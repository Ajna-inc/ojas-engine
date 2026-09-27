//! qwen35moe MoE FFN kernels (Qwen3.6-A3B family). Own MSL source string;
//! kernels registered via `MOE_KERNEL_NAMES`.
//!
//! The expert GEMVs use the qmv_fast pattern (as `gemv_q4_fast` in the main kernel
//! set): uint16 vectorized weight loads, mask-in-place no-shift dequant (nibble masked
//! in its byte lane = nib×16^lane, with x pre-divided so the products come out right),
//! 4 output rows per simdgroup. That reads ~30% faster than the plain uchar2 pattern on
//! the same shapes, and expert weights are ~580 MB/token on the 35B-A3B, the largest
//! single stream in MoE decode.

/// Kernels to register from this module's source.
pub const MOE_KERNEL_NAMES: &[&str] = &["gemv_w32", "gemv_w32_accum", "gemv_w32_bias", "moe_topk", "moe_topk_nonorm", "moe_gu_q4", "moe_down_q4", "lg_copy",
    "gemv_w32_m", "moe_topk_m", "moe_gu_q4_m", "moe_down_q4_m", "moe_gu_q8", "moe_down_q8", "moe_gu_q4k", "moe_topk_v3", "moe_down_q80", "moe_down_q50", "moe_down_q5k", "moe_down_q6k",
    "moe_gu_q8_oai", "moe_down_q8_oai", "moe_gu_q8_oai_m", "moe_down_q8_oai_m"];

pub const MOE_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ===== qmv_fast-style symmetric-q4 dot helpers (see gemv_q4_fast in qwen.rs).
// Q4FAST_DOT processes 16 weights of row r into res_[r]: nibbles masked IN PLACE
// (& 0x00f0 = nib*16 etc.) against x pre-divided by 16^lane, so no shift ALU.
// Symmetric q4: dot = scale*(Σx·nib) - 8*scale*(Σx).
#define Q4FAST_DOT(wr_, r_, xt_, xsum_, sc_, res_) { \
    device const uint16_t* ws = (device const uint16_t*)((wr_) + (r_)*rb); \
    float s = float((sc_)[(r_)*nblk]); float acc = 0.0; \
    for (uint i = 0u; i < 4u; i++) { uint16_t wv = ws[i]; \
        acc += (xt_)[4u*i]*float(wv & 0x000fu) + (xt_)[4u*i+1u]*float(wv & 0x00f0u) \
             + (xt_)[4u*i+2u]*float(wv & 0x0f00u) + (xt_)[4u*i+3u]*float(wv & 0xf000u); } \
    (res_)[r_] += s*acc - 8.0*s*(xsum_); }

// Load 16 x values (pre-divided for the mask trick), optionally pre-scaled by w_
// (expert routing weight — the dot is linear in x, so scaling x scales the dot).
#define Q4FAST_LOADXW(xr_, xt_, xsum_, w_) \
    float xsum_ = 0.0; \
    for (uint i = 0u; i < 16u; i += 4u) { \
        float a=(xr_)[i]*(w_), b=(xr_)[i+1u]*(w_), c=(xr_)[i+2u]*(w_), d=(xr_)[i+3u]*(w_); \
        xsum_ += a+b+c+d; \
        (xt_)[i]=a; (xt_)[i+1u]=b*(1.0/16.0); (xt_)[i+2u]=c*(1.0/256.0); (xt_)[i+3u]=d*(1.0/4096.0); }

// f32-weight GEMV (router ffn_gate_inp [N,K] f32; also the 1-row shexp gate).
// The router ships as F32 in the GGUF (the reference runs it as f32 mul_mv too).
kernel void gemv_w32(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    float p = 0.0;
    if (K % 4u == 0u) {
        device const float4* row = (device const float4*)(w + (ulong)n*(ulong)K);
        device const float4* xv = (device const float4*)x;
        for (uint k = lane; k < K/4u; k += 32u) { p += dot(row[k], xv[k]); }
    } else {
        for (uint k = lane; k < K; k += 32u) { p += w[(ulong)n*K+k] * x[k]; }
    }
    p = simd_sum(p);
    if (lane == 0u) { y[n] = p; }
}

kernel void gemv_w32_accum(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    float p = 0.0;
    if (K % 4u == 0u) {
        device const float4* row = (device const float4*)(w + (ulong)n*(ulong)K);
        device const float4* xv = (device const float4*)x;
        for (uint k = lane; k < K/4u; k += 32u) { p += dot(row[k], xv[k]); }
    } else {
        for (uint k = lane; k < K; k += 32u) { p += w[(ulong)n*K+k] * x[k]; }
    }
    p = simd_sum(p);
    if (lane == 0u) { y[n] += p; }
}

kernel void gemv_w32_bias(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* bias [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    float p = 0.0;
    if (K % 4u == 0u) {
        device const float4* row = (device const float4*)(w + (ulong)n*(ulong)K);
        device const float4* xv = (device const float4*)x;
        for (uint k = lane; k < K/4u; k += 32u) { p += dot(row[k], xv[k]); }
    } else {
        for (uint k = lane; k < K; k += 32u) { p += w[(ulong)n*K+k] * x[k]; }
    }
    p = simd_sum(p);
    if (lane == 0u) { y[n] = p + bias[n]; }
}

// Router: softmax over E logits, select top-KSEL experts, weights renormalized over
// the selection (norm_topk_prob — ref qwen35moe.cpp build_moe_ffn). One threadgroup,
// 32 threads. E <= 1024.
// Routing-telemetry snapshot: copy this layer's router logits into a
// persistent [n_layers x n_expert] buffer (read back per token for the
// routing-agreement metric — top-k overlap / JS vs a reference run).
kernel void lg_copy(device const float* lg [[buffer(0)]], device float* dst [[buffer(1)]],
    constant uint& off [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    dst[off + gid] = lg[gid];
}
kernel void moe_topk(device const float* lg [[buffer(0)]], device uint* idx [[buffer(1)]],
    device float* wgt [[buffer(2)]], constant uint& E [[buffer(3)]], constant uint& KSEL [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float pr[1024];
    threadgroup float tw[32];
    float mx = -1e30;
    for (uint e = lane; e < E; e += 32u) { mx = max(mx, lg[e]); }
    mx = simd_max(mx);
    float sum = 0.0;
    for (uint e = lane; e < E; e += 32u) { float v = exp(lg[e] - mx); pr[e] = v; sum += v; }
    sum = simd_sum(sum);
    float wsum = 0.0;
    for (uint j = 0u; j < KSEL; j++) {
        float bv = -1.0; uint bi = 0u;
        for (uint e = lane; e < E; e += 32u) { if (pr[e] > bv) { bv = pr[e]; bi = e; } }
        float m2 = simd_max(bv);
        uint cand = (bv == m2) ? bi : 0xffffffffu;
        cand = simd_min(cand);
        if (lane == 0u) { idx[j] = cand; tw[j] = m2/sum; pr[cand] = -1.0; }
        wsum += m2/sum;                    // uniform across lanes
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < KSEL) { wgt[lane] = tw[lane]/wsum; }
}
// DeepSeek-V2 routing (norm_topk_prob=false): raw softmax probs, no /wsum renorm.
kernel void moe_topk_nonorm(device const float* lg [[buffer(0)]], device uint* idx [[buffer(1)]],
    device float* wgt [[buffer(2)]], constant uint& E [[buffer(3)]], constant uint& KSEL [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float pr[1024]; threadgroup float tw[32];
    float mx = -1e30;
    for (uint e = lane; e < E; e += 32u) { mx = max(mx, lg[e]); }
    mx = simd_max(mx);
    float sum = 0.0;
    for (uint e = lane; e < E; e += 32u) { float v = exp(lg[e] - mx); pr[e] = v; sum += v; }
    sum = simd_sum(sum);
    for (uint j = 0u; j < KSEL; j++) {
        float bv = -1.0; uint bi = 0u;
        for (uint e = lane; e < E; e += 32u) { if (pr[e] > bv) { bv = pr[e]; bi = e; } }
        float m2 = simd_max(bv);
        uint cand = (bv == m2) ? bi : 0xffffffffu;
        cand = simd_min(cand);
        if (lane == 0u) { idx[j] = cand; tw[j] = m2/sum; pr[cand] = -1.0; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < KSEL) { wgt[lane] = tw[lane]; }
}

// ===== Batched (chunked-prefill) variants: same math per row, with all M rows'
// expert work in flight in one dispatch. The speed comes from concurrency —
// threadgroups routed to the same expert run simultaneously, so the expert's weight
// stream is read once through the cache hierarchy instead of once per token (per-token
// MoE prefill re-reads ~580 MB/token on the 35B-A3B).

// Batched f32-weight GEMV: row = tg.y; x row-major [M,K], y [M,N].
kernel void gemv_w32_m(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint n = tg.x*(ts.x/32u) + sgid; if (n >= N) { return; }
    device const float4* row = (device const float4*)(w + (ulong)n*(ulong)K);
    device const float4* xv = (device const float4*)(x + (ulong)tg.y*(ulong)K);
    uint K4 = K/4u; float p = 0.0;
    for (uint k = lane; k < K4; k += 32u) { p += dot(row[k], xv[k]); }
    p = simd_sum(p);
    if (lane == 0u) { y[(ulong)tg.y*(ulong)N + n] = p; }
}

// Batched router top-k: one threadgroup per row; lg [M,E], idx/wgt [M,KSEL].
kernel void moe_topk_m(device const float* lg [[buffer(0)]], device uint* idx [[buffer(1)]],
    device float* wgt [[buffer(2)]], constant uint& E [[buffer(3)]], constant uint& KSEL [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float pr[1024];
    threadgroup float tw[32];
    device const float* lr = lg + (ulong)row*(ulong)E;
    float mx = -1e30;
    for (uint e = lane; e < E; e += 32u) { mx = max(mx, lr[e]); }
    mx = simd_max(mx);
    float sum = 0.0;
    for (uint e = lane; e < E; e += 32u) { float v = exp(lr[e] - mx); pr[e] = v; sum += v; }
    sum = simd_sum(sum);
    float wsum = 0.0;
    for (uint j = 0u; j < KSEL; j++) {
        float bv = -1.0; uint bi = 0u;
        for (uint e = lane; e < E; e += 32u) { if (pr[e] > bv) { bv = pr[e]; bi = e; } }
        float m2 = simd_max(bv);
        uint cand = (bv == m2) ? bi : 0xffffffffu;
        cand = simd_min(cand);
        if (lane == 0u) { idx[row*KSEL + j] = cand; tw[j] = m2/sum; pr[cand] = -1.0; }
        wsum += m2/sum;                    // uniform across lanes
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < KSEL) { wgt[row*KSEL + lane] = tw[lane]/wsum; }
}

// Batched per-expert SwiGLU: assignment j = tg.y spans [M*KSEL); row = j/KSEL,
// expert = idx[j] (idx is the batched [M,KSEL] table), output act[j][N].
kernel void moe_gu_q4_m(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    device const half* sg [[buffer(6)]], device const half* su [[buffer(7)]],
    device const uint* idx [[buffer(8)]], constant uint& KSEL [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    ulong rowb = (ulong)idx[j]*(ulong)N + out_row;
    uint nblk = K/32u; uint rb = K/2u;
    device const uchar* wgr = wg + rowb*(ulong)rb + lane*8u;
    device const uchar* wur = wu + rowb*(ulong)rb + lane*8u;
    device const half* sgc = sg + rowb*(ulong)nblk + lane/2u;
    device const half* suc = su + rowb*(ulong)nblk + lane/2u;
    device const float* xr = x + (ulong)(j/KSEL)*(ulong)K + lane*16u;
    float rg[4] = {0.0, 0.0, 0.0, 0.0};
    float ru[4] = {0.0, 0.0, 0.0, 0.0};
    uint kfull = (K/512u)*512u;
    for (uint k = 0u; k < kfull; k += 512u) {
        float xt[16]; Q4FAST_LOADXW(xr, xt, xsum, 1.0)
        for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(wgr, r, xt, xsum, sgc, rg) Q4FAST_DOT(wur, r, xt, xsum, suc, ru) }
        wgr += 256u; wur += 256u; sgc += 16u; suc += 16u; xr += 512u;
    }
    if (lane*16u < K - kfull) {
        float xt[16]; Q4FAST_LOADXW(xr, xt, xsum, 1.0)
        for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(wgr, r, xt, xsum, sgc, rg) Q4FAST_DOT(wur, r, xt, xsum, suc, ru) }
    }
    for (uint r = 0u; r < 4u; r++) {
        float g = simd_sum(rg[r]); float u = simd_sum(ru[r]);
        if (lane == 0u) { act[(ulong)j*(ulong)N + out_row + r] = (g/(1.0 + exp(-g)))*u; }
    }
}

// Batched weighted down-projection + shared-expert add: row = tg.y;
// x[row][n] += Σ_j wgt[row][j]·(Wd_{e_j}[n]·act[row*KSEL+j]) + shx[row][n]·sigmoid(shg[row]).
kernel void moe_down_q4_m(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* sd [[buffer(5)]], device const uint* idx [[buffer(6)]],
    device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint row = tg.y;
    uint out_row = tg.x*(ts.x/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    uint nblk = K/32u; uint rb = K/2u;
    float res[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint j = 0u; j < KSEL; j++) {
        ulong rowb = (ulong)idx[row*KSEL + j]*(ulong)N + out_row;
        device const uchar* wr = wd + rowb*(ulong)rb + lane*8u;
        device const half* sc = sd + rowb*(ulong)nblk + lane/2u;
        device const float* xr = act + (ulong)(row*KSEL + j)*(ulong)K + lane*16u;
        float w = wgt[row*KSEL + j];
        uint kfull = (K/512u)*512u;
        for (uint k = 0u; k < kfull; k += 512u) {
            float xt[16]; Q4FAST_LOADXW(xr, xt, xsum, w)
            for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(wr, r, xt, xsum, sc, res) }
            wr += 256u; sc += 16u; xr += 512u;
        }
        if (lane*16u < K - kfull) {
            float xt[16]; Q4FAST_LOADXW(xr, xt, xsum, w)
            for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(wr, r, xt, xsum, sc, res) }
        }
    }
    for (uint r = 0u; r < 4u; r++) {
        float v = simd_sum(res[r]);
        if (lane == 0u) {
            uint n = out_row + r;
            x[(ulong)row*(ulong)N + n] += v + shx[(ulong)row*(ulong)N + n]/(1.0 + exp(-shg[row]));
        }
    }
}

// Per-expert SwiGLU: act[j][n] = silu(Wg_e·h)·(Wu_e·h), e = idx[tg.y]. Expert e's
// rows are the block e*N..(e+1)*N of the stacked [n_expert*N, K] q4 weight.
// Fast pattern: 4 rows/simdgroup × 2 simdgroups = 8 rows/tg, 64 threads.
kernel void moe_gu_q4(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    device const half* sg [[buffer(6)]], device const half* su [[buffer(7)]],
    device const uint* idx [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    ulong rowb = (ulong)idx[j]*(ulong)N + out_row;
    uint nblk = K/32u; uint rb = K/2u;
    device const uchar* wgr = wg + rowb*(ulong)rb + lane*8u;
    device const uchar* wur = wu + rowb*(ulong)rb + lane*8u;
    device const half* sgc = sg + rowb*(ulong)nblk + lane/2u;
    device const half* suc = su + rowb*(ulong)nblk + lane/2u;
    device const float* xr = x + lane*16u;
    float rg[4] = {0.0, 0.0, 0.0, 0.0};
    float ru[4] = {0.0, 0.0, 0.0, 0.0};
    uint kfull = (K/512u)*512u;
    for (uint k = 0u; k < kfull; k += 512u) {
        float xt[16]; Q4FAST_LOADXW(xr, xt, xsum, 1.0)
        for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(wgr, r, xt, xsum, sgc, rg) Q4FAST_DOT(wur, r, xt, xsum, suc, ru) }
        wgr += 256u; wur += 256u; sgc += 16u; suc += 16u; xr += 512u;
    }
    if (lane*16u < K - kfull) {
        float xt[16]; Q4FAST_LOADXW(xr, xt, xsum, 1.0)
        for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(wgr, r, xt, xsum, sgc, rg) Q4FAST_DOT(wur, r, xt, xsum, suc, ru) }
    }
    for (uint r = 0u; r < 4u; r++) {
        float g = simd_sum(rg[r]); float u = simd_sum(ru[r]);
        if (lane == 0u) { act[j*N + out_row + r] = (g/(1.0 + exp(-g)))*u; }
    }
}

// Weighted expert down-projection FUSED with the shared-expert add:
// x[n] += Σ_j wgt[j]·(Wd_{e_j}[n]·act[j]) + shx[n]·sigmoid(shg[0]).
// The routing weight folds into the x load (dot is linear in x).
kernel void moe_down_q4(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* sd [[buffer(5)]], device const uint* idx [[buffer(6)]],
    device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    uint nblk = K/32u; uint rb = K/2u;
    float res[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint j = 0u; j < KSEL; j++) {
        ulong rowb = (ulong)idx[j]*(ulong)N + out_row;
        device const uchar* wr = wd + rowb*(ulong)rb + lane*8u;
        device const half* sc = sd + rowb*(ulong)nblk + lane/2u;
        device const float* xr = act + (ulong)j*(ulong)K + lane*16u;
        float w = wgt[j];
        uint kfull = (K/512u)*512u;
        for (uint k = 0u; k < kfull; k += 512u) {
            float xt[16]; Q4FAST_LOADXW(xr, xt, xsum, w)
            for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(wr, r, xt, xsum, sc, res) }
            wr += 256u; sc += 16u; xr += 512u;
        }
        if (lane*16u < K - kfull) {
            float xt[16]; Q4FAST_LOADXW(xr, xt, xsum, w)
            for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(wr, r, xt, xsum, sc, res) }
        }
    }
    for (uint r = 0u; r < 4u; r++) {
        float v = simd_sum(res[r]);
        if (lane == 0u) { uint n = out_row + r; x[n] += v + shx[n]/(1.0 + exp(-shg[0])); }
    }
}

// ===== Q8 expert kernels (int8 weights + per-row f32 scale, quantize_row_i8 layout).
// Same math/dispatch as the q4 pair; ~0.4% error (vs Q4's ~2%) — used for deepseek2
// where Q4 requant error compounds through the MLA latent + 27 MoE layers.
kernel void moe_gu_q8(device const float* x [[buffer(0)]], device const char* wg [[buffer(1)]],
    device const char* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    device const float* sg [[buffer(6)]], device const float* su [[buffer(7)]],
    device const uint* idx [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    for (uint r = 0u; r < 4u; r++) {
        ulong row = (ulong)idx[j]*(ulong)N + (ulong)(out_row + r);
        device const char* wgr = wg + row*(ulong)K;
        device const char* wur = wu + row*(ulong)K;
        float g = 0.0, u = 0.0;
        for (uint k = lane; k < K; k += 32u) { float xk = x[k]; g += float(wgr[k])*xk; u += float(wur[k])*xk; }
        g = simd_sum(g)*sg[row]; u = simd_sum(u)*su[row];
        if (lane == 0u) { act[j*N + out_row + r] = (g/(1.0 + exp(-g)))*u; }
    }
}
kernel void moe_down_q8(device const float* act [[buffer(0)]], device const char* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* sd [[buffer(5)]], device const uint* idx [[buffer(6)]],
    device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    float res[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint j = 0u; j < KSEL; j++) {
        float w = wgt[j];
        device const float* xr = act + (ulong)j*(ulong)K;
        for (uint r = 0u; r < 4u; r++) {
            ulong row = (ulong)idx[j]*(ulong)N + (ulong)(out_row + r);
            device const char* wr = wd + row*(ulong)K;
            float acc = 0.0;
            for (uint k = lane; k < K; k += 32u) { acc += float(wr[k])*xr[k]; }
            res[r] += simd_sum(acc)*sd[row]*w;
        }
    }
    for (uint r = 0u; r < 4u; r++) {
        if (lane == 0u) { uint n = out_row + r; x[n] += res[r] + shx[n]/(1.0 + exp(-shg[0])); }
    }
}

// ===== GPT-OSS expert kernels: Q8 weights + per-row scale + per-row BIAS, SwiGLU-OAI
// gate, and NO shared expert. bias tensors are [ffn_exp/d, n_expert] dims →
// expert-major layout b[expert*N + row], matching the weight row index idx[j]*N+row.
kernel void moe_gu_q8_oai(device const float* x [[buffer(0)]], device const char* wg [[buffer(1)]],
    device const char* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    device const float* sg [[buffer(6)]], device const float* su [[buffer(7)]],
    device const uint* idx [[buffer(8)]],
    device const float* bg [[buffer(9)]], device const float* bu [[buffer(10)]],
    constant float& alpha [[buffer(11)]], constant float& limit [[buffer(12)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    for (uint r = 0u; r < 4u; r++) {
        ulong row = (ulong)idx[j]*(ulong)N + (ulong)(out_row + r);
        device const char* wgr = wg + row*(ulong)K;
        device const char* wur = wu + row*(ulong)K;
        float g = 0.0, u = 0.0;
        for (uint k = lane; k < K; k += 32u) { float xk = x[k]; g += float(wgr[k])*xk; u += float(wur[k])*xk; }
        g = simd_sum(g)*sg[row] + bg[row];
        u = simd_sum(u)*su[row] + bu[row];
        if (lane == 0u) {
            float xg = min(g, limit);            // SwiGLU-OAI: clamp gate to limit
            float yu = clamp(u, -limit, limit);  // clamp up to [-limit, limit]
            act[j*N + out_row + r] = (xg/(1.0 + exp(-alpha*xg))) * (yu + 1.0);
        }
    }
}
kernel void moe_down_q8_oai(device const float* act [[buffer(0)]], device const char* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* sd [[buffer(5)]], device const uint* idx [[buffer(6)]],
    device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* bd [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    float res[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint j = 0u; j < KSEL; j++) {
        float w = wgt[j];
        device const float* xr = act + (ulong)j*(ulong)K;
        for (uint r = 0u; r < 4u; r++) {
            ulong row = (ulong)idx[j]*(ulong)N + (ulong)(out_row + r);
            device const char* wr = wd + row*(ulong)K;
            float acc = 0.0;
            for (uint k = lane; k < K; k += 32u) { acc += float(wr[k])*xr[k]; }
            res[r] += (simd_sum(acc)*sd[row] + bd[row]) * w;   // per-expert down bias, then route weight
        }
    }
    for (uint r = 0u; r < 4u; r++) {
        if (lane == 0u) { x[out_row + r] += res[r]; }          // no shared expert
    }
}

// ===== GPT-OSS BATCHED (M-token) biased SwiGLU-OAI MoE — for chunked prefill.
// idx/wgt/act are [M*KSEL]-shaped (token m's experts at m*KSEL..); x/out are [M,*].
kernel void moe_gu_q8_oai_m(device const float* x [[buffer(0)]], device const char* wg [[buffer(1)]],
    device const char* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    device const float* sg [[buffer(6)]], device const float* su [[buffer(7)]],
    device const uint* idx [[buffer(8)]], constant uint& KSEL [[buffer(9)]],
    device const float* bg [[buffer(10)]], device const float* bu [[buffer(11)]],
    constant float& alpha [[buffer(12)]], constant float& limit [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;                                  // token m's expert e: j = m*KSEL + e
    uint out_row = tg.x*(ts.x/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    device const float* xr = x + (ulong)(j/KSEL)*(ulong)K;   // token m's hidden
    for (uint r = 0u; r < 4u; r++) {
        ulong row = (ulong)idx[j]*(ulong)N + (ulong)(out_row + r);
        device const char* wgr = wg + row*(ulong)K;
        device const char* wur = wu + row*(ulong)K;
        float g = 0.0, u = 0.0;
        for (uint k = lane; k < K; k += 32u) { float xk = xr[k]; g += float(wgr[k])*xk; u += float(wur[k])*xk; }
        g = simd_sum(g)*sg[row] + bg[row];
        u = simd_sum(u)*su[row] + bu[row];
        if (lane == 0u) {
            float xg = min(g, limit); float yu = clamp(u, -limit, limit);
            act[j*N + out_row + r] = (xg/(1.0 + exp(-alpha*xg))) * (yu + 1.0);
        }
    }
}
kernel void moe_down_q8_oai_m(device const float* act [[buffer(0)]], device const char* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* sd [[buffer(5)]], device const uint* idx [[buffer(6)]],
    device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* bd [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint mtok = tg.y;                               // token index
    uint out_row = tg.x*(ts.x/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    float res[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint e = 0u; e < KSEL; e++) {
        uint j = mtok*KSEL + e; float w = wgt[j];
        device const float* xr = act + (ulong)j*(ulong)K;
        for (uint r = 0u; r < 4u; r++) {
            ulong row = (ulong)idx[j]*(ulong)N + (ulong)(out_row + r);
            device const char* wr = wd + row*(ulong)K;
            float acc = 0.0;
            for (uint k = lane; k < K; k += 32u) { acc += float(wr[k])*xr[k]; }
            res[r] += (simd_sum(acc)*sd[row] + bd[row]) * w;
        }
    }
    for (uint r = 0u; r < 4u; r++) { if (lane == 0u) { x[(ulong)mtok*(ulong)N + out_row + r] += res[r]; } }
}

// Q4_K expert SwiGLU (native Q4_K gate/up, faithful). 1 output row per simdgroup.
#define Q4K_DOT_M(WR, SB, XB, ACC) { \
    device const uchar* _b = (WR) + (ulong)(SB)*144u; \
    float _d = float(*(device const half*)_b); float _dm = float(*(device const half*)(_b+2)); \
    device const uchar* _sc = _b+4; device const uchar* _qs = _b+16; \
    device const float* _xb = (XB) + (ulong)(SB)*256u; \
    for (uint _j=0u;_j<8u;_j++){ uint _s,_m; \
      if(_j<4u){_s=_sc[_j]&63u;_m=_sc[_j+4u]&63u;} \
      else{_s=(_sc[_j+4u]&0x0Fu)|((_sc[_j-4u]>>6)<<4);_m=(_sc[_j+4u]>>4)|((_sc[_j]>>6)<<4);} \
      float _d1=_d*float(_s); float _m1=_dm*float(_m); \
      device const uchar* _qq=_qs+(_j>>1)*32u; uint _hi=_j&1u; \
      for(uint _l=0u;_l<32u;_l++){ uint _nb=_hi?(uint(_qq[_l])>>4):(uint(_qq[_l])&0x0Fu); \
        (ACC) += (_d1*float(_nb)-_m1)*_xb[_j*32u+_l]; } } }
kernel void moe_gu_q4k(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], device const uint* idx [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N) { return; }
    uint nsb = K/256u;
    device const uchar* wgr = wg + ((ulong)idx[j]*(ulong)N + out_row)*(ulong)nsb*144u;
    device const uchar* wur = wu + ((ulong)idx[j]*(ulong)N + out_row)*(ulong)nsb*144u;
    float g = 0.0, u = 0.0;
    for (uint sb = lane; sb < nsb; sb += 32u) { Q4K_DOT_M(wgr, sb, x, g) Q4K_DOT_M(wur, sb, x, u) }
    g = simd_sum(g); u = simd_sum(u);
    if (lane == 0u) { act[j*N + out_row] = (g/(1.0 + exp(-g)))*u; }
}

// DeepSeek-V3 / GLM-5.2 router: choice = sigmoid(logit) + bias (exp_probs_b) for SELECTION;
// weight = sigmoid(logit) (unbiased); optional renorm over the top-k; × routed_scaling_factor.
kernel void moe_topk_v3(device const float* lg [[buffer(0)]], device const float* bias [[buffer(1)]],
    device uint* idx [[buffer(2)]], device float* wgt [[buffer(3)]], constant uint& E [[buffer(4)]],
    constant uint& KSEL [[buffer(5)]], constant float& rscale [[buffer(6)]], constant uint& norm [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sg[1024]; threadgroup float ch[1024]; threadgroup float tw[32];
    for (uint e = lane; e < E; e += 32u) { float s = 1.0/(1.0+exp(-lg[e])); sg[e] = s; ch[e] = s + bias[e]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0u; j < KSEL; j++) {
        float bv = -1e30; uint bi = 0u;
        for (uint e = lane; e < E; e += 32u) { if (ch[e] > bv) { bv = ch[e]; bi = e; } }
        float m2 = simd_max(bv);
        uint cand = (bv == m2) ? bi : 0xffffffffu; cand = simd_min(cand);
        if (lane == 0u) { idx[j] = cand; tw[j] = sg[cand]; ch[cand] = -1e30; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        float sum = 0.0; for (uint j = 0u; j < KSEL; j++) sum += tw[j];
        float inv = (norm != 0u && sum > 0.0) ? (1.0/sum) : 1.0;
        for (uint j = 0u; j < KSEL; j++) wgt[j] = tw[j] * inv * rscale;
    }
}

// Native Q8_0 expert down-proj (block {half d; i8 qs[32]}=34B/32) — for direct-mmap streaming
// of GGUF Q8_0 down_exps. Same accumulate as moe_down_q8 but reads the native block format.
// Gate/up over raw GGUF Q8_0 expert rows (34 B / 32 weights: f16 scale + int8).
// Distinct from moe_gu_q8, which reads the requantized int8 + f32 row-scale layout.
// Needed because the qwen4exp MTP draft head ships Q8_0 experts.
kernel void moe_gu_q80(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], device const uint* idx [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N) { return; }
    uint nblk = K/32u; ulong rowbytes = (ulong)nblk*34u;
    ulong row = (ulong)idx[j]*(ulong)N + (ulong)out_row;
    device const uchar* gr = wg + row*rowbytes;
    device const uchar* ur = wu + row*rowbytes;
    float gs = 0.0, us = 0.0;
    for (uint blk = lane; blk < nblk; blk += 32u) {
        device const uchar* gb = gr + (ulong)blk*34u;
        device const uchar* ub = ur + (ulong)blk*34u;
        float gd = float(*(device const half*)gb);
        float ud = float(*(device const half*)ub);
        device const char* gq = (device const char*)(gb + 2);
        device const char* uq = (device const char*)(ub + 2);
        float sg = 0.0, su = 0.0;
        for (uint i = 0u; i < 32u; i++) { float xv = x[blk*32u + i]; sg += float(gq[i]) * xv; su += float(uq[i]) * xv; }
        gs += gd * sg; us += ud * su;
    }
    gs = simd_sum(gs); us = simd_sum(us);
    if (lane == 0u) { act[(ulong)j*(ulong)N + out_row] = (gs/(1.0 + exp(-gs))) * us; }
}

kernel void moe_down_q80(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    uint nblk = K/32u; ulong rowbytes = (ulong)nblk*34u;
    float res[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint j = 0u; j < KSEL; j++) {
        float w = wgt[j];
        device const float* xr = act + (ulong)j*(ulong)K;
        for (uint r = 0u; r < 4u; r++) {
            ulong row = (ulong)idx[j]*(ulong)N + (ulong)(out_row + r);
            device const uchar* wr = wd + row*rowbytes;
            float acc = 0.0;
            for (uint blk = lane; blk < nblk; blk += 32u) {
                device const uchar* b = wr + (ulong)blk*34u;
                float d = float(*(device const half*)b);
                device const char* q = (device const char*)(b + 2);
                float s = 0.0;
                for (uint i = 0u; i < 32u; i++) s += float(q[i]) * xr[blk*32u + i];
                acc += d * s;
            }
            res[r] += simd_sum(acc) * w;
        }
    }
    for (uint r = 0u; r < 4u; r++) {
        if (lane == 0u) { uint n = out_row + r; x[n] += res[r] + shx[n]/(1.0 + exp(-shg[0])); }
    }
}

// Native Q5_0 expert down-proj (block {half d; u8 qh[4]; u8 qs[16]}=22B/32; w=d*(q-16),
// q = low/high nibble | (qh bit). For direct-mmap streaming of GGUF Q5_0 down_exps.
// Batched twins of the raw-Q8_0 pair. tg.y is the token for down (each token
// sums its own KSEL experts) and a flat (token, slot) index for gate/up.
kernel void moe_gu_q80_m(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], device const uint* idx [[buffer(8)]],
    constant uint& KSEL [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N) { return; }
    uint nblk = K/32u; ulong rowbytes = (ulong)nblk*34u;
    ulong row = (ulong)idx[j]*(ulong)N + (ulong)out_row;
    device const uchar* gr = wg + row*rowbytes;
    device const uchar* ur = wu + row*rowbytes;
    device const float* xt = x + (ulong)(j/KSEL)*(ulong)K;
    float gs = 0.0, us = 0.0;
    for (uint blk = lane; blk < nblk; blk += 32u) {
        device const uchar* gb = gr + (ulong)blk*34u;
        device const uchar* ub = ur + (ulong)blk*34u;
        float gd = float(*(device const half*)gb);
        float ud = float(*(device const half*)ub);
        device const char* gq = (device const char*)(gb + 2);
        device const char* uq = (device const char*)(ub + 2);
        float sg = 0.0, su = 0.0;
        for (uint i = 0u; i < 32u; i++) { float xv = xt[blk*32u + i]; sg += float(gq[i]) * xv; su += float(uq[i]) * xv; }
        gs += gd * sg; us += ud * su;
    }
    gs = simd_sum(gs); us = simd_sum(us);
    if (lane == 0u) { act[(ulong)j*(ulong)N + out_row] = (gs/(1.0 + exp(-gs))) * us; }
}

kernel void moe_down_q80_m(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint rowt = tg.y;
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N) { return; }
    uint nblk = K/32u; ulong rowbytes = (ulong)nblk*34u;
    float total = 0.0;
    for (uint j = 0u; j < KSEL; j++) {
        float w = wgt[rowt*KSEL + j];
        device const float* xr = act + (ulong)(rowt*KSEL + j)*(ulong)K;
        ulong row = (ulong)idx[rowt*KSEL + j]*(ulong)N + (ulong)out_row;
        device const uchar* wr = wd + row*rowbytes;
        float acc = 0.0;
        for (uint blk = lane; blk < nblk; blk += 32u) {
            device const uchar* b = wr + (ulong)blk*34u;
            float d = float(*(device const half*)b);
            device const char* q = (device const char*)(b + 2);
            float sacc = 0.0;
            for (uint i = 0u; i < 32u; i++) sacc += float(q[i]) * xr[blk*32u + i];
            acc += d * sacc;
        }
        total += simd_sum(acc) * w;
    }
    if (lane == 0u) {
        ulong o = (ulong)rowt*(ulong)N + out_row;
        x[o] += total + shx[o]/(1.0 + exp(-shg[rowt]));
    }
}

kernel void moe_down_q50(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    uint nblk = K/32u; ulong rowbytes = (ulong)nblk*22u;
    float res[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint j = 0u; j < KSEL; j++) {
        float w = wgt[j];
        device const float* xr = act + (ulong)j*(ulong)K;
        for (uint r = 0u; r < 4u; r++) {
            ulong row = (ulong)idx[j]*(ulong)N + (ulong)(out_row + r);
            device const uchar* wr = wd + row*rowbytes;
            float acc = 0.0;
            for (uint blk = lane; blk < nblk; blk += 32u) {
                device const uchar* b = wr + (ulong)blk*22u;
                float d = float(*(device const half*)b);
                uint qh = uint(b[2]) | (uint(b[3])<<8) | (uint(b[4])<<16) | (uint(b[5])<<24);
                device const uchar* qs = b + 6;
                float s = 0.0;
                for (uint wi = 0u; wi < 32u; wi++) {
                    uint nib = (wi < 16u) ? uint(qs[wi] & 0x0Fu) : uint(qs[wi - 16u] >> 4);
                    uint q = nib | (((qh >> wi) & 1u) << 4);
                    s += (d*(float(q) - 16.0)) * xr[blk*32u + wi];
                }
                acc += s;
            }
            res[r] += simd_sum(acc) * w;
        }
    }
    for (uint r = 0u; r < 4u; r++) {
        if (lane == 0u) { uint n = out_row + r; x[n] += res[r] + shx[n]/(1.0 + exp(-shg[0])); }
    }
}

// Native Q5_K expert down-proj (176B/256 super-block: d,dmin + 6-bit sub-scales + 5th bit qh)
// for direct-mmap streaming of GGUF Q5_K down_exps (dynamic-quant GLM-5.2 GGUFs). Same accumulate as
// moe_down_q80; dequant per gguf.rs Q5_K (get_scale_min_k4 + qh 5th bit).
kernel void moe_down_q5k(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    uint nsb = K/256u; ulong rowbytes = (ulong)nsb*176u;
    float res[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint j = 0u; j < KSEL; j++) {
        float w = wgt[j];
        device const float* xr = act + (ulong)j*(ulong)K;
        for (uint r = 0u; r < 4u; r++) {
            ulong row = (ulong)idx[j]*(ulong)N + (ulong)(out_row + r);
            device const uchar* wr = wd + row*rowbytes;
            float acc = 0.0;
            for (uint sb = lane; sb < nsb; sb += 32u) {
                device const uchar* b = wr + (ulong)sb*176u;
                float d = float(*(device const half*)b); float dm = float(*(device const half*)(b+2));
                device const uchar* sc = b + 4; device const uchar* qh = b + 16; device const uchar* qs = b + 48;
                device const float* xb = xr + (ulong)sb*256u;
                for (uint c = 0u; c < 4u; c++) {
                    // get_scale_min_k4 for j0=2c and j1=2c+1
                    uint j0 = 2u*c, j1 = 2u*c + 1u; uint s0,m0,s1,m1;
                    if (j0 < 4u) { s0 = sc[j0]&63u; m0 = sc[j0+4u]&63u; } else { s0=(sc[j0+4u]&0x0Fu)|((sc[j0-4u]>>6)<<4); m0=(sc[j0+4u]>>4)|((sc[j0]>>6)<<4); }
                    if (j1 < 4u) { s1 = sc[j1]&63u; m1 = sc[j1+4u]&63u; } else { s1=(sc[j1+4u]&0x0Fu)|((sc[j1-4u]>>6)<<4); m1=(sc[j1+4u]>>4)|((sc[j1]>>6)<<4); }
                    float dl0=d*float(s0), ml0=dm*float(m0), dl1=d*float(s1), ml1=dm*float(m1);
                    device const uchar* ql = qs + c*32u;
                    uint u0 = 1u << (2u*c), u1 = 2u << (2u*c);
                    for (uint l = 0u; l < 32u; l++) {
                        float hi0 = (qh[l] & u0) ? 16.0 : 0.0;
                        acc += (dl0*(float(ql[l]&0x0Fu)+hi0) - ml0) * xb[(2u*c)*32u + l];
                        float hi1 = (qh[l] & u1) ? 16.0 : 0.0;
                        acc += (dl1*(float(ql[l]>>4)+hi1) - ml1) * xb[(2u*c+1u)*32u + l];
                    }
                }
            }
            res[r] += simd_sum(acc) * w;
        }
    }
    for (uint r = 0u; r < 4u; r++) {
        if (lane == 0u) { uint n = out_row + r; x[n] += res[r] + shx[n]/(1.0 + exp(-shg[0])); }
    }
}

// Native Q6_K (type 14, 210 B/256) MoE down projection — same accumulation shape as
// moe_down_q5k; dequant per block_q6_K (ql 4b | qh 2b | int8 scales | half d).
kernel void moe_down_q6k(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    uint nsb = K/256u; ulong rowbytes = (ulong)nsb*210u;
    float res[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint j = 0u; j < KSEL; j++) {
        float w = wgt[j];
        device const float* xr = act + (ulong)j*(ulong)K;
        for (uint r = 0u; r < 4u; r++) {
            ulong row = (ulong)idx[j]*(ulong)N + (ulong)(out_row + r);
            device const uchar* wr = wd + row*rowbytes;
            float acc = 0.0;
            for (uint sb = lane; sb < nsb; sb += 32u) {
                device const uchar* b = wr + (ulong)sb*210u;
                float d = float(*(device const half*)(b + 208u));
                device const uchar* ql0 = b;
                device const uchar* qh0 = b + 128u;
                device const char*  sc0 = (device const char*)(b + 192u);
                device const float* xb = xr + (ulong)sb*256u;
                for (uint hf = 0u; hf < 2u; hf++) {
                    device const uchar* ql = ql0 + hf*64u;
                    device const uchar* qh = qh0 + hf*32u;
                    device const char*  sc = sc0 + hf*8u;
                    device const float* yb = xb + hf*128u;
                    for (uint l = 0u; l < 32u; l++) {
                        uint is = l/16u;
                        int q1 = int((ql[l]&0x0Fu)     | (((qh[l]>>0)&3u)<<4)) - 32;
                        int q2 = int((ql[l+32u]&0x0Fu) | (((qh[l]>>2)&3u)<<4)) - 32;
                        int q3 = int((ql[l]>>4)        | (((qh[l]>>4)&3u)<<4)) - 32;
                        int q4 = int((ql[l+32u]>>4)    | (((qh[l]>>6)&3u)<<4)) - 32;
                        acc += d*float(sc[is+0])*float(q1) * yb[l];
                        acc += d*float(sc[is+2])*float(q2) * yb[l+32u];
                        acc += d*float(sc[is+4])*float(q3) * yb[l+64u];
                        acc += d*float(sc[is+6])*float(q4) * yb[l+96u];
                    }
                }
            }
            res[r] += simd_sum(acc) * w;
        }
    }
    for (uint r = 0u; r < 4u; r++) {
        if (lane == 0u) { uint n = out_row + r; x[n] += res[r] + shx[n]/(1.0 + exp(-shg[0])); }
    }
}
"#;
