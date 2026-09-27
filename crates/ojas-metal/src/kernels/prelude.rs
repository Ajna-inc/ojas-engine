pub const PRELUDE: &str = r#"
#include <metal_stdlib>
using namespace metal;












// FFN gate activation: act=0 SiLU (Qwen/Llama), act=1 GeLU tanh-approx (Gemma).
// The cube can overflow to inf → fast-math tanh(inf)=NaN, so clamp the tanh arg.
inline float ffn_act(float g, uint act) {
    if (act == 1u) {
        float inner = 0.7978845608f*(g + 0.044715f*g*g*g);
        inner = clamp(inner, -30.0f, 30.0f);
        return 0.5f*g*(1.0f + tanh(inner));
    }
    return g/(1.0f+exp(-g));
}










// GEMV: y[nn] = Σ_k x[k]*w[nn,k], weight f16 [N,K] row-major. One simdgroup per
// output row, the best occupancy for small-model f16 decode: row-blocking hurt here,
// having no dequant to amortize and cutting the threadgroup count 4×.
// Vectorized half4/float4 loads (K mult of 4). Variants: plain / +bias / +=accum.
#define GEMV_DOT \
    uint n = tgid*8u + sgid; \
    if (n >= N) { return; } \
    device const half4* row = (device const half4*)(w + (ulong)n*(ulong)K); \
    device const float4* xv = (device const float4*)x; \
    uint K4 = K/4u; float p = 0.0; \
    for (uint k = lane; k < K4; k += 32u) { p += dot(float4(row[k]), xv[k]); } \
    p = simd_sum(p);




// Q8 GEMV: int8 weights (1 byte, half the bytes of f16) + per-row f32 scale that
// factors out of the sum. char4-vectorized; scale/bias/residual apply after. A
// 4-accumulator unroll was neutral — the inner loop is memory-bound, not
// FMA-latency-bound.
// The row index derives from simdgroups-per-threadgroup (ts/32) so the host tunes
// rows/threadgroup by choosing the thread count: small-N GEMVs (o_proj, ffn_down)
// use fewer rows/tg and so more threadgroups, hiding memory latency better (they
// sat at 240-335 GB/s against lm_head's 395 for want of threadgroups).
#define GEMV_Q8 \
    uint n = tgid*(ts/32u) + sgid; \
    if (n >= N) { return; } \
    device const char4* row = (device const char4*)(w + (ulong)n*(ulong)K); \
    device const float4* xv = (device const float4*)x; \
    uint K4 = K/4u; float p = 0.0; \
    for (uint k = lane; k < K4; k += 32u) { p += dot(float4(row[k]), xv[k]); } \
    p = simd_sum(p) * scale[n];




// Split-K Q8 GEMV (helper_mv_reduce_and_write pattern): one output row per
// threadgroup, its `nsg` simdgroups each reducing a different stripe of K and
// combining through threadgroup memory. For low-N long-K GEMVs (ffn_down N=896
// K=4864) that is nsg× more simdgroups and nsg× shorter loops; with 1 simdgroup the
// row was memory-starved at 242 GB/s. Synchronized by a threadgroup barrier, no grid
// sync needed.
#define GEMV_Q8_KSPLIT \
    uint n = tgid; \
    if (n >= N) { return; } \
    uint nsg = ts/32u; \
    device const char4* row = (device const char4*)(w + (ulong)n*(ulong)K); \
    device const float4* xv = (device const float4*)x; \
    uint K4 = K/4u; float pp = 0.0; \
    for (uint k = sgid*32u + lane; k < K4; k += nsg*32u) { pp += dot(float4(row[k]), xv[k]); } \
    pp = simd_sum(pp); \
    threadgroup float part[32]; \
    if (lane == 0u) { part[sgid] = pp; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float p = 0.0; \
    bool w0 = (sgid == 0u && lane == 0u); \
    if (sgid == 0u) { float v = (lane < nsg) ? part[lane] : 0.0; p = simd_sum(v) * scale[n]; }



// ===== NATIVE Q4 GEMV: per-32-block asymmetric 4-bit (nibbles + f16 scale + f16 min).
// Reads ~5 bits/weight (vs q8's 8) → ~1.6× less weight bandwidth. Each 32-block: 16
// bytes of nibbles (2/byte). Dequant w = scale*nib + min, so dot = scale*Σ(x*nib) +
// min*Σx. One simdgroup/row; lanes stride over the K/32 blocks. Buffers: w4(nibbles),
// scale(half), mn(half). Scale/bias/residual apply after (like q8).
// Scalar Q4 block (packing byte j = elem 2j low | 2j+1 high). axq=Σ(x·nib), sx=Σx.
// Reads each x once; manual float4 vectorization regressed twice, since this is
// overhead/latency-bound rather than ALU-bound.
#define Q4BLK(row, b, xb_, axq, sx) { \
    device const uchar* bl = (row) + (b)*16u; \
    device const float* xb = (xb_) + (b)*32u; \
    for (uint j = 0u; j < 16u; j++) { uchar by = bl[j]; \
        float x0 = xb[2u*j], x1 = xb[2u*j+1u]; \
        axq += x0*float(by & 0xFu) + x1*float(by >> 4u); sx += x0 + x1; } }
// Symmetric Q4_0: w = scale*(nib-8), so dot = scale*(Σx·nib - 8·Σx) = scale*(axq - 8*sx).
// No per-block min → 0.5625 bytes/weight (matches Q4_K), one fewer memory stream.
#define GEMV_Q4 \
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; } \
    uint nblk = K/32u; \
    device const uchar* row = w4 + (ulong)n*(ulong)(K/2u); \
    device const half* rsc = scale + (ulong)n*(ulong)nblk; \
    float p = 0.0; \
    for (uint b = lane; b < nblk; b += 32u) { \
        float axq = 0.0, sx = 0.0; \
        Q4BLK(row, b, x, axq, sx); \
        p += float(rsc[b])*(axq - 8.0*sx); \
    } \
    p = simd_sum(p);



// qmv_fast-style Q4 GEMV: 4 output rows/simdgroup (2 sg/tg = 8 rows/tg), each lane
// processing 16 weights/iter via uint16 vectorized loads. Nibbles are masked in place
// (& 0x00f0 = nib*16) and x is pre-divided (x/16), so x_pre*nib_masked = x*nib with no
// shift ALU. Symmetric Q4: dot = scale*(Σx·nib) - 8*scale*(Σx). Requires N%8==0.
// #define processes 16 weights for row r into acc (mask-in-place, pre-divided xt).
#define Q4FAST_DOT(wr_, r_, xt_, xsum_, sc_, res_) { \
    device const uint16_t* ws = (device const uint16_t*)((wr_) + (r_)*rb); \
    float s = float((sc_)[(r_)*nblk]); float acc = 0.0; \
    for (uint i = 0u; i < 4u; i++) { uint16_t wv = ws[i]; \
        acc += (xt_)[4u*i]*float(wv & 0x000fu) + (xt_)[4u*i+1u]*float(wv & 0x00f0u) \
             + (xt_)[4u*i+2u]*float(wv & 0x0f00u) + (xt_)[4u*i+3u]*float(wv & 0xf000u); } \
    (res_)[r_] += s*acc - 8.0*s*(xsum_); }
#define Q4FAST_LOADX(xr_, xt_, xsum_) \
    float xsum_ = 0.0; \
    for (uint i = 0u; i < 16u; i += 4u) { \
        float4 v4 = *(device const float4*)((xr_) + i); \
        xsum_ += v4.x+v4.y+v4.z+v4.w; \
        (xt_)[i]=v4.x; (xt_)[i+1u]=v4.y*(1.0/16.0); (xt_)[i+2u]=v4.z*(1.0/256.0); (xt_)[i+3u]=v4.w*(1.0/4096.0); }

// ---- Q4L: Q4_K's exact values in the fast Q4 layout -------------------------
//
// Q4_K decodes as `w = d1*nib - m1` per 32-weight sub-block. That is affine, and
// `Q4FAST_DOT` is already affine (`s*acc - 8*s*xsum`), so only the layout stops the
// tuned Q4 kernel from reading Q4_K directly: Q4_K interleaves a 16-byte header every
// 128 bytes of nibbles, which breaks coalescing and forces every lane to re-read the
// header. Measured 135 GB/s against 354 for the same weights laid out contiguously.
//
// Q4L keeps Q4_K's quantization exactly (same nibbles, same d1/m1) and only moves
// the bytes: nibbles contiguous per row in the fast kernel's pair order, with d1 and
// -m1 hoisted into two side arrays. Accuracy is Q4_K's; speed is the tuned family's.
// The relayout happens once, on the GPU, at load.
//
// The side arrays are f32, not f16. Q4_K forms d1 = d(f16) * scale(6-bit) in f32 at
// decode time, so rounding that product to f16 would lose precision Q4_K actually
// has — measured as 2.3e-3 error against the f64 oracle, versus 3e-7 with f32.
// The cost is 0.25 bits/weight on a ~4.5-bit format.
// One ushort4 vector load, not four scalar uint16 loads: this was the whole gap
// between the Q4L family and the Q8 kernels. Q8 loads char4 and reaches 389 GB/s of
// DRAM traffic; Q4L with four 2-byte scalar loads per step managed 154-276 depending
// on shape, and probes showed it insensitive to byte count, ALU count and row
// blocking, every variant keeping the same scalar loads. The address is 8B-aligned by
// construction (base + row*(K/2) + lane*8, K%16==0).
#define Q4L_DOT(wr_, r_, xt_, xsum_, a_, b_, res_) { \
    ushort4 wv4 = *(device const ushort4*)((wr_) + (r_)*rb); \
    float A = float((a_)[(r_)*nblk]); float B = float((b_)[(r_)*nblk]); float acc = 0.0; \
    for (uint i = 0u; i < 4u; i++) { uint16_t wv = wv4[i]; \
        acc += (xt_)[4u*i]*float(wv & 0x000fu) + (xt_)[4u*i+1u]*float(wv & 0x00f0u) \
             + (xt_)[4u*i+2u]*float(wv & 0x0f00u) + (xt_)[4u*i+3u]*float(wv & 0xf000u); } \
    (res_)[r_] += A*acc + B*(xsum_); }

#define GEMV_Q4L_FAST \
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; } \
    uint nblk = K/32u; uint rb = K/2u; \
    device const uchar* wr = w4 + (ulong)out_row*(ulong)rb + (ulong)lane*8u; \
    device const half* qa = qa_ + (ulong)out_row*(ulong)nblk + lane/2u; \
    device const half* qb = qb_ + (ulong)out_row*(ulong)nblk + lane/2u; \
    device const float* xr = x + lane*16u; \
    float res[4] = {0.0, 0.0, 0.0, 0.0}; \
    uint kfull = (K/512u)*512u; \
    for (uint k = 0u; k < kfull; k += 512u) { \
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum) \
        for (uint r = 0u; r < 4u; r++) Q4L_DOT(wr, r, xt, xsum, qa, qb, res) \
        wr += 256u; qa += 16u; qb += 16u; xr += 512u; \
    } \
    if (lane*16u < K - kfull) { \
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum) \
        for (uint r = 0u; r < 4u; r++) Q4L_DOT(wr, r, xt, xsum, qa, qb, res) \
    } \
    for (uint r = 0u; r < 4u; r++) res[r] = simd_sum(res[r]);

#define GEMV_Q4_FAST \
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; } \
    uint nblk = K/32u; uint rb = K/2u; \
    device const uchar* wr = w4 + (ulong)out_row*(ulong)rb + (ulong)lane*8u; \
    device const half* sc = scale + (ulong)out_row*(ulong)nblk + lane/2u; \
    device const float* xr = x + lane*16u; \
    float res[4] = {0.0, 0.0, 0.0, 0.0}; \
    uint kfull = (K/512u)*512u; \
    for (uint k = 0u; k < kfull; k += 512u) { \
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum) \
        for (uint r = 0u; r < 4u; r++) Q4FAST_DOT(wr, r, xt, xsum, sc, res) \
        wr += 256u; sc += 16u; xr += 512u; \
    } \
    if (lane*16u < K - kfull) { \
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum) \
        for (uint r = 0u; r < 4u; r++) Q4FAST_DOT(wr, r, xt, xsum, sc, res) \
    } \
    for (uint r = 0u; r < 4u; r++) res[r] = simd_sum(res[r]);



// K-SPLIT Q4: ONE output row per threadgroup, its `nsg` simdgroups (ts/32) each reduce
// a STRIPE of the K/32 blocks (all nsg*32 threads process blocks in parallel), combined
// via threadgroup memory. Fixes the short-loop overhead of gemv_q4 (which was 39% BW
// util: for K=2048 only ~2 blocks/lane) — same technique as gemv_q8_ksplit.
#define GEMV_Q4_KSPLIT \
    uint n = tgid; if (n >= N) { return; } \
    uint nsg = ts/32u; uint nblk = K/32u; uint gtid = sgid*32u + lane; \
    device const uchar* row = w4 + (ulong)n*(ulong)(K/2u); \
    device const half* rsc = scale + (ulong)n*(ulong)nblk; \
    float pp = 0.0; \
    for (uint b = gtid; b < nblk; b += nsg*32u) { \
        float axq = 0.0, sx = 0.0; \
        Q4BLK(row, b, x, axq, sx); \
        pp += float(rsc[b])*(axq - 8.0*sx); \
    } \
    pp = simd_sum(pp); \
    threadgroup float part[32]; \
    if (lane == 0u) { part[sgid] = pp; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float p = 0.0; bool w0 = (sgid == 0u && lane == 0u); \
    if (sgid == 0u) { float v = (lane < nsg) ? part[lane] : 0.0; p = simd_sum(v); }






// Q8 GEMV, 4 rows per simdgroup (qmv_fast structure). Each simdgroup keeps 4
// independent accumulators so 4 FMA/load chains stay in flight, moving the kernel
// from latency-bound (~30% of bandwidth) toward bandwidth-bound. x is loaded once per
// k and reused across the 4 rows. Threadgroup = 2 simdgroups (64 threads) = 8 rows/tg,
// same grid as gemv_q8. Requires N % 8 == 0, which every matmul here satisfies.
#define GEMV_Q8_R4 \
    uint row0 = tgid*8u + sgid*4u; \
    device const char4* r0 = (device const char4*)(w + (ulong)(row0+0u)*(ulong)K); \
    device const char4* r1 = (device const char4*)(w + (ulong)(row0+1u)*(ulong)K); \
    device const char4* r2 = (device const char4*)(w + (ulong)(row0+2u)*(ulong)K); \
    device const char4* r3 = (device const char4*)(w + (ulong)(row0+3u)*(ulong)K); \
    device const float4* xv = (device const float4*)x; \
    uint K4 = K/4u; float p0=0.0,p1=0.0,p2=0.0,p3=0.0; \
    for (uint k = lane; k < K4; k += 32u) { \
        float4 xx = xv[k]; \
        p0 += dot(float4(r0[k]), xx); p1 += dot(float4(r1[k]), xx); \
        p2 += dot(float4(r2[k]), xx); p3 += dot(float4(r3[k]), xx); \
    } \
    p0 = simd_sum(p0)*scale[row0+0u]; p1 = simd_sum(p1)*scale[row0+1u]; \
    p2 = simd_sum(p2)*scale[row0+2u]; p3 = simd_sum(p3)*scale[row0+3u];






// Fused RMSNorm + Q/K/V projection. Each threadgroup RMS-normalizes the residual
// x into threadgroup memory once (nw = norm weight), then its simdgroups project
// from that on-chip copy — saves a dispatch AND serves the GEMV input from fast
// threadgroup memory instead of device. XN_MAX bounds the model dim.
#define XN_MAX 2048




// ================= Multi-token (batched) kernels for speculative decoding =====
// Process M tokens (M<=8) in ONE dispatch. Weight-reuse GEMV: each simdgroup
// loads a weight row once and applies it to all M token activations (X is [M,K]
// row-major). This is what makes verifying a K-token draft cost ~1 forward.























// ============== qwen35 prefill (M-token chunk) kernels ==============
// Q4 weight-reuse GEMV for M tokens: each simdgroup owns one output row and dequants
// each 32-block once, applying it to all M token rows (X [M,K] row-major, Y [M,N]
// row-major). Reading weights once per chunk instead of once per token is what makes
// chunked prefill ~M× faster, decode being weight-bandwidth-bound.
#define GEMV_M_Q4_BODY \
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; } \
    device const uchar2* row = (device const uchar2*)(w4 + (ulong)n*(ulong)(K/2u)); \
    device const half* rsc = scale + (ulong)n*(ulong)(K/32u); \
    uint K4 = K/4u; \
    float p[256]; for (uint m=0u;m<M;m++) { p[m]=0.0; } \
    for (uint k = lane; k < K4; k += 32u) { \
        uchar2 by = row[k]; \
        float4 wv = (float4(float(by.x & 0xFu), float(by.x >> 4u), float(by.y & 0xFu), float(by.y >> 4u)) - 8.0) * float(rsc[k/8u]); \
        for (uint m=0u;m<M;m++) { device const float4* xm = (device const float4*)(x + (ulong)m*(ulong)K); p[m] += dot(wv, xm[k]); } \
    }











// Compile-time M=2 batched GEMV for MTP verify: two tokens share each weight read
// (spec-verify ~ the cost of one decode forward). Same qmv_fast pattern as the
// decode kernel (uint16 loads, mask-in-place no-shift dequant, 4 rows/simdgroup,
// 64 threads = 8 rows/tg) with dual x streams.
#define GEMV_M2_FAST_BODY \
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; } \
    uint nblk = K/32u; uint rb = K/2u; \
    device const uchar* wr = w4 + (ulong)out_row*(ulong)rb + lane*8u; \
    device const half* sc = scale + (ulong)out_row*(ulong)nblk + lane/2u; \
    device const float* xr0 = x + lane*16u; \
    device const float* xr1 = x + (ulong)K + lane*16u; \
    float r0[4] = {0.0, 0.0, 0.0, 0.0}; \
    float r1[4] = {0.0, 0.0, 0.0, 0.0}; \
    uint kfull = (K/512u)*512u; \
    for (uint k = 0u; k < kfull; k += 512u) { \
        float xt0[16]; Q4FAST_LOADX(xr0, xt0, xsum0) \
        float xt1[16]; Q4FAST_LOADX(xr1, xt1, xsum1) \
        for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(wr, r, xt0, xsum0, sc, r0) Q4FAST_DOT(wr, r, xt1, xsum1, sc, r1) } \
        wr += 256u; sc += 16u; xr0 += 512u; xr1 += 512u; \
    } \
    if (lane*16u < K - kfull) { \
        float xt0[16]; Q4FAST_LOADX(xr0, xt0, xsum0) \
        float xt1[16]; Q4FAST_LOADX(xr1, xt1, xsum1) \
        for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(wr, r, xt0, xsum0, sc, r0) Q4FAST_DOT(wr, r, xt1, xsum1, sc, r1) } \
    }

















// ===================== Q4_K native dequant (faithful, no requant) =====================
// block_q4_K = { half d; half dmin; u8 scales[12]; u8 qs[128]; } = 144 B / 256 weights.
// 8 sub-blocks of 32; sub-block scale/min are 6-bit (get_scale_min_k4), scaled by d/dmin.
// Q4K_DOT adds super-block SB of row WR (dot with XB) into ACC.
// Q4_K dot, one super-block, all 32 lanes cooperating.
//
// The obvious loop — `for (sb = lane; sb < nsb; sb += 32)` — gives one 256-weight
// super-block per lane, so a K=2048 row has 8 super-blocks and leaves 24 of 32 lanes
// idle. That, not the arithmetic, is why the native path measured slower than Q8
// despite reading half the bytes.
//
// Here lane L owns 8 consecutive weights: sub-block `j = L>>2`, offset `(L&3)*8`
// inside it. An 8-run never straddles a 32-weight sub-block, so the 6-bit scale/min
// pair is constant per lane and unpacked once. Loads are uchar4/float4: `_b` is
// 16-aligned (144 B stride), and the float offsets land on multiples of 16 B.
// Q4_K dot, one lane = one 64-weight nibble group.
//
// `Q4K_DOT_L` fixed occupancy but left the arithmetic lopsided: every lane unpacks a
// 6-bit scale/min pair to do only 8 multiply-adds over 8 bytes, measuring 206 GB/s
// against the Q8 gemv's 386 on the same machine — that overhead, not memory.
//
// Here lane L takes nibble group `g = L & 3` of super-block `SB`: the group's whole
// 32-byte run, whose low nibbles are sub-block 2g and high nibbles 2g+1. One lane
// unpacks two scale pairs and does 64 MACs over 32 contiguous bytes — 4x fewer
// unpacks per weight than the per-8 split, with wide aligned loads.
#define Q4K_DOT_G(WR, SB, XB, G, ACC) { \
    device const uchar* _b = (WR) + (ulong)(SB)*144u; \
    float _d = float(*(device const half*)_b); float _dm = float(*(device const half*)(_b+2)); \
    device const uchar* _sc = _b+4; \
    uint _jl=(G)*2u, _jh=(G)*2u+1u; uint _sl,_ml,_sh,_mh; \
    if(_jl<4u){_sl=_sc[_jl]&63u;_ml=_sc[_jl+4u]&63u;} \
    else{_sl=(_sc[_jl+4u]&0x0Fu)|((_sc[_jl-4u]>>6)<<4);_ml=(_sc[_jl+4u]>>4)|((_sc[_jl]>>6)<<4);} \
    if(_jh<4u){_sh=_sc[_jh]&63u;_mh=_sc[_jh+4u]&63u;} \
    else{_sh=(_sc[_jh+4u]&0x0Fu)|((_sc[_jh-4u]>>6)<<4);_mh=(_sc[_jh+4u]>>4)|((_sc[_jh]>>6)<<4);} \
    float _dl=_d*float(_sl), _mll=_dm*float(_ml); \
    float _dh=_d*float(_sh), _mhh=_dm*float(_mh); \
    device const uchar4* _q4 = (device const uchar4*)(_b + 16u + (G)*32u); \
    device const float4* _xl = (device const float4*)((XB) + (ulong)(SB)*256u + _jl*32u); \
    device const float4* _xh = (device const float4*)((XB) + (ulong)(SB)*256u + _jh*32u); \
    for(uint _i=0u;_i<8u;_i++){ \
      uchar4 _v=_q4[_i]; \
      (ACC) += dot(_dl*float4(_v & (uchar4)0x0F)-_mll, _xl[_i]); \
      (ACC) += dot(_dh*float4(_v >> (uchar4)4)-_mhh, _xh[_i]); } }

#define Q4K_DOT_L(WR, SB, XB, LANE, ACC) { \
    device const uchar* _b = (WR) + (ulong)(SB)*144u; \
    float _d = float(*(device const half*)_b); float _dm = float(*(device const half*)(_b+2)); \
    device const uchar* _sc = _b+4; \
    uint _j = (LANE) >> 2; uint _o = ((LANE) & 3u) * 8u; \
    uint _s,_m; \
    if(_j<4u){_s=_sc[_j]&63u;_m=_sc[_j+4u]&63u;} \
    else{_s=(_sc[_j+4u]&0x0Fu)|((_sc[_j-4u]>>6)<<4);_m=(_sc[_j+4u]>>4)|((_sc[_j]>>6)<<4);} \
    float _d1=_d*float(_s), _m1=_dm*float(_m); \
    device const uchar4* _q4 = (device const uchar4*)(_b + 16u + (_j>>1)*32u + _o); \
    device const float4* _x4 = (device const float4*)((XB) + (ulong)(SB)*256u + _j*32u + _o); \
    uint _hi = _j & 1u; \
    for (uint _i=0u;_i<2u;_i++){ \
      uchar4 _v=_q4[_i]; \
      float4 _nb = _hi ? float4(_v >> (uchar4)4) : float4(_v & (uchar4)0x0F); \
      (ACC) += dot(_d1*_nb-_m1, _x4[_i]); } }

#define Q4K_DOT(WR, SB, XB, ACC) { \
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
// Q6_K (native GGUF): block = 210 B per 256 weights —
//   u8 ql[128] (low nibbles) | u8 qh[64] (2 high bits, 4 weights per byte)
//   | i8 scales[16] (one per 16 weights) | half d (super-block scale).
// Two halves of 128 weights; each advances ql by 64, qh by 32, scales by 8.
// Weight = d * scales[is] * ((ql_nibble | (qh_bits << 4)) - 32).
// Mirrors ojas-formats `dequant_to_f16` arm 14, which matches the reference
// `dequantize_row_q6_K` — the CPU oracle for this kernel.
// 210 is only 2-byte aligned, so this reads bytes and one `half` at +208 (even). No
// 4- or 16-byte loads: the block stride would break them.
// Q6_K dot, one super-block, all 32 lanes cooperating — same fix as Q4K_DOT_L.
//
// Lane L owns 8 consecutive weights: half `h = L>>4`, offset `(L&15)*8` inside it.
// A 128-weight half is four 32-weight groups (low nibbles of ql[0..32], of
// ql[32..64], then the high nibbles of each), and an 8-run stays inside one group,
// so the group index, the 2-bit shift into qh, and the int8 scale are all constant
// per lane.
//
// Stays scalar rather than uchar4: the 210 B block stride is only 2-byte aligned,
// so vector loads off `_b` are not safe. The win here is occupancy, not width.
#define Q6K_DOT_L(WR, SB, XB, LANE, ACC) { \
    device const uchar* _b = (WR) + (ulong)(SB)*210u; \
    float _d = float(*(device const half*)(_b + 208u)); \
    uint _h = (LANE) >> 4; uint _o = ((LANE) & 15u) * 8u; \
    uint _q = _o >> 5; uint _li = _o & 31u; uint _is = _li >> 4; \
    device const uchar* _qlp = _b + _h*64u + ((_q & 1u) ? 32u : 0u) + _li; \
    device const uchar* _qhp = _b + 128u + _h*32u + _li; \
    float _dl = _d * float(((device const char*)(_b + 192u))[_h*8u + _is + _q*2u]); \
    uint _sh = _q * 2u; bool _high = _q >= 2u; \
    device const float* _xp = (XB) + (ulong)(SB)*256u + _h*128u + _o; \
    for (uint _i=0u;_i<8u;_i++){ \
      uint _lo = uint(_qlp[_i]); uint _hb = (uint(_qhp[_i]) >> _sh) & 3u; \
      float _v = _dl * float(int((_high ? (_lo>>4) : (_lo&0x0Fu)) | (_hb<<4)) - 32); \
      (ACC) += _v * _xp[_i]; } }

#define Q6K_DOT(WR, SB, XB, ACC) { \
    device const uchar* _b = (WR) + (ulong)(SB)*210u; \
    device const uchar* _ql = _b; device const uchar* _qh = _b + 128u; \
    device const char* _sc = (device const char*)(_b + 192u); \
    float _d = float(*(device const half*)(_b + 208u)); \
    device const float* _xb = (XB) + (ulong)(SB)*256u; \
    for (uint _h=0u;_h<2u;_h++){ \
      uint _qlb=_h*64u, _qhb=_h*32u, _scb=_h*8u, _yb=_h*128u; \
      for (uint _l=0u;_l<32u;_l++){ uint _is=_l>>4; \
        uint _lo=uint(_ql[_qlb+_l]), _lo2=uint(_ql[_qlb+_l+32u]), _hb=uint(_qh[_qhb+_l]); \
        float _q1=float(int((_lo &0x0Fu)|(( _hb     &3u)<<4))-32); \
        float _q2=float(int((_lo2&0x0Fu)|(((_hb>>2)&3u)<<4))-32); \
        float _q3=float(int(( _lo >>4   )|(((_hb>>4)&3u)<<4))-32); \
        float _q4=float(int(( _lo2>>4   )|(((_hb>>6)&3u)<<4))-32); \
        (ACC) += _d*float(_sc[_scb+_is    ])*_q1*_xb[_yb+_l]; \
        (ACC) += _d*float(_sc[_scb+_is+2u])*_q2*_xb[_yb+_l+32u]; \
        (ACC) += _d*float(_sc[_scb+_is+4u])*_q3*_xb[_yb+_l+64u]; \
        (ACC) += _d*float(_sc[_scb+_is+6u])*_q4*_xb[_yb+_l+96u]; } } }
// Row walkers: identical block math to Q4K_DOT / Q6K_DOT, but they expose each
// weight as `_v` at column `_idx` and run BODY, instead of folding into a dot.
// Used by the GPU requantizers (and anything else needing raw weights).

// Decode one weight at index `i` of a Q6_K row.
//
// Q6K_ROW walks a super-block in its native order; this is the inverse map, for
// callers that want a specific element (an embedding gather) or a run of them (a
// GEMM A-tile fill). Layout per 210-byte super-block: ql[128] qh[64] sc[16] d.
//
//   h = i/128, r = i%128, q = r/32, l = r%32
//   value = d * sc[8h + l/16 + 2q] * ((ql[64h + 32(q&1) + l] >> 4(q>=2) & 0xF)
//                                     | ((qh[32h + l] >> 2q & 3) << 4) - 32)
//
// A 16-aligned run of 16 shares one super-block, one h, one q and one l/16, so the
// scale and `d` hoist out of the loop entirely, which is what makes the tiled fill
// cheap.
#define Q6K_AT(ROW, I, OUT) { \
    uint _sb = (I) / 256u, _io = (I) % 256u; \
    device const uchar* _b = (ROW) + (ulong)_sb*210ul; \
    uint _h = _io / 128u, _r = _io % 128u, _q = _r / 32u, _l = _r % 32u; \
    float _d = float(*(device const half*)(_b + 208u)); \
    float _s = _d * float(((device const char*)(_b + 192u))[_h*8u + _l/16u + 2u*_q]); \
    uint _lo = (uint(_b[_h*64u + (_q & 1u)*32u + _l]) >> ((_q >= 2u) ? 4u : 0u)) & 0x0Fu; \
    uint _hi = (uint(_b[128u + _h*32u + _l]) >> (2u*_q)) & 3u; \
    OUT = _s * float(int(_lo | (_hi << 4u)) - 32); }


// Q5_K row walker. Block = 176 B / 256 weights:
//   half d | half dmin | u8 scales[12] (6-bit d/m pairs) | u8 qh[32] | u8 qs[128]
// Four 64-weight chunks; each consumes two 6-bit scale/min pairs and one high-bit
// mask that shifts left by 2 per chunk. Mirrors `dequant_to_f16` arm 13.

// Q2_K row walker. Block = 84 B / 256 weights:
//   u8 scales[16] | u8 qs[64] | half d | half dmin
// Unlike every other K-quant, d/dmin sit at the end of the block. Sixteen
// sub-blocks of 16; each scales[] byte packs a 4-bit scale (low nibble) and a
// 4-bit min (high). w = d*sc*q - dmin*m. Mirrors `dequant_to_f16` arm 10.

// Pull scale `I` (0..15) out of Q3_K's four unpacked scale words.
//
// Selected with nested ternaries rather than a local `uint _aux[4]`: a
// runtime-indexed thread-local array spills to device-backed thread memory on Apple
// GPUs, the trap that made `gemv_mv_q4l` run 47 ms against 19 ms. The unpacked field
// is 6 bits (max 0x3F), so it is never negative as an int8 and the sign extension the
// format relies on is a no-op here.

// Q3_K row walker. Block = 110 B / 256 weights:
//   u8 hmask[32] | u8 qs[64] | u8 scales[12] | half d
// 3-bit: 2 low bits in qs, the 3rd in hmask as a per-weight inverted bit (set = 0,
// clear = subtract 4). The 12 scale bytes hold sixteen 6-bit biased scales in the
// kmask1/kmask2 packing, reproduced so values match bit for bit. `_m` advances once
// per j and carries across the two halves: 1,2,4,...,128.
// Mirrors `dequant_to_f16` arm 11.

// The IQ4 codebook: the IQ4_NL/XS formats' fixed dequant values, shared by both. It
// lives in `constant` address space because, unlike a thread-local array, a runtime
// index into constant memory does not spill to device-backed thread memory, so the
// lookup stays a load rather than a stall.

// IQ4_NL row walker. Block = 18 B / 32 weights: half d | u8 qs[16].
// "NL" = non-linear: the 4 bits index the codebook above rather than a uniform ramp.
// Low nibbles fill the first half of the block, high nibbles the second; they are not
// interleaved. Mirrors `dequant_to_f16` arm 20.

// IQ4_XS row walker. Block = 136 B / 256 weights:
//   half d | u16 scales_h | u8 scales_l[4] | u8 qs[128]
// Same codebook as IQ4_NL, but eight sub-blocks of 32 share one `d` and each
// carries a 6-bit scale split across two arrays: 4 low bits nibble-packed in
// scales_l, 2 high bits in scales_h. Biased by 32 like the K-quants.
// Mirrors `dequant_to_f16` arm 23.

// Legacy 32-weight walkers. In all four, weights [0..16) are the low nibbles of
// qs[0..16] and [16..32) are the high nibbles, not interleaved 2j/2j+1 per byte.
// Mirrors `dequant_to_f16` arms 2/3/6/7.

// Q4_0: block { half d; u8 qs[16]; } = 18 B / 32; w = d*(nib-8)

// Q4_1: block { half d; half m; u8 qs[16]; } = 20 B / 32; w = nib*d + m

// Q5_0: block { half d; u8 qh[4]; u8 qs[16]; } = 22 B / 32; w = d*(q-16).
// The 5th bit of weight j lives in bit j of the qh word (bit j+16 for the high
// nibbles), not alongside the nibble.

// Q5_1: block { half d; half m; u8 qh[4]; u8 qs[16]; } = 24 B / 32; w = q*d + m

// Q8_0 row walker. Block = 34 B / 32 weights: half d | i8 qs[32]; w = d*q.

// Q2_0 (ternary g128): block = 34 B per 128 weights — f16 scale
// then 32 B of 2-bit codes packed sequentially (weight j at byte j/4, bits
// (j%4)*2), w = (code-1)*scale. The sequential packing is what lets one byte
// feed exactly one float4 of activations, so the inner loop is a plain dot().
// Never requantize these: every non-zero weight sits at +-amax, so Q4_0's
// asymmetric clamp biases every positive weight low and wrecks the model.
// Q2_0 (ternary g128) after the load-time repack: codes are 32 B/block contiguous
// (so 16-byte loads are legal — the on-disk 34 B stride is not 4-aligned), scales
// split into their own f16 array. Layout per row n: codes[n*nblk*32], scales[n*nblk].
// w = (code-1)*scale, codes packed sequentially (weight j at byte j/4, bits (j%4)*2),
// so one code byte feeds exactly one float4 of activations.
//
// One simdgroup per output row, lanes striding whole 128-weight blocks. 4
// rows/simdgroup, to amortize activation loads, measured worse (42.8 -> 36.1): x is
// only ~16 KB and stays cache-resident across rows, so there is no activation DRAM
// traffic to save, and the per-row scale loads plus the bounds-check `break` cost
// more than they save.
#define Q20_DOT(cr, sr, b, x, acc) { \
    float d = float((sr)[(b)]); \
    device const uint4* c4 = (device const uint4*)((cr) + (ulong)(b)*32ul); \
    device const float4* x4 = (device const float4*)((x) + (b)*128u); \
    float s = 0.0; \
    for (uint h = 0u; h < 2u; h++) { \
        uint4 u4 = c4[h]; \
        for (uint w = 0u; w < 4u; w++) { \
            uint u = u4[w]; \
            for (uint t = 0u; t < 4u; t++) { \
                uint by = (u >> (t*8u)) & 0xFFu; \
                float4 wv = float4(float(by & 3u), float((by >> 2) & 3u), \
                                   float((by >> 4) & 3u), float((by >> 6) & 3u)) - 1.0; \
                s += dot(wv, x4[h*16u + w*4u + t]); \
            } \
        } \
    } \
    acc += d * s; \
}








"#;
