//! Training kernel family (t_*): fwd/bwd/optimizer — entry names aligned with CUDA.
pub const TRAIN_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// logits[c*V+v] = dot(h[c], w[v]) — one simdgroup per (v, c)
kernel void flce_logits(device const float* h [[buffer(0)]], device const half* w [[buffer(1)]],
    device float* logits [[buffer(2)]], constant uint& D [[buffer(3)]], constant uint& V [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint v = tg.x*8u + sgid;
    if (v >= V) { return; }
    device const half4* wr = (device const half4*)(w + (ulong)v*(ulong)D);
    float acc[8] = {0.0};
    for (uint k = lane; k < D/4u; k += 32u) {
        float4 wv = float4(wr[k]);
        for (uint c = 0; c < M; c++) {
            acc[c] += dot(wv, ((device const float4*)(h + (ulong)c*(ulong)D))[k]);
        }
    }
    for (uint c = 0; c < M; c++) {
        float a = simd_sum(acc[c]);
        if (lane == 0) { logits[(ulong)c*(ulong)V + v] = a; }
    }
}

// Per row: loss[c0+c] = lse - logit[target]; overwrite row with
// (softmax - onehot) * scale. One threadgroup (1024 threads) per row.
kernel void flce_row(device float* logits [[buffer(0)]], device const float* tgt [[buffer(1)]],
    device float* loss [[buffer(2)]], constant uint& V [[buffer(3)]],
    constant float& scale [[buffer(4)]], constant uint& c0 [[buffer(5)]],
    uint c [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint nth [[threads_per_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    device float* row = logits + (ulong)c*(ulong)V;
    threadgroup float red[32];
    uint nsg = nth / 32u;
    float m = -INFINITY;
    for (uint v = tid; v < V; v += nth) { m = max(m, row[v]); }
    m = simd_max(m);
    if (lane == 0) { red[sgid] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float x = (lane < nsg) ? red[lane] : -INFINITY;
        x = simd_max(x);
        if (lane == 0) { red[0] = x; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0;
    for (uint v = tid; v < V; v += nth) { s += exp(row[v] - m); }
    s = simd_sum(s);
    if (lane == 0) { red[sgid] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float x = (lane < nsg) ? red[lane] : 0.0;
        x = simd_sum(x);
        if (lane == 0) { red[0] = x; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = red[0];
    uint y = (uint)tgt[c0 + c];
    bool ign = y >= V;                       // ignore-index (masked target)
    if (tid == 0) { loss[c0 + c] = ign ? 0.0f : m + log(s) - row[y]; }
    threadgroup_barrier(mem_flags::mem_device); // loss reads row[y] before overwrite
    for (uint v = tid; v < V; v += nth) {
        float p = exp(row[v] - m) / s;
        row[v] = ign ? 0.0f : (p - (v == y ? 1.0f : 0.0f)) * scale;
    }
}

// x[t,:] = emb[tok[t],:]
kernel void t_embed_fwd(device const float* tok [[buffer(0)]], device const half* emb [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& D [[buffer(3)]], constant uint& TOT [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }
    uint t = g / D, i = g % D;
    x[g] = float(emb[(ulong)((uint)tok[t])*D + i]);
}
// demb[v,:] = sum_{t: tok[t]==v} dx[t,:] — deterministic (tg per vocab row)
kernel void t_embed_bwd(device const float* tok [[buffer(0)]], device const float* dx [[buffer(1)]],
    device float* demb [[buffer(2)]], constant uint& D [[buffer(3)]], constant uint& T [[buffer(4)]],
    uint v [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint nth [[threads_per_threadgroup]]) {
    for (uint i = tid; i < D; i += nth) {
        float acc = 0.0;
        for (uint t = 0; t < T; t++) {
            if ((uint)tok[t] == v) { acc += dx[(ulong)t*D + i]; }
        }
        demb[(ulong)v*D + i] = acc;
    }
}

// Fused AdamW, f32 moments. One thread per param.
kernel void t_adamw(device float* w [[buffer(0)]], device const float* g [[buffer(1)]],
    device float* m [[buffer(2)]], device float* v [[buffer(3)]],
    constant float& lr [[buffer(4)]], constant float& b1 [[buffer(5)]], constant float& b2 [[buffer(6)]],
    constant float& bc1 [[buffer(7)]], constant float& bc2 [[buffer(8)]], // 1/(1-b^t)
    constant float& wd [[buffer(9)]], constant uint& N [[buffer(10)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= N) { return; }
    float gi = g[i];
    float mi = b1*m[i] + (1.0f - b1)*gi;
    float vi = b2*v[i] + (1.0f - b2)*gi*gi;
    m[i] = mi; v[i] = vi;
    float mh = mi*bc1, vh = vi*bc2;
    w[i] -= lr*(mh/(sqrt(vh) + 1e-8f) + wd*w[i]);
}

// Fused AdamW with compact moments: m f16, v log-u8 per 256-block.
// One SIMDGROUP per block (8 params/lane, vectorized): absmax via simd_max
// only — zero threadgroup barriers. Fast exp2/log2 bit-hack codes (~1% err).
// SR write keeps f16 weights unbiased.
kernel void t_adamw_8h(device half* w [[buffer(0)]], device const float* g [[buffer(1)]],
    device half* mh [[buffer(2)]], device uchar* vq [[buffer(3)]],
    device float* vsc [[buffer(4)]],
    constant float& lr [[buffer(5)]], constant float& b1 [[buffer(6)]], constant float& b2 [[buffer(7)]],
    constant float& bc1 [[buffer(8)]], constant float& bc2 [[buffer(9)]],
    constant float& wd [[buffer(10)]], constant uint& N [[buffer(11)]],
    constant uint& stp [[buffer(12)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const float K2 = 33.219281f; // log2(1e10)
    uint blk = tgid*8u + sgid;
    uint i0 = blk*256u + uint(lane)*8u;
    float vs_old = vsc[blk];
    float mi[8], vi[8], wnew[8];
    float vmax = 0.0;
    for (uint k = 0; k < 8; k++) {
        uint i = i0 + k;
        if (i >= N) { vi[k] = 0.0; mi[k] = 0.0; wnew[k] = 0.0; continue; }
        float gi = g[i];
        float m = b1*float(mh[i]) + (1.0f - b1)*gi;
        uchar c = vq[i];
        float xe = -K2*float(255u - c)/255.0f + 126.94269504f;
        float vprev = (c == 0) ? 0.0f : vs_old*as_type<float>(uint(xe*8388608.0f));
        float v = b2*vprev + (1.0f - b2)*gi*gi;
        float wf = float(w[i]);
        wnew[k] = wf - lr*((m*bc1)/(sqrt(v*bc2) + 1e-8f) + wd*wf);
        mi[k] = m; vi[k] = v;
        vmax = max(vmax, v);
    }
    vmax = simd_max(vmax);
    for (uint k = 0; k < 8; k++) {
        uint i = i0 + k;
        if (i >= N) { continue; }
        float x = wnew[k];
        half h0 = half(x);
        float f0 = float(h0);
        if (f0 != x && isfinite(x)) {
            ushort b = as_type<ushort>(h0);
            bool up = x > f0;
            ushort bn = ((f0 >= 0.0f) == up) ? b + 1 : b - 1;
            if (f0 == 0.0f) { bn = up ? ushort(1) : ushort(0x8001); }
            half h1 = as_type<half>(bn);
            uint r = (i ^ (stp*2654435761u))*2246822519u;
            r ^= r >> 15; r *= 2654435761u; r ^= r >> 13;
            float u = float(r & 0xFFFFFFu)/16777216.0f;
            float pr = (x - f0)/(float(h1) - f0);
            w[i] = (u < pr) ? h1 : h0;
        } else { w[i] = h0; }
        mh[i] = half(mi[k]);
        if (vi[k] <= 0.0f || vmax <= 0.0f) { vq[i] = 0; }
        else {
            float l2 = float(as_type<uint>(vi[k]/vmax))*1.1920929e-7f - 126.94269504f;
            vq[i] = uchar(clamp(round(255.0f + 255.0f/K2*l2), 1.0f, 255.0f));
        }
    }
    if (lane == 0) { vsc[blk] = vmax; }
}

// dh[c,i] = sum_v dl[c,v]*w[v,i] — one threadgroup per c, coalesced over w rows.
// Threads own strided columns; D/nth accumulators in registers (<=16).
kernel void flce_dh(device const float* dl [[buffer(0)]], device const half* w [[buffer(1)]],
    device float* dh [[buffer(2)]], constant uint& D [[buffer(3)]], constant uint& V [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]]) {
    uint i = tg.x*256u + tid2.x;
    if (i >= D) { return; }
    float acc[8] = {0.0};
    for (uint v = 0; v < V; v++) {
        float wv = (float)w[(ulong)v*(ulong)D + i];
        for (uint c = 0; c < M; c++) { acc[c] += dl[(ulong)c*(ulong)V + v]*wv; }
    }
    for (uint c = 0; c < M; c++) { dh[(ulong)c*(ulong)D + i] = acc[c]; }
}

// dw[v,i] += sum_c dl[c,v]*h[c,i] — one threadgroup per v row.
kernel void flce_dw(device const float* dl [[buffer(0)]], device const float* h [[buffer(1)]],
    device float* dw [[buffer(2)]], constant uint& D [[buffer(3)]], constant uint& V [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    uint v [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint nth [[threads_per_threadgroup]]) {
    for (uint i = tid; i < D; i += nth) {
        float acc = 0.0;
        for (uint c = 0; c < C; c++) { acc += dl[(ulong)c*(ulong)V + v] * h[(ulong)c*(ulong)D + i]; }
        dw[(ulong)v*(ulong)D + i] += acc;
    }
}

// ======== layer training kernels (f32, HF layout; validated vs grad.rs) ========
constant float TEPS = 1e-6f;
inline float tsig(float x) { return 1.0f/(1.0f+exp(-x)); }
inline float tsilu(float x) { return x*tsig(x); }
inline float tdsilu(float x) { float s = tsig(x); return s*(1.0f + x*(1.0f-s)); }

// y[m,o] = dot(x[m,:], w[o,:])  (w f32 [OUT,IN] row-major).
// m-tiled: each weight read is amortized over TMT tokens (x stays in cache).
#define TMT 16
kernel void t_gemm_xwT(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& IN [[buffer(3)]], constant uint& OUT [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint o = tg.x*8u + sgid;
    uint m0 = tg.y*TMT;
    if (o >= OUT) { return; }
    uint mtn = min(uint(TMT), M - m0);
    device const float4* wr = (device const float4*)(w + (ulong)o*IN);
    float acc[TMT] = {0.0};
    for (uint k = lane; k < IN/4u; k += 32u) {
        float4 wv = wr[k];
        for (uint mt = 0; mt < mtn; mt++) {
            acc[mt] += dot(wv, ((device const float4*)(x + (ulong)(m0 + mt)*IN))[k]);
        }
    }
    for (uint mt = 0; mt < mtn; mt++) {
        float a = simd_sum(acc[mt]);
        if (lane == 0) { y[(ulong)(m0 + mt)*OUT + o] = a; }
    }
}

// dx[m,i] (+)= sum_o dy[m,o]*w[o,i] — grid (IN/1024, m-blocks).
// dy tile staged in threadgroup memory (kills the 16x redundant scalar loads);
// float4 on i (4x work per load instruction). IN must be %4.
kernel void t_gemm_dx(device const float* dy [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* dx [[buffer(2)]], constant uint& IN [[buffer(3)]], constant uint& OUT [[buffer(4)]],
    constant uint& accum [[buffer(5)]], constant uint& M [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]]) {
    uint tid = tid2.x;
    uint i4 = tg.x*256u + tid;              // float4 index
    uint m0 = tg.y*8u;
    uint mtn = min(8u, M - m0);
    threadgroup float dys[8*64];
    float4 acc[8];
    for (uint mt = 0; mt < 8; mt++) { acc[mt] = float4(0.0); }
    bool live = i4*4u < IN;
    for (uint o0 = 0; o0 < OUT; o0 += 64u) {
        for (uint l = 0; l < 2; l++) {
            uint idx = tid*2u + l;          // covers 8*64 = 512 slots
            uint mt = idx / 64u, oo = idx % 64u;
            dys[idx] = (mt < mtn && o0 + oo < OUT) ? dy[(ulong)(m0 + mt)*OUT + o0 + oo] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            uint on = min(64u, OUT - o0);
            for (uint oo = 0; oo < on; oo++) {
                float4 wv = ((device const float4*)(w + (ulong)(o0 + oo)*IN))[i4];
                for (uint mt = 0; mt < mtn; mt++) { acc[mt] += wv*dys[mt*64u + oo]; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!live) { return; }
    for (uint mt = 0; mt < mtn; mt++) {
        device float4* dxr = (device float4*)(dx + (ulong)(m0 + mt)*IN);
        dxr[i4] = (accum != 0 ? dxr[i4] : float4(0.0)) + acc[mt];
    }
}

// dw[o,i] = sum_m dy[m,o]*x[m,i] — grid (IN/1024, OUT/16). dy tile staged
// in threadgroup memory once (M <= 256); float4 on i.
kernel void t_gemm_dw(device const float* dy [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* dw [[buffer(2)]], constant uint& IN [[buffer(3)]], constant uint& OUT [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]]) {
    uint tid = tid2.x;
    uint i4 = tg.x*256u + tid;
    uint o0 = tg.y*8u;
    uint otn = min(8u, OUT - o0);
    threadgroup float dyt[256*8];
    for (uint idx = tid; idx < M*8u; idx += 256u) {
        uint mm = idx / 8u, ot = idx % 8u;
        dyt[idx] = (ot < otn) ? dy[(ulong)mm*OUT + o0 + ot] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i4*4u >= IN) { return; }
    float4 acc[8];
    for (uint ot = 0; ot < 8; ot++) { acc[ot] = float4(0.0); }
    for (uint m = 0; m < M; m++) {
        float4 xv = ((device const float4*)(x + (ulong)m*IN))[i4];
        for (uint ot = 0; ot < otn; ot++) { acc[ot] += xv*dyt[m*8u + ot]; }
    }
    for (uint ot = 0; ot < otn; ot++) {
        ((device float4*)(dw + (ulong)(o0 + ot)*IN))[i4] = acc[ot];
    }
}

// RMSNorm rows (mean variant): R rows of width N; one simdgroup per row.
kernel void t_rms_fwd(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]], device float* rsave [[buffer(3)]],
    constant uint& N [[buffer(4)]], constant uint& R [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x*8u + sgid;
    if (r >= R) { return; }
    device const float* xr = x + (ulong)r*N;
    float s = 0.0;
    for (uint i = lane; i < N; i += 32u) { s += xr[i]*xr[i]; }
    s = simd_sum(s);
    float rr = rsqrt(s/float(N) + TEPS);
    for (uint i = lane; i < N; i += 32u) { y[(ulong)r*N + i] = xr[i]*rr*w[i]; }
    if (lane == 0) { rsave[r] = rr; }
}
kernel void t_rms_bwd(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device const float* rsave [[buffer(2)]], device const float* dy [[buffer(3)]],
    device float* dx [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& R [[buffer(6)]],
    constant uint& accum [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x*8u + sgid;
    if (r >= R) { return; }
    device const float* xr = x + (ulong)r*N;
    device const float* dyr = dy + (ulong)r*N;
    float rr = rsave[r];
    float dot = 0.0;
    for (uint i = lane; i < N; i += 32u) { dot += dyr[i]*w[i]*xr[i]; }
    dot = simd_sum(dot);
    float c = rr*rr*rr*dot/float(N);
    for (uint i = lane; i < N; i += 32u) {
        ulong di = (ulong)r*N + i;
        float v = rr*dyr[i]*w[i] - xr[i]*c;
        dx[di] = (accum != 0 ? dx[di] : 0.0f) + v;
    }
}
kernel void t_rms_dw(device const float* x [[buffer(0)]], device const float* rsave [[buffer(1)]],
    device const float* dy [[buffer(2)]], device float* dw [[buffer(3)]],
    constant uint& N [[buffer(4)]], constant uint& R [[buffer(5)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= N) { return; }
    float acc = 0.0;
    for (uint r = 0; r < R; r++) { acc += dy[(ulong)r*N + g]*x[(ulong)r*N + g]*rsave[r]; }
    dw[g] = acc;
}

// gated per-head RMSNorm: rows R = T*HV of width S; og = o*r*nw*silu(z)
kernel void t_gnorm_fwd(device const float* o [[buffer(0)]], device const float* z [[buffer(1)]],
    device const float* nw [[buffer(2)]], device float* og [[buffer(3)]], device float* rsave [[buffer(4)]],
    constant uint& S [[buffer(5)]], constant uint& R [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x*8u + sgid;
    if (r >= R) { return; }
    float s = 0.0;
    for (uint i = lane; i < S; i += 32u) { float v = o[(ulong)r*S+i]; s += v*v; }
    s = simd_sum(s);
    float rr = rsqrt(s/float(S) + TEPS);
    for (uint i = lane; i < S; i += 32u) {
        og[(ulong)r*S+i] = o[(ulong)r*S+i]*rr*nw[i]*tsilu(z[(ulong)r*S+i]);
    }
    if (lane == 0) { rsave[r] = rr; }
}
kernel void t_gnorm_bwd(device const float* o [[buffer(0)]], device const float* z [[buffer(1)]],
    device const float* nw [[buffer(2)]], device const float* rsave [[buffer(3)]],
    device const float* dog [[buffer(4)]], device float* do_ [[buffer(5)]], device float* dz [[buffer(6)]],
    constant uint& S [[buffer(7)]], constant uint& R [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x*8u + sgid;
    if (r >= R) { return; }
    float rr = rsave[r];
    float dot = 0.0;
    for (uint i = lane; i < S; i += 32u) {
        float don = dog[(ulong)r*S+i]*tsilu(z[(ulong)r*S+i]);
        dot += don*nw[i]*o[(ulong)r*S+i];
    }
    dot = simd_sum(dot);
    float c = rr*rr*rr*dot/float(S);
    for (uint i = lane; i < S; i += 32u) {
        ulong ix = (ulong)r*S+i;
        float don = dog[ix]*tsilu(z[ix]);
        do_[ix] = rr*don*nw[i] - o[ix]*c;
        dz[ix] = dog[ix]*o[ix]*rr*nw[i]*tdsilu(z[ix]);
    }
}
kernel void t_gnorm_dnw(device const float* o [[buffer(0)]], device const float* z [[buffer(1)]],
    device const float* rsave [[buffer(2)]], device const float* dog [[buffer(3)]],
    device float* dnw [[buffer(4)]], constant uint& S [[buffer(5)]], constant uint& R [[buffer(6)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= S) { return; }
    float acc = 0.0;
    for (uint r = 0; r < R; r++) {
        acc += dog[(ulong)r*S+g]*tsilu(z[(ulong)r*S+g])*o[(ulong)r*S+g]*rsave[r];
    }
    dnw[g] = acc;
}

// GDN gates: bet = sig(b_raw); sp = softplus(a_in+dt); gex = exp(sp*acoef)
kernel void t_gates_fwd(device const float* a_in [[buffer(0)]], device const float* b_raw [[buffer(1)]],
    device const float* dt [[buffer(2)]], device const float* alog [[buffer(3)]],
    device float* bet [[buffer(4)]], device float* sp [[buffer(5)]], device float* gex [[buffer(6)]],
    constant uint& HV [[buffer(7)]], constant uint& TOT [[buffer(8)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }
    uint h = g % HV;
    bet[g] = tsig(b_raw[g]);
    float xg = a_in[g] + dt[h];
    float s = xg > 20.0f ? xg : log(1.0f + exp(xg));
    sp[g] = s;
    gex[g] = exp(s*(-exp(alog[h])));
}
// elementwise chains: d_ain = d_gex*gex*acoef*sig(a_in+dt); d_braw = d_bet*bet*(1-bet)
kernel void t_gates_bwd(device const float* d_gex [[buffer(0)]], device const float* d_bet [[buffer(1)]],
    device const float* gex [[buffer(2)]], device const float* bet [[buffer(3)]],
    device const float* a_in [[buffer(4)]], device const float* dt [[buffer(5)]],
    device const float* alog [[buffer(6)]], device float* d_ain [[buffer(7)]], device float* d_braw [[buffer(8)]],
    constant uint& HV [[buffer(9)]], constant uint& TOT [[buffer(10)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }
    uint h = g % HV;
    d_ain[g] = d_gex[g]*gex[g]*(-exp(alog[h]))*tsig(a_in[g] + dt[h]);
    d_braw[g] = d_bet[g]*bet[g]*(1.0f - bet[g]);
}
// per-head reductions over T: d_dt[h] = sum_t d_ain; d_alog[h] = (sum_t d_gex*gex*sp)*(-exp(alog))
kernel void t_gates_dtb(device const float* d_ain [[buffer(0)]], device const float* d_gex [[buffer(1)]],
    device const float* gex [[buffer(2)]], device const float* sp [[buffer(3)]],
    device const float* alog [[buffer(4)]], device float* d_dt [[buffer(5)]], device float* d_alog [[buffer(6)]],
    constant uint& HV [[buffer(7)]], constant uint& T [[buffer(8)]],
    uint h [[thread_position_in_grid]]) {
    if (h >= HV) { return; }
    float sdt = 0.0, sa = 0.0;
    for (uint t = 0; t < T; t++) {
        sdt += d_ain[t*HV + h];
        sa += d_gex[t*HV + h]*gex[t*HV + h]*sp[t*HV + h];
    }
    d_dt[h] = sdt;
    d_alog[h] = sa*(-exp(alog[h]));
}

// causal depthwise conv (K=4) + SiLU, fresh sequence (zero left context)
kernel void t_conv_fwd(device const float* qkv [[buffer(0)]], device const float* cw [[buffer(1)]],
    device float* acc [[buffer(2)]], device float* conv [[buffer(3)]],
    constant uint& C [[buffer(4)]], constant uint& T [[buffer(5)]],
    uint2 g [[thread_position_in_grid]]) {
    uint c = g.x, t = g.y;
    if (c >= C) { return; }
    float s = 0.0;
    for (uint j = 0; j < 4; j++) {
        if (t + j >= 3) { s += cw[c*4u + j]*qkv[(ulong)(t + j - 3)*C + c]; }
    }
    acc[(ulong)t*C + c] = s;
    conv[(ulong)t*C + c] = tsilu(s);
}
kernel void t_conv_bwd(device const float* dconv [[buffer(0)]], device const float* acc [[buffer(1)]],
    device const float* cw [[buffer(2)]], device float* dqkv [[buffer(3)]],
    constant uint& C [[buffer(4)]], constant uint& T [[buffer(5)]],
    uint2 g [[thread_position_in_grid]]) {
    uint c = g.x, t = g.y; // t = source position
    if (c >= C) { return; }
    float s = 0.0;
    for (uint m = 0; m < 4; m++) {
        uint tt = t + m;
        if (tt < T) { s += dconv[(ulong)tt*C + c]*tdsilu(acc[(ulong)tt*C + c])*cw[c*4u + (3u - m)]; }
    }
    dqkv[(ulong)t*C + c] = s;
}
kernel void t_conv_dw(device const float* dconv [[buffer(0)]], device const float* acc [[buffer(1)]],
    device const float* qkv [[buffer(2)]], device float* dcw [[buffer(3)]],
    constant uint& C [[buffer(4)]], constant uint& T [[buffer(5)]],
    uint2 g [[thread_position_in_grid]]) {
    uint c = g.x, j = g.y;
    if (c >= C || j >= 4) { return; }
    float s = 0.0;
    for (uint t = 0; t < T; t++) {
        if (t + j >= 3) { s += dconv[(ulong)t*C + c]*tdsilu(acc[(ulong)t*C + c])*qkv[(ulong)(t + j - 3)*C + c]; }
    }
    dcw[c*4u + j] = s;
}

// GDN forward (training): saves per-token state history + sk/dlt.
// HF layout: v-head h uses k/q-head h/group. Grid (S/4, HV), 128 threads.
kernel void t_dn_fwd(device const float* conv [[buffer(0)]], device const float* gex [[buffer(1)]],
    device const float* bet [[buffer(2)]], device float* o [[buffer(3)]],
    device float* st_hist [[buffer(4)]], device float* sk_all [[buffer(5)]], device float* dlt_all [[buffer(6)]],
    constant uint& S [[buffer(7)]], constant uint& HK [[buffer(8)]], constant uint& HV [[buffer(9)]],
    constant uint& C [[buffer(10)]], constant uint& T [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    uint h = tg.y;
    uint col = tg.x*4u + sgid;
    uint kh = h / (HV/HK);
    float scale = 1.0/sqrt(float(S));
    ulong hss = (ulong)HV*S*S;
    float ls[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint t = 0; t < T; t++) {
        device float* snap = st_hist + (ulong)t*hss + (ulong)(h*S + col)*S;
        for (short j = 0; j < 4; j++) { snap[lane*4u + j] = ls[j]; }
        device const float* row = conv + (ulong)t*C;
        float qv[4], kv[4];
        float sq = 0.0, s2 = 0.0;
        for (short j = 0; j < 4; j++) {
            uint is = lane*4u + j;
            qv[j] = row[kh*S + is];
            kv[j] = row[HK*S + kh*S + is];
            sq += qv[j]*qv[j]; s2 += kv[j]*kv[j];
        }
        sq = simd_sum(sq); s2 = simd_sum(s2);
        float qn = rsqrt(sq + TEPS)*scale;
        float kn = rsqrt(s2 + TEPS);
        float g = gex[t*HV + h];
        float b = bet[t*HV + h];
        float sk = 0.0;
        for (short j = 0; j < 4; j++) { ls[j] *= g; sk += ls[j]*kv[j]; }
        sk = simd_sum(sk);
        float d = (row[2u*HK*S + h*S + col] - sk*kn)*b;
        float y = 0.0;
        for (short j = 0; j < 4; j++) { ls[j] += kv[j]*kn*d; y += ls[j]*qv[j]; }
        y = simd_sum(y);
        if (lane == 0) {
            o[(ulong)t*(HV*S) + h*S + col] = y*qn;
            sk_all[((ulong)t*HV + h)*S + col] = sk;
            dlt_all[((ulong)t*HV + h)*S + col] = d;
        }
    }
    device float* snap = st_hist + (ulong)T*hss + (ulong)(h*S + col)*S;
    for (short j = 0; j < 4; j++) { snap[lane*4u + j] = ls[j]; }
}

// GDN backward: reverse token scan. One threadgroup (128 threads) per head;
// thread tid owns j = tid. dS carried in device buffer. dq/dk staged per-head
// (v-head pairs share a q/k slot — folded by t_dnqk_fold, no atomics).
kernel void t_dn_bwd(device const float* conv [[buffer(0)]], device const float* gex [[buffer(1)]],
    device const float* bet [[buffer(2)]], device const float* d_o [[buffer(3)]],
    device const float* st_hist [[buffer(4)]], device const float* sk_all [[buffer(5)]],
    device const float* dlt_all [[buffer(6)]], device float* dS [[buffer(7)]],
    device float* dqk [[buffer(8)]], device float* dconv [[buffer(9)]],
    device float* d_gex [[buffer(10)]], device float* d_bet [[buffer(11)]],
    constant uint& S [[buffer(12)]], constant uint& HK [[buffer(13)]], constant uint& HV [[buffer(14)]],
    constant uint& C [[buffer(15)]], constant uint& T [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    uint h = tg.y, cg = tg.x, tid = tid2.x;   // cg = column group (4 x 32 cols)
    uint kh = h / (HV/HK);
    float scale = 1.0/sqrt(float(S));
    ulong hss = (ulong)HV*S*S;
    threadgroup float red[4];
    device float* dsb = dS + (ulong)h*S*S;
    #define TG_SUM(val, out) { float p_ = simd_sum(val); if (lane==0) { red[sgid] = p_; } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        out = red[0]+red[1]+red[2]+red[3]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    for (int t = int(T) - 1; t >= 0; t--) {
        device const float* row = conv + (ulong)t*C;
        float qj = row[kh*S + tid];
        float kj = row[HK*S + kh*S + tid];
        float sq, s2;
        TG_SUM(qj*qj, sq); TG_SUM(kj*kj, s2);
        sq += TEPS; s2 += TEPS;
        float rq = rsqrt(sq);
        float qn = rq*scale;
        float kn = rsqrt(s2);
        float gg = gex[t*HV + h];
        float bb = bet[t*HV + h];
        device const float* sprev = st_hist + (ulong)t*hss + (ulong)h*S*S;
        device const float* s2s = st_hist + (ulong)(t + 1)*hss + (ulong)h*S*S;
        float dq = 0.0, dk = 0.0, dqn = 0.0, dkn = 0.0, dg_s = 0.0, db_s = 0.0;
        for (uint col = cg*32u; col < cg*32u + 32u; col++) {
            float do_c = d_o[(ulong)t*(HV*S) + h*S + col];
            float sk_c = sk_all[((ulong)t*HV + h)*S + col];
            float dl_c = dlt_all[((ulong)t*HV + h)*S + col];
            float v_c = row[2u*HK*S + h*S + col];
            float s2r = s2s[col*S + tid];
            float spr = sprev[col*S + tid];
            float dsr = dsb[col*S + tid];
            float yraw;
            TG_SUM(s2r*qj, yraw);
            dqn += do_c*yraw;
            dq += do_c*qn*s2r;
            dsr += do_c*qn*qj;              // dS2 complete
            float dd;
            TG_SUM(dsr*kj, dd);
            float ddlt = dd*kn;
            dk += kn*dsr*dl_c;
            dkn += dd*dl_c;
            if (tid == col) { dconv[(ulong)t*C + 2u*HK*S + h*S + col] = ddlt*bb; }
            // ddlt/do_c/dd/yraw are thread-uniform after TG_SUM, so these
            // scalar accumulators hold the true total in every thread.
            db_s += ddlt*(v_c - sk_c*kn);
            float dsk = -ddlt*bb*kn;
            dkn += -ddlt*bb*sk_c;
            float ds1 = dsr + dsk*kj;
            dk += dsk*spr*gg;
            float dgp;
            TG_SUM(ds1*spr, dgp);
            dg_s += dgp;
            dsb[col*S + tid] = ds1*gg;
        }
        // norm-factor terms depend on the full-head dqn/dkn sums; emit the
        // partial sums and fold them (with the norm terms) in t_dnqk_fold.
        dqk[((((ulong)t*HV + h)*4u + cg)*4u + 0u)*S + tid] = dq;
        dqk[((((ulong)t*HV + h)*4u + cg)*4u + 1u)*S + tid] = dk;
        if (tid == 0) {
            dqk[((((ulong)t*HV + h)*4u + cg)*4u + 2u)*S + 0u] = dqn;
            dqk[((((ulong)t*HV + h)*4u + cg)*4u + 2u)*S + 1u] = dkn;
            d_gex[(t*HV + h)*4u + cg] = dg_s;
            d_bet[(t*HV + h)*4u + cg] = db_s;
        }
    }
    #undef TG_SUM
}

// Fold: sum 4 column-group partials per v-head, apply the deferred
// L2-norm-factor terms (need full-head dqn/dkn), then sum the v-head group
// into the shared q/k slots of dconv. Deterministic fixed-order sums.
kernel void t_dnqk_fold(device const float* dqk [[buffer(0)]], device float* dconv [[buffer(1)]],
    device const float* conv [[buffer(2)]],
    constant uint& S [[buffer(3)]], constant uint& HK [[buffer(4)]], constant uint& HV [[buffer(5)]],
    constant uint& C [[buffer(6)]], constant uint& T [[buffer(7)]],
    uint2 g [[thread_position_in_grid]]) {
    uint j = g.x, x = g.y;         // x = t*HK + khead
    if (j >= S) { return; }
    uint t = x / HK, kh = x % HK;
    uint grp = HV/HK;
    float scale = 1.0/sqrt(float(S));
    device const float* row = conv + (ulong)t*C;
    float qj = row[kh*S + j];
    float kj = row[HK*S + kh*S + j];
    // recompute rq/kn for the norm-factor terms (cheap: S-dot via serial? no —
    // read all S elems per thread would be S ops; instead each thread computes
    // its own j term only, needing sq/s2 sums: recompute via loop over S)
    float sq = 0.0, s2 = 0.0;
    for (uint i = 0; i < S; i++) {
        float q = row[kh*S + i]; float k = row[HK*S + kh*S + i];
        sq += q*q; s2 += k*k;
    }
    sq += 1e-6f; s2 += 1e-6f;
    float rq = rsqrt(sq);
    float kn = rsqrt(s2);
    float dq = 0.0, dk = 0.0;
    for (uint u = 0; u < grp; u++) {
        uint h = kh*grp + u;
        float dqn = 0.0, dkn = 0.0;
        for (uint cgi = 0; cgi < 4; cgi++) {
            ulong base = (((ulong)t*HV + h)*4u + cgi)*4u;
            dq += dqk[(base + 0u)*S + j];
            dk += dqk[(base + 1u)*S + j];
            dqn += dqk[(base + 2u)*S + 0u];
            dkn += dqk[(base + 2u)*S + 1u];
        }
        dq += dqn*scale*(-qj*rq*rq*rq);
        dk += dkn*(-kj*kn*kn*kn);
    }
    dconv[(ulong)t*C + kh*S + j] = dq;
    dconv[(ulong)t*C + HK*S + kh*S + j] = dk;
}

// ======== MMA GEMMs (simdgroup_matrix, f32) — ported from gemm_mm_f16 ========
// y[m,n] = dot(x[m,:K], w[n,:K])  — 64(N)x32(M) tile, 128 threads, 4 simdgroups.
// N%64==0, K%32==0 required (guaranteed by call sites; small-N uses t_gemm_xwT).
kernel void t_mm_xwT(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[64*32];
    threadgroup float sb[32*32];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;
    uint lr0 = tiitg/2u;
    uint il0 = tiitg%2u;
    device const float* arow = w + (ulong)(r0+lr0)*(ulong)K;
    uint lr1 = tiitg/4u;
    uint sxb = tiitg%4u;
    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile: 16 elems of weight row lr0 at k = lk + il0*16 + i
            device const float* bp = arow + lk + il0*16u;
            uint sy = lr0/8u, lx = lr0%8u;
            for (short i = 0; i < 16; i++) {
                uint sx = 2u*il0 + uint(i)/8u;
                sa[64u*(8u*sx+sy) + 8u*(uint(i)%8u) + lx] = bp[i];
            }
        }
        {   // B tile: token lr1, 8 k at lk + 8*sxb (zero-pad tokens >= M)
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            device const float4* xr = (device const float4*)(x + (ulong)(t0+lr1)*(ulong)K + lk + 8u*sxb);
            bool ok = t0 + lr1 < M;
            threadgroup float4* dstb = (threadgroup float4*)(sb + 64u*ib + 8u*ly);
            dstb[0] = ok ? xr[0] : float4(0.0);
            dstb[1] = ok ? xr[1] : float4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const float* lsmb = sb + 2u*64u*(sgitg/2u);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) { simdgroup_load(ma[i], lsma + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) { simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) { simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]); }
            lsma += 8*64; lsmb += 4*64;
        }
    }
    device float* C = y + (r0 + 32u*(sgitg & 1u)) + (ulong)(t0 + 16u*(sgitg >> 1u))*(ulong)N;
    if (t0 + 32u <= M) {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false);
        }
    } else {
        // tail: stage to threadgroup and copy row-wise with bounds check
        threadgroup float* stg = sa; // reuse (64x32 >= 32x64)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], stg + (sgitg%2u)*32u + 8*(i%4) + (ulong)(8*(i/4) + 16u*(sgitg>>1))*64u, 64, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tiitg; idx < 32u*64u; idx += 128u) {
            uint tt = idx/64u, rr = idx%64u;
            if (t0 + tt < M) { y[(ulong)(t0+tt)*(ulong)N + r0 + rr] = stg[tt*64u + rr]; }
        }
    }
}

// f16-weight twin of t_mm_xwT (trainer f16 weight storage), 128 threads, 4 simdgroups.
// N%64==0, K%32==0 required (guaranteed by call sites; small-N uses t_gemm_xwT).
kernel void t_mm_xwT_h(device const float* x [[buffer(0)]], device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[64*32];
    threadgroup float sb[32*32];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;
    uint lr0 = tiitg/2u;
    uint il0 = tiitg%2u;
    device const half* arow = w + (ulong)(r0+lr0)*(ulong)K;
    uint lr1 = tiitg/4u;
    uint sxb = tiitg%4u;
    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile: 16 elems of weight row lr0 at k = lk + il0*16 + i
            device const half* bp = arow + lk + il0*16u;
            uint sy = lr0/8u, lx = lr0%8u;
            for (short i = 0; i < 16; i++) {
                uint sx = 2u*il0 + uint(i)/8u;
                sa[64u*(8u*sx+sy) + 8u*(uint(i)%8u) + lx] = float(bp[i]);
            }
        }
        {   // B tile: token lr1, 8 k at lk + 8*sxb (zero-pad tokens >= M)
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            device const float4* xr = (device const float4*)(x + (ulong)(t0+lr1)*(ulong)K + lk + 8u*sxb);
            bool ok = t0 + lr1 < M;
            threadgroup float4* dstb = (threadgroup float4*)(sb + 64u*ib + 8u*ly);
            dstb[0] = ok ? xr[0] : float4(0.0);
            dstb[1] = ok ? xr[1] : float4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const float* lsmb = sb + 2u*64u*(sgitg/2u);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) { simdgroup_load(ma[i], lsma + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) { simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) { simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]); }
            lsma += 8*64; lsmb += 4*64;
        }
    }
    device float* C = y + (r0 + 32u*(sgitg & 1u)) + (ulong)(t0 + 16u*(sgitg >> 1u))*(ulong)N;
    if (t0 + 32u <= M) {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false);
        }
    } else {
        // tail: stage to threadgroup and copy row-wise with bounds check
        threadgroup float* stg = sa; // reuse (64x32 >= 32x64)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], stg + (sgitg%2u)*32u + 8*(i%4) + (ulong)(8*(i/4) + 16u*(sgitg>>1))*64u, 64, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tiitg; idx < 32u*64u; idx += 128u) {
            uint tt = idx/64u, rr = idx%64u;
            if (t0 + tt < M) { y[(ulong)(t0+tt)*(ulong)N + r0 + rr] = stg[tt*64u + rr]; }
        }
    }
}

// rp-only RMSNorm (1/rms per row, no h2 write) — the reduction half of a fused norm->GEMM.
kernel void t_rmsrp(device const float* x [[buffer(0)]], device float* rsave [[buffer(1)]],
    constant uint& N [[buffer(2)]], constant uint& R [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x*8u + sgid;
    if (r >= R) { return; }
    device const float* xr = x + (ulong)r*N;
    float s = 0.0;
    for (uint i = lane; i < N; i += 32u) { s += xr[i]*xr[i]; }
    s = simd_sum(s);
    if (lane == 0) { rsave[r] = rsqrt(s/float(N) + TEPS); }
}

// FUSED RMSNorm->GEMM: y = (rmsnorm(x) with weight nw, scale rp) @ w^T. Applies the norm during
// the x-tile load — never materializes h2 (hides ~90% of norm memory-IO). Mirrors t_mm_xwT_h.
kernel void t_mm_rmsnorm_xwT(device const float* x [[buffer(0)]], device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]], device const float* nw [[buffer(3)]], device const float* rp [[buffer(4)]],
    constant uint& K [[buffer(5)]], constant uint& N [[buffer(6)]], constant uint& M [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[64*32];
    threadgroup float sb[32*32];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;
    uint lr0 = tiitg/2u, il0 = tiitg%2u;
    device const half* arow = w + (ulong)(r0+lr0)*(ulong)K;
    uint lr1 = tiitg/4u, sxb = tiitg%4u;
    simdgroup_float8x8 ma[4]; simdgroup_float8x8 mb[2]; simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        { device const half* bp = arow + lk + il0*16u; uint sy = lr0/8u, lx = lr0%8u;
          for (short i = 0; i < 16; i++) { uint sx = 2u*il0 + uint(i)/8u; sa[64u*(8u*sx+sy) + 8u*(uint(i)%8u) + lx] = float(bp[i]); } }
        { uint sy = lr1/8u, ly = lr1%8u; uint ib = 4u*sxb + sy; uint token = t0 + lr1;
          bool ok = token < M; float rrow = ok ? rp[token] : 0.0f;
          device const float4* xr = (device const float4*)(x + (ulong)token*(ulong)K + lk + 8u*sxb);
          device const float4* nwr = (device const float4*)(nw + lk + 8u*sxb);
          threadgroup float4* dstb = (threadgroup float4*)(sb + 64u*ib + 8u*ly);
          float4 x0 = ok ? xr[0] : float4(0.0); float4 x1 = ok ? xr[1] : float4(0.0);
          dstb[0] = x0 * rrow * nwr[0]; dstb[1] = x1 * rrow * nwr[1]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const float* lsmb = sb + 2u*64u*(sgitg/2u);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) { simdgroup_load(ma[i], lsma + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) { simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) { simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]); }
            lsma += 8*64; lsmb += 4*64;
        }
    }
    device float* C = y + (r0 + 32u*(sgitg & 1u)) + (ulong)(t0 + 16u*(sgitg >> 1u))*(ulong)N;
    if (t0 + 32u <= M) {
        for (short i = 0; i < 8; i++) { simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false); }
    } else {
        threadgroup float* stg = sa;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 8; i++) { simdgroup_store(mc[i], stg + (sgitg%2u)*32u + 8*(i%4) + (ulong)(8*(i/4) + 16u*(sgitg>>1))*64u, 64, 0, false); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tiitg; idx < 32u*64u; idx += 128u) { uint tt = idx/64u, rr = idx%64u; if (t0+tt < M) { y[(ulong)(t0+tt)*(ulong)N + r0 + rr] = stg[tt*64u + rr]; } }
    }
}

// GROUPED MoE GEMM (Megablocks dMoE): one launch does ALL experts. m-tile `tgpig.x` covers
// gathered rows [trow0..tmend) using expert texp's weight from wall[ne,N,K]. Mirrors t_mm_xwT_h.
kernel void t_mm_grp_xwT(device const float* x [[buffer(0)]], device const half* wall [[buffer(1)]],
    device float* y [[buffer(2)]], device const uint* texp [[buffer(3)]],
    device const uint* trow0 [[buffer(4)]], device const uint* tmend [[buffer(5)]],
    constant uint& K [[buffer(6)]], constant uint& N [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[64*32];
    threadgroup float sb[32*32];
    uint mtile = tgpig.x;
    uint expert = texp[mtile];
    const uint r0 = tgpig.y*64u;
    const uint t0 = trow0[mtile];
    const uint M = tmend[mtile];
    device const half* w = wall + (ulong)expert*(ulong)N*(ulong)K;
    uint lr0 = tiitg/2u, il0 = tiitg%2u;
    device const half* arow = w + (ulong)(r0+lr0)*(ulong)K;
    uint lr1 = tiitg/4u, sxb = tiitg%4u;
    simdgroup_float8x8 ma[4]; simdgroup_float8x8 mb[2]; simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        { device const half* bp = arow + lk + il0*16u; uint sy = lr0/8u, lx = lr0%8u;
          for (short i = 0; i < 16; i++) { uint sx = 2u*il0 + uint(i)/8u; sa[64u*(8u*sx+sy) + 8u*(uint(i)%8u) + lx] = float(bp[i]); } }
        { uint sy = lr1/8u, ly = lr1%8u; uint ib = 4u*sxb + sy;
          device const float4* xr = (device const float4*)(x + (ulong)(t0+lr1)*(ulong)K + lk + 8u*sxb);
          bool ok = t0 + lr1 < M;
          threadgroup float4* dstb = (threadgroup float4*)(sb + 64u*ib + 8u*ly);
          dstb[0] = ok ? xr[0] : float4(0.0); dstb[1] = ok ? xr[1] : float4(0.0); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const float* lsmb = sb + 2u*64u*(sgitg/2u);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) { simdgroup_load(ma[i], lsma + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) { simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) { simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]); }
            lsma += 8*64; lsmb += 4*64;
        }
    }
    device float* C = y + (r0 + 32u*(sgitg & 1u)) + (ulong)(t0 + 16u*(sgitg >> 1u))*(ulong)N;
    if (t0 + 32u <= M) {
        for (short i = 0; i < 8; i++) { simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false); }
    } else {
        threadgroup float* stg = sa;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 8; i++) { simdgroup_store(mc[i], stg + (sgitg%2u)*32u + 8*(i%4) + (ulong)(8*(i/4) + 16u*(sgitg>>1))*64u, 64, 0, false); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tiitg; idx < 32u*64u; idx += 128u) { uint tt = idx/64u, rr = idx%64u; if (t0+tt < M) { y[(ulong)(t0+tt)*(ulong)N + r0 + rr] = stg[tt*64u + rr]; } }
    }
}

// dx[m,i] = sum_o dy[m,o]*w[o,i] — same skeleton, A tile staged transposed
// from w (A'[i,o] = w[o,i]); coalesced: 32 w-rows x 64 contiguous cols.
// IN%64==0, OUT%32==0.
kernel void t_mm_dx(device const float* dy [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* dx [[buffer(2)]], constant uint& IN [[buffer(3)]], constant uint& OUT [[buffer(4)]],
    constant uint& accum [[buffer(5)]], constant uint& M [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[64*32];
    threadgroup float sb[32*32];
    const uint r0 = tgpig.y*64u;   // i block (IN)
    const uint t0 = tgpig.x*32u;   // m block
    uint lr1 = tiitg/4u;
    uint sxb = tiitg%4u;
    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < OUT; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile transposed: each thread reads 16 contiguous i of w row (lk+lo)
            uint lo = tiitg/8u;            // 16 o-rows per pass, 2 passes
            uint seg = tiitg%8u;           // 8 segments x 8 i
            for (uint pass = 0; pass < 2; pass++) {
                uint o = lo + 16u*pass;
                device const float* wr = w + (ulong)(lk+o)*(ulong)IN + r0 + 8u*seg;
                uint kk = o;               // A' k-index = o
                for (short i = 0; i < 8; i++) {
                    uint ii = 8u*seg + uint(i);   // i within 64
                    uint sx = kk/8u, sy = ii/8u, lx = ii%8u;
                    sa[64u*(8u*sx+sy) + 8u*(kk%8u) + lx] = wr[i];
                }
            }
        }
        {   // B tile: token lr1, 8 o at lk + 8*sxb
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            device const float4* xr = (device const float4*)(dy + (ulong)(t0+lr1)*(ulong)OUT + lk + 8u*sxb);
            bool ok = t0 + lr1 < M;
            threadgroup float4* dstb = (threadgroup float4*)(sb + 64u*ib + 8u*ly);
            dstb[0] = ok ? xr[0] : float4(0.0);
            dstb[1] = ok ? xr[1] : float4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const float* lsmb = sb + 2u*64u*(sgitg/2u);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) { simdgroup_load(ma[i], lsma + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) { simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) { simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]); }
            lsma += 8*64; lsmb += 4*64;
        }
    }
    threadgroup float* stg = sa;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], stg + (sgitg%2u)*32u + 8*(i%4) + (ulong)(8*(i/4) + 16u*(sgitg>>1))*64u, 64, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tiitg; idx < 32u*64u; idx += 128u) {
        uint tt = idx/64u, rr = idx%64u;
        if (t0 + tt < M) {
            ulong di = (ulong)(t0+tt)*(ulong)IN + r0 + rr;
            dx[di] = (accum != 0 ? dx[di] : 0.0f) + stg[tt*64u + rr];
        }
    }
}

// f16-weight twin of t_mm_dx (FLCE dh: dh = dlogits @ lm_head)
// from w (A'[i,o] = w[o,i]); coalesced: 32 w-rows x 64 contiguous cols.
// IN%64==0, OUT%32==0.
kernel void t_mm_dx_h(device const float* dy [[buffer(0)]], device const half* w [[buffer(1)]],
    device float* dx [[buffer(2)]], constant uint& IN [[buffer(3)]], constant uint& OUT [[buffer(4)]],
    constant uint& accum [[buffer(5)]], constant uint& M [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[64*32];
    threadgroup float sb[32*32];
    const uint r0 = tgpig.y*64u;   // i block (IN)
    const uint t0 = tgpig.x*32u;   // m block
    uint lr1 = tiitg/4u;
    uint sxb = tiitg%4u;
    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < OUT; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile transposed: each thread reads 16 contiguous i of w row (lk+lo)
            uint lo = tiitg/8u;            // 16 o-rows per pass, 2 passes
            uint seg = tiitg%8u;           // 8 segments x 8 i
            for (uint pass = 0; pass < 2; pass++) {
                uint o = lo + 16u*pass;
                device const half* wr = w + (ulong)(lk+o)*(ulong)IN + r0 + 8u*seg;
                uint kk = o;               // A' k-index = o
                for (short i = 0; i < 8; i++) {
                    uint ii = 8u*seg + uint(i);   // i within 64
                    uint sx = kk/8u, sy = ii/8u, lx = ii%8u;
                    sa[64u*(8u*sx+sy) + 8u*(kk%8u) + lx] = float(wr[i]);
                }
            }
        }
        {   // B tile: token lr1, 8 o at lk + 8*sxb
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            device const float4* xr = (device const float4*)(dy + (ulong)(t0+lr1)*(ulong)OUT + lk + 8u*sxb);
            bool ok = t0 + lr1 < M;
            threadgroup float4* dstb = (threadgroup float4*)(sb + 64u*ib + 8u*ly);
            dstb[0] = ok ? xr[0] : float4(0.0);
            dstb[1] = ok ? xr[1] : float4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const float* lsmb = sb + 2u*64u*(sgitg/2u);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) { simdgroup_load(ma[i], lsma + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) { simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) { simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]); }
            lsma += 8*64; lsmb += 4*64;
        }
    }
    threadgroup float* stg = sa;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], stg + (sgitg%2u)*32u + 8*(i%4) + (ulong)(8*(i/4) + 16u*(sgitg>>1))*64u, 64, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tiitg; idx < 32u*64u; idx += 128u) {
        uint tt = idx/64u, rr = idx%64u;
        if (t0 + tt < M) {
            ulong di = (ulong)(t0+tt)*(ulong)IN + r0 + rr;
            dx[di] = (accum != 0 ? dx[di] : 0.0f) + stg[tt*64u + rr];
        }
    }
}

// dw[o,i] = sum_m dy[m,o]*x[m,i] — MMA outer-product GEMM (reduction dim = M
// tokens, staged in 32-slabs). A tile = dy^T [o(64), m(32)]; B tile = x^T
// [i(32), m(32)]; mc = [i][o]; stored transposed into dw[o, i]. OUT%64==0,
// IN%32==0; M padded in staging.
kernel void t_mm_dw(device const float* dy [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* dw [[buffer(2)]], constant uint& IN [[buffer(3)]], constant uint& OUT [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[64*32];
    threadgroup float sb[32*32];
    const uint r0 = tgpig.y*64u;   // o block (OUT)
    const uint t0 = tgpig.x*32u;   // i block (IN)
    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < M; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile transposed from dy: 32 m-rows x 64 contiguous o
            uint lm = tiitg/8u;            // 16 m-rows per pass, 2 passes
            uint seg = tiitg%8u;
            for (uint pass = 0; pass < 2; pass++) {
                uint m = lm + 16u*pass;
                bool ok = lk + m < M;
                device const float* dyr = dy + (ulong)(lk+m)*(ulong)OUT + r0 + 8u*seg;
                for (short i = 0; i < 8; i++) {
                    uint oo = 8u*seg + uint(i);
                    uint sx = m/8u, sy = oo/8u, lx = oo%8u;
                    sa[64u*(8u*sx+sy) + 8u*(m%8u) + lx] = ok ? dyr[i] : 0.0f;
                }
            }
        }
        {   // B tile transposed from x: 32 m-rows x 32 contiguous i
            uint lm = tiitg/4u;            // 32 m-rows, 4 threads each covering 8 i
            uint seg = tiitg%4u;
            bool ok = lk + lm < M;
            device const float* xr = x + (ulong)(lk+lm)*(ulong)IN + t0 + 8u*seg;
            for (short i = 0; i < 8; i++) {
                uint ii = 8u*seg + uint(i);
                uint sy2 = ii/8u, ly = ii%8u;
                uint ib = 4u*(lm/8u) + sy2;
                sb[64u*ib + 8u*ly + (lm%8u)] = ok ? xr[i] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const float* lsmb = sb + 2u*64u*(sgitg/2u);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) { simdgroup_load(ma[i], lsma + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) { simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) { simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]); }
            lsma += 8*64; lsmb += 4*64;
        }
    }
    // mc blocks are [i][o]; store transposed into dw[o, i] (ld = IN)
    for (short i = 0; i < 8; i++) {
        uint oo = r0 + 32u*(sgitg & 1u) + 8u*(i%4);
        uint ii = t0 + 16u*(sgitg >> 1u) + 8u*(i/4);
        simdgroup_store(mc[i], dw + (ulong)oo*(ulong)IN + ii, IN, 0, true);
    }
}

// sum the 4 column-group partials of d_gex/d_bet
kernel void t_dng_fold(device const float* gp [[buffer(0)]], device const float* bp [[buffer(1)]],
    device float* g_out [[buffer(2)]], device float* b_out [[buffer(3)]],
    constant uint& TOT [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= TOT) { return; }
    g_out[i] = gp[i*4u] + gp[i*4u+1u] + gp[i*4u+2u] + gp[i*4u+3u];
    b_out[i] = bp[i*4u] + bp[i*4u+1u] + bp[i*4u+2u] + bp[i*4u+3u];
}

// SwiGLU elementwise fwd/bwd + add
kernel void t_swiglu_fwd(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]],
    device float* act [[buffer(2)]], constant uint& N [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < N) { act[i] = tsilu(g[i])*u[i]; }
}
kernel void t_swiglu_bwd(device const float* dact [[buffer(0)]], device const float* g [[buffer(1)]],
    device const float* u [[buffer(2)]], device float* dg [[buffer(3)]], device float* du [[buffer(4)]],
    constant uint& N [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i < N) { dg[i] = dact[i]*u[i]*tdsilu(g[i]); du[i] = dact[i]*tsilu(g[i]); }
}
kernel void t_add(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
    constant uint& N [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < N) { dst[i] += src[i]; }
}
kernel void t_fill(device float* dst [[buffer(0)]],
    constant uint& N [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < N) { dst[i] = 0.0f; }
}
kernel void t_copy(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
    constant uint& N [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < N) { dst[i] = src[i]; }
}

// ======== attention decoder layer (fused [q|gate], QK-norm, partial rope, GQA) ========

// q0[t,h,:HD] = qfull[t,h*2HD:+HD]  (dir=0) / merge back (dir=1)
kernel void t_qsplit(device float* qfull [[buffer(0)]], device float* q0 [[buffer(1)]],
    constant uint& HD [[buffer(2)]], constant uint& NH [[buffer(3)]], constant uint& dir [[buffer(4)]],
    constant uint& TOT [[buffer(5)]], uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }               // TOT = T*NH*HD
    uint i = g % HD, h = (g / HD) % NH, t = g / (HD*NH);
    ulong fi = (ulong)t*(2u*NH*HD) + h*2u*HD + i;
    if (dir == 0) { q0[g] = qfull[fi]; } else { qfull[fi] = q0[g]; }
}

// partial NEOX rope in place: rows = T*NH of width HD, rotate first ROT dims.
// bwd=1 applies the transpose (inverse rotation).
kernel void t_rope(device float* x [[buffer(0)]], constant uint& HD [[buffer(1)]],
    constant uint& NH [[buffer(2)]], constant uint& ROT [[buffer(3)]],
    constant uint& bwd [[buffer(4)]], constant uint& TOT [[buffer(5)]],
    constant uint& block [[buffer(6)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }               // TOT = T*NH*(ROT/2)
    uint half_ = ROT/2u;
    uint i = g % half_, h = (g / half_) % NH, pos = g / (half_*NH);
    float fr = pow(1e6f, -2.0f*float(i)/float(ROT));  // Qwen3-0.6B rope_theta (an earlier model used 1e7)
    float ang = float(pos % block)*fr;                // ANGLE resets per packed block...
    float cs = cos(ang), sn = sin(ang);
    ulong base = (ulong)(pos*NH + h)*HD;              // ...but buffer INDEX uses the actual position
    float a = x[base + i], b = x[base + i + half_];
    if (bwd == 0) { x[base + i] = a*cs - b*sn; x[base + i + half_] = a*sn + b*cs; }
    else          { x[base + i] = a*cs + b*sn; x[base + i + half_] = -a*sn + b*cs; }
}

// causal GQA attention forward, saving probs. One thread per (h, ti).
kernel void t_attn_fwd(device const float* q2 [[buffer(0)]], device const float* k2 [[buffer(1)]],
    device const float* v [[buffer(2)]], device float* p [[buffer(3)]], device float* aog [[buffer(4)]],
    constant uint& NH [[buffer(5)]], constant uint& NKV [[buffer(6)]], constant uint& HD [[buffer(7)]],
    constant uint& T [[buffer(8)]], constant uint& causal [[buffer(9)]], constant uint& block [[buffer(10)]], uint2 g [[thread_position_in_grid]]) {
    uint h = g.x, ti = g.y;
    if (h >= NH || ti >= T) { return; }
    uint bs = (ti/block)*block; uint lo = bs; uint hi = causal ? (ti + 1u) : (bs + block);
    uint kvh = h / (NH/NKV);
    float scale = rsqrt(float(HD));
    device const float* qr = q2 + (ulong)(ti*NH + h)*HD;
    device float* pr = p + ((ulong)h*T + ti)*T;
    float mx = -INFINITY;
    for (uint si = lo; si < hi; si++) {
        device const float* kr = k2 + (ulong)(si*NKV + kvh)*HD;
        float s = 0.0;
        for (uint i = 0; i < HD; i++) { s += qr[i]*kr[i]; }
        s *= scale;
        pr[si] = s;
        mx = max(mx, s);
    }
    float den = 0.0;
    for (uint si = lo; si < hi; si++) { pr[si] = exp(pr[si] - mx); den += pr[si]; }
    for (uint si = lo; si < hi; si++) { pr[si] /= den; }
    for (uint i = 0; i < HD; i++) {
        float acc = 0.0;
        for (uint si = lo; si < hi; si++) { acc += pr[si]*v[(ulong)(si*NKV + kvh)*HD + i]; }
        aog[(ulong)(ti*NH + h)*HD + i] = acc;
    }
}

// ao = aog * sigmoid(gate half of qfull); bwd fills daog + dqfull gate half.
kernel void t_attngate_fwd(device const float* aog [[buffer(0)]], device const float* qfull [[buffer(1)]],
    device float* ao [[buffer(2)]], constant uint& HD [[buffer(3)]], constant uint& NH [[buffer(4)]],
    constant uint& TOT [[buffer(5)]], uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }
    uint i = g % HD, h = (g / HD) % NH, t = g / (HD*NH);
    float zz = qfull[(ulong)t*(2u*NH*HD) + h*2u*HD + HD + i];
    ao[g] = aog[g]*tsig(zz);
}
kernel void t_attngate_bwd(device const float* dao [[buffer(0)]], device const float* aog [[buffer(1)]],
    device float* qfull [[buffer(2)]], device float* daog [[buffer(3)]], device float* dqfull [[buffer(4)]],
    constant uint& HD [[buffer(5)]], constant uint& NH [[buffer(6)]], constant uint& TOT [[buffer(7)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }
    uint i = g % HD, h = (g / HD) % NH, t = g / (HD*NH);
    ulong zi = (ulong)t*(2u*NH*HD) + h*2u*HD + HD + i;
    float sg = tsig(qfull[zi]);
    daog[g] = dao[g]*sg;
    dqfull[zi] = dao[g]*aog[g]*sg*(1.0f - sg);
}

// backward stage 1: dscores (softmax bwd) + dq2. One thread per (h, ti).
kernel void t_attn_dscore(device const float* q2 [[buffer(0)]], device const float* k2 [[buffer(1)]],
    device const float* v [[buffer(2)]], device const float* p [[buffer(3)]],
    device const float* daog [[buffer(4)]], device float* ds [[buffer(5)]], device float* dq2 [[buffer(6)]],
    constant uint& NH [[buffer(7)]], constant uint& NKV [[buffer(8)]], constant uint& HD [[buffer(9)]],
    constant uint& T [[buffer(10)]], constant uint& causal [[buffer(11)]], constant uint& block [[buffer(12)]], uint2 g [[thread_position_in_grid]]) {
    uint h = g.x, ti = g.y;
    if (h >= NH || ti >= T) { return; }
    uint bs = (ti/block)*block; uint lo = bs; uint hi = causal ? (ti + 1u) : (bs + block);
    uint kvh = h / (NH/NKV);
    float scale = rsqrt(float(HD));
    device const float* dar = daog + (ulong)(ti*NH + h)*HD;
    device const float* pr = p + ((ulong)h*T + ti)*T;
    device float* dsr = ds + ((ulong)h*T + ti)*T;
    float rowdot = 0.0;
    for (uint si = lo; si < hi; si++) {
        device const float* vr = v + (ulong)(si*NKV + kvh)*HD;
        float dp = 0.0;
        for (uint i = 0; i < HD; i++) { dp += dar[i]*vr[i]; }
        dsr[si] = dp;
        rowdot += pr[si]*dp;
    }
    for (uint si = lo; si < hi; si++) { dsr[si] = pr[si]*(dsr[si] - rowdot)*scale; }
    for (uint i = 0; i < HD; i++) {
        float acc = 0.0;
        for (uint si = lo; si < hi; si++) { acc += dsr[si]*k2[(ulong)(si*NKV + kvh)*HD + i]; }
        dq2[(ulong)(ti*NH + h)*HD + i] = acc;
    }
}

// backward stage 2: dk2/dv — one threadgroup per (si, kvh), threads over i.
kernel void t_attn_dkv(device const float* q2 [[buffer(0)]], device const float* p [[buffer(1)]],
    device const float* ds [[buffer(2)]], device const float* daog [[buffer(3)]],
    device float* dk2 [[buffer(4)]], device float* dv [[buffer(5)]],
    constant uint& NH [[buffer(6)]], constant uint& NKV [[buffer(7)]], constant uint& HD [[buffer(8)]],
    constant uint& T [[buffer(9)]], constant uint& causal [[buffer(10)]], constant uint& block [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]], uint2 nth2 [[threads_per_threadgroup]]) {
    uint si = tg.x, kvh = tg.y;
    uint bs = (si/block)*block; uint be = bs + block; uint lo = causal ? si : bs;
    uint tid = tid2.x, nth = nth2.x;
    uint grp = NH/NKV;
    for (uint i = tid; i < HD; i += nth) {
        float dk = 0.0, dvv = 0.0;
        for (uint u = 0; u < grp; u++) {
            uint h = kvh*grp + u;
            for (uint ti = lo; ti < be; ti++) {
                dk += ds[((ulong)h*T + ti)*T + si]*q2[(ulong)(ti*NH + h)*HD + i];
                dvv += p[((ulong)h*T + ti)*T + si]*daog[(ulong)(ti*NH + h)*HD + i];
            }
        }
        dk2[(ulong)(si*NKV + kvh)*HD + i] = dk;
        dv[(ulong)(si*NKV + kvh)*HD + i] = dvv;
    }
}

// ======== MoE routing: top-k softmax gate (fwd) + its jacobian (bwd) ========
// One thread per token. NE experts, K active. K,NE small (<=64/<=8).
kernel void t_moe_gate_fwd(device const float* logits [[buffer(0)]],
    device float* gates [[buffer(1)]], device uint* topk [[buffer(2)]],
    constant uint& T [[buffer(3)]], constant uint& NE [[buffer(4)]], constant uint& K [[buffer(5)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= T) { return; }
    device const float* lg = logits + (ulong)t*NE;
    uint idx[8]; float val[8];
    for (uint j = 0; j < K; j++) {
        float best = -1e30f; uint bi = 0;
        for (uint e = 0; e < NE; e++) {
            bool taken = false;
            for (uint m = 0; m < j; m++) { if (idx[m] == e) { taken = true; } }
            if (!taken && lg[e] > best) { best = lg[e]; bi = e; }
        }
        idx[j] = bi; val[j] = best;
    }
    float mx = val[0];
    for (uint j = 1; j < K; j++) { mx = max(mx, val[j]); }
    float sum = 0.0f;
    for (uint j = 0; j < K; j++) { val[j] = exp(val[j] - mx); sum += val[j]; }
    for (uint e = 0; e < NE; e++) { gates[(ulong)t*NE + e] = 0.0f; }
    for (uint j = 0; j < K; j++) {
        gates[(ulong)t*NE + idx[j]] = val[j] / sum;
        topk[(ulong)t*K + j] = idx[j];
    }
}

// d_logits from d_gate via the softmax-over-top-k jacobian (0 for unselected experts).
kernel void t_moe_gate_bwd(device const float* dgate [[buffer(0)]],
    device const float* gates [[buffer(1)]], device const uint* topk [[buffer(2)]],
    device float* dlogits [[buffer(3)]],
    constant uint& T [[buffer(4)]], constant uint& NE [[buffer(5)]], constant uint& K [[buffer(6)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= T) { return; }
    for (uint e = 0; e < NE; e++) { dlogits[(ulong)t*NE + e] = 0.0f; }
    float s[8], dg[8]; float dot = 0.0f;
    for (uint j = 0; j < K; j++) {
        uint e = topk[(ulong)t*K + j];
        s[j] = gates[(ulong)t*NE + e]; dg[j] = dgate[(ulong)t*NE + e];
        dot += s[j]*dg[j];
    }
    for (uint j = 0; j < K; j++) {
        uint e = topk[(ulong)t*K + j];
        dlogits[(ulong)t*NE + e] = s[j]*(dg[j] - dot);
    }
}

// out[t,:] += gates[t,EIDX] * src[t,:]   (gated accumulate of an expert's output)
kernel void t_moe_acc(device float* out [[buffer(0)]], device const float* src [[buffer(1)]],
    device const float* gates [[buffer(2)]], constant uint& D [[buffer(3)]], constant uint& NE [[buffer(4)]],
    constant uint& EIDX [[buffer(5)]], constant uint& TOT [[buffer(6)]], uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }
    uint t = g / D;
    out[g] += gates[(ulong)t*NE + EIDX] * src[g];
}

// dst[t,:] = gates[t,EIDX] * src[t,:]   (scale an expert's incoming grad by its gate)
kernel void t_moe_rowscale(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
    device const float* gates [[buffer(2)]], constant uint& D [[buffer(3)]], constant uint& NE [[buffer(4)]],
    constant uint& EIDX [[buffer(5)]], constant uint& TOT [[buffer(6)]], uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }
    uint t = g / D;
    dst[g] = gates[(ulong)t*NE + EIDX] * src[g];
}

// gathered[r,:] = h2[gidx[r],:]   (route tokens into per-expert contiguous groups)
kernel void t_moe_gather(device float* gathered [[buffer(0)]], device const float* h2 [[buffer(1)]],
    device const uint* gidx [[buffer(2)]], constant uint& D [[buffer(3)]], constant uint& TOT [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }
    uint r = g / D, i = g % D;
    gathered[g] = h2[(ulong)gidx[r]*D + i];
}

// out[t,:] = shared[t,:] + sum_j grow_gate[row]*eo[row,:]   where row = tok2rows[t,j]
// (scatter each token's k expert outputs back, gate-weighted; no atomics — one write/token)
kernel void t_moe_scatter_k(device float* out [[buffer(0)]], device const float* shared [[buffer(1)]],
    device const float* eo [[buffer(2)]], device const uint* tok2rows [[buffer(3)]], device const float* grow_gate [[buffer(4)]],
    constant uint& D [[buffer(5)]], constant uint& K [[buffer(6)]], constant uint& TOT [[buffer(7)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }
    uint t = g / D, i = g % D;
    float s = shared[g];
    for (uint j = 0; j < K; j++) {
        uint r = tok2rows[(ulong)t*K + j];
        s += grow_gate[r] * eo[(ulong)r*D + i];
    }
    out[g] = s;
}

// --- GPU-side MoE routing (removes the CPU roundtrip in gather/scatter) ---
// 1) histogram: count[e] = #assignments to expert e
kernel void t_moe_route_count(device atomic_uint* count [[buffer(0)]], device const uint* topk [[buffer(1)]],
    constant uint& NA [[buffer(2)]], uint a [[thread_position_in_grid]]) {
    if (a >= NA) { return; }
    atomic_fetch_add_explicit(&count[topk[a]], 1u, memory_order_relaxed);
}
// 2) exclusive prefix sum -> offset[ne+1] (ne small, single thread)
kernel void t_moe_route_offset(device const uint* count [[buffer(0)]], device uint* offset [[buffer(1)]],
    constant uint& NE [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t != 0) { return; }
    uint acc = 0u;
    for (uint e = 0; e < NE; e++) { offset[e] = acc; acc += count[e]; }
    offset[NE] = acc;
}
// 3) scatter each assignment into its expert's segment (atomic fill), build gather/scatter maps
kernel void t_moe_route_scatter(device const uint* topk [[buffer(0)]], device const float* gates [[buffer(1)]],
    device const uint* offset [[buffer(2)]], device atomic_uint* fill [[buffer(3)]],
    device uint* gidx [[buffer(4)]], device uint* tok2 [[buffer(5)]], device float* gg [[buffer(6)]], device uint* rexp [[buffer(7)]],
    constant uint& NE [[buffer(8)]], constant uint& K [[buffer(9)]], constant uint& T [[buffer(10)]],
    uint a [[thread_position_in_grid]]) {
    if (a >= T*K) { return; }
    uint t = a / K;
    uint e = topk[a];
    uint pos = offset[e] + atomic_fetch_add_explicit(&fill[e], 1u, memory_order_relaxed);
    gidx[pos] = t; tok2[a] = pos; gg[pos] = gates[(ulong)t*NE + e]; rexp[pos] = e;
}

// Build the m-tile map for t_mm_grp_xwT from GPU routing offsets: for each expert
// ex with gathered rows [off[ex], off[ex+1]) emit 32-row tiles (texp, trow0, tmend).
// Unused tile slots get tmend=0 (row guards make them inert). Single thread —
// NE and tile counts are tiny. Enables a fixed worst-case dispatch: no readback.
kernel void t_moe_tilemap(device const uint* off [[buffer(0)]],
    device uint* texp [[buffer(1)]], device uint* trow0 [[buffer(2)]], device uint* tmend [[buffer(3)]],
    constant uint& NE [[buffer(4)]], constant uint& NMT [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) { return; }
    uint s = 0u;
    for (uint ex = 0u; ex < NE; ex++) {
        uint a = off[ex], b = off[ex+1u];
        for (uint t0 = a; t0 < b && s < NMT; t0 += 32u) {
            texp[s] = ex; trow0[s] = t0; tmend[s] = b; s++;
        }
    }
    for (; s < NMT; s++) { texp[s] = 0u; trow0[s] = 0u; tmend[s] = 0u; }
}

// --- Flash Attention forward (online softmax, no T x T materialization) ---
// One thread per (head, query). Streams KV, keeping running max m, sum l, output acc.
// Layout [T, H, HD]. Bidirectional (diffusion). HD <= 128.
kernel void t_flash_attn_fwd(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device float* O [[buffer(3)]],
    constant uint& T [[buffer(4)]], constant uint& HD [[buffer(5)]], constant uint& H [[buffer(6)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= H*T) { return; }
    uint head = g % H, qi = g / H;
    float scale = 1.0f/sqrt(float(HD));
    device const float* q = Q + ((ulong)qi*H + head)*HD;
    float m = -1e30f, l = 0.0f;
    float acc[128];
    for (uint i = 0; i < HD; i++) { acc[i] = 0.0f; }
    for (uint kj = 0; kj < T; kj++) {
        device const float* kk = K + ((ulong)kj*H + head)*HD;
        float s = 0.0f; for (uint i = 0; i < HD; i++) { s += q[i]*kk[i]; }
        s *= scale;
        float mnew = max(m, s);
        float corr = exp(m - mnew);
        float p = exp(s - mnew);
        l = corr*l + p;
        device const float* vv = V + ((ulong)kj*H + head)*HD;
        for (uint i = 0; i < HD; i++) { acc[i] = corr*acc[i] + p*vv[i]; }
        m = mnew;
    }
    device float* o = O + ((ulong)qi*H + head)*HD;
    for (uint i = 0; i < HD; i++) { o[i] = acc[i]/l; }
}

// --- Flash Attention forward, MMA (simdgroup_matrix), online softmax ---
// One threadgroup (1 simdgroup, 32 threads) per 8-query block of one head. Streams K/V in
// 8-key blocks from device; QK^T and P@V use simdgroup_matrix; softmax kept in threadgroup.
// grid = (T/8, H). Layout [T,H,HD]. HD multiple of 8, <= 128. Bidirectional.
kernel void t_flash_mma_fwd(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device float* O [[buffer(3)]], device float* L [[buffer(4)]],
    constant uint& T [[buffer(5)]], constant uint& HD [[buffer(6)]], constant uint& H [[buffer(7)]],
    constant uint& W [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]]) {
    uint qi0 = tgpig.x*8u, head = tgpig.y;
    uint rowstride = H*HD;
    threadgroup float sS[64];
    threadgroup float sPV[8*128];
    threadgroup float Oacc[8*128];
    threadgroup float mm[8], ll[8], cc[8];
    for (uint idx = tiitg; idx < 8u*HD; idx += 32u) { Oacc[idx] = 0.0f; }
    if (tiitg < 8u) { mm[tiitg] = -1e30f; ll[tiitg] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = 1.0f/sqrt(float(HD));
    for (uint kb = 0; kb < T; kb += 8u) {
        if (W != 0u && (qi0 > kb ? qi0 - kb : kb - qi0) > W) { continue; }   // local window
        // S = Q @ K^T  [8,8]
        simdgroup_float8x8 sacc = make_filled_simdgroup_matrix<float, 8>(0.f);
        for (uint h = 0; h < HD; h += 8u) {
            simdgroup_float8x8 qt, kt;
            simdgroup_load(qt, Q + (ulong)(qi0*H + head)*HD + h, rowstride, 0, false);
            simdgroup_load(kt, K + (ulong)(kb*H + head)*HD + h, rowstride, 0, true);   // K^T
            simdgroup_multiply_accumulate(sacc, qt, kt, sacc);
        }
        simdgroup_store(sacc, sS, 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax on the 8x8 tile (lanes 0..7 = rows)
        if (tiitg < 8u) {
            uint r = tiitg;
            float rm = -1e30f;
            for (uint j = 0; j < 8u; j++) { sS[r*8+j] *= scale; rm = max(rm, sS[r*8+j]); }
            float mnew = max(mm[r], rm);
            float corr = exp(mm[r] - mnew);
            float ps = 0.0f;
            for (uint j = 0; j < 8u; j++) { float p = exp(sS[r*8+j] - mnew); sS[r*8+j] = p; ps += p; }
            ll[r] = corr*ll[r] + ps; mm[r] = mnew; cc[r] = corr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // PV = P @ V  [8,HD]
        simdgroup_float8x8 pt;
        simdgroup_load(pt, sS, 8, 0, false);
        for (uint h = 0; h < HD; h += 8u) {
            simdgroup_float8x8 vt, pv = make_filled_simdgroup_matrix<float, 8>(0.f);
            simdgroup_load(vt, V + (ulong)(kb*H + head)*HD + h, rowstride, 0, false);
            simdgroup_multiply_accumulate(pv, pt, vt, pv);
            simdgroup_store(pv, sPV + h, HD, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tiitg; idx < 8u*HD; idx += 32u) { uint r = idx/HD; Oacc[idx] = cc[r]*Oacc[idx] + sPV[idx]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint idx = tiitg; idx < 8u*HD; idx += 32u) {
        uint r = idx/HD, i = idx%HD;
        O[(ulong)((qi0+r)*H + head)*HD + i] = Oacc[idx]/ll[r];
    }
    if (tiitg < 8u) { L[(ulong)(qi0+tiitg)*H + head] = mm[tiitg] + log(ll[tiitg]); }
}

// D[row] = sum_i dO[row,i]*O[row,i]  (row = query*H+head)  — for the softmax backward
kernel void t_flash_drow(device const float* dO [[buffer(0)]], device const float* O [[buffer(1)]],
    device float* D [[buffer(2)]], constant uint& HD [[buffer(3)]], constant uint& NR [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= NR) { return; }
    float s = 0.0f; for (uint i = 0; i < HD; i++) { s += dO[(ulong)g*HD+i]*O[(ulong)g*HD+i]; }
    D[g] = s;
}

// Flash backward dQ (per 8-query block): recompute S,P from L; dP=dO@V^T; dS=P*(dP-D); dQ=scale*dS@K
kernel void t_flash_mma_dq(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device const float* dO [[buffer(3)]], device const float* Lb [[buffer(4)]],
    device const float* Db [[buffer(5)]], device float* dQ [[buffer(6)]],
    constant uint& T [[buffer(7)]], constant uint& HD [[buffer(8)]], constant uint& H [[buffer(9)]],
    constant uint& W [[buffer(10)]],
    uint2 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]]) {
    uint qi0 = tgpig.x*8u, head = tgpig.y; uint rs = H*HD;
    threadgroup float sS[64]; threadgroup float sDS[64]; threadgroup float dQacc[8*128];
    threadgroup float Ll[8], Dd[8];
    for (uint idx = tiitg; idx < 8u*HD; idx += 32u) { dQacc[idx] = 0.0f; }
    if (tiitg < 8u) { Ll[tiitg] = Lb[(ulong)(qi0+tiitg)*H+head]; Dd[tiitg] = Db[(ulong)(qi0+tiitg)*H+head]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = 1.0f/sqrt(float(HD));
    for (uint kb = 0; kb < T; kb += 8u) {
        if (W != 0u && (qi0 > kb ? qi0 - kb : kb - qi0) > W) { continue; }   // local window
        simdgroup_float8x8 sacc = make_filled_simdgroup_matrix<float, 8>(0.f);
        simdgroup_float8x8 dpacc = make_filled_simdgroup_matrix<float, 8>(0.f);
        for (uint h = 0; h < HD; h += 8u) {
            simdgroup_float8x8 qt, kt, dot, vt;
            simdgroup_load(qt, Q + (ulong)(qi0*H+head)*HD + h, rs, 0, false);
            simdgroup_load(kt, K + (ulong)(kb*H+head)*HD + h, rs, 0, true);
            simdgroup_multiply_accumulate(sacc, qt, kt, sacc);
            simdgroup_load(dot, dO + (ulong)(qi0*H+head)*HD + h, rs, 0, false);
            simdgroup_load(vt, V + (ulong)(kb*H+head)*HD + h, rs, 0, true);
            simdgroup_multiply_accumulate(dpacc, dot, vt, dpacc);   // dP = dO @ V^T
        }
        simdgroup_store(sacc, sS, 8, 0, false);
        simdgroup_store(dpacc, sDS, 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg < 8u) { uint r = tiitg;
            for (uint j = 0; j < 8u; j++) { float p = exp(sS[r*8+j]*scale - Ll[r]); sDS[r*8+j] = p*(sDS[r*8+j] - Dd[r]); } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dst; simdgroup_load(dst, sDS, 8, 0, false);
        for (uint h = 0; h < HD; h += 8u) {
            simdgroup_float8x8 kt2, dq = make_filled_simdgroup_matrix<float, 8>(0.f);
            simdgroup_load(kt2, K + (ulong)(kb*H+head)*HD + h, rs, 0, false);
            simdgroup_multiply_accumulate(dq, dst, kt2, dq);        // dQ += dS @ K
            simdgroup_store(dq, sS + 0, 8, 0, false);               // reuse sS as 8x8 scratch
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = tiitg/8u; r < 8u; r += 4u) { dQacc[r*HD + h + tiitg%8u] += sS[r*8 + tiitg%8u]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint idx = tiitg; idx < 8u*HD; idx += 32u) { uint r = idx/HD, i = idx%HD; dQ[(ulong)((qi0+r)*H+head)*HD + i] = dQacc[idx]*scale; }
}

// Flash backward dK,dV (per 8-key block): loop queries; dV+=P^T@dO; dK+=scale*dS^T@Q
kernel void t_flash_mma_dkv(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device const float* dO [[buffer(3)]], device const float* Lb [[buffer(4)]],
    device const float* Db [[buffer(5)]], device float* dK [[buffer(6)]], device float* dV [[buffer(7)]],
    constant uint& T [[buffer(8)]], constant uint& HD [[buffer(9)]], constant uint& H [[buffer(10)]],
    constant uint& W [[buffer(11)]],
    uint2 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]]) {
    uint kj0 = tgpig.x*8u, head = tgpig.y; uint rs = H*HD;
    threadgroup float sS[64]; threadgroup float sP[64]; threadgroup float sDS[64];
    threadgroup float dKacc[8*128]; threadgroup float dVacc[8*128];
    for (uint idx = tiitg; idx < 8u*HD; idx += 32u) { dKacc[idx] = 0.0f; dVacc[idx] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = 1.0f/sqrt(float(HD));
    for (uint qb = 0; qb < T; qb += 8u) {
        if (W != 0u && (kj0 > qb ? kj0 - qb : qb - kj0) > W) { continue; }   // local window (symmetric with fwd)
        simdgroup_float8x8 sacc = make_filled_simdgroup_matrix<float, 8>(0.f);
        simdgroup_float8x8 dpacc = make_filled_simdgroup_matrix<float, 8>(0.f);
        for (uint h = 0; h < HD; h += 8u) {
            simdgroup_float8x8 qt, kt, dot, vt;
            simdgroup_load(qt, Q + (ulong)(qb*H+head)*HD + h, rs, 0, false);
            simdgroup_load(kt, K + (ulong)(kj0*H+head)*HD + h, rs, 0, true);
            simdgroup_multiply_accumulate(sacc, qt, kt, sacc);      // S[q,k]
            simdgroup_load(dot, dO + (ulong)(qb*H+head)*HD + h, rs, 0, false);
            simdgroup_load(vt, V + (ulong)(kj0*H+head)*HD + h, rs, 0, true);
            simdgroup_multiply_accumulate(dpacc, dot, vt, dpacc);   // dP[q,k]
        }
        simdgroup_store(sacc, sS, 8, 0, false);
        simdgroup_store(dpacc, sDS, 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg < 8u) { uint q = tiitg; float L = Lb[(ulong)(qb+q)*H+head], D = Db[(ulong)(qb+q)*H+head];
            for (uint j = 0; j < 8u; j++) { float p = exp(sS[q*8+j]*scale - L); sP[q*8+j] = p; sDS[q*8+j] = p*(sDS[q*8+j] - D); } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 pT, dsT; simdgroup_load(pT, sP, 8, 0, true); simdgroup_load(dsT, sDS, 8, 0, true);   // transposed
        for (uint h = 0; h < HD; h += 8u) {
            simdgroup_float8x8 dot2, qt2, dv = make_filled_simdgroup_matrix<float, 8>(0.f), dk = make_filled_simdgroup_matrix<float, 8>(0.f);
            simdgroup_load(dot2, dO + (ulong)(qb*H+head)*HD + h, rs, 0, false);
            simdgroup_multiply_accumulate(dv, pT, dot2, dv);        // dV += P^T @ dO
            simdgroup_load(qt2, Q + (ulong)(qb*H+head)*HD + h, rs, 0, false);
            simdgroup_multiply_accumulate(dk, dsT, qt2, dk);        // dK += dS^T @ Q
            simdgroup_store(dv, sS, 8, 0, false); threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = tiitg/8u; r < 8u; r += 4u) { dVacc[r*HD + h + tiitg%8u] += sS[r*8 + tiitg%8u]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(dk, sS, 8, 0, false); threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = tiitg/8u; r < 8u; r += 4u) { dKacc[r*HD + h + tiitg%8u] += sS[r*8 + tiitg%8u]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint idx = tiitg; idx < 8u*HD; idx += 32u) { uint r = idx/HD, i = idx%HD;
        dK[(ulong)((kj0+r)*H+head)*HD + i] = dKacc[idx]*scale; dV[(ulong)((kj0+r)*H+head)*HD + i] = dVacc[idx]; }
}

// --- Muon optimizer helpers (Newton-Schulz orthogonalization) ---
kernel void t_scale(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
    constant float& s [[buffer(2)]], constant uint& N [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < N) { dst[i] = s*src[i]; }
}
kernel void t_lincomb2(device float* dst [[buffer(0)]], device const float* x [[buffer(1)]],
    device const float* y [[buffer(2)]], constant float& a [[buffer(3)]], constant float& b [[buffer(4)]],
    constant uint& N [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i < N) { dst[i] = a*x[i] + b*y[i]; }
}
kernel void t_sumsq(device const float* src [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& N [[buffer(2)]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sh[256];
    float s = 0.0f;
    for (uint i = tid; i < N; i += 256u) { s += src[i]*src[i]; }
    sh[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = 128u; st > 0u; st >>= 1u) { if (tid < st) { sh[tid] += sh[tid+st]; } threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (tid == 0u) { out[0] = sh[0]; }
}
// Parallel sum-of-squares: G threadgroups grid-stride the buffer -> G partial sums out[0..G].
// Caller sums the G partials (CPU). Fixes t_sumsq's single-threadgroup serialization on big buffers.
kernel void t_sumsq_g(device const float* src [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& N [[buffer(2)]], uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint ntg [[threadgroups_per_grid]]) {
    threadgroup float sh[256];
    float s = 0.0f;
    for (uint i = tgp*256u + tid; i < N; i += ntg*256u) { s += src[i]*src[i]; }
    sh[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = 128u; st > 0u; st >>= 1u) { if (tid < st) { sh[tid] += sh[tid+st]; } threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (tid == 0u) { out[tgp] = sh[0]; }
}
kernel void t_scale_rnorm(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
    device const float* nrm [[buffer(2)]], constant uint& N [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < N) { dst[i] = src[i] * rsqrt(nrm[0] + 1e-12f); }
}
kernel void t_transpose(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
    constant uint& R [[buffer(2)]], constant uint& C [[buffer(3)]], uint g [[thread_position_in_grid]]) {
    if (g >= R*C) { return; }
    uint r = g/C, c = g%C; dst[(ulong)c*R + r] = src[g];
}
kernel void t_muon_update(device half* w [[buffer(0)]], device const float* o [[buffer(1)]],
    constant float& lrs [[buffer(2)]], constant uint& N [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < N) { w[i] = half(float(w[i]) - lrs*o[i]); }
}

// --- backward gather/scatter ---
// d_eo[r,:] = grow_gate[r] * dout[gidx[r],:]   (gather d_out into per-expert groups, gate-scaled)
kernel void t_moe_gather_scaled(device float* deo [[buffer(0)]], device const float* dout [[buffer(1)]],
    device const uint* gidx [[buffer(2)]], device const float* gg [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& TOT [[buffer(5)]], uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }
    uint r = g / D, i = g % D;
    deo[g] = gg[r] * dout[(ulong)gidx[r]*D + i];
}

// d_gate_row[r] = <eo[r,:], dout[gidx[r],:]>   (gate gradient, per gathered row)
kernel void t_moe_dgate_gs(device float* dgr [[buffer(0)]], device const float* eo [[buffer(1)]],
    device const float* dout [[buffer(2)]], device const uint* gidx [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& NR [[buffer(5)]], uint r [[thread_position_in_grid]]) {
    if (r >= NR) { return; }
    ulong tb = (ulong)gidx[r]*D; float s = 0.0f;
    for (uint i = 0; i < D; i++) { s += eo[(ulong)r*D + i]*dout[tb + i]; }
    dgr[r] = s;
}

// d_h2[t,:] += sum_j dgath[tok2rows[t,j],:]   (scatter expert input-grads back, accumulate on shared's)
kernel void t_moe_scatter_dh2(device float* dh2 [[buffer(0)]], device const float* dgath [[buffer(1)]],
    device const uint* tok2 [[buffer(2)]], constant uint& D [[buffer(3)]], constant uint& K [[buffer(4)]],
    constant uint& TOT [[buffer(5)]], uint g [[thread_position_in_grid]]) {
    if (g >= TOT) { return; }
    uint t = g / D, i = g % D; float s = dh2[g];
    for (uint j = 0; j < K; j++) { uint r = tok2[(ulong)t*K + j]; s += dgath[(ulong)r*D + i]; }
    dh2[g] = s;
}

// dgate_dense[gidx[r], rexp[r]] = dgr[r]   (scatter row gate-grads into the dense [T,ne] layout)
kernel void t_moe_scatter_dgate(device float* dgd [[buffer(0)]], device const float* dgr [[buffer(1)]],
    device const uint* gidx [[buffer(2)]], device const uint* rexp [[buffer(3)]],
    constant uint& NE [[buffer(4)]], constant uint& NR [[buffer(5)]], uint r [[thread_position_in_grid]]) {
    if (r >= NR) { return; }
    dgd[(ulong)gidx[r]*NE + rexp[r]] = dgr[r];
}

// dgate[t,EIDX] = <expert_out[t,:], d_out[t,:]>   (gate gradient, per token)
kernel void t_moe_dgate(device float* dgate [[buffer(0)]], device const float* oe [[buffer(1)]],
    device const float* dout [[buffer(2)]], constant uint& D [[buffer(3)]], constant uint& NE [[buffer(4)]],
    constant uint& EIDX [[buffer(5)]], constant uint& T [[buffer(6)]], uint t [[thread_position_in_grid]]) {
    if (t >= T) { return; }
    float s = 0.0f;
    for (uint i = 0; i < D; i++) { s += oe[(ulong)t*D + i]*dout[(ulong)t*D + i]; }
    dgate[(ulong)t*NE + EIDX] = s;
}

// ===================== Tokenizer-free front-end: segment scatter-mean pool =====================
// bytes[T,D] -> patches[KP,D].  hp[k,:] = mean_{t: pid[t]==k} z[t,:]  (k in 1..KP; patch 0 = BOS).
// Atomic-free: one thread per (patch k, dim i) gathers its segment. pid[t] in 1..=KMAX (0 reserved
// for BOS). count[k] passed in (computed CPU-side from pid, matching the MoE-routing convention).
// t_pool_fwd fwd: hp[0,:]=bos[:]; hp[k>0,:]=sum_{t:pid[t]==k} z[t,:] / max(count[k],1).
kernel void t_pool_fwd(device float* hp [[buffer(0)]], device const float* z [[buffer(1)]],
    device const uint* pid [[buffer(2)]], device const uint* count [[buffer(3)]], device const float* bos [[buffer(4)]],
    constant uint& D [[buffer(5)]], constant uint& T [[buffer(6)]], constant uint& KP [[buffer(7)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= KP*D) { return; }
    uint k = g / D, i = g % D;
    if (k == 0u) { hp[g] = bos[i]; return; }
    float acc = 0.0f;
    for (uint t = 0; t < T; t++) { if (pid[t] == k) { acc += z[(ulong)t*D + i]; } }
    float c = float(count[k]); if (c < 1.0f) { c = 1.0f; }
    hp[g] = acc / c;
}
// t_pool_bwd bwd: dz[t,:] = dhp[pid[t],:] / max(count[pid[t]],1).  (BOS grad handled separately.)
kernel void t_pool_bwd(device float* dz [[buffer(0)]], device const float* dhp [[buffer(1)]],
    device const uint* pid [[buffer(2)]], device const uint* count [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& T [[buffer(5)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= T*D) { return; }
    uint t = g / D, i = g % D;
    uint k = pid[t];
    float c = float(count[k]); if (c < 1.0f) { c = 1.0f; }
    dz[g] = dhp[(ulong)k*D + i] / c;
}

// ===================== Tokenizer-free front-end: byte->patch cross-attention =====================
// Scalar online-softmax (correctness-first, mirrors t_flash_attn_fwd). Q=byte queries [T,H,HD],
// K/V = patch keys/vals [KP,H,HD]. Band mask: query t attends patch kp iff (kp==0 || kp < pid[t]).
// Saves L[t,head]=logsumexp for the backward. scale = 1/sqrt(HD). HD <= 128.
kernel void t_xattn_fwd(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* Vv [[buffer(2)]], device float* O [[buffer(3)]], device float* L [[buffer(4)]],
    device const uint* pid [[buffer(5)]], constant uint& T [[buffer(6)]], constant uint& KP [[buffer(7)]],
    constant uint& HD [[buffer(8)]], constant uint& H [[buffer(9)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= H*T) { return; }
    uint head = g % H, qi = g / H;
    float scale = 1.0f/sqrt(float(HD));
    uint pt = pid[qi];
    device const float* q = Q + ((ulong)qi*H + head)*HD;
    float m = -1e30f, l = 0.0f; float acc[128];
    for (uint i = 0; i < HD; i++) { acc[i] = 0.0f; }
    for (uint kp = 0; kp < KP; kp++) {
        if (!(kp == 0u || kp < pt)) { continue; }
        device const float* kk = K + ((ulong)kp*H + head)*HD;
        float s = 0.0f; for (uint i = 0; i < HD; i++) { s += q[i]*kk[i]; }
        s *= scale;
        float mnew = max(m, s); float corr = exp(m - mnew); float p = exp(s - mnew);
        l = corr*l + p;
        device const float* vv = Vv + ((ulong)kp*H + head)*HD;
        for (uint i = 0; i < HD; i++) { acc[i] = corr*acc[i] + p*vv[i]; }
        m = mnew;
    }
    device float* o = O + ((ulong)qi*H + head)*HD;
    for (uint i = 0; i < HD; i++) { o[i] = acc[i]/l; }
    L[(ulong)qi*H + head] = m + log(l);
}
// dQ: one thread per (query,head). D[qi,head]=dO.O (via t_flash_drow). Recompute p from L.
kernel void t_xattn_dq(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* Vv [[buffer(2)]], device const float* dO [[buffer(3)]], device const float* L [[buffer(4)]],
    device const float* Dr [[buffer(5)]], device float* dQ [[buffer(6)]], device const uint* pid [[buffer(7)]],
    constant uint& T [[buffer(8)]], constant uint& KP [[buffer(9)]], constant uint& HD [[buffer(10)]], constant uint& H [[buffer(11)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= H*T) { return; }
    uint head = g % H, qi = g / H;
    float scale = 1.0f/sqrt(float(HD));
    uint pt = pid[qi];
    device const float* q = Q + ((ulong)qi*H + head)*HD;
    device const float* d = dO + ((ulong)qi*H + head)*HD;
    float Lq = L[(ulong)qi*H + head], Dq = Dr[(ulong)qi*H + head];
    float dq[128]; for (uint i = 0; i < HD; i++) { dq[i] = 0.0f; }
    for (uint kp = 0; kp < KP; kp++) {
        if (!(kp == 0u || kp < pt)) { continue; }
        device const float* kk = K + ((ulong)kp*H + head)*HD;
        device const float* vv = Vv + ((ulong)kp*H + head)*HD;
        float s = 0.0f, dp = 0.0f;
        for (uint i = 0; i < HD; i++) { s += q[i]*kk[i]; dp += d[i]*vv[i]; }
        float p = exp(s*scale - Lq);
        float ds = p*(dp - Dq);                       // dS wrt scaled score
        for (uint i = 0; i < HD; i++) { dq[i] += ds*kk[i]; }
    }
    device float* o = dQ + ((ulong)qi*H + head)*HD;
    for (uint i = 0; i < HD; i++) { o[i] = dq[i]*scale; }
}
// dK,dV: one thread per (patch kp,head). Loop queries that attend this patch.
kernel void t_xattn_dkv(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* Vv [[buffer(2)]], device const float* dO [[buffer(3)]], device const float* L [[buffer(4)]],
    device const float* Dr [[buffer(5)]], device float* dK [[buffer(6)]], device float* dV [[buffer(7)]], device const uint* pid [[buffer(8)]],
    constant uint& T [[buffer(9)]], constant uint& KP [[buffer(10)]], constant uint& HD [[buffer(11)]], constant uint& H [[buffer(12)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= H*KP) { return; }
    uint head = g % H, kp = g / H;
    float scale = 1.0f/sqrt(float(HD));
    device const float* kk = K + ((ulong)kp*H + head)*HD;
    device const float* vv = Vv + ((ulong)kp*H + head)*HD;
    float dk[128], dv[128];
    for (uint i = 0; i < HD; i++) { dk[i] = 0.0f; dv[i] = 0.0f; }
    for (uint qi = 0; qi < T; qi++) {
        uint pt = pid[qi];
        if (!(kp == 0u || kp < pt)) { continue; }
        device const float* q = Q + ((ulong)qi*H + head)*HD;
        device const float* d = dO + ((ulong)qi*H + head)*HD;
        float Lq = L[(ulong)qi*H + head], Dq = Dr[(ulong)qi*H + head];
        float s = 0.0f, dp = 0.0f;
        for (uint i = 0; i < HD; i++) { s += q[i]*kk[i]; dp += d[i]*vv[i]; }
        float p = exp(s*scale - Lq);
        float ds = p*(dp - Dq);
        for (uint i = 0; i < HD; i++) { dv[i] += p*d[i]; dk[i] += ds*q[i]; }
    }
    device float* ok = dK + ((ulong)kp*H + head)*HD;
    device float* ov = dV + ((ulong)kp*H + head)*HD;
    for (uint i = 0; i < HD; i++) { ok[i] = dk[i]*scale; ov[i] = dv[i]; }
}
"#;
