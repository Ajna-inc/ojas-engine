// Bodies compile against kernels::PRELUDE (shared defines + helpers).
pub const BODY: &str = r#"
// Broadcast a length-N bias across all M rows of x[M,N] (total = M*N threads).
kernel void add_rowbias_m(device float* x [[buffer(0)]], device const float* b [[buffer(1)]],
    constant uint& N [[buffer(2)]], constant uint& total [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid < total) { x[gid] += b[gid % N]; }
}

// Gemma4: q-norm + k-norm + weightless V-norm in one dispatch (one fewer per layer).
// heads [0,nq)=q, [nq,nq+nk)=k, [nq+nk,nq+nk+nv)=v (weight vw, e.g. all-ones).
kernel void qkv_rmsnorm(device float* vq [[buffer(0)]], device float* vk [[buffer(1)]],
    device float* vv [[buffer(2)]], device const float* qw [[buffer(3)]],
    device const float* kw [[buffer(4)]], device const float* vw [[buffer(5)]],
    constant uint& hd [[buffer(6)]], constant uint& nq [[buffer(7)]], constant uint& nk [[buffer(8)]],
    constant uint& nv [[buffer(9)]], constant float& eps [[buffer(10)]],
    uint2 gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint head = gid.x; uint tok = gid.y;
    device float* v; device const float* w; uint h; uint rowdim;
    if (head < nq)          { v = vq; w = qw; h = head;          rowdim = nq*hd; }
    else if (head < nq+nk)  { v = vk; w = kw; h = head - nq;     rowdim = nk*hd; }
    else                    { v = vv; w = vw; h = head - nq - nk; rowdim = nv*hd; }
    uint base = tok*rowdim + h*hd;
    float ss = 0.0;
    for (uint i = lane; i < hd; i += 32u) { float x = v[base+i]; ss += x*x; }
    ss = simd_sum(ss);
    float inv = rsqrt(ss/float(hd) + eps);
    for (uint i = lane; i < hd; i += 32u) { v[base+i] = v[base+i]*inv*w[i]; }
}

kernel void gemv_f16(device const float* x [[buffer(0)]], device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    GEMV_DOT
    if (lane == 0u) { y[n] = p; }
}

// M-row f16 GEMV, the f16 twin of gemv_m_q8: reads each weight row once and
// produces M outputs from it. Looping gemv_f16 per token instead re-reads the whole
// weight matrix for every row, which on a model whose dense skeleton is ~88% of
// per-token bytes turns a 2-token verify into two full forwards' worth of traffic.
// M <= 8 (the accumulator array), K % 4 == 0 (half4/float4 loads).
kernel void gemv_m_f16(device const float* x [[buffer(0)]], device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid; if (n >= N) { return; }
    device const half4* row = (device const half4*)(w + (ulong)n*(ulong)K);
    uint K4 = K/4u;
    float p[8]; for (uint m=0u;m<M;m++) p[m]=0.0;
    for (uint k = lane; k < K4; k += 32u) {
        float4 w4 = float4(row[k]);
        for (uint m=0u;m<M;m++) { device const float4* xm=(device const float4*)(x + (ulong)m*(ulong)K); p[m]+=dot(w4, xm[k]); }
    }
    for (uint m=0u;m<M;m++) { float r = simd_sum(p[m]); if (lane==0u) y[m*N+n]=r; }
}

kernel void gemv_bias(device const float* x [[buffer(0)]], device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* bias [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    GEMV_DOT
    if (lane == 0u) { y[n] = p + bias[n]; }
}

kernel void gemv_accum(device const float* x [[buffer(0)]], device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    GEMV_DOT
    if (lane == 0u) { y[n] += p; }
}

kernel void gemv_q8(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q8
    if (lane == 0u) { y[n] = p; }
}

kernel void gemv_q8_bias(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]], device const float* bias [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q8
    if (lane == 0u) { y[n] = p + bias[n]; }
}

kernel void gemv_q8_accum(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q8
    if (lane == 0u) { y[n] += p; }
}

kernel void gemv_q8_ksplit(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q8_KSPLIT
    if (w0) { y[n] = p; }
}

kernel void gemv_q8_ksplit_accum(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q8_KSPLIT
    if (w0) { y[n] += p; }
}

kernel void gemv_q4(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q4
    if (lane == 0u) { y[n] = p; }
}

kernel void gemv_q4_accum(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q4
    if (lane == 0u) { y[n] += p; }
}

// Q4L gemv: Q4_K values, fast layout. Buffers mirror gemv_q4_fast plus a second
// f16 side array (d1 and -m1 rather than one symmetric scale).
kernel void gemv_q4l(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], device const half* qb_ [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q4L_FAST
    if (lane == 0u) { for (uint r = 0u; r < 4u; r++) if (out_row+r < N) y[out_row+r] = res[r]; }
}

kernel void gemv_q4l_accum(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], device const half* qb_ [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q4L_FAST
    if (lane == 0u) { for (uint r = 0u; r < 4u; r++) if (out_row+r < N) y[out_row+r] += res[r]; }
}

// Fused SwiGLU over Q4L. Same 4-rows-per-simdgroup shape as gemv_q4l, with gate
// and up sharing one pass over the activations — the largest category per layer.
kernel void ffn_gu_q4l(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& act [[buffer(8)]],
    device const half* ga [[buffer(6)]], device const half* gb [[buffer(7)]],
    device const half* ua [[buffer(9)]], device const half* ub [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    uint nblk = K/32u; uint rb = K/2u;
    device const uchar* gr = wg + (ulong)out_row*(ulong)rb + (ulong)lane*8u;
    device const uchar* ur = wu + (ulong)out_row*(ulong)rb + (ulong)lane*8u;
    device const half* gaa = ga + (ulong)out_row*(ulong)nblk + lane/2u;
    device const half* gbb = gb + (ulong)out_row*(ulong)nblk + lane/2u;
    device const half* uaa = ua + (ulong)out_row*(ulong)nblk + lane/2u;
    device const half* ubb = ub + (ulong)out_row*(ulong)nblk + lane/2u;
    device const float* xr = x + lane*16u;
    float rg[4] = {0.0, 0.0, 0.0, 0.0};
    float ru[4] = {0.0, 0.0, 0.0, 0.0};
    uint kfull = (K/512u)*512u;
    for (uint k = 0u; k < kfull; k += 512u) {
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
        for (uint r = 0u; r < 4u; r++) { Q4L_DOT(gr, r, xt, xsum, gaa, gbb, rg) Q4L_DOT(ur, r, xt, xsum, uaa, ubb, ru) }
        gr += 256u; ur += 256u; gaa += 16u; gbb += 16u; uaa += 16u; ubb += 16u; xr += 512u;
    }
    if (lane*16u < K - kfull) {
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
        for (uint r = 0u; r < 4u; r++) { Q4L_DOT(gr, r, xt, xsum, gaa, gbb, rg) Q4L_DOT(ur, r, xt, xsum, uaa, ubb, ru) }
    }
    for (uint r = 0u; r < 4u; r++) { rg[r] = simd_sum(rg[r]); ru[r] = simd_sum(ru[r]); }
    if (lane == 0u) {
        for (uint r = 0u; r < 4u; r++) if (out_row+r < N) out[out_row+r] = ffn_act(rg[r], act)*ru[r];
    }
}

// M-row Q4L gemv: the batched-prefill counterpart of gemv_q4l. Reads each weight
// block once and dots it against all M activation rows, so prefill is compute-bound
// rather than bandwidth-bound like decode (one token at a time measured 90 tok/s
// against the reference's 1018 on the same model).
//
// Nibbles are unpacked once per block into registers and reused across the M rows;
// unpacking per row would put back the ALU cost that the batching removes.
// Lane-partitioned verify matvec (kernel_mul_mv_ext structure).
//
// Every other Q4L kernel here gives a lane N rows and has it stride the whole K, so
// live registers scale as rows x tokens; hoisting the nibble unpack out of the token
// loop needs the unpacked weights live, and at 4 rows/lane there is no room for it.
// The reference partitions the simdgroup instead (kernel_mul_mv_ext_q4x4_f32_impl):
// NXPSG lanes cooperate on one row and 32/NXPSG rows run per simdgroup, so a lane
// holds exactly one row's dequantized block and the hoist fits.
//
//     ours (4 rows/lane)   rows*tokens acc + 4 rows of weights + xt[16]  ~48 regs
//     theirs (1 row/lane)  tokens acc + one 16-float block + inline x    ~26 regs
//
// The block is therefore dequantized once per k-chunk and dotted against all R1PTG
// tokens, worth -28% of ALU. Activations are read as float4 and consumed
// immediately rather than staged into a register array, worth another -37%. The tail
// reduction is a partial shuffle over NXPSG lanes, not a full simd_sum.
//
// Q4L makes this cheaper than it is for the reference: nibbles are already
// contiguous per row with the scales hoisted into side arrays, so a lane owns a
// contiguous run of 16-weight chunks (one ushort4 each) instead of walking the
// interleaved 144-byte Q4_K super-block. Chunk c covers weights [16c, 16c+16),
// inside 32-weight block c/2 — needs K % 32 == 0.
//
// NXPSG and R1PTG are compile-time (one kernel per pair): an array subscripted by a
// runtime loop variable cannot stay in registers.
#define XV_KERNEL(NAME, NXPSG, R1PTG) \
kernel void NAME(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]], \
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], \
    device const half* qa_ [[buffer(5)]], device const half* qb_ [[buffer(6)]], \
    constant uint& M [[buffer(7)]], constant uint& accum [[buffer(8)]], \
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) { \
    const uint nypsg = 32u/(NXPSG); \
    uint tx = lane % (NXPSG); \
    uint ty = lane / (NXPSG); \
    uint out_row = tgid*(ts/(NXPSG)) + sgid*nypsg + ty; \
    if (out_row >= N) { return; } \
    uint nblk = K/32u; uint nch = K/16u; \
    device const uchar* row = w4 + (ulong)out_row*(ulong)(K/2u); \
    device const half* qa = qa_ + (ulong)out_row*(ulong)nblk; \
    device const half* qb = qb_ + (ulong)out_row*(ulong)nblk; \
    float sumf[R1PTG]; \
    for (uint m = 0u; m < (R1PTG); m++) { sumf[m] = 0.0; } \
    for (uint c = tx; c < nch; c += (NXPSG)) { \
        ushort4 wv = *(device const ushort4*)(row + (ulong)c*8u); \
        float A = float(qa[c/2u]); \
        float B = float(qb[c/2u]); \
        /* dequantize the 16-weight chunk ONCE: mask, convert, and fold in A and \
           the mask trick's 1/16^j so the activations below are read raw. */ \
        float lx[16]; \
        for (uint i = 0u; i < 4u; i++) { \
            ushort v = wv[i]; \
            lx[4u*i+0u] = float(v & 0x000fu)*A; \
            lx[4u*i+1u] = float(v & 0x00f0u)*(A*(1.0/16.0)); \
            lx[4u*i+2u] = float(v & 0x0f00u)*(A*(1.0/256.0)); \
            lx[4u*i+3u] = float(v & 0xf000u)*(A*(1.0/4096.0)); \
        } \
        for (uint m = 0u; m < (R1PTG); m++) { \
            device const float* xm = x + (ulong)m*(ulong)K + (ulong)c*16u; \
            float acc = 0.0, xsum = 0.0; \
            for (uint i = 0u; i < 4u; i++) { \
                float4 xv = *(device const float4*)(xm + 4u*i); \
                xsum += xv.x + xv.y + xv.z + xv.w; \
                acc += lx[4u*i+0u]*xv.x + lx[4u*i+1u]*xv.y + lx[4u*i+2u]*xv.z + lx[4u*i+3u]*xv.w; \
            } \
            sumf[m] += acc + B*xsum; \
        } \
    } \
    for (uint m = 0u; m < (R1PTG); m++) { \
        if ((NXPSG) >= 32u) { sumf[m] += simd_shuffle_down(sumf[m], 16u); } \
        if ((NXPSG) >= 16u) { sumf[m] += simd_shuffle_down(sumf[m],  8u); } \
        if ((NXPSG) >=  8u) { sumf[m] += simd_shuffle_down(sumf[m],  4u); } \
        if ((NXPSG) >=  4u) { sumf[m] += simd_shuffle_down(sumf[m],  2u); } \
        if ((NXPSG) >=  2u) { sumf[m] += simd_shuffle_down(sumf[m],  1u); } \
    } \
    if (tx == 0u) { \
        for (uint m = 0u; m < (R1PTG); m++) { \
            if (accum != 0u) { y[(ulong)m*(ulong)N + out_row] += sumf[m]; } \
            else             { y[(ulong)m*(ulong)N + out_row]  = sumf[m]; } \
        } \
    } \
}

XV_KERNEL(gemv_x8_1_q4l,  8u, 1u)
XV_KERNEL(gemv_x8_2_q4l,  8u, 2u)
XV_KERNEL(gemv_x8_3_q4l,  8u, 3u)
XV_KERNEL(gemv_x8_4_q4l,  8u, 4u)
XV_KERNEL(gemv_x8_5_q4l,  8u, 5u)
XV_KERNEL(gemv_x8_6_q4l,  8u, 6u)
XV_KERNEL(gemv_x8_7_q4l,  8u, 7u)
XV_KERNEL(gemv_x8_8_q4l,  8u, 8u)
// NXPSG is picked by shape, not by K alignment as the reference does it. Rows per
// threadgroup are 256/NXPSG, so a narrow-N matmul starves for threadgroups at
// NXPSG=4 (ffn_down N=2048 gets 32 of them on a 38-core GPU). Measured at M=4,
// ms/call, against the row-blocked kernel:
//
//                       ffn_gu (K2048,N11008)   ffn_down (K11008,N2048)
//     row-blocked            0.1874                  0.1271
//     NXPSG=4                0.1685  (-10%)          0.1413  (+11%)
//     NXPSG=8                0.1947  (+4%)           0.1221  (-4%)
//     NXPSG=16               0.2705                  0.1694
//
// On qkv/o_proj (K2048, N<=2048) the row-blocked kernel wins outright (o_proj
// 0.0175 vs 0.0247), so those keep it. NXPSG=16 loses everywhere and is kept only as
// the sweep's upper point.
XV_KERNEL(gemv_x4_1_q4l,  4u, 1u)
XV_KERNEL(gemv_x4_2_q4l,  4u, 2u)
XV_KERNEL(gemv_x4_3_q4l,  4u, 3u)
XV_KERNEL(gemv_x4_4_q4l,  4u, 4u)
XV_KERNEL(gemv_x4_5_q4l,  4u, 5u)
XV_KERNEL(gemv_x4_6_q4l,  4u, 6u)
XV_KERNEL(gemv_x4_7_q4l,  4u, 7u)
XV_KERNEL(gemv_x4_8_q4l,  4u, 8u)
XV_KERNEL(gemv_x16_4_q4l, 16u, 4u)

// Small-M Q4L matvec for the speculative verify: 4 rows x up to 4 tokens per
// simdgroup pair, tokens split across simdgroups instead of blocked in registers.
//
// Blocking all 4 tokens in registers (res[4][4] + wv[4][4] + xt[16], ~55 live
// registers) measured 2.7-4.5x the decode kernels' cost on the same weights — the
// register-occupancy cliff that decides every experiment in this file. Instead a
// pair of simdgroups shares the same 4 rows: each loads the row bytes itself (the
// second load hits L2, since the pair runs concurrently in one threadgroup) and
// handles only 2 tokens, so live registers stay near the decode kernel's budget
// while DRAM still sees the weights about once.
kernel void gemv_m4_q4l(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], device const half* qb_ [[buffer(6)]],
    constant uint& M [[buffer(7)]], constant uint& accum [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint pair = sgid / 2u;                       // row group within the tg
    uint th   = sgid % 2u;                       // token half: tokens [2*th, 2*th+2)
    uint out_row = tgid*(ts/64u*4u) + pair*4u; if (out_row >= N) { return; }
    uint m0 = th*2u;
    if (m0 >= M) { return; }
    uint mm = min(M - m0, 2u);
    uint nblk = K/32u; uint rb = K/2u;
    device const uchar* wr = w4 + (ulong)out_row*(ulong)rb + (ulong)lane*8u;
    device const half* qa = qa_ + (ulong)out_row*(ulong)nblk + lane/2u;
    device const half* qb = qb_ + (ulong)out_row*(ulong)nblk + lane/2u;
    float r0[4] = {0.0, 0.0, 0.0, 0.0};
    float r1[4] = {0.0, 0.0, 0.0, 0.0};
    uint kfull = (K/512u)*512u;
    for (uint k = 0u; k < kfull + 512u; k += 512u) {
        bool tail = (k >= kfull);
        if (tail && !(lane*16u < K - kfull)) { break; }
        {
            device const float* xr = x + (ulong)m0*(ulong)K + k + lane*16u;
            float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
            for (uint r = 0u; r < 4u; r++) Q4L_DOT(wr, r, xt, xsum, qa, qb, r0)
        }
        if (mm > 1u) {
            device const float* xr = x + (ulong)(m0 + 1u)*(ulong)K + k + lane*16u;
            float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
            for (uint r = 0u; r < 4u; r++) Q4L_DOT(wr, r, xt, xsum, qa, qb, r1)
        }
        wr += 256u; qa += 16u; qb += 16u;
    }
    for (uint r = 0u; r < 4u; r++) { r0[r] = simd_sum(r0[r]); r1[r] = simd_sum(r1[r]); }
    if (lane == 0u) {
        for (uint r = 0u; r < 4u; r++) {
            uint o = out_row + r;
            if (o >= N) { break; }
            {
                if (accum != 0u) { y[(ulong)m0*(ulong)N + o] += r0[r]; }
                else             { y[(ulong)m0*(ulong)N + o]  = r0[r]; }
            }
            if (mm > 1u) {
                if (accum != 0u) { y[(ulong)(m0+1u)*(ulong)N + o] += r1[r]; }
                else             { y[(ulong)(m0+1u)*(ulong)N + o]  = r1[r]; }
            }
        }
    }
}
// Ablation probes: numerically wrong by construction, for timing only. The M=4
// verify costs 2.2-2.6x its own decode cost per category on weights it reads exactly
// once; these separate ALU cost (the nibble unpack, redone per token) from load
// issue (the second token's activation stream).
//
// _alu keeps every load and deletes ~90% of the arithmetic.
// _ld  keeps every arithmetic op and deletes the second token's x loads.
#define Q4L_DOT_NOALU(wr_, r_, xt_, xsum_, a_, b_, res_) { \
    ushort4 wv4 = *(device const ushort4*)((wr_) + (r_)*rb); \
    float A = float((a_)[(r_)*nblk]); float B = float((b_)[(r_)*nblk]); \
    (res_)[r_] += A*float(wv4[0])*(xt_)[0] + B*(xsum_); }

kernel void gemv_m4_alu(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], device const half* qb_ [[buffer(6)]],
    constant uint& M [[buffer(7)]], constant uint& accum [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint pair = sgid / 2u; uint th = sgid % 2u;
    uint out_row = tgid*(ts/64u*4u) + pair*4u; if (out_row >= N) { return; }
    uint m0 = th*2u; if (m0 >= M) { return; }
    uint mm = min(M - m0, 2u);
    uint nblk = K/32u; uint rb = K/2u;
    device const uchar* wr = w4 + (ulong)out_row*(ulong)rb + (ulong)lane*8u;
    device const half* qa = qa_ + (ulong)out_row*(ulong)nblk + lane/2u;
    device const half* qb = qb_ + (ulong)out_row*(ulong)nblk + lane/2u;
    float r0[4] = {0.0, 0.0, 0.0, 0.0};
    float r1[4] = {0.0, 0.0, 0.0, 0.0};
    uint kfull = (K/512u)*512u;
    for (uint k = 0u; k < kfull + 512u; k += 512u) {
        bool tail = (k >= kfull);
        if (tail && !(lane*16u < K - kfull)) { break; }
        {
            device const float* xr = x + (ulong)m0*(ulong)K + k + lane*16u;
            float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
            for (uint r = 0u; r < 4u; r++) Q4L_DOT_NOALU(wr, r, xt, xsum, qa, qb, r0)
        }
        if (mm > 1u) {
            device const float* xr = x + (ulong)(m0 + 1u)*(ulong)K + k + lane*16u;
            float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
            for (uint r = 0u; r < 4u; r++) Q4L_DOT_NOALU(wr, r, xt, xsum, qa, qb, r1)
        }
        wr += 256u; qa += 16u; qb += 16u;
    }
    for (uint r = 0u; r < 4u; r++) { r0[r] = simd_sum(r0[r]); r1[r] = simd_sum(r1[r]); }
    if (lane == 0u) {
        for (uint r = 0u; r < 4u; r++) {
            uint o = out_row + r; if (o >= N) { break; }
            if (accum != 0u) { y[(ulong)m0*(ulong)N + o] += r0[r]; } else { y[(ulong)m0*(ulong)N + o] = r0[r]; }
            if (mm > 1u) {
                if (accum != 0u) { y[(ulong)(m0+1u)*(ulong)N + o] += r1[r]; }
                else             { y[(ulong)(m0+1u)*(ulong)N + o]  = r1[r]; }
            }
        }
    }
}

kernel void gemv_m4_ld(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], device const half* qb_ [[buffer(6)]],
    constant uint& M [[buffer(7)]], constant uint& accum [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint pair = sgid / 2u; uint th = sgid % 2u;
    uint out_row = tgid*(ts/64u*4u) + pair*4u; if (out_row >= N) { return; }
    uint m0 = th*2u; if (m0 >= M) { return; }
    uint mm = min(M - m0, 2u);
    uint nblk = K/32u; uint rb = K/2u;
    device const uchar* wr = w4 + (ulong)out_row*(ulong)rb + (ulong)lane*8u;
    device const half* qa = qa_ + (ulong)out_row*(ulong)nblk + lane/2u;
    device const half* qb = qb_ + (ulong)out_row*(ulong)nblk + lane/2u;
    float r0[4] = {0.0, 0.0, 0.0, 0.0};
    float r1[4] = {0.0, 0.0, 0.0, 0.0};
    uint kfull = (K/512u)*512u;
    for (uint k = 0u; k < kfull + 512u; k += 512u) {
        bool tail = (k >= kfull);
        if (tail && !(lane*16u < K - kfull)) { break; }
        device const float* xr = x + (ulong)m0*(ulong)K + k + lane*16u;
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
        for (uint r = 0u; r < 4u; r++) Q4L_DOT(wr, r, xt, xsum, qa, qb, r0)
        // second token: same activations reused, so the arithmetic is unchanged
        // but its 4 float4 loads are gone.
        if (mm > 1u) { for (uint r = 0u; r < 4u; r++) Q4L_DOT(wr, r, xt, xsum, qa, qb, r1) }
        wr += 256u; qa += 16u; qb += 16u;
    }
    for (uint r = 0u; r < 4u; r++) { r0[r] = simd_sum(r0[r]); r1[r] = simd_sum(r1[r]); }
    if (lane == 0u) {
        for (uint r = 0u; r < 4u; r++) {
            uint o = out_row + r; if (o >= N) { break; }
            if (accum != 0u) { y[(ulong)m0*(ulong)N + o] += r0[r]; } else { y[(ulong)m0*(ulong)N + o] = r0[r]; }
            if (mm > 1u) {
                if (accum != 0u) { y[(ulong)(m0+1u)*(ulong)N + o] += r1[r]; }
                else             { y[(ulong)(m0+1u)*(ulong)N + o]  = r1[r]; }
            }
        }
    }
}

kernel void gemv_m8_q4l(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], device const half* qb_ [[buffer(6)]],
    constant uint& M [[buffer(7)]], constant uint& accum [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*2u) + sgid*2u; if (out_row >= N) { return; }
    uint nblk = K/32u; uint rb = K/2u;
    uint mm = min(M, 8u);
    device const uchar* wr = w4 + (ulong)out_row*(ulong)rb + (ulong)lane*8u;
    device const half* qa = qa_ + (ulong)out_row*(ulong)nblk + lane/2u;
    device const half* qb = qb_ + (ulong)out_row*(ulong)nblk + lane/2u;
    float res[2][8];
    for (uint r = 0u; r < 2u; r++) { for (uint m = 0u; m < 8u; m++) { res[r][m] = 0.0; } }

    uint kfull = (K/512u)*512u;
    for (uint k = 0u; k < kfull + 512u; k += 512u) {
        bool tail = (k >= kfull);
        if (tail && !(lane*16u < K - kfull)) { break; }
        // weights for this lane's 4 rows -> registers, read once for all M tokens
        uint16_t wv[2][4]; float A[2], B[2];
        for (uint r = 0u; r < 2u; r++) {
            device const uint16_t* ws = (device const uint16_t*)(wr + (ulong)r*(ulong)rb);
            wv[r][0] = ws[0]; wv[r][1] = ws[1]; wv[r][2] = ws[2]; wv[r][3] = ws[3];
            A[r] = float(qa[(ulong)r*(ulong)nblk]);
            B[r] = float(qb[(ulong)r*(ulong)nblk]);
        }
        for (uint m = 0u; m < mm; m++) {
            device const float* xr = x + (ulong)m*(ulong)K + k + lane*16u;
            float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
            for (uint r = 0u; r < 2u; r++) {
                float acc = 0.0;
                for (uint i = 0u; i < 4u; i++) {
                    uint16_t v = wv[r][i];
                    acc += xt[4u*i]*float(v & 0x000fu) + xt[4u*i+1u]*float(v & 0x00f0u)
                         + xt[4u*i+2u]*float(v & 0x0f00u) + xt[4u*i+3u]*float(v & 0xf000u);
                }
                res[r][m] += A[r]*acc + B[r]*xsum;
            }
        }
        wr += 256u; qa += 16u; qb += 16u;
    }
    for (uint r = 0u; r < 2u; r++) { for (uint m = 0u; m < mm; m++) { res[r][m] = simd_sum(res[r][m]); } }
    if (lane == 0u) {
        for (uint r = 0u; r < 2u; r++) {
            uint o = out_row + r;
            if (o >= N) { break; }
            for (uint m = 0u; m < mm; m++) {
                if (accum != 0u) { y[(ulong)m*(ulong)N + o] += res[r][m]; }
                else             { y[(ulong)m*(ulong)N + o]  = res[r][m]; }
            }
        }
    }
}


kernel void gemv_m_q4l(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], device const half* qb_ [[buffer(6)]],
    constant uint& M [[buffer(7)]], constant uint& ACC [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nblk = K/32u;
    device const uchar* wr = w4 + (ulong)n*(ulong)(K/2u);
    device const half* aa = qa_ + (ulong)n*(ulong)nblk;
    device const half* bb = qb_ + (ulong)n*(ulong)nblk;
    // M is walked in groups of 8: only 8 accumulators fit in registers, and the
    // caller keeps the prefill chunk at 8 so weights are still read once per group.
    // The loop keeps a larger M correct rather than writing past the array, which
    // corrupts output without crashing.
    for (uint m0 = 0u; m0 < M; m0 += 8u) {
        uint mm = min(8u, M - m0);
        float p[8], xs[8];
        for (uint m = 0u; m < mm; m++) { p[m] = 0.0; xs[m] = 0.0; }
        for (uint b = lane; b < nblk; b += 32u) {
            device const uchar4* q = (device const uchar4*)(wr + b*16u);
            float A = float(aa[b]), B = float(bb[b]);
            float acc[8];
            for (uint m = 0u; m < mm; m++) { acc[m] = 0.0; }
            // One uchar4 covers 8 weights. Unpack it into two float4 registers and
            // reuse across the M rows — materialising all 32 weights in an array
            // spills to memory and costs more than the batching saves.
            for (uint i = 0u; i < 4u; i++) {
                uchar4 c = q[i];
                float4 lo = float4(c & (uchar4)0x0F);   // weights 8i+0,2,4,6
                float4 hi = float4(c >> (uchar4)4);     // weights 8i+1,3,5,7
                for (uint m = 0u; m < mm; m++) {
                    device const float4* xm = (device const float4*)(x + (ulong)(m0+m)*(ulong)K + b*32u);
                    float4 xa = xm[2u*i], xb = xm[2u*i + 1u];
                    acc[m] += lo.x*xa.x + hi.x*xa.y + lo.y*xa.z + hi.y*xa.w
                            + lo.z*xb.x + hi.z*xb.y + lo.w*xb.z + hi.w*xb.w;
                    xs[m]  += xa.x+xa.y+xa.z+xa.w + xb.x+xb.y+xb.z+xb.w;
                }
            }
            for (uint m = 0u; m < mm; m++) { p[m] += A*acc[m]; }
            // B applies to the block's activation sum; fold it per block.
            for (uint m = 0u; m < mm; m++) { p[m] += B*(xs[m]); xs[m] = 0.0; }
        }
        for (uint m = 0u; m < mm; m++) {
            float r = simd_sum(p[m]);
            if (lane == 0u) {
                if (ACC != 0u) { y[(ulong)(m0+m)*(ulong)N + n] += r; } else { y[(ulong)(m0+m)*(ulong)N + n] = r; }
            }
        }
    }
}

// Relayout Q4_K -> Q4L, once at load, on the GPU. One lane per 32-weight sub-block.
// Repacks the nibbles too: Q4_K stores a sub-block as the low (or high) nibbles of 32
// consecutive bytes; the fast kernel wants weight pairs sharing a byte.
kernel void relayout_q4k_q4l(device const uchar* w [[buffer(0)]], device uchar* nib [[buffer(1)]],
    device half* qa [[buffer(2)]], device half* qb [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nblk = K/32u; uint nsb = K/256u;
    device const uchar* wr = w + (ulong)n*(ulong)nsb*144u;
    device uchar* nr = nib + (ulong)n*(ulong)(K/2u);
    for (uint b = lane; b < nblk; b += 32u) {
        uint sb = b >> 3, j = b & 7u, g = j >> 1, hi = j & 1u;
        device const uchar* blk = wr + (ulong)sb*144u;
        float d = float(*(device const half*)blk);
        float dm = float(*(device const half*)(blk+2));
        device const uchar* sc = blk + 4u;
        uint s_, m_;
        if (j < 4u) { s_ = sc[j] & 63u; m_ = sc[j+4u] & 63u; }
        else { s_ = (sc[j+4u] & 0x0Fu) | ((sc[j-4u] >> 6) << 4); m_ = (sc[j+4u] >> 4) | ((sc[j] >> 6) << 4); }
        qa[(ulong)n*(ulong)nblk + b] = half(d * float(s_));
        qb[(ulong)n*(ulong)nblk + b] = half(-dm * float(m_));
        device const uchar* qq = blk + 16u + g*32u;
        for (uint i = 0u; i < 16u; i++) {
            uint n0 = hi ? (uint(qq[2u*i]) >> 4) : (uint(qq[2u*i]) & 0x0Fu);
            uint n1 = hi ? (uint(qq[2u*i+1u]) >> 4) : (uint(qq[2u*i+1u]) & 0x0Fu);
            nr[b*16u + i] = uchar(n0 | (n1 << 4));
        }
    }
}

// ---- Measured negatives around the tuned Q4 matvec -------------------------
//
// Three hypotheses for why the 4-bit decode matvec reads weights at a lower GB/s
// than the f16 one. All three are wired behind OJAS_Q4F (see
// decoder/dispatch.rs::q4f_override) and were measured on qwen35 (d=1024, ffn=3584)
// against the shipped `gemv_q4_fast`; all three lost or did nothing, and are kept
// reachable so the claims stay checkable. Run-to-run noise was ~19%, so only large
// effects are meaningful.
//
//   1. 8-byte vector weight load (gemv_q4_fast_v4, OJAS_Q4F=vec4:64). `Q4FAST_DOT`
//      reads its 8 weight bytes as four scalar uint16_t loads where `Q4L_DOT` reads
//      the same 8 bytes as one ushort4. Null here: ffn_down 222 vs 226 GB/s,
//      ssm_proj 246 vs 244, o_proj 208 vs 197 — the Metal compiler already merges
//      the four contiguous loads. The default stays as the prelude has it.
//
//   2. 8 rows per simdgroup (gemv_q4_fast8, OJAS_Q4F=fast8:64). A Q4 matvec reads
//      1.78 bytes of f32 activation per byte of weight where an f16 one reads 0.5,
//      so halving activation traffic per output row should matter. Worse, and not
//      marginally: 224 -> 179 GB/s over the whole GEMV set. Eight live accumulators
//      cost more occupancy than the saved traffic buys.
//
//   3. Activation traffic removed entirely (gemv_q4_xabl, OJAS_Q4F=xabl:64). Null:
//      223 -> 227 GB/s, inside noise. Activation traffic is not the ceiling, which
//      retires hypothesis 2's premise.
//
// What does move the number: the same kernels timed in the concurrent encoder the
// decoder actually opens, rather than a serial one, reach 327 GB/s — the f16 rate. A
// single Q4 dispatch moves 3.6x fewer bytes than the f16 one for the same launch, so
// it never reaches steady state alone. The lever is dispatch granularity and
// overlap, not the inner loop.

// 8 rows per simdgroup: half the activation traffic per output row, twice the live
// accumulators. Measured worse — see (2) above.
#define GEMV_Q4_FAST8 \
    uint out_row = tgid*(ts/32u*8u) + sgid*8u; if (out_row >= N) { return; } \
    uint nblk = K/32u; uint rb = K/2u; \
    device const uchar* wr = w4 + (ulong)out_row*(ulong)rb + (ulong)lane*8u; \
    device const half* sc = scale + (ulong)out_row*(ulong)nblk + lane/2u; \
    device const float* xr = x + lane*16u; \
    float res[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; \
    uint kfull = (K/512u)*512u; \
    for (uint k = 0u; k < kfull; k += 512u) { \
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum) \
        for (uint r = 0u; r < 8u; r++) Q4FAST_DOT(wr, r, xt, xsum, sc, res) \
        wr += 256u; sc += 16u; xr += 512u; \
    } \
    if (lane*16u < K - kfull) { \
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum) \
        for (uint r = 0u; r < 8u; r++) Q4FAST_DOT(wr, r, xt, xsum, sc, res) \
    } \
    for (uint r = 0u; r < 8u; r++) res[r] = simd_sum(res[r]);

kernel void gemv_q4_fast8(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q4_FAST8
    if (lane == 0u) { for (uint r = 0u; r < 8u; r++) if (out_row+r < N) y[out_row+r] = res[r]; }
}

kernel void gemv_q4_fast8_accum(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q4_FAST8
    if (lane == 0u) { for (uint r = 0u; r < 8u; r++) if (out_row+r < N) y[out_row+r] += res[r]; }
}

// One ushort4 weight load instead of four scalar uint16_t ones. The address is
// 8-byte aligned by construction at every call site (w4 + out_row*(K/2) + lane*8,
// with K % 32 == 0 enforced for every Q4 tensor at load), which the scalar form
// cannot tell the compiler. Measured null — see (1) above.
#define Q4FAST_DOT_V4(wr_, r_, xt_, xsum_, sc_, res_) { \
    ushort4 wv4 = *(device const ushort4*)((wr_) + (r_)*rb); \
    float s = float((sc_)[(r_)*nblk]); float acc = 0.0; \
    for (uint i = 0u; i < 4u; i++) { uint16_t wv = wv4[i]; \
        acc += (xt_)[4u*i]*float(wv & 0x000fu) + (xt_)[4u*i+1u]*float(wv & 0x00f0u) \
             + (xt_)[4u*i+2u]*float(wv & 0x0f00u) + (xt_)[4u*i+3u]*float(wv & 0xf000u); } \
    (res_)[r_] += s*acc - 8.0*s*(xsum_); }

kernel void gemv_q4_fast_v4(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    uint nblk = K/32u; uint rb = K/2u;
    device const uchar* wr = w4 + (ulong)out_row*(ulong)rb + (ulong)lane*8u;
    device const half* sc = scale + (ulong)out_row*(ulong)nblk + lane/2u;
    device const float* xr = x + lane*16u;
    float res[4] = {0.0, 0.0, 0.0, 0.0};
    uint kfull = (K/512u)*512u;
    for (uint k = 0u; k < kfull; k += 512u) {
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
        for (uint r = 0u; r < 4u; r++) Q4FAST_DOT_V4(wr, r, xt, xsum, sc, res)
        wr += 256u; sc += 16u; xr += 512u;
    }
    if (lane*16u < K - kfull) {
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
        for (uint r = 0u; r < 4u; r++) Q4FAST_DOT_V4(wr, r, xt, xsum, sc, res)
    }
    for (uint r = 0u; r < 4u; r++) res[r] = simd_sum(res[r]);
    if (lane == 0u) { for (uint r = 0u; r < 4u; r++) y[out_row+r] = res[r]; }
}

// Diagnostic only, numerically wrong by construction. Identical to gemv_q4_fast
// except the activation pointer never advances, so every iteration re-reads the same
// 64 bytes of x (L1-hot) while the weight stream is untouched, isolating whether the
// ceiling is activation traffic. Measured null — see (3) above. Reachable only via
// OJAS_Q4F=xabl:<threads>, which no shipping path sets.
kernel void gemv_q4_xabl(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    uint nblk = K/32u; uint rb = K/2u;
    device const uchar* wr = w4 + (ulong)out_row*(ulong)rb + (ulong)lane*8u;
    device const half* sc = scale + (ulong)out_row*(ulong)nblk + lane/2u;
    device const float* xr = x + lane*16u;
    float res[4] = {0.0, 0.0, 0.0, 0.0};
    uint kfull = (K/512u)*512u;
    for (uint k = 0u; k < kfull; k += 512u) {
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
        for (uint r = 0u; r < 4u; r++) Q4FAST_DOT(wr, r, xt, xsum, sc, res)
        wr += 256u; sc += 16u;               // xr deliberately NOT advanced
    }
    if (lane*16u < K - kfull) {
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
        for (uint r = 0u; r < 4u; r++) Q4FAST_DOT(wr, r, xt, xsum, sc, res)
    }
    for (uint r = 0u; r < 4u; r++) res[r] = simd_sum(res[r]);
    if (lane == 0u) { for (uint r = 0u; r < 4u; r++) y[out_row+r] = res[r]; }
}

kernel void gemv_q4_fast(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q4_FAST
    if (lane == 0u) { for (uint r = 0u; r < 4u; r++) y[out_row+r] = res[r]; }
}

kernel void gemv_q4_fast_accum(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q4_FAST
    if (lane == 0u) { for (uint r = 0u; r < 4u; r++) y[out_row+r] += res[r]; }
}

// Dispatching non-multiple-of-8 N.
//
// `gemv_q4_fast` handles 4 rows per simdgroup and stores all four unconditionally,
// so the dispatcher gated it on N % 8 == 0 and sent everything else to `gemv_q4`.
// The only shape that fails that gate is the lm_head — at K=1024, N=65425 that is
// 12% of the token's weight bytes in one dispatch, alone in its barrier region, so
// its isolated rate is its real rate. Measured at prec=2: `gemv_q4` 237 GB/s against
// `gemv_q4_fast`'s 381-387 on the FFN shapes in the same run. The autotuner's
// candidate set was `gemv_q4` and `gemv_q4_ksplit`, so the faster kernel was never
// eligible for comparison.
//
// A guarded-store variant is wrong: guarding the store leaves the loads overrunning,
// since `Q4FAST_DOT` reads row `out_row + r` for all four r before anything is
// stored — an out-of-bounds read of up to 3 rows (~1.5 KB) past the weight and scale
// buffers. `mm()` instead dispatches this kernel over the 4-row-aligned prefix
// (N4 = N/4*4, passed as its N so its own guard does the work) and sweeps the <= 3
// leftover rows with `gemv_q4` at a buffer offset, where every access is in bounds by
// construction. See the N % 8 != 0 branch in decoder/dispatch.rs.

kernel void gemv_q4_ksplit(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q4_KSPLIT
    if (w0) { y[n] = p; }
}

kernel void gemv_q4_ksplit_accum(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_Q4_KSPLIT
    if (w0) { y[n] += p; }
}

// Fused Q/K/V for Q4 (with bias). Row index picks q/k/v.
kernel void qkv_q4(device const float* x [[buffer(0)]],
    device const uchar* wq [[buffer(1)]], device const uchar* wk [[buffer(2)]], device const uchar* wv [[buffer(3)]],
    device float* yq [[buffer(4)]], device float* yk [[buffer(5)]], device float* yv [[buffer(6)]],
    constant uint& K [[buffer(7)]], constant uint& Nq [[buffer(8)]], constant uint& Nkv [[buffer(9)]],
    device const half* sq [[buffer(10)]], device const half* sk [[buffer(11)]], device const half* sv [[buffer(12)]],
    device const float* bq [[buffer(13)]], device const float* bk [[buffer(14)]], device const float* bv [[buffer(15)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid; uint total = Nq + 2u*Nkv; if (n >= total) { return; }
    device const uchar* w4; device const half* scale;
    device const float* bias; device float* y; uint r;
    if (n < Nq)          { w4=wq; scale=sq; bias=bq; y=yq; r=n; }
    else if (n < Nq+Nkv) { w4=wk; scale=sk; bias=bk; y=yk; r=n-Nq; }
    else                 { w4=wv; scale=sv; bias=bv; y=yv; r=n-Nq-Nkv; }
    uint nblk = K/32u;
    device const uchar* row = w4 + (ulong)r*(ulong)(K/2u);
    device const half* rsc = scale + (ulong)r*(ulong)nblk;
    float p = 0.0;
    for (uint b = lane; b < nblk; b += 32u) {
        float axq = 0.0, sx = 0.0;
        Q4BLK(row, b, x, axq, sx);
        p += float(rsc[b])*(axq - 8.0*sx);
    }
    p = simd_sum(p);
    if (lane == 0u) { y[r] = p + bias[r]; }
}

// K-split fused Q/K/V for Q4: one output row per threadgroup, simdgroups split the
// K-blocks (fixes short-loop overhead on the small-N qkv, K=2048).
kernel void qkv_q4_ksplit(device const float* x [[buffer(0)]],
    device const uchar* wq [[buffer(1)]], device const uchar* wk [[buffer(2)]], device const uchar* wv [[buffer(3)]],
    device float* yq [[buffer(4)]], device float* yk [[buffer(5)]], device float* yv [[buffer(6)]],
    constant uint& K [[buffer(7)]], constant uint& Nq [[buffer(8)]], constant uint& Nkv [[buffer(9)]],
    device const half* sq [[buffer(10)]], device const half* sk [[buffer(11)]], device const half* sv [[buffer(12)]],
    device const float* bq [[buffer(13)]], device const float* bk [[buffer(14)]], device const float* bv [[buffer(15)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid; uint total = Nq + 2u*Nkv; if (n >= total) { return; }
    device const uchar* w4; device const half* scale;
    device const float* bias; device float* y; uint r;
    if (n < Nq)          { w4=wq; scale=sq; bias=bq; y=yq; r=n; }
    else if (n < Nq+Nkv) { w4=wk; scale=sk; bias=bk; y=yk; r=n-Nq; }
    else                 { w4=wv; scale=sv; bias=bv; y=yv; r=n-Nq-Nkv; }
    uint nsg = ts/32u; uint nblk = K/32u; uint gtid = sgid*32u + lane;
    device const uchar* row = w4 + (ulong)r*(ulong)(K/2u);
    device const half* rsc = scale + (ulong)r*(ulong)nblk;
    float pp = 0.0;
    for (uint b = gtid; b < nblk; b += nsg*32u) {
        float axq = 0.0, sx = 0.0;
        Q4BLK(row, b, x, axq, sx);
        pp += float(rsc[b])*(axq - 8.0*sx);
    }
    pp = simd_sum(pp);
    threadgroup float part[32];
    if (lane == 0u) { part[sgid] = pp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) { float v = (lane < nsg) ? part[lane] : 0.0; float p = simd_sum(v);
        if (lane == 0u) { y[r] = p + bias[r]; } }
}

// Fused SwiGLU gate/up for Q4, qmv_fast style: 4 output rows/simdgroup (8 rows/tg),
// uint16 vectorized loads, mask-in-place + pre-divided x (no shifts). x loaded once and
// reused across all 8 gate/up dots. Requires N%8==0 (all FFN dims are).
kernel void ffn_gu_q4(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    device const half* sg [[buffer(6)]], device const half* su [[buffer(7)]], constant uint& act [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u*4u) + sgid*4u; if (out_row >= N) { return; }
    uint nblk = K/32u; uint rb = K/2u;
    device const uchar* gr = wg + (ulong)out_row*(ulong)rb + (ulong)lane*8u;
    device const uchar* ur = wu + (ulong)out_row*(ulong)rb + (ulong)lane*8u;
    device const half* scg = sg + (ulong)out_row*(ulong)nblk + lane/2u;
    device const half* scu = su + (ulong)out_row*(ulong)nblk + lane/2u;
    device const float* xr = x + lane*16u;
    float rg[4] = {0.0,0.0,0.0,0.0}, ru[4] = {0.0,0.0,0.0,0.0};
    uint kfull = (K/512u)*512u;
    for (uint k = 0u; k < kfull; k += 512u) {
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
        for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(gr, r, xt, xsum, scg, rg) Q4FAST_DOT(ur, r, xt, xsum, scu, ru) }
        gr += 256u; ur += 256u; scg += 16u; scu += 16u; xr += 512u;
    }
    if (lane*16u < K - kfull) {
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
        for (uint r = 0u; r < 4u; r++) { Q4FAST_DOT(gr, r, xt, xsum, scg, rg) Q4FAST_DOT(ur, r, xt, xsum, scu, ru) }
    }
    for (uint r = 0u; r < 4u; r++) { rg[r] = simd_sum(rg[r]); ru[r] = simd_sum(ru[r]); }
    if (lane == 0u) { for (uint r = 0u; r < 4u; r++) out[out_row+r] = ffn_act(rg[r], act)*ru[r]; }
}

kernel void gemv_q8_r4(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    GEMV_Q8_R4
    if (lane == 0u) { y[row0+0u]=p0; y[row0+1u]=p1; y[row0+2u]=p2; y[row0+3u]=p3; }
}

kernel void gemv_q8_r4_bias(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]], device const float* bias [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    GEMV_Q8_R4
    if (lane == 0u) { y[row0+0u]=p0+bias[row0+0u]; y[row0+1u]=p1+bias[row0+1u];
                      y[row0+2u]=p2+bias[row0+2u]; y[row0+3u]=p3+bias[row0+3u]; }
}

kernel void gemv_q8_r4_accum(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    GEMV_Q8_R4
    if (lane == 0u) { y[row0+0u]+=p0; y[row0+1u]+=p1; y[row0+2u]+=p2; y[row0+3u]+=p3; }
}

// Fused Q/K/V projection: all three read the same input h, so one dispatch instead
// of three (saves 2 dispatch launches/layer × n_layers). The grid covers Nq+2*Nkv
// rows; each simdgroup's row index picks which of q/k/v it is and writes to that
// separate output buffer, leaving rope/store_kv downstream unchanged.
// Q4L twin of qkv_q8: one fused dispatch for the three attention projections,
// reading Q4_K's own values instead of a Q8 requantization of them.
//
// Without it attn_q/k/v fall off the native path (the fused Q8 kernel indexes w8
// directly) and are requantized to Q8 at 1.0625 B/weight against Q4L's 0.5625 — on
// qwen2-3B, 200 MB/token instead of 106, which comes straight off a bandwidth-bound
// decode rate. Three separate gemv_q4l dispatches would fix the bytes but add two
// dispatches per layer at ~5 us of sync each.
//
// One output row per simdgroup, lanes split K — qkv_q8's shape, not the
// 4-rows-per-simdgroup shape of GEMV_Q4L_FAST, because rows here straddle three
// matrices with different N and a 4-row group would cross those boundaries.
kernel void qkv_q4l(device const float* x [[buffer(0)]],
    device const uchar* wq [[buffer(1)]], device const uchar* wk [[buffer(2)]], device const uchar* wv [[buffer(3)]],
    device float* yq [[buffer(4)]], device float* yk [[buffer(5)]], device float* yv [[buffer(6)]],
    constant uint& K [[buffer(7)]], constant uint& Nq [[buffer(8)]], constant uint& Nkv [[buffer(9)]],
    device const half* aq [[buffer(10)]], device const half* bq_ [[buffer(11)]],
    device const half* ak [[buffer(12)]], device const half* bk_ [[buffer(13)]],
    device const half* av [[buffer(14)]], device const half* bv_ [[buffer(15)]],
    device const float* biq [[buffer(16)]], device const float* bik [[buffer(17)]], device const float* biv [[buffer(18)]],
    constant uint& has_bias [[buffer(19)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    // One output row per simdgroup, lanes split K — qkv_q8's shape. A
    // 4-rows-per-simdgroup variant (GEMV_Q4L_FAST's shape, with row groups indexed
    // within each matrix so a group never straddles q/k/v) failed decode_gate on its
    // row math and was no faster (qkv 0.612 -> 0.592 ms, whole floor 7.306 -> 7.483).
    uint nsg = ts/32u;
    uint n = tgid*nsg + sgid;
    uint total = Nq + 2u*Nkv;
    if (n >= total) { return; }
    device const uchar* w; device const half* aa; device const half* bb;
    device const float* bias; device float* y; uint r;
    if (n < Nq)          { w=wq; aa=aq; bb=bq_; bias=biq; y=yq; r=n; }
    else if (n < Nq+Nkv) { w=wk; aa=ak; bb=bk_; bias=bik; y=yk; r=n-Nq; }
    else                 { w=wv; aa=av; bb=bv_; bias=biv; y=yv; r=n-Nq-Nkv; }

    uint nblk = K/32u; uint rb = K/2u;
    device const uchar* wr = w + (ulong)r*(ulong)rb + (ulong)lane*8u;
    device const half* qa = aa + (ulong)r*(ulong)nblk + lane/2u;
    device const half* qb = bb + (ulong)r*(ulong)nblk + lane/2u;
    device const float* xr = x + lane*16u;
    float res[1] = {0.0};
    uint kfull = (K/512u)*512u;
    for (uint k = 0u; k < kfull; k += 512u) {
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
        Q4L_DOT(wr, 0u, xt, xsum, qa, qb, res)
        wr += 256u; qa += 16u; qb += 16u; xr += 512u;
    }
    if (lane*16u < K - kfull) {
        float xt[16]; Q4FAST_LOADX(xr, xt, xsum)
        Q4L_DOT(wr, 0u, xt, xsum, qa, qb, res)
    }
    res[0] = simd_sum(res[0]);
    if (lane == 0u) { y[r] = res[0] + ((has_bias != 0u) ? bias[r] : 0.0); }
}
kernel void qkv_q8(device const float* x [[buffer(0)]],
    device const char* wq [[buffer(1)]], device const char* wk [[buffer(2)]], device const char* wv [[buffer(3)]],
    device float* yq [[buffer(4)]], device float* yk [[buffer(5)]], device float* yv [[buffer(6)]],
    constant uint& K [[buffer(7)]], constant uint& Nq [[buffer(8)]], constant uint& Nkv [[buffer(9)]],
    device const float* sq [[buffer(10)]], device const float* sk [[buffer(11)]], device const float* sv [[buffer(12)]],
    device const float* bq [[buffer(13)]], device const float* bk [[buffer(14)]], device const float* bv [[buffer(15)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid;
    uint total = Nq + 2u*Nkv;
    if (n >= total) { return; }
    // select region -> weight/scale/bias/output + local row
    device const char* w; device const float* scale; device const float* bias;
    device float* y; uint r;
    if (n < Nq)            { w=wq; scale=sq; bias=bq; y=yq; r=n; }
    else if (n < Nq+Nkv)   { w=wk; scale=sk; bias=bk; y=yk; r=n-Nq; }
    else                   { w=wv; scale=sv; bias=bv; y=yv; r=n-Nq-Nkv; }
    device const char4* row = (device const char4*)(w + (ulong)r*(ulong)K);
    device const float4* xv = (device const float4*)x;
    uint K4 = K/4u; float p = 0.0;
    for (uint k = lane; k < K4; k += 32u) { p += dot(float4(row[k]), xv[k]); }
    p = simd_sum(p) * scale[r];
    if (lane == 0u) { y[r] = p + bias[r]; }
}

// Fused Q/K/V for f16 weights (half, no per-row scale) — the f16 analog of qkv_q8,
// so the f16 path also does QKV in ONE dispatch (was 3) → parity with the q8 path.
kernel void qkv_f16(device const float* x [[buffer(0)]],
    device const half* wq [[buffer(1)]], device const half* wk [[buffer(2)]], device const half* wv [[buffer(3)]],
    device float* yq [[buffer(4)]], device float* yk [[buffer(5)]], device float* yv [[buffer(6)]],
    constant uint& K [[buffer(7)]], constant uint& Nq [[buffer(8)]], constant uint& Nkv [[buffer(9)]],
    device const float* bq [[buffer(13)]], device const float* bk [[buffer(14)]], device const float* bv [[buffer(15)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid; uint total = Nq + 2u*Nkv; if (n >= total) { return; }
    device const half* w; device const float* bias; device float* y; uint r;
    if (n < Nq)            { w=wq; bias=bq; y=yq; r=n; }
    else if (n < Nq+Nkv)   { w=wk; bias=bk; y=yk; r=n-Nq; }
    else                   { w=wv; bias=bv; y=yv; r=n-Nq-Nkv; }
    device const half4* row = (device const half4*)(w + (ulong)r*(ulong)K);
    device const float4* xv = (device const float4*)x;
    uint K4 = K/4u; float p = 0.0;
    for (uint k = lane; k < K4; k += 32u) { p += dot(float4(row[k]), xv[k]); }
    p = simd_sum(p);
    if (lane == 0u) { y[r] = p + bias[r]; }
}



// Fused SwiGLU first half: out[n] = silu(Wg·x)·(Wu·x) in one dispatch (replaces
// gate GEMV + up GEMV + swiglu = 3 dispatches). Reads x once per row.
kernel void ffn_gu_f16(device const float* x [[buffer(0)]], device const half* wg [[buffer(1)]],
    device const half* wu [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& act [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid; if (n >= N) { return; }
    device const half4* rg = (device const half4*)(wg + (ulong)n*(ulong)K);
    device const half4* ru = (device const half4*)(wu + (ulong)n*(ulong)K);
    device const float4* xv = (device const float4*)x; uint K4 = K/4u;
    float pg = 0.0, pu = 0.0;
    for (uint k = lane; k < K4; k += 32u) { float4 xf = xv[k]; pg += dot(float4(rg[k]), xf); pu += dot(float4(ru[k]), xf); }
    pg = simd_sum(pg); pu = simd_sum(pu);
    if (lane == 0u) { out[n] = ffn_act(pg, act)*pu; }
}

kernel void ffn_gu_q8(device const float* x [[buffer(0)]], device const char* wg [[buffer(1)]],
    device const char* wu [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    device const float* sg [[buffer(6)]], device const float* su [[buffer(7)]], constant uint& act [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    device const char4* rg = (device const char4*)(wg + (ulong)n*(ulong)K);
    device const char4* ru = (device const char4*)(wu + (ulong)n*(ulong)K);
    device const float4* xv = (device const float4*)x; uint K4 = K/4u;
    float pg = 0.0, pu = 0.0;
    for (uint k = lane; k < K4; k += 32u) { float4 xf = xv[k]; pg += dot(float4(rg[k]), xf); pu += dot(float4(ru[k]), xf); }
    pg = simd_sum(pg)*sg[n]; pu = simd_sum(pu)*su[n];
    if (lane == 0u) { out[n] = ffn_act(pg, act)*pu; }
}

kernel void gemv_m_q8(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]], constant uint& M [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid; if (n >= N) { return; }
    device const char4* row = (device const char4*)(w + (ulong)n*(ulong)K);
    uint K4 = K/4u;
    float p[8]; for (uint m=0u;m<M;m++) p[m]=0.0;
    for (uint k = lane; k < K4; k += 32u) {
        float4 w4 = float4(row[k]);
        for (uint m=0u;m<M;m++) { device const float4* xm=(device const float4*)(x + (ulong)m*(ulong)K); p[m]+=dot(w4, xm[k]); }
    }
    float s = scale[n];
    for (uint m=0u;m<M;m++) { float r = simd_sum(p[m])*s; if (lane==0u) y[m*N+n]=r; }
}

kernel void gemv_m_q8_bias(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]], constant uint& M [[buffer(6)]], device const float* bias [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid; if (n >= N) { return; }
    device const char4* row = (device const char4*)(w + (ulong)n*(ulong)K); uint K4 = K/4u;
    float p[8]; for (uint m=0u;m<M;m++) p[m]=0.0;
    for (uint k = lane; k < K4; k += 32u) { float4 w4=float4(row[k]);
        for (uint m=0u;m<M;m++){ device const float4* xm=(device const float4*)(x+(ulong)m*(ulong)K); p[m]+=dot(w4,xm[k]); } }
    float s=scale[n], b=bias[n];
    for (uint m=0u;m<M;m++){ float r=simd_sum(p[m])*s+b; if(lane==0u) y[m*N+n]=r; }
}

kernel void gemv_m_q8_accum(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]], constant uint& M [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid; if (n >= N) { return; }
    device const char4* row = (device const char4*)(w + (ulong)n*(ulong)K); uint K4 = K/4u;
    float p[8]; for (uint m=0u;m<M;m++) p[m]=0.0;
    for (uint k = lane; k < K4; k += 32u) { float4 w4=float4(row[k]);
        for (uint m=0u;m<M;m++){ device const float4* xm=(device const float4*)(x+(ulong)m*(ulong)K); p[m]+=dot(w4,xm[k]); } }
    float s=scale[n];
    for (uint m=0u;m<M;m++){ float r=simd_sum(p[m])*s; if(lane==0u) y[m*N+n]+=r; }
}

// Fused Q/K/V projection for M tokens in one dispatch (multi-token analog of
// qkv_q8). Row index picks q/k/v weight+scale+bias+output; weight row loaded once
// and reused across all M tokens. Outputs are separate [M,Nout] buffers.
kernel void qkv_m_q8(device const float* x [[buffer(0)]],
    device const char* wq [[buffer(1)]], device const char* wk [[buffer(2)]], device const char* wv [[buffer(3)]],
    device float* yq [[buffer(4)]], device float* yk [[buffer(5)]], device float* yv [[buffer(6)]],
    constant uint& K [[buffer(7)]], constant uint& Nq [[buffer(8)]], constant uint& Nkv [[buffer(9)]],
    device const float* sq [[buffer(10)]], device const float* sk [[buffer(11)]], device const float* sv [[buffer(12)]],
    device const float* bq [[buffer(13)]], device const float* bk [[buffer(14)]], device const float* bv [[buffer(15)]],
    constant uint& M [[buffer(16)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid; uint total = Nq + 2u*Nkv; if (n >= total) { return; }
    device const char* w; device const float* scale; device const float* bias;
    device float* y; uint r; uint Nout;
    if (n < Nq)          { w=wq; scale=sq; bias=bq; y=yq; r=n;          Nout=Nq;  }
    else if (n < Nq+Nkv) { w=wk; scale=sk; bias=bk; y=yk; r=n-Nq;      Nout=Nkv; }
    else                 { w=wv; scale=sv; bias=bv; y=yv; r=n-Nq-Nkv;  Nout=Nkv; }
    device const char4* row = (device const char4*)(w + (ulong)r*(ulong)K); uint K4 = K/4u;
    float p[8]; for (uint m=0u;m<M;m++) p[m]=0.0;
    for (uint k = lane; k < K4; k += 32u) { float4 w4 = float4(row[k]);
        for (uint m=0u;m<M;m++){ device const float4* xm=(device const float4*)(x+(ulong)m*(ulong)K); p[m]+=dot(w4,xm[k]); } }
    float s=scale[r], b=bias[r];
    for (uint m=0u;m<M;m++){ float rr=simd_sum(p[m])*s+b; if(lane==0u) y[m*Nout+r]=rr; }
}

// Large-M (≤32) variants of the fused qkv + biased GEMV for gpt-oss chunked
// prefill. Identical weight-row-reuse design (each weight row loaded once,
// dotted against all M tokens) but a p[32] accumulator so a chunk can hold up to
// 32 tokens — 4× fewer MoE-expert-streaming passes than the M≤8 spec kernels.
// Register pressure is higher, but far outweighed by the drop in weight traffic.
kernel void qkv_mg_q8(device const float* x [[buffer(0)]],
    device const char* wq [[buffer(1)]], device const char* wk [[buffer(2)]], device const char* wv [[buffer(3)]],
    device float* yq [[buffer(4)]], device float* yk [[buffer(5)]], device float* yv [[buffer(6)]],
    constant uint& K [[buffer(7)]], constant uint& Nq [[buffer(8)]], constant uint& Nkv [[buffer(9)]],
    device const float* sq [[buffer(10)]], device const float* sk [[buffer(11)]], device const float* sv [[buffer(12)]],
    device const float* bq [[buffer(13)]], device const float* bk [[buffer(14)]], device const float* bv [[buffer(15)]],
    constant uint& M [[buffer(16)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid; uint total = Nq + 2u*Nkv; if (n >= total) { return; }
    device const char* w; device const float* scale; device const float* bias;
    device float* y; uint r; uint Nout;
    if (n < Nq)          { w=wq; scale=sq; bias=bq; y=yq; r=n;          Nout=Nq;  }
    else if (n < Nq+Nkv) { w=wk; scale=sk; bias=bk; y=yk; r=n-Nq;      Nout=Nkv; }
    else                 { w=wv; scale=sv; bias=bv; y=yv; r=n-Nq-Nkv;  Nout=Nkv; }
    device const char4* row = (device const char4*)(w + (ulong)r*(ulong)K); uint K4 = K/4u;
    float p[32]; for (uint m=0u;m<M;m++) p[m]=0.0;
    for (uint k = lane; k < K4; k += 32u) { float4 w4 = float4(row[k]);
        for (uint m=0u;m<M;m++){ device const float4* xm=(device const float4*)(x+(ulong)m*(ulong)K); p[m]+=dot(w4,xm[k]); } }
    float s=scale[r], b=bias[r];
    for (uint m=0u;m<M;m++){ float rr=simd_sum(p[m])*s+b; if(lane==0u) y[m*Nout+r]=rr; }
}

kernel void gemv_mg_q8_bias(device const float* x [[buffer(0)]], device const char* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]], constant uint& M [[buffer(6)]], device const float* bias [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid; if (n >= N) { return; }
    device const char4* row = (device const char4*)(w + (ulong)n*(ulong)K); uint K4 = K/4u;
    float p[32]; for (uint m=0u;m<M;m++) p[m]=0.0;
    for (uint k = lane; k < K4; k += 32u) { float4 w4=float4(row[k]);
        for (uint m=0u;m<M;m++){ device const float4* xm=(device const float4*)(x+(ulong)m*(ulong)K); p[m]+=dot(w4,xm[k]); } }
    float s=scale[n], b=bias[n];
    for (uint m=0u;m<M;m++){ float r=simd_sum(p[m])*s+b; if(lane==0u) y[m*N+n]=r; }
}

kernel void ffn_gu_m_q8(device const float* x [[buffer(0)]], device const char* wg [[buffer(1)]],
    device const char* wu [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    device const float* sg [[buffer(6)]], device const float* su [[buffer(7)]], constant uint& M [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tgid*8u + sgid; if (n >= N) { return; }
    device const char4* rg=(device const char4*)(wg+(ulong)n*(ulong)K);
    device const char4* ru=(device const char4*)(wu+(ulong)n*(ulong)K); uint K4=K/4u;
    float pg[8], pu[8]; for(uint m=0u;m<M;m++){pg[m]=0.0;pu[m]=0.0;}
    for (uint k=lane;k<K4;k+=32u){ float4 g4=float4(rg[k]), u4=float4(ru[k]);
        for(uint m=0u;m<M;m++){ device const float4* xm=(device const float4*)(x+(ulong)m*(ulong)K); float4 xf=xm[k]; pg[m]+=dot(g4,xf); pu[m]+=dot(u4,xf); } }
    float fg=sg[n], fu=su[n];
    for(uint m=0u;m<M;m++){ float g=simd_sum(pg[m])*fg, u=simd_sum(pu[m])*fu; if(lane==0u) out[m*N+n]=(g/(1.0+exp(-g)))*u; }
}

kernel void gemv_m_q4(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]], constant uint& M [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_M_Q4_BODY
    for (uint m=0u;m<M;m++) { float r = simd_sum(p[m]); if (lane==0u) { y[(ulong)m*(ulong)N+n] = r; } }
}

kernel void gemv_m_q4_accum(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]], constant uint& M [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_M_Q4_BODY
    for (uint m=0u;m<M;m++) { float r = simd_sum(p[m]); if (lane==0u) { y[(ulong)m*(ulong)N+n] += r; } }
}

// Simdgroup-matrix (MMA) GEMM for prefill chunks (kernel_mul_mm
// skeleton, adapted to our symmetric per-32-block q4). 128 threads / 4 simdgroups
// per threadgroup compute a 64(N-rows)×32(tokens) C tile over k-tiles of 32:
// A dequants to half into sa (8×8-block layout, k-major within block = A^T);
// B (x [M,K] f32) loads as half into sb (token-major blocks); then 8×8
// simdgroup_multiply_accumulate: mc(tok,row) += mb(tok,k) × ma(k,row).
// The full 32-token tile is stored unconditionally: tokens >= M compute zero (sb is
// zero-padded) and are written to y anyway, which accum adds as 0. The cooperative
// simdgroup_store cannot skip sub-tile rows, so the caller must size y to
// ceil(M/32)*32 rows for that store to stay in bounds; the MAXM-sized instance
// buffers satisfy this (MAXM=256, a multiple of 32; see decoder/load.rs). The
// invariant governs every gemm_mm_* MMA kernel below. Needs N%64==0, K%32==0.
kernel void gemm_mm_q4(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[64*32];
    threadgroup half sb[32*32];
    threadgroup float idm[64];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;                // token tile base (chunks > 32 tokens)
    uint lr0 = tiitg/2u;                        // A: row within tile (0..63)
    uint il0 = tiitg%2u;                        // A: which 16-elem half of the 32-k block
    device const uchar* arow = w4 + (ulong)(r0+lr0)*(ulong)(K/2u);
    device const half*  asc = scale + (ulong)(r0+lr0)*(ulong)(K/32u);
    uint lr1 = tiitg/4u;                        // B: token (0..31)
    uint sxb = tiitg%4u;                        // B: k-octet
    if (tiitg < 64u) { idm[tiitg] = (tiitg/8u == tiitg%8u) ? 1.0 : 0.0; }
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile: dequant 16 elems of row lr0 at k = lk + il0*16 + i
            half s = asc[lk/32u];
            device const uchar* bp = arow + lk/2u + il0*8u;
            uint sy = lr0/8u, lx = lr0%8u;
            for (short i = 0; i < 16; i++) {
                uchar byv = bp[i/2];
                half v = half(int((i & 1) ? (byv >> 4u) : (byv & 0xFu)) - 8) * s;
                uint sx = 2u*il0 + uint(i)/8u;
                sa[64u*(8u*sx+sy) + 8u*(uint(i)%8u) + lx] = v;
            }
        }
        {   // B tile: token lr1, 8 consecutive k at lk + 8*sxb (zero-pad tokens >= M).
            // Vectorized float4 loads → half4 stores (the reference loads B as 2x4 vectors).
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            device const float4* xr = (device const float4*)(x + (ulong)(t0+lr1)*(ulong)K + lk + 8u*sxb);
            bool ok = t0 + lr1 < M;
            threadgroup half4* dstb = (threadgroup half4*)(sb + 64u*ib + 8u*ly);
            dstb[0] = ok ? half4(xr[0]) : half4(0.0);
            dstb[1] = ok ? half4(xr[1]) : half4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const half* lsmb = sb + 2u*64u*(sgitg/2u);
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
    // C tile: this simdgroup owns rows r0 + 32*(sgitg&1) .. +32, tokens 16*(sgitg>>1) .. +16
    device float* C = y + (r0 + 32u*(sgitg & 1u)) + (ulong)(t0 + 16u*(sgitg >> 1u))*(ulong)N;
    if (accum != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 mid, mo;
        simdgroup_load(mid, idm, 8, 0, false);
        for (short i = 0; i < 8; i++) {
            device float* Ci = C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N;
            simdgroup_load(mo, Ci, N, 0, false);
            simdgroup_multiply_accumulate(mc[i], mc[i], mid, mo);   // mc = mc*I + old
            simdgroup_store(mc[i], Ci, N, 0, false);
        }
    } else {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false);
        }
    }
}

// Q8 twin of gemm_mm_q4 (mul_mm MMA GEMM). Weights are per-row symmetric int8 (one
// f32 scale per output row, constant over k), so the A tile just loads int8*scale
// into half — no nibble unpack, no per-block scale. Same 64(N)×32(tok) tile, weights
// loaded once and reused across all 32 tokens via simdgroup_multiply_accumulate.
// The batched-forward speed path, replacing the p[8] tiled GEMVs that re-read the
// weights once per 8-row tile. Needs N%64==0, K%32==0. accum!=0 => y += C.
// Q4L GEMM: 64(N) x 64(M) tile, 64 threads (2 simdgroups), 32 accumulators per
// simdgroup, serpentine token order.
//
// An earlier 64x64 tile kept 4 simdgroups (mc[16] each) and lost. This config
// (bm=64, bn=64, bk=16, wm=1, wn=2) halves the simdgroups instead: each of the 2
// owns all 64 rows x 32 tokens (mc[8][4]), so every staged fragment feeds 4-8 MMAs
// instead of 2-4 — double the arithmetic intensity per shared-memory load at the
// same barrier count. K-slab 16 (half a Q4L block; the scale pair still hoists).
kernel void gemm_mm_q4l_mlx(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[16*64];   // blocked 8x8: [k-octet 0..1][row-octet 0..7], within [k][row]
    threadgroup half sb[64*16];   // blocked 8x8: [k-octet 0..1][tok-octet 0..7], within [tok][k]
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*64u;
    uint nblk = K/32u;
    uint lr = tiitg;                       // 0..63: A row, and B token
    device const uchar* arow = w4 + (ulong)(r0+lr)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr)*(ulong)nblk;
    device const float* xrow = x + (ulong)(t0+lr)*(ulong)K;
    bool okb = t0 + lr < M;
    simdgroup_half8x8 ma[8];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[8][4];
    for (short r = 0; r < 8; r++) for (short t = 0; t < 4; t++) mc[r][t] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint lk = 0u; lk < K; lk += 16u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A: row lr, 16 k. One scale pair (16-aligned run inside one block).
            half A = arow_a[lk/32u];
            half B = arow_b[lk/32u];
            device const uchar4* bp4 = (device const uchar4*)(arow + lk/2u);
            uchar4 b0 = bp4[0], b1 = bp4[1];
            uint sy = lr/8u, lx = lr%8u;
            // block b = ko*8 + sy; within: [k][row] => 64*b + 8*(k%8) + lx
            threadgroup half* d0 = sa + 64u*sy + lx;          // ko=0 (k 0..7)
            threadgroup half* d1 = sa + 64u*(8u+sy) + lx;     // ko=1 (k 8..15)
            d0[ 0] = A*half(b0.x & 0x0Fu) + B;  d0[ 8] = A*half(b0.x >> 4) + B;
            d0[16] = A*half(b0.y & 0x0Fu) + B;  d0[24] = A*half(b0.y >> 4) + B;
            d0[32] = A*half(b0.z & 0x0Fu) + B;  d0[40] = A*half(b0.z >> 4) + B;
            d0[48] = A*half(b0.w & 0x0Fu) + B;  d0[56] = A*half(b0.w >> 4) + B;
            d1[ 0] = A*half(b1.x & 0x0Fu) + B;  d1[ 8] = A*half(b1.x >> 4) + B;
            d1[16] = A*half(b1.y & 0x0Fu) + B;  d1[24] = A*half(b1.y >> 4) + B;
            d1[32] = A*half(b1.z & 0x0Fu) + B;  d1[40] = A*half(b1.z >> 4) + B;
            d1[48] = A*half(b1.w & 0x0Fu) + B;  d1[56] = A*half(b1.w >> 4) + B;
        }
        {   // B: token lr, 16 k. within-block [tok][k]: 64*(ko*8 + tok/8) + 8*(tok%8) + k%8
            uint ty = lr/8u, tl = lr%8u;
            threadgroup half* d0 = sb + 64u*ty + 8u*tl;
            threadgroup half* d1 = sb + 64u*(8u+ty) + 8u*tl;
            device const float4* xr = (device const float4*)(xrow + lk);
            float4 v0 = okb ? xr[0] : float4(0.0);
            float4 v1 = okb ? xr[1] : float4(0.0);
            float4 v2 = okb ? xr[2] : float4(0.0);
            float4 v3 = okb ? xr[3] : float4(0.0);
            *(threadgroup half4*)(d0)      = half4(v0);
            *(threadgroup half4*)(d0 + 4)  = half4(v1);
            *(threadgroup half4*)(d1)      = half4(v2);
            *(threadgroup half4*)(d1 + 4)  = half4(v3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tokb = uint(sgitg)*4u;        // this sg's 4 token-octets (0..3 or 4..7)
        for (short ko = 0; ko < 2; ko++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short r = 0; r < 8; r++) { simdgroup_load(ma[r], sa + 64u*(uint(ko)*8u + uint(r)), 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short t = 0; t < 4; t++) { simdgroup_load(mb[t], sb + 64u*(uint(ko)*8u + tokb + uint(t)), 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            // Indices must stay fully static: a runtime-indexed simdgroup-matrix
            // array always spills, and computing t at runtime for the serpentine
            // order spilled all of mc — 37 ms/call against the base kernel's 2.6.
#pragma clang loop unroll(full)
            for (short r = 0; r < 8; r++) {
#pragma clang loop unroll(full)
                for (short t = 0; t < 4; t++) {
                    simdgroup_multiply_accumulate(mc[r][t], mb[t], ma[r], mc[r][t]);
                }
            }
        }
    }
    // store: mc[r][t] covers rows r0+8r, tokens t0 + sg*32 + 8t. accum via the
    // identity-MMA trick (mc = mc*I + C), same as gemm_mm_q4l.
    uint tokoff = t0 + uint(sgitg)*32u;
    if (accum != 0u) {
        threadgroup float idm[64];
        if (tiitg < 64u) { idm[tiitg] = (tiitg/8u == tiitg%8u) ? 1.0 : 0.0; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 mid, mo;
        simdgroup_load(mid, idm, 8, 0, false);
        for (short r = 0; r < 8; r++) {
            for (short t = 0; t < 4; t++) {
                device float* C = y + (r0 + uint(r)*8u) + (ulong)(tokoff + uint(t)*8u)*(ulong)N;
                simdgroup_load(mo, C, N, 0, false);
                simdgroup_multiply_accumulate(mc[r][t], mc[r][t], mid, mo);
                simdgroup_store(mc[r][t], C, N, 0, false);
            }
        }
    } else {
        for (short r = 0; r < 8; r++) {
            for (short t = 0; t < 4; t++) {
                device float* C = y + (r0 + uint(r)*8u) + (ulong)(tokoff + uint(t)*8u)*(ulong)N;
                simdgroup_store(mc[r][t], C, N, 0, false);
            }
        }
    }
}

kernel void gemm_mm_q4l(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[64*32];
    threadgroup half sb[32*32];
    threadgroup float idm[64];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;
    uint lr0 = tiitg/2u;                        // A: row within tile (0..63)
    uint il0 = tiitg%2u;                        // A: which 16-elem half of the 32-k block
    uint nblk = K/32u;
    device const uchar* arow = w4 + (ulong)(r0+lr0)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr0)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr0)*(ulong)nblk;
    uint lr1 = tiitg/4u;                        // B: token (0..31)
    uint sxb = tiitg%4u;                        // B: k-octet
    if (tiitg < 64u) { idm[tiitg] = (tiitg/8u == tiitg%8u) ? 1.0 : 0.0; }
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile: 16 Q4L weights of row lr0 at k = lk + il0*16 + i, dequantised
            // into the shared tile. k0 is 16-aligned and blocks are 32 wide, so all
            // 16 sit in one block and share its A/B pair — the per-lane header
            // re-reads that make the Q4_K layout slow at decode simply do not arise
            // here, because the tile is filled once and then MMA'd.
            //
            // This kernel is occupancy-limited by register pressure, so anything
            // adding live state to the inner loop loses: a 64x64 tile halving the
            // A-fill cost per output measured 193.5 -> 201.1 ms (mc grows 8 -> 16
            // accumulators), and double-buffering the next K-tile into registers
            // measured 193.6 -> 201.0 ms. Cut registers elsewhere first.
            uint k0 = lk + il0*16u;
            half A = arow_a[k0/32u];
            half B = arow_b[k0/32u];
            uint sy = lr0/8u, lx = lr0%8u;
            // The 16 nibbles are 8 contiguous bytes: two uchar4 loads rather than
            // 16 scalar ones reading every byte twice, once per nibble. The
            // destination indices collapse to a fixed stride-8 run from one base
            // (sx takes two values, the second +512 halfs), so the per-element index
            // arithmetic disappears too.
            device const uchar4* bp4 = (device const uchar4*)(arow + k0/2u);
            uchar4 b0 = bp4[0], b1 = bp4[1];
            threadgroup half* dst = sa + 64u*(16u*il0 + sy) + lx;
            dst[  0] = A*half(b0.x & 0x0Fu) + B;  dst[  8] = A*half(b0.x >> 4) + B;
            dst[ 16] = A*half(b0.y & 0x0Fu) + B;  dst[ 24] = A*half(b0.y >> 4) + B;
            dst[ 32] = A*half(b0.z & 0x0Fu) + B;  dst[ 40] = A*half(b0.z >> 4) + B;
            dst[ 48] = A*half(b0.w & 0x0Fu) + B;  dst[ 56] = A*half(b0.w >> 4) + B;
            dst[512] = A*half(b1.x & 0x0Fu) + B;  dst[520] = A*half(b1.x >> 4) + B;
            dst[528] = A*half(b1.y & 0x0Fu) + B;  dst[536] = A*half(b1.y >> 4) + B;
            dst[544] = A*half(b1.z & 0x0Fu) + B;  dst[552] = A*half(b1.z >> 4) + B;
            dst[560] = A*half(b1.w & 0x0Fu) + B;  dst[568] = A*half(b1.w >> 4) + B;
        }
        {   // B tile: token lr1, 8 consecutive k at lk + 8*sxb (zero-pad tokens >= M)
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            device const float4* xr = (device const float4*)(x + (ulong)(t0+lr1)*(ulong)K + lk + 8u*sxb);
            bool ok = t0 + lr1 < M;
            threadgroup half4* dstb = (threadgroup half4*)(sb + 64u*ib + 8u*ly);
            dstb[0] = ok ? half4(xr[0]) : half4(0.0);
            dstb[1] = ok ? half4(xr[1]) : half4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const half* lsmb = sb + 2u*64u*(sgitg/2u);
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
    if (accum != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 mid, mo;
        simdgroup_load(mid, idm, 8, 0, false);
        for (short i = 0; i < 8; i++) {
            device float* Ci = C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N;
            simdgroup_load(mo, Ci, N, 0, false);
            simdgroup_multiply_accumulate(mc[i], mc[i], mid, mo);
            simdgroup_store(mc[i], Ci, N, 0, false);
        }
    } else {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false);
        }
    }
}

// Batched Q6_K matmul — the lm_head for prefill and speculative verify.
//
// Without it token_embd cannot stay native: the batched path's lm_head goes through
// an MMA GEMM, and with only Q8/Q4L variants token_embd had to be requantized to Q8,
// costing 331 MB/token instead of 255 on a 152k vocab for a tensor the tied head
// reads in full every token.
//
// Structure is gemm_mm_q4l's: 64x32 tile, 128 threads, mc[8]. Only the A-tile fill
// differs — Q6_K super-blocks instead of Q4L nibbles plus side arrays.
// Half-activation twin of gemm_mm_q4l, for the GEMMs whose activation slab is too
// big to sit in cache. ffn_down's B operand is M x ffn f32 = 11.3 MB at M=256,
// re-read by every one of the 32 N-tiles (~361 MB per call), and it measures the
// worst efficiency of the big GEMMs (7.3 TFLOP/s vs ffn_gu's 9.1). Feeding it
// half activations (converted once per call by copy_f32_half into the xh scratch)
// halves that re-read traffic and turns the staging into plain 16-byte copies.
// Split-K twin of gemm_mm_q4l. Partition k-ranges are whole 32-element Q4L blocks,
// so scales never straddle a boundary. Partials are fp32, not the reference's f16:
// they must survive an accum into the residual stream. Reduced by splitk_accum.
kernel void gemm_mm_q4l_sk(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    constant uint& nsplit [[buffer(9)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[64*32];
    threadgroup half sb[32*32];
    threadgroup float idm[64];
    // Split-K: grid.z partitions each own a contiguous, 32-aligned K-range and
    // write plain fp32 partials at y + z*(M*N); a second pass sums them. Long-K
    // narrow-N shapes (ffn_down: K=11008, N=2048) are tile-count starved — 256
    // threadgroups over 344 serial K-steps — and split-K is grid parallelism the
    // shape cannot otherwise expose.
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;
    uint kper = ((K / nsplit) / 32u) * 32u;
    uint kbeg = tgpig.z * kper;
    uint kend = (tgpig.z + 1u == nsplit) ? K : kbeg + kper;
    device float* yz = y + (ulong)tgpig.z * (ulong)M * (ulong)N;
    uint lr0 = tiitg/2u;                        // A: row within tile (0..63)
    uint il0 = tiitg%2u;                        // A: which 16-elem half of the 32-k block
    uint nblk = K/32u;
    device const uchar* arow = w4 + (ulong)(r0+lr0)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr0)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr0)*(ulong)nblk;
    uint lr1 = tiitg/4u;                        // B: token (0..31)
    uint sxb = tiitg%4u;                        // B: k-octet
    if (tiitg < 64u) { idm[tiitg] = (tiitg/8u == tiitg%8u) ? 1.0 : 0.0; }
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = kbeg; lk < kend; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile: 16 Q4L weights of row lr0 at k = lk + il0*16 + i, dequantised
            // into the shared tile. k0 is 16-aligned and blocks are 32 wide, so all
            // 16 sit in one block and share its A/B pair — the per-lane header
            // re-reads that make the Q4_K layout slow at decode simply do not arise
            // here, because the tile is filled once and then MMA'd.
            //
            // This kernel is occupancy-limited by register pressure, so anything
            // adding live state to the inner loop loses: a 64x64 tile halving the
            // A-fill cost per output measured 193.5 -> 201.1 ms (mc grows 8 -> 16
            // accumulators), and double-buffering the next K-tile into registers
            // measured 193.6 -> 201.0 ms. Cut registers elsewhere first.
            uint k0 = lk + il0*16u;
            half A = arow_a[k0/32u];
            half B = arow_b[k0/32u];
            uint sy = lr0/8u, lx = lr0%8u;
            // The 16 nibbles are 8 contiguous bytes: two uchar4 loads rather than
            // 16 scalar ones reading every byte twice, once per nibble. The
            // destination indices collapse to a fixed stride-8 run from one base
            // (sx takes two values, the second +512 halfs), so the per-element index
            // arithmetic disappears too.
            device const uchar4* bp4 = (device const uchar4*)(arow + k0/2u);
            uchar4 b0 = bp4[0], b1 = bp4[1];
            threadgroup half* dst = sa + 64u*(16u*il0 + sy) + lx;
            dst[  0] = A*half(b0.x & 0x0Fu) + B;  dst[  8] = A*half(b0.x >> 4) + B;
            dst[ 16] = A*half(b0.y & 0x0Fu) + B;  dst[ 24] = A*half(b0.y >> 4) + B;
            dst[ 32] = A*half(b0.z & 0x0Fu) + B;  dst[ 40] = A*half(b0.z >> 4) + B;
            dst[ 48] = A*half(b0.w & 0x0Fu) + B;  dst[ 56] = A*half(b0.w >> 4) + B;
            dst[512] = A*half(b1.x & 0x0Fu) + B;  dst[520] = A*half(b1.x >> 4) + B;
            dst[528] = A*half(b1.y & 0x0Fu) + B;  dst[536] = A*half(b1.y >> 4) + B;
            dst[544] = A*half(b1.z & 0x0Fu) + B;  dst[552] = A*half(b1.z >> 4) + B;
            dst[560] = A*half(b1.w & 0x0Fu) + B;  dst[568] = A*half(b1.w >> 4) + B;
        }
        {   // B tile: token lr1, 8 consecutive k at lk + 8*sxb (zero-pad tokens >= M)
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            device const float4* xr = (device const float4*)(x + (ulong)(t0+lr1)*(ulong)K + lk + 8u*sxb);
            bool ok = t0 + lr1 < M;
            threadgroup half4* dstb = (threadgroup half4*)(sb + 64u*ib + 8u*ly);
            dstb[0] = ok ? half4(xr[0]) : half4(0.0);
            dstb[1] = ok ? half4(xr[1]) : half4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const half* lsmb = sb + 2u*64u*(sgitg/2u);
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
    device float* C = yz + (r0 + 32u*(sgitg & 1u)) + (ulong)(t0 + 16u*(sgitg >> 1u))*(ulong)N;
    if (false) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 mid, mo;
        simdgroup_load(mid, idm, 8, 0, false);
        for (short i = 0; i < 8; i++) {
            device float* Ci = C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N;
            simdgroup_load(mo, Ci, N, 0, false);
            simdgroup_multiply_accumulate(mc[i], mc[i], mid, mo);
            simdgroup_store(mc[i], Ci, N, 0, false);
        }
    } else {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false);
        }
    }
}

// Batched Q6_K matmul — the lm_head for prefill and speculative verify.
//
// Without it token_embd cannot stay native: the batched path's lm_head goes through
// an MMA GEMM, and with only Q8/Q4L variants token_embd had to be requantized to Q8,
// costing 331 MB/token instead of 255 on a 152k vocab for a tensor the tied head
// reads in full every token.
//
// Structure is gemm_mm_q4l's: 64x32 tile, 128 threads, mc[8]. Only the A-tile fill
// differs — Q6_K super-blocks instead of Q4L nibbles plus side arrays.
// Half-activation twin of gemm_mm_q4l, for the GEMMs whose activation slab is too
// big to sit in cache. ffn_down's B operand is M x ffn f32 = 11.3 MB at M=256,
// re-read by every one of the 32 N-tiles (~361 MB per call), and it measures the
// worst efficiency of the big GEMMs (7.3 TFLOP/s vs ffn_gu's 9.1). Feeding it
// half activations (converted once per call by copy_f32_half into the xh scratch)
// halves that re-read traffic and turns the staging into plain 16-byte copies.

// Sum split-K partials and ACCUMULATE into the destination (ffn_down's semantics:
// x += W.act). One thread per output element; entirely bandwidth-bound and tiny
// next to the GEMM it serves.
kernel void splitk_accum(device const float* part [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& total [[buffer(2)]], constant uint& nsplit [[buffer(3)]],
    constant uint& accum [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= total) { return; }
    float acc = 0.0;
    for (uint p = 0u; p < nsplit; p++) { acc += part[(ulong)p * (ulong)total + g]; }
    if (accum != 0u) { out[g] += acc; } else { out[g] = acc; }
}

kernel void gemm_mm_q4l_hb(device const half* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[64*32];
    threadgroup half sb[32*32];
    threadgroup float idm[64];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;
    uint lr0 = tiitg/2u;                        // A: row within tile (0..63)
    uint il0 = tiitg%2u;                        // A: which 16-elem half of the 32-k block
    uint nblk = K/32u;
    device const uchar* arow = w4 + (ulong)(r0+lr0)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr0)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr0)*(ulong)nblk;
    uint lr1 = tiitg/4u;                        // B: token (0..31)
    uint sxb = tiitg%4u;                        // B: k-octet
    if (tiitg < 64u) { idm[tiitg] = (tiitg/8u == tiitg%8u) ? 1.0 : 0.0; }
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile: 16 Q4L weights of row lr0 at k = lk + il0*16 + i, dequantised
            // into the shared tile. k0 is 16-aligned and blocks are 32 wide, so all
            // 16 sit in one block and share its A/B pair — the per-lane header
            // re-reads that make the Q4_K layout slow at decode simply do not arise
            // here, because the tile is filled once and then MMA'd.
            //
            // This kernel is occupancy-limited by register pressure, so anything
            // adding live state to the inner loop loses: a 64x64 tile halving the
            // A-fill cost per output measured 193.5 -> 201.1 ms (mc grows 8 -> 16
            // accumulators), and double-buffering the next K-tile into registers
            // measured 193.6 -> 201.0 ms. Cut registers elsewhere first.
            uint k0 = lk + il0*16u;
            half A = arow_a[k0/32u];
            half B = arow_b[k0/32u];
            uint sy = lr0/8u, lx = lr0%8u;
            // The 16 nibbles are 8 contiguous bytes: two uchar4 loads rather than
            // 16 scalar ones reading every byte twice, once per nibble. The
            // destination indices collapse to a fixed stride-8 run from one base
            // (sx takes two values, the second +512 halfs), so the per-element index
            // arithmetic disappears too.
            device const uchar4* bp4 = (device const uchar4*)(arow + k0/2u);
            uchar4 b0 = bp4[0], b1 = bp4[1];
            threadgroup half* dst = sa + 64u*(16u*il0 + sy) + lx;
            dst[  0] = A*half(b0.x & 0x0Fu) + B;  dst[  8] = A*half(b0.x >> 4) + B;
            dst[ 16] = A*half(b0.y & 0x0Fu) + B;  dst[ 24] = A*half(b0.y >> 4) + B;
            dst[ 32] = A*half(b0.z & 0x0Fu) + B;  dst[ 40] = A*half(b0.z >> 4) + B;
            dst[ 48] = A*half(b0.w & 0x0Fu) + B;  dst[ 56] = A*half(b0.w >> 4) + B;
            dst[512] = A*half(b1.x & 0x0Fu) + B;  dst[520] = A*half(b1.x >> 4) + B;
            dst[528] = A*half(b1.y & 0x0Fu) + B;  dst[536] = A*half(b1.y >> 4) + B;
            dst[544] = A*half(b1.z & 0x0Fu) + B;  dst[552] = A*half(b1.z >> 4) + B;
            dst[560] = A*half(b1.w & 0x0Fu) + B;  dst[568] = A*half(b1.w >> 4) + B;
        }
        {   // B tile from HALF activations: one 16-byte load stages all 8 elements.
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            device const half4* xr = (device const half4*)(x + (ulong)(t0+lr1)*(ulong)K + lk + 8u*sxb);
            bool ok = t0 + lr1 < M;
            threadgroup half4* dstb = (threadgroup half4*)(sb + 64u*ib + 8u*ly);
            dstb[0] = ok ? xr[0] : half4(0.0);
            dstb[1] = ok ? xr[1] : half4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const half* lsmb = sb + 2u*64u*(sgitg/2u);
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
    if (accum != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 mid, mo;
        simdgroup_load(mid, idm, 8, 0, false);
        for (short i = 0; i < 8; i++) {
            device float* Ci = C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N;
            simdgroup_load(mo, Ci, N, 0, false);
            simdgroup_multiply_accumulate(mc[i], mc[i], mid, mo);
            simdgroup_store(mc[i], Ci, N, 0, false);
        }
    } else {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false);
        }
    }
}

// Batched Q6_K matmul — the lm_head for prefill and speculative verify.
//
// Without it token_embd cannot stay native: the batched path's lm_head goes through
// an MMA GEMM, and with only Q8/Q4L variants token_embd had to be requantized to Q8,
// costing 331 MB/token instead of 255 on a 152k vocab for a tensor the tied head
// reads in full every token.
//
// Structure is gemm_mm_q4l's: 64x32 tile, 128 threads, mc[8]. Only the A-tile fill
// differs — Q6_K super-blocks instead of Q4L nibbles plus side arrays.

kernel void gemm_mm_q6k(device const float* x [[buffer(0)]], device const uchar* w6 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[64*32];
    threadgroup half sb[32*32];
    threadgroup float idm[64];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;
    uint lr0 = tiitg/2u;                        // A: row within tile (0..63)
    uint il0 = tiitg%2u;                        // A: which 16-elem half of the 32-k block
    uint nsb = K/256u;
    device const uchar* arow = w6 + (ulong)(r0+lr0)*(ulong)nsb*210ul;
    uint lr1 = tiitg/4u;                        // B: token (0..31)
    uint sxb = tiitg%4u;                        // B: k-octet
    if (tiitg < 64u) { idm[tiitg] = (tiitg/8u == tiitg%8u) ? 1.0 : 0.0; }
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile: 16 Q6_K weights of row lr0 at k = lk + il0*16, dequantised
            // into the shared tile. The run is 16-aligned, so it sits inside one
            // super-block with a single h/q/scale — everything but the ql/qh bytes
            // hoists out of the loop.
            uint k0 = lk + il0*16u;
            uint sb = k0 / 256u, io = k0 % 256u;
            device const uchar* b = arow + (ulong)sb*210ul;
            uint h = io / 128u, r = io % 128u, q = r / 32u, l0 = r % 32u;
            float dq = float(*(device const half*)(b + 208u));
            float sc = dq * float(((device const char*)(b + 192u))[h*8u + l0/16u + 2u*q]);
            uint qlb = h*64u + (q & 1u)*32u, qhb = 128u + h*32u;
            uint shl = (q >= 2u) ? 4u : 0u, shh = 2u*q;
            uint sy = lr0/8u, lx = lr0%8u;
            threadgroup half* dst = sa + 64u*(16u*il0 + sy) + lx;
            for (uint i = 0u; i < 16u; i++) {
                uint l = l0 + i;
                uint lo = (uint(b[qlb + l]) >> shl) & 0x0Fu;
                uint hi = (uint(b[qhb + l]) >> shh) & 3u;
                half v = half(sc * float(int(lo | (hi << 4u)) - 32));
                dst[(i < 8u) ? (8u*i) : (512u + 8u*(i - 8u))] = v;
            }
        }
        {   // B tile: token lr1, 8 consecutive k at lk + 8*sxb (zero-pad tokens >= M)
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            device const float4* xr = (device const float4*)(x + (ulong)(t0+lr1)*(ulong)K + lk + 8u*sxb);
            bool ok = t0 + lr1 < M;
            threadgroup half4* dstb = (threadgroup half4*)(sb + 64u*ib + 8u*ly);
            dstb[0] = ok ? half4(xr[0]) : half4(0.0);
            dstb[1] = ok ? half4(xr[1]) : half4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const half* lsmb = sb + 2u*64u*(sgitg/2u);
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
    if (accum != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 mid, mo;
        simdgroup_load(mid, idm, 8, 0, false);
        for (short i = 0; i < 8; i++) {
            device float* Ci = C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N;
            simdgroup_load(mo, Ci, N, 0, false);
            simdgroup_multiply_accumulate(mc[i], mc[i], mid, mo);
            simdgroup_store(mc[i], Ci, N, 0, false);
        }
    } else {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false);
        }
    }
}


kernel void gemm_mm_q8(device const float* x [[buffer(0)]], device const char* w8 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const float* scale [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[64*32];
    threadgroup half sb[32*32];
    threadgroup float idm[64];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;
    uint lr0 = tiitg/2u;                        // A: row within tile (0..63)
    uint il0 = tiitg%2u;                        // A: which 16-elem half of the 32-k block
    device const char* arow = w8 + (ulong)(r0+lr0)*(ulong)K;
    half srow = half(scale[r0+lr0]);            // per-row scale, constant over k
    uint lr1 = tiitg/4u;                        // B: token (0..31)
    uint sxb = tiitg%4u;                        // B: k-octet
    if (tiitg < 64u) { idm[tiitg] = (tiitg/8u == tiitg%8u) ? 1.0 : 0.0; }
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile: 16 int8 elems of row lr0 at k = lk + il0*16 + i, scaled to half
            device const char* bp = arow + lk + il0*16u;
            uint sy = lr0/8u, lx = lr0%8u;
            for (short i = 0; i < 16; i++) {
                half v = half(int(bp[i])) * srow;
                uint sx = 2u*il0 + uint(i)/8u;
                sa[64u*(8u*sx+sy) + 8u*(uint(i)%8u) + lx] = v;
            }
        }
        {   // B tile: token lr1, 8 consecutive k at lk + 8*sxb (zero-pad tokens >= M)
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            device const float4* xr = (device const float4*)(x + (ulong)(t0+lr1)*(ulong)K + lk + 8u*sxb);
            bool ok = t0 + lr1 < M;
            threadgroup half4* dstb = (threadgroup half4*)(sb + 64u*ib + 8u*ly);
            dstb[0] = ok ? half4(xr[0]) : half4(0.0);
            dstb[1] = ok ? half4(xr[1]) : half4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const half* lsmb = sb + 2u*64u*(sgitg/2u);
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
    if (accum != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 mid, mo;
        simdgroup_load(mid, idm, 8, 0, false);
        for (short i = 0; i < 8; i++) {
            device float* Ci = C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N;
            simdgroup_load(mo, Ci, N, 0, false);
            simdgroup_multiply_accumulate(mc[i], mc[i], mid, mo);
            simdgroup_store(mc[i], Ci, N, 0, false);
        }
    } else {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false);
        }
    }
}

// gemm_mm_f16 for any N and K: the same 64(N)x32(tok) MMA tile, with loads past N or
// K reading zero, and an output tile that reaches past N or M staged in threadgroup
// memory and stored element by element. For the shapes gemm_mm_f16 cannot take
// (N % 64 or K % 32 not zero); tokens past M are never stored.
kernel void gemm_mm_f16_edge(device const float* x [[buffer(0)]], device const half* w16 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half sa[64*32];
    threadgroup half sb[32*32];
    threadgroup float stage[4*8*64];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;
    uint lr0 = tiitg/2u;
    uint il0 = tiitg%2u;
    bool row_ok = r0 + lr0 < N;
    device const half* arow = w16 + (ulong)(row_ok ? r0 + lr0 : 0u)*(ulong)K;
    uint lr1 = tiitg/4u;
    uint sxb = tiitg%4u;
    bool tok_ok = t0 + lr1 < M;
    device const float* xrow = x + (ulong)(tok_ok ? t0 + lr1 : 0u)*(ulong)K;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile: 16 half elems of row lr0 at k = lk + il0*16 + i, zero past N or K
            uint sy = lr0/8u, lx = lr0%8u;
            for (short i = 0; i < 16; i++) {
                uint k = lk + il0*16u + uint(i);
                half v = (row_ok && k < K) ? arow[k] : half(0.0);
                uint sx = 2u*il0 + uint(i)/8u;
                sa[64u*(8u*sx+sy) + 8u*(uint(i)%8u) + lx] = v;
            }
        }
        {   // B tile: token lr1, 8 k at lk + 8*sxb, zero past M or K
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            threadgroup half* dstb = sb + 64u*ib + 8u*ly;
            for (short j = 0; j < 8; j++) {
                uint k = lk + 8u*sxb + uint(j);
                dstb[j] = (tok_ok && k < K) ? half(xrow[k]) : half(0.0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const half* lsmb = sb + 2u*64u*(sgitg/2u);
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
    // Block i of this simdgroup holds tokens t0 + 16*(sg>>1) + 8*(i/4) + r and
    // columns r0 + 32*(sg&1) + 8*(i%4) + c, element (r, c).
    threadgroup float* st = stage + 8u*64u*sgitg;
    for (short i = 0; i < 8; i++) { simdgroup_store(mc[i], st + 64*i, 8, 0, false); }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 8u*64u; e += 32u) {
        uint i = e/64u, r = (e%64u)/8u, c = e%8u;
        uint t = t0 + 16u*(sgitg >> 1u) + 8u*(i/4u) + r;
        uint n = r0 + 32u*(sgitg & 1u) + 8u*(i%4u) + c;
        if (t < M && n < N) {
            ulong o = (ulong)t*(ulong)N + n;
            y[o] = accum != 0u ? y[o] + st[e] : st[e];
        }
    }
}

// F16 twin of gemm_mm_q8: weights are raw half (w16), loaded directly into the A tile —
// no dequant, no scale. For the diffusion forward at prec=0 (f16), to test whether Q8 is
// degrading the (RL-sharpened) model. Same 64(N)×32(tok) MMA tile. N%64==0, K%32==0.
kernel void gemm_mm_f16(device const float* x [[buffer(0)]], device const half* w16 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[64*32];
    threadgroup half sb[32*32];
    threadgroup float idm[64];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;
    uint lr0 = tiitg/2u;
    uint il0 = tiitg%2u;
    device const half* arow = w16 + (ulong)(r0+lr0)*(ulong)K;
    uint lr1 = tiitg/4u;
    uint sxb = tiitg%4u;
    if (tiitg < 64u) { idm[tiitg] = (tiitg/8u == tiitg%8u) ? 1.0 : 0.0; }
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A tile: 16 half elems of row lr0 at k = lk + il0*16 + i (direct half load)
            device const half* bp = arow + lk + il0*16u;
            uint sy = lr0/8u, lx = lr0%8u;
            for (short i = 0; i < 16; i++) {
                half v = bp[i];
                uint sx = 2u*il0 + uint(i)/8u;
                sa[64u*(8u*sx+sy) + 8u*(uint(i)%8u) + lx] = v;
            }
        }
        {   // B tile: token lr1, 8 k at lk + 8*sxb (zero-pad tokens >= M)
            uint sy = lr1/8u, ly = lr1%8u;
            uint ib = 4u*sxb + sy;
            device const float4* xr = (device const float4*)(x + (ulong)(t0+lr1)*(ulong)K + lk + 8u*sxb);
            bool ok = t0 + lr1 < M;
            threadgroup half4* dstb = (threadgroup half4*)(sb + 64u*ib + 8u*ly);
            dstb[0] = ok ? half4(xr[0]) : half4(0.0);
            dstb[1] = ok ? half4(xr[1]) : half4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u*64u*(sgitg%2u);
        threadgroup const half* lsmb = sb + 2u*64u*(sgitg/2u);
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
    if (accum != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 mid, mo;
        simdgroup_load(mid, idm, 8, 0, false);
        for (short i = 0; i < 8; i++) {
            device float* Ci = C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N;
            simdgroup_load(mo, Ci, N, 0, false);
            simdgroup_multiply_accumulate(mc[i], mc[i], mid, mo);
            simdgroup_store(mc[i], Ci, N, 0, false);
        }
    } else {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false);
        }
    }
}

kernel void gemv_m2_q4(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_M2_FAST_BODY
    for (uint r = 0u; r < 4u; r++) {
        float a = simd_sum(r0[r]); float b = simd_sum(r1[r]);
        if (lane == 0u) { y[out_row + r] = a; y[(ulong)N + out_row + r] = b; }
    }
}

kernel void gemv_m2_q4_accum(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* scale [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    GEMV_M2_FAST_BODY
    for (uint r = 0u; r < 4u; r++) {
        float a = simd_sum(r0[r]); float b = simd_sum(r1[r]);
        if (lane == 0u) { y[out_row + r] += a; y[(ulong)N + out_row + r] += b; }
    }
}

kernel void gemv_q4k(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u; device const uchar* wr = w + (ulong)n*(ulong)nsb*144u;
    float acc = 0.0;
    for (uint base = 0u; base < nsb; base += 8u) { uint sb = base + (lane >> 2); 
        if (sb < nsb) { Q4K_DOT_G(wr, sb, x, lane & 3u, acc) } }
    acc = simd_sum(acc);
    if (lane == 0u) { y[n] = acc; }
}

kernel void gemv_q20(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* sc [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nblk = K/128u;
    device const uchar* wr = w + (ulong)n*(ulong)nblk*32ul;
    device const half* sr = sc + (ulong)n*(ulong)nblk;
    float acc = 0.0;
    for (uint b = lane; b < nblk; b += 32u) { Q20_DOT(wr, sr, b, x, acc) }
    acc = simd_sum(acc);
    if (lane == 0u) { y[n] = acc; }
}

kernel void gemv_q20_accum(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* sc [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nblk = K/128u;
    device const uchar* wr = w + (ulong)n*(ulong)nblk*32ul;
    device const half* sr = sc + (ulong)n*(ulong)nblk;
    float acc = 0.0;
    for (uint b = lane; b < nblk; b += 32u) { Q20_DOT(wr, sr, b, x, acc) }
    acc = simd_sum(acc);
    if (lane == 0u) { y[n] += acc; }
}

// Fused SwiGLU first half: gate and up dots share one pass over the activations.
kernel void ffn_gu_q20(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& act [[buffer(8)]],
    device const half* sg [[buffer(6)]], device const half* su [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nblk = K/128u;
    device const uchar* gr = wg + (ulong)n*(ulong)nblk*32ul;
    device const uchar* ur = wu + (ulong)n*(ulong)nblk*32ul;
    device const half* gs = sg + (ulong)n*(ulong)nblk;
    device const half* us = su + (ulong)n*(ulong)nblk;
    float pg = 0.0, pu = 0.0;
    for (uint b = lane; b < nblk; b += 32u) { Q20_DOT(gr, gs, b, x, pg) Q20_DOT(ur, us, b, x, pu) }
    pg = simd_sum(pg); pu = simd_sum(pu);
    if (lane == 0u) { out[n] = ffn_act(pg, act)*pu; }
}

// ---- GPU requantization: K-quant -> Q8 -------------------------------------
//
// Decode is fastest on Q8 (one int8*scale per weight); loading is fastest when
// K-quant blocks go to the GPU untouched. The CPU conversion costs minutes —
// dequantize every block to f16, then scan each row twice to quantize. These kernels
// do the same arithmetic on the GPU as a one-shot pass over resident weights.
//
// Layout out: `q[n*K + c]` int8 row-major + one f32 scale per row (amax/127),
// matching `quantize_row_i8`. Two passes over the row rather than staging it in
// threadgroup memory — ffn_down rows are K=11008, which does not fit.
kernel void requant_q4k_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u; device const uchar* wr = w + (ulong)n*(ulong)nsb*144u;
    float amax = 0.0;
    for (uint sb = lane; sb < nsb; sb += 32u) { Q4K_ROW(wr, sb, amax = max(amax, fabs(_v));) }
    amax = simd_max(amax);
    float s = amax > 0.0 ? amax/127.0 : 1.0;
    if (lane == 0u) { sc[n] = s; }
    float inv = 1.0/s;
    device char* qr = q + (ulong)n*(ulong)K;
    for (uint sb = lane; sb < nsb; sb += 32u) {
        Q4K_ROW(wr, sb, qr[_idx] = char(clamp(rint(_v*inv), -127.0, 127.0));)
    }
}

kernel void requant_q6k_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u; device const uchar* wr = w + (ulong)n*(ulong)nsb*210u;
    float amax = 0.0;
    for (uint sb = lane; sb < nsb; sb += 32u) { Q6K_ROW(wr, sb, amax = max(amax, fabs(_v));) }
    amax = simd_max(amax);
    float s = amax > 0.0 ? amax/127.0 : 1.0;
    if (lane == 0u) { sc[n] = s; }
    float inv = 1.0/s;
    device char* qr = q + (ulong)n*(ulong)K;
    for (uint sb = lane; sb < nsb; sb += 32u) {
        Q6K_ROW(wr, sb, qr[_idx] = char(clamp(rint(_v*inv), -127.0, 127.0));)
    }
}

kernel void requant_q5k_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u; device const uchar* wr = w + (ulong)n*(ulong)nsb*176u;
    float amax = 0.0;
    for (uint sb = lane; sb < nsb; sb += 32u) { Q5K_ROW(wr, sb, amax = max(amax, fabs(_v));) }
    amax = simd_max(amax);
    float s = amax > 0.0 ? amax/127.0 : 1.0;
    if (lane == 0u) { sc[n] = s; }
    float inv = 1.0/s;
    device char* qr = q + (ulong)n*(ulong)K;
    for (uint sb = lane; sb < nsb; sb += 32u) {
        Q5K_ROW(wr, sb, qr[_idx] = char(clamp(rint(_v*inv), -127.0, 127.0));)
    }
}

kernel void requant_q2k_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u; device const uchar* wr = w + (ulong)n*(ulong)nsb*84u;
    float amax = 0.0;
    for (uint sb = lane; sb < nsb; sb += 32u) { Q2K_ROW(wr, sb, amax = max(amax, fabs(_v));) }
    amax = simd_max(amax);
    float s = amax > 0.0 ? amax/127.0 : 1.0;
    if (lane == 0u) { sc[n] = s; }
    float inv = 1.0/s;
    device char* qr = q + (ulong)n*(ulong)K;
    for (uint sb = lane; sb < nsb; sb += 32u) {
        Q2K_ROW(wr, sb, qr[_idx] = char(clamp(rint(_v*inv), -127.0, 127.0));)
    }
}

kernel void requant_q3k_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u; device const uchar* wr = w + (ulong)n*(ulong)nsb*110u;
    float amax = 0.0;
    for (uint sb = lane; sb < nsb; sb += 32u) { Q3K_ROW(wr, sb, amax = max(amax, fabs(_v));) }
    amax = simd_max(amax);
    float s = amax > 0.0 ? amax/127.0 : 1.0;
    if (lane == 0u) { sc[n] = s; }
    float inv = 1.0/s;
    device char* qr = q + (ulong)n*(ulong)K;
    for (uint sb = lane; sb < nsb; sb += 32u) {
        Q3K_ROW(wr, sb, qr[_idx] = char(clamp(rint(_v*inv), -127.0, 127.0));)
    }
}

kernel void requant_iq4nl_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nblk = K/32u; device const uchar* wr = w + (ulong)n*(ulong)nblk*18u;
    float amax = 0.0;
    for (uint b = lane; b < nblk; b += 32u) { IQ4NL_ROW(wr, b, amax = max(amax, fabs(_v));) }
    amax = simd_max(amax);
    float s = amax > 0.0 ? amax/127.0 : 1.0;
    if (lane == 0u) { sc[n] = s; }
    float inv = 1.0/s;
    device char* qr = q + (ulong)n*(ulong)K;
    for (uint b = lane; b < nblk; b += 32u) {
        IQ4NL_ROW(wr, b, qr[_idx] = char(clamp(rint(_v*inv), -127.0, 127.0));)
    }
}

kernel void requant_iq4xs_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u; device const uchar* wr = w + (ulong)n*(ulong)nsb*136u;
    float amax = 0.0;
    for (uint sb = lane; sb < nsb; sb += 32u) { IQ4XS_ROW(wr, sb, amax = max(amax, fabs(_v));) }
    amax = simd_max(amax);
    float s = amax > 0.0 ? amax/127.0 : 1.0;
    if (lane == 0u) { sc[n] = s; }
    float inv = 1.0/s;
    device char* qr = q + (ulong)n*(ulong)K;
    for (uint sb = lane; sb < nsb; sb += 32u) {
        IQ4XS_ROW(wr, sb, qr[_idx] = char(clamp(rint(_v*inv), -127.0, 127.0));)
    }
}

// Q8_0 is already int8, but with a per-32 half scale; our decode kernels want one
// f32 scale per row. This rescales rather than re-quantizes, so it is exact up to
// the single rounding at the end.
kernel void requant_q80_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nblk = K/32u; device const uchar* wr = w + (ulong)n*(ulong)nblk*34u;
    float amax = 0.0;
    for (uint b = lane; b < nblk; b += 32u) { Q80_ROW(wr, b, amax = max(amax, fabs(_v));) }
    amax = simd_max(amax);
    float s = amax > 0.0 ? amax/127.0 : 1.0;
    if (lane == 0u) { sc[n] = s; }
    float inv = 1.0/s;
    device char* qr = q + (ulong)n*(ulong)K;
    for (uint b = lane; b < nblk; b += 32u) {
        Q80_ROW(wr, b, qr[_idx] = char(clamp(rint(_v*inv), -127.0, 127.0));)
    }
}

// Q8 -> our own Q4 (4-bit, per-32-block f16 scale).
//
// Three 4-bit paths, measured on the same model and M2 Max, per-token GPU time
// summed over all categories:
//
//   Q8 (per-row int8)          12.32 ms   354 GB/s on ffn_gu
//   native GGUF Q4_K           16.79 ms   135 GB/s   <- half the bytes, 3x the time
//   our Q4 (per-32 f16 scale)   9.39 ms   the tuned family: _fast/_ksplit/r4/autotune
//
// Native Q4_K loses because its kernel is one untuned shape. The fast route is
// therefore to reach Q8 on the GPU (already done, from any source format) and take
// one more pass to Q4 here.
//
// Chaining Q4_K -> Q8 -> Q4 rather than going direct costs almost nothing: Q8 holds
// ~8 bits of a 4.5-bit source, so the error is dominated by the final 4-bit step,
// which is the same either way. In exchange every readable quantization reaches this
// kernel through one path.
kernel void requant_q8_to_q4(device const char* q8 [[buffer(0)]], device const float* s8 [[buffer(1)]],
    device uchar* nib [[buffer(2)]], device half* sc4 [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nblk = K/32u;
    device const char* qr = q8 + (ulong)n*(ulong)K;
    float srow = s8[n];
    device uchar* nr = nib + (ulong)n*(ulong)(K/2u);
    device half* sr = sc4 + (ulong)n*(ulong)nblk;
    for (uint b = lane; b < nblk; b += 32u) {
        uint base = b*32u;
        float amax = 0.0;
        for (uint j=0u;j<32u;j++) { amax = max(amax, fabs(float(qr[base+j])*srow)); }
        // amax/8 (not /7): the nibble is a biased 0..15 code centred on 8.
        float s = amax > 0.0 ? amax/8.0 : 1.0;
        sr[b] = half(s);
        float inv = 1.0/s;
        for (uint j=0u;j<16u;j++) {
            float v0 = float(qr[base+2u*j])*srow;
            float v1 = float(qr[base+2u*j+1u])*srow;
            uint q0 = uint(clamp(rint(v0*inv)+8.0, 0.0, 15.0));
            uint q1 = uint(clamp(rint(v1*inv)+8.0, 0.0, 15.0));
            nr[b*16u+j] = uchar(q0 | (q1<<4));
        }
    }
}

// Direct Q8_0 -> tuned Q4 layout. This avoids the intermediate per-row Q8
// requantization, so the only new rounding is the requested 4-bit head.
kernel void requant_q80_q4(device const uchar* w [[buffer(0)]], device uchar* nib [[buffer(1)]],
    device half* sc4 [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) return;
    uint nblk=K/32u; device const uchar* wr=w+(ulong)n*(ulong)nblk*34ul;
    device uchar* nr=nib+(ulong)n*(ulong)(K/2u); device half* sr=sc4+(ulong)n*(ulong)nblk;
    for(uint b=lane;b<nblk;b+=32u) {
        device const uchar* z=wr+(ulong)b*34ul; float d=float(*reinterpret_cast<device const half*>(z));
        device const char* q=(device const char*)(z+2u); float amax=0.0f;
        for(uint j=0u;j<32u;j++) amax=max(amax,fabs(float(q[j])*d));
        float s=amax>0.0f?amax/8.0f:1.0f;sr[b]=half(s);float inv=1.0f/s;
        for(uint j=0u;j<16u;j++) {
            uint q0=uint(clamp(rint(float(q[2u*j])*d*inv)+8.0f,0.0f,15.0f));
            uint q1=uint(clamp(rint(float(q[2u*j+1u])*d*inv)+8.0f,0.0f,15.0f));
            nr[b*16u+j]=uchar(q0|(q1<<4));
        }
    }
}

// Fused SwiGLU over native K-quant weights: gate and up share one pass over the
// activations, as in ffn_gu_q8/_q20. These two matrices are the largest per layer,
// so keeping them native is most of the load-time win. The fused kernel needs both
// operands in the same format, which the loader enforces.
kernel void ffn_gu_q4k(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& act [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u;
    device const uchar* gr = wg + (ulong)n*(ulong)nsb*144u;
    device const uchar* ur = wu + (ulong)n*(ulong)nsb*144u;
    float pg = 0.0, pu = 0.0;
    for (uint base = 0u; base < nsb; base += 8u) { uint sb = base + (lane >> 2);
        if (sb < nsb) { Q4K_DOT_G(gr, sb, x, lane & 3u, pg) Q4K_DOT_G(ur, sb, x, lane & 3u, pu) } }
    pg = simd_sum(pg); pu = simd_sum(pu);
    if (lane == 0u) { out[n] = ffn_act(pg, act)*pu; }
}

kernel void ffn_gu_q6k(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& act [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u;
    device const uchar* gr = wg + (ulong)n*(ulong)nsb*210u;
    device const uchar* ur = wu + (ulong)n*(ulong)nsb*210u;
    float pg = 0.0, pu = 0.0;
    for (uint sb = 0u; sb < nsb; sb++) { Q6K_DOT_L(gr, sb, x, lane, pg) Q6K_DOT_L(ur, sb, x, lane, pu) }
    pg = simd_sum(pg); pu = simd_sum(pu);
    if (lane == 0u) { out[n] = ffn_act(pg, act)*pu; }
}

// Native Q6_K gemv. `*_K_M` files keep attn_qkv / ffn_down / output at Q6_K, so
// without it every one of those tensors takes the CPU dequant->requant path at load
// (145 s on a 1.8 GB model). Same shape as gemv_q4k; only the block stride (210 vs
// 144) and the dot macro differ.
kernel void gemv_q6k(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u; device const uchar* wr = w + (ulong)n*(ulong)nsb*210u;
    float acc = 0.0;
    for (uint sb = 0u; sb < nsb; sb++) { Q6K_DOT_L(wr, sb, x, lane, acc) }
    acc = simd_sum(acc);
    if (lane == 0u) { y[n] = acc; }
}

kernel void gemv_q6k_accum(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u; device const uchar* wr = w + (ulong)n*(ulong)nsb*210u;
    float acc = 0.0;
    for (uint sb = 0u; sb < nsb; sb++) { Q6K_DOT_L(wr, sb, x, lane, acc) }
    acc = simd_sum(acc);
    if (lane == 0u) { y[n] += acc; }
}

kernel void gemv_q4k_accum(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; }
    uint nsb = K/256u; device const uchar* wr = w + (ulong)n*(ulong)nsb*144u;
    float acc = 0.0;
    for (uint base = 0u; base < nsb; base += 8u) { uint sb = base + (lane >> 2); 
        if (sb < nsb) { Q4K_DOT_G(wr, sb, x, lane & 3u, acc) } }
    acc = simd_sum(acc);
    if (lane == 0u) { y[n] += acc; }
}

"#;
