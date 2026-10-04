//! Quantised-weight prefill GEMMs: the CUDA twins of Metal's `gemm_mm_*` family
//! (`ojas-metal/src/kernels/gemv.rs`), same entry names, same weight layouts, same contract:
//! f32 activations `x[M, K]` (row-major), quantised weights `W[N, K]` (one output per row, so
//! `y = x · Wᵀ`), f32 accumulation, f32 `y[M, N]`, `accum != 0` adds into `y`.
//!
//! One tensor-core pipeline serves all six: `gemm_mm_f16`'s m16n8k16 f16 mma with
//! ldmatrix fragments, except that the W tile is dequantised to f16 in shared memory by a
//! per-format loader instead of being copied. Each thread owns one 32-block of the 64-wide
//! k step; the raw bytes of step s+1 are read into registers while step s's MMAs run
//! (`gq_core` below). Like Metal, activations are rounded to f16 on their way into the
//! tile; the tensor core sums each 32 products in f16 and the partial sums are
//! accumulated in f32.
//!
//! | entry              | weights (row n)                                                     |
//! |--------------------|---------------------------------------------------------------------|
//! | `gemm_mm_q4`       | Q4 "pair" (`ojas_formats::quant::q4_pair_*`): `w4[N][K/2]`, byte j of a 32-block = elems 2j (low nibble), 2j+1 (high); f16 `scale[N][K/32]`; `v = (q - 8)·s` |
//! | `gemm_mm_q4l`      | Q4L (Q4_K relaid by `relayout_q4k_q4l`): same nibble packing, f16 `qa[N][K/32]`, `qb[N][K/32]`; `v = qa·q + qb` |
//! | `gemm_mm_q4l_hb`   | Q4L with f16 activations `x` (`copy_f32_half` first)                |
//! | `gemm_mm_q6k`      | GGML Q6_K super-blocks, 210 B / 256 weights, row-major              |
//! | `gemm_mm_q8`       | per-row symmetric int8 `w8[N][K]`, f32 `scale[N]`. The int8 values are exact in f16, so the scale is applied in f32 to the accumulator (Metal rounds it to half first) |
//! | `gemm_mm_q8_0`     | GGML Q8_0 blocks, 34 B / 32 weights (half scale, 32 int8), row-major: a GGUF's own Q8_0 tensors, read in place |
//!
//! Launch geometry: a 64-token x 128-output tile per block of 256 threads. Pointers come
//! first, then the u32 constants in Metal's buffer order:
//!
//! | entry              | args                                               | grid                                     |
//! |--------------------|----------------------------------------------------|------------------------------------------|
//! | `gemm_mm_q4`       | x, w4, y, scale, K, N, accum, M                    | `[ceil(N/128), ceil(M/64), 1]`           |
//! | `gemm_mm_q4l`      | x, w4, y, qa, qb, K, N, accum, M                   | `[ceil(N/128), ceil(M/64), 1]`           |
//! | `gemm_mm_q4l_hb`   | xh(f16), w4, y, qa, qb, K, N, accum, M             | `[ceil(N/128), ceil(M/64), 1]`           |
//! | `gemm_mm_q6k`      | x, w6, y, K, N, accum, M                           | `[ceil(N/128), ceil(M/64), 1]`           |
//! | `gemm_mm_q8`       | x, w8, y, scale(f32), K, N, accum, M               | `[ceil(N/128), ceil(M/64), 1]`           |
//! | `gemm_mm_q8_0`     | x, w(blocks), y, K, N, accum, M                    | `[ceil(N/128), ceil(M/64), 1]`           |
//! | `gemm_mm_q4l_sk`   | xh(f16), w4, y, qa, qb, K, N, M, nsplit            | `[ceil(N/128), ceil(M/64), nsplit]`      |
//! | `gemm_mm_q8_0_sk`  | xh(f16), w, y, K, N, M, nsplit                     | `[ceil(N/128), ceil(M/64), nsplit]`      |
//! | `gemm_mm_q6k_sk`   | xh(f16), w6, y, K, N, M, nsplit                    | `[ceil(N/128), ceil(M/64), nsplit]`      |
//! | `gemm_mm_f16_sk`   | x, w(f16), y, K, N, M, nsplit                      | `[ceil(N/128), ceil(M/64), nsplit]`      |
//! | `gemm_mm_{q4l,q8_0}{,_sk}_h` | as the entry without `_h`                | same                                     |
//!
//! The `_h` entries sum each 32 products in f16 on the tensor core (full rate on GeForce
//! parts) before accumulating in f32: 10-20 % faster, 1e-3 of relative error on a hidden
//! state against 1e-4 for the f32-accumulating entries. The decision models' one-pass
//! (`fast`) mode uses them; the exact and split modes use the plain entries.
//!
//! The `_sk` forms exist for short chunks: at M ≤ 128 a plain launch is one or two rows of
//! N/128 blocks, each walking all of K alone, which leaves most of the GPU idle; split-K
//! puts `nsplit` blocks on every tile, each adding its slice of K into `y` with f32 atomics
//! (so `y` holds what is to be accumulated into: zero it first for a plain product). The
//! quantized `_sk` forms take the activations as f16 (`copy_f32_half` once per matmul): the
//! tile would round them to f16 anyway, and converting once spares every block that reads
//! the same rows its own conversion.
//!
//! Shapes: any M and N (bounds masked on load and store; unlike Metal nothing is written
//! past row M, so y needs no padding). K % 32 == 0, K % 256 for Q6_K, as the block formats
//! require. `_sk`'s partition is Metal's: `kper = ((K / nsplit) / 32) · 32`, the last one
//! takes the remainder.
pub const BODY: &str = r#"
#define GQ_BN 128
#define GQ_BK 64
#define GQ_SROW 72
#define GQ_GROUP 8

// 8 nibbles of `u` (byte j: low then high) -> 4 half2 of a*q + c, one rounding each.
__device__ __forceinline__ void gq_nib8(unsigned u, float a, float c, unsigned* o) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned by = (u >> (8 * j)) & 0xffu;
        __half2 h = __floats2half2_rn(fmaf(a, (float)(by & 15u), c), fmaf(a, (float)(by >> 4), c));
        o[j] = *(unsigned*)&h;
    }
}

// 32 halves (16 packed pairs) to d, 16-byte aligned.
__device__ __forceinline__ void gq_put32(__half* d, const unsigned* o) {
    #pragma unroll
    for (int i = 0; i < 4; i++)
        *(uint4*)(d + 8 * i) = make_uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
}

// ---- weight formats. fetch(n, k): raw bytes of weights k..k+31 of row n into registers
// (k is 32-aligned: one 32-block, or a 32-run of a Q6_K super-block); put(d): dequantise
// those 32 to f16 at d (16-byte aligned); cs(n): per-output scale applied to the f32
// accumulator (1 unless the format has one).

// Q4 pair layout, symmetric: v = (q - 8) * s
struct GqQ4 {
    const unsigned char* w; const __half* s; int K;
    uint4 b; float sc;
    __device__ __forceinline__ void fetch(int n, int k) {
        b = *(const uint4*)(w + (long)n * (K >> 1) + (k >> 1));
        sc = __half2float(s[(long)n * (K >> 5) + (k >> 5)]);
    }
    __device__ __forceinline__ void put(__half* d) const {
        unsigned o[16];
        gq_nib8(b.x, sc, -8.f * sc, o);
        gq_nib8(b.y, sc, -8.f * sc, o + 4);
        gq_nib8(b.z, sc, -8.f * sc, o + 8);
        gq_nib8(b.w, sc, -8.f * sc, o + 12);
        gq_put32(d, o);
    }
    __device__ __forceinline__ float cs(int) const { return 1.f; }
};

// A nibble q as the half 1024 + q is its bits or-ed into 0x6400; two of them land in one
// register by masking the byte's halves into place. Subtracting 1024 (exact) leaves q as
// a half, and one fused multiply-add per pair gives qa*q + qb rounded once — what the f32
// path rounds to as well, at a third of the instructions.
__device__ __forceinline__ unsigned gq_nib2h(unsigned byte) {
    return 0x64006400u | (byte & 0x0fu) | ((byte & 0xf0u) << 12);
}

__device__ __forceinline__ void gq_nib8h(unsigned u, __half2 a2, __half2 c2, unsigned* o) {
    const __half2 k1024 = __floats2half2_rn(1024.f, 1024.f);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned h = gq_nib2h((u >> (8 * j)) & 0xffu);
        __half2 q = __hsub2(*(const __half2*)&h, k1024);
        __half2 v = __hfma2(q, a2, c2);
        o[j] = *(unsigned*)&v;
    }
}

// Q4L: same nibble packing, v = qa * q + qb
struct GqQ4L {
    const unsigned char* w; const __half* qa; const __half* qb; int K;
    uint4 b; __half2 a2, c2;
    __device__ __forceinline__ void fetch(int n, int k) {
        b = *(const uint4*)(w + (long)n * (K >> 1) + (k >> 1));
        long bi = (long)n * (K >> 5) + (k >> 5);
        a2 = __half2half2(qa[bi]);
        c2 = __half2half2(qb[bi]);
    }
    __device__ __forceinline__ void put(__half* d) const {
        unsigned o[16];
        gq_nib8h(b.x, a2, c2, o);
        gq_nib8h(b.y, a2, c2, o + 4);
        gq_nib8h(b.z, a2, c2, o + 8);
        gq_nib8h(b.w, a2, c2, o + 12);
        gq_put32(d, o);
    }
    __device__ __forceinline__ float cs(int) const { return 1.f; }
};

// 4 int8 in `u` -> 2 half2 (exact)
__device__ __forceinline__ void gq_i8x4(unsigned u, unsigned* o) {
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        float lo = (float)(signed char)((u >> (16 * j)) & 0xffu);
        float hi = (float)(signed char)((u >> (16 * j + 8)) & 0xffu);
        __half2 h = __floats2half2_rn(lo, hi);
        o[j] = *(unsigned*)&h;
    }
}

// Q8: per-row int8, exact in f16; the row scale goes on the accumulator
struct GqQ8 {
    const signed char* w; const float* s; int K;
    uint4 b0, b1;
    __device__ __forceinline__ void fetch(int n, int k) {
        const uint4* p = (const uint4*)(w + (long)n * K + k);
        b0 = p[0]; b1 = p[1];
    }
    __device__ __forceinline__ void put(__half* d) const {
        unsigned o[16];
        gq_i8x4(b0.x, o); gq_i8x4(b0.y, o + 2); gq_i8x4(b0.z, o + 4); gq_i8x4(b0.w, o + 6);
        gq_i8x4(b1.x, o + 8); gq_i8x4(b1.y, o + 10); gq_i8x4(b1.z, o + 12); gq_i8x4(b1.w, o + 14);
        gq_put32(d, o);
    }
    __device__ __forceinline__ float cs(int n) const { return s[n]; }
};

// GGML Q8_0: { half d; i8 qs[32]; } per 32, 34 B: v = d * q. The 32 qs start 2 bytes into
// a 2-byte-aligned block, so they are read as the nine 4-byte words that cover them and
// funnel-shifted into place (no divergence: the shift is 0 or 16 per thread). An int8 q
// flipped to q + 128 is the half 1152 + q when or-ed into 0x6400; one subtraction and one
// multiply per pair.
struct GqQ80 {
    const unsigned char* w; int K;
    unsigned wv[9]; int sh; __half2 d2;
    __device__ __forceinline__ void fetch(int n, int k) {
        const unsigned char* b = w + ((long)n * (K >> 5) + (k >> 5)) * 34;
        d2 = __half2half2(*(const __half*)b);
        const unsigned long a = (unsigned long)(b + 2);
        const unsigned* p = (const unsigned*)(a & ~3ul);
        sh = (a & 2) ? 16 : 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) wv[i] = p[i];
        wv[8] = sh ? p[8] : 0u;
    }
    __device__ __forceinline__ void put(__half* d) const {
        const __half2 k1152 = __floats2half2_rn(1152.f, 1152.f);
        unsigned o[16];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned q = __funnelshift_r(wv[i], wv[i + 1], sh);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                unsigned bytes = ((q >> (16 * j)) & 0xffffu) ^ 0x8080u;
                unsigned h = 0x64006400u | (bytes & 0xffu) | ((bytes & 0xff00u) << 8);
                __half2 v = __hmul2(__hsub2(*(const __half2*)&h, k1152), d2);
                o[2 * i + j] = *(unsigned*)&v;
            }
        }
        gq_put32(d, o);
    }
    __device__ __forceinline__ float cs(int) const { return 1.f; }
};

// Plain f16 rows, copied through the same pipeline so split-K serves f16 weights too.
struct GqF16 {
    const __half* w; int K;
    uint4 b[4];
    __device__ __forceinline__ void fetch(int n, int k) {
        const uint4* p = (const uint4*)(w + (long)n * K + k);
        #pragma unroll
        for (int i = 0; i < 4; i++) b[i] = p[i];
    }
    __device__ __forceinline__ void put(__half* d) const {
        #pragma unroll
        for (int i = 0; i < 4; i++) *(uint4*)(d + 8 * i) = b[i];
    }
    __device__ __forceinline__ float cs(int) const { return 1.f; }
};

// GGML Q6_K: { u8 ql[128]; u8 qh[64]; i8 scales[16]; half d; } per 256. A 32-aligned run of
// 32 spans two scales; its ql / qh are 32 consecutive bytes each (2-byte aligned: blocks
// are 210 B, read as u16).
struct GqQ6K {
    const unsigned char* w; int K;
    unsigned ql[8], qh[8]; float sc0, sc1; int shl, shh;
    __device__ __forceinline__ void fetch(int n, int k) {
        const unsigned char* b = w + ((long)n * (K >> 8) + (k >> 8)) * 210;
        int io = k & 255, h = io >> 7, q = (io & 127) >> 5;
        float dq = __half2float(*(const __half*)(b + 208));
        const signed char* sc = (const signed char*)b + 192 + h * 8 + 2 * q;
        sc0 = dq * (float)sc[0];
        sc1 = dq * (float)sc[1];
        const unsigned short* pl = (const unsigned short*)(b + h * 64 + (q & 1) * 32);
        const unsigned short* ph = (const unsigned short*)(b + 128 + h * 32);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            ql[i] = (unsigned)pl[2 * i] | ((unsigned)pl[2 * i + 1] << 16);
            qh[i] = (unsigned)ph[2 * i] | ((unsigned)ph[2 * i + 1] << 16);
        }
        shl = (q >= 2) ? 4 : 0;
        shh = 2 * q;
    }
    __device__ __forceinline__ void put(__half* d) const {
        unsigned o[16];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const float sc = i < 4 ? sc0 : sc1;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                float v[2];
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int bit = 16 * j + 8 * e;
                    unsigned lo = ((ql[i] >> bit) >> shl) & 15u;
                    unsigned hi = ((qh[i] >> bit) >> shh) & 3u;
                    v[e] = sc * (float)((int)(lo | (hi << 4)) - 32);
                }
                __half2 hv = __floats2half2_rn(v[0], v[1]);
                o[2 * i + j] = *(unsigned*)&hv;
            }
        }
        gq_put32(d, o);
    }
    __device__ __forceinline__ float cs(int) const { return 1.f; }
};

// ---------------------------------------------------------------------------------------
// gq_core: y[M,N] (+)= x[M, kbeg..kend) · W[N, kbeg..kend)^T.
//
// Block tile BM tokens x 128 outputs, k step 64, NW warps in a (BM/32) x (NW/(BM/32)) grid,
// each warp 32 tokens x (128/WN) outputs. BM = 128 is gemm_mm_f16's tile; 64 and 32 pad
// short chunks less. In the W tile thread t owns row t/2, 32-block t%2 of the step: one
// 16-byte read of nibbles (or 32 B of int8) and one scale per thread per step, a warp's
// reads filling whole 32-byte sectors. One shared-memory stage with the next step's raw
// bytes staged in registers: at every step the registers are dequantised into the tile,
// the next step's reads are issued, and the MMAs run while they land.
// HACC: the tensor core sums each 32 products in f16 (full rate on GeForce parts, where
// f32 accumulation runs at half rate) and the partial sums are added to f32 accumulators;
// the `_h` entries, 10-20 % faster on Kev's shapes, for the decision models' one-pass
// mode. Without it every product lands in f32: the exact and split modes' entries.
// Grouped raster over (N tiles, M tiles) as in gemm_mm_f16, so a wave shares x and W in L2.
// A split's range may end on a half step: the second 32-block of that step is zero.
// Rows past M are padding: a warp whose 32 rows are all padding issues no MMAs, and one
// whose second 16 rows are skips theirs, so a ragged M costs its tokens, not its tile.
// ---------------------------------------------------------------------------------------
template <class WF, int BM, bool XH, bool ATOMIC, bool HACC, int NW>
__device__ __forceinline__ void gq_core(const void* xv, const WF& wf0, float* y, int K, int N, int M,
                                        int accum, int kbeg, int kend) {
    constexpr int NT = 32 * NW;                            // threads
    constexpr int WM = BM / 32, WN = NW / WM, WNW = GQ_BN / WN, NI = WNW / 8;
    constexpr int FPT = (GQ_BN * 2) / NT;                  // 32-blocks of W per thread per step
    constexpr int XCH = XH ? BM * 8 : BM * 16;             // 16-byte x chunks in the tile
    constexpr int XI = (XCH + NT - 1) / NT;                // per thread (idle past XCH)
    __shared__ __align__(16) __half sA[BM * GQ_SROW];
    __shared__ __align__(16) __half sB[GQ_BN * GQ_SROW];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, R = lane >> 2, Q = lane & 3;
    const int nN = gridDim.x, nM = gridDim.y;
    const int lin = blockIdx.y * nN + blockIdx.x;
    const int per_group = GQ_GROUP * nN;
    const int first_m = (lin / per_group) * GQ_GROUP;
    const int gsz = min(nM - first_m, GQ_GROUP);
    const int in_g = lin % per_group;
    const int m0 = (first_m + in_g % gsz) * BM;
    const int n0 = (in_g / gsz) * GQ_BN;
    // warps wm = 0 are warps 0..WN-1: one per SM sub-partition, so when the rows of
    // wm = 1 are all padding and those warps skip the MMAs, every sub-partition is relieved
    const int wm = warp / WN, wn = warp % WN;
    // 16-row halves of this warp's 32 rows that hold tokens (the rest is padding past M)
    const bool live0 = m0 + wm * 32 < M, live1 = m0 + wm * 32 + 16 < M;
    const int nk = (kend - kbeg + GQ_BK - 1) / GQ_BK;
    // W rows and 32-blocks this thread fills (item = row * 2 + block); rows past N read
    // row N-1 (never stored)
    WF wf[FPT];
    int wrow[FPT], wk[FPT];
    __half* wdst[FPT];
    bool have[FPT];
    #pragma unroll
    for (int f = 0; f < FPT; f++) {
        const int item = tid + f * NT;
        wf[f] = wf0;
        wrow[f] = min(n0 + (item >> 1), N - 1);
        wk[f] = (item & 1) * 32;
        wdst[f] = &sB[(item >> 1) * GQ_SROW + wk[f]];
        have[f] = false;
    }

    float acc[2][NI][4];
    #pragma unroll
    for (int a = 0; a < 2; a++)
        #pragma unroll
        for (int b = 0; b < NI; b++)
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

    float4 xa[XI];
    uint4 xq[XI];
    auto load_x = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < XI; i++) {
            int idx = tid + i * NT;
            if (idx >= XCH) break;
            if (XH) {
                int r = idx >> 3, c = (idx & 7) * 8, m = m0 + r;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (m < M && k0 + c < kend) v = *(const uint4*)((const __half*)xv + (long)m * K + k0 + c);
                xq[i] = v;
            } else {
                int r = idx >> 4, c = (idx & 15) * 4, m = m0 + r;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (m < M && k0 + c < kend) v = *(const float4*)((const float*)xv + (long)m * K + k0 + c);
                xa[i] = v;
            }
        }
    };
    auto store_x = [&]() {
        #pragma unroll
        for (int i = 0; i < XI; i++) {
            int idx = tid + i * NT;
            if (idx >= XCH) break;
            if (XH) {
                int r = idx >> 3, c = (idx & 7) * 8;
                *(uint4*)(&sA[r * GQ_SROW + c]) = xq[i];
            } else {
                int r = idx >> 4, c = (idx & 15) * 4;
                __half2 lo = __floats2half2_rn(xa[i].x, xa[i].y);
                __half2 hi = __floats2half2_rn(xa[i].z, xa[i].w);
                uint2 u;
                u.x = *(unsigned*)&lo;
                u.y = *(unsigned*)&hi;
                *(uint2*)(&sA[r * GQ_SROW + c]) = u;
            }
        }
    };
    auto fetch_w = [&](int k0) {
        #pragma unroll
        for (int f = 0; f < FPT; f++) {
            have[f] = k0 + wk[f] < kend;
            if (have[f]) wf[f].fetch(wrow[f], k0 + wk[f]);
        }
    };
    auto put_w = [&]() {
        #pragma unroll
        for (int f = 0; f < FPT; f++) {
            if (have[f]) {
                wf[f].put(wdst[f]);
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) *(uint4*)(wdst[f] + 8 * i) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    };
    auto compute = [&]() {
        if (!live0) return;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            unsigned hacc[2][NI][2];                   // f16 partial sums (HACC only)
            #pragma unroll
            for (int a = 0; a < 2; a++)
                #pragma unroll
                for (int b = 0; b < NI; b++) { hacc[a][b][0] = 0u; hacc[a][b][1] = 0u; }
            #pragma unroll
            for (int ks = half * 32; ks < half * 32 + 32; ks += 16) {
                unsigned a[2][4], b[NI][2];
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    if (mi == 1 && !live1) break;
                    const __half* ap = sA + (wm * 32 + mi * 16 + (lane & 15)) * GQ_SROW + ks + (lane >> 4) * 8;
                    unsigned sa = (unsigned)__cvta_generic_to_shared(ap);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                        : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3]) : "r"(sa));
                }
                #pragma unroll
                for (int ni = 0; ni < NI; ni += 2) {
                    const __half* bp = sB + (wn * WNW + ni * 8 + (lane >> 4) * 8 + (lane & 7)) * GQ_SROW
                        + ks + ((lane >> 3) & 1) * 8;
                    unsigned sb = (unsigned)__cvta_generic_to_shared(bp);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                        : "=r"(b[ni][0]), "=r"(b[ni][1]), "=r"(b[ni + 1][0]), "=r"(b[ni + 1][1]) : "r"(sb));
                }
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    if (mi == 1 && !live1) break;
                    #pragma unroll
                    for (int ni = 0; ni < NI; ni++) {
                        if constexpr (HACC) {
                            asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                                : "+r"(hacc[mi][ni][0]), "+r"(hacc[mi][ni][1])
                                : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[ni][0]), "r"(b[ni][1]));
                        } else {
                            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]), "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                                : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[ni][0]), "r"(b[ni][1]));
                        }
                    }
                }
            }
            if constexpr (HACC) {
                #pragma unroll
                for (int mi = 0; mi < 2; mi++)
                    #pragma unroll
                    for (int ni = 0; ni < NI; ni++) {
                        float2 lo = __half22float2(*(const __half2*)&hacc[mi][ni][0]);
                        float2 hi = __half22float2(*(const __half2*)&hacc[mi][ni][1]);
                        acc[mi][ni][0] += lo.x; acc[mi][ni][1] += lo.y; acc[mi][ni][2] += hi.x; acc[mi][ni][3] += hi.y;
                    }
            }
        }
    };

    if (nk > 0) { fetch_w(kbeg); load_x(kbeg); }
    for (int kt = 0; kt < nk; kt++) {
        put_w();
        store_x();
        __syncthreads();
        if (kt + 1 < nk) { const int k1 = kbeg + (kt + 1) * GQ_BK; fetch_w(k1); load_x(k1); }
        compute();
        __syncthreads();
    }

    // C fragment reg l: token row (l/2)*8 + R, output col 2Q + (l&1)
    const bool pair = (N & 1) == 0;
    #pragma unroll
    for (int ni = 0; ni < NI; ni++) {
        const int n = n0 + wn * WNW + ni * 8 + 2 * Q;
        const float s0 = n < N ? wf0.cs(n) : 0.f, s1 = n + 1 < N ? wf0.cs(n + 1) : 0.f;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                int m = m0 + wm * 32 + mi * 16 + hh * 8 + R;
                if (m >= M) continue;
                float* yr = y + (long)m * N;
                float v0 = acc[mi][ni][hh * 2] * s0, v1 = acc[mi][ni][hh * 2 + 1] * s1;
                if (ATOMIC) {
                    if (n < N)     atomicAdd(yr + n, v0);
                    if (n + 1 < N) atomicAdd(yr + n + 1, v1);
                } else if (pair && n + 1 < N) {
                    float2* p = (float2*)(yr + n);
                    if (accum) { float2 o = *p; v0 += o.x; v1 += o.y; }
                    *p = make_float2(v0, v1);
                } else {
                    if (n < N)     { if (accum) v0 += yr[n];     yr[n] = v0; }
                    if (n + 1 < N) { if (accum) v1 += yr[n + 1]; yr[n + 1] = v1; }
                }
            }
    }
}

// Split-K: grid.z = nsplit partitions of whole 32-blocks, each adding its part into y.
#define GQ_SPLIT \
    const int z = blockIdx.z, kper = ((K / nsplit) / 32) * 32; \
    const int kbeg = z * kper, kend = (z + 1 == nsplit) ? K : kbeg + kper;

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q4(const float* x,
    const unsigned char* w4, float* y, const __half* scale, int K, int N, int accum, int M) {
    GqQ4 wf; wf.w = w4; wf.s = scale; wf.K = K;
    gq_core<GqQ4, 64, false, false, false, 8>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q4l(const float* x,
    const unsigned char* w4, float* y, const __half* qa, const __half* qb, int K, int N,
    int accum, int M) {
    GqQ4L wf; wf.w = w4; wf.qa = qa; wf.qb = qb; wf.K = K;
    gq_core<GqQ4L, 64, false, false, false, 8>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q4l_hb(const __half* x,
    const unsigned char* w4, float* y, const __half* qa, const __half* qb, int K, int N,
    int accum, int M) {
    GqQ4L wf; wf.w = w4; wf.qa = qa; wf.qb = qb; wf.K = K;
    gq_core<GqQ4L, 64, true, false, false, 8>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q6k(const float* x,
    const unsigned char* w6, float* y, int K, int N, int accum, int M) {
    GqQ6K wf; wf.w = w6; wf.K = K;
    gq_core<GqQ6K, 64, false, false, false, 8>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q8(const float* x,
    const signed char* w8, float* y, const float* scale, int K, int N, int accum, int M) {
    GqQ8 wf; wf.w = w8; wf.s = scale; wf.K = K;
    gq_core<GqQ8, 64, false, false, false, 8>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q8_0(const float* x,
    const unsigned char* w, float* y, int K, int N, int accum, int M) {
    GqQ80 wf; wf.w = w; wf.K = K;
    gq_core<GqQ80, 64, false, false, false, 8>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q4l_sk(const __half* x,
    const unsigned char* w4, float* y, const __half* qa, const __half* qb, int K, int N, int M, int nsplit) {
    GqQ4L wf; wf.w = w4; wf.qa = qa; wf.qb = qb; wf.K = K;
    GQ_SPLIT
    gq_core<GqQ4L, 64, true, true, false, 8>(x, wf, y, K, N, M, 1, kbeg, kend);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q8_0_sk(const __half* x,
    const unsigned char* w, float* y, int K, int N, int M, int nsplit) {
    GqQ80 wf; wf.w = w; wf.K = K;
    GQ_SPLIT
    gq_core<GqQ80, 64, true, true, false, 8>(x, wf, y, K, N, M, 1, kbeg, kend);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q6k_sk(const __half* x,
    const unsigned char* w6, float* y, int K, int N, int M, int nsplit) {
    GqQ6K wf; wf.w = w6; wf.K = K;
    GQ_SPLIT
    gq_core<GqQ6K, 64, true, true, false, 8>(x, wf, y, K, N, M, 1, kbeg, kend);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_f16_sk(const float* x,
    const __half* w, float* y, int K, int N, int M, int nsplit) {
    GqF16 wf; wf.w = w; wf.K = K;
    GQ_SPLIT
    gq_core<GqF16, 64, false, true, false, 8>(x, wf, y, K, N, M, 1, kbeg, kend);
}

// f16 partial sums (HACC): the decision models' fast path.
extern "C" __global__ void __launch_bounds__(256) gemm_mm_q4l_h(const float* x,
    const unsigned char* w4, float* y, const __half* qa, const __half* qb, int K, int N,
    int accum, int M) {
    GqQ4L wf; wf.w = w4; wf.qa = qa; wf.qb = qb; wf.K = K;
    gq_core<GqQ4L, 64, false, false, true, 8>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q8_0_h(const float* x,
    const unsigned char* w, float* y, int K, int N, int accum, int M) {
    GqQ80 wf; wf.w = w; wf.K = K;
    gq_core<GqQ80, 64, false, false, true, 8>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q4l_sk_h(const __half* x,
    const unsigned char* w4, float* y, const __half* qa, const __half* qb, int K, int N, int M, int nsplit) {
    GqQ4L wf; wf.w = w4; wf.qa = qa; wf.qb = qb; wf.K = K;
    GQ_SPLIT
    gq_core<GqQ4L, 64, true, true, true, 8>(x, wf, y, K, N, M, 1, kbeg, kend);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q8_0_sk_h(const __half* x,
    const unsigned char* w, float* y, int K, int N, int M, int nsplit) {
    GqQ80 wf; wf.w = w; wf.K = K;
    GQ_SPLIT
    gq_core<GqQ80, 64, true, true, true, 8>(x, wf, y, K, N, M, 1, kbeg, kend);
}
"#;

pub const NAMES: &[&str] = &[
    "gemm_mm_q4",
    "gemm_mm_q4l",
    "gemm_mm_q4l_hb",
    "gemm_mm_q6k",
    "gemm_mm_q8",
    "gemm_mm_q8_0",
    "gemm_mm_q4l_sk",
    "gemm_mm_q8_0_sk",
    "gemm_mm_q6k_sk",
    "gemm_mm_f16_sk",
    "gemm_mm_q4l_h",
    "gemm_mm_q8_0_h",
    "gemm_mm_q4l_sk_h",
    "gemm_mm_q8_0_sk_h",
];
