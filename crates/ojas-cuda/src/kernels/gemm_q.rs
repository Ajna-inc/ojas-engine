//! Quantised-weight prefill GEMMs: the CUDA twins of Metal's `gemm_mm_*` family
//! (`ojas-metal/src/kernels/gemv.rs`), same entry names, same weight layouts, same contract:
//! f32 activations `x[M, K]` (row-major), quantised weights `W[N, K]` (one output per row, so
//! `y = x · Wᵀ`), f32 accumulation, f32 `y[M, N]`, `accum != 0` adds into `y`.
//!
//! One tensor-core pipeline serves all seven: `gemm_mm_f16`'s m16n8k16 f16/f32 mma with
//! ldmatrix fragments, except that the W tile is dequantised to f16 in shared memory by a
//! per-format loader instead of being copied. The raw quant bytes for k-step s+1 are loaded
//! into registers before the mma of step s and dequantised + stored after it (the register
//! staging `gemm_mm_f16` uses for x), so the load latency hides behind the mma. Like Metal,
//! activations are rounded to f16 on their way into the tile.
//!
//! | entry              | weights (row n)                                                     |
//! |--------------------|---------------------------------------------------------------------|
//! | `gemm_mm_q4`       | Q4 "pair" (`ojas_formats::quant::q4_pair_*`): `w4[N][K/2]`, byte j of a 32-block = elems 2j (low nibble), 2j+1 (high); f16 `scale[N][K/32]`; `v = (q - 8)·s` |
//! | `gemm_mm_q4l`      | Q4L (Q4_K relaid by `relayout_q4k_q4l`): same nibble packing, f16 `qa[N][K/32]`, `qb[N][K/32]`; `v = qa·q + qb` |
//! | `gemm_mm_q4l_mlx`  | Q4L, 64-token tile (Metal's 64×64 "MLX" config; here 64 tokens × 128 outputs) — less padding waste at small M |
//! | `gemm_mm_q4l_sk`   | Q4L split-K: grid.z partitions of whole 32-blocks write plain f32 partials at `y + z·M·N`; `splitk_accum` sums them (and applies accum) |
//! | `gemm_mm_q4l_hb`   | Q4L with f16 activations `x` (`copy_f32_half` first)                |
//! | `gemm_mm_q6k`      | GGML Q6_K super-blocks, 210 B / 256 weights, row-major              |
//! | `gemm_mm_q8`       | per-row symmetric int8 `w8[N][K]`, f32 `scale[N]`. The int8 values are exact in f16, so the scale is applied in f32 to the accumulator (Metal rounds it to half first) |
//!
//! Launch geometry. Pointers come first, then the u32 constants in Metal's buffer order:
//!
//! | entry              | args                                               | grid                                     | block |
//! |--------------------|----------------------------------------------------|------------------------------------------|-------|
//! | `gemm_mm_q4`       | x, w4, y, scale, K, N, accum, M                    | `[ceil(N/128), ceil(M/128), 1]`          | 256   |
//! | `gemm_mm_q4l`      | x, w4, y, qa, qb, K, N, accum, M                   | `[ceil(N/128), ceil(M/128), 1]`          | 256   |
//! | `gemm_mm_q4l_mlx`  | x, w4, y, qa, qb, K, N, accum, M                   | `[ceil(N/128), ceil(M/64), 1]`           | 256   |
//! | `gemm_mm_q4l_sk`   | x, w4, part, qa, qb, K, N, accum(ignored), M, nsplit | `[ceil(N/128), ceil(M/128), nsplit]`   | 256   |
//! | `gemm_mm_q4l_hb`   | xh(f16), w4, y, qa, qb, K, N, accum, M             | `[ceil(N/128), ceil(M/128), 1]`          | 256   |
//! | `gemm_mm_q6k`      | x, w6, y, K, N, accum, M                           | `[ceil(N/128), ceil(M/128), 1]`          | 256   |
//! | `gemm_mm_q8`       | x, w8, y, scale(f32), K, N, accum, M               | `[ceil(N/128), ceil(M/128), 1]`          | 256   |
//! | `splitk_accum`     | part, out, total(=M·N), nsplit, accum              | `[ceil(total/256), 1, 1]`                | 256   |
//!
//! Shapes: any M and N (bounds masked on load and store; unlike Metal nothing is written
//! past row M, so y needs no padding). K % 32 == 0, K % 256 for Q6_K, as the block formats
//! require. `_sk`'s partition is Metal's: `kper = ((K / nsplit) / 32) · 32`, the last one
//! takes the remainder; the partial buffer is `nsplit · M · N` floats.
pub const BODY: &str = r#"
#define GQ_BN 128
#define GQ_BK 32
#define GQ_SROW 40
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

__device__ __forceinline__ void gq_put16(__half* d, const unsigned* o) {
    *(uint4*)d = make_uint4(o[0], o[1], o[2], o[3]);
    *(uint4*)(d + 8) = make_uint4(o[4], o[5], o[6], o[7]);
}

// ---- weight formats. fetch(n, k): raw bytes of weights k..k+15 of row n into registers
// (k is 16-aligned, so the run sits inside one 32-block / one Q6_K super-block);
// put(d): dequantise those 16 to f16 at d (16-byte aligned); cs(n): per-output scale applied
// to the f32 accumulator (1 unless the format has one).

// Q4 pair layout, symmetric: v = (q - 8) * s
struct GqQ4 {
    const unsigned char* w; const __half* s; int K;
    uint2 b; float sc;
    __device__ __forceinline__ void fetch(int n, int k) {
        b = *(const uint2*)(w + (long)n * (K >> 1) + (k >> 1));
        sc = __half2float(s[(long)n * (K >> 5) + (k >> 5)]);
    }
    __device__ __forceinline__ void put(__half* d) const {
        unsigned o[8];
        gq_nib8(b.x, sc, -8.f * sc, o);
        gq_nib8(b.y, sc, -8.f * sc, o + 4);
        gq_put16(d, o);
    }
    __device__ __forceinline__ float cs(int) const { return 1.f; }
};

// Q4L: same nibble packing, v = qa * q + qb
struct GqQ4L {
    const unsigned char* w; const __half* qa; const __half* qb; int K;
    uint2 b; float a, c;
    __device__ __forceinline__ void fetch(int n, int k) {
        b = *(const uint2*)(w + (long)n * (K >> 1) + (k >> 1));
        long bi = (long)n * (K >> 5) + (k >> 5);
        a = __half2float(qa[bi]);
        c = __half2float(qb[bi]);
    }
    __device__ __forceinline__ void put(__half* d) const {
        unsigned o[8];
        gq_nib8(b.x, a, c, o);
        gq_nib8(b.y, a, c, o + 4);
        gq_put16(d, o);
    }
    __device__ __forceinline__ float cs(int) const { return 1.f; }
};

// Q8: per-row int8, exact in f16; the row scale goes on the accumulator
struct GqQ8 {
    const signed char* w; const float* s; int K;
    uint4 b;
    __device__ __forceinline__ void fetch(int n, int k) {
        b = *(const uint4*)(w + (long)n * K + k);
    }
    __device__ __forceinline__ void put(__half* d) const {
        unsigned o[8];
        const unsigned u[4] = {b.x, b.y, b.z, b.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                float lo = (float)(signed char)((u[i] >> (16 * j)) & 0xffu);
                float hi = (float)(signed char)((u[i] >> (16 * j + 8)) & 0xffu);
                __half2 h = __floats2half2_rn(lo, hi);
                o[2 * i + j] = *(unsigned*)&h;
            }
        }
        gq_put16(d, o);
    }
    __device__ __forceinline__ float cs(int n) const { return s[n]; }
};

// GGML Q6_K: { u8 ql[128]; u8 qh[64]; i8 scales[16]; half d; } per 256. A 16-aligned run of
// 16 shares one scale; ql/qh are 16 consecutive bytes each (2-byte aligned: blocks are 210 B).
struct GqQ6K {
    const unsigned char* w; int K;
    unsigned ql[4], qh[4]; float sc; int shl, shh;
    __device__ __forceinline__ void fetch(int n, int k) {
        const unsigned char* b = w + ((long)n * (K >> 8) + (k >> 8)) * 210;
        int io = k & 255, h = io >> 7, r = io & 127, q = r >> 5, l0 = r & 31;
        float dq = __half2float(*(const __half*)(b + 208));
        sc = dq * (float)((const signed char*)b)[192 + h * 8 + (l0 >> 4) + 2 * q];
        const unsigned short* pl = (const unsigned short*)(b + h * 64 + (q & 1) * 32 + l0);
        const unsigned short* ph = (const unsigned short*)(b + 128 + h * 32 + l0);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            ql[i] = (unsigned)pl[2 * i] | ((unsigned)pl[2 * i + 1] << 16);
            qh[i] = (unsigned)ph[2 * i] | ((unsigned)ph[2 * i + 1] << 16);
        }
        shl = (q >= 2) ? 4 : 0;
        shh = 2 * q;
    }
    __device__ __forceinline__ void put(__half* d) const {
        unsigned o[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
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
                __half2 h = __floats2half2_rn(v[0], v[1]);
                o[2 * i + j] = *(unsigned*)&h;
            }
        }
        gq_put16(d, o);
    }
    __device__ __forceinline__ float cs(int) const { return 1.f; }
};

// ---------------------------------------------------------------------------------------
// gq_core: y[M,N] (+)= x[M, kbeg..kend) · W[N, kbeg..kend)^T.
//
// Block tile BM tokens x 128 outputs, k step 32, 8 warps in a (BM/32) x (8/(BM/32)) grid,
// each warp 32 tokens x (128/WN) outputs. BM = 128 is gemm_mm_f16's tile; BM = 64 halves the
// token padding for short chunks. In the W tile thread t owns row t/2, weights 16·(t%2)..+16
// of the k step — one fetch/put per thread per step. Two smem stages; both operands
// register-staged (loads for step s+1 issued before step s's mma, stored after it).
// Grouped raster over (N tiles, M tiles) as in gemm_mm_f16, so a wave shares x and W in L2.
// ---------------------------------------------------------------------------------------
template <class WF, int BM, bool XH>
__device__ __forceinline__ void gq_core(const void* xv, WF wf, float* y, int K, int N, int M,
                                        int accum, int kbeg, int kend) {
    constexpr int WM = BM / 32, WN = 8 / WM, WNW = GQ_BN / WN, NI = WNW / 8;
    constexpr int XI = XH ? BM * 4 / 256 : BM * 8 / 256;   // 16-byte x chunks per thread
    __shared__ __align__(16) __half sA[2][BM * GQ_SROW];
    __shared__ __align__(16) __half sB[2][GQ_BN * GQ_SROW];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, R = lane >> 2, Q = lane & 3;
    const int nN = gridDim.x, nM = gridDim.y;
    const int lin = blockIdx.y * nN + blockIdx.x;
    const int per_group = GQ_GROUP * nN;
    const int first_m = (lin / per_group) * GQ_GROUP;
    const int gsz = min(nM - first_m, GQ_GROUP);
    const int in_g = lin % per_group;
    const int m0 = (first_m + in_g % gsz) * BM;
    const int n0 = (in_g / gsz) * GQ_BN;
    const int wm = warp % WM, wn = warp / WM;
    const int nk = (kend - kbeg) / GQ_BK;
    // W row this thread fills; rows past N read row N-1 (their outputs are never stored)
    const int wrow = min(n0 + (tid >> 1), N - 1), wk = (tid & 1) * 16;
    const int wdst = (tid >> 1) * GQ_SROW + wk;

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
            int idx = tid + i * 256;
            if (XH) {
                int r = idx >> 2, c = (idx & 3) * 8, m = m0 + r;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (m < M) v = *(const uint4*)((const __half*)xv + (long)m * K + k0 + c);
                xq[i] = v;
            } else {
                int r = idx >> 3, c = (idx & 7) * 4, m = m0 + r;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (m < M) v = *(const float4*)((const float*)xv + (long)m * K + k0 + c);
                xa[i] = v;
            }
        }
    };
    auto store_x = [&](int st) {
        #pragma unroll
        for (int i = 0; i < XI; i++) {
            int idx = tid + i * 256;
            if (XH) {
                int r = idx >> 2, c = (idx & 3) * 8;
                *(uint4*)(&sA[st][r * GQ_SROW + c]) = xq[i];
            } else {
                int r = idx >> 3, c = (idx & 7) * 4;
                __half2 lo = __floats2half2_rn(xa[i].x, xa[i].y);
                __half2 hi = __floats2half2_rn(xa[i].z, xa[i].w);
                uint2 u;
                u.x = *(unsigned*)&lo;
                u.y = *(unsigned*)&hi;
                *(uint2*)(&sA[st][r * GQ_SROW + c]) = u;
            }
        }
    };

    if (nk > 0) {
        wf.fetch(wrow, kbeg + wk);
        load_x(kbeg);
        wf.put(&sB[0][wdst]);
        store_x(0);
    }
    for (int kt = 0; kt < nk; kt++) {
        const int cur = kt & 1;
        __syncthreads();
        const bool more = kt + 1 < nk;
        if (more) {
            const int k1 = kbeg + (kt + 1) * GQ_BK;
            wf.fetch(wrow, k1 + wk);
            load_x(k1);
        }
        const __half* A = sA[cur];
        const __half* B = sB[cur];
        #pragma unroll
        for (int ks = 0; ks < GQ_BK; ks += 16) {
            unsigned a[2][4], b[NI][2];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const __half* ap = A + (wm * 32 + mi * 16 + (lane & 15)) * GQ_SROW + ks + (lane >> 4) * 8;
                unsigned sa = (unsigned)__cvta_generic_to_shared(ap);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3]) : "r"(sa));
            }
            #pragma unroll
            for (int ni = 0; ni < NI; ni += 2) {
                const __half* bp = B + (wn * WNW + ni * 8 + (lane >> 4) * 8 + (lane & 7)) * GQ_SROW
                    + ks + ((lane >> 3) & 1) * 8;
                unsigned sb = (unsigned)__cvta_generic_to_shared(bp);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(b[ni][0]), "=r"(b[ni][1]), "=r"(b[ni + 1][0]), "=r"(b[ni + 1][1]) : "r"(sb));
            }
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < NI; ni++)
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]), "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[ni][0]), "r"(b[ni][1]));
        }
        // the stage refilled here was last read in step kt-1; every warp has passed this
        // step's barrier since, so the write cannot race a reader
        if (more) {
            wf.put(&sB[cur ^ 1][wdst]);
            store_x(cur ^ 1);
        }
    }

    // C fragment reg l: token row (l/2)*8 + R, output col 2Q + (l&1)
    const bool pair = (N & 1) == 0;
    #pragma unroll
    for (int ni = 0; ni < NI; ni++) {
        const int n = n0 + wn * WNW + ni * 8 + 2 * Q;
        const float s0 = n < N ? wf.cs(n) : 0.f, s1 = n + 1 < N ? wf.cs(n + 1) : 0.f;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                int m = m0 + wm * 32 + mi * 16 + hh * 8 + R;
                if (m >= M) continue;
                float* yr = y + (long)m * N;
                float v0 = acc[mi][ni][hh * 2] * s0, v1 = acc[mi][ni][hh * 2 + 1] * s1;
                if (pair && n + 1 < N) {
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

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q4(const float* x,
    const unsigned char* w4, float* y, const __half* scale, int K, int N, int accum, int M) {
    GqQ4 wf; wf.w = w4; wf.s = scale; wf.K = K;
    gq_core<GqQ4, 128, false>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q4l(const float* x,
    const unsigned char* w4, float* y, const __half* qa, const __half* qb, int K, int N,
    int accum, int M) {
    GqQ4L wf; wf.w = w4; wf.qa = qa; wf.qb = qb; wf.K = K;
    gq_core<GqQ4L, 128, false>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q4l_mlx(const float* x,
    const unsigned char* w4, float* y, const __half* qa, const __half* qb, int K, int N,
    int accum, int M) {
    GqQ4L wf; wf.w = w4; wf.qa = qa; wf.qb = qb; wf.K = K;
    gq_core<GqQ4L, 64, false>(x, wf, y, K, N, M, accum, 0, K);
}

// Split-K: grid.z = nsplit partitions of whole 32-blocks, plain f32 partials at y + z*M*N.
extern "C" __global__ void __launch_bounds__(256) gemm_mm_q4l_sk(const float* x,
    const unsigned char* w4, float* y, const __half* qa, const __half* qb, int K, int N,
    int accum, int M, int nsplit) {
    GqQ4L wf; wf.w = w4; wf.qa = qa; wf.qb = qb; wf.K = K;
    const int z = blockIdx.z, kper = ((K / nsplit) / 32) * 32;
    const int kbeg = z * kper, kend = (z + 1 == nsplit) ? K : kbeg + kper;
    gq_core<GqQ4L, 128, false>(x, wf, y + (long)z * M * N, K, N, M, 0, kbeg, kend);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q4l_hb(const __half* x,
    const unsigned char* w4, float* y, const __half* qa, const __half* qb, int K, int N,
    int accum, int M) {
    GqQ4L wf; wf.w = w4; wf.qa = qa; wf.qb = qb; wf.K = K;
    gq_core<GqQ4L, 128, true>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q6k(const float* x,
    const unsigned char* w6, float* y, int K, int N, int accum, int M) {
    GqQ6K wf; wf.w = w6; wf.K = K;
    gq_core<GqQ6K, 128, false>(x, wf, y, K, N, M, accum, 0, K);
}

extern "C" __global__ void __launch_bounds__(256) gemm_mm_q8(const float* x,
    const signed char* w8, float* y, const float* scale, int K, int N, int accum, int M) {
    GqQ8 wf; wf.w = w8; wf.s = scale; wf.K = K;
    gq_core<GqQ8, 128, false>(x, wf, y, K, N, M, accum, 0, K);
}

// Sum split-K partials into out (accum != 0: out += sum). Metal: gemv.rs `splitk_accum`.
extern "C" __global__ void splitk_accum(const float* part, float* out, unsigned int total,
    unsigned int nsplit, unsigned int accum) {
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;
    float acc = 0.f;
    for (unsigned int p = 0; p < nsplit; p++) acc += part[(unsigned long long)p * total + g];
    if (accum) out[g] += acc; else out[g] = acc;
}
"#;

pub const NAMES: &[&str] = &[
    "gemm_mm_q4",
    "gemm_mm_q4l",
    "gemm_mm_q4l_mlx",
    "gemm_mm_q4l_sk",
    "gemm_mm_q4l_hb",
    "gemm_mm_q6k",
    "gemm_mm_q8",
    "splitk_accum",
];
