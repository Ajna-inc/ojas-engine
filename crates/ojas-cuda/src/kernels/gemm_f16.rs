//! F16-weight matmuls: the dense path of an F16 model (surya-2's LLM and vision tower).
//!
//! Same names and data contract as Metal (`ojas-metal/src/kernels/gemv.rs`): f32 activations
//! `x[M, K]` (row-major), F16 weights `W[N, K]` (row-major, one output per row, so
//! `y = x · Wᵀ`), f32 accumulation, f32 `y[M, N]`.
//!
//! * `gemm_mm_f16`  — prefill / vision tower, tensor cores. Like Metal it rounds each
//!   activation to f16 on its way into the tile (the mma takes f16 on both sides) and
//!   accumulates in f32. Unlike Metal it takes any M, N, K (Metal: N % 64, K % 32, and it
//!   stores whole 32-token tiles past M) — bounds are zero-filled on load and masked on store.
//! * `gemv_f16`     — one token; memory-bound, the weight read is the whole cost.
//! * `gemv_m_f16`   — M ≤ 8 tokens reading each weight row once (Metal's contract; M > 8 is
//!   still correct here, it just re-reads W once per 8 rows).
//!
//! Launch geometry:
//!
//! | entry         | args (pointers, then u32)            | grid                              | block |
//! |---------------|--------------------------------------|-----------------------------------|-------|
//! | `gemm_mm_f16` | x, w, y, K, N, accum, M               | `[ceil(N/128), ceil(M/128), 1]`   | 256   |
//! | `gemv_f16`    | x, w, y, K, N                         | `[ceil(N/8), 1, 1]`               | 256   |
//! | `gemv_m_f16`  | x, w, y, K, N, M                      | `[ceil(N/8), 1, 1]`               | 256   |
//!
//! All three are correct for any K; K % 8 == 0 (every surya-2 shape) takes the 16-byte
//! vector / `cp.async` path, anything else a scalar path.
pub const BODY: &str = r#"
// ---------------------------------------------------------------------------------------
// gemm_mm_f16: y[M,N] (+)= x[M,K] · W[N,K]^T on m16n8k16 tensor cores.
//
// Block tile 128 tokens x 128 outputs, k step 32, 8 warps in a 4 (tokens) x 2 (outputs)
// grid, 32 x 64 per warp — the cnn_igemm fragment code (ldmatrix A and B, row padding 40
// halves so the 8 ldmatrix rows land in distinct banks). Two smem stages:
//   * W (already f16) arrives by 16-byte cp.async with zero-fill past N / K;
//   * x is f32 in global, so it is register-staged: the float4 loads for step k+1 issue
//     before the mma of step k and are converted + stored after it, hiding their latency
//     the way the async copy does.
// Blocks are raster-grouped 8 token-tiles at a time so one wave shares both the x rows and
// the W rows it needs in L2; plain row-major order re-streams all of W from DRAM once per
// token tile.
// ---------------------------------------------------------------------------------------
#define GF_BM 128
#define GF_BN 128
#define GF_BK 32
#define GF_SROW 40
#define GF_GROUP 8

extern "C" __global__ void __launch_bounds__(256) gemm_mm_f16(const float* x, const __half* w,
    float* y, int K, int N, int accum, int M) {
    __shared__ __align__(16) __half sA[2][GF_BM * GF_SROW];
    __shared__ __align__(16) __half sB[2][GF_BN * GF_SROW];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, R = lane >> 2, Q = lane & 3;
    // grouped raster: grid is (nN, nM); remap so consecutive blocks walk GF_GROUP token tiles
    const int nN = gridDim.x, nM = gridDim.y;
    const int lin = blockIdx.y * nN + blockIdx.x;
    const int per_group = GF_GROUP * nN;
    const int first_m = (lin / per_group) * GF_GROUP;
    const int gsz = min(nM - first_m, GF_GROUP);
    const int in_g = lin % per_group;
    const int m0 = (first_m + in_g % gsz) * GF_BM;
    const int n0 = (in_g / gsz) * GF_BN;
    const int wm = warp & 3, wn = warp >> 2;
    const bool vec = (K & 7) == 0;
    const int nk = (K + GF_BK - 1) / GF_BK;

    float acc[2][8][4];
    #pragma unroll
    for (int a = 0; a < 2; a++)
        #pragma unroll
        for (int b = 0; b < 8; b++)
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

    // x tile: 128 rows x 32 k = 1024 float4, 4 per thread; thread -> (row idx>>3, k4 idx&7)
    float4 xa[4];
    auto load_x = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256, r = idx >> 3, k = k0 + (idx & 7) * 4, m = m0 + r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m < M) {
                const float* p = x + (long)m * K + k;
                if (vec) {
                    if (k < K) v = *(const float4*)p;
                } else {
                    if (k < K) v.x = p[0];
                    if (k + 1 < K) v.y = p[1];
                    if (k + 2 < K) v.z = p[2];
                    if (k + 3 < K) v.w = p[3];
                }
            }
            xa[i] = v;
        }
    };
    auto store_x = [&](int st) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256, r = idx >> 3, c = (idx & 7) * 4;
            __half2 lo = __floats2half2_rn(xa[i].x, xa[i].y);
            __half2 hi = __floats2half2_rn(xa[i].z, xa[i].w);
            uint2 u;
            u.x = *(unsigned*)&lo;
            u.y = *(unsigned*)&hi;
            *(uint2*)(&sA[st][r * GF_SROW + c]) = u;
        }
    };
    // W tile: 128 rows x 32 halves = 512 16-byte chunks, 2 per thread
    auto load_w = [&](int st, int k0) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * 256, r = idx >> 2, seg = idx & 3, k = k0 + seg * 8, n = n0 + r;
            __half* d = &sB[st][r * GF_SROW + seg * 8];
            if (vec) {
                int bytes = (n < N && k < K) ? 16 : 0;
                const __half* src = w + (bytes ? (long)n * K + k : 0);
                unsigned dst = (unsigned)__cvta_generic_to_shared(d);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                             :: "r"(dst), "l"(src), "r"(bytes));
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    d[j] = (n < N && k + j < K) ? w[(long)n * K + k + j] : __float2half(0.f);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    load_w(0, 0);
    load_x(0);
    store_x(0);
    for (int kt = 0; kt < nk; kt++) {
        const int cur = kt & 1;
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const bool more = kt + 1 < nk;
        if (more) { load_w(cur ^ 1, (kt + 1) * GF_BK); load_x((kt + 1) * GF_BK); }
        const __half* A = sA[cur];
        const __half* B = sB[cur];
        #pragma unroll
        for (int ks = 0; ks < GF_BK; ks += 16) {
            unsigned a[2][4], b[8][2];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const __half* ap = A + (wm * 32 + mi * 16 + (lane & 15)) * GF_SROW + ks + (lane >> 4) * 8;
                unsigned sa = (unsigned)__cvta_generic_to_shared(ap);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3]) : "r"(sa));
            }
            #pragma unroll
            for (int ni = 0; ni < 8; ni += 2) {
                const __half* bp = B + (wn * 64 + ni * 8 + (lane >> 4) * 8 + (lane & 7)) * GF_SROW
                    + ks + ((lane >> 3) & 1) * 8;
                unsigned sb = (unsigned)__cvta_generic_to_shared(bp);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(b[ni][0]), "=r"(b[ni][1]), "=r"(b[ni + 1][0]), "=r"(b[ni + 1][1]) : "r"(sb));
            }
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 8; ni++)
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]), "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[ni][0]), "r"(b[ni][1]));
        }
        // the stage being refilled was last read in step kt-1, and every warp has passed this
        // step's barrier since, so writing it now cannot race a reader
        if (more) store_x(cur ^ 1);
    }

    // C fragment reg l: token row (l/2)*8 + R, output col 2Q + (l&1)
    const bool pair = (N & 1) == 0;   // float2 stores need an 8-byte-aligned m*N + n
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            int m = m0 + wm * 32 + mi * 16 + hh * 8 + R;
            if (m >= M) continue;
            float* yr = y + (long)m * N;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                int n = n0 + wn * 64 + ni * 8 + 2 * Q;
                float v0 = acc[mi][ni][hh * 2], v1 = acc[mi][ni][hh * 2 + 1];
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

// ---------------------------------------------------------------------------------------
// gemv_f16 / gemv_m_f16: one warp per weight row, 8 rows per 256-thread block (Metal's
// geometry). The weight row is streamed with 16-byte non-caching loads (8 halves per lane
// per step, 512 halves per warp step), unrolled so each lane keeps several loads in flight:
// one outstanding 16-byte load per lane caps a naive row-per-warp GEMV well under the
// bandwidth roof. x is small and reused by every row, so it takes the normal cached path.
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ uint4 gf_ld_stream(const __half* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

// dot of 8 halves (packed in a uint4) with 8 floats at xp
__device__ __forceinline__ float gf_dot8(uint4 h, const float* xp, float s) {
    float4 x0 = *(const float4*)xp, x1 = *(const float4*)(xp + 4);
    float2 a = __half22float2(*(__half2*)&h.x), b = __half22float2(*(__half2*)&h.y);
    float2 c = __half22float2(*(__half2*)&h.z), d = __half22float2(*(__half2*)&h.w);
    s = fmaf(a.x, x0.x, s); s = fmaf(a.y, x0.y, s); s = fmaf(b.x, x0.z, s); s = fmaf(b.y, x0.w, s);
    s = fmaf(c.x, x1.x, s); s = fmaf(c.y, x1.y, s); s = fmaf(d.x, x1.z, s); s = fmaf(d.y, x1.w, s);
    return s;
}

// MM rows of x against one weight row; p[m] per-lane partials (not yet warp-reduced).
template <int MM>
__device__ __forceinline__ void gf_rowdot(const float* x, const __half* row, int K, int lane,
                                          float* p) {
    #pragma unroll
    for (int m = 0; m < MM; m++) p[m] = 0.f;
    if ((K & 7) == 0) {
        const int K8 = K >> 3;
        int c = lane;
        // 4 independent 16-byte loads in flight per lane
        for (; c + 96 < K8; c += 128) {
            uint4 h0 = gf_ld_stream(row + 8 * c), h1 = gf_ld_stream(row + 8 * (c + 32));
            uint4 h2 = gf_ld_stream(row + 8 * (c + 64)), h3 = gf_ld_stream(row + 8 * (c + 96));
            #pragma unroll
            for (int m = 0; m < MM; m++) {
                const float* xm = x + (long)m * K;
                p[m] = gf_dot8(h0, xm + 8 * c, p[m]);
                p[m] = gf_dot8(h1, xm + 8 * (c + 32), p[m]);
                p[m] = gf_dot8(h2, xm + 8 * (c + 64), p[m]);
                p[m] = gf_dot8(h3, xm + 8 * (c + 96), p[m]);
            }
        }
        for (; c < K8; c += 32) {
            uint4 h = gf_ld_stream(row + 8 * c);
            #pragma unroll
            for (int m = 0; m < MM; m++) p[m] = gf_dot8(h, x + (long)m * K + 8 * c, p[m]);
        }
    } else {
        for (int k = lane; k < K; k += 32) {
            float wv = __half2float(row[k]);
            #pragma unroll
            for (int m = 0; m < MM; m++) p[m] = fmaf(wv, x[(long)m * K + k], p[m]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) gemv_f16(const float* x, const __half* w,
    float* y, int K, int N) {
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (n >= N) return;
    float p[1];
    gf_rowdot<1>(x, w + (long)n * K, K, lane, p);
    float s = warp_sum(p[0]);
    if (lane == 0) y[n] = s;
}

template <int MM>
__device__ __forceinline__ void gf_rows(const float* x, const __half* row, float* y, int K,
                                        int N, int n, int lane) {
    float p[MM];
    gf_rowdot<MM>(x, row, K, lane, p);
    #pragma unroll
    for (int m = 0; m < MM; m++) {
        float s = warp_sum(p[m]);
        if (lane == 0) y[(long)m * N + n] = s;
    }
}

// Split form for the larger M: a warp owns RR weight rows over 1/RR of K (the block still
// covers 8 rows with 8 warps), so every x value a lane loads feeds RR rows instead of one.
// At one row per warp a lane reads 2·M bytes of x per byte of weight — 16 at M = 8 — and
// that L1 traffic, not DRAM, caps M = 8 at half the bandwidth roof. The RR·M partials meet
// in shared memory. K % 8 == 0 only (the 16-byte path).
template <int MM, int RR>
__device__ __forceinline__ void gf_split(const float* x, const __half* w, float* y, int K, int N,
                                         float* red) {
    constexpr int G = 8 / RR;   // row groups per block; RR K-segments each
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = warp % G, s = warp / G;
    const int nb = blockIdx.x * 8 + g * RR;
    const int K8 = K >> 3, L = (K8 + RR - 1) / RR;
    const int c0 = s * L, c1 = min(K8, c0 + L);
    const __half* rows[RR];
    #pragma unroll
    for (int r = 0; r < RR; r++) rows[r] = w + (long)min(nb + r, N - 1) * K;  // past N: discarded
    float p[RR][MM];
    #pragma unroll
    for (int r = 0; r < RR; r++)
        #pragma unroll
        for (int m = 0; m < MM; m++) p[r][m] = 0.f;
    #pragma unroll 1
    for (int c = c0 + lane; c < c1; c += 32) {
        uint4 h[RR];
        #pragma unroll
        for (int r = 0; r < RR; r++) h[r] = gf_ld_stream(rows[r] + 8 * c);
        #pragma unroll
        for (int m = 0; m < MM; m++) {
            const float* xp = x + (long)m * K + 8 * c;
            #pragma unroll
            for (int r = 0; r < RR; r++) p[r][m] = gf_dot8(h[r], xp, p[r][m]);
        }
    }
    #pragma unroll
    for (int r = 0; r < RR; r++)
        #pragma unroll
        for (int m = 0; m < MM; m++) {
            float v = warp_sum(p[r][m]);
            if (lane == 0) red[(warp * RR + r) * MM + m] = v;
        }
    __syncthreads();
    // row j of the block = group j / RR, row j % RR; sum its RR K-segments
    for (int t = threadIdx.x; t < 8 * MM; t += blockDim.x) {
        const int j = t / MM, m = t % MM, n = blockIdx.x * 8 + j;
        const int gg = j / RR, r = j % RR;
        float v = 0.f;
        #pragma unroll
        for (int ss = 0; ss < RR; ss++) v += red[((ss * G + gg) * RR + r) * MM + m];
        if (n < N) y[(long)m * N + n] = v;
    }
    __syncthreads();   // red is reused by the next 8-row chunk
}

// RR = 2 measured best on the RTX 3060: M = 8 over the lm_head went 183 -> 255 GB/s. RR = 4
// was slower at every shape (the RR*MM accumulators crowd out the loads in flight), as was
// unrolling the k loop — one 16-byte load per row per iteration keeps occupancy up.
#define GF_SPLIT_RR 2

// M-row GEMV: each weight row is read once and produces M outputs (y[m*N + n]). Metal caps
// M at 8 (its accumulator array); here M > 8 runs in chunks of 8, each one a re-read of W.
// M >= 5 with K % 8 == 0 takes the split form above; otherwise one weight row per warp.
extern "C" __global__ void __launch_bounds__(256) gemv_m_f16(const float* x, const __half* w,
    float* y, int K, int N, int M) {
    if (M >= 5 && (K & 7) == 0) {
        __shared__ float red[8 * GF_SPLIT_RR * 8];   // (warp, row, token) partials
        for (int m0 = 0; m0 < M; m0 += 8) {
            const float* xm = x + (long)m0 * K;
            float* ym = y + (long)m0 * N;
            switch (min(M - m0, 8)) {
                case 1: gf_split<1, GF_SPLIT_RR>(xm, w, ym, K, N, red); break;
                case 2: gf_split<2, GF_SPLIT_RR>(xm, w, ym, K, N, red); break;
                case 3: gf_split<3, GF_SPLIT_RR>(xm, w, ym, K, N, red); break;
                case 4: gf_split<4, GF_SPLIT_RR>(xm, w, ym, K, N, red); break;
                case 5: gf_split<5, GF_SPLIT_RR>(xm, w, ym, K, N, red); break;
                case 6: gf_split<6, GF_SPLIT_RR>(xm, w, ym, K, N, red); break;
                case 7: gf_split<7, GF_SPLIT_RR>(xm, w, ym, K, N, red); break;
                default: gf_split<8, GF_SPLIT_RR>(xm, w, ym, K, N, red); break;
            }
        }
        return;
    }
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (n >= N) return;
    const __half* row = w + (long)n * K;
    for (int m0 = 0; m0 < M; m0 += 8) {
        const float* xm = x + (long)m0 * K;
        float* ym = y + (long)m0 * N;
        switch (min(M - m0, 8)) {
            case 1: gf_rows<1>(xm, row, ym, K, N, n, lane); break;
            case 2: gf_rows<2>(xm, row, ym, K, N, n, lane); break;
            case 3: gf_rows<3>(xm, row, ym, K, N, n, lane); break;
            case 4: gf_rows<4>(xm, row, ym, K, N, n, lane); break;
            case 5: gf_rows<5>(xm, row, ym, K, N, n, lane); break;
            case 6: gf_rows<6>(xm, row, ym, K, N, n, lane); break;
            case 7: gf_rows<7>(xm, row, ym, K, N, n, lane); break;
            default: gf_rows<8>(xm, row, ym, K, N, n, lane); break;
        }
    }
}
"#;

pub const NAMES: &[&str] = &["gemm_mm_f16", "gemv_f16", "gemv_m_f16"];
