// Kernel bodies are byte-identical to the reference CUDA backend; entry names are
// canonicalized to the Metal-canonical ones.
// Compiles against kernels::PRELUDE (cuda_fp16 + warp reduction helpers).
pub const BODY: &str = r#"
// Scalar reference GEMV (block_q4_0 packing: byte j = elem j low | elem j+16 high).
// Correct but slow (~10% BW); kept for cross-checking the fast dp4a kernel.
extern "C" __global__ void gemv_q4(const float* x, const unsigned char* w4,
                                   float* y, const __half* scale, int K, int N) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * warps + warp;
    if (n >= N) return;
    int nblk = K >> 5;
    const unsigned char* row = w4 + (long)n * (K / 2);
    const __half* rsc = scale + (long)n * nblk;
    float p = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        const unsigned char* bl = row + b * 16;
        const float* xb = x + b * 32;
        float axq = 0.0f, sx = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned char by = bl[j];
            float x0 = xb[j], x1 = xb[j + 16];        // block_q4_0 element order
            axq += x0 * (float)(by & 0xF) + x1 * (float)(by >> 4);
            sx += x0 + x1;
        }
        p += __half2float(rsc[b]) * (axq - 8.0f * sx);
    }
    p = warp_sum(p);
    if (lane == 0) y[n] = p;
}

// Quantize an f32 activation vector to q8_1: per 32-block, d8 = amax/127, int8
// quants, and d8sum = d8*Σq8 (so the symmetric −8 offset applies once per block
// in the dot). One warp per 32-block. Done once per activation, reused across
// every weight matrix that consumes it (reference quantize.cu pattern).
extern "C" __global__ void quantize_q8_1(const float* x, signed char* q8,
                                         float* d8, float* d8sum, int K) {
    int blk = blockIdx.x, lane = threadIdx.x;   // block_dim = 32
    int base = blk * 32;
    float v = x[base + lane];
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float d = a / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    int qi = max(-127, min(127, __float2int_rn(v * id)));
    q8[base + lane] = (signed char)qi;
    float s = (float)qi;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) { d8[blk] = d; d8sum[blk] = d * s; }
}

// Per-token q8 quant for the batched mma8 GEMM. x[M*K] token-major → q8[M*K] +
// d8[M] (one scale per token = max|x|/127). Grid.x = M (one block/token, 256 thr).
extern "C" __global__ void quantize_q8_pertoken(const float* x, signed char* q8,
                                                float* d8, int K) {
    int tok = blockIdx.x, tid = threadIdx.x;   // block_dim = 256
    const float* xr = x + (long)tok * K;
    signed char* qr = q8 + (long)tok * K;
    __shared__ float sm[256];
    float a = 0.f;
    for (int i = tid; i < K; i += 256) a = fmaxf(a, fabsf(xr[i]));
    sm[tid] = a; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]); __syncthreads(); }
    float d = sm[0] / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    for (int i = tid; i < K; i += 256) {
        int qi = max(-127, min(127, __float2int_rn(xr[i] * id)));
        qr[i] = (signed char)qi;
    }
    if (tid == 0) d8[tok] = d;
}

// W4A8 dp4a decode GEMV (reference mmvq Q4_0 path). 128 threads = 4 warps
// cooperate on one output row (grid.x = N). Two threads per 32-block (vdr=2):
// each loads 2 weight int32 (coalesced, adjacent lanes → adjacent 8-byte words)
// + matching q8 int32, and does 4 __dp4a. Nibbles stay unsigned 0..15; the
// symmetric −8 folds into 4*d8sum per thread (2 threads → 8*d8sum/block).
extern "C" __global__ void __launch_bounds__(128, 8)
gemv_q4_dp4a(const int* w, const __half* scale,
             const signed char* q8, const float* d8,
             const float* d8sum, float* y, int K, int N) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x, nblk = K >> 5;
    const int* wrow = w + (long)row * (K / 8);   // K/8 int32 per row
    const __half* srow = scale + (long)row * nblk;
    int sub = tid & 1;                           // 0 → weight words 0,1 ; 1 → 2,3
    float tmp = 0.0f;
    for (int blk = tid >> 1; blk < nblk; blk += 64) {
        const int* wq = wrow + blk * 4;
        const int* q8blk = (const int*)(q8 + blk * 32);
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int wi = sub * 2 + i;
            int v = wq[wi];
            int vi0 = (v >> 0) & 0x0f0f0f0f;      // 4 low nibbles  → elems [wi*4 .. +4)
            int vi1 = (v >> 4) & 0x0f0f0f0f;      // 4 high nibbles → elems [wi*4+16 .. +4)
            sumi = __dp4a(vi0, q8blk[wi], sumi);
            sumi = __dp4a(vi1, q8blk[wi + 4], sumi);
        }
        tmp += __half2float(srow[blk]) * ((float)sumi * d8[blk] - 4.0f * d8sum[blk]);
    }
    // warp-reduce each of the 4 warps, then combine via 4-slot shared (1 sync).
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) tmp += __shfl_down_sync(0xffffffffu, tmp, o);
    __shared__ float sm[4];
    if ((tid & 31) == 0) sm[tid >> 5] = tmp;
    __syncthreads();
    if (tid == 0) y[row] = sm[0] + sm[1] + sm[2] + sm[3];
}

// Thread-count-tunable decode GEMV (same math as gemv_q4_dp4a but block size is
// dynamic: stride = blockDim.x/2, warps combined via sm[nthr/32]). Lets the decode
// autotuner pick the fastest threads/row ∈ {64,128,256} per (N,K) shape/device.
extern "C" __global__ void __launch_bounds__(256)
gemv_q4_dp4a_t(const int* w, const __half* scale,
               const signed char* q8, const float* d8,
               const float* d8sum, float* y, int K, int N) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x, nblk = K >> 5, nthr = blockDim.x;
    const int* wrow = w + (long)row * (K / 8);
    const __half* srow = scale + (long)row * nblk;
    int sub = tid & 1;
    float tmp = 0.0f;
    for (int blk = tid >> 1; blk < nblk; blk += nthr >> 1) {
        const int* wq = wrow + blk * 4;
        const int* q8blk = (const int*)(q8 + blk * 32);
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int wi = sub * 2 + i;
            int v = wq[wi];
            sumi = __dp4a((v >> 0) & 0x0f0f0f0f, q8blk[wi], sumi);
            sumi = __dp4a((v >> 4) & 0x0f0f0f0f, q8blk[wi + 4], sumi);
        }
        tmp += __half2float(srow[blk]) * ((float)sumi * d8[blk] - 4.0f * d8sum[blk]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) tmp += __shfl_down_sync(0xffffffffu, tmp, o);
    __shared__ float sm[32];
    if ((tid & 31) == 0) sm[tid >> 5] = tmp;
    __syncthreads();
    if (tid == 0) { float s = 0.f; int nw = nthr >> 5; for (int i = 0; i < nw; i++) s += sm[i]; y[row] = s; }
}

// Batched Q4 prefill GEMV: reads each weight row once and applies it to M token
// activations (weight bandwidth amortized M×). One warp per output row; each
// lane owns whole 32-blocks and dp4a's the block against all M tokens' q8.
// q8/d8/d8sum are [M, K] / [M, nblk]; y is [M, N] row-major (token-major).
#define PREFILL_MAXM 32

extern "C" __global__ void gemm_q4_dp4a(const int* w, const __half* scale,
    const signed char* q8, const float* d8, const float* d8sum,
    float* y, int K, int N, int M) {
    int row = blockIdx.x;
    if (row >= N) return;
    int lane = threadIdx.x & 31;                 // one warp per row (block_dim=32)
    int nblk = K >> 5;
    const int* wrow = w + (long)row * (K / 8);
    const __half* srow = scale + (long)row * nblk;
    float acc[PREFILL_MAXM];
    #pragma unroll
    for (int m = 0; m < PREFILL_MAXM; m++) acc[m] = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        const int* wq = wrow + b * 4;            // 4 weight int32 = one 32-block
        float wsc = __half2float(srow[b]);
        int vi0[4], vi1[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) { int v = wq[i]; vi0[i] = (v >> 0) & 0x0f0f0f0f; vi1[i] = (v >> 4) & 0x0f0f0f0f; }
        for (int m = 0; m < M; m++) {            // reuse the loaded weight across M tokens
            const int* q8b = (const int*)(q8 + (long)m * K + b * 32);
            int sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) { sumi = __dp4a(vi0[i], q8b[i], sumi); sumi = __dp4a(vi1[i], q8b[i + 4], sumi); }
            acc[m] += wsc * ((float)sumi * d8[(long)m * nblk + b] - 8.0f * d8sum[(long)m * nblk + b]);
        }
    }
    for (int m = 0; m < M; m++) {
        float t = acc[m];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
        if (lane == 0) y[(long)m * N + row] = t;
    }
}

// Combined kernel: ldmatrix (l-form) + 2-block K-chunk (halves syncs, 8KB
// shared keeps ~6 blocks/SM). 64 rows × 64 tokens/block, 8 warps. Grid=(N/64,M/64).
#define KC2 2

// mma18 = mma14 but B loaded manually from shared (no ldmatrix); the reference notes
// load_generic beats load_ldmatrix for B. B operand mirrors A: token=tt*8+lane/4,
// kint32=bb*8+(lane%4)|+4. Removes 8 ldmatrix/block. Grid=(N/128,M/64).
extern "C" __global__ void __launch_bounds__(256, 2) gemm_q4_s8_mma18(
    const int* w, const __half* scale, const signed char* q8, const float* d8,
    float* y, int K, int N, int M) {
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int rowbase = blockIdx.x * 128, tokbase = blockIdx.y * 64;
    int trow = rowbase + warp * 16;
    int nblk = K >> 5, nchunk = nblk / KC2;
    const int ST = KC2 * 8, SZA = 128 * KC2 * 8, SZB = 64 * KC2 * 8;
    __shared__ __align__(16) int sA[2 * 128 * KC2 * 8];
    __shared__ __align__(16) int sB[2 * 64 * KC2 * 8];
    __shared__ float sSA[2 * 128 * KC2];
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;
    int q = lane / 4, r = lane % 4;
    #define STA18(ci, buf) do { \
        int kc = (ci) * KC2; int* dA = sA + (buf) * SZA; float* dS = sSA + (buf) * (128 * KC2); \
        for (int idx = tid; idx < 128 * KC2 * 4; idx += 256) { \
            int rr = idx / (KC2 * 4), bb = (idx / 4) % KC2, kqsx = idx & 3; \
            int qs0 = ((const int*)(w + (long)(rowbase + rr) * (K / 8)))[(kc + bb) * 4 + kqsx]; \
            int base = rr * ST + bb * 8 + kqsx; \
            dA[base]     = __vsubss4((qs0 >> 0) & 0x0f0f0f0f, 0x08080808); \
            dA[base + 4] = __vsubss4((qs0 >> 4) & 0x0f0f0f0f, 0x08080808); \
        } \
        for (int idx = tid; idx < 128 * KC2; idx += 256) { \
            int rr = idx / KC2, bb = idx % KC2; \
            dS[idx] = __half2float(scale[(long)(rowbase + rr) * nblk + kc + bb]); \
        } \
    } while (0)
    #define STB18(ci, buf) do { \
        int kc = (ci) * KC2; int* dB = sB + (buf) * SZB; \
        for (int idx = tid; idx < 64 * KC2 * 2; idx += 256) { \
            int unit = idx >> 1, half = idx & 1, tok = unit / KC2, bb = unit % KC2; \
            const signed char* src = q8 + (long)(tokbase + tok) * K + (kc + bb) * 32 + half * 16; \
            unsigned dst = (unsigned)__cvta_generic_to_shared(dB + tok * ST + bb * 8 + half * 4); \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); \
        } \
    } while (0)
    STA18(0, 0); STB18(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    for (int ci = 0; ci < nchunk; ci++) {
        int cur = ci & 1, nxt = cur ^ 1;
        if (ci + 1 < nchunk) { STB18(ci + 1, nxt); asm volatile("cp.async.commit_group;\n" ::); STA18(ci + 1, nxt); }
        const int* cA = sA + cur * SZA;
        const int* cB = sB + cur * SZB;
        const float* cS = sSA + cur * (128 * KC2);
        #pragma unroll
        for (int bb = 0; bb < KC2; bb++) {
            const int* xsA = cA + warp * 16 * ST + bb * 8 + (lane % 16) * ST + (lane / 16) * 4;
            int A0, A1, A2, A3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(A0), "=r"(A1), "=r"(A2), "=r"(A3) : "l"(xsA));
            float dAlo = cS[(warp * 16 + q) * KC2 + bb];
            float dAhi = cS[(warp * 16 + 8 + q) * KC2 + bb];
            #pragma unroll
            for (int tt = 0; tt < 8; tt++) {
                int B0 = cB[(tt * 8 + q) * ST + bb * 8 + r];
                int B1 = cB[(tt * 8 + q) * ST + bb * 8 + r + 4];
                int D0 = 0, D1 = 0, D2 = 0, D3 = 0;
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
                    : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));
                acc[tt * 4 + 0] += (float)D0 * dAlo;
                acc[tt * 4 + 1] += (float)D1 * dAlo;
                acc[tt * 4 + 2] += (float)D2 * dAhi;
                acc[tt * 4 + 3] += (float)D3 * dAhi;
            }
        }
        if (ci + 1 < nchunk) asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
    #undef STA18
    #undef STB18
    #pragma unroll
    for (int tt = 0; tt < 8; tt++) {
        int gilo = lane / 4, gihi = 8 + lane / 4, gj0 = (lane % 4) * 2, gj1 = gj0 + 1;
        float dB0 = d8[tokbase + tt * 8 + gj0];
        float dB1 = d8[tokbase + tt * 8 + gj1];
        y[(long)(tokbase + tt * 8 + gj0) * N + trow + gilo] = acc[tt * 4 + 0] * dB0;
        y[(long)(tokbase + tt * 8 + gj1) * N + trow + gilo] = acc[tt * 4 + 1] * dB1;
        y[(long)(tokbase + tt * 8 + gj0) * N + trow + gihi] = acc[tt * 4 + 2] * dB0;
        y[(long)(tokbase + tt * 8 + gj1) * N + trow + gihi] = acc[tt * 4 + 3] * dB1;
    }
}

// mma19: the full MMQ port. 128×128 tile (mmq_y=mmq_x=128, nwarps=8, acc[64]) +
// opt-in dynamic shared (2 blocks/SM) + cp.async activations (double-buffered) +
// manual B loads (no ldmatrix) + A reused across all 128
// tokens. Grid=(N/128, M/128). Needs set_attribute(MAX_DYNAMIC_SHARED, >=48896)
// and shared_mem_bytes=48896.
extern "C" __global__ void __launch_bounds__(256, 2) gemm_q4_s8_mma19(
    const int* w, const __half* scale, const signed char* q8, const float* d8,
    float* y, int K, int N, int M) {
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int rowbase = blockIdx.x * 128, tokbase = blockIdx.y * 128;
    int trow = rowbase + warp * 16;
    int nblk = K >> 5, nchunk = nblk / KC2;
    const int ST = KC2 * 8, SZA = 128 * KC2 * 8, SZB = 128 * KC2 * 8;
    extern __shared__ __align__(16) int smem19[];
    int* sA = smem19;                     // 2*SZA
    int* sB = sA + 2 * SZA;               // 2*SZB
    float* sSA = (float*)(sB + 2 * SZB);  // 2*128*KC2
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.f;
    int q = lane / 4, r = lane % 4;
    #define STA19(ci, buf) do { \
        int kc = (ci) * KC2; int* dA = sA + (buf) * SZA; float* dS = sSA + (buf) * (128 * KC2); \
        for (int idx = tid; idx < 128 * KC2 * 4; idx += 256) { \
            int rr = idx / (KC2 * 4), bb = (idx / 4) % KC2, kqsx = idx & 3; \
            int qs0 = ((const int*)(w + (long)(rowbase + rr) * (K / 8)))[(kc + bb) * 4 + kqsx]; \
            int base = rr * ST + bb * 8 + kqsx; \
            dA[base]     = __vsubss4((qs0 >> 0) & 0x0f0f0f0f, 0x08080808); \
            dA[base + 4] = __vsubss4((qs0 >> 4) & 0x0f0f0f0f, 0x08080808); \
        } \
        for (int idx = tid; idx < 128 * KC2; idx += 256) { \
            int rr = idx / KC2, bb = idx % KC2; \
            dS[idx] = __half2float(scale[(long)(rowbase + rr) * nblk + kc + bb]); \
        } \
    } while (0)
    #define STB19(ci, buf) do { \
        int kc = (ci) * KC2; int* dB = sB + (buf) * SZB; \
        for (int idx = tid; idx < 128 * KC2 * 2; idx += 256) { \
            int unit = idx >> 1, half = idx & 1, tok = unit / KC2, bb = unit % KC2; \
            const signed char* src = q8 + (long)(tokbase + tok) * K + (kc + bb) * 32 + half * 16; \
            unsigned dst = (unsigned)__cvta_generic_to_shared(dB + tok * ST + bb * 8 + half * 4); \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); \
        } \
    } while (0)
    STA19(0, 0); STB19(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    for (int ci = 0; ci < nchunk; ci++) {
        int cur = ci & 1, nxt = cur ^ 1;
        if (ci + 1 < nchunk) { STB19(ci + 1, nxt); asm volatile("cp.async.commit_group;\n" ::); STA19(ci + 1, nxt); }
        const int* cA = sA + cur * SZA;
        const int* cB = sB + cur * SZB;
        const float* cS = sSA + cur * (128 * KC2);
        #pragma unroll
        for (int bb = 0; bb < KC2; bb++) {
            const int* xsA = cA + warp * 16 * ST + bb * 8 + (lane % 16) * ST + (lane / 16) * 4;
            int A0, A1, A2, A3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(A0), "=r"(A1), "=r"(A2), "=r"(A3) : "l"(xsA));
            float dAlo = cS[(warp * 16 + q) * KC2 + bb];
            float dAhi = cS[(warp * 16 + 8 + q) * KC2 + bb];
            #pragma unroll
            for (int tt = 0; tt < 16; tt++) {
                int B0 = cB[(tt * 8 + q) * ST + bb * 8 + r];
                int B1 = cB[(tt * 8 + q) * ST + bb * 8 + r + 4];
                int D0 = 0, D1 = 0, D2 = 0, D3 = 0;
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
                    : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));
                acc[tt * 4 + 0] += (float)D0 * dAlo;
                acc[tt * 4 + 1] += (float)D1 * dAlo;
                acc[tt * 4 + 2] += (float)D2 * dAhi;
                acc[tt * 4 + 3] += (float)D3 * dAhi;
            }
        }
        if (ci + 1 < nchunk) asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
    #undef STA19
    #undef STB19
    #pragma unroll
    for (int tt = 0; tt < 16; tt++) {
        int gilo = lane / 4, gihi = 8 + lane / 4, gj0 = (lane % 4) * 2, gj1 = gj0 + 1;
        float dB0 = d8[tokbase + tt * 8 + gj0];
        float dB1 = d8[tokbase + tt * 8 + gj1];
        y[(long)(tokbase + tt * 8 + gj0) * N + trow + gilo] = acc[tt * 4 + 0] * dB0;
        y[(long)(tokbase + tt * 8 + gj1) * N + trow + gilo] = acc[tt * 4 + 1] * dB1;
        y[(long)(tokbase + tt * 8 + gj0) * N + trow + gihi] = acc[tt * 4 + 2] * dB0;
        y[(long)(tokbase + tt * 8 + gj1) * N + trow + gihi] = acc[tt * 4 + 3] * dB1;
    }
}

// mma19_sk = mma19 + split-K over the K reduction axis. For small matrices (small N
// and few tokens) the plain (N/128, M/128) grid launches too few blocks to fill all
// SMs: N=1024, M=128 gives only 8 blocks on a 28-SM card (29% util). This variant
// adds a grid.z = SPLITK dimension: each z-block reduces only its slice [ci0,ci1) of
// the K-chunks and atomicAdds its scaled partial into y, which must be pre-zeroed.
// SPLITK× more blocks → full occupancy. The per-token act
// scale dB factors out of the K sum, so scaling each partial by dB before the add
// is exact. Requires nchunk % gridDim.z == 0.
extern "C" __global__ void __launch_bounds__(256, 2) gemm_q4_s8_mma19_sk(
    const int* w, const __half* scale, const signed char* q8, const float* d8,
    float* y, int K, int N, int M) {
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int rowbase = blockIdx.x * 128, tokbase = blockIdx.y * 128;
    int trow = rowbase + warp * 16;
    int nblk = K >> 5, nchunk = nblk / KC2;
    int ncz = nchunk / gridDim.z, ci0 = blockIdx.z * ncz, ci1 = ci0 + ncz;
    const int ST = KC2 * 8, SZA = 128 * KC2 * 8, SZB = 128 * KC2 * 8;
    extern __shared__ __align__(16) int smem19[];
    int* sA = smem19;                     // 2*SZA
    int* sB = sA + 2 * SZA;               // 2*SZB
    float* sSA = (float*)(sB + 2 * SZB);  // 2*128*KC2
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.f;
    int q = lane / 4, r = lane % 4;
    #define STA19S(ci, buf) do { \
        int kc = (ci) * KC2; int* dA = sA + (buf) * SZA; float* dS = sSA + (buf) * (128 * KC2); \
        for (int idx = tid; idx < 128 * KC2 * 4; idx += 256) { \
            int rr = idx / (KC2 * 4), bb = (idx / 4) % KC2, kqsx = idx & 3; \
            int qs0 = ((const int*)(w + (long)(rowbase + rr) * (K / 8)))[(kc + bb) * 4 + kqsx]; \
            int base = rr * ST + bb * 8 + kqsx; \
            dA[base]     = __vsubss4((qs0 >> 0) & 0x0f0f0f0f, 0x08080808); \
            dA[base + 4] = __vsubss4((qs0 >> 4) & 0x0f0f0f0f, 0x08080808); \
        } \
        for (int idx = tid; idx < 128 * KC2; idx += 256) { \
            int rr = idx / KC2, bb = idx % KC2; \
            dS[idx] = __half2float(scale[(long)(rowbase + rr) * nblk + kc + bb]); \
        } \
    } while (0)
    #define STB19S(ci, buf) do { \
        int kc = (ci) * KC2; int* dB = sB + (buf) * SZB; \
        for (int idx = tid; idx < 128 * KC2 * 2; idx += 256) { \
            int unit = idx >> 1, half = idx & 1, tok = unit / KC2, bb = unit % KC2; \
            const signed char* src = q8 + (long)(tokbase + tok) * K + (kc + bb) * 32 + half * 16; \
            unsigned dst = (unsigned)__cvta_generic_to_shared(dB + tok * ST + bb * 8 + half * 4); \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); \
        } \
    } while (0)
    int cur = 0;
    STA19S(ci0, 0); STB19S(ci0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    for (int ci = ci0; ci < ci1; ci++) {
        int nxt = cur ^ 1;
        if (ci + 1 < ci1) { STB19S(ci + 1, nxt); asm volatile("cp.async.commit_group;\n" ::); STA19S(ci + 1, nxt); }
        const int* cA = sA + cur * SZA;
        const int* cB = sB + cur * SZB;
        const float* cS = sSA + cur * (128 * KC2);
        #pragma unroll
        for (int bb = 0; bb < KC2; bb++) {
            const int* xsA = cA + warp * 16 * ST + bb * 8 + (lane % 16) * ST + (lane / 16) * 4;
            int A0, A1, A2, A3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(A0), "=r"(A1), "=r"(A2), "=r"(A3) : "l"(xsA));
            float dAlo = cS[(warp * 16 + q) * KC2 + bb];
            float dAhi = cS[(warp * 16 + 8 + q) * KC2 + bb];
            #pragma unroll
            for (int tt = 0; tt < 16; tt++) {
                int B0 = cB[(tt * 8 + q) * ST + bb * 8 + r];
                int B1 = cB[(tt * 8 + q) * ST + bb * 8 + r + 4];
                int D0 = 0, D1 = 0, D2 = 0, D3 = 0;
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
                    : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));
                acc[tt * 4 + 0] += (float)D0 * dAlo;
                acc[tt * 4 + 1] += (float)D1 * dAlo;
                acc[tt * 4 + 2] += (float)D2 * dAhi;
                acc[tt * 4 + 3] += (float)D3 * dAhi;
            }
        }
        if (ci + 1 < ci1) asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        cur = nxt;
    }
    #undef STA19S
    #undef STB19S
    #pragma unroll
    for (int tt = 0; tt < 16; tt++) {
        int gilo = lane / 4, gihi = 8 + lane / 4, gj0 = (lane % 4) * 2, gj1 = gj0 + 1;
        float dB0 = d8[tokbase + tt * 8 + gj0];
        float dB1 = d8[tokbase + tt * 8 + gj1];
        atomicAdd(&y[(long)(tokbase + tt * 8 + gj0) * N + trow + gilo], acc[tt * 4 + 0] * dB0);
        atomicAdd(&y[(long)(tokbase + tt * 8 + gj1) * N + trow + gilo], acc[tt * 4 + 1] * dB1);
        atomicAdd(&y[(long)(tokbase + tt * 8 + gj0) * N + trow + gihi], acc[tt * 4 + 2] * dB0);
        atomicAdd(&y[(long)(tokbase + tt * 8 + gj1) * N + trow + gihi], acc[tt * 4 + 3] * dB1);
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) gemm_s8_s8_mma19(
    const signed char* w8, const __half* scale, const signed char* q8, const float* d8,
    float* y, int K, int N, int M) {
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int rowbase = blockIdx.x * 128, tokbase = blockIdx.y * 128;
    int trow = rowbase + warp * 16;
    int nblk = K >> 5, nchunk = nblk / KC2;
    // ST padded +4 (KC2*8=16 is power-of-2 → ldmatrix-A + manual-B reads alias
    // shared banks 8-way; +4 → 20 spreads across 8 banks, still 16B-aligned).
    // (256,1) → 1 block/SM already, so the extra shared is free.
    const int ST = KC2 * 8 + 4, SZA = 128 * ST, SZB = 128 * ST;
    extern __shared__ __align__(16) int smemS8[];
    int* sA = smemS8;                     // 2*SZA
    int* sB = sA + 2 * SZA;               // 2*SZB
    float* sSA = (float*)(sB + 2 * SZB);  // 2*128*KC2
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.f;
    int q = lane / 4, r = lane % 4;
    #define STA_S8(ci, buf) do { \
        int kc = (ci) * KC2; int* dA = sA + (buf) * SZA; float* dS = sSA + (buf) * (128 * KC2); \
        for (int idx = tid; idx < 128 * KC2 * 2; idx += 256) { \
            int unit = idx >> 1, half = idx & 1, rr = unit / KC2, bb = unit % KC2; \
            const signed char* src = w8 + (long)(rowbase + rr) * K + (kc + bb) * 32 + half * 16; \
            unsigned dst = (unsigned)__cvta_generic_to_shared(dA + rr * ST + bb * 8 + half * 4); \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); \
        } \
        for (int idx = tid; idx < 128 * KC2; idx += 256) { \
            int rr = idx / KC2, bb = idx % KC2; \
            dS[idx] = __half2float(scale[(long)(rowbase + rr) * nblk + kc + bb]); \
        } \
    } while (0)
    #define STB_S8(ci, buf) do { \
        int kc = (ci) * KC2; int* dB = sB + (buf) * SZB; \
        for (int idx = tid; idx < 128 * KC2 * 2; idx += 256) { \
            int unit = idx >> 1, half = idx & 1, tok = unit / KC2, bb = unit % KC2; \
            const signed char* src = q8 + (long)(tokbase + tok) * K + (kc + bb) * 32 + half * 16; \
            unsigned dst = (unsigned)__cvta_generic_to_shared(dB + tok * ST + bb * 8 + half * 4); \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); \
        } \
    } while (0)
    STA_S8(0, 0); STB_S8(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    for (int ci = 0; ci < nchunk; ci++) {
        int cur = ci & 1, nxt = cur ^ 1;
        if (ci + 1 < nchunk) { STB_S8(ci + 1, nxt); STA_S8(ci + 1, nxt); asm volatile("cp.async.commit_group;\n" ::); }
        const int* cA = sA + cur * SZA;
        const int* cB = sB + cur * SZB;
        const float* cS = sSA + cur * (128 * KC2);
        #pragma unroll
        for (int bb = 0; bb < KC2; bb++) {
            const int* xsA = cA + warp * 16 * ST + bb * 8 + (lane % 16) * ST + (lane / 16) * 4;
            int A0, A1, A2, A3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(A0), "=r"(A1), "=r"(A2), "=r"(A3) : "l"(xsA));
            float dAlo = cS[(warp * 16 + q) * KC2 + bb];
            float dAhi = cS[(warp * 16 + 8 + q) * KC2 + bb];
            #pragma unroll
            for (int tt = 0; tt < 16; tt++) {
                int B0 = cB[(tt * 8 + q) * ST + bb * 8 + r];
                int B1 = cB[(tt * 8 + q) * ST + bb * 8 + r + 4];
                int D0 = 0, D1 = 0, D2 = 0, D3 = 0;
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
                    : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));
                acc[tt * 4 + 0] += (float)D0 * dAlo;
                acc[tt * 4 + 1] += (float)D1 * dAlo;
                acc[tt * 4 + 2] += (float)D2 * dAhi;
                acc[tt * 4 + 3] += (float)D3 * dAhi;
            }
        }
        if (ci + 1 < nchunk) asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
    #undef STA_S8
    #undef STB_S8
    #pragma unroll
    for (int tt = 0; tt < 16; tt++) {
        int gilo = lane / 4, gihi = 8 + lane / 4, gj0 = (lane % 4) * 2, gj1 = gj0 + 1;
        float dB0 = d8[tokbase + tt * 8 + gj0], dB1 = d8[tokbase + tt * 8 + gj1];
        y[(long)(tokbase + tt * 8 + gj0) * N + trow + gilo] = acc[tt * 4 + 0] * dB0;
        y[(long)(tokbase + tt * 8 + gj1) * N + trow + gilo] = acc[tt * 4 + 1] * dB1;
        y[(long)(tokbase + tt * 8 + gj0) * N + trow + gihi] = acc[tt * 4 + 2] * dB0;
        y[(long)(tokbase + tt * 8 + gj1) * N + trow + gihi] = acc[tt * 4 + 3] * dB1;
    }
}

// INT8 GEMM + SPLIT-K: gemm_s8_s8_mma19 with grid.z=SPLITK partitioning the K
// chunks across blocks (atomicAdd into a pre-zeroed y). For grid-starved small-M
// shapes (mp=128, N=1024 → only 8 blocks) this fills the SMs while keeping int8's
// speed + padded (ST=20) bank-conflict-free shared. Requires nchunk % gridDim.z==0.
extern "C" __global__ void __launch_bounds__(256, 1) gemm_s8_mma19_sk(
    const signed char* w8, const __half* scale, const signed char* q8, const float* d8,
    float* y, int K, int N, int M) {
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int rowbase = blockIdx.x * 128, tokbase = blockIdx.y * 128;
    int trow = rowbase + warp * 16;
    int nblk = K >> 5, nchunk = nblk / KC2;
    int ncz = nchunk / gridDim.z, ci0 = blockIdx.z * ncz, ci1 = ci0 + ncz;
    const int ST = KC2 * 8 + 4, SZA = 128 * ST, SZB = 128 * ST;
    extern __shared__ __align__(16) int smemSK[];
    int* sA = smemSK;
    int* sB = sA + 2 * SZA;
    float* sSA = (float*)(sB + 2 * SZB);
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.f;
    int q = lane / 4, r = lane % 4;
    #define STA_SK(ci, buf) do { \
        int kc = (ci) * KC2; int* dA = sA + (buf) * SZA; float* dS = sSA + (buf) * (128 * KC2); \
        for (int idx = tid; idx < 128 * KC2 * 2; idx += 256) { \
            int unit = idx >> 1, half = idx & 1, rr = unit / KC2, bb = unit % KC2; \
            const signed char* src = w8 + (long)(rowbase + rr) * K + (kc + bb) * 32 + half * 16; \
            unsigned dst = (unsigned)__cvta_generic_to_shared(dA + rr * ST + bb * 8 + half * 4); \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); \
        } \
        for (int idx = tid; idx < 128 * KC2; idx += 256) { \
            int rr = idx / KC2, bb = idx % KC2; \
            dS[idx] = __half2float(scale[(long)(rowbase + rr) * nblk + kc + bb]); \
        } \
    } while (0)
    #define STB_SK(ci, buf) do { \
        int kc = (ci) * KC2; int* dB = sB + (buf) * SZB; \
        for (int idx = tid; idx < 128 * KC2 * 2; idx += 256) { \
            int unit = idx >> 1, half = idx & 1, tok = unit / KC2, bb = unit % KC2; \
            const signed char* src = q8 + (long)(tokbase + tok) * K + (kc + bb) * 32 + half * 16; \
            unsigned dst = (unsigned)__cvta_generic_to_shared(dB + tok * ST + bb * 8 + half * 4); \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); \
        } \
    } while (0)
    int cur = 0;
    STA_SK(ci0, 0); STB_SK(ci0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    for (int ci = ci0; ci < ci1; ci++) {
        int nxt = cur ^ 1;
        if (ci + 1 < ci1) { STB_SK(ci + 1, nxt); STA_SK(ci + 1, nxt); asm volatile("cp.async.commit_group;\n" ::); }
        const int* cA = sA + cur * SZA;
        const int* cB = sB + cur * SZB;
        const float* cS = sSA + cur * (128 * KC2);
        #pragma unroll
        for (int bb = 0; bb < KC2; bb++) {
            const int* xsA = cA + warp * 16 * ST + bb * 8 + (lane % 16) * ST + (lane / 16) * 4;
            int A0, A1, A2, A3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(A0), "=r"(A1), "=r"(A2), "=r"(A3) : "l"(xsA));
            float dAlo = cS[(warp * 16 + q) * KC2 + bb];
            float dAhi = cS[(warp * 16 + 8 + q) * KC2 + bb];
            #pragma unroll
            for (int tt = 0; tt < 16; tt++) {
                int B0 = cB[(tt * 8 + q) * ST + bb * 8 + r];
                int B1 = cB[(tt * 8 + q) * ST + bb * 8 + r + 4];
                int D0 = 0, D1 = 0, D2 = 0, D3 = 0;
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
                    : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));
                acc[tt * 4 + 0] += (float)D0 * dAlo;
                acc[tt * 4 + 1] += (float)D1 * dAlo;
                acc[tt * 4 + 2] += (float)D2 * dAhi;
                acc[tt * 4 + 3] += (float)D3 * dAhi;
            }
        }
        if (ci + 1 < ci1) asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        cur = nxt;
    }
    #undef STA_SK
    #undef STB_SK
    #pragma unroll
    for (int tt = 0; tt < 16; tt++) {
        int gilo = lane / 4, gihi = 8 + lane / 4, gj0 = (lane % 4) * 2, gj1 = gj0 + 1;
        float dB0 = d8[tokbase + tt * 8 + gj0], dB1 = d8[tokbase + tt * 8 + gj1];
        atomicAdd(&y[(long)(tokbase + tt * 8 + gj0) * N + trow + gilo], acc[tt * 4 + 0] * dB0);
        atomicAdd(&y[(long)(tokbase + tt * 8 + gj1) * N + trow + gilo], acc[tt * 4 + 1] * dB1);
        atomicAdd(&y[(long)(tokbase + tt * 8 + gj0) * N + trow + gihi], acc[tt * 4 + 2] * dB0);
        atomicAdd(&y[(long)(tokbase + tt * 8 + gj1) * N + trow + gihi], acc[tt * 4 + 3] * dB1);
    }
}

// Batched W4A8 GEMV for small / non-128 N (alpha,beta): y[m*N+n] = d8[m] *
// Σ_b wscale[n,b]·(Σ_{k∈b} q8[m,k]·(nib−8)). One warp per (m,n). Per-token act.
extern "C" __global__ void gemv_q4_batched(const unsigned char* w4, const __half* scale,
    const signed char* q8, const float* d8, float* y, int K, int N, int M) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int m = warp / N, n = warp % N;
    if (m >= M) return;
    int nblk = K >> 5;
    const unsigned char* row = w4 + (long)n * (K / 2);
    const __half* rsc = scale + (long)n * nblk;
    const signed char* qr = q8 + (long)m * K;
    float p = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        const unsigned char* bl = row + b * 16;
        const signed char* xb = qr + b * 32;
        int axq = 0, sx = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned char by = bl[j];
            int x0 = xb[j], x1 = xb[j + 16];
            axq += x0 * (int)(by & 0xF) + x1 * (int)(by >> 4);
            sx += x0 + x1;
        }
        p += __half2float(rsc[b]) * (float)(axq - 8 * sx);
    }
    p = warp_all_sum(p);
    if (lane == 0) y[(long)m * N + n] = p * d8[m];
}
"#;

/// Canonical entry names registered from this family.
pub const NAMES: &[&str] = &[
    "gemv_q4",
    "quantize_q8_1",
    "quantize_q8_pertoken",
    "gemv_q4_dp4a",
    "gemv_q4_dp4a_t",
    "gemm_q4_dp4a",
    "gemm_q4_s8_mma18",
    "gemm_q4_s8_mma19",
    "gemm_q4_s8_mma19_sk",
    "gemm_s8_s8_mma19",
    "gemm_s8_mma19_sk",
    "gemv_q4_batched",
];
