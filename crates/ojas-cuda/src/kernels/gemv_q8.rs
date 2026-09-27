//! The Q8 GEMV family: CUDA twins of `ojas-metal`'s `gemv_q8` group.
//!
//! These read the row-quantized Q8 layout the decoder uses for weights: a row is `K` plain
//! `int8` values plus one `float` scale per output row. That is not GGUF's `block_q8_0` (32
//! values with a per-block half scale), which `gemv.rs` handles. The two must stay apart: the
//! decoder's requant path produces this layout, and mixing the readers halves accuracy silently
//! rather than failing.
//!
//! Structure follows Metal, since the canonical entry names and argument order are the
//! cross-backend contract (`ojas-core/src/kernel.rs`): one warp per output row, `char4`/`float4`
//! loads striding by the warp width, a warp reduction, then the row scale. `simd_sum` becomes
//! `warp_sum` and `threadgroup_position_in_grid` becomes `blockIdx.x`; nothing else differs, and
//! a divergence here is a correctness bug on one backend that the other will not see.
//!
//! Arguments mirror the Metal buffer order with the `constant uint&` slots hoisted into the
//! trailing int list, which is the CUDA dialect's convention (`gemv.rs`).

/// `ffn_act` and the char4/float4 dot helper, shared by every entry below.
pub const HELPERS: &str = r#"
// Metal's ffn_act: act==1 is GeLU (tanh approximation, clamped), otherwise SiLU.
__device__ __forceinline__ float ffn_act(float g, unsigned int act) {
    if (act == 1u) {
        float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
        inner = fminf(fmaxf(inner, -30.0f), 30.0f);
        return 0.5f * g * (1.0f + tanhf(inner));
    }
    return g / (1.0f + __expf(-g));
}

// dot(float4(char4), float4) — the inner product Metal gets from its vector types.
__device__ __forceinline__ float dot_c4(const char4 w, const float4 x) {
    return (float)w.x * x.x + (float)w.y * x.y + (float)w.z * x.z + (float)w.w * x.w;
}

// One warp reduces one row of a Q8 weight matrix against x. K must be a multiple of 4.
__device__ __forceinline__ float q8_row_dot(const char* w, const float* x, int K, int lane) {
    const char4* row = (const char4*)w;
    const float4* xv = (const float4*)x;
    int K4 = K >> 2;
    float p = 0.0f;
    for (int k = lane; k < K4; k += 32) p += dot_c4(row[k], xv[k]);
    return warp_sum(p);
}
"#;

pub const BODY: &str = r#"
// y[n] = scale[n] * dot(w[n], x)
extern "C" __global__ void gemv_q8(const float* x, const char* w, float* y,
                                   const float* scale, int K, int N) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * warps + warp;
    if (n >= N) return;
    float p = q8_row_dot(w + (long)n * (long)K, x, K, lane);
    if (lane == 0) y[n] = p * scale[n];
}

// y[n] += scale[n] * dot(w[n], x) — residual accumulation without a second pass.
extern "C" __global__ void gemv_q8_accum(const float* x, const char* w, float* y,
                                         const float* scale, int K, int N) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * warps + warp;
    if (n >= N) return;
    float p = q8_row_dot(w + (long)n * (long)K, x, K, lane);
    if (lane == 0) y[n] += p * scale[n];
}

// y[n] = scale[n] * dot(w[n], x) + bias[n]
extern "C" __global__ void gemv_q8_bias(const float* x, const char* w, float* y,
                                        const float* scale, const float* bias, int K, int N) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * warps + warp;
    if (n >= N) return;
    float p = q8_row_dot(w + (long)n * (long)K, x, K, lane);
    if (lane == 0) y[n] = p * scale[n] + bias[n];
}

// Split-K: one row per block, each warp taking a different stripe of K, combined in shared
// memory. Metal added this because a single-warp row on a long-K / low-N shape (ffn_down,
// N=896 K=4864) was memory-starved; the same arithmetic applies here.
extern "C" __global__ void gemv_q8_ksplit(const float* x, const char* w, float* y,
                                          const float* scale, int K, int N) {
    // static, not `extern __shared__`: KernelRuntime::dispatch always launches with zero
    // dynamic shared bytes, so a dynamic array here has no storage at all. 32 warps is the
    // most a 1024-thread block can hold.
    __shared__ float part[32];
    int nsg = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x;
    if (n >= N) return;
    const char4* row = (const char4*)(w + (long)n * (long)K);
    const float4* xv = (const float4*)x;
    int K4 = K >> 2;
    float p = 0.0f;
    for (int k = warp * 32 + lane; k < K4; k += nsg * 32) p += dot_c4(row[k], xv[k]);
    p = warp_sum(p);
    if (lane == 0) part[warp] = p;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < nsg; i++) s += part[i];
        y[n] = s * scale[n];
    }
}

// Fused gate/up SwiGLU: out[n] = act(gate[n]) * up[n], both rows read once.
extern "C" __global__ void ffn_gu_q8(const float* x, const char* wg, const char* wu, float* out,
                                     const float* sg, const float* su, int K, int N, int act) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * warps + warp;
    if (n >= N) return;
    const char4* rg = (const char4*)(wg + (long)n * (long)K);
    const char4* ru = (const char4*)(wu + (long)n * (long)K);
    const float4* xv = (const float4*)x;
    int K4 = K >> 2;
    float pg = 0.0f, pu = 0.0f;
    for (int k = lane; k < K4; k += 32) {
        float4 xf = xv[k];
        pg += dot_c4(rg[k], xf);
        pu += dot_c4(ru[k], xf);
    }
    pg = warp_sum(pg);
    pu = warp_sum(pu);
    if (lane == 0) out[n] = ffn_act(pg * sg[n], (unsigned int)act) * (pu * su[n]);
}

// Fused q/k/v projection: one launch covers Nq + 2*Nkv rows, each warp picking its region.
// bias may be null (Qwen3/Gemma have none; Qwen2 does).
extern "C" __global__ void qkv_q8(const float* x, const char* wq, const char* wk, const char* wv,
                                  float* yq, float* yk, float* yv,
                                  const float* sq, const float* sk, const float* sv,
                                  const float* bq, const float* bk, const float* bv,
                                  int K, int Nq, int Nkv) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * warps + warp;
    int total = Nq + 2 * Nkv;
    if (n >= total) return;
    const char* w; const float* scale; const float* bias; float* y; int r;
    if (n < Nq)            { w = wq; scale = sq; bias = bq; y = yq; r = n; }
    else if (n < Nq + Nkv) { w = wk; scale = sk; bias = bk; y = yk; r = n - Nq; }
    else                   { w = wv; scale = sv; bias = bv; y = yv; r = n - Nq - Nkv; }
    float p = q8_row_dot(w + (long)r * (long)K, x, K, lane);
    if (lane == 0) y[r] = p * scale[r] + (bias ? bias[r] : 0.0f);
}

// Embedding gather straight out of Q8 storage — no dequantised copy of the table.
extern "C" __global__ void embed_q8(const char* table, const float* scale, float* out,
                                    int token, int D) {
    const char* row = table + (long)token * (long)D;
    float s = scale[token];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < D; i += gridDim.x * blockDim.x) {
        out[i] = (float)row[i] * s;
    }
}
// ---- GGUF block_q8_0, read directly, integer dot product -------------------------------------
//
// The kernels above read the engine's row-Q8 layout, which costs a repack at load and 2.9 % of
// the top logit against f32. These read `block_q8_0` as the file stores it (34 bytes per 32
// values: an f16 scale then 32 int8) and multiply it against activations quantised by
// `quantize_q8_1` / `rmsnorm_q8`, so the inner loop is `__dp4a` on packed int8 rather than f32
// fma. That is how llama.cpp gets its throughput, and it removes the repack entirely.
//
// Four lanes per block, not one. Giving each lane a whole 32-value block and striding by 32 puts
// adjacent lanes 34*32 = 1088 bytes apart: every lane of the warp misses into a different sector
// and the row read costs 32 transactions instead of a handful. Splitting each block across four
// lanes makes the warp cover 8 consecutive blocks, 272 contiguous bytes, per iteration. Same
// arithmetic, one memory pattern the coalescer can serve.
//
// The weight row is not 4-byte aligned (34 bytes per block) so a quad cannot be an `int` load,
// but `b + 2` is always even, so it is two 16-bit loads rather than four 8-bit ones. The
// activation side is always 4-byte aligned and loads as an `int`.

__device__ __forceinline__ int q80_wquad(const unsigned char* p) {
    const unsigned short* h = (const unsigned short*)p;      // p is even; see above
    return (int)h[0] | ((int)h[1] << 16);
}

// One warp against one row of `block_q8_0` weights; lane 0 holds the row dot (warp_sum is a
// shfl_down reduction, like Metal simd_sum).
__device__ __forceinline__ float q80_row_dp4a(const unsigned char* row, const signed char* q8,
                                              const float* d8, int nblk, int lane) {
    int sub = lane & 3;              // which 8 of the block's 32 values
    float acc = 0.0f;
    for (int blk = lane >> 2; blk < nblk; blk += 8) {
        const unsigned char* b = row + (long)blk * 34;
        float dw = __half2float(*(const __half*)b);
        const unsigned char* wq = b + 2 + sub * 8;
        const int* xq = (const int*)(q8 + blk * 32 + sub * 8);
        int sumi = __dp4a(q80_wquad(wq), xq[0], 0);
        sumi = __dp4a(q80_wquad(wq + 4), xq[1], sumi);
        acc += dw * d8[blk] * (float)sumi;
    }
    return warp_sum(acc);
}

extern "C" __global__ void gemv_q80_dp4a(const unsigned char* w, const signed char* q8,
                                         const float* d8, float* y, int K, int N) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * warps + warp;
    if (n >= N) return;
    int nblk = K >> 5;
    float acc = q80_row_dp4a(w + (long)n * (long)nblk * 34, q8, d8, nblk, lane);
    if (lane == 0) y[n] = acc;
}

// y[n] = dot + bias[n] — Qwen2 carries q/k/v bias, and dropping it sends the logits to noise.
extern "C" __global__ void gemv_q80_dp4a_bias(const unsigned char* w, const signed char* q8,
                                              const float* d8, float* y, const float* bias,
                                              int K, int N) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * warps + warp;
    if (n >= N) return;
    int nblk = K >> 5;
    float acc = q80_row_dp4a(w + (long)n * (long)nblk * 34, q8, d8, nblk, lane);
    if (lane == 0) y[n] = acc + bias[n];
}

// y[n] += dot — the residual add for o_proj and ffn_down, so no second pass over the stream.
extern "C" __global__ void gemv_q80_dp4a_accum(const unsigned char* w, const signed char* q8,
                                               const float* d8, float* y, int K, int N) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * warps + warp;
    if (n >= N) return;
    int nblk = K >> 5;
    float acc = q80_row_dp4a(w + (long)n * (long)nblk * 34, q8, d8, nblk, lane);
    if (lane == 0) y[n] += acc;
}

// Fused q/k/v: one launch covers Nq + 2*Nkv rows against the one quantised activation. `bias`
// may be null. Saves two launches per layer, which at 24 layers is 48 per token.
extern "C" __global__ void qkv_q80_dp4a(const unsigned char* wq_, const unsigned char* wk_,
                                        const unsigned char* wv_, const signed char* q8,
                                        const float* d8, float* q, float* k, float* v,
                                        const float* bias, int K, int Nq, int Nkv) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * warps + warp;
    int total = Nq + 2 * Nkv;
    if (n >= total) return;
    int nblk = K >> 5;
    long rowbytes = (long)nblk * 34;
    const unsigned char* w; float* out; int row; int boff;
    if (n < Nq)              { w = wq_; out = q; row = n;             boff = 0; }
    else if (n < Nq + Nkv)   { w = wk_; out = k; row = n - Nq;        boff = Nq; }
    else                     { w = wv_; out = v; row = n - Nq - Nkv;  boff = Nq + Nkv; }
    float acc = q80_row_dp4a(w + (long)row * rowbytes, q8, d8, nblk, lane);
    if (lane == 0) out[row] = bias ? acc + bias[boff + row] : acc;
}

// Fused SwiGLU gate/up: out[n] = silu(gate[n]) * up[n], one launch, the activation read once.
// act==1 selects GeLU, matching `ffn_gu_q8` and Metal's ffn_act.
extern "C" __global__ void ffn_gu_q80_dp4a(const unsigned char* wg, const unsigned char* wu,
                                           const signed char* q8, const float* d8, float* out,
                                           int K, int N, int act) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * warps + warp;
    if (n >= N) return;
    int nblk = K >> 5;
    long rowbytes = (long)nblk * 34;
    float g = q80_row_dp4a(wg + (long)n * rowbytes, q8, d8, nblk, lane);
    float u = q80_row_dp4a(wu + (long)n * rowbytes, q8, d8, nblk, lane);
    if (lane == 0) out[n] = ffn_act(g, act) * u;
}

// Embedding gather straight out of `block_q8_0` storage: no dequantised table, no repack, and
// the table is usually the largest tensor in a small model.
// Device-token variant: the id comes from `ctl[0]` instead of a launch argument, so a whole
// decode step can be captured as a CUDA graph and replayed without re-recording per token.
// `argmax_ctl` is what writes it.
extern "C" __global__ void embed_q80_g(const unsigned char* table, float* out,
                                       const int* ctl, int d) {
    int nblk = d >> 5;
    const unsigned char* row = table + (long)ctl[0] * (long)nblk * 34;
    for (int blk = blockIdx.x * blockDim.x + threadIdx.x; blk < nblk;
         blk += gridDim.x * blockDim.x) {
        const unsigned char* b = row + (long)blk * 34;
        float dw = __half2float(*(const __half*)b);
        const signed char* q = (const signed char*)(b + 2);
        #pragma unroll
        for (int i = 0; i < 32; i++) out[blk * 32 + i] = (float)q[i] * dw;
    }
}

extern "C" __global__ void embed_q80(const unsigned char* table, float* out, int token, int d) {
    int nblk = d >> 5;
    const unsigned char* row = table + (long)token * (long)nblk * 34;
    for (int blk = blockIdx.x * blockDim.x + threadIdx.x; blk < nblk;
         blk += gridDim.x * blockDim.x) {
        const unsigned char* b = row + (long)blk * 34;
        float dw = __half2float(*(const __half*)b);
        const signed char* q = (const signed char*)(b + 2);
        #pragma unroll
        for (int i = 0; i < 32; i++) out[blk * 32 + i] = (float)q[i] * dw;
    }
}
"#;

/// Canonical entry names registered from this family.
pub const NAMES: &[&str] = &[
    "gemv_q8",
    "gemv_q8_accum",
    "gemv_q8_bias",
    "gemv_q8_ksplit",
    "ffn_gu_q8",
    "qkv_q8",
    "embed_q8",
    "gemv_q80_dp4a",
    "gemv_q80_dp4a_bias",
    "gemv_q80_dp4a_accum",
    "qkv_q80_dp4a",
    "ffn_gu_q80_dp4a",
    "embed_q80",
    "embed_q80_g",
];
