// Kernel bodies are byte-identical to the reference CUDA backend; entry names are
// canonicalized to the Metal-canonical ones.
// Compiles against kernels::PRELUDE (cuda_fp16 + warp reduction helpers).
pub const BODY: &str = r#"
// copy src[0..n) into dst[off..off+n)  (KV-cache append).
extern "C" __global__ void store_kv(float* dst, const float* src, int n, int off) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) dst[off + g] = src[g];
}

// store src[0..n) into dst at offset ctl[1]*kvdim (KV-cache append at pos).
extern "C" __global__ void store_kv_g(__half* dst, const float* src, const int* ctl,
                                      int n, int kvdim) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) dst[(long)ctl[1] * kvdim + g] = __float2half(src[g]);
}

// ---- KV context-reuse (llama's n_cache_reuse): shift a cached KV chunk to a new
// position so it can back a different absolute position in the next prompt. ----
//
// Copy a contiguous KV chunk of `n` positions (all kvdim elems each): the dst
// position (dst0+i) gets the src position (src0+i). Used to gather a cache chunk
// into scratch (src0=c0,dst0=0) and to scatter it back (src0=0,dst0=p0). Going
// through scratch avoids the in-place read/write race when src/dst ranges overlap
// (the common shift-left / sliding-window case).
extern "C" __global__ void kv_copy_off(const __half* src, __half* dst,
        int src0, int dst0, int n, int kvdim) {
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long)n * kvdim) return;
    int i = g / kvdim, e = g % kvdim;
    dst[(long)(dst0 + i) * kvdim + e] = src[(long)(src0 + i) * kvdim + e];
}

// Fused gather+re-RoPE (replaces separate gather-K, gather-V, rope passes): read a
// KV chunk `[c0, c0+n)` and write it to scratch — K re-RoPE'd by `delta`, V copied.
// One thread per (position i, kv-head kvh, dim d): every thread copies V[d]; threads
// with d < n_rot/2 additionally rotate the K pair (d, d+n_rot/2). grid = n*n_kv*hd.
extern "C" __global__ void kv_gather_rope(const __half* kc, const __half* vc,
        __half* sk, __half* sv, int c0, int n, int delta,
        int kvdim, int hd, int n_kv, int n_rot, float base) {
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)n * n_kv * hd;
    if (g >= total) return;
    int perh = n_kv * hd;
    int i = (int)(g / perh), rem = (int)(g % perh), kvh = rem / hd, d = rem % hd;
    long sbase = (long)(c0 + i) * kvdim + (long)kvh * hd;
    long dbase = (long)i * kvdim + (long)kvh * hd;
    sv[dbase + d] = vc[sbase + d];              // V: verbatim
    int rf = n_rot >> 1;
    if (d < rf) {                               // K: rotate pair (d, d+rf) by delta
        float freq = 1.0f / powf(base, 2.0f * (float)d / (float)n_rot);
        float ang = (float)delta * freq, cs = cosf(ang), sn = sinf(ang);
        float x0 = __half2float(kc[sbase + d]), x1 = __half2float(kc[sbase + d + rf]);
        sk[dbase + d]      = __float2half(x0 * cs - x1 * sn);
        sk[dbase + d + rf] = __float2half(x0 * sn + x1 * cs);
    } else if (d >= n_rot) {                     // K: any non-rotary tail, copy
        sk[dbase + d] = kc[sbase + d];
    }
}

// Fused scatter of a gathered chunk back into the cache at `[p0, p0+n)`, K and V in
// one pass. grid = n*kvdim.
extern "C" __global__ void kv_scatter2(const __half* sk, const __half* sv,
        __half* kc, __half* vc, int p0, int n, int kvdim) {
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long)n * kvdim) return;
    int i = (int)(g / kvdim), e = (int)(g % kvdim);
    long d = (long)(p0 + i) * kvdim + e, s = (long)i * kvdim + e;
    kc[d] = sk[s]; vc[d] = sv[s];
}

// Re-RoPE a gathered K scratch chunk by `delta` positions, in place. RoPE composes:
// a key stored at absolute pos p is R(p)*K_raw, so relocating it to pos p+delta is
// applying the extra rotation R(delta)*(stored K). NeoX pairs (j, j+n_rot/2),
// freq = base^(-2j/n_rot). V carries no RoPE, so it is copied unchanged. `delta` may
// be negative (shift-left). grid over (position i, kv-head kvh, pair j<n_rot/2).
extern "C" __global__ void kv_rope_shift(__half* sk, int n, int delta,
        int kvdim, int hd, int n_kv, int n_rot, float base) {
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int rf = n_rot >> 1;
    long total = (long)n * n_kv * rf;
    if (g >= total) return;
    int per = n_kv * rf;
    int i = g / per, rem = (int)(g % per), kvh = rem / rf, j = rem % rf;
    long b = (long)i * kvdim + (long)kvh * hd;
    float freq = 1.0f / powf(base, 2.0f * (float)j / (float)n_rot);
    float ang = (float)delta * freq, cs = cosf(ang), sn = sinf(ang);
    float x0 = __half2float(sk[b + j]), x1 = __half2float(sk[b + j + rf]);
    sk[b + j]      = __float2half(x0 * cs - x1 * sn);
    sk[b + j + rf] = __float2half(x0 * sn + x1 * cs);
}

// GQA decode attention: one block per query head. Scores q·k over all `seq`
// cached positions → softmax → weighted sum of v. group = nq/nkv (GQA ratio),
// so kv-head = head/group. KV cache is f32 here (bring-up; the Metal path uses
// f16 to halve KV bandwidth). Bounded seq (<=4096) in shared `sc`.
extern "C" __global__ void attention_short(const float* q, const float* kc,
    const float* vc, float* out, int hd, int kvdim, int seq, int group, float scale) {
    __shared__ float sc[4096];
    __shared__ float part[32];
    __shared__ float qsh[512];
    int head = blockIdx.x, lid = threadIdx.x, ts = blockDim.x;
    int lane = lid & 31, sg = lid >> 5, nsg = ts >> 5;
    int kvh = head / group;
    const float* qh = q + head * hd;
    for (int i = lid; i < hd; i += ts) qsh[i] = qh[i];
    __syncthreads();
    // scores: one warp per position stripe
    for (int t = sg; t < seq; t += nsg) {
        const float* kt = kc + (long)t * kvdim + kvh * hd;
        float sv = 0.0f;
        for (int i = lane; i < hd; i += 32) sv += qsh[i] * kt[i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sv += __shfl_down_sync(0xffffffffu, sv, o);
        if (lane == 0) sc[t] = sv * scale;
    }
    __syncthreads();
    // softmax max
    float lmax = -1e30f;
    for (int t = lid; t < seq; t += ts) lmax = fmaxf(lmax, sc[t]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
    if (lane == 0) part[sg] = lmax;
    __syncthreads();
    float mx = -1e30f;
    for (int j = 0; j < nsg; j++) mx = fmaxf(mx, part[j]);
    // exp + sum
    float lsum = 0.0f;
    for (int t = lid; t < seq; t += ts) { float e = expf(sc[t] - mx); sc[t] = e; lsum += e; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
    if (lane == 0) part[sg] = lsum;
    __syncthreads();
    float sum = 0.0f;
    for (int j = 0; j < nsg; j++) sum += part[j];
    // weighted sum of v
    for (int i = lid; i < hd; i += ts) {
        float acc = 0.0f;
        for (int t = 0; t < seq; t++) acc += sc[t] * vc[(long)t * kvdim + kvh * hd + i];
        out[head * hd + i] = acc / sum;
    }
}

// Graph variant: seq = ctl[1]+1 read from device.
extern "C" __global__ void attention_short_g(const float* q, const __half* kc,
    const __half* vc, float* out, const int* ctl, int hd, int kvdim, int group, float scale) {
    __shared__ float sc[4096];
    __shared__ float part[32];
    __shared__ float qsh[512];
    int head = blockIdx.x, lid = threadIdx.x, ts = blockDim.x;
    int lane = lid & 31, sg = lid >> 5, nsg = ts >> 5;
    int seq = ctl[1] + 1;
    int kvh = head / group;
    const float* qh = q + head * hd;
    for (int i = lid; i < hd; i += ts) qsh[i] = qh[i];
    __syncthreads();
    for (int t = sg; t < seq; t += nsg) {
        const __half* kt = kc + (long)t * kvdim + kvh * hd;
        float sv = 0.0f;
        for (int i = lane; i < hd; i += 32) sv += qsh[i] * __half2float(kt[i]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sv += __shfl_down_sync(0xffffffffu, sv, o);
        if (lane == 0) sc[t] = sv * scale;
    }
    __syncthreads();
    float lmax = -1e30f;
    for (int t = lid; t < seq; t += ts) lmax = fmaxf(lmax, sc[t]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
    if (lane == 0) part[sg] = lmax;
    __syncthreads();
    float mx = -1e30f;
    for (int j = 0; j < nsg; j++) mx = fmaxf(mx, part[j]);
    float lsum = 0.0f;
    for (int t = lid; t < seq; t += ts) { float e = expf(sc[t] - mx); sc[t] = e; lsum += e; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
    if (lane == 0) part[sg] = lsum;
    __syncthreads();
    float sum = 0.0f;
    for (int j = 0; j < nsg; j++) sum += part[j];
    for (int i = lid; i < hd; i += ts) {
        float acc = 0.0f;
        for (int t = 0; t < seq; t++) acc += sc[t] * __half2float(vc[(long)t * kvdim + kvh * hd + i]);
        out[head * hd + i] = acc / sum;
    }
}

// Store M tokens' k/v (src[M*kvdim]) into cache at positions pos_base..pos_base+M.
extern "C" __global__ void store_kv_m(__half* dst, const float* src,
    int kvdim, int pos_base, int M) {
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long)M * kvdim) return;
    int m = g / kvdim, i = g % kvdim;
    dst[(long)(pos_base + m) * kvdim + i] = __float2half(src[g]);
}

extern "C" __global__ void store_kv_m_g(__half* dst, const float* src, const int* pctl,
    int kvdim, int M) {
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long)M * kvdim) return;
    int m = g / kvdim, i = g % kvdim;
    dst[(long)(pctl[0] + m) * kvdim + i] = __float2half(src[g]);
}

// Causal GQA attention for M query positions. grid=(n_head, M), block=256.
// Query m (pos = pos_base + m) attends to cache positions 0..pos.
extern "C" __global__ void attention_m_short(const float* q, const float* kc,
    const float* vc, float* out, int hd, int kvdim, int qdim, int pos_base, int M,
    int group, float scale) {
    __shared__ float sc[4096];
    __shared__ float part[32];
    __shared__ float qsh[512];
    int head = blockIdx.x, m = blockIdx.y, lid = threadIdx.x, ts = blockDim.x;
    int lane = lid & 31, sg = lid >> 5, nsg = ts >> 5;
    int seq = pos_base + m + 1;
    int kvh = head / group;
    const float* qh = q + (long)m * qdim + head * hd;
    for (int i = lid; i < hd; i += ts) qsh[i] = qh[i];
    __syncthreads();
    for (int t = sg; t < seq; t += nsg) {
        const float* kt = kc + (long)t * kvdim + kvh * hd;
        float sv = 0.0f;
        for (int i = lane; i < hd; i += 32) sv += qsh[i] * kt[i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sv += __shfl_down_sync(0xffffffffu, sv, o);
        if (lane == 0) sc[t] = sv * scale;
    }
    __syncthreads();
    float lmax = -1e30f;
    for (int t = lid; t < seq; t += ts) lmax = fmaxf(lmax, sc[t]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
    if (lane == 0) part[sg] = lmax;
    __syncthreads();
    float mx = -1e30f;
    for (int j = 0; j < nsg; j++) mx = fmaxf(mx, part[j]);
    float lsum = 0.0f;
    for (int t = lid; t < seq; t += ts) { float e = expf(sc[t] - mx); sc[t] = e; lsum += e; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
    if (lane == 0) part[sg] = lsum;
    __syncthreads();
    float sum = 0.0f;
    for (int j = 0; j < nsg; j++) sum += part[j];
    for (int i = lid; i < hd; i += ts) {
        float acc = 0.0f;
        for (int t = 0; t < seq; t++) acc += sc[t] * vc[(long)t * kvdim + kvh * hd + i];
        out[(long)m * qdim + head * hd + i] = acc / sum;
    }
}
"#;

/// Canonical entry names registered from this family.
pub const NAMES: &[&str] = &[
    "store_kv",
    "store_kv_g",
    "kv_copy_off",
    "kv_gather_rope",
    "kv_scatter2",
    "kv_rope_shift",
    "attention_short",
    "attention_short_g",
    "store_kv_m",
    "store_kv_m_g",
    "attention_m_short",
];
