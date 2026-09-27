// Kernel bodies are byte-identical to the reference CUDA backend; entry names are
// canonicalized to the Metal-canonical ones.
// Compiles against kernels::PRELUDE (cuda_fp16 + warp reduction helpers).
pub const BODY: &str = r#"
// RMSNorm over `d` elements, one block (Metal single-threadgroup kernel:
// warp-reduce then an 8-way shared tree). out = x * rsqrt(mean(x^2)+eps) * w.
extern "C" __global__ void rmsnorm(const float* x, const float* w, float* out,
                                   int d, float eps) {
    __shared__ float part[32];
    int lid = threadIdx.x, ts = blockDim.x, lane = lid & 31, sg = lid >> 5;
    float s = 0.0f;
    for (int i = lid; i < d; i += ts) { float v = x[i]; s += v * v; }
    s = warp_sum(s);
    if (lane == 0) part[sg] = s;
    __syncthreads();
    int nsg = ts >> 5;
    float tot = 0.0f;
    for (int j = 0; j < nsg; j++) tot += part[j];
    float inv = rsqrtf(tot / (float)d + eps);
    for (int i = lid; i < d; i += ts) out[i] = x[i] * inv * w[i];
}

// Fused RMSNorm + q8_1 quant: the norm output feeds only GEMVs, so quantize it
// in-kernel and never materialize the f32 vector. One block; phase 1 = rms into
// shared, phase 2 = 8 warps quantize the d/32 blocks (d<=4096). Removes a whole
// quantize kernel + the f32 round-trip at every norm site (~65/token).
extern "C" __global__ void rmsnorm_q8(const float* x, const float* w,
    signed char* q8, float* d8, float* d8sum, int d, float eps) {
    __shared__ float sh[4096];
    __shared__ float part[32];
    int lid = threadIdx.x, ts = blockDim.x, lane = lid & 31, sg = lid >> 5;
    float s = 0.0f;
    for (int i = lid; i < d; i += ts) { float v = x[i]; s += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) part[sg] = s;
    __syncthreads();
    int nsg = ts >> 5;
    float tot = 0.0f;
    for (int j = 0; j < nsg; j++) tot += part[j];
    float inv = rsqrtf(tot / (float)d + eps);
    for (int i = lid; i < d; i += ts) sh[i] = x[i] * inv * w[i];
    __syncthreads();
    int nblk = d >> 5;
    for (int b = sg; b < nblk; b += nsg) {
        float v = sh[b * 32 + lane];
        float a = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
        float dd = a / 127.0f;
        float id = dd > 0.0f ? 1.0f / dd : 0.0f;
        int qi = max(-127, min(127, __float2int_rn(v * id)));
        q8[b * 32 + lane] = (signed char)qi;
        float ss = (float)qi;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) { d8[b] = dd; d8sum[b] = dd * ss; }
    }
}

// act = silu(gate)*up  (SwiGLU second half). silu_mul and swiglu are identical
// math; both names exist because the Metal runner uses each in a different path.
extern "C" __global__ void silu_mul(const float* gate, const float* up,
                                    float* act, int total) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;
    float v = gate[g];
    act[g] = (v / (1.0f + expf(-v))) * up[g];
}

extern "C" __global__ void swiglu(const float* gate, const float* up,
                                  float* out, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    float v = gate[g];
    out[g] = (v / (1.0f + expf(-v))) * up[g];
}

// NeoX-style RoPE on one buffer: pairs (i, i+hd/2) within each head are rotated
// by pos*freq(i). One thread per (head, i < hd/2).
extern "C" __global__ void rope(float* v, int hd, int pos, float base, int total) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int hf = hd / 2, head = gid / hf, i = gid % hf, b = head * hd;
    float freq = 1.0f / powf(base, 2.0f * (float)i / (float)hd);
    float ang = (float)pos * freq, s = sinf(ang), c = cosf(ang);
    float x0 = v[b + i], x1 = v[b + hf + i];
    v[b + i] = x0 * c - x1 * s;
    v[b + hf + i] = x0 * s + x1 * c;
}

extern "C" __global__ void add_inplace(float* x, const float* y, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) x[g] += y[g];
}

extern "C" __global__ void mul_scalar(float* x, int n, float s) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) x[g] *= s;
}

extern "C" __global__ void copy_buf(float* dst, const float* src, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) dst[g] = src[g];
}

// GPU argmax over `n` logits, one block; shared-mem tree reduction. out[0]=index.
extern "C" __global__ void argmax(const float* logits, unsigned int* out, int n) {
    __shared__ float vals[1024];
    __shared__ unsigned int idxs[1024];
    int lid = threadIdx.x, ts = blockDim.x;
    float bv = -1e30f; unsigned int bi = 0u;
    for (int i = lid; i < n; i += ts) { float x = logits[i]; if (x > bv) { bv = x; bi = (unsigned)i; } }
    vals[lid] = bv; idxs[lid] = bi;
    __syncthreads();
    for (int off = ts / 2; off > 0; off >>= 1) {
        if (lid < off && vals[lid + off] > vals[lid]) { vals[lid] = vals[lid + off]; idxs[lid] = idxs[lid + off]; }
        __syncthreads();
    }
    if (lid == 0) out[0] = idxs[0];
}

// Greedy sample straight into the control buffer: ctl[0] = argmax, ctl[1] += 1.
//
// This is what makes a decode step position-independent. With the token id and the position
// both in device memory, every launch in the step has arguments that never change, so the step
// records once as a CUDA graph and replays per token: no host readback between tokens, and the
// ~200 launches per token collapse into one graph launch. A host that wants the id can still
// read ctl[0], but decoding the next token does not require it.
extern "C" __global__ void argmax_ctl(const float* logits, int* ctl, int n) {
    __shared__ float vals[1024];
    __shared__ unsigned int idxs[1024];
    int lid = threadIdx.x, ts = blockDim.x;
    float bv = -1e30f; unsigned int bi = 0u;
    for (int i = lid; i < n; i += ts) { float x = logits[i]; if (x > bv) { bv = x; bi = (unsigned)i; } }
    vals[lid] = bv; idxs[lid] = bi;
    __syncthreads();
    for (int off = ts / 2; off > 0; off >>= 1) {
        if (lid < off && vals[lid + off] > vals[lid]) { vals[lid] = vals[lid + off]; idxs[lid] = idxs[lid + off]; }
        __syncthreads();
    }
    if (lid == 0) { ctl[0] = (int)idxs[0]; ctl[1] = ctl[1] + 1; }
}

// Fused NeoX rope on q and k in one launch, position from ctl[1]. Two launches per layer times
// 24 layers is 48 per token; this halves that. Lanes below `qtotal` rotate q, the rest k.
extern "C" __global__ void rope_qk_g(float* q, float* k, const int* ctl, int hd, int n_rot,
                                     float base, int qtotal, int ktotal) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= qtotal + ktotal) return;
    float* v = gid < qtotal ? q : k;
    int g = gid < qtotal ? gid : gid - qtotal;
    int pos = ctl[1];
    int rf = n_rot / 2, head = g / rf, j = g % rf, b = head * hd;
    float freq = 1.0f / powf(base, 2.0f * (float)j / (float)n_rot);
    float ang = (float)pos * freq, s = sinf(ang), c = cosf(ang);
    float x0 = v[b + j], x1 = v[b + rf + j];
    v[b + j] = x0 * c - x1 * s;
    v[b + rf + j] = x0 * s + x1 * c;
}

// Append both k and v to their f16 caches in one launch (`store_kv_g` twice, fused).
extern "C" __global__ void store_kv2_g(__half* kdst, __half* vdst, const float* ksrc,
                                       const float* vsrc, const int* ctl, int n, int kvdim) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    long off = (long)ctl[1] * kvdim + g;
    kdst[off] = __float2half(ksrc[g]);
    vdst[off] = __float2half(vsrc[g]);
}

// Q4 embedding lookup: dequant row `token` of token_embd into x[0..d].
// Q4 layout: block byte j = elem 2j low nibble | elem 2j+1 high nibble.
extern "C" __global__ void embed_q4(const unsigned char* emb, float* x, const __half* scale,
                                    int d, int token) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= d) return;
    int nblk = d / 32;
    const unsigned char* row = emb + (long)token * (d / 2);
    int b = gid / 32, within = gid % 32;
    int byte = within < 16 ? within : within - 16;   // block_q4_0: low=first16, high=last16
    float sc = __half2float(scale[(long)token * nblk + b]);
    unsigned char by = row[b * 16 + byte];
    int nib = within < 16 ? (by & 0xF) : (by >> 4);
    x[gid] = sc * ((float)nib - 8.0f);
}

// Partial NeoX RoPE: rotate only the first `n_rot` dims of each head (pairs
// (j, j+n_rot/2)), leaving dims [n_rot, hd) untouched. total = n_heads*(n_rot/2).
extern "C" __global__ void rope_partial(float* v, int hd, int n_rot, int pos,
                                        float base, int total) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int rf = n_rot / 2, head = gid / rf, j = gid % rf, b = head * hd;
    float freq = 1.0f / powf(base, 2.0f * (float)j / (float)n_rot);
    float ang = (float)pos * freq, s = sinf(ang), c = cosf(ang);
    float x0 = v[b + j], x1 = v[b + rf + j];
    v[b + j] = x0 * c - x1 * s;
    v[b + rf + j] = x0 * s + x1 * c;
}

extern "C" __global__ void embed_q4_g(const unsigned char* emb, float* x, const int* ctl,
                                      const __half* scale, int d) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= d) return;
    int token = ctl[0];
    int nblk = d / 32;
    const unsigned char* row = emb + (long)token * (d / 2);
    int b = gid / 32, within = gid % 32;
    int byte = within < 16 ? within : within - 16;
    float sc = __half2float(scale[(long)token * nblk + b]);
    unsigned char by = row[b * 16 + byte];
    int nib = within < 16 ? (by & 0xF) : (by >> 4);
    x[gid] = sc * ((float)nib - 8.0f);
}

extern "C" __global__ void rope_partial_g(float* v, const int* ctl, int hd, int n_rot,
                                          float base, int total) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int pos = ctl[1];
    int rf = n_rot / 2, head = gid / rf, j = gid % rf, b = head * hd;
    float freq = 1.0f / powf(base, 2.0f * (float)j / (float)n_rot);
    float ang = (float)pos * freq, s = sinf(ang), c = cosf(ang);
    float x0 = v[b + j], x1 = v[b + rf + j];
    v[b + j] = x0 * c - x1 * s;
    v[b + rf + j] = x0 * s + x1 * c;
}

// qwen35 gated attention: attn_q projects to per-head [q(hd) | gate(hd)] chunks
// (stride 2*hd). Split the contiguous q out of qfull. qfull rows [M,2*qdim].
extern "C" __global__ void qgate_split(const float* qfull, float* q,
                                       int hd, int qdim, int M) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= M * qdim) return;
    int m = g / qdim, r = g % qdim, h = r / hd, i = r % hd;
    q[g] = qfull[(long)m * (2 * qdim) + h * 2 * hd + i];
}

// attn[g] *= sigmoid(gate), gate = second hd of each 2*hd chunk of qfull.
extern "C" __global__ void gate_mul_sigmoid(float* attn, const float* qfull,
                                            int hd, int qdim, int M) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= M * qdim) return;
    int m = g / qdim, r = g % qdim, h = r / hd, i = r % hd;
    float z = qfull[(long)m * (2 * qdim) + h * 2 * hd + hd + i];
    attn[g] *= 1.0f / (1.0f + expf(-z));
}

// Fused gate_mul_sigmoid + q8 quant (output feeds only the attn_output GEMV). M=1.
extern "C" __global__ void gate_mul_sigmoid_q8(const float* attn, const float* qfull,
    signed char* q8, float* d8, float* d8sum, int hd, int qdim) {
    int blk = blockIdx.x, lane = threadIdx.x;
    int g = blk * 32 + lane;
    if (g >= qdim) return;
    int h = g / hd, i = g % hd;
    float z = qfull[h * 2 * hd + hd + i];
    float v = attn[g] * (1.0f / (1.0f + expf(-z)));
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float dd = a / 127.0f;
    float id = dd > 0.0f ? 1.0f / dd : 0.0f;
    int qi = max(-127, min(127, __float2int_rn(v * id)));
    q8[g] = (signed char)qi;
    float ssum = (float)qi;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssum += __shfl_xor_sync(0xffffffffu, ssum, o);
    if (lane == 0) { d8[blk] = dd; d8sum[blk] = dd * ssum; }
}

// Qwen3 QK-norm: per-head RMSNorm over head_dim on q and k, before RoPE.
// One warp per head; q heads [0,nq) use qw, k heads [nq,nq+nk) use kw.
extern "C" __global__ void qk_rmsnorm(float* vq, float* vk, const float* qw,
                                      const float* kw, int hd, int nq, int nk, float eps) {
    int head = blockIdx.x, tok = blockIdx.y, lane = threadIdx.x;  // 32 threads
    bool isq = head < nq;
    float* v = isq ? vq : vk;
    const float* w = isq ? qw : kw;
    int h = isq ? head : (head - nq);
    int rowdim = isq ? nq * hd : nk * hd;
    long base = (long)tok * rowdim + (long)h * hd;
    float ss = 0.0f;
    for (int i = lane; i < hd; i += 32) { float x = v[base + i]; ss += x * x; }
    ss = warp_all_sum(ss);
    float inv = rsqrtf(ss / (float)hd + eps);
    for (int i = lane; i < hd; i += 32) v[base + i] = v[base + i] * inv * w[i];
}

// Gated RMSNorm per head: x = (rmsnorm(x)·w)·silu(z). One warp per (head, token).
// base = token*rs + head*hd; w is shared per head [hd].
extern "C" __global__ void gated_rmsnorm(float* x, const float* w, const float* z,
                                         int hd, float eps, int rs) {
    int head = blockIdx.x, tok = blockIdx.y, lane = threadIdx.x;  // 32 threads
    long base = (long)tok * rs + (long)head * hd;
    float ss = 0.0f;
    for (int i = lane; i < hd; i += 32) { float v = x[base + i]; ss += v * v; }
    ss = warp_all_sum(ss);
    float inv = rsqrtf(ss / (float)hd + eps);
    for (int i = lane; i < hd; i += 32) {
        float zz = z[base + i];
        float sz = zz / (1.0f + expf(-zz));
        x[base + i] = x[base + i] * inv * w[i] * sz;
    }
}

// Fused gated_rmsnorm + q8 quant (output feeds only the ssm_out GEMV). hd=head_v
// (128 = 4 blocks of 32); one warp per (head, token). rs = d_inner.
extern "C" __global__ void gated_rmsnorm_q8(const float* x, const float* w, const float* z,
    signed char* q8, float* d8, float* d8sum, int hd, float eps, int rs) {
    int head = blockIdx.x, tok = blockIdx.y, lane = threadIdx.x;
    long base = (long)tok * rs + (long)head * hd;
    float ss = 0.0f;
    for (int i = lane; i < hd; i += 32) { float v = x[base + i]; ss += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float inv = rsqrtf(ss / (float)hd + eps);
    int nb = hd / 32;
    for (int j = 0; j < nb; j++) {
        int i = 32 * j + lane;
        float zz = z[base + i];
        float sz = zz / (1.0f + expf(-zz));
        float v = x[base + i] * inv * w[i] * sz;
        float a = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
        float dd = a / 127.0f;
        float id = dd > 0.0f ? 1.0f / dd : 0.0f;
        int qi = max(-127, min(127, __float2int_rn(v * id)));
        int blk = head * nb + j;
        q8[blk * 32 + lane] = (signed char)qi;
        float ssum = (float)qi;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ssum += __shfl_xor_sync(0xffffffffu, ssum, o);
        if (lane == 0) { d8[blk] = dd; d8sum[blk] = dd * ssum; }
    }
}

// Fused SwiGLU + q8 quant: act = silu(gate)*up, quantized to q8_1 in one pass.
// Removes the separate silu_mul + quantize kernels and the round-trip of the
// wide activation buffer (ffn=12288). One warp per 32-block.
extern "C" __global__ void silu_mul_q8(const float* gate, const float* up,
    signed char* q8, float* d8, float* d8sum, int total) {
    int blk = blockIdx.x, lane = threadIdx.x;   // 32 threads = one 32-block
    int i = blk * 32 + lane;
    if (i >= total) return;
    float g = gate[i];
    float act = (g / (1.0f + expf(-g))) * up[i];
    float a = fabsf(act);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float d = a / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    int qi = max(-127, min(127, __float2int_rn(act * id)));
    q8[i] = (signed char)qi;
    float s = (float)qi;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) { d8[blk] = d; d8sum[blk] = d * s; }
}

// Embed M tokens: x[m*d + i] = dequant row tokens[m] of token_embd (Q4).
extern "C" __global__ void embed_m_q4(const unsigned char* emb, const int* tokens,
    float* x, const __half* scale, int d, int M) {
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)M * d) return;
    int m = gid / d, col = gid % d;
    int token = tokens[m];
    int nblk = d / 32;
    const unsigned char* row = emb + (long)token * (d / 2);
    int b = col / 32, within = col % 32;
    int byte = within < 16 ? within : within - 16;
    float sc = __half2float(scale[(long)token * nblk + b]);
    unsigned char by = row[b * 16 + byte];
    int nib = within < 16 ? (by & 0xF) : (by >> 4);
    x[gid] = sc * ((float)nib - 8.0f);
}

// Batched RMSNorm → f32 (feeds quantize_q8_pertoken then mma14). One block/token.
extern "C" __global__ void rmsnorm_m(const float* x, const float* w,
    float* out, int d, float eps) {
    int m = blockIdx.x, lid = threadIdx.x, ts = blockDim.x, lane = lid & 31, sg = lid >> 5;
    __shared__ float part[32];
    const float* xr = x + (long)m * d;
    float* outr = out + (long)m * d;
    float s = 0.0f;
    for (int i = lid; i < d; i += ts) { float v = xr[i]; s += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) part[sg] = s;
    __syncthreads();
    int nsg = ts >> 5;
    float tot = 0.0f;
    for (int j = 0; j < nsg; j++) tot += part[j];
    float inv = rsqrtf(tot / (float)d + eps);
    for (int i = lid; i < d; i += ts) outr[i] = xr[i] * inv * w[i];
}

// Partial NeoX RoPE over M tokens; token m uses pos = pos_base + m.
extern "C" __global__ void rope_partial_m(float* v, int hd, int n_rot,
    int pos_base, float base, int n_heads, int rowdim, int M) {
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int rf = n_rot / 2;
    long total = (long)M * n_heads * rf;
    if (gid >= total) return;
    int per = n_heads * rf;
    int m = gid / per, rem = gid % per, head = rem / rf, j = rem % rf;
    int pos = pos_base + m;
    long b = (long)m * rowdim + (long)head * hd;
    float freq = 1.0f / powf(base, 2.0f * (float)j / (float)n_rot);
    float ang = (float)pos * freq, s = sinf(ang), c = cosf(ang);
    float x0 = v[b + j], x1 = v[b + rf + j];
    v[b + j] = x0 * c - x1 * s;
    v[b + rf + j] = x0 * s + x1 * c;
}

// Device-position variants (pos_base from pctl[0]) for graph-captured prefill.
extern "C" __global__ void rope_partial_m_g(float* v, const int* pctl, int hd, int n_rot,
    float base, int n_heads, int rowdim, int M) {
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int rf = n_rot / 2;
    long total = (long)M * n_heads * rf;
    if (gid >= total) return;
    int per = n_heads * rf;
    int m = gid / per, rem = gid % per, head = rem / rf, j = rem % rf;
    int pos = pctl[0] + m;
    long b = (long)m * rowdim + (long)head * hd;
    float freq = 1.0f / powf(base, 2.0f * (float)j / (float)n_rot);
    float ang = (float)pos * freq, s = sinf(ang), c = cosf(ang);
    float x0 = v[b + j], x1 = v[b + rf + j];
    v[b + j] = x0 * c - x1 * s;
    v[b + rf + j] = x0 * s + x1 * c;
}

// Fused gate_mul_sigmoid + per-token q8 quant (attention output → attn_output GEMM).
// attn *= sigmoid(gate from qfull), then per-token quant. One block/token over qdim.
extern "C" __global__ void gate_mul_sigmoid_q8pt(const float* attn, const float* qfull,
    signed char* q8, float* d8, int hd, int qdim) {
    int m = blockIdx.x, lid = threadIdx.x, ts = blockDim.x, lane = lid & 31, sg = lid >> 5;
    __shared__ float part[32];
    __shared__ float bc;
    const float* ar = attn + (long)m * qdim;
    const float* qf = qfull + (long)m * 2 * qdim;
    signed char* qr = q8 + (long)m * qdim;
    float amax = 0.f;
    for (int i = lid; i < qdim; i += ts) {
        int h = i / hd, ii = i % hd;
        float z = qf[h * 2 * hd + hd + ii];
        amax = fmaxf(amax, fabsf(ar[i] * (1.0f / (1.0f + expf(-z)))));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) part[sg] = amax;
    __syncthreads();
    int nsg = ts >> 5;
    float gmax = 0.f;
    for (int j = 0; j < nsg; j++) gmax = fmaxf(gmax, part[j]);
    float dd = gmax / 127.0f;
    if (lid == 0) { d8[m] = dd; bc = dd > 0.f ? 1.0f / dd : 0.f; }
    __syncthreads();
    float id = bc;
    for (int i = lid; i < qdim; i += ts) {
        int h = i / hd, ii = i % hd;
        float z = qf[h * 2 * hd + hd + ii];
        float v = ar[i] * (1.0f / (1.0f + expf(-z)));
        qr[i] = (signed char)max(-127, min(127, __float2int_rn(v * id)));
    }
}

// Fused gated_rmsnorm + per-token q8 quant (SSM out → ssm_out GEMM). One block of
// Hv warps per token: warp = head, rmsnorm over head_v + silu(z) gate → shared,
// then per-token amax over d_inner → quantize. rs = d_inner, hd = head_v.
extern "C" __global__ void gated_rmsnorm_q8pt(const float* x, const float* w, const float* z,
    signed char* q8, float* d8, int hd, float eps, int rs, int Hv) {
    int m = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ float sh[];   // rs floats (normed+gated), + Hv for warp-maxes at [rs..]
    float* wmax = sh + rs;
    long base = (long)m * rs + (long)warp * hd;
    float ss = 0.f;
    for (int i = lane; i < hd; i += 32) { float v = x[base + i]; ss += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float inv = rsqrtf(ss / (float)hd + eps);
    float amax = 0.f;
    for (int i = lane; i < hd; i += 32) {
        float zz = z[base + i]; float sz = zz / (1.0f + expf(-zz));
        float v = x[base + i] * inv * w[i] * sz;
        sh[warp * hd + i] = v; amax = fmaxf(amax, fabsf(v));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) wmax[warp] = amax;
    __syncthreads();
    float gmax = 0.f;
    for (int j = 0; j < Hv; j++) gmax = fmaxf(gmax, wmax[j]);
    float dd = gmax / 127.0f, id = dd > 0.f ? 1.0f / dd : 0.f;
    if (threadIdx.x == 0) d8[m] = dd;
    signed char* qr = q8 + (long)m * rs;
    for (int i = threadIdx.x; i < rs; i += blockDim.x)
        qr[i] = (signed char)max(-127, min(127, __float2int_rn(sh[i] * id)));
}

// Fused rmsnorm + per-token q8 quant → q8[M*d] + d8[M]. Skips the f32 intermediate
// and a separate quant launch. One block/token; two block-reductions (rms, amax).
extern "C" __global__ void rmsnorm_q8pt(const float* x, const float* w,
    signed char* q8, float* d8, int d, float eps) {
    int m = blockIdx.x, lid = threadIdx.x, ts = blockDim.x, lane = lid & 31, sg = lid >> 5;
    __shared__ float part[32];
    __shared__ float bc;   // broadcast 1/scale
    const float* xr = x + (long)m * d;
    signed char* qr = q8 + (long)m * d;
    float s = 0.f;
    for (int i = lid; i < d; i += ts) { float v = xr[i]; s += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) part[sg] = s;
    __syncthreads();
    int nsg = ts >> 5;
    float tot = 0.f;
    for (int j = 0; j < nsg; j++) tot += part[j];
    float inv = rsqrtf(tot / (float)d + eps);
    float amax = 0.f;
    for (int i = lid; i < d; i += ts) amax = fmaxf(amax, fabsf(xr[i] * inv * w[i]));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) part[sg] = amax;
    __syncthreads();
    float gmax = 0.f;
    for (int j = 0; j < nsg; j++) gmax = fmaxf(gmax, part[j]);
    float dd = gmax / 127.0f;
    if (lid == 0) { d8[m] = dd; bc = dd > 0.f ? 1.0f / dd : 0.f; }
    __syncthreads();
    float id = bc;
    for (int i = lid; i < d; i += ts) {
        int qi = max(-127, min(127, __float2int_rn(xr[i] * inv * w[i] * id)));
        qr[i] = (signed char)qi;
    }
}

// Fused silu(gate)*up + per-token q8 quant → q8[M*ffn] + d8[M]. One block/token,
// recompute silu twice (cheap) to avoid a full-row shared buffer.
extern "C" __global__ void silu_mul_q8pt(const float* gate, const float* up,
    signed char* q8, float* d8, int ffn) {
    int m = blockIdx.x, lid = threadIdx.x, ts = blockDim.x, lane = lid & 31, sg = lid >> 5;
    __shared__ float part[32];
    __shared__ float bc;
    const float* gr = gate + (long)m * ffn;
    const float* ur = up + (long)m * ffn;
    signed char* qr = q8 + (long)m * ffn;
    float amax = 0.f;
    for (int i = lid; i < ffn; i += ts) { float g = gr[i]; float a = (g / (1.0f + expf(-g))) * ur[i]; amax = fmaxf(amax, fabsf(a)); }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) part[sg] = amax;
    __syncthreads();
    int nsg = ts >> 5;
    float gmax = 0.f;
    for (int j = 0; j < nsg; j++) gmax = fmaxf(gmax, part[j]);
    float dd = gmax / 127.0f;
    if (lid == 0) { d8[m] = dd; bc = dd > 0.f ? 1.0f / dd : 0.f; }
    __syncthreads();
    float id = bc;
    for (int i = lid; i < ffn; i += ts) {
        float g = gr[i]; float a = (g / (1.0f + expf(-g))) * ur[i];
        qr[i] = (signed char)max(-127, min(127, __float2int_rn(a * id)));
    }
}
"#;

/// Canonical entry names registered from this family.
pub const NAMES: &[&str] = &[
    "rmsnorm",
    "rmsnorm_q8",
    "silu_mul",
    "swiglu",
    "rope",
    "add_inplace",
    "mul_scalar",
    "copy_buf",
    "argmax",
    "argmax_ctl",
    "rope_qk_g",
    "store_kv2_g",
    "embed_q4",
    "rope_partial",
    "embed_q4_g",
    "rope_partial_g",
    "qgate_split",
    "gate_mul_sigmoid",
    "gate_mul_sigmoid_q8",
    "qk_rmsnorm",
    "gated_rmsnorm",
    "gated_rmsnorm_q8",
    "silu_mul_q8",
    "embed_m_q4",
    "rmsnorm_m",
    "rope_partial_m",
    "rope_partial_m_g",
    "gate_mul_sigmoid_q8pt",
    "gated_rmsnorm_q8pt",
    "rmsnorm_q8pt",
    "silu_mul_q8pt",
];
