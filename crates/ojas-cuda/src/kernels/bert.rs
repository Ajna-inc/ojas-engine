//! The ModernBERT text encoder's kernels (`crate::bert`): what the vision family does not
//! already provide for a packed batch of independent sequences. Metal-canonical names where
//! Metal has the op (`ops.rs` / `attn.rs` there); the `bert_` entries are CUDA's own.
//!
//! | entry                     | args (pointers, then u32 / f32)                                   | grid                      | block |
//! |---------------------------|-------------------------------------------------------------------|---------------------------|-------|
//! | `bert_embed_f16`          | emb(f16 [vocab, d]), tokens(u32 [M]), x(f32 [M, d]), d, M         | `[ceil(M*d/256), 1, 1]`   | 256   |
//! | `bert_layernorm_m`        | x, w, out, b, d, eps(f32), has_bias                               | `[M, 1, 1]`               | 256   |
//! | `bert_rope_m`             | v(f32 [M, R]), pos(u32 [M]), hd, base(f32), R, M                  | `[ceil(M*R/2/256), 1, 1]` | 256   |
//! | `attention_m_bidir_span`  | q, kc(f16), vc(f16), out, span(u32 [2M]), hd, kvdim, total, group, scale(f32), n_head | `[M*n_head, 1, 1]` | 256 |
//! | `ffn_gu_rows`             | x([M, 2f]), out([M, f]), f, n(=M*f), act                           | `[ceil(n/256), 1, 1]`     | 256   |
//! | `act_m`                   | x, out, n, act                                                    | `[ceil(n/256), 1, 1]`     | 256   |
//! | `ple_gather`              | table([*, hd]), rows(i32 [n]), emb([n, hd]), hd, n                | `[ceil(n*hd/256), 1, 1]`  | 256   |
//!
//! `act`: 1 tanh-GELU, 3 erf-GELU (ModernBERT, PyTorch `nn.GELU()`), 4 ReLU, else SiLU — the
//! codes of Metal's `ffn_act`.
//!
//! `attention_m_bidir_span` is `attention_m_bidir` (`attn_bidir.rs`) with a per-row key
//! range: query row `m` attends to keys `[span[2m], span[2m+1])` instead of `[0, total)`.
//! One range per row expresses both things a packed text-encoder batch needs: sequence
//! boundaries and a symmetric local window (ModernBERT's sliding layers, `|i - j| <= w`),
//! intersected on the host. The online softmax, the warp split and the store are the
//! bidirectional kernel's; only the loop bounds move. Every span must be non-empty.
//!
//! Compiles against kernels::PRELUDE.
pub const BODY: &str = r#"
__device__ __forceinline__ float bert_act(float g, unsigned act) {
    if (act == 1u) {
        float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
        inner = fminf(fmaxf(inner, -30.0f), 30.0f);
        return 0.5f * g * (1.0f + tanhf(inner));
    }
    if (act == 3u) return 0.5f * g * (1.0f + erff(g * 0.70710678118654752f));
    if (act == 4u) return fmaxf(g, 0.0f);
    return g / (1.0f + expf(-g));
}

// x[m] = float(emb[tokens[m]]), M rows of d.
extern "C" __global__ void bert_embed_f16(const __half* emb, const unsigned* tokens, float* x,
    unsigned d, unsigned M) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (unsigned long long)M * d) return;
    unsigned m = (unsigned)(g / d), i = (unsigned)(g % d);
    x[g] = __half2float(emb[(unsigned long long)tokens[m] * d + i]);
}

// LayerNorm over M rows of x[M, d], bias optional (has_bias == 0 reads no b). One block per
// row; `out` may alias `x`. The same two-pass population variance as vit_layernorm_m.
extern "C" __global__ void bert_layernorm_m(const float* x, const float* w, float* out,
    const float* b, unsigned d, float eps, unsigned has_bias) {
    __shared__ float part[256];
    __shared__ float part2[256];
    unsigned m = blockIdx.x, lid = threadIdx.x, ts = blockDim.x;
    const float* xm = x + (unsigned long long)m * d;
    float* om = out + (unsigned long long)m * d;
    float s = 0.0f;
    for (unsigned i = lid; i < d; i += ts) s += xm[i];
    part[lid] = s; __syncthreads();
    for (unsigned off = ts / 2u; off > 0u; off >>= 1u) {
        if (lid < off) part[lid] += part[lid + off];
        __syncthreads();
    }
    float mean = part[0] / (float)d;
    float s2 = 0.0f;
    for (unsigned i = lid; i < d; i += ts) { float v = xm[i] - mean; s2 += v * v; }
    part2[lid] = s2; __syncthreads();
    for (unsigned off = ts / 2u; off > 0u; off >>= 1u) {
        if (lid < off) part2[lid] += part2[lid + off];
        __syncthreads();
    }
    float inv = rsqrtf(part2[0] / (float)d + eps);
    for (unsigned i = lid; i < d; i += ts) {
        float y = (xm[i] - mean) * inv * w[i];
        om[i] = has_bias ? y + b[i] : y;
    }
}

// NEOX rotary embedding in place on M rows of v[M, R] (R = n_head * hd), one position per
// row: pair (j, j + hd/2) of every head rotates by theta_j = pos[m] * base^(-2j/hd). One
// thread per rotated pair: total = M * (R/hd) * (hd/2).
extern "C" __global__ void bert_rope_m(float* v, const unsigned* pos, unsigned hd, float base,
    unsigned R, unsigned M) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned nd = hd / 2u, nh = R / hd, per_row = nh * nd;
    if (g >= (unsigned long long)M * per_row) return;
    unsigned m = (unsigned)(g / per_row), rem = (unsigned)(g % per_row);
    unsigned head = rem / nd, j = rem % nd;
    float th = (float)pos[m] * powf(base, -2.0f * (float)j / (float)hd);
    float s, c;
    sincosf(th, &s, &c);
    unsigned long long bb = (unsigned long long)m * R + (unsigned long long)head * hd;
    float x0 = v[bb + j], x1 = v[bb + nd + j];
    v[bb + j] = x0 * c - x1 * s;
    v[bb + nd + j] = x0 * s + x1 * c;
}

// ---- attention_m_bidir with per-row key spans (see the module docs) ----
extern "C" __global__ void attention_m_bidir_span(const float* q, const __half* kc,
    const __half* vc, float* out, const unsigned* span, unsigned hd, unsigned kvdim,
    unsigned total, unsigned group, float scale, unsigned n_head) {
    __shared__ float qsh[512];
    __shared__ float tacc[8 * 512];
    __shared__ float tm[8];
    __shared__ float tl[8];
    unsigned tg = blockIdx.x, lid = threadIdx.x, ts = blockDim.x;
    unsigned m = tg / n_head, head = tg % n_head, kvh = head / group;
    unsigned lo = span[2u * m], seq = span[2u * m + 1u];   // keys [lo, seq) only
    (void)total;
    unsigned R = n_head * hd;
    unsigned sgid = lid / 32u, lane = lid % 32u, nsg = ts / 32u;
    const float* qh = q + (unsigned long long)m * R + head * hd;
    for (unsigned i = lid; i < hd; i += ts) qsh[i] = qh[i];
    __syncthreads();
    float mi = -1e30f, li = 0.0f;
    float acc[16];
    #pragma unroll
    for (int c = 0; c < 16; c++) acc[c] = 0.0f;
    unsigned nch = (hd + 31u) / 32u;
    for (unsigned t = lo + sgid; t < seq; t += nsg) {
        const __half* kt = kc + (unsigned long long)t * kvdim + kvh * hd;
        float sv = 0.0f;
        for (unsigned i = lane; i < hd; i += 32u) sv += qsh[i] * __half2float(kt[i]);
        sv = warp_all_sum(sv) * scale;
        float mn = fmaxf(mi, sv); float corr = expf(mi - mn); float pw = expf(sv - mn);
        li = li * corr + pw;
        const __half* vt = vc + (unsigned long long)t * kvdim + kvh * hd;
        #pragma unroll
        for (unsigned c = 0u; c < 16u; c++) {
            unsigned i = lane + 32u * c;
            if (c < nch && i < hd) acc[c] = acc[c] * corr + pw * __half2float(vt[i]);
        }
        mi = mn;
    }
    if (lane == 0u) { tm[sgid] = mi; tl[sgid] = li; }
    #pragma unroll
    for (unsigned c = 0u; c < 16u; c++) {
        unsigned i = lane + 32u * c;
        if (c < nch && i < hd) tacc[sgid * hd + i] = acc[c];
    }
    __syncthreads();
    float gm = -1e30f;
    for (unsigned j = 0u; j < nsg; j++) gm = fmaxf(gm, tm[j]);
    float gl = 0.0f;
    for (unsigned j = 0u; j < nsg; j++) gl += tl[j] * expf(tm[j] - gm);
    for (unsigned i = lid; i < hd; i += ts) {
        float o = 0.0f;
        for (unsigned j = 0u; j < nsg; j++) o += tacc[j * hd + i] * expf(tm[j] - gm);
        out[(unsigned long long)m * R + head * hd + i] = o / gl;
    }
}

// Gated MLP activation over fused [g | u] rows of width 2f: out[m, i] = act(g) * u.
extern "C" __global__ void ffn_gu_rows(const float* x, float* out, unsigned f, unsigned n,
    unsigned act) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    unsigned long long r = (g / f) * (unsigned long long)(2u * f) + (g % f);
    out[g] = bert_act(x[r], act) * x[r + f];
}

// Elementwise activation; `out` may alias `x`.
extern "C" __global__ void act_m(const float* x, float* out, unsigned n, unsigned act) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) out[g] = bert_act(x[g], act);
}

// Row gather: emb[h, :] = table[rows[h], :] for h < n, rows of width hd.
extern "C" __global__ void ple_gather(const float* table, const int* rows, float* emb,
    unsigned hd, unsigned n) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (unsigned long long)hd * n) return;
    unsigned h = (unsigned)(g / hd), i = (unsigned)(g % hd);
    emb[g] = table[(unsigned long long)rows[h] * hd + i];
}
"#;

pub const NAMES: &[&str] = &[
    "bert_embed_f16", "bert_layernorm_m", "bert_rope_m", "attention_m_bidir_span",
    "ffn_gu_rows", "act_m", "ple_gather",
];
