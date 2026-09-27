//! Multi-token RoPE twins of Metal's `rope_m` and `rope_qk_store_m`
//! (`ojas-metal/src/kernels/ops.rs`), including sectioned M-RoPE (OFF / MROPE / IMROPE /
//! VISION) for surya-2's qwen35 decoder over an image span.
//!
//! Names, argument order and semantics follow Metal, except for the crate rule that every
//! pointer precedes every scalar: Metal's `mpos` (buffer 14, after the scalars) moves to just
//! after the last pointer. A CUDA argument list is positional and complete, so `mpos` cannot
//! be left unbound — a caller that never enables M-RoPE passes any live buffer there; at
//! mode 0 it is never dereferenced.
//!
//! The arithmetic is Metal's line for line (unsigned indices, `1/pow(base, 2j/rd)`, angle =
//! float(pos)·freq, f16 round-to-nearest KV store), so the plain path evaluates exactly the
//! `rope_partial_m` expression.
//!
//! Launch: 1-D, `grid = ceil(total / block)`, any block.
//!   rope_m:          total = M * R/2
//!   rope_qk_store_m: total = M * (Aq + Ak + kvdim)
pub const BODY: &str = r#"
// RoPE (NeoX pairing, full head, exponent over hd) for M rows. R = elements per row
// (n_head*hd for q, n_kv*hd for k); token m uses pos = base_pos + m.
extern "C" __global__ void rope_m(float* v, unsigned hd, unsigned base_pos, float base,
                                  unsigned R, unsigned M) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned hf = hd / 2u; unsigned halfPerRow = R / 2u; unsigned total = M * halfPerRow;
    if (gid >= total) return;
    unsigned m = gid / halfPerRow; unsigned rem = gid % halfPerRow;
    unsigned head = rem / hf; unsigned i = rem % hf;
    unsigned long long b = (unsigned long long)m * R + head * hd; unsigned pos = base_pos + m;
    float freq = 1.0f / powf(base, 2.0f * (float)i / (float)hd);
    float ang = (float)pos * freq, s = sinf(ang), c = cosf(ang);
    float x0 = v[b + i], x1 = v[b + hf + i];
    v[b + i] = x0 * c - x1 * s; v[b + hf + i] = x0 * s + x1 * c;
}

#define MROPE_SHIFT 8u

// Sectioned M-RoPE stream selection — Metal `mrope_sel` verbatim.
// mpos = [s0 s1 s2 s3 | per token m: t h w e]; sections are counted in pairs.
// mode 1 MROPE contiguous, 2 IMROPE interleaved t h w (qwen35), 3 VISION contiguous with
// theta restarting per section. Returns (position for the angle, exponent index).
__device__ __forceinline__ uint2 mrope_sel(const unsigned* mpos, unsigned mode,
                                           unsigned m, unsigned j) {
    unsigned s0 = mpos[0], s1 = mpos[1], s2 = mpos[2], s3 = mpos[3];
    unsigned sect = s0 + s1 + s2 + s3;
    if (sect == 0u) return make_uint2(mpos[4u + 4u * m], j);
    unsigned sector = j % sect;
    unsigned sel = 0u, start = 0u;
    if (mode == 2u) {
        unsigned r = sector % 3u;
        if      (r == 1u && sector < 3u * s1) { sel = 1u; }
        else if (r == 2u && sector < 3u * s2) { sel = 2u; }
        else if (r == 0u && sector < 3u * s0) { sel = 0u; }
        else                                  { sel = 3u; }
    } else {
        if      (sector < s0)           { sel = 0u; start = 0u; }
        else if (sector < s0 + s1)      { sel = 1u; start = s0; }
        else if (sector < s0 + s1 + s2) { sel = 2u; start = s0 + s1; }
        else                            { sel = 3u; start = s0 + s1 + s2; }
    }
    return make_uint2(mpos[4u + 4u * m + sel], (mode == 3u) ? (sector - start) : j);
}

// Fused RoPE(q,k) + f16 store(k,v) for M tokens (Metal `rope_qk_store_m`). Per token:
// q pairs (Aq = n_head*hd/2) | k pairs (Ak, rotated straight into kc) | v copies (kvdim).
// `neox` = (mode << MROPE_SHIFT) | pairing; rd = rotated dims (partial rope).
// The angle takes the section's position stream; the cache row is always base_pos + m.
extern "C" __global__ void rope_qk_store_m(float* vq, float* vk, const float* vv,
    __half* kc, __half* vc, const unsigned* mpos,
    unsigned hd, unsigned base_pos, float base, unsigned Aq, unsigned Ak, unsigned kvdim,
    unsigned M, unsigned neox, unsigned rd) {
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned perTok = Aq + Ak + kvdim;
    if (gid >= (unsigned long long)M * perTok) return;
    unsigned m = (unsigned)(gid / perTok); unsigned w = (unsigned)(gid % perTok);
    unsigned hf = hd / 2u; unsigned rf = rd / 2u; unsigned pos = base_pos + m;
    unsigned nx = neox & 1u; unsigned mrope = neox >> MROPE_SHIFT;
    if (w < Aq) {
        unsigned head = w / hf; unsigned j = w % hf;
        unsigned long long b = (unsigned long long)m * (2u * Aq) + head * hd;
        if (j >= rf) return;
        unsigned long long a0 = nx ? b + j : b + 2u * j;
        unsigned long long a1 = nx ? b + rf + j : b + 2u * j + 1u;
        uint2 pj = make_uint2(pos, j);
        if (mrope != 0u) pj = mrope_sel(mpos, mrope, m, j);
        float freq = 1.0f / powf(base, 2.0f * (float)pj.y / (float)rd);
        float ang = (float)pj.x * freq, s = sinf(ang), c = cosf(ang);
        float x0 = vq[a0], x1 = vq[a1];
        vq[a0] = x0 * c - x1 * s; vq[a1] = x0 * s + x1 * c;
    } else if (w < Aq + Ak) {
        unsigned wk = w - Aq; unsigned head = wk / hf; unsigned j = wk % hf;
        unsigned long long b = (unsigned long long)m * kvdim + head * hd;
        unsigned long long row = (unsigned long long)(base_pos + m) * kvdim;
        if (j >= rf) {
            unsigned e0 = head * hd + rd + 2u * (j - rf);
            unsigned long long src = (unsigned long long)m * kvdim + e0;
            kc[row + e0] = __float2half(vk[src]); kc[row + e0 + 1u] = __float2half(vk[src + 1u]);
            return;
        }
        unsigned long long a0 = nx ? b + j : b + 2u * j;
        unsigned long long a1 = nx ? b + rf + j : b + 2u * j + 1u;
        unsigned o0 = nx ? j : 2u * j; unsigned o1 = nx ? rf + j : 2u * j + 1u;
        uint2 pj = make_uint2(pos, j);
        if (mrope != 0u) pj = mrope_sel(mpos, mrope, m, j);
        float freq = 1.0f / powf(base, 2.0f * (float)pj.y / (float)rd);
        float ang = (float)pj.x * freq, s = sinf(ang), c = cosf(ang);
        float x0 = vk[a0], x1 = vk[a1];
        float n0 = x0 * c - x1 * s, n1 = x0 * s + x1 * c;
        unsigned long long cb = row + head * hd;
        kc[cb + o0] = __float2half(n0); kc[cb + o1] = __float2half(n1);
    } else {
        unsigned e = w - Aq - Ak;
        vc[(unsigned long long)(base_pos + m) * kvdim + e] =
            __float2half(vv[(unsigned long long)m * kvdim + e]);
    }
}
"#;

pub const NAMES: &[&str] = &["rope_m", "rope_qk_store_m"];
