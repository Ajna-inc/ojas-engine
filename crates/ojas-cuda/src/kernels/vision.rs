//! ViT tower primitives: the CUDA twins of `ojas-metal/src/kernels/vision.rs`, plus the glue
//! the surya-2 tower (`ojas-models/src/decoder/vision.rs::encode_vit`) dispatches that CUDA
//! lacked — `add_rowbias_m` and `copy_f32_half`.
//!
//! Entries keep their Metal names and argument semantics; only the order differs. CUDA
//! `KernelRuntime::dispatch` appends all buffers then all u32 constants, so every pointer
//! precedes every scalar (`tests/conformance.rs::pointers_precede_scalars_in_every_signature`);
//! within each group Metal's slot order is kept, and float scalars (`eps`, `base`) travel as
//! `f32::to_bits`.
//!
//! | entry              | CUDA buffers (in order)   | CUDA scalars (in order) | Metal slots            |
//! |--------------------|---------------------------|-------------------------|------------------------|
//! | vit_layernorm_m    | x, w, out, b              | d, eps                  | 0 x 1 w 2 out 5 b; 3 d 4 eps |
//! | vit_gelu           | x, out                    | n                       | 0 x 1 out; 2 n         |
//! | vit_patchify       | img, out                  | W, H, C, P, total       | 0,1; 2..6              |
//! | vit_rope           | v, mpos                   | hd, base, R, M          | 0 v 5 mpos; 1 hd 2 base 3 R 4 M |
//! | vit_qkv_split      | qkv, q, k, v              | d, total                | 0..3; 4, 5             |
//! | vit_merge_permute  | src, dst                  | d, pw, total            | 0, 1; 2..4             |
//! | add_rowbias_m      | x, b                      | N, total                | 0, 1; 2, 3             |
//! | copy_f32_half      | src, dst(half)            | n                       | 0, 1; 2                |
//!
//! Grids are the Metal ones (threadgroups -> blocks): `vit_layernorm_m` is one block per row
//! with a power-of-two block <= 256; everything else is one thread per element (or per rotated
//! pair for `vit_rope`), any block width.
//!
//! Activations stay f32 end to end: surya's ViT residual stream reaches 1500-3000, inside
//! f16's per-element range but not once squared or summed, and the Metal tower keeps them f32
//! for the same reason. The only f16 here is `copy_f32_half`, which feeds K/V to the
//! bidirectional attention kernels (they read `const __half*`).
//!
//! The numerical contracts are transcribed from the Metal file statement for statement.
//! Oracle: `ojas_cpu::cpu_math::{layernorm, gelu}` and
//! `ojas_cpu::cpu_vit::{patchify, merge_permutation, mrope_positions, vision_rope}`, gated by
//! `tests/surya_vision_kernels.rs`.

pub const NAMES: &[&str] = &[
    "vit_layernorm_m",
    "vit_gelu",
    "vit_patchify",
    "vit_rope",
    "vit_qkv_split",
    "vit_merge_permute",
    "add_rowbias_m",
    "copy_f32_half",
];

// Compiles against kernels::PRELUDE (cuda_fp16 + warp reduction helpers).
pub const BODY: &str = r#"
// Metal's `ffn_act(g, 1u)`: GELU, tanh approximation, inner term clamped to +-30 (fast-math
// tanh(inf) is NaN; both backends clamp so they evaluate the same expression). Local name so
// it cannot collide with a future PRELUDE `ffn_act`.
__device__ __forceinline__ float vit_gelu_tanh(float g) {
    float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
    inner = fminf(fmaxf(inner, -30.0f), 30.0f);
    return 0.5f * g * (1.0f + tanhf(inner));
}

// LayerNorm with bias over M rows of x[M,d]. One block per row, blockDim a power of two <= 256.
//   y[i] = (x[i] - mean) * rsqrt(var + eps) * w[i] + b[i]
//   mean = sum(x)/d,  var = sum((x-mean)^2)/d     <- population variance, eps inside, two passes
// `out` may alias `x`: both reductions complete before the store loop, and a thread rewrites
// only the elements it read itself.
extern "C" __global__ void vit_layernorm_m(const float* x, const float* w, float* out,
    const float* b, unsigned int d, float eps) {
    __shared__ float part[256];
    __shared__ float part2[256];
    unsigned int m = blockIdx.x, lid = threadIdx.x, ts = blockDim.x;
    const float* xm = x + (unsigned long long)m * d;
    float* om = out + (unsigned long long)m * d;
    float s = 0.0f;
    for (unsigned int i = lid; i < d; i += ts) s += xm[i];
    part[lid] = s; __syncthreads();
    for (unsigned int off = ts / 2u; off > 0u; off >>= 1u) {
        if (lid < off) part[lid] += part[lid + off];
        __syncthreads();
    }
    float mean = part[0] / (float)d;
    float s2 = 0.0f;
    for (unsigned int i = lid; i < d; i += ts) { float v = xm[i] - mean; s2 += v * v; }
    part2[lid] = s2; __syncthreads();
    for (unsigned int off = ts / 2u; off > 0u; off >>= 1u) {
        if (lid < off) part2[lid] += part2[lid + off];
        __syncthreads();
    }
    float inv = rsqrtf(part2[0] / (float)d + eps);
    for (unsigned int i = lid; i < d; i += ts) om[i] = (xm[i] - mean) * inv * w[i] + b[i];
}

// Elementwise GELU (fc1 -> GELU -> fc2, no gate/up split). `out` may alias `x`.
extern "C" __global__ void vit_gelu(const float* x, float* out, unsigned int n) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = vit_gelu_tanh(x[gid]);
}

// im2col for non-overlapping patches, one thread per output element.
// in : planar CHW f32 img[c*H*W + y*W + x]
// out: [T, C*P*P], T = (W/P)*(H/P) row-major patches, each row (ic, ky, kx) with kx fastest.
// A partial trailing patch is dropped (pw = W/P). total = (W/P)*(H/P)*C*P*P.
extern "C" __global__ void vit_patchify(const float* img, float* out,
    unsigned int W, unsigned int H, unsigned int C, unsigned int P, unsigned int total) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    unsigned int pw = W / P, pp = P * P, row = C * pp;
    unsigned int t = gid / row, e = gid % row;
    unsigned int py = t / pw, px = t % pw;
    unsigned int ic = e / pp, r = e % pp;
    unsigned int ky = r / P, kx = r % P;
    unsigned int y = py * P + ky, xx = px * P + kx;
    out[gid] = img[(unsigned long long)ic * H * W + (unsigned long long)y * W + xx];
}

// Sectioned VISION M-RoPE, non-storing, in place on M rows of v[M, R] (R = n_head*hd).
//   pair j in [0, hd/2) rotates (v[j], v[j + hd/2])            <- NEOX, whole head
//   sector = j % (s0+s1+s2+s3); stream = section of `sector` -> mpos[4 + 4*m + stream]
//   theta  = pos * theta_scale^(sector - section_start)       <- indep_sects, repeated multiply
// mpos: [s0,s1,s2,s3] then (t,h,w,e) per token — `ojas_metal::kernels::ops::mrope_desc`.
// sect == 0 degenerates to plain NEOX driven by stream 0. One thread per rotated pair:
// total = M*(R/hd)*(hd/2).
extern "C" __global__ void vit_rope(float* v, const unsigned int* mpos,
    unsigned int hd, float base, unsigned int R, unsigned int M) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nd = hd / 2u;
    unsigned int nh = R / hd;
    unsigned int perRow = nh * nd;
    if (gid >= M * perRow) return;
    unsigned int m = gid / perRow, rem = gid % perRow;
    unsigned int head = rem / nd, j = rem % nd;
    unsigned int s0 = mpos[0], s1 = mpos[1], s2 = mpos[2], s3 = mpos[3];
    unsigned int sect = s0 + s1 + s2 + s3;
    unsigned int sel = 0u, start = 0u, sector = j;
    if (sect != 0u) {
        sector = j % sect;
        if      (sector < s0)           { sel = 0u; start = 0u; }
        else if (sector < s0 + s1)      { sel = 1u; start = s0; }
        else if (sector < s0 + s1 + s2) { sel = 2u; start = s0 + s1; }
        else                            { sel = 3u; start = s0 + s1 + s2; }
    }
    float tsc = powf(base, -2.0f / (float)nd);
    float th = (float)mpos[4u + 4u * m + sel];
    for (unsigned int e = start; e < sector; e++) th *= tsc;
    float s = sinf(th), c = cosf(th);
    unsigned long long bb = (unsigned long long)m * R + (unsigned long long)(head * hd);
    float x0 = v[bb + j], x1 = v[bb + nd + j];
    v[bb + j] = x0 * c - x1 * s;
    v[bb + nd + j] = x0 * s + x1 * c;
}

// Split a fused attn_qkv row [Q|K|V] (row stride 3*d) into three contiguous [M,d] streams.
// total = M*d.
extern "C" __global__ void vit_qkv_split(const float* qkv, float* q, float* k, float* v,
    unsigned int d, unsigned int total) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    unsigned int m = gid / d, i = gid % d;
    unsigned long long r = (unsigned long long)m * (3ull * d);
    q[gid] = qkv[r + i];
    k[gid] = qkv[r + d + i];
    v[gid] = qkv[r + 2ull * d + i];
}

// 2x2 spatial merge reorder (before block 0; token count unchanged). dst slot t holds source
// patch   b = t/4, r = t%4, hb = pw/2, y = 2*(b/hb) + r/2, x = 2*(b%hb) + r%2
// i.e. `ojas_cpu::cpu_vit::merge_permutation(pw, ph)` in closed form. dst must not alias src.
// total = T*d, pw even.
extern "C" __global__ void vit_merge_permute(const float* src, float* dst,
    unsigned int d, unsigned int pw, unsigned int total) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    unsigned int t = gid / d, i = gid % d;
    unsigned int b = t >> 2, r = t & 3u, hb = pw >> 1;
    unsigned int y = 2u * (b / hb) + (r >> 1), xx = 2u * (b % hb) + (r & 1u);
    dst[gid] = src[(unsigned long long)(y * pw + xx) * d + i];
}

// Broadcast a length-N bias across all rows of x[M,N], in place (total = M*N threads).
// With N == total it is a plain elementwise add, which is how encode_vit adds the position
// embedding. Metal: gemv.rs `add_rowbias_m`.
extern "C" __global__ void add_rowbias_m(float* x, const float* b, unsigned int N,
    unsigned int total) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total) x[gid] += b[gid % N];
}

// f32 -> f16 copy (round to nearest even), n elements. Metal: ops.rs `copy_f32_half`.
extern "C" __global__ void copy_f32_half(const float* src, __half* dst, unsigned int n) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) dst[gid] = __float2half_rn(src[gid]);
}
"#;
