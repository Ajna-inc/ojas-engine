//! Encoder tower primitives: the kernels a bidirectional encoder needs that the
//! decoder families do not already provide. The qwen3vl ViT tower and the
//! ModernBERT text encoder both run on them, through the shared block loop in
//! `ojas-models/src/decoder/encoder.rs`. The `vit_` prefix is historical; the CUDA
//! twins in `ojas-cuda/src/kernels/vision.rs` carry the same names.
//!
//! Everything else an encoder runs is an existing kernel and must stay that way:
//! activations are `act_m` / `ffn_gu_rows` over the shared `ffn_act` (`ops.rs`,
//! `prelude.rs`), per-projection bias is `add_rowbias_m` (`gemv.rs`), every linear goes
//! through `projm` (`gemm_mm_f16_fat` at f16, `gemm_fat.rs`), and non-causal attention is
//! `attention_m_bidir` (`attn_core.rs`), its per-row key-span variants
//! (`attn::attn_bidir_span_src`, `attn::attn_mma_span_src`) or
//! `attention_m_mma_bidir_<hd>` (`attn.rs`). The family is registered in `SPLIT`
//! (`mod.rs`), so the PRELUDE and `ffn_act` come for free.
//!
//! 1. `vit_layernorm_m` — mean-subtracting LayerNorm, bias optional. Every other norm
//!    in the tree is RMS (`rmsnorm_m` in `ops.rs`, and its twins in `gemv.rs`,
//!    `qwen4exp.rs`, `mla.rs` and `train.rs`) and LN cannot be a flag on them: RMS carries one accumulator
//!    (Σx²) and no bias, LN needs Σx too. It is a structural copy of `rmsnorm_m` —
//!    same threadgroup-per-row grid, `part[256]` tree reduction and `(3,d)`/`(4,eps)`
//!    constant slots — plus a second accumulator, a bias buffer at slot 5 and its
//!    presence flag at slot 6.
//! 2. `vit_patchify` — the patch embedding's im2col, not a convolution: 16×16
//!    stride-16 patches do not overlap, so it is a pure permutation with no
//!    duplication, and its `[T, C*P*P]` result feeds the patch-embed GEMM directly.
//! 3. `vit_rope` — sectioned M-RoPE driven by a per-row position table, writing only
//!    the rotated vector. With every section size zero and `freq_dims = hd` it is a
//!    plain NEOX rope on stream 0, which is how the text encoder restarts positions
//!    per packed sequence. `rope_qk_store_m` (`ops.rs`) has the section logic but also
//!    writes a KV cache the tower lacks (its K and V are per-layer temporaries);
//!    `rope_m` (`ops.rs`) does not store but takes a scalar `base_pos`, which cannot
//!    express a patch's (y, x). Sharing with `ops.rs` would mean moving text into the
//!    PRELUDE, as `ops` and `vision` are separate SPLIT units.
//! 4. `vit_qkv_prep` — the attention inputs in one pass over the fused `attn_qkv`
//!    output: split `[Q|K|V]`, rotate Q and K with `vit_rope`'s arithmetic, store Q as
//!    f32 and K, V as the half the attention kernels read. The encoder block runs this;
//!    `vit_qkv_split`, `vit_rope` and `copy_f32_half` in sequence are its oracle, and
//!    `tests/vision_kernels.rs` holds the two bit-identical. `vit_qkv_split` and
//!    `vit_rope` also keep the names the CUDA twins carry.
//! 5. `vit_merge_permute` — the 2×2 spatial reorder that runs before block 0.
//! 6. `vit_gelu` — `act_m` with act 1, kept only for the CUDA twin's name (see the
//!    kernel).
//!
//! # Constraints these kernels encode
//!
//! LayerNorm uses the population variance (divide by n) with `eps` inside the sqrt,
//! `y = (x - mean)/sqrt(var + eps)`, in two passes — both as in ggml's `ggml_norm`
//! (`ggml-cpu/ops.cpp:3694`) and the oracle `ojas_cpu::cpu_math::layernorm`
//! (`cpu_math.rs:468`), and both pinned by `tests/vision_kernels.rs`: `eps` outside the
//! sqrt is invisible on ordinary rows and wrong by 3.3× on a near-constant one, which
//! is what a ViT's `post_ln` sees, and the one-pass Σx²-mean² identity cancels on those
//! same rows and can go negative.
//!
//! The ViT's GELU is the tanh approximation (`act_m` with act 1), as in the
//! reference; ModernBERT's is the exact erf form (act 3). See `ffn_act` in the
//! prelude.
//!
//! Patchify row order is load-bearing: any other nesting compiles, runs, and gives a
//! subtly wrong tower. See `vit_patchify` below; oracle `ojas_cpu::cpu_vit::patchify`.
//!
//! Vision M-RoPE has three silent traps, each with a test against
//! `ojas_cpu::cpu_vit::vision_rope`. `indep_sects` is on, so the frequency ramp
//! restarts at every section boundary and the exponent is `sector - section_start`,
//! not `sector`. The NEOX pairing spans the whole head — pair `j` is `(j, j + hd/2)`
//! — and every channel is rotated; vision is the one rope mode with no pass-through
//! tail. And the sections do not collapse into a plain 64-wide rope: with
//! `n_dims = hd/2 = 32` rotated pairs against `sect_dims = 4*(hd/4) = 64` only
//! sectors 0..31 are reached, so sections 2 and 3 (the `w`/`e` position streams) are
//! unreachable and the result is two independent 32-wide NEOX ropes — dims
//! {0..16}∪{32..48} driven by `pos[0]`, dims {16..32}∪{48..64} by `pos[1]`.
//! Simplifying it to one rope passes a t == h test and fails on every real image.
//!
//! `vit_merge_permute`'s index arithmetic must be the same loop nest as the M-RoPE
//! position table (reference `clip.cpp:4780`, oracle
//! `ojas_cpu::cpu_vit::merge_permutation`). Out of step, the encoder still runs and
//! the page transcribes as almost the right text.

// Bodies compile against kernels::PRELUDE (shared defines + helpers).
pub const BODY: &str = r#"
// LayerNorm with bias over M rows of x[M,d]: one threadgroup per row, `ts` a power
// of two and <= 256 (the `part` arrays). Structural twin of `rmsnorm_m`
// (ops.rs:357) — same grid, tree reduction and constant slots — so a call site
// swapping RMS for LN binds one extra buffer and changes nothing else.
//
//   y[i] = (x[i] - mean) * rsqrt(var + eps) * w[i] + b[i]
//   mean = Σx/d,  var = Σ(x-mean)²/d       <- population variance, eps inside
//
// Two passes, as in ggml's `ggml_norm` (ggml-cpu/ops.cpp:3694) and
// ojas_cpu::cpu_math::layernorm:468. The Σx²-mean² identity is not used: it cancels
// when a row is near-constant (|x| large, spread small) and can land on a negative
// variance, the regime a ViT's post_ln lives in.
//
// buffers: 0 x[M,d] f32  1 w[d] f32  2 out[M,d] f32  5 b[d] f32
// consts : 3 d (uint)    4 eps (float)   6 has_bias (uint)
// With has_bias == 0 the bias is not read (ModernBERT's norms carry none); bind any
// buffer at slot 5.
// grid   : M threadgroups x ts threads. `out` may alias `x`: both reductions finish
//          before the store loop, and a thread only rewrites the elements it reads
//          itself (tests/vision_kernels.rs asserts it).
kernel void vit_layernorm_m(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& d [[buffer(3)]], constant float& eps [[buffer(4)]],
    device const float* b [[buffer(5)]], constant uint& has_bias [[buffer(6)]],
    uint m [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    device const float* xm = x + (ulong)m*(ulong)d; device float* om = out + (ulong)m*(ulong)d;
    threadgroup float part[256]; threadgroup float part2[256];
    float s=0.0;
    for (uint i=lid;i<d;i+=ts) s+=xm[i];
    part[lid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off=ts/2u;off>0u;off>>=1u){ if(lid<off) part[lid]+=part[lid+off]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mean=part[0]/float(d);
    // Second accumulator needs its own array: reusing `part` would race the
    // broadcast read of part[0] above against thread 0's write below.
    float s2=0.0;
    for (uint i=lid;i<d;i+=ts){ float v=xm[i]-mean; s2+=v*v; }
    part2[lid]=s2; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off=ts/2u;off>0u;off>>=1u){ if(lid<off) part2[lid]+=part2[lid+off]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv=rsqrt(part2[0]/float(d)+eps);
    if (has_bias != 0u) { for (uint i=lid;i<d;i+=ts) om[i]=(xm[i]-mean)*inv*w[i]+b[i]; }
    else                { for (uint i=lid;i<d;i+=ts) om[i]=(xm[i]-mean)*inv*w[i]; }
}

// Elementwise tanh GELU: `act_m` with act 1, kept as its own entry because the CUDA
// twin (`ojas-cuda/src/kernels/vision.rs`) is named `vit_gelu` and the cross-backend
// parity ratchet counts shared names. Retire both together once the CUDA encoder
// path adopts `act_m`. Encoders on Metal dispatch `act_m`.
// buffers: 0 x f32  1 out f32 (may alias x)   consts: 2 n (uint)
kernel void vit_gelu(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) { out[gid] = ffn_act(x[gid], 1u); }
}

// im2col for NON-OVERLAPPING patches: a gather, one thread per output element.
//
// in : planar CHW f32, img[c*H*W + y*W + x]      (clip.cpp:4485 `inp_raw`)
// out: [T, C*P*P] f32, T = (W/P)*(H/P) patches in row-major (py*pw + px) order,
//      each row ordered (ic, ky, kx) with kx fastest — ggml im2col row order
//      `dst_data[iic*(KH*KW) + ikh*KW + ikw]`, i.e. the layout of one output
//      channel of v.patch_embd.weight [kw, kh, ic, oc] (ggml-cpu/ops.cpp:6417).
//      Feeds gemm_mm_f16 as its f32 activation with K = C*P*P.
// A partial trailing patch is dropped (pw = W/P), matching ojas_cpu::cpu_vit.
//
// buffers: 0 img f32  1 out f32
// consts : 2 W  3 H  4 C  5 P  6 total = (W/P)*(H/P)*C*P*P
kernel void vit_patchify(device const float* img [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& W [[buffer(2)]], constant uint& H [[buffer(3)]],
    constant uint& C [[buffer(4)]], constant uint& P [[buffer(5)]],
    constant uint& total [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= total) { return; }
    uint pw = W/P; uint pp = P*P; uint row = C*pp;
    uint t = gid/row, e = gid%row;             // patch index, element within patch
    uint py = t/pw, px = t%pw;                 // patch grid position (row-major)
    uint ic = e/pp, r = e%pp;                  // channel-major within the row
    uint ky = r/P, kx = r%P;                   // kx fastest
    uint y = py*P + ky, xx = px*P + kx;
    out[gid] = img[(ulong)ic*(ulong)H*(ulong)W + (ulong)y*(ulong)W + (ulong)xx];
}

// Sectioned vision M-RoPE, non-storing, in place on M rows of v[M, R].
//
// Transcribed from `ggml_mrope_cache_init` + `rotate_pairs` (ggml-cpu/ops.cpp) for
// `mode == GGML_ROPE_TYPE_VISION`, which llama.cpp reaches with `n_dims = hd/2`,
// `sections = {hd/4} x 4`, `freq_base 10000`, `freq_scale 1`, no YaRN and no
// freq_factors (`qwen3vl.cpp:104`). Oracle: `ojas_cpu::cpu_vit::vision_rope`.
//
//   pair j in [0, hd/2) rotates (v[j], v[j + hd/2])      <- NEOX, whole head
//   sector      = j % (s0+s1+s2+s3)
//   stream      = the section `sector` falls in -> mpos[4 + 4*m + stream]
//   exponent    = sector - section_start                 <- indep_sects: restart
//   theta       = pos * theta_scale^exponent,  theta_scale = base^(-2/n_dims)
//
// `theta` is built by repeated multiplication rather than `pow(theta_scale, e)`:
// that is what ggml does, and bit-parity with a reference dump requires it. The
// trip count is `sector - start` < hd/4 (16 here), so the loop is cheap.
//
// `mpos` has the same layout `rope_qk_store_m` takes at buffer(14) and that
// `kernels::ops::mrope_desc` builds — `[s0,s1,s2,s3]` then `(t,h,w,e)` per token —
// so one host-side descriptor serves both the decoder's rope and the tower's.
// Sections are counted in cos/sin pairs. `sect == 0` (no sections declared)
// degenerates to a plain NEOX rope driven by stream 0, which makes the buffer safe
// to reuse for a non-sectioned caller.
//
// `freq_dims` is ggml's `n_dims`, the width the frequency ramp is spread over:
// theta_scale = base^(-2/freq_dims). Vision rope passes hd/2 (ggml's vision mode
// rotates the whole head but ramps over half of it); a plain NEOX text rope passes
// hd, giving the usual theta_j = pos * base^(-2j/hd).
//
// buffers: 0 v[M,R] f32 (in place)   5 mpos u32
// consts : 1 hd   3 R = n_head*hd   4 M   6 freq_dims   |   float: 2 base
// grid   : ceil(M*(R/hd)*(hd/2) / 64) x 64
// Rotation angle of pair `j` in row `m`: the section lookup and the indep_sects ramp
// described above. Shared by `vit_rope` and `vit_qkv_prep` so both evaluate the
// same expression.
inline float vit_rope_angle(device const uint* mpos, uint m, uint j, float base, uint freq_dims) {
    uint s0=mpos[0], s1=mpos[1], s2=mpos[2], s3=mpos[3];
    uint sect = s0+s1+s2+s3;
    uint sel = 0u, start = 0u, sector = j;
    if (sect != 0u) {
        sector = j % sect;
        if      (sector < s0)       { sel = 0u; start = 0u; }
        else if (sector < s0+s1)    { sel = 1u; start = s0; }
        else if (sector < s0+s1+s2) { sel = 2u; start = s0+s1; }
        else                        { sel = 3u; start = s0+s1+s2; }
    }
    float ts = pow(base, -2.0/float(freq_dims));
    float th = float(mpos[4u + 4u*m + sel]);
    for (uint e = start; e < sector; e++) { th *= ts; }   // indep_sects ramp
    return th;
}

kernel void vit_rope(device float* v [[buffer(0)]], constant uint& hd [[buffer(1)]],
    constant float& base [[buffer(2)]], constant uint& R [[buffer(3)]],
    constant uint& M [[buffer(4)]], device const uint* mpos [[buffer(5)]],
    constant uint& freq_dims [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    uint nd = hd/2u;                            // rotated pairs per head
    uint nh = R/hd;                             // heads per row
    uint perRow = nh*nd;
    if (gid >= M*perRow) { return; }
    uint m = gid/perRow, rem = gid%perRow;
    uint head = rem/nd, j = rem%nd;
    float th = vit_rope_angle(mpos, m, j, base, freq_dims);
    float s = sin(th), c = cos(th);
    ulong b = (ulong)m*(ulong)R + (ulong)(head*hd);
    float x0 = v[b+j], x1 = v[b+nd+j];
    v[b+j] = x0*c - x1*s; v[b+nd+j] = x0*s + x1*c;
}

// The encoder's attention inputs in one pass over the fused QKV projection: split
// the [Q|K|V] row, rotate Q and K by `vit_rope`'s arithmetic, store Q as f32 and K,
// V as the half the attention kernels read. Replaces vit_qkv_split, two vit_rope
// and two copy_f32_half dispatches, and their round trips through memory; the
// results are those five kernels' bit for bit.
//
// buffers: 0 qkv[M,3d] f32  1 q[M,d] f32  2 kh[M,d] half  3 vh[M,d] half  5 mpos
// consts : 4 d  6 hd  7 M  8 freq_dims (0: no rotation)  |  float: 9 base
// grid   : one thread per (row, head, rotated pair), M*d/2 threads
kernel void vit_qkv_prep(device const float* qkv [[buffer(0)]], device float* q [[buffer(1)]],
    device half* kh [[buffer(2)]], device half* vh [[buffer(3)]],
    constant uint& d [[buffer(4)]], device const uint* mpos [[buffer(5)]],
    constant uint& hd [[buffer(6)]], constant uint& M [[buffer(7)]],
    constant uint& freq_dims [[buffer(8)]], constant float& base [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    uint nd = hd/2u;
    uint perRow = d/2u;
    if (gid >= M*perRow) { return; }
    uint m = gid/perRow, rem = gid%perRow;
    uint head = rem/nd, j = rem%nd;
    ulong r = (ulong)m*(ulong)(3u*d) + (ulong)(head*hd);
    ulong o = (ulong)m*(ulong)d + (ulong)(head*hd);
    float q0 = qkv[r+j],       q1 = qkv[r+nd+j];
    float k0 = qkv[r+d+j],     k1 = qkv[r+d+nd+j];
    if (freq_dims != 0u) {
        float th = vit_rope_angle(mpos, m, j, base, freq_dims);
        float s = sin(th), c = cos(th);
        float a = q0, b = q1;  q0 = a*c - b*s;  q1 = a*s + b*c;
        a = k0; b = k1;        k0 = a*c - b*s;  k1 = a*s + b*c;
    }
    q[o+j] = q0; q[o+nd+j] = q1;
    kh[o+j] = half(k0); kh[o+nd+j] = half(k1);
    vh[o+j] = half(qkv[r+2u*d+j]); vh[o+nd+j] = half(qkv[r+2u*d+nd+j]);
}

// Split a fused attn_qkv row into three contiguous d-wide streams.
//
// `attn_qkv.weight` is one [d, 3d] tensor, so `projm` lands [Q|K|V] interleaved at
// a 3*d row stride (reference `qwen3vl.cpp:84-97` takes three strided views of it).
// Every consumer downstream wants contiguous rows: `vit_rope` strides by
// `n_head*hd`, and `attention_m_bidir` / `attention_m_mma_bidir_*` stride `kc`/`vc`
// by `kvdim`. Copying once here is cheaper than teaching four kernels a second
// stride.
//
// The encoder block now runs `vit_qkv_prep`, which does this split, the rotation and
// the half conversion in one pass; this kernel is its test oracle and keeps the name
// of its CUDA twin.
//
// buffers: 0 qkv[M,3d] f32  1 q[M,d]  2 k[M,d]  3 v[M,d]
// consts : 4 d   5 total = M*d        grid: ceil(total/64) x 64
kernel void vit_qkv_split(device const float* qkv [[buffer(0)]], device float* q [[buffer(1)]],
    device float* k [[buffer(2)]], device float* v [[buffer(3)]],
    constant uint& d [[buffer(4)]], constant uint& total [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) { return; }
    uint m = gid/d, i = gid%d;
    ulong r = (ulong)m*(ulong)(3u*d);
    q[gid] = qkv[r + i];
    k[gid] = qkv[r + (ulong)d + i];
    v[gid] = qkv[r + (ulong)(2u*d) + i];
}

// 2x2 spatial merge: the reorder that runs before block 0.
//
// The reference expresses it as permute/reshape/permute (`qwen3vl.cpp:18-31`) and
// lands back on [n_embd, n_patches_x*n_patches_y], the same token count. Its only
// job is to make each group of four consecutive tokens a 2x2 spatial block, so that
// the projector's `reshape(n_embd*4, n_pos/4)` picks up neighbours for free.
//
// Unrolled, the destination order visits (y_block, x_block, dy, dx) with dx
// fastest, i.e. dst slot `d` holds source patch
//
//     b = d/4, r = d%4, hb = pw/2
//     y = 2*(b/hb) + r/2,  x = 2*(b%hb) + r%2,  src = y*pw + x
//
// which is `ojas_cpu::cpu_vit::merge_permutation(pw, ph)` in closed form — no index
// buffer, and the same arithmetic the M-RoPE position table must use.
//
// buffers: 0 src[T,d] f32  1 dst[T,d] f32 (must not alias src)
// consts : 2 d   3 pw (patches per row, even)   4 total = T*d
kernel void vit_merge_permute(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
    constant uint& d [[buffer(2)]], constant uint& pw [[buffer(3)]],
    constant uint& total [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= total) { return; }
    uint t = gid/d, i = gid%d;
    uint b = t >> 2, r = t & 3u, hb = pw >> 1;
    uint y = 2u*(b/hb) + (r >> 1), x = 2u*(b%hb) + (r & 1u);
    dst[gid] = src[(ulong)(y*pw + x)*(ulong)d + i];
}
"#;
