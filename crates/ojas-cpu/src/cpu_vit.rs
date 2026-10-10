//! Pure-CPU qwen3vl-style vision encoder (surya-2 `mmproj`) — the numerical
//! oracle the Metal ViT is validated against. Correctness over speed; nothing
//! here is tuned for throughput.
//!
//! Shape: 12 pre-norm blocks over the full patch count (16 384 for a 4 096-token
//! page), LayerNorm-with-bias (not RMSNorm), biases on every projection, fused
//! QKV, vision M-RoPE on Q and K in every block, bidirectional attention, GELU
//! MLP, then `v.post_ln` and a two-layer `qwen3vl_merger` projector.
//!
//! Five details are transcribed from the reference rather than derived from the
//! config, and each has a test:
//!
//! 1. The patch embed is a gather, not a convolution: 16×16 stride-16 patches
//!    are non-overlapping, so im2col is a permutation feeding a plain matmul
//!    ([`patchify`]).
//! 2. The two patch-embed convs fold. `temporal_patch_size = 2` makes the
//!    reference run `conv(w0,x) + conv(w1,x)` over the same image for a still
//!    (`clip_graph_qwen2vl::build_inp_with_temporal_merge`, qwen2vl.cpp:3), so
//!    `w0+w1` at load is one conv ([`fold_patch_weights`]).
//! 3. The 2×2 spatial merge runs before block 0 and is a reorder, not a
//!    reduction: token count is unchanged, and it exists only so the final
//!    `reshape(n_embd*4, n_pos/4)` picks up spatial neighbours
//!    ([`merge_permutation`]). Merging at the end instead is a common bug.
//! 4. The learned 48×48 position embedding is bilinearly resized with
//!    ALIGN_CORNERS to the patch grid, and early-returns unmodified at an
//!    exactly 48×48 grid ([`resize_position_embeddings`]).
//! 5. Vision M-RoPE (`GGML_ROPE_TYPE_VISION`): four sections of `d_head/4`, base
//!    10 000, NEOX pairing `(j, j+d_head/2)`, each section restarting the
//!    frequency ramp (`indep_sects`) ([`vision_rope`]). Its position order must
//!    match the spatial permute exactly ([`mrope_positions`]); out of step it
//!    produces text that is only nearly right.
//!
//! Preprocessing is a separate stage: [`CpuVit::forward`] takes an
//! already-normalized planar-CHW f32 buffer plus `(width, height)`.
//!
//! `OJAS_VIT_F32=1` keeps every matmul weight in f32 instead of the GGUF's f16
//! (the whisper loader has the same switch).

use crate::cpu_math::{gelu, layernorm, matmul as mm, W};
use anyhow::{bail, Context, Result};
use half::f16;
use half::slice::HalfFloatSliceExt;
use ojas_formats::gguf::Gguf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `GGML_ROPE_TYPE_VISION` freq base for the qwen3vl tower (clip.cpp `build()`).
const ROPE_BASE: f32 = 10000.0;

// ---------------------------------------------------------------------------
// pure index / numeric helpers — each independently testable
// ---------------------------------------------------------------------------

/// Fold the two patch-embed convolutions into one.
///
/// For a still image the reference feeds the same pixels to `patch_embeddings_0`
/// and `patch_embeddings_1` and adds the results (qwen2vl.cpp:3, the `n_batch ==
/// 1` arm), so `conv(w0,x) + conv(w1,x) == conv(w0+w1, x)` and the second conv
/// is redundant. The fold is done in f32 regardless of the storage tier: `w0`
/// and `w1` are f16 in the GGUF and their sum is not, in general, an f16.
pub fn fold_patch_weights(w0: &[f32], w1: &[f32]) -> Result<Vec<f32>> {
    if w0.len() != w1.len() {
        bail!("patch_embd fold: {} vs {} elements", w0.len(), w1.len());
    }
    Ok(w0.iter().zip(w1).map(|(a, b)| a + b).collect())
}

/// im2col for non-overlapping patches: a gather, no duplication.
///
/// `img` is planar CHW (`img[c*h*w + y*w + x]`) — the layout clip.cpp builds for
/// `inp_raw` (clip.cpp:4553, "the channel dim is unrolled"). Output is one row
/// per patch, `channels * patch * patch` wide, ordered `(ic, ky, kx)` with `kx`
/// fastest: ggml's im2col row order
/// (`dst_data[iic*(KH*KW) + ikh*KW + ikw]`, ggml-cpu/ops.cpp), which is also the
/// layout of one output channel of `v.patch_embd.weight` `[kw, kh, ic, oc]`.
/// Patch (token) order is row-major `y*pw + x`.
pub fn patchify(img: &[f32], width: usize, height: usize, channels: usize, patch: usize) -> Vec<f32> {
    let (pw, ph) = (width / patch, height / patch);
    let row = channels * patch * patch;
    let mut out = vec![0f32; pw * ph * row];
    for py in 0..ph {
        for px in 0..pw {
            let dst = (py * pw + px) * row;
            for ic in 0..channels {
                for ky in 0..patch {
                    let src = ic * height * width + (py * patch + ky) * width + px * patch;
                    let d = dst + ic * patch * patch + ky * patch;
                    out[d..d + patch].copy_from_slice(&img[src..src + patch]);
                }
            }
        }
    }
    out
}

/// Destination-slot → source-patch index for the 2×2 spatial merge that runs
/// before block 0.
///
/// The reference expresses it as permute/reshape/permute
/// (qwen3vl.cpp, "// spatial merge"); unrolled, the contiguous layout it lands
/// on visits `(y_block, x_block, dy, dx)` with `dx` fastest:
///
/// ```text
/// for y in (0..ph).step_by(2) { for x in (0..pw).step_by(2) {
///   for dy in 0..2 { for dx in 0..2 { emit (y+dy)*pw + (x+dx) } } } }
/// ```
///
/// Token count is unchanged. The same permutation is applied to the resized
/// position embedding.
pub fn merge_permutation(pw: usize, ph: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(pw * ph);
    for y in (0..ph).step_by(2) {
        for x in (0..pw).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    perm.push((y + dy) * pw + (x + dx));
                }
            }
        }
    }
    perm
}

/// The four M-RoPE position channels per token, in permuted token order.
///
/// Transcribed verbatim from clip.cpp:4787 (`PROJECTOR_TYPE_QWEN3VL` arm). The
/// loop nest is the same one as in [`merge_permutation`]; if the two disagree
/// the encoder still runs and the output is subtly wrong, so they are tested
/// against each other.
///
/// Channels 2 and 3 duplicate 0 and 1 in the reference and are never read at
/// `d_head = 64`: with `n_dims = d_head/2 = 32` there are only 32 rotated pairs
/// while `sect_dims = 4 * (d_head/4) = 64`, so sectors 32..63 (which select
/// `theta_w` / `theta_e`) are unreachable. Kept so the transcription stays
/// literal.
pub fn mrope_positions(pw: usize, ph: usize) -> Vec<[i32; 4]> {
    let mut pos = Vec::with_capacity(pw * ph);
    for y in (0..ph).step_by(2) {
        for x in (0..pw).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    let (t, h) = ((y + dy) as i32, (x + dx) as i32);
                    pos.push([t, h, t, h]);
                }
            }
        }
    }
    pos
}

/// Bilinear resize of the learned position embedding, ALIGN_CORNERS.
///
/// `src` is `[side*side][n_embd]` row-major in grid index `g = y*side + x`
/// (`v.position_embd.weight` is `[n_embd, side*side]` in GGUF `ne` order).
/// Returns `[height*width][n_embd]` in the same convention.
///
/// The early return is part of the contract:
/// `clip_graph::resize_position_embeddings` (clip.cpp:312) returns the tensor
/// unmodified when the grid is already `side × side`, so a 48×48 page must be
/// bit-identical to the raw weight, not merely close as a resize with sf == 1
/// would be.
///
/// The interpolation is `ggml_compute_forward_upscale_f32`'s
/// `GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS` branch: pixel
/// offset 0, `sf = (dst-1)/(src-1)` (falling back to `dst/src` when either side
/// is 1), corner indices clamped before the fractional part is taken.
pub fn resize_position_embeddings(
    src: &[f32], n_embd: usize, side: usize, width: usize, height: usize,
) -> Vec<f32> {
    if width == side && height == side {
        return src.to_vec(); // early return — bit-identical, per clip.cpp:322
    }
    let sf0 = if width > 1 && side > 1 { (width - 1) as f32 / (side - 1) as f32 } else { width as f32 / side as f32 };
    let sf1 = if height > 1 && side > 1 { (height - 1) as f32 / (side - 1) as f32 } else { height as f32 / side as f32 };
    let mut out = vec![0f32; width * height * n_embd];
    for iy in 0..height {
        let y = iy as f32 / sf1;
        let y0f = y.floor();
        let y0 = (y0f as i64).clamp(0, side as i64 - 1) as usize;
        let y1 = (y0f as i64 + 1).clamp(0, side as i64 - 1) as usize;
        let dy = (y - y0 as f32).clamp(0.0, 1.0);
        for ix in 0..width {
            let x = ix as f32 / sf0;
            let x0f = x.floor();
            let x0 = (x0f as i64).clamp(0, side as i64 - 1) as usize;
            let x1 = (x0f as i64 + 1).clamp(0, side as i64 - 1) as usize;
            let dx = (x - x0 as f32).clamp(0.0, 1.0);
            let (ra, rb) = (y0 * side + x0, y0 * side + x1);
            let (rc, rd) = (y1 * side + x0, y1 * side + x1);
            let dst = (iy * width + ix) * n_embd;
            for c in 0..n_embd {
                let a = src[ra * n_embd + c];
                let b = src[rb * n_embd + c];
                let cc = src[rc * n_embd + c];
                let d = src[rd * n_embd + c];
                out[dst + c] = a * (1.0 - dx) * (1.0 - dy)
                    + b * dx * (1.0 - dy)
                    + cc * (1.0 - dx) * dy
                    + d * dx * dy;
            }
        }
    }
    out
}

/// Vision M-RoPE, in place on one head vector of `head_dim` values.
///
/// Transcribed from `ggml_mrope_cache_init` + `rotate_pairs` (ggml-cpu/ops.cpp)
/// for `mode == GGML_ROPE_TYPE_VISION`, which llama.cpp reaches with
/// `n_dims = head_dim/2`, `sections = {head_dim/4} × 4`, `freq_scale = 1`,
/// `ext_factor = 0`, `attn_factor = 1` — no YaRN, no freq_factors.
///
/// Three properties a from-scratch implementation tends to get wrong:
///   * NEOX pairing spans the whole head: pair `j` is `(j, j + head_dim/2)` for
///     `j` in `0..head_dim/2`, and every channel is rotated (vision is the one
///     rope mode with no pass-through tail).
///   * `indep_sects` is on: each section restarts the frequency ramp, so pair
///     `j` uses `theta_scale^(j - section_start)`, not `theta_scale^j`.
///   * `theta_scale = base^(-2/n_dims)` with `n_dims = head_dim/2`, and the
///     angle is built by repeated multiplication, not `powf` — the GPU kernel
///     must do the same for bit-parity with ggml.
pub fn vision_rope(v: &mut [f32], head_dim: usize, pos: [i32; 4], base: f32) {
    let ne0 = head_dim;
    let n_dims = head_dim / 2;
    let sections = [head_dim / 4, head_dim / 4, head_dim / 4, head_dim / 4];
    let theta_scale = base.powf(-2.0 / n_dims as f32);

    let sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    let sec_w = sections[1] + sections[0];
    let sec_e = sections[2] + sec_w;

    let (bt, bh, bw, be) = (pos[0] as f32, pos[1] as f32, pos[2] as f32, pos[3] as f32);
    let (mut theta_t, mut theta_h, mut theta_w, mut theta_e) = (bt, bh, bw, be);

    let mut cache = vec![0f32; ne0];
    let mut i0 = 0;
    while i0 < ne0 {
        let sector = (i0 / 2) % sect_dims;
        // indep_sects: reset the ramp at each section boundary
        if sector == 0 {
            theta_t = bt;
        } else if sector == sections[0] {
            theta_h = bh;
        } else if sector == sec_w {
            theta_w = bw;
        } else if sector == sec_e {
            theta_e = be;
        }
        let theta = if sector >= sections[0] && sector < sec_w {
            theta_h
        } else if sector >= sec_w && sector < sec_w + sections[2] {
            theta_w
        } else if sector >= sec_w + sections[2] {
            theta_e
        } else {
            theta_t
        };
        let (s, c) = theta.sin_cos();
        cache[i0] = c;
        cache[i0 + 1] = s;
        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
        i0 += 2;
    }

    // rotate_pairs(ne0, n_dims, cache, src, dst) — scale 2, so ic = i0/2
    let mut i0 = 0;
    while i0 < ne0 {
        let ic = i0 / 2;
        let (c, s) = (cache[i0], cache[i0 + 1]);
        let x0 = v[ic];
        let x1 = v[ic + n_dims];
        v[ic] = x0 * c - x1 * s;
        v[ic + n_dims] = x0 * s + x1 * c;
        i0 += 2;
    }
}

/// Softmax over `s`, returning the denominator and leaving the unnormalized
/// `exp(s - max)` weights in place.
///
/// Normalization convention: accumulate-then-divide. The caller accumulates
/// `Σ e_u · v_u` with the unnormalized weights and divides the accumulator by
/// `den` once, at the end; it does not scale the weights first. The two differ
/// in the last bits (`n_pos` roundings per output element vs one), so the Metal
/// side must make the same choice, and does: `attention_m_bidir`
/// (`kernels/attn_core.rs:262`) and `attention_m_mma` are online-softmax kernels
/// that finish with `o/gl`. `cpu_whisper.rs`'s encoder attention matches too.
///
/// llama.cpp is not internally consistent here. `ggml_flash_attn_ext` — what
/// clip.cpp actually runs, since `build_attn` enables it whenever the backend
/// supports it — is accumulate-then-divide and agrees with this file.
/// `ggml_compute_forward_soft_max_f32` (the CPU fallback) scales the probability
/// vector by `1/sum` before the KQV matmul, so a `-ngl 0` reference dump differs
/// from this oracle in the last bits of every attention output.
fn softmax_in_place(s: &mut [f32]) -> f32 {
    let mx = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut den = 0.0f32;
    for v in s.iter_mut() {
        *v = (*v - mx).exp();
        den += *v;
    }
    den
}

/// Parallel map over token rows writing disjoint `row`-sized spans of `out`,
/// on the shared `cpu_math` worker pool (`std::thread::scope` per call costs
/// ~0.5 ms, and a 12-layer ViT issues 12 of these).
fn par_rows(n: usize, row: usize, threads: usize, out: &mut [f32], f: impl Fn(usize, &mut [f32]) + Sync) {
    debug_assert_eq!(out.len(), n * row);
    let base = out.as_mut_ptr() as usize;
    let next = AtomicUsize::new(0);
    let body = |_id: usize, _nt: usize| loop {
        let t = next.fetch_add(1, Ordering::Relaxed);
        if t >= n {
            break;
        }
        // SAFETY: `next` hands each token index to exactly one worker, and
        // token t owns the disjoint span [t*row, (t+1)*row) of `out`, which
        // outlives the call because `parallel` joins before returning.
        let dst = unsafe { std::slice::from_raw_parts_mut((base as *mut f32).add(t * row), row) };
        f(t, dst);
    };
    crate::cpu_math::parallel(threads, &body);
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

struct Block {
    ln1: (Vec<f32>, Vec<f32>),
    ln2: (Vec<f32>, Vec<f32>),
    qkv: W,
    qkv_b: Vec<f32>,
    o: W,
    o_b: Vec<f32>,
    up: W,
    up_b: Vec<f32>,
    down: W,
    down_b: Vec<f32>,
}

/// Intermediate activations, for stage-by-stage parity against
/// `reference_probe` / `mtmd_get_output_embd`, so a projector, permute-order or
/// M-RoPE bug localizes instead of surfacing as subtly wrong text.
pub struct VitTrace {
    /// Patch grid actually used, `(patches_x, patches_y)`.
    pub grid: (usize, usize),
    /// `v.post_ln` output, `[n_pos][n_embd]`, in permuted token order.
    pub post_ln: Vec<f32>,
    /// `mm.0` output before the GELU, `[n_pos/4][n_embd*4]`.
    pub mm0: Vec<f32>,
    /// `mm.2` output — what the decoder consumes, `[n_pos/4][projection_dim]`.
    pub out: Vec<f32>,
}

pub struct CpuVit {
    pub n_embd: usize,
    pub n_layers: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub patch: usize,
    pub channels: usize,
    /// Side of the learned position-embedding grid (`image_size / patch_size`).
    pub pos_side: usize,
    /// Spatial merge factor (2 → 2×2 blocks).
    pub merge: usize,
    pub proj_dim: usize,
    pub eps: f32,
    /// `w0 + w1`, `[n_embd][channels*patch*patch]`.
    patch_w: W,
    patch_b: Vec<f32>,
    /// `[pos_side*pos_side][n_embd]`.
    pos_embd: Vec<f32>,
    blocks: Vec<Block>,
    post_ln: (Vec<f32>, Vec<f32>),
    mm0: (W, Vec<f32>),
    mm2: (W, Vec<f32>),
    threads: usize,
    dotprod: bool,
}

fn read_f32(g: &mut Gguf, name: &str) -> Result<Vec<f32>> {
    let (_dims, ty, b) = g.read_tensor(name).with_context(|| format!("reading {name}"))?;
    match ty {
        0 => Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        1 => {
            let v: Vec<f16> = b.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
            let mut f = vec![0f32; v.len()];
            v.convert_to_f32_slice(&mut f);
            Ok(f)
        }
        t => bail!("{name}: unsupported GGUF type {t}"),
    }
}

/// Matmul weights keep the GGUF's own precision (f16 here) unless
/// `OJAS_VIT_F32` is set, which separates "the GPU kernel is wrong" from "f16
/// rounding differs" — the same switch as `OJAS_WHISPER_F32`. `cpu_whisper`'s
/// `readf`/`readw` bail on anything but f32 and so are unusable for an f16
/// mmproj; this uses the `cpu_qwen.rs` ty-1/ty-0 pattern instead.
fn read_w(g: &mut Gguf, name: &str, f32_mode: bool) -> Result<W> {
    let (_dims, ty, b) = g.read_tensor(name).with_context(|| format!("reading {name}"))?;
    match ty {
        0 => Ok(W::F32(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())),
        1 => {
            let v: Vec<f16> = b.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
            if f32_mode {
                let mut f = vec![0f32; v.len()];
                v.convert_to_f32_slice(&mut f);
                return Ok(W::F32(f));
            }
            Ok(W::F16(v))
        }
        t => bail!("{name}: unsupported GGUF type {t}"),
    }
}

/// `y[t] = W·x[t] + bias` over a flat `[T*k]` activation buffer, returning a
/// flat `[T*n]` one. Thin shim over `cpu_math::matmul` — the contract there is
/// `y[t][n] = Σ_k x[t][k]·w[n,k]` with row-major `[n][k]` weights, which is
/// exactly GGUF `ne = [k, n]`.
fn linear(w: &W, n: usize, k: usize, x: &[f32], bias: Option<&[f32]>, threads: usize, dotprod: bool) -> Vec<f32> {
    let t = x.len() / k;
    let xs: Vec<&[f32]> = (0..t).map(|i| &x[i * k..(i + 1) * k]).collect();
    let mut outs = vec![vec![0f32; n]; t];
    mm(w, n, k, &xs, bias, &mut outs, threads, dotprod);
    outs.concat()
}

impl CpuVit {
    pub fn load(g: &mut Gguf) -> Result<CpuVit> {
        if g.arch() != "clip" {
            bail!("CpuVit needs general.architecture=clip (got {})", g.arch());
        }
        let mu = |g: &Gguf, k: &str| g.meta_u32(&format!("clip.vision.{k}")).unwrap_or(0) as usize;
        let n_embd = mu(g, "embedding_length");
        let n_layers = mu(g, "block_count");
        let n_head = mu(g, "attention.head_count");
        let ffn = mu(g, "feed_forward_length");
        let patch = mu(g, "patch_size");
        let image_size = mu(g, "image_size");
        let proj_dim = mu(g, "projection_dim");
        let merge = mu(g, "spatial_merge_size").max(1);
        let eps = g.meta_f32("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6);
        if n_embd == 0 || n_layers == 0 || n_head == 0 || patch == 0 || image_size == 0 {
            bail!("mmproj is missing clip.vision.* metadata (embd={n_embd} blocks={n_layers} heads={n_head} patch={patch} image={image_size})");
        }
        if n_embd % n_head != 0 {
            bail!("n_embd {n_embd} not divisible by n_head {n_head}");
        }
        let head_dim = n_embd / n_head;
        if head_dim % 4 != 0 {
            bail!("vision M-RoPE needs head_dim {head_dim} divisible by 4 (4 sections of head_dim/4)");
        }
        if merge != 2 {
            bail!("only spatial_merge_size=2 is implemented (got {merge})");
        }
        // Everything downstream assumes no deepstack: surya-2 ships twelve
        // `false`s here and zero `v.deepstack.*` tensors. Qwen3.5-VL proper does
        // use it, so fail loudly rather than silently dropping features.
        if g.int_arr("clip.vision.is_deepstack_layers").is_some_and(|v| v.iter().any(|&b| b != 0)) {
            bail!("this mmproj declares deepstack layers; CpuVit does not implement deepstack");
        }
        let pos_side = image_size / patch;

        let f32_mode = ojas_core::config::flag("OJAS_VIT_F32");
        let dotprod = crate::cpu_math::fast_i8();
        let threads = ojas_core::config::var("OJAS_CPU_THREADS").ok().and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .or_else(crate::cpu_math::perf_cores)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));

        // patch embed: fold the two convs (see `fold_patch_weights`)
        let (pdims, _, _) = g.read_tensor("v.patch_embd.weight").context("reading v.patch_embd.weight")?;
        if pdims.len() != 4 {
            bail!("v.patch_embd.weight: expected 4-D [kw,kh,ic,oc], got {pdims:?}");
        }
        let (kw, kh, channels, oc) = (pdims[0] as usize, pdims[1] as usize, pdims[2] as usize, pdims[3] as usize);
        if kw != patch || kh != patch || oc != n_embd {
            bail!("v.patch_embd.weight {pdims:?} disagrees with patch_size {patch} / n_embd {n_embd}");
        }
        let w0 = read_f32(g, "v.patch_embd.weight")?;
        let w1 = read_f32(g, "v.patch_embd.weight.1")?;
        let folded = fold_patch_weights(&w0, &w1)?;
        // The fold must happen in f32; the storage tier then matches every other
        // weight so the oracle and the GPU round in the same place.
        let patch_w = if f32_mode {
            W::F32(folded)
        } else {
            W::F16(folded.iter().map(|&v| f16::from_f32(v)).collect())
        };

        let ln = |g: &mut Gguf, n: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((read_f32(g, &format!("{n}.weight"))?, read_f32(g, &format!("{n}.bias"))?))
        };

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("v.blk.{i}.{s}");
            blocks.push(Block {
                ln1: ln(g, &p("ln1"))?,
                ln2: ln(g, &p("ln2"))?,
                qkv: read_w(g, &p("attn_qkv.weight"), f32_mode)?,
                qkv_b: read_f32(g, &p("attn_qkv.bias"))?,
                o: read_w(g, &p("attn_out.weight"), f32_mode)?,
                o_b: read_f32(g, &p("attn_out.bias"))?,
                up: read_w(g, &p("ffn_up.weight"), f32_mode)?,
                up_b: read_f32(g, &p("ffn_up.bias"))?,
                down: read_w(g, &p("ffn_down.weight"), f32_mode)?,
                down_b: read_f32(g, &p("ffn_down.bias"))?,
            });
        }

        let pos_embd = read_f32(g, "v.position_embd.weight")?;
        if pos_embd.len() != pos_side * pos_side * n_embd {
            bail!("v.position_embd.weight has {} elements, expected {}x{}x{}",
                  pos_embd.len(), pos_side, pos_side, n_embd);
        }

        let m = CpuVit {
            n_embd, n_layers, n_head, head_dim, ffn, patch, channels, pos_side, merge, proj_dim, eps,
            patch_w,
            patch_b: read_f32(g, "v.patch_embd.bias")?,
            pos_embd,
            blocks,
            post_ln: ln(g, "v.post_ln")?,
            mm0: (read_w(g, "mm.0.weight", f32_mode)?, read_f32(g, "mm.0.bias")?),
            mm2: (read_w(g, "mm.2.weight", f32_mode)?, read_f32(g, "mm.2.bias")?),
            threads, dotprod,
        };
        // The projector's input width is the 2x2 merge, so the first layer must
        // take n_embd * merge^2. A failure here means the merge is applied in
        // the wrong place — at the end instead of before block 0.
        if m.mm0.1.len() != n_embd * merge * merge {
            bail!("mm.0.bias is {} wide, expected n_embd*merge^2 = {}", m.mm0.1.len(), n_embd * merge * merge);
        }
        if m.mm2.1.len() != proj_dim {
            bail!("mm.2.bias is {} wide, expected clip.vision.projection_dim = {proj_dim}", m.mm2.1.len());
        }
        tracing::info!(target: "cpu:vit",
            "vit d={n_embd} blocks={n_layers} heads={n_head} hd={head_dim} ffn={ffn} patch={patch} \
             pos_grid={pos_side}x{pos_side} merge={merge} proj={proj_dim} eps={eps:e} | f32={f32_mode} threads={threads}");
        Ok(m)
    }

    /// Merged token count for an image of this size — what the decoder will see.
    pub fn n_merged_tokens(&self, width: usize, height: usize) -> usize {
        (width / self.patch) * (height / self.patch) / (self.merge * self.merge)
    }

    /// `img` is an already-normalized planar-CHW f32 buffer
    /// (`img[c*h*w + y*w + x]`, `channels*height*width` long). Returns the
    /// projector output, `[n_merged_tokens * projection_dim]` row-major.
    ///
    /// Preprocessing (bicubic resize to the pixel budget, mean/std normalize) is
    /// a separate, separately testable stage and is not done here.
    pub fn forward(&self, img: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        Ok(self.forward_trace(img, width, height)?.out)
    }

    /// [`CpuVit::forward`] plus the three coarse checkpoints. See [`VitTrace`].
    pub fn forward_trace(&self, img: &[f32], width: usize, height: usize) -> Result<VitTrace> {
        self.forward_tapped(img, width, height, &mut |_, _, _| {})
    }

    /// [`CpuVit::forward_trace`] with a tap fired at every point the reference
    /// `clip_graph::cb()` fires, so a per-layer dump pairs with a
    /// `reference_probe` vision trace file-for-file.
    ///
    /// `tap(name, layer, values)` uses llama.cpp's own tensor names
    /// (`models/qwen3vl.cpp`, `clip.cpp build_norm/build_attn/build_ffn`):
    /// `patch_bias`, `inp_pos_emb`, then per layer `ln1`, `Qcur`, `Kcur`,
    /// `Vcur`, `Qcur_rope`, `Kcur_rope`, `kqv_out`, `attn_out`, `ffn_inp`,
    /// `ffn_inp_normed`, `ffn_up_b`, `ffn_gelu`, `ffn_out`, `layer_out`, then
    /// `norm_b` at `layer = n_layers` for `v.post_ln`, and finally the
    /// projector's `ffn_up_b` / `ffn_gelu` with `layer = None`.
    ///
    /// Two reference checkpoints are not tapped: `ffn_up` and `ffn_down` are
    /// cb'd before their bias add (clip.cpp `build_ffn`), and
    /// `cpu_math::matmul` folds the bias into the same expression as the dot, so
    /// the pre-bias value is not a tensor this implementation ever holds.
    /// `ffn_up_b` / `ffn_out` are the post-bias twins of the same matmul.
    ///
    /// Every value handed to the tap is `[n_pos][row]` row-major, what ggml's
    /// `ne = [row, n_pos]` dumps to, so the buffers are directly comparable with
    /// no reshape.
    pub fn forward_tapped(
        &self,
        img: &[f32],
        width: usize,
        height: usize,
        tap: &mut dyn FnMut(&str, Option<usize>, &[f32]),
    ) -> Result<VitTrace> {
        let (d, hd, nh, p) = (self.n_embd, self.head_dim, self.n_head, self.patch);
        let step = p * self.merge;
        if width == 0 || height == 0 || width % step != 0 || height % step != 0 {
            bail!("image {width}x{height} must be a non-zero multiple of patch_size*merge = {step}");
        }
        if img.len() != self.channels * width * height {
            bail!("image buffer has {} values, expected {}x{}x{}", img.len(), self.channels, height, width);
        }
        let (pw, ph) = (width / p, height / p);
        let n_pos = pw * ph;

        // 1. patch embed — a gather + one matmul with the folded conv weight
        let rows = patchify(img, width, height, self.channels, p);
        let k = self.channels * p * p;
        let embedded = linear(&self.patch_w, d, k, &rows, None, self.threads, self.dotprod);

        // 2. spatial permute before block 0 (token count unchanged)
        let perm = merge_permutation(pw, ph);
        let mut x = vec![0f32; n_pos * d];
        for (dst, &src) in perm.iter().enumerate() {
            x[dst * d..(dst + 1) * d].copy_from_slice(&embedded[src * d..(src + 1) * d]);
        }

        // 3. + patch bias, + bilinearly-resized position embedding (permuted the
        //    same way; the reference applies the identical permute nest to it).
        //    Two separate passes, in the reference's order (`+ patch_bias`, then
        //    `+ learned_pos_embd`): folding them into `x + (b + p)` rounds
        //    differently and would cost bit-parity, while two passes give the
        //    same per-element expression and make the intermediate tappable.
        for dst in 0..n_pos {
            let row = &mut x[dst * d..(dst + 1) * d];
            for i in 0..d {
                row[i] += self.patch_b[i];
            }
        }
        tap("patch_bias", None, &x);
        let pos = resize_position_embeddings(&self.pos_embd, d, self.pos_side, pw, ph);
        for (dst, &src) in perm.iter().enumerate() {
            let row = &mut x[dst * d..(dst + 1) * d];
            let pe = &pos[src * d..(src + 1) * d];
            for i in 0..d {
                row[i] += pe[i];
            }
        }
        tap("inp_pos_emb", None, &x);

        // 4. transformer blocks
        let mpos = mrope_positions(pw, ph);
        debug_assert_eq!(mpos.len(), n_pos);
        let scale = 1.0 / (hd as f32).sqrt();
        for (il, blk) in self.blocks.iter().enumerate() {
            // pre-norm -> fused QKV
            let mut h = vec![0f32; n_pos * d];
            for t in 0..n_pos {
                let n = layernorm(&x[t * d..(t + 1) * d], &blk.ln1.0, &blk.ln1.1, self.eps);
                h[t * d..(t + 1) * d].copy_from_slice(&n);
            }
            tap("ln1", Some(il), &h);
            let qkv = linear(&blk.qkv, 3 * d, d, &h, Some(&blk.qkv_b), self.threads, self.dotprod);

            // split [Q | K | V] and rope Q/K per head
            let mut q = vec![0f32; n_pos * d];
            let mut kk = vec![0f32; n_pos * d];
            let mut vv = vec![0f32; n_pos * d];
            for t in 0..n_pos {
                let r = &qkv[t * 3 * d..(t + 1) * 3 * d];
                q[t * d..(t + 1) * d].copy_from_slice(&r[0..d]);
                kk[t * d..(t + 1) * d].copy_from_slice(&r[d..2 * d]);
                vv[t * d..(t + 1) * d].copy_from_slice(&r[2 * d..3 * d]);
            }
            // The reference cb's Qcur/Kcur/Vcur on the pre-rope views, so the
            // tap has to fire before the rotation, not after it.
            tap("Qcur", Some(il), &q);
            tap("Kcur", Some(il), &kk);
            tap("Vcur", Some(il), &vv);
            for t in 0..n_pos {
                for head in 0..nh {
                    let lo = t * d + head * hd;
                    vision_rope(&mut q[lo..lo + hd], hd, mpos[t], ROPE_BASE);
                    vision_rope(&mut kk[lo..lo + hd], hd, mpos[t], ROPE_BASE);
                }
            }
            tap("Qcur_rope", Some(il), &q);
            tap("Kcur_rope", Some(il), &kk);

            // bidirectional attention (no mask, no causality)
            let mut attn = vec![0f32; n_pos * d];
            let (qr, kr, vr) = (&q, &kk, &vv);
            par_rows(n_pos, d, self.threads, &mut attn, |t, out| {
                let mut sc = vec![0f32; n_pos];
                for head in 0..nh {
                    let qh = &qr[t * d + head * hd..t * d + (head + 1) * hd];
                    for (u, s) in sc.iter_mut().enumerate() {
                        *s = crate::cpu_math::dot_f32(qh, &kr[u * d + head * hd..u * d + (head + 1) * hd]) * scale;
                    }
                    let den = softmax_in_place(&mut sc);
                    // accumulate-then-divide (see `softmax_in_place`)
                    for i in 0..hd {
                        let mut acc = 0.0f32;
                        for (u, &s) in sc.iter().enumerate() {
                            acc += s * vr[u * d + head * hd + i];
                        }
                        out[head * hd + i] = acc / den;
                    }
                }
            });

            tap("kqv_out", Some(il), &attn);

            let o = linear(&blk.o, d, d, &attn, Some(&blk.o_b), self.threads, self.dotprod);
            tap("attn_out", Some(il), &o);
            for i in 0..n_pos * d {
                x[i] += o[i];
            }
            tap("ffn_inp", Some(il), &x);

            // pre-norm -> GELU MLP
            let mut h2 = vec![0f32; n_pos * d];
            for t in 0..n_pos {
                let n = layernorm(&x[t * d..(t + 1) * d], &blk.ln2.0, &blk.ln2.1, self.eps);
                h2[t * d..(t + 1) * d].copy_from_slice(&n);
            }
            tap("ffn_inp_normed", Some(il), &h2);
            let mut a = linear(&blk.up, self.ffn, d, &h2, Some(&blk.up_b), self.threads, self.dotprod);
            tap("ffn_up_b", Some(il), &a);
            for v in a.iter_mut() {
                *v = gelu(*v);
            }
            tap("ffn_gelu", Some(il), &a);
            let f2 = linear(&blk.down, d, self.ffn, &a, Some(&blk.down_b), self.threads, self.dotprod);
            tap("ffn_out", Some(il), &f2);
            for i in 0..n_pos * d {
                x[i] += f2[i];
            }
            tap("layer_out", Some(il), &x);
        }

        // 5. post-LN, then the qwen3vl_merger projector over 2x2-merged rows
        for t in 0..n_pos {
            let n = layernorm(&x[t * d..(t + 1) * d], &self.post_ln.0, &self.post_ln.1, self.eps);
            x[t * d..(t + 1) * d].copy_from_slice(&n);
        }
        // build_norm's post-layer-norm is the one cb("norm_b", il) the layer
        // loop does not rename, and it runs with il == n_layer.
        tap("norm_b", Some(self.n_layers), &x);
        // The 2x2 merge is a pure reshape here: the permute at step 2 already
        // made each group of 4 tokens spatial neighbours, so `[n_embd, n_pos]`
        // reinterpreted as `[n_embd*4, n_pos/4]` is the merged sequence.
        let kmm = d * self.merge * self.merge;
        let n_mm0 = self.mm0.1.len();
        let mm0 = linear(&self.mm0.0, n_mm0, kmm, &x, Some(&self.mm0.1), self.threads, self.dotprod);
        // The projector's build_ffn runs with il == -1, so its stages land on
        // bare names in the reference trace.
        tap("ffn_up_b", None, &mm0);
        let mut act = mm0.clone();
        for v in act.iter_mut() {
            *v = gelu(*v);
        }
        tap("ffn_gelu", None, &act);
        let out = linear(&self.mm2.0, self.proj_dim, n_mm0, &act, Some(&self.mm2.1), self.threads, self.dotprod);

        Ok(VitTrace { grid: (pw, ph), post_ln: x, mm0, out })
    }
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random (plain LCG), so a failure reproduces without
    /// a seed crate. The values only need to be non-degenerate.
    fn prand(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    fn conv_once(w: &[f32], n_embd: usize, rows: &[f32], k: usize) -> Vec<f32> {
        linear(&W::F32(w.to_vec()), n_embd, k, rows, None, 1, false)
    }

    // 1. the w0+w1 fold ------------------------------------------------------

    /// With exactly-representable integer weights and pixels every intermediate
    /// is exact, so `conv(w0+w1, x)` must equal `conv(w0,x) + conv(w1,x)` bit for
    /// bit. This pins the fold and the gather index order, with no
    /// floating-point slack to hide a wiring error.
    #[test]
    fn patch_fold_is_bit_exact_on_exact_inputs() {
        let (w, h, c, p, d) = (4usize, 6usize, 3usize, 2usize, 5usize);
        let k = c * p * p;
        let w0: Vec<f32> = (0..d * k).map(|i| ((i % 17) as f32) - 8.0).collect();
        let w1: Vec<f32> = (0..d * k).map(|i| ((i % 11) as f32) - 5.0).collect();
        let img: Vec<f32> = (0..c * w * h).map(|i| ((i % 13) as f32) - 6.0).collect();
        let rows = patchify(&img, w, h, c, p);

        let folded = fold_patch_weights(&w0, &w1).unwrap();
        let a = conv_once(&folded, d, &rows, k);
        let b0 = conv_once(&w0, d, &rows, k);
        let b1 = conv_once(&w1, d, &rows, k);
        let b: Vec<f32> = b0.iter().zip(&b1).map(|(x, y)| x + y).collect();

        assert_eq!(a.len(), (w / p) * (h / p) * d);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "element {i}: {x} vs {y}");
        }
    }

    /// Same identity on real-valued weights. Not bit-exact here — `Σ(a+b)x` and
    /// `Σax + Σbx` reassociate — so this is a wiring check at 1e-6, and the
    /// reason the fold is done in f32 rather than f16.
    #[test]
    fn patch_fold_matches_two_convs_on_random_inputs() {
        let (w, h, c, p, d) = (8usize, 4usize, 3usize, 2usize, 7usize);
        let k = c * p * p;
        let w0 = prand(d * k, 1);
        let w1 = prand(d * k, 2);
        let img = prand(c * w * h, 3);
        let rows = patchify(&img, w, h, c, p);

        let a = conv_once(&fold_patch_weights(&w0, &w1).unwrap(), d, &rows, k);
        let b0 = conv_once(&w0, d, &rows, k);
        let b1 = conv_once(&w1, d, &rows, k);
        for i in 0..a.len() {
            let e = (a[i] - (b0[i] + b1[i])).abs();
            assert!(e < 1e-6, "element {i}: {} vs {}", a[i], b0[i] + b1[i]);
        }
    }

    /// The gather must reproduce ggml's im2col row order `(ic, ky, kx)`, kx
    /// fastest, with patches in row-major `y*pw + x` order.
    #[test]
    fn patchify_row_order_matches_im2col() {
        let (w, h, c, p) = (4usize, 4usize, 2usize, 2usize);
        // img[c][y][x] = 100*c + 10*y + x
        let mut img = vec![0f32; c * w * h];
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    img[ch * h * w + y * w + x] = (100 * ch + 10 * y + x) as f32;
                }
            }
        }
        let rows = patchify(&img, w, h, c, p);
        let k = c * p * p;
        // patch (y=1, x=0) is token index 1*2 + 0 = 2; top-left pixel (2,0)
        let t2 = &rows[2 * k..3 * k];
        assert_eq!(t2, &[20.0, 21.0, 30.0, 31.0, 120.0, 121.0, 130.0, 131.0]);
    }

    // 2. the 2x2 spatial permute --------------------------------------------

    /// Synthetic grid where patch (y,x) holds `y*1000 + x`, so the permuted
    /// sequence is readable by eye and the expectation is written out in full.
    #[test]
    fn merge_permutation_order_square() {
        let (pw, ph) = (4usize, 4usize);
        let grid: Vec<f32> = (0..ph).flat_map(|y| (0..pw).map(move |x| (y * 1000 + x) as f32)).collect();
        let perm = merge_permutation(pw, ph);
        let got: Vec<f32> = perm.iter().map(|&s| grid[s]).collect();
        #[rustfmt::skip]
        let want: Vec<f32> = vec![
            // y-block 0, x-block 0 | x-block 2
            0.0, 1.0, 1000.0, 1001.0,   2.0, 3.0, 1002.0, 1003.0,
            // y-block 2, x-block 0 | x-block 2
            2000.0, 2001.0, 3000.0, 3001.0,   2002.0, 2003.0, 3002.0, 3003.0,
        ];
        assert_eq!(got, want);
        assert_eq!(perm.len(), pw * ph, "the merge is a reorder, not a reduction");
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..pw * ph).collect::<Vec<_>>(), "permutation must be a bijection");
    }

    /// Non-square grid: the nest is y-outer / x-inner and the row stride is pw,
    /// which a transposed implementation gets wrong only when pw != ph.
    #[test]
    fn merge_permutation_order_non_square() {
        let (pw, ph) = (6usize, 2usize);
        let grid: Vec<f32> = (0..ph).flat_map(|y| (0..pw).map(move |x| (y * 1000 + x) as f32)).collect();
        let got: Vec<f32> = merge_permutation(pw, ph).iter().map(|&s| grid[s]).collect();
        #[rustfmt::skip]
        let want: Vec<f32> = vec![
            0.0, 1.0, 1000.0, 1001.0,
            2.0, 3.0, 1002.0, 1003.0,
            4.0, 5.0, 1004.0, 1005.0,
        ];
        assert_eq!(got, want);
    }

    /// The M-RoPE position nest and the permute nest are the same loop; if they
    /// ever drift the encoder still runs and the output is subtly wrong.
    #[test]
    fn mrope_positions_track_the_permutation() {
        for (pw, ph) in [(4usize, 4usize), (6, 2), (2, 8)] {
            let perm = merge_permutation(pw, ph);
            let pos = mrope_positions(pw, ph);
            assert_eq!(perm.len(), pos.len());
            for (p, (&src, pv)) in perm.iter().zip(&pos).enumerate() {
                let (y, x) = (src / pw, src % pw);
                assert_eq!(pv, &[y as i32, x as i32, y as i32, x as i32], "slot {p} of {pw}x{ph}");
            }
        }
    }

    // 3. position-embedding resize ------------------------------------------

    /// clip.cpp:322 early-returns the tensor unmodified at the native grid. It
    /// must be bit-identical, not merely close: a resize with sf == 1 would
    /// round, and the 48x48 page is the one case checkable against the raw GGUF
    /// weight.
    #[test]
    fn position_embedding_48x48_early_returns_unmodified() {
        let (side, d) = (48usize, 3usize);
        let src = prand(side * side * d, 7);
        let out = resize_position_embeddings(&src, d, side, side, side);
        assert_eq!(out.len(), src.len());
        for (i, (a, b)) in out.iter().zip(&src).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "element {i}");
        }
    }

    /// Align-corners upsample, hand-computed: side 2 -> 3x3 gives sf = 2, so the
    /// sample points are 0, 0.5, 1 on both axes and every value is an exact
    /// binary fraction.
    #[test]
    fn position_embedding_bilinear_align_corners_upsample() {
        let src = vec![0.0f32, 1.0, 2.0, 3.0]; // grid (y,x): (0,0)=0 (0,1)=1 (1,0)=2 (1,1)=3
        let out = resize_position_embeddings(&src, 1, 2, 3, 3);
        let want = [0.0f32, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0];
        assert_eq!(out.len(), 9);
        for (i, (a, b)) in out.iter().zip(&want).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} vs {b}");
        }
    }

    /// Non-square target from a square source: sf0 and sf1 are computed
    /// independently, and a downscale on one axis must not leak into the other.
    /// Also pins the align-corners invariant — the four corners are exact
    /// source samples.
    #[test]
    fn position_embedding_bilinear_non_square_target() {
        let side = 3usize;
        // src[y][x] = 10*y + x
        let src: Vec<f32> = (0..side).flat_map(|y| (0..side).map(move |x| (10 * y + x) as f32)).collect();
        let (w, h) = (4usize, 2usize);
        let out = resize_position_embeddings(&src, 1, side, w, h);
        assert_eq!(out.len(), w * h);
        let want = [
            0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0, // y = 0
            20.0, 20.0 + 2.0 / 3.0, 20.0 + 4.0 / 3.0, 22.0, // y = 2 (sf1 = 0.5)
        ];
        for (i, (a, b)) in out.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-5, "element {i}: {a} vs {b}");
        }
        // align-corners: corners are exactly the source corners
        assert_eq!(out[0], src[0]);
        assert_eq!(out[w - 1], src[side - 1]);
        assert_eq!(out[(h - 1) * w], src[(side - 1) * side]);
        assert_eq!(out[h * w - 1], src[side * side - 1]);
    }

    /// The resize runs per channel; channel c must not read channel c-1.
    #[test]
    fn position_embedding_resize_is_per_channel() {
        let (side, d) = (2usize, 2usize);
        // channel 0 = 0,1,2,3 ; channel 1 = 100,101,102,103
        let src: Vec<f32> = (0..side * side).flat_map(|g| [g as f32, 100.0 + g as f32]).collect();
        let out = resize_position_embeddings(&src, d, side, 3, 3);
        for g in 0..9 {
            assert!((out[g * d + 1] - (out[g * d] + 100.0)).abs() < 1e-5, "grid {g}");
        }
    }

    // 4. vision M-RoPE -------------------------------------------------------

    /// With `t == h` the sectioned vision rope degenerates into two independent
    /// plain NEOX ropes of width `head_dim/2`, because `indep_sects` restarts
    /// the frequency ramp at each section boundary:
    ///
    ///   * section 0 covers pairs (j, j+32) for j in 0..16 — dims {0..16} ∪ {32..48}
    ///   * section 1 covers pairs (j, j+32) for j in 16..32 — dims {16..32} ∪ {48..64}
    ///
    /// so each group of 32 channels is exactly `cpu_math::rope(_, 32, pos, base)`
    /// (whose pairing is (i, i+16) over its own 32 values). It is not the same as
    /// a plain rope over all 64 dims. Tolerance rather than bit-equality because
    /// ggml builds the angle by repeated multiplication while `cpu_math::rope`
    /// uses `powf`; the repeated-multiply form is normative.
    #[test]
    fn vision_mrope_reduces_to_plain_rope_when_t_equals_h() {
        let hd = 64usize;
        let half = hd / 2;
        let base = ROPE_BASE;
        for pos in [0i32, 1, 7, 47] {
            let v0 = prand(hd, 11 + pos as u64);
            let mut got = v0.clone();
            vision_rope(&mut got, hd, [pos, pos, pos, pos], base);

            // group A: dims 0..16 and 32..48 ; group B: dims 16..32 and 48..64
            for (g, lo) in [(0usize, 0usize), (1, half / 2)] {
                let sect = half / 2; // 16
                let mut want: Vec<f32> = Vec::with_capacity(half);
                want.extend_from_slice(&v0[lo..lo + sect]);
                want.extend_from_slice(&v0[half + lo..half + lo + sect]);
                crate::cpu_math::rope(&mut want, half, pos as usize, base);
                for i in 0..sect {
                    let (a, b) = (got[lo + i], want[i]);
                    assert!((a - b).abs() < 1e-4, "pos {pos} group {g} low dim {i}: {a} vs {b}");
                    let (a, b) = (got[half + lo + i], want[sect + i]);
                    assert!((a - b).abs() < 1e-4, "pos {pos} group {g} high dim {i}: {a} vs {b}");
                }
            }
        }
    }

    /// The two sections must use different position channels: t drives dims
    /// 0..16 / 32..48, h drives dims 16..32 / 48..64. An implementation whose
    /// sections collapse to a plain rope passes the test above and fails this
    /// one.
    #[test]
    fn vision_mrope_sections_use_separate_position_channels() {
        let hd = 64usize;
        let v0 = prand(hd, 23);
        let mut a = v0.clone();
        let mut b = v0.clone();
        vision_rope(&mut a, hd, [3, 5, 3, 5], ROPE_BASE);
        vision_rope(&mut b, hd, [3, 9, 3, 9], ROPE_BASE); // only h differs
        for i in (0..16).chain(32..48) {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "dim {i} must depend on t only");
        }
        let moved = (16..32).chain(48..64).any(|i| (a[i] - b[i]).abs() > 1e-6);
        assert!(moved, "dims 16..32/48..64 must depend on h");
    }

    /// Position 0 is the identity (all angles are zero), for every channel.
    #[test]
    fn vision_mrope_at_zero_is_identity() {
        let hd = 64usize;
        let v0 = prand(hd, 31);
        let mut v = v0.clone();
        vision_rope(&mut v, hd, [0, 0, 0, 0], ROPE_BASE);
        for i in 0..hd {
            assert!((v[i] - v0[i]).abs() < 1e-6, "dim {i}");
        }
    }

    // 5. shared numerics -----------------------------------------------------

    /// LayerNorm: population variance, eps inside the sqrt. Both choices are
    /// checked against a closed form rather than a second implementation.
    #[test]
    fn layernorm_uses_population_variance_and_eps_inside_sqrt() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32; 4];
        let b = [0.0f32; 4];
        let eps = 1e-6f32;
        let got = layernorm(&x, &w, &b, eps);
        let mean = 2.5f32;
        let var = ((1.5f32).powi(2) * 2.0 + (0.5f32).powi(2) * 2.0) / 4.0; // population: /n
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..4 {
            assert!((got[i] - (x[i] - mean) * inv).abs() < 1e-6, "dim {i}");
        }
        // eps must be inside the sqrt: a constant row normalizes to 0, not NaN
        let flat = layernorm(&[5.0f32; 4], &w, &b, eps);
        assert!(flat.iter().all(|v| v.abs() < 1e-3), "constant row -> {flat:?}");
    }

    /// GELU is the tanh approximation, not erf. At x = 1 the two differ in the
    /// 4th decimal, which is enough to separate them.
    #[test]
    fn gelu_is_the_tanh_approximation() {
        assert!((gelu(0.0)).abs() < 1e-9);
        assert!((gelu(1.0) - 0.841_192_0).abs() < 1e-5, "{}", gelu(1.0));
        assert!((gelu(-1.0) + 0.158_808_0).abs() < 1e-5, "{}", gelu(-1.0));
        assert!((gelu(10.0) - 10.0).abs() < 1e-4);
    }

    /// accumulate-then-divide: the weights left behind are unnormalized and the
    /// returned denominator is their sum. The Metal side must make the same
    /// choice.
    #[test]
    fn softmax_convention_is_accumulate_then_divide() {
        let mut s = [1.0f32, 2.0, 3.0];
        let den = softmax_in_place(&mut s);
        assert!((s.iter().sum::<f32>() - den).abs() < 1e-6);
        assert!((s[2] - 1.0).abs() < 1e-6, "max element exponentiates to exactly 1");
        let probs: Vec<f32> = s.iter().map(|v| v / den).collect();
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    // 6. end-to-end wiring ---------------------------------------------------

    /// A tiny synthetic tower, exercised on both position-embedding paths: a
    /// grid equal to `pos_side` (early return) and a larger one (bilinear
    /// resize). Checks shapes, finiteness and that the merged token count is
    /// `n_pos / 4` — i.e. that the merge happened once, at the end.
    fn tiny_vit() -> CpuVit {
        let (d, nh, hd, ffn, p, c, side, proj) = (4usize, 1usize, 4usize, 8usize, 2usize, 3usize, 2usize, 3usize);
        let k = c * p * p;
        let mf = 4usize;
        CpuVit {
            n_embd: d, n_layers: 2, n_head: nh, head_dim: hd, ffn, patch: p, channels: c,
            pos_side: side, merge: 2, proj_dim: proj, eps: 1e-6,
            patch_w: W::F32(prand(d * k, 101)),
            patch_b: prand(d, 102),
            pos_embd: prand(side * side * d, 103),
            blocks: (0..2)
                .map(|i| Block {
                    ln1: (vec![1.0; d], vec![0.0; d]),
                    ln2: (vec![1.0; d], vec![0.0; d]),
                    qkv: W::F32(prand(3 * d * d, 200 + i)),
                    qkv_b: prand(3 * d, 210 + i),
                    o: W::F32(prand(d * d, 220 + i)),
                    o_b: prand(d, 230 + i),
                    up: W::F32(prand(ffn * d, 240 + i)),
                    up_b: prand(ffn, 250 + i),
                    down: W::F32(prand(d * ffn, 260 + i)),
                    down_b: prand(d, 270 + i),
                })
                .collect(),
            post_ln: (vec![1.0; d], vec![0.0; d]),
            mm0: (W::F32(prand(d * mf * d * mf, 301)), prand(d * mf, 302)),
            mm2: (W::F32(prand(proj * d * mf, 303)), prand(proj, 304)),
            threads: 1,
            dotprod: false,
        }
    }

    #[test]
    fn forward_shapes_on_both_position_embedding_paths() {
        let m = tiny_vit();
        for (w, h) in [(4usize, 4usize), (8, 4), (8, 8)] {
            let img = prand(m.channels * w * h, 999 + w as u64);
            let tr = m.forward_trace(&img, w, h).unwrap();
            let n_pos = (w / m.patch) * (h / m.patch);
            assert_eq!(tr.grid, (w / m.patch, h / m.patch));
            assert_eq!(tr.post_ln.len(), n_pos * m.n_embd);
            assert_eq!(tr.mm0.len(), n_pos / 4 * m.n_embd * 4);
            assert_eq!(tr.out.len(), m.n_merged_tokens(w, h) * m.proj_dim);
            assert_eq!(tr.out.len(), n_pos / 4 * m.proj_dim);
            assert!(tr.out.iter().all(|v| v.is_finite()), "{w}x{h} produced non-finite output");
        }
    }

    /// Non-multiples of `patch*merge` must be rejected, not silently truncated:
    /// the reference asserts the same thing (qwen2vl.cpp:6).
    #[test]
    fn forward_rejects_unaligned_images() {
        let m = tiny_vit();
        assert!(m.forward(&vec![0.0; m.channels * 6 * 4], 6, 4).is_err());
        assert!(m.forward(&vec![0.0; m.channels * 4 * 4 - 1], 4, 4).is_err());
    }

    /// Attention is bidirectional: token 0's output must depend on the last
    /// token's pixels. A causal mask left in by accident makes this constant.
    #[test]
    fn attention_is_bidirectional() {
        let m = tiny_vit();
        let (w, h) = (8usize, 4usize);
        let mut img = prand(m.channels * w * h, 555);
        let a = m.forward_trace(&img, w, h).unwrap().post_ln;
        // perturb the bottom-right patch only
        for c in 0..m.channels {
            for y in h - m.patch..h {
                for x in w - m.patch..w {
                    img[c * h * w + y * w + x] += 1.0;
                }
            }
        }
        let b = m.forward_trace(&img, w, h).unwrap().post_ln;
        let moved = (0..m.n_embd).any(|i| (a[i] - b[i]).abs() > 1e-6);
        assert!(moved, "token 0 did not see the last patch — attention is masked");
    }
}
