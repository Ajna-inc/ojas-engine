#![allow(clippy::too_many_arguments)]
//! The qwen3vl ViT tower and `qwen3vl_merger` projector on Metal, the vision half
//! of surya-2, entered through [`DecoderGpu::encode_image`].
//!
//! Twelve pre-norm blocks over the full patch count, LayerNorm-with-bias (not
//! RMSNorm), a bias on every projection, fused QKV, vision M-RoPE on Q and K in
//! every block, bidirectional attention, a GELU MLP, then `v.post_ln` and a
//! two-layer projector. The numerical oracle is `ojas_cpu::cpu_vit::CpuVit`;
//! `examples/vision_gate.rs` compares against it.
//!
//! A sibling entry rather than a decoder phase: `encode_span_phase`
//! (`graph_decode.rs:212`) asserts one causal decoder stack over `arch.layers` — a
//! KV cache per layer, a scalar position per token, causal masking, `self.d` as the
//! width — while the tower is bidirectional, 768 wide against the decoder's 1024,
//! has no KV cache, and has 2-D positions. Like `embed_ids` (`entries.rs:31`) it
//! gets its own entry and command buffer, runs before prefill, and hands back rows
//! that `prefill_embeds` injects.
//!
//! The twelve blocks are the shared encoder block (`encoder.rs`), which the text
//! encoder also runs; this file owns what is specific to the tower: the patch
//! embedding, the 2x2 merge, the position embedding, the M-RoPE table and the
//! projector. Every matmul goes through `projm` (`dispatch.rs`): all eight ViT
//! shapes satisfy its `N%64==0, K%32==0` guard, and it keeps the weight lookup inside
//! `Weights::repr`'s single probe order (`mod.rs`).
//! Structurally this is `forward_diffusion_range` (`batch.rs:33`) — already a
//! bidirectional, fully-biased, M-token encoder — with RMSNorm swapped for
//! LayerNorm and SwiGLU for GELU, including the `mk`/`big` idiom for working sets
//! past the arena's capacity and the 32-row padding cooperative `simdgroup_store`
//! forces.
//!
//! Four details are transcribed from `tools/mtmd/models/qwen3vl.cpp` and the CPU
//! oracle rather than derived from the config, each with a test on the oracle side:
//!
//! 1. The patch embed is a gather plus a matmul, not a convolution — 16x16
//!    stride-16 patches do not overlap — and the model's two patch-embed convs fold
//!    into one weight at load (`load.rs`, `VIT_PATCH_FOLD`), because for a still
//!    image the reference runs both over the same pixels and adds.
//! 2. The 2x2 spatial merge runs before block 0 and is a reorder, not a reduction.
//!    The token count is unchanged; it makes each group of four tokens a spatial
//!    block so the projector's `reshape(n_embd*4, n_pos/4)` picks up neighbours.
//!    Merging at the end instead is a common bug.
//! 3. The learned 48x48 position embedding is bilinearly resized with align_corners
//!    to the actual grid, and returns unmodified at exactly 48x48 — a 768x768 page,
//!    the common case.
//! 4. Vision M-RoPE is sectioned, `indep_sects` is on, and it does not reduce to a
//!    plain rope. See the `vit_rope` comment in `kernels/vision.rs`.
//!
//! Preprocessing is not here: [`DecoderGpu::encode_image`] takes an
//! already-normalized planar-CHW f32 buffer, as `CpuVit::forward` does, so the two
//! can be handed identical bytes and the encoder is testable independently of the
//! resampler (`ojas_cpu::VitPreproc`).

use super::*;
use anyhow::{ensure, Result};
use objc::{msg_send, sel, sel_impl};

/// Name of the folded patch-embed weight `load.rs` synthesizes from
/// `v.patch_embd.weight + v.patch_embd.weight.1`.
///
/// It needs its own name rather than shadowing `v.patch_embd.weight`: the on-disk
/// tensor is 4-D `[kw,kh,ic,oc]`, so `wshape` records `(16, 36864)`, which is the
/// wrong `(K, N)` for this GEMM and would trip `check_shape`.
pub(crate) const VIT_PATCH_FOLD: &str = "v.patch_embd.fold.weight";

/// Coarse checkpoints of one encode, the GPU twin of `ojas_cpu::cpu_vit::VitTrace`.
///
/// The three tensors localize a failure: a projector bug moves `out` but not
/// `post_ln`, a merge-order or M-RoPE bug moves `post_ln`, and a patch-embed bug
/// moves layer 0. They are the final contents of buffers the graph already wrote,
/// read after the single commit; the GELU is routed to a separate buffer so `mm0`
/// survives as its pre-activation value, matching the oracle's tap.
pub struct VitGpuTrace {
    /// Patch grid actually used, `(patches_x, patches_y)`.
    pub grid: (usize, usize),
    /// `v.post_ln` output, `[n_pos][d_v]`, in permuted token order.
    pub post_ln: Vec<f32>,
    /// `mm.0` output before the GELU, `[n_pos/4][mm_hidden]`.
    pub mm0: Vec<f32>,
    /// `mm.2` output — what the decoder consumes, `[n_pos/4][projection_dim]`.
    pub out: Vec<f32>,
    /// Residual stream after each block, `[n_pos][d_v]` each. Empty unless the
    /// caller asked for it; filling it costs one command buffer per layer.
    pub layer_out: Vec<Vec<f32>>,
    /// GPU execution time of the encode, from the command buffers' own timestamps.
    pub gpu_s: f64,
}

/// Destination-slot -> source-patch index for the 2x2 spatial merge.
///
/// Kept as a loop nest rather than a closed form so it stays a literal
/// transcription (`qwen3vl.cpp:18-31` unrolled; oracle
/// `ojas_cpu::cpu_vit::merge_permutation`). `vit_merge_permute` derives the same
/// mapping in closed form on the GPU, and `encode_image` debug_asserts the two
/// agree: if they drift, the encoder still runs and the page transcribes as almost
/// the right text.
fn merge_permutation(pw: usize, ph: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(pw * ph);
    for y in (0..ph).step_by(2) {
        for x in (0..pw).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 { perm.push((y + dy) * pw + (x + dx)); }
            }
        }
    }
    perm
}

/// The four M-RoPE position channels per token, in permuted token order.
///
/// Verbatim from `clip.cpp:4780`'s `PROJECTOR_TYPE_QWEN3VL` arm. The nest must stay
/// byte-for-byte [`merge_permutation`]'s: out of step with the permute, every block
/// ropes the wrong patch.
///
/// Channels 2 and 3 duplicate 0 and 1 and are never read at `hd = 64` (see
/// `vit_rope`); kept so the transcription stays literal.
fn mrope_positions(pw: usize, ph: usize) -> Vec<[u32; 4]> {
    let mut pos = Vec::with_capacity(pw * ph);
    for y in (0..ph).step_by(2) {
        for x in (0..pw).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    let (t, h) = ((y + dy) as u32, (x + dx) as u32);
                    pos.push([t, h, t, h]);
                }
            }
        }
    }
    pos
}

/// Bilinear resize of the learned position embedding, align_corners.
///
/// `src` is `[side*side][d]` row-major in grid index `g = y*side + x`
/// (`v.position_embd.weight` is `[d, side*side]` in GGUF `ne` order). Returns
/// `[height*width][d]` in the same convention.
///
/// The early return is part of the contract:
/// `clip_graph::resize_position_embeddings` (`clip.cpp:312`) returns the tensor
/// unmodified when the grid is already `side x side`, so a 768x768 page must be
/// bit-identical to the raw weight, not merely close as a resize with `sf == 1`
/// would be. The interpolation is `ggml_compute_forward_upscale_f32`'s
/// `BILINEAR | ALIGN_CORNERS` branch: pixel offset 0, `sf = (dst-1)/(src-1)`
/// falling back to `dst/src` when either side is 1, and corner indices clamped
/// before the fractional part is taken.
///
/// Duplicated from `ojas_cpu::cpu_vit::resize_position_embeddings` rather than
/// called, because `ojas-cpu` is only a dev-dependency here: the oracle cannot be a
/// runtime dependency of the thing it audits. `examples/vision_gate.rs` compares
/// the two end to end, which keeps them in step.
fn resize_position_embeddings(src: &[f32], d: usize, side: usize, width: usize, height: usize) -> Vec<f32> {
    if width == side && height == side { return src.to_vec(); }   // clip.cpp:322
    let sf0 = if width > 1 && side > 1 { (width - 1) as f32 / (side - 1) as f32 } else { width as f32 / side as f32 };
    let sf1 = if height > 1 && side > 1 { (height - 1) as f32 / (side - 1) as f32 } else { height as f32 / side as f32 };
    let mut out = vec![0f32; width * height * d];
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
            let (rc, rdd) = (y1 * side + x0, y1 * side + x1);
            let dst = (iy * width + ix) * d;
            for c in 0..d {
                let a = src[ra * d + c];
                let b = src[rb * d + c];
                let cc = src[rc * d + c];
                let e = src[rdd * d + c];
                out[dst + c] = a * (1.0 - dx) * (1.0 - dy) + b * dx * (1.0 - dy)
                    + cc * (1.0 - dx) * dy + e * dx * dy;
            }
        }
    }
    out
}

/// Host view of a shared Metal buffer's first `n` floats.
///
/// SAFETY: every buffer here is `StorageModeShared` and the caller has already
/// waited on the command buffer that wrote it.
fn readf(b: &metal::Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) }.to_vec()
}

impl<'a> DecoderGpu<'a> {
    /// The tower's geometry, or `None` when no mmproj was attached.
    pub fn has_vision(&self) -> bool { self.arch.vision.is_some() }

    /// Merged token count an image of this size produces — what the decoder sees,
    /// and the row count of [`DecoderGpu::encode_image`]'s output.
    pub fn n_merged_tokens(&self, width: usize, height: usize) -> Option<usize> {
        let v = self.arch.vision?;
        let (p, m) = (v.patch as usize, v.merge as usize);
        Some((width / p) * (height / p) / (m * m))
    }

    /// The merged token grid `(columns, rows)` an image of this size encodes to;
    /// [`DecoderGpu::encode_image`]'s rows are this grid in raster order.
    pub fn vision_grid(&self, width: usize, height: usize) -> Option<(usize, usize)> {
        let v = self.arch.vision?;
        let (p, m) = (v.patch as usize, v.merge as usize);
        Some((width / p / m, height / p / m))
    }

    /// The tower's patch side and spatial merge, which size its input.
    pub fn vision_patch_merge(&self) -> Option<(usize, usize)> {
        self.arch.vision.map(|v| (v.patch as usize, v.merge as usize))
    }

    /// Projector output width (`clip.vision.projection_dim`), which is by
    /// construction the decoder's embedding width — `mmproj::validate` refuses a
    /// file where it is not.
    pub fn vision_proj_dim(&self) -> Option<usize> { self.arch.vision.map(|v| v.proj_dim as usize) }

    /// Whether the tower's weights are resident in the mmproj's own f16 rather than
    /// requantized at load. The loader keeps them in f16 at every precision.
    ///
    /// Decides whether a comparison against `ojas_cpu::CpuVit` means anything: the
    /// oracle reads the GGUF's f16 directly, so a requantized tower would run
    /// different weights and the measured cosine would price the requantizer, not
    /// the encoder. On surya-2 at 36 patches, a tower requantized through
    /// `quantize_row_i8` gives a projector cosine of 0.998657 against 0.999999 in
    /// f16, with layer 0 already at 0.999930.
    pub fn vision_weights_f16(&self) -> bool {
        self.arch.vision.is_some() && self.wt.repr("v.blk.0.attn_qkv.weight") == Repr::F16
    }

    /// Encode one image. `img` is an already-normalized planar-CHW f32 buffer
    /// (`img[c*H*W + y*W + x]`, `channels*height*width` long — what
    /// `ojas_cpu::VitPreproc::preprocess` returns). Returns the projector output,
    /// `[n_merged_tokens * projection_dim]` row-major: exactly `prefill_embeds`'
    /// input.
    pub fn encode_image(&self, img: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        Ok(self.encode_image_trace(img, width, height)?.out)
    }

    /// [`DecoderGpu::encode_image`] plus the three coarse checkpoints.
    pub fn encode_image_trace(&self, img: &[f32], width: usize, height: usize) -> Result<VitGpuTrace> {
        self.encode_vit(img, width, height, false)
    }

    /// [`DecoderGpu::encode_image_trace`] plus the residual stream after every
    /// block, so a divergence against the CPU oracle localizes to a layer rather
    /// than being inferred from the projector output. Costs one command buffer per
    /// layer; the arithmetic is unchanged, since Metal orders dispatches within a
    /// serial encoder as it orders command buffers.
    pub fn encode_image_layer_trace(&self, img: &[f32], width: usize, height: usize) -> Result<VitGpuTrace> {
        self.encode_vit(img, width, height, true)
    }

    fn encode_vit(&self, img: &[f32], width: usize, height: usize, want_layers: bool) -> Result<VitGpuTrace> {
        let v = self.arch.vision
            .ok_or_else(|| anyhow::anyhow!("no vision tower: this model was loaded without an mmproj \
                                            (pass --mmproj or put the sidecar next to the GGUF)"))?;
        let (pp, mg, dv) = (v.patch as usize, v.merge as usize, v.d as usize);
        let step = pp * mg;
        ensure!(width > 0 && height > 0 && width % step == 0 && height % step == 0,
            "image {width}x{height} must be a non-zero multiple of patch_size*merge = {step}");
        let chan = v.channels as usize;
        ensure!(img.len() == chan * width * height,
            "image buffer has {} values, expected {chan}x{height}x{width}", img.len());
        let (pw, ph) = (width / pp, height / pp);
        let n_pos = pw * ph;

        // The loader keeps every tower weight in its f16 (`load.rs`). Check that held
        // before dispatching, rather than panicking on a missing key inside the
        // encoder. `Weights::repr` is the only legal probe order.
        for n in ["v.blk.0.attn_qkv.weight", VIT_PATCH_FOLD, "mm.0.weight", "mm.2.weight"] {
            let got = self.wt.repr(n);
            ensure!(got == Repr::F16,
                "vision weight {n} resolved to {got:?}; the tower is loaded in f16 at every precision");
        }

        let (nh, hd, ffn) = (v.n_head as usize, v.hd as usize, v.ffn as usize);
        let mmh = v.mm_hidden as usize;
        let kmm = mg * mg * dv;                    // projector contraction width
        let pd = v.proj_dim as usize;
        let kpatch = chan * pp * pp;               // patchify row width
        let n_mm = n_pos / (mg * mg);              // merged tokens
        ensure!(n_mm > 0, "image {width}x{height} yields no merged tokens");

        // Working set. GEMM outputs are stored in whole 32-row tiles (cooperative
        // simdgroup_store cannot skip sub-tile rows), so every row count is padded
        // to 32. The projector also reinterprets `x` as [n_mm, 4*d_v], and rounding
        // n_mm up to 32 can demand more tower rows than pad32(n_pos) at small grids.
        // Same reasoning and the same `mk`/`big` fallback as `batch.rs:44-62`.
        let pad32 = |r: usize| (r + 31) / 32 * 32;
        let mmrows = pad32(n_mm);
        let rows = pad32(n_pos).max(4 * mmrows);
        let big = n_pos > v.max_patches as usize;
        if big {
            tracing::debug!(target: "vision", "{n_pos} patches exceeds the {} the arena was sized for; \
                allocating temporaries (set OJAS_VIT_MAX_PATCHES={n_pos} to keep them resident)", v.max_patches);
        }
        let mk = |need: usize, base: &metal::Buffer| if big { buf(self.gpu, need) } else { base.clone() };
        let mkz = |need: usize, base: &metal::Buffer| if big { buf_zeroed(self.gpu, need) } else { base.clone() };
        let imgb = mk(chan * width * height, &self.st.vimg);
        let rowsb = mk((rows * kpatch).max(mmrows * mmh), &self.st.vrows);
        let x = mk(rows * dv, &self.st.vx);
        let h = mk(rows * dv, &self.st.vh);
        let qkv = mk(rows * 3 * dv, &self.st.vqkv);
        let q = mk(rows * dv, &self.st.vq);
        let khb = mkz(((rows + 32) * dv + 1) / 2, &self.st.vkh);
        let vhb = mkz(((rows + 32) * dv + 1) / 2, &self.st.vvh);
        let ffnb = mk((rows * ffn).max(mmrows * mmh), &self.st.vffn);
        let peb = mk(n_pos * dv, &self.st.vpe);
        let mposb = mk(4 + 4 * n_pos, &self.st.vmpos);
        let outb = mk(mmrows * pd, &self.st.vout);

        // ---- host inputs: pixels, the resized position embedding, the M-RoPE table
        // `st.*` are StorageModeShared, so a host write is visible to the next
        // command buffer (the property `span.rs:15` and `trainer.rs:637` rely on).
        unsafe { std::ptr::copy_nonoverlapping(img.as_ptr(), imgb.contents() as *mut f32, img.len()); }

        let perm = merge_permutation(pw, ph);
        debug_assert_eq!(perm.len(), n_pos, "the 2x2 merge is a reorder, not a reduction");
        // The GPU kernel derives the same mapping in closed form; assert they agree
        // for this grid rather than trusting the algebra.
        debug_assert!(
            (0..n_pos).all(|t| {
                let (b, r, hb) = (t / 4, t % 4, pw / 2);
                perm[t] == (2 * (b / hb) + r / 2) * pw + 2 * (b % hb) + r % 2
            }),
            "vit_merge_permute's closed form has drifted from merge_permutation at {pw}x{ph}");

        // The position embedding depends only on (pw, ph), so for same-size pages
        // this ~7 MB read + resize + permute is recomputed identically per page.
        // Caching it behind a `Cell<(usize, usize)>` is worth a few milliseconds
        // against a ~100 ms encode, so it is not done here.
        let pos_name = "v.position_embd.weight";
        let pos_buf = self.wt.w32.get(pos_name)
            .ok_or_else(|| anyhow::anyhow!("{pos_name} is not resident as f32 (repr {:?})", self.wt.repr(pos_name)))?;
        let side = v.pos_side as usize;
        let pe = resize_position_embeddings(&readf(pos_buf, side * side * dv), dv, side, pw, ph);
        let mut pe_perm = vec![0f32; n_pos * dv];
        for (dst, &src) in perm.iter().enumerate() {
            pe_perm[dst * dv..(dst + 1) * dv].copy_from_slice(&pe[src * dv..(src + 1) * dv]);
        }
        unsafe { std::ptr::copy_nonoverlapping(pe_perm.as_ptr(), peb.contents() as *mut f32, pe_perm.len()); }

        // Sections are `{hd/4} x 4` (qwen3vl.cpp:14), counted in cos/sin pairs, and
        // sum to hd (64) rather than to the hd/2 (32) rotated pairs that exist,
        // which is why sections 2 and 3 are unreachable. Passed as declared:
        // `vit_rope` reproduces ggml's arithmetic without pre-simplifying it.
        let sects = [v.hd / 4; 4];
        let desc = ojas_metal::kernels::ops::mrope_desc(sects, &mrope_positions(pw, ph));
        debug_assert_eq!(desc.len(), 4 + 4 * n_pos);
        unsafe { std::ptr::copy_nonoverlapping(desc.as_ptr(), mposb.contents() as *mut u32, desc.len()); }

        // ---- graph
        let (d32, m32) = (dv as u32, n_pos as u32);
        // MMA flash attention shares one KV pass across 32 queries; the per-(query,
        // head)-threadgroup kernel streams all of K and V per query, ~824 GB/layer
        // at page scale. MMA is Apple7+ only (`load.rs:344`'s `keep` closure drops
        // every `*mma*` kernel elsewhere), so probe for the pipeline rather than
        // assume it. OJAS_VIT_NO_MMA forces the streaming kernel on a GPU that has
        // both, which is how the non-Apple7 path gets exercised here;
        // `examples/vision_gate.rs` runs the same sizes under it.
        let mma = format!("attention_m_mma_bidir_{hd}");
        let use_mma = self.p.contains_key(&mma) && !ojas_core::config::flag("OJAS_VIT_NO_MMA");
        let mut gpu_s = 0.0f64;
        let mut layer_out: Vec<Vec<f32>> = Vec::new();
        let t0 = std::time::Instant::now();

        let mut cb = self.gpu.command_buffer();
        let mut enc = cb.new_compute_command_encoder();

        // 1. patch embed: gather + one matmul with the folded conv weight.
        self.enc_reduce(&enc, "vit_patchify", &[(&imgb, 0), (&rowsb, 1)],
            &[(2, width as u32), (3, height as u32), (4, chan as u32), (5, pp as u32),
              (6, (n_pos * kpatch) as u32)],
            &[], ((n_pos * kpatch + 255) / 256) as u64, 256);
        self.projm(&enc, &rowsb, 0, VIT_PATCH_FOLD, &h, kpatch as u32, d32, m32, false);
        // 2. the 2x2 spatial permute, before block 0. Token count unchanged.
        self.enc_reduce(&enc, "vit_merge_permute", &[(&h, 0), (&x, 1)],
            &[(2, d32), (3, pw as u32), (4, (n_pos * dv) as u32)],
            &[], ((n_pos * dv + 63) / 64) as u64, 64);
        // 3. + patch bias, then + the permuted position embedding: two passes, in
        //    the reference's order. Folding them into `x + (bias + pe)` rounds
        //    differently for no gain; kept apart, this is the same per-element
        //    expression the oracle evaluates. `add_rowbias_m` with N == total is a
        //    plain elementwise add (`x[g] += b[g % N]`), which is how the full
        //    [n_pos, d] position tensor rides a bias kernel.
        self.enc_reduce(&enc, "add_rowbias_m", &[(&x, 0), (&self.wt.w32["v.patch_embd.bias"], 1)],
            &[(2, d32), (3, (n_pos * dv) as u32)], &[], ((n_pos * dv + 63) / 64) as u64, 64);
        self.enc_reduce(&enc, "add_rowbias_m", &[(&x, 0), (&peb, 1)],
            &[(2, (n_pos * dv) as u32), (3, (n_pos * dv) as u32)], &[], ((n_pos * dv + 63) / 64) as u64, 64);

        // 4. twelve pre-norm blocks, through the shared encoder block
        let geom = encoder::Geom { d: dv, n_head: nh, hd, ffn, eps: v.eps };
        let scratch = encoder::Scratch {
            x: &x, h: &h, qkv: &qkv, q: &q, kh: &khb, vh: &vhb,
            ffn: &ffnb, ffn_wide: &ffnb, pos: &mposb,
        };
        for l in 0..v.layers as usize {
            let p = |s: &str| format!("v.blk.{l}.{s}");
            let norm = |n: &str| encoder::Norm { weight: p(&format!("{n}.weight")), bias: Some(p(&format!("{n}.bias"))) };
            let block = encoder::Block {
                attn_norm: Some(norm("ln1")),
                qkv: p("attn_qkv.weight"), qkv_bias: Some(p("attn_qkv.bias")),
                out: p("attn_out.weight"), out_bias: Some(p("attn_out.bias")),
                ffn_norm: norm("ln2"),
                up: p("ffn_up.weight"), up_bias: Some(p("ffn_up.bias")),
                down: p("ffn_down.weight"), down_bias: Some(p("ffn_down.bias")),
                mlp: encoder::Mlp::Plain,
                act: encoder::Act::GeluTanh,
                // ggml's vision rope spreads the frequency ramp over half the head.
                rope: Some(encoder::Rope { base: v.rope_base, freq_dims: v.hd / 2 }),
            };
            self.encode_block(&enc, &block, &geom, &scratch, encoder::Keys::All { mma: use_mma }, n_pos);
            if want_layers {
                enc.end_encoding();
                gpu_s += commit_vit(cb, &format!("vit block {l}"))?;
                layer_out.push(readf(&x, n_pos * dv));
                cb = self.gpu.command_buffer();
                enc = cb.new_compute_command_encoder();
            }
        }

        // 5. post-LN, in place: `vit_layernorm_m`'s reductions both complete before
        //    its store loop, and a thread only rewrites what it read. Then the
        //    projector over 2x2-merged rows. The merge is a pure reshape — step 2
        //    already made every group of four tokens spatial neighbours, so
        //    [n_pos, d_v] read as [n_pos/4, 4*d_v] is the merged sequence, and
        //    `projm`'s K does the reinterpreting.
        let post_ln = encoder::Norm { weight: "v.post_ln.weight".into(), bias: Some("v.post_ln.bias".into()) };
        self.enc_layernorm(&enc, &x, &x, &post_ln, d32, v.eps, m32);
        self.projm(&enc, &x, 0, "mm.0.weight", &ffnb, kmm as u32, mmh as u32, n_mm as u32, false);
        self.enc_reduce(&enc, "add_rowbias_m", &[(&ffnb, 0), (&self.wt.w32["mm.0.bias"], 1)],
            &[(2, mmh as u32), (3, (n_mm * mmh) as u32)], &[], ((n_mm * mmh + 63) / 64) as u64, 64);
        // GELU to a DIFFERENT buffer, so `ffnb` survives the encode as mm.0's
        // pre-activation value — the checkpoint the oracle taps as `ffn_up_b`.
        self.enc_act(&enc, &ffnb, &rowsb, encoder::Act::GeluTanh, (n_mm * mmh) as u32);
        self.projm(&enc, &rowsb, 0, "mm.2.weight", &outb, mmh as u32, pd as u32, n_mm as u32, false);
        self.enc_reduce(&enc, "add_rowbias_m", &[(&outb, 0), (&self.wt.w32["mm.2.bias"], 1)],
            &[(2, pd as u32), (3, (n_mm * pd) as u32)], &[], ((n_mm * pd + 63) / 64) as u64, 64);
        enc.end_encoding();
        gpu_s += commit_vit(cb, "vit encode")?;
        self.gpu_s.set(self.gpu_s.get() + gpu_s);
        tracing::debug!(target: "vision", "encoded {width}x{height} -> {pw}x{ph} patches -> {n_mm} tokens \
            in {:.1} ms wall / {:.1} ms gpu ({})", t0.elapsed().as_secs_f64() * 1e3, gpu_s * 1e3,
            if use_mma { "mma" } else { "streaming" });

        Ok(VitGpuTrace {
            grid: (pw, ph),
            post_ln: readf(&x, n_pos * dv),
            mm0: readf(&ffnb, n_mm * mmh),
            out: readf(&outb, n_mm * pd),
            layer_out,
            gpu_s,
        })
    }
}

/// Commit one vision command buffer and return its GPU time.
///
/// `commit_and_wait_checked` is the only place in the tree that surfaces an
/// `MTLCommandBufferError` to the caller, and a vision encode is the shape that
/// trips error 8 (OutOfMemory) at page scale, so the status is checked.
fn commit_vit(cb: &metal::CommandBufferRef, label: &str) -> Result<f64> {
    ojas_metal::commit_and_wait_checked(cb, label).map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
    Ok((ge - gs).max(0.0))
}
