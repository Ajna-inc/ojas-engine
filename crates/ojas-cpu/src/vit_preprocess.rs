//! Document-image preprocessing for a Qwen3-VL-style vision encoder — the
//! front half of the OCR path.
//!
//! A bit-compatible port of llama.cpp's mtmd preprocessor
//! (`tools/mtmd/mtmd-image.cpp` at revision `434ddbbc0`); each ported routine
//! cites its file:line. Every downstream number — patch embeddings, ViT
//! activations, decoder logits — depends on the exact bytes this file
//! produces, so an approximate resize puts logit parity out of reach.
//!
//! Three details are load-bearing:
//!
//! 1. The size policy is not "scale to fit". `calc_size_preserved_ratio`
//!    (mtmd-image.cpp:122) rounds each edge to a multiple of
//!    `patch_size * n_merge` first, and only then pulls the area into the
//!    `[min_pixels, max_pixels]` budget — flooring by the factor when over
//!    budget, ceiling when under. Round/floor/ceil are not interchangeable.
//!    "smart_resize" in the transformers code is the same function.
//! 2. The resampler works in the u8 domain with fixed-point weights.
//!    `resize_pillow` (mtmd-image.cpp:195) is Pillow's separable resampler
//!    with 22 fractional bits and a `clip8` per output pixel; the per-pass
//!    rounding and the [0,255] clamp are part of the algorithm, so resampling
//!    in f32 and quantizing afterwards gives different bytes. Bicubic here is
//!    Pillow's a = -0.5, not the a = -0.75 that GGML/PyTorch/OpenCV use.
//! 3. Qwen3-VL resizes with `PAD_CEIL`, not a stretch. `image_resize_pad`
//!    defaults to `PAD_CEIL` (clip-model.h:67) and the qwen3vl branch
//!    (clip.cpp:1651-1668) never overrides it: the image is scaled by
//!    `min(tw/sw, th/sh)`, ceil'd, and composited centred into a black canvas
//!    of the target size. One axis is exact, the other carries up to
//!    `align_size - 1` rows/columns of padding.
//!
//! The revision pin matters. At `434ddbbc0` the Qwen3-VL arm of `clip.cpp`
//! sets `image_resize_algo = RESIZE_ALGO_BICUBIC` and `img_tool::resize`
//! routes every algo through `resize_pillow`, which is what this file ports
//! and what agrees with Pillow — and therefore with the `transformers`
//! `Qwen2VLImageProcessor` the checkpoint was trained through. At `bb4caa754`
//! (llama.cpp 0.2.0) the same arm sets `RESIZE_ALGO_BILINEAR`, which
//! dispatches to `resize_bilinear`: a naive ALIGN_CORNERS lerp with a
//! truncating `uint8_t` cast, not Pillow's triangle filter, and not
//! expressible as a [`ResizeAlgo`] here. On a 150 dpi page
//! (900x1450 -> 896x1440) the two differ on 1.14% of bytes by up to 8/255, so
//! a 0.2.0 `inp_raw` dump is not a valid oracle for this file; validate the
//! encoder with `OJAS_VIT_INP_RAW` (examples/vit_parity_gate.rs) until the
//! oracle is rebuilt from the pinned revision.
//!
//! Parameters for surya-2, read off the GGUF: `patch_size = 16`,
//! `spatial_merge_size = 2` → `align_size = 32`;
//! `set_limit_image_tokens(8, 4096)` (clip.cpp:1660) with
//! `patch_area = 16*16*2*2 = 1024` → 8 192 … 4 194 304 pixels; BICUBIC
//! (clip.cpp:1656); `image_mean = image_std = [0.5, 0.5, 0.5]`.
//!
//! Output layout is planar CHW f32 — `idx = c*(W*H) + y*W + x` — matching what
//! llama.cpp builds into `inp_raw` (clip.cpp:4530-4565) and what the patch
//! embed wants: `v.patch_embd.weight` is `[16,16,3,768]` in ggml `ne` order
//! (dims[0] fastest), so one output channel's 768 weights run x-fastest, then
//! y, then c. Patchifying planar CHW into `[T, 768]` rows is a pure gather
//! with no transpose.
//!
//! Pure CPU: index and integer arithmetic only, no Metal and no model. The
//! golden values in the test module were produced by compiling the reference
//! routines standalone and printing their output.

use anyhow::{bail, Result};

/// Pillow-compatible resampling kernels. Mirrors `resize_algo`
/// (clip-model.h:33).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeAlgo {
    Bilinear,
    Bicubic,
    Lanczos,
}

/// Padding style for [`resize`]. Mirrors `pad_style` (clip-model.h:43).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadStyle {
    /// No padding; direct (aspect-breaking) resize to the target dimensions.
    None,
    /// Aspect-preserving pad, `ceil` rounding. The llama.cpp default and what
    /// Qwen3-VL uses.
    Ceil,
    /// Aspect-preserving pad, nearest-integer rounding (Pillow byte-parity).
    Nearest,
}

// ---------------------------------------------------------------------------
// size policy — port of img_tool::calc_size_preserved_ratio, mtmd-image.cpp:122
// ---------------------------------------------------------------------------

/// Options for [`calc_size_preserved_ratio`]. Field-for-field
/// `img_tool::calc_size_opt` (mtmd-image.cpp:107). `0` disables a limit.
#[derive(Clone, Copy, Debug)]
pub struct CalcSizeOpt {
    pub align_size: i32,
    pub min_pixels: i32,
    pub max_pixels: i32,
    /// Applied *before* min/max_pixels, so min_pixels can push an edge back
    /// above it. Unused by Qwen3-VL (it takes the round-by-factor branch).
    pub longest_edge: i32,
}

impl Default for CalcSizeOpt {
    fn default() -> Self {
        CalcSizeOpt { align_size: 1, min_pixels: 0, max_pixels: 0, longest_edge: 0 }
    }
}

// Rounding helpers; lambdas in the reference (mtmd-image.cpp:128-130). The
// divide and round happen in f32, not f64, and not on the product — keep the
// types.
#[inline]
fn round_by_factor(x: f32, f: i32) -> i32 {
    (x / f as f32).round() as i32 * f
}
#[inline]
fn ceil_by_factor(x: f32, f: i32) -> i32 {
    (x / f as f32).ceil() as i32 * f
}
#[inline]
fn floor_by_factor(x: f32, f: i32) -> i32 {
    (x / f as f32).floor() as i32 * f
}

/// Size of the resized image: aspect ratio preserved, each edge a multiple of
/// `align_size`, area pulled into `[min_pixels, max_pixels]`. Takes and
/// returns `(width, height)`. ("smart_resize" in the transformers code.)
///
/// Port of `img_tool::calc_size_preserved_ratio`, mtmd-image.cpp:122-157, with
/// one intentional deviation: area comparisons are done in `i64` where the
/// reference multiplies two `int`s and overflows above ~2^31 pixels (a
/// 46 341² image). Every in-range input agrees exactly.
pub fn calc_size_preserved_ratio(size: (i32, i32), o: &CalcSizeOpt) -> (i32, i32) {
    assert!(o.align_size > 0, "align_size must be > 0");
    let (width, height) = size;
    if width <= 0 || height <= 0 {
        return (0, 0);
    }

    let (mut w_bar, mut h_bar);
    if o.longest_edge > 0 {
        let scale =
            (o.longest_edge as f32 / width as f32).min(o.longest_edge as f32 / height as f32);
        w_bar = ceil_by_factor(width as f32 * scale, o.align_size);
        h_bar = ceil_by_factor(height as f32 * scale, o.align_size);
    } else {
        // always align up first
        w_bar = o.align_size.max(round_by_factor(width as f32, o.align_size));
        h_bar = o.align_size.max(round_by_factor(height as f32, o.align_size));
    }

    let area = h_bar as i64 * w_bar as i64;
    if o.max_pixels > 0 && area > o.max_pixels as i64 {
        // over budget: shrink by sqrt of the area ratio, then floor by factor
        let beta = (height as f32 * width as f32 / o.max_pixels as f32).sqrt();
        h_bar = o.align_size.max(floor_by_factor(height as f32 / beta, o.align_size));
        w_bar = o.align_size.max(floor_by_factor(width as f32 / beta, o.align_size));
    } else if o.min_pixels > 0 && area < o.min_pixels as i64 {
        // under budget: grow by sqrt of the area ratio, then ceil by factor.
        // The reference has no `max(align_size, ..)` clamp on this branch.
        let beta = (o.min_pixels as f32 / (height as f32 * width as f32)).sqrt();
        h_bar = ceil_by_factor(height as f32 * beta, o.align_size);
        w_bar = ceil_by_factor(width as f32 * beta, o.align_size);
    }

    (w_bar, h_bar)
}

// ---------------------------------------------------------------------------
// resampler — port of img_tool::resize_pillow, mtmd-image.cpp:195-470
// ---------------------------------------------------------------------------

/// 22 = 32 (i32) − 8 (u8 pixels) − 2 (accumulation headroom). mtmd-image.cpp:203.
const PRECISION_BITS: i32 = 32 - 8 - 2;

/// Filter weight at distance `x` from a pixel centre. mtmd-image.cpp:226-255.
///
/// Bicubic uses Pillow's a = -0.5; GGML/PyTorch use a = -0.75 and produce
/// visibly different bytes. All arithmetic is `f64`, as in the reference.
#[inline]
fn resample_filter(x: f64, algo: ResizeAlgo) -> f64 {
    if algo == ResizeAlgo::Lanczos {
        if (-3.0..3.0).contains(&x) {
            fn sinc(v: f64) -> f64 {
                if v == 0.0 {
                    return 1.0;
                }
                let pi_v = v * std::f64::consts::PI;
                pi_v.sin() / pi_v
            }
            return sinc(x) * sinc(x / 3.0);
        }
        return 0.0;
    }

    let x = if x < 0.0 { -x } else { x };

    if algo == ResizeAlgo::Bilinear {
        return if x < 1.0 { 1.0 - x } else { 0.0 };
    }

    const A: f64 = -0.5;
    if x < 1.0 {
        return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    }
    if x < 2.0 {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
    }
    0.0 // zero outside [-2, 2]
}

/// Clamp a de-scaled fixed-point accumulator to `u8`. mtmd-image.cpp:258-262.
#[inline]
fn clip8(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

/// Filter coefficients for one dimension. Returns
/// `(ksize, bounds, weights)` where `bounds[xx*2+0] = xmin`,
/// `bounds[xx*2+1] = count`, and `weights[xx*ksize + i]` is the fixed-point
/// weight of input pixel `xmin + i`.
///
/// Port of the `precompute_weights` lambda, mtmd-image.cpp:278-358. `f64`
/// throughout, matching the reference's `double`.
fn precompute_weights(
    in_size: i32,
    out_size: i32,
    algo: ResizeAlgo,
) -> (usize, Vec<i32>, Vec<i32>) {
    assert!(in_size > 0 && out_size > 0);

    let filter_support: f64 = match algo {
        ResizeAlgo::Bilinear => 1.0,
        ResizeAlgo::Bicubic => 2.0,
        ResizeAlgo::Lanczos => 3.0,
    };

    // ratio of input range to output size
    let scale = in_size as f64 / out_size as f64;
    // upsampling (scale < 1) keeps the filter sharp; downsampling widens it
    let filterscale = if scale < 1.0 { 1.0 } else { scale };

    let support = filter_support * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;

    let out_n = out_size as usize;
    let mut pre_weights = vec![0f64; out_n * ksize];
    let mut bounds = vec![0i32; out_n * 2];

    for xx in 0..out_n {
        // pixel-centre convention: +0.5
        let center = (xx as f64 + 0.5) * scale;
        let mut ww = 0.0f64;
        let ss = 1.0 / filterscale;

        // `as i32` truncates toward zero like C's `(int)` cast; the value can
        // only be negative here, where the clamp to 0 makes trunc and floor
        // agree.
        let mut xmin = (center - support + 0.5) as i32;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i32;
        if xmax > in_size {
            xmax = in_size;
        }
        xmax -= xmin; // now a count
        let xcnt = xmax.max(0) as usize;
        debug_assert!(xcnt <= ksize);

        for x in 0..xcnt {
            let w = resample_filter((x as f64 + xmin as f64 - center + 0.5) * ss, algo);
            pre_weights[xx * ksize + x] = w;
            ww += w;
        }
        // normalize to sum 1.0 (preserves brightness)
        if ww != 0.0 {
            for x in 0..xcnt {
                pre_weights[xx * ksize + x] /= ww;
            }
        }
        // remaining kernel slots stay zero (already zero-initialized)

        bounds[xx * 2] = xmin;
        bounds[xx * 2 + 1] = xcnt as i32;
    }

    // float -> fixed point. Pillow adds +/- 0.5 then truncates toward zero;
    // `round()` would round twice. mtmd-image.cpp:348-356.
    let fxp_scale = (2.0f64).powi(PRECISION_BITS);
    let weights: Vec<i32> = pre_weights
        .iter()
        .map(|&w| (w * fxp_scale + if w < 0.0 { -0.5 } else { 0.5 }) as i32)
        .collect();

    (ksize, bounds, weights)
}

/// Horizontal pass: width `in_nx` -> `out_nx`, height untouched.
/// mtmd-image.cpp:360-400.
fn resample_horizontal(
    src: &[u8],
    in_nx: usize,
    in_ny: usize,
    out_nx: usize,
    ksize: usize,
    bounds: &[i32],
    weights: &[i32],
) -> Vec<u8> {
    let mut out = vec![0u8; out_nx * in_ny * 3];
    for yy in 0..in_ny {
        let src_row = &src[yy * in_nx * 3..(yy + 1) * in_nx * 3];
        let dst_row = &mut out[yy * out_nx * 3..(yy + 1) * out_nx * 3];
        for xx in 0..out_nx {
            let xmin = bounds[xx * 2] as usize;
            let xcnt = bounds[xx * 2 + 1] as usize;
            let k = &weights[xx * ksize..xx * ksize + ksize];
            // accumulators pre-loaded with the 0.5 rounding bias
            let mut ss0 = 1i32 << (PRECISION_BITS - 1);
            let mut ss1 = 1i32 << (PRECISION_BITS - 1);
            let mut ss2 = 1i32 << (PRECISION_BITS - 1);
            let p = &src_row[xmin * 3..(xmin + xcnt) * 3];
            for x in 0..xcnt {
                ss0 += p[x * 3] as i32 * k[x];
                ss1 += p[x * 3 + 1] as i32 * k[x];
                ss2 += p[x * 3 + 2] as i32 * k[x];
            }
            dst_row[xx * 3] = clip8(ss0 >> PRECISION_BITS);
            dst_row[xx * 3 + 1] = clip8(ss1 >> PRECISION_BITS);
            dst_row[xx * 3 + 2] = clip8(ss2 >> PRECISION_BITS);
        }
    }
    out
}

/// Vertical pass: height -> `out_ny`, width untouched. Accumulates whole rows.
/// mtmd-image.cpp:402-438.
fn resample_vertical(
    src: &[u8],
    in_nx: usize,
    out_ny: usize,
    ksize: usize,
    bounds: &[i32],
    weights: &[i32],
) -> Vec<u8> {
    let row_elems = in_nx * 3;
    let mut out = vec![0u8; row_elems * out_ny];
    let mut acc = vec![0i32; row_elems];
    for yy in 0..out_ny {
        let ymin = bounds[yy * 2] as usize;
        let ycnt = bounds[yy * 2 + 1] as usize;
        let k = &weights[yy * ksize..yy * ksize + ksize];
        acc.iter_mut().for_each(|a| *a = 1i32 << (PRECISION_BITS - 1));
        for y in 0..ycnt {
            let src_row = &src[(ymin + y) * row_elems..(ymin + y + 1) * row_elems];
            let w = k[y];
            for i in 0..row_elems {
                acc[i] += src_row[i] as i32 * w;
            }
        }
        let dst_row = &mut out[yy * row_elems..(yy + 1) * row_elems];
        for i in 0..row_elems {
            dst_row[i] = clip8(acc[i] >> PRECISION_BITS);
        }
    }
    out
}

/// Pillow-compatible separable resample of an interleaved RGB u8 image.
/// Horizontal pass then vertical pass, each in the u8 domain.
///
/// Port of `img_tool::resize_pillow`, mtmd-image.cpp:195-470.
pub fn resize_pillow(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    algo: ResizeAlgo,
) -> Vec<u8> {
    assert_eq!(src.len(), src_w * src_h * 3, "src is not W*H*3 interleaved RGB");

    let need_h = dst_w != src_w;
    let need_v = dst_h != src_h;

    match (need_h, need_v) {
        (true, true) => {
            let (kh, bh, wh) = precompute_weights(src_w as i32, dst_w as i32, algo);
            let (kv, bv, wv) = precompute_weights(src_h as i32, dst_h as i32, algo);
            let tmp = resample_horizontal(src, src_w, src_h, dst_w, kh, &bh, &wh);
            resample_vertical(&tmp, dst_w, dst_h, kv, &bv, &wv)
        }
        (true, false) => {
            let (kh, bh, wh) = precompute_weights(src_w as i32, dst_w as i32, algo);
            resample_horizontal(src, src_w, src_h, dst_w, kh, &bh, &wh)
        }
        (false, true) => {
            let (kv, bv, wv) = precompute_weights(src_h as i32, dst_h as i32, algo);
            resample_vertical(src, src_w, dst_h, kv, &bv, &wv)
        }
        (false, false) => src.to_vec(),
    }
}

/// Resize to exactly `(dst_w, dst_h)`, optionally aspect-preserving with a
/// centred pad. Returns an interleaved RGB u8 buffer of `dst_w*dst_h*3`.
///
/// Port of `img_tool::resize`, mtmd-image.cpp:39-90.
pub fn resize(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    algo: ResizeAlgo,
    padding: PadStyle,
    pad_color: [u8; 3],
) -> Vec<u8> {
    assert_eq!(src.len(), src_w * src_h * 3, "src is not W*H*3 interleaved RGB");

    // same size -> plain copy, checked before the padding branch
    if dst_w == src_w && dst_h == src_h {
        return src.to_vec();
    }

    if padding == PadStyle::None {
        return resize_pillow(src, src_w, src_h, dst_w, dst_h, algo);
    }

    // aspect-preserving: one axis lands exactly on the target, the other is
    // short by up to align_size-1 and gets centred in pad_color.
    let scale_w = dst_w as f32 / src_w as f32;
    let scale_h = dst_h as f32 / src_h as f32;
    let scale = scale_w.min(scale_h);

    let (new_w, new_h) = if padding == PadStyle::Nearest {
        (
            ((src_w as f32 * scale).round() as i64).min(dst_w as i64) as usize,
            ((src_h as f32 * scale).round() as i64).min(dst_h as i64) as usize,
        )
    } else {
        (
            ((src_w as f32 * scale).ceil() as i64).min(dst_w as i64) as usize,
            ((src_h as f32 * scale).ceil() as i64).min(dst_h as i64) as usize,
        )
    };

    let resized = resize_pillow(src, src_w, src_h, new_w, new_h, algo);

    // fill with pad_color, then composite (mtmd-image.cpp:159-193)
    let mut dst = vec![0u8; dst_w * dst_h * 3];
    for px in dst.chunks_exact_mut(3) {
        px.copy_from_slice(&pad_color);
    }

    let (off_x, off_y) = if padding == PadStyle::Nearest {
        (
            ((dst_w as f32 - new_w as f32) / 2.0).round() as i64,
            ((dst_h as f32 - new_h as f32) / 2.0).round() as i64,
        )
    } else {
        ((dst_w as i64 - new_w as i64) / 2, (dst_h as i64 - new_h as i64) / 2)
    };

    for y in 0..new_h {
        let dy = y as i64 + off_y;
        if dy < 0 || dy >= dst_h as i64 {
            continue;
        }
        for x in 0..new_w {
            let dx = x as i64 + off_x;
            if dx < 0 || dx >= dst_w as i64 {
                continue;
            }
            let si = (y * new_w + x) * 3;
            let di = (dy as usize * dst_w + dx as usize) * 3;
            dst[di..di + 3].copy_from_slice(&resized[si..si + 3]);
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// normalization + layout
// ---------------------------------------------------------------------------

/// Interleaved RGB u8 -> planar CHW f32, normalized per channel.
///
/// `out[c*(w*h) + y*w + x] = (rgb[3*(y*w+x)+c] / 255 - mean[c]) / std[c]`.
///
/// The `/255` then `(v-mean)/std` split is `clip_image_f32::from_u8`
/// (clip-impl.h:707) followed by `::normalize` (clip-impl.h:726). Keep both
/// steps: for `mean = std = 0.5` they collapse algebraically to `x/127.5 - 1`,
/// but the collapsed form is not bit-identical in f32. The CHW transpose is
/// clip.cpp:4530-4565.
pub fn normalize_planar(
    rgb: &[u8],
    w: usize,
    h: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Vec<f32> {
    assert_eq!(rgb.len(), w * h * 3, "input is not W*H*3 interleaved RGB");
    let n = w * h;
    let mut out = vec![0f32; n * 3];
    for i in 0..n {
        for c in 0..3 {
            let v = rgb[i * 3 + c] as f32 / 255.0;
            out[c * n + i] = (v - mean[c]) / std[c];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// top-level entry
// ---------------------------------------------------------------------------

/// Everything the preprocessor needs, defaulted to surya-2 / Qwen3-VL.
///
/// The token budget is in merged tokens (what the decoder sees), the same unit
/// `set_limit_image_tokens` takes (clip-model.h:190): pixel budget = tokens x
/// `patch_size^2 * n_merge^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VitPreproc {
    pub patch_size: i32,
    /// `clip.vision.spatial_merge_size` — patch merges per side.
    pub n_merge: i32,
    pub min_tokens: i32,
    pub max_tokens: i32,
    pub algo: ResizeAlgo,
    pub pad: PadStyle,
    pub pad_color: [u8; 3],
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for VitPreproc {
    fn default() -> Self {
        Self::qwen3vl()
    }
}

impl VitPreproc {
    /// surya-2 / Qwen3-VL as shipped: patch 16, merge 2, BICUBIC, PAD_CEIL,
    /// mean = std = 0.5, `set_limit_image_tokens(8, 4096)` (clip.cpp:1656-1660).
    pub const fn qwen3vl() -> Self {
        VitPreproc {
            patch_size: 16,
            n_merge: 2,
            min_tokens: 8,
            max_tokens: 4096,
            algo: ResizeAlgo::Bicubic,
            pad: PadStyle::Ceil,
            pad_color: [0, 0, 0],
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }

    /// Override the merged-token budget — `--image-min-tokens` /
    /// `--image-max-tokens` upstream. ViT attention is quadratic in patch
    /// count, so this is the cheapest knob for cost.
    pub const fn with_token_budget(mut self, min_tokens: i32, max_tokens: i32) -> Self {
        self.min_tokens = min_tokens;
        self.max_tokens = max_tokens;
        self
    }

    /// `patch_size * n_merge` — every edge is a multiple of this (32 here).
    pub const fn align_size(&self) -> i32 {
        self.patch_size * self.n_merge
    }

    /// Pixels per merged token: `patch_size^2 * n_merge^2` (1024 here).
    /// clip-model.h:191.
    pub const fn patch_area(&self) -> i32 {
        self.patch_size * self.patch_size * self.n_merge * self.n_merge
    }

    /// The size policy this config implies.
    pub const fn calc_size_opt(&self) -> CalcSizeOpt {
        CalcSizeOpt {
            align_size: self.align_size(),
            min_pixels: self.min_tokens * self.patch_area(),
            max_pixels: self.max_tokens * self.patch_area(),
            longest_edge: 0,
        }
    }

    /// Target `(width, height)` for a source image, without touching pixels.
    /// Useful for sizing buffers and for the position-embedding resize.
    pub fn target_size(&self, w: usize, h: usize) -> (usize, usize) {
        let (tw, th) =
            calc_size_preserved_ratio((w as i32, h as i32), &self.calc_size_opt());
        (tw.max(0) as usize, th.max(0) as usize)
    }

    /// Full pipeline: size policy -> u8-domain bicubic resize with centred pad
    /// -> normalize -> planar CHW f32.
    ///
    /// `rgb` is interleaved RGB, `w*h*3` bytes. Returns
    /// `(out_w, out_h, planar)` with `planar.len() == out_w*out_h*3`, ready
    /// for the patch-embed gather.
    pub fn preprocess(&self, rgb: &[u8], w: usize, h: usize) -> Result<(usize, usize, Vec<f32>)> {
        if w == 0 || h == 0 {
            bail!("vit_preprocess: empty image ({w}x{h})");
        }
        if w > i32::MAX as usize || h > i32::MAX as usize {
            bail!("vit_preprocess: image too large ({w}x{h})");
        }
        if rgb.len() != w * h * 3 {
            bail!(
                "vit_preprocess: expected {} bytes of interleaved RGB for {w}x{h}, got {}",
                w * h * 3,
                rgb.len()
            );
        }

        let (tw, th) = self.target_size(w, h);
        if tw == 0 || th == 0 {
            bail!("vit_preprocess: size policy produced an empty target for {w}x{h}");
        }

        let resized =
            resize(rgb, w, h, tw, th, self.algo, self.pad, self.pad_color);
        let planar = normalize_planar(&resized, tw, th, &self.mean, &self.std);
        Ok((tw, th, planar))
    }
}

// ---------------------------------------------------------------------------
// decoding (the only use of the `image` crate)
// ---------------------------------------------------------------------------

/// Decode PNG/JPEG/… bytes to interleaved RGB8 `(width, height, pixels)`.
///
/// The `image` crate is a container/codec dependency only; its filters are not
/// Pillow's and would break parity, so the resize above is this file's port.
///
/// EXIF orientation is applied, as cv2.imread does by default — the reference
/// the detector pipeline was trained through.
pub fn decode_rgb8(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
    use image::ImageDecoder;
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    let img = img.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok((w, h, img.into_raw()))
}

/// [`decode_rgb8`] from a path, with the format sniffed from the contents.
pub fn decode_rgb8_path(path: impl AsRef<std::path::Path>) -> Result<(usize, usize, Vec<u8>)> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("vit_preprocess: reading {}: {e}", path.display()))?;
    decode_rgb8(&bytes)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Goldens in this module came from compiling `tools/mtmd/mtmd-image.cpp`'s
    /// routines standalone (llama.cpp `434ddbbc0`), not from this
    /// implementation; regenerate them by re-running the reference.
    const SURYA: CalcSizeOpt = CalcSizeOpt {
        align_size: 32,
        min_pixels: 8 * 1024,       // set_limit_image_tokens(8, ..)
        max_pixels: 4096 * 1024,    // .., 4096), patch_area = 16*16*2*2
        longest_edge: 0,
    };

    /// Same LCG the reference harness used, so both sides see identical bytes.
    fn lcg_image(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..w * h * 3)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 16) & 0xFF) as u8
            })
            .collect()
    }

    fn fnv1a(v: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in v {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    // ---- size policy ------------------------------------------------------

    /// surya-2 as shipped. Covers exactly-at-max (2048x2048 and 4096x1024 are
    /// 4 194 304 px on the nose and must not be shrunk), over-max with a
    /// ragged edge (4000x3000 rounds 3000 up to 3008 first, then floors),
    /// under-min (anything tiny grows to >= 8192 px), the exactly-8192 case
    /// (100x50 -> 128x64), and an extreme aspect ratio (5000x17) where the min
    /// clamp never fires because the aligned area already clears it.
    #[test]
    fn size_policy_surya2_budget() {
        let cases: &[((i32, i32), (i32, i32))] = &[
            ((2048, 2048), (2048, 2048)), // exactly max_pixels
            ((4096, 1024), (4096, 1024)), // exactly max_pixels, 4:1
            ((4000, 3000), (2336, 1760)), // over max, ragged height
            ((8000, 8000), (2048, 2048)), // far over max
            ((2560, 1440), (2560, 1440)), // in budget, already aligned
            ((1700, 2200), (1696, 2208)), // in budget, both edges ragged
            ((1024, 1024), (1024, 1024)), // in budget
            ((2048, 2049), (2048, 2048)), // 1 px over -> rounds back down
            ((2040, 2050), (2048, 2048)), // both edges round to the cap
            ((100, 50), (128, 64)),       // under min -> exactly 8192 px
            ((64, 64), (96, 96)),         // under min
            ((32, 32), (96, 96)),         // under min, already aligned
            ((31, 31), (96, 96)),         // under min, rounds up to align
            ((1, 1), (96, 96)),           // degenerate but valid
            ((33, 17), (128, 96)),        // ragged + under min
            ((96, 32), (160, 64)),        // ragged aspect under min
            ((5000, 17), (4992, 32)),     // extreme aspect, min never fires
        ];
        for &(inp, want) in cases {
            assert_eq!(calc_size_preserved_ratio(inp, &SURYA), want, "input {inp:?}");
        }
    }

    /// Same images at a 1024-token budget, where the floor-by-factor branch
    /// does the work. 2048x2049 lands on 992x1024, not 1024x1024: the floor is
    /// applied to the original edge over beta, so an odd source can lose a
    /// whole block. That asymmetry is the reference's.
    #[test]
    fn size_policy_smaller_budget() {
        let opt = CalcSizeOpt { max_pixels: 1024 * 1024, ..SURYA };
        let cases: &[((i32, i32), (i32, i32))] = &[
            ((2048, 2048), (1024, 1024)),
            ((4000, 3000), (1152, 864)),
            ((4096, 1024), (2048, 512)),
            ((8000, 8000), (1024, 1024)),
            ((1700, 2200), (896, 1152)),
            ((2560, 1440), (1344, 768)),
            ((1024, 1024), (1024, 1024)), // exactly at the new max
            ((2048, 2049), (992, 1024)),
            ((2040, 2050), (992, 1024)),
            ((100, 50), (128, 64)),       // min branch, unchanged
            ((5000, 17), (4992, 32)),
        ];
        for &(inp, want) in cases {
            assert_eq!(calc_size_preserved_ratio(inp, &opt), want, "input {inp:?}");
        }
    }

    /// A non-power-of-two align (28 = qwen2-VL's 14x2) with a different
    /// budget, so the factor arithmetic cannot pass by accident on 32's bit
    /// patterns.
    #[test]
    fn size_policy_align_28() {
        let opt = CalcSizeOpt {
            align_size: 28,
            min_pixels: 3136,
            max_pixels: 12_845_056,
            longest_edge: 0,
        };
        let cases: &[((i32, i32), (i32, i32))] = &[
            ((2048, 2048), (2044, 2044)),
            ((4000, 3000), (4004, 2996)),
            ((8000, 8000), (3584, 3584)),
            ((1024, 1024), (1036, 1036)),
            ((64, 64), (56, 56)),
            ((31, 31), (84, 84)),
            ((33, 17), (84, 56)),
            ((5000, 17), (5012, 28)),
            ((2560, 1440), (2548, 1428)),
        ];
        for &(inp, want) in cases {
            assert_eq!(calc_size_preserved_ratio(inp, &opt), want, "input {inp:?}");
        }
    }

    /// Non-positive edges return (0,0) rather than panicking or clamping up
    /// (mtmd-image.cpp:126).
    #[test]
    fn size_policy_rejects_degenerate_input() {
        assert_eq!(calc_size_preserved_ratio((0, 10), &SURYA), (0, 0));
        assert_eq!(calc_size_preserved_ratio((10, 0), &SURYA), (0, 0));
        assert_eq!(calc_size_preserved_ratio((-4, 10), &SURYA), (0, 0));
    }

    /// The policy's output must be align-aligned and inside the token budget —
    /// the invariant the ViT graph relies on. Swept over a spread of aspect
    /// ratios rather than the pinned table.
    #[test]
    fn size_policy_invariants_hold() {
        let p = VitPreproc::qwen3vl();
        for &w in &[1usize, 7, 33, 64, 199, 612, 1024, 1700, 2551, 4000, 9000] {
            for &h in &[1usize, 7, 33, 64, 199, 612, 1024, 1700, 2551, 4000, 9000] {
                let (tw, th) = p.target_size(w, h);
                assert_eq!(tw % 32, 0, "{w}x{h} -> {tw}x{th}");
                assert_eq!(th % 32, 0, "{w}x{h} -> {tw}x{th}");
                let tokens = (tw * th) / p.patch_area() as usize;
                assert!(
                    (8..=4096).contains(&tokens),
                    "{w}x{h} -> {tw}x{th} = {tokens} tokens, outside [8, 4096]"
                );
            }
        }
    }

    // ---- filter kernel ----------------------------------------------------

    /// Pillow's bicubic, a = -0.5. All values are exact in binary floating
    /// point, so `==` is the right comparison: f(0.5) = 0.5625 identifies
    /// a = -0.5 uniquely (a = -0.75 gives 0.58203125), and f(1.5) = -0.0625
    /// pins the second lobe's sign (a = -0.75 gives -0.09375).
    #[test]
    fn bicubic_filter_known_values() {
        let f = |x: f64| resample_filter(x, ResizeAlgo::Bicubic);
        assert_eq!(f(0.0), 1.0);
        assert_eq!(f(0.25), 0.8671875);
        assert_eq!(f(0.5), 0.5625);
        assert_eq!(f(1.0), 0.0); // both branches agree here (-0.0 == 0.0)
        assert_eq!(f(1.5), -0.0625);
        assert_eq!(f(2.0), 0.0);
        assert_eq!(f(2.5), 0.0);
        // even
        assert_eq!(f(-0.25), f(0.25));
        assert_eq!(f(-0.5), f(0.5));
        assert_eq!(f(-1.5), f(1.5));
        // the negative lobe is real: a Catmull-Rom / a=-0.75 mix-up loses it
        assert!(f(1.25) < 0.0 && f(1.75) < 0.0);

        // sanity on the other two kernels, which share the code path
        let bl = |x: f64| resample_filter(x, ResizeAlgo::Bilinear);
        assert_eq!(bl(0.0), 1.0);
        assert_eq!(bl(0.5), 0.5);
        assert_eq!(bl(1.0), 0.0);
        let lz = |x: f64| resample_filter(x, ResizeAlgo::Lanczos);
        assert_eq!(lz(0.0), 1.0);
        assert!(lz(1.0).abs() < 1e-15);
        assert_eq!(lz(3.0), 0.0);
    }

    // ---- resampler --------------------------------------------------------

    /// Target == source is a copy, not a resample. The reference short-circuits
    /// in both `resize` (mtmd-image.cpp:52) and `resize_pillow`
    /// (mtmd-image.cpp:466); without that, a filter not exactly interpolating
    /// at integer offsets would perturb every pixel.
    #[test]
    fn identity_resize_is_a_copy() {
        let src = lcg_image(3, 2, 5);
        assert_eq!(
            resize(&src, 3, 2, 3, 2, ResizeAlgo::Bicubic, PadStyle::Ceil, [0, 0, 0]),
            src
        );
        assert_eq!(resize_pillow(&src, 3, 2, 3, 2, ResizeAlgo::Bicubic), src);

        let big = lcg_image(40, 24, 11);
        assert_eq!(resize_pillow(&big, 40, 24, 40, 24, ResizeAlgo::Bicubic), big);
    }

    /// Downsample, both axes. Hand-built pixels keep a byte-level diff against
    /// the reference readable. The `0 6 14` first pixel identifies the u8 path:
    /// the fixed-point accumulator plus clip8 rounds it differently from an f32
    /// resize.
    #[test]
    fn downsample_4x3_to_3x2_matches_reference() {
        #[rustfmt::skip]
        let src: Vec<u8> = vec![
            0,0,0,      10,20,30,   200,100,50,  255,255,255,
            5,5,5,      60,70,80,   90,10,20,    128,128,128,
            250,0,0,    0,250,0,    0,0,250,     40,40,40,
        ];
        let got = resize_pillow(&src, 4, 3, 3, 2, ResizeAlgo::Bicubic);
        #[rustfmt::skip]
        let want: Vec<u8> = vec![
            0,6,14,     97,45,36,   209,180,167,
            129,59,10,  26,98,104,  56,39,99,
        ];
        assert_eq!(got, want);
    }

    /// Upsample, both axes. Exercises the `filterscale = 1.0` branch (filter
    /// kept sharp) and the negative lobe: 2x2 -> 5x5 of saturated colours
    /// overshoots, and `clip8` clamps it — behaviour an f32 resampler would not
    /// reproduce.
    #[test]
    fn upsample_2x2_to_5x5_matches_reference() {
        let src: Vec<u8> = vec![0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255];
        let got = resize_pillow(&src, 2, 2, 5, 5, ResizeAlgo::Bicubic);
        #[rustfmt::skip]
        let want: Vec<u8> = vec![
            0,0,0,     19,0,0,      141,0,0,     255,0,0,     255,0,0,
            0,17,0,    16,16,1,     120,8,8,     222,1,16,    238,0,17,
            0,128,0,   9,119,9,     64,64,64,    119,9,119,   128,0,128,
            0,238,0,   1,222,16,    8,120,120,   16,16,222,   17,0,238,
            0,255,0,   0,255,19,    0,141,141,   0,19,255,    0,0,255,
        ];
        assert_eq!(got, want);
    }

    /// Single-axis resizes take the one-pass branches, separate code in the
    /// reference (mtmd-image.cpp:472-480) and easy to get wrong by running the
    /// two-pass path with a no-op dimension.
    #[test]
    fn single_axis_passes_match_reference() {
        let h_src = lcg_image(5, 2, 7);
        assert_eq!(
            h_src,
            vec![
                32, 218, 204, 78, 161, 20, 105, 152, 155, 227, 9, 0, 172, 63, 227, 4, 161, 7,
                95, 38, 202, 68, 137, 199, 142, 173, 201, 84, 63, 108
            ],
            "the test LCG must match the one the reference harness used"
        );
        assert_eq!(
            resize_pillow(&h_src, 5, 2, 3, 2, ResizeAlgo::Bicubic),
            vec![47, 199, 128, 129, 120, 79, 199, 36, 132, 40, 106, 88, 95, 123, 211, 110, 113, 147]
        );

        let v_src = lcg_image(2, 5, 7);
        assert_eq!(
            resize_pillow(&v_src, 2, 5, 2, 3, ResizeAlgo::Bicubic),
            vec![63, 194, 182, 145, 94, 6, 143, 69, 205, 69, 121, 48, 122, 113, 203, 72, 98, 152]
        );
    }

    /// PAD_CEIL: 6x4 into an 8x8 canvas. scale = min(8/6, 8/4) = 4/3, so the
    /// content becomes 8x6 (width exact, height short by 2) and sits at
    /// offset_y = (8-6)/2 = 1 — one black row top and bottom, no horizontal
    /// padding. A wrong offset or axis still produces a plausible-looking image.
    #[test]
    fn pad_ceil_centers_content_matches_reference() {
        let src = lcg_image(6, 4, 99);
        let got = resize(&src, 6, 4, 8, 8, ResizeAlgo::Bicubic, PadStyle::Ceil, [0, 0, 0]);
        #[rustfmt::skip]
        let want: Vec<u8> = vec![
            0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,
            60,17,6, 40,27,10, 25,122,107, 63,214,240, 227,48,192, 223,79,147, 162,151,70, 116,186,0,
            124,79,36, 40,123,78, 19,146,120, 99,133,136, 178,141,139, 125,82,95, 124,108,46, 187,198,19,
            153,172,65, 68,206,150, 60,149,129, 139,61,19, 124,228,87, 39,98,59, 84,84,52, 220,216,82,
            20,243,54, 159,128,135, 211,69,114, 134,118,18, 127,175,114, 115,141,131, 91,162,159, 69,234,201,
            18,185,94, 94,60,87, 125,60,95, 97,200,108, 142,212,70, 109,120,101, 103,126,133, 133,222,141,
            54,112,133, 11,22,48, 5,75,82, 66,255,189, 155,255,23, 82,89,54, 111,72,79, 234,206,53,
            0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,
        ];
        assert_eq!(got, want);
        // the pad is on the height axis only
        assert!(want[0..24].iter().all(|&b| b == 0));
        assert!(want[7 * 8 * 3..].iter().all(|&b| b == 0));
    }

    // ---- end to end -------------------------------------------------------

    /// The whole pipeline on a 64x40 image with surya-2's parameters.
    ///
    /// The policy sends 64x40 to 128x96 (under min_pixels: beta =
    /// sqrt(8192/2560) = 1.789, 40*beta ceils to 96, 64*beta ceils to 128).
    /// PAD_CEIL then scales by min(128/64, 96/40) = 2.0, giving 128x80 content
    /// centred at offset_y = 8, so rows 0..7 and 88..95 are pad; with
    /// mean = std = 0.5 a black pad pixel is exactly -1.0.
    ///
    /// Both hashes and the probes are the reference's output.
    #[test]
    fn end_to_end_matches_reference() {
        let p = VitPreproc::qwen3vl();
        let src = lcg_image(64, 40, 2024);

        assert_eq!(p.target_size(64, 40), (128, 96));

        // the u8 stage alone, so a failure localizes to resize vs normalize
        let resized = resize(&src, 64, 40, 128, 96, p.algo, p.pad, p.pad_color);
        assert_eq!(resized.len(), 128 * 96 * 3);
        assert_eq!(fnv1a(&resized), 15_332_016_313_800_890_671);

        let (w, h, planar) = p.preprocess(&src, 64, 40).unwrap();
        assert_eq!((w, h), (128, 96));
        assert_eq!(planar.len(), 128 * 96 * 3);

        let bytes: Vec<u8> = planar.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(fnv1a(&bytes), 11_303_846_539_709_460_700);

        // planar CHW: idx = c*(w*h) + y*w + x
        let n = w * h;
        let at = |c: usize, y: usize, x: usize| planar[c * n + y * w + x];
        assert_eq!(at(0, 8, 0), -0.584_313_75);
        assert_eq!(at(1, 8, 0), 0.905_882_36);
        assert_eq!(at(2, 8, 0), -0.356_862_724);
        assert_eq!(at(0, 50, 64), 0.113_725_543);
        assert_eq!(at(1, 50, 64), 0.129_411_817);
        assert_eq!(at(2, 50, 64), 0.019_607_901_6);
        assert_eq!(at(0, 87, 127), 0.600_000_024);

        // the pad rows, top and bottom, on every channel
        for c in 0..3 {
            for y in (0..8).chain(88..96) {
                for x in 0..w {
                    assert_eq!(at(c, y, x), -1.0, "pad pixel c={c} y={y} x={x}");
                }
            }
        }
        // and the content rows are not all pad
        assert!((8..88).any(|y| (0..w).any(|x| at(0, y, x) != -1.0)));
    }

    /// The planar transpose and the normalization, isolated from the resize.
    #[test]
    fn normalize_planar_is_chw_and_scaled() {
        // 2x1 image: one white pixel, one mid-grey
        let rgb = vec![255u8, 0, 128, 0, 255, 64];
        let out = normalize_planar(&rgb, 2, 1, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]);
        assert_eq!(out.len(), 6);
        // R plane, then G, then B
        assert_eq!(out[0], 1.0); // 255 -> +1
        assert_eq!(out[1], -1.0); // 0 -> -1
        assert_eq!(out[2], -1.0);
        assert_eq!(out[3], 1.0);
        assert_eq!(out[4], (128.0 / 255.0 - 0.5) / 0.5);
        assert_eq!(out[5], (64.0 / 255.0 - 0.5) / 0.5);

        // non-trivial mean/std are applied per channel, in from_u8 order
        let out = normalize_planar(&rgb, 2, 1, &[0.1, 0.2, 0.3], &[0.4, 0.5, 0.6]);
        assert_eq!(out[0], (1.0 - 0.1) / 0.4);
        assert_eq!(out[2], (0.0 - 0.2) / 0.5);
        assert_eq!(out[4], (128.0f32 / 255.0 - 0.3) / 0.6);
    }

    /// A budget change must move the target, and the merged token count must
    /// respect the new cap.
    #[test]
    fn token_budget_is_respected() {
        let full = VitPreproc::qwen3vl();
        let half = VitPreproc::qwen3vl().with_token_budget(8, 1024);
        assert_eq!(full.patch_area(), 1024);
        assert_eq!(full.align_size(), 32);

        let (fw, fh) = full.target_size(2480, 3508); // A4 @ 300 dpi
        let (hw, hh) = half.target_size(2480, 3508);
        assert!(fw * fh <= 4096 * 1024 && fw * fh > 1024 * 1024);
        assert!(hw * hh <= 1024 * 1024);
        assert_eq!((fw % 32, fh % 32, hw % 32, hh % 32), (0, 0, 0, 0));
    }

    #[test]
    fn preprocess_rejects_bad_input() {
        let p = VitPreproc::qwen3vl();
        assert!(p.preprocess(&[], 0, 10).is_err());
        assert!(p.preprocess(&[0; 12], 2, 3).is_err()); // needs 18 bytes
        assert!(p.preprocess(&[0; 18], 2, 3).is_ok());
    }

    /// The `image` crate is only a decoder here: a 2x2 PNG round-trips to the
    /// interleaved RGB8 the rest of this file expects.
    #[test]
    fn decode_rgb8_reads_png() {
        let pixels: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
        let mut png = Vec::new();
        {
            let img: image::RgbImage =
                image::ImageBuffer::from_raw(2, 2, pixels.clone()).unwrap();
            image::DynamicImage::ImageRgb8(img)
                .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
                .unwrap();
        }
        let (w, h, rgb) = decode_rgb8(&png).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(rgb, pixels);
        assert!(decode_rgb8(b"not an image").is_err());
    }
}
