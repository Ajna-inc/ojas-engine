//! Detector preprocessing: cv2-compatible bilinear resize and the Ultralytics
//! letterbox. Not `vit_preprocess::resize`, which is Pillow antialiased
//! resampling — correct for the ViT encoder it serves and silently wrong for YOLO.
//!
//! The resize replicates cv2 `INTER_LINEAR` on 8-bit data: half-pixel centres,
//! no antialias, 11-bit fixed-point coefficients and the SIMD path's
//! `(b·(row>>4))>>16` accumulation, so bytes match what the model was trained
//! through.

/// How a frame was placed into the model input square.
#[derive(Debug, Clone, Copy)]
pub struct Letterbox {
    /// frame → model scale factor (min ratio)
    pub scale: f32,
    /// left pad in model pixels
    pub pad_x: usize,
    /// top pad in model pixels
    pub pad_y: usize,
    pub frame_w: usize,
    pub frame_h: usize,
    pub target: usize,
}

impl Letterbox {
    /// Map a model-space coordinate back to frame pixels (unclipped).
    pub fn to_frame(&self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.pad_x as f32) / self.scale, (y - self.pad_y as f32) / self.scale)
    }
}

const COEF_BITS: i32 = 11;
const COEF_SCALE: f32 = (1 << COEF_BITS) as f32; // 2048

/// One axis of cv2 linear resampling: per dst index, (src index, coeff pair).
fn linear_coeffs(src: usize, dst: usize) -> Vec<(usize, i32, i32)> {
    let scale = src as f32 / dst as f32;
    (0..dst)
        .map(|d| {
            let fx = (d as f32 + 0.5) * scale - 0.5;
            let mut sx = fx.floor() as isize;
            let mut fx = fx - sx as f32;
            if sx < 0 {
                sx = 0;
                fx = 0.0;
            }
            if sx as usize >= src - 1 {
                sx = src as isize - 2;
                fx = 1.0;
            }
            let sx = sx.max(0) as usize; // src == 1 guard
            let a1 = (fx * COEF_SCALE).round() as i32;
            (sx, (1 << COEF_BITS) - a1, a1)
        })
        .collect()
}

/// cv2 `INTER_LINEAR` resize of interleaved RGB8 (also grayscale via ch = 1).
pub fn resize_bilinear_rgb8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, ch: usize) -> Vec<u8> {
    assert_eq!(src.len(), sw * sh * ch);
    if (sw, sh) == (dw, dh) {
        return src.to_vec();
    }
    if sw == 1 && sh == 1 {
        let mut out = Vec::with_capacity(dw * dh * ch);
        for _ in 0..dw * dh {
            out.extend_from_slice(&src[..ch]);
        }
        return out;
    }
    let cx = linear_coeffs(sw, dw);
    let cy = linear_coeffs(sh, dh);
    // horizontal pass into i32 rows (values ≤ 255 · 2048), then the cv2 SIMD
    // vertical: ((b0·(r0>>4))>>16 + (b1·(r1>>4))>>16 + 2) >> 2
    let mut out = vec![0u8; dw * dh * ch];
    let mut row0 = vec![0i32; dw * ch];
    let mut row1 = vec![0i32; dw * ch];
    let mut cached: (isize, isize) = (-1, -1);
    let hresize = |sy: usize, row: &mut [i32]| {
        let s = &src[sy * sw * ch..(sy + 1) * sw * ch];
        for (d, &(sx, a0, a1)) in cx.iter().enumerate() {
            let sx1 = (sx + 1).min(sw - 1); // single-column source: both taps read col 0
            for c in 0..ch {
                row[d * ch + c] = a0 * s[sx * ch + c] as i32 + a1 * s[sx1 * ch + c] as i32;
            }
        }
    };
    for (dy, &(sy, b0, b1)) in cy.iter().enumerate() {
        let (need0, need1) = (sy as isize, (sy as isize + 1).min(sh as isize - 1));
        if cached == (need1, need0) {
            std::mem::swap(&mut row0, &mut row1);
            cached = (need0, need1);
        }
        if cached.0 != need0 {
            hresize(need0 as usize, &mut row0);
        }
        if cached.1 != need1 {
            hresize(need1 as usize, &mut row1);
        }
        cached = (need0, need1);
        let dst = &mut out[dy * dw * ch..(dy + 1) * dw * ch];
        for i in 0..dw * ch {
            let v = ((b0 * (row0[i] >> 4)) >> 16) + ((b1 * (row1[i] >> 4)) >> 16);
            dst[i] = ((v + 2) >> 2).clamp(0, 255) as u8;
        }
    }
    out
}

/// NV12 (Y plane `pitch` bytes/row, then the interleaved UV plane at
/// `uv_off`) to interleaved RGB8 with OpenCV's `COLOR_YUV2RGB_NV12` fixed
/// point (BT.601 limited range, 20-bit coefficients) — checked byte-exact
/// against cv2 5.0. Decoder output (NVDEC) enters the pipeline through this.
pub fn nv12_to_rgb8(nv12: &[u8], w: usize, h: usize, pitch: usize, uv_off: usize) -> Vec<u8> {
    const CY: i32 = 1_220_542;
    const CUB: i32 = 2_116_026;
    const CUG: i32 = -409_993;
    const CVG: i32 = -852_492;
    const CVR: i32 = 1_673_527;
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let yy = nv12[y * pitch + x] as i32;
            let uvp = uv_off + (y / 2) * pitch + (x & !1);
            let (u, v) = (nv12[uvp] as i32 - 128, nv12[uvp + 1] as i32 - 128);
            let yv = (yy - 16).max(0) * CY;
            let half = 1 << 19;
            let o = (y * w + x) * 3;
            out[o] = ((yv + half + CVR * v) >> 20).clamp(0, 255) as u8;
            out[o + 1] = ((yv + half + CVG * v + CUG * u) >> 20).clamp(0, 255) as u8;
            out[o + 2] = ((yv + half + CUB * u) >> 20).clamp(0, 255) as u8;
        }
    }
    out
}

/// Ultralytics letterbox placement of a `w × h` frame in a `target` square:
/// (placement, resized w, resized h). Shared by the CPU path below and the
/// GPU preprocessing kernel so both place pixels identically.
pub fn letterbox_geom(w: usize, h: usize, target: usize) -> (Letterbox, usize, usize) {
    let r = (target as f32 / w as f32).min(target as f32 / h as f32);
    let new_w = ((w as f32 * r).round() as usize).clamp(1, target);
    let new_h = ((h as f32 * r).round() as usize).clamp(1, target);
    let dw = (target - new_w) as f32 / 2.0;
    let dh = (target - new_h) as f32 / 2.0;
    let left = (dw - 0.1).round() as usize;
    let top = (dh - 0.1).round() as usize;
    (Letterbox { scale: r, pad_x: left, pad_y: top, frame_w: w, frame_h: h, target }, new_w, new_h)
}

/// Ultralytics letterbox: scale by the min ratio, centre with pad 114, the
/// `round(d/2 ∓ 0.1)` split, then `x/255` RGB → planar CHW f32
/// written into `out` (`3 · target · target`, one batch slot).
pub fn letterbox_rgb8_into(rgb: &[u8], w: usize, h: usize, target: usize, out: &mut [f32]) -> Letterbox {
    assert_eq!(rgb.len(), w * h * 3);
    assert_eq!(out.len(), 3 * target * target);
    let (lb, new_w, new_h) = letterbox_geom(w, h, target);
    let resized = resize_bilinear_rgb8(rgb, w, h, new_w, new_h, 3);
    let (left, top) = (lb.pad_x, lb.pad_y);

    const PAD: f32 = 114.0 / 255.0;
    out.fill(PAD);
    let plane = target * target;
    for y in 0..new_h {
        let src_row = &resized[y * new_w * 3..(y + 1) * new_w * 3];
        let dst_off = (top + y) * target + left;
        for x in 0..new_w {
            out[dst_off + x] = src_row[x * 3] as f32 / 255.0;
            out[plane + dst_off + x] = src_row[x * 3 + 1] as f32 / 255.0;
            out[2 * plane + dst_off + x] = src_row[x * 3 + 2] as f32 / 255.0;
        }
    }
    lb
}

/// Pixel normalization an OCR export expects. Conversions differ: the
/// standard PaddleOCR pipeline feeds `(x/255 − 0.5)/0.5`, but some exports
/// fold mean/std into the stem conv/BN and want raw `x/255` — feeding the
/// signed form to those double-normalizes and drives the net into its noise
/// attractor. Recorded per model in models/MANIFEST.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OcrNorm {
    /// `(x/255 − 0.5)/0.5`, pad −1 — the standard PaddleOCR contract.
    #[default]
    Signed,
    /// `x/255`, pad 0 — for exports with normalization folded in-graph.
    Unit,
}

/// YOLOX letterbox: same cv2 resize, but the image sits top-left (pads go
/// bottom/right), raw 0–255 values (YOLOX removed input normalization), BGR plane
/// order, pad 114.0.
pub fn letterbox_yolox_bgr_into(rgb: &[u8], w: usize, h: usize, target: usize, out: &mut [f32]) -> Letterbox {
    assert_eq!(rgb.len(), w * h * 3);
    assert_eq!(out.len(), 3 * target * target);
    let r = (target as f32 / w as f32).min(target as f32 / h as f32);
    let new_w = ((w as f32 * r) as usize).clamp(1, target);
    let new_h = ((h as f32 * r) as usize).clamp(1, target);
    let resized = resize_bilinear_rgb8(rgb, w, h, new_w, new_h, 3);
    out.fill(114.0);
    let plane = target * target;
    for y in 0..new_h {
        let src_row = &resized[y * new_w * 3..(y + 1) * new_w * 3];
        let dst_off = y * target;
        for x in 0..new_w {
            // BGR plane order
            out[dst_off + x] = src_row[x * 3 + 2] as f32;
            out[plane + dst_off + x] = src_row[x * 3 + 1] as f32;
            out[2 * plane + dst_off + x] = src_row[x * 3] as f32;
        }
    }
    Letterbox { scale: r, pad_x: 0, pad_y: 0, frame_w: w, frame_h: h, target }
}

/// PP-OCR recognition preprocessing: resize to `height` keeping aspect (cap at
/// `width`), right-pad, normalize per `norm`, planar CHW into `out`.
///
/// `swap_rb` writes the planes in BGR order: Paddle-native exports were trained on
/// cv2 BGR frames, and feeding them RGB flips the colour channels, which changes
/// the reading on coloured plates.
/// Resized width of a `w × h` crop in the OCR input (height fixed, capped).
pub fn ocr_width(w: usize, h: usize, height: usize, width: usize) -> usize {
    let ratio = w as f32 / h as f32;
    ((height as f32 * ratio).ceil() as usize).clamp(1, width)
}

/// (scale, shift, pad) of an OCR normalization: value = x·scale + shift.
pub fn ocr_affine(norm: OcrNorm) -> (f32, f32, f32) {
    match norm {
        OcrNorm::Signed => (2.0 / 255.0, -1.0, -1.0),
        OcrNorm::Unit => (1.0 / 255.0, 0.0, 0.0),
    }
}

pub fn ocr_resize_into(rgb: &[u8], w: usize, h: usize, height: usize, width: usize, norm: OcrNorm, swap_rb: bool, out: &mut [f32]) {
    place_into(rgb, w, h, ocr_width(w, h, height, width), height, width, norm, swap_rb, out)
}

/// Stretch to exactly `width` × `height` (classifier and embedding inputs),
/// normalised like OCR input; `swap_rb` feeds BGR planes.
pub fn stretch_into(rgb: &[u8], w: usize, h: usize, height: usize, width: usize, norm: OcrNorm, swap_rb: bool, out: &mut [f32]) {
    place_into(rgb, w, h, width, height, width, norm, swap_rb, out)
}

/// Resize to `new_w` × `height`, left-aligned in `width`, normalised; the rest padded.
#[allow(clippy::too_many_arguments)]
fn place_into(rgb: &[u8], w: usize, h: usize, new_w: usize, height: usize, width: usize, norm: OcrNorm, swap_rb: bool, out: &mut [f32]) {
    assert_eq!(rgb.len(), w * h * 3);
    assert_eq!(out.len(), 3 * height * width);
    let resized = resize_bilinear_rgb8(rgb, w, h, new_w, height, 3);
    let (scale, shift, pad) = ocr_affine(norm);
    out.fill(pad);
    let plane = height * width;
    for y in 0..height {
        for x in 0..new_w {
            for c in 0..3 {
                let dst_c = if swap_rb { 2 - c } else { c };
                out[dst_c * plane + y * width + x] = resized[(y * new_w + x) * 3 + c] as f32 * scale + shift;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_resize_is_passthrough() {
        let src: Vec<u8> = (0..48).map(|i| i as u8).collect(); // 4x4 rgb
        assert_eq!(resize_bilinear_rgb8(&src, 4, 4, 4, 4, 3), src);
    }

    #[test]
    fn upscale_2x_interpolates_midpoints() {
        // 2x1 grayscale [0, 100] -> 4x1: centres at src 0.25/0.75 clamp to edges around border
        let out = resize_bilinear_rgb8(&[0, 100], 2, 1, 4, 1, 1);
        assert_eq!(out[0], 0);
        assert_eq!(out[3], 100);
        assert!(out[1] < out[2], "monotone: {out:?}");
        // 4x1 downscale to 2x1 averages pairs
        let out = resize_bilinear_rgb8(&[10, 30, 50, 70], 4, 1, 2, 1, 1);
        assert_eq!(out, vec![20, 60]);
    }

    #[test]
    fn letterbox_geometry_1080p() {
        // 1920x1080 -> 640: r = 1/3, new = 640x360, dh = 140 -> top 140
        let rgb = vec![0u8; 1920 * 1080 * 3];
        let mut out = vec![0f32; 3 * 640 * 640];
        let lb = letterbox_rgb8_into(&rgb, 1920, 1080, 640, &mut out);
        assert!((lb.scale - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!((lb.pad_x, lb.pad_y), (0, 140));
        // pad rows are 114/255, content rows are 0
        assert!((out[0] - 114.0 / 255.0).abs() < 1e-6);
        assert_eq!(out[140 * 640], 0.0);
        // round-trip a model-space point
        let (fx, fy) = lb.to_frame(320.0, 320.0);
        assert!((fx - 960.0).abs() < 1e-3 && (fy - 540.0).abs() < 1e-3);
    }

    #[test]
    fn odd_pad_split_matches_ultralytics_rounding() {
        // 100x35 -> 64: r = 0.64, new = 64x22, dh = 21 -> top = round(20.9)=21, bottom 21
        let rgb = vec![255u8; 100 * 35 * 3];
        let mut out = vec![0f32; 3 * 64 * 64];
        let lb = letterbox_rgb8_into(&rgb, 100, 35, 64, &mut out);
        assert_eq!(lb.pad_y, 21);
        // 33x10 -> 32: new = 32x10, dh=11 -> top round(10.9) = 11
        let rgb = vec![255u8; 33 * 10 * 3];
        let mut out = vec![0f32; 3 * 32 * 32];
        let lb = letterbox_rgb8_into(&rgb, 33, 10, 32, &mut out);
        assert_eq!(lb.pad_y, 11);
    }

    #[test]
    fn ocr_resize_pads_right() {
        let rgb = vec![255u8; 10 * 10 * 3]; // square crop -> new_w = 48 columns of content
        let mut out = vec![0f32; 3 * 48 * 320];
        ocr_resize_into(&rgb, 10, 10, 48, 320, OcrNorm::Signed, false, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6); // white -> (1-0.5)/0.5 = 1
        assert!((out[47] - 1.0).abs() < 1e-6); // last content column
        assert_eq!(out[48], -1.0); // beyond new_w: pad -1
        // unit norm: white -> 1.0, pad -> 0.0
        ocr_resize_into(&rgb, 10, 10, 48, 320, OcrNorm::Unit, false, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert_eq!(out[48], 0.0);
    }

    #[test]
    fn ocr_bgr_swaps_planes() {
        // one pure-red pixel: RGB planes (1,-1,-1); BGR planes (-1,-1,1)
        let rgb = [255u8, 0, 0];
        let mut out = vec![0f32; 3 * 48 * 320];
        ocr_resize_into(&rgb, 1, 1, 48, 320, OcrNorm::Signed, false, &mut out);
        let plane = 48 * 320;
        assert_eq!((out[0], out[plane], out[2 * plane]), (1.0, -1.0, -1.0));
        ocr_resize_into(&rgb, 1, 1, 48, 320, OcrNorm::Signed, true, &mut out);
        assert_eq!((out[0], out[plane], out[2 * plane]), (-1.0, -1.0, 1.0));
    }
}

/// Per-channel input normalisation: value = u8 · scale + shift, one pair per RGB channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChanNorm {
    pub scale: [f32; 3],
    pub shift: [f32; 3],
}

impl ChanNorm {
    /// The OCR / classifier norms, the same on every channel.
    pub fn of(norm: OcrNorm) -> Self {
        let (s, b, _) = ocr_affine(norm);
        ChanNorm { scale: [s; 3], shift: [b; 3] }
    }
    /// (x/255 − mean)/std per channel — ImageNet-style (OSNet, RTMPose).
    pub fn mean_std(mean: [f32; 3], std: [f32; 3]) -> Self {
        let mut n = ChanNorm { scale: [0.0; 3], shift: [0.0; 3] };
        for c in 0..3 {
            n.scale[c] = 1.0 / (255.0 * std[c]);
            n.shift[c] = -mean[c] / std[c];
        }
        n
    }
    pub fn apply(&self, c: usize, u: u8) -> f32 {
        u as f32 * self.scale[c] + self.shift[c]
    }
}

/// How a box maps onto a model input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Window {
    /// the box itself, stretched to the input (torchreid, CLIP processors)
    Stretch,
    /// the box grown to `scale`× about its centre, aspect fixed to the input, zero outside the
    /// frame (mmpose top-down: centre + scale, padding 1.25)
    Around { scale: f32 },
}

/// The pixels a window actually reads and where they land in a `width`×`height` input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowGeom {
    /// frame rectangle read (clipped to the frame)
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    /// size it is resized to and its offset in the input; the rest is fill
    pub nw: usize,
    pub nh: usize,
    pub left: usize,
    pub top: usize,
    /// window origin in frame pixels and input pixels per frame pixel — to map keypoints back:
    /// frame = origin + input / scale
    pub ox: f32,
    pub oy: f32,
    pub sx: f32,
    pub sy: f32,
}

impl WindowGeom {
    /// Input-pixel coordinates → frame pixels.
    pub fn to_frame(&self, x: f32, y: f32) -> (f32, f32) {
        (self.ox + x / self.sx, self.oy + y / self.sy)
    }
}

/// Geometry of `bx` (x0, y0, w, h in frame pixels) into a `width`×`height` input.
pub fn window_geom(bx: [f32; 4], fw: usize, fh: usize, width: usize, height: usize, window: Window) -> WindowGeom {
    let (fwf, fhf) = (fw as f32, fh as f32);
    let (wx0, wy0, ww, wh) = match window {
        Window::Stretch => (bx[0], bx[1], bx[2], bx[3]),
        Window::Around { scale } => {
            let (cx, cy) = (bx[0] + bx[2] / 2.0, bx[1] + bx[3] / 2.0);
            let (mut ww, mut wh) = (bx[2] * scale, bx[3] * scale);
            let aspect = width as f32 / height as f32;
            if ww > wh * aspect {
                wh = ww / aspect;
            } else {
                ww = wh * aspect;
            }
            (cx - ww / 2.0, cy - wh / 2.0, ww, wh)
        }
    };
    let (ww, wh) = (ww.max(1.0), wh.max(1.0));
    let (sx, sy) = (width as f32 / ww, height as f32 / wh);
    // the part of the window inside the frame, on whole pixels
    let cx0 = wx0.max(0.0).floor().min(fwf - 1.0);
    let cy0 = wy0.max(0.0).floor().min(fhf - 1.0);
    let cx1 = (wx0 + ww).min(fwf).ceil().max(cx0 + 1.0);
    let cy1 = (wy0 + wh).min(fhf).ceil().max(cy0 + 1.0);
    let nw = (((cx1 - cx0) * sx).round() as usize).clamp(1, width);
    let nh = (((cy1 - cy0) * sy).round() as usize).clamp(1, height);
    let left = (((cx0 - wx0) * sx).round().max(0.0) as usize).min(width - nw);
    let top = (((cy0 - wy0) * sy).round().max(0.0) as usize).min(height - nh);
    WindowGeom { x0: cx0 as usize, y0: cy0 as usize, w: (cx1 - cx0) as usize, h: (cy1 - cy0) as usize, nw, nh, left, top, ox: wx0, oy: wy0, sx, sy }
}

/// CPU twin of the device crop kernel for a window: the read rectangle resized to
/// (nw, nh), placed at (left, top) in the `width`×`height` input, the rest `fill_u8`
/// normalised like a pixel; per-channel norm; `swap_rb` feeds BGR planes.
#[allow(clippy::too_many_arguments)]
pub fn window_into(rgb: &[u8], w: usize, h: usize, g: &WindowGeom, width: usize, height: usize, norm: &ChanNorm, fill_u8: u8, swap_rb: bool, out: &mut [f32]) {
    assert_eq!(rgb.len(), w * h * 3);
    assert_eq!(out.len(), 3 * height * width);
    let mut sub = Vec::with_capacity(g.w * g.h * 3);
    for y in g.y0..g.y0 + g.h {
        sub.extend_from_slice(&rgb[(y * w + g.x0) * 3..(y * w + g.x0 + g.w) * 3]);
    }
    let resized = resize_bilinear_rgb8(&sub, g.w, g.h, g.nw, g.nh, 3);
    let plane = height * width;
    for c in 0..3 {
        let dst_c = if swap_rb { 2 - c } else { c };
        out[dst_c * plane..(dst_c + 1) * plane].fill(norm.apply(c, fill_u8));
    }
    for y in 0..g.nh {
        for x in 0..g.nw {
            for c in 0..3 {
                let dst_c = if swap_rb { 2 - c } else { c };
                out[dst_c * plane + (y + g.top) * width + x + g.left] = norm.apply(c, resized[(y * g.nw + x) * 3 + c]);
            }
        }
    }
}
