//! GPU preprocessing: crops of uploaded u8 RGB frames, cv2-exact resized and
//! placed straight into a model's padded-NHWC f16 input (`cnn_crop_resize_u8`).
//! Geometry comes from the same functions the CPU path uses (`pre.rs`), so the
//! GPU input is byte-identical to CPU preprocessing followed by an f16 upload.

use anyhow::{ensure, Result};
use ojas_core::conv::Storage;


use crate::gpu::GpuDev;

use crate::pre::{letterbox_geom, ocr_affine, ocr_width, thumb_geom, thumbs_from_f16, window_geom, ChanNorm, Letterbox, OcrNorm, Window};
pub use crate::pre::{Thumb, ThumbGeom, ThumbSpec};
use crate::yolo::Detection;

/// Pixel layout of a device frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixFmt {
    /// interleaved RGB8
    Rgb8,
    /// NV12: Y plane, then interleaved UV at `uv_off` bytes (decoder output)
    Nv12 { uv_off: usize },
}

/// A frame already on the device, anywhere: its address, row pitch (bytes)
/// and layout. Frames of one crop batch may live in different buffers
/// (e.g. one per camera decoder).
#[derive(Debug, Clone, Copy)]
pub struct DevFrame {
    pub ptr: u64,
    pub pitch: usize,
    pub w: usize,
    pub h: usize,
    pub fmt: PixFmt,
}

/// A rectangle of one frame (pixels, clipped by the caller).
#[derive(Debug, Clone, Copy)]
pub struct Roi {
    pub frame: usize,
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
}

impl Roi {
    /// All of frame `frame`.
    pub fn whole(frame: usize, f: &DevFrame) -> Roi {
        Roi { frame, x0: 0, y0: 0, w: f.w, h: f.h }
    }

    /// Detections in this region's pixels (what a detector run on the region
    /// returns once its letterbox is undone) to frame pixels.
    pub fn to_frame(&self, dets: &mut [Detection]) {
        let (ox, oy) = (self.x0 as f32, self.y0 as f32);
        for d in dets.iter_mut() {
            d.x0 += ox;
            d.x1 += ox;
            d.y0 += oy;
            d.y1 += oy;
            if let Some(kp) = &mut d.keypoints {
                for p in kp.iter_mut() {
                    p[0] += ox;
                    p[1] += oy;
                }
            }
        }
    }
}

/// A decoder-scaled surface (`DecoderOptions::output`): the frame has the surface's size,
/// not the stream's, so everything here works in surface pixels until `decoded_to_source`.
#[cfg(feature = "cuda")]
impl DevFrame {
    /// A decoded frame as a device frame (its own size, pitch and chroma offset).
    pub fn of_decoded(f: &ojas_cuda::nvdec::DecodedFrame) -> DevFrame {
        DevFrame { ptr: f.ptr, pitch: f.pitch, w: f.w, h: f.h, fmt: PixFmt::Nv12 { uv_off: f.uv_off } }
    }
}

/// The picture inside decoded frame `frame` (all of it unless letterboxed): the ROI a
/// detector should see, so letterbox bars never reach it.
#[cfg(feature = "cuda")]
pub fn decoded_content(frame: usize, g: &ojas_cuda::nvdec::OutputGeom) -> Roi {
    let [x0, y0, x1, y1] = g.rect;
    Roi { frame, x0, y0, w: x1 - x0, h: y1 - y0 }
}

/// Detections in decoded-surface pixels to the stream's own pixels.
#[cfg(feature = "cuda")]
pub fn decoded_to_source(g: &ojas_cuda::nvdec::OutputGeom, dets: &mut [Detection]) {
    for d in dets.iter_mut() {
        (d.x0, d.y0) = g.to_source(d.x0, d.y0);
        (d.x1, d.y1) = g.to_source(d.x1, d.y1);
        if let Some(kp) = &mut d.keypoints {
            for p in kp.iter_mut() {
                (p[0], p[1]) = g.to_source(p[0], p[1]);
            }
        }
    }
}

/// How crops are placed into the model input.
#[derive(Debug, Clone, Copy)]
pub enum Placement {
    /// Ultralytics letterbox into a `target` square: pad 114, x/255, RGB.
    Letterbox { target: usize },
    /// PP-OCR: height fixed, aspect kept, left-aligned in `width`.
    Ocr { height: usize, width: usize, norm: OcrNorm, bgr: bool },
    /// Classifier / embedding input: stretched to exactly `width` × `height`, OCR-style norm.
    Stretch { height: usize, width: usize, norm: OcrNorm, bgr: bool },
    /// Per-channel normalisation and a window rule (the person models: OSNet's ImageNet
    /// mean/std on the stretched box, RTMPose's 1.25× window with zero fill). The ROI is the
    /// box; the window is derived from it.
    Norm { height: usize, width: usize, norm: ChanNorm, window: Window, fill_u8: u8, bgr: bool },
}

/// Descriptors for `cnn_crop_resize_u8` (14 words per crop) and each crop's
/// letterbox (for mapping detections back; `None` for OCR placement). The
/// letterbox is that of the region alone (`frame_w`/`frame_h` are the ROI's):
/// `yolo::to_frame` with it gives region pixels, `Roi::to_frame` frame pixels.
/// Taps are addressed in frame coordinates, so an NV12 region may start on
/// any pixel (chroma is looked up as for the whole frame).
pub fn descriptors(frames: &[DevFrame], rois: &[Roi], place: Placement) -> Result<(Vec<u32>, Vec<Option<Letterbox>>)> {
    let mut d = Vec::with_capacity(rois.len() * 10);
    let mut lbs = Vec::with_capacity(rois.len());
    for r in rois {
        ensure!(r.frame < frames.len(), "roi {r:?}: {} frames", frames.len());
        let f = frames[r.frame];
        ensure!(r.w > 0 && r.h > 0 && r.x0 + r.w <= f.w && r.y0 + r.h <= f.h, "roi {r:?} outside frame {f:?}");
        let mut r = *r;
        let (nw, nh, left, top, lb) = match place {
            Placement::Letterbox { target } => {
                let (lb, nw, nh) = letterbox_geom(r.w, r.h, target);
                (nw, nh, lb.pad_x, lb.pad_y, Some(lb))
            }
            Placement::Ocr { height, width, .. } => (ocr_width(r.w, r.h, height, width), height, 0, 0, None),
            Placement::Stretch { height, width, .. } => (width, height, 0, 0, None),
            Placement::Norm { height, width, window, .. } => {
                let g = window_geom([r.x0 as f32, r.y0 as f32, r.w as f32, r.h as f32], f.w, f.h, width, height, window);
                r = Roi { frame: r.frame, x0: g.x0, y0: g.y0, w: g.w, h: g.h };
                (g.nw, g.nh, g.left, g.top, None)
            }
        };
        let (fmt, uv_off) = match f.fmt {
            PixFmt::Rgb8 => (0u32, 0u32),
            PixFmt::Nv12 { uv_off } => (1, uv_off as u32),
        };
        d.extend([f.ptr as u32, (f.ptr >> 32) as u32, f.pitch as u32, fmt, uv_off]);
        d.extend([r.x0, r.y0, r.w, r.h, nw, nh, left, top, 0].map(|v| v as u32));
        lbs.push(lb);
    }
    Ok((d, lbs))
}

/// Enqueue the crop/resize of `n_items` crops (descriptors already on the
/// device) into `dst` (a model input storage, one crop per image slot).
pub fn launch<G: GpuDev>(g: &G, enc: &G::Enc, desc: &G::Buf,
              dst: (&G::Buf, u64), st: &Storage, n_items: usize, place: Placement) -> Result<()> {
    // per channel: value = u/div (div > 0) or u·scale + shift; fill is the normalised pad value
    let (tw, th, div, scale, shift, fill, swap) = match place {
        Placement::Letterbox { target } => (target, target, 255.0f32, [0.0; 3], [0.0; 3], [114.0f32 / 255.0; 3], 0),
        Placement::Ocr { height, width, norm, bgr } | Placement::Stretch { height, width, norm, bgr } => {
            let (scale, shift, pad) = ocr_affine(norm);
            (width, height, 0.0, [scale; 3], [shift; 3], [pad; 3], bgr as u32)
        }
        Placement::Norm { height, width, norm, fill_u8, bgr, .. } => {
            (width, height, 0.0, norm.scale, norm.shift, [norm.apply(0, fill_u8), norm.apply(1, fill_u8), norm.apply(2, fill_u8)], bgr as u32)
        }
    };
    ensure!(st.h == th && st.w == tw && st.c >= 3, "placement {place:?} vs storage {st:?}");
    let consts = [tw as u32, th as u32, st.pad as u32, st.wp() as u32, st.img() as u32, st.cs as u32, div.to_bits(),
                  scale[0].to_bits(), scale[1].to_bits(), scale[2].to_bits(), shift[0].to_bits(), shift[1].to_bits(), shift[2].to_bits(),
                  fill[0].to_bits(), fill[1].to_bits(), fill[2].to_bits(), swap];
    g.dispatch(enc, "cnn_crop_resize_u8", &[(desc, 0), dst], &consts,
               [((tw * th) as u32).div_ceil(256), n_items as u32, 1], [256, 1, 1])
}

/// Copy device frame `f` to the host as packed RGB8, taking every `step`-th
/// pixel (1 = full size; 4 = a quarter-size thumbnail). Returns (w, h) of
/// `out`. Colour conversion as the crop kernel (cv2 BT.601 for NV12).
pub fn frame_to_rgb8<G: GpuDev>(g: &G, f: &DevFrame, step: usize, scratch: &mut Option<G::Buf>, out: &mut Vec<u8>) -> Result<(usize, usize)> {
    let step = step.max(1);
    let (ow, oh) = (f.w.div_ceil(step), f.h.div_ceil(step));
    let (fmt, uv_off) = match f.fmt {
        PixFmt::Rgb8 => (0u32, 0u32),
        PixFmt::Nv12 { uv_off } => (1, uv_off as u32),
    };
    let desc = [f.ptr as u32, (f.ptr >> 32) as u32, f.pitch as u32, fmt, uv_off];
    let dbuf = g.upload_bytes(&desc.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    let need = ow * oh * 3;
    if scratch.as_ref().is_none_or(|b| G::buf_len(b) < need) {
        *scratch = Some(g.alloc_bytes(need)?);
    }
    let buf = scratch.as_ref().unwrap();
    let enc = g.begin();
    g.dispatch(&enc, "cnn_frame_rgb8", &[(&dbuf, 0), (buf, 0)], &[ow as u32, oh as u32, step as u32],
               [((ow * oh) as u32).div_ceil(256), 1, 1], [256, 1, 1])?;
    g.submit(enc)?;
    out.resize(need, 0);
    g.read_bytes(buf, 0, out)?;
    Ok((ow, oh))
}

/// Crop descriptors for thumbnails of `rois` (`pre::thumb_geom` sizes), each placed in the
/// top-left corner of a shared canvas, and that canvas's (width, height).
pub fn thumb_descriptors(frames: &[DevFrame], rois: &[Roi], spec: ThumbSpec) -> Result<(Vec<u32>, Vec<ThumbGeom>, (usize, usize))> {
    ensure!(spec.width > 0 && spec.height > 0, "thumbnail {spec:?}");
    let mut d = Vec::with_capacity(rois.len() * 14);
    let mut geoms = Vec::with_capacity(rois.len());
    for r in rois {
        ensure!(r.frame < frames.len(), "roi {r:?}: {} frames", frames.len());
        let f = frames[r.frame];
        ensure!(r.w > 0 && r.h > 0 && r.x0 + r.w <= f.w && r.y0 + r.h <= f.h, "roi {r:?} outside frame {f:?}");
        let g = thumb_geom(r.x0, r.y0, r.w, r.h, spec);
        let (fmt, uv_off) = match f.fmt {
            PixFmt::Rgb8 => (0u32, 0u32),
            PixFmt::Nv12 { uv_off } => (1, uv_off as u32),
        };
        d.extend([f.ptr as u32, (f.ptr >> 32) as u32, f.pitch as u32, fmt, uv_off]);
        d.extend([r.x0, r.y0, r.w, r.h, g.w, g.h, 0, 0, 0].map(|v| v as u32));
        geoms.push(g);
    }
    let canvas = (geoms.iter().map(|g| g.w).max().unwrap_or(1), geoms.iter().map(|g| g.h).max().unwrap_or(1));
    Ok((d, geoms, canvas))
}

/// Small host copies of `rois` (whole frames or crops): resized on the device by the crop
/// kernel (cv2-linear, the bytes of `pre::thumb_rgb8` on the same region), all downloaded
/// with one read. `scratch` is kept by the caller between calls.
pub fn thumbs<G: GpuDev>(g: &G, frames: &[DevFrame], rois: &[Roi], spec: ThumbSpec, scratch: &mut Option<G::Buf>) -> Result<Vec<Thumb>> {
    if rois.is_empty() {
        return Ok(vec![]);
    }
    let (d, geoms, (tw, th)) = thumb_descriptors(frames, rois, spec)?;
    // the kernel's output as whole numbers 0..=255 in f16 (exact), unpadded RGB
    let st = Storage { c: 3, cs: 3, h: th, w: tw, pad: 0, spare: 0 };
    let place = Placement::Norm { height: th, width: tw, norm: ChanNorm { scale: [1.0; 3], shift: [0.0; 3] }, window: Window::Stretch, fill_u8: 0, bgr: false };
    let need = rois.len() * tw * th * 6;
    if scratch.as_ref().is_none_or(|b| G::buf_len(b) < need) {
        *scratch = Some(g.alloc_bytes(need)?);
    }
    let buf = scratch.as_ref().unwrap();
    let dbuf = g.upload_bytes(&d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    let enc = g.begin();
    launch(g, &enc, &dbuf, (buf, 0), &st, rois.len(), place)?;
    g.submit(enc)?;
    let mut bytes = vec![0u8; need];
    g.read_bytes(buf, 0, &mut bytes)?;
    Ok(thumbs_from_f16(&bytes, &geoms, tw, th))
}

/// One whole-frame thumbnail per frame, or none when `spec` is `None` (the default).
pub fn frame_thumbs<G: GpuDev>(g: &G, frames: &[DevFrame], spec: Option<ThumbSpec>, scratch: &mut Option<G::Buf>) -> Result<Vec<Thumb>> {
    let Some(spec) = spec else { return Ok(vec![]) };
    let whole: Vec<Roi> = frames.iter().enumerate().map(|(i, f)| Roi::whole(i, f)).collect();
    thumbs(g, frames, &whole, spec, scratch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre::{letterbox_rgb8_into, nv12_to_rgb8, thumb_rgb8};
    use crate::yolo;

    fn noise(n: usize, mut s: u32) -> Vec<u8> {
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// One tap as the kernel's `cnn_tap_rgb`: frame coordinates, NV12 chroma at (y/2, x & !1).
    fn tap(mem: &[u8], d: &[u32], x: usize, y: usize) -> [i32; 3] {
        let (pitch, fmt, uv_off) = (d[2] as usize, d[3], d[4] as usize);
        if fmt == 0 {
            let p = y * pitch + x * 3;
            return [mem[p] as i32, mem[p + 1] as i32, mem[p + 2] as i32];
        }
        let yy = mem[y * pitch + x] as i32;
        let uvp = uv_off + (y / 2) * pitch + (x & !1);
        let (u, v) = (mem[uvp] as i32 - 128, mem[uvp + 1] as i32 - 128);
        let yv = (yy - 16).max(0) * 1_220_542;
        [
            ((yv + (1 << 19) + 1_673_527 * v) >> 20).clamp(0, 255),
            ((yv + (1 << 19) - 852_492 * v - 409_993 * u) >> 20).clamp(0, 255),
            ((yv + (1 << 19) + 2_116_026 * u) >> 20).clamp(0, 255),
        ]
    }

    /// `cnn_crop_resize_u8` for the letterbox placement, transcribed: one 14-word descriptor
    /// over the frame's bytes into planar CHW f32 (the kernel writes the same values as f16
    /// into the padded input).
    fn kernel_twin(mem: &[u8], d: &[u32], target: usize) -> Vec<f32> {
        let u = twin_u8(mem, d, target, target, 114);
        let plane = target * target;
        let mut out = vec![0f32; 3 * plane];
        for p in 0..plane {
            for c in 0..3 {
                out[c * plane + p] = u[p * 3 + c] as f32 / 255.0;
            }
        }
        out
    }

    /// The kernel's u8 values before normalisation, interleaved RGB over a `tw` × `th` canvas.
    fn twin_u8(mem: &[u8], d: &[u32], tw: usize, th: usize, fill: i32) -> Vec<u8> {
        let [rx0, ry0, rw, rh, nw, nh, left, top] = [5, 6, 7, 8, 9, 10, 11, 12].map(|i| d[i] as i32);
        let axis = |o: i32, r: i32, n: i32| -> (i32, i32, i32, i32) {
            let f = (o as f32 + 0.5) * (r as f32 / n as f32) - 0.5;
            let mut s = f.floor() as i32;
            let mut f = f - s as f32;
            if s < 0 {
                (s, f) = (0, 0.0);
            }
            if s >= r - 1 {
                (s, f) = (r - 2, 1.0);
            }
            let s = s.max(0);
            let a1 = (f * 2048.0).round() as i32;
            (s, (s + 1).min(r - 1), 2048 - a1, a1)
        };
        let mut out = vec![0u8; 3 * tw * th];
        for p in 0..tw * th {
            let (ox, oy) = ((p % tw) as i32 - left, (p / tw) as i32 - top);
            let u = if ox < 0 || ox >= nw || oy < 0 || oy >= nh {
                [fill; 3]
            } else if rw == nw && rh == nh {
                tap(mem, d, (rx0 + ox) as usize, (ry0 + oy) as usize)
            } else {
                let (sx, sx1, a0, a1) = axis(ox, rw, nw);
                let (sy, sy1, b0, b1) = axis(oy, rh, nh);
                let t = |x: i32, y: i32| tap(mem, d, (rx0 + x) as usize, (ry0 + y) as usize);
                let (t00, t01, t10, t11) = (t(sx, sy), t(sx1, sy), t(sx, sy1), t(sx1, sy1));
                [0, 1, 2].map(|c| {
                    let h0 = a0 * t00[c] + a1 * t01[c];
                    let h1 = a0 * t10[c] + a1 * t11[c];
                    ((((b0 * (h0 >> 4)) >> 16) + ((b1 * (h1 >> 4)) >> 16) + 2) >> 2).clamp(0, 255)
                })
            };
            for c in 0..3 {
                out[p * 3 + c] = u[c] as u8;
            }
        }
        out
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn decoder_scaled_detections_map_back_to_the_stream() {
        use ojas_cuda::nvdec::{output_geom, DecodedFrame, Fit, OutputSize};
        // 1920x1080 letterboxed by the decoder into 640x640: picture rows 140..500
        let g = output_geom(1920, 1080, Some(OutputSize { width: 640, height: 640, fit: Fit::Letterbox }));
        let f = DevFrame::of_decoded(&DecodedFrame { ptr: 64, pitch: 640, w: g.w, h: g.h, uv_off: 640 * 640, timestamp: 0 });
        assert_eq!((f.w, f.h, f.pitch, f.fmt), (640, 640, 640, PixFmt::Nv12 { uv_off: 640 * 640 }));
        let roi = decoded_content(0, &g);
        assert_eq!((roi.x0, roi.y0, roi.w, roi.h), (0, 140, 640, 360));
        assert!(descriptors(&[f], &[roi], Placement::Letterbox { target: 640 }).is_ok());
        // a box the detector reports inside the ROI, moved to the surface, then to the stream
        let kp = [[100.0, 30.0], [200.0, 30.0], [200.0, 90.0], [100.0, 90.0]];
        let mut dets = vec![Detection { class: 0, score: 0.9, x0: 100.0, y0: 30.0, x1: 200.0, y1: 90.0, keypoints: Some(kp) }];
        roi.to_frame(&mut dets);
        decoded_to_source(&g, &mut dets);
        let d = &dets[0];
        let near = |a: f32, b: f32| (a - b).abs() < 1e-2;
        assert!(near(d.x0, 300.0) && near(d.y0, 90.0) && near(d.x1, 600.0) && near(d.y1, 270.0), "{d:?}");
        let k = d.keypoints.unwrap();
        assert!(near(k[2][0], 600.0) && near(k[2][1], 270.0), "{k:?}");
    }

    #[test]
    fn thumbnails_match_the_cpu_thumbnail() {
        // RGB8 with a wide pitch, and NV12 in the decoder layout
        let (w, h, pitch) = (64usize, 36usize, 64 * 3 + 7);
        let rgb = noise(w * h * 3, 3);
        let mut mem = vec![0u8; pitch * h];
        for y in 0..h {
            mem[y * pitch..y * pitch + w * 3].copy_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
        }
        let nv_pitch = 80usize;
        let nv = noise(nv_pitch * h * 3 / 2, 5);
        let uv_off = nv_pitch * h;
        let nv_rgb = nv12_to_rgb8(&nv, w, h, nv_pitch, uv_off);
        let frames = [
            (DevFrame { ptr: 0, pitch, w, h, fmt: PixFmt::Rgb8 }, &mem, &rgb),
            (DevFrame { ptr: 0, pitch: nv_pitch, w, h, fmt: PixFmt::Nv12 { uv_off } }, &nv, &nv_rgb),
        ];
        for (f, bytes, host) in frames {
            // whole frame, an odd-origin crop, and a crop smaller than the thumbnail (copied 1:1)
            let rois = [Roi::whole(0, &f), Roi { frame: 0, x0: 7, y0: 3, w: 41, h: 21 }, Roi { frame: 0, x0: 50, y0: 30, w: 9, h: 5 }];
            for spec in [ThumbSpec { width: 16, height: 16, keep_aspect: true }, ThumbSpec { width: 24, height: 10, keep_aspect: false }] {
                let (d, geoms, (tw, th)) = thumb_descriptors(&[f], &rois, spec).unwrap();
                assert!(tw <= spec.width && th <= spec.height);
                for (i, r) in rois.iter().enumerate() {
                    let want = thumb_rgb8(host, w, h, [r.x0, r.y0, r.w, r.h], spec);
                    assert_eq!(geoms[i], want.geom);
                    let canvas = twin_u8(bytes, &d[i * 14..(i + 1) * 14], tw, th, 0);
                    let got: Vec<u8> = (0..want.geom.h).flat_map(|y| canvas[y * tw * 3..(y * tw + want.geom.w) * 3].iter().copied()).collect();
                    assert!(got == want.rgb, "{r:?} {spec:?} {:?}", f.fmt);
                }
            }
        }
        assert!(thumb_descriptors(&[frames[0].0], &[Roi { frame: 0, x0: 60, y0: 0, w: 5, h: 5 }], ThumbSpec { width: 8, height: 8, keep_aspect: true }).is_err());
    }

    fn crop_rgb(rgb: &[u8], w: usize, r: &Roi) -> Vec<u8> {
        (r.y0..r.y0 + r.h).flat_map(|y| rgb[(y * w + r.x0) * 3..(y * w + r.x0 + r.w) * 3].iter().copied()).collect()
    }

    /// The CPU path on the same region: crop, then the Ultralytics letterbox.
    fn oracle(rgb: &[u8], w: usize, r: &Roi, target: usize) -> (Vec<f32>, Letterbox) {
        let mut out = vec![0f32; 3 * target * target];
        let lb = letterbox_rgb8_into(&crop_rgb(rgb, w, r), r.w, r.h, target, &mut out);
        (out, lb)
    }

    fn same(a: &Letterbox, b: &Letterbox) -> bool {
        (a.scale, a.pad_x, a.pad_y, a.frame_w, a.frame_h, a.target) == (b.scale, b.pad_x, b.pad_y, b.frame_w, b.frame_h, b.target)
    }

    #[test]
    fn whole_frame_roi_is_the_frame_letterbox() {
        let f = DevFrame { ptr: 0x1_2345_6780, pitch: 1920 * 3, w: 1920, h: 1080, fmt: PixFmt::Rgb8 };
        let place = Placement::Letterbox { target: 640 };
        let (d, lbs) = descriptors(&[f], &[Roi::whole(0, &f)], place).unwrap();
        let (lb, nw, nh) = letterbox_geom(1920, 1080, 640);
        assert_eq!(d, [0x2345_6780, 1, 1920 * 3, 0, 0, 0, 0, 1920, 1080, nw as u32, nh as u32, lb.pad_x as u32, lb.pad_y as u32, 0]);
        assert!(same(&lbs[0].unwrap(), &lb));
        // the literal whole-frame ROI callers wrote before `Roi::whole`
        let (d2, _) = descriptors(&[f], &[Roi { frame: 0, x0: 0, y0: 0, w: 1920, h: 1080 }], place).unwrap();
        assert_eq!(d, d2);
        // and mapping back through it moves nothing
        let det = Detection { class: 0, score: 0.9, x0: 10.0, y0: 20.0, x1: 30.0, y1: 40.0, keypoints: None };
        let mut dets = vec![det];
        Roi::whole(0, &f).to_frame(&mut dets);
        assert_eq!(dets[0], det);
    }

    #[test]
    fn roi_letterbox_matches_cpu_on_the_region() {
        // odd frame size, row pitch wider than the pixels
        let (w, h, pitch) = (61usize, 47usize, 61 * 3 + 5);
        let rgb = noise(w * h * 3, 7);
        let mut mem = vec![0u8; pitch * h];
        for y in 0..h {
            mem[y * pitch..y * pitch + w * 3].copy_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
        }
        let f = DevFrame { ptr: 0, pitch, w, h, fmt: PixFmt::Rgb8 };
        let rois = [
            Roi::whole(0, &f),
            Roi { frame: 0, x0: 13, y0: 5, w: 31, h: 17 },  // off-centre, odd origin and size
            Roi { frame: 0, x0: 0, y0: 0, w: 20, h: 47 },   // left edge, full height (portrait)
            Roi { frame: 0, x0: 40, y0: 30, w: 21, h: 17 }, // bottom-right corner
            Roi { frame: 0, x0: 3, y0: 46, w: 58, h: 1 },   // one row, last row
            Roi { frame: 0, x0: 60, y0: 2, w: 1, h: 40 },   // one column, last column
            Roi { frame: 0, x0: 9, y0: 9, w: 32, h: 32 },   // exactly the input: no resize
            Roi { frame: 0, x0: 30, y0: 20, w: 1, h: 1 },
        ];
        for target in [32usize, 33, 96] {
            let (d, lbs) = descriptors(&[f], &rois, Placement::Letterbox { target }).unwrap();
            for (i, r) in rois.iter().enumerate() {
                let (want, lb) = oracle(&rgb, w, r, target);
                assert!(same(&lbs[i].unwrap(), &lb), "{r:?} -> {target}");
                assert!(kernel_twin(&mem, &d[i * 14..(i + 1) * 14], target) == want, "{r:?} -> {target}");
            }
        }
    }

    #[test]
    fn nv12_roi_at_any_origin_matches_cpu() {
        // decoder layout: Y rows, then interleaved UV rows, both `pitch` wide
        let (w, h, pitch) = (64usize, 36usize, 80usize);
        let nv = noise(pitch * h * 3 / 2, 11);
        let uv_off = pitch * h;
        let rgb = nv12_to_rgb8(&nv, w, h, pitch, uv_off);
        let f = DevFrame { ptr: 0, pitch, w, h, fmt: PixFmt::Nv12 { uv_off } };
        let rois = [
            Roi::whole(0, &f),
            Roi { frame: 0, x0: 8, y0: 4, w: 40, h: 20 },   // even origin
            Roi { frame: 0, x0: 7, y0: 3, w: 41, h: 21 },   // odd origin, odd size
            Roi { frame: 0, x0: 1, y0: 0, w: 63, h: 36 },   // odd x, touching three edges
            Roi { frame: 0, x0: 33, y0: 17, w: 31, h: 19 }, // bottom-right corner
            Roi { frame: 0, x0: 5, y0: 35, w: 9, h: 1 },    // last row
        ];
        for target in [48usize, 31] {
            let (d, _) = descriptors(&[f], &rois, Placement::Letterbox { target }).unwrap();
            for (i, r) in rois.iter().enumerate() {
                assert_eq!(d[i * 14 + 3..i * 14 + 5], [1, uv_off as u32]);
                assert!(kernel_twin(&nv, &d[i * 14..(i + 1) * 14], target) == oracle(&rgb, w, r, target).0, "{r:?} -> {target}");
            }
        }
    }

    #[test]
    fn detections_map_back_through_letterbox_and_roi() {
        let f = DevFrame { ptr: 0, pitch: 1920, w: 1920, h: 1080, fmt: PixFmt::Nv12 { uv_off: 1920 * 1080 } };
        // 601x333 at an odd origin -> 640: scale 640/601, pads top and bottom
        let r = Roi { frame: 0, x0: 1001, y0: 437, w: 601, h: 333 };
        let (_, lbs) = descriptors(&[f], &[r], Placement::Letterbox { target: 640 }).unwrap();
        let lb = lbs[0].unwrap();
        assert_eq!((lb.frame_w, lb.frame_h), (601, 333));
        assert!(lb.pad_x == 0 && lb.pad_y > 0);
        // a box and its corners in frame pixels, as the model would report them in its input
        let frame_box = [1100.5f32, 500.25, 1400.0, 700.75];
        let to_model = |x: f32, y: f32| ((x - r.x0 as f32) * lb.scale + lb.pad_x as f32, (y - r.y0 as f32) * lb.scale + lb.pad_y as f32);
        let (mx0, my0) = to_model(frame_box[0], frame_box[1]);
        let (mx1, my1) = to_model(frame_box[2], frame_box[3]);
        let kp = [[mx0, my0], [mx1, my0], [mx1, my1], [mx0, my1]];
        // the second box runs past the region on every side
        let mut dets = vec![
            Detection { class: 2, score: 0.9, x0: mx0, y0: my0, x1: mx1, y1: my1, keypoints: Some(kp) },
            Detection { class: 2, score: 0.8, x0: -50.0, y0: -50.0, x1: 700.0, y1: 700.0, keypoints: None },
        ];
        yolo::to_frame(&mut dets, &lb);
        r.to_frame(&mut dets);
        let near = |a: f32, b: f32| (a - b).abs() < 1e-2;
        let d = &dets[0];
        assert!(near(d.x0, frame_box[0]) && near(d.y0, frame_box[1]) && near(d.x1, frame_box[2]) && near(d.y1, frame_box[3]), "{d:?}");
        let k = d.keypoints.unwrap();
        assert!(near(k[0][0], frame_box[0]) && near(k[0][1], frame_box[1]) && near(k[2][0], frame_box[2]) && near(k[2][1], frame_box[3]), "{k:?}");
        // clipped to the region, in frame pixels
        let d = &dets[1];
        assert_eq!((d.x0, d.y0, d.x1, d.y1), (1001.0, 437.0, 1602.0, 770.0));
    }

    #[test]
    fn rois_outside_the_frame_are_rejected() {
        let f = DevFrame { ptr: 0, pitch: 300, w: 100, h: 50, fmt: PixFmt::Rgb8 };
        let place = Placement::Letterbox { target: 64 };
        for r in [
            Roi { frame: 0, x0: 90, y0: 0, w: 11, h: 10 },
            Roi { frame: 0, x0: 0, y0: 41, w: 10, h: 10 },
            Roi { frame: 0, x0: 0, y0: 0, w: 0, h: 10 },
            Roi { frame: 1, x0: 0, y0: 0, w: 10, h: 10 },
        ] {
            assert!(descriptors(&[f], &[r], place).is_err(), "{r:?}");
        }
        // touching the far edges is inside
        assert!(descriptors(&[f], &[Roi { frame: 0, x0: 90, y0: 40, w: 10, h: 10 }], place).is_ok());
    }
}
