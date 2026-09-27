//! GPU preprocessing: crops of uploaded u8 RGB frames, cv2-exact resized and
//! placed straight into a model's padded-NHWC f16 input (`cnn_crop_resize_u8`).
//! Geometry comes from the same functions the CPU path uses (`pre.rs`), so the
//! GPU input is byte-identical to CPU preprocessing followed by an f16 upload.

use anyhow::{ensure, Result};
use ojas_core::conv::Storage;


use crate::gpu::GpuDev;

use crate::pre::{letterbox_geom, ocr_affine, ocr_width, window_geom, ChanNorm, Letterbox, OcrNorm, Window};

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
/// letterbox (for mapping detections back; `None` for OCR placement).
pub fn descriptors(frames: &[DevFrame], rois: &[Roi], place: Placement) -> Result<(Vec<u32>, Vec<Option<Letterbox>>)> {
    let mut d = Vec::with_capacity(rois.len() * 10);
    let mut lbs = Vec::with_capacity(rois.len());
    for r in rois {
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
