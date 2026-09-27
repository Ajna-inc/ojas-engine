//! Detection data: COCO-format annotations, image decoding, the RT-DETR train
//! augmentations and a threaded batch loader.
//!
//! Augmentations follow the RT-DETRv2 / IISc UVH-26 recipe (torchvision v2):
//! RandomPhotometricDistort (p 0.5), RandomZoomOut (fill 0, side 1..4, p 0.5),
//! RandomIoUCrop (p 0.8), SanitizeBoundingBoxes (min 1 px), RandomHorizontalFlip,
//! Resize 640×640. The geometric ops compose into one source window, so the
//! zoom-out canvas is never materialised: every output pixel is resampled once
//! from the decoded image with PIL's antialiased bilinear filter (triangle,
//! support scaled with the downsampling factor), zeros outside the image.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};

use crate::models::detr_loss::{Rng, Target};

/// One annotated image.
#[derive(Clone, Debug)]
pub struct Sample {
    pub path: PathBuf,
    pub width: usize,
    pub height: usize,
    /// class id and box (x, y, w, h) in pixels
    pub boxes: Vec<(usize, [f32; 4])>,
    pub image_id: i64,
}

/// A COCO detection file. `root`: directory searched (recursively, one level of
/// sub-folders like UVH-26's `data/000/`) for the image files.
pub fn load_coco(json: &Path, root: &Path) -> Result<Vec<Sample>> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(json).with_context(|| format!("{}", json.display()))?)?;
    // file name → path
    let mut files: HashMap<String, PathBuf> = HashMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).with_context(|| format!("{}", d.display()))? {
            let p = e?.path();
            if p.is_dir() {
                stack.push(p);
            } else if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
                files.insert(n.to_string(), p.clone());
            }
        }
    }
    let mut by_id: HashMap<i64, usize> = HashMap::new();
    let mut out = vec![];
    for im in v["images"].as_array().ok_or_else(|| anyhow!("no images"))? {
        let name = im["file_name"].as_str().unwrap_or_default();
        // a relative path under root first (BMD-45: images_000/41.png — base names repeat across folders)
        let direct = root.join(name);
        let path = if name.contains('/') && direct.is_file() {
            direct
        } else if let Some(p) = files.get(name).or_else(|| files.get(Path::new(name).file_name().and_then(|n| n.to_str()).unwrap_or(""))) {
            p.clone()
        } else {
            continue;
        };
        by_id.insert(im["id"].as_i64().unwrap_or(-1), out.len());
        out.push(Sample {
            path,
            width: im["width"].as_u64().unwrap_or(0) as usize,
            height: im["height"].as_u64().unwrap_or(0) as usize,
            boxes: vec![],
            image_id: im["id"].as_i64().unwrap_or(-1),
        });
    }
    for a in v["annotations"].as_array().ok_or_else(|| anyhow!("no annotations"))? {
        if a["iscrowd"].as_i64().unwrap_or(0) != 0 {
            continue;
        }
        let Some(&i) = by_id.get(&a["image_id"].as_i64().unwrap_or(-1)) else { continue };
        let b = a["bbox"].as_array().ok_or_else(|| anyhow!("annotation without bbox"))?;
        let bb = [0, 1, 2, 3].map(|k| b[k].as_f64().unwrap_or(0.0) as f32);
        out[i].boxes.push((a["category_id"].as_u64().unwrap_or(0) as usize, bb));
    }
    Ok(out)
}

/// Decoded RGB image, f32 in 0..1, HWC.
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<f32>,
}

pub fn decode(path: &Path) -> Result<Image> {
    let img = image::open(path).with_context(|| format!("{}", path.display()))?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok(Image { w, h, rgb: img.into_raw().iter().map(|&v| v as f32 / 255.0).collect() })
}

/// PIL-style separable resampling weights for one axis: output index → (first
/// source index, weights), the source window [x0, x0 + span) mapped onto `out`
/// pixels; sources outside [0, n) contribute zero (zero padding).
fn axis_weights(n: usize, x0: f64, span: f64, out: usize) -> Vec<(isize, Vec<f32>)> {
    let scale = span / out as f64;
    let support = scale.max(1.0); // bilinear support 1, widened when downsampling
    (0..out)
        .map(|o| {
            let center = x0 + (o as f64 + 0.5) * scale;
            let lo = (center - support).floor() as isize;
            let hi = (center + support).ceil() as isize;
            // PIL: weights over the in-image sources, renormalised (all zero only
            // when the whole footprint is outside the image: zoom-out fill)
            let mut ws = vec![];
            let mut sum = 0.0f64;
            for s in lo..hi {
                let inside = s >= 0 && s < n as isize;
                let w = if inside { (1.0 - ((s as f64 + 0.5 - center) / support).abs()).max(0.0) } else { 0.0 };
                ws.push(w);
                sum += w;
            }
            let ws: Vec<f32> = ws.iter().map(|&w| if sum > 0.0 { (w / sum) as f32 } else { 0.0 }).collect();
            (lo, ws)
        })
        .collect()
}

/// Resample the source window (x0, y0, w, h) — may extend outside the image —
/// to ow×oh, CHW, optional horizontal flip.
fn resample(img: &Image, win: [f64; 4], ow: usize, oh: usize, flip: bool) -> Vec<f32> {
    let wx = axis_weights(img.w, win[0], win[2], ow);
    let wy = axis_weights(img.h, win[1], win[3], oh);
    // horizontal pass over the rows the vertical pass will need
    let (ylo, yhi) = (wy.iter().map(|(l, _)| *l).min().unwrap_or(0).max(0), wy.iter().map(|(l, w)| *l + w.len() as isize).max().unwrap_or(0).min(img.h as isize));
    let rows = (yhi - ylo).max(0) as usize;
    let mut tmp = vec![0.0f32; rows * ow * 3];
    for r in 0..rows {
        let src = &img.rgb[(ylo as usize + r) * img.w * 3..(ylo as usize + r + 1) * img.w * 3];
        for (o, (lo, ws)) in wx.iter().enumerate() {
            let mut acc = [0.0f32; 3];
            for (i, &w) in ws.iter().enumerate() {
                if w != 0.0 {
                    let s = (*lo + i as isize) as usize * 3;
                    acc[0] += w * src[s];
                    acc[1] += w * src[s + 1];
                    acc[2] += w * src[s + 2];
                }
            }
            let oo = if flip { ow - 1 - o } else { o };
            tmp[(r * ow + oo) * 3..(r * ow + oo) * 3 + 3].copy_from_slice(&acc);
        }
    }
    let mut chw = vec![0.0f32; 3 * ow * oh];
    for (oy, (lo, ws)) in wy.iter().enumerate() {
        for ox in 0..ow {
            let mut acc = [0.0f32; 3];
            for (i, &w) in ws.iter().enumerate() {
                let y = *lo + i as isize;
                if w != 0.0 && y >= ylo && y < yhi {
                    let t = ((y - ylo) as usize * ow + ox) * 3;
                    acc[0] += w * tmp[t];
                    acc[1] += w * tmp[t + 1];
                    acc[2] += w * tmp[t + 2];
                }
            }
            for c in 0..3 {
                chw[(c * oh + oy) * ow + ox] = acc[c];
            }
        }
    }
    chw
}

fn box_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    // xyxy
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let i = iw * ih;
    i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)
}

/// torchvision RandomIoUCrop: a crop window (x, y, w, h) in canvas coords, or None.
fn iou_crop(rng: &mut Rng, cw: f32, ch: f32, boxes: &[[f32; 4]]) -> Option<[f32; 4]> {
    let opts = [0.0f32, 0.1, 0.3, 0.5, 0.7, 0.9, 2.0]; // 2.0 = no crop
    loop {
        let min_iou = opts[(rng.next_u32() as usize) % opts.len()];
        if min_iou >= 1.0 {
            return None;
        }
        for _ in 0..40 {
            let (sw, sh) = (0.3 + 0.7 * rng.uniform(), 0.3 + 0.7 * rng.uniform());
            let (nw, nh) = (cw * sw, ch * sh);
            let ar = nw / nh;
            if !(0.5..=2.0).contains(&ar) {
                continue;
            }
            let (l, t) = ((cw - nw) * rng.uniform(), (ch - nh) * rng.uniform());
            let (r, b) = (l + nw, t + nh);
            if l == r || t == b {
                continue;
            }
            let win = [l, t, r, b];
            let centers_in = boxes.iter().any(|bx| {
                let (cx, cy) = ((bx[0] + bx[2]) * 0.5, (bx[1] + bx[3]) * 0.5);
                l < cx && cx < r && t < cy && cy < b
            });
            if !centers_in {
                continue;
            }
            if boxes.iter().any(|&bx| box_iou(bx, win) >= min_iou) {
                return Some([l, t, nw, nh]);
            }
        }
    }
}

/// RandomPhotometricDistort on a CHW image in 0..1 (each op p 0.5): brightness
/// ×[0.875, 1.125], contrast ×[0.5, 1.5] (before or after saturation),
/// saturation ×[0.5, 1.5], hue ±0.05; channel permutation p 0.5.
fn photometric(rng: &mut Rng, img: &mut [f32], n: usize) {
    let gray_mean = |img: &[f32]| -> f32 { (0..n).map(|i| 0.299 * img[i] + 0.587 * img[n + i] + 0.114 * img[2 * n + i]).sum::<f32>() / n as f32 };
    let blend = |img: &mut [f32], other: &dyn Fn(usize, usize, &[f32]) -> f32, f: f32| {
        let copy = img.to_vec();
        for c in 0..3 {
            for i in 0..n {
                img[c * n + i] = (f * copy[c * n + i] + (1.0 - f) * other(c, i, &copy)).clamp(0.0, 1.0);
            }
        }
    };
    if rng.uniform() < 0.5 {
        let f = 0.875 + 0.25 * rng.uniform();
        img.iter_mut().for_each(|v| *v = (*v * f).clamp(0.0, 1.0));
    }
    let contrast_first = rng.uniform() < 0.5;
    let contrast = |rng: &mut Rng, img: &mut [f32]| {
        if rng.uniform() < 0.5 {
            let f = 0.5 + rng.uniform();
            let m = gray_mean(img);
            blend(img, &|_, _, _| m, f);
        }
    };
    if contrast_first {
        contrast(rng, img);
    }
    if rng.uniform() < 0.5 {
        let f = 0.5 + rng.uniform();
        blend(img, &|_, i, src: &[f32]| 0.299 * src[i] + 0.587 * src[n + i] + 0.114 * src[2 * n + i], f);
    }
    if rng.uniform() < 0.5 {
        let dh = -0.05 + 0.1 * rng.uniform();
        for i in 0..n {
            let (r, g, b) = (img[i], img[n + i], img[2 * n + i]);
            let mx = r.max(g).max(b);
            let mn = r.min(g).min(b);
            let d = mx - mn;
            let mut h = if d == 0.0 { 0.0 } else if mx == r { ((g - b) / d).rem_euclid(6.0) } else if mx == g { (b - r) / d + 2.0 } else { (r - g) / d + 4.0 } / 6.0;
            let s = if mx == 0.0 { 0.0 } else { d / mx };
            h = (h + dh).rem_euclid(1.0);
            let (hh, v) = (h * 6.0, mx);
            let c = v * s;
            let x = c * (1.0 - (hh.rem_euclid(2.0) - 1.0).abs());
            let (r1, g1, b1) = match hh as usize {
                0 => (c, x, 0.0),
                1 => (x, c, 0.0),
                2 => (0.0, c, x),
                3 => (0.0, x, c),
                4 => (x, 0.0, c),
                _ => (c, 0.0, x),
            };
            let m = v - c;
            img[i] = r1 + m;
            img[n + i] = g1 + m;
            img[2 * n + i] = b1 + m;
        }
    }
    if !contrast_first {
        contrast(rng, img);
    }
    if rng.uniform() < 0.5 {
        let mut p = [0usize, 1, 2];
        for i in (1..3).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            p.swap(i, j);
        }
        let copy = img.to_vec();
        for c in 0..3 {
            img[c * n..(c + 1) * n].copy_from_slice(&copy[p[c] * n..(p[c] + 1) * n]);
        }
    }
}

/// Which of the per-sample augmentations run.
#[derive(Clone, Copy, Debug)]
pub struct Ops {
    /// RandomPhotometricDistort (p 0.5)
    pub photometric: bool,
    /// RandomZoomOut (p 0.5) then RandomIoUCrop (p 0.8)
    pub zoom_crop: bool,
    /// RandomHorizontalFlip (p 0.5)
    pub flip: bool,
}

impl Ops {
    pub const FULL: Ops = Ops { photometric: true, zoom_crop: true, flip: true };
    pub const NONE: Ops = Ops { photometric: false, zoom_crop: false, flip: false };
    pub const FLIP: Ops = Ops { photometric: false, zoom_crop: false, flip: true };
}

/// One training (augment) or evaluation (resize only) sample → (CHW image
/// out×out, target with cxcywh boxes normalised to the output).
pub fn prepare(s: &Sample, out: usize, augment: bool, rng: &mut Rng) -> Result<(Vec<f32>, Target)> {
    prepare_ops(s, out, if augment { Ops::FULL } else { Ops::NONE }, rng)
}

pub fn prepare_ops(s: &Sample, out: usize, ops: Ops, rng: &mut Rng) -> Result<(Vec<f32>, Target)> {
    let img = decode(&s.path)?;
    // boxes in xyxy source pixels
    let boxes: Vec<(usize, [f32; 4])> = s.boxes.iter().map(|&(c, b)| (c, [b[0], b[1], b[0] + b[2], b[1] + b[3]])).collect();
    Ok(place(&img, boxes, out, ops, rng))
}

/// The geometric and photometric ops on a decoded image with xyxy pixel boxes → out×out.
fn place(img: &Image, mut boxes: Vec<(usize, [f32; 4])>, out: usize, ops: Ops, rng: &mut Rng) -> (Vec<f32>, Target) {
    let (w, h) = (img.w as f32, img.h as f32);
    // geometry: canvas = zoom-out (image at ox, oy), then a crop window in the canvas
    let (mut ox, mut oy, mut cw, mut ch) = (0.0f32, 0.0f32, w, h);
    let mut win = [0.0f32, 0.0, w, h];
    let mut flip = false;
    if ops.zoom_crop {
        if rng.uniform() < 0.5 {
            let r = 1.0 + 3.0 * rng.uniform();
            (cw, ch) = (w * r, h * r);
            ox = ((cw - w) * rng.uniform()).floor();
            oy = ((ch - h) * rng.uniform()).floor();
            win = [0.0, 0.0, cw, ch];
        }
        let canvas_boxes: Vec<[f32; 4]> = boxes.iter().map(|(_, b)| [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]).collect();
        if rng.uniform() < 0.8 && !canvas_boxes.is_empty() {
            if let Some(c) = iou_crop(rng, cw, ch, &canvas_boxes) {
                win = c;
                // IoU crop keeps boxes whose centre is inside
                boxes.retain(|(_, b)| {
                    let (cx, cy) = ((b[0] + b[2]) * 0.5 + ox, (b[1] + b[3]) * 0.5 + oy);
                    c[0] < cx && cx < c[0] + c[2] && c[1] < cy && cy < c[1] + c[3]
                });
            }
        }
    }
    if ops.flip {
        flip = rng.uniform() < 0.5;
    }
    // map boxes into the output: canvas → window → out×out (clipped), then sanitise
    let (sx, sy) = (out as f32 / win[2], out as f32 / win[3]);
    let mut tgt = Target::default();
    for (c, b) in boxes {
        let mut x1 = ((b[0] + ox - win[0]) * sx).clamp(0.0, out as f32);
        let mut x2 = ((b[2] + ox - win[0]) * sx).clamp(0.0, out as f32);
        let y1 = ((b[1] + oy - win[1]) * sy).clamp(0.0, out as f32);
        let y2 = ((b[3] + oy - win[1]) * sy).clamp(0.0, out as f32);
        if flip {
            (x1, x2) = (out as f32 - x2, out as f32 - x1);
        }
        if x2 - x1 < 1.0 || y2 - y1 < 1.0 {
            continue;
        }
        let o = out as f32;
        tgt.labels.push(c);
        tgt.boxes.push([(x1 + x2) * 0.5 / o, (y1 + y2) * 0.5 / o, (x2 - x1) / o, (y2 - y1) / o]);
    }
    // pixels: the window in source coordinates is (win.x − ox, win.y − oy)
    let mut chw = resample(img, [(win[0] - ox) as f64, (win[1] - oy) as f64, win[2] as f64, win[3] as f64], out, out, flip);
    if ops.photometric && rng.uniform() < 0.5 {
        photometric(rng, &mut chw, out * out);
    }
    (chw, tgt)
}

/// DEIM's Mosaic (`use_cache: False`): the sample and three others drawn uniformly, each
/// resized to a short side of `size`, pasted 2×2 at the corners of a zero canvas twice the
/// largest, then RandomAffine (±10°, ±10 % translation, scale 0.5..1.5; nearest, fill 0) on the
/// canvas. Returns the canvas and its xyxy pixel boxes (clamped, degenerate ones dropped).
fn mosaic(samples: &[Sample], first: usize, size: usize, rng: &mut Rng) -> Result<(Image, Vec<(usize, [f32; 4])>)> {
    let mut picks = vec![first];
    for _ in 0..3 {
        picks.push((rng.next_u32() as usize) % samples.len());
    }
    let mut parts = vec![];
    for &i in &picks {
        let img = decode(&samples[i].path)?;
        // torchvision Resize(size): the short side to `size`, the long side truncated
        let (nw, nh) = if img.w <= img.h { (size, size * img.h / img.w) } else { (size * img.w / img.h, size) };
        let chw = resample(&img, [0.0, 0.0, img.w as f64, img.h as f64], nw, nh, false);
        let (sx, sy) = (nw as f32 / img.w as f32, nh as f32 / img.h as f32);
        let boxes: Vec<(usize, [f32; 4])> = samples[i].boxes.iter().map(|&(c, b)| (c, [b[0] * sx, b[1] * sy, (b[0] + b[2]) * sx, (b[1] + b[3]) * sy])).collect();
        parts.push((nw, nh, chw, boxes));
    }
    let mw = parts.iter().map(|p| p.0).max().unwrap();
    let mh = parts.iter().map(|p| p.1).max().unwrap();
    let (cw, ch) = (2 * mw, 2 * mh);
    let mut canvas = vec![0.0f32; cw * ch * 3];
    let mut boxes = vec![];
    for (k, (nw, nh, chw, bx)) in parts.into_iter().enumerate() {
        let (ox, oy) = ((k % 2) * mw, (k / 2) * mh);
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..3 {
                    canvas[((oy + y) * cw + ox + x) * 3 + c] = chw[(c * nh + y) * nw + x];
                }
            }
        }
        boxes.extend(bx.into_iter().map(|(c, b)| (c, [b[0] + ox as f32, b[1] + oy as f32, b[2] + ox as f32, b[3] + oy as f32])));
    }
    // RandomAffine.get_params: angle, integer translation, scale; about the canvas centre
    let angle = (-10.0 + 20.0 * rng.uniform()).to_radians();
    let tx = ((-0.1 + 0.2 * rng.uniform()) * cw as f32).round();
    let ty = ((-0.1 + 0.2 * rng.uniform()) * ch as f32).round();
    let sc = 0.5 + rng.uniform();
    let (cx, cy) = (cw as f32 * 0.5, ch as f32 * 0.5);
    let (cs, sn) = (angle.cos(), angle.sin());
    let fwd = |x: f32, y: f32| -> (f32, f32) {
        let (dx, dy) = (x - cx, y - cy);
        (cx + tx + sc * (cs * dx - sn * dy), cy + ty + sc * (sn * dx + cs * dy))
    };
    let mut warped = vec![0.0f32; cw * ch * 3];
    for y in 0..ch {
        for x in 0..cw {
            let (dx, dy) = (x as f32 + 0.5 - cx - tx, y as f32 + 0.5 - cy - ty);
            let (sx, sy) = (cx + (cs * dx + sn * dy) / sc, cy + (-sn * dx + cs * dy) / sc);
            if sx >= 0.0 && sy >= 0.0 && (sx as usize) < cw && (sy as usize) < ch {
                let (si, di) = (((sy as usize) * cw + sx as usize) * 3, (y * cw + x) * 3);
                warped[di..di + 3].copy_from_slice(&canvas[si..si + 3]);
            }
        }
    }
    let boxes = boxes
        .into_iter()
        .filter_map(|(c, b)| {
            let pts = [fwd(b[0], b[1]), fwd(b[2], b[1]), fwd(b[0], b[3]), fwd(b[2], b[3])];
            let x1 = pts.iter().map(|p| p.0).fold(f32::MAX, f32::min).clamp(0.0, cw as f32);
            let x2 = pts.iter().map(|p| p.0).fold(f32::MIN, f32::max).clamp(0.0, cw as f32);
            let y1 = pts.iter().map(|p| p.1).fold(f32::MAX, f32::min).clamp(0.0, ch as f32);
            let y2 = pts.iter().map(|p| p.1).fold(f32::MIN, f32::max).clamp(0.0, ch as f32);
            (x2 - x1 >= 1.0 && y2 - y1 >= 1.0).then_some((c, [x1, y1, x2, y2]))
        })
        .collect();
    Ok((Image { w: cw, h: ch, rgb: warped }, boxes))
}

/// DEIM's epoch-staged augmentation (`policy.epoch: [a, b, c]`, `mosaic_prob`, and the collate
/// function's mixup): before `a` and from `c` on only the flip; in [a, b) each sample is a
/// mosaic with `mosaic_prob` (photometric + flip, no zoom-out / IoU crop) or the full set; in
/// [b, c) the full set without mosaic. Mixup with `mixup_prob` per batch in `mixup_epochs`.
#[derive(Clone, Copy, Debug)]
pub struct AugPolicy {
    pub stages: [usize; 3],
    pub mosaic_prob: f32,
    pub mosaic_size: usize,
    pub mixup_prob: f32,
    pub mixup_epochs: [usize; 2],
}

impl AugPolicy {
    /// `training/deim/step3_ojas_n32.yml`: stages [4, flat, stop], mosaic and mixup at 0.5 in
    /// [4, flat).
    pub fn deim(flat: usize, stop: usize) -> Self {
        AugPolicy { stages: [4, flat, stop], mosaic_prob: 0.5, mosaic_size: 320, mixup_prob: 0.5, mixup_epochs: [4, flat] }
    }

    fn sample_ops(&self, epoch: usize, rng: &mut Rng) -> (bool, Ops) {
        let [a, b, c] = self.stages;
        if epoch < a || epoch >= c {
            (false, Ops::FLIP)
        } else if epoch < b && rng.uniform() <= self.mosaic_prob {
            (true, Ops { photometric: true, zoom_crop: false, flip: true })
        } else {
            (false, Ops::FULL)
        }
    }
}

/// One sample under the policy at `epoch`.
pub fn prepare_policy(samples: &[Sample], i: usize, out: usize, epoch: usize, p: &AugPolicy, rng: &mut Rng) -> Result<(Vec<f32>, Target)> {
    let (mos, ops) = p.sample_ops(epoch, rng);
    if mos {
        let (img, boxes) = mosaic(samples, i, p.mosaic_size, rng)?;
        Ok(place(&img, boxes, out, ops, rng))
    } else {
        prepare_ops(&samples[i], out, ops, rng)
    }
}

/// The collate function's mixup: each image blended with the previous one in the batch
/// (β ∈ [0.45, 0.55], rounded to 6 places) and given both images' boxes.
pub fn mixup(images: &mut [f32], targets: &mut [Target], epoch: usize, p: &AugPolicy, rng: &mut Rng) {
    let b = targets.len();
    if b < 2 || !(p.mixup_epochs[0] <= epoch && epoch < p.mixup_epochs[1]) || rng.uniform() >= p.mixup_prob {
        return;
    }
    let beta = ((0.45 + 0.1 * rng.uniform()) as f64 * 1e6).round() as f32 / 1e6;
    let n = images.len() / b;
    let (orig, orig_t) = (images.to_vec(), targets.to_vec());
    for i in 0..b {
        let prev = (i + b - 1) % b;
        for k in 0..n {
            images[i * n + k] = beta * orig[i * n + k] + (1.0 - beta) * orig[prev * n + k];
        }
        targets[i].labels.extend(&orig_t[prev].labels);
        targets[i].boxes.extend(&orig_t[prev].boxes);
    }
}

/// A batch: images [B, 3, out, out] and their targets (and source samples).
pub struct Batch {
    pub images: Vec<f32>,
    pub targets: Vec<Target>,
    pub samples: Vec<usize>,
}

/// Threaded loader over one pass of `order` (sample indices), `batch` at a time.
pub fn loader(samples: Arc<Vec<Sample>>, order: Vec<usize>, batch: usize, out: usize, augment: bool, workers: usize, seed: u64) -> Receiver<Result<Batch>> {
    loader_with(samples, order, batch, out, augment, None, workers, seed)
}

/// The loader for one training epoch under an augmentation policy (mosaic, mixup, stages).
pub fn loader_epoch(samples: Arc<Vec<Sample>>, order: Vec<usize>, batch: usize, out: usize, epoch: usize, policy: AugPolicy, workers: usize, seed: u64) -> Receiver<Result<Batch>> {
    loader_with(samples, order, batch, out, true, Some((epoch, policy)), workers, seed)
}

#[allow(clippy::too_many_arguments)]
fn loader_with(samples: Arc<Vec<Sample>>, order: Vec<usize>, batch: usize, out: usize, augment: bool, policy: Option<(usize, AugPolicy)>, workers: usize, seed: u64) -> Receiver<Result<Batch>> {
    let (tx, rx) = sync_channel::<Result<Batch>>(4);
    let batches: Vec<Vec<usize>> = order.chunks(batch).filter(|c| c.len() == batch || !augment).map(|c| c.to_vec()).collect();
    let next = Arc::new(Mutex::new(0usize));
    // batches are produced out of order by several workers and re-sequenced here
    let (btx, brx) = sync_channel::<(usize, Result<Batch>)>(workers * 2);
    let batches = Arc::new(batches);
    for w in 0..workers {
        let (samples, batches, next, btx) = (samples.clone(), batches.clone(), next.clone(), btx.clone());
        std::thread::spawn(move || loop {
            let i = {
                let mut n = next.lock().unwrap();
                let i = *n;
                *n += 1;
                i
            };
            if i >= batches.len() {
                break;
            }
            let mut rng = Rng::new(seed ^ ((i as u64) << 20) ^ w as u64);
            let mut images = Vec::with_capacity(batches[i].len() * 3 * out * out);
            let mut targets = vec![];
            let r = (|| -> Result<Batch> {
                for &si in &batches[i] {
                    let (im, t) = match &policy {
                        Some((epoch, p)) => prepare_policy(&samples, si, out, *epoch, p, &mut rng)?,
                        None => prepare(&samples[si], out, augment, &mut rng)?,
                    };
                    images.extend(im);
                    targets.push(t);
                }
                if let Some((epoch, p)) = &policy {
                    mixup(&mut images, &mut targets, *epoch, p, &mut rng);
                }
                Ok(Batch { images, targets, samples: batches[i].clone() })
            })();
            if btx.send((i, r)).is_err() {
                break;
            }
        });
    }
    drop(btx);
    std::thread::spawn(move || {
        let mut pending: HashMap<usize, Result<Batch>> = HashMap::new();
        let mut want = 0;
        for (i, b) in brx {
            pending.insert(i, b);
            while let Some(b) = pending.remove(&want) {
                if tx.send(b).is_err() {
                    return;
                }
                want += 1;
            }
        }
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four 64×48 images, each one colour with one box; forced mosaic, then mixup.
    #[test]
    fn mosaic_and_mixup_keep_boxes_consistent() {
        let dir = std::env::temp_dir().join(format!("ojas_mosaic_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut samples = vec![];
        for k in 0..4u8 {
            let mut img = image::RgbImage::new(64, 48);
            for (x, y, p) in img.enumerate_pixels_mut() {
                // the box region bright, the rest dark
                let inside = (16..40).contains(&x) && (12..36).contains(&y);
                *p = image::Rgb(if inside { [200, 60 * k, 255 - 50 * k] } else { [10, 10, 10] });
            }
            let path = dir.join(format!("{k}.png"));
            img.save(&path).unwrap();
            samples.push(Sample { path, width: 64, height: 48, boxes: vec![(k as usize, [16.0, 12.0, 24.0, 24.0])], image_id: k as i64 });
        }
        let p = AugPolicy { stages: [4, 40, 64], mosaic_prob: 1.0, mosaic_size: 32, mixup_prob: 1.0, mixup_epochs: [4, 40] };
        let mut rng = Rng::new(7);
        let out = 64;
        let mut images = vec![];
        let mut targets = vec![];
        let mut saw_boxes = 0;
        for i in 0..4 {
            let (im, t) = prepare_policy(&samples, i, out, 10, &p, &mut rng).unwrap();
            assert_eq!(im.len(), 3 * out * out);
            for b in &t.boxes {
                assert!(b.iter().all(|v| v.is_finite()) && b[2] > 0.0 && b[3] > 0.0);
                assert!(b[0] - b[2] / 2.0 >= -1e-4 && b[0] + b[2] / 2.0 <= 1.0 + 1e-4);
                // the box centre lands on a bright pixel of its image
                let (cx, cy) = ((b[0] * out as f32) as usize, (b[1] * out as f32) as usize);
                let px: f32 = (0..3).map(|c| im[(c * out + cy.min(out - 1)) * out + cx.min(out - 1)]).sum();
                assert!(px > 0.3, "box centre on background: {b:?} {px}");
            }
            saw_boxes += t.boxes.len();
            images.extend(im);
            targets.push(t);
        }
        assert!(saw_boxes > 4, "mosaic should carry boxes from several images: {saw_boxes}");
        // policy stages: epoch 0 and 64 are flip-only, 50 is the full set without mosaic
        assert!(matches!(p.sample_ops(0, &mut rng), (false, o) if !o.zoom_crop && o.flip));
        assert!(matches!(p.sample_ops(64, &mut rng), (false, o) if !o.photometric));
        assert!(matches!(p.sample_ops(50, &mut rng), (false, o) if o.zoom_crop));
        let before: Vec<usize> = targets.iter().map(|t| t.labels.len()).collect();
        let orig = images.clone();
        mixup(&mut images, &mut targets, 10, &p, &mut rng);
        for i in 0..4 {
            assert_eq!(targets[i].labels.len(), before[i] + before[(i + 3) % 4]);
        }
        // a blend of two images: between them everywhere
        let n = 3 * out * out;
        for k in (0..n).step_by(97) {
            let (a, b) = (orig[k], orig[n * 3 + k]);
            assert!(images[k] >= a.min(b) - 1e-5 && images[k] <= a.max(b) + 1e-5);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
