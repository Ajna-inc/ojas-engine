//! YOLOv8/11 head decode and NMS.
//!
//! The dense export emits `[N, 4+nc, A]` (e.g. `[1, 84, 8400]` at 640):
//! `cx, cy, w, h` in model pixels plus per-class scores already sigmoided.
//! Decode filters by confidence, NMS is class-aware by default with a
//! deterministic tie-break (score desc, then anchor index), and boxes are
//! mapped back through the letterbox and clipped to the frame.

use crate::pre::Letterbox;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub class: u16,
    pub score: f32,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    /// Plate corners TL/TR/BR/BL when the model has a pose head; None for plain
    /// box detectors.
    pub keypoints: Option<[[f32; 2]; 4]>,
}

impl Detection {
    pub fn area(&self) -> f32 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }
    pub fn iou(&self, o: &Detection) -> f32 {
        let ix0 = self.x0.max(o.x0);
        let iy0 = self.y0.max(o.y0);
        let ix1 = self.x1.min(o.x1);
        let iy1 = self.y1.min(o.y1);
        let inter = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
        let union = self.area() + o.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecodeCfg {
    pub conf: f32,
    pub iou: f32,
    pub max_det: usize,
    /// keep only these class ids (None = all)
    pub classes: Option<Vec<u16>>,
    pub class_agnostic_nms: bool,
}

impl Default for DecodeCfg {
    fn default() -> Self {
        DecodeCfg { conf: 0.25, iou: 0.45, max_det: 300, classes: None, class_agnostic_nms: false }
    }
}

/// Head layout: how the export arranges its outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadLayout {
    /// Ultralytics `[1, 4+nc, A]`: channel-major, class scores sigmoided.
    ChannelsFirst,
    /// YOLOX raw head `[1, A, 5+nc]`: per-anchor rows of
    /// `reg_x, reg_y, reg_w, reg_h, obj, cls…` needing grid+stride decode
    /// (`cx = (reg_x + gx)·s`, `w = e^{reg_w}·s`); score = obj · cls.
    AnchorsFirstObj,
    /// DETR family: two outputs, `logits [1,Q,C]` + `boxes [1,Q,4]`, stretched
    /// 0–1 RGB input, no NMS (`crate::detr::decode`).
    Detr,
}

/// YOLOX grid decode of `[A, 5+nc]` raw rows. Anchors are the FPN levels (strides
/// 8, 16, 32) concatenated, each level row-major with x fastest, giving
/// `(t/8)² + (t/16)² + (t/32)²` anchors for input `t`.
pub fn decode_anchors_first_obj(pred: &[f32], nc: usize, anchors: usize, input: usize, cfg: &DecodeCfg) -> Vec<Detection> {
    debug_assert_eq!(pred.len(), anchors * (5 + nc));
    debug_assert_eq!(anchors, (input / 8).pow(2) + (input / 16).pow(2) + (input / 32).pow(2));
    let row_len = 5 + nc;
    let class_ok = |c: usize| cfg.classes.as_ref().is_none_or(|cs| cs.contains(&(c as u16)));
    let mut out = Vec::new();
    let mut a = 0usize;
    for stride in [8usize, 16, 32] {
        let side = input / stride;
        for gy in 0..side {
            for gx in 0..side {
                let row = &pred[a * row_len..(a + 1) * row_len];
                a += 1;
                let obj = row[4];
                if obj < cfg.conf {
                    continue; // score = obj·cls ≤ obj
                }
                let mut best = 0usize;
                let mut best_s = -1.0f32;
                for c in 0..nc {
                    let s = row[5 + c];
                    if s > best_s && class_ok(c) {
                        best_s = s;
                        best = c;
                    }
                }
                let score = obj * best_s;
                if score < cfg.conf {
                    continue;
                }
                let s = stride as f32;
                let cx = (row[0] + gx as f32) * s;
                let cy = (row[1] + gy as f32) * s;
                let w = row[2].min(20.0).exp() * s;
                let h = row[3].min(20.0).exp() * s;
                out.push(Detection {
                    class: best as u16,
                    score,
                    x0: cx - w / 2.0,
                    y0: cy - h / 2.0,
                    x1: cx + w / 2.0,
                    y1: cy + h / 2.0,
                    keypoints: None,
                });
            }
        }
    }
    out
}

/// Decode one image's `[4+nc, A]` slice (channel-major, anchors contiguous —
/// exactly the export layout) into candidate boxes in model space.
pub fn decode(pred: &[f32], nc: usize, anchors: usize, cfg: &DecodeCfg) -> Vec<Detection> {
    debug_assert_eq!(pred.len(), (4 + nc) * anchors);
    let mut out = Vec::new();
    let class_ok = |c: usize| cfg.classes.as_ref().is_none_or(|cs| cs.contains(&(c as u16)));
    for a in 0..anchors {
        let mut best = 0usize;
        let mut best_s = -1.0f32;
        for c in 0..nc {
            let s = pred[(4 + c) * anchors + a];
            if s > best_s && class_ok(c) {
                best_s = s;
                best = c;
            }
        }
        if best_s < cfg.conf {
            continue;
        }
        let cx = pred[a];
        let cy = pred[anchors + a];
        let w = pred[2 * anchors + a];
        let h = pred[3 * anchors + a];
        out.push(Detection {
            class: best as u16,
            score: best_s,
            x0: cx - w / 2.0,
            y0: cy - h / 2.0,
            x1: cx + w / 2.0,
            y1: cy + h / 2.0,
            keypoints: None,
        });
    }
    out
}

/// Greedy NMS. Sorting is (score desc, then original order) so equal scores
/// resolve identically run to run and batch to batch.
pub fn nms(mut dets: Vec<Detection>, cfg: &DecodeCfg) -> Vec<Detection> {
    let mut idx: Vec<usize> = (0..dets.len()).collect();
    idx.sort_by(|&a, &b| dets[b].score.partial_cmp(&dets[a].score).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    let mut keep: Vec<Detection> = Vec::new();
    'cand: for &i in &idx {
        if keep.len() >= cfg.max_det {
            break;
        }
        for k in &keep {
            if (cfg.class_agnostic_nms || k.class == dets[i].class) && k.iou(&dets[i]) > cfg.iou {
                continue 'cand;
            }
        }
        keep.push(dets[i]);
    }
    dets.clear();
    keep
}

/// Map model-space detections back to frame pixels and clip.
pub fn to_frame(dets: &mut [Detection], lb: &Letterbox) {
    for d in dets.iter_mut() {
        let (x0, y0) = lb.to_frame(d.x0, d.y0);
        let (x1, y1) = lb.to_frame(d.x1, d.y1);
        d.x0 = x0.clamp(0.0, lb.frame_w as f32);
        d.y0 = y0.clamp(0.0, lb.frame_h as f32);
        d.x1 = x1.clamp(0.0, lb.frame_w as f32);
        d.y1 = y1.clamp(0.0, lb.frame_h as f32);
        if let Some(kp) = &mut d.keypoints {
            for p in kp.iter_mut() {
                let (x, y) = lb.to_frame(p[0], p[1]);
                p[0] = x.clamp(0.0, lb.frame_w as f32);
                p[1] = y.clamp(0.0, lb.frame_h as f32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(class: u16, score: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> Detection {
        Detection { class, score, x0, y0, x1, y1, keypoints: None }
    }

    #[test]
    fn decode_layout_and_threshold() {
        // nc = 2, anchors = 3; anchor 1 has class-1 score 0.9, box (10,20,4,6)
        let nc = 2;
        let a = 3;
        let mut pred = vec![0.0f32; (4 + nc) * a];
        pred[1] = 10.0; // cx
        pred[a + 1] = 20.0; // cy
        pred[2 * a + 1] = 4.0; // w
        pred[3 * a + 1] = 6.0; // h
        pred[(4 + 1) * a + 1] = 0.9;
        pred[4 * a] = 0.2; // anchor 0 class 0, below conf
        let dets = decode(&pred, nc, a, &DecodeCfg::default());
        assert_eq!(dets.len(), 1);
        let d = dets[0];
        assert_eq!(d.class, 1);
        assert_eq!((d.x0, d.y0, d.x1, d.y1), (8.0, 17.0, 12.0, 23.0));
    }

    #[test]
    fn class_filter_picks_allowed_class() {
        // one anchor where class 0 scores higher than class 2, but only 2 allowed
        let nc = 3;
        let a = 1;
        let mut pred = vec![0.0f32; (4 + nc) * a];
        pred[2 * a] = 2.0;
        pred[3 * a] = 2.0;
        pred[4 * a] = 0.9; // class 0
        pred[6 * a] = 0.6; // class 2
        let cfg = DecodeCfg { classes: Some(vec![2]), ..Default::default() };
        let dets = decode(&pred, nc, a, &cfg);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class, 2);
        assert!((dets[0].score - 0.6).abs() < 1e-6);
    }

    #[test]
    fn nms_suppresses_overlap_keeps_classes_apart() {
        let cfg = DecodeCfg { iou: 0.5, ..Default::default() };
        let dets = vec![
            det(0, 0.9, 0.0, 0.0, 10.0, 10.0),
            det(0, 0.8, 1.0, 1.0, 11.0, 11.0), // IoU ~0.68 with first -> suppressed
            det(1, 0.7, 0.0, 0.0, 10.0, 10.0), // other class -> kept
            det(0, 0.6, 50.0, 50.0, 60.0, 60.0),
        ];
        let kept = nms(dets, &cfg);
        assert_eq!(kept.len(), 3);
        assert_eq!(kept[0].score, 0.9);
        assert_eq!(kept[1].class, 1);
        // agnostic mode suppresses the class-1 twin too
        let cfg2 = DecodeCfg { iou: 0.5, class_agnostic_nms: true, ..Default::default() };
        let dets2 = vec![det(0, 0.9, 0.0, 0.0, 10.0, 10.0), det(1, 0.7, 0.0, 0.0, 10.0, 10.0)];
        assert_eq!(nms(dets2, &cfg2).len(), 1);
    }

    #[test]
    fn deterministic_tie_break() {
        let cfg = DecodeCfg { iou: 0.9, max_det: 2, ..Default::default() };
        let dets = vec![det(0, 0.5, 0.0, 0.0, 1.0, 1.0), det(0, 0.5, 5.0, 5.0, 6.0, 6.0), det(0, 0.5, 9.0, 9.0, 10.0, 10.0)];
        let a = nms(dets.clone(), &cfg);
        let b = nms(dets, &cfg);
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
        assert_eq!((a[0].x0, a[1].x0), (0.0, 5.0)); // original order among ties
    }

    #[test]
    fn frame_mapping_clips() {
        let lb = Letterbox { scale: 1.0 / 3.0, pad_x: 0, pad_y: 140, frame_w: 1920, frame_h: 1080, target: 640 };
        let mut dets = vec![det(0, 0.9, -5.0, 130.0, 320.0, 500.0)];
        to_frame(&mut dets, &lb);
        assert_eq!(dets[0].x0, 0.0); // clipped
        assert!((dets[0].y0 - 0.0).abs() < 1e-3); // (130-140)/scale clipped to 0
        assert!((dets[0].x1 - 960.0).abs() < 1e-3);
        assert!((dets[0].y1 - 1080.0).abs() < 1e-3);
    }
}
