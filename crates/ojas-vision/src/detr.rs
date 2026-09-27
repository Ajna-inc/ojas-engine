//! DETR-family head decode (D-FINE / RT-DETR / DEIM exports): `logits [N,Q,C]`
//! (pre-sigmoid) and `boxes [N,Q,4]` (`cx, cy, w, h` as fractions of the input,
//! which the frame is stretched to — no letterbox, RGB 0–1). One hypothesis per
//! query, its top class, so there is no NMS: score = sigmoid(top logit), keep
//! ≥ `conf` (and in `classes` when given), best first, at most `max_det`.
//! Boxes come back in the pixels of the `w × h` region the query saw, clipped.

use crate::yolo::{DecodeCfg, Detection};

/// Decode one image's `[Q, C]` logits and `[Q, 4]` boxes for a `w × h` region.
pub fn decode(logits: &[f32], boxes: &[f32], nc: usize, queries: usize, w: usize, h: usize, cfg: &DecodeCfg) -> Vec<Detection> {
    debug_assert_eq!(logits.len(), queries * nc);
    debug_assert_eq!(boxes.len(), queries * 4);
    let (fw, fh) = (w as f32, h as f32);
    let mut dets = Vec::new();
    for q in 0..queries {
        let row = &logits[q * nc..(q + 1) * nc];
        let (mut best, mut bl) = (0usize, f32::NEG_INFINITY);
        for (c, &l) in row.iter().enumerate() {
            if l > bl {
                (best, bl) = (c, l);
            }
        }
        if !cfg.classes.as_ref().is_none_or(|cs| cs.contains(&(best as u16))) {
            continue;
        }
        let score = 1.0 / (1.0 + (-bl).exp());
        if score < cfg.conf {
            continue;
        }
        let b = &boxes[q * 4..q * 4 + 4];
        let (cx, cy, bw, bh) = (b[0] * fw, b[1] * fh, b[2] * fw, b[3] * fh);
        dets.push(Detection {
            class: best as u16,
            score,
            x0: (cx - bw / 2.0).clamp(0.0, fw),
            y0: (cy - bh / 2.0).clamp(0.0, fh),
            x1: (cx + bw / 2.0).clamp(0.0, fw),
            y1: (cy + bh / 2.0).clamp(0.0, fh),
            keypoints: None,
        });
    }
    // best first; equal scores keep query order, so a batch never reorders a frame's boxes
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    dets.truncate(cfg.max_det);
    dets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_class_sigmoid_and_frame_pixels() {
        // 2 queries, 3 classes, a 200x100 region
        let logits = [0.0, 2.0, -1.0, 3.0, 0.0, 0.0];
        let boxes = [0.5, 0.5, 0.2, 0.4, 0.0, 0.0, 0.5, 0.5];
        let cfg = DecodeCfg { conf: 0.5, ..Default::default() };
        let d = decode(&logits, &boxes, 3, 2, 200, 100, &cfg);
        assert_eq!(d.len(), 2);
        assert_eq!((d[0].class, d[1].class), (0, 1)); // best first: sigmoid(3) > sigmoid(2)
        assert!((d[0].score - 1.0 / (1.0 + (-3.0f32).exp())).abs() < 1e-6);
        // query 1: centre (100, 50), 40 x 40
        assert_eq!((d[1].x0, d[1].y0, d[1].x1, d[1].y1), (80.0, 30.0, 120.0, 70.0));
        // query 0's box is half outside: clipped
        assert_eq!((d[0].x0, d[0].y0, d[0].x1, d[0].y1), (0.0, 0.0, 50.0, 25.0));
        // class filter applies to the query's top class, not a lesser one
        let cfg = DecodeCfg { conf: 0.5, classes: Some(vec![1]), ..Default::default() };
        assert_eq!(decode(&logits, &boxes, 3, 2, 200, 100, &cfg).len(), 1);
        let cfg = DecodeCfg { conf: 0.5, max_det: 1, ..Default::default() };
        assert_eq!(decode(&logits, &boxes, 3, 2, 200, 100, &cfg).len(), 1);
    }
}
