//! COCO bbox evaluation (pycocotools COCOeval, iouType "bbox", area "all",
//! maxDets 100): AP@[.5:.95], AP50, AP75 — and the RT-DETR post-processor that
//! turns (logits, boxes) into scored detections.

use std::collections::HashMap;

/// One detection in original-image pixels (x1, y1, x2, y2).
#[derive(Clone, Copy, Debug)]
pub struct Det {
    pub label: usize,
    pub score: f32,
    pub xyxy: [f32; 4],
}

/// RTDETRPostProcessor (focal): sigmoid scores over queries × classes, the top
/// `k` pairs, boxes (cxcywh 0..1) scaled to `w`×`h`.
pub fn postprocess(logits: &[f32], boxes: &[f32], classes: usize, k: usize, w: f32, h: f32) -> Vec<Det> {
    let mut s: Vec<(f32, usize)> = logits.iter().enumerate().map(|(i, &v)| (1.0 / (1.0 + (-v).exp()), i)).collect();
    s.sort_by(|a, b| b.0.total_cmp(&a.0));
    s.truncate(k);
    s.iter()
        .map(|&(score, i)| {
            let (q, label) = (i / classes, i % classes);
            let b = &boxes[q * 4..q * 4 + 4];
            Det { label, score, xyxy: [(b[0] - 0.5 * b[2]) * w, (b[1] - 0.5 * b[3]) * h, (b[0] + 0.5 * b[2]) * w, (b[1] + 0.5 * b[3]) * h] }
        })
        .collect()
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let i = iw * ih;
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    if u <= 0.0 { 0.0 } else { i / u }
}

#[derive(Debug, Clone, Copy)]
pub struct CocoStats {
    pub ap: f32,
    pub ap50: f32,
    pub ap75: f32,
    pub categories: usize,
}

/// `gts[i]`: (label, xyxy) ground truth of image i; `dets[i]` its detections.
pub fn coco_map(gts: &[Vec<(usize, [f32; 4])>], dets: &[Vec<Det>]) -> CocoStats {
    let thresholds: Vec<f32> = (0..10).map(|i| 0.5 + 0.05 * i as f32).collect();
    let max_dets = 100;
    let mut cats: Vec<usize> = gts.iter().flatten().map(|g| g.0).collect();
    cats.sort();
    cats.dedup();
    // per category: (score, matched per threshold) over all images, and the gt count
    let mut per_cat: HashMap<usize, (Vec<(f32, Vec<bool>)>, usize)> = HashMap::new();
    for (img, d) in dets.iter().enumerate() {
        // COCOeval keeps each image's top max_dets detections per category
        let mut by_cat: HashMap<usize, Vec<&Det>> = HashMap::new();
        for x in d {
            by_cat.entry(x.label).or_default().push(x);
        }
        let mut gt_cats: HashMap<usize, Vec<[f32; 4]>> = HashMap::new();
        for g in &gts[img] {
            gt_cats.entry(g.0).or_default().push(g.1);
        }
        let all: std::collections::HashSet<usize> = by_cat.keys().chain(gt_cats.keys()).copied().collect();
        for c in all {
            let g = gt_cats.get(&c).cloned().unwrap_or_default();
            let mut ds = by_cat.get(&c).cloned().unwrap_or_default();
            ds.sort_by(|a, b| b.score.total_cmp(&a.score));
            ds.truncate(max_dets);
            let e = per_cat.entry(c).or_insert((vec![], 0));
            e.1 += g.len();
            let mut matched_rows: Vec<Vec<bool>> = vec![vec![false; thresholds.len()]; ds.len()];
            for (ti, &t) in thresholds.iter().enumerate() {
                let mut used = vec![false; g.len()];
                for (di, dd) in ds.iter().enumerate() {
                    // best unmatched gt with IoU ≥ t (pycocotools: iou ≥ min(t, 1 − 1e-10))
                    let mut best = (t.min(1.0 - 1e-10), None);
                    for (gi, gb) in g.iter().enumerate() {
                        if used[gi] {
                            continue;
                        }
                        let v = iou(dd.xyxy, *gb);
                        if v >= best.0 {
                            best = (v, Some(gi));
                        }
                    }
                    if let Some(gi) = best.1 {
                        used[gi] = true;
                        matched_rows[di][ti] = true;
                    }
                }
            }
            for (dd, m) in ds.iter().zip(matched_rows) {
                e.0.push((dd.score, m));
            }
        }
    }
    let rec_thrs: Vec<f32> = (0..101).map(|i| i as f32 / 100.0).collect();
    let (mut sum, mut sum50, mut sum75, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for c in &cats {
        let Some((d, npos)) = per_cat.get_mut(c) else { continue };
        if *npos == 0 {
            continue;
        }
        // mergesort (stable) by score, descending — pycocotools uses kind='mergesort'
        d.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut ap_t = vec![0.0f64; thresholds.len()];
        for ti in 0..thresholds.len() {
            let (mut tp, mut fp) = (0.0f64, 0.0f64);
            let mut rc = vec![];
            let mut pr = vec![];
            for (_, m) in d.iter() {
                if m[ti] { tp += 1.0 } else { fp += 1.0 }
                rc.push(tp / *npos as f64);
                pr.push(tp / (tp + fp + f64::EPSILON));
            }
            for i in (1..pr.len()).rev() {
                if pr[i] > pr[i - 1] {
                    pr[i - 1] = pr[i];
                }
            }
            let mut q = 0.0f64;
            for &r in &rec_thrs {
                let idx = rc.partition_point(|&x| x < r as f64);
                if idx < pr.len() {
                    q += pr[idx];
                }
            }
            ap_t[ti] = q / rec_thrs.len() as f64;
        }
        sum += ap_t.iter().sum::<f64>() / thresholds.len() as f64;
        sum50 += ap_t[0];
        sum75 += ap_t[5];
        n += 1;
    }
    let d = n.max(1) as f64;
    CocoStats { ap: (sum / d) as f32, ap50: (sum50 / d) as f32, ap75: (sum75 / d) as f32, categories: n }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_and_half_detections() {
        let gts = vec![vec![(1, [0.0, 0.0, 10.0, 10.0]), (2, [20.0, 20.0, 40.0, 40.0])]];
        let perfect = vec![vec![Det { label: 1, score: 0.9, xyxy: [0.0, 0.0, 10.0, 10.0] }, Det { label: 2, score: 0.8, xyxy: [20.0, 20.0, 40.0, 40.0] }]];
        let s = coco_map(&gts, &perfect);
        assert!((s.ap - 1.0).abs() < 1e-6, "{s:?}");
        // class 2 missed entirely: mAP = (1 + 0) / 2
        let half = vec![vec![Det { label: 1, score: 0.9, xyxy: [0.0, 0.0, 10.0, 10.0] }]];
        let s = coco_map(&gts, &half);
        assert!((s.ap - 0.5).abs() < 1e-6, "{s:?}");
        // a box shifted to IoU 0.6: counts at thresholds .5, .55, .6 only (3 of 10)
        let shifted = vec![vec![Det { label: 1, score: 0.9, xyxy: [2.5, 0.0, 12.5, 10.0] }]];
        let s = coco_map(&vec![vec![(1, [0.0, 0.0, 10.0, 10.0])]], &shifted);
        assert!((s.ap - 0.3).abs() < 1e-6 && s.ap50 == 1.0 && s.ap75 == 0.0, "{s:?}");
    }
}
