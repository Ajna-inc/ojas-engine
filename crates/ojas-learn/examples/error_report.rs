//! Error report for a detector on a COCO-format set: per-class AP and recall, recall by object
//! size, a TIDE-style breakdown of what each missed ground-truth box lost to (classification,
//! localization, both, or nothing near it), duplicate predictions, background false positives and
//! the confusion pairs — separating localization, recall and subtype confusion as error sources.
//! `error_report model.pth prefix annotations.json image_root [limit] [batch] [score]`
use std::collections::HashMap;
use std::sync::Arc;

use ojas_learn::cuda::Cuda;
use ojas_learn::data::{load_coco, loader};
use ojas_learn::eval::{coco_map, postprocess, Det};
use ojas_learn::models::rtdetr::{Config, RtDetr, Store};
use ojas_learn::Tape;

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let i = iw * ih;
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    if u <= 0.0 { 0.0 } else { i / u }
}

fn area(b: [f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

/// COCO size bands by ground-truth area.
fn size_of(b: [f32; 4]) -> usize {
    let a = area(b);
    if a < 32.0 * 32.0 {
        0
    } else if a < 96.0 * 96.0 {
        1
    } else {
        2
    }
}

/// Class-agnostic, score-ordered, one-to-one matching of unique queries.
/// Rows are passenger GT classes; columns are all predicted classes, then missed.
fn passenger_confusion(gts: &[Vec<(usize, [f32; 4])>], dets: &[Vec<Det>], threshold: f32) -> [[usize; 16]; 4] {
    let mut counts = [[0; 16]; 4];
    for (gt, predictions) in gts.iter().zip(dets) {
        let mut used = vec![false; gt.len()];
        let mut kept: Vec<_> = predictions.iter().filter(|d| d.score >= threshold).collect();
        kept.sort_by(|a, b| b.score.total_cmp(&a.score));
        for d in kept {
            let mut best = (0.5, None);
            for (j, g) in gt.iter().enumerate() {
                if !used[j] {
                    let overlap = iou(d.xyxy, g.1);
                    if overlap >= best.0 { best = (overlap, Some(j)); }
                }
            }
            if let Some(j) = best.1 {
                used[j] = true;
                if (1..=4).contains(&gt[j].0) { counts[gt[j].0 - 1][d.label] += 1; }
            }
        }
        for (j, g) in gt.iter().enumerate() {
            if !used[j] && (1..=4).contains(&g.0) { counts[g.0 - 1][15] += 1; }
        }
    }
    counts
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let limit: usize = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(usize::MAX);
    let batch: usize = a.get(6).map(|v| v.parse().unwrap()).unwrap_or(8);
    let score_thr: f32 = a.get(7).map(|v| v.parse().unwrap()).unwrap_or(0.3);
    let names: Vec<String> = match std::env::var("CLASS_NAMES") {
        Ok(p) => {
            let j: serde_json::Value = serde_json::from_slice(&std::fs::read(&p)?)?;
            let mut v = vec!["-".to_string(); 15];
            for c in j["categories"].as_array().cloned().unwrap_or_default() {
                let (id, n) = (c["id"].as_u64().unwrap_or(0) as usize, c["name"].as_str().unwrap_or("?").to_string());
                if id < v.len() {
                    v[id] = n;
                }
            }
            v
        }
        Err(_) => (0..15).map(|i| i.to_string()).collect(),
    };
    let mut samples = load_coco(a[3].as_ref(), a[4].as_ref())?;
    samples.truncate(limit);
    println!("{} images, {} boxes, score threshold {score_thr}\n", samples.len(), samples.iter().map(|s| s.boxes.len()).sum::<usize>());
    let be = Cuda::new(0)?;
    let st = Store::from_tensors(&be, &ojas_formats::pth::load(&std::fs::read(&a[1])?)?, &a[2]);
    let cfg = Config::r18vd(15);
    let m = RtDetr { cfg: cfg.clone(), st: &st, train: false, var: Default::default() };
    let samples = Arc::new(samples);
    let t0 = std::time::Instant::now();
    let (mut gts, mut dets): (Vec<Vec<(usize, [f32; 4])>>, Vec<Vec<Det>>) = (vec![], vec![]);
    let mut raw: Vec<(Vec<f32>, Vec<f32>, f32, f32)> = vec![];
    for b in loader(samples.clone(), (0..samples.len()).collect(), batch, 640, false, 12, 0) {
        let b = b?;
        let n = b.samples.len();
        let mut t = Tape::new(&be);
        let x = t.input(&b.images, &[n, 3, 640, 640]);
        let o = m.forward(&mut t, x, None)?;
        let (lg, bx) = (t.value(o.logits[0]), t.value(o.boxes[0]));
        for (i, &si) in b.samples.iter().enumerate() {
            let s = &samples[si];
            let q = cfg.num_queries;
            raw.push((lg[i * q * 15..(i + 1) * q * 15].to_vec(), bx[i * q * 4..(i + 1) * q * 4].to_vec(), s.width as f32, s.height as f32));
            dets.push(postprocess(&lg[i * q * 15..(i + 1) * q * 15], &bx[i * q * 4..(i + 1) * q * 4], 15, 300, s.width as f32, s.height as f32));
            gts.push(s.boxes.iter().map(|&(c, b)| (c, [b[0], b[1], b[0] + b[2], b[1] + b[3]])).collect());
        }
        if dets.len() % (batch * 50) == 0 {
            eprintln!("  {} / {}", dets.len(), samples.len());
        }
    }
    eprintln!("  inference done in {:.0} s", t0.elapsed().as_secs_f64());
    // one hypothesis per query (its best class): what a deployment keeps
    let per_query: Vec<Vec<Det>> = raw
        .iter()
        .map(|(l, b, w, h)| {
            (0..l.len() / 15)
                .map(|qi| {
                    let row = &l[qi * 15..qi * 15 + 15];
                    let (mut best, mut lab) = (f32::NEG_INFINITY, 1usize);
                    for (c, &v) in row.iter().enumerate().skip(1) {
                        if v > best {
                            best = v;
                            lab = c;
                        }
                    }
                    let bo = &b[qi * 4..qi * 4 + 4];
                    Det { label: lab, score: 1.0 / (1.0 + (-best).exp()), xyxy: [(bo[0] - 0.5 * bo[2]) * w, (bo[1] - 0.5 * bo[3]) * h, (bo[0] + 0.5 * bo[2]) * w, (bo[1] + 0.5 * bo[3]) * h] }
                })
                .collect()
        })
        .collect();

    let passenger = passenger_confusion(&gts, &per_query, score_thr);
    let total: usize = passenger.iter().flatten().sum();
    let missed: usize = passenger.iter().map(|r| r[15]).sum();
    let correct: usize = (0..4).map(|i| passenger[i][i + 1]).sum();
    println!("## Passenger subtype baseline (unique queries)\n");
    println!("Class-agnostic greedy matching against ALL GT, IoU >= 0.5, score >= {score_thr}; one query and one GT per match. No GT-based routing.");
    println!("GT {total}; localized {}; correct subtype {correct}; missed {missed}.", total - missed);
    if total > missed { println!("Conditional subtype accuracy {:.4}", correct as f64 / (total - missed) as f64); }
    if total > 0 { println!("Localization coverage {:.4}; correct subtype / all passenger GT {:.4}", (total - missed) as f64 / total as f64, correct as f64 / total as f64); }
    println!("Unmatched predictions are excluded here; detector AP and precision below account for false positives.\n");
    println!("| GT | Hatchback | Sedan | SUV | MUV | other predicted class | missed |");
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for (i, row) in passenger.iter().enumerate() {
        println!("| {} | {} | {} | {} | {} | {} | {} |", ["Hatchback", "Sedan", "SUV", "MUV"][i], row[1], row[2], row[3], row[4], row[0] + row[5..15].iter().sum::<usize>(), row[15]);
    }

    // ---- per class: AP (all detections) and recall / precision at the threshold
    let all = coco_map(&gts, &dets);
    println!("## Overall\n\nmAP@[.5:.95] {:.4}  AP50 {:.4}  AP75 {:.4}  ({} categories)\n", all.ap, all.ap50, all.ap75, all.categories);
    let mut cls: Vec<usize> = gts.iter().flatten().map(|g| g.0).collect();
    cls.sort();
    cls.dedup();
    println!("## Per class\n");
    println!("| class | GT | AP | AP50 | recall@{score_thr} | precision@{score_thr} |");
    println!("|---|---:|---:|---:|---:|---:|");
    let mut rows: Vec<(f32, String)> = vec![];
    for &c in &cls {
        let g: Vec<Vec<(usize, [f32; 4])>> = gts.iter().map(|v| v.iter().filter(|x| x.0 == c).cloned().collect()).collect();
        let d: Vec<Vec<Det>> = dets.iter().map(|v| v.iter().filter(|x| x.label == c).cloned().collect()).collect();
        let s = coco_map(&g, &d);
        let n_gt: usize = g.iter().map(|v| v.len()).sum();
        // recall / precision at the score threshold, IoU 0.5
        let (mut tp, mut fp, mut matched) = (0usize, 0usize, 0usize);
        for (gi, di) in g.iter().zip(d.iter()) {
            let mut used = vec![false; gi.len()];
            let mut kept: Vec<&Det> = di.iter().filter(|x| x.score >= score_thr).collect();
            kept.sort_by(|a, b| b.score.total_cmp(&a.score));
            for x in kept {
                let mut best = (0.5f32, None);
                for (j, gg) in gi.iter().enumerate() {
                    if used[j] {
                        continue;
                    }
                    let v = iou(x.xyxy, gg.1);
                    if v >= best.0 {
                        best = (v, Some(j));
                    }
                }
                match best.1 {
                    Some(j) => {
                        used[j] = true;
                        tp += 1;
                    }
                    None => fp += 1,
                }
            }
            matched += used.iter().filter(|x| **x).count();
        }
        let rec = if n_gt > 0 { matched as f32 / n_gt as f32 } else { 0.0 };
        let prec = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
        rows.push((s.ap, format!("| {} | {n_gt} | {:.4} | {:.4} | {:.3} | {:.3} |", names.get(c).cloned().unwrap_or(c.to_string()), s.ap, s.ap50, rec, prec)));
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, r) in &rows {
        println!("{r}");
    }

    // ---- score-threshold trade: what a deployment keeps, finds and invents
    println!("\n## Score threshold sweep (one hypothesis per query)\n");
    println!("| score ≥ | detections / image | recall | precision |");
    println!("|---:|---:|---:|---:|");
    for thr in [0.1f32, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6] {
        let (mut kept, mut found, mut gt_n) = (0usize, 0usize, 0usize);
        for (g, d) in gts.iter().zip(per_query.iter()) {
            let mut k: Vec<&Det> = d.iter().filter(|x| x.score >= thr).collect();
            k.sort_by(|a, b| b.score.total_cmp(&a.score));
            kept += k.len();
            gt_n += g.len();
            let mut used = vec![false; g.len()];
            for x in k {
                let mut best = (0.5f32, None);
                for (j, gg) in g.iter().enumerate() {
                    if used[j] || gg.0 != x.label {
                        continue;
                    }
                    let v = iou(x.xyxy, gg.1);
                    if v >= best.0 {
                        best = (v, Some(j));
                    }
                }
                if let Some(j) = best.1 {
                    used[j] = true;
                }
            }
            found += used.iter().filter(|x| **x).count();
        }
        println!("| {thr} | {:.1} | {:.3} | {:.3} |", kept as f32 / gts.len() as f32, found as f32 / gt_n.max(1) as f32, found as f32 / kept.max(1) as f32);
    }

    // ---- what each ground-truth box lost to (TIDE-style), and the false positives
    let (mut miss, mut cls_err, mut loc_err, mut both, mut nothing, mut hit) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut by_size = [[0usize; 2]; 3]; // [size][0]=gt, [1]=found
    let (mut dup, mut bg_fp, mut kept_total) = (0usize, 0usize, 0usize);
    let mut confusion: HashMap<(usize, usize), usize> = HashMap::new();
    for (gi, di) in gts.iter().zip(dets.iter()) {
        let mut kept: Vec<&Det> = di.iter().filter(|x| x.score >= score_thr).collect();
        kept.sort_by(|a, b| b.score.total_cmp(&a.score));
        kept_total += kept.len();
        let mut used = vec![false; gi.len()];
        let mut det_matched = vec![false; kept.len()];
        // greedy match, same class, IoU >= 0.5
        for (k, x) in kept.iter().enumerate() {
            let mut best = (0.5f32, None);
            for (j, gg) in gi.iter().enumerate() {
                if used[j] || gg.0 != x.label {
                    continue;
                }
                let v = iou(x.xyxy, gg.1);
                if v >= best.0 {
                    best = (v, Some(j));
                }
            }
            if let Some(j) = best.1 {
                used[j] = true;
                det_matched[k] = true;
            }
        }
        for (j, gg) in gi.iter().enumerate() {
            let sz = size_of(gg.1);
            by_size[sz][0] += 1;
            if used[j] {
                hit += 1;
                by_size[sz][1] += 1;
                continue;
            }
            miss += 1;
            // what was near it
            let mut best_any = (0.0f32, 0usize);
            let mut best_same = 0.0f32;
            for x in &kept {
                let v = iou(x.xyxy, gg.1);
                if v > best_any.0 {
                    best_any = (v, x.label);
                }
                if x.label == gg.0 {
                    best_same = best_same.max(v);
                }
            }
            let good_box = best_any.0 >= 0.5;
            let loose_box = best_any.0 >= 0.1;
            if good_box && best_any.1 != gg.0 {
                cls_err += 1;
                *confusion.entry((gg.0, best_any.1)).or_default() += 1;
            } else if best_same >= 0.1 && best_same < 0.5 {
                loc_err += 1;
            } else if loose_box {
                both += 1;
                *confusion.entry((gg.0, best_any.1)).or_default() += 1;
            } else {
                nothing += 1;
            }
        }
        for (k, x) in kept.iter().enumerate() {
            if det_matched[k] {
                continue;
            }
            let near = gi.iter().map(|g| iou(x.xyxy, g.1)).fold(0.0f32, f32::max);
            if near >= 0.5 {
                dup += 1;
            } else if near < 0.1 {
                bg_fp += 1;
            }
        }
    }
    let gt_total = hit + miss;
    let pc = |n: usize| 100.0 * n as f32 / gt_total.max(1) as f32;
    println!("\n## Where the ground truth goes (IoU 0.5, score ≥ {score_thr})\n");
    println!("| outcome | boxes | share |");
    println!("|---|---:|---:|");
    println!("| detected | {hit} | {:.1} % |", pc(hit));
    println!("| missed: **nothing predicted there** | {nothing} | {:.1} % |", pc(nothing));
    println!("| missed: **wrong class**, box was good | {cls_err} | {:.1} % |", pc(cls_err));
    println!("| missed: **box too loose**, class was right | {loc_err} | {:.1} % |", pc(loc_err));
    println!("| missed: wrong class *and* loose box | {both} | {:.1} % |", pc(both));
    println!("\nFalse positives at this threshold: {dup} duplicates, {bg_fp} on background, of {kept_total} kept predictions.");
    println!("\n## Recall by object size\n");
    println!("| size | GT | detected | recall |");
    println!("|---|---:|---:|---:|");
    for (i, n) in ["small (<32²)", "medium (<96²)", "large"].iter().enumerate() {
        let (g, f) = (by_size[i][0], by_size[i][1]);
        println!("| {n} | {g} | {f} | {:.3} |", f as f32 / g.max(1) as f32);
    }
    // ---- what raising the deployment confidence threshold costs
    println!("\n## Confidence threshold sweep (one hypothesis per query would differ; this is the published post-process)\n");
    println!("| score ≥ | detections / image | recall | precision |");
    println!("|---:|---:|---:|---:|");
    for thr in [0.1f32, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6] {
        let (mut kept, mut found, mut gt_n, mut tp) = (0usize, 0usize, 0usize, 0usize);
        for (g, d) in gts.iter().zip(dets.iter()) {
            let mut k: Vec<&Det> = d.iter().filter(|x| x.score >= thr).collect();
            k.sort_by(|a, b| b.score.total_cmp(&a.score));
            kept += k.len();
            gt_n += g.len();
            let mut used = vec![false; g.len()];
            for x in &k {
                let mut best = (0.5f32, None);
                for (j, gg) in g.iter().enumerate() {
                    if used[j] || gg.0 != x.label {
                        continue;
                    }
                    let v = iou(x.xyxy, gg.1);
                    if v >= best.0 {
                        best = (v, Some(j));
                    }
                }
                if let Some(j) = best.1 {
                    used[j] = true;
                    tp += 1;
                }
            }
            found += used.iter().filter(|x| **x).count();
        }
        println!("| {thr} | {:.1} | {:.3} | {:.3} |", kept as f32 / gts.len() as f32, found as f32 / gt_n.max(1) as f32, tp as f32 / kept.max(1) as f32);
    }

    let mut cf: Vec<((usize, usize), usize)> = confusion.into_iter().collect();
    cf.sort_by_key(|x| std::cmp::Reverse(x.1));
    println!("\n## Top confusions (ground truth → what was predicted there)\n");
    println!("| ground truth | predicted | boxes |");
    println!("|---|---|---:|");
    for ((g, p), n) in cf.into_iter().take(12) {
        println!("| {} | {} | {n} |", names.get(g).cloned().unwrap_or(g.to_string()), names.get(p).cloned().unwrap_or(p.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passenger_matching_counts_wrong_classes_and_misses_without_reusing_boxes() {
        let box_a = [0., 0., 10., 10.];
        let box_b = [20., 0., 30., 10.];
        let gt = vec![vec![(1, box_a), (2, box_a), (8, box_b), (3, [40., 0., 50., 10.])]];
        let predictions = vec![vec![
            Det { label: 4, score: 0.9, xyxy: box_a },
            Det { label: 1, score: 0.8, xyxy: box_b },
            Det { label: 3, score: 0.1, xyxy: [40., 0., 50., 10.] },
        ]];
        let counts = passenger_confusion(&gt, &predictions, 0.3);
        assert_eq!(counts.iter().map(|r| r[4]).sum::<usize>(), 1);
        assert_eq!(counts.iter().map(|r| r[15]).sum::<usize>(), 2);
        assert_eq!(counts.iter().map(|r| r[1]).sum::<usize>(), 0);
        assert_eq!(counts.iter().flatten().sum::<usize>(), 3);
    }
}
