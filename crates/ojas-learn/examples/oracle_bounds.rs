//! mAP available to a re-classifier that never moves a box.
//!
//! One hypothesis per query (its highest class score), so every predicted box is matched at most
//! once. The published post-processor keeps the top 300 query × class pairs instead, letting
//! several hypotheses share one query's box; an oracle can then match them to different
//! ground-truth objects and invent headroom in crowded scenes. That baseline is reported alongside
//! for comparison.
//!
//! Oracles (boxes and scores never change, only labels of matched detections):
//!   - per family, one family at a time, so a gain can be attributed;
//!   - all families together, under the coarse tree and under a merged
//!     `goods` = Truck + LCV tree;
//!   - unrestricted (any class), the ceiling of perfect classification.
//! `oracle_bounds model.pth prefix annotations.json image_root [limit] [batch]`
use std::sync::Arc;

use ojas_learn::cuda::Cuda;
use ojas_learn::data::{load_coco, loader};
use ojas_learn::eval::{coco_map, postprocess, Det};
use ojas_learn::models::rtdetr::{Config, RtDetr, Store};
use ojas_learn::Tape;

/// The coarse class tree over UVH-26 ids 1..14.
fn family(c: usize) -> &'static str {
    match c {
        1 | 2 | 3 | 4 => "passenger",      // Hatchback, Sedan, SUV, MUV
        5 | 10 => "heavy passenger",        // Bus, Mini-bus
        6 => "heavy goods",                 // Truck
        7 => "three-wheel",                 // Three-wheeler
        8 | 12 => "two-wheel",              // Two-wheeler, Bicycle
        9 | 11 | 13 => "light commercial",  // LCV, Tempo-traveller, Van
        _ => "other",                       // Others
    }
}

/// The same tree with Truck and LCV merged.
fn family_goods_merged(c: usize) -> &'static str {
    match c {
        6 | 9 => "goods",                   // Truck, LCV
        11 | 13 => "light commercial",      // Tempo-traveller, Van
        _ => family(c),
    }
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let i = iw * ih;
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    if u <= 0.0 { 0.0 } else { i / u }
}

/// Greedy per-image matching by score: each detection takes the best free GT at IoU ≥ 0.5.
fn matches(gt: &[(usize, [f32; 4])], d: &[Det]) -> Vec<Option<usize>> {
    let mut order: Vec<usize> = (0..d.len()).collect();
    order.sort_by(|&a, &b| d[b].score.total_cmp(&d[a].score));
    let mut used = vec![false; gt.len()];
    let mut out = vec![None; d.len()];
    for k in order {
        let mut best = (0.5f32, None);
        for (j, g) in gt.iter().enumerate() {
            if used[j] {
                continue;
            }
            let v = iou(d[k].xyxy, g.1);
            if v >= best.0 {
                best = (v, Some(j));
            }
        }
        if let Some(j) = best.1 {
            used[j] = true;
            out[k] = Some(j);
        }
    }
    out
}

/// Relabel matched detections where `allow(predicted, truth)` says a re-classifier could reach
/// that correction; returns the new detections and how many changed.
fn oracle(gts: &[Vec<(usize, [f32; 4])>], dets: &[Vec<Det>], allow: impl Fn(usize, usize) -> bool) -> (Vec<Vec<Det>>, usize) {
    let mut out = dets.to_vec();
    let mut n = 0;
    for (i, (g, d)) in gts.iter().zip(dets.iter()).enumerate() {
        for (k, m) in matches(g, d).iter().enumerate() {
            let Some(j) = *m else { continue };
            let truth = g[j].0;
            if d[k].label != truth && allow(d[k].label, truth) {
                out[i][k].label = truth;
                n += 1;
            }
        }
    }
    (out, n)
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let limit: usize = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(usize::MAX);
    let batch: usize = a.get(6).map(|v| v.parse().unwrap()).unwrap_or(8);
    let mut samples = load_coco(a[3].as_ref(), a[4].as_ref())?;
    samples.truncate(limit);
    println!("# {}\n\n{} images, {} boxes", a[3].rsplit('/').next().unwrap_or(""), samples.len(), samples.iter().map(|s| s.boxes.len()).sum::<usize>());
    let be = Cuda::new(0)?;
    let st = Store::from_tensors(&be, &ojas_formats::pth::load(&std::fs::read(&a[1])?)?, &a[2]);
    let cfg = Config::r18vd(15);
    let m = RtDetr { cfg: cfg.clone(), st: &st, train: false, var: Default::default() };
    let samples = Arc::new(samples);
    let mut gts: Vec<Vec<(usize, [f32; 4])>> = vec![];
    let mut published: Vec<Vec<Det>> = vec![]; // top 300 query × class (what we reported before)
    let mut per_query: Vec<Vec<Det>> = vec![]; // one hypothesis per query
    for b in loader(samples.clone(), (0..samples.len()).collect(), batch, 640, false, 12, 0) {
        let b = b?;
        let n = b.samples.len();
        let mut t = Tape::new(&be);
        let x = t.input(&b.images, &[n, 3, 640, 640]);
        let o = m.forward(&mut t, x, None)?;
        let (lg, bx) = (t.value(o.logits[0]), t.value(o.boxes[0]));
        let q = cfg.num_queries;
        for (i, &si) in b.samples.iter().enumerate() {
            let s = &samples[si];
            let (l, bb) = (&lg[i * q * 15..(i + 1) * q * 15], &bx[i * q * 4..(i + 1) * q * 4]);
            published.push(postprocess(l, bb, 15, 300, s.width as f32, s.height as f32));
            // one hypothesis per query: its best class (ids 1..14; 0 is unused)
            let mut one = vec![];
            for qi in 0..q {
                let row = &l[qi * 15..qi * 15 + 15];
                let (mut best, mut lab) = (f32::NEG_INFINITY, 1usize);
                for (c, &v) in row.iter().enumerate().skip(1) {
                    if v > best {
                        best = v;
                        lab = c;
                    }
                }
                let bo = &bb[qi * 4..qi * 4 + 4];
                let (w, h) = (s.width as f32, s.height as f32);
                one.push(Det {
                    label: lab,
                    score: 1.0 / (1.0 + (-best).exp()),
                    xyxy: [(bo[0] - 0.5 * bo[2]) * w, (bo[1] - 0.5 * bo[3]) * h, (bo[0] + 0.5 * bo[2]) * w, (bo[1] + 0.5 * bo[3]) * h],
                });
            }
            per_query.push(one);
            gts.push(s.boxes.iter().map(|&(c, b)| (c, [b[0], b[1], b[0] + b[2], b[1] + b[3]])).collect());
        }
        if gts.len() % (batch * 50) == 0 {
            eprintln!("  {} / {}", gts.len(), samples.len());
        }
    }

    let pub_base = coco_map(&gts, &published);
    let base = coco_map(&gts, &per_query);
    println!("\n| variant | relabelled | mAP@[.5:.95] | Δ vs its baseline | AP50 |");
    println!("|---|---:|---:|---:|---:|");
    println!("| published post-process (top 300 query × class) | 0 | {:.4} | — | {:.4} |", pub_base.ap, pub_base.ap50);
    println!("| **one hypothesis per query** (the honest baseline) | 0 | {:.4} | — | {:.4} |", base.ap, base.ap50);
    // one family at a time, so the gain can be attributed
    for fam in ["passenger", "two-wheel", "heavy passenger", "light commercial"] {
        let (d, n) = oracle(&gts, &per_query, |p, t| family(p) == fam && family(t) == fam);
        let s = coco_map(&gts, &d);
        println!("| {fam} oracle only | {n} | {:.4} | +{:.4} | {:.4} |", s.ap, s.ap - base.ap, s.ap50);
    }
    let (d, n) = oracle(&gts, &per_query, |p, t| family_goods_merged(p) == "goods" && family_goods_merged(t) == "goods");
    let s = coco_map(&gts, &d);
    println!("| goods oracle only (Truck + LCV merged) | {n} | {:.4} | +{:.4} | {:.4} |", s.ap, s.ap - base.ap, s.ap50);
    let (d, n) = oracle(&gts, &per_query, |p, t| family(p) == family(t));
    let s = coco_map(&gts, &d);
    println!("| all within-family (plan's tree) | {n} | {:.4} | +{:.4} | {:.4} |", s.ap, s.ap - base.ap, s.ap50);
    let (d, n) = oracle(&gts, &per_query, |p, t| family_goods_merged(p) == family_goods_merged(t));
    let s = coco_map(&gts, &d);
    println!("| all within-family (goods merged) | {n} | {:.4} | +{:.4} | {:.4} |", s.ap, s.ap - base.ap, s.ap50);
    let (d, n) = oracle(&gts, &per_query, |_, _| true);
    let s = coco_map(&gts, &d);
    println!("| unrestricted class oracle | {n} | {:.4} | +{:.4} | {:.4} |", s.ap, s.ap - base.ap, s.ap50);
    Ok(())
}
