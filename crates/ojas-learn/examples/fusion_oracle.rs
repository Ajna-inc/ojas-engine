//! Ceiling of fusing a crop expert into the detector's class distribution without discarding its
//! secondary hypotheses.
//!
//! The deployed pipeline flattens the top 300 query × class pairs, and keeping several hypotheses
//! per query is worth ~0.08 mAP, so a crop classifier has to fuse a distribution rather than
//! overwrite a label. For every query uniquely matched to a ground-truth box, a perfect
//! within-family crop expert adds a residual
//!
//!     fused[c] = detector[c] + alpha * (+1 if c is the truth,
//!                                       -1 if c is another member of the
//!                                          predicted class's family,
//!                                        0 otherwise)
//!
//! then the ordinary post-processor runs. alpha = 0 is the unmodified detector; hard label
//! replacement is the other extreme, reported by `oracle_bounds`.
//! `fusion_oracle model.pth prefix annotations.json image_root [limit] [batch]`
use std::sync::Arc;

use ojas_learn::cuda::Cuda;
use ojas_learn::data::{load_coco, loader};
use ojas_learn::eval::{coco_map, Det};
use ojas_learn::models::rtdetr::{Config, RtDetr, Store};
use ojas_learn::Tape;

/// Coarse families over UVH-26 ids 1..14, with Truck + LCV merged as `goods`, which is worth
/// +0.007 ST / +0.012 MV over the unmerged tree.
fn family(c: usize) -> u8 {
    match c {
        1 | 2 | 3 | 4 => 1,   // passenger: Hatchback, Sedan, SUV, MUV
        5 | 10 => 2,          // heavy passenger: Bus, Mini-bus
        6 | 9 => 3,           // goods: Truck, LCV
        7 => 4,               // three-wheel
        8 | 12 => 5,          // two-wheel
        11 | 13 => 6,         // light commercial: Tempo-traveller, Van
        _ => 7,               // other
    }
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let i = iw * ih;
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    if u <= 0.0 { 0.0 } else { i / u }
}

fn xyxy(b: &[f32], w: f32, h: f32) -> [f32; 4] {
    [(b[0] - 0.5 * b[2]) * w, (b[1] - 0.5 * b[3]) * h, (b[0] + 0.5 * b[2]) * w, (b[1] + 0.5 * b[3]) * h]
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
    let q = cfg.num_queries;
    // cache the raw head outputs so the alpha sweep needs no further inference
    let mut logits: Vec<Vec<f32>> = vec![];
    let mut boxes: Vec<Vec<f32>> = vec![];
    let mut gts: Vec<Vec<(usize, [f32; 4])>> = vec![];
    let mut wh: Vec<(f32, f32)> = vec![];
    for b in loader(samples.clone(), (0..samples.len()).collect(), batch, 640, false, 12, 0) {
        let b = b?;
        let n = b.samples.len();
        let mut t = Tape::new(&be);
        let x = t.input(&b.images, &[n, 3, 640, 640]);
        let o = m.forward(&mut t, x, None)?;
        let (lg, bx) = (t.value(o.logits[0]), t.value(o.boxes[0]));
        for (i, &si) in b.samples.iter().enumerate() {
            let s = &samples[si];
            logits.push(lg[i * q * 15..(i + 1) * q * 15].to_vec());
            boxes.push(bx[i * q * 4..(i + 1) * q * 4].to_vec());
            gts.push(s.boxes.iter().map(|&(c, b)| (c, [b[0], b[1], b[0] + b[2], b[1] + b[3]])).collect());
            wh.push((s.width as f32, s.height as f32));
        }
        if gts.len() % (batch * 50) == 0 {
            eprintln!("  {} / {}", gts.len(), samples.len());
        }
    }
    eprintln!("  cached {} images", gts.len());

    // which query owns which ground-truth box: one hypothesis per query, greedy by score
    let mut truth_of: Vec<Vec<Option<usize>>> = vec![]; // per image, per query
    for (i, g) in gts.iter().enumerate() {
        let (w, h) = wh[i];
        let mut top: Vec<(f32, usize, [f32; 4])> = (0..q)
            .map(|qi| {
                let row = &logits[i][qi * 15..qi * 15 + 15];
                let (mut best, mut lab) = (f32::NEG_INFINITY, 1usize);
                for (c, &v) in row.iter().enumerate().skip(1) {
                    if v > best {
                        best = v;
                        lab = c;
                    }
                }
                (1.0 / (1.0 + (-best).exp()), lab, xyxy(&boxes[i][qi * 4..qi * 4 + 4], w, h))
            })
            .collect();
        let mut order: Vec<usize> = (0..q).collect();
        order.sort_by(|&x, &y| top[y].0.total_cmp(&top[x].0));
        let mut used = vec![false; g.len()];
        let mut owner = vec![None; q];
        for qi in order {
            let mut best = (0.5f32, None);
            for (j, gg) in g.iter().enumerate() {
                if used[j] {
                    continue;
                }
                let v = iou(top[qi].2, gg.1);
                if v >= best.0 {
                    best = (v, Some(j));
                }
            }
            if let Some(j) = best.1 {
                used[j] = true;
                owner[qi] = Some(g[j].0);
            }
        }
        top.clear();
        truth_of.push(owner);
    }

    println!("\n| fusion alpha | mAP@[.5:.95] | Δ vs detector | AP50 | truth in query's top-1 | kept same-family alternatives |");
    println!("|---:|---:|---:|---:|---:|---:|");
    let mut base_ap = 0.0f32;
    for (ai, &alpha) in [0.0f32, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0].iter().enumerate() {
        let mut dets: Vec<Vec<Det>> = vec![];
        let (mut top1_right, mut matched, mut alt_kept) = (0usize, 0usize, 0usize);
        for i in 0..gts.len() {
            let (w, h) = wh[i];
            let mut l = logits[i].clone();
            for qi in 0..q {
                let Some(truth) = truth_of[i][qi] else { continue };
                let row = &mut l[qi * 15..qi * 15 + 15];
                // the expert only sees the family the detector already chose
                let (mut best, mut lab) = (f32::NEG_INFINITY, 1usize);
                for (c, &v) in row.iter().enumerate().skip(1) {
                    if v > best {
                        best = v;
                        lab = c;
                    }
                }
                if family(lab) != family(truth) {
                    continue; // outside a within-family expert's reach
                }
                for c in 1..15 {
                    if c == truth {
                        row[c] += alpha;
                    } else if family(c) == family(lab) {
                        row[c] -= alpha;
                    }
                }
                let (mut nb, mut nl) = (f32::NEG_INFINITY, 1usize);
                for c in 1..15 {
                    if row[c] > nb {
                        nb = row[c];
                        nl = c;
                    }
                }
                matched += 1;
                if nl == truth {
                    top1_right += 1;
                }
            }
            // the published post-processor, keeping each pair's query so the surviving
            // same-family alternatives can be counted
            let mut pairs: Vec<(f32, usize, usize)> = l.iter().enumerate().map(|(k, &v)| (1.0 / (1.0 + (-v).exp()), k / 15, k % 15)).filter(|x| x.2 != 0).collect();
            pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
            pairs.truncate(300);
            let mut d = Vec::with_capacity(pairs.len());
            for &(score, qi, label) in &pairs {
                if let Some(t) = truth_of[i][qi] {
                    if label != t && family(label) == family(t) {
                        alt_kept += 1;
                    }
                }
                d.push(Det { label, score, xyxy: xyxy(&boxes[i][qi * 4..qi * 4 + 4], w, h) });
            }
            dets.push(d);
        }
        let s = coco_map(&gts, &dets);
        if ai == 0 {
            base_ap = s.ap;
        }
        println!("| {alpha} | {:.4} | {:+.4} | {:.4} | {:.3} | {alt_kept} |", s.ap, s.ap - base_ap, s.ap50, top1_right as f32 / matched.max(1) as f32);
    }
    Ok(())
}
