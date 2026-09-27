//! Find detections that never move. On frames sampled across a whole recording, a box that keeps
//! appearing in the same place is either a parked vehicle or a sign board, hoarding, pillar or
//! shadow the detector calls a vehicle. Groups detections into location clusters across time and
//! reports how much of the output is stationary, by class and score, saving crops of the worst
//! offenders.
//! `static_fp model.pth prefix frames_dir out_dir [score] [batch]`
use std::sync::Arc;

use ojas_learn::cuda::Cuda;
use ojas_learn::data::{decode, loader, Sample};
use ojas_learn::models::rtdetr::{Config, RtDetr, Store};
use ojas_learn::Tape;

const NAMES: [&str; 15] = ["-", "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"];

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let i = iw * ih;
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    if u <= 0.0 { 0.0 } else { i / u }
}

struct Cluster {
    box_sum: [f32; 4],
    n: usize,
    frames: Vec<usize>,
    scores: Vec<f32>,
    labels: Vec<usize>,
    first: (usize, [f32; 4]),
}

impl Cluster {
    fn mean(&self) -> [f32; 4] {
        [self.box_sum[0] / self.n as f32, self.box_sum[1] / self.n as f32, self.box_sum[2] / self.n as f32, self.box_sum[3] / self.n as f32]
    }
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() > 4, "static_fp model.pth prefix frames_dir out_dir [score] [batch]");
    let out_dir = std::path::PathBuf::from(&a[4]);
    std::fs::create_dir_all(&out_dir)?;
    let score_thr: f32 = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(0.35);
    let batch: usize = a.get(6).map(|v| v.parse().unwrap()).unwrap_or(8);
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&a[3])?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "png" || x == "jpg")).collect();
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no frames in {}", a[3]);
    let first = decode(&files[0])?;
    let samples: Vec<Sample> = files.iter().enumerate().map(|(i, p)| Sample { path: p.clone(), width: first.w, height: first.h, boxes: vec![], image_id: i as i64 }).collect();
    println!("{} frames, {}×{}, score ≥ {score_thr}", samples.len(), first.w, first.h);
    let be = Cuda::new(0)?;
    let st = Store::from_tensors(&be, &ojas_formats::pth::load(&std::fs::read(&a[1])?)?, &a[2]);
    let cfg = Config::r18vd(15);
    let m = RtDetr { cfg: cfg.clone(), st: &st, train: false, var: Default::default() };
    let n_frames = samples.len();
    let samples = Arc::new(samples);
    let q = cfg.num_queries;
    let (w, h) = (first.w as f32, first.h as f32);
    // one hypothesis per query above the threshold: what a deployment would keep
    let mut per_frame: Vec<Vec<(usize, f32, [f32; 4])>> = vec![vec![]; n_frames];
    for b in loader(samples.clone(), (0..n_frames).collect(), batch, 640, false, 8, 0) {
        let b = b?;
        let n = b.samples.len();
        let mut t = Tape::new(&be);
        let x = t.input(&b.images, &[n, 3, 640, 640]);
        let o = m.forward(&mut t, x, None)?;
        let (lg, bx) = (t.value(o.logits[0]), t.value(o.boxes[0]));
        for (i, &si) in b.samples.iter().enumerate() {
            let (l, bb) = (&lg[i * q * 15..(i + 1) * q * 15], &bx[i * q * 4..(i + 1) * q * 4]);
            let mut keep = vec![];
            for qi in 0..q {
                let row = &l[qi * 15..qi * 15 + 15];
                let (mut best, mut lab) = (f32::NEG_INFINITY, 1usize);
                for (c, &v) in row.iter().enumerate().skip(1) {
                    if v > best {
                        best = v;
                        lab = c;
                    }
                }
                let s = 1.0 / (1.0 + (-best).exp());
                if s < score_thr {
                    continue;
                }
                let bo = &bb[qi * 4..qi * 4 + 4];
                keep.push((lab, s, [(bo[0] - 0.5 * bo[2]) * w, (bo[1] - 0.5 * bo[3]) * h, (bo[0] + 0.5 * bo[2]) * w, (bo[1] + 0.5 * bo[3]) * h]));
            }
            per_frame[si] = keep;
        }
    }
    let total: usize = per_frame.iter().map(|v| v.len()).sum();
    println!("{total} detections kept, {:.1} per frame", total as f32 / n_frames as f32);
    // the class histogram, so two models' views of the same footage can be compared
    let mut hist: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for d in per_frame.iter().flatten() {
        *hist.entry(d.0).or_default() += 1;
    }
    let mut rows: Vec<(usize, usize)> = hist.into_iter().collect();
    rows.sort_by_key(|x| std::cmp::Reverse(x.1));
    println!("\n| class | detections | share |");
    println!("|---|---:|---:|");
    for (c, n) in &rows {
        println!("| {} | {n} | {:.1} % |", NAMES.get(*c).unwrap_or(&"?"), 100.0 * *n as f32 / total.max(1) as f32);
    }

    // group detections that sit in the same place across frames
    let mut clusters: Vec<Cluster> = vec![];
    for (fi, dets) in per_frame.iter().enumerate() {
        for &(lab, s, bx) in dets {
            let mut hit = None;
            let mut best = 0.6f32;
            for (ci, c) in clusters.iter().enumerate() {
                let v = iou(bx, c.mean());
                if v >= best {
                    best = v;
                    hit = Some(ci);
                }
            }
            match hit {
                Some(ci) => {
                    let c = &mut clusters[ci];
                    for k in 0..4 {
                        c.box_sum[k] += bx[k];
                    }
                    c.n += 1;
                    if *c.frames.last().unwrap() != fi {
                        c.frames.push(fi);
                    }
                    c.scores.push(s);
                    c.labels.push(lab);
                }
                None => clusters.push(Cluster { box_sum: bx, n: 1, frames: vec![fi], scores: vec![s], labels: vec![lab], first: (fi, bx) }),
            }
        }
    }
    // a cluster is stationary when it appears in a large share of the sampled frames, which
    // span the whole recording — a passing vehicle cannot
    let share = |c: &Cluster| c.frames.len() as f32 / n_frames as f32;
    let mut stat: Vec<&Cluster> = clusters.iter().filter(|c| share(c) >= 0.4).collect();
    stat.sort_by(|a, b| (b.frames.len(), b.n).cmp(&(a.frames.len(), a.n)));
    let in_static: usize = stat.iter().map(|c| c.n).sum();
    println!("\n## Stationary detections (same place in ≥ 40 % of frames spread over the recording)\n");
    println!("{} clusters, {in_static} detections — **{:.1} % of everything the detector emitted**", stat.len(), 100.0 * in_static as f32 / total.max(1) as f32);
    let mut by_class: std::collections::HashMap<usize, (usize, f32)> = std::collections::HashMap::new();
    for c in &stat {
        for (l, s) in c.labels.iter().zip(c.scores.iter()) {
            let e = by_class.entry(*l).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += s;
        }
    }
    let mut rows: Vec<(usize, (usize, f32))> = by_class.into_iter().collect();
    rows.sort_by_key(|x| std::cmp::Reverse(x.1 .0));
    println!("\n| class called | stationary detections | mean score |");
    println!("|---|---:|---:|");
    for (l, (n, s)) in &rows {
        println!("| {} | {n} | {:.3} |", NAMES.get(*l).unwrap_or(&"?"), s / *n as f32);
    }
    println!("\n| # | class | frames present | mean score | box (x1,y1,x2,y2) | crop |");
    println!("|---:|---|---:|---:|---|---|");
    for (i, c) in stat.iter().take(20).enumerate() {
        let b = c.mean();
        let mut lab = c.labels.clone();
        lab.sort();
        let modal = lab[lab.len() / 2];
        let ms = c.scores.iter().sum::<f32>() / c.scores.len() as f32;
        // save the crop from the frame it first appeared in
        let img = decode(&files[c.first.0])?;
        let (x0, y0) = (b[0].max(0.0) as usize, b[1].max(0.0) as usize);
        let (x1, y1) = ((b[2] as usize).min(img.w), (b[3] as usize).min(img.h));
        let name = format!("static_{i:02}_{}_{:.2}.png", NAMES.get(modal).unwrap_or(&"?"), ms);
        if x1 > x0 + 4 && y1 > y0 + 4 {
            let (cw, ch) = (x1 - x0, y1 - y0);
            let mut buf = Vec::with_capacity(cw * ch * 3);
            for y in y0..y1 {
                for x in x0..x1 {
                    let i0 = (y * img.w + x) * 3;
                    buf.extend_from_slice(&[(img.rgb[i0] * 255.0) as u8, (img.rgb[i0 + 1] * 255.0) as u8, (img.rgb[i0 + 2] * 255.0) as u8]);
                }
            }
            image::save_buffer(out_dir.join(&name), &buf, cw as u32, ch as u32, image::ColorType::Rgb8)?;
        }
        println!("| {i} | {} | {} / {n_frames} | {ms:.3} | {:.0},{:.0},{:.0},{:.0} | {name} |", NAMES.get(modal).unwrap_or(&"?"), c.frames.len(), b[0], b[1], b[2], b[3]);
    }
    println!("\nCrops written to {}", out_dir.display());
    Ok(())
}
