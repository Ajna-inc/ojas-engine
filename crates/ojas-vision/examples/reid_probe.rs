//! How well an appearance embedding tells people apart on recorded footage, without
//! identity labels. Consecutive frames of one camera (~0.5 s apart) show the same
//! people a few pixels on, so a box in frame t and the box nearest it in frame t+1
//! (same size band) are taken as one person and every other box in frame t+1 as
//! someone else. For each such pair the probe checks whether the embedding ranks the
//! true continuation first among the next frame's people — one-step
//! re-identification by appearance alone, scored over thousands of pairs and by
//! person height. It also reports the cosine gap between same-person and
//! other-person pairs.
//!
//! With `TEXT=siglip2_text.json` (SigLIP: text embeddings of attribute phrases, with the logit
//! scale and bias) it also writes, per phrase, a contact sheet of the crops the model ranks
//! highest — search by description, judged by eye.
//!
//! `reid_probe <frames.json> <persons.dets.json> <model.onnx> <out_dir> [score 0.4] [min_h 60] [max_pairs 2000]`
//! Env: `MODEL=osnet|siglip` (input size and normalisation), `OJAS_DEVICE=cuda:0|cpu`, `TEXT=…`.
use std::collections::{BTreeMap, HashMap};

use image::{imageops, Rgb, RgbImage};
use serde_json::Value;

#[derive(Clone, Copy)]
struct Box_ {
    id: i64,
    b: [f32; 4],
}

struct Embedder {
    g: ojas_vision::ir::Graph,
    w: usize,
    h: usize,
    mean: [f32; 3],
    std: [f32; 3],
    out: usize, // which graph output is the embedding
    cpu: Option<ojas_vision::exec_cpu::CpuExecutor>,
    #[cfg(feature = "cuda")]
    gpu: Option<ojas_vision::exec_gpu::CudaExecutor>,
    input: Vec<f32>,
}

impl Embedder {
    fn load(path: &str, kind: &str, device: &str) -> anyhow::Result<Self> {
        let (w, h, mean, std, binds) = match kind {
            "osnet" => (128, 256, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], HashMap::new()),
            "siglip" => (224, 224, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], HashMap::from([("batch_size".to_string(), 1usize), ("num_channels".to_string(), 3), ("height".to_string(), 224), ("width".to_string(), 224)])),
            other => anyhow::bail!("MODEL={other}: osnet or siglip"),
        };
        let model = ojas_formats::onnx::load(path)?;
        let mut g = ojas_vision::import(&model, &binds)?;
        ojas_vision::passes::optimize(&mut g);
        let out = g.outputs.len() - 1; // OSNet: features; SigLIP: [last_hidden_state, pooler_output]
        let mut e = Embedder { w, h, mean, std, out, cpu: None, input: vec![0.0; 3 * w * h], g, #[cfg(feature = "cuda")] gpu: None };
        match device {
            #[cfg(feature = "cuda")]
            d if d.starts_with("cuda") => {
                ojas_vision::passes::lower_for_gpu(&mut e.g);
                e.gpu = Some(ojas_vision::exec_gpu::CudaExecutor::new(&e.g, 0)?);
            }
            _ => e.cpu = Some(ojas_vision::exec_cpu::CpuExecutor::new(&e.g, std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))),
        }
        Ok(e)
    }

    /// The crop (stretched to the input, as torchreid / the SigLIP processor do), L2-normalised embedding.
    fn embed(&mut self, crop: &RgbImage) -> anyhow::Result<Vec<f32>> {
        let small = imageops::resize(crop, self.w as u32, self.h as u32, imageops::FilterType::Triangle);
        let (w, h) = (self.w, self.h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    self.input[c * w * h + y * w + x] = (small.get_pixel(x as u32, y as u32).0[c] as f32 / 255.0 - self.mean[c]) / self.std[c];
                }
            }
        }
        #[cfg(feature = "cuda")]
        let outs = match self.gpu.as_mut() {
            Some(ex) => ex.run(&self.input)?,
            None => self.cpu.as_mut().unwrap().run(&self.g, &[&self.input])?,
        };
        #[cfg(not(feature = "cuda"))]
        let outs = self.cpu.as_mut().unwrap().run(&self.g, &[&self.input])?;
        let mut v = outs[self.out].clone();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        v.iter_mut().for_each(|x| *x /= n);
        Ok(v)
    }
}

fn crop(img: &RgbImage, b: [f32; 4]) -> RgbImage {
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let x0 = b[0].max(0.0).min(iw - 1.0);
    let y0 = b[1].max(0.0).min(ih - 1.0);
    let x1 = (b[0] + b[2]).min(iw).max(x0 + 1.0);
    let y1 = (b[1] + b[3]).min(ih).max(y0 + 1.0);
    imageops::crop_imm(img, x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32).to_image()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

const BANDS: [(f32, &str); 4] = [(100.0, "60-100"), (150.0, "100-150"), (250.0, "150-250"), (f32::MAX, "≥250")];

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 5, "reid_probe <frames.json> <persons.dets.json> <model.onnx> <out_dir> [score] [min_h] [max_pairs]");
    let score: f32 = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(0.4);
    let min_h: f32 = a.get(6).and_then(|v| v.parse().ok()).unwrap_or(60.0);
    let max_pairs: usize = a.get(7).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let kind = std::env::var("MODEL").unwrap_or_else(|_| "osnet".into());
    let device = std::env::var("OJAS_DEVICE").unwrap_or_else(|_| "cpu".into());
    let out_dir = std::path::Path::new(&a[4]);
    std::fs::create_dir_all(out_dir)?;

    // frames per camera in time order, boxes per frame
    let frames: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let mut files: HashMap<i64, String> = HashMap::new();
    let mut by_cam: BTreeMap<String, Vec<(i64, i64)>> = BTreeMap::new(); // (utc_ms, id)
    for im in frames["images"].as_array().unwrap() {
        let id = im["id"].as_i64().unwrap();
        files.insert(id, im["file_name"].as_str().unwrap().to_string());
        by_cam.entry(im["camera"].as_str().unwrap_or("?").to_string()).or_default().push((im["utc_ms"].as_i64().unwrap_or(0), id));
    }
    let dets: Vec<Value> = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    let mut boxes: HashMap<i64, Vec<Box_>> = HashMap::new();
    for d in &dets {
        if (d["score"].as_f64().unwrap() as f32) < score {
            continue;
        }
        let b: Vec<f32> = d["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        if b[3] < min_h {
            continue;
        }
        let id = d["image_id"].as_i64().unwrap();
        boxes.entry(id).or_default().push(Box_ { id, b: [b[0], b[1], b[2], b[3]] });
    }

    // pairs: (box in t, its continuation in t+1, the other boxes in t+1)
    let mut pairs: Vec<(Box_, Box_, Vec<Box_>)> = vec![];
    for (_, mut fr) in by_cam {
        fr.sort_unstable();
        for w in fr.windows(2) {
            let ((t0, f0), (t1, f1)) = (w[0], w[1]);
            if t1 - t0 > 1500 {
                continue;
            }
            let (Some(b0), Some(b1)) = (boxes.get(&f0), boxes.get(&f1)) else { continue };
            if b1.len() < 2 {
                continue; // rank-1 would be trivial
            }
            let mut used = vec![false; b1.len()];
            for x in b0 {
                let (cx, cy, hx) = (x.b[0] + x.b[2] / 2.0, x.b[1] + x.b[3] / 2.0, x.b[3]);
                let mut best: Option<(f32, usize)> = None;
                for (j, y) in b1.iter().enumerate() {
                    if used[j] {
                        continue;
                    }
                    let (dx, dy) = (y.b[0] + y.b[2] / 2.0 - cx, y.b[1] + y.b[3] / 2.0 - cy);
                    let dist = (dx * dx + dy * dy).sqrt() / hx;
                    let ratio = y.b[3] / hx;
                    if dist < 0.6 && (0.7..1.43).contains(&ratio) && best.is_none_or(|(d, _)| dist < d) {
                        best = Some((dist, j));
                    }
                }
                if let Some((_, j)) = best {
                    used[j] = true;
                    let others: Vec<Box_> = b1.iter().enumerate().filter(|(k, _)| *k != j).map(|(_, y)| *y).collect();
                    pairs.push((*x, b1[j], others));
                }
            }
        }
    }
    let step = (pairs.len() / max_pairs).max(1);
    let pairs: Vec<_> = pairs.into_iter().step_by(step).take(max_pairs).collect();
    println!("{} consecutive-frame pairs with ≥ 1 distractor (score ≥ {score}, height ≥ {min_h} px), {kind} on {device}", pairs.len());

    // embed every box involved, one image decode per frame
    let mut emb = Embedder::load(&a[3], &kind, &device)?;
    let mut need: BTreeMap<i64, Vec<[f32; 4]>> = BTreeMap::new();
    for (x, y, others) in &pairs {
        need.entry(x.id).or_default().push(x.b);
        need.entry(y.id).or_default().push(y.b);
        for o in others {
            need.entry(o.id).or_default().push(o.b);
        }
    }
    let key = |id: i64, b: [f32; 4]| (id, [b[0] as i32, b[1] as i32, b[2] as i32, b[3] as i32]);
    let mut vecs: HashMap<(i64, [i32; 4]), Vec<f32>> = HashMap::new();
    let mut thumbs: Vec<((i64, [i32; 4]), RgbImage)> = vec![];
    let mut ms = 0.0f64;
    let mut n_crops = 0usize;
    for (id, bs) in &need {
        let img = image::open(&files[id])?.to_rgb8();
        for &b in bs {
            let k = key(*id, b);
            if vecs.contains_key(&k) {
                continue;
            }
            let c = crop(&img, b);
            let t = std::time::Instant::now();
            let v = emb.embed(&c)?;
            ms += t.elapsed().as_secs_f64() * 1e3;
            n_crops += 1;
            if thumbs.len() < 4000 {
                thumbs.push((k, imageops::resize(&c, 96, 192, imageops::FilterType::Triangle)));
            }
            vecs.insert(k, v);
        }
    }
    println!("{n_crops} crops embedded, {:.2} ms per crop", ms / n_crops.max(1) as f64);

    // one-step re-identification
    let mut per_band: Vec<(usize, usize, f64, f64, usize, usize)> = vec![(0, 0, 0.0, 0.0, 0, 0); BANDS.len()]; // pairs, rank-1 hits, Σcos pos, Σcos neg, n neg, neg above pos
    for (x, y, others) in &pairs {
        let vx = &vecs[&key(x.id, x.b)];
        let vy = &vecs[&key(y.id, y.b)];
        let pos = cos(vx, vy);
        let band = BANDS.iter().position(|(top, _)| x.b[3] < *top).unwrap();
        let s = &mut per_band[band];
        s.0 += 1;
        s.2 += pos as f64;
        let mut hit = true;
        for o in others {
            let neg = cos(vx, &vecs[&key(o.id, o.b)]);
            s.3 += neg as f64;
            s.4 += 1;
            if neg >= pos {
                hit = false;
                s.5 += 1;
            }
        }
        s.1 += hit as usize;
    }
    println!("\n{:<10}{:>7}{:>9}{:>12}{:>12}{:>16}", "height px", "pairs", "rank-1", "cos same", "cos other", "other ≥ same");
    let (mut tp, mut th) = (0, 0);
    for (i, (n, hits, sp, sn, nn, above)) in per_band.iter().enumerate() {
        if *n > 0 {
            println!("{:<10}{n:>7}{:>8.1}%{:>12.3}{:>12.3}{:>15.1}%", BANDS[i].1, 100.0 * *hits as f64 / *n as f64, sp / *n as f64, sn / (*nn).max(1) as f64, 100.0 * *above as f64 / (*nn).max(1) as f64);
            tp += n;
            th += hits;
        }
    }
    println!("{:<10}{tp:>7}{:>8.1}%", "all", 100.0 * th as f64 / tp.max(1) as f64);

    // search by description
    if let Ok(tp) = std::env::var("TEXT") {
        let t: Value = serde_json::from_slice(&std::fs::read(&tp)?)?;
        let (scale, bias) = (t["logit_scale"].as_f64().unwrap() as f32, t["logit_bias"].as_f64().unwrap() as f32);
        println!("\nsearch by description over {} crops (top 12 per phrase; sheet per phrase):", thumbs.len());
        for (phrase, v) in t["phrases"].as_object().unwrap() {
            let tv: Vec<f32> = v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let mut scored: Vec<(f32, usize)> = thumbs.iter().enumerate().map(|(i, (k, _))| (cos(&vecs[k], &tv), i)).collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            let top: Vec<_> = scored.iter().take(12).collect();
            let probs: Vec<String> = top.iter().map(|(c, _)| format!("{:.2}", 1.0 / (1.0 + (-(scale * c + bias)).exp()))).collect();
            let mut sheet = RgbImage::from_pixel(12 * 96, 192, Rgb([0, 0, 0]));
            for (i, (_, ti)) in top.iter().enumerate() {
                imageops::overlay(&mut sheet, &thumbs[*ti].1, (i * 96) as i64, 0);
            }
            let fname = format!("text_{}.jpg", phrase.replace(' ', "_"));
            sheet.save(out_dir.join(&fname))?;
            println!("  {phrase:<18} p(top12) {}", probs.join(" "));
        }
    }
    Ok(())
}
