//! How many people a detector sees on recorded footage, and how big they are — the
//! numbers that decide which person lanes (pose, gait, face) a camera can support. A
//! face needs ~40 px, so a standing person of ~280 px; 2-D pose is usable from
//! ~100 px; appearance ReID from ~60 px.
//!
//! Runs one detector over a COCO image list, keeps one class (COCO `person` = 0 by default),
//! writes plain COCO detections (`image_id, category_id 1, bbox xywh, score`) so the review
//! tools can compare two detectors, and prints the per-camera counts and the height histogram.
//!
//! A dense-head export (YOLO) goes through `Detector`; a DETR export (`logits [1,Q,C]` +
//! `boxes [1,Q,4]`, e.g. D-FINE COCO) through the raw executors with the benchmark's
//! preprocessing (stretch to the input square, 0–1 RGB) and sigmoid scores.
//!
//! `person_survey <frames.json> <model.onnx> <out.dets.json> [conf 0.25] [limit]`
//! Env: `OJAS_DEVICE=cuda:0|vulkan:0|cpu`, `CLASS=0`, `DECODERS=8`.
use std::collections::{BTreeMap, HashMap};
use std::sync::{mpsc, Arc};

use ojas_vision::{Detection, Device, DetectorCfg, Frame, Runtime, RuntimeCfg};
use serde_json::{json, Value};

/// A DETR-family detector on the raw executors: one frame per call.
struct Detr {
    g: ojas_vision::ir::Graph,
    size: usize,
    classes: usize,
    cpu: Option<ojas_vision::exec_cpu::CpuExecutor>,
    #[cfg(feature = "cuda")]
    gpu: Option<ojas_vision::exec_gpu::CudaExecutor>,
    input: Vec<f32>,
}

impl Detr {
    fn load(path: &str, device: Device) -> anyhow::Result<Self> {
        let model = ojas_formats::onnx::load(path)?;
        let mut g = ojas_vision::import(&model, &HashMap::new())?;
        ojas_vision::passes::optimize(&mut g);
        anyhow::ensure!(g.outputs.len() == 2, "not a DETR export");
        let size = g.shape(g.inputs[0])[2];
        let classes = g.shape(g.outputs[0])[2];
        let mut d = Detr { size, classes, cpu: None, input: vec![0.0; 3 * size * size], g, #[cfg(feature = "cuda")] gpu: None };
        match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(o) => {
                ojas_vision::passes::lower_for_gpu(&mut d.g);
                d.gpu = Some(ojas_vision::exec_gpu::CudaExecutor::new(&d.g, o)?);
            }
            _ => d.cpu = Some(ojas_vision::exec_cpu::CpuExecutor::new(&d.g, std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))),
        }
        Ok(d)
    }

    fn run(&mut self, w: usize, h: usize, rgb: &[u8], class: u16, conf: f32) -> anyhow::Result<Vec<Detection>> {
        let s = self.size;
        let small = ojas_vision::pre::resize_bilinear_rgb8(rgb, w, h, s, s, 3);
        for c in 0..3 {
            for i in 0..s * s {
                self.input[c * s * s + i] = small[i * 3 + c] as f32 / 255.0;
            }
        }
        #[cfg(feature = "cuda")]
        let outs = match self.gpu.as_mut() {
            Some(ex) => ex.run(&self.input)?,
            None => self.cpu.as_mut().unwrap().run(&self.g, &[&self.input])?,
        };
        #[cfg(not(feature = "cuda"))]
        let outs = self.cpu.as_mut().unwrap().run(&self.g, &[&self.input])?;
        let (logits, boxes) = (&outs[0], &outs[1]);
        let c = self.classes;
        let mut dets = vec![];
        for q in 0..boxes.len() / 4 {
            let score = 1.0 / (1.0 + (-logits[q * c + class as usize]).exp());
            if score < conf {
                continue;
            }
            // the query's top class must be ours (one hypothesis per query, as the benchmark scores it)
            if (0..c).any(|k| k != class as usize && logits[q * c + k] > logits[q * c + class as usize]) {
                continue;
            }
            let b = &boxes[q * 4..q * 4 + 4];
            let (cx, cy, bw, bh) = (b[0] * w as f32, b[1] * h as f32, b[2] * w as f32, b[3] * h as f32);
            dets.push(Detection { class, score, x0: cx - bw / 2.0, y0: cy - bh / 2.0, x1: cx + bw / 2.0, y1: cy + bh / 2.0, keypoints: None });
        }
        Ok(dets)
    }
}

const BINS: [(usize, &str); 7] = [(30, "<30"), (60, "30-60"), (100, "60-100"), (150, "100-150"), (200, "150-200"), (280, "200-280"), (usize::MAX, "≥280")];

fn bin(h: usize) -> usize {
    BINS.iter().position(|(top, _)| h < *top).unwrap()
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 4, "person_survey <frames.json> <model.onnx> <out.dets.json> [conf] [limit]");
    let conf: f32 = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(0.25);
    let limit: usize = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let class: u16 = std::env::var("CLASS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let decoders: usize = std::env::var("DECODERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let device = match std::env::var("OJAS_DEVICE").unwrap_or_else(|_| "cpu".into()) {
        d if d.starts_with("cuda") => Device::Cuda(d.split_once(':').and_then(|(_, i)| i.parse().ok()).unwrap_or(0)),
        d if d.starts_with("vulkan") => Device::Vulkan(d.split_once(':').and_then(|(_, i)| i.parse().ok()).unwrap_or(0)),
        _ => Device::Cpu,
    };
    let frames: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let images: Vec<(i64, String, String)> = frames["images"]
        .as_array()
        .unwrap()
        .iter()
        .take(limit)
        .map(|im| (im["id"].as_i64().unwrap(), im["file_name"].as_str().unwrap().to_string(), im["camera"].as_str().unwrap_or("?").to_string()))
        .collect();

    let rt = Runtime::new(RuntimeCfg { threads: None, device })?;
    let mut detr = Detr::load(&a[2], device).ok();
    let mut det = match detr {
        Some(_) => None,
        None => Some(rt.detector(&a[2], DetectorCfg { conf, classes: Some(vec![class]), ..Default::default() })?),
    };
    println!("{} frames, {} ({}) on {device:?}, class {class}, conf {conf}", images.len(), a[2].rsplit('/').next().unwrap_or(""), if detr.is_some() { "DETR" } else { "dense head" });

    // JPEG decode in worker threads, in order: worker k decodes frames k, k+n, k+2n, ...
    let images = Arc::new(images);
    let mut rxs = vec![];
    for k in 0..decoders {
        let (tx, rx) = mpsc::sync_channel::<(usize, usize, Vec<u8>)>(4);
        rxs.push(rx);
        let images = images.clone();
        std::thread::spawn(move || {
            for i in (k..images.len()).step_by(decoders) {
                let (w, h, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(&images[i].1).expect("decode");
                if tx.send((w, h, rgb)).is_err() {
                    break;
                }
            }
        });
    }

    let t0 = std::time::Instant::now();
    let mut dets_out: Vec<Value> = vec![];
    let mut per_cam: BTreeMap<String, (usize, usize, [usize; 7])> = BTreeMap::new(); // frames, persons, height bins
    let mut per_frame: Vec<usize> = vec![];
    let mut hist = [0usize; 7];
    let batch = 8;
    let mut i = 0usize;
    while i < images.len() {
        let n = batch.min(images.len() - i);
        let mut bufs = Vec::with_capacity(n);
        for j in 0..n {
            bufs.push(rxs[(i + j) % decoders].recv()?);
        }
        let results: Vec<Vec<Detection>> = match (&mut det, &mut detr) {
            (Some(det), _) => {
                let frames: Vec<Frame> = bufs.iter().map(|(w, h, rgb)| Frame::Rgb8 { w: *w, h: *h, data: rgb }).collect();
                det.run(&frames)?
            }
            (None, Some(dt)) => bufs.iter().map(|(w, h, rgb)| dt.run(*w, *h, rgb, class, conf)).collect::<anyhow::Result<_>>()?,
            _ => unreachable!(),
        };
        for (j, dets) in results.into_iter().enumerate() {
            let (id, _, cam) = &images[i + j];
            let e = per_cam.entry(cam.clone()).or_default();
            e.0 += 1;
            e.1 += dets.len();
            per_frame.push(dets.len());
            for d in dets {
                let h = (d.y1 - d.y0).max(0.0) as usize;
                hist[bin(h)] += 1;
                e.2[bin(h)] += 1;
                dets_out.push(json!({"image_id": id, "category_id": 1, "bbox": [d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0], "score": d.score}));
            }
        }
        i += n;
        if i % 2000 < n {
            eprintln!("  {i}/{} ({:.0} frames/s)", images.len(), i as f64 / t0.elapsed().as_secs_f64());
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    std::fs::write(&a[3], serde_json::to_string(&dets_out)?)?;

    per_frame.sort_unstable();
    let total: usize = per_frame.iter().sum();
    let pct = |p: f64| per_frame[((per_frame.len() - 1) as f64 * p) as usize];
    println!("\n{} people in {} frames in {secs:.0} s ({:.0} frames/s): {:.2} per frame, p50 {}, p90 {}, max {}; {:.1} % of frames have nobody", total, per_frame.len(), per_frame.len() as f64 / secs, total as f64 / per_frame.len() as f64, pct(0.5), pct(0.9), per_frame.last().unwrap_or(&0), 100.0 * per_frame.iter().filter(|&&n| n == 0).count() as f64 / per_frame.len() as f64);
    println!("\nbox height (px)   {}", BINS.iter().map(|(_, n)| format!("{n:>8}")).collect::<String>());
    println!("all               {}", hist.iter().map(|c| format!("{c:>8}")).collect::<String>());
    println!("share             {}", hist.iter().map(|c| format!("{:>7.1}%", 100.0 * *c as f64 / total.max(1) as f64)).collect::<String>());
    let usable = |h: &[usize; 7], from: usize| 100.0 * h[from..].iter().sum::<usize>() as f64 / h.iter().sum::<usize>().max(1) as f64;
    println!("\nusable for:  ReID (≥60 px) {:.1} %   pose/gait (≥100 px) {:.1} %   face (≥280 px) {:.1} %", usable(&hist, 2), usable(&hist, 3), usable(&hist, 6));
    println!("\n{:<20}{:>8}{:>9}{:>10}{:>10}{:>10}{:>10}", "camera", "frames", "people", "per frame", "≥60 px", "≥100 px", "≥280 px");
    for (cam, (f, p, h)) in &per_cam {
        println!("{cam:<20}{f:>8}{p:>9}{:>10.2}{:>9.1}%{:>9.1}%{:>9.1}%", *p as f64 / *f as f64, usable(h, 2), usable(h, 3), usable(h, 6));
    }
    println!("\n→ {}", a[3]);
    Ok(())
}
