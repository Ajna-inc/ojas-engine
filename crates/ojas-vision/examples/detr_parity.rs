//! Compares the detections the engine decodes from a DETR export against
//! onnxruntime's — the gate for a new D-FINE / DEIM model.
//!
//! Runs the export over the first `limit` frames of a COCO image list with the engine's own
//! preprocessing (stretch to the input square, 0–1 RGB — `person_survey`'s DETR path), and writes
//! per frame `<work>/<id>.in.f32` (the preprocessed input, once) and `<work>/<id>.<device>.{0,1}.f32`
//! (raw logits and boxes). When `<work>/<id>.ref.{0,1}.f32` exist (onnxruntime on the same
//! `.in.f32`: `training/deim/export_onnx.py ref --onnx … --dir <work>`) it compares: the raw max
//! |Δ| per output, and the decoded detections (top class per query, sigmoid score ≥ `conf`, no
//! NMS): fraction of the reference boxes matched by ours at IoU ≥ 0.9 with the same class, the
//! reverse fraction, the worst IoU among the matched pairs, the largest score and box-corner
//! deviation among them. Also writes `<work>/<device>.dets.json` (COCO: `category_id` = class + 1).
//!
//! `detr_parity <frames.json> <model.onnx> <work_dir> [conf 0.4] [limit 20]`
//! Env: `OJAS_DEVICE=cpu|cuda:0` (CUDA needs `--features cuda`), `batch` is bound to 1.
use std::collections::HashMap;

use serde_json::{json, Value};

struct Det {
    class: usize,
    score: f32,
    /// normalised cxcywh, as the model emits it
    b: [f32; 4],
}

struct Detr {
    g: ojas_vision::ir::Graph,
    size: usize,
    classes: usize,
    cpu: Option<ojas_vision::exec_cpu::CpuExecutor>,
    #[cfg(feature = "cuda")]
    gpu: Option<ojas_vision::exec_gpu::CudaExecutor>,
}

impl Detr {
    fn load(path: &str, device: &str) -> anyhow::Result<Self> {
        let model = ojas_formats::onnx::load(path)?;
        let mut binds = HashMap::new();
        binds.insert("batch".to_string(), 1usize);
        let mut g = ojas_vision::import(&model, &binds)?;
        ojas_vision::passes::optimize(&mut g);
        anyhow::ensure!(g.outputs.len() == 2 && g.shape(g.outputs[1]).last() == Some(&4), "not a DETR export (logits + boxes)");
        let size = g.shape(g.inputs[0])[2];
        let classes = g.shape(g.outputs[0])[2];
        let mut d = Detr { size, classes, cpu: None, g, #[cfg(feature = "cuda")] gpu: None };
        match device {
            #[cfg(feature = "cuda")]
            d_ if d_.starts_with("cuda") => {
                let ordinal = d_.split_once(':').and_then(|(_, i)| i.parse().ok()).unwrap_or(0);
                ojas_vision::passes::lower_for_gpu(&mut d.g);
                d.gpu = Some(ojas_vision::exec_gpu::CudaExecutor::new(&d.g, ordinal)?);
            }
            "cpu" => d.cpu = Some(ojas_vision::exec_cpu::CpuExecutor::new(&d.g, std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))),
            other => anyhow::bail!("OJAS_DEVICE {other:?}: cpu or cuda:N (with --features cuda)"),
        }
        Ok(d)
    }

    fn preprocess(&self, w: usize, h: usize, rgb: &[u8]) -> Vec<f32> {
        let s = self.size;
        let small = ojas_vision::pre::resize_bilinear_rgb8(rgb, w, h, s, s, 3);
        let mut x = vec![0.0f32; 3 * s * s];
        for c in 0..3 {
            for i in 0..s * s {
                x[c * s * s + i] = small[i * 3 + c] as f32 / 255.0;
            }
        }
        x
    }

    fn run(&mut self, input: &[f32]) -> anyhow::Result<Vec<Vec<f32>>> {
        #[cfg(feature = "cuda")]
        if let Some(ex) = self.gpu.as_mut() {
            return ex.run(input);
        }
        self.cpu.as_mut().unwrap().run(&self.g, &[input])
    }
}

/// Top class per query, sigmoid score, keep ≥ conf (no NMS) — what `person.rs`'s DetrHead does.
fn decode(logits: &[f32], boxes: &[f32], classes: usize, conf: f32) -> Vec<Det> {
    let mut out = vec![];
    for q in 0..boxes.len() / 4 {
        let row = &logits[q * classes..(q + 1) * classes];
        let (class, &best) = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        let score = 1.0 / (1.0 + (-best).exp());
        if score >= conf {
            out.push(Det { class, score, b: [boxes[q * 4], boxes[q * 4 + 1], boxes[q * 4 + 2], boxes[q * 4 + 3]] });
        }
    }
    out
}

fn corners(b: &[f32; 4]) -> [f32; 4] {
    [b[0] - b[2] / 2.0, b[1] - b[3] / 2.0, b[0] + b[2] / 2.0, b[1] + b[3] / 2.0]
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (a, b) = (corners(a), corners(b));
    let i = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i).max(1e-9)
}

fn read_f32(path: &str) -> Option<Vec<f32>> {
    let b = std::fs::read(path).ok()?;
    Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn write_f32(path: &str, v: &[f32]) -> std::io::Result<()> {
    std::fs::write(path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}

/// One-directional match: for each `from` box the best same-class IoU in `to`.
/// Returns (matched at ≥ 0.9, worst IoU among matched, max |Δscore|, max corner Δ in px).
fn agree(from: &[Det], to: &[Det], w: f32, h: f32) -> (usize, f32, f32, f32) {
    let (mut n, mut worst, mut ds, mut db) = (0usize, 1.0f32, 0.0f32, 0.0f32);
    for r in from {
        if let Some((i, o)) = to.iter().filter(|o| o.class == r.class).map(|o| (iou(&o.b, &r.b), o)).max_by(|a, b| a.0.total_cmp(&b.0)) {
            if i >= 0.9 {
                n += 1;
                worst = worst.min(i);
                ds = ds.max((o.score - r.score).abs());
                let (a, b) = (corners(&o.b), corners(&r.b));
                for k in 0..4 {
                    db = db.max((a[k] - b[k]).abs() * if k % 2 == 0 { w } else { h });
                }
            }
        }
    }
    (n, worst, ds, db)
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 4, "detr_parity <frames.json> <model.onnx> <work_dir> [conf 0.4] [limit 20]");
    let conf: f32 = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(0.4);
    let limit: usize = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(20);
    let device = std::env::var("OJAS_DEVICE").unwrap_or_else(|_| "cpu".into());
    let tag = device.split(':').next().unwrap().to_string();
    let work = &a[3];
    std::fs::create_dir_all(work)?;

    let frames: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let images: Vec<(i64, String)> = frames["images"].as_array().unwrap().iter().take(limit).map(|im| (im["id"].as_i64().unwrap(), im["file_name"].as_str().unwrap().to_string())).collect();
    let t = std::time::Instant::now();
    let mut detr = Detr::load(&a[2], &device)?;
    println!("{} on {device}: input {}, {} classes, loaded in {:.1} s", a[2].rsplit('/').next().unwrap_or(""), detr.size, detr.classes, t.elapsed().as_secs_f64());

    let (mut n_ref, mut n_ours, mut m_fwd, mut m_rev) = (0usize, 0usize, 0usize, 0usize);
    let (mut worst_iou, mut max_ds, mut max_db, mut raw) = (1.0f32, 0.0f32, 0.0f32, [0.0f32; 2]);
    let mut compared = 0usize;
    let mut dets_out: Vec<Value> = vec![];
    let mut ms = 0.0f64;
    for (id, path) in &images {
        let (w, h, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(path)?;
        let in_path = format!("{work}/{id}.in.f32");
        let input = match read_f32(&in_path) {
            Some(x) if x.len() == 3 * detr.size * detr.size => x,
            _ => {
                let x = detr.preprocess(w, h, &rgb);
                write_f32(&in_path, &x)?;
                x
            }
        };
        let t = std::time::Instant::now();
        let outs = detr.run(&input)?;
        ms += t.elapsed().as_secs_f64() * 1e3;
        for (k, o) in outs.iter().enumerate() {
            write_f32(&format!("{work}/{id}.{tag}.{k}.f32"), o)?;
        }
        let ours = decode(&outs[0], &outs[1], detr.classes, conf);
        for d in &ours {
            let c = corners(&d.b);
            dets_out.push(json!({"image_id": id, "category_id": d.class + 1, "bbox": [c[0] * w as f32, c[1] * h as f32, (c[2] - c[0]) * w as f32, (c[3] - c[1]) * h as f32], "score": d.score}));
        }
        n_ours += ours.len();
        if let (Some(rl), Some(rb)) = (read_f32(&format!("{work}/{id}.ref.0.f32")), read_f32(&format!("{work}/{id}.ref.1.f32"))) {
            anyhow::ensure!(rl.len() == outs[0].len() && rb.len() == outs[1].len(), "reference shape differs for frame {id}");
            for (k, (o, r)) in [(&outs[0], &rl), (&outs[1], &rb)].into_iter().enumerate() {
                raw[k] = raw[k].max(o.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max));
            }
            let refs = decode(&rl, &rb, detr.classes, conf);
            let (f, wi, ds, db) = agree(&refs, &ours, w as f32, h as f32);
            let (r, ..) = agree(&ours, &refs, w as f32, h as f32);
            n_ref += refs.len();
            m_fwd += f;
            m_rev += r;
            worst_iou = worst_iou.min(wi);
            max_ds = max_ds.max(ds);
            max_db = max_db.max(db);
            compared += 1;
        }
    }
    std::fs::write(format!("{work}/{tag}.dets.json"), serde_json::to_string(&dets_out)?)?;
    println!("{} frames, {:.1} ms/frame on {device}, {n_ours} detections ≥ {conf} → {work}/{tag}.dets.json", images.len(), ms / images.len() as f64);
    if compared == 0 {
        println!("no reference outputs in {work} — run `export_onnx.py ref --onnx <model> --dir {work}` and rerun");
        return Ok(());
    }
    println!("vs onnxruntime on {compared} frames: raw max |Δ| logits {:.2e}, boxes {:.2e}", raw[0], raw[1]);
    println!(
        "  detections ≥ {conf}: reference {n_ref}, ours {n_ours}; reference matched by ours (same class, IoU ≥ 0.9) {m_fwd}/{n_ref} = {:.4}, ours matched by reference {m_rev}/{n_ours} = {:.4}",
        m_fwd as f64 / n_ref.max(1) as f64,
        m_rev as f64 / n_ours.max(1) as f64
    );
    println!("  among matched pairs: worst IoU {worst_iou:.4}, max |Δscore| {max_ds:.4}, max box-corner Δ {max_db:.2} px");
    Ok(())
}
