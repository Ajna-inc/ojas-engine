//! Vision commands: `ojas detect`, `ojas plate`, `ojas vbench`. CPU backend.

use anyhow::{bail, ensure, Context, Result};
use ojas_vision::{DetectorCfg, Frame, OcrCfg, OcrNorm, Runtime, RuntimeCfg};

use crate::flags::RunOpts;

/// COCO class names, printed when the model has exactly 80 classes.
pub(crate) const COCO: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
];

pub(crate) fn class_name(names: Option<&[String]>, nc: usize, id: u16) -> String {
    if let Some(n) = names.and_then(|n| n.get(id as usize)) {
        return n.clone();
    }
    if nc == 80 {
        COCO.get(id as usize).map(|s| s.to_string()).unwrap_or_else(|| id.to_string())
    } else {
        id.to_string()
    }
}

fn detector_cfg(opts: &RunOpts) -> DetectorCfg {
    DetectorCfg {
        conf: opts.conf.unwrap_or(0.25),
        iou: opts.iou.unwrap_or(0.45),
        ..Default::default()
    }
}

fn image_paths(path: &str) -> Result<Vec<String>> {
    let meta = std::fs::metadata(path).with_context(|| format!("opening {path}"))?;
    if meta.is_file() {
        return Ok(vec![path.to_string()]);
    }
    let mut v: Vec<String> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp" | "webp"))
        })
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    v.sort();
    ensure!(!v.is_empty(), "{path}: no images found");
    Ok(v)
}

/// `ojas detect model.onnx image|dir [--conf] [--iou] [--json] [--device cpu]`
pub fn detect(model: &str, input: &str, opts: &RunOpts) -> Result<()> {
    let rt = Runtime::new(RuntimeCfg::default())?;
    let mut det = rt.detector(model, detector_cfg(opts))?;
    let nc = det.classes();
    let names = det.class_names().map(|n| n.to_vec());
    let names = names.as_deref();
    let paths = image_paths(input)?;
    let mut first = true;
    if opts.json {
        println!("[");
    }
    for p in &paths {
        let (w, h, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(p)?;
        let t0 = std::time::Instant::now();
        let dets = det.run(&[Frame::Rgb8 { w, h, data: &rgb }])?.remove(0);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if opts.json {
            if !first {
                println!(",");
            }
            first = false;
            let boxes: Vec<String> = dets
                .iter()
                .map(|d| {
                    format!(
                        "{{\"class\":{},\"name\":{:?},\"score\":{:.4},\"box\":[{:.1},{:.1},{:.1},{:.1}]}}",
                        d.class,
                        class_name(names, nc, d.class),
                        d.score,
                        d.x0,
                        d.y0,
                        d.x1,
                        d.y1
                    )
                })
                .collect();
            print!("{{\"image\":{p:?},\"ms\":{ms:.1},\"detections\":[{}]}}", boxes.join(","));
        } else {
            println!("{p}  ({w}x{h}, {ms:.1} ms, {} detections)", dets.len());
            for d in &dets {
                println!(
                    "  {:<14} {:.3}  [{:>6.1}, {:>6.1}, {:>6.1}, {:>6.1}]",
                    class_name(names, nc, d.class),
                    d.score,
                    d.x0,
                    d.y0,
                    d.x1,
                    d.y1
                );
            }
        }
    }
    if opts.json {
        println!("\n]");
    }
    Ok(())
}

/// `ojas plate det.onnx rec.onnx dict.txt image|dir` — detect plates, crop,
/// read. The detector is a single-class plate model; crops get 8% padding.
pub fn plate(det_model: &str, rec_model: &str, dict: &str, input: &str, opts: &RunOpts) -> Result<()> {
    let rt = Runtime::new(RuntimeCfg::default())?;
    let mut det = rt.detector(det_model, DetectorCfg { conf: opts.conf.unwrap_or(0.3), ..detector_cfg(opts) })?;
    // --ocr-norm signed|unit with an optional ",bgr" suffix (e.g. "signed,bgr")
    let (norm_s, bgr) = match opts.ocr_norm.as_deref() {
        None => ("signed", false),
        Some(s) => match s.strip_suffix(",bgr") {
            Some(base) => (base, true),
            None => (s, false),
        },
    };
    let norm = match norm_s {
        "signed" | "" => OcrNorm::Signed,
        "unit" => OcrNorm::Unit,
        other => bail!("--ocr-norm {other:?}: want signed or unit, optionally with ,bgr"),
    };
    let mut ocr = rt.plate_ocr(rec_model, dict, OcrCfg { norm, bgr, ..Default::default() })?;
    for p in image_paths(input)? {
        let (w, h, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(&p)?;
        let dets = det.run(&[Frame::Rgb8 { w, h, data: &rgb }])?.remove(0);
        println!("{p}  ({} plates)", dets.len());
        for d in &dets {
            // crop with 8% padding
            let pw = (d.x1 - d.x0) * 0.08;
            let ph = (d.y1 - d.y0) * 0.08;
            let x0 = (d.x0 - pw).max(0.0) as usize;
            let y0 = (d.y0 - ph).max(0.0) as usize;
            let x1 = ((d.x1 + pw) as usize).min(w);
            let y1 = ((d.y1 + ph) as usize).min(h);
            if x1 <= x0 + 2 || y1 <= y0 + 2 {
                continue;
            }
            let (cw, chh) = (x1 - x0, y1 - y0);
            let mut crop = vec![0u8; cw * chh * 3];
            for y in 0..chh {
                let src = &rgb[((y0 + y) * w + x0) * 3..((y0 + y) * w + x1) * 3];
                crop[y * cw * 3..(y + 1) * cw * 3].copy_from_slice(src);
            }
            let read = ocr.run(&[Frame::Rgb8 { w: cw, h: chh, data: &crop }])?.remove(0);
            println!(
                "  [{:>6.1},{:>6.1},{:>6.1},{:>6.1}] conf {:.2} -> {:?} (mean {:.2})",
                d.x0, d.y0, d.x1, d.y1, d.score, read.text, read.mean_conf
            );
        }
    }
    Ok(())
}

/// `ojas vbench model.onnx [-r reps]` — forward-pass timing, batch 1.
pub fn vbench(model: &str, opts: &RunOpts) -> Result<()> {
    if !model.ends_with(".onnx") {
        bail!("vbench takes an .onnx model");
    }
    let iters = (opts.reps.max(1) * 10) as u32;
    for threads in [1usize, 0] {
        let rt = Runtime::new(RuntimeCfg { threads: (threads > 0).then_some(threads), ..Default::default() })?;
        let mut det = rt.detector(model, DetectorCfg::default())?;
        let label = if threads == 1 { "1 thread " } else { "all cores" };
        let r = det.bench_detailed(iters, opts.json)?;
        let (med, min) = (r.median.as_secs_f64() * 1e3, r.min.as_secs_f64() * 1e3);
        println!(
            "{model}  {label}  median {med:8.2} ms  min {min:8.2} ms   {:6.1} fps(min)   ({iters} iters)",
            1e3 / min
        );
        if opts.json {
            // --json doubles as the profile switch here: per-op breakdown
            for (op, secs, calls) in r.per_op.iter().take(12) {
                println!("    {op:<18} {:8.2} ms  x{calls}", secs * 1e3);
            }
        }
    }
    Ok(())
}
