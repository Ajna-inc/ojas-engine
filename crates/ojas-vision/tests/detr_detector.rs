//! `Detector` on a DETR export (D-FINE-S COCO, `logits [N,300,80]` + `boxes [N,300,4]`,
//! dynamic batch): its boxes are the survey's raw-executor decoding (`examples/person_survey.rs`
//! `Detr`: frame stretched to the square, 0–1 RGB, sigmoid of the query's top logit, no NMS)
//! on gold frames, a batch is the frames one by one, and the CUDA backend agrees with the CPU.
//!
//! Needs `models/dfine-s-coco-dyn.onnx` (`OJAS_MODELS`) and the gold frames:
//! `OJAS_GOLD_FRAMES` is a review manifest (`[{"file": …}, …]`) or a directory of
//! JPEGs. Skips without them.

use std::collections::HashMap;
use std::path::PathBuf;

use ojas_vision::{Detection, Device, DetectorCfg, Frame, Runtime, RuntimeCfg};

const MODEL: &str = "dfine-s-coco-dyn.onnx";
const CONF: f32 = 0.25;

fn model() -> Option<PathBuf> {
    let dir = std::env::var("OJAS_MODELS").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"));
    let p = dir.join(MODEL);
    p.is_file().then_some(p)
}

/// The first three distinct frames of the gold set.
fn gold_frames() -> Option<Vec<(usize, usize, Vec<u8>)>> {
    let src = PathBuf::from(std::env::var("OJAS_GOLD_FRAMES").ok()?);
    let mut files: Vec<PathBuf> = if src.is_dir() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&src).ok()?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "jpg")).collect();
        v.sort();
        v
    } else if src.is_file() {
        let text = std::fs::read_to_string(&src).ok()?;
        let entries: Vec<serde_json::Value> = serde_json::from_str(&text).ok()?;
        let mut v: Vec<PathBuf> = vec![];
        for e in entries {
            if let Some(f) = e["file"].as_str().map(PathBuf::from) {
                if !v.contains(&f) {
                    v.push(f);
                }
            }
        }
        v
    } else {
        return None;
    };
    files.truncate(3);
    if files.len() < 3 {
        return None;
    }
    files.iter().map(|f| ojas_cpu::vit_preprocess::decode_rgb8_path(f.to_str()?).ok().map(|(w, h, rgb)| (w as usize, h as usize, rgb))).collect()
}

/// person_survey's `Detr::run` on the CPU executors: every class, score ≥ `conf`, clipped
/// to the frame, best first.
fn survey_decode(path: &str, frames: &[(usize, usize, Vec<u8>)], conf: f32) -> Vec<Vec<Detection>> {
    let model = ojas_formats::onnx::load(path).unwrap();
    let mut g = ojas_vision::import(&model, &HashMap::from([("batch".to_string(), 1usize)])).unwrap();
    ojas_vision::passes::optimize(&mut g);
    assert_eq!(g.outputs.len(), 2, "a DETR export");
    let s = g.shape(g.inputs[0])[2];
    let c = g.shape(g.outputs[0])[2];
    let mut exec = ojas_vision::exec_cpu::CpuExecutor::new(&g, 4);
    let mut out = vec![];
    for (w, h, rgb) in frames {
        let (w, h) = (*w, *h);
        let small = ojas_vision::pre::resize_bilinear_rgb8(rgb, w, h, s, s, 3);
        let mut input = vec![0.0f32; 3 * s * s];
        for ch in 0..3 {
            for i in 0..s * s {
                input[ch * s * s + i] = small[i * 3 + ch] as f32 / 255.0;
            }
        }
        let outs = exec.run(&g, &[&input]).unwrap();
        let (logits, boxes) = (&outs[0], &outs[1]);
        let mut dets = vec![];
        for q in 0..boxes.len() / 4 {
            let (mut class, mut best) = (0usize, f32::NEG_INFINITY);
            for k in 0..c {
                if logits[q * c + k] > best {
                    (class, best) = (k, logits[q * c + k]);
                }
            }
            let score = 1.0 / (1.0 + (-best).exp());
            if score < conf {
                continue;
            }
            let b = &boxes[q * 4..q * 4 + 4];
            let (cx, cy, bw, bh) = (b[0] * w as f32, b[1] * h as f32, b[2] * w as f32, b[3] * h as f32);
            let (fw, fh) = (w as f32, h as f32);
            dets.push(Detection { class: class as u16, score, x0: (cx - bw / 2.0).clamp(0.0, fw), y0: (cy - bh / 2.0).clamp(0.0, fh), x1: (cx + bw / 2.0).clamp(0.0, fw), y1: (cy + bh / 2.0).clamp(0.0, fh), keypoints: None });
        }
        dets.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.push(dets);
    }
    out
}

fn assert_same(a: &[Detection], b: &[Detection], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: {} vs {} boxes", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.class, y.class, "{what}: box {i} class");
        for (n, (p, q)) in [(x.score, y.score), (x.x0, y.x0), (x.y0, y.y0), (x.x1, y.x1), (x.y1, y.y1)].into_iter().enumerate() {
            assert!((p - q).abs() <= tol, "{what}: box {i} field {n}: {p} vs {q}");
        }
    }
}

#[test]
fn detector_decodes_the_detr_export_like_the_survey() {
    let (Some(model), Some(frames)) = (model(), gold_frames()) else {
        eprintln!("skipped: no {MODEL} / gold frames");
        return;
    };
    let path = model.to_str().unwrap();
    let want = survey_decode(path, &frames, CONF);
    assert!(want.iter().map(Vec::len).sum::<usize>() >= 3, "the gold frames hold detections");

    let rt = Runtime::new(RuntimeCfg { threads: Some(4), device: Device::Cpu }).unwrap();
    let mut det = rt.detector(path, DetectorCfg { conf: CONF, ..Default::default() }).unwrap();
    assert_eq!((det.head(), det.classes(), det.input_size(), det.backend()), ("detr", 80, 640, "cpu"));
    let fs: Vec<Frame> = frames.iter().map(|(w, h, rgb)| Frame::Rgb8 { w: *w, h: *h, data: rgb }).collect();
    // one call for the three frames: the same boxes as the survey's decode ...
    let got = det.run(&fs).unwrap();
    assert_eq!(got.len(), 3);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_same(g, w, 1e-3, &format!("frame {i}"));
    }
    // ... and as each frame alone
    for (i, f) in fs.iter().enumerate() {
        assert_same(&det.run(&[*f]).unwrap()[0], &got[i], 0.0, &format!("frame {i} alone"));
    }
    // the threshold is on the sigmoid score, `max_det` caps a frame, a class filter keeps the
    // query's top class only
    let mut strict = rt.detector(path, DetectorCfg { conf: 0.5, max_det: 2, classes: Some(vec![0, 2]), ..Default::default() }).unwrap();
    for (i, f) in fs.iter().enumerate() {
        let d = strict.run(&[*f]).unwrap().remove(0);
        let mut w: Vec<Detection> = want[i].iter().copied().filter(|d| d.score >= 0.5 && (d.class == 0 || d.class == 2)).collect();
        w.truncate(2);
        assert_same(&d, &w, 1e-3, &format!("frame {i} filtered"));
    }
    eprintln!("{} boxes on 3 gold frames match the survey decode", got.iter().map(Vec::len).sum::<usize>());
}

/// The CUDA plan (fp16, batch buckets 1/4/8, the crop kernel's stretch) finds the CPU's
/// boxes and nothing else: of the boxes with a margin over the threshold on either side, all
/// but a few small ones have a twin of the same class on the other side within 3 px, and the
/// twins' scores agree to 0.15 (fp16 moves a 24-px motorcycle from 0.47 to 0.36; the median
/// deviation is ~0.005). The batch of three is the frames one by one.
#[cfg(feature = "cuda")]
#[test]
fn cuda_detector_agrees_with_the_cpu() {
    let (Some(model), Some(frames)) = (model(), gold_frames()) else {
        eprintln!("skipped: no {MODEL} / gold frames");
        return;
    };
    if Runtime::probe().gpus.is_empty() {
        eprintln!("skipped: no CUDA GPU");
        return;
    }
    let path = model.to_str().unwrap();
    let cfg = || DetectorCfg { conf: CONF, ..Default::default() };
    let mut cpu = Runtime::new(RuntimeCfg { threads: Some(4), device: Device::Cpu }).unwrap().detector(path, cfg()).unwrap();
    let mut gpu = Runtime::new(RuntimeCfg { threads: Some(4), device: Device::Cuda(0) }).unwrap().detector(path, cfg()).unwrap();
    assert_eq!((gpu.head(), gpu.backend(), gpu.classes()), ("detr", "cuda", 80));
    let fs: Vec<Frame> = frames.iter().map(|(w, h, rgb)| Frame::Rgb8 { w: *w, h: *h, data: rgb }).collect();
    let (a, b) = (cpu.run(&fs).unwrap(), gpu.run(&fs).unwrap());
    // of the same-class boxes within 3 px, the one closest in score (a DETR head can put a
    // weak duplicate query right on a strong box): (score deviation, edge deviation)
    let twin = |d: &Detection, among: &[Detection]| -> Option<(f32, f32)> {
        among
            .iter()
            .filter(|o| o.class == d.class)
            .map(|o| ((o.score - d.score).abs(), [o.x0 - d.x0, o.y0 - d.y0, o.x1 - d.x1, o.y1 - d.y1].iter().fold(0.0f32, |m, v| m.max(v.abs()))))
            .filter(|(_, px)| *px <= 3.0)
            .min_by(|x, y| x.0.total_cmp(&y.0))
    };
    let sure = |v: &[Detection]| v.iter().filter(|d| d.score >= CONF + 0.1).copied().collect::<Vec<_>>();
    let (mut devs, mut worst_px, mut lost) = (vec![], 0.0f32, vec![]);
    for (i, (ca, cb)) in a.iter().zip(&b).enumerate() {
        eprintln!("frame {i}: {} boxes on the CPU, {} on CUDA (≥ {:.2}: {} / {})", ca.len(), cb.len(), CONF + 0.1, sure(ca).len(), sure(cb).len());
        for (side, mine, theirs) in [("CPU", ca, cb), ("CUDA", cb, ca)] {
            for d in sure(mine) {
                match twin(&d, theirs) {
                    Some((s, px)) => {
                        assert!(s <= 0.15, "frame {i}: {side} box {d:?} scores {s:.3} away from its twin");
                        devs.push(s);
                        worst_px = worst_px.max(px);
                    }
                    None => lost.push(format!("frame {i}: {side} box {d:?} has no twin")),
                }
            }
        }
    }
    let n = devs.len();
    assert!(n >= 100, "{n} twins");
    assert!(lost.len() * 100 <= n * 3 && lost.len() <= 6, "{} of {n} sure boxes have no twin:\n{}", lost.len(), lost.join("\n"));
    devs.sort_by(|x, y| x.total_cmp(y));
    assert!(devs[n / 2] <= 0.02, "median score deviation {}", devs[n / 2]);
    // the batch of three is the frames one by one (each through the batch-1 plan)
    for (i, f) in fs.iter().enumerate() {
        let alone = gpu.run(&[*f]).unwrap().remove(0);
        let missing = sure(&b[i]).iter().filter(|d| twin(d, &alone).is_none()).count();
        assert!(missing <= 1, "frame {i}: {missing} boxes of the batch are missing alone");
    }
    eprintln!("{n} twins across CPU and CUDA ({} without): score deviation median {:.4}, max {:.4}; worst edge {worst_px:.2} px", lost.len(), devs[n / 2], devs[n - 1]);
    for l in &lost {
        eprintln!("  {l}");
    }
}
