//! Gate A: one `Model` API over every decoder.
//! - classify / embed on a tiny model built in memory, whose outputs are
//!   computed by hand here (no onnxruntime, no Python);
//! - the same tiny models on CUDA and Vulkan agree with the CPU;
//! - detect / OCR through `Model` are bit-identical to `Detector` / `PlateOcr` on
//!   the parity images (needs the model files).

use ojas_formats::onnx::{AttrValue, OnnxAttr, OnnxDim, OnnxGraph, OnnxModel, OnnxNode, OnnxTensor, OnnxValueInfo};
use ojas_vision::{Device, DetectorCfg, Frame, ModelKind, OcrCfg, OcrNorm, Output, Runtime, RuntimeCfg, TensorHead, TensorSpec};
use std::path::PathBuf;

const S: usize = 8; // model input side

/// x[N,3,S,S] → GlobalAveragePool → Flatten → Gemm(I₃) [→ Softmax]: the
/// output is the per-channel mean of the normalised input.
fn tiny(softmax: bool) -> OnnxModel {
    let vi = |name: &str, dims: Vec<OnnxDim>| OnnxValueInfo { name: name.into(), elem_type: 1, dims };
    let node = |op: &str, ins: &[&str], out: &str, attrs: Vec<OnnxAttr>| OnnxNode {
        name: out.into(),
        op_type: op.into(),
        domain: String::new(),
        inputs: ins.iter().map(|s| s.to_string()).collect(),
        outputs: vec![out.into()],
        attrs,
    };
    let mut nodes = vec![
        node("GlobalAveragePool", &["x"], "gap", vec![]),
        node("Flatten", &["gap"], "flat", vec![OnnxAttr { name: "axis".into(), value: AttrValue::I(1) }]),
        node("Gemm", &["flat", "w", "b"], if softmax { "logits" } else { "y" }, vec![]),
    ];
    if softmax {
        nodes.push(node("Softmax", &["logits"], "y", vec![OnnxAttr { name: "axis".into(), value: AttrValue::I(1) }]));
    }
    let eye = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    OnnxModel {
        ir_version: 8,
        producer_name: "model_api test".into(),
        opsets: vec![(String::new(), 17)],
        graph: OnnxGraph {
            name: "tiny".into(),
            nodes,
            initializers: vec![OnnxTensor::from_f32("w", &[3, 3], eye), OnnxTensor::from_f32("b", &[3], vec![0.0; 3])],
            inputs: vec![vi("x", vec![OnnxDim::Param("N".into()), OnnxDim::Value(3), OnnxDim::Value(S as i64), OnnxDim::Value(S as i64)])],
            outputs: vec![vi("y", vec![OnnxDim::Param("N".into()), OnnxDim::Value(3)])],
            value_info: vec![],
        },
    }
}

fn spec(head: TensorHead, norm: OcrNorm, bgr: bool) -> TensorSpec {
    TensorSpec { width: S, height: S, norm, bgr, head, ..Default::default() }
}

fn solid(w: usize, h: usize, rgb: [u8; 3]) -> Vec<u8> {
    rgb.iter().copied().cycle().take(w * h * 3).collect()
}

fn cpu() -> Runtime {
    Runtime::new(RuntimeCfg { threads: Some(2), device: Device::Cpu }).unwrap()
}

fn softmax(v: [f32; 3]) -> [f32; 3] {
    let e = v.map(|x| x.exp());
    let s: f32 = e.iter().sum();
    e.map(|x| x / s)
}

fn labels(o: &Output) -> Vec<(usize, f32)> {
    match o {
        Output::Labels(l) => l.clone(),
        other => panic!("not labels: {other:?}"),
    }
}

fn vector(o: &Output) -> Vec<f32> {
    match o {
        Output::Vector(v) => v.clone(),
        other => panic!("not a vector: {other:?}"),
    }
}

#[test]
fn classify_gives_top_k_probabilities() {
    let red = solid(20, 10, [255, 0, 0]);
    for in_graph in [false, true] {
        let mut m = cpu().tensor_model(&tiny(in_graph), spec(TensorHead::Classify { top_k: 2 }, OcrNorm::Unit, false)).unwrap();
        let out = m.run(&[Frame::Rgb8 { w: 20, h: 10, data: &red }]).unwrap();
        let l = labels(&out[0]);
        let p = softmax([1.0, 0.0, 0.0]);
        assert_eq!(l.len(), 2);
        assert_eq!(l[0].0, 0);
        assert!((l[0].1 - p[0]).abs() < 1e-5, "{l:?} vs {p:?} (softmax in graph: {in_graph})");
        // classes 1 and 2 tie: the lower id wins
        assert_eq!(l[1].0, 1);
        assert!((l[1].1 - p[1]).abs() < 1e-5);
    }
}

#[test]
fn bgr_and_signed_norm_reach_the_model() {
    let red = solid(8, 8, [255, 0, 0]);
    let f = [Frame::Rgb8 { w: 8, h: 8, data: &red }];
    // BGR planes: red lands in channel 2
    let mut m = cpu().tensor_model(&tiny(false), spec(TensorHead::Classify { top_k: 1 }, OcrNorm::Unit, true)).unwrap();
    assert_eq!(labels(&m.run(&f).unwrap()[0])[0].0, 2);
    // signed: R = 1, G = B = −1
    let mut m = cpu().tensor_model(&tiny(false), spec(TensorHead::Embed, OcrNorm::Signed, false)).unwrap();
    let v = vector(&m.run(&f).unwrap()[0]);
    let want = [1.0, -1.0, -1.0].map(|x: f32| x / 3f32.sqrt());
    for (a, b) in v.iter().zip(want) {
        assert!((a - b).abs() < 1e-5, "{v:?} vs {want:?}");
    }
}

#[test]
fn embed_is_l2_normalised() {
    let yellow = solid(16, 16, [255, 255, 0]);
    let mut m = cpu().tensor_model(&tiny(false), spec(TensorHead::Embed, OcrNorm::Unit, false)).unwrap();
    let v = vector(&m.run(&[Frame::Rgb8 { w: 16, h: 16, data: &yellow }]).unwrap()[0]);
    let h = 1.0 / 2f32.sqrt();
    assert!((v[0] - h).abs() < 1e-5 && (v[1] - h).abs() < 1e-5 && v[2].abs() < 1e-5, "{v:?}");
}

#[test]
fn a_batch_equals_one_at_a_time() {
    let imgs = [(20, 10, [255, 0, 0]), (8, 8, [0, 255, 0]), (33, 17, [10, 20, 250])];
    let data: Vec<Vec<u8>> = imgs.iter().map(|(w, h, c)| solid(*w, *h, *c)).collect();
    let frames: Vec<Frame> = imgs.iter().zip(&data).map(|((w, h, _), d)| Frame::Rgb8 { w: *w, h: *h, data: d }).collect();
    let mut m = cpu().tensor_model(&tiny(false), spec(TensorHead::Embed, OcrNorm::Unit, false)).unwrap();
    let all = m.run(&frames).unwrap();
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(format!("{:?}", all[i]), format!("{:?}", m.run(std::slice::from_ref(f)).unwrap()[0]));
    }
}

/// Every GPU this build and machine offer, as runtimes.
fn gpus() -> Vec<Runtime> {
    let info = Runtime::probe();
    let mut out = vec![];
    for g in &info.gpus {
        out.push(Runtime::new(RuntimeCfg { threads: Some(2), device: Device::Cuda(g.ordinal) }).unwrap());
    }
    for v in info.vulkan.iter().filter(|v| v.kind == "discrete" || v.kind == "integrated") {
        out.push(Runtime::new(RuntimeCfg { threads: Some(2), device: Device::Vulkan(v.index) }).unwrap());
    }
    out
}

#[test]
fn gpus_agree_with_the_cpu() {
    let rts = gpus();
    if rts.is_empty() {
        eprintln!("skipped: no GPU backend in this build / machine");
        return;
    }
    let imgs: Vec<(usize, usize, Vec<u8>)> = (0..6).map(|i| (7 + i * 5, 5 + i * 3, (0..(7 + i * 5) * (5 + i * 3) * 3).map(|k| ((k * 37 + i * 11) % 256) as u8).collect())).collect();
    let frames: Vec<Frame> = imgs.iter().map(|(w, h, d)| Frame::Rgb8 { w: *w, h: *h, data: d }).collect();
    for head in [TensorHead::Classify { top_k: 3 }, TensorHead::Embed] {
        let want = cpu().tensor_model(&tiny(false), spec(head, OcrNorm::Signed, false)).unwrap().run(&frames).unwrap();
        for rt in &rts {
            let mut m = rt.tensor_model(&tiny(false), spec(head, OcrNorm::Signed, false)).unwrap();
            assert_ne!(m.backend(), "cpu");
            let got = m.run(&frames).unwrap();
            for (g, w) in got.iter().zip(&want) {
                let (gv, wv): (Vec<f32>, Vec<f32>) = match (g, w) {
                    (Output::Labels(a), Output::Labels(b)) => (a.iter().map(|x| x.1).collect(), b.iter().map(|x| x.1).collect()),
                    (Output::Vector(a), Output::Vector(b)) => (a.clone(), b.clone()),
                    other => panic!("{other:?}"),
                };
                // fp16 on the GPU: the stretch resize and the mean differ by a few 1e-3
                for (a, b) in gv.iter().zip(&wv) {
                    assert!((a - b).abs() < 2e-2, "{} {head:?}: {gv:?} vs cpu {wv:?}", m.backend());
                }
            }
        }
    }
}

fn dir(var: &str, default: &str) -> Option<PathBuf> {
    let p = std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(default));
    p.is_dir().then_some(p)
}

/// Detect and OCR through `Model` are the existing wrappers, bit for bit.
#[test]
fn detect_and_ocr_through_model_are_unchanged() {
    let (Some(models), Some(images)) = (dir("OJAS_MODELS", "../../models"), dir("OJAS_PARITY_IMAGES", "../../fixtures/plates")) else {
        eprintln!("skipped: no models / parity images");
        return;
    };
    let rt = cpu();
    let det = models.join("plate-v9t-384.onnx");
    let (ocr, dict) = (models.join("awiros_rec.onnx"), models.join("awiros_dict.txt"));
    let dcfg = || DetectorCfg { conf: 0.15, max_det: 20, class_agnostic_nms: true, ..Default::default() };
    let ocfg = || OcrCfg { norm: OcrNorm::Signed, bgr: true, ..Default::default() };
    let mut d1 = rt.detector(det.to_str().unwrap(), dcfg()).unwrap();
    let mut d2 = rt.model(det.to_str().unwrap(), ModelKind::Detect(dcfg())).unwrap();
    let mut o1 = rt.plate_ocr(ocr.to_str().unwrap(), dict.to_str().unwrap(), ocfg()).unwrap();
    let mut o2 = rt.model(ocr.to_str().unwrap(), ModelKind::Ocr { dict: dict.to_str().unwrap(), cfg: ocfg() }).unwrap();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&images).unwrap().filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "jpg" || x == "png")).collect();
    files.sort();
    assert!(files.len() >= 20, "{} images in {}", files.len(), images.display());
    for f in &files {
        let (w, h, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(f.to_str().unwrap()).unwrap();
        let (w, h) = (w as usize, h as usize);
        let fr = [Frame::Rgb8 { w, h, data: &rgb }];
        let a = d1.run(&fr).unwrap().remove(0);
        match d2.run(&fr).unwrap().remove(0) {
            Output::Boxes(b) => assert_eq!(format!("{a:?}"), format!("{b:?}"), "{}", f.display()),
            other => panic!("{other:?}"),
        }
        let r1 = o1.run(&fr).unwrap().remove(0);
        match o2.run(&fr).unwrap().remove(0) {
            Output::Text(r2) => assert_eq!(format!("{r1:?}"), format!("{r2:?}"), "{}", f.display()),
            other => panic!("{other:?}"),
        }
    }
}

/// The GPU lowering pass (GlobalAvgPool / Gemm) touches none of the running
/// ANPR graphs, so their GPU plans are exactly what they were before it.
#[test]
fn production_graphs_need_no_lowering() {
    let Some(models) = dir("OJAS_MODELS", "../../models") else {
        eprintln!("skipped: no models");
        return;
    };
    for (file, width) in [("yolo11n.onnx", None), ("plate-v9t-384.onnx", None), ("awiros_rec.onnx", Some(320))] {
        // `models/` is committed but the weights in it are not, so a clean
        // checkout has the directory and none of the graphs.
        let path = models.join(file);
        if !path.is_file() {
            eprintln!("skipped {file}: not present");
            continue;
        }
        let m = ojas_formats::onnx::load(path.to_str().unwrap()).unwrap();
        let mut binds = std::collections::HashMap::new();
        for vi in &m.graph.inputs {
            for (i, d) in vi.dims.iter().enumerate() {
                if let OnnxDim::Param(p) = d {
                    binds.insert(p.clone(), if i == 0 { 1 } else { width.unwrap() });
                }
            }
        }
        let mut g = ojas_vision::import(&m, &binds).unwrap();
        ojas_vision::passes::optimize(&mut g);
        assert_eq!(ojas_vision::passes::lower_for_gpu(&mut g), 0, "{file}");
    }
}
