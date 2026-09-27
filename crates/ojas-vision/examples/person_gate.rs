//! The person models through the public `Model` API, CPU against the GPU, on real crops: OSNet
//! (ImageNet mean/std, stretched box → 512-d embedding), RTMPose (1.25× window, zero fill →
//! 17 keypoints) and SigLIP 2 (±0.5, stretched → 768-d). Reports the CPU-vs-GPU agreement per
//! model (cosine for embeddings, pixels for keypoints) and ms per crop at the batch the GPU
//! chooses — the gate for the person lanes, like `pipeline_gate` for plates.
//!
//! `person_gate <frames.json> <persons.dets.json> <osnet.onnx> <rtmpose.onnx> <siglip_vision.onnx> [crops 64] [score 0.4] [min_h 100]`
//! Env: `OJAS_GATE_DEVICE=cuda:0|vulkan:0` (default cuda:0).
use ojas_vision::model::{ModelKind, Output, TensorHead, TensorSpec};
use ojas_vision::pre::{OcrNorm, Window};
use ojas_vision::{Device, Frame, Runtime, RuntimeCfg};
use serde_json::Value;

const IMAGENET: ([f32; 3], [f32; 3]) = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);

fn specs() -> [(&'static str, TensorSpec); 3] {
    [
        ("osnet", TensorSpec { width: 128, height: 256, mean_std: Some(IMAGENET), window: Window::Stretch, head: TensorHead::Embed, ..Default::default() }),
        ("rtmpose", TensorSpec { width: 192, height: 256, mean_std: Some(IMAGENET), window: Window::Around { scale: 1.25 }, fill_u8: 0, head: TensorHead::Pose { split: 2.0 }, ..Default::default() }),
        ("siglip", TensorSpec { width: 224, height: 224, norm: OcrNorm::Signed, window: Window::Stretch, head: TensorHead::Embed, ..Default::default() }),
    ]
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 6, "person_gate <frames.json> <persons.dets.json> <osnet.onnx> <rtmpose.onnx> <siglip.onnx> [crops] [score] [min_h]");
    let n_crops: usize = a.get(6).and_then(|v| v.parse().ok()).unwrap_or(64);
    let score: f32 = a.get(7).and_then(|v| v.parse().ok()).unwrap_or(0.4);
    let min_h: f32 = a.get(8).and_then(|v| v.parse().ok()).unwrap_or(100.0);
    let gpu = match std::env::var("OJAS_GATE_DEVICE").unwrap_or_else(|_| "cuda:0".into()) {
        d if d.starts_with("vulkan") => Device::Vulkan(d.split_once(':').and_then(|(_, i)| i.parse().ok()).unwrap_or(0)),
        d => Device::Cuda(d.split_once(':').and_then(|(_, i)| i.parse().ok()).unwrap_or(0)),
    };

    // crops: every k-th qualifying detection, spread over the file; the box itself is the frame
    // handed to the model (the window rule grows it inside the model)
    let frames: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let files: std::collections::HashMap<i64, String> = frames["images"].as_array().unwrap().iter().map(|im| (im["id"].as_i64().unwrap(), im["file_name"].as_str().unwrap().to_string())).collect();
    let dets: Vec<Value> = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    let picks: Vec<(i64, [f32; 4])> = dets
        .iter()
        .filter(|d| (d["score"].as_f64().unwrap() as f32) >= score)
        .filter_map(|d| {
            let b: Vec<f32> = d["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            (b[3] >= min_h).then(|| (d["image_id"].as_i64().unwrap(), [b[0], b[1], b[2], b[3]]))
        })
        .collect();
    let step = (picks.len() / n_crops).max(1);
    let picks: Vec<_> = picks.into_iter().step_by(step).take(n_crops).collect();
    // crops with 20 % context so the pose window has room; the model sees the crop as its box
    let mut crops: Vec<(usize, usize, Vec<u8>)> = vec![];
    let mut cache: Option<(i64, image::RgbImage)> = None;
    for (id, b) in &picks {
        if cache.as_ref().map(|c| c.0) != Some(*id) {
            cache = Some((*id, image::open(&files[id])?.to_rgb8()));
        }
        let img = &cache.as_ref().unwrap().1;
        let (iw, ih) = (img.width() as f32, img.height() as f32);
        let (px, py) = (b[2] * 0.2, b[3] * 0.2);
        let x0 = (b[0] - px).max(0.0);
        let y0 = (b[1] - py).max(0.0);
        let x1 = (b[0] + b[2] + px).min(iw);
        let y1 = (b[1] + b[3] + py).min(ih);
        let c = image::imageops::crop_imm(img, x0 as u32, y0 as u32, (x1 - x0).max(2.0) as u32, (y1 - y0).max(2.0) as u32).to_image();
        crops.push((c.width() as usize, c.height() as usize, c.into_raw()));
    }
    println!("{} crops (score ≥ {score}, height ≥ {min_h} px)\n", crops.len());
    let frames_in: Vec<Frame> = crops.iter().map(|(w, h, d)| Frame::Rgb8 { w: *w, h: *h, data: d }).collect();

    let cpu = Runtime::new(RuntimeCfg { threads: None, device: Device::Cpu })?;
    let dev = Runtime::new(RuntimeCfg { threads: None, device: gpu })?;
    let mut all_ok = true;
    for ((name, spec), path) in specs().into_iter().zip(&a[3..6]) {
        let mut mc = cpu.model(path, ModelKind::Tensor(spec))?;
        let mut mg = dev.model(path, ModelKind::Tensor(spec))?;
        let t = std::time::Instant::now();
        let oc = mc.run(&frames_in)?;
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / frames_in.len() as f64;
        mg.run(&frames_in[..1])?; // warm-up: kernels, autotune
        let t = std::time::Instant::now();
        let og = mg.run(&frames_in)?;
        let gpu_ms = t.elapsed().as_secs_f64() * 1e3 / frames_in.len() as f64;
        let (mut worst, mut kp_px, mut kp_n, mut kp_far) = (1.0f32, 0.0f32, 0usize, 0usize);
        if std::env::var("VERBOSE").is_ok() {
            for (i, (c, g)) in oc.iter().zip(&og).take(3).enumerate() {
                if let (Output::Keypoints(kc), Output::Keypoints(kg)) = (c, g) {
                    println!("  crop {i} ({}x{}): cpu {:?}\n                 gpu {:?}", crops[i].0, crops[i].1, &kc[..5].iter().map(|k| (k[0].round(), k[1].round(), (k[2] * 100.0).round() / 100.0)).collect::<Vec<_>>(), &kg[..5].iter().map(|k| (k[0].round(), k[1].round(), (k[2] * 100.0).round() / 100.0)).collect::<Vec<_>>());
                }
            }
        }
        for (c, g) in oc.iter().zip(&og) {
            match (c, g) {
                (Output::Vector(vc), Output::Vector(vg)) => worst = worst.min(cos(vc, vg)),
                (Output::Keypoints(kc), Output::Keypoints(kg)) => {
                    // confident on both sides: a flat SimCC peak may flip bins under f16, a
                    // confident one may not move more than a bin (0.5 px at split 2)
                    for (p, q) in kc.iter().zip(kg) {
                        if p[2] >= 0.5 && q[2] >= 0.5 {
                            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
                            kp_px = kp_px.max(d);
                            kp_n += 1;
                            kp_far += (d > 2.0) as usize;
                        }
                    }
                }
                _ => anyhow::bail!("{name}: output kinds differ"),
            }
        }
        let ok = match spec.head {
            TensorHead::Pose { .. } => kp_far * 200 <= kp_n.max(1), // ≥ 99.5 % within 2 px
            _ => worst >= 0.995,
        };
        all_ok &= ok;
        match spec.head {
            TensorHead::Pose { .. } => println!("{name:<8} {} vs cpu: {kp_n} confident keypoints, {kp_far} moved > 2 px (max {kp_px:.1} px); CPU {cpu_ms:.1} ms/crop, GPU {gpu_ms:.2} ms/crop  {}", mg.backend(), if ok { "OK" } else { "FAIL" }),
            _ => println!("{name:<8} {} vs cpu: worst cosine {worst:.5}; CPU {cpu_ms:.1} ms/crop, GPU {gpu_ms:.2} ms/crop  {}", mg.backend(), if ok { "OK" } else { "FAIL" }),
        }
    }
    println!("\nperson models CPU == GPU: {all_ok}");
    Ok(())
}
