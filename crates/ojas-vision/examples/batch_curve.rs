//! The measured batch curve of a detector: ms per forward and ms per frame at
//! batch 1, 2, 4, 8, 16 — what a dispatcher needs before it decides to wait for a
//! batch. Lingering only pays when the curve is steep enough to beat the delay it
//! adds.
//! `batch_curve model.onnx [reps] [width] [height]`  (`OJAS_DEVICE=cuda:0`)
use anyhow::Result;
use ojas_vision::{Device, DetectorCfg, Frame, Runtime, RuntimeCfg};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() > 1, "batch_curve model.onnx [reps] [width] [height]");
    let reps: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(30);
    let (w, h): (usize, usize) = (a.get(3).and_then(|v| v.parse().ok()).unwrap_or(1920), a.get(4).and_then(|v| v.parse().ok()).unwrap_or(1080));
    let device = match std::env::var("OJAS_DEVICE").unwrap_or_else(|_| "cpu".into()) {
        d if d.starts_with("cuda") => Device::Cuda(d.split_once(':').and_then(|(_, i)| i.parse().ok()).unwrap_or(0)),
        d if d.starts_with("vulkan") => Device::Vulkan(d.split_once(':').and_then(|(_, i)| i.parse().ok()).unwrap_or(0)),
        _ => Device::Cpu,
    };
    let rt = Runtime::new(RuntimeCfg { threads: None, device })?;
    let mut det = rt.detector(&a[1], DetectorCfg { conf: 0.25, ..Default::default() })?;
    // frames with structure, so preprocessing and NMS do realistic work
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            rgb[i] = (x % 251) as u8;
            rgb[i + 1] = (y % 241) as u8;
            rgb[i + 2] = ((x ^ y) % 239) as u8;
        }
    }
    println!("{} at {w}×{h} on {device:?}, {reps} reps\n", a[1].rsplit('/').next().unwrap_or(""));
    println!("| batch | ms / forward | ms / frame | frames/s | vs batch 1 |");
    println!("|---:|---:|---:|---:|---:|");
    let mut base = 0.0f64;
    for (bi, batch) in [1usize, 2, 4, 8, 16].iter().enumerate() {
        let frames: Vec<Frame> = (0..*batch).map(|_| Frame::Rgb8 { w, h, data: &rgb }).collect();
        det.run(&frames)?; // warm up: kernels compile, plans settle
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = std::time::Instant::now();
            det.run(&frames)?;
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(f64::total_cmp);
        let med = times[times.len() / 2];
        let per_frame = med / *batch as f64;
        if bi == 0 {
            base = per_frame;
        }
        println!("| {batch} | {med:.2} | {per_frame:.2} | {:.0} | {:.2}× |", 1000.0 / per_frame, base / per_frame);
    }
    println!("\nA dispatcher should wait for a batch only while the per-frame gain exceeds the wait it adds.");
    Ok(())
}
