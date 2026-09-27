//! Checkpoint timing including cached crop JPEG decoding and preprocessing.
use anyhow::{ensure, Result};
use ojas_learn::{cuda::Cuda, models::rtdetr::Store, passenger, Tape};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "passenger_crop_bench expert.safetensors manifest.jsonl");
    let be = Cuda::new(0)?;
    let st = Store::from_safetensors(&be, &args[1], "model.")?;
    let mut crops = vec![];
    for line in std::fs::read_to_string(&args[2])?.lines() {
        let row: serde_json::Value = serde_json::from_str(line)?;
        if row["split"] == "dev" {
            crops.push(std::fs::read(row["path"].as_str().unwrap())?);
            if crops.len() == 64 { break; }
        }
    }
    ensure!(crops.len() >= 16, "need at least 16 development crops");
    let parameters: usize = st.params.values().map(|p| p.shape.iter().product::<usize>()).sum();
    for batch in [1, 4, 16] {
        let (mut total, mut forward) = (vec![], vec![]);
        for i in 0..60 {
            let start = Instant::now();
            let mut inputs = vec![];
            for j in 0..batch {
                let img = image::load_from_memory(&crops[(i * batch + j) % crops.len()])?.to_rgb8();
                inputs.extend(passenger::preprocess(&img, 224, false, 1.));
            }
            let gpu = Instant::now();
            let mut t = Tape::new(&be);
            let x = t.input(&inputs, &[batch, 3, 224, 224]);
            let logits = passenger::forward(&mut t, &st, x)?;
            let values = t.value(logits);
            let probs = passenger::probabilities(&values);
            ensure!(probs.iter().flatten().all(|v| v.is_finite()), "nonfinite predictions");
            let gpu_ms = gpu.elapsed().as_secs_f64() * 1000.;
            let total_ms = start.elapsed().as_secs_f64() * 1000.;
            if i >= 10 { total.push(total_ms); forward.push(gpu_ms); }
        }
        total.sort_by(f64::total_cmp); forward.sort_by(f64::total_cmp);
        println!("{}", serde_json::json!({"batch":batch,"size":224,"parameters":parameters,
            "crop_pipeline_p50_ms":total[25],"crop_pipeline_p95_ms":total[47],
            "upload_forward_download_p50_ms":forward[25],"upload_forward_download_p95_ms":forward[47],
            "warmups":10,"iterations":50,"scope":"actual checkpoint, Ojas CUDA BF16 tape; crop JPEG bytes preloaded; includes JPEG decode, resize/letterbox, tensor preparation and GPU transfer; excludes full-frame detector, box cropping, disk IO, tracking and scheduling"}));
    }
    Ok(())
}
