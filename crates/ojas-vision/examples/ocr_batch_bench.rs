//! OCR batching measurement: ms/crop at each plan batch size, plus the
//! batch-composition determinism check (batched text must equal batch-1).
//!
//! cargo run -p ojas-vision --release --example ocr_batch_bench -- \
//!     rec.onnx dict.txt [norm=unit] [bgr] <crops_dir> [n_crops]

use anyhow::{bail, ensure, Result};
use ojas_vision::{Frame, OcrCfg, OcrNorm, Runtime, RuntimeCfg};

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().collect();
    let norm = if let Some(p) = args.iter().position(|a| a == "norm=unit") {
        args.remove(p);
        OcrNorm::Unit
    } else {
        OcrNorm::Signed
    };
    let bgr = if let Some(p) = args.iter().position(|a| a == "bgr") {
        args.remove(p);
        true
    } else {
        false
    };
    if args.len() < 4 {
        bail!("usage: ocr_batch_bench <rec.onnx> <dict.txt> [norm=unit] [bgr] <crops_dir> [n]");
    }
    let n: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(64);

    // load crops
    let mut paths: Vec<_> = std::fs::read_dir(&args[3])?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jpg" || x == "png"))
        .collect();
    paths.sort();
    paths.truncate(n);
    ensure!(!paths.is_empty(), "no crops in {}", args[3]);
    let images: Vec<(usize, usize, Vec<u8>)> =
        paths.iter().map(|p| ojas_cpu::vit_preprocess::decode_rgb8_path(p)).collect::<Result<_>>()?;
    let frames: Vec<Frame> = images.iter().map(|(w, h, d)| Frame::Rgb8 { w: *w, h: *h, data: d }).collect();
    println!("model {}  crops {}  threads auto", args[1], frames.len());

    let rt = Runtime::new(RuntimeCfg::default())?;
    let mut ocr = rt.plate_ocr(&args[1], &args[2], OcrCfg { norm, bgr, ..Default::default() })?;

    // reference: batch-1 reads for the determinism check
    let mut reference = Vec::new();
    for f in &frames {
        reference.push(ocr.run(std::slice::from_ref(f))?.remove(0).text);
    }

    for &b in &[1usize, 4, 8, 16, 32, 64] {
        if b > frames.len() {
            break;
        }
        let t_warm = std::time::Instant::now();
        ocr.warmup(&[b])?;
        let warm_ms = t_warm.elapsed().as_secs_f64() * 1e3;
        // min-of-reps over the whole set, chunked at exactly b
        let mut best = f64::MAX;
        let mut texts = Vec::new();
        for _ in 0..5 {
            texts.clear();
            let t0 = std::time::Instant::now();
            for chunk in frames.chunks(b) {
                for r in ocr.run(chunk)? {
                    texts.push(r.text);
                }
            }
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let mismatches = texts.iter().zip(&reference).filter(|(a, b)| a != b).count();
        println!(
            "batch {b:>2}: {:7.2} ms/crop  ({:6.1} ms total, plan build {:5.0} ms, determinism vs batch-1: {})",
            best * 1e3 / frames.len() as f64,
            best * 1e3,
            warm_ms,
            if mismatches == 0 { "OK".to_string() } else { format!("{mismatches} MISMATCHES") }
        );
    }
    Ok(())
}
