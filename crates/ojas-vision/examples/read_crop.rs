//! Read plate crops with the OCR model directly (no detector).
//!
//! `cargo run -p ojas-vision --release --example read_crop -- rec.onnx dict.txt crop.png [more...]`
//!
//! This is the OCR accuracy harness's inner loop: exact-match / CER scripts
//! feed labelled crop sets through it.

use anyhow::{bail, Result};
use ojas_vision::{Frame, OcrCfg, OcrNorm, Runtime, RuntimeCfg};

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().collect();
    let norm = if let Some(pos) = args.iter().position(|a| a == "norm=unit") {
        args.remove(pos);
        OcrNorm::Unit
    } else {
        OcrNorm::Signed
    };
    let bgr = if let Some(pos) = args.iter().position(|a| a == "bgr") {
        args.remove(pos);
        true
    } else {
        false
    };
    if args.len() < 4 {
        bail!("usage: read_crop <rec.onnx> <dict.txt> [norm=unit] [bgr] <crop image>...");
    }
    let rt = Runtime::new(RuntimeCfg::default())?;
    let mut ocr = rt.plate_ocr(&args[1], &args[2], OcrCfg { norm, bgr, ..Default::default() })?;
    for path in &args[3..] {
        let (w, h, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(path)?;
        let t0 = std::time::Instant::now();
        let read = ocr.run(&[Frame::Rgb8 { w, h, data: &rgb }])?.remove(0);
        println!(
            "{path} ({w}x{h}, {:.1} ms): {:?} mean_conf {:.3} per-char {:?}",
            t0.elapsed().as_secs_f64() * 1e3,
            read.text,
            read.mean_conf,
            read.char_conf.iter().map(|c| (c * 100.0).round() / 100.0).collect::<Vec<_>>()
        );
    }
    Ok(())
}
