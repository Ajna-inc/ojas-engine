//! OCR accuracy harness: exact-match and CER per crop-height bucket.
//!
//! cargo run -p ojas-vision --release --example ocr_bench -- \
//!     rec.onnx dict.txt [norm=unit] [bgr] labels.tsv crops_dir
//!
//! labels.tsv: `<filename>\t<ground truth>` per line. Comparison is case-folded,
//! since the deployment normalizer uppercases anyway. The per-height table is what
//! the read gate's "TooSmall" threshold is set from.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use ojas_vision::{Frame, OcrCfg, OcrNorm, Runtime, RuntimeCfg};

fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[derive(Default)]
struct Bucket {
    n: usize,
    exact: usize,
    edits: usize,
    ref_chars: usize,
    conf_sum: f64,
}

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
    if args.len() != 5 {
        bail!("usage: ocr_bench <rec.onnx> <dict.txt> [norm=unit] [bgr] <labels.tsv> <crops_dir>");
    }
    let rt = Runtime::new(RuntimeCfg::default())?;
    let mut ocr = rt.plate_ocr(&args[1], &args[2], OcrCfg { norm, bgr, ..Default::default() })?;

    let labels = std::fs::read_to_string(&args[3]).with_context(|| args[3].clone())?;
    let dir = &args[4];
    let mut buckets: BTreeMap<usize, Bucket> = BTreeMap::new();
    let mut total = 0usize;
    for line in labels.lines().filter(|l| !l.trim().is_empty()) {
        let (file, truth) = line.split_once('\t').with_context(|| format!("bad label line {line:?}"))?;
        let (w, h, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(format!("{dir}/{file}"))?;
        let read = ocr.run(&[Frame::Rgb8 { w, h, data: &rgb }])?.remove(0);
        let got: Vec<char> = read.text.to_uppercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        let want: Vec<char> = truth.to_uppercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        let b = buckets.entry(h).or_default();
        b.n += 1;
        b.exact += usize::from(got == want);
        b.edits += edit_distance(&got, &want);
        b.ref_chars += want.len();
        b.conf_sum += read.mean_conf as f64;
        total += 1;
    }
    println!("model {}  norm {:?}  bgr {}  crops {}", args[1], norm, bgr, total);
    println!("{:>8} {:>6} {:>12} {:>8} {:>10}", "height", "n", "exact-match", "CER", "mean_conf");
    for (h, b) in &buckets {
        println!(
            "{:>7}px {:>6} {:>11.1}% {:>7.1}% {:>10.3}",
            h,
            b.n,
            100.0 * b.exact as f64 / b.n as f64,
            100.0 * b.edits as f64 / b.ref_chars.max(1) as f64,
            b.conf_sum / b.n as f64
        );
    }
    Ok(())
}
