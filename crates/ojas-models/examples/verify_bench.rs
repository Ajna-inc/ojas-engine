//! Where a speculative verify's wall time goes: kernels, or everything else.
//!
//! profile_batch reports the per-category kernel floor and spec_bench reports the wall.
//! When those two disagree the difference is not a kernel problem, and at M=4 they
//! disagree a lot; this says by how much, and splits the remainder into GPU-reported
//! execution and CPU time.
//!
//! The baseline is the same split for a single decode forward: decode has the same
//! per-dispatch drain and the same encoder.
//!
//! usage: verify_bench <gguf> [prec]

use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: verify_bench <gguf> [prec]");
    let prec: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;

    // warm
    for p in 0..64 { m.forward_id(if p == 0 { 1 } else { 100 }, p); }
    let toks: Vec<u32> = (0..8).map(|i| 100u32 + i as u32).collect();
    for _ in 0..3 { let _ = m.forward_batch_ids(&toks[..4], 64); }

    println!("{:<16} {:>9} {:>9} {:>9} {:>8}", "path", "wall ms", "gpu ms", "off-gpu", "off %");
    let mut row = |label: String, reps: usize, f: &dyn Fn()| {
        // best-of: the minimum is the least-contended sample, same rule the
        // other benches here use.
        let mut best_wall = f64::INFINITY;
        let mut best_gpu = 0.0;
        for _ in 0..7 {
            let g0 = m.gpu_seconds();
            let t0 = std::time::Instant::now();
            for _ in 0..reps { f(); }
            let wall = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let gms = (m.gpu_seconds() - g0) * 1e3 / reps as f64;
            if wall < best_wall { best_wall = wall; best_gpu = gms; }
        }
        let off = best_wall - best_gpu;
        println!("{:<16} {:>9.3} {:>9.3} {:>9.3} {:>7.1}%",
            label, best_wall, best_gpu, off, 100.0 * off / best_wall);
    };

    row("decode (M=1)".into(), 8, &|| { m.forward_id(100, 64); });
    for bm in [1usize, 2, 4, 8] {
        row(format!("verify M={bm}"), 4, &|| { let _ = m.forward_batch_ids(&toks[..bm], 64); });
    }
    Ok(())
}
