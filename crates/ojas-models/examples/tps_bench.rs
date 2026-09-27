//! Raw decode throughput: `forward_id` in a loop, no server, no tokenizer.
//!
//! Separates engine cost from everything around it. The profiler reports summed GPU
//! kernel time; this reports wall time per token, so the difference is submission
//! latency and CPU-side encoding.
//!
//! usage: tps_bench <gguf> [n_tokens] [prec]

use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: tps_bench <gguf> [n] [prec]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let prec: u8 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    ojas_core::logging::init();   // OJAS_LOG="info,tok=trace" for the per-phase split
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 2048, prec, None, None)?;

    // Warm: autotune plans, pipeline compilation, first-touch page faults.
    let mut t = 1u32;
    for p in 0..32 {
        t = m.forward_id(t, p);
    }

    let g0 = m.gpu_seconds();
    let t0 = std::time::Instant::now();
    for p in 32..(32 + n) {
        t = m.forward_id(t, p);
    }
    let el = t0.elapsed().as_secs_f64();
    // GPU busy time vs wall: the difference is everything that is not the GPU
    // executing — CPU encoding of the ~290 dispatches, submission, and the
    // round-trip wait. Which one it is decides whether the fix is fewer kernels or
    // a different submission strategy.
    let gs = m.gpu_seconds() - g0;
    println!(
        "raw decode: {:.1} tok/s  ({:.3} ms/token, {n} tokens, prec={prec})",
        n as f64 / el,
        el * 1000.0 / n as f64
    );
    println!(
        "  gpu busy {:.3} ms/token  |  off-gpu {:.3} ms/token ({:.0}%)",
        gs * 1000.0 / n as f64,
        (el - gs) * 1000.0 / n as f64,
        (el - gs) / el * 100.0
    );
    Ok(())
}
