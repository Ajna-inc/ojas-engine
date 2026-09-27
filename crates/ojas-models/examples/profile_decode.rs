//! Per-kernel decode profile: where a token's time actually goes.
//!
//! `prec` matters: 1 requantizes everything to Q8, 2 uses the tuned 4-bit family, 3
//! keeps Q4_K/Q6_K native. They have very different byte counts per token, so profiling
//! the wrong one measures a path nobody runs.
//!
//! `DecoderGpu::profile` reports through `tracing` at DEBUG, so a subscriber must be
//! installed or the run prints nothing at all. The filter defaults to debug so the plain
//! invocation reports; OJAS_LOG still overrides.
//!
//! usage: profile_decode <gguf> [pos] [prec]

use anyhow::Result;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: profile_decode <gguf> [pos] [prec]");
    let pos: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let prec: u8 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    if std::env::var("OJAS_LOG").is_err() {
        // SAFETY: single-threaded, before any other thread exists.
        unsafe { std::env::set_var("OJAS_LOG", "debug"); }
    }
    ojas_core::logging::init();
    ojas_models::bench::report_load();
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 2048, prec, None, None)?;
    // Warm the caches / autotune with a few real steps first.
    for p in 0..pos {
        m.forward_id(if p == 0 { 1u32 } else { 100u32 }, p);
    }
    // Wall-clock reference for the category sum: the category floor alone says nothing
    // without the real per-token GPU time.
    let g0 = m.gpu_seconds();
    let t0 = std::time::Instant::now();
    for p in pos..(pos + 64) { m.forward_id(100u32, p); }
    let el = t0.elapsed().as_secs_f64();
    let gs = m.gpu_seconds() - g0;
    tracing::debug!(target: "profile",
        "REAL decode: {:.3} ms/token wall, {:.3} ms/token gpu-busy ({:.1} tok/s)",
        el * 1e3 / 64.0, gs * 1e3 / 64.0, 64.0 / el);
    m.profile(pos);
    Ok(())
}
