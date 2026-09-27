//! Per-category GPU profile of the batched prefill path.
//!
//! Decode is bandwidth-bound; prefill at large M is supposed to be compute-bound,
//! so the GEMMs should dominate and everything else should be noise. This prints
//! the split so "prefill is slow" can be turned into "this kernel is slow".
//!
//! usage: profile_prefill <gguf> [m] [pos] [prec]

use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: profile_prefill <gguf> [m] [pos] [prec]");
    let m: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let pos: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(512);
    let prec: u8 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let mo = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;
    // Warm: one real batched pass so autotune/pipelines are hot.
    let toks: Vec<u32> = (0..m).map(|i| 100u32 + (i % 50)).collect();
    mo.prefill(&toks, 0);
    mo.profile_batch(m, pos);
    Ok(())
}
