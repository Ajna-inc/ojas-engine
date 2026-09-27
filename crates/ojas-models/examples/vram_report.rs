//! Reports how much GPU memory a model occupies, per source format.
//!
//! The loader requantizes most formats to Q8 (prec=1) or Q4L (prec=2) at load, so the
//! resident footprint is set by `prec`, not by what the file on disk costs. For a 2-bit IQ
//! model that is a large expansion, and it is invisible in every other measurement here:
//! `du` reports the file, RSS conflates the mmap'd source pages with the buffers, and
//! tok/s does not mention memory.
//!
//! `currentAllocatedSize` is the authoritative number: it is what Metal says it handed
//! out, so it counts the requantized buffers and nothing else.
//!
//! usage: vram_report <gguf> [prec]

use anyhow::Result;
use ojas_core::Device as _;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: vram_report <gguf> [prec]");
    let prec: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let ctx: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(4096);

    let gpu = ojas_metal::MetalGpu::new()?;
    let before = gpu.device.current_allocated_size();
    let mut g = ojas_formats::gguf::Gguf::open(&path)?;
    let file_bytes = std::fs::metadata(&path)?.len();
    let params: u64 = g.tensors.values()
        .filter(|i| i.dims.len() >= 2)
        .map(|i| i.dims.iter().product::<u64>())
        .sum();
    let t0 = std::time::Instant::now();
    let _m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, ctx, prec, None, None)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let after = gpu.device.current_allocated_size();

    let gb = |b: u64| b as f64 / 1e9;
    let resident = after.saturating_sub(before);
    println!("  file        {:7.2} GB   ({:.2} bits/weight on disk)",
        gb(file_bytes), file_bytes as f64 * 8.0 / params as f64);
    println!("  gpu alloc   {:7.2} GB   ({:.2} bits/weight resident)  prec={prec}",
        gb(resident), resident as f64 * 8.0 / params as f64);
    println!("  expansion   {:7.2}x   (ctx={ctx})", resident as f64 / file_bytes as f64);
    println!("  load        {load_ms:7.0} ms");
    Ok(())
}
