//! Whether expert streaming survives PCIe.
//!
//! On Apple the streamed-expert path costs a page-cache read into memory the GPU already
//! addresses, which is how a 27 GB model runs in a 3.18 GB resident set. On NVIDIA the same
//! gather crosses PCIe, so the memory advantage depends on whether the link can carry it at the
//! target token rate. Three numbers answer that:
//!
//! 1. pageable host → device bandwidth (what a naive `Vec` gather gets)
//! 2. pinned host → device bandwidth (page-locked staging, no driver bounce buffer)
//! 3. the achievable token rate for a given per-token expert footprint
//!
//! `cargo run --release -p ojas-cuda --example stream_probe [-- <MiB per token>]`
use anyhow::Result;
use ojas_cuda::CudaGpu;
use std::time::Instant;

fn main() -> Result<()> {
    let per_token_mib: f64 = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(58.0); // 10 routed experts of ~5.8 MiB at IQ4

    let gpu = CudaGpu::new(0)?;
    let chunk = 64 << 20; // 64 MiB, big enough that per-call overhead is not the story
    let reps = 8;

    // ---- pageable
    let pageable = vec![7u8; chunk];
    let mut dst = gpu.alloc_bytes(chunk)?;
    gpu.upload_bytes(&pageable[..1024])?; // warm the context
    let t = Instant::now();
    for _ in 0..reps {
        dst = gpu.upload_bytes(&pageable)?;
    }
    gpu.sync()?;
    let pageable_gbs = (chunk as f64 * reps as f64) / t.elapsed().as_secs_f64() / 1e9;

    // ---- pinned
    let mut pinned = gpu.alloc_pinned(chunk)?;
    pinned.as_mut_slice()?.fill(7u8);
    gpu.upload_pinned_async(&pinned, &mut dst)?;
    gpu.sync()?;
    let t = Instant::now();
    for _ in 0..reps {
        gpu.upload_pinned_async(&pinned, &mut dst)?;
    }
    gpu.sync()?;
    let pinned_gbs = (chunk as f64 * reps as f64) / t.elapsed().as_secs_f64() / 1e9;

    println!("host → device bandwidth on this link");
    println!("  pageable  {pageable_gbs:6.2} GB/s");
    println!("  pinned    {pinned_gbs:6.2} GB/s   ({:.2}× pageable)", pinned_gbs / pageable_gbs);

    // ---- what that means for streamed experts
    let per_token_gb = per_token_mib / 1024.0;
    println!("\nstreamed experts at {per_token_mib:.1} MiB per token");
    for (label, bw) in [("pageable", pageable_gbs), ("pinned", pinned_gbs)] {
        let tok_s = bw / per_token_gb;
        println!("  {label:9} {tok_s:7.1} tok/s ceiling  ({:.2} ms per token of transfer)",
                 1000.0 * per_token_gb / bw);
    }
    println!("\nFor reference, Metal's measured Flash Next rate is 23.16 tok/s on unified memory.");
    println!("A ceiling above that means PCIe is");
    println!("not what limits us; a ceiling below it means the streamed path needs a resident");
    println!("cache large enough to keep the miss rate under the difference.");
    Ok(())
}
