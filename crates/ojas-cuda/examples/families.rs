//! Compile every kernel family on this GPU and report what NVRTC said.
//!
//! `families_compile` in the conformance suite fails with "'x' did not load", which does not say
//! whether the family failed to compile, failed to link, or simply lacks that entry. This prints
//! the compiler's own message per family, which is what identifies a new architecture (sm_120 /
//! Blackwell was the first) rejecting something the older ones accepted.
//!
//! `cargo run --release -p ojas-cuda --example families [-- --entries]`
use ojas_core::kernel::KernelRuntime;

fn main() -> anyhow::Result<()> {
    let show_entries = std::env::args().any(|a| a == "--entries");
    let mut gpu = ojas_cuda::CudaGpu::new(0)?;
    let caps = gpu.caps().clone();
    println!("device tier {:?}, simd {}, f16 {}, features {:?}", caps.tier, caps.simd_width, caps.f16_compute, caps.features);

    let families: Vec<&'static str> = ojas_cuda::kernels::families().map(|(f, _)| f).collect();
    let (mut ok, mut bad) = (0, 0);
    for f in families {
        match gpu.ensure_family(f) {
            Ok(()) => {
                let names: Vec<&str> = ojas_cuda::kernels::all_names().filter(|n| ojas_cuda::kernels::family_of(n) == Some(f)).collect();
                let missing: Vec<&&str> = names.iter().filter(|n| !gpu.has_kernel(n)).collect();
                if missing.is_empty() {
                    println!("  ok   {f:12} {} entries", names.len());
                } else {
                    println!("  PART {f:12} {} entries, {} did not resolve: {:?}", names.len(), missing.len(), &missing[..missing.len().min(6)]);
                    bad += 1;
                    continue;
                }
                if show_entries {
                    for n in names {
                        println!("         {n}");
                    }
                }
                ok += 1;
            }
            Err(e) => {
                println!("  FAIL {f:12} {}", format!("{e:#}").lines().take(6).collect::<Vec<_>>().join(" | "));
                bad += 1;
            }
        }
    }
    println!("{ok} families compiled, {bad} failed");
    if bad > 0 {
        std::process::exit(1);
    }
    Ok(())
}
