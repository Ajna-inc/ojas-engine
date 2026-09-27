//! Investigate the 48 GiB pooled-expert corruption.
//!
//! usage: flash_pool_audit model [tokens=24]
//!
//! Generates greedily, then audits the expert cache against the model file and
//! checks live pooled allocations for overlap. Run it at a cache size that is
//! known good (16/32 GiB) and at the size that is known bad (48 GiB, which needs
//! OJAS_UNSAFE_POOL_OVERRIDE=1) and compare.
//!
//! Outcomes:
//!   corrupt entries  -> admission or lifetime: cached bytes stopped matching disk
//!   overlaps         -> two live allocations share GPU address space
//!   neither          -> the bytes are right, so look at addressing or
//!                       synchronisation instead (command-buffer status, the
//!                       address tables, or the kernel's own indexing)
use anyhow::{ensure, Result};
use ojas_core::Model;
use ojas_models::decoder::DecoderGpu;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() >= 2, "usage: flash_pool_audit model [tokens]");
    let want: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(24);
    ojas_core::logging::init();

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&args[1])?;
    let arch = g.arch();
    ensure!(arch == "qwen4exp", "this audit targets Flash (qwen4exp), got {arch}");
    let bpe = ojas_tokenize::Bpe::from_gguf(&g);
    let prompt: Vec<u32> = bpe
        .encode("The capital of France is")
        .into_iter()
        .map(|v| v as u32)
        .collect();

    let m = DecoderGpu::load(&gpu, &mut g, 2048, 4, None, None)?;

    // Audit before any decode: whatever the loader admitted should already match.
    let before = m.audit_expert_cache();
    println!(
        "after load : entries={} corrupt={} overlaps={} unresolved={} checked={:.2} GB",
        before.entries,
        before.corrupt.len(),
        before.overlaps.len(),
        before.unresolved,
        before.bytes_checked as f64 / 1e9
    );

    m.reset_session();
    let engine = ojas_infer::EngineCore::new(&m);
    let out = engine.generate(&prompt, want, true);

    let after = m.audit_expert_cache();
    println!(
        "after {:>3} tok: entries={} corrupt={} overlaps={} unresolved={} checked={:.2} GB",
        out.len(),
        after.entries,
        after.corrupt.len(),
        after.overlaps.len(),
        after.unresolved,
        after.bytes_checked as f64 / 1e9
    );

    for c in after.corrupt.iter().take(8) {
        println!(
            "  CORRUPT layer={} expert={} kind={} len={} first_diff={} cached=0x{:02x} source=0x{:02x}",
            c.layer, c.expert, c.kind, c.len, c.first_diff, c.cached, c.source
        );
    }
    for (a, b) in after.overlaps.iter().take(8) {
        println!("  OVERLAP keys 0x{a:x} and 0x{b:x}");
    }

    println!("first 12 tokens: {:?}", &out[..out.len().min(12)]);
    println!(
        "verdict: {}",
        if after.ok() {
            "cache bytes match the file and no allocations overlap — look at addressing/synchronisation"
        } else {
            "cache contents are wrong — admission or lifetime"
        }
    );
    Ok(())
}
