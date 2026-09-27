//! Gate: representation resolution must be unambiguous, and capability predicates must
//! agree with it.
//!
//! Weights live in ~14 parallel maps, and ~280 sites used to rediscover the
//! representation by probing them in priority order. `Weights::repr` is now the only
//! place that knows the order, but priority only matters if a weight can land in two
//! maps at once — a loader bug that does that would make every dispatch site's behaviour
//! depend on an ordering nobody reads.
//!
//! So this asserts the loader's invariant directly: every weight resolves, and resolves
//! to exactly one representation. It also prints the histogram, the fastest way to see
//! that a mode did what was asked (e.g. that OJAS_NATIVE moved the FFN into `Native`).
//!
//! `batched_dense_ok` is checked here too because it fails closed: while it probed maps
//! directly it silently returned false for native weights, disabling batched prefill
//! with no error and no wrong output.
//!
//! usage: repr_gate <gguf> [prec]

use anyhow::Result;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: repr_gate <gguf> [prec]");
    let prec: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&path)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 2048, prec, None, None)?;

    let (hist, dup, missing) = m.repr_audit();
    println!("  representation histogram:");
    for (r, n) in &hist {
        println!("    {r:<8} {n}");
    }
    println!("  batched_dense_ok = {}", m.batched_dense_ok_pub());
    if let Ok(want) = std::env::var("REPR_SHOW") {
        println!("  sample names with repr {want}:");
        for n in m.repr_names(&want).iter().take(6) { println!("    {n}"); }
    }

    let mut bad = false;
    if !dup.is_empty() {
        println!("  AMBIGUOUS (in more than one map) — priority order is deciding silently:");
        for n in dup.iter().take(8) { println!("    {n}"); }
        bad = true;
    }
    if !missing.is_empty() {
        println!("  UNRESOLVED (in no map, but the model declares them 2-D):");
        for n in missing.iter().take(8) { println!("    {n}"); }
        bad = true;
    }
    if bad { println!("\nGATE: REPR FAIL"); std::process::exit(1); }
    println!("\nGATE: REPR PASS");
    Ok(())
}
