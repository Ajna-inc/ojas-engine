//! What an M-token verify costs against a single forward.
//!
//! Speculation buys tokens at the price of a wider verify. On an MoE that price
//! is not flat: every extra row widens the routed-expert union, and the gather
//! pays for it. If the curve is linear in M there is nothing to amortize and
//! drafting deeper cannot help, however good the draft head is.
//!
//! usage: verify_cost <gguf> [prec]
use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: verify_cost <gguf> [prec]");
    let prec: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    ojas_core::logging::init();
    let idle = ojas_models::bench::report_load();
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;
    let ids: Vec<u32> = (0..24).map(|i| 100u32 + (i as u32 % 40)).collect();

    m.reset_state();
    m.prefill(&ids, 0);
    let base = ids.len();
    // warm: first pass pays cold expert misses that no later pass sees
    let _ = m.forward_id(ids[0], base);

    let best = |f: &dyn Fn()| -> f64 {
        let mut v: Vec<f64> = (0..3).map(|_| {
            let t = std::time::Instant::now(); f(); t.elapsed().as_secs_f64()
        }).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[0] * 1e3
    };
    let single = best(&|| { m.forward_id(ids[0], base); });
    println!("\n  single forward        {single:7.1} ms");
    println!("  {:>3}  {:>10}  {:>8}  {:>14}", "M", "verify ms", "vs 1x", "ms per token");
    for mm in [2usize, 3, 4, 6, 8] {
        let batch: Vec<u32> = (0..mm).map(|i| ids[i % ids.len()]).collect();
        let ms = best(&|| { m.verify_cost_probe(&batch, base); });
        println!("  {mm:>3}  {ms:>10.1}  {:>7.2}x  {:>13.1}", ms / single, ms / mm as f64);
    }
    if !idle { println!("\n  ** machine was loaded — ratios hold better than absolutes **"); }
    Ok(())
}
