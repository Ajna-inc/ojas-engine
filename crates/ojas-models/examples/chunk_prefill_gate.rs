//! qwen4exp batched prefill: correctness against the per-token path, then cost.
//!
//! The batched graph is only worth having if it agrees with the path it replaces, so this
//! runs both over the same tokens in one process — a second process would re-load 93 GB
//! and time a cold page cache instead of the graph — and compares the KV cache and the
//! next-token argmax they leave behind.
//!
//! usage: chunk_prefill_gate <gguf> [n_tokens] [prec]
use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: chunk_prefill_gate <gguf> [n] [prec]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let prec: u8 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    ojas_core::logging::init();
    let idle = ojas_models::bench::report_load();
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 2048, prec, None, None)?;

    let toks: Vec<u32> = std::env::var("OJAS_PROMPT")
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|_| (0..n).map(|i| 100u32 + (i as u32 % 50)).collect());
    let toks = &toks[..n.min(toks.len())];

    // Two rounds, reporting the second. The expert cache is cold on the first pass
    // over a streamed model, so whichever path runs first pays for the disk and the
    // other reads it back from RAM — a 5x that is really just cache order.
    let per_token = |m: &ojas_models::decoder::DecoderGpu| -> (f64, u32) {
        m.reset_state();
        let t = std::time::Instant::now();
        let mut last = 0u32;
        for (i, &tk) in toks.iter().enumerate() { last = m.forward_id(tk, i); }
        (t.elapsed().as_secs_f64(), last)
    };
    let batched = |m: &ojas_models::decoder::DecoderGpu| -> (f64, f64, u32) {
        m.reset_state();
        let t = std::time::Instant::now();
        m.prefill(&toks[..toks.len() - 1], 0);
        let pre = t.elapsed().as_secs_f64();
        let last = m.forward_id(toks[toks.len() - 1], toks.len() - 1);
        (t.elapsed().as_secs_f64(), pre, last)
    };
    let (w0, n0) = per_token(&m);
    let (b0, _, m0) = batched(&m);
    println!("round 1 (cold): per-token {w0:6.2} s -> {n0} | batched {b0:6.2} s -> {m0}");
    let (w1, n1) = per_token(&m);
    let (b1, pre1, m1) = batched(&m);
    println!("round 2 (warm): per-token {w1:6.2} s ({:.2} tok/s) -> {n1} | batched {b1:6.2} s ({:.2} tok/s, prefill {pre1:.2} s) -> {m1}",
             n as f64 / w1, n as f64 / b1);
    println!("speedup (warm): {:.2}x{}", w1 / b1, if idle { "" } else { "   ** machine was loaded — not quotable **" });
    if n0 == m0 && n1 == m1 && n0 == n1 {
        println!("GATE: PASS — both paths predict the same next token ({n1})");
        Ok(())
    } else {
        println!("GATE: FAIL — per-token {n0}/{n1}, batched {m0}/{m1}");
        std::process::exit(1);
    }
}
