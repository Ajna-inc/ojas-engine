//! Speculative-decoding throughput: prompt-lookup vs single-token, same model.
//!
//! The win is workload-shaped. The drafter can only propose text it has already
//! seen, so output that reuses context (summarization, extraction, code edits,
//! JSON, repeated structure, agentic tool loops) accepts several tokens per
//! forward, while novel prose accepts ~1 and falls back to single-token decode.
//! Both are measured here.
//!
//! usage: spec_bench <gguf> [n_tokens]

use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: spec_bench <gguf> [n]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let prec: u8 = std::env::var("OJAS_PREC").ok().and_then(|v| v.parse().ok()).unwrap_or(3);

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;
    let mut e = ojas_infer::EngineCore::new(m);
    e.eos = None;

    // repetitive (drafter hits) vs novel-ish (drafter mostly misses)
    let unit: Vec<u32> = vec![785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13];
    let mut rep: Vec<u32> = Vec::new();
    for _ in 0..6 { rep.extend_from_slice(&unit); }
    let novel: Vec<u32> = (0..60).map(|i| 900u32 + (i * 37) % 4000).collect();

    // Cost of one batched verify vs one single-token forward. Spec only pays off if
    // an M-token verify costs much less than M single forwards.
    {
        let mm = e.model();
        for &bm in &[1usize, 2, 4, 8, 16] {
            let toks: Vec<u32> = (0..bm).map(|i| 100u32 + i as u32).collect();
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t0 = std::time::Instant::now();
                let _ = mm.forward_batch_ids(&toks, 64);
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                if ms < best { best = ms; }
            }
            println!("  verify M={bm:<3} {best:6.2} ms  ({:.2} ms/token if all accepted)", best / bm as f64);
        }
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            let _ = mm.forward_id(100, 64);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if ms < best { best = ms; }
        }
        println!("  single forward {best:6.2} ms");
    }

    for (label, prompt) in [("repetitive", &rep), ("novel", &novel)] {
        let mut best = 0.0f64;
        for _ in 0..3 {
            e.model().reset_state();
            // Time decode only. generate() prefills the prompt first, and counting
            // that understates the rate; reference implementations report decode
            // separately from encode, so this has to match to be comparable.
            let mut t0 = None;
            let mut ntok = 0usize;
            let mut cb = |_t: u32| { if t0.is_none() { t0 = Some(std::time::Instant::now()); } ntok += 1; true };
            let _ = e.generate_with(prompt, n, None, &mut |_, _| {}, &mut cb);
            if let Some(t) = t0 {
                let tps = (ntok.saturating_sub(1)) as f64 / t.elapsed().as_secs_f64();
                if tps > best { best = tps; }
            }
        }
        println!("  {label:<11} {best:7.1} tok/s");
    }
    Ok(())
}
