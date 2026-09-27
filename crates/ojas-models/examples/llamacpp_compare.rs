//! ojas in llama-bench's units, so the two can be put side by side.
//!
//! llama-bench reports pp<N> (prompt processing: N tokens through prefill) and
//! tg<M> (text generation: M tokens decoded one at a time), both as tok/s. This runs the
//! same two phases so the numbers mean the same thing. It does not tokenize — token ids
//! are synthetic — because llama-bench does not either.
//!
//! usage: llamacpp_compare <gguf> [n_prompt] [n_gen] [prec]
use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: llamacpp_compare <gguf> [pp] [tg] [prec]");
    let pp: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let tg: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    let prec: u8 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    let reps: usize = std::env::var("OJAS_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    anyhow::ensure!(pp > 0 && tg > 0 && reps > 0, "prompt, generation and repetitions must be nonzero");
    ojas_core::logging::init();
    let idle = ojas_models::bench::report_load();
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let ctx = (pp + tg + 64).max(2048);
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, ctx, prec, None, None)?;

    // Synthetic ids inside the vocabulary, as llama-bench does.
    let toks: Vec<u32> = (0..pp).map(|i| 100u32 + (i as u32 % 64)).collect();

    let prefill_once = || -> f64 {
        m.reset_state();
        let t = std::time::Instant::now();
        m.prefill(&toks, 0);
        t.elapsed().as_secs_f64()
    };
    let decode_once = || -> f64 {
        m.reset_state();
        m.prefill(&toks[..1], 0);
        let mut cur = toks[0];
        let t = std::time::Instant::now();
        for i in 0..tg { cur = m.forward_id(cur, 1 + i); }
        t.elapsed().as_secs_f64()
    };

    // Warm up both phases first: on a streamed model the first pass pays cold expert
    // misses, so charging them to whichever phase runs first inverts the A/B.
    let _ = prefill_once();
    let _ = decode_once();

    let median = |f: &dyn Fn() -> f64| -> (f64, f64) {
        let mut v: Vec<f64> = (0..reps).map(|_| f()).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ({ let mid = v.len()/2; if v.len()%2 == 0 { (v[mid-1]+v[mid])/2.0 } else { v[mid] } }, (v[v.len() - 1] - v[0]) / v[0] * 100.0)
    };
    let (p_s, p_spread) = median(&prefill_once);
    let (d_s, d_spread) = median(&decode_once);

    println!("warm median across {reps} repetitions; synthetic token workload");
    println!();
    println!("  model      {}", std::path::Path::new(&gguf).file_name().unwrap().to_string_lossy());
    println!("  pp{pp:<8}  {:8.2} tok/s   (spread +{p_spread:.0}%)", pp as f64 / p_s);
    println!("  tg{tg:<8}  {:8.2} tok/s   (spread +{d_spread:.0}%)", tg as f64 / d_s);
    if !idle || p_spread.max(d_spread) > 15.0 {
        println!("  ** NOT QUOTABLE: {} **",
            if !idle { "machine was loaded" } else { "spread > 15%" });
    }
    Ok(())
}
