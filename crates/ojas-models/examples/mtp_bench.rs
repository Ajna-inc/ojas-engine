//! Self-speculative decode with the model's own MTP head vs plain decode.
//!
//! Qwen3.8 ships a NextN/MTP draft block (`nextn_predict_layers = 1`): a
//! next-next-token head sharing the backbone, so it is the intended drafter for this
//! model. Against a separate small draft model it needs no second KV cache and no
//! second set of weights resident, and it agrees with the target more often, being the
//! target's own head.
//!
//! One `mtp_step` is a single command buffer that drafts and then verifies
//! [cur, draft] through the main stack, so a rejected draft costs the same forward the
//! plain path would have done anyway, and an accepted one yields two tokens for that
//! one forward.
//!
//! Reports acceptance rate alongside throughput, since the speedup is a function of it.
//!
//! usage: mtp_bench <gguf> [n_tokens] [prec]

use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: mtp_bench <gguf> [n] [prec]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(48);
    let prec: u8 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let idle = ojas_models::bench::report_load();
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&path)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;
    if !m.has_mtp() {
        eprintln!("this GGUF ships no MTP draft block — nothing to measure");
        std::process::exit(2);
    }

    // OJAS_PROMPT: token ids in this model's vocabulary. The default below is a Qwen2.5
    // prompt and is nonsense to anything else — fed to Qwen3.8 (248320 tokens, where
    // 9707 is ".Q") it produces punctuation-only output that reads like an engine bug.
    let prompt: Vec<u32> = match std::env::var("OJAS_PROMPT") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => vec![9707, 0, 2585, 646, 358, 7789, 498, 3351, 30],
    };

    // Repeat each phase and take the best. A streamed model's per-token cost swings with
    // page-cache and expert-cache state: single-shot runs of this bench reported
    // 282..448 ms/token for configurations whose gather hit rate was byte-identical, a
    // spread wide enough to invert the speedup it reports. OJAS_BENCH_REPS=1 is
    // single-shot.
    let reps: usize = std::env::var("OJAS_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);

    // ---- baseline: plain single-token decode ----
    let plain_once = |m: &ojas_models::decoder::DecoderGpu| -> f64 {
        m.reset_state();
        m.prefill(&prompt[..prompt.len() - 1], 0);
        let mut pos = prompt.len() - 1;
        let mut cur = prompt[prompt.len() - 1];
        let t = std::time::Instant::now();
        for _ in 0..n { cur = m.forward_id(cur, pos); pos += 1; }
        t.elapsed().as_secs_f64()
    };
    // Warmup: the first pass pays cold-start expert misses no later pass sees, and
    // charging them to whichever phase runs first once made this bench report a 4.9x
    // that was really cache order.
    let _ = plain_once(&m);
    m.gather_stats_reset();
    let mut plain_all: Vec<f64> = (0..reps).map(|_| plain_once(&m)).collect();
    let (ph, pr, _) = m.gather_stats();
    plain_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let plain = plain_all[0];
    let plain_spread = (plain_all[plain_all.len() - 1] - plain) / plain * 100.0;

    // ---- MTP self-speculation ----
    let mut spec_all: Vec<f64> = Vec::with_capacity(reps);
    let (mut end_cur, mut end_pos) = (0u32, 0usize);
    let (mut emitted, mut steps, mut accepted) = (0usize, 0usize, 0usize);
    let (mut t_step, mut t_roll) = (0.0f64, 0.0f64);
    m.gather_stats_reset();
    for rep in 0..reps {
    if rep > 0 { emitted = 0; steps = 0; accepted = 0; t_step = 0.0; t_roll = 0.0; }
    m.reset_state();
    m.prefill(&prompt[..prompt.len() - 1], 0);
    let mut pos = prompt.len() - 1;
    let mut cur = prompt[prompt.len() - 1];
    let t1 = std::time::Instant::now();
    // Verify leaves row 0 = hidden at pos, row 1 = hidden at pos+1; the draft needs
    // the row for its own predecessor, so an accepted step reads row 1.
    let mut hrow = 0usize;
    while emitted < n {
        let ts = std::time::Instant::now();
        // one command buffer: draft, then verify [cur, draft] through the stack
        let (a0, a1, draft) = m.mtp_step(cur, pos, hrow);
        t_step += ts.elapsed().as_secs_f64();
        emitted += 1;
        steps += 1;
        if draft == a0 {
            // the draft matched, so its verify row is valid and that forward produced
            // two tokens
            emitted += 1;
            accepted += 1;
            pos += 2;
            cur = a1;
            hrow = 1;
        } else {
            // rejected: only position `pos` is committed. KV needs no rollback
            // (the next verify overwrites the same slots) but the SSM/conv state
            // advanced through the draft and must be restored.
            let tr = std::time::Instant::now();
            m.mtp_rollback();
            t_roll += tr.elapsed().as_secs_f64();
            pos += 1;
            cur = a0;
            hrow = 0;
        }
    }
    spec_all.push(t1.elapsed().as_secs_f64());
    (end_cur, end_pos) = (cur, pos);
    }
    let (sh, sr, cached) = m.gather_stats();
    spec_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let spec = spec_all[0];
    let spec_spread = (spec_all[spec_all.len() - 1] - spec) / spec * 100.0;

    // Draft vs verify: which half a losing step loses to. One sample, taken after the
    // loop at a longer position, so it does not reconcile with the mtp_step average
    // above and is only good for the ratio between the two.
    let td = std::time::Instant::now();
    let d1 = m.mtp_draft(end_cur, end_pos, 0, true);
    let draft_ms = td.elapsed().as_secs_f64() * 1e3;
    let tv = std::time::Instant::now();
    let _ = m.mtp_verify(end_cur, d1, end_pos);
    let verify_ms = tv.elapsed().as_secs_f64() * 1e3;

    println!("  tokens                {n}   (best of {reps})");
    println!("  plain decode          {:7.2} tok/s   ({:.2} ms/token, spread +{plain_spread:.0}%)", n as f64 / plain, plain * 1e3 / n as f64);
    println!("  MTP speculative       {:7.2} tok/s   ({:.2} ms/token, spread +{spec_spread:.0}%)", emitted as f64 / spec, spec * 1e3 / emitted as f64);
    if plain_spread.max(spec_spread) > 15.0 || !idle {
        println!("  ** do not quote the speedup: {} **",
            if !idle { "machine was loaded" } else { "spread > 15%, the machine moved under the run" });
    }
    println!("  speedup               {:7.3}x", (emitted as f64 / spec) / (n as f64 / plain));
    println!("  acceptance            {:7.1}%  ({accepted} of {steps} drafts)", accepted as f64 * 100.0 / steps as f64);
    println!("  tokens per forward    {:7.3}", emitted as f64 / steps as f64);
    println!("  -- where the time goes --");
    println!("  mtp_step              {:7.1} ms/call  ({:.0}% of total)", t_step * 1e3 / steps as f64, t_step / spec * 100.0);
    println!("  mtp_rollback          {:7.1} ms/call  ({:.0}% of total)", t_roll * 1e3 / (steps - accepted).max(1) as f64, t_roll / spec * 100.0);
    println!("  plain forward         {:7.1} ms       (for reference)", plain * 1e3 / n as f64);
    println!("  draft block           {draft_ms:7.1} ms   (1 sample)");
    println!("  verify (2 tokens)     {verify_ms:7.1} ms   (1 sample)");
    // Acceptance at which one step beats one plain forward. Above it speculation
    // pays; below it the step is pure overhead.
    let step_ms = t_step * 1e3 / steps as f64;
    println!("  break-even acceptance {:7.1}%", (step_ms / (plain * 1e3 / n as f64) - 1.0) * 100.0);
    // A streamed model's throughput tracks its expert-gather hit rate: a plain decode
    // and a 2-token verify read the same experts and differ only in how many each one
    // misses.
    println!("  -- expert gather --");
    println!("  plain hit rate        {:7.1}%  ({ph} of {pr} reads)", ph as f64 * 100.0 / pr.max(1) as f64);
    println!("  spec  hit rate        {:7.1}%  ({sh} of {sr} reads)", sh as f64 * 100.0 / sr.max(1) as f64);
    println!("  LRU resident          {:7.1} GB", cached as f64 / (1u64 << 30) as f64);
    Ok(())
}
