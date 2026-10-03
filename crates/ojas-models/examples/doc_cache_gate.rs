//! Document-cache gate: how close a page reused at a new position comes to
//! processing it, and what it saves.
//!
//! Each page is first read after system prompt A, which stores it in the document
//! cache. Then it is asked about after a different system prompt B, twice: processed
//! in full, and with the page served from the cache. The reused run is scored by
//! teacher-forced agreement: along the fresh run's answer, the share of positions
//! where it predicts the same next token. The gate also reports the reused run's
//! greedy answer and the prompt-processing time both ways.
//!
//! usage: doc_cache_gate <gguf> [prec] [min agreement, default 0.95]
//! Tune with OJAS_DOC_RECOMPUTE and OJAS_DOC_TAIL.

use anyhow::{ensure, Result};
use ojas_core::Model;
use std::time::Instant;

type Turns = Vec<(String, String)>;

const SYSTEM_A: &str = "You are a careful assistant inside a web browser. Answer questions about the page \
accurately and briefly, quote numbers exactly, and say so when the page does not contain the answer.";
const SYSTEM_B: &str = "You are a research assistant. Read the document the user shares and answer in plain \
sentences. Cite the section a number comes from, and never guess a value the document does not state.";

fn page(topic: &str, seed: usize) -> String {
    (0..45).map(|i| format!(
        "Section {i}: the {topic} report lists item {} at {} units, up {}% on the prior period, recorded by \
         team {} in region {}.\n",
        (i * 7 + seed) % 97, (i * 131 + seed * 17) % 1000, (i + seed) % 23, (i + seed) % 9, (i * 3 + seed) % 5)).collect()
}

/// A prompt, all but its last token processed: returns the time taken.
fn process(model: &dyn Model, ids: &[u32], docs: &[(usize, usize)]) -> f64 {
    let t0 = Instant::now();
    let pre = &ids[..ids.len() - 1];
    model.set_prefix_docs(docs, true);
    let start = model.reuse_prefix_len(pre);
    if start < pre.len() { model.prefill(&pre[start..], start); }
    t0.elapsed().as_secs_f64()
}

/// Greedy continuation of a processed prompt.
fn continue_greedy(model: &dyn Model, ids: &[u32], n: usize, stop: &[u32]) -> Vec<u32> {
    let (mut cur, mut pos, mut out) = (*ids.last().unwrap(), ids.len() - 1, Vec::new());
    while out.len() < n {
        cur = model.forward_id(cur, pos);
        pos += 1;
        out.push(cur);
        if stop.contains(&cur) { break; }
    }
    out
}

/// Share of `answer`'s tokens the processed prompt predicts as its argmax, feeding
/// the answer itself.
fn agreement(model: &dyn Model, ids: &[u32], answer: &[u32]) -> f64 {
    let (mut cur, mut pos, mut same) = (*ids.last().unwrap(), ids.len() - 1, 0);
    for &want in answer {
        let logits = model.forward_logits(cur, pos).expect("this model exposes logits");
        let top = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap() as u32;
        same += usize::from(top == want);
        cur = want;
        pos += 1;
    }
    same as f64 / answer.len().max(1) as f64
}

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: doc_cache_gate <gguf> [prec] [min agreement]");
    let prec: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let min_agreement: f64 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0.95);
    if std::env::var("OJAS_DOC_CACHE_GB").is_err() { std::env::set_var("OJAS_DOC_CACHE_GB", "2"); }
    ojas_core::logging::init();
    let cfg = ojas_core::config::EngineConfig::current();
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let arch = g.arch();
    let bpe = ojas_tokenize::tokenizer::Bpe::from_gguf(&g);
    let stop = ojas_tokenize::tokenizer::eog_token_ids(&g, &arch);
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 8192, prec, None, None)?;
    let model: &dyn Model = &m;
    ensure!(model.prefix_cache_stats().is_some_and(|s| s.docs.budget > 0),
        "this model and precision do not use the document cache");
    // Exact prefix reuse would restore a repeated prompt whole; the gate measures the
    // document cache alone.
    model.set_prefix_reuse(false);
    let tokens = |text: &str| -> Vec<u32> { bpe.encode(text).into_iter().map(|v| v as u32).collect() };
    let encode = |system: &str, turns: &Turns| -> (Vec<u32>, (usize, usize)) {
        let ids = tokens(&ojas_tokenize::tokenizer::chat_transcript(&arch, system, turns));
        (ids, ojas_tokenize::transcript_span(&arch, system, turns, 0, tokens))
    };
    println!("  {arch} | prec {prec} | recompute {} tokens, replay {:.0}% of the document\n",
        cfg.doc_recompute, cfg.doc_tail * 100.0);

    let mut failures = Vec::new();
    let mut scores = Vec::new();
    for (i, topic) in ["sales", "support", "hiring", "travel"].iter().enumerate() {
        let doc = page(topic, i);
        let read: Turns = vec![("user".into(), doc.clone()), ("user".into(), "Which item has the largest value?".into())];
        let ask: Turns = vec![("user".into(), doc), ("user".into(), "List the three items with the smallest values.".into())];
        let (first, span_a) = encode(SYSTEM_A, &read);
        process(model, &first, &[span_a]);
        let (ids, span) = encode(SYSTEM_B, &ask);

        let fresh_s = process(model, &ids, &[]);
        let fresh = continue_greedy(model, &ids, 64, &stop);
        let fresh_check = { process(model, &ids, &[]); agreement(model, &ids, &fresh) };

        let reused_s = process(model, &ids, &[span]);
        let reused_tokens = model.prefix_cache_stats().map_or(0, |s| s.last.doc_reused_tokens);
        let reused = continue_greedy(model, &ids, 64, &stop);
        process(model, &ids, &[span]);
        let score = agreement(model, &ids, &fresh);
        let same = reused.iter().zip(&fresh).take_while(|(a, b)| a == b).count();
        println!("  {topic:<8} doc {:>4} tokens, {reused_tokens:>4} from the cache | prompt {fresh_s:.2}s fresh, \
                  {reused_s:.2}s reused | agreement {:.1}% | greedy answers match for {same}/{} tokens",
            span.1 - span.0, score * 100.0, fresh.len());
        if fresh_check < 1.0 { failures.push(format!("{topic}: the fresh run does not reproduce itself")); }
        if reused_tokens == 0 { failures.push(format!("{topic}: the document was not reused")); }
        scores.push(score);
    }
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    println!("\n  mean agreement {:.1}% (gate: {:.0}%)", mean * 100.0, min_agreement * 100.0);
    if mean < min_agreement { failures.push(format!("agreement {:.1}% is below the gate", mean * 100.0)); }
    for f in &failures { println!("  FAIL {f}"); }
    ensure!(failures.is_empty(), "GATE: DOC CACHE FAIL ({} problems)", failures.len());
    println!("\nGATE: DOC CACHE PASS");
    Ok(())
}
