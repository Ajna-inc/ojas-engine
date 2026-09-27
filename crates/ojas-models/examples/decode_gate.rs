//! Gate: decode must stay token-identical to the last recorded run.
//!
//! Catches regressions — whether a change moved a token — against a recorded token
//! sequence rather than a second engine, so it is model-agnostic, needs no sibling
//! repo, and does not load a second copy of the weights.
//!
//! Missing baselines fail. `OJAS_REGOLD=1` records one; use it only when a token
//! change is intended and understood, and say why in the commit.
//!
//! Generation stops at the model's EOS token. Running the full `n_tok` regardless left
//! post-EOS continuation in the baseline (a model finishing its turn at token 7
//! contributed 13 more tokens), and that region is out-of-distribution and numerically
//! chaotic: on the Qwen2.5-0.5B quant sweep IQ2_M / IQ3_M / Q2_K all failed purely
//! there while agreeing on every real token. The comparison covers only tokens the
//! model meant to emit.
//!
//! usage: decode_gate <gguf> [n_tokens] [prec]

use anyhow::Result;
use ojas_core::Model;
use std::path::PathBuf;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: decode_gate <gguf> [n] [prec]");
    let n_tok: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let prec: u8 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    // OJAS_PROMPT overrides the built-in token ids. The default is a Qwen2.5 chat
    // prompt, which is meaningless to a model with a different vocabulary: Qwen3.8 has
    // 248320 tokens, where 9707 is ".Q" and 2585 is "ru", and feeding it the default
    // produces punctuation-only output that reads like an engine bug.
    let prompt_override: Option<Vec<u32>> = match std::env::var("OJAS_PROMPT") {
        Ok(v) => Some(v.split(',').map(|t| t.trim().parse()).collect::<std::result::Result<Vec<_>, _>>()?),
        Err(_) => None,
    };

    anyhow::ensure!(n_tok > 0, "output length must be nonzero");
    ojas_core::logging::init();   // RUST_LOG=arch=info to see the derived hparams
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    // Absent metadata leaves `eos` as None, which runs the full `n_tok` — the right
    // fallback for a GGUF that declares no EOS.
    let eos = g.meta_u32("tokenizer.ggml.eos_token_id");
    // Resolve the default prompt against this model's vocabulary: the Qwen2.5 ids are
    // out of range for surya-2 (qwen35, 65425 tokens) and panicked with "prefill exceeds
    // model bounds", which left that OCR model with no golden baseline while every
    // quantization tier was tuned against it. `vocab` comes from token_embd's row count
    // because that is what `arch.vocab`, and so the prefill bounds check, is derived
    // from; the tokens array can be absent, and reading it instead would let the panic
    // back in.
    let vocab = g.tensors.get("token_embd.weight")
        .and_then(|t| t.dims.get(1).copied()).unwrap_or(u64::MAX) as usize;
    let prompt: Vec<u32> = prompt_override.unwrap_or_else(|| {
        let qwen25 = vec![151644u32, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198];
        if qwen25.iter().all(|&t| (t as usize) < vocab) { qwen25 }
        // Every id below must be in range for any vocab this falls back to.
        else {
            // A digit run rather than a chat prefix, chosen by measurement: a ChatML
            // prefix makes an OCR model emit EOS after one token (no image, nothing to
            // say), and a 1-token baseline gates almost nothing. A digit run continues
            // for the full n_tok, so 20 decode steps are compared. Low ids only — the
            // Qwen2.5 role id 77091 ("assistant") exceeds surya-2's 65425 and would
            // reintroduce the panic this fallback exists to avoid.
            vec![14, 15, 16, 17, 18, 19]
        }
    });
    anyhow::ensure!(!prompt.is_empty(), "prompt must be nonempty");
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 2048, prec, None, None)?;
    m.prefill(&prompt[..prompt.len() - 1], 0);
    let mut got = Vec::with_capacity(n_tok);
    let mut t = prompt[prompt.len() - 1];
    let mut hit_eos = false;
    for i in 0..n_tok {
        t = m.forward_id(t, prompt.len() - 1 + i);
        got.push(t);
        // Keep the EOS token itself: it is a real prediction, and dropping it would hide
        // a model that stops in the wrong place.
        if Some(t) == eos { hit_eos = true; break; }
    }

    // Key on the model file plus the shape of the run, so several models can be gated
    // from the same checkout without colliding.
    let stem = PathBuf::from(&gguf)
        .file_stem().map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    // A custom prompt gets its own key, hashed from OJAS_PROMPT: a (model, n, prec) key
    // alone let a run with OJAS_PROMPT set compare against — and overwrite — the
    // default-prompt baseline, reporting a mismatch that was only a different input.
    // Default-prompt keys are unchanged, so existing baselines keep their names.
    let key = match std::env::var("OJAS_PROMPT") {
        Ok(v) => {
            let mut h: u64 = 0xcbf29ce484222325;
            for b in v.trim().as_bytes() { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
            format!("{stem}-n{n_tok}-p{prec}-{:08x}", h as u32)
        }
        Err(_) => format!("{stem}-n{n_tok}-p{prec}"),
    };
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/decode");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{key}.txt"));

    let line = got.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
    let regold = std::env::var("OJAS_REGOLD").as_deref() == Ok("1");
    let prev = std::fs::read_to_string(&path).ok().map(|s| s.trim().to_string());

    match prev {
        Some(want) if !regold => {
            println!("want: {want}");
            println!("got:  {line}");
            if want == line {
                    println!("GATE: TOKEN-IDENTICAL PASS ({} tokens{}, {key})",
                    got.len(), if hit_eos { " to EOS" } else { "" });
            } else {
                let w: Vec<&str> = want.split(',').collect();
                let gt: Vec<&str> = line.split(',').collect();
                let at = w.iter().zip(&gt).position(|(a, b)| a != b);
                println!("GATE: TOKEN MISMATCH FAIL (first diff at {at:?}, {key})");
                std::process::exit(1);
            }
        }
        _ => {
            anyhow::ensure!(regold, "missing decode baseline {}; explicitly set OJAS_REGOLD=1 to record", path.display());
            std::fs::write(&path, format!("{line}\n"))?;
            println!("got:  {line}");
            println!("GATE: RECORDED {} ({} tokens{})", path.display(),
                got.len(), if hit_eos { " to EOS" } else { "" });
        }
    }
    Ok(())
}
