//! The subcommands: run, chat, bench, tokenize, info.
//!
//! `stream`, `stop_ids`, `check_fits` and `banner` are `pub(crate)` so `ocr`
//! shares them instead of forking the decode loop: a second copy would let the
//! partial-UTF-8 detokenizer, the TTFT accounting and the prefill progress
//! callback drift.

use crate::backend::{with_model, ModelInfo};
use crate::detok::Detok;
use crate::flags::RunOpts;
use anyhow::{bail, Context, Result};
use ojas_core::Model;
use ojas_infer::EngineCore;
use ojas_tokenize::Bpe;
use std::io::{IsTerminal, Write};

/// Wrap a user turn the way this architecture expects, unless asked not to.
fn build_prompt(bpe: &Bpe, info: &ModelInfo, opts: &RunOpts, text: &str) -> Vec<u32> {
    let s = if opts.raw {
        text.to_string()
    } else if opts.system.is_empty() {
        ojas_tokenize::chat_template(&info.arch, text)
    } else {
        ojas_tokenize::chat_transcript(&info.arch, &opts.system, &[("user".into(), text.into())])
    };
    bpe.encode(&s).into_iter().map(|v| v as u32).collect()
}

/// The two ids that can end a turn, as (primary, secondary).
///
/// `EngineCore` holds one stop id, but a templated turn can end at the template's
/// terminator (`<|im_end|>` and friends) or at the GGUF's own EOS, and which one
/// the model reaches is not knowable in advance. The primary goes to the engine;
/// the caller checks the secondary in its token callback.
pub(crate) fn stop_ids(bpe: &Bpe, info: &ModelInfo, raw: bool) -> (Option<u32>, Option<u32>) {
    if raw {
        return (info.eos, None);
    }
    let marker = bpe.encode(ojas_tokenize::chat_eos(&info.arch));
    match marker.as_slice() {
        // Usable only when the template marker is a single token.
        [one] if Some(*one as u32) != info.eos => (Some(*one as u32), info.eos),
        _ => (info.eos, None),
    }
}

pub(crate) fn check_fits(prompt: usize, want: usize, info: &ModelInfo) -> Result<()> {
    if prompt >= info.context {
        bail!(
            "prompt is {prompt} tokens but the context holds {} — raise it with -c/--ctx-size",
            info.context
        );
    }
    if prompt + want > info.context {
        tracing::warn!(
            "prompt {prompt} + {want} requested exceeds context {}; output will be clipped",
            info.context
        );
    }
    Ok(())
}

pub(crate) fn banner(info: &ModelInfo, prompt_tokens: usize) {
    eprintln!(
        "  {} | {} | {} layers | ctx {} | vocab {} | MTP {} | loaded in {:.1}s | prompt {} tok",
        info.arch,
        info.backend,
        info.n_layers,
        info.context,
        info.vocab,
        if info.has_mtp { "yes" } else { "no" },
        info.load_secs,
        prompt_tokens,
    );
}

/// Stream one generation to stdout. Returns (emitted, ttft_s, decode_s).
///
/// `guard` runs after each emitted token and stops generation when it returns
/// false, as reaching an end-of-generation id would; text already produced is
/// still flushed and detokenized cleanly. `run` passes a guard that never fires,
/// `ocr` the n-gram repetition detector.
pub(crate) fn stream(
    core: &EngineCore<&dyn Model>,
    bpe: &Bpe,
    ids: &[u32],
    want: usize,
    opts: &RunOpts,
    also_stop: Option<u32>,
    guard: &mut dyn FnMut(u32) -> bool,
) -> (usize, f64, f64) {
    let t0 = std::time::Instant::now();
    let mut first: Option<std::time::Instant> = None;
    let mut last = t0;
    let mut n = 0usize;
    let mut d = Detok::default();
    let mut out = std::io::stdout();
    let progress = std::io::stderr().is_terminal();

    core.generate_with(
        ids,
        want,
        opts.sampling(),
        &mut |done, total| {
            // Carriage-return progress only works on a terminal; piped or
            // redirected, `\r` erases nothing and the percentages interleave
            // with the captured output.
            if !progress || total == 0 {
                return;
            }
            if done < total {
                eprint!("\r  prompt {}%", done * 100 / total);
            } else {
                eprint!("\r             \r");
            }
            let _ = std::io::stderr().flush();
        },
        &mut |t| {
            if Some(t) == also_stop {
                return false;
            }
            if first.is_none() {
                first = Some(std::time::Instant::now());
            }
            last = std::time::Instant::now();
            n += 1;
            let piece = d.push(bpe, t);
            if !piece.is_empty() {
                let _ = out.write_all(piece.as_bytes());
                let _ = out.flush();
            }
            // After the write, so a guard that stops the run still leaves the
            // token that tripped it visible.
            guard(t)
        },
    );
    let tail = d.finish();
    if !tail.is_empty() {
        let _ = out.write_all(tail.as_bytes());
        let _ = out.flush();
    }

    let ttft = first.map(|f| f.duration_since(t0).as_secs_f64()).unwrap_or(0.0);
    // Decode time spans the n-1 gaps between emitted tokens, excluding the first
    // token's latency, which is prefill rather than decode.
    let decode = first.map(|f| last.duration_since(f).as_secs_f64()).unwrap_or(0.0);
    (n, ttft, decode)
}

fn rate(n: usize, secs: f64) -> f64 {
    if n < 2 || secs <= 0.0 {
        return 0.0;
    }
    (n - 1) as f64 / secs
}

pub fn run(model: &str, opts: &RunOpts, positional: Option<&str>, context: usize) -> Result<()> {
    let text = opts.resolve_prompt(positional)?;
    with_model(model, opts.device, context, opts.precision, |m, bpe, info| {
        let ids = build_prompt(bpe, info, opts, &text);
        check_fits(ids.len(), opts.n_predict, info)?;
        banner(info, ids.len());

        let (primary, secondary) = stop_ids(bpe, info, opts.raw);
        let mut core = EngineCore::new(m);
        core.eos = primary;
        core.eog = info.eog.clone();

        let (n, ttft, decode) =
            stream(&core, bpe, &ids, opts.n_predict, opts, secondary, &mut |_| true);
        // Output printed before the fault came out of valid work; anything after
        // it would not. Fail rather than let a truncated answer look complete.
        if let Some(err) = ojas_core::device_fault::peek() {
            eprintln!();
            anyhow::bail!("device fault after {n} tokens; output is truncated and the session \
                           must be reloaded: {err}");
        }
        eprintln!(
            "\n\n  {n} tokens | first token {ttft:.2}s | {:.2} tok/s",
            rate(n, decode)
        );
        Ok(())
    })
}

pub fn chat(model: &str, opts: &RunOpts, context: usize) -> Result<()> {
    with_model(model, opts.device, context, opts.precision, |m, bpe, info| {
        banner(info, 0);
        eprintln!("  type a message, or /exit to quit, /reset to clear history\n");

        let (primary, secondary) = stop_ids(bpe, info, false);
        let mut core = EngineCore::new(m);
        core.eos = primary;
        core.eog = info.eog.clone();

        let mut turns: Vec<(String, String)> = Vec::new();
        loop {
            eprint!("> ");
            let _ = std::io::stderr().flush();
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).context("reading stdin")? == 0 {
                eprintln!();
                return Ok(()); // EOF, e.g. piped input that ran out
            }
            let line = line.trim();
            match line {
                "" => continue,
                "/exit" | "/quit" => return Ok(()),
                "/reset" => {
                    turns.clear();
                    eprintln!("  history cleared");
                    continue;
                }
                _ => {}
            }

            turns.push(("user".into(), line.to_string()));
            let text = ojas_tokenize::chat_transcript(&info.arch, &opts.system, &turns);
            let ids: Vec<u32> = bpe.encode(&text).into_iter().map(|v| v as u32).collect();
            if ids.len() >= info.context {
                eprintln!(
                    "  history is {} tokens and the context holds {} — /reset to continue",
                    ids.len(),
                    info.context
                );
                turns.pop();
                continue;
            }

            // Capture the reply so the next turn can include it.
            let mut reply = String::new();
            let mut d = Detok::default();
            let mut out = std::io::stdout();
            core.generate_with(
                &ids,
                opts.n_predict,
                opts.sampling(),
                &mut |_, _| {},
                &mut |t| {
                    if Some(t) == secondary {
                        return false;
                    }
                    let piece = d.push(bpe, t);
                    if !piece.is_empty() {
                        reply.push_str(&piece);
                        let _ = out.write_all(piece.as_bytes());
                        let _ = out.flush();
                    }
                    true
                },
            );
            let tail = d.finish();
            if !tail.is_empty() {
                reply.push_str(&tail);
                let _ = out.write_all(tail.as_bytes());
            }
            if let Some(err) = ojas_core::device_fault::peek() {
                println!();
                anyhow::bail!("device fault mid-conversation; the session must be reloaded: {err}");
            }
            println!("\n");
            let _ = out.flush();
            turns.push(("assistant".into(), reply));
        }
    })
}

pub fn bench(model: &str, opts: &RunOpts, positional: Option<&str>, context: usize) -> Result<()> {
    let text = opts
        .resolve_prompt(positional)
        .unwrap_or_else(|_| "Write a short paragraph about the sea.".to_string());
    with_model(model, opts.device, context, opts.precision, |m, bpe, info| {
        let ids = build_prompt(bpe, info, opts, &text);
        check_fits(ids.len(), opts.n_predict, info)?;
        banner(info, ids.len());

        let (primary, secondary) = stop_ids(bpe, info, opts.raw);
        let mut core = EngineCore::new(m);
        core.eos = primary;
        core.eog = info.eog.clone();

        // One unmeasured pass first: the first call pays for page residency and
        // queue warmup, which would otherwise bias the reported rate.
        eprintln!("  warmup...");
        let _ = core.generate_with(&ids, opts.n_predict.min(16), opts.sampling(), &mut |_, _| {}, &mut |t| {
            Some(t) != secondary
        });

        let mut decode_rates = Vec::new();
        let mut ttfts = Vec::new();
        for r in 0..opts.reps.max(1) {
            let t0 = std::time::Instant::now();
            let mut first: Option<std::time::Instant> = None;
            let mut last = t0;
            let mut n = 0usize;
            core.generate_with(&ids, opts.n_predict, opts.sampling(), &mut |_, _| {}, &mut |t| {
                if Some(t) == secondary {
                    return false;
                }
                if first.is_none() {
                    first = Some(std::time::Instant::now());
                }
                last = std::time::Instant::now();
                n += 1;
                true
            });
            let ttft = first.map(|f| f.duration_since(t0).as_secs_f64()).unwrap_or(0.0);
            let decode = first.map(|f| last.duration_since(f).as_secs_f64()).unwrap_or(0.0);
            let tps = rate(n, decode);
            decode_rates.push(tps);
            ttfts.push(ttft);
            eprintln!("  rep {}: {n} tokens | first token {ttft:.2}s | {tps:.2} tok/s", r + 1);
        }

        let median = |mut v: Vec<f64>| -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if v.is_empty() { 0.0 } else { v[v.len() / 2] }
        };
        let lo = decode_rates.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = decode_rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "\n{} | {} | prompt {} tok | {} reps\n  decode  median {:.2} tok/s  range {:.2}-{:.2}\n  first token median {:.2}s",
            info.arch,
            info.backend,
            ids.len(),
            decode_rates.len(),
            median(decode_rates),
            lo,
            hi,
            median(ttfts),
        );
        Ok(())
    })
}

pub fn tokenize(model: &str, opts: &RunOpts, positional: Option<&str>) -> Result<()> {
    let text = opts.resolve_prompt(positional)?;
    let g = ojas_formats::gguf::Gguf::open(model).with_context(|| format!("opening {model}"))?;
    let arch = g.arch();
    let bpe = Bpe::from_gguf(&g);
    let s = if opts.raw { text.clone() } else { ojas_tokenize::chat_template(&arch, &text) };
    let ids = bpe.encode(&s);
    println!("{} tokens", ids.len());
    for id in &ids {
        println!("{id:>7}  {:?}", bpe.decode(*id));
    }
    Ok(())
}

pub fn info(model: &str) -> Result<()> {
    use ojas_formats::gguf::Meta;
    if model.ends_with(".onnx") {
        return info_onnx(model);
    }
    let g = ojas_formats::gguf::Gguf::open(model).with_context(|| format!("opening {model}"))?;
    let arch = g.arch();
    println!("file        {model}");
    println!("arch        {arch}");
    println!("tensors     {}", g.tensors.len());
    println!("shards      {}", g.shard_paths().len());

    let total: u64 = g
        .tensors
        .values()
        .map(|t| t.dims.iter().product::<u64>())
        .sum();
    println!("parameters  {:.2}B", total as f64 / 1e9);

    // Quantization mix. The directory label is not the tensor type, so anything
    // that dispatches on quantization must count the tensors, not read the name.
    let mut by_type: std::collections::HashMap<&'static str, usize> = Default::default();
    for t in g.tensors.values() {
        *by_type.entry(ojas_formats::gguf::gguf_type_name(t.ggml_type)).or_default() += 1;
    }
    let mut mix: Vec<_> = by_type.into_iter().collect();
    mix.sort_by(|a, b| b.1.cmp(&a.1));
    println!(
        "quant mix   {}",
        mix.iter().map(|(n, c)| format!("{n}x{c}")).collect::<Vec<_>>().join(" ")
    );

    println!("\nmetadata");
    let mut keys: Vec<&String> = g.meta.keys().collect();
    keys.sort();
    for k in keys {
        // The token table is tens of thousands of strings; print its size only.
        let v = match &g.meta[k] {
            Meta::U32(v) => v.to_string(),
            Meta::U64(v) => v.to_string(),
            Meta::I32(v) => v.to_string(),
            Meta::F32(v) => v.to_string(),
            Meta::Bool(v) => v.to_string(),
            Meta::Str(s) if s.len() <= 120 => format!("{s:?}"),
            Meta::Str(s) => format!("<{} chars>", s.len()),
            Meta::StrArr(a) => format!("<{} strings>", a.len()),
            Meta::IntArr(a) => format!("<{} ints>", a.len()),
            Meta::FloatArr(a) => format!("<{} floats>", a.len()),
            Meta::Arr => "<array>".into(),
        };
        println!("  {k:<44} {v}");
    }
    Ok(())
}

/// `ojas info model.onnx` — producer, opset, IO shapes, op histogram,
/// parameter bytes. Pure ojas-formats reader; no backend is loaded.
fn info_onnx(model: &str) -> Result<()> {
    use ojas_formats::onnx::{dtype_name, OnnxDim};
    let m = ojas_formats::onnx::load(model).with_context(|| format!("opening {model}"))?;
    println!("file        {model}");
    println!("format      onnx (ir_version {})", m.ir_version);
    println!("producer    {}", if m.producer_name.is_empty() { "<unknown>" } else { &m.producer_name });
    let opsets: Vec<String> = m
        .opsets
        .iter()
        .map(|(d, v)| if d.is_empty() { format!("ai.onnx v{v}") } else { format!("{d} v{v}") })
        .collect();
    println!("opset       {}", opsets.join(", "));
    println!("graph       {}", m.graph.name);
    println!("nodes       {}", m.graph.nodes.len());
    println!("weights     {} tensors, {:.2} MB", m.graph.initializers.len(), m.parameter_bytes() as f64 / 1e6);

    let fmt_dims = |dims: &[OnnxDim]| -> String {
        let parts: Vec<String> = dims
            .iter()
            .map(|d| match d {
                OnnxDim::Value(v) => v.to_string(),
                OnnxDim::Param(p) => p.clone(),
                OnnxDim::Unknown => "?".into(),
            })
            .collect();
        format!("[{}]", parts.join(", "))
    };
    println!("\ninputs");
    for i in &m.graph.inputs {
        println!("  {:<28} {} {}", i.name, dtype_name(i.elem_type), fmt_dims(&i.dims));
    }
    println!("outputs");
    for o in &m.graph.outputs {
        println!("  {:<28} {} {}", o.name, dtype_name(o.elem_type), fmt_dims(&o.dims));
    }
    println!("\nops");
    for (op, count) in m.op_histogram() {
        println!("  {op:<28} {count}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info_for(arch: &str, eos: Option<u32>) -> ModelInfo {
        ModelInfo {
            arch: arch.into(),
            eos,
            eog: eos.into_iter().collect(),
            vocab: 100,
            context: 128,
            backend: "cpu",
            has_mtp: false,
            n_layers: 1,
            hidden_dim: 1,
            load_secs: 0.0,
        }
    }

    #[test]
    fn a_prompt_at_or_past_the_context_is_rejected() {
        let info = info_for("qwen3", None);
        assert!(check_fits(128, 1, &info).is_err(), "prompt == context must fail");
        assert!(check_fits(200, 1, &info).is_err());
        assert!(check_fits(64, 8, &info).is_ok());
    }

    /// Overlong requests warn and clip rather than fail: the engine already clamps
    /// `max_tokens` to the remaining room.
    #[test]
    fn a_fitting_prompt_with_an_overlong_request_is_allowed() {
        let info = info_for("qwen3", None);
        assert!(check_fits(120, 999, &info).is_ok());
    }

    #[test]
    fn decode_rate_measures_intervals_not_tokens() {
        // 5 tokens across 2 seconds is 4 intervals -> 2 tok/s.
        assert!((rate(5, 2.0) - 2.0).abs() < 1e-9);
        // A single token has no interval to measure, so it has no rate.
        assert_eq!(rate(1, 1.0), 0.0);
        assert_eq!(rate(0, 1.0), 0.0);
        assert_eq!(rate(5, 0.0), 0.0, "zero elapsed must not divide");
    }
}
