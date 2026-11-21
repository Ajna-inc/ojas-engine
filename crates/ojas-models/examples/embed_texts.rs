//! Embed texts with a text-encoder GGUF on Metal: one unit-length vector per text,
//! pooled the way the file declares (`pooling_type`: mean, first token or last token)
//! and passed through its dense maps, many texts per GPU pass.
//!
//! usage: embed_texts <model.gguf> <texts.jsonl> <out.f16>
//!            [--prompt TEXT | --prompt-name NAME] [--dim N]
//!            [--max-tokens 256] [--pass-tokens 8192]
//!
//! Each line of the input is a JSON string, or an object with a `text` field. A prompt
//! is prepended to every text before tokenization and pooled with it; `--prompt-name`
//! looks one up among the prompts the file carries (`<arch>.embedding.prompt_names`,
//! e.g. `Document`, `QuestionAnswering`). `--dim` keeps the first N values of each
//! vector before normalizing (Matryoshka truncation). `--max-tokens` bounds a text's
//! tokens, the special tokens included.
//!
//! The output is the rows back to back as little-endian f16, `width` values each; the
//! width and count are printed and written beside it as `<out>.json`.

use anyhow::{bail, Context, Result};
use ojas_decision::json::Json;
use ojas_formats::gguf::{Gguf, Meta};
use ojas_models::decoder::DecoderGpu;
use ojas_tokenize::Bpe;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (model_path, input, output) = (args.next().context("model")?, args.next().context("texts")?, args.next().context("output")?);
    let (mut max_tokens, mut pass_tokens) = (256usize, 8192usize);
    let (mut prompt, mut prompt_name, mut dim) = (None::<String>, None::<String>, None::<usize>);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--max-tokens" => max_tokens = args.next().context("--max-tokens N")?.parse()?,
            "--pass-tokens" => pass_tokens = args.next().context("--pass-tokens N")?.parse()?,
            "--prompt" => prompt = Some(args.next().context("--prompt TEXT")?),
            "--prompt-name" => prompt_name = Some(args.next().context("--prompt-name NAME")?),
            "--dim" => dim = Some(args.next().context("--dim N")?.parse()?),
            other => bail!("unknown argument {other}"),
        }
    }
    let mut g = Gguf::open(&model_path)?;
    if let Some(name) = &prompt_name {
        prompt = Some(named_prompt(&g, name)?);
    }
    let prompt = prompt.unwrap_or_default();
    let tok = Bpe::from_gguf(&g);
    let specials = Specials::from_gguf(&g)?;
    g.sidecars = false;
    let gpu = ojas_metal::MetalGpu::new()?;
    let enc = DecoderGpu::load(&gpu, &mut g, 64, 0, None, None)?;
    let full = enc.text_embedding_width().context("not a text encoder")?;
    let width = dim.unwrap_or(full);
    anyhow::ensure!(width > 0 && width <= full, "--dim {width} outside 1..={full}");
    anyhow::ensure!(max_tokens > specials.count(), "--max-tokens {max_tokens} leaves no room for text");

    let texts: Vec<String> = std::fs::read_to_string(&input)?.lines().filter(|l| !l.trim().is_empty()).map(|l| {
        match Json::parse(l) {
            Ok(Json::Str(s)) => s,
            Ok(j) => j.get("text").and_then(Json::as_str).unwrap_or_default().to_string(),
            Err(_) => l.to_string(),
        }
    }).collect();
    let mut out = BufWriter::new(std::fs::File::create(&output)?);
    let start = Instant::now();
    let (mut pass, mut used, mut done) = (Vec::new(), 0usize, 0usize);
    let flush = |pass: &mut Vec<Vec<u32>>, out: &mut BufWriter<std::fs::File>| -> Result<usize> {
        if pass.is_empty() { return Ok(0); }
        for v in enc.embed_text(pass)? {
            let v = &v[..width];
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in v { out.write_all(&half::f16::from_f32(x / norm).to_le_bytes())?; }
        }
        let n = pass.len();
        pass.clear();
        Ok(n)
    };
    let mut tok_s = 0.0f64;
    for text in &texts {
        let t0 = Instant::now();
        let ids = specials.wrap(tok.encode(&format!("{prompt}{text}")).into_iter().map(|t| t as u32), max_tokens);
        tok_s += t0.elapsed().as_secs_f64();
        if used + ids.len() > pass_tokens && !pass.is_empty() {
            done += flush(&mut pass, &mut out)?;
            used = 0;
            if done % 65536 < pass_tokens / 8 {
                eprintln!("{done}/{} texts, {:.0}/s", texts.len(), done as f64 / start.elapsed().as_secs_f64());
            }
        }
        used += ids.len();
        pass.push(ids);
    }
    done += flush(&mut pass, &mut out)?;
    out.flush()?;
    std::fs::write(format!("{output}.json"), format!("{{\"width\": {width}, \"rows\": {done}}}\n"))?;
    eprintln!("{done} texts embedded, width {width}, in {:.1} s ({:.0}/s; GPU {:.1} s, tokenizing {:.1} s)",
        start.elapsed().as_secs_f64(), done as f64 / start.elapsed().as_secs_f64(), enc.gpu_seconds(), tok_s);
    Ok(())
}

/// The prompt the file stores under `name` (`<arch>.embedding.prompt_names` and
/// `<arch>.embedding.prompts`, parallel arrays).
fn named_prompt(g: &Gguf, name: &str) -> Result<String> {
    let arch = g.arch();
    let names = g.str_arr(&format!("{arch}.embedding.prompt_names")).context("this file carries no named prompts")?;
    let prompts = g.str_arr(&format!("{arch}.embedding.prompts")).context("this file carries no named prompts")?;
    names.iter().position(|n| n == name).and_then(|i| prompts.get(i)).cloned()
        .with_context(|| format!("no prompt named {name:?}; the file has {names:?}"))
}

/// The special tokens the file asks to surround a text with.
struct Specials {
    bos: Option<u32>,
    eos: Option<u32>,
}

impl Specials {
    fn from_gguf(g: &Gguf) -> Result<Specials> {
        let flag = |k: &str| matches!(g.meta.get(&format!("tokenizer.ggml.add_{k}_token")), Some(Meta::Bool(true)));
        let id = |k: &str| g.meta_u32(&format!("tokenizer.ggml.{k}_token_id")).with_context(|| format!("no tokenizer.ggml.{k}_token_id"));
        Ok(Specials {
            bos: if flag("bos") { Some(id("bos")?) } else { None },
            eos: if flag("eos") { Some(id("eos")?) } else { None },
        })
    }

    fn count(&self) -> usize { self.bos.is_some() as usize + self.eos.is_some() as usize }

    /// `text` with the special tokens around it, cut to at most `max` tokens in all.
    fn wrap(&self, text: impl Iterator<Item = u32>, max: usize) -> Vec<u32> {
        let mut ids = Vec::with_capacity(max);
        ids.extend(self.bos);
        ids.extend(text.take(max - self.count()));
        ids.extend(self.eos);
        ids
    }
}
