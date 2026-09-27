//! Run a model on a prompt and print what it says.
//!
//! usage: run <gguf> "<prompt>" [max_tokens] [prec]
use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: run <gguf> \"<prompt>\" [n] [prec]");
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "What is the capital of France?".into());
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let prec: u8 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    ojas_core::logging::init();

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let arch = g.arch();
    let eos = g.meta_u32("tokenizer.ggml.eos_token_id");
    let bpe = ojas_tokenize::tokenizer::Bpe::from_gguf(&g);
    let text = ojas_tokenize::tokenizer::chat_template(&arch, &prompt);
    let ids: Vec<u32> = bpe.encode(&text).into_iter().map(|v| v as u32).collect();

    let t_load = std::time::Instant::now();
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;
    let load_s = t_load.elapsed().as_secs_f64();

    let t0 = std::time::Instant::now();
    let mut core = ojas_infer::EngineCore::new(m);
    core.eos = eos;   // without it generation runs past the stop token
    print!("\n{prompt}\n\n");
    use std::io::Write;
    let mut first: Option<std::time::Instant> = None;
    let mut n_out = 0usize;
    let out = core.generate_with(&ids, n, None, &mut |_, _| {}, &mut |t| {
        if first.is_none() { first = Some(std::time::Instant::now()); }
        n_out += 1;
        print!("{}", bpe.decode(t as usize));
        let _ = std::io::stdout().flush();
        true
    });
    let total = t0.elapsed().as_secs_f64();
    let ttft = first.map(|f| f.duration_since(t0).as_secs_f64()).unwrap_or(total);
    let gen = (total - ttft).max(1e-9);
    println!("\n\n  load {load_s:.1}s | {} prompt tok | first token {ttft:.2}s | {} tok in {gen:.1}s ({:.2} tok/s)",
             ids.len(), out.len(), (out.len().saturating_sub(1)) as f64 / gen);
    Ok(())
}
