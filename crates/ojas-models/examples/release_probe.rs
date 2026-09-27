//! Replays an independent reference's token sequence and emits comparable logits.
//! usage: release_probe model reference.ids output.f32 [precision] [prefill]
use anyhow::{ensure, Result};
use ojas_core::Model;
use std::io::Write;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len() >= 4, "usage: release_probe model reference.ids output.f32 [precision] [prefill]");
    let values = std::fs::read_to_string(&a[2])?.split_whitespace().map(str::parse::<u32>).collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(values.len() > 1, "empty reference sequence");
    let n = values[0] as usize; let tokens = &values[1..];
    ensure!(n > 0 && n <= tokens.len(), "invalid prompt length");
    let prec = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(4u8);
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&a[1])?;
    let vocab = g.str_arr("tokenizer.ggml.tokens").map(|v| v.len()).unwrap_or(0);
    ensure!(tokens.iter().all(|t| (*t as usize) < vocab), "token outside vocabulary");
    if let Some(prompt_path) = a.get(6) {
        let prompt = std::fs::read_to_string(prompt_path)?;
        let bpe = ojas_tokenize::Bpe::from_gguf(&g);
        let own: Vec<u32> = bpe.encode(&prompt).into_iter().map(|t| t as u32).collect();
        ensure!(own == tokens[..n], "tokenizer differs from independent reference");
    }
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 512.max(tokens.len()), prec, None, None)?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[3])?);
    let start = if a.get(5).is_some_and(|s| s == "prefill") { m.prefill(&tokens[..n-1], 0); n-1 } else { 0 };
    for (i, &t) in tokens.iter().enumerate().skip(start) {
        let row = m.forward_logits(t, i).ok_or_else(|| anyhow::anyhow!("logits unavailable"))?;
        ensure!(row.len() == vocab && row.iter().all(|v| v.is_finite()), "invalid logits at {i}");
        if i >= n-1 { for v in row { out.write_all(&v.to_le_bytes())?; } }
    }
    out.flush()?;
    Ok(())
}
