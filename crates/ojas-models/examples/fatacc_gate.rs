//! Logits digest after a real prefill chunk: the A/B instrument for any change to the
//! batched GEMM (tile shape, accumulator width, staging).
//!
//! Run it twice with the kernel toggled and diff. For the last position of a 256-token
//! prefill it prints the top-5 token ids, the top-1 logit gap (how near a tie the argmax
//! is), the L2 norm, and a checksum over the whole vocab. Cosine between two runs is what
//! matters, so the digest makes a real drift visible while ignoring last-bit noise.
//!
//! A synthetic oracle will not do: the accumulator question is about a 2048-term sum over
//! the activation distribution the model really produces, and uniform noise cancels
//! differently, understating the error.
//!
//! usage: fatacc_gate <gguf> [ntokens]
//!   OJAS_FAT_H=1  -> f16 accumulators      (control: unset = f32)

use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: fatacc_gate <gguf> [n]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&path)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, 3, None, None)?;

    let toks: Vec<u32> = (0..n).map(|i| 100u32 + (i % 400) as u32).collect();
    let rows = m.forward_batch(&toks, 0);
    let last = rows.last().expect("no logits row");

    let mut idx: Vec<usize> = (0..last.len()).collect();
    idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    let top: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, last[i])).collect();
    let gap = last[idx[0]] - last[idx[1]];
    let norm = last.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
    // order-sensitive checksum over the full vocab, so a drift anywhere shows up
    let sum: f64 = last.iter().enumerate().map(|(i, v)| (i as f64 + 1.0) * (*v as f64)).sum();

    println!("top5   {:?}", top.iter().map(|(i, _)| *i).collect::<Vec<_>>());
    println!("logits {:?}", top.iter().map(|(_, v)| *v).collect::<Vec<_>>());
    println!("gap    {gap:.6}");
    println!("l2     {norm:.6}");
    println!("chk    {sum:.3}");
    Ok(())
}
