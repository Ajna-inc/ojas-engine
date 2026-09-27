//! Gate: the K-quant dequantizers must reproduce the reference values.
//!
//! A wrong block layout loads, runs, and produces confident nonsense. These
//! formats are easy to get subtly wrong: Q2_K is the one K-quant that stores
//! d/dmin at the end of the block rather than the start, and Q3_K's sixteen
//! 6-bit scales are packed across 12 bytes with a bit-interleaving that has to
//! be reproduced exactly.
//!
//! The oracle is gguf-py's `gguf.quants.dequantize` (scripts/kquant_ref.py), so
//! this compares against the reference implementation rather than against a
//! reading of the spec.
//!
//! usage: kquant_gate <gguf> <tensor-name> [n]   — prints the dequant as text

use anyhow::Result;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: kquant_gate <gguf> <tensor> [n]");
    let name = std::env::args().nth(2).expect("tensor name");
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(4096);

    let mut g = ojas_formats::gguf::Gguf::open(&path)?;
    let total = g.tensors[&name].dims.iter().product::<u64>() as usize;
    let (_dims, ty, raw) = g.read_tensor_raw(&name)?;
    let f16 = ojas_formats::gguf::dequant_to_f16(&raw, ty, total);
    tracing::info!(target: "ojas", "type {ty} ({}) elems {total}", ojas_formats::gguf::gguf_type_name(ty));
    for c in f16.chunks_exact(2).take(n) {
        println!("{:.6}", half::f16::from_le_bytes([c[0], c[1]]).to_f32());
    }
    Ok(())
}
