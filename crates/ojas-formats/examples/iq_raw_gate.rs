//! Dequantize random blocks and dump them, for diffing against gguf-py.
//!
//! The IQ2/IQ3 formats cannot be produced by `llama-quantize` without an
//! importance matrix, which needs a working reference inference build, so this
//! takes the other route.
//!
//! Every bit pattern is a legal block for these types: each grid index is masked
//! into range (`& 511`, `& 0x300`, a byte into a 256-entry table) and each sign
//! index is 7 bits into a 128-entry table. Random bytes are therefore valid
//! input, and better input than a real model's weights: a trained tensor touches
//! whatever subset of the codebook it happens to need, while random blocks sweep
//! all 1024 grid entries and all 128 sign words.
//!
//! Rust generates the bytes and writes them out; Python reads that same file, so
//! the two sides need no seed agreement and cannot drift apart.
//!
//! usage: iq_raw_gate <ggml_type> <n_blocks> <out.bin>

use std::io::Write;

use ojas_formats::synth;

fn main() -> std::io::Result<()> {
    let ty: u32 = std::env::args().nth(1).expect("usage: iq_raw_gate <type> <n> <out>").parse().unwrap();
    let nb: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    let out_path = std::env::args().nth(3).unwrap();

    // Shared with the unit tests so the two cannot drift apart.
    let bytes = synth::blocks(ty, nb);
    std::fs::File::create(&out_path)?.write_all(&bytes)?;

    let (_, epb) = synth::block_shape(ty).unwrap();
    let vals = ojas_formats::gguf::dequant_to_f16(&bytes, ty, nb * epb);
    let mut s = String::new();
    for c in vals.chunks_exact(2) {
        s.push_str(&format!("{:.6}\n", half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()));
    }
    print!("{s}");
    Ok(())
}
