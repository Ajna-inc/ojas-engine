//! Regression lock for every GGUF quantization format this crate decodes.
//!
//! The correctness proof lives in `examples/iq_raw_gate` +
//! `scripts/iq_raw_ref.py`, which check each dequantizer value-for-value against
//! gguf-py. That needs Python and a checked-out reference tree, so it cannot run
//! in `cargo test`.
//!
//! This locks the result of that comparison: the digests below were taken from
//! the implementation on the day every format passed with zero mismatched
//! values. It cannot prove a format is right, only that nothing has changed
//! since it was proven right — weights that are wrong but not obviously wrong
//! are the failure mode here.
//!
//! If a digest here changes, re-run the gate against gguf-py before updating it.

use ojas_formats::{gguf, synth};

/// FNV-1a over the formatted values. Chosen over a real hash only to avoid
/// adding a dependency; collision resistance is irrelevant for a change detector.
fn digest(vals: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for c in vals.chunks_exact(2) {
        let v = half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
        for b in format!("{v:.6}").as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

#[test]
fn every_quant_format_decodes_unchanged() {
    // (GGUF type, expected value count, expected digest)
    const CASES: &[(u32, usize, u64)] = &[
        (10, 10240, 0xbda788876d584945), // Q2_K
        (11, 10240, 0xa3fcfcd97ce8a0a9), // Q3_K
        (12, 10240, 0x6536b83e8b2292b4), // Q4_K
        (13, 10240, 0x361cbd8f84e79ade), // Q5_K
        (14, 10240, 0x38ecabe858257db6), // Q6_K
        (16, 10240, 0xc04cd5bdb6938e01), // IQ2_XXS
        (17, 10240, 0x3602669b51117b9d), // IQ2_XS
        (18, 10240, 0xc84a275076678dab), // IQ3_XXS
        (19, 10240, 0xfae4b5a7dd7618f8), // IQ1_S
        (20, 1280,  0xa5d4cf4e5db3a1bf), // IQ4_NL
        (21, 10240, 0x8e41a16a45b56b9a), // IQ3_S
        (22, 10240, 0x246928ac497add0c), // IQ2_S
        (23, 10240, 0x1429cafb4df9af64), // IQ4_XS
        (29, 10240, 0xcc6c7931b65f407c), // IQ1_M
    ];
    let mut fails = vec![];
    for &(ty, want_n, want_d) in CASES {
        let (_, epb) = synth::block_shape(ty).expect("block shape");
        let raw = synth::blocks(ty, 40);
        let out = gguf::dequant_to_f16(&raw, ty, 40 * epb);
        let n = out.len() / 2;
        let d = digest(&out);
        if n != want_n || d != want_d {
            fails.push(format!("        ({ty}, {n}, 0x{d:016x}),"));
        }
    }
    assert!(fails.is_empty(), "quant format digests changed:\n{}", fails.join("\n"));
}
