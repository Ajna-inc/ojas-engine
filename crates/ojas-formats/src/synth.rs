//! Deterministic synthetic quant blocks, for testing dequantizers without a model.
//!
//! Most IQ types cannot be produced by `llama-quantize` without an importance
//! matrix, so there is no easy way to obtain a real tensor of (say) IQ2_XXS to
//! test against. Random blocks solve that, and are the better input regardless:
//! every bit pattern is a legal block for these formats — grid indices are
//! masked into range and sign indices are 7 bits into a 128-entry table — so
//! random data sweeps the whole codebook, while a trained tensor touches only
//! the entries it happens to need.

/// Bytes per block, and weights per block, for a GGUF type.
pub fn block_shape(t: u32) -> Option<(usize, usize)> {
    Some(match t {
        2 => (18, 32), 3 => (20, 32), 6 => (22, 32), 7 => (24, 32), 8 => (34, 32),
        10 => (84, 256), 11 => (110, 256), 12 => (144, 256), 13 => (176, 256), 14 => (210, 256),
        16 => (66, 256), 17 => (74, 256), 18 => (98, 256), 19 => (50, 256),
        20 => (18, 32), 21 => (110, 256), 22 => (82, 256), 23 => (136, 256), 29 => (56, 256),
        _ => return None,
    })
}

/// `nb` blocks of type `t`, byte-for-byte reproducible across runs and machines.
///
/// Every byte is random except the f16 scale fields. Those are pinned because a
/// random u16 can decode to inf or NaN, which poisons every value in the block.
/// The offsets differ per format, and getting them wrong silently reintroduces
/// the inf: Q3_K keeps `d` at the end of the block, Q6_K likewise, Q4_K/Q5_K
/// carry a second `dmin`, and IQ1_M has no `d` field at all.
pub fn blocks(t: u32, nb: usize) -> Vec<u8> {
    let (bb, _) = block_shape(t).unwrap_or_else(|| panic!("synth: no block shape for GGUF type {t}"));
    let mut st: u64 = 0x2545_F491_4F6C_DD1D;
    let mut bytes = vec![0u8; nb * bb];
    for b in bytes.iter_mut() {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        *b = (st >> 24) as u8;
    }
    for i in 0..nb {
        let off = i * bb;
        let d = half::f16::from_f32(0.05 + (i % 7) as f32 * 0.01).to_bits().to_le_bytes();
        let dmin = half::f16::from_f32(0.01).to_bits().to_le_bytes();
        let set = |b: &mut Vec<u8>, at: usize, v: [u8; 2]| {
            b[off + at..off + at + 2].copy_from_slice(&v);
        };
        match t {
            10 => { set(&mut bytes, 80, d); set(&mut bytes, 82, dmin); }
            11 => set(&mut bytes, 108, d),
            12 | 13 => { set(&mut bytes, 0, d); set(&mut bytes, 2, dmin); }
            14 => set(&mut bytes, 208, d),
            29 => {
                // IQ1_M's f16 scale is reassembled from the top nibble of each of
                // the four trailing scale words, low nibble first.
                let db = half::f16::from_f32(0.05).to_bits();
                for w in 0..4 {
                    let at = off + 48 + 2 * w;
                    let cur = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
                    let v = (cur & 0x0FFF) | (((db >> (4 * w)) & 0xF) << 12);
                    bytes[at..at + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            _ => set(&mut bytes, 0, d),
        }
    }
    bytes
}
