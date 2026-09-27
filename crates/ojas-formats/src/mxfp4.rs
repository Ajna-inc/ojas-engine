//! MXFP4 (OCP microscaling FP4) decode + the repack the loader needs.
//!
//! MXFP4 is a block-scaled 4-bit float: a 4-bit OCP E2M1 code selects a
//! magnitude from a 16-entry LUT, and one E8M0 (pure-exponent) scale is shared
//! across every 32 weights. This is the format modern open-weight checkpoints
//! ship the MoE experts in — gpt-oss is the motivating case.
//!
//! # Layouts
//!
//! HF checkpoints store the two halves as separate tensors: `<name>.blocks`
//! (`U8`, last dim `K/2` — two nibbles per byte) and `<name>.scales` (`U8`, last
//! dim `K/32` — one e8m0 exponent per 32 weights). ojas's quant codegen wants a
//! single contiguous block, so [`pack_from_hf`] interleaves them into ojas's
//! `{ u8 scale; u8 qs[16] } = 17 B / 32` block — the layout `MXFP4_SUB` in
//! `ojas_core::quant_src` decodes. The nibble packing is identical on both sides
//! (weight `2j` is the low nibble of byte `j`, `2j+1` the high nibble), so the
//! repack is a memcpy of the 16 weight bytes behind the scale byte.
//!
//! [`dequant`] and [`dequant_blocks`] are the CPU reference: unit-tested
//! byte-for-byte against hand-computed values, and used to widen an MXFP4 tensor
//! to f32 for the dense/BF16 fallback path when a native kernel is not wired for
//! a given tensor.

/// The OCP E2M1 codebook. `value = (-1)^s * (exp==0 ? man*0.5 : (1+man*0.5) *
/// 2^(exp-1))` for the nibble `s:exp(2):man(1)`. Symmetric about zero.
pub const MXFP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Weights per MXFP4 block (one shared e8m0 scale covers this many).
pub const GROUP: usize = 32;
/// Packed weight bytes per block (`GROUP/2` nibbles).
pub const QS_BYTES: usize = GROUP / 2;
/// ojas block size: one scale byte + the packed nibbles.
pub const BLOCK_BYTES: usize = 1 + QS_BYTES;

/// The e8m0 scale `2^(e-127)`, formed by placing the exponent into the float32
/// exponent field (`e << 23`) rather than through `exp2` — matching the GPU
/// decoder and the reference kernel exactly. `e==0` is `+0.0`, `e==0xFF` is
/// `+inf` (the MX spec's reserved NaN slot; callers should never see it in a
/// well-formed checkpoint).
#[inline]
pub fn e8m0_scale(e: u8) -> f32 {
    f32::from_bits((e as u32) << 23)
}

/// Decode one 32-weight group from its `QS_BYTES` packed nibble bytes and its
/// e8m0 scale exponent, in weight order (`2j` = low nibble, `2j+1` = high).
pub fn dequant_group(qs: &[u8], scale: u8) -> [f32; GROUP] {
    debug_assert!(qs.len() >= QS_BYTES);
    let d = e8m0_scale(scale);
    let mut out = [0.0f32; GROUP];
    for j in 0..QS_BYTES {
        let byte = qs[j] as usize;
        out[2 * j] = d * MXFP4_LUT[byte & 0x0F];
        out[2 * j + 1] = d * MXFP4_LUT[byte >> 4];
    }
    out
}

/// Decode `n_groups` HF-layout groups from the separate blocks + scales tensors
/// (both row-major over the same leading dims) to a flat `n_groups*32` f32 buffer.
pub fn dequant(blocks: &[u8], scales: &[u8], n_groups: usize) -> Vec<f32> {
    assert!(blocks.len() >= n_groups * QS_BYTES, "blocks too short");
    assert!(scales.len() >= n_groups, "scales too short");
    let mut out = Vec::with_capacity(n_groups * GROUP);
    for g in 0..n_groups {
        let qs = &blocks[g * QS_BYTES..g * QS_BYTES + QS_BYTES];
        out.extend_from_slice(&dequant_group(qs, scales[g]));
    }
    out
}

/// Decode ojas-packed 17-byte blocks (`{ u8 scale; u8 qs[16] }`) to f32. This is
/// the CPU twin of `MXFP4_SUB`, so a test can gate the two against each other.
pub fn dequant_blocks(packed: &[u8], n_weights: usize) -> Vec<f32> {
    let n_groups = n_weights / GROUP;
    assert_eq!(n_weights % GROUP, 0, "MXFP4 decodes whole 32-weight groups");
    assert!(packed.len() >= n_groups * BLOCK_BYTES, "packed too short");
    let mut out = Vec::with_capacity(n_weights);
    for g in 0..n_groups {
        let b = &packed[g * BLOCK_BYTES..g * BLOCK_BYTES + BLOCK_BYTES];
        out.extend_from_slice(&dequant_group(&b[1..], b[0]));
    }
    out
}

/// Repack HF's separate `blocks` + `scales` into ojas's contiguous 17-byte
/// blocks, ready to upload as a native MXFP4 weight.
///
/// `blocks` is `rows * (k/2)` bytes and `scales` is `rows * (k/32)` bytes, both
/// row-major with the K dimension innermost — the HF `*_blocks` / `*_scales`
/// layout. Output is `rows * (k/32) * 17` bytes: for each row, each 32-weight
/// group becomes `[scale_byte, 16 nibble bytes]`.
pub fn pack_from_hf(blocks: &[u8], scales: &[u8], rows: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0, "MXFP4 K must be a multiple of 32, got {k}");
    let groups_per_row = k / GROUP;
    let row_qs = k / 2; // packed weight bytes per row
    assert_eq!(blocks.len(), rows * row_qs, "blocks size mismatch");
    assert_eq!(scales.len(), rows * groups_per_row, "scales size mismatch");
    let mut out = vec![0u8; rows * groups_per_row * BLOCK_BYTES];
    for r in 0..rows {
        for g in 0..groups_per_row {
            let dst = (r * groups_per_row + g) * BLOCK_BYTES;
            out[dst] = scales[r * groups_per_row + g];
            let src = r * row_qs + g * QS_BYTES;
            out[dst + 1..dst + 1 + QS_BYTES].copy_from_slice(&blocks[src..src + QS_BYTES]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_is_ocp_e2m1() {
        // Signs mirror the magnitudes; index 8 is -0.0.
        for i in 0..8 {
            assert_eq!(MXFP4_LUT[i], -MXFP4_LUT[i + 8], "sign pair {i}");
        }
        assert_eq!(MXFP4_LUT[7], 6.0);
        assert_eq!(MXFP4_LUT[15], -6.0);
    }

    #[test]
    fn e8m0_is_power_of_two() {
        assert_eq!(e8m0_scale(127), 1.0);
        assert_eq!(e8m0_scale(128), 2.0);
        assert_eq!(e8m0_scale(129), 4.0);
        assert_eq!(e8m0_scale(126), 0.5);
        assert_eq!(e8m0_scale(120), 2f32.powi(-7));
        assert_eq!(e8m0_scale(0), 0.0);
    }

    #[test]
    fn group_decode_byte_exact() {
        // scale=1.0; nibbles sweep the codebook, low then high per byte.
        let qs = [0x10u8, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, 0, 0, 0, 0, 0, 0, 0, 0];
        let got = dequant_group(&qs, 127);
        let expect = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0f32,
        ];
        for i in 0..16 {
            assert_eq!(got[i], expect[i], "weight {i}");
        }
        for i in 16..32 {
            assert_eq!(got[i], 0.0);
        }
    }

    #[test]
    fn scale_multiplies_every_weight() {
        // scale=4.0; every byte lo=5 (3.0) hi=13 (-3.0) -> 12.0 / -12.0.
        let qs = [0xD5u8; QS_BYTES];
        let got = dequant_group(&qs, 129);
        for j in 0..QS_BYTES {
            assert_eq!(got[2 * j], 12.0);
            assert_eq!(got[2 * j + 1], -12.0);
        }
    }

    #[test]
    fn hf_pack_then_block_decode_matches_direct_dequant() {
        // Two rows, K=64 (two groups/row). Arbitrary-but-legal bytes.
        let rows = 2;
        let k = 64;
        let gpr = k / GROUP;
        let row_qs = k / 2;
        let mut blocks = vec![0u8; rows * row_qs];
        let mut scales = vec![0u8; rows * gpr];
        let mut st: u32 = 0x1234_5678;
        for b in blocks.iter_mut() {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            *b = (st >> 8) as u8;
        }
        // keep scales in a sane exponent range so no inf/underflow surprises
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 120 + (i as u8 % 8);
        }
        // direct HF dequant (reference)
        let direct = dequant(&blocks, &scales, rows * gpr);
        // pack -> ojas blocks -> block decode
        let packed = pack_from_hf(&blocks, &scales, rows, k);
        assert_eq!(packed.len(), rows * gpr * BLOCK_BYTES);
        let via_blocks = dequant_blocks(&packed, rows * k);
        assert_eq!(direct.len(), via_blocks.len());
        for (i, (a, b)) in direct.iter().zip(via_blocks.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "weight {i}: {a} vs {b}");
        }
    }

    #[test]
    fn pack_places_scale_ahead_of_its_nibbles() {
        // row 0 group 1 must carry scales[1] and blocks[16..32].
        let rows = 1;
        let k = 64;
        let blocks: Vec<u8> = (0..32u8).collect();
        let scales = [200u8, 201];
        let packed = pack_from_hf(&blocks, &scales, rows, k);
        // group 0
        assert_eq!(packed[0], 200);
        assert_eq!(&packed[1..17], &blocks[0..16]);
        // group 1
        assert_eq!(packed[17], 201);
        assert_eq!(&packed[18..34], &blocks[16..32]);
    }
}
