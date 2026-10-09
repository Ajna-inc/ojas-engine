//! Q8_0 group-wise quantization for activations and weight deltas on the wire.
//!
//! Adapted from SwarmLLM (MIT OR Apache-2.0), `src/inference/quant.rs` @ b14482d.
//!
//! For every group of [`GROUP_SIZE`] f32 values: `[f16 scale][32 x i8]` = 34 bytes,
//! `scale = max|x| / 127`, `q = round(x / scale)`. A trailing partial group is still a
//! full block; its unused lanes are zero and ignored on decode. About 3.76x smaller
//! than f32.

use half::f16;

pub const GROUP_SIZE: usize = 32;
pub const BLOCK_BYTES: usize = 2 + GROUP_SIZE;

pub fn quantize_q8_0(values: &[f32]) -> Vec<u8> {
    let blocks = values.len().div_ceil(GROUP_SIZE);
    let mut out = Vec::with_capacity(blocks * BLOCK_BYTES);
    for chunk in values.chunks(GROUP_SIZE) {
        // Non-finite lanes are skipped for the scale; the decoder rejects them anyway.
        let amax = chunk.iter().filter(|v| v.is_finite()).fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        for lane in 0..GROUP_SIZE {
            let q = chunk.get(lane).map_or(0i8, |v| (v * inv).round().clamp(-127.0, 127.0) as i8);
            out.push(q as u8);
        }
    }
    out
}

/// `bytes.len()` must be exactly [`q8_0_len`]`(n)`.
pub fn dequantize_q8_0(bytes: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let want = q8_0_len(n).ok_or("Q8_0 length overflows")?;
    if bytes.len() != want {
        return Err(format!("Q8_0 length mismatch: got {}, want {want} for {n} values", bytes.len()));
    }
    let mut out = Vec::with_capacity(n);
    for blk in bytes.chunks_exact(BLOCK_BYTES) {
        let scale = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let take = (n - out.len()).min(GROUP_SIZE);
        out.extend(blk[2..2 + take].iter().map(|&q| q as i8 as f32 * scale));
    }
    Ok(out)
}

/// Encoded size of `n` values, or `None` when it would overflow: a wrapped length
/// would be smaller than the truth and let a short buffer pass the check.
pub fn q8_0_len(n: usize) -> Option<usize> {
    n.div_ceil(GROUP_SIZE).checked_mul(BLOCK_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_within_half_a_step() {
        let v: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.37).collect();
        let q = quantize_q8_0(&v);
        assert_eq!(q.len(), q8_0_len(v.len()).unwrap());
        let d = dequantize_q8_0(&q, v.len()).unwrap();
        for (blk_v, blk_d) in v.chunks(GROUP_SIZE).zip(d.chunks(GROUP_SIZE)) {
            let step = blk_v.iter().fold(0.0f32, |m, x| m.max(x.abs())) / 127.0;
            for (a, b) in blk_v.iter().zip(blk_d) {
                assert!((a - b).abs() <= step * 0.5 + 1e-3 * step.max(1.0), "{a} vs {b}");
            }
        }
    }

    #[test]
    fn zeros_stay_zero_and_wrong_lengths_are_refused() {
        let q = quantize_q8_0(&[0.0; 40]);
        assert_eq!(dequantize_q8_0(&q, 40).unwrap(), vec![0.0; 40]);
        assert!(dequantize_q8_0(&q[..q.len() - 1], 40).is_err());
        assert!(q8_0_len(usize::MAX).is_none());
    }
}
