//! Row quantization.
//!
//! Two distinct Q4 formats exist on purpose; they feed different kernels:
//! - `q4_pair_*`: interleaved pairs (elem 2j low nibble, 2j+1 high), scale = amax/8.
//!   Metal `gemv_q4` family layout.
//! - `q4_ggml_*`: split halves (elem j low, j+16 high), scale d = mx/-8 (block_q4_0).
//!   CUDA dp4a/mma layout.

use half::f16;

fn f16_at(bytes: &[u8], i: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])).to_f32()
}

/// Per-row symmetric int8 from f16 bytes. Returns (q[n*k], scale[n]).
pub fn q8_rows_from_f16(f16_bytes: &[u8], n: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; n * k];
    let mut scale = vec![1.0f32; n];
    let nthreads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1).clamp(1, n.max(1));
    let rows_per = (n + nthreads - 1) / nthreads;
    std::thread::scope(|s| {
        for (ti, (qc, sc)) in q.chunks_mut(rows_per * k).zip(scale.chunks_mut(rows_per)).enumerate() {
            let row0 = ti * rows_per;
            s.spawn(move || {
                for r in 0..(qc.len() / k) {
                    let gbase = (row0 + r) * k;
                    let mut amax = 0.0f32;
                    for c in 0..k { amax = amax.max(f16_at(f16_bytes, gbase + c).abs()); }
                    let sv = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                    sc[r] = sv;
                    for c in 0..k {
                        qc[r * k + c] = (f16_at(f16_bytes, gbase + c) / sv).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            });
        }
    });
    (q, scale)
}

/// Interleaved-pair Q4 from f16 bytes (Metal layout). Returns (nibbles[n*k/2], f16-bit scales[n*k/32]).
pub fn q4_pair_from_f16(f16_bytes: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u16>) {
    let nblk = k / 32;
    let mut nib = vec![0u8; n * (k / 2)];
    let mut scales = vec![0u16; n * nblk];
    for row in 0..n {
        for b in 0..nblk {
            let base = row * k + b * 32;
            let mut vals = [0f32; 32];
            let mut amax = 0.0f32;
            for j in 0..32 {
                vals[j] = f16_at(f16_bytes, base + j);
                amax = amax.max(vals[j].abs());
            }
            let sc = if amax > 0.0 { amax / 8.0 } else { 1.0 };
            scales[row * nblk + b] = f16::from_f32(sc).to_bits();
            let inv = 1.0 / sc;
            let nb = row * (k / 2) + b * 16;
            for j in 0..16 {
                let q0 = ((vals[2 * j] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                let q1 = ((vals[2 * j + 1] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                nib[nb + j] = q0 | (q1 << 4);
            }
        }
    }
    (nib, scales)
}

/// block_q4_0 from f32 (CUDA layout). Returns (nibbles[n*k/2], f16 scales[n*k/32]).
pub fn q4_ggml_from_f32(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f16>) {
    assert!(k % 32 == 0);
    let nblk = k / 32;
    let rowbytes = k / 2;
    let mut w4 = vec![0u8; n * rowbytes];
    let mut scale = vec![f16::ZERO; n * nblk];
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4);
    let chunk = n.div_ceil(threads);
    std::thread::scope(|s| {
        let w = &w;
        for (ti, (w4c, scc)) in w4.chunks_mut(chunk * rowbytes).zip(scale.chunks_mut(chunk * nblk)).enumerate() {
            s.spawn(move || {
                let row0 = ti * chunk;
                for r in 0..(w4c.len() / rowbytes) {
                    let row = row0 + r;
                    for b in 0..nblk {
                        let blk = &w[row * k + b * 32..row * k + b * 32 + 32];
                        let (mut amax, mut mx) = (0f32, 0f32);
                        for &x in blk {
                            if x.abs() > amax { amax = x.abs(); mx = x; }
                        }
                        let d = mx / -8.0;
                        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                        scc[r * nblk + b] = f16::from_f32(d);
                        for j in 0..16 {
                            let lo = ((blk[j] * id).round() + 8.0).clamp(0.0, 15.0) as u8;
                            let hi = ((blk[j + 16] * id).round() + 8.0).clamp(0.0, 15.0) as u8;
                            w4c[r * rowbytes + b * 16 + j] = lo | (hi << 4);
                        }
                    }
                }
            });
        }
    });
    (w4, scale)
}

pub fn dequant_q4_ggml(w4: &[u8], scale: &[f16], n: usize, k: usize) -> Vec<f32> {
    let nblk = k / 32;
    let mut out = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..nblk {
            let s = scale[row * nblk + b].to_f32();
            for j in 0..16 {
                let by = w4[row * (k / 2) + b * 16 + j];
                out[row * k + b * 32 + j] = s * ((by & 0xF) as f32 - 8.0);
                out[row * k + b * 32 + j + 16] = s * ((by >> 4) as f32 - 8.0);
            }
        }
    }
    out
}

/// int8 (nibble-8) unpack of the block_q4_0 layout, for cp.async GEMM paths.
pub fn unpack_q4_to_s8(w4: &[u8], n: usize, k: usize) -> Vec<i8> {
    let mut out = vec![0i8; n * k];
    for row in 0..n {
        for b in 0..k / 32 {
            for j in 0..16 {
                let by = w4[row * (k / 2) + b * 16 + j];
                out[row * k + b * 32 + j] = (by & 0xF) as i8 - 8;
                out[row * k + b * 32 + j + 16] = (by >> 4) as i8 - 8;
            }
        }
    }
    out
}

pub fn bytes_of_u16(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

pub fn bytes_of_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_ggml_roundtrip_close() {
        let n = 4;
        let k = 64;
        let w: Vec<f32> = (0..n * k).map(|i| ((i * 37 % 97) as f32 - 48.0) / 10.0).collect();
        let (w4, sc) = q4_ggml_from_f32(&w, n, k);
        let deq = dequant_q4_ggml(&w4, &sc, n, k);
        let s8 = unpack_q4_to_s8(&w4, n, k);
        for i in 0..n * k {
            assert!((w[i] - deq[i]).abs() < 0.5, "i={i} {} vs {}", w[i], deq[i]);
        }
        let nblk = k / 32;
        for row in 0..n {
            for b in 0..nblk {
                let s = sc[row * nblk + b].to_f32();
                for j in 0..32 {
                    let idx = row * k + b * 32 + j;
                    assert_eq!(deq[idx], s * s8[idx] as f32);
                }
            }
        }
    }

    #[test]
    fn q8_rows_scale() {
        let n = 2;
        let k = 8;
        let f: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0, -0.1, 0.2, 2.5, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let h: Vec<u16> = f.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let (q, s) = q8_rows_from_f16(bytes_of_u16(&h), n, k);
        assert!((s[0] - 3.0 / 127.0).abs() < 1e-6);
        assert_eq!(q[3], 127);
        assert_eq!(s[1], 1.0);
        assert!(q[8..].iter().all(|&x| x == 0));
    }
}

/// Q4_K super-blocks (144 B / 256) relaid for the Q4L GEMM kernels: nibbles with pairs
/// sharing a byte, and per 32-block f16 `qa = d·sc`, `qb = -dmin·m`, so every value is
/// `qa·q + qb`. `w` is `n` rows of `k` weights, `k % 256 == 0`. A CPU transcription of
/// Metal's load-time `relayout_q4k_q4l` kernel (ojas-metal gemv.rs), so it carries the
/// same f16 rounding of the two products.
pub fn relayout_q4k_q4l(w: &[u8], k: usize, n: usize) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
    assert!(k % 256 == 0 && w.len() == n * k / 256 * 144, "relayout_q4k_q4l: {n} rows of {k} need {} bytes, got {}", n * k / 256 * 144, w.len());
    let (nblk, nsb) = (k / 32, k / 256);
    let mut nib = vec![0u8; n * k / 2];
    let mut qa = vec![0u16; n * nblk];
    let mut qb = vec![0u16; n * nblk];
    for r in 0..n {
        for b in 0..nblk {
            let (sb, j) = (b >> 3, b & 7);
            let (g, hi) = (j >> 1, j & 1);
            let blk = &w[(r * nsb + sb) * 144..][..144];
            let d = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
            let dm = f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
            let sc = &blk[4..16];
            let (s_, m_) = if j < 4 {
                ((sc[j] & 63) as u32, (sc[j + 4] & 63) as u32)
            } else {
                (((sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4)) as u32,
                 ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) as u32)
            };
            qa[r * nblk + b] = f16::from_f32(d * s_ as f32).to_bits();
            qb[r * nblk + b] = f16::from_f32(-dm * m_ as f32).to_bits();
            let qq = &blk[16 + g * 32..][..32];
            for i in 0..16 {
                let pick = |v: u8| if hi == 1 { v >> 4 } else { v & 15 };
                nib[r * (k / 2) + b * 16 + i] = pick(qq[2 * i]) | (pick(qq[2 * i + 1]) << 4);
            }
        }
    }
    (nib, qa, qb)
}
