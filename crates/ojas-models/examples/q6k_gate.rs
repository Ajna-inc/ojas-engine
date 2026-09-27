//! Gate: the native K-quant gemv kernels must agree with the reference dequant math on
//! real tensors. Covers `gemv_q4k` (ty 12) and `gemv_q6k` (ty 14).
//!
//! Oracle: `dequantize_row_q6_K` from the reference implementation, transcribed here in
//! f64 and run over the same raw super-blocks the kernel reads. Going through
//! `dequant_to_f16` instead would round every weight to f16 while the kernel keeps f32;
//! the kernel is the more accurate of the two, so that comparison would measure the
//! oracle's error, not the kernel's.
//!
//! Both sides dot against one fixed pseudo-random vector and are compared by cosine plus
//! error normalised to the output RMS. Element-wise relative error is the wrong bar: a
//! 4096-term dot product cancels, so a result near zero has a huge relative error while
//! being perfectly correct.
//!
//! usage: cargo run -p ojas-models --example q6k_gate <gguf>

use anyhow::{bail, Result};
use ojas_core::Device;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: q6k_gate <gguf>");
    let mut g = ojas_formats::gguf::Gguf::open(&path)?;

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut all_pass = true;

    // (GGUF type, kernel, bytes per super-block, label)
    for &(ty, entry, blk, label) in &[
        (12u32, "gemv_q4k", 144usize, "Q4_K"),
        (14u32, "gemv_q6k", 210usize, "Q6_K"),
    ] {
        let mut pick: Option<(String, Vec<u64>)> = None;
        let mut names: Vec<String> = g.tensors.keys().cloned().collect();
        names.sort();
        for name in names {
            let info = &g.tensors[&name];
            if info.ggml_type == ty && info.dims.len() == 2 && info.dims[0] % 256 == 0 {
                pick = Some((name.clone(), info.dims.clone()));
                break;
            }
        }
        let Some((name, dims)) = pick else {
            println!("{label:6} (absent from this file — skipped)");
            continue;
        };

        let k = dims[0] as usize;
        let n_test = (dims[1] as usize).min(64);
        let (_d, ty_raw, raw) = g.read_tensor_raw(&name)?;
        assert_eq!(ty_raw, ty);

        // Deterministic activations — reproducible, no rand dependency.
        let mut seed = 0x1234_5678u32;
        let x: Vec<f32> = (0..k)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect();

        // ---- CPU oracle in f64, over the same raw blocks (see module docs).
        let nsb = k / 256;
        let mut want = vec![0f64; n_test];
        for (row, w) in want.iter_mut().enumerate() {
            let row_base = row * nsb * blk;
            let mut acc = 0f64;
            for sb in 0..nsb {
                let b = &raw[row_base + sb * blk..row_base + (sb + 1) * blk];
                if ty == 12 {
                    let d = half::f16::from_le_bytes([b[0], b[1]]).to_f32() as f64;
                    let dmin = half::f16::from_le_bytes([b[2], b[3]]).to_f32() as f64;
                    let sc = &b[4..16];
                    let qs = &b[16..144];
                    for j in 0..8usize {
                        let (s, m) = if j < 4 {
                            ((sc[j] & 63) as f64, (sc[j + 4] & 63) as f64)
                        } else {
                            (
                                ((sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4)) as f64,
                                ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) as f64,
                            )
                        };
                        let (d1, m1) = (d * s, dmin * m);
                        let qq = &qs[(j >> 1) * 32..(j >> 1) * 32 + 32];
                        let hi = j & 1 == 1;
                        for l in 0..32usize {
                            let nb = if hi { qq[l] >> 4 } else { qq[l] & 0x0F } as f64;
                            acc += (d1 * nb - m1) * x[sb * 256 + j * 32 + l] as f64;
                        }
                    }
                } else {
                    let d = half::f16::from_le_bytes([b[208], b[209]]).to_f32() as f64;
                    for h in 0..2usize {
                        let (ql, qh, sc) = (&b[h * 64..], &b[128 + h * 32..], &b[192 + h * 8..]);
                        for l in 0..32usize {
                            let is = l / 16;
                            let q1 = ((ql[l] & 0xF) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) - 32;
                            let q2 = ((ql[l + 32] & 0xF) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
                            let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
                            let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;
                            let o = sb * 256 + h * 128 + l;
                            acc += d * sc[is] as i8 as f64 * q1 as f64 * x[o] as f64;
                            acc += d * sc[is + 2] as i8 as f64 * q2 as f64 * x[o + 32] as f64;
                            acc += d * sc[is + 4] as i8 as f64 * q3 as f64 * x[o + 64] as f64;
                            acc += d * sc[is + 6] as i8 as f64 * q4 as f64 * x[o + 96] as f64;
                        }
                    }
                }
            }
            *w = acc;
        }

        // ---- GPU: the kernel under test.
        let src = ojas_metal::kernels::source_of(entry).expect("kernel in a family");
        let pipe = gpu.pipeline(src, entry)?;
        let xb = gpu.upload(&x);
        let wb = gpu.upload_u8(&raw[..n_test * nsb * blk]);
        let yb = gpu.alloc(n_test);
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&xb.buf), 0);
        enc.set_buffer(1, Some(&wb.buf), 0);
        enc.set_buffer(2, Some(&yb.buf), 0);
        let (ku, nu) = (k as u32, n_test as u32);
        enc.set_bytes(3, 4, &ku as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &nu as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((n_test + 7) / 8) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = gpu.read(&yb);

        let (mut dot, mut na, mut nb, mut max_abs) = (0f64, 0f64, 0f64, 0f64);
        for row in 0..n_test {
            let (a, b) = (want[row], got[row] as f64);
            dot += a * b;
            na += a * a;
            nb += b * b;
            max_abs = max_abs.max((a - b).abs());
        }
        let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 0.0 };
        let err = max_abs / (na / n_test as f64).sqrt().max(1e-30);
        let pass = cos > 0.999_999 && err < 1e-4;
        all_pass &= pass;
        println!(
            "{label:6} {name:30} K={k:<6} cos {cos:.9}  err/rms {err:.2e}  {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    if all_pass {

        println!("GATE: NATIVE KQUANT GEMV PASS");
        Ok(())
    } else {
        bail!("GATE: NATIVE KQUANT GEMV FAIL")
    }
}
