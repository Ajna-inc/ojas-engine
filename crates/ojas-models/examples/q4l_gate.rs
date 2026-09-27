//! Gate: Q4L (relayout + gemv) must reproduce Q4_K's exact values.
//!
//! Q4L is not a new quantization — it is Q4_K's nibbles and scales moved into the
//! layout the tuned Q4 kernel wants. Nibbles must survive exactly; the per-block
//! `d1`/`-m1` are stored as f16, so the dot lands ~2e-3 off the f64 oracle.
//!
//! That tolerance is measured, not slack: f32 side arrays give 8e-7 here but cost 1.5
//! bits/weight (6.0 vs 5.0), and `scripts/parity.py` reports byte-identical per-layer
//! cosines either way (0.9928 worst, logits 0.9964), so the f16 rounding sits far below
//! Q4_K's own 4.5-bit noise. The bar here is that the relayout did not lose or reorder a
//! nibble, which a mistake would miss by orders of magnitude; per-layer parity guards
//! accuracy.
//!
//! usage: cargo run -p ojas-models --example q4l_gate <gguf>

use anyhow::{bail, Result};
use ojas_core::Device;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: q4l_gate <gguf>");
    let mut g = ojas_formats::gguf::Gguf::open(&path)?;
    let gpu = ojas_metal::MetalGpu::new()?;

    let mut pick = None;
    let mut names: Vec<String> = g.tensors.keys().cloned().collect();
    names.sort();
    for name in names {
        let i = &g.tensors[&name];
        if i.ggml_type == 12 && i.dims.len() == 2 && i.dims[0] % 256 == 0 {
            pick = Some((name.clone(), i.dims.clone()));
            break;
        }
    }
    let Some((name, dims)) = pick else { bail!("no 2-D Q4_K tensor in {path}") };
    let k = dims[0] as usize;
    let n = (dims[1] as usize).min(64);
    let nsb = k / 256;
    let nblk = k / 32;
    println!("tensor {name}  K={k}  testing {n} rows");

    let (_d, _ty, raw) = g.read_tensor_raw(&name)?;

    // Deterministic activations.
    let mut seed = 0x1234_5678u32;
    let x: Vec<f32> = (0..k)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect();

    // f64 oracle over the raw Q4_K blocks (the reference dequant_row_q4_K).
    let mut want = vec![0f64; n];
    for (row, w) in want.iter_mut().enumerate() {
        let mut acc = 0f64;
        for sb in 0..nsb {
            let b = &raw[(row * nsb + sb) * 144..(row * nsb + sb + 1) * 144];
            let d = half::f16::from_le_bytes([b[0], b[1]]).to_f32() as f64;
            let dmin = half::f16::from_le_bytes([b[2], b[3]]).to_f32() as f64;
            let (sc, qs) = (&b[4..16], &b[16..144]);
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
        }
        *w = acc;
    }

    // GPU: relayout, then the fast gemv over the new layout.
    let wb = gpu.upload_u8(&raw[..n * nsb * 144]);
    let nib = gpu.alloc((n * (k / 2) + 3) / 4);
    let qa = gpu.alloc(n * nblk);
    let qb = gpu.alloc(n * nblk);
    let (ku, nu) = (k as u32, n as u32);
    let c = |i: u32| &ku as *const u32 as *const std::ffi::c_void as *const std::ffi::c_void;
    let _ = c;

    let run = |entry: &str, bufs: &[(&metal::Buffer, u64)], tgs: u64, tpg: u64| -> Result<()> {
        let src = ojas_metal::kernels::source_of(entry).expect("kernel");
        let pipe = gpu.pipeline(src, entry)?;
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        for (i, (b, off)) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), *off);
        }
        enc.dispatch_thread_groups(metal::MTLSize::new(tgs, 1, 1), metal::MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    };

    // relayout(w, nib, qa, qb, K, N)
    {
        let src = ojas_metal::kernels::source_of("relayout_q4k_q4l").expect("kernel");
        let pipe = gpu.pipeline(src, "relayout_q4k_q4l")?;
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&wb.buf), 0);
        enc.set_buffer(1, Some(&nib.buf), 0);
        enc.set_buffer(2, Some(&qa.buf), 0);
        enc.set_buffer(3, Some(&qb.buf), 0);
        enc.set_bytes(4, 4, &ku as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &nu as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((n + 7) / 8) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let _ = run;

    let xb = gpu.upload(&x);
    let yb = gpu.alloc(n);
    {
        let src = ojas_metal::kernels::source_of("gemv_q4l").expect("kernel");
        let pipe = gpu.pipeline(src, "gemv_q4l")?;
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&xb.buf), 0);
        enc.set_buffer(1, Some(&nib.buf), 0);
        enc.set_buffer(2, Some(&yb.buf), 0);
        enc.set_bytes(3, 4, &ku as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &nu as *const u32 as *const std::ffi::c_void);
        enc.set_buffer(5, Some(&qa.buf), 0);
        enc.set_buffer(6, Some(&qb.buf), 0);
        // 4 rows per simdgroup, 8 simdgroups per threadgroup = 32 rows/tg.
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((n + 31) / 32) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let got = gpu.read(&yb);

    let (mut dot, mut na, mut nb2, mut mx) = (0f64, 0f64, 0f64, 0f64);
    for r in 0..n {
        let (a, b) = (want[r], got[r] as f64);
        dot += a * b;
        na += a * a;
        nb2 += b * b;
        mx = mx.max((a - b).abs());
    }
    let cosv = if na > 0.0 && nb2 > 0.0 { dot / (na.sqrt() * nb2.sqrt()) } else { 0.0 };
    let err = mx / (na / n as f64).sqrt().max(1e-30);
    println!("cos = {cosv:.9}   err/rms = {err:.2e}");
    if cosv > 0.999_99 && err < 1e-2 {
        println!("GATE: Q4L PASS");
        Ok(())
    } else {
        bail!("GATE: Q4L FAIL")
    }
}
