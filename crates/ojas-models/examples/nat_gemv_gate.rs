//! Checks the native (no-requant) matvec kernels against an f64 reference.
//!
//! These kernels read GGUF blocks straight from the mmap and dequantize inside the
//! kernel, so nothing downstream can catch a decode slip: there is no Q8 intermediate
//! to compare against, and a wrong weight just produces a plausible number. The
//! reference is `dequant_to_f16` — which `iq_raw_gate` checks value-for-value against
//! gguf-py — followed by a dot product in f64.
//!
//! Tolerance covers f32 accumulation noise over K terms: the kernel sums in f32 with a
//! different association order than the serial reference, so exact equality is not
//! available. A decode bug misses by orders of magnitude. Pass bar is worst relative
//! error < 2e-3 across all five checks.
//!
//! The batched (`_m`) form is checked at M>1, not M=1: at M=1 the [M][K] and [M][N]
//! indexing collapses to the plain case, so a transposed index is invisible. Each row
//! gets different activations so a row mix-up shows up too. The accumulate form runs
//! onto a pre-seeded buffer, and `_bias` is checked against a reference bias vector.
//!
//! usage: nat_gemv_gate [K] [N]

use anyhow::Result;
use metal::{MTLResourceOptions, MTLSize};
use ojas_core::Device as _;
use ojas_formats::synth;
use std::ffi::c_void;

fn main() -> Result<()> {
    let k: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1280);
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let gpu = ojas_metal::MetalGpu::new()?;
    let dev = &gpu.device;

    // Activations: deterministic, mixed sign, O(1) so the dot is well-scaled. MB rows
    // of different activations, laid out [M][K] as the batched kernel expects. Row 0
    // doubles as the M=1 input.
    const MB: usize = 3;
    let xm: Vec<f32> = (0..MB * k)
        .map(|i| { let (r, c) = (i / k, i % k); ((c % 23) as f32 - 11.0) * 0.05 * (1.0 + r as f32) })
        .collect();
    let xs: Vec<f32> = xm[..k].to_vec();
    let xbuf = dev.new_buffer_with_data(
        xm.as_ptr() as *const c_void, (MB * k * 4) as u64, MTLResourceOptions::StorageModeShared);

    let mut fails = 0usize;
    println!("  {:<9}{:>6}{:>10}{:>9}{:>9}{:>9}{:>9}{:>9}",
        "format", "type", "gemv", "bias", "accum", "m>1", "m-acc", "verdict");
    for f in ojas_metal::kernels::nat::FORMATS {
        let (tag, ty) = (f.tag, &f.ty);
        let (bb, wpb) = (f.block_bytes as usize, f.weights as usize);
        if k % wpb != 0 { continue; }
        let nblk = k / wpb;
        let raw = synth::blocks(*ty, nblk * n);
        assert_eq!(raw.len(), nblk * n * bb);

        // Reference: dequantize the whole [N][K] tile, dot each row in f64.
        let f16 = ojas_formats::gguf::dequant_to_f16(&raw, *ty, nblk * n * wpb);
        let mut want = vec![0f64; n];
        for (r, wv) in want.iter_mut().enumerate() {
            let mut s = 0f64;
            for c in 0..k {
                let o = (r * k + c) * 2;
                s += half::f16::from_bits(u16::from_le_bytes([f16[o], f16[o + 1]])).to_f64()
                    * xs[c] as f64;
            }
            *wv = s;
        }
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
        // Per-row references for the batched check.
        let mut want_m = vec![vec![0f64; n]; MB];
        for r in 0..MB {
            for (row, wv) in want_m[r].iter_mut().enumerate() {
                let mut s = 0f64;
                for c in 0..k {
                    let o = (row * k + c) * 2;
                    s += half::f16::from_bits(u16::from_le_bytes([f16[o], f16[o + 1]])).to_f64()
                        * xm[r * k + c] as f64;
                }
                *wv = s;
            }
        }

        let wbuf = dev.new_buffer_with_data(
            raw.as_ptr() as *const c_void, raw.len() as u64, MTLResourceOptions::StorageModeShared);
        let ybuf = dev.new_buffer((n * MB * 4).max(4) as u64, MTLResourceOptions::StorageModeShared);
        let (ku, nu) = (k as u32, n as u32);

        // Bias: distinctive values so a dropped or mis-indexed bias is obvious.
        let bias: Vec<f32> = (0..n).map(|i| 1000.0 + i as f32).collect();
        let bbuf = dev.new_buffer_with_data(
            bias.as_ptr() as *const c_void, (n * 4) as u64, MTLResourceOptions::StorageModeShared);

        let run = |entry: &str, m: Option<u32>, acc: u32, seed: Option<&[f32]>| -> Vec<f32> {
            let rows_out = m.unwrap_or(1) as usize * n;
            if let Some(sv) = seed {
                unsafe { std::ptr::copy_nonoverlapping(sv.as_ptr(), ybuf.contents() as *mut f32, rows_out) };
            }
            let src = ojas_metal::kernels::source_of(entry)
                .unwrap_or_else(|| panic!("kernel {entry} not found"));
            let pipe = gpu.pipeline(src, entry).expect("compile");
            let cb = gpu.command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&pipe);
            e.set_buffer(0, Some(&xbuf), 0);
            e.set_buffer(1, Some(&wbuf), 0);
            e.set_buffer(2, Some(&ybuf), 0);
            e.set_bytes(3, 4, &ku as *const u32 as *const c_void);
            e.set_bytes(4, 4, &nu as *const u32 as *const c_void);
            if entry.ends_with("_bias") { e.set_buffer(5, Some(&bbuf), 0); }
            if let Some(mm) = m {
                e.set_bytes(7, 4, &mm as *const u32 as *const c_void);
                e.set_bytes(8, 4, &acc as *const u32 as *const c_void);
            }
            // Match the shipping launch shape: these kernels derive their row
            // assignment from it, so another geometry would test a kernel nobody runs.
            let (thr, rows) = if entry.ends_with("_m") {
                (256u32, 8u32)   // batched form is still one row per simdgroup
            } else {
                ojas_metal::kernels::nat::nat_launch(*ty).expect("launch shape")
            };
            e.dispatch_thread_groups(
                MTLSize::new(((n as u32 + rows - 1) / rows) as u64, 1, 1),
                MTLSize::new(thr as u64, 1, 1));
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            unsafe { std::slice::from_raw_parts(ybuf.contents() as *const f32, rows_out) }.to_vec()
        };

        let rel = |got: &[f32], want: &[f64]| -> f64 {
            (0..want.len()).fold(0f64, |mx, i| mx.max((got[i] as f64 - want[i]).abs() / scale))
        };

        // 1. plain matvec
        let e1 = rel(&run(&format!("gemv_nat_{tag}"), None, 0, None), &want);
        // 2. bias: y = W.x + b
        let wb: Vec<f64> = (0..n).map(|i| want[i] + bias[i] as f64).collect();
        let e2 = rel(&run(&format!("gemv_nat_{tag}_bias"), None, 0, None), &wb);
        // 3. accum: seed y, expect y += W.x
        let seed: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let wa: Vec<f64> = (0..n).map(|i| want[i] + seed[i] as f64).collect();
        let e3 = rel(&run(&format!("gemv_nat_{tag}_accum"), None, 0, Some(&seed)), &wa);
        // 4. batched at M>1, each row a different activation vector. At M=1 the
        //    [M][K]/[M][N] indexing collapses and a transpose bug hides.
        let gm = run(&format!("gemv_nat_{tag}_m"), Some(MB as u32), 0, None);
        let mut e4 = 0f64;
        for r in 0..MB {
            for c in 0..n {
                e4 = e4.max((gm[r * n + c] as f64 - want_m[r][c]).abs() / scale);
            }
        }
        // 5. batched accumulate
        let seed_m: Vec<f32> = (0..MB * n).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let gma = run(&format!("gemv_nat_{tag}_m"), Some(MB as u32), 1, Some(&seed_m));
        let mut e5 = 0f64;
        for r in 0..MB {
            for c in 0..n {
                let w = want_m[r][c] + seed_m[r * n + c] as f64;
                e5 = e5.max((gma[r * n + c] as f64 - w).abs() / scale);
            }
        }

        let worst = e1.max(e2).max(e3).max(e4).max(e5);
        let ok = worst < 2e-3;
        if !ok { fails += 1; }
        println!("  {:<9}{:>6}{:>10.1e}{:>9.1e}{:>9.1e}{:>9.1e}{:>9.1e}{:>9}",
            tag, ty, e1, e2, e3, e4, e5, if ok { "ok" } else { "FAIL" });
    }
    if fails == 0 { println!("\nGATE: NATIVE-GEMV PASS"); Ok(()) }
    else { println!("\nGATE: NATIVE-GEMV FAIL ({fails})"); std::process::exit(1) }
}
