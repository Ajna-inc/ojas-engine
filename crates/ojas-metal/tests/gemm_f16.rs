//! `gemm_mm_f16_fat` (`kernels/gemm_fat.rs`), the encoders' f16 GEMM, against an f64
//! reference over the f16-rounded operands it multiplies.
//!
//! Covered: partial token tiles (M not a multiple of 64, including 1), long K, split-K
//! partitions reduced by `splitk_accum`, and the accumulate form the residual
//! projections use. The output sits between guard regions and is followed by rows past
//! M, which the kernel must never store: its callers do not pad for it.

// The Metal device is macOS-only (`ojas-metal/src/lib.rs`); only its `kernels` source
// table builds elsewhere. These tests drive a real `MetalGpu`.
#![cfg(target_os = "macos")]

use metal::{Buffer, MTLResourceOptions, MTLSize};
use ojas_core::Device;
use ojas_metal::MetalGpu;
use std::ffi::c_void;

const GUARD: usize = 64;
const GUARD_FILL: f32 = 12345.0;

fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

fn upload<T>(gpu: &MetalGpu, v: &[T]) -> Buffer {
    gpu.device.new_buffer_with_data(v.as_ptr() as *const c_void, std::mem::size_of_val(v) as u64,
        MTLResourceOptions::StorageModeShared)
}

fn host(buf: &Buffer, n: usize) -> &[f32] {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }
}

#[test]
fn gemm_mm_f16_fat_matches_f64_reference() {
    let gpu = match MetalGpu::new() {
        Ok(g) => g,
        Err(e) => { eprintln!("gemm_f16: no Metal device ({e}); skipping"); return; }
    };
    if !gpu.native_reduce {
        eprintln!("gemm_f16: simdgroup_matrix needs Apple7/Mac2; skipping");
        return;
    }
    let fat = gpu.pipeline(ojas_metal::kernels::gemm_fat::GEMM_FAT_KERNELS, "gemm_mm_f16_fat")
        .expect("gemm_mm_f16_fat pipeline");
    let reduce = gpu.pipeline(ojas_metal::kernels::family_source("gemv").expect("gemv family"), "splitk_accum")
        .expect("splitk_accum pipeline");
    // (M, K, N): single row, a partial tile, a short request, and the long-K down shape.
    let shapes = [(1u32, 64u32, 64u32), (37, 96, 128), (115, 1024, 1024), (70, 2624, 1024)];
    for &(m, k, n) in &shapes {
        let auto = ojas_metal::kernels::gemm_fat::f16_fat_splits(m, n, k, u32::MAX);
        let mut splits = vec![1u32, 3, auto];
        splits.retain(|&s| k / s >= 32);
        splits.dedup();
        for &nsplit in &splits {
            for accum in [0u32, 1] {
                let mut seed = m ^ (k << 4) ^ (n << 9) ^ nsplit ^ (accum << 20);
                let x: Vec<f32> = (0..(m * k) as usize).map(|_| lcg(&mut seed)).collect();
                let w: Vec<half::f16> = (0..(n * k) as usize).map(|_| half::f16::from_f32(lcg(&mut seed) * 0.05)).collect();
                let y0: Vec<f32> = (0..(m * n) as usize).map(|_| lcg(&mut seed)).collect();
                // y: leading guard, M*N output rows, then a full tile of rows past M and
                // a trailing guard, all of which must come back untouched.
                let tail = 64 * n as usize;
                let mut init = vec![GUARD_FILL; GUARD + (m * n) as usize + tail + GUARD];
                init[GUARD..GUARD + (m * n) as usize].copy_from_slice(&y0);
                let (xb, wb, yb) = (upload(&gpu, &x), upload(&gpu, &w), upload(&gpu, &init));
                let part = gpu.device.new_buffer(((nsplit * m * n) as u64 * 4).max(4), MTLResourceOptions::StorageModeShared);

                let cb = gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&fat);
                enc.set_buffer(0, Some(&xb), 0);
                enc.set_buffer(1, Some(&wb), 0);
                if nsplit > 1 { enc.set_buffer(2, Some(&part), 0); } else { enc.set_buffer(2, Some(&yb), (GUARD * 4) as u64); }
                for (i, v) in [(3u64, k), (4, n), (6, accum), (7, m), (9, nsplit)] {
                    enc.set_bytes(i, 4, &v as *const u32 as *const c_void);
                }
                enc.dispatch_thread_groups(MTLSize::new(m.div_ceil(64) as u64, (n / 64) as u64, nsplit as u64),
                                           MTLSize::new(128, 1, 1));
                if nsplit > 1 {
                    let total = m * n;
                    enc.set_compute_pipeline_state(&reduce);
                    enc.set_buffer(0, Some(&part), 0);
                    enc.set_buffer(1, Some(&yb), (GUARD * 4) as u64);
                    for (i, v) in [(2u64, total), (3, nsplit), (4, accum)] {
                        enc.set_bytes(i, 4, &v as *const u32 as *const c_void);
                    }
                    enc.dispatch_thread_groups(MTLSize::new(total.div_ceil(256) as u64, 1, 1), MTLSize::new(256, 1, 1));
                }
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();

                let label = format!("M={m} K={k} N={n} splits={nsplit} accum={accum}");
                let all = host(&yb, init.len());
                assert!(all[..GUARD].iter().all(|&v| v == GUARD_FILL), "{label}: stored before the output");
                assert!(all[GUARD + (m * n) as usize..].iter().all(|&v| v == GUARD_FILL),
                    "{label}: stored past row M-1");
                let y = &all[GUARD..GUARD + (m * n) as usize];
                let mut worst = 0f64;
                for t in 0..m as usize {
                    for r in 0..n as usize {
                        let dot: f64 = (0..k as usize).map(|j| {
                            half::f16::from_f32(x[t * k as usize + j]).to_f64() * w[r * k as usize + j].to_f64()
                        }).sum();
                        let want = dot + if accum != 0 { y0[t * n as usize + r] as f64 } else { 0.0 };
                        worst = worst.max((y[t * n as usize + r] as f64 - want).abs() / want.abs().max(1.0));
                    }
                }
                assert!(worst < 1e-4, "{label}: worst relative error {worst:.3e}");
            }
        }
    }
}
