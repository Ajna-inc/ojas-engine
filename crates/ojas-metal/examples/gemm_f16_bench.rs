//! f16-weight GEMM throughput on encoder shapes: the staged `gemm_mm_f16` against
//! `gemm_mm_f16_fat`, without and with split-K, each checked against an f64 reference.
//!
//! Shapes are ModernBERT-large's four projections (qkv 1024->3072, out 1024->1024,
//! up 1024->5248, down 2624->1024) at one short sequence (115 rows) and a packed
//! seven-question request (731 rows). Timing is GPU time of `REPS` back-to-back
//! dispatches in one command buffer; the best of five batches is reported, since the
//! GPU is shared with the desktop. `--accum` measures the `y += x.W` form the
//! residual projections use. The first line is the simdgroup-matrix ceiling.
//!
//! Run: cargo run --release -p ojas-metal --example gemm_f16_bench [-- --accum]

use metal::{MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device;
use std::ffi::c_void;

const REPS: usize = 20;

fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

fn main() -> anyhow::Result<()> {
    let accum = std::env::args().any(|a| a == "--accum") as u32;
    let gpu = ojas_metal::MetalGpu::new()?;
    peak(&gpu)?;
    let staged = gpu.pipeline(ojas_metal::kernels::family_source("gemv").expect("gemv family"), "gemm_mm_f16")?;
    let reduce = gpu.pipeline(ojas_metal::kernels::family_source("gemv").expect("gemv family"), "splitk_accum")?;
    let fat = gpu.pipeline(ojas_metal::kernels::gemm_fat::GEMM_FAT_KERNELS, "gemm_mm_f16_fat")?;
    let shapes = [(1024u32, 3072u32, "qkv"), (1024, 1024, "out"), (1024, 5248, "up"), (2624, 1024, "down")];
    println!("{:>5} {:>5} {:>5}  {:>5}  {:>20}  {:>20}  {:>26}", "M", "K", "N", "", "staged", "fat", "fat + split-K");
    for m in [115u32, 731, 4096] {
        let mut totals = [0f64; 3];
        for &(k, n, name) in &shapes {
            let mut seed = k ^ (n << 3) ^ m;
            let x: Vec<f32> = (0..(m * k) as usize).map(|_| lcg(&mut seed)).collect();
            let w: Vec<half::f16> = (0..(n * k) as usize).map(|_| half::f16::from_f32(lcg(&mut seed) * 0.05)).collect();
            let y0: Vec<f32> = (0..(m * n) as usize).map(|_| lcg(&mut seed)).collect();
            let up = |v: *const c_void, bytes: usize| gpu.device.new_buffer_with_data(v, bytes as u64,
                MTLResourceOptions::StorageModeShared);
            let (xb, wb) = (up(x.as_ptr() as _, x.len() * 4), up(w.as_ptr() as _, w.len() * 2));
            // The staged kernel stores whole 32-row tiles, so y carries padding rows.
            let yb = gpu.device.new_buffer((m.div_ceil(32) * 32 * n) as u64 * 4, MTLResourceOptions::StorageModeShared);
            let nsplit = ojas_metal::kernels::gemm_fat::f16_fat_splits(m, n, k, u32::MAX);
            let part = gpu.device.new_buffer((nsplit * m * n) as u64 * 4, MTLResourceOptions::StorageModeShared);
            let mut line = format!("{m:>5} {k:>5} {n:>5}  {name:>5}");
            for (i, splits) in [(0usize, 0u32), (1, 1), (2, nsplit)] {
                let run = |reps: usize| -> f64 {
                    unsafe { std::ptr::copy_nonoverlapping(y0.as_ptr(), yb.contents() as *mut f32, y0.len()) };
                    let cb = gpu.command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    for _ in 0..reps {
                        let pipe = if i == 0 { &staged } else { &fat };
                        enc.set_compute_pipeline_state(pipe);
                        enc.set_buffer(0, Some(&xb), 0);
                        enc.set_buffer(1, Some(&wb), 0);
                        enc.set_buffer(2, Some(if splits > 1 { &part } else { &yb }), 0);
                        for (idx, v) in [(3u64, k), (4, n), (6, accum), (7, m), (9, splits.max(1))] {
                            enc.set_bytes(idx, 4, &v as *const u32 as *const c_void);
                        }
                        let grid = if i == 0 { MTLSize::new(m.div_ceil(32) as u64, (n / 64) as u64, 1) }
                                   else { MTLSize::new(m.div_ceil(64) as u64, (n / 64) as u64, splits as u64) };
                        enc.dispatch_thread_groups(grid, MTLSize::new(128, 1, 1));
                        if splits > 1 {
                            let total = m * n;
                            enc.set_compute_pipeline_state(&reduce);
                            enc.set_buffer(0, Some(&part), 0);
                            enc.set_buffer(1, Some(&yb), 0);
                            for (idx, v) in [(2u64, total), (3, splits), (4, accum)] {
                                enc.set_bytes(idx, 4, &v as *const u32 as *const c_void);
                            }
                            enc.dispatch_thread_groups(MTLSize::new(total.div_ceil(256) as u64, 1, 1), MTLSize::new(256, 1, 1));
                        }
                    }
                    enc.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    let (s, e): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
                    (e - s) / reps as f64
                };
                run(1);
                // f64 over the f16-rounded inputs the kernels multiply, on a strided
                // sample of outputs; one dispatch, so `accum` adds y0 exactly once.
                let y = unsafe { std::slice::from_raw_parts(yb.contents() as *const f32, (m * n) as usize) };
                let mut worst = 0f64;
                for idx in (0..(m * n) as usize).step_by(97) {
                    let (t, r) = (idx / n as usize, idx % n as usize);
                    let dot: f64 = (0..k as usize).map(|j| {
                        half::f16::from_f32(x[t * k as usize + j]).to_f64() * w[r * k as usize + j].to_f64()
                    }).sum();
                    let want = dot + if accum != 0 { y0[idx] as f64 } else { 0.0 };
                    worst = worst.max((y[idx] as f64 - want).abs() / want.abs().max(1.0));
                }
                assert!(worst < 1e-3, "{name} M={m} variant {i}: worst rel err {worst:.3e}");
                let s = (0..5).map(|_| run(REPS)).fold(f64::INFINITY, f64::min);
                totals[i] += s;
                let tflops = 2.0 * m as f64 * k as f64 * n as f64 / s / 1e12;
                let tag = if i == 2 { format!(" x{splits}") } else { String::new() };
                line += &format!("  {:>7.3} ms {:>5.2} TF{tag:<4}", s * 1e3, tflops);
            }
            println!("{line}");
        }
        println!("{:>5} per ModernBERT-large layer: staged {:.3} ms, fat {:.3} ms, fat + split-K {:.3} ms\n",
            m, totals[0] * 1e3, totals[1] * 1e3, totals[2] * 1e3);
    }
    Ok(())
}

/// The simdgroup-matrix ceiling: the GEMM inner step (four A and four B fragments
/// from cache into sixteen float accumulators) with no tile fills or barriers.
fn peak(gpu: &ojas_metal::MetalGpu) -> anyhow::Result<()> {
    const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void mma_peak(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
    device const half* in [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    simdgroup_half8x8 a[4], b[4];
    simdgroup_float8x8 c[16];
    for (short i = 0; i < 16; i++) { c[i] = make_filled_simdgroup_matrix<float, 8>(float(i + sg)); }
    for (uint it = 0; it < iters; it++) {
        device const half* p = in + ((it * 8u) & 63u) * 64u;
        for (short i = 0; i < 4; i++) { simdgroup_load(a[i], p + i * 64, 8); }
        for (short i = 0; i < 4; i++) { simdgroup_load(b[i], p + (4 + i) * 64, 8); }
        for (short i = 0; i < 16; i++) { simdgroup_multiply_accumulate(c[i], a[i / 4], b[i % 4], c[i]); }
    }
    // Every accumulator is stored, so none of the sixteen chains is dead code.
    for (short i = 0; i < 16; i++) { simdgroup_store(c[i], out + ((tg * 4u + sg) * 16u + i) * 64u, 8); }
}
"#;
    let pipe = gpu.pipeline(SRC, "mma_peak")?;
    let (groups, iters) = (38u64 * 64, 4096u32);
    let out = gpu.device.new_buffer(groups * 4 * 16 * 64 * 4, MTLResourceOptions::StorageModeShared);
    let operand: Vec<half::f16> = (0..64 * 64).map(|i| half::f16::from_f32((i % 7) as f32 * 1e-3)).collect();
    let inb = gpu.device.new_buffer_with_data(operand.as_ptr() as *const c_void, (operand.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared);
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&out), 0);
        enc.set_bytes(1, 4, &iters as *const u32 as *const c_void);
        enc.set_buffer(2, Some(&inb), 0);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let (s, e): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        best = best.min(e - s);
    }
    let flops = groups as f64 * 4.0 * iters as f64 * 16.0 * 2.0 * 512.0;
    println!("simdgroup-matrix f16 ceiling: {:.2} TF/s\n", flops / best / 1e12);
    Ok(())
}
