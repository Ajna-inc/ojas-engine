//! Native GGUF Q8_0 projection comparison. Synthetic finite blocks, rotating order.
use anyhow::{ensure, Result};
use metal::{MTLResourceOptions as Opt, MTLSize};
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device as _;
use std::ffi::c_void;
fn main() -> Result<()> {
    let gpu = ojas_metal::MetalGpu::new()?;
    let src = ojas_metal::kernels::family_source("gemv").unwrap();
    let names = [
        "gemv_nat_q80_m",
        "gemv_nat_q80_m_cooperative",
        "gemv_nat_q80_m_cooperative4",
    ];
    let pipes = names
        .iter()
        .map(|n| gpu.pipeline(src, n))
        .collect::<Result<Vec<_>>>()?;
    let upload = |p: *const c_void, bytes: usize| {
        gpu.device
            .new_buffer_with_data(p, bytes as u64, Opt::StorageModeShared)
    };
    for (k, n) in [
        (2560usize, 13usize),
        (2560, 640),
        (10240, 320),
        (2560, 10240),
        (6144, 2560),
        (320, 10240),
    ] {
        let mut w = vec![0u8; k / 32 * n * 34];
        let mut state = 1234567u32;
        for b in w.chunks_exact_mut(34) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let scale = half::f16::from_f32((state % 63 + 1) as f32 / 4096.0)
                .to_bits()
                .to_le_bytes();
            b[..2].copy_from_slice(&scale);
            for v in &mut b[2..] {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                *v = state as u8;
            }
        }
        let wb = upload(w.as_ptr().cast(), w.len());
        for m in [1usize, 2, 3, 4, 7] {
            let x: Vec<f32> = (0..k * m)
                .map(|i| ((i * 137 % 1021) as f32 - 510.0) / 511.0)
                .collect();
            let xb = upload(x.as_ptr().cast(), x.len() * 4);
            let y = gpu
                .device
                .new_buffer((n * m * 4) as u64, Opt::StorageModeShared);
            let mut expected = vec![];
            let mut times = [vec![], vec![], vec![]];
            for rep in 0..8 {
                for order in 0..3 {
                    let variant = if rep == 0 { order } else { (order + rep) % 3 };
                    let cb = gpu.command_buffer();
                    let e = cb.new_compute_command_encoder();
                    e.set_compute_pipeline_state(&pipes[variant]);
                    e.set_buffer(0, Some(&xb), 0);
                    e.set_buffer(1, Some(&wb), 0);
                    e.set_buffer(2, Some(&y), 0);
                    for (idx, val) in [(3, k as u32), (4, n as u32), (7, m as u32), (8, 0u32)] {
                        e.set_bytes(idx, 4, (&val as *const u32).cast());
                    }
                    e.dispatch_thread_groups(
                        MTLSize::new(n.div_ceil(8) as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                    e.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    ensure!(
                        cb.status() == metal::MTLCommandBufferStatus::Completed,
                        "GPU command failed"
                    );
                    let output =
                        unsafe { std::slice::from_raw_parts(y.contents().cast::<f32>(), n * m) };
                    if rep == 0 && variant == 0 {
                        expected = output.to_vec();
                    }
                    ensure!(output.iter().all(|v| v.is_finite()), "nonfinite output");
                    let maxerr = output
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    ensure!(
                        maxerr < 0.0001,
                        "kernel mismatch K={k} N={n} M={m}: {maxerr}"
                    );
                    if rep == 0 && variant == 0 && n == 13 {
                        for row in 0..m {
                            for out in 0..n {
                                let mut want = 0.0f64;
                                for col in 0..k {
                                    let b = (out * k / 32 + col / 32) * 34;
                                    let scale =
                                        half::f16::from_bits(u16::from_le_bytes([w[b], w[b + 1]]))
                                            .to_f32();
                                    want += f64::from(scale)
                                        * f64::from(w[b + 2 + col % 32] as i8)
                                        * f64::from(x[row * k + col]);
                                }
                                ensure!(
                                    (output[row * n + out] as f64 - want).abs() < 0.0001,
                                    "CPU mismatch"
                                );
                            }
                        }
                    }
                    if rep > 0 {
                        let start: f64 = unsafe { msg_send![cb, GPUStartTime] };
                        let end: f64 = unsafe { msg_send![cb, GPUEndTime] };
                        ensure!(end > start, "no GPU timestamps");
                        times[variant].push((end - start) * 1000.0);
                    }
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            println!("{{\"k\":{k},\"n\":{n},\"m\":{m},\"baseline_ms\":{},\"cooperative_ms\":{},\"cooperative4_ms\":{},\"speedup\":{},\"correct\":true}}",times[0][3],times[1][3],times[2][3],times[0][3]/times[1][3]);
        }
    }
    Ok(())
}
