//! Isolated scalar Q8 geometry sweep. Synthetic weights; production code unchanged.
use anyhow::{ensure, Result};
use metal::{MTLResourceOptions as Opt, MTLSize};
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device as _;
use std::ffi::c_void;
fn main() -> Result<()> {
    ensure!(
        std::env::var_os("OJAS_NO_FAST").is_none(),
        "unset OJAS_NO_FAST to benchmark the production scalar kernel"
    );
    let gpu = ojas_metal::MetalGpu::new()?;
    let src = ojas_metal::kernels::family_source("gemv").unwrap();
    // (rows per threadgroup, SIMDgroups, lanes per quant block).
    let configs = [
        (2, 4, 4),
        (2, 2, 4),
        (2, 8, 4),
        (4, 4, 4),
        (4, 8, 4),
        (2, 4, 8),
        (4, 4, 8),
    ];
    let mut pipes = Vec::new();
    for (rows, groups, lanes) in configs {
        let start = src.find("#define NAT_GEMV_Q80(").unwrap();
        let end = start + src[start..].find("// Batched (prefill)").unwrap();
        let body = src[start..end]
            .replace("_part[8]", &format!("_part[{}]", rows * groups))
            .replace("_r * 4u", &format!("_r * {groups}u"));
        let body = if lanes == 8 {
            body.replace("lane / 4u", "lane / 8u")
                .replace("lane % 4u", "lane % 8u")
                .replace("sgid * 8u", "sgid * 4u")
                .replace("_nsg * 8u", "_nsg * 4u")
                .replace("_il * 8u", "_il * 4u")
                .replace("_j < 8u", "_j < 4u")
        } else {
            body
        };
        let mut code = format!("{}{}{}", &src[..start], body, &src[end..]);
        code = code.replace(
            "NAT_GEMV_Q80(Q80_SUB, 34u, 32u, 1u, 32, 2,",
            &format!("NAT_GEMV_Q80(Q80_SUB, 34u, 32u, 1u, 32, {rows},"),
        );
        ensure!(
            code.contains(&format!("32, {rows}, y[n] = acc)")),
            "row replacement failed"
        );
        pipes.push(gpu.pipeline(&code, "gemv_nat_q80")?);
    }
    let upload = |p: *const c_void, bytes: usize| {
        gpu.device
            .new_buffer_with_data(p, bytes as u64, Opt::StorageModeShared)
    };
    for (k, n) in [
        (2560usize, 13usize),
        (2560, 48),
        (10240, 320),
        (320, 10240),
        (2560, 6144),
        (2560, 10240),
        (640, 2560),
        (2560, 640),
        (6144, 2560),
        (2560, 2560),
        (2560, 512),
        (2560, 12288),
    ] {
        // Limit an investigation to the production GDN projection shapes.
        if std::env::var_os("OJAS_BENCH_GDN_ONLY").is_some()
            && !matches!(
                (k, n),
                (2560, 48) | (2560, 6144) | (2560, 10240) | (6144, 2560)
            )
        {
            continue;
        }
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
        // Keep the rotating pool above the M2 Max system cache even for small matrices.
        let pool = if n == 13 {
            8
        } else {
            (128usize * 1024 * 1024).div_ceil(w.len()).max(8)
        };
        let weights: Vec<_> = (0..pool)
            .map(|_| upload(w.as_ptr().cast(), w.len()))
            .collect();
        for copies in [1usize, pool] {
            let m = 1usize;
            let x: Vec<f32> = (0..k * m)
                .map(|i| ((i * 137 % 1021) as f32 - 510.0) / 511.0)
                .collect();
            let xb = upload(x.as_ptr().cast(), x.len() * 4);
            let y = gpu
                .device
                .new_buffer(((n * m + 16) * 4) as u64, Opt::StorageModeShared);
            let mut expected = vec![];
            let mut times = vec![vec![]; configs.len()];
            let mut worst_error = 0f32;
            for rep in 0..19 {
                for order in 0..configs.len() {
                    let variant = if rep == 0 {
                        order
                    } else {
                        (order + rep) % configs.len()
                    };
                    unsafe {
                        let ys =
                            std::slice::from_raw_parts_mut(y.contents().cast::<f32>(), n * m + 16);
                        ys[..n * m].fill(f32::NAN);
                        ys[n * m..].fill(12345.);
                    }
                    let cb = gpu.command_buffer();
                    let e = cb.new_compute_command_encoder();
                    for wb in &weights[..copies] {
                        e.set_compute_pipeline_state(&pipes[variant]);
                        e.set_buffer(0, Some(&xb), 0);
                        e.set_buffer(1, Some(wb), 0);
                        e.set_buffer(2, Some(&y), 0);
                        for (idx, val) in [(3, k as u32), (4, n as u32), (7, m as u32), (8, 0u32)] {
                            e.set_bytes(idx, 4, (&val as *const u32).cast());
                        }
                        e.dispatch_thread_groups(
                            MTLSize::new(n.div_ceil(configs[variant].0) as u64, 1, 1),
                            MTLSize::new((configs[variant].1 * 32) as u64, 1, 1),
                        );
                    }
                    e.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    ensure!(
                        cb.status() == metal::MTLCommandBufferStatus::Completed,
                        "GPU command failed"
                    );
                    ensure!(
                        unsafe {
                            std::slice::from_raw_parts(y.contents().cast::<f32>().add(n * m), 16)
                        }
                        .iter()
                        .all(|&v| v == 12345.),
                        "output guard overwritten"
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
                    worst_error = worst_error.max(maxerr);
                    ensure!(
                        maxerr < 0.0001,
                        "kernel mismatch K={k} N={n} M={m}: {maxerr}"
                    );
                    if rep == 0 && variant == 0 {
                        for row in 0..m {
                            for out in (0..n)
                                .filter(|&r| n == 13 || r % (n / 17).max(1) == 0 || r == n - 1)
                            {
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
                    if rep >= 4 {
                        let start: f64 = unsafe { msg_send![cb, GPUStartTime] };
                        let end: f64 = unsafe { msg_send![cb, GPUEndTime] };
                        ensure!(end > start, "no GPU timestamps");
                        times[variant].push((end - start) * 1000.0 / copies as f64);
                    }
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            for (v, (rows, groups, lanes)) in configs.iter().enumerate() {
                println!("{{\"k\":{k},\"n\":{n},\"copies\":{copies},\"rows\":{rows},\"groups\":{groups},\"lanes\":{lanes},\"median_ms\":{},\"p20_ms\":{},\"p80_ms\":{},\"speedup\":{},\"max_error\":{worst_error}}}",times[v][7],times[v][3],times[v][11],times[0][7]/times[v][7]);
            }
        }
    }
    Ok(())
}
