//! Isolated production-shaped F32 alpha/beta projections + activation fusion.
//! No model load or production-kernel modifications.
use anyhow::{ensure, Result};
use metal::{MTLResourceOptions as Opt, MTLSize};
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device as _;

fn main() -> Result<()> {
    let gpu = ojas_metal::MetalGpu::new()?;
    let mv = gpu.pipeline(
        ojas_metal::kernels::family_source("moe").unwrap(),
        "gemv_w32",
    )?;
    let ab = gpu.pipeline(ojas_metal::kernels::family_source("ssm").unwrap(), "ssm_ab")?;
    let fused = gpu.pipeline(r#"
#include <metal_stdlib>
using namespace metal;
kernel void fused(device const float4* x [[buffer(0)]], device const float4* wa [[buffer(1)]],
device const float4* wb [[buffer(2)]], device float* ya [[buffer(3)]], device float* yb [[buffer(4)]],
device const float* dt [[buffer(5)]],device const float* a [[buffer(6)]],
uint tg [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]],uint ts [[threads_per_threadgroup]]) {
uint row=tg*(ts/32)+sg;if(row>=48)return;
float pa=0.,pb=0.;for(uint k=lane;k<640;k+=32){float4 v=x[k];pa+=dot(wa[row*640+k],v);pb+=dot(wb[row*640+k],v);}
pa=simd_sum(pa);pb=simd_sum(pb);
if(lane==0){float z=pa+dt[row];ya[row]=(z>20.?z:log(1.+exp(z)))*a[row];yb[row]=1./(1.+exp(-pb));}
}
"#, "fused")?;
    let (k, n) = (2560usize, 48usize);
    let upload = |v: &[f32]| {
        gpu.device.new_buffer_with_data(
            v.as_ptr().cast(),
            (v.len() * 4) as u64,
            Opt::StorageModeShared,
        )
    };
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 137 % 1021) as f32 - 510.) / 511.)
        .collect();
    let wa: Vec<f32> = (0..k * n)
        .map(|i| ((i * 71 % 997) as f32 - 498.) / 10000.)
        .collect();
    let wb: Vec<f32> = (0..k * n)
        .map(|i| ((i * 197 % 991) as f32 - 495.) / 11000.)
        .collect();
    let dt: Vec<f32> = (0..n).map(|i| i as f32 / 20. - 1.).collect();
    let a: Vec<f32> = (0..n).map(|i| -(i as f32 + 1.) / 50.).collect();
    let xb = upload(&x);
    let dtb = upload(&dt);
    let abuf = upload(&a);
    let pool: Vec<_> = (0..160).map(|_| (upload(&wa), upload(&wb))).collect();
    let ya = upload(&vec![12345.; n + 16]);
    let yb = upload(&vec![12345.; n + 16]);
    let configs = [0u64, 32, 64, 128, 256];
    let mut refa = Vec::new();
    let mut refb = Vec::new();
    let mut errors = vec![0f32; configs.len()];
    for copies in [1usize, 36, 160] {
        let mut times = vec![Vec::new(); configs.len()];
        for rep in 0..25 {
            for order in 0..configs.len() {
                let v = if rep == 0 {
                    order
                } else {
                    (order + rep) % configs.len()
                };
                let cb = gpu.command_buffer();
                let e = cb.new_compute_command_encoder();
                for (wa, wb) in &pool[..copies] {
                    if v == 0 {
                        for (w, y) in [(wa, &ya), (wb, &yb)] {
                            e.set_compute_pipeline_state(&mv);
                            for (i, b) in [(0, &xb), (1, w), (2, y)] {
                                e.set_buffer(i, Some(b), 0);
                            }
                            for (i, val) in [(3, k as u32), (4, n as u32)] {
                                e.set_bytes(i, 4, (&val as *const u32).cast());
                            }
                            e.dispatch_thread_groups(
                                MTLSize::new(6, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        }
                        e.set_compute_pipeline_state(&ab);
                        for (i, b) in [(0, &ya), (1, &yb), (2, &dtb), (3, &abuf)] {
                            e.set_buffer(i, Some(b), 0);
                        }
                        for i in [4, 5] {
                            let val = 48u32;
                            e.set_bytes(i, 4, (&val as *const u32).cast());
                        }
                        e.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(64, 1, 1));
                    } else {
                        e.set_compute_pipeline_state(&fused);
                        for (i, b) in [
                            (0, &xb),
                            (1, wa),
                            (2, wb),
                            (3, &ya),
                            (4, &yb),
                            (5, &dtb),
                            (6, &abuf),
                        ] {
                            e.set_buffer(i, Some(b), 0);
                        }
                        e.dispatch_thread_groups(
                            MTLSize::new(48 / (configs[v] / 32), 1, 1),
                            MTLSize::new(configs[v], 1, 1),
                        );
                    }
                }
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                ensure!(
                    cb.status() == metal::MTLCommandBufferStatus::Completed,
                    "GPU failure"
                );
                let start: f64 = unsafe { msg_send![cb, GPUStartTime] };
                let end: f64 = unsafe { msg_send![cb, GPUEndTime] };
                ensure!(end > start, "invalid timestamps");
                if rep >= 5 {
                    times[v].push((end - start) * 1000. / copies as f64);
                }
                let ga = unsafe { std::slice::from_raw_parts(ya.contents().cast::<f32>(), n + 16) };
                let gb = unsafe { std::slice::from_raw_parts(yb.contents().cast::<f32>(), n + 16) };
                ensure!(
                    ga[n..].iter().chain(&gb[n..]).all(|&v| v == 12345.),
                    "guard overwritten"
                );
                if rep == 0 && v == 0 {
                    refa = ga[..n].to_vec();
                    refb = gb[..n].to_vec();
                }
                for (g, r) in ga[..n].iter().zip(&refa).chain(gb[..n].iter().zip(&refb)) {
                    ensure!(g.is_finite() && (g - r).abs() < 2e-5, "GPU mismatch");
                    errors[v] = errors[v].max((g - r).abs());
                }
            }
        }
        let mut cpuerr = 0f64;
        for row in 0..n {
            let dot = |w: &[f32]| {
                (0..k)
                    .map(|i| w[row * k + i] as f64 * x[i] as f64)
                    .sum::<f64>()
            };
            let z = dot(&wa) + dt[row] as f64;
            let ca = (if z > 20. { z } else { z.exp().ln_1p() }) * a[row] as f64;
            let cb = 1. / (1. + (-dot(&wb)).exp());
            cpuerr = cpuerr
                .max((refa[row] as f64 - ca).abs())
                .max((refb[row] as f64 - cb).abs());
        }
        ensure!(cpuerr < 2e-5, "CPU mismatch {cpuerr}");
        for (t, threads) in times.iter_mut().zip(configs) {
            t.sort_by(f64::total_cmp);
            let v = configs.iter().position(|&x| x == threads).unwrap();
            println!("{{\"copies\":{copies},\"threads\":{threads},\"median_ms\":{},\"p20_ms\":{},\"p80_ms\":{},\"max_error\":{},\"cpu_error\":{cpuerr}}}",t[10],t[4],t[16],errors[v]);
        }
    }
    Ok(())
}
