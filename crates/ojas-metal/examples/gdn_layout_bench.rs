//! Isolated decode experiment; never changes the production kernel or loads weights.
use anyhow::{ensure, Result};
use metal::{MTLResourceOptions as Opt, MTLSize};
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device as _;
fn main() -> Result<()> {
    let gpu = ojas_metal::MetalGpu::new()?;
    let src = ojas_metal::kernels::family_source("ssm").unwrap();
    let variants = [
        (4, false, false),
        (1, false, false),
        (2, false, false),
        (8, false, false),
        (4, true, false),
        (1, true, false),
        (2, true, false),
        (8, true, false),
        (4, false, true),
        (4, true, true),
        (8, true, true),
    ];
    let mut pipes = Vec::new();
    for (groups, strided, prenorm) in variants {
        let mut code = src.replace("tg.x*4u + sgid", &format!("tg.x*{groups}u + sgid"));
        if strided {
            code = code.replace("lane*4u + j", "lane + j*32u");
        }
        if prenorm {
            code = code.replace("constant uint& clamp_l2 [[buffer(14)]],", "constant uint& clamp_l2 [[buffer(14)]], device const float2* norms [[buffer(15)]],");
            code = code.replace("sq += qv[j]*qv[j]; s2 += kv[j]*kv[j];", "");
            code = code.replace("sq = simd_sum(sq); s2 = simd_sum(s2);", "");
            code = code.replace(
                "float qn = clamp_l2 ? scale/max(sqrt(sq), eps) : rsqrt(sq + eps)*scale;",
                "float qn = norms[hk].x;",
            );
            code = code.replace(
                "float kn = clamp_l2 ? 1.0/max(sqrt(s2), eps) : rsqrt(s2 + eps);",
                "float kn = norms[hk].y;",
            );
            ensure!(
                code.contains("float qn = norms[hk].x;"),
                "normalizer replacement failed"
            );
        }
        pipes.push(gpu.pipeline(&code, "deltanet_fused")?);
    }
    let norm_pipe = gpu.pipeline(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void norms(device const float* x [[buffer(0)]], device float2* out [[buffer(1)]],
 uint h [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
 float sq=0., sk=0.;
 for(uint j=0;j<4;j++){uint i=lane*4+j;float q=x[h*128+i], k=x[16*128+h*128+i];sq+=q*q;sk+=k*k;}
 sq=simd_sum(sq);sk=simd_sum(sk);
 if(lane==0)out[h]=float2(rsqrt(sq+1e-6f)/sqrt(128.f),rsqrt(sk+1e-6f));
}
"#,
        "norms",
    )?;
    let (s, hk, hv) = (128usize, 16usize, 48usize);
    let channels = (2 * hk + hv) * s;
    let state_len = hv * s * s;
    let upload = |v: &[f32]| {
        gpu.device.new_buffer_with_data(
            v.as_ptr().cast(),
            (v.len() * 4) as u64,
            Opt::StorageModeShared,
        )
    };
    let input: Vec<f32> = (0..channels)
        .map(|i| ((i * 137 % 1021) as f32 - 510.) / 511.)
        .collect();
    let initial: Vec<f32> = (0..state_len + 16)
        .map(|i| {
            if i >= state_len {
                12345.
            } else {
                ((i * 71 % 997) as f32 - 498.) / 10000.
            }
        })
        .collect();
    let x = upload(&input);
    let norms = upload(&vec![0.; hk * 2]);
    let gate = upload(&vec![-0.2; hv]);
    let beta = upload(&vec![0.4; hv]);
    let states: Vec<_> = (0..36).map(|_| upload(&initial)).collect();
    let outputs: Vec<_> = (0..36)
        .map(|_| upload(&vec![12345.; hv * s + 16]))
        .collect();
    let read = |b: &metal::Buffer, n: usize| unsafe {
        std::slice::from_raw_parts(b.contents().cast::<f32>(), n).to_vec()
    };
    let mut ref_state = Vec::new();
    let mut ref_out = Vec::new();
    for layers in [1usize, 36] {
        let mut times = vec![Vec::new(); variants.len()];
        let mut errors = vec![0f32; variants.len()];
        for rep in 0..19 {
            for order in 0..variants.len() {
                let v = if rep == 0 {
                    order
                } else {
                    (order + rep) % variants.len()
                };
                for state in &states[..layers] {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            initial.as_ptr(),
                            state.contents().cast(),
                            initial.len(),
                        );
                    }
                }
                let cb = gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                for layer in 0..layers {
                    if variants[v].2 {
                        enc.set_compute_pipeline_state(&norm_pipe);
                        enc.set_buffer(0, Some(&x), 0);
                        enc.set_buffer(1, Some(&norms), 0);
                        enc.dispatch_thread_groups(
                            MTLSize::new(hk as u64, 1, 1),
                            MTLSize::new(32, 1, 1),
                        );
                    }
                    enc.set_compute_pipeline_state(&pipes[v]);
                    for (i, b) in [
                        (0, &states[layer]),
                        (1, &x),
                        (2, &gate),
                        (3, &beta),
                        (4, &outputs[layer]),
                        (11, &states[layer]),
                        (15, &norms),
                    ] {
                        enc.set_buffer(i, Some(b), 0);
                    }
                    for (i, val) in [
                        (5, s as u32),
                        (6, hk as u32),
                        (7, hv as u32),
                        (8, channels as u32),
                        (9, 1),
                        (12, u32::MAX),
                        (13, 0),
                        (14, 0),
                    ] {
                        enc.set_bytes(i, 4, (&val as *const u32).cast());
                    }
                    enc.set_bytes(10, 4, (&1e-6f32 as *const f32).cast());
                    enc.dispatch_thread_groups(
                        MTLSize::new((s / variants[v].0) as u64, hv as u64, 1),
                        MTLSize::new((variants[v].0 * 32) as u64, 1, 1),
                    );
                }
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                ensure!(
                    cb.status() == metal::MTLCommandBufferStatus::Completed,
                    "GPU failure"
                );
                let start: f64 = unsafe { msg_send![cb, GPUStartTime] };
                let end: f64 = unsafe { msg_send![cb, GPUEndTime] };
                ensure!(end > start, "missing GPU timestamps");
                if rep >= 4 {
                    times[v].push((end - start) * 1000.);
                }
                for layer in 0..layers {
                    let st = read(&states[layer], state_len + 16);
                    let out = read(&outputs[layer], hv * s + 16);
                    ensure!(
                        st[state_len..]
                            .iter()
                            .chain(&out[hv * s..])
                            .all(|&x| x == 12345.),
                        "guard overwrite"
                    );
                    if rep == 0 && v == 0 && layer == 0 {
                        ref_state = st.clone();
                        ref_out = out.clone();
                    }
                    for (a, b) in st.iter().zip(&ref_state).chain(out.iter().zip(&ref_out)) {
                        ensure!(a.is_finite() && (*a - *b).abs() < 2e-5, "parity failed");
                        errors[v] = errors[v].max((*a - *b).abs());
                    }
                }
            }
        }
        // Independent f64 one-step oracle for every output and updated state element.
        let mut cpu_error = 0f64;
        for h in 0..hv {
            let key = h % hk;
            let q = &input[key * s..(key + 1) * s];
            let k = &input[(hk + key) * s..(hk + key + 1) * s];
            let qn = 1.
                / ((q.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() + 1e-6).sqrt()
                    * (s as f64).sqrt());
            let kn = 1. / (k.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() + 1e-6).sqrt();
            for col in 0..s {
                let off = (h * s + col) * s;
                let decay = (-0.2f32 as f64).exp();
                let sk = (0..s)
                    .map(|j| initial[off + j] as f64 * decay * k[j] as f64)
                    .sum::<f64>()
                    * kn;
                let delta = (input[2 * hk * s + h * s + col] as f64 - sk) * 0.4f32 as f64;
                let mut y = 0.;
                for j in 0..s {
                    let val = initial[off + j] as f64 * decay + k[j] as f64 * kn * delta;
                    cpu_error = cpu_error.max((ref_state[off + j] as f64 - val).abs());
                    y += val * q[j] as f64;
                }
                cpu_error = cpu_error.max((ref_out[h * s + col] as f64 - y * qn).abs());
            }
        }
        ensure!(cpu_error < 2e-5, "CPU mismatch {cpu_error}");
        for t in &mut times {
            t.sort_by(f64::total_cmp);
        }
        for (v, (groups, strided, prenorm)) in variants.iter().enumerate() {
            println!("{{\"layers\":{layers},\"groups\":{groups},\"strided\":{strided},\"prenorm\":{prenorm},\"median_ms\":{},\"p20_ms\":{},\"p80_ms\":{},\"speedup\":{},\"max_error\":{},\"cpu_error\":{cpu_error}}}",times[v][7],times[v][3],times[v][11],times[0][7]/times[v][7],errors[v]);
        }
    }
    Ok(())
}
