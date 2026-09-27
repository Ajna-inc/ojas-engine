//! Screening experiment: expose selected experts as independent GPU work.
use anyhow::{ensure, Result};
use metal::{MTLResourceOptions as Opt, MTLSize};
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device as _;
fn main() -> Result<()> {
    let gpu = ojas_metal::MetalGpu::new()?;
    let src = ojas_metal::kernels::family_source("moe_iq").unwrap();
    let baseline = gpu.pipeline(src, "moe_down_iq4nl")?;
    let start = src.find("kernel void moe_down_iq4nl(").unwrap();
    let end = start
        + src[start..]
            .find("kernel void moe_down_iq4nl_parallel(")
            .unwrap();
    let body = src[start..end]
        .replace(
            "kernel void moe_down_iq4nl(",
            "kernel void moe_down_parallel(",
        )
        .replace(
            "uint tgid [[threadgroup_position_in_grid]]",
            "uint2 tg [[threadgroup_position_in_grid]]",
        )
        .replace("tgid*(ts/32u)", "tg.x*(ts.x/32u)")
        .replace(
            "uint ts [[threads_per_threadgroup]]",
            "uint2 ts [[threads_per_threadgroup]]",
        )
        .replace(
            "for (uint j = 0u; j < KSEL; j++)",
            "for (uint j = tg.y; j < tg.y+1u; j++)",
        )
        .replace(
            "float sh = shx[out_row] / (1.0f + exp(-shg[0]));",
            "float sh = 0.0f;",
        )
        .replace("x[out_row] += total + sh;", "x[tg.y*N+out_row] = total;");
    let candidate = gpu.pipeline(&format!("{src}\n{body}"), "moe_down_parallel")?;
    let cooperated = body
        .replace(
            "for (uint b = lane; b < nb; b += 32u)",
            "for (uint b = lane/4u; b < nb; b += 8u)",
        )
        .replace(
            "s += d * iq4nl_block(blk, aj + b*32u);",
            r#"
            float dot=0.;
            for(uint z=0;z<8;z++) {
                uint ii=(lane%4u)*8u+z;
                uint packed=blk[2u+(ii%16u)];
                uint q=ii<16u ? (packed&15u) : (packed>>4u);
                dot+=float(kvalues_iq4nl[q])*aj[b*32u+ii];
            }
            s+=d*dot;
        "#,
        );
    let cooperative = gpu.pipeline(&format!("{src}\n{cooperated}"), "moe_down_parallel")?;
    // Sweep how many lanes cooperate on one 32-weight IQ4_NL block.  Four is
    // the production setting; wider subgroups shorten each lane's serial LUT
    // dependency chain at the cost of fewer independent blocks in flight.
    let prod_start = src.find("kernel void moe_down_iq4nl_parallel(").unwrap();
    let prod_end = prod_start
        + src[prod_start..]
            .find("kernel void moe_down_iq4nl_finish(")
            .unwrap();
    let prod = &src[prod_start..prod_end];
    let width_kernel = |width: u32| -> Result<_> {
        let values = 32 / width;
        let blocks = 32 / width;
        let name = format!("moe_down_iq4nl_w{width}");
        let body = prod
            .replace("moe_down_iq4nl_parallel", &name)
            .replace("lane/4u", &format!("lane/{width}u"))
            .replace("b += 8u", &format!("b += {blocks}u"))
            .replace("(lane & 3u)*8u", &format!("(lane % {width}u)*{values}u"))
            .replace("z < 8u", &format!("z < {values}u"));
        gpu.pipeline(&format!("{src}\n{body}"), &name)
    };
    let width2 = width_kernel(2)?;
    let width8 = width_kernel(8)?;
    let width16 = width_kernel(16)?;
    let reduce = gpu.pipeline(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void finish(device const float* partial [[buffer(0)]],device float* out [[buffer(1)]],
device const float* shared [[buffer(2)]],device const float* gate [[buffer(3)]],
constant uint& n [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 if(i>=n)return; float sum=0.;for(uint j=0;j<10;j++)sum+=partial[j*n+i];
 out[i]+=sum+shared[i]/(1.+exp(-gate[0]));
}
"#,
        "finish",
    )?;
    let (k, n, experts) = (640usize, 2560usize, 32usize);
    let wb_len = k / 32 * 18 * n * experts;
    let mut weights = vec![0u8; wb_len];
    let mut seed = 1234567u32;
    for b in weights.chunks_exact_mut(18) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        b[..2].copy_from_slice(
            &half::f16::from_f32((seed % 63 + 1) as f32 / 8192.)
                .to_bits()
                .to_le_bytes(),
        );
        for v in &mut b[2..] {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *v = seed as u8;
        }
    }
    let upload = |v: &[f32]| {
        gpu.device.new_buffer_with_data(
            v.as_ptr().cast(),
            (v.len() * 4) as u64,
            Opt::StorageModeShared,
        )
    };
    // 16 pools, 147 MiB of selected expert weights touched per timed batch.
    let pools: Vec<_> = (0..16)
        .map(|_| {
            gpu.device.new_buffer_with_data(
                weights.as_ptr().cast(),
                wb_len as u64,
                Opt::StorageModeShared,
            )
        })
        .collect();
    let acts: Vec<f32> = (0..10 * k)
        .map(|i| ((i * 137 % 1021) as f32 - 510.) / 511.)
        .collect();
    let act = upload(&acts);
    let ids = [17u32, 3, 29, 7, 0, 23, 9, 31, 12, 5];
    let idx = gpu
        .device
        .new_buffer_with_data(ids.as_ptr().cast(), 40, Opt::StorageModeShared);
    let probs: Vec<f32> = (1..=10).map(|i| i as f32 / 55.).collect();
    let wgt = upload(&probs);
    let sh = upload(&vec![0.01; n]);
    let gate = upload(&[0.2]);
    let outs: Vec<_> = (0..16).map(|_| upload(&vec![12345.; n + 16])).collect();
    let partial = upload(&vec![12345.; 10 * n + 16]);
    // kind: 0 serial-expert reference, 1 split scalar blocks, 2 legacy
    // four-lane candidate, 3/4/5 production-shaped 2/8/16-lane candidates.
    let configs = [
        (0u8, 64u32, 1u32),
        (1, 64, 1),
        (2, 128, 4),
        (2, 256, 4),
        (3, 64, 2),
        (3, 128, 2),
        (3, 256, 2),
        (4, 64, 8),
        (4, 128, 8),
        (4, 256, 8),
        (5, 64, 16),
        (5, 128, 16),
        (5, 256, 16),
    ];
    let mut reference = Vec::new();
    let mut worst = 0f32;
    for copies in [1usize, 16] {
        let mut times = vec![Vec::new(); configs.len()];
        for rep in 0..19 {
            for order in 0..configs.len() {
                let v = if rep == 0 {
                    order
                } else {
                    (order + rep) % configs.len()
                };
                let (kind, threads, _width) = configs[v];
                let split = kind != 0;
                for out in &outs[..copies] {
                    unsafe {
                        std::slice::from_raw_parts_mut(out.contents().cast::<f32>(), n).fill(0.);
                    }
                }
                let cb = gpu.command_buffer();
                for i in 0..copies {
                    let e = cb.new_compute_command_encoder();
                    e.set_compute_pipeline_state(match kind {
                        0 => &baseline,
                        1 => &candidate,
                        2 => &cooperative,
                        3 => &width2,
                        4 => &width8,
                        5 => &width16,
                        _ => unreachable!(),
                    });
                    for (index, b) in [
                        (0, &act),
                        (1, &pools[i]),
                        (2, if split { &partial } else { &outs[i] }),
                        (6, &idx),
                        (7, &wgt),
                        (9, &sh),
                        (10, &gate),
                    ] {
                        e.set_buffer(index, Some(b), 0);
                    }
                    for (index, val) in [(3, k as u32), (4, n as u32), (8, 10u32)] {
                        e.set_bytes(index, 4, (&val as *const u32).cast());
                    }
                    e.dispatch_thread_groups(
                        MTLSize::new(
                            n.div_ceil(threads as usize / 32) as u64,
                            if split { 10 } else { 1 },
                            1,
                        ),
                        MTLSize::new(threads as u64, 1, 1),
                    );
                    e.end_encoding();
                    if split {
                        let e = cb.new_compute_command_encoder();
                        e.set_compute_pipeline_state(&reduce);
                        for (index, b) in [(0, &partial), (1, &outs[i]), (2, &sh), (3, &gate)] {
                            e.set_buffer(index, Some(b), 0);
                        }
                        e.set_bytes(4, 4, (&(n as u32) as *const u32).cast());
                        e.dispatch_thread_groups(
                            MTLSize::new(n.div_ceil(64) as u64, 1, 1),
                            MTLSize::new(64, 1, 1),
                        );
                        e.end_encoding();
                    }
                }
                cb.commit();
                cb.wait_until_completed();
                ensure!(
                    cb.status() == metal::MTLCommandBufferStatus::Completed,
                    "GPU failure"
                );
                let start: f64 = unsafe { msg_send![cb, GPUStartTime] };
                let end: f64 = unsafe { msg_send![cb, GPUEndTime] };
                ensure!(end > start, "missing timestamps");
                if rep >= 4 {
                    times[v].push((end - start) * 1000. / copies as f64);
                }
                for out in &outs[..copies] {
                    let got =
                        unsafe { std::slice::from_raw_parts(out.contents().cast::<f32>(), n + 16) };
                    ensure!(got[n..].iter().all(|&x| x == 12345.), "guard overwritten");
                    if rep == 0 && v == 0 {
                        reference = got[..n].to_vec();
                    }
                    for (a, b) in got[..n].iter().zip(&reference) {
                        worst = worst.max((a - b).abs());
                        ensure!(a.is_finite() && (a - b).abs() < 1e-4, "output mismatch");
                    }
                }
                ensure!(
                    unsafe {
                        std::slice::from_raw_parts(partial.contents().cast::<f32>().add(10 * n), 16)
                    }
                    .iter()
                    .all(|&x| x == 12345.),
                    "partial guard overwritten"
                );
            }
        }
        let codebook = [
            -127i32, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ];
        let mut cpuerr = 0f64;
        for row in (0..n).step_by(67) {
            let mut expected = 0.01f32 as f64 / (1. + (-0.2f32 as f64).exp());
            for j in 0..10 {
                let mut sum = 0.;
                for c in 0..k {
                    let off = (ids[j] as usize * n + row) * (k / 32 * 18) + c / 32 * 18;
                    let d =
                        half::f16::from_bits(u16::from_le_bytes([weights[off], weights[off + 1]]))
                            .to_f32();
                    let q = weights[off + 2 + c % 16];
                    let val = if c % 32 < 16 { q & 15 } else { q >> 4 };
                    sum += d as f64 * codebook[val as usize] as f64 * acts[j * k + c] as f64;
                }
                expected += probs[j] as f64 * sum;
            }
            cpuerr = cpuerr.max((reference[row] as f64 - expected).abs());
        }
        ensure!(cpuerr < 1e-4, "CPU error {cpuerr}");
        for t in &mut times {
            t.sort_by(f64::total_cmp);
        }
        for (v, (kind, threads, width)) in configs.iter().enumerate() {
            println!("{{\"copies\":{copies},\"kind\":{kind},\"lanes_per_block\":{width},\"threads\":{threads},\"median_ms\":{},\"p20_ms\":{},\"p80_ms\":{},\"speedup_vs_serial\":{},\"speedup_vs_production\":{},\"max_error\":{worst},\"cpu_error\":{cpuerr}}}",times[v][7],times[v][3],times[v][11],times[0][7]/times[v][7],times[2][7]/times[v][7]);
        }
    }
    Ok(())
}
