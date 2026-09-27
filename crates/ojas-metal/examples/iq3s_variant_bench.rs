//! Isolated IQ3_S gate/up experiment. Production kernels remain unchanged.
use anyhow::{ensure, Result};
use metal::{MTLResourceOptions as Opt, MTLSize};
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device as _;

fn variant_source(src: &str, activation: bool, table: bool, split: bool, id: usize) -> String {
    let begin = src.find("inline float iq3s_group(").unwrap();
    let end = begin + src[begin..].find("// ---- IQ4_XS").unwrap();
    let mut body = src[begin..end].to_string();
    body = body.replace("iq3s_group", &format!("iq3s_group_v{id}"));
    body = body.replace("moe_gu_iq3s", &format!("moe_gu_iq3s_v{id}"));

    if activation {
        body = body.replace("device const float* xg)", "thread const float* xg)");
        body = body.replace(
            "device const float* xg = x + b*256u + g*32u;",
            "float xv[32]; for (uint z=0u; z<32u; ++z) { xv[z]=x[b*256u+g*32u+z]; } thread const float* xg=xv;",
        );
    }
    if table {
        body = body.replace(
            &format!("thread const float* xg) {{",),
            &format!("thread const float* xg, threadgroup const uint* grid) {{"),
        );
        body = body.replace(
            "device const float* xg) {",
            "device const float* xg, threadgroup const uint* grid) {",
        );
        body = body.replace("iq3s_grid[i1]", "grid[i1]");
        body = body.replace("iq3s_grid[i2]", "grid[i2]");
        // Match llama.cpp's Metal path: after staging uint32 entries, address
        // their four bytes directly. Merely staging the table while retaining
        // eight shifts/masks per pair does not test the reference technique.
        body = body.replace(
            "uint g1 = grid[i1], g2 = grid[i2];",
            "threadgroup const uchar* g1 = ((threadgroup const uchar*)grid) + 4u*i1; threadgroup const uchar* g2 = ((threadgroup const uchar*)grid) + 4u*i2;",
        );
        body = body.replace(
            "float v1 = (float)((g1 >> (8u*j)) & 255u);",
            "float v1 = (float)g1[j];",
        );
        body = body.replace(
            "float v2 = (float)((g2 >> (8u*j)) & 255u);",
            "float v2 = (float)g2[j];",
        );
        body = body.replace(
            "uint j = tg.y;",
            "threadgroup uint grid[512]; uint lid=sgid*32u+lane; for(uint z=lid;z<512u;z+=ts.x){grid[z]=iq3s_grid[z];} threadgroup_barrier(mem_flags::mem_threadgroup); uint j = tg.y;",
        );
        body = body.replace(
            &format!("iq3s_group_v{id}(gb, g, xg)"),
            &format!("iq3s_group_v{id}(gb, g, xg, grid)"),
        );
        body = body.replace(
            &format!("iq3s_group_v{id}(ub, g, xg)"),
            &format!("iq3s_group_v{id}(ub, g, xg, grid)"),
        );
        body = body.replace(
            &format!("iq3s_group_v{id}_cached(gb, g, xv)"),
            &format!("iq3s_group_v{id}_cached(gb, g, xv, grid)"),
        );
        body = body.replace(
            &format!("iq3s_group_v{id}_cached(ub, g, xv)"),
            &format!("iq3s_group_v{id}_cached(ub, g, xv, grid)"),
        );
    }
    if split {
        body = body.replace("float gsum = 0.0f;", "float2 gsum = float2(0.0f);");
        body = body.replace("gsum += v1 *", "gsum.x += v1 *");
        body = body.replace("gsum += v2 *", "gsum.y += v2 *");
        body = body.replace(")) * gsum;", ")) * (gsum.x + gsum.y);");
    }
    format!("{src}\n{body}")
}

fn row_source(src: &str, rows: u32) -> String {
    let begin = src.find("inline float iq3s_group_cached(").unwrap();
    let end = begin + src[begin..].find("// ---- IQ4_XS").unwrap();
    let name = format!("moe_gu_iq3s_rows{rows}");
    let helper = format!("iq3s_group_cached_rows{rows}");
    let mut body = src[begin..end]
        .replace("iq3s_group_cached", &helper)
        .replace("moe_gu_iq3s", &name);
    body = body
        .replace(
            "uint row0 = tg.x*8u + sgid*4u;",
            &format!("uint row0 = tg.x*(ts.x/32u)*{rows}u + sgid*{rows}u;"),
        )
        .replace(
            "float gs[4] = {0.0f, 0.0f, 0.0f, 0.0f};",
            &format!("float gs[{rows}] = {{0.0f}};"),
        )
        .replace(
            "float us[4] = {0.0f, 0.0f, 0.0f, 0.0f};",
            &format!("float us[{rows}] = {{0.0f}};"),
        )
        .replace("r < 4u", &format!("r < {rows}u"));
    format!("{src}\n{body}")
}

fn main() -> Result<()> {
    let gpu = ojas_metal::MetalGpu::new()?;
    let src = ojas_metal::kernels::family_source("moe_iq").unwrap();
    let configs = [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (false, false, true),
        (true, true, false),
        (true, false, true),
        (false, true, true),
        (true, true, true),
    ];
    let baseline = gpu.pipeline(src, "moe_gu_iq3s")?;
    let mut pipes = vec![baseline];
    for (id, &(activation, table, split)) in configs.iter().enumerate().skip(1) {
        pipes.push(gpu.pipeline(
            &variant_source(src, activation, table, split, id),
            &format!("moe_gu_iq3s_v{id}"),
        )?);
    }

    let (k, n, selected, experts, copies) = (2560usize, 640usize, 10usize, 32usize, 5usize);
    let row_bytes = k / 256 * 110;
    let tensor_bytes = experts * n * row_bytes;
    let mut packed = vec![0u8; tensor_bytes];
    let mut seed = 0x7a31_8f25u32;
    for block in packed.chunks_exact_mut(110) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        block[..2].copy_from_slice(
            &half::f16::from_f32((seed % 31 + 1) as f32 / 8192.0)
                .to_bits()
                .to_le_bytes(),
        );
        for byte in &mut block[2..] {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *byte = seed as u8;
        }
    }
    let weights: Vec<_> = (0..copies * 2)
        .map(|_| {
            gpu.device.new_buffer_with_data(
                packed.as_ptr().cast(),
                tensor_bytes as u64,
                Opt::StorageModeShared,
            )
        })
        .collect();
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 137 % 1021) as f32 - 510.0) / 511.0)
        .collect();
    let upload_f32 = |values: &[f32]| {
        gpu.device.new_buffer_with_data(
            values.as_ptr().cast(),
            (values.len() * 4) as u64,
            Opt::StorageModeShared,
        )
    };
    let xb = upload_f32(&x);
    let expert_ids = [17u32, 3, 29, 7, 0, 23, 9, 31, 12, 5];
    let ids = gpu.device.new_buffer_with_data(
        expert_ids.as_ptr().cast(),
        (selected * 4) as u64,
        Opt::StorageModeShared,
    );
    let outputs: Vec<_> = (0..copies)
        .map(|_| upload_f32(&vec![12345.0; selected * n + 16]))
        .collect();
    let mut reference = Vec::new();
    let mut max_error = vec![0.0f32; configs.len()];
    let mut times = vec![Vec::new(); configs.len()];

    for rep in 0..19 {
        for order in 0..configs.len() {
            let variant = if rep == 0 {
                order
            } else {
                (order + rep) % configs.len()
            };
            for output in &outputs {
                unsafe {
                    let values = std::slice::from_raw_parts_mut(
                        output.contents().cast::<f32>(),
                        selected * n + 16,
                    );
                    values[..selected * n].fill(f32::NAN);
                    values[selected * n..].fill(12345.0);
                }
            }
            let cb = gpu.command_buffer();
            for copy in 0..copies {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipes[variant]);
                enc.set_buffer(0, Some(&xb), 0);
                enc.set_buffer(1, Some(&weights[copy * 2]), 0);
                enc.set_buffer(2, Some(&weights[copy * 2 + 1]), 0);
                enc.set_buffer(3, Some(&outputs[copy]), 0);
                enc.set_bytes(4, 4, (&(k as u32) as *const u32).cast());
                enc.set_bytes(5, 4, (&(n as u32) as *const u32).cast());
                enc.set_buffer(8, Some(&ids), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(8) as u64, selected as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
                enc.end_encoding();
            }
            cb.commit();
            cb.wait_until_completed();
            ensure!(
                cb.status() == metal::MTLCommandBufferStatus::Completed,
                "GPU command failed"
            );
            let start: f64 = unsafe { msg_send![cb, GPUStartTime] };
            let end: f64 = unsafe { msg_send![cb, GPUEndTime] };
            ensure!(end > start, "GPU timestamps unavailable");
            if rep >= 4 {
                times[variant].push((end - start) * 1000.0 / copies as f64);
            }
            for output in &outputs {
                let got = unsafe {
                    std::slice::from_raw_parts(output.contents().cast::<f32>(), selected * n + 16)
                };
                ensure!(
                    got[selected * n..].iter().all(|&v| v == 12345.0),
                    "output guard overwritten"
                );
                if rep == 0 && variant == 0 {
                    reference = got[..selected * n].to_vec();
                }
                for (&actual, &want) in got[..selected * n].iter().zip(&reference) {
                    let error = (actual - want).abs();
                    max_error[variant] = max_error[variant].max(error);
                    ensure!(
                        actual.is_finite() && error < 1e-3,
                        "variant {variant} mismatch: {error}"
                    );
                }
            }
        }
    }

    let mut cpu_error = 0.0f32;
    for slot in 0..selected {
        for row in (0..n).step_by(37) {
            let expert = expert_ids[slot] as usize;
            let offset = (expert * n + row) * row_bytes;
            let gate = ojas_cpu::cpu_math::dot_iq3s(&packed[offset..offset + row_bytes], &x);
            let up_offset = offset;
            let up = ojas_cpu::cpu_math::dot_iq3s(&packed[up_offset..up_offset + row_bytes], &x);
            let want = (gate / (1.0 + (-gate).exp())) * up;
            cpu_error = cpu_error.max((reference[slot * n + row] - want).abs());
        }
    }
    ensure!(cpu_error < 2e-3, "CPU reference mismatch: {cpu_error}");
    for values in &mut times {
        values.sort_by(f64::total_cmp);
    }
    for (variant, &(activation, table, split)) in configs.iter().enumerate() {
        println!(
            "{{\"activation_cache\":{activation},\"threadgroup_table\":{table},\"split_accumulator\":{split},\"median_ms\":{},\"p20_ms\":{},\"p80_ms\":{},\"speedup\":{},\"max_error\":{},\"cpu_error\":{cpu_error}}}",
            times[variant][7],
            times[variant][3],
            times[variant][11],
            times[0][7] / times[variant][7],
            max_error[variant]
        );
    }

    // Geometry sweep of the accepted activation-cache body. Keep the arithmetic
    // fixed and vary only rows per SIMDgroup and SIMDgroups per threadgroup.
    let row_counts = [1u32, 2, 4, 8];
    let mut row_pipes = Vec::new();
    for rows in row_counts {
        row_pipes.push(gpu.pipeline(&row_source(src, rows), &format!("moe_gu_iq3s_rows{rows}"))?);
    }
    let row_configs: Vec<_> = row_counts
        .into_iter()
        .flat_map(|rows| [32u32, 64, 128, 256].map(|threads| (rows, threads)))
        .collect();
    let mut row_times = vec![Vec::new(); row_configs.len()];
    let mut row_error = vec![0.0f32; row_configs.len()];
    for rep in 0..19 {
        for order in 0..row_configs.len() {
            let variant = if rep == 0 {
                order
            } else {
                (order + rep) % row_configs.len()
            };
            let (rows, threads) = row_configs[variant];
            for output in &outputs {
                unsafe {
                    std::slice::from_raw_parts_mut(output.contents().cast::<f32>(), selected * n)
                        .fill(f32::NAN);
                }
            }
            let cb = gpu.command_buffer();
            for copy in 0..copies {
                let enc = cb.new_compute_command_encoder();
                let pi = row_counts.iter().position(|&r| r == rows).unwrap();
                enc.set_compute_pipeline_state(&row_pipes[pi]);
                for (i, b) in [
                    (0, &xb),
                    (1, &weights[copy * 2]),
                    (2, &weights[copy * 2 + 1]),
                    (3, &outputs[copy]),
                    (8, &ids),
                ] {
                    enc.set_buffer(i, Some(b), 0);
                }
                enc.set_bytes(4, 4, (&(k as u32) as *const u32).cast());
                enc.set_bytes(5, 4, (&(n as u32) as *const u32).cast());
                let covered = rows as usize * (threads as usize / 32);
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(covered) as u64, selected as u64, 1),
                    MTLSize::new(threads as u64, 1, 1),
                );
                enc.end_encoding();
            }
            cb.commit();
            cb.wait_until_completed();
            ensure!(
                cb.status() == metal::MTLCommandBufferStatus::Completed,
                "geometry GPU command failed"
            );
            let start: f64 = unsafe { msg_send![cb, GPUStartTime] };
            let end: f64 = unsafe { msg_send![cb, GPUEndTime] };
            if rep >= 4 {
                row_times[variant].push((end - start) * 1000.0 / copies as f64);
            }
            for output in &outputs {
                let got = unsafe {
                    std::slice::from_raw_parts(output.contents().cast::<f32>(), selected * n + 16)
                };
                ensure!(
                    got[selected * n..].iter().all(|&v| v == 12345.0),
                    "geometry guard overwritten"
                );
                for (&actual, &want) in got[..selected * n].iter().zip(&reference) {
                    let error = (actual - want).abs();
                    row_error[variant] = row_error[variant].max(error);
                    ensure!(actual.is_finite() && error < 1e-3, "geometry mismatch");
                }
            }
        }
    }
    for values in &mut row_times {
        values.sort_by(f64::total_cmp);
    }
    let production = row_configs
        .iter()
        .position(|&(rows, threads)| rows == 4 && threads == 64)
        .unwrap();
    for (variant, &(rows, threads)) in row_configs.iter().enumerate() {
        println!(
            "{{\"geometry\":true,\"rows\":{rows},\"threads\":{threads},\"median_ms\":{},\"p20_ms\":{},\"p80_ms\":{},\"speedup_vs_production\":{},\"max_error\":{}}}",
            row_times[variant][7],
            row_times[variant][3],
            row_times[variant][11],
            row_times[production][7] / row_times[variant][7],
            row_error[variant]
        );
    }
    Ok(())
}
