//! Production-shaped qwen4exp HC chain: seven-dispatch baseline vs three passes.
//! Synthetic Q8_0/F32 weights; no production dispatch changes.
use anyhow::{ensure, Result};
use metal::{MTLResourceOptions as Opt, MTLSize};
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device as _;
use std::ffi::c_void;

const D: usize = 2560;
const HC: usize = 4;
const K: usize = D * HC;
const LR: usize = 320;
const Q8_BLOCK: usize = 34;

const FUSED: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float q8(device const uchar * w, uint row, uint k, uint K) {
    ulong b = ((ulong)row * (K / 32u) + k / 32u) * 34u;
    ushort bits = (ushort)w[b] | ((ushort)w[b + 1u] << 8);
    return float(as_type<half>(bits)) * float(as_type<char>(w[b + 2u + k % 32u]));
}

// Rows 0..319 are Q8_0 down weights. Rows 320..323 are the F32 injection
// weights. One dispatch also applies the two different output activations.
kernel void hc_down_inject(device const float * xn [[buffer(0)]],
    device const uchar * down [[buffer(1)]], device const float * inject [[buffer(2)]],
    device float * lo [[buffer(3)]], device float * inj [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4];
    float acc = 0.0f;
    if (row < 320u) {
        for (uint b = tid/4u; b < 320u; b += 32u) {
            ulong off = ((ulong)row*320ul + b)*34ul;
            ushort bits = (ushort)down[off] | ((ushort)down[off+1u] << 8);
            float d = float(as_type<half>(bits));
            uint col = b*32u + (tid & 3u)*8u;
            float dot = 0.0f;
            for (uint z = 0u; z < 8u; ++z) dot += xn[col+z]*float(as_type<char>(down[off+2u+(tid&3u)*8u+z]));
            acc += d*dot;
        }
    } else {
        uint r = row - 320u;
        for (uint i = tid; i < 10240u; i += 128u) acc += xn[i] * inject[r * 10240u + i];
    }
    acc = simd_sum(acc);
    if (lane == 0u) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float z = part[0] + part[1] + part[2] + part[3];
        if (row < 320u) {
            z *= 0.25f;
            lo[row] = z / (1.0f + exp(-z));
        } else {
            inj[row - 320u] = z;
        }
    }
}

// One threadgroup owns hidden feature i across all four HC streams. It reads
// the same low-rank activation for four Q8_0 up rows, applies each gate to xn,
// and writes the collapsed mean without materializing graw/gated.
kernel void hc_up_gate_collapse(device const float * lo [[buffer(0)]],
    device const uchar * up [[buffer(1)]], device const float * xn [[buffer(2)]],
    device float * mixed [[buffer(3)]],
    uint i [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4][4];
    float4 acc = 0.0f;
    for (uint b = tid/8u; b < 10u; b += 16u) {
        uint qoff = (tid & 7u)*4u;
        uint col = b*32u + qoff;
        float4 dot = 0.0f;
        for (uint c = 0u; c < 4u; ++c) {
            uint row = c*2560u + i;
            ulong off = ((ulong)row*10ul + b)*34ul;
            ushort bits = (ushort)up[off] | ((ushort)up[off+1u] << 8);
            float d = float(as_type<half>(bits));
            float s = 0.0f;
            for (uint z = 0u; z < 4u; ++z) s += lo[col+z]*float(as_type<char>(up[off+2u+qoff+z]));
            dot[c] = d*s;
        }
        acc += dot;
    }
    acc.x = simd_sum(acc.x); acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z); acc.w = simd_sum(acc.w);
    if (lane == 0u) part[sg][0] = acc.x;
    if (lane == 0u) part[sg][1] = acc.y;
    if (lane == 0u) part[sg][2] = acc.z;
    if (lane == 0u) part[sg][3] = acc.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float out = 0.0f;
        for (uint c = 0u; c < 4u; ++c) {
            float z = part[0][c] + part[1][c] + part[2][c] + part[3][c];
            out += xn[c * 2560u + i] / (1.0f + exp(-z));
        }
        mixed[i] = out * 0.25f;
    }
}
"#;

fn main() -> Result<()> {
    ensure!(
        std::env::var_os("OJAS_NO_FAST").is_none(),
        "unset OJAS_NO_FAST"
    );
    let gpu = ojas_metal::MetalGpu::new()?;
    let qsrc = ojas_metal::kernels::family_source("gemv").unwrap();
    let hsrc = ojas_metal::kernels::family_source("qwen4exp").unwrap();
    let msrc = ojas_metal::kernels::family_source("moe").unwrap();
    let q80 = gpu.pipeline(qsrc, "gemv_nat_q80")?;
    let norm = gpu.pipeline(hsrc, "hc_rmsnorm")?;
    let silu = gpu.pipeline(hsrc, "hc_silu_scale")?;
    let gate = gpu.pipeline(hsrc, "hc_gate")?;
    let collapse = gpu.pipeline(hsrc, "hc_collapse")?;
    let f32mv = gpu.pipeline(msrc, "gemv_w32")?;
    let fused_down = gpu.pipeline(FUSED, "hc_down_inject")?;
    let fused_up = gpu.pipeline(FUSED, "hc_up_gate_collapse")?;

    let upload = |p: *const c_void, bytes: usize| {
        gpu.device
            .new_buffer_with_data(p, bytes as u64, Opt::StorageModeShared)
    };
    let fbuf = |v: &[f32]| upload(v.as_ptr().cast(), v.len() * 4);
    let mut seed = 0x9e37_79b9u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let xn: Vec<f32> = (0..K)
        .map(|_| (next() as i32 as f32) / i32::MAX as f32)
        .collect();
    let gamma: Vec<f32> = (0..K).map(|i| 0.8 + (i % 31) as f32 / 100.).collect();
    let make_q8 = |rows: usize, cols: usize, next: &mut dyn FnMut() -> u32| {
        let mut w = vec![0u8; rows * cols / 32 * Q8_BLOCK];
        for b in w.chunks_exact_mut(Q8_BLOCK) {
            let s = half::f16::from_f32((next() % 31 + 1) as f32 / 4096.)
                .to_bits()
                .to_le_bytes();
            b[..2].copy_from_slice(&s);
            for q in &mut b[2..] {
                *q = next() as u8;
            }
        }
        w
    };
    let down = make_q8(LR, K, &mut next);
    let up = make_q8(K, LR, &mut next);
    let inject: Vec<f32> = (0..HC * K)
        .map(|_| (next() as i32 as f32) / i32::MAX as f32 / 100.)
        .collect();
    let x = fbuf(&xn);
    let g = fbuf(&gamma);
    // 96 distinct Metal allocations reproduce the production chain's cold
    // footprint while keeping contents identical for a stable correctness gate.
    let weights: Vec<_> = (0..96)
        .map(|_| {
            (
                upload(down.as_ptr().cast(), down.len()),
                upload(up.as_ptr().cast(), up.len()),
                fbuf(&inject),
            )
        })
        .collect();
    let alloc = |n: usize| {
        gpu.device
            .new_buffer((n * 4) as u64, Opt::StorageModeShared)
    };
    let xn_b = alloc(K);
    let lo_b = alloc(LR);
    let raw_b = alloc(K);
    let gated_b = alloc(K);
    let mixed_b = alloc(D);
    let inj_b = alloc(HC);
    let xn_c = alloc(K);
    let lo_c = alloc(LR);
    let mixed_c = alloc(D);
    let inj_c = alloc(HC);

    let barrier = |e: &metal::ComputeCommandEncoderRef| unsafe {
        let _: () = msg_send![e, memoryBarrierWithScope: 1u64];
    };
    let set_u32 = |e: &metal::ComputeCommandEncoderRef, i: u64, v: u32| {
        e.set_bytes(i, 4, (&v as *const u32).cast());
    };
    let qmv = |e: &metal::ComputeCommandEncoderRef,
               w: &metal::BufferRef,
               input: &metal::BufferRef,
               output: &metal::BufferRef,
               k: usize,
               n: usize| {
        e.set_compute_pipeline_state(&q80);
        e.set_buffer(0, Some(input), 0);
        e.set_buffer(1, Some(w), 0);
        e.set_buffer(2, Some(output), 0);
        for (i, v) in [(3, k as u32), (4, n as u32), (7, 1), (8, 0)] {
            set_u32(e, i, v);
        }
        e.dispatch_thread_groups(
            MTLSize::new(n.div_ceil(2) as u64, 1, 1),
            MTLSize::new(128, 1, 1),
        );
    };
    let norm_run = |e: &metal::ComputeCommandEncoderRef, out: &metal::BufferRef| {
        e.set_compute_pipeline_state(&norm);
        e.set_buffer(0, Some(&x), 0);
        e.set_buffer(1, Some(&g), 0);
        e.set_buffer(2, Some(out), 0);
        set_u32(e, 3, D as u32);
        let eps = 1e-6f32;
        e.set_bytes(4, 4, (&eps as *const f32).cast());
        set_u32(e, 5, HC as u32);
        e.dispatch_thread_groups(MTLSize::new(HC as u64, 1, 1), MTLSize::new(256, 1, 1));
    };

    let mut reference_mixed = Vec::new();
    let mut reference_inj = Vec::new();
    let mut times = [Vec::new(), Vec::new()];
    for copies in [1usize, 96] {
        times.iter_mut().for_each(Vec::clear);
        let mut max_error = 0f32;
        for rep in 0..25 {
            for turn in 0..2 {
                let variant = if rep == 0 { turn } else { (turn + rep) % 2 };
                let cb = gpu.command_buffer();
                let e = cb.new_compute_command_encoder();
                for (down_b, up_b, inject_b) in &weights[..copies] {
                    if variant == 0 {
                        norm_run(&e, &xn_b);
                        barrier(&e);
                        qmv(&e, down_b, &xn_b, &lo_b, K, LR);
                        barrier(&e);
                        e.set_compute_pipeline_state(&silu);
                        e.set_buffer(0, Some(&lo_b), 0);
                        set_u32(&e, 1, LR as u32);
                        set_u32(&e, 2, HC as u32);
                        e.dispatch_thread_groups(
                            MTLSize::new(LR.div_ceil(64) as u64, 1, 1),
                            MTLSize::new(64, 1, 1),
                        );
                        barrier(&e);
                        qmv(&e, up_b, &lo_b, &raw_b, LR, K);
                        barrier(&e);
                        e.set_compute_pipeline_state(&gate);
                        e.set_buffer(0, Some(&xn_b), 0);
                        e.set_buffer(1, Some(&raw_b), 0);
                        e.set_buffer(2, Some(&gated_b), 0);
                        set_u32(&e, 3, K as u32);
                        e.dispatch_thread_groups(
                            MTLSize::new(K.div_ceil(64) as u64, 1, 1),
                            MTLSize::new(64, 1, 1),
                        );
                        barrier(&e);
                        e.set_compute_pipeline_state(&collapse);
                        e.set_buffer(0, Some(&gated_b), 0);
                        e.set_buffer(1, Some(&mixed_b), 0);
                        set_u32(&e, 2, D as u32);
                        set_u32(&e, 3, HC as u32);
                        e.dispatch_thread_groups(
                            MTLSize::new(D.div_ceil(64) as u64, 1, 1),
                            MTLSize::new(64, 1, 1),
                        );
                        e.set_compute_pipeline_state(&f32mv);
                        e.set_buffer(0, Some(&xn_b), 0);
                        e.set_buffer(1, Some(inject_b), 0);
                        e.set_buffer(2, Some(&inj_b), 0);
                        set_u32(&e, 3, K as u32);
                        set_u32(&e, 4, HC as u32);
                        e.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
                    } else {
                        norm_run(&e, &xn_c);
                        barrier(&e);
                        e.set_compute_pipeline_state(&fused_down);
                        e.set_buffer(0, Some(&xn_c), 0);
                        e.set_buffer(1, Some(down_b), 0);
                        e.set_buffer(2, Some(inject_b), 0);
                        e.set_buffer(3, Some(&lo_c), 0);
                        e.set_buffer(4, Some(&inj_c), 0);
                        e.dispatch_thread_groups(
                            MTLSize::new((LR + HC) as u64, 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                        barrier(&e);
                        e.set_compute_pipeline_state(&fused_up);
                        e.set_buffer(0, Some(&lo_c), 0);
                        e.set_buffer(1, Some(up_b), 0);
                        e.set_buffer(2, Some(&xn_c), 0);
                        e.set_buffer(3, Some(&mixed_c), 0);
                        e.dispatch_thread_groups(
                            MTLSize::new(D as u64, 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                    }
                }
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                ensure!(
                    cb.status() == metal::MTLCommandBufferStatus::Completed,
                    "GPU command failed"
                );
                let (mb, ib) = if variant == 0 {
                    (&mixed_b, &inj_b)
                } else {
                    (&mixed_c, &inj_c)
                };
                let got_m = unsafe { std::slice::from_raw_parts(mb.contents().cast::<f32>(), D) };
                let got_i = unsafe { std::slice::from_raw_parts(ib.contents().cast::<f32>(), HC) };
                ensure!(
                    got_m.iter().chain(got_i).all(|v| v.is_finite()),
                    "nonfinite output"
                );
                if rep == 0 && variant == 0 {
                    reference_mixed = got_m.to_vec();
                    reference_inj = got_i.to_vec();
                }
                if !reference_mixed.is_empty() {
                    let err = got_m
                        .iter()
                        .zip(&reference_mixed)
                        .chain(got_i.iter().zip(&reference_inj))
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    max_error = max_error.max(err);
                    ensure!(err < 2e-4, "candidate mismatch: {err}");
                }
                if rep >= 5 {
                    let start: f64 = unsafe { msg_send![cb, GPUStartTime] };
                    let end: f64 = unsafe { msg_send![cb, GPUEndTime] };
                    ensure!(end > start, "missing GPU timestamp");
                    times[variant].push((end - start) * 1000.0 / copies as f64);
                }
            }
        }
        for t in &mut times {
            t.sort_by(f64::total_cmp);
        }
        println!("{{\"copies\":{copies},\"baseline_ms\":{},\"candidate_ms\":{},\"speedup\":{},\"saving_ms_per_hc\":{},\"max_error\":{max_error}}}",
            times[0][10], times[1][10], times[0][10]/times[1][10], times[0][10]-times[1][10]);
    }
    Ok(())
}
