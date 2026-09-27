// The Metal device is macOS-only (`ojas-metal/src/lib.rs`); only its `kernels` source table
// builds elsewhere. These tests drive a real `MetalGpu`, so they compile only where one exists.
#![cfg(target_os = "macos")]

use ojas_core::{Device, KernelRuntime, Tier};
use ojas_metal::MetalGpu;

const TEST_FAMILY: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void vadd(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]], constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) { c[i] = a[i] + b[i]; }
}
kernel void rowsum(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& k [[buffer(2)]],
    uint r [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    float s = 0.0;
    for (uint i = lane; i < k; i += 32) { s += x[r * k + i]; }
    s = simd_sum(s);
    if (lane == 0) { out[r] = s; }
}
"#;

#[test]
fn compile_dispatch_verify() {
    let mut gpu = MetalGpu::new().unwrap();
    assert_eq!(gpu.caps().tier, Tier::A);
    gpu.register_family("test", TEST_FAMILY);
    gpu.ensure_family("test").unwrap();
    assert!(gpu.has_kernel("vadd") && gpu.has_kernel("rowsum"));

    let n = 1024usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| 2.0 * i as f32).collect();
    let (ba, bb, bc) = (gpu.upload(&a), gpu.upload(&b), gpu.alloc(n));
    let enc = gpu.begin();
    gpu.dispatch(
        &enc,
        "vadd",
        &[(&ba, 0), (&bb, 0), (&bc, 0)],
        &[n as u32],
        [(n as u32).div_ceil(256), 1, 1],
        [256, 1, 1],
    )
    .unwrap();
    let x: Vec<f32> = (0..8 * 64).map(|i| (i % 7) as f32).collect();
    let bx = gpu.upload(&x);
    let bs = gpu.alloc(8);
    gpu.dispatch(
        &enc,
        "rowsum",
        &[(&bx, 0), (&bs, 0)],
        &[64],
        [8, 1, 1],
        [32, 1, 1],
    )
    .unwrap();
    gpu.submit(enc).unwrap();

    let c = gpu.read(&bc);
    assert!(c.iter().enumerate().all(|(i, &v)| v == 3.0 * i as f32));
    let s = gpu.read(&bs);
    for r in 0..8 {
        let expect: f32 = (0..64).map(|i| ((r * 64 + i) % 7) as f32).sum();
        assert!(
            (s[r] - expect).abs() < 1e-3,
            "row {r}: {} vs {expect}",
            s[r]
        );
    }
}

/// Every kernel family must compile and build all of its pipelines on this GPU.
///
/// `manifest.rs` checks that kernel names resolve within the source, not that the source
/// is valid MSL. A family that fails to compile otherwise surfaces only when a model needs
/// it, which on a large model is minutes into a load, far from the edit that broke it.
/// Compiling every family here costs seconds.
#[test]
fn every_family_compiles() {
    let gpu = MetalGpu::new().unwrap();
    let mut total = 0usize;
    for (fam, src) in ojas_metal::kernels::families() {
        let pipes = gpu
            .compile_all(src, |_| true)
            .unwrap_or_else(|e| panic!("kernel family {fam} failed to compile: {e}"));
        assert!(!pipes.is_empty(), "kernel family {fam} built no pipelines");
        total += pipes.len();
    }
    assert!(total > 190, "expected the full kernel surface, got {total}");
}

/// `gemv_w32` against a plain f32 dot product.
///
/// It is the fallback `mm()` takes for weights a GGUF keeps in f32, which on the streamed
/// path includes `ssm_alpha`/`ssm_beta` — the two projections feeding the gated-DeltaNet
/// decay gate, where an error compounds down the layer stack rather than staying local.
/// Nothing else exercises it until a model ships those tensors as f32.
#[test]
fn gemv_w32_matches_cpu_dot() {
    use metal::MTLResourceOptions;
    use ojas_core::Device as _;
    let gpu = MetalGpu::new().unwrap();
    let src = ojas_metal::kernels::family_source("moe").expect("moe family");
    let pipes = gpu
        .compile_all(src, |n| {
            matches!(n, "gemv_w32" | "gemv_w32_accum" | "gemv_w32_bias")
        })
        .unwrap();
    assert_eq!(pipes.len(), 3);
    for (entry, pipe) in pipes {
        // Shapes that matter here: K=2560 with the small N of an alpha/beta projection,
        // and an N that is not a multiple of the 8 rows a threadgroup covers.
        for (k, n) in [
            (2560usize, 48usize),
            (2560, 32),
            (2560, 13),
            (512, 8),
            (13, 7),
        ] {
            let w: Vec<f32> = (0..k * n)
                .map(|i| ((i * 2654435761) % 1024) as f32 / 512.0 - 1.0)
                .collect();
            let x: Vec<f32> = (0..k)
                .map(|i| ((i * 40503) % 1024) as f32 / 512.0 - 1.0)
                .collect();
            let want: Vec<f32> = (0..n)
                .map(|r| {
                    (0..k).map(|j| w[r * k + j] * x[j]).sum::<f32>()
                        + if entry.ends_with("_accum") {
                            0.25
                        } else if entry.ends_with("_bias") {
                            r as f32 / 10.0
                        } else {
                            0.0
                        }
                })
                .collect();

            let dev = gpu.device.clone();
            let wb = dev.new_buffer_with_data(
                w.as_ptr() as *const _,
                (w.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let xb = dev.new_buffer_with_data(
                x.as_ptr() as *const _,
                (x.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let init = vec![0.25f32; n];
            let biases: Vec<f32> = (0..n).map(|r| r as f32 / 10.0).collect();
            let yb = dev.new_buffer_with_data(
                init.as_ptr() as *const _,
                (n * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let bb = dev.new_buffer_with_data(
                biases.as_ptr() as *const _,
                (n * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let cb = gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_buffer(1, Some(&wb), 0);
            enc.set_buffer(2, Some(&yb), 0);
            if entry.ends_with("_bias") {
                enc.set_buffer(5, Some(&bb), 0);
            }
            let (kk, nn) = (k as u32, n as u32);
            enc.set_bytes(3, 4, &kk as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &nn as *const u32 as *const std::ffi::c_void);
            // The dispatch mm() uses: 8 rows per 256-thread threadgroup.
            enc.dispatch_thread_groups(
                metal::MTLSize::new(((nn + 7) / 8) as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let got = unsafe { std::slice::from_raw_parts(yb.contents() as *const f32, n) };
            for r in 0..n {
                let rel = (got[r] - want[r]).abs() / want[r].abs().max(1e-3);
                assert!(
                    rel < 1e-4,
                    "gemv_w32 K={k} N={n} row {r}: gpu {} vs cpu {}",
                    got[r],
                    want[r]
                );
            }
        }
    }
}

/// Independent scalar causal convolution checks history, dilation and GGUF
/// [kernel, channel] strides, including a restored speculative branch.
#[test]
fn ple_dilated_history_matches_scalar_and_restores() {
    use metal::{MTLResourceOptions, MTLSize};
    let gpu = MetalGpu::new().unwrap();
    let src = ojas_metal::kernels::family_source("qwen4exp").unwrap();
    let pipe = gpu.pipeline(src, "ple_finish").unwrap();
    let (d, hc, kern, dilation) = (65u32, 4u32, 4u32, 3u32);
    let channels = (d * hc) as usize;
    let hist = ((kern - 1) * dilation) as usize;
    let upload = |x: &[f32]| {
        gpu.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let weights: Vec<f32> = (0..channels * kern as usize)
        .map(|i| ((i * 17 % 31) as f32 - 15.0) / 31.0)
        .collect();
    let w = upload(&weights);
    let value = upload(&vec![0.2; d as usize]);
    let gate = upload(&vec![0.7; hc as usize]);
    // Canary prefix verifies that this kernel respects the state-buffer offset
    // used to place PLE after DeltaNet's existing convolution history.
    let prefix = 32usize;
    let mut initial = vec![0.0; prefix + channels * hist];
    initial[..prefix].fill(123.0);
    let state = upload(&initial);
    let inputs: Vec<Vec<f32>> = (0..18)
        .map(|t| {
            (0..channels)
                .map(|c| ((t * 7 + c * 3) % 29) as f32 / 29.0 - 0.5)
                .collect()
        })
        .collect();
    let mut saved = Vec::new();
    for t in (0..18).chain(6..18) {
        if t == 6 && !saved.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    saved.as_ptr(),
                    state.contents() as *mut f32,
                    saved.len(),
                );
            }
        }
        let x = upload(&inputs[t]);
        let out = upload(&vec![0.3; channels]);
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        for (i, b) in [&out, &value, &gate, &x, &w].iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        for (i, v) in [(5, d), (6, hc), (8, kern), (9, dilation)] {
            enc.set_bytes(i, 4, &v as *const u32 as *const _);
        }
        enc.set_buffer(7, Some(&state), (prefix * 4) as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(d.div_ceil(64) as u64, hc as u64, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, channels) };
        for c in 0..channels {
            let mut cv = 0.0;
            for k in 0..kern as usize {
                let lag = (kern as usize - 1 - k) * dilation as usize;
                if t >= lag {
                    cv += weights[c * kern as usize + k] * inputs[t - lag][c];
                }
            }
            let want = 0.3 + 0.2 * 0.7 + cv / (1.0 + (-cv).exp());
            assert!(
                (got[c] - want).abs() < 2e-6,
                "token {t} channel {c}: {} != {want}",
                got[c]
            );
        }
        let current =
            unsafe { std::slice::from_raw_parts(state.contents() as *const f32, initial.len()) };
        assert!(current[..prefix].iter().all(|v| *v == 123.0));
        if t == 5 {
            saved = current.to_vec();
        }
    }
}

/// Q8_0 embedding lookup must preserve values that are not representable in
/// F16. Also exercise signed bytes, multiple rows, buffer offsets and tail lanes.
#[test]
fn native_q8_embedding_preserves_float_products() {
    use metal::{MTLResourceOptions, MTLSize};
    let gpu = MetalGpu::new().unwrap();
    let pipe = gpu
        .pipeline(
            ojas_metal::kernels::family_source("ops").unwrap(),
            "embed_gguf_q8_0",
        )
        .unwrap();
    let d = 96u32;
    let mut raw = vec![0u8; 64 + 6 * 34];
    let scales = [0.333251953125f32, 1.0, 2.001953125];
    let mut expected = vec![vec![0f32; 96]; 2];
    for row in 0..2 {
        for block in 0..3 {
            let scale = scales[(row + block) % 3];
            let at = 64 + (row * 3 + block) * 34;
            raw[at..at + 2].copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            for i in 0..32 {
                let q = ((row * 97 + block * 29 + i * 7) % 256) as i16 - 128;
                raw[at + 2 + i] = q as i8 as u8;
                expected[row][block * 32 + i] = scale * q as f32;
            }
        }
    }
    assert!(expected
        .iter()
        .flatten()
        .any(|&v| half::f16::from_f32(v).to_f32() != v));
    let emb = gpu.device.new_buffer_with_data(
        raw.as_ptr() as *const _,
        raw.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    for token in 0..2u32 {
        let initial = vec![12345.0f32; 112];
        let out = gpu.device.new_buffer_with_data(
            initial.as_ptr() as *const _,
            448,
            MTLResourceOptions::StorageModeShared,
        );
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&emb), 64);
        enc.set_buffer(1, Some(&out), 32);
        enc.set_bytes(2, 4, &d as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(3, 4, &token as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(MTLSize::new(2, 1, 1), MTLSize::new(64, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, 112) };
        assert_eq!(&got[8..104], expected[token as usize].as_slice());
        assert!(got[..8].iter().chain(&got[104..]).all(|&v| v == 12345.0));
    }
}

/// The decode Q8_0 matvec splits K across four SIMDgroups and reduces their
/// partials through threadgroup memory. Exercise an odd output tail and a real
/// Flash input width so a wrong row stride or incomplete K partition is visible.
#[test]
fn native_q8_matvec_ksplit_matches_cpu() {
    use metal::{MTLResourceOptions, MTLSize};
    let gpu = MetalGpu::new().unwrap();
    let pipe = gpu
        .pipeline(
            ojas_metal::kernels::family_source("gemv").unwrap(),
            "gemv_nat_q80",
        )
        .unwrap();
    let (k, n) = (2560usize, 13usize);
    let nb = k / 32;
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 37 % 257) as f32 - 128.0) / 127.0)
        .collect();
    let mut raw = vec![0u8; n * nb * 34];
    let mut want = vec![0.0f32; n];
    for r in 0..n {
        for b in 0..nb {
            let scale = half::f16::from_f32(0.003 + ((r + b) % 7) as f32 * 0.001).to_f32();
            let at = (r * nb + b) * 34;
            raw[at..at + 2].copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            for j in 0..32 {
                let q = ((r * 17 + b * 11 + j * 5) % 31) as i8 - 15;
                raw[at + 2 + j] = q as u8;
                want[r] += scale * q as f32 * x[b * 32 + j];
            }
        }
    }
    let xb = gpu.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let wb = gpu.device.new_buffer_with_data(
        raw.as_ptr() as *const _,
        raw.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let guard = vec![12345.0f32; n + 4];
    let yb = gpu.device.new_buffer_with_data(
        guard.as_ptr() as *const _,
        (guard.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(0, Some(&xb), 0);
    enc.set_buffer(1, Some(&wb), 0);
    enc.set_buffer(2, Some(&yb), 8);
    let (kk, nn) = (k as u32, n as u32);
    enc.set_bytes(3, 4, &kk as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &nn as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(nn.div_ceil(2) as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let got = unsafe { std::slice::from_raw_parts(yb.contents() as *const f32, n + 4) };
    assert_eq!(&got[..2], &[12345.0, 12345.0]);
    assert_eq!(&got[n + 2..], &[12345.0, 12345.0]);
    for r in 0..n {
        assert!(
            (got[r + 2] - want[r]).abs() < 2e-4,
            "row {r}: {} != {}",
            got[r + 2],
            want[r]
        );
    }
}

/// Closed-form one-step DeltaNet: Q=K=c, V varies by column, zero initial
/// state, beta=1. Clamp normalization gives a unit Q.K above the epsilon
/// floor; adding epsilon to the squared norm produces a different answer.
#[test]
fn deltanet_respects_architecture_l2_normalization() {
    use metal::{MTLResourceOptions, MTLSize};
    let gpu = MetalGpu::new().unwrap();
    let pipe = gpu
        .pipeline(
            ojas_metal::kernels::family_source("ssm").unwrap(),
            "deltanet_fused",
        )
        .unwrap();
    let upload = |v: &[f32]| {
        gpu.device.new_buffer_with_data(
            v.as_ptr() as *const _,
            (v.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let s = 128usize;
    let eps = 1e-6f32;
    for clamp in [0u32, 1] {
        for c in [0.0f32, 1e-9, 1e-5, 0.5] {
            let mut qkv = vec![c; 3 * s];
            for col in 0..s {
                qkv[2 * s + col] = (col + 1) as f32 / s as f32;
            }
            let state = upload(&vec![0.0; s * s]);
            let snap = upload(&vec![-999.0; s * s]);
            let x = upload(&qkv);
            let gate = upload(&[0.0]);
            let beta = upload(&[1.0]);
            let out = upload(&vec![0.0; s]);
            let cb = gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [
                (0, &state),
                (1, &x),
                (2, &gate),
                (3, &beta),
                (4, &out),
                (11, &snap),
            ] {
                enc.set_buffer(i, Some(b), 0);
            }
            for (i, v) in [
                (5, s as u32),
                (6, 1),
                (7, 1),
                (8, 3 * s as u32),
                (9, 1),
                (12, 0),
                (13, 0),
                (14, clamp),
            ] {
                enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void);
            }
            enc.set_bytes(10, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new((s / 4) as u64, 1, 1), MTLSize::new(128, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let norm2 = s as f64 * (c as f64).powi(2);
            let denom = if clamp == 1 {
                norm2.sqrt().max(eps as f64)
            } else {
                (norm2 + eps as f64).sqrt()
            };
            let gain = norm2 / (denom * denom) / (s as f64).sqrt();
            let got = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, s) };
            for col in 0..s {
                let want = qkv[2 * s + col] as f64 * gain;
                assert!(
                    (got[col] as f64 - want).abs() < 1e-6,
                    "clamp={clamp} c={c} col={col}: {} vs {want}",
                    got[col]
                );
            }
            let final_state =
                unsafe { std::slice::from_raw_parts(state.contents() as *const f32, s * s) };
            let snapshot =
                unsafe { std::slice::from_raw_parts(snap.contents() as *const f32, s * s) };
            assert_eq!(final_state, snapshot);
        }
    }
}

#[test]
fn conv_prefix_snapshots_preserve_appended_history_slots() {
    use metal::{MTLResourceOptions, MTLSize};
    let gpu = MetalGpu::new().unwrap();
    let pipe = gpu
        .pipeline(
            ojas_metal::kernels::family_source("ssm").unwrap(),
            "conv1d_prefill",
        )
        .unwrap();
    let upload = |v: &[f32]| {
        gpu.device.new_buffer_with_data(
            v.as_ptr() as *const _,
            (v.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let x = upload(&[1., 2., 3., 4., 5., 6., 7., 8.]);
    let state = upload(&[-3., -2., -1., 0.]);
    let weights = upload(&[0.25; 6]);
    // Four conv floats plus three appended PLE floats per snapshot; guard tail.
    let snap = upload(&[12345.; 36]);
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    for (i, b) in [(0, &x), (1, &state), (2, &weights), (6, &snap)] {
        enc.set_buffer(i, Some(b), 0);
    }
    for (i, v) in [(3, 2u32), (4, 3), (5, 4), (7, 0x8000_0000 | 7)] {
        enc.set_bytes(i, 4, &v as *const u32 as *const _);
    }
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(64, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let got = unsafe { std::slice::from_raw_parts(snap.contents() as *const f32, 36) };
    for row in 0..4 {
        let first = 2. * row as f32 - 1.;
        assert_eq!(
            &got[row * 7..row * 7 + 4],
            &[first, first + 1., first + 2., first + 3.]
        );
        assert_eq!(&got[row * 7 + 4..row * 7 + 7], &[12345.; 3]);
    }
    assert_eq!(&got[28..], &[12345.; 8]);
}

#[test]
fn deltanet_prefix_snapshots_match_independent_prefix_runs() {
    use metal::{MTLResourceOptions, MTLSize};
    let gpu = MetalGpu::new().unwrap();
    let pipe = gpu
        .pipeline(
            ojas_metal::kernels::family_source("ssm").unwrap(),
            "deltanet_fused",
        )
        .unwrap();
    let upload = |v: &[f32]| {
        gpu.device.new_buffer_with_data(
            v.as_ptr() as *const _,
            (v.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let s = 128usize;
    let values: Vec<f32> = (0..4 * 3 * s)
        .map(|i| ((i * 17 % 101) as f32 - 50.) / 100.)
        .collect();
    let x = upload(&values);
    let gate = upload(&[-0.2, -0.3, -0.1, -0.4]);
    let beta = upload(&[0.2, 0.7, 0.4, 0.9]);
    let run = |m: u32, mode: u32| {
        let state = upload(&vec![0.01; s * s]);
        let snap = upload(&vec![12345.; 4 * s * s + 16]);
        let out = upload(&vec![0.; 4 * s]);
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        for (i, b) in [
            (0, &state),
            (1, &x),
            (2, &gate),
            (3, &beta),
            (4, &out),
            (11, &snap),
        ] {
            enc.set_buffer(i, Some(b), 0);
        }
        for (i, v) in [
            (5, s as u32),
            (6, 1),
            (7, 1),
            (8, 3 * s as u32),
            (9, m),
            (12, mode),
            (13, 0),
            (14, 1),
        ] {
            enc.set_bytes(i, 4, &v as *const u32 as *const _);
        }
        enc.set_bytes(10, 4, &1e-6f32 as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new((s / 4) as u64, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let read = |b: &metal::Buffer| unsafe {
            std::slice::from_raw_parts(b.contents() as *const f32, (b.length() / 4) as usize)
                .to_vec()
        };
        (read(&state), read(&snap), read(&out))
    };
    let (final_state, snap, out) = run(4, 0x8000_0000);
    for m in 1..=4 {
        let (state, unused, reference) = run(m, u32::MAX);
        assert_eq!(
            &snap[(m as usize - 1) * s * s..m as usize * s * s],
            state.as_slice()
        );
        assert_eq!(&out[..m as usize * s], &reference[..m as usize * s]);
        assert!(unused.iter().all(|&v| v == 12345.));
    }
    assert_eq!(final_state, &snap[3 * s * s..4 * s * s]);
    assert_eq!(&snap[4 * s * s..], &[12345.; 16]);
}

/// Native block scales, signed bytes, ragged M/N, offsets and accumulating stores.
#[test]
fn cooperative_q80_matches_cpu_with_guards() {
    use metal::{MTLResourceOptions as Opt, MTLSize};
    let gpu = MetalGpu::new().unwrap();
    let src = ojas_metal::kernels::family_source("gemv").unwrap();
    for entry in ["gemv_nat_q80_m_cooperative", "gemv_nat_q80_m_cooperative4"] {
        let pipe = gpu.pipeline(src, entry).unwrap();
        let upload = |p: *const std::ffi::c_void, bytes: usize| {
            gpu.device
                .new_buffer_with_data(p, bytes as u64, Opt::StorageModeShared)
        };
        for k in [32usize, 320, 2560, 6144] {
            let n = 13usize;
            // Offset 16 remains aligned for the f16 scale. Suffix catches overreads
            // that contaminate valid output; output guard values catch overstores.
            let mut w = vec![0u8; 16 + k / 32 * n * 34 + 16];
            let mut state = 987654321u32;
            for b in w[16..16 + k / 32 * n * 34].chunks_exact_mut(34) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let scale = half::f16::from_f32((state % 127 + 1) as f32 / 8192.0);
                b[..2].copy_from_slice(&scale.to_bits().to_le_bytes());
                for v in &mut b[2..] {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    *v = state as u8;
                }
            }
            let wb = upload(w.as_ptr().cast(), w.len());
            for m in [1usize, 2, 3, 4, 7, 8] {
                for accum in [0u32, 1] {
                    let mut x = vec![123.0f32; 4 + k * m + 4];
                    for (i, v) in x[4..4 + k * m].iter_mut().enumerate() {
                        *v = ((i * 137 % 1021) as f32 - 510.0) / 511.0;
                    }
                    let mut y = vec![321.0f32; 4 + n * m + 4];
                    y[4..4 + n * m].fill(0.25);
                    let xb = upload(x.as_ptr().cast(), x.len() * 4);
                    let yb = upload(y.as_ptr().cast(), y.len() * 4);
                    let cb = gpu.command_buffer();
                    let e = cb.new_compute_command_encoder();
                    e.set_compute_pipeline_state(&pipe);
                    e.set_buffer(0, Some(&xb), 16);
                    e.set_buffer(1, Some(&wb), 16);
                    e.set_buffer(2, Some(&yb), 16);
                    for (idx, val) in [(3, k as u32), (4, n as u32), (7, m as u32), (8, accum)] {
                        e.set_bytes(idx, 4, (&val as *const u32).cast());
                    }
                    e.dispatch_thread_groups(
                        MTLSize::new(n.div_ceil(8) as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                    e.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
                    let got =
                        unsafe { std::slice::from_raw_parts(yb.contents().cast::<f32>(), y.len()) };
                    assert_eq!(&got[..4], &[321.0; 4]);
                    assert_eq!(&got[4 + n * m..], &[321.0; 4]);
                    for row in 0..m {
                        for out in 0..n {
                            let mut want = if accum == 1 { 0.25f64 } else { 0.0 };
                            for col in 0..k {
                                let b = 16 + (out * k / 32 + col / 32) * 34;
                                let scale =
                                    half::f16::from_bits(u16::from_le_bytes([w[b], w[b + 1]]))
                                        .to_f32();
                                want += f64::from(scale)
                                    * f64::from(w[b + 2 + col % 32] as i8)
                                    * f64::from(x[4 + row * k + col]);
                            }
                            let actual = got[4 + row * n + out] as f64;
                            assert!(
                                actual.is_finite() && (actual - want).abs() < 0.0001,
                                "K={k} M={m} ACC={accum} row={row} out={out}: {actual} vs {want}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn cached_iq3s_gate_up_matches_cpu_with_guards() {
    use metal::{MTLResourceOptions as Opt, MTLSize};
    let gpu = MetalGpu::new().unwrap();
    let pipe = gpu
        .pipeline(
            ojas_metal::kernels::family_source("moe_iq").unwrap(),
            "moe_gu_iq3s",
        )
        .unwrap();
    let (k, n, experts, selected) = (2560usize, 13usize, 3usize, 2usize);
    let row_bytes = k / 256 * 110;
    let mut seed = 0xa751_38fdu32;
    let mut make_weights = || {
        let mut bytes = vec![0u8; experts * n * row_bytes];
        for block in bytes.chunks_exact_mut(110) {
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
        bytes
    };
    let gate_weights = make_weights();
    let up_weights = make_weights();
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 137 % 1021) as f32 - 510.0) / 511.0)
        .collect();
    let ids = [2u32, 0];
    let upload = |p: *const std::ffi::c_void, bytes: usize| {
        gpu.device
            .new_buffer_with_data(p, bytes as u64, Opt::StorageModeShared)
    };
    let xb = upload(x.as_ptr().cast(), x.len() * 4);
    let gb = upload(gate_weights.as_ptr().cast(), gate_weights.len());
    let ub = upload(up_weights.as_ptr().cast(), up_weights.len());
    let ib = upload(ids.as_ptr().cast(), ids.len() * 4);
    let initial = vec![12345.0f32; selected * n + 16];
    let output = upload(initial.as_ptr().cast(), initial.len() * 4);
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    for (index, buffer) in [(0, &xb), (1, &gb), (2, &ub), (3, &output), (8, &ib)] {
        enc.set_buffer(index, Some(buffer), 0);
    }
    for (index, value) in [(4, k as u32), (5, n as u32)] {
        enc.set_bytes(index, 4, (&value as *const u32).cast());
    }
    enc.dispatch_thread_groups(
        MTLSize::new(n.div_ceil(8) as u64, selected as u64, 1),
        MTLSize::new(64, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
    let got =
        unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), selected * n + 16) };
    assert!(got[selected * n..].iter().all(|&v| v == 12345.0));
    for slot in 0..selected {
        for row in 0..n {
            let offset = (ids[slot] as usize * n + row) * row_bytes;
            let gate = ojas_cpu::cpu_math::dot_iq3s(&gate_weights[offset..offset + row_bytes], &x);
            let up = ojas_cpu::cpu_math::dot_iq3s(&up_weights[offset..offset + row_bytes], &x);
            let want = gate / (1.0 + (-gate).exp()) * up;
            let actual = got[slot * n + row];
            assert!(
                actual.is_finite() && (actual - want).abs() < 2e-3,
                "slot={slot} row={row}: {actual} != {want}"
            );
        }
    }
}

/// A change in one token/stream must not leak into other NextN inputs.
/// Pooling followed by broadcasting fails this isolation property.
#[test]
fn nextn_preserves_stream_identity() {
    use metal::{MTLResourceOptions as Opt, MTLSize};
    let gpu = MetalGpu::new().unwrap();
    let pipe = gpu
        .pipeline(
            ojas_metal::kernels::family_source("qwen4exp").unwrap(),
            "nextn_concat_streams",
        )
        .unwrap();
    let (d, hc, m) = (65usize, 4usize, 3usize);
    let emb: Vec<f32> = (0..m * d).map(|i| (i % 13) as f32).collect();
    let hidden: Vec<f32> = (0..m * hc * d).map(|i| i as f32 / 1024.0).collect();
    let upload = |v: &[f32]| {
        gpu.device.new_buffer_with_data(
            v.as_ptr().cast(),
            (v.len() * 4) as u64,
            Opt::StorageModeShared,
        )
    };
    let eb = upload(&emb);
    let hb = upload(&hidden);
    let init = vec![-999.0f32; m * hc * 2 * d + 8];
    let out = upload(&init);
    let run = || {
        let cb = gpu.command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(&pipe);
        e.set_buffer(0, Some(&eb), 0);
        e.set_buffer(1, Some(&hb), 0);
        e.set_buffer(2, Some(&out), 16);
        for (i, v) in [(3, d as u32), (4, hc as u32)] {
            e.set_bytes(i, 4, (&v as *const u32).cast());
        }
        e.dispatch_thread_groups(
            MTLSize::new(d.div_ceil(64) as u64, hc as u64, m as u64),
            MTLSize::new(64, 1, 1),
        );
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
        unsafe { std::slice::from_raw_parts(out.contents().cast::<f32>(), init.len()) }.to_vec()
    };
    let before = run();
    assert_eq!(&before[..4], &[-999.0; 4]);
    assert_eq!(&before[before.len() - 4..], &[-999.0; 4]);
    for t in 0..m {
        for c in 0..hc {
            let row = 4 + (t * hc + c) * 2 * d;
            assert_eq!(&before[row..row + d], &emb[t * d..(t + 1) * d]);
            assert_eq!(
                &before[row + d..row + 2 * d],
                &hidden[(t * hc + c) * d..(t * hc + c + 1) * d]
            );
        }
    }
    let changed = (hc + 2) * d + 17;
    unsafe {
        *hb.contents().cast::<f32>().add(changed) += 7.0;
    }
    let after = run();
    let differences: Vec<_> = before
        .iter()
        .zip(&after)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(differences, vec![4 + (hc + 2) * 2 * d + d + 17]);
}
