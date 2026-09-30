//! The encoder kernels (`kernels/vision.rs`, plus `act_m` from `kernels/ops.rs`),
//! gated against the CPU oracle.
//!
//! `ojas_cpu::cpu_math::layernorm`, `ojas_cpu::cpu_math::gelu` and
//! `ojas_cpu::cpu_vit::patchify` are the same functions the CPU ViT tower runs, so a
//! disagreement here is a real GPU/CPU divergence rather than two independent
//! transcriptions of a paper (the convention `ojas-models/examples/moe_iq_gate.rs:58`
//! set).
//!
//! The three bugs this catches:
//!
//! 1. `eps` outside the sqrt in `vit_layernorm_m`. `1/(sqrt(var)+eps)` instead of
//!    `1/sqrt(var+eps)` agrees to ~1e-5 on any row with ordinary spread and is wrong
//!    by 3.3x on a near-constant one, which is what a ViT's `post_ln` sees.
//!    `NEAR_CONSTANT` is that row, and `layernorm_eps_is_inside_the_sqrt` asserts the
//!    row is discriminating before asserting the kernel is right, so the gate cannot
//!    degrade into a tautology.
//! 2. `Σx²/d - mean²` instead of two passes. On the same near-constant row the
//!    one-pass identity cancels catastrophically and can produce a negative variance.
//!    Covered by that row at a large offset (`NEAR_CONSTANT_BIG`).
//! 3. A transposed patch row in `vit_patchify`. `(ic, ky, kx)` with `kx` fastest is
//!    ggml's im2col order; `(ky, kx, ic)` is the natural order for interleaved RGB.
//!    Both produce a correctly-shaped `[T, C*P*P]` matrix and a tower that is merely
//!    bad. The image here encodes `(c, y, x)` in the value itself (`c*1e6 + y*1e3 + x`,
//!    exact in f32), so a wrong nesting is caught positionally — element by element,
//!    naming the coordinate it read — rather than by a statistic.
//!
//! Two idioms carried from `tests/dispatch.rs`: guard regions on both sides
//! of every output buffer (`dispatch.rs:216`/`:235-238`/`:261`), so an over-store is
//! caught even when the numbers agree, and ragged shapes
//! (`dispatch.rs:93`/`:213`/`:232`) — row and element counts that are not multiples of
//! the threadgroup width, including 1.

// The Metal device is macOS-only (`ojas-metal/src/lib.rs`); only its `kernels` source
// table builds elsewhere. These tests drive a real `MetalGpu`.
#![cfg(target_os = "macos")]

use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
use ojas_core::Device;
use ojas_metal::MetalGpu;
use std::ffi::c_void;

/// f32 guard values on each side of every output buffer (256 B, comfortably past
/// Metal's 4 B setBuffer alignment).
const GUARD: usize = 64;
const GUARD_FILL: f32 = 12345.0;

fn upload(gpu: &MetalGpu, v: &[f32]) -> Buffer {
    gpu.device.new_buffer_with_data(v.as_ptr() as *const c_void, (v.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared)
}

/// An output buffer of `n` floats flanked by `GUARD` guard values on each side.
/// Bind it at offset `GUARD*4`; check it with [`assert_guards`].
fn guarded(gpu: &MetalGpu, n: usize) -> Buffer {
    upload(gpu, &vec![GUARD_FILL; n + 2 * GUARD])
}

fn host(buf: &Buffer, n: usize) -> &[f32] {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }
}

fn payload(buf: &Buffer, n: usize) -> &[f32] {
    &host(buf, n + 2 * GUARD)[GUARD..GUARD + n]
}

fn assert_guards(label: &str, buf: &Buffer, n: usize) {
    let all = host(buf, n + 2 * GUARD);
    assert!(all[..GUARD].iter().all(|&v| v == GUARD_FILL), "{label}: stored BEFORE the output region");
    assert!(all[GUARD + n..].iter().all(|&v| v == GUARD_FILL), "{label}: stored PAST the output region");
}

fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

fn pipe(gpu: &MetalGpu, entry: &str) -> ComputePipelineState {
    let src = ojas_metal::kernels::source_of(entry).unwrap_or_else(|| panic!("{entry} is not a registered kernel"));
    gpu.pipeline(src, entry).unwrap_or_else(|e| panic!("{entry} pipeline: {e}"))
}

fn gpu_or_skip(label: &str) -> Option<MetalGpu> {
    match MetalGpu::new() {
        Ok(g) => Some(g),
        Err(e) => { eprintln!("{label}: no Metal device ({e}); skipping"); None }
    }
}

// ---------------------------------------------------------------------------
// vit_layernorm_m
// ---------------------------------------------------------------------------

/// The ViT eps. The mmproj ships 1e-6; 1e-5 is the whisper/transformers default. Both
/// are tested, since the eps-placement bug's visibility scales with it.
const EPS: [f32; 2] = [1e-6, 1e-5];

/// (rows, d, threads-per-threadgroup), deliberately ragged: d=768 is the real ViT
/// width and a multiple of every ts here; 13/100/255/257/1000 are not (257 also leaves
/// a 1-element tail past ts=256, and 13 < ts so most of `part[]` holds the
/// zero-initialised identity of the sum). ts sweeps 32/64/256 so each d is exercised
/// both as a multiple and as a non-multiple of the threadgroup width. m=1/2/3/7
/// mirrors `dispatch.rs:232`.
const LN_SHAPES: &[(usize, usize, u64)] = &[
    (1, 768, 256), (7, 768, 256), (3, 768, 64), (2, 768, 32),
    (1, 13, 32), (3, 13, 256), (7, 100, 64), (2, 255, 256),
    (3, 257, 256), (1, 1000, 64), (2, 1000, 256),
];

/// One `vit_layernorm_m` dispatch. `out` is bound past its leading guard.
fn ln_run(gpu: &MetalGpu, p: &ComputePipelineState, x: &Buffer, w: &Buffer, b: &Buffer,
          out: &Buffer, d: u32, eps: f32, m: u64, ts: u64, has_bias: bool) {
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(p);
    enc.set_buffer(0, Some(x), 0);
    enc.set_buffer(1, Some(w), 0);
    enc.set_buffer(2, Some(out), (GUARD * 4) as u64);
    enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
    enc.set_buffer(5, Some(b), 0);
    let hb = has_bias as u32;
    enc.set_bytes(6, 4, &hb as *const u32 as *const c_void);
    enc.dispatch_thread_groups(MTLSize::new(m, 1, 1), MTLSize::new(ts, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

#[test]
fn vit_layernorm_m_matches_cpu_oracle() {
    let Some(gpu) = gpu_or_skip("vision/layernorm") else { return };
    let p = pipe(&gpu, "vit_layernorm_m");
    let mut worst_rel = 0f64;
    let mut worst_at = String::new();
    for &(m, d, ts) in LN_SHAPES {
        for &eps in &EPS {
            let mut seed = 0x5EEDu32 ^ ((d as u32) << 7) ^ (m as u32) ^ (ts as u32);
            // Scale varies per row so the mean is genuinely non-zero — an LN that
            // forgot to subtract it would still pass on zero-mean rows.
            let x: Vec<f32> = (0..m * d).map(|i| lcg(&mut seed) * 3.0 + (i / d) as f32 * 2.0 - 1.0).collect();
            let w: Vec<f32> = (0..d).map(|_| lcg(&mut seed) + 1.0).collect();
            let bias: Vec<f32> = (0..d).map(|_| lcg(&mut seed) * 0.5).collect();
            let (xb, wb, bb) = (upload(&gpu, &x), upload(&gpu, &w), upload(&gpu, &bias));
            let ob = guarded(&gpu, m * d);
            ln_run(&gpu, &p, &xb, &wb, &bb, &ob, d as u32, eps, m as u64, ts, true);
            let got = payload(&ob, m * d);
            assert_guards(&format!("layernorm m={m} d={d} ts={ts}"), &ob, m * d);
            for r in 0..m {
                let want = ojas_cpu::cpu_math::layernorm(&x[r * d..(r + 1) * d], &w, &bias, eps);
                // Normalized by the ROW's magnitude, not the element's: the two
                // implementations differ only in summation order, so the error is
                // a property of the row's reduction and dividing by an element
                // that happens to sit near a zero crossing measures nothing.
                let scale = want.iter().fold(0f64, |a, &v| a.max(v.abs() as f64)).max(1e-6);
                for i in 0..d {
                    let (g, e) = (got[r * d + i] as f64, want[i] as f64);
                    assert!(g.is_finite(), "layernorm m={m} d={d} ts={ts} row {r} elem {i}: {g}");
                    let rel = (g - e).abs() / scale;
                    if rel > worst_rel { worst_rel = rel; worst_at = format!("m={m} d={d} ts={ts} eps={eps:e} row {r} elem {i} ({g} vs {e}, row scale {scale:.4})"); }
                }
            }
        }
    }
    println!("vit_layernorm_m: worst rel err {worst_rel:.3e} at {worst_at}");
    assert!(worst_rel < 1e-5, "worst rel err {worst_rel:.3e} at {worst_at}");
}

/// Alternating ±1e-3 about a mean of 1.0: variance 1e-6, an order of magnitude
/// BELOW eps=1e-5. This is where `1/sqrt(var+eps)` and `1/(sqrt(var)+eps)` split
/// (301.5 vs 990.1), and it is the regime a ViT's `post_ln` actually runs in.
const NEAR_CONSTANT_SPREAD: f32 = 1e-3;
/// The same row offset far from zero. `Σx²/d - mean²` loses ~13 of f32's 24 bits
/// here (mean² ≈ 1e6, variance 1e-6) and lands anywhere from 0 to negative; the
/// two-pass form does not. Nothing about the *placement* of eps changes, so this
/// row isolates the one-pass/two-pass choice specifically.
const NEAR_CONSTANT_BIG: f32 = 1000.0;

#[test]
fn vit_layernorm_m_eps_is_inside_the_sqrt() {
    let Some(gpu) = gpu_or_skip("vision/layernorm-eps") else { return };
    let p = pipe(&gpu, "vit_layernorm_m");
    let eps = 1e-5f32;
    for &offset in &[0.0f32, NEAR_CONSTANT_BIG] {
        // d=257: odd, prime, and one past the 256-thread threadgroup width, so
        // exactly one element lands in the strided tail of both passes.
        let d = 257usize;
        let x: Vec<f32> = (0..d).map(|i| offset + 1.0 + if i % 2 == 0 { NEAR_CONSTANT_SPREAD } else { -NEAR_CONSTANT_SPREAD }).collect();
        let w = vec![1.0f32; d];
        let bias = vec![0.0f32; d];

        // f64 reference for both eps placements, so the assertion below is proved
        // to be discriminating rather than assumed to be.
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
        let inv_inside = 1.0 / (var + eps as f64).sqrt();
        let inv_outside = 1.0 / (var.sqrt() + eps as f64);
        let ratio = inv_outside / inv_inside;
        println!("layernorm eps gate (offset {offset}): var={var:.3e} inside={inv_inside:.4} \
                  outside={inv_outside:.4} (ratio {ratio:.3}x)");
        assert!(ratio > 2.0, "the near-constant row stopped discriminating between eps placements \
            (ratio {ratio:.3}x) — this gate would now pass either way");

        let (xb, wb, bb) = (upload(&gpu, &x), upload(&gpu, &w), upload(&gpu, &bias));
        let ob = guarded(&gpu, d);
        ln_run(&gpu, &p, &xb, &wb, &bb, &ob, d as u32, eps, 1, 256, true);
        let got = payload(&ob, d);
        assert_guards("layernorm eps gate", &ob, d);

        // What the Sum(x^2)/d - mean^2 identity would give, evaluated in f32 the way
        // a one-pass kernel would. At offset 1000 both terms are ~1.002e6 and f32's
        // ulp there is 0.0625, six orders above the real variance of 1e-6, so the
        // identity returns 0 (inv 316.2 instead of 302.1) or a negative variance
        // (inv NaN). Computed here rather than stated in prose.
        let (mut s1, mut s2) = (0f32, 0f32);
        for &v in &x { s1 += v; s2 += v * v; }
        let var_onepass = s2 / d as f32 - (s1 / d as f32) * (s1 / d as f32);
        let inv_onepass = if var_onepass + eps > 0.0 { (1.0 / (var_onepass + eps).sqrt()) as f64 } else { f64::NAN };
        println!("layernorm one-pass gate (offset {offset}): Sum(x^2)/d - mean^2 = {var_onepass:e} \
                  (true {var:.3e}) -> inv {inv_onepass:.4} vs {inv_inside:.4}");

        // Both f32 implementations are graded against the same f64 reference. At
        // offset 1000 the row is 1001 +/- 1e-3 and the f32 accumulation of a sum
        // ~2.6e5 perturbs the mean by a few 1e-5 — a few parts in 1e3 of the
        // deviation, for the CPU oracle as much as for the GPU. Grading the GPU
        // against the CPU alone would report that inherent loss as a GPU defect.
        let want = ojas_cpu::cpu_math::layernorm(&x, &w, &bias, eps);
        let scale = inv_inside * NEAR_CONSTANT_SPREAD as f64;
        let (mut worst_gpu, mut worst_cpu) = (0f64, 0f64);
        for i in 0..d {
            let dev = (x[i] as f64 - mean) * inv_inside;
            let wrong = (x[i] as f64 - mean) * inv_outside;
            let g = got[i] as f64;
            assert!(g.is_finite(), "offset {offset} elem {i}: GPU produced {g} — a negative variance \
                reaching rsqrt is what the one-pass identity does");
            assert!((g - wrong).abs() > (g - dev).abs(), "offset {offset} elem {i}: GPU {g} is closer to \
                the eps-OUTSIDE-sqrt answer {wrong} than to the eps-INSIDE answer {dev}");
            if inv_onepass.is_finite() {
                let onepass = (x[i] as f64 - mean) * inv_onepass;
                assert!((g - onepass).abs() >= (g - dev).abs(), "offset {offset} elem {i}: GPU {g} is \
                    closer to the one-pass Sum(x^2)-mean^2 answer {onepass} than to the two-pass {dev}");
            }
            worst_gpu = worst_gpu.max((g - dev).abs() / scale);
            worst_cpu = worst_cpu.max((want[i] as f64 - dev).abs() / scale);
        }
        println!("layernorm eps gate (offset {offset}): worst rel err vs f64 truth — gpu {worst_gpu:.3e}, cpu oracle {worst_cpu:.3e}");
        // 1e-2 is loose in absolute terms because the floor here is f32 accumulation
        // of the mean, not the kernel. It is still 3.3x under an eps-outside kernel's
        // 2.3x miss and 4.7x under the 4.7% a one-pass identity lands at; the two
        // elementwise assertions above are the sharp part of this gate.
        assert!(worst_gpu < 1e-2, "offset {offset}: gpu worst rel err {worst_gpu:.3e} vs f64 truth");
        assert!(worst_gpu <= worst_cpu.max(1e-6) * 8.0, "offset {offset}: the GPU two-pass reduction is \
            materially worse than the CPU oracle's ({worst_gpu:.3e} vs {worst_cpu:.3e}) — suspect the \
            Sum(x^2)/d - mean^2 identity rather than two passes");
    }
}

/// The final store loop only rewrites elements the storing thread has itself
/// already read twice, and both reductions are complete before it runs — so
/// `out` may alias `x`. The header claims that; this proves it rather than
/// leaving the next caller to find out.
#[test]
fn vit_layernorm_m_is_safe_in_place() {
    let Some(gpu) = gpu_or_skip("vision/layernorm-inplace") else { return };
    let p = pipe(&gpu, "vit_layernorm_m");
    let (m, d, eps) = (3usize, 768usize, 1e-6f32);
    let mut seed = 0xA11A5u32;
    let x: Vec<f32> = (0..m * d).map(|i| lcg(&mut seed) * 3.0 + (i / d) as f32 * 2.0 - 1.0).collect();
    let w: Vec<f32> = (0..d).map(|_| lcg(&mut seed) + 1.0).collect();
    let bias: Vec<f32> = (0..d).map(|_| lcg(&mut seed) * 0.5).collect();
    let (wb, bb) = (upload(&gpu, &w), upload(&gpu, &bias));

    // Reference run, out-of-place.
    let xb = upload(&gpu, &x);
    let ob = guarded(&gpu, m * d);
    ln_run(&gpu, &p, &xb, &wb, &bb, &ob, d as u32, eps, m as u64, 256, true);
    let want: Vec<f32> = payload(&ob, m * d).to_vec();

    // One buffer bound to slots 0 and 2. Guards must still hold.
    let inplace = guarded(&gpu, m * d);
    unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), (inplace.contents() as *mut f32).add(GUARD), m * d) };
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&p);
    enc.set_buffer(0, Some(&inplace), (GUARD * 4) as u64);
    enc.set_buffer(1, Some(&wb), 0);
    enc.set_buffer(2, Some(&inplace), (GUARD * 4) as u64);
    let dd = d as u32;
    enc.set_bytes(3, 4, &dd as *const u32 as *const c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
    enc.set_buffer(5, Some(&bb), 0);
    let hb = 1u32;
    enc.set_bytes(6, 4, &hb as *const u32 as *const c_void);
    enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let got = payload(&inplace, m * d);
    assert_guards("layernorm in-place", &inplace, m * d);
    assert_eq!(got, want.as_slice(), "in-place layernorm diverged from the out-of-place run");
}

/// `has_bias == 0` is the ModernBERT form: `y = (x - mean)/sqrt(var + eps) * w`. Slot 5
/// is bound to a buffer of NaNs, so a kernel that still read the bias would poison
/// every output rather than shift it by a plausible amount.
#[test]
fn vit_layernorm_m_without_bias_never_reads_slot_5() {
    let Some(gpu) = gpu_or_skip("vision/layernorm-nobias") else { return };
    let p = pipe(&gpu, "vit_layernorm_m");
    for &(m, d, ts) in &[(1usize, 1024usize, 256u64), (5, 768, 256), (3, 257, 64)] {
        let mut seed = 0xB1A5u32 ^ d as u32;
        let x: Vec<f32> = (0..m * d).map(|i| lcg(&mut seed) * 3.0 + (i / d) as f32 - 1.0).collect();
        let w: Vec<f32> = (0..d).map(|_| lcg(&mut seed) + 1.0).collect();
        let (xb, wb, nanb) = (upload(&gpu, &x), upload(&gpu, &w), upload(&gpu, &vec![f32::NAN; d]));
        let ob = guarded(&gpu, m * d);
        ln_run(&gpu, &p, &xb, &wb, &nanb, &ob, d as u32, 1e-5, m as u64, ts, false);
        assert_guards(&format!("layernorm no-bias m={m} d={d}"), &ob, m * d);
        let got = payload(&ob, m * d);
        let zero = vec![0f32; d];
        for r in 0..m {
            let want = ojas_cpu::cpu_math::layernorm(&x[r * d..(r + 1) * d], &w, &zero, 1e-5);
            let scale = want.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-6);
            for i in 0..d {
                let g = got[r * d + i];
                assert!(g.is_finite(), "no-bias m={m} d={d} row {r} elem {i}: {g} (bias slot was read)");
                assert!((g - want[i]).abs() / scale < 1e-5, "no-bias m={m} d={d} row {r} elem {i}: {g} vs {}", want[i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// act_m: the elementwise activation (the ViT MLP runs act 1, tanh GELU)
// ---------------------------------------------------------------------------

/// Ragged element counts: 1 and 7 are sub-threadgroup, 13 is `dispatch.rs:93`'s
/// awkward K, 255/257 straddle the 256-thread group, 4096 is a real ViT row
/// (768*... rounded) and 16385 leaves a 1-thread tail.
const GELU_N: &[usize] = &[1, 7, 13, 255, 256, 257, 4096, 16385];

fn act_run(gpu: &MetalGpu, p: &ComputePipelineState, x: &Buffer, out: &Buffer, out_off: u64, n: u32, act: u32) {
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(p);
    enc.set_buffer(0, Some(x), 0);
    enc.set_buffer(1, Some(out), out_off);
    enc.set_bytes(2, 4, &n as *const u32 as *const c_void);
    enc.set_bytes(3, 4, &act as *const u32 as *const c_void);
    enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

/// Inputs spanning the curve plus the saturation tails the PRELUDE's clamp exists for:
/// past |x| ~ 40 the cube overflows f32 and fast-math `tanh(inf)` is NaN, so the kernel
/// clamps the tanh argument. The CPU oracle does not clamp, and both must still land on
/// `gelu(x) -> x` / `-> 0`.
fn gelu_input(i: usize, n: usize) -> f32 {
    match i % 8 {
        0 => 0.0,
        1 => 40.0,
        2 => -40.0,
        3 => 1e4,
        4 => -1e4,
        5 => 1.702,
        6 => -1.702,
        _ => (i as f32 / n as f32) * 16.0 - 8.0,
    }
}

/// Every activation code against its CPU oracle: 1 is the ViT's tanh GELU, 3 the exact
/// erf GELU (ModernBERT, PyTorch `nn.GELU()`), 4 ReLU. The erf oracle is f64
/// (`cpu_math::erf`), independent of the kernel's Abramowitz-Stegun approximation.
#[test]
fn act_m_matches_cpu_oracle() {
    let Some(gpu) = gpu_or_skip("encoder/act") else { return };
    let p = pipe(&gpu, "act_m");
    let oracles: [(u32, &str, fn(f32) -> f32); 3] = [
        (1, "gelu-tanh", ojas_cpu::cpu_math::gelu),
        (3, "gelu-erf", ojas_cpu::cpu_math::gelu_erf),
        (4, "relu", |x| x.max(0.0)),
    ];
    for (act, name, oracle) in oracles {
        let mut worst = 0f64;
        let mut worst_at = String::new();
        for &n in GELU_N {
            let x: Vec<f32> = (0..n).map(|i| gelu_input(i, n)).collect();
            let xb = upload(&gpu, &x);
            let ob = guarded(&gpu, n);
            act_run(&gpu, &p, &xb, &ob, (GUARD * 4) as u64, n as u32, act);
            let got = payload(&ob, n);
            assert_guards(&format!("{name} n={n}"), &ob, n);
            for i in 0..n {
                let e = oracle(x[i]) as f64;
                let g = got[i] as f64;
                assert!(g.is_finite(), "{name} n={n} i={i} x={}: got {g}", x[i]);
                // Relative to the magnitude of the input: GELU(x) -> x in the tail,
                // so an absolute tolerance would be meaningless at x=1e4.
                let tol_base = e.abs().max(x[i].abs() as f64).max(1.0);
                let rel = (g - e).abs() / tol_base;
                if rel > worst { worst = rel; worst_at = format!("n={n} i={i} x={} ({g} vs {e})", x[i]); }
            }
        }
        println!("act_m {name}: worst rel err {worst:.3e} at {worst_at}");
        assert!(worst < 1e-6, "{name}: worst rel err {worst:.3e} at {worst_at}");
    }
}

/// The kernel is one thread per element, so `out` may alias `x`. The ViT MLP
/// wants that (fc1 writes a scratch row, GELU rewrites it in place); asserting it
/// here makes it a contract instead of an accident.
#[test]
fn act_m_is_safe_in_place() {
    let Some(gpu) = gpu_or_skip("encoder/act-inplace") else { return };
    let p = pipe(&gpu, "act_m");
    let n = 3072usize;
    let x: Vec<f32> = (0..n).map(|i| gelu_input(i, n)).collect();
    // One buffer, bound to both slots: guards on either side still have to hold.
    let xb = guarded(&gpu, n);
    unsafe {
        let p0 = (xb.contents() as *mut f32).add(GUARD);
        std::ptr::copy_nonoverlapping(x.as_ptr(), p0, n);
    }
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&p);
    enc.set_buffer(0, Some(&xb), (GUARD * 4) as u64);
    enc.set_buffer(1, Some(&xb), (GUARD * 4) as u64);
    let (nn, act) = (n as u32, 1u32);
    enc.set_bytes(2, 4, &nn as *const u32 as *const c_void);
    enc.set_bytes(3, 4, &act as *const u32 as *const c_void);
    enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let got = payload(&xb, n);
    assert_guards("gelu in-place", &xb, n);
    for i in 0..n {
        let e = ojas_cpu::cpu_math::gelu(x[i]) as f64;
        let tol = e.abs().max(x[i].abs() as f64).max(1.0) * 1e-6;
        assert!((got[i] as f64 - e).abs() <= tol, "in-place i={i} x={}: {} vs {e}", x[i], got[i]);
    }
}

// ---------------------------------------------------------------------------
// vit_patchify
// ---------------------------------------------------------------------------

/// `(c, y, x) -> c*1e6 + y*1e3 + x`. Every coordinate of every shape below keeps
/// this under 2^24, so the value is an EXACT f32 and uniquely names its source
/// pixel: a wrong index order is caught by reading the coordinate back, not by a
/// statistic that a permutation could satisfy by accident.
fn pixel(c: usize, y: usize, x: usize) -> f32 {
    (c * 1_000_000 + y * 1_000 + x) as f32
}

fn synth_image(w: usize, h: usize, c: usize) -> Vec<f32> {
    let mut img = vec![0f32; c * h * w];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                img[ch * h * w + y * w + x] = pixel(ch, y, x);
            }
        }
    }
    img
}

/// (W, H, C, P), ragged: 50x34 is not a multiple of 16 (the trailing strip is dropped,
/// matching `cpu_vit::patchify`'s `width/patch`), 24x40 with P=8 gives a 3x5 patch grid
/// whose element count is not a multiple of 256, C=1 exercises the degenerate channel
/// loop, and 64x64/P=16 is the real 16-patch ViT tile with K = 3*16*16 = 768.
const IMG_SHAPES: &[(usize, usize, usize, usize)] = &[
    (64, 64, 3, 16), (48, 32, 3, 16), (50, 34, 3, 16),
    (24, 40, 3, 8), (16, 16, 3, 16), (32, 16, 1, 16), (16, 16, 3, 4),
];

fn patchify_run(gpu: &MetalGpu, p: &ComputePipelineState, img: &Buffer, out: &Buffer,
                w: u32, h: u32, c: u32, patch: u32, total: u32) {
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(p);
    enc.set_buffer(0, Some(img), 0);
    enc.set_buffer(1, Some(out), (GUARD * 4) as u64);
    for (slot, v) in [(2u64, w), (3, h), (4, c), (5, patch), (6, total)] {
        enc.set_bytes(slot, 4, &v as *const u32 as *const c_void);
    }
    enc.dispatch_thread_groups(MTLSize::new((total as u64).div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

#[test]
fn vit_patchify_matches_cpu_oracle() {
    let Some(gpu) = gpu_or_skip("vision/patchify") else { return };
    let p = pipe(&gpu, "vit_patchify");
    for &(w, h, c, patch) in IMG_SHAPES {
        let img = synth_image(w, h, c);
        let want = ojas_cpu::cpu_vit::patchify(&img, w, h, c, patch);
        let (pw, ph) = (w / patch, h / patch);
        let total = pw * ph * c * patch * patch;
        assert_eq!(want.len(), total);

        let ib = upload(&gpu, &img);
        let ob = guarded(&gpu, total);
        patchify_run(&gpu, &p, &ib, &ob, w as u32, h as u32, c as u32, patch as u32, total as u32);
        let got = payload(&ob, total);
        let label = format!("patchify {w}x{h}x{c} P={patch}");
        assert_guards(&label, &ob, total);
        // Exact: this kernel moves floats, it does not compute with them. Any
        // difference at all is an index bug.
        for i in 0..total {
            assert_eq!(got[i], want[i], "{label}: element {i} (patch {}, slot {}) — GPU read \
                pixel (c={}, y={}, x={}), oracle read (c={}, y={}, x={})",
                i / (c * patch * patch), i % (c * patch * patch),
                (got[i] as i64) / 1_000_000, ((got[i] as i64) / 1_000) % 1_000, (got[i] as i64) % 1_000,
                (want[i] as i64) / 1_000_000, ((want[i] as i64) / 1_000) % 1_000, (want[i] as i64) % 1_000);
        }
        println!("{label}: {} patches x {} = {total} elements exact vs cpu_vit::patchify",
                 pw * ph, c * patch * patch);
    }
}

/// The oracle and the kernel could in principle share a wrong convention. This
/// asserts the row order directly against the reference's index expression —
/// `dst_data[iic*(KH*KW) + ikh*KW + ikw]` (`ggml-cpu/ops.cpp:6417`) — with no
/// CPU implementation in the loop.
#[test]
fn vit_patchify_row_order_is_channel_major_kx_fastest() {
    let Some(gpu) = gpu_or_skip("vision/patchify-order") else { return };
    let p = pipe(&gpu, "vit_patchify");
    let (w, h, c, patch) = (64usize, 48usize, 3usize, 16usize);
    let (pw, ph) = (w / patch, h / patch);
    let row = c * patch * patch;
    let total = pw * ph * row;
    assert_eq!(row, 768, "the ViT patch row is 3*16*16");

    let ib = upload(&gpu, &synth_image(w, h, c));
    let ob = guarded(&gpu, total);
    patchify_run(&gpu, &p, &ib, &ob, w as u32, h as u32, c as u32, patch as u32, total as u32);
    let got = payload(&ob, total);
    assert_guards("patchify order", &ob, total);

    for py in 0..ph {
        for px in 0..pw {
            let t = py * pw + px; // patch (token) order: row-major y*pw + x
            for ic in 0..c {
                for ky in 0..patch {
                    for kx in 0..patch {
                        let e = ic * patch * patch + ky * patch + kx; // kx FASTEST
                        let want = pixel(ic, py * patch + ky, px * patch + kx);
                        assert_eq!(got[t * row + e], want,
                            "patch ({py},{px}) slot (ic={ic}, ky={ky}, kx={kx}) at row offset {e}: \
                             got pixel (c={}, y={}, x={}), want (c={ic}, y={}, x={})",
                            (got[t * row + e] as i64) / 1_000_000,
                            ((got[t * row + e] as i64) / 1_000) % 1_000,
                            (got[t * row + e] as i64) % 1_000,
                            py * patch + ky, px * patch + kx);
                    }
                }
            }
        }
    }
    println!("vit_patchify: ({pw}x{ph}) patches, row order (ic, ky, kx) with kx fastest, \
              K={row} — verified positionally against ggml-cpu/ops.cpp:6417");
}

/// `vit_gelu` exists for the CUDA twin's name; it must stay `act_m` with act 1 exactly,
/// or the two Metal entry points would quietly diverge.
#[test]
fn vit_gelu_is_act_m_with_tanh_gelu() {
    let Some(gpu) = gpu_or_skip("vision/vit-gelu") else { return };
    let (vg, am) = (pipe(&gpu, "vit_gelu"), pipe(&gpu, "act_m"));
    let n = 4099usize;
    let x: Vec<f32> = (0..n).map(|i| gelu_input(i, n)).collect();
    let xb = upload(&gpu, &x);
    let (a, b) = (guarded(&gpu, n), guarded(&gpu, n));
    act_run(&gpu, &am, &xb, &a, (GUARD * 4) as u64, n as u32, 1);
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&vg);
    enc.set_buffer(0, Some(&xb), 0);
    enc.set_buffer(1, Some(&b), (GUARD * 4) as u64);
    let nn = n as u32;
    enc.set_bytes(2, 4, &nn as *const u32 as *const c_void);
    enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    assert_guards("vit_gelu", &b, n);
    let (pa, pb) = (payload(&a, n), payload(&b, n));
    assert!(pa.iter().zip(pb).all(|(u, v)| u.to_bits() == v.to_bits()), "vit_gelu diverged from act_m(act=1)");
}

// ---------------------------------------------------------------------------
// vit_qkv_prep and ffn_gu_rows (the encoder block's fused steps)
// ---------------------------------------------------------------------------

/// One dispatch of `pipe` with buffers, u32 and f32 constants, `groups` x `threads`.
fn dispatch(gpu: &MetalGpu, pipe: &ComputePipelineState, bufs: &[(u64, &Buffer)], ints: &[(u64, u32)],
            floats: &[(u64, f32)], groups: u64, threads: u64) {
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    for &(i, b) in bufs { enc.set_buffer(i, Some(b), 0); }
    for &(i, v) in ints { enc.set_bytes(i, 4, &v as *const u32 as *const c_void); }
    for &(i, v) in floats { enc.set_bytes(i, 4, &v as *const f32 as *const c_void); }
    enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(threads, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

fn upload_u32(gpu: &MetalGpu, v: &[u32]) -> Buffer {
    gpu.device.new_buffer_with_data(v.as_ptr() as *const c_void, (v.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared)
}

fn words(buf: &Buffer, n: usize) -> &[u32] {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const u32, n) }
}

fn halves(buf: &Buffer, n: usize) -> &[u16] {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const u16, n) }
}

/// `vit_qkv_prep` claims to be `vit_qkv_split`, `vit_rope` on Q and K, and
/// `copy_f32_half` on K and V, fused. Held to that bit for bit in the three ways the
/// encoders call it: the text rope (no sections, ramp over `hd`), the vision M-RoPE
/// (four sections, ramp over `hd/2`) and no rotation at all (the Laya head).
#[test]
fn vit_qkv_prep_is_split_rope_and_half_fused() {
    let Some(gpu) = gpu_or_skip("encoder/qkv-prep") else { return };
    let (prep, split, rope, half) = (pipe(&gpu, "vit_qkv_prep"), pipe(&gpu, "vit_qkv_split"),
                                     pipe(&gpu, "vit_rope"), pipe(&gpu, "copy_f32_half"));
    let hd = 64u32;
    for &(m, d) in &[(1usize, 128u32), (37, 768), (130, 1024)] {
        for (mode, sections, freq_dims, base) in [("text", [0u32; 4], hd, 160000.0f32),
                                                  ("vision", [hd / 4; 4], hd / 2, 10000.0),
                                                  ("none", [0u32; 4], 0, 0.0)] {
            let mut seed = 0x9E37u32 ^ (m as u32) ^ d;
            let qkv: Vec<f32> = (0..m * 3 * d as usize).map(|_| lcg(&mut seed) * 4.0).collect();
            // Positions restart partway, as packed sequences do; the vision streams
            // differ per channel, as a patch's (y, x) does.
            let mut mpos = sections.to_vec();
            for r in 0..m as u32 { mpos.extend_from_slice(&[r % 23, r / 5, r % 23, r / 5]); }
            let (qkvb, posb) = (upload(&gpu, &qkv), upload_u32(&gpu, &mpos));
            let dm = m as u32 * d;
            let (q, kh, vh) = (upload(&gpu, &vec![0.0; dm as usize]), upload(&gpu, &vec![0.0; dm as usize / 2 + 1]),
                               upload(&gpu, &vec![0.0; dm as usize / 2 + 1]));
            let threads = dm / 2;
            dispatch(&gpu, &prep, &[(0, &qkvb), (1, &q), (2, &kh), (3, &vh), (5, &posb)],
                &[(4, d), (6, hd), (7, m as u32), (8, freq_dims)], &[(9, base)], threads.div_ceil(64) as u64, 64);

            let (q2, k2, v2) = (upload(&gpu, &vec![0.0; dm as usize]), upload(&gpu, &vec![0.0; dm as usize]),
                                upload(&gpu, &vec![0.0; dm as usize]));
            dispatch(&gpu, &split, &[(0, &qkvb), (1, &q2), (2, &k2), (3, &v2)], &[(4, d), (5, dm)], &[],
                dm.div_ceil(64) as u64, 64);
            if freq_dims != 0 {
                let pairs = m as u32 * (d / hd) * (hd / 2);
                for t in [&q2, &k2] {
                    dispatch(&gpu, &rope, &[(0, t), (5, &posb)], &[(1, hd), (3, d), (4, m as u32), (6, freq_dims)],
                        &[(2, base)], pairs.div_ceil(64) as u64, 64);
                }
            }
            let (kh2, vh2) = (upload(&gpu, &vec![0.0; dm as usize / 2 + 1]), upload(&gpu, &vec![0.0; dm as usize / 2 + 1]));
            for (src, dst) in [(&k2, &kh2), (&v2, &vh2)] {
                dispatch(&gpu, &half, &[(0, src), (1, dst)], &[(2, dm)], &[], dm.div_ceil(256) as u64, 256);
            }
            let label = format!("{mode} m={m} d={d}");
            assert_eq!(words(&q, dm as usize), words(&q2, dm as usize), "{label}: Q differs from split + rope");
            assert_eq!(halves(&kh, dm as usize), halves(&kh2, dm as usize), "{label}: K differs from split + rope + half");
            assert_eq!(halves(&vh, dm as usize), halves(&vh2, dm as usize), "{label}: V differs from split + half");
        }
    }
}

/// `ffn_gu_rows` over ModernBERT's fused `[gate | up]` rows: `act(gate) * up`, first
/// half activated, against the f64 oracles, at ragged widths and row counts.
#[test]
fn ffn_gu_rows_matches_cpu_oracle() {
    let Some(gpu) = gpu_or_skip("encoder/ffn-gu-rows") else { return };
    let p = pipe(&gpu, "ffn_gu_rows");
    let oracles: [(u32, &str, fn(f32) -> f32); 3] = [
        (1, "gelu-tanh", ojas_cpu::cpu_math::gelu),
        (3, "gelu-erf", ojas_cpu::cpu_math::gelu_erf),
        (4, "relu", |x| x.max(0.0)),
    ];
    for &(m, f) in &[(1usize, 2624usize), (7, 1152), (3, 13)] {
        let mut seed = 0x61u32 ^ (m * f) as u32;
        let x: Vec<f32> = (0..m * 2 * f).map(|_| lcg(&mut seed) * 6.0).collect();
        let xb = upload(&gpu, &x);
        for (act, name, oracle) in oracles {
            let ob = guarded(&gpu, m * f);
            let cb = gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&p);
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_buffer(1, Some(&ob), (GUARD * 4) as u64);
            let n = (m * f) as u32;
            for (i, v) in [(2u64, f as u32), (3, n), (4, act)] { enc.set_bytes(i, 4, &v as *const u32 as *const c_void); }
            enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(256) as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_guards(&format!("ffn_gu_rows {name} m={m} f={f}"), &ob, m * f);
            let got = payload(&ob, m * f);
            for r in 0..m {
                for i in 0..f {
                    let (g, u) = (x[r * 2 * f + i], x[r * 2 * f + f + i]);
                    let want = oracle(g) as f64 * u as f64;
                    // The activation's error scales with |g| (the erf approximation is
                    // good to 1.5e-7 absolute, times 0.5|g|) and the product scales it
                    // by |u|, so the bound is relative to both, as in act_m's test.
                    let tol = 1e-6 * (g.abs() as f64).max(1.0) * (u.abs() as f64).max(1.0);
                    assert!((got[r * f + i] as f64 - want).abs() <= tol,
                        "ffn_gu_rows {name} m={m} f={f} row {r} col {i}: {} vs {want}", got[r * f + i]);
                }
            }
        }
    }
}
