//! Quantised prefill GEMMs on CUDA — the `gemm_q` family, twins of Metal's `gemm_mm_q4`,
//! `gemm_mm_q4l{,_sk,_hb}`, `gemm_mm_q6k`, `gemm_mm_q8`, plus `gemm_mm_q8_0` and the split-K
//! forms — against a CPU f32 reference built from the repo's own quantisers / dequantisers.
//!
//! GPU-only, so every test is `#[ignore]`:
//!
//! ```text
//! SP=~/.local/lib/python3.10/site-packages/nvidia
//! LD_LIBRARY_PATH=$SP/cuda_nvrtc/lib:$SP/cublas/lib OJAS_CUDA_INCLUDE=$SP/cuda_runtime/include \
//!   cargo test --release -p ojas-cuda --test prefill_gemm_q -- --ignored --nocapture
//! ```
//!
//! Weights per format, and where the reference W comes from:
//! * Q4 (pair layout): `ojas_formats::quant::q4_pair_from_f16`. The repo has no CPU
//!   dequantiser for this layout (only Metal kernels read it), so `dequant_q4_pair` below is
//!   its inverse, `(q - 8) · s`.
//! * Q4L: synthetic Q4_K blocks (`ojas_formats::synth::blocks(12, ..)`), reference W =
//!   `ojas_formats::gguf::dequant_to_f16(.., 12, ..)`; the kernel's inputs are those blocks
//!   relaid by `quant::relayout_q4k_q4l` (a CPU transcription of the Metal load-time kernel), so the
//!   kernel also carries the relayout's rounding of `d·sc` and `dmin·m` to f16.
//! * Q6_K: `synth::blocks(14, ..)`, reference `dequant_to_f16(.., 14, ..)`.
//! * Q8: `ojas_formats::quant::q8_rows_from_f16`, reference `q · scale`.
//!
//! Two references, as in `surya_gemm_f16`: `f32` (x as given) gives the error against the
//! model's math; `x16` (x rounded to f16, as the kernel does on its way into the tile) isolates
//! the kernel's own error. Errors are `max |got - ref| / rms(ref)` and rms(difference) /
//! rms(ref). Timings share the GPU with other work and are indicative.
use half::f16;
use ojas_core::{Device, KernelRuntime};
use ojas_cuda::CudaGpu;
use ojas_formats::{gguf, quant, synth};

type Buf = <CudaGpu as Device>::Buf;
type Enc = <CudaGpu as Device>::Enc;

fn gpu() -> CudaGpu {
    let mut g = CudaGpu::new(0).expect("these tests require a CUDA box: cargo test -- --ignored");
    g.ensure_family("gemm_q").expect("gemm_q family compiles");
    g.ensure_family("gemm_f16").expect("gemm_f16 family compiles");
    g
}

fn tvec(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 9) as f32 / (1 << 23) as f32) - 1.0
        })
        .collect()
}

fn round_f16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&a| f16::from_f32(a).to_f32()).collect()
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&a| f16::from_f32(a).to_bits().to_le_bytes()).collect()
}

fn f16_bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|a| a.to_le_bytes()).collect()
}

// ---------------- weight formats ----------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fmt { Q4, Q4l, Q6k, Q8, Q80, F16 }

/// Device-side weights plus the f32 reference W[n, k] the CPU dequantiser produces.
struct Weights {
    bufs: Vec<Buf>,   // w, then the side arrays in kernel-argument order
    wref: Vec<f32>,
}

/// Inverse of `quant::q4_pair_from_f16`: byte j of a 32-block holds elements 2j (low) and
/// 2j+1 (high), `v = (q - 8) · s`.
fn dequant_q4_pair(nib: &[u8], scale: &[u16], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * k];
    for r in 0..n {
        for b in 0..k / 32 {
            let s = f16::from_bits(scale[r * (k / 32) + b]).to_f32();
            for j in 0..16 {
                let by = nib[r * (k / 2) + b * 16 + j];
                out[r * k + b * 32 + 2 * j] = ((by & 15) as f32 - 8.0) * s;
                out[r * k + b * 32 + 2 * j + 1] = ((by >> 4) as f32 - 8.0) * s;
            }
        }
    }
    out
}

fn make_weights(g: &CudaGpu, fmt: Fmt, n: usize, k: usize, seed: u32) -> Weights {
    match fmt {
        Fmt::Q4 | Fmt::Q8 => {
            let s = 1.7 / (k as f32).sqrt();
            let w: Vec<f32> = tvec(n * k, seed).iter().map(|v| v * s).collect();
            let wb = f16_bytes(&w);
            if fmt == Fmt::Q4 {
                let (nib, sc) = quant::q4_pair_from_f16(&wb, n, k);
                let wref = dequant_q4_pair(&nib, &sc, n, k);
                Weights { bufs: vec![g.upload_bytes(&nib).unwrap(), g.upload_bytes(&u16_bytes(&sc)).unwrap()],
                          wref }
            } else {
                let (q, sc) = quant::q8_rows_from_f16(&wb, n, k);
                let wref: Vec<f32> = (0..n * k).map(|i| q[i] as f32 * sc[i / k]).collect();
                let qb: Vec<u8> = q.iter().map(|&v| v as u8).collect();
                Weights { bufs: vec![g.upload_bytes(&qb).unwrap(), g.upload(&sc)], wref }
            }
        }
        Fmt::Q4l => {
            let raw = synth::blocks(12, n * k / 256);
            let wref = f16_bytes_to_f32(&gguf::dequant_to_f16(&raw, 12, n * k));
            let (nib, qa, qb) = quant::relayout_q4k_q4l(&raw, k, n);
            Weights { bufs: vec![g.upload_bytes(&nib).unwrap(), g.upload_bytes(&u16_bytes(&qa)).unwrap(),
                                 g.upload_bytes(&u16_bytes(&qb)).unwrap()], wref }
        }
        Fmt::Q6k => {
            let raw = synth::blocks(14, n * k / 256);
            let wref = f16_bytes_to_f32(&gguf::dequant_to_f16(&raw, 14, n * k));
            Weights { bufs: vec![g.upload_bytes(&raw).unwrap()], wref }
        }
        Fmt::Q80 => {
            let raw = synth::blocks(8, n * k / 32);
            let wref = f16_bytes_to_f32(&gguf::dequant_to_f16(&raw, 8, n * k));
            Weights { bufs: vec![g.upload_bytes(&raw).unwrap()], wref }
        }
        Fmt::F16 => {
            let s = 1.7 / (k as f32).sqrt();
            let w: Vec<f32> = tvec(n * k, seed).iter().map(|v| v * s).collect();
            let wb = f16_bytes(&w);
            Weights { bufs: vec![g.upload_bytes(&wb).unwrap()], wref: f16_bytes_to_f32(&wb) }
        }
    }
}

// ---------------- kernels under test ----------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kern { Q4, Q4l, Q4lHb, Q6k, Q8, Q80, Q4lSk, Q80Sk, Q6kSk, F16Sk, Q4lH, Q80H, Q4lSkH, Q80SkH }

impl Kern {
    const ALL: [Kern; 14] = [Kern::Q4, Kern::Q4l, Kern::Q4lHb, Kern::Q6k, Kern::Q8, Kern::Q80,
                             Kern::Q4lSk, Kern::Q80Sk, Kern::Q6kSk, Kern::F16Sk,
                             Kern::Q4lH, Kern::Q80H, Kern::Q4lSkH, Kern::Q80SkH];
    fn split_k(self) -> bool { matches!(self, Kern::Q4lSk | Kern::Q80Sk | Kern::Q6kSk | Kern::F16Sk | Kern::Q4lSkH | Kern::Q80SkH) }
    fn x_half(self) -> bool { matches!(self, Kern::Q4lHb | Kern::Q4lSk | Kern::Q80Sk | Kern::Q6kSk | Kern::Q4lSkH | Kern::Q80SkH) }
    fn name(self) -> &'static str {
        match self {
            Kern::Q4 => "gemm_mm_q4", Kern::Q4l => "gemm_mm_q4l", Kern::Q4lHb => "gemm_mm_q4l_hb",
            Kern::Q6k => "gemm_mm_q6k", Kern::Q8 => "gemm_mm_q8", Kern::Q80 => "gemm_mm_q8_0",
            Kern::Q4lSk => "gemm_mm_q4l_sk", Kern::Q80Sk => "gemm_mm_q8_0_sk", Kern::Q6kSk => "gemm_mm_q6k_sk",
            Kern::F16Sk => "gemm_mm_f16_sk",
            Kern::Q4lH => "gemm_mm_q4l_h", Kern::Q80H => "gemm_mm_q8_0_h",
            Kern::Q4lSkH => "gemm_mm_q4l_sk_h", Kern::Q80SkH => "gemm_mm_q8_0_sk_h",
        }
    }
    fn fmt(self) -> Fmt {
        match self {
            Kern::Q4 => Fmt::Q4, Kern::Q6k | Kern::Q6kSk => Fmt::Q6k, Kern::Q8 => Fmt::Q8,
            Kern::Q80 | Kern::Q80Sk | Kern::Q80H | Kern::Q80SkH => Fmt::Q80, Kern::F16Sk => Fmt::F16, _ => Fmt::Q4l,
        }
    }
}

/// Metal's partition count (dispatch.rs), on this kernel's 64x128 tile: enough z-slices to
/// reach ~2 waves of blocks, at most 8, each at least one 32-block.
fn nsplit_for(m: usize, k: usize, n: usize) -> usize {
    let tiles = m.div_ceil(64) * n.div_ceil(128);
    let mut ns = (56 / tiles.max(1)).clamp(1, 8);
    while ns > 1 && k / ns < 32 { ns -= 1; }
    ns
}

/// Launch `kern`: y (+)= x · Wᵀ (the split-K forms add into y, zeroed first unless `accum`).
#[allow(clippy::too_many_arguments)]
fn launch(g: &CudaGpu, enc: &Enc, kern: Kern, x: &Buf, w: &Weights, y: &Buf, m: usize, k: usize, n: usize,
          accum: bool, nsplit: usize) {
    let grid = [n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1];
    let mut bufs: Vec<(&Buf, u64)> = vec![(x, 0), (&w.bufs[0], 0), (y, 0)];
    bufs.extend(w.bufs[1..].iter().map(|b| (b, 0)));
    let (k, n, m, ac) = (k as u32, n as u32, m as u32, accum as u32);
    if kern.split_k() {
        if !accum { g.zero_bytes(y, 0, (m * n) as usize * 4).unwrap(); }
        g.dispatch(enc, kern.name(), &bufs, &[k, n, m, nsplit as u32], [grid[0], grid[1], nsplit as u32], [256, 1, 1]).unwrap();
    } else {
        g.dispatch(enc, kern.name(), &bufs, &[k, n, ac, m], grid, [256, 1, 1]).unwrap();
    }
}

// ---------------- reference + metrics ----------------

/// Both references in one pass: (x · Wᵀ, round_f16(x) · Wᵀ), f32 with 8-lane partial sums,
/// threaded over output rows (or columns when M is small).
fn reference(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let x16 = round_f16(x);
    let dot2 = |xa: &[f32], xb: &[f32], wr: &[f32]| -> (f32, f32) {
        let (mut a, mut b) = ([0f32; 8], [0f32; 8]);
        for ((p, q), r) in xa.chunks_exact(8).zip(xb.chunks_exact(8)).zip(wr.chunks_exact(8)) {
            for i in 0..8 {
                a[i] += p[i] * r[i];
                b[i] += q[i] * r[i];
            }
        }
        (a.iter().sum(), b.iter().sum())
    };
    let threads = std::thread::available_parallelism().map_or(8, |t| t.get());
    let mut y = vec![0f32; m * n];
    let mut y16 = vec![0f32; m * n];
    // one task = a band of columns for every row (balanced for any M)
    let cols = n.div_ceil(threads * 4).max(1);
    std::thread::scope(|s| {
        let x16 = &x16;
        let parts: Vec<_> = (0..n.div_ceil(cols))
            .map(|c| {
                s.spawn(move || {
                    let (j0, j1) = (c * cols, ((c + 1) * cols).min(n));
                    let mut out = vec![(0f32, 0f32); m * (j1 - j0)];
                    for r in 0..m {
                        for j in j0..j1 {
                            out[r * (j1 - j0) + j - j0] =
                                dot2(&x[r * k..][..k], &x16[r * k..][..k], &w[j * k..][..k]);
                        }
                    }
                    (j0, j1, out)
                })
            })
            .collect();
        for p in parts {
            let (j0, j1, out) = p.join().unwrap();
            for r in 0..m {
                for j in j0..j1 {
                    let (a, b) = out[r * (j1 - j0) + j - j0];
                    y[r * n + j] = a;
                    y16[r * n + j] = b;
                }
            }
        }
    });
    (y, y16)
}

/// (max |got - want| / rms(want), rms(got - want) / rms(want)).
fn errs(got: &[f32], want: &[f32]) -> (f32, f32) {
    let rms = (want.iter().map(|v| (v * v) as f64).sum::<f64>() / want.len() as f64).sqrt();
    let mut emax = 0f64;
    let mut esq = 0f64;
    for (g, w) in got.iter().zip(want) {
        assert!(g.is_finite(), "non-finite output");
        let d = (g - w).abs() as f64;
        emax = emax.max(d);
        esq += d * d;
    }
    let rms = rms.max(1e-30);
    ((emax / rms) as f32, ((esq / want.len() as f64).sqrt() / rms) as f32)
}

fn time(g: &CudaGpu, iters: usize, mut f: impl FnMut(&CudaGpu, &Enc)) -> (f64, f64) {
    use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
    let enc = g.begin();
    for _ in 0..2 {
        f(g, &enc);
    }
    let ctx = enc.context().clone();
    let (e0, e1) = (ctx.new_event(Some(CU_EVENT_DEFAULT)).unwrap(),
                    ctx.new_event(Some(CU_EVENT_DEFAULT)).unwrap());
    let mut ts: Vec<f64> = (0..iters)
        .map(|_| {
            e0.record(&enc).unwrap();
            f(g, &enc);
            e1.record(&enc).unwrap();
            e1.synchronize().unwrap();
            e0.elapsed_ms(&e1).unwrap() as f64 * 1e-3
        })
        .collect();
    g.submit(enc).unwrap();
    ts.sort_by(f64::total_cmp);
    (ts[0], ts[ts.len() / 2])
}

/// Error bounds against the x16 reference (the kernel's own error). The f16 W grid is the
/// reference's for Q4 / Q6_K (exact dequant values) and Q8 (int8 exact, scale in f32); Q4L
/// carries the relayout's f16 rounding of qa / qb on top.
fn bound_x16(fmt: Fmt) -> (f32, f32) {
    match fmt {
        Fmt::Q4l => (2e-2, 2e-3),
        _ => (5e-3, 5e-4),
    }
}

struct Case { e32: (f32, f32), e16: (f32, f32), secs: f64 }

#[allow(clippy::too_many_arguments)]
fn run_case(g: &CudaGpu, kern: Kern, w: &Weights, m: usize, k: usize, n: usize, accum: bool,
            nsplit: usize, timed: bool) -> Case {
    let seed = (m * 31 + k * 7 + n) as u32;
    let x = tvec(m * k, seed);
    let y0: Vec<f32> = if accum { tvec(m * n, seed ^ 0xacc) } else { vec![0.0; m * n] };
    let (mut want, mut want16) = reference(&x, &w.wref, m, k, n);
    for ((a, b), c) in want.iter_mut().zip(want16.iter_mut()).zip(&y0) {
        *a += c;
        *b += c;
    }
    // guard row past M: nothing may be written there
    const SENT: f32 = 12345.5;
    let mut yinit = y0.clone();
    yinit.extend(std::iter::repeat(SENT).take(n));
    let xd = if kern.x_half() { g.upload_f16(&x) } else { g.upload(&x) };
    let yd = g.upload(&yinit);
    let enc = g.begin();
    launch(g, &enc, kern, &xd, w, &yd, m, k, n, accum, nsplit);
    g.submit(enc).unwrap();
    let mut got = vec![0f32; m * n + n];
    g.read(&yd, &mut got);
    assert!(got[m * n..].iter().all(|&v| v == SENT), "{} wrote past row M={m}", kern.name());
    let got = &got[..m * n];
    let e32 = errs(got, &want);
    let e16 = errs(got, &want16);
    let (secs, med) = if timed {
        time(g, 20, |g, enc| launch(g, enc, kern, &xd, w, &yd, m, k, n, false, nsplit))
    } else {
        (0.0, 0.0)
    };
    let tf = |s: f64| 2.0 * (m * k * n) as f64 / s / 1e12;
    let ts = if timed {
        format!("  {:8.3} ms {:6.2} TFLOP/s (median {:5.2})", secs * 1e3, tf(secs), tf(med))
    } else {
        String::new()
    };
    let sk = if kern.split_k() { format!(" z={nsplit}") } else { String::new() };
    println!("  {:16} M={m:5} K={k:5} N={n:5}{}{sk:4}  f32 max {:.2e} rms {:.2e} | x16 max {:.2e} rms {:.2e}{ts}",
             kern.name(), if accum { " +acc" } else { "     " }, e32.0, e32.1, e16.0, e16.1);
    let (bmax, brms) = bound_x16(kern.fmt());
    assert!(e16.0 < bmax && e16.1 < brms,
            "{} M={m} K={k} N={n}: kernel error {:?} vs the x16 reference (bound {bmax}/{brms})",
            kern.name(), e16);
    assert!(e32.0 < 2e-2 && e32.1 < 3e-3,
            "{} M={m} K={k} N={n}: error {:?} vs the f32 reference", kern.name(), e32);
    Case { e32, e16, secs }
}

/// The prefill grid for one kernel: M ∈ {1, 17, 512, 4180} × (K, N) ∈ {(1024, 1024),
/// (2048, 3584), (3584, 1024)}, timed at M ≥ 512; then ragged M / N and the accumulate flag.
fn sweep(kern: Kern) {
    let g = gpu();
    println!("{}", kern.name());
    let mut worst32 = (0f32, 0f32);
    let mut worst16 = (0f32, 0f32);
    let mut note = |c: &Case| {
        worst32 = (worst32.0.max(c.e32.0), worst32.1.max(c.e32.1));
        worst16 = (worst16.0.max(c.e16.0), worst16.1.max(c.e16.1));
    };
    let mut best_tf = 0f64;
    for (k, n) in [(1024, 1024), (2048, 3584), (3584, 1024)] {
        let w = make_weights(&g, kern.fmt(), n, k, (k * 3 + n) as u32);
        for m in [1, 17, 512, 4180] {
            let c = run_case(&g, kern, &w, m, k, n, false, nsplit_for(m, k, n), m >= 512);
            note(&c);
            if m == 4180 {
                best_tf = best_tf.max(2.0 * (m * k * n) as f64 / c.secs / 1e12);
            }
        }
    }
    // ragged: N not a tile multiple (even and odd), M off-tile, K a multiple of 256 only
    for (m, k, n, acc) in [(77, 1280, 1000, false), (129, 1280, 1001, true), (5, 2304, 77, false),
                           (300, 3584, 1024, true)] {
        let w = make_weights(&g, kern.fmt(), n, k, (k + n) as u32);
        note(&run_case(&g, kern, &w, m, k, n, acc, nsplit_for(m, k, n), false));
        if kern.split_k() {
            // an odd partition count whose last slice takes a remainder
            note(&run_case(&g, kern, &w, m, k, n, acc, 3, false));
        }
    }
    println!("  worst: f32 max {:.2e} rms {:.2e} | x16 max {:.2e} rms {:.2e}; best M=4180 {:.2} TFLOP/s",
             worst32.0, worst32.1, worst16.0, worst16.1, best_tf);
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_q4() { sweep(Kern::Q4) }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_q4l() { sweep(Kern::Q4l) }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_q4l_hb() { sweep(Kern::Q4lHb) }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_q6k() { sweep(Kern::Q6k) }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_q8() { sweep(Kern::Q8) }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_q8_0() { sweep(Kern::Q80) }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_q4l_sk() { sweep(Kern::Q4lSk) }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_q8_0_sk() { sweep(Kern::Q80Sk) }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_q6k_sk() { sweep(Kern::Q6kSk) }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_f16_sk() { sweep(Kern::F16Sk) }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_h() { for k in [Kern::Q4lH, Kern::Q80H, Kern::Q4lSkH, Kern::Q80SkH] { sweep(k) } }

/// Fails if an entry of the `gemm_q` family has no test above.
#[test]
fn every_gemm_q_entry_is_tested() {
    let tested: Vec<&str> = Kern::ALL.iter().map(|k| k.name()).collect();
    for name in ojas_cuda::kernels::gemm_q::NAMES {
        assert!(tested.contains(name), "{name} has no test here");
    }
}

/// The Q4_K / Q8_0 projections of Kev-4B at the chunk sizes a decision request prefills,
/// timed with the runner's rule (split-K up to 128 rows, the plain tile above): milliseconds,
/// the weight read against the card's bandwidth, and the arithmetic rate.
#[test]
#[ignore = "GPU"]
fn kev_shapes() {
    let g = gpu();
    // (name, K, N, fmt): gate|up, down, the SSM in-projection, ssm_out / attn_output
    let shapes = [("ffn gate|up", 2560, 18432, Fmt::Q4l), ("ffn down", 9216, 2560, Fmt::Q4l),
                  ("ssm in", 2560, 12352, Fmt::Q80), ("ssm out", 4096, 2560, Fmt::Q80)];
    let bits = |f: Fmt| match f { Fmt::Q4l => 4.5, Fmt::Q80 => 8.5, _ => 16.0 };
    for k in [Kern::Q4lSkH, Kern::Q80SkH, Kern::Q4lH, Kern::Q80H] {
        println!("  {:<18} {} blocks per SM", k.name(), g.blocks_per_sm(k.name(), 256).unwrap());
    }
    for (name, k, n, fmt) in shapes {
        let w = make_weights(&g, fmt, n, k, 7);
        for m in [16usize, 32, 64, 96, 128, 192, 256] {
            let kerns = match (fmt, m <= 128) {
                (Fmt::Q4l, true) => [Kern::Q4lSk, Kern::Q4lSkH], (Fmt::Q4l, false) => [Kern::Q4l, Kern::Q4lH],
                (_, true) => [Kern::Q80Sk, Kern::Q80SkH], (_, false) => [Kern::Q80, Kern::Q80H],
            };
            let tiles = m.div_ceil(64) * n.div_ceil(128);
            let mut ns = if m <= 128 { (224usize).div_ceil(tiles.max(1)).clamp(1, 8) } else { 1 };
            while ns > 1 && k / ns < 32 { ns -= 1; }
            let x = tvec(m * k, 3);
            let mut line = format!("  {name:<12} K={k:5} N={n:5} M={m:3} z={ns}:");
            for kern in kerns {
                let xd = if kern.x_half() { g.upload_f16(&x) } else { g.upload(&x) };
                let yd = g.alloc(m * n);
                let (secs, _) = time(&g, 20, |g, enc| launch(g, enc, kern, &xd, &w, &yd, m, k, n, false, ns));
                line += &format!("  {:<18} {:.3} ms {:>5.0} GB/s {:>5.1} TFLOP/s", kern.name(), secs * 1e3,
                                 (n * k) as f64 * bits(fmt) / 8.0 / secs / 1e9, 2.0 * (m * n * k) as f64 / secs / 1e12);
            }
            println!("{line}");
        }
    }
}
