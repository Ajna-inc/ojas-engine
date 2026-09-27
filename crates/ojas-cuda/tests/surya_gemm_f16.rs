//! F16-weight matmuls for surya-2 on CUDA: `gemm_mm_f16` (prefill / vision tower, tensor
//! cores), `gemv_f16` (one token) and `gemv_m_f16` (a few tokens), against a CPU f32 reference.
//!
//! GPU-only, so every test is `#[ignore]`:
//!
//! ```text
//! SP=~/.local/lib/python3.10/site-packages/nvidia
//! LD_LIBRARY_PATH=$SP/cuda_nvrtc/lib:$SP/cublas/lib OJAS_CUDA_INCLUDE=$SP/cuda_runtime/include \
//!   cargo test --release -p ojas-cuda --test surya_gemm_f16 -- --ignored --nocapture
//! ```
//!
//! Two references for the GEMM, because the kernel (like Metal's) rounds each activation to
//! f16 on its way into the tensor-core tile:
//! * `f32`: x as given — the error against the model's math;
//! * `x16`: x rounded to f16 first — the kernel's own error (accumulation order only).
//!
//! Errors are `max |got - ref| / rms(ref)` (relative, and it does not explode on the near-zero
//! outputs a random dot product produces) and the max element-wise relative error over outputs
//! with `|ref| >= rms(ref)`. Timings share the GPU with other work and are indicative.
use ojas_core::{Device, KernelRuntime};
use ojas_cuda::CudaGpu;

fn gpu() -> CudaGpu {
    let mut g = CudaGpu::new(0).expect("these tests require a CUDA box: cargo test -- --ignored");
    g.ensure_family("gemm_f16").expect("gemm_f16 family compiles");
    g
}

/// Deterministic pseudo-random values in [-1, 1) (no rand dep).
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
    v.iter().map(|&a| half::f16::from_f32(a).to_f32()).collect()
}

/// Weights `[n, k]`, already on the f16 grid so the f32 reference sees the exact values the GPU does.
fn weights(n: usize, k: usize, seed: u32) -> Vec<f32> {
    let s = 1.0 / (k as f32).sqrt();
    round_f16(&tvec(n * k, seed).iter().map(|v| v * s * 1.7).collect::<Vec<_>>())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// CPU f32 reference: y[m, n] = Σ_k x[m, k] · w[n, k]. Threaded over outputs.
fn reference(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    let threads = std::thread::available_parallelism().map_or(8, |t| t.get());
    // split by output rows when there are enough of them, else by output columns
    std::thread::scope(|s| {
        if m >= threads {
            let rows = m.div_ceil(threads);
            for (ci, chunk) in y.chunks_mut(rows * n).enumerate() {
                s.spawn(move || {
                    for (r, yr) in chunk.chunks_mut(n).enumerate() {
                        let xr = &x[(ci * rows + r) * k..][..k];
                        for (j, o) in yr.iter_mut().enumerate() {
                            *o = dot(xr, &w[j * k..][..k]);
                        }
                    }
                });
            }
        } else {
            let cols = n.div_ceil(threads);
            let parts: Vec<_> = (0..n.div_ceil(cols))
                .map(|c| {
                    s.spawn(move || {
                        let (j0, j1) = (c * cols, ((c + 1) * cols).min(n));
                        let mut out = vec![0.0f32; m * (j1 - j0)];
                        for r in 0..m {
                            for j in j0..j1 {
                                out[r * (j1 - j0) + j - j0] = dot(&x[r * k..][..k], &w[j * k..][..k]);
                            }
                        }
                        (j0, j1, out)
                    })
                })
                .collect();
            for p in parts {
                let (j0, j1, out) = p.join().unwrap();
                for r in 0..m {
                    y[r * n + j0..r * n + j1].copy_from_slice(&out[r * (j1 - j0)..][..j1 - j0]);
                }
            }
        }
    });
    y
}

/// (max |got - want| / rms(want), max relative error over |want| >= rms(want)).
fn errs(got: &[f32], want: &[f32]) -> (f32, f32) {
    let rms = (want.iter().map(|v| (v * v) as f64).sum::<f64>() / want.len() as f64).sqrt() as f32;
    let mut e_abs = 0.0f32;
    let mut e_rel = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        assert!(g.is_finite(), "non-finite output");
        let d = (g - w).abs();
        e_abs = e_abs.max(d);
        if w.abs() >= rms {
            e_rel = e_rel.max(d / w.abs());
        }
    }
    (e_abs / rms.max(1e-30), e_rel)
}

/// Seconds per call, timed launch by launch with CUDA events: the minimum over `iters`
/// launches after a warm-up. Sharing the GPU only ever makes a launch slower, so the fastest
/// is closest to the kernel alone; the median is returned too, as the contended figure.
fn time(g: &CudaGpu, iters: usize, mut f: impl FnMut(&CudaGpu, &<CudaGpu as Device>::Enc, usize)) -> (f64, f64) {
    use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
    let enc = g.begin();
    for i in 0..2 {
        f(g, &enc, i);
    }
    let ctx = enc.context().clone();
    let (e0, e1) = (ctx.new_event(Some(CU_EVENT_DEFAULT)).unwrap(),
                    ctx.new_event(Some(CU_EVENT_DEFAULT)).unwrap());
    let mut ts: Vec<f64> = (0..iters)
        .map(|i| {
            e0.record(&enc).unwrap();
            f(g, &enc, i);
            e1.record(&enc).unwrap();
            e1.synchronize().unwrap();
            e0.elapsed_ms(&e1).unwrap() as f64 * 1e-3
        })
        .collect();
    g.submit(enc).unwrap();
    ts.sort_by(f64::total_cmp);
    (ts[0], ts[ts.len() / 2])
}

type Buf = <CudaGpu as Device>::Buf;

fn gemm(g: &CudaGpu, enc: &<CudaGpu as Device>::Enc, x: &Buf, w: &Buf, y: &Buf,
        m: usize, k: usize, n: usize, accum: bool) {
    g.dispatch(enc, "gemm_mm_f16", &[(x, 0), (w, 0), (y, 0)],
               &[k as u32, n as u32, accum as u32, m as u32],
               [n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1], [256, 1, 1])
        .unwrap();
}

/// One GEMM case: checks both references, that rows past M are never written, and — with
/// `accum` — that y is added to rather than overwritten. Returns (err vs f32, seconds/call).
fn gemm_case(g: &CudaGpu, m: usize, k: usize, n: usize, accum: bool, timed: bool) -> (f32, f64) {
    let seed = (m * 31 + k * 7 + n) as u32;
    let x = tvec(m * k, seed);
    let w = weights(n, k, seed ^ 0x5eed);
    let y0: Vec<f32> = if accum { tvec(m * n, seed ^ 0xacc) } else { vec![0.0; m * n] };
    let add = |mut r: Vec<f32>| { for (a, b) in r.iter_mut().zip(&y0) { *a += b; } r };
    let want = add(reference(&x, &w, m, k, n));
    let want16 = add(reference(&round_f16(&x), &w, m, k, n));

    // one guard row past M: the kernel must not store whole tiles the way Metal's does
    const SENT: f32 = 12345.5;
    let mut yinit = y0.clone();
    yinit.extend(std::iter::repeat(SENT).take(n));
    let (xd, wd, yd) = (g.upload(&x), g.upload_f16(&w), g.upload(&yinit));
    let enc = g.begin();
    gemm(g, &enc, &xd, &wd, &yd, m, k, n, accum);
    g.submit(enc).unwrap();
    let mut got = vec![0.0f32; m * n + n];
    g.read(&yd, &mut got);
    assert!(got[m * n..].iter().all(|&v| v == SENT), "gemm_mm_f16 wrote past row M={m}");
    let got = &got[..m * n];
    let (e32, r32) = errs(got, &want);
    let (e16, r16) = errs(got, &want16);

    let (secs, med) = if timed {
        time(g, 40, |g, enc, _| gemm(g, enc, &xd, &wd, &yd, m, k, n, false))
    } else {
        (0.0, 0.0)
    };
    let tflops = |s: f64| 2.0 * (m * k * n) as f64 / s / 1e12;
    let tf = if timed { format!("{:6.3} ms {:5.2} TFLOP/s (median {:5.2})", secs * 1e3,
                                tflops(secs), tflops(med)) } else { String::new() };
    println!("  M={m:5} K={k:5} N={n:5}{}  err/rms f32 {e32:.2e} (rel {r32:.2e})  x16 {e16:.2e} (rel {r16:.2e})  {tf}",
             if accum { " +acc" } else { "     " });
    assert!(e16 < 1e-4, "gemm_mm_f16 M={m} K={k} N={n}: kernel error {e16} vs the f16-x reference");
    assert!(e32 < 5e-3, "gemm_mm_f16 M={m} K={k} N={n}: error {e32} vs the f32 reference");
    (e32, secs)
}

/// The prefill shapes: a ~4,200-token page through surya-2's projections (K, N from
/// {768, 1024, 2048, 3072, 3584}).
#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_f16_surya_prefill_shapes() {
    let g = gpu();
    println!("gemm_mm_f16, M = 4180 (prefill)");
    let shapes = [(1024, 1024), (1024, 2048), (2048, 1024), (1024, 3584), (3584, 1024),
                  (768, 3072), (3072, 768), (768, 768), (2048, 2048)];
    let mut worst = 0.0f32;
    for (k, n) in shapes {
        worst = worst.max(gemm_case(&g, 4180, k, n, false, true).0);
    }
    println!("  worst err/rms vs f32: {worst:.2e}");
}

/// Ragged M (1, 3, 17, 4180), N and K that are not tile multiples (including K % 8 != 0,
/// the scalar path), and the accumulate flag.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemm_mm_f16_ragged_and_accum() {
    let g = gpu();
    println!("gemm_mm_f16, ragged shapes");
    for m in [1, 3, 17, 129, 4180] {
        gemm_case(&g, m, 1024, 3584, false, true);
    }
    for (m, k, n) in [(100, 1000, 1000),   // K % 32 = 8, N % 128 = 104
                      (129, 776, 770),     // K % 32 = 8, N % 128 = 2
                      (33, 1030, 77),      // K % 8 = 6: scalar path, odd... N = 77
                      (17, 3584, 1025),    // odd N: scalar stores
                      (4180, 3072, 768)] {
        gemm_case(&g, m, k, n, false, false);
    }
    for (m, k, n) in [(17, 1024, 3584), (4180, 3584, 1024), (33, 1030, 77)] {
        gemm_case(&g, m, k, n, true, false);
    }
}

fn gemv_m(g: &CudaGpu, enc: &<CudaGpu as Device>::Enc, x: &Buf, w: &Buf, y: &Buf,
          m: usize, k: usize, n: usize) {
    let grid = [n.div_ceil(8) as u32, 1, 1];
    if m == 0 {
        g.dispatch(enc, "gemv_f16", &[(x, 0), (w, 0), (y, 0)], &[k as u32, n as u32], grid, [256, 1, 1])
    } else {
        g.dispatch(enc, "gemv_m_f16", &[(x, 0), (w, 0), (y, 0)], &[k as u32, n as u32, m as u32],
                   grid, [256, 1, 1])
    }
    .unwrap();
}

/// One GEMV case (`m == 0` means `gemv_f16`, else `gemv_m_f16` with M = m). Checks against the
/// f32 reference, then times it over enough distinct weight copies (≥ 48 MB) that the read
/// comes from DRAM, not the 3 MB L2. Returns (err, GB/s of weight read).
fn gemv_case(g: &CudaGpu, m: usize, k: usize, n: usize, timed: bool) -> (f32, f64) {
    let rows = m.max(1);
    let seed = (rows * 131 + k * 7 + n) as u32;
    let x = tvec(rows * k, seed);
    let w = weights(n, k, seed ^ 0x77);
    let want = reference(&x, &w, rows, k, n);
    let (xd, yd) = (g.upload(&x), g.alloc(rows * n));
    let copies = if timed { (48usize << 20).div_ceil(n * k * 2).clamp(1, 64) } else { 1 };
    let wds: Vec<Buf> = (0..copies).map(|_| g.upload_f16(&w)).collect();
    let enc = g.begin();
    gemv_m(g, &enc, &xd, &wds[0], &yd, m, k, n);
    g.submit(enc).unwrap();
    let mut got = vec![0.0f32; rows * n];
    g.read(&yd, &mut got);
    let (e, r) = errs(&got, &want);
    let (gbs, gmed, us) = if timed {
        let iters = (copies * 4).max(60);
        let (s, med) = time(g, iters, |g, enc, i| gemv_m(g, enc, &xd, &wds[i % copies], &yd, m, k, n));
        ((n * k * 2) as f64 / s / 1e9, (n * k * 2) as f64 / med / 1e9, s)
    } else {
        (0.0, 0.0, 0.0)
    };
    let name = if m == 0 { "gemv_f16    ".to_string() } else { format!("gemv_m_f16 M={m}") };
    println!("  {name:15} K={k:5} N={n:6}  err/rms {e:.2e} (rel {r:.2e}){}",
             if timed { format!("  {:7.2} us  {gbs:6.1} GB/s weight read (median {gmed:6.1})", us * 1e6) }
             else { String::new() });
    assert!(e < 1e-5, "{name} K={k} N={n}: error {e} vs the f32 reference");
    (e, gbs)
}

/// Decode, one token: surya-2's per-token projections and the lm_head, vs the ~360 GB/s roof.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemv_f16_surya_decode_shapes() {
    let g = gpu();
    println!("gemv_f16 (M = 1)");
    for (k, n) in [(1024, 65425), (1024, 6144), (1024, 4096), (3584, 1024), (2048, 1024),
                   (1024, 3584), (1024, 2048), (1024, 512), (768, 3072), (3072, 768), (1024, 16)] {
        gemv_case(&g, 0, k, n, true);
    }
    for (k, n) in [(1030, 77), (776, 1001), (5, 3)] {   // K % 8 != 0, ragged N
        gemv_case(&g, 0, k, n, false);
    }
}

/// Decode, a few tokens: each weight row is read once for all M. Prints the batching gain
/// against M separate `gemv_f16` reads of the same matrix.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn gemv_m_f16_surya_decode_shapes() {
    let g = gpu();
    println!("gemv_m_f16");
    for (k, n) in [(1024, 65425), (1024, 3584), (3584, 1024), (1024, 2048)] {
        let (_, base) = gemv_case(&g, 0, k, n, true);
        for m in [1, 2, 3, 4, 5, 6, 8] {
            let (_, gbs) = gemv_case(&g, m, k, n, true);
            println!("      M={m}: {:.2}x the rows/s of M x gemv_f16", m as f64 * gbs / base);
        }
    }
    for (m, k, n) in [(3, 1030, 77), (8, 776, 1001), (11, 1024, 515), (5, 13, 9)] {
        gemv_case(&g, m, k, n, false);   // K % 8 != 0, ragged N, M > 8 (chunked)
    }
}
