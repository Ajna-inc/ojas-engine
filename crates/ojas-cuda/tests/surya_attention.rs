//! Bidirectional (non-causal) attention for the surya-2 vision tower on CUDA:
//! `attention_m_bidir` (streaming reference) and `attention_m_mma_bidir_{64,128}` (tensor-core
//! flash attention), both against a CPU softmax(Q·Kᵀ·scale)·V with no mask.
//!
//! Inputs follow the tower exactly: Q f32 `[n, heads*hd]`, K/V f32 rounded to f16 by the copy
//! that precedes attention (`copy_f32_half`), `scale = 1/sqrt(hd)`, `total = mtok = n`. The CPU
//! reference sees the same f16-rounded K/V, so it measures the kernels and not the input copy,
//! and accumulates in f64.
//!
//! Tolerances (outputs are convex combinations of V ∈ [-1, 1], logits have std ~3):
//! * `attention_m_bidir` (f32 Q, f32 softmax) vs CPU: max abs ≤ 1e-4.
//! * `attention_m_mma_bidir_*` rounds Q and the probabilities P to f16 (as Metal's MMA kernel
//!   and ggml's flash-attn do), a relative 2^-11 on each — max abs ≤ 1e-3 against both the CPU
//!   and `attention_m_bidir`.
//!
//! Large shapes check every row of the MMA kernel against `attention_m_bidir` on the GPU and a
//! strided sample of rows (every head) against the CPU, which cannot afford 16k² × 12 heads.
//!
//! GPU only:
//! `cargo test --release -p ojas-cuda --test surya_attention -- --ignored --nocapture`

use std::sync::Mutex;
use std::time::Instant;

use ojas_core::{Device, KernelRuntime};
use ojas_cuda::kernels::attn_bidir::{bidir_launch, mma_bidir_launch};
use ojas_cuda::{CuBuf, CudaGpu};

/// Timings are meaningless with two shapes on the GPU at once.
static SERIAL: Mutex<()> = Mutex::new(());

const TOL_SIMPLE: f32 = 1e-4;
const TOL_MMA: f32 = 1e-3;

fn tvec(n: usize, seed: u32, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (((s >> 9) as f32 / (1 << 23) as f32) - 1.0) * amp
        })
        .collect()
}

fn f16_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
}

/// softmax(q·kᵀ·scale)·v for the given query rows (all heads), f64 accumulation.
/// Returns `rows.len() * heads * hd` values, row-major like `out`.
fn cpu_ref(q: &[f32], k: &[f32], v: &[f32], n: usize, heads: usize, hd: usize, scale: f32,
           rows: &[usize]) -> Vec<f32> {
    let d = heads * hd;
    let mut res = vec![0f32; rows.len() * d];
    let nth = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4);
    let per = rows.len().div_ceil(nth).max(1);
    std::thread::scope(|sc| {
        for (chunk, rs) in res.chunks_mut(per * d).zip(rows.chunks(per)) {
            sc.spawn(move || {
                let mut s = vec![0f64; n];
                for (ri, &m) in rs.iter().enumerate() {
                    for h in 0..heads {
                        let qh = &q[m * d + h * hd..][..hd];
                        let mut mx = f64::NEG_INFINITY;
                        for t in 0..n {
                            let kt = &k[t * d + h * hd..][..hd];
                            let dot: f64 = qh.iter().zip(kt).map(|(a, b)| *a as f64 * *b as f64).sum();
                            s[t] = dot * scale as f64;
                            mx = mx.max(s[t]);
                        }
                        let mut den = 0f64;
                        let mut acc = vec![0f64; hd];
                        for t in 0..n {
                            let p = (s[t] - mx).exp();
                            den += p;
                            let vt = &v[t * d + h * hd..][..hd];
                            for i in 0..hd { acc[i] += p * vt[i] as f64; }
                        }
                        for i in 0..hd { chunk[ri * d + h * hd + i] = (acc[i] / den) as f32; }
                    }
                }
            });
        }
    });
    res
}

#[derive(Default, Debug)]
struct Err { abs: f32, rel: f32 }

/// max |got-want| and max relative error over entries with |want| ≥ 0.05 (the outputs are
/// averages that can sit arbitrarily close to zero, where a relative error means nothing).
fn err(got: &[f32], want: &[f32]) -> Err {
    let mut e = Err::default();
    for (g, w) in got.iter().zip(want) {
        assert!(g.is_finite(), "non-finite output {g}");
        let d = (g - w).abs();
        e.abs = e.abs.max(d);
        if w.abs() >= 0.05 { e.rel = e.rel.max(d / w.abs()); }
    }
    e
}

fn gather(full: &[f32], rows: &[usize], d: usize) -> Vec<f32> {
    rows.iter().flat_map(|&m| full[m * d..(m + 1) * d].iter().copied()).collect()
}

struct Case { g: CudaGpu, q: CuBuf, k: CuBuf, v: CuBuf, out: CuBuf }

fn dispatch(c: &Case, name: &str, n: u32, heads: u32, hd: u32, scale: f32,
            launch: ([u32; 3], [u32; 3])) {
    let enc = c.g.begin();
    let consts = [hd, heads * hd, n, 1, scale.to_bits(), heads, n];
    // attention_m_bidir takes no mtok (Metal buffer 10 exists only on the MMA kernel)
    let consts = if name == "attention_m_bidir" { &consts[..6] } else { &consts[..] };
    c.g.dispatch(&enc, name, &[(&c.q, 0), (&c.k, 0), (&c.v, 0), (&c.out, 0)], consts,
                 launch.0, launch.1).unwrap();
    c.g.submit(enc).unwrap();
}

/// Run `name` `iters` times after one warm-up; returns (output, ms per call).
fn run(c: &mut Case, name: &str, n: usize, heads: usize, hd: usize, scale: f32, iters: usize)
       -> (Vec<f32>, f64) {
    let launch = if name == "attention_m_bidir" {
        bidir_launch(heads as u32, n as u32)
    } else {
        mma_bidir_launch(heads as u32, n as u32)
    };
    // poison the output so a kernel that skips rows cannot pass on a stale result
    c.out = c.g.upload(&vec![f32::NAN; n * heads * hd]);
    dispatch(c, name, n as u32, heads as u32, hd as u32, scale, launch);
    let t0 = Instant::now();
    for _ in 0..iters {
        dispatch(c, name, n as u32, heads as u32, hd as u32, scale, launch);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let mut o = vec![0f32; n * heads * hd];
    c.g.read(&c.out, &mut o);
    (o, ms)
}

fn tflops(n: usize, heads: usize, hd: usize, ms: f64) -> f64 {
    // QKᵀ and P·V, 2 flops per MAC each
    4.0 * (n * n) as f64 * (heads * hd) as f64 / (ms * 1e-3) / 1e12
}

fn shape(n: usize, heads: usize, hd: usize, mma_iters: usize, simple_iters: usize) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut g = CudaGpu::new(0).expect("requires a CUDA GPU");
    g.ensure_family("attn_bidir").expect("attn_bidir family compiles");
    let d = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    // amplitude 3: logit std ~3 (peaky rows), which is where an f16 softmax would show
    let q = tvec(n * d, 11 + n as u32, 3.0);
    let k = f16_round(&tvec(n * d, 23 + n as u32, 3.0));
    let v = f16_round(&tvec(n * d, 37 + n as u32, 1.0));
    let mut c = Case {
        q: g.upload(&q), k: g.upload_f16(&k), v: g.upload_f16(&v), out: g.alloc(n * d), g,
    };

    let mma = format!("attention_m_mma_bidir_{hd}");
    let (o_mma, ms_mma) = run(&mut c, &mma, n, heads, hd, scale, mma_iters);
    let (o_simple, ms_simple) = run(&mut c, "attention_m_bidir", n, heads, hd, scale, simple_iters);

    let rows: Vec<usize> = if n <= 1024 {
        (0..n).collect()
    } else {
        let s = 192;
        let mut r: Vec<usize> = (0..s).map(|i| (i * n / s + i % 7).min(n - 1)).collect();
        r.push(n - 1);
        r.dedup();
        r
    };
    let t = Instant::now();
    let want = cpu_ref(&q, &k, &v, n, heads, hd, scale, &rows);
    let cpu_s = t.elapsed().as_secs_f64();
    let e_simple = err(&gather(&o_simple, &rows, d), &want);
    let e_mma = err(&gather(&o_mma, &rows, d), &want);
    let e_pair = err(&o_mma, &o_simple);

    println!("\n== {n} tokens x {heads} heads x hd {hd}  (CPU ref on {} rows x all heads, {cpu_s:.1} s)",
             rows.len());
    println!("  {mma:<26} vs CPU: max abs {:.3e}  max rel {:.3e}", e_mma.abs, e_mma.rel);
    println!("  {:<26} vs CPU: max abs {:.3e}  max rel {:.3e}", "attention_m_bidir", e_simple.abs, e_simple.rel);
    println!("  {mma:<26} vs attention_m_bidir (all {n} rows): max abs {:.3e}  max rel {:.3e}",
             e_pair.abs, e_pair.rel);
    println!("  {mma:<26} {ms_mma:9.3} ms  {:6.2} TFLOP/s", tflops(n, heads, hd, ms_mma));
    println!("  {:<26} {ms_simple:9.3} ms  {:6.2} TFLOP/s", "attention_m_bidir", tflops(n, heads, hd, ms_simple));

    assert!(e_simple.abs <= TOL_SIMPLE, "attention_m_bidir vs CPU {:?} > {TOL_SIMPLE}", e_simple);
    assert!(e_mma.abs <= TOL_MMA, "{mma} vs CPU {:?} > {TOL_MMA}", e_mma);
    assert!(e_pair.abs <= TOL_MMA, "{mma} vs attention_m_bidir {:?} > {TOL_MMA}", e_pair);
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn bidir_tiny_17() { shape(17, 12, 64, 20, 20); }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn bidir_1024() { shape(1024, 12, 64, 20, 5); }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn bidir_4096() { shape(4096, 12, 64, 10, 1); }

#[test]
#[ignore = "requires NVIDIA GPU"]
fn bidir_16384() { shape(16384, 12, 64, 5, 1); }

/// hd 128 at a ragged length (not a multiple of the 32-key tile or the 64-row block).
#[test]
#[ignore = "requires NVIDIA GPU"]
fn bidir_hd128_1000() { shape(1000, 8, 128, 20, 5); }
