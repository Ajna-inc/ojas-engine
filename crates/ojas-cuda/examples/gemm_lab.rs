//! Lab for the tiled 1x1-conv GEMM: Y[m][n] = sum_k X[m][k] * W[n][k].
//! X rows are pixels (row offsets from a table, channels contiguous), W is
//! [N x K] row-major, Y rows via an output offset table. Checks against a CPU
//! reference on sampled entries and times the OCR's wide 1x1 shapes (b32).
//!
//! `gemm_lab <kernel.c> [-DKNOB=..;..]...`

use std::time::Instant;

use ojas_core::{Device, KernelRuntime};
use ojas_cuda::CudaGpu;

// (M = pixels, K = cin, N = cout, cuDNN b32 ms)
const SHAPES: &[(usize, usize, usize, f64)] = &[
    (15360, 2176, 512, 1.431),
    (7680, 3328, 1024, 2.201),
    (15360, 1664, 512, 1.089),
    (30720, 704, 256, 0.483),
    (7680, 1024, 2048, 1.404),
    (15360, 512, 1024, 0.732),
    (30720, 256, 512, 0.392),
    (7680, 384, 384, 0.110),
];

fn tvec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
    }).collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let src = std::fs::read_to_string(&args[1])?;
    let variants: Vec<String> = if args.len() > 2 { args[2..].to_vec() } else { vec![String::new()] };
    let g = CudaGpu::new(0)?;
    for v in &variants {
        let defs: String = v.split(';').filter(|d| !d.is_empty()).map(|d| {
            let (k, val) = d.split_once('=').unwrap_or((d, ""));
            format!("#define {k} {val}\n")
        }).collect();
        let full: &'static str = Box::leak(format!("{defs}{src}").into_boxed_str());
        let f = g.pipeline(full, "lab_gemm")?;
        let mut line = format!("{:<28}", if v.is_empty() { "baseline" } else { v });
        let (mut tot, mut ref_tot) = (0.0, 0.0);
        for &(m, k, n, cudnn) in SHAPES {
            // X as padded-NHWC-like rows: pixel stride = k + 16 (a channel slice)
            let ld = k + 16;
            let x = tvec(m * ld, 3, 1.0);
            let wt = tvec(n * k, 5, 1.0 / (k as f32).sqrt());
            let npad = n.div_ceil(256) * 256;
            let kpad = k.div_ceil(32) * 32;
            let mut wp = vec![0.0f32; npad * kpad];
            for r in 0..n {
                wp[r * kpad..r * kpad + k].copy_from_slice(&wt[r * k..r * k + k]);
            }
            let xoff: Vec<f32> = (0..m).map(|i| f32::from_bits((i * ld) as u32)).collect();
            let yoff: Vec<f32> = (0..m).map(|i| f32::from_bits((i * n) as u32)).collect();
            let (xd, wd, yd, xo, yo) = (g.upload_f16(&x), g.upload_f16(&wp), g.upload_f16(&vec![0.0; m * n]), g.upload(&xoff), g.upload(&yoff));
            let consts = [m, n, k, kpad].map(|c| c as u32);
            let bn: usize = std::env::var("BN").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
            let bm: usize = std::env::var("BM").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
            let grid = [m.div_ceil(bm) as u32, n.div_ceil(bn) as u32, 1];
            let smem: u32 = std::env::var("SMEM").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let go = || {
                g.dispatch_pipeline(&f, &[(&xd, 0), (&wd, 0), (&yd, 0), (&xo, 0), (&yo, 0)], &consts, grid, [256, 1, 1], smem).unwrap();
                g.submit(g.begin()).unwrap();
            };
            go();
            // check sampled entries against f64 on f16-rounded inputs
            let mut y = vec![0.0f32; m * n];
            g.read(&yd, &mut y);
            let r16 = |a: f32| half::f16::from_f32(a).to_f32() as f64;
            let mut worst = 0.0f64;
            for t in 0..512 {
                let (i, j) = ((t * 7919) % m, (t * 104729) % n);
                let want: f64 = (0..k).map(|kk| r16(x[i * ld + kk]) * r16(wt[j * k + kk])).sum();
                worst = worst.max((y[i * n + j] as f64 - want).abs() / want.abs().max(0.05));
            }
            for _ in 0..3 { go(); }
            let mut ts: Vec<f64> = (0..11).map(|_| { let t = Instant::now(); go(); t.elapsed().as_secs_f64() * 1e3 }).collect();
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mark = if worst < 5e-3 { " " } else { "!" };
            line += &format!(" {:>6.3}{mark}", ts[5]);
            tot += ts[5];
            ref_tot += cudnn;
        }
        println!("{line} | sum {tot:.3} ms (cuDNN {ref_tot:.3}, ratio {:.2})", tot / ref_tot);
    }
    Ok(())
}
