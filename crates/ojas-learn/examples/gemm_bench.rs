//! Raw GEMM throughput of the learn kernels: `gemm_bench [m n k] [ta tb]`.
use std::time::Instant;

use ojas_learn::backend::{Backend, Gemm};
use ojas_learn::cuda::{Cuda, Prec};

fn main() -> anyhow::Result<()> {
    let a: Vec<usize> = std::env::args().skip(1).map(|v| v.parse().unwrap()).collect();
    let (m, n, k) = (a.first().copied().unwrap_or(2048), a.get(1).copied().unwrap_or(2048), a.get(2).copied().unwrap_or(2048));
    let (ta, tb) = (a.get(3).is_some_and(|&v| v == 1), a.get(4).is_some_and(|&v| v == 1));
    let mut be = Cuda::new(0)?;
    let x = be.upload(&vec![0.5; m * k]);
    let y = be.upload(&vec![0.25; k * n]);
    let c = be.alloc(m * n);
    for prec in [Prec::F32, Prec::Tf32, Prec::Bf16] {
        be.prec = prec;
        let g = Gemm { ta, tb, ..Gemm::plain(m, n, k, ta, tb) };
        be.gemm(&x, &y, &c, &g);
        be.download(&c);
        let reps = 10;
        let t = Instant::now();
        for _ in 0..reps {
            be.gemm(&x, &y, &c, &g);
        }
        be.download(&c);
        let s = t.elapsed().as_secs_f64() / reps as f64;
        println!("{prec:?} {m}x{n}x{k} ta={ta} tb={tb}: {:.2} ms, {:.1} TFLOPS", s * 1e3, 2.0 * (m * n * k) as f64 / s / 1e12);
    }
    Ok(())
}
