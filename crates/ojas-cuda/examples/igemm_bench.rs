//! Conv variants head to head on chosen shapes (fp16, NHWC), each checked
//! against the direct kernel. Shapes: `cin,h,w,cout,k,s[,batch]` arguments;
//! defaults are the heaviest layers of the three production models.
//!
//! `cargo run --release -p ojas-cuda --example igemm_bench -- [shape..]`

use ojas_cuda::conv::{ConvGeom, ConvPlan, ConvVariant};
use ojas_cuda::device::CudaGpu;
use ojas_core::{Device, KernelRuntime};
use std::time::Instant;

fn tvec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let shapes: Vec<[usize; 7]> = if args.is_empty() {
        vec![
            [3328, 3, 80, 1024, 1, 1, 32],
            [2176, 6, 80, 512, 1, 1, 32],
            [512, 6, 80, 1024, 1, 1, 32],
            [64, 80, 80, 64, 3, 1, 8],
            [128, 80, 80, 128, 3, 2, 8],
            [16, 320, 320, 32, 3, 2, 8],
            [64, 48, 48, 64, 3, 1, 32],
            [16, 96, 96, 16, 3, 1, 32],
        ]
    } else {
        args.iter()
            .map(|a| {
                let v: Vec<usize> = a.split(',').map(|x| x.parse().unwrap()).collect();
                [v[0], v[1], v[2], v[3], v[4], v[5], *v.get(6).unwrap_or(&8)]
            })
            .collect()
    };
    let mut g = CudaGpu::new(0)?;
    g.ensure_family("cnn")?;
    const REPS: usize = 20;
    for (si, &[cin, h, w, cout, k, s, n]) in shapes.iter().enumerate() {
        let geom = ConvGeom::square(cin, h, w, cout, k, s, k / 2);
        let wt = tvec(cout * cin * k * k, si as u64 + 7, 1.0 / ((cin * k * k) as f32).sqrt());
        let bias = tvec(cout, si as u64 + 9, 0.1);
        let mut plan = ConvPlan::new(&g, geom, &wt, Some(&bias), 1, 0)?;
        let (ist, ost) = (plan.input, plan.output);
        let x = g.upload_f16(&tvec(n * ist.img(), si as u64 + 3, 1.0));
        let y = g.upload_f16(&vec![0.0; n * ost.img()]);
        let (oh, ow) = geom.out_hw();
        let flops = 2.0 * (n * oh * ow * cout * cin * k * k) as f64;
        let mut reference: Option<Vec<f32>> = None;
        let mut line = format!("{cin:>5}x{h}x{w} -> {cout:<5} k{k} s{s} n{n:<3}|");
        for v in plan.variants() {
            plan.variant = v;
            let enc = g.begin();
            plan.dispatch(&g, &enc, &x, &y, n)?;
            g.submit(enc)?;
            let mut out = vec![0.0f32; n * ost.img()];
            g.read(&y, &mut out);
            let err = match &reference {
                None => {
                    reference = Some(out);
                    0.0
                }
                Some(r) => r.iter().zip(&out).map(|(a, b)| (a - b).abs() / a.abs().max(1e-1)).fold(0.0f32, f32::max),
            };
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let enc = g.begin();
                let t = Instant::now();
                for _ in 0..REPS {
                    plan.dispatch(&g, &enc, &x, &y, n)?;
                }
                g.submit(enc)?;
                best = best.min(t.elapsed().as_secs_f64() / REPS as f64);
            }
            let tag = match v {
                ConvVariant::DirectFused => "DF".into(),
                ConvVariant::DirectSplit => "DS".into(),
                ConvVariant::Gemm(b) => format!("G{b}"),
                ConvVariant::GemmFused(b) => format!("GF{b}"),
            };
            let bad = if err > 2e-2 { format!(" BAD {err:.1e}") } else { String::new() };
            line += &format!(" {tag} {:.0}us {:.1}T{bad} ", best * 1e6, flops / best / 1e12);
        }
        println!("{line}");
    }
    Ok(())
}
