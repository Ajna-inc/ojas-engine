//! Per-layer fp16 convolution: ojas `cnn` kernels vs cuDNN (autotuned).
//!
//! 1. survey the model's conv layers → `target/baseline/yolo11n_conv_layers.json`
//! 2. `scripts/release/cudnn_conv_layers.py <layers> > target/baseline/cudnn_layers.out`
//! 3. `cargo run --release -p ojas-cuda --example conv_bench -- <layers> <cudnn.out> [batch..]`
//!
//! Batch 1 is also checked against the CPU oracle (`ojas_cpu::cpu_cnn::conv2d`).

use std::time::Instant;

use ojas_core::{Device, KernelRuntime};
use ojas_cpu::cpu_cnn::{conv2d, Act, ConvShape};
use ojas_cuda::conv::{nchw_to_padded_nhwc, padded_nhwc_to_nchw, ConvGeom, ConvPlan};
use ojas_cuda::{CuBuf, CudaGpu};
use serde_json::Value;

#[derive(Clone, Copy)]
struct Layer { cin: usize, cout: usize, h: usize, w: usize, k: usize, s: usize, p: usize, g: usize, count: usize }

fn tvec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
    }).collect()
}

enum Prepared {
    Direct { plan: ConvPlan, x: CuBuf, y: CuBuf, n: usize, oshape: (usize, usize) },
    Depthwise { x: CuBuf, w: CuBuf, b: CuBuf, y: CuBuf, consts: Vec<u32>, total: usize, oshape: (usize, usize) },
    DwNhwc { x: CuBuf, w: CuBuf, y: CuBuf, consts: Vec<u32>, grid: [u32; 3], n: usize, oshape: (usize, usize) },
}

fn prepare(g: &CudaGpu, l: Layer, n: usize, x: &[f32], w: &[f32]) -> Prepared {
    let (oh, ow) = ((l.h + 2 * l.p - l.k) / l.s + 1, (l.w + 2 * l.p - l.k) / l.s + 1);
    if l.g > 1 {
        let total = n * l.cout * oh * ow;
        if std::env::var("DW_NCHW").is_ok() {
            return Prepared::Depthwise {
                x: g.upload_f16(x), w: g.upload_f16(w), b: g.upload_f16(&[0.0]), y: g.upload_f16(&vec![0.0; total]),
                consts: vec![total as u32, l.cout as u32, 1, l.w as u32, l.h as u32, ow as u32, oh as u32,
                             l.k as u32, l.k as u32, l.s as u32, l.s as u32, l.p as u32, l.p as u32, 1, 1, 0],
                total, oshape: (oh, ow),
            };
        }
        let xp = nchw_to_padded_nhwc(x, n, l.cin, l.h, l.w, l.p, 0, l.cin);
        let wp = l.w + 2 * l.p;
        return Prepared::DwNhwc {
            x: g.upload_f16(&xp), w: g.upload_f16(w), y: g.upload_f16(&vec![0.0; total]),
            consts: vec![n as u32, l.cin as u32, l.h as u32, l.w as u32, l.p as u32, wp as u32, ((l.h + 2 * l.p) * wp * l.cin) as u32,
                         oh as u32, ow as u32, l.k as u32, l.s as u32, l.s as u32, l.p as u32, l.p as u32, 0, ow as u32,
                         (oh * ow * l.cout) as u32, l.cin as u32, l.cout as u32],
            grid: [(n * oh) as u32, (ow * l.cout).div_ceil(256) as u32, 1], n, oshape: (oh, ow),
        };
    }
    // channel storage padding only when phantom columns cannot fix kw*cin (the RGB stem)
    let mut geom = ConvGeom::square(l.cin, l.h, l.w, l.cout, l.k, l.s, l.p);
    let mut cs = l.cin;
    while geom.kw_eff().is_none() {
        cs += 1;
        geom.cin = cs;
    }
    let mut ws = vec![0.0f32; l.cout * cs * l.k * l.k];
    for oc in 0..l.cout {
        for ic in 0..l.cin {
            let (d, s) = ((oc * cs + ic) * l.k * l.k, (oc * l.cin + ic) * l.k * l.k);
            ws[d..d + l.k * l.k].copy_from_slice(&w[s..s + l.k * l.k]);
        }
    }
    let mut plan = ConvPlan::new(g, geom, &ws, None, 0, 0).unwrap();
    if std::env::var("STANDALONE").is_ok() {
        let src: &'static str = Box::leak(format!("#include <cuda_fp16.h>\n{}", ojas_cuda::kernels::cnn::expand(ojas_cuda::kernels::cnn::BODY)).into_boxed_str());
        plan.func = Some(g.pipeline(src, "cnn_conv_f16").unwrap());
    }
    let xp = nchw_to_padded_nhwc(x, n, l.cin, l.h, l.w, l.p, geom.spare_cols(), cs);
    Prepared::Direct { plan, x: g.upload_f16(&xp), y: g.upload_f16(&vec![0.0; n * oh * ow * l.cout]), n, oshape: (oh, ow) }
}

fn run(g: &CudaGpu, pr: &Prepared) {
    let enc = g.begin();
    match pr {
        Prepared::Direct { plan, x, y, n, .. } => plan.dispatch(g, &enc, x, y, *n).unwrap(),
        Prepared::DwNhwc { x, w, y, consts, grid, .. } => {
            g.dispatch(&enc, "cnn_dwconv_nhwc_f16", &[(x, 0), (w, 0), (y, 0)], consts, *grid, [256, 1, 1]).unwrap();
        }
        Prepared::Depthwise { x, w, b, y, consts, total, .. } => {
            g.dispatch(&enc, "cnn_dwconv_f16", &[(x, 0), (w, 0), (b, 0), (y, 0)], consts,
                       [(*total as u32).div_ceil(256), 1, 1], [256, 1, 1]).unwrap();
        }
    }
    g.submit(enc).unwrap();
}

/// Batch-0 output in NCHW f32.
fn output(g: &CudaGpu, pr: &Prepared, l: Layer) -> Vec<f32> {
    match pr {
        Prepared::Direct { y, n, oshape: (oh, ow), .. } => {
            let mut raw = vec![0.0f32; n * oh * ow * l.cout];
            g.read(y, &mut raw);
            padded_nhwc_to_nchw(&raw[..oh * ow * l.cout], 1, l.cout, *oh, *ow, 0)
        }
        Prepared::DwNhwc { y, n, oshape: (oh, ow), .. } => {
            let mut raw = vec![0.0f32; n * oh * ow * l.cout];
            g.read(y, &mut raw);
            padded_nhwc_to_nchw(&raw[..oh * ow * l.cout], 1, l.cout, *oh, *ow, 0)
        }
        Prepared::Depthwise { y, oshape: (oh, ow), .. } => {
            let mut raw = vec![0.0f32; l.cout * oh * ow];
            g.read(y, &mut raw);
            raw
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let layers: Vec<Layer> = serde_json::from_str::<Value>(&std::fs::read_to_string(&args[1])?)?
        .as_array().unwrap().iter().map(|v| {
            let f = |k: &str| v[k].as_u64().unwrap() as usize;
            Layer { cin: f("cin"), cout: f("cout"), h: f("h"), w: f("w"), k: f("k"), s: f("s"), p: f("p"), g: f("g"), count: f("count") }
        }).collect();
    let cudnn_txt = std::fs::read_to_string(&args[2])?;
    let cudnn: Value = serde_json::from_str(cudnn_txt.lines().last().unwrap())?;
    let batches: Vec<usize> = if args.len() > 3 { args[3..].iter().map(|s| s.parse().unwrap()).collect() } else { vec![1, 8, 32] };

    let mut g = CudaGpu::new(0)?;
    g.ensure_family("cnn")?;
    for &n in &batches {
        let (mut ours_total, mut cudnn_total) = (0.0f64, 0.0f64);
        println!("\n== fp16 batch {n}");
        println!("{:>4} {:>4} {:>4}x{:<4} k{} s{} g{:<3} {:>9} {:>9} {:>6} {:>9}", "cin", "cout", "h", "w", "", "", "", "ours ms", "cuDNN ms", "ratio", "check");
        for (li, &l) in layers.iter().enumerate() {
            let j = (l.cin / l.g) * l.k * l.k;
            let x = tvec(n * l.cin * l.h * l.w, li as u64 + 1, 1.0);
            let w = tvec(l.cout * j, li as u64 + 1000, 1.0 / (j as f32).sqrt());
            let pr = prepare(&g, l, n, &x, &w);
            let mut check = String::from("-");
            if n == 1 {
                run(&g, &pr);
                let got = output(&g, &pr, l);
                // oracle on the same f16-rounded inputs the GPU saw
                let r16 = |v: &[f32]| v.iter().map(|&a| half::f16::from_f32(a).to_f32()).collect::<Vec<_>>();
                let s = ConvShape { n: 1, cin: l.cin, h: l.h, w: l.w, cout: l.cout, kh: l.k, kw: l.k, group: l.g,
                                    stride: [l.s; 2], pads: [l.p; 4], dilation: [1, 1] };
                let mut want = vec![0.0f32; got.len()];
                conv2d(&r16(&x), &r16(&w), None, &s, Act::None, 4, &mut want);
                let want16 = r16(&want);
                let err = got.iter().zip(&want16).map(|(a, b)| (a - b).abs() / b.abs().max(1e-2)).fold(0.0f32, f32::max);
                check = if err < 2e-3 { format!("ok {err:.0e}") } else { format!("BAD {err:.1e}") };
            }
            for _ in 0..3 { run(&g, &pr); }
            let mut ts = vec![];
            for _ in 0..15 {
                let t = Instant::now();
                run(&g, &pr);
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ours = ts[ts.len() / 2];
            let cd = cudnn["rows"].as_array().unwrap().iter().find(|r| {
                r["dtype"] == "f16" && r["batch"] == n as u64 && r["cin"] == l.cin as u64 && r["cout"] == l.cout as u64
                    && r["h"] == l.h as u64 && r["w"] == l.w as u64 && r["k"] == l.k as u64 && r["s"] == l.s as u64 && r["g"] == l.g as u64
            }).and_then(|r| r["ms"].as_f64()).unwrap_or(f64::NAN);
            ours_total += ours * l.count as f64;
            cudnn_total += cd * l.count as f64;
            println!("{:>4} {:>4} {:>4}x{:<4} k{} s{} g{:<3} {:>9.3} {:>9.3} {:>6.2} {:>9}", l.cin, l.cout, l.h, l.w, l.k, l.s, l.g, ours, cd, ours / cd, check);
        }
        println!("TOTAL (count-weighted) batch {n}: ours {ours_total:.2} ms  cuDNN {cudnn_total:.2} ms  ratio {:.2}", ours_total / cudnn_total);
    }
    Ok(())
}
