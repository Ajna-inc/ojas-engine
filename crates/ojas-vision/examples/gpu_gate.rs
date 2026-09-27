//! CUDA executor vs the CPU executor on the same input, then GPU timing.
//!
//! `cargo run --release -p ojas-vision --features cuda --example gpu_gate -- model.onnx [batch] [iters]`
//!
//! Parity: the fp16 GPU graph against the f32 CPU oracle on every graph
//! output (cosine, max |diff| relative to the output scale). Timing: device
//! forward only (input already uploaded) and end to end (upload + forward +
//! readback), median of `iters`.

use std::collections::HashMap;
use std::time::Instant;

use ojas_vision::exec_cpu::CpuExecutor;
use ojas_vision::exec_cuda::CudaExecutor;

fn tvec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 40) as f32 / (1u64 << 24) as f32
    }).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let batch: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1);
    let iters: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(50);
    let model = ojas_formats::onnx::load(&args[1])?;
    let mut binds = HashMap::new();
    binds.insert("batch".to_string(), batch);
    let mut g = ojas_vision::import(&model, &binds)?;
    ojas_vision::passes::optimize(&mut g);
    let in_shape = g.shape(g.inputs[0]).to_vec();
    anyhow::ensure!(in_shape[0] == batch, "model batch is {} (static export?); re-export or pass the matching batch", in_shape[0]);
    let x = tvec(g.tensors[g.inputs[0]].numel(), 7);

    let t = Instant::now();
    let mut gpu = CudaExecutor::new(&g, 0)?;
    println!("plan: {:.1} ms, device buffers {:.1} MB, concat inputs still copied: {}",
             t.elapsed().as_secs_f64() * 1e3, gpu.device_bytes() as f64 / 1e6, gpu.concat_copies);
    let got = gpu.run(&x)?;

    // parity on image 0 only when batched (the CPU oracle is slow)
    let mut cpu = CpuExecutor::new(&g, 8);
    let want = cpu.run(&g, &[&x])?;
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let maxd = a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let nan = a.iter().filter(|v| !v.is_finite()).count();
        println!("output {i} {:?}: cosine {:.7}  max|d| {:.3e} (scale {:.1})  non-finite {nan}",
                 g.shape(g.outputs[i]), dot / (na * nb), maxd, scale);
    }

    for _ in 0..5 { gpu.forward()?; }
    let dev = median((0..iters).map(|_| { let t = Instant::now(); gpu.forward().unwrap(); t.elapsed().as_secs_f64() * 1e3 }).collect());
    let e2e = median((0..iters.min(30)).map(|_| { let t = Instant::now(); gpu.run(&x).unwrap(); t.elapsed().as_secs_f64() * 1e3 }).collect());
    println!("batch {batch}: device forward {dev:.3} ms ({:.1} frames/s) | end to end {e2e:.3} ms", batch as f64 * 1e3 / dev);

    if std::env::var("PROFILE").is_ok() {
        gpu.profile = true;
        for _ in 0..10 { gpu.forward()?; }
        let mut v: Vec<_> = gpu.op_times.iter().map(|(k, (t, n))| (*k, *t / 10.0 * 1e3, *n / 10)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (k, ms, n) in v { println!("  {k:<28} {ms:>8.3} ms  x{n}"); }
    }
    Ok(())
}
