//! Finds where the GPU executor first disagrees with the CPU oracle, by binary search
//! over the optimized graph's nodes: each probe re-imports the model, makes one node's
//! output the graph output, runs it on both executors from the same input, and
//! compares. Prints the first diverging node, its op, and the ops feeding it.
//!
//! `graph_bisect model.onnx input.f32 [rel_tol 0.05] [dim=value ...]` (needs `--features cuda`)
use std::collections::HashMap;

fn read_f32(path: &str) -> anyhow::Result<Vec<f32>> {
    let b = std::fs::read(path)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn graph(model: &ojas_formats::onnx::OnnxModel) -> anyhow::Result<ojas_vision::ir::Graph> {
    // symbolic dims from the command line (`batch_size=1 height=224 …`); `batch` defaults to 1
    let mut binds: HashMap<String, usize> = std::env::args().skip(4).filter_map(|kv| kv.split_once('=').and_then(|(k, v)| v.parse().ok().map(|v| (k.to_string(), v)))).collect();
    binds.entry("batch".to_string()).or_insert(1);
    let mut g = ojas_vision::import(model, &binds)?;
    ojas_vision::passes::optimize(&mut g);
    Ok(g)
}

/// (relative max |Δ|, max |cpu|, numel) of node `k`'s first output, CPU vs CUDA.
fn probe(model: &ojas_formats::onnx::OnnxModel, input: &[f32], k: usize) -> anyhow::Result<(f32, f32, usize)> {
    let mut gc = graph(model)?;
    let t = gc.nodes[k].outputs[0];
    gc.outputs = vec![t];
    let cpu = ojas_vision::exec_cpu::CpuExecutor::new(&gc, 8).run(&gc, &[input])?.remove(0);
    let mut gg = graph(model)?;
    gg.outputs = vec![t];
    ojas_vision::passes::lower_for_gpu(&mut gg);
    let mut ex = ojas_vision::exec_gpu::CudaExecutor::new(&gg, 0)?;
    ex.run(input)?;
    let gpu = ex.run(input)?.remove(0);
    anyhow::ensure!(cpu.len() == gpu.len(), "node {k}: cpu {} values, gpu {}", cpu.len(), gpu.len());
    let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    Ok((d / scale.max(1e-6), scale, cpu.len()))
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 3, "graph_bisect model.onnx input.f32 [rel_tol]");
    let tol: f32 = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.05);
    let model = ojas_formats::onnx::load(&a[1])?;
    let input = read_f32(&a[2])?;
    let g = graph(&model)?;
    let n = g.nodes.len();
    println!("{n} nodes after optimisation; tolerance {tol} (relative to the tensor's max |value|)");
    let (last_r, ..) = probe(&model, &input, n - 1)?;
    println!("last node: rel Δ {last_r:.4}");
    if last_r <= tol {
        println!("no divergence");
        return Ok(());
    }
    // first k with a mismatch, assuming mismatches persist downstream
    let (mut lo, mut hi) = (0usize, n - 1);
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (r, scale, numel) = probe(&model, &input, mid)?;
        println!("  node {mid:>4} {:<28} rel Δ {r:.4}  (max |cpu| {scale:.3}, {numel} values)", format!("{:?}", g.nodes[mid].op).chars().take(28).collect::<String>());
        if r > tol {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let nd = &g.nodes[lo];
    println!("\nfirst diverging node {lo}: {} {:?}", nd.name, nd.op);
    println!("  output shape {:?}", g.shape(nd.outputs[0]));
    for &i in &nd.inputs {
        let producer = g.nodes.iter().position(|m| m.outputs.contains(&i));
        println!("  input {i} shape {:?} <- {}", g.shape(i), producer.map(|p| format!("node {p} {:?}", g.nodes[p].op)).unwrap_or_else(|| if g.weights[i].is_some() { "weight".into() } else { "graph input".into() }));
    }
    Ok(())
}
