//! Forward time of one model on a GPU backend, per batch, and where it goes
//! (per op kind, synced after every step). `exec_profile <model.onnx> <cuda|vulkan[:idx]> [batches] [width]`
//! e.g. `exec_profile yolo11n-dyn.onnx vulkan:0 1,8`
use std::time::Instant;

use anyhow::Result;
use ojas_vision::exec_gpu::GpuExecutor;
use ojas_vision::gpu::GpuDev;

fn bench<G: GpuDev>(g: &ojas_vision::ir::Graph, ordinal: usize) -> Result<()> {
    let mut ex = GpuExecutor::<G>::new(g, ordinal)?;
    for _ in 0..3 {
        ex.forward_device()?;
    }
    let mut ts = vec![];
    for _ in 0..10 {
        let t = Instant::now();
        ex.forward_device()?;
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.total_cmp(b));
    ex.profile = true;
    for _ in 0..3 {
        ex.forward_device()?;
    }
    let total: f64 = ex.op_times.values().map(|v| v.0).sum();
    let mut ops: Vec<_> = ex.op_times.iter().map(|(k, v)| (*k, v.0 / 3.0 * 1e3, v.1 / 3)).collect();
    ops.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("  {} batch {}: forward median {:.2} ms (min {:.2}); per step synced {:.1} ms:", G::BACKEND, ex.batch(), ts[5], ts[0], total / 3.0 * 1e3);
    for (k, ms, n) in ops.iter().take(std::env::var("OJAS_PROFILE_TOP").ok().and_then(|v| v.parse().ok()).unwrap_or(8)) {
        println!("    {k:48} {ms:7.2} ms  x{n}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let model = ojas_formats::onnx::load(&a[1])?;
    let (kind, idx) = a[2].split_once(':').unwrap_or((a[2].as_str(), "0"));
    let idx: usize = idx.parse()?;
    let batches: Vec<usize> = a.get(3).map(|s| s.split(',').map(|b| b.parse().unwrap()).collect()).unwrap_or(vec![1]);
    let width: Option<usize> = a.get(4).map(|w| w.parse().unwrap());
    for b in batches {
        let mut binds = std::collections::HashMap::from([("batch".to_string(), b)]);
        if let Some(w) = width {
            for vi in &model.graph.inputs {
                if let Some(ojas_formats::onnx::OnnxDim::Param(p)) = vi.dims.get(3) {
                    binds.insert(p.clone(), w);
                }
            }
        }
        let mut g = ojas_vision::import(&model, &binds)?;
        ojas_vision::passes::optimize(&mut g);
        ojas_vision::passes::lower_for_gpu(&mut g);
        match kind {
            #[cfg(feature = "cuda")]
            "cuda" => bench::<ojas_cuda::CudaGpu>(&g, idx)?,
            #[cfg(feature = "vulkan")]
            "vulkan" => bench::<ojas_vulkan::VkGpu>(&g, idx)?,
            other => anyhow::bail!("backend {other} not in this build"),
        }
    }
    Ok(())
}
