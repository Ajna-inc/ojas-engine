//! Per-batch execution time of one graph: upload + forward + output read, with the
//! batch dimension bound at import. It says whether the device is saturated by one
//! item (batching buys nothing) or underfilled (batching buys throughput).
//!
//! It also verifies that the graph really executes the batch it reports. A static
//! batch-1 export cannot be batched, and dividing its constant time by a requested
//! batch manufactures a linear speed-up out of nothing, so this tool refuses such an
//! input rather than printing it. The timing includes host transfers, so it is not a
//! pure kernel number; time transfers separately to attribute the gain.
//! `gpu_forward_batch model-dyn.onnx [reps]`   (`OJAS_DEVICE=cuda:0`)
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() > 1, "gpu_forward_batch model-dyn.onnx [reps]");
    let reps: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(20);
    anyhow::ensure!(reps > 0, "reps must be positive");
    let device = std::env::var("OJAS_DEVICE").unwrap_or_else(|_| "cpu".into());
    // a CUDA request on a build without the feature must fail, not quietly run on the CPU
    #[cfg(not(feature = "cuda"))]
    anyhow::ensure!(!device.starts_with("cuda"), "OJAS_DEVICE={device} but this build has no `cuda` feature");
    anyhow::ensure!(device == "cpu" || device.starts_with("cuda") || device.starts_with("vulkan"), "OJAS_DEVICE={device}: want cpu, cuda[:N] or vulkan[:N]");
    let model = ojas_formats::onnx::parse(&std::fs::read(&a[1])?)?;
    println!("{} on {device}, {reps} reps — time includes upload and output read\n", a[1].rsplit('/').next().unwrap_or(""));
    println!("| batch | ms / call | ms / item | items/s | vs batch 1 |");
    println!("|---:|---:|---:|---:|---:|");
    let mut base = 0.0f64;
    for (bi, batch) in [1usize, 2, 4, 8, 16].iter().enumerate() {
        // bind dim 0 (batch) for every symbolic input dimension
        let mut binds: HashMap<String, usize> = HashMap::new();
        for vi in &model.graph.inputs {
            for (i, d) in vi.dims.iter().enumerate() {
                if let ojas_formats::onnx::OnnxDim::Param(p) = d {
                    anyhow::ensure!(i == 0, "input {}: symbolic dim at position {i} is not the batch", vi.name);
                    binds.insert(p.clone(), *batch);
                }
            }
        }
        let mut g = ojas_vision::import(&model, &binds)?;
        ojas_vision::passes::optimize(&mut g);
        // the graph must actually carry the batch the timing is divided by
        let in_shape = g.shape(g.inputs[0]).to_vec();
        let got = *in_shape.first().unwrap_or(&0);
        anyhow::ensure!(
            got == *batch,
            "{}: input is {in_shape:?} — this export runs batch {got}, not {batch}. \
             A static batch-1 export cannot be batched; dividing its constant time by {batch} \
             would report a {batch}× speed-up that never happened. Export with a dynamic batch \
             dimension (e.g. yolo11n-dyn.onnx).",
            a[1]
        );
        let n: usize = in_shape.iter().product();
        let input = vec![0.5f32; n];
        let med = match device.as_str() {
            #[cfg(feature = "cuda")]
            d if d.starts_with("cuda") => {
                let ordinal = d.split_once(':').and_then(|(_, i)| i.parse().ok()).unwrap_or(0);
                ojas_vision::passes::lower_for_gpu(&mut g);
                let mut ex = ojas_vision::exec_gpu::CudaExecutor::new(&g, ordinal)?;
                ex.run(&input)?; // warm-up: compile kernels, settle buffers
                let mut ts = vec![];
                for _ in 0..reps {
                    let t = std::time::Instant::now();
                    ex.run(&input)?;
                    ts.push(t.elapsed().as_secs_f64() * 1e3);
                }
                ts.sort_by(f64::total_cmp);
                ts[ts.len() / 2]
            }
            _ => {
                let mut ex = ojas_vision::exec_cpu::CpuExecutor::new(&g, std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
                ex.run(&g, &[&input])?;
                let mut ts = vec![];
                for _ in 0..reps {
                    let t = std::time::Instant::now();
                    ex.run(&g, &[&input])?;
                    ts.push(t.elapsed().as_secs_f64() * 1e3);
                }
                ts.sort_by(f64::total_cmp);
                ts[ts.len() / 2]
            }
        };
        let per = med / *batch as f64;
        if bi == 0 {
            base = per;
        }
        println!("| {batch} | {med:.2} | {per:.2} | {:.0} | {:.2}× |", 1000.0 / per, base / per);
    }
    println!("\nFlat per-item time = the device is already full at batch 1; a falling curve = it is not.");
    Ok(())
}
