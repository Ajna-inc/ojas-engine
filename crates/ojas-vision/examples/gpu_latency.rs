//! Batch-one deployment-executor timing: gpu_latency model.onnx [iterations] [input.f32].
//! Warm synchronized forward timing excludes decoding/preprocessing/postprocessing.
use anyhow::{ensure, Result};
use std::time::Instant;
use ojas_vision::exec_gpu::GpuExecutor;

fn stats(mut values: Vec<f64>) -> serde_json::Value {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    serde_json::json!({"n":n,"mean_ms":values.iter().sum::<f64>()/n as f64,
        "p50_ms":values[n/2],"p95_ms":values[(n as f64*0.95).ceil() as usize-1],"min_ms":values[0],"max_ms":values[n-1]})
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() >= 2, "gpu_latency model.onnx [iterations] [input.f32]");
    let iterations: usize = args.get(2).map(|v| v.parse()).transpose()?.unwrap_or(100);
    ensure!(iterations >= 20, "at least 20 iterations");
    let start = Instant::now();
    let model = ojas_formats::onnx::load(&args[1])?;
    let mut graph = ojas_vision::import(&model, &std::collections::HashMap::from([("batch".into(), 1)]))?;
    let input_shape = graph.shape(graph.inputs[0]).to_vec();
    ensure!(input_shape[0] == 1, "static model is not batch one");
    let weight_elements: usize = graph.weights.iter().flatten().map(Vec::len).sum();
    ojas_vision::passes::optimize(&mut graph);
    ojas_vision::passes::lower_for_gpu(&mut graph);
    let output_shapes: Vec<_> = graph.outputs.iter().map(|&i| graph.shape(i).to_vec()).collect();
    let mut ex = GpuExecutor::<ojas_cuda::CudaGpu>::new(&graph, 0)?;
    ensure!(ex.batch() == 1, "executor batch is not one");
    let input: Vec<f32> = if let Some(path) = args.get(3) {
        let bytes = std::fs::read(path)?;
        ensure!(bytes.len() % 4 == 0, "invalid f32 input");
        bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    } else { vec![0.5; input_shape.iter().product()] };
    ensure!(input.len() == input_shape.iter().product::<usize>(), "input dimensions mismatch");
    let outputs = ex.run(&input)?;
    ensure!(outputs.iter().flatten().all(|x| x.is_finite()), "nonfinite model output");
    for _ in 0..10 { ex.forward_device()?; }
    let initialization_ms = start.elapsed().as_secs_f64() * 1000.;
    let mut forward = Vec::new();
    for _ in 0..iterations {
        let now = Instant::now();
        ex.forward_device()?;
        forward.push(now.elapsed().as_secs_f64() * 1000.);
    }
    let mut host = Vec::new();
    for _ in 0..iterations {
        let now = Instant::now();
        let _ = ex.run(&input)?;
        host.push(now.elapsed().as_secs_f64() * 1000.);
    }
    println!("{}", serde_json::json!({"model":args[1],"onnx_bytes":std::fs::metadata(&args[1])?.len(),
        "input_shape":input_shape,"output_shapes":output_shapes,"imported_weight_elements":weight_elements,
        "executor_device_bytes":ex.device_bytes(),"executor_plan_bytes":ex.plan_bytes(),
        "initialization_and_warmup_ms":initialization_ms,"warmup":10,"forward":stats(forward),"upload_forward_download":stats(host),
        "scope":"CUDA deployment executor, batch 1, synchronized, warmed; excludes image decode, resize, postprocess and tracking"}));
    Ok(())
}
