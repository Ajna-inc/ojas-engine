//! Device memory per planned executor: `plan_mem <model.onnx> <batch>..`
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let model = ojas_formats::onnx::load(&a[1])?;
    for b in &a[2..] {
        let mut binds = HashMap::new();
        binds.insert("batch".to_string(), b.parse::<usize>()?);
        let mut g = ojas_vision::import(&model, &binds)?;
        ojas_vision::passes::optimize(&mut g);
        let ex = ojas_vision::exec_cuda::CudaExecutor::new(&g, 0)?;
        let mb = |n: usize| n as f64 / (1 << 20) as f64;
        println!("batch {b:>3}: total {:7.1} MB | conv plans {:6.1} MB | buffers {:7.1} MB",
                 mb(ex.device_bytes()), mb(ex.plan_bytes()), mb(ex.device_bytes() - ex.plan_bytes()));
    }
    Ok(())
}
