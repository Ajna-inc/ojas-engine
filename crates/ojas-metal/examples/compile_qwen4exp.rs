//! Validates that the qwen4exp hyper-connection kernel family compiles and every
//! pipeline builds on this GPU. Run: cargo run -p ojas-metal --example compile_qwen4exp
fn main() -> anyhow::Result<()> {
    let gpu = ojas_metal::MetalGpu::new()?;
    let src = ojas_metal::kernels::family_source("qwen4exp").expect("qwen4exp family exists");
    let pipes = gpu.compile_all(src, |_| true)?;
    println!("qwen4exp: {} pipelines compiled + built OK", pipes.len());
    for (name, _) in &pipes { println!("  {name}"); }
    Ok(())
}
