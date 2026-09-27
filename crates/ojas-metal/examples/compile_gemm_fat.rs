//! Validates that the gemm_fat family MSL compiles and every pipeline builds on
//! this GPU — catches MSL breakage without needing a model. Run: cargo run -p
//! ojas-metal --example compile_gemm_fat
fn main() -> anyhow::Result<()> {
    let gpu = ojas_metal::MetalGpu::new()?;
    let src = ojas_metal::kernels::gemm_fat::GEMM_FAT_KERNELS;
    let pipes = gpu.compile_all(src, |_| true)?;
    println!("gemm_fat: {} pipelines compiled + built OK", pipes.len());
    for (name, _) in &pipes { println!("  {name}"); }
    Ok(())
}
