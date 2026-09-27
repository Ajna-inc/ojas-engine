//! GPU-time attribution for one real Qwen4Exp resident scalar token.
use anyhow::Result;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: flash_resident_profile <gguf>");
    ojas_core::logging::init();
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut gguf = ojas_formats::gguf::Gguf::open(&path)?;
    let model = ojas_models::decoder::DecoderGpu::load(&gpu, &mut gguf, 2048, 4, None, None)?;
    for pos in 0..64 {
        model.forward_id(if pos == 0 { 1 } else { 100 }, pos);
    }
    let before = model.gpu_seconds();
    let start = std::time::Instant::now();
    for pos in 64..96 {
        model.forward_id(100, pos);
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1e3 / 32.0;
    let gpu_ms = (model.gpu_seconds() - before) * 1e3 / 32.0;
    println!("production wall={wall_ms:.3} ms/token gpu={gpu_ms:.3} ms/token");
    let rows = model.profile_qwen4exp_resident_token(100, 96);
    let total: f64 = rows.iter().map(|(_, ms)| ms).sum();
    for (name, ms) in rows {
        println!("{name:>15}: {ms:8.3} ms  {:5.1}%", ms / total * 100.0);
    }
    println!("split GPU sum: {total:.3} ms");
    Ok(())
}
