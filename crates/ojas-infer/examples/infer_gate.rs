//! Verify EngineCore::generate is token-identical on a fixed greedy sequence
//! (Qwen3-0.6B-f16: open GGUF → DecoderGpu → generate).
//! usage: infer_gate <gguf>
fn main() -> anyhow::Result<()> {
    let gguf = std::env::args().nth(1).expect("gguf path");
    let prompt: Vec<u32> = vec![151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198];
    let expect: Vec<u32> = vec![
        151667, 198, 32313, 11, 279, 1196, 1101, 1053, 330, 9707, 8958, 773, 358, 1184, 311,
        5889, 34901, 13, 6771, 752, 1744, 13, 5512, 11, 358, 1265, 24645, 862, 42113, 13,
        10696, 1977,
    ];

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 2048, 1, None, None)?;
    let engine = ojas_infer::EngineCore::new(m);
    let out = engine.generate(&prompt, 32, true);

    println!("expect: {expect:?}");
    println!("got:    {out:?}");
    assert_eq!(out, expect, "EngineCore output diverged from decode_gate");
    println!("GATE: ENGINECORE TOKEN-IDENTICAL PASS");
    Ok(())
}
