//! Gate: speculative decoding must emit exactly the tokens plain greedy decoding
//! does. A drafted token is kept only where it equals the model's own argmax, so a
//! mismatch means the accept test or the KV bookkeeping is wrong.
//!
//! The prompt deliberately contains repetition: the drafter can only propose text
//! it has already seen, so a novel prompt would exercise only the fallback path.
//!
//! usage: spec_gate <gguf> [n_tokens]

use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: spec_gate <gguf> [n]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(96);
    let prec: u8 = std::env::var("OJAS_PREC").ok().and_then(|v| v.parse().ok()).unwrap_or(3);

    // A prompt with real internal repetition, so lookup actually fires.
    let unit: Vec<u32> = vec![785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13];
    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..6 { prompt.extend_from_slice(&unit); }

    let run = |spec: bool| -> Result<Vec<u32>> {
        // Fresh process state per run: same model, same prompt, only the flag differs.
        let gpu = ojas_metal::MetalGpu::new()?;
        let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
        let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;
        m.reset_state();
        let mut e = ojas_infer::EngineCore::new(m);
        e.eos = None;
        let _ = spec;
        Ok(e.generate(&prompt, n, true))
    };

    // The two paths are selected by OJAS_NO_SPEC, read once per process, so this
    // gate runs one of them and compares against the other via a child process.
    if std::env::var("OJAS_SPEC_GATE_CHILD").is_ok() {
        let toks = run(false)?;
        println!("{}", toks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
        return Ok(());
    }

    let with_spec = run(true)?;
    let exe = std::env::current_exe()?;
    let out = std::process::Command::new(exe)
        .arg(&gguf).arg(n.to_string())
        .env("OJAS_SPEC_GATE_CHILD", "1")
        .env("OJAS_NO_SPEC", "1")
        .env("OJAS_PREC", prec.to_string())
        .output()?;
    let txt = String::from_utf8_lossy(&out.stdout);
    let without: Vec<u32> = txt.trim().split(',').filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok()).collect();

    println!("spec   : {:?}", &with_spec[..with_spec.len().min(24)]);
    println!("nospec : {:?}", &without[..without.len().min(24)]);
    if with_spec.is_empty() || without.is_empty() {
        println!("GATE: SPEC INCONCLUSIVE (empty output)");
        std::process::exit(1);
    }
    if with_spec == without {
        println!("GATE: SPEC EXACT PASS ({} tokens)", with_spec.len());
        Ok(())
    } else {
        let d = with_spec.iter().zip(&without).position(|(a, b)| a != b);
        println!("GATE: SPEC MISMATCH FAIL (first diff at {d:?})");
        std::process::exit(1);
    }
}
