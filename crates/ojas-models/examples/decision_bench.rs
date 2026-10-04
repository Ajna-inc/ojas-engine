//! Decision request latency on Metal: one request's first 1, ..., n questions, warm.
//!
//! Reported per prefix: tokens, and the median and minimum of wall time (rendering,
//! tokenization, the GPU pass and the host scorer) and of GPU time.
//!
//! With `--profile`, also prints the encoder's per-category GPU time at each size
//! (`DecoderGpu::profile_text`).
//!
//! usage: decision_bench <model.gguf> [request.json] [reps] [--profile]
//!        request.json is `{"state": ..., "questions": {...}}`; without one the example
//!        runs `decision::BENCH_REQUEST`, the 7-question email case.

use anyhow::{ensure, Context, Result};
use ojas_decision::{json::Json, DecisionModel, Request, BENCH_REQUEST};

fn stats(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[0])
}

fn main() -> Result<()> {
    let profile = std::env::args().any(|a| a == "--profile");
    let args: Vec<String> = std::env::args().skip(1).filter(|a| a != "--profile").collect();
    ensure!(!args.is_empty(), "usage: decision_bench <model.gguf> [request.json] [reps] [--profile]");
    // After the model: a number is the repetition count, anything else the request file.
    let reps: usize = args[1..].iter().find_map(|a| a.parse().ok()).unwrap_or(20).max(1);
    let text = match args[1..].iter().find(|a| a.parse::<usize>().is_err()) {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?,
        None => BENCH_REQUEST.to_string(),
    };
    let metal = ojas_metal::MetalGpu::new()?;
    let gpu = ojas_models::decision_backend::MetalDecision(&metal);
    let model = DecisionModel::load(&gpu, &args[0])?;
    let req = model.request(&Json::parse(&text)?)?;
    let mut sizes: Vec<usize> = vec![1, req.questions.len().div_ceil(2), req.questions.len()];
    sizes.dedup();
    println!("{:>3}  {:>6}  {:>16}  {:>16}", "B", "tokens", "wall p50 / min", "gpu p50 / min");
    for b in sizes {
        let part = Request { state: req.state.clone(), questions: req.questions[..b].to_vec(), images: req.images.clone() };
        for _ in 0..3 { model.decide(&part)?; }
        let (mut wall, mut dev, mut tokens) = (Vec::new(), Vec::new(), 0);
        for _ in 0..reps {
            let t = std::time::Instant::now();
            let d = model.decide(&part)?;
            wall.push(t.elapsed().as_secs_f64() * 1e3);
            dev.push(d.gpu_s * 1e3);
            tokens = d.input_tokens;
        }
        let ((w50, wmin), (g50, gmin)) = (stats(wall), stats(dev));
        println!("{b:>3}  {tokens:>6}  {w50:>7.2} / {wmin:>6.2}  {g50:>7.2} / {gmin:>6.2}  ms");
        if profile {
            for (name, ms) in model.profile_gpu(&part)? { println!("       {name:<18} {ms:8.3} ms"); }
        }
    }
    Ok(())
}
