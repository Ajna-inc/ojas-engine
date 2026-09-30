//! Laya request latency on Metal: one request's first 1, ..., n questions, warm.
//!
//! Every question is one packed sequence, so the prefixes measure how a request scales with
//! its question count. Reported per prefix: tokens, and the median and minimum of wall time
//! (tokenization, sequence building, the GPU forward and the host head) and of GPU time.
//!
//! With `--profile`, also prints the encoder's per-category GPU time at each size
//! (`DecoderGpu::profile_text`).
//!
//! usage: laya_bench <laya.gguf> [request.json] [reps] [--profile]
//!        request.json is `{"state": ..., "questions": {...}}`; without one the example
//!        runs `laya::BENCH_REQUEST`, the 7-question email case.

use anyhow::{ensure, Context, Result};
use ojas_models::laya::{json::Json, Laya, Question, BENCH_REQUEST};

fn stats(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[0])
}

fn main() -> Result<()> {
    let profile = std::env::args().any(|a| a == "--profile");
    let args: Vec<String> = std::env::args().skip(1).filter(|a| a != "--profile").collect();
    ensure!(!args.is_empty(), "usage: laya_bench <laya.gguf> [request.json] [reps] [--profile]");
    // After the model: a number is the repetition count, anything else the request file.
    let reps: usize = args[1..].iter().find_map(|a| a.parse().ok()).unwrap_or(20).max(1);
    let text = match args[1..].iter().find(|a| a.parse::<usize>().is_err()) {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?,
        None => BENCH_REQUEST.to_string(),
    };
    let req = Json::parse(&text)?;
    let state = req.get("state").context("request without state")?;
    let questions: Vec<Question> = req.get("questions").and_then(Json::as_object).context("request without questions")?
        .iter().map(|(id, q)| Question::from_json(id, q)).collect::<Result<_>>()?;
    let gpu = ojas_metal::MetalGpu::new()?;
    let laya = Laya::load(&gpu, &args[0])?;
    let mut sizes: Vec<usize> = vec![1, questions.len().div_ceil(2), questions.len()];
    sizes.dedup();
    println!("{:>3}  {:>6}  {:>16}  {:>16}", "B", "tokens", "wall p50 / min", "gpu p50 / min");
    for b in sizes {
        let qs = &questions[..b];
        for _ in 0..3 { laya.decide(state, qs)?; }
        let (mut wall, mut dev, mut tokens) = (Vec::new(), Vec::new(), 0);
        for _ in 0..reps {
            let t = std::time::Instant::now();
            let d = laya.decide(state, qs)?;
            wall.push(t.elapsed().as_secs_f64() * 1e3);
            dev.push(d.gpu_s * 1e3);
            tokens = d.input_tokens;
        }
        let ((w50, wmin), (g50, gmin)) = (stats(wall), stats(dev));
        println!("{b:>3}  {tokens:>6}  {w50:>7.2} / {wmin:>6.2}  {g50:>7.2} / {gmin:>6.2}  ms");
        if profile {
            for (name, ms) in laya.profile(state, qs)? { println!("       {name:<18} {ms:8.3} ms"); }
        }
    }
    Ok(())
}
