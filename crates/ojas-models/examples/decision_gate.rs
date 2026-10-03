//! Gate: DECISION — a decision model must reproduce a reference response.
//!
//! The reference is the response another implementation returned for the same model
//! file and request (`{"model", "answers", "usage"}`); the checks are
//! [`ojas_models::decision::parity`]'s.
//!
//! usage: decision_gate <model.gguf> <request.json> <reference.json> [tolerance]

use anyhow::{bail, ensure, Result};
use ojas_models::decision::json::Json;
use ojas_models::decision::parity::{compare, CHOICE_MARGIN, MAX_PROBABILITY_DIFF};
use ojas_models::decision::{DecisionModel, QuestionKind};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 4 || args.len() == 5, "usage: decision_gate <model.gguf> <request.json> <reference.json> [tolerance]");
    let tolerance: f64 = args.get(4).map_or(Ok(MAX_PROBABILITY_DIFF), |t| t.parse())?;
    let body = Json::parse(&std::fs::read_to_string(&args[2])?)?;
    let reference = Json::parse(&std::fs::read_to_string(&args[3])?)?;
    let gpu = ojas_metal::MetalGpu::new()?;
    let t_load = std::time::Instant::now();
    let model = DecisionModel::load(&gpu, &args[1])?;
    println!("  model      {}  (loaded in {:.2} s)", args[1], t_load.elapsed().as_secs_f64());
    let req = model.request(&body)?;
    for _ in 0..2 { model.decide(&req)?; }
    let t = std::time::Instant::now();
    let decision = model.decide(&req)?;
    println!("  {} questions, {} tokens, {:.1} ms wall / {:.1} ms gpu",
        req.questions.len(), decision.input_tokens, t.elapsed().as_secs_f64() * 1e3, decision.gpu_s * 1e3);

    let parity = compare(&decision, &reference, tolerance)?;
    for (a, p) in decision.answers.iter().zip(&parity.answers) {
        let summary = match a.kind {
            QuestionKind::Choice => format!("choice {}", a.choice()),
            QuestionKind::Score => format!("score {:.4}", a.score()),
            QuestionKind::Noul => format!("noul {:.4}", a.noul()),
        };
        println!("    {:<20} {summary:<28} max|d| {:.2e}", a.id, p.max_diff);
    }
    println!("  worst probability difference {:.2e}", parity.worst());
    if !parity.passed() {
        for f in &parity.failures { println!("  FAIL {f}"); }
        bail!("GATE: DECISION FAIL ({} checks)", parity.failures.len());
    }
    println!("GATE: DECISION PASS (tokens exact; p <= {tolerance}; choice where margin >= {CHOICE_MARGIN})");
    Ok(())
}
