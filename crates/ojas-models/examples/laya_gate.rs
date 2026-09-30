//! Gate: LAYA — `ojas_models::laya::Laya` must reproduce the upstream PyTorch model.
//!
//! The reference record comes from `scripts/release/laya_reference.py`, which runs the
//! checkpoint's own `rl_agent_api.py` in f32 on the CPU over `scripts/release/laya_cases.json`.
//! Every question is checked at three levels, so a failure localizes:
//!
//! 1. token ids and option-marker positions, exactly: the tokenizer, the JSON state
//!    serialization, the question rendering and the truncation budget all feed this;
//! 2. the raw scorer logits, reported as the largest absolute difference;
//! 3. the calibrated probabilities (max abs difference <= 0.01), the argmax wherever the
//!    reference's top two differ by at least 0.02, and the act probability (<= 0.01).
//!
//! The Metal path rounds activations to f16 inside every GEMM and reads attention K/V as
//! f16, which the f32 reference does not, so bit equality is not expected past step 1.
//! The thresholds are fixed here and are not relaxed to obtain a pass.
//!
//! usage: laya_gate <laya.gguf> <laya_cases.json> <reference.json>

use anyhow::{bail, ensure, Context, Result};
use ojas_models::laya::{json::Json, Laya, Question};

const MAX_PROB_DIFF: f64 = 0.01;
const MAX_ACT_DIFF: f64 = 0.01;
const ARGMAX_MARGIN: f64 = 0.02;

fn floats(v: Option<&Json>) -> Result<Vec<f64>> {
    v.and_then(Json::as_array).context("expected an array")?
        .iter().map(|x| x.as_f64().context("expected a number")).collect()
}

fn argmax(v: &[f64]) -> usize {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 4, "usage: laya_gate <laya.gguf> <laya_cases.json> <reference.json>");
    let cases = Json::parse(&std::fs::read_to_string(&args[2])?)?;
    let reference = Json::parse(&std::fs::read_to_string(&args[3])?)?;
    let gpu = ojas_metal::MetalGpu::new()?;
    let t_load = std::time::Instant::now();
    let laya = Laya::load(&gpu, &args[1])?;
    println!("  model      {}  (loaded in {:.2} s)", args[1], t_load.elapsed().as_secs_f64());
    println!("  reference  {}", reference.get("reference").and_then(Json::as_str).unwrap_or("?"));

    let ref_cases = reference.get("cases").and_then(Json::as_array).context("reference has no cases")?;
    let mut failures: Vec<String> = Vec::new();
    let (mut worst_p, mut worst_logit, mut worst_act) = (0f64, 0f64, 0f64);
    for (case, rc) in cases.as_array().context("cases must be a list")?.iter().zip(ref_cases) {
        let label = case.get("label").and_then(Json::as_str).unwrap_or("?");
        let state = case.get("state").context("case without state")?;
        let questions: Vec<Question> = case.get("questions").and_then(Json::as_object).context("case without questions")?
            .iter().map(|(id, q)| Question::from_json(id, q)).collect::<Result<_>>()?;
        let seqs = laya.sequences(state, &questions)?;
        let t = std::time::Instant::now();
        let decision = laya.decide(state, &questions)?;
        let wall = t.elapsed().as_secs_f64() * 1e3;
        println!("\n  ---- {label}: {} questions, {} tokens, {wall:.1} ms wall / {:.1} ms gpu",
            questions.len(), decision.input_tokens, decision.gpu_s * 1e3);

        let ref_answers = rc.get("answers").and_then(Json::as_array).context("reference case without answers")?;
        ensure!(ref_answers.len() == questions.len(), "{label}: reference has {} answers", ref_answers.len());
        for ((ans, (ids, markers)), ra) in decision.answers.iter().zip(&seqs).zip(ref_answers) {
            let id = format!("{label}/{}", ans.id);
            let ref_ids: Vec<u32> = floats(ra.get("ids"))?.into_iter().map(|x| x as u32).collect();
            let ref_markers: Vec<usize> = floats(ra.get("markers"))?.into_iter().map(|x| x as usize).collect();
            if *ids != ref_ids {
                let at = ids.iter().zip(&ref_ids).position(|(a, b)| a != b).unwrap_or(ids.len().min(ref_ids.len()));
                failures.push(format!("{id}: token ids differ at position {at} (ojas {} tokens, reference {})",
                    ids.len(), ref_ids.len()));
                continue;
            }
            if *markers != ref_markers { failures.push(format!("{id}: marker positions differ")); continue; }

            let ref_p = floats(ra.get("probabilities"))?;
            let ref_logits = floats(ra.get("logits"))?;
            let got_p: Vec<f64> = ans.probabilities.iter().map(|&p| p as f64).collect();
            let dp = got_p.iter().zip(&ref_p).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            let dz = ans.logits.iter().zip(&ref_logits).map(|(&a, b)| (a as f64 - b).abs()).fold(0.0, f64::max);
            let da = (ans.act_probability as f64 - ra.get("act_probability").and_then(Json::as_f64).unwrap_or(f64::NAN)).abs();
            let mut sorted = ref_p.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let margin = sorted[0] - sorted.get(1).copied().unwrap_or(0.0);
            let argmax_ok = margin < ARGMAX_MARGIN || argmax(&got_p) == argmax(&ref_p);
            worst_p = worst_p.max(dp);
            worst_logit = worst_logit.max(dz);
            worst_act = worst_act.max(da);
            println!("    {:<20} ids ok ({:>3})  p max|d| {dp:.2e}  logit max|d| {dz:.2e}  act |d| {da:.2e}  top {:<14} {}",
                ans.id, ids.len(), ans.labels[argmax(&got_p)], if argmax_ok { "" } else { "ARGMAX DIFFERS" });
            if dp > MAX_PROB_DIFF { failures.push(format!("{id}: probability differs by {dp:.4} > {MAX_PROB_DIFF}")); }
            if da > MAX_ACT_DIFF || da.is_nan() { failures.push(format!("{id}: act probability differs by {da:.4}")); }
            if !argmax_ok { failures.push(format!("{id}: argmax differs with reference margin {margin:.4}")); }
        }
    }
    println!("\n  worst: probability {worst_p:.2e}, logit {worst_logit:.2e}, act {worst_act:.2e}");
    if !failures.is_empty() {
        for f in &failures { println!("  FAIL {f}"); }
        bail!("GATE: LAYA FAIL ({} checks)", failures.len());
    }
    println!("\nGATE: LAYA PASS (ids exact; p <= {MAX_PROB_DIFF}; act <= {MAX_ACT_DIFF}; argmax where margin >= {ARGMAX_MARGIN})");
    Ok(())
}
