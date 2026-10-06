//! Time training steps of an encoder decision model on Metal: forward, backward and an
//! AdamW update over every tensor, on one request's questions.
//!
//! ```text
//! cargo run --release -p ojas-learn --features metal --example marker_step -- \
//!     ~/models/decision/Laya-Q8_0.gguf crates/ojas-decision/tests/decision/requests/ticket.json [steps]
//! ```

use anyhow::{Context, Result};
use ojas_decision::json::Json;
use ojas_decision::DecisionModel;
use ojas_learn::metal::Metal;
use ojas_learn::models::modern_bert::{LearnDecision, MarkerSeq};
use ojas_learn::{AdamW, Tape, Unary};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (model_path, request_path) = (args.get(1).context("model path")?, args.get(2).context("request path")?);
    let steps: usize = args.get(3).map_or(Ok(5), |s| s.parse())?;
    let be = Metal::new()?;
    let gpu = LearnDecision::new(&be);
    let model = DecisionModel::load(&gpu, model_path)?;
    let body = Json::parse(&std::fs::read_to_string(request_path)?).map_err(|e| anyhow::anyhow!("{e}"))?;
    let req = model.request(&body)?;
    let prompts = model.marker_prompts(&req)?;
    let seqs: Vec<MarkerSeq> = prompts.iter()
        .map(|p| MarkerSeq { ids: p.ids.clone(), qtype: p.qtype, markers: p.markers.clone() }).collect();
    let tokens: usize = seqs.iter().map(|s| s.ids.len()).sum();
    let bert = &model.marker_backend().context("not an encoder model")?.model;
    let mut opt = AdamW { lr: 1e-5, wd: 0.0, ..AdamW::default() };
    println!("{}: {} questions, {tokens} tokens, {} tensors", model.name(), seqs.len(), bert.params().len());
    for step in 0..steps {
        let start = Instant::now();
        let mut t = Tape::new(&be);
        let s = bert.scores(&mut t, &seqs)?;
        // Cross-entropy towards each question's first option: a stand-in loss of the
        // shape every training objective here takes.
        let mut first = 0;
        let mut picked = Vec::with_capacity(seqs.len());
        for (seq, p) in seqs.iter().zip(&prompts) {
            let n = seq.markers.len();
            let row = t.slice(s, 0, first, first + n)?;
            let row = t.scale(row, 1.0 / p.temperature)?;
            let row = t.reshape(row, &[1, n])?;
            let probs = t.softmax(row);
            picked.push(t.slice(probs, 1, 0, 1)?);
            first += n;
        }
        let picked = t.concat(&picked, 1)?;
        let logp = t.unary(Unary::Log, picked);
        let loss = t.sum_scaled(logp, -1.0 / seqs.len() as f32);
        let forward = start.elapsed();
        t.backward(loss)?;
        let backward = start.elapsed();
        opt.begin();
        for param in bert.params() {
            if let Some(g) = t.param_var(param).and_then(|v| t.grad(v)) {
                opt.update(&be, param, g, 0.0);
            }
        }
        let loss = t.value(loss)[0];
        let total = start.elapsed();
        println!("step {step}: loss {loss:.4}  forward {:.0} ms  backward {:.0} ms  total {:.0} ms",
                 forward.as_secs_f64() * 1e3, (backward - forward).as_secs_f64() * 1e3, total.as_secs_f64() * 1e3);
    }
    Ok(())
}
