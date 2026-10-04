//! The trainable ModernBERT encoder and marker head against the reference responses:
//! the decision parity suite of `ojas_decision::testkit`, on a training backend, for
//! the encoder models (tinylaya, Laya, Julia-1).
//!
//! ```text
//! OJAS_DECISION_MODELS=~/models/decision cargo test --release -p ojas-learn \
//!     --features metal --test modern_bert -- --ignored
//! ```

use ojas_decision::testkit;
use ojas_learn::models::modern_bert::LearnDecision;
use std::path::PathBuf;

const ENCODERS: [&str; 3] = ["tinylaya-Q8_0.gguf", "Laya-Q8_0.gguf", "Julia-1-Q8_0.gguf"];

/// A directory holding only the encoder models of `OJAS_DECISION_MODELS`, so the suite
/// skips the causal models a training backend does not load.
fn encoders_only() -> PathBuf {
    let (from, dir) = (testkit::models_dir(), std::env::temp_dir().join(format!("ojas-learn-encoders-{}", std::process::id())));
    std::fs::create_dir_all(&dir).unwrap();
    for name in ENCODERS {
        let (src, dst) = (from.join(name), dir.join(name));
        if src.is_file() && !dst.exists() {
            std::os::unix::fs::symlink(&src, &dst).unwrap();
        }
    }
    dir
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn encoder_decisions_match_the_reference_responses() {
    let be = ojas_learn::metal::Metal::new().expect("a Metal device");
    testkit::decisions_match_the_reference_responses(&LearnDecision(&be), &encoders_only());
}

#[cfg(not(feature = "metal"))]
#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn encoder_decisions_match_the_reference_responses() {
    testkit::decisions_match_the_reference_responses(&LearnDecision(&ojas_learn::cpu::Cpu), &encoders_only());
}

/// Gradients reach the whole model: a few AdamW steps of cross-entropy towards the
/// option tinylaya likes least make it the answer the decision pipeline gives.
#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn training_moves_the_served_answer() {
    use ojas_decision::DecisionModel;
    use ojas_learn::models::modern_bert::MarkerSeq;
    use ojas_learn::{AdamW, Tape};

    let path = testkit::models_dir().join("tinylaya-Q8_0.gguf");
    let be = ojas_learn::cpu::Cpu;
    let gpu = LearnDecision(&be);
    let model = DecisionModel::load(&gpu, path.to_str().unwrap()).unwrap();
    let body = testkit::read_json(&testkit::fixtures().join("requests/ticket.json"));
    let req = model.request(&body).unwrap();
    let route = req.questions.iter().position(|q| q.id == "route").unwrap();
    let before = model.decide(&req).unwrap().answers[route].clone();
    let target = (0..before.probabilities.len())
        .min_by(|&a, &b| before.probabilities[a].total_cmp(&before.probabilities[b])).unwrap();

    let prompt = model.marker_prompts(&req).unwrap().swap_remove(route);
    let seq = MarkerSeq { ids: prompt.ids, qtype: prompt.qtype, markers: prompt.markers };
    let bert = &model.marker_backend().unwrap().model;
    let n = seq.markers.len();
    let mut opt = AdamW { lr: 1e-4, wd: 0.0, ..AdamW::default() };
    for _ in 0..20 {
        let mut t = Tape::new(&be);
        let s = bert.scores(&mut t, std::slice::from_ref(&seq)).unwrap();
        let s = t.scale(s, 1.0 / prompt.temperature).unwrap();
        let s = t.reshape(s, &[1, n]).unwrap();
        let p = t.softmax(s);
        let p = t.slice(p, 1, target, target + 1).unwrap();
        let logp = t.unary(ojas_learn::Unary::Log, p);
        let loss = t.scale(logp, -1.0).unwrap();
        let loss = t.sum(loss);
        t.backward(loss).unwrap();
        opt.begin();
        let mut updated = 0;
        for param in bert.params() {
            if let Some(g) = t.param_var(param).and_then(|v| t.grad(v)) {
                opt.update(&be, param, g, 0.0);
                updated += 1;
            }
        }
        assert_eq!(updated, bert.params().len(), "every tensor of the model receives a gradient");
    }
    let after = model.decide(&req).unwrap().answers[route].clone();
    eprintln!("route: {:?} -> {:?} (target {})", before.probabilities, after.probabilities, before.keys[target]);
    assert_eq!(after.best(), target, "training did not move the answer to {}", before.keys[target]);
}
