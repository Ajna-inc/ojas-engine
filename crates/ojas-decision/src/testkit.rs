//! The decision parity suite, run by each backend's own test: decision models against
//! reference responses, from `tests/decision/` of this crate.
//!
//! `cases.json` names each model file, the projector beside it when its requests
//! carry images, and the requests it answers (`requests/<name>.json`). For each,
//! `references/<model>.<request>.json` is the response the reference implementation
//! returned for the same file and request. Every answer is held to [`crate::parity`],
//! within the case's `tolerance` when it names one: the spread between the
//! reference's own CPU and GPU runs on that model.
//!
//! The model files are not in the repository. A backend's test runs when
//! `OJAS_DECISION_MODELS` names the directory holding them, and skips a case whose
//! model is absent:
//!
//! ```text
//! OJAS_DECISION_MODELS=~/models/decision cargo test --release -p ojas-models \
//!     --test decision_parity -- --ignored          # Metal
//! OJAS_DECISION_MODELS=~/models/decision cargo test --release -p ojas-cuda \
//!     --test decision_parity -- --ignored          # CUDA
//! ```
//!
//! `tinylev` and `tinykev` are derived from `tinyopenjev` by
//! `scripts/decision_test_models.py`.

use crate::json::Json;
use crate::parity::{compare, MAX_PROBABILITY_DIFF};
use crate::{DecisionGpu, DecisionModel};
use std::path::{Path, PathBuf};

/// `tests/decision/` of this crate's checkout.
pub fn fixtures() -> PathBuf { Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/decision") }

/// The directory `OJAS_DECISION_MODELS` names.
pub fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("OJAS_DECISION_MODELS").expect("OJAS_DECISION_MODELS names the model directory"))
}

pub fn read_json(path: &Path) -> Json {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    Json::parse(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}

/// Every case of `cases.json` whose model is in `dir`, against its references. Panics
/// on the first model that fails to load and, at the end, on any failed check.
pub fn decisions_match_the_reference_responses<G: DecisionGpu>(gpu: &G, dir: &Path) {
    let cases = read_json(&fixtures().join("cases.json"));
    let (mut ran, mut failures) = (0, Vec::new());
    for case in cases.as_array().expect("cases.json is a list") {
        let file = case.get("model").and_then(Json::as_str).expect("a case names its model");
        let path = dir.join(file);
        let projector = case.get("projector").and_then(Json::as_str);
        if !path.is_file() || projector.is_some_and(|p| !dir.join(p).is_file()) {
            eprintln!("skipped {file}: not in {}", dir.display());
            continue;
        }
        let model = DecisionModel::load(gpu, path.to_str().unwrap()).unwrap_or_else(|e| panic!("loading {file}: {e:#}"));
        assert_eq!(model.takes_images(), projector.is_some(), "{file}: image support");
        let stem = file.trim_end_matches(".gguf");
        let tolerance = case.get("tolerance").and_then(Json::as_f64).unwrap_or(MAX_PROBABILITY_DIFF);
        for name in case.get("requests").and_then(Json::as_array).expect("a case lists its requests") {
            let name = name.as_str().expect("request names are strings");
            let body = read_json(&fixtures().join(format!("requests/{name}.json")));
            let reference = read_json(&fixtures().join(format!("references/{stem}.{name}.json")));
            let decision = model.request(&body).and_then(|req| model.decide(&req))
                .unwrap_or_else(|e| panic!("{stem}/{name}: {e:#}"));
            let parity = compare(&decision, &reference, tolerance).unwrap_or_else(|e| panic!("{stem}/{name}: {e:#}"));
            eprintln!("{stem}/{name}: {} tokens, worst probability difference {:.2e}", decision.input_tokens, parity.worst());
            failures.extend(parity.failures.iter().map(|f| format!("{stem}/{name}: {f}")));
            ran += 1;
        }
    }
    assert!(ran > 0, "no case ran: none of the models is in {}", dir.display());
    assert!(failures.is_empty(), "{} checks failed:\n{}", failures.len(), failures.join("\n"));
}

/// A causal model continues a repeated state from the prefix it holds, and answers
/// the same as when it computed the prefix.
pub fn a_repeated_state_is_continued_from_and_answers_the_same<G: DecisionGpu>(gpu: &G, dir: &Path) {
    let mut ran = 0;
    for file in ["tinyopenjev-Q8_0.gguf", "tinylev-Q8_0.gguf", "tinykev-Q8_0.gguf"] {
        let path = dir.join(file);
        if !path.is_file() { eprintln!("skipped {file}: not in {}", dir.display()); continue; }
        let model = DecisionModel::load(gpu, path.to_str().unwrap()).unwrap();
        let request = |name: &str| model.request(&read_json(&fixtures().join(format!("requests/{name}.json")))).unwrap();
        let (ticket, nested) = (request("ticket"), request("nested"));
        let first = model.decide(&ticket).unwrap();
        let again = model.decide(&ticket).unwrap();
        let other = model.decide(&nested).unwrap();
        let after_other = model.decide(&ticket).unwrap();
        assert!(again.cached_tokens > first.cached_tokens, "{file}: the repeat reused nothing more");
        assert!(after_other.cached_tokens < again.cached_tokens, "{file}: a different state was continued from");
        assert!(other.answers.len() == nested.questions.len());
        for (a, b) in first.answers.iter().zip(&again.answers).chain(first.answers.iter().zip(&after_other.answers)) {
            for (x, y) in a.probabilities.iter().zip(&b.probabilities) {
                assert!((x - y).abs() < 1e-4, "{file}/{}: {x} then {y}", a.id);
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no model is in {}", dir.display());
}
