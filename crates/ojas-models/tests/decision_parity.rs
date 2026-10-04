//! Decision models on Metal against the reference responses: the suite in
//! `ojas_decision::testkit`, on a `MetalGpu`.
//!
//! ```text
//! OJAS_DECISION_MODELS=~/models/decision cargo test --release -p ojas-models \
//!     --test decision_parity -- --ignored
//! ```

use ojas_decision::testkit;
use ojas_models::decision_backend::MetalDecision;

#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn decisions_match_the_reference_responses() {
    let gpu = ojas_metal::MetalGpu::new().expect("a Metal GPU");
    testkit::decisions_match_the_reference_responses(&MetalDecision(&gpu), &testkit::models_dir());
}

#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn a_repeated_state_is_continued_from_and_answers_the_same() {
    let gpu = ojas_metal::MetalGpu::new().expect("a Metal GPU");
    testkit::a_repeated_state_is_continued_from_and_answers_the_same(&MetalDecision(&gpu), &testkit::models_dir());
}
