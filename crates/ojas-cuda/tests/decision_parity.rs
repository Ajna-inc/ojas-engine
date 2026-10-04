//! Decision models on CUDA against the reference responses: the suite in
//! `ojas_decision::testkit`, on `CudaDecision`.
//!
//! ```text
//! OJAS_DECISION_MODELS=~/models/decision cargo test --release -p ojas-cuda \
//!     --test decision_parity -- --ignored --nocapture --test-threads 1
//! ```

use ojas_cuda::CudaDecision;
use ojas_decision::testkit;

#[test]
#[ignore = "needs a CUDA GPU and the decision model files; set OJAS_DECISION_MODELS"]
fn decisions_match_the_reference_responses() {
    let gpu = CudaDecision::new(0).expect("a CUDA device");
    testkit::decisions_match_the_reference_responses(&gpu, &testkit::models_dir());
}

#[test]
#[ignore = "needs a CUDA GPU and the decision model files; set OJAS_DECISION_MODELS"]
fn a_repeated_state_is_continued_from_and_answers_the_same() {
    let gpu = CudaDecision::new(0).expect("a CUDA device");
    testkit::a_repeated_state_is_continued_from_and_answers_the_same(&gpu, &testkit::models_dir());
}
