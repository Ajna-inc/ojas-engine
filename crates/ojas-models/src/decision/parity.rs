//! Comparing a decision with a reference response for the same model and request.
//!
//! Three checks, so a failure localizes:
//!
//! 1. `usage.input_tokens`, exactly: the template rendering, the tokenizer, the head
//!    budget and the image layout all feed it;
//! 2. every probability, and a noul's probability, within a tolerance:
//!    [`MAX_PROBABILITY_DIFF`] unless the model's own numerics call for more;
//! 3. a choice's pick wherever the reference's top two differ by at least
//!    [`CHOICE_MARGIN`].
//!
//! Two implementations round differently on the GPU, so bit equality is not expected
//! past the first check.

use super::json::Json;
use super::{Decision, QuestionKind};
use anyhow::{Context, Result};

/// Largest difference allowed in any probability, by default. A larger quantized
/// model can be more sensitive than this to rounding alone: the tolerance for it is
/// set from the spread between two runs of the reference itself (its CPU and GPU
/// backends) on the same request.
pub const MAX_PROBABILITY_DIFF: f64 = 0.01;
/// Reference margin between the top two options above which a choice must agree.
pub const CHOICE_MARGIN: f64 = 0.02;

/// How one answer compares.
#[derive(Clone, Debug)]
pub struct AnswerParity {
    pub id: String,
    /// The largest probability difference; infinite when a value is not finite.
    pub max_diff: f64,
}

/// How a decision compares with a reference response.
#[derive(Clone, Debug)]
pub struct Parity {
    pub answers: Vec<AnswerParity>,
    /// Every check that failed, described.
    pub failures: Vec<String>,
}

impl Parity {
    pub fn passed(&self) -> bool { self.failures.is_empty() }

    /// The largest probability difference over every answer.
    pub fn worst(&self) -> f64 { self.answers.iter().map(|a| a.max_diff).fold(0.0, f64::max) }
}

/// Compare `decision` with `reference`, a `{"answers", "usage"}` response, allowing
/// probabilities to differ by up to `tolerance`.
pub fn compare(decision: &Decision, reference: &Json, tolerance: f64) -> Result<Parity> {
    let mut failures = Vec::new();
    let ref_tokens = reference.get("usage").and_then(|u| u.get("input_tokens")).and_then(Json::as_f64)
        .context("the reference has no usage.input_tokens")?;
    if decision.input_tokens as f64 != ref_tokens {
        failures.push(format!("input tokens: {}, reference {ref_tokens}", decision.input_tokens));
    }
    let ref_answers = reference.get("answers").context("the reference has no answers")?;
    let mut answers = Vec::with_capacity(decision.answers.len());
    for a in &decision.answers {
        let ra = ref_answers.get(&a.id).with_context(|| format!("the reference has no answer {:?}", a.id))?;
        let (got, want): (Vec<f64>, Vec<f64>) = if a.kind == QuestionKind::Noul {
            (vec![a.noul()], vec![ra.get("noul").and_then(Json::as_f64).with_context(|| format!("{}: no reference noul", a.id))?])
        } else {
            let rp = ra.get("probabilities").with_context(|| format!("{}: no reference probabilities", a.id))?;
            let want = a.keys.iter()
                .map(|k| rp.get(k).and_then(Json::as_f64).with_context(|| format!("{}: no reference probability for {k:?}", a.id)))
                .collect::<Result<Vec<_>>>()?;
            (a.probabilities.clone(), want)
        };
        // A value that is not finite is a failure, never a small difference.
        let max_diff = got.iter().zip(&want).map(|(x, y)| (x - y).abs())
            .fold(0.0, |m: f64, d| if d.is_nan() { f64::INFINITY } else { m.max(d) });
        if max_diff > tolerance {
            failures.push(format!("{}: a probability differs by {max_diff:.4}", a.id));
        }
        if a.kind == QuestionKind::Choice {
            let mut sorted = want.clone();
            sorted.sort_by(|x, y| y.total_cmp(x));
            let margin = sorted[0] - sorted.get(1).copied().unwrap_or(0.0);
            if margin >= CHOICE_MARGIN && ra.get("choice").and_then(Json::as_str) != Some(a.choice()) {
                failures.push(format!("{}: the choice differs where the reference leads by {margin:.4}", a.id));
            }
        }
        answers.push(AnswerParity { id: a.id.clone(), max_diff });
    }
    Ok(Parity { answers, failures })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::Answer;

    fn decision(p: &[f64], tokens: usize) -> Decision {
        Decision {
            answers: vec![Answer {
                id: "route".into(), kind: QuestionKind::Choice, keys: vec!["a".into(), "b".into()],
                descriptions: vec![Json::Null; 2], probabilities: p.to_vec(),
            }],
            input_tokens: tokens, cached_tokens: 0, gpu_s: 0.0,
        }
    }

    fn reference() -> Json {
        Json::parse(r#"{"answers": {"route": {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}}},
                        "usage": {"input_tokens": 10, "output_tokens": 0}}"#).unwrap()
    }

    #[test]
    fn close_answers_with_equal_tokens_pass() {
        let p = compare(&decision(&[0.705, 0.295], 10), &reference(), MAX_PROBABILITY_DIFF).unwrap();
        assert!(p.passed(), "{:?}", p.failures);
        assert!((p.worst() - 0.005).abs() < 1e-9);
    }

    #[test]
    fn token_counts_differences_and_a_flipped_choice_each_fail() {
        assert_eq!(compare(&decision(&[0.7, 0.3], 11), &reference(), MAX_PROBABILITY_DIFF).unwrap().failures.len(), 1);
        assert_eq!(compare(&decision(&[0.4, 0.6], 10), &reference(), MAX_PROBABILITY_DIFF).unwrap().failures.len(), 2);
        let nan = compare(&decision(&[f64::NAN, 0.3], 10), &reference(), MAX_PROBABILITY_DIFF).unwrap();
        assert!(!nan.passed() && nan.worst().is_infinite());
    }
}
