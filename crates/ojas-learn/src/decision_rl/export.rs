//! Saving a trained encoder as the GGUF the decision server loads: the source file's
//! metadata, the trained tensors, and temperatures refitted to the model's new scores.

use super::ModernBert;
use crate::backend::Backend;
use anyhow::{Context, Result};
use ojas_decision::QuestionKind;
use ojas_formats::gguf::{Gguf, Meta};
use ojas_formats::gguf_write::{write_gguf, TensorOut};
use std::path::Path;

/// Write `model` to `dst` with `src`'s metadata. `temperatures` are
/// `(kind-and-bucket, T)` pairs as [`fit_temperatures`] returns them; only the keys
/// `src` carries are replaced, so a file that never had a bucket's key gets none.
/// Matrices are stored F16, vectors F32.
pub fn export<B: Backend>(be: &B, model: &ModernBert<B>, src: &Path, dst: &Path, temperatures: &[(String, f32)]) -> Result<()> {
    let g = Gguf::open(src.to_str().context("source path")?)?;
    let prefix = format!("{}.decision.temperature.", g.arch());
    let set: Vec<(String, f32)> = temperatures.iter()
        .map(|(k, t)| (format!("{prefix}{k}"), *t))
        .filter(|(k, _)| matches!(g.meta.get(k), Some(Meta::F32(_))))
        .collect();
    let set_refs: Vec<(&str, f32)> = set.iter().map(|(k, t)| (k.as_str(), *t)).collect();
    let values: Vec<Vec<f32>> = model.params().iter().map(|p| be.download(&p.val)).collect();
    let tensors: Vec<TensorOut<'_>> = model.params().iter().zip(&values).map(|(p, data)| TensorOut {
        name: &p.name,
        dims: p.shape.iter().rev().map(|&d| d as u64).collect(),
        data,
        f16: p.shape.len() == 2 && p.shape.iter().all(|&d| d >= 64),
    }).collect();
    write_gguf(src, dst, &set_refs, &tensors)
}

/// One question's raw option scores and the target distribution they are fitted to.
pub struct Scored {
    pub kind: QuestionKind,
    pub scores: Vec<f32>,
    pub target: Vec<f64>,
}

/// The option-count buckets of an encoder model's temperatures: `(most options, name)`.
const BUCKETS: &[(usize, &str)] = &[(2, "2"), (5, "3_5"), (10, "6_10"), (usize::MAX, "11")];

/// Temperatures that make the softmax of the scores closest (in cross-entropy) to the
/// targets: one per question kind over all its questions, and one per kind and
/// option-count bucket over the questions in it, named as the metadata keys name them
/// (`choice`, `choice.3_5`, …). Kinds and buckets with no questions get nothing.
pub fn fit_temperatures(questions: &[Scored]) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    for kind in [QuestionKind::Choice, QuestionKind::Score, QuestionKind::Noul] {
        let of_kind: Vec<&Scored> = questions.iter().filter(|q| q.kind == kind).collect();
        if of_kind.is_empty() { continue; }
        out.push((kind.name().to_string(), best_temperature(&of_kind)));
        for (most, bucket) in BUCKETS {
            let in_bucket: Vec<&Scored> = of_kind.iter().copied().filter(|q| q.scores.len() <= *most
                && BUCKETS.iter().find(|(m, _)| q.scores.len() <= *m).map(|(_, b)| b) == Some(bucket)).collect();
            if !in_bucket.is_empty() {
                out.push((format!("{}.{bucket}", kind.name()), best_temperature(&in_bucket)));
            }
        }
    }
    out
}

/// Mean cross-entropy of the targets against the scores softmaxed at `t`.
fn cross_entropy(questions: &[&Scored], t: f64) -> f64 {
    let mut total = 0.0;
    for q in questions {
        let mx = q.scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let e: Vec<f64> = q.scores.iter().map(|&s| ((s as f64 - mx) / t).exp()).collect();
        let z: f64 = e.iter().sum();
        total -= q.target.iter().zip(&e).map(|(p, ei)| p * (ei / z).max(1e-12).ln()).sum::<f64>();
    }
    total / questions.len() as f64
}

/// Golden-section search over `log T` in `[0.05, 20]`.
fn best_temperature(questions: &[&Scored]) -> f32 {
    let (mut lo, mut hi) = (0.05f64.ln(), 20f64.ln());
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (hi - phi * (hi - lo), lo + phi * (hi - lo));
    let (mut fa, mut fb) = (cross_entropy(questions, a.exp()), cross_entropy(questions, b.exp()));
    for _ in 0..60 {
        if fa < fb {
            hi = b; b = a; fb = fa;
            a = hi - phi * (hi - lo); fa = cross_entropy(questions, a.exp());
        } else {
            lo = a; a = b; fa = fb;
            b = lo + phi * (hi - lo); fb = cross_entropy(questions, b.exp());
        }
    }
    ((lo + hi) / 2.0).exp() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fitted_temperature_recovers_the_one_the_targets_were_made_with() {
        let mut rng = crate::decision_rl::tasks::Rng::new(11);
        let made_with = 2.5f64;
        let questions: Vec<Scored> = (0..200).map(|_| {
            let n = 2 + rng.below(4);
            let scores: Vec<f32> = (0..n).map(|_| (rng.below(1000) as f32 / 100.0) - 5.0).collect();
            let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let e: Vec<f64> = scores.iter().map(|&s| ((s as f64 - mx) / made_with).exp()).collect();
            let z: f64 = e.iter().sum();
            Scored { kind: QuestionKind::Choice, scores, target: e.iter().map(|v| v / z).collect() }
        }).collect();
        let fitted = fit_temperatures(&questions);
        let (_, t) = fitted.iter().find(|(k, _)| k == "choice").unwrap();
        assert!((*t as f64 - made_with).abs() < 0.05, "fitted {t}, made with {made_with}");
        assert!(fitted.iter().any(|(k, _)| k == "choice.2") && fitted.iter().any(|(k, _)| k == "choice.3_5"));
    }
}
