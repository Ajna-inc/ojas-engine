//! Laya: calibrated typed decisions from one encoder pass.
//!
//! A Laya checkpoint (`convaiinnovations/laya`, converted by
//! `scripts/laya_convert.py`) is a ModernBERT encoder plus a small decision head.
//! Given a state (text or JSON) and typed questions, it returns a calibrated
//! distribution over each question's options. There is no generation: every
//! question becomes one sequence ([`sequence`]), all sequences of a request are
//! packed into one GPU forward ([`DecoderGpu::laya_forward`]), and the host
//! finishes the scorer's last projection, the calibration and the act head.
//!
//! Numerics follow the reference `rl_agent_api.py`: logits are divided by the
//! temperature fitted for the question type and option count, confidence is one
//! minus the normalized entropy, and the act head reads the head's `[CLS]` output
//! plus four features of the uncalibrated distribution.

pub mod json;
pub mod sequence;

use crate::decoder::DecoderGpu;
use anyhow::{ensure, Context, Result};
use json::Json;
use ojas_core::math::gelu_erf;
use ojas_formats::gguf::Gguf;
use ojas_metal::MetalGpu;
use ojas_tokenize::Bpe;
use sequence::SequenceSpec;
pub use sequence::{Question, QuestionKind};
use std::collections::HashMap;

/// The model card's email triage, seven questions of all three types, as
/// `{"state": ..., "questions": {...}}` (731 tokens on the English checkpoint): the
/// request `ojas bench` and `examples/laya_bench.rs` run by default.
pub const BENCH_REQUEST: &str = r#"{
  "state": {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan. Our account manager has not responded in a week and this is the third time this has happened."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "other": "everything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": [
        "not urgent",
        "soon",
        "critical deadline or blocking issue"
      ]
    },
    "churn_risk": {
      "type": "noul",
      "instructions": "Does the user threaten to cancel or leave?"
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the user explicitly request a refund?"
    },
    "angry": {
      "type": "noul",
      "instructions": "Is the sender angry?"
    },
    "invoice_number": {
      "type": "noul",
      "instructions": "Does this mention a specific invoice number?"
    },
    "spam": {
      "type": "noul",
      "instructions": "Is this spam or a phishing attempt?"
    }
  }
}"#;

/// One question's answer.
#[derive(Clone, Debug)]
pub struct Answer {
    pub id: String,
    pub kind: QuestionKind,
    /// Option labels, in the order of `probabilities`.
    pub labels: Vec<String>,
    /// Calibrated probability per option.
    pub probabilities: Vec<f32>,
    /// Raw scorer logit per option, before calibration.
    pub logits: Vec<f32>,
    /// `1 - H(p) / ln(k)`, clipped to `[0, 1]`.
    pub confidence: f32,
    /// The act head's probability for its first action, the reference's
    /// `rl_agent.act_probability` (acting on the answer, as opposed to escalating).
    pub act_probability: f32,
}

impl Answer {
    /// The most probable option's label.
    pub fn choice(&self) -> &str {
        let best = self.probabilities.iter().enumerate()
            .fold(0, |b, (i, &p)| if p > self.probabilities[b] { i } else { b });
        &self.labels[best]
    }

    /// Expected level of a score question: `sum(i * p_i)`.
    pub fn expected_level(&self) -> f32 {
        self.probabilities.iter().enumerate().map(|(i, p)| i as f32 * p).sum()
    }

    /// Probability that a noul question's statement holds.
    pub fn p_true(&self) -> f32 { self.probabilities.get(1).copied().unwrap_or(0.0) }
}

/// Answers to one request.
#[derive(Clone, Debug)]
pub struct Decision {
    pub answers: Vec<Answer>,
    /// Tokens across every question's sequence.
    pub input_tokens: usize,
    /// GPU time of the forward pass, seconds.
    pub gpu_s: f64,
}

/// Host-side head weights: the scorer's `d -> 1` projection and the act head.
struct HostHead {
    scorer_w: Vec<f32>,
    scorer_b: f32,
    /// `[n_hidden][d + 4]`, PyTorch `Linear` layout.
    act_fc_w: Vec<f32>,
    act_fc_b: Vec<f32>,
    /// `[n_act][n_hidden]`.
    act_out_w: Vec<f32>,
    act_out_b: Vec<f32>,
}

fn read_f32(g: &mut Gguf, name: &str) -> Result<(Vec<u64>, Vec<f32>)> {
    let (dims, ty, bytes) = g.read_tensor(name)?;
    let v = match ty {
        0 => bytes.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
        1 => bytes.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c).to_f32()).collect(),
        t => anyhow::bail!("{name}: ggml type {t} is not f32 or f16"),
    };
    Ok((dims, v))
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let mx = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = z.iter().map(|&v| ((v - mx) as f64).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|&v| (v / s) as f32).collect()
}

/// `(W x + b)` for a row-major `[n_out][n_in]` weight.
fn linear(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    let n_in = x.len();
    b.iter().enumerate().map(|(o, &bo)| {
        bo + w[o * n_in..(o + 1) * n_in].iter().zip(x).map(|(a, c)| a * c).sum::<f32>()
    }).collect()
}

/// A loaded Laya model.
pub struct Laya<'a> {
    dec: DecoderGpu<'a>,
    tok: Bpe,
    spec: SequenceSpec,
    head: HostHead,
    d: usize,
    temperature: [f32; 3],
    temperature_by_options: HashMap<String, f32>,
}

impl<'a> Laya<'a> {
    /// Load a GGUF written by `scripts/laya_convert.py`.
    pub fn load(gpu: &'a MetalGpu, path: &str) -> Result<Self> {
        let mut g = Gguf::open(path).with_context(|| format!("opening {path}"))?;
        ensure!(g.arch() == "modern-bert" && g.meta_u32("laya.head.block_count").is_some(),
            "{path} is not a Laya model (architecture {:?}, no laya.head.* metadata); \
             convert the checkpoint with scripts/laya_convert.py", g.arch());
        let id = |k: &str| g.meta_u32(&format!("tokenizer.ggml.{k}"))
            .ok_or_else(|| anyhow::anyhow!("{path}: missing tokenizer.ggml.{k}"));
        let spec = SequenceSpec {
            cls: id("bos_token_id")?,
            sep: id("seperator_token_id")?,
            mask: id("mask_token_id")?,
            mask_text: String::new(),
            max_len: g.meta_u32("laya.max_len").unwrap_or(512) as usize,
            head_max_len: g.meta_u32("laya.head_max_len").unwrap_or(192) as usize,
        };
        let temps = g.float_arr("laya.temperature").cloned().unwrap_or_else(|| vec![1.0; 3]);
        ensure!(temps.len() == 3 && temps.iter().all(|&t| t > 0.0), "{path}: laya.temperature must be three positive values");
        let keys = g.str_arr("laya.temperature_by_options.keys").cloned().unwrap_or_default();
        let vals = g.float_arr("laya.temperature_by_options.values").cloned().unwrap_or_default();
        ensure!(keys.len() == vals.len(), "{path}: temperature_by_options keys and values differ in length");
        let tok = Bpe::from_gguf(&g);
        let spec = SequenceSpec { mask_text: tok.decode(spec.mask as usize), ..spec };
        ensure!(!spec.mask_text.is_empty(), "{path}: the mask token has no text");

        let (_, scorer_w) = read_f32(&mut g, "laya.scorer_out.weight")?;
        let (_, scorer_b) = read_f32(&mut g, "laya.scorer_out.bias")?;
        let (fc_dims, act_fc_w) = read_f32(&mut g, "laya.act_fc.weight")?;
        let (_, act_fc_b) = read_f32(&mut g, "laya.act_fc.bias")?;
        let (_, act_out_w) = read_f32(&mut g, "laya.act_out.weight")?;
        let (_, act_out_b) = read_f32(&mut g, "laya.act_out.bias")?;

        // A small context: the encoder does not use the decoder's KV cache.
        let dec = DecoderGpu::load(gpu, &mut g, 64, 4, None, None)?;
        let d = dec.text_encoder_width().ok_or_else(|| anyhow::anyhow!("{path}: not a text encoder"))?;
        ensure!(dec.has_laya_head(), "{path}: the encoder loaded without its Laya head");
        ensure!(scorer_w.len() == d && scorer_b.len() == 1, "laya.scorer_out has the wrong shape");
        ensure!(fc_dims.first() == Some(&(d as u64 + 4)) && act_fc_w.len() == act_fc_b.len() * (d + 4),
            "laya.act_fc.weight is {fc_dims:?}, expected [{}, {}]", d + 4, act_fc_b.len());
        ensure!(act_out_w.len() == act_out_b.len() * act_fc_b.len(), "laya.act_out has the wrong shape");
        let max_len = spec.max_len.min(dec.text_encoder_max_positions().unwrap_or(spec.max_len));
        Ok(Laya {
            dec, tok, spec: SequenceSpec { max_len, ..spec }, d,
            head: HostHead { scorer_w, scorer_b: scorer_b[0], act_fc_w, act_fc_b, act_out_w, act_out_b },
            temperature: [temps[0], temps[1], temps[2]],
            temperature_by_options: keys.into_iter().zip(vals).collect(),
        })
    }

    /// Longest sequence a question may produce.
    pub fn max_len(&self) -> usize { self.spec.max_len }

    /// Build every question's token sequence for `state`.
    pub fn sequences(&self, state: &Json, questions: &[Question]) -> Result<Vec<(Vec<u32>, Vec<usize>)>> {
        let st = sequence::state_ids(&self.tok, &self.spec, state);
        questions.iter().map(|q| sequence::build_sequence(&self.tok, &self.spec, q, &st)).collect()
    }

    /// Per-category GPU time of the encoder on this request's sequences; see
    /// [`DecoderGpu::profile_text`].
    pub fn profile(&self, state: &Json, questions: &[Question]) -> Result<Vec<(String, f64)>> {
        let seqs: Vec<Vec<u32>> = self.sequences(state, questions)?.into_iter().map(|(ids, _)| ids).collect();
        self.dec.profile_text(&seqs)
    }

    /// Answer `questions` about `state` in one forward pass.
    pub fn decide(&self, state: &Json, questions: &[Question]) -> Result<Decision> {
        ensure!(!questions.is_empty(), "no questions to answer");
        let built = self.sequences(state, questions)?;
        let (seqs, markers): (Vec<Vec<u32>>, Vec<Vec<usize>>) = built.into_iter().unzip();
        let qtypes: Vec<u32> = questions.iter().map(|q| q.kind.index()).collect();
        let out = self.dec.laya_forward(&seqs, &qtypes, &markers)?;
        let answers = questions.iter().enumerate().map(|(i, q)| {
            let logits: Vec<f32> = out.scorer_hidden[i].iter().map(|h| {
                self.head.scorer_b + h.iter().zip(&self.head.scorer_w).map(|(a, w)| a * w).sum::<f32>()
            }).collect();
            self.answer(q, &logits, &out.pooled[i])
        }).collect();
        Ok(Decision { answers, input_tokens: seqs.iter().map(Vec::len).sum(), gpu_s: out.gpu_s })
    }

    fn temperature(&self, kind: QuestionKind, k: usize) -> f32 {
        let size = match k { 0..=2 => "2", 3..=5 => "3-5", 6..=10 => "6-10", _ => "11+" };
        self.temperature_by_options.get(&format!("{}:{size}", kind.name())).copied()
            .unwrap_or(self.temperature[kind.index() as usize])
    }

    fn answer(&self, q: &Question, logits: &[f32], pooled: &[f32]) -> Answer {
        let k = logits.len();
        let t = self.temperature(q.kind, k);
        let probabilities = softmax(&logits.iter().map(|z| z / t).collect::<Vec<_>>());
        let entropy = |p: &[f32]| -> f32 { -p.iter().map(|&v| v * v.max(1e-12).ln()).sum::<f32>() };
        let confidence = if k < 2 { 1.0 } else { (1.0 - entropy(&probabilities) / (k as f32).ln()).clamp(0.0, 1.0) };

        // Act head features, from the uncalibrated distribution (reference: the
        // softmax of the detached logits, entropy floored at 1e-9).
        let raw = softmax(logits);
        let mut sorted = raw.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let k_eff = k.max(2) as f32;
        let ent = -raw.iter().map(|&v| v * v.max(1e-9).ln()).sum::<f32>() / k_eff.ln();
        let mut feat = pooled.to_vec();
        feat.extend_from_slice(&[sorted[0], sorted[0] - sorted.get(1).copied().unwrap_or(0.0), ent, k_eff / 255.0]);
        debug_assert_eq!(feat.len(), self.d + 4);
        let hidden: Vec<f32> = linear(&self.head.act_fc_w, &self.head.act_fc_b, &feat).into_iter().map(gelu_erf).collect();
        let act = softmax(&linear(&self.head.act_out_w, &self.head.act_out_b, &hidden));

        Answer {
            id: q.id.clone(), kind: q.kind, labels: q.labels.clone(), probabilities, logits: logits.to_vec(), confidence,
            act_probability: act[0],
        }
    }
}
