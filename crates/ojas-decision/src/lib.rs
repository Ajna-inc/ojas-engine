//! Decision models: calibrated answers to typed questions about a state.
//!
//! A request is `{"state": ..., "questions": {id: question}}`, where a question is a
//! `choice` among named options, a `score` on an ordered scale, or a `noul` (the
//! probability that a statement holds). Nothing is generated. Each question's
//! prompt is rendered from the template the model file ships, and the model's
//! scores for the options are read in one of a few ways, chosen by the
//! `decision.type` the file names, through one table (`PROFILES`):
//!
//! - markers: an encoder with a head scored at a marker token before each option
//!   (`marker`); every question of a request runs in one GPU pass.
//! - labels and pointer: a causal language model, scored by the next-token logits
//!   of labels that name the options, or by projected hidden states at the token
//!   that ends each option (`causal`).
//!
//! A request may also carry images (`media`), which a causal model with its
//! vision projector reads through its own tower.
//!
//! The scores are divided by a temperature fitted per question type and option
//! count, then normalized; the answer carries the distribution, a summary value
//! and a confidence. [`parity`] compares a decision with another implementation's
//! response for the same request.

pub mod backend;
mod causal;
pub mod json;
mod marker;
mod media;
pub mod parity;
mod template;
pub mod testkit;

pub use backend::{CausalBackend, CausalLoad, DecisionGpu, MarkerBackend, MarkerHeadOut, PromptRows, SlotPrefill, MAX_SLOTS};
pub use media::{Image, MAX_IMAGES};

use anyhow::{bail, Context, Result};
use json::Json;
use causal::{Causal, LabelSet, Read};
use ojas_formats::gguf::{Gguf, Meta};
use std::collections::HashMap;
use std::fmt;

/// An email triage request with seven questions of all three types: the request
/// `ojas bench` and `examples/decision_bench.rs` run by default.
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

/// The question types, in the order of a model's question-type rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuestionKind {
    /// Pick one of several named options.
    Choice,
    /// Pick a level on an ordered scale.
    Score,
    /// The probability that a statement holds.
    Noul,
}

impl QuestionKind {
    pub fn name(self) -> &'static str {
        match self { Self::Choice => "choice", Self::Score => "score", Self::Noul => "noul" }
    }

    pub fn index(self) -> u32 {
        match self { Self::Choice => 0, Self::Score => 1, Self::Noul => 2 }
    }
}

/// One option of a question: its key and its description as the request gave it.
#[derive(Clone, Debug)]
pub struct QuestionOption {
    pub key: String,
    pub description: Json,
}

/// One typed question. `options` are in the order the prompt shows them.
#[derive(Clone, Debug)]
pub struct Question {
    pub id: String,
    pub kind: QuestionKind,
    pub instructions: Json,
    pub options: Vec<QuestionOption>,
}

/// A request the caller got wrong, as opposed to a failure of the model.
#[derive(Debug)]
pub struct InvalidRequest(pub String);

impl fmt::Display for InvalidRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}

impl std::error::Error for InvalidRequest {}

/// A well-formed request asking for something this model cannot do.
#[derive(Debug)]
pub struct Unsupported(pub String);

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}

impl std::error::Error for Unsupported {}

fn invalid(msg: impl Into<String>) -> anyhow::Error { InvalidRequest(msg.into()).into() }

/// A parsed request.
#[derive(Clone, Debug)]
pub struct Request {
    /// The state, without the image parts its messages held.
    pub state: Json,
    pub questions: Vec<Question>,
    /// The request's images, in the order the prompt shows them.
    pub images: Vec<Image>,
}

/// One question's answer.
#[derive(Clone, Debug)]
pub struct Answer {
    pub id: String,
    pub kind: QuestionKind,
    /// Option keys in request order; a noul's are `false` and `true`.
    pub keys: Vec<String>,
    /// Option descriptions, in the order of `keys`.
    pub descriptions: Vec<Json>,
    /// Calibrated probability per key.
    pub probabilities: Vec<f64>,
}

impl Answer {
    /// The most probable option's index.
    pub fn best(&self) -> usize {
        self.probabilities.iter().enumerate().fold(0, |b, (i, &p)| if p > self.probabilities[b] { i } else { b })
    }

    /// The most probable option's key.
    pub fn choice(&self) -> &str { &self.keys[self.best()] }

    /// The expected level of a score: `sum(i * p_i)`.
    pub fn score(&self) -> f64 { self.probabilities.iter().enumerate().map(|(i, p)| i as f64 * p).sum() }

    /// The probability that a noul's statement holds.
    pub fn noul(&self) -> f64 {
        self.keys.iter().position(|k| k == "true").map_or(0.0, |i| self.probabilities[i])
    }

    /// How far the distribution is from uniform: for a choice, the most probable
    /// option's lead over `1/n`; for a score, one minus its mean distance to the
    /// mode relative to that of a uniform distribution. Both in `[0, 1]`.
    pub fn confidence(&self) -> f64 {
        let p = &self.probabilities;
        let n = p.len();
        if n < 2 { return 1.0; }
        match self.kind {
            QuestionKind::Score => {
                let mode = self.best();
                let dist: f64 = p.iter().enumerate().map(|(i, &q)| q * (i as f64 - mode as f64).abs()).sum();
                let uniform: f64 = (0..n).map(|i| (i as f64 - (n - 1) as f64 / 2.0).abs() / n as f64).sum();
                (1.0 - dist / uniform).max(0.0)
            }
            _ => {
                let u = 1.0 / n as f64;
                ((p[self.best()] - u) / (1.0 - u)).max(0.0)
            }
        }
    }

    /// The answer object of a response.
    pub fn to_json(&self) -> Json {
        let mut kv = vec![("type".to_string(), Json::Str(self.kind.name().into()))];
        let probabilities = || Json::Object(self.keys.iter().cloned().zip(self.probabilities.iter().map(|&p| Json::Float(p))).collect());
        match self.kind {
            QuestionKind::Choice => {
                kv.push(("choice".into(), Json::Str(self.choice().into())));
                kv.push(("probabilities".into(), probabilities()));
                kv.push(("confidence".into(), Json::Float(self.confidence())));
            }
            QuestionKind::Score => {
                kv.push(("score".into(), Json::Float(self.score())));
                kv.push(("legend".into(), Json::Object(self.keys.iter().cloned().zip(self.descriptions.iter().cloned()).collect())));
                kv.push(("probabilities".into(), probabilities()));
                kv.push(("confidence".into(), Json::Float(self.confidence())));
            }
            QuestionKind::Noul => kv.push(("noul".into(), Json::Float(self.noul()))),
        }
        Json::Object(kv)
    }
}

/// Answers to one request.
#[derive(Clone, Debug)]
pub struct Decision {
    pub answers: Vec<Answer>,
    /// Tokens across every prompt evaluated.
    pub input_tokens: usize,
    /// Of `input_tokens`, those reused from a shared prefix rather than computed.
    pub cached_tokens: usize,
    /// GPU time, seconds.
    pub gpu_s: f64,
}

impl Decision {
    /// The response body: `{"model", "answers", "usage"}`.
    pub fn to_json(&self, model: &str) -> Json {
        let answers = Json::Object(self.answers.iter().map(|a| (a.id.clone(), a.to_json())).collect());
        let usage = Json::Object(vec![
            ("input_tokens".into(), Json::Int(self.input_tokens.to_string())),
            ("output_tokens".into(), Json::Int("0".into())),
        ]);
        Json::Object(vec![("model".into(), Json::Str(model.into())), ("answers".into(), answers), ("usage".into(), usage)])
    }
}

/// How the scores of a question's options are read.
#[derive(Clone, Copy, Debug)]
enum Readout {
    /// An encoder head, scored at the marker token that opens each option.
    Markers,
    /// A causal language model's labels or pointers.
    Causal(Causal),
}

/// What a `decision.type` means: how its prompts are built and its scores read.
#[derive(Clone, Copy, Debug)]
struct Profile {
    readout: Readout,
    /// A noul's options are shown `true` first.
    noul_true_first: bool,
    /// Most options per question; a label readout may allow fewer.
    max_options: usize,
    /// Temperature buckets by option count: `(most options, name)`, ascending.
    buckets: &'static [(usize, &'static str)],
}

const OPTION_COUNT_BUCKETS: &[(usize, &str)] = &[(2, "2"), (5, "3_5"), (10, "6_10"), (usize::MAX, "11")];
const OPTION_SIZE_BUCKETS: &[(usize, &str)] = &[(8, "small"), (26, "mid"), (usize::MAX, "large")];

/// Every `decision.type` this engine serves.
const PROFILES: &[(&str, Profile)] = &[
    ("laya", Profile { readout: Readout::Markers, noul_true_first: false, max_options: 255, buckets: OPTION_COUNT_BUCKETS }),
    ("openjev", Profile {
        readout: Readout::Causal(Causal {
            read: Read::Labels(LabelSet::Letters), reversed_choice: false, noul_ratings: None, sorted_keys: false, text_only: false, images: true,
        }),
        noul_true_first: true, max_options: 255, buckets: OPTION_COUNT_BUCKETS,
    }),
    ("lev", Profile {
        readout: Readout::Causal(Causal {
            read: Read::Labels(LabelSet::Codes), reversed_choice: true, noul_ratings: Some(9), sorted_keys: true, text_only: false, images: false,
        }),
        noul_true_first: false, max_options: 255, buckets: OPTION_SIZE_BUCKETS,
    }),
    ("kev", Profile {
        readout: Readout::Causal(Causal {
            read: Read::Pointer { marker: "<|box_end|>" }, reversed_choice: false, noul_ratings: None, sorted_keys: false, text_only: true, images: false,
        }),
        noul_true_first: false, max_options: 255, buckets: OPTION_COUNT_BUCKETS,
    }),
];

/// Most levels a score question may have.
const MAX_SCORE_LEVELS: usize = 10;

/// The fitted temperatures, keyed `<type>` and `<type>.<bucket>`.
#[derive(Clone, Debug, Default)]
struct Calibration(HashMap<String, f32>);

impl Calibration {
    fn from_gguf(g: &Gguf, prefix: &str) -> Result<Self> {
        let mut t = HashMap::new();
        for (key, v) in &g.meta {
            let Some(name) = key.strip_prefix(prefix) else { continue };
            let value = match v {
                Meta::F32(x) => *x,
                other => bail!("{key} is {other:?}, not a temperature"),
            };
            if value.is_nan() || value <= 0.0 { bail!("{key} = {value} is not a positive temperature"); }
            t.insert(name.to_string(), value);
        }
        Ok(Calibration(t))
    }

    /// The temperature for `kind` with `n` options: the bucket's, else the type's,
    /// else 1.
    fn temperature(&self, kind: QuestionKind, n: usize, buckets: &[(usize, &str)]) -> f32 {
        let bucket = buckets.iter().find(|(most, _)| n <= *most).map(|(_, b)| *b);
        bucket.and_then(|b| self.0.get(&format!("{}.{b}", kind.name())))
            .or_else(|| self.0.get(kind.name()))
            .copied().unwrap_or(1.0)
    }
}

/// A request body for a model of `profile`.
fn parse_request(body: &Json, profile: &Profile) -> Result<Request> {
    let state = match body.get("state") {
        None | Some(Json::Null) => return Err(invalid("\"state\" must be provided")),
        Some(s) => s.clone(),
    };
    let questions = match body.get("questions") {
        Some(Json::Object(q)) if !q.is_empty() => q,
        _ => return Err(invalid("\"questions\" must be a non-empty object")),
    };
    let questions = questions.iter().map(|(id, q)| parse_question(id, q, profile)).collect::<Result<_>>()?;
    let (state, images) = media::take_images(body, &state)?;
    Ok(Request { state, questions, images })
}

/// One question, with its options in the order the prompt shows them.
fn parse_question(id: &str, q: &Json, profile: &Profile) -> Result<Question> {
    let err = |msg: &str| invalid(format!("questions.{id}: {msg}"));
    if q.as_object().is_none() { return Err(err("must be an object")); }
    let instructions = match q.get("instructions") {
        None | Some(Json::Null) => return Err(err("\"instructions\" must be provided")),
        Some(i) => i.clone(),
    };
    let criteria = q.get("criteria").unwrap_or(&Json::Null);
    let option = |key: &str, description: &Json| QuestionOption { key: key.to_string(), description: description.clone() };
    let (kind, options) = match q.get("type").and_then(Json::as_str) {
        Some("choice") => match criteria {
            Json::Object(kv) if !kv.is_empty() => (QuestionKind::Choice, kv.iter().map(|(k, d)| option(k, d)).collect()),
            _ => return Err(err("\"criteria\" must be a non-empty object")),
        },
        Some("score") => match criteria {
            Json::Array(levels) if (2..=MAX_SCORE_LEVELS).contains(&levels.len()) =>
                (QuestionKind::Score, levels.iter().enumerate().map(|(i, d)| option(&i.to_string(), d)).collect()),
            _ => return Err(err(&format!("\"criteria\" must be an array of 2 to {MAX_SCORE_LEVELS} levels"))),
        },
        Some("noul") => {
            if !matches!(criteria, Json::Null | Json::Object(_)) { return Err(err("\"criteria\" must be an object")); }
            let side = |k: &str| option(k, criteria.get(k).unwrap_or(&Json::Null));
            let mut options = vec![side("false"), side("true")];
            if profile.noul_true_first { options.reverse(); }
            (QuestionKind::Noul, options)
        }
        _ => return Err(err("\"type\" must be one of: choice, score, noul")),
    };
    if options.len() > profile.max_options {
        return Err(err(&format!("too many options ({}), this model supports at most {}", options.len(), profile.max_options)));
    }
    Ok(Question { id: id.to_string(), kind, instructions, options })
}

/// A small tensor as f32. `read_tensor` hands back f32 or f16 for every type.
fn read_f32(g: &mut Gguf, name: &str) -> Result<Vec<f32>> {
    let (_, ty, bytes) = g.read_tensor(name)?;
    Ok(match ty {
        0 => bytes.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
        _ => bytes.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c).to_f32()).collect(),
    })
}

/// What a readout returns for a request.
struct Scored {
    /// Per question, per option order, one score per output.
    scores: Vec<Vec<Vec<f32>>>,
    /// Tokens across every prompt evaluated.
    tokens: usize,
    /// Of `tokens`, those reused from a shared prefix rather than computed.
    reused: usize,
    gpu_s: f64,
}

enum Engine<M, C> {
    Markers(marker::MarkerHead<M>),
    Causal(causal::CausalHead<C>),
}

/// One question's prompt for an encoder model, as its head reads it: the tokens, the
/// question type (its row of the type embedding), the positions of the option
/// markers in the request's option order, and the calibration temperature its
/// option scores are divided by.
#[derive(Clone, Debug)]
pub struct MarkerPrompt {
    pub ids: Vec<u32>,
    pub qtype: u32,
    pub markers: Vec<usize>,
    pub temperature: f32,
}

/// A loaded decision model, on the backend `G` loaded it with.
pub struct DecisionModel<'a, G: DecisionGpu + 'a> {
    name: String,
    /// The file's `decision.type`.
    kind: String,
    profile: Profile,
    calibration: Calibration,
    engine: Engine<G::Marker<'a>, G::Causal<'a>>,
}

/// The `decision.type` a GGUF names, read from its header.
pub fn decision_type(path: &str) -> Option<String> {
    let g = Gguf::open(path).ok()?;
    match g.meta.get(&format!("{}.decision.type", g.arch())) {
        Some(Meta::Str(t)) => Some(t.clone()),
        _ => None,
    }
}

impl<'a, G: DecisionGpu + 'a> DecisionModel<'a, G> {
    pub fn load(gpu: &'a G, path: &str) -> Result<Self> {
        let mut g = Gguf::open(path).with_context(|| format!("opening {path}"))?;
        let prefix = format!("{}.decision.", g.arch());
        let kind = match g.meta.get(&format!("{prefix}type")) {
            Some(Meta::Str(t)) => t.clone(),
            _ => bail!("{path} is not a decision model (no {prefix}type)"),
        };
        let mut profile = PROFILES.iter().find(|(t, _)| *t == kind).map(|(_, p)| *p)
            .with_context(|| format!("{path}: decision type {kind:?} is not supported"))?;
        let calibration = Calibration::from_gguf(&g, &format!("{prefix}temperature."))?;
        let source = match g.meta.get("tokenizer.chat_template.systemone") {
            Some(Meta::Str(s)) => s.clone(),
            _ => bail!("{path}: no decision template (tokenizer.chat_template.systemone)"),
        };
        let template = template::Template::new(&source)?;
        let engine = match profile.readout {
            Readout::Markers => Engine::Markers(marker::MarkerHead::load(gpu, &mut g, template, &prefix)?),
            Readout::Causal(spec) => {
                let head = causal::CausalHead::load(gpu, &mut g, template, spec)?;
                profile.max_options = profile.max_options.min(head.max_options());
                Engine::Causal(head)
            }
        };
        let name = std::path::Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or(&kind).to_string();
        Ok(DecisionModel { name, kind, profile, calibration, engine })
    }

    /// The model's name: its file name without the extension.
    pub fn name(&self) -> &str { &self.name }

    /// The file's `decision.type`.
    pub fn kind(&self) -> &str { &self.kind }

    /// Most options a question may have.
    pub fn max_options(&self) -> usize { self.profile.max_options }

    /// Parse `{"state": ..., "questions": {...}}`. Errors are [`InvalidRequest`] or
    /// [`Unsupported`].
    pub fn request(&self, body: &Json) -> Result<Request> {
        let req = parse_request(body, &self.profile)?;
        self.validate(&req)?;
        Ok(req)
    }

    /// Check what a request asks of this model: at most [`MAX_IMAGES`] images, and
    /// none unless it takes them. Errors are [`InvalidRequest`] or [`Unsupported`].
    pub fn validate(&self, req: &Request) -> Result<()> {
        if req.images.len() > MAX_IMAGES {
            return Err(invalid(format!("too many images, the maximum is {MAX_IMAGES}")));
        }
        if !req.images.is_empty() && !self.takes_images() {
            return Err(Unsupported(match self.profile.readout {
                Readout::Causal(c) if c.images => "images need the model's vision projector: an mmproj GGUF beside the model file".into(),
                _ => "this model does not take images".into(),
            }).into());
        }
        Ok(())
    }

    /// Whether requests may carry images.
    pub fn takes_images(&self) -> bool {
        match &self.engine {
            Engine::Markers(_) => false,
            Engine::Causal(c) => c.takes_images(),
        }
    }

    /// Answer every question of `req`.
    pub fn decide(&self, req: &Request) -> Result<Decision> {
        self.decide_all(&[req]).pop().expect("one result per request")
    }

    /// Answer several requests, each on its own: one request's failure is its own
    /// result. An encoder model runs the questions of all of them in shared GPU
    /// passes; a causal model runs requests asked in one prompt together, a slot
    /// each, and the others one at a time.
    pub fn decide_all(&self, reqs: &[&Request]) -> Vec<Result<Decision>> {
        let scored: Vec<Result<Scored>> = match &self.engine {
            Engine::Markers(m) => m.scores(reqs),
            Engine::Causal(c) => c.scores_many(&reqs.iter().map(|r| (&r.state, r.questions.as_slice(), r.images.as_slice())).collect::<Vec<_>>()),
        };
        reqs.iter().zip(scored).map(|(req, s)| s.map(|s| {
            let answers = req.questions.iter().zip(&s.scores).map(|(q, v)| self.answer(q, v)).collect();
            Decision { answers, input_tokens: s.tokens, cached_tokens: s.reused, gpu_s: s.gpu_s }
        })).collect()
    }

    /// The calibrated distribution over `q`'s options from their scores, one list
    /// per option order: the request's, then reversed. The orders are averaged.
    fn answer(&self, q: &Question, variants: &[Vec<f32>]) -> Answer {
        let t = self.calibration.temperature(q.kind, q.options.len(), self.profile.buckets) as f64;
        let n = variants[0].len();
        let mut probabilities = vec![0.0f64; n];
        for (v, scores) in variants.iter().enumerate() {
            let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f64> = scores.iter().map(|&s| ((s - mx) as f64 / t).exp()).collect();
            let sum: f64 = e.iter().sum();
            for (i, p) in e.iter().enumerate() {
                probabilities[if v == 0 { i } else { n - 1 - i }] += p / sum / variants.len() as f64;
            }
        }
        // A rating readout: the expected rating, as the probability of `true`.
        if q.kind == QuestionKind::Noul && n != q.options.len() {
            let p: f64 = probabilities.iter().enumerate().map(|(r, p)| p * r as f64 / (n - 1) as f64).sum();
            probabilities = vec![1.0 - p, p];
        }
        let mut options = q.options.clone();
        if q.kind == QuestionKind::Noul && self.profile.noul_true_first {
            options.reverse();
            probabilities.reverse();
        }
        Answer {
            id: q.id.clone(), kind: q.kind,
            keys: options.iter().map(|o| o.key.clone()).collect(),
            descriptions: options.into_iter().map(|o| o.description).collect(),
            probabilities,
        }
    }

    /// The loaded encoder of an encoder model, for a host that trains it; `None` for a
    /// causal model.
    pub fn marker_backend(&self) -> Option<&G::Marker<'a>> {
        match &self.engine {
            Engine::Markers(m) => Some(m.backend()),
            Engine::Causal(_) => None,
        }
    }

    /// Each question's encoder prompt, exactly as [`Self::decide`] builds it.
    pub fn marker_prompts(&self, req: &Request) -> Result<Vec<MarkerPrompt>> {
        let Engine::Markers(m) = &self.engine else { bail!("{} is not an encoder model", self.name) };
        let (seqs, markers) = m.sequences(&req.state, &req.questions)?;
        Ok(req.questions.iter().zip(seqs).zip(markers).map(|((q, ids), markers)| MarkerPrompt {
            ids, qtype: q.kind.index(), markers,
            temperature: self.calibration.temperature(q.kind, q.options.len(), self.profile.buckets),
        }).collect())
    }

    /// Per-category GPU time of the encoder over `req`'s prompts.
    pub fn profile_gpu(&self, req: &Request) -> Result<Vec<(String, f64)>> {
        match &self.engine {
            Engine::Markers(m) => m.profile(&req.state, &req.questions),
            Engine::Causal(_) => bail!("GPU profiling covers encoder models"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(kind: QuestionKind, p: &[f64]) -> Answer {
        Answer {
            id: "q".into(), kind, keys: (0..p.len()).map(|i| i.to_string()).collect(),
            descriptions: vec![Json::Null; p.len()], probabilities: p.to_vec(),
        }
    }

    fn parse(body: &str) -> Result<Request> { parse_request(&Json::parse(body).unwrap(), &PROFILES[0].1) }

    #[test]
    fn requests_are_validated_as_the_reference_validates_them() {
        let state = r#""I was charged twice.""#;
        for body in [
            r#"{"questions": {"q": {"type": "noul", "instructions": "x"}}}"#.to_string(),
            format!(r#"{{"state": {state}}}"#),
            format!(r#"{{"state": {state}, "questions": {{}}}}"#),
            format!(r#"{{"state": {state}, "questions": {{"q": {{"type": "unknown", "instructions": "x"}}}}}}"#),
            format!(r#"{{"state": {state}, "questions": {{"q": {{"type": "noul"}}}}}}"#),
            format!(r#"{{"state": {state}, "questions": {{"q": {{"type": "choice", "instructions": "x"}}}}}}"#),
            format!(r#"{{"state": {state}, "questions": {{"q": {{"type": "choice", "instructions": "x", "criteria": {{}}}}}}}}"#),
            format!(r#"{{"state": {state}, "questions": {{"q": {{"type": "score", "instructions": "x", "criteria": ["only one"]}}}}}}"#),
        ] {
            let err = parse(&body).expect_err(&body);
            assert!(err.downcast_ref::<InvalidRequest>().is_some(), "{body}: {err}");
        }
        let not_a_data_url = format!(r#"{{"state": {state}, "questions": {{"q": {{"type": "noul", "instructions": "x"}}}}, "images": ["https://example.com/a.png"]}}"#);
        assert!(parse(&not_a_data_url).unwrap_err().downcast_ref::<InvalidRequest>().is_some());
    }

    #[test]
    fn questions_keep_their_option_order() {
        let req = parse(r#"{"state": {"a": 1}, "questions": {
            "route": {"type": "choice", "instructions": "x", "criteria": {"z": null, "a": "first"}},
            "level": {"type": "score", "instructions": "x", "criteria": ["low", "high"]},
            "flag": {"type": "noul", "instructions": "x", "criteria": {"true": "yes"}}}}"#).unwrap();
        let keys = |i: usize| req.questions[i].options.iter().map(|o| o.key.as_str()).collect::<Vec<_>>();
        assert_eq!((keys(0), keys(1), keys(2)), (vec!["z", "a"], vec!["0", "1"], vec!["false", "true"]));
        assert_eq!(req.questions[2].options[1].description, Json::Str("yes".into()));
    }

    #[test]
    fn confidence_is_zero_for_uniform_and_one_for_certain() {
        for kind in [QuestionKind::Choice, QuestionKind::Score] {
            assert!(answer(kind, &[0.25; 4]).confidence().abs() < 1e-12);
            assert!((answer(kind, &[0.0, 0.0, 1.0, 0.0]).confidence() - 1.0).abs() < 1e-12);
        }
        assert!((answer(QuestionKind::Choice, &[0.6, 0.2, 0.2]).confidence() - 0.4).abs() < 1e-12);
    }

    #[test]
    fn temperature_prefers_the_bucket_then_the_type() {
        let c = Calibration([("choice".to_string(), 2.0), ("choice.3_5".to_string(), 3.0)].into_iter().collect());
        assert_eq!(c.temperature(QuestionKind::Choice, 4, OPTION_COUNT_BUCKETS), 3.0);
        assert_eq!(c.temperature(QuestionKind::Choice, 12, OPTION_COUNT_BUCKETS), 2.0);
        assert_eq!(c.temperature(QuestionKind::Score, 4, OPTION_COUNT_BUCKETS), 1.0);
    }
}
