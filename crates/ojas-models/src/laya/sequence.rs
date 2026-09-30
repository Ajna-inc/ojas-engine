//! Typed questions and the token sequence Laya scores them from.
//!
//! A transcription of the reference `rl_common.py` (`render_options`,
//! `build_sequence`) and `rl_agent_api.py` (`_to_internal`). Every question is one
//! sequence:
//!
//! ```text
//! [CLS] "<type> question: <instructions>" [SEP] [MASK] " opt0" [MASK] " opt1" ... [SEP] <state> [SEP]
//! ```
//!
//! and the head scores each option at its `[MASK]`. The truncation arithmetic is
//! the reference's exactly, since it decides which tokens the model sees.

use super::json::Json;
use anyhow::{bail, ensure, Result};
use ojas_tokenize::Bpe;

/// The three question types, in `type_emb` row order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuestionKind {
    /// Pick one of several named options.
    Choice,
    /// Pick a level on an ordered scale.
    Score,
    /// Yes or no: probability that a statement holds.
    Noul,
}

impl QuestionKind {
    pub fn name(self) -> &'static str {
        match self { Self::Choice => "choice", Self::Score => "score", Self::Noul => "noul" }
    }

    pub fn index(self) -> u32 {
        match self { Self::Choice => 0, Self::Score => 1, Self::Noul => 2 }
    }

    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "choice" => Ok(Self::Choice),
            "score" => Ok(Self::Score),
            "noul" => Ok(Self::Noul),
            other => bail!("unknown question type {other:?} (expected choice, score or noul)"),
        }
    }
}

/// One typed question, ready to encode.
#[derive(Clone, Debug)]
pub struct Question {
    pub id: String,
    pub kind: QuestionKind,
    pub instructions: String,
    /// Output label per option: the choice names, `"0".."k-1"` for a score, and
    /// `["false", "true"]` for a noul.
    pub labels: Vec<String>,
    /// Option text as the model reads it, in label order.
    pub options: Vec<String>,
}

/// A criterion's text, `None` when absent or empty (the reference tests truthiness).
fn text_of(v: &Json, what: &str) -> Result<Option<String>> {
    match v {
        Json::Null => Ok(None),
        Json::Str(s) if s.is_empty() => Ok(None),
        Json::Str(s) => Ok(Some(s.clone())),
        _ => bail!("{what} must be a string or null"),
    }
}

impl Question {
    /// Build from the request shape `{"type", "instructions", "criteria"}`.
    pub fn from_json(id: &str, q: &Json) -> Result<Question> {
        let kind = QuestionKind::from_name(q.get("type").and_then(Json::as_str)
            .ok_or_else(|| anyhow::anyhow!("question {id:?}: missing \"type\""))?)?;
        // Non-string instructions are serialized with Python's default json.dumps.
        let instructions = match q.get("instructions") {
            Some(Json::Str(s)) => s.clone(),
            Some(other) => other.to_python(true),
            None => bail!("question {id:?}: missing \"instructions\""),
        };
        let crit = q.get("criteria").unwrap_or(&Json::Null);
        let (labels, options): (Vec<String>, Vec<String>) = match kind {
            QuestionKind::Choice => {
                let named: Vec<(String, Option<String>)> = match crit {
                    Json::Object(kv) => kv.iter()
                        .map(|(k, v)| Ok((k.clone(), text_of(v, &format!("question {id:?} option {k:?}"))?)))
                        .collect::<Result<_>>()?,
                    Json::Array(items) => items.iter()
                        .map(|v| v.as_str().map(|s| (s.to_string(), None))
                            .ok_or_else(|| anyhow::anyhow!("question {id:?}: choice options must be strings")))
                        .collect::<Result<_>>()?,
                    _ => bail!("question {id:?}: a choice needs \"criteria\" as an object or a list"),
                };
                named.into_iter().map(|(k, v)| {
                    let text = match &v { Some(d) => format!("{k}: {d}"), None => k.clone() };
                    (k, text)
                }).unzip()
            }
            QuestionKind::Score => {
                let Json::Array(levels) = crit else { bail!("question {id:?}: a score needs \"criteria\" as a list") };
                levels.iter().enumerate().map(|(i, v)| {
                    let c = v.as_str().ok_or_else(|| anyhow::anyhow!("question {id:?}: score levels must be strings"))?;
                    Ok((i.to_string(), format!("level {i}: {c}")))
                }).collect::<Result<Vec<_>>>()?.into_iter().unzip()
            }
            QuestionKind::Noul => {
                let side = |key: &str, default: &str| -> Result<String> {
                    let v = crit.get(key).map(|v| text_of(v, &format!("question {id:?} {key}"))).transpose()?.flatten();
                    Ok(format!("{key}: {}", v.unwrap_or_else(|| default.to_string())))
                };
                (vec!["false".into(), "true".into()],
                 vec![side("false", "no, the statement does not hold")?, side("true", "yes, the statement holds")?])
            }
        };
        ensure!(options.len() >= 2, "question {id:?}: needs at least two options, has {}", options.len());
        Ok(Question { id: id.to_string(), kind, instructions, labels, options })
    }
}

/// Token ids and budgets `build_sequence` needs.
#[derive(Clone, Debug)]
pub struct SequenceSpec {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    /// The mask token's text (`[MASK]`, `<mask>`): the reference replaces it with a
    /// space wherever it occurs in instructions, options or the state.
    pub mask_text: String,
    pub max_len: usize,
    pub head_max_len: usize,
}

/// Tokens per option text before the budget is applied.
const MAX_OPTION_TOKENS: usize = 48;
/// Head budget left for the question text below which options shrink evenly.
const MIN_QUESTION_BUDGET: usize = 16;

fn encode(tok: &Bpe, text: &str) -> Vec<u32> {
    tok.encode(text).into_iter().map(|t| t as u32).collect()
}

/// One question's sequence and the positions of its option markers. `state_ids` is
/// the state, tokenized once and shared by every question of a request.
pub fn build_sequence(tok: &Bpe, spec: &SequenceSpec, q: &Question, state_ids: &[u32]) -> Result<(Vec<u32>, Vec<usize>)> {
    let mask_text = spec.mask_text.as_str();
    let ins = q.instructions.replace(mask_text, " ");
    let head_ids = encode(tok, &format!("{} question: {ins}", q.kind.name()));
    let mut opt_ids: Vec<Vec<u32>> = q.options.iter().map(|o| {
        let mut ids = vec![spec.mask];
        ids.extend(encode(tok, &format!(" {}", o.replace(mask_text, " "))).into_iter().take(MAX_OPTION_TOKENS));
        ids
    }).collect();
    let used = |o: &[Vec<u32>]| o.iter().map(Vec::len).sum::<usize>();
    let mut opt_budget = spec.head_max_len as isize - used(&opt_ids) as isize;
    if opt_budget < MIN_QUESTION_BUDGET as isize {
        let per = ((spec.head_max_len as isize - MIN_QUESTION_BUDGET as isize)
            .div_euclid(opt_ids.len().max(1) as isize)).max(4) as usize;
        for o in &mut opt_ids { o.truncate(per); }
        opt_budget = spec.head_max_len as isize - used(&opt_ids) as isize;
    }
    let keep = opt_budget.max(8) as usize;
    let mut ids = vec![spec.cls];
    ids.extend(head_ids.iter().take(keep));
    ids.push(spec.sep);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(spec.sep);
    let room = spec.max_len.saturating_sub(ids.len() + 1);
    ids.extend(state_ids.iter().take(room));
    ids.push(spec.sep);
    ids.truncate(spec.max_len);
    markers.retain(|&m| m < spec.max_len);
    ensure!(markers.len() == q.options.len(),
        "question {:?}: its options do not fit in head_max_len={} tokens", q.id, spec.head_max_len);
    Ok((ids, markers))
}

/// Tokenize the state the way the reference does: a JSON object or array is first
/// written as `json.dumps(state, ensure_ascii=False)`; text is used as is.
pub fn state_ids(tok: &Bpe, spec: &SequenceSpec, state: &Json) -> Vec<u32> {
    let text = match state {
        Json::Str(s) => s.clone(),
        other => other.to_python(false),
    };
    encode(tok, &text.replace(spec.mask_text.as_str(), " "))
}
