//! Token log-probabilities for a completion request: OpenAI's `logprobs` and
//! `top_logprobs`.
//!
//! The distribution reported at each step is the model's own, over the tokens that
//! step could have produced: a softmax of the raw logits restricted to the ids the
//! request's grammar or JSON schema allows there and to ids that are not suppressed
//! (`ignore_eos`). It is taken before the sampler: temperature, top-p, top-k and the
//! repetition penalty do not change it. So a log-probability says how strongly the
//! model preferred a token among its legal choices, whatever the sampling settings,
//! and two requests that differ only in temperature report the same numbers for the
//! same text. A token a constraint forces has log-probability 0. At temperature 0
//! the chosen token is the first entry of `top_logprobs`, exact ties aside.
//!
//! The numbers are exact, never estimated. The request's constraint is wrapped in
//! an [`Observer`], a [`LogitProcessor`], and the engine turns speculative decoding
//! off while any processor is attached, so every token is chosen on the host from
//! the full logits of an ordinary forward and the observer sees exactly those. A
//! drafted token is never reported. The prompt-prefix cache restores state that is
//! bit-identical to processing the prompt, so a cached prompt reports what an
//! uncached one does. A model whose backend does not expose logits answers 501
//! instead of generating. The observer reads the logits and never edits them, so
//! asking for log-probabilities does not change which tokens are generated.
//!
//! A request that does not ask for log-probabilities gets no observer and takes
//! the token path it always did. One that does pays for reading the logits back
//! and one pass over them per token; with a grammar it also pays for the set of
//! allowed tokens, computed once for each grammar state it visits.
//!
//! One entry is reported per completion token counted in `usage`, in order, which
//! includes a token whose text a stop string held back. An entry's `token` is that
//! token's bytes decoded alone, so a character split across tokens reads as U+FFFD
//! there; `bytes` are exact.

use anyhow::Result;
use ojas_grammar::GrammarProcessor;
use ojas_infer::LogitProcessor;
use serde_json::{json, Map, Value};
use std::sync::{Arc, Mutex};

/// Most alternatives reported per token.
const MAX_TOP: usize = 20;

/// The alternatives a request wants per token, or `None` when it asks for no
/// log-probabilities. Chat takes `logprobs: true` and `top_logprobs: N`; the
/// completion routes take `logprobs: N`.
fn requested(body: &Value, chat: bool) -> Result<Option<usize>> {
    let count = |key: &str, v: &Value| match v.as_u64() {
        Some(n) if n as usize <= MAX_TOP => Ok(n as usize),
        _ => anyhow::bail!("\"{key}\" must be an integer from 0 to {MAX_TOP}"),
    };
    let on = body.get("logprobs").filter(|v| !v.is_null());
    if !chat {
        return match on {
            None | Some(Value::Bool(false)) => Ok(None),
            Some(Value::Bool(true)) => Ok(Some(0)),
            Some(v) => count("logprobs", v).map(Some),
        };
    }
    let on = match on {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => anyhow::bail!("\"logprobs\" must be a boolean"),
    };
    match body.get("top_logprobs").filter(|v| !v.is_null()) {
        Some(_) if !on => anyhow::bail!("\"top_logprobs\" needs \"logprobs\": true"),
        Some(v) => count("top_logprobs", v).map(Some),
        None => Ok(on.then_some(0)),
    }
}

/// The logit processor a request runs with, if any.
pub(crate) type Processor = Option<Box<dyn LogitProcessor + Send>>;

/// Read a request's log-probability fields and box its `constraint` for the
/// engine. When the request asks for log-probabilities the constraint is wrapped
/// to observe every step, and the collector for the reply is returned with it.
pub(crate) fn attach(body: &Value, chat: bool, constraint: Option<GrammarProcessor>) -> Result<(Option<Logprobs>, Processor)> {
    let Some(top) = requested(body, chat)? else {
        return Ok((None, constraint.map(|p| Box::new(p) as Box<dyn LogitProcessor + Send>)));
    };
    let last = Arc::new(Mutex::new(None));
    let observer = Observer { inner: constraint, top, logits: Vec::new(), norm: 0.0, best: Vec::new(), last: last.clone() };
    Ok((Some(Logprobs { top, chat, last, entries: Vec::new(), offset: 0 }), Some(Box::new(observer))))
}

/// The log of the summed probability mass of the tokens `allowed` marks (all of
/// them without it), and the `top` most likely of them, best first (ties by id).
/// Only finite logits count: a suppressed token's is negative infinity. A token's
/// log-probability is its logit minus the returned normaliser.
fn distribution(logits: &[f32], top: usize, allowed: Option<&[bool]>) -> (f64, Vec<(u32, f32)>) {
    // A running log-sum-exp: `sum` is the mass seen so far relative to `max`.
    let (mut max, mut sum) = (f64::NEG_INFINITY, 0.0f64);
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(top + 1);
    for (i, &l) in logits.iter().enumerate() {
        if !l.is_finite() || allowed.is_some_and(|a| !a.get(i).copied().unwrap_or(false)) { continue; }
        let x = l as f64;
        if x > max {
            sum = sum * (max - x).exp() + 1.0;
            max = x;
        } else {
            sum += (x - max).exp();
        }
        if best.len() < top || best.last().is_some_and(|&(_, low)| l > low) {
            let at = best.partition_point(|&(_, b)| b >= l);
            best.insert(at, (i as u32, l));
            best.truncate(top);
        }
    }
    (max + sum.ln(), best)
}

/// One step as the observer saw it: the chosen token's log-probability and the
/// most likely alternatives.
struct Step {
    token: u32,
    logprob: f64,
    top: Vec<(u32, f64)>,
}

/// Watches every step of one request from the logit hook, in front of the
/// request's own constraint when it has one, without changing what is generated:
/// the logits are read, never edited, and every call is passed on.
struct Observer {
    inner: Option<GrammarProcessor>,
    top: usize,
    /// This step's logits, its normaliser and its best tokens, kept from `process`
    /// until `accept` names the chosen token.
    logits: Vec<f32>,
    norm: f64,
    best: Vec<(u32, f32)>,
    /// The last accepted step, for the reply to take with the token.
    last: Arc<Mutex<Option<Step>>>,
}

impl LogitProcessor for Observer {
    fn process(&mut self, logits: &mut [f32]) {
        if let Some(p) = &mut self.inner { p.process(logits); }
        self.logits.clear();
        self.logits.extend_from_slice(logits);
        let allowed = self.inner.as_mut().map(|p| p.allowed_tokens());
        (self.norm, self.best) = distribution(logits, self.top, allowed.as_deref());
    }

    fn allows(&mut self, token: u32) -> bool {
        match &mut self.inner {
            Some(p) => p.allows(token),
            None => true,
        }
    }

    fn accept(&mut self, token: u32) {
        if let Some(p) = &mut self.inner { p.accept(token); }
        let logprob = |l: f32| l as f64 - self.norm;
        let step = self.logits.get(token as usize).filter(|l| l.is_finite()).map(|&l| Step {
            token, logprob: logprob(l), top: self.best.iter().map(|&(t, l)| (t, logprob(l))).collect(),
        });
        *self.last.lock().unwrap() = step;
    }

    fn finished(&self) -> bool { self.inner.as_ref().is_some_and(|p| p.finished()) }
}

/// A token as a response names it: its bytes, and those bytes as text.
struct Piece {
    text: String,
    bytes: Vec<u8>,
}

impl Piece {
    fn new(bytes: Vec<u8>) -> Self { Piece { text: String::from_utf8_lossy(&bytes).into_owned(), bytes } }
}

struct Entry {
    piece: Piece,
    /// `None` only if the step was never observed, reported as `null` rather than
    /// a made-up number.
    logprob: Option<f64>,
    top: Vec<(Piece, f64)>,
    /// Characters of the earlier tokens' text.
    offset: usize,
}

/// The log-probabilities of one reply: collects each token's as it is emitted and
/// shapes them for the route that asked.
pub(crate) struct Logprobs {
    top: usize,
    chat: bool,
    last: Arc<Mutex<Option<Step>>>,
    entries: Vec<Entry>,
    offset: usize,
}

impl Logprobs {
    /// Record emitted token `t`, whose step the observer just accepted, and return
    /// its log-probabilities in the response's shape, for a streamed chunk.
    /// `bytes_of` gives a token's bytes.
    pub(crate) fn token(&mut self, bytes_of: impl Fn(u32) -> Vec<u8>, t: u32) -> Value {
        let step = self.last.lock().unwrap().take().filter(|s| s.token == t);
        let piece = Piece::new(bytes_of(t));
        let offset = self.offset;
        self.offset += piece.text.chars().count();
        let (logprob, top) = match step {
            Some(s) => (Some(s.logprob), s.top.into_iter().map(|(id, lp)| (Piece::new(bytes_of(id)), lp)).collect()),
            None => (None, Vec::new()),
        };
        self.entries.push(Entry { piece, logprob, top, offset });
        self.shaped(&self.entries[self.entries.len() - 1..])
    }

    /// Each token's length in bytes and its log-probability, in order.
    pub(crate) fn steps(&self) -> Vec<(usize, Option<f64>)> {
        self.entries.iter().map(|e| (e.piece.bytes.len(), e.logprob)).collect()
    }

    /// Every token's log-probabilities, for a whole response.
    pub(crate) fn whole(&self) -> Value { self.shaped(&self.entries) }

    /// `entries` as chat's `{"content": [{token, logprob, bytes, top_logprobs}]}` or
    /// as the completion routes' parallel lists, where `text_offset` counts
    /// characters from the start of the generated text and `top_logprobs` is null
    /// when no alternatives were asked for.
    fn shaped(&self, entries: &[Entry]) -> Value {
        if self.chat {
            let content: Vec<Value> = entries.iter().map(|e| json!({
                "token": e.piece.text, "logprob": e.logprob, "bytes": e.piece.bytes,
                "top_logprobs": e.top.iter()
                    .map(|(p, lp)| json!({"token": p.text, "logprob": lp, "bytes": p.bytes})).collect::<Vec<_>>(),
            })).collect();
            return json!({"content": content, "refusal": null});
        }
        let top = (self.top > 0).then(|| entries.iter()
            .map(|e| Value::Object(e.top.iter().map(|(p, lp)| (p.text.clone(), json!(lp))).collect::<Map<_, _>>()))
            .collect::<Vec<_>>());
        json!({
            "tokens": entries.iter().map(|e| e.piece.text.as_str()).collect::<Vec<_>>(),
            "token_logprobs": entries.iter().map(|e| e.logprob).collect::<Vec<_>>(),
            "top_logprobs": top,
            "text_offset": entries.iter().map(|e| e.offset).collect::<Vec<_>>(),
        })
    }
}

/// Put log-probabilities where a response or stream event carries them: in its one
/// choice on the OpenAI routes, at the top level on `/completion`.
pub(crate) fn put(response: &mut Value, oai: bool, logprobs: Value) {
    if oai { response["choices"][0]["logprobs"] = logprobs } else { response["logprobs"] = logprobs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_grammar::{Grammar, TokenVocab};

    /// Every token's log-probability under `distribution`, for checking against a
    /// softmax computed the plain way.
    fn logprobs_of(logits: &[f32], allowed: Option<&[bool]>) -> Vec<(u32, f64)> {
        let (norm, best) = distribution(logits, logits.len(), allowed);
        best.into_iter().map(|(t, l)| (t, l as f64 - norm)).collect()
    }

    #[test]
    fn log_probabilities_are_a_log_softmax_sorted_best_first() {
        let logits = [1.0f32, 3.0, -2.0, 0.5, 3.5];
        let z: f64 = logits.iter().map(|&l| (l as f64).exp()).sum();
        let got = logprobs_of(&logits, None);
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), [4, 1, 0, 3, 2]);
        for &(t, lp) in &got {
            assert!((lp - ((logits[t as usize] as f64).exp() / z).ln()).abs() < 1e-12, "token {t}: {lp}");
        }
        assert!((got.iter().map(|g| g.1.exp()).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_top_list_is_the_best_few_and_ties_keep_id_order() {
        let logits = [0.0f32, 2.0, 2.0, 5.0, -1.0, 2.0];
        let (norm, best) = distribution(&logits, 3, None);
        assert_eq!(best, [(3, 5.0), (1, 2.0), (2, 2.0)]);
        // The normaliser covers every token, not only the listed ones.
        let z: f64 = logits.iter().map(|&l| (l as f64).exp()).sum();
        assert!((norm - z.ln()).abs() < 1e-12);
        assert!(best.iter().map(|&(_, l)| (l as f64 - norm).exp()).sum::<f64>() < 1.0);
        assert!(distribution(&logits, 0, None).1.is_empty());
    }

    #[test]
    fn large_logits_do_not_overflow() {
        let (norm, best) = distribution(&[1000.0, 999.0, -1000.0], 2, None);
        assert!((norm - (1000.0 + (1.0 + (-1.0f64).exp()).ln())).abs() < 1e-9);
        assert_eq!(best[0].0, 0);
        // Rising logits take the other branch of the running sum.
        let (rising, _) = distribution(&[-1000.0, 999.0, 1000.0], 0, None);
        assert!((rising - norm).abs() < 1e-9);
    }

    #[test]
    fn forbidden_tokens_take_no_probability() {
        let logits = [4.0f32, f32::NEG_INFINITY, 1.0, 9.0, f32::NAN, 2.0];
        // Token 3, the model's favourite, is not allowed here.
        let got = logprobs_of(&logits, Some(&[true, true, true, false, true, true]));
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), [0, 5, 2]);
        assert!((got.iter().map(|g| g.1.exp()).sum::<f64>() - 1.0).abs() < 1e-12);
        // A forced token is certain, and a mask shorter than the logits allows nothing past its end.
        assert_eq!(logprobs_of(&logits, Some(&[false, false, true])), [(2, 0.0)]);
    }

    /// A grammar over the tokens `a`..`e` (ids 0..5), with id 5 ending generation.
    fn grammar(src: &str) -> GrammarProcessor {
        let bytes = (0..5u8).map(|t| vec![b'a' + t]).chain([b"<end>".to_vec()]).collect();
        let vocab = TokenVocab::new(bytes, vec![false, false, false, false, false, true], &[5]);
        GrammarProcessor::new(Arc::new(Grammar::parse(src).unwrap()), Arc::new(vocab))
    }

    fn bytes_of(t: u32) -> Vec<u8> { vec![b'a' + t as u8] }

    #[test]
    fn the_observer_reports_the_constrained_distribution_and_leaves_the_logits_alone() {
        // First `a`, `c` or `e`; then only `c`; then the end.
        let (lp, processor) = attach(&json!({"logprobs": true, "top_logprobs": 2}), true,
            Some(grammar("root ::= [ace] \"c\""))).unwrap();
        let (mut lp, mut observer) = (lp.unwrap(), processor.unwrap());
        let mut logits = vec![1.0f32, 8.0, 2.0, 7.0, 0.0, 9.0];
        let before = logits.clone();
        observer.process(&mut logits);
        assert_eq!(logits, before);
        assert!(observer.allows(2) && !observer.allows(1) && !observer.allows(5));
        observer.accept(2);
        assert!(!observer.finished());

        let v = lp.token(bytes_of, 2);
        let entry = &v["content"][0];
        let z = 1f64.exp() + 2f64.exp() + 1.0;
        assert_eq!((entry["token"].as_str(), entry["bytes"].clone()), (Some("c"), json!([b'c'])));
        assert!((entry["logprob"].as_f64().unwrap() - (2.0 - z.ln())).abs() < 1e-12);
        let top = entry["top_logprobs"].as_array().unwrap();
        assert_eq!(top.iter().map(|t| t["token"].as_str().unwrap()).collect::<Vec<_>>(), ["c", "a"]);
        assert_eq!(top[0]["logprob"], entry["logprob"]);
        assert_eq!(top[1]["bytes"], json!([b'a']));

        // The grammar forces `c`, whatever the model prefers.
        observer.process(&mut logits);
        observer.accept(2);
        assert!(observer.finished());
        let v = lp.token(bytes_of, 2);
        assert_eq!(v["content"][0]["logprob"], json!(0.0));
        assert_eq!(v["content"][0]["top_logprobs"].as_array().unwrap().len(), 1);

        // A step the client never sees (the end id) does not leak into a later token.
        observer.process(&mut logits);
        observer.accept(5);
        assert!(lp.token(bytes_of, 2)["content"][0]["logprob"].is_null());
        assert_eq!(lp.whole()["content"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn completion_routes_report_parallel_lists() {
        let (lp, processor) = attach(&json!({"logprobs": 2}), false, None).unwrap();
        let (mut lp, mut observer) = (lp.unwrap(), processor.unwrap());
        for (logits, chosen) in [([0.0f32, 3.0, 1.0], 1u32), ([2.0, 0.0, 0.0], 0)] {
            observer.process(&mut logits.clone());
            assert!(observer.allows(chosen));
            observer.accept(chosen);
            let chunk = lp.token(|t| if t == 1 { "é!".as_bytes().to_vec() } else { bytes_of(t) }, chosen);
            assert_eq!(chunk["tokens"].as_array().unwrap().len(), 1);
        }
        let v = lp.whole();
        assert_eq!(v["tokens"], json!(["é!", "a"]));
        assert_eq!(v["text_offset"], json!([0, 2]), "offsets count characters");
        let z = (1.0 + 3f64.exp() + 1f64.exp()).ln();
        assert!((v["token_logprobs"][0].as_f64().unwrap() - (3.0 - z)).abs() < 1e-12);
        let top = v["top_logprobs"][0].as_object().unwrap();
        assert_eq!(top.len(), 2);
        assert!((top["é!"].as_f64().unwrap() - (3.0 - z)).abs() < 1e-12 && (top["c"].as_f64().unwrap() - (1.0 - z)).abs() < 1e-12);

        let (lp, processor) = attach(&json!({"logprobs": 0}), false, None).unwrap();
        let (mut lp, mut observer) = (lp.unwrap(), processor.unwrap());
        observer.process(&mut [0.0, 1.0]);
        observer.accept(1);
        let v = lp.token(bytes_of, 1);
        assert!(v["top_logprobs"].is_null() && v["token_logprobs"][0].is_number());
    }

    #[test]
    fn a_token_whose_step_was_not_observed_has_no_number() {
        let (lp, _observer) = attach(&json!({"logprobs": true}), true, None).unwrap();
        let v = lp.unwrap().token(bytes_of, 3);
        assert!(v["content"][0]["logprob"].is_null());
    }

    #[test]
    fn request_fields_are_read_per_route() {
        let top = |body: Value, chat: bool| requested(&body, chat);
        assert_eq!(top(json!({}), true).unwrap(), None);
        assert_eq!(top(json!({"logprobs": false}), true).unwrap(), None);
        assert_eq!(top(json!({"logprobs": null, "top_logprobs": null}), true).unwrap(), None);
        assert_eq!(top(json!({"logprobs": true}), true).unwrap(), Some(0));
        assert_eq!(top(json!({"logprobs": true, "top_logprobs": 20}), true).unwrap(), Some(20));
        assert!(top(json!({"logprobs": true, "top_logprobs": 21}), true).is_err());
        assert!(top(json!({"logprobs": true, "top_logprobs": -1}), true).is_err());
        assert!(top(json!({"top_logprobs": 3}), true).is_err());
        assert!(top(json!({"logprobs": 3}), true).is_err());
        assert_eq!(top(json!({}), false).unwrap(), None);
        assert_eq!(top(json!({"logprobs": 5}), false).unwrap(), Some(5));
        assert_eq!(top(json!({"logprobs": 0}), false).unwrap(), Some(0));
        assert_eq!(top(json!({"logprobs": true}), false).unwrap(), Some(0));
        assert!(top(json!({"logprobs": 21}), false).is_err());
        assert!(top(json!({"logprobs": "yes"}), false).is_err());
        // Without the fields a request runs with its constraint alone, or with nothing.
        let (lp, processor) = attach(&json!({}), true, None).unwrap();
        assert!(lp.is_none() && processor.is_none());
        let (lp, processor) = attach(&json!({}), true, Some(grammar("root ::= \"a\""))).unwrap();
        assert!(lp.is_none() && processor.is_some());
    }

    #[test]
    fn log_probabilities_sit_in_the_choice_or_at_the_top_level() {
        let mut oai = json!({"choices": [{"index": 0}]});
        put(&mut oai, true, json!({"content": []}));
        assert_eq!(oai["choices"][0]["logprobs"], json!({"content": []}));
        let mut native = json!({"content": "x"});
        put(&mut native, false, json!({"tokens": []}));
        assert_eq!(native["logprobs"], json!({"tokens": []}));
    }
}
