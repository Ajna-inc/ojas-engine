//! `ojas decide`: typed decisions with a Laya model (Metal).
//!
//! ```text
//! ojas decide laya.gguf --state '{"subject": "...", "body": "..."}' \
//!     --questions '{"urgency": {"type": "score", "instructions": "How urgent?",
//!                               "criteria": ["not urgent", "soon", "blocking"]}}'
//! ```
//!
//! The state is JSON (an object or array) or plain text; anything that does not
//! parse as a JSON object or array is taken as text. `--json` prints the response in
//! the shape of the reference `laya` package: `{"model", "answers", "usage"}`, with
//! probabilities rounded to four decimals as it rounds them.
//!
//! `ojas serve laya.gguf` serves the same through `POST /v1/decide` (also `/decide`),
//! whose body is `{"state": ..., "questions": {...}}` and whose response is the
//! `--json` output. Requests are answered one at a time, as `ojas serve` answers
//! completions; the body is read with the order-preserving JSON reader, since a
//! choice question's option order is part of the model's input.

use crate::flags::RunOpts;
use crate::serve::{send, send_err, send_json, serve_http};
use anyhow::{Context, Result};
use ojas_models::laya::json::Json;
use ojas_models::laya::{Answer, Decision, Laya, Question, QuestionKind};

fn read(inline: &Option<String>, file: &Option<String>, what: &str) -> Result<String> {
    if let Some(f) = file {
        return std::fs::read_to_string(f).with_context(|| format!("reading {what} file {f}"));
    }
    inline.clone().with_context(|| format!("no {what}: pass --{what} or --{what}-file"))
}

/// `read`, or `default` when neither the inline flag nor the file flag was given. A
/// flag that was given and cannot be read is an error, not a fallback.
fn read_or(inline: &Option<String>, file: &Option<String>, what: &str, default: &str) -> Result<String> {
    if inline.is_none() && file.is_none() { return Ok(default.to_string()); }
    read(inline, file, what)
}

/// A JSON object or array is the reference's structured state; anything else is
/// plain text, exactly as given.
fn parse_state(text: &str) -> Json {
    match Json::parse(text) {
        Ok(v @ (Json::Object(_) | Json::Array(_))) => v,
        _ => Json::Str(text.to_string()),
    }
}

fn parse_questions(text: &str) -> Result<Vec<Question>> {
    let q = Json::parse(text).context("--questions is not valid JSON")?;
    let entries = q.as_object().context("--questions must be a JSON object of {id: question}")?;
    anyhow::ensure!(!entries.is_empty(), "--questions is empty");
    entries.iter().map(|(id, v)| Question::from_json(id, v)).collect()
}

fn round4(x: f32) -> Json { Json::Float((x as f64 * 1e4).round() / 1e4) }

fn answer_json(a: &Answer) -> Json {
    let probs = Json::Object(a.labels.iter().cloned().zip(a.probabilities.iter().map(|&p| round4(p))).collect());
    let ext = Json::Object(vec![("act_probability".into(), round4(a.act_probability))]);
    let mut kv: Vec<(String, Json)> = vec![("type".into(), Json::Str(a.kind.name().into()))];
    match a.kind {
        QuestionKind::Choice => {
            kv.push(("choice".into(), Json::Str(a.choice().into())));
            kv.push(("probabilities".into(), probs));
            kv.push(("confidence".into(), round4(a.confidence)));
        }
        QuestionKind::Score => {
            kv.push(("score".into(), round4(a.expected_level())));
            kv.push(("probabilities".into(), probs));
            kv.push(("confidence".into(), round4(a.confidence)));
        }
        QuestionKind::Noul => kv.push(("noul".into(), round4(a.p_true()))),
    }
    kv.push(("rl_agent".into(), ext));
    Json::Object(kv)
}

/// The reference package's response shape: `{"model", "answers", "usage"}`.
fn response_json(d: &Decision) -> Json {
    let answers = Json::Object(d.answers.iter().map(|a| (a.id.clone(), answer_json(a))).collect());
    let usage = Json::Object(vec![
        ("input_tokens".into(), Json::Int(d.input_tokens.to_string())),
        ("output_tokens".into(), Json::Int("0".into())),
    ]);
    Json::Object(vec![("model".into(), Json::Str("laya".into())), ("answers".into(), answers), ("usage".into(), usage)])
}

fn print_table(d: &Decision, wall_ms: f64) {
    for a in &d.answers {
        let head = match a.kind {
            QuestionKind::Choice => format!("choice {}", a.choice()),
            QuestionKind::Score => format!("score {:.3}", a.expected_level()),
            QuestionKind::Noul => format!("noul p(true) = {:.4}", a.p_true()),
        };
        println!("[{}] {head}   confidence {:.3}   act {:.3}", a.id, a.confidence, a.act_probability);
        for (label, p) in a.labels.iter().zip(&a.probabilities) {
            let bar = "#".repeat((p * 24.0).round() as usize);
            println!("    {:<24} {:<24} {p:.4}", truncate(label, 24), bar);
        }
    }
    println!("{} questions, {} tokens, {wall_ms:.1} ms wall / {:.1} ms gpu",
        d.answers.len(), d.input_tokens, d.gpu_s * 1e3);
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { s.chars().take(n - 1).chain(['…']).collect() }
}

pub fn decide(model: &str, opts: &RunOpts) -> Result<()> {
    let state = parse_state(&read(&opts.state, &opts.state_file, "state")?);
    let questions = parse_questions(&read(&opts.questions, &opts.questions_file, "questions")?)?;
    let gpu = ojas_metal::MetalGpu::new().context("`decide` needs a Metal GPU")?;
    let laya = Laya::load(&gpu, model)?;

    let t = std::time::Instant::now();
    let decision = laya.decide(&state, &questions)?;
    let wall_ms = t.elapsed().as_secs_f64() * 1e3;
    if opts.json {
        println!("{}", response_json(&decision).to_python(false));
    } else {
        print_table(&decision, wall_ms);
    }
    Ok(())
}

/// Whether `path` is a Laya GGUF (a `modern-bert` encoder carrying `laya.head.*`),
/// read from the header alone.
pub fn is_laya(path: &str) -> bool {
    ojas_formats::gguf::Gguf::open(path)
        .map(|g| g.arch() == "modern-bert" && g.meta_u32("laya.head.block_count").is_some())
        .unwrap_or(false)
}

/// One `POST /v1/decide` body: the state and the questions.
fn decide_request(body: &[u8]) -> Result<(Json, Vec<Question>)> {
    let text = std::str::from_utf8(body).context("request body is not UTF-8")?;
    let req = Json::parse(text).context("request body is not valid JSON")?;
    let state = match req.get("state") {
        Some(Json::Str(s)) => parse_state(s),
        Some(v) => v.clone(),
        None => anyhow::bail!("request has no \"state\""),
    };
    let questions = req.get("questions").and_then(Json::as_object)
        .context("request needs \"questions\" as an object of {id: question}")?;
    anyhow::ensure!(!questions.is_empty(), "\"questions\" is empty");
    Ok((state, questions.iter().map(|(id, q)| Question::from_json(id, q)).collect::<Result<_>>()?))
}

/// `ojas serve` for a Laya model.
pub fn serve(model: &str, opts: &RunOpts) -> Result<()> {
    let gpu = ojas_metal::MetalGpu::new().context("serving a Laya model needs a Metal GPU")?;
    let t = std::time::Instant::now();
    let laya = Laya::load(&gpu, model)?;
    let addr = format!("{}:{}", opts.host, opts.port);
    let listener = std::net::TcpListener::bind(&addr).with_context(|| format!("binding {addr}"))?;
    eprintln!("  laya | metal | max_len {} | loaded in {:.1}s", laya.max_len(), t.elapsed().as_secs_f64());
    eprintln!("  listening on http://{addr}  (one request at a time)");
    eprintln!("  POST /v1/decide  GET /health  GET /v1/models");
    serve_http(&listener, |stream, req, path| match (req.method.as_str(), path) {
        ("GET", "/health") => match ojas_core::device_fault::peek() {
            Some(err) => send_json(stream, "503 Service Unavailable",
                &serde_json::json!({"status": "error", "error": err.to_string()})),
            None => send_json(stream, "200 OK", &serde_json::json!({"status": "ok"})),
        },
        ("GET", "/v1/models") => send_json(stream, "200 OK", &serde_json::json!({
            "object": "list", "data": [{"id": "laya", "object": "model", "owned_by": "ojas"}]
        })),
        ("POST", "/v1/decide" | "/decide") => match decide_request(&req.body) {
            Err(e) => send_err(stream, "400 Bad Request", &format!("{e:#}")),
            Ok((state, questions)) => match laya.decide(&state, &questions) {
                Ok(d) => send(stream, "200 OK", "application/json", response_json(&d).to_python(false).as_bytes()),
                Err(e) => send_err(stream, "400 Bad Request", &format!("{e:#}")),
            },
        },
        _ => send_err(stream, "404 Not Found", &format!("no route for {} {}", req.method, path)),
    })
}

/// `ojas bench` for a Laya model: decision latency over `-r` repetitions after a
/// warmup, wall time (tokenization, GPU pass, host head) and GPU time. The state and
/// the questions each default to `laya::BENCH_REQUEST`'s when their flags are absent.
pub fn bench(model: &str, opts: &RunOpts) -> Result<()> {
    let builtin = Json::parse(ojas_models::laya::BENCH_REQUEST).context("the built-in bench request")?;
    let field = |k: &str| builtin.get(k).map(|v| v.to_python(false)).unwrap_or_default();
    let state = parse_state(&read_or(&opts.state, &opts.state_file, "state", &field("state"))?);
    let questions = parse_questions(&read_or(&opts.questions, &opts.questions_file, "questions", &field("questions"))?)?;
    let gpu = ojas_metal::MetalGpu::new().context("`bench` on a Laya model needs a Metal GPU")?;
    let laya = Laya::load(&gpu, model)?;
    eprintln!("  warmup...");
    for _ in 0..3 { laya.decide(&state, &questions)?; }
    let (mut wall, mut dev, mut tokens) = (Vec::new(), Vec::new(), 0usize);
    for r in 0..opts.reps.max(1) {
        let t = std::time::Instant::now();
        let d = laya.decide(&state, &questions)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        tokens = d.input_tokens;
        eprintln!("  rep {}: {ms:.2} ms wall, {:.2} ms gpu", r + 1, d.gpu_s * 1e3);
        wall.push(ms);
        dev.push(d.gpu_s * 1e3);
    }
    let stats = |mut v: Vec<f64>| { v.sort_by(f64::total_cmp); (v[v.len() / 2], v[0]) };
    let ((w50, wmin), (g50, gmin)) = (stats(wall), stats(dev));
    println!("\nlaya | metal | {} questions, {tokens} tokens | {} reps\n  wall median {w50:.2} ms  best {wmin:.2} ms  \
              ({:.1} decisions/s)\n  gpu  median {g50:.2} ms  best {gmin:.2} ms",
        questions.len(), opts.reps.max(1), 1e3 / w50);
    Ok(())
}
