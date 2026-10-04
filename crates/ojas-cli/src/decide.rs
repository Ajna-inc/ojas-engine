//! `ojas decide`: typed decisions with a decision model, on Metal or CUDA.
//!
//! ```text
//! ojas decide model.gguf --state '{"subject": "...", "body": "..."}' \
//!     --questions '{"urgency": {"type": "score", "instructions": "How urgent?",
//!                               "criteria": ["not urgent", "soon", "blocking"]}}'
//! ```
//!
//! The state is JSON (an object or array) or plain text; anything that does not
//! parse as a JSON object or array is taken as text. `--image file` adds an image,
//! for a model that takes them. `--json` prints the response body:
//! `{"model", "answers", "usage"}`. `--requests-file F` answers a file of request
//! bodies instead, one per line, in batches, one response per output line.
//!
//! `ojas serve model.gguf` serves the same through `POST /v1/systemone` (also
//! `/v1/decide` and `/decide`), whose body is `{"state": ..., "questions": {...}}`
//! with optional `"images"` as data URLs.
//! Requests that arrive together are answered together; the body is read with the
//! order-preserving JSON reader, since a choice question's option order is part of
//! the model's input.

// Without a GPU backend in the build every entry point refuses before loading, and the
// code past the refusal is never reached.
#![cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code, unused_variables, unreachable_code))]

use crate::backend::Device;
use crate::flags::RunOpts;
use crate::serve::{send, send_err, send_json, serve_http_batches};
use anyhow::{Context, Result};
use ojas_decision::json::Json;
use ojas_decision::{Decision, DecisionGpu, DecisionModel, Image, InvalidRequest, QuestionKind, Request, Unsupported};

/// The GPU backends a decision model can run on in this build.
#[derive(Clone, Copy, Debug)]
enum Backend {
    #[cfg(target_os = "macos")]
    Metal,
    #[cfg(feature = "cuda")]
    Cuda,
}

/// The backend `--device` asks for: Metal on macOS, CUDA when built with the `cuda`
/// feature; `auto` prefers Metal, then CUDA. There is no CPU decision path.
fn pick(device: Device) -> Result<Backend> {
    let want = match device {
        Device::Auto => "auto",
        Device::Metal => "metal",
        Device::Cpu => anyhow::bail!("decision models run on a GPU (Metal or CUDA), not on the CPU"),
        Device::Cuda => "cuda",
    };
    #[cfg(target_os = "macos")]
    if matches!(want, "auto" | "metal") { return Ok(Backend::Metal); }
    #[cfg(feature = "cuda")]
    if matches!(want, "auto" | "cuda") { return Ok(Backend::Cuda); }
    anyhow::bail!("decision models need a Metal GPU or a build with --features cuda (asked for --device {want})")
}

/// Load the decision model at `$path` on the backend `$device` names and run `$body`
/// with it. A macro rather than a function: the model's type carries the backend, and
/// each backend is its own arm so the others need not be compiled in.
macro_rules! with_decision_model {
    ($path:expr, $device:expr, |$model:ident| $body:expr) => {{
        let path: &str = $path;
        match pick($device)? {
            #[cfg(target_os = "macos")]
            Backend::Metal => {
                let gpu = ojas_metal::MetalGpu::new().context("opening the Metal GPU")?;
                let backend = ojas_models::decision_backend::MetalDecision(&gpu);
                let $model = DecisionModel::load(&backend, path)?;
                $body
            }
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                let gpu = ojas_cuda::CudaDecision::new(0).context("opening the CUDA device")?;
                let $model = DecisionModel::load(&gpu, path)?;
                $body
            }
        }
    }};
}

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

/// A request from the state text, the questions text and image files.
fn request<G: DecisionGpu>(model: &DecisionModel<G>, state: &str, questions: &str, images: &[String]) -> Result<Request> {
    let questions = Json::parse(questions).context("--questions is not valid JSON")?;
    let body = Json::Object(vec![("state".into(), parse_state(state)), ("questions".into(), questions)]);
    let mut req = model.request(&body)?;
    for path in images {
        let bytes = std::fs::read(path).with_context(|| format!("reading image {path}"))?;
        req.images.push(Image::decode(&bytes).with_context(|| format!("decoding image {path}"))?);
    }
    model.validate(&req)?;
    Ok(req)
}

fn print_table(d: &Decision, wall_ms: f64) {
    for a in &d.answers {
        let head = match a.kind {
            QuestionKind::Choice => format!("choice {}", a.choice()),
            QuestionKind::Score => format!("score {:.3}", a.score()),
            QuestionKind::Noul => format!("noul p(true) = {:.4}", a.noul()),
        };
        match a.kind {
            QuestionKind::Noul => println!("[{}] {head}", a.id),
            _ => println!("[{}] {head}   confidence {:.3}", a.id, a.confidence()),
        }
        for (key, p) in a.keys.iter().zip(&a.probabilities) {
            let bar = "#".repeat((p * 24.0).round() as usize);
            println!("    {:<24} {:<24} {p:.4}", truncate(key, 24), bar);
        }
    }
    println!("{} questions, {} tokens, {wall_ms:.1} ms wall / {:.1} ms gpu",
        d.answers.len(), d.input_tokens, d.gpu_s * 1e3);
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { s.chars().take(n - 1).chain(['…']).collect() }
}

/// The answers to a file of request bodies, one JSON object per line: each line's
/// response (or `{"error": ...}`) on its own output line, in order. Requests are
/// answered `MAX_BATCH` at a time, as `serve` answers requests that arrive together.
fn decide_file<G: DecisionGpu>(model: &DecisionModel<G>, path: &str) -> Result<()> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading requests file {path}"))?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let t = std::time::Instant::now();
    let (mut answered, mut refused) = (0usize, 0usize);
    for chunk in lines.chunks(MAX_BATCH) {
        let parsed: Vec<Result<Request, Refusal>> = chunk.iter().map(|l| parse_body(model, l.as_bytes())).collect();
        let ok: Vec<&Request> = parsed.iter().filter_map(|r| r.as_ref().ok()).collect();
        let mut decided = model.decide_all(&ok).into_iter();
        for p in &parsed {
            let line = match p {
                Err((_, msg)) => Err(msg.clone()),
                Ok(_) => decided.next().expect("one result per request").map_err(|e| refusal(e).1),
            };
            match line {
                Ok(d) => { answered += 1; println!("{}", d.to_json(model.name()).to_python(false)); }
                Err(msg) => {
                    refused += 1;
                    println!("{}", Json::Object(vec![("error".into(), Json::Str(msg))]).to_python(false));
                }
            }
        }
    }
    eprintln!("{answered} answered, {refused} refused, in {:.1} s", t.elapsed().as_secs_f64());
    Ok(())
}

pub fn decide(path: &str, opts: &RunOpts) -> Result<()> {
    if let Some(requests) = &opts.requests_file {
        return with_decision_model!(path, opts.device, |model| decide_file(&model, requests));
    }
    let state = read(&opts.state, &opts.state_file, "state")?;
    let questions = read(&opts.questions, &opts.questions_file, "questions")?;
    with_decision_model!(path, opts.device, |model| {
        let req = request(&model, &state, &questions, &opts.images)?;
        let t = std::time::Instant::now();
        let decision = model.decide(&req)?;
        let wall_ms = t.elapsed().as_secs_f64() * 1e3;
        if opts.json {
            println!("{}", decision.to_json(model.name()).to_python(false));
        } else {
            print_table(&decision, wall_ms);
        }
        Ok(())
    })
}

/// Whether `path` is a decision model, read from the header alone.
pub fn is_decision_model(path: &str) -> bool { ojas_decision::decision_type(path).is_some() }

/// An error reply: the status and the message.
type Refusal = (&'static str, String);

fn refusal(e: anyhow::Error) -> Refusal {
    let status = if e.downcast_ref::<InvalidRequest>().is_some() { "400 Bad Request" }
                 else if e.downcast_ref::<Unsupported>().is_some() { "501 Not Implemented" }
                 else { "500 Internal Server Error" };
    (status, format!("{e:#}"))
}

/// A request body, parsed for `model`.
fn parse_body<G: DecisionGpu>(model: &DecisionModel<G>, body: &[u8]) -> Result<Request, Refusal> {
    let body = std::str::from_utf8(body).ok().and_then(|text| Json::parse(text).ok())
        .ok_or(("400 Bad Request", "the request body is not valid JSON".to_string()))?;
    model.request(&body).map_err(refusal)
}

/// Write a reply on its own thread, so a client slow to read holds up no one.
fn reply(mut stream: std::net::TcpStream, result: Result<String, Refusal>) {
    std::thread::spawn(move || match result {
        Ok(body) => send(&mut stream, "200 OK", "application/json", body.as_bytes()),
        Err((status, msg)) => send_err(&mut stream, status, &msg),
    });
}

/// Most requests answered together.
const MAX_BATCH: usize = 64;

/// Counters `GET /metrics` reports, in the Prometheus text format.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Metrics {
    /// Requests answered.
    requests: u64,
    /// Requests refused or failed.
    errors: u64,
    /// Prompt tokens computed.
    prompt_tokens: u64,
    /// Prompt tokens reused from a shared prefix rather than computed.
    prompt_tokens_cached: u64,
    /// GPU time spent answering, seconds.
    gpu_seconds: f64,
}

impl Metrics {
    fn record(&mut self, d: &Decision) {
        self.requests += 1;
        self.prompt_tokens += (d.input_tokens - d.cached_tokens) as u64;
        self.prompt_tokens_cached += d.cached_tokens as u64;
        self.gpu_seconds += d.gpu_s;
    }

    fn render(&self) -> String {
        let counters: [(&str, &str, String); 5] = [
            ("requests_total", "Decision requests answered.", self.requests.to_string()),
            ("request_errors_total", "Decision requests refused or failed.", self.errors.to_string()),
            ("prompt_tokens_total", "Prompt tokens computed.", self.prompt_tokens.to_string()),
            ("prompt_tokens_cached_total", "Prompt tokens reused from a shared prefix.", self.prompt_tokens_cached.to_string()),
            ("gpu_seconds_total", "GPU time spent answering, in seconds.", format!("{:.6}", self.gpu_seconds)),
        ];
        counters.iter().map(|(name, help, value)| {
            format!("# HELP ojas_decision_{name} {help}\n# TYPE ojas_decision_{name} counter\nojas_decision_{name} {value}\n")
        }).collect()
    }
}

/// `ojas serve` for a decision model. Requests that arrive while others are being
/// answered wait, and are then answered together: an encoder model runs all of
/// their questions in shared GPU passes.
pub fn serve(path: &str, opts: &RunOpts) -> Result<()> {
    let t = std::time::Instant::now();
    with_decision_model!(path, opts.device, |model| serve_model(&model, opts, t.elapsed().as_secs_f64()))
}

fn serve_model<G: DecisionGpu>(model: &DecisionModel<G>, opts: &RunOpts, load_s: f64) -> Result<()> {
    let addr = format!("{}:{}", opts.host, opts.port);
    let listener = std::net::TcpListener::bind(&addr).with_context(|| format!("binding {addr}"))?;
    eprintln!("  {} | decision model | {} | loaded in {load_s:.1}s", model.name(), G::NAME);
    eprintln!("  listening on http://{addr}  (up to {MAX_BATCH} requests answered together)");
    eprintln!("  POST /v1/systemone  GET /health  GET /v1/models  GET /props  GET /metrics");
    let props = serde_json::json!({
        "model": model.name(), "decision_type": model.kind(), "max_options": model.max_options(),
        "images": model.takes_images(), "max_images": ojas_decision::MAX_IMAGES, "max_batch": MAX_BATCH,
    });
    let mut metrics = Metrics::default();
    serve_http_batches(&listener, MAX_BATCH, |batch| {
        let mut pending: Vec<(std::net::TcpStream, Request)> = Vec::new();
        for (mut stream, req, route) in batch {
            match (req.method.as_str(), route.as_str()) {
                ("POST", "/v1/systemone" | "/v1/decide" | "/decide") => match parse_body(model, &req.body) {
                    Ok(r) => pending.push((stream, r)),
                    Err(refused) => {
                        metrics.errors += 1;
                        reply(stream, Err(refused));
                    }
                },
                ("GET", "/metrics") => send(&mut stream, "200 OK", "text/plain; version=0.0.4", metrics.render().as_bytes()),
                ("GET", "/props") => send_json(&mut stream, "200 OK", &props),
                ("GET", "/health") => match ojas_core::device_fault::peek() {
                    Some(err) => send_json(&mut stream, "503 Service Unavailable",
                        &serde_json::json!({"status": "error", "error": err.to_string()})),
                    None => send_json(&mut stream, "200 OK", &serde_json::json!({"status": "ok"})),
                },
                ("GET", "/v1/models") => send_json(&mut stream, "200 OK", &serde_json::json!({
                    "object": "list", "data": [{"id": model.name(), "object": "model", "owned_by": "ojas"}]
                })),
                _ => send_err(&mut stream, "404 Not Found", &format!("no route for {} {}", req.method, route)),
            }
        }
        if pending.is_empty() { return; }
        let results = model.decide_all(&pending.iter().map(|(_, r)| r).collect::<Vec<_>>());
        for ((stream, _), result) in pending.into_iter().zip(results) {
            match &result {
                Ok(d) => metrics.record(d),
                Err(_) => metrics.errors += 1,
            }
            reply(stream, result.map(|d| d.to_json(model.name()).to_python(false)).map_err(refusal));
        }
    })
}

/// `ojas bench` for a decision model: latency over `-r` repetitions after a warmup,
/// wall time (rendering, tokenization, GPU pass, host scorer) and GPU time. The state
/// and the questions each default to `decision::BENCH_REQUEST`'s when their flags
/// are absent.
pub fn bench(path: &str, opts: &RunOpts) -> Result<()> {
    let builtin = Json::parse(ojas_decision::BENCH_REQUEST).context("the built-in bench request")?;
    let field = |k: &str| builtin.get(k).map(|v| v.to_python(false)).unwrap_or_default();
    let state = read_or(&opts.state, &opts.state_file, "state", &field("state"))?;
    let questions = read_or(&opts.questions, &opts.questions_file, "questions", &field("questions"))?;
    with_decision_model!(path, opts.device, |model| bench_model(&model, opts, &state, &questions))
}

fn bench_model<G: DecisionGpu>(model: &DecisionModel<G>, opts: &RunOpts, state: &str, questions: &str) -> Result<()> {
    let req = request(model, state, questions, &opts.images)?;
    eprintln!("  warmup...");
    for _ in 0..3 { model.decide(&req)?; }
    let (mut wall, mut dev, mut tokens) = (Vec::new(), Vec::new(), 0usize);
    for r in 0..opts.reps.max(1) {
        let t = std::time::Instant::now();
        let d = model.decide(&req)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        tokens = d.input_tokens;
        eprintln!("  rep {}: {ms:.2} ms wall, {:.2} ms gpu", r + 1, d.gpu_s * 1e3);
        wall.push(ms);
        dev.push(d.gpu_s * 1e3);
    }
    let stats = |mut v: Vec<f64>| { v.sort_by(f64::total_cmp); (v[v.len() / 2], v[0]) };
    let ((w50, wmin), (g50, gmin)) = (stats(wall), stats(dev));
    println!("\n{} | {} | {} questions, {tokens} tokens | {} reps\n  wall median {w50:.2} ms  best {wmin:.2} ms  \
              ({:.1} decisions/s)\n  gpu  median {g50:.2} ms  best {gmin:.2} ms",
        model.name(), G::NAME, req.questions.len(), opts.reps.max(1), 1e3 / w50);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_count_computed_and_reused_tokens_apart() {
        let mut m = Metrics::default();
        m.record(&Decision { answers: Vec::new(), input_tokens: 300, cached_tokens: 120, gpu_s: 0.5 });
        m.errors += 1;
        let text = m.render();
        for line in ["ojas_decision_requests_total 1", "ojas_decision_request_errors_total 1",
                     "ojas_decision_prompt_tokens_total 180", "ojas_decision_prompt_tokens_cached_total 120",
                     "ojas_decision_gpu_seconds_total 0.500000", "# TYPE ojas_decision_requests_total counter"] {
            assert!(text.lines().any(|l| l == line), "missing {line:?} in\n{text}");
        }
    }

    #[test]
    fn refusals_carry_the_status_of_their_cause() {
        assert_eq!(refusal(InvalidRequest("bad".into()).into()).0, "400 Bad Request");
        assert_eq!(refusal(Unsupported("no images".into()).into()).0, "501 Not Implemented");
        assert_eq!(refusal(anyhow::anyhow!("device lost")).0, "500 Internal Server Error");
        let wrapped = anyhow::Error::from(InvalidRequest("bad".into())).context("questions.q");
        assert_eq!(refusal(wrapped), ("400 Bad Request", "questions.q: bad".to_string()));
    }
}
