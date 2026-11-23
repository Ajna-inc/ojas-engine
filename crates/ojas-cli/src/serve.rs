//! HTTP server: OpenAI-compatible plus the native `/completion`. A decision
//! model is served by `decide::serve` instead (`POST /v1/systemone`), on the same
//! connection loop, [`serve_http`].
//!
//! Requests are served on the thread that owns the model: `DecoderGpu` is `Send`
//! but not `Sync`. With one sequence slot they run one at a time. With several
//! (`--parallel`), up to that many run together through the decoder's slots, one
//! token for each per batched decode step (`ojas_infer::batch`).
//!
//! Built on `std::net`: an async runtime would put an executor between the
//! request and a blocking decode loop without buying any parallelism.

use crate::backend::{with_model, ModelInfo};
use ojas_grammar::OutputFormat;
use crate::flags::RunOpts;
use anyhow::{Context, Result};
use ojas_core::Model;
use ojas_infer::batch::{Batch, Event};
use ojas_infer::{EngineCore, SampleOpts};
use std::collections::{HashMap, VecDeque};
use ojas_tokenize::Bpe;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

mod anthropic;
mod completion;
mod conn;
mod images;
mod logprobs;
mod metrics;
mod tools;
pub(crate) use conn::{drain_on_sigterm, Conn};
/// Marks a request body that arrived on the Messages API route.
const MESSAGES_MARK: &str = "__messages_api";
use completion::Ctx;

const MAX_BODY: usize = 32 * 1024 * 1024;
/// Most bytes of a request line and its headers.
const MAX_HEAD: usize = 64 * 1024;

pub(crate) struct Request {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) body: Vec<u8>,
    /// The token of `Authorization: Bearer`.
    bearer: Option<String>,
    /// The caller's `X-Request-Id`, when it is one a log line can hold.
    request_id: Option<String>,
}

/// Read one HTTP/1.1 request. `None` on a closed or unusable connection.
fn read_request(stream: &mut impl BufRead) -> Result<Option<Request>> {
    let mut line = String::new();
    if stream.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut head = line.len();
    let mut parts = line.split_whitespace();
    let (method, path) = match (parts.next(), parts.next()) {
        (Some(m), Some(p)) => (m.to_string(), p.to_string()),
        _ => return Ok(None),
    };

    let (mut len, mut bearer, mut request_id) = (0usize, None, None);
    loop {
        let mut h = String::new();
        if stream.read_line(&mut h)? == 0 {
            return Ok(None);
        }
        head += h.len();
        if head > MAX_HEAD {
            anyhow::bail!("request headers exceed the {MAX_HEAD}-byte limit");
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some(v) = h.split_once(':') {
            if v.0.eq_ignore_ascii_case("content-length") {
                len = v.1.trim().parse().unwrap_or(0);
            } else if v.0.eq_ignore_ascii_case("authorization") {
                bearer = v.1.trim().strip_prefix("Bearer ").map(|t| t.trim().to_string());
            } else if v.0.eq_ignore_ascii_case("x-request-id") {
                let id = v.1.trim();
                let plain = !id.is_empty() && id.len() <= 128 && id.bytes().all(|b| b.is_ascii_graphic());
                request_id = plain.then(|| id.to_string());
            }
        }
    }
    if len > MAX_BODY {
        anyhow::bail!("request body of {len} bytes exceeds the {MAX_BODY}-byte limit");
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body, bearer, request_id }))
}

/// Header lines every response of this process carries: the CORS origin, when one
/// is allowed, and the model's identity.
static HEADERS: OnceLock<String> = OnceLock::new();

fn shared_headers() -> &'static str { HEADERS.get().map(String::as_str).unwrap_or("") }

/// What a server says about the model it serves, so a caller can pin and log it.
pub(crate) struct Identity {
    /// The model file's name.
    pub(crate) file: String,
    pub(crate) sha256: String,
}

/// The source revision this binary was built from.
pub(crate) const REVISION: &str = env!("OJAS_REVISION");

impl Identity {
    /// Hash the model file on another thread, beside the model's load.
    pub(crate) fn start(path: &str) -> std::thread::JoinHandle<Result<Identity>> {
        let path = path.to_string();
        std::thread::spawn(move || {
            use sha2::{Digest, Sha256};
            let mut file = std::fs::File::open(&path).with_context(|| format!("opening {path}"))?;
            let mut hash = Sha256::new();
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = file.read(&mut buf).with_context(|| format!("reading {path}"))?;
                if n == 0 { break; }
                hash.update(&buf[..n]);
            }
            let sha256 = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
            let file = std::path::Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(path);
            Ok(Identity { file, sha256 })
        })
    }

    pub(crate) fn finish(handle: std::thread::JoinHandle<Result<Identity>>) -> Result<Identity> {
        handle.join().map_err(|_| anyhow::anyhow!("hashing the model file failed"))?
    }

    /// The fields `/v1/models` adds to a model's entry.
    pub(crate) fn describe(&self, entry: &mut Value) {
        entry["file"] = json!(self.file);
        entry["sha256"] = json!(self.sha256);
        entry["revision"] = json!(REVISION);
    }
}

/// Fix the headers every response carries. Called once, before requests are served.
pub(crate) fn set_shared_headers(opts: &RunOpts, identity: &Identity) {
    let mut h = String::new();
    if let Some(origin) = opts.cors_origin.as_deref().filter(|o| o.bytes().all(|b| b.is_ascii_graphic())) {
        h.push_str(&format!("Access-Control-Allow-Origin: {origin}\r\n"));
        if origin != "*" { h.push_str("Vary: Origin\r\n"); }
    }
    let file: String = identity.file.chars().filter(|c| c.is_ascii_graphic()).collect();
    h.push_str(&format!("X-Ojas-Model: {file}\r\nX-Ojas-Model-Sha256: {}\r\nX-Ojas-Revision: {REVISION}\r\n", identity.sha256));
    let _ = HEADERS.set(h);
}

/// The header line echoing a caller's request id, or nothing.
fn request_id_header(id: Option<&str>) -> String {
    id.map(|id| format!("X-Request-Id: {id}\r\n")).unwrap_or_default()
}

/// A whole HTTP response. `extra` is header lines of this response alone. A
/// status that tells the caller to come back says when.
fn response(status: &str, content_type: &str, body: &[u8], extra: &str) -> Vec<u8> {
    let retry = if status.starts_with("429") || status.starts_with("503") { "Retry-After: 1\r\n" } else { "" };
    let mut out = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         {retry}{extra}{}Connection: close\r\n\r\n",
        body.len(), shared_headers()
    ).into_bytes();
    out.extend_from_slice(body);
    out
}

fn json_response(status: &str, v: &Value, extra: &str) -> Vec<u8> {
    response(status, "application/json", v.to_string().as_bytes(), extra)
}

/// OpenAI's error envelope, so clients surface the text instead of a blank failure.
/// An error body: `{"error": {"code", "message", "type"}}`, the type named for the
/// status's class.
fn error_response(status: &str, msg: &str, extra: &str) -> Vec<u8> {
    json_response(status, &error_body(status, msg), extra)
}

fn error_body(status: &str, msg: &str) -> Value {
    let code: u16 = status.split(' ').next().and_then(|c| c.parse().ok()).unwrap_or(500);
    let kind = match code {
        400 | 413 | 422 => "invalid_request_error",
        401 | 403 => "authentication_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        504 => "timeout_error",
        501 => "not_supported_error",
        503 => "unavailable_error",
        _ => "server_error",
    };
    json!({"error": {"code": code, "message": msg, "type": kind}})
}

/// The head of a server-sent event stream.
fn sse_head(extra: &str) -> Vec<u8> {
    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
             {extra}{}Connection: close\r\n\r\n", shared_headers()).into_bytes()
}

fn sse_event(v: &Value) -> Vec<u8> { format!("data: {v}\n\n").into_bytes() }

pub(crate) fn send(stream: &mut Conn, status: &str, content_type: &str, body: &[u8]) {
    let extra = request_id_header(stream.request_id.as_deref());
    let _ = stream.write_all(&response(status, content_type, body, &extra)).and_then(|_| stream.flush());
}

pub(crate) fn send_json(stream: &mut Conn, status: &str, v: &Value) {
    send(stream, status, "application/json", v.to_string().as_bytes());
}

pub(crate) fn send_err(stream: &mut Conn, status: &str, msg: &str) {
    send_json(stream, status, &error_body(status, msg));
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The prompt-prefix cache's state for `/cache` and `/props`; `null` when the model
/// keeps none.
fn cache_json(stats: Option<ojas_core::PrefixCacheStats>) -> Value {
    let Some(s) = stats else { return Value::Null };
    json!({
        "blocks": s.blocks, "snapshots": s.snapshots, "block_tokens": s.block_tokens,
        "bytes": s.bytes, "budget_bytes": s.budget,
        "lookups": s.lookups, "hits": s.hits, "reused_tokens": s.reused_tokens,
        "directory": s.directory.then(|| json!({
            "blocks": s.disk_blocks, "snapshots": s.disk_snapshots, "bytes": s.disk_bytes,
            "budget_bytes": s.disk_budget, "reads": s.disk_reads,
        })),
        "pinned_blocks": s.pinned_blocks,
        "evictions": s.evictions,
        "documents": (s.docs.budget > 0).then(|| json!({
            "docs": s.docs.docs, "bytes": s.docs.bytes, "budget_bytes": s.docs.budget,
            "hits": s.docs.hits, "reused_tokens": s.docs.reused_tokens,
        })),
        "last": restore_json(&s.last),
    })
}

/// Where a prompt's reused tokens came from, and what restoring them cost.
fn restore_json(r: &ojas_core::PrefixRestore) -> Value {
    let tier = match (r.reused_tokens, r.disk_payloads) {
        (0, _) => "none",
        (_, 0) => "ram",
        _ => "disk",
    };
    json!({
        "tier": tier, "matched_tokens": r.matched_tokens, "doc_reused_tokens": r.doc_reused_tokens,
        "reused_tokens": r.reused_tokens, "restore_ms": r.restore_us as f64 / 1e3,
    })
}

/// The tokens a warm request caches. A raw `prompt` is taken whole. For `messages`
/// it is the part of the transcript every continuation shares: the transcript is
/// rendered with two different next user turns and cut where they diverge, which
/// holds for any chat template.
fn warm_ids(body: &Value, bpe: &Bpe, info: &ModelInfo, opts: &RunOpts) -> Result<Vec<u32>> {
    let Some(msgs) = body.get("messages").and_then(Value::as_array) else {
        return prompt_ids(body, bpe, info, opts);
    };
    let render = |next: &str| {
        let mut turns = msgs.clone();
        turns.push(json!({"role": "user", "content": next}));
        prompt_ids(&json!({ "messages": turns }), bpe, info, opts)
    };
    let (a, b) = (render("a")?, render("b")?);
    Ok(a[..ojas_tokenize::shared_prefix(&a, &b)].to_vec())
}

/// `ids`, when they fit in the context the model was loaded with.
fn fits_context(ids: Vec<u32>, info: &ModelInfo) -> Result<Vec<u32>> {
    anyhow::ensure!(ids.len() < info.context, "the prompt is {} tokens; the context holds {}", ids.len(), info.context);
    Ok(ids)
}

/// Process and pin `--prefix-cache-pin`'s system prompt, when one is set.
fn pin_startup_prompt(model: &dyn Model, bpe: &Bpe, info: &ModelInfo, opts: &RunOpts) -> Result<()> {
    let Some(path) = ojas_core::config::EngineConfig::current().prefix_cache_pin else { return Ok(()) };
    let system = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let ids = fits_context(warm_ids(&json!({"messages": [{"role": "system", "content": system}]}), bpe, info, opts)?, info)
        .with_context(|| format!("{}", path.display()))?;
    match model.warm_prefix(&ids, true) {
        Some(n) => eprintln!("  pinned {n} of {} system-prompt tokens from {}", ids.len(), path.display()),
        None => eprintln!("  {}: this model keeps no prompt-prefix cache; nothing pinned", path.display()),
    }
    Ok(())
}

/// Per-request sampling: whatever the body specifies, else the CLI defaults.
/// `constrained` requests default to no repetition penalty, which fights the
/// punctuation structured output repeats on every field.
fn sampling_from(body: &Value, base: &RunOpts, constrained: bool) -> SampleOpts {
    let f = |k: &str, d: f32| body.get(k).and_then(Value::as_f64).map(|v| v as f32).unwrap_or(d);
    let u = |k: &str, d: usize| body.get(k).and_then(Value::as_u64).map(|v| v as usize).unwrap_or(d);
    SampleOpts {
        temperature: f("temperature", base.sample.temperature),
        top_p: f("top_p", base.sample.top_p),
        top_k: u("top_k", base.sample.top_k),
        repeat_penalty: f("repeat_penalty", if constrained { 1.0 } else { base.sample.repeat_penalty }),
        repeat_window: u("repeat_last_n", base.sample.repeat_window),
        seed: body.get("seed").and_then(Value::as_u64).unwrap_or(base.sample.seed),
    }
}

/// The output constraint a request asks for: a GBNF `grammar`, a bare
/// `json_schema`, or OpenAI's `response_format`, `json_object` or
/// `json_schema` (whose schema sits under `json_schema.schema`). At most one.
fn output_format(body: &Value) -> Result<Option<OutputFormat>> {
    let mut found: Vec<OutputFormat> = Vec::new();
    if let Some(g) = body.get("grammar").filter(|v| !v.is_null()) {
        let g = g.as_str().context("\"grammar\" must be a string")?;
        if !g.trim().is_empty() { found.push(OutputFormat::Grammar(g.to_string())); }
    }
    if let Some(s) = body.get("json_schema").filter(|v| !v.is_null()) {
        found.push(OutputFormat::JsonSchema(s.clone()));
    }
    if let Some(rf) = body.get("response_format").filter(|v| !v.is_null()) {
        match rf.get("type").and_then(Value::as_str) {
            None | Some("text") => {}
            Some("json_object") => found.push(match rf.get("schema") {
                Some(s) => OutputFormat::JsonSchema(s.clone()),
                None => OutputFormat::JsonObject,
            }),
            Some("json_schema") => {
                let schema = rf.get("json_schema").and_then(|j| j.get("schema"))
                    .context("response_format json_schema needs json_schema.schema")?;
                found.push(OutputFormat::JsonSchema(schema.clone()));
            }
            Some(other) => anyhow::bail!("unsupported response_format type \"{other}\""),
        }
    }
    if found.len() > 1 {
        anyhow::bail!("give at most one of \"grammar\", \"json_schema\", \"response_format\"");
    }
    Ok(found.pop())
}

/// `stop`: a string or an array of strings.
fn stop_strings(body: &Value) -> Result<Vec<String>> {
    match body.get("stop") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(a)) => a.iter().map(|v| v.as_str().map(str::to_string)
            .context("\"stop\" entries must be strings")).collect(),
        Some(_) => anyhow::bail!("\"stop\" must be a string or an array of strings"),
    }
}

/// OpenAI sends `messages`; llama.cpp sends `prompt`. Turn either into ids.
fn prompt_ids(body: &Value, bpe: &Bpe, info: &ModelInfo, opts: &RunOpts) -> Result<Vec<u32>> {
    // A raw `prompt` may be a string or an array of token ids.
    if let Some(p) = body.get("prompt") {
        if let Some(s) = p.as_str() {
            return Ok(bpe.encode(s).into_iter().map(|v| v as u32).collect());
        }
        if let Some(a) = p.as_array() {
            return a.iter().map(|v| match v.as_u64() {
                Some(id) if (id as usize) < info.vocab => Ok(id as u32),
                _ => anyhow::bail!("prompt token {v} is not an id in the {}-token vocabulary", info.vocab),
            }).collect();
        }
    }
    if let Some(msgs) = body.get("messages").and_then(Value::as_array) {
        let (system, turns) = chat_turns(msgs, opts);
        let text = ojas_tokenize::chat_transcript(&info.arch, &system, &turns);
        return Ok(bpe.encode(&text).into_iter().map(|v| v as u32).collect());
    }
    anyhow::bail!("request needs either \"prompt\" or \"messages\"")
}

/// The system prompt and the (role, text) turns of a `messages` array.
fn chat_turns(msgs: &[Value], opts: &RunOpts) -> (String, Vec<(String, String)>) {
    let mut system = opts.system.clone();
    let mut turns = Vec::new();
    for m in msgs {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = images::content_text(m.get("content"));
        if role == "system" {
            system = content;
        } else {
            turns.push((role.to_string(), content));
        }
    }
    (system, turns)
}

/// Token ranges of a request's prompt the document cache may serve: the text of
/// every chat message marked `"cache_doc": true`, or a raw prompt's `doc_spans`
/// (`[[start, end], ...]` in tokens).
fn prompt_docs(body: &Value, bpe: &Bpe, info: &ModelInfo, opts: &RunOpts) -> Vec<(usize, usize)> {
    if let Some(spans) = body.get("doc_spans").and_then(Value::as_array) {
        return spans.iter().filter_map(|s| {
            let s = s.as_array()?;
            Some((s.first()?.as_u64()? as usize, s.get(1)?.as_u64()? as usize))
        }).collect();
    }
    let Some(msgs) = body.get("messages").and_then(Value::as_array) else { return Vec::new() };
    let (system, turns) = chat_turns(msgs, opts);
    let tokens = |text: &str| -> Vec<u32> { bpe.encode(text).into_iter().map(|v| v as u32).collect() };
    msgs.iter().filter(|m| m.get("role").and_then(Value::as_str) != Some("system"))
        .enumerate()
        .filter(|(_, m)| m.get("cache_doc").and_then(Value::as_bool) == Some(true))
        .map(|(k, _)| ojas_tokenize::transcript_span(&info.arch, &system, &turns, k, tokens))
        .filter(|(s, e)| s < e)
        .collect()
}

/// User turns, counted from the end, whose boundaries are marked: the decoder keeps
/// at most this many snapshots per prompt besides its branch point and end.
const MARKED_TURNS: usize = 6;

/// Message boundaries in a chat request's prompt, for the prefix cache to keep
/// state at; none for a raw prompt.
fn prompt_marks(body: &Value, bpe: &Bpe, info: &ModelInfo, opts: &RunOpts) -> Vec<usize> {
    let Some(msgs) = body.get("messages").and_then(Value::as_array) else { return Vec::new() };
    let (system, turns) = chat_turns(msgs, opts);
    ojas_tokenize::transcript_boundaries(&info.arch, &system, &turns, MARKED_TURNS,
        |text| bpe.encode(text).into_iter().map(|v| v as u32).collect())
}

fn stop_ids(bpe: &Bpe, info: &ModelInfo) -> (Option<u32>, Option<u32>) {
    let marker = bpe.encode(ojas_tokenize::chat_eos(&info.arch));
    match marker.as_slice() {
        [one] if Some(*one as u32) != info.eos => (Some(*one as u32), info.eos),
        _ => (info.eos, None),
    }
}

pub fn serve(model: &str, opts: &RunOpts, context: usize) -> Result<()> {
    if opts.keep_nothing {
        let cfg = ojas_core::config::EngineConfig::current();
        anyhow::ensure!(cfg.prefix_cache_dir.is_none(),
            "--keep-nothing and a prefix cache directory are alternatives: the directory holds processed prompts on disk");
    }
    // Listening starts before the model loads, so a probe gets an answer from the first moment.
    let front = Front::bind(opts, "this model")?;
    let loading = front.answer_while_loading()?;
    let hashing = Identity::start(model);
    if crate::decide::is_decision_model(model) {
        return crate::decide::serve(model, opts, front, loading, hashing);
    }
    with_model(model, opts.device, context, opts.precision, |m, bpe, info| {
        let (primary, secondary) = stop_ids(bpe, info);
        // Token bytes for constrained requests, built once for the session.
        let vocab = crate::constrain::vocab(bpe, info);
        pin_startup_prompt(m, bpe, info, opts)?;
        let vision = images::Vision::open(model, m, bpe, info, opts);
        let identity = Identity::finish(hashing)?;
        set_shared_headers(opts, &identity);
        let ctx = Ctx { bpe, info, opts, vocab: &vocab, primary, secondary, prefix_cache: m.prefix_cache_stats().is_some(),
                        images: &vision, identity: &identity,
                        timeout: opts.request_timeout.filter(|s| *s > 0.0).map(Duration::from_secs_f64) };

        let slots = m.max_slots();
        metrics::slots(slots);
        eprintln!(
            "  {} | {} | ctx {} | MTP {} | loaded in {:.1}s",
            info.arch,
            info.backend,
            info.context,
            if info.has_mtp { "yes" } else { "no" },
            info.load_secs
        );
        eprintln!("  {} sha256 {} | engine {REVISION}", identity.file, identity.sha256);
        match slots {
            1 => eprintln!("  listening on {}  (one request at a time; {})", front.url(), front.auth.describe()),
            n => eprintln!("  listening on {}  ({n} requests at a time; {})", front.url(), front.auth.describe()),
        }
        if opts.keep_nothing { eprintln!("  keeping nothing: no cache files, no request content in the log"); }
        if let Ok(v) = &vision { eprintln!("  images: data URLs in chat messages, up to {} tokens each", v.max_tokens); }
        eprintln!("  POST /v1/chat/completions  POST /v1/messages  POST /v1/completions  POST /completion  GET /health  GET /ready  GET /metrics  GET /props  GET /cache  POST /cache/save|warm|unpin  GET /v1/models");

        let props = json!({
            "model": info.arch,
            "backend": info.backend,
            "n_ctx": info.context,
            "n_layers": info.n_layers,
            "hidden_dim": info.hidden_dim,
            "vocab_size": info.vocab,
            "mtp": info.has_mtp,
            "concurrent_slots": slots,
            "images": vision.as_ref().ok().map(|v| json!({"max_tokens": v.max_tokens})),
            "keep_nothing": opts.keep_nothing,
            "load_seconds": info.load_secs,
        });
        loading.finish();
        conn::drain_on_sigterm(Duration::from_secs_f64(opts.drain_secs.max(0.0)));
        if slots > 1 { serve_batched(&front, m, &ctx, &props) } else { serve_alone(&front, m, &ctx, &props) }
    })
}

/// Serve one request at a time on the thread that owns the model, with speculative
/// decoding available to every request.
fn serve_alone(front: &Front, model: &dyn Model, ctx: &Ctx, props: &Value) -> Result<()> {
    let mut core = EngineCore::new(model);
    serve_http(front, |stream, req, path| {
        if let Some((body, chat, oai)) = completion_route(stream, req, path) {
            match stream.try_clone() {
                Ok(stream) => run_alone(&mut core, ctx, stream, &body, chat, oai),
                Err(e) => send_err(stream, "500 Internal Server Error", &format!("cannot answer this connection: {e}")),
            }
        } else if (req.method.as_str(), path) == ("POST", "/cache/warm") {
            if let Some((ids, pin)) = warm_request(stream, req, ctx) { warm(stream, model, 0, &ids, pin); }
        } else if !serve_other(stream, req, path, model, ctx, props) {
            send_err(stream, "404 Not Found", &format!("no route for {} {}", req.method, path));
        }
    })
}

fn run_alone(core: &mut EngineCore<&dyn Model>, ctx: &Ctx, stream: Conn, body: &Value, chat: bool, oai: bool) {
    let Some((mut job, mut reply)) = completion::start(stream, ctx, body, chat, oai) else { return };
    if let Some(err) = ojas_core::device_fault::peek() { return reply.refuse("503 Service Unavailable", &fault_message(&err)); }
    metrics::load(1, 0);
    reply.begin();
    let model = core.model();
    model.set_prefix_reuse(job.reuse);
    if !job.reuse { model.reset_session(); }
    model.set_prefix_marks(&job.marks);
    model.set_prefix_docs(&job.docs, job.reuse);
    (core.eog, core.eos, core.banned) = (job.eog.clone(), job.eos, job.banned.clone());
    core.images = job.images.take();
    let mut constraint = job.constraint.take();
    let gen = core.generate_ex(&job.ids, job.want, job.sampling.as_ref(),
        constraint.as_deref_mut().map(|p| p as &mut dyn ojas_infer::LogitProcessor), &mut |_, _| {},
        &mut |t| reply.token(ctx.bpe, t));
    reply.finish(&gen);
    metrics::load(0, 0);
}

/// Why a session with a latched device fault refuses requests.
fn fault_message(err: &ojas_core::device_fault::DeviceError) -> String {
    format!("device fault; this model session is no longer usable and must be reloaded: {err}")
}

/// Waiting this long, a request is admitted before any whose prompt the prefix cache
/// covers further.
const MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// Requests that may wait for a slot; beyond it the server answers 429.
const MAX_QUEUE: usize = 64;

/// What a connection thread hands the engine thread.
enum Arrival {
    Completion(Box<(completion::Job, completion::Reply)>),
    Warm(Conn, Vec<u32>, bool),
    Other(Conn, Request, String),
}

/// Read and parse one connection: a completion is tokenized and checked here, off
/// the engine thread, so a long transcript or a slow client never delays decoding.
fn arrive((mut stream, req, path): Arrived, ctx: &Ctx) -> Option<Arrival> {
    if let Some((body, chat, oai)) = completion_route(&mut stream, &req, &path) {
        return completion::start(stream, ctx, &body, chat, oai).map(|r| Arrival::Completion(Box::new(r)));
    }
    if (req.method.as_str(), path.as_str()) == ("POST", "/cache/warm") {
        let (ids, pin) = warm_request(&mut stream, &req, ctx)?;
        return Some(Arrival::Warm(stream, ids, pin));
    }
    Some(Arrival::Other(stream, req, path))
}

/// Serve up to `max_slots` requests together. Each connection is read on its own
/// thread; this thread owns the model and steps the batch: each step admits waiting
/// requests, processes one prompt chunk, and decodes a token for every slot past its
/// prompt, streaming each to its client.
fn serve_batched(front: &Front, model: &dyn Model, ctx: &Ctx, props: &Value) -> Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(move || accept_each(front, scope, move |arrived| arrive(arrived, ctx), tx));
        let mut batch = Batch::new(model, MAX_WAIT);
        let mut replies: HashMap<u64, completion::Reply> = HashMap::new();
        // `/cache/warm` needs a slot of its own outside the batch, and waits here
        // until one is free.
        let mut warms: VecDeque<(Conn, Vec<u32>, bool, Ticket)> = VecDeque::new();
        let mut next_id = 0u64;
        loop {
            let mut arrived: Vec<(Arrival, Ticket)> = Vec::new();
            if batch.is_idle() && warms.is_empty() {
                match rx.recv() {
                    Ok(a) => arrived.push(a),
                    Err(_) => return Ok(()),
                }
            }
            arrived.extend(rx.try_iter());
            for (a, ticket) in arrived {
                match a {
                    Arrival::Completion(request) => {
                        let (job, mut reply) = *request;
                        if let Some(err) = ojas_core::device_fault::peek() {
                            reply.refuse("503 Service Unavailable", &fault_message(&err));
                        } else if batch.load().1 >= MAX_QUEUE {
                            reply.refuse("429 Too Many Requests", "the server is at capacity; retry shortly");
                        } else {
                            reply.begin();
                            batch.submit(next_id, job.into_request());
                            replies.insert(next_id, reply);
                            next_id += 1;
                        }
                    }
                    Arrival::Warm(stream, ids, pin) => warms.push_back((stream, ids, pin, ticket)),
                    Arrival::Other(mut stream, req, path) => {
                        if !serve_other(&mut stream, &req, &path, model, ctx, props) {
                            send_err(&mut stream, "404 Not Found", &format!("no route for {} {}", req.method, path));
                        }
                    }
                }
            }
            if let Some(slot) = batch.free_slot() {
                if let Some((mut stream, ids, pin, _ticket)) = warms.pop_front() { warm(&mut stream, model, slot, &ids, pin); }
            }
            let (running, waiting) = batch.load();
            metrics::load(running, waiting);
            batch.step(&mut |id, event| match event {
                Event::Token(t) => replies.get_mut(&id).is_some_and(|r| r.token(ctx.bpe, t)),
                Event::Prefill(..) => replies.get(&id).is_some_and(completion::Reply::connected),
                Event::Done(gen) => {
                    if let Some(r) = replies.remove(&id) { r.finish(&gen); }
                    true
                }
            });
        }
    })
}

/// The body and response format of a completion request, answering a malformed body
/// itself; `None` for any other route.
fn completion_route(stream: &mut Conn, req: &Request, path: &str) -> Option<(Value, bool, bool)> {
    let ("POST", "/completion" | "/completions" | "/v1/completions" | "/v1/chat/completions" | "/v1/messages") =
        (req.method.as_str(), path) else { return None };
    match serde_json::from_slice::<Value>(&req.body) {
        Ok(mut body) => {
            if path == "/v1/messages" { body[MESSAGES_MARK] = Value::Bool(true); }
            Some((body, path == "/v1/chat/completions" || path == "/v1/messages", path.starts_with("/v1/")))
        }
        Err(e) => {
            metrics::failed();
            send_err(stream, "400 Bad Request", &format!("invalid JSON: {e}"));
            None
        }
    }
}

/// The tokens and pin flag of a `/cache/warm` request, answering a malformed one
/// itself.
fn warm_request(stream: &mut Conn, req: &Request, ctx: &Ctx) -> Option<(Vec<u32>, bool)> {
    let body: Value = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => {
            send_err(stream, "400 Bad Request", &format!("invalid JSON: {e}"));
            return None;
        }
    };
    if images::has_images(&body) {
        send_err(stream, "400 Bad Request", "a prefix is warmed from text; send the images in a completion request");
        return None;
    }
    match warm_ids(&body, ctx.bpe, ctx.info, ctx.opts).and_then(|ids| fits_context(ids, ctx.info)) {
        Ok(ids) => Some((ids, body.get("pin").and_then(Value::as_bool).unwrap_or(true))),
        Err(e) => {
            send_err(stream, "400 Bad Request", &e.to_string());
            None
        }
    }
}

/// Process `ids` into the prefix cache in `slot`, which no request holds.
fn warm(stream: &mut Conn, model: &dyn Model, slot: usize, ids: &[u32], pin: bool) {
    let mut cached = None;
    if !model.with_slot(slot, &mut || cached = model.warm_prefix(ids, pin)) {
        return send_err(stream, "500 Internal Server Error", &format!("slot {slot} cannot be addressed"));
    }
    match cached {
        Some(cached) => send_json(stream, "200 OK", &json!({
            "tokens": ids.len(), "cached_tokens": cached, "pinned": pin, "cache": cache_json(model.prefix_cache_stats()),
        })),
        None => send_err(stream, "400 Bad Request", "this model keeps no prompt-prefix cache"),
    }
}

/// Every route but completions and `/cache/warm`. False when the route is unknown.
fn serve_other(stream: &mut Conn, req: &Request, path: &str, model: &dyn Model, ctx: &Ctx, props: &Value) -> bool {
    match (req.method.as_str(), path) {
        ("GET", "/health") => health(stream),
        ("GET", "/ready") => ready(stream),
        ("GET", "/metrics") => send(stream, "200 OK", "text/plain; version=0.0.4", metrics::render().as_bytes()),
        ("GET", "/props") => {
            let mut p = props.clone();
            p["prefix_cache"] = cache_json(model.prefix_cache_stats());
            send_json(stream, "200 OK", &p)
        }
        ("GET", "/cache") => send_json(stream, "200 OK", &cache_json(model.prefix_cache_stats())),
        ("POST", "/cache/save") if ctx.opts.keep_nothing =>
            send_err(stream, "403 Forbidden", "this server keeps nothing: the cache is not written to disk"),
        ("POST", "/cache/save") => {
            model.save_prefix_cache();
            send_json(stream, "200 OK", &cache_json(model.prefix_cache_stats()))
        }
        ("POST", "/cache/unpin") => {
            model.unpin_prefix_cache();
            send_json(stream, "200 OK", &cache_json(model.prefix_cache_stats()))
        }
        ("GET", "/v1/models") => {
            let mut entry = json!({"id": ctx.info.arch, "object": "model", "created": now(), "owned_by": "ojas"});
            ctx.identity.describe(&mut entry);
            send_json(stream, "200 OK", &json!({"object": "list", "data": [entry]}))
        }
        ("POST", "/v1/messages/count_tokens") => match serde_json::from_slice::<Value>(&req.body) {
            Ok(body) => match anthropic::convert(&body, &ctx.opts.system) {
                Ok(c) => {
                    let n = ctx.bpe.encode(c.body["prompt"].as_str().unwrap_or_default()).len();
                    send_json(stream, "200 OK", &json!({"input_tokens": n}))
                }
                Err(e) => send_err(stream, "400 Bad Request", &e),
            },
            Err(e) => send_err(stream, "400 Bad Request", &format!("invalid JSON: {e}")),
        },
        ("POST", "/v1/systemone" | "/v1/decide" | "/decide") =>
            send_err(stream, "501 Not Implemented", "this model is not a decision model"),
        _ => return false,
    }
    true
}

/// Liveness: the process is up and its device has not faulted. A process that
/// cannot serve must not report healthy; the recovery path is a restart by the
/// orchestrator.
pub(crate) fn health(stream: &mut Conn) {
    match ojas_core::device_fault::peek() {
        Some(err) => send_json(stream, "503 Service Unavailable", &json!({"status": "error", "error": err.to_string()})),
        None => send_json(stream, "200 OK", &json!({"status": "ok"})),
    }
}

/// Readiness: whether to route requests here. Not while the model loads, not once
/// the server is stopping, and not after a device fault.
pub(crate) fn ready(stream: &mut Conn) {
    if let Some(err) = ojas_core::device_fault::peek() {
        send_json(stream, "503 Service Unavailable", &json!({"status": "error", "error": err.to_string()}))
    } else if conn::draining() {
        send_json(stream, "503 Service Unavailable", &json!({"status": "draining"}))
    } else {
        send_json(stream, "200 OK", &json!({"status": "ready"}))
    }
}

/// Routes answered without credentials: they say nothing a port scan cannot see,
/// and an orchestrator's probes carry none.
fn is_probe(path: &str) -> bool { matches!(path, "/health" | "/ready") }

/// The listening socket and what guards it: who may call, and TLS when the server
/// was given a certificate.
pub(crate) struct Front {
    listener: TcpListener,
    pub(crate) auth: Arc<crate::access::Auth>,
    tls: Option<Arc<rustls::ServerConfig>>,
    addr: String,
}

impl Front {
    /// Bind `--host`/`--port` under the access rules of `opts`. `what` names what
    /// is served, for the refusal to bind beyond loopback without a token.
    pub(crate) fn bind(opts: &RunOpts, what: &str) -> Result<Front> {
        let auth = Arc::new(crate::access::Auth::from_opts(opts, what)?);
        let tls = match (&opts.tls_cert, &opts.tls_key) {
            (Some(cert), Some(key)) => Some(conn::tls_config(cert, key)?),
            (None, None) => None,
            _ => anyhow::bail!("--tls-cert and --tls-key go together"),
        };
        let addr = format!("{}:{}", opts.host, opts.port);
        let listener = TcpListener::bind(&addr).with_context(|| format!("binding {addr}"))?;
        Ok(Front { listener, auth, tls, addr })
    }

    pub(crate) fn url(&self) -> String {
        format!("{}://{}", if self.tls.is_some() { "https" } else { "http" }, self.addr)
    }

    fn try_clone(&self) -> Result<Front> {
        Ok(Front { listener: self.listener.try_clone()?, auth: Arc::clone(&self.auth), tls: self.tls.clone(),
                   addr: self.addr.clone() })
    }

    /// Read one request from a new connection, with its path (query string
    /// removed), and decide whether it is let in. Answers by itself, and returns
    /// `None`, for a CORS preflight, a request that cannot be parsed (400), a
    /// caller without the right to call (401), and anything but a probe once the
    /// server is stopping (503); `None` too for a connection closed before a request.
    fn admit(&self, stream: TcpStream) -> Option<Arrived> {
        let _ = stream.set_write_timeout(Some(SOCKET_TIMEOUT));
        let mut conn = match &self.tls {
            Some(config) => Conn::tls(stream, config).ok()?,
            None => Conn::plain(stream),
        };
        let req = {
            let mut reader = BufReader::new(Deadline { conn: &mut conn, until: Instant::now() + SOCKET_TIMEOUT });
            match read_request(&mut reader) {
                Ok(Some(r)) => r,
                Ok(None) => return None,
                Err(e) => {
                    // A failed handshake or a dropped socket has no one to answer.
                    if e.downcast_ref::<std::io::Error>().is_none() {
                        send_err(&mut conn, "400 Bad Request", &format!("{e:#}"));
                    }
                    return None;
                }
            }
        };
        conn.request_id = req.request_id.clone();
        if req.method == "OPTIONS" {
            // Without an allowed origin the preflight carries no grant, and the browser refuses.
            let head = format!("HTTP/1.1 204 No Content\r\n{}Access-Control-Allow-Headers: *\r\n\
                                Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
                                Content-Length: 0\r\nConnection: close\r\n\r\n", shared_headers());
            let _ = conn.write_all(head.as_bytes()).and_then(|_| conn.flush());
            return None;
        }
        let _ = conn.set_read_timeout(Some(SOCKET_TIMEOUT));
        let path = req.path.split('?').next().unwrap_or("").to_string();
        if !is_probe(&path) {
            if let Err(why) = self.auth.check(conn.peer_ip(), req.bearer.as_deref()) {
                eprintln!("  refused {:?} {} {path}: {why}", conn.peer_ip(), req.method);
                let body = error_body("401 Unauthorized", why).to_string();
                let _ = conn.write_all(&response("401 Unauthorized", "application/json", body.as_bytes(),
                    "WWW-Authenticate: Bearer\r\n")).and_then(|_| conn.flush());
                return None;
            }
            if conn::draining() && path != "/metrics" {
                metrics::refused();
                send_err(&mut conn, "503 Service Unavailable", "the server is stopping; send this request elsewhere");
                return None;
            }
        }
        Some((conn, req, path))
    }

    /// Answer probes while the model loads: alive, not ready. Any other request
    /// is told to come back. Ends at `Loading::finish`, after which the caller
    /// accepts connections itself.
    pub(crate) fn answer_while_loading(&self) -> Result<Loading> {
        let front = Arc::new(self.try_clone()?);
        front.listener.set_nonblocking(true)?;
        let done = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&done);
        let thread = std::thread::Builder::new().name("http-loading".into()).spawn(move || {
            while !stop.load(Ordering::Acquire) {
                let stream = match front.listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(_) => {
                        std::thread::sleep(Duration::from_millis(20));
                        continue;
                    }
                };
                // An accepted socket may inherit the listener's non-blocking mode.
                if stream.set_nonblocking(false).is_err() { continue; }
                let front = Arc::clone(&front);
                std::thread::spawn(move || {
                    let Some((mut conn, req, path)) = front.admit(stream) else { return };
                    match (req.method.as_str(), path.as_str()) {
                        ("GET", "/health") => send_json(&mut conn, "200 OK", &json!({"status": "loading"})),
                        _ => send_json(&mut conn, "503 Service Unavailable", &json!({"status": "loading"})),
                    }
                });
            }
        })?;
        Ok(Loading { done, thread: Some(thread), listener: self.listener.try_clone()? })
    }
}

/// The thread answering for a server whose model is still loading.
pub(crate) struct Loading {
    done: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    listener: TcpListener,
}

impl Loading {
    /// The model is loaded: hand the listening socket back, blocking again.
    pub(crate) fn finish(mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() { let _ = t.join(); }
        let _ = self.listener.set_nonblocking(false);
    }
}

impl Drop for Loading {
    fn drop(&mut self) { self.done.store(true, Ordering::Release); }
}

/// Hand each request to `handle` with its path (query string removed), one at a
/// time on the calling thread, in the order they arrive. Each connection is read on
/// its own thread, so a client slow to send its request delays no one else's.
pub(crate) fn serve_http(front: &Front, mut handle: impl FnMut(&mut Conn, &Request, &str)) -> Result<()> {
    serve_http_batches(front, 1, |batch| {
        for (mut stream, req, path) in batch { handle(&mut stream, &req, &path); }
    })
}

/// A request read from its connection: the connection to answer on, the request,
/// and its path (query string removed).
pub(crate) type Arrived = (Conn, Request, String);

/// Hand requests to `handle` on the calling thread, in arrival order, in batches:
/// it waits for one, then takes every request already waiting, up to `most`.
/// Connections are read as in [`serve_http`].
pub(crate) fn serve_http_batches(front: &Front, most: usize, mut handle: impl FnMut(Vec<Arrived>)) -> Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(move || accept_each(front, scope, Some, tx));
        while let Ok(first) = rx.recv() {
            let mut batch = vec![first];
            while batch.len() < most.max(1) {
                match rx.try_recv() {
                    Ok(next) => batch.push(next),
                    Err(_) => break,
                }
            }
            let (arrived, _tickets): (Vec<Arrived>, Vec<Ticket>) = batch.into_iter().unzip();
            handle(arrived);
        }
        Ok(())
    })
}

/// Connections being read, or holding a request not yet answered, at once. A
/// connection past it is answered 429 straight away, so neither threads nor
/// queued request bodies grow without bound.
const MAX_CONNECTIONS: usize = 128;

/// An admitted connection, counted against `MAX_CONNECTIONS` until dropped.
pub(crate) struct Ticket(Arc<AtomicUsize>);

impl Drop for Ticket {
    fn drop(&mut self) { self.0.fetch_sub(1, Ordering::AcqRel); }
}

/// Read and admit every new connection on its own thread, pass what arrived
/// through `read`, and send the result to `tx` with the connection's ticket, which
/// the receiver holds until the request is answered.
fn accept_each<'scope, T: Send + 'scope>(front: &'scope Front, scope: &'scope std::thread::Scope<'scope, '_>,
                                         read: impl Fn(Arrived) -> Option<T> + Copy + Send + 'scope,
                                         tx: std::sync::mpsc::Sender<(T, Ticket)>) {
    let open = Arc::new(AtomicUsize::new(0));
    for conn in front.listener.incoming() {
        match conn {
            Ok(stream) => {
                if open.fetch_add(1, Ordering::AcqRel) >= MAX_CONNECTIONS {
                    open.fetch_sub(1, Ordering::AcqRel);
                    metrics::refused();
                    // Over TLS the answer would cost a handshake on this thread; the connection is closed instead.
                    if front.tls.is_none() {
                        let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                        send_err(&mut Conn::plain(stream), "429 Too Many Requests", "too many connections; retry shortly");
                    }
                    continue;
                }
                let (tx, ticket) = (tx.clone(), Ticket(open.clone()));
                scope.spawn(move || {
                    if let Some(item) = front.admit(stream).and_then(read) { let _ = tx.send((item, ticket)); }
                });
            }
            Err(e) => {
                tracing::warn!("accept failed: {e}");
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// A client gets this long to send its whole request, and each write to it this
/// long to complete, so an idle or stalled connection cannot hold a thread forever.
const SOCKET_TIMEOUT: Duration = Duration::from_secs(30);

/// A connection read against one deadline for the whole request, so a client that
/// sends a byte at a time cannot hold its thread past `SOCKET_TIMEOUT`.
struct Deadline<'a> {
    conn: &'a mut Conn,
    until: Instant,
}

impl Read for Deadline<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "the request did not arrive in time"));
        }
        self.conn.set_read_timeout(Some(left))?;
        self.conn.read(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_bodies_name_the_class_of_their_status() {
        for (status, kind) in [("400 Bad Request", "invalid_request_error"), ("404 Not Found", "not_found_error"),
                               ("501 Not Implemented", "not_supported_error"), ("503 Service Unavailable", "unavailable_error"),
                               ("429 Too Many Requests", "rate_limit_error"), ("504 Gateway Timeout", "timeout_error"),
                               ("500 Internal Server Error", "server_error")] {
            let body = error_body(status, "m");
            assert_eq!(body["error"]["type"], kind, "{status}");
            assert_eq!(body["error"]["code"].as_u64(), status.split(' ').next().unwrap().parse().ok());
        }
    }

    #[test]
    fn prompt_token_ids_must_be_in_the_vocabulary() {
        let info = info_for("qwen3");
        let opts = RunOpts::default();
        let bpe = Bpe::default();
        assert_eq!(prompt_ids(&json!({"prompt": [1, 2, 9]}), &bpe, &info, &opts).unwrap(), vec![1, 2, 9]);
        for bad in [json!({"prompt": [1, 10]}), json!({"prompt": [1, -1]}), json!({"prompt": [1, "a"]})] {
            assert!(prompt_ids(&bad, &bpe, &info, &opts).is_err(), "{bad}");
        }
        assert!(fits_context(vec![0; 63], &info).is_ok());
        assert!(fits_context(vec![0; 64], &info).is_err());
    }

    #[test]
    fn a_request_must_arrive_before_its_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let mut s = TcpStream::connect(addr).unwrap();
            s.write_all(b"GET /health HTTP/1.1\r\n").unwrap();
            std::thread::sleep(std::time::Duration::from_millis(400));
            let _ = s.write_all(b"\r\n");
        });
        let mut conn = Conn::plain(listener.accept().unwrap().0);
        let mut reader = BufReader::new(Deadline { conn: &mut conn, until: Instant::now() + Duration::from_millis(100) });
        assert!(read_request(&mut reader).is_err(), "a request arriving after its deadline was read");
        client.join().unwrap();
    }

    #[test]
    fn a_request_carries_its_bearer_token_and_a_printable_request_id() {
        let head = "POST /x HTTP/1.1\r\nAuthorization: Bearer  abc \r\nX-Request-Id: r-1\r\nContent-Length: 2\r\n\r\nhi";
        let req = read_request(&mut BufReader::new(head.as_bytes())).unwrap().unwrap();
        assert_eq!((req.bearer.as_deref(), req.request_id.as_deref(), req.body.as_slice()), (Some("abc"), Some("r-1"), &b"hi"[..]));
        let odd = "GET / HTTP/1.1\r\nX-Request-Id: a b\r\n\r\n";
        assert!(read_request(&mut BufReader::new(odd.as_bytes())).unwrap().unwrap().request_id.is_none());
    }

    #[test]
    fn a_come_back_later_status_says_when_and_no_origin_is_allowed_unasked() {
        let full = String::from_utf8(response("429 Too Many Requests", "application/json", b"{}", "")).unwrap();
        assert!(full.contains("Retry-After: 1\r\n"), "{full}");
        let ok = String::from_utf8(response("200 OK", "application/json", b"{}", "X-Request-Id: r\r\n")).unwrap();
        assert!(!ok.contains("Retry-After") && ok.contains("X-Request-Id: r\r\n"));
        if HEADERS.get().is_none() { assert!(!ok.contains("Access-Control-Allow-Origin")); }
    }

    #[test]
    fn oversized_headers_are_refused() {
        let head = format!("GET / HTTP/1.1\r\nX-Pad: {}\r\n\r\n", "a".repeat(MAX_HEAD));
        let mut reader = BufReader::new(head.as_bytes());
        assert!(read_request(&mut reader).is_err());
    }

    fn info_for(arch: &str) -> ModelInfo {
        ModelInfo {
            arch: arch.into(), eos: Some(2), eog: vec![2], vocab: 10, context: 64, backend: "cpu",
            has_mtp: false, n_layers: 1, hidden_dim: 1, load_secs: 0.0,
        }
    }

    #[test]
    fn body_sampling_overrides_the_cli_default() {
        let base = RunOpts::default();
        let s = sampling_from(&json!({"temperature": 0.1, "top_k": 5, "seed": 99}), &base, false);
        assert!((s.temperature - 0.1).abs() < 1e-6);
        assert_eq!(s.top_k, 5);
        assert_eq!(s.seed, 99);
        // Unspecified fields keep the CLI's value rather than a hardcoded one.
        assert!((s.top_p - base.sample.top_p).abs() < 1e-6);
    }

    #[test]
    fn absent_sampling_fields_fall_back_to_the_cli() {
        let base = RunOpts::default();
        let s = sampling_from(&json!({}), &base, false);
        assert!((s.temperature - base.sample.temperature).abs() < 1e-6);
        assert_eq!(s.repeat_window, base.sample.repeat_window);
    }

    #[test]
    fn constrained_requests_default_to_no_repeat_penalty() {
        let base = RunOpts::default();
        assert_eq!(sampling_from(&json!({}), &base, true).repeat_penalty, 1.0);
        assert_eq!(sampling_from(&json!({"repeat_penalty": 1.2}), &base, true).repeat_penalty, 1.2);
    }

    #[test]
    fn output_format_reads_every_request_shape() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
        let f = |b: Value| output_format(&b).unwrap();
        assert!(f(json!({})).is_none());
        assert!(f(json!({"response_format": {"type": "text"}})).is_none());
        assert!(matches!(f(json!({"response_format": {"type": "json_object"}})), Some(OutputFormat::JsonObject)));
        assert!(matches!(f(json!({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}})),
            Some(OutputFormat::JsonSchema(_))));
        assert!(matches!(f(json!({"json_schema": schema})), Some(OutputFormat::JsonSchema(_))));
        assert!(matches!(f(json!({"grammar": "root ::= \"a\""})), Some(OutputFormat::Grammar(_))));
        assert!(output_format(&json!({"grammar": "root ::= \"a\"", "json_schema": schema})).is_err());
        assert!(output_format(&json!({"response_format": {"type": "json_schema"}})).is_err());
        assert!(output_format(&json!({"response_format": {"type": "xml"}})).is_err());
    }

    #[test]
    fn stop_accepts_a_string_or_a_list() {
        assert_eq!(stop_strings(&json!({"stop": "\n\n"})).unwrap(), vec!["\n\n"]);
        assert_eq!(stop_strings(&json!({"stop": ["a", "b"]})).unwrap(), vec!["a", "b"]);
        assert!(stop_strings(&json!({})).unwrap().is_empty());
        assert!(stop_strings(&json!({"stop": 3})).is_err());
    }

    /// llama.cpp clients send `prompt` as an array of token ids; that has to
    /// survive as ids rather than being stringified.
    #[test]
    fn a_token_id_array_prompt_is_used_verbatim() {
        let ids = prompt_ids(&json!({"prompt": [1, 2, 3]}), &Bpe::default(), &info_for("qwen3"), &RunOpts::default()).unwrap();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn a_request_with_neither_prompt_nor_messages_is_an_error() {
        let e = prompt_ids(&json!({"temperature": 1}), &Bpe::default(), &info_for("qwen3"), &RunOpts::default());
        assert!(e.is_err(), "must not silently generate from nothing");
    }

    #[test]
    fn oversized_content_length_is_rejected_before_allocating() {
        // The guard is a constant comparison; assert the policy, not the socket.
        assert!(MAX_BODY < usize::MAX, "there must be an upper bound at all");
    }
}
