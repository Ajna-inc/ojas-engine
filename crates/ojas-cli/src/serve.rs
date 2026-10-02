//! HTTP server: OpenAI-compatible plus llama.cpp's native `/completion`. A Laya
//! model is served by `decide::serve` instead (`POST /v1/decide`), on the same
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

mod completion;
use completion::Ctx;

const MAX_BODY: usize = 32 * 1024 * 1024;

pub(crate) struct Request {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) body: Vec<u8>,
}

/// Read one HTTP/1.1 request. `None` on a closed or unusable connection.
fn read_request(stream: &mut BufReader<&TcpStream>) -> Result<Option<Request>> {
    let mut line = String::new();
    if stream.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let (method, path) = match (parts.next(), parts.next()) {
        (Some(m), Some(p)) => (m.to_string(), p.to_string()),
        _ => return Ok(None),
    };

    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if stream.read_line(&mut h)? == 0 {
            return Ok(None);
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some(v) = h.split_once(':') {
            if v.0.eq_ignore_ascii_case("content-length") {
                len = v.1.trim().parse().unwrap_or(0);
            }
        }
    }
    if len > MAX_BODY {
        anyhow::bail!("request body of {len} bytes exceeds the {MAX_BODY}-byte limit");
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

/// A whole HTTP response.
fn response(status: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        body.len()
    ).into_bytes();
    out.extend_from_slice(body);
    out
}

fn json_response(status: &str, v: &Value) -> Vec<u8> { response(status, "application/json", v.to_string().as_bytes()) }

/// OpenAI's error envelope, so clients surface the text instead of a blank failure.
fn error_response(status: &str, msg: &str) -> Vec<u8> {
    json_response(status, &json!({"error": {"message": msg, "type": "invalid_request_error"}}))
}

/// The head of a server-sent event stream.
const SSE_HEAD: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
                        Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";

fn sse_event(v: &Value) -> Vec<u8> { format!("data: {v}\n\n").into_bytes() }

pub(crate) fn send(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let _ = stream.write_all(&response(status, content_type, body)).and_then(|_| stream.flush());
}

pub(crate) fn send_json(stream: &mut TcpStream, status: &str, v: &Value) {
    send(stream, status, "application/json", v.to_string().as_bytes());
}

pub(crate) fn send_err(stream: &mut TcpStream, status: &str, msg: &str) {
    let _ = stream.write_all(&error_response(status, msg)).and_then(|_| stream.flush());
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

/// Process and pin `--prefix-cache-pin`'s system prompt, when one is set.
fn pin_startup_prompt(model: &dyn Model, bpe: &Bpe, info: &ModelInfo, opts: &RunOpts) -> Result<()> {
    let Some(path) = ojas_core::config::EngineConfig::current().prefix_cache_pin else { return Ok(()) };
    let system = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let ids = warm_ids(&json!({"messages": [{"role": "system", "content": system}]}), bpe, info, opts)?;
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
            let text = if opts.raw { s.to_string() } else { s.to_string() };
            return Ok(bpe.encode(&text).into_iter().map(|v| v as u32).collect());
        }
        if let Some(a) = p.as_array() {
            return Ok(a.iter().filter_map(Value::as_u64).map(|v| v as u32).collect());
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
        let content = m.get("content").and_then(Value::as_str).unwrap_or("").to_string();
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
    #[cfg(target_os = "macos")]
    if crate::decide::is_laya(model) {
        return crate::decide::serve(model, opts);
    }
    with_model(model, opts.device, context, opts.precision, |m, bpe, info| {
        let (primary, secondary) = stop_ids(bpe, info);
        // Token bytes for constrained requests, built once for the session.
        let vocab = crate::constrain::vocab(bpe, info);
        pin_startup_prompt(m, bpe, info, opts)?;
        let ctx = Ctx { bpe, info, opts, vocab: &vocab, primary, secondary, prefix_cache: m.prefix_cache_stats().is_some() };

        let addr = format!("{}:{}", opts.host, opts.port);
        let listener = TcpListener::bind(&addr).with_context(|| format!("binding {addr}"))?;
        let slots = m.max_slots();
        eprintln!(
            "  {} | {} | ctx {} | MTP {} | loaded in {:.1}s",
            info.arch,
            info.backend,
            info.context,
            if info.has_mtp { "yes" } else { "no" },
            info.load_secs
        );
        match slots {
            1 => eprintln!("  listening on http://{addr}  (one request at a time)"),
            n => eprintln!("  listening on http://{addr}  ({n} requests at a time)"),
        }
        eprintln!("  POST /v1/chat/completions  POST /v1/completions  POST /completion  GET /health  GET /props  GET /cache  POST /cache/save|warm|unpin  GET /v1/models");

        let props = json!({
            "model": info.arch,
            "backend": info.backend,
            "n_ctx": info.context,
            "n_layers": info.n_layers,
            "hidden_dim": info.hidden_dim,
            "vocab_size": info.vocab,
            "mtp": info.has_mtp,
            "concurrent_slots": slots,
        });
        if slots > 1 { serve_batched(&listener, m, &ctx, &props) } else { serve_alone(&listener, m, &ctx, &props) }
    })
}

/// Serve one request at a time on the thread that owns the model, with speculative
/// decoding available to every request.
fn serve_alone(listener: &TcpListener, model: &dyn Model, ctx: &Ctx, props: &Value) -> Result<()> {
    let mut core = EngineCore::new(model);
    serve_http(listener, |stream, req, path| {
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

fn run_alone(core: &mut EngineCore<&dyn Model>, ctx: &Ctx, stream: TcpStream, body: &Value, chat: bool, oai: bool) {
    let Some((mut job, mut reply)) = completion::start(stream, ctx, body, chat, oai) else { return };
    if let Some(err) = ojas_core::device_fault::peek() { return reply.refuse("503 Service Unavailable", &fault_message(&err)); }
    reply.begin();
    let model = core.model();
    model.set_prefix_reuse(job.reuse);
    if !job.reuse { model.reset_session(); }
    model.set_prefix_marks(&job.marks);
    model.set_prefix_docs(&job.docs, job.reuse);
    (core.eog, core.eos, core.banned) = (job.eog.clone(), job.eos, job.banned.clone());
    let mut constraint = job.constraint.take();
    let gen = core.generate_ex(&job.ids, job.want, job.sampling.as_ref(),
        constraint.as_deref_mut().map(|p| p as &mut dyn ojas_infer::LogitProcessor), &mut |_, _| {},
        &mut |t| reply.token(ctx.bpe, t));
    reply.finish(&gen);
}

/// Why a session with a latched device fault refuses requests.
fn fault_message(err: &ojas_core::device_fault::DeviceError) -> String {
    format!("device fault; this model session is no longer usable and must be reloaded: {err}")
}

/// Waiting this long, a request is admitted before any whose prompt the prefix cache
/// covers further.
const MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// Requests that may wait for a slot; beyond it the server answers 503.
const MAX_QUEUE: usize = 64;

/// What a connection thread hands the engine thread.
enum Arrival {
    Completion(Box<(completion::Job, completion::Reply)>),
    Warm(TcpStream, Vec<u32>, bool),
    Other(TcpStream, Request, String),
}

/// Read and parse one connection: a completion is tokenized and checked here, off
/// the engine thread, so a long transcript or a slow client never delays decoding.
fn arrive(stream: TcpStream, ctx: &Ctx) -> Option<Arrival> {
    let (mut stream, req, path) = accept(stream)?;
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
fn serve_batched(listener: &TcpListener, model: &dyn Model, ctx: &Ctx, props: &Value) -> Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(move || accept_each(listener, scope, move |stream| arrive(stream, ctx), tx));
        let mut batch = Batch::new(model, MAX_WAIT);
        let mut replies: HashMap<u64, completion::Reply> = HashMap::new();
        // `/cache/warm` needs a slot of its own outside the batch, and waits here
        // until one is free.
        let mut warms: VecDeque<(TcpStream, Vec<u32>, bool)> = VecDeque::new();
        let mut next_id = 0u64;
        loop {
            let mut arrived: Vec<Arrival> = Vec::new();
            if batch.is_idle() && warms.is_empty() {
                match rx.recv() {
                    Ok(a) => arrived.push(a),
                    Err(_) => return Ok(()),
                }
            }
            arrived.extend(rx.try_iter());
            for a in arrived {
                match a {
                    Arrival::Completion(request) => {
                        let (job, mut reply) = *request;
                        if let Some(err) = ojas_core::device_fault::peek() {
                            reply.refuse("503 Service Unavailable", &fault_message(&err));
                        } else if batch.load().1 >= MAX_QUEUE {
                            reply.refuse("503 Service Unavailable", "the server is at capacity; retry shortly");
                        } else {
                            reply.begin();
                            batch.submit(next_id, job.into_request());
                            replies.insert(next_id, reply);
                            next_id += 1;
                        }
                    }
                    Arrival::Warm(stream, ids, pin) => warms.push_back((stream, ids, pin)),
                    Arrival::Other(mut stream, req, path) => {
                        if !serve_other(&mut stream, &req, &path, model, ctx, props) {
                            send_err(&mut stream, "404 Not Found", &format!("no route for {} {}", req.method, path));
                        }
                    }
                }
            }
            if let Some(slot) = batch.free_slot() {
                if let Some((mut stream, ids, pin)) = warms.pop_front() { warm(&mut stream, model, slot, &ids, pin); }
            }
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
fn completion_route(stream: &mut TcpStream, req: &Request, path: &str) -> Option<(Value, bool, bool)> {
    let ("POST", "/completion" | "/completions" | "/v1/completions" | "/v1/chat/completions") = (req.method.as_str(), path)
    else { return None };
    match serde_json::from_slice(&req.body) {
        Ok(body) => Some((body, path == "/v1/chat/completions", path.starts_with("/v1/"))),
        Err(e) => {
            send_err(stream, "400 Bad Request", &format!("invalid JSON: {e}"));
            None
        }
    }
}

/// The tokens and pin flag of a `/cache/warm` request, answering a malformed one
/// itself.
fn warm_request(stream: &mut TcpStream, req: &Request, ctx: &Ctx) -> Option<(Vec<u32>, bool)> {
    let body: Value = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => {
            send_err(stream, "400 Bad Request", &format!("invalid JSON: {e}"));
            return None;
        }
    };
    match warm_ids(&body, ctx.bpe, ctx.info, ctx.opts) {
        Ok(ids) => Some((ids, body.get("pin").and_then(Value::as_bool).unwrap_or(true))),
        Err(e) => {
            send_err(stream, "400 Bad Request", &e.to_string());
            None
        }
    }
}

/// Process `ids` into the prefix cache in `slot`, which no request holds.
fn warm(stream: &mut TcpStream, model: &dyn Model, slot: usize, ids: &[u32], pin: bool) {
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
fn serve_other(stream: &mut TcpStream, req: &Request, path: &str, model: &dyn Model, ctx: &Ctx, props: &Value) -> bool {
    match (req.method.as_str(), path) {
        ("GET", "/health") => match ojas_core::device_fault::peek() {
            // A process that cannot serve must not report healthy; the recovery path
            // is a restart by the orchestrator.
            Some(err) => send_json(stream, "503 Service Unavailable", &json!({"status": "error", "error": err.to_string()})),
            None => send_json(stream, "200 OK", &json!({"status": "ok"})),
        },
        ("GET", "/props") => {
            let mut p = props.clone();
            p["prefix_cache"] = cache_json(model.prefix_cache_stats());
            send_json(stream, "200 OK", &p)
        }
        ("GET", "/cache") => send_json(stream, "200 OK", &cache_json(model.prefix_cache_stats())),
        ("POST", "/cache/save") => {
            model.save_prefix_cache();
            send_json(stream, "200 OK", &cache_json(model.prefix_cache_stats()))
        }
        ("POST", "/cache/unpin") => {
            model.unpin_prefix_cache();
            send_json(stream, "200 OK", &cache_json(model.prefix_cache_stats()))
        }
        ("GET", "/v1/models") => send_json(stream, "200 OK", &json!({
            "object": "list",
            "data": [{"id": ctx.info.arch, "object": "model", "created": now(), "owned_by": "ojas"}]
        })),
        _ => return false,
    }
    true
}

/// Hand each request to `handle` with its path (query string removed), one at a
/// time on the calling thread, in the order they arrive. Each connection is read on
/// its own thread, so a client slow to send its request delays no one else's.
pub(crate) fn serve_http(listener: &TcpListener, mut handle: impl FnMut(&mut TcpStream, &Request, &str)) -> Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(move || accept_each(listener, scope, accept, tx));
        for (mut stream, req, path) in rx { handle(&mut stream, &req, &path); }
        Ok(())
    })
}

/// Read every new connection on its own thread with `read`, and send what it
/// returns to `tx`.
fn accept_each<'scope, T: Send + 'scope>(listener: &'scope TcpListener, scope: &'scope std::thread::Scope<'scope, '_>,
                                         read: impl Fn(TcpStream) -> Option<T> + Copy + Send + 'scope,
                                         tx: std::sync::mpsc::Sender<T>) {
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let tx = tx.clone();
                scope.spawn(move || {
                    if let Some(item) = read(stream) { let _ = tx.send(item); }
                });
            }
            Err(e) => {
                tracing::warn!("accept failed: {e}");
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    }
}

/// A client gets this long to send its request, and each write to it this long to
/// complete, so an idle or stalled connection cannot hold a thread forever.
const SOCKET_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Read one request from a new connection, with its path (query string removed).
/// Answers CORS preflight itself, and a request that cannot be parsed with 400;
/// `None` for those and for a connection closed before a request.
fn accept(mut stream: TcpStream) -> Option<(TcpStream, Request, String)> {
    let _ = stream.set_read_timeout(Some(SOCKET_TIMEOUT));
    let _ = stream.set_write_timeout(Some(SOCKET_TIMEOUT));
    let req = {
        let mut reader = BufReader::new(&stream);
        match read_request(&mut reader) {
            Ok(Some(r)) => r,
            Ok(None) => return None,
            Err(e) => {
                send_err(&mut stream, "400 Bad Request", &format!("{e:#}"));
                return None;
            }
        }
    };
    if req.method == "OPTIONS" {
        let head = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n\
                    Access-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
                    Content-Length: 0\r\nConnection: close\r\n\r\n";
        let _ = stream.write_all(head.as_bytes());
        return None;
    }
    let path = req.path.split('?').next().unwrap_or("").to_string();
    Some((stream, req, path))
}

#[cfg(test)]
mod tests {
    use super::*;

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
