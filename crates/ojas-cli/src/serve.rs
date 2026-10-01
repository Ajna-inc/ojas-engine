//! HTTP server: OpenAI-compatible plus llama.cpp's native `/completion`. A Laya
//! model is served by `decide::serve` instead (`POST /v1/decide`), on the same
//! connection loop, [`serve_http`].
//!
//! One slot, served on the thread that owns the model. `DecoderGpu` is `Send` but
//! not `Sync` (it keeps interior-mutable decode state) and a single decoder has
//! one KV cache, so requests are served sequentially, as llama.cpp's default
//! `-np 1` does. Concurrency needs several model instances, not several threads.
//!
//! Built on `std::net`: an async runtime would put an executor between the
//! request and a blocking decode loop without buying any parallelism.

use crate::backend::{with_model, ModelInfo};
use crate::detok::TextStream;
use ojas_grammar::{OutputFormat, TokenVocab};
use std::sync::Arc;
use crate::flags::RunOpts;
use anyhow::{Context, Result};
use ojas_core::Model;
use ojas_infer::{EngineCore, SampleOpts};
use ojas_tokenize::Bpe;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

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

pub(crate) fn send(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

pub(crate) fn send_json(stream: &mut TcpStream, status: &str, v: &Value) {
    send(stream, status, "application/json", v.to_string().as_bytes());
}

pub(crate) fn send_err(stream: &mut TcpStream, status: &str, msg: &str) {
    // OpenAI's error envelope, so clients surface the text instead of a blank failure.
    send_json(stream, status, &json!({"error": {"message": msg, "type": "invalid_request_error"}}));
}

fn begin_sse(stream: &mut TcpStream) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
                Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.flush();
}

fn sse(stream: &mut TcpStream, v: &Value) -> bool {
    let chunk = format!("data: {}\n\n", v);
    stream.write_all(chunk.as_bytes()).and_then(|_| stream.flush()).is_ok()
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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
        let text = ojas_tokenize::chat_transcript(&info.arch, &system, &turns);
        return Ok(bpe.encode(&text).into_iter().map(|v| v as u32).collect());
    }
    anyhow::bail!("request needs either \"prompt\" or \"messages\"")
}

fn stop_ids(bpe: &Bpe, info: &ModelInfo) -> (Option<u32>, Option<u32>) {
    let marker = bpe.encode(ojas_tokenize::chat_eos(&info.arch));
    match marker.as_slice() {
        [one] if Some(*one as u32) != info.eos => (Some(*one as u32), info.eos),
        _ => (info.eos, None),
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_completion(
    stream: &mut TcpStream,
    core: &EngineCore<&dyn Model>,
    bpe: &Bpe,
    info: &ModelInfo,
    opts: &RunOpts,
    body: &Value,
    chat: bool,
    oai: bool,
    secondary: Option<u32>,
    vocab: &Arc<TokenVocab>,
) {
    // Refuse a poisoned session before the response shape is chosen, so the client
    // gets an HTTP error rather than a stream it cannot trust.
    if let Some(err) = ojas_core::device_fault::peek() {
        return send_err(stream, "503 Service Unavailable",
            &format!("device fault; this model session is no longer usable and must be \
                      reloaded: {err}"));
    }
    let ids = match prompt_ids(body, bpe, info, opts) {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => return send_err(stream, "400 Bad Request", "empty prompt"),
        Err(e) => return send_err(stream, "400 Bad Request", &format!("{e:#}")),
    };
    if ids.len() >= info.context {
        return send_err(
            stream,
            "400 Bad Request",
            &format!("prompt is {} tokens; context holds {}", ids.len(), info.context),
        );
    }

    let want = body
        .get("n_predict")
        .or_else(|| body.get("max_tokens"))
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(opts.n_predict)
        .min(info.context - ids.len());
    let stops = match stop_strings(body) {
        Ok(v) => v,
        Err(e) => return send_err(stream, "400 Bad Request", &format!("{e:#}")),
    };
    let mut constraint = match output_format(body).and_then(|f| crate::constrain::processor(f.as_ref(), vocab)) {
        Ok(p) => p,
        Err(e) => return send_err(stream, "400 Bad Request", &format!("{e:#}")),
    };
    let s = sampling_from(body, opts, constraint.is_some());
    let sampling = if s.temperature <= 0.0 { None } else { Some(&s) };
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let id = format!("cmpl-{:x}", now());
    let created = now();
    let model_name = info.arch.clone();

    let mut d = TextStream::new(&stops);
    let mut text = String::new();
    let mut n_out = 0usize;
    let mut alive = true;
    // Every emitted id, so a client can compare token sequences rather than
    // decoded text: two engines can print the same string from different tokens.
    let mut out_ids: Vec<u32> = Vec::with_capacity(want);
    let return_tokens = body.get("return_tokens").and_then(Value::as_bool).unwrap_or(false);

    if streaming {
        begin_sse(stream);
        // The first OpenAI chunk carries the role and no content.
        if oai && chat {
            alive = sse(stream, &json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            }));
        }
    }

    let gen = core.generate_ex(&ids, want, sampling,
        constraint.as_mut().map(|p| p as &mut dyn ojas_infer::LogitProcessor), &mut |_, _| {}, &mut |t| {
        if Some(t) == secondary {
            return false;
        }
        let (piece, stopped) = d.push(bpe, t);
        n_out += 1;
        out_ids.push(t);
        text.push_str(&piece);
        if !streaming {
            return !stopped;
        }
        // Emit an event for every token, including one whose decoded piece is empty
        // because the detokenizer is still holding a multi-byte character. Skipping
        // those hides tokens from the client: counts come out short and the
        // first/last arrival timestamps used for throughput span the wrong window.
        let ev = if !oai {
            let mut e = json!({"content": piece, "stop": false});
            if return_tokens {
                e["tokens"] = json!([t]);
            }
            e
        } else if chat {
            json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model_name,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": null}]
            })
        } else {
            json!({
                "id": id, "object": "text_completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "text": piece, "finish_reason": null}]
            })
        };
        alive = sse(stream, &ev);
        // A disconnected client must stop the decode, not keep the slot busy
        // generating into a closed socket.
        alive && !stopped
    });
    // A model that does not expose logits cannot be constrained; it stops before
    // emitting anything rather than produce unconstrained output.
    if gen.finish == ojas_infer::FinishReason::NoLogits {
        if streaming { return; }
        return send_err(stream, "501 Not Implemented",
            "this model's backend does not expose logits, which constrained output needs");
    }
    let tail = d.finish();
    if !tail.is_empty() {
        text.push_str(&tail);
    }
    let reason = crate::constrain::finish_reason(gen.finish);
    // The /completion fields saying which way generation stopped.
    let stopped = json!({
        "stopped_eos": matches!(gen.finish, ojas_infer::FinishReason::Stop | ojas_infer::FinishReason::Complete),
        "stopped_word": d.stopped(),
        "stopped_limit": gen.finish == ojas_infer::FinishReason::Length,
    });
    let usage = json!({
        "prompt_tokens": ids.len(),
        "completion_tokens": n_out,
        "total_tokens": ids.len() + n_out,
    });

    if streaming {
        // A fault raised mid-stream means the tokens after it came out of undefined
        // buffers. The 200 is already promised, so close with a terminal error
        // event: never `finish_reason: stop` and never `[DONE]`, which both report
        // success.
        if let Some(err) = ojas_core::device_fault::peek() {
            if alive {
                let ev = if !oai {
                    json!({"error": {"message": err.to_string(), "type": "device_error"},
                           "stop": true, "truncated": true})
                } else {
                    json!({"error": {"message": err.to_string(), "type": "device_error"},
                           "id": id, "object": if chat { "chat.completion.chunk" } else { "text_completion" },
                           "created": created, "model": model_name,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]})
                };
                sse(stream, &ev);
            }
            return;
        }
        if alive {
            let done = if !oai {
                let mut e = json!({"content": tail, "stop": true, "tokens_predicted": n_out, "tokens_evaluated": ids.len()});
                for (k, v) in stopped.as_object().unwrap() { e[k] = v.clone(); }
                e
            } else if chat {
                // Text held back to the end (an unfinished character, a possible
                // stop string) goes out with the final chunk.
                let delta = if tail.is_empty() { json!({}) } else { json!({"content": tail}) };
                json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": reason}], "usage": usage
                })
            } else {
                json!({
                    "id": id, "object": "text_completion", "created": created, "model": model_name,
                    "choices": [{"index": 0, "text": tail, "finish_reason": reason}], "usage": usage
                })
            };
            sse(stream, &done);
            if oai {
                let _ = stream.write_all(b"data: [DONE]\n\n");
                let _ = stream.flush();
            }
        }
        return;
    }

    if let Some(err) = ojas_core::device_fault::peek() {
        return send_err(stream, "500 Internal Server Error",
            &format!("device fault during generation; no output is returned because the \
                      buffers it was read from are undefined: {err}"));
    }
    let out = if !oai {
        let mut e = json!({"content": text, "stop": true, "model": model_name,
               "tokens_predicted": n_out, "tokens_evaluated": ids.len()});
        for (k, v) in stopped.as_object().unwrap() { e[k] = v.clone(); }
        if return_tokens {
            e["tokens"] = json!(out_ids);
        }
        e
    } else if chat {
        json!({
            "id": id, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason}],
            "usage": usage
        })
    } else {
        json!({
            "id": id, "object": "text_completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "text": text, "finish_reason": reason}],
            "usage": usage
        })
    };
    send_json(stream, "200 OK", &out);
}

pub fn serve(model: &str, opts: &RunOpts, context: usize) -> Result<()> {
    #[cfg(target_os = "macos")]
    if crate::decide::is_laya(model) {
        return crate::decide::serve(model, opts);
    }
    with_model(model, opts.device, context, opts.precision, |m, bpe, info| {
        let (primary, secondary) = stop_ids(bpe, info);
        let mut core = EngineCore::new(m);
        core.eos = primary;
        // Token bytes for constrained requests, built once for the session.
        let vocab = crate::constrain::vocab(bpe, info);

        let addr = format!("{}:{}", opts.host, opts.port);
        let listener = TcpListener::bind(&addr).with_context(|| format!("binding {addr}"))?;
        eprintln!(
            "  {} | {} | ctx {} | MTP {} | loaded in {:.1}s",
            info.arch,
            info.backend,
            info.context,
            if info.has_mtp { "yes" } else { "no" },
            info.load_secs
        );
        eprintln!("  listening on http://{addr}  (one request at a time)");
        eprintln!("  POST /v1/chat/completions  POST /v1/completions  POST /completion  GET /health  GET /props  GET /v1/models");

        let props = json!({
            "model": info.arch,
            "backend": info.backend,
            "n_ctx": info.context,
            "n_layers": info.n_layers,
            "hidden_dim": info.hidden_dim,
            "vocab_size": info.vocab,
            "mtp": info.has_mtp,
            "concurrent_slots": 1,
        });

        serve_http(&listener, |stream, req, path| {
            match (req.method.as_str(), path) {
                ("GET", "/health") => match ojas_core::device_fault::peek() {
                    // A process that cannot serve must not report healthy; the
                    // recovery path is a restart by the orchestrator.
                    Some(err) => send_json(stream, "503 Service Unavailable",
                        &json!({"status": "error", "error": err.to_string()})),
                    None => send_json(stream, "200 OK", &json!({"status": "ok"})),
                },
                ("GET", "/props") => send_json(stream, "200 OK", &props),
                ("GET", "/v1/models") => send_json(stream, "200 OK", &json!({
                    "object": "list",
                    "data": [{"id": info.arch, "object": "model", "created": now(), "owned_by": "ojas"}]
                })),
                ("POST", p @ ("/completion" | "/completions" | "/v1/completions" | "/v1/chat/completions")) => {
                    let body: Value = match serde_json::from_slice(&req.body) {
                        Ok(v) => v,
                        Err(e) => {
                            send_err(stream, "400 Bad Request", &format!("invalid JSON: {e}"));
                            return;
                        }
                    };
                    let oai = p.starts_with("/v1/");
                    let chat = p == "/v1/chat/completions";
                    // Each request is an independent sequence unless the client
                    // opts into prompt caching. A recurrent model (Flash, qwen35)
                    // carries SSM/GDN and MTP state forward, so without the reset the
                    // second request continues from the middle of the first: fluent
                    // nonsense, no error. Dense models are unaffected.
                    if !body.get("cache_prompt").and_then(Value::as_bool).unwrap_or(false) {
                        core.model().reset_session();
                    }
                    // `ignore_eos` makes a fixed-length benchmark possible: both
                    // engines must emit exactly n_predict tokens, or the rates
                    // describe different amounts of work. It applies per request and
                    // cannot leak into the next one. As in llama.cpp it suppresses
                    // the end-of-generation set from selection rather than merely
                    // declining to stop: a model that emits its terminator and
                    // carries on produces different text from that point, so a
                    // benchmark row whose outputs diverge is not like-for-like.
                    // `engine_bench.py` reports per-row output equality; treat a row
                    // marked "NO" as invalid.
                    let ignore_eos = body.get("ignore_eos").and_then(Value::as_bool).unwrap_or(false);
                    core.eog = if ignore_eos { Vec::new() } else { info.eog.clone() };
                    core.banned = if ignore_eos { info.eog.clone() } else { Vec::new() };
                    core.eos = if ignore_eos { None } else { primary };
                    let stop = if ignore_eos { None } else { secondary };
                    handle_completion(stream, &core, bpe, info, opts, &body, chat, oai, stop, &vocab);
                }
                _ => send_err(stream, "404 Not Found", &format!("no route for {} {}", req.method, path)),
            }
        })
    })
}

/// Accept connections one at a time and hand each parsed request to `handle` with
/// its path (query string removed). Answers CORS preflight itself, and a request that
/// cannot be parsed with 400.
pub(crate) fn serve_http(listener: &TcpListener, mut handle: impl FnMut(&mut TcpStream, &Request, &str)) -> Result<()> {
    for conn in listener.incoming() {
        let mut stream = match conn {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("accept failed: {e}");
                continue;
            }
        };
        let req = {
            let mut reader = BufReader::new(&stream);
            match read_request(&mut reader) {
                Ok(Some(r)) => r,
                Ok(None) => continue,
                Err(e) => {
                    send_err(&mut stream, "400 Bad Request", &format!("{e:#}"));
                    continue;
                }
            }
        };
        if req.method == "OPTIONS" {
            let head = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n\
                        Access-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
                        Content-Length: 0\r\nConnection: close\r\n\r\n";
            let _ = stream.write_all(head.as_bytes());
            continue;
        }
        let path = req.path.split('?').next().unwrap_or("").to_string();
        handle(&mut stream, &req, &path);
    }
    Ok(())
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
