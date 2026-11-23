//! One completion request: what to generate (`Job`) and how to answer it (`Reply`).
//!
//! The split lets one request run either way: alone through `EngineCore`, with the
//! reply fed from its token callback, or in a batch of slots, with the job submitted
//! to the batch and the reply kept until the batch reports the request done. Either
//! way the client sees the same response format, stop handling and usage.
//!
//! A reply writes through its own thread (`Outbox`), so a client that reads slowly
//! or not at all holds up only its own request, never the thread decoding for
//! everyone.

use super::anthropic;
use super::logprobs::{self, Logprobs};
use super::tools::{self, ToolReply};
use super::{error_response, json_response, metrics, now, output_format, prompt_docs, prompt_ids, prompt_marks,
            request_id_header, restore_json, sampling_from, send_err, sse_event, sse_head, stop_strings, Conn};
use crate::backend::ModelInfo;
use crate::detok::TextStream;
use crate::flags::RunOpts;
use ojas_grammar::TokenVocab;
use ojas_infer::{batch, FinishReason, Generation, LogitProcessor, SampleOpts};
use ojas_tokenize::Bpe;
use serde_json::{json, Value};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

/// Writes a reply may queue ahead of its client; one that falls this far behind is
/// treated as gone.
const OUTBOX: usize = 4096;

/// Tokens between checks that a client still waiting for a whole response is there.
const ALIVE_EVERY: usize = 32;

/// Tells apart requests that arrive within one second, in their ids.
static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// What every request on this server shares.
pub(crate) struct Ctx<'a> {
    pub(crate) bpe: &'a Bpe,
    pub(crate) info: &'a ModelInfo,
    pub(crate) opts: &'a RunOpts,
    pub(crate) vocab: &'a Arc<TokenVocab>,
    /// The end-of-generation id, and a chat template's end-of-turn id when it is a
    /// different single token.
    pub(crate) primary: Option<u32>,
    pub(crate) secondary: Option<u32>,
    /// Whether the model keeps a prompt-prefix cache.
    pub(crate) prefix_cache: bool,
    /// What the server needs to take images, or why it takes none.
    pub(crate) images: &'a Result<super::images::Vision, String>,
    /// The model file this server answers from.
    pub(crate) identity: &'a super::Identity,
    /// Longest a request may take from its arrival, when the server sets one.
    pub(crate) timeout: Option<std::time::Duration>,
}

/// What a request asks the model to generate.
pub(crate) struct Job {
    pub(crate) ids: Vec<u32>,
    pub(crate) marks: Vec<usize>,
    pub(crate) docs: Vec<(usize, usize)>,
    /// The prompt's images, when it has any.
    pub(crate) images: Option<ojas_infer::image::PromptImages>,
    pub(crate) want: usize,
    pub(crate) sampling: Option<SampleOpts>,
    pub(crate) constraint: Option<Box<dyn LogitProcessor + Send>>,
    /// Whether the prompt may restore from the prefix cache. Off, the request is an
    /// independent sequence: a recurrent model would otherwise continue from the
    /// previous request's state, which reads as fluent nonsense with no error.
    pub(crate) reuse: bool,
    /// Ids that end generation, and ids never chosen.
    pub(crate) eog: Vec<u32>,
    pub(crate) eos: Option<u32>,
    pub(crate) banned: Vec<u32>,
    /// Bulk work nobody is waiting on: it runs after every other request.
    pub(crate) background: bool,
}

impl Job {
    /// The ids `EngineCore::is_stop` would stop at, as one list.
    fn stop_ids(&self) -> Vec<u32> {
        if self.eog.is_empty() { self.eos.into_iter().collect() } else { self.eog.clone() }
    }

    pub(crate) fn into_request(self) -> batch::Request {
        let stop = self.stop_ids();
        batch::Request {
            prompt: self.ids, max_tokens: self.want, opts: self.sampling,
            processor: self.constraint.map(|p| p as Box<dyn LogitProcessor>), stop,
            banned: self.banned, marks: self.marks, docs: self.docs, reuse: self.reuse, images: self.images,
            background: self.background,
        }
    }
}

/// A connection's writes, made on their own thread.
struct Outbox {
    tx: mpsc::SyncSender<Vec<u8>>,
    gone: Arc<AtomicBool>,
    /// The connection itself, kept to notice a client that hung up.
    stream: Conn,
    tx_ping: mpsc::SyncSender<Vec<u8>>,
    /// Ends the keep-alive thread.
    quiet: Arc<AtomicBool>,
}

impl Drop for Outbox {
    fn drop(&mut self) { self.quiet.store(true, Ordering::Relaxed); }
}

impl Outbox {
    fn new(stream: &Conn) -> std::io::Result<Self> {
        let (mut writer, stream) = (stream.try_clone()?, stream.try_clone()?);
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(OUTBOX);
        let gone = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&gone);
        std::thread::Builder::new().name("http-reply".into()).spawn(move || {
            for bytes in rx {
                if writer.write_all(&bytes).and_then(|_| writer.flush()).is_err() {
                    flag.store(true, Ordering::Relaxed);
                    return;
                }
            }
        })?;
        Ok(Outbox { tx_ping: tx.clone(), tx, gone, stream, quiet: Arc::new(AtomicBool::new(false)) })
    }

    /// Queue bytes for the client; false when it is gone or too far behind.
    fn send(&self, bytes: Vec<u8>) -> bool {
        if self.gone.load(Ordering::Relaxed) || self.tx.try_send(bytes).is_err() {
            self.gone.store(true, Ordering::Relaxed);
            return false;
        }
        true
    }

    /// Send an SSE ping every few seconds until `hush`: a client reads a long silent
    /// prompt pass as a dead connection.
    fn keep_alive(&self) {
        let (tx, quiet, gone) = (self.tx_ping.clone(), Arc::clone(&self.quiet), Arc::clone(&self.gone));
        let ping = anthropic::event("ping", &json!({"type": "ping"}));
        let _ = std::thread::Builder::new().name("http-ping".into()).spawn(move || {
            let mut waited = 0;
            while !quiet.load(Ordering::Relaxed) && !gone.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
                waited += 1;
                if waited % 50 == 0 && !quiet.load(Ordering::Relaxed) && tx.try_send(ping.clone()).is_err() { return; }
            }
        });
    }

    fn hush(&self) { self.quiet.store(true, Ordering::Relaxed); }

    /// Whether the client is still connected, without waiting.
    fn connected(&self) -> bool {
        !self.gone.load(Ordering::Relaxed) && !self.stream.closed()
    }
}

/// How a request is answered: the connection, the response format, and the text so
/// far.
pub(crate) struct Reply {
    out: Outbox,
    streaming: bool,
    oai: bool,
    chat: bool,
    return_tokens: bool,
    id: String,
    created: u64,
    model_name: String,
    prompt_tokens: usize,
    /// A chat template's end-of-turn id: ends the reply without being part of it.
    end_of_turn: Option<u32>,
    text_stream: TextStream,
    text: String,
    out_ids: Vec<u32>,
    /// Token log-probabilities, when the request asked for them.
    logprobs: Option<Logprobs>,
    alive: bool,
    /// Set for a Messages API request: how to turn the text into content blocks.
    anth: Option<anthropic::Shape>,
    /// Set for a chat request with tools: how to find the calls in the text.
    tools: Option<ToolReply>,
    /// Set for an OCR request: how to turn the model's markup into a page.
    ocr: Option<super::ocr_route::Shape>,
    began: std::time::Instant,
    /// How long the first token took, once there is one.
    first_token: Option<std::time::Duration>,
    /// When the request runs out of time, and whether it did.
    deadline: Option<std::time::Instant>,
    timed_out: std::cell::Cell<bool>,
    /// The caller's request id, for the log and as a response header line.
    request_id: Option<String>,
}

/// Parse a completion request, answering it at once with an error when it is
/// malformed. Nothing is sent on success until `Reply::begin`.
pub(crate) fn start(stream: Conn, ctx: &Ctx, body: &Value, chat: bool, oai: bool) -> Option<(Job, Reply)> {
    let (bpe, info, opts) = (ctx.bpe, ctx.info, ctx.opts);
    let mut stream = stream;
    let began = std::time::Instant::now();
    let converted;
    let (body, anth, tools) = if body.get(super::MESSAGES_MARK).is_some() {
        match anthropic::convert(body, &opts.system) {
            Ok(c) => { converted = c.body; (&converted, Some(c.shape), None) }
            Err(e) => return refuse(stream, &e),
        }
    } else if chat && oai {
        match tools::convert(body, &info.arch, &opts.system) {
            Ok(Some((internal, reply))) => { converted = internal; (&converted, None, Some(reply)) }
            Ok(None) => (body, None, None),
            Err(e) => return refuse(stream, &e),
        }
    } else { (body, None, None) };
    let ocr = match super::ocr_route::Shape::of(ctx, body) {
        Ok(shape) => shape,
        Err(why) => {
            metrics::failed();
            send_err(&mut stream, "501 Not Implemented", &why);
            return None;
        }
    };
    // A request with image parts brings its own prompt, boundaries and image rows.
    let built = match super::images::build(ctx, body) {
        Ok(b) => b,
        Err(e) => return refuse(stream, &format!("{e:#}")),
    };
    let (built_ids, built_marks, images) = match built {
        Some(b) => (Some(b.ids), Some(b.marks), Some(b.images)),
        None => (None, None, None),
    };
    let ids = match built_ids.map_or_else(|| prompt_ids(body, bpe, info, opts), Ok) {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => return refuse(stream, "empty prompt"),
        Err(e) => return refuse(stream, &format!("{e:#}")),
    };
    if ids.len() >= info.context {
        return refuse(stream, &format!("prompt is {} tokens; context holds {}", ids.len(), info.context));
    }
    let stops = match stop_strings(body) {
        Ok(v) => v,
        Err(e) => return refuse(stream, &format!("{e:#}")),
    };
    let constraint = match output_format(body).and_then(|f| crate::constrain::processor(f.as_ref(), ctx.vocab)) {
        Ok(p) => p,
        Err(e) => return refuse(stream, &format!("{e:#}")),
    };
    let sampling = sampling_from(body, opts, constraint.is_some());
    let (logprobs, constraint) = match logprobs::attach(body, chat, constraint) {
        Ok(v) => v,
        Err(e) => return refuse(stream, &format!("{e:#}")),
    };
    let want = body.get("n_predict").or_else(|| body.get("max_tokens")).and_then(Value::as_u64)
        .map(|v| v as usize).unwrap_or(opts.n_predict).min(info.context - ids.len());
    // `ignore_eos` makes a fixed-length benchmark possible: both engines must emit
    // exactly n_predict tokens, or the rates describe different amounts of work. It
    // suppresses the end-of-generation set from selection rather than merely
    // declining to stop: a model that emits its terminator and carries on produces
    // different text from that point, so a benchmark whose outputs diverge is not
    // like-for-like.
    let ignore_eos = body.get("ignore_eos").and_then(Value::as_bool).unwrap_or(false);
    // `"priority": "batch"`, or the same in an `X-Priority` header a proxy can set.
    let background = match body.get("priority").and_then(Value::as_str) {
        None | Some("interactive") => stream.background,
        Some("batch") => true,
        Some(other) => return refuse(stream, &format!("unknown priority \"{other}\" (want \"interactive\" or \"batch\")")),
    };
    // A request may ask for less time than the server allows, never more.
    let asked = body.get("timeout_ms").and_then(Value::as_f64).filter(|ms| *ms > 0.0)
        .map(|ms| std::time::Duration::from_secs_f64(ms / 1e3));
    let timeout = match (ctx.timeout, asked) {
        (Some(most), Some(asked)) => Some(most.min(asked)),
        (most, asked) => most.or(asked),
    };
    let job = Job {
        marks: built_marks.unwrap_or_else(|| if ctx.prefix_cache { prompt_marks(body, bpe, info, opts) } else { Vec::new() }),
        docs: if ctx.prefix_cache && images.is_none() { prompt_docs(body, bpe, info, opts) } else { Vec::new() },
        images,
        want,
        sampling: (sampling.temperature > 0.0).then_some(sampling),
        constraint,
        reuse: body.get("cache_prompt").and_then(Value::as_bool).unwrap_or(ctx.prefix_cache),
        eog: if ignore_eos { Vec::new() } else { info.eog.clone() },
        eos: if ignore_eos { None } else { ctx.primary },
        banned: if ignore_eos { info.eog.clone() } else { Vec::new() },
        background,
        ids,
    };
    let out = match Outbox::new(&stream) {
        Ok(out) => out,
        Err(e) => {
            metrics::failed();
            send_err(&mut stream, "500 Internal Server Error", &format!("cannot answer this connection: {e}"));
            return None;
        }
    };
    let reply = Reply {
        streaming: body.get("stream").and_then(Value::as_bool).unwrap_or(false), oai, chat,
        return_tokens: body.get("return_tokens").and_then(Value::as_bool).unwrap_or(false),
        id: format!("cmpl-{:x}{:04x}", now(), SERIAL.fetch_add(1, Ordering::Relaxed) & 0xffff), created: now(),
        model_name: info.arch.clone(),
        prompt_tokens: job.ids.len(), end_of_turn: if ignore_eos { None } else { ctx.secondary },
        text_stream: TextStream::new(&stops), text: String::new(), out_ids: Vec::with_capacity(want), alive: true,
        logprobs, out, anth, tools, ocr, began, first_token: None, deadline: timeout.map(|t| began + t),
        timed_out: std::cell::Cell::new(false), request_id: stream.request_id.clone(),
    };
    Some((job, reply))
}

fn refuse(mut stream: Conn, msg: &str) -> Option<(Job, Reply)> {
    metrics::failed();
    send_err(&mut stream, "400 Bad Request", msg);
    None
}

impl Reply {
    /// Commit to answering: a stream's header and first chunk go out now, a whole
    /// response only at `finish`.
    pub(crate) fn begin(&mut self) {
        eprintln!("  request {}{}: {} prompt tokens", self.id, self.tag(), self.prompt_tokens);
        if !self.streaming { return; }
        self.alive = self.out.send(sse_head(&self.headers()));
        if self.anth.is_some() {
            let start = anthropic::message(&self.id, &self.model_name, Vec::new(), "", self.prompt_tokens, 0);
            self.alive &= self.out.send(anthropic::event("message_start", &json!({"type": "message_start", "message": start})));
            self.out.keep_alive();
            return;
        }
        // The first OpenAI chunk carries the role and no content.
        if self.oai && self.chat {
            let first = json!({
                "id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            });
            self.alive &= self.out.send(sse_event(&first));
        }
    }

    /// The caller's request id as a response header line, or nothing.
    fn headers(&self) -> String { request_id_header(self.request_id.as_deref()) }

    /// The caller's request id as it is written in a log line.
    fn tag(&self) -> String { self.request_id.as_ref().map(|id| format!(" [{id}]")).unwrap_or_default() }

    /// Answer with an error instead, before `begin`.
    pub(crate) fn refuse(self, status: &str, msg: &str) {
        if status.starts_with("429") { metrics::refused() } else { metrics::failed() }
        self.out.send(error_response(status, msg, &self.headers()));
    }

    /// Whether the request ran out of time; remembered, so the reply says so.
    fn expired(&self) -> bool {
        if self.deadline.is_some_and(|d| std::time::Instant::now() >= d) { self.timed_out.set(true); }
        self.timed_out.get()
    }

    /// Whether to go on with the request: the client is still waiting and there is time left.
    pub(crate) fn connected(&self) -> bool { !self.expired() && self.alive && self.out.connected() }

    /// End a request that never started: 504 when it ran out of time waiting,
    /// nothing when its client left.
    pub(crate) fn give_up(self) {
        if self.timed_out.get() {
            self.fail("504 Gateway Timeout", "timeout_error", "the request ran out of time before it was answered");
        }
    }

    /// One chunk of a streamed chat completion.
    fn chat_chunk(&self, delta: Value, finish: Value) -> Value {
        json!({
            "id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
        })
    }

    /// Take one generated token; false ends the request (end of turn, a stop
    /// string, or a client that went away).
    pub(crate) fn token(&mut self, bpe: &Bpe, t: u32) -> bool {
        if Some(t) == self.end_of_turn || self.expired() { return false; }
        if self.first_token.is_none() { self.first_token = Some(self.began.elapsed()); }
        let (piece, stopped) = self.text_stream.push(bpe, t);
        self.out_ids.push(t);
        self.text.push_str(&piece);
        let logprobs = self.logprobs.as_mut().map(|l| l.token(|id| bpe.decode_bytes(id as usize), t));
        if self.ocr.as_mut().is_some_and(|page| !page.token(&self.out_ids)) { return false; }
        if self.anth.is_some() {
            // Buffered: blocks are cut from the whole text. A ping keeps a long reply's connection warm.
            if self.streaming && self.out_ids.len().is_multiple_of(ALIVE_EVERY) {
                self.alive = self.out.send(anthropic::event("ping", &json!({"type": "ping"})));
            } else if self.out_ids.len().is_multiple_of(ALIVE_EVERY) {
                self.alive = self.out.connected();
            }
            return self.alive && !stopped;
        }
        if let Some(tools) = self.tools.as_mut() {
            // Only text that cannot be part of a call is shown as it comes.
            let shown = tools.delta(&self.text).to_string();
            if self.streaming && !shown.is_empty() {
                self.alive = self.out.send(sse_event(&self.chat_chunk(json!({"content": shown}), Value::Null)));
            } else if self.out_ids.len().is_multiple_of(ALIVE_EVERY) {
                self.alive = self.out.connected();
            }
            return self.alive && !stopped;
        }
        if !self.streaming {
            if self.out_ids.len().is_multiple_of(ALIVE_EVERY) { self.alive = self.out.connected(); }
            return self.alive && !stopped;
        }
        // Emit an event for every token, including one whose decoded piece is empty
        // because the detokenizer is still holding a multi-byte character. Skipping
        // those hides tokens from the client: counts come out short and the first
        // and last arrival timestamps used for throughput span the wrong window.
        let mut ev = if !self.oai {
            let mut e = json!({"content": piece, "stop": false});
            if self.return_tokens { e["tokens"] = json!([t]); }
            e
        } else if self.chat {
            json!({
                "id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model_name,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": null}]
            })
        } else {
            json!({
                "id": self.id, "object": "text_completion", "created": self.created, "model": self.model_name,
                "choices": [{"index": 0, "text": piece, "finish_reason": null}]
            })
        };
        if let Some(l) = logprobs { logprobs::put(&mut ev, self.oai, l); }
        self.alive = self.out.send(sse_event(&ev));
        // A disconnected client must stop the decode, not keep it generating into a
        // closed socket.
        self.alive && !stopped
    }

    /// Answer the request now that generation has ended.
    pub(crate) fn finish(mut self, gen: &Generation) {
        match gen.finish {
            // A model that does not expose logits cannot be constrained; it stops
            // before emitting anything rather than produce unconstrained output.
            FinishReason::NoLogits => {
                return self.fail("501 Not Implemented", "server_error",
                    "this model's backend does not expose logits, which this request needs");
            }
            FinishReason::Refused => {
                return self.fail("400 Bad Request", "invalid_request_error", "the model cannot run this request");
            }
            _ => {}
        }
        // A fault means the tokens after it came out of undefined buffers.
        if let Some(err) = ojas_core::device_fault::peek() {
            return self.fail("500 Internal Server Error", "device_error", &format!(
                "device fault during generation; no output is returned because the buffers it was read from are \
                 undefined: {err}"));
        }
        if self.timed_out.get() {
            return self.fail("504 Gateway Timeout", "timeout_error", "the request ran out of time before it was answered");
        }
        let tail = self.text_stream.finish();
        self.text.push_str(&tail);
        let steps = self.logprobs.as_ref().map(Logprobs::steps);
        let page = match self.ocr.take().map(|o| o.page(&self.text, steps.as_deref(), gen.finish == FinishReason::Length)).transpose() {
            Ok(page) => page,
            Err(why) => return self.fail("501 Not Implemented", "not_supported_error", &why),
        };
        let n_out = self.out_ids.len();
        eprintln!("  request {}{}: {} tokens out, {} cached, {:.1}s total", self.id, self.tag(), n_out, gen.cached_tokens,
                  self.began.elapsed().as_secs_f32());
        metrics::answered(&metrics::Answered {
            prompt_tokens: self.prompt_tokens, cached_tokens: gen.cached_tokens, completion_tokens: n_out,
            first_token: self.first_token, total: self.began.elapsed(),
        });
        let headers = self.headers();
        let reason = crate::constrain::finish_reason(gen.finish);
        // The /completion fields saying which way generation stopped.
        let stopped = json!({
            "stopped_eos": matches!(gen.finish, FinishReason::Stop | FinishReason::Complete),
            "stopped_word": self.text_stream.stopped(),
            "stopped_limit": gen.finish == FinishReason::Length,
        });
        let mut usage = json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": n_out,
            "total_tokens": self.prompt_tokens + n_out,
            "prompt_tokens_details": {"cached_tokens": gen.cached_tokens},
        });
        if let Some(r) = &gen.prompt_cache { usage["prompt_cache"] = restore_json(r); }
        let (id, created, model) = (&self.id, self.created, &self.model_name);

        if let Some(mut page) = page {
            page["usage"] = usage;
            self.out.send(json_response("200 OK", &page, &headers));
            return;
        }

        if let Some(shape) = &self.anth {
            self.out.hush();
            let parsed = shape.parse(&self.text);
            let stop_reason = if !parsed.calls.is_empty() { "tool_use" }
                else if gen.finish == FinishReason::Length { "max_tokens" }
                else if self.text_stream.stopped() { "stop_sequence" } else { "end_turn" };
            let content = anthropic::blocks(&parsed, &id.replace("cmpl-", ""));
            if self.streaming {
                if self.alive { self.out.send(anthropic::stream_events(&content, stop_reason, n_out)); }
            } else {
                let m = anthropic::message(id, model, content, stop_reason, self.prompt_tokens, n_out);
                self.out.send(json_response("200 OK", &m, &headers));
            }
            return;
        }

        if let Some(tools) = &self.tools {
            let (parsed, rest) = tools.finish(&self.text);
            let call_id = id.trim_start_matches("cmpl-");
            let reason = if parsed.calls.is_empty() { reason } else { "tool_calls" };
            if self.streaming {
                if !self.alive { return; }
                if !rest.is_empty() { self.out.send(sse_event(&self.chat_chunk(json!({"content": rest}), Value::Null))); }
                if !parsed.calls.is_empty() {
                    let calls = tools::calls_json(&parsed.calls, call_id, true);
                    self.out.send(sse_event(&self.chat_chunk(json!({"tool_calls": calls}), Value::Null)));
                }
                let mut done = self.chat_chunk(json!({}), json!(reason));
                done["usage"] = usage;
                self.out.send(sse_event(&done));
                self.out.send(b"data: [DONE]\n\n".to_vec());
                return;
            }
            // A reply that is only calls has no content, as the API has it.
            let content = if parsed.text.is_empty() && !parsed.calls.is_empty() { Value::Null } else { json!(parsed.text) };
            let mut message = json!({"role": "assistant", "content": content});
            if !parsed.calls.is_empty() { message["tool_calls"] = json!(tools::calls_json(&parsed.calls, call_id, false)); }
            let mut out = json!({
                "id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": reason}], "usage": usage
            });
            if let Some(l) = &self.logprobs { logprobs::put(&mut out, self.oai, l.whole()); }
            self.out.send(json_response("200 OK", &out, &headers));
            return;
        }

        if self.streaming {
            if !self.alive { return; }
            let done = if !self.oai {
                let mut e = json!({"content": tail, "stop": true, "tokens_predicted": n_out,
                    "tokens_evaluated": self.prompt_tokens, "tokens_cached": gen.cached_tokens});
                for (k, v) in stopped.as_object().unwrap() { e[k] = v.clone(); }
                e
            } else if self.chat {
                // Text held back to the end (an unfinished character, a possible stop
                // string) goes out with the final chunk.
                let delta = if tail.is_empty() { json!({}) } else { json!({"content": tail}) };
                json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": reason}], "usage": usage
                })
            } else {
                json!({
                    "id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{"index": 0, "text": tail, "finish_reason": reason}], "usage": usage
                })
            };
            self.out.send(sse_event(&done));
            if self.oai { self.out.send(b"data: [DONE]\n\n".to_vec()); }
            return;
        }
        let mut out = if !self.oai {
            let mut e = json!({"content": self.text, "stop": true, "model": model, "tokens_predicted": n_out,
                "tokens_evaluated": self.prompt_tokens, "tokens_cached": gen.cached_tokens});
            for (k, v) in stopped.as_object().unwrap() { e[k] = v.clone(); }
            if self.return_tokens { e["tokens"] = json!(self.out_ids); }
            e
        } else if self.chat {
            json!({
                "id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": self.text}, "finish_reason": reason}],
                "usage": usage
            })
        } else {
            json!({
                "id": id, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": self.text, "finish_reason": reason}],
                "usage": usage
            })
        };
        if let Some(l) = &self.logprobs { logprobs::put(&mut out, self.oai, l.whole()); }
        self.out.send(json_response("200 OK", &out, &headers));
    }

    /// End the request with an error: an HTTP error before any of a stream was
    /// sent, else a terminal error event, never `finish_reason: stop` or `[DONE]`,
    /// which both report success.
    fn fail(self, status: &str, kind: &str, msg: &str) {
        if !self.streaming { return self.refuse(status, msg); }
        metrics::failed();
        if !self.alive { return; }
        if self.anth.is_some() {
            self.out.hush();
            let ev = json!({"type": "error", "error": {"type": kind, "message": msg}});
            self.out.send(anthropic::event("error", &ev));
            return;
        }
        let ev = if !self.oai {
            json!({"error": {"message": msg, "type": kind}, "stop": true, "truncated": true})
        } else {
            json!({"error": {"message": msg, "type": kind},
                   "id": self.id, "object": if self.chat { "chat.completion.chunk" } else { "text_completion" },
                   "created": self.created, "model": self.model_name,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]})
        };
        self.out.send(sse_event(&ev));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::{TcpListener, TcpStream};

    fn info() -> ModelInfo {
        ModelInfo {
            arch: "qwen3".into(), eos: Some(2), eog: vec![2], vocab: 16, context: 64, backend: "cpu",
            has_mtp: false, n_layers: 1, hidden_dim: 1, load_secs: 0.0,
        }
    }

    /// The server's end of a loopback connection, and the client's.
    fn connection() -> (Conn, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        (Conn::plain(listener.accept().unwrap().0), client)
    }

    /// Everything the client receives until the reply's writer closes.
    fn received(mut client: TcpStream) -> String {
        let mut text = String::new();
        client.read_to_string(&mut text).unwrap();
        text
    }

    fn generation(tokens: Vec<u32>, finish: FinishReason) -> Generation {
        Generation { tokens, finish, cached_tokens: 0, prompt_cache: None }
    }

    /// Start a request with `body` on a fresh connection, with end-of-turn id 7.
    fn started(body: Value, chat: bool) -> (Job, Reply, TcpStream) {
        let (bpe, info, opts) = (Bpe::default(), info(), RunOpts::default());
        let vocab = Arc::new(TokenVocab::new(Vec::new(), Vec::new(), &[]));
        let no_images = Err("no projector".to_string());
        let identity = crate::serve::Identity { file: "m.gguf".into(), sha256: "00".into() };
        let ctx = Ctx { bpe: &bpe, info: &info, opts: &opts, vocab: &vocab, primary: Some(2), secondary: Some(7),
                        prefix_cache: false, images: &no_images, identity: &identity, timeout: None };
        let (mut server, client) = connection();
        server.request_id = Some("req-7".into());
        let (job, reply) = start(server, &ctx, &body, chat, true).expect("a valid request");
        (job, reply, client)
    }

    #[test]
    fn a_streamed_chat_reply_has_its_role_chunk_tokens_final_chunk_and_done() {
        let (job, mut reply, client) = started(json!({"prompt": [1, 3, 4], "stream": true, "max_tokens": 4}), true);
        assert_eq!((job.ids.len(), job.want, job.eog.clone()), (3, 4, vec![2]));
        reply.begin();
        let bpe = Bpe::default();
        assert!(reply.token(&bpe, 5) && reply.token(&bpe, 6));
        reply.finish(&generation(vec![5, 6], FinishReason::Length));
        let text = received(client);
        assert!(text.starts_with("HTTP/1.1 200 OK") && text.contains("text/event-stream"));
        assert!(text.contains(r#""delta":{"role":"assistant"}"#));
        assert_eq!(text.matches(r#""finish_reason":null"#).count(), 3, "the role chunk and one per token");
        assert!(text.contains(r#""finish_reason":"length""#) && text.contains(r#""completion_tokens":2"#));
        assert!(text.ends_with("data: [DONE]\n\n"));
    }

    #[test]
    fn requested_log_probabilities_ride_each_chunk_and_the_whole_response() {
        for stream in [true, false] {
            let body = json!({"prompt": [1, 3], "stream": stream, "logprobs": true, "top_logprobs": 2});
            let (mut job, mut reply, client) = started(body, true);
            reply.begin();
            // The engine shows the request's processor each step's logits, then the token.
            let mut observer = job.constraint.take().expect("log-probabilities are observed from the logit hook");
            observer.process(&mut [0.0, 0.0, 0.0, 0.0, 1.0, 3.0]);
            observer.accept(5);
            assert!(reply.token(&Bpe::default(), 5));
            reply.finish(&generation(vec![5], FinishReason::Length));
            let text = received(client);
            assert_eq!(text.matches(r#""logprobs":{"content":[{"#).count(), 1, "{text}");
            assert_eq!(text.matches(r#"{"token":"","logprob":-"#).count(), 3, "the token and two alternatives: {text}");
        }
        let (job, _, _) = started(json!({"prompt": [1, 3]}), true);
        assert!(job.constraint.is_none(), "nothing observes a request that did not ask");
    }

    #[test]
    fn a_request_out_of_time_is_answered_504_and_its_id_is_echoed() {
        let (_, mut reply, client) = started(json!({"prompt": [1, 3], "timeout_ms": 1}), true);
        reply.begin();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(!reply.connected() && !reply.token(&Bpe::default(), 5));
        reply.finish(&generation(Vec::new(), FinishReason::Caller));
        let text = received(client);
        assert!(text.starts_with("HTTP/1.1 504") && text.contains("timeout_error"), "{text}");
        assert!(text.contains("X-Request-Id: req-7\r\n"), "{text}");
    }

    #[test]
    fn a_reply_with_tools_is_a_message_with_tool_calls() {
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object",
            "properties": {"n": {"type": "integer"}}}}}]);
        for stream in [false, true] {
            // The default tokenizer neither encodes nor decodes text, so the prompt is given
            // as ids and the reply's text is set as generated.
            let body = json!({"messages": [{"role": "user", "content": "x"}], "tools": tools});
            let (_, mut reply, client) = started(json!({"prompt": [1, 3], "stream": stream}), true);
            reply.tools = tools::convert(&body, "qwen3", "").unwrap().map(|(_, reply)| reply);
            reply.begin();
            reply.text = "On it.\n\n<tool_call>\n<function=f>\n<parameter=n>\n4\n</parameter>\n</function>\n</tool_call>".into();
            reply.finish(&generation(vec![5], FinishReason::Stop));
            let text = received(client);
            assert!(text.contains(r#""finish_reason":"tool_calls""#), "{text}");
            assert!(text.contains(r#""function":{"name":"f","arguments":"{\"n\":4}"}"#), "{text}");
            assert!(text.contains(r#""content":"On it.""#), "{text}");
            assert_eq!(text.contains("data: [DONE]"), stream);
        }
    }

    #[test]
    fn the_end_of_turn_id_ends_a_reply_without_being_part_of_it() {
        let (_, mut reply, client) = started(json!({"prompt": [1, 3]}), true);
        reply.begin();
        let bpe = Bpe::default();
        assert!(reply.token(&bpe, 5));
        assert!(!reply.token(&bpe, 7));
        reply.finish(&generation(vec![5, 7], FinishReason::Caller));
        let text = received(client);
        assert!(text.starts_with("HTTP/1.1 200 OK") && text.contains(r#""completion_tokens":1"#), "{text}");
    }

    #[test]
    fn a_refusal_before_anything_was_sent_is_an_http_error() {
        let (_, reply, client) = started(json!({"prompt": [1, 3], "stream": true}), true);
        reply.refuse("503 Service Unavailable", "the server is at capacity");
        let text = received(client);
        assert!(text.starts_with("HTTP/1.1 503") && text.contains("the server is at capacity"));
    }

    #[test]
    fn a_stream_that_fails_ends_with_an_error_event_and_no_done() {
        let (_, mut reply, client) = started(json!({"prompt": [1, 3], "stream": true}), true);
        reply.begin();
        reply.finish(&generation(Vec::new(), FinishReason::NoLogits));
        let text = received(client);
        assert!(text.contains(r#""finish_reason":"error""#) && !text.contains("[DONE]"), "{text}");
    }

    #[test]
    fn a_client_that_hung_up_is_noticed() {
        let (_, reply, client) = started(json!({"prompt": [1, 3]}), false);
        assert!(reply.connected());
        drop(client);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while reply.connected() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(!reply.connected());
    }
}
