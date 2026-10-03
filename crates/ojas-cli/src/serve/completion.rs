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

use super::{error_response, json_response, now, output_format, prompt_docs, prompt_ids, prompt_marks, restore_json,
            sampling_from, send_err, sse_event, stop_strings, SSE_HEAD};
use crate::backend::ModelInfo;
use crate::detok::TextStream;
use crate::flags::RunOpts;
use ojas_grammar::TokenVocab;
use ojas_infer::{batch, FinishReason, Generation, LogitProcessor, SampleOpts};
use ojas_tokenize::Bpe;
use serde_json::{json, Value};
use std::io::Write;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

/// Writes a reply may queue ahead of its client; one that falls this far behind is
/// treated as gone.
const OUTBOX: usize = 4096;

/// Tokens between checks that a client still waiting for a whole response is there.
const ALIVE_EVERY: usize = 32;

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
}

/// What a request asks the model to generate.
pub(crate) struct Job {
    pub(crate) ids: Vec<u32>,
    pub(crate) marks: Vec<usize>,
    pub(crate) docs: Vec<(usize, usize)>,
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
            banned: self.banned, marks: self.marks, docs: self.docs, reuse: self.reuse,
        }
    }
}

/// A connection's writes, made on their own thread.
struct Outbox {
    tx: mpsc::SyncSender<Vec<u8>>,
    gone: Arc<AtomicBool>,
    /// The connection itself, kept to notice a client that hung up.
    stream: TcpStream,
}

impl Outbox {
    fn new(stream: &TcpStream) -> std::io::Result<Self> {
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
        Ok(Outbox { tx, gone, stream })
    }

    /// Queue bytes for the client; false when it is gone or too far behind.
    fn send(&self, bytes: Vec<u8>) -> bool {
        if self.gone.load(Ordering::Relaxed) || self.tx.try_send(bytes).is_err() {
            self.gone.store(true, Ordering::Relaxed);
            return false;
        }
        true
    }

    /// Whether the client is still connected, without waiting.
    fn connected(&self) -> bool {
        !self.gone.load(Ordering::Relaxed) && !peer_closed(&self.stream)
    }
}

/// Whether the peer closed the connection: a non-blocking peek reads end of stream.
#[cfg(unix)]
fn peer_closed(stream: &TcpStream) -> bool {
    use std::os::fd::AsRawFd;
    let mut byte = 0u8;
    let n = unsafe {
        libc::recv(stream.as_raw_fd(), (&mut byte as *mut u8).cast(), 1, libc::MSG_PEEK | libc::MSG_DONTWAIT)
    };
    n == 0
}

#[cfg(not(unix))]
fn peer_closed(_stream: &TcpStream) -> bool { false }

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
    alive: bool,
}

/// Parse a completion request, answering it at once with an error when it is
/// malformed. Nothing is sent on success until `Reply::begin`.
pub(crate) fn start(stream: TcpStream, ctx: &Ctx, body: &Value, chat: bool, oai: bool) -> Option<(Job, Reply)> {
    let (bpe, info, opts) = (ctx.bpe, ctx.info, ctx.opts);
    let ids = match prompt_ids(body, bpe, info, opts) {
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
    let want = body.get("n_predict").or_else(|| body.get("max_tokens")).and_then(Value::as_u64)
        .map(|v| v as usize).unwrap_or(opts.n_predict).min(info.context - ids.len());
    // `ignore_eos` makes a fixed-length benchmark possible: both engines must emit
    // exactly n_predict tokens, or the rates describe different amounts of work. It
    // suppresses the end-of-generation set from selection rather than merely
    // declining to stop: a model that emits its terminator and carries on produces
    // different text from that point, so a benchmark whose outputs diverge is not
    // like-for-like.
    let ignore_eos = body.get("ignore_eos").and_then(Value::as_bool).unwrap_or(false);
    let job = Job {
        marks: if ctx.prefix_cache { prompt_marks(body, bpe, info, opts) } else { Vec::new() },
        docs: if ctx.prefix_cache { prompt_docs(body, bpe, info, opts) } else { Vec::new() },
        want,
        sampling: (sampling.temperature > 0.0).then_some(sampling),
        constraint: constraint.map(|p| Box::new(p) as Box<dyn LogitProcessor + Send>),
        reuse: body.get("cache_prompt").and_then(Value::as_bool).unwrap_or(ctx.prefix_cache),
        eog: if ignore_eos { Vec::new() } else { info.eog.clone() },
        eos: if ignore_eos { None } else { ctx.primary },
        banned: if ignore_eos { info.eog.clone() } else { Vec::new() },
        ids,
    };
    let mut stream = stream;
    let out = match Outbox::new(&stream) {
        Ok(out) => out,
        Err(e) => {
            send_err(&mut stream, "500 Internal Server Error", &format!("cannot answer this connection: {e}"));
            return None;
        }
    };
    let reply = Reply {
        streaming: body.get("stream").and_then(Value::as_bool).unwrap_or(false), oai, chat,
        return_tokens: body.get("return_tokens").and_then(Value::as_bool).unwrap_or(false),
        id: format!("cmpl-{:x}", now()), created: now(), model_name: info.arch.clone(),
        prompt_tokens: job.ids.len(), end_of_turn: if ignore_eos { None } else { ctx.secondary },
        text_stream: TextStream::new(&stops), text: String::new(), out_ids: Vec::with_capacity(want), alive: true,
        out,
    };
    Some((job, reply))
}

fn refuse(mut stream: TcpStream, msg: &str) -> Option<(Job, Reply)> {
    send_err(&mut stream, "400 Bad Request", msg);
    None
}

impl Reply {
    /// Commit to answering: a stream's header and first chunk go out now, a whole
    /// response only at `finish`.
    pub(crate) fn begin(&mut self) {
        if !self.streaming { return; }
        self.alive = self.out.send(SSE_HEAD.as_bytes().to_vec());
        // The first OpenAI chunk carries the role and no content.
        if self.oai && self.chat {
            let first = json!({
                "id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            });
            self.alive &= self.out.send(sse_event(&first));
        }
    }

    /// Answer with an error instead, before `begin`.
    pub(crate) fn refuse(self, status: &str, msg: &str) { self.out.send(error_response(status, msg)); }

    /// Whether the client is still waiting.
    pub(crate) fn connected(&self) -> bool { self.alive && self.out.connected() }

    /// Take one generated token; false ends the request (end of turn, a stop
    /// string, or a client that went away).
    pub(crate) fn token(&mut self, bpe: &Bpe, t: u32) -> bool {
        if Some(t) == self.end_of_turn { return false; }
        let (piece, stopped) = self.text_stream.push(bpe, t);
        self.out_ids.push(t);
        self.text.push_str(&piece);
        if !self.streaming {
            if self.out_ids.len().is_multiple_of(ALIVE_EVERY) { self.alive = self.out.connected(); }
            return self.alive && !stopped;
        }
        // Emit an event for every token, including one whose decoded piece is empty
        // because the detokenizer is still holding a multi-byte character. Skipping
        // those hides tokens from the client: counts come out short and the first
        // and last arrival timestamps used for throughput span the wrong window.
        let ev = if !self.oai {
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
        let tail = self.text_stream.finish();
        self.text.push_str(&tail);
        let n_out = self.out_ids.len();
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
        let out = if !self.oai {
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
        self.out.send(json_response("200 OK", &out));
    }

    /// End the request with an error: an HTTP error before any of a stream was
    /// sent, else a terminal error event, never `finish_reason: stop` or `[DONE]`,
    /// which both report success.
    fn fail(self, status: &str, kind: &str, msg: &str) {
        if !self.streaming { return self.refuse(status, msg); }
        if !self.alive { return; }
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
    use std::net::TcpListener;

    fn info() -> ModelInfo {
        ModelInfo {
            arch: "qwen3".into(), eos: Some(2), eog: vec![2], vocab: 16, context: 64, backend: "cpu",
            has_mtp: false, n_layers: 1, hidden_dim: 1, load_secs: 0.0,
        }
    }

    /// The server's end of a loopback connection, and the client's.
    fn connection() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        (listener.accept().unwrap().0, client)
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
        let ctx = Ctx { bpe: &bpe, info: &info, opts: &opts, vocab: &vocab, primary: Some(2), secondary: Some(7),
                        prefix_cache: false };
        let (server, client) = connection();
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
