//! One completion request: what to generate (`Job`) and how to answer it (`Reply`).
//!
//! The split lets one request run either way: alone through `EngineCore`, with the
//! reply fed from its token callback, or in a batch of slots, with the job submitted
//! to the batch and the reply kept until the batch reports the request done. Either
//! way the client sees the same stream, the same stop handling and the same usage.

use super::{begin_sse, now, output_format, prompt_docs, prompt_ids, prompt_marks, restore_json, sampling_from,
            send_err, send_json, sse, stop_strings};
use crate::backend::ModelInfo;
use crate::detok::TextStream;
use crate::flags::RunOpts;
use ojas_grammar::TokenVocab;
use ojas_infer::{batch, FinishReason, Generation, LogitProcessor, SampleOpts};
use ojas_tokenize::Bpe;
use serde_json::{json, Value};
use std::io::Write;
use std::net::TcpStream;
use std::sync::Arc;

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
    pub(crate) constraint: Option<Box<dyn LogitProcessor>>,
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
            prompt: self.ids, max_tokens: self.want, opts: self.sampling, processor: self.constraint, stop,
            banned: self.banned, marks: self.marks, docs: self.docs, reuse: self.reuse,
        }
    }
}

/// How a request is answered: the connection, the response format, and the text so
/// far.
pub(crate) struct Reply {
    stream: TcpStream,
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
/// malformed or the model cannot serve it.
pub(crate) fn start(mut stream: TcpStream, ctx: &Ctx, body: &Value, chat: bool, oai: bool) -> Option<(Job, Reply)> {
    // Refuse a poisoned session before the response shape is chosen, so the client
    // gets an HTTP error rather than a stream it cannot trust.
    if let Some(err) = ojas_core::device_fault::peek() {
        send_err(&mut stream, "503 Service Unavailable",
            &format!("device fault; this model session is no longer usable and must be reloaded: {err}"));
        return None;
    }
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
        constraint: constraint.map(|p| Box::new(p) as Box<dyn LogitProcessor>),
        reuse: body.get("cache_prompt").and_then(Value::as_bool).unwrap_or(ctx.prefix_cache),
        eog: if ignore_eos { Vec::new() } else { info.eog.clone() },
        eos: if ignore_eos { None } else { ctx.primary },
        banned: if ignore_eos { info.eog.clone() } else { Vec::new() },
        ids,
    };
    let mut reply = Reply {
        streaming: body.get("stream").and_then(Value::as_bool).unwrap_or(false), oai, chat,
        return_tokens: body.get("return_tokens").and_then(Value::as_bool).unwrap_or(false),
        id: format!("cmpl-{:x}", now()), created: now(), model_name: info.arch.clone(),
        prompt_tokens: job.ids.len(), end_of_turn: if ignore_eos { None } else { ctx.secondary },
        text_stream: TextStream::new(&stops), text: String::new(), out_ids: Vec::with_capacity(want), alive: true,
        stream,
    };
    if reply.streaming {
        begin_sse(&mut reply.stream);
        // The first OpenAI chunk carries the role and no content.
        if oai && chat {
            let first = json!({
                "id": reply.id, "object": "chat.completion.chunk", "created": reply.created, "model": reply.model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            });
            reply.alive = sse(&mut reply.stream, &first);
        }
    }
    Some((job, reply))
}

fn refuse(mut stream: TcpStream, msg: &str) -> Option<(Job, Reply)> {
    send_err(&mut stream, "400 Bad Request", msg);
    None
}

impl Reply {
    /// Take one generated token; false ends the request (end of turn, a stop
    /// string, or a client that went away).
    pub(crate) fn token(&mut self, bpe: &Bpe, t: u32) -> bool {
        if Some(t) == self.end_of_turn { return false; }
        let (piece, stopped) = self.text_stream.push(bpe, t);
        self.out_ids.push(t);
        self.text.push_str(&piece);
        if !self.streaming { return !stopped; }
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
        self.alive = sse(&mut self.stream, &ev);
        // A disconnected client must stop the decode, not keep it generating into a
        // closed socket.
        self.alive && !stopped
    }

    /// Answer the request now that generation has ended.
    pub(crate) fn finish(mut self, gen: &Generation) {
        // A model that does not expose logits cannot be constrained; it stops before
        // emitting anything rather than produce unconstrained output.
        if gen.finish == FinishReason::NoLogits {
            if !self.streaming {
                send_err(&mut self.stream, "501 Not Implemented",
                    "this model's backend does not expose logits, which constrained output needs");
            }
            return;
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
        let fault = ojas_core::device_fault::peek();

        if self.streaming {
            if !self.alive { return; }
            // A fault raised mid-stream means the tokens after it came out of
            // undefined buffers. The 200 is already promised, so close with a terminal
            // error event: never `finish_reason: stop` and never `[DONE]`, which both
            // report success.
            if let Some(err) = fault {
                let ev = if !self.oai {
                    json!({"error": {"message": err.to_string(), "type": "device_error"}, "stop": true, "truncated": true})
                } else {
                    json!({"error": {"message": err.to_string(), "type": "device_error"},
                           "id": id, "object": if self.chat { "chat.completion.chunk" } else { "text_completion" },
                           "created": created, "model": model,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]})
                };
                sse(&mut self.stream, &ev);
                return;
            }
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
            sse(&mut self.stream, &done);
            if self.oai {
                let _ = self.stream.write_all(b"data: [DONE]\n\n");
                let _ = self.stream.flush();
            }
            return;
        }

        if let Some(err) = fault {
            return send_err(&mut self.stream, "500 Internal Server Error",
                &format!("device fault during generation; no output is returned because the buffers it was \
                          read from are undefined: {err}"));
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
        send_json(&mut self.stream, "200 OK", &out);
    }
}
