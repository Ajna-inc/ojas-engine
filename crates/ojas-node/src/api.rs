//! The local API on 127.0.0.1: OpenAI-compatible completions, routed through the
//! swarm, behind a bearer token.

use crate::app::{App, ModelEntry};
use crate::route::{self, Piece};
use crate::text::{TextStream, Tok};
use anyhow::Result;
use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event as Sse, KeepAlive};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use ojas_swarm_proto::{Finish, ModelId, Sampling};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;

pub async fn serve(app: Arc<App>, port: u16, token: String) -> Result<()> {
    let token = Arc::new(token);
    let r = Router::new()
        .route("/status", get(status))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .layer(middleware::from_fn_with_state(token, auth))
        .with_state(app);
    let l = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    tracing::info!("api on http://{}", l.local_addr()?);
    axum::serve(l, r).await?;
    Ok(())
}

async fn auth(State(token): State<Arc<String>>, req: Request, next: Next) -> Response {
    let ok = req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).is_some_and(|t| ct_eq(t.as_bytes(), token.as_bytes()));
    if !ok {
        return err(StatusCode::UNAUTHORIZED, "missing or wrong bearer token");
    }
    next.run(req).await
}

/// Comparison time independent of where the strings differ.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// A refused request, boxed: a `Response` is large to carry in every `Result`.
type Fail = Box<Response>;

fn fail(code: StatusCode, msg: &str) -> Fail {
    Box::new(err(code, msg))
}

fn err(code: StatusCode, msg: &str) -> Response {
    (code, Json(json!({"error": {"message": msg, "type": code.canonical_reason().unwrap_or("error")}}))).into_response()
}

async fn status(State(app): State<Arc<App>>) -> Json<Value> {
    let workers: Vec<Value> = app
        .slots
        .iter()
        .map(|s| {
            let w = s.cur.read().unwrap().clone();
            json!({
                "device": s.device,
                "restarts": s.restarts.load(std::sync::atomic::Ordering::Relaxed),
                "alive": w.as_ref().is_some_and(|w| w.alive()),
                "backend": w.as_ref().map(|w| w.caps.backend.as_str()),
                "busy": w.as_ref().map(|w| w.busy.load(std::sync::atomic::Ordering::Relaxed)),
                "models": w.as_ref().map(|w| w.models.lock().unwrap().iter().map(|(p, i)| json!({"path": p, "id": i.identity.model, "arch": i.arch})).collect::<Vec<_>>()),
            })
        })
        .collect();
    let table: Vec<Value> = app
        .table
        .lock()
        .unwrap()
        .iter()
        .map(|(p, s)| json!({"peer": p.to_string(), "age_ms": s.at.elapsed().as_millis() as u64, "announce": s.announce}))
        .collect();
    Json(json!({
        "peer_id": app.node.peer_id().to_string(),
        "pool": app.pool,
        "listen": app.node.listen_addrs().await.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
        "members": app.node.members().iter().map(|p| p.to_string()).collect::<Vec<_>>(),
        "peers": app.node.peers().await,
        "workers": workers,
        "table": table,
        "models": app.models.iter().map(|m| json!({"name": m.name, "id": m.id(), "path": m.path})).collect::<Vec<_>>(),
        "coordinator": app.coordinator.is_some(),
        "train": *app.train_status.lock().unwrap(),
    }))
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    let mut data: Vec<Value> = app.models.iter().filter_map(|m| m.id().map(|id| (m, id))).map(|(m, id)| json!({"id": m.name, "object": "model", "owned_by": "ojas", "ojas_model_id": id})).collect();
    // Models other members serve that this node has no tokenizer for: listed by id
    // so a client can see them, though only raw-token prompts can use them here.
    let known: Vec<ModelId> = app.models.iter().filter_map(|m| m.id()).collect();
    let mut seen = Vec::new();
    for s in app.table.lock().unwrap().values() {
        for m in &s.announce.models {
            if !known.contains(&m.model) && !seen.contains(&m.model) {
                seen.push(m.model);
                data.push(json!({"id": m.model, "object": "model", "owned_by": "swarm", "arch": m.arch}));
            }
        }
    }
    Json(json!({"object": "list", "data": data}))
}

struct Job {
    name: String,
    model: ModelId,
    tok: Arc<Tok>,
    prompt: Vec<u32>,
    max_tokens: u32,
    sampling: Sampling,
    stops: Vec<String>,
    stream: bool,
}

fn resolve(app: &App, body: &Value) -> Result<(Arc<ModelEntry>, ModelId, Arc<Tok>), Fail> {
    let want = body.get("model").and_then(Value::as_str).unwrap_or("");
    let entry = app
        .models
        .iter()
        .find(|m| m.name == want || m.id().is_some_and(|id| id.to_hex() == want))
        .or_else(|| if want.is_empty() && app.models.len() == 1 { app.models.first() } else { None })
        .cloned()
        .ok_or_else(|| fail(StatusCode::NOT_FOUND, &format!("no model {want:?} configured on this node (its tokenizer is needed here)")))?;
    let id = entry.id().ok_or_else(|| fail(StatusCode::SERVICE_UNAVAILABLE, &format!("model {} has no identity yet: it is loading, or give its `id` in the config", entry.name)))?;
    let tok = entry.tok().ok_or_else(|| fail(StatusCode::SERVICE_UNAVAILABLE, &format!("tokenizer for {} is not loaded", entry.name)))?;
    Ok((entry, id, tok))
}

fn job(app: &App, body: &Value, chat: bool) -> Result<Job, Fail> {
    let (entry, model, tok) = resolve(app, body)?;
    let bad = |m: &str| fail(StatusCode::BAD_REQUEST, m);
    let prompt = if chat {
        let msgs = body.get("messages").and_then(Value::as_array).ok_or_else(|| bad("\"messages\" must be an array"))?;
        let mut system = String::new();
        let mut turns = Vec::new();
        for m in msgs {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            let content = match m.get("content") {
                Some(Value::String(s)) => s.clone(),
                // Content parts: keep the text ones.
                Some(Value::Array(parts)) => parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join(""),
                _ => String::new(),
            };
            if role == "system" || role == "developer" {
                system = content;
            } else {
                turns.push((role.to_string(), content));
            }
        }
        tok.chat(&system, &turns)
    } else {
        match body.get("prompt") {
            Some(Value::String(s)) => tok.encode(s),
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| v.as_u64().filter(|&id| (id as usize) < tok.vocab).map(|id| id as u32))
                .collect::<Option<Vec<u32>>>()
                .ok_or_else(|| bad("\"prompt\" token ids must be in the vocabulary"))?,
            _ => return Err(bad("\"prompt\" must be a string or an array of token ids")),
        }
    };
    if prompt.is_empty() {
        return Err(bad("empty prompt"));
    }
    let f = |k: &str, d: f64| body.get(k).and_then(Value::as_f64).unwrap_or(d) as f32;
    let max_tokens = body.get("max_completion_tokens").or_else(|| body.get("max_tokens")).and_then(Value::as_u64).unwrap_or(512).clamp(1, 1 << 20) as u32;
    let sampling = Sampling {
        temperature: f("temperature", 0.8),
        top_p: f("top_p", 1.0),
        top_k: body.get("top_k").and_then(Value::as_u64).unwrap_or(0) as u32,
        repeat_penalty: f("repeat_penalty", 1.0),
        seed: body.get("seed").and_then(Value::as_u64).unwrap_or_else(rand::random),
    };
    let stops = match body.get("stop") {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        Some(_) => return Err(bad("\"stop\" must be a string or an array of strings")),
    };
    Ok(Job { name: entry.name.clone(), model, tok, prompt, max_tokens, sampling, stops, stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false) })
}

enum Step {
    Text(String),
    End { finish: &'static str, prompt_tokens: u32, completion_tokens: u32, error: Option<String> },
}

/// Detokenise a routed generation into text steps. Ends at the model's
/// end-of-generation token, a stop string, `max_tokens`, or the worker's `Done`.
async fn run(job: &Job, mut r: route::Route, tx: mpsc::Sender<Step>) {
    let mut text = TextStream::new(&job.stops);
    let mut n = 0u32;
    let prompt_tokens = job.prompt.len() as u32;
    let end = |finish, n, error| Step::End { finish, prompt_tokens, completion_tokens: n, error };
    let last = loop {
        match r.rx.recv().await {
            Some(Ok(Piece::Token(t))) => {
                n += 1;
                if job.tok.is_eog(t) {
                    break end("stop", n, None);
                }
                let (s, hit) = text.push(&job.tok.bpe, t);
                if !s.is_empty() && tx.send(Step::Text(s)).await.is_err() {
                    return;
                }
                if hit {
                    break end("stop", n, None);
                }
                if n >= job.max_tokens {
                    break end("length", n, None);
                }
            }
            Some(Ok(Piece::Done { prompt_tokens: pt, completion_tokens, finish })) => {
                let f = match finish {
                    Finish::Length => "length",
                    Finish::Fault | Finish::Refused => "error",
                    _ => "stop",
                };
                let error = (f == "error").then(|| format!("{finish:?}"));
                // The worker's count is the truth if it gave one (it may add BOS).
                break Step::End { finish: f, prompt_tokens: if pt > 0 { pt } else { prompt_tokens }, completion_tokens: completion_tokens.max(n), error };
            }
            Some(Err(e)) => break end("error", n, Some(format!("{e:#}"))),
            None => break end("error", n, Some("generation ended without Done".into())),
        }
    };
    // Dropping the route here cancels the generation wherever it runs.
    drop(r);
    let tail = text.finish();
    if !tail.is_empty() {
        let _ = tx.send(Step::Text(tail)).await;
    }
    let _ = tx.send(last).await;
}

async fn chat(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    respond(app, body, true).await
}

async fn completions(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    respond(app, body, false).await
}

async fn respond(app: Arc<App>, body: Value, chat: bool) -> Response {
    let job = match job(&app, &body, chat) {
        Ok(j) => j,
        Err(r) => return *r,
    };
    let r = match route::generate(&app, job.model, job.prompt.clone(), job.max_tokens, job.sampling.clone()).await {
        Ok(r) => r,
        Err(e) => return err(StatusCode::BAD_GATEWAY, &format!("{e:#}")),
    };
    let served = json!({"served_by": r.served_by, "backend": r.backend.as_str()});
    let id = format!("{}-{}", if chat { "chatcmpl" } else { "cmpl" }, app.req_id().0);
    let created = ojas_net::unix_now();
    let (tx, mut rx) = mpsc::channel(64);
    let stream = job.stream;
    let name = job.name.clone();
    tokio::spawn(async move { run(&job, r, tx).await });

    if !stream {
        let mut content = String::new();
        let mut fin = ("error", 0, 0, Some("no output".to_string()));
        while let Some(s) = rx.recv().await {
            match s {
                Step::Text(t) => content.push_str(&t),
                Step::End { finish, prompt_tokens, completion_tokens, error } => fin = (finish, prompt_tokens, completion_tokens, error),
            }
        }
        if let (true, Some(e)) = (content.is_empty(), &fin.3) {
            return err(StatusCode::BAD_GATEWAY, e);
        }
        let choice = if chat {
            json!({"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": fin.0})
        } else {
            json!({"index": 0, "text": content, "finish_reason": fin.0})
        };
        return Json(json!({
            "id": id, "object": if chat { "chat.completion" } else { "text_completion" }, "created": created, "model": name,
            "choices": [choice],
            "usage": {"prompt_tokens": fin.1, "completion_tokens": fin.2, "total_tokens": fin.1 + fin.2},
            "ojas": served,
        }))
        .into_response();
    }

    let object = if chat { "chat.completion.chunk" } else { "text_completion" };
    let chunk = move |delta: Value, finish: Value| -> Sse {
        let choice = if chat { json!({"index": 0, "delta": delta, "finish_reason": finish}) } else { json!({"index": 0, "text": delta, "finish_reason": finish}) };
        Sse::default().data(json!({"id": id, "object": object, "created": created, "model": name, "choices": [choice], "ojas": served}).to_string())
    };
    let (etx, erx) = mpsc::channel::<Sse>(64);
    tokio::spawn(async move {
        if chat && etx.send(chunk(json!({"role": "assistant"}), Value::Null)).await.is_err() {
            return;
        }
        while let Some(s) = rx.recv().await {
            let ev = match s {
                Step::Text(t) => chunk(if chat { json!({"content": t}) } else { json!(t) }, Value::Null),
                Step::End { finish, error, .. } => {
                    let mut c = chunk(if chat { json!({}) } else { json!("") }, json!(finish));
                    if let Some(e) = error {
                        c = Sse::default().data(json!({"error": {"message": e}}).to_string());
                        let _ = etx.send(c).await;
                        c = chunk(if chat { json!({}) } else { json!("") }, json!(finish));
                    }
                    c
                }
            };
            if etx.send(ev).await.is_err() {
                return;
            }
        }
        let _ = etx.send(Sse::default().data("[DONE]")).await;
    });
    let s = tokio_stream::wrappers::ReceiverStream::new(erx).map(Ok::<_, Infallible>);
    axum::response::Sse::new(s).keep_alive(KeepAlive::default()).into_response()
}
