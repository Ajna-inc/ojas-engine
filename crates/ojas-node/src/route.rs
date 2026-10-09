//! Where a generation runs: a local worker holding the model, else a member that
//! announces it, failing over to the next candidate until one produces its first
//! token. After the first token there is no failover: the client has already seen
//! output, and replaying it from another node could diverge.

use crate::app::App;
use anyhow::{anyhow, bail, Result};
use ojas_net::PeerId;
use ojas_swarm_proto::ipc::WorkerMsg;
use ojas_swarm_proto::peer::{PeerReq, PeerResp};
use ojas_swarm_proto::{Backend, Finish, GenerateReq, ModelId, Sampling};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Prefill of a long prompt on a CPU member is slow; this bounds it, not decode.
const FIRST_TOKEN: Duration = Duration::from_secs(300);
const NEXT_TOKEN: Duration = Duration::from_secs(120);

#[derive(Debug)]
pub enum Piece {
    Token(u32),
    Done { prompt_tokens: u32, completion_tokens: u32, finish: Finish },
}

pub struct Route {
    /// "local" or the serving member's PeerId.
    pub served_by: String,
    pub backend: Backend,
    pub rx: mpsc::Receiver<Result<Piece>>,
}

pub async fn generate(app: &Arc<App>, model: ModelId, prompt: Vec<u32>, max_tokens: u32, sampling: Sampling) -> Result<Route> {
    let g = GenerateReq { req: app.req_id(), model, prompt, max_tokens, sampling, want_text: false };
    let mut errs: Vec<String> = Vec::new();
    for w in app.local_for(&model) {
        match local(app, &w, g.clone()).await {
            Ok(r) => return Ok(r),
            Err(e) => errs.push(format!("local {}: {e:#}", w.device)),
        }
    }
    for p in app.remote_for(&model) {
        match remote(app, p, g.clone()).await {
            Ok(r) => return Ok(r),
            Err(e) => {
                tracing::warn!(peer = %p, "failing over: {e:#}");
                errs.push(format!("{p}: {e:#}"));
            }
        }
    }
    if errs.is_empty() {
        bail!("no local worker or member serves model {}", model.to_hex());
    }
    bail!("every candidate for model {} failed: {}", model.short(), errs.join("; "))
}

fn piece_of_worker(m: WorkerMsg) -> Result<Option<Piece>> {
    Ok(match m {
        WorkerMsg::Token { token, .. } => Some(Piece::Token(token)),
        WorkerMsg::Done { prompt_tokens, completion_tokens, finish, .. } => Some(Piece::Done { prompt_tokens, completion_tokens, finish }),
        WorkerMsg::Error { message, .. } => bail!("{message}"),
        _ => None,
    })
}

async fn local(app: &Arc<App>, w: &Arc<crate::worker::Worker>, g: GenerateReq) -> Result<Route> {
    let model = g.model;
    let mut gen = w.generate(g)?;
    let first = loop {
        let m = tokio::time::timeout(FIRST_TOKEN, gen.next()).await.map_err(|_| anyhow!("no first token within {FIRST_TOKEN:?}"))?;
        match piece_of_worker(m.ok_or_else(|| anyhow!("worker died before the first token"))?)? {
            Some(p) => break p,
            None => continue,
        }
    };
    let (tx, rx) = mpsc::channel(256);
    let app2 = app.clone();
    tokio::spawn(async move {
        let t0 = Instant::now();
        let mut next = Some(first);
        loop {
            let p = match next.take() {
                Some(p) => Ok(p),
                None => match gen.next().await {
                    None => Err(anyhow!("worker died mid-generation")),
                    Some(m) => match piece_of_worker(m) {
                        Ok(Some(p)) => Ok(p),
                        Ok(None) => continue,
                        Err(e) => Err(e),
                    },
                },
            };
            let end = !matches!(p, Ok(Piece::Token(_)));
            if let Ok(Piece::Done { completion_tokens, .. }) = &p {
                app2.note_rate(model, *completion_tokens, t0.elapsed());
            }
            // A dropped receiver is a client that went away: dropping `gen` cancels.
            if tx.send(p).await.is_err() || end {
                return;
            }
        }
    });
    Ok(Route { served_by: "local".into(), backend: w.caps.backend, rx })
}

fn piece_of_peer(m: PeerResp) -> Result<Option<(Piece, Option<Backend>)>> {
    Ok(match m {
        PeerResp::Token { token, .. } => Some((Piece::Token(token), None)),
        PeerResp::Done { prompt_tokens, completion_tokens, finish, backend, .. } => Some((Piece::Done { prompt_tokens, completion_tokens, finish }, Some(backend))),
        PeerResp::Error { message, .. } => bail!("{message}"),
        _ => None,
    })
}

async fn remote(app: &Arc<App>, peer: PeerId, g: GenerateReq) -> Result<Route> {
    let mut s = tokio::time::timeout(Duration::from_secs(15), app.node.open_stream(peer)).await.map_err(|_| anyhow!("stream open timed out"))??;
    s.send(&PeerReq::Generate(g), &[]).await?;
    let first = loop {
        let m = tokio::time::timeout(FIRST_TOKEN, s.recv::<PeerResp>()).await.map_err(|_| anyhow!("no first token within {FIRST_TOKEN:?}"))??;
        match piece_of_peer(m.ok_or_else(|| anyhow!("stream closed before the first token"))?.0)? {
            Some(p) => break p,
            None => continue,
        }
    };
    let backend = app.table.lock().unwrap().get(&peer).and_then(|s| s.announce.workers.first().map(|w| w.backend)).unwrap_or(Backend::Cpu);
    let (tx, rx) = mpsc::channel(256);
    tokio::spawn(async move {
        let mut next = Some(first.0);
        loop {
            let p = match next.take() {
                Some(p) => Ok(p),
                None => match tokio::time::timeout(NEXT_TOKEN, s.recv::<PeerResp>()).await {
                    Err(_) => Err(anyhow!("{peer} stalled")),
                    Ok(Err(e)) => Err(e),
                    Ok(Ok(None)) => Err(anyhow!("{peer} closed the stream mid-generation")),
                    Ok(Ok(Some((m, _)))) => match piece_of_peer(m) {
                        Ok(Some((p, _))) => Ok(p),
                        Ok(None) => continue,
                        Err(e) => Err(e),
                    },
                },
            };
            let end = !matches!(p, Ok(Piece::Token(_)));
            if tx.send(p).await.is_err() || end {
                return;
            }
        }
    });
    Ok(Route { served_by: peer.to_string(), backend, rx })
}
