//! The daemon's state and its long-running tasks: worker supervision, the
//! announce loop, the peer table, and the answers to members' requests.

use crate::config::Config;
use crate::text::Tok;
use crate::worker::{Spec, Worker};
use anyhow::{anyhow, Result};
use ojas_net::blob::BlobStore;
use ojas_net::{Event, Node, PeerId, PeerStream};
use ojas_swarm_proto::ipc::WorkerMsg;
use ojas_swarm_proto::peer::{Announce, PeerReq, PeerResp, ServedModel, WorkerSummary};
use ojas_swarm_proto::{features, GenerateReq, ModelId, ReqId, PROTO_VERSION};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Notify};

pub struct ModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub load: bool,
    pub context: u32,
    /// From the config, or learned when a local worker loads the file.
    pub id: RwLock<Option<ModelId>>,
    pub tok: RwLock<Option<Arc<Tok>>>,
}

impl ModelEntry {
    pub fn id(&self) -> Option<ModelId> {
        *self.id.read().unwrap()
    }
    pub fn tok(&self) -> Option<Arc<Tok>> {
        self.tok.read().unwrap().clone()
    }
}

pub struct Slot {
    pub device: String,
    pub cur: RwLock<Option<Arc<Worker>>>,
    pub restarts: AtomicU32,
}

pub struct Seen {
    pub announce: Announce,
    pub at: Instant,
}

/// The coordinator a `--coordinator` node feeds training and blob requests to.
pub trait Coordinate: Send {
    fn handle(&mut self, from: &str, req: PeerReq, payload: Vec<u8>) -> (PeerResp, Vec<u8>);
}

pub struct App {
    pub node: Node,
    pub cfg: Config,
    pub pool: String,
    pub models: Vec<Arc<ModelEntry>>,
    pub slots: Vec<Arc<Slot>>,
    pub table: Mutex<HashMap<PeerId, Seen>>,
    pub tok_s: Mutex<HashMap<ModelId, f32>>,
    pub next_req: AtomicU64,
    pub blobs: Mutex<BlobStore>,
    pub coordinator: Option<Arc<Mutex<Box<dyn Coordinate>>>>,
    pub train_status: Mutex<serde_json::Value>,
    pub announce_now: Notify,
    pub announce_every: Duration,
    pub worker_up: Notify,
}

impl App {
    pub fn req_id(&self) -> ReqId {
        ReqId(self.next_req.fetch_add(1, Ordering::Relaxed))
    }

    pub fn live_workers(&self) -> Vec<Arc<Worker>> {
        self.slots.iter().filter_map(|s| s.cur.read().unwrap().clone()).filter(|w| w.alive()).collect()
    }

    /// Local workers holding `m`, least busy first.
    pub fn local_for(&self, m: &ModelId) -> Vec<Arc<Worker>> {
        let mut v: Vec<_> = self.live_workers().into_iter().filter(|w| w.holds(m)).collect();
        v.sort_by_key(|w| w.busy.load(Ordering::Relaxed));
        v
    }

    /// Members announcing `m`: least busy, then fastest measured decode, then the
    /// PeerId so the order is stable.
    pub fn remote_for(&self, m: &ModelId) -> Vec<PeerId> {
        let stale = self.announce_every * 4;
        let t = self.table.lock().unwrap();
        let mut v: Vec<(u32, f32, PeerId)> = t
            .iter()
            .filter(|(p, s)| s.at.elapsed() < stale && self.node.is_member(p))
            .filter_map(|(p, s)| {
                let sm = s.announce.models.iter().find(|x| x.model == *m)?;
                Some((s.announce.busy, sm.tok_s.unwrap_or(-1.0), *p))
            })
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)).then(a.2.cmp(&b.2)));
        v.into_iter().map(|x| x.2).collect()
    }

    pub fn note_rate(&self, m: ModelId, tokens: u32, decode: Duration) {
        if tokens < 2 || decode.is_zero() {
            return;
        }
        let r = (tokens - 1) as f32 / decode.as_secs_f32();
        let mut t = self.tok_s.lock().unwrap();
        let e = t.entry(m).or_insert(r);
        *e = 0.7 * *e + 0.3 * r;
        drop(t);
        self.announce_now.notify_one();
    }

    pub fn announce(&self) -> Announce {
        let workers = self.live_workers();
        let tok_s = self.tok_s.lock().unwrap().clone();
        let mut models: Vec<ServedModel> = Vec::new();
        for w in &workers {
            for (_, i) in w.models.lock().unwrap().iter() {
                if !models.iter().any(|m| m.model == i.identity.model) {
                    models.push(ServedModel {
                        model: i.identity.model,
                        arch_id: i.identity.arch,
                        arch: i.arch.clone(),
                        backend: i.backend,
                        context: i.context,
                        tok_s: tok_s.get(&i.identity.model).copied(),
                        name: self.models.iter().find(|e| e.id() == Some(i.identity.model)).map(|e| e.name.clone()),
                    });
                }
            }
        }
        let feats = workers.iter().fold(0, |f, w| f | w.caps.features) | if models.is_empty() { 0 } else { features::GENERATE };
        Announce {
            proto: PROTO_VERSION,
            features: feats,
            engine_version: env!("CARGO_PKG_VERSION").into(),
            workers: workers
                .iter()
                .map(|w| WorkerSummary { backend: w.caps.backend, device: w.caps.device.clone(), memory_bytes: w.caps.memory_bytes, slots: w.caps.slots })
                .collect(),
            models,
            busy: workers.iter().map(|w| w.busy.load(Ordering::Relaxed)).sum(),
            unix_ms: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0),
            // TODO: the node's reserved relay circuits, once members behind NAT are tested.
            relays: Vec::new(),
        }
    }
}

pub async fn announce_loop(app: Arc<App>) {
    loop {
        let _ = app.node.publish(&app.announce());
        let _ = tokio::time::timeout(app.announce_every, app.announce_now.notified()).await;
        // Coalesce bursts (several models loading at once) into one message.
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Keep one worker running on a device: start it, load the configured models,
/// and restart it whenever it exits or reports a fatal error.
pub async fn supervise(app: Arc<App>, slot: Arc<Slot>, spec: Spec) {
    let mut backoff = Duration::from_millis(500);
    for epoch in 0u64.. {
        let started = Instant::now();
        match Worker::spawn(&spec, epoch).await {
            Ok((mut child, w)) => {
                tracing::info!(device = %slot.device, backend = w.caps.backend.as_str(), "worker ready: {}", w.caps.device);
                for m in app.models.iter().filter(|m| m.load) {
                    match w.load(&m.path, m.context, m.id()).await {
                        Ok(info) => {
                            tracing::info!("{} loaded as {}", m.name, info.identity.model.short());
                            *m.id.write().unwrap() = Some(info.identity.model);
                        }
                        Err(e) => tracing::error!("{e:#}"),
                    }
                }
                *slot.cur.write().unwrap() = Some(w.clone());
                app.worker_up.notify_waiters();
                app.announce_now.notify_one();
                tokio::select! {
                    s = child.wait() => tracing::warn!(device = %slot.device, "worker exited: {:?}", s),
                    _ = w.ended.notified() => {
                        tracing::warn!(device = %slot.device, "worker connection ended; restarting it");
                        let _ = child.start_kill();
                        let _ = child.wait().await;
                    }
                }
                *slot.cur.write().unwrap() = None;
                app.announce_now.notify_one();
            }
            Err(e) => tracing::error!(device = %slot.device, "worker failed to start: {e:#}"),
        }
        slot.restarts.fetch_add(1, Ordering::Relaxed);
        if started.elapsed() > Duration::from_secs(60) {
            backoff = Duration::from_millis(500);
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}

/// Wait for a live worker on any device.
pub async fn any_worker(app: &App) -> Arc<Worker> {
    loop {
        let n = app.worker_up.notified();
        if let Some(w) = app.live_workers().into_iter().next() {
            return w;
        }
        let _ = tokio::time::timeout(Duration::from_secs(1), n).await;
    }
}

pub async fn events(app: Arc<App>, mut rx: mpsc::Receiver<Event>) {
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Rpc { peer, req, payload, reply } => {
                let app = app.clone();
                tokio::spawn(async move {
                    let (r, p) = answer(&app, peer, req, payload).await;
                    reply.send(r, p);
                });
            }
            Event::Stream(s) => {
                tokio::spawn(serve_stream(app.clone(), s));
            }
            Event::Announce { peer, announce } => {
                app.table.lock().unwrap().insert(peer, Seen { announce, at: Instant::now() });
            }
            Event::MemberUp(_) => app.announce_now.notify_one(),
            Event::MemberDown(p) | Event::Rejected { peer: p, .. } => {
                app.table.lock().unwrap().remove(&p);
            }
        }
    }
}

async fn answer(app: &Arc<App>, peer: PeerId, req: PeerReq, payload: Vec<u8>) -> (PeerResp, Vec<u8>) {
    if let Some(c) = &app.coordinator {
        if !matches!(req, PeerReq::Generate(_) | PeerReq::Score(_)) {
            let c = c.clone();
            let from = peer.to_string();
            // The outer step can be heavy; keep it off the async workers.
            return tokio::task::spawn_blocking(move || c.lock().unwrap().handle(&from, req, payload))
                .await
                .unwrap_or_else(|e| (PeerResp::Error { message: format!("coordinator panicked: {e}"), retry: false }, vec![]));
        }
    }
    if let Some(r) = app.blobs.lock().unwrap().handle(&req, &payload) {
        return r;
    }
    let msg = match req {
        PeerReq::Generate(_) => format!("Generate goes on {}", ojas_swarm_proto::peer::PROTO_STREAM),
        PeerReq::Score(_) => "scoring is not served by this node".into(),
        _ => "this node does not coordinate a training run".into(),
    };
    (PeerResp::Error { message: msg, retry: false }, vec![])
}

/// A member's generation, served by a local worker.
async fn serve_stream(app: Arc<App>, mut s: PeerStream) {
    let peer = s.peer;
    let g = match tokio::time::timeout(Duration::from_secs(30), s.recv::<PeerReq>()).await {
        Ok(Ok(Some((PeerReq::Generate(g), _)))) => g,
        Ok(Ok(Some((other, _)))) => {
            let _ = s.send(&PeerResp::Error { message: format!("expected Generate, got {other:?}"), retry: false }, &[]).await;
            return;
        }
        _ => return,
    };
    let their = g.req;
    let r = run_local(&app, g, |m| {
        let resp = match m {
            WorkerMsg::Token { token, text, .. } => PeerResp::Token { req: their, token, text },
            WorkerMsg::Done { prompt_tokens, completion_tokens, finish, .. } => {
                PeerResp::Done { req: their, prompt_tokens, completion_tokens, finish, backend: ojas_swarm_proto::Backend::Cpu }
            }
            WorkerMsg::Error { message, .. } => PeerResp::Error { message, retry: false },
            _ => return None,
        };
        Some(resp)
    }, &mut s)
    .await;
    if let Err(e) = r {
        tracing::debug!(%peer, "remote generation ended: {e:#}");
        let _ = s.send(&PeerResp::Error { message: e.to_string(), retry: true }, &[]).await;
    }
    s.close().await;
}

/// Run `g` on the least busy local worker holding its model, writing each frame
/// `map` produces to `s`. Errors before anything was written are retryable.
async fn run_local(app: &Arc<App>, mut g: GenerateReq, map: impl Fn(WorkerMsg) -> Option<PeerResp>, s: &mut PeerStream) -> Result<()> {
    let w = app.local_for(&g.model).into_iter().next().ok_or_else(|| anyhow!("model {} is not loaded here", g.model.short()))?;
    let backend = w.caps.backend;
    let model = g.model;
    g.req = app.req_id();
    let mut gen = w.generate(g)?;
    let mut first: Option<Instant> = None;
    loop {
        let m = gen.next().await.ok_or_else(|| anyhow!("worker on {} died mid-request", w.device))?;
        let (tokens, end) = match &m {
            WorkerMsg::Token { .. } => {
                first.get_or_insert_with(Instant::now);
                (0, false)
            }
            WorkerMsg::Done { completion_tokens, .. } => (*completion_tokens, true),
            WorkerMsg::Error { .. } => (0, true),
            _ => (0, false),
        };
        // Measured before the write: a requester that stops reading at its own
        // max_tokens must not cost this node its decode rate.
        if end {
            if let Some(f) = first {
                app.note_rate(model, tokens, f.elapsed());
            }
        }
        if let Some(mut out) = map(m) {
            if let PeerResp::Done { backend: b, .. } = &mut out {
                *b = backend;
            }
            s.send(&out, &[]).await?;
        }
        if end {
            return Ok(());
        }
    }
}
