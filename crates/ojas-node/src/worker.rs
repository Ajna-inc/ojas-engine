//! One engine worker: its process, its IPC connection, and the requests in flight.
//!
//! Generations are multiplexed by `ReqId`. Everything else (Load, TrainBegin,
//! TrainRound) carries no id, so those run one at a time through [`Worker::control`]
//! and the next control-type answer belongs to whoever holds the lock.
//!
//! A worker that dies takes its in-flight work with it: every generation channel is
//! closed, which its reader sees as an end without `Done`. That is what lets a
//! router fail over before the first token.

use crate::ipc::{self, Listener};
use anyhow::{anyhow, bail, Context, Result};
use ojas_swarm_proto::ipc::{NodeMsg, WorkerMsg};
use ojas_swarm_proto::{GenerateReq, ModelId, ModelInfo, ReqId, WorkerCaps, PROTO_VERSION};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, Notify};

type Out = (NodeMsg, Vec<u8>);

pub struct Spec {
    pub bin: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub device: String,
    pub socket: PathBuf,
}

pub struct Worker {
    pub device: String,
    /// Increments on every restart of this device's worker; training state lives in
    /// the process and is lost with it.
    pub epoch: u64,
    pub caps: WorkerCaps,
    pub models: Mutex<Vec<(PathBuf, ModelInfo)>>,
    pub busy: AtomicU32,
    out: mpsc::UnboundedSender<Out>,
    gens: Arc<Mutex<HashMap<ReqId, mpsc::UnboundedSender<WorkerMsg>>>>,
    ctl: tokio::sync::Mutex<mpsc::UnboundedReceiver<(WorkerMsg, Vec<u8>)>>,
    dead: Arc<AtomicBool>,
    /// Signalled once when the connection ends or the worker reports a fatal error.
    pub ended: Arc<Notify>,
}

impl Worker {
    /// Start the process and wait for it to connect and say `Ready`.
    pub async fn spawn(spec: &Spec, epoch: u64) -> Result<(Child, Arc<Worker>)> {
        let (listener, addr) = Listener::bind(&spec.socket).await?;
        let mut cmd = Command::new(&spec.bin);
        cmd.args(&spec.args).arg("--socket").arg(&addr).arg("--device").arg(&spec.device).kill_on_drop(true);
        cmd.envs(spec.env.iter().map(|(k, v)| (k, v)));
        let mut child = cmd.spawn().with_context(|| format!("starting worker {}", spec.bin.display()))?;
        let io = tokio::select! {
            c = tokio::time::timeout(Duration::from_secs(60), listener.accept()) => c.map_err(|_| anyhow!("worker did not connect within 60 s"))??,
            s = child.wait() => bail!("worker exited before connecting: {}", s?),
        };
        let (mut r, w) = tokio::io::split(io);
        let mut skipped = vec![];
        let caps = match tokio::time::timeout(Duration::from_secs(60), ipc::recv::<_, WorkerMsg>(&mut r, &mut skipped)).await {
            Ok(Ok(Some((WorkerMsg::Ready(c), _)))) => c,
            Ok(Ok(Some((m, _)))) => bail!("worker spoke before Ready: {m:?}"),
            Ok(Ok(None)) => bail!("worker closed before Ready"),
            Ok(Err(e)) => return Err(e),
            Err(_) => bail!("worker sent no Ready within 60 s"),
        };
        if caps.proto != PROTO_VERSION {
            bail!("worker speaks proto {}, this node {PROTO_VERSION}", caps.proto);
        }
        let (out, out_rx) = mpsc::unbounded_channel();
        let (ctl_tx, ctl_rx) = mpsc::unbounded_channel();
        let worker = Arc::new(Worker {
            device: spec.device.clone(),
            epoch,
            caps,
            models: Mutex::new(Vec::new()),
            busy: AtomicU32::new(0),
            out,
            gens: Arc::new(Mutex::new(HashMap::new())),
            ctl: tokio::sync::Mutex::new(ctl_rx),
            dead: Arc::new(AtomicBool::new(false)),
            ended: Arc::new(Notify::new()),
        });
        tokio::spawn(writer(w, out_rx));
        tokio::spawn(reader(r, skipped, worker.gens.clone(), ctl_tx, worker.dead.clone(), worker.ended.clone()));
        Ok((child, worker))
    }

    pub fn alive(&self) -> bool {
        !self.dead.load(Ordering::Relaxed)
    }

    pub fn holds(&self, m: &ModelId) -> bool {
        self.alive() && self.models.lock().unwrap().iter().any(|(_, i)| i.identity.model == *m)
    }

    pub fn send(&self, m: NodeMsg, payload: Vec<u8>) -> Result<()> {
        self.out.send((m, payload)).map_err(|_| anyhow!("worker on {} is gone", self.device))
    }

    /// One non-generation request and its answer. `None` timeout waits as long as
    /// the worker lives (a training round on a CPU can take a while).
    pub async fn control(&self, m: NodeMsg, payload: Vec<u8>, timeout: Option<Duration>) -> Result<(WorkerMsg, Vec<u8>)> {
        let mut ctl = self.ctl.lock().await;
        // Answers to an earlier caller that gave up must not be taken for ours.
        while ctl.try_recv().is_ok() {}
        self.send(m, payload)?;
        let next = async {
            loop {
                match ctl.recv().await {
                    Some((WorkerMsg::Progress { .. }, _)) => continue,
                    Some(x) => return Ok(x),
                    None => bail!("worker on {} died", self.device),
                }
            }
        };
        match timeout {
            Some(t) => tokio::time::timeout(t, next).await.map_err(|_| anyhow!("worker on {} did not answer within {t:?}", self.device))?,
            None => next.await,
        }
    }

    pub async fn load(&self, path: &std::path::Path, context: u32, expect: Option<ModelId>) -> Result<ModelInfo> {
        let m = NodeMsg::Load { path: path.to_string_lossy().into_owned(), context, expect };
        match self.control(m, vec![], Some(Duration::from_secs(1800))).await? {
            (WorkerMsg::Loaded(info), _) => {
                self.models.lock().unwrap().push((path.to_path_buf(), info.clone()));
                Ok(info)
            }
            (WorkerMsg::Error { message, .. }, _) => bail!("loading {}: {message}", path.display()),
            (other, _) => bail!("loading {}: unexpected {other:?}", path.display()),
        }
    }

    /// Start a generation. The returned [`Gen`] cancels it if dropped early.
    pub fn generate(self: &Arc<Self>, g: GenerateReq) -> Result<Gen> {
        let (tx, rx) = mpsc::unbounded_channel();
        let req = g.req;
        self.gens.lock().unwrap().insert(req, tx);
        if let Err(e) = self.send(NodeMsg::Generate(g), vec![]) {
            self.gens.lock().unwrap().remove(&req);
            return Err(e);
        }
        self.busy.fetch_add(1, Ordering::Relaxed);
        Ok(Gen { rx, worker: self.clone(), req, finished: false })
    }

    pub fn shutdown(&self) {
        let _ = self.send(NodeMsg::Shutdown, vec![]);
    }
}

pub struct Gen {
    rx: mpsc::UnboundedReceiver<WorkerMsg>,
    worker: Arc<Worker>,
    req: ReqId,
    finished: bool,
}

impl Gen {
    /// `Token`, `Done` or `Error`; `None` if the worker died mid-request.
    pub async fn next(&mut self) -> Option<WorkerMsg> {
        loop {
            let m = self.rx.recv().await?;
            match m {
                WorkerMsg::Progress { .. } => continue,
                WorkerMsg::Done { .. } | WorkerMsg::Error { .. } => self.finished = true,
                _ => {}
            }
            return Some(m);
        }
    }
}

impl Drop for Gen {
    fn drop(&mut self) {
        self.worker.busy.fetch_sub(1, Ordering::Relaxed);
        self.worker.gens.lock().unwrap().remove(&self.req);
        if !self.finished {
            let _ = self.worker.send(NodeMsg::Cancel { req: self.req }, vec![]);
        }
    }
}

async fn writer(mut w: WriteHalf<Box<dyn ipc::Io>>, mut rx: mpsc::UnboundedReceiver<Out>) {
    while let Some((m, p)) = rx.recv().await {
        if let Err(e) = ipc::send(&mut w, &m, &p).await {
            tracing::warn!("{e}");
            break;
        }
    }
    let _ = w.shutdown().await;
}

async fn reader(
    mut r: ReadHalf<Box<dyn ipc::Io>>,
    mut skipped: Vec<String>,
    gens: Arc<Mutex<HashMap<ReqId, mpsc::UnboundedSender<WorkerMsg>>>>,
    ctl: mpsc::UnboundedSender<(WorkerMsg, Vec<u8>)>,
    dead: Arc<AtomicBool>,
    ended: Arc<Notify>,
) {
    loop {
        let (m, p) = match ipc::recv::<_, WorkerMsg>(&mut r, &mut skipped).await {
            Ok(Some(x)) => x,
            Ok(None) => break,
            Err(e) => {
                tracing::warn!("{e}");
                break;
            }
        };
        let to_gen = |req: ReqId, m: WorkerMsg, last: bool| {
            let mut g = gens.lock().unwrap();
            let tx = if last { g.remove(&req) } else { g.get(&req).cloned() };
            if let Some(tx) = tx {
                let _ = tx.send(m);
            }
        };
        match m {
            WorkerMsg::Token { req, .. } | WorkerMsg::Progress { req: Some(req), .. } => to_gen(req, m, false),
            WorkerMsg::Done { req, .. } => to_gen(req, m, true),
            WorkerMsg::Error { req: Some(req), fatal, ref message } => {
                if fatal {
                    tracing::error!("worker fatal error: {message}");
                }
                to_gen(req, m, true);
                if fatal {
                    break;
                }
            }
            WorkerMsg::Error { fatal: true, ref message, .. } => {
                tracing::error!("worker fatal error: {message}");
                let _ = ctl.send((m, p));
                break;
            }
            WorkerMsg::Progress { req: None, phase, done, total } => tracing::debug!("worker {phase:?} {done}/{total}"),
            WorkerMsg::Bye => break,
            WorkerMsg::Ready(_) => {}
            other => {
                let _ = ctl.send((other, p));
            }
        }
    }
    dead.store(true, Ordering::Relaxed);
    gens.lock().unwrap().clear();
    ended.notify_one();
}
