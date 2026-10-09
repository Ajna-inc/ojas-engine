//! Coordinator side of a DiLoCo run. Transport agnostic: the node passes each training
//! or blob request it receives to [`Coordinator::handle`] and sends back what it returns.
//!
//! One run, fixed at start by a [`RunConfig`]. Members pull: join, sync, fetch θ (and the
//! data) as blobs, train, put their delta as a blob, push. A round closes when every
//! bound member has pushed, or its deadline has passed — in both cases only with at
//! least `min_members` deltas. Then the deltas are averaged (weighted by tokens trained,
//! members in id order, f64 accumulation), the outer Nesterov step is applied, and the
//! new θ, its momentum, the round and the member table are checkpointed atomically.
//!
//! Nothing a peer sends reaches θ unchecked: a delta must come from the peer bound to
//! its member slot, for the current round, from the current base (`ModelId`, i.e. the
//! exact bytes of θ) and the run's schema (`ArchId`; matching length is not enough),
//! with a verified blob that decodes to exactly `n_params` finite values and a report
//! whose shard ranges are the ones this member was assigned.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use ojas_learn::cpu::Cpu;
use ojas_learn::models::tiny_gpt::{self, TinyGpt};
use ojas_swarm_proto::peer::{PeerReq, PeerResp, RoundStatus, TrainState, BLOB_CHUNK};
use ojas_swarm_proto::{ArchId, Backend, BlobId, DType, DataRef, DeltaReport, ModelId, RoundSpec, Tensor, TinyGptConfig, TrainModel, TrainSpec};
use serde::{Deserialize, Serialize};

use crate::train::data;

/// A run's configuration, read once at start (JSON). Relative paths resolve against
/// the config file's directory.
///
/// ```json
/// {
///   "run": "bytes-demo",
///   "model": { "kind": "tiny_gpt", "vocab": 256, "ctx": 128, "d_model": 128,
///              "n_layers": 4, "n_heads": 4, "d_ff": 512, "init_seed": 1 },
///   "data": "train.ojtk",            // token file (see `train::data`)
///   "heldout": "heldout.ojtk",       // optional; absent: evaluate on the head of `data`
///   "inner_steps": 50, "batch": 8, "inner_lr": 0.001, "weight_decay": 0.1,
///   "outer_lr": 0.7, "outer_momentum": 0.9,
///   "rounds": 100, "min_members": 1, "max_members": 8,
///   "round_deadline_secs": 900,
///   "checkpoint_dir": "ckpt",
///   "eval_seqs": 32
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub run: String,
    pub model: TrainModel,
    pub data: PathBuf,
    #[serde(default)]
    pub heldout: Option<PathBuf>,
    pub inner_steps: u32,
    pub batch: u32,
    pub inner_lr: f32,
    #[serde(default)]
    pub weight_decay: f32,
    /// Outer (Nesterov SGD) learning rate on the averaged pseudo-gradient.
    #[serde(default = "default_outer_lr")]
    pub outer_lr: f32,
    #[serde(default = "default_outer_momentum")]
    pub outer_momentum: f32,
    /// The run is done once this many rounds have closed.
    pub rounds: u32,
    #[serde(default = "default_one")]
    pub min_members: u32,
    /// Member slots; also the shard count, so the data split never changes mid-run.
    pub max_members: u32,
    #[serde(default = "default_deadline")]
    pub round_deadline_secs: u64,
    pub checkpoint_dir: PathBuf,
    /// Held-out sequences evaluated (on the CPU) after every round.
    #[serde(default = "default_eval_seqs")]
    pub eval_seqs: u32,
}

fn default_outer_lr() -> f32 {
    0.7
}
fn default_outer_momentum() -> f32 {
    0.9
}
fn default_one() -> u32 {
    1
}
fn default_deadline() -> u64 {
    900
}
fn default_eval_seqs() -> u32 {
    32
}

impl RunConfig {
    fn tiny(&self) -> &TinyGptConfig {
        let TrainModel::TinyGpt(c) = &self.model;
        c
    }

    fn check(&self) -> Result<()> {
        tiny_gpt::check_config(self.tiny())?;
        ensure!(!self.run.is_empty(), "run name is empty");
        ensure!(self.inner_steps > 0 && self.batch > 0, "inner_steps and batch must be positive");
        ensure!(self.inner_lr.is_finite() && self.inner_lr > 0.0, "inner_lr {}", self.inner_lr);
        ensure!(self.weight_decay.is_finite() && self.weight_decay >= 0.0, "weight_decay {}", self.weight_decay);
        ensure!(self.outer_lr.is_finite() && self.outer_lr > 0.0, "outer_lr {}", self.outer_lr);
        ensure!(self.outer_momentum.is_finite() && (0.0..1.0).contains(&self.outer_momentum), "outer_momentum {}", self.outer_momentum);
        ensure!(self.max_members > 0 && (1..=self.max_members).contains(&self.min_members), "min_members {} / max_members {}", self.min_members, self.max_members);
        ensure!(self.rounds > 0, "rounds must be positive");
        ensure!(self.round_deadline_secs > 0, "round_deadline_secs must be positive");
        Ok(())
    }
}

/// The outer step. `deltas` = (tokens, `theta_start - theta_end`) per member, in member-id
/// order (the order is part of the result: f64 sums are not associative). The
/// pseudo-gradient is the token-weighted mean `g = Σ (tᵢ / Σt) Δᵢ`, so a member that
/// trained on more tokens counts proportionally more; then Nesterov SGD as PyTorch
/// defines it: `m ← μ m + g`, `θ ← θ − lr (g + μ m)`. Accumulates in f64; nothing is
/// written (θ or momentum) unless every result is finite.
pub fn aggregate(theta: &[f32], momentum: &mut [f32], deltas: &[(u64, &[f32])], outer_lr: f32, mu: f32) -> Result<Vec<f32>> {
    ensure!(!deltas.is_empty(), "aggregate: no deltas");
    ensure!(momentum.len() == theta.len(), "aggregate: momentum {} for θ {}", momentum.len(), theta.len());
    for (t, d) in deltas {
        ensure!(d.len() == theta.len(), "aggregate: delta of {} for θ {}", d.len(), theta.len());
        ensure!(*t > 0, "aggregate: a delta trained on 0 tokens");
    }
    let total: f64 = deltas.iter().map(|(t, _)| *t as f64).sum();
    let w: Vec<f64> = deltas.iter().map(|(t, _)| *t as f64 / total).collect();
    let (lr, mu) = (outer_lr as f64, mu as f64);
    let mut out = Vec::with_capacity(theta.len());
    let mut m_new = Vec::with_capacity(theta.len());
    for j in 0..theta.len() {
        let mut g = 0.0f64;
        for (i, (_, d)) in deltas.iter().enumerate() {
            g += w[i] * d[j] as f64;
        }
        let m = mu * momentum[j] as f64 + g;
        let th = theta[j] as f64 - lr * (g + mu * m);
        let (m32, th32) = (m as f32, th as f32);
        ensure!(m32.is_finite() && th32.is_finite(), "aggregate: non-finite result at {j}");
        m_new.push(m32);
        out.push(th32);
    }
    momentum.copy_from_slice(&m_new);
    Ok(out)
}

/// Mean next-token loss of θ on the first `max_seqs` sequences of a token file, on the CPU.
pub fn eval_loss(cfg: &TinyGptConfig, theta: &[f32], file: &[u8], max_seqs: u32) -> Result<f32> {
    let toks = data::Tokens::parse(file)?;
    ensure!(toks.vocab <= cfg.vocab, "eval file vocab {} exceeds the model's {}", toks.vocab, cfg.vocab);
    let ctx = cfg.ctx as usize;
    let n = toks.n_seqs(ctx).min(max_seqs.max(1) as u64);
    ensure!(n > 0, "eval file is shorter than one window");
    let model = TinyGpt::new(&Cpu, cfg, Some(theta))?;
    let (mut sum, mut buf) = (0.0f64, Vec::new());
    let mut s = 0;
    while s < n {
        let b = (n - s).min(8);
        buf.clear();
        for k in s..s + b {
            toks.window(k, ctx, &mut buf)?;
        }
        sum += model.loss(&Cpu, &buf, b as usize)? as f64 * b as f64;
        s += b;
    }
    Ok((sum / n as f64) as f32)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemberRec {
    pub peer: String,
    pub backend: Backend,
    pub device: String,
}

/// One closed round.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoundLog {
    pub round: u32,
    /// (member, tokens, member's mean inner loss), in member order
    pub members: Vec<(u32, u64, f32)>,
    /// held-out loss of the θ this round produced
    pub heldout_loss: Option<f32>,
    pub base_after: ModelId,
}

/// A summary for logs.
#[derive(Clone, Debug, PartialEq)]
pub struct Status {
    pub run: String,
    pub round: u32,
    pub rounds: u32,
    pub base: ModelId,
    pub bound: u32,
    pub received: u32,
    pub done: bool,
    pub last_heldout: Option<f32>,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "run {} round {}/{} base {} members {} received {}", self.run, self.round, self.rounds, self.base.short(), self.bound, self.received)?;
        if let Some(l) = self.last_heldout {
            write!(f, " heldout {l:.4}")?;
        }
        if self.done {
            write!(f, " (done)")?;
        }
        Ok(())
    }
}

struct Upload {
    blob: BlobId,
    total: u64,
    buf: Vec<u8>,
    verified: bool,
}

struct Submitted {
    report: DeltaReport,
    delta: Vec<f32>,
    blob: BlobId,
}

pub struct Coordinator {
    cfg: RunConfig,
    arch: ArchId,
    n_params: usize,
    data: Vec<u8>,
    data_ref: DataRef,
    heldout: Option<Vec<u8>>,
    theta: Vec<f32>,
    momentum: Vec<f32>,
    base: ModelId,
    theta_blob: Vec<u8>,
    theta_id: BlobId,
    round: u32,
    opened: Instant,
    members: Vec<Option<MemberRec>>,
    deltas: BTreeMap<u32, Submitted>,
    /// at most one in-flight or verified delta blob per peer, each at most θ's size
    uploads: HashMap<String, Upload>,
    initial_heldout: Option<f32>,
    history: Vec<RoundLog>,
}

const CKPT_MAGIC: &[u8; 4] = b"OJCK";
const CKPT_VERSION: u32 = 1;
const CKPT_FILE: &str = "latest.ckpt";

#[derive(Serialize, Deserialize)]
struct CkptHeader {
    run: String,
    arch: ArchId,
    round: u32,
    base: ModelId,
    n_params: u64,
    /// the token file the cursors index into: resuming over other data would silently
    /// change what every recorded shard range means
    data: BlobId,
    members: Vec<Option<MemberRec>>,
    initial_heldout: Option<f32>,
    history: Vec<RoundLog>,
}

fn refuse(msg: impl Into<String>) -> (PeerResp, Vec<u8>) {
    (PeerResp::Error { message: msg.into(), retry: false }, Vec::new())
}

fn nack(reason: impl Into<String>) -> (PeerResp, Vec<u8>) {
    (PeerResp::TrainAck { accepted: false, reason: reason.into() }, Vec::new())
}

impl Coordinator {
    /// Read the run config, its token files, and resume from `checkpoint_dir` if a
    /// checkpoint is there.
    pub fn from_config_file(path: impl AsRef<Path>) -> Result<Coordinator> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).with_context(|| format!("reading run config {}", path.display()))?;
        let mut cfg: RunConfig = serde_json::from_str(&text).with_context(|| format!("parsing run config {}", path.display()))?;
        let dir = path.parent().unwrap_or(Path::new("."));
        let rel = |p: &Path| if p.is_relative() { dir.join(p) } else { p.to_path_buf() };
        cfg.data = rel(&cfg.data);
        cfg.heldout = cfg.heldout.as_deref().map(rel);
        cfg.checkpoint_dir = rel(&cfg.checkpoint_dir);
        let data = std::fs::read(&cfg.data).with_context(|| format!("reading data {}", cfg.data.display()))?;
        let heldout = match &cfg.heldout {
            Some(p) => Some(std::fs::read(p).with_context(|| format!("reading held-out data {}", p.display()))?),
            None => None,
        };
        Coordinator::new(cfg, data, heldout)
    }

    /// From an in-memory config and token files (paths in `cfg` other than
    /// `checkpoint_dir` are not read).
    pub fn new(cfg: RunConfig, data: Vec<u8>, heldout: Option<Vec<u8>>) -> Result<Coordinator> {
        cfg.check()?;
        let tiny = cfg.tiny().clone();
        let toks = data::Tokens::parse(&data).context("training data")?;
        ensure!(toks.vocab <= tiny.vocab, "data vocab {} exceeds the model's {}", toks.vocab, tiny.vocab);
        ensure!(toks.n_seqs(tiny.ctx as usize) > 0, "training data is shorter than one window");
        if let Some(h) = &heldout {
            let t = data::Tokens::parse(h).context("held-out data")?;
            ensure!(t.vocab <= tiny.vocab && t.n_seqs(tiny.ctx as usize) > 0, "held-out data: vocab {} / {} tokens", t.vocab, t.n);
        }
        let theta = tiny_gpt::init(&tiny);
        let id = tiny_gpt::identity(&tiny, &theta)?;
        let n_params = theta.len();
        let data_ref = DataRef { blob: BlobId::of(&data), bytes: data.len() as u64 };
        let theta_blob = Tensor::vector(theta.clone()).encode(DType::F32);
        let mut c = Coordinator {
            arch: id.arch,
            n_params,
            data,
            data_ref,
            heldout,
            momentum: vec![0.0; n_params],
            base: id.model,
            theta_id: BlobId::of(&theta_blob),
            theta_blob,
            theta,
            round: 0,
            opened: Instant::now(),
            members: vec![None; cfg.max_members as usize],
            deltas: BTreeMap::new(),
            uploads: HashMap::new(),
            initial_heldout: None,
            history: Vec::new(),
            cfg,
        };
        if !c.resume()? {
            c.initial_heldout = c.eval().ok();
            std::fs::create_dir_all(&c.cfg.checkpoint_dir).with_context(|| format!("creating {}", c.cfg.checkpoint_dir.display()))?;
            c.checkpoint()?;
        }
        tracing::info!("coordinator: {}", c.status());
        Ok(c)
    }

    // ------------------------------------------------------------ accessors ---

    pub fn config(&self) -> &RunConfig {
        &self.cfg
    }
    pub fn theta(&self) -> &[f32] {
        &self.theta
    }
    pub fn momentum(&self) -> &[f32] {
        &self.momentum
    }
    pub fn round(&self) -> u32 {
        self.round
    }
    pub fn base(&self) -> ModelId {
        self.base
    }
    pub fn arch(&self) -> ArchId {
        self.arch
    }
    pub fn data_ref(&self) -> &DataRef {
        &self.data_ref
    }
    pub fn history(&self) -> &[RoundLog] {
        &self.history
    }
    pub fn initial_heldout(&self) -> Option<f32> {
        self.initial_heldout
    }
    pub fn done(&self) -> bool {
        self.round >= self.cfg.rounds
    }
    pub fn members(&self) -> &[Option<MemberRec>] {
        &self.members
    }

    pub fn status(&self) -> Status {
        Status {
            run: self.cfg.run.clone(),
            round: self.round,
            rounds: self.cfg.rounds,
            base: self.base,
            bound: self.bound_count(),
            received: self.deltas.len() as u32,
            done: self.done(),
            last_heldout: self.history.last().and_then(|h| h.heldout_loss).or(self.initial_heldout),
        }
    }

    fn bound_count(&self) -> u32 {
        self.members.iter().filter(|m| m.is_some()).count() as u32
    }

    fn cursor(&self) -> u64 {
        self.round as u64 * self.cfg.inner_steps as u64 * self.cfg.batch as u64
    }

    fn spec(&self, member: u32) -> TrainSpec {
        TrainSpec {
            run: self.cfg.run.clone(),
            model: self.cfg.model.clone(),
            arch: self.arch,
            data: self.data_ref.clone(),
            shard: member,
            n_shards: self.cfg.max_members,
            inner_steps: self.cfg.inner_steps,
            batch: self.cfg.batch,
            inner_lr: self.cfg.inner_lr,
            weight_decay: self.cfg.weight_decay,
        }
    }

    fn slot_of(&self, peer: &str) -> Option<u32> {
        self.members.iter().position(|m| m.as_ref().is_some_and(|m| m.peer == peer)).map(|i| i as u32)
    }

    /// `member` exists and is bound to `from`.
    fn check_bound(&self, from: &str, member: u32) -> std::result::Result<(), String> {
        match self.members.get(member as usize) {
            Some(Some(m)) if m.peer == from => Ok(()),
            Some(Some(_)) => Err(format!("member {member} is bound to another peer")),
            Some(None) => Err(format!("member {member} is not bound; join first")),
            None => Err(format!("no member {member}")),
        }
    }

    fn state(&self, member: u32) -> TrainState {
        let status = if self.done() {
            RoundStatus::Done
        } else if self.deltas.contains_key(&member) {
            RoundStatus::Submitted
        } else {
            RoundStatus::Open
        };
        TrainState {
            round: RoundSpec { round: self.round, base: self.base, cursor: self.cursor() },
            theta: self.theta_id,
            theta_bytes: self.theta_blob.len() as u64,
            status,
            received: self.deltas.len() as u32,
            expected: self.bound_count(),
        }
    }

    // -------------------------------------------------------------- handle ---

    /// `from` is the requesting peer's id (libp2p PeerId as a string), so a member slot
    /// can be bound to the peer that joined it.
    pub fn handle(&mut self, from: &str, req: PeerReq, payload: Vec<u8>) -> (PeerResp, Vec<u8>) {
        self.tick_at(Instant::now());
        match req {
            PeerReq::TrainJoin { run, backend, device, .. } => self.join(from, &run, backend, device),
            PeerReq::TrainSync { run, member } => {
                if run != self.cfg.run {
                    return refuse(format!("unknown run {run}"));
                }
                match self.check_bound(from, member) {
                    Ok(()) => (PeerResp::TrainState(self.state(member)), Vec::new()),
                    Err(e) => refuse(e),
                }
            }
            PeerReq::BlobGet { blob, offset, len } => self.blob_get(from, blob, offset, len),
            PeerReq::BlobPut { blob, offset, total } => self.blob_put(from, blob, offset, total, payload),
            PeerReq::TrainPush { member, report, delta, delta_bytes } => {
                let r = self.push(from, member, report, delta, delta_bytes);
                self.tick_at(Instant::now());
                r
            }
            PeerReq::TrainLeave { run, member } => {
                if run != self.cfg.run {
                    return refuse(format!("unknown run {run}"));
                }
                if let Err(e) = self.check_bound(from, member) {
                    return refuse(e);
                }
                // an accepted delta stays: it is valid work for this round
                self.members[member as usize] = None;
                self.uploads.remove(from);
                tracing::info!("coordinator: member {member} ({from}) left");
                self.tick_at(Instant::now());
                (PeerResp::Ok, Vec::new())
            }
            PeerReq::Generate(_) | PeerReq::Score(_) => refuse("a coordinator does not serve models"),
        }
    }

    fn join(&mut self, from: &str, run: &str, backend: Backend, device: String) -> (PeerResp, Vec<u8>) {
        if run != self.cfg.run {
            return refuse(format!("unknown run {run}"));
        }
        let slot = match self.slot_of(from) {
            Some(s) => s,
            None => match self.members.iter().position(|m| m.is_none()) {
                Some(s) => s as u32,
                None => return (PeerResp::Error { message: format!("run {run} is full ({} members)", self.cfg.max_members), retry: true }, Vec::new()),
            },
        };
        self.members[slot as usize] = Some(MemberRec { peer: from.to_string(), backend, device });
        tracing::info!("coordinator: member {slot} = {from} ({})", backend.as_str());
        (PeerResp::TrainJoined { member: slot, spec: self.spec(slot) }, Vec::new())
    }

    fn blob_get(&self, from: &str, blob: BlobId, offset: u64, len: u32) -> (PeerResp, Vec<u8>) {
        // the pool is invite-only, and the data is the run's: serve bound members only
        if self.slot_of(from).is_none() {
            return refuse("not a member of this run");
        }
        let bytes: &[u8] = if blob == self.theta_id {
            &self.theta_blob
        } else if blob == self.data_ref.blob {
            &self.data
        } else {
            return refuse(format!("unknown blob {} (θ is {})", blob.short(), self.theta_id.short()));
        };
        let total = bytes.len() as u64;
        if offset > total {
            return refuse(format!("offset {offset} past the end ({total})"));
        }
        let end = (offset + len.min(BLOB_CHUNK) as u64).min(total);
        (PeerResp::Blob { blob, offset, total }, bytes[offset as usize..end as usize].to_vec())
    }

    /// Chunks arrive in order (the member sends them sequentially); a repeated chunk is
    /// acknowledged without effect, so a retry after a lost response is harmless.
    fn blob_put(&mut self, from: &str, blob: BlobId, offset: u64, total: u64, chunk: Vec<u8>) -> (PeerResp, Vec<u8>) {
        if self.slot_of(from).is_none() {
            return refuse("not a member of this run");
        }
        // a delta is never larger than θ in f32: that bounds what one peer can make us hold
        let max = self.theta_blob.len() as u64;
        if total == 0 || total > max {
            return refuse(format!("blob of {total} bytes; a delta is at most {max}"));
        }
        if chunk.len() as u64 > BLOB_CHUNK as u64 || offset.checked_add(chunk.len() as u64).is_none_or(|e| e > total) {
            return refuse(format!("chunk {offset}+{} outside a blob of {total}", chunk.len()));
        }
        let up = self.uploads.entry(from.to_string()).or_insert_with(|| Upload { blob, total, buf: Vec::new(), verified: false });
        if up.blob != blob || up.total != total {
            // a new blob replaces this peer's previous one
            *up = Upload { blob, total, buf: Vec::new(), verified: false };
        }
        let have = up.buf.len() as u64;
        if offset == have {
            up.buf.extend_from_slice(&chunk);
        } else if offset + (chunk.len() as u64) <= have {
            if up.buf[offset as usize..offset as usize + chunk.len()] != chunk[..] {
                self.uploads.remove(from);
                return refuse("a resent chunk differs from the first copy; upload dropped");
            }
        } else {
            return refuse(format!("out-of-order chunk at {offset}; next expected {have}"));
        }
        if up.buf.len() as u64 == up.total && !up.verified {
            if BlobId::of(&up.buf) != up.blob {
                self.uploads.remove(from);
                return refuse("blob does not hash to its id; upload dropped");
            }
            up.verified = true;
        }
        (PeerResp::Ok, Vec::new())
    }

    fn push(&mut self, from: &str, member: u32, report: DeltaReport, delta: BlobId, delta_bytes: u64) -> (PeerResp, Vec<u8>) {
        if let Err(e) = self.check_bound(from, member) {
            return nack(e);
        }
        if report.run != self.cfg.run {
            return nack(format!("delta for run {}, this is {}", report.run, self.cfg.run));
        }
        if self.done() {
            return nack("the run is done");
        }
        if report.round != self.round {
            return nack(format!("delta for round {}, the current round is {}", report.round, self.round));
        }
        if report.base != self.base {
            return nack(format!("delta from base {}, the round's base is {}", report.base.short(), self.base.short()));
        }
        if report.arch != self.arch {
            return nack(format!("delta for arch {}, the run's is {}", report.arch.short(), self.arch.short()));
        }
        if let Some(s) = self.deltas.get(&member) {
            return if s.blob == delta {
                (PeerResp::TrainAck { accepted: true, reason: "already received".into() }, Vec::new())
            } else {
                nack(format!("member {member} already submitted a delta for round {}", self.round))
            };
        }
        // the report must describe exactly the data this member was assigned
        let (first, stride) = data::stride_of(self.cursor(), member, self.cfg.max_members);
        let seqs = self.cfg.inner_steps as u64 * self.cfg.batch as u64;
        let ctx = self.cfg.tiny().ctx as u64;
        if report.first_seq != first || report.stride != stride || report.seqs != seqs || report.tokens != seqs * ctx {
            return nack(format!(
                "report covers first {} stride {} seqs {} tokens {}; assigned first {first} stride {stride} seqs {seqs} tokens {}",
                report.first_seq,
                report.stride,
                report.seqs,
                report.tokens,
                seqs * ctx
            ));
        }
        if !report.mean_loss.is_finite() || !report.final_loss.is_finite() {
            return nack("report loss is not finite");
        }
        let bytes = match self.uploads.get(from) {
            Some(u) if u.blob == delta && u.verified => &u.buf,
            Some(u) if u.blob == delta => return nack(format!("delta blob incomplete ({} of {} bytes)", u.buf.len(), u.total)),
            _ => return nack(format!("delta blob {} was not put", delta.short())),
        };
        if bytes.len() as u64 != delta_bytes {
            return nack(format!("delta_bytes {delta_bytes}, the blob is {}", bytes.len()));
        }
        let t = match Tensor::decode(bytes) {
            Ok((t, used)) if used == bytes.len() => t,
            Ok((_, used)) => return nack(format!("delta blob has {} trailing bytes", bytes.len() - used)),
            Err(e) => return nack(format!("delta does not decode: {e}")),
        };
        if t.shape != [self.n_params] {
            return nack(format!("delta shape {:?}, the model has {} parameters", t.shape, self.n_params));
        }
        // Tensor::decode refuses NaN/Inf; check again so the invariant does not rest on it
        if t.data.iter().any(|v| !v.is_finite()) {
            return nack("delta is not finite");
        }
        self.uploads.remove(from);
        tracing::info!("coordinator: round {} delta from member {member} ({}, {} tokens, loss {:.4})", self.round, report.backend.as_str(), report.tokens, report.mean_loss);
        self.deltas.insert(member, Submitted { report, delta: t.data, blob: delta });
        (PeerResp::TrainAck { accepted: true, reason: String::new() }, Vec::new())
    }

    // -------------------------------------------------------------- rounds ---

    /// Close the round if it is due at `now`: every bound member submitted, or the
    /// deadline passed — either way with at least `min_members` deltas. Returns whether
    /// a round closed. [`handle`](Self::handle) calls it on every request; a node may
    /// also call it on a timer.
    pub fn tick_at(&mut self, now: Instant) -> bool {
        if self.done() || (self.deltas.len() as u32) < self.cfg.min_members {
            return false;
        }
        let all = self.members.iter().enumerate().all(|(i, m)| m.is_none() || self.deltas.contains_key(&(i as u32)));
        let late = now.saturating_duration_since(self.opened) >= Duration::from_secs(self.cfg.round_deadline_secs);
        if !(all || late) {
            return false;
        }
        match self.close_round() {
            Ok(()) => true,
            Err(e) => {
                // θ is untouched; the round restarts with fresh deltas
                tracing::warn!("coordinator: round {} not applied: {e:#}", self.round);
                self.deltas.clear();
                self.opened = Instant::now();
                false
            }
        }
    }

    fn close_round(&mut self) -> Result<()> {
        let subs: Vec<(u64, &[f32])> = self.deltas.values().map(|s| (s.report.tokens, s.delta.as_slice())).collect();
        let mut mom = self.momentum.clone();
        let theta = aggregate(&self.theta, &mut mom, &subs, self.cfg.outer_lr, self.cfg.outer_momentum)?;
        let id = tiny_gpt::identity(self.cfg.tiny(), &theta)?;
        let members = self.deltas.iter().map(|(m, s)| (*m, s.report.tokens, s.report.mean_loss)).collect();
        let closed = self.round;
        self.theta = theta;
        self.momentum = mom;
        self.base = id.model;
        self.theta_blob = Tensor::vector(self.theta.clone()).encode(DType::F32);
        self.theta_id = BlobId::of(&self.theta_blob);
        self.round += 1;
        self.deltas.clear();
        // anything half-uploaded was computed from the old base
        self.uploads.clear();
        let heldout_loss = match self.eval() {
            Ok(l) => Some(l),
            Err(e) => {
                tracing::warn!("coordinator: held-out eval failed: {e:#}");
                None
            }
        };
        self.history.push(RoundLog { round: closed, members, heldout_loss, base_after: self.base });
        self.checkpoint()?;
        self.opened = Instant::now();
        tracing::info!("coordinator: closed round {closed}: {}", self.status());
        Ok(())
    }

    fn eval(&self) -> Result<f32> {
        let file = self.heldout.as_deref().unwrap_or(&self.data);
        eval_loss(self.cfg.tiny(), &self.theta, file, self.cfg.eval_seqs)
    }

    // ---------------------------------------------------------- checkpoint ---

    pub fn checkpoint_path(&self) -> PathBuf {
        self.cfg.checkpoint_dir.join(CKPT_FILE)
    }

    /// `"OJCK" | u32 version | u64 header_len | header JSON | θ tensor | momentum tensor`,
    /// written to a temporary file, synced, then renamed over the previous one: a crash
    /// leaves either the old checkpoint or the new one, never a torn one.
    fn checkpoint(&self) -> Result<()> {
        use std::io::Write;
        let h = CkptHeader {
            run: self.cfg.run.clone(),
            arch: self.arch,
            round: self.round,
            base: self.base,
            n_params: self.n_params as u64,
            data: self.data_ref.blob,
            members: self.members.clone(),
            initial_heldout: self.initial_heldout,
            history: self.history.clone(),
        };
        let hj = serde_json::to_vec(&h)?;
        let mut out = Vec::with_capacity(16 + hj.len() + 2 * self.theta_blob.len());
        out.extend_from_slice(CKPT_MAGIC);
        out.extend_from_slice(&CKPT_VERSION.to_le_bytes());
        out.extend_from_slice(&(hj.len() as u64).to_le_bytes());
        out.extend_from_slice(&hj);
        out.extend_from_slice(&self.theta_blob);
        out.extend_from_slice(&Tensor::vector(self.momentum.clone()).encode(DType::F32));
        let path = self.checkpoint_path();
        let tmp = path.with_extension("ckpt.tmp");
        {
            let mut f = std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
            f.write_all(&out)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path).with_context(|| format!("renaming {} over {}", tmp.display(), path.display()))?;
        Ok(())
    }

    /// Load `checkpoint_dir/latest.ckpt` if present. A checkpoint of another run or
    /// another schema is an error, not something to silently start over from.
    fn resume(&mut self) -> Result<bool> {
        let path = self.checkpoint_path();
        let b = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let ctx = || format!("checkpoint {}", path.display());
        ensure!(b.len() >= 16 && &b[..4] == CKPT_MAGIC, "{}: not a checkpoint", ctx());
        ensure!(u32::from_le_bytes(b[4..8].try_into().unwrap()) == CKPT_VERSION, "{}: unknown version", ctx());
        let hl = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
        ensure!(b.len() >= 16 + hl, "{}: truncated header", ctx());
        let h: CkptHeader = serde_json::from_slice(&b[16..16 + hl]).with_context(ctx)?;
        ensure!(h.run == self.cfg.run, "{}: run {}, config says {}", ctx(), h.run, self.cfg.run);
        ensure!(h.arch == self.arch, "{}: arch {}, config builds {}", ctx(), h.arch.short(), self.arch.short());
        ensure!(h.n_params as usize == self.n_params, "{}: {} params", ctx(), h.n_params);
        ensure!(h.data == self.data_ref.blob, "{}: trained on data {}, config gives {}", ctx(), h.data.short(), self.data_ref.blob.short());
        let rest = &b[16 + hl..];
        let (theta, used) = Tensor::decode(rest).with_context(ctx)?;
        let (mom, used2) = Tensor::decode(&rest[used..]).with_context(ctx)?;
        ensure!(used + used2 == rest.len(), "{}: trailing bytes", ctx());
        ensure!(theta.shape == [self.n_params] && mom.shape == [self.n_params], "{}: tensor shapes", ctx());
        let id = tiny_gpt::identity(self.cfg.tiny(), &theta.data)?;
        ensure!(id.model == h.base, "{}: θ hashes to {}, header says {}", ctx(), id.model.short(), h.base.short());
        let mut members = h.members;
        ensure!(members.iter().skip(self.cfg.max_members as usize).all(|m| m.is_none()), "{}: members beyond max_members", ctx());
        members.resize(self.cfg.max_members as usize, None);
        self.theta = theta.data;
        self.momentum = mom.data;
        self.base = id.model;
        self.theta_blob = Tensor::vector(self.theta.clone()).encode(DType::F32);
        self.theta_id = BlobId::of(&self.theta_blob);
        self.round = h.round;
        self.members = members;
        self.initial_heldout = h.initial_heldout;
        self.history = h.history;
        self.opened = Instant::now();
        tracing::info!("coordinator: resumed {} at round {}", path.display(), self.round);
        Ok(true)
    }
}
