//! `ojas engine-worker` — the per-device compute process of the swarm.
//!
//! The node starts one per device and listens; the worker connects to it, sends
//! `Ready`, and then answers `ipc::NodeMsg` frames until the node hangs up or says
//! `Shutdown`. It opens no other socket: everything network-facing lives in the node,
//! and killing this process is how the node frees the device or recovers from a fault.
//!
//! Shape:
//!
//! * A reader thread owns the socket's read half and feeds a channel. That is what
//!   lets a `Cancel` reach a generation in progress: the decode loop polls the channel
//!   from its token callback, and the reader also raises `STREAM_CANCEL` directly so a
//!   long prefill stops between chunks.
//! * The model lives only inside `backend::with_model`'s callback (a Metal decoder
//!   borrows its device). So the main loop waits for `Load`; the callback serves until
//!   `Unload`, another `Load`, `Shutdown` or EOF, then returns so the next model can be
//!   loaded in its place.
//! * Training is a separate state that survives model loads: a run's model is built
//!   by `ojas_swarm::train::begin` and kept until `TrainEnd`. It shares the device with
//!   a served model if one is loaded; on unified memory (Metal, CPU) both fit for the
//!   models DiLoCo trains here, and a run that does not fit fails its `TrainBegin`
//!   rather than evicting what is being served.

use crate::backend::{self, with_model, Device};
use crate::detok::Detok;
use anyhow::{bail, Context, Result};
use ojas_core::cancel::STREAM_CANCEL;
use ojas_core::{device_fault, Model};
use ojas_infer::{EngineCore, FinishReason, SampleOpts};
use ojas_swarm::train::RoundTrainer;
use ojas_swarm_proto::frame::{self, FrameError, Limits};
use ojas_swarm_proto::ipc::{NodeMsg, Phase, WorkerMsg};
use ojas_swarm_proto::{
    features, Backend, DType, Finish, GenerateReq, Identity, ModelId, ModelInfo, ReqId, Sampling, ScoreReq,
    Tensor, WorkerCaps, PROTO_VERSION,
};
use ojas_tokenize::Bpe;
use std::collections::{HashSet, VecDeque};
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

/// Architectures the Metal decoder loads. It has no list of its own (it builds by
/// mechanism and warns on partial support), so this mirrors what it accepts.
pub const METAL_ARCHS: &[&str] =
    &["qwen2", "qwen3", "llama", "gemma3", "glm-dsa", "deepseek2", "qwen35", "qwen35moe", "qwen4exp", "gpt-oss"];

const DEFAULT_CONTEXT: usize = 4096;
/// Positions per batched scoring forward: the decoders' batch ceiling.
const SCORE_CHUNK: usize = 256;

/// `argv` is the whole command line, `argv[1] == "engine-worker"`.
pub fn main(argv: Vec<String>) -> Result<()> {
    // --socket and --context are this command's own; everything else is an engine
    // or runtime flag (--device, --precision, -c, ...) and goes through the shared
    // parser, which rejects anything unknown.
    let (mut socket, mut context, mut rest) = (None, None, Vec::with_capacity(argv.len()));
    let mut it = argv.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--socket" => socket = Some(it.next().context("--socket needs an address")?),
            "--context" => context = Some(it.next().context("--context needs a number")?.parse::<usize>()?),
            _ => rest.push(a),
        }
    }
    let (cfg, opts, pos) = crate::flags::parse(rest)?;
    if pos.len() > 2 {
        bail!("engine-worker takes no positional arguments (got {:?})", &pos[2..]);
    }
    let socket = socket.context("usage: ojas engine-worker --socket <unix:PATH|tcp:HOST:PORT> --device <auto|metal|cuda|cpu> [--context N]")?;
    let context = context.or(cfg.ctx).unwrap_or(DEFAULT_CONTEXT);
    let _ = ojas_core::config::EngineConfig::install(cfg);

    let device = resolve(opts.device);
    let (r, w) = connect(&socket)?;
    let (tx, rx) = mpsc::channel();
    let active = Arc::new(AtomicU64::new(0));
    spawn_reader(r, tx, active.clone());
    let mut wk = Worker {
        rx,
        pending: VecDeque::new(),
        out: BufWriter::new(w),
        active,
        device,
        precision: opts.precision,
        default_context: context,
        trainer: None,
        dead: None,
    };
    let r = wk.run();
    if let Some(e) = wk.dead.take() {
        bail!("node connection lost: {e}");
    }
    r
}

/// `Auto` becomes a concrete device once, at start: a worker is one device, and an
/// explicit device never falls back (`with_model`'s rule), so a model that cannot
/// run here is an error the node sees rather than a silent CPU fallback.
fn resolve(d: Device) -> Device {
    if d != Device::Auto {
        return d;
    }
    #[cfg(target_os = "macos")]
    if ojas_metal::MetalGpu::new().is_ok() {
        return Device::Metal;
    }
    #[cfg(feature = "cuda")]
    if ojas_cuda::device::CudaGpu::device_count().is_ok_and(|n| n > 0) {
        return Device::Cuda;
    }
    Device::Cpu
}

fn backend_of(d: Device) -> Backend {
    match d {
        Device::Metal => Backend::Metal,
        Device::Cuda => Backend::Cuda,
        Device::Cpu | Device::Auto => Backend::Cpu,
    }
}

type Halves = (Box<dyn Read + Send>, Box<dyn Write + Send>);

fn connect(addr: &str) -> Result<Halves> {
    if let Some(p) = addr.strip_prefix("unix:") {
        #[cfg(unix)]
        {
            let s = std::os::unix::net::UnixStream::connect(p).with_context(|| format!("connecting to {addr}"))?;
            return Ok((Box::new(s.try_clone()?), Box::new(s)));
        }
        #[cfg(not(unix))]
        bail!("unix sockets are not available here ({p}); use tcp:HOST:PORT");
    }
    if let Some(a) = addr.strip_prefix("tcp:") {
        let s = std::net::TcpStream::connect(a).with_context(|| format!("connecting to {addr}"))?;
        // One small frame per token: Nagle would hold each back for the previous ACK.
        s.set_nodelay(true)?;
        return Ok((Box::new(s.try_clone()?), Box::new(s)));
    }
    bail!("--socket {addr:?}: want unix:PATH or tcp:HOST:PORT")
}

enum Event {
    Msg(NodeMsg, Vec<u8>),
    Eof,
    /// The stream cannot be read further (oversized or undecodable frame, I/O error).
    Broken(String),
}

/// `active` holds the id + 1 of the generation in flight, 0 for none.
fn spawn_reader(r: Box<dyn Read + Send>, tx: Sender<Event>, active: Arc<AtomicU64>) {
    std::thread::Builder::new()
        .name("ipc-reader".into())
        .spawn(move || {
            let mut r = BufReader::new(r);
            let mut skipped = HashSet::new();
            let stop_active = || {
                if active.load(Ordering::Acquire) != 0 {
                    STREAM_CANCEL.store(true, Ordering::Relaxed);
                }
            };
            loop {
                let got = frame::read_known::<_, NodeMsg>(&mut r, Limits::IPC, &mut |t| {
                    if skipped.insert(t.to_string()) {
                        tracing::warn!("skipping unknown message {t:?} (newer node?)");
                    }
                });
                let ev = match got {
                    Ok((m, p)) => {
                        if let NodeMsg::Cancel { req } = &m {
                            if active.load(Ordering::Acquire) == req.0.wrapping_add(1) {
                                STREAM_CANCEL.store(true, Ordering::Relaxed);
                            }
                        }
                        Event::Msg(m, p)
                    }
                    Err(e) if e.is_eof() => {
                        stop_active();
                        let _ = tx.send(Event::Eof);
                        return;
                    }
                    Err(e) => {
                        stop_active();
                        let _ = tx.send(Event::Broken(e.to_string()));
                        return;
                    }
                };
                if tx.send(ev).is_err() {
                    return;
                }
            }
        })
        .expect("spawning the IPC reader");
}

/// What the serving loop does next.
enum Step {
    Idle,
    Load { path: String, context: u32, expect: Option<ModelId> },
    Exit,
    Broken(String),
}

struct Worker {
    rx: Receiver<Event>,
    /// Events read while a generation was running and not meant for it.
    pending: VecDeque<Event>,
    out: BufWriter<Box<dyn Write + Send>>,
    active: Arc<AtomicU64>,
    device: Device,
    precision: Option<u8>,
    default_context: usize,
    trainer: Option<(String, Box<dyn RoundTrainer>)>,
    /// Set when a write to the node failed: nothing more can be said to it.
    dead: Option<String>,
}

impl Worker {
    fn run(&mut self) -> Result<()> {
        let caps = self.caps();
        tracing::info!("engine-worker: {} ({}), {} MB", caps.device, caps.backend.as_str(), caps.memory_bytes >> 20);
        self.send(&WorkerMsg::Ready(caps), &[]);
        let mut step = Step::Idle;
        loop {
            if self.dead.is_some() {
                return Ok(());
            }
            step = match step {
                Step::Exit => return Ok(()),
                Step::Broken(e) => bail!("node stream broken: {e}"),
                Step::Load { path, context, expect } => self.load(&path, context, expect),
                Step::Idle => match self.next_event() {
                    Event::Eof => Step::Exit,
                    Event::Broken(e) => Step::Broken(e),
                    Event::Msg(m, p) => self.idle(m, p),
                },
            };
        }
    }

    fn caps(&self) -> WorkerCaps {
        let backend = backend_of(self.device);
        let (device, memory_bytes) = device_facts(self.device);
        let archs: &[&str] = match backend {
            Backend::Metal => METAL_ARCHS,
            Backend::Cuda => backend::CUDA_ARCHS,
            Backend::Cpu => backend::CPU_ARCHS,
        };
        let trainable = ojas_swarm::train::trainable();
        let mut feats = features::GENERATE | features::SCORE;
        if !trainable.is_empty() {
            feats |= features::TRAIN_DILOCO;
        }
        WorkerCaps {
            proto: PROTO_VERSION,
            features: feats,
            backend,
            device,
            memory_bytes,
            archs: archs.iter().map(|s| s.to_string()).collect(),
            trainable,
            // Requests are served one at a time; slotted batching is the node's to
            // ask for once the worker multiplexes them.
            slots: 1,
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn next_event(&mut self) -> Event {
        self.pending.pop_front().unwrap_or_else(|| self.rx.recv().unwrap_or(Event::Eof))
    }

    /// Drain what the reader has queued without blocking. True when generation `req`
    /// must stop: it was cancelled, or the node is gone.
    fn poll(&mut self, req: ReqId) -> bool {
        let mut stop = false;
        while let Ok(ev) = self.rx.try_recv() {
            match ev {
                Event::Msg(NodeMsg::Cancel { req: r }, _) if r == req => stop = true,
                Event::Msg(NodeMsg::Cancel { .. }, _) => {}
                ev @ (Event::Eof | Event::Broken(_)) => {
                    self.pending.push_back(ev);
                    stop = true;
                }
                ev => self.pending.push_back(ev),
            }
        }
        stop || self.dead.is_some()
    }

    fn send(&mut self, m: &WorkerMsg, payload: &[u8]) {
        if self.dead.is_some() {
            return;
        }
        match frame::write(&mut self.out, m, payload, Limits::IPC) {
            Ok(()) => {}
            // Refused before a byte was written: the stream is still aligned, so the
            // node can be told instead.
            Err(FrameError::TooLarge { what, len, max }) => {
                let message = format!("reply {what} of {len} bytes exceeds the IPC limit {max}");
                let req = req_of(m);
                let _ = frame::write(&mut self.out, &WorkerMsg::Error { req, message, fatal: false }, &[], Limits::IPC);
            }
            Err(e) => self.dead = Some(e.to_string()),
        }
    }

    fn error(&mut self, req: Option<ReqId>, message: impl Into<String>, fatal: bool) {
        let message = message.into();
        tracing::warn!("{}{message}", if fatal { "fatal: " } else { "" });
        self.send(&WorkerMsg::Error { req, message, fatal }, &[]);
    }

    /// Without a model loaded.
    fn idle(&mut self, m: NodeMsg, p: Vec<u8>) -> Step {
        match m {
            NodeMsg::Load { path, context, expect } => Step::Load { path, context, expect },
            NodeMsg::Unload { .. } | NodeMsg::Cancel { .. } => Step::Idle,
            NodeMsg::Generate(GenerateReq { req, .. }) | NodeMsg::Score(ScoreReq { req, .. }) => {
                self.error(Some(req), "no model loaded", false);
                Step::Idle
            }
            m => self.common(m, p),
        }
    }

    /// Messages answered the same with or without a model: training and shutdown.
    fn common(&mut self, m: NodeMsg, p: Vec<u8>) -> Step {
        match m {
            NodeMsg::TrainBegin { spec } => {
                let theta0 = match decode_vec(&p) {
                    Ok(t) => t,
                    Err(e) => {
                        self.error(None, format!("TrainBegin θ0: {e}"), false);
                        return Step::Idle;
                    }
                };
                // A new run replaces the old one: its weights would only hold memory.
                self.trainer = None;
                match ojas_swarm::train::begin(&spec, theta0.as_deref(), backend_of(self.device)) {
                    Ok(t) => {
                        let msg = WorkerMsg::TrainReady { run: spec.run.clone(), identity: t.identity(), n_params: t.n_params() };
                        self.trainer = Some((spec.run, t));
                        self.send(&msg, &[]);
                    }
                    Err(e) => self.error(None, format!("TrainBegin {}: {e:#}", spec.run), device_fault::is_faulted()),
                }
            }
            NodeMsg::TrainRound { round, data_path } => self.train_round(round, &data_path, &p),
            NodeMsg::TrainEnd { run } => {
                if self.trainer.as_ref().is_some_and(|(r, _)| *r == run) {
                    self.trainer = None;
                }
            }
            NodeMsg::Shutdown => {
                self.send(&WorkerMsg::Bye, &[]);
                return Step::Exit;
            }
            NodeMsg::Cancel { .. } => {}
            // Model messages are routed before this point.
            NodeMsg::Load { .. } | NodeMsg::Unload { .. } | NodeMsg::Generate(_) | NodeMsg::Score(_) => {}
        }
        Step::Idle
    }

    fn train_round(&mut self, round: ojas_swarm_proto::RoundSpec, data_path: &str, p: &[u8]) {
        let Some((run, mut t)) = self.trainer.take() else {
            self.error(None, "TrainRound without TrainBegin", false);
            return;
        };
        let res = (|| -> Result<(ojas_swarm_proto::DeltaReport, Vec<f32>)> {
            if let Some(theta) = decode_vec(p)? {
                t.set_weights(&theta)?;
            }
            let data = std::fs::read(data_path).with_context(|| format!("reading {data_path}"))?;
            self.send(&WorkerMsg::Progress { req: None, phase: Phase::Training, done: 0, total: 1 }, &[]);
            t.round(&round, &data)
        })();
        match res {
            Ok((report, delta)) => {
                let payload = Tensor::vector(delta).encode(DType::F32);
                let msg = WorkerMsg::Delta { report, identity: t.identity() };
                self.send(&msg, &payload);
            }
            Err(e) => self.error(None, format!("TrainRound {} of {run}: {e:#}", round.round), device_fault::is_faulted()),
        }
        self.trainer = Some((run, t));
    }

    fn load(&mut self, path: &str, context: u32, expect: Option<ModelId>) -> Step {
        let t0 = std::time::Instant::now();
        let out = &mut self.out;
        let mut hashing = |done: u64, total: u64| {
            let _ = frame::write(out, &WorkerMsg::Progress { req: None, phase: Phase::Hashing, done, total }, &[], Limits::IPC);
        };
        let (identity, file_bytes) = match ojas_formats::swarm_id::identity(path, &mut hashing) {
            Ok(x) => x,
            Err(e) => {
                self.error(None, format!("Load {path}: {e:#}"), false);
                return Step::Idle;
            }
        };
        if let Some(want) = expect {
            if want != identity.model {
                self.error(None, format!("Load {path}: content is {} but {} was expected", identity.model, want), false);
                return Step::Idle;
            }
        }
        self.send(&WorkerMsg::Progress { req: None, phase: Phase::Loading, done: 0, total: 1 }, &[]);
        let context = if context == 0 { self.default_context } else { context as usize };
        let (device, precision) = (self.device, self.precision);
        let r = with_model(path, device, context, precision, |m, bpe, info| {
            let mi = ModelInfo {
                identity,
                arch: info.arch.clone(),
                n_layers: info.n_layers as u32,
                hidden_dim: info.hidden_dim as u32,
                vocab: info.vocab as u32,
                context: info.context.min(u32::MAX as usize) as u32,
                eog: info.eog.clone(),
                file_bytes,
                backend: backend_of(device),
                load_ms: t0.elapsed().as_millis().min(u32::MAX as u128) as u32,
            };
            tracing::info!("loaded {path} as {} ({} on {}) in {} ms", identity.model.short(), mi.arch, mi.backend.as_str(), mi.load_ms);
            self.send(&WorkerMsg::Loaded(mi), &[]);
            Ok(self.serve(m, bpe, info, identity))
        });
        match r {
            Ok(step) => step,
            Err(e) => {
                let fatal = device_fault::is_faulted();
                self.error(None, format!("Load {path}: {e:#}"), fatal);
                Step::Idle
            }
        }
    }

    /// The loaded state. Returns when the model must go.
    fn serve(&mut self, m: &dyn Model, bpe: &Bpe, info: &backend::ModelInfo, id: Identity) -> Step {
        let mut core = EngineCore::new(m);
        // The node sends templated ids, so the turn may end at the GGUF's EOS or at
        // any template terminator: the full EOG set, as `ojas run --raw` does.
        core.eos = info.eos;
        core.eog = info.eog.clone();
        loop {
            if self.dead.is_some() {
                return Step::Exit;
            }
            let (msg, p) = match self.next_event() {
                Event::Eof => return Step::Exit,
                Event::Broken(e) => return Step::Broken(e),
                Event::Msg(msg, p) => (msg, p),
            };
            match msg {
                // A replacement whose content is wrong must not cost the model being
                // served, so `expect` is checked before this one is dropped. The id is
                // cached, so `load` asking again is free.
                NodeMsg::Load { path, context, expect: Some(want) } => {
                    match ojas_formats::swarm_id::identity(&path, &mut |_, _| {}) {
                        Ok((got, _)) if got.model == want => return Step::Load { path, context, expect: Some(want) },
                        Ok((got, _)) => self.error(None, format!("Load {path}: content is {} but {want} was expected; still serving {}", got.model, id.model), false),
                        Err(e) => self.error(None, format!("Load {path}: {e:#}; still serving {}", id.model), false),
                    }
                }
                NodeMsg::Load { path, context, expect: None } => return Step::Load { path, context, expect: None },
                NodeMsg::Unload { model } if model == id.model => return Step::Idle,
                NodeMsg::Unload { model } => self.error(None, format!("Unload {model}: not loaded (have {})", id.model), false),
                NodeMsg::Generate(req) => self.generate(&core, bpe, info, id.model, req),
                NodeMsg::Score(req) => self.score(m, info, id.model, req),
                msg => {
                    if let Step::Exit = self.common(msg, p) {
                        return Step::Exit;
                    }
                }
            }
        }
    }

    /// Shared request checks. `Err` = already answered.
    fn admit(&mut self, req: ReqId, want: ModelId, have: ModelId) -> Result<(), ()> {
        if want != have {
            self.error(Some(req), format!("model {want} is not loaded (have {have})"), false);
            return Err(());
        }
        if let Some(f) = device_fault::peek() {
            self.error(Some(req), format!("device faulted earlier; restart this worker: {f}"), true);
            return Err(());
        }
        Ok(())
    }

    fn generate(&mut self, core: &EngineCore<&dyn Model>, bpe: &Bpe, info: &backend::ModelInfo, have: ModelId, g: GenerateReq) {
        let req = g.req;
        if self.admit(req, g.model, have).is_err() {
            return;
        }
        let n_prompt = g.prompt.len() as u32;
        let done = |finish, completion_tokens| WorkerMsg::Done { req, prompt_tokens: n_prompt, completion_tokens, finish };
        // An id past the vocabulary would index past the embedding table.
        let bad_id = info.vocab > 0 && g.prompt.iter().any(|&t| t as usize >= info.vocab);
        if g.prompt.is_empty() || g.prompt.len() >= info.context || bad_id {
            self.send(&done(Finish::Refused, 0), &[]);
            return;
        }
        let max = (g.max_tokens as usize).min(info.context - g.prompt.len());
        if max == 0 {
            self.send(&done(Finish::Length, 0), &[]);
            return;
        }
        // A cancel that arrived before the request started.
        if self.poll(req) || self.pending.iter().any(|e| matches!(e, Event::Msg(NodeMsg::Cancel { req: r }, _) if *r == req)) {
            self.send(&done(Finish::Cancelled, 0), &[]);
            return;
        }

        // Requests are independent: a recurrent model would otherwise answer this
        // one from the middle of the last.
        core.model().reset_session();
        let opts = sample_opts(&g.sampling);
        self.active.store(req.0.wrapping_add(1), Ordering::Release);
        ojas_core::cancel::clear_cancel();
        let me = std::cell::RefCell::new(&mut *self);
        let (mut sent, mut stopped) = (0u32, false);
        let mut detok = Detok::default();
        let gen = core.generate_ex(
            &g.prompt,
            max,
            opts.as_ref(),
            None,
            &mut |done, total| {
                let mut w = me.borrow_mut();
                if w.poll(req) {
                    STREAM_CANCEL.store(true, Ordering::Relaxed);
                }
                if total > 0 {
                    w.send(&WorkerMsg::Progress { req: Some(req), phase: Phase::Prefill, done: done as u64, total: total as u64 }, &[]);
                }
            },
            &mut |t| {
                let mut w = me.borrow_mut();
                if stopped || w.poll(req) {
                    stopped = true;
                    return false;
                }
                let text = g.want_text.then(|| detok.push(bpe, t));
                w.send(&WorkerMsg::Token { req, token: t, text }, &[]);
                sent += 1;
                w.dead.is_none()
            },
        );
        drop(me);
        self.active.store(0, Ordering::Release);
        ojas_core::cancel::clear_cancel();
        if let Some(f) = device_fault::peek() {
            self.error(Some(req), format!("device fault after {sent} tokens; output after it is untrustworthy: {f}"), true);
            return;
        }
        let finish = match gen.finish {
            _ if stopped => Finish::Cancelled,
            FinishReason::Stop | FinishReason::Complete | FinishReason::NoAllowedToken => Finish::Stop,
            FinishReason::Length => Finish::Length,
            FinishReason::Caller | FinishReason::Cancelled => Finish::Cancelled,
            FinishReason::Fault => Finish::Fault,
            FinishReason::NoLogits | FinishReason::Refused => Finish::Refused,
        };
        self.send(&done(finish, sent), &[]);
    }

    fn score(&mut self, m: &dyn Model, info: &backend::ModelInfo, have: ModelId, s: ScoreReq) {
        let req = s.req;
        if self.admit(req, s.model, have).is_err() {
            return;
        }
        let (n, from) = (s.tokens.len(), s.from as usize);
        let why = if from == 0 {
            Some("from must be at least 1: the first token has no context to be scored against".to_string())
        } else if from >= n {
            Some(format!("nothing to score: from {from} with {n} tokens"))
        } else if n > info.context {
            Some(format!("{n} tokens exceed the context {}", info.context))
        } else if info.vocab > 0 && s.tokens.iter().any(|&t| t as usize >= info.vocab) {
            Some(format!("a token id is outside the vocabulary of {}", info.vocab))
        } else {
            None
        };
        if let Some(why) = why {
            self.error(Some(req), why, false);
            return;
        }
        m.reset_session();
        self.active.store(req.0.wrapping_add(1), Ordering::Release);
        ojas_core::cancel::clear_cancel();
        let r = score(m, &s.tokens, from, &mut || self.poll(req));
        self.active.store(0, Ordering::Release);
        ojas_core::cancel::clear_cancel();
        if let Some(f) = device_fault::peek() {
            self.error(Some(req), format!("device fault while scoring: {f}"), true);
            return;
        }
        match r {
            Ok(lp) => self.send(&WorkerMsg::Scores { req }, &Tensor::vector(lp).encode(DType::F32)),
            Err(e) => self.error(Some(req), format!("{e:#}"), false),
        }
    }
}

/// Log-probability of `tokens[i]` given `tokens[..i]`, for `i` in `from..`.
///
/// The prefix is prefilled exactly as a generation of `tokens[..from]` would be
/// (all but its last token), then the remaining positions run as decode steps, so
/// the score of a greedy continuation is computed on the same path that chose it.
/// Models with a batched logits forward (Metal dense decoders) score
/// [`SCORE_CHUNK`] positions per pass; the others go one position at a time, which
/// is exact but costs a forward per token.
pub fn score(m: &dyn Model, tokens: &[u32], from: usize, cancelled: &mut dyn FnMut() -> bool) -> Result<Vec<f32>> {
    let pre = &tokens[..from - 1];
    if pre.is_empty() {
        m.prefill(&[], 0);
    } else {
        let mut done = m.reuse_prefix_len(pre).min(pre.len());
        while done < pre.len() {
            if cancelled() {
                bail!("cancelled");
            }
            let end = (done + SCORE_CHUNK).min(pre.len());
            m.prefill(&pre[done..end], done);
            done = end;
        }
    }
    let feed = &tokens[from - 1..tokens.len() - 1];
    let target = &tokens[from..];
    let mut out = Vec::with_capacity(target.len());
    let mut batched = true;
    let mut i = 0;
    while i < feed.len() {
        if cancelled() {
            bail!("cancelled");
        }
        let pos = from - 1 + i;
        if batched {
            let end = (i + SCORE_CHUNK).min(feed.len());
            match m.forward_batch_logits(&feed[i..end], pos) {
                Some(rows) if rows.len() == end - i => {
                    for (row, &t) in rows.iter().zip(&target[i..end]) {
                        out.push(logprob(row, t)?);
                    }
                    i = end;
                    continue;
                }
                _ => batched = false,
            }
        }
        let row = m.forward_logits(feed[i], pos).context("this model does not expose logits")?;
        out.push(logprob(&row, target[i])?);
        i += 1;
    }
    Ok(out)
}

fn logprob(logits: &[f32], t: u32) -> Result<f32> {
    let x = *logits.get(t as usize).context("token id past the logits")?;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max as f64 + logits.iter().map(|&l| ((l - max) as f64).exp()).sum::<f64>().ln();
    let lp = (x as f64 - lse) as f32;
    // A NaN here is a faulted kernel, not a probability; the tensor codec would
    // refuse it on the other side anyway.
    anyhow::ensure!(lp.is_finite(), "non-finite log-probability (logits contain NaN or Inf)");
    Ok(lp)
}

fn sample_opts(s: &Sampling) -> Option<SampleOpts> {
    if s.temperature <= 0.0 {
        return None;
    }
    Some(SampleOpts {
        temperature: s.temperature,
        top_p: if s.top_p > 0.0 && s.top_p <= 1.0 { s.top_p } else { 1.0 },
        top_k: s.top_k as usize,
        repeat_penalty: if s.repeat_penalty > 0.0 { s.repeat_penalty } else { 1.0 },
        repeat_window: SampleOpts::default().repeat_window,
        seed: s.seed,
    })
}

/// An optional f32 vector payload: empty = none.
fn decode_vec(p: &[u8]) -> Result<Option<Vec<f32>>> {
    if p.is_empty() {
        return Ok(None);
    }
    let (t, _) = Tensor::decode(p).map_err(|e| anyhow::anyhow!("{e}"))?;
    anyhow::ensure!(t.shape.len() == 1, "expected a vector, got shape {:?}", t.shape);
    Ok(Some(t.data))
}

fn req_of(m: &WorkerMsg) -> Option<ReqId> {
    match m {
        WorkerMsg::Token { req, .. } | WorkerMsg::Done { req, .. } | WorkerMsg::Scores { req } => Some(*req),
        WorkerMsg::Progress { req, .. } | WorkerMsg::Error { req, .. } => *req,
        _ => None,
    }
}

/// Device name and the memory a model can use on it.
fn device_facts(d: Device) -> (String, u64) {
    match d {
        #[cfg(target_os = "macos")]
        Device::Metal => match ojas_metal::MetalGpu::new() {
            Ok(g) => (g.device.name().to_string(), g.device.recommended_max_working_set_size()),
            Err(_) => ("metal".into(), 0),
        },
        #[cfg(feature = "cuda")]
        Device::Cuda => match ojas_cuda::device::CudaGpu::new(0).and_then(|g| g.properties()) {
            Ok(p) => (p.name, p.memory_bytes as u64),
            Err(_) => ("cuda".into(), 0),
        },
        _ => (cpu_name(), system_memory()),
    }
}

fn cpu_name() -> String {
    #[cfg(target_os = "macos")]
    if let Some(s) = sysctl_str("machdep.cpu.brand_string") {
        return s;
    }
    #[cfg(target_os = "linux")]
    if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(l) = s.lines().find(|l| l.starts_with("model name")) {
            if let Some((_, v)) = l.split_once(':') {
                return v.trim().to_string();
            }
        }
    }
    format!("{} cpu", std::env::consts::ARCH)
}

fn system_memory() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut v: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = c"hw.memsize";
        let rc = unsafe { libc::sysctlbyname(name.as_ptr(), &mut v as *mut u64 as *mut _, &mut len, std::ptr::null_mut(), 0) };
        if rc == 0 {
            return v;
        }
    }
    #[cfg(target_os = "linux")]
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for key in ["MemAvailable:", "MemTotal:"] {
            if let Some(kb) = s.lines().find_map(|l| l.strip_prefix(key)).and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok()) {
                return kb * 1024;
            }
        }
    }
    0
}

#[cfg(target_os = "macos")]
fn sysctl_str(name: &str) -> Option<String> {
    let c = std::ffi::CString::new(name).ok()?;
    let mut len = 0usize;
    unsafe {
        if libc::sysctlbyname(c.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) != 0 || len == 0 {
            return None;
        }
        let mut buf = vec![0u8; len];
        if libc::sysctlbyname(c.as_ptr(), buf.as_mut_ptr() as *mut _, &mut len, std::ptr::null_mut(), 0) != 0 {
            return None;
        }
        buf.truncate(buf.iter().position(|&b| b == 0).unwrap_or(len));
        String::from_utf8(buf).ok()
    }
}
