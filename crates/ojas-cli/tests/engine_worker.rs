//! `ojas engine-worker` against a fake node: the test listens, spawns the built
//! binary, and speaks `ipc` frames to it. The model is a tiny random-weight qwen3
//! written at run time (`support/tiny_gguf.rs`), served on the CPU.
//!
//! The real-hardware comparison (Metal against CPU on a real checkpoint) is ignored
//! by default:
//!
//! ```text
//! OJAS_TEST_GGUF=~/models/Qwen3-0.6B-Q8_0.gguf cargo test --release -p ojas-cli \
//!     --test engine_worker -- --ignored --nocapture metal_and_cpu
//! ```

#[path = "support/tiny_gguf.rs"]
mod tiny_gguf;

use ojas_infer::EngineCore;
use ojas_swarm_proto::frame::{self, Limits};
use ojas_swarm_proto::ipc::{NodeMsg, WorkerMsg};
use ojas_swarm_proto::*;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(120);

struct Dir(PathBuf);
impl Dir {
    fn new() -> Dir {
        static N: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!("ojas-ew-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
    fn join(&self, s: &str) -> PathBuf {
        self.0.join(s)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A fake node with one worker attached.
struct Node {
    child: Option<Child>,
    s: TcpStream,
    caps: WorkerCaps,
    next_req: u64,
}

impl Node {
    fn start(device: &str, cache: &Path) -> Node {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_ojas"))
            .args(["engine-worker", "--socket", &format!("tcp:{addr}"), "--device", device])
            .env("OJAS_CACHE_DIR", cache)
            .env("OJAS_CPU_THREADS", "2")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let s = accept(&l, TIMEOUT);
        let mut n = Node { child: Some(child), s, caps: dummy_caps(), next_req: 1 };
        match n.recv() {
            (WorkerMsg::Ready(c), _) => n.caps = c,
            (m, _) => panic!("expected Ready, got {m:?}"),
        }
        n
    }

    fn send(&mut self, m: &NodeMsg, payload: &[u8]) {
        frame::write(&mut self.s, m, payload, Limits::IPC).unwrap();
    }

    fn recv(&mut self) -> (WorkerMsg, Vec<u8>) {
        frame::read::<_, WorkerMsg>(&mut self.s, Limits::IPC).unwrap()
    }

    /// The next message that is not `Progress`.
    fn reply(&mut self) -> (WorkerMsg, Vec<u8>) {
        loop {
            match self.recv() {
                (WorkerMsg::Progress { .. }, _) => continue,
                r => return r,
            }
        }
    }

    fn req(&mut self) -> ReqId {
        self.next_req += 1;
        ReqId(self.next_req)
    }

    fn load(&mut self, path: &Path, context: u32, expect: Option<ModelId>) -> Result<ModelInfo, String> {
        self.send(&NodeMsg::Load { path: path.to_string_lossy().into(), context, expect }, &[]);
        match self.reply() {
            (WorkerMsg::Loaded(i), _) => Ok(i),
            (WorkerMsg::Error { message, fatal: false, .. }, _) => Err(message),
            (m, _) => panic!("unexpected reply to Load: {m:?}"),
        }
    }

    fn start_generate(&mut self, model: ModelId, prompt: &[u32], max_tokens: u32, want_text: bool) -> ReqId {
        let req = self.req();
        let g = GenerateReq { req, model, prompt: prompt.to_vec(), max_tokens, sampling: Sampling::greedy(), want_text };
        self.send(&NodeMsg::Generate(g), &[]);
        req
    }

    /// Tokens, their text, the finish reason, and each token's arrival time.
    fn generate(&mut self, model: ModelId, prompt: &[u32], max_tokens: u32) -> Gen {
        let req = self.start_generate(model, prompt, max_tokens, true);
        self.collect(req, None)
    }

    /// Read one generation to its end; after `cancel_after` tokens, send `Cancel`.
    fn collect(&mut self, req: ReqId, cancel_after: Option<usize>) -> Gen {
        let mut g = Gen { tokens: vec![], text: String::new(), finish: Finish::Stop, at: vec![] };
        loop {
            match self.reply() {
                (WorkerMsg::Token { req: r, token, text }, _) => {
                    assert_eq!(r, req);
                    g.tokens.push(token);
                    g.text.push_str(text.as_deref().unwrap_or(""));
                    g.at.push(Instant::now());
                    if cancel_after == Some(g.tokens.len()) {
                        self.send(&NodeMsg::Cancel { req }, &[]);
                    }
                }
                (WorkerMsg::Done { req: r, completion_tokens, finish, .. }, _) => {
                    assert_eq!(r, req);
                    assert_eq!(completion_tokens as usize, g.tokens.len(), "Done counts what was streamed");
                    g.finish = finish;
                    return g;
                }
                (m, _) => panic!("unexpected message during generation: {m:?}"),
            }
        }
    }

    fn score(&mut self, model: ModelId, tokens: &[u32], from: u32) -> Result<Vec<f32>, String> {
        let req = self.req();
        self.send(&NodeMsg::Score(ScoreReq { req, model, tokens: tokens.to_vec(), from }), &[]);
        match self.reply() {
            (WorkerMsg::Scores { req: r }, p) => {
                assert_eq!(r, req);
                let (t, used) = Tensor::decode(&p).unwrap();
                assert_eq!(used, p.len());
                Ok(t.data)
            }
            (WorkerMsg::Error { req: r, message, fatal: false }, _) => {
                assert_eq!(r, Some(req));
                Err(message)
            }
            (m, _) => panic!("unexpected reply to Score: {m:?}"),
        }
    }

    fn shutdown(mut self) {
        self.send(&NodeMsg::Shutdown, &[]);
        assert!(matches!(self.reply().0, WorkerMsg::Bye));
        let st = self.wait(TIMEOUT).expect("worker did not exit after Bye");
        assert!(st.success(), "exit status {st:?}");
    }

    fn wait(&mut self, timeout: Duration) -> Option<ExitStatus> {
        let c = self.child.as_mut().unwrap();
        let t0 = Instant::now();
        while t0.elapsed() < timeout {
            if let Some(st) = c.try_wait().unwrap() {
                return Some(st);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        None
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

struct Gen {
    tokens: Vec<u32>,
    text: String,
    finish: Finish,
    at: Vec<Instant>,
}

fn accept(l: &TcpListener, timeout: Duration) -> TcpStream {
    l.set_nonblocking(true).unwrap();
    let t0 = Instant::now();
    loop {
        match l.accept() {
            Ok((s, _)) => {
                s.set_nonblocking(false).unwrap();
                s.set_read_timeout(Some(TIMEOUT)).unwrap();
                s.set_nodelay(true).unwrap();
                return s;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && t0.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(10))
            }
            Err(e) => panic!("worker never connected: {e}"),
        }
    }
}

fn dummy_caps() -> WorkerCaps {
    WorkerCaps { proto: 0, features: 0, backend: Backend::Cpu, device: String::new(), memory_bytes: 0, archs: vec![], trainable: vec![], slots: 0, engine_version: String::new() }
}

/// A tiny model in `dir`, and a worker with its identity cache there too.
fn setup() -> (Dir, PathBuf, Node) {
    let dir = Dir::new();
    let model = dir.join("tiny.gguf");
    tiny_gguf::write(&model, 1);
    let node = Node::start("cpu", &dir.join("cache"));
    (dir, model, node)
}

fn prompt() -> Vec<u32> {
    b"the swarm says".iter().map(|&b| b as u32).collect()
}

#[test]
fn ready_reports_the_device_and_shutdown_says_bye() {
    let (_d, _m, node) = setup();
    let c = &node.caps;
    assert_eq!(c.proto, PROTO_VERSION);
    assert_eq!(c.backend, Backend::Cpu);
    assert_ne!(c.features & features::GENERATE, 0);
    assert_ne!(c.features & features::SCORE, 0);
    assert!(c.archs.iter().any(|a| a == "qwen3"), "{:?}", c.archs);
    assert!(c.trainable.iter().any(|t| t == "tiny_gpt"));
    assert!(!c.device.is_empty() && !c.engine_version.is_empty());
    assert!(c.slots >= 1);
    node.shutdown();
}

#[test]
fn identity_is_stable_tracks_content_and_gates_loading() {
    let (d, model, mut node) = setup();
    let a = node.load(&model, 256, None).unwrap();
    assert_eq!(a.arch, "qwen3");
    assert_eq!((a.n_layers, a.hidden_dim, a.vocab), (2, 64, 257));
    assert_eq!(a.backend, Backend::Cpu);
    assert_eq!(a.file_bytes, std::fs::metadata(&model).unwrap().len());
    assert!(a.eog.contains(&tiny_gguf::EOS));
    // Second load comes from the cache; a fresh worker with an empty cache re-hashes.
    assert_eq!(node.load(&model, 256, None).unwrap().identity, a.identity);
    let mut fresh = Node::start("cpu", &d.join("cache2"));
    assert_eq!(fresh.load(&model, 256, None).unwrap().identity, a.identity, "cache and re-hash agree");
    fresh.shutdown();

    // One weight byte changed: same layout, different content.
    let edited = d.join("edited.gguf");
    let w = tiny_gguf::write(&edited, 1);
    let mut bytes = std::fs::read(&edited).unwrap();
    bytes[w.offsets["blk.1.ffn_up.weight"]] ^= 1;
    std::fs::write(&edited, bytes).unwrap();
    let b = node.load(&edited, 256, None).unwrap();
    assert_eq!(b.identity.arch, a.identity.arch);
    assert_ne!(b.identity.model, a.identity.model);

    // `expect` refuses the wrong content and keeps serving what was loaded.
    let e = node.load(&model, 256, Some(b.identity.model)).unwrap_err();
    assert!(e.contains("expected"), "{e}");
    assert_eq!(node.generate(b.identity.model, &prompt(), 4).tokens.len(), 4, "still serving the edited model");
    // A request for a model that is not loaded is refused.
    let req = node.start_generate(a.identity.model, &prompt(), 4, false);
    match node.reply().0 {
        WorkerMsg::Error { req: r, fatal: false, message } => {
            assert_eq!(r, Some(req));
            assert!(message.contains("not loaded"), "{message}");
        }
        m => panic!("expected a refusal, got {m:?}"),
    }
    assert_eq!(node.load(&model, 256, Some(a.identity.model)).unwrap().identity, a.identity);
    node.shutdown();
}

#[test]
fn greedy_generation_is_deterministic_and_matches_the_engine() {
    let (_d, model, mut node) = setup();
    let info = node.load(&model, 512, None).unwrap();
    let p = prompt();
    let a = node.generate(info.identity.model, &p, 40);
    let b = node.generate(info.identity.model, &p, 40);
    assert_eq!(a.finish, Finish::Length);
    assert_eq!(a.tokens.len(), 40);
    assert_eq!(a.tokens, b.tokens, "greedy is deterministic across requests");

    // The same model run in this process through EngineCore directly.
    let mut g = ojas_formats::gguf::Gguf::open(model.to_str().unwrap()).unwrap();
    let m = ojas_cpu::CpuQwen::load(&mut g).unwrap();
    let mut core = EngineCore::new(&m);
    core.eos = g.meta_u32("tokenizer.ggml.eos_token_id");
    core.eog = ojas_tokenize::eog_token_ids(&g, "qwen3");
    assert_eq!(core.generate(&p, 40, true), a.tokens);

    // Byte-level vocabulary: the streamed text is the tokens' bytes.
    let bytes: Vec<u8> = a.tokens.iter().map(|&t| t as u8).collect();
    let want = String::from_utf8_lossy(&bytes);
    assert_eq!(a.text.trim_end_matches('\u{fffd}'), want.trim_end_matches('\u{fffd}'));

    // A prompt that does not fit is refused, not clipped.
    let long: Vec<u32> = (0..600).map(|i| (i % 200) as u32).collect();
    assert_eq!(node.generate(info.identity.model, &long, 4).finish, Finish::Refused);
    node.shutdown();
}

#[test]
fn cancel_stops_a_generation_mid_stream() {
    let (_d, model, mut node) = setup();
    let info = node.load(&model, 1 << 20, None).unwrap();
    let max = 500_000;
    let req = node.start_generate(info.identity.model, &prompt(), max, false);
    let t0 = Instant::now();
    let g = node.collect(req, Some(16));
    assert_eq!(g.finish, Finish::Cancelled);
    assert!(g.tokens.len() >= 16 && g.tokens.len() < max as usize, "{} tokens", g.tokens.len());
    assert!(t0.elapsed() < Duration::from_secs(60));
    // Cancelling an unknown or finished request is harmless, and serving continues.
    node.send(&NodeMsg::Cancel { req }, &[]);
    node.send(&NodeMsg::Cancel { req: ReqId(999_999) }, &[]);
    assert_eq!(node.generate(info.identity.model, &prompt(), 5).tokens.len(), 5);
    node.shutdown();
}

#[test]
fn scores_are_log_probabilities_consistent_with_greedy() {
    let (_d, model, mut node) = setup();
    let info = node.load(&model, 512, None).unwrap();
    let id = info.identity.model;
    let p = prompt();
    let g = node.generate(id, &p, 6).tokens;
    let mut seq = p.clone();
    seq.extend(&g);
    let from = p.len() as u32;
    let lp = node.score(id, &seq, from).unwrap();
    assert_eq!(lp.len(), g.len());
    assert!(lp.iter().all(|v| v.is_finite() && *v <= 0.0), "{lp:?}");

    // At each generated position, every alternative: the greedy token has the largest
    // log-probability, the distribution sums to one, and the batch score agrees.
    for i in 0..g.len() {
        let ctx = &seq[..p.len() + i];
        let all: Vec<f32> = (0..info.vocab)
            .map(|t| {
                let mut s = ctx.to_vec();
                s.push(t);
                let r = node.score(id, &s, ctx.len() as u32).unwrap();
                assert_eq!(r.len(), 1);
                r[0]
            })
            .collect();
        let best = (0..all.len()).max_by(|&a, &b| all[a].total_cmp(&all[b])).unwrap();
        assert_eq!(best as u32, g[i], "position {i}: argmax of the scores is the greedy token");
        let mass: f64 = all.iter().map(|&l| (l as f64).exp()).sum();
        assert!((mass - 1.0).abs() < 1e-3, "position {i}: probabilities sum to {mass}");
        assert!((all[g[i] as usize] - lp[i]).abs() < 1e-4, "position {i}: {} vs {}", all[g[i] as usize], lp[i]);
    }

    // Requests that cannot be scored are refused without killing the worker.
    assert!(node.score(id, &seq, 0).is_err());
    assert!(node.score(id, &seq, seq.len() as u32).is_err());
    assert!(node.score(id, &[1, 2, 9999], 1).is_err());
    node.shutdown();
}

/// Requests are independent: what one leaves in the model's state must not change
/// the next. Scores a sequence after an unrelated generation and compares with a
/// worker that has served nothing.
#[test]
fn requests_do_not_leak_state_into_each_other() {
    let (d, model, mut node) = setup();
    let id = node.load(&model, 512, None).unwrap().identity.model;
    let other: Vec<u32> = b"something else entirely".iter().map(|&b| b as u32).collect();
    node.generate(id, &other, 16);
    let mut seq = prompt();
    seq.extend(b" and listens".iter().map(|&b| b as u32));
    let after = node.score(id, &seq, 4).unwrap();
    let mut fresh = Node::start("cpu", &d.join("cache"));
    fresh.load(&model, 512, None).unwrap();
    assert_eq!(after, fresh.score(id, &seq, 4).unwrap());
    let g1 = node.generate(id, &prompt(), 16).tokens;
    assert_eq!(g1, fresh.generate(id, &prompt(), 16).tokens);
    fresh.shutdown();
    node.shutdown();
}

#[test]
fn an_unknown_message_is_skipped_and_the_next_one_answered() {
    let (_d, model, mut node) = setup();
    frame::write(&mut node.s, &serde_json::json!({"t": "FromTheFuture", "x": [1, 2, 3]}), b"some payload", Limits::IPC).unwrap();
    let info = node.load(&model, 256, None).unwrap();
    frame::write(&mut node.s, &serde_json::json!({"t": "AlsoNew"}), &[], Limits::IPC).unwrap();
    assert_eq!(node.generate(info.identity.model, &prompt(), 3).tokens.len(), 3);
    node.shutdown();
}

#[test]
fn a_garbage_frame_ends_the_worker_with_an_error() {
    let dir = Dir::new();
    // A header length past the IPC limit.
    let mut node = Node::start("cpu", &dir.join("cache"));
    node.s.write_all(&u32::MAX.to_le_bytes()).unwrap();
    let st = node.wait(Duration::from_secs(30)).expect("worker hung on an oversized frame");
    assert!(!st.success(), "{st:?}");
    // A known tag that does not decode.
    let mut node = Node::start("cpu", &dir.join("cache"));
    frame::write(&mut node.s, &serde_json::json!({"t": "Load", "path": 5}), &[], Limits::IPC).unwrap();
    let st = node.wait(Duration::from_secs(30)).expect("worker hung on a malformed frame");
    assert!(!st.success(), "{st:?}");
    // Not JSON at all.
    let mut node = Node::start("cpu", &dir.join("cache"));
    node.s.write_all(&[4, 0, 0, 0, 0xff, 0xfe, 0, 1, 0, 0, 0, 0]).unwrap();
    let st = node.wait(Duration::from_secs(30)).expect("worker hung on a non-JSON header");
    assert!(!st.success(), "{st:?}");
}

#[test]
fn eof_from_the_node_exits_cleanly() {
    let dir = Dir::new();
    let mut node = Node::start("cpu", &dir.join("cache"));
    node.s.shutdown(std::net::Shutdown::Both).unwrap();
    let st = node.wait(Duration::from_secs(30)).expect("worker did not exit on EOF");
    assert!(st.success(), "{st:?}");
}

#[cfg(unix)]
#[test]
fn connects_over_a_unix_socket() {
    let dir = Dir::new();
    let sock = dir.join("w.sock");
    let l = std::os::unix::net::UnixListener::bind(&sock).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_ojas"))
        .args(["engine-worker", "--socket", &format!("unix:{}", sock.display()), "--device", "cpu"])
        .env("OJAS_CACHE_DIR", dir.join("cache"))
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let (mut s, _) = l.accept().unwrap();
    s.set_read_timeout(Some(TIMEOUT)).unwrap();
    assert!(matches!(frame::read::<_, WorkerMsg>(&mut s, Limits::IPC).unwrap().0, WorkerMsg::Ready(_)));
    frame::write(&mut s, &NodeMsg::Shutdown, &[], Limits::IPC).unwrap();
    assert!(matches!(frame::read::<_, WorkerMsg>(&mut s, Limits::IPC).unwrap().0, WorkerMsg::Bye));
    assert!(child.wait().unwrap().success());
}

fn train_spec() -> TrainSpec {
    let cfg = TinyGptConfig { vocab: 256, ctx: 32, d_model: 32, n_layers: 1, n_heads: 2, d_ff: 64, init_seed: 7 };
    TrainSpec {
        run: "test-run".into(),
        model: TrainModel::TinyGpt(cfg),
        arch: ArchId([0; 32]),
        data: DataRef { blob: BlobId([0; 32]), bytes: 0 },
        shard: 0,
        n_shards: 1,
        inner_steps: 2,
        batch: 2,
        inner_lr: 1e-3,
        weight_decay: 0.0,
    }
}

/// Whatever `ojas_swarm::train::begin` does, a training failure is not a worker
/// failure: serving goes on.
#[test]
fn training_errors_are_not_fatal() {
    let (_d, model, mut node) = setup();
    let info = node.load(&model, 256, None).unwrap();
    node.send(&NodeMsg::TrainBegin { spec: train_spec() }, &[]);
    match node.reply().0 {
        WorkerMsg::TrainReady { run, n_params, .. } => {
            assert_eq!(run, "test-run");
            assert!(n_params > 0);
        }
        WorkerMsg::Error { fatal: false, .. } => {}
        m => panic!("unexpected reply to TrainBegin: {m:?}"),
    }
    // A round whose data file does not exist fails on its own.
    let round = RoundSpec { round: 0, base: ModelId([0; 32]), cursor: 0 };
    node.send(&NodeMsg::TrainRound { round, data_path: "/nonexistent/shard.bin".into() }, &[]);
    assert!(matches!(node.reply().0, WorkerMsg::Error { fatal: false, .. }));
    node.send(&NodeMsg::TrainEnd { run: "test-run".into() }, &[]);
    assert_eq!(node.generate(info.identity.model, &prompt(), 3).tokens.len(), 3);
    node.shutdown();
}

/// A whole round on the CPU: the coordinator's θ0 in, a delta of the same length out,
/// from the base the coordinator named, to weights with the same layout.
#[test]
fn a_training_round_returns_a_delta() {
    let dir = Dir::new();
    let text: Vec<u8> = b"the quick brown fox jumps over the lazy dog. ".iter().cycle().take(4096).copied().collect();
    let data = ojas_swarm::train::data::from_bytes(&text);
    let mut spec = train_spec();
    let cfg = ojas_swarm::coord::RunConfig {
        run: spec.run.clone(),
        model: spec.model.clone(),
        data: dir.join("unused"),
        heldout: None,
        inner_steps: spec.inner_steps,
        batch: spec.batch,
        inner_lr: spec.inner_lr,
        weight_decay: spec.weight_decay,
        outer_lr: 0.7,
        outer_momentum: 0.9,
        rounds: 1,
        min_members: 1,
        max_members: 1,
        round_deadline_secs: 60,
        checkpoint_dir: dir.join("ckpt"),
        eval_seqs: 1,
    };
    let coord = ojas_swarm::coord::Coordinator::new(cfg, data.clone(), None).unwrap();
    spec.arch = coord.arch();
    spec.data = coord.data_ref().clone();
    let theta0 = Tensor::vector(coord.theta().to_vec()).encode(DType::F32);
    let shard = dir.join("shard.bin");
    std::fs::write(&shard, &data).unwrap();

    let mut node = Node::start("cpu", &dir.join("cache"));
    node.send(&NodeMsg::TrainBegin { spec }, &theta0);
    let (identity, n) = match node.reply().0 {
        WorkerMsg::TrainReady { identity, n_params, .. } => (identity, n_params as usize),
        m => panic!("TrainBegin: {m:?}"),
    };
    assert_eq!(identity.model, coord.base(), "θ0 hashes to the coordinator's base");
    assert_eq!(n, coord.theta().len());
    let round = RoundSpec { round: 0, base: identity.model, cursor: 0 };
    node.send(&NodeMsg::TrainRound { round, data_path: shard.to_string_lossy().into() }, &[]);
    let after = match node.reply() {
        (WorkerMsg::Delta { report, identity: after }, p) => {
            assert_eq!(report.base, identity.model);
            assert_eq!(report.backend, Backend::Cpu);
            assert!(report.final_loss.is_finite() && report.tokens > 0);
            assert_ne!(after.model, identity.model, "the weights moved");
            assert_eq!(after.arch, identity.arch);
            let d = Tensor::decode(&p).unwrap().0.data;
            assert_eq!(d.len(), n);
            assert!(d.iter().any(|v| *v != 0.0));
            after
        }
        (m, _) => panic!("TrainRound: {m:?}"),
    };
    // A round whose base is not the current weights is refused; resetting θ with a
    // payload makes the original base valid again.
    let stale = RoundSpec { round: 1, base: identity.model, cursor: 4 };
    node.send(&NodeMsg::TrainRound { round: stale.clone(), data_path: shard.to_string_lossy().into() }, &[]);
    assert!(matches!(node.reply().0, WorkerMsg::Error { fatal: false, .. }), "weights are at {}", after.model.short());
    node.send(&NodeMsg::TrainRound { round: stale, data_path: shard.to_string_lossy().into() }, &theta0);
    assert!(matches!(node.reply().0, WorkerMsg::Delta { .. }));
    node.send(&NodeMsg::TrainEnd { run: "test-run".into() }, &[]);
    node.shutdown();
}

/// Metal against CPU on a real checkpoint: greedy outputs and teacher-forced scores.
#[test]
#[ignore = "needs OJAS_TEST_GGUF and, for the Metal half, an Apple GPU"]
fn metal_and_cpu_agree_on_a_real_model() {
    let Ok(path) = std::env::var("OJAS_TEST_GGUF") else {
        eprintln!("OJAS_TEST_GGUF not set; skipping");
        return;
    };
    let g = ojas_formats::gguf::Gguf::open(&path).unwrap();
    let bpe = ojas_tokenize::Bpe::from_gguf(&g);
    let text = ojas_tokenize::chat_template(&g.arch(), "Name three primary colours.");
    let p: Vec<u32> = bpe.encode(&text).into_iter().map(|t| t as u32).collect();
    let n = std::env::var("OJAS_TEST_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(48u32);
    let dir = Dir::new();
    let devices: Vec<&str> = if cfg!(target_os = "macos") { vec!["metal", "cpu"] } else { vec!["cpu"] };
    let mut runs = Vec::new();
    for dev in devices {
        let mut node = Node::start(dev, &dir.join("cache"));
        let t0 = Instant::now();
        let info = node.load(Path::new(&path), 2048, None).unwrap();
        let load = t0.elapsed();
        let gen = node.generate(info.identity.model, &p, n);
        let rate = if gen.at.len() > 1 {
            (gen.at.len() - 1) as f64 / gen.at.last().unwrap().duration_since(gen.at[0]).as_secs_f64()
        } else {
            0.0
        };
        eprintln!(
            "{dev}: {} on {} | load {:.1}s | {} tokens ({:?}) | {rate:.1} tok/s decode\n  {:?}",
            info.identity.model.short(), node.caps.device, load.as_secs_f64(), gen.tokens.len(), gen.finish, gen.text
        );
        runs.push((dev, node, info, gen));
    }
    if runs.len() < 2 {
        return;
    }
    assert_eq!(runs[0].2.identity, runs[1].2.identity, "both devices hash the file the same");
    let (a, b) = (&runs[0].3.tokens, &runs[1].3.tokens);
    let same = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    eprintln!("greedy: {same} of {} leading tokens agree (metal vs cpu)", a.len().min(b.len()));
    // Score the Metal continuation on both devices.
    let mut seq = p.clone();
    seq.extend(a);
    let id = runs[0].2.identity.model;
    // Reload first: the CPU decoder's KV is append-only (see
    // `requests_do_not_leak_state_into_each_other`), so a score after a generation
    // with another continuation would read the generation's rows.
    for (_, node, _, _) in runs.iter_mut() {
        node.load(Path::new(&path), 2048, None).unwrap();
    }
    let t0 = Instant::now();
    let sm = runs[0].1.score(id, &seq, p.len() as u32).unwrap();
    let tm = t0.elapsed();
    let t0 = Instant::now();
    let sc = runs[1].1.score(id, &seq, p.len() as u32).unwrap();
    let tc = t0.elapsed();
    let d: Vec<f32> = sm.iter().zip(&sc).map(|(x, y)| (x - y).abs()).collect();
    let max = d.iter().copied().fold(0.0f32, f32::max);
    let mean = d.iter().sum::<f32>() / d.len() as f32;
    eprintln!(
        "score {} positions: metal {:.2}s, cpu {:.2}s | |Δ logprob| max {max:.4} mean {mean:.4} | mean logprob metal {:.3} cpu {:.3}",
        d.len(), tm.as_secs_f64(), tc.as_secs_f64(),
        sm.iter().sum::<f32>() / sm.len() as f32, sc.iter().sum::<f32>() / sc.len() as f32
    );
    for (_, node, _, _) in runs {
        node.shutdown();
    }
}
