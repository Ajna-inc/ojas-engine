//! `ojas-node mock-worker`: the engine-worker IPC with no engine behind it, for
//! tests that must run without a GPU or a model.
//!
//! Deterministic: `Load` reports a ModelId that is BLAKE3 of the path, and
//! `Generate` echoes the prompt's tokens (cycled to `max_tokens`), which are valid
//! ids in whatever vocabulary the prompt came from. A training round returns 1% of
//! θ as the delta, with a report covering exactly the stride the coordinator
//! assigned. Faults are injected by environment:
//!
//! * `OJAS_MOCK_CRASH_AT_REQ=N` — exit before the first token of the Nth Generate;
//! * `OJAS_MOCK_FATAL_AT_REQ=N` — answer the Nth Generate with a fatal error;
//! * `OJAS_MOCK_CRASH_AFTER=N` — exit after N tokens in total;
//! * `OJAS_MOCK_SLOW_MS=N` — sleep N ms before each token.

use crate::ipc;
use anyhow::Result;
use ojas_swarm_proto::ipc::{NodeMsg, WorkerMsg};
use ojas_swarm_proto::tensor::{DType, Tensor};
use ojas_swarm_proto::*;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

fn env_u64(k: &str) -> Option<u64> {
    std::env::var(k).ok().and_then(|v| v.parse().ok())
}

fn backend_of(device: &str) -> Backend {
    match device {
        "metal" => Backend::Metal,
        "cuda" => Backend::Cuda,
        _ => Backend::Cpu,
    }
}

pub async fn run(socket: &str, device: &str) -> Result<()> {
    let io = ipc::connect(socket).await?;
    let (mut r, mut w) = tokio::io::split(io);
    let (tx, mut rx) = mpsc::unbounded_channel::<(WorkerMsg, Vec<u8>)>();
    let backend = backend_of(device);
    let caps = WorkerCaps {
        proto: PROTO_VERSION,
        features: features::GENERATE | features::TRAIN_DILOCO,
        backend,
        device: format!("mock-{device}"),
        memory_bytes: 1 << 30,
        archs: vec![],
        trainable: vec!["tiny_gpt".into()],
        slots: 4,
        engine_version: "mock".into(),
    };
    ipc::send(&mut w, &WorkerMsg::Ready(caps), &[]).await?;
    tokio::spawn(async move {
        while let Some((m, p)) = rx.recv().await {
            if ipc::send(&mut w, &m, &p).await.is_err() {
                std::process::exit(0);
            }
        }
    });

    let cancelled = Arc::new(Mutex::new(HashSet::<ReqId>::new()));
    let sent = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut gens = 0u64;
    let mut theta: Vec<f32> = Vec::new();
    let mut spec: Option<TrainSpec> = None;
    let mut skipped = vec![];
    while let Some((m, payload)) = ipc::recv::<_, NodeMsg>(&mut r, &mut skipped).await? {
        match m {
            NodeMsg::Load { path, context, .. } => {
                let model = ModelId(*blake3::hash(path.as_bytes()).as_bytes());
                let arch = ArchId(*blake3::hash(format!("mock-arch:{path}").as_bytes()).as_bytes());
                let (name, vocab) = match ojas_formats::gguf::Gguf::open(&path) {
                    Ok(g) => (g.arch(), g.str_arr("tokenizer.ggml.tokens").map_or(0, |t| t.len() as u32)),
                    Err(_) => ("mock".into(), 0),
                };
                let info = ModelInfo { identity: Identity { arch, model }, arch: name, n_layers: 1, hidden_dim: 8, vocab, context, eog: vec![], file_bytes: 0, backend, load_ms: 1 };
                let _ = tx.send((WorkerMsg::Loaded(info), vec![]));
            }
            NodeMsg::Generate(g) => {
                gens += 1;
                if env_u64("OJAS_MOCK_CRASH_AT_REQ") == Some(gens) {
                    std::process::exit(3);
                }
                if env_u64("OJAS_MOCK_FATAL_AT_REQ") == Some(gens) {
                    let _ = tx.send((WorkerMsg::Error { req: Some(g.req), message: "mock fatal fault".into(), fatal: true }, vec![]));
                    continue;
                }
                let (tx, cancelled, sent) = (tx.clone(), cancelled.clone(), sent.clone());
                tokio::spawn(async move {
                    let slow = env_u64("OJAS_MOCK_SLOW_MS").unwrap_or(0);
                    let crash_after = env_u64("OJAS_MOCK_CRASH_AFTER");
                    let n = if g.prompt.is_empty() { 0 } else { g.max_tokens as usize };
                    let mut finish = Finish::Length;
                    let mut out = 0u32;
                    for i in 0..n {
                        if cancelled.lock().unwrap().remove(&g.req) {
                            finish = Finish::Cancelled;
                            break;
                        }
                        if slow > 0 {
                            tokio::time::sleep(Duration::from_millis(slow)).await;
                        }
                        if crash_after.is_some_and(|c| sent.load(std::sync::atomic::Ordering::Relaxed) >= c) {
                            std::process::exit(4);
                        }
                        let token = g.prompt[i % g.prompt.len()];
                        let _ = tx.send((WorkerMsg::Token { req: g.req, token, text: None }, vec![]));
                        sent.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        out += 1;
                    }
                    let _ = tx.send((WorkerMsg::Done { req: g.req, prompt_tokens: g.prompt.len() as u32, completion_tokens: out, finish }, vec![]));
                });
            }
            NodeMsg::Cancel { req } => {
                cancelled.lock().unwrap().insert(req);
            }
            NodeMsg::TrainBegin { spec: sp } => {
                theta = if payload.is_empty() { vec![0.5; 64] } else { Tensor::decode(&payload).map(|t| t.0.data).unwrap_or_default() };
                let id = identity_of(&theta);
                let run = sp.run.clone();
                spec = Some(sp);
                let _ = tx.send((WorkerMsg::TrainReady { run, identity: id, n_params: theta.len() as u64 }, vec![]));
            }
            NodeMsg::TrainRound { round, data_path } => {
                if !payload.is_empty() {
                    match Tensor::decode(&payload) {
                        Ok((t, _)) => theta = t.data,
                        Err(e) => {
                            let _ = tx.send((WorkerMsg::Error { req: None, message: format!("bad θ: {e}"), fatal: false }, vec![]));
                            continue;
                        }
                    }
                }
                let Some(sp) = &spec else {
                    let _ = tx.send((WorkerMsg::Error { req: None, message: "TrainRound before TrainBegin".into(), fatal: false }, vec![]));
                    continue;
                };
                if std::fs::metadata(&data_path).is_err() {
                    let _ = tx.send((WorkerMsg::Error { req: None, message: format!("no data at {data_path}"), fatal: false }, vec![]));
                    continue;
                }
                // A fixed fraction of θ: deterministic, and visibly applied by an outer step.
                let delta: Vec<f32> = theta.iter().map(|v| v * 0.01).collect();
                for (t, d) in theta.iter_mut().zip(&delta) {
                    *t -= d;
                }
                let TrainModel::TinyGpt(m) = &sp.model;
                // The stride the coordinator assigned: shard `shard` of `n_shards` at the cursor.
                let seqs = sp.inner_steps as u64 * sp.batch as u64;
                let report = DeltaReport {
                    run: sp.run.clone(),
                    round: round.round,
                    base: round.base,
                    arch: sp.arch,
                    first_seq: round.cursor * sp.n_shards as u64 + sp.shard as u64,
                    stride: sp.n_shards as u64,
                    seqs,
                    tokens: seqs * m.ctx as u64,
                    mean_loss: 1.0 / (1.0 + round.round as f32),
                    final_loss: 1.0 / (2.0 + round.round as f32),
                    backend,
                    wall_ms: 1,
                };
                let _ = tx.send((WorkerMsg::Delta { report, identity: identity_of(&theta) }, Tensor::vector(delta).encode(DType::F32)));
            }
            NodeMsg::TrainEnd { .. } | NodeMsg::Unload { .. } | NodeMsg::Score(_) => {}
            NodeMsg::Shutdown => break,
        }
    }
    let _ = tx.send((WorkerMsg::Bye, vec![]));
    tokio::time::sleep(Duration::from_millis(50)).await;
    Ok(())
}

fn identity_of(theta: &[f32]) -> Identity {
    let mut b = IdentityBuilder::new("mock").dim("n", theta.len() as u64);
    let bytes: Vec<u8> = theta.iter().flat_map(|v| v.to_le_bytes()).collect();
    b.tensor("theta", "f32", &[theta.len() as u64], &bytes);
    b.finish()
}
