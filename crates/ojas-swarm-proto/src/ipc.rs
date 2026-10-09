//! Node <-> engine-worker messages, framed by [`crate::frame`] with [`Limits::IPC`].
//!
//! One worker process per device. The node starts it with
//! `ojas engine-worker --socket <path> --device <metal|cuda|cpu>`, the worker connects,
//! sends [`WorkerMsg::Ready`], and then answers requests. The worker never opens a
//! network socket. Killing it is how a node frees device memory or recovers from a
//! fault; [`WorkerMsg::Error`] with `fatal` asks for exactly that.
//!
//! The message set follows SwarmLLM's `DaemonMsg`/`WorkerMsg` (MIT OR Apache-2.0,
//! `src/inference/worker_ipc.rs` @ b14482d), cut to what ojas serves today and
//! extended with model loading by identity, scoring and DiLoCo training.
//!
//! [`Limits::IPC`]: crate::frame::Limits::IPC

use crate::frame::Tagged;
use crate::identity::{Identity, ModelId};
use crate::types::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t")]
pub enum NodeMsg {
    /// Load a GGUF for serving. With `expect`, the worker refuses a file whose
    /// content does not hash to it.
    Load { path: String, context: u32, expect: Option<ModelId> },
    Unload { model: ModelId },
    /// Streamed back as `Token`... then `Done`.
    Generate(GenerateReq),
    /// Answered by `Scores`; payload = f32 tensor `[tokens.len() - from]` of log-probs.
    Score(ScoreReq),
    /// Best-effort, idempotent. An unknown id is ignored.
    Cancel { req: ReqId },
    /// Build the training model for a run. Payload = θ0 tensor (f32 vector); empty means
    /// initialise from the config's seed. Answered by `TrainReady`.
    TrainBegin { spec: TrainSpec },
    /// Run one round of inner steps from θ = payload (f32 vector, or empty to continue
    /// from the current weights if they already hash to `round.base`). The data shard is
    /// the file at `data_path`, which the node has fetched and verified. Answered by `Delta`.
    TrainRound { round: RoundSpec, data_path: String },
    TrainEnd { run: String },
    Shutdown,
}

impl Tagged for NodeMsg {
    const TAGS: &'static [&'static str] =
        &["Load", "Unload", "Generate", "Score", "Cancel", "TrainBegin", "TrainRound", "TrainEnd", "Shutdown"];
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t")]
pub enum WorkerMsg {
    Ready(WorkerCaps),
    Loaded(ModelInfo),
    Token { req: ReqId, token: u32, #[serde(default, skip_serializing_if = "Option::is_none")] text: Option<String> },
    Done { req: ReqId, prompt_tokens: u32, completion_tokens: u32, finish: Finish },
    Scores { req: ReqId },
    TrainReady { run: String, identity: Identity, n_params: u64 },
    /// Payload = delta tensor (f32 vector, `theta_start - theta_end`). `identity` is
    /// the worker's weights *after* the round, before the outer step.
    Delta { report: DeltaReport, identity: Identity },
    Progress { req: Option<ReqId>, phase: Phase, done: u64, total: u64 },
    /// `fatal`: device state is corrupt or exhausted; kill and restart this worker.
    Error { req: Option<ReqId>, message: String, fatal: bool },
    Bye,
}

impl Tagged for WorkerMsg {
    const TAGS: &'static [&'static str] =
        &["Ready", "Loaded", "Token", "Done", "Scores", "TrainReady", "Delta", "Progress", "Error", "Bye"];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Hashing,
    Loading,
    Prefill,
    Training,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{self, Limits};
    use crate::identity::{ArchId, BlobId};

    fn id() -> Identity {
        Identity { arch: ArchId([1; 32]), model: ModelId([2; 32]) }
    }

    /// Every variant, so a variant added without its tag fails here.
    fn all_node() -> Vec<NodeMsg> {
        let spec = TrainSpec {
            run: "r".into(),
            model: TrainModel::TinyGpt(TinyGptConfig::small(256)),
            arch: ArchId([1; 32]),
            data: DataRef { blob: BlobId([3; 32]), bytes: 10 },
            shard: 0,
            n_shards: 2,
            inner_steps: 10,
            batch: 4,
            inner_lr: 1e-3,
            weight_decay: 0.0,
        };
        vec![
            NodeMsg::Load { path: "m.gguf".into(), context: 2048, expect: None },
            NodeMsg::Unload { model: ModelId([2; 32]) },
            NodeMsg::Generate(GenerateReq { req: ReqId(1), model: ModelId([2; 32]), prompt: vec![1, 2], max_tokens: 4, sampling: Sampling::greedy(), want_text: true }),
            NodeMsg::Score(ScoreReq { req: ReqId(2), model: ModelId([2; 32]), tokens: vec![1, 2, 3], from: 1 }),
            NodeMsg::Cancel { req: ReqId(1) },
            NodeMsg::TrainBegin { spec },
            NodeMsg::TrainRound { round: RoundSpec { round: 0, base: ModelId([2; 32]), cursor: 0 }, data_path: "d".into() },
            NodeMsg::TrainEnd { run: "r".into() },
            NodeMsg::Shutdown,
        ]
    }

    fn all_worker() -> Vec<WorkerMsg> {
        let caps = WorkerCaps { proto: 1, features: 0, backend: Backend::Cpu, device: "cpu".into(), memory_bytes: 1, archs: vec![], trainable: vec![], slots: 1, engine_version: "0".into() };
        let report = DeltaReport { run: "r".into(), round: 0, base: ModelId([2; 32]), arch: ArchId([1; 32]), first_seq: 0, stride: 2, seqs: 5, tokens: 640, mean_loss: 3.0, final_loss: 2.5, backend: Backend::Cpu, wall_ms: 10 };
        vec![
            WorkerMsg::Ready(caps),
            WorkerMsg::Loaded(ModelInfo { identity: id(), arch: "qwen3".into(), n_layers: 28, hidden_dim: 1024, vocab: 151936, context: 2048, eog: vec![1], file_bytes: 9, backend: Backend::Cpu, load_ms: 5 }),
            WorkerMsg::Token { req: ReqId(1), token: 5, text: Some("hi".into()) },
            WorkerMsg::Done { req: ReqId(1), prompt_tokens: 2, completion_tokens: 1, finish: Finish::Stop },
            WorkerMsg::Scores { req: ReqId(2) },
            WorkerMsg::TrainReady { run: "r".into(), identity: id(), n_params: 100 },
            WorkerMsg::Delta { report, identity: id() },
            WorkerMsg::Progress { req: None, phase: Phase::Loading, done: 1, total: 2 },
            WorkerMsg::Error { req: None, message: "x".into(), fatal: true },
            WorkerMsg::Bye,
        ]
    }

    #[test]
    fn every_message_round_trips_and_has_a_tag() {
        let node = all_node();
        assert_eq!(node.len(), NodeMsg::TAGS.len());
        for m in node {
            let mut b = Vec::new();
            frame::write(&mut b, &m, b"p", Limits::IPC).unwrap();
            assert_eq!(frame::read::<_, NodeMsg>(&mut &b[..], Limits::IPC).unwrap().0, m);
            let v = serde_json::to_value(&m).unwrap();
            assert!(NodeMsg::TAGS.contains(&v["t"].as_str().unwrap()));
        }
        let worker = all_worker();
        assert_eq!(worker.len(), WorkerMsg::TAGS.len());
        for m in worker {
            let mut b = Vec::new();
            frame::write(&mut b, &m, &[], Limits::IPC).unwrap();
            assert_eq!(frame::read::<_, WorkerMsg>(&mut &b[..], Limits::IPC).unwrap().0, m);
            let v = serde_json::to_value(&m).unwrap();
            assert!(WorkerMsg::TAGS.contains(&v["t"].as_str().unwrap()));
        }
    }
}
