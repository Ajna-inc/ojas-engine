//! Types shared by the IPC and the peer protocol.

use crate::identity::{ArchId, BlobId, Identity, ModelId};
use serde::{Deserialize, Serialize};

/// A request id, unique within the node that issued it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReqId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Metal,
    Cuda,
    Cpu,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
            Backend::Cpu => "cpu",
        }
    }
}

/// What a worker can do, reported once at start.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkerCaps {
    pub proto: u32,
    pub features: u64,
    pub backend: Backend,
    /// "Apple M2 Max", "NVIDIA GeForce RTX 3060", CPU brand.
    pub device: String,
    /// Device memory available for models: VRAM, or unified/system RAM.
    pub memory_bytes: u64,
    /// GGUF architectures this worker can serve.
    pub archs: Vec<String>,
    /// Model kinds this worker can train (see [`TrainModel`]).
    pub trainable: Vec<String>,
    /// Concurrent generations the loaded model supports.
    pub slots: u32,
    pub engine_version: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    /// <= 0 is greedy.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl Sampling {
    pub fn greedy() -> Sampling {
        Sampling { temperature: 0.0, top_p: 1.0, top_k: 0, repeat_penalty: 1.0, seed: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Finish {
    /// End-of-generation token.
    Stop,
    /// `max_tokens` or the context ran out.
    Length,
    Cancelled,
    /// The device faulted; tokens after the fault were not sent.
    Fault,
    /// The model cannot run this request (prompt longer than the context, ...).
    Refused,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerateReq {
    pub req: ReqId,
    pub model: ModelId,
    /// Token ids, already templated. Tokenising on the requester keeps the worker
    /// and every peer agnostic of chat templates.
    pub prompt: Vec<u32>,
    pub max_tokens: u32,
    pub sampling: Sampling,
    /// Send each token's text as well as its id.
    #[serde(default)]
    pub want_text: bool,
}

/// Teacher-forced scoring: log-probability of each token given those before it, in
/// one prefill. Used to compare backends and to check another node's output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreReq {
    pub req: ReqId,
    pub model: ModelId,
    pub tokens: Vec<u32>,
    /// Score positions `from..tokens.len()`; earlier ones are context only.
    pub from: u32,
}

/// Facts about a loaded model a node needs for routing and templating.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub identity: Identity,
    pub arch: String,
    pub n_layers: u32,
    pub hidden_dim: u32,
    pub vocab: u32,
    pub context: u32,
    pub eog: Vec<u32>,
    pub file_bytes: u64,
    pub backend: Backend,
    pub load_ms: u32,
}

// ------------------------------------------------------------------ training

/// A model a DiLoCo run trains. Every member must build the same one: the run's
/// `ArchId` is checked at join and on every delta.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrainModel {
    /// Small causal transformer on the `ojas-learn` tape: trains on CPU, CUDA and Metal.
    TinyGpt(TinyGptConfig),
}

impl TrainModel {
    pub fn kind(&self) -> &'static str {
        match self {
            TrainModel::TinyGpt(_) => "tiny_gpt",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TinyGptConfig {
    pub vocab: u32,
    pub ctx: u32,
    pub d_model: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub d_ff: u32,
    /// Seed for the initial weights. Every member starts from the same θ0 because the
    /// coordinator sends it; the seed only makes θ0 reproducible.
    pub init_seed: u64,
}

impl TinyGptConfig {
    /// A model small enough to train on a laptop CPU in minutes.
    pub fn small(vocab: u32) -> TinyGptConfig {
        TinyGptConfig { vocab, ctx: 128, d_model: 128, n_layers: 4, n_heads: 4, d_ff: 512, init_seed: 1 }
    }
}

/// Training data: a token file, either local to the member or a blob the
/// coordinator serves.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataRef {
    pub blob: BlobId,
    pub bytes: u64,
}

/// Everything a member needs to run inner steps. Fixed for a run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainSpec {
    pub run: String,
    pub model: TrainModel,
    pub arch: ArchId,
    pub data: DataRef,
    /// This member's stride through the data: sequences `shard, shard + n_shards, ...`.
    pub shard: u32,
    pub n_shards: u32,
    pub inner_steps: u32,
    pub batch: u32,
    pub inner_lr: f32,
    pub weight_decay: f32,
}

/// One round of inner steps from `base`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoundSpec {
    pub round: u32,
    pub base: ModelId,
    /// Sequence offset to resume the shard's stride from, so no member repeats data
    /// across rounds and a restarted member continues where it left off.
    pub cursor: u64,
}

/// What a member sends back with its delta (`theta_start - theta_end`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeltaReport {
    pub run: String,
    pub round: u32,
    pub base: ModelId,
    pub arch: ArchId,
    /// Sequence indices actually used, as `[first, first + stride * count)`.
    pub first_seq: u64,
    pub stride: u64,
    pub seqs: u64,
    pub tokens: u64,
    pub mean_loss: f32,
    pub final_loss: f32,
    pub backend: Backend,
    pub wall_ms: u32,
}
