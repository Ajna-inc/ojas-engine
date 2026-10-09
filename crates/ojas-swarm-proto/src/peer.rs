//! Node <-> node messages over libp2p, framed by [`crate::frame`] with [`Limits::PEER`].
//!
//! * [`PROTO_RPC`]: one request, one response (`request_response`).
//! * [`PROTO_STREAM`]: a per-request stream for token streaming. The opener writes
//!   one [`PeerReq::Generate`], the server writes [`PeerResp::Token`] frames and ends
//!   with [`PeerResp::Done`] or [`PeerResp::Error`]. One persistent stream per request
//!   rather than a request/response per token: SwarmLLM measured ~100 ms/token for the
//!   latter.
//! * [`TOPIC_NODES`]: gossipsub topic carrying [`Announce`].
//!
//! Anything larger than [`Limits::PEER`] moves as blob chunks (`BlobGet`/`BlobPut`),
//! addressed by BLAKE3 and verified on arrival.
//!
//! DiLoCo is pull-based: members dial the coordinator, never the reverse, so a
//! member behind NAT needs no inbound connectivity. A member joins, then loops:
//! `TrainSync` -> fetch θ -> train -> `BlobPut` delta -> `TrainPush`.
//!
//! [`Limits::PEER`]: crate::frame::Limits::PEER

use crate::frame::Tagged;
use crate::identity::{ArchId, BlobId, ModelId};
use crate::types::*;
use serde::{Deserialize, Serialize};

pub const PROTO_RPC: &str = "/ojas/rpc/1.0.0";
pub const PROTO_STREAM: &str = "/ojas/stream/1.0.0";
pub const TOPIC_NODES: &str = "ojas/nodes/1";
/// libp2p identify `protocol_version`. A peer that does not announce it is not an
/// ojas node, however healthy its connection: completing identify proves nothing.
pub const AGENT_PROTOCOL: &str = "ojas/1";

/// Largest chunk a `BlobGet`/`BlobPut` carries.
pub const BLOB_CHUNK: u32 = 8 << 20;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t")]
pub enum PeerReq {
    /// Streamed (on [`PROTO_STREAM`]) as `Token`... then `Done`.
    Generate(GenerateReq),
    /// Response `Scores` with an f32 tensor payload.
    Score(ScoreReq),
    /// Response `Blob` with up to `len` bytes from `offset`.
    BlobGet { blob: BlobId, offset: u64, len: u32 },
    /// Payload = bytes at `offset` of a blob of `total` bytes. The receiver keeps
    /// chunks and verifies the whole blob against `blob` once it is complete.
    BlobPut { blob: BlobId, offset: u64, total: u64 },
    TrainJoin { run: String, backend: Backend, device: String, memory_bytes: u64, engine_version: String },
    TrainSync { run: String, member: u32 },
    /// Sent after the delta blob has been put in full.
    TrainPush { member: u32, report: DeltaReport, delta: BlobId, delta_bytes: u64 },
    TrainLeave { run: String, member: u32 },
}

impl Tagged for PeerReq {
    const TAGS: &'static [&'static str] =
        &["Generate", "Score", "BlobGet", "BlobPut", "TrainJoin", "TrainSync", "TrainPush", "TrainLeave"];
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t")]
pub enum PeerResp {
    Token { req: ReqId, token: u32, #[serde(default, skip_serializing_if = "Option::is_none")] text: Option<String> },
    Done { req: ReqId, prompt_tokens: u32, completion_tokens: u32, finish: Finish, backend: Backend },
    Scores { req: ReqId },
    /// Payload = the bytes.
    Blob { blob: BlobId, offset: u64, total: u64 },
    Ok,
    TrainJoined { member: u32, spec: TrainSpec },
    TrainState(TrainState),
    TrainAck { accepted: bool, #[serde(default)] reason: String },
    /// `retry`: transient (busy, not ready yet); try again later.
    Error { message: String, #[serde(default)] retry: bool },
}

impl Tagged for PeerResp {
    const TAGS: &'static [&'static str] =
        &["Token", "Done", "Scores", "Blob", "Ok", "TrainJoined", "TrainState", "TrainAck", "Error"];
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainState {
    pub round: RoundSpec,
    /// Weights for `round.base`, as a blob of an f32 vector tensor.
    pub theta: BlobId,
    pub theta_bytes: u64,
    pub status: RoundStatus,
    /// Members whose delta for this round has been accepted.
    pub received: u32,
    pub expected: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundStatus {
    /// Accepting deltas for `round`.
    Open,
    /// This member's delta for `round` is in; wait for the next round.
    Submitted,
    /// The run has finished.
    Done,
}

/// Periodic gossip: what this node offers. Small; capabilities only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Announce {
    pub proto: u32,
    pub features: u64,
    pub engine_version: String,
    pub workers: Vec<WorkerSummary>,
    pub models: Vec<ServedModel>,
    /// Generations in flight across all workers.
    pub busy: u32,
    pub unix_ms: u64,
    /// Relay circuit addresses this node can be reached through when it is behind NAT.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relays: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkerSummary {
    pub backend: Backend,
    pub device: String,
    pub memory_bytes: u64,
    pub slots: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ServedModel {
    pub model: ModelId,
    pub arch_id: ArchId,
    pub arch: String,
    pub backend: Backend,
    pub context: u32,
    /// Measured decode rate, if any yet.
    pub tok_s: Option<f32>,
    /// The API name the serving node gives it, so a model can be asked for by name on
    /// a node that does not hold it. Routing still checks `model`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{self, Limits};

    #[test]
    fn tags_cover_every_variant() {
        let reqs = vec![
            PeerReq::Generate(GenerateReq { req: ReqId(1), model: ModelId([0; 32]), prompt: vec![1], max_tokens: 1, sampling: Sampling::greedy(), want_text: false }),
            PeerReq::Score(ScoreReq { req: ReqId(1), model: ModelId([0; 32]), tokens: vec![1, 2], from: 1 }),
            PeerReq::BlobGet { blob: BlobId([0; 32]), offset: 0, len: 1 },
            PeerReq::BlobPut { blob: BlobId([0; 32]), offset: 0, total: 1 },
            PeerReq::TrainJoin { run: "r".into(), backend: Backend::Cpu, device: "d".into(), memory_bytes: 1, engine_version: "v".into() },
            PeerReq::TrainSync { run: "r".into(), member: 0 },
            PeerReq::TrainPush { member: 0, report: DeltaReport { run: "r".into(), round: 0, base: ModelId([0; 32]), arch: ArchId([0; 32]), first_seq: 0, stride: 1, seqs: 1, tokens: 1, mean_loss: 1.0, final_loss: 1.0, backend: Backend::Cpu, wall_ms: 1 }, delta: BlobId([0; 32]), delta_bytes: 4 },
            PeerReq::TrainLeave { run: "r".into(), member: 0 },
        ];
        assert_eq!(reqs.len(), PeerReq::TAGS.len());
        for m in reqs {
            let mut b = Vec::new();
            frame::write(&mut b, &m, &[], Limits::PEER).unwrap();
            assert_eq!(frame::read::<_, PeerReq>(&mut &b[..], Limits::PEER).unwrap().0, m);
        }
        let resps = vec![
            PeerResp::Token { req: ReqId(1), token: 2, text: None },
            PeerResp::Done { req: ReqId(1), prompt_tokens: 1, completion_tokens: 1, finish: Finish::Length, backend: Backend::Metal },
            PeerResp::Scores { req: ReqId(1) },
            PeerResp::Blob { blob: BlobId([0; 32]), offset: 0, total: 1 },
            PeerResp::Ok,
            PeerResp::TrainJoined { member: 0, spec: TrainSpec { run: "r".into(), model: TrainModel::TinyGpt(TinyGptConfig::small(64)), arch: ArchId([0; 32]), data: DataRef { blob: BlobId([0; 32]), bytes: 1 }, shard: 0, n_shards: 1, inner_steps: 1, batch: 1, inner_lr: 1e-3, weight_decay: 0.0 } },
            PeerResp::TrainState(TrainState { round: RoundSpec { round: 0, base: ModelId([0; 32]), cursor: 0 }, theta: BlobId([0; 32]), theta_bytes: 1, status: RoundStatus::Open, received: 0, expected: 2 }),
            PeerResp::TrainAck { accepted: true, reason: String::new() },
            PeerResp::Error { message: "e".into(), retry: true },
        ];
        assert_eq!(resps.len(), PeerResp::TAGS.len());
        for m in resps {
            let mut b = Vec::new();
            frame::write(&mut b, &m, &[], Limits::PEER).unwrap();
            assert_eq!(frame::read::<_, PeerResp>(&mut &b[..], Limits::PEER).unwrap().0, m);
        }
    }
}
