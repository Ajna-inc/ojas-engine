//! ojas-swarm-proto — the wire contract of the ojas swarm.
//!
//! Two processes per machine:
//!
//! * `ojas-node` — the libp2p daemon: discovery, NAT and relays, routing, the DiLoCo
//!   coordinator client. No GPU code.
//! * `ojas engine-worker` — one per device (Metal, CUDA or CPU): loads models, generates,
//!   scores and trains.
//!
//! This crate is everything the two agree on, and everything two nodes agree on:
//! [`ipc`] (node <-> worker), [`peer`] (node <-> node), [`tensor`] (self-describing
//! tensors), [`identity`] (what makes two models the same), and [`frame`] (the framing
//! under all of it). It builds for every target and does no I/O beyond framing.
//!
//! Evolution is additive: new messages are gated on a [`features`] bit both sides
//! advertise, a variant is never repurposed, and [`PROTO_VERSION`] changes only on a
//! real break. A peer with a different `PROTO_VERSION` is refused at handshake.
//!
//! Parts are adapted from SwarmLLM (MIT OR Apache-2.0, <https://github.com/enapt/SwarmLLM>
//! @ b14482d); see `third_party/swarmllm/`. Each adapted file says which.

pub mod frame;
pub mod identity;
pub mod ipc;
pub mod peer;
pub mod quant;
pub mod tensor;
pub mod types;

pub use identity::{ArchId, BlobId, Identity, IdentityBuilder, ModelId};
pub use tensor::{DType, Tensor};
pub use types::*;

pub const PROTO_VERSION: u32 = 1;

/// Capability bits. A message gated on a bit is only sent to a peer that set it.
pub mod features {
    pub const GENERATE: u64 = 1 << 0;
    pub const SCORE: u64 = 1 << 1;
    pub const TRAIN_DILOCO: u64 = 1 << 2;
    /// Reserved: layer-range forward for pipelines.
    pub const FORWARD_SPAN: u64 = 1 << 3;
}
