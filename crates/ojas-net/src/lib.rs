//! ojas-net — the libp2p layer of the ojas swarm.
//!
//! A [`Node`] owns a libp2p `Swarm` in one tokio task and is driven through a
//! cloneable handle. It speaks the contract in `ojas-swarm-proto`:
//!
//! * [`PROTO_RPC`] — one request, one response, for blobs and training;
//! * [`PROTO_STREAM`] — one libp2p-stream per generation, tokens as frames;
//! * [`TOPIC_NODES`] — gossipsub carrying [`Announce`].
//!
//! Every pool is invite-only ([`pool`]): a peer that cannot prove membership is
//! disconnected, and nothing it sends reaches the application.
//!
//! Transport, behaviour layout, relay and discovery choices are adapted from SwarmLLM
//! (MIT OR Apache-2.0, <https://github.com/enapt/SwarmLLM> @ b14482d); see
//! `third_party/swarmllm/`. Each adapted file says which.
//!
//! [`PROTO_RPC`]: ojas_swarm_proto::peer::PROTO_RPC
//! [`PROTO_STREAM`]: ojas_swarm_proto::peer::PROTO_STREAM
//! [`TOPIC_NODES`]: ojas_swarm_proto::peer::TOPIC_NODES
//! [`Announce`]: ojas_swarm_proto::peer::Announce

pub mod behaviour;
pub mod blob;
pub mod codec;
pub mod discovery;
pub mod key;
pub mod node;
pub mod pool;
pub mod relay;
pub mod stream;

pub use libp2p::{Multiaddr, PeerId};
pub use node::{Event, NetConfig, Node, PeerView, Responder};
pub use pool::{Credential, Invite, Pool};
pub use stream::PeerStream;

/// Seconds since the Unix epoch; invites and announces are stamped with it.
pub fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
