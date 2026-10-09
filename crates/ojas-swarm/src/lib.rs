//! ojas-swarm — training in the swarm.
//!
//! * [`train`] — the member side: build a run's model on this device and run one DiLoCo
//!   round of inner steps, returning `theta_start - theta_end`. Called by
//!   `ojas engine-worker` for `TrainBegin` / `TrainRound`.
//! * [`coord`] — the coordinator side: hands out data shards, collects deltas, refuses
//!   ones from another base or schema, applies the outer step, and serves θ. Transport
//!   agnostic: `ojas-node --coordinator` feeds it [`PeerReq`]s.
//! * [`local`] — an in-process member driving a coordinator directly (tests, examples).
//!
//! [`PeerReq`]: ojas_swarm_proto::peer::PeerReq

pub mod coord;
pub mod local;
pub mod train;

#[cfg(target_os = "macos")]
mod legacy;
#[cfg(target_os = "macos")]
pub use legacy::{diloco_worker, pipe_test, worker, D, NL};
