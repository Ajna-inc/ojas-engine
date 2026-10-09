//! Framed per-request streams on [`PROTO_STREAM`].
//!
//! Adapted from SwarmLLM (MIT OR Apache-2.0), `src/network/pipeline_stream.rs`
//! @ b14482d: one libp2p-stream per request instead of a request/response per
//! token, which they measured at ~100 ms of framing and correlation overhead a
//! token. Ours carries the proto's frames, so either end reads with
//! [`recv`](PeerStream::recv) and skips frames from a newer peer.
//!
//! [`PROTO_STREAM`]: ojas_swarm_proto::peer::PROTO_STREAM

use anyhow::{anyhow, Result};
use libp2p::PeerId;
use ojas_swarm_proto::frame::{self, FrameError, Limits, Tagged};
use serde::Serialize;
use tokio::io::AsyncWriteExt;
use tokio_util::compat::{Compat, FuturesAsyncReadCompatExt};

pub struct PeerStream {
    pub peer: PeerId,
    io: Compat<libp2p::Stream>,
}

impl PeerStream {
    pub fn new(peer: PeerId, s: libp2p::Stream) -> PeerStream {
        PeerStream { peer, io: s.compat() }
    }

    pub async fn send<T: Serialize>(&mut self, msg: &T, payload: &[u8]) -> Result<()> {
        frame::nonblocking::write(&mut self.io, msg, payload, Limits::PEER).await.map_err(|e| anyhow!("stream to {}: {e}", self.peer))
    }

    /// The next known frame; `None` when the peer closed the stream cleanly.
    pub async fn recv<T: Tagged>(&mut self) -> Result<Option<(T, Vec<u8>)>> {
        loop {
            match frame::nonblocking::read::<_, T>(&mut self.io, Limits::PEER).await {
                Ok(m) => return Ok(Some(m)),
                Err(FrameError::Unknown(t)) => tracing::debug!(peer = %self.peer, "skipping unknown stream frame {t:?}"),
                Err(e) if e.is_eof() => return Ok(None),
                Err(e) => return Err(anyhow!("stream from {}: {e}", self.peer)),
            }
        }
    }

    pub async fn close(mut self) {
        let _ = self.io.shutdown().await;
    }
}
