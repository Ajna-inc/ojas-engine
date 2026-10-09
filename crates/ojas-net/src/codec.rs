//! `request_response` codec for [`PROTO_RPC`]: one `frame` each way, `Limits::PEER`.
//!
//! Adapted in idea from SwarmLLM (MIT OR Apache-2.0), `src/network/protocol/mod.rs`
//! @ b14482d: binary data rides out of the JSON header. Theirs tags each payload
//! with a type byte; ours is the proto's `[header][payload]` frame, so the codec
//! is a thin adapter between futures-io and tokio-io.
//!
//! [`PROTO_RPC`]: ojas_swarm_proto::peer::PROTO_RPC

use async_trait::async_trait;
use futures::{AsyncRead, AsyncWrite, AsyncWriteExt};
use libp2p::StreamProtocol;
use ojas_swarm_proto::frame::{self, FrameError, Limits};
use ojas_swarm_proto::peer::{PeerReq, PeerResp};
use std::io;
use tokio_util::compat::{FuturesAsyncReadCompatExt, FuturesAsyncWriteCompatExt};

pub type Request = (PeerReq, Vec<u8>);
pub type Response = (PeerResp, Vec<u8>);

#[derive(Clone, Default)]
pub struct RpcCodec;

fn io_err(e: FrameError) -> io::Error {
    match e {
        FrameError::Io(e) => e,
        other => io::Error::new(io::ErrorKind::InvalidData, other.to_string()),
    }
}

#[async_trait]
impl libp2p::request_response::Codec for RpcCodec {
    type Protocol = StreamProtocol;
    type Request = Request;
    type Response = Response;

    async fn read_request<T: AsyncRead + Unpin + Send>(&mut self, _: &StreamProtocol, io: &mut T) -> io::Result<Request> {
        // An unknown request cannot be answered meaningfully: a newer peer should
        // have gated it on a feature bit. Failing the inbound is the honest reply.
        frame::nonblocking::read(&mut io.compat(), Limits::PEER).await.map_err(io_err)
    }

    async fn read_response<T: AsyncRead + Unpin + Send>(&mut self, _: &StreamProtocol, io: &mut T) -> io::Result<Response> {
        match frame::nonblocking::read(&mut io.compat(), Limits::PEER).await {
            Ok(r) => Ok(r),
            // A response from a newer build: surface it as an error the caller can
            // act on instead of failing the request at the transport.
            Err(FrameError::Unknown(t)) => {
                Ok((PeerResp::Error { message: format!("peer sent unknown response {t:?}"), retry: false }, Vec::new()))
            }
            Err(e) => Err(io_err(e)),
        }
    }

    async fn write_request<T: AsyncWrite + Unpin + Send>(&mut self, _: &StreamProtocol, io: &mut T, (m, p): Request) -> io::Result<()> {
        frame::nonblocking::write(&mut io.compat_write(), &m, &p, Limits::PEER).await.map_err(io_err)?;
        io.close().await
    }

    async fn write_response<T: AsyncWrite + Unpin + Send>(&mut self, _: &StreamProtocol, io: &mut T, (m, p): Response) -> io::Result<()> {
        frame::nonblocking::write(&mut io.compat_write(), &m, &p, Limits::PEER).await.map_err(io_err)?;
        io.close().await
    }
}
