//! The node <-> worker socket: `unix:<path>` or `tcp:127.0.0.1:<port>`.
//!
//! The node listens and the worker connects, so the node never races a worker
//! that has not bound yet, and a worker cannot outlive the address it was given.
//! Windows has no unix sockets in tokio: there it is loopback TCP.

use anyhow::{bail, Context, Result};
use ojas_swarm_proto::frame::{self, FrameError, Limits, Tagged};
use serde::Serialize;
use tokio::io::{AsyncRead, AsyncWrite};

pub trait Io: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Io for T {}

pub enum Listener {
    Tcp(tokio::net::TcpListener),
    #[cfg(unix)]
    Unix(tokio::net::UnixListener, std::path::PathBuf),
}

impl Listener {
    /// A fresh listener; `unix_path` is used where unix sockets exist.
    pub async fn bind(unix_path: &std::path::Path) -> Result<(Listener, String)> {
        #[cfg(unix)]
        {
            let _ = std::fs::remove_file(unix_path);
            if let Some(d) = unix_path.parent() {
                std::fs::create_dir_all(d)?;
            }
            // sun_path is ~104 bytes: a deep data dir falls back to loopback TCP.
            if unix_path.as_os_str().len() < 100 {
                let l = tokio::net::UnixListener::bind(unix_path).with_context(|| format!("binding {}", unix_path.display()))?;
                return Ok((Listener::Unix(l, unix_path.to_path_buf()), format!("unix:{}", unix_path.display())));
            }
        }
        let _ = unix_path;
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let a = l.local_addr()?;
        Ok((Listener::Tcp(l), format!("tcp:{a}")))
    }

    pub async fn accept(&self) -> Result<Box<dyn Io>> {
        Ok(match self {
            Listener::Tcp(l) => {
                let (s, peer) = l.accept().await?;
                if !peer.ip().is_loopback() {
                    bail!("worker connection from non-loopback {peer}");
                }
                s.set_nodelay(true)?;
                Box::new(s)
            }
            #[cfg(unix)]
            Listener::Unix(l, _) => Box::new(l.accept().await?.0),
        })
    }
}

#[cfg(unix)]
impl Drop for Listener {
    fn drop(&mut self) {
        if let Listener::Unix(_, p) = self {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// The worker side: connect to a `--socket` address.
pub async fn connect(addr: &str) -> Result<Box<dyn Io>> {
    if let Some(a) = addr.strip_prefix("tcp:") {
        let s = tokio::net::TcpStream::connect(a).await.with_context(|| format!("connecting {a}"))?;
        s.set_nodelay(true)?;
        return Ok(Box::new(s));
    }
    #[cfg(unix)]
    if let Some(p) = addr.strip_prefix("unix:") {
        return Ok(Box::new(tokio::net::UnixStream::connect(p).await.with_context(|| format!("connecting {p}"))?));
    }
    bail!("socket address must be unix:<path> or tcp:<host>:<port>, got {addr:?}")
}

pub async fn send<W: AsyncWrite + Unpin, T: Serialize>(w: &mut W, m: &T, payload: &[u8]) -> Result<()> {
    frame::nonblocking::write(w, m, payload, Limits::IPC).await.map_err(|e| anyhow::anyhow!("ipc write: {e}"))
}

/// The next frame this build knows; `None` at a clean end of stream. Unknown
/// frames come from a newer peer and are skipped, logged once each.
pub async fn recv<R: AsyncRead + Unpin, T: Tagged>(r: &mut R, skipped: &mut Vec<String>) -> Result<Option<(T, Vec<u8>)>> {
    loop {
        match frame::nonblocking::read::<_, T>(r, Limits::IPC).await {
            Ok(m) => return Ok(Some(m)),
            Err(FrameError::Unknown(t)) => {
                if !skipped.contains(&t) {
                    tracing::warn!("skipping unknown ipc message {t:?} (newer peer?)");
                    skipped.push(t);
                }
            }
            Err(e) if e.is_eof() => return Ok(None),
            Err(e) => bail!("ipc read: {e}"),
        }
    }
}
