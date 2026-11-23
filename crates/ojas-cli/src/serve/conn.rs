//! One client connection, plain or TLS, and the count of those still being
//! answered, which a drain waits on.
//!
//! A connection is cloned to write from another thread than the one that read the
//! request. Every clone counts as the same open request: it is done when the last
//! one is dropped, which is when its last byte has been handed to the socket.

use anyhow::{Context, Result};
use std::io::{self, Read, Write};
use std::net::{IpAddr, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

type Tls = rustls::StreamOwned<rustls::ServerConnection, TcpStream>;

/// Requests read from a connection and not yet fully answered.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Set once the process was asked to stop: nothing new is admitted.
static DRAINING: AtomicBool = AtomicBool::new(false);

/// Counted in `ACTIVE` while alive.
struct Busy;

impl Busy {
    fn new() -> Arc<Busy> {
        ACTIVE.fetch_add(1, Ordering::AcqRel);
        Arc::new(Busy)
    }
}

impl Drop for Busy {
    fn drop(&mut self) { ACTIVE.fetch_sub(1, Ordering::AcqRel); }
}

/// A TLS session, closed properly when the last clone of its connection goes: a
/// client reading a stream to its end otherwise sees a truncated connection.
struct TlsIo(Mutex<Tls>);

impl Drop for TlsIo {
    fn drop(&mut self) {
        let t = self.0.get_mut().unwrap_or_else(|e| e.into_inner());
        t.conn.send_close_notify();
        while t.conn.wants_write() {
            if t.conn.write_tls(&mut t.sock).is_err() { break; }
        }
    }
}

pub(crate) struct Conn {
    /// The socket itself: timeouts, the peer's address, and whether it hung up.
    raw: TcpStream,
    tls: Option<Arc<TlsIo>>,
    /// The caller's `X-Request-Id`, echoed on the response and in the log.
    pub(crate) request_id: Option<String>,
    _busy: Arc<Busy>,
}

impl Conn {
    pub(crate) fn plain(stream: TcpStream) -> Conn {
        Conn { raw: stream, tls: None, request_id: None, _busy: Busy::new() }
    }

    /// A connection that speaks TLS; the handshake happens on its first read.
    pub(crate) fn tls(stream: TcpStream, config: &Arc<rustls::ServerConfig>) -> io::Result<Conn> {
        let session = rustls::ServerConnection::new(Arc::clone(config)).map_err(io::Error::other)?;
        let raw = stream.try_clone()?;
        let tls = Arc::new(TlsIo(Mutex::new(rustls::StreamOwned::new(session, stream))));
        Ok(Conn { raw, tls: Some(tls), request_id: None, _busy: Busy::new() })
    }

    pub(crate) fn try_clone(&self) -> io::Result<Conn> {
        Ok(Conn { raw: self.raw.try_clone()?, tls: self.tls.clone(), request_id: self.request_id.clone(),
                  _busy: Arc::clone(&self._busy) })
    }

    pub(crate) fn set_read_timeout(&self, d: Option<Duration>) -> io::Result<()> { self.raw.set_read_timeout(d) }

    pub(crate) fn peer_ip(&self) -> Option<IpAddr> { self.raw.peer_addr().ok().map(|a| a.ip()) }

    /// Whether the peer closed the connection, without waiting.
    pub(crate) fn closed(&self) -> bool {
        let Some(tls) = &self.tls else { return peer_closed(&self.raw) };
        // A writer holds the session: the client was there a moment ago.
        let Ok(mut t) = tls.0.try_lock() else { return false };
        if self.raw.set_nonblocking(true).is_err() { return false; }
        let t = &mut *t;
        let closed = loop {
            match t.conn.read_tls(&mut t.sock) {
                Ok(0) => break true,
                Ok(_) => match t.conn.process_new_packets() {
                    Ok(state) if !state.peer_has_closed() => continue,
                    _ => break true,
                },
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break false,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => break true,
            }
        };
        let _ = self.raw.set_nonblocking(false);
        closed
    }
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match &self.tls {
            Some(tls) => tls.0.lock().unwrap_or_else(|e| e.into_inner()).read(buf),
            None => self.raw.read(buf),
        }
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &self.tls {
            Some(tls) => tls.0.lock().unwrap_or_else(|e| e.into_inner()).write(buf),
            None => self.raw.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match &self.tls {
            Some(tls) => tls.0.lock().unwrap_or_else(|e| e.into_inner()).flush(),
            None => self.raw.flush(),
        }
    }
}

/// Whether the peer closed the connection: a non-blocking peek reads end of stream.
#[cfg(unix)]
fn peer_closed(stream: &TcpStream) -> bool {
    use std::os::fd::AsRawFd;
    let mut byte = 0u8;
    let n = unsafe {
        libc::recv(stream.as_raw_fd(), (&mut byte as *mut u8).cast(), 1, libc::MSG_PEEK | libc::MSG_DONTWAIT)
    };
    n == 0
}

#[cfg(not(unix))]
fn peer_closed(_stream: &TcpStream) -> bool { false }

/// The server side of TLS from a PEM certificate chain and its private key.
pub(crate) fn tls_config(cert: &str, key: &str) -> Result<Arc<rustls::ServerConfig>> {
    use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer};
    let chain = CertificateDer::pem_file_iter(cert).map_err(|e| anyhow::anyhow!("{e}"))
        .and_then(|certs| certs.collect::<Result<Vec<_>, _>>().map_err(|e| anyhow::anyhow!("{e}")))
        .with_context(|| format!("reading the certificate chain {cert}"))?;
    anyhow::ensure!(!chain.is_empty(), "{cert} holds no certificate");
    let key = PrivateKeyDer::from_pem_file(key).map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("reading the private key {key}"))?;
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions().context("TLS protocol versions")?
        .with_no_client_auth()
        .with_single_cert(chain, key).context("the certificate and key do not form a usable identity")?;
    Ok(Arc::new(config))
}

/// Requests read and not yet fully answered.
pub(crate) fn active() -> usize { ACTIVE.load(Ordering::Acquire) }

/// Whether the process is finishing what it has before it exits.
pub(crate) fn draining() -> bool { DRAINING.load(Ordering::Acquire) }

/// On SIGTERM stop admitting requests, let those in progress finish for up to
/// `grace`, then exit. An orchestrator stops a replica this way; a second signal
/// is not needed, and SIGINT still stops at once.
pub(crate) fn drain_on_sigterm(grace: Duration) {
    #[cfg(unix)]
    {
        extern "C" fn on_term(_: libc::c_int) { DRAINING.store(true, Ordering::Release); }
        let handler = on_term as extern "C" fn(libc::c_int);
        unsafe { libc::signal(libc::SIGTERM, handler as libc::sighandler_t) };
    }
    let _ = std::thread::Builder::new().name("drain".into()).spawn(move || {
        while !draining() { std::thread::sleep(Duration::from_millis(100)); }
        eprintln!("  stopping: {} requests in progress, up to {:.0}s to finish them", active(), grace.as_secs_f32());
        let until = std::time::Instant::now() + grace;
        // Idle twice in a row: a request admitted just before the signal has been counted by then.
        let mut idle = 0;
        while idle < 2 && std::time::Instant::now() < until {
            idle = if active() == 0 { idle + 1 } else { 0 };
            std::thread::sleep(Duration::from_millis(100));
        }
        if active() > 0 { eprintln!("  stopping with {} requests unfinished", active()); }
        std::process::exit(0);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn a_request_is_open_until_every_clone_of_its_connection_is_dropped() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let conn = Conn::plain(listener.accept().unwrap().0);
        let busy = Arc::clone(&conn._busy);
        let writer = conn.try_clone().unwrap();
        assert_eq!(Arc::strong_count(&busy), 3);
        drop(conn);
        assert_eq!(Arc::strong_count(&busy), 2, "the writer still holds the request open");
        assert!(!writer.closed());
        drop(client);
        let until = std::time::Instant::now() + Duration::from_secs(2);
        while !writer.closed() && std::time::Instant::now() < until { std::thread::sleep(Duration::from_millis(10)); }
        assert!(writer.closed());
        drop(writer);
        assert_eq!(Arc::strong_count(&busy), 1);
    }
}
