//! Framing shared by the node <-> worker IPC and node <-> node streams:
//!
//! ```text
//! [u32 LE header_len][header: JSON, tagged by "t"][u32 LE payload_len][payload]
//! ```
//!
//! Large binary data (tensors, blob chunks) rides in the payload, never in the JSON:
//! a `Vec<u8>` as JSON is about five times its size.
//!
//! Adapted from SwarmLLM (MIT OR Apache-2.0), `src/inference/worker_ipc.rs` @ b14482d,
//! including its rule for version skew: a frame whose tag this build does not know is
//! consumed whole and reported as [`FrameError::Unknown`], so the reader can skip it
//! and stay aligned. A *known* tag that fails to decode is a broken stream and is
//! fatal. A peer gates any new message on a negotiated feature bit anyway; skipping
//! is the safety net for a newer peer that got it wrong.

use serde::{de::DeserializeOwned, Serialize};
use std::io::{self, Read, Write};

/// Per-direction size limits. Every length prefix is checked against these before
/// anything is allocated.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub header: u32,
    pub payload: u32,
}

impl Limits {
    /// Local IPC between a node and its own worker. Large enough for a whole set of
    /// small-model weights in one frame; bigger tensors go as chunks.
    pub const IPC: Limits = Limits { header: 4 << 20, payload: 1 << 30 };
    /// Between nodes. Anything bigger travels as blob chunks.
    pub const PEER: Limits = Limits { header: 1 << 20, payload: 64 << 20 };
}

/// A message enum that knows its own tags, so an unknown one can be told apart from
/// a corrupt one without reading serde's error text.
pub trait Tagged: Serialize + DeserializeOwned {
    const TAGS: &'static [&'static str];
}

#[derive(Debug)]
pub enum FrameError {
    Io(io::Error),
    /// A length prefix exceeded [`Limits`].
    TooLarge { what: &'static str, len: u32, max: u32 },
    /// A tag this build does not know. The frame has been consumed.
    Unknown(String),
    /// A known tag that did not decode, or a header that is not a tagged object.
    Malformed(String),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::Io(e) => write!(f, "{e}"),
            FrameError::TooLarge { what, len, max } => write!(f, "{what} of {len} bytes exceeds {max}"),
            FrameError::Unknown(t) => write!(f, "unknown message {t:?} (newer peer?)"),
            FrameError::Malformed(e) => write!(f, "malformed frame: {e}"),
        }
    }
}
impl std::error::Error for FrameError {}

impl From<io::Error> for FrameError {
    fn from(e: io::Error) -> Self {
        FrameError::Io(e)
    }
}

impl FrameError {
    /// The stream ended cleanly between frames.
    pub fn is_eof(&self) -> bool {
        matches!(self, FrameError::Io(e) if e.kind() == io::ErrorKind::UnexpectedEof)
    }
}

pub fn encode_header<T: Serialize>(msg: &T, lim: Limits) -> Result<Vec<u8>, FrameError> {
    let json = serde_json::to_vec(msg).map_err(|e| FrameError::Malformed(e.to_string()))?;
    check(json.len(), lim.header, "header")?;
    Ok(json)
}

fn check(len: usize, max: u32, what: &'static str) -> Result<u32, FrameError> {
    let l = u32::try_from(len).map_err(|_| FrameError::TooLarge { what, len: u32::MAX, max })?;
    if l > max {
        return Err(FrameError::TooLarge { what, len: l, max });
    }
    Ok(l)
}

pub fn decode_header<T: Tagged>(header: &[u8]) -> Result<T, FrameError> {
    match serde_json::from_slice::<T>(header) {
        Ok(m) => Ok(m),
        Err(e) => {
            let tag = serde_json::from_slice::<serde_json::Value>(header)
                .ok()
                .and_then(|v| v.get("t").and_then(|t| t.as_str()).map(str::to_string));
            match tag {
                Some(t) if !T::TAGS.contains(&t.as_str()) => Err(FrameError::Unknown(t)),
                _ => Err(FrameError::Malformed(e.to_string())),
            }
        }
    }
}

/// Write one frame. Not buffered: wrap the writer in a `BufWriter` and flush, or
/// pass a stream that is already buffered.
pub fn write<W: Write, T: Serialize>(w: &mut W, msg: &T, payload: &[u8], lim: Limits) -> Result<(), FrameError> {
    let h = encode_header(msg, lim)?;
    let pl = check(payload.len(), lim.payload, "payload")?;
    w.write_all(&(h.len() as u32).to_le_bytes())?;
    w.write_all(&h)?;
    w.write_all(&pl.to_le_bytes())?;
    w.write_all(payload)?;
    w.flush()?;
    Ok(())
}

/// Read one raw frame: header bytes and payload.
pub fn read_raw<R: Read>(r: &mut R, lim: Limits) -> Result<(Vec<u8>, Vec<u8>), FrameError> {
    let hl = read_u32(r)?;
    if hl == 0 || hl > lim.header {
        return Err(FrameError::TooLarge { what: "header", len: hl, max: lim.header });
    }
    let mut h = vec![0u8; hl as usize];
    r.read_exact(&mut h)?;
    let pl = read_u32(r)?;
    if pl > lim.payload {
        return Err(FrameError::TooLarge { what: "payload", len: pl, max: lim.payload });
    }
    let mut p = vec![0u8; pl as usize];
    r.read_exact(&mut p)?;
    Ok((h, p))
}

pub fn read<R: Read, T: Tagged>(r: &mut R, lim: Limits) -> Result<(T, Vec<u8>), FrameError> {
    let (h, p) = read_raw(r, lim)?;
    Ok((decode_header(&h)?, p))
}

/// [`read`], skipping frames from a newer peer. Each skipped tag is passed to
/// `on_skip` so it can be logged once.
pub fn read_known<R: Read, T: Tagged>(r: &mut R, lim: Limits, on_skip: &mut dyn FnMut(&str))
                                      -> Result<(T, Vec<u8>), FrameError> {
    loop {
        match read(r, lim) {
            Err(FrameError::Unknown(t)) => on_skip(&t),
            other => return other,
        }
    }
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[cfg(feature = "tokio")]
pub mod nonblocking {
    //! The same framing over tokio streams.
    use super::*;
    use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

    pub async fn write<W: AsyncWrite + Unpin, T: Serialize>(w: &mut W, msg: &T, payload: &[u8], lim: Limits)
                                                            -> Result<(), FrameError> {
        let h = encode_header(msg, lim)?;
        let pl = check(payload.len(), lim.payload, "payload")?;
        let mut head = Vec::with_capacity(8 + h.len());
        head.extend_from_slice(&(h.len() as u32).to_le_bytes());
        head.extend_from_slice(&h);
        head.extend_from_slice(&pl.to_le_bytes());
        w.write_all(&head).await?;
        w.write_all(payload).await?;
        w.flush().await?;
        Ok(())
    }

    pub async fn read_raw<R: AsyncRead + Unpin>(r: &mut R, lim: Limits) -> Result<(Vec<u8>, Vec<u8>), FrameError> {
        let hl = r.read_u32_le().await?;
        if hl == 0 || hl > lim.header {
            return Err(FrameError::TooLarge { what: "header", len: hl, max: lim.header });
        }
        let mut h = vec![0u8; hl as usize];
        r.read_exact(&mut h).await?;
        let pl = r.read_u32_le().await?;
        if pl > lim.payload {
            return Err(FrameError::TooLarge { what: "payload", len: pl, max: lim.payload });
        }
        let mut p = vec![0u8; pl as usize];
        r.read_exact(&mut p).await?;
        Ok((h, p))
    }

    pub async fn read<R: AsyncRead + Unpin, T: Tagged>(r: &mut R, lim: Limits) -> Result<(T, Vec<u8>), FrameError> {
        let (h, p) = read_raw(r, lim).await?;
        Ok((decode_header(&h)?, p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    #[serde(tag = "t")]
    enum Old {
        Ping { n: u32 },
        Bye,
    }
    impl Tagged for Old {
        const TAGS: &'static [&'static str] = &["Ping", "Bye"];
    }

    #[derive(Serialize)]
    #[serde(tag = "t")]
    enum New {
        Ping { n: u32 },
        Shiny { x: String },
    }

    #[test]
    fn frames_round_trip_with_payload() {
        let mut buf = Vec::new();
        write(&mut buf, &Old::Ping { n: 7 }, b"abc", Limits::IPC).unwrap();
        write(&mut buf, &Old::Bye, &[], Limits::IPC).unwrap();
        let mut r = &buf[..];
        assert_eq!(read::<_, Old>(&mut r, Limits::IPC).unwrap(), (Old::Ping { n: 7 }, b"abc".to_vec()));
        assert_eq!(read::<_, Old>(&mut r, Limits::IPC).unwrap(), (Old::Bye, vec![]));
        assert!(read::<_, Old>(&mut r, Limits::IPC).unwrap_err().is_eof());
    }

    #[test]
    fn an_unknown_message_is_skipped_and_the_stream_stays_aligned() {
        let mut buf = Vec::new();
        write(&mut buf, &New::Shiny { x: "from the future".into() }, b"payload", Limits::IPC).unwrap();
        write(&mut buf, &New::Ping { n: 3 }, &[], Limits::IPC).unwrap();
        let mut skipped = vec![];
        let (m, _) = read_known::<_, Old>(&mut &buf[..], Limits::IPC, &mut |t| skipped.push(t.to_string())).unwrap();
        assert_eq!(m, Old::Ping { n: 3 });
        assert_eq!(skipped, ["Shiny"]);
    }

    #[test]
    fn a_known_tag_that_does_not_decode_is_fatal() {
        let mut buf = Vec::new();
        write(&mut buf, &serde_json::json!({"t": "Ping", "n": "seven"}), &[], Limits::IPC).unwrap();
        assert!(matches!(read::<_, Old>(&mut &buf[..], Limits::IPC), Err(FrameError::Malformed(_))));
    }

    #[test]
    fn oversized_prefixes_are_refused_before_allocating() {
        let lim = Limits { header: 64, payload: 16 };
        let mut buf = Vec::new();
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(read_raw(&mut &buf[..], lim), Err(FrameError::TooLarge { what: "header", .. })));
        let mut buf = Vec::new();
        write(&mut buf, &Old::Bye, &[], Limits::IPC).unwrap();
        buf.truncate(buf.len() - 4);
        buf.extend_from_slice(&1000u32.to_le_bytes());
        assert!(matches!(read_raw(&mut &buf[..], lim), Err(FrameError::TooLarge { what: "payload", .. })));
        assert!(matches!(write(&mut Vec::new(), &Old::Bye, &[0; 17], lim), Err(FrameError::TooLarge { .. })));
    }

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn async_and_blocking_framing_agree() {
        let mut buf = Vec::new();
        nonblocking::write(&mut buf, &Old::Ping { n: 9 }, b"xy", Limits::PEER).await.unwrap();
        assert_eq!(read::<_, Old>(&mut &buf[..], Limits::PEER).unwrap().0, Old::Ping { n: 9 });
        let mut b2 = Vec::new();
        write(&mut b2, &Old::Ping { n: 9 }, b"xy", Limits::PEER).unwrap();
        assert_eq!(buf, b2);
        let (m, p) = nonblocking::read::<_, Old>(&mut &b2[..], Limits::PEER).await.unwrap();
        assert_eq!((m, p), (Old::Ping { n: 9 }, b"xy".to_vec()));
    }
}
