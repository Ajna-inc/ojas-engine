//! Content-addressed blob transfer over [`PROTO_RPC`] in [`BLOB_CHUNK`] pieces.
//!
//! Every blob is named by its BLAKE3 ([`BlobId::of`]) and verified whole on
//! arrival, at both ends: a member never trains on θ the coordinator did not send,
//! and the coordinator never averages a delta that was damaged in transit.
//!
//! [`PROTO_RPC`]: ojas_swarm_proto::peer::PROTO_RPC

use crate::node::Node;
use anyhow::{anyhow, bail, Result};
use libp2p::PeerId;
use ojas_swarm_proto::peer::{PeerReq, PeerResp, BLOB_CHUNK};
use ojas_swarm_proto::BlobId;
use std::collections::HashMap;
use std::time::Duration;

const RETRIES: u32 = 5;

async fn rpc_retrying(node: &Node, peer: PeerId, req: PeerReq, payload: &[u8]) -> Result<(PeerResp, Vec<u8>)> {
    let mut wait = Duration::from_millis(200);
    for attempt in 0.. {
        match node.rpc(peer, req.clone(), payload.to_vec()).await {
            Ok((PeerResp::Error { retry: true, message }, _)) if attempt < RETRIES => {
                tracing::debug!("blob rpc to {peer}: {message}; retrying");
            }
            Err(e) if attempt < RETRIES => tracing::debug!("blob rpc to {peer}: {e}; retrying"),
            other => return other,
        }
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(Duration::from_secs(5));
    }
    unreachable!()
}

/// Fetch blob `id` of `total` bytes from `peer`, verified.
pub async fn get(node: &Node, peer: PeerId, id: BlobId, total: u64) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(usize::try_from(total)?);
    while (out.len() as u64) < total {
        let offset = out.len() as u64;
        let len = (total - offset).min(BLOB_CHUNK as u64) as u32;
        match rpc_retrying(node, peer, PeerReq::BlobGet { blob: id, offset, len }, &[]).await? {
            (PeerResp::Blob { blob, offset: o, total: t }, bytes) if blob == id && o == offset && t == total => {
                if bytes.is_empty() || bytes.len() > len as usize {
                    bail!("blob {}: chunk at {offset} has {} bytes, asked for {len}", id.short(), bytes.len());
                }
                out.extend_from_slice(&bytes);
            }
            (PeerResp::Error { message, .. }, _) => bail!("blob {}: {message}", id.short()),
            (other, _) => bail!("blob {}: unexpected answer {other:?}", id.short()),
        }
    }
    if BlobId::of(&out) != id {
        bail!("blob {} failed verification: content hashes to {}", id.short(), BlobId::of(&out).short());
    }
    Ok(out)
}

/// Upload `bytes` to `peer` as a blob; returns its id.
pub async fn put(node: &Node, peer: PeerId, bytes: &[u8]) -> Result<BlobId> {
    let id = BlobId::of(bytes);
    let total = bytes.len() as u64;
    let chunks: Vec<&[u8]> = if bytes.is_empty() { vec![&[]] } else { bytes.chunks(BLOB_CHUNK as usize).collect() };
    let mut offset = 0u64;
    for c in chunks {
        match rpc_retrying(node, peer, PeerReq::BlobPut { blob: id, offset, total }, c).await? {
            (PeerResp::Ok, _) => {}
            (PeerResp::Error { message, .. }, _) => bail!("putting blob {}: {message}", id.short()),
            (other, _) => bail!("putting blob {}: unexpected answer {other:?}", id.short()),
        }
        offset += c.len() as u64;
    }
    Ok(id)
}

/// Blobs a node holds and serves, and uploads it is assembling.
pub struct BlobStore {
    done: HashMap<BlobId, Vec<u8>>,
    partial: HashMap<BlobId, Partial>,
    max_blob: u64,
}

struct Partial {
    buf: Vec<u8>,
    /// Received byte ranges, merged.
    have: Vec<(u64, u64)>,
}

impl BlobStore {
    pub fn new(max_blob: u64) -> BlobStore {
        BlobStore { done: HashMap::new(), partial: HashMap::new(), max_blob }
    }

    pub fn insert(&mut self, bytes: Vec<u8>) -> BlobId {
        let id = BlobId::of(&bytes);
        self.done.insert(id, bytes);
        id
    }

    pub fn get(&self, id: &BlobId) -> Option<&[u8]> {
        self.done.get(id).map(|v| v.as_slice())
    }

    pub fn take(&mut self, id: &BlobId) -> Option<Vec<u8>> {
        self.done.remove(id)
    }

    /// Answer a `BlobGet` or `BlobPut`; `None` for any other request.
    pub fn handle(&mut self, req: &PeerReq, payload: &[u8]) -> Option<(PeerResp, Vec<u8>)> {
        let err = |m: String| (PeerResp::Error { message: m, retry: false }, Vec::new());
        Some(match *req {
            PeerReq::BlobGet { blob, offset, len } => match self.done.get(&blob) {
                Some(b) if offset < b.len() as u64 || (b.is_empty() && offset == 0) => {
                    let end = (offset + len.min(BLOB_CHUNK) as u64).min(b.len() as u64);
                    (PeerResp::Blob { blob, offset, total: b.len() as u64 }, b[offset as usize..end as usize].to_vec())
                }
                Some(_) => err(format!("offset {offset} is past the end of blob {}", blob.short())),
                None => err(format!("no blob {}", blob.short())),
            },
            PeerReq::BlobPut { blob, offset, total } => match self.put_chunk(blob, offset, total, payload) {
                Ok(()) => (PeerResp::Ok, Vec::new()),
                Err(e) => err(e.to_string()),
            },
            _ => return None,
        })
    }

    fn put_chunk(&mut self, id: BlobId, offset: u64, total: u64, chunk: &[u8]) -> Result<()> {
        if self.done.contains_key(&id) {
            return Ok(());
        }
        if total > self.max_blob {
            bail!("blob of {total} bytes exceeds the {} byte limit", self.max_blob);
        }
        let end = offset.checked_add(chunk.len() as u64).filter(|&e| e <= total).ok_or_else(|| anyhow!("chunk at {offset} overruns a {total} byte blob"))?;
        let p = self.partial.entry(id).or_insert_with(|| Partial { buf: vec![0; total as usize], have: Vec::new() });
        if p.buf.len() as u64 != total {
            bail!("blob {} was started with a different size", id.short());
        }
        p.buf[offset as usize..end as usize].copy_from_slice(chunk);
        p.have.push((offset, end));
        p.have.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(p.have.len());
        for &(s, e) in &p.have {
            match merged.last_mut() {
                Some(last) if s <= last.1 => last.1 = last.1.max(e),
                _ => merged.push((s, e)),
            }
        }
        p.have = merged;
        if p.have == [(0, total)] || total == 0 {
            let p = self.partial.remove(&id).unwrap();
            if BlobId::of(&p.buf) != id {
                bail!("blob {} failed verification", id.short());
            }
            self.done.insert(id, p.buf);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_in_any_order_assemble_and_verify() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let id = BlobId::of(&data);
        let mut s = BlobStore::new(1 << 20);
        for (off, c) in [(600u64, &data[600..]), (0, &data[..300]), (300, &data[300..600])] {
            let (r, _) = s.handle(&PeerReq::BlobPut { blob: id, offset: off, total: 1000 }, c).unwrap();
            assert_eq!(r, PeerResp::Ok);
        }
        assert_eq!(s.get(&id), Some(&data[..]));
        let (r, b) = s.handle(&PeerReq::BlobGet { blob: id, offset: 900, len: 500 }, &[]).unwrap();
        assert_eq!(r, PeerResp::Blob { blob: id, offset: 900, total: 1000 });
        assert_eq!(b, &data[900..]);
    }

    #[test]
    fn a_damaged_or_oversized_upload_is_refused() {
        let data = vec![5u8; 64];
        let id = BlobId::of(&data);
        let mut s = BlobStore::new(1 << 20);
        let mut bad = data.clone();
        bad[3] = 0;
        let (r, _) = s.handle(&PeerReq::BlobPut { blob: id, offset: 0, total: 64 }, &bad).unwrap();
        assert!(matches!(r, PeerResp::Error { .. }));
        assert!(s.get(&id).is_none());
        let (r, _) = s.handle(&PeerReq::BlobPut { blob: id, offset: 0, total: 1 << 30 }, &data).unwrap();
        assert!(matches!(r, PeerResp::Error { .. }));
        let (r, _) = s.handle(&PeerReq::BlobPut { blob: id, offset: 60, total: 64 }, &data).unwrap();
        assert!(matches!(r, PeerResp::Error { .. }), "overrun");
    }
}
