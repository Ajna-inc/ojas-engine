//! An in-process DiLoCo member: the node's member loop (join, sync, fetch θ and data as
//! blob chunks, train, put the delta in chunks, push) driven straight against a
//! [`Coordinator`], with no network. For tests, the `diloco_local` example, and checking a
//! run config before deploying it.

use anyhow::{bail, ensure, Context, Result};
use ojas_swarm_proto::peer::{PeerReq, PeerResp, RoundStatus, TrainState};
use ojas_swarm_proto::{Backend, BlobId, DType, DeltaReport, Tensor, TrainSpec};

use crate::coord::Coordinator;
use crate::train::{self, RoundTrainer};

pub struct LocalMember {
    pub peer: String,
    pub id: u32,
    pub spec: TrainSpec,
    pub backend: Backend,
    /// the run's token file, fetched and verified at join
    pub data: Vec<u8>,
    pub trainer: Option<Box<dyn RoundTrainer>>,
    /// blob chunk size for gets and puts (≤ `peer::BLOB_CHUNK`)
    pub chunk: u32,
}

fn ok(r: (PeerResp, Vec<u8>)) -> Result<(PeerResp, Vec<u8>)> {
    if let PeerResp::Error { message, .. } = &r.0 {
        bail!("coordinator refused: {message}");
    }
    Ok(r)
}

/// Fetch a whole blob in chunks and verify it against its id.
pub fn fetch(c: &mut Coordinator, peer: &str, blob: BlobId, chunk: u32) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let (r, bytes) = ok(c.handle(peer, PeerReq::BlobGet { blob, offset: out.len() as u64, len: chunk }, vec![]))?;
        let PeerResp::Blob { total, .. } = r else { bail!("BlobGet answered {r:?}") };
        ensure!(bytes.len() as u64 <= total - out.len() as u64, "blob overran its total");
        out.extend_from_slice(&bytes);
        if out.len() as u64 == total {
            break;
        }
        ensure!(!bytes.is_empty(), "empty chunk before the end of the blob");
    }
    ensure!(BlobId::of(&out) == blob, "fetched blob does not hash to {}", blob.short());
    Ok(out)
}

/// Put `bytes` as a blob in chunks; returns its id.
pub fn put(c: &mut Coordinator, peer: &str, bytes: &[u8], chunk: u32) -> Result<BlobId> {
    let blob = BlobId::of(bytes);
    for (i, ch) in bytes.chunks(chunk as usize).enumerate() {
        ok(c.handle(peer, PeerReq::BlobPut { blob, offset: (i * chunk as usize) as u64, total: bytes.len() as u64 }, ch.to_vec()))?;
    }
    Ok(blob)
}

impl LocalMember {
    pub fn join(c: &mut Coordinator, peer: &str, backend: Backend) -> Result<LocalMember> {
        let req = PeerReq::TrainJoin { run: c.config().run.clone(), backend, device: "local".into(), memory_bytes: 0, engine_version: env!("CARGO_PKG_VERSION").into() };
        let PeerResp::TrainJoined { member, spec } = ok(c.handle(peer, req, vec![]))?.0 else { bail!("TrainJoin not answered by TrainJoined") };
        let chunk = 1 << 16;
        let data = fetch(c, peer, spec.data.blob, chunk)?;
        Ok(LocalMember { peer: peer.into(), id: member, spec, backend, data, trainer: None, chunk })
    }

    pub fn sync(&self, c: &mut Coordinator) -> Result<TrainState> {
        match ok(c.handle(&self.peer, PeerReq::TrainSync { run: self.spec.run.clone(), member: self.id }, vec![]))?.0 {
            PeerResp::TrainState(s) => Ok(s),
            r => bail!("TrainSync answered {r:?}"),
        }
    }

    /// Train and push this round's delta. `None`: nothing to do (already submitted, or
    /// the run is done).
    pub fn train(&mut self, c: &mut Coordinator) -> Result<Option<(DeltaReport, Vec<f32>)>> {
        let st = self.sync(c)?;
        if st.status != RoundStatus::Open {
            return Ok(None);
        }
        // fetch θ only when ours is not the round's base (by content hash)
        if self.trainer.as_ref().map(|t| t.identity().model) != Some(st.round.base) {
            let theta = Tensor::decode(&fetch(c, &self.peer, st.theta, self.chunk)?).context("θ blob")?.0.data;
            match &mut self.trainer {
                Some(t) => t.set_weights(&theta)?,
                None => self.trainer = Some(train::begin(&self.spec, Some(&theta), self.backend)?),
            }
        }
        let (report, delta) = self.trainer.as_mut().unwrap().round(&st.round, &self.data)?;
        let bytes = Tensor::vector(delta.clone()).encode(DType::F32);
        let blob = put(c, &self.peer, &bytes, self.chunk)?;
        let push = PeerReq::TrainPush { member: self.id, report: report.clone(), delta: blob, delta_bytes: bytes.len() as u64 };
        match c.handle(&self.peer, push, vec![]).0 {
            PeerResp::TrainAck { accepted: true, .. } => Ok(Some((report, delta))),
            r => bail!("delta refused: {r:?}"),
        }
    }
}

/// Pseudo-English from a fixed grammar, `bytes` long: enough structure for a byte model
/// to learn in minutes, enough variety that it cannot just memorise.
pub fn toy_corpus(bytes: usize, seed: u64) -> Vec<u8> {
    let subj = ["the cat", "a dog", "the old man", "my sister", "the bird", "a child", "our team", "the farmer"];
    let verb = ["sees", "likes", "finds", "carries", "paints", "follows", "keeps", "wants"];
    let obj = ["the red ball", "a small boat", "the bread", "an apple", "the green hat", "a long rope", "the map", "some water"];
    let tail = [".", " today.", " again.", " at night.", " by the river."];
    let mut s = seed;
    let mut r = |n: usize| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as usize) % n
    };
    let mut out = String::new();
    while out.len() < bytes {
        out.push_str(&format!("{} {} {}{} ", subj[r(8)], verb[r(8)], obj[r(8)], tail[r(5)]));
    }
    out.into_bytes()[..bytes].to_vec()
}
