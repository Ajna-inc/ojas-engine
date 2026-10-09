//! DiLoCo member: pull work from the coordinator, train on a local worker, push
//! the delta back. Every request is dialled out from here, so a member behind NAT
//! needs no inbound reachability.

use crate::app::{any_worker, App};
use anyhow::{anyhow, bail, Context, Result};
use ojas_net::{blob, relay, Multiaddr, PeerId};
use ojas_swarm_proto::ipc::{NodeMsg, WorkerMsg};
use ojas_swarm_proto::peer::{PeerReq, PeerResp, RoundStatus, TrainState};
use ojas_swarm_proto::tensor::Tensor;
use ojas_swarm_proto::{BlobId, TrainSpec};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

pub async fn member(app: Arc<App>, coordinator: Multiaddr, run: String) {
    let mut wait = Duration::from_secs(1);
    loop {
        match session(&app, &coordinator, &run).await {
            Ok(()) => {
                status(&app, json!({"run": run, "state": "done"}));
                tracing::info!("training run {run} finished");
                return;
            }
            Err(e) => {
                tracing::warn!("training run {run}: {e:#}; rejoining in {wait:?}");
                status(&app, json!({"run": run, "state": "retrying", "error": format!("{e:#}")}));
            }
        }
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(Duration::from_secs(60));
    }
}

fn status(app: &App, v: serde_json::Value) {
    *app.train_status.lock().unwrap() = v;
}

async fn rpc(app: &App, peer: PeerId, req: PeerReq, payload: Vec<u8>) -> Result<(PeerResp, Vec<u8>)> {
    match app.node.rpc(peer, req, payload).await? {
        (PeerResp::Error { message, retry }, _) => bail!("coordinator: {message}{}", if retry { " (transient)" } else { "" }),
        r => Ok(r),
    }
}

async fn connect(app: &App, coord: &Multiaddr) -> Result<PeerId> {
    let peer = relay::target_peer(coord).context("coordinator address must end in /p2p/<PeerId>")?;
    for _ in 0..30 {
        if app.node.is_member(&peer) {
            return Ok(peer);
        }
        app.node.dial(coord.clone())?;
        if app.node.wait_member(&peer).await {
            return Ok(peer);
        }
    }
    bail!("could not reach the coordinator {peer} as a pool member")
}

async fn session(app: &Arc<App>, coord: &Multiaddr, run: &str) -> Result<()> {
    let peer = connect(app, coord).await?;
    let mut w = any_worker(app).await;
    let (member, spec): (u32, TrainSpec) = match rpc(
        app,
        peer,
        PeerReq::TrainJoin { run: run.into(), backend: w.caps.backend, device: w.caps.device.clone(), memory_bytes: w.caps.memory_bytes, engine_version: w.caps.engine_version.clone() },
        vec![],
    )
    .await?
    {
        (PeerResp::TrainJoined { member, spec }, _) => (member, spec),
        (other, _) => bail!("TrainJoin answered {other:?}"),
    };
    tracing::info!("joined run {run} as member {member}, shard {}/{}", spec.shard, spec.n_shards);
    let data = blob::get(&app.node, peer, spec.data.blob, spec.data.bytes).await.context("fetching training data")?;
    let data_path = write_data(app, run, spec.data.blob, &data)?;
    drop(data);

    let mut begun: Option<u64> = None;
    let mut poll = Duration::from_millis(500);
    loop {
        let st: TrainState = match rpc(app, peer, PeerReq::TrainSync { run: run.into(), member }, vec![]).await? {
            (PeerResp::TrainState(s), _) => s,
            (other, _) => bail!("TrainSync answered {other:?}"),
        };
        status(app, json!({"run": run, "member": member, "round": st.round.round, "status": st.status, "received": st.received, "expected": st.expected}));
        match st.status {
            RoundStatus::Done => {
                let _ = w.send(NodeMsg::TrainEnd { run: run.into() }, vec![]);
                let _ = app.node.rpc(peer, PeerReq::TrainLeave { run: run.into(), member }, vec![]).await;
                return Ok(());
            }
            RoundStatus::Submitted => {
                tokio::time::sleep(poll).await;
                poll = (poll * 2).min(Duration::from_secs(10));
                continue;
            }
            RoundStatus::Open => poll = Duration::from_millis(500),
        }
        let theta = blob::get(&app.node, peer, st.theta, st.theta_bytes).await.context("fetching θ")?;
        Tensor::decode(&theta).map_err(|e| anyhow!("θ for round {} is not a tensor: {e}", st.round.round))?;
        if !w.alive() {
            w = any_worker(app).await;
        }
        // Training state lives in the worker process: a restarted worker needs a
        // fresh TrainBegin.
        if begun != Some(w.epoch) {
            match w.control(NodeMsg::TrainBegin { spec: spec.clone() }, theta.clone(), None).await? {
                (WorkerMsg::TrainReady { n_params, .. }, _) => tracing::info!("worker ready to train {n_params} parameters"),
                (WorkerMsg::Error { message, .. }, _) => bail!("TrainBegin: {message}"),
                (other, _) => bail!("TrainBegin answered {other:?}"),
            }
            begun = Some(w.epoch);
        }
        let round = st.round.clone();
        let (report, delta) = match w.control(NodeMsg::TrainRound { round: round.clone(), data_path: data_path.to_string_lossy().into_owned() }, theta, None).await? {
            (WorkerMsg::Delta { report, .. }, d) => (report, d),
            (WorkerMsg::Error { message, .. }, _) => bail!("round {}: {message}", round.round),
            (other, _) => bail!("TrainRound answered {other:?}"),
        };
        let delta_bytes = delta.len() as u64;
        let id = blob::put(&app.node, peer, &delta).await.context("uploading delta")?;
        match rpc(app, peer, PeerReq::TrainPush { member, report: report.clone(), delta: id, delta_bytes }, vec![]).await? {
            (PeerResp::TrainAck { accepted: true, .. }, _) => tracing::info!("round {} delta accepted (loss {:.4})", round.round, report.final_loss),
            (PeerResp::TrainAck { accepted: false, reason }, _) => tracing::warn!("round {} delta refused: {reason}", round.round),
            (other, _) => bail!("TrainPush answered {other:?}"),
        }
    }
}

fn write_data(app: &App, run: &str, id: BlobId, data: &[u8]) -> Result<PathBuf> {
    let dir = app.cfg.data_dir().join("train");
    std::fs::create_dir_all(&dir)?;
    let safe: String = run.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    let p = dir.join(format!("{safe}-{}.bin", id.short()));
    let tmp = p.with_extension("part");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, &p)?;
    Ok(p)
}
