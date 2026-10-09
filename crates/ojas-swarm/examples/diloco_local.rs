//! A DiLoCo run in one process: a coordinator and N members talking through
//! `Coordinator::handle` exactly as over the network (blob chunks included), printing
//! the held-out loss after every round.
//!
//! ```text
//! cargo run --release -p ojas-swarm --example diloco_local -- \
//!     [--members 3] [--rounds 8] [--steps 20] [--batch 8] [--lr 3e-3]
//!     [--text corpus.txt | --bytes 65536]   # a byte-level file, or the built-in toy corpus
//!     [--model toy|small] [--metal] [--compare]
//! ```
//! `--metal` puts member 0 on Metal (build with `--features metal`); `--compare` repeats
//! the run with a single member at the same per-member steps.

use std::time::Instant;

use anyhow::{bail, Result};
use ojas_swarm::coord::{Coordinator, RunConfig};
use ojas_swarm::local::{toy_corpus, LocalMember};
use ojas_swarm::train::data;
use ojas_swarm_proto::{Backend, TinyGptConfig, TrainModel};

#[derive(Clone)]
struct Args {
    members: u32,
    rounds: u32,
    steps: u32,
    batch: u32,
    lr: f32,
    text: Option<String>,
    bytes: usize,
    model: String,
    metal: bool,
    compare: bool,
}

fn args() -> Result<Args> {
    let mut a = Args { members: 3, rounds: 8, steps: 20, batch: 8, lr: 3e-3, text: None, bytes: 1 << 16, model: "toy".into(), metal: false, compare: false };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| anyhow::anyhow!("{k} needs a value"));
        match k.as_str() {
            "--members" => a.members = v()?.parse()?,
            "--rounds" => a.rounds = v()?.parse()?,
            "--steps" => a.steps = v()?.parse()?,
            "--batch" => a.batch = v()?.parse()?,
            "--lr" => a.lr = v()?.parse()?,
            "--text" => a.text = Some(v()?),
            "--bytes" => a.bytes = v()?.parse()?,
            "--model" => a.model = v()?,
            "--metal" => a.metal = true,
            "--compare" => a.compare = true,
            _ => bail!("unknown argument {k}"),
        }
    }
    Ok(a)
}

fn run(a: &Args, members: u32, train: &[u8], held: &[u8]) -> Result<Vec<f32>> {
    let model = match a.model.as_str() {
        "small" => TinyGptConfig::small(256),
        "toy" => TinyGptConfig { vocab: 256, ctx: 64, d_model: 64, n_layers: 2, n_heads: 4, d_ff: 256, init_seed: 1 },
        m => bail!("--model {m}: toy or small"),
    };
    let dir = std::env::temp_dir().join(format!("ojas-diloco-local-{}-{members}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = RunConfig {
        run: "diloco-local".into(),
        model: TrainModel::TinyGpt(model),
        data: "memory".into(),
        heldout: None,
        inner_steps: a.steps,
        batch: a.batch,
        inner_lr: a.lr,
        weight_decay: 0.01,
        outer_lr: 0.7,
        outer_momentum: 0.9,
        rounds: a.rounds,
        min_members: members,
        max_members: members,
        round_deadline_secs: 3600,
        checkpoint_dir: dir.clone(),
        eval_seqs: 64,
    };
    let mut c = Coordinator::new(cfg, train.to_vec(), Some(held.to_vec()))?;
    println!("{members} member(s), {} parameters, round 0 held-out {:.4}", c.theta().len(), c.initial_heldout().unwrap_or(f32::NAN));
    let mut ms = Vec::new();
    for i in 0..members {
        let be = if a.metal && i == 0 { Backend::Metal } else { Backend::Cpu };
        ms.push(LocalMember::join(&mut c, &format!("local-{i}"), be)?);
    }
    let mut curve = Vec::new();
    while !c.done() {
        let t0 = Instant::now();
        let mut losses = Vec::new();
        for m in &mut ms {
            if let Some((r, _)) = m.train(&mut c)? {
                losses.push(format!("{}:{:.3}", r.backend.as_str(), r.mean_loss));
            }
        }
        let h = c.history().last().unwrap();
        let l = h.heldout_loss.unwrap_or(f32::NAN);
        curve.push(l);
        println!("round {:3}  held-out {l:.4}  inner [{}]  {:.1}s", h.round, losses.join(" "), t0.elapsed().as_secs_f32());
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(curve)
}

fn main() -> Result<()> {
    let a = args()?;
    let text = match &a.text {
        Some(p) => std::fs::read(p)?,
        None => toy_corpus(a.bytes, 42),
    };
    let (train, held) = data::split_heldout(&data::from_bytes(&text), 0.1)?;
    let many = run(&a, a.members, &train, &held)?;
    if a.compare && a.members != 1 {
        let one = run(&Args { metal: false, ..a.clone() }, 1, &train, &held)?;
        println!("final held-out: {} members {:.4}, 1 member {:.4}", a.members, many.last().unwrap(), one.last().unwrap());
    }
    Ok(())
}
