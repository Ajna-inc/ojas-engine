//! ojas-swarm — decentralized pipeline workers over the ojas wire protocol.
use anyhow::Result;
use ojas_core::{diloco, wire, Gate, Learner};
use ojas_metal::MetalGpu;
use ojas_train::Trainer;
use std::net::{TcpListener, TcpStream};

pub const NL: usize = 28;
pub const D: usize = 1024;

/// Pipeline worker for layers [lo, hi). With hi >= NL it takes the tail role: loss plus
/// surprise-gated learning from the serving forward, with activations left resident so no second
/// forward is needed.
pub fn worker(model_dir: &str, lo: usize, hi: usize, listen: u16, next_addr: &str,
              t_max: usize, lr: f32, k_sigma: f32) -> Result<()> {
    let tail = hi >= NL;
    let gpu = MetalGpu::new()?;
    let mut tr = Trainer::new(&gpu, model_dir, t_max, lr)?;
    let srv = TcpListener::bind(("0.0.0.0", listen))?;
    tracing::info!(target: "ojas:worker", "{lo}..{hi} :{listen} -> {next_addr}{}", if tail { " TAIL+LEARNER" } else { "" });
    let (mut up, _) = srv.accept()?;
    let mut down = TcpStream::connect(next_addr)?;
    let mut gate = Gate::new(k_sigma);
    loop {
        let (kind, id, toks, x) = wire::recv(&mut up, D)?;
        if kind == wire::STOP { let _ = wire::send(&mut down, wire::STOP, 0, &[], None); break; }
        if !tail {
            let out = tr.fwd_span_range(&toks, x.as_deref(), lo, hi)?;
            wire::send(&mut down, kind, id, &toks, Some(&out))?;
            continue;
        }
        let tgts: Vec<u32> = toks.iter().skip(1).cloned().chain([u32::MAX]).collect();
        let (loss, _) = Learner::step_core(&mut tr, &toks, x.as_deref(), &tgts, lo, false)?;
        let mut updated = false;
        if kind == wire::SERVE && gate.observe(loss) {
            Learner::bwd_from(&mut tr, lo)?;
            tr.step += 1;
            updated = true;
        }
        if gate.n_seen > 0 && gate.n_seen % 100 == 0 && kind == wire::SERVE {
            tracing::debug!(target: "tail", "served {} updated {} mu {:.3}", gate.n_seen, gate.n_upd, gate.mu);
        }
        wire::send_reply(&mut down, id, loss, updated)?;
    }
    Ok(())
}

/// DiLoCo data-parallel worker: train locally for `inner` steps, then exchange only the
/// accumulated parameter difference with the hub. Where `worker` above splits one model across
/// machines, every peer here holds the whole model and a disjoint slice of the data.
///
/// `shard`/`n_shards` select this worker's stride through the corpus; the samples a worker took are
/// recorded rather than reconstructed from a seed.
pub fn diloco_worker(model_dir: &str, data_path: &str, hub_addr: &str,
                     shard: usize, n_shards: usize, inner: usize, lr: f32) -> Result<()> {
    anyhow::ensure!(shard < n_shards, "shard {shard} out of range for {n_shards}");
    let bytes = std::fs::read(data_path)?;
    let rd = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let (t, n) = (rd(4) as usize, rd(8) as usize);
    let seq = |i: usize| -> Vec<u32> {
        let base = 12 + i * t * 8;
        (0..t).map(|j| rd(base + j * 4)).collect()
    };
    let gpu = MetalGpu::new()?;
    let mut tr = Trainer::new(&gpu, model_dir, t, lr)?;
    let sig = tr.weight_sig();
    let mut hub = TcpStream::connect(hub_addr)?;
    wire::send_vec(&mut hub, sig, &tr.weights_f32())?;
    tracing::info!(target: "diloco", "joined {hub_addr} sig {sig:08x} shard {shard}/{n_shards} ({n} seqs T={t})");

    let mut taken = shard;                       // this worker's stride through the corpus
    loop {
        let (round, theta) = wire::recv_vec(&mut hub)?;
        if round == diloco::DONE { break; }
        tr.set_weights_f32(&theta)?;
        let mut loss = 0.0;
        for _ in 0..inner {
            let s = seq(taken % n);
            let tg: Vec<u32> = s.iter().skip(1).cloned().chain([u32::MAX]).collect();
            loss += tr.train_step(&s, &tg)?.0;
            taken += n_shards;
        }
        // delta = theta_start - theta_now, so the hub subtracts a descent direction
        let now = tr.weights_f32();
        let delta: Vec<f32> = theta.iter().zip(&now).map(|(a, b)| a - b).collect();
        wire::send_vec(&mut hub, round, &delta)?;
        tracing::info!(target: "diloco", "round {round}: {inner} steps, mean loss {:.4}, {} samples used",
                       loss / inner as f32, (taken - shard) / n_shards);
    }
    tracing::info!(target: "diloco", "run complete, {} samples used", (taken - shard) / n_shards);
    Ok(())
}

/// Localhost or LAN driver: a parity probe against the monolith, then gated serve traffic.
pub fn pipe_test(model_dir: &str, data_path: &str, first_addr: &str,
                 n_req: usize, ret_port: u16) -> Result<()> {
    let bytes = std::fs::read(data_path)?;
    let rd = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let (t, n) = (rd(4) as usize, rd(8) as usize);
    let seq = |i: usize| -> Vec<u32> {
        let base = 12 + i * t * 8;
        (0..t).map(|j| rd(base + j * 4)).collect()
    };
    let ret = TcpListener::bind(("0.0.0.0", ret_port))?;
    let mut w0 = TcpStream::connect(first_addr)?;
    let (mut back, _) = ret.accept()?;
    tracing::debug!(target: "pipe-test", "connected ({n} seqs T={t})");

    let gpu = MetalGpu::new()?;
    let mut mono = Trainer::new(&gpu, model_dir, t, 1e-5)?;
    let s0 = seq(n - 1);
    let tg0: Vec<u32> = s0.iter().skip(1).cloned().chain([u32::MAX]).collect();
    let (ml, _) = mono.step_core(&s0, None, &tg0, 0, false)?;
    drop(mono);
    wire::send(&mut w0, wire::VAL, 0, &s0, None)?;
    let (_, pl, _) = wire::recv_reply(&mut back)?;
    tracing::info!(target: "parity", "monolith {ml:.4} vs pipeline {pl:.4} (delta {:.5})", (ml - pl).abs());

    let val = |w0: &mut TcpStream, back: &mut TcpStream| -> Result<f32> {
        let mut tot = 0.0;
        for i in (n - 32)..n {
            wire::send(w0, wire::VAL, i as u32, &seq(i), None)?;
            tot += wire::recv_reply(back)?.1;
        }
        Ok(tot / 32.0)
    };
    let v0 = val(&mut w0, &mut back)?;
    tracing::info!(target: "pipe-test", "init val {v0:.4}");
    let t0 = std::time::Instant::now();
    let mut lat = vec![];
    let mut rng = 0x9E3779B97F4A7C15u64;
    for step in 1..=n_req {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let i = ((rng >> 33) as usize) % (n - 32);
        let ts = std::time::Instant::now();
        wire::send(&mut w0, wire::SERVE, step as u32, &seq(i), None)?;
        let (_, l, upd) = wire::recv_reply(&mut back)?;
        lat.push(ts.elapsed().as_secs_f32() * 1000.0);
        if step % 50 == 0 {
            tracing::debug!(target: "ojas", "  req {step:4}  loss {l:.3} upd={upd}  e2e {:.0}ms  [val {:.4}]",
                      lat.last().unwrap(), val(&mut w0, &mut back)?);
        }
    }
    let vf = val(&mut w0, &mut back)?;
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tracing::info!(target: "OJAS PIPE RESULT", "val {v0:.4} -> {vf:.4} over {n_req} reqs | {:.0} tok/s | median e2e {:.0}ms",
              (n_req * t) as f32 / t0.elapsed().as_secs_f32(), lat[lat.len() / 2]);
    wire::send(&mut w0, wire::STOP, 0, &[], None)?;
    Ok(())
}
