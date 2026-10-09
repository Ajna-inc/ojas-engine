//! DiLoCo end to end, in process: the coordinator fed `PeerReq`s exactly as the node
//! would, members built with `train::begin` and driven through `RoundTrainer::round`,
//! θ, data and deltas moved as BLAKE3-addressed blob chunks.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use ojas_swarm::coord::{aggregate, eval_loss, Coordinator, RunConfig};
use ojas_swarm::local::{self, toy_corpus, LocalMember};
use ojas_swarm::train::{self, data};
use ojas_swarm_proto::peer::{PeerReq, PeerResp, RoundStatus, TrainState};
use ojas_swarm_proto::{ArchId, Backend, BlobId, DType, DeltaReport, ModelId, Tensor, TinyGptConfig, TrainModel};

// ------------------------------------------------------------------ helpers ---

fn tmpdir(tag: &str) -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let d = std::env::temp_dir().join(format!("ojas-diloco-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn toy_model() -> TinyGptConfig {
    TinyGptConfig { vocab: 256, ctx: 32, d_model: 32, n_layers: 2, n_heads: 2, d_ff: 64, init_seed: 11 }
}

fn config(tag: &str, model: TinyGptConfig, max_members: u32) -> RunConfig {
    RunConfig {
        run: format!("test-{tag}"),
        model: TrainModel::TinyGpt(model),
        data: "unused".into(),
        heldout: None,
        inner_steps: 8,
        batch: 4,
        inner_lr: 3e-3,
        weight_decay: 0.01,
        outer_lr: 0.7,
        outer_momentum: 0.9,
        rounds: 6,
        min_members: 1,
        max_members,
        round_deadline_secs: 600,
        checkpoint_dir: tmpdir(tag),
        eval_seqs: 16,
    }
}

fn files(bytes: usize) -> (Vec<u8>, Vec<u8>) {
    data::split_heldout(&data::from_bytes(&toy_corpus(bytes, 42)), 0.1).unwrap()
}

type Member = LocalMember;

fn fetch(c: &mut Coordinator, peer: &str, blob: BlobId, chunk: u32) -> Vec<u8> {
    local::fetch(c, peer, blob, chunk).unwrap()
}

fn put(c: &mut Coordinator, peer: &str, bytes: &[u8], chunk: u32) -> BlobId {
    local::put(c, peer, bytes, chunk).unwrap()
}

fn sync(c: &mut Coordinator, m: &Member) -> TrainState {
    m.sync(c).unwrap()
}

fn join(c: &mut Coordinator, peer: &str) -> Member {
    LocalMember::join(c, peer, Backend::Cpu).unwrap()
}

fn train_one(c: &mut Coordinator, m: &mut Member) -> Option<(DeltaReport, Vec<f32>)> {
    m.train(c).unwrap()
}

fn push_raw(c: &mut Coordinator, peer: &str, member: u32, report: &DeltaReport, bytes: &[u8]) -> PeerResp {
    let blob = put(c, peer, bytes, 1 << 14);
    c.handle(peer, PeerReq::TrainPush { member, report: report.clone(), delta: blob, delta_bytes: bytes.len() as u64 }, vec![]).0
}

fn refused(r: &PeerResp, why: &str) {
    match r {
        PeerResp::TrainAck { accepted: false, reason } => eprintln!("refused as expected ({why}): {reason}"),
        PeerResp::Error { message, .. } => eprintln!("refused as expected ({why}): {message}"),
        _ => panic!("{why}: accepted: {r:?}"),
    }
}

// -------------------------------------------------------------------- tests ---

#[test]
fn aggregate_matches_a_hand_computed_example() {
    // g = 0.25·[0.5, 1.0] + 0.75·[0.1, −0.2] = [0.2, 0.1]
    // m = 0.9·[0.1, −0.2] + g = [0.29, −0.08]
    // θ = [1, 2] − 0.5·(g + 0.9·m) = [1 − 0.5·0.461, 2 − 0.5·0.028] = [0.7695, 1.986]
    let theta = [1.0f32, 2.0];
    let mut m = [0.1f32, -0.2];
    let (a, b) = ([0.5f32, 1.0], [0.1f32, -0.2]);
    let out = aggregate(&theta, &mut m, &[(100, &a), (300, &b)], 0.5, 0.9).unwrap();
    let close = |x: f32, y: f32| (x - y).abs() < 1e-6;
    assert!(close(out[0], 0.7695) && close(out[1], 1.986), "{out:?}");
    assert!(close(m[0], 0.29) && close(m[1], -0.08), "{m:?}");
    // a non-finite result touches nothing
    let mut m2 = [0.0f32; 2];
    let big = [f32::MAX, 0.0];
    assert!(aggregate(&theta, &mut m2, &[(1, &big)], 1e30, 0.9).is_err());
    assert_eq!(m2, [0.0, 0.0]);
    assert!(aggregate(&theta, &mut m2, &[(0, &a)], 0.5, 0.9).is_err(), "zero tokens");
}

#[test]
fn bad_deltas_and_strangers_are_refused() {
    let small = TinyGptConfig { vocab: 256, ctx: 8, d_model: 8, n_layers: 1, n_heads: 2, d_ff: 16, init_seed: 1 };
    let mut cfg = config("refuse", small, 2);
    cfg.inner_steps = 2;
    cfg.batch = 2;
    let (train_f, held) = files(4000);
    let mut c = Coordinator::new(cfg, train_f, Some(held)).unwrap();
    let a = join(&mut c, "peer-a");
    let mut b = join(&mut c, "peer-b");
    assert_eq!((a.id, b.id), (0, 1));
    // re-joining keeps the slot
    let again = join(&mut c, "peer-a");
    assert_eq!(again.id, 0);

    let st = sync(&mut c, &a);
    let theta = Tensor::decode(&fetch(&mut c, "peer-a", st.theta, 1 << 12)).unwrap().0.data;
    let mut t = train::begin(&a.spec, Some(&theta), Backend::Cpu).unwrap();
    let (report, delta) = t.round(&st.round, &a.data).unwrap();
    let good = Tensor::vector(delta.clone()).encode(DType::F32);

    // a member refuses to train from weights that are not the round's base
    let mut wrong = st.round.clone();
    wrong.base = ModelId([7; 32]);
    assert!(t.round(&wrong, &a.data).is_err());

    // strangers
    refused(&c.handle("mallory", PeerReq::BlobPut { blob: BlobId::of(&good), offset: 0, total: good.len() as u64 }, good[..64].to_vec()).0, "unbound peer blob put");
    refused(&c.handle("mallory", PeerReq::TrainPush { member: 0, report: report.clone(), delta: BlobId::of(&good), delta_bytes: good.len() as u64 }, vec![]).0, "unbound peer push");
    refused(&c.handle("mallory", PeerReq::TrainSync { run: a.spec.run.clone(), member: 0 }, vec![]).0, "unbound peer sync");
    refused(&c.handle("mallory", PeerReq::BlobGet { blob: st.theta, offset: 0, len: 10 }, vec![]).0, "unbound peer blob get");
    put(&mut c, "peer-b", &good, 1 << 14);
    refused(&c.handle("peer-b", PeerReq::TrainPush { member: 0, report: report.clone(), delta: BlobId::of(&good), delta_bytes: good.len() as u64 }, vec![]).0, "member 0 pushed by member 1's peer");

    let mut r = report.clone();
    r.base = ModelId([9; 32]);
    refused(&push_raw(&mut c, "peer-a", 0, &r, &good), "foreign base");
    let mut r = report.clone();
    r.arch = ArchId([1; 32]);
    refused(&push_raw(&mut c, "peer-a", 0, &r, &good), "wrong arch");
    let short = Tensor::vector(delta[..delta.len() - 1].to_vec()).encode(DType::F32);
    refused(&push_raw(&mut c, "peer-a", 0, &report, &short), "wrong length");
    let mut nan = delta.clone();
    nan[3] = f32::NAN;
    refused(&push_raw(&mut c, "peer-a", 0, &report, &Tensor::vector(nan).encode(DType::F32)), "NaN delta");
    let mut r = report.clone();
    r.first_seq += 1;
    refused(&push_raw(&mut c, "peer-a", 0, &r, &good), "shard range not the assigned one");
    // a corrupted chunk fails verification
    let mut bad = good.clone();
    bad[40] ^= 1;
    let blob = BlobId::of(&good);
    let r = c.handle("peer-a", PeerReq::BlobPut { blob, offset: 0, total: bad.len() as u64 }, bad);
    refused(&r.0, "resent chunk that differs");
    let r = c.handle("peer-a", PeerReq::BlobPut { blob: BlobId([5; 32]), offset: 0, total: good.len() as u64 }, good.clone());
    refused(&r.0, "blob that does not hash to its id");
    refused(&c.handle("peer-a", PeerReq::BlobPut { blob, offset: 0, total: 1 << 40 }, vec![]).0, "oversized blob");
    assert_eq!(sync(&mut c, &a).received, 0, "nothing accepted so far");

    // the genuine one is accepted; the round waits for member 1
    assert_eq!(push_raw(&mut c, "peer-a", 0, &report, &good), PeerResp::TrainAck { accepted: true, reason: String::new() });
    assert_eq!(sync(&mut c, &a).status, RoundStatus::Submitted);
    assert_eq!(c.round(), 0);
    train_one(&mut c, &mut b).unwrap();
    assert_eq!(c.round(), 1, "round closed once both bound members pushed");
    // late delta for round 0
    refused(&push_raw(&mut c, "peer-a", 0, &report, &good), "late delta for an old round");
    // leaving frees the slot for someone else
    assert_eq!(c.handle("peer-a", PeerReq::TrainLeave { run: a.spec.run.clone(), member: 0 }, vec![]).0, PeerResp::Ok);
    let carol = join(&mut c, "peer-c");
    assert_eq!(carol.id, 0);
    refused(&c.handle("peer-a", PeerReq::TrainSync { run: a.spec.run.clone(), member: 0 }, vec![]).0, "slot rebound to another peer");
}

#[test]
fn deadline_closes_a_round_without_the_straggler() {
    let small = TinyGptConfig { vocab: 256, ctx: 8, d_model: 8, n_layers: 1, n_heads: 2, d_ff: 16, init_seed: 1 };
    let mut cfg = config("deadline", small, 3);
    cfg.inner_steps = 2;
    cfg.batch = 2;
    cfg.min_members = 2;
    cfg.round_deadline_secs = 60;
    let (train_f, held) = files(4000);
    let mut c = Coordinator::new(cfg, train_f, Some(held)).unwrap();
    let mut ms: Vec<Member> = (0..3).map(|i| join(&mut c, &format!("peer-{i}"))).collect();
    train_one(&mut c, &mut ms[0]).unwrap();
    assert!(!c.tick_at(Instant::now() + Duration::from_secs(120)), "one delta is below min_members, deadline or not");
    train_one(&mut c, &mut ms[1]).unwrap();
    assert!(!c.tick_at(Instant::now()), "before the deadline, the straggler is waited for");
    assert_eq!(c.round(), 0);
    assert!(c.tick_at(Instant::now() + Duration::from_secs(61)));
    assert_eq!(c.round(), 1);
    let h = &c.history()[0];
    assert_eq!(h.members.iter().map(|m| m.0).collect::<Vec<_>>(), vec![0, 1]);
    // the straggler's next sync is the new round
    let st = sync(&mut c, &ms[2]);
    assert_eq!((st.round.round, st.status), (1, RoundStatus::Open));
    assert_eq!(st.round.cursor, 4);
}

#[test]
fn resume_from_checkpoint_restores_theta_round_and_members() {
    let small = TinyGptConfig { vocab: 256, ctx: 8, d_model: 8, n_layers: 1, n_heads: 2, d_ff: 16, init_seed: 1 };
    let mut cfg = config("resume", small, 2);
    cfg.inner_steps = 2;
    cfg.batch = 2;
    let (train_f, held) = files(4000);
    let mut c = Coordinator::new(cfg.clone(), train_f.clone(), Some(held.clone())).unwrap();
    let mut ms: Vec<Member> = (0..2).map(|i| join(&mut c, &format!("peer-{i}"))).collect();
    for _ in 0..2 {
        for m in &mut ms {
            train_one(&mut c, m).unwrap();
        }
    }
    assert_eq!(c.round(), 2);
    let (theta, mom, base, members, hist) = (c.theta().to_vec(), c.momentum().to_vec(), c.base(), c.members().to_vec(), c.history().to_vec());
    assert!(mom.iter().any(|v| *v != 0.0));
    drop(c);
    let mut c2 = Coordinator::new(cfg.clone(), train_f.clone(), Some(held.clone())).unwrap();
    assert_eq!(c2.round(), 2);
    assert_eq!(c2.theta(), &theta[..]);
    assert_eq!(c2.momentum(), &mom[..]);
    assert_eq!(c2.base(), base);
    assert_eq!(c2.members(), &members[..]);
    assert_eq!(c2.history(), &hist[..]);
    // the members carry on against the resumed coordinator
    for m in &mut ms {
        train_one(&mut c2, m).unwrap();
    }
    assert_eq!(c2.round(), 3);
    // a checkpoint of another run is refused, not silently replaced
    let mut other = cfg.clone();
    other.run = "another".into();
    assert!(Coordinator::new(other, train_f, Some(held)).is_err());
}

#[test]
fn config_file_round_trip() {
    let dir = tmpdir("cfgfile");
    let (train_f, held) = files(4000);
    std::fs::write(dir.join("train.ojtk"), &train_f).unwrap();
    std::fs::write(dir.join("held.ojtk"), &held).unwrap();
    let json = r#"{
        "run": "file-run",
        "model": {"kind": "tiny_gpt", "vocab": 256, "ctx": 8, "d_model": 8, "n_layers": 1, "n_heads": 2, "d_ff": 16, "init_seed": 3},
        "data": "train.ojtk", "heldout": "held.ojtk",
        "inner_steps": 2, "batch": 2, "inner_lr": 0.001,
        "rounds": 3, "max_members": 2, "checkpoint_dir": "ckpt"
    }"#;
    std::fs::write(dir.join("run.json"), json).unwrap();
    let c = Coordinator::from_config_file(dir.join("run.json")).unwrap();
    assert_eq!((c.config().outer_lr, c.config().outer_momentum, c.config().min_members), (0.7, 0.9, 1));
    assert!(dir.join("ckpt/latest.ckpt").exists());
    assert_eq!(c.data_ref().blob, BlobId::of(&train_f));
    assert!(c.initial_heldout().unwrap() > 5.0, "byte model at init ≈ ln 256");
    // the shipped sample config parses
    let sample = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/run.json")).unwrap();
    let s: RunConfig = serde_json::from_str(&sample).unwrap();
    assert_eq!(s.model, TrainModel::TinyGpt(TinyGptConfig::small(256)));
}

#[test]
fn token_files_and_shards() {
    let f = data::from_bytes(b"hello world, hello swarm");
    let t = data::Tokens::parse(&f).unwrap();
    assert_eq!((t.vocab, t.width, t.n), (256, 2, 24));
    assert_eq!(t.n_seqs(4), 5);
    let mut w = Vec::new();
    t.window(1, 4, &mut w).unwrap();
    assert_eq!(w, b"o wor".iter().map(|&b| b as u32).collect::<Vec<_>>());
    assert!(t.window(5, 4, &mut w).is_err());
    assert!(data::Tokens::parse(&f[..f.len() - 1]).is_err(), "truncated");
    let wide = data::encode(70000, &[1, 69999, 5]).unwrap();
    assert_eq!(data::Tokens::parse(&wide).unwrap().get(1), 69999);
    // members of one round never share a sequence; rounds continue the stride
    let (steps, batch, n) = (3u64, 2u64, 3u32);
    let mut seen = std::collections::HashSet::new();
    for round in 0..2u64 {
        for shard in 0..n {
            let (first, stride) = data::stride_of(round * steps * batch, shard, n);
            for k in 0..steps * batch {
                assert!(seen.insert(data::seq_index(first, stride, k, 1_000)), "repeat");
            }
        }
    }
    assert_eq!(seen.len(), 36);
}

/// Several rounds of 3 CPU members vs 1 member at equal per-member steps.
#[test]
fn three_members_train_and_beat_one() {
    let (train_f, held) = files(24_000);
    let run = |members: u32, tag: &str| -> (f32, Vec<f32>) {
        let cfg = config(tag, toy_model(), members);
        let mut c = Coordinator::new(cfg, train_f.clone(), Some(held.clone())).unwrap();
        let mut ms: Vec<Member> = (0..members).map(|i| join(&mut c, &format!("{tag}-{i}"))).collect();
        let t0 = Instant::now();
        while !c.done() {
            for m in &mut ms {
                train_one(&mut c, m);
            }
        }
        let curve: Vec<f32> = c.history().iter().map(|h| h.heldout_loss.unwrap()).collect();
        eprintln!("{members} member(s): held-out {:.4} -> {curve:.4?} ({:.1}s)", c.initial_heldout().unwrap(), t0.elapsed().as_secs_f32());
        (c.initial_heldout().unwrap(), curve)
    };
    let (init3, three) = run(3, "e2e3");
    let (init1, one) = run(1, "e2e1");
    assert_eq!(init3, init1, "same θ0");
    assert!(three.last().unwrap() < &(init3 - 0.5), "3 members learned: {init3} -> {three:?}");
    assert!(three.windows(2).filter(|w| w[1] < w[0]).count() >= three.len() / 2, "mostly decreasing: {three:?}");
    assert!(three.last().unwrap() <= &(one.last().unwrap() + 0.02), "3 members {three:?} vs 1 member {one:?}");
}

/// The coordinator's eval and a member's own loss agree on the same θ (one CPU path).
#[test]
fn eval_is_the_model_loss() {
    let (train_f, held) = files(4000);
    let m = toy_model();
    let theta = ojas_learn::models::tiny_gpt::init(&m);
    let a = eval_loss(&m, &theta, &held, 4).unwrap();
    let b = eval_loss(&m, &theta, &held, 4).unwrap();
    assert_eq!(a, b);
    assert!((a - 256f32.ln()).abs() < 0.2, "{a}");
    let _ = train_f;
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_and_cpu_members_agree_on_a_round() {
    let (train_f, held) = files(24_000);
    let cfg = config("metal", toy_model(), 2);
    let mut c = Coordinator::new(cfg, train_f, Some(held)).unwrap();
    let a = join(&mut c, "cpu");
    let st = sync(&mut c, &a);
    let theta = Tensor::decode(&fetch(&mut c, "cpu", st.theta, 1 << 16)).unwrap().0.data;
    // the same shard on both, so the data is identical too
    let mut cpu = train::begin(&a.spec, Some(&theta), Backend::Cpu).unwrap();
    let mut gpu = match train::begin(&a.spec, Some(&theta), Backend::Metal) {
        Ok(g) => g,
        Err(e) if format!("{e:#}").contains("no Metal device") => return eprintln!("skipped: {e:#}"),
        Err(e) => panic!("{e:#}"),
    };
    assert_eq!(gpu.identity(), cpu.identity(), "same θ on both devices hashes the same");
    let (rc, dc) = cpu.round(&st.round, &a.data).unwrap();
    let (rg, dg) = gpu.round(&st.round, &a.data).unwrap();
    let diff: Vec<f32> = dc.iter().zip(&dg).map(|(x, y)| (x - y).abs()).collect();
    let max = diff.iter().fold(0.0f32, |a, &b| a.max(b));
    let mean = diff.iter().map(|&d| d as f64).sum::<f64>() / diff.len() as f64;
    let scale = dc.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let mean_mag = dc.iter().map(|&d| d.abs() as f64).sum::<f64>() / dc.len() as f64;
    eprintln!(
        "metal vs cpu delta after {} steps: max |diff| {max:.3e} mean |diff| {mean:.3e}; max |delta| {scale:.3e} mean |delta| {mean_mag:.3e}; loss cpu {} metal {}",
        rc.seqs / 4,
        rc.mean_loss,
        rg.mean_loss
    );
    assert_eq!((rc.first_seq, rc.stride, rc.seqs, rc.tokens), (rg.first_seq, rg.stride, rg.seqs, rg.tokens));
    assert_eq!(rg.backend, Backend::Metal);
    assert!((rc.mean_loss - rg.mean_loss).abs() < 1e-3, "loss {} vs {}", rc.mean_loss, rg.mean_loss);
    // relative to the update's size, not the gradient's: AdamW divides by √v, so where a
    // gradient is ~0 rounding noise can move a weight by a sizeable fraction of lr
    assert!(max < 5e-3 * scale, "max |diff| {max} vs max |delta| {scale}");
    assert!(mean < 1e-4 * mean_mag, "mean |diff| {mean} vs mean |delta| {mean_mag}");
}

/// The window order is a bijection of `0..n` (so shards stay disjoint) and does not
/// read the corpus in file order (so a round sees the whole corpus, not one file).
#[test]
fn window_order_is_a_shuffle_of_every_window() {
    for n in [1u64, 2, 3, 7, 64, 1000, 7256, 65_536] {
        let mut seen = vec![false; n as usize];
        for i in 0..n {
            let p = data::permute(i, n);
            assert!(p < n && !std::mem::replace(&mut seen[p as usize], true), "n={n}: {i} -> {p} repeats");
        }
    }
    // 400 consecutive stream positions (one round of the lab run) cover the corpus
    let n = 7256u64;
    let hit: Vec<u64> = (0..400).map(|k| data::seq_index(0, 1, k, n)).collect();
    let tenths: std::collections::BTreeSet<u64> = hit.iter().map(|s| s * 10 / n).collect();
    assert_eq!(tenths.len(), 10, "one round should touch every tenth of the corpus");
}
