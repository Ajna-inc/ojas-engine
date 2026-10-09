//! Member side of a DiLoCo run. The contract `engine-worker` builds against.
//!
//! A member holds the run's model on one device between rounds. Each round it checks
//! that its weights are exactly the coordinator's base (by content hash), runs
//! `inner_steps` AdamW steps over its own stride of the token file, and returns
//! `theta_start - theta_end`. AdamW's moments persist across rounds, as DiLoCo
//! prescribes; [`RoundTrainer::save_state`] lets the worker checkpoint them.

use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use ojas_learn::backend::Backend as Device;
use ojas_learn::models::tiny_gpt::{self, OptState, TinyGpt};
use ojas_swarm_proto::{Backend, DeltaReport, Identity, RoundSpec, TrainModel, TrainSpec};

/// A run's model, resident on one device, between rounds.
pub trait RoundTrainer {
    fn identity(&self) -> Identity;
    fn n_params(&self) -> u64;
    /// Trainable parameters flattened in the model's canonical order, f32.
    fn weights(&self) -> Vec<f32>;
    /// Replace the weights (the coordinator's θ). Resets nothing else: inner optimizer
    /// state persists across rounds, as DiLoCo prescribes.
    fn set_weights(&mut self, theta: &[f32]) -> Result<()>;
    /// `spec.inner_steps` steps from the current weights over this member's stride of
    /// `data` (a token file, see [`data`]), starting at `round.cursor`. Returns the
    /// report and `theta_start - theta_end`. Refuses if the current weights do not hash
    /// to `round.base`, and if any loss, gradient or weight goes non-finite; a refused
    /// round leaves weights and optimizer state as they were.
    fn round(&mut self, round: &RoundSpec, data: &[u8]) -> Result<(DeltaReport, Vec<f32>)>;
    /// Weights plus inner optimizer state (AdamW step and moments), so a restarted
    /// worker resumes with the same moments rather than cold ones.
    fn save_state(&self) -> Result<Vec<u8>> {
        bail!("this trainer does not checkpoint its optimizer state")
    }
    fn load_state(&mut self, state: &[u8]) -> Result<()> {
        let _ = state;
        bail!("this trainer does not checkpoint its optimizer state")
    }
}

/// Build the run's model on `backend`. `theta0` = the coordinator's weights; `None`
/// initialises from the config's seed.
pub fn begin(spec: &TrainSpec, theta0: Option<&[f32]>, backend: Backend) -> Result<Box<dyn RoundTrainer>> {
    ensure!(spec.n_shards > 0 && spec.shard < spec.n_shards, "train spec: shard {} of {}", spec.shard, spec.n_shards);
    ensure!(spec.inner_steps > 0 && spec.batch > 0, "train spec: inner_steps and batch must be positive");
    ensure!(
        spec.inner_lr.is_finite() && spec.inner_lr > 0.0 && spec.weight_decay.is_finite() && spec.weight_decay >= 0.0,
        "train spec: bad inner_lr {} / weight_decay {}",
        spec.inner_lr,
        spec.weight_decay
    );
    match backend {
        Backend::Cpu => Ok(Box::new(GptTrainer::new(ojas_learn::cpu::Cpu, Backend::Cpu, spec, theta0)?)),
        Backend::Metal => metal(spec, theta0),
        Backend::Cuda => cuda(spec, theta0),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal(spec: &TrainSpec, theta0: Option<&[f32]>) -> Result<Box<dyn RoundTrainer>> {
    let be = ojas_learn::metal::Metal::new().context("starting the Metal training backend")?;
    Ok(Box::new(GptTrainer::new(be, Backend::Metal, spec, theta0)?))
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn metal(_: &TrainSpec, _: Option<&[f32]>) -> Result<Box<dyn RoundTrainer>> {
    bail!("this build cannot train on Metal: build ojas-swarm with `--features metal` on macOS")
}

#[cfg(feature = "cuda")]
fn cuda(spec: &TrainSpec, theta0: Option<&[f32]>) -> Result<Box<dyn RoundTrainer>> {
    // one engine-worker per device: the worker's device is ordinal 0 of what it sees
    let be = ojas_learn::cuda::Cuda::new(0).context("starting the CUDA training backend")?;
    Ok(Box::new(GptTrainer::new(be, Backend::Cuda, spec, theta0)?))
}

#[cfg(not(feature = "cuda"))]
fn cuda(_: &TrainSpec, _: Option<&[f32]>) -> Result<Box<dyn RoundTrainer>> {
    bail!("this build cannot train on CUDA: build ojas-swarm with `--features cuda`")
}

/// Model kinds this build can train, for `WorkerCaps::trainable`.
pub fn trainable() -> Vec<String> {
    vec!["tiny_gpt".into()]
}

/// TinyGpt on one device.
pub struct GptTrainer<B: Device> {
    be: B,
    kind: Backend,
    spec: TrainSpec,
    model: TinyGpt<B>,
}

const STATE_MAGIC: &[u8; 4] = b"OJOS";
const STATE_VERSION: u32 = 1;

impl<B: Device> GptTrainer<B> {
    pub fn new(be: B, kind: Backend, spec: &TrainSpec, theta0: Option<&[f32]>) -> Result<Self> {
        let TrainModel::TinyGpt(cfg) = &spec.model;
        let model = TinyGpt::new(&be, cfg, theta0)?;
        let id = model.identity(&be)?;
        // the layout this build produces must be the run's: a member whose code builds
        // another schema (another version, another config) must not train for it
        ensure!(id.arch == spec.arch, "run {} expects arch {}, this build of tiny_gpt produces {}", spec.run, spec.arch.short(), id.arch.short());
        Ok(GptTrainer { be, kind, spec: spec.clone(), model })
    }

    fn snapshot(&self) -> (Vec<f32>, OptState) {
        (self.model.flatten(&self.be), self.model.opt_state(&self.be))
    }

    fn restore(&mut self, s: &(Vec<f32>, OptState)) -> Result<()> {
        self.model.unflatten(&self.be, &s.0)?;
        self.model.set_opt_state(&self.be, &s.1)
    }
}

impl<B: Device> RoundTrainer for GptTrainer<B> {
    fn identity(&self) -> Identity {
        self.model.identity(&self.be).expect("the model's own weights always match its layout")
    }

    fn n_params(&self) -> u64 {
        self.model.n_params() as u64
    }

    fn weights(&self) -> Vec<f32> {
        self.model.flatten(&self.be)
    }

    fn set_weights(&mut self, theta: &[f32]) -> Result<()> {
        self.model.unflatten(&self.be, theta)
    }

    fn round(&mut self, round: &RoundSpec, data: &[u8]) -> Result<(DeltaReport, Vec<f32>)> {
        let t0 = Instant::now();
        let start = self.snapshot();
        let id = tiny_gpt::identity(&self.model.cfg, &start.0)?;
        ensure!(id.model == round.base, "round {}: weights are {}, the round's base is {}", round.round, id.model.short(), round.base.short());
        let toks = data::Tokens::parse(data)?;
        let cfg = &self.model.cfg;
        ensure!(toks.vocab <= cfg.vocab, "token file vocab {} exceeds the model's {}", toks.vocab, cfg.vocab);
        let ctx = cfg.ctx as usize;
        let n_seqs = toks.n_seqs(ctx);
        ensure!(n_seqs > 0, "token file holds {} tokens, fewer than one window of {}", toks.n, ctx + 1);
        let (steps, batch) = (self.spec.inner_steps as usize, self.spec.batch as usize);
        let (first, stride) = data::stride_of(round.cursor, self.spec.shard, self.spec.n_shards);
        let (lr, wd) = (self.spec.inner_lr, self.spec.weight_decay);

        let mut buf = Vec::with_capacity(batch * (ctx + 1));
        let (mut sum, mut last) = (0.0f64, 0.0f32);
        for s in 0..steps {
            buf.clear();
            for j in 0..batch {
                let seq = data::seq_index(first, stride, (s * batch + j) as u64, n_seqs);
                toks.window(seq, ctx, &mut buf)?;
            }
            match self.model.step(&self.be, &buf, batch, lr, wd) {
                Ok(o) => {
                    sum += o.loss as f64;
                    last = o.loss;
                }
                Err(e) => {
                    // a failed round leaves the member exactly as it was: weights and moments
                    self.restore(&start).context("restoring the round's starting state")?;
                    return Err(e.context(format!("round {} inner step {s}", round.round)));
                }
            }
        }
        let end = self.model.flatten(&self.be);
        let delta: Vec<f32> = start.0.iter().zip(&end).map(|(a, b)| a - b).collect();
        if delta.iter().any(|v| !v.is_finite()) {
            self.restore(&start)?;
            bail!("round {}: delta is not finite", round.round);
        }
        let seqs = (steps * batch) as u64;
        let report = DeltaReport {
            run: self.spec.run.clone(),
            round: round.round,
            base: round.base,
            arch: id.arch,
            first_seq: first,
            stride,
            seqs,
            tokens: seqs * ctx as u64,
            mean_loss: (sum / steps as f64) as f32,
            final_loss: last,
            backend: self.kind,
            wall_ms: t0.elapsed().as_millis().min(u32::MAX as u128) as u32,
        };
        Ok((report, delta))
    }

    /// `"OJOS" | u32 version | u32 adam_step | u64 n | θ[n] | m[n] | v[n]`, little-endian f32.
    fn save_state(&self) -> Result<Vec<u8>> {
        let (theta, o) = self.snapshot();
        let mut out = Vec::with_capacity(20 + 12 * theta.len());
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&o.step.to_le_bytes());
        out.extend_from_slice(&(theta.len() as u64).to_le_bytes());
        for v in theta.iter().chain(&o.m).chain(&o.v) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        Ok(out)
    }

    fn load_state(&mut self, s: &[u8]) -> Result<()> {
        ensure!(s.len() >= 20 && &s[..4] == STATE_MAGIC, "not a trainer state");
        let u32_at = |o: usize| u32::from_le_bytes(s[o..o + 4].try_into().unwrap());
        ensure!(u32_at(4) == STATE_VERSION, "trainer state version {}", u32_at(4));
        let step = u32_at(8);
        let n = u64::from_le_bytes(s[12..20].try_into().unwrap()) as usize;
        ensure!(n == self.model.n_params() && s.len() == 20 + 12 * n, "trainer state for {n} params, model has {}", self.model.n_params());
        let f: Vec<f32> = s[20..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let snap = (f[..n].to_vec(), OptState { step, m: f[n..2 * n].to_vec(), v: f[2 * n..].to_vec() });
        self.restore(&snap)
    }
}

/// Token-file format for training data.
///
/// ```text
/// offset  size  field
///      0     4  magic "OJTK"
///      4     4  version (1)
///      8     4  vocab: every token is < vocab
///     12     4  width: bytes per token, 2 (u16) or 4 (u32)
///     16     8  n_tokens
///     24     .  n_tokens tokens, little-endian
/// ```
/// The file is exactly `24 + width * n_tokens` bytes; anything else is refused.
///
/// **Sequences.** Sequence `j` is the window of `ctx + 1` tokens starting at `j * ctx`
/// (inputs its first `ctx`, targets its last `ctx`): windows overlap by one token and
/// `n_seqs = (n_tokens - 1) / ctx`.
///
/// **Shards.** A member with shard `s` of `n_shards`, given `cursor` `c`, reads the
/// stream `g(k) = (c + k) * n_shards + s`, `k = 0, 1, ...`: `first_seq = c * n_shards
/// + s`, `stride = n_shards`, and row `b` of inner step `i` is `k = i * batch + b`. The
/// sequence read is `g(k) mod n_seqs` (wrapping is a new epoch). The coordinator sets
/// `c = round * inner_steps * batch`, so the members of one round read disjoint
/// sequences and nobody repeats one before the data wraps.
pub mod data {
    use anyhow::{ensure, Result};

    pub const MAGIC: &[u8; 4] = b"OJTK";
    pub const VERSION: u32 = 1;
    pub const HEADER: usize = 24;

    /// Encode `tokens` (all `< vocab`) as a token file.
    pub fn encode(vocab: u32, tokens: &[u32]) -> Result<Vec<u8>> {
        ensure!(vocab > 0 && tokens.iter().all(|&t| t < vocab), "token outside vocab {vocab}");
        let width: u32 = if vocab <= 1 << 16 { 2 } else { 4 };
        let mut out = Vec::with_capacity(HEADER + width as usize * tokens.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&vocab.to_le_bytes());
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for &t in tokens {
            if width == 2 {
                out.extend_from_slice(&(t as u16).to_le_bytes());
            } else {
                out.extend_from_slice(&t.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// A byte-level corpus: each byte is a token, vocab 256.
    pub fn from_bytes(text: &[u8]) -> Vec<u8> {
        let t: Vec<u32> = text.iter().map(|&b| b as u32).collect();
        encode(256, &t).expect("bytes are < 256")
    }

    /// Split a token file into (train, held-out) files: the last `frac` of the tokens
    /// are held out, contiguously, so no held-out window overlaps a training one.
    pub fn split_heldout(file: &[u8], frac: f64) -> Result<(Vec<u8>, Vec<u8>)> {
        ensure!(frac > 0.0 && frac < 1.0, "held-out fraction {frac} must be in (0, 1)");
        let t = Tokens::parse(file)?;
        let all: Vec<u32> = (0..t.n).map(|i| t.get(i)).collect();
        let cut = ((t.n as f64) * (1.0 - frac)).round() as usize;
        ensure!(cut >= 2 && t.n - cut >= 2, "{} tokens are too few to split", t.n);
        Ok((encode(t.vocab, &all[..cut])?, encode(t.vocab, &all[cut..])?))
    }

    /// A parsed view of a token file (no copy).
    pub struct Tokens<'a> {
        pub vocab: u32,
        pub width: usize,
        pub n: usize,
        body: &'a [u8],
    }

    impl<'a> Tokens<'a> {
        pub fn parse(file: &'a [u8]) -> Result<Tokens<'a>> {
            ensure!(file.len() >= HEADER && &file[..4] == MAGIC, "not a token file (magic)");
            let u32_at = |o: usize| u32::from_le_bytes(file[o..o + 4].try_into().unwrap());
            ensure!(u32_at(4) == VERSION, "token file version {}", u32_at(4));
            let (vocab, width) = (u32_at(8), u32_at(12) as usize);
            ensure!(width == 2 || width == 4, "token width {width}");
            ensure!(vocab > 0, "token file vocab 0");
            let n = u64::from_le_bytes(file[16..24].try_into().unwrap());
            let need = (n as u128) * width as u128 + HEADER as u128;
            ensure!(need == file.len() as u128, "token file is {} bytes, header says {need}", file.len());
            Ok(Tokens { vocab, width, n: n as usize, body: &file[HEADER..] })
        }

        pub fn get(&self, i: usize) -> u32 {
            let o = i * self.width;
            if self.width == 2 {
                u16::from_le_bytes([self.body[o], self.body[o + 1]]) as u32
            } else {
                u32::from_le_bytes(self.body[o..o + 4].try_into().unwrap())
            }
        }

        pub fn n_seqs(&self, ctx: usize) -> u64 {
            if self.n < 2 || ctx == 0 {
                0
            } else {
                ((self.n - 1) / ctx) as u64
            }
        }

        /// Append sequence `seq` (`ctx + 1` tokens) to `out`, checking each against vocab.
        pub fn window(&self, seq: u64, ctx: usize, out: &mut Vec<u32>) -> Result<()> {
            ensure!(seq < self.n_seqs(ctx), "sequence {seq} of {}", self.n_seqs(ctx));
            let s = seq as usize * ctx;
            for i in s..s + ctx + 1 {
                let t = self.get(i);
                ensure!(t < self.vocab, "token {t} at {i} outside vocab {}", self.vocab);
                out.push(t);
            }
            Ok(())
        }
    }

    /// `(first_seq, stride)` of shard `shard` of `n_shards` at `cursor`.
    pub fn stride_of(cursor: u64, shard: u32, n_shards: u32) -> (u64, u64) {
        (cursor * n_shards as u64 + shard as u64, n_shards as u64)
    }

    /// The sequence read `k`-th from `first` at `stride`, wrapping over `n_seqs`.
    ///
    /// The stream position is mapped through [`permute`] before it names a window: a
    /// corpus is concatenated files, so reading it in file order hands one round a
    /// single file (measured: two rounds of a code corpus spent inside one generated
    /// table, and held-out loss went 3.94 → 7.25 while the members' own loss fell).
    /// The permutation is a bijection, so shards stay disjoint and no window repeats
    /// before the data wraps; it depends only on `n_seqs`, so every member agrees.
    pub fn seq_index(first: u64, stride: u64, k: u64, n_seqs: u64) -> u64 {
        permute(((first as u128 + stride as u128 * k as u128) % n_seqs as u128) as u64, n_seqs)
    }

    /// `i -> (a*i + b) mod n` with `gcd(a, n) = 1`: a fixed shuffle of `0..n` that needs no
    /// table. `a` is near `n/φ`, which spreads neighbours far apart.
    pub fn permute(i: u64, n: u64) -> u64 {
        if n <= 2 {
            return i % n.max(1);
        }
        fn gcd(mut a: u64, mut b: u64) -> u64 {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        }
        let mut a = ((n as f64 * 0.618_033_988_75) as u64).max(1);
        while gcd(a, n) != 1 {
            a += 1;
        }
        ((a as u128 * i as u128 + (n / 3) as u128) % n as u128) as u64
    }
}
