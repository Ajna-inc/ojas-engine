//! TinyGpt: a small causal transformer on the tape, the model a DiLoCo swarm run trains.
//!
//! GPT-2 layout: token + learned positional embedding, pre-LN blocks (causal multi-head
//! attention, GELU MLP), final LN, LM head **tied** to the token embedding (it halves the
//! embedding parameters and is what GPT-2 does; the head's gradient and the lookup's
//! gradient accumulate on the same leaf). Mean cross-entropy over next tokens.
//!
//! Everything is composed from existing tape primitives, so the same code trains on the
//! CPU reference, Metal and CUDA with nothing new to port:
//! * embedding lookup = `gather_rows` on the [1, V, D] view of the embedding;
//! * attention = matmul / permute / softmax with an additive causal mask (a constant
//!   input, −1e9 above the diagonal: finite, so a fully masked row cannot produce
//!   `inf − inf`);
//! * cross-entropy = −mean log softmax(logits)[target], via `gather_last` on the
//!   softmax. The target probability is floored at 1e-30 before the log so an
//!   underflowed probability gives a large finite loss rather than −log 0.
//!
//! Parameters live in a canonical order: sorted by name. `flatten`, `identity` and the
//! wire vector all use it, so θ means the same thing on every member.

use std::collections::HashMap;

use anyhow::{bail, ensure, Result};
use ojas_swarm_proto::{Identity, IdentityBuilder, TinyGptConfig};

use crate::backend::{Backend, Binary, Unary};
use crate::tape::{AdamW, Param, Tape, Var};

pub const ARCH: &str = "tiny_gpt";
const LN_EPS: f32 = 1e-5;
const MASK: f32 = -1e9;
const P_FLOOR: f32 = 1e-30;

fn dims(c: &TinyGptConfig) -> (usize, usize, usize, usize, usize, usize) {
    (c.vocab as usize, c.ctx as usize, c.d_model as usize, c.n_layers as usize, c.n_heads as usize, c.d_ff as usize)
}

pub fn check_config(c: &TinyGptConfig) -> Result<()> {
    let (v, t, d, l, h, f) = dims(c);
    ensure!(v >= 2 && t >= 1 && d >= 1 && l >= 1 && h >= 1 && f >= 1, "tiny_gpt: every dimension must be positive: {c:?}");
    ensure!(d % h == 0, "tiny_gpt: d_model {d} not divisible by n_heads {h}");
    // the [1, V, D] embedding view and the gather index ride as f32: keep them exact
    ensure!(v < (1 << 24) && t < (1 << 24), "tiny_gpt: vocab/ctx too large for f32 indices");
    Ok(())
}

/// Every trainable tensor, name and shape, in canonical (sorted-name) order.
pub fn param_specs(c: &TinyGptConfig) -> Vec<(String, Vec<usize>)> {
    let (v, t, d, l, _, f) = dims(c);
    let mut s: Vec<(String, Vec<usize>)> = vec![
        ("tok_emb.weight".into(), vec![v, d]),
        ("pos_emb.weight".into(), vec![t, d]),
        ("ln_f.weight".into(), vec![d]),
        ("ln_f.bias".into(), vec![d]),
    ];
    for i in 0..l {
        let p = |n: &str| format!("blk.{i}.{n}");
        for ln in ["ln1", "ln2"] {
            s.push((p(&format!("{ln}.weight")), vec![d]));
            s.push((p(&format!("{ln}.bias")), vec![d]));
        }
        for w in ["q", "k", "v", "o"] {
            s.push((p(&format!("attn.{w}.weight")), vec![d, d]));
            s.push((p(&format!("attn.{w}.bias")), vec![d]));
        }
        s.push((p("mlp.up.weight"), vec![f, d]));
        s.push((p("mlp.up.bias"), vec![f]));
        s.push((p("mlp.down.weight"), vec![d, f]));
        s.push((p("mlp.down.bias"), vec![d]));
    }
    s.sort_by(|a, b| a.0.cmp(&b.0));
    s
}

pub fn n_params(c: &TinyGptConfig) -> usize {
    param_specs(c).iter().map(|(_, s)| s.iter().product::<usize>()).sum()
}

/// Decoupled weight decay applies to matrices (linears and the embeddings), never to
/// LayerNorm gains or biases: decaying a gain toward 0 fights the normalisation.
pub fn decays(name: &str) -> bool {
    name.ends_with(".weight") && !name.contains(".ln") && !name.starts_with("ln_")
}

/// SplitMix64: tiny, seedable, the same on every platform.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// uniform in (0, 1]
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

/// θ0 from `init_seed`, flattened in canonical order. GPT-2 init: N(0, 0.02) for
/// embeddings and linears, the two residual projections (`attn.o`, `mlp.down`) scaled
/// by 1/√(2L) so the residual stream's variance does not grow with depth; biases 0,
/// LN gains 1. Each tensor draws from its own stream (seed ⊕ hash(name)), so one
/// tensor's values do not depend on the others' order or sizes. Computed in f64 on the
/// host and rounded once: every backend uploads bit-identical θ0.
pub fn init(c: &TinyGptConfig) -> Vec<f32> {
    let l = c.n_layers as f64;
    let mut out = Vec::with_capacity(n_params(c));
    for (name, shape) in param_specs(c) {
        let n: usize = shape.iter().product();
        if name.ends_with(".bias") {
            out.extend(std::iter::repeat_n(0.0, n));
        } else if name.contains("ln") {
            out.extend(std::iter::repeat_n(1.0, n));
        } else {
            let std = if name.ends_with("attn.o.weight") || name.ends_with("mlp.down.weight") { 0.02 / (2.0 * l).sqrt() } else { 0.02 };
            let mut r = Rng(c.init_seed ^ fnv(&name));
            out.extend((0..n).map(|_| (r.normal() * std) as f32));
        }
    }
    out
}

/// Layout + content identity of θ (`arch` "tiny_gpt", dims from the config, every
/// tensor as name / "f32" / shape / little-endian bytes).
pub fn identity(c: &TinyGptConfig, flat: &[f32]) -> Result<Identity> {
    let specs = param_specs(c);
    let n: usize = specs.iter().map(|(_, s)| s.iter().product::<usize>()).sum();
    ensure!(flat.len() == n, "tiny_gpt identity: {} values for {n} parameters", flat.len());
    let mut b = IdentityBuilder::new(ARCH)
        .dim("vocab", c.vocab as u64)
        .dim("ctx", c.ctx as u64)
        .dim("d_model", c.d_model as u64)
        .dim("n_layers", c.n_layers as u64)
        .dim("n_heads", c.n_heads as u64)
        .dim("d_ff", c.d_ff as u64);
    let mut off = 0;
    let mut bytes = Vec::new();
    for (name, shape) in &specs {
        let k: usize = shape.iter().product();
        bytes.clear();
        flat[off..off + k].iter().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
        let sh: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        b.tensor(name, "f32", &sh, &bytes);
        off += k;
    }
    Ok(b.finish())
}

/// Mean next-token cross-entropy of `batch` windows of `t + 1` tokens each (`tokens` row-major,
/// `t ≤ ctx`), with `w` the parameters as tape vars in canonical order.
pub fn forward_loss<B: Backend>(t: &mut Tape<'_, B>, c: &TinyGptConfig, w: &[Var], tokens: &[u32], batch: usize) -> Result<Var> {
    let (v, ctx, d, layers, heads, _) = dims(c);
    let specs = param_specs(c);
    ensure!(w.len() == specs.len(), "tiny_gpt: {} parameter vars for {} tensors", w.len(), specs.len());
    ensure!(batch > 0 && tokens.len() % batch == 0, "tiny_gpt: {} tokens in {batch} rows", tokens.len());
    let win = tokens.len() / batch;
    ensure!(win >= 2 && win - 1 <= ctx, "tiny_gpt: window {win} must be 2..={}", ctx + 1);
    if let Some(&bad) = tokens.iter().find(|&&x| x as usize >= v) {
        bail!("tiny_gpt: token {bad} outside vocab {v}");
    }
    let tl = win - 1;
    let idx: HashMap<&str, usize> = specs.iter().enumerate().map(|(i, (n, _))| (n.as_str(), i)).collect();
    let p = |n: &str| w[idx[n]];
    let (inp, tgt): (Vec<usize>, Vec<usize>) = (0..batch * tl).map(|i| (tokens[(i / tl) * win + i % tl] as usize, tokens[(i / tl) * win + i % tl + 1] as usize)).unzip();

    let emb = p("tok_emb.weight");
    let emb3 = t.reshape(emb, &[1, v, d])?;
    let x = t.gather_rows(emb3, &inp, batch * tl)?;
    let x = t.reshape(x, &[batch, tl, d])?;
    let pos = p("pos_emb.weight");
    let pos = if tl < ctx { t.slice(pos, 0, 0, tl)? } else { pos };
    let mut x = t.add(x, pos)?;

    let mask: Vec<f32> = (0..tl * tl).map(|i| if i % tl > i / tl { MASK } else { 0.0 }).collect();
    let mask = t.input(&mask, &[tl, tl]);
    let hd = d / heads;
    let heads_of = |t: &mut Tape<'_, B>, y: Var| -> Result<Var> {
        let y = t.reshape(y, &[batch, tl, heads, hd])?;
        t.permute(y, &[0, 2, 1, 3])
    };
    for i in 0..layers {
        let n = |s: &str| format!("blk.{i}.{s}");
        let h = t.layer_norm(x, p(&n("ln1.weight")), p(&n("ln1.bias")), LN_EPS);
        let q = t.linear(h, p(&n("attn.q.weight")), Some(p(&n("attn.q.bias"))))?;
        let k = t.linear(h, p(&n("attn.k.weight")), Some(p(&n("attn.k.bias"))))?;
        let vv = t.linear(h, p(&n("attn.v.weight")), Some(p(&n("attn.v.bias"))))?;
        let (q, k, vv) = (heads_of(t, q)?, heads_of(t, k)?, heads_of(t, vv)?);
        let s = t.matmul_opts(q, k, true)?;
        let s = t.scale(s, 1.0 / (hd as f32).sqrt())?;
        let s = t.add(s, mask)?;
        let a = t.softmax(s);
        let a = t.matmul(a, vv)?;
        let a = t.permute(a, &[0, 2, 1, 3])?;
        let a = t.reshape(a, &[batch, tl, d])?;
        let a = t.linear(a, p(&n("attn.o.weight")), Some(p(&n("attn.o.bias"))))?;
        x = t.add(x, a)?;
        let h = t.layer_norm(x, p(&n("ln2.weight")), p(&n("ln2.bias")), LN_EPS);
        let f = t.linear(h, p(&n("mlp.up.weight")), Some(p(&n("mlp.up.bias"))))?;
        let f = t.gelu(f);
        let f = t.linear(f, p(&n("mlp.down.weight")), Some(p(&n("mlp.down.bias"))))?;
        x = t.add(x, f)?;
    }
    let x = t.layer_norm(x, p("ln_f.weight"), p("ln_f.bias"), LN_EPS);
    let logits = t.linear(x, emb, None)?;
    let prob = t.softmax(logits);
    let pt = t.gather_last(prob, &tgt, 1)?;
    let floor = t.input(&[P_FLOOR], &[1]);
    let pt = t.binary(Binary::Max, pt, floor)?;
    let lp = t.unary(Unary::Log, pt);
    Ok(t.sum_scaled(lp, -1.0 / (batch * tl) as f32))
}

/// One accepted optimizer step.
#[derive(Clone, Copy, Debug)]
pub struct StepOut {
    pub loss: f32,
    /// global L2 norm of the gradient before clipping
    pub grad_norm: f32,
}

/// The model resident on a backend, with its AdamW state.
pub struct TinyGpt<B: Backend> {
    pub cfg: TinyGptConfig,
    pub params: Vec<Param<B>>,
    pub opt: AdamW,
    /// global-norm gradient clip (GPT practice; ≤ 0 disables)
    pub clip: f32,
}

/// Inner optimizer state, host side: what a member checkpoints so a restart does not
/// reset AdamW's moments.
#[derive(Clone, Debug, PartialEq)]
pub struct OptState {
    pub step: u32,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl<B: Backend> TinyGpt<B> {
    /// `theta` in canonical order (`None`: [`init`] from the config's seed).
    pub fn new(be: &B, cfg: &TinyGptConfig, theta: Option<&[f32]>) -> Result<Self> {
        check_config(cfg)?;
        let owned;
        let theta = match theta {
            Some(t) => t,
            None => {
                owned = init(cfg);
                &owned
            }
        };
        let specs = param_specs(cfg);
        let n: usize = specs.iter().map(|(_, s)| s.iter().product::<usize>()).sum();
        ensure!(theta.len() == n, "tiny_gpt: θ has {} values, the config needs {n}", theta.len());
        ensure!(theta.iter().all(|v| v.is_finite()), "tiny_gpt: θ is not finite");
        let mut off = 0;
        let params = specs
            .into_iter()
            .map(|(name, shape)| {
                let k: usize = shape.iter().product();
                off += k;
                Param::new(be, name, &shape, &theta[off - k..off])
            })
            .collect();
        Ok(TinyGpt { cfg: cfg.clone(), params, opt: AdamW { wd: 0.0, ..AdamW::default() }, clip: 1.0 })
    }

    pub fn n_params(&self) -> usize {
        self.params.iter().map(|p| p.shape.iter().product::<usize>()).sum()
    }

    pub fn flatten(&self, be: &B) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_params());
        self.params.iter().for_each(|p| out.extend(be.download(&p.val)));
        out
    }

    /// Replace the weights; optimizer state is kept.
    pub fn unflatten(&mut self, be: &B, theta: &[f32]) -> Result<()> {
        ensure!(theta.len() == self.n_params(), "tiny_gpt: θ has {} values, the model {}", theta.len(), self.n_params());
        ensure!(theta.iter().all(|v| v.is_finite()), "tiny_gpt: θ is not finite");
        let mut off = 0;
        for p in &mut self.params {
            let k: usize = p.shape.iter().product();
            p.val = std::rc::Rc::new(be.upload(&theta[off..off + k]));
            off += k;
        }
        Ok(())
    }

    pub fn identity(&self, be: &B) -> Result<Identity> {
        identity(&self.cfg, &self.flatten(be))
    }

    pub fn opt_state(&self, be: &B) -> OptState {
        let (mut m, mut v) = (Vec::new(), Vec::new());
        for p in &self.params {
            m.extend(be.download(&p.m));
            v.extend(be.download(&p.v));
        }
        OptState { step: self.opt.step, m, v }
    }

    pub fn set_opt_state(&mut self, be: &B, s: &OptState) -> Result<()> {
        let n = self.n_params();
        ensure!(s.m.len() == n && s.v.len() == n, "tiny_gpt: optimizer state for {} params, model has {n}", s.m.len());
        ensure!(s.m.iter().chain(&s.v).all(|x| x.is_finite()), "tiny_gpt: optimizer state is not finite");
        let mut off = 0;
        for p in &mut self.params {
            let k: usize = p.shape.iter().product();
            p.m = be.upload(&s.m[off..off + k]);
            p.v = be.upload(&s.v[off..off + k]);
            off += k;
        }
        self.opt.step = s.step;
        Ok(())
    }

    /// Mean cross-entropy without a backward pass.
    pub fn loss(&self, be: &B, tokens: &[u32], batch: usize) -> Result<f32> {
        let mut t = Tape::new(be);
        let w: Vec<Var> = self.params.iter().map(|p| t.param(p)).collect();
        let l = forward_loss(&mut t, &self.cfg, &w, tokens, batch)?;
        Ok(t.value(l)[0])
    }

    /// Loss and gradients (canonical order, host side): for tests and backend comparison.
    pub fn loss_and_grads(&self, be: &B, tokens: &[u32], batch: usize) -> Result<(f32, Vec<f32>)> {
        let mut t = Tape::new(be);
        let w: Vec<Var> = self.params.iter().map(|p| t.param(p)).collect();
        let l = forward_loss(&mut t, &self.cfg, &w, tokens, batch)?;
        t.backward(l)?;
        let mut g = Vec::with_capacity(self.n_params());
        for (p, v) in self.params.iter().zip(&w) {
            g.extend(t.grad_vec(*v).unwrap_or_else(|| vec![0.0; p.shape.iter().product()]));
        }
        Ok((t.value(l)[0], g))
    }

    /// One AdamW step at `lr` with decoupled weight decay `wd` (matrices only, see
    /// [`decays`]). The loss and the gradient are checked finite before anything touches
    /// the weights; a refused step leaves weights and moments untouched. The weights
    /// are checked after the update too, but undoing that is the caller's job (it
    /// holds the snapshot).
    pub fn step(&mut self, be: &B, tokens: &[u32], batch: usize, lr: f32, wd: f32) -> Result<StepOut> {
        let mut t = Tape::new(be);
        let w: Vec<Var> = self.params.iter().map(|p| t.param(p)).collect();
        let l = forward_loss(&mut t, &self.cfg, &w, tokens, batch)?;
        let loss = t.value(l)[0];
        ensure!(loss.is_finite(), "tiny_gpt: non-finite loss {loss}");
        t.backward(l)?;
        // one device reduction and one download for the whole gradient; NaN and Inf
        // both survive a sum of squares
        let acc = be.alloc(1);
        for v in &w {
            match t.grad(*v) {
                Some(g) => be.sumsq(g, &acc, true),
                None => bail!("tiny_gpt: a parameter got no gradient"),
            }
        }
        let norm = (be.download(&acc)[0] as f64).sqrt() as f32;
        ensure!(norm.is_finite(), "tiny_gpt: non-finite gradient (norm {norm})");
        let s = if self.clip > 0.0 && norm > self.clip { self.clip / norm } else { 1.0 };
        self.opt.lr = lr;
        self.opt.begin();
        for (p, v) in self.params.iter().zip(&w) {
            let g = t.grad(*v).unwrap();
            if s < 1.0 {
                be.scale(g, s);
            }
            self.opt.update(be, p, g, if decays(&p.name) { wd } else { 0.0 });
        }
        drop(t);
        let acc = be.alloc(1);
        self.params.iter().for_each(|p| be.sumsq(&p.val, &acc, true));
        ensure!(be.download(&acc)[0].is_finite(), "tiny_gpt: weights went non-finite");
        Ok(StepOut { loss, grad_norm: norm })
    }
}
