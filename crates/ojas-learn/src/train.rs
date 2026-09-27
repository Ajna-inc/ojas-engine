//! DETR training loop pieces: AdamW parameter groups (the IISc / RT-DETRv2 recipe, or DEIM's),
//! linear warmup or DEIM's flat-cosine schedule, global-norm gradient clipping, the model EMA
//! and safetensors checkpoints. [`Trainer`] trains RT-DETRv2; [`DfineTrainer`] trains D-FINE
//! with the DEIM loss.

use anyhow::Result;

use crate::backend::Backend;
use crate::models::deim_loss::DeimCriterion;
use crate::models::detr_loss::{denoising, Criterion, DnGroup, Rng, Target};
use crate::models::dfine::{Dfine, DfineConfig};
use crate::models::rtdetr::{Config, RtDetr, Store, Variant};
use crate::tape::{BnRunning, Param, Tape};

#[derive(Clone, Debug)]
pub struct TrainCfg {
    pub lr: f32,
    /// backbone weights that are not norms
    pub lr_backbone: f32,
    pub weight_decay: f32,
    pub betas: (f32, f32),
    pub warmup_steps: u64,
    pub clip: f32,
    pub ema_decay: f32,
    pub ema_warmup: f32,
    pub num_denoising: usize,
    pub label_noise: f32,
    pub box_noise: f32,
    /// None: linear warm-up to a constant lr (RT-DETRv2); Some: DEIM's flat-cosine
    pub schedule: Option<FlatCosine>,
    /// parameter-group rule
    pub groups: Groups,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Groups {
    /// `^(?=.*backbone)(?!.*norm|bn).*$` → lr_backbone; encoder/decoder norms → no decay
    RtDetrV2,
    /// `^(?=.*backbone)(?!.*bn).*$` → lr_backbone; `^(?=.*(?:norm|bn)).*$` → no decay
    Deim,
}

/// DEIM's `flat_cosine_schedule`: quadratic warm-up, flat until `flat`, cosine down to
/// `gamma`·lr, held there for the last `no_aug` iterations. All in iterations.
#[derive(Clone, Copy, Debug)]
pub struct FlatCosine {
    pub total: u64,
    pub warmup: u64,
    pub flat: u64,
    pub no_aug: u64,
    pub gamma: f32,
}

impl FlatCosine {
    /// From epochs, as FlatCosineLRScheduler builds it.
    pub fn from_epochs(iters_per_epoch: u64, epochs: u64, warmup_iter: u64, flat_epochs: u64, no_aug_epochs: u64, gamma: f32) -> Self {
        FlatCosine { total: iters_per_epoch * epochs, warmup: warmup_iter, flat: iters_per_epoch * flat_epochs, no_aug: iters_per_epoch * no_aug_epochs, gamma }
    }

    /// The lr multiplier at iteration `it` (0-based): lr = base · factor.
    pub fn factor(&self, it: u64) -> f32 {
        let min = self.gamma;
        if it <= self.warmup {
            (it as f32 / self.warmup.max(1) as f32).powi(2)
        } else if it <= self.flat {
            1.0
        } else if it >= self.total - self.no_aug {
            min
        } else {
            let x = (it - self.flat) as f64 / (self.total - self.flat - self.no_aug) as f64;
            min + (1.0 - min) * (0.5 * (1.0 + (std::f64::consts::PI * x).cos())) as f32
        }
    }
}

impl TrainCfg {
    /// IISc's UVH-26 RT-DETRv2-S settings.
    pub fn rtdetrv2_uvh26() -> Self {
        TrainCfg { lr: 1e-4, lr_backbone: 1e-5, weight_decay: 1e-4, betas: (0.9, 0.999), warmup_steps: 2000, clip: 0.1, ema_decay: 0.9999, ema_warmup: 2000.0, num_denoising: 100, label_noise: 0.5, box_noise: 1.0, schedule: None, groups: Groups::RtDetrV2 }
    }

    /// `training/deim/step3_ojas_n32.yml` (batch 48): lr 6e-4, backbone 3e-4, wd 1e-4, clip 0.1,
    /// warm-up 1333 iterations, EMA 0.99985 with 667 warm-up updates. The schedule is flat-cosine
    /// with `lr_gamma: 1.0`, inherited from DEIM's `deim_hgnetv2_n_coco.yml` — min lr = base lr,
    /// so after the warm-up the lr is constant (the PyTorch run logs base = min for every group).
    pub fn deim_ojas_n32(iters_per_epoch: u64) -> Self {
        TrainCfg {
            lr: 6e-4,
            lr_backbone: 3e-4,
            weight_decay: 1e-4,
            betas: (0.9, 0.999),
            warmup_steps: 1333,
            clip: 0.1,
            ema_decay: 0.99985,
            ema_warmup: 667.0,
            num_denoising: 100,
            label_noise: 0.5,
            box_noise: 1.0,
            schedule: Some(FlatCosine::from_epochs(iters_per_epoch, 72, 1333, 40, 8, 1.0)),
            groups: Groups::Deim,
        }
    }

    /// The lr multiplier at iteration `it`.
    pub fn lr_factor(&self, it: u64) -> f32 {
        match &self.schedule {
            Some(s) => s.factor(it),
            None => ((it + 1) as f32 / self.warmup_steps.max(1) as f32).min(1.0),
        }
    }

    /// (lr, weight decay) of a parameter, by the config's regex groups:
    /// `^(?=.*backbone)(?!.*norm|bn).*$` → lr_backbone;
    /// `^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$` → no weight decay.
    pub fn group(&self, name: &str) -> (f32, f32) {
        let norm = name.contains("norm") || name.contains("bn");
        if self.groups == Groups::Deim {
            return if name.contains("backbone") && !name.contains("bn") {
                (self.lr_backbone, self.weight_decay)
            } else if norm {
                (self.lr, 0.0)
            } else {
                (self.lr, self.weight_decay)
            };
        }
        if name.contains("backbone") && !name.contains("norm") {
            (self.lr_backbone, self.weight_decay)
        } else if (name.contains("encoder") || name.contains("decoder")) && norm {
            (self.lr, 0.0)
        } else {
            (self.lr, self.weight_decay)
        }
    }
}

impl<B: Backend> Store<B> {
    /// A device copy (the EMA model).
    pub fn duplicate(&self, be: &B) -> Self {
        let copy = |x: &B::Buf| {
            let y = be.alloc(be.len(x));
            be.axpby(x, &y, 1.0, 0.0);
            y
        };
        Store {
            params: self.params.iter().map(|(k, p)| (k.clone(), Param { name: p.name.clone(), shape: p.shape.clone(), val: std::rc::Rc::new(copy(&p.val)), m: be.alloc(1), v: be.alloc(1) })).collect(),
            bns: self.bns.iter().map(|(k, r)| (k.clone(), BnRunning { mean: copy(&r.mean), var: copy(&r.var), momentum: r.momentum, eps: r.eps })).collect(),
            consts: self.consts.clone(),
        }
    }

    /// Every float state tensor by name: parameters and BN running statistics.
    pub fn named_state(&self) -> Vec<(String, &B::Buf, Vec<usize>)> {
        let mut v: Vec<(String, &B::Buf, Vec<usize>)> = self.params.iter().map(|(k, p)| (k.clone(), &*p.val, p.shape.clone())).collect();
        for (k, r) in &self.bns {
            v.push((format!("{k}.running_mean"), &r.mean, vec![]));
            v.push((format!("{k}.running_var"), &r.var, vec![]));
        }
        v
    }
}

/// ModelEMA: every float state tensor, decay·ema + (1 − decay)·model, with
/// decay = d·(1 − exp(−updates / warmups)).
pub struct Ema<B: Backend> {
    pub store: Store<B>,
    pub updates: u64,
}

impl<B: Backend> Ema<B> {
    pub fn new(be: &B, st: &Store<B>) -> Self {
        Ema { store: st.duplicate(be), updates: 0 }
    }

    pub fn update(&mut self, be: &B, st: &Store<B>, decay: f32, warmup: f32) {
        self.updates += 1;
        let d = decay * (1.0 - (-(self.updates as f32) / warmup).exp());
        for (k, p) in &st.params {
            be.axpby(&p.val, &self.store.params[k].val, 1.0 - d, d);
        }
        for (k, r) in &st.bns {
            let e = &self.store.bns[k];
            be.axpby(&r.mean, &e.mean, 1.0 - d, d);
            be.axpby(&r.var, &e.var, 1.0 - d, d);
        }
    }
}

/// After the backward: global-norm clip, AdamW per parameter group at the scheduled lr, EMA.
/// `step` is the 1-based optimizer step. Returns (gradient norm, lr factor).
fn apply_update<B: Backend>(be: &B, st: &Store<B>, ema: &mut Ema<B>, t: &Tape<B>, cfg: &TrainCfg, step: u64, frozen: &[String]) -> (f32, f32) {
    let acc = be.alloc(1);
    let grads: Vec<(&String, &Param<B>, &B::Buf)> = st.params.iter().filter_map(|(k, p)| t.param_var(p).and_then(|v| t.grad(v)).map(|g| (k, p, g))).collect();
    for (_, _, g) in &grads {
        be.sumsq(g, &acc, true);
    }
    let norm = be.download(&acc)[0].sqrt();
    if cfg.clip > 0.0 && norm > cfg.clip {
        let c = cfg.clip / (norm + 1e-6);
        for (_, _, g) in &grads {
            be.scale(g, c);
        }
    }
    let factor = cfg.lr_factor(step - 1);
    for (k, p, g) in &grads {
        if frozen.iter().any(|f| k.starts_with(f.as_str())) {
            continue;
        }
        let (lr, wd) = cfg.group(k);
        be.adamw(&p.val, g, &p.m, &p.v, lr * factor, cfg.betas.0, cfg.betas.1, 1e-8, wd, step as u32);
    }
    ema.update(be, st, cfg.ema_decay, cfg.ema_warmup);
    (norm, factor)
}

/// The denoising queries on the tape: class embeddings (padding slots zeroed, as
/// nn.Embedding(padding_idx) reads a zero row and gives it no gradient), noised boxes, mask.
fn dn_inputs<B: Backend>(t: &mut Tape<B>, st: &Store<B>, g: &DnGroup, b: usize, hidden: usize, num_queries: usize) -> Result<(crate::tape::Var, crate::tape::Var, crate::tape::Var)> {
    let d = g.num_dn;
    let emb = t.param(&st.params["decoder.denoising_class_embed.weight"]);
    let rows = t.shape(emb)[0];
    let emb = t.reshape(emb, &[1, rows, hidden])?;
    let content = t.gather_rows(emb, &g.classes, b * d)?;
    let content = t.reshape(content, &[b, d, hidden])?;
    let pad = t.input(&g.pad, &[b, d, 1]);
    let content = t.mul(content, pad)?;
    let boxes = t.input(&g.boxes_unact, &[b, d, 4]);
    let n = d + num_queries;
    let mask = t.input(&g.mask, &[n, n]);
    Ok((content, boxes, mask))
}

fn save_stores<B: Backend>(be: &B, stores: [(&str, &Store<B>); 2], path: &std::path::Path) -> Result<()> {
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![];
    for (prefix, st) in stores {
        for (k, buf, shape) in st.named_state() {
            let data = be.download(buf);
            let shape = if shape.is_empty() { vec![data.len()] } else { shape };
            tensors.push((format!("{prefix}.{k}"), shape, data));
        }
        for (k, (shape, data)) in &st.consts {
            tensors.push((format!("{prefix}.{k}"), shape.clone(), data.clone()));
        }
    }
    ojas_formats::safetensors::write_f32(path, &tensors)
}

/// D-FINE (HGNetv2 + D-FINE encoder / decoder) with the DEIM loss.
pub struct DfineTrainer<'b, B: Backend> {
    pub be: &'b B,
    pub cfg: TrainCfg,
    pub model: DfineConfig,
    pub st: Store<B>,
    pub ema: Ema<B>,
    pub crit: DeimCriterion,
    /// optimizer steps taken
    pub step: u64,
    pub rng: Rng,
}

impl<'b, B: Backend> DfineTrainer<'b, B> {
    pub fn new(be: &'b B, model: DfineConfig, st: Store<B>, cfg: TrainCfg, seed: u64) -> Self {
        let ema = Ema::new(be, &st);
        let crit = DeimCriterion::deim(model.num_classes);
        DfineTrainer { be, cfg, model, st, ema, crit, step: 0, rng: Rng::new(seed) }
    }

    /// DEIM's EMA restart at the augmentation stop: a new decay, the warm-up ramp kept.
    pub fn restart_ema(&mut self, decay: f32) {
        self.cfg.ema_decay = decay;
    }

    /// One optimisation step on a batch [B, 3, H, W].
    pub fn step(&mut self, images: &[f32], targets: &[Target], hw: [usize; 2]) -> Result<StepLog> {
        let be = self.be;
        let b = targets.len();
        let m = Dfine { cfg: self.model.clone(), st: &self.st, train: true };
        let mut t = Tape::new(be);
        let x = t.input(images, &[b, 3, hw[0], hw[1]]);
        // D-FINE draws no denoising group for a batch without boxes
        let dn = if targets.iter().all(|t| t.labels.is_empty()) {
            None
        } else {
            denoising(targets, self.model.num_classes, self.model.num_queries, self.cfg.num_denoising, self.cfg.label_noise, self.cfg.box_noise, &mut self.rng)
        };
        let dn_vars = match &dn {
            Some(g) => Some(dn_inputs(&mut t, &self.st, g, b, self.model.hidden, self.model.num_queries)?),
            None => None,
        };
        let out = m.forward(&mut t, x, dn_vars)?;
        let (total, terms) = self.crit.forward(&mut t, &out, targets, dn.as_ref())?;
        for (_, v) in &terms {
            t.keep(*v);
        }
        t.backward(total)?;
        self.step += 1;
        let (norm, factor) = apply_update(be, &self.st, &mut self.ema, &t, &self.cfg, self.step, &[]);
        let terms: Vec<(String, f32)> = terms.iter().map(|(n, v)| (n.clone(), t.value(*v)[0])).collect();
        Ok(StepLog { updated: true, loss: t.value(total)[0], terms, grad_norm: norm, lr: self.cfg.lr * factor })
    }

    /// Save model + EMA state as safetensors ("model.<name>", "ema.<name>").
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        save_stores(self.be, [("model", &self.st), ("ema", &self.ema.store)], path)
    }
}

pub struct StepLog {
    /// false: the surprise gate skipped the backward and the update
    pub updated: bool,
    pub loss: f32,
    pub terms: Vec<(String, f32)>,
    pub grad_norm: f32,
    pub lr: f32,
}

pub struct Trainer<'b, B: Backend> {
    pub be: &'b B,
    pub cfg: TrainCfg,
    pub model: Config,
    pub st: Store<B>,
    pub ema: Ema<B>,
    pub crit: Criterion,
    pub step: u64,
    pub rng: Rng,
    pub var: Variant,
    /// surprise gate on the batch loss (update only when loss > μ + kσ)
    pub gate: Option<ojas_core::Gate>,
    /// control for the gate: skip each update with this probability, at random
    pub random_skip: Option<f32>,
    pub updates: u64,
}

impl<'b, B: Backend> Trainer<'b, B> {
    pub fn new(be: &'b B, model: Config, st: Store<B>, cfg: TrainCfg, seed: u64) -> Self {
        let ema = Ema::new(be, &st);
        let crit = Criterion::rtdetrv2(model.num_classes);
        Trainer { be, cfg, model, st, ema, crit, step: 0, rng: Rng::new(seed), var: Variant::default(), gate: None, random_skip: None, updates: 0 }
    }

    /// One optimisation step on a batch [B, 3, H, W].
    pub fn step(&mut self, images: &[f32], targets: &[Target], hw: [usize; 2]) -> Result<StepLog> {
        let be = self.be;
        let timing = std::env::var("OJAS_LEARN_TIMING").is_ok();
        let t_start = std::time::Instant::now();
        let mark = |what: &str, sync: &dyn Fn()| {
            if timing {
                sync();
                eprintln!("  {what:10} {:6.1} ms", t_start.elapsed().as_secs_f64() * 1e3);
            }
        };
        let b = targets.len();
        let m = RtDetr { cfg: self.model.clone(), st: &self.st, train: true, var: self.var.clone() };
        let mut t = Tape::new(be);
        let x = t.input(images, &[b, 3, hw[0], hw[1]]);
        // contrastive denoising queries
        let dn = denoising(targets, self.model.num_classes, self.model.num_queries, self.cfg.num_denoising, self.cfg.label_noise, self.cfg.box_noise, &mut self.rng);
        let dn_vars = match &dn {
            Some(g) => Some(dn_inputs(&mut t, &self.st, g, b, self.model.hidden, self.model.num_queries)?),
            None => None,
        };
        let out = m.forward(&mut t, x, dn_vars)?;
        mark("forward", &|| { be.download(&be.alloc(1)); });
        let (total, terms) = self.crit.forward(&mut t, &out, targets, dn.as_ref())?;
        mark("criterion", &|| { be.download(&be.alloc(1)); });
        for (_, v) in &terms {
            t.keep(*v);
        }
        let loss_now = t.value(total)[0];
        let skip = match (self.gate.as_mut(), self.random_skip) {
            (Some(g), _) => !g.observe(loss_now),
            (None, Some(p)) => self.rng.uniform() < p,
            _ => false,
        };
        {
            if skip {
                self.step += 1;
                let terms: Vec<(String, f32)> = terms.iter().map(|(n, v)| (n.clone(), t.value(*v)[0])).collect();
                return Ok(StepLog { updated: false, loss: loss_now, terms, grad_norm: 0.0, lr: 0.0 });
            }
        }
        t.backward(total)?;
        mark("backward", &|| { be.download(&be.alloc(1)); });
        self.step += 1;
        self.updates += 1;
        let (norm, factor) = apply_update(be, &self.st, &mut self.ema, &t, &self.cfg, self.step, &self.var.frozen);
        mark("optimizer", &|| { be.download(&be.alloc(1)); });
        let terms: Vec<(String, f32)> = terms.iter().map(|(n, v)| (n.clone(), t.value(*v)[0])).collect();
        Ok(StepLog { updated: true, loss: t.value(total)[0], terms, grad_norm: norm, lr: self.cfg.lr * factor })
    }

    /// Save model + EMA state as safetensors ("model.<name>", "ema.<name>").
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        save_stores(self.be, [("model", &self.st), ("ema", &self.ema.store)], path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_cosine_matches_deim() {
        // 932 iterations per epoch, 72 epochs, warm-up 1333, flat 40, no-aug 8, γ 0.5
        let s = FlatCosine::from_epochs(932, 72, 1333, 40, 8, 0.5);
        assert_eq!(s.factor(0), 0.0);
        assert!((s.factor(666) - (666.0f32 / 1333.0).powi(2)).abs() < 1e-6);
        assert_eq!(s.factor(1333), 1.0);
        assert_eq!(s.factor(932 * 40), 1.0);
        // halfway through the cosine: 0.5 + 0.5·0.5
        let mid = 932 * 40 + (932 * 72 - 932 * 40 - 932 * 8) / 2;
        assert!((s.factor(mid) - 0.75).abs() < 1e-4, "{}", s.factor(mid));
        assert_eq!(s.factor(932 * 64), 0.5);
        assert_eq!(s.factor(932 * 72 - 1), 0.5);
    }

    #[test]
    fn deim_parameter_groups() {
        let c = TrainCfg::deim_ojas_n32(932);
        assert_eq!(c.group("backbone.stages.0.blocks.0.layers.0.conv.weight"), (3e-4, 1e-4));
        assert_eq!(c.group("backbone.stem.stem1.lab.scale"), (3e-4, 1e-4));
        // backbone BN at the base lr, no decay
        assert_eq!(c.group("backbone.stem.stem1.bn.weight"), (6e-4, 0.0));
        assert_eq!(c.group("decoder.decoder.layers.0.norm1.weight"), (6e-4, 0.0));
        assert_eq!(c.group("encoder.fpn_blocks.0.cv1.norm.bias"), (6e-4, 0.0));
        assert_eq!(c.group("decoder.dec_bbox_head.0.layers.0.bias"), (6e-4, 1e-4));
    }
}
