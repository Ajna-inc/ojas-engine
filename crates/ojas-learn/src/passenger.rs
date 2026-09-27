//! Four-way crop classifier on the PResNet-18-vd backbone. RGB / 255, aspect-preserving black
//! letterbox; BatchNorm statistics frozen.
use anyhow::{ensure, Result};
use crate::{Backend, Param, Tape, Unary, Var};
use crate::models::{detr_loss::Rng, rtdetr::{Config, RtDetr, Store}};

pub const NAMES: [&str; 4] = ["Hatchback", "Sedan", "SUV", "MUV"];

/// Temporary SAM ascent, restoring the weights exactly even on error. Optimizer moments are never
/// perturbed. The caller supplies the global norm of the un-clipped gradients over the same active
/// parameter set.
pub struct SamPerturbation<'a, B: Backend> {
    be: &'a B,
    saved: Vec<(&'a Param<B>, B::Buf)>,
    restored: bool,
}

impl<'a, B: Backend> SamPerturbation<'a, B> {
    pub fn new(be: &'a B, grads: &[(&'a Param<B>, &B::Buf)], rho: f32, norm: f32) -> Result<Self> {
        ensure!(rho.is_finite() && rho > 0. && norm.is_finite() && norm >= 0. && !grads.is_empty(), "invalid SAM perturbation");
        let mut guard = Self { be, saved: vec![], restored: false };
        for &(p, g) in grads {
            let backup = be.alloc(be.len(&p.val));
            be.axpby(&p.val, &backup, 1., 0.);
            guard.saved.push((p, backup));
            be.axpby(g, &p.val, rho / (norm + 1e-12), 1.);
        }
        Ok(guard)
    }

    pub fn delta_norm(&self) -> f32 {
        let norm = self.be.alloc(1);
        for (p, backup) in &self.saved {
            let delta = self.be.alloc(self.be.len(backup));
            self.be.axpby(&p.val, &delta, 1., 0.);
            self.be.axpby(backup, &delta, -1., 1.);
            self.be.sumsq(&delta, &norm, true);
        }
        self.be.download(&norm)[0].sqrt()
    }

    pub fn restore(&mut self) {
        if !self.restored {
            for (p, backup) in &self.saved { self.be.axpby(backup, &p.val, 1., 0.); }
            self.restored = true;
        }
    }

    pub fn restoration_is_exact(&self) -> bool {
        self.restored && self.saved.iter().all(|(p, backup)| self.be.download(&p.val) == self.be.download(backup))
    }
}

impl<B: Backend> Drop for SamPerturbation<'_, B> {
    fn drop(&mut self) { self.restore(); }
}

pub fn init<B: Backend>(be: &B, path: &str, seed: u64) -> Result<Store<B>> {
    init_stage(be, path, seed, 2)
}

pub fn init_stage<B: Backend>(be: &B, path: &str, seed: u64, stage: usize) -> Result<Store<B>> {
    ensure!((1..=3).contains(&stage), "invalid classifier backbone stage");
    let tensors = ojas_formats::pth::load(&std::fs::read(path)?)?;
    let tensors: Vec<_> = tensors.into_iter().filter(|(n, _)| n.starts_with("ema.module.backbone.")).collect();
    ensure!(!tensors.is_empty(), "no EMA backbone in checkpoint");
    let mut st = Store::from_tensors(be, &tensors, "ema.module.");
    for omitted in stage + 1..=3 {
        let prefix = format!("backbone.res_layers.{omitted}.");
        st.params.retain(|name, _| !name.starts_with(&prefix));
        st.bns.retain(|name, _| !name.starts_with(&prefix));
    }
    let mut rng = Rng::new(seed);
    let channels = 64usize << stage;
    let weights: Vec<_> = (0..4 * channels).map(|_| (rng.uniform() * 2. - 1.) / (channels as f32).sqrt()).collect();
    st.params.insert("classifier.weight".into(), Param::new(be, "classifier.weight", &[4, channels], &weights));
    st.params.insert("classifier.bias".into(), Param::new(be, "classifier.bias", &[4], &[0.; 4]));
    Ok(st)
}

pub fn forward<B: Backend>(t: &mut Tape<B>, st: &Store<B>, x: Var) -> Result<Var> {
    let model = RtDetr { cfg: Config::r18vd(15), st, train: false, var: Default::default() };
    let channels = st.params["classifier.weight"].shape[1];
    let stage = match channels { 128 => 1, 256 => 2, 512 => 3, _ => anyhow::bail!("unsupported classifier width") };
    let features = model.backbone_through(t, x, stage)?;
    let h = *features.last().unwrap();
    let shape = t.shape(h).to_vec();
    let pooled = t.avg_pool2d(h, [shape[2], shape[3]], [1, 1], [0; 4], false, true)?;
    let pooled = t.reshape(pooled, &[shape[0], channels])?;
    let w = t.param(&st.params["classifier.weight"]);
    let b = t.param(&st.params["classifier.bias"]);
    t.linear(pooled, w, Some(b))
}

/// Stable log-softmax cross-entropy, including gradients for very wrong, confident logits.
pub fn cross_entropy<B: Backend>(t: &mut Tape<B>, logits: Var, labels: &[usize]) -> Result<Var> {
    ensure!(t.shape(logits) == [labels.len(), 4] && !labels.is_empty(), "CE expects nonempty [B,4]");
    ensure!(labels.iter().all(|&c| c < 4), "label outside four-way taxonomy");
    let maxima: Vec<_> = t.value(logits).chunks(4).map(|r| r.iter().copied().fold(f32::NEG_INFINITY, f32::max)).collect();
    let shift = t.input(&maxima, &[labels.len(), 1]);
    let shifted = t.sub(logits, shift)?;
    let exp = t.unary(Unary::Exp, shifted);
    let sum = t.sum_axis(exp, 1)?;
    let logsum = t.unary(Unary::Log, sum);
    let logsum = t.reshape(logsum, &[labels.len(), 1])?;
    let logprob = t.sub(shifted, logsum)?;
    let mut onehot = vec![0.; labels.len() * 4];
    for (i, &c) in labels.iter().enumerate() { onehot[i * 4 + c] = 1.; }
    let target = t.input(&onehot, &[labels.len(), 4]);
    let selected = t.mul(logprob, target)?;
    Ok(t.sum_scaled(selected, -1. / labels.len() as f32))
}

pub fn preprocess(img: &image::RgbImage, size: usize, flip: bool, brightness: f32) -> Vec<f32> {
    let scale = size as f64 / img.width().max(img.height()) as f64;
    let w = ((img.width() as f64 * scale).round() as u32).clamp(1, size as u32);
    let h = ((img.height() as f64 * scale).round() as u32).clamp(1, size as u32);
    let small = image::imageops::resize(img, w, h, image::imageops::FilterType::Triangle);
    let mut out = vec![0.; 3 * size * size];
    let (ox, oy) = ((size - w as usize) / 2, (size - h as usize) / 2);
    for y in 0..h as usize { for x in 0..w as usize {
        let pixel = small.get_pixel(if flip { w - 1 - x as u32 } else { x as u32 }, y as u32);
        for c in 0..3 { out[c * size * size + (y + oy) * size + x + ox] = (pixel[c] as f32 / 255. * brightness).clamp(0., 1.); }
    }}
    out
}

pub fn probabilities(logits: &[f32]) -> Vec<[f32; 4]> {
    logits.chunks_exact(4).map(|r| {
        let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut p = [0.; 4];
        for i in 0..4 { p[i] = (r[i] - m).exp(); }
        let sum: f32 = p.iter().sum();
        for x in &mut p { *x /= sum; }
        p
    }).collect()
}

pub fn metrics(predictions: &[[f32; 4]], labels: &[usize]) -> serde_json::Value {
    let mut confusion = [[0usize; 4]; 4];
    let mut nll = 0.;
    for (p, &label) in predictions.iter().zip(labels) {
        let best = (0..4).max_by(|&i, &j| p[i].total_cmp(&p[j])).unwrap();
        confusion[label][best] += 1;
        nll -= p[label].max(1e-30).ln() as f64;
    }
    let total = labels.len();
    let correct: usize = (0..4).map(|i| confusion[i][i]).sum();
    let mut f1 = 0.;
    let mut recall = [0.; 4];
    for c in 0..4 {
        let gt: usize = confusion[c].iter().sum();
        let predicted: usize = confusion.iter().map(|r| r[c]).sum();
        recall[c] = confusion[c][c] as f64 / gt.max(1) as f64;
        f1 += 2. * confusion[c][c] as f64 / (gt + predicted).max(1) as f64;
    }
    serde_json::json!({"n":total,"accuracy":correct as f64 / total.max(1) as f64,"macro_f1":f1/4.,"nll":nll/total.max(1) as f64,"per_class_recall":recall,"confusion":confusion})
}

pub fn save<B: Backend>(be: &B, st: &Store<B>, path: &std::path::Path) -> Result<()> {
    let tensors: Vec<_> = st.named_state().into_iter().map(|(name, buf, shape)| {
        let data = be.download(buf);
        let shape = if shape.is_empty() { vec![data.len()] } else { shape };
        (format!("model.{name}"), shape, data)
    }).collect();
    ojas_formats::safetensors::write_f32(path, &tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Cpu;
    #[test]
    fn sam_uses_global_norm_and_restores_on_error_without_touching_moments() {
        let be = Cpu;
        let p = Param::new(&be, "p", &[2], &[1., -2.]);
        let q = Param::new(&be, "q", &[1], &[0.5]);
        let g = be.upload(&[3., 4.]);
        let h = be.upload(&[12.]);
        let result: Result<()> = (|| {
            let guard = SamPerturbation::new(&be, &[(&p, &g), (&q, &h)], 0.13, 13.)?;
            assert!((guard.delta_norm() - 0.13).abs() < 1e-6);
            assert!((be.download(&p.val)[0] - 1.03).abs() < 1e-6);
            assert!((be.download(&q.val)[0] - 0.62).abs() < 1e-6);
            anyhow::bail!("simulated second-pass failure")
        })();
        assert!(result.is_err());
        assert_eq!(be.download(&p.val), [1., -2.]);
        assert_eq!(be.download(&q.val), [0.5]);
        assert_eq!(be.download(&p.m), [0., 0.]);
        assert_eq!(be.download(&p.v), [0., 0.]);
    }

    #[test]
    fn sam_second_gradient_is_evaluated_at_perturbed_weights() {
        let be = Cpu;
        let p = Param::new(&be, "p", &[2], &[1., 2.]);
        let mut first = Tape::new(&be);
        let x = first.param(&p);
        let square = first.mul(x, x).unwrap();
        let loss = first.sum_scaled(square, 0.5);
        first.backward(loss).unwrap();
        let mut guard = SamPerturbation::new(&be, &[(&p, first.grad(x).unwrap())], 0.1, 5f32.sqrt()).unwrap();
        let expected = be.download(&p.val);
        drop(first);
        let mut second = Tape::new(&be);
        let x = second.param(&p);
        let square = second.mul(x, x).unwrap();
        let loss = second.sum_scaled(square, 0.5);
        second.backward(loss).unwrap();
        guard.restore();
        assert!(guard.restoration_is_exact());
        assert_eq!(be.download(&p.val), [1., 2.]);
        let gradient = second.grad_vec(x).unwrap();
        for (g, e) in gradient.iter().zip(expected) { assert!((g - e).abs() < 1e-6); }
        assert!(gradient[0] > 1. && gradient[1] > 2.);
    }
    #[test]
    fn ce_gradient_matches_softmax_and_stays_finite_at_extreme_logits() {
        let be = Cpu;
        let values = [1000., -1000., 0., 1., 0.2, -0.3, 0.7, 1.];
        let labels = [1, 2];
        let mut t = Tape::new(&be);
        let x = t.var(&values, &[2, 4]);
        let loss = cross_entropy(&mut t, x, &labels).unwrap();
        assert!(t.value(loss)[0].is_finite() && t.value(loss)[0] > 1000.);
        t.backward(loss).unwrap();
        let grad = t.grad_vec(x).unwrap();
        for (i, p) in probabilities(&values).iter().enumerate() { for c in 0..4 {
            let expected = (p[c] - if c == labels[i] { 1. } else { 0. }) / 2.;
            assert!((grad[i * 4 + c] - expected).abs() < 1e-6);
        }}
    }
    #[test]
    fn letterbox_preserves_aspect_and_rgb_order() {
        let img = image::RgbImage::from_pixel(4, 2, image::Rgb([255, 128, 0]));
        let out = preprocess(&img, 4, false, 1.);
        assert_eq!(out[0], 0.);
        assert_eq!(out[4], 1.);
        assert!((out[16 + 4] - 128./255.).abs() < 1e-6);
        assert_eq!(out[32 + 4], 0.);
    }
}
