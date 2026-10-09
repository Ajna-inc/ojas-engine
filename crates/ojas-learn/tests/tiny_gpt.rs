//! TinyGpt on the tape: the whole model against finite differences, loss going down on a
//! repetitive corpus, and (feature `metal`) the Metal backend against the CPU reference.

use ojas_learn::check::gradcheck;
use ojas_learn::cpu::Cpu;
use ojas_learn::models::tiny_gpt::{self, forward_loss, TinyGpt};
use ojas_learn::Var;
use ojas_swarm_proto::TinyGptConfig;

fn toy(vocab: u32) -> TinyGptConfig {
    TinyGptConfig { vocab, ctx: 6, d_model: 8, n_layers: 2, n_heads: 2, d_ff: 12, init_seed: 7 }
}

/// Larger-than-init weights so attention is not uniform and every path carries gradient.
fn spread(theta: &[f32]) -> Vec<f32> {
    theta.iter().enumerate().map(|(i, v)| v * 4.0 + ((i as f32) * 0.618).sin() * 0.02).collect()
}

#[test]
fn layout_is_canonical_and_identity_tracks_content() {
    let c = TinyGptConfig::small(256);
    let specs = tiny_gpt::param_specs(&c);
    assert!(specs.windows(2).all(|w| w[0].0 < w[1].0), "sorted by name");
    let theta = tiny_gpt::init(&c);
    assert_eq!(theta.len(), tiny_gpt::n_params(&c));
    assert_eq!(theta, tiny_gpt::init(&c), "init is deterministic");
    let a = tiny_gpt::identity(&c, &theta).unwrap();
    let mut t2 = theta.clone();
    t2[123] += 1e-6;
    let b = tiny_gpt::identity(&c, &t2).unwrap();
    assert_eq!(a.arch, b.arch);
    assert_ne!(a.model, b.model);
    let other = tiny_gpt::identity(&TinyGptConfig { init_seed: 2, ..c.clone() }, &tiny_gpt::init(&TinyGptConfig { init_seed: 2, ..c.clone() })).unwrap();
    assert_eq!(a.arch, other.arch, "the seed is not layout");
    assert_ne!(a.model, other.model);
    eprintln!("TinyGptConfig::small(256): {} parameters", theta.len());
}

#[test]
fn whole_model_gradcheck() {
    let c = toy(11);
    let specs = tiny_gpt::param_specs(&c);
    let theta = spread(&tiny_gpt::init(&c));
    let mut inputs = Vec::new();
    let mut off = 0;
    for (_, s) in &specs {
        let n: usize = s.iter().product();
        inputs.push((theta[off..off + n].to_vec(), s.clone()));
        off += n;
    }
    // two rows: one full window (ctx + 1) each, repeated tokens included
    let tokens: Vec<u32> = vec![1, 4, 4, 9, 0, 3, 10, 2, 7, 7, 5, 1, 6, 8];
    // eps 2e-3: LayerNorm over 8 small-variance features is curved enough that 1e-2
    // measures the curvature, not the gradient; below ~1e-3 f32 loss rounding dominates
    let m = gradcheck(&inputs, 2e-3, |t, v: &[Var]| forward_loss(t, &c, v, &tokens, 2).unwrap());
    eprintln!("tiny_gpt gradcheck: {m:?} ({} params)", theta.len());
    assert!(m.err < 1e-2, "{m:?} in {}", specs[m.input].0);
}

#[test]
fn shorter_windows_than_ctx_work() {
    let c = toy(11);
    let m = TinyGpt::new(&Cpu, &c, None).unwrap();
    let l = m.loss(&Cpu, &[1, 2, 3, 4, 5, 6, 7, 8], 2).unwrap();
    assert!((l - (11f32).ln()).abs() < 0.1, "init loss {l} should be near ln V");
    assert!(m.loss(&Cpu, &[1, 2, 3, 11], 1).is_err(), "token outside vocab");
}

#[test]
fn loss_falls_on_a_repetitive_corpus() {
    let c = TinyGptConfig { vocab: 16, ctx: 8, d_model: 16, n_layers: 1, n_heads: 2, d_ff: 32, init_seed: 3 };
    let text: Vec<u32> = (0..400).map(|i| [3, 1, 4, 1, 5, 9, 2, 6, 5, 3][i % 10]).collect();
    let mut m = TinyGpt::new(&Cpu, &c, None).unwrap();
    let batch = 4;
    let mut first = 0.0;
    let mut last = 0.0;
    for step in 0..200 {
        let mut b = Vec::new();
        for j in 0..batch {
            let s = ((step * batch + j) * 7) % (text.len() - 9);
            b.extend_from_slice(&text[s..s + 9]);
        }
        let o = m.step(&Cpu, &b, batch, 1e-2, 0.0).unwrap();
        if step == 0 {
            first = o.loss;
        }
        last = o.loss;
    }
    eprintln!("repetitive corpus: loss {first:.3} -> {last:.3}");
    assert!(last < 0.5 * first && last < 0.6, "loss {first} -> {last}");
}

#[test]
fn a_poisoned_step_is_refused_before_touching_weights() {
    let c = toy(11);
    let mut m = TinyGpt::new(&Cpu, &c, None).unwrap();
    let before = m.flatten(&Cpu);
    let mut bad = before.clone();
    bad[0] = f32::NAN;
    assert!(m.unflatten(&Cpu, &bad).is_err(), "non-finite θ refused");
    // a huge embedding overflows the forward to Inf/NaN
    let huge: Vec<f32> = before.iter().map(|v| v * 1e30).collect();
    m.unflatten(&Cpu, &huge).unwrap();
    let r = m.step(&Cpu, &[1, 2, 3, 4, 5, 6, 7], 1, 1e-3, 0.0);
    assert!(r.is_err(), "{r:?}");
    assert_eq!(m.flatten(&Cpu), huge, "weights untouched by the refused step");
    assert_eq!(m.opt.step, 0);
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;
    use ojas_learn::metal::Metal;

    #[test]
    fn metal_matches_cpu_forward_and_grads() {
        let gpu = match Metal::new() {
            Ok(m) => m,
            Err(e) if format!("{e:#}").contains("no Metal device") => return eprintln!("skipped: {e:#}"),
            Err(e) => panic!("{e:#}"),
        };
        let c = TinyGptConfig { vocab: 256, ctx: 16, d_model: 32, n_layers: 2, n_heads: 4, d_ff: 64, init_seed: 5 };
        let theta = spread(&tiny_gpt::init(&c));
        let tokens: Vec<u32> = (0..3 * 17).map(|i| ((i * 37 + 11) % 256) as u32).collect();
        let (lc, gc) = TinyGpt::new(&Cpu, &c, Some(&theta)).unwrap().loss_and_grads(&Cpu, &tokens, 3).unwrap();
        let (lm, gm) = TinyGpt::new(&gpu, &c, Some(&theta)).unwrap().loss_and_grads(&gpu, &tokens, 3).unwrap();
        let scale = gc.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let worst = gc.iter().zip(&gm).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("tiny_gpt metal vs cpu: loss {lc} vs {lm}, worst grad |diff| {worst:.2e} (max |grad| {scale:.2e})");
        assert!((lc - lm).abs() < 1e-4 * lc.abs().max(1.0), "loss {lc} vs {lm}");
        assert!(worst < 1e-4 * scale.max(1.0), "grads differ by {worst}");
    }
}
