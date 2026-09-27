//! End to end: a small CNN (conv → BN → ReLU → pool → conv → global pool →
//! linear) learns a synthetic task with AdamW — on the CPU reference, and on
//! CUDA where it must track the CPU run step for step.

use ojas_learn::backend::Backend;
use ojas_learn::cpu::Cpu;
use ojas_learn::{AdamW, BnRunning, Param, Tape};

fn vals(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Images 8×8×1: class 1 = a bright 3×3 blob somewhere, class 0 = noise only.
fn batch(step: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x = vals(n * 64, 1000 + step as u64, 0.3);
    let mut y = vec![0.0; n];
    for i in 0..n {
        if (i + step) % 2 == 0 {
            y[i] = 1.0;
            let (cy, cx) = (1 + (i * 3 + step) % 5, 1 + (i * 5 + step * 3) % 5);
            for dy in 0..3 {
                for dx in 0..3 {
                    x[i * 64 + (cy + dy - 1) * 8 + cx + dx - 1] += 1.5;
                }
            }
        }
    }
    (x, y)
}

struct Net<B: Backend> {
    c1: Param<B>,
    g1: Param<B>,
    b1: Param<B>,
    run1: BnRunning<B>,
    c2: Param<B>,
    fc: Param<B>,
    fb: Param<B>,
}

impl<B: Backend> Net<B> {
    fn new(be: &B) -> Self {
        Net {
            c1: Param::new(be, "c1", &[8, 1, 3, 3], &vals(72, 1, 0.4)),
            g1: Param::new(be, "g1", &[8], &[1.0; 8]),
            b1: Param::new(be, "b1", &[8], &[0.0; 8]),
            run1: BnRunning::new(be, 8),
            c2: Param::new(be, "c2", &[8, 8, 3, 3], &vals(576, 2, 0.15)),
            fc: Param::new(be, "fc", &[1, 8], &vals(8, 3, 0.3)),
            fb: Param::new(be, "fb", &[1], &[0.0]),
        }
    }

    /// One AdamW step; returns the loss (squared error on sigmoid(logit)).
    fn step(&self, be: &B, opt: &mut AdamW, x: &[f32], y: &[f32]) -> f32 {
        let n = y.len();
        let mut t = Tape::new(be);
        let xv = t.input(x, &[n, 1, 8, 8]);
        let yv = t.input(y, &[n, 1]);
        let (c1, g1, b1, c2, fc, fb) = (t.param(&self.c1), t.param(&self.g1), t.param(&self.b1), t.param(&self.c2), t.param(&self.fc), t.param(&self.fb));
        let h = t.conv2d(xv, c1, None, [1, 1], [1; 4], 1).unwrap();
        let h = t.batch_norm2d(h, g1, b1, &self.run1, true).unwrap();
        let h = t.relu(h);
        let h = t.max_pool2d(h, [2, 2], [2, 2], [0; 4], false).unwrap();
        let h = t.conv2d(h, c2, None, [1, 1], [1; 4], 1).unwrap();
        let h = t.relu(h);
        let h = t.avg_pool2d(h, [4, 4], [4, 4], [0; 4], false, false).unwrap();
        let h = t.reshape(h, &[n, 8]).unwrap();
        let logit = t.linear(h, fc, Some(fb)).unwrap();
        let p = t.sigmoid(logit);
        let d = t.sub(p, yv).unwrap();
        let d2 = t.mul(d, d).unwrap();
        let loss = t.mean(d2);
        t.backward(loss).unwrap();
        opt.begin();
        for (p, v, wd) in [(&self.c1, c1, 1e-4), (&self.g1, g1, 0.0), (&self.b1, b1, 0.0), (&self.c2, c2, 1e-4), (&self.fc, fc, 1e-4), (&self.fb, fb, 0.0)] {
            opt.update(be, p, t.grad(v).unwrap(), wd);
        }
        t.value(loss)[0]
    }
}

fn train<B: Backend>(be: &B, steps: usize) -> Vec<f32> {
    let net = Net::new(be);
    let mut opt = AdamW { lr: 1e-2, ..Default::default() };
    (0..steps)
        .map(|s| {
            let (x, y) = batch(s, 16);
            net.step(be, &mut opt, &x, &y)
        })
        .collect()
}

#[test]
fn small_cnn_learns_on_cpu() {
    let l = train(&Cpu, 60);
    let (first, last) = (l[..5].iter().sum::<f32>() / 5.0, l[l.len() - 5..].iter().sum::<f32>() / 5.0);
    eprintln!("cpu loss {first:.4} -> {last:.4}");
    assert!(last < first * 0.2, "loss {first} -> {last}");
}

#[cfg(feature = "cuda")]
#[test]
fn small_cnn_learns_on_cuda_like_the_cpu() {
    let gpu = match ojas_learn::cuda::Cuda::exact(0) {
        Ok(g) => g,
        Err(e) if format!("{e:#}").contains("CUDA unavailable") => return,
        Err(e) => panic!("{e:#}"),
    };
    let (c, g) = (train(&Cpu, 60), train(&gpu, 60));
    for (i, (a, b)) in c.iter().zip(&g).enumerate().take(20) {
        assert!((a - b).abs() < 2e-3 * a.abs().max(0.05), "step {i}: cpu {a} cuda {b}");
    }
    let last = g[g.len() - 5..].iter().sum::<f32>() / 5.0;
    eprintln!("cuda loss {:.4} -> {last:.4} (cpu {:.4})", g[0], c[c.len() - 1]);
    assert!(last < g[..5].iter().sum::<f32>() / 5.0 * 0.2);
}
