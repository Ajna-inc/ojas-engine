//! Training-step speed of RT-DETRv2-S's backbone (PResNet-18-vd, BN in train
//! mode) at 640×640: forward, backward and AdamW, per batch.
//! `backbone_bench [batch] [iters]` (CUDA).

use std::time::Instant;

use ojas_learn::backend::Backend;
use ojas_learn::cuda::Cuda;
use ojas_learn::{AdamW, BnRunning, Param, Tape, Var};

struct ConvBn {
    w: Param<Cuda>,
    g: Param<Cuda>,
    b: Param<Cuda>,
    run: BnRunning<Cuda>,
    stride: usize,
    k: usize,
}

fn init(n: usize, fan_in: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let scale = (2.0 / fan_in as f32).sqrt();
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

impl ConvBn {
    fn new(be: &Cuda, cin: usize, cout: usize, k: usize, stride: usize, seed: u64) -> Self {
        ConvBn {
            w: Param::new(be, "w", &[cout, cin, k, k], &init(cout * cin * k * k, cin * k * k, seed)),
            g: Param::new(be, "g", &[cout], &vec![1.0; cout]),
            b: Param::new(be, "b", &[cout], &vec![0.0; cout]),
            run: BnRunning::new(be, cout),
            stride,
            k,
        }
    }

    fn fwd(&self, t: &mut Tape<Cuda>, x: Var, relu: bool, vars: &mut Vec<(Var, *const Param<Cuda>)>) -> Var {
        let (w, g, b) = (t.param(&self.w), t.param(&self.g), t.param(&self.b));
        vars.extend([(w, &self.w as *const _), (g, &self.g as *const _), (b, &self.b as *const _)]);
        let p = self.k / 2;
        let h = t.conv2d(x, w, None, [self.stride; 2], [p; 4], 1).unwrap();
        let h = t.batch_norm2d(h, g, b, &self.run, true).unwrap();
        if relu { t.relu(h) } else { h }
    }
}

struct Block {
    a: ConvBn,
    b: ConvBn,
    short: Option<ConvBn>,
    pool: bool,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let (batch, iters) = (args.first().copied().unwrap_or(4), args.get(1).copied().unwrap_or(3));
    let be = Cuda::new(0)?;
    let mut seed = 0;
    let mut cb = |cin, cout, k, s| {
        seed += 1;
        ConvBn::new(&be, cin, cout, k, s, seed)
    };
    let stem = [cb(3, 32, 3, 2), cb(32, 32, 3, 1), cb(32, 64, 3, 1)];
    let mut blocks = vec![];
    let mut cin = 64;
    for (i, &c) in [64usize, 128, 256, 512].iter().enumerate() {
        for j in 0..2 {
            let s = if i > 0 && j == 0 { 2 } else { 1 };
            let short = (cin != c || s != 1).then(|| cb(cin, c, 1, 1));
            blocks.push(Block { a: cb(cin, c, 3, s), b: cb(c, c, 3, 1), short, pool: s == 2 });
            cin = c;
        }
    }
    let x: Vec<f32> = init(batch * 3 * 640 * 640, 1, 99);
    let mut opt = AdamW { lr: 1e-4, ..Default::default() };
    for it in 0..iters + 1 {
        let t0 = Instant::now();
        let mut t = Tape::new(&be);
        let mut vars = vec![];
        let xv = t.input(&x, &[batch, 3, 640, 640]);
        let mut h = xv;
        for s in &stem {
            h = s.fwd(&mut t, h, true, &mut vars);
        }
        h = t.max_pool2d(h, [3, 3], [2, 2], [1; 4], false)?;
        let mut outs = vec![];
        for (bi, b) in blocks.iter().enumerate() {
            let a = b.a.fwd(&mut t, h, true, &mut vars);
            let a = b.b.fwd(&mut t, a, false, &mut vars);
            let sc = match &b.short {
                Some(s) => {
                    let p = if b.pool { t.avg_pool2d(h, [2, 2], [2, 2], [0; 4], true, false)? } else { h };
                    s.fwd(&mut t, p, false, &mut vars)
                }
                None => h,
            };
            let y = t.add(a, sc)?;
            h = t.relu(y);
            if bi % 2 == 1 && bi >= 3 {
                outs.push(h);
            }
        }
        let mut loss = t.mean(outs[0]);
        for &o in &outs[1..] {
            let m = t.mean(o);
            loss = t.add(loss, m)?;
        }
        let _ = t.value(loss); // sync
        let t1 = Instant::now();
        t.backward(loss)?;
        let _ = t.grad_vec(xv); // no grad on the input — sync on something small instead
        let _ = be.download(t.grad(vars[0].0).unwrap());
        let t2 = Instant::now();
        opt.begin();
        for &(v, p) in &vars {
            let p = unsafe { &*p };
            opt.update(&be, p, t.grad(v).unwrap(), 1e-4);
        }
        let _ = be.download(&vars.last().map(|&(_, p)| unsafe { &*p }).unwrap().val);
        let t3 = Instant::now();
        if it == 0 {
            be.take_profile();
        }
        if it > 0 {
            let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
            println!(
                "batch {batch}: forward {:.0} ms, backward {:.0} ms, adamw {:.1} ms → {:.1} img/s ({} params tensors)",
                ms(t0, t1),
                ms(t1, t2),
                ms(t2, t3),
                batch as f64 / (t3 - t0).as_secs_f64(),
                vars.len()
            );
        }
    }
    let prof = be.take_profile();
    if !prof.is_empty() {
        println!("per kernel over {iters} steps (synced):");
        for (k, ms, n) in prof.iter().take(12) {
            println!("  {k:24} {:8.1} ms/step  x{}", ms / iters as f64, n / iters);
        }
    }
    Ok(())
}
