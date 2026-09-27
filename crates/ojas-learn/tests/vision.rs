//! Vision ops (conv, BatchNorm, pooling, upsample, GridSample, row gather):
//! gradients vs finite differences on the CPU reference, then CUDA vs CPU on
//! larger shapes — the same generic program run on both backends.

use ojas_learn::backend::Backend;
use ojas_learn::check::gradcheck;
use ojas_learn::cpu::Cpu;
use ojas_learn::{BnRunning, Tape, Var};

fn vals(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn inp(shape: &[usize], seed: u64) -> (Vec<f32>, Vec<usize>) {
    (vals(shape.iter().product(), seed), shape.to_vec())
}

fn weighted_sum<B: Backend>(t: &mut Tape<B>, y: Var) -> Var {
    let n: usize = t.shape(y).iter().product();
    let w: Vec<f32> = (0..n).map(|i| 0.3 + (i % 7) as f32 * 0.17).collect();
    let s = t.shape(y).to_vec();
    let wv = t.input(&w, &s);
    let p = t.mul(y, wv).unwrap();
    t.sum(p)
}

/// Grid points whose pixel coordinates keep ≥ 0.2 from integer boundaries (the
/// bilinear weights' kinks), with some outside the image (zeros padding).
fn grid(n: usize, ho: usize, wo: usize, h: usize, w: usize, seed: u64) -> (Vec<f32>, Vec<usize>) {
    let r = vals(n * ho * wo * 2, seed);
    let g: Vec<f32> = r
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let size = if i % 2 == 0 { w } else { h } as f32;
            // pixel coordinate in [-1.5, size + 0.5), fractional part in [0.3, 0.7]
            let p = ((v * 0.5 + 0.5) * (size + 2.0)).floor() - 1.5 + 0.3 + 0.4 * ((v * 97.0).rem_euclid(1.0));
            (2.0 * p + 1.0) / size - 1.0
        })
        .collect();
    (g, vec![n, ho, wo, 2])
}

// ------------------------------------------------------ the programs ---

fn conv_prog<B: Backend>(t: &mut Tape<B>, v: &[Var], stride: usize, pad: usize, groups: usize) -> Var {
    let y = t.conv2d(v[0], v[1], Some(v[2]), [stride, stride], [pad; 4], groups).unwrap();
    weighted_sum(t, y)
}

fn bn_prog<B: Backend>(t: &mut Tape<B>, v: &[Var], train: bool) -> Var {
    let c = t.shape(v[0])[1];
    let run = BnRunning::new(t.be, c);
    let y = t.batch_norm2d(v[0], v[1], v[2], &run, train).unwrap();
    // square so the train-mode loss is not flat along the normalised directions
    let y2 = t.mul(y, y).unwrap();
    weighted_sum(t, y2)
}

fn pool_prog<B: Backend>(t: &mut Tape<B>, v: &[Var]) -> Var {
    let a = t.max_pool2d(v[0], [3, 3], [2, 2], [1, 1, 1, 1], false).unwrap();
    let b = t.avg_pool2d(v[0], [2, 2], [2, 2], [0; 4], true, false).unwrap();
    let u = t.upsample_nearest(b, 2, 2).unwrap();
    let (sa, su) = (weighted_sum(t, a), weighted_sum(t, u));
    t.add(sa, su).unwrap()
}

fn grid_prog<B: Backend>(t: &mut Tape<B>, v: &[Var]) -> Var {
    let y = t.grid_sample(v[0], v[1]).unwrap();
    weighted_sum(t, y)
}

fn gather_prog<B: Backend>(t: &mut Tape<B>, v: &[Var]) -> Var {
    let k = 3;
    let n = t.shape(v[0])[1];
    let idx: Vec<usize> = (0..t.shape(v[0])[0] * k).map(|i| (i * 7 + 2) % n).collect();
    let y = t.gather_rows(v[0], &idx, k).unwrap();
    weighted_sum(t, y)
}

// ------------------------------------------------------ CPU gradcheck ---

fn check(name: &str, inputs: &[(Vec<f32>, Vec<usize>)], eps: f32, tol: f32, f: impl Fn(&mut Tape<Cpu>, &[Var]) -> Var) {
    let m = gradcheck(inputs, eps, f);
    assert!(m.err < tol, "{name}: input {} [{}] analytic {} numeric {} (err {})", m.input, m.index, m.analytic, m.numeric, m.err);
    eprintln!("{name}: worst err {:.2e}", m.err);
}

#[test]
fn conv_gradients() {
    check("conv 3x3 s1 p1", &[inp(&[2, 3, 5, 6], 1), inp(&[4, 3, 3, 3], 2), inp(&[4], 3)], 1e-2, 3e-3, |t, v| conv_prog(t, v, 1, 1, 1));
    check("conv 3x3 s2 p1", &[inp(&[1, 2, 7, 6], 4), inp(&[3, 2, 3, 3], 5), inp(&[3], 6)], 1e-2, 3e-3, |t, v| conv_prog(t, v, 2, 1, 1));
    check("conv 1x1", &[inp(&[2, 4, 3, 3], 7), inp(&[5, 4, 1, 1], 8), inp(&[5], 9)], 1e-2, 3e-3, |t, v| conv_prog(t, v, 1, 0, 1));
    check("conv grouped", &[inp(&[1, 4, 5, 5], 10), inp(&[6, 2, 3, 3], 11), inp(&[6], 12)], 1e-2, 3e-3, |t, v| conv_prog(t, v, 1, 1, 2));
    check("conv depthwise", &[inp(&[2, 3, 5, 5], 13), inp(&[3, 1, 3, 3], 14), inp(&[3], 15)], 1e-2, 3e-3, |t, v| conv_prog(t, v, 2, 1, 3));
}

#[test]
fn batchnorm_gradients() {
    let x = inp(&[3, 2, 3, 4], 20);
    check("bn train", &[x.clone(), inp(&[2], 21), inp(&[2], 22)], 1e-2, 5e-3, |t, v| bn_prog(t, v, true));
    check("bn eval", &[x, inp(&[2], 23), inp(&[2], 24)], 1e-2, 3e-3, |t, v| bn_prog(t, v, false));
}

#[test]
fn pool_upsample_gradients() {
    // well-separated values: no max-pool ties within a finite-difference step
    let n = 2 * 2 * 7 * 6;
    let x: Vec<f32> = (0..n).map(|i| ((i * 37 + 11) % n) as f32 * 0.05 - 1.0).collect();
    // the loss sums ~100 terms in f32: a 2e-2 step (below half the 0.05 value spacing) keeps
    // its rounding under the tolerance without changing any max-pool winner
    check("maxpool / avgpool (ceil) / upsample", &[(x, vec![2, 2, 7, 6])], 2e-2, 5e-3, pool_prog::<Cpu>);
}

#[test]
fn grid_sample_gradients() {
    check("grid_sample", &[inp(&[2, 3, 5, 6], 30), grid(2, 4, 3, 5, 6, 31)], 2e-3, 5e-3, grid_prog::<Cpu>);
}

#[test]
fn gather_rows_gradients() {
    check("gather_rows", &[inp(&[2, 5, 4], 40)], 1e-2, 1e-3, gather_prog::<Cpu>);
}

#[test]
fn detach_blocks_the_gradient() {
    let be = Cpu;
    let mut t = Tape::new(&be);
    let x = t.var(&[1.0, 2.0, 3.0], &[3]);
    let d = t.detach(x);
    let a = t.mul(x, d).unwrap(); // d(x·stop(x))/dx = stop(x), not 2x
    let l = t.sum(a);
    t.backward(l).unwrap();
    assert_eq!(t.grad_vec(x).unwrap(), vec![1.0, 2.0, 3.0]);
}

// ------------------------------------------------------ CUDA vs CPU ---

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use ojas_learn::cuda::Cuda;

    fn gpu() -> Option<Cuda> {
        match Cuda::exact(0) {
            Ok(c) => Some(c),
            Err(e) if format!("{e:#}").contains("CUDA unavailable") => {
                eprintln!("skipped: no CUDA device");
                None
            }
            Err(e) => panic!("CUDA backend failed to start: {e:#}"),
        }
    }

    fn run<B: Backend>(be: &B, inputs: &[(Vec<f32>, Vec<usize>)], prog: impl Fn(&mut Tape<B>, &[Var]) -> Var) -> (f32, Vec<Vec<f32>>) {
        let mut t = Tape::new(be);
        let vs: Vec<Var> = inputs.iter().map(|(d, s)| t.var(d, s)).collect();
        let l = prog(&mut t, &vs);
        t.backward(l).unwrap();
        (t.value(l)[0], vs.iter().map(|&v| t.grad_vec(v).unwrap()).collect())
    }

    fn close(name: &str, c: (f32, Vec<Vec<f32>>), g: (f32, Vec<Vec<f32>>), tol: f32) {
        let rel = |a: f32, b: f32| (a - b).abs() / a.abs().max(1.0);
        assert!(rel(c.0, g.0) < tol.max(1e-4), "{name}: loss cpu {} cuda {}", c.0, g.0);
        let mut worst = 0.0f32;
        for (k, (a, b)) in c.1.iter().zip(&g.1).enumerate() {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                let e = rel(*x, *y);
                assert!(e < tol, "{name}: grad {k}[{i}] cpu {x} cuda {y}");
                worst = worst.max(e);
            }
        }
        eprintln!("{name}: loss {} vs {}, worst grad err {worst:.2e}", c.0, g.0);
    }

    #[test]
    fn conv_matches() {
        let Some(g) = gpu() else { return };
        // a backbone-like layer: 3x3 s2 over 40x36, channels off the GEMM tile edges
        let ins = [inp(&[2, 24, 40, 36], 1), inp(&[40, 24, 3, 3], 2), inp(&[40], 3)];
        close("conv 3x3 s2", run(&Cpu, &ins, |t, v| conv_prog(t, v, 2, 1, 1)), run(&g, &ins, |t, v| conv_prog(t, v, 2, 1, 1)), 1e-3);
        let ins = [inp(&[2, 32, 20, 20], 4), inp(&[32, 1, 3, 3], 5), inp(&[32], 6)];
        close("conv depthwise", run(&Cpu, &ins, |t, v| conv_prog(t, v, 1, 1, 32)), run(&g, &ins, |t, v| conv_prog(t, v, 1, 1, 32)), 1e-3);
    }

    #[test]
    fn batchnorm_matches() {
        let Some(g) = gpu() else { return };
        let ins = [inp(&[4, 48, 17, 19], 7), inp(&[48], 8), inp(&[48], 9)];
        close("bn train", run(&Cpu, &ins, |t, v| bn_prog(t, v, true)), run(&g, &ins, |t, v| bn_prog(t, v, true)), 1e-3);
        close("bn eval", run(&Cpu, &ins, |t, v| bn_prog(t, v, false)), run(&g, &ins, |t, v| bn_prog(t, v, false)), 1e-4);
    }

    #[test]
    fn pool_upsample_match() {
        let Some(g) = gpu() else { return };
        let n = 3 * 16 * 33 * 31;
        let x: Vec<f32> = (0..n).map(|i| ((i * 7919 + 13) % n) as f32 / n as f32).collect();
        let ins = [(x, vec![3, 16, 33, 31])];
        close("pools / upsample", run(&Cpu, &ins, pool_prog::<Cpu>), run(&g, &ins, pool_prog::<Cuda>), 1e-5);
    }

    #[test]
    fn grid_sample_matches() {
        let Some(g) = gpu() else { return };
        // the deformable-attention shape: 8 heads × 32 channels sampled at 300×4 points
        let ins = [inp(&[8, 32, 40, 40], 10), grid(8, 300, 4, 40, 40, 11)];
        close("grid_sample", run(&Cpu, &ins, grid_prog::<Cpu>), run(&g, &ins, grid_prog::<Cuda>), 1e-3);
    }

    /// Tensor-core precisions against the f32 CPU reference, by norm, including
    /// split-K weight gradients (K = N·OH·OW) and the implicit im2col: TF32 within
    /// 0.3%, bf16 (the training default) within 1%.
    #[test]
    fn tensor_core_conv_close_to_f32() {
        let mut g = match Cuda::new(0) {
            Ok(c) => c,
            Err(e) if format!("{e:#}").contains("CUDA unavailable") => return,
            Err(e) => panic!("{e:#}"),
        };
        for (prec, tol) in [(ojas_learn::cuda::Prec::Tf32, 3e-3f32), (ojas_learn::cuda::Prec::Bf16, 1e-2)] {
        g.prec = prec;
        for (name, ins, s, p, gr) in [
            ("conv 3x3 s1 (split-K dW)", vec![inp(&[2, 64, 48, 48], 21), inp(&[64, 64, 3, 3], 22), inp(&[64], 23)], 1, 1, 1),
            ("conv 1x1 s2", vec![inp(&[2, 96, 30, 30], 24), inp(&[130, 96, 1, 1], 25), inp(&[130], 26)], 2, 0, 1),
        ] {
            let c = run(&Cpu, &ins, |t, v| conv_prog(t, v, s, p, gr));
            let t = run(&g, &ins, |t, v| conv_prog(t, v, s, p, gr));
            for (k, (a, b)) in c.1.iter().zip(&t.1).enumerate() {
                let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
                let den: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(num / den < tol, "{prec:?} {name}: grad {k} relative error {}", num / den);
                eprintln!("{prec:?} {name}: grad {k} relative error {:.2e}", num / den);
            }
        }
        }
    }

    #[test]
    fn gather_rows_matches() {
        let Some(g) = gpu() else { return };
        let ins = [inp(&[2, 1000, 64], 12)];
        close("gather_rows", run(&Cpu, &ins, gather_prog::<Cpu>), run(&g, &ins, gather_prog::<Cuda>), 1e-5);
    }
}
