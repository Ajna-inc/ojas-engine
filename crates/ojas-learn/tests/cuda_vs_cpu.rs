//! The CUDA backend against the CPU reference: the same tape program on both,
//! forward values and every gradient compared. Sizes cross the GEMM tile edges
//! (64) and exceed one block (256) per softmax / LayerNorm row.
#![cfg(feature = "cuda")]

use ojas_learn::backend::Backend;
use ojas_learn::cpu::Cpu;
use ojas_learn::cuda::Cuda;
use ojas_learn::{AdamW, Binary, Param, Tape, Unary, Var};

fn vals(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn cuda() -> Option<Cuda> {
    match Cuda::exact(0) {
        Ok(c) => Some(c),
        // only a missing driver / device skips; a kernel that fails to compile is a failure
        Err(e) if format!("{e:#}").contains("CUDA unavailable") => {
            eprintln!("skipped: no CUDA device ({e:#})");
            None
        }
        Err(e) => panic!("CUDA backend failed to start: {e:#}"),
    }
}

/// Run `f` on both backends with the same inputs (all differentiable); return
/// (cpu, cuda) × (loss, grads).
fn both<F>(inputs: &[(Vec<f32>, Vec<usize>)], f: F) -> Option<((f32, Vec<Vec<f32>>), (f32, Vec<Vec<f32>>))>
where
    F: Fn(&mut dyn TapeOps, &[Var]) -> Var,
{
    let gpu = cuda()?;
    fn run<B: Backend>(be: &B, inputs: &[(Vec<f32>, Vec<usize>)], f: &dyn Fn(&mut dyn TapeOps, &[Var]) -> Var) -> (f32, Vec<Vec<f32>>)
    where
        for<'a> Tape<'a, B>: TapeOps,
    {
        let mut t = Tape::new(be);
        let vs: Vec<Var> = inputs.iter().map(|(d, s)| t.var(d, s)).collect();
        let l = f(&mut t, &vs);
        t.backward(l).unwrap();
        (t.value(l)[0], vs.iter().map(|&v| t.grad_vec(v).unwrap()).collect())
    }
    Some((run(&Cpu, inputs, &f), run(&gpu, inputs, &f)))
}

/// Object-safe view of the tape ops the tests use, so one closure drives both backends.
trait TapeOps {
    fn unary(&mut self, op: Unary, x: Var) -> Var;
    fn binary(&mut self, op: Binary, a: Var, b: Var) -> Var;
    fn matmul_opts(&mut self, a: Var, b: Var, tb: bool) -> Var;
    fn linear(&mut self, x: Var, w: Var, b: Option<Var>) -> Var;
    fn softmax(&mut self, x: Var) -> Var;
    fn layer_norm(&mut self, x: Var, g: Var, b: Var) -> Var;
    fn permute(&mut self, x: Var, p: &[usize]) -> Var;
    fn slice(&mut self, x: Var, axis: usize, s: usize, e: usize) -> Var;
    fn concat(&mut self, xs: &[Var], axis: usize) -> Var;
    fn weighted_sum(&mut self, y: Var) -> Var;
}

impl<B: Backend> TapeOps for Tape<'_, B> {
    fn unary(&mut self, op: Unary, x: Var) -> Var {
        Tape::unary(self, op, x)
    }
    fn binary(&mut self, op: Binary, a: Var, b: Var) -> Var {
        Tape::binary(self, op, a, b).unwrap()
    }
    fn matmul_opts(&mut self, a: Var, b: Var, tb: bool) -> Var {
        Tape::matmul_opts(self, a, b, tb).unwrap()
    }
    fn linear(&mut self, x: Var, w: Var, b: Option<Var>) -> Var {
        Tape::linear(self, x, w, b).unwrap()
    }
    fn softmax(&mut self, x: Var) -> Var {
        Tape::softmax(self, x)
    }
    fn layer_norm(&mut self, x: Var, g: Var, b: Var) -> Var {
        Tape::layer_norm(self, x, g, b, 1e-5)
    }
    fn permute(&mut self, x: Var, p: &[usize]) -> Var {
        Tape::permute(self, x, p).unwrap()
    }
    fn slice(&mut self, x: Var, axis: usize, s: usize, e: usize) -> Var {
        Tape::slice(self, x, axis, s, e).unwrap()
    }
    fn concat(&mut self, xs: &[Var], axis: usize) -> Var {
        Tape::concat(self, xs, axis).unwrap()
    }
    fn weighted_sum(&mut self, y: Var) -> Var {
        let n: usize = self.shape(y).iter().product();
        let w: Vec<f32> = (0..n).map(|i| 0.3 + (i % 7) as f32 * 0.17).collect();
        let s = self.shape(y).to_vec();
        let wv = self.input(&w, &s);
        let p = Tape::mul(self, y, wv).unwrap();
        self.sum(p)
    }
}

fn close(name: &str, cpu: &(f32, Vec<Vec<f32>>), gpu: &(f32, Vec<Vec<f32>>), tol: f32) {
    let rel = |a: f32, b: f32| (a - b).abs() / a.abs().max(1.0);
    // the loss is a sum of thousands of terms: CPU adds in order, CUDA as a tree
    assert!(rel(cpu.0, gpu.0) < tol.max(1e-4), "{name}: loss cpu {} cuda {}", cpu.0, gpu.0);
    let mut worst = 0.0f32;
    for (k, (a, b)) in cpu.1.iter().zip(&gpu.1).enumerate() {
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let e = rel(*x, *y);
            assert!(e < tol, "{name}: grad {k}[{i}] cpu {x} cuda {y}");
            worst = worst.max(e);
        }
    }
    eprintln!("{name}: loss {} vs {}, worst grad err {worst:.2e}", cpu.0, gpu.0);
}

#[test]
fn elementwise_and_broadcast() {
    for op in [Unary::Relu, Unary::Sigmoid, Unary::Silu, Unary::Tanh, Unary::Gelu, Unary::Exp, Unary::Neg] {
        let Some((c, g)) = both(&[(vals(3000, 1), vec![3, 1000])], |t, v| {
            let y = t.unary(op, v[0]);
            t.weighted_sum(y)
        }) else { return };
        close(&format!("{op:?}"), &c, &g, 1e-4);
    }
    for op in [Binary::Add, Binary::Sub, Binary::Mul, Binary::Max, Binary::Min] {
        let Some((c, g)) = both(&[(vals(2 * 70 * 33, 2), vec![2, 70, 33]), (vals(33, 3), vec![33])], |t, v| {
            let y = t.binary(op, v[0], v[1]);
            t.weighted_sum(y)
        }) else { return };
        close(&format!("{op:?}"), &c, &g, 1e-4);
    }
}

#[test]
fn gemm_every_layout() {
    // batched, shared, trans_b, linear — M, N, K all off the 64 / 16 tile edges
    let Some((c, g)) = both(&[(vals(3 * 130 * 70, 4), vec![3, 130, 70]), (vals(3 * 70 * 90, 5), vec![3, 70, 90])], |t, v| {
        let y = t.matmul_opts(v[0], v[1], false);
        t.weighted_sum(y)
    }) else { return };
    close("matmul batched", &c, &g, 1e-3);
    let Some((c, g)) = both(&[(vals(3 * 130 * 70, 6), vec![3, 130, 70]), (vals(90 * 70, 7), vec![90, 70]), (vals(90, 8), vec![90])], |t, v| {
        let y = t.linear(v[0], v[1], Some(v[2]));
        t.weighted_sum(y)
    }) else { return };
    close("linear", &c, &g, 1e-3);
    let Some((c, g)) = both(&[(vals(2 * 77 * 40, 9), vec![2, 77, 40]), (vals(2 * 65 * 40, 10), vec![2, 65, 40])], |t, v| {
        let y = t.matmul_opts(v[0], v[1], true);
        t.weighted_sum(y)
    }) else { return };
    close("matmul trans_b", &c, &g, 1e-3);
}

#[test]
fn softmax_layernorm_wide_rows() {
    let Some((c, g)) = both(&[(vals(5 * 700, 11), vec![5, 700])], |t, v| {
        let y = t.softmax(v[0]);
        t.weighted_sum(y)
    }) else { return };
    close("softmax", &c, &g, 1e-4);
    let Some((c, g)) = both(&[(vals(300 * 520, 12), vec![300, 520]), (vals(520, 13), vec![520]), (vals(520, 14), vec![520])], |t, v| {
        let y = t.layer_norm(v[0], v[1], v[2]);
        t.weighted_sum(y)
    }) else { return };
    close("layer_norm", &c, &g, 1e-3);
}

#[test]
fn data_movement() {
    let Some((c, g)) = both(&[(vals(2 * 3 * 40 * 50, 15), vec![2, 3, 40, 50]), (vals(2 * 5 * 40 * 50, 16), vec![2, 5, 40, 50])], |t, v| {
        let a = t.permute(v[0], &[0, 2, 3, 1]);
        let b = t.slice(v[1], 1, 1, 4);
        let b = t.permute(b, &[0, 2, 3, 1]);
        let y = t.concat(&[a, b], 3);
        t.weighted_sum(y)
    }) else { return };
    close("permute/slice/concat", &c, &g, 1e-5);
}

#[test]
fn adamw_steps_match() {
    let Some(gpu) = cuda() else { return };
    fn steps<B: Backend>(be: &B) -> Vec<f32> {
        let p = Param::new(be, "w", &[64, 32], &vals(64 * 32, 20));
        let x = vals(16 * 32, 21);
        let mut opt = AdamW { lr: 1e-2, wd: 0.05, ..Default::default() };
        for _ in 0..5 {
            let mut t = Tape::new(be);
            let xv = t.input(&x, &[16, 32]);
            let w = t.param(&p);
            let y = t.linear(xv, w, None).unwrap();
            let y = t.unary(Unary::Tanh, y);
            let l = t.mean(y);
            t.backward(l).unwrap();
            opt.begin();
            opt.update(be, &p, t.grad(w).unwrap(), opt.wd);
        }
        be.download(&p.val)
    }
    let (c, g) = (steps(&Cpu), steps(&gpu));
    let worst = c.iter().zip(&g).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "AdamW after 5 steps: worst |Δ| {worst}");
    eprintln!("adamw 5 steps: worst |Δ| {worst:.2e}");
}
