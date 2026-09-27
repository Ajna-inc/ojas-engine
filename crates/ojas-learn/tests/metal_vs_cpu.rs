//! The Metal backend against the CPU reference: the same tape program on both, forward values and
//! every gradient compared. Mirrors `cuda_vs_cpu.rs` -- same cases, same tolerances, same `TapeOps`
//! shim -- swapped onto `metal::Metal`.
#![cfg(feature = "metal")]

use ojas_learn::backend::Backend;
use ojas_learn::cpu::Cpu;
use ojas_learn::metal::Metal;
use ojas_learn::{AdamW, BnRunning, Binary, Param, Tape, Unary, Var};

fn vals(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn metal() -> Option<Metal> {
    match Metal::new() {
        Ok(m) => Some(m),
        Err(e) if format!("{e:#}").contains("no Metal device") => {
            eprintln!("skipped: no Metal device ({e:#})");
            None
        }
        Err(e) => panic!("Metal backend failed to start: {e:#}"),
    }
}

/// Run `f` on both backends with the same inputs (all differentiable); return
/// (cpu, metal) x (loss, grads).
fn both<F>(inputs: &[(Vec<f32>, Vec<usize>)], f: F) -> Option<((f32, Vec<Vec<f32>>), (f32, Vec<Vec<f32>>))>
where
    F: Fn(&mut dyn TapeOps, &[Var]) -> Var,
{
    let gpu = metal()?;
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
    assert!(rel(cpu.0, gpu.0) < tol.max(1e-4), "{name}: loss cpu {} metal {}", cpu.0, gpu.0);
    let mut worst = 0.0f32;
    for (k, (a, b)) in cpu.1.iter().zip(&gpu.1).enumerate() {
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let e = rel(*x, *y);
            assert!(e < tol, "{name}: grad {k}[{i}] cpu {x} metal {y}");
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
    // batched, shared, trans_b, linear -- M, N, K all off the 64 tile edge
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

fn weighted_sum<B: Backend>(t: &mut Tape<'_, B>, y: Var) -> Var {
    let n: usize = t.shape(y).iter().product();
    let w: Vec<f32> = (0..n).map(|i| 0.3 + (i % 7) as f32 * 0.17).collect();
    let s = t.shape(y).to_vec();
    let wv = t.input(&w, &s);
    let p = Tape::mul(t, y, wv).unwrap();
    t.sum(p)
}

/// One tape program run to (loss, every input's gradient) on a given backend. Each vision-op test
/// below builds a generic function rather than a closure, since a closure cannot implement both
/// `Fn(&mut Tape<Cpu>,..)` and `Fn(&mut Tape<Metal>,..)`, and calls it once per backend.
fn run<B: Backend>(be: &B, inputs: &[(Vec<f32>, Vec<usize>)], f: impl Fn(&mut Tape<'_, B>, &[Var]) -> Var) -> (f32, Vec<Vec<f32>>) {
    let mut t = Tape::new(be);
    let vs: Vec<Var> = inputs.iter().map(|(d, s)| t.var(d, s)).collect();
    let l = f(&mut t, &vs);
    t.backward(l).unwrap();
    (t.value(l)[0], vs.iter().map(|&v| t.grad_vec(v).unwrap()).collect())
}

#[test]
fn conv_pool_gridsample() {
    let Some(gpu) = metal() else { return };

    // NCHW 2x3x17x19, a 4x3x3x3 kernel: shapes off every tile/window edge.
    fn conv<B: Backend>(be: &B) -> (f32, Vec<Vec<f32>>) {
        run(be, &[(vals(2 * 3 * 17 * 19, 30), vec![2, 3, 17, 19]), (vals(4 * 3 * 3 * 3, 31), vec![4, 3, 3, 3]), (vals(4, 32), vec![4])], |t, v| {
            let y = t.conv2d(v[0], v[1], Some(v[2]), [2, 2], [1, 1, 1, 1], 1).unwrap();
            weighted_sum(t, y)
        })
    }
    close("conv2d", &conv(&Cpu), &conv(&gpu), 1e-3);

    fn maxp<B: Backend>(be: &B) -> (f32, Vec<Vec<f32>>) {
        run(be, &[(vals(2 * 3 * 17 * 19, 33), vec![2, 3, 17, 19])], |t, v| {
            let y = t.max_pool2d(v[0], [3, 3], [2, 2], [1, 1, 1, 1], false).unwrap();
            weighted_sum(t, y)
        })
    }
    close("max_pool2d", &maxp(&Cpu), &maxp(&gpu), 1e-4);

    fn avgp<B: Backend>(be: &B) -> (f32, Vec<Vec<f32>>) {
        run(be, &[(vals(2 * 3 * 17 * 19, 34), vec![2, 3, 17, 19])], |t, v| {
            let y = t.avg_pool2d(v[0], [3, 3], [2, 2], [1, 1, 1, 1], false, false).unwrap();
            weighted_sum(t, y)
        })
    }
    close("avg_pool2d", &avgp(&Cpu), &avgp(&gpu), 1e-4);

    // BatchNorm2d, train mode (bn_stats + bn_apply + bn_bwd + bn_wgrad): the
    // sequential double-reduce shape.
    fn bn<B: Backend>(be: &B) -> (f32, Vec<Vec<f32>>) {
        run(be, &[(vals(2 * 5 * 17 * 19, 35), vec![2, 5, 17, 19]), (vals(5, 36), vec![5]), (vals(5, 37), vec![5])], |t, v| {
            let bnr = BnRunning::new(t.be, 5);
            let y = t.batch_norm2d(v[0], v[1], v[2], &bnr, true).unwrap();
            weighted_sum(t, y)
        })
    }
    close("batch_norm2d", &bn(&Cpu), &bn(&gpu), 1e-3);

    // grid values near [-1, 1]: half in-bounds, half sampling past the edge, so
    // GridSample's zero-padding branch runs on both backends.
    fn gs<B: Backend>(be: &B) -> (f32, Vec<Vec<f32>>) {
        let grid: Vec<f32> = vals(2 * 5 * 6 * 2, 38).iter().map(|v| v * 1.3).collect();
        run(be, &[(vals(2 * 3 * 17 * 19, 39), vec![2, 3, 17, 19]), (grid, vec![2, 5, 6, 2])], |t, v| {
            let y = t.grid_sample(v[0], v[1]).unwrap();
            weighted_sum(t, y)
        })
    }
    close("grid_sample", &gs(&Cpu), &gs(&gpu), 1e-3);
}

#[test]
fn gather_rows_matches() {
    let Some(gpu) = metal() else { return };
    fn gr<B: Backend>(be: &B) -> (f32, Vec<Vec<f32>>) {
        run(be, &[(vals(2 * 9 * 4, 40), vec![2, 9, 4])], |t, v| {
            let idx = [0usize, 3, 5, 8, 1, 2, 0, 4, 7, 6];
            let y = t.gather_rows(v[0], &idx, 5).unwrap();
            weighted_sum(t, y)
        })
    }
    close("gather_rows", &gr(&Cpu), &gr(&gpu), 1e-5);
}

#[test]
fn adamw_steps_match() {
    let Some(gpu) = metal() else { return };
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
    assert!(worst < 1e-5, "AdamW after 5 steps: worst |delta| {worst}");
    eprintln!("adamw 5 steps: worst |delta| {worst:.2e}");
}
