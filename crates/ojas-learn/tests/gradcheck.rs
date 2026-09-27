//! Every tape op's backward against central finite differences (CPU reference).

use ojas_learn::check::gradcheck;
use ojas_learn::cpu::Cpu;
use ojas_learn::{Binary, Tape, Unary, Var};

/// Deterministic values in [-1, 1), kept away from 0 (ReLU / max kinks).
fn vals(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
            if v.abs() < 0.05 { v + 0.1 } else { v }
        })
        .collect()
}

fn inp(shape: &[usize], seed: u64) -> (Vec<f32>, Vec<usize>) {
    (vals(shape.iter().product(), seed), shape.to_vec())
}

/// A loss that weights every output element differently, so a wrong index in a
/// backward cannot cancel out.
fn weighted_sum(t: &mut Tape<Cpu>, y: Var) -> Var {
    let n: usize = t.shape(y).iter().product();
    let w: Vec<f32> = (0..n).map(|i| 0.3 + (i % 7) as f32 * 0.17).collect();
    let s = t.shape(y).to_vec();
    let wv = t.input(&w, &s);
    let p = t.mul(y, wv).unwrap();
    t.sum(p)
}

fn assert_ok(name: &str, m: ojas_learn::check::Mismatch, tol: f32) {
    assert!(m.err < tol, "{name}: input {} [{}] analytic {} numeric {} (err {})", m.input, m.index, m.analytic, m.numeric, m.err);
    eprintln!("{name}: worst err {:.2e}", m.err);
}

#[test]
fn unary_ops() {
    for op in [Unary::Relu, Unary::Sigmoid, Unary::Silu, Unary::Tanh, Unary::Gelu, Unary::Exp, Unary::Neg] {
        let m = gradcheck(&[inp(&[3, 5], 1)], 1e-2, |t, v| {
            let y = t.unary(op, v[0]);
            weighted_sum(t, y)
        });
        assert_ok(&format!("{op:?}"), m, 2e-3);
    }
    // log / sqrt need positive inputs
    for op in [Unary::Log, Unary::Sqrt] {
        let x: Vec<f32> = vals(15, 2).iter().map(|v| v.abs() + 0.5).collect();
        let m = gradcheck(&[(x, vec![3, 5])], 1e-2, |t, v| {
            let y = t.unary(op, v[0]);
            weighted_sum(t, y)
        });
        assert_ok(&format!("{op:?}"), m, 2e-3);
    }
}

#[test]
fn binary_ops_with_broadcasting() {
    let cases: &[(&[usize], &[usize])] = &[(&[2, 3, 4], &[2, 3, 4]), (&[2, 3, 4], &[4]), (&[2, 1, 4], &[3, 1]), (&[1], &[2, 3])];
    for &(sa, sb) in cases {
        for op in [Binary::Add, Binary::Sub, Binary::Mul, Binary::Div, Binary::Max, Binary::Min] {
            let (mut a, mut b) = (inp(sa, 3), inp(sb, 4));
            if op == Binary::Div {
                b.0.iter_mut().for_each(|v| *v = v.abs() + 0.5);
            }
            if matches!(op, Binary::Max | Binary::Min) {
                // keep every a/b pair ≥ 0.05 apart so the finite difference never straddles the kink
                a.0.iter_mut().for_each(|v| *v = (*v * 10.0).round() / 10.0 + 0.05);
                b.0.iter_mut().for_each(|v| *v = (*v * 10.0).round() / 10.0);
            }
            let m = gradcheck(&[a, b], 1e-2, |t, v| {
                let y = t.binary(op, v[0], v[1]).unwrap();
                weighted_sum(t, y)
            });
            assert_ok(&format!("{op:?} {sa:?}×{sb:?}"), m, 5e-3);
        }
    }
}

#[test]
fn matmul_batched_shared_and_transposed() {
    // batched × batched
    let m = gradcheck(&[inp(&[2, 3, 4], 5), inp(&[2, 4, 5], 6)], 1e-2, |t, v| {
        let y = t.matmul(v[0], v[1]).unwrap();
        weighted_sum(t, y)
    });
    assert_ok("matmul batched", m, 2e-3);
    // batched × shared
    let m = gradcheck(&[inp(&[2, 3, 4], 7), inp(&[4, 5], 8)], 1e-2, |t, v| {
        let y = t.matmul(v[0], v[1]).unwrap();
        weighted_sum(t, y)
    });
    assert_ok("matmul shared", m, 2e-3);
    // linear: x · wᵀ + b
    let m = gradcheck(&[inp(&[2, 3, 4], 9), inp(&[5, 4], 10), inp(&[5], 11)], 1e-2, |t, v| {
        let y = t.linear(v[0], v[1], Some(v[2])).unwrap();
        weighted_sum(t, y)
    });
    assert_ok("linear", m, 2e-3);
    // batched × batched transposed (attention q·kᵀ)
    let m = gradcheck(&[inp(&[2, 3, 4], 12), inp(&[2, 5, 4], 13)], 1e-2, |t, v| {
        let y = t.matmul_opts(v[0], v[1], true).unwrap();
        weighted_sum(t, y)
    });
    assert_ok("matmul batched trans_b", m, 2e-3);
}

#[test]
fn softmax_and_layernorm() {
    let m = gradcheck(&[inp(&[3, 7], 14)], 1e-2, |t, v| {
        let y = t.softmax(v[0]);
        weighted_sum(t, y)
    });
    assert_ok("softmax", m, 2e-3);
    let m = gradcheck(&[inp(&[4, 6], 15), inp(&[6], 16), inp(&[6], 17)], 1e-2, |t, v| {
        let y = t.layer_norm(v[0], v[1], v[2], 1e-5);
        weighted_sum(t, y)
    });
    assert_ok("layer_norm", m, 5e-3);
}

#[test]
fn data_movement() {
    let m = gradcheck(&[inp(&[2, 3, 4], 18)], 1e-2, |t, v| {
        let y = t.permute(v[0], &[2, 0, 1]).unwrap();
        weighted_sum(t, y)
    });
    assert_ok("permute", m, 1e-3);
    let m = gradcheck(&[inp(&[2, 5, 3], 19)], 1e-2, |t, v| {
        let y = t.slice(v[0], 1, 1, 4).unwrap();
        weighted_sum(t, y)
    });
    assert_ok("slice", m, 1e-3);
    let m = gradcheck(&[inp(&[2, 2, 3], 20), inp(&[2, 4, 3], 21)], 1e-2, |t, v| {
        let y = t.concat(&[v[0], v[1]], 1).unwrap();
        weighted_sum(t, y)
    });
    assert_ok("concat", m, 1e-3);
    let m = gradcheck(&[inp(&[2, 6], 22)], 1e-2, |t, v| {
        let y = t.reshape(v[0], &[3, 4]).unwrap();
        weighted_sum(t, y)
    });
    assert_ok("reshape", m, 1e-3);
}

#[test]
fn sum_axis_clamp_inverse_sigmoid() {
    for axis in 0..3 {
        let m = gradcheck(&[inp(&[2, 3, 4], 50)], 1e-2, |t, v| {
            let y = t.sum_axis(v[0], axis).unwrap();
            weighted_sum(t, y)
        });
        assert_ok(&format!("sum_axis {axis}"), m, 1e-3);
    }
    // inputs in (0.1, 0.9): inside the clip range, away from its kinks
    let x: Vec<f32> = vals(12, 51).iter().map(|v| 0.5 + 0.4 * v).collect();
    let m = gradcheck(&[(x, vec![3, 4])], 1e-3, |t, v| {
        let y = t.inverse_sigmoid(v[0], 1e-5).unwrap();
        weighted_sum(t, y)
    });
    assert_ok("inverse_sigmoid", m, 5e-3);
}

#[test]
fn a_tensor_read_twice_accumulates() {
    // y = x·x + permute(x)ᵀ-ish reuse: x feeds three ops
    let m = gradcheck(&[inp(&[3, 3], 23)], 1e-2, |t, v| {
        let a = t.mul(v[0], v[0]).unwrap();
        let b = t.permute(v[0], &[1, 0]).unwrap();
        let c = t.add(a, b).unwrap();
        let d = t.matmul(c, v[0]).unwrap();
        weighted_sum(t, d)
    });
    assert_ok("reuse", m, 3e-3);
}

#[test]
fn transformer_block() {
    // pre-norm attention + GELU MLP: the ops of a DETR decoder layer
    let (s, d) = (5, 8);
    let inputs = vec![inp(&[s, d], 30), inp(&[d, d], 31), inp(&[d, d], 32), inp(&[d, d], 33), inp(&[d], 34), inp(&[d], 35), inp(&[16, d], 36), inp(&[d, 16], 37)];
    let m = gradcheck(&inputs, 1e-2, |t, v| {
        let h = t.layer_norm(v[0], v[4], v[5], 1e-5);
        let q = t.linear(h, v[1], None).unwrap();
        let k = t.linear(h, v[2], None).unwrap();
        let vv = t.linear(h, v[3], None).unwrap();
        let sc = t.matmul_opts(q, k, true).unwrap();
        let sc = t.scale(sc, 1.0 / (d as f32).sqrt()).unwrap();
        let p = t.softmax(sc);
        let a = t.matmul(p, vv).unwrap();
        let x1 = t.add(v[0], a).unwrap();
        let f = t.linear(x1, v[6], None).unwrap();
        let f = t.gelu(f);
        let f = t.linear(f, v[7], None).unwrap();
        let y = t.add(x1, f).unwrap();
        weighted_sum(t, y)
    });
    assert_ok("transformer block", m, 5e-3);
}
