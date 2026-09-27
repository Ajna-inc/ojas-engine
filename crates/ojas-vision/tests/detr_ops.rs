//! The CUDA kernels for GridSample, TopK, TopKGather and Reduce against the CPU
//! reference (`detr_ops`), on the cases that break them: sample points outside the
//! image and on its edges, a score row full of exact ties, 8400 candidates → 300.
//! Values are multiples of 1/64 so f16 storage holds them exactly, making a tie a tie
//! on both backends so that only the outputs' own f16 rounding differs.
#![cfg(feature = "cuda")]

use ojas_vision::exec_cpu::CpuExecutor;
use ojas_vision::exec_gpu::CudaExecutor;
use ojas_vision::ir::{Graph, Node, Op, ReduceOp, TensorKind};

fn exact(i: usize, modulus: usize) -> f32 {
    ((i * 37 + 11) % modulus) as f32 / 64.0 - 1.0
}

fn run_both(g: &Graph, input: &[f32]) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let cpu = CpuExecutor::new(g, 4).run(g, &[input]).unwrap();
    let mut gpu = match CudaExecutor::new(g, 0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e:#})");
            return None;
        }
    };
    Some((cpu, gpu.run(input).unwrap()))
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + x.abs()), "{what}[{i}]: cpu {x} cuda {y}");
    }
}

fn node(op: Op, inputs: Vec<usize>, outputs: Vec<usize>) -> Node {
    Node { name: format!("{op:?}"), op, inputs, outputs }
}

#[test]
fn grid_sample_matches_the_cpu_inside_on_the_edges_and_outside() {
    let mut g = Graph::default();
    let x = g.add_tensor("x", vec![1, 4, 6, 8], TensorKind::Input);
    // (x, y) points over [-1.3, 1.3]: outside, exactly on ±1, and between pixels
    let pts: Vec<f32> = (0..5 * 7 * 2).map(|i| -1.3 + 2.6 * ((i * 29) % 70) as f32 / 69.0).collect();
    let mut grid = pts;
    grid[0..4].copy_from_slice(&[-1.0, -1.0, 1.0, 1.0]);
    let gw = g.add_weight("grid", vec![1, 5, 7, 2], grid);
    let y = g.add_tensor("y", vec![1, 4, 5, 7], TensorKind::Value);
    g.nodes.push(node(Op::GridSample, vec![x, gw], vec![y]));
    g.inputs.push(x);
    g.outputs.push(y);
    let input: Vec<f32> = (0..4 * 6 * 8).map(|i| exact(i, 128)).collect();
    let Some((cpu, gpu)) = run_both(&g, &input) else { return };
    close(&cpu[0], &gpu[0], 2e-3, "grid_sample");
    assert!(cpu[0].iter().any(|&v| v == 0.0) && cpu[0].iter().any(|&v| v != 0.0), "some points outside, some inside");
}

#[test]
fn topk_and_the_fused_gather_pick_the_same_rows_through_ties() {
    // scores: 8400 = 1 x 84 x 100 image, as the DETR encoder's candidates; only 97 distinct values
    let mut g = Graph::default();
    let x = g.add_tensor("x", vec![1, 1, 84, 100], TensorKind::Input);
    let s = g.add_tensor("scores", vec![1, 8400], TensorKind::Value);
    g.tensors[s].alias_of = Some(x);
    g.nodes.push(node(Op::View, vec![x], vec![s]));
    // each row names itself exactly in f16: (row / 64, row % 64, 0.5, -1)
    let data: Vec<f32> = (0..8400).flat_map(|r| [(r / 64) as f32, (r % 64) as f32, 0.5, -1.0]).collect();
    let d = g.add_weight("data", vec![1, 8400, 4], data);
    let top = g.add_tensor("top", vec![1, 300], TensorKind::Value);
    let rows = g.add_tensor("rows", vec![1, 300, 4], TensorKind::Value);
    g.nodes.push(node(Op::TopK { k: 300 }, vec![s], vec![top]));
    g.nodes.push(node(Op::TopKGather { k: 300 }, vec![s, d], vec![rows]));
    g.inputs.push(x);
    g.outputs.extend([top, rows]);
    let input: Vec<f32> = (0..8400).map(|i| exact(i, 97)).collect();
    let Some((cpu, gpu)) = run_both(&g, &input) else { return };
    assert_eq!(cpu[0], gpu[0], "topk values, largest first");
    assert!(cpu[0].windows(2).all(|w| w[0] >= w[1]));
    assert_eq!(cpu[1], gpu[1], "the same rows in the same order (ties to the lower index)");
    // ties really happened, and the lower index came first
    let firsts: Vec<f32> = cpu[1].chunks(4).map(|r| r[0] * 64.0 + r[1]).collect();
    let tied = cpu[0].windows(2).zip(firsts.windows(2)).filter(|(v, _)| v[0] == v[1]).count();
    assert!(tied > 100, "{tied} tied neighbours");
    assert!(cpu[0].windows(2).zip(firsts.windows(2)).all(|(v, r)| v[0] != v[1] || r[0] < r[1]));
}

#[test]
fn reductions_sum_max_mean_over_trailing_axes() {
    for (kind, axes, out) in [(ReduceOp::Sum, vec![3], vec![1, 4, 6]), (ReduceOp::Max, vec![3], vec![1, 4, 6]), (ReduceOp::Mean, vec![2, 3], vec![1, 4])] {
        let mut g = Graph::default();
        let x = g.add_tensor("x", vec![1, 4, 6, 8], TensorKind::Input);
        let y = g.add_tensor("y", out, TensorKind::Value);
        g.nodes.push(node(Op::Reduce { kind, axes, keepdims: false }, vec![x], vec![y]));
        g.inputs.push(x);
        g.outputs.push(y);
        let input: Vec<f32> = (0..4 * 6 * 8).map(|i| exact(i, 128)).collect();
        let Some((cpu, gpu)) = run_both(&g, &input) else { return };
        close(&cpu[0], &gpu[0], 2e-3, &format!("{kind:?}"));
    }
}
