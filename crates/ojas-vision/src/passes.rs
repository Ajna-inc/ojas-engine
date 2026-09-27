//! IR rewrites, run to fixpoint after import:
//!
//! 1. Conv ← Mul/Add per-channel folding (BatchNorm arrives decomposed).
//! 2. `x · σ(x)` → SiLU.
//! 3. Conv/Gemm + unary activation fusion into the epilogue.
//! 4. Dead-code elimination.
//!
//! All rewrites keep the node list topologically ordered: a rewrite only ever
//! removes nodes or moves an output binding onto an earlier node.

use std::collections::HashMap;

use crate::ir::{BinaryOp, Graph, Op, TensorId, TensorKind, UnaryOp};

/// Run every pass to fixpoint, then DCE. Returns a per-pass hit count for
/// logging (`vision:load`).
pub fn optimize(g: &mut Graph) -> PassStats {
    let mut stats = PassStats::default();
    loop {
        let mut changed = 0;
        changed += fuse_silu(g);
        changed += fuse_gelu_tanh(g);
        changed += fuse_layer_norm(g);
        changed += fuse_linear(g);
        stats.silu += changed;
        let bn = fold_channel_affine_into_conv(g);
        stats.bn_folds += bn;
        changed += bn;
        let act = fuse_producer_act(g);
        stats.act_fusions += act;
        changed += act;
        let ss = fuse_scale_shift(g);
        stats.scale_shifts += ss;
        changed += ss;
        if changed == 0 {
            break;
        }
    }
    stats.dce = dce(g);
    stats
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PassStats {
    pub silu: usize,
    pub bn_folds: usize,
    pub act_fusions: usize,
    pub scale_shifts: usize,
    pub dce: usize,
}

/// tensor id → number of uses (node inputs + graph outputs).
fn use_counts(g: &Graph) -> Vec<usize> {
    let mut c = vec![0usize; g.tensors.len()];
    for n in &g.nodes {
        for &i in &n.inputs {
            c[i] += 1;
        }
    }
    for &o in &g.outputs {
        c[o] += 1;
    }
    c
}

/// tensor id → producing node index.
fn producers(g: &Graph) -> HashMap<TensorId, usize> {
    let mut m = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for &o in &n.outputs {
            m.insert(o, i);
        }
    }
    m
}

fn remove_nodes(g: &mut Graph, dead: &[bool]) {
    let mut i = 0;
    g.nodes.retain(|_| {
        let keep = !dead[i];
        i += 1;
        keep
    });
}

/// Mul(x, Sigmoid(x)) → Silu(x).
fn fuse_silu(g: &mut Graph) -> usize {
    let uses = use_counts(g);
    let prod = producers(g);
    let mut dead = vec![false; g.nodes.len()];
    let mut hits = 0;
    for mi in 0..g.nodes.len() {
        if dead[mi] || !matches!(g.nodes[mi].op, Op::Binary(BinaryOp::Mul)) {
            continue;
        }
        let (a, b) = (g.nodes[mi].inputs[0], g.nodes[mi].inputs[1]);
        // find which side is Sigmoid(other side)
        let matched = [(a, b), (b, a)].into_iter().find_map(|(x, s)| {
            let si = *prod.get(&s)?;
            if dead[si] {
                return None;
            }
            match g.nodes[si].op {
                Op::Unary(UnaryOp::Sigmoid) if g.nodes[si].inputs[0] == x && uses[s] == 1 => Some((x, si)),
                _ => None,
            }
        });
        if let Some((x, si)) = matched {
            let out = g.nodes[mi].outputs[0];
            g.nodes[mi] = crate::ir::Node {
                name: g.nodes[mi].name.clone(),
                op: Op::Unary(UnaryOp::Silu),
                inputs: vec![x],
                outputs: vec![out],
            };
            dead[si] = true;
            hits += 1;
        }
    }
    remove_nodes(g, &dead);
    hits
}

/// The tanh GELU as exporters spell it — ½·x·(1+tanh(√(2/π)·(x+0.044715·x³))), with the
/// constants already folded into ScaleShift nodes — back into one `GeluTanh`. In f16 the x³
/// overflows long before the GELU does, so the fused kernel (f32 inside) is the only correct way
/// to run it on the GPU.
fn fuse_gelu_tanh(g: &mut Graph) -> usize {
    let uses = use_counts(g);
    let prod = producers(g);
    let mut consumers: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for &t in &n.inputs {
            consumers.entry(t).or_default().push(i);
        }
    }
    let single = |t: TensorId| -> Option<usize> {
        match consumers.get(&t) {
            Some(v) if v.len() == 1 && uses[t] == 1 => Some(v[0]),
            _ => None,
        }
    };
    let near = |a: f32, b: f32| (a - b).abs() <= 1e-3 * b.abs();
    fn ss(g: &Graph, i: usize) -> Option<(f32, f32)> {
        match g.nodes[i].op {
            Op::ScaleShift { scale, shift } => Some((scale, shift)),
            _ => None,
        }
    }
    let mut dead = vec![false; g.nodes.len()];
    let mut hits = 0;
    'tanh: for ti in 0..g.nodes.len() {
        if dead[ti] || !matches!(g.nodes[ti].op, Op::Unary(UnaryOp::Tanh)) {
            continue;
        }
        // backwards: Tanh(ScaleShift_0.7979(Add(x, ScaleShift_0.044715(x³))))
        let Some(&s1) = prod.get(&g.nodes[ti].inputs[0]) else { continue };
        let Some((sc, sh)) = ss(g, s1) else { continue };
        if !near(sc, 0.797_884_56) || sh != 0.0 {
            continue;
        }
        let Some(&ai) = prod.get(&g.nodes[s1].inputs[0]) else { continue };
        if !matches!(g.nodes[ai].op, Op::Binary(BinaryOp::Add)) {
            continue;
        }
        let (p, q) = (g.nodes[ai].inputs[0], g.nodes[ai].inputs[1]);
        let mut found: Option<(TensorId, Vec<usize>)> = None;
        for (x, c) in [(p, q), (q, p)] {
            let Some(&s2) = prod.get(&c) else { continue };
            let Some((sc, sh)) = ss(g, s2) else { continue };
            if !near(sc, 0.044715) || sh != 0.0 {
                continue;
            }
            let Some(&ci) = prod.get(&g.nodes[s2].inputs[0]) else { continue };
            let mut chain = vec![s2, ci];
            let cube = match &g.nodes[ci].op {
                Op::Binary(BinaryOp::Pow) => {
                    g.nodes[ci].inputs[0] == x && g.weight(g.nodes[ci].inputs[1]).is_some_and(|w| w.len() == 1 && w[0] == 3.0)
                }
                Op::Binary(BinaryOp::Mul) => {
                    // x²·x in either order, x² = Mul(x, x)
                    let (m0, m1) = (g.nodes[ci].inputs[0], g.nodes[ci].inputs[1]);
                    let mut ok = false;
                    for (sq, xx) in [(m0, m1), (m1, m0)] {
                        if xx == x {
                            if let Some(&si) = prod.get(&sq) {
                                if matches!(g.nodes[si].op, Op::Binary(BinaryOp::Mul)) && g.nodes[si].inputs == [x, x] {
                                    chain.push(si);
                                    ok = true;
                                    break;
                                }
                            }
                        }
                    }
                    ok
                }
                _ => false,
            };
            if cube {
                found = Some((x, chain));
                break;
            }
        }
        let Some((x, mut chain)) = found else { continue };
        chain.extend([s1, ai, ti]);
        // forwards: (1 + tanh) · x · ½, the ½ before or after the product with x
        let Some(o1) = single(g.nodes[ti].outputs[0]) else { continue };
        if ss(g, o1) != Some((1.0, 1.0)) {
            continue;
        }
        let Some(mi) = single(g.nodes[o1].outputs[0]) else { continue };
        if !matches!(g.nodes[mi].op, Op::Binary(BinaryOp::Mul)) {
            continue;
        }
        let t1 = g.nodes[o1].outputs[0];
        let other = if g.nodes[mi].inputs[0] == t1 { g.nodes[mi].inputs[1] } else { g.nodes[mi].inputs[0] };
        chain.extend([o1, mi]);
        let last = if other == x {
            let Some(hi) = single(g.nodes[mi].outputs[0]) else { continue };
            if ss(g, hi) != Some((0.5, 0.0)) {
                continue;
            }
            chain.push(hi);
            hi
        } else {
            let Some(&hi) = prod.get(&other) else { continue };
            if ss(g, hi) != Some((0.5, 0.0)) || g.nodes[hi].inputs[0] != x {
                continue;
            }
            chain.push(hi);
            mi
        };
        // every intermediate feeds only the chain
        for &i in &chain {
            if dead[i] || (i != last && g.nodes[i].outputs.iter().any(|&o| uses[o] != 1)) {
                continue 'tanh;
            }
        }
        let out = g.nodes[last].outputs[0];
        g.nodes[last] = crate::ir::Node { name: g.nodes[last].name.clone(), op: Op::Unary(UnaryOp::GeluTanh), inputs: vec![x], outputs: vec![out] };
        for &i in &chain {
            if i != last {
                dead[i] = true;
            }
        }
        hits += 1;
    }
    remove_nodes(g, &dead);
    hits
}

/// A last-axis LayerNorm spelled out by the exporter — mean, centre, square, mean, +eps, sqrt,
/// divide, ·γ, +β — back into one `LayerNorm`: the variance of f16 activations overflows in the
/// decomposition, the fused kernel accumulates in f32.
fn fuse_layer_norm(g: &mut Graph) -> usize {
    let uses = use_counts(g);
    let mut consumers: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for &t in &n.inputs {
            consumers.entry(t).or_default().push(i);
        }
    }
    let single = |t: TensorId| -> Option<usize> {
        match consumers.get(&t) {
            Some(v) if v.len() == 1 && uses[t] == 1 => Some(v[0]),
            _ => None,
        }
    };
    fn last_axis_mean(g: &Graph, i: usize) -> bool {
        let rank = g.shape(g.nodes[i].inputs[0]).len();
        matches!(&g.nodes[i].op, Op::ReduceMean { axes, keepdims: true } if axes.as_slice() == [rank - 1])
    }
    let mut dead = vec![false; g.nodes.len()];
    let mut hits = 0;
    'mean: for mi in 0..g.nodes.len() {
        if dead[mi] || !last_axis_mean(g, mi) {
            continue;
        }
        let x = g.nodes[mi].inputs[0];
        let m = g.nodes[mi].outputs[0];
        // d = x - mean
        let Some(di) = single(m) else { continue };
        if !matches!(g.nodes[di].op, Op::Binary(BinaryOp::Sub)) || g.nodes[di].inputs != [x, m] {
            continue;
        }
        let d = g.nodes[di].outputs[0];
        // d is used twice: squared, and divided by the std
        let Some(dc) = consumers.get(&d) else { continue };
        if dc.len() != 2 || uses[d] != 2 {
            continue;
        }
        let sq_i = dc.iter().copied().find(|&i| match &g.nodes[i].op {
            Op::Binary(BinaryOp::Pow) => g.nodes[i].inputs[0] == d && g.weight(g.nodes[i].inputs[1]).is_some_and(|w| w.len() == 1 && w[0] == 2.0),
            Op::Binary(BinaryOp::Mul) => g.nodes[i].inputs == [d, d],
            _ => false,
        });
        let Some(sq_i) = sq_i else { continue };
        let Some(vi) = single(g.nodes[sq_i].outputs[0]) else { continue };
        if !last_axis_mean(g, vi) {
            continue;
        }
        let Some(ei) = single(g.nodes[vi].outputs[0]) else { continue };
        let eps = match g.nodes[ei].op {
            Op::ScaleShift { scale, shift } if scale == 1.0 && shift > 0.0 => shift,
            _ => continue,
        };
        let Some(si) = single(g.nodes[ei].outputs[0]) else { continue };
        if !matches!(g.nodes[si].op, Op::Unary(UnaryOp::Sqrt)) {
            continue;
        }
        let sd = g.nodes[si].outputs[0];
        let Some(qi) = single(sd) else { continue };
        if !matches!(g.nodes[qi].op, Op::Binary(BinaryOp::Div)) || g.nodes[qi].inputs != [d, sd] {
            continue;
        }
        // ·γ then +β (both weights)
        let Some(gi) = single(g.nodes[qi].outputs[0]) else { continue };
        if !matches!(g.nodes[gi].op, Op::Binary(BinaryOp::Mul)) {
            continue;
        }
        let nrm = g.nodes[qi].outputs[0];
        let gamma = if g.nodes[gi].inputs[0] == nrm { g.nodes[gi].inputs[1] } else { g.nodes[gi].inputs[0] };
        let c = *g.shape(x).last().unwrap();
        if !g.is_weight(gamma) || g.tensors[gamma].numel() != c {
            continue;
        }
        let Some(bi) = single(g.nodes[gi].outputs[0]) else { continue };
        if !matches!(g.nodes[bi].op, Op::Binary(BinaryOp::Add)) {
            continue;
        }
        let y1 = g.nodes[gi].outputs[0];
        let beta = if g.nodes[bi].inputs[0] == y1 { g.nodes[bi].inputs[1] } else { g.nodes[bi].inputs[0] };
        if !g.is_weight(beta) || g.tensors[beta].numel() != c {
            continue;
        }
        let chain = [mi, di, sq_i, vi, ei, si, qi, gi];
        for &i in &chain {
            if dead[i] {
                continue 'mean;
            }
        }
        let out = g.nodes[bi].outputs[0];
        g.nodes[bi] = crate::ir::Node { name: g.nodes[bi].name.clone(), op: Op::LayerNorm { eps }, inputs: vec![x, gamma, beta], outputs: vec![out] };
        for &i in &chain {
            dead[i] = true;
        }
        hits += 1;
    }
    remove_nodes(g, &dead);
    hits
}

/// A Linear layer as HF exports it — MatMul(x[…, K], W[K, N] const), then Add of an [N]
/// const — becomes View → Gemm → View over the flattened rows, so it runs on the tensor-core
/// GEMM (a 1×1 conv over the rows) instead of the plain matmul kernel. The views are free.
fn fuse_linear(g: &mut Graph) -> usize {
    let uses = use_counts(g);
    let mut consumers: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for &t in &n.inputs {
            consumers.entry(t).or_default().push(i);
        }
    }
    // node index → (x, w, bias, the node whose output is the final one, out)
    let mut plan: Vec<(usize, TensorId, TensorId, Option<TensorId>, usize, TensorId)> = vec![];
    let mut taken = vec![false; g.nodes.len()];
    for mi in 0..g.nodes.len() {
        if !matches!(g.nodes[mi].op, Op::MatMul) {
            continue;
        }
        let (x, w, y) = (g.nodes[mi].inputs[0], g.nodes[mi].inputs[1], g.nodes[mi].outputs[0]);
        if !g.is_weight(w) || g.is_weight(x) {
            continue;
        }
        let (xs, ws) = (g.shape(x).to_vec(), g.shape(w).to_vec());
        if ws.len() != 2 || xs.len() < 2 || xs[xs.len() - 1] != ws[0] || ws[0] % 16 != 0 {
            continue;
        }
        let n = ws[1];
        // an [N] bias right after it (its only use)
        let mut last = mi;
        let mut bias = None;
        if let Some(cs) = consumers.get(&y) {
            if cs.len() == 1 && uses[y] == 1 {
                let ai = cs[0];
                if matches!(g.nodes[ai].op, Op::Binary(BinaryOp::Add)) {
                    let other = if g.nodes[ai].inputs[0] == y { g.nodes[ai].inputs[1] } else { g.nodes[ai].inputs[0] };
                    if g.is_weight(other) && g.tensors[other].numel() == n {
                        bias = Some(other);
                        last = ai;
                    }
                }
            }
        }
        if taken[mi] || taken[last] {
            continue;
        }
        taken[mi] = true;
        taken[last] = true;
        plan.push((mi, x, w, bias, last, g.nodes[last].outputs[0]));
    }
    if plan.is_empty() {
        return 0;
    }
    // rebuild the node list: each MatMul becomes View → Gemm → View at its place, its Add goes
    let mut replace: HashMap<usize, Vec<crate::ir::Node>> = HashMap::new();
    let mut drop = vec![false; g.nodes.len()];
    for (mi, x, w, bias, last, out) in &plan {
        let (xs, ws) = (g.shape(*x).to_vec(), g.shape(*w).to_vec());
        let (k, n) = (ws[0], ws[1]);
        let rows: usize = xs[..xs.len() - 1].iter().product();
        let name = g.nodes[*mi].name.clone();
        let xv = g.add_tensor(format!("{name}.rows"), vec![rows, k], TensorKind::Value);
        g.tensors[xv].alias_of = Some(g.storage_root(*x));
        let gy = g.add_tensor(format!("{name}.gemm"), vec![rows, n], TensorKind::Value);
        g.tensors[*out].alias_of = Some(gy);
        let mut inputs = vec![xv, *w];
        if let Some(b) = bias {
            // the bias may be stored as [1, N] or [1, 1, N]; the gemm wants [N]
            let bid = if g.shape(*b).len() == 1 { *b } else { let d = g.weight(*b).unwrap().to_vec(); g.add_weight(format!("{name}.bias"), vec![n], d) };
            inputs.push(bid);
        }
        replace.insert(
            *mi,
            vec![
                crate::ir::Node { name: format!("{name}.rows"), op: Op::View, inputs: vec![*x], outputs: vec![xv] },
                crate::ir::Node { name: format!("{name}.gemm"), op: Op::Gemm { trans_b: false, act: None }, inputs, outputs: vec![gy] },
                crate::ir::Node { name: format!("{name}.out"), op: Op::View, inputs: vec![gy], outputs: vec![*out] },
            ],
        );
        if last != mi {
            drop[*last] = true;
        }
    }
    let old = std::mem::take(&mut g.nodes);
    for (i, node) in old.into_iter().enumerate() {
        if drop[i] {
            continue;
        }
        match replace.remove(&i) {
            Some(nodes) => g.nodes.extend(nodes),
            None => g.nodes.push(node),
        }
    }
    plan.len()
}

/// Conv → Mul([1,C,1,1] const) and Conv → Add([1,C,1,1] const) folds. BN was
/// decomposed to exactly this at import; the fold math ran in f64 there, so
/// scaling weights here is a single f32 multiply per element.
fn fold_channel_affine_into_conv(g: &mut Graph) -> usize {
    let uses = use_counts(g);
    let prod = producers(g);
    let mut dead = vec![false; g.nodes.len()];
    let mut hits = 0;
    for bi in 0..g.nodes.len() {
        if dead[bi] {
            continue;
        }
        let is_mul = matches!(g.nodes[bi].op, Op::Binary(BinaryOp::Mul));
        let is_add = matches!(g.nodes[bi].op, Op::Binary(BinaryOp::Add));
        if !is_mul && !is_add {
            continue;
        }
        let (x, w) = (g.nodes[bi].inputs[0], g.nodes[bi].inputs[1]);
        // constant on the right, per-channel [1,C,1,1] against a rank-4 x
        if !g.is_weight(w) || g.is_weight(x) {
            continue;
        }
        let ws = g.shape(w).to_vec();
        let cout = match ws.as_slice() {
            [1, c, 1, 1] => *c,
            _ => continue,
        };
        let Some(&ci) = prod.get(&x) else { continue };
        if dead[ci] || uses[x] != 1 {
            continue;
        }
        let Op::Conv { act, .. } = g.nodes[ci].op else { continue };
        if act.is_some() {
            continue; // activation already fused: affine cannot move past it
        }
        let conv_w = g.nodes[ci].inputs[1];
        if g.shape(conv_w)[0] != cout {
            continue;
        }
        let chan = g.weight(w).unwrap().to_vec();
        if is_mul {
            // scale conv weights + bias per out-channel
            let per = g.tensors[g.storage_root(conv_w)].numel() / cout;
            let wid = g.storage_root(conv_w);
            let data = g.weights[wid].as_mut().unwrap();
            for co in 0..cout {
                for v in &mut data[co * per..(co + 1) * per] {
                    *v *= chan[co];
                }
            }
            if let Some(&bias) = g.nodes[ci].inputs.get(2) {
                let bid = g.storage_root(bias);
                let b = g.weights[bid].as_mut().unwrap();
                for co in 0..cout {
                    b[co] *= chan[co];
                }
            }
        } else {
            // add to conv bias (create it if missing)
            match g.nodes[ci].inputs.get(2).copied() {
                Some(bias) => {
                    let bid = g.storage_root(bias);
                    let b = g.weights[bid].as_mut().unwrap();
                    for co in 0..cout {
                        b[co] += chan[co];
                    }
                }
                None => {
                    let name = format!("{}.bias", g.nodes[ci].name);
                    let bid = g.add_weight(name, vec![cout], chan);
                    g.nodes[ci].inputs.push(bid);
                }
            }
        }
        // conv now produces the affine's output directly
        let out = g.nodes[bi].outputs[0];
        g.nodes[ci].outputs[0] = out;
        dead[bi] = true;
        hits += 1;
    }
    remove_nodes(g, &dead);
    hits
}

fn fusable(u: UnaryOp) -> bool {
    matches!(
        u,
        UnaryOp::Relu | UnaryOp::Sigmoid | UnaryOp::Silu | UnaryOp::HardSigmoid | UnaryOp::HardSwish | UnaryOp::Tanh
    )
}

/// Conv/Gemm (no act) followed by a single-consumer fusable unary → epilogue.
fn fuse_producer_act(g: &mut Graph) -> usize {
    let uses = use_counts(g);
    let prod = producers(g);
    let mut dead = vec![false; g.nodes.len()];
    let mut hits = 0;
    for ui in 0..g.nodes.len() {
        if dead[ui] {
            continue;
        }
        let Op::Unary(u) = g.nodes[ui].op else { continue };
        if !fusable(u) {
            continue;
        }
        let x = g.nodes[ui].inputs[0];
        let Some(&pi) = prod.get(&x) else { continue };
        if dead[pi] || uses[x] != 1 {
            continue;
        }
        let out = g.nodes[ui].outputs[0];
        match &mut g.nodes[pi].op {
            Op::Conv { act, .. } | Op::Gemm { act, .. } if act.is_none() => {
                *act = Some(u);
                g.nodes[pi].outputs[0] = out;
                dead[ui] = true;
                hits += 1;
            }
            _ => {}
        }
    }
    remove_nodes(g, &dead);
    hits
}

/// Scalar Mul / Add / Sub by a one-element constant → `ScaleShift`, and a
/// ScaleShift that only feeds another merges into it: x·s₁+t₁ then ·s₂+t₂ =
/// x·(s₁s₂) + (t₁s₂+t₂). HGNetV2 (D-FINE's backbone) follows every conv with
/// one, and on the GPU a scalar op on NHWC storage otherwise costs two layout
/// conversions.
fn fuse_scale_shift(g: &mut Graph) -> usize {
    let mut hits = 0;
    for i in 0..g.nodes.len() {
        let Op::Binary(op) = g.nodes[i].op else { continue };
        let (a, b) = (g.nodes[i].inputs[0], g.nodes[i].inputs[1]);
        let scalar = |t: TensorId| g.weight(t).filter(|w| w.len() == 1).map(|w| w[0]);
        let (x, v, weight_left) = match (scalar(a), scalar(b)) {
            (None, Some(v)) if !g.is_weight(a) => (a, v, false),
            (Some(v), None) if !g.is_weight(b) => (b, v, true),
            _ => continue,
        };
        if g.shape(x) != g.shape(g.nodes[i].outputs[0]) {
            continue;
        }
        let (scale, shift) = match (op, weight_left) {
            (BinaryOp::Mul, _) => (v, 0.0),
            (BinaryOp::Add, _) => (1.0, v),
            (BinaryOp::Sub, false) => (1.0, -v),
            (BinaryOp::Sub, true) => (-1.0, v),
            _ => continue,
        };
        g.nodes[i].op = Op::ScaleShift { scale, shift };
        g.nodes[i].inputs = vec![x];
        hits += 1;
    }
    let uses = use_counts(g);
    let prod = producers(g);
    let mut dead = vec![false; g.nodes.len()];
    for ui in 0..g.nodes.len() {
        let Op::ScaleShift { scale: s2, shift: t2 } = g.nodes[ui].op else { continue };
        let x = g.nodes[ui].inputs[0];
        let Some(&pi) = prod.get(&x) else { continue };
        if dead[pi] || uses[x] != 1 {
            continue;
        }
        let Op::ScaleShift { scale: s1, shift: t1 } = g.nodes[pi].op else { continue };
        g.nodes[pi].op = Op::ScaleShift { scale: s1 * s2, shift: t1 * s2 + t2 };
        g.nodes[pi].outputs[0] = g.nodes[ui].outputs[0];
        dead[ui] = true;
        hits += 1;
    }
    remove_nodes(g, &dead);
    hits
}

/// Drop nodes whose outputs reach nothing. Walk backwards once — the node
/// list is topological, so a single reverse sweep settles reachability.
fn dce(g: &mut Graph) -> usize {
    let mut live = vec![false; g.tensors.len()];
    for &o in &g.outputs {
        live[o] = true;
    }
    let mut dead = vec![false; g.nodes.len()];
    let mut removed = 0;
    for i in (0..g.nodes.len()).rev() {
        let any_live = g.nodes[i].outputs.iter().any(|&o| live[o]);
        if !any_live {
            dead[i] = true;
            removed += 1;
            continue;
        }
        for &inp in &g.nodes[i].inputs {
            live[inp] = true;
            // keep alias roots alive too: a View output borrows its storage
            live[g.storage_root(inp)] = true;
        }
    }
    remove_nodes(g, &dead);
    // free payloads of weights nothing references (BN params after folding)
    let uses = use_counts(g);
    for (i, w) in g.weights.iter_mut().enumerate() {
        if w.is_some() && uses[i] == 0 && g.tensors[i].kind == TensorKind::Weight && !g.outputs.contains(&i) {
            *w = None;
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Node, TensorKind};

    fn value(g: &mut Graph, name: &str, shape: &[usize]) -> TensorId {
        g.add_tensor(name, shape.to_vec(), TensorKind::Value)
    }

    #[test]
    fn silu_and_conv_act_fuse() {
        // conv -> sigmoid/mul pair -> output
        let mut g = Graph::default();
        let x = g.add_tensor("x", vec![1, 2, 4, 4], TensorKind::Input);
        g.inputs.push(x);
        let w = g.add_weight("w", vec![3, 2, 1, 1], vec![1.0; 6]);
        let c = value(&mut g, "c", &[1, 3, 4, 4]);
        g.nodes.push(Node {
            name: "conv".into(),
            op: Op::Conv { group: 1, strides: [1, 1], pads: [0; 4], dilations: [1, 1], act: None },
            inputs: vec![x, w],
            outputs: vec![c],
        });
        let s = value(&mut g, "sig", &[1, 3, 4, 4]);
        g.nodes.push(Node { name: "sig".into(), op: Op::Unary(UnaryOp::Sigmoid), inputs: vec![c], outputs: vec![s] });
        let m = value(&mut g, "mul", &[1, 3, 4, 4]);
        g.nodes.push(Node { name: "mul".into(), op: Op::Binary(BinaryOp::Mul), inputs: vec![c, s], outputs: vec![m] });
        g.outputs.push(m);

        let stats = optimize(&mut g);
        assert_eq!(stats.silu, 1);
        assert_eq!(stats.act_fusions, 1);
        assert_eq!(g.nodes.len(), 1, "one fused conv remains: {:?}", g.nodes);
        match g.nodes[0].op {
            Op::Conv { act: Some(UnaryOp::Silu), .. } => {}
            ref other => panic!("expected fused conv+silu, got {other:?}"),
        }
        assert_eq!(g.nodes[0].outputs[0], m);
    }

    #[test]
    fn bn_affine_folds_into_conv() {
        // conv (no bias) -> mul [1,C,1,1] -> add [1,C,1,1]; C = 2
        let mut g = Graph::default();
        let x = g.add_tensor("x", vec![1, 1, 2, 2], TensorKind::Input);
        g.inputs.push(x);
        let w = g.add_weight("w", vec![2, 1, 1, 1], vec![1.0, 2.0]);
        let c = value(&mut g, "c", &[1, 2, 2, 2]);
        g.nodes.push(Node {
            name: "conv".into(),
            op: Op::Conv { group: 1, strides: [1, 1], pads: [0; 4], dilations: [1, 1], act: None },
            inputs: vec![x, w],
            outputs: vec![c],
        });
        let a = g.add_weight("bn_a", vec![1, 2, 1, 1], vec![10.0, 100.0]);
        let m = value(&mut g, "m", &[1, 2, 2, 2]);
        g.nodes.push(Node { name: "m".into(), op: Op::Binary(BinaryOp::Mul), inputs: vec![c, a], outputs: vec![m] });
        let b = g.add_weight("bn_b", vec![1, 2, 1, 1], vec![0.5, -0.5]);
        let o = value(&mut g, "o", &[1, 2, 2, 2]);
        g.nodes.push(Node { name: "a".into(), op: Op::Binary(BinaryOp::Add), inputs: vec![m, b], outputs: vec![o] });
        g.outputs.push(o);

        let stats = optimize(&mut g);
        assert_eq!(stats.bn_folds, 2);
        assert_eq!(g.nodes.len(), 1);
        // weights scaled: [1*10, 2*100]; bias created: [0.5, -0.5]
        assert_eq!(g.weight(g.nodes[0].inputs[1]).unwrap(), &[10.0, 200.0]);
        assert_eq!(g.weight(g.nodes[0].inputs[2]).unwrap(), &[0.5, -0.5]);
        assert_eq!(g.nodes[0].outputs[0], o);
    }

    #[test]
    fn dce_drops_unreachable() {
        let mut g = Graph::default();
        let x = g.add_tensor("x", vec![4], TensorKind::Input);
        g.inputs.push(x);
        let live = value(&mut g, "live", &[4]);
        g.nodes.push(Node { name: "keep".into(), op: Op::Unary(UnaryOp::Relu), inputs: vec![x], outputs: vec![live] });
        let dead1 = value(&mut g, "dead1", &[4]);
        g.nodes.push(Node { name: "drop1".into(), op: Op::Unary(UnaryOp::Sigmoid), inputs: vec![x], outputs: vec![dead1] });
        let dead2 = value(&mut g, "dead2", &[4]);
        g.nodes.push(Node { name: "drop2".into(), op: Op::Unary(UnaryOp::Relu), inputs: vec![dead1], outputs: vec![dead2] });
        g.outputs.push(live);

        let stats = optimize(&mut g);
        assert_eq!(stats.dce, 2);
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].name, "keep");
    }
}

/// Rewrite the classifier-head ops the GPU executor has no kernel for onto
/// ones it has (GPU plans only; the CPU runs the graph as imported):
/// `GlobalAvgPool` → `AvgPool` over the whole map, and `Gemm` → `MatMul`
/// (+ bias `Add`, + activation), with a transposed copy of a `trans_b` weight.
/// Returns how many nodes were rewritten.
/// A weight Gemm on 2-D rows whose K is a multiple of 16 and whose activation
/// the conv epilogue has runs as a 1×1 implicit GEMM (tensor cores) on the GPU.
pub fn gemm_on_tensor_cores(g: &Graph, inputs: &[TensorId], act: Option<UnaryOp>) -> bool {
    let xs = g.shape(inputs[0]);
    g.is_weight(inputs[1]) && xs.len() == 2 && xs[1] % 16 == 0 && matches!(act, None | Some(UnaryOp::Relu | UnaryOp::Silu | UnaryOp::Sigmoid))
}

pub fn lower_for_gpu(g: &mut Graph) -> usize {
    let mut out = Vec::with_capacity(g.nodes.len());
    let mut n_lowered = 0;
    for node in std::mem::take(&mut g.nodes) {
        match node.op.clone() {
            Op::GlobalAvgPool => {
                let s = g.shape(node.inputs[0]).to_vec();
                let op = Op::AvgPool { kernel: [s[2], s[3]], strides: [1, 1], pads: [0; 4], count_include_pad: false, ceil: false };
                out.push(crate::ir::Node { op, ..node });
                n_lowered += 1;
            }
            // a Gemm the tensor-core path takes (a 1×1 igemm over the flat rows) stays a Gemm
            Op::Gemm { act, .. } if gemm_on_tensor_cores(g, &node.inputs, act) => out.push(node),
            Op::Gemm { trans_b, act } if g.is_weight(node.inputs[1]) => {
                let (x, w, y) = (node.inputs[0], node.inputs[1], node.outputs[0]);
                let ys = g.shape(y).to_vec();
                let rhs = if trans_b {
                    let (ws, data) = (g.shape(w).to_vec(), g.weight(w).unwrap().to_vec());
                    let (n, k) = (ws[0], ws[1]);
                    let mut t = vec![0.0; n * k];
                    for r in 0..n {
                        for c in 0..k {
                            t[c * n + r] = data[r * k + c];
                        }
                    }
                    g.add_weight(format!("{}.wt", node.name), vec![k, n], t)
                } else {
                    w
                };
                let mut last = x;
                let mut steps: Vec<(Op, Vec<TensorId>)> = vec![(Op::MatMul, vec![x, rhs])];
                if let Some(&b) = node.inputs.get(2) {
                    steps.push((Op::Binary(BinaryOp::Add), vec![TensorId::MAX, b]));
                }
                if let Some(u) = act {
                    steps.push((Op::Unary(u), vec![TensorId::MAX]));
                }
                let n_steps = steps.len();
                for (i, (op, mut ins)) in steps.into_iter().enumerate() {
                    for t in ins.iter_mut().filter(|t| **t == TensorId::MAX) {
                        *t = last;
                    }
                    let o = if i + 1 == n_steps { y } else { g.add_tensor(format!("{}.{i}", node.name), ys.clone(), TensorKind::Value) };
                    out.push(crate::ir::Node { name: format!("{}.{i}", node.name), op, inputs: ins, outputs: vec![o] });
                    last = o;
                }
                n_lowered += 1;
            }
            _ => out.push(node),
        }
    }
    g.nodes = out;
    n_lowered
}
