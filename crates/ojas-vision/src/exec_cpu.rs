//! CPU graph executor: walks the optimized IR in order, dispatching to
//! `ojas_cpu::cpu_cnn`. Activation buffers are recycled through a free pool
//! driven by tensor liveness, so steady-state runs allocate nothing new.

use anyhow::{anyhow, bail, ensure, Result};
use ojas_cpu::cpu_cnn::{self, Act, BinOp, ConvShape, PoolShape};

use crate::ir::{broadcast_strides, BinaryOp, Graph, Op, TensorId, TensorKind, UnaryOp};

pub struct CpuExecutor {
    pub threads: usize,
    /// last step (node index) each storage root is read at; usize::MAX = graph output
    last_use: Vec<usize>,
    /// reusable activation buffers
    pool: Vec<Vec<f32>>,
    /// when true, accumulate per-op-kind wall time into `op_times`
    pub profile: bool,
    /// (op kind, total seconds, calls) — filled only when `profile`
    pub op_times: std::collections::HashMap<&'static str, (f64, usize)>,
}

fn op_kind(op: &Op) -> &'static str {
    match op {
        Op::Conv { group, .. } if *group > 1 => "conv(grouped/dw)",
        Op::Conv { .. } => "conv",
        Op::MatMul => "matmul",
        Op::Gemm { .. } => "gemm",
        Op::Unary(_) => "unary",
        Op::Binary(_) => "binary",
        Op::MaxPool { .. } => "maxpool",
        Op::AvgPool { .. } => "avgpool",
        Op::GlobalAvgPool => "gavgpool",
        Op::ResizeNearest { .. } => "resize",
        Op::Concat { .. } => "concat",
        Op::Split { .. } => "split",
        Op::Slice { .. } => "slice",
        Op::Transpose { .. } => "transpose",
        Op::Softmax { .. } => "softmax",
        Op::ReduceMean { .. } => "reduce_mean",
        Op::LayerNorm { .. } => "layernorm",
        Op::View => "view",
        Op::Reduce { .. } => "reduce",
        Op::GridSample => "grid_sample",
        Op::TopK { .. } => "topk",
        Op::TopKGather { .. } => "topk_gather",
        Op::Pad { .. } => "pad",
        Op::ScaleShift { .. } => "scale_shift",
    }
}

fn act_of(u: UnaryOp) -> Option<Act> {
    Some(match u {
        UnaryOp::Relu => Act::Relu,
        UnaryOp::Sigmoid => Act::Sigmoid,
        UnaryOp::Silu => Act::Silu,
        UnaryOp::HardSigmoid => Act::HardSigmoid,
        UnaryOp::HardSwish => Act::HardSwish,
        UnaryOp::Tanh => Act::Tanh,
        _ => return None,
    })
}

impl CpuExecutor {
    pub fn new(g: &Graph, threads: usize) -> Self {
        let mut last_use = vec![0usize; g.tensors.len()];
        for (step, n) in g.nodes.iter().enumerate() {
            for &i in &n.inputs {
                last_use[g.storage_root(i)] = step;
            }
        }
        for &o in &g.outputs {
            last_use[g.storage_root(o)] = usize::MAX;
        }
        CpuExecutor {
            threads: threads.max(1),
            last_use,
            pool: Vec::new(),
            profile: false,
            op_times: Default::default(),
        }
    }

    fn take_buf(&mut self, len: usize) -> Vec<f32> {
        // best-fit from the pool to keep large planes for large tensors
        let mut best: Option<(usize, usize)> = None; // (index, capacity)
        for (i, b) in self.pool.iter().enumerate() {
            let cap = b.capacity();
            if cap >= len && best.is_none_or(|(_, c)| cap < c) {
                best = Some((i, cap));
            }
        }
        match best {
            Some((i, _)) => {
                let mut b = self.pool.swap_remove(i);
                b.clear();
                b.resize(len, 0.0);
                b
            }
            None => vec![0.0; len],
        }
    }

    /// Run the graph. `inputs` follow `g.inputs` order; each must match the
    /// bound shape exactly. Returns the output tensors in `g.outputs` order.
    pub fn run(&mut self, g: &Graph, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        self.run_with_capture(g, inputs, &mut |_, _| {})
    }

    /// `run`, additionally handing every produced tensor (name, data) to
    /// `capture` — the parity gate uses this to localize divergence per node.
    pub fn run_with_capture(
        &mut self,
        g: &Graph,
        inputs: &[&[f32]],
        capture: &mut dyn FnMut(&str, &[f32]),
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(inputs.len() == g.inputs.len(), "expected {} inputs, got {}", g.inputs.len(), inputs.len());
        let mut bufs: Vec<Option<Vec<f32>>> = vec![None; g.tensors.len()];
        for (k, (&id, &data)) in g.inputs.iter().zip(inputs).enumerate() {
            let want = g.tensors[id].numel();
            ensure!(data.len() == want, "input {k}: expected {want} elements ({:?}), got {}", g.shape(id), data.len());
            let mut b = self.take_buf(want);
            b.copy_from_slice(data);
            bufs[id] = Some(b);
        }

        for step in 0..g.nodes.len() {
            let t0 = self.profile.then(std::time::Instant::now);
            self.exec_node(g, step, &mut bufs)?;
            if let Some(t0) = t0 {
                let e = self.op_times.entry(op_kind(&g.nodes[step].op)).or_insert((0.0, 0));
                e.0 += t0.elapsed().as_secs_f64();
                e.1 += 1;
            }
            for &o in &g.nodes[step].outputs {
                let root = g.storage_root(o);
                if let Some(b) = bufs[root].as_deref() {
                    capture(&g.tensors[o].name, &b[..g.tensors[o].numel()]);
                }
            }
            // recycle buffers whose last read was this step
            let node = &g.nodes[step];
            for &i in &node.inputs {
                let root = g.storage_root(i);
                if self.last_use[root] == step && g.tensors[root].kind != TensorKind::Weight {
                    if let Some(b) = bufs[root].take() {
                        self.pool.push(b);
                    }
                }
            }
        }

        let mut outs = Vec::with_capacity(g.outputs.len());
        for &o in &g.outputs {
            let root = g.storage_root(o);
            let data = match (&bufs[root], g.weight(o)) {
                (Some(b), _) => b[..g.tensors[o].numel()].to_vec(),
                (None, Some(w)) => w.to_vec(),
                (None, None) => bail!("output {} was never produced", g.tensors[o].name),
            };
            outs.push(data);
        }
        Ok(outs)
    }

    fn exec_node(&mut self, g: &Graph, step: usize, bufs: &mut Vec<Option<Vec<f32>>>) -> Result<()> {
        // borrow an input tensor's storage (weight or live value buffer)
        macro_rules! data {
            ($id:expr) => {{
                let id: TensorId = $id;
                let root = g.storage_root(id);
                match g.weights[root].as_deref() {
                    Some(w) => w,
                    None => bufs[root]
                        .as_deref()
                        .ok_or_else(|| anyhow!("tensor {} read before it is produced", g.tensors[id].name))?,
                }
            }};
        }

        let node = &g.nodes[step];
        let t = self.threads;
        match &node.op {
            Op::View => {
                // no storage: output aliases input's root. Nothing to compute.
            }
            Op::Conv { group, strides, pads, dilations, act } => {
                let (x, w) = (node.inputs[0], node.inputs[1]);
                let xs = g.shape(x);
                let ws = g.shape(w);
                let s = ConvShape {
                    n: xs[0],
                    cin: xs[1],
                    h: xs[2],
                    w: xs[3],
                    cout: ws[0],
                    kh: ws[2],
                    kw: ws[3],
                    group: *group,
                    stride: *strides,
                    pads: *pads,
                    dilation: *dilations,
                };
                let a = match act {
                    None => Act::None,
                    Some(u) => act_of(*u).ok_or_else(|| anyhow!("conv: unfusable activation {u:?}"))?,
                };
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                let bias = match node.inputs.get(2) {
                    Some(&b) => Some(data!(b)),
                    None => None,
                };
                cpu_cnn::conv2d(data!(x), data!(w), bias, &s, a, t, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::Unary(u) => {
                let x = node.inputs[0];
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                let xin = data!(x);
                match u {
                    UnaryOp::Sigmoid => out.iter_mut().zip(xin).for_each(|(o, &v)| *o = cpu_cnn::sigmoid(v)),
                    UnaryOp::Relu => out.iter_mut().zip(xin).for_each(|(o, &v)| *o = v.max(0.0)),
                    UnaryOp::Tanh => out.iter_mut().zip(xin).for_each(|(o, &v)| *o = v.tanh()),
                    UnaryOp::Sqrt => out.iter_mut().zip(xin).for_each(|(o, &v)| *o = v.sqrt()),
                    UnaryOp::Erf => out.iter_mut().zip(xin).for_each(|(o, &v)| *o = cpu_cnn::erf(v)),
                    UnaryOp::GeluErf => out.iter_mut().zip(xin).for_each(|(o, &v)| *o = cpu_cnn::gelu_erf(v)),
                    UnaryOp::GeluTanh => out.iter_mut().zip(xin).for_each(|(o, &v)| *o = cpu_cnn::gelu_tanh(v)),
                    UnaryOp::HardSigmoid => {
                        out.iter_mut().zip(xin).for_each(|(o, &v)| *o = cpu_cnn::apply_act(Act::HardSigmoid, v))
                    }
                    UnaryOp::HardSwish => {
                        out.iter_mut().zip(xin).for_each(|(o, &v)| *o = cpu_cnn::apply_act(Act::HardSwish, v))
                    }
                    UnaryOp::Silu => out.iter_mut().zip(xin).for_each(|(o, &v)| *o = cpu_cnn::apply_act(Act::Silu, v)),
                    UnaryOp::Neg => out.iter_mut().zip(xin).for_each(|(o, &v)| *o = -v),
                    UnaryOp::Exp => out.iter_mut().zip(xin).for_each(|(o, &v)| *o = v.exp()),
                    UnaryOp::Log => out.iter_mut().zip(xin).for_each(|(o, &v)| *o = v.ln()),
                }
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::Binary(b) => {
                let (ai, bi) = (node.inputs[0], node.inputs[1]);
                let out_id = node.outputs[0];
                let shape = g.shape(out_id).to_vec();
                let sa = broadcast_strides(g.shape(ai), &shape);
                let sb = broadcast_strides(g.shape(bi), &shape);
                let op = match b {
                    BinaryOp::Add => BinOp::Add,
                    BinaryOp::Sub => BinOp::Sub,
                    BinaryOp::Mul => BinOp::Mul,
                    BinaryOp::Div => BinOp::Div,
                    BinaryOp::Pow => BinOp::Pow,
                    BinaryOp::Max => BinOp::Max,
                    BinaryOp::Min => BinOp::Min,
                };
                let mut out = self.take_buf(g.tensors[out_id].numel());
                cpu_cnn::binary_bcast(op, data!(ai), &sa, data!(bi), &sb, &shape, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::MaxPool { kernel, strides, pads, ceil } | Op::AvgPool { kernel, strides, pads, ceil, .. } => {
                let x = node.inputs[0];
                let xs = g.shape(x);
                let s = PoolShape { n: xs[0], c: xs[1], h: xs[2], w: xs[3], kernel: *kernel, stride: *strides, pads: *pads, ceil: *ceil };
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                match &node.op {
                    Op::MaxPool { .. } => cpu_cnn::maxpool2d(data!(x), &s, t, &mut out),
                    Op::AvgPool { count_include_pad, .. } => {
                        cpu_cnn::avgpool2d(data!(x), &s, *count_include_pad, t, &mut out)
                    }
                    _ => unreachable!(),
                }
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::GlobalAvgPool => {
                let x = node.inputs[0];
                let xs = g.shape(x);
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                cpu_cnn::global_avgpool(data!(x), xs[0], xs[1], xs[2] * xs[3], &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::ResizeNearest { scale_h, scale_w } => {
                let x = node.inputs[0];
                let xs = g.shape(x);
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                cpu_cnn::resize_nearest(data!(x), xs[0] * xs[1], xs[2], xs[3], *scale_h, *scale_w, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::Concat { axis } => {
                let shapes: Vec<Vec<usize>> = node.inputs.iter().map(|&i| g.shape(i).to_vec()).collect();
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                {
                    let datas: Vec<&[f32]> = node
                        .inputs
                        .iter()
                        .map(|&i| -> Result<&[f32]> { Ok(data!(i)) })
                        .collect::<Result<_>>()?;
                    let pairs: Vec<(&[f32], &[usize])> =
                        datas.into_iter().zip(shapes.iter().map(|s| s.as_slice())).collect();
                    cpu_cnn::concat(&pairs, *axis, &mut out);
                }
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::Split { axis, parts } => {
                let x = node.inputs[0];
                let xs = g.shape(x).to_vec();
                let mut outs: Vec<Vec<f32>> = node.outputs.iter().map(|&o| self.take_buf(g.tensors[o].numel())).collect();
                {
                    let mut views: Vec<&mut [f32]> = outs.iter_mut().map(|v| v.as_mut_slice()).collect();
                    cpu_cnn::split(data!(x), &xs, *axis, parts, &mut views);
                }
                for (&o, buf) in node.outputs.iter().zip(outs) {
                    bufs[g.storage_root(o)] = Some(buf);
                }
            }
            Op::Slice { starts, ends, steps } => {
                let x = node.inputs[0];
                let xs = g.shape(x).to_vec();
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                cpu_cnn::slice_copy(data!(x), &xs, starts, ends, steps, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::Transpose { perm } => {
                let x = node.inputs[0];
                let xs = g.shape(x).to_vec();
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                cpu_cnn::transpose(data!(x), &xs, perm, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::Softmax { axis } => {
                let x = node.inputs[0];
                let xs = g.shape(x).to_vec();
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                cpu_cnn::softmax_axis(data!(x), &xs, *axis, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::Reduce { kind, axes, .. } => {
                let x = node.inputs[0];
                let xs = g.shape(x).to_vec();
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                crate::detr_ops::reduce(data!(x), &xs, axes, *kind, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::GridSample => {
                let (x, grid) = (node.inputs[0], node.inputs[1]);
                let (xs, gs) = (g.shape(x).to_vec(), g.shape(grid).to_vec());
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                crate::detr_ops::grid_sample(data!(x), &xs, data!(grid), &gs, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::TopK { k } => {
                let x = node.inputs[0];
                let len = *g.shape(x).last().unwrap();
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                crate::detr_ops::topk(data!(x), len, *k, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::ScaleShift { scale, shift } => {
                let x = node.inputs[0];
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                for (o, &v) in out.iter_mut().zip(data!(x)) {
                    *o = v * scale + shift;
                }
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::Pad { pads, value } => {
                let x = node.inputs[0];
                let xs = g.shape(x).to_vec();
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                crate::detr_ops::pad(data!(x), &xs, *pads, *value, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::TopKGather { k } => {
                let (scores, d) = (node.inputs[0], node.inputs[1]);
                let ds = g.shape(d).to_vec();
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                crate::detr_ops::topk_gather(data!(scores), data!(d), ds[0], ds[1], ds[2], *k, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::ReduceMean { axes, .. } => {
                let x = node.inputs[0];
                let xs = g.shape(x).to_vec();
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                cpu_cnn::reduce_mean(data!(x), &xs, axes, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::LayerNorm { eps } => {
                let (x, w) = (node.inputs[0], node.inputs[1]);
                let xs = g.shape(x).to_vec();
                let d = *xs.last().unwrap();
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                let bias = match node.inputs.get(2) {
                    Some(&b) => Some(data!(b)),
                    None => None,
                };
                cpu_cnn::layernorm_lastaxis(data!(x), d, data!(w), bias, *eps, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::MatMul => {
                let (ai, bi) = (node.inputs[0], node.inputs[1]);
                let (sa, sb) = (g.shape(ai).to_vec(), g.shape(bi).to_vec());
                let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
                let n = sb[sb.len() - 1];
                let a_batch: usize = sa[..sa.len() - 2].iter().product::<usize>().max(1);
                let b_batch: usize = sb[..sb.len() - 2].iter().product::<usize>().max(1);
                ensure!(
                    b_batch == a_batch || b_batch == 1,
                    "MatMul: unsupported batch broadcast {sa:?} x {sb:?}"
                );
                let out_id = node.outputs[0];
                let mut out = self.take_buf(g.tensors[out_id].numel());
                cpu_cnn::matmul_batched(data!(ai), data!(bi), a_batch, b_batch, m, k, n, t, &mut out);
                bufs[g.storage_root(out_id)] = Some(out);
            }
            Op::Gemm { trans_b, act } => {
                let (xi, wi) = (node.inputs[0], node.inputs[1]);
                let xs = g.shape(xi).to_vec();
                let ws = g.shape(wi).to_vec();
                let (m, k) = (xs[0], xs[1]);
                let out_id = node.outputs[0];
                let n = g.shape(out_id)[1];
                let a = match act {
                    None => Act::None,
                    Some(u) => act_of(*u).ok_or_else(|| anyhow!("gemm: unfusable activation {u:?}"))?,
                };
                let mut out = self.take_buf(g.tensors[out_id].numel());
                let bias = match node.inputs.get(2) {
                    Some(&b) => Some(data!(b)),
                    None => None,
                };
                if *trans_b {
                    cpu_cnn::gemm_nt(data!(xi), data!(wi), bias, m, k, n, a, t, &mut out);
                } else {
                    ensure!(ws == [k, n], "Gemm: weight shape {ws:?}");
                    cpu_cnn::matmul_batched(data!(xi), data!(wi), 1, 1, m, k, n, t, &mut out);
                    if let Some(b) = bias {
                        for row in out.chunks_exact_mut(n) {
                            for (o, &bv) in row.iter_mut().zip(b) {
                                *o += bv;
                            }
                        }
                    }
                    if a != Act::None {
                        for v in out.iter_mut() {
                            *v = cpu_cnn::apply_act(a, *v);
                        }
                    }
                }
                bufs[g.storage_root(out_id)] = Some(out);
            }
        }
        Ok(())
    }
}
