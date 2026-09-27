//! Typed IR for CNN graphs: static shapes, f32 tensors, NCHW layout.
//!
//! The importer lowers ONNX into this; passes rewrite it; the planner and
//! executors consume it. Everything is bound to concrete shapes — there are
//! no symbolic dims past import.

use anyhow::{bail, ensure, Result};

pub type TensorId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Sigmoid,
    Relu,
    Tanh,
    Sqrt,
    /// Exact erf(x) — ONNX `Erf`, not the ggml tanh-GELU.
    Erf,
    /// 0.5·x·(1+erf(x/√2)) — ONNX `Gelu` default.
    GeluErf,
    /// 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³))) — the tanh GELU (SigLIP, GPT-2), fused from its
    /// exported decomposition because x³ overflows f16 long before the GELU does.
    GeluTanh,
    HardSigmoid,
    HardSwish,
    Silu,
    Neg,
    Exp,
    Log,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Mean,
}

/// Activation fused into a producer (conv / gemm) epilogue.
pub type FusedAct = Option<UnaryOp>;

#[derive(Debug, Clone)]
pub enum Op {
    /// inputs: [x, w, (bias)] — w is `[Cout, Cin/group, kh, kw]`.
    Conv {
        group: usize,
        strides: [usize; 2],
        /// [top, left, bottom, right]
        pads: [usize; 4],
        dilations: [usize; 2],
        act: FusedAct,
    },
    /// inputs: [a, b] — numpy matmul over the last two axes, leading axes broadcast.
    MatMul,
    /// inputs: [x, w, (bias)] — y = x·wᵀ when `trans_b`, plus fused activation.
    Gemm { trans_b: bool, act: FusedAct },
    Unary(UnaryOp),
    /// numpy broadcasting between the two inputs.
    Binary(BinaryOp),
    MaxPool {
        kernel: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        ceil: bool,
    },
    AvgPool {
        kernel: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        count_include_pad: bool,
        ceil: bool,
    },
    GlobalAvgPool,
    /// Nearest, asymmetric + floor (the Ultralytics export); integer scale only.
    ResizeNearest { scale_h: usize, scale_w: usize },
    Concat { axis: usize },
    /// One input, N outputs — `parts` are extents along `axis`.
    Split { axis: usize, parts: Vec<usize> },
    /// Normalised at import: one entry per axis of the input, step ≥ 1.
    Slice {
        starts: Vec<usize>,
        ends: Vec<usize>,
        steps: Vec<usize>,
    },
    Transpose { perm: Vec<usize> },
    Softmax { axis: usize },
    ReduceMean { axes: Vec<usize>, keepdims: bool },
    /// Last-axis LayerNorm with weight (+ optional bias) as inputs [x, w, (b)].
    LayerNorm { eps: f32 },
    /// ONNX ReduceSum / ReduceMax / ReduceMean over `axes` (sorted, unique).
    Reduce { kind: ReduceOp, axes: Vec<usize>, keepdims: bool },
    /// inputs [x (N,C,H,W), grid (N,Ho,Wo,2)] → (N,C,Ho,Wo). Bilinear,
    /// zeros padding, align_corners = false: PyTorch's `grid_sampler_2d` as
    /// DETR deformable attention uses it (grid in [-1, 1], x then y).
    GridSample,
    /// The k largest values along the last axis, largest first (ties: the
    /// lower index first, as ONNX TopK). Values only: indices never leave a kernel.
    TopK { k: usize },
    /// inputs [scores (B,N), data (B,N,C)] → (B,k,C): the rows of `data` at
    /// the k largest scores, in score order. ONNX TopK → Unsqueeze/Tile →
    /// GatherElements (DETR query selection) fused, so the integer indices stay
    /// inside the kernel (an f16 activation cannot even hold 8400 exactly).
    TopKGather { k: usize },
    /// y = x·scale + shift (scalar Mul / Add / Sub by a constant, fused by
    /// `passes`: HGNetV2's learnable affine block after every conv).
    ScaleShift { scale: f32, shift: f32 },
    /// ONNX Pad, constant mode, on the spatial axes of NCHW: [top, left, bottom, right].
    Pad { pads: [usize; 4], value: f32 },
    /// Metadata-only reshape/flatten/squeeze/identity: same storage, new shape.
    View,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    /// Graph input, bound by the caller.
    Input,
    /// Initializer payload owned by the graph.
    Weight,
    /// Produced by a node.
    Value,
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: TensorKind,
    /// `View` outputs alias their input's storage. Executors and the planner
    /// resolve through this chain to the root tensor.
    pub alias_of: Option<TensorId>,
}

impl TensorMeta {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub op: Op,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

#[derive(Debug, Default)]
pub struct Graph {
    pub tensors: Vec<TensorMeta>,
    /// Weight payloads, indexed by TensorId (None for non-weights).
    pub weights: Vec<Option<Vec<f32>>>,
    /// Nodes in topological order.
    pub nodes: Vec<Node>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

impl Graph {
    pub fn add_tensor(&mut self, name: impl Into<String>, shape: Vec<usize>, kind: TensorKind) -> TensorId {
        self.tensors.push(TensorMeta { name: name.into(), shape, kind, alias_of: None });
        self.weights.push(None);
        self.tensors.len() - 1
    }

    pub fn add_weight(&mut self, name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) -> TensorId {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        let id = self.add_tensor(name, shape, TensorKind::Weight);
        self.weights[id] = Some(data);
        id
    }

    pub fn shape(&self, id: TensorId) -> &[usize] {
        &self.tensors[id].shape
    }

    /// Follow the alias chain to the tensor that owns storage.
    pub fn storage_root(&self, id: TensorId) -> TensorId {
        let mut t = id;
        while let Some(a) = self.tensors[t].alias_of {
            t = a;
        }
        t
    }

    pub fn is_weight(&self, id: TensorId) -> bool {
        self.weights[self.storage_root(id)].is_some()
    }

    pub fn weight(&self, id: TensorId) -> Option<&[f32]> {
        self.weights[self.storage_root(id)].as_deref()
    }

    /// Bytes of activation storage if every value tensor lived at once
    /// (upper bound; the executor's reuse pool stays well under it).
    pub fn activation_bytes_upper(&self) -> usize {
        self.tensors
            .iter()
            .enumerate()
            .filter(|(i, t)| t.kind == TensorKind::Value && t.alias_of.is_none() && self.weights[*i].is_none())
            .map(|(_, t)| t.numel() * 4)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Shape helpers shared by import-time inference and the executors.
// ---------------------------------------------------------------------------

/// Output spatial extent of a conv/pool window.
pub fn window_out(input: usize, kernel: usize, pad_lo: usize, pad_hi: usize, stride: usize, dilation: usize) -> Result<usize> {
    let eff = (kernel - 1) * dilation + 1;
    let padded = input + pad_lo + pad_hi;
    ensure!(padded >= eff, "window {kernel}x (dil {dilation}) does not fit input {input} with pads {pad_lo}+{pad_hi}");
    Ok((padded - eff) / stride + 1)
}

/// Numpy broadcast of two shapes.
pub fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let db = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            (x, y) => bail!("cannot broadcast {a:?} with {b:?} (dim {i}: {x} vs {y})"),
        };
    }
    Ok(out)
}

/// Row-major strides for a shape.
pub fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Strides for reading `src_shape` broadcast up to `dst_shape` (0-stride on
/// broadcast axes), aligned to dst rank.
pub fn broadcast_strides(src_shape: &[usize], dst_shape: &[usize]) -> Vec<usize> {
    let src_strides = strides_of(src_shape);
    let off = dst_shape.len() - src_shape.len();
    let mut out = vec![0usize; dst_shape.len()];
    for i in 0..src_shape.len() {
        out[off + i] = if src_shape[i] == 1 && dst_shape[off + i] != 1 { 0 } else { src_strides[i] };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_rules() {
        assert_eq!(broadcast(&[1, 84, 8400], &[1, 84, 1]).unwrap(), vec![1, 84, 8400]);
        assert_eq!(broadcast(&[4], &[3, 1]).unwrap(), vec![3, 4]);
        assert!(broadcast(&[2, 3], &[4, 3]).is_err());
    }

    #[test]
    fn window_extents() {
        // 3x3 s2 p1 on 640 -> 320 (YOLO stem)
        assert_eq!(window_out(640, 3, 1, 1, 2, 1).unwrap(), 320);
        // SPPF 5x5 s1 p2 keeps extent
        assert_eq!(window_out(20, 5, 2, 2, 1, 1).unwrap(), 20);
    }

    #[test]
    fn alias_resolution() {
        let mut g = Graph::default();
        let a = g.add_tensor("a", vec![1, 4], TensorKind::Value);
        let b = g.add_tensor("b", vec![4], TensorKind::Value);
        g.tensors[b].alias_of = Some(a);
        let c = g.add_tensor("c", vec![2, 2], TensorKind::Value);
        g.tensors[c].alias_of = Some(b);
        assert_eq!(g.storage_root(c), a);
    }

    #[test]
    fn broadcast_stride_zeroing() {
        // [1,84,1] read as [1,84,8400]: last axis repeats
        assert_eq!(broadcast_strides(&[1, 84, 1], &[1, 84, 8400]), vec![84, 1, 0]);
        // scalar-ish [1] against [2,3]
        assert_eq!(broadcast_strides(&[1], &[2, 3]), vec![0, 0]);
    }
}
