//! ONNX → IR lowering.
//!
//! Walks the graph in topological order keeping, per ONNX value name, its
//! inferred static shape, its IR tensor (if it exists at runtime), and — when
//! the value is computable at import time — its constant payload. Shape-math
//! subgraphs (Shape → Gather → Concat → Reshape and friends) fold away here;
//! everything else lowers to a typed `ir::Op` with concrete shapes.
//!
//! Symbolic dims (`dim_param`) must be bound by the caller or import fails.

use std::collections::HashMap;

use anyhow::{anyhow, bail, ensure, Context, Result};
use ojas_formats::onnx::{AttrValue, OnnxModel, OnnxNode, OnnxDim, DT_FLOAT};

use crate::ir::{broadcast, window_out, BinaryOp, Graph, Op, ReduceOp, TensorId, TensorKind, UnaryOp};

/// A value computable at import time.
#[derive(Debug, Clone)]
enum SVal {
    I(Vec<i64>),
    F(Vec<f32>),
}

impl SVal {
    fn as_i64(&self) -> Result<&[i64]> {
        match self {
            SVal::I(v) => Ok(v),
            SVal::F(_) => bail!("expected integer constant, found float"),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Val {
    shape: Vec<usize>,
    id: Option<TensorId>,
    sval: Option<SVal>,
    /// TopK indices (scores tensor, k), carried through Unsqueeze / Tile /
    /// Expand until a GatherElements consumes them as one fused TopKGather.
    topk: Option<(TensorId, usize)>,
}

struct Ctx {
    g: Graph,
    env: HashMap<String, Val>,
    opset: i64,
    /// Every value some node or the graph output reads (unread TopK values are not computed).
    used: std::collections::HashSet<String>,
}

/// Keep import-time constants only when small — Resize scales, shape vectors,
/// DFL aranges. Big f32 initializers stay weights-only.
const SVAL_MAX_ELEMS: usize = 65_536;

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

fn attr_i(n: &OnnxNode, name: &str, default: i64) -> Result<i64> {
    match n.attr(name) {
        None => Ok(default),
        Some(AttrValue::I(v)) => Ok(*v),
        Some(other) => bail!("node {}: attribute {name} has wrong type {other:?}", n.op_type),
    }
}

fn attr_f(n: &OnnxNode, name: &str, default: f32) -> Result<f32> {
    match n.attr(name) {
        None => Ok(default),
        Some(AttrValue::F(v)) => Ok(*v),
        Some(other) => bail!("node {}: attribute {name} has wrong type {other:?}", n.op_type),
    }
}

fn attr_is(n: &OnnxNode, name: &str) -> Result<Option<Vec<i64>>> {
    match n.attr(name) {
        None => Ok(None),
        Some(AttrValue::Ints(v)) => Ok(Some(v.clone())),
        Some(other) => bail!("node {}: attribute {name} has wrong type {other:?}", n.op_type),
    }
}

fn attr_s<'a>(n: &'a OnnxNode, name: &str, default: &'a str) -> Result<&'a str> {
    match n.attr(name) {
        None => Ok(default),
        Some(AttrValue::S(v)) => Ok(v.as_str()),
        Some(other) => bail!("node {}: attribute {name} has wrong type {other:?}", n.op_type),
    }
}

fn norm_axis(axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    ensure!(a >= 0 && (a as usize) < rank, "axis {axis} out of range for rank {rank}");
    Ok(a as usize)
}

fn to_usize_shape(v: &[i64], what: &str) -> Result<Vec<usize>> {
    v.iter()
        .map(|&d| usize::try_from(d).map_err(|_| anyhow!("{what}: negative dim {d}")))
        .collect()
}

/// [kh, kw] / [sh, sw] pairs from list attrs, defaulting to `default` each.
fn pair(v: Option<Vec<i64>>, default: usize, what: &str) -> Result<[usize; 2]> {
    match v {
        None => Ok([default, default]),
        Some(v) => {
            ensure!(v.len() == 2, "{what}: expected 2 spatial values, got {v:?} (only 2-D convs/pools)");
            Ok([usize::try_from(v[0])?, usize::try_from(v[1])?])
        }
    }
}

/// Effective [t, l, b, r] pads for a conv/pool node: explicit `pads` when
/// auto_pad is NOTSET, else computed from SAME_UPPER / SAME_LOWER / VALID
/// (static shapes make the SAME pads concrete at import).
fn resolve_pads(
    n: &OnnxNode,
    input_hw: [usize; 2],
    kernel: [usize; 2],
    strides: [usize; 2],
    dilations: [usize; 2],
    what: &str,
) -> Result<[usize; 4]> {
    match attr_s(n, "auto_pad", "NOTSET")? {
        "NOTSET" => pads4(attr_is(n, "pads")?, what),
        "VALID" => Ok([0; 4]),
        mode @ ("SAME_UPPER" | "SAME_LOWER") => {
            let mut p = [0usize; 4];
            for axis in 0..2 {
                let eff = (kernel[axis] - 1) * dilations[axis] + 1;
                let out = input_hw[axis].div_ceil(strides[axis]);
                let total = ((out - 1) * strides[axis] + eff).saturating_sub(input_hw[axis]);
                let lo = if mode == "SAME_UPPER" { total / 2 } else { total.div_ceil(2) };
                p[axis] = lo; // top / left
                p[axis + 2] = total - lo; // bottom / right
            }
            Ok(p)
        }
        other => bail!("{what}: auto_pad {other:?} unsupported"),
    }
}

/// ONNX pads [t, l, b, r] for 2-D ops.
fn pads4(v: Option<Vec<i64>>, what: &str) -> Result<[usize; 4]> {
    match v {
        None => Ok([0; 4]),
        Some(v) => {
            ensure!(v.len() == 4, "{what}: expected 4 pad values, got {v:?}");
            Ok([
                usize::try_from(v[0])?,
                usize::try_from(v[1])?,
                usize::try_from(v[2])?,
                usize::try_from(v[3])?,
            ])
        }
    }
}

// ---------------------------------------------------------------------------
// Ctx: value bookkeeping
// ---------------------------------------------------------------------------

impl Ctx {
    fn val(&self, name: &str) -> Result<&Val> {
        self.env
            .get(name)
            .ok_or_else(|| anyhow!("value {name:?} used before it is produced"))
    }

    fn shape_of(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.val(name)?.shape.clone())
    }

    fn sval(&self, name: &str) -> Result<&SVal> {
        self.val(name)?
            .sval
            .as_ref()
            .ok_or_else(|| anyhow!("value {name:?} must be a compile-time constant here (dynamic shapes are not supported)"))
    }

    fn ints(&self, name: &str) -> Result<Vec<i64>> {
        Ok(self.sval(name)?.as_i64()?.to_vec())
    }

    /// Runtime tensor id for a name, materializing folded f32 constants into
    /// weights when a real op consumes them.
    fn tensor(&mut self, name: &str) -> Result<TensorId> {
        let v = self.val(name)?.clone();
        if let Some(id) = v.id {
            return Ok(id);
        }
        ensure!(v.topk.is_none(), "value {name:?}: TopK indices used other than to gather rows (GatherElements) — unsupported");
        match v.sval {
            Some(SVal::F(data)) => {
                let id = self.g.add_weight(name, v.shape.clone(), data);
                self.env.get_mut(name).unwrap().id = Some(id);
                Ok(id)
            }
            Some(SVal::I(_)) => bail!("value {name:?} is an integer tensor consumed at runtime — unsupported"),
            None => bail!("value {name:?} has no runtime tensor"),
        }
    }

    fn bind_output(&mut self, name: &str, id: TensorId, shape: Vec<usize>) {
        self.env.insert(name.to_string(), Val { shape, id: Some(id), sval: None, topk: None });
    }

    fn bind_const(&mut self, name: &str, shape: Vec<usize>, sval: SVal) {
        self.env.insert(name.to_string(), Val { shape, id: None, sval: Some(sval), topk: None });
    }

    fn value_out(&mut self, name: &str, shape: Vec<usize>) -> TensorId {
        let id = self.g.add_tensor(name, shape.clone(), TensorKind::Value);
        self.bind_output(name, id, shape);
        id
    }

    /// A view output: same storage as `src`, new shape.
    fn view_out(&mut self, node_name: &str, name: &str, src: TensorId, shape: Vec<usize>) {
        let root = self.g.storage_root(src);
        let id = self.g.add_tensor(name, shape.clone(), TensorKind::Value);
        self.g.tensors[id].alias_of = Some(root);
        self.g.nodes.push(crate::ir::Node {
            name: node_name.to_string(),
            op: Op::View,
            inputs: vec![src],
            outputs: vec![id],
        });
        self.bind_output(name, id, shape);
    }
}

// ---------------------------------------------------------------------------
// Constant evaluation of exporter shape-math
// ---------------------------------------------------------------------------

/// Try to compute this node at import time. Returns true if fully handled
/// (outputs bound as constants, no IR emitted).
fn try_const_eval(ctx: &mut Ctx, n: &OnnxNode) -> Result<bool> {
    let all_static = |ctx: &Ctx, names: &[String]| names.iter().filter(|s| !s.is_empty()).all(|s| {
        ctx.env.get(s.as_str()).is_some_and(|v| v.sval.is_some())
    });

    match n.op_type.as_str() {
        "Constant" => {
            let t = match n.attr("value") {
                Some(AttrValue::T(t)) => t,
                _ => bail!("Constant node {} without tensor value", n.name),
            };
            let shape = to_usize_shape(&t.dims, "Constant")?;
            let sval = if t.dtype == DT_FLOAT || t.dtype == ojas_formats::onnx::DT_FLOAT16 || t.dtype == ojas_formats::onnx::DT_DOUBLE {
                SVal::F(t.f32_data()?)
            } else {
                SVal::I(t.i64_data()?)
            };
            ctx.bind_const(&n.outputs[0], shape, sval);
            Ok(true)
        }
        "Shape" => {
            let shape = ctx.shape_of(&n.inputs[0])?;
            let start = attr_i(n, "start", 0)?;
            let end = attr_i(n, "end", shape.len() as i64)?;
            let s = norm_axis(start.min(shape.len() as i64 - 1).max(-(shape.len() as i64)), shape.len())?;
            let e = if end >= shape.len() as i64 { shape.len() } else { norm_axis(end, shape.len())? };
            let dims: Vec<i64> = shape[s..e].iter().map(|&d| d as i64).collect();
            ctx.bind_const(&n.outputs[0], vec![dims.len()], SVal::I(dims));
            Ok(true)
        }
        // Pure data ops: fold only when every input is already constant.
        "Gather" | "Cast" | "Concat" | "Unsqueeze" | "Squeeze" | "Reshape" | "Slice" | "Add" | "Sub" | "Mul"
        | "Div" | "Range" | "ConstantOfShape" | "Expand" | "Identity" | "Floor" | "Ceil"
            if all_static(ctx, &n.inputs) && !n.inputs.is_empty() =>
        {
            const_eval_data(ctx, n)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn const_eval_data(ctx: &mut Ctx, n: &OnnxNode) -> Result<()> {
    let get = |ctx: &Ctx, i: usize| -> Result<(Vec<usize>, SVal)> {
        let v = ctx.val(&n.inputs[i])?;
        Ok((v.shape.clone(), v.sval.clone().unwrap()))
    };
    let out = n.outputs[0].clone();
    match n.op_type.as_str() {
        "Identity" => {
            let (shape, v) = get(ctx, 0)?;
            ctx.bind_const(&out, shape, v);
        }
        "Cast" => {
            let to = attr_i(n, "to", 0)? as u32;
            let (shape, v) = get(ctx, 0)?;
            let v = match (v, to) {
                (SVal::I(x), DT_FLOAT) => SVal::F(x.iter().map(|&i| i as f32).collect()),
                (SVal::F(x), t) if t == ojas_formats::onnx::DT_INT64 || t == ojas_formats::onnx::DT_INT32 => {
                    SVal::I(x.iter().map(|&f| f as i64).collect())
                }
                (v, _) => v, // same-family cast
            };
            ctx.bind_const(&out, shape, v);
        }
        "Gather" => {
            let (dshape, data) = get(ctx, 0)?;
            let (ishape, idx) = get(ctx, 1)?;
            let axis = norm_axis(attr_i(n, "axis", 0)?, dshape.len().max(1))?;
            ensure!(dshape.len() == 1 && axis == 0, "const Gather: only 1-D data supported (shape math)");
            let idx = idx.as_i64()?.to_vec();
            let pick = |i: i64| -> Result<usize> {
                let i = if i < 0 { i + dshape[0] as i64 } else { i };
                ensure!(i >= 0 && (i as usize) < dshape[0], "const Gather index {i} out of range");
                Ok(i as usize)
            };
            let sval = match data {
                SVal::I(d) => SVal::I(idx.iter().map(|&i| Ok(d[pick(i)?])).collect::<Result<_>>()?),
                SVal::F(d) => SVal::F(idx.iter().map(|&i| Ok(d[pick(i)?])).collect::<Result<_>>()?),
            };
            ctx.bind_const(&out, ishape, sval);
        }
        "Concat" => {
            let mut dims = Vec::new();
            let mut all_i = Vec::new();
            let mut all_f = Vec::new();
            let mut float = false;
            for (k, name) in n.inputs.iter().enumerate() {
                let v = ctx.val(name)?;
                ensure!(v.shape.len() == 1, "const Concat: only 1-D supported");
                dims.push(v.shape[0]);
                match v.sval.clone().unwrap() {
                    SVal::I(x) => all_i.extend(x),
                    SVal::F(x) => {
                        float = true;
                        all_f.extend(x);
                    }
                }
                ensure!(!(float && k > 0 && !all_i.is_empty()), "const Concat: mixed dtypes");
            }
            let total: usize = dims.iter().sum();
            let sval = if float { SVal::F(all_f) } else { SVal::I(all_i) };
            ctx.bind_const(&out, vec![total], sval);
        }
        "Unsqueeze" | "Squeeze" | "Reshape" => {
            // Shape-only change on constants: payload is unchanged; recompute shape.
            let (shape, v) = get(ctx, 0)?;
            let numel: usize = shape.iter().product();
            let new_shape = match n.op_type.as_str() {
                "Unsqueeze" => {
                    let axes = if n.inputs.len() > 1 { ctx.ints(&n.inputs[1])? } else { attr_is(n, "axes")?.unwrap_or_default() };
                    let rank = shape.len() + axes.len();
                    let mut set: Vec<usize> = axes.iter().map(|&a| norm_axis(a, rank)).collect::<Result<_>>()?;
                    set.sort_unstable();
                    let mut s = shape.clone();
                    for &a in &set {
                        s.insert(a, 1);
                    }
                    s
                }
                "Squeeze" => shape.iter().copied().filter(|&d| d != 1).collect(),
                _ => {
                    let target = ctx.ints(&n.inputs[1])?;
                    resolve_reshape(&shape, &target)?
                }
            };
            ensure!(new_shape.iter().product::<usize>() == numel, "const reshape changes element count");
            ctx.bind_const(&out, new_shape, v);
        }
        "Slice" => {
            let (shape, v) = get(ctx, 0)?;
            ensure!(shape.len() == 1, "const Slice: only 1-D supported");
            let starts = ctx.ints(&n.inputs[1])?;
            let ends = ctx.ints(&n.inputs[2])?;
            let d = shape[0] as i64;
            let clamp = |x: i64| -> usize { (if x < 0 { x + d } else { x }).clamp(0, d) as usize };
            let (s, e) = (clamp(starts[0]), clamp(ends[0]));
            ensure!(s <= e, "const Slice: start {s} > end {e}");
            let sval = match v {
                SVal::I(x) => SVal::I(x[s..e].to_vec()),
                SVal::F(x) => SVal::F(x[s..e].to_vec()),
            };
            ctx.bind_const(&out, vec![e - s], sval);
        }
        "Add" | "Sub" | "Mul" | "Div" => {
            let (ashape, a) = get(ctx, 0)?;
            let (bshape, b) = get(ctx, 1)?;
            let shape = broadcast(&ashape, &bshape)?;
            let numel: usize = shape.iter().product();
            ensure!(numel <= SVAL_MAX_ELEMS, "const arithmetic too large");
            let idx = |s: &[usize], flat: usize| -> usize {
                // broadcast read for 0-d/1-d cases used in shape math
                let n: usize = s.iter().product();
                if n <= 1 { 0 } else { flat % n }
            };
            let sval = match (a, b) {
                (SVal::I(x), SVal::I(y)) => SVal::I((0..numel)
                    .map(|f| {
                        let (p, q) = (x[idx(&ashape, f)], y[idx(&bshape, f)]);
                        match n.op_type.as_str() {
                            "Add" => p + q,
                            "Sub" => p - q,
                            "Mul" => p * q,
                            _ => p / q,
                        }
                    })
                    .collect()),
                (a, b) => {
                    let xf: Vec<f32> = match a { SVal::F(v) => v, SVal::I(v) => v.iter().map(|&i| i as f32).collect() };
                    let yf: Vec<f32> = match b { SVal::F(v) => v, SVal::I(v) => v.iter().map(|&i| i as f32).collect() };
                    SVal::F((0..numel)
                        .map(|f| {
                            let (p, q) = (xf[idx(&ashape, f)], yf[idx(&bshape, f)]);
                            match n.op_type.as_str() {
                                "Add" => p + q,
                                "Sub" => p - q,
                                "Mul" => p * q,
                                _ => p / q,
                            }
                        })
                        .collect())
                }
            };
            ctx.bind_const(&out, shape, sval);
        }
        "Range" => {
            let start = ctx.ints(&n.inputs[0])?[0];
            let limit = ctx.ints(&n.inputs[1])?[0];
            let delta = ctx.ints(&n.inputs[2])?[0];
            ensure!(delta != 0, "Range with zero delta");
            let mut v = Vec::new();
            let mut x = start;
            while (delta > 0 && x < limit) || (delta < 0 && x > limit) {
                ensure!(v.len() < SVAL_MAX_ELEMS, "Range too large");
                v.push(x);
                x += delta;
            }
            ctx.bind_const(&out, vec![v.len()], SVal::I(v));
        }
        "ConstantOfShape" => {
            let shape = to_usize_shape(&ctx.ints(&n.inputs[0])?, "ConstantOfShape")?;
            let numel: usize = shape.iter().product();
            ensure!(numel <= SVAL_MAX_ELEMS, "ConstantOfShape too large");
            let sval = match n.attr("value") {
                Some(AttrValue::T(t)) if t.dtype == DT_FLOAT => SVal::F(vec![t.f32_data()?[0]; numel]),
                Some(AttrValue::T(t)) => SVal::I(vec![t.i64_data()?[0]; numel]),
                None => SVal::F(vec![0.0; numel]),
                other => bail!("ConstantOfShape: bad value attr {other:?}"),
            };
            ctx.bind_const(&out, shape, sval);
        }
        "Expand" => {
            let (ishape, v) = get(ctx, 0)?;
            let target = to_usize_shape(&ctx.ints(&n.inputs[1])?, "Expand")?;
            let shape = broadcast(&ishape, &target)?;
            let numel: usize = shape.iter().product();
            ensure!(numel <= SVAL_MAX_ELEMS, "const Expand too large");
            let src_n: usize = ishape.iter().product::<usize>().max(1);
            let sval = match v {
                SVal::I(x) => SVal::I((0..numel).map(|f| x[f % src_n]).collect()),
                SVal::F(x) => SVal::F((0..numel).map(|f| x[f % src_n]).collect()),
            };
            ctx.bind_const(&out, shape, sval);
        }
        "Floor" | "Ceil" => {
            let (shape, v) = get(ctx, 0)?;
            let sval = match v {
                SVal::F(x) => SVal::F(x.iter().map(|&f| if n.op_type == "Floor" { f.floor() } else { f.ceil() }).collect()),
                v @ SVal::I(_) => v,
            };
            ctx.bind_const(&out, shape, sval);
        }
        other => bail!("const eval: unhandled op {other}"),
    }
    Ok(())
}

/// ONNX Reshape target semantics: 0 copies the input dim, one -1 is inferred.
fn resolve_reshape(input: &[usize], target: &[i64]) -> Result<Vec<usize>> {
    let numel: usize = input.iter().product();
    let mut out = Vec::with_capacity(target.len());
    let mut infer = None;
    for (i, &t) in target.iter().enumerate() {
        match t {
            0 => {
                ensure!(i < input.len(), "Reshape: 0-dim {i} beyond input rank");
                out.push(input[i]);
            }
            -1 => {
                ensure!(infer.is_none(), "Reshape: more than one -1");
                infer = Some(i);
                out.push(1);
            }
            d if d > 0 => out.push(d as usize),
            d => bail!("Reshape: bad target dim {d}"),
        }
    }
    let known: usize = out.iter().product();
    if let Some(i) = infer {
        ensure!(known > 0 && numel % known == 0, "Reshape: cannot infer -1 ({numel} vs {known})");
        out[i] = numel / known;
    } else {
        ensure!(known == numel, "Reshape: element count mismatch ({numel} -> {known})");
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Import driver
// ---------------------------------------------------------------------------

/// Lower an ONNX model to the IR, binding every symbolic dim via `binds`
/// (e.g. `{"batch": 1}`). Shapes are static from here on.
pub fn import(model: &OnnxModel, binds: &HashMap<String, usize>) -> Result<Graph> {
    let used = model.graph.nodes.iter().flat_map(|n| n.inputs.iter().cloned()).chain(model.graph.outputs.iter().map(|o| o.name.clone())).collect();
    let mut ctx = Ctx { g: Graph::default(), env: HashMap::new(), opset: model.default_opset().unwrap_or(13), used };

    // Initializers: f32 → weights (+ small const payload); ints → const only.
    let initializer_names: std::collections::HashSet<&str> =
        model.graph.initializers.iter().map(|t| t.name.as_str()).collect();
    for t in &model.graph.initializers {
        let shape = to_usize_shape(&t.dims, &t.name)?;
        let numel: usize = shape.iter().product();
        match t.dtype {
            DT_FLOAT | ojas_formats::onnx::DT_FLOAT16 | ojas_formats::onnx::DT_DOUBLE => {
                let data = t.f32_data().with_context(|| format!("initializer {}", t.name))?;
                let sval = (numel <= SVAL_MAX_ELEMS).then(|| SVal::F(data.clone()));
                let id = ctx.g.add_weight(&t.name, shape.clone(), data);
                ctx.env.insert(t.name.clone(), Val { shape, id: Some(id), sval, topk: None });
            }
            _ => {
                let data = t.i64_data().with_context(|| format!("initializer {}", t.name))?;
                ctx.bind_const(&t.name, shape, SVal::I(data));
            }
        }
    }

    // Graph inputs (initializers may legally repeat in `input`; skip those).
    for vi in &model.graph.inputs {
        if initializer_names.contains(vi.name.as_str()) {
            continue;
        }
        let mut shape = Vec::with_capacity(vi.dims.len());
        for (i, d) in vi.dims.iter().enumerate() {
            match d {
                OnnxDim::Value(v) => shape.push(usize::try_from(*v).map_err(|_| anyhow!("input {}: bad dim {v}", vi.name))?),
                OnnxDim::Param(p) => match binds.get(p) {
                    Some(&v) => shape.push(v),
                    None => bail!("input {}: unbound dim_param {p:?} — bind it (e.g. batch=1)", vi.name),
                },
                OnnxDim::Unknown => bail!("input {}: dim {i} has neither value nor name", vi.name),
            }
        }
        let id = ctx.g.add_tensor(&vi.name, shape.clone(), TensorKind::Input);
        ctx.g.inputs.push(id);
        ctx.bind_output(&vi.name, id, shape);
    }

    for n in &model.graph.nodes {
        let ctx_msg = || format!("lowering node {} ({})", if n.name.is_empty() { &n.outputs[0] } else { &n.name }, n.op_type);
        if carry_topk(&mut ctx, n).with_context(ctx_msg)? {
            continue;
        }
        if try_const_eval(&mut ctx, n).with_context(ctx_msg)? {
            continue;
        }
        lower_node(&mut ctx, n).with_context(ctx_msg)?;
    }

    for o in &model.graph.outputs {
        let id = ctx
            .tensor(&o.name)
            .with_context(|| format!("graph output {}", o.name))?;
        ctx.g.outputs.push(id);
    }
    Ok(ctx.g)
}

/// TopK indices flow through shape-only ops (Unsqueeze, Expand, Tile …) to
/// the GatherElements that uses them; bind the marker instead of a tensor.
fn carry_topk(ctx: &mut Ctx, n: &OnnxNode) -> Result<bool> {
    const SHAPE_ONLY: &[&str] = &["Unsqueeze", "Squeeze", "Reshape", "Expand", "Tile", "Cast", "Identity"];
    let Some(first) = n.inputs.first() else { return Ok(false) };
    let Some(marker) = ctx.env.get(first.as_str()).and_then(|v| v.topk) else { return Ok(false) };
    ensure!(SHAPE_ONLY.contains(&n.op_type.as_str()), "{}: TopK indices used by {} — only GatherElements (row gather) is supported", n.op_type, n.op_type);
    let xs = ctx.shape_of(first)?;
    let axes = |ctx: &Ctx| -> Result<Vec<i64>> {
        Ok(if n.inputs.len() > 1 && !n.inputs[1].is_empty() { ctx.ints(&n.inputs[1])? } else { attr_is(n, "axes")?.unwrap_or_default() })
    };
    let shape = match n.op_type.as_str() {
        "Unsqueeze" => {
            let ax = axes(ctx)?;
            let rank = xs.len() + ax.len();
            let mut at: Vec<usize> = ax.iter().map(|&a| norm_axis(a, rank)).collect::<Result<_>>()?;
            at.sort_unstable();
            let mut s = xs.clone();
            for a in at {
                s.insert(a.min(s.len()), 1);
            }
            s
        }
        "Squeeze" => {
            let drop: Vec<usize> = axes(ctx)?.iter().map(|&a| norm_axis(a, xs.len())).collect::<Result<_>>()?;
            xs.iter().enumerate().filter(|(i, &d)| !(drop.contains(i) || drop.is_empty() && d == 1)).map(|(_, &d)| d).collect()
        }
        "Reshape" => resolve_reshape(&xs, &ctx.ints(&n.inputs[1])?)?,
        "Expand" => broadcast(&xs, &to_usize_shape(&ctx.ints(&n.inputs[1])?, "Expand")?)?,
        "Tile" => {
            let reps = ctx.ints(&n.inputs[1])?;
            ensure!(reps.len() == xs.len(), "Tile: repeats {reps:?} vs rank {}", xs.len());
            xs.iter().zip(&reps).map(|(&d, &r)| d * r as usize).collect()
        }
        _ => xs.clone(),
    };
    ctx.env.insert(n.outputs[0].clone(), Val { shape, id: None, sval: None, topk: Some(marker) });
    Ok(true)
}

/// Normalised, sorted, unique reduction axes (from the input or the attribute; none = all).
fn reduce_axes(ctx: &Ctx, n: &OnnxNode, rank: usize) -> Result<Vec<usize>> {
    let raw = if n.inputs.len() > 1 && !n.inputs[1].is_empty() {
        ctx.ints(&n.inputs[1])?
    } else {
        attr_is(n, "axes")?.unwrap_or_else(|| (0..rank as i64).collect())
    };
    let mut axes: Vec<usize> = raw.iter().map(|&a| norm_axis(a, rank)).collect::<Result<_>>()?;
    axes.sort_unstable();
    axes.dedup();
    Ok(axes)
}

/// A scalar constant input (Clip bounds): None when the input is absent.
fn scalar_input(ctx: &Ctx, n: &OnnxNode, i: usize) -> Result<Option<f32>> {
    match n.inputs.get(i).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(name) => {
            let v: Vec<f32> = match ctx.sval(name)? {
                SVal::F(v) => v.clone(),
                SVal::I(v) => v.iter().map(|&x| x as f32).collect(),
            };
            ensure!(v.len() == 1, "{}: bound {name:?} is not a scalar", n.op_type);
            Ok(Some(v[0]))
        }
    }
}

// ---------------------------------------------------------------------------
// Per-op lowering
// ---------------------------------------------------------------------------

fn lower_node(ctx: &mut Ctx, n: &OnnxNode) -> Result<()> {
    let node_name = if n.name.is_empty() { n.outputs[0].clone() } else { n.name.clone() };
    let emit = |ctx: &mut Ctx, op: Op, inputs: Vec<TensorId>, out_shapes: Vec<Vec<usize>>, outs: &[String]| {
        let mut outputs = Vec::with_capacity(outs.len());
        for (name, shape) in outs.iter().zip(out_shapes) {
            outputs.push(ctx.value_out(name, shape));
        }
        ctx.g.nodes.push(crate::ir::Node { name: node_name.clone(), op, inputs, outputs });
    };

    match n.op_type.as_str() {
        "Conv" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let w = ctx.tensor(&n.inputs[1])?;
            let bias = if n.inputs.len() > 2 && !n.inputs[2].is_empty() { Some(ctx.tensor(&n.inputs[2])?) } else { None };
            let xs = ctx.g.shape(x).to_vec();
            let ws = ctx.g.shape(w).to_vec();
            ensure!(xs.len() == 4 && ws.len() == 4, "Conv: only 2-D convs (x {xs:?}, w {ws:?})");
            let group = usize::try_from(attr_i(n, "group", 1)?)?;
            let strides = pair(attr_is(n, "strides")?, 1, "Conv strides")?;
            let dilations = pair(attr_is(n, "dilations")?, 1, "Conv dilations")?;
            let pads = resolve_pads(n, [xs[2], xs[3]], [ws[2], ws[3]], strides, dilations, "Conv pads")?;
            if let Some(k) = attr_is(n, "kernel_shape")? {
                ensure!(k.len() == 2 && k[0] as usize == ws[2] && k[1] as usize == ws[3], "Conv: kernel_shape {k:?} vs weight {ws:?}");
            }
            ensure!(xs[1] == ws[1] * group, "Conv: channels {} vs weight Cin {} x group {group}", xs[1], ws[1]);
            let oh = window_out(xs[2], ws[2], pads[0], pads[2], strides[0], dilations[0])?;
            let ow = window_out(xs[3], ws[3], pads[1], pads[3], strides[1], dilations[1])?;
            let out_shape = vec![xs[0], ws[0], oh, ow];
            let mut inputs = vec![x, w];
            if let Some(b) = bias {
                ensure!(ctx.g.shape(b) == [ws[0]], "Conv: bias shape mismatch");
                inputs.push(b);
            }
            emit(ctx, Op::Conv { group, strides, pads, dilations, act: None }, inputs, vec![out_shape], &n.outputs);
        }
        "BatchNormalization" => {
            // Decompose to per-channel scale·x + shift; a later pass folds it
            // into a preceding conv. Fold math in f64.
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            ensure!(xs.len() >= 2, "BatchNormalization: rank < 2");
            let eps = attr_f(n, "epsilon", 1e-5)? as f64;
            let mut bn_param = |i: usize, what: &str| -> Result<Vec<f32>> {
                let id = ctx.tensor(&n.inputs[i])?;
                Ok(ctx.g.weight(id).ok_or_else(|| anyhow!("BN {what} must be constant"))?.to_vec())
            };
            let scale = bn_param(1, "scale")?;
            let b = bn_param(2, "bias")?;
            let mean = bn_param(3, "mean")?;
            let var = bn_param(4, "var")?;
            let c = xs[1];
            ensure!(scale.len() == c && b.len() == c && mean.len() == c && var.len() == c, "BN: channel mismatch");
            let mut a = vec![0f32; c];
            let mut sh = vec![0f32; c];
            for i in 0..c {
                let s = scale[i] as f64 / (var[i] as f64 + eps).sqrt();
                a[i] = s as f32;
                sh[i] = (b[i] as f64 - mean[i] as f64 * s) as f32;
            }
            let mut wshape = vec![1usize; xs.len()];
            wshape[1] = c;
            let aw = ctx.g.add_weight(format!("{node_name}.bn_scale"), wshape.clone(), a);
            let bw = ctx.g.add_weight(format!("{node_name}.bn_shift"), wshape, sh);
            let mid = ctx.g.add_tensor(format!("{node_name}.scaled"), xs.clone(), TensorKind::Value);
            ctx.g.nodes.push(crate::ir::Node {
                name: format!("{node_name}.mul"),
                op: Op::Binary(BinaryOp::Mul),
                inputs: vec![x, aw],
                outputs: vec![mid],
            });
            let out = ctx.value_out(&n.outputs[0], xs);
            ctx.g.nodes.push(crate::ir::Node {
                name: format!("{node_name}.add"),
                op: Op::Binary(BinaryOp::Add),
                inputs: vec![mid, bw],
                outputs: vec![out],
            });
        }
        // --- unary ---
        "Sigmoid" | "Relu" | "Tanh" | "Sqrt" | "Erf" | "Exp" | "Log" | "Neg" | "HardSwish" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let shape = ctx.g.shape(x).to_vec();
            let u = match n.op_type.as_str() {
                "Sigmoid" => UnaryOp::Sigmoid,
                "Relu" => UnaryOp::Relu,
                "Tanh" => UnaryOp::Tanh,
                "Sqrt" => UnaryOp::Sqrt,
                "Erf" => UnaryOp::Erf,
                "Exp" => UnaryOp::Exp,
                "Log" => UnaryOp::Log,
                "Neg" => UnaryOp::Neg,
                _ => UnaryOp::HardSwish, // ONNX HardSwish is fixed alpha=1/6 beta=1/2
            };
            emit(ctx, Op::Unary(u), vec![x], vec![shape], &n.outputs);
        }
        "HardSigmoid" => {
            let alpha = attr_f(n, "alpha", 0.2)?;
            let beta = attr_f(n, "beta", 0.5)?;
            // Only the torch/PP-OCR parameterization is implemented.
            ensure!((alpha - 1.0 / 6.0).abs() < 1e-4 && (beta - 0.5).abs() < 1e-6,
                "HardSigmoid: only alpha=1/6, beta=0.5 supported (got {alpha}, {beta})");
            let x = ctx.tensor(&n.inputs[0])?;
            let shape = ctx.g.shape(x).to_vec();
            emit(ctx, Op::Unary(UnaryOp::HardSigmoid), vec![x], vec![shape], &n.outputs);
        }
        "Gelu" => {
            ensure!(attr_s(n, "approximate", "none")? == "none", "Gelu: tanh approximation not supported");
            let x = ctx.tensor(&n.inputs[0])?;
            let shape = ctx.g.shape(x).to_vec();
            emit(ctx, Op::Unary(UnaryOp::GeluErf), vec![x], vec![shape], &n.outputs);
        }
        // --- binary (broadcasting) ---
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Max" | "Min" => {
            let a = ctx.tensor(&n.inputs[0])?;
            let b = ctx.tensor(&n.inputs[1])?;
            if n.op_type == "Max" || n.op_type == "Min" {
                ensure!(n.inputs.len() == 2, "{}: only 2 inputs supported", n.op_type);
            }
            let shape = broadcast(ctx.g.shape(a), ctx.g.shape(b))?;
            let op = match n.op_type.as_str() {
                "Add" => BinaryOp::Add,
                "Sub" => BinaryOp::Sub,
                "Mul" => BinaryOp::Mul,
                "Div" => BinaryOp::Div,
                "Pow" => BinaryOp::Pow,
                "Max" => BinaryOp::Max,
                _ => BinaryOp::Min,
            };
            emit(ctx, Op::Binary(op), vec![a, b], vec![shape], &n.outputs);
        }
        // --- pooling ---
        "MaxPool" | "AveragePool" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            ensure!(xs.len() == 4, "{}: only 2-D pooling", n.op_type);
            ensure!(n.outputs.len() == 1, "MaxPool: Indices output unsupported");
            let ceil = attr_i(n, "ceil_mode", 0)? != 0;
            let kernel = pair(attr_is(n, "kernel_shape")?, 1, "pool kernel")?;
            let strides = pair(attr_is(n, "strides")?, 1, "pool strides")?;
            let pads = resolve_pads(n, [xs[2], xs[3]], kernel, strides, [1, 1], "pool pads")?;
            if let Some(d) = attr_is(n, "dilations")? {
                ensure!(d.iter().all(|&v| v == 1), "pool dilations unsupported");
            }
            let oh = ojas_cpu::cpu_cnn::pool_out(xs[2], kernel[0], pads[0], pads[2], strides[0], ceil);
            let ow = ojas_cpu::cpu_cnn::pool_out(xs[3], kernel[1], pads[1], pads[3], strides[1], ceil);
            let out_shape = vec![xs[0], xs[1], oh, ow];
            let op = if n.op_type == "MaxPool" {
                Op::MaxPool { kernel, strides, pads, ceil }
            } else {
                Op::AvgPool { kernel, strides, pads, count_include_pad: attr_i(n, "count_include_pad", 0)? != 0, ceil }
            };
            emit(ctx, op, vec![x], vec![out_shape], &n.outputs);
        }
        "GlobalAveragePool" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            ensure!(xs.len() == 4, "GlobalAveragePool: rank 4 only");
            emit(ctx, Op::GlobalAvgPool, vec![x], vec![vec![xs[0], xs[1], 1, 1]], &n.outputs);
        }
        "Resize" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            ensure!(xs.len() == 4, "Resize: rank 4 only");
            let mode = attr_s(n, "mode", "nearest")?;
            let coord = attr_s(n, "coordinate_transformation_mode", "half_pixel")?;
            let nearest = attr_s(n, "nearest_mode", "round_prefer_floor")?;
            ensure!(mode == "nearest", "Resize: mode {mode:?} unsupported (only nearest — trap T4)");
            ensure!(coord == "asymmetric", "Resize: coordinate mode {coord:?} unsupported (only asymmetric)");
            ensure!(nearest == "floor", "Resize: nearest_mode {nearest:?} unsupported (only floor)");
            // scales (input 2) or sizes (input 3), must be static
            let (sh, sw) = if n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                let s = match ctx.sval(&n.inputs[2])? {
                    SVal::F(v) => v.clone(),
                    SVal::I(v) => v.iter().map(|&i| i as f32).collect(),
                };
                ensure!(s.len() == 4 && s[0] == 1.0 && s[1] == 1.0, "Resize: batch/channel scaling unsupported ({s:?})");
                (s[2], s[3])
            } else if n.inputs.len() > 3 && !n.inputs[3].is_empty() {
                let sizes = ctx.ints(&n.inputs[3])?;
                ensure!(sizes.len() == 4, "Resize sizes: {sizes:?}");
                (sizes[2] as f32 / xs[2] as f32, sizes[3] as f32 / xs[3] as f32)
            } else {
                bail!("Resize without scales or sizes")
            };
            ensure!(sh.fract() == 0.0 && sw.fract() == 0.0 && sh >= 1.0 && sw >= 1.0,
                "Resize: non-integer scale ({sh}, {sw}) unsupported");
            let (sh, sw) = (sh as usize, sw as usize);
            let out_shape = vec![xs[0], xs[1], xs[2] * sh, xs[3] * sw];
            emit(ctx, Op::ResizeNearest { scale_h: sh, scale_w: sw }, vec![x], vec![out_shape], &n.outputs);
        }
        // --- tensor plumbing ---
        "Concat" => {
            let ids: Vec<TensorId> = n.inputs.iter().map(|i| ctx.tensor(i)).collect::<Result<_>>()?;
            let shapes: Vec<Vec<usize>> = ids.iter().map(|&i| ctx.g.shape(i).to_vec()).collect();
            let rank = shapes[0].len();
            let axis = norm_axis(attr_i(n, "axis", 0)?, rank)?;
            let mut out_shape = shapes[0].clone();
            out_shape[axis] = shapes.iter().map(|s| s[axis]).sum();
            for s in &shapes {
                ensure!(s.len() == rank, "Concat: rank mismatch");
                for (d, (&a, &b)) in s.iter().zip(&out_shape).enumerate() {
                    ensure!(d == axis || a == b, "Concat: dim {d} mismatch {shapes:?}");
                }
            }
            emit(ctx, Op::Concat { axis }, ids, vec![out_shape], &n.outputs);
        }
        "Split" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            let axis = norm_axis(attr_i(n, "axis", 0)?, xs.len())?;
            let parts: Vec<usize> = if n.inputs.len() > 1 && !n.inputs[1].is_empty() {
                to_usize_shape(&ctx.ints(&n.inputs[1])?, "Split")?
            } else if let Some(v) = attr_is(n, "split")? {
                to_usize_shape(&v, "Split")?
            } else {
                let num = usize::try_from(attr_i(n, "num_outputs", n.outputs.len() as i64)?)?;
                ensure!(num > 0 && xs[axis] % num == 0, "Split: {} not divisible by {num}", xs[axis]);
                vec![xs[axis] / num; num]
            };
            ensure!(parts.iter().sum::<usize>() == xs[axis], "Split: parts {parts:?} vs dim {}", xs[axis]);
            ensure!(parts.len() == n.outputs.len(), "Split: {} parts vs {} outputs", parts.len(), n.outputs.len());
            let out_shapes: Vec<Vec<usize>> = parts
                .iter()
                .map(|&p| {
                    let mut s = xs.clone();
                    s[axis] = p;
                    s
                })
                .collect();
            emit(ctx, Op::Split { axis, parts }, vec![x], out_shapes, &n.outputs);
        }
        "Slice" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            let rank = xs.len();
            let (raw_starts, raw_ends, raw_axes, raw_steps) = if n.inputs.len() > 1 {
                let starts = ctx.ints(&n.inputs[1])?;
                let ends = ctx.ints(&n.inputs[2])?;
                let axes = if n.inputs.len() > 3 && !n.inputs[3].is_empty() {
                    ctx.ints(&n.inputs[3])?
                } else {
                    (0..starts.len() as i64).collect()
                };
                let steps = if n.inputs.len() > 4 && !n.inputs[4].is_empty() {
                    ctx.ints(&n.inputs[4])?
                } else {
                    vec![1; starts.len()]
                };
                (starts, ends, axes, steps)
            } else {
                // opset < 10: attributes
                let starts = attr_is(n, "starts")?.ok_or_else(|| anyhow!("Slice without starts"))?;
                let ends = attr_is(n, "ends")?.ok_or_else(|| anyhow!("Slice without ends"))?;
                let axes = attr_is(n, "axes")?.unwrap_or_else(|| (0..starts.len() as i64).collect());
                let steps = vec![1; starts.len()];
                (starts, ends, axes, steps)
            };
            let mut starts = vec![0usize; rank];
            let mut ends: Vec<usize> = xs.clone();
            let mut steps = vec![1usize; rank];
            for (k, &ax) in raw_axes.iter().enumerate() {
                let a = norm_axis(ax, rank)?;
                let d = xs[a] as i64;
                let clamp = |v: i64| (if v < 0 { v + d } else { v }).clamp(0, d) as usize;
                starts[a] = clamp(raw_starts[k]);
                ends[a] = clamp(raw_ends[k].min(i64::MAX / 2));
                ensure!(raw_steps[k] >= 1, "Slice: negative/zero step {} unsupported", raw_steps[k]);
                steps[a] = raw_steps[k] as usize;
                ensure!(starts[a] <= ends[a], "Slice: start {} > end {} on axis {a}", starts[a], ends[a]);
            }
            let out_shape: Vec<usize> = (0..rank).map(|a| (ends[a] - starts[a]).div_ceil(steps[a])).collect();
            emit(ctx, Op::Slice { starts, ends, steps }, vec![x], vec![out_shape], &n.outputs);
        }
        "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" | "Identity" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            let out_shape = match n.op_type.as_str() {
                "Reshape" => {
                    let target = ctx.ints(&n.inputs[1])?;
                    // allowzero only changes what a literal 0 means; targets
                    // without zeros behave identically under both modes.
                    ensure!(
                        attr_i(n, "allowzero", 0)? == 0 || !target.contains(&0),
                        "Reshape: allowzero with a 0 target dim unsupported"
                    );
                    resolve_reshape(&xs, &target)?
                }
                "Flatten" => {
                    let axis = norm_axis(attr_i(n, "axis", 1)?.min(xs.len() as i64), xs.len() + 1)?;
                    let (a, b) = xs.split_at(axis);
                    vec![a.iter().product::<usize>().max(1), b.iter().product::<usize>().max(1)]
                }
                "Squeeze" => {
                    let axes = if n.inputs.len() > 1 && !n.inputs[1].is_empty() {
                        ctx.ints(&n.inputs[1])?
                    } else {
                        attr_is(n, "axes")?.unwrap_or_default()
                    };
                    if axes.is_empty() {
                        xs.iter().copied().filter(|&d| d != 1).collect()
                    } else {
                        let drop: Vec<usize> = axes.iter().map(|&a| norm_axis(a, xs.len())).collect::<Result<_>>()?;
                        xs.iter()
                            .enumerate()
                            .filter(|(i, _)| !drop.contains(i))
                            .map(|(_, &d)| d)
                            .collect()
                    }
                }
                "Unsqueeze" => {
                    let axes = if n.inputs.len() > 1 && !n.inputs[1].is_empty() {
                        ctx.ints(&n.inputs[1])?
                    } else {
                        attr_is(n, "axes")?.ok_or_else(|| anyhow!("Unsqueeze without axes"))?
                    };
                    let rank = xs.len() + axes.len();
                    let mut at: Vec<usize> = axes.iter().map(|&a| norm_axis(a, rank)).collect::<Result<_>>()?;
                    at.sort_unstable();
                    let mut s = xs.clone();
                    for &a in &at {
                        s.insert(a.min(s.len()), 1);
                    }
                    s
                }
                _ => xs.clone(),
            };
            ctx.view_out(&node_name, &n.outputs[0], x, out_shape);
        }
        "Transpose" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            let perm = match attr_is(n, "perm")? {
                Some(p) => p.iter().map(|&a| norm_axis(a, xs.len())).collect::<Result<Vec<_>>>()?,
                None => (0..xs.len()).rev().collect(),
            };
            ensure!(perm.len() == xs.len(), "Transpose: perm rank mismatch");
            if perm.iter().enumerate().all(|(i, &p)| i == p) {
                ctx.view_out(&node_name, &n.outputs[0], x, xs);
            } else {
                let out_shape: Vec<usize> = perm.iter().map(|&p| xs[p]).collect();
                emit(ctx, Op::Transpose { perm }, vec![x], vec![out_shape], &n.outputs);
            }
        }
        // --- math ---
        "MatMul" => {
            let a = ctx.tensor(&n.inputs[0])?;
            let mut b = ctx.tensor(&n.inputs[1])?;
            let (sa, mut sb) = (ctx.g.shape(a).to_vec(), ctx.g.shape(b).to_vec());
            if sb.len() == 1 && sa.len() >= 2 {
                // numpy: a 1-D right operand is a column [K, 1] whose axis drops from the result
                // (D-FINE's FDR integral: bin probabilities · bin values)
                let w = ctx.g.weight(b).ok_or_else(|| anyhow!("MatMul: a 1-D runtime right operand is unsupported"))?.to_vec();
                b = ctx.g.add_weight(format!("{node_name}/column"), vec![sb[0], 1], w);
                sb = vec![sb[0], 1];
                ensure!(sa[sa.len() - 1] == sb[0], "MatMul: inner dims {:?} vs {:?}", sa, sb);
                let mut col = sa.clone();
                *col.last_mut().unwrap() = 1;
                let tmp = ctx.value_out(&format!("{node_name}/col_out"), col);
                ctx.g.nodes.push(crate::ir::Node { name: node_name.clone(), op: Op::MatMul, inputs: vec![a, b], outputs: vec![tmp] });
                ctx.view_out(&format!("{node_name}/drop"), &n.outputs[0], tmp, sa[..sa.len() - 1].to_vec());
                return Ok(());
            }
            ensure!(sa.len() >= 2 && sb.len() >= 2, "MatMul: 1-D operands unsupported");
            let (m, ka) = (sa[sa.len() - 2], sa[sa.len() - 1]);
            let (kb, nn) = (sb[sb.len() - 2], sb[sb.len() - 1]);
            ensure!(ka == kb, "MatMul: inner dims {ka} vs {kb}");
            let lead = broadcast(&sa[..sa.len() - 2], &sb[..sb.len() - 2])?;
            let mut out_shape = lead;
            out_shape.push(m);
            out_shape.push(nn);
            emit(ctx, Op::MatMul, vec![a, b], vec![out_shape], &n.outputs);
        }
        "Gemm" => {
            ensure!(attr_i(n, "transA", 0)? == 0, "Gemm: transA unsupported");
            ensure!((attr_f(n, "alpha", 1.0)? - 1.0).abs() < 1e-6, "Gemm: alpha != 1 unsupported");
            ensure!((attr_f(n, "beta", 1.0)? - 1.0).abs() < 1e-6, "Gemm: beta != 1 unsupported");
            let trans_b = attr_i(n, "transB", 0)? != 0;
            let a = ctx.tensor(&n.inputs[0])?;
            let b = ctx.tensor(&n.inputs[1])?;
            let (sa, sb) = (ctx.g.shape(a).to_vec(), ctx.g.shape(b).to_vec());
            ensure!(sa.len() == 2 && sb.len() == 2, "Gemm: rank-2 only");
            let (m, k) = (sa[0], sa[1]);
            let (nn, kb) = if trans_b { (sb[0], sb[1]) } else { (sb[1], sb[0]) };
            ensure!(k == kb, "Gemm: inner dims {k} vs {kb}");
            let mut inputs = vec![a, b];
            if n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                let c = ctx.tensor(&n.inputs[2])?;
                let cs = ctx.g.shape(c);
                ensure!(cs == [nn] || cs == [1, nn], "Gemm: bias shape {cs:?} unsupported");
                inputs.push(c);
            }
            emit(ctx, Op::Gemm { trans_b, act: None }, inputs, vec![vec![m, nn]], &n.outputs);
        }
        "Softmax" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            let default_axis = if ctx.opset >= 13 { -1 } else { 1 };
            let axis = norm_axis(attr_i(n, "axis", default_axis)?, xs.len())?;
            if ctx.opset < 13 {
                ensure!(axis == xs.len() - 1, "Softmax: opset<13 flatten semantics only supported on the last axis");
            }
            emit(ctx, Op::Softmax { axis }, vec![x], vec![xs], &n.outputs);
        }
        "ReduceSum" | "ReduceMax" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            ensure!(attr_i(n, "noop_with_empty_axes", 0)? == 0, "{}: noop_with_empty_axes unsupported", n.op_type);
            let keepdims = attr_i(n, "keepdims", 1)? != 0;
            let axes = reduce_axes(ctx, n, xs.len())?;
            let out_shape: Vec<usize> = if keepdims {
                xs.iter().enumerate().map(|(i, &d)| if axes.contains(&i) { 1 } else { d }).collect()
            } else {
                xs.iter().enumerate().filter(|(i, _)| !axes.contains(i)).map(|(_, &d)| d).collect()
            };
            let kind = if n.op_type == "ReduceSum" { ReduceOp::Sum } else { ReduceOp::Max };
            emit(ctx, Op::Reduce { kind, axes, keepdims }, vec![x], vec![out_shape], &n.outputs);
        }
        "GridSample" => {
            ensure!(attr_s(n, "mode", "bilinear")? == "bilinear", "GridSample: only bilinear");
            ensure!(attr_s(n, "padding_mode", "zeros")? == "zeros", "GridSample: only zeros padding");
            ensure!(attr_i(n, "align_corners", 0)? == 0, "GridSample: only align_corners = 0");
            let (x, grid) = (ctx.tensor(&n.inputs[0])?, ctx.tensor(&n.inputs[1])?);
            let (xs, gs) = (ctx.g.shape(x).to_vec(), ctx.g.shape(grid).to_vec());
            ensure!(xs.len() == 4 && gs.len() == 4 && gs[3] == 2 && gs[0] == xs[0], "GridSample: x {xs:?} / grid {gs:?} (4-D, grid (N,H,W,2))");
            emit(ctx, Op::GridSample, vec![x, grid], vec![vec![xs[0], xs[1], gs[1], gs[2]]], &n.outputs);
        }
        "TopK" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            let k = ctx.ints(&n.inputs[1])?;
            ensure!(k.len() == 1 && k[0] > 0, "TopK: k {k:?}");
            let k = k[0] as usize;
            ensure!(norm_axis(attr_i(n, "axis", -1)?, xs.len())? == xs.len() - 1, "TopK: only the last axis");
            ensure!(attr_i(n, "largest", 1)? == 1 && attr_i(n, "sorted", 1)? == 1, "TopK: only largest, sorted");
            ensure!(k <= *xs.last().unwrap(), "TopK: k {k} > axis {}", xs.last().unwrap());
            let mut out = xs.clone();
            *out.last_mut().unwrap() = k;
            // values: computed only if something reads them; indices: a marker for GatherElements
            if ctx.used.contains(n.outputs[0].as_str()) {
                emit(ctx, Op::TopK { k }, vec![x], vec![out.clone()], &n.outputs[..1]);
            }
            if let Some(idx) = n.outputs.get(1).filter(|s| !s.is_empty() && ctx.used.contains(s.as_str())) {
                ensure!(xs.len() == 2, "TopK: indices are gathered only from 2-D scores (B,N), got {xs:?}");
                ctx.env.insert(idx.clone(), Val { shape: out, id: None, sval: None, topk: Some((x, k)) });
            }
        }
        "GatherElements" => {
            let (scores, k) = ctx.val(&n.inputs[1])?.topk.ok_or_else(|| anyhow!("GatherElements: only with TopK indices (DETR query selection)"))?;
            let d = ctx.tensor(&n.inputs[0])?;
            let ds = ctx.g.shape(d).to_vec();
            let ss = ctx.g.shape(scores).to_vec();
            let is = ctx.shape_of(&n.inputs[1])?;
            ensure!(norm_axis(attr_i(n, "axis", 0)?, ds.len())? == 1, "GatherElements: only axis 1");
            ensure!(ds.len() == 3 && ds[..2] == ss[..] && is == [ds[0], k, ds[2]], "GatherElements: data {ds:?}, scores {ss:?}, indices {is:?} — rows (B,N,C) by (B,k)");
            emit(ctx, Op::TopKGather { k }, vec![scores, d], vec![is], &n.outputs);
        }
        "Pad" => {
            ensure!(attr_s(n, "mode", "constant")? == "constant", "Pad: only constant mode");
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            ensure!(xs.len() == 4, "Pad: rank 4 only");
            let p = if n.inputs.len() > 1 && !n.inputs[1].is_empty() { ctx.ints(&n.inputs[1])? } else { attr_is(n, "pads")?.unwrap_or_default() };
            ensure!(p.len() == 8 && p.iter().all(|&v| v >= 0), "Pad: pads {p:?} (8 non-negative)");
            ensure!(p[0] == 0 && p[1] == 0 && p[4] == 0 && p[5] == 0, "Pad: only the spatial axes ({p:?})");
            ensure!(n.inputs.get(3).is_none_or(|s| s.is_empty()), "Pad: axes input unsupported");
            let value = scalar_input(ctx, n, 2)?.unwrap_or(0.0);
            let pads = [p[2] as usize, p[3] as usize, p[6] as usize, p[7] as usize];
            let out = vec![xs[0], xs[1], xs[2] + pads[0] + pads[2], xs[3] + pads[1] + pads[3]];
            emit(ctx, Op::Pad { pads, value }, vec![x], vec![out], &n.outputs);
        }
        "Clip" => {
            // Clip(x, min, max) with constant bounds = Min(Max(x, min), max)
            let mut x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            let (lo, hi) = (scalar_input(ctx, n, 1)?, scalar_input(ctx, n, 2)?);
            let steps: Vec<(BinaryOp, f32)> = [(BinaryOp::Max, lo), (BinaryOp::Min, hi)].into_iter().filter_map(|(o, v)| v.map(|v| (o, v))).collect();
            if steps.is_empty() {
                ctx.view_out(&node_name, &n.outputs[0], x, xs);
            } else {
                for (i, (op, v)) in steps.iter().enumerate() {
                    let bound = ctx.g.add_weight(format!("{node_name}/bound{i}"), vec![1], vec![*v]);
                    let name = if i + 1 == steps.len() { n.outputs[0].clone() } else { format!("{node_name}/clip{i}") };
                    let y = ctx.value_out(&name, xs.clone());
                    ctx.g.nodes.push(crate::ir::Node { name: format!("{node_name}/{i}"), op: Op::Binary(*op), inputs: vec![x, bound], outputs: vec![y] });
                    x = y;
                }
            }
        }
        "Gather" => {
            // runtime data, one constant index: a slice (and a squeeze for a scalar index)
            let d = ctx.tensor(&n.inputs[0])?;
            let ds = ctx.g.shape(d).to_vec();
            let axis = norm_axis(attr_i(n, "axis", 0)?, ds.len())?;
            let idx = ctx.ints(&n.inputs[1])?;
            let ishape = ctx.shape_of(&n.inputs[1])?;
            ensure!(idx.len() == 1, "Gather: runtime data with {} indices unsupported (one constant index only)", idx.len());
            let i = if idx[0] < 0 { idx[0] + ds[axis] as i64 } else { idx[0] } as usize;
            ensure!(i < ds[axis], "Gather: index {} out of range {}", idx[0], ds[axis]);
            let (mut starts, mut ends) = (vec![0; ds.len()], ds.clone());
            starts[axis] = i;
            ends[axis] = i + 1;
            let mut sliced = ds.clone();
            sliced[axis] = 1;
            let tmp = ctx.value_out(&format!("{node_name}/slice"), sliced.clone());
            ctx.g.nodes.push(crate::ir::Node { name: node_name.clone(), op: Op::Slice { starts, ends, steps: vec![1; ds.len()] }, inputs: vec![d], outputs: vec![tmp] });
            let mut out = sliced;
            if ishape.is_empty() {
                out.remove(axis);
            }
            ctx.view_out(&format!("{node_name}/view"), &n.outputs[0], tmp, out);
        }
        "ReduceMean" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            let keepdims = attr_i(n, "keepdims", 1)? != 0;
            let axes_raw = if n.inputs.len() > 1 && !n.inputs[1].is_empty() {
                ctx.ints(&n.inputs[1])?
            } else {
                attr_is(n, "axes")?.unwrap_or_else(|| (0..xs.len() as i64).collect())
            };
            let mut axes: Vec<usize> = axes_raw.iter().map(|&a| norm_axis(a, xs.len())).collect::<Result<_>>()?;
            axes.sort_unstable();
            axes.dedup();
            let out_shape: Vec<usize> = if keepdims {
                xs.iter().enumerate().map(|(i, &d)| if axes.contains(&i) { 1 } else { d }).collect()
            } else {
                xs.iter().enumerate().filter(|(i, _)| !axes.contains(i)).map(|(_, &d)| d).collect()
            };
            emit(ctx, Op::ReduceMean { axes, keepdims }, vec![x], vec![out_shape], &n.outputs);
        }
        "Tile" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            let reps = to_usize_shape(&ctx.ints(&n.inputs[1])?, "Tile")?;
            ensure!(reps.len() == xs.len(), "Tile: repeats {reps:?} vs rank {}", xs.len());
            if reps.iter().all(|&r| r == 1) {
                ctx.view_out(&node_name, &n.outputs[0], x, xs);
            } else if let Some(w) = ctx.g.weight(x).map(|w| w.to_vec()) {
                // a constant tiled by a constant (a pooling probe repeated over the batch): materialise it
                let out_shape: Vec<usize> = xs.iter().zip(&reps).map(|(&d, &r)| d * r).collect();
                let data = tile_data(&w, &xs, &out_shape);
                let id = ctx.g.add_weight(&n.outputs[0], out_shape.clone(), data);
                ctx.bind_output(&n.outputs[0], id, out_shape);
            } else {
                bail!("Tile: runtime repeats {reps:?} of a non-constant input unsupported");
            }
        }
        "InstanceNormalization" => {
            // per (image, channel) mean/variance over the spatial axes, then scale·x̂ + bias,
            // expressed with the reduce / broadcast ops every backend has
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            ensure!(xs.len() >= 3, "InstanceNormalization: rank < 3");
            let eps = attr_f(n, "epsilon", 1e-5)?;
            let c = xs[1];
            let sid = ctx.tensor(&n.inputs[1])?;
            let scale = ctx.g.weight(sid).ok_or_else(|| anyhow!("InstanceNormalization: scale must be constant"))?.to_vec();
            let bid = ctx.tensor(&n.inputs[2])?;
            let bias = ctx.g.weight(bid).ok_or_else(|| anyhow!("InstanceNormalization: bias must be constant"))?.to_vec();
            ensure!(scale.len() == c && bias.len() == c, "InstanceNormalization: channel mismatch");
            let axes: Vec<usize> = (2..xs.len()).collect();
            let mut red = xs.clone();
            for &a in &axes {
                red[a] = 1;
            }
            let mut wshape = vec![1usize; xs.len()];
            wshape[1] = c;
            let sw = ctx.g.add_weight(format!("{node_name}.scale"), wshape.clone(), scale);
            let bw = ctx.g.add_weight(format!("{node_name}.bias"), wshape, bias);
            let ew = ctx.g.add_weight(format!("{node_name}.eps"), vec![1usize; xs.len()], vec![eps]);
            let push = |ctx: &mut Ctx, what: &str, op: Op, inputs: Vec<TensorId>, shape: Vec<usize>| -> TensorId {
                let id = ctx.g.add_tensor(format!("{node_name}.{what}"), shape, TensorKind::Value);
                ctx.g.nodes.push(crate::ir::Node { name: format!("{node_name}.{what}"), op, inputs, outputs: vec![id] });
                id
            };
            let m = push(ctx, "mean", Op::ReduceMean { axes: axes.clone(), keepdims: true }, vec![x], red.clone());
            let d = push(ctx, "centred", Op::Binary(BinaryOp::Sub), vec![x, m], xs.clone());
            let sq = push(ctx, "sq", Op::Binary(BinaryOp::Mul), vec![d, d], xs.clone());
            let v = push(ctx, "var", Op::ReduceMean { axes, keepdims: true }, vec![sq], red.clone());
            let ve = push(ctx, "var_eps", Op::Binary(BinaryOp::Add), vec![v, ew], red.clone());
            let sd = push(ctx, "std", Op::Unary(UnaryOp::Sqrt), vec![ve], red);
            let nrm = push(ctx, "norm", Op::Binary(BinaryOp::Div), vec![d, sd], xs.clone());
            let sc = push(ctx, "scaled", Op::Binary(BinaryOp::Mul), vec![nrm, sw], xs.clone());
            emit(ctx, Op::Binary(BinaryOp::Add), vec![sc, bw], vec![xs], &n.outputs);
        }
        "LayerNormalization" => {
            let x = ctx.tensor(&n.inputs[0])?;
            let xs = ctx.g.shape(x).to_vec();
            let axis = norm_axis(attr_i(n, "axis", -1)?, xs.len())?;
            ensure!(axis == xs.len() - 1, "LayerNormalization: only last-axis supported");
            ensure!(n.outputs.len() == 1, "LayerNormalization: mean/invstd outputs unsupported");
            let eps = attr_f(n, "epsilon", 1e-5)?;
            let w = ctx.tensor(&n.inputs[1])?;
            let mut inputs = vec![x, w];
            if n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                inputs.push(ctx.tensor(&n.inputs[2])?);
            }
            emit(ctx, Op::LayerNorm { eps }, inputs, vec![xs], &n.outputs);
        }
        other => bail!("unsupported ONNX op {other} — extend ojas-vision or re-export the model"),
    }
    Ok(())
}

/// `x` (shape `xs`) repeated to `out` (each dim a multiple): output index i reads x at i mod xs.
fn tile_data(x: &[f32], xs: &[usize], out: &[usize]) -> Vec<f32> {
    let n: usize = out.iter().product();
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let (mut rem, mut src, mut stride) = (i, 0usize, 1usize);
        for d in (0..out.len()).rev() {
            let o = rem % out[d];
            rem /= out[d];
            src += (o % xs[d]) * stride;
            stride *= xs[d];
        }
        y.push(x[src]);
    }
    y
}
