//! GPU graph executor (fp16), for any backend implementing [`GpuDev`] (CUDA,
//! Vulkan). Mirrors `exec_cpu.rs`: walks the optimized IR
//! in order, but plans once — layouts, buffers, conv weight packing, launch
//! parameters — and then replays a flat list of kernel launches on one stream
//! with a single sync per forward.
//!
//! Storage: 4-D activations between convolutions live as padded NHWC images
//! (`ojas_cuda::conv::Storage`, a 1-pixel zero border plus one spare column),
//! which the tensor-core conv reads in place; channel concat/split and
//! same-shape adds work on those buffers directly (the border stays zero).
//! Everything else (attention internals, the detection head) is contiguous
//! row-major. Conversions between the two are inserted where a tensor crosses
//! and cached per tensor.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{bail, ensure, Context, Result};
use ojas_core::conv::{ConvGeom, Storage};


use crate::gpu::{GpuConv, GpuDev};

use crate::ir::{broadcast_strides, strides_of, BinaryOp, ReduceOp, Graph, Node, Op, TensorId, UnaryOp};

/// Border and spare columns of every spatial activation.
const PAD: usize = 1;
const SPARE: usize = 1;

#[derive(Clone, Copy, Debug)]
enum Layout {
    Spatial(Storage),
    Flat,
}

/// One planned tensor: a buffer, the channel offset of its first channel
/// (spatial channel slices of a wider buffer), and its layout.
#[derive(Clone, Copy, Debug)]
struct Placed {
    buf: usize,
    off: usize,
    layout: Layout,
}

impl Placed {
    fn at(&self) -> (usize, u64) {
        (self.buf, self.off as u64 * 2)
    }
}

enum Step<G: GpuDev> {
    Conv { plan: G::Conv, x: (usize, u64), y: (usize, u64), n: usize },
    Launch { name: &'static str, bufs: Vec<(usize, u64)>, consts: Vec<u32>, grid: [u32; 3] },
}

/// Device memory shared by executors that never run at the same time — the
/// stages and batch buckets of one pipeline, where every call syncs before
/// the next. Sized to its largest member and allocated on first use, so all
/// members must be planned before any of them runs. Members re-zero the pad
/// cells of their spatial buffers when another member ran in between.
pub struct ArenaOf<G: GpuDev> {
    need: AtomicUsize,
    buf: OnceLock<G::Buf>,
    owner: AtomicU64,
}

#[cfg(feature = "cuda")]
pub type Arena = ArenaOf<ojas_cuda::CudaGpu>;

impl<G: GpuDev> ArenaOf<G> {
    pub fn new() -> Arc<Self> {
        Arc::new(ArenaOf { need: AtomicUsize::new(0), buf: OnceLock::new(), owner: AtomicU64::new(0) })
    }

    /// Bytes the arena holds (or will, once allocated).
    pub fn bytes(&self) -> usize {
        self.buf.get().map_or(self.need.load(Ordering::Relaxed), |b| G::buf_len(b))
    }

    fn reserve(&self, bytes: usize) -> Result<()> {
        let prev = self.need.fetch_max(bytes, Ordering::Relaxed);
        if let Some(b) = self.buf.get() {
            ensure!(G::buf_len(b) >= bytes.max(prev), "arena already allocated ({} B); plan every member before running any",
                    G::buf_len(b));
        }
        Ok(())
    }

    fn get(&self, g: &G) -> Result<&G::Buf> {
        if self.buf.get().is_none() {
            let b = g.alloc_bytes(self.need.load(Ordering::Relaxed))?;
            let _ = self.buf.set(b);
        }
        Ok(self.buf.get().unwrap())
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub struct GpuExecutor<G: GpuDev> {
    gpu: G,
    bufs: Vec<G::Buf>,
    /// per buffer: byte offset into the arena (None: its own allocation in `bufs`)
    arena_off: Vec<Option<u64>>,
    arena: Option<Arc<ArenaOf<G>>>,
    /// arena buffers whose pad cells must be re-zeroed after another member ran
    /// pad-zeroing table for the arena buffers: (table, buffers, blocks per
    /// buffer: one warp per contiguous border run)
    pad_tab: Option<(G::Buf, u32, u32)>,
    arena_need: usize,
    id: u64,
    tuned: bool,
    steps: Vec<Step<G>>,
    /// flat f16 staging buffer for the (single) graph input, NCHW
    input_stage: usize,
    input_numel: usize,
    /// the padded-NHWC input the graph actually reads (step 0 fills it from
    /// the staging buffer; device-side preprocessing writes it directly)
    input_buf: usize,
    input_storage: Storage,
    batch: usize,
    outputs: Vec<(usize, usize)>, // (buffer, numel)
    output_shapes: Vec<Vec<usize>>,
    /// concat inputs that still needed a copy (placement conflicts)
    pub concat_copies: usize,
    /// when true, `forward` syncs after every step and records per-kind time
    pub profile: bool,
    pub op_times: HashMap<&'static str, (f64, usize)>,
    /// per step: the IR op kind it came from
    origin: Vec<String>,
    /// the forward recorded once (CUDA Graphs), per first step, replayed with
    /// one launch; `OJAS_CUDA_GRAPHS=0` turns it off
    graphs: HashMap<usize, G::Graph>,
}

/// The cnn_act_f code of a unary op.
fn unary_code(u: UnaryOp) -> Result<usize> {
    Ok(match u {
        UnaryOp::Silu => 1,
        UnaryOp::Sigmoid => 2,
        UnaryOp::Relu => 3,
        UnaryOp::Tanh => 4,
        UnaryOp::Sqrt => 5,
        UnaryOp::Erf => 6,
        UnaryOp::GeluErf => 7,
        UnaryOp::HardSigmoid => 8,
        UnaryOp::HardSwish => 9,
        UnaryOp::Neg => 10,
        UnaryOp::Exp => 11,
        UnaryOp::Log => 12,
        UnaryOp::GeluTanh => 14,
    })
}

fn grid1(n: usize) -> [u32; 3] {
    [(n as u32).div_ceil(256), 1, 1]
}

/// Zero-copy concat placement. Walks the graph backwards (outer concats
/// first) and assigns each channel-concat input the slot it will occupy in
/// the concat's buffer, so its producer writes there directly. A split whose
/// parts appear in the concat consecutively and in order is placed as a
/// whole; its parts then land in their slots as views. Returns
/// tensor -> (root tensor that owns the buffer, channel offset).
fn plan_placement(g: &Graph) -> HashMap<TensorId, (TensorId, usize)> {
    let mut producer: HashMap<TensorId, usize> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for &o in &n.outputs {
            producer.insert(o, i);
        }
    }
    let spatial_4d = |t: TensorId| g.shape(t).len() == 4 && !g.is_weight(t) && !g.inputs.contains(&t);
    let writes_storage = |t: TensorId| -> bool {
        match producer.get(&t).map(|&i| &g.nodes[i].op) {
            Some(Op::Conv { .. } | Op::MaxPool { .. } | Op::AvgPool { .. } | Op::ResizeNearest { .. }) => true,
            Some(Op::Concat { axis }) => *axis == 1,
            Some(Op::Binary(BinaryOp::Add)) => {
                let n = &g.nodes[producer[&t]];
                n.inputs.iter().all(|&i| g.shape(i) == g.shape(t) && !g.is_weight(i))
            }
            _ => false,
        }
    };
    let mut place: HashMap<TensorId, (TensorId, usize)> = HashMap::new();
    for node in g.nodes.iter().rev() {
        let Op::Concat { axis: 1 } = node.op else { continue };
        let y = node.outputs[0];
        if !spatial_4d(y) {
            continue;
        }
        let (root, base) = place.get(&y).copied().unwrap_or((y, 0));
        let mut run = 0;
        let mut k = 0;
        while k < node.inputs.len() {
            let x = node.inputs[k];
            let c = g.shape(x)[1];
            // a split feeding its parts in order: place the split input whole
            if let Some(&pi) = producer.get(&x) {
                if let Op::Split { axis: 1, parts } = &g.nodes[pi].op {
                    let s = g.nodes[pi].inputs[0];
                    let outs = &g.nodes[pi].outputs;
                    let first = outs.iter().position(|&o| o == x).unwrap();
                    let fits = first == 0
                        && outs.len() <= node.inputs.len() - k
                        && outs.iter().enumerate().all(|(j, &o)| node.inputs[k + j] == o);
                    if fits && spatial_4d(s) && !place.contains_key(&s) && writes_storage(s) {
                        place.insert(s, (root, base + run));
                        run += parts.iter().sum::<usize>();
                        k += outs.len();
                        continue;
                    }
                }
            }
            if spatial_4d(x) && !place.contains_key(&x) && writes_storage(x) {
                place.insert(x, (root, base + run));
            }
            run += c;
            k += 1;
        }
    }
    place
}

/// What a planned buffer holds, for liveness-based reuse: spatial buffers
/// are shared only between identical storage geometries (their zero borders
/// are never written, so they stay valid), flat ones by size.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    Persistent,
    Flat,
    Spatial { hp: usize, wp: usize, cs: usize, n: usize, pad: usize, h: usize, w: usize, c: usize },
}

struct Planner<'a, G: GpuDev> {
    g: &'a Graph,
    gpu: &'a G,
    sizes: Vec<usize>, // buffer element counts (f16)
    kinds: Vec<Kind>,
    steps: Vec<Step<G>>,
    placed: HashMap<TensorId, Placed>,
    place: HashMap<TensorId, (TensorId, usize)>,
    /// root tensor -> its buffer (allocated on first use)
    roots: HashMap<TensorId, usize>,
    /// cached layout conversions: (tensor, want_flat) -> placement
    converted: HashMap<(TensorId, bool), Placed>,
    weights: HashMap<TensorId, usize>,
    uploads: Vec<(usize, Vec<f32>)>, // buffer, contents (f16 on upload)
    concat_copies: usize,
    /// nodes already done inside an earlier step (a conv's fused epilogue)
    fused: std::collections::HashSet<usize>,
    /// zero border of every spatial buffer: the largest conv pad in the graph (≥ 1),
    /// so a 7×7 head reads the same padded layout as the 3×3 stem
    pad: usize,
}

impl<'a, G: GpuDev> Planner<'a, G> {
    fn alloc(&mut self, numel: usize) -> usize {
        self.alloc_kind(numel, Kind::Flat)
    }

    fn alloc_kind(&mut self, numel: usize, kind: Kind) -> usize {
        self.sizes.push(numel.max(1));
        self.kinds.push(kind);
        self.sizes.len() - 1
    }

    fn alloc_spatial(&mut self, st: &Storage, n: usize) -> usize {
        self.alloc_kind(n * st.img(), Kind::Spatial { hp: st.hp(), wp: st.wp(), cs: st.cs, n, pad: st.pad, h: st.h, w: st.w, c: st.c })
    }

    fn launch(&mut self, name: &'static str, bufs: Vec<(usize, u64)>, consts: Vec<usize>, n: usize) {
        self.steps.push(Step::Launch { name, bufs, consts: consts.into_iter().map(|c| c as u32).collect(), grid: grid1(n) });
    }

    fn batch(&self, t: TensorId) -> usize {
        self.g.shape(t)[0]
    }

    fn dense_storage(&self, t: TensorId) -> Storage {
        let s = self.g.shape(t);
        Storage { c: s[1], cs: s[1], h: s[2], w: s[3], pad: self.pad, spare: SPARE }
    }

    /// Weight tensor as a flat f16 buffer (uploaded once).
    fn weight(&mut self, t: TensorId) -> usize {
        if let Some(&b) = self.weights.get(&t) {
            return b;
        }
        let data = self.g.weight(t).expect("weight").to_vec();
        let b = self.alloc_kind(data.len(), Kind::Persistent);
        self.uploads.push((b, data));
        self.weights.insert(t, b);
        b
    }

    fn get(&self, t: TensorId) -> Result<Placed> {
        self.placed.get(&t).copied().with_context(|| format!("tensor {} not produced", self.g.tensors[t].name))
    }

    /// `t` contiguous (row-major, logical shape): (buffer, byte offset 0).
    fn flat(&mut self, t: TensorId) -> Result<(usize, u64)> {
        if self.g.is_weight(t) {
            return Ok((self.weight(t), 0));
        }
        let p = self.get(t)?;
        match p.layout {
            Layout::Flat => Ok(p.at()),
            Layout::Spatial(st) => {
                if let Some(c) = self.converted.get(&(t, true)) {
                    return Ok(c.at());
                }
                let numel = self.batch(t) * st.c * st.h * st.w;
                let b = self.alloc(numel);
                self.launch("cnn_nhwc_to_nchw_f16", vec![p.at(), (b, 0)],
                            vec![numel, st.c, st.h, st.w, st.pad, st.wp(), st.img(), st.cs], numel);
                let c = Placed { buf: b, off: 0, layout: Layout::Flat };
                self.converted.insert((t, true), c);
                Ok(c.at())
            }
        }
    }

    /// `t` as a padded NHWC image (converting a contiguous 4-D tensor if needed).
    fn spatial(&mut self, t: TensorId) -> Result<(Placed, Storage)> {
        let p = self.get(t)?;
        match p.layout {
            Layout::Spatial(st) => Ok((p, st)),
            Layout::Flat => {
                ensure!(self.g.shape(t).len() == 4, "tensor {} is not 4-D", self.g.tensors[t].name);
                if let Some(c) = self.converted.get(&(t, false)).copied() {
                    let Layout::Spatial(st) = c.layout else { unreachable!() };
                    return Ok((c, st));
                }
                let st = self.dense_storage(t);
                let b = self.alloc_spatial(&st, self.batch(t));
                let numel = self.batch(t) * st.c * st.h * st.w;
                self.launch("cnn_nchw_to_nhwc_f16", vec![p.at(), (b, 0)],
                            vec![numel, st.c, st.h, st.w, st.pad, st.wp(), st.img(), st.cs], numel);
                let c = Placed { buf: b, off: 0, layout: Layout::Spatial(st) };
                self.converted.insert((t, false), c);
                Ok((c, st))
            }
        }
    }

    /// Spatial `t` with its channels packed per pixel (copies a channel slice
    /// into its own buffer when the consumer cannot read strided channels).
    fn spatial_dense(&mut self, t: TensorId) -> Result<(Placed, Storage)> {
        let (p, st) = self.spatial(t)?;
        if st.cs == st.c {
            return Ok((p, st));
        }
        let d = self.dense_storage(t);
        let b = self.alloc_spatial(&d, self.batch(t));
        let n = self.batch(t) * st.hp() * st.wp() * st.c;
        self.launch("cnn_axis_copy_f16", vec![(p.buf, 0), (b, 0)], vec![n, st.c, 1, st.cs, p.off, st.c, 0], n);
        Ok((Placed { buf: b, off: 0, layout: Layout::Spatial(d) }, d))
    }

    fn is_spatial(&self, t: TensorId) -> bool {
        matches!(self.placed.get(&t), Some(Placed { layout: Layout::Spatial(_), .. }))
    }

    fn root_buf(&mut self, root: TensorId) -> usize {
        if let Some(&b) = self.roots.get(&root) {
            return b;
        }
        let st = self.dense_storage(root);
        let b = self.alloc_spatial(&st, self.batch(root));
        self.roots.insert(root, b);
        b
    }

    /// Output storage for spatial `t`: its concat slot if placed, else its own.
    fn new_spatial(&mut self, t: TensorId) -> (Placed, Storage) {
        let (root, off) = self.place.get(&t).copied().unwrap_or((t, 0));
        let buf = self.root_buf(root);
        let mut st = self.dense_storage(t);
        st.cs = self.g.shape(root)[1];
        let p = Placed { buf, off, layout: Layout::Spatial(st) };
        self.placed.insert(t, p);
        (p, st)
    }

    fn new_flat(&mut self, t: TensorId) -> (usize, u64) {
        let b = self.alloc(self.g.tensors[t].numel());
        self.placed.insert(t, Placed { buf: b, off: 0, layout: Layout::Flat });
        (b, 0)
    }

    /// The one node that reads `y` (not a graph output), for fusing it into
    /// the step that writes `y`.
    fn sole_reader(&self, step: usize, y: TensorId) -> Option<(usize, &'a Node)> {
        let g = self.g;
        if g.outputs.contains(&y) {
            return None;
        }
        let mut users = g.nodes.iter().enumerate().skip(step + 1).filter(|(_, n)| n.inputs.contains(&y));
        let first = users.next()?;
        users.next().is_none().then_some(first)
    }

    /// The positive-scale ScaleShift that is the only reader of `y` (HGNetv2's
    /// LearnableAffineBlock after Conv+ReLU): (node, scale, shift, its output).
    fn affine_after(&self, step: usize, y: TensorId) -> Option<(usize, f32, f32, TensorId)> {
        match self.sole_reader(step, y)? {
            (i, Node { op: Op::ScaleShift { scale, shift }, outputs, .. }) if *scale > 0.0 => Some((i, *scale, *shift, outputs[0])),
            _ => None,
        }
    }

    fn node(&mut self, step: usize) -> Result<()> {
        let g = self.g;
        let node = &g.nodes[step];
        match &node.op {
            Op::Conv { group, strides, pads, dilations, act } => {
                let (x, w, y) = (node.inputs[0], node.inputs[1], node.outputs[0]);
                let (xs, ws) = (g.shape(x).to_vec(), g.shape(w).to_vec());
                ensure!(dilations == &[1, 1], "conv: dilation unsupported");
                let act_code: u32 = match act {
                    None => 0,
                    Some(UnaryOp::Silu) => 1,
                    Some(UnaryOp::Sigmoid) => 2,
                    Some(UnaryOp::Relu) => 3,
                    Some(UnaryOp::HardSigmoid) => 8,
                    Some(UnaryOp::HardSwish) => 9,
                    Some(u) => bail!("conv: fused activation {u:?} not on CUDA yet"),
                };
                let bias = node.inputs.get(2).map(|&b| g.weight(b).expect("bias weight").to_vec());
                let n = xs[0];
                let (kh, kw) = (ws[2], ws[3]);
                let (sh, sw) = (strides[0], strides[1]);
                let depthwise = *group == xs[1] && ws[0] == xs[1] && ws[1] == 1 && *group > 1;
                if !depthwise {
                    ensure!(pads.iter().all(|&p| p <= self.pad), "conv: pads {pads:?} exceed the storage border {}", self.pad);
                }
                if !depthwise && *group == 1 && kh == 1 && kw == 1 && xs[2] == 1 && xs[3] == 1 && xs[1] % 16 != 0 {
                    // a 1×1 conv on a 1×1 map is one GEMV per image (a channel gate's FC); the
                    // tensor-core tiles need kw·cin % 16 == 0, so it runs on the plain matmul
                    // kernel instead: y[n, cout] = x[n, cin] · Wᵀ, then bias and activation
                    let (cin, cout) = (xs[1], ws[0]);
                    let wt = g.weight(w).unwrap();
                    let mut wtr = vec![0.0f32; cin * cout];
                    for oc in 0..cout {
                        for ic in 0..cin {
                            wtr[ic * cout + oc] = wt[oc * cin + ic];
                        }
                    }
                    let wb = self.alloc_kind(wtr.len(), Kind::Persistent);
                    self.uploads.push((wb, wtr));
                    let xb = self.flat(x)?;
                    let yb = self.new_flat(y);
                    let total = n * cout;
                    self.launch("cnn_matmul_f16", vec![xb, (wb, 0), yb], vec![total, n, cin, cout, 0], total);
                    if bias.is_some() || act_code != 0 {
                        let b = bias.clone().unwrap_or_else(|| vec![0.0; cout]);
                        let bb = self.alloc_kind(b.len(), Kind::Persistent);
                        self.uploads.push((bb, b));
                        self.launch("cnn_bias_act_f16", vec![yb, (bb, 0)], vec![total, 1, cout, act_code as usize], total);
                    }
                    return Ok(());
                }
                if *group == 1 {
                    let cin = xs[1];
                    let (xp, st) = if cin % 16 == 0 { self.spatial(x)? } else { self.spatial_dense(x)? };
                    let mut wt = g.weight(w).unwrap().to_vec();
                    // Conv+ReLU then s·x+t (s > 0) in one conv: s·relu(z)+t = max(s·z + t, t),
                    // s folded into the weights and bias, t after the bias as the floor
                    let (mut y, mut act_code, mut bias) = (y, act_code, bias);
                    if act_code == 3 && G::CONV_FLOOR {
                        if let Some((i, sc, t, z)) = self.affine_after(step, y) {
                            let cout = ws[0];
                            wt.iter_mut().for_each(|v| *v *= sc);
                            let mut b: Vec<f32> = bias.unwrap_or_else(|| vec![0.0; cout]).iter().map(|v| v * sc + t).collect();
                            b.extend(std::iter::repeat_n(t, cout));
                            (y, act_code, bias) = (z, 13, Some(b));
                            self.fused.insert(i);
                        }
                    }
                    let (yp, ost) = self.new_spatial(y);
                    // channel storage may exceed the logical channels (RGB input)
                    if st.c != cin {
                        let cout = ws[0];
                        let kk = kh * kw;
                        let mut wp = vec![0.0f32; cout * st.c * kk];
                        for oc in 0..cout {
                            for ic in 0..cin {
                                let (d, sidx) = ((oc * st.c + ic) * kk, (oc * cin + ic) * kk);
                                wp[d..d + kk].copy_from_slice(&wt[sidx..sidx + kk]);
                            }
                        }
                        wt = wp;
                    }
                    let geom = ConvGeom { cin: st.c, h: xs[2], w: xs[3], cout: ws[0], kh, kw, sh, sw, pads: *pads };
                    let plan = self.gpu.conv_plan(geom, &wt, bias.as_deref(), act_code, st, ost)?;
                    self.steps.push(Step::Conv { plan, x: xp.at(), y: yp.at(), n });
                } else if !depthwise {
                    // general grouped conv: one conv per group on channel slices
                    let (cin_g, cout_g) = (xs[1] / group, ws[0] / group);
                    let (xp, st) = self.spatial(x)?;
                    let (yp, ost) = self.new_spatial(y);
                    let wt = g.weight(w).unwrap();
                    let per = cout_g * cin_g * kh * kw;
                    for gi in 0..*group {
                        let mut ist = st;
                        ist.c = cin_g;
                        let mut xg = Placed { buf: xp.buf, off: xp.off + gi * cin_g, layout: Layout::Spatial(ist) };
                        if cin_g % 16 != 0 {
                            // pack the slice so phantom columns can read contiguous rows
                            let d = Storage { c: cin_g, cs: cin_g, ..st };
                            let b = self.alloc_spatial(&d, n);
                            let cnt = n * st.hp() * st.wp() * cin_g;
                            self.launch("cnn_axis_copy_f16", vec![(xp.buf, 0), (b, 0)], vec![cnt, cin_g, 1, st.cs, xg.off, cin_g, 0], cnt);
                            xg = Placed { buf: b, off: 0, layout: Layout::Spatial(d) };
                            ist = d;
                        }
                        let mut og = ost;
                        og.c = cout_g;
                        let geom = ConvGeom { cin: cin_g, h: xs[2], w: xs[3], cout: cout_g, kh, kw, sh, sw, pads: *pads };
                        let bg = bias.as_ref().map(|b| b[gi * cout_g..(gi + 1) * cout_g].to_vec());
                        let plan = self.gpu.conv_plan(geom, &wt[gi * per..(gi + 1) * per], bg.as_deref(), act_code, ist, og)?;
                        self.steps.push(Step::Conv { plan, x: xg.at(), y: (yp.buf, (yp.off + gi * cout_g) as u64 * 2), n });
                    }
                } else {
                    ensure!(kh == kw && kh * kh <= 25, "depthwise: square kernels up to 5x5");
                    let (xp, st) = self.spatial(x)?;
                    let (yp, ost) = self.new_spatial(y);
                    let wb = self.weight(w);
                    let (oh, ow) = (ost.h, ost.w);
                    let c = xs[1];
                    let dw2 = (kh == 3 || kh == 5) && c % 2 == 0 && st.cs % 2 == 0 && ost.cs % 2 == 0
                        && xp.off % 2 == 0 && yp.off % 2 == 0 && std::env::var("OJAS_DW2").map_or(true, |v| v != "0")
                        && self.gpu.has_kernel("cnn_dwconv2_nhwc_f16");
                    if dw2 {
                        let bb = match node.inputs.get(2) {
                            Some(&b) => (self.weight(b), 0),
                            None => (wb, 0),
                        };
                        self.steps.push(Step::Launch {
                            name: "cnn_dwconv2_nhwc_f16",
                            bufs: vec![xp.at(), (wb, 0), yp.at(), bb],
                            consts: [c, st.h, st.w, st.pad, st.wp(), st.img(), oh, ow, kh, sh, sw, pads[0], pads[1],
                                     ost.pad, ost.wp(), ost.img(), st.cs, ost.cs, act_code as usize, bias.is_some() as usize]
                                .map(|v| v as u32)
                                .to_vec(),
                            grid: [(n * oh) as u32, c.div_ceil(64) as u32, 1],
                        });
                        return Ok(());
                    }
                    self.steps.push(Step::Launch {
                        name: "cnn_dwconv_nhwc_f16",
                        bufs: vec![xp.at(), (wb, 0), yp.at()],
                        consts: [n, c, st.h, st.w, st.pad, st.wp(), st.img(), oh, ow, kh, sh, sw, pads[0], pads[1],
                                 ost.pad, ost.wp(), ost.img(), st.cs, ost.cs]
                            .map(|v| v as u32)
                            .to_vec(),
                        grid: [(n * oh) as u32, (ow * c).div_ceil(256) as u32, 1],
                    });
                    if bias.is_some() || act_code != 0 {
                        let bb = match node.inputs.get(2) {
                            Some(&b) => (self.weight(b), 0),
                            None => yp.at(),
                        };
                        let total = n * oh * ow * c;
                        self.launch("cnn_bias_act_nhwc_f16", vec![yp.at(), bb],
                                    vec![total, oh, ow, ost.pad, ost.wp(), c, ost.img(), act_code as usize,
                                         bias.is_some() as usize, ost.cs], total);
                    }
                }
            }
            Op::MaxPool { kernel, strides, pads, .. } | Op::AvgPool { kernel, strides, pads, .. } => {
                // ceil mode only changes the output extent (already in the IR
                // shape); windows are bounds-checked against the logical image
                let (x, y) = (node.inputs[0], node.outputs[0]);
                let (xp, st) = self.spatial(x)?;
                let (yp, ost) = self.new_spatial(y);
                let total = self.batch(y) * ost.c * ost.h * ost.w;
                let mut consts = vec![total, st.c, st.h, st.w, st.pad, st.wp(), st.img(), ost.h, ost.w, ost.pad, ost.wp(), ost.img(),
                                      kernel[0], kernel[1], strides[0], strides[1], pads[0], pads[1], st.cs, ost.cs];
                let name = match &node.op {
                    Op::AvgPool { count_include_pad, .. } => {
                        consts.push(*count_include_pad as usize);
                        "cnn_avgpool_nhwc_f16"
                    }
                    _ => "cnn_maxpool_nhwc_f16",
                };
                self.launch(name, vec![xp.at(), yp.at()], consts, total);
            }
            Op::ResizeNearest { scale_h, scale_w } => {
                let (x, y) = (node.inputs[0], node.outputs[0]);
                let (xp, st) = self.spatial(x)?;
                let (yp, ost) = self.new_spatial(y);
                let total = self.batch(y) * ost.c * ost.h * ost.w;
                self.launch("cnn_upsample_nhwc_f16", vec![xp.at(), yp.at()],
                            vec![total, st.c, st.h, st.w, st.pad, st.wp(), st.img(), ost.h, ost.w, ost.pad, ost.wp(), ost.img(),
                                 *scale_h, *scale_w, st.cs, ost.cs], total);
            }
            Op::Concat { axis } => {
                let y = node.outputs[0];
                let ys = g.shape(y).to_vec();
                if *axis == 1 && ys.len() == 4 {
                    let (yp, ost) = self.new_spatial(y);
                    let pixels = self.batch(y) * ost.hp() * ost.wp();
                    let mut run = 0;
                    for &i in &node.inputs {
                        let (ip, ist) = self.spatial(i)?;
                        ensure!(ist.hp() == ost.hp() && ist.wp() == ost.wp(), "concat: storage mismatch");
                        if !(ip.buf == yp.buf && ip.off == yp.off + run) {
                            let n = pixels * ist.c;
                            self.launch("cnn_axis_copy_f16", vec![(ip.buf, 0), (yp.buf, 0)],
                                        vec![n, ist.c, 1, ist.cs, ip.off, ost.cs, yp.off + run], n);
                            self.concat_copies += 1;
                        }
                        run += ist.c;
                    }
                } else {
                    let yb = self.new_flat(y);
                    let outer: usize = ys[..*axis].iter().product();
                    let inner: usize = ys[axis + 1..].iter().product();
                    let mut off = 0;
                    for &i in &node.inputs {
                        let ib = self.flat(i)?;
                        let len = g.shape(i)[*axis];
                        let n = outer * len * inner;
                        self.launch("cnn_axis_copy_f16", vec![ib, yb], vec![n, len, inner, len, 0, ys[*axis], off], n);
                        off += len;
                    }
                }
            }
            Op::Split { axis, parts } => {
                let x = node.inputs[0];
                let xs = g.shape(x).to_vec();
                if *axis == 1 && xs.len() == 4 && self.is_spatial(x) {
                    // channel slices of the input: views, no copy
                    let (xp, st) = self.spatial(x)?;
                    let mut off = 0;
                    for (&o, &len) in node.outputs.iter().zip(parts) {
                        let mut ost = st;
                        ost.c = len;
                        self.placed.insert(o, Placed { buf: xp.buf, off: xp.off + off, layout: Layout::Spatial(ost) });
                        off += len;
                    }
                } else {
                    let xb = self.flat(x)?;
                    let outer: usize = xs[..*axis].iter().product();
                    let inner: usize = xs[axis + 1..].iter().product();
                    let mut off = 0;
                    for (&o, &len) in node.outputs.iter().zip(parts) {
                        let ob = self.new_flat(o);
                        let n = outer * len * inner;
                        self.launch("cnn_axis_copy_f16", vec![xb, ob], vec![n, len, inner, xs[*axis], off, len, 0], n);
                        off += len;
                    }
                }
            }
            Op::Binary(op) => {
                let (a, b, y) = (node.inputs[0], node.inputs[1], node.outputs[0]);
                // x·x → ReduceSum (trailing axes) → Sqrt is an L2 norm; the sum of squares
                // overflows f16 long before the norm does, so it runs as one kernel that
                // accumulates in f32 and stores only the root.
                if *op == BinaryOp::Mul && a == b {
                    let consumers = |t: TensorId| -> Vec<usize> { g.nodes.iter().enumerate().filter(|(_, m)| m.inputs.contains(&t)).map(|(i, _)| i).collect() };
                    if let &[r] = consumers(y).as_slice() {
                        if let Op::Reduce { kind: ReduceOp::Sum, axes, .. } = &g.nodes[r].op {
                            let xs = g.shape(a).to_vec();
                            let trailing = axes.iter().enumerate().all(|(i, &ax)| ax == xs.len() - axes.len() + i);
                            if let (&[q], true) = (consumers(g.nodes[r].outputs[0]).as_slice(), trailing) {
                                if matches!(g.nodes[q].op, Op::Unary(UnaryOp::Sqrt)) {
                                    let len: usize = axes.iter().map(|&ax| xs[ax]).product();
                                    let rows = g.tensors[a].numel() / len;
                                    let xb = self.flat(a)?;
                                    let yb = self.new_flat(g.nodes[q].outputs[0]);
                                    self.launch("cnn_reduce_last_f16", vec![xb, yb], vec![rows, len, 3], rows);
                                    self.fused.insert(r);
                                    self.fused.insert(q);
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
                let code = match op {
                    BinaryOp::Add => 0,
                    BinaryOp::Sub => 1,
                    BinaryOp::Mul => 2,
                    BinaryOp::Div => 3,
                    BinaryOp::Pow => 4,
                    BinaryOp::Max => 5,
                    BinaryOp::Min => 6,
                };
                let ys = g.shape(y).to_vec();
                let same = g.shape(a) == ys.as_slice() && g.shape(b) == ys.as_slice();
                let spatial = same && ys.len() == 4 && code == 0 && !g.is_weight(a) && !g.is_weight(b)
                    && (self.is_spatial(a) || self.is_spatial(b));
                if spatial {
                    let (ap, ast) = self.spatial(a)?;
                    let (bp, bst) = self.spatial(b)?;
                    // the activation that alone reads the sum (ResNet's Add → ReLU) goes in the same pass
                    let act = match self.sole_reader(step, y) {
                        Some((i, Node { op: Op::Unary(u), outputs, .. })) if self.gpu.has_kernel("cnn_add_act_nhwc_f16") => {
                            unary_code(*u).ok().map(|c| (i, c, outputs[0]))
                        }
                        _ => None,
                    };
                    let y = act.map_or(y, |a| a.2);
                    let (yp, ost) = self.new_spatial(y);
                    let n = self.batch(y) * ost.c * ost.h * ost.w;
                    let mut consts = vec![n, ost.c, ost.h, ost.w, ost.pad, ost.wp(), ast.cs, ast.img(), bst.cs, bst.img(), ost.cs, ost.img()];
                    if let Some((i, code, _)) = act {
                        consts.push(code);
                        self.fused.insert(i);
                        self.launch("cnn_add_act_nhwc_f16", vec![ap.at(), bp.at(), yp.at()], consts, n);
                    } else {
                        self.launch("cnn_add_nhwc_f16", vec![ap.at(), bp.at(), yp.at()], consts, n);
                    }
                } else {
                    let (ab, bb) = (self.flat(a)?, self.flat(b)?);
                    let yb = self.new_flat(y);
                    // size-1 output axes carry no index: drop them; adjacent axes both
                    // operands walk contiguously (or both broadcast) are one axis — a
                    // batched 5-D deformable-attention product folds its batch into the
                    // query axis; then pad to rank 4
                    let (fa, fb) = (broadcast_strides(g.shape(a), &ys), broadcast_strides(g.shape(b), &ys));
                    let mut axes: Vec<(usize, usize, usize)> = (0..ys.len()).filter(|&d| ys[d] != 1).map(|d| (ys[d], fa[d], fb[d])).collect();
                    let mut i = 0;
                    while i + 1 < axes.len() {
                        let ((d0, a0, b0), (d1, a1, b1)) = (axes[i], axes[i + 1]);
                        if a0 == a1 * d1 && b0 == b1 * d1 {
                            axes[i] = (d0 * d1, a1, b1);
                            axes.remove(i + 1);
                        } else {
                            i += 1;
                        }
                    }
                    ensure!(axes.len() <= 4, "binary: {} non-unit axes > 4 ({:?} · {:?})", axes.len(), g.shape(a), g.shape(b));
                    let pad4 = |v: Vec<usize>, fill: usize| -> Vec<usize> { let mut o = vec![fill; 4 - v.len()]; o.extend(v); o };
                    let dims = pad4(axes.iter().map(|x| x.0).collect(), 1);
                    let sa = pad4(axes.iter().map(|x| x.1).collect(), 0);
                    let sb = pad4(axes.iter().map(|x| x.2).collect(), 0);
                    let n: usize = ys.iter().product();
                    self.launch("cnn_binary_bcast_f16", vec![ab, bb, yb],
                                vec![n, dims[1], dims[2], dims[3], sa[0], sa[1], sa[2], sa[3], sb[0], sb[1], sb[2], sb[3], code], n);
                }
            }
            Op::Unary(u) => {
                let (x, y) = (node.inputs[0], node.outputs[0]);
                let code = unary_code(*u)?;
                if self.is_spatial(x) && g.shape(y).len() == 4 && self.gpu.has_kernel("cnn_act_nhwc_f16") {
                    // stay in NHWC (e.g. a ReLU between convs)
                    let (xp, xst) = self.spatial(x)?;
                    let (yp, ost) = self.new_spatial(y);
                    let n = self.batch(y) * ost.c * ost.h * ost.w;
                    self.launch("cnn_act_nhwc_f16", vec![xp.at(), yp.at()],
                                vec![n, ost.c, ost.h, ost.w, ost.pad, ost.wp(), xst.cs, xst.img(), ost.cs, ost.img(), code], n);
                    return Ok(());
                }
                let xb = self.flat(x)?;
                let yb = self.new_flat(y);
                let n = g.tensors[y].numel();
                self.launch("cnn_act_f16", vec![xb, yb], vec![n, code], n);
            }
            Op::View => {
                let (x, y) = (node.inputs[0], node.outputs[0]);
                let (xb, xo) = self.flat(x)?;
                self.placed.insert(y, Placed { buf: xb, off: (xo / 2) as usize, layout: Layout::Flat });
            }
            Op::Transpose { perm } => {
                let (x, y) = (node.inputs[0], node.outputs[0]);
                let xs = g.shape(x).to_vec();
                let xb = self.flat(x)?;
                let yb = self.new_flat(y);
                let st = strides_of(&xs);
                let ys = g.shape(y).to_vec();
                // size-1 output axes carry no index: drop them, pad to rank 6
                let keep: Vec<usize> = (0..ys.len()).filter(|&d| ys[d] != 1).collect();
                ensure!(keep.len() <= 6, "transpose: {} non-unit axes > 6", keep.len());
                let lead = 6 - keep.len();
                let mut dims = vec![1; lead];
                dims.extend(keep.iter().map(|&d| ys[d]));
                let mut s6 = vec![0; lead];
                s6.extend(keep.iter().map(|&d| st[perm[d]]));
                let n: usize = ys.iter().product();
                self.launch("cnn_permute_f16", vec![xb, yb],
                            vec![n, dims[1], dims[2], dims[3], dims[4], dims[5], s6[0], s6[1], s6[2], s6[3], s6[4], s6[5]], n);
            }
            Op::LayerNorm { eps } => {
                let (x, w, y) = (node.inputs[0], node.inputs[1], node.outputs[0]);
                let d = *g.shape(x).last().unwrap();
                let rows = g.tensors[x].numel() / d;
                let xb = self.flat(x)?;
                let wb = (self.weight(w), 0);
                let bb = match node.inputs.get(2) {
                    Some(&b) => (self.weight(b), 0),
                    None => wb,
                };
                let yb = self.new_flat(y);
                self.launch("cnn_layernorm_warp_f16", vec![xb, wb, bb, yb],
                            vec![rows * 32, d, eps.to_bits() as usize, (node.inputs.len() > 2) as usize], rows * 32);
            }
            Op::MatMul => {
                let (a, b, y) = (node.inputs[0], node.inputs[1], node.outputs[0]);
                let (sa, sb) = (g.shape(a).to_vec(), g.shape(b).to_vec());
                let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
                let nn = sb[sb.len() - 1];
                let a_batch: usize = sa[..sa.len() - 2].iter().product::<usize>().max(1);
                let b_batch: usize = sb[..sb.len() - 2].iter().product::<usize>().max(1);
                ensure!(b_batch == a_batch || b_batch == 1, "matmul: batch broadcast {sa:?} x {sb:?}");
                let (ab, bb) = (self.flat(a)?, self.flat(b)?);
                let yb = self.new_flat(y);
                let total = a_batch * m * nn;
                self.launch("cnn_matmul_f16", vec![ab, bb, yb], vec![total, m, k, nn, (b_batch > 1) as usize], total);
            }
            Op::Softmax { axis } => {
                let (x, y) = (node.inputs[0], node.outputs[0]);
                let xs = g.shape(x).to_vec();
                let xb = self.flat(x)?;
                let yb = self.new_flat(y);
                let dim = xs[*axis];
                let inner: usize = xs[axis + 1..].iter().product();
                let slices = g.tensors[x].numel() / dim;
                if inner == 1 {
                    // the last axis (attention rows): a warp per row, coalesced
                    self.launch("cnn_softmax_rows_f16", vec![xb, yb], vec![slices * 32, dim], slices * 32);
                } else {
                    self.launch("cnn_softmax_f16", vec![xb, yb], vec![slices, dim, inner], slices);
                }
            }
            Op::Slice { starts, ends, steps } => {
                let (x, y) = (node.inputs[0], node.outputs[0]);
                let xs = g.shape(x).to_vec();
                let cut: Vec<usize> = (0..xs.len()).filter(|&d| starts[d] != 0 || ends[d] != xs[d]).collect();
                ensure!(cut.len() <= 1 && steps.iter().all(|&s| s == 1), "slice: only one axis, step 1");
                if cut == [1] && xs.len() == 4 && self.is_spatial(x) {
                    // channel slice of a spatial tensor: a view, like split
                    let (xp, st) = self.spatial(x)?;
                    let mut ost = st;
                    ost.c = ends[1] - starts[1];
                    self.placed.insert(y, Placed { buf: xp.buf, off: xp.off + starts[1], layout: Layout::Spatial(ost) });
                    return Ok(());
                }
                let xb = self.flat(x)?;
                let yb = self.new_flat(y);
                let ax = cut.first().copied().unwrap_or(0);
                let len = ends[ax] - starts[ax];
                let outer: usize = xs[..ax].iter().product();
                let inner: usize = xs[ax + 1..].iter().product();
                let n = outer * len * inner;
                self.launch("cnn_axis_copy_f16", vec![xb, yb], vec![n, len, inner, xs[ax], starts[ax], len, 0], n);
            }
            Op::Gemm { trans_b, act } => {
                // y[M,N] = x[M,K]·Wᵀ + b as a 1×1 conv over a 1×M "image" of K channels:
                // the flat rows are exactly that NHWC storage, so the tensor-core igemm
                // reads and writes them in place (lower_for_gpu keeps only such Gemms)
                ensure!(crate::passes::gemm_on_tensor_cores(g, &node.inputs, *act), "gemm: not a tensor-core shape (lower it)");
                let (x, w, y) = (node.inputs[0], node.inputs[1], node.outputs[0]);
                let (xs, ws) = (g.shape(x).to_vec(), g.shape(w).to_vec());
                let (m, k) = (xs[0], xs[1]);
                let data = g.weight(w).unwrap();
                let (nn, wt) = if *trans_b {
                    (ws[0], data.to_vec())
                } else {
                    let nn = ws[1];
                    let mut t = vec![0.0f32; nn * k];
                    for r in 0..k {
                        for c in 0..nn {
                            t[c * k + r] = data[r * nn + c];
                        }
                    }
                    (nn, t)
                };
                let bias = node.inputs.get(2).map(|&b| g.weight(b).expect("gemm bias weight").to_vec());
                let act_code = match act {
                    None => 0,
                    Some(UnaryOp::Silu) => 1,
                    Some(UnaryOp::Sigmoid) => 2,
                    Some(UnaryOp::Relu) => 3,
                    Some(UnaryOp::HardSigmoid) => 8,
                    Some(UnaryOp::HardSwish) => 9,
                    Some(u) => bail!("gemm: fused activation {u:?} not on CUDA yet"),
                };
                let xb = self.flat(x)?;
                let yb = self.new_flat(y);
                ensure!(xb.1 % 16 == 0 && yb.1 % 16 == 0, "gemm: rows not 16-byte aligned");
                let geom = ConvGeom { cin: k, h: 1, w: m, cout: nn, kh: 1, kw: 1, sh: 1, sw: 1, pads: [0; 4] };
                let st_in = Storage { c: k, cs: k, h: 1, w: m, pad: 0, spare: 0 };
                let st_out = Storage { c: nn, cs: nn, h: 1, w: m, pad: 0, spare: 0 };
                let plan = self.gpu.conv_plan(geom, &wt, bias.as_deref(), act_code, st_in, st_out)?;
                self.steps.push(Step::Conv { plan, x: xb, y: yb, n: 1 });
            }
            Op::ReduceMean { axes, .. } | Op::Reduce { axes, .. } => {
                let kind = match &node.op {
                    Op::Reduce { kind, .. } => *kind,
                    _ => ReduceOp::Mean,
                };
                let (x, y) = (node.inputs[0], node.outputs[0]);
                let xs = g.shape(x).to_vec();
                ensure!(axes.iter().enumerate().all(|(i, &a)| a == xs.len() - axes.len() + i), "reduce: only trailing axes on the GPU ({axes:?} of {xs:?})");
                let len: usize = axes.iter().map(|&a| xs[a]).product();
                let rows = g.tensors[x].numel() / len;
                let code = match kind {
                    ReduceOp::Sum => 0,
                    ReduceOp::Max => 1,
                    ReduceOp::Mean => 2,
                };
                let xb = self.flat(x)?;
                let yb = self.new_flat(y);
                self.launch("cnn_reduce_last_f16", vec![xb, yb], vec![rows, len, code], rows);
            }
            Op::GridSample => {
                let (x, grid, y) = (node.inputs[0], node.inputs[1], node.outputs[0]);
                let (xs, gs) = (g.shape(x).to_vec(), g.shape(grid).to_vec());
                let (xb, gb) = (self.flat(x)?, self.flat(grid)?);
                let yb = self.new_flat(y);
                let n = xs[0] * gs[1] * gs[2];
                self.launch("cnn_grid_sample_f16", vec![xb, gb, yb], vec![n, xs[1], xs[2], xs[3], gs[1], gs[2]], n);
            }
            Op::ScaleShift { scale, shift } => {
                let (x, y) = (node.inputs[0], node.outputs[0]);
                let (s, t) = (scale.to_bits() as usize, shift.to_bits() as usize);
                if self.is_spatial(x) {
                    // stay in NHWC: no layout round trip around the conv that produced it
                    let (xp, xst) = self.spatial(x)?;
                    let (yp, ost) = self.new_spatial(y);
                    let n = self.batch(y) * ost.c * ost.h * ost.w;
                    self.launch("cnn_scale_shift_nhwc_f16", vec![xp.at(), yp.at()],
                                vec![n, ost.c, ost.h, ost.w, ost.pad, ost.wp(), xst.cs, xst.img(), ost.cs, ost.img(), s, t], n);
                } else {
                    let xb = self.flat(x)?;
                    let yb = self.new_flat(y);
                    let n = g.tensors[y].numel();
                    self.launch("cnn_scale_shift_f16", vec![xb, yb], vec![n, s, t], n);
                }
            }
            Op::Pad { pads, value } => {
                let (x, y) = (node.inputs[0], node.outputs[0]);
                let (xs, ys) = (g.shape(x).to_vec(), g.shape(y).to_vec());
                let xb = self.flat(x)?;
                let yb = self.new_flat(y);
                let n = g.tensors[y].numel();
                self.launch("cnn_pad4_f16", vec![xb, yb], vec![n, xs[2], xs[3], ys[2], ys[3], pads[0], pads[1], value.to_bits() as usize], n);
            }
            Op::TopK { k } | Op::TopKGather { k } if *k > 512 => bail!("TopK k = {k}: the CUDA kernels hold at most 512 (CNN_TOPK_MAXK)"),
            Op::TopK { k } => {
                let (x, y) = (node.inputs[0], node.outputs[0]);
                let len = *g.shape(x).last().unwrap();
                let n = g.tensors[x].numel();
                let xb = self.flat(x)?;
                let yb = self.new_flat(y);
                // a 256-thread block per row (radix select in shared memory)
                let rows = n / len;
                self.launch("cnn_topk_last_f16", vec![xb, yb], vec![len, *k], rows * 256);
            }
            Op::TopKGather { k } => {
                let (scores, d, y) = (node.inputs[0], node.inputs[1], node.outputs[0]);
                let ds = g.shape(d).to_vec();
                let (sb, db) = (self.flat(scores)?, self.flat(d)?);
                let yb = self.new_flat(y);
                self.launch("cnn_topk_gather_f16", vec![sb, db, yb], vec![ds[1], ds[2], *k], ds[0] * 256);
            }
            other => bail!("op {other:?} not on CUDA yet"),
        }
        Ok(())
    }
}

/// Liveness-based buffer sharing: logical buffers whose step intervals do not
/// overlap share one physical buffer (same `Kind` key; flat ones grow to the
/// largest). Returns logical -> physical and the physical sizes.
fn reuse_buffers<G: GpuDev>(steps: &[Step<G>], sizes: &[usize], kinds: &[Kind]) -> (Vec<usize>, Vec<usize>) {
    let n = sizes.len();
    let (mut first, mut last) = (vec![usize::MAX; n], vec![0usize; n]);
    for (i, s) in steps.iter().enumerate() {
        let used: Vec<usize> = match s {
            Step::Conv { x, y, .. } => vec![x.0, y.0],
            Step::Launch { bufs, .. } => bufs.iter().map(|b| b.0).collect(),
        };
        for b in used {
            first[b] = first[b].min(i);
            last[b] = last[b].max(i);
        }
    }
    let mut map = vec![usize::MAX; n];
    let mut phys: Vec<usize> = vec![];
    let mut free: HashMap<Kind, Vec<usize>> = HashMap::new();
    // persistent and never-used buffers keep their own storage
    for b in 0..n {
        if kinds[b] == Kind::Persistent || first[b] == usize::MAX {
            map[b] = phys.len();
            phys.push(sizes[b]);
        }
    }
    let mut starts: Vec<Vec<usize>> = vec![vec![]; steps.len()];
    let mut ends: Vec<Vec<usize>> = vec![vec![]; steps.len()];
    for b in 0..n {
        if map[b] == usize::MAX {
            starts[first[b]].push(b);
            ends[last[b]].push(b);
        }
    }
    for i in 0..steps.len() {
        for &b in &starts[i] {
            let key = match kinds[b] {
                Kind::Flat => Kind::Flat,
                k => k,
            };
            let pool = free.entry(key).or_default();
            // best fit: the smallest free buffer that holds it, else the largest (grown)
            let pick = pool
                .iter()
                .enumerate()
                .filter(|(_, &p)| phys[p] >= sizes[b])
                .min_by_key(|(_, &p)| phys[p])
                .map(|(j, _)| j)
                .or_else(|| (0..pool.len()).max_by_key(|&j| phys[pool[j]]));
            match pick {
                Some(j) => {
                    let p = pool.swap_remove(j);
                    phys[p] = phys[p].max(sizes[b]);
                    map[b] = p;
                }
                None => {
                    map[b] = phys.len();
                    phys.push(sizes[b]);
                }
            }
        }
        for &b in &ends[i] {
            free.entry(kinds[b]).or_default().push(map[b]);
        }
    }
    (map, phys)
}

/// Per-conv variant choice by measurement on the real buffers (cuDNN
/// "benchmark" style): each candidate runs `REPS` times, the fastest median
/// wins. Buffers hold zeros at plan time, which is fine for timing.
fn autotune<'a, G: GpuDev>(g: &G, buf: &dyn Fn(usize) -> (&'a G::Buf, u64), steps: &mut [Step<G>]) -> Result<()> {
    const REPS: usize = 5;
    /// dispatches per timed submit: one submit per sample made the timings
    /// mostly submit latency, and the picks (and fp16 results) vary per run
    const BURST: usize = 4;
    let log = std::env::var("OJAS_TUNE_LOG").is_ok();
    // bring the GPU to its working clock first: short bursts at idle clocks
    // time 2x slower and in random order
    let t0 = std::time::Instant::now();
    while t0.elapsed().as_millis() < 300 {
        let enc = g.begin();
        for s in steps.iter() {
            if let Step::Conv { plan, x, y, n } = s {
                let (xb, yb) = (buf(x.0), buf(y.0));
                plan.dispatch_at(g, &enc, (xb.0, xb.1 + x.1), (yb.0, yb.1 + y.1), *n)?;
            }
        }
        g.submit(enc)?;
    }
    for s in steps.iter_mut() {
        let Step::Conv { plan, x, y, n } = s else { continue };
        let cands = plan.variants();
        if cands.len() < 2 {
            continue;
        }
        let mut best = (f64::INFINITY, plan.variant());
        let mut seen = vec![];
        for v in cands {
            plan.set_variant(v);
            let mut ts = Vec::with_capacity(REPS);
            for _ in 0..REPS + 1 {
                let enc = g.begin();
                let t = std::time::Instant::now();
                let (xb, yb) = (buf(x.0), buf(y.0));
                for _ in 0..BURST {
                    plan.dispatch_at(g, &enc, (xb.0, xb.1 + x.1), (yb.0, yb.1 + y.1), *n)?;
                }
                g.submit(enc)?;
                ts.push(t.elapsed().as_secs_f64() / BURST as f64);
            }
            ts.remove(0);
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            seen.push(format!("{v:?} {:.1}", ts[REPS / 2] * 1e6));
            if ts[REPS / 2] < best.0 {
                best = (ts[REPS / 2], v);
            }
        }
        plan.set_variant(best.1);
        if log {
            let gm = plan.geom();
            eprintln!("tune n{} {}x{}x{} -> {} k{}x{} s{} | {} | pick {:?}", n, gm.cin, gm.h, gm.w, gm.cout, gm.kh, gm.kw,
                      gm.sh, seen.join(", "), best.1);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub type CudaExecutor = GpuExecutor<ojas_cuda::CudaGpu>;
#[cfg(feature = "vulkan")]
pub type VkExecutor = GpuExecutor<ojas_vulkan::VkGpu>;

impl<G: GpuDev> GpuExecutor<G> {
    /// Plan `g` for device `ordinal`. The graph's batch is baked into its shapes.
    pub fn new(g: &Graph, ordinal: usize) -> Result<Self> {
        Self::new_in(g, ordinal, None)
    }

    /// `new` with the activation buffers placed in a shared `arena` (weights,
    /// the input and the outputs keep their own memory). Autotuning then runs
    /// on the first forward, once the arena exists.
    pub fn new_in(g: &Graph, ordinal: usize, arena: Option<&Arc<ArenaOf<G>>>) -> Result<Self> {
        let mut gpu = G::open(ordinal)?;
        gpu.ensure_family("cnn")?;
        ensure!(g.inputs.len() == 1, "GPU executor: one graph input");
        let input = g.inputs[0];
        let xs = g.shape(input).to_vec();
        ensure!(xs.len() == 4, "GPU executor: 4-D image input");
        let pad = g.nodes.iter().filter_map(|n| match &n.op { Op::Conv { pads, .. } => pads.iter().max().copied(), _ => None }).max().unwrap_or(0).max(PAD);
        let mut p = Planner {
            g,
            gpu: &gpu,
            sizes: vec![],
            kinds: vec![],
            steps: vec![],
            placed: HashMap::new(),
            place: plan_placement(g),
            roots: HashMap::new(),
            converted: HashMap::new(),
            weights: HashMap::new(),
            uploads: vec![],
            concat_copies: 0,
            fused: Default::default(),
            pad,
        };
        // input: flat NCHW staging -> padded NHWC with channel storage rounded
        // up to a multiple of 4 (RGB -> 4 so the stem conv's kw*C reaches 16)
        let input_numel = g.tensors[input].numel();
        let stage = p.alloc_kind(input_numel, Kind::Persistent);
        let cs = xs[1].div_ceil(4) * 4;
        let st = Storage { c: cs, cs, h: xs[2], w: xs[3], pad, spare: SPARE };
        let ib = p.alloc_kind(xs[0] * st.img(), Kind::Persistent);
        let n = input_numel;
        p.launch("cnn_nchw_to_nhwc_f16", vec![(stage, 0), (ib, 0)], vec![n, xs[1], xs[2], xs[3], st.pad, st.wp(), st.img(), st.cs], n);
        p.placed.insert(input, Placed { buf: ib, off: 0, layout: Layout::Spatial(st) });
        // per step: the IR op it came from (the profile's OJAS_PROFILE_NODE rows)
        let mut origin: Vec<String> = vec!["input".into(); p.steps.len()];
        for step in 0..g.nodes.len() {
            if p.fused.contains(&step) {
                continue;
            }
            p.node(step).with_context(|| format!("node {} ({:?})", g.nodes[step].name, g.nodes[step].op))?;
            let op = format!("{:?}", g.nodes[step].op);
            let short = op.split([' ', '(', '{']).next().unwrap_or("").to_string();
            origin.resize(p.steps.len(), short);
        }
        let mut outputs = vec![];
        for &o in &g.outputs {
            let (b, off) = p.flat(o)?;
            ensure!(off == 0, "graph output is a view at an offset");
            outputs.push((b, g.tensors[o].numel()));
        }
        let Planner { sizes, mut kinds, steps, uploads, concat_copies, .. } = p;
        let input_storage = st;
        for &(b, _) in &outputs {
            kinds[b] = Kind::Persistent;
        }
        let (map, sizes) = if std::env::var("OJAS_REUSE").map_or(true, |v| v != "0") {
            reuse_buffers(&steps, &sizes, &kinds)
        } else {
            ((0..sizes.len()).collect(), sizes)
        };
        let mut steps = steps;
        for s in steps.iter_mut() {
            match s {
                Step::Conv { x, y, .. } => {
                    x.0 = map[x.0];
                    y.0 = map[y.0];
                }
                Step::Launch { bufs, .. } => {
                    for b in bufs.iter_mut() {
                        b.0 = map[b.0];
                    }
                }
            }
        }
        let outputs: Vec<(usize, usize)> = outputs.into_iter().map(|(b, n)| (map[b], n)).collect();
        let uploads: Vec<(usize, Vec<f32>)> = uploads.into_iter().map(|(b, d)| (map[b], d)).collect();
        let (stage, ib) = (map[stage], map[ib]);
        // physical buffers that may live in the arena: used by a step, not
        // persistent (input stage, input, outputs, weights), not uploaded
        let mut phys_kind: Vec<Option<Kind>> = vec![None; sizes.len()];
        for (b, &k) in kinds.iter().enumerate() {
            let slot = &mut phys_kind[map[b]];
            if k == Kind::Persistent || slot.is_some_and(|s| s == Kind::Persistent) {
                *slot = Some(Kind::Persistent);
            } else {
                *slot = Some(k);
            }
        }
        let mut used = vec![false; sizes.len()];
        for s in &steps {
            match s {
                Step::Conv { x, y, .. } => {
                    used[x.0] = true;
                    used[y.0] = true;
                }
                Step::Launch { bufs, .. } => bufs.iter().for_each(|b| used[b.0] = true),
            }
        }
        for (b, _) in &uploads {
            phys_kind[*b] = Some(Kind::Persistent);
        }
        let mut arena_off = vec![None; sizes.len()];
        let mut pads = vec![];
        let mut total = 0usize;
        if let Some(ar) = arena {
            for b in 0..sizes.len() {
                let Some(k) = phys_kind[b] else { continue };
                if k == Kind::Persistent || !used[b] {
                    continue;
                }
                total = total.div_ceil(256) * 256;
                arena_off[b] = Some(total as u64);
                total += sizes[b] * 2;
                if let Kind::Spatial { .. } = k {
                    pads.push((b, k));
                }
            }
            ar.reserve(total)?;
        }
        // one row per spatial arena buffer: offset (halves), n, hp, wp, cs, pad, h, w, c
        let mut rows: Vec<u32> = vec![];
        let mut most = 0usize;
        for &(b, k) in &pads {
            let Kind::Spatial { hp, wp, cs, n, pad, h, w, c } = k else { continue };
            ensure!(arena_off[b].unwrap() / 2 < u32::MAX as u64, "arena offset past 32 bits");
            rows.extend([(arena_off[b].unwrap() / 2) as usize, n, hp, wp, cs, pad, h, w, c].map(|v| v as u32));
            most = most.max(n * (h + 1 + if c < cs { h * w } else { 0 }));
        }
        let pad_tab = (!rows.is_empty()).then(|| {
            let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.upload_bytes(&bytes).map(|t| (t, (rows.len() / 9) as u32, (most.div_ceil(8) as u32).min(1024)))
        }).transpose()?;
        let mut bufs: Vec<G::Buf> = sizes
            .iter()
            .zip(&arena_off)
            .map(|(&n, off)| gpu.upload_f16(&vec![0.0; if off.is_some() { 1 } else { n }]))
            .collect();
        for (b, data) in uploads {
            bufs[b] = gpu.upload_f16(&data);
        }
        let tune = std::env::var("OJAS_AUTOTUNE").map_or(true, |v| v != "0");
        if tune && arena.is_none() {
            autotune(&gpu, &|i| (&bufs[i], 0), &mut steps)?;
        }
        Ok(GpuExecutor {
            gpu,
            bufs,
            arena_off,
            arena: arena.cloned(),
            pad_tab,
            arena_need: total,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            tuned: !tune || arena.is_none(),
            steps,
            input_stage: stage,
            input_numel,
            input_buf: ib,
            input_storage,
            batch: xs[0],
            output_shapes: g.outputs.iter().map(|&o| g.shape(o).to_vec()).collect(),
            outputs,
            concat_copies,
            profile: false,
            origin,
            graphs: HashMap::new(),
            op_times: HashMap::new(),
        })
    }

    /// Bytes of device memory held by activation + weight buffers and conv
    /// plans (arena members count the arena bytes they use, not the arena).
    pub fn device_bytes(&self) -> usize {
        let own: usize = self.bufs.iter().zip(&self.arena_off).filter(|(_, o)| o.is_none()).map(|(b, _)| G::buf_len(b)).sum();
        own + self.arena_bytes() + self.plan_bytes()
    }

    /// Bytes of the shared arena this executor uses (0 without one).
    pub fn arena_bytes(&self) -> usize {
        self.arena_need
    }

    /// Bytes held by conv plans (packed weights, GEMM weights, offset tables).
    pub fn plan_bytes(&self) -> usize {
        self.steps.iter().map(|s| if let Step::Conv { plan, .. } = s { GpuConv::device_bytes(plan) } else { 0 }).sum()
    }

    /// Upload `input` (NCHW f32, the graph's shape), run, return outputs (f32).
    pub fn run(&mut self, input: &[f32]) -> Result<Vec<Vec<f32>>> {
        ensure!(input.len() == self.input_numel, "input: {} elements, want {}", input.len(), self.input_numel);
        // in place: the recorded steps (and a captured graph) address this buffer
        let bytes: Vec<u8> = input.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
        self.gpu.write_bytes(&mut self.bufs[self.input_stage], 0, &bytes)?;
        self.forward()?;
        let mut outs = vec![];
        for &(b, n) in &self.outputs {
            let mut v = vec![0.0f32; n];
            self.gpu.read(&self.bufs[b], &mut v);
            outs.push(v);
        }
        Ok(outs)
    }

    pub fn gpu(&self) -> &G {
        &self.gpu
    }

    /// The padded-NHWC input buffer and its layout (one image per `img()`
    /// elements); fill it on the device and call `forward_device`.
    pub fn input(&self) -> (&G::Buf, Storage) {
        (&self.bufs[self.input_buf], self.input_storage)
    }

    /// Shape of graph output `i` (batch included).
    pub fn output_shapes(&self) -> &[Vec<usize>] {
        &self.output_shapes
    }

    pub fn output_shape(&self, i: usize) -> &[usize] {
        &self.output_shapes[i]
    }

    /// Batch the graph was planned for.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Forward from the padded input buffer (skips the host staging step).
    pub fn forward_device(&mut self) -> Result<()> {
        self.run_steps(1)
    }

    /// Device buffer of graph output `i` (contiguous, logical shape).
    pub fn output(&self, i: usize) -> &G::Buf {
        &self.bufs[self.outputs[i].0]
    }

    /// First `images` images of graph output `i`, as f32.
    pub fn read_output(&self, i: usize, images: usize) -> Result<Vec<f32>> {
        let (b, numel) = self.outputs[i];
        let per = numel / self.batch();
        let n = per * images.min(self.batch());
        let mut raw = vec![0u8; n * 2];
        self.gpu.read_bytes(&self.bufs[b], 0, &mut raw)?;
        Ok(raw.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect())
    }

    /// Enqueue every step and sync once (the input stage already uploaded).
    pub fn forward(&mut self) -> Result<()> {
        self.run_steps(0)
    }

    /// Buffer `i` as (allocation, byte offset).
    fn buf(&self, i: usize) -> (&G::Buf, u64) {
        match (self.arena_off[i], &self.arena) {
            (Some(o), Some(a)) => (a.buf.get().expect("arena allocated before use"), o),
            _ => (&self.bufs[i], 0),
        }
    }

    /// Arena members: allocate the arena and autotune on first use. Returns
    /// whether the pad cells need re-zeroing (another member ran in between).
    fn claim(&mut self) -> Result<bool> {
        let Some(ar) = self.arena.clone() else { return Ok(false) };
        ar.get(&self.gpu)?;
        let rezero = ar.owner.swap(self.id, Ordering::Relaxed) != self.id;
        if !self.tuned {
            if rezero {
                let enc = self.gpu.begin();
                self.zero_pads(&enc)?;
                self.gpu.submit(enc)?;
            }
            let mut steps = std::mem::take(&mut self.steps);
            let r = autotune(&self.gpu, &|i| self.buf(i), &mut steps);
            self.steps = steps;
            r?;
            self.tuned = true;
            return Ok(false);
        }
        Ok(rezero)
    }

    /// Enqueue one step; returns its kind for the profile.
    fn dispatch_step(&self, enc: &G::Enc, i: usize) -> Result<&'static str> {
        let s = &self.steps[i];
        let g = &self.gpu;
        Ok(match s {
            Step::Conv { plan, x, y, n } => {
                let (xb, yb) = (self.buf(x.0), self.buf(y.0));
                plan.dispatch_at(g, enc, (xb.0, xb.1 + x.1), (yb.0, yb.1 + y.1), *n)?;
                if self.profile && std::env::var("OJAS_PROFILE_CONV").is_ok() {
                    // one row per geometry and variant (profiling only; the labels are few)
                    let c = plan.geom();
                    let label = format!("conv {}x{}x{}->{} k{}x{} s{} {:?}", c.h, c.w, c.cin, c.cout, c.kh, c.kw, c.sh, plan.variant());
                    return Ok(Box::leak(label.into_boxed_str()));
                }
                "conv"
            }
            Step::Launch { name, bufs, consts, grid } => {
                let b: Vec<(&G::Buf, u64)> = bufs.iter().map(|&(i, o)| { let (b, base) = self.buf(i); (b, base + o) }).collect();
                g.dispatch(enc, name, &b, consts, *grid, [256, 1, 1])?;
                if let Some(o) = self.profile.then(|| std::env::var("OJAS_PROFILE_NODE").ok()).flatten().and(self.origin.get(i)) {
                    return Ok(Box::leak(format!("{name} <- {o}").into_boxed_str()));
                }
                name
            }
        })
    }

    fn zero_pads(&self, enc: &G::Enc) -> Result<()> {
        if let Some((tab, rows, blocks)) = &self.pad_tab {
            let ar = self.arena.as_ref().unwrap().buf.get().expect("arena allocated before use");
            self.gpu.dispatch(enc, "cnn_zero_pad_f16", &[(ar, 0), (tab, 0)], &[], [*blocks, *rows, 1], [256, 1, 1])?;
        }
        Ok(())
    }

    fn run_steps(&mut self, first: usize) -> Result<()> {
        let rezero = self.claim()?;
        let graphs = !self.profile && std::env::var("OJAS_CUDA_GRAPHS").map_or(true, |v| v != "0");
        if graphs {
            if rezero {
                let enc = self.gpu.begin();
                self.zero_pads(&enc)?;
                self.gpu.submit(enc)?;
            }
            if !self.graphs.contains_key(&first) {
                // record the steps once (the conv variants are tuned by now)
                let mut run = |enc: &G::Enc| -> Result<()> {
                    for i in first..self.steps.len() {
                        self.dispatch_step(enc, i)?;
                    }
                    Ok(())
                };
                if let Some(gr) = self.gpu.capture(&mut run)? {
                    self.graphs.insert(first, gr);
                }
            }
            if let Some(gr) = self.graphs.get(&first) {
                return self.gpu.replay(gr);
            }
        }
        let g = &self.gpu;
        let mut enc = g.begin();
        if rezero && !graphs {
            self.zero_pads(&enc)?;
        }
        for i in first..self.steps.len() {
            let t0 = self.profile.then(std::time::Instant::now);
            let kind = self.dispatch_step(&enc, i)?;
            if let Some(t0) = t0 {
                g.submit(enc)?;
                enc = g.begin();
                let e = self.op_times.entry(kind).or_insert((0.0, 0));
                e.0 += t0.elapsed().as_secs_f64();
                e.1 += 1;
            }
        }
        g.submit(enc)
    }
}
