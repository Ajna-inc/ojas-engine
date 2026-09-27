//! Define-by-run reverse-mode autodiff. A model is plain Rust calling tape ops;
//! every op records what its backward needs, and `backward` walks the tape in
//! reverse. Nodes are created in execution order, so reverse creation order is
//! a valid reverse topological order.
//!
//! Buffers are reference-counted: `reshape` shares its input's buffer, and a
//! `Param` lends its weights to the tape without a copy (the optimizer then
//! updates them in place).

use std::rc::Rc;

use anyhow::{bail, ensure, Result};

use crate::backend::{strides, Backend, Bcast, Binary, Gemm, Unary, Win};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(pub usize);

enum Op<B: Backend> {
    Leaf,
    Unary(Unary, Var),
    Binary(Binary, Var, Var, Bcast),
    /// C = A·op(B); `shared_b`: one B for every batch
    MatMul { a: Var, b: Var, g: Gemm, shared_b: bool },
    Softmax(Var),
    LayerNorm { x: Var, g: Var, b: Var, mean: B::Buf, rstd: B::Buf },
    Reshape(Var),
    Permute(Var, Vec<usize>),
    Slice { x: Var, axis: usize, start: usize },
    Concat(Vec<Var>, usize),
    Sum { x: Var, scale: f32 },
    /// sum over one axis (dropped from the shape)
    SumAxis { x: Var, axis: usize },
    /// NCHW conv, im2col + GEMM per image; `win` over one image (planes = C)
    Conv { x: Var, w: Var, b: Option<Var>, win: Win, n: usize, cout: usize, groups: usize },
    /// training mode: batch statistics; eval: the running ones (no batch terms in the backward)
    BatchNorm { x: Var, g: Var, b: Var, mean: B::Buf, rstd: B::Buf, train: bool },
    MaxPool { x: Var, idx: B::Buf, win: Win },
    AvgPool { x: Var, win: Win, cip: bool },
    Upsample { x: Var, fy: usize, fx: usize },
    GridSample { x: Var, grid: Var },
    GatherRows { x: Var, idx: B::Buf, b: usize, n: usize, k: usize, c: usize },
}

/// BatchNorm running statistics (not trained; updated in training mode).
pub struct BnRunning<B: Backend> {
    pub mean: B::Buf,
    pub var: B::Buf,
    pub momentum: f32,
    pub eps: f32,
}

impl<B: Backend> BnRunning<B> {
    pub fn new(be: &B, c: usize) -> Self {
        BnRunning { mean: be.alloc(c), var: be.upload(&vec![1.0; c]), momentum: 0.1, eps: 1e-5 }
    }
}

struct Node<B: Backend> {
    val: Rc<B::Buf>,
    shape: Vec<usize>,
    grad: Option<B::Buf>,
    req: bool,
    op: Op<B>,
}

/// A trainable tensor that outlives a tape: weights plus AdamW moments.
pub struct Param<B: Backend> {
    pub name: String,
    pub shape: Vec<usize>,
    pub val: Rc<B::Buf>,
    pub m: B::Buf,
    pub v: B::Buf,
}

impl<B: Backend> Param<B> {
    pub fn new(be: &B, name: impl Into<String>, shape: &[usize], init: &[f32]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(init.len(), n, "param init length");
        Param { name: name.into(), shape: shape.to_vec(), val: Rc::new(be.upload(init)), m: be.alloc(n), v: be.alloc(n) }
    }
}

pub struct Tape<'b, B: Backend> {
    pub be: &'b B,
    nodes: Vec<Node<B>>,
    /// one leaf per parameter, however often the model reads it (its gradient then sums every use)
    params: std::collections::HashMap<*const B::Buf, Var>,
    /// nodes whose value and gradient survive `backward` (besides leaves and the loss)
    keep: std::collections::HashSet<usize>,
    /// stands in for a freed value
    freed: Option<Rc<B::Buf>>,
}

impl<'b, B: Backend> Tape<'b, B> {
    pub fn new(be: &'b B) -> Self {
        Tape { be, nodes: vec![], params: Default::default(), keep: Default::default(), freed: None }
    }

    fn push(&mut self, val: B::Buf, shape: Vec<usize>, op: Op<B>, parents: &[Var]) -> Var {
        self.push_rc(Rc::new(val), shape, op, parents)
    }

    fn push_rc(&mut self, val: Rc<B::Buf>, shape: Vec<usize>, op: Op<B>, parents: &[Var]) -> Var {
        let req = parents.iter().any(|p| self.nodes[p.0].req);
        self.nodes.push(Node { val, shape, grad: None, req, op });
        Var(self.nodes.len() - 1)
    }

    /// A constant (no gradient).
    pub fn input(&mut self, data: &[f32], shape: &[usize]) -> Var {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "input length vs shape {shape:?}");
        let val = self.be.upload(data);
        self.leaf(Rc::new(val), shape, false)
    }

    /// A leaf that receives a gradient (tests, inputs to differentiate).
    pub fn var(&mut self, data: &[f32], shape: &[usize]) -> Var {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "var length vs shape {shape:?}");
        let val = self.be.upload(data);
        self.leaf(Rc::new(val), shape, true)
    }

    /// Bring a parameter onto the tape (shares its buffer); the same parameter
    /// always maps to the same leaf.
    pub fn param(&mut self, p: &Param<B>) -> Var {
        let key = Rc::as_ptr(&p.val);
        if let Some(&v) = self.params.get(&key) {
            return v;
        }
        let v = self.leaf(p.val.clone(), &p.shape, true);
        self.params.insert(key, v);
        v
    }

    /// The leaf of a parameter this tape has read (None: unused).
    pub fn param_var(&self, p: &Param<B>) -> Option<Var> {
        self.params.get(&Rc::as_ptr(&p.val)).copied()
    }

    fn leaf(&mut self, val: Rc<B::Buf>, shape: &[usize], req: bool) -> Var {
        self.nodes.push(Node { val, shape: shape.to_vec(), grad: None, req, op: Op::Leaf });
        Var(self.nodes.len() - 1)
    }

    pub fn shape(&self, v: Var) -> &[usize] {
        &self.nodes[v.0].shape
    }

    pub fn value(&self, v: Var) -> Vec<f32> {
        self.be.download(&self.nodes[v.0].val)
    }

    pub fn buf(&self, v: Var) -> &B::Buf {
        &self.nodes[v.0].val
    }

    /// The gradient after `backward` (None: not on the path to the loss).
    pub fn grad(&self, v: Var) -> Option<&B::Buf> {
        self.nodes[v.0].grad.as_ref()
    }

    pub fn grad_vec(&self, v: Var) -> Option<Vec<f32>> {
        self.grad(v).map(|g| self.be.download(g))
    }

    fn numel(&self, v: Var) -> usize {
        self.nodes[v.0].shape.iter().product()
    }

    // ---------------------------------------------------------------- ops ---

    pub fn unary(&mut self, op: Unary, x: Var) -> Var {
        let n = self.numel(x);
        let y = self.be.alloc_out(n);
        self.be.unary(op, &self.nodes[x.0].val, &y);
        self.push(y, self.nodes[x.0].shape.clone(), Op::Unary(op, x), &[x])
    }

    pub fn relu(&mut self, x: Var) -> Var {
        self.unary(Unary::Relu, x)
    }
    pub fn sigmoid(&mut self, x: Var) -> Var {
        self.unary(Unary::Sigmoid, x)
    }
    pub fn silu(&mut self, x: Var) -> Var {
        self.unary(Unary::Silu, x)
    }
    pub fn gelu(&mut self, x: Var) -> Var {
        self.unary(Unary::Gelu, x)
    }

    pub fn binary(&mut self, op: Binary, a: Var, b: Var) -> Result<Var> {
        let (bc, out) = Bcast::new(&self.nodes[a.0].shape, &self.nodes[b.0].shape)?;
        let y = self.be.alloc_out(bc.n);
        self.be.binary(op, &self.nodes[a.0].val, &self.nodes[b.0].val, &y, &bc);
        Ok(self.push(y, out, Op::Binary(op, a, b, bc), &[a, b]))
    }

    pub fn add(&mut self, a: Var, b: Var) -> Result<Var> {
        self.binary(Binary::Add, a, b)
    }
    pub fn sub(&mut self, a: Var, b: Var) -> Result<Var> {
        self.binary(Binary::Sub, a, b)
    }
    pub fn mul(&mut self, a: Var, b: Var) -> Result<Var> {
        self.binary(Binary::Mul, a, b)
    }
    pub fn div(&mut self, a: Var, b: Var) -> Result<Var> {
        self.binary(Binary::Div, a, b)
    }

    pub fn scale(&mut self, x: Var, s: f32) -> Result<Var> {
        let c = self.input(&[s], &[1]);
        self.mul(x, c)
    }

    /// a [.., M, K] · b [.., K, N] (or b [K, N] shared by every batch); with
    /// `trans_b`, b is stored [.., N, K] (PyTorch Linear weights).
    pub fn matmul_opts(&mut self, a: Var, b: Var, trans_b: bool) -> Result<Var> {
        let (sa, sb) = (self.nodes[a.0].shape.clone(), self.nodes[b.0].shape.clone());
        ensure!(sa.len() >= 2 && sb.len() >= 2, "matmul needs rank ≥ 2: {sa:?} · {sb:?}");
        let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
        let (kb, n) = if trans_b { (sb[sb.len() - 1], sb[sb.len() - 2]) } else { (sb[sb.len() - 2], sb[sb.len() - 1]) };
        ensure!(k == kb, "matmul inner dims: {sa:?} · {sb:?} (trans_b {trans_b})");
        let batch: usize = sa[..sa.len() - 2].iter().product();
        let shared_b = sb.len() == 2;
        if !shared_b {
            ensure!(sb[..sb.len() - 2] == sa[..sa.len() - 2], "matmul batch dims differ: {sa:?} · {sb:?}");
        }
        let g = Gemm { m, n, k, ta: false, tb: trans_b, batch, sa: m * k, sb: if shared_b { 0 } else { k * n }, sc: m * n, alpha: 1.0, beta: 0.0, oa: 0, ob: 0, oc: 0 };
        let y = self.be.alloc_out(batch * m * n);
        self.be.gemm(&self.nodes[a.0].val, &self.nodes[b.0].val, &y, &g);
        let mut out = sa[..sa.len() - 2].to_vec();
        out.extend([m, n]);
        Ok(self.push(y, out, Op::MatMul { a, b, g, shared_b }, &[a, b]))
    }

    pub fn matmul(&mut self, a: Var, b: Var) -> Result<Var> {
        self.matmul_opts(a, b, false)
    }

    /// PyTorch Linear: x [.., in] · wᵀ (w [out, in]) + b [out].
    pub fn linear(&mut self, x: Var, w: Var, b: Option<Var>) -> Result<Var> {
        let sx = self.nodes[x.0].shape.clone();
        let rows: usize = sx[..sx.len() - 1].iter().product();
        let x2 = self.reshape(x, &[rows, sx[sx.len() - 1]])?;
        let mut y = self.matmul_opts(x2, w, true)?;
        if let Some(b) = b {
            y = self.add(y, b)?;
        }
        let mut out = sx[..sx.len() - 1].to_vec();
        out.push(self.nodes[w.0].shape[0]);
        self.reshape(y, &out)
    }

    /// Softmax over the last axis.
    pub fn softmax(&mut self, x: Var) -> Var {
        let s = self.nodes[x.0].shape.clone();
        let cols = *s.last().unwrap();
        let rows = self.numel(x) / cols;
        let y = self.be.alloc_out(rows * cols);
        self.be.softmax(&self.nodes[x.0].val, &y, rows, cols);
        self.push(y, s, Op::Softmax(x), &[x])
    }

    /// LayerNorm over the last axis with weight g and bias b ([cols]).
    pub fn layer_norm(&mut self, x: Var, g: Var, b: Var, eps: f32) -> Var {
        let s = self.nodes[x.0].shape.clone();
        let cols = *s.last().unwrap();
        let rows = self.numel(x) / cols;
        let (y, mean, rstd) = (self.be.alloc_out(rows * cols), self.be.alloc_out(rows), self.be.alloc_out(rows));
        self.be.layernorm(&self.nodes[x.0].val, &self.nodes[g.0].val, &self.nodes[b.0].val, &y, &mean, &rstd, rows, cols, eps);
        self.push(y, s, Op::LayerNorm { x, g, b, mean, rstd }, &[x, g, b])
    }

    pub fn reshape(&mut self, x: Var, shape: &[usize]) -> Result<Var> {
        ensure!(shape.iter().product::<usize>() == self.numel(x), "reshape {:?} → {shape:?}", self.nodes[x.0].shape);
        let val = self.nodes[x.0].val.clone();
        Ok(self.push_rc(val, shape.to_vec(), Op::Reshape(x), &[x]))
    }

    /// y = x.permute(perm)
    pub fn permute(&mut self, x: Var, perm: &[usize]) -> Result<Var> {
        let s = self.nodes[x.0].shape.clone();
        ensure!(perm.len() == s.len(), "permute {perm:?} of rank {}", s.len());
        let st = strides(&s);
        let out: Vec<usize> = perm.iter().map(|&p| s[p]).collect();
        let xs: Vec<usize> = perm.iter().map(|&p| st[p]).collect();
        let y = self.be.alloc_out(self.numel(x));
        self.be.copy_strided(&self.nodes[x.0].val, 0, &xs, &y, 0, &strides(&out), &out, false);
        Ok(self.push(y, out, Op::Permute(x, perm.to_vec()), &[x]))
    }

    /// x[.., start..end, ..] on `axis`
    pub fn slice(&mut self, x: Var, axis: usize, start: usize, end: usize) -> Result<Var> {
        let s = self.nodes[x.0].shape.clone();
        ensure!(axis < s.len() && start < end && end <= s[axis], "slice {start}..{end} of axis {axis} in {s:?}");
        let st = strides(&s);
        let mut out = s.clone();
        out[axis] = end - start;
        let y = self.be.alloc_out(out.iter().product());
        self.be.copy_strided(&self.nodes[x.0].val, start * st[axis], &st, &y, 0, &strides(&out), &out, false);
        Ok(self.push(y, out, Op::Slice { x, axis, start }, &[x]))
    }

    pub fn concat(&mut self, xs: &[Var], axis: usize) -> Result<Var> {
        let s0 = self.nodes[xs[0].0].shape.clone();
        let mut out = s0.clone();
        out[axis] = 0;
        for &x in xs {
            let s = &self.nodes[x.0].shape;
            ensure!(s.len() == s0.len() && (0..s.len()).all(|d| d == axis || s[d] == s0[d]), "concat shapes {s0:?} vs {s:?}");
            out[axis] += s[axis];
        }
        let ost = strides(&out);
        let y = self.be.alloc_out(out.iter().product());
        let mut off = 0;
        for &x in xs {
            let s = self.nodes[x.0].shape.clone();
            self.be.copy_strided(&self.nodes[x.0].val, 0, &strides(&s), &y, off * ost[axis], &ost, &s, false);
            off += s[axis];
        }
        Ok(self.push(y, out, Op::Concat(xs.to_vec(), axis), xs))
    }

    /// Sum over `axis`, which is removed from the shape.
    pub fn sum_axis(&mut self, x: Var, axis: usize) -> Result<Var> {
        let s = self.nodes[x.0].shape.clone();
        ensure!(axis < s.len(), "sum_axis {axis} of {s:?}");
        let mut keep = s.clone();
        keep[axis] = 1;
        let y = self.be.alloc_out(keep.iter().product());
        self.be.reduce_to(&self.nodes[x.0].val, &s, &y, &keep, false);
        let mut out = s.clone();
        out.remove(axis);
        Ok(self.push(y, out, Op::SumAxis { x, axis }, &[x]))
    }

    pub fn add_scalar(&mut self, x: Var, c: f32) -> Result<Var> {
        let k = self.input(&[c], &[1]);
        self.add(x, k)
    }

    /// min(max(x, lo), hi)
    pub fn clamp(&mut self, x: Var, lo: f32, hi: f32) -> Result<Var> {
        let (l, h) = (self.input(&[lo], &[1]), self.input(&[hi], &[1]));
        let y = self.binary(Binary::Max, x, l)?;
        self.binary(Binary::Min, y, h)
    }

    /// log(x / (1 − x)) with x clipped to [0, 1] and both terms to ≥ eps (DETR's inverse_sigmoid).
    pub fn inverse_sigmoid(&mut self, x: Var, eps: f32) -> Result<Var> {
        let x = self.clamp(x, 0.0, 1.0)?;
        let one = self.input(&[1.0], &[1]);
        let om = self.sub(one, x)?;
        let (a, b) = (self.clamp(x, eps, f32::MAX)?, self.clamp(om, eps, f32::MAX)?);
        let r = self.div(a, b)?;
        Ok(self.unary(Unary::Log, r))
    }

    /// Σ x (scaled), as a [1] tensor.
    pub fn sum_scaled(&mut self, x: Var, scale: f32) -> Var {
        let y = self.be.alloc_out(1);
        self.be.sum(&self.nodes[x.0].val, &y, scale, false);
        self.push(y, vec![1], Op::Sum { x, scale }, &[x])
    }

    pub fn sum(&mut self, x: Var) -> Var {
        self.sum_scaled(x, 1.0)
    }

    pub fn mean(&mut self, x: Var) -> Var {
        let n = self.numel(x) as f32;
        self.sum_scaled(x, 1.0 / n)
    }

    /// Cut the gradient: the same values, as a constant.
    pub fn detach(&mut self, x: Var) -> Var {
        let val = self.nodes[x.0].val.clone();
        let shape = self.nodes[x.0].shape.clone();
        self.leaf(val, &shape, false)
    }

    /// PyTorch Conv2d on NCHW: x [N,C,H,W], w [Cout, C/groups, kh, kw], b [Cout];
    /// pads [top, left, bottom, right].
    pub fn conv2d(&mut self, x: Var, w: Var, b: Option<Var>, stride: [usize; 2], pads: [usize; 4], groups: usize) -> Result<Var> {
        let (xs, ws) = (self.nodes[x.0].shape.clone(), self.nodes[w.0].shape.clone());
        ensure!(xs.len() == 4 && ws.len() == 4, "conv2d wants NCHW x and OIHW w: {xs:?}, {ws:?}");
        let (n, c, h, wd) = (xs[0], xs[1], xs[2], xs[3]);
        let (cout, cg, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
        ensure!(c % groups == 0 && cout % groups == 0 && cg == c / groups, "conv2d groups {groups}: x {xs:?}, w {ws:?}");
        let win = Win::new(c, h, wd, [kh, kw], stride, pads, false);
        let (ohw, cgkk, coutg) = (win.oh * win.ow, cg * kh * kw, cout / groups);
        let y = self.be.alloc_out(n * cout * ohw);
        // Y[img][g] = W[g] · im2col(x[img][g]), the unfolding done on the fly
        let wg = Win { planes: cg, ..win };
        if groups == 1 {
            let g = Gemm { m: cout, n: ohw, k: cgkk, ta: false, tb: false, batch: n, sa: 0, sb: c * h * wd, sc: cout * ohw, alpha: 1.0, beta: 0.0, oa: 0, ob: 0, oc: 0 };
            self.be.gemm_im2col(&self.nodes[w.0].val, &self.nodes[x.0].val, &y, &g, &wg);
        } else {
            for img in 0..n {
                let g = Gemm { m: coutg, n: ohw, k: cgkk, ta: false, tb: false, batch: groups, sa: coutg * cgkk, sb: cg * h * wd, sc: coutg * ohw, alpha: 1.0, beta: 0.0, oa: 0, ob: img * c * h * wd, oc: img * cout * ohw };
                self.be.gemm_im2col(&self.nodes[w.0].val, &self.nodes[x.0].val, &y, &g, &wg);
            }
        }
        let y = match b {
            Some(bv) => {
                ensure!(self.nodes[bv.0].shape == [cout], "conv2d bias {:?} for {cout} channels", self.nodes[bv.0].shape);
                let (bc, _) = Bcast::new(&[n, cout, win.oh, win.ow], &[cout, 1, 1])?;
                let yb = self.be.alloc_out(bc.n);
                self.be.binary(Binary::Add, &y, &self.nodes[bv.0].val, &yb, &bc);
                yb
            }
            None => y,
        };
        let parents: Vec<Var> = [Some(x), Some(w), b].into_iter().flatten().collect();
        Ok(self.push(y, vec![n, cout, win.oh, win.ow], Op::Conv { x, w, b, win, n, cout, groups }, &parents))
    }

    /// BatchNorm2d. `train`: normalise by batch statistics and update `run`;
    /// otherwise by `run` (frozen statistics).
    pub fn batch_norm2d(&mut self, x: Var, g: Var, b: Var, run: &BnRunning<B>, train: bool) -> Result<Var> {
        let s = self.nodes[x.0].shape.clone();
        ensure!(s.len() == 4, "batch_norm2d wants NCHW: {s:?}");
        let (n, c, hw) = (s[0], s[1], s[2] * s[3]);
        let (mean, rstd) = if train {
            let (mean, rstd) = (self.be.alloc_out(c), self.be.alloc_out(c));
            self.be.bn_stats(&self.nodes[x.0].val, &mean, &rstd, n, c, hw, run.eps);
            let m = (n * hw) as f32;
            self.be.bn_running(&mean, &rstd, &run.mean, &run.var, c, run.momentum, run.eps, m / (m - 1.0).max(1.0));
            (mean, rstd)
        } else {
            let (rm, rv) = (self.be.download(&run.mean), self.be.download(&run.var));
            let rs: Vec<f32> = rv.iter().map(|v| 1.0 / (v + run.eps).sqrt()).collect();
            (self.be.upload(&rm), self.be.upload(&rs))
        };
        let y = self.be.alloc_out(n * c * hw);
        self.be.bn_apply(&self.nodes[x.0].val, &mean, &rstd, &self.nodes[g.0].val, &self.nodes[b.0].val, &y, n, c, hw);
        Ok(self.push(y, s, Op::BatchNorm { x, g, b, mean, rstd, train }, &[x, g, b]))
    }

    /// pads [top, left, bottom, right]
    pub fn max_pool2d(&mut self, x: Var, k: [usize; 2], stride: [usize; 2], pads: [usize; 4], ceil: bool) -> Result<Var> {
        let s = self.nodes[x.0].shape.clone();
        ensure!(s.len() == 4, "max_pool2d wants NCHW: {s:?}");
        let win = Win::new(s[0] * s[1], s[2], s[3], k, stride, pads, ceil);
        let (y, idx) = (self.be.alloc_out(win.planes * win.oh * win.ow), self.be.alloc_out(win.planes * win.oh * win.ow));
        self.be.maxpool(&self.nodes[x.0].val, &y, &idx, &win);
        Ok(self.push(y, vec![s[0], s[1], win.oh, win.ow], Op::MaxPool { x, idx, win }, &[x]))
    }

    pub fn avg_pool2d(&mut self, x: Var, k: [usize; 2], stride: [usize; 2], pads: [usize; 4], ceil: bool, count_include_pad: bool) -> Result<Var> {
        let s = self.nodes[x.0].shape.clone();
        ensure!(s.len() == 4, "avg_pool2d wants NCHW: {s:?}");
        let win = Win::new(s[0] * s[1], s[2], s[3], k, stride, pads, ceil);
        let y = self.be.alloc_out(win.planes * win.oh * win.ow);
        self.be.avgpool(&self.nodes[x.0].val, &y, &win, count_include_pad);
        Ok(self.push(y, vec![s[0], s[1], win.oh, win.ow], Op::AvgPool { x, win, cip: count_include_pad }, &[x]))
    }

    /// Nearest-neighbour upsampling by integer factors (F.interpolate, mode="nearest").
    pub fn upsample_nearest(&mut self, x: Var, fy: usize, fx: usize) -> Result<Var> {
        let s = self.nodes[x.0].shape.clone();
        ensure!(s.len() == 4, "upsample wants NCHW: {s:?}");
        let y = self.be.alloc_out(self.numel(x) * fy * fx);
        self.be.upsample(&self.nodes[x.0].val, &y, s[0] * s[1], s[2], s[3], fy, fx);
        Ok(self.push(y, vec![s[0], s[1], s[2] * fy, s[3] * fx], Op::Upsample { x, fy, fx }, &[x]))
    }

    /// F.grid_sample(x [N,C,H,W], grid [N,Ho,Wo,2]), bilinear, zeros, align_corners = false.
    pub fn grid_sample(&mut self, x: Var, grid: Var) -> Result<Var> {
        let (xs, gs) = (self.nodes[x.0].shape.clone(), self.nodes[grid.0].shape.clone());
        ensure!(xs.len() == 4 && gs.len() == 4 && gs[3] == 2 && gs[0] == xs[0], "grid_sample {xs:?} with grid {gs:?}");
        let y = self.be.alloc_out(xs[0] * xs[1] * gs[1] * gs[2]);
        self.be.grid_sample(&self.nodes[x.0].val, &self.nodes[grid.0].val, &y, xs[0], xs[1], xs[2], xs[3], gs[1], gs[2]);
        Ok(self.push(y, vec![xs[0], xs[1], gs[1], gs[2]], Op::GridSample { x, grid }, &[x, grid]))
    }

    /// x [B,N,C] → [B,K,C]: rows idx[b][k] of batch b (TopK query selection).
    pub fn gather_rows(&mut self, x: Var, idx: &[usize], k: usize) -> Result<Var> {
        let s = self.nodes[x.0].shape.clone();
        ensure!(s.len() == 3 && idx.len() == s[0] * k, "gather_rows from {s:?} with {} indices (k {k})", idx.len());
        ensure!(idx.iter().all(|&i| i < s[1]), "gather_rows index out of range (N {})", s[1]);
        let (b, n, c) = (s[0], s[1], s[2]);
        let iv: Vec<f32> = idx.iter().map(|&i| i as f32).collect();
        let ib = self.be.upload(&iv);
        let y = self.be.alloc_out(b * k * c);
        self.be.gather_rows(&self.nodes[x.0].val, &ib, &y, b, n, k, c);
        Ok(self.push(y, vec![b, k, c], Op::GatherRows { x, idx: ib, b, n, k, c }, &[x]))
    }

    /// Zero padding of one axis (`before` / `after` elements), as `F.pad` with zeros. Built from
    /// `concat` with constant zero blocks, so its backward is concat's slice. Not the same as a
    /// pooling op's own padding, which pads with −∞.
    pub fn pad(&mut self, x: Var, axis: usize, before: usize, after: usize) -> Result<Var> {
        if before == 0 && after == 0 {
            return Ok(x);
        }
        let s = self.nodes[x.0].shape.clone();
        ensure!(axis < s.len(), "pad axis {axis} of {s:?}");
        let zeros = |n: usize, t: &mut Self| {
            let mut z = s.clone();
            z[axis] = n;
            t.input(&vec![0.0; z.iter().product()], &z)
        };
        let mut parts = vec![];
        if before > 0 {
            parts.push(zeros(before, self));
        }
        parts.push(x);
        if after > 0 {
            parts.push(zeros(after, self));
        }
        self.concat(&parts, axis)
    }

    /// `F.pad(x, (left, right, top, bottom))` on an NCHW tensor, zero fill.
    pub fn pad2d(&mut self, x: Var, left: usize, right: usize, top: usize, bottom: usize) -> Result<Var> {
        ensure!(self.nodes[x.0].shape.len() == 4, "pad2d wants NCHW: {:?}", self.nodes[x.0].shape);
        let x = self.pad(x, 3, left, right)?;
        self.pad(x, 2, top, bottom)
    }

    /// Per-row gather along the last axis: x [.., K], `idx` holds `k` column indices for each of
    /// the M = Π(leading dims) rows, row-major → [.., k]. `torch.gather(x, -1, idx)`; the
    /// gradient scatters back (repeated indices accumulate).
    pub fn gather_last(&mut self, x: Var, idx: &[usize], k: usize) -> Result<Var> {
        let s = self.nodes[x.0].shape.clone();
        ensure!(!s.is_empty(), "gather_last of a scalar");
        let last = *s.last().unwrap();
        let m: usize = s[..s.len() - 1].iter().product();
        ensure!(idx.len() == m * k, "gather_last: {} indices for {m} rows × {k}", idx.len());
        let x3 = self.reshape(x, &[m, last, 1])?;
        let y = self.gather_rows(x3, idx, k)?;
        let mut out = s[..s.len() - 1].to_vec();
        out.push(k);
        self.reshape(y, &out)
    }

    /// The `k` largest values of each row along the last axis, in descending order, with their
    /// indices (`torch.topk(x, k, dim=-1)`). Selection happens on the host; the values stay on
    /// the tape, so the gradient reaches exactly the selected elements.
    pub fn topk_last(&mut self, x: Var, k: usize) -> Result<(Var, Vec<usize>)> {
        let s = self.nodes[x.0].shape.clone();
        let last = *s.last().ok_or_else(|| anyhow::anyhow!("topk of a scalar"))?;
        ensure!(k >= 1 && k <= last, "topk k {k} of last axis {last}");
        let vals = self.value(x);
        let mut idx = Vec::with_capacity(vals.len() / last * k);
        let mut order: Vec<usize> = Vec::with_capacity(last);
        for row in vals.chunks(last) {
            order.clear();
            order.extend(0..last);
            // descending; the lower index first on ties (stable sort)
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            idx.extend_from_slice(&order[..k]);
        }
        let y = self.gather_last(x, &idx, k)?;
        Ok((y, idx))
    }

    // ----------------------------------------------------------- backward ---

    /// Keep `v`'s value and gradient through `backward` (by default only leaves and
    /// the loss keep theirs: every other node frees both once its backward ran).
    pub fn keep(&mut self, v: Var) {
        self.keep.insert(v.0);
    }

    fn parents(&self, i: usize) -> Vec<Var> {
        match &self.nodes[i].op {
            Op::Leaf => vec![],
            Op::Unary(_, x) | Op::Softmax(x) | Op::Reshape(x) | Op::Permute(x, _) => vec![*x],
            Op::Slice { x, .. } | Op::Sum { x, .. } | Op::SumAxis { x, .. } | Op::MaxPool { x, .. } | Op::AvgPool { x, .. } | Op::Upsample { x, .. } | Op::GatherRows { x, .. } => vec![*x],
            Op::Binary(_, a, b, _) | Op::MatMul { a, b, .. } => vec![*a, *b],
            Op::LayerNorm { x, g, b, .. } | Op::BatchNorm { x, g, b, .. } => vec![*x, *g, *b],
            Op::Concat(xs, _) => xs.clone(),
            Op::Conv { x, w, b, .. } => [Some(*x), Some(*w), *b].into_iter().flatten().collect(),
            Op::GridSample { x, grid } => vec![*x, *grid],
        }
    }

    /// Gradients of the scalar `loss` w.r.t. every node that needs one. Walks the
    /// tape in reverse; a node's gradient is created when its first consumer
    /// contributes, and a non-leaf node's value and gradient are freed as soon
    /// as its own backward has run (every consumer has already run by then).
    pub fn backward(&mut self, loss: Var) -> Result<()> {
        if self.numel(loss) != 1 {
            bail!("backward from a non-scalar {:?}", self.nodes[loss.0].shape);
        }
        if !self.nodes[loss.0].req {
            bail!("loss does not depend on anything that needs a gradient");
        }
        let g = self.be.alloc(1);
        self.be.fill(&g, 1.0);
        self.nodes[loss.0].grad = Some(g);
        self.keep.insert(loss.0);
        for i in (0..=loss.0).rev() {
            if !self.nodes[i].req || self.nodes[i].grad.is_none() {
                continue;
            }
            for p in self.parents(i) {
                if self.nodes[p.0].req && self.nodes[p.0].grad.is_none() {
                    let n = self.numel(p);
                    self.nodes[p.0].grad = Some(self.be.alloc(n));
                }
            }
            self.backward_node(i)?;
            if !matches!(self.nodes[i].op, Op::Leaf) && !self.keep.contains(&i) {
                self.nodes[i].grad = None;
                let freed = self.freed.get_or_insert_with(|| Rc::new(self.be.alloc(1))).clone();
                self.nodes[i].val = freed;
            }
        }
        Ok(())
    }

    fn backward_node(&self, i: usize) -> Result<()> {
        let be = self.be;
        let node = &self.nodes[i];
        let dy = node.grad.as_ref().unwrap();
        let val = |v: Var| -> &B::Buf { &self.nodes[v.0].val };
        let grad = |v: Var| -> Option<&B::Buf> { if self.nodes[v.0].req { self.nodes[v.0].grad.as_ref() } else { None } };
        match &node.op {
            Op::Leaf => {}
            Op::Unary(op, x) => {
                if let Some(gx) = grad(*x) {
                    be.unary_bwd(*op, val(*x), &node.val, dy, gx, true);
                }
            }
            Op::Binary(op, a, b, bc) => {
                let out = &node.shape;
                for (which, v) in [(0u8, *a), (1u8, *b)] {
                    if let Some(gv) = grad(v) {
                        let t = be.alloc_out(bc.n);
                        be.binary_grad(*op, which, val(*a), val(*b), dy, &t, bc);
                        be.reduce_to(&t, out, gv, &self.nodes[v.0].shape, true);
                    }
                }
            }
            Op::MatMul { a, b, g, shared_b } => {
                // C = A·op(B): dA = dC·op(B)ᵀ, dB = Aᵀ·dC (or its transpose when B is stored N×K)
                if let Some(ga) = grad(*a) {
                    let gd = Gemm { m: g.m, n: g.k, k: g.n, ta: false, tb: !g.tb, batch: g.batch, sa: g.m * g.n, sb: g.sb, sc: g.m * g.k, alpha: 1.0, beta: 1.0, oa: 0, ob: 0, oc: 0 };
                    be.gemm(dy, val(*b), ga, &gd);
                }
                if let Some(gb) = grad(*b) {
                    let (batch, k_red) = if *shared_b { (1, g.batch * g.m) } else { (g.batch, g.m) };
                    let (bs_a, bs_c) = if *shared_b { (0, 0) } else { (g.m * g.k, g.m * g.n) };
                    if g.tb {
                        // B stored N×K: dB = dCᵀ·A  (N×K)
                        let gd = Gemm { m: g.n, n: g.k, k: k_red, ta: true, tb: false, batch, sa: bs_c, sb: bs_a, sc: g.n * g.k, alpha: 1.0, beta: 1.0, oa: 0, ob: 0, oc: 0 };
                        be.gemm(dy, val(*a), gb, &gd);
                    } else {
                        // B stored K×N: dB = Aᵀ·dC  (K×N)
                        let gd = Gemm { m: g.k, n: g.n, k: k_red, ta: true, tb: false, batch, sa: bs_a, sb: bs_c, sc: g.k * g.n, alpha: 1.0, beta: 1.0, oa: 0, ob: 0, oc: 0 };
                        be.gemm(val(*a), dy, gb, &gd);
                    }
                }
            }
            Op::Softmax(x) => {
                if let Some(gx) = grad(*x) {
                    let cols = *node.shape.last().unwrap();
                    be.softmax_bwd(&node.val, dy, gx, self.numel(Var(i)) / cols, cols, true);
                }
            }
            Op::LayerNorm { x, g, b, mean, rstd } => {
                let cols = *node.shape.last().unwrap();
                let rows = self.numel(Var(i)) / cols;
                if let Some(gx) = grad(*x) {
                    be.layernorm_bwd(val(*x), val(*g), mean, rstd, dy, gx, rows, cols, true);
                }
                if grad(*g).is_some() || grad(*b).is_some() {
                    let (tg, tb) = (be.alloc_out(cols), be.alloc_out(cols));
                    be.layernorm_wgrad(val(*x), mean, rstd, dy, &tg, &tb, rows, cols, false);
                    if let Some(gg) = grad(*g) {
                        be.axpby(&tg, gg, 1.0, 1.0);
                    }
                    if let Some(gb) = grad(*b) {
                        be.axpby(&tb, gb, 1.0, 1.0);
                    }
                }
            }
            Op::Reshape(x) => {
                if let Some(gx) = grad(*x) {
                    be.axpby(dy, gx, 1.0, 1.0);
                }
            }
            Op::Permute(x, perm) => {
                if let Some(gx) = grad(*x) {
                    // walk the output index space: dy is contiguous there, gx through the permuted strides
                    let xst = strides(&self.nodes[x.0].shape);
                    let gs: Vec<usize> = perm.iter().map(|&p| xst[p]).collect();
                    be.copy_strided(dy, 0, &strides(&node.shape), gx, 0, &gs, &node.shape, true);
                }
            }
            Op::Slice { x, axis, start } => {
                if let Some(gx) = grad(*x) {
                    let xst = strides(&self.nodes[x.0].shape);
                    be.copy_strided(dy, 0, &strides(&node.shape), gx, start * xst[*axis], &xst, &node.shape, true);
                }
            }
            Op::Concat(xs, axis) => {
                let ost = strides(&node.shape);
                let mut off = 0;
                for &x in xs {
                    let s = self.nodes[x.0].shape.clone();
                    if let Some(gx) = grad(x) {
                        be.copy_strided(dy, off * ost[*axis], &ost, gx, 0, &strides(&s), &s, true);
                    }
                    off += s[*axis];
                }
            }
            Op::SumAxis { x, axis } => {
                if let Some(gx) = grad(*x) {
                    // broadcast dy back along the summed axis: stride 0 there
                    let xs = self.nodes[x.0].shape.clone();
                    let mut ds = strides(&node.shape);
                    ds.insert(*axis, 0);
                    be.copy_strided(dy, 0, &ds, gx, 0, &strides(&xs), &xs, true);
                }
            }
            Op::Sum { x, scale } => {
                if let Some(gx) = grad(*x) {
                    be.bcast_scalar(dy, gx, *scale, true);
                }
            }

            Op::Conv { x, w, b, win, n, cout, groups } => {
                let (c, ohw) = (win.planes, win.oh * win.ow);
                let (cg, coutg) = (c / groups, cout / groups);
                let cgkk = cg * win.kh * win.kw;
                if let Some(bv) = b {
                    if let Some(gb) = grad(*bv) {
                        be.channel_sum(dy, gb, *n, *cout, ohw, true);
                    }
                }
                let (gx, gw) = (grad(*x), grad(*w));
                let wg = Win { planes: cg, ..*win };
                if let Some(gw) = gw {
                    // dW[g] += dY[img][g] · im2col(x[img][g])ᵀ
                    for img in 0..*n {
                        let g = Gemm { m: coutg, n: cgkk, k: ohw, ta: false, tb: true, batch: *groups, sa: coutg * ohw, sb: cg * win.h * win.w, sc: coutg * cgkk, alpha: 1.0, beta: 1.0, oa: img * cout * ohw, ob: img * c * win.h * win.w, oc: 0 };
                        be.gemm_im2col(dy, val(*x), gw, &g, &wg);
                    }
                }
                if gx.is_some() {
                    let col = be.alloc_out(c * win.kh * win.kw * ohw);
                    for img in 0..*n {
                        if let Some(gx) = gx {
                            // dcol[g] = W[g]ᵀ · dY[img][g]; dx[img] += col2im(dcol)
                            let g = Gemm { m: cgkk, n: ohw, k: coutg, ta: true, tb: false, batch: *groups, sa: coutg * cgkk, sb: coutg * ohw, sc: cgkk * ohw, alpha: 1.0, beta: 0.0, oa: 0, ob: img * cout * ohw, oc: 0 };
                            be.gemm(val(*w), dy, &col, &g);
                            be.col2im(&col, gx, img * c * win.h * win.w, win, true);
                        }
                    }
                }
            }
            Op::BatchNorm { x, g, b, mean, rstd, train } => {
                let s = &self.nodes[x.0].shape;
                let (n, c, hw) = (s[0], s[1], s[2] * s[3]);
                let (dg, db) = (be.alloc_out(c), be.alloc_out(c));
                be.bn_wgrad(val(*x), mean, rstd, dy, &dg, &db, n, c, hw);
                if let Some(gx) = grad(*x) {
                    if *train {
                        be.bn_bwd(val(*x), mean, rstd, val(*g), &dg, &db, dy, gx, n, c, hw, true);
                    } else {
                        // frozen statistics: dx = g·rstd·dy (the batch terms vanish)
                        let z = be.alloc(c);
                        be.bn_bwd(val(*x), mean, rstd, val(*g), &z, &z, dy, gx, n, c, hw, true);
                    }
                }
                if let Some(gg) = grad(*g) {
                    be.axpby(&dg, gg, 1.0, 1.0);
                }
                if let Some(gb) = grad(*b) {
                    be.axpby(&db, gb, 1.0, 1.0);
                }
            }
            Op::MaxPool { x, idx, win } => {
                if let Some(gx) = grad(*x) {
                    be.maxpool_bwd(dy, idx, gx, win, true);
                }
            }
            Op::AvgPool { x, win, cip } => {
                if let Some(gx) = grad(*x) {
                    be.avgpool_bwd(dy, gx, win, *cip, true);
                }
            }
            Op::Upsample { x, fy, fx } => {
                if let Some(gx) = grad(*x) {
                    let s = &self.nodes[x.0].shape;
                    be.upsample_bwd(dy, gx, s[0] * s[1], s[2], s[3], *fy, *fx, true);
                }
            }
            Op::GridSample { x, grid } => {
                let (xs, gs) = (&self.nodes[x.0].shape, &self.nodes[grid.0].shape);
                let (gx, gg) = (grad(*x), grad(*grid));
                if gx.is_some() || gg.is_some() {
                    be.grid_sample_bwd(val(*x), val(*grid), dy, gx, gg, xs[0], xs[1], xs[2], xs[3], gs[1], gs[2], true);
                }
            }
            Op::GatherRows { x, idx, b, n, k, c } => {
                if let Some(gx) = grad(*x) {
                    be.gather_rows_bwd(dy, idx, gx, *b, *n, *k, *c);
                }
            }
        }
        Ok(())
    }
}

/// AdamW over `Param`s (PyTorch `torch.optim.AdamW` semantics).
pub struct AdamW {
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub wd: f32,
    pub step: u32,
}

impl Default for AdamW {
    fn default() -> Self {
        AdamW { lr: 1e-3, b1: 0.9, b2: 0.999, eps: 1e-8, wd: 1e-2, step: 0 }
    }
}

impl AdamW {
    /// Start a step (bias correction counts steps from 1).
    pub fn begin(&mut self) {
        self.step += 1;
    }

    pub fn update<B: Backend>(&self, be: &B, p: &Param<B>, grad: &B::Buf, wd: f32) {
        be.adamw(&p.val, grad, &p.m, &p.v, self.lr, self.b1, self.b2, self.eps, wd, self.step);
    }
}

#[cfg(test)]
mod tests {
    use crate::check::gradcheck;

    fn ramp(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.37 + seed).sin() * 1.3).collect()
    }

    /// weighted sum so every output element has a distinct gradient
    fn weighted(t: &mut crate::Tape<crate::cpu::Cpu>, y: crate::Var) -> crate::Var {
        let n: usize = t.shape(y).iter().product();
        let w = t.input(&ramp(n, 0.5), t.shape(y).to_vec().as_slice());
        let p = t.mul(y, w).unwrap();
        t.sum(p)
    }

    #[test]
    fn pad2d_matches_f_pad_and_differentiates() {
        let x = ramp(2 * 3 * 4 * 5, 0.1);
        let be = crate::cpu::Cpu;
        let mut t = crate::Tape::new(&be);
        let v = t.input(&x, &[2, 3, 4, 5]);
        let y = t.pad2d(v, 0, 1, 2, 1).unwrap();
        assert_eq!(t.shape(y), &[2, 3, 7, 6]);
        let out = t.value(y);
        // interior equals the input, borders are zero (F.pad(x, (0, 1, 2, 1)))
        for n in 0..2 { for c in 0..3 { for h in 0..7 { for w in 0..6 {
            let got = out[((n * 3 + c) * 7 + h) * 6 + w];
            let want = if (2..6).contains(&h) && w < 5 { x[((n * 3 + c) * 4 + h - 2) * 5 + w] } else { 0.0 };
            assert_eq!(got, want);
        }}}}
        let m = gradcheck(&[(x, vec![2, 3, 4, 5])], 1e-2, |t, v| { let y = t.pad2d(v[0], 0, 1, 2, 1).unwrap(); weighted(t, y) });
        assert!(m.err < 2e-3, "{m:?}");
    }

    #[test]
    fn gather_last_accumulates_repeated_indices() {
        let x = ramp(3 * 4 * 6, 0.2);
        let idx = vec![0, 5, 5, 2, 1, 1, 3, 0, 4, 4, 4, 2, 5, 0, 1, 1, 2, 3, 3, 3, 0, 0, 0, 5];
        let be = crate::cpu::Cpu;
        let mut t = crate::Tape::new(&be);
        let v = t.input(&x, &[3, 4, 6]);
        let y = t.gather_last(v, &idx, 2).unwrap();
        assert_eq!(t.shape(y), &[3, 4, 2]);
        let out = t.value(y);
        for r in 0..12 { for j in 0..2 { assert_eq!(out[r * 2 + j], x[r * 6 + idx[r * 2 + j]]); } }
        let m = gradcheck(&[(x, vec![3, 4, 6])], 1e-2, |t, v| { let y = t.gather_last(v[0], &idx, 2).unwrap(); weighted(t, y) });
        assert!(m.err < 2e-3, "{m:?}");
    }

    #[test]
    fn topk_last_selects_descending_and_routes_gradient() {
        let x: Vec<f32> = (0..5 * 33).map(|i| ((i * 7919) % 165) as f32 * 0.05).collect();
        let be = crate::cpu::Cpu;
        let mut t = crate::Tape::new(&be);
        let v = t.input(&x, &[5, 33]);
        let (y, idx) = t.topk_last(v, 4).unwrap();
        let out = t.value(y);
        for r in 0..5 {
            let mut row: Vec<f32> = x[r * 33..(r + 1) * 33].to_vec();
            row.sort_by(|a, b| b.total_cmp(a));
            assert_eq!(&out[r * 4..(r + 1) * 4], &row[..4]);
            for j in 0..4 { assert_eq!(x[r * 33 + idx[r * 4 + j]], out[r * 4 + j]); }
        }
        let m = gradcheck(&[(x, vec![5, 33])], 1e-2, |t, v| { let (y, _) = t.topk_last(v[0], 4).unwrap(); weighted(t, y) });
        assert!(m.err < 2e-3, "{m:?}");
    }
}
