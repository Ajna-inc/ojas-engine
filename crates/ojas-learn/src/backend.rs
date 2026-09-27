//! The primitive set every training backend implements. The tape is written
//! against this trait only; the CPU backend is the reference each device
//! backend is checked against, primitive by primitive.
//!
//! All tensors are contiguous row-major f32. Output buffers are passed by
//! shared reference (device memory, or interior mutability on the CPU).
//! `accum` adds into the output instead of overwriting it.

pub const MAX_RANK: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unary {
    Relu = 0,
    Sigmoid = 1,
    Silu = 2,
    Tanh = 3,
    /// 0.5·x·(1+erf(x/√2))
    Gelu = 4,
    Exp = 5,
    Log = 6,
    Neg = 7,
    Sqrt = 8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    /// ties take a's gradient
    Max = 4,
    Min = 5,
}

/// A broadcast binary op laid out for the kernels: the output dims and each
/// operand's strides over them (0 on broadcast axes), right-aligned to 6 axes.
#[derive(Clone, Copy, Debug)]
pub struct Bcast {
    pub dims: [usize; MAX_RANK],
    pub sa: [usize; MAX_RANK],
    pub sb: [usize; MAX_RANK],
    pub n: usize,
}

/// Right-align `shape` to 6 axes, padding with 1.
pub fn pad6(shape: &[usize]) -> [usize; MAX_RANK] {
    assert!(shape.len() <= MAX_RANK, "rank {} > {MAX_RANK}", shape.len());
    let mut d = [1; MAX_RANK];
    d[MAX_RANK - shape.len()..].copy_from_slice(shape);
    d
}

pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// NumPy broadcasting of two shapes.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> anyhow::Result<Vec<usize>> {
    let r = a.len().max(b.len());
    let (pa, pb) = (pad_to(a, r), pad_to(b, r));
    (0..r)
        .map(|i| match (pa[i], pb[i]) {
            (x, y) if x == y => Ok(x),
            (1, y) => Ok(y),
            (x, 1) => Ok(x),
            (x, y) => anyhow::bail!("cannot broadcast {a:?} with {b:?} (axis {i}: {x} vs {y})"),
        })
        .collect()
}

fn pad_to(s: &[usize], r: usize) -> Vec<usize> {
    let mut v = vec![1; r - s.len()];
    v.extend_from_slice(s);
    v
}

impl Bcast {
    pub fn new(a: &[usize], b: &[usize]) -> anyhow::Result<(Self, Vec<usize>)> {
        let out = broadcast_shape(a, b)?;
        let dims = pad6(&out);
        let st = |s: &[usize]| {
            let p = pad6(s);
            let full = strides(&p);
            let mut r = [0; MAX_RANK];
            for d in 0..MAX_RANK {
                r[d] = if p[d] == 1 && dims[d] != 1 { 0 } else { full[d] };
            }
            r
        };
        Ok((Bcast { dims, sa: st(a), sb: st(b), n: out.iter().product() }, out))
    }
}

/// Batched row-major GEMM: C = alpha·op(A)·op(B) + beta·C.
#[derive(Clone, Copy, Debug)]
pub struct Gemm {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// A stored K×M
    pub ta: bool,
    /// B stored N×K
    pub tb: bool,
    pub batch: usize,
    /// per-batch element strides (0 = the same matrix for every batch)
    pub sa: usize,
    pub sb: usize,
    pub sc: usize,
    pub alpha: f32,
    pub beta: f32,
    /// element offsets of A, B, C in their buffers
    pub oa: usize,
    pub ob: usize,
    pub oc: usize,
}

impl Gemm {
    /// One C = op(A)·op(B), no batching, no offsets.
    pub fn plain(m: usize, n: usize, k: usize, ta: bool, tb: bool) -> Self {
        Gemm { m, n, k, ta, tb, batch: 1, sa: 0, sb: 0, sc: 0, alpha: 1.0, beta: 0.0, oa: 0, ob: 0, oc: 0 }
    }
}

/// A sliding-window geometry over `planes` H×W images (conv, pooling):
/// kernel kh×kw, stride, top/left padding, output OH×OW.
#[derive(Clone, Copy, Debug)]
pub struct Win {
    pub planes: usize,
    pub h: usize,
    pub w: usize,
    pub oh: usize,
    pub ow: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub pt: usize,
    pub pl: usize,
}

impl Win {
    /// Output extent for pads [top, left, bottom, right] (floor mode, or ceil
    /// mode where the last window must start inside the image or left padding).
    pub fn new(planes: usize, h: usize, w: usize, k: [usize; 2], s: [usize; 2], pads: [usize; 4], ceil: bool) -> Self {
        let ext = |i: usize, k: usize, s: usize, a: usize, b: usize| {
            let span = i + a + b - k;
            let mut o = if ceil { span.div_ceil(s) + 1 } else { span / s + 1 };
            // PyTorch: the last pooling window must start inside the input or left padding
            if ceil && (o - 1) * s >= i + a {
                o -= 1;
            }
            o
        };
        Win {
            planes,
            h,
            w,
            oh: ext(h, k[0], s[0], pads[0], pads[2]),
            ow: ext(w, k[1], s[1], pads[1], pads[3]),
            kh: k[0],
            kw: k[1],
            sh: s[0],
            sw: s[1],
            pt: pads[0],
            pl: pads[1],
        }
    }
}

pub trait Backend {
    type Buf;

    /// Zero-filled.
    fn alloc(&self, n: usize) -> Self::Buf;
    /// Unspecified contents: for outputs the next primitive overwrites entirely.
    fn alloc_out(&self, n: usize) -> Self::Buf {
        self.alloc(n)
    }
    fn upload(&self, v: &[f32]) -> Self::Buf;
    fn download(&self, b: &Self::Buf) -> Vec<f32>;
    fn len(&self, b: &Self::Buf) -> usize;

    fn fill(&self, y: &Self::Buf, v: f32);
    /// y = a·x + b·y
    fn axpby(&self, x: &Self::Buf, y: &Self::Buf, a: f32, b: f32);
    fn unary(&self, op: Unary, x: &Self::Buf, y: &Self::Buf);
    /// dx (+)= f'(x)·dy (y = f(x) from the forward)
    fn unary_bwd(&self, op: Unary, x: &Self::Buf, y: &Self::Buf, dy: &Self::Buf, dx: &Self::Buf, accum: bool);
    fn binary(&self, op: Binary, a: &Self::Buf, b: &Self::Buf, y: &Self::Buf, bc: &Bcast);
    /// t = dy·∂(a op b)/∂a (which = 0) or ∂b (which = 1), over the full output shape
    fn binary_grad(&self, op: Binary, which: u8, a: &Self::Buf, b: &Self::Buf, dy: &Self::Buf, t: &Self::Buf, bc: &Bcast);
    /// g (+)= t summed over the axes where `target` (right-aligned) is 1 and `full` is not
    fn reduce_to(&self, t: &Self::Buf, full: &[usize], g: &Self::Buf, target: &[usize], accum: bool);
    /// y[yoff + Σc·ys] (+)= x[xoff + Σc·xs] over `dims`
    #[allow(clippy::too_many_arguments)]
    fn copy_strided(&self, x: &Self::Buf, xoff: usize, xs: &[usize], y: &Self::Buf, yoff: usize, ys: &[usize], dims: &[usize], accum: bool);
    fn gemm(&self, a: &Self::Buf, b: &Self::Buf, c: &Self::Buf, g: &Gemm);
    /// `gemm` whose B is the im2col matrix of the image(s) in `x` (batch z at
    /// element offset g.ob + z·g.sb, `win.planes` channels): B(k, n) = col(k, n),
    /// or col(n, k) with g.tb. No unfolded copy needs to exist.
    fn gemm_im2col(&self, a: &Self::Buf, x: &Self::Buf, c: &Self::Buf, g: &Gemm, win: &Win) {
        // reference: unfold each batch, then an ordinary GEMM
        let rows = win.planes * win.kh * win.kw;
        let col = self.alloc(rows * win.oh * win.ow);
        for z in 0..g.batch {
            self.im2col(x, g.ob + z * g.sb, &col, win);
            let one = Gemm { batch: 1, sa: 0, sb: 0, sc: 0, oa: g.oa + z * g.sa, ob: 0, oc: g.oc + z * g.sc, ..*g };
            self.gemm(a, &col, c, &one);
        }
    }
    fn softmax(&self, x: &Self::Buf, y: &Self::Buf, rows: usize, cols: usize);
    fn softmax_bwd(&self, y: &Self::Buf, dy: &Self::Buf, dx: &Self::Buf, rows: usize, cols: usize, accum: bool);
    #[allow(clippy::too_many_arguments)]
    fn layernorm(&self, x: &Self::Buf, g: &Self::Buf, b: &Self::Buf, y: &Self::Buf, mean: &Self::Buf, rstd: &Self::Buf, rows: usize, cols: usize, eps: f32);
    #[allow(clippy::too_many_arguments)]
    fn layernorm_bwd(&self, x: &Self::Buf, g: &Self::Buf, mean: &Self::Buf, rstd: &Self::Buf, dy: &Self::Buf, dx: &Self::Buf, rows: usize, cols: usize, accum: bool);
    #[allow(clippy::too_many_arguments)]
    fn layernorm_wgrad(&self, x: &Self::Buf, mean: &Self::Buf, rstd: &Self::Buf, dy: &Self::Buf, dg: &Self::Buf, db: &Self::Buf, rows: usize, cols: usize, accum: bool);
    /// y[0] (+)= scale·Σx
    fn sum(&self, x: &Self::Buf, y: &Self::Buf, scale: f32, accum: bool);
    /// y[0] (+)= Σ x²
    fn sumsq(&self, x: &Self::Buf, y: &Self::Buf, accum: bool);
    /// y *= s
    fn scale(&self, y: &Self::Buf, s: f32);
    /// dx[i] (+)= scale·dy[0]
    fn bcast_scalar(&self, dy: &Self::Buf, dx: &Self::Buf, scale: f32, accum: bool);
    // ------------------------------------------------------------ vision ---
    /// col [planes·kh·kw, OH·OW] ← one image at element offset `xoff` of x (planes = C)
    fn im2col(&self, x: &Self::Buf, xoff: usize, col: &Self::Buf, g: &Win);
    /// dx[dxoff..] (+)= col2im(col)
    fn col2im(&self, col: &Self::Buf, dx: &Self::Buf, dxoff: usize, g: &Win, accum: bool);
    /// db[c] (+)= Σ_{n,p} dy[n][c][p]
    fn channel_sum(&self, dy: &Self::Buf, db: &Self::Buf, n: usize, c: usize, hw: usize, accum: bool);
    #[allow(clippy::too_many_arguments)]
    fn bn_stats(&self, x: &Self::Buf, mean: &Self::Buf, rstd: &Self::Buf, n: usize, c: usize, hw: usize, eps: f32);
    #[allow(clippy::too_many_arguments)]
    fn bn_apply(&self, x: &Self::Buf, mean: &Self::Buf, rstd: &Self::Buf, g: &Self::Buf, b: &Self::Buf, y: &Self::Buf, n: usize, c: usize, hw: usize);
    #[allow(clippy::too_many_arguments)]
    fn bn_wgrad(&self, x: &Self::Buf, mean: &Self::Buf, rstd: &Self::Buf, dy: &Self::Buf, dg: &Self::Buf, db: &Self::Buf, n: usize, c: usize, hw: usize);
    #[allow(clippy::too_many_arguments)]
    fn bn_bwd(&self, x: &Self::Buf, mean: &Self::Buf, rstd: &Self::Buf, g: &Self::Buf, dg: &Self::Buf, db: &Self::Buf, dy: &Self::Buf, dx: &Self::Buf, n: usize, c: usize, hw: usize, accum: bool);
    #[allow(clippy::too_many_arguments)]
    fn bn_running(&self, mean: &Self::Buf, rstd: &Self::Buf, rm: &Self::Buf, rv: &Self::Buf, c: usize, momentum: f32, eps: f32, unbias: f32);
    fn maxpool(&self, x: &Self::Buf, y: &Self::Buf, idx: &Self::Buf, g: &Win);
    fn maxpool_bwd(&self, dy: &Self::Buf, idx: &Self::Buf, dx: &Self::Buf, g: &Win, accum: bool);
    fn avgpool(&self, x: &Self::Buf, y: &Self::Buf, g: &Win, count_include_pad: bool);
    fn avgpool_bwd(&self, dy: &Self::Buf, dx: &Self::Buf, g: &Win, count_include_pad: bool, accum: bool);
    #[allow(clippy::too_many_arguments)]
    fn upsample(&self, x: &Self::Buf, y: &Self::Buf, planes: usize, h: usize, w: usize, fy: usize, fx: usize);
    #[allow(clippy::too_many_arguments)]
    fn upsample_bwd(&self, dy: &Self::Buf, dx: &Self::Buf, planes: usize, h: usize, w: usize, fy: usize, fx: usize, accum: bool);
    /// x [n,c,h,w], grid [n,ho,wo,2] → y [n,c,ho,wo] (bilinear, zeros, align_corners = false)
    #[allow(clippy::too_many_arguments)]
    fn grid_sample(&self, x: &Self::Buf, grid: &Self::Buf, y: &Self::Buf, n: usize, c: usize, h: usize, w: usize, ho: usize, wo: usize);
    /// dx += (always accumulates), dgrid (+)=
    #[allow(clippy::too_many_arguments)]
    fn grid_sample_bwd(&self, x: &Self::Buf, grid: &Self::Buf, dy: &Self::Buf, dx: Option<&Self::Buf>, dgrid: Option<&Self::Buf>, n: usize, c: usize, h: usize, w: usize, ho: usize, wo: usize, accum_dgrid: bool);
    /// y[b][k] = x[b][idx[b][k]] (rows of c floats; idx holds integers as f32)
    #[allow(clippy::too_many_arguments)]
    fn gather_rows(&self, x: &Self::Buf, idx: &Self::Buf, y: &Self::Buf, b: usize, n: usize, k: usize, c: usize);
    /// dx[b][idx[b][k]] += dy[b][k] (always accumulates)
    #[allow(clippy::too_many_arguments)]
    fn gather_rows_bwd(&self, dy: &Self::Buf, idx: &Self::Buf, dx: &Self::Buf, b: usize, n: usize, k: usize, c: usize);

    #[allow(clippy::too_many_arguments)]
    fn adamw(&self, p: &Self::Buf, g: &Self::Buf, m: &Self::Buf, v: &Self::Buf, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, step: u32);
}
