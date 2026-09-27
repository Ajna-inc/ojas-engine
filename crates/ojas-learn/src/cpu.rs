//! The CPU reference backend: plain Rust, one loop per primitive, written for
//! clarity, not speed. Device backends are checked against it.

use std::cell::RefCell;

use crate::backend::{pad6, strides, Backend, Bcast, Binary, Gemm, Unary, Win, MAX_RANK};

pub struct Cpu;

pub type CpuBuf = RefCell<Vec<f32>>;

fn unary_f(x: f32, op: Unary) -> f32 {
    match op {
        Unary::Relu => x.max(0.0),
        Unary::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Unary::Silu => x / (1.0 + (-x).exp()),
        Unary::Tanh => x.tanh(),
        Unary::Gelu => 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)),
        Unary::Exp => x.exp(),
        Unary::Log => x.ln(),
        Unary::Neg => -x,
        Unary::Sqrt => x.sqrt(),
    }
}

fn unary_d(x: f32, y: f32, op: Unary) -> f32 {
    match op {
        Unary::Relu => (x > 0.0) as u8 as f32,
        Unary::Sigmoid => y * (1.0 - y),
        Unary::Silu => {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 + x * (1.0 - s))
        }
        Unary::Tanh => 1.0 - y * y,
        Unary::Gelu => {
            0.5 * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)) + x * 0.398_942_3 * (-0.5 * x * x).exp()
        }
        Unary::Exp => y,
        Unary::Log => 1.0 / x,
        Unary::Neg => -1.0,
        Unary::Sqrt => 0.5 / y,
    }
}

/// erf via f64 (Abramowitz–Stegun 7.1.26 is too coarse for gradient checks):
/// the series / continued fraction split used by most libms.
fn erf(x: f32) -> f32 {
    let x = x as f64;
    let t = x.abs();
    let r = if t < 2.5 {
        // Maclaurin series
        let mut sum = t;
        let mut term = t;
        let x2 = t * t;
        let mut n = 0.0;
        loop {
            n += 1.0;
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() < 1e-17 {
                break;
            }
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // continued fraction for erfc
        let mut f = 0.0;
        for k in (1..60).rev() {
            f = (k as f64 / 2.0) / (t + f);
        }
        1.0 - (-t * t).exp() / std::f64::consts::PI.sqrt() / (t + f)
    };
    (if x < 0.0 { -r } else { r }) as f32
}

fn binary_f(a: f32, b: f32, op: Binary) -> f32 {
    match op {
        Binary::Add => a + b,
        Binary::Sub => a - b,
        Binary::Mul => a * b,
        Binary::Div => a / b,
        Binary::Max => if a >= b { a } else { b },
        Binary::Min => if a <= b { a } else { b },
    }
}

fn binary_d(a: f32, b: f32, op: Binary, which: u8) -> f32 {
    let w = which == 1;
    match op {
        Binary::Add => 1.0,
        Binary::Sub => if w { -1.0 } else { 1.0 },
        Binary::Mul => if w { a } else { b },
        Binary::Div => if w { -a / (b * b) } else { 1.0 / b },
        Binary::Max => (if w { b > a } else { a >= b }) as u8 as f32,
        Binary::Min => (if w { b < a } else { a <= b }) as u8 as f32,
    }
}

fn offsets(i: usize, bc: &Bcast) -> (usize, usize) {
    let (mut rem, mut oa, mut ob) = (i, 0, 0);
    for d in (0..MAX_RANK).rev() {
        let c = rem % bc.dims[d];
        rem /= bc.dims[d];
        oa += c * bc.sa[d];
        ob += c * bc.sb[d];
    }
    (oa, ob)
}

fn put(y: &mut [f32], i: usize, v: f32, accum: bool) {
    if accum {
        y[i] += v
    } else {
        y[i] = v
    }
}

impl Backend for Cpu {
    type Buf = CpuBuf;

    fn alloc(&self, n: usize) -> CpuBuf {
        RefCell::new(vec![0.0; n])
    }
    fn upload(&self, v: &[f32]) -> CpuBuf {
        RefCell::new(v.to_vec())
    }
    fn download(&self, b: &CpuBuf) -> Vec<f32> {
        b.borrow().clone()
    }
    fn len(&self, b: &CpuBuf) -> usize {
        b.borrow().len()
    }

    fn fill(&self, y: &CpuBuf, v: f32) {
        y.borrow_mut().iter_mut().for_each(|e| *e = v);
    }

    fn axpby(&self, x: &CpuBuf, y: &CpuBuf, a: f32, b: f32) {
        let x = x.borrow();
        for (yi, xi) in y.borrow_mut().iter_mut().zip(x.iter()) {
            *yi = a * xi + b * *yi;
        }
    }

    fn unary(&self, op: Unary, x: &CpuBuf, y: &CpuBuf) {
        let x = x.borrow();
        for (yi, xi) in y.borrow_mut().iter_mut().zip(x.iter()) {
            *yi = unary_f(*xi, op);
        }
    }

    fn unary_bwd(&self, op: Unary, x: &CpuBuf, y: &CpuBuf, dy: &CpuBuf, dx: &CpuBuf, accum: bool) {
        let (x, y, dy) = (x.borrow(), y.borrow(), dy.borrow());
        let mut dx = dx.borrow_mut();
        for i in 0..x.len() {
            put(&mut dx, i, unary_d(x[i], y[i], op) * dy[i], accum);
        }
    }

    fn binary(&self, op: Binary, a: &CpuBuf, b: &CpuBuf, y: &CpuBuf, bc: &Bcast) {
        let (a, b) = (a.borrow(), b.borrow());
        let mut y = y.borrow_mut();
        for i in 0..bc.n {
            let (oa, ob) = offsets(i, bc);
            y[i] = binary_f(a[oa], b[ob], op);
        }
    }

    fn binary_grad(&self, op: Binary, which: u8, a: &CpuBuf, b: &CpuBuf, dy: &CpuBuf, t: &CpuBuf, bc: &Bcast) {
        let (a, b, dy) = (a.borrow(), b.borrow(), dy.borrow());
        let mut t = t.borrow_mut();
        for i in 0..bc.n {
            let (oa, ob) = offsets(i, bc);
            t[i] = dy[i] * binary_d(a[oa], b[ob], op, which);
        }
    }

    fn reduce_to(&self, t: &CpuBuf, full: &[usize], g: &CpuBuf, target: &[usize], accum: bool) {
        let (f, tg) = (pad6(full), pad6(target));
        let fs = strides(&f);
        let ts = strides(&tg);
        let t = t.borrow();
        let mut g = g.borrow_mut();
        if !accum {
            g.iter_mut().for_each(|v| *v = 0.0);
        }
        // every full element adds into its target element, in full index order
        let mut acc = vec![0.0f32; g.len()];
        for (i, &v) in t.iter().enumerate() {
            let mut o = 0;
            for d in 0..MAX_RANK {
                let c = i / fs[d] % f[d];
                if tg[d] != 1 {
                    o += c * ts[d];
                }
            }
            acc[o] += v;
        }
        for (gi, a) in g.iter_mut().zip(acc) {
            *gi += a;
        }
    }

    fn copy_strided(&self, x: &CpuBuf, xoff: usize, xs: &[usize], y: &CpuBuf, yoff: usize, ys: &[usize], dims: &[usize], accum: bool) {
        let n: usize = dims.iter().product();
        let x = x.borrow();
        let mut y = y.borrow_mut();
        for i in 0..n {
            let (mut rem, mut ox, mut oy) = (i, xoff, yoff);
            for d in (0..dims.len()).rev() {
                let c = rem % dims[d];
                rem /= dims[d];
                ox += c * xs[d];
                oy += c * ys[d];
            }
            put(&mut y, oy, x[ox], accum);
        }
    }

    fn gemm(&self, a: &CpuBuf, b: &CpuBuf, c: &CpuBuf, g: &Gemm) {
        let (a, b) = (a.borrow(), b.borrow());
        let mut c = c.borrow_mut();
        for z in 0..g.batch {
            let (ao, bo, co) = (g.oa + z * g.sa, g.ob + z * g.sb, g.oc + z * g.sc);
            for i in 0..g.m {
                for j in 0..g.n {
                    let mut s = 0.0f32;
                    for kk in 0..g.k {
                        let av = if g.ta { a[ao + kk * g.m + i] } else { a[ao + i * g.k + kk] };
                        let bv = if g.tb { b[bo + j * g.k + kk] } else { b[bo + kk * g.n + j] };
                        s += av * bv;
                    }
                    let o = co + i * g.n + j;
                    c[o] = if g.beta == 0.0 { g.alpha * s } else { g.alpha * s + g.beta * c[o] };
                }
            }
        }
    }

    fn softmax(&self, x: &CpuBuf, y: &CpuBuf, rows: usize, cols: usize) {
        let x = x.borrow();
        let mut y = y.borrow_mut();
        for r in 0..rows {
            let xr = &x[r * cols..(r + 1) * cols];
            let m = xr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = xr.iter().map(|v| (v - m).exp()).sum();
            for c in 0..cols {
                y[r * cols + c] = (xr[c] - m).exp() / s;
            }
        }
    }

    fn softmax_bwd(&self, y: &CpuBuf, dy: &CpuBuf, dx: &CpuBuf, rows: usize, cols: usize, accum: bool) {
        let (y, dy) = (y.borrow(), dy.borrow());
        let mut dx = dx.borrow_mut();
        for r in 0..rows {
            let s: f32 = (0..cols).map(|c| y[r * cols + c] * dy[r * cols + c]).sum();
            for c in 0..cols {
                let i = r * cols + c;
                put(&mut dx, i, y[i] * (dy[i] - s), accum);
            }
        }
    }

    fn layernorm(&self, x: &CpuBuf, g: &CpuBuf, b: &CpuBuf, y: &CpuBuf, mean: &CpuBuf, rstd: &CpuBuf, rows: usize, cols: usize, eps: f32) {
        let (x, g, b) = (x.borrow(), g.borrow(), b.borrow());
        let (mut y, mut mean, mut rstd) = (y.borrow_mut(), mean.borrow_mut(), rstd.borrow_mut());
        for r in 0..rows {
            let xr = &x[r * cols..(r + 1) * cols];
            let mu = xr.iter().sum::<f32>() / cols as f32;
            let var = xr.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / cols as f32;
            let rs = 1.0 / (var + eps).sqrt();
            for c in 0..cols {
                y[r * cols + c] = (xr[c] - mu) * rs * g[c] + b[c];
            }
            mean[r] = mu;
            rstd[r] = rs;
        }
    }

    fn layernorm_bwd(&self, x: &CpuBuf, g: &CpuBuf, mean: &CpuBuf, rstd: &CpuBuf, dy: &CpuBuf, dx: &CpuBuf, rows: usize, cols: usize, accum: bool) {
        let (x, g, mean, rstd, dy) = (x.borrow(), g.borrow(), mean.borrow(), rstd.borrow(), dy.borrow());
        let mut dx = dx.borrow_mut();
        for r in 0..rows {
            let (mu, rs) = (mean[r], rstd[r]);
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for c in 0..cols {
                let i = r * cols + c;
                let (gh, xh) = (dy[i] * g[c], (x[i] - mu) * rs);
                s1 += gh;
                s2 += gh * xh;
            }
            let (s1, s2) = (s1 / cols as f32, s2 / cols as f32);
            for c in 0..cols {
                let i = r * cols + c;
                let (gh, xh) = (dy[i] * g[c], (x[i] - mu) * rs);
                put(&mut dx, i, rs * (gh - s1 - xh * s2), accum);
            }
        }
    }

    fn layernorm_wgrad(&self, x: &CpuBuf, mean: &CpuBuf, rstd: &CpuBuf, dy: &CpuBuf, dg: &CpuBuf, db: &CpuBuf, rows: usize, cols: usize, accum: bool) {
        let (x, mean, rstd, dy) = (x.borrow(), mean.borrow(), rstd.borrow(), dy.borrow());
        let (mut dg, mut db) = (dg.borrow_mut(), db.borrow_mut());
        for c in 0..cols {
            let (mut sg, mut sb) = (0.0f32, 0.0f32);
            for r in 0..rows {
                let i = r * cols + c;
                sg += dy[i] * (x[i] - mean[r]) * rstd[r];
                sb += dy[i];
            }
            put(&mut dg, c, sg, accum);
            put(&mut db, c, sb, accum);
        }
    }

    fn sum(&self, x: &CpuBuf, y: &CpuBuf, scale: f32, accum: bool) {
        let s: f32 = x.borrow().iter().sum();
        put(&mut y.borrow_mut(), 0, s * scale, accum);
    }

    fn sumsq(&self, x: &CpuBuf, y: &CpuBuf, accum: bool) {
        let s: f32 = x.borrow().iter().map(|v| v * v).sum();
        put(&mut y.borrow_mut(), 0, s, accum);
    }

    fn scale(&self, y: &CpuBuf, s: f32) {
        y.borrow_mut().iter_mut().for_each(|v| *v *= s);
    }

    fn bcast_scalar(&self, dy: &CpuBuf, dx: &CpuBuf, scale: f32, accum: bool) {
        let d = dy.borrow()[0];
        let mut dx = dx.borrow_mut();
        for i in 0..dx.len() {
            put(&mut dx, i, scale * d, accum);
        }
    }

    fn im2col(&self, x: &CpuBuf, xoff: usize, col: &CpuBuf, g: &Win) {
        let x = x.borrow();
        let mut col = col.borrow_mut();
        let ohw = g.oh * g.ow;
        for c in 0..g.planes {
            for ky in 0..g.kh {
                for kx in 0..g.kw {
                    let row = (c * g.kh + ky) * g.kw + kx;
                    for oy in 0..g.oh {
                        for ox in 0..g.ow {
                            let iy = (oy * g.sh + ky) as isize - g.pt as isize;
                            let ix = (ox * g.sw + kx) as isize - g.pl as isize;
                            let inside = iy >= 0 && (iy as usize) < g.h && ix >= 0 && (ix as usize) < g.w;
                            col[row * ohw + oy * g.ow + ox] = if inside { x[xoff + (c * g.h + iy as usize) * g.w + ix as usize] } else { 0.0 };
                        }
                    }
                }
            }
        }
    }

    fn col2im(&self, col: &CpuBuf, dx: &CpuBuf, dxoff: usize, g: &Win, accum: bool) {
        let col = col.borrow();
        let mut dx = dx.borrow_mut();
        let n = g.planes * g.h * g.w;
        if !accum {
            dx[dxoff..dxoff + n].iter_mut().for_each(|v| *v = 0.0);
        }
        // scatter in col order (the CUDA kernel gathers; sums match to rounding)
        let mut acc = vec![0.0f32; n];
        let ohw = g.oh * g.ow;
        for c in 0..g.planes {
            for ky in 0..g.kh {
                for kx in 0..g.kw {
                    let row = (c * g.kh + ky) * g.kw + kx;
                    for oy in 0..g.oh {
                        for ox in 0..g.ow {
                            let iy = (oy * g.sh + ky) as isize - g.pt as isize;
                            let ix = (ox * g.sw + kx) as isize - g.pl as isize;
                            if iy >= 0 && (iy as usize) < g.h && ix >= 0 && (ix as usize) < g.w {
                                acc[(c * g.h + iy as usize) * g.w + ix as usize] += col[row * ohw + oy * g.ow + ox];
                            }
                        }
                    }
                }
            }
        }
        for (d, a) in dx[dxoff..dxoff + n].iter_mut().zip(acc) {
            *d += a;
        }
    }

    fn channel_sum(&self, dy: &CpuBuf, db: &CpuBuf, n: usize, c: usize, hw: usize, accum: bool) {
        let dy = dy.borrow();
        let mut db = db.borrow_mut();
        for ch in 0..c {
            let s: f32 = (0..n).map(|b| dy[(b * c + ch) * hw..(b * c + ch + 1) * hw].iter().sum::<f32>()).sum();
            put(&mut db, ch, s, accum);
        }
    }

    fn bn_stats(&self, x: &CpuBuf, mean: &CpuBuf, rstd: &CpuBuf, n: usize, c: usize, hw: usize, eps: f32) {
        let x = x.borrow();
        let (mut mean, mut rstd) = (mean.borrow_mut(), rstd.borrow_mut());
        let m = (n * hw) as f32;
        for ch in 0..c {
            let it = || (0..n).flat_map(move |b| (b * c + ch) * hw..(b * c + ch + 1) * hw);
            let mu = it().map(|i| x[i]).sum::<f32>() / m;
            let var = it().map(|i| (x[i] - mu) * (x[i] - mu)).sum::<f32>() / m;
            mean[ch] = mu;
            rstd[ch] = 1.0 / (var + eps).sqrt();
        }
    }

    fn bn_apply(&self, x: &CpuBuf, mean: &CpuBuf, rstd: &CpuBuf, g: &CpuBuf, b: &CpuBuf, y: &CpuBuf, n: usize, c: usize, hw: usize) {
        let (x, mean, rstd, g, b) = (x.borrow(), mean.borrow(), rstd.borrow(), g.borrow(), b.borrow());
        let mut y = y.borrow_mut();
        for i in 0..n * c * hw {
            let ch = i / hw % c;
            y[i] = (x[i] - mean[ch]) * rstd[ch] * g[ch] + b[ch];
        }
    }

    fn bn_wgrad(&self, x: &CpuBuf, mean: &CpuBuf, rstd: &CpuBuf, dy: &CpuBuf, dg: &CpuBuf, db: &CpuBuf, n: usize, c: usize, hw: usize) {
        let (x, mean, rstd, dy) = (x.borrow(), mean.borrow(), rstd.borrow(), dy.borrow());
        let (mut dg, mut db) = (dg.borrow_mut(), db.borrow_mut());
        for ch in 0..c {
            let (mut sg, mut sb) = (0.0f32, 0.0f32);
            for b in 0..n {
                for i in (b * c + ch) * hw..(b * c + ch + 1) * hw {
                    sg += dy[i] * (x[i] - mean[ch]) * rstd[ch];
                    sb += dy[i];
                }
            }
            dg[ch] = sg;
            db[ch] = sb;
        }
    }

    fn bn_bwd(&self, x: &CpuBuf, mean: &CpuBuf, rstd: &CpuBuf, g: &CpuBuf, dg: &CpuBuf, db: &CpuBuf, dy: &CpuBuf, dx: &CpuBuf, n: usize, c: usize, hw: usize, accum: bool) {
        let (x, mean, rstd, g, dg, db, dy) = (x.borrow(), mean.borrow(), rstd.borrow(), g.borrow(), dg.borrow(), db.borrow(), dy.borrow());
        let mut dx = dx.borrow_mut();
        let inv_m = 1.0 / (n * hw) as f32;
        for i in 0..n * c * hw {
            let ch = i / hw % c;
            let xh = (x[i] - mean[ch]) * rstd[ch];
            put(&mut dx, i, g[ch] * rstd[ch] * (dy[i] - db[ch] * inv_m - xh * dg[ch] * inv_m), accum);
        }
    }

    fn bn_running(&self, mean: &CpuBuf, rstd: &CpuBuf, rm: &CpuBuf, rv: &CpuBuf, c: usize, momentum: f32, eps: f32, unbias: f32) {
        let (mean, rstd) = (mean.borrow(), rstd.borrow());
        let (mut rm, mut rv) = (rm.borrow_mut(), rv.borrow_mut());
        for ch in 0..c {
            let var = 1.0 / (rstd[ch] * rstd[ch]) - eps;
            rm[ch] = (1.0 - momentum) * rm[ch] + momentum * mean[ch];
            rv[ch] = (1.0 - momentum) * rv[ch] + momentum * var * unbias;
        }
    }

    fn maxpool(&self, x: &CpuBuf, y: &CpuBuf, idx: &CpuBuf, g: &Win) {
        let x = x.borrow();
        let (mut y, mut idx) = (y.borrow_mut(), idx.borrow_mut());
        for p in 0..g.planes {
            for oy in 0..g.oh {
                for ox in 0..g.ow {
                    let (mut best, mut bi) = (f32::NEG_INFINITY, -1isize);
                    for ky in 0..g.kh {
                        for kx in 0..g.kw {
                            let iy = (oy * g.sh + ky) as isize - g.pt as isize;
                            let ix = (ox * g.sw + kx) as isize - g.pl as isize;
                            if iy < 0 || iy as usize >= g.h || ix < 0 || ix as usize >= g.w {
                                continue;
                            }
                            let v = x[(p * g.h + iy as usize) * g.w + ix as usize];
                            if v > best || bi < 0 {
                                best = v;
                                bi = iy * g.w as isize + ix;
                            }
                        }
                    }
                    let o = (p * g.oh + oy) * g.ow + ox;
                    y[o] = best;
                    idx[o] = bi as f32;
                }
            }
        }
    }

    fn maxpool_bwd(&self, dy: &CpuBuf, idx: &CpuBuf, dx: &CpuBuf, g: &Win, accum: bool) {
        let (dy, idx) = (dy.borrow(), idx.borrow());
        let mut dx = dx.borrow_mut();
        if !accum {
            dx.iter_mut().for_each(|v| *v = 0.0);
        }
        for p in 0..g.planes {
            for o in 0..g.oh * g.ow {
                let oi = p * g.oh * g.ow + o;
                dx[p * g.h * g.w + idx[oi] as usize] += dy[oi];
            }
        }
    }

    fn avgpool(&self, x: &CpuBuf, y: &CpuBuf, g: &Win, count_include_pad: bool) {
        let x = x.borrow();
        let mut y = y.borrow_mut();
        for p in 0..g.planes {
            for oy in 0..g.oh {
                for ox in 0..g.ow {
                    let mut s = 0.0f32;
                    for ky in 0..g.kh {
                        for kx in 0..g.kw {
                            let iy = (oy * g.sh + ky) as isize - g.pt as isize;
                            let ix = (ox * g.sw + kx) as isize - g.pl as isize;
                            if iy >= 0 && (iy as usize) < g.h && ix >= 0 && (ix as usize) < g.w {
                                s += x[(p * g.h + iy as usize) * g.w + ix as usize];
                            }
                        }
                    }
                    y[(p * g.oh + oy) * g.ow + ox] = s / avg_div(oy, ox, g, count_include_pad);
                }
            }
        }
    }

    fn avgpool_bwd(&self, dy: &CpuBuf, dx: &CpuBuf, g: &Win, count_include_pad: bool, accum: bool) {
        let dy = dy.borrow();
        let mut dx = dx.borrow_mut();
        if !accum {
            dx.iter_mut().for_each(|v| *v = 0.0);
        }
        for p in 0..g.planes {
            for oy in 0..g.oh {
                for ox in 0..g.ow {
                    let d = dy[(p * g.oh + oy) * g.ow + ox] / avg_div(oy, ox, g, count_include_pad);
                    for ky in 0..g.kh {
                        for kx in 0..g.kw {
                            let iy = (oy * g.sh + ky) as isize - g.pt as isize;
                            let ix = (ox * g.sw + kx) as isize - g.pl as isize;
                            if iy >= 0 && (iy as usize) < g.h && ix >= 0 && (ix as usize) < g.w {
                                dx[(p * g.h + iy as usize) * g.w + ix as usize] += d;
                            }
                        }
                    }
                }
            }
        }
    }

    fn upsample(&self, x: &CpuBuf, y: &CpuBuf, planes: usize, h: usize, w: usize, fy: usize, fx: usize) {
        let x = x.borrow();
        let mut y = y.borrow_mut();
        let (oh, ow) = (h * fy, w * fx);
        for p in 0..planes {
            for oy in 0..oh {
                for ox in 0..ow {
                    y[(p * oh + oy) * ow + ox] = x[(p * h + oy / fy) * w + ox / fx];
                }
            }
        }
    }

    fn upsample_bwd(&self, dy: &CpuBuf, dx: &CpuBuf, planes: usize, h: usize, w: usize, fy: usize, fx: usize, accum: bool) {
        let dy = dy.borrow();
        let mut dx = dx.borrow_mut();
        let (oh, ow) = (h * fy, w * fx);
        for p in 0..planes {
            for iy in 0..h {
                for ix in 0..w {
                    let mut s = 0.0f32;
                    for a in 0..fy {
                        for b in 0..fx {
                            s += dy[(p * oh + iy * fy + a) * ow + ix * fx + b];
                        }
                    }
                    put(&mut dx, (p * h + iy) * w + ix, s, accum);
                }
            }
        }
    }

    fn grid_sample(&self, x: &CpuBuf, grid: &CpuBuf, y: &CpuBuf, n: usize, c: usize, h: usize, w: usize, ho: usize, wo: usize) {
        let (x, grid) = (x.borrow(), grid.borrow());
        let mut y = y.borrow_mut();
        for i in 0..n * ho * wo {
            let (b, pix) = (i / (ho * wo), i % (ho * wo));
            let s = Sample::at(grid[i * 2], grid[i * 2 + 1], h, w);
            for ch in 0..c {
                let pb = (b * c + ch) * h * w;
                y[(b * c + ch) * ho * wo + pix] = s.corners().iter().map(|&(o, wt)| x[pb + o] * wt).sum();
            }
        }
    }

    fn grid_sample_bwd(&self, x: &CpuBuf, grid: &CpuBuf, dy: &CpuBuf, dx: Option<&CpuBuf>, dgrid: Option<&CpuBuf>, n: usize, c: usize, h: usize, w: usize, ho: usize, wo: usize, accum_dgrid: bool) {
        let (x, grid, dy) = (x.borrow(), grid.borrow(), dy.borrow());
        let mut dx = dx.map(|d| d.borrow_mut());
        let mut dgrid = dgrid.map(|d| d.borrow_mut());
        for i in 0..n * ho * wo {
            let (b, pix) = (i / (ho * wo), i % (ho * wo));
            let s = Sample::at(grid[i * 2], grid[i * 2 + 1], h, w);
            let (mut gix, mut giy) = (0.0f32, 0.0f32);
            for ch in 0..c {
                let pb = (b * c + ch) * h * w;
                let g = dy[(b * c + ch) * ho * wo + pix];
                let v = |k: usize| s.valid[k].then(|| x[pb + s.off[k]]).unwrap_or(0.0);
                let (vnw, vne, vsw, vse) = (v(0), v(1), v(2), v(3));
                gix += g * ((vne - vnw) * (1.0 - s.fy) + (vse - vsw) * s.fy);
                giy += g * ((vsw - vnw) * (1.0 - s.fx) + (vse - vne) * s.fx);
                if let Some(dx) = dx.as_mut() {
                    for (o, wt) in s.corners() {
                        dx[pb + o] += g * wt;
                    }
                }
            }
            if let Some(dg) = dgrid.as_mut() {
                put(dg, i * 2, gix * w as f32 * 0.5, accum_dgrid);
                put(dg, i * 2 + 1, giy * h as f32 * 0.5, accum_dgrid);
            }
        }
    }

    fn gather_rows(&self, x: &CpuBuf, idx: &CpuBuf, y: &CpuBuf, b: usize, n: usize, k: usize, c: usize) {
        let (x, idx) = (x.borrow(), idx.borrow());
        let mut y = y.borrow_mut();
        for bk in 0..b * k {
            let r = idx[bk] as usize;
            let src = ((bk / k) * n + r) * c;
            y[bk * c..(bk + 1) * c].copy_from_slice(&x[src..src + c]);
        }
    }

    fn gather_rows_bwd(&self, dy: &CpuBuf, idx: &CpuBuf, dx: &CpuBuf, b: usize, n: usize, k: usize, c: usize) {
        let (dy, idx) = (dy.borrow(), idx.borrow());
        let mut dx = dx.borrow_mut();
        for bk in 0..b * k {
            let r = idx[bk] as usize;
            let dst = ((bk / k) * n + r) * c;
            for j in 0..c {
                dx[dst + j] += dy[bk * c + j];
            }
        }
    }

    fn adamw(&self, p: &CpuBuf, g: &CpuBuf, m: &CpuBuf, v: &CpuBuf, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, step: u32) {
        let (bc1, bc2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
        let g = g.borrow();
        let (mut p, mut m, mut v) = (p.borrow_mut(), m.borrow_mut(), v.borrow_mut());
        for i in 0..p.len() {
            let pi = p[i] * (1.0 - lr * wd);
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            p[i] = pi - lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
        }
    }
}

/// PyTorch AvgPool2d divisor.
fn avg_div(oy: usize, ox: usize, g: &Win, cip: bool) -> f32 {
    let (y0, x0) = ((oy * g.sh) as isize - g.pt as isize, (ox * g.sw) as isize - g.pl as isize);
    let (y1, x1) = (y0 + g.kh as isize, x0 + g.kw as isize);
    let (h, w) = (g.h as isize, g.w as isize);
    if cip {
        ((y1.min(h + g.pt as isize) - y0) * (x1.min(w + g.pl as isize) - x0)) as f32
    } else {
        ((y1.min(h) - y0.max(0)) * (x1.min(w) - x0.max(0))) as f32
    }
}

/// One bilinear sample point (PyTorch grid_sampler_2d, align_corners = false):
/// the four corners nw, ne, sw, se, their in-image flags and weights.
struct Sample {
    off: [usize; 4],
    valid: [bool; 4],
    wt: [f32; 4],
    fx: f32,
    fy: f32,
}

impl Sample {
    fn at(gx: f32, gy: f32, h: usize, w: usize) -> Self {
        let ix = ((gx + 1.0) * w as f32 - 1.0) * 0.5;
        let iy = ((gy + 1.0) * h as f32 - 1.0) * 0.5;
        let (x0, y0) = (ix.floor() as isize, iy.floor() as isize);
        let (fx, fy) = (ix - x0 as f32, iy - y0 as f32);
        let pts = [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)];
        let wt = [(1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy];
        let mut off = [0; 4];
        let mut valid = [false; 4];
        for (k, &(px, py)) in pts.iter().enumerate() {
            valid[k] = px >= 0 && (px as usize) < w && py >= 0 && (py as usize) < h;
            if valid[k] {
                off[k] = py as usize * w + px as usize;
            }
        }
        Sample { off, valid, wt, fx, fy }
    }

    fn corners(&self) -> Vec<(usize, f32)> {
        (0..4).filter(|&k| self.valid[k]).map(|k| (self.off[k], self.wt[k])).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::erf;

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [(0.0f32, 0.0f32), (0.5, 0.520_499_9), (1.0, 0.842_700_8), (2.0, 0.995_322_3), (3.0, 0.999_977_9), (-1.0, -0.842_700_8)] {
            assert!((erf(x) - want).abs() < 1e-6, "erf({x}) = {} want {want}", erf(x));
        }
    }
}
