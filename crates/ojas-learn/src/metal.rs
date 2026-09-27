//! Metal backend: the `cnn_train` kernel family (`ojas-metal` `kernels/cnn_train.rs`), f32,
//! CPU-parity naive GEMM only (no simdgroup_matrix tensor-core path yet). Every launch goes through
//! `KernelRuntime::dispatch`, the same contract `ojas-learn/src/cuda.rs` uses, so this file mirrors
//! it call for call.

use anyhow::Result;
use ojas_core::{Device, KernelRuntime};
use ojas_metal::{MBuf, MetalGpu};

use crate::backend::{pad6, Backend, Bcast, Binary, Gemm, Unary, Win, MAX_RANK};

pub struct Metal {
    pub gpu: MetalGpu,
}

impl Metal {
    pub fn new() -> Result<Self> {
        let mut gpu = MetalGpu::new()?;
        gpu.ensure_family("cnn_train")?;
        Ok(Metal { gpu })
    }

    fn go(&self, name: &str, bufs: &[&MBuf], consts: &[u32], grid: [u32; 3], block: u32) {
        let b: Vec<(&MBuf, usize)> = bufs.iter().map(|b| (*b, 0)).collect();
        self.go_at(name, &b, consts, grid, block);
    }

    fn go_at(&self, name: &str, bufs: &[(&MBuf, usize)], consts: &[u32], grid: [u32; 3], block: u32) {
        let enc = self.gpu.begin();
        let b: Vec<(&MBuf, u64)> = bufs.iter().map(|&(b, off)| (b, off as u64 * 4)).collect();
        self.gpu
            .dispatch(&enc, name, &b, consts, grid, [block, 1, 1])
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        self.gpu.submit(enc).unwrap_or_else(|e| panic!("{name} submit: {e:#}"));
    }

    fn go1_at(&self, name: &str, bufs: &[(&MBuf, usize)], consts: &[u32], n: usize) {
        if n > 0 {
            self.go_at(name, bufs, consts, [n.div_ceil(256) as u32, 1, 1], 256);
        }
    }

    /// One thread per element.
    fn go1(&self, name: &str, bufs: &[&MBuf], consts: &[u32], n: usize) {
        if n > 0 {
            self.go(name, bufs, consts, [n.div_ceil(256) as u32, 1, 1], 256);
        }
    }
}

fn f(v: f32) -> u32 {
    v.to_bits()
}

fn bcast_consts(bc: &Bcast, op: u32) -> Vec<u32> {
    let mut c = vec![bc.n as u32, op];
    c.extend(bc.dims.iter().map(|&v| v as u32));
    c.extend(bc.sa.iter().map(|&v| v as u32));
    c.extend(bc.sb.iter().map(|&v| v as u32));
    c
}

/// Window constants, same order as `ojas-learn/src/cuda.rs`. Conv (im2col /
/// col2im): C H W kh kw sh sw pt pl OH OW. Pooling: planes H W OH OW kh kw sh sw pt pl.
fn win_consts(g: &Win, pool: bool) -> Vec<u32> {
    let v = if pool {
        [g.planes, g.h, g.w, g.oh, g.ow, g.kh, g.kw, g.sh, g.sw, g.pt, g.pl]
    } else {
        [g.planes, g.h, g.w, g.kh, g.kw, g.sh, g.sw, g.pt, g.pl, g.oh, g.ow]
    };
    v.iter().map(|&x| x as u32).collect()
}

fn pad_strides(s: &[usize]) -> [usize; MAX_RANK] {
    let mut r = [0; MAX_RANK];
    r[MAX_RANK - s.len()..].copy_from_slice(s);
    r
}

impl Backend for Metal {
    type Buf = MBuf;

    fn alloc(&self, n: usize) -> MBuf {
        let b = self.gpu.alloc(n.max(1));
        // `MetalGpu::alloc` hands back a fresh `newBufferWithLength:`, whose contents
        // Metal does not guarantee are zero, while this trait promises zero-filled
        // buffers. Unified memory makes the memset cheap.
        unsafe { std::ptr::write_bytes(b.buf.contents() as *mut u8, 0, b.len * 4) };
        b
    }
    fn upload(&self, v: &[f32]) -> MBuf {
        if v.is_empty() {
            return self.alloc(1);
        }
        self.gpu.upload(v)
    }
    fn download(&self, b: &MBuf) -> Vec<f32> {
        self.gpu.read(b)
    }
    fn len(&self, b: &MBuf) -> usize {
        b.len
    }

    fn fill(&self, y: &MBuf, v: f32) {
        self.go1("learn_fill", &[y], &[self.len(y) as u32, f(v)], self.len(y));
    }

    fn axpby(&self, x: &MBuf, y: &MBuf, a: f32, b: f32) {
        let n = self.len(y);
        self.go1("learn_axpby", &[x, y], &[n as u32, f(a), f(b)], n);
    }

    fn unary(&self, op: Unary, x: &MBuf, y: &MBuf) {
        let n = self.len(y);
        self.go1("learn_unary", &[x, y], &[n as u32, op as u32], n);
    }

    fn unary_bwd(&self, op: Unary, x: &MBuf, y: &MBuf, dy: &MBuf, dx: &MBuf, accum: bool) {
        let n = self.len(dx);
        self.go1("learn_unary_bwd", &[x, y, dy, dx], &[n as u32, op as u32, accum as u32], n);
    }

    fn binary(&self, op: Binary, a: &MBuf, b: &MBuf, y: &MBuf, bc: &Bcast) {
        self.go1("learn_binary", &[a, b, y], &bcast_consts(bc, op as u32), bc.n);
    }

    fn binary_grad(&self, op: Binary, which: u8, a: &MBuf, b: &MBuf, dy: &MBuf, t: &MBuf, bc: &Bcast) {
        let mut c = vec![which as u32];
        c.extend(bcast_consts(bc, op as u32));
        self.go1("learn_binary_grad", &[a, b, dy, t], &c, bc.n);
    }

    fn reduce_to(&self, t: &MBuf, full: &[usize], g: &MBuf, target: &[usize], accum: bool) {
        // one general kernel; no CUDA-style row-split fast path yet
        let (fd, td) = (pad6(full), pad6(target));
        let mut red = 0u32;
        for d in 0..MAX_RANK {
            if td[d] == 1 && fd[d] != 1 {
                red |= 1 << d;
            }
        }
        let m: usize = target.iter().product();
        let mut c = vec![m as u32, accum as u32, red];
        c.extend(fd.iter().map(|&v| v as u32));
        self.go1("learn_reduce_to", &[t, g], &c, m);
    }

    fn copy_strided(&self, x: &MBuf, xoff: usize, xs: &[usize], y: &MBuf, yoff: usize, ys: &[usize], dims: &[usize], accum: bool) {
        let n: usize = dims.iter().product();
        let (d, sx, sy) = (pad6(dims), pad_strides(xs), pad_strides(ys));
        let mut c = vec![n as u32, accum as u32];
        c.extend(d.iter().map(|&v| v as u32));
        c.extend(sx.iter().map(|&v| v as u32));
        c.extend(sy.iter().map(|&v| v as u32));
        c.extend([xoff as u32, yoff as u32]);
        self.go1("learn_copy_strided", &[x, y], &c, n);
    }

    fn gemm(&self, a: &MBuf, b: &MBuf, c: &MBuf, g: &Gemm) {
        let consts = [
            g.m as u32, g.n as u32, g.k as u32, g.ta as u32, g.tb as u32,
            g.sa as u32, g.sb as u32, g.sc as u32, f(g.alpha), f(g.beta),
        ];
        let grid = [g.n.div_ceil(64) as u32, g.m.div_ceil(64) as u32, g.batch as u32];
        self.go_at("learn_gemm", &[(a, g.oa), (b, g.ob), (c, g.oc)], &consts, grid, 256);
    }

    fn softmax(&self, x: &MBuf, y: &MBuf, rows: usize, cols: usize) {
        self.go("learn_softmax", &[x, y], &[rows as u32, cols as u32], [rows as u32, 1, 1], 256);
    }

    fn softmax_bwd(&self, y: &MBuf, dy: &MBuf, dx: &MBuf, rows: usize, cols: usize, accum: bool) {
        self.go("learn_softmax_bwd", &[y, dy, dx], &[rows as u32, cols as u32, accum as u32], [rows as u32, 1, 1], 256);
    }

    fn layernorm(&self, x: &MBuf, g: &MBuf, b: &MBuf, y: &MBuf, mean: &MBuf, rstd: &MBuf, rows: usize, cols: usize, eps: f32) {
        self.go("learn_layernorm", &[x, g, b, y, mean, rstd], &[rows as u32, cols as u32, f(eps)], [rows as u32, 1, 1], 256);
    }

    fn layernorm_bwd(&self, x: &MBuf, g: &MBuf, mean: &MBuf, rstd: &MBuf, dy: &MBuf, dx: &MBuf, rows: usize, cols: usize, accum: bool) {
        self.go("learn_layernorm_bwd", &[x, g, mean, rstd, dy, dx], &[rows as u32, cols as u32, accum as u32], [rows as u32, 1, 1], 256);
    }

    fn layernorm_wgrad(&self, x: &MBuf, mean: &MBuf, rstd: &MBuf, dy: &MBuf, dg: &MBuf, db: &MBuf, rows: usize, cols: usize, accum: bool) {
        self.go1("learn_layernorm_wgrad", &[x, mean, rstd, dy, dg, db], &[rows as u32, cols as u32, accum as u32], cols);
    }

    fn sum(&self, x: &MBuf, y: &MBuf, scale: f32, accum: bool) {
        self.go("learn_sum", &[x, y], &[self.len(x) as u32, f(scale), accum as u32], [1, 1, 1], 1024);
    }

    fn sumsq(&self, x: &MBuf, y: &MBuf, accum: bool) {
        self.go("learn_sumsq", &[x, y], &[self.len(x) as u32, accum as u32], [1, 1, 1], 1024);
    }

    fn scale(&self, y: &MBuf, s: f32) {
        let n = self.len(y);
        self.go1("learn_scale", &[y], &[n as u32, f(s)], n);
    }

    fn bcast_scalar(&self, dy: &MBuf, dx: &MBuf, scale: f32, accum: bool) {
        let n = self.len(dx);
        self.go1("learn_bcast_scalar", &[dy, dx], &[n as u32, f(scale), accum as u32], n);
    }

    fn im2col(&self, x: &MBuf, xoff: usize, col: &MBuf, g: &Win) {
        let n = g.planes * g.kh * g.kw * g.oh * g.ow;
        self.go1_at("learn_im2col", &[(x, xoff), (col, 0)], &win_consts(g, false), n);
    }

    fn col2im(&self, col: &MBuf, dx: &MBuf, dxoff: usize, g: &Win, accum: bool) {
        let mut c = win_consts(g, false);
        c.push(accum as u32);
        self.go1_at("learn_col2im", &[(col, 0), (dx, dxoff)], &c, g.planes * g.h * g.w);
    }

    fn channel_sum(&self, dy: &MBuf, db: &MBuf, n: usize, c: usize, hw: usize, accum: bool) {
        self.go("learn_channel_sum", &[dy, db], &[n as u32, c as u32, hw as u32, accum as u32], [c as u32, 1, 1], 256);
    }

    fn bn_stats(&self, x: &MBuf, mean: &MBuf, rstd: &MBuf, n: usize, c: usize, hw: usize, eps: f32) {
        self.go("learn_bn_stats", &[x, mean, rstd], &[n as u32, c as u32, hw as u32, f(eps)], [c as u32, 1, 1], 256);
    }

    fn bn_apply(&self, x: &MBuf, mean: &MBuf, rstd: &MBuf, g: &MBuf, b: &MBuf, y: &MBuf, n: usize, c: usize, hw: usize) {
        let t = n * c * hw;
        self.go1("learn_bn_apply", &[x, mean, rstd, g, b, y], &[t as u32, c as u32, hw as u32], t);
    }

    fn bn_wgrad(&self, x: &MBuf, mean: &MBuf, rstd: &MBuf, dy: &MBuf, dg: &MBuf, db: &MBuf, n: usize, c: usize, hw: usize) {
        self.go("learn_bn_wgrad", &[x, mean, rstd, dy, dg, db], &[n as u32, c as u32, hw as u32], [c as u32, 1, 1], 256);
    }

    fn bn_bwd(&self, x: &MBuf, mean: &MBuf, rstd: &MBuf, g: &MBuf, dg: &MBuf, db: &MBuf, dy: &MBuf, dx: &MBuf, n: usize, c: usize, hw: usize, accum: bool) {
        let t = n * c * hw;
        let inv_m = 1.0 / (n * hw) as f32;
        self.go1("learn_bn_bwd", &[x, mean, rstd, g, dg, db, dy, dx], &[t as u32, c as u32, hw as u32, f(inv_m), accum as u32], t);
    }

    fn bn_running(&self, mean: &MBuf, rstd: &MBuf, rm: &MBuf, rv: &MBuf, c: usize, momentum: f32, eps: f32, unbias: f32) {
        self.go1("learn_bn_running", &[mean, rstd, rm, rv], &[c as u32, f(momentum), f(eps), f(unbias)], c);
    }

    fn maxpool(&self, x: &MBuf, y: &MBuf, idx: &MBuf, g: &Win) {
        self.go1("learn_maxpool", &[x, y, idx], &win_consts(g, true), g.planes * g.oh * g.ow);
    }

    fn maxpool_bwd(&self, dy: &MBuf, idx: &MBuf, dx: &MBuf, g: &Win, accum: bool) {
        let mut c = win_consts(g, true);
        c.push(accum as u32);
        self.go1("learn_maxpool_bwd", &[dy, idx, dx], &c, g.planes * g.h * g.w);
    }

    fn avgpool(&self, x: &MBuf, y: &MBuf, g: &Win, count_include_pad: bool) {
        let mut c = win_consts(g, true);
        c.push(count_include_pad as u32);
        self.go1("learn_avgpool", &[x, y], &c, g.planes * g.oh * g.ow);
    }

    fn avgpool_bwd(&self, dy: &MBuf, dx: &MBuf, g: &Win, count_include_pad: bool, accum: bool) {
        let mut c = win_consts(g, true);
        c.extend([count_include_pad as u32, accum as u32]);
        self.go1("learn_avgpool_bwd", &[dy, dx], &c, g.planes * g.h * g.w);
    }

    fn upsample(&self, x: &MBuf, y: &MBuf, planes: usize, h: usize, w: usize, fy: usize, fx: usize) {
        self.go1("learn_upsample", &[x, y], &[planes as u32, h as u32, w as u32, fy as u32, fx as u32], planes * h * fy * w * fx);
    }

    fn upsample_bwd(&self, dy: &MBuf, dx: &MBuf, planes: usize, h: usize, w: usize, fy: usize, fx: usize, accum: bool) {
        self.go1("learn_upsample_bwd", &[dy, dx], &[planes as u32, h as u32, w as u32, fy as u32, fx as u32, accum as u32], planes * h * w);
    }

    fn grid_sample(&self, x: &MBuf, grid: &MBuf, y: &MBuf, n: usize, c: usize, h: usize, w: usize, ho: usize, wo: usize) {
        self.go1("learn_grid_sample", &[x, grid, y], &[n, c, h, w, ho, wo].map(|v| v as u32), n * ho * wo);
    }

    fn grid_sample_bwd(&self, x: &MBuf, grid: &MBuf, dy: &MBuf, dx: Option<&MBuf>, dgrid: Option<&MBuf>, n: usize, c: usize, h: usize, w: usize, ho: usize, wo: usize, accum_dgrid: bool) {
        // a missing output gets a harmless stand-in (the kernel skips it by flag)
        let (dxb, dgb) = (dx.unwrap_or(dy), dgrid.unwrap_or(dy));
        let consts = [n, c, h, w, ho, wo, dx.is_some() as usize, dgrid.is_some() as usize, accum_dgrid as usize].map(|v| v as u32);
        self.go1("learn_grid_sample_bwd", &[x, grid, dy, dxb, dgb], &consts, n * ho * wo);
    }

    fn gather_rows(&self, x: &MBuf, idx: &MBuf, y: &MBuf, b: usize, n: usize, k: usize, c: usize) {
        self.go1("learn_gather_rows", &[x, idx, y], &[b, n, k, c].map(|v| v as u32), b * k * c);
    }

    fn gather_rows_bwd(&self, dy: &MBuf, idx: &MBuf, dx: &MBuf, b: usize, n: usize, k: usize, c: usize) {
        self.go1("learn_gather_rows_bwd", &[dy, idx, dx], &[b, n, k, c].map(|v| v as u32), b * k * c);
    }

    fn adamw(&self, p: &MBuf, g: &MBuf, m: &MBuf, v: &MBuf, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, step: u32) {
        let (bc1, bc2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
        let n = self.len(p);
        self.go1("learn_adamw", &[p, g, m, v], &[n as u32, f(lr), f(b1), f(b2), f(eps), f(wd), f(bc1), f(bc2)], n);
    }
}
