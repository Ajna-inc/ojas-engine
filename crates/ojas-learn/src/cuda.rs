//! CUDA backend: the `learn` kernel family (ojas-cuda `kernels/learn.rs`),
//! f32, one stream. Every launch goes through `KernelRuntime::dispatch`.

use anyhow::Result;
use ojas_core::{Device, KernelRuntime};
use ojas_cuda::{CuBuf, CudaGpu};

use crate::backend::{pad6, Backend, Bcast, Binary, Gemm, Unary, Win, MAX_RANK};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prec {
    F32,
    Tf32,
    Bf16,
}

pub struct Cuda {
    pub gpu: CudaGpu,
    /// GEMM precision: exact f32 CUDA cores (the CPU-comparison tests), TF32
    /// tensor cores, or bf16 tensor cores with f32 accumulation (default).
    pub prec: Prec,
    sms: usize,
    /// OJAS_LEARN_PROFILE=1: sync after every launch and sum the time per kernel
    profile: Option<std::cell::RefCell<std::collections::HashMap<&'static str, (f64, usize)>>>,
}

impl Cuda {
    pub fn new(ordinal: usize) -> Result<Self> {
        let mut gpu = CudaGpu::new(ordinal)?;
        gpu.ensure_family("learn")?;
        gpu.keep_pool_memory()?;
        let profile = std::env::var("OJAS_LEARN_PROFILE").is_ok_and(|v| v == "1").then(Default::default);
        let prec = match std::env::var("OJAS_LEARN_PREC").as_deref() {
            Ok("f32") => Prec::F32,
            Ok("tf32") => Prec::Tf32,
            _ => Prec::Bf16,
        };
        let sms = gpu.properties().map(|p| p.multiprocessors).unwrap_or(28);
        Ok(Cuda { gpu, prec, sms, profile })
    }

    /// Exact f32 GEMMs (no TF32): for comparisons against the CPU reference.
    pub fn exact(ordinal: usize) -> Result<Self> {
        let mut c = Self::new(ordinal)?;
        c.prec = Prec::F32;
        Ok(c)
    }

    /// Per-kernel time since the last call (profile mode), largest first.
    pub fn take_profile(&self) -> Vec<(&'static str, f64, usize)> {
        let Some(p) = &self.profile else { return vec![] };
        let mut v: Vec<_> = p.borrow_mut().drain().map(|(k, (t, n))| (k, t, n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    fn go(&self, name: &'static str, bufs: &[&CuBuf], consts: &[u32], grid: [u32; 3], block: u32) {
        let b: Vec<(&CuBuf, usize)> = bufs.iter().map(|b| (*b, 0)).collect();
        self.go_at(name, &b, consts, grid, block);
    }

    /// Buffers at element offsets.
    fn go_at(&self, name: &'static str, bufs: &[(&CuBuf, usize)], consts: &[u32], grid: [u32; 3], block: u32) {
        let enc = self.gpu.begin();
        let b: Vec<(&CuBuf, u64)> = bufs.iter().map(|&(b, off)| (b, off as u64 * 4)).collect();
        let t0 = self.profile.as_ref().map(|_| {
            self.gpu.submit(self.gpu.begin()).ok();
            std::time::Instant::now()
        });
        self.gpu
            .dispatch(&enc, name, &b, consts, grid, [block, 1, 1])
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        if let (Some(p), Some(t0)) = (&self.profile, t0) {
            self.gpu.submit(enc).ok();
            let mut m = p.borrow_mut();
            let e = m.entry(name).or_insert((0.0, 0));
            e.0 += t0.elapsed().as_secs_f64() * 1e3;
            e.1 += 1;
        }
    }

    fn go1_at(&self, name: &'static str, bufs: &[(&CuBuf, usize)], consts: &[u32], n: usize) {
        if n > 0 {
            self.go_at(name, bufs, consts, [n.div_ceil(256) as u32, 1, 1], 256);
        }
    }

    /// One thread per element.
    fn go1(&self, name: &'static str, bufs: &[&CuBuf], consts: &[u32], n: usize) {
        if n > 0 {
            self.go(name, bufs, consts, [n.div_ceil(256) as u32, 1, 1], 256);
        }
    }

    fn gemm_tc(&self, a: &CuBuf, b: &CuBuf, c: &CuBuf, g: &Gemm, win: Option<&Win>) {
        if let Some(p) = &self.profile {
            let key = match (win.is_some(), g.tb, g.ta) {
                (true, false, _) => "flops conv fwd",
                (true, true, _) => "flops conv dW",
                (false, _, true) => "flops conv dX / matmul Aᵀ",
                _ => "flops matmul",
            };
            let mut m = p.borrow_mut();
            let e = m.entry(key).or_insert((0.0, 0));
            e.0 += 2.0 * (g.m * g.n * g.k * g.batch) as f64 / 1e9; // GFLOP in the time column
            e.1 += 1;
        }
        let consts = [g.m as u32, g.n as u32, g.k as u32, g.ta as u32, g.tb as u32, g.sa as u32, g.sb as u32, g.sc as u32, f(g.alpha), f(g.beta)];
        let (tile, kstep, kernel, block) = match self.prec {
            Prec::Bf16 => {
                let name = match (g.ta, g.tb, win.is_some()) {
                    (false, false, false) => "learn_gemm_bf16_000",
                    (true, false, false) => "learn_gemm_bf16_100",
                    (false, true, false) => "learn_gemm_bf16_010",
                    (true, true, false) => "learn_gemm_bf16_110",
                    (false, false, true) => "learn_gemm_bf16_001",
                    (false, true, true) => "learn_gemm_bf16_011",
                    (true, _, true) => panic!("implicit im2col with a transposed A"),
                };
                (128, 32, name, 256)
            }
            _ => (64, 16, "learn_gemm_tc", 128),
        };
        let tiles = g.n.div_ceil(tile) * g.m.div_ceil(tile);
        if self.prec == Prec::F32 {
            let grid = [g.n.div_ceil(64) as u32, g.m.div_ceil(64) as u32, g.batch as u32];
            self.go_at("learn_gemm", &[(a, g.oa), (b, g.ob), (c, g.oc)], &consts, grid, 256);
            return;
        }
        // split K when the output tiles alone cannot fill the GPU (weight gradients:
        // small M×N, K = every output pixel)
        let blocks = tiles * g.batch;
        let want = 4 * self.sms;
        let splits = if blocks < want && g.k >= 512 { want.div_ceil(blocks).min(g.k / 256).max(1) } else { 1 };
        let kchunk = g.k.div_ceil(splits).div_ceil(kstep) * kstep;
        let splits = g.k.div_ceil(kchunk);
        let mut c2 = consts.to_vec();
        c2.extend([kchunk as u32, splits as u32]);
        match win {
            Some(w) => c2.extend([1, w.planes, w.h, w.w, w.kh, w.kw, w.sh, w.sw, w.pt, w.pl, w.oh, w.ow].map(|v| v as u32)),
            None => c2.extend([0u32; 12]),
        }
        if self.prec == Prec::Bf16 {
            // float4 loads need the contiguous axis, offsets and batch strides 4-aligned
            let lda = if g.ta { g.m } else { g.k };
            let ldb = if g.tb { g.k } else { g.n };
            let va = lda % 4 == 0 && g.oa % 4 == 0 && g.sa % 4 == 0;
            let vb = win.is_none() && ldb % 4 == 0 && g.ob % 4 == 0 && g.sb % 4 == 0;
            c2.extend([va as u32, vb as u32]);
        }
        let grid = [g.n.div_ceil(tile) as u32, g.m.div_ceil(tile) as u32, (g.batch * splits) as u32];
        if splits == 1 {
            self.go_at(kernel, &[(a, g.oa), (b, g.ob), (c, g.oc)], &c2, grid, block);
        } else {
            let mn = g.m * g.n;
            let ws = self.alloc_out(g.batch * splits * mn);
            self.go_at(kernel, &[(a, g.oa), (b, g.ob), (&ws, 0)], &c2, grid, block);
            self.go1_at("learn_splitk_reduce", &[(&ws, 0), (c, g.oc)], &[mn as u32, g.batch as u32, splits as u32, g.sc as u32, f(g.beta)], g.batch * mn);
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

impl Backend for Cuda {
    type Buf = CuBuf;

    fn alloc(&self, n: usize) -> CuBuf {
        self.gpu.alloc(n.max(1))
    }
    fn alloc_out(&self, n: usize) -> CuBuf {
        // OJAS_LEARN_ZERO_OUT=1: zero-fill anyway (debugging reads of unwritten memory)
        if std::env::var("OJAS_LEARN_ZERO_OUT").is_ok_and(|v| v == "1") {
            return self.gpu.alloc(n.max(1));
        }
        self.gpu.alloc_uninit(n).expect("cuda alloc")
    }
    fn upload(&self, v: &[f32]) -> CuBuf {
        if v.is_empty() {
            return self.gpu.alloc(1);
        }
        self.gpu.upload(v)
    }
    fn download(&self, b: &CuBuf) -> Vec<f32> {
        let mut out = vec![0.0; self.len(b)];
        self.gpu.read(b, &mut out);
        out
    }
    fn len(&self, b: &CuBuf) -> usize {
        b.bytes.len() / 4
    }

    fn fill(&self, y: &CuBuf, v: f32) {
        self.go1("learn_fill", &[y], &[self.len(y) as u32, f(v)], self.len(y));
    }

    fn axpby(&self, x: &CuBuf, y: &CuBuf, a: f32, b: f32) {
        let n = self.len(y);
        self.go1("learn_axpby", &[x, y], &[n as u32, f(a), f(b)], n);
    }

    fn unary(&self, op: Unary, x: &CuBuf, y: &CuBuf) {
        let n = self.len(y);
        self.go1("learn_unary", &[x, y], &[n as u32, op as u32], n);
    }

    fn unary_bwd(&self, op: Unary, x: &CuBuf, y: &CuBuf, dy: &CuBuf, dx: &CuBuf, accum: bool) {
        let n = self.len(dx);
        self.go1("learn_unary_bwd", &[x, y, dy, dx], &[n as u32, op as u32, accum as u32], n);
    }

    fn binary(&self, op: Binary, a: &CuBuf, b: &CuBuf, y: &CuBuf, bc: &Bcast) {
        self.go1("learn_binary", &[a, b, y], &bcast_consts(bc, op as u32), bc.n);
    }

    fn binary_grad(&self, op: Binary, which: u8, a: &CuBuf, b: &CuBuf, dy: &CuBuf, t: &CuBuf, bc: &Bcast) {
        let mut c = vec![which as u32];
        c.extend(bcast_consts(bc, op as u32));
        self.go1("learn_binary_grad", &[a, b, dy, t], &c, bc.n);
    }

    fn reduce_to(&self, t: &CuBuf, full: &[usize], g: &CuBuf, target: &[usize], accum: bool) {
        let (fd, td) = (pad6(full), pad6(target));
        let mut red = 0u32;
        for d in 0..MAX_RANK {
            if td[d] == 1 && fd[d] != 1 {
                red |= 1 << d;
            }
        }
        let m: usize = target.iter().product();
        // reduced axes all before the kept ones (bias-like [R, C] → [C]): split the rows
        let first_kept = (0..MAX_RANK).find(|&d| red >> d & 1 == 0 && fd[d] != 1).unwrap_or(MAX_RANK);
        let prefix = (first_kept..MAX_RANK).all(|d| red >> d & 1 == 0);
        let r: usize = fd.iter().product::<usize>() / m.max(1);
        if prefix && r >= 64 && m >= 1 {
            let splits = (r / 32).clamp(1, 256);
            let chunk = r.div_ceil(splits);
            let splits = r.div_ceil(chunk);
            let part = self.alloc_out(splits * m);
            self.go("learn_reduce_rows_partial", &[t, &part], &[r as u32, m as u32, chunk as u32], [m.div_ceil(256) as u32, splits as u32, 1], 256);
            self.go1("learn_reduce_rows_final", &[&part, g], &[m as u32, splits as u32, accum as u32], m);
            return;
        }
        let mut c = vec![m as u32, accum as u32, red];
        c.extend(fd.iter().map(|&v| v as u32));
        self.go1("learn_reduce_to", &[t, g], &c, m);
    }

    fn copy_strided(&self, x: &CuBuf, xoff: usize, xs: &[usize], y: &CuBuf, yoff: usize, ys: &[usize], dims: &[usize], accum: bool) {
        let n: usize = dims.iter().product();
        let (d, sx, sy) = (pad6(dims), pad_strides(xs), pad_strides(ys));
        let mut c = vec![n as u32, accum as u32];
        c.extend(d.iter().map(|&v| v as u32));
        c.extend(sx.iter().map(|&v| v as u32));
        c.extend(sy.iter().map(|&v| v as u32));
        c.extend([xoff as u32, yoff as u32]);
        self.go1("learn_copy_strided", &[x, y], &c, n);
    }

    fn gemm(&self, a: &CuBuf, b: &CuBuf, c: &CuBuf, g: &Gemm) {
        self.gemm_tc(a, b, c, g, None);
    }

    fn gemm_im2col(&self, a: &CuBuf, x: &CuBuf, c: &CuBuf, g: &Gemm, win: &Win) {
        // Implicit unfolding pays off for the forward (one launch for the whole
        // batch); the weight gradient is faster from an explicit unfold (measured
        // on the PResNet-18 backbone). OJAS_LEARN_IMPLICIT = fwd | dw | all | none.
        let mode = std::env::var("OJAS_LEARN_IMPLICIT").unwrap_or_else(|_| "fwd".into());
        let implicit = match mode.as_str() {
            "none" => false,
            "dw" => g.tb,
            "all" => true,
            _ => !g.tb,
        };
        if self.prec == Prec::F32 || !implicit {
            // the exact path has no implicit variant: unfold, then GEMM
            let rows = win.planes * win.kh * win.kw;
            let col = self.alloc_out(rows * win.oh * win.ow);
            for z in 0..g.batch {
                self.im2col(x, g.ob + z * g.sb, &col, win);
                let one = Gemm { batch: 1, sa: 0, sb: 0, sc: 0, oa: g.oa + z * g.sa, ob: 0, oc: g.oc + z * g.sc, ..*g };
                self.gemm(a, &col, c, &one);
            }
            return;
        }
        self.gemm_tc(a, x, c, g, Some(win));
    }

    fn softmax(&self, x: &CuBuf, y: &CuBuf, rows: usize, cols: usize) {
        self.go("learn_softmax", &[x, y], &[rows as u32, cols as u32], [rows as u32, 1, 1], 256);
    }

    fn softmax_bwd(&self, y: &CuBuf, dy: &CuBuf, dx: &CuBuf, rows: usize, cols: usize, accum: bool) {
        self.go("learn_softmax_bwd", &[y, dy, dx], &[rows as u32, cols as u32, accum as u32], [rows as u32, 1, 1], 256);
    }

    fn layernorm(&self, x: &CuBuf, g: &CuBuf, b: &CuBuf, y: &CuBuf, mean: &CuBuf, rstd: &CuBuf, rows: usize, cols: usize, eps: f32) {
        self.go("learn_layernorm", &[x, g, b, y, mean, rstd], &[rows as u32, cols as u32, f(eps)], [rows as u32, 1, 1], 256);
    }

    fn layernorm_bwd(&self, x: &CuBuf, g: &CuBuf, mean: &CuBuf, rstd: &CuBuf, dy: &CuBuf, dx: &CuBuf, rows: usize, cols: usize, accum: bool) {
        self.go("learn_layernorm_bwd", &[x, g, mean, rstd, dy, dx], &[rows as u32, cols as u32, accum as u32], [rows as u32, 1, 1], 256);
    }

    fn layernorm_wgrad(&self, x: &CuBuf, mean: &CuBuf, rstd: &CuBuf, dy: &CuBuf, dg: &CuBuf, db: &CuBuf, rows: usize, cols: usize, accum: bool) {
        self.go1("learn_layernorm_wgrad", &[x, mean, rstd, dy, dg, db], &[rows as u32, cols as u32, accum as u32], cols);
    }

    fn sum(&self, x: &CuBuf, y: &CuBuf, scale: f32, accum: bool) {
        self.go("learn_sum", &[x, y], &[self.len(x) as u32, f(scale), accum as u32], [1, 1, 1], 1024);
    }

    fn sumsq(&self, x: &CuBuf, y: &CuBuf, accum: bool) {
        self.go("learn_sumsq", &[x, y], &[self.len(x) as u32, accum as u32], [1, 1, 1], 1024);
    }

    fn scale(&self, y: &CuBuf, s: f32) {
        let n = self.len(y);
        self.go1("learn_scale", &[y], &[n as u32, f(s)], n);
    }

    fn bcast_scalar(&self, dy: &CuBuf, dx: &CuBuf, scale: f32, accum: bool) {
        let n = self.len(dx);
        self.go1("learn_bcast_scalar", &[dy, dx], &[n as u32, f(scale), accum as u32], n);
    }

    fn im2col(&self, x: &CuBuf, xoff: usize, col: &CuBuf, g: &Win) {
        let n = g.planes * g.kh * g.kw * g.oh * g.ow;
        self.go1_at("learn_im2col", &[(x, xoff), (col, 0)], &win_consts(g, false), n);
    }

    fn col2im(&self, col: &CuBuf, dx: &CuBuf, dxoff: usize, g: &Win, accum: bool) {
        let mut c = win_consts(g, false);
        c.push(accum as u32);
        self.go1_at("learn_col2im", &[(col, 0), (dx, dxoff)], &c, g.planes * g.h * g.w);
    }

    fn channel_sum(&self, dy: &CuBuf, db: &CuBuf, n: usize, c: usize, hw: usize, accum: bool) {
        self.go("learn_channel_sum", &[dy, db], &[n as u32, c as u32, hw as u32, accum as u32], [c as u32, 1, 1], 256);
    }

    fn bn_stats(&self, x: &CuBuf, mean: &CuBuf, rstd: &CuBuf, n: usize, c: usize, hw: usize, eps: f32) {
        self.go("learn_bn_stats", &[x, mean, rstd], &[n as u32, c as u32, hw as u32, f(eps)], [c as u32, 1, 1], 256);
    }

    fn bn_apply(&self, x: &CuBuf, mean: &CuBuf, rstd: &CuBuf, g: &CuBuf, b: &CuBuf, y: &CuBuf, n: usize, c: usize, hw: usize) {
        let t = n * c * hw;
        self.go1("learn_bn_apply", &[x, mean, rstd, g, b, y], &[t as u32, c as u32, hw as u32], t);
    }

    fn bn_wgrad(&self, x: &CuBuf, mean: &CuBuf, rstd: &CuBuf, dy: &CuBuf, dg: &CuBuf, db: &CuBuf, n: usize, c: usize, hw: usize) {
        self.go("learn_bn_wgrad", &[x, mean, rstd, dy, dg, db], &[n as u32, c as u32, hw as u32], [c as u32, 1, 1], 256);
    }

    fn bn_bwd(&self, x: &CuBuf, mean: &CuBuf, rstd: &CuBuf, g: &CuBuf, dg: &CuBuf, db: &CuBuf, dy: &CuBuf, dx: &CuBuf, n: usize, c: usize, hw: usize, accum: bool) {
        let t = n * c * hw;
        let inv_m = 1.0 / (n * hw) as f32;
        self.go1("learn_bn_bwd", &[x, mean, rstd, g, dg, db, dy, dx], &[t as u32, c as u32, hw as u32, f(inv_m), accum as u32], t);
    }

    fn bn_running(&self, mean: &CuBuf, rstd: &CuBuf, rm: &CuBuf, rv: &CuBuf, c: usize, momentum: f32, eps: f32, unbias: f32) {
        self.go1("learn_bn_running", &[mean, rstd, rm, rv], &[c as u32, f(momentum), f(eps), f(unbias)], c);
    }

    fn maxpool(&self, x: &CuBuf, y: &CuBuf, idx: &CuBuf, g: &Win) {
        self.go1("learn_maxpool", &[x, y, idx], &win_consts(g, true), g.planes * g.oh * g.ow);
    }

    fn maxpool_bwd(&self, dy: &CuBuf, idx: &CuBuf, dx: &CuBuf, g: &Win, accum: bool) {
        let mut c = win_consts(g, true);
        c.push(accum as u32);
        self.go1("learn_maxpool_bwd", &[dy, idx, dx], &c, g.planes * g.h * g.w);
    }

    fn avgpool(&self, x: &CuBuf, y: &CuBuf, g: &Win, count_include_pad: bool) {
        let mut c = win_consts(g, true);
        c.push(count_include_pad as u32);
        self.go1("learn_avgpool", &[x, y], &c, g.planes * g.oh * g.ow);
    }

    fn avgpool_bwd(&self, dy: &CuBuf, dx: &CuBuf, g: &Win, count_include_pad: bool, accum: bool) {
        let mut c = win_consts(g, true);
        c.extend([count_include_pad as u32, accum as u32]);
        self.go1("learn_avgpool_bwd", &[dy, dx], &c, g.planes * g.h * g.w);
    }

    fn upsample(&self, x: &CuBuf, y: &CuBuf, planes: usize, h: usize, w: usize, fy: usize, fx: usize) {
        self.go1("learn_upsample", &[x, y], &[planes as u32, h as u32, w as u32, fy as u32, fx as u32], planes * h * fy * w * fx);
    }

    fn upsample_bwd(&self, dy: &CuBuf, dx: &CuBuf, planes: usize, h: usize, w: usize, fy: usize, fx: usize, accum: bool) {
        self.go1("learn_upsample_bwd", &[dy, dx], &[planes as u32, h as u32, w as u32, fy as u32, fx as u32, accum as u32], planes * h * w);
    }

    fn grid_sample(&self, x: &CuBuf, grid: &CuBuf, y: &CuBuf, n: usize, c: usize, h: usize, w: usize, ho: usize, wo: usize) {
        self.go1("learn_grid_sample", &[x, grid, y], &[n, c, h, w, ho, wo].map(|v| v as u32), n * ho * wo);
    }

    fn grid_sample_bwd(&self, x: &CuBuf, grid: &CuBuf, dy: &CuBuf, dx: Option<&CuBuf>, dgrid: Option<&CuBuf>, n: usize, c: usize, h: usize, w: usize, ho: usize, wo: usize, accum_dgrid: bool) {
        // a missing output gets a harmless stand-in (the kernel skips it by flag)
        let (dxb, dgb) = (dx.unwrap_or(dy), dgrid.unwrap_or(dy));
        let consts = [n, c, h, w, ho, wo, dx.is_some() as usize, dgrid.is_some() as usize, accum_dgrid as usize].map(|v| v as u32);
        self.go1("learn_grid_sample_bwd", &[x, grid, dy, dxb, dgb], &consts, n * ho * wo);
    }

    fn gather_rows(&self, x: &CuBuf, idx: &CuBuf, y: &CuBuf, b: usize, n: usize, k: usize, c: usize) {
        self.go1("learn_gather_rows", &[x, idx, y], &[b, n, k, c].map(|v| v as u32), b * k * c);
    }

    fn gather_rows_bwd(&self, dy: &CuBuf, idx: &CuBuf, dx: &CuBuf, b: usize, n: usize, k: usize, c: usize) {
        self.go1("learn_gather_rows_bwd", &[dy, idx, dx], &[b, n, k, c].map(|v| v as u32), b * k * c);
    }

    fn adamw(&self, p: &CuBuf, g: &CuBuf, m: &CuBuf, v: &CuBuf, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, step: u32) {
        let (bc1, bc2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
        let n = self.len(p);
        self.go1("learn_adamw", &[p, g, m, v], &[n as u32, f(lr), f(b1), f(b2), f(eps), f(wd), f(bc1), f(bc2)], n);
    }
}

/// Window constants. Conv (im2col / col2im): C H W kh kw sh sw pt pl OH OW.
/// Pooling: planes H W OH OW kh kw sh sw pt pl.
fn win_consts(g: &Win, pool: bool) -> Vec<u32> {
    let v = if pool {
        [g.planes, g.h, g.w, g.oh, g.ow, g.kh, g.kw, g.sh, g.sw, g.pt, g.pl]
    } else {
        [g.planes, g.h, g.w, g.kh, g.kw, g.sh, g.sw, g.pt, g.pl, g.oh, g.ow]
    };
    v.iter().map(|&x| x as u32).collect()
}

/// Right-align strides to 6 axes (padding axes get stride 0; their extent is 1).
fn pad_strides(s: &[usize]) -> [usize; MAX_RANK] {
    let mut r = [0; MAX_RANK];
    r[MAX_RANK - s.len()..].copy_from_slice(s);
    r
}
