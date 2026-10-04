//! Metal backend: the `cnn_train` kernel family (`ojas-metal` `kernels/cnn_train.rs`), f32. Every
//! launch goes through `KernelRuntime::dispatch`, the same contract `ojas-learn/src/cuda.rs` uses,
//! so this file mirrors it call for call.
//!
//! Launches are recorded into one open command buffer instead of each paying a commit and a wait:
//! a training step is thousands of small kernels, and a CPU-GPU round trip per kernel costs more
//! than most of them take to run. The buffer is committed every [`FLUSH_EVERY`] dispatches so the
//! GPU starts while the host keeps encoding, and the host waits only when it reads a result
//! ([`Backend::download`]) or calls [`Metal::sync`].
//!
//! Why this is safe without further bookkeeping:
//! - `dispatch` closes every launch with a buffer-scope memory barrier, and command buffers on one
//!   queue run in commit order, so each kernel sees every earlier kernel's writes.
//! - Command buffers retain the buffers they reference, so a buffer the tape drops while work on
//!   it is still queued stays alive until that work completes.
//! - The host writes only into buffers it has just created (`upload`), which no queued work can
//!   reference; every host read goes through `download`, which drains the queue first.
//!
//! Buffers are recycled ([`LBuf`]): creating and releasing a Metal buffer is a kernel call each
//! way, and a step allocates thousands. A dropped buffer goes back to a per-length free list and
//! `alloc` hands it out again. Queued work may still reference it under its previous owner, which
//! is harmless because the new owner only reaches it through later kernels (zero-fill included),
//! ordered behind that work; `upload`, the one host write, always takes a fresh buffer.
//!
//! `alloc`'s zero-fill is deferred until something reads the buffer. Most zeroed buffers are
//! gradients, and when the first contribution covers the whole buffer it simply stores instead
//! of accumulating, so neither the fill launch nor the read of zeros happens (`Metal::claim`).
//!
//! GEMMs run on the simdgroup matrix units where the GPU has them (`cnn_train_mma`), splitting
//! K when a product has too few output tiles to fill the GPU.

use anyhow::Result;
use ojas_core::{Device, KernelRuntime};
use ojas_metal::{MBuf, MetalEnc, MetalGpu};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::rc::Rc;

use crate::backend::{pad6, Backend, Bcast, Binary, Gemm, Unary, Win, MAX_RANK};

/// Dispatches recorded before the open command buffer is committed. Small enough that the GPU
/// starts early in a long step, large enough that commit overhead stays negligible.
const FLUSH_EVERY: usize = 128;

/// Released buffers by length, waiting to be handed out again.
#[derive(Default)]
struct Pool {
    free: HashMap<usize, Vec<MBuf>>,
    bytes: usize,
    /// Most bytes kept on the free lists; a buffer released past this is freed instead.
    cap: usize,
}

impl Pool {
    fn take(&mut self, n: usize) -> Option<MBuf> {
        let b = self.free.get_mut(&n)?.pop()?;
        self.bytes -= n * 4;
        Some(b)
    }

    fn put(&mut self, b: MBuf) {
        if self.bytes + b.len * 4 <= self.cap {
            self.bytes += b.len * 4;
            self.free.entry(b.len).or_default().push(b);
        }
    }
}

/// A training buffer: an `MBuf` that returns to its backend's free list when dropped.
pub struct LBuf {
    buf: ManuallyDrop<MBuf>,
    pool: Rc<RefCell<Pool>>,
    /// Promised zero by `alloc` but not yet written: the fill is deferred (see `Metal::claim`).
    zero: Cell<bool>,
}

impl Deref for LBuf {
    type Target = MBuf;
    fn deref(&self) -> &MBuf {
        &self.buf
    }
}

impl Drop for LBuf {
    fn drop(&mut self) {
        // SAFETY: `buf` is never touched again after this.
        let b = unsafe { ManuallyDrop::take(&mut self.buf) };
        self.pool.borrow_mut().put(b);
    }
}

pub struct Metal {
    pub gpu: MetalGpu,
    pool: Rc<RefCell<Pool>>,
    /// The command buffer being recorded, if any.
    open: RefCell<Option<MetalEnc>>,
    /// Dispatches recorded into `open`.
    queued: Cell<usize>,
    /// Committed command buffers not yet known to have completed.
    inflight: RefCell<Vec<ojas_metal::metal::CommandBuffer>>,
    /// The simdgroup-matrix GEMMs (`cnn_train_mma`) are compiled: Apple family 7+.
    mma: bool,
    /// Split-K partial sums, grown on demand and shared by every split GEMM: queued GEMMs
    /// run in order, so each one's partials are consumed before the next overwrites them.
    scratch: RefCell<Option<LBuf>>,
    /// Per-kernel GPU time (profile mode): name -> (ms, launches).
    profile: Option<RefCell<HashMap<String, (f64, usize)>>>,
}

impl Metal {
    pub fn new() -> Result<Self> {
        let mut gpu = MetalGpu::new()?;
        gpu.ensure_family("cnn_train")?;
        let mma = gpu.device.supports_family(ojas_metal::metal::MTLGPUFamily::Apple7);
        if mma {
            gpu.register_family("cnn_train_mma", ojas_metal::kernels::cnn_train::MMA_BODY);
            gpu.ensure_family("cnn_train_mma")?;
        }
        // Cache at most a quarter of the working set the device recommends, so recycled
        // buffers never crowd out whatever else shares the GPU.
        let cap = (gpu.device.recommended_max_working_set_size() / 4) as usize;
        let pool = Rc::new(RefCell::new(Pool { cap, ..Pool::default() }));
        Ok(Metal { gpu, pool, open: RefCell::new(None), queued: Cell::new(0), inflight: RefCell::new(vec![]), mma, scratch: RefCell::new(None), profile: None })
    }

    /// Profile mode: every launch is committed and waited for in its own command buffer and
    /// that buffer's GPU execution time charged to the kernel name. Much slower than the
    /// batched default; read with [`Self::take_profile`].
    pub fn profiled() -> Result<Self> {
        let mut m = Self::new()?;
        m.profile = Some(RefCell::new(HashMap::new()));
        Ok(m)
    }

    /// Per-kernel time since the last call (profile mode), largest first: (name, ms, launches).
    pub fn take_profile(&self) -> Vec<(String, f64, usize)> {
        let Some(p) = &self.profile else { return vec![] };
        let mut v: Vec<_> = p.borrow_mut().drain().map(|(k, (t, n))| (k, t, n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// Free every cached buffer (the allocations a finished workload no longer needs).
    pub fn release_cached(&self) {
        let mut p = self.pool.borrow_mut();
        p.free.clear();
        p.bytes = 0;
    }

    fn wrap(&self, b: MBuf) -> LBuf {
        LBuf { buf: ManuallyDrop::new(b), pool: self.pool.clone(), zero: Cell::new(false) }
    }

    /// Commit everything recorded so far and wait for the GPU to finish it. Panics if a
    /// command buffer failed: the outputs it should have written are not there.
    pub fn sync(&self) {
        self.commit_open();
        for cb in self.inflight.borrow_mut().drain(..) {
            if let Err(e) = ojas_metal::wait_checked(&cb, "ojas-learn") {
                panic!("Metal training command buffer failed: {e:?}");
            }
        }
    }

    /// End and commit the open command buffer without waiting for it.
    fn commit_open(&self) {
        if let Some(enc) = self.open.borrow_mut().take() {
            enc.enc.end_encoding();
            enc.cb.commit();
            let mut inflight = self.inflight.borrow_mut();
            inflight.retain(|cb| cb.status() != ojas_metal::metal::MTLCommandBufferStatus::Completed);
            inflight.push(enc.cb);
        }
        self.queued.set(0);
    }

    /// Settle a deferred zero-fill for an output the next launch overwrites in full, or that
    /// it accumulates into in full: the fill is skipped, and true tells the caller to store
    /// instead of accumulate (adding to zeros is storing). False for a buffer with no fill
    /// pending, one the launch covers only partly, or one it also reads (`reads`), which
    /// `go_at` zero-fills first.
    fn claim(&self, b: &LBuf, whole: bool, reads: &[&LBuf]) -> bool {
        whole && !reads.iter().any(|r| std::ptr::eq(*r, b)) && b.zero.replace(false)
    }

    /// [`Self::claim`] over several outputs written under one `accum` flag: all of them or none.
    fn claim_all(&self, bs: &[&LBuf], whole: bool, reads: &[&LBuf]) -> bool {
        let aliased = bs.iter().any(|b| reads.iter().any(|r| std::ptr::eq(*r, *b)));
        if whole && !aliased && bs.iter().all(|b| b.zero.get()) {
            bs.iter().for_each(|b| b.zero.set(false));
            return true;
        }
        false
    }

    fn go(&self, name: &str, bufs: &[&LBuf], consts: &[u32], grid: [u32; 3], block: u32) {
        let b: Vec<(&LBuf, usize)> = bufs.iter().map(|b| (*b, 0)).collect();
        self.go_at(name, &b, consts, grid, block);
    }

    /// Record one launch, buffers at element offsets.
    fn go_at(&self, name: &str, bufs: &[(&LBuf, usize)], consts: &[u32], grid: [u32; 3], block: u32) {
        // any deferred zero-fill this launch did not claim is due now
        for &(b, _) in bufs {
            if b.zero.replace(false) {
                self.go1("learn_fill", &[b], &[b.len as u32, f(0.0)], b.len);
            }
        }
        let b: Vec<(&MBuf, u64)> = bufs.iter().map(|&(b, off)| (&**b, off as u64 * 4)).collect();
        {
            let mut open = self.open.borrow_mut();
            let enc = open.get_or_insert_with(|| self.gpu.begin());
            self.gpu
                .dispatch(enc, name, &b, consts, grid, [block, 1, 1])
                .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        }
        self.queued.set(self.queued.get() + 1);
        if let Some(p) = &self.profile {
            let cb = self.open.borrow().as_ref().map(|e| e.cb.clone());
            self.sync();
            let ms = cb.map_or(0.0, |cb| ojas_metal::gpu_seconds(&cb) * 1e3);
            let mut m = p.borrow_mut();
            let e = m.entry(name.to_string()).or_insert((0.0, 0));
            e.0 += ms;
            e.1 += 1;
        } else if self.queued.get() >= FLUSH_EVERY {
            self.commit_open();
        }
    }

    fn go1_at(&self, name: &str, bufs: &[(&LBuf, usize)], consts: &[u32], n: usize) {
        if n > 0 {
            self.go_at(name, bufs, consts, [n.div_ceil(256) as u32, 1, 1], 256);
        }
    }

    /// One thread per element.
    fn go1(&self, name: &str, bufs: &[&LBuf], consts: &[u32], n: usize) {
        if n > 0 {
            self.go(name, bufs, consts, [n.div_ceil(256) as u32, 1, 1], 256);
        }
    }
}

impl Drop for Metal {
    /// An encoder released without `endEncoding` is a Metal assertion, and queued work may
    /// still be writing buffers the caller is about to drop.
    fn drop(&mut self) {
        self.commit_open();
        for cb in self.inflight.borrow_mut().drain(..) {
            cb.wait_until_completed();
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

/// The same strided copy over fewer axes: unit axes dropped, and an axis merged into the
/// next one wherever both sides step through them as one contiguous run.
fn fold_axes(dims: &[usize], xs: &[usize], ys: &[usize]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let (mut d, mut x, mut y): (Vec<usize>, Vec<usize>, Vec<usize>) = (vec![], vec![], vec![]);
    for i in 0..dims.len() {
        if dims[i] == 1 {
            continue;
        }
        if let (Some(&px), Some(&py)) = (x.last(), y.last()) {
            if px == xs[i] * dims[i] && py == ys[i] * dims[i] {
                *d.last_mut().unwrap() *= dims[i];
                *x.last_mut().unwrap() = xs[i];
                *y.last_mut().unwrap() = ys[i];
                continue;
            }
        }
        d.push(dims[i]);
        x.push(xs[i]);
        y.push(ys[i]);
    }
    (d, x, y)
}

fn pad_strides(s: &[usize]) -> [usize; MAX_RANK] {
    let mut r = [0; MAX_RANK];
    r[MAX_RANK - s.len()..].copy_from_slice(s);
    r
}

impl Backend for Metal {
    type Buf = LBuf;

    fn trim(&self) {
        self.sync();
        self.release_cached();
    }

    fn alloc(&self, n: usize) -> LBuf {
        // `newBufferWithLength:` does not promise zeroed contents, recycled buffers hold old
        // data, and this trait promises zeros. The fill runs on the GPU (the host would first
        // have to wait for queued work on a recycled buffer) and is deferred: most zeroed
        // buffers are gradients whose first writer covers them in full, and then it is
        // dropped altogether (`claim`).
        let b = self.alloc_out(n);
        b.zero.set(true);
        b
    }
    fn alloc_out(&self, n: usize) -> LBuf {
        let n = n.max(1);
        let b = self.pool.borrow_mut().take(n);
        self.wrap(b.unwrap_or_else(|| self.gpu.alloc(n)))
    }
    fn upload(&self, v: &[f32]) -> LBuf {
        if v.is_empty() {
            return self.alloc(1);
        }
        self.wrap(self.gpu.upload(v))
    }
    fn download(&self, b: &LBuf) -> Vec<f32> {
        if b.zero.get() {
            return vec![0.0; b.len];
        }
        self.sync();
        self.gpu.read(b)
    }
    fn len(&self, b: &LBuf) -> usize {
        b.len
    }

    fn fill(&self, y: &LBuf, v: f32) {
        self.claim(y, true, &[]);
        self.go1("learn_fill", &[y], &[self.len(y) as u32, f(v)], self.len(y));
    }

    fn axpby(&self, x: &LBuf, y: &LBuf, a: f32, b: f32) {
        let n = self.len(y);
        let b = if self.claim(y, true, &[x]) { 0.0 } else { b };
        self.go1("learn_axpby", &[x, y], &[n as u32, f(a), f(b)], n);
    }

    fn unary(&self, op: Unary, x: &LBuf, y: &LBuf) {
        let n = self.len(y);
        self.go1("learn_unary", &[x, y], &[n as u32, op as u32], n);
    }

    fn unary_bwd(&self, op: Unary, x: &LBuf, y: &LBuf, dy: &LBuf, dx: &LBuf, accum: bool) {
        let n = self.len(dx);
        let accum = !self.claim(dx, true, &[x, y, dy]) && accum;
        self.go1("learn_unary_bwd", &[x, y, dy, dx], &[n as u32, op as u32, accum as u32], n);
    }

    fn binary(&self, op: Binary, a: &LBuf, b: &LBuf, y: &LBuf, bc: &Bcast) {
        self.go1("learn_binary", &[a, b, y], &bcast_consts(bc, op as u32), bc.n);
    }

    fn binary_grad(&self, op: Binary, which: u8, a: &LBuf, b: &LBuf, dy: &LBuf, t: &LBuf, bc: &Bcast) {
        let mut c = vec![which as u32];
        c.extend(bcast_consts(bc, op as u32));
        self.go1("learn_binary_grad", &[a, b, dy, t], &c, bc.n);
    }

    fn reduce_to(&self, t: &LBuf, full: &[usize], g: &LBuf, target: &[usize], accum: bool) {
        let (fd, td) = (pad6(full), pad6(target));
        let mut red = 0u32;
        for d in 0..MAX_RANK {
            if td[d] == 1 && fd[d] != 1 {
                red |= 1 << d;
            }
        }
        let m: usize = target.iter().product();
        let accum = !self.claim(g, m == g.len, &[t]) && accum;
        // reduced axes all before the kept ones (bias-like [R, C] -> [C], or nothing reduced):
        // column sums, split over rows when there are many
        let first_kept = (0..MAX_RANK).find(|&d| red >> d & 1 == 0 && fd[d] != 1).unwrap_or(MAX_RANK);
        let prefix = (first_kept..MAX_RANK).all(|d| red >> d & 1 == 0);
        let r: usize = fd.iter().product::<usize>() / m.max(1);
        if prefix && m >= 1 {
            if r < 64 {
                self.go1("learn_reduce_rows_final", &[t, g], &[m as u32, r as u32, accum as u32], m);
                return;
            }
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

    fn copy_strided(&self, x: &LBuf, xoff: usize, xs: &[usize], y: &LBuf, yoff: usize, ys: &[usize], dims: &[usize], accum: bool) {
        // distinct destinations, as many as y holds, from its start: y is covered in full
        let accum = !self.claim(y, yoff == 0 && dims.iter().product::<usize>() == y.len, &[x]) && accum;
        let (mut dims, mut xs, mut ys) = fold_axes(dims, xs, ys);
        // float4 at a time when the innermost axis is a contiguous run of whole, aligned quads
        let r = dims.len();
        let quads = r > 0
            && xs[r - 1] == 1
            && ys[r - 1] == 1
            && dims[r - 1].is_multiple_of(4)
            && xoff.is_multiple_of(4)
            && yoff.is_multiple_of(4)
            && xs[..r - 1].iter().chain(&ys[..r - 1]).all(|s| s.is_multiple_of(4));
        if quads {
            dims[r - 1] /= 4;
            xs[r - 1] = 4;
            ys[r - 1] = 4;
        }
        let n: usize = dims.iter().product();
        let (d, sx, sy) = (pad6(&dims), pad_strides(&xs), pad_strides(&ys));
        let mut c = vec![n as u32, accum as u32];
        c.extend(d.iter().map(|&v| v as u32));
        c.extend(sx.iter().map(|&v| v as u32));
        c.extend(sy.iter().map(|&v| v as u32));
        c.extend([xoff as u32, yoff as u32]);
        self.go1(if quads { "learn_copy_strided4" } else { "learn_copy_strided" }, &[x, y], &c, n);
    }

    fn gemm(&self, a: &LBuf, b: &LBuf, c: &LBuf, g: &Gemm) {
        let whole = g.oc == 0 && (g.batch == 1 || g.sc == g.m * g.n) && g.batch * g.m * g.n == c.len;
        let g = &Gemm { beta: if self.claim(c, whole, &[a, b]) { 0.0 } else { g.beta }, ..*g };
        let grid = [g.n.div_ceil(64) as u32, g.m.div_ceil(64) as u32, g.batch as u32];
        if !self.mma {
            let consts = [
                g.m as u32, g.n as u32, g.k as u32, g.ta as u32, g.tb as u32,
                g.sa as u32, g.sb as u32, g.sc as u32, f(g.alpha), f(g.beta),
            ];
            self.go_at("learn_gemm", &[(a, g.oa), (b, g.ob), (c, g.oc)], &consts, grid, 256);
            return;
        }
        let name = match (g.ta, g.tb) {
            (false, false) => "learn_gemm_mma_nn",
            (false, true) => "learn_gemm_mma_nt",
            (true, false) => "learn_gemm_mma_tn",
            (true, true) => "learn_gemm_mma_tt",
        };
        // Split K when the output tiles alone cannot fill the GPU, keeping at least
        // MIN_KBLOCKS runs of 32 along K per split so each threadgroup still streams a long run.
        const TARGET_GROUPS: usize = 192;
        const MIN_KBLOCKS: usize = 4;
        let tiles = grid[0] as usize * grid[1] as usize * g.batch;
        let kblocks = g.k.div_ceil(32).max(1);
        let want = if tiles < TARGET_GROUPS { TARGET_GROUPS.div_ceil(tiles) } else { 1 };
        let per = kblocks.div_ceil(want.min(kblocks / MIN_KBLOCKS).max(1));
        let splits = kblocks.div_ceil(per);
        let kchunk = per * 32;
        // float4 loads need every row start 16-byte aligned: base offset, batch stride and
        // row length all multiples of 4 floats.
        let aligned = |off: usize, stride: usize, ld: usize| off.is_multiple_of(4) && (g.batch == 1 || stride.is_multiple_of(4)) && ld.is_multiple_of(4);
        let vec = aligned(g.oa, g.sa, if g.ta { g.m } else { g.k }) as u32
            | (aligned(g.ob, g.sb, if g.tb { g.k } else { g.n }) as u32) << 1
            | (aligned(g.oc, g.sc, g.n) as u32) << 2;
        let consts = [
            g.m as u32, g.n as u32, g.k as u32, g.sa as u32, g.sb as u32, g.sc as u32, f(g.alpha), f(g.beta),
            splits as u32, kchunk as u32, vec,
        ];
        if splits == 1 {
            self.go_at(name, &[(a, g.oa), (b, g.ob), (c, g.oc), (c, g.oc)], &consts, grid, 128);
            return;
        }
        let parts = g.batch * splits * g.m * g.n;
        let mut scratch = self.scratch.borrow_mut();
        if scratch.as_ref().is_none_or(|s| s.len < parts) {
            // Queued work still holding the old buffer keeps it alive (command buffers
            // retain what they reference); later work is ordered behind it by the barriers.
            *scratch = Some(self.alloc_out(parts));
        }
        let p = scratch.as_ref().unwrap();
        let split_grid = [grid[0], grid[1], (g.batch * splits) as u32];
        self.go_at(name, &[(a, g.oa), (b, g.ob), (c, g.oc), (p, 0)], &consts, split_grid, 128);
        let rc = [g.m as u32, g.n as u32, g.batch as u32, g.sc as u32, splits as u32, f(g.alpha), f(g.beta)];
        self.go1_at("learn_gemm_mma_reduce", &[(p, 0), (c, g.oc)], &rc, g.batch * g.m * g.n);
    }

    fn softmax(&self, x: &LBuf, y: &LBuf, rows: usize, cols: usize) {
        self.go("learn_softmax", &[x, y], &[rows as u32, cols as u32], [rows as u32, 1, 1], 256);
    }

    fn softmax_bwd(&self, y: &LBuf, dy: &LBuf, dx: &LBuf, rows: usize, cols: usize, accum: bool) {
        let accum = !self.claim(dx, rows * cols == dx.len, &[y, dy]) && accum;
        self.go("learn_softmax_bwd", &[y, dy, dx], &[rows as u32, cols as u32, accum as u32], [rows as u32, 1, 1], 256);
    }

    fn layernorm(&self, x: &LBuf, g: &LBuf, b: &LBuf, y: &LBuf, mean: &LBuf, rstd: &LBuf, rows: usize, cols: usize, eps: f32) {
        self.go("learn_layernorm", &[x, g, b, y, mean, rstd], &[rows as u32, cols as u32, f(eps)], [rows as u32, 1, 1], 256);
    }

    fn layernorm_bwd(&self, x: &LBuf, g: &LBuf, mean: &LBuf, rstd: &LBuf, dy: &LBuf, dx: &LBuf, rows: usize, cols: usize, accum: bool) {
        let accum = !self.claim(dx, rows * cols == dx.len, &[x, g, mean, rstd, dy]) && accum;
        self.go("learn_layernorm_bwd", &[x, g, mean, rstd, dy, dx], &[rows as u32, cols as u32, accum as u32], [rows as u32, 1, 1], 256);
    }

    fn layernorm_wgrad(&self, x: &LBuf, mean: &LBuf, rstd: &LBuf, dy: &LBuf, dg: &LBuf, db: &LBuf, rows: usize, cols: usize, accum: bool) {
        let accum = !self.claim_all(&[dg, db], cols == dg.len && cols == db.len, &[x, mean, rstd, dy]) && accum;
        self.go1("learn_layernorm_wgrad", &[x, mean, rstd, dy, dg, db], &[rows as u32, cols as u32, accum as u32], cols);
    }

    fn sum(&self, x: &LBuf, y: &LBuf, scale: f32, accum: bool) {
        self.go("learn_sum", &[x, y], &[self.len(x) as u32, f(scale), accum as u32], [1, 1, 1], 1024);
    }

    fn sumsq(&self, x: &LBuf, y: &LBuf, accum: bool) {
        self.go("learn_sumsq", &[x, y], &[self.len(x) as u32, accum as u32], [1, 1, 1], 1024);
    }

    fn scale(&self, y: &LBuf, s: f32) {
        let n = self.len(y);
        self.go1("learn_scale", &[y], &[n as u32, f(s)], n);
    }

    fn bcast_scalar(&self, dy: &LBuf, dx: &LBuf, scale: f32, accum: bool) {
        let n = self.len(dx);
        let accum = !self.claim(dx, true, &[dy]) && accum;
        self.go1("learn_bcast_scalar", &[dy, dx], &[n as u32, f(scale), accum as u32], n);
    }

    fn im2col(&self, x: &LBuf, xoff: usize, col: &LBuf, g: &Win) {
        let n = g.planes * g.kh * g.kw * g.oh * g.ow;
        self.go1_at("learn_im2col", &[(x, xoff), (col, 0)], &win_consts(g, false), n);
    }

    fn col2im(&self, col: &LBuf, dx: &LBuf, dxoff: usize, g: &Win, accum: bool) {
        let mut c = win_consts(g, false);
        c.push(accum as u32);
        self.go1_at("learn_col2im", &[(col, 0), (dx, dxoff)], &c, g.planes * g.h * g.w);
    }

    fn channel_sum(&self, dy: &LBuf, db: &LBuf, n: usize, c: usize, hw: usize, accum: bool) {
        self.go("learn_channel_sum", &[dy, db], &[n as u32, c as u32, hw as u32, accum as u32], [c as u32, 1, 1], 256);
    }

    fn bn_stats(&self, x: &LBuf, mean: &LBuf, rstd: &LBuf, n: usize, c: usize, hw: usize, eps: f32) {
        self.go("learn_bn_stats", &[x, mean, rstd], &[n as u32, c as u32, hw as u32, f(eps)], [c as u32, 1, 1], 256);
    }

    fn bn_apply(&self, x: &LBuf, mean: &LBuf, rstd: &LBuf, g: &LBuf, b: &LBuf, y: &LBuf, n: usize, c: usize, hw: usize) {
        let t = n * c * hw;
        self.go1("learn_bn_apply", &[x, mean, rstd, g, b, y], &[t as u32, c as u32, hw as u32], t);
    }

    fn bn_wgrad(&self, x: &LBuf, mean: &LBuf, rstd: &LBuf, dy: &LBuf, dg: &LBuf, db: &LBuf, n: usize, c: usize, hw: usize) {
        self.go("learn_bn_wgrad", &[x, mean, rstd, dy, dg, db], &[n as u32, c as u32, hw as u32], [c as u32, 1, 1], 256);
    }

    fn bn_bwd(&self, x: &LBuf, mean: &LBuf, rstd: &LBuf, g: &LBuf, dg: &LBuf, db: &LBuf, dy: &LBuf, dx: &LBuf, n: usize, c: usize, hw: usize, accum: bool) {
        let t = n * c * hw;
        let inv_m = 1.0 / (n * hw) as f32;
        self.go1("learn_bn_bwd", &[x, mean, rstd, g, dg, db, dy, dx], &[t as u32, c as u32, hw as u32, f(inv_m), accum as u32], t);
    }

    fn bn_running(&self, mean: &LBuf, rstd: &LBuf, rm: &LBuf, rv: &LBuf, c: usize, momentum: f32, eps: f32, unbias: f32) {
        self.go1("learn_bn_running", &[mean, rstd, rm, rv], &[c as u32, f(momentum), f(eps), f(unbias)], c);
    }

    fn maxpool(&self, x: &LBuf, y: &LBuf, idx: &LBuf, g: &Win) {
        self.go1("learn_maxpool", &[x, y, idx], &win_consts(g, true), g.planes * g.oh * g.ow);
    }

    fn maxpool_bwd(&self, dy: &LBuf, idx: &LBuf, dx: &LBuf, g: &Win, accum: bool) {
        let mut c = win_consts(g, true);
        c.push(accum as u32);
        self.go1("learn_maxpool_bwd", &[dy, idx, dx], &c, g.planes * g.h * g.w);
    }

    fn avgpool(&self, x: &LBuf, y: &LBuf, g: &Win, count_include_pad: bool) {
        let mut c = win_consts(g, true);
        c.push(count_include_pad as u32);
        self.go1("learn_avgpool", &[x, y], &c, g.planes * g.oh * g.ow);
    }

    fn avgpool_bwd(&self, dy: &LBuf, dx: &LBuf, g: &Win, count_include_pad: bool, accum: bool) {
        let mut c = win_consts(g, true);
        c.extend([count_include_pad as u32, accum as u32]);
        self.go1("learn_avgpool_bwd", &[dy, dx], &c, g.planes * g.h * g.w);
    }

    fn upsample(&self, x: &LBuf, y: &LBuf, planes: usize, h: usize, w: usize, fy: usize, fx: usize) {
        self.go1("learn_upsample", &[x, y], &[planes as u32, h as u32, w as u32, fy as u32, fx as u32], planes * h * fy * w * fx);
    }

    fn upsample_bwd(&self, dy: &LBuf, dx: &LBuf, planes: usize, h: usize, w: usize, fy: usize, fx: usize, accum: bool) {
        self.go1("learn_upsample_bwd", &[dy, dx], &[planes as u32, h as u32, w as u32, fy as u32, fx as u32, accum as u32], planes * h * w);
    }

    fn grid_sample(&self, x: &LBuf, grid: &LBuf, y: &LBuf, n: usize, c: usize, h: usize, w: usize, ho: usize, wo: usize) {
        self.go1("learn_grid_sample", &[x, grid, y], &[n, c, h, w, ho, wo].map(|v| v as u32), n * ho * wo);
    }

    fn grid_sample_bwd(&self, x: &LBuf, grid: &LBuf, dy: &LBuf, dx: Option<&LBuf>, dgrid: Option<&LBuf>, n: usize, c: usize, h: usize, w: usize, ho: usize, wo: usize, accum_dgrid: bool) {
        // a missing output gets a harmless stand-in (the kernel skips it by flag)
        let (dxb, dgb) = (dx.unwrap_or(dy), dgrid.unwrap_or(dy));
        let consts = [n, c, h, w, ho, wo, dx.is_some() as usize, dgrid.is_some() as usize, accum_dgrid as usize].map(|v| v as u32);
        self.go1("learn_grid_sample_bwd", &[x, grid, dy, dxb, dgb], &consts, n * ho * wo);
    }

    fn gather_rows(&self, x: &LBuf, idx: &LBuf, y: &LBuf, b: usize, n: usize, k: usize, c: usize) {
        self.go1("learn_gather_rows", &[x, idx, y], &[b, n, k, c].map(|v| v as u32), b * k * c);
    }

    fn gather_rows_bwd(&self, dy: &LBuf, idx: &LBuf, dx: &LBuf, b: usize, n: usize, k: usize, c: usize) {
        self.go1("learn_gather_rows_bwd", &[dy, idx, dx], &[b, n, k, c].map(|v| v as u32), b * k * c);
    }

    fn adamw(&self, p: &LBuf, g: &LBuf, m: &LBuf, v: &LBuf, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, step: u32) {
        let (bc1, bc2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
        let n = self.len(p);
        self.go1("learn_adamw", &[p, g, m, v], &[n as u32, f(lr), f(b1), f(b2), f(eps), f(wd), f(bc1), f(bc2)], n);
    }
}
