//! Shared CPU math for all CPU-tier models: weight storage (q8/f32/f16),
//! the SDOT int8 dot kernel, the row-parallel work-stealing matmul, and the
//! numeric helpers — a single copy shared by all the cpu_* models.

use half::f16;
use half::slice::HalfFloatSliceExt;
use std::sync::atomic::Ordering;

/// Big-weight storage. Q8 (per-row symmetric int8 + f32 scale) is the fast
/// path on CPUs with int8 dot-product instructions: decode is byte-bound
/// (~40-53 GB/s streaming ceiling measured), so 1 byte/param beats 4, and NEON
/// SDOT does 16 MACs/instruction. f32 is the exact tier; f16 is a half-RAM
/// fallback for big models.
pub enum W {
    Q8 { q: Vec<i8>, scale: Vec<f32> },
    F32(Vec<f32>),
    F16(Vec<f16>),
    /// Ternary Q2_0 (g128) kept in its raw GGUF form — 2.125 bits/weight
    /// resident instead of 8, a ~3.8x cut in bytes/token on a byte-bound
    /// decode. Dequant happens inside [`dot_q2_0`], so there is no requant
    /// accuracy loss.
    Q20 { raw: Vec<u8> },
}

// ---- persistent worker pool ------------------------------------------------
// `std::thread::scope` spawns and joins on every call. A decode step issues ~253
// matmuls, so at the default width that is ~3k thread creations per token;
// measured spawn/join was ~150 us per matmul, two thirds of the step. These
// workers are created once and parked on a condvar between jobs.

struct PoolState {
    gen: u64,
    quit: bool,
    /// (erased `&F`, trampoline that restores its type and calls it)
    job: Option<(*const (), unsafe fn(*const (), usize, usize))>,
    next_id: usize,
    done: usize,
    nt: usize,
}
// SAFETY: the payload pointer is only dereferenced between posting a job and
// `run` observing done == workers, and `run` does not return until then — so
// the referent outlives every use. `F: Sync` makes the shared &F sound.
unsafe impl Send for PoolState {}

struct PoolInner {
    m: std::sync::Mutex<PoolState>,
    cv_work: std::sync::Condvar,
    cv_done: std::sync::Condvar,
}

pub struct Pool {
    inner: std::sync::Arc<PoolInner>,
    workers: usize,
    /// Held for the duration of a job. The job slot is single-occupancy: a
    /// second caller (another thread driving another model, or parallel tests)
    /// posting while a job runs would repoint the workers at ITS closure while
    /// the first caller still counts their completions — a use-after-return.
    /// A contended caller runs its job alone instead; every body already
    /// supports `(0, 1)` (that is what `parallel` passes for threads <= 1),
    /// and each output keeps its fixed per-element order, so results are
    /// bit-identical either way.
    busy: std::sync::Mutex<()>,
}

unsafe fn trampoline<F: Fn(usize, usize) + Sync>(p: *const (), id: usize, nt: usize) {
    (*(p as *const F))(id, nt)
}

impl Pool {
    fn new(threads: usize) -> Pool {
        let workers = threads.saturating_sub(1); // the caller is worker `workers`
        let inner = std::sync::Arc::new(PoolInner {
            m: std::sync::Mutex::new(PoolState {
                gen: 0, quit: false, job: None, next_id: 0, done: 0, nt: threads,
            }),
            cv_work: std::sync::Condvar::new(),
            cv_done: std::sync::Condvar::new(),
        });
        for _ in 0..workers {
            let inner = inner.clone();
            std::thread::spawn(move || {
                let mut seen = 0u64;
                loop {
                    let (f, p, id, nt) = {
                        let mut st = inner.m.lock().unwrap();
                        while st.gen == seen && !st.quit {
                            st = inner.cv_work.wait(st).unwrap();
                        }
                        if st.quit { return; }
                        seen = st.gen;
                        let id = st.next_id;
                        st.next_id += 1;
                        let (p, f) = st.job.unwrap();
                        (f, p, id, st.nt)
                    };
                    // SAFETY: see `unsafe impl Send for PoolState`.
                    unsafe { f(p, id, nt) };
                    let mut st = inner.m.lock().unwrap();
                    st.done += 1;
                    if st.done == nt - 1 { inner.cv_done.notify_all(); }
                }
            });
        }
        Pool { inner, workers, busy: std::sync::Mutex::new(()) }
    }

    /// Run `f(worker_id, nt)` on every worker plus the calling thread, and block
    /// until all shares finish.
    fn run<F: Fn(usize, usize) + Sync>(&self, f: &F) {
        let Ok(_busy) = self.busy.try_lock() else { f(0, 1); return; };
        let nt = self.workers + 1;
        {
            let mut st = self.inner.m.lock().unwrap();
            st.job = Some((f as *const F as *const (), trampoline::<F>));
            st.gen += 1;
            st.next_id = 0;
            st.done = 0;
            st.nt = nt;
        }
        self.inner.cv_work.notify_all();
        f(self.workers, nt); // caller takes the last share
        let mut st = self.inner.m.lock().unwrap();
        while st.done < self.workers {
            st = self.inner.cv_done.wait(st).unwrap();
        }
    }
}

/// Count the fast cores. Every worker gets an equal shot at the row queue, so a
/// slow core does not merely contribute less, it holds the join. Measured on M2
/// Max (8P+4E): 8 threads 22.7 tok/s vs 12 threads 20.5.
///
/// macOS names the clusters directly. On Linux/Android this reads the per-core
/// max frequencies and drops the slowest cluster (a Tensor G5 is 2x2.25 +
/// 5x3.05 + 1x3.78 GHz, so it keeps the six big cores).
pub fn perf_cores() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.logicalcpu"]).output().ok()?;
        let n: usize = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
        return if n > 0 { Some(n) } else { None };
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut freqs = Vec::new();
        for i in 0..256 {
            let p = format!("/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq");
            match std::fs::read_to_string(&p) {
                Ok(s) => freqs.push(s.trim().parse::<u64>().ok()?),
                Err(_) => break,
            }
        }
        let min = *freqs.iter().min()?;
        let fast = freqs.iter().filter(|&&f| f > min).count();
        // uniform SoC (no little cluster) → use everything
        if fast == 0 { Some(freqs.len()).filter(|&n| n > 0) } else { Some(fast) }
    }
}

static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();

/// Process-wide pool, sized by the first caller (the model's thread count).
fn pool(threads: usize) -> &'static Pool {
    POOL.get_or_init(|| Pool::new(threads.max(1)))
}

/// Run `f(worker_id, nt)` across the shared pool. For callers outside `matmul`
/// that have their own work-stealing loop (per-head attention, say).
pub fn parallel<F: Fn(usize, usize) + Sync>(threads: usize, f: &F) {
    if threads <= 1 { f(0, 1); } else { pool(threads).run(f); }
}

pub fn quant_row_i8(row: &[f32]) -> (Vec<i8>, f32) {
    let m = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let scale = if m > 0.0 { m / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    (row.iter().map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8).collect(), scale)
}


/// y[t][n] = Σ_k x[t][k]·w[n,k] (+bias) for a batch of activations.
/// Row-outer with atomic work-stealing (P/E-core safe); Q8 weights integer-dot
/// against per-vector-quantized activations. Deterministic: each output
/// element is written by exactly one thread, fixed per-dot order.
#[allow(clippy::too_many_arguments)]
pub fn matmul(w: &W, n: usize, k: usize, xs: &[&[f32]], bias: Option<&[f32]>,
              outs: &mut [Vec<f32>], threads: usize, dotprod: bool) {
    let m = xs.len();
    let t_prep = std::time::Instant::now();
    let xq: Vec<(Vec<i8>, f32)> = if matches!(w, W::Q8 { .. } | W::Q20 { .. }) {
        xs.iter().map(|x| quant_row_i8(x)).collect()
    } else { Vec::new() };
    // Ternary rows all read the same activation vector, so de-interleave it once
    // here rather than re-doing a vld4q_s8 in every one of the n row-dots.
    let xp: Vec<(Vec<i8>, Vec<i32>)> = if matches!(w, W::Q20 { .. }) {
        xq.iter().map(|(q, _)| deinterleave4_i8(q)).collect()
    } else { Vec::new() };
    let prep_ns = t_prep.elapsed().as_nanos() as u64;
    // Scoped-spawn cost is ~0.5 ms; below ~4M MACs single-thread wins (a
    // 512-dim decoder matvec is ~26 us of SDOT, so threading it is a loss).
    let nt = if n * k * m < (1 << 22) { 1 } else { threads.min(n.max(1)) };
    let outs_addr = outs.as_mut_ptr() as usize;
    const BLOCK: usize = 16;
    let next = std::sync::atomic::AtomicUsize::new(0);
    {
        let body = |_id: usize, _nt: usize| {
            let next = &next;
            let xq = &xq;
            let xp = &xp;
            {
                let mut scratch = vec![0f32; k];
                loop {
                    let r0 = next.fetch_add(BLOCK, Ordering::Relaxed);
                    if r0 >= n { break; }
                    for row in r0..(r0 + BLOCK).min(n) {
                        let b = bias.map_or(0.0, |b| b[row]);
                        if let W::Q8 { q, scale } = w {
                            let qrow = &q[row * k..(row + 1) * k];
                            let sw = scale[row];
                            for (t, (qx, sx)) in xq.iter().enumerate() {
                                let acc = dot_i8(qrow, qx, dotprod);
                                let s = b + acc as f32 * sw * sx;
                                // SAFETY: each row is claimed by exactly one
                                // thread; threads write disjoint elements.
                                unsafe {
                                    let ov = &mut *(outs_addr as *mut Vec<f32>).add(t);
                                    ov[row] = s;
                                }
                            }
                            continue;
                        }
                        if let W::Q20 { raw } = w {
                            let rb = k / 128 * 34;
                            let rrow = &raw[row * rb..(row + 1) * rb];
                            for (t, (_, sx)) in xq.iter().enumerate() {
                                let s = b + dot_q2_0(rrow, &xp[t].0, &xp[t].1, k, *sx, dotprod);
                                // SAFETY: as above — disjoint writes per row.
                                unsafe {
                                    let ov = &mut *(outs_addr as *mut Vec<f32>).add(t);
                                    ov[row] = s;
                                }
                            }
                            continue;
                        }
                        let wrow: &[f32] = match w {
                            W::F32(v) => &v[row * k..(row + 1) * k],
                            W::F16(v) => {
                                v[row * k..(row + 1) * k].convert_to_f32_slice(&mut scratch);
                                &scratch
                            }
                            W::Q8 { .. } | W::Q20 { .. } => unreachable!(),
                        };
                        for (t, x) in xs.iter().enumerate() {
                            let s = b + dot_f32(wrow, x);
                            // SAFETY: as above — disjoint writes per row.
                            unsafe {
                                let ov = &mut *(outs_addr as *mut Vec<f32>).add(t);
                                ov[row] = s;
                            }
                        }
                    }
                }
            }
        };
        // Every worker runs the same work-stealing loop; `next` hands out row
        // blocks, so the split is dynamic and little cores can't gate a join.
        if nt <= 1 { body(0, 1); } else { pool(threads).run(&body); }
    }
    if PROF.load(Ordering::Relaxed) {
        T_MM.fetch_add(t_prep.elapsed().as_nanos() as u64, Ordering::Relaxed);
        T_PREP.fetch_add(prep_ns, Ordering::Relaxed);
    }
}

// OJAS_CPU_PROF=1 accounting: total time inside matmul, and how much of that was
// the serial activation prep (quantize + de-interleave) before any worker starts.
pub static PROF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static T_MM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static T_PREP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Time in the single-threaded per-head attention loop (scores/softmax/AV).
pub static T_ATTN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn matvec(w: &W, n: usize, k: usize, x: &[f32], bias: Option<&[f32]>,
              threads: usize, dotprod: bool) -> Vec<f32> {
    let mut out = vec![vec![0f32; n]];
    matmul(w, n, k, &[x], bias, &mut out, threads, dotprod);
    out.pop().unwrap()
}

/// Q2_0 (ternary g128) block = 34 B per 128 weights: f16 scale + 32 B of 2-bit
/// codes, packed sequentially (weight j at byte j/4, bits (j%4)*2 — verified
/// against the reference `dequantize_row_q2_0`). Code c dequantizes to (c-1)·scale,
/// so the scale factors out of the block and the inner product is signed
/// accumulate only — no per-weight multiply.
///
/// De-interleave int8 activations by stride 4 into 4 contiguous planes:
/// `out[c*k/4 + j] = xq[4j + c]`.
///
/// Q2_0's sequential packing means the weights extracted at shift `2c` are
/// exactly positions {4j+c}, so they pair with plane `c`. Doing it once per
/// matmul rather than once per output row replaces a `vld4q_s8` (4-way
/// de-interleaving load) per 64 weights per row with a plain contiguous load,
/// over the thousands of rows a matmul runs against one activation.
///
/// Also returns per-128-block activation sums, which let the kernel SDOT the
/// raw codes and correct once per block (`Σ(c-1)·x = Σc·x - Σx`), deleting the
/// four `vsubq_s8` per 64 weights from the inner loop. They are row-independent
/// and so also computed once per matmul.
pub fn deinterleave4_i8(xq: &[i8]) -> (Vec<i8>, Vec<i32>) {
    let q = xq.len() / 4;
    let mut out = vec![0i8; q * 4];
    for j in 0..q {
        let b = j * 4;
        out[j] = xq[b];
        out[q + j] = xq[b + 1];
        out[2 * q + j] = xq[b + 2];
        out[3 * q + j] = xq[b + 3];
    }
    let sums = xq.chunks(128).map(|c| c.iter().map(|&v| v as i32).sum()).collect();
    (out, sums)
}

/// Activations are int8 (one scale for the whole vector, as in the Q8 path) so
/// the inner product runs on SDOT. `xp` is the de-interleaved plane layout from
/// [`deinterleave4_i8`]; `k` is the row length in weights.
pub fn dot_q2_0(row: &[u8], xp: &[i8], bs: &[i32], k: usize, xs: f32, dotprod: bool) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if dotprod {
        // SAFETY: gated on runtime dotprod detection at load.
        return unsafe { dot_q2_0_planes(row, xp, bs, k, xs) };
    }
    let _ = dotprod;
    let q = k / 4;
    let mut total = 0f32;
    for (bi, blk) in row.chunks_exact(34).enumerate() {
        let d = f16b(blk, 0);
        let (mut a0, mut a1, mut a2, mut a3) = (0i32, 0i32, 0i32, 0i32);
        for j in 0..32 {
            let byte = blk[2 + j];
            let p = bi * 32 + j;
            a0 += (byte & 3) as i32 * xp[p] as i32;
            a1 += ((byte >> 2) & 3) as i32 * xp[q + p] as i32;
            a2 += ((byte >> 4) & 3) as i32 * xp[2 * q + p] as i32;
            a3 += (byte >> 6) as i32 * xp[3 * q + p] as i32;
        }
        total += d * (a0 + a1 + a2 + a3 - bs[bi]) as f32;
    }
    total * xs
}

/// NEON ternary dot over pre-de-interleaved activation planes: four plain
/// contiguous loads replace the per-row `vld4q_s8`, leaving the loop at
/// 4 SDOTs + 3 shifts + 3 ands + 4 subs + 5 loads per 64 weights.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q2_0_planes(row: &[u8], xp: &[i8], bs: &[i32], k: usize, xs: f32) -> f32 {
    use std::arch::aarch64::*;
    let three = vdupq_n_u8(3);
    let q = k / 4;
    let p0 = xp.as_ptr();
    let mut total = 0f32;
    for (bi, blk) in row.chunks_exact(34).enumerate() {
        let d = f16b(blk, 0);
        let mut acc = vdupq_n_s32(0);
        for h in 0..2 {
            // 16 code bytes = 64 weights; plane offset advances by 16 lanes.
            // SDOT the raw codes (0..3, safely inside i8) — the -1 per weight is
            // folded into the per-block `bs` correction below.
            let off = bi * 32 + h * 16;
            let c = vld1q_u8(blk.as_ptr().add(2 + h * 16));
            let w0 = vreinterpretq_s8_u8(vandq_u8(c, three));
            let w1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(c, 2), three));
            let w2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(c, 4), three));
            let w3 = vreinterpretq_s8_u8(vshrq_n_u8(c, 6));
            acc = vdotq_s32(acc, w0, vld1q_s8(p0.add(off)));
            acc = vdotq_s32(acc, w1, vld1q_s8(p0.add(q + off)));
            acc = vdotq_s32(acc, w2, vld1q_s8(p0.add(2 * q + off)));
            acc = vdotq_s32(acc, w3, vld1q_s8(p0.add(3 * q + off)));
        }
        total += d * (vaddvq_s32(acc) - bs[bi]) as f32;
    }
    total * xs
}

/// Dequantize one Q2_0 row (`k` weights) to f32 — the embedding-lookup path,
/// which needs the row itself rather than a dot against it.
pub fn dequant_q2_0_row(row: &[u8], k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(k);
    for blk in row.chunks_exact(34) {
        let d = f16b(blk, 0);
        for j in 0..128 {
            let q = (blk[2 + j / 4] >> ((j % 4) * 2)) & 3;
            out.push(d * (q as f32 - 1.0));
        }
    }
    out.truncate(k);
    out
}

/// int8 dot: NEON SDOT (16 MACs/instruction) where available; scalar fallback.
/// Exact integer math → deterministic. i8·i8 over k≤16384 cannot overflow i32.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8], dotprod: bool) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if dotprod {
        // SAFETY: gated on runtime dotprod detection at load.
        return unsafe { dot_i8_sdot(a, b) };
    }
    let _ = dotprod;
    let n = a.len().min(b.len());
    let mut s = 0i32;
    for i in 0..n { s += a[i] as i32 * b[i] as i32; }
    s
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_i8_sdot(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let n32 = n & !31;
    let mut i = 0;
    while i < n32 {
        acc0 = vdotq_s32(acc0, vld1q_s8(a.as_ptr().add(i)), vld1q_s8(b.as_ptr().add(i)));
        acc1 = vdotq_s32(acc1, vld1q_s8(a.as_ptr().add(i + 16)), vld1q_s8(b.as_ptr().add(i + 16)));
        i += 32;
    }
    let mut s = vaddvq_s32(acc0) + vaddvq_s32(acc1);
    while i < n { s += a[i] as i32 * b[i] as i32; i += 1; }
    s
}

/// f32 dot with 4 independent accumulators — LLVM autovectorizes this to
/// NEON/AVX FMA. Fixed summation order → deterministic across runs.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut s0, mut s1, mut s2, mut s3) = (0f32, 0f32, 0f32, 0f32);
    let n4 = n & !3;
    let mut i = 0;
    while i < n4 {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = (s0 + s1) + (s2 + s3);
    while i < n { s += a[i] * b[i]; i += 1; }
    s
}

pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * inv * b).collect()
}

pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for (a, b) in x.iter_mut().zip(w) { *a *= inv * b; }
}

/// LayerNorm with bias — `(x - mean) / sqrt(var + eps) * w + b`.
///
/// Two details are not interchangeable with the obvious alternatives, since
/// this is the CPU oracle the GPU kernels are validated against:
///   * the variance is the population variance (divide by n, not n-1), and
///   * `eps` lives inside the sqrt, not added to the reciprocal afterwards.
/// Both match `ggml_compute_forward_norm_f32` (ggml-cpu/ops.cpp) exactly.
///
/// `eps` is a parameter because the callers disagree: whisper uses 1e-5, the
/// qwen3vl/surya-2 vision tower uses 1e-6 (`clip.vision.attention.layer_norm_epsilon`).
pub fn layernorm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().zip(w.iter().zip(b)).map(|(v, (w, b))| (v - mean) * inv * w + b).collect()
}

pub fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// GELU, tanh approximation — transcribed from `ggml_gelu_f32`
/// (ggml-cpu/vec.h): `0.5x(1 + tanh(sqrt(2/pi)·x·(1 + 0.044715x²)))`.
///
/// Not the erf form. The two differ by ~1e-3 around |x| ≈ 2, and
/// `kernels/gemm_fat.rs:626` records that a more accurate variant moved a
/// logits checksum by 0.20%. Every model whose reference uses the tanh form —
/// CPU and Metal (`kernels/prelude.rs` `ffn_act(g, 1u)`) — must use this
/// expression in this association order. Models trained with PyTorch's default
/// `nn.GELU()` (ModernBERT) use [`gelu_erf`] instead.
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_560_802_865_4;
    const COEF_A: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + COEF_A * x * x)).tanh())
}

/// GELU, exact form, and the f64 error function it is built on. They live in
/// `ojas-core` so that the Laya host code (`ojas-models`) and this oracle share one
/// implementation; the Metal kernel's `erf_as` is an independent approximation.
pub use ojas_core::math::{erf, gelu_erf};

/// NeoX-style RoPE in place on a per-head vector (head_dim), position `pos`.
pub fn rope(v: &mut [f32], head_dim: usize, pos: usize, base: f32) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
        let ang = pos as f32 * freq;
        let (s, c) = ang.sin_cos();
        let x0 = v[i];
        let x1 = v[i + half];
        v[i] = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
}

pub fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    let mut bv = f32::MIN;
    for (i, &x) in v.iter().enumerate() {
        if x > bv { bv = x; bi = i; }
    }
    bi
}


// ===================== direct K-quant dots =====================
// Dot GGUF Q4_K / Q5_K rows directly against block-quantized activations — no
// dequant, no requant, raw bytes stay cache-resident. Activation format is
// Q8_K-flavoured: per-256 superblock f32 scale + i8[256] + per-32 block sums
// (the sums turn the k-quant `dmin` term into a precomputed correction).

/// Q8_K-style quantized activation (built once per matmul).
pub struct Q8kAct {
    pub scales: Vec<f32>, // per 256-superblock
    pub q: Vec<i8>,
    pub bsums: Vec<i32>,  // 8 per superblock (per-32 sums)
}

pub fn quant_q8k(x: &[f32]) -> Q8kAct {
    let nsb = x.len().div_ceil(256);
    let mut scales = Vec::with_capacity(nsb);
    let mut q = vec![0i8; nsb * 256];
    let mut bsums = vec![0i32; nsb * 8];
    for sb in 0..nsb {
        let seg = &x[sb * 256..(sb * 256 + 256).min(x.len())];
        let m = seg.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let scale = if m > 0.0 { m / 127.0 } else { 1.0 };
        let inv = 1.0 / scale;
        scales.push(scale);
        for (i, &v) in seg.iter().enumerate() {
            let qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
            q[sb * 256 + i] = qi;
            bsums[sb * 8 + i / 32] += qi as i32;
        }
    }
    Q8kAct { scales, q, bsums }
}

#[inline]
fn scale_min_k4(j: usize, s: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((s[j] & 63) as f32, (s[j + 4] & 63) as f32)
    } else {
        let d = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        let m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
        (d as f32, m as f32)
    }
}

#[inline]
fn f16b(b: &[u8], off: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[off], b[off + 1]])).to_f32()
}

/// 32-element integer dot: u8 nibble-expanded weights (0..15 or 0..31) × i8.
#[inline]
fn idot32(w: &[u8; 32], x: &[i8], dotprod: bool) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if dotprod {
        // SAFETY: gated on runtime dotprod detection.
        return unsafe { idot32_sdot(w, x) };
    }
    let _ = dotprod;
    let mut s = 0i32;
    for i in 0..32 { s += w[i] as i32 * x[i] as i32; }
    s
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn idot32_sdot(w: &[u8; 32], x: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    // weights are 0..31 → fit i8; reinterpret and SDOT
    let w0 = vld1q_s8(w.as_ptr() as *const i8);
    let w1 = vld1q_s8(w.as_ptr().add(16) as *const i8);
    let x0 = vld1q_s8(x.as_ptr());
    let x1 = vld1q_s8(x.as_ptr().add(16));
    let acc = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1);
    vaddvq_s32(acc)
}

/// Dot one Q4_K row (raw GGUF bytes, 144 B / 256 weights) against a Q8_K act.
pub fn dot_q4k(row: &[u8], act: &Q8kAct, dotprod: bool) -> f32 {
    let mut total = 0f32;
    for (sb, blk) in row.chunks_exact(144).enumerate() {
        let d = f16b(blk, 0);
        let dmin = f16b(blk, 2);
        let sc = &blk[4..16];
        let qs = &blk[16..144];
        let sx = act.scales[sb];
        let xq = &act.q[sb * 256..(sb + 1) * 256];
        let bs = &act.bsums[sb * 8..(sb + 1) * 8];
        let mut sumd = 0f32; // d   · Σ sc_j · idot_j
        let mut summ = 0f32; // dmin· Σ  m_j · bsum_j
        let mut lo = [0u8; 32];
        let mut hi = [0u8; 32];
        for c in 0..4 {
            let q = &qs[c * 32..c * 32 + 32];
            for l in 0..32 { lo[l] = q[l] & 0xF; hi[l] = q[l] >> 4; }
            let (s1, m1) = scale_min_k4(c * 2, sc);
            let (s2, m2) = scale_min_k4(c * 2 + 1, sc);
            sumd += s1 * idot32(&lo, &xq[c * 64..c * 64 + 32], dotprod) as f32;
            sumd += s2 * idot32(&hi, &xq[c * 64 + 32..c * 64 + 64], dotprod) as f32;
            summ += m1 * bs[c * 2] as f32 + m2 * bs[c * 2 + 1] as f32;
        }
        total += sx * (d * sumd - dmin * summ);
    }
    total
}

/// Dot one Q5_K row (raw GGUF bytes, 176 B / 256 weights) against a Q8_K act.
pub fn dot_q5k(row: &[u8], act: &Q8kAct, dotprod: bool) -> f32 {
    let mut total = 0f32;
    for (sb, blk) in row.chunks_exact(176).enumerate() {
        let d = f16b(blk, 0);
        let dmin = f16b(blk, 2);
        let sc = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        let sx = act.scales[sb];
        let xq = &act.q[sb * 256..(sb + 1) * 256];
        let bs = &act.bsums[sb * 8..(sb + 1) * 8];
        let mut sumd = 0f32;
        let mut summ = 0f32;
        let mut lo = [0u8; 32];
        let mut hi = [0u8; 32];
        let (mut u1, mut u2) = (1u8, 2u8);
        for c in 0..4 {
            let q = &qs[c * 32..c * 32 + 32];
            for l in 0..32 {
                lo[l] = (q[l] & 0xF) | if qh[l] & u1 != 0 { 16 } else { 0 };
                hi[l] = (q[l] >> 4) | if qh[l] & u2 != 0 { 16 } else { 0 };
            }
            let (s1, m1) = scale_min_k4(c * 2, sc);
            let (s2, m2) = scale_min_k4(c * 2 + 1, sc);
            sumd += s1 * idot32(&lo, &xq[c * 64..c * 64 + 32], dotprod) as f32;
            sumd += s2 * idot32(&hi, &xq[c * 64 + 32..c * 64 + 64], dotprod) as f32;
            summ += m1 * bs[c * 2] as f32 + m2 * bs[c * 2 + 1] as f32;
            u1 <<= 2; u2 <<= 2;
        }
        total += sx * (d * sumd - dmin * summ);
    }
    total
}

/// Dot one Q6_K row (raw GGUF bytes, 210 B / 256 weights) against a Q8_K act.
/// Q6_K: { ql[128] low-4, qh[64] hi-2, i8 scales[16] per-16, half d } — no dmin.
pub fn dot_q6k(row: &[u8], act: &Q8kAct, dotprod: bool) -> f32 {
    let mut total = 0f32;
    let mut w = [0u8; 32]; // 6-bit values 0..63 fit u8/i8 for the int dot
    for (sb, blk) in row.chunks_exact(210).enumerate() {
        let d = f16b(blk, 208);
        let sx = act.scales[sb];
        let xq = &act.q[sb * 256..(sb + 1) * 256];
        let bs = &act.bsums[sb * 8..(sb + 1) * 8];
        let mut sum = 0f32;   // Σ sc·idot per 16-elem group
        let mut sub = 0f32;   // the −32 correction term (per-16 activation sums)
        // scales are per-16; a per-32 idot mixes two scales — handle by splitting each
        // 32-run into two 16-halves for the scale/bsum bookkeeping.
        for half_i in 0..2 {
            let ql = &blk[half_i * 64..half_i * 64 + 64];
            let qh = &blk[128 + half_i * 32..128 + half_i * 32 + 32];
            let sc = &blk[192 + half_i * 8..192 + half_i * 8 + 8];
            let xh = &xq[half_i * 128..half_i * 128 + 128];
            for quarter in 0..4 {
                // reconstruct the 32 weights of this quarter (rows y[l+32*quarter])
                for l in 0..32 {
                    w[l] = match quarter {
                        0 => (ql[l] & 0xF) | ((qh[l] & 3) << 4),
                        1 => (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4),
                        2 => (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4),
                        _ => (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4),
                    };
                }
                let x32 = &xh[quarter * 32..quarter * 32 + 32];
                // per-16 scales: scale index = is + 2*quarter (is = l/16)
                let s0 = sc[quarter * 2] as i8 as f32;
                let s1 = sc[quarter * 2 + 1] as i8 as f32;
                let (mut i0, mut i1) = (0i32, 0i32);
                let (mut x0, mut x1) = (0i32, 0i32);
                for l in 0..16 {
                    i0 += w[l] as i32 * x32[l] as i32;
                    x0 += x32[l] as i32;
                    i1 += w[l + 16] as i32 * x32[l + 16] as i32;
                    x1 += x32[l + 16] as i32;
                }
                let _ = dotprod; // 16-lane groups: scalar path (dotprod unused here)
                sum += s0 * i0 as f32 + s1 * i1 as f32;
                sub += s0 * x0 as f32 + s1 * x1 as f32;
            }
            let _ = bs;
        }
        total += sx * d * (sum - 32.0 * sub);
    }
    total
}

/// IQ2_XXS (2.0625 bpw, GGUF type 16): per 256 weights, f16 d + 32×u16.
/// Each 32-weight group is 2 u32s: aux0 = 4 grid-index bytes (256-entry
/// 8-byte-magnitude codebook), aux1 = 4×7-bit sign words + 4-bit scale.
/// f32 dot (no activation quant — these rows are memory-bound streamed).
pub fn dot_iq2xxs(row: &[u8], x: &[f32]) -> f32 {
    use ojas_formats::iq_tables::{IQ2XXS_GRID, KSIGNS_IQ2XS};
    let mut sum = 0f32;
    for (b, xb) in x.chunks_exact(256).enumerate() {
        let blk = &row[b * 66..(b + 1) * 66];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qs = &blk[2..66];
        let mut bsum = 0f32;
        for g in 0..8usize {
            let a0 = u32::from_le_bytes([qs[8 * g], qs[8 * g + 1], qs[8 * g + 2], qs[8 * g + 3]]);
            let a1 = u32::from_le_bytes([qs[8 * g + 4], qs[8 * g + 5], qs[8 * g + 6], qs[8 * g + 7]]);
            let db = (0.5 + (a1 >> 28) as f32) * 0.25;
            let mut gsum = 0f32;
            for l in 0..4usize {
                let grid = IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize];
                let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                let xg = &xb[g * 32 + l * 8..g * 32 + l * 8 + 8];
                for j in 0..8usize {
                    let mag = ((grid >> (8 * j)) & 255) as f32;
                    let v = if (signs >> j) & 1 != 0 { -mag } else { mag };
                    gsum += v * xg[j];
                }
            }
            bsum += db * gsum;
        }
        sum += d * bsum;
    }
    sum
}

/// Reference-shape dequant (validation + repacker use): one row of k weights.
pub fn dequant_iq2xxs(row: &[u8], k: usize) -> Vec<f32> {
    use ojas_formats::iq_tables::{IQ2XXS_GRID, KSIGNS_IQ2XS};
    let mut y = Vec::with_capacity(k);
    for b in 0..k / 256 {
        let blk = &row[b * 66..(b + 1) * 66];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qs = &blk[2..66];
        for g in 0..8usize {
            let a0 = u32::from_le_bytes([qs[8 * g], qs[8 * g + 1], qs[8 * g + 2], qs[8 * g + 3]]);
            let a1 = u32::from_le_bytes([qs[8 * g + 4], qs[8 * g + 5], qs[8 * g + 6], qs[8 * g + 7]]);
            let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
            for l in 0..4usize {
                let grid = IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize];
                let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                for j in 0..8usize {
                    let mag = ((grid >> (8 * j)) & 255) as f32;
                    y.push(db * if (signs >> j) & 1 != 0 { -mag } else { mag });
                }
            }
        }
    }
    y
}

/// IQ3_XXS (3.0625 bpw, type 18): 98 B/256 = f16 d + 64 grid-index bytes
/// (u32 grid, 4 magnitudes each; 8 per 32-group) + 8×u32 scales/signs.
pub fn dequant_iq3xxs(row: &[u8], k: usize) -> Vec<f32> {
    use ojas_formats::iq_tables::{IQ3XXS_GRID, KSIGNS_IQ2XS};
    let mut y = Vec::with_capacity(k);
    for b in 0..k / 256 {
        let blk = &row[b * 98..(b + 1) * 98];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qs = &blk[2..66];
        let ss = &blk[66..98];
        for g in 0..8usize {
            let aux = u32::from_le_bytes([ss[4 * g], ss[4 * g + 1], ss[4 * g + 2], ss[4 * g + 3]]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4usize {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[qs[8 * g + 2 * l] as usize];
                let g2 = IQ3XXS_GRID[qs[8 * g + 2 * l + 1] as usize];
                for j in 0..4usize {
                    let m = ((g1 >> (8 * j)) & 255) as f32;
                    y.push(db * if (signs >> j) & 1 != 0 { -m } else { m });
                }
                for j in 0..4usize {
                    let m = ((g2 >> (8 * j)) & 255) as f32;
                    y.push(db * if (signs >> (j + 4)) & 1 != 0 { -m } else { m });
                }
            }
        }
    }
    y
}

/// IQ2_S (2.5625 bpw, type 22): 82 B/256 = f16 d + qs[64] (32 grid-low +
/// 32 raw sign bytes) + qh[8] (2 high grid bits per entry) + scales[8]
/// (two 4-bit halves per 32-group).
/// Dispatch dequant by GGUF type (IQ family) — for re-encode simulations.
pub fn dequant_iq(ggml_type: u32, row: &[u8], k: usize) -> Vec<f32> {
    match ggml_type {
        16 => dequant_iq2xxs(row, k),
        18 => dequant_iq3xxs(row, k),
        22 => dequant_iq2s(row, k),
        23 => dequant_iq4xs(row, k),
        t => panic!("dequant_iq: unsupported type {t}"),
    }
}

pub fn dequant_iq2s(row: &[u8], k: usize) -> Vec<f32> {
    use ojas_formats::iq_tables::IQ2S_GRID;
    let mut y = Vec::with_capacity(k);
    for b in 0..k / 256 {
        let blk = &row[b * 82..(b + 1) * 82];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let (qs, signs, qh, scales) = (&blk[2..34], &blk[34..66], &blk[66..74], &blk[74..82]);
        for g in 0..8usize {
            let db = [
                d * (0.5 + (scales[g] & 0xf) as f32) * 0.25,
                d * (0.5 + (scales[g] >> 4) as f32) * 0.25,
            ];
            for l in 0..4usize {
                let dl = db[l / 2];
                let gi = qs[4 * g + l] as usize | (((qh[g] as usize) << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[gi];
                let sb = signs[4 * g + l];
                for j in 0..8usize {
                    let m = ((grid >> (8 * j)) & 255) as f32;
                    y.push(dl * if (sb >> j) & 1 != 0 { -m } else { m });
                }
            }
        }
    }
    y
}

/// IQ4_XS (4.25 bpw, type 23): 136 B/256 = f16 d + u16 scales_h +
/// scales_l[4] + qs[128]; 6-bit scale − 32, 16-entry non-linear LUT,
/// nibbles map to lanes j and j+16.
pub fn dequant_iq4xs(row: &[u8], k: usize) -> Vec<f32> {
    use ojas_formats::iq_tables::KVALUES_IQ4NL;
    let mut y = Vec::with_capacity(k);
    for b in 0..k / 256 {
        let blk = &row[b * 136..(b + 1) * 136];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let sh = u16::from_le_bytes([blk[2], blk[3]]);
        let sl = &blk[4..8];
        let qs = &blk[8..136];
        for g in 0..8usize {
            let ls = ((sl[g / 2] >> (4 * (g % 2))) & 0xf) as i32 | ((((sh >> (2 * g)) & 3) as i32) << 4);
            let dl = d * (ls - 32) as f32;
            let mut lane = [0f32; 32];
            for j in 0..16usize {
                let q = qs[16 * g + j];
                lane[j] = dl * KVALUES_IQ4NL[(q & 0xf) as usize] as f32;
                lane[j + 16] = dl * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
            }
            y.extend_from_slice(&lane);
        }
    }
    y
}

/// Direct (allocation-free) f32 dots per IQ format — decode in registers,
/// accumulate against x. These run per expert row on the streaming hot path.
pub fn dot_iq3xxs(row: &[u8], x: &[f32]) -> f32 {
    use ojas_formats::iq_tables::{IQ3XXS_GRID, KSIGNS_IQ2XS};
    let mut sum = 0f32;
    for (b, xb) in x.chunks_exact(256).enumerate() {
        let blk = &row[b * 98..(b + 1) * 98];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let (qs, ss) = (&blk[2..66], &blk[66..98]);
        let mut bsum = 0f32;
        for g in 0..8usize {
            let aux = u32::from_le_bytes([ss[4 * g], ss[4 * g + 1], ss[4 * g + 2], ss[4 * g + 3]]);
            let db = 0.5 + (aux >> 28) as f32;
            let mut gsum = 0f32;
            for l in 0..4usize {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[qs[8 * g + 2 * l] as usize];
                let g2 = IQ3XXS_GRID[qs[8 * g + 2 * l + 1] as usize];
                let xg = &xb[g * 32 + l * 8..g * 32 + l * 8 + 8];
                for j in 0..4usize {
                    let m1 = ((g1 >> (8 * j)) & 255) as f32;
                    let m2 = ((g2 >> (8 * j)) & 255) as f32;
                    gsum += if (signs >> j) & 1 != 0 { -m1 } else { m1 } * xg[j];
                    gsum += if (signs >> (j + 4)) & 1 != 0 { -m2 } else { m2 } * xg[j + 4];
                }
            }
            bsum += db * gsum;
        }
        sum += d * 0.5 * bsum;
    }
    sum
}

pub fn dot_iq2s(row: &[u8], x: &[f32]) -> f32 {
    use ojas_formats::iq_tables::IQ2S_GRID;
    let mut sum = 0f32;
    for (b, xb) in x.chunks_exact(256).enumerate() {
        let blk = &row[b * 82..(b + 1) * 82];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let (qs, signs, qh, scales) = (&blk[2..34], &blk[34..66], &blk[66..74], &blk[74..82]);
        let mut bsum = 0f32;
        for g in 0..8usize {
            let db = [0.5 + (scales[g] & 0xf) as f32, 0.5 + (scales[g] >> 4) as f32];
            for l in 0..4usize {
                let gi = qs[4 * g + l] as usize | (((qh[g] as usize) << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[gi];
                let sb = signs[4 * g + l];
                let xg = &xb[g * 32 + l * 8..g * 32 + l * 8 + 8];
                let mut gsum = 0f32;
                for j in 0..8usize {
                    let m = ((grid >> (8 * j)) & 255) as f32;
                    gsum += if (sb >> j) & 1 != 0 { -m } else { m } * xg[j];
                }
                bsum += db[l / 2] * gsum;
            }
        }
        sum += d * 0.25 * bsum;
    }
    sum
}

pub fn dot_iq4xs(row: &[u8], x: &[f32]) -> f32 {
    use ojas_formats::iq_tables::KVALUES_IQ4NL;
    let mut sum = 0f32;
    for (b, xb) in x.chunks_exact(256).enumerate() {
        let blk = &row[b * 136..(b + 1) * 136];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let sh = u16::from_le_bytes([blk[2], blk[3]]);
        let sl = &blk[4..8];
        let qs = &blk[8..136];
        let mut bsum = 0f32;
        for g in 0..8usize {
            let ls = ((sl[g / 2] >> (4 * (g % 2))) & 0xf) as i32 | ((((sh >> (2 * g)) & 3) as i32) << 4);
            let xg = &xb[g * 32..g * 32 + 32];
            let mut gsum = 0f32;
            for j in 0..16usize {
                let q = qs[16 * g + j];
                gsum += KVALUES_IQ4NL[(q & 0xf) as usize] as f32 * xg[j];
                gsum += KVALUES_IQ4NL[(q >> 4) as usize] as f32 * xg[j + 16];
            }
            bsum += (ls - 32) as f32 * gsum;
        }
        sum += d * bsum;
    }
    sum
}

/// Q8_0 dot against f32 activations: 32 weights per 34-byte block, one f16 scale
/// then 32 int8. The raw GGUF layout, not the requantized int8 + f32 row scale.
pub fn dot_q80(row: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0f32;
    for (b, xb) in x.chunks_exact(32).enumerate() {
        let blk = &row[b * 34..(b + 1) * 34];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let mut acc = 0f32;
        for j in 0..32usize { acc += (blk[2 + j] as i8) as f32 * xb[j]; }
        sum += d * acc;
    }
    sum
}

/// IQ4_NL dot against f32 activations.
///
/// The cheapest IQ format to decode: 32 weights per 18-byte block — one f16
/// scale and 16 packed nibbles — with no superblock and no per-group sub-scale,
/// so the nibble IS the codebook index. `dot_iq4xs` above is the same codebook
/// wrapped in a 256-weight superblock with 6-bit group scales.
///
/// Layout trap: the 16 low nibbles produce the first 16 outputs and the 16 high
/// nibbles the next 16 — split, not interleaved (the q4_0 convention). Mirrors
/// `gguf::dequant_to_f16`'s type-20 arm, which is checked against gguf-py.
pub fn dot_iq4nl(row: &[u8], x: &[f32]) -> f32 {
    use ojas_formats::iq_tables::KVALUES_IQ4NL;
    let mut sum = 0f32;
    for (b, xb) in x.chunks_exact(32).enumerate() {
        let blk = &row[b * 18..(b + 1) * 18];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qs = &blk[2..18];
        let mut bsum = 0f32;
        for j in 0..16usize {
            let q = qs[j];
            bsum += KVALUES_IQ4NL[(q & 0xf) as usize] as f32 * xb[j];
            bsum += KVALUES_IQ4NL[(q >> 4) as usize] as f32 * xb[j + 16];
        }
        sum += d * bsum;
    }
    sum
}

/// IQ3_S dot against f32 activations.
///
/// 256 weights per 110-byte block: `{ half d; u8 qs[64]; u8 qh[8]; u8 signs[32];
/// u8 scales[4] }`. Three details differ from the other IQ types, and missing
/// any of them gives a silently wrong answer:
///   * the grid has 512 entries, and the 9th index bit comes from `qh`
///   * signs are raw bits (via KMASK_IQ2XS), not a ksigns lookup
///   * the 4-bit scale is applied as `(1 + 2s)`, not as `s` or `s - 32`
///
/// Indexing collapses neatly per 32-weight group `g`: `qs` offset `8g`, `signs`
/// offset `4g`, `qh[g]`, and the scale nibble `scales[g/2] >> 4(g&1)`. Mirrors
/// `gguf::dequant_to_f16`'s type-21 arm.
pub fn dot_iq3s(row: &[u8], x: &[f32]) -> f32 {
    use ojas_formats::iq_tables::{IQ3S_GRID, KMASK_IQ2XS};
    let mut sum = 0f32;
    for (b, xb) in x.chunks_exact(256).enumerate() {
        let blk = &row[b * 110..(b + 1) * 110];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let (qs, qh, sg, sc) = (&blk[2..66], &blk[66..74], &blk[74..106], &blk[106..110]);
        let mut bsum = 0f32;
        for g in 0..8usize {
            let h = qh[g] as usize;
            let (qo, so) = (8 * g, 4 * g);
            let xg = &xb[g * 32..g * 32 + 32];
            let mut gsum = 0f32;
            for l in 0..4usize {
                let g1 = IQ3S_GRID[qs[qo + 2 * l] as usize | ((h << (8 - 2 * l)) & 256)].to_le_bytes();
                let g2 = IQ3S_GRID[qs[qo + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256)].to_le_bytes();
                let s = sg[so + l];
                for j in 0..4usize {
                    let s1 = if s & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let s2 = if s & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    gsum += g1[j] as f32 * s1 * xg[8 * l + j];
                    gsum += g2[j] as f32 * s2 * xg[8 * l + j + 4];
                }
            }
            bsum += (1.0 + 2.0 * ((sc[g >> 1] >> (4 * (g & 1))) & 0xF) as f32) * gsum;
        }
        sum += d * bsum;
    }
    sum
}

/// SDOT-path IQ2_XXS dot vs quantized activations: decode ±grid magnitudes
/// into an i8 lane buffer (pure integers), SDOT each 32-group against the
/// q8 activations, one float multiply per group (the dot_q4k shape).
pub fn dot_iq2xxs_q8k(row: &[u8], act: &Q8kAct, dotprod: bool) -> f32 {
    use ojas_formats::iq_tables::{IQ2XXS_GRID, KSIGNS_IQ2XS};
    let mut sum = 0f32;
    let nsb = act.scales.len();
    let mut w = [0i8; 256];
    for b in 0..nsb {
        let blk = &row[b * 66..(b + 1) * 66];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qs = &blk[2..66];
        let mut sc = [0f32; 8];
        for g in 0..8usize {
            let a0 = u32::from_le_bytes([qs[8 * g], qs[8 * g + 1], qs[8 * g + 2], qs[8 * g + 3]]);
            let a1 = u32::from_le_bytes([qs[8 * g + 4], qs[8 * g + 5], qs[8 * g + 6], qs[8 * g + 7]]);
            sc[g] = (0.5 + (a1 >> 28) as f32) * 0.25;
            for l in 0..4usize {
                let grid = IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize];
                let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                let base = g * 32 + l * 8;
                for j in 0..8usize {
                    let m = ((grid >> (8 * j)) & 255) as i8;
                    w[base + j] = if (signs >> j) & 1 != 0 { -m } else { m };
                }
            }
        }
        let xq = &act.q[b * 256..(b + 1) * 256];
        let mut bsum = 0f32;
        for g in 0..8usize {
            let id = dot_i8(&w[g * 32..(g + 1) * 32], &xq[g * 32..(g + 1) * 32], dotprod);
            bsum += sc[g] * id as f32;
        }
        sum += d * act.scales[b] * bsum;
    }
    sum
}

/// SDOT-path IQ3_XXS dot vs quantized activations (same shape; 3-bit grid).
pub fn dot_iq3xxs_q8k(row: &[u8], act: &Q8kAct, dotprod: bool) -> f32 {
    use ojas_formats::iq_tables::{IQ3XXS_GRID, KSIGNS_IQ2XS};
    let mut sum = 0f32;
    let nsb = act.scales.len();
    let mut w = [0i8; 256];
    for b in 0..nsb {
        let blk = &row[b * 98..(b + 1) * 98];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let (qs, ss) = (&blk[2..66], &blk[66..98]);
        let mut sc = [0f32; 8];
        for g in 0..8usize {
            let aux = u32::from_le_bytes([ss[4 * g], ss[4 * g + 1], ss[4 * g + 2], ss[4 * g + 3]]);
            sc[g] = (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4usize {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[qs[8 * g + 2 * l] as usize];
                let g2 = IQ3XXS_GRID[qs[8 * g + 2 * l + 1] as usize];
                let base = g * 32 + l * 8;
                for j in 0..4usize {
                    let m1 = ((g1 >> (8 * j)) & 255) as i8;
                    let m2 = ((g2 >> (8 * j)) & 255) as i8;
                    w[base + j] = if (signs >> j) & 1 != 0 { -m1 } else { m1 };
                    w[base + j + 4] = if (signs >> (j + 4)) & 1 != 0 { -m2 } else { m2 };
                }
            }
        }
        let xq = &act.q[b * 256..(b + 1) * 256];
        let mut bsum = 0f32;
        for g in 0..8usize {
            let id = dot_i8(&w[g * 32..(g + 1) * 32], &xq[g * 32..(g + 1) * 32], dotprod);
            bsum += sc[g] * id as f32;
        }
        sum += d * act.scales[b] * bsum;
    }
    sum
}

/// f32 dot over any supported raw IQ row (slow path; SDOT variants above).
pub fn dot_iq(ggml_type: u32, row: &[u8], x: &[f32]) -> f32 {
    match ggml_type {
        16 => dot_iq2xxs(row, x),
        18 => dot_iq3xxs(row, x),
        8  => dot_q80(row, x),
        20 => dot_iq4nl(row, x),
        21 => dot_iq3s(row, x),
        22 => dot_iq2s(row, x),
        23 => dot_iq4xs(row, x),
        t => panic!("dot_iq: unsupported type {t}"),
    }
}

/// Row-parallel matvec over RAW K-quant rows (Q4_K 12 / Q5_K 13 / Q6_K 14).
pub fn matvec_kq(raw: &[u8], ggml_type: u32, n: usize, k: usize, x: &[f32],
                 threads: usize, dotprod: bool) -> Vec<f32> {
    let act = quant_q8k(x);
    let row_bytes = k / 256 * match ggml_type {
        12 => 144,
        13 => 176,
        14 => 210,
        16 => 66,  // IQ2_XXS
        18 => 98,  // IQ3_XXS
        8  => 272, // Q8_0 — 34 B/32 weights, i.e. 272 B per 256
        20 => 144, // IQ4_NL
        21 => 110, // IQ3_S — 18 B/32 weights, i.e. 144 B per 256, so the k/256 form holds
        22 => 82,  // IQ2_S
        23 => 136, // IQ4_XS
        t => panic!("matvec_kq: unsupported GGUF type {t}"),
    };
    let mut out = vec![0f32; n];
    let nt = threads.min(n.max(1));
    let out_addr = out.as_mut_ptr() as usize;
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..nt {
            let next = &next;
            let act = &act;
            sc.spawn(move || loop {
                let r0 = next.fetch_add(8, Ordering::Relaxed);
                if r0 >= n { break; }
                for row in r0..(r0 + 8).min(n) {
                    let rb = &raw[row * row_bytes..(row + 1) * row_bytes];
                    let v = match ggml_type {
                        12 => dot_q4k(rb, act, dotprod),
                        13 => dot_q5k(rb, act, dotprod),
                        16 => dot_iq2xxs_q8k(rb, act, dotprod),
                        18 => dot_iq3xxs_q8k(rb, act, dotprod),
                        8 | 20 | 21 | 22 | 23 => dot_iq(ggml_type, rb, x),
                        _ => dot_q6k(rb, act, dotprod),
                    };
                    // SAFETY: each row is written by exactly one thread.
                    unsafe { *(out_addr as *mut f32).add(row) = v; }
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod iq4nl_tests {
    use super::dot_iq4nl;
    use ojas_formats::iq_tables::KVALUES_IQ4NL;

    /// Build one IQ4_NL block: f16 scale `d_bits`, then 16 nibble pairs.
    fn block(d_bits: u16, qs: [u8; 16]) -> Vec<u8> {
        let mut b = d_bits.to_le_bytes().to_vec();
        b.extend_from_slice(&qs);
        b
    }

    /// The 16 low nibbles feed outputs 0..16 and the 16 high nibbles feed
    /// outputs 16..32; interleaving them instead still produces plausible
    /// numbers.
    #[test]
    fn low_nibbles_are_the_first_half() {
        const ONE: u16 = 0x3C00; // f16 1.0 — leaves the codebook value bare
        // qs[0] = 0x93 -> low nibble 3, high nibble 9.
        let mut qs = [0u8; 16];
        qs[0] = 0x93;
        let row = block(ONE, qs);

        // one-hot at position 0 must select the low nibble of qs[0]
        let mut x = vec![0f32; 32];
        x[0] = 1.0;
        assert_eq!(dot_iq4nl(&row, &x), KVALUES_IQ4NL[3] as f32);

        // one-hot at position 16 must select the high nibble of qs[0]
        let mut x = vec![0f32; 32];
        x[16] = 1.0;
        assert_eq!(dot_iq4nl(&row, &x), KVALUES_IQ4NL[9] as f32);
    }

    /// IQ3_S against the same independent decoder. Random bytes are a valid block
    /// for this format (every `qs` value indexes the grid, `qh` only adds a bit,
    /// and signs/scales are unconstrained), so this sweeps the whole 512-entry
    /// grid rather than the handful a real tensor happens to use.
    #[test]
    fn iq3s_agrees_with_dequant_reference() {
        use super::dot_iq3s;
        let (nblk, k) = (6usize, 6 * 256usize);
        let mut lcg: u32 = 0x9e37_79b9;
        let mut rnd = || { lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); lcg };
        let mut row: Vec<u8> = (0..nblk * 110).map(|_| (rnd() >> 11) as u8).collect();
        // Random bytes are a valid payload but not a valid scale — an arbitrary f16
        // bit pattern is NaN/Inf often enough to poison the comparison, so pin the
        // two scale bytes per block to exact values.
        let scales: [u16; 4] = [0x3C00, 0x3800, 0x3400, 0xBE00]; // 1.0, 0.5, 0.25, -1.5
        for b in 0..nblk {
            row[b * 110..b * 110 + 2].copy_from_slice(&scales[b % scales.len()].to_le_bytes());
        }
        let x: Vec<f32> = (0..k).map(|_| (rnd() >> 8) as f32 / 8_388_608.0 - 1.0).collect();

        let deq = ojas_formats::gguf::dequant_to_f16(&row, 21, k);
        let mut want = 0f32;
        for i in 0..k {
            let w = half::f16::from_le_bytes([deq[2 * i], deq[2 * i + 1]]).to_f32();
            want += w * x[i];
        }
        let got = dot_iq3s(&row, &x);
        let tol = 1e-3 * want.abs().max(1.0);
        assert!((got - want).abs() <= tol, "dot_iq3s {got} vs dequant reference {want}");
    }

    /// Cross-check against `gguf::dequant_to_f16`, a separately written decoder
    /// whose type-20 arm is checked against gguf-py. Agreement is close but not
    /// exact: that path rounds each `d * kvalue` to f16 before the dot, while
    /// this one accumulates in f32. The bound is loose enough to absorb that and
    /// far too tight for a decode bug, which misses by orders of magnitude.
    #[test]
    fn agrees_with_dequant_reference() {
        let (nblk, k) = (12usize, 12 * 32usize);
        let mut lcg: u32 = 0x1234_5678;
        let mut rnd = || { lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); lcg };

        let scales: [u16; 4] = [0x3C00, 0x3800, 0x3400, 0xBE00]; // 1.0, 0.5, 0.25, -1.5
        let mut row = Vec::with_capacity(nblk * 18);
        for b in 0..nblk {
            let mut qs = [0u8; 16];
            for q in qs.iter_mut() { *q = (rnd() >> 13) as u8; }
            row.extend_from_slice(&block(scales[b % scales.len()], qs));
        }
        let x: Vec<f32> = (0..k).map(|_| (rnd() >> 8) as f32 / 8_388_608.0 - 1.0).collect();

        let deq = ojas_formats::gguf::dequant_to_f16(&row, 20, k);
        let mut want = 0f32;
        for i in 0..k {
            let w = half::f16::from_le_bytes([deq[2 * i], deq[2 * i + 1]]).to_f32();
            want += w * x[i];
        }
        let got = dot_iq4nl(&row, &x);
        let tol = 1e-3 * want.abs().max(1.0);
        assert!((got - want).abs() <= tol, "dot_iq4nl {got} vs dequant reference {want}");
    }
}
