//! What a surya-2-shaped decode pass costs at M rows instead of 1.
//!
//! A surya-2 page is ~2400 output tokens and decode is ~88% of the page cost, so the
//! metric for a thousand-page volume is tokens/second aggregated over pages, not the
//! latency of one page. Sequence batching pays only if the per-token weight read — the
//! same ~353 MB (Q4L) / ~1129 MB (F16) for every sequence — can be shared across N
//! sequences inside one kernel launch, so this probe measures that read at M rows.
//!
//! The model is not loaded: the weight bytes are a fixed pattern and the outputs are
//! garbage, only the timing is read. What is measured is the memory system and the
//! kernel's M-scaling on the exact (K,N) shapes surya-2 decodes through, with one
//! distinct buffer per layer so the 24-layer stream exceeds any cache. Re-dispatching
//! one layer 24 times would measure the SLC rather than DRAM and flatter every M
//! equally.
//!
//! Three parts:
//!   1. `roof`   — a pure streaming read over the same footprint: the machine's
//!                 achievable read bandwidth, which every other result is a fraction of.
//!   2. `shapes` — per-(K,N) ms and GB/s at M = 1,2,4,8, showing which projections
//!                 amortize and which are already latency-floored.
//!   3. `step`   — the whole per-token weight read, all 24 layers + lm_head chained in
//!                 one command buffer, at M = 1,2,4,8. The ratio M * t(1) / t(M) is the
//!                 throughput multiplier sequence batching could deliver for the
//!                 weight-read term.
//!
//! Kernel selection mirrors `decoder/dispatch.rs` (`mm` for M=1, the m-row /
//! lane-partitioned family for 2..=8, `gemm_mm_*` above that), so the numbers describe
//! the code that would actually run.
//!
//! usage: ocr_batch_roof [f16|q4l|both] [roof|shapes|step|all]

use anyhow::Result;
use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use std::collections::HashMap;
use std::ffi::c_void;

/// Pure streaming read. `acc ^= src[i]` over uint4 so the compiler cannot drop
/// the loads, one write per simdgroup so the store traffic is ~0.1% of the read.
/// Consecutive lanes take consecutive uint4 => fully coalesced 512 B per simdgroup.
const ROOF_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void stream_read(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
    constant uint& n4 [[buffer(2)]], constant uint& total [[buffer(3)]],
    uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint4 acc = uint4(0u);
    for (uint i = gid; i < n4; i += total) { acc ^= src[i]; }
    uint a = acc.x ^ acc.y ^ acc.z ^ acc.w;
    a = simd_sum(a);
    if (lane == 0u) { out[(gid >> 5) & 1023u] = a; }
}
// Read-modify-write over the recurrent state. Unlike the weights, EVERY sequence
// has its own copy, so this term scales linearly with N no matter how the decode
// is batched. This measures its floor.
kernel void state_rmw(device float4* s [[buffer(0)]],
    constant uint& n4 [[buffer(1)]], constant uint& total [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    for (uint i = gid; i < n4; i += total) { s[i] = s[i]*1.0009765625f + 1e-7f; }
}
kernel void nullk(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0xffffffffu) { out[0] = gid; }
}
"#;

/// The 2-D weights a surya-2 SSM (gated-DeltaNet) layer reads per decoded token.
/// (K, N) in ggml order: the GGUF stores [ne0=K, ne1=N], one N-row of K values.
const SSM_SHAPES: &[(&str, u32, u32)] = &[
    ("attn_qkv", 1024, 6144),
    ("attn_gate", 1024, 2048),
    ("ssm_alpha", 1024, 16),
    ("ssm_beta", 1024, 16),
    ("ssm_out", 2048, 1024),
    ("ffn_gate", 1024, 3584),
    ("ffn_up", 1024, 3584),
    ("ffn_down", 3584, 1024),
];
/// Layers 3, 7, 11, 15, 19, 23 (full_attention_interval = 4).
const ATTN_SHAPES: &[(&str, u32, u32)] = &[
    ("attn_q", 1024, 4096),
    ("attn_k", 1024, 512),
    ("attn_v", 1024, 512),
    ("attn_output", 2048, 1024),
    ("ffn_gate", 1024, 3584),
    ("ffn_up", 1024, 3584),
    ("ffn_down", 3584, 1024),
];
const HEAD: (&str, u32, u32) = ("lm_head", 1024, 65425);
const N_SSM: usize = 18;
const N_ATTN: usize = 6;

struct Mat {
    label: &'static str,
    k: u32,
    n: u32,
    /// F16: the whole weight. Q4L: nibbles.
    w: Buffer,
    /// Q4L side scale arrays (d1 and -m1), None for F16.
    qa: Option<Buffer>,
    qb: Option<Buffer>,
}

impl Mat {
    fn weights(&self) -> f64 { self.k as f64 * self.n as f64 }
    fn bytes(&self, q4l: bool) -> f64 { self.weights() * if q4l { 0.625 } else { 2.0 } }
}

fn mk(dev: &metal::Device, bytes: usize, fill: u8) -> Buffer {
    let b = dev.new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared);
    unsafe { std::ptr::write_bytes(b.contents() as *mut u8, fill, bytes.max(4)) };
    b
}

/// One weight set. Fill bytes are chosen so every f16 they decode to is a small normal
/// number: 0x3030 ~ 0.131, 0x1c1c ~ 0.00401. Denormals and NaNs are not a correctness
/// problem here (outputs are discarded) but could perturb timing.
fn make(dev: &metal::Device, label: &'static str, k: u32, n: u32, q4l: bool) -> Mat {
    let kn = k as usize * n as usize;
    if q4l {
        Mat {
            label, k, n,
            w: mk(dev, kn / 2, 0x5a),
            qa: Some(mk(dev, kn / 32 * 2, 0x1c)),
            qb: Some(mk(dev, kn / 32 * 2, 0x1c)),
        }
    } else {
        Mat { label, k, n, w: mk(dev, kn * 2, 0x30), qa: None, qb: None }
    }
}

/// Mirrors `dispatch.rs::mm` (M=1) and the `m <= mrow_max()` / GEMM branches of
/// `gemm_named` (M > 1). Returns the kernel name so the report can show which kernel
/// produced each number; the routing is shape-dependent.
fn kernel_for(q4l: bool, k: u32, n: u32, m: u32) -> (String, u32, u32) {
    // -> (name, threads_per_tg, rows_per_tg); rows_per_tg 0 means "2-D tile grid"
    if !q4l {
        if m == 1 { return ("gemv_f16".into(), 256, 8); }
        if m <= 8 { return ("gemv_m_f16".into(), 256, 8); }
        if n % 64 == 0 && k % 32 == 0 { return ("gemm_mm_f16".into(), 128, 0); }
        return ("gemv_m_f16".into(), 256, 8); // N%64!=0 (lm_head): no tile path
    }
    let t = 256u32;
    if m == 1 { return ("gemv_q4l".into(), t, t / 32 * 4); }
    if m <= 8 {
        // dispatch.rs:989-1005 fallback routing (OJAS_NO_XVTUNE form).
        // OJAS_BR_XV pins NXPSG for every shape (0 = force the row-blocked
        // m4/m8 kernels), so the M=8 routing can be A/B'd against the default.
        let xn = match std::env::var("OJAS_BR_XV").ok().and_then(|v| v.parse::<u32>().ok()) {
            Some(v) => v,
            None => if n >= 4096 { 4 } else if k >= 4096 && n <= 4096 { 8 } else { 0 },
        };
        if xn > 0 && k % 32 == 0 { return (format!("gemv_x{xn}_{m}_q4l"), t, t / xn); }
        if m > 4 { return ("gemv_m8_q4l".into(), t, t / 32 * 2); }
        return ("gemv_m4_q4l".into(), t, t / 64 * 4);
    }
    if n % 64 == 0 && k % 32 == 0 { return ("gemm_mm_q4l".into(), 128, 0); }
    ("gemv_m_q4l".into(), 256, 8)
}

#[allow(clippy::too_many_arguments)]
fn encode(e: &metal::ComputeCommandEncoderRef, p: &HashMap<String, ComputePipelineState>,
          mat: &Mat, x: &Buffer, y: &Buffer, m: u32, q4l: bool) {
    let (name, t, rows) = kernel_for(q4l, mat.k, mat.n, m);
    let pipe = p.get(&name).unwrap_or_else(|| panic!("kernel {name} not compiled"));
    e.set_compute_pipeline_state(pipe);
    e.set_buffer(0, Some(x), 0);
    e.set_buffer(1, Some(&mat.w), 0);
    e.set_buffer(2, Some(y), 0);
    e.set_bytes(3, 4, &mat.k as *const u32 as *const c_void);
    e.set_bytes(4, 4, &mat.n as *const u32 as *const c_void);
    let ac = 0u32;
    if q4l {
        e.set_buffer(5, Some(mat.qa.as_ref().unwrap()), 0);
        e.set_buffer(6, Some(mat.qb.as_ref().unwrap()), 0);
        if name.starts_with("gemm_mm_q4l") {
            // gemm_mm_q4l: qa at 5, accum at 6, M at 7, qb at 8 (dispatch.rs:900).
            e.set_bytes(6, 4, &ac as *const u32 as *const c_void);
            e.set_bytes(7, 4, &m as *const u32 as *const c_void);
            e.set_buffer(8, Some(mat.qb.as_ref().unwrap()), 0);
        } else if m > 1 {
            e.set_bytes(7, 4, &m as *const u32 as *const c_void);
            e.set_bytes(8, 4, &ac as *const u32 as *const c_void);
        }
    } else if name == "gemv_m_f16" {
        e.set_bytes(5, 4, &m as *const u32 as *const c_void);
    } else if name == "gemm_mm_f16" {
        e.set_bytes(6, 4, &ac as *const u32 as *const c_void);
        e.set_bytes(7, 4, &m as *const u32 as *const c_void);
    }
    if rows == 0 {
        e.dispatch_thread_groups(MTLSize::new(m.div_ceil(32) as u64, (mat.n / 64) as u64, 1),
                                 MTLSize::new(t as u64, 1, 1));
    } else {
        e.dispatch_thread_groups(MTLSize::new(mat.n.div_ceil(rows) as u64, 1, 1),
                                 MTLSize::new(t as u64, 1, 1));
    }
}

/// Best-of-`reps` GPU time (GPUStartTime/GPUEndTime, so a busy CPU perturbs it
/// far less than wall clock) for one command buffer built by `f`.
fn time_cb(gpu: &ojas_metal::MetalGpu, warm: usize, reps: usize,
           f: &dyn Fn(&metal::ComputeCommandEncoderRef)) -> f64 {
    for _ in 0..warm {
        let cb = gpu.command_buffer();
        let e = cb.new_compute_command_encoder();
        f(&e);
        e.end_encoding(); cb.commit(); cb.wait_until_completed();
    }
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let cb = gpu.command_buffer();
        let e = cb.new_compute_command_encoder();
        f(&e);
        e.end_encoding(); cb.commit(); cb.wait_until_completed();
        let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        best = best.min((ge - gs) * 1e3);
    }
    best
}

fn main() -> Result<()> {
    let which = std::env::args().nth(1).unwrap_or_else(|| "both".into());
    let part = std::env::args().nth(2).unwrap_or_else(|| "all".into());
    let ms: Vec<u32> = std::env::var("OJAS_BR_M").ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 3, 4, 6, 8]);

    let gpu = ojas_metal::MetalGpu::new()?;
    println!("device: {}  (native_reduce={})", gpu.device.name(), gpu.native_reduce);

    let roof_pipes: HashMap<String, ComputePipelineState> =
        gpu.compile_all(ROOF_SRC, |_| true)?.into_iter().collect();

    // ---------------------------------------------------------------- 1. roof
    if part == "roof" || part == "all" {
        const ROOF_MB: usize = 512;
        let n_bytes = ROOF_MB * 1024 * 1024;
        let src = mk(&gpu.device, n_bytes, 0x11);
        let out = mk(&gpu.device, 4096, 0);
        let n4 = (n_bytes / 16) as u32;
        println!("\n== 1. streaming-read roof ({ROOF_MB} MB, one pass per dispatch) ==");
        let mut peak = 0.0f64;
        for &tgs in &[512u32, 1024, 2048, 4096, 8192] {
            let total = tgs * 256;
            let ms_t = time_cb(&gpu, 2, 7, &|e| {
                e.set_compute_pipeline_state(&roof_pipes["stream_read"]);
                e.set_buffer(0, Some(&src), 0);
                e.set_buffer(1, Some(&out), 0);
                e.set_bytes(2, 4, &n4 as *const u32 as *const c_void);
                e.set_bytes(3, 4, &total as *const u32 as *const c_void);
                e.dispatch_thread_groups(MTLSize::new(tgs as u64, 1, 1), MTLSize::new(256, 1, 1));
            });
            let gbs = n_bytes as f64 / (ms_t / 1e3) / 1e9;
            println!("   {tgs:5} threadgroups x 256   {ms_t:7.3} ms   {gbs:6.1} GB/s");
            peak = peak.max(gbs);
        }
        println!("   -> achievable read bandwidth: {peak:.1} GB/s");

        // Fixed cost of a dispatch, for scaling the per-token dispatch count.
        let nullo = mk(&gpu.device, 4096, 0);
        for &cnt in &[200usize, 400] {
            let t = time_cb(&gpu, 2, 7, &|e| {
                for _ in 0..cnt {
                    e.set_compute_pipeline_state(&roof_pipes["nullk"]);
                    e.set_buffer(0, Some(&nullo), 0);
                    e.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
                }
            });
            println!("   {cnt} null dispatches: {t:7.3} ms  ({:.3} us/dispatch)", t * 1e3 / cnt as f64);
        }

        // Per-sequence recurrent state: ssm_state is d_state*d_state*dt_rank
        // = 128*128*16 f32 = 1 MiB per SSM layer, conv_state (K-1)*conv_ch
        // = 3*6144 f32 = 72 KiB, over 18 SSM layers. Read and written every token,
        // and not shared between sequences.
        let per_seq = 18 * (128 * 128 * 16 + 3 * 6144) * 4usize;
        let st = mk(&gpu.device, per_seq * 8, 0);
        println!("\n   per-sequence recurrent state: {:.1} MB (18 SSM layers), read+write every token",
                 per_seq as f64 / 1e6);
        for n_seq in [1usize, 2, 4, 8] {
            let n4 = (per_seq * n_seq / 16) as u32;
            let total = 8192u32 * 256;
            let t = time_cb(&gpu, 2, 7, &|e| {
                e.set_compute_pipeline_state(&roof_pipes["state_rmw"]);
                e.set_buffer(0, Some(&st), 0);
                e.set_bytes(1, 4, &n4 as *const u32 as *const c_void);
                e.set_bytes(2, 4, &total as *const u32 as *const c_void);
                e.dispatch_thread_groups(MTLSize::new(8192, 1, 1), MTLSize::new(256, 1, 1));
            });
            println!("   N={n_seq}: {t:7.4} ms  ({:.1} GB/s of the {:.0} MB round trip)",
                     2.0 * (per_seq * n_seq) as f64 / (t / 1e3) / 1e9,
                     2.0 * (per_seq * n_seq) as f64 / 1e6);
        }
    }

    for q4l in [false, true] {
        if which == "f16" && q4l { continue; }
        if which == "q4l" && !q4l { continue; }
        let tag = if q4l { "Q4L (prec 2)" } else { "F16 (prec 4, the `ocr` default)" };
        let bpw = if q4l { 0.625 } else { 2.0 };

        // One distinct buffer per (layer, tensor): 24 real layers, so the stream is the
        // model's full per-token footprint and cannot sit in cache.
        let mut mats: Vec<Mat> = Vec::new();
        for _ in 0..N_SSM { for &(l, k, n) in SSM_SHAPES { mats.push(make(&gpu.device, l, k, n, q4l)); } }
        for _ in 0..N_ATTN { for &(l, k, n) in ATTN_SHAPES { mats.push(make(&gpu.device, l, k, n, q4l)); } }
        mats.push(make(&gpu.device, HEAD.0, HEAD.1, HEAD.2, q4l));
        let total_bytes: f64 = mats.iter().map(|m| m.bytes(q4l)).sum();
        let maxk = mats.iter().map(|m| m.k).max().unwrap();
        let maxn = mats.iter().map(|m| m.n).max().unwrap();

        let gemv_src = ojas_metal::kernels::family_source("gemv").unwrap();
        let gemm_src = ojas_metal::kernels::family_source("gemv").unwrap();
        let want: Vec<String> = {
            let mut v = vec![];
            for &m in ms.iter() {
                for &(_, k, n) in SSM_SHAPES.iter().chain(ATTN_SHAPES).chain(std::iter::once(&HEAD)) {
                    v.push(kernel_for(q4l, k, n, m).0);
                }
            }
            v.sort(); v.dedup(); v
        };
        let mut pipes: HashMap<String, ComputePipelineState> =
            gpu.compile_all(gemv_src, |nm| want.iter().any(|w| w == nm))?.into_iter().collect();
        for fam in ["gemm_fat", "attn", "ops"] {
            let missing: Vec<&String> = want.iter().filter(|w| !pipes.contains_key(*w)).collect();
            if missing.is_empty() { break; }
            if let Some(src) = ojas_metal::kernels::family_source(fam) {
                for (nm, p) in gpu.compile_all(src, |nm| want.iter().any(|w| w == nm))? { pipes.insert(nm, p); }
            }
        }
        // gemm_mm_* live in the gemm_fat module, which is pasted into the prelude.
        if want.iter().any(|w| !pipes.contains_key(w)) {
            let src = ojas_metal::kernels::source_of("gemm_mm_f16")
                .or_else(|| ojas_metal::kernels::source_of("gemm_mm_q4l"));
            if let Some(src) = src {
                for (nm, p) in gpu.compile_all(src, |nm| want.iter().any(|w| w == nm))? { pipes.insert(nm, p); }
            }
        }
        let _ = gemm_src;
        let missing: Vec<&String> = want.iter().filter(|w| !pipes.contains_key(*w)).collect();
        if !missing.is_empty() { println!("   (not compiled, skipped: {missing:?})"); }

        let mmax = (*ms.iter().max().unwrap() as usize).max(8);
        let x = mk(&gpu.device, mmax * maxk as usize * 4, 0x3c);
        let y = mk(&gpu.device, mmax * maxn as usize * 4, 0);

        println!("\n======== {tag} ========");
        println!("  per-token weight read: {:.1} MB over {} matvec dispatches",
                 total_bytes / 1e6, mats.len());

        // ------------------------------------------------------- 2. per shape
        if part == "shapes" || part == "all" {
            println!("\n== 2. per-shape M-scaling (~192 MB of distinct weight sets per shape, best of 7) ==");
            let hdr: String = ms.iter().map(|m| format!("{:>24}", format!("M={m}"))).collect();
            println!("  {:<12} {:>6} {:>6} |{hdr}", "tensor", "K", "N");
            // Sized per shape so every set is ~192 MB, 4x the M2 Max system-level cache.
            // A fixed count leaves the small matrices resident and reports a bandwidth
            // DRAM never sees: attn_gate measured 455 then 202 GB/s across two runs at
            // 8 copies of a 4 MB matrix.
            let copies_for = |k: u32, n: u32| -> usize {
                let b = k as f64 * n as f64 * bpw;
                ((192e6 / b).ceil() as usize).clamp(3, 256)
            };
            let seen: Vec<(&str, u32, u32)> = {
                let mut v: Vec<(&str, u32, u32)> = SSM_SHAPES.iter()
                    .chain(ATTN_SHAPES).chain(std::iter::once(&HEAD))
                    .map(|&(l, k, n)| (l, k, n)).collect();
                v.sort_by_key(|&(_, k, n)| (k, n)); v.dedup_by_key(|&mut (_, k, n)| (k, n)); v
            };
            let rounds: usize = std::env::var("OJAS_BR_ROUNDS").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(6);
            for (label, k, n) in seen {
                let copies = copies_for(k, n);
                let set: Vec<Mat> = (0..copies).map(|_| make(&gpu.device, label, k, n, q4l)).collect();
                // Interleaved across rounds for the same reason as part 3.
                let mut best = vec![f64::INFINITY; ms.len()];
                for r in 0..rounds {
                    for (i, &m) in ms.iter().enumerate() {
                        if !pipes.contains_key(&kernel_for(q4l, k, n, m).0) { continue; }
                        let t = time_cb(&gpu, usize::from(r == 0), 2,
                                        &|e| { for mt in &set { encode(e, &pipes, mt, &x, &y, m, q4l); } })
                                / copies as f64;
                        if t < best[i] { best[i] = t; }
                    }
                }
                let mut cells = vec![];
                let mut base = 0.0f64;
                for (i, &m) in ms.iter().enumerate() {
                    if !best[i].is_finite() { cells.push(format!("{:>24}", "-")); continue; }
                    let t = best[i];
                    let gbs = k as f64 * n as f64 * bpw / (t / 1e3) / 1e9;
                    if m == 1 { base = t; }
                    let sp = m as f64 * base / t;
                    let (kn, _, _) = kernel_for(q4l, k, n, m);
                    let short = kn.trim_start_matches("gemv_").trim_start_matches("gemm_mm_");
                    cells.push(format!("{t:7.4} {gbs:4.0}GB/s {sp:4.2}x {:<7}", &short[..short.len().min(7)]));
                }
                println!("  {label:<12} {k:>6} {n:>6} | {}", cells.join(" "));
            }
        }

        // -------------------------------------------------- 3. whole-token step
        if part == "step" || part == "all" {
            println!("\n== 3. whole per-token weight read, 24 layers + lm_head, one command buffer ==");
            // The M values are interleaved across rounds and the per-M minimum kept.
            // Running M=1 to completion and then M=8 to completion would record how
            // machine load drifted rather than M: on a shared box a `cargo build`
            // starting mid-sweep charged the later M values ~2x and inverted the result.
            // Contention can only make a cell slower, so min-over-rounds recovers the
            // uncontended cost for every M from the same window.
            let rounds: usize = std::env::var("OJAS_BR_ROUNDS").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(6);
            let mut best: Vec<f64> = vec![f64::INFINITY; ms.len()];
            let mut ok: Vec<bool> = ms.iter()
                .map(|&m| mats.iter().all(|mt| pipes.contains_key(&kernel_for(q4l, mt.k, mt.n, m).0)))
                .collect();
            for r in 0..rounds {
                for (i, &m) in ms.iter().enumerate() {
                    if !ok[i] { continue; }
                    let t = time_cb(&gpu, usize::from(r == 0), 2,
                                    &|e| { for mt in &mats { encode(e, &pipes, mt, &x, &y, m, q4l); } });
                    if t < best[i] { best[i] = t; }
                }
            }
            for (i, _) in ms.iter().enumerate() { if !best[i].is_finite() { ok[i] = false; } }
            println!("  {:>3} {:>10} {:>11} {:>10} {:>12} {:>9}",
                     "M", "ms/step", "GB/s", "ms/token", "tok/s", "speedup");
            let mut base = 0.0f64;
            for (i, &m) in ms.iter().enumerate() {
                if !ok[i] { println!("  {m:>3}   (kernel missing, skipped)"); continue; }
                let t = best[i];
                let per_tok = t / m as f64;
                if m == 1 { base = per_tok; }
                println!("  {m:>3} {t:>10.3} {:>11.1} {per_tok:>10.4} {:>12.1} {:>8.2}x",
                         total_bytes / (t / 1e3) / 1e9, 1e3 / per_tok, base / per_tok);
            }
        }
    }
    Ok(())
}
