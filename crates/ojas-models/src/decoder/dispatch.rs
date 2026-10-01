#![allow(clippy::too_many_arguments)]
use super::*;
use objc::{msg_send, sel, sel_impl};
use metal::MTLSize;
use std::ffi::c_void;
 // re-export

/// OJAS_NAT_T: threads per threadgroup for the native-quant matvecs. One output
/// row per simdgroup, so a group covers threads/32 rows, which also sets how many
/// threadgroups a given N produces — fewer threads gives the scheduler more groups
/// to interleave, so the best value is worth sweeping per machine.
fn nat_threads_override() -> Option<u32> {
    static V: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OJAS_NAT_T").ok().and_then(|v| v.parse().ok()))
}

/// OJAS_NO_GEMM_FAT=1 forces the staged 64x32 GEMM — the fat-tile A/B control.
fn no_gemm_fat() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OJAS_NO_GEMM_FAT").is_ok())
}

/// OJAS_GEMM_MLX=1 routes batched Q4L matmuls through a 64x64 2-simdgroup tile.
fn gemm_mlx() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OJAS_GEMM_MLX").is_ok())
}

/// OJAS_GEMM_AC=1 opts in to the async-copy GEMM (measured slower for Q4L).
fn gemm_ac_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OJAS_GEMM_AC").is_ok())
}

/// Largest M the register-blocked m-row gemv kernels claim before the padded MMA
/// GEMM takes over. The GEMM pads M up to a 32-row tile, so a verify-sized batch
/// runs a mostly-empty tile: at M=8 the padded GEMM measured 48.5-48.9 ms against
/// gemv_m8_q4l's 42.4, so 8 belongs to the m-row kernel. Beyond 8 the m-row
/// fallback loses badly (M=16: 177.5 ms vs the GEMM's 48.1) because gemv_m_q4l
/// walks M in groups of 8 with no weight reuse across groups. OJAS_MROW_MAX
/// overrides.
///
/// M below which the native batched path loops the M=1 kernel instead of using
/// the `_m` kernel. OJAS_NAT_MLOOP overrides it for sweeping.
///
/// It was 96 while the batched kernel was a generic body that lost to a plain loop
/// everywhere below M~100. Rebuilt on the M=1 body (NAT_GEMV_M_ROW) it wins from
/// M=2 up, so the threshold is now just "more than one row". Sum of per-category
/// GPU time, Qwen3.8-27B UD-IQ2_XXS:
///
/// ```text
///   M     batched     loop
///   2       72.4 ms   83.7 ms   batched 1.16x
///   4      135.5     159.7      batched 1.18x
///   8      260.1     316.4      batched 1.22x
///  32     1023.0    1260.2      batched 1.23x
/// ```
///
/// The `_m` kernel loses to a plain loop over the whole speculation range. It
/// amortizes the weight read across rows, but it is a generic body at one row per
/// simdgroup, while the M=1 kernel gets each format's tuned nr0/nsg and its
/// hand-written specialization, and that gap outweighs the amortization until the
/// weight read dominates around M~100. Above the crossover the batched kernel does
/// win and prefill runs there, so this stays a threshold rather than a deletion.
pub(crate) fn nat_mloop() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OJAS_NAT_MLOOP").ok().and_then(|s| s.parse().ok()).unwrap_or(2))
}

pub(crate) fn mrow_max() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("OJAS_MROW_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
    })
}

/// Lanes per row for the lane-partitioned verify matvec (gemv_x<NXPSG>_<M>_q4l).
/// 8 (the default) means "route by shape"; 0 disables the family; any other
/// value pins that NXPSG for every shape, for A/B sweeps.
pub(crate) fn xv_nxpsg() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("OJAS_XV").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
    })
}

/// Minimum K for the split-K GEMM. At the shipped 4096 only ffn_down qualifies,
/// leaving o_proj and qkv's q — both n=2048, 256 tiles — on the plain tile at 79%
/// of the best measured rate. OJAS_SK_KMIN sweeps it.
pub(crate) fn sk_kmin() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("OJAS_SK_KMIN").ok().and_then(|v| v.parse().ok()).unwrap_or(4096)
    })
}

/// True unless OJAS_Q4F_TAIL=0. Routes the N % 8 != 0 shapes (in practice the
/// lm_head alone) through `gemv_q4_fast` over a 4-row-aligned prefix plus a
/// `gemv_q4` tail, instead of `gemv_q4` over the whole matrix. Worth 237 -> 428
/// GB/s on that dispatch and +2.2% end-to-end decode on surya-2 at prec=2.
pub(crate) fn q4f_tail() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OJAS_Q4F_TAIL").map(|v| v != "0").unwrap_or(true))
}

/// A/B knob for the tuned-Q4 (prec=2) M=1 matvec: `OJAS_Q4F=<kernel>:<threads>`.
/// `fast` is the incumbent (gemv_q4_fast, 4 rows/simdgroup); `fast8`, `vec4` and
/// `xabl` are the three measured negatives documented in kernels/gemv.rs (`xabl` is
/// numerically wrong, a diagnostic only); `ksplit` and `plain` are the
/// gemv_q4_ksplit / gemv_q4 launch shapes. Unset = the shipped choice.
///
/// The Q4 fast path hardcoded `(64, 8)` and returned before consulting
/// `tune.gemv_plan`, so the autotuner's entries for these shapes — which do exist,
/// `Apple M2 Max#q4v2 3584 1024` among them — went unused. This knob measures the
/// alternatives on the real forward rather than on a micro-bench.
pub(crate) fn q4f_override() -> Option<(&'static str, u32)> {
    static V: std::sync::OnceLock<Option<(&'static str, u32)>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("OJAS_Q4F").ok()?;
        let (k, t) = v.split_once(':')?;
        let t: u32 = t.parse().ok()?;
        let k: &'static str = match k {
            "fast" => "fast", "fast8" => "fast8", "xabl" => "xabl", "vec4" => "vec4", "ksplit" => "ksplit", "plain" => "plain",
            _ => return None,
        };
        Some((k, t))
    })
}

/// Fallback threadgroup size when a shape has no tuned entry. OJAS_Q4L_TG pins
/// it for A/B; 256 is the M2 Max value the whole family used to hardcode.
pub(crate) fn q4l_tg() -> u32 {
    static TG: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *TG.get_or_init(|| {
        std::env::var("OJAS_Q4L_TG").ok().and_then(|v| v.parse().ok()).unwrap_or(256)
    })
}

impl<'a> DecoderGpu<'a> {
    /// The `up` GEMM with the SwiGLU epilogue fused into its store: writes
    /// silu(gate)*up straight to `out`, so no separate silu_mul dispatch and no
    /// round trip through `up`. Returns false when the shape or the weight map
    /// does not qualify, and the caller falls back to the split form.
    pub(crate) fn gemm_up_silu(&self, enc: &metal::ComputeCommandEncoderRef, name: &str,
                x: &metal::Buffer, gate: &metal::Buffer, out: &metal::Buffer,
                k: u32, n: u32, m: u32, act: u32) -> bool {
        // Opt-in, default off: worth 5.3 ms of a 344 ms prefill on the critical
        // path, but not numerically neutral. The logits checksum moves 0.20% and
        // the top1-top2 gap narrows 2.2%, against 0.072%/+0.8% for the split-K
        // reassociation that was adopted. A bisect (identity epilogue into `up`,
        // silu_mul left in place) proved the GEMM half bit-exact, so the drift is
        // entirely the elementwise epilogue, where fast-math rounding should be
        // ~1e-7 rather than 1e-3. Ruled out: vector-vs-scalar exp, the fragment
        // gate load (replaced with explicit addressing), barrier visibility (same
        // under OJAS_SERIAL), buffer aliasing and coverage. Remaining suspect is
        // the separate translation unit: gemm_fat.rs compiles under `#pragma METAL
        // internals : enable`, which may lower the transcendentals differently
        // than ops.rs does for silu_mul. OJAS_FUSED_SILU=1 measures it; the
        // explanation should close before this becomes a default.
        if std::env::var("OJAS_FUSED_SILU").is_err() { return false; }
        if !(self.gpu.native_reduce && n % 64 == 0 && k % 32 == 0 && m > mrow_max()) { return false; }
        if !self.p.contains_key("gemm_mm_q4l_fatx2_silu") { return false; }
        let Some(w) = self.wt.w4l.get(name) else { return false };
        self.check_shape(name, k, n); // fused up-proj: (d, ff)
        enc.set_compute_pipeline_state(&self.p["gemm_mm_q4l_fatx2_silu"]);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
        enc.set_buffer(5, Some(&self.wt.q4l_a[name]), 0);
        let zero = 0u32;
        enc.set_bytes(6, 4, &zero as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
        enc.set_buffer(8, Some(&self.wt.q4l_b[name]), 0);
        enc.set_buffer(9, Some(gate), 0);
        enc.set_bytes(10, 4, &act as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(((m + 63) / 64) as u64, (n / 64) as u64, 1),
                                   MTLSize::new(128, 1, 1));
        true
    }

    /// Tuned threadgroup size for a Q4L gemv shape, else the fallback.
    /// OJAS_Q4L_TG overrides everything (A/B knob).
    pub(crate) fn q4l_threads(&self, k: u32, n: u32) -> u32 {
        if std::env::var("OJAS_Q4L_TG").is_ok() { return q4l_tg(); }
        self.tune.q4l_tg.get(&(k, n)).copied().unwrap_or_else(q4l_tg)
    }

    pub(crate) fn enc1d(&self, enc: &metal::ComputeCommandEncoderRef, kern: &str, bufs: &[(&metal::Buffer, u64)], ints: &[(u32, u32)], floats: &[(u32, f32)], threads: u64) {
        enc.set_compute_pipeline_state(&self.p[kern]);
        for (b, idx) in bufs {
            enc.set_buffer(*idx, Some(b), 0);
        }
        for (idx, val) in ints {
            enc.set_bytes(*idx as u64, 4, val as *const u32 as *const c_void);
        }
        for (idx, val) in floats {
            enc.set_bytes(*idx as u64, 4, val as *const f32 as *const c_void);
        }
        let per = 64u64;
        let groups = (threads + per - 1) / per;
        enc.dispatch_thread_groups(MTLSize::new(groups.max(1), 1, 1), MTLSize::new(per, 1, 1));
    }

    /// Matmul dispatch, picking f16 or Q8 kernel. `kind` ∈ plain/bias/accum.
    /// For q8 plain/accum, uses the autotuned per-(K,N) plan (ksplit + thread count);
    /// falls back to a safe default for shapes not in the table.

    /// One row of a native matvec, with the activation and output rows selected by
    /// buffer offset rather than by a kernel argument.
    ///
    /// The batched native path uses this for small M: the `_m` kernel is a generic
    /// body at nr0=1, while the M=1 kernel gets the per-format nr0/nsg launch shape
    /// and the hand-written specializations, and the difference is bigger than the
    /// batching saves. On Qwen3.8-27B UD-IQ2_XXS, ffn_gu costs 23.3 ms at M=1 and
    /// 71.9 ms through the `_m` kernel at M=2 — 3.08x for two rows, where looping
    /// the M=1 kernel is 2.0x by construction.
    ///
    /// Offsets are in bytes; both are float-aligned, which is all Metal asks.
    pub(crate) fn nat_row(&self, enc: &metal::ComputeCommandEncoderRef, wname: &str, ty: u32,
               w: &metal::Buffer, x: &metal::Buffer, xoff: u64,
               y: &metal::Buffer, yoff: u64, k: u32, n: u32, accum: bool) {
        let entry = ojas_metal::kernels::nat::nat_entry(ty, if accum { "_accum" } else { "" })
            .unwrap_or_else(|| panic!("no native kernel for GGUF type {ty} ({wname})"));
        enc.set_compute_pipeline_state(&self.p[&entry]);
        enc.set_buffer(0, Some(x), xoff);
        enc.set_buffer(1, Some(w), self.wt.w_off.get(wname).copied().unwrap_or(0));
        enc.set_buffer(2, Some(y), yoff);
        enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
        let (t, rows) = ojas_metal::kernels::nat::nat_launch(ty)
            .expect("native format without a launch shape");
        enc.dispatch_thread_groups(MTLSize::new(((n + rows - 1) / rows) as u64, 1, 1),
                                   MTLSize::new(t as u64, 1, 1));
    }

    /// Native batched matvec: M rows of `x` against one weight matrix.
    ///
    /// Splits M into a whole number of MTILE-row passes plus a remainder handled one
    /// row at a time, because the batched kernel pads a ragged tile by recomputing
    /// the last row: at MTILE=2 an odd M costs what M+1 costs. Left unsplit, M=3
    /// measured 135.4 ms against 124.1 for a plain loop, the one width where
    /// batching lost. Split, the M=1 kernel takes the odd row.
    ///
    /// Below `nat_mloop()` rows it is all loop; M=1 has nothing to batch.
    pub(crate) fn nat_batched(&self, enc: &metal::ComputeCommandEncoderRef, wname: &str,
                   x: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32, m: u32, accum: bool) {
        let w = &self.wt.wq[wname];
        let ty = self.wt.w_qtype[wname];
        let off = self.wt.w_off.get(wname).copied().unwrap_or(0);
        if self.kquant_fat(ty, enc, x, w, off, y, k, n, m, accum) { return; }
        // Cooperative Q8 wins on wide projections, but loses on narrow HC
        // down projections. Four-row tiles avoid rereading weights at M=3/4.
        let cooperative = ty == 8 && self.arch.qwen4exp.is_some()
            && self.cfg.flash_q8_cooperative && n >= 1024 && (2..=8).contains(&m);
        // The cooperative kernel handles a ragged tile in one dispatch.
        let mtile = if cooperative { 1 } else { ojas_metal::kernels::nat::m_mtile() };
        let mb = if m < nat_mloop() { 0 } else { m - m % mtile };
        for r in mb as u64..m as u64 {
            self.nat_row(enc, wname, ty, w, x, r * k as u64 * 4, y, r * n as u64 * 4,
                         k, n, accum);
        }
        if mb == 0 { return; }
        let tag = ojas_metal::kernels::nat::nat_entry(ty, "")
            .unwrap_or_else(|| panic!("no native batched kernel for GGUF type {ty} ({wname})"));
        let entry = if cooperative {
            if m >= 3 { "gemv_nat_q80_m_cooperative4".into() }
            else { "gemv_nat_q80_m_cooperative".into() }
        } else { ojas_metal::kernels::nat::m_entry(tag.trim_start_matches("gemv_nat_")) };
        enc.set_compute_pipeline_state(&self.p[&entry]);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), self.wt.w_off.get(wname).copied().unwrap_or(0));
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &mb as *const u32 as *const c_void);
        let acc = u32::from(accum);
        enc.set_bytes(8, 4, &acc as *const u32 as *const c_void);
        let (thr, rpg) = if cooperative { (128, 8) } else {
            ojas_metal::kernels::nat::nat_launch_m(ty)
                .expect("native format without a batched launch shape")
        };
        enc.dispatch_thread_groups(MTLSize::new(((n + rpg - 1) / rpg) as u64, 1, 1),
                                   MTLSize::new(thr as u64, 1, 1));
    }

    pub(crate) fn mm(&self, enc: &metal::ComputeCommandEncoderRef, kind: &str, wname: &str, x: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32, bias: Option<&metal::Buffer>) {
        self.check_shape(wname, k, n); // the single per-token projection dispatch — covers q/k/v/o/ffn/ssm
        if self.wt.repr(wname) == Repr::F32 {
            let kernel = match kind { "plain" => "gemv_w32", "accum" => "gemv_w32_accum", "bias" => "gemv_w32_bias", _ => panic!("unknown F32 matvec kind {kind}") };
            enc.set_compute_pipeline_state(&self.p[kernel]);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(&self.wt.w32[wname]), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            if kind == "bias" { enc.set_buffer(5, Some(bias.expect("missing F32 bias")), 0); }
            enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            return;
        }
        // Mixed-precision Q4 experiment. A global Q4 model takes the full
        // autotuned branch below; this path serves selected Q4 matrices among
        // otherwise native weights (currently output head and FFN down).
        if !self.wt.q4 && self.wt.w4.contains_key(wname) {
            assert!(matches!(kind, "plain" | "accum"), "mixed Q4 supports plain/accum only");
            assert_eq!(n % 8, 0, "mixed Q4 matrix requires eight-row alignment");
            enc.set_compute_pipeline_state(&self.p[if kind == "accum" {
                "gemv_q4_fast_accum"
            } else {
                "gemv_q4_fast"
            }]);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(&self.wt.w4[wname]), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            enc.set_buffer(5, Some(&self.wt.scale4[wname]), 0);
            enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(8) as u64, 1, 1), MTLSize::new(64, 1, 1));
            return;
        }
        // Native quantized path: the GGUF's own blocks, decoded in-kernel. Checked
        // first because when it is populated the weight exists in no other map.
        if let Some(w) = self.wt.wq.get(wname) {
            let ty = self.wt.w_qtype[wname];
            let suffix = match kind { "accum" => "_accum", "bias" => "_bias", _ => "" };
            let entry = ojas_metal::kernels::nat::nat_entry(ty, suffix)
                .unwrap_or_else(|| panic!("no native kernel for GGUF type {ty} ({wname})"));
            enc.set_compute_pipeline_state(&self.p[&entry]);
            enc.set_buffer(0, Some(x), 0);
            // Zero-copy buffers start on a page boundary below the tensor; the
            // leftover bytes ride in `w_off`.
            enc.set_buffer(1, Some(w), self.wt.w_off.get(wname).copied().unwrap_or(0));
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            if kind == "bias" {
                enc.set_buffer(5, Some(bias.expect("bias kind without a bias buffer")), 0);
            }
            // Launch shape comes from the format: the kernel puts nr0 rows on each
            // simdgroup and runs nsg simdgroups, so a threadgroup covers nsg*nr0
            // rows. These kernels hold ~32 activation registers, so nsg stays small
            // (2 for most formats); more simdgroups would cost occupancy rather than
            // buy it. OJAS_NAT_T overrides the thread count for sweeping.
            let (t, rows) = match nat_threads_override() {
                // rows must be recomputed from the overridden thread count: the
                // kernel derives its row index from `ts`, so keeping the default
                // row stride here would silently skip rows.
                Some(o) => ojas_metal::kernels::nat::nat_launch_with(ty, o),
                None => ojas_metal::kernels::nat::nat_launch(ty),
            }.expect("native format without a launch shape");
            enc.dispatch_thread_groups(MTLSize::new(((n + rows - 1) / rows) as u64, 1, 1),
                                       MTLSize::new(t as u64, 1, 1));
            return;
        }
        // Native ternary Q2_0 path. plain/accum only (qwen3 has no qkv bias).
        if let Some(w) = self.wt.w20.get(wname) {
            enc.set_compute_pipeline_state(&self.p[if kind == "accum" { "gemv_q20_accum" } else { "gemv_q20" }]);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(w), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            enc.set_buffer(5, Some(&self.wt.s20[wname]), 0);
            enc.dispatch_thread_groups(MTLSize::new(((n + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            return;
        }
        // Native Q4_K / Q6_K kept in their own maps (a Q6_K token table, prec 3's
        // all-native fallback): the same blocks `wq` holds, so the same native GEMV
        // bodies (`NAT_GEMV_Q4K`/`NAT_GEMV_Q6K`). plain/accum only.
        for (map, ty) in [(&self.wt.w4k, 12u32), (&self.wt.w6k, 14)] {
            let Some(w) = map.get(wname) else { continue };
            let tag = if ty == 12 { "q4k" } else { "q6k" };
            enc.set_compute_pipeline_state(&self.p[&if kind == "accum" { format!("gemv_nat_{tag}_accum") } else { format!("gemv_nat_{tag}") }]);
            enc.set_buffer(0, Some(x), 0);
            // Zero-copy buffers start at a page boundary below the tensor; the
            // leftover bytes ride in `w_off`. Copied buffers have no offset.
            enc.set_buffer(1, Some(w), self.wt.w_off.get(wname).copied().unwrap_or(0));
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            let (threads, rows) = ojas_metal::kernels::nat::nat_launch(ty).expect("K-quant launch shape");
            enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(rows) as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
            return;
        }
        // Q4L: Q4_K values in the tuned layout. 4 output rows per simdgroup, so the
        // threadgroup covers 32 rows.
        if let Some(w) = self.wt.w4l.get(wname) {
            // A K-split variant (one row per threadgroup, 8 simdgroups slicing K, so
            // more independent request streams in flight) measured slower: ffn_down
            // 1.70 -> 2.15 ms, o_proj 0.39 -> 0.73. With the other probes (bytes
            // -20%: null; ALU -92%: null; rows/sg 4/2/1: null; ushort4 weight loads:
            // null; float4 activation loads: null) that pins the Q4L family at ~530
            // Gweights/s for reasons none of those knobs touch. It is already the
            // fastest per-weight rate in the engine — the Q8 lm_head does 360 Gw/s —
            // so treat 530 as this GPU's practical ceiling for this access pattern
            // before assuming a kernel bug.
            enc.set_compute_pipeline_state(&self.p[if kind == "accum" { "gemv_q4l_accum" } else { "gemv_q4l" }]);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(w), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            enc.set_buffer(5, Some(&self.wt.q4l_a[wname]), 0);
            enc.set_buffer(6, Some(&self.wt.q4l_b[wname]), 0);
            let t = self.q4l_threads(k, n);
            let rows = t / 32 * 4;                      // 4 rows per simdgroup
            enc.dispatch_thread_groups(MTLSize::new(((n + rows - 1) / rows) as u64, 1, 1), MTLSize::new(t as u64, 1, 1));
            return;
        }
        // Native Q4 path (o_proj/ffn_down/lm_head are plain/accum, no bias). Long-K
        // small-N uses K-split (1 row/tg, 8 simdgroups split blocks) for BW; else the
        // plain 8-rows/tg kernel (lm_head huge-N already has plenty of threadgroups).
        if self.wt.q4 {
            // Some GGUFs keep small matrices unquantized (dynamic-quant GGUFs: ssm_alpha/beta
            // are F32 2D). Fall back to the f32/f16 GEMV for those.
            if !self.wt.w4.contains_key(wname) {
                if let Some(w) = self.wt.w32.get(wname) {
                    assert!(kind == "plain", "f32 fallback supports plain only ({wname})");
                    enc.set_compute_pipeline_state(&self.p["gemv_w32"]);
                    enc.set_buffer(0, Some(x), 0);
                    enc.set_buffer(1, Some(w), 0);
                    enc.set_buffer(2, Some(y), 0);
                    enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(((n + 7)/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
                    return;
                }
            }
            // qmv_fast-style kernel (8 rows/tg, uint16 loads, no-shift dequant) —
            // fastest when N%8==0 (all our matmul shapes). Handles any K via a tail loop.
            if n % 8 == 0 {
                // fast kernel: 4 rows/simdgroup, 2 simdgroups/tg (64 threads, 8 rows/tg),
                // matching the reference. Fixed rather than autotuned: the isolated
                // micro-bench favors 256 but the real interleaved forward prefers 64.
                // OJAS_Q4F=<kernel>:<threads> overrides for A/B; see q4f_override.
                let (kern, t) = q4f_override().unwrap_or(("fast", 64));
                let bind = |kn: &str| {
                    enc.set_compute_pipeline_state(&self.p[kn]);
                    enc.set_buffer(0, Some(x), 0);
                    enc.set_buffer(1, Some(&self.wt.w4[wname]), 0);
                    enc.set_buffer(2, Some(y), 0);
                    enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                    enc.set_buffer(5, Some(&self.wt.scale4[wname]), 0);
                };
                let acc = kind == "accum";
                match kern {
                    // One row per threadgroup, `t` threads slicing K — for
                    // parallelism, not tiling: `gemv_q4_fast` puts 4 rows on a
                    // simdgroup, so a whole matvec is only 8*N threads (8192 for
                    // ffn_down's N=1024, about a quarter of what this GPU can hold).
                    "ksplit" => {
                        bind(if acc { "gemv_q4_ksplit_accum" } else { "gemv_q4_ksplit" });
                        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(t as u64, 1, 1));
                    }
                    // one row per simdgroup: t/32 rows per threadgroup.
                    "plain" => {
                        let rows = (t / 32).max(1);
                        bind(if acc { "gemv_q4_accum" } else { "gemv_q4" });
                        enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(rows) as u64, 1, 1), MTLSize::new(t as u64, 1, 1));
                    }
                    // A/B variant: same kernel with one ushort4 weight load.
                    "vec4" => {
                        let rows = (t / 32).max(1) * 4;
                        bind("gemv_q4_fast_v4");
                        enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(rows) as u64, 1, 1), MTLSize::new(t as u64, 1, 1));
                    }
                    // Diagnostic, numerically wrong: see gemv_q4_xabl.
                    "xabl" => {
                        let rows = (t / 32).max(1) * 4;
                        bind("gemv_q4_xabl");
                        enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(rows) as u64, 1, 1), MTLSize::new(t as u64, 1, 1));
                    }
                    // 8 rows per simdgroup: half the activation traffic per output
                    // row, twice the live accumulators.
                    "fast8" => {
                        let rows = (t / 32).max(1) * 8;
                        bind(if acc { "gemv_q4_fast8_accum" } else { "gemv_q4_fast8" });
                        enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(rows) as u64, 1, 1), MTLSize::new(t as u64, 1, 1));
                    }
                    // `gemv_q4_fast` derives its row from `ts` (out_row =
                    // tgid*(ts/32*4) + sgid*4), so rows per threadgroup must be
                    // recomputed from the thread count or the grid skips rows.
                    _ => {
                        let rows = (t / 32).max(1) * 4;
                        bind(if acc { "gemv_q4_fast_accum" } else { "gemv_q4_fast" });
                        enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(rows) as u64, 1, 1), MTLSize::new(t as u64, 1, 1));
                    }
                }
                return;
            }
            // N % 8 != 0 — on every model served that is the lm_head and nothing else.
            // Routing it to gemv_q4/gemv_q4_ksplit — the obvious choice, since
            // `gemv_q4_fast` stores 4 rows at a time — measured 237 GB/s on surya-2 at
            // prec=2, against 381-387 for the same kernel on the FFN shapes, on 12% of
            // the token's weight bytes in a dispatch with nothing to overlap against. See the note in
            // kernels/gemv.rs.
            //
            // Split rather than guard: `gemv_q4_fast` reads rows out_row..out_row+3
            // before it stores any of them, so clamping the stores still overruns the
            // weight and scale buffers by up to 3 rows. Instead run it over the
            // 4-row-aligned prefix only — pass N4 as its N, so its own `out_row >= N`
            // guard retires the partial groups — and sweep the <= 3 remaining rows
            // with `gemv_q4` at a row offset. Every access is then in bounds by
            // construction.
            //
            // OJAS_Q4F_TAIL=0 restores the old single-kernel routing for A/B.
            if q4f_tail() {
                let n4 = n / 4 * 4;          // rows the 4-row kernel can own outright
                let acc = kind == "accum";
                if n4 > 0 {
                    let (t, rows) = (64u32, 8u32);
                    enc.set_compute_pipeline_state(&self.p[if acc { "gemv_q4_fast_accum" } else { "gemv_q4_fast" }]);
                    enc.set_buffer(0, Some(x), 0);
                    enc.set_buffer(1, Some(&self.wt.w4[wname]), 0);
                    enc.set_buffer(2, Some(y), 0);
                    enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &n4 as *const u32 as *const c_void);
                    enc.set_buffer(5, Some(&self.wt.scale4[wname]), 0);
                    enc.dispatch_thread_groups(MTLSize::new(n4.div_ceil(rows) as u64, 1, 1), MTLSize::new(t as u64, 1, 1));
                }
                // Tail: rows [n4, n). At most 3, so one threadgroup. Reached by
                // offsetting the buffers, which keeps `gemv_q4`'s own row indexing
                // (row 0 = the first tail row) and needs no kernel change. Offsets
                // are 4-byte aligned by construction: n4 % 4 == 0, the Q4 row is
                // K/2 bytes with K % 32 == 0, and the scale row is K/32 halves.
                let tail = n - n4;
                if tail > 0 {
                    let nblk = (k / 32) as u64;
                    enc.set_compute_pipeline_state(&self.p[if acc { "gemv_q4_accum" } else { "gemv_q4" }]);
                    enc.set_buffer(0, Some(x), 0);
                    enc.set_buffer(1, Some(&self.wt.w4[wname]), n4 as u64 * (k / 2) as u64);
                    enc.set_buffer(2, Some(y), n4 as u64 * 4);
                    enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &tail as *const u32 as *const c_void);
                    enc.set_buffer(5, Some(&self.wt.scale4[wname]), n4 as u64 * nblk * 2);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(64, 1, 1));
                }
                return;
            }
            // autotuned plan (ksplit + threads) per shape; fallback heuristic if absent.
            let plan = self.tune.gemv_plan.get(&(k, n)).copied()
                .unwrap_or(GemvPlan { ksplit: k >= 2048 && n <= 8192, threads: 256 });
            enc.set_compute_pipeline_state(&self.p[match (plan.ksplit, kind) {
                (true, "accum") => "gemv_q4_ksplit_accum",
                (true, _) => "gemv_q4_ksplit",
                (false, "accum") => "gemv_q4_accum",
                (false, _) => "gemv_q4",
            }]);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(&self.wt.w4[wname]), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            enc.set_buffer(5, Some(&self.wt.scale4[wname]), 0);
            if plan.ksplit {
                enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(plan.threads, 1, 1));
            } else {
                let rows = (plan.threads / 32).max(1) as u32;
                enc.dispatch_thread_groups(MTLSize::new(((n + rows - 1) / rows) as u64, 1, 1), MTLSize::new(plan.threads, 1, 1));
            }
            return;
        }
        // F32-kept weights. The q4 path above has this fallback; without it here, any
        // tensor the file stores as f32 — on the streamed path a good part of the
        // skeleton, since `read_tensor` hands those back unconverted — reaches the
        // `w16[wname]` index below and panics with "no entry found for key" and no
        // tensor name.
        if !self.wt.w16.contains_key(wname) && !self.wt.w8.contains_key(wname) {
            if let Some(w) = self.wt.w32.get(wname) {
                assert!(kind == "plain", "f32 fallback supports plain only ({wname})");
                enc.set_compute_pipeline_state(&self.p["gemv_w32"]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(y), 0);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((n + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
                return;
            }
        }
        // Autotuned plan for this shape (q8, non-bias). Default: 8 rows/tg, no split.
        let plan = if self.wt.q8 && kind != "bias" {
            self.tune.gemv_plan.get(&(k, n)).copied().unwrap_or(GemvPlan { ksplit: false, threads: 256 })
        } else {
            GemvPlan { ksplit: false, threads: 256 }
        };
        let (kern, w, scale): (&str, &metal::Buffer, Option<&metal::Buffer>) = if self.wt.q8 {
            let kn = if plan.ksplit {
                if kind == "accum" { "gemv_q8_ksplit_accum" } else { "gemv_q8_ksplit" }
            } else {
                match kind { "bias" => "gemv_q8_bias", "accum" => "gemv_q8_accum", _ => "gemv_q8" }
            };
            (kn, &self.wt.w8[wname], Some(&self.wt.scale8[wname]))
        } else {
            let kn = match kind { "bias" => "gemv_bias", "accum" => "gemv_accum", _ => "gemv_f16" };
            (kn, self.wt.w16.get(wname)
                    .unwrap_or_else(|| panic!("{wname} is in no weight map ({:?})", self.wt.repr(wname))), None)
        };
        enc.set_compute_pipeline_state(&self.p[kern]);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
        // f16: bias at 5. Q8: scale at 5, bias at 6.
        match (scale, bias) {
            (Some(s), Some(b)) => { enc.set_buffer(5, Some(s), 0); enc.set_buffer(6, Some(b), 0); }
            (Some(s), None) => { enc.set_buffer(5, Some(s), 0); }
            (None, Some(b)) => { enc.set_buffer(5, Some(b), 0); }
            (None, None) => {}
        }
        // q8 kernels use tgid*(ts/32)+sgid so rows/tg = threads/32; ksplit uses one
        // row/tg (its simdgroups split K). f16 kernels hardcode 8 rows/tg (256 threads).
        if self.wt.q8 {
            if plan.ksplit {
                enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(plan.threads, 1, 1));
            } else {
                let rows = (plan.threads / 32).max(1) as u32;
                enc.dispatch_thread_groups(MTLSize::new(((n + rows - 1) / rows) as u64, 1, 1), MTLSize::new(plan.threads, 1, 1));
            }
        } else {
            enc.dispatch_thread_groups(MTLSize::new(((n + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
    }

    /// Fused Q/K/V projection in one dispatch (q8 only). Reads normalized `h`, writes
    /// q,k,v. RMSNorm stays a separate dispatch: folded in here, its reduction gets
    /// replicated across every threadgroup and regresses.
    pub(crate) fn qkv(&self, enc: &metal::ComputeCommandEncoderRef, lp: &dyn Fn(&str) -> String,
           h: &metal::Buffer, k_in: u32, nq: u32, nkv: u32) {
        self.check_qkv(lp, k_in, nq, nkv);
        let g = |s: &str| lp(s);
        enc.set_compute_pipeline_state(&self.p["qkv_q8"]);
        enc.set_buffer(0, Some(h), 0);
        enc.set_buffer(1, Some(&self.wt.w8[&g("attn_q.weight")]), 0);
        enc.set_buffer(2, Some(&self.wt.w8[&g("attn_k.weight")]), 0);
        enc.set_buffer(3, Some(&self.wt.w8[&g("attn_v.weight")]), 0);
        enc.set_buffer(4, Some(&self.st.q), 0);
        enc.set_buffer(5, Some(&self.st.k), 0);
        enc.set_buffer(6, Some(&self.st.v), 0);
        enc.set_bytes(7, 4, &k_in as *const u32 as *const c_void);
        enc.set_bytes(8, 4, &nq as *const u32 as *const c_void);
        enc.set_bytes(9, 4, &nkv as *const u32 as *const c_void);
        enc.set_buffer(10, Some(&self.wt.scale8[&g("attn_q.weight")]), 0);
        enc.set_buffer(11, Some(&self.wt.scale8[&g("attn_k.weight")]), 0);
        enc.set_buffer(12, Some(&self.wt.scale8[&g("attn_v.weight")]), 0);
        enc.set_buffer(13, Some(&self.wt.w32[&g("attn_q.bias")]), 0);
        enc.set_buffer(14, Some(&self.wt.w32[&g("attn_k.bias")]), 0);
        enc.set_buffer(15, Some(&self.wt.w32[&g("attn_v.bias")]), 0);
        let total = nq + 2 * nkv;
        enc.dispatch_thread_groups(MTLSize::new(((total + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Fused Q/K/V for the f16 path (half weights, no scale). One dispatch (was 3).
    pub(crate) fn qkv_f16(&self, enc: &metal::ComputeCommandEncoderRef, lp: &dyn Fn(&str) -> String,
               h: &metal::Buffer, k_in: u32, nq: u32, nkv: u32) {
        self.check_qkv(lp, k_in, nq, nkv);
        let g = |s: &str| lp(s);
        enc.set_compute_pipeline_state(&self.p["qkv_f16"]);
        enc.set_buffer(0, Some(h), 0);
        enc.set_buffer(1, Some(&self.wt.w16[&g("attn_q.weight")]), 0);
        enc.set_buffer(2, Some(&self.wt.w16[&g("attn_k.weight")]), 0);
        enc.set_buffer(3, Some(&self.wt.w16[&g("attn_v.weight")]), 0);
        enc.set_buffer(4, Some(&self.st.q), 0);
        enc.set_buffer(5, Some(&self.st.k), 0);
        enc.set_buffer(6, Some(&self.st.v), 0);
        enc.set_bytes(7, 4, &k_in as *const u32 as *const c_void);
        enc.set_bytes(8, 4, &nq as *const u32 as *const c_void);
        enc.set_bytes(9, 4, &nkv as *const u32 as *const c_void);
        enc.set_buffer(13, Some(&self.wt.w32[&g("attn_q.bias")]), 0);
        enc.set_buffer(14, Some(&self.wt.w32[&g("attn_k.bias")]), 0);
        enc.set_buffer(15, Some(&self.wt.w32[&g("attn_v.bias")]), 0);
        let total = nq + 2 * nkv;
        enc.dispatch_thread_groups(MTLSize::new(((total + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Fused Q/K/V for the native Q4 path (nibbles + f16 scale/min + bias).
    ///
    /// Fused Q4L q/k/v projection — one dispatch for all three, reading Q4_K's own
    /// values rather than a Q8 requantization (0.5625 vs 1.0625 B/weight).
    pub(crate) fn qkv_q4l(&self, enc: &metal::ComputeCommandEncoderRef, lp: &dyn Fn(&str) -> String,
                          x: &metal::Buffer, k: u32, nq: u32, nkv: u32) {
        self.check_qkv(lp, k, nq, nkv);
        let (qn, kn, vn) = (lp("attn_q.weight"), lp("attn_k.weight"), lp("attn_v.weight"));
        enc.set_compute_pipeline_state(&self.p["qkv_q4l"]);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(&self.wt.w4l[&qn]), 0);
        enc.set_buffer(2, Some(&self.wt.w4l[&kn]), 0);
        enc.set_buffer(3, Some(&self.wt.w4l[&vn]), 0);
        enc.set_buffer(4, Some(&self.st.q), 0);
        enc.set_buffer(5, Some(&self.st.k), 0);
        enc.set_buffer(6, Some(&self.st.v), 0);
        enc.set_bytes(7, 4, &k as *const u32 as *const c_void);
        enc.set_bytes(8, 4, &nq as *const u32 as *const c_void);
        enc.set_bytes(9, 4, &nkv as *const u32 as *const c_void);
        enc.set_buffer(10, Some(&self.wt.q4l_a[&qn]), 0);
        enc.set_buffer(11, Some(&self.wt.q4l_b[&qn]), 0);
        enc.set_buffer(12, Some(&self.wt.q4l_a[&kn]), 0);
        enc.set_buffer(13, Some(&self.wt.q4l_b[&kn]), 0);
        enc.set_buffer(14, Some(&self.wt.q4l_a[&vn]), 0);
        enc.set_buffer(15, Some(&self.wt.q4l_b[&vn]), 0);
        // Bias is optional (Qwen2 has it, Llama/Qwen3 do not). `ones` is a harmless
        // stand-in buffer when has_bias is 0 — the kernel never reads it.
        let qb = lp("attn_q.bias");
        let has_bias = self.wt.w32.contains_key(&qb) as u32;
        let bq = if has_bias == 1 { &self.wt.w32[&qb] } else { &self.st.ones };
        let bk = if has_bias == 1 { &self.wt.w32[&lp("attn_k.bias")] } else { &self.st.ones };
        let bv = if has_bias == 1 { &self.wt.w32[&lp("attn_v.bias")] } else { &self.st.ones };
        enc.set_buffer(16, Some(bq), 0);
        enc.set_buffer(17, Some(bk), 0);
        enc.set_buffer(18, Some(bv), 0);
        enc.set_bytes(19, 4, &has_bias as *const u32 as *const c_void);
        let total = nq + 2 * nkv;
        enc.dispatch_thread_groups(MTLSize::new(((total + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub(crate) fn qkv_q4(&self, enc: &metal::ComputeCommandEncoderRef, lp: &dyn Fn(&str) -> String,
              h: &metal::Buffer, k_in: u32, nq: u32, nkv: u32) {
        self.check_qkv(lp, k_in, nq, nkv);
        let g = |s: &str| lp(s);
        let ksplit = k_in >= 2048; // long K → split blocks across simdgroups
        enc.set_compute_pipeline_state(&self.p[if ksplit { "qkv_q4_ksplit" } else { "qkv_q4" }]);
        enc.set_buffer(0, Some(h), 0);
        enc.set_buffer(1, Some(&self.wt.w4[&g("attn_q.weight")]), 0);
        enc.set_buffer(2, Some(&self.wt.w4[&g("attn_k.weight")]), 0);
        enc.set_buffer(3, Some(&self.wt.w4[&g("attn_v.weight")]), 0);
        enc.set_buffer(4, Some(&self.st.q), 0);
        enc.set_buffer(5, Some(&self.st.k), 0);
        enc.set_buffer(6, Some(&self.st.v), 0);
        enc.set_bytes(7, 4, &k_in as *const u32 as *const c_void);
        enc.set_bytes(8, 4, &nq as *const u32 as *const c_void);
        enc.set_bytes(9, 4, &nkv as *const u32 as *const c_void);
        enc.set_buffer(10, Some(&self.wt.scale4[&g("attn_q.weight")]), 0);
        enc.set_buffer(11, Some(&self.wt.scale4[&g("attn_k.weight")]), 0);
        enc.set_buffer(12, Some(&self.wt.scale4[&g("attn_v.weight")]), 0);
        enc.set_buffer(13, Some(&self.wt.w32[&g("attn_q.bias")]), 0);
        enc.set_buffer(14, Some(&self.wt.w32[&g("attn_k.bias")]), 0);
        enc.set_buffer(15, Some(&self.wt.w32[&g("attn_v.bias")]), 0);
        let total = nq + 2 * nkv;
        // ksplit: 1 row/tg (grid=total); plain: 8 rows/tg (grid=total/8).
        let grid = if ksplit { total as u64 } else { ((total + 7) / 8) as u64 };
        enc.dispatch_thread_groups(MTLSize::new(grid, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// `enc_reduce` with a byte offset per buffer, for addressing one token's row
    /// of a batched scratch buffer.
    pub(crate) fn enc_reduce_off(&self, enc: &metal::ComputeCommandEncoderRef, kern: &str,
                                 bufs: &[(&metal::Buffer, u64, u64)], ints: &[(u32, u32)],
                                 floats: &[(u32, f32)], groups: u64, per: u64) {
        enc.set_compute_pipeline_state(&self.p[kern]);
        for (b, idx, off) in bufs { enc.set_buffer(*idx, Some(b), *off); }
        for (idx, val) in ints { enc.set_bytes(*idx as u64, 4, val as *const u32 as *const c_void); }
        for (idx, val) in floats { enc.set_bytes(*idx as u64, 4, val as *const f32 as *const c_void); }
        enc.dispatch_thread_groups(metal::MTLSize::new(groups, 1, 1), metal::MTLSize::new(per, 1, 1));
    }

    pub(crate) fn enc_reduce(&self, enc: &metal::ComputeCommandEncoderRef, kern: &str, bufs: &[(&metal::Buffer, u64)], ints: &[(u32, u32)], floats: &[(u32, f32)], groups: u64, per: u64) {
        enc.set_compute_pipeline_state(&self.p[kern]);
        for (b, idx) in bufs {
            enc.set_buffer(*idx, Some(b), 0);
        }
        for (idx, val) in ints {
            enc.set_bytes(*idx as u64, 4, val as *const u32 as *const c_void);
        }
        for (idx, val) in floats {
            enc.set_bytes(*idx as u64, 4, val as *const f32 as *const c_void);
        }
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(per, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    /// Dispatch a p[8]-limited batched GEMV kernel (`*_m_q8`) over M rows in ≤8-row
    /// tiles. Those kernels accumulate into a `float p[8]` register array and overflow
    /// for M>8 (of the batched paths only gpt-oss/qwen35 chunked; dense-qwen2 did
    /// not). Row-major buffers are offset per tile: `rowbufs` = (buffer_index,
    /// buffer, row_stride_elems); stride 0 binds at offset 0 (weights/scales/bias).
    /// `mint_idx` is the M constant slot.
    pub(crate) fn tile8(&self, enc: &metal::ComputeCommandEncoderRef, kern: &str, m: u32, groups: u64,
             rowbufs: &[(u64, &metal::Buffer, u32)], ints: &[(u64, u32)], floats: &[(u64, f32)],
             mint_idx: u64) {
        let mut r0 = 0u32;
        while r0 < m {
            let mt = (m - r0).min(8);
            enc.set_compute_pipeline_state(&self.p[kern]);
            for &(idx, buf, stride) in rowbufs {
                let off = if stride == 0 { 0 } else { (r0 as u64) * (stride as u64) * 4 };
                enc.set_buffer(idx, Some(buf), off);
            }
            for &(idx, val) in ints { enc.set_bytes(idx, 4, &val as *const u32 as *const c_void); }
            for &(idx, val) in floats { enc.set_bytes(idx, 4, &val as *const f32 as *const c_void); }
            enc.set_bytes(mint_idx, 4, &mt as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(256, 1, 1));
            r0 += mt;
        }
    }

    /// Batched Q8 GEMM `y[M,N] = x[M,K]·W[N,K]^T` (+ accum), no bias. Uses the MMA
    /// kernel (weights streamed once, reused across all M tokens) when dims align and
    /// the GPU has simdgroup_matrix; else falls back to the tiled p[8] GEMV. This is
    /// the diffusion/batched-forward speed path.
    ///
    /// Batched matmul by weight name, picking the format. Q4L has its own M-row
    /// kernel; everything else falls through to the Q8 path. Returns false when the
    /// tensor is in neither, so callers can bail rather than index a missing key.
    ///
    /// Guards the o_proj bug class: a projection GEMM dispatched with the wrong
    /// contraction/output dim (e.g. `d` where the weight is really `n_head*hd`),
    /// silently correct only on models where the two coincide. The GGUF header is the
    /// ground truth. debug_assert, so it is free in release but trips in every test.
    #[inline]
    pub(crate) fn check_shape(&self, name: &str, k: u32, n: u32) {
        if let Some(&(wk, wn)) = self.wt.wshape.get(name) {
            debug_assert!(k == wk && n == wn,
                "GEMM shape mismatch for {name}: caller passed (K={k}, N={n}) but the \
                 weight is (K={wk}, N={wn}). A projection was dispatched with the wrong \
                 dim — check d vs qdim(=n_head*hd) vs kvdim(=n_kv*hd).");
        }
    }

    /// Same guard for a fused qkv dispatch (one kernel reads all three weights):
    /// q is (k_in, nq=qdim), k/v are (k_in, nkv=kvdim). A qdim / kvdim / d mix-up
    /// hides here on non-GQA or qdim==d models.
    #[inline]
    pub(crate) fn check_qkv(&self, lp: &dyn Fn(&str) -> String, k_in: u32, nq: u32, nkv: u32) {
        self.check_shape(&lp("attn_q.weight"), k_in, nq);
        self.check_shape(&lp("attn_k.weight"), k_in, nkv);
        self.check_shape(&lp("attn_v.weight"), k_in, nkv);
    }

    /// Same guard for a MoE expert block. Per-expert the gate/up experts contract
    /// (d, ffn_exp) and the down experts (ffn_exp, d); wshape holds the per-expert
    /// (K, N) for the 3D expert stacks, so this catches a d/ffn_exp swap.
    #[inline]
    pub(crate) fn check_moe(&self, lp: &dyn Fn(&str) -> String, d: u32, fe: u32) {
        self.check_shape(&lp("ffn_gate_exps.weight"), d, fe);
        self.check_shape(&lp("ffn_up_exps.weight"), d, fe);
        self.check_shape(&lp("ffn_down_exps.weight"), fe, d);
    }

    pub(crate) fn gemm_named(&self, enc: &metal::ComputeCommandEncoderRef, name: &str,
                 x: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32, m: u32, accum: bool) -> bool {
        self.check_shape(name, k, n);
        // Native quantized path. The M-row kernel decodes each weight block once per
        // group of 8 tokens, so the codebook work amortizes across the batch, though
        // it cannot use the matrix units the way the Q4L tile GEMM does. Prefill is
        // slower here than after a requant; the trade is that the model fits at all.
        if self.wt.wq.contains_key(name) {
            self.nat_batched(enc, name, x, y, k, n, m, accum);
            return true;
        }
        if let Some(w) = self.wt.w4l.get(name) {
            // Prefer the MMA tile GEMM when the shape allows: it dequantises into
            // threadgroup memory once and then runs on the matrix units, which a
            // gemv-shaped M-row kernel cannot match (83 vs 417 tok/s prefill before
            // this existed).
            //
            // The mrow_max() cut stays at 8. Raising it to 32 is worse — at M=8 the
            // whole batched pass went 52.6 -> 87.3 ms — because the padded GEMM beats
            // gemv_m_q4l even when it throws away 3/4 of the tile, despite the grid
            // collapsing to N/64 threadgroups (32 of them for n=2048 on a 38-core GPU).
            // NK=64 (double K-slab per barrier pair, halving threadgroup barriers)
            // measured prefill floor 193.2 -> 214.4 ms: the 12.5 KB of staged tiles
            // cost more resident threadgroups than the halved barrier count saved. Q8
            // prefill, whose A-fill is a trivial char->half, measures within 1% of Q4L
            // prefill, so the A-tile dequant is not the GEMM's cost either. This 64x32
            // structure caps near 8 TFLOP/s, and only a redesign of the staging (larger
            // per-simdgroup tiles fed without proportional register/shmem growth)
            // moves it.
            //
            // Split-K for long-K narrow-N (ffn_down): grid parallelism the shape
            // cannot otherwise expose. Partition count targets >=1024 threadgroups
            // (the reference qmm_splitk targets 512; this GPU measured best near 1376).
            // OJAS_NO_SK opts out.
            if self.gpu.native_reduce && n % 64 == 0 && k % 32 == 0 && m > mrow_max()
                && k >= sk_kmin() && n <= 4096
                // skbuf holds 8*MAXM*d partials; a big-M (m>MAXM) diffusion forward
                // would overrun it, so those fall back to the plain GEMM.
                && m <= MAXM as u32
                && std::env::var("OJAS_NO_SK").is_err() {
                let tiles = ((m + 31) / 32) * (n / 64);
                let mut nsplit = (1024 / tiles.max(1)).clamp(1, 8);
                while nsplit > 1 && (k / nsplit) < 32 { nsplit -= 1; }
                if nsplit > 1 {
                    // Fat-tile split-K twin (64x64, 16 fragments/simdgroup) instead of
                    // the 64x32 base. Its tile is twice as wide in tokens, so the
                    // partition count is recomputed against the halved tile grid.
                    //
                    // The two effects are orthogonal — split-K supplies the
                    // threadgroups this shape cannot expose, the fat tile supplies
                    // arithmetic intensity per staged fragment — and ffn_down needs
                    // both: fat alone loses here (55.8 vs 50.3 ms). Together, ffn_down
                    // 50.3 -> 47.1 ms and prefill 1454 -> 1481 tok/s over 3
                    // drift-checked rounds. Output is not bit-identical, since
                    // different partition boundaries reassociate the fp32 sum, but the
                    // drift is 0.07% on the vocab checksum and the top1-top2 gap widens
                    // 1.1173 -> 1.1262: reassociation, not precision loss, and the
                    // opposite signature to the rejected f16 accumulator.
                    // Token-identical on the 3B and the 9B hybrid.
                    // OJAS_NO_SKFAT falls back to the 64x32 split-K.
                    let skfat = std::env::var("OJAS_NO_SKFAT").is_err()
                        && self.p.contains_key("gemm_mm_q4l_skfat");
                    if skfat {
                        let tiles_f = ((m + 63) / 64) * (n / 64);
                        let mut ns = (1024 / tiles_f.max(1)).clamp(1, 8);
                        while ns > 1 && (k / ns) < 32 { ns -= 1; }
                        nsplit = ns;
                    }
                    enc.set_compute_pipeline_state(&self.p[if skfat { "gemm_mm_q4l_skfat" } else { "gemm_mm_q4l_sk" }]);
                    enc.set_buffer(0, Some(x), 0);
                    enc.set_buffer(1, Some(w), 0);
                    enc.set_buffer(2, Some(&self.st.skbuf), 0);
                    enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                    enc.set_buffer(5, Some(&self.wt.q4l_a[name]), 0);
                    let ac0 = 0u32;
                    enc.set_bytes(6, 4, &ac0 as *const u32 as *const c_void);
                    enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
                    enc.set_buffer(8, Some(&self.wt.q4l_b[name]), 0);
                    enc.set_bytes(9, 4, &nsplit as *const u32 as *const c_void);
                    let mtile = if skfat { 64 } else { 32 };
                    enc.dispatch_thread_groups(
                        MTLSize::new(((m + mtile - 1) / mtile) as u64, (n / 64) as u64, nsplit as u64),
                        MTLSize::new(128, 1, 1));
                    self.barc(enc);
                    let total = m * n;
                    let ac = accum as u32;
                    self.enc_reduce(enc, "splitk_accum",
                        &[(&self.st.skbuf, 0), (y, 1)],
                        &[(2, total), (3, nsplit), (4, ac)], &[],
                        ((total + 255) / 256) as u64, 256);
                    return true;
                }
            }
            // Huge-activation-slab shapes (ffn_down: K >= 4096, accum): convert the
            // activations to half once and run the half-B tile. The slab is re-read by
            // every N-tile, so halving its bytes should pay for the convert many times
            // over. Needs a barrier between convert and GEMM under the concurrent
            // batch encoder; barc() is harmless on a serial one.
            if self.gpu.native_reduce && n % 64 == 0 && k % 32 == 0 && m > mrow_max()
                && k >= 4096 && accum
                // Opt-in: measured null (57.1 vs 56.7 ms). The slab re-reads were
                // already SLC-served, so halving their bytes only bought back the
                // convert cost. ffn_down's 7.3 TFLOP/s is tile-count starvation (256
                // threadgroups vs ffn_gu's 1376; the 128-tg fat tile was slower
                // still), not bandwidth.
                && std::env::var("OJAS_HB").is_ok() {
                self.enc_reduce(enc, "copy_f32_half", &[(x, 0), (&self.st.xh, 1)],
                    &[(2, m * k)], &[], ((m * k + 255) / 256) as u64, 256);
                self.barc(enc);
                enc.set_compute_pipeline_state(&self.p["gemm_mm_q4l_hb"]);
                enc.set_buffer(0, Some(&self.st.xh), 0);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(y), 0);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.set_buffer(5, Some(&self.wt.q4l_a[name]), 0);
                let ac = accum as u32;
                enc.set_bytes(6, 4, &ac as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
                enc.set_buffer(8, Some(&self.wt.q4l_b[name]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((m + 31) / 32) as u64, (n / 64) as u64, 1), MTLSize::new(128, 1, 1));
                return true;
            }
            // Fat-tile GEMM (custom fragment storage, 64x64, 4 simdgroups, 16 fragments
            // each — a config the opaque simdgroup_matrix API cannot hold in
            // registers). Shape-routed to where it measured faster: large-N non-accum
            // matmuls, i.e. ffn_gate/ffn_up (2.555 vs 2.633 ms/call). On small N
            // (qkv's 512-row k/v: 21.8 vs 17.4) and accum/long-K (ffn_down: 1.647 vs
            // 1.570) the 64x32 base wins, because more tiles beats more intensity
            // there. A K-slab-64 twin measured 107.5 vs 92.0 on ffn_gu: 18.4 KB of
            // staged tiles kills occupancy, again showing barrier count is not this
            // GEMM's cost.
            //
            // OJAS_NO_GEMM_FAT opts out for A/B. OJAS_FAT_ALL relaxes the shape gate
            // so the fat tile can be measured on ffn_down (n=2048, accum), where it
            // lost in an earlier round that predates both split-K and fatx2.
            let fat_all = std::env::var("OJAS_FAT_ALL").is_ok();
            if self.gpu.native_reduce && n % 64 == 0 && k % 32 == 0 && m > mrow_max()
                && (fat_all || (n >= 4096 && !accum))
                && self.p.contains_key("gemm_mm_q4l_fat") && !no_gemm_fat() {
                // fatx (activation fragments straight from device) measured fastest:
                // 91.5 vs 92.2 (fat) vs 94.8 (base) ms on ffn_gu. OJAS_FAT_STAGED
                // falls back to the staged-B variant.
                let staged = std::env::var("OJAS_FAT_STAGED").is_ok();
                // OJAS_FAT_H=1: f16 accumulators (the M1/M2 config) — half the
                // accumulator registers, at the cost of f16 accumulation over K.
                let fatk = if staged { "gemm_mm_q4l_fat" }
                    else if std::env::var("OJAS_FAT_H").is_ok() && self.p.contains_key("gemm_mm_q4l_fatxh") { "gemm_mm_q4l_fatxh" }
                    // Double-buffered A tile: one barrier per K-slab instead of
                    // two. Bit-identical output, +1% end-to-end (1460 -> 1473
                    // tok/s over 3 interleaved rounds; ffn_gu 91.9 -> 90.7 ms).
                    // It costs 4.6 KB of threadgroup memory that fatx was not
                    // using and no registers, which is the resource the other
                    // variants here run out of.
                    else if !std::env::var("OJAS_NO_FAT2").is_ok() && self.p.contains_key("gemm_mm_q4l_fatx2") { "gemm_mm_q4l_fatx2" }
                    else { "gemm_mm_q4l_fatx" };
                enc.set_compute_pipeline_state(&self.p[fatk]);
                let fat8 = false;
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(y), 0);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.set_buffer(5, Some(&self.wt.q4l_a[name]), 0);
                let ac = accum as u32;
                enc.set_bytes(6, 4, &ac as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
                enc.set_buffer(8, Some(&self.wt.q4l_b[name]), 0);
                if fat8 {
                    enc.dispatch_thread_groups(MTLSize::new(((m + 127) / 128) as u64, (n / 64) as u64, 1), MTLSize::new(256, 1, 1));
                } else {
                    enc.dispatch_thread_groups(MTLSize::new(((m + 63) / 64) as u64, (n / 64) as u64, 1), MTLSize::new(128, 1, 1));
                }
                return true;
            }
            // A 64x64 2-simdgroup GEMM, behind OJAS_GEMM_MLX. As written it spills:
            // mc[8][4] = 32 simdgroup_float8x8 accumulators is beyond what the
            // black-box simdgroup_matrix API keeps in registers (35 ms/call vs 2.6
            // base, static indices notwithstanding). Holding 32+ fragments requires
            // custom fragment storage (morton-order per-lane offsets, plain float2
            // loads) instead of simdgroup_load; the kernel is kept as the starting
            // point for that port.
            if self.gpu.native_reduce && n % 64 == 0 && k % 32 == 0 && m > mrow_max()
                && gemm_mlx() {
                enc.set_compute_pipeline_state(&self.p["gemm_mm_q4l_mlx"]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(y), 0);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.set_buffer(5, Some(&self.wt.q4l_a[name]), 0);
                let ac = accum as u32;
                enc.set_bytes(6, 4, &ac as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
                enc.set_buffer(8, Some(&self.wt.q4l_b[name]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((m + 63) / 64) as u64, (n / 64) as u64, 1), MTLSize::new(64, 1, 1));
                return true;
            }
            // Async-copy GEMM, opt-in (OJAS_GEMM_AC=1) because it measured net negative
            // for this kernel: 193.7 -> 227.3 ms prefill floor as first built
            // (row-major A + transposed MMA loads), 202.1 with the blocked A restored.
            // The activation tile is only 2 KB, so its cooperative staging was never a
            // cost, and the big tile (weights) cannot be DMA'd because the copy engine
            // cannot dequantize. MPS-style kernels win with async copy on dense f16
            // GEMMs where both tiles are large plain copies. Byte-identical output was
            // verified (ebb79eb8), so the plumbing is sound if a dense-f16 path ever
            // wants it.
            if self.gpu.native_reduce && n % 64 == 0 && k % 32 == 0 && m > mrow_max()
                && self.p.contains_key("gemm_mm_q4l_ac") && gemm_ac_on() {
                self.enc_reduce(enc, "copy_f32_half", &[(x, 0), (&self.st.xh, 1)],
                    &[(2, m * k)], &[], ((m * k + 255) / 256) as u64, 256);
                self.barc(enc); // GEMM reads st.xh — order the convert before it (concurrent encoder)
                enc.set_compute_pipeline_state(&self.p["gemm_mm_q4l_ac"]);
                enc.set_buffer(0, Some(&self.st.xh), 0);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(y), 0);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.set_buffer(5, Some(&self.wt.q4l_a[name]), 0);
                let ac = accum as u32;
                enc.set_bytes(6, 4, &ac as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
                enc.set_buffer(8, Some(&self.wt.q4l_b[name]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((m + 31) / 32) as u64, (n / 64) as u64, 1), MTLSize::new(128, 1, 1));
                return true;
            }
            if self.gpu.native_reduce && n % 64 == 0 && k % 32 == 0 && m > mrow_max() {
                // 64x32, not 64x64. A 64x64 twin halves the A-tile dequant cost per
                // output and is still slower (ffn_gu 95.4 -> 98.4 ms, ffn_down 56.6 ->
                // 60.3, whole pass 193.5 -> 201.1): mc goes from 8 to 16 simdgroup
                // accumulators, 32 floats per lane, and the occupancy that costs
                // outweighs the fill it saves. Same shape of result as the 4-row Q20
                // experiment noted in prelude.rs.
                enc.set_compute_pipeline_state(&self.p["gemm_mm_q4l"]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(y), 0);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.set_buffer(5, Some(&self.wt.q4l_a[name]), 0);
                let ac = accum as u32;
                enc.set_bytes(6, 4, &ac as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
                enc.set_buffer(8, Some(&self.wt.q4l_b[name]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((m + 31) / 32) as u64, (n / 64) as u64, 1), MTLSize::new(128, 1, 1));
                return true;
            }
            // A 64x8 MMA tile for this range is worse: verify M=4 went 22.5 -> 35.4 ms,
            // ffn_down 5.0 -> 13.3. Shrinking mc from 8 to 2 quarters the matrix work
            // but leaves the A-tile dequant fill unchanged, so the fill-to-compute
            // ratio gets 4x worse and the staging dominates. Small M wants the
            // register-blocked gemv below, which dequants inline per lane and never
            // stages a tile.
            if m <= 8 {
                // 4 rows x 4 tokens up to M=4, then 2 rows x 8 tokens — same 16
                // accumulators per lane either way.
                let wide = m > 4;
                // gemv_mv<M>_q4l hoists the nibble unpack out of the token loop (see
                // the kernel comment); M is compile-time so the accumulators stay in
                // registers. OJAS_MV=0 falls back for A/B.
                // OJAS_MPROBE=alu|ld swaps in the ablation kernels: same loads with
                // the arithmetic gutted, or same arithmetic with the second token's
                // activation loads gone. Both are numerically wrong by construction
                // and are timing instruments only.
                let probe = std::env::var("OJAS_MPROBE").ok();
                // Lane-partitioned kernel (the reference mul_mv_ext shape): NXPSG lanes
                // per row, so a lane holds one row and the dequant hoists across
                // tokens. Needs K % 32 == 0 for the chunk->block map. Shape-routed,
                // because it wins only where the row-blocked kernel is weakest and the
                // winning NXPSG differs by shape (see the table in gemv.rs): wide-N
                // ffn_gate/ffn_up take 4 lanes/row, long-K narrow-N ffn_down takes 8,
                // and qkv and o_proj keep the row-blocked kernel, faster on both.
                let xn = match xv_nxpsg() {
                    0 => 0,
                    v if v != 8 => v,                       // OJAS_XV pins a value
                    // Autotuned per (K,N) per device when available. The fallback below
                    // is the M2 Max hand-tuning and is only a guess on another GPU: the
                    // winner is decided by threadgroups-produced (256/NXPSG rows per tg)
                    // against threadgroups-needed, and the latter is a property of the
                    // chip, not of the shape.
                    _ if std::env::var("OJAS_NO_XVTUNE").is_ok() => {
                        if n >= 4096 { 4 } else if k >= 4096 && n <= 4096 { 8 } else { 0 }
                    }
                    _ => match self.tune.xv_plan.get(&(k, n)) {
                        Some(&v) => v,
                        None if n >= 4096 => 4,
                        None if k >= 4096 && n <= 4096 => 8,
                        None => 0,
                    },
                };
                let xname = format!("gemv_x{xn}_{m}_q4l");
                let xv_ok = xn > 0 && k % 32 == 0 && self.p.contains_key(&xname);
                let kname: &str = match probe.as_deref() {
                    Some("alu") => "gemv_m4_alu",
                    Some("ld") => "gemv_m4_ld",
                    Some("old") if wide => "gemv_m8_q4l",
                    Some("old") => "gemv_m4_q4l",
                    _ if xv_ok => &xname,
                    _ if wide => "gemv_m8_q4l",
                    _ => "gemv_m4_q4l",
                };
                enc.set_compute_pipeline_state(&self.p[kname]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(y), 0);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.set_buffer(5, Some(&self.wt.q4l_a[name]), 0);
                enc.set_buffer(6, Some(&self.wt.q4l_b[name]), 0);
                enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
                let ac = accum as u32;
                enc.set_bytes(8, 4, &ac as *const u32 as *const c_void);
                let t = self.q4l_threads(k, n);
                // Rows per threadgroup must come from the kernel actually chosen, not
                // from whether the lane-partitioned one was eligible: deriving it from
                // `xv_ok` while a probe overrode `kname` dispatches half the
                // threadgroups for the row-blocked kernels, which then compute half the
                // output rows and measure ~35% "faster".
                // m4/alu/ld: 4 rows per simdgroup pair (tokens split across the pair);
                // m8: 2 rows per simdgroup; x<n>: t/n rows.
                let rows = if kname.starts_with("gemv_x") { t / xn }
                    else if kname == "gemv_m8_q4l" { t / 32 * 2 }
                    else { t / 64 * 4 };
                enc.dispatch_thread_groups(MTLSize::new(((n + rows - 1) / rows) as u64, 1, 1), MTLSize::new(t as u64, 1, 1));
                return true;
            }
            enc.set_compute_pipeline_state(&self.p["gemv_m_q4l"]);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(w), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            enc.set_buffer(5, Some(&self.wt.q4l_a[name]), 0);
            enc.set_buffer(6, Some(&self.wt.q4l_b[name]), 0);
            enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
            let ac = accum as u32;
            enc.set_bytes(8, 4, &ac as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(((n + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            return true;
        }
        // Native Q4_K: the fat GEMM at prefill widths, else `mm()`'s own Q4_K
        // contract row by row.
        if let Some(w) = self.wt.w4k.get(name) {
            let off = self.wt.w_off.get(name).copied().unwrap_or(0);
            if self.kquant_fat(12, enc, x, w, off, y, k, n, m, accum) { return true; }
            for row in 0..m as u64 {
                enc.set_compute_pipeline_state(&self.p[if accum { "gemv_q4k_accum" } else { "gemv_q4k" }]);
                enc.set_buffer(0, Some(x), row * (k as u64) * 4);
                enc.set_buffer(1, Some(w), off);
                enc.set_buffer(2, Some(y), row * (n as u64) * 4);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((n + 1) / 2) as u64, 1, 1), MTLSize::new(64, 1, 1));
            }
            return true;
        }
        // Native Q6_K (a Q4_K_M file's attn_qkv, kept native at prec 3). Without a
        // batched branch for this store, qwen35's chunk prefill asserts "missing MTP
        // matrix dispatch" on the first prompt. Prefill widths take the fat GEMM;
        // below them this loops `mm()`'s own Q6_K contract row by row.
        if let Some(w) = self.wt.w6k.get(name) {
            let off = self.wt.w_off.get(name).copied().unwrap_or(0);
            if self.kquant_fat(14, enc, x, w, off, y, k, n, m, accum) { return true; }
            for row in 0..m as u64 {
                enc.set_compute_pipeline_state(&self.p[if accum { "gemv_q6k_accum" } else { "gemv_q6k" }]);
                enc.set_buffer(0, Some(x), row * (k as u64) * 4);
                enc.set_buffer(1, Some(w), off);
                enc.set_buffer(2, Some(y), row * (n as u64) * 4);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((n + 1) / 2) as u64, 1, 1), MTLSize::new(64, 1, 1));
            }
            return true;
        }
        match (self.wt.w8.get(name), self.wt.scale8.get(name)) {
            (Some(w), Some(sc)) => {
                self.gemm8(enc, x, w, sc, y, k, n, m, accum);
                true
            }
            // No bucket held `name`. Every caller ignores this `false` (`let _ =`), so a
            // missing weight leaves the output buffer stale instead of erroring — which
            // is how an MLA arch (no attn_k/attn_v) slipping into the dense batched
            // graph corrupts rather than crashes. A weight the dense graph asks for must
            // exist, so trip in debug at the dispatch rather than chasing garbage
            // downstream.
            _ => {
                debug_assert!(false,
                    "gemm_named: no weight/handler for {name} — a forward path is \
                     dispatching a projection this arch does not have (e.g. an MLA \
                     model routed through the dense graph).");
                false
            }
        }
    }

    pub(crate) fn gemm8(&self, enc: &metal::ComputeCommandEncoderRef, x: &metal::Buffer, w: &metal::Buffer,
             scale: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32, m: u32, accum: bool) {
        self.gemm8_off(enc, x, 0, w, scale, y, k, n, m, accum);
    }

    /// gemm8 with a byte offset into x (to run the GEMM on a row-slice of x, e.g. lm_head
    /// over only the answer rows so logits stay small for long prompts).
    pub(crate) fn gemm8_off(&self, enc: &metal::ComputeCommandEncoderRef, x: &metal::Buffer, x_off: u64,
                 w: &metal::Buffer, scale: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32, m: u32, accum: bool) {
            // The mrow_max() cut stays at 8; raising it to 32 took the whole batched
            // pass from 52.6 to 87.3 ms at M=8. The padded GEMM beats gemv_m_q4l even
            // when it throws away 3/4 of the tile, despite the grid collapsing to N/64
            // threadgroups (32 of them for n=2048 on a 38-core GPU).
            // Prefill widths take the fat tile (64x64, double-buffered weights); a
            // verify step's handful of rows keeps the 32-row base tile. Long-K
            // narrow-N shapes (ffn_down) split K the way the Q4L path does, for the
            // same reason: N = 2560 gives 160 tiles for 38 cores. The partials go to
            // `st.skbuf` and a barrier precedes the reduce; no other GEMM runs beside
            // these accumulating projections. OJAS_NO_Q8_FAT opts out for A/B.
            if self.gpu.native_reduce && n % 64 == 0 && k % 32 == 0 && m > 32
                && self.p.contains_key("gemm_mm_q8_fat") && std::env::var("OJAS_NO_Q8_FAT").is_err() {
                let mut nsplit = 1u32;
                if k >= sk_kmin() && n <= 4096 && m <= MAXM as u32 && std::env::var("OJAS_NO_SK").is_err() {
                    let tiles = m.div_ceil(64) * (n / 64);
                    nsplit = (1024 / tiles.max(1)).clamp(1, 8);
                    let room = self.st.skbuf.length() / 4;
                    while nsplit > 1 && ((k / nsplit) < 32 || (nsplit * m * n) as u64 > room) { nsplit -= 1; }
                }
                enc.set_compute_pipeline_state(&self.p["gemm_mm_q8_fat"]);
                enc.set_buffer(0, Some(x), x_off);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(if nsplit > 1 { &self.st.skbuf } else { y }), 0);
                enc.set_buffer(5, Some(scale), 0);
                for (i, v) in [(3u64, k), (4, n), (6, accum as u32), (7, m), (9, nsplit)] {
                    enc.set_bytes(i, 4, &v as *const u32 as *const c_void);
                }
                enc.dispatch_thread_groups(MTLSize::new(m.div_ceil(64) as u64, (n / 64) as u64, nsplit as u64),
                                           MTLSize::new(128, 1, 1));
                if nsplit > 1 {
                    self.barc(enc);
                    let total = m * n;
                    self.enc_reduce(enc, "splitk_accum", &[(&self.st.skbuf, 0), (y, 1)],
                        &[(2, total), (3, nsplit), (4, accum as u32)], &[], total.div_ceil(256) as u64, 256);
                }
                return;
            }
            if self.gpu.native_reduce && n % 64 == 0 && k % 32 == 0 && m > mrow_max() {
            enc.set_compute_pipeline_state(&self.p["gemm_mm_q8"]);
            enc.set_buffer(0, Some(x), x_off);
            enc.set_buffer(1, Some(w), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            let ac = accum as u32;
            enc.set_bytes(6, 4, &ac as *const u32 as *const c_void);
            enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
            enc.set_buffer(5, Some(scale), 0);
            enc.dispatch_thread_groups(MTLSize::new(((m + 31) / 32) as u64, (n / 64) as u64, 1), MTLSize::new(128, 1, 1));
        } else {
            // non-MMA fallback (family 6): x_off unused (diffusion requires family 7; lm_head
            // dims always take the MMA branch, and other calls pass x_off=0).
            debug_assert!(x_off == 0, "gemm8_off fallback needs x_off==0");
            let kern = if accum { "gemv_m_q8_accum" } else { "gemv_m_q8" };
            self.tile8(enc, kern, m, ((n + 7) / 8) as u64,
                &[(0, x, k), (1, w, 0), (2, y, n), (5, scale, 0)], &[(3, k), (4, n)], &[], 6);
        }
    }

    /// f16 batched GEMM (weights are raw half in w16, no scale). MMA only (family 7).
    pub(crate) fn gemm16_off(&self, enc: &metal::ComputeCommandEncoderRef, x: &metal::Buffer, x_off: u64,
                  w: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32, m: u32, accum: bool) {
        enc.set_compute_pipeline_state(&self.p["gemm_mm_f16"]);
        enc.set_buffer(0, Some(x), x_off);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
        let ac = accum as u32;
        enc.set_bytes(6, 4, &ac as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(((m + 31) / 32) as u64, (n / 64) as u64, 1), MTLSize::new(128, 1, 1));
    }

    /// Projection over a weight stored as Q8 (w8+scale8) or f16 (w16), whichever the
    /// loader chose for it. Serves the diffusion forward at either precision and the
    /// vision tower, which stays f16 at every precision.
    pub(crate) fn projm(&self, enc: &metal::ComputeCommandEncoderRef, x: &metal::Buffer, x_off: u64,
             wname: &str, y: &metal::Buffer, k: u32, n: u32, m: u32, accum: bool) {
        self.check_shape(wname, k, n);
        if let (Some(w), Some(s)) = (self.wt.w8.get(wname), self.wt.scale8.get(wname)) {
            self.gemm8_off(enc, x, x_off, w, s, y, k, n, m, accum);
        } else if self.p.contains_key("gemm_mm_f16_fat") {
            self.gemm_fat(enc, "gemm_mm_f16_fat", x, x_off, &self.wt.w16[wname], 0, y, k, n, m, accum, true);
        } else {
            self.gemm16_off(enc, x, x_off, &self.wt.w16[wname], y, k, n, m, accum);
        }
    }

    /// A `gemm_fat.rs` fat GEMM over f16, Q4_K or Q6_K weights as stored (`entry`
    /// names the format). `split_k` allows split-K into `st.skbuf` for the narrow shapes a
    /// short request cannot fill the GPU with (see
    /// `ojas_metal::kernels::gemm_fat::f16_fat_splits`); it needs a serial encoder,
    /// since the reduce follows the GEMM with no barrier and every split shares one
    /// scratch buffer.
    pub(crate) fn gemm_fat(&self, enc: &metal::ComputeCommandEncoderRef, entry: &str, x: &metal::Buffer, x_off: u64,
                  w: &metal::Buffer, w_off: u64, y: &metal::Buffer, k: u32, n: u32, m: u32, accum: bool, split_k: bool) {
        let scratch = (self.st.skbuf.length() / 4).min(u32::MAX as u64) as u32;
        let nsplit = if split_k { ojas_metal::kernels::gemm_fat::f16_fat_splits(m, n, k, scratch) } else { 1 };
        enc.set_compute_pipeline_state(&self.p[entry]);
        enc.set_buffer(0, Some(x), x_off);
        enc.set_buffer(1, Some(w), w_off);
        enc.set_buffer(2, Some(if nsplit > 1 { &self.st.skbuf } else { y }), 0);
        for (i, v) in [(3u64, k), (4, n), (6, accum as u32), (7, m), (9, nsplit)] {
            enc.set_bytes(i, 4, &v as *const u32 as *const c_void);
        }
        enc.dispatch_thread_groups(MTLSize::new(m.div_ceil(64) as u64, (n / 64) as u64, nsplit as u64),
                                   MTLSize::new(128, 1, 1));
        if nsplit > 1 {
            let total = m * n;
            self.enc_reduce(enc, "splitk_accum", &[(&self.st.skbuf, 0), (y, 1)],
                &[(2, total), (3, nsplit), (4, accum as u32)], &[], total.div_ceil(256) as u64, 256);
        }
    }

    /// `y[M,n] = x[M,k] @ W^T` for a GGUF Q4_K (type 12) or Q6_K (14) matrix at a
    /// prefill width, through the fat GEMM over its blocks as stored. False, having
    /// encoded nothing, for any other type, a short M, a shape the tile does not
    /// cover, or a GPU without simdgroup matrices; the caller then keeps its GEMV.
    ///
    /// Against the batched GEMV it replaces at these widths: Qwen3.5 4B Q4_K_M at
    /// precision 4, 591-token prompt, first token 0.95 s instead of 7.7 s (M2 Max).
    pub(crate) fn kquant_fat(&self, ty: u32, enc: &metal::ComputeCommandEncoderRef, x: &metal::Buffer,
                             w: &metal::Buffer, w_off: u64, y: &metal::Buffer, k: u32, n: u32, m: u32, accum: bool) -> bool {
        let entry = match ty { 12 => "gemm_mm_q4k_fat", 14 => "gemm_mm_q6k_fat", _ => return false };
        if m < 8 || n % 64 != 0 || k % 256 != 0 || !self.gpu.native_reduce || !self.p.contains_key(entry) {
            return false;
        }
        // No split-K: the chunk graph encodes concurrently (gate and up in flight at once).
        self.gemm_fat(enc, entry, x, 0, w, w_off, y, k, n, m, accum, false);
        true
    }

    /// Memory barrier for concurrent-dispatch encoders (qwen35/moe decode uses
    /// MTLDispatchTypeConcurrent so independent kernels overlap; barriers mark real
    /// data dependencies). Only meaningful there — that is the SSM path. Dense models
    /// use a serial encoder, where Metal already orders every dispatch, and these
    /// calls compile out.
    ///
    /// Moving dense onto the concurrent encoder does not pay: with these barriers it
    /// runs 94.8 tok/s against the serial path's 99.4, because an explicit
    /// memoryBarrierWithScope costs more than the serial encoder's implicit ordering.
    /// Without the barriers it reports 132.7 tok/s and fails decode_gate three times
    /// out of three, so that number is a race rather than a result — but it bounds the
    /// prize: sync of any kind costs ~2.6 ms/token, and claiming it needs fewer
    /// dispatches, not cheaper barriers.
    ///
    /// Encode a top-8-of-logits selection (one row) into tmp[1..9].
    pub(crate) fn enc_topk1(&self, enc: &metal::ComputeCommandEncoderRef) {
        enc.set_compute_pipeline_state(&self.p["topk_m"]);
        enc.set_buffer(0, Some(&self.st.logits), 0);
        enc.set_buffer(1, Some(&self.st.tmp), 4);
        let (vocab, m, kk) = (self.arch.vocab as u32, 1u32, 8u32);
        for (bi, v) in [(2u32, vocab), (3, m), (4, kk)] {
            enc.set_bytes(bi as u64, 4, &v as *const u32 as *const c_void);
        }
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Unconditional buffer-scope barrier, for encoders known to be concurrent (the
    /// batched prefill path). `bar()` stays gated for the mixed callers.
    pub(crate) fn barc(&self, enc: &metal::ComputeCommandEncoderRef) {
        unsafe { let _: () = msg_send![enc, memoryBarrierWithScope: 1u64]; }
    }

    pub(crate) fn bar(&self, enc: &metal::ComputeCommandEncoderRef) {
        // qwen4exp scalar decode deliberately opens a serial encoder even when
        // OJAS_SERIAL is unset (entries.rs). Metal supplies dependency ordering
        // there; explicit buffer barriers only add latency.
        if self.arch.ssm.is_some() && !self.cfg.serial && self.arch.qwen4exp.is_none() {
            unsafe { let _: () = msg_send![enc, memoryBarrierWithScope: 1u64]; } // MTLBarrierScopeBuffers
        }
    }

}
