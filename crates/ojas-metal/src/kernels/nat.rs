//! Metal-side glue for the portable native-quant matvecs.
//!
//! The decoders, the format table and the kernel bodies all live in
//! `ojas_core::quant_src` so Metal and CUDA share them. This file only supplies
//! what is Metal-specific: assembling the source and naming the entry points.

pub use ojas_core::quant_src::{format_of, Dialect, QFormat, FORMATS};

/// Decoders + whole-block forms + matvec bodies, in Metal dialect. Prepended to
/// every kernel family that instantiates a native matvec.
pub fn metal_decoders() -> String {
    format!("{}\n{}\n{}\n{}",
        ojas_core::quant_src::prologue(Dialect::Metal),
        ojas_core::quant_src::DECODERS,
        ojas_core::quant_src::row_macros(),
        ojas_core::quant_src::GEMV_BODIES)
}

/// Kernel entry name for `ty` and one of "", "_accum", "_bias", "_m".
/// None when the format has no native path (F16/F32/BF16, ternary Q2_0).
pub fn nat_entry(ty: u32, suffix: &str) -> Option<String> {
    format_of(ty).map(|f| format!("gemv_nat_{}{suffix}", f.tag))
}

/// Weights per block — the dispatcher needs it for the K % weights == 0 check.
pub fn nat_wpb(ty: u32) -> Option<u32> {
    format_of(ty).map(|f| f.weights)
}

/// (threads per threadgroup, rows per threadgroup) for a native matvec.
///
/// Generic kernels derive their row from the thread count — `row0 =
/// (tgid*(ts/32) + sgid) * nr0` — so their rows per threadgroup are
/// `(threads/32) * nr0`. Specialized Q8 instead assigns `row0 = tgid*nr0` and uses its
/// simdgroups to split K, so it always owns exactly `nr0` rows. Using the generic grid
/// for it silently skips output rows.
pub fn nat_launch(ty: u32) -> Option<(u32, u32)> {
    format_of(ty).map(|f| (f.nsg * 32, if ty == 8 { f.nr0 } else { f.nsg * f.nr0 }))
}

/// One batched-kernel configuration: (kind, chunk columns, M tile, rows per
/// simdgroup).
///
/// Kind 0 = device-read generic, 1 = shared-memory staged, 2 = device-read with
/// x reused in registers, 3 = the M=1 body with MTILE activation rows (walker
/// fusion preserved -- see NAT_GEMV_M_ROW).
///
/// For kind 3, `nr0 == 0` means "use this format's own nr0", so the batched
/// kernel inherits the same per-format launch shape the M=1 kernel was tuned to.
pub type MCfg = (u8, u32, u32, u32);

/// Configurations the `OJAS_TUNE_M` sweep compares. The point of listing the
/// device-read body here is that it is the incumbent: staging costs threadgroup
/// memory (which caps occupancy) and two barriers per chunk, and whether that
/// buys anything is a measurement.
pub const M_VARIANTS: &[MCfg] = &[
    (0, 0, 8, 1),      // v0: device-read generic — the incumbent
    (1, 512, 4, 8),    // v1: best staged config (lost; kept measurable)
    (2, 0, 2, 4),      // v2: register-reuse (lost; kept measurable)
    (3, 0, 2, 0),      // v3: walker-fused, 2 rows of x, format's own nr0
    (3, 0, 2, 1),      // v4: walker-fused, 2 rows, one row per simdgroup
    (3, 0, 2, 2),      // v5: walker-fused, 2 rows, two rows per simdgroup
    (3, 0, 4, 0),      // v6: walker-fused, 4 rows of x
    (3, 0, 4, 1),      // v7: walker-fused, 4 rows, one row per simdgroup
];

/// The best config the sweep has measured. Changing it changes the shipped kernel, so
/// it stays next to the variants it was chosen from.
///
/// v3 — the M=1 body with two activation rows, at the format's own nr0/nsg.
/// Sum of per-category GPU time at M=2 on Qwen3.8-27B UD-IQ2_XXS:
///
/// ```text
///   v0 generic device-read     121.8 ms
///   v3 walker-fused MTILE=2     72.8      <- shipped
///   v4 walker-fused nr0=1      138.1
///   v5 walker-fused nr0=2       86.3
///   v6 walker-fused MTILE=4    343.3
///   loop of the M=1 kernel      82.2
/// ```
///
/// MTILE=4 falls off a cliff because `_yl[4][32]` no longer fits in registers.
pub const M_DEFAULT: MCfg = M_VARIANTS[3];

fn m_body_call(f: &QFormat, c: MCfg) -> String {
    let (w, bb, wp, sb) = (f.walker, f.block_bytes, f.weights, f.subs);
    let (staged, chunk, mtile) = (c.0, c.1, c.2);
    let (sw, nr0) = (f.sub_weights(), m_nr0(f, c));
    match staged {
        3 => format!("NAT_GEMV_M_ROW({w}_SUB, {bb}u, {wp}u, {sb}u, {sw}, {nr0}, {mtile}u)"),
        1 => format!("NAT_GEMV_M_STAGED({w}_SUB, {bb}u, {wp}u, {sb}u, {chunk}u, {mtile}u, {nr0}u)"),
        2 => format!("NAT_GEMV_M_REG({w}_SUB, {bb}u, {wp}u, {sb}u, {mtile}u, {nr0}u)"),
        _ => format!("NAT_GEMV_M_DEV({w}_SUB, {bb}u, {wp}u, {sb}u, {mtile}u)"),
    }
}

/// Rows per simdgroup for `c` on format `f` — resolves kind 3's "0 means the
/// format's own nr0". Kind 0 is fixed at one row per simdgroup.
fn m_nr0(f: &QFormat, c: MCfg) -> u32 {
    match c.0 {
        0 => 1,
        3 if c.3 == 0 => f.nr0,
        _ => c.3,
    }
}

/// (threads, rows per threadgroup) for a batched config on a given format.
///
/// Kind 3 inherits the format's `nsg` as well as its `nr0`, because those two
/// were tuned together for the M=1 kernel and it is the same body.
pub fn m_launch(f: &QFormat, c: MCfg) -> (u32, u32) {
    let threads = if c.0 == 3 { f.nsg * 32 } else { 256 };
    (threads, (threads / 32).max(1) * m_nr0(f, c))
}

/// (threads, rows per threadgroup) for the SHIPPED batched kernel on `ty`.
///
/// Callers must take the grid from this and not from the M=1 `nat_launch`: the two
/// disagree whenever the batched config uses a different rows-per-simdgroup, and the
/// wrong one silently drops output rows.
pub fn nat_launch_m(ty: u32) -> Option<(u32, u32)> {
    format_of(ty).map(|f| m_launch(f, m_default()))
}

/// The batched config in force. OJAS_NAT_MVAR selects one of `M_VARIANTS` for
/// sweeping; unset uses `M_DEFAULT`.
pub fn m_default() -> MCfg {
    static V: std::sync::OnceLock<MCfg> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("OJAS_NAT_MVAR").ok().and_then(|s| s.parse::<usize>().ok())
            .and_then(|i| M_VARIANTS.get(i).copied()).unwrap_or(M_DEFAULT)
    })
}

/// Activation rows the batched kernel pads to. Callers must not hand it an M that is
/// not a multiple of this.
///
/// Only kind 3 pads: it clamps the ragged rows to the last valid one and computes them
/// twice, so M=3 costs what M=4 costs. The other bodies carry a runtime
/// `mm = min(MTILE, M-m0)` guard and take any M, so they report 1 and are never split.
///
/// A wrong value corrupts nothing but measures the wrong kernel: returning 8 for the
/// generic body made `m - m % 8` zero at M=2, so a sweep of the generic batched kernel
/// ran the loop instead and reported the loop's time under the batched kernel's name.
pub fn m_mtile() -> u32 { let c = m_default(); if c.0 == 3 { c.2 } else { 1 } }

/// Entry name for the batched kernel in force: the plain `_m` when it is the
/// default config, else the sweep variant's own entry point.
pub fn m_entry(tag: &str) -> String {
    match std::env::var("OJAS_NAT_MVAR").ok().and_then(|s| s.parse::<usize>().ok()) {
        Some(i) if i < M_VARIANTS.len() => format!("gemv_nat_{tag}_m_v{i}"),
        _ => format!("gemv_nat_{tag}_m"),
    }
}

/// Same, for an overridden thread count (sweeping).
pub fn nat_launch_with(ty: u32, threads: u32) -> Option<(u32, u32)> {
    if threads == 0 || threads % 32 != 0 || (ty == 8 && threads > 128) { return None; }
    format_of(ty).map(|f| (threads, if ty == 8 { f.nr0 } else { (threads / 32).max(1) * f.nr0 }))
}

/// Formats whose decoders need the IQ codebooks (they live in `requant_iq`).
pub fn grid_formats() -> impl Iterator<Item = &'static QFormat> {
    FORMATS.iter().filter(|f| f.needs_grids)
}

/// Formats whose decoders need no codebook (compiled into the `gemv` family).
pub fn plain_formats() -> impl Iterator<Item = &'static QFormat> {
    FORMATS.iter().filter(|f| !f.needs_grids)
}

/// The `kernel void ...` signatures for one set of formats. Generated rather
/// than written: 4 forms x 19 formats, and the names must be literal text for
/// the kernel registry's scan to find them.
pub fn instantiate(fmts: impl Iterator<Item = &'static QFormat>) -> String {
    const SIG: &str = "device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],\n    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],\n    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],\n    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]";
    const SIG_B: &str = "device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],\n    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],\n    device const float* bs [[buffer(5)]],\n    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],\n    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]";
    const SIG_M: &str = "device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],\n    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],\n    constant uint& M [[buffer(7)]], constant uint& ACC [[buffer(8)]],\n    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],\n    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]";
    let mut s = String::from("\n// native (no-requant) matvecs — bodies in ojas_core::quant_src\n");
    // Tuning variants: the same kernel at several rows-per-simdgroup, so the sweep
    // can measure NR rather than assume it. NR sizes `sumf[NR]` and so must be a
    // compile-time constant, hence separate entry points. Gated because 19 formats x 4
    // variants is a lot of kernels for a build that will never dispatch them.
    let tuning = std::env::var("OJAS_TUNE_KERNELS").is_ok();
    for f in fmts {
        let (t, w, bb, wp, sb) = (f.tag, f.walker, f.block_bytes, f.weights, f.subs);
        let (sw, nr) = (f.sub_weights(), f.nr0);
        // Specializations are on by default, disabled with OJAS_NO_FAST. Measured on
        // Qwen3.8-27B UD-IQ2_XXS (100 of ~500 tensors are IQ1_S): 66.85 -> 60.15
        // ms/token, a 10% win at 0.6% control drift.
        //
        // Two ways that measurement goes wrong: a contended machine with an
        // A-then-B ordering reported ~2x slower, and a 0.5B model reported a wash
        // because its FFN is cache-resident, leaving nothing for an ALU-against-memory
        // trade to bite on.
        let body = if std::env::var("OJAS_NO_FAST").is_ok() {
            "NAT_GEMV_BODY"
        } else {
            f.fast_body.unwrap_or("NAT_GEMV_BODY")
        };
        s.push_str(&format!("kernel void gemv_nat_{t}({SIG}) {{\n    {body}({w}_SUB, {bb}u, {wp}u, {sb}u, {sw}, {nr}, y[n] = acc)\n}}\n\n"));
        s.push_str(&format!("kernel void gemv_nat_{t}_accum({SIG}) {{\n    {body}({w}_SUB, {bb}u, {wp}u, {sb}u, {sw}, {nr}, y[n] += acc)\n}}\n\n"));
        s.push_str(&format!("kernel void gemv_nat_{t}_bias({SIG_B}) {{\n    {body}({w}_SUB, {bb}u, {wp}u, {sb}u, {sw}, {nr}, y[n] = acc + bs[n])\n}}\n\n"));
        s.push_str(&format!("kernel void gemv_nat_{t}_m({SIG_M}) {{\n    {mbody}\n}}\n\n",
            mbody = m_body_call(f, M_DEFAULT)));
        if f.ty == 8 {
            for (entry, tile) in [("gemv_nat_q80_m_cooperative", 2), ("gemv_nat_q80_m_cooperative4", 4)] {
                let body = Q80_M_COOPERATIVE.replace("@MTILE@", &tile.to_string());
                s.push_str(&format!("kernel void {entry}({SIG_M}) {{\n{body}\n}}\n"));
            }
        }
        if tuning {
            for (i, cfg) in M_VARIANTS.iter().enumerate() {
                s.push_str(&format!("kernel void gemv_nat_{t}_m_v{i}({SIG_M}) {{\n    {mbody}\n}}\n\n",
                    mbody = m_body_call(f, *cfg)));
            }
        }
        if tuning {
            for r in [1u32, 2, 4, 8] {
                s.push_str(&format!("kernel void gemv_nat_{t}_r{r}({SIG}) {{\n    {body}({w}_SUB, {bb}u, {wp}u, {sb}u, {sw}, {r}, y[n] = acc)\n}}\n\n"));
            }
        }
    }
    s
}

// Eight lanes cooperate on each native Q8_0 block; each lane handles four
// values. Avoids large activation register arrays. GGUF scales stay f16.
const Q80_M_COOPERATIVE: &str = r#"
    uint row0 = (tgid * (ts/32u) + sgid) * 2u;
    if (row0 >= N) return;
    uint nb = K/32u;
    for (uint m0 = 0; m0 < M; m0 += @MTILE@u) {
        float sums[@MTILE@][2] = {};
        for (uint b = lane/8u; b < nb; b += 4u) {
            #pragma unroll
            for (uint r = 0; r < 2u; r++) {
                if (row0+r >= N) break;
                device const uchar* block = w + (ulong(row0+r)*nb+b)*34u;
                float scale = float(*reinterpret_cast<device const half*>(block));
                #pragma unroll
                for (uint j = 0; j < 4u; j++) {
                    uint i = (lane%8u)*4u+j;
                    float v = scale * float(as_type<char>(block[2u+i]));
                    #pragma unroll
                    for (uint m = 0; m < @MTILE@u; m++)
                        sums[m][r] += v * x[ulong(min(m0+m,M-1u))*K+b*32u+i];
                }
            }
        }
        #pragma unroll
        for (uint r = 0; r < 2u; r++) {
            if (row0+r >= N) break;
            #pragma unroll
            for (uint m = 0; m < @MTILE@u; m++) {
                if (m0+m >= M) break;
                float sum = simd_sum(sums[m][r]);
                if (lane == 0u) {
                    ulong index = ulong(m0+m)*N+row0+r;
                    if (ACC) y[index] += sum; else y[index] = sum;
                }
            }
        }
    }
"#;
