//! CUDA glue for the portable native-quant matvecs.
//!
//! Mirrors `ojas-metal/src/kernels/nat.rs`. The decoders, the format table and the matvec
//! bodies come from `ojas_core::quant_src`, the same text Metal compiles, so a new quantization
//! format is added once and both backends get it. Only the kernel signatures differ, and they
//! are generated from the same table.
//!
//! CUDA launch shape matches the Metal one so the dispatcher logic is shared:
//! one output row per warp, `blockDim.x/32` rows per block, `lane` walking
//! (block, sub-block) work items strided by 32.

use ojas_core::iq_grids;
use ojas_core::quant_src::{prologue, Dialect, QFormat, DECODERS, FORMATS, GEMV_BODIES};

/// Decoders + whole-block forms + matvec bodies in CUDA dialect, plus the warp
/// intrinsics the bodies expect (`QSUM` maps to `warp_sum` from the prelude).
pub fn cuda_decoders() -> String {
    format!("{}\n{}\n{}\n{}",
        prologue(Dialect::Cuda),
        DECODERS,
        ojas_core::quant_src::row_macros(),
        GEMV_BODIES)
}

/// The IQ codebooks the grid formats index, in CUDA dialect.
///
/// Emitted only into the `gemv_iq` family, for the same reason Metal keeps them in
/// `requant_iq`: the tables total ~41 KB of `__constant__` that no other family reads.
pub fn cuda_grids() -> String {
    iq_grids::grids(
        iq_grids::Dialect::Cuda,
        &["kmask_iq2xs", "ksigns_iq2xs", "iq2xxs_grid", "iq2xs_grid", "iq2s_grid",
          "iq3xxs_grid", "iq3s_grid", "iq1s_grid", "iq1s_grid_gpu"],
    )
}

/// Formats whose decoders need no codebook. Compiled into the `gemv` family.
pub fn plain_formats() -> impl Iterator<Item = &'static QFormat> {
    FORMATS.iter().filter(|f| !f.needs_grids)
}

/// Formats whose decoders index an IQ codebook. Compiled into `gemv_iq`, which is the only
/// family carrying the grids.
pub fn grid_formats() -> impl Iterator<Item = &'static QFormat> {
    FORMATS.iter().filter(|f| f.needs_grids)
}

/// Kernel entry name for `ty`, or None when this backend has no native path.
///
/// Grid formats are included. A `None` here would make a caller requantise an IQ3_S tensor
/// instead of reading it natively, which is the cost the native path exists to avoid.
pub fn nat_entry(ty: u32, suffix: &str) -> Option<String> {
    FORMATS.iter().find(|f| f.ty == ty).map(|f| format!("gemv_nat_{}{suffix}", f.tag))
}

/// Every entry name this module emits — for the manifest test.
pub fn names() -> Vec<String> {
    let mut v = vec![];
    for f in FORMATS.iter() {
        for s in ["", "_accum", "_bias", "_m"] {
            v.push(format!("gemv_nat_{}{s}", f.tag));
        }
    }
    v
}

/// `extern "C" __global__` signatures. Generated from the shared table, so the
/// two backends cannot drift in which formats they claim to support.
pub fn instantiate(fmts: impl Iterator<Item = &'static QFormat>) -> String {
    const SIG: &str = "const float* x, const uchar* w, float* y, uint K, uint N";
    // `bs` sits with the other pointers, before the scalars: `KernelRuntime::dispatch` appends
    // buffers then consts, so a trailing pointer would land in a scalar's argument slot.
    const SIG_B: &str = "const float* x, const uchar* w, float* y, const float* bs, uint K, uint N";
    const SIG_M: &str = "const float* x, const uchar* w, float* y, uint K, uint N, uint M, uint ACC";
    // CUDA has no threadgroup-scoped builtins; derive the Metal-shaped ids so the
    // shared bodies compile unchanged.
    const IDS: &str = "    uint tgid = blockIdx.x, ts = blockDim.x;\n    uint sgid = threadIdx.x >> 5, lane = threadIdx.x & 31u;";
    let mut s = String::from("\n// native (no-requant) matvecs — bodies in ojas_core::quant_src\n");
    for f in fmts {
        let (t, w, bb, wp, sb) = (f.tag, f.walker, f.block_bytes, f.weights, f.subs);
        let (sw, nr) = (f.sub_weights(), f.nr0);
        let body = f.fast_body.unwrap_or("NAT_GEMV_BODY");
        s.push_str(&format!("extern \"C\" __global__ void gemv_nat_{t}({SIG}) {{\n{IDS}\n    {body}({w}_SUB, {bb}u, {wp}u, {sb}u, {sw}, {nr}, y[n] = acc)\n}}\n\n"));
        s.push_str(&format!("extern \"C\" __global__ void gemv_nat_{t}_accum({SIG}) {{\n{IDS}\n    {body}({w}_SUB, {bb}u, {wp}u, {sb}u, {sw}, {nr}, y[n] += acc)\n}}\n\n"));
        s.push_str(&format!("extern \"C\" __global__ void gemv_nat_{t}_bias({SIG_B}) {{\n{IDS}\n    {body}({w}_SUB, {bb}u, {wp}u, {sb}u, {sw}, {nr}, y[n] = acc + bs[n])\n}}\n\n"));
        s.push_str(&format!("extern \"C\" __global__ void gemv_nat_{t}_m({SIG_M}) {{\n{IDS}\n    NAT_GEMV_M_DEV({w}_SUB, {bb}u, {wp}u, {sb}u, 8u)\n}}\n\n"));
    }
    s
}
