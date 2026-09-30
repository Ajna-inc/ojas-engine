//! Portable GGUF block decoders: one source of truth, emitted per GPU dialect.
//!
//! Written once per backend, the same block layouts are one more chance per copy to
//! transpose a nibble, and a wrong weight does not crash — it produces fluent
//! garbage. So the GPU decoders live here in a dialect-neutral C subset, and each
//! backend supplies a small prologue for the few things that differ (address-space
//! qualifiers, the f16 load, the warp reduction). Adding a quantization format is
//! one `*_SUB` macro and one table row, for both backends; adding a backend is one
//! prologue.
//!
//! The sub-block contract: every decoder is `FMT_SUB(WR, BLK, IL, BODY)`, decoding
//! only sub-block `IL` of block `BLK` and handing each weight to BODY as `_v` at
//! column `_idx`, mirroring the reference `dequantize_q4_K(xb, il, reg)`. The split
//! is for speed: walking whole blocks makes one lane decode all 256 weights
//! serially, which measured ~48 GB/s against the tuned requantized path's ~372 GB/s
//! on the same model. With sub-blocks, 8 lanes cooperate on one block at 32 weights
//! each, reading consecutive bytes of it instead of striding a block apart.
//!
//! `FMT_ROW` is a loop over `FMT_SUB`, so the load-time requantizers keep working
//! unchanged and keep gating the decomposition: a wrong `_SUB` split fails
//! `iq_raw_gate` and `requant_gate` on the format that broke.
//!
//! All 256-weight formats decompose into 8 sub-blocks of 32; the 32-weight legacy
//! formats are a single sub-block.

/// GPU source dialect. Both are C++-ish; only qualifiers and intrinsics differ.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dialect {
    Metal,
    Cuda,
}

/// One quantization format: how to find its blocks and how wide a sub-block is.
#[derive(Clone, Copy, Debug)]
pub struct QFormat {
    /// Short kernel-name tag, e.g. "iq2xxs" -> `gemv_nat_iq2xxs`.
    pub tag: &'static str,
    /// The `FMT_SUB` / `FMT_ROW` macro prefix.
    pub walker: &'static str,
    /// GGUF type id.
    pub ty: u32,
    /// Bytes per block on disk.
    pub block_bytes: u32,
    /// Weights per block.
    pub weights: u32,
    /// Sub-blocks per block (lane-cooperation granularity).
    pub subs: u32,
    /// True when the decoder needs the IQ codebooks (kept out of the shared
    /// prelude — they are ~15 KB of constant tables).
    pub needs_grids: bool,
    /// Output rows per simdgroup. More rows amortize the staged activation load
    /// across more work; too many exhaust the register budget.
    ///
    /// Seeded from the reference `N_R0_*`; these are the values behind the
    /// 60.15 ms/token measurement.
    ///
    /// A static table only approximates it: `examples/nat_tune` finds the optimum is
    /// shape-dependent (at K=5120/N=4096 q2k wants nr0=1, at K=17408/N=5120 nr0=8;
    /// iq2xxs moves 2->4 and iq3s 2->4 across the same pair), so one constant per
    /// format cannot serve a model whose FFN alone uses two shapes. Substituting the
    /// single-shape winners gave no end-to-end gain; the fix is per-(K,N) autotuning,
    /// as `autotune.rs` already does for the Q4L/Q8 families.
    pub nr0: u32,
    /// Simdgroups per threadgroup; threads = nsg*32. Shape-dependent like `nr0`:
    /// the sweep prefers 1 (a 32-thread threadgroup) at one shape and 2-4 at
    /// another.
    pub nsg: u32,
    /// Optional specialized matvec body, for formats where the generic one leaves
    /// measurable speed on the table. The generic body consumes the walker's
    /// finished per-weight `_v`, with scale and offset already folded together:
    /// correct for every format, but it forbids the algebraic factoring that makes
    /// IQ1_S fast (see `NAT_GEMV_IQ1S`). That is the cost of one decoder serving
    /// both the requantizer and the matvec, so specializing is opt-in per format
    /// and justified by measurement.
    pub fast_body: Option<&'static str>,
}

impl QFormat {
    /// Weights per sub-block — the unit of work one lane decodes.
    pub const fn sub_weights(&self) -> u32 { self.weights / self.subs }
}

pub const FORMATS: &[QFormat] = &[
    // legacy 32-weight blocks: one sub-block each
    QFormat { tag: "q40",    walker: "Q40",    ty: 2,  block_bytes: 18,  weights: 32,  subs: 1, needs_grids: false , nr0: 2, nsg: 4, fast_body: None },
    QFormat { tag: "q41",    walker: "Q41",    ty: 3,  block_bytes: 20,  weights: 32,  subs: 1, needs_grids: false , nr0: 2, nsg: 4, fast_body: None },
    QFormat { tag: "q50",    walker: "Q50",    ty: 6,  block_bytes: 22,  weights: 32,  subs: 1, needs_grids: false , nr0: 2, nsg: 4, fast_body: None },
    QFormat { tag: "q51",    walker: "Q51",    ty: 7,  block_bytes: 24,  weights: 32,  subs: 1, needs_grids: false , nr0: 2, nsg: 4, fast_body: None },
    QFormat { tag: "q80",    walker: "Q80",    ty: 8,  block_bytes: 34,  weights: 32,  subs: 1, needs_grids: false , nr0: 2, nsg: 4, fast_body: Some("NAT_GEMV_Q80") },
    QFormat { tag: "iq4nl",  walker: "IQ4NL",  ty: 20, block_bytes: 18,  weights: 32,  subs: 1, needs_grids: false , nr0: 2, nsg: 2, fast_body: None },
    // MXFP4 (4-bit float, e.g. gpt-oss). E2M1 nibble + one e8m0 scale per 32;
    // ojas packs it as { u8 scale; u8 qs[16] } = 17 B / 32, one sub-block like the
    // legacy formats. `ty` is synthetic (139): this is a safetensors format, not a
    // GGUF type, so it never collides with a real GGUF type id on the native path.
    QFormat { tag: "mxfp4",  walker: "MXFP4",  ty: 139, block_bytes: 17, weights: 32, subs: 1, needs_grids: false , nr0: 2, nsg: 4, fast_body: None },
    // K-quants: 8 sub-blocks of 32
    QFormat { tag: "q2k",    walker: "Q2K",    ty: 10, block_bytes: 84,  weights: 256, subs: 8, needs_grids: false , nr0: 4, nsg: 2, fast_body: None },
    QFormat { tag: "q3k",    walker: "Q3K",    ty: 11, block_bytes: 110, weights: 256, subs: 8, needs_grids: false , nr0: 2, nsg: 2, fast_body: Some("NAT_GEMV_Q3K") },
    QFormat { tag: "q4k",    walker: "Q4K",    ty: 12, block_bytes: 144, weights: 256, subs: 8, needs_grids: false , nr0: 2, nsg: 2, fast_body: Some("NAT_GEMV_Q4K") },
    QFormat { tag: "q5k",    walker: "Q5K",    ty: 13, block_bytes: 176, weights: 256, subs: 8, needs_grids: false , nr0: 1, nsg: 2, fast_body: Some("NAT_GEMV_Q5K") },
    QFormat { tag: "q6k",    walker: "Q6K",    ty: 14, block_bytes: 210, weights: 256, subs: 8, needs_grids: false , nr0: 2, nsg: 2, fast_body: Some("NAT_GEMV_Q6K") },
    QFormat { tag: "iq4xs",  walker: "IQ4XS",  ty: 23, block_bytes: 136, weights: 256, subs: 8, needs_grids: false , nr0: 2, nsg: 2, fast_body: Some("NAT_GEMV_IQ4XS") },
    // IQ family: 8 sub-blocks of 32, codebook-driven
    QFormat { tag: "iq1s",   walker: "IQ1S",   ty: 19, block_bytes: 50,  weights: 256, subs: 8, needs_grids: true , nr0: 4, nsg: 2, fast_body: Some("NAT_GEMV_IQ1S") },
    QFormat { tag: "iq1m",   walker: "IQ1M",   ty: 29, block_bytes: 56,  weights: 256, subs: 8, needs_grids: true , nr0: 4, nsg: 2, fast_body: None },
    QFormat { tag: "iq2xxs", walker: "IQ2XXS", ty: 16, block_bytes: 66,  weights: 256, subs: 8, needs_grids: true , nr0: 4, nsg: 2, fast_body: None },
    QFormat { tag: "iq2xs",  walker: "IQ2XS",  ty: 17, block_bytes: 74,  weights: 256, subs: 8, needs_grids: true , nr0: 4, nsg: 2, fast_body: None },
    QFormat { tag: "iq2s",   walker: "IQ2S",   ty: 22, block_bytes: 82,  weights: 256, subs: 8, needs_grids: true , nr0: 4, nsg: 2, fast_body: None },
    QFormat { tag: "iq3xxs", walker: "IQ3XXS", ty: 18, block_bytes: 98,  weights: 256, subs: 8, needs_grids: true , nr0: 4, nsg: 2, fast_body: None },
    QFormat { tag: "iq3s",   walker: "IQ3S",   ty: 21, block_bytes: 110, weights: 256, subs: 8, needs_grids: true , nr0: 2, nsg: 2, fast_body: None },
];

pub fn format_of(ty: u32) -> Option<&'static QFormat> {
    FORMATS.iter().find(|f| f.ty == ty)
}

// ---- MoE expert kernels -------------------------------------------------
//
// Separate from `FORMATS`: `QFormat` describes the native matvec (walker,
// nr0/nsg, fast_body), while the MoE expert kernels are a different family with
// their own launch rules, and a format can be usable in one and not the other.
//
// This table covers the native/streamed path only, where the kernel reads the
// GGUF's own blocks. `moe_gu_q4`/`moe_down_q4` (Q4L) and `moe_gu_q8`/`moe_down_q8`
// (int8 + f32 row scale) are requantized layouts rather than GGUF types, so they
// are chosen by representation at the call site and are absent here.

/// One MoE expert kernel: the entry point and the launch shape it needs.
///
/// `launch` is (threads per threadgroup, output rows per threadgroup); the grid is
/// `ceil(rows / launch.1)` threadgroups. It is declared here rather than inferred
/// from the kernel name, so a new IQ kernel with a different row count cannot be
/// mis-dispatched without a diagnostic.
#[derive(Clone, Copy, Debug)]
pub struct MoeKernel {
    pub entry: &'static str,
    pub launch: (u32, u32),
}

/// Which expert projection a tensor feeds. The two roles have different kernel
/// sets — a format valid as gate/up is not necessarily valid as down — so a lookup
/// that ignores the role can hand blocks to another format's walker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRole { GateUp, Down }

/// GGUF type -> expert kernels, per role.
///
/// `None` means no kernel for this format in this role, and callers must refuse the
/// model rather than substitute another format's kernel. A catch-all fallback
/// (`_ => "moe_gu_q4k"`, `_ => "moe_down_q80"`) lets a Q8_0 gate/up or Q4_K down
/// tensor past the loader's allowlist and decodes it with the wrong walker: wrong
/// numbers, no error, no crash.
pub struct MoeFormat {
    pub ty: u32,
    pub gu: Option<MoeKernel>,
    pub down: Option<MoeKernel>,
    /// Batched (M>1) twins, for prefill and speculative verify. `None` means the
    /// format decodes one token at a time, so a caller wanting a batch falls back
    /// to per-token rather than to another format's kernel.
    pub gu_m: Option<MoeKernel>,
    pub down_m: Option<MoeKernel>,
}

const fn mk(entry: &'static str, threads: u32, rows: u32) -> Option<MoeKernel> {
    Some(MoeKernel { entry, launch: (threads, rows) })
}

pub const MOE_FORMATS: &[MoeFormat] = &[
    // down: the q80-family kernels decode 4 rows per simdgroup -> 8 rows/tg at 64 threads.
    MoeFormat { ty: 6,  gu: None, down: mk("moe_down_q50", 64, 8),
                gu_m: None, down_m: None },                                          // Q5_0
    MoeFormat { ty: 8,  gu: mk("moe_gu_q80", 256, 8), down: mk("moe_down_q80", 64, 8),
                gu_m: mk("moe_gu_q80_m", 256, 8), down_m: mk("moe_down_q80_m", 64, 2) }, // Q8_0
    MoeFormat { ty: 12, gu: mk("moe_gu_q4k", 256, 8), down: None,
                gu_m: None, down_m: None },                                          // Q4_K
    MoeFormat { ty: 13, gu: None, down: mk("moe_down_q5k", 64, 8),
                gu_m: None, down_m: None },                                          // Q5_K
    MoeFormat { ty: 14, gu: None, down: mk("moe_down_q6k", 64, 8),
                gu_m: None, down_m: None },                                          // Q6_K
    // down: the IQ kernels decode 1 row per simdgroup -> 2 rows/tg at 64 threads.
    MoeFormat { ty: 16, gu: mk("moe_gu_iq2xxs", 256, 8), down: mk("moe_down_iq2xxs", 64, 2),
                gu_m: None, down_m: None },                                          // IQ2_XXS
    MoeFormat { ty: 18, gu: None, down: mk("moe_down_iq3xxs", 64, 2),
                gu_m: None, down_m: None },                                          // IQ3_XXS
    MoeFormat { ty: 20, gu: None, down: mk("moe_down_iq4nl", 64, 2),
                gu_m: None, down_m: mk("moe_down_iq4nl_m", 64, 2) },                 // IQ4_NL
    MoeFormat { ty: 21, gu: mk("moe_gu_iq3s", 64, 8), down: None,
                gu_m: mk("moe_gu_iq3s_m", 256, 8), down_m: None },                   // IQ3_S
    MoeFormat { ty: 22, gu: mk("moe_gu_iq2s", 256, 8), down: None,
                gu_m: None, down_m: None },                                          // IQ2_S
    MoeFormat { ty: 23, gu: mk("moe_gu_iq4xs", 256, 8), down: mk("moe_down_iq4xs", 64, 2),
                gu_m: mk("moe_gu_iq4xs_m", 256, 8), down_m: None },                  // IQ4_XS
];

/// The expert kernel for `ty` in `role`, or `None` when the format has none.
pub fn moe_kernel(ty: u32, role: MoeRole) -> Option<MoeKernel> {
    MOE_FORMATS.iter().find(|f| f.ty == ty).and_then(|f| match role {
        MoeRole::GateUp => f.gu,
        MoeRole::Down => f.down,
    })
}

/// The batched (M>1) twin. `None` means this format has no batched kernel, and
/// the caller must run per-token rather than substitute anything else.
pub fn moe_kernel_m(ty: u32, role: MoeRole) -> Option<MoeKernel> {
    MOE_FORMATS.iter().find(|f| f.ty == ty).and_then(|f| match role {
        MoeRole::GateUp => f.gu_m,
        MoeRole::Down => f.down_m,
    })
}

/// Role a `*_exps.` tensor name plays. The loader uses this to validate a model
/// before upload, so the graph can assume a kernel exists.
pub fn moe_role_of(name: &str) -> MoeRole {
    if name.contains("ffn_down_exps") { MoeRole::Down } else { MoeRole::GateUp }
}

/// The per-dialect mapping for everything the decoders need that differs between
/// backends. It is meant to stay tiny: growth here means the decoders are leaking
/// platform assumptions.
pub fn prologue(d: Dialect) -> &'static str {
    match d {
        Dialect::Metal => r#"
// ---- dialect: Metal ----
#define QG        device const      // pointer into global (device) memory
#define QCONSTANT constant          // read-only table visible to all threads
#define QF16(P)   float(*(device const half*)(P))
#define QHBITS(B) float(as_type<half>(ushort(B)))
#define QFBITS(U) as_type<float>(uint(U))
#define QSUM(V)   simd_sum(V)
// Threadgroup/shared memory. TWO macros, not one: the storage class on the
// DECLARATION and the address space on a POINTER are the same word in MSL and
// different words in CUDA (`__shared__` is illegal on a pointer type there).
#define QSHARED   threadgroup
#define QSPTR     threadgroup
#define QBAR()    threadgroup_barrier(mem_flags::mem_threadgroup)
#define QU64      ulong
#define QUNROLL   _Pragma("unroll")
"#,
        Dialect::Cuda => r#"
// ---- dialect: CUDA ----
#include <cuda_fp16.h>
typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
#define QG        const
#define QCONSTANT __device__ const
#define QF16(P)   __half2float(*(const __half*)(P))
#define QHBITS(B) __half2float(__ushort_as_half((unsigned short)(B)))
#define QFBITS(U) __uint_as_float((unsigned int)(U))
#define QSUM(V)   warp_sum(V)
#define QSHARED   __shared__
#define QSPTR                       /* CUDA shared pointers are plain pointers */
#define QBAR()    __syncthreads()
#define QU64      unsigned long long
#define QUNROLL   _Pragma("unroll")
"#,
    }
}

/// The block decoders themselves — identical text for every backend.
pub const DECODERS: &str = r#"
// ============================ block decoders =================================
// Contract: FMT_SUB(WR, BLK, IL, BODY) decodes sub-block IL of block BLK and
// runs BODY once per weight with `_v` = value and `_idx` = column in the row.
// FMT_ROW(WR, BLK, BODY) is the whole block, defined as a loop over sub-blocks.

// ---- legacy 32-weight formats: one sub-block, IL unused -----------------------
// In all four, weights [0..16) are the LOW nibbles of qs[0..16] and [16..32) the
// HIGH nibbles — NOT interleaved 2j/2j+1 per byte.

// Q4_0: { half d; u8 qs[16]; } = 18 B / 32; w = d*(nib-8)
#define Q40_SUB(WR, BLK, IL, BODY) { (void)(IL); \
    QG uchar* _b = (WR) + (QU64)(BLK)*18u; \
    float _d = QF16(_b); QG uchar* _q = _b+2u; uint _base = (BLK)*32u; \
    for (uint _j=0u;_j<16u;_j++){ \
      { float _v=_d*(float(uint(_q[_j])&0x0Fu)-8.0f); uint _sub=_j; uint _idx=_base+_sub;     (void)_idx; (void)_sub; BODY } \
      { float _v=_d*(float(uint(_q[_j])>>4)-8.0f);    uint _sub=16u+_j; uint _idx=_base+_sub; (void)_idx; (void)_sub; BODY } } }

// Q4_1: { half d; half m; u8 qs[16]; } = 20 B / 32; w = nib*d + m
#define Q41_SUB(WR, BLK, IL, BODY) { (void)(IL); \
    QG uchar* _b = (WR) + (QU64)(BLK)*20u; \
    float _d = QF16(_b); float _m = QF16(_b+2u); \
    QG uchar* _q = _b+4u; uint _base = (BLK)*32u; \
    for (uint _j=0u;_j<16u;_j++){ \
      { float _v=_d*float(uint(_q[_j])&0x0Fu)+_m; uint _sub=_j; uint _idx=_base+_sub;     (void)_idx; (void)_sub; BODY } \
      { float _v=_d*float(uint(_q[_j])>>4)+_m;    uint _sub=16u+_j; uint _idx=_base+_sub; (void)_idx; (void)_sub; BODY } } }

// Q5_0: { half d; u8 qh[4]; u8 qs[16]; } = 22 B / 32; w = d*(q-16).
// The 5th bit of weight j is bit j of the qh word (bit j+16 for the high
// nibbles) — it does NOT sit next to the nibble.
#define Q50_SUB(WR, BLK, IL, BODY) { (void)(IL); \
    QG uchar* _b = (WR) + (QU64)(BLK)*22u; \
    float _d = QF16(_b); \
    uint _qh = uint(_b[2])|(uint(_b[3])<<8)|(uint(_b[4])<<16)|(uint(_b[5])<<24); \
    QG uchar* _q = _b+6u; uint _base = (BLK)*32u; \
    for (uint _j=0u;_j<16u;_j++){ \
      { uint _c=(uint(_q[_j])&0x0Fu)|(((_qh>>_j)&1u)<<4); \
        float _v=_d*(float(_c)-16.0f); uint _sub=_j; uint _idx=_base+_sub;     (void)_idx; (void)_sub; BODY } \
      { uint _c=(uint(_q[_j])>>4)|(((_qh>>(_j+16u))&1u)<<4); \
        float _v=_d*(float(_c)-16.0f); uint _sub=16u+_j; uint _idx=_base+_sub; (void)_idx; (void)_sub; BODY } } }

// Q5_1: { half d; half m; u8 qh[4]; u8 qs[16]; } = 24 B / 32; w = q*d + m
#define Q51_SUB(WR, BLK, IL, BODY) { (void)(IL); \
    QG uchar* _b = (WR) + (QU64)(BLK)*24u; \
    float _d = QF16(_b); float _m = QF16(_b+2u); \
    uint _qh = uint(_b[4])|(uint(_b[5])<<8)|(uint(_b[6])<<16)|(uint(_b[7])<<24); \
    QG uchar* _q = _b+8u; uint _base = (BLK)*32u; \
    for (uint _j=0u;_j<16u;_j++){ \
      { uint _c=(uint(_q[_j])&0x0Fu)|(((_qh>>_j)&1u)<<4); \
        float _v=_d*float(_c)+_m; uint _sub=_j; uint _idx=_base+_sub;     (void)_idx; (void)_sub; BODY } \
      { uint _c=(uint(_q[_j])>>4)|(((_qh>>(_j+16u))&1u)<<4); \
        float _v=_d*float(_c)+_m; uint _sub=16u+_j; uint _idx=_base+_sub; (void)_idx; (void)_sub; BODY } } }

// Q8_0: { half d; i8 qs[32]; } = 34 B / 32; w = d*q
#define Q80_SUB(WR, BLK, IL, BODY) { (void)(IL); \
    QG uchar* _b = (WR) + (QU64)(BLK)*34u; \
    float _d = QF16(_b); QG char* _q = (QG char*)(_b+2u); uint _base = (BLK)*32u; \
    for (uint _l=0u;_l<32u;_l++){ float _v=_d*float(_q[_l]); \
      uint _sub=_l; uint _idx=_base+_sub; (void)_idx; (void)_sub; BODY } }

// IQ4_NL: { half d; u8 qs[16]; } = 18 B / 32. "NL" = non-linear: the 4 bits index
// a 16-entry codebook of unevenly spaced levels, not a uniform ramp.
#define IQ4NL_SUB(WR, BLK, IL, BODY) { (void)(IL); \
    QG uchar* _b = (WR) + (QU64)(BLK)*18u; \
    float _d = QF16(_b); QG uchar* _qs = _b+2u; uint _base = (BLK)*32u; \
    for (uint _j=0u;_j<16u;_j++){ \
      { float _v=_d*float(kvalues_iq4nl_p[uint(_qs[_j])&0x0Fu]); \
        uint _sub=_j; uint _idx=_base+_sub;     (void)_idx; (void)_sub; BODY } \
      { float _v=_d*float(kvalues_iq4nl_p[uint(_qs[_j])>>4]); \
        uint _sub=16u+_j; uint _idx=_base+_sub; (void)_idx; (void)_sub; BODY } } }

// MXFP4: { u8 scale; u8 qs[16]; } = 17 B / 32. The 4-bit code is OCP E2M1 (a
// 16-entry LUT), and the scale is an e8m0 (pure exponent) shared by all 32
// weights: 2^(e-127), formed by placing e into the float32 exponent field
// (e<<23) rather than through exp2, exactly as the reference kernel does.
//
// The nibbles are INTERLEAVED, unlike the GGUF legacy formats: weight 2j is the
// LOW nibble of qs[j] and weight 2j+1 the HIGH nibble (so consecutive weights
// share a byte). This matches the HF `*_blocks` tensor's K/2-byte rows, so the
// loader packs those bytes verbatim behind the scale byte.
#define MXFP4_SUB(WR, BLK, IL, BODY) { (void)(IL); \
    QG uchar* _b = (WR) + (QU64)(BLK)*17u; \
    float _d = QFBITS(uint(_b[0]) << 23); QG uchar* _qs = _b+1u; uint _base = (BLK)*32u; \
    for (uint _j=0u;_j<16u;_j++){ \
      { float _v=_d*kvalues_mxfp4[uint(_qs[_j])&0x0Fu]; \
        uint _sub=2u*_j;    uint _idx=_base+_sub; (void)_idx; (void)_sub; BODY } \
      { float _v=_d*kvalues_mxfp4[uint(_qs[_j])>>4]; \
        uint _sub=2u*_j+1u; uint _idx=_base+_sub; (void)_idx; (void)_sub; BODY } } }

// The OCP E2M1 codebook (sign,exp2,man1). Signed levels, symmetric about 0.
// `QCONSTANT` so a runtime nibble index is a constant-memory load, not a spill.
QCONSTANT float kvalues_mxfp4[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,-0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};

// ---- K-quants: 8 sub-blocks of 32 --------------------------------------------

// Q2_K: { u8 scales[16]; u8 qs[64]; half d; half dmin; } = 84 B / 256.
// Unlike every other K-quant, d/dmin sit at the END. Sub-block IL selects
// half n=IL>>2 and shift j=IL&3; each covers 32 weights as two runs of 16.
#define Q2K_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*84u; \
    QG uchar* _sc = _b; QG uchar* _qs = _b+16u; \
    float _d = QF16(_b+80u); float _dm = QF16(_b+82u); \
    uint _n = (IL)>>2, _j = (IL)&3u; \
    QG uchar* _q = _qs + _n*32u; uint _sh = 2u*_j; uint _is = _n*8u + _j*2u; \
    uint _s1 = uint(_sc[_is]), _s2 = uint(_sc[_is+1u]); \
    float _dl1=_d*float(_s1&0x0Fu), _ml1=_dm*float(_s1>>4); \
    float _dl2=_d*float(_s2&0x0Fu), _ml2=_dm*float(_s2>>4); \
    uint _o = (BLK)*256u + _n*128u + _j*32u; \
    for (uint _l=0u;_l<16u;_l++){ \
      { float _v=_dl1*float((uint(_q[_l])>>_sh)&3u)-_ml1; \
        uint _sub=_l; uint _idx=_o+_sub;      (void)_idx; (void)_sub; BODY } \
      { float _v=_dl2*float((uint(_q[_l+16u])>>_sh)&3u)-_ml2; \
        uint _sub=16u+_l; uint _idx=_o+_sub;  (void)_idx; (void)_sub; BODY } } }

// Q3_K: { u8 hmask[32]; u8 qs[64]; u8 scales[12]; half d; } = 110 B / 256.
// 3-bit: 2 low bits in qs, the 3rd in hmask as a per-weight INVERTED bit (set =
// 0, clear = subtract 4). The hmask bit for sub-block IL is bit IL — which is
// why the whole-block form advanced `m` once per sub-block.
#define Q3K_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*110u; \
    QG uchar* _hm = _b; QG uchar* _qs = _b+32u; QG uchar* _raw = _b+96u; \
    float _dall = QF16(_b+108u); \
    uint _a0=uint(_raw[0])|(uint(_raw[1])<<8)|(uint(_raw[2])<<16)|(uint(_raw[3])<<24); \
    uint _a1=uint(_raw[4])|(uint(_raw[5])<<8)|(uint(_raw[6])<<16)|(uint(_raw[7])<<24); \
    uint _a2=uint(_raw[8])|(uint(_raw[9])<<8)|(uint(_raw[10])<<16)|(uint(_raw[11])<<24); \
    uint _km1=0x03030303u, _km2=0x0f0f0f0fu, _tmp=_a2; \
    uint _w2=((_a0>>4)&_km2)|(((_tmp>>4)&_km1)<<4); \
    uint _w3=((_a1>>4)&_km2)|(((_tmp>>6)&_km1)<<4); \
    uint _w0=(_a0&_km2)|((_tmp&_km1)<<4); \
    uint _w1=(_a1&_km2)|(((_tmp>>2)&_km1)<<4); \
    uint _n=(IL)>>2, _j=(IL)&3u; \
    QG uchar* _q=_qs+_n*32u; uint _sh=2u*_j; uint _is=_n*8u+_j*2u; uint _m=1u<<(IL); \
    int _sc1, _sc2; \
    Q3K_SC(_w0,_w1,_w2,_w3,_is,_sc1) Q3K_SC(_w0,_w1,_w2,_w3,_is+1u,_sc2) \
    float _dl1=_dall*float(_sc1-32), _dl2=_dall*float(_sc2-32); \
    uint _o=(BLK)*256u+_n*128u+_j*32u; \
    for (uint _l=0u;_l<16u;_l++){ \
      { float _h=((uint(_hm[_l])&_m)!=0u)?0.0f:4.0f; \
        float _v=_dl1*(float((uint(_q[_l])>>_sh)&3u)-_h); \
        uint _sub=_l; uint _idx=_o+_sub;      (void)_idx; (void)_sub; BODY } \
      { float _h=((uint(_hm[_l+16u])&_m)!=0u)?0.0f:4.0f; \
        float _v=_dl2*(float((uint(_q[_l+16u])>>_sh)&3u)-_h); \
        uint _sub=16u+_l; uint _idx=_o+_sub;  (void)_idx; (void)_sub; BODY } } }

// Pull scale I (0..15) from Q3_K's four unpacked scale words. Nested ternaries
// rather than a local array: a runtime-indexed thread-local array spills to
// device-backed thread memory on Apple GPUs (the trap that made an earlier
// matvec run 47 ms against 19). The field is 6 bits so it is never negative as
// an int8 and the sign extension is a no-op here.
#define Q3K_SC(W0,W1,W2,W3,I,OUT) { uint _w=(I)>>2; \
    uint _ww=(_w==0u)?(W0):((_w==1u)?(W1):((_w==2u)?(W2):(W3))); \
    OUT = int((_ww >> (8u*((I)&3u))) & 0xFFu); }

// Q4_K: { half d; half dmin; u8 scales[12]; u8 qs[128]; } = 144 B / 256.
// Sub-block IL is one of eight 32-weight groups; the 6-bit scale/min pair is
// packed two ways depending on IL<4.
#define Q4K_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*144u; \
    float _d = QF16(_b); float _dm = QF16(_b+2u); \
    QG uchar* _sc = _b+4u; QG uchar* _qs = _b+16u; \
    uint _j=(IL); uint _s,_mn; \
    if(_j<4u){_s=_sc[_j]&63u;_mn=_sc[_j+4u]&63u;} \
    else{_s=(_sc[_j+4u]&0x0Fu)|((_sc[_j-4u]>>6)<<4);_mn=(_sc[_j+4u]>>4)|((_sc[_j]>>6)<<4);} \
    float _d1=_d*float(_s); float _m1=_dm*float(_mn); \
    QG uchar* _qq=_qs+(_j>>1)*32u; uint _hi=_j&1u; \
    uint _o=(BLK)*256u+_j*32u; \
    for(uint _l=0u;_l<32u;_l++){ uint _nb=_hi?(uint(_qq[_l])>>4):(uint(_qq[_l])&0x0Fu); \
      float _v=_d1*float(_nb)-_m1; uint _sub=_l; uint _idx=_o+_sub; (void)_idx; (void)_sub; BODY } }

// Q5_K: { half d; half dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; } = 176 B / 256.
// Four 64-weight chunks, each two sub-blocks; the high-bit mask shifts left by 2
// per chunk, so sub-block IL uses bit IL of each qh byte.
#define Q5K_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*176u; \
    float _d = QF16(_b); float _dm = QF16(_b+2u); \
    QG uchar* _sc = _b+4u; QG uchar* _qh = _b+16u; QG uchar* _qs = _b+48u; \
    uint _j=(IL); uint _c=_j>>1; uint _half=_j&1u; uint _s,_mn; \
    if(_j<4u){_s=_sc[_j]&63u;_mn=_sc[_j+4u]&63u;} \
    else{_s=(_sc[_j+4u]&0x0Fu)|((_sc[_j-4u]>>6)<<4);_mn=(_sc[_j+4u]>>4)|((_sc[_j]>>6)<<4);} \
    float _dl=_d*float(_s), _ml=_dm*float(_mn); \
    QG uchar* _ql=_qs+_c*32u; uint _bit=1u<<_j; \
    uint _o=(BLK)*256u+_c*64u+_half*32u; \
    for(uint _l=0u;_l<32u;_l++){ \
      float _h=((uint(_qh[_l])&_bit)!=0u)?16.0f:0.0f; \
      uint _nb=_half?(uint(_ql[_l])>>4):(uint(_ql[_l])&0x0Fu); \
      float _v=_dl*(float(_nb)+_h)-_ml; uint _sub=_l; uint _idx=_o+_sub; (void)_idx; (void)_sub; BODY } }

// Q6_K: { u8 ql[128]; u8 qh[64]; i8 scales[16]; half d; } = 210 B / 256.
// Each half of the block interleaves FOUR 32-weight streams; sub-block IL picks
// half h=IL>>2 and stream s=IL&3. Stream s reads ql at +32*(s&1), takes the high
// nibble when s>=2, shifts qh by 2s and uses scale index 2s.
#define Q6K_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*210u; \
    QG uchar* _ql = _b; QG uchar* _qh = _b + 128u; \
    QG char* _sc = (QG char*)(_b + 192u); \
    float _d = QF16(_b + 208u); \
    uint _h=(IL)>>2, _s=(IL)&3u; \
    uint _qlb=_h*64u + (_s&1u)*32u, _qhb=_h*32u, _scb=_h*8u + 2u*_s; \
    uint _o=(BLK)*256u + _h*128u + _s*32u; \
    for (uint _l=0u;_l<32u;_l++){ uint _is=_l>>4; \
      uint _lo=uint(_ql[_qlb+_l]); \
      uint _nb=(_s>=2u)?(_lo>>4):(_lo&0x0Fu); \
      uint _hb=(uint(_qh[_qhb+_l])>>(2u*_s))&3u; \
      float _v=_d*float(_sc[_scb+_is])*float(int(_nb|(_hb<<4))-32); \
      uint _sub=_l; uint _idx=_o+_sub; (void)_idx; (void)_sub; BODY } }

// IQ4_XS: { half d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; } = 136 B / 256.
// Same codebook as IQ4_NL; eight sub-blocks share one `d`, each with a 6-bit
// scale split across scales_l (nibble) and scales_h (2 bits), biased by 32.
#define IQ4XS_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*136u; \
    float _d = QF16(_b); \
    uint _sh = uint(_b[2]) | (uint(_b[3])<<8); \
    QG uchar* _sl = _b+4u; QG uchar* _qs = _b+8u; \
    uint _ib=(IL); \
    uint _ls = ((uint(_sl[_ib>>1]) >> (4u*(_ib&1u))) & 0x0Fu) | (((_sh >> (2u*_ib)) & 3u) << 4); \
    float _dl = _d*float(int(_ls)-32); \
    QG uchar* _q = _qs + _ib*16u; \
    uint _o = (BLK)*256u + _ib*32u; \
    for (uint _j=0u;_j<16u;_j++){ \
      { float _v=_dl*float(kvalues_iq4nl_p[uint(_q[_j])&0x0Fu]); \
        uint _sub=_j; uint _idx=_o+_sub;      (void)_idx; (void)_sub; BODY } \
      { float _v=_dl*float(kvalues_iq4nl_p[uint(_q[_j])>>4]); \
        uint _sub=16u+_j; uint _idx=_o+_sub;  (void)_idx; (void)_sub; BODY } } }

// The IQ4 codebook. `QCONSTANT` because a runtime index into constant memory is
// a load, not a spill — unlike a thread-local array.
QCONSTANT int kvalues_iq4nl_p[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};


// ---- IQ family: 8 sub-blocks of 32, codebook-driven ---------------------------
// These formats store no weights. A group of 8 shares one CODEBOOK entry giving
// 8 magnitudes, plus a sign word and a scale — that indirection is where 2 bits
// per weight comes from. Grids hold MAGNITUDES packed as bytes inside a u64
// (IQ2, 8 per entry) or a u32 (IQ3, 4 per entry); a SET sign bit means negative.
#define IQ_SGN(S, J) ((((S) & kmask_iq2xs[(J)]) != 0) ? -1.0f : 1.0f)
#define IQ_B(G, J)   float(((G) >> (8u*(J))) & 0xFFu)
// IQ1's grid bytes are SIGNED, unlike every other grid. Reading them unsigned
// silently folds the negative half of the range onto 128..255.
#define IQ_SB(G, J)  float(char(uchar(((G) >> (8u*(J))) & 0xFFu)))

// IQ2_XXS: { half d; u16 qs[32]; } = 66 B / 256. Each 8-byte stride holds four
// grid-index bytes then a word packing the sub-block scale (top nibble) and four
// 7-bit sign-table indices.
#define IQ2XXS_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*66u; \
    float _d = QF16(_b); QG uchar* _qs = _b+2u; \
    uint _ib=(IL), _o8=8u*_ib; \
    uint _a0 = uint(_qs[_o8])|(uint(_qs[_o8+1u])<<8)|(uint(_qs[_o8+2u])<<16)|(uint(_qs[_o8+3u])<<24); \
    uint _a1 = uint(_qs[_o8+4u])|(uint(_qs[_o8+5u])<<8)|(uint(_qs[_o8+6u])<<16)|(uint(_qs[_o8+7u])<<24); \
    float _db = _d*(0.5f+float(_a1>>28))*0.25f; \
    uint _o = (BLK)*256u + _ib*32u; \
    for (uint _l=0u;_l<4u;_l++){ \
      QU64 _g = iq2xxs_grid[(_a0 >> (8u*_l)) & 0xFFu]; \
      uchar _s = ksigns_iq2xs[(_a1 >> (7u*_l)) & 127u]; \
      for (uint _j=0u;_j<8u;_j++){ \
        float _v = _db*IQ_B(_g,_j)*IQ_SGN(_s,_j); \
        uint _sub = _l*8u+_j; uint _idx = _o+_sub; (void)_idx; (void)_sub; BODY } } }

// IQ2_XS: { half d; u16 qs[32]; u8 scales[8]; } = 74 B / 256. Grid index and sign
// index share one u16 (9 bits + 7), which frees the scale into its own nibble.
// Lanes 0-1 of each group take the low nibble, 2-3 the high.
#define IQ2XS_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*74u; \
    float _d = QF16(_b); QG uchar* _qs = _b+2u; QG uchar* _sc = _b+66u; \
    uint _ib=(IL); \
    float _db0 = _d*(0.5f+float(uint(_sc[_ib])&0x0Fu))*0.25f; \
    float _db1 = _d*(0.5f+float(uint(_sc[_ib])>>4))*0.25f; \
    uint _o = (BLK)*256u + _ib*32u; \
    for (uint _l=0u;_l<4u;_l++){ uint _i=4u*_ib+_l; \
      uint _q = uint(_qs[2u*_i]) | (uint(_qs[2u*_i+1u])<<8); \
      QU64 _g = iq2xs_grid[_q & 511u]; \
      uchar _s = ksigns_iq2xs[_q >> 9]; \
      float _dl = (_l < 2u) ? _db0 : _db1; \
      for (uint _j=0u;_j<8u;_j++){ \
        float _v = _dl*IQ_B(_g,_j)*IQ_SGN(_s,_j); \
        uint _sub = _l*8u+_j; uint _idx = _o+_sub; (void)_idx; (void)_sub; BODY } } }

// IQ2_S: { half d; u8 qs[64]; u8 qh[8]; u8 scales[8]; } = 82 B / 256. qs splits in
// half: 32 index bytes (2 more index bits from qh), then 32 RAW sign bytes rather
// than indices into ksigns — the extra bits are what buys accuracy over IQ2_XS.
#define IQ2S_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*82u; \
    float _d = QF16(_b); QG uchar* _qs = _b+2u; \
    QG uchar* _qh = _b+66u; QG uchar* _sc = _b+74u; \
    uint _ib=(IL); \
    float _db0 = _d*(0.5f+float(uint(_sc[_ib])&0x0Fu))*0.25f; \
    float _db1 = _d*(0.5f+float(uint(_sc[_ib])>>4))*0.25f; \
    uint _o = (BLK)*256u + _ib*32u; \
    for (uint _l=0u;_l<4u;_l++){ \
      uint _gi = uint(_qs[4u*_ib+_l]) | ((uint(_qh[_ib]) << (8u-2u*_l)) & 0x300u); \
      QU64 _g = iq2s_grid[_gi]; \
      uchar _s = _qs[32u+4u*_ib+_l]; \
      float _dl = (_l < 2u) ? _db0 : _db1; \
      for (uint _j=0u;_j<8u;_j++){ \
        float _v = _dl*IQ_B(_g,_j)*IQ_SGN(_s,_j); \
        uint _sub = _l*8u+_j; uint _idx = _o+_sub; (void)_idx; (void)_sub; BODY } } }

// IQ3_XXS: { half d; u8 qs[96]; } = 98 B / 256. Grid entries hold only 4
// magnitudes, so a group of 8 needs TWO indices; lanes 0-3 come from the first
// and 4-7 from the second, each reading the sign word at its own lane.
#define IQ3XXS_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*98u; \
    float _d = QF16(_b); QG uchar* _qs = _b+2u; QG uchar* _ss = _b+2u+64u; \
    uint _ib=(IL), _o4=4u*_ib; \
    uint _a = uint(_ss[_o4])|(uint(_ss[_o4+1u])<<8)|(uint(_ss[_o4+2u])<<16)|(uint(_ss[_o4+3u])<<24); \
    float _db = _d*(0.5f+float(_a>>28))*0.5f; \
    QG uchar* _q = _qs + 8u*_ib; \
    uint _o = (BLK)*256u + _ib*32u; \
    for (uint _l=0u;_l<4u;_l++){ \
      uchar _s = ksigns_iq2xs[(_a >> (7u*_l)) & 127u]; \
      uint _g1 = iq3xxs_grid[_q[2u*_l]]; uint _g2 = iq3xxs_grid[_q[2u*_l+1u]]; \
      for (uint _j=0u;_j<4u;_j++){ \
        float _v = _db*IQ_B(_g1,_j)*IQ_SGN(_s,_j); \
        uint _sub = _l*8u+_j; uint _idx = _o+_sub; (void)_idx; (void)_sub; BODY } \
      for (uint _j=0u;_j<4u;_j++){ \
        float _v = _db*IQ_B(_g2,_j)*IQ_SGN(_s,_j+4u); \
        uint _sub = _l*8u+4u+_j; uint _idx = _o+_sub; (void)_idx; (void)_sub; BODY } } }

// IQ3_S: { half d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]; } = 110 B / 256.
// One scale byte covers two sub-blocks, and the scale form is (1 + 2*s) — DIFFERENT
// from every other IQ type here, which use (0.5 + s).
#define IQ3S_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*110u; \
    float _d = QF16(_b); QG uchar* _qs = _b+2u; QG uchar* _qh = _b+66u; \
    QG uchar* _sg = _b+74u; QG uchar* _sc = _b+106u; \
    uint _p=(IL)>>1, _h2=(IL)&1u; \
    float _db = (_h2==0u) ? _d*(1.0f+2.0f*float(uint(_sc[_p])&0x0Fu)) \
                          : _d*(1.0f+2.0f*float(uint(_sc[_p])>>4)); \
    uint _h = uint(_qh[(IL)]); \
    uint _qo = 16u*_p + 8u*_h2, _so = 8u*_p + 4u*_h2; \
    uint _o = (BLK)*256u + (IL)*32u; \
    for (uint _l=0u;_l<4u;_l++){ \
      uint _g1 = iq3s_grid[uint(_qs[_qo+2u*_l])    | ((_h << (8u-2u*_l)) & 256u)]; \
      uint _g2 = iq3s_grid[uint(_qs[_qo+2u*_l+1u]) | ((_h << (7u-2u*_l)) & 256u)]; \
      uchar _s = _sg[_so+_l]; \
      for (uint _j=0u;_j<4u;_j++){ \
        float _v = _db*IQ_B(_g1,_j)*IQ_SGN(_s,_j); \
        uint _sub = _l*8u+_j; uint _idx = _o+_sub; (void)_idx; (void)_sub; BODY } \
      for (uint _j=0u;_j<4u;_j++){ \
        float _v = _db*IQ_B(_g2,_j)*IQ_SGN(_s,_j+4u); \
        uint _sub = _l*8u+4u+_j; uint _idx = _o+_sub; (void)_idx; (void)_sub; BODY } } }

// IQ1_S: { half d; u8 qs[32]; u16 qh[8]; } = 50 B / 256. Each sub-block has a
// 3-bit scale and a sign bit choosing DELTA, a constant offset added to every
// magnitude — at one bit per weight there is no room for a per-weight offset.
#define IQ1S_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*50u; \
    float _d = QF16(_b); QG uchar* _qs = _b+2u; QG uchar* _qhb = _b+34u; \
    uint _ib=(IL); \
    uint _qh = uint(_qhb[2u*_ib]) | (uint(_qhb[2u*_ib+1u])<<8); \
    float _dl = _d*float(2u*((_qh>>12)&7u)+1u); \
    float _dt = ((_qh & 0x8000u)!=0u) ? -0.125f : 0.125f; \
    uint _o = (BLK)*256u + _ib*32u; \
    for (uint _l=0u;_l<4u;_l++){ \
      QU64 _g = iq1s_grid[uint(_qs[4u*_ib+_l]) | (((_qh >> (3u*_l)) & 7u) << 8)]; \
      for (uint _j=0u;_j<8u;_j++){ \
        float _v = _dl*(IQ_SB(_g,_j)+_dt); \
        uint _sub = _l*8u+_j; uint _idx = _o+_sub; (void)_idx; (void)_sub; BODY } } }

// IQ1_M: { u8 qs[32]; u8 qh[16]; u8 scales[8]; } = 56 B / 256. The ONLY block in
// the family with no `d` field: the f16 scale is reassembled from the top nibble
// of each of the four trailing scale words, low nibble first.
#define IQ1M_SUB(WR, BLK, IL, BODY) { \
    QG uchar* _b = (WR) + (QU64)(BLK)*56u; \
    QG uchar* _qs = _b; QG uchar* _qh = _b+32u; QG uchar* _scb = _b+48u; \
    uint _s0=uint(_scb[0])|(uint(_scb[1])<<8), _s1=uint(_scb[2])|(uint(_scb[3])<<8); \
    uint _s2=uint(_scb[4])|(uint(_scb[5])<<8), _s3=uint(_scb[6])|(uint(_scb[7])<<8); \
    uint _dbits = (_s0>>12) | ((_s1>>8)&0x00F0u) | ((_s2>>4)&0x0F00u) | (_s3&0xF000u); \
    float _d = QHBITS(_dbits); \
    uint _ib=(IL); \
    uint _w = (_ib>>1)==0u?_s0:((_ib>>1)==1u?_s1:((_ib>>1)==2u?_s2:_s3)); \
    uint _shf = 6u*(_ib&1u); \
    float _dl1 = _d*float(2u*((_w>>_shf)&7u)+1u); \
    float _dl2 = _d*float(2u*((_w>>(_shf+3u))&7u)+1u); \
    uint _h0 = uint(_qh[2u*_ib]), _h1 = uint(_qh[2u*_ib+1u]); \
    uint _o = (BLK)*256u + _ib*32u; \
    for (uint _l=0u;_l<4u;_l++){ \
      uint _gi, _hh; \
      if (_l==0u)      { _gi = uint(_qs[4u*_ib+0u]) | ((_h0<<8)&0x700u); _hh=_h0; } \
      else if (_l==1u) { _gi = uint(_qs[4u*_ib+1u]) | ((_h0<<4)&0x700u); _hh=_h0; } \
      else if (_l==2u) { _gi = uint(_qs[4u*_ib+2u]) | ((_h1<<8)&0x700u); _hh=_h1; } \
      else             { _gi = uint(_qs[4u*_ib+3u]) | ((_h1<<4)&0x700u); _hh=_h1; } \
      uint _mask = ((_l&1u)==0u) ? 0x08u : 0x80u; \
      float _dt = ((_hh & _mask)!=0u) ? -0.125f : 0.125f; \
      float _dl = (_l<2u) ? _dl1 : _dl2; \
      QU64 _g = iq1s_grid[_gi]; \
      for (uint _j=0u;_j<8u;_j++){ \
        float _v = _dl*(IQ_SB(_g,_j)+_dt); \
        uint _sub = _l*8u+_j; uint _idx = _o+_sub; (void)_idx; (void)_sub; BODY } } }

"#;

/// `FMT_ROW` for each format: the whole block as a loop over its sub-blocks. The
/// load-time requantizers use these, so they also gate the sub-block
/// decomposition — a wrong split fails `iq_raw_gate` for that format.
pub fn row_macros() -> String {
    let mut s = String::from("\n// ---- whole-block forms, generated from the sub-block decoders ----\n");
    for f in FORMATS {
        s.push_str(&format!(
            "#define {w}_ROW(WR, SB, BODY) {{ for (uint _il_=0u;_il_<{n}u;_il_++) {{ {w}_SUB(WR, SB, _il_, BODY) }} }}\n",
            w = f.walker, n = f.subs));
    }
    s
}

/// Matvec bodies, lane-cooperative: work items are (block, sub-block) pairs and
/// lane L takes items L, L+32, ..., so eight consecutive lanes decode eight
/// sub-blocks of the same block. Each lane decodes 32 weights and neighbouring
/// lanes read neighbouring bytes. A one-block-per-lane form instead decodes all
/// 256 weights on one lane, with consecutive lanes a whole block apart, and
/// measured ~48 GB/s against the requantized path's ~372.
///
/// Signatures live at the instantiation site, not here: the Metal kernel registry
/// finds entry points by scanning source for `kernel void <name>(`, so a name
/// produced by macro token-pasting would be invisible to it.
pub const GEMV_BODIES: &str = r#"
// The factoring that Q3_K, IQ1_S and these two share, stated once:
//
//   every one of these formats decodes to  w = A*q + B  with A and B constant
//   over a group, so   SUM_j y_j*w_j  ==  A * SUM_j y_j*q_j  +  B * SUM_j y_j
//
// The generic body cannot use it — it consumes the walker's finished per-weight
// `_v`, which has already folded A and B together. That is the cost of one
// decoder serving the requantizer and the matvec both, and it is why these are
// per-format opt-ins rather than a rewrite of the walker protocol.
//
// The saving is one multiply and one add per weight out of two of each, plus
// whatever per-weight branching the format's offset needed.

// IQ4_XS: w = dl * kvalues[nibble], dl constant over a 32-weight sub-block.
// Accumulate SUM(y*kv) and apply dl once. Was 245 Gw/s, 48% of the best format.
#define NAT_GEMV_IQ4XS(WALK_SUB, BLKB, WPB, SUBS, SUBW, NR, STORE) \
    uint _row0 = (tgid*(ts/32u) + sgid) * (NR); \
    if (_row0 >= N) { return; } \
    uint nb = K/(WPB); \
    uint _tot = nb*(SUBS); \
    float _sumf[NR]; \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { _sumf[_r] = 0.0f; } \
    float _yl[SUBW]; \
    for (uint _t = lane; _t < _tot; _t += 32u) { \
        uint _blk = _t/(SUBS), _il = _t%(SUBS); \
        uint _cb = _blk*(WPB) + _il*(SUBW); \
        QUNROLL for (uint _i = 0u; _i < (SUBW); _i++) { _yl[_i] = x[_cb+_i]; } \
        QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
            if (_row0+_r >= N) { break; } \
            QG uchar* _b = w + (QU64)(_row0+_r)*(QU64)nb*(BLKB) + (QU64)_blk*136u; \
            float _d = QF16(_b); \
            uint _shx = uint(_b[2]) | (uint(_b[3])<<8); \
            QG uchar* _sl = _b+4u; QG uchar* _q = _b+8u+_il*16u; \
            uint _ls = ((uint(_sl[_il>>1]) >> (4u*(_il&1u))) & 0x0Fu) \
                     | (((_shx >> (2u*_il)) & 3u) << 4); \
            float _acc4 = 0.0f; \
            QUNROLL for (uint _j = 0u; _j < 16u; _j++) { \
                uint _byte = uint(_q[_j]); \
                _acc4 += _yl[_j]*float(kvalues_iq4nl_p[_byte & 0x0Fu]) \
                       + _yl[16u+_j]*float(kvalues_iq4nl_p[_byte >> 4]); } \
            _sumf[_r] += _d*float(int(_ls)-32) * _acc4; \
        } \
    } \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
        float acc = QSUM(_sumf[_r]); \
        if (lane == 0u && _row0+_r < N) { uint n = _row0+_r; STORE; } \
    }

// Q5_K: w = dl*(nibble + 16*highbit) - ml, with dl/ml constant over the
// sub-block. The -ml was a subtract per weight; it becomes -ml*SUM(y) once.
// Was 276 Gw/s, 54% of the best format.
#define NAT_GEMV_Q5K(WALK_SUB, BLKB, WPB, SUBS, SUBW, NR, STORE) \
    uint _row0 = (tgid*(ts/32u) + sgid) * (NR); \
    if (_row0 >= N) { return; } \
    uint nb = K/(WPB); \
    uint _tot = nb*(SUBS); \
    float _sumf[NR]; \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { _sumf[_r] = 0.0f; } \
    float _yl[SUBW]; \
    for (uint _t = lane; _t < _tot; _t += 32u) { \
        uint _blk = _t/(SUBS), _il = _t%(SUBS); \
        uint _cb = _blk*(WPB) + _il*(SUBW); \
        float _ysum = 0.0f; \
        QUNROLL for (uint _i = 0u; _i < (SUBW); _i++) { _yl[_i] = x[_cb+_i]; _ysum += _yl[_i]; } \
        uint _c5 = _il>>1, _hf5 = _il&1u, _bit5 = 1u<<_il; \
        QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
            if (_row0+_r >= N) { break; } \
            QG uchar* _b = w + (QU64)(_row0+_r)*(QU64)nb*(BLKB) + (QU64)_blk*176u; \
            float _d = QF16(_b), _dm = QF16(_b+2u); \
            QG uchar* _sc = _b+4u; QG uchar* _qh = _b+16u; QG uchar* _ql = _b+48u+_c5*32u; \
            uint _s5, _m5; \
            if (_il < 4u) { _s5=_sc[_il]&63u; _m5=_sc[_il+4u]&63u; } \
            else { _s5=(_sc[_il+4u]&0x0Fu)|((_sc[_il-4u]>>6)<<4); \
                   _m5=(_sc[_il+4u]>>4)|((_sc[_il]>>6)<<4); } \
            float _q5 = 0.0f; \
            QUNROLL for (uint _l = 0u; _l < 32u; _l++) { \
                float _h5 = ((uint(_qh[_l]) & _bit5) != 0u) ? 16.0f : 0.0f; \
                uint _nb5 = _hf5 ? (uint(_ql[_l])>>4) : (uint(_ql[_l])&0x0Fu); \
                _q5 += _yl[_l]*(float(_nb5)+_h5); } \
            _sumf[_r] += _d*float(_s5)*_q5 - _dm*float(_m5)*_ysum; \
        } \
    } \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
        float acc = QSUM(_sumf[_r]); \
        if (lane == 0u && _row0+_r < N) { uint n = _row0+_r; STORE; } \
    }

// Q4_K, specialized. The generic body reads one 32-weight sub-block per lane and
// re-derives its 6-bit scale and min from the packed bytes for every row, which
// held Q4_K at 289 Gw/s (K=2560, N=9216, M2 Max) against the tuned Q4L layout's
// ~530 on the same weights.
//
// Here eight lanes share a super-block and four super-blocks are in flight per
// simdgroup. Lane (ix, iq, ir) = (lane/8, lane%8/4, lane%4) owns eight weights of
// each of the sub-blocks 2iq, 2iq+1, 2iq+4 and 2iq+5, so its 32 activations and
// their four sums are loaded once per super-block and reused across all NR rows.
// The quants are read as 16-bit words: masking a word leaves nibble v at a fixed
// power-of-two multiple (v, 16v, 256v, 4096v), which the scale absorbs, and the
// four scales/mins the lane needs come out of three words with the kmask ops.
// The min term is -dmin*m*SUM(y), once per sub-block, as in Q5_K.
#define NAT_GEMV_Q4K(WALK_SUB, BLKB, WPB, SUBS, SUBW, NR, STORE) \
    uint _row0 = (tgid*(ts/32u) + sgid) * (NR); \
    if (_row0 >= N) { return; } \
    uint nb = K/256u; \
    uint _ix = lane/8u, _iq = (lane%8u)/4u, _ir = lane%4u; \
    float _sumf[NR]; \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { _sumf[_r] = 0.0f; } \
    float _yl[16], _yh[16]; \
    for (uint _ib = _ix; _ib < nb; _ib += 4u) { \
        QG float* _y4 = x + (QU64)_ib*256u + 64u*_iq + 8u*_ir; \
        float _s0 = 0.0f, _s1 = 0.0f, _s2 = 0.0f, _s3 = 0.0f; \
        QUNROLL for (uint _i = 0u; _i < 8u; _i++) { \
            _yl[_i] = _y4[_i];         _s0 += _yl[_i]; \
            _yl[_i+8u] = _y4[_i+32u];  _s1 += _yl[_i+8u]; \
            _yh[_i] = _y4[_i+128u];    _s2 += _yh[_i]; \
            _yh[_i+8u] = _y4[_i+160u]; _s3 += _yh[_i+8u]; } \
        QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
            if (_row0+_r >= N) { break; } \
            QG uchar* _b = w + ((QU64)(_row0+_r)*(QU64)nb + _ib)*144u; \
            QG ushort* _sw = (QG ushort*)(_b + 4u) + _iq; \
            uint _w0 = _sw[0], _w2 = _sw[2], _w4 = _sw[4]; \
            uint _sc01 = _w0 & 0x3F3Fu, _mn01 = _w2 & 0x3F3Fu; \
            uint _sc45 = (_w4 & 0x0F0Fu) | ((_w0 & 0xC0C0u) >> 2); \
            uint _mn45 = ((_w4 >> 4) & 0x0F0Fu) | ((_w2 & 0xC0C0u) >> 2); \
            QG ushort* _q1 = (QG ushort*)(_b + 16u) + 16u*_iq + 4u*_ir; \
            QG ushort* _q2 = _q1 + 32u; \
            float _a0 = 0.0f, _a1 = 0.0f, _a2 = 0.0f, _a3 = 0.0f; \
            float _c0 = 0.0f, _c1 = 0.0f, _c2 = 0.0f, _c3 = 0.0f; \
            QUNROLL for (uint _i = 0u; _i < 8u; _i += 2u) { \
                uint _u1 = _q1[_i/2u], _u2 = _q2[_i/2u]; \
                _a0 += _yl[_i]*float(_u1 & 0x000Fu);    _a1 += _yl[_i+1u]*float(_u1 & 0x0F00u); \
                _a2 += _yl[_i+8u]*float(_u1 & 0x00F0u); _a3 += _yl[_i+9u]*float(_u1 & 0xF000u); \
                _c0 += _yh[_i]*float(_u2 & 0x000Fu);    _c1 += _yh[_i+1u]*float(_u2 & 0x0F00u); \
                _c2 += _yh[_i+8u]*float(_u2 & 0x00F0u); _c3 += _yh[_i+9u]*float(_u2 & 0xF000u); } \
            _sumf[_r] += QF16(_b) * ((_a0 + _a1*(1.0f/256.0f))*float(_sc01 & 0xFFu) \
                                   + (_a2 + _a3*(1.0f/256.0f))*float(_sc01 >> 8)*(1.0f/16.0f) \
                                   + (_c0 + _c1*(1.0f/256.0f))*float(_sc45 & 0xFFu) \
                                   + (_c2 + _c3*(1.0f/256.0f))*float(_sc45 >> 8)*(1.0f/16.0f)) \
                       - QF16(_b + 2u) * (_s0*float(_mn01 & 0xFFu) + _s1*float(_mn01 >> 8) \
                                        + _s2*float(_mn45 & 0xFFu) + _s3*float(_mn45 >> 8)); \
        } \
    } \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
        float acc = QSUM(_sumf[_r]); \
        if (lane == 0u && _row0+_r < N) { uint n = _row0+_r; STORE; } \
    }

// Q6_K, specialized on the same plan: sixteen lanes share a super-block and two
// super-blocks are in flight. Lane (ix, ip, il) = (lane%2, lane/16, lane/2%8)
// owns weights l0..l0+3 (l0 = 4il) of each quarter of half ip, which share one
// qh byte per l and one int8 scale per quarter, so the 16 activations are loaded
// once per super-block and reused across all NR rows. The generic body measured
// 282 Gw/s here. Scales are signed bytes, read as (b ^ 0x80) - 128 so the
// arithmetic does not depend on the dialect's `char` signedness.
#define NAT_GEMV_Q6K(WALK_SUB, BLKB, WPB, SUBS, SUBW, NR, STORE) \
    uint _row0 = (tgid*(ts/32u) + sgid) * (NR); \
    if (_row0 >= N) { return; } \
    uint nb = K/256u; \
    uint _ix = lane%2u, _ip = lane/16u, _l0 = 4u*((lane/2u)%8u); \
    uint _is = 8u*_ip + _l0/16u, _yo = 128u*_ip + _l0; \
    uint _qol = 64u*_ip + _l0, _qoh = 128u + 32u*_ip + _l0; \
    float _sumf[NR]; \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { _sumf[_r] = 0.0f; } \
    float _yv[16]; \
    for (uint _ib = _ix; _ib < nb; _ib += 2u) { \
        QG float* _y = x + (QU64)_ib*256u + _yo; \
        QUNROLL for (uint _l = 0u; _l < 4u; _l++) { \
            _yv[_l] = _y[_l]; _yv[4u+_l] = _y[_l+32u]; _yv[8u+_l] = _y[_l+64u]; _yv[12u+_l] = _y[_l+96u]; } \
        QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
            if (_row0+_r >= N) { break; } \
            QG uchar* _b = w + ((QU64)(_row0+_r)*(QU64)nb + _ib)*210u; \
            QG uchar* _q1 = _b + _qol; QG uchar* _q2 = _q1 + 32u; QG uchar* _qh = _b + _qoh; \
            QG uchar* _sc = _b + 192u + _is; \
            float _t0 = 0.0f, _t1 = 0.0f, _t2 = 0.0f, _t3 = 0.0f; \
            QUNROLL for (uint _l = 0u; _l < 4u; _l++) { \
                uint _h = _qh[_l], _v1 = _q1[_l], _v2 = _q2[_l]; \
                _t0 += _yv[_l]    *float(int((_v1 & 0x0Fu) | ((_h & 0x03u) << 4)) - 32); \
                _t1 += _yv[4u+_l] *float(int((_v2 & 0x0Fu) | ((_h & 0x0Cu) << 2)) - 32); \
                _t2 += _yv[8u+_l] *float(int((_v1 >> 4)    |  (_h & 0x30u))       - 32); \
                _t3 += _yv[12u+_l]*float(int((_v2 >> 4)    | ((_h & 0xC0u) >> 2)) - 32); } \
            _sumf[_r] += QF16(_b + 208u) * ( \
                  _t0*float(int(uint(_sc[0]) ^ 0x80u) - 128) + _t1*float(int(uint(_sc[2]) ^ 0x80u) - 128) \
                + _t2*float(int(uint(_sc[4]) ^ 0x80u) - 128) + _t3*float(int(uint(_sc[6]) ^ 0x80u) - 128)); \
        } \
    } \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
        float acc = QSUM(_sumf[_r]); \
        if (lane == 0u && _row0+_r < N) { uint n = _row0+_r; STORE; } \
    }

// Q3_K, specialized. The generic body measured 168 Gw/s against 400 for IQ3_S
// at the SAME 3.44 bits/weight — same data density, 2.4x the throughput — so
// the cost is decode, not bandwidth. Two causes, both fixed here:
//
//  1. `Q3K_SUB` unpacks ALL SIXTEEN 6-bit scales (four kmask word ops) and then
//     uses two of them. Per sub-block. That is the full unpack done eight times
//     per block to read one eighth of it each time. `Q3K_SC1` computes one scale
//     directly from the three bytes it actually depends on.
//
//  2. The third bit lives in hmask as a per-weight INVERTED flag, evaluated as a
//     select-and-multiply per weight. But w = dl*(q - hi) with hi in {0,4}, so
//         SUM y*w == dl * ( SUM y*q  -  4 * SUM_{hmask clear} y )
//     and the correction collapses to one multiply per 16 weights. Same trick as
//     the IQ1_S delta; it generalizes to any format whose offset is per-group.
//
// Verified against the CPU dequantizer, which does the unfactored arithmetic on
// the unpacked-all-sixteen scales — two different formulations agreeing.
#define Q3K_SC1(RAW, I, OUT) { uint _w3=(I)>>2, _b3=(I)&3u; \
    uint _lo3 = ((_w3 & 1u)==0u) ? uint((RAW)[_b3]) : uint((RAW)[4u+_b3]); \
    uint _nb3 = (_w3 < 2u) ? (_lo3 & 0x0Fu) : (_lo3 >> 4); \
    uint _hb3 = (uint((RAW)[8u+_b3]) >> (2u*_w3)) & 3u; \
    OUT = int(_nb3 | (_hb3 << 4)); }

#define NAT_GEMV_Q3K(WALK_SUB, BLKB, WPB, SUBS, SUBW, NR, STORE) \
    uint _row0 = (tgid*(ts/32u) + sgid) * (NR); \
    if (_row0 >= N) { return; } \
    uint nb = K/(WPB); \
    uint _tot = nb*(SUBS); \
    float _sumf[NR]; \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { _sumf[_r] = 0.0f; } \
    float _yl[SUBW]; \
    for (uint _t = lane; _t < _tot; _t += 32u) { \
        uint _blk = _t/(SUBS), _il = _t%(SUBS); \
        uint _cb = _blk*(WPB) + _il*(SUBW); \
        QUNROLL for (uint _i = 0u; _i < (SUBW); _i++) { _yl[_i] = x[_cb+_i]; } \
        uint _n3 = _il>>2, _j3 = _il&3u, _sh3 = 2u*_j3, _m3 = 1u<<_il; \
        QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
            if (_row0+_r >= N) { break; } \
            QG uchar* _b = w + (QU64)(_row0+_r)*(QU64)nb*(BLKB) + (QU64)_blk*110u; \
            float _dall = QF16(_b+108u); \
            QG uchar* _hm = _b; QG uchar* _q = _b+32u+_n3*32u; QG uchar* _raw = _b+96u; \
            QUNROLL for (uint _h3 = 0u; _h3 < 2u; _h3++) { \
                int _sc3; Q3K_SC1(_raw, 2u*_il+_h3, _sc3) \
                float _dl3 = _dall*float(_sc3-32); \
                float _qs3 = 0.0f, _cs3 = 0.0f; \
                uint _o3 = _h3*16u; \
                QUNROLL for (uint _l = 0u; _l < 16u; _l++) { \
                    float _yv = _yl[_o3+_l]; \
                    _qs3 += _yv * float((uint(_q[_o3+_l])>>_sh3)&3u); \
                    if ((uint(_hm[_o3+_l]) & _m3) == 0u) { _cs3 += _yv; } } \
                _sumf[_r] += _dl3 * (_qs3 - 4.0f*_cs3); \
            } \
        } \
    } \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
        float acc = QSUM(_sumf[_r]); \
        if (lane == 0u && _row0+_r < N) { uint n = _row0+_r; STORE; } \
    }

// IQ1_S, specialized. Two departures from the generic body, both taken from
// the reference kernel_mul_mv_iq1_s_f32_impl:
//
//  1. It reads `iq1s_grid_gpu` (u32, four bytes of PACKED NIBBLES) instead of
//     `iq1s_grid` (u64, eight signed bytes). Same codebook, half the traffic —
//     and the codebook is read once per 8 weights, so it is not a small term.
//
//  2. The DELTA is factored out of the inner loop. IQ1_S weights are
//     dl*(g - 1 +/- delta), and
//         SUM_j y_j * dl*(g_j - 1 +/- d) == dl * ( SUM_j y_j*g_j + SUM_j y_j * (-1 +/- d) )
//     so the offset costs ONE multiply-add per 32 weights instead of 32. The
//     activation sum is free — it is accumulated while staging `_yl`.
//
// The generic body cannot do this: it consumes the walker's finished per-weight
// value `_v`, which has already folded scale and offset together. That is the
// price of one decoder serving both the requantizer and the matvec, and it is
// why specialization here is opt-in per format rather than a rewrite of the
// walker protocol.
#define NAT_GEMV_IQ1S(WALK_SUB, BLKB, WPB, SUBS, SUBW, NR, STORE) \
    uint _row0 = (tgid*(ts/32u) + sgid) * (NR); \
    if (_row0 >= N) { return; } \
    uint nb = K/(WPB); \
    uint _tot = nb*(SUBS); \
    float _sumf[NR]; \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { _sumf[_r] = 0.0f; } \
    float _yl[SUBW]; \
    for (uint _t = lane; _t < _tot; _t += 32u) { \
        uint _blk = _t/(SUBS), _il = _t%(SUBS); \
        uint _cb = _blk*(WPB) + _il*(SUBW); \
        float _ysum = 0.0f; \
        QUNROLL for (uint _i = 0u; _i < (SUBW); _i++) { _yl[_i] = x[_cb+_i]; _ysum += _yl[_i]; } \
        QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
            if (_row0+_r >= N) { break; } \
            QG uchar* _b = w + (QU64)(_row0+_r)*(QU64)nb*(BLKB) + (QU64)_blk*50u; \
            float _d = QF16(_b); \
            QG uchar* _qs = _b + 2u + 4u*_il; \
            uint _qh = uint(_b[34u+2u*_il]) | (uint(_b[35u+2u*_il])<<8); \
            float _s = 0.0f; \
            QUNROLL for (uint _l = 0u; _l < 4u; _l++) { \
                uint _g = iq1s_grid_gpu[uint(_qs[_l]) | (((_qh >> (3u*_l)) & 7u) << 8)]; \
                QUNROLL for (uint _j = 0u; _j < 4u; _j++) { \
                    uint _by = (_g >> (8u*_j)) & 0xFFu; \
                    _s += _yl[_l*8u+_j]*float(_by & 0x0Fu) \
                        + _yl[_l*8u+4u+_j]*float(_by >> 4); } } \
            float _off = ((_qh & 0x8000u)!=0u) ? (-1.0f-0.125f) : (-1.0f+0.125f); \
            _sumf[_r] += _d * (_s + _ysum*_off) * float(2u*((_qh>>12)&7u)+1u); \
        } \
    } \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
        float acc = QSUM(_sumf[_r]); \
        if (lane == 0u && _row0+_r < N) { uint n = _row0+_r; STORE; } \
    }

// LANE-COOPERATIVE + REGISTER-STAGED + MULTI-ROW.
//
// Three things happen here, and they compound:
//
//  1. Work items are (block, sub-block) pairs taken by lane L strided by 32, so
//     eight consecutive lanes decode eight sub-blocks of the SAME block. One
//     lane decodes 32 weights, not 256, and neighbours read neighbouring bytes.
//
//  2. The 32 activations for a sub-block are staged into REGISTERS once and
//     reused for every weight and every row. This is the big one: at 2 bits per
//     weight the weight stream is N*K*0.26 bytes but re-reading x per row is
//     N*K*4 — fifteen times larger. `_sub` (offset within the sub-block) is what
//     makes the index static; a runtime index into a thread-local array spills
//     to device memory on Apple GPUs and would undo the whole optimization.
//
//  3. NR output rows per simdgroup divide that activation load further.
//
// This is the shape of the reference kernel_mul_mv_<type>_f32_impl, arrived at the
// same way: their yl[32] + N_R0 rows + lane-strided sub-blocks.
#define NAT_GEMV_BODY(WALK_SUB, BLKB, WPB, SUBS, SUBW, NR, STORE) \
    uint _row0 = (tgid*(ts/32u) + sgid) * (NR); \
    if (_row0 >= N) { return; } \
    uint nb = K/(WPB); \
    uint _tot = nb*(SUBS); \
    float _sumf[NR]; \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { _sumf[_r] = 0.0f; } \
    float _yl[SUBW]; \
    for (uint _t = lane; _t < _tot; _t += 32u) { \
        uint _blk = _t/(SUBS), _il = _t%(SUBS); \
        uint _cb = _blk*(WPB) + _il*(SUBW); \
        QUNROLL for (uint _i = 0u; _i < (SUBW); _i++) { _yl[_i] = x[_cb+_i]; } \
        QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
            if (_row0+_r >= N) { break; } \
            QG uchar* wr = w + (QU64)(_row0+_r)*(QU64)nb*(BLKB); \
            WALK_SUB(wr, _blk, _il, _sumf[_r] += _v * _yl[_sub];) \
        } \
    } \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
        float acc = QSUM(_sumf[_r]); \
        if (lane == 0u && _row0+_r < N) { uint n = _row0+_r; STORE; } \
    }

// Q8_0 specialization matching the reference kernel's work assignment. Four
// neighbouring lanes consume one 32-weight block (8 values each), while the
// SIMDgroups split K for the same NR output rows. The generic body assigns each
// SIMDgroup different rows; on Flash's wide Q8 projections that leaves four
// times fewer threadgroups and makes every group walk all of K serially.
#define NAT_GEMV_Q80(WALK_SUB, BLKB, WPB, SUBS, SUBW, NR, STORE) \
    uint _row0 = tgid * (NR); \
    if (_row0 >= N) { return; } \
    uint _nsg = ts / 32u; \
    uint _ix = lane / 4u, _il = lane % 4u; \
    uint _nb = K / 32u; \
    float _sumf[NR]; \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { _sumf[_r] = 0.0f; } \
    for (uint _ib = sgid * 8u + _ix; _ib < _nb; _ib += _nsg * 8u) { \
        uint _col = _ib * 32u + _il * 8u; \
        QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
            if (_row0 + _r >= N) { break; } \
            QG uchar* _b = w + (QU64)(_row0 + _r) * (QU64)_nb * 34u + (QU64)_ib * 34u; \
            float _d = QF16(_b); QG char* _q = (QG char*)(_b + 2u); \
            float _s = 0.0f; \
            QUNROLL for (uint _j = 0u; _j < 8u; _j++) { _s += float(_q[_il * 8u + _j]) * x[_col + _j]; } \
            _sumf[_r] += _s * _d; \
        } \
    } \
    QSHARED float _part[(NR)*4u]; \
    QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
        float _v = QSUM(_sumf[_r]); \
        if (lane == 0u) { _part[_r * 4u + sgid] = _v; } \
    } \
    QBAR(); \
    if (sgid == 0u && lane < (NR) && _row0 + lane < N) { \
        uint _r = lane; float acc = 0.0f; \
        for (uint _s = 0u; _s < _nsg; _s++) { acc += _part[_r * 4u + _s]; } \
        uint n = _row0 + _r; STORE; \
    }

// Batched (prefill) form: x is [M][K] row-major, y is [M][N], ACC selects +=.
//
// The weights of a sub-block are decoded ONCE into registers, then each row does
// a CONTIGUOUS dot against them. That ordering is the whole point:
//
//   the previous version put the row loop INSIDE the walker, so every weight
//   cost M scattered device loads — rows are K*4 bytes apart, so no coalescing
//   and no reuse. Measured at M=2 it cost 8-10x the M=1 decode kernel where it
//   should cost ~2x (0.49 vs 0.057 ms on iq2xxs), and an MTP verify came to
//   946 ms against 59 for the tuned path.
//
//   Now the 32 x reads per row are sequential, which the decode path already
//   relies on, and the weight decode is amortized across all M rows instead of
//   being repeated per row.
//
// `wv[SUBW]` is indexed by `_sub`, which is a compile-time constant after
// unrolling — the same reason `_sub` exists for the decode body. A runtime index
// here would spill and undo everything.
// Batched (M rows of x) native matvec.
//
// TWO THINGS DOMINATE THIS KERNEL, and only one of them is the weights.
//
// The first version gave each simdgroup ONE output row and had it read all of x
// from device memory. That costs N*M*K*4 bytes of activation traffic against
// N*K*bpw/8 of weight traffic -- at 2 bits/weight and M=3 that is 46x MORE
// TRAFFIC ON x THAN ON THE WEIGHTS. A low-bit format makes the weight stream so
// small that the activations, not the weights, set the runtime. Measured 8-10x
// the M=1 decode kernel per token, which is what "batching" was supposed to
// amortize away.
//
// So x is staged in shared memory and reused, on two axes at once:
//   - a CHUNK-column tile of x is loaded cooperatively by the whole block
//     and read by all NSG simdgroups in it, and
//   - each simdgroup owns NR0 output rows, so one staged tile serves NR0 rows.
// Activation traffic falls by NSG*NR0 (32 at the default 256 threads), which
// brings it under the weight stream instead of 46x over it.
//
// M is tiled at MTILE so `p[MTILE][NR0]` and the staging buffer stay
// compile-time sized. M>MTILE re-reads the weights once per tile; speculation
// runs M<=4, so in practice there is one tile and the weights are read once.
//
// NOTE THE ABSENCE OF AN EARLY `return`. Out-of-range rows are dropped per-row
// at the store, not by exiting the thread, because every thread in the block
// has to reach the staging barriers.
#define NAT_GEMV_M_STAGED(WALK_SUB, BLKB, WPB, SUBS, NAT_M_CHUNK, NAT_M_MTILE, NAT_M_NR0) \
    QSHARED float _xs[(NAT_M_MTILE)*(NAT_M_CHUNK)]; \
    uint _tid = sgid*32u + lane; \
    uint _n0 = (tgid*(ts/32u) + sgid) * (NAT_M_NR0); \
    uint nb = K/(WPB); \
    for (uint m0 = 0u; m0 < M; m0 += NAT_M_MTILE) { \
        uint mm = min(NAT_M_MTILE, M - m0); \
        float p[(NAT_M_MTILE)][(NAT_M_NR0)]; \
        QUNROLL for (uint m = 0u; m < (NAT_M_MTILE); m++) { \
            QUNROLL for (uint _r = 0u; _r < (NAT_M_NR0); _r++) { p[m][_r] = 0.0f; } } \
        for (uint c0 = 0u; c0 < K; c0 += NAT_M_CHUNK) { \
            uint _cn = min(NAT_M_CHUNK, K - c0); \
            QBAR(); \
            for (uint _i = _tid; _i < mm*_cn; _i += ts) { \
                uint _m = _i/_cn, _c = _i - _m*_cn; \
                _xs[_m*(NAT_M_CHUNK) + _c] = x[(QU64)(m0+_m)*(QU64)K + (QU64)(c0+_c)]; } \
            QBAR(); \
            uint _b0 = c0/(WPB), _tot = (_cn/(WPB))*(SUBS); \
            for (uint _t = lane; _t < _tot; _t += 32u) { \
                uint _blk = _b0 + _t/(SUBS), _il = _t%(SUBS); \
                uint _off = (_t/(SUBS))*(WPB) + _il*(SUBW_OF(WPB, SUBS)); \
                QUNROLL for (uint _r = 0u; _r < (NAT_M_NR0); _r++) { \
                    uint _n = _n0 + _r; \
                    if (_n >= N) { continue; } \
                    QG uchar* wr = w + (QU64)_n*(QU64)nb*(BLKB); \
                    float _wv[SUBW_OF(WPB, SUBS)]; \
                    WALK_SUB(wr, _blk, _il, _wv[_sub] = _v;) \
                    for (uint m = 0u; m < mm; m++) { \
                        QSPTR float* _xr = _xs + m*(NAT_M_CHUNK) + _off; \
                        float _a = 0.0f; \
                        QUNROLL for (uint _i = 0u; _i < (SUBW_OF(WPB, SUBS)); _i++) { \
                            _a += _wv[_i] * _xr[_i]; } \
                        p[m][_r] += _a; } } \
            } \
        } \
        QUNROLL for (uint _r = 0u; _r < (NAT_M_NR0); _r++) { \
            for (uint m = 0u; m < mm; m++) { \
                float r = QSUM(p[m][_r]); \
                uint _n = _n0 + _r; \
                if (lane == 0u && _n < N) { \
                    if (ACC != 0u) { y[(QU64)(m0+m)*(QU64)N + _n] += r; } \
                    else { y[(QU64)(m0+m)*(QU64)N + _n] = r; } } } } \
    }


// The device-read predecessor, kept as a MEASURABLE ALTERNATIVE rather than
// deleted. It reads x straight from device memory with no staging and no tile
// reuse; the staged body above is only worth its shared memory and its
// barriers if it beats this, and that is a question for the sweep, not for a
// bandwidth argument on paper. (The first bandwidth argument said staging would
// win by a wide margin. It lost.)
#define NAT_GEMV_M_DEV(WALK_SUB, BLKB, WPB, SUBS, MTILE) \
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; } \
    uint nb = K/(WPB); \
    QG uchar* wr = w + (QU64)n*(QU64)nb*(BLKB); \
    uint _tot = nb*(SUBS); \
    for (uint m0 = 0u; m0 < M; m0 += (MTILE)) { \
        uint mm = min((uint)(MTILE), M - m0); \
        float p[MTILE]; \
        for (uint m = 0u; m < mm; m++) { p[m] = 0.0f; } \
        for (uint _t = lane; _t < _tot; _t += 32u) { \
            uint _blk = _t/(SUBS), _il = _t%(SUBS); \
            uint _cb = _blk*(WPB) + _il*(SUBW_OF(WPB, SUBS)); \
            float _wv[SUBW_OF(WPB, SUBS)]; \
            WALK_SUB(wr, _blk, _il, _wv[_sub] = _v;) \
            for (uint m = 0u; m < mm; m++) { \
                QG float* _xr = x + (QU64)(m0+m)*(QU64)K + _cb; \
                float _a = 0.0f; \
                QUNROLL for (uint _i = 0u; _i < (SUBW_OF(WPB, SUBS)); _i++) { \
                    _a += _wv[_i] * _xr[_i]; } \
                p[m] += _a; } \
        } \
        for (uint m = 0u; m < mm; m++) { \
            float r = QSUM(p[m]); \
            if (lane == 0u) { \
                if (ACC != 0u) { y[(QU64)(m0+m)*(QU64)N + n] += r; } \
                else { y[(QU64)(m0+m)*(QU64)N + n] = r; } } } \
    }

// Third structure: device reads, but x held in REGISTERS and reused across NR0
// output rows.
//
// The staged body reduced activation traffic and still lost, because of where
// the reads land rather than how many there are: lane L reads columns
// [L*SUBW, L*SUBW+SUBW), so every lane in the simdgroup addresses the same
// threadgroup-memory bank and the 32 reads serialize. Padding cannot fix a
// stride that IS the bank count.
//
// So keep the loads in device memory, where the cache handles the stride, and
// attack the count instead: load one sub-block of x per m into registers ONCE,
// then decode NR0 different weight rows against it. x loads drop by NR0 with no
// shared memory, no barriers and no occupancy cost.
//
// Both loops over m are QUNROLL over the compile-time MTILE with a runtime
// guard, NOT `for m < mm`: `_yl[m]` with a runtime m is a dynamic index into a
// thread-private array, which on this GPU spills the whole array to device
// memory and costs more than the loads it was meant to save.
#define NAT_GEMV_M_REG(WALK_SUB, BLKB, WPB, SUBS, MTILE, NR0) \
    uint _n0 = (tgid*(ts/32u) + sgid) * (NR0); \
    if (_n0 >= N) { return; } \
    uint nb = K/(WPB); \
    uint _tot = nb*(SUBS); \
    for (uint m0 = 0u; m0 < M; m0 += (MTILE)) { \
        uint mm = min((uint)(MTILE), M - m0); \
        float p[MTILE][NR0]; \
        QUNROLL for (uint m = 0u; m < (MTILE); m++) { \
            QUNROLL for (uint _r = 0u; _r < (NR0); _r++) { p[m][_r] = 0.0f; } } \
        for (uint _t = lane; _t < _tot; _t += 32u) { \
            uint _blk = _t/(SUBS), _il = _t%(SUBS); \
            uint _cb = _blk*(WPB) + _il*(SUBW_OF(WPB, SUBS)); \
            float _yl[MTILE][SUBW_OF(WPB, SUBS)]; \
            QUNROLL for (uint m = 0u; m < (MTILE); m++) { \
                if (m >= mm) { break; } \
                QG float* _xr = x + (QU64)(m0+m)*(QU64)K + _cb; \
                QUNROLL for (uint _i = 0u; _i < (SUBW_OF(WPB, SUBS)); _i++) { \
                    _yl[m][_i] = _xr[_i]; } } \
            QUNROLL for (uint _r = 0u; _r < (NR0); _r++) { \
                if (_n0+_r >= N) { break; } \
                QG uchar* wr = w + (QU64)(_n0+_r)*(QU64)nb*(BLKB); \
                float _wv[SUBW_OF(WPB, SUBS)]; \
                WALK_SUB(wr, _blk, _il, _wv[_sub] = _v;) \
                QUNROLL for (uint m = 0u; m < (MTILE); m++) { \
                    if (m >= mm) { break; } \
                    float _a = 0.0f; \
                    QUNROLL for (uint _i = 0u; _i < (SUBW_OF(WPB, SUBS)); _i++) { \
                        _a += _wv[_i] * _yl[m][_i]; } \
                    p[m][_r] += _a; } } \
        } \
        QUNROLL for (uint _r = 0u; _r < (NR0); _r++) { \
            if (_n0+_r >= N) { break; } \
            QUNROLL for (uint m = 0u; m < (MTILE); m++) { \
                if (m >= mm) { break; } \
                float r = QSUM(p[m][_r]); \
                if (lane == 0u) { \
                    if (ACC != 0u) { y[(QU64)(m0+m)*(QU64)N + _n0+_r] += r; } \
                    else { y[(QU64)(m0+m)*(QU64)N + _n0+_r] = r; } } } } \
    }

// Fourth structure, and the one that works: THE M=1 BODY WITH MORE THAN ONE
// ACTIVATION ROW.
//
// The three attempts above all shared a mistake that is invisible until you put
// them next to NAT_GEMV_BODY: they materialize the decoded sub-block into a
// `_wv[SUBW]` array and then dot it against x. The M=1 body never does that. It
// passes the multiply-accumulate INTO the walker, so each decoded `_v` is
// consumed the instant it exists and never occupies a register:
//
//     WALK_SUB(wr, _blk, _il, _sumf[_r] += _v * _yl[_sub];)
//
// Materializing costs SUBW=32 extra live registers and a second pass over the
// sub-block, on a kernel whose occupancy is already set by register pressure.
// That is why "batched" kept losing to simply running the M=1 kernel twice: the
// comparison was never batching-vs-not, it was this body against a worse one.
//
// So: keep the walker fusion, keep NR rows per simdgroup, and give it MTILE
// activation rows. Registers go from _yl[SUBW] to _yl[MTILE][SUBW], which is the
// real cost and the reason MTILE stays small.
//
// THE HOT LOOP CARRIES NO RUNTIME BRANCH, and that is load-bearing. The first
// version guarded each accumulate with `if (_m < mm)` for the ragged last M
// tile. That branch stopped the compiler unrolling the walker's 32-iteration
// inner loop, which made `_sub` a runtime index into `_yl[MTILE][SUBW]`, which
// moved the whole array off registers and onto the stack: measured 451 ms
// against the generic body's 120 at M=2, a 3.8x LOSS from one `if`.
//
// Instead the activation row is CLAMPED (`min(m0+_m, M-1)`) so every lane always
// has a valid row to read, the accumulate runs unconditionally, and the only
// guard is at the store. Rows past M compute a duplicate of the last row and
// throw it away — a few wasted FMAs on the final tile, against keeping the
// array in registers.
#define NAT_GEMV_M_ROW(WALK_SUB, BLKB, WPB, SUBS, SUBW, NR, MTILE) \
    uint _row0 = (tgid*(ts/32u) + sgid) * (NR); \
    if (_row0 >= N) { return; } \
    uint nb = K/(WPB); \
    uint _tot = nb*(SUBS); \
    for (uint m0 = 0u; m0 < M; m0 += (MTILE)) { \
        float _sumf[MTILE][NR]; \
        QUNROLL for (uint _m = 0u; _m < (MTILE); _m++) { \
            QUNROLL for (uint _r = 0u; _r < (NR); _r++) { _sumf[_m][_r] = 0.0f; } } \
        float _yl[MTILE][SUBW]; \
        for (uint _t = lane; _t < _tot; _t += 32u) { \
            uint _blk = _t/(SUBS), _il = _t%(SUBS); \
            uint _cb = _blk*(WPB) + _il*(SUBW); \
            QUNROLL for (uint _m = 0u; _m < (MTILE); _m++) { \
                QG float* _xr = x + (QU64)min(m0+_m, M-1u)*(QU64)K + _cb; \
                QUNROLL for (uint _i = 0u; _i < (SUBW); _i++) { _yl[_m][_i] = _xr[_i]; } } \
            QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
                if (_row0+_r >= N) { break; } \
                QG uchar* wr = w + (QU64)(_row0+_r)*(QU64)nb*(BLKB); \
                WALK_SUB(wr, _blk, _il, \
                    QUNROLL for (uint _m = 0u; _m < (MTILE); _m++) { \
                        _sumf[_m][_r] += _v * _yl[_m][_sub]; }) \
            } \
        } \
        QUNROLL for (uint _r = 0u; _r < (NR); _r++) { \
            if (_row0+_r >= N) { break; } \
            QUNROLL for (uint _m = 0u; _m < (MTILE); _m++) { \
                if (m0+_m >= M) { break; } \
                float acc = QSUM(_sumf[_m][_r]); \
                uint n = _row0+_r; \
                if (lane == 0u) { \
                    if (ACC != 0u) { y[(QU64)(m0+_m)*(QU64)N + n] += acc; } \
                    else { y[(QU64)(m0+_m)*(QU64)N + n] = acc; } } } } \
    }

// weights per sub-block, as a macro so the array size stays compile-time
#define SUBW_OF(WPB, SUBS) ((WPB)/(SUBS))
"#;

#[cfg(test)]
mod mxfp4_tests {
    use super::*;

    fn mxfp4_row() -> &'static QFormat {
        FORMATS.iter().find(|f| f.tag == "mxfp4").expect("mxfp4 row present")
    }

    #[test]
    fn geometry_is_a_17byte_single_subblock_32weight_block() {
        let f = mxfp4_row();
        assert_eq!(f.ty, 139, "synthetic, non-GGUF type id");
        assert_eq!(f.block_bytes, 17);
        assert_eq!(f.weights, 32);
        assert_eq!(f.subs, 1);
        assert_eq!(f.sub_weights(), 32);
        assert!(!f.needs_grids, "MXFP4 uses a 16-entry LUT, not the IQ codebooks");
        // bits/weight the loader uses for its native-vs-requant math: 17*8/32.
        assert!((f.block_bytes as f32 * 8.0 / f.weights as f32 - 4.25).abs() < 1e-6);
        // format_of resolves it, and it is not any real GGUF type in the table.
        assert!(format_of(139).is_some());
    }

    #[test]
    fn decoder_and_lut_reach_both_dialects_without_metalisms() {
        // The decoder text is dialect-neutral; the exponent bitcast goes through
        // QFBITS so `as_type<` never leaks into CUDA.
        assert!(DECODERS.contains("MXFP4_SUB"));
        assert!(DECODERS.contains("kvalues_mxfp4"));
        assert!(DECODERS.contains("QFBITS(uint(_b[0]) << 23)"));
        assert!(!DECODERS.contains("as_type<"), "decoder body must stay dialect-neutral");
        let metal = format!("{}{}", prologue(Dialect::Metal), DECODERS);
        let cuda = format!("{}{}", prologue(Dialect::Cuda), DECODERS);
        assert!(metal.contains("as_type<float>(uint(U))"), "Metal QFBITS present");
        assert!(cuda.contains("__uint_as_float"), "CUDA QFBITS present");
        assert!(!cuda.contains("as_type<"), "no Metal-ism in the CUDA source");
        // The whole-block form is generated for it too (used by any requantizer).
        assert!(row_macros().contains("MXFP4_ROW"));
    }

    // The OCP E2M1 codebook, hand-verified: value = (-1)^s * (subnormal ?
    // man*0.5 : (1+man*0.5) * 2^(exp-1)).
    const LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    fn e8m0(e: u8) -> f32 { f32::from_bits((e as u32) << 23) }

    /// Rust reimplementation of MXFP4_SUB: decode one 17-byte ojas block to 32
    /// f32, in the exact index order the GPU walker emits (`_sub`).
    fn decode_block(blk: &[u8; 17]) -> [f32; 32] {
        let d = e8m0(blk[0]);
        let mut out = [0.0f32; 32];
        for j in 0..16 {
            let byte = blk[1 + j] as usize;
            out[2 * j] = d * LUT[byte & 0x0F]; // low nibble  -> weight 2j
            out[2 * j + 1] = d * LUT[byte >> 4]; // high nibble -> weight 2j+1
        }
        out
    }

    #[test]
    fn byte_exact_hand_computed_blocks() {
        // Block A: scale exponent 127 -> 2^0 = 1.0. Bytes chosen so the nibbles
        // sweep the whole codebook. qs[j] = (hi<<4)|lo.
        // weight order after decode: [lo0,hi0, lo1,hi1, ...].
        let mut a = [0u8; 17];
        a[0] = 127; // scale = 1.0
        // qs[0] = 0x10 -> lo=0 (0.0), hi=1 (0.5)
        // qs[1] = 0x32 -> lo=2 (1.0), hi=3 (1.5)
        // qs[2] = 0x54 -> lo=4 (2.0), hi=5 (3.0)
        // qs[3] = 0x76 -> lo=6 (4.0), hi=7 (6.0)
        // qs[4] = 0x98 -> lo=8 (-0.0), hi=9 (-0.5)
        // qs[5] = 0xBA -> lo=10 (-1.0), hi=11 (-1.5)
        // qs[6] = 0xDC -> lo=12 (-2.0), hi=13 (-3.0)
        // qs[7] = 0xFE -> lo=14 (-4.0), hi=15 (-6.0)
        for (j, &b) in [0x10u8, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE].iter().enumerate() {
            a[1 + j] = b;
        }
        // qs[8..16] = 0 -> all 0.0
        let got = decode_block(&a);
        let expect_first16 = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0f32,
        ];
        for i in 0..16 {
            assert_eq!(got[i], expect_first16[i], "weight {i}");
        }
        for i in 16..32 {
            assert_eq!(got[i], 0.0, "weight {i} (zero nibble)");
        }

        // Block B: scale exponent 129 -> 2^2 = 4.0, applied to nibble 5 (=3.0)
        // in every position -> 12.0, and nibble 13 (=-3.0) -> -12.0.
        let mut b = [0u8; 17];
        b[0] = 129;
        for j in 0..16 { b[1 + j] = 0xD5; } // lo=5 (3.0), hi=13 (-3.0)
        let got = decode_block(&b);
        for j in 0..16 {
            assert_eq!(got[2 * j], 12.0, "even weight {}", 2 * j);
            assert_eq!(got[2 * j + 1], -12.0, "odd weight {}", 2 * j + 1);
        }

        // Block C: scale exponent 126 -> 2^-1 = 0.5, nibble 2 (=1.0) -> 0.5.
        let mut c = [0u8; 17];
        c[0] = 126;
        c[1] = 0x22; // lo=2 (1.0), hi=2 (1.0)
        let got = decode_block(&c);
        assert_eq!(got[0], 0.5);
        assert_eq!(got[1], 0.5);
    }
}
