//! GPU requantization for the IQ family: IQ block -> int8 + one f32 scale/row.
//!
//! An IQ2 70B is ~18 GB on disk and ~140 GB as f16, so the CPU dequantize-to-f16
//! fallback does not merely load such a model slowly, it cannot load it at all.
//! Requantizing on the GPU keeps the whole conversion inside the resident pass and
//! hands the decode kernels the Q8 they already know, so no per-format GEMM or GEMV
//! kernel is needed.
//!
//! It is a separate family rather than part of the shared prelude because the
//! codebooks are ~15 KB of constant data and the prelude is prepended to every other
//! kernel family that compiles.
//!
//! Decode mirrors `dequant_to_f16` arms 16/17/18/21/22, which are checked
//! against gguf-py on random blocks by `iq_raw_gate`.

pub const REQUANT_IQ_KERNEL_NAMES: &[&str] = &[
    "requant_iq2xxs_q8", "requant_iq2xs_q8", "requant_iq2s_q8",
    "requant_iq3xxs_q8", "requant_iq3s_q8", "requant_iq1s_q8", "requant_iq1m_q8",
];

pub const BODY: &str = r#"
// The IQ grids hold magnitudes only, packed as bytes inside a u64 (IQ2, 8 per entry)
// or a u32 (IQ3, 4 per entry). Signs come from a separate word, one bit per lane; a
// set bit means negative.
// IQ1's grid bytes are signed, unlike every other grid here. Reading them unsigned
// silently folds the negative half of the range onto 128..255.

// IQ2_XXS: block { half d; u16 qs[32]; } = 66 B / 256. Each group of 8 weights
// costs one byte of grid index; the second u32 of every 8-byte stride packs the
// sub-block scale (top nibble) and four 7-bit sign-table indices.

// IQ2_XS: block { half d; u16 qs[32]; u8 scales[8]; } = 74 B / 256. Grid index
// and sign index share one u16 (9 bits + 7 bits), freeing the scale into its own
// nibble array. Lanes 0-1 of each group take the low nibble, 2-3 the high.

// IQ2_S: block { half d; u8 qs[64]; u8 qh[8]; u8 scales[8]; } = 82 B / 256.
// qs splits in half: 32 index bytes (2 more index bits come from qh), then 32
// RAW sign bytes rather than indices into ksigns.

// IQ3_XXS: block { half d; u8 qs[96]; } = 98 B / 256. Grid entries hold only 4
// magnitudes, so a group of 8 needs two indices; lanes 0-3 come from the first
// and 4-7 from the second, each reading the sign word at its own lane.

// IQ3_S: block { half d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]; } = 110 B / 256.
// One scale byte covers two sub-blocks of 32, so this walks four PAIRS rather
// than eight singles. The scale form is (1 + 2*s) — different from every other
// IQ type here, which use (0.5 + s).

// IQ1_S: block { half d; u8 qs[32]; u16 qh[8]; } = 50 B / 256. 1.56 bpw. Each
// sub-block has a 3-bit scale and a sign bit choosing DELTA, a constant offset
// added to every magnitude — at one bit per weight there is no room for a
// per-weight offset. Mirrors `dequant_to_f16` arm 19.

// IQ1_M: block { u8 qs[32]; u8 qh[16]; u8 scales[8]; } = 56 B / 256. 1.75 bpw,
// and the only block in the family with no `d` field: the f16 scale is
// reassembled from the top nibble of each of the four trailing scale words, low
// nibble first. Mirrors `dequant_to_f16` arm 29.

// One row per simdgroup, two passes: find the row amax, then write int8. Same
// shape as the K-quant requantizers in the gemv family — ffn_down rows are far
// too long to stage in threadgroup memory.
//
// Only the body is a macro. The `kernel void <name>(` line is spelled out for each
// entry because the kernel registry finds entry points by scanning source text for
// that literal, so a macro-generated name is invisible to it.
#define IQ_REQUANT_BODY(WALK, BLKB) \
    uint n = tgid*(ts/32u) + sgid; if (n >= N) { return; } \
    uint nsb = K/256u; device const uchar* wr = w + (ulong)n*(ulong)nsb*(BLKB); \
    float amax = 0.0f; \
    for (uint sb = lane; sb < nsb; sb += 32u) { WALK(wr, sb, amax = max(amax, fabs(_v));) } \
    amax = simd_max(amax); \
    float s = amax > 0.0f ? amax/127.0f : 1.0f; \
    if (lane == 0u) { sc[n] = s; } \
    float inv = 1.0f/s; \
    device char* qr = q + (ulong)n*(ulong)K; \
    for (uint sb = lane; sb < nsb; sb += 32u) { \
        WALK(wr, sb, qr[_idx] = char(clamp(rint(_v*inv), -127.0f, 127.0f));) \
    }

kernel void requant_iq2xxs_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    IQ_REQUANT_BODY(IQ2XXS_ROW, 66u)
}

kernel void requant_iq2xs_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    IQ_REQUANT_BODY(IQ2XS_ROW, 74u)
}

kernel void requant_iq2s_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    IQ_REQUANT_BODY(IQ2S_ROW, 82u)
}

kernel void requant_iq3xxs_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    IQ_REQUANT_BODY(IQ3XXS_ROW, 98u)
}

kernel void requant_iq3s_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    IQ_REQUANT_BODY(IQ3S_ROW, 110u)
}

kernel void requant_iq1s_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    IQ_REQUANT_BODY(IQ1S_ROW, 50u)
}

kernel void requant_iq1m_q8(device const uchar* w [[buffer(0)]], device char* q [[buffer(1)]],
    device float* sc [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    IQ_REQUANT_BODY(IQ1M_ROW, 56u)
}


"#;
