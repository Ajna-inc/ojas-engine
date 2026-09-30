//! Long-context attention kernels (flash-decoding + tiled prefill), modeled on the
//! flash_attn_ext family (see reference/).
//!
//! The score-array kernels (attention_short/_m_short) are fastest at seq <= 2048;
//! beyond that two structural problems appear:
//!  - decode: one threadgroup per head = 16 threadgroups walking the whole history
//!    serially, leaving the GPU idle (63 → 5 tok/s by 32k).
//!  - prefill: one query per threadgroup re-reads the entire KV per query, O(T²)
//!    bytes (401 → 63 tok/s prefill by 32k).
//!
//! `attention_part` + `attention_merge`: flash-decoding. The KV sequence is split
//! across NWG threadgroups per head (like the nwg=32 vec path); each
//! computes a streaming-softmax partial (m, l, acc[hd]) over its slice; a small
//! merge kernel combines partials exactly.
//!
//! `attention_m_mma`: prefill flash-attention on the simdgroup matrix units
//! (32 queries/tg, 8×8 MMA tiles for Q·K^T and P·V, hd-specialized variants).
//!
//! `page_minmax`/`page_select`/`attention_part_sparse`: Quest-style page-sparse
//! decode (OJAS_SPARSE), exact softmax over a per-head top-K page subset.
//! All sizes derived from hd ≤ 512; portable across Apple GPUs (32-wide simds,
//! ≥28KB threadgroup memory).

/// Kernels to register from this module's source.
pub const ATTN_KERNEL_NAMES: &[&str] = &["attention_part", "attention_merge",
    "attention_m_mma", "attention_m_mma_dq", "page_minmax", "page_select", "attention_part_sparse"];

/// Max KV-split workgroups per head (scratch is sized for this).
pub const ATTN_NWG: usize = 32;

/// Quest-style page-sparse decode: page size in KV positions.
pub const PAGE: usize = 64;
/// Max pages a head can select (plist row width; 256 pages = 16k tokens budget).
pub const MAXSEL: usize = 256;

/// Head dims that get a compile-time-specialized MMA prefill kernel
/// (`attention_m_mma_<hd>`): with hd pinned, the tile-array indices and the
/// dd/nt loops are fully static, so the compiler unrolls them and keeps the
/// simdgroup matrices in registers (runtime-indexed matrix arrays can spill).
pub const ATTN_HD_SPECIAL: &[u32] = &[128, 256];

/// Head dims that get a BIDIRECTIONAL (non-causal) MMA kernel
/// (`attention_m_mma_bidir_<hd>` + its f16-Q twin). The ViT tower this exists for
/// is hd=64; add a dim here when another encoder needs one.
pub const ATTN_BIDIR_HD: &[u32] = &[64];

/// Generate the hd-specialized MMA prefill kernel source (the reference compiles a
/// template instantiation per head dim for the same reason).
pub fn attn_mma_dq_hd_src(hd: u32) -> String {
    let s = ATTN_KERNELS;
    let start = s.find("kernel void attention_m_mma_dq(").expect("dq kernel in source");
    let end = start + s[start..].find("\nkernel void ").expect("kernel after dq");
    let body = s[start..end]
        .replace("attention_m_mma_dq(", &format!("attention_m_mma_dq_{hd}("))
        .replace("constant uint& hd [[buffer(4)]]", "constant uint& hd_rt [[buffer(4)]]")
        .replace(
            "const uint ts = 256u;",
            &format!("const uint ts = 256u; const uint hd = {hd}u; (void)hd_rt;"),
        );
    format!("#include <metal_stdlib>\nusing namespace metal;\n{body}")
}

pub fn attn_mma_hd_src(hd: u32) -> String {
    let s = ATTN_KERNELS;
    let start = s.find("kernel void attention_m_mma(").expect("mma kernel in source");
    let end = start + s[start..].find("\nkernel void ").expect("kernel after mma");
    // The staged-Q tile is declared for the hd <= 256 worst case (16.9 KB). With
    // hd pinned it only needs hd + 8, which on a 128-dim head is 8.7 KB — and
    // this kernel's threadgroup memory is what caps how many of it fit per core.
    let sq = hd + 8;
    let body = s[start..end]
        .replace("attention_m_mma(", &format!("attention_m_mma_{hd}("))
        .replace("constant uint& hd [[buffer(4)]]", "constant uint& hd_rt [[buffer(4)]]")
        .replace("threadgroup half  sq[32*264];", &format!("threadgroup half  sq[32*{sq}];"))
        .replace("const uint SQ = 264u;", &format!("const uint SQ = {sq}u;"))
        .replace(
            "const uint ts = 256u;",
            &format!("const uint ts = 256u; const uint hd = {hd}u; (void)hd_rt;"),
        );
    format!("#include <metal_stdlib>\nusing namespace metal;\n{body}")
}

/// Strip causality out of an MMA kernel body: buffer(6) stops being `base_pos` (KV
/// rows before this tile) and becomes `total` (the whole KV length), the same slot
/// and the same meaning `attention_m_bidir` gives it. Causality lives in exactly two
/// expressions — the KV block-loop bound and the per-row `ok0`/`ok1` column mask —
/// and both become `total`.
///
/// The assert guards against a rewrite that misses a site: that would compile and
/// return causal results under a bidirectional name, which no shape check catches.
fn mma_bidir_body(body: &str) -> String {
    let out = body
        .replace("constant uint& base_pos [[buffer(6)]]", "constant uint& total [[buffer(6)]]")
        .replace("uint maxseq = base_pos + q0 + nq;", "uint maxseq = total;")
        .replace("// longest causal row in this tile", "// BIDIRECTIONAL: every row sees all of K/V")
        .replace("uint myseq = base_pos + q0 + r + 1u;", "uint myseq = total;")
        .replace("// causal bound (valid rows only)", "// BIDIRECTIONAL: no causal bound");
    assert!(!out.contains("base_pos"), "MMA causality moved; the bidirectional rewrite is stale");
    out
}

/// Bidirectional (non-causal) MMA prefill: every query row attends to all `total` KV
/// positions. `attention_m_bidir` (attn_core) is correct for that shape and stays the
/// oracle, but it gives each (query, head) its own threadgroup, so every query streams
/// the head's entire K and V — at 16384 patches x 12 heads, ~824 GB per layer and ~2 s
/// on an M2 Max. This shares one KV pass across 32 queries, the 1/BQ traffic scaling
/// that keeps 32k causal prefill off the memory wall.
///
/// Generated from `attention_m_mma` by the same string rewrite `attn_mma_hd_src` uses
/// (this repo has no Metal function constants), so the tiling, the online softmax and
/// the store path are that kernel byte for byte.
pub fn attn_mma_bidir_src(hd: u32) -> String {
    let s = ATTN_KERNELS;
    let start = s.find("kernel void attention_m_mma(").expect("mma kernel in source");
    let end = start + s[start..].find("\nkernel void ").expect("kernel after mma");
    let sq = hd + 8;                                   // staged-Q stride, as in attn_mma_hd_src
    let body = mma_bidir_body(&s[start..end])
        .replace("attention_m_mma(", &format!("attention_m_mma_bidir_{hd}("))
        .replace("constant uint& hd [[buffer(4)]]", "constant uint& hd_rt [[buffer(4)]]")
        .replace("threadgroup half  sq[32*264];", &format!("threadgroup half  sq[32*{sq}];"))
        .replace("const uint SQ = 264u;", &format!("const uint SQ = {sq}u;"))
        .replace(
            "const uint ts = 256u;",
            &format!("const uint ts = 256u; const uint hd = {hd}u; (void)hd_rt;"),
        );
    format!("#include <metal_stdlib>\nusing namespace metal;\n{body}")
}

/// f16-Q twin of `attn_mma_bidir_src`: Q read as fragments straight from device
/// (written by `q_to_half`, zeroed past the token count) instead of staged into a
/// threadgroup array. Same two-site rewrite, applied to `attention_m_mma_dq`.
pub fn attn_mma_dq_bidir_src(hd: u32) -> String {
    let s = ATTN_KERNELS;
    let start = s.find("kernel void attention_m_mma_dq(").expect("dq kernel in source");
    let end = start + s[start..].find("\nkernel void ").expect("kernel after dq");
    let body = mma_bidir_body(&s[start..end])
        .replace("attention_m_mma_dq(", &format!("attention_m_mma_dq_bidir_{hd}("))
        .replace("constant uint& hd [[buffer(4)]]", "constant uint& hd_rt [[buffer(4)]]")
        .replace(
            "const uint ts = 256u;",
            &format!("const uint ts = 256u; const uint hd = {hd}u; (void)hd_rt;"),
        );
    format!("#include <metal_stdlib>\nusing namespace metal;\n{body}")
}

/// Name of the kernel [`attn_bidir_span_src`] generates.
pub const ATTN_BIDIR_SPAN: &str = "attention_m_bidir_span";

/// `attention_m_bidir` with a per-row key range: query row `m` attends to keys
/// `[span[m].x, span[m].y)` instead of `[0, total)`.
///
/// One range per row expresses both things a packed text-encoder batch needs:
/// sequence boundaries (several independent sequences share one row buffer) and a
/// symmetric local window (ModernBERT's sliding layers, `|i - j| <= w`), each
/// intersected on the host. The online softmax, the simdgroup split and the store are
/// `attention_m_bidir`'s, byte for byte; only the loop bounds move, so that kernel
/// stays the oracle for this one. Every span must be non-empty.
///
/// buffers: as `attention_m_bidir`, plus 10 `span[M]` (uint2).
pub fn attn_bidir_span_src() -> String {
    let s = super::attn_core::BODY;
    let start = s.find("kernel void attention_m_bidir(").expect("attention_m_bidir in attn_core");
    let end = start + s[start..].find("\nkernel void ").expect("kernel after attention_m_bidir");
    let mut body = s[start..end].to_string();
    for (from, to) in [
        ("attention_m_bidir(", "attention_m_bidir_span("),
        ("constant uint& n_head [[buffer(9)]],",
         "constant uint& n_head [[buffer(9)]], device const uint2* span [[buffer(10)]],"),
        ("uint seq = total;                      // BIDIRECTIONAL: attend to every position",
         "uint2 sp = span[m]; uint seq = sp.y; (void)total;   // keys [sp.x, sp.y) only"),
        ("for (uint t = sgid; t < seq; t += nsg)", "for (uint t = sp.x + sgid; t < seq; t += nsg)"),
    ] {
        assert!(body.contains(from), "attention_m_bidir changed shape; the span rewrite of `{from}` is stale");
        body = body.replacen(from, to, 1);
    }
    format!("#include <metal_stdlib>\nusing namespace metal;\n{body}")
}

/// Name of the kernel [`attn_mma_span_src`] generates for head dim `hd`.
pub fn attn_mma_span_name(hd: u32) -> String { format!("attention_m_mma_span_{hd}") }

/// MMA attention over packed sequences with per-row key spans: the tiled twin of
/// [`attn_bidir_span_src`], generated from `attention_m_mma` so its tiling, online
/// softmax and store path are that kernel's byte for byte.
///
/// Each threadgroup takes one query tile from `tiles` (slot 11), a `uint4` of
/// `(q0, nq, klo, khi)`: up to 32 query rows starting at `q0`, all from one sequence,
/// and the key range `[klo, khi)` covering every row's span. Rows are then masked to
/// their own `span[row]` (slot 12, `uint2`, as the scalar kernel reads it), which is
/// how a sliding window clips inside a tile. The grid is `(n_head, n_tiles)`.
///
/// Key blocks are read eight rows at a time past `khi`; those rows are masked, and the
/// caller keeps K and V finite (zeroed) at least 64 rows past the last token so the
/// masked products stay zero.
pub fn attn_mma_span_src(hd: u32) -> String {
    let s = ATTN_KERNELS;
    let start = s.find("kernel void attention_m_mma(").expect("mma kernel in source");
    let end = start + s[start..].find("\nkernel void ").expect("kernel after mma");
    let sq = hd + 8;
    let mut body = s[start..end].to_string();
    for (from, to) in [
        ("attention_m_mma(".to_string(), format!("{}(", attn_mma_span_name(hd))),
        ("constant uint& mtok [[buffer(10)]],".into(),
         "constant uint& mtok [[buffer(10)]], device const uint4* tiles [[buffer(11)]], \
          device const uint2* span [[buffer(12)]],".into()),
        ("constant uint& hd [[buffer(4)]]".into(), "constant uint& hd_rt [[buffer(4)]]".into()),
        ("threadgroup half  sq[32*264];".into(), format!("threadgroup half  sq[32*{sq}];")),
        ("const uint SQ = 264u;".into(), format!("const uint SQ = {sq}u;")),
        ("const uint ts = 256u;".into(), format!("const uint ts = 256u; const uint hd = {hd}u; (void)hd_rt;")),
        ("uint q0 = qt*32u;\n    uint nq = min(32u, mtok - q0);".into(),
         "uint4 td = tiles[qt]; uint q0 = td.x; uint nq = td.y; (void)mtok;".into()),
        ("uint maxseq = base_pos + q0 + nq;              // longest causal row in this tile".into(),
         "uint klo = td.z; uint maxseq = td.w; (void)base_pos;   // keys [klo, maxseq) cover every row".into()),
        ("for (uint c0 = 0u; c0 < maxseq; c0 += C) {".into(), "for (uint c0 = klo; c0 < maxseq; c0 += C) {".into()),
        ("uint myseq = base_pos + q0 + r + 1u;   // causal bound (valid rows only)\n            \
          bool ok0 = (r < nq) && (c0 + lane < myseq);\n            \
          bool ok1 = (r < nq) && (c0 + lane + 32u < myseq);".into(),
         "uint2 rsp = span[q0 + min(r, nq - 1u)];   // this row's keys [rsp.x, rsp.y)\n            \
          bool ok0 = (r < nq) && (c0 + lane >= rsp.x) && (c0 + lane < rsp.y);\n            \
          bool ok1 = (r < nq) && (c0 + lane + 32u >= rsp.x) && (c0 + lane + 32u < rsp.y);".into()),
    ] {
        assert!(body.contains(from.as_str()), "attention_m_mma changed shape; the span rewrite of `{from}` is stale");
        body = body.replacen(from.as_str(), &to, 1);
    }
    assert!(!body.contains("base_pos + q0"), "a causal bound survived the span rewrite");
    format!("#include <metal_stdlib>\nusing namespace metal;\n{body}")
}

pub const ATTN_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---- flash-decoding: one threadgroup per (head, kv-slice) → partial (m, l, acc).
// part layout: [head][wg][hd + 2] floats; slice w covers positions [w*chunk, end).
// Vectorized (flash_attn_ext_vec pattern): Q lives in per-lane float4 registers, K/V
// rows are read as half4, and each simdgroup scores 4 positions per pass so the four
// simd_sum reductions pipeline instead of serializing.
// Lane e owns elements [c*128 + lane*4, +4) of the head — the same indexing is
// used for the V accumulator so the threadgroup merge stays element-aligned.
kernel void attention_part(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* part [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kvdim [[buffer(5)]], constant uint& seq [[buffer(6)]],
    constant uint& group [[buffer(7)]], constant float& scale [[buffer(8)]], constant uint& nwg [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint ts = 256u;
    threadgroup float tacc[8*512];
    threadgroup float tm[8];
    threadgroup float tl[8];
    uint head = tg.x; uint wg = tg.y;
    uint kvh = head/group;
    uint chunk = (seq + nwg - 1u) / nwg;
    uint t0 = wg * chunk;
    uint t1 = min(t0 + chunk, seq);
    uint nsg = ts / 32u;
    uint nc4 = (hd + 127u)/128u;                   // float4 chunks per lane (hd ≤ 512)
    device const float4* qh4 = (device const float4*)(q + head*hd);
    float4 qr[4];
    for (uint c = 0u; c < nc4; c++) {
        uint e = c*32u + lane;                     // float4 index within the head
        qr[c] = (e*4u < hd) ? qh4[e] : float4(0.0);
    }
    float mi = -1e30, li = 0.0;
    float4 acc[4] = {float4(0.0), float4(0.0), float4(0.0), float4(0.0)};
    ulong kv4 = (ulong)kvdim/4u;                   // row stride in half4 units
    ulong base4 = (ulong)(kvh*hd)/4u;              // head offset in half4 units
    for (uint t = t0 + sg*4u; t < t1; t += nsg*4u) {
        // score up to 4 positions: independent dots + reductions overlap
        float sv[4];
        for (uint j = 0u; j < 4u; j++) {
            float s = 0.0;
            if (t + j < t1) {
                device const half4* kt4 = (device const half4*)kc + (ulong)(t + j)*kv4 + base4;
                for (uint c = 0u; c < nc4; c++) {
                    uint e = c*32u + lane;
                    if (e*4u < hd) { s += dot(float4(kt4[e]), qr[c]); }
                }
            }
            sv[j] = simd_sum(s)*scale;
        }
        // online softmax + V accumulation, in position order (exact)
        for (uint j = 0u; j < 4u; j++) {
            if (t + j >= t1) { break; }
            float mn = max(mi, sv[j]);
            float corr = exp(mi - mn); float pw = exp(sv[j] - mn);
            li = li*corr + pw;
            device const half4* vt4 = (device const half4*)vc + (ulong)(t + j)*kv4 + base4;
            for (uint c = 0u; c < nc4; c++) {
                uint e = c*32u + lane;
                float4 v = (e*4u < hd) ? float4(vt4[e]) : float4(0.0);
                acc[c] = acc[c]*corr + pw*v;
            }
            mi = mn;
        }
    }
    if (lane == 0u) { tm[sg] = mi; tl[sg] = li; }
    threadgroup float4* tacc4 = (threadgroup float4*)tacc;
    for (uint c = 0u; c < nc4; c++) {
        uint e = c*32u + lane;
        if (e*4u < hd) { tacc4[(sg*hd)/4u + e] = acc[c]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = -1e30; for (uint j = 0u; j < nsg; j++) { gm = max(gm, tm[j]); }
    float gl = 0.0; for (uint j = 0u; j < nsg; j++) { gl += tl[j]*exp(tm[j] - gm); }
    device float* po = part + (ulong)(head*nwg + wg)*(ulong)(hd + 2u);
    for (uint i = lid; i < hd; i += ts) {
        float o = 0.0;
        for (uint j = 0u; j < nsg; j++) { o += tacc[j*hd + i]*exp(tm[j] - gm); }
        po[i] = o;                                 // unnormalized weighted V
    }
    if (lid == 0u) { po[hd] = gm; po[hd + 1u] = gl; }
}

// merge partials exactly: out = Σ_w acc_w·exp(m_w − M) / Σ_w l_w·exp(m_w − M)
kernel void attention_merge(device const float* part [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& hd [[buffer(2)]], constant uint& nwg [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint ts [[threads_per_threadgroup]]) {
    device const float* ph = part + (ulong)(head*nwg)*(ulong)(hd + 2u);
    float gm = -1e30;
    for (uint w = 0u; w < nwg; w++) { gm = max(gm, ph[w*(hd + 2u) + hd]); }
    float gl = 0.0;
    for (uint w = 0u; w < nwg; w++) { gl += ph[w*(hd + 2u) + hd + 1u]*exp(ph[w*(hd + 2u) + hd] - gm); }
    for (uint i = lid; i < hd; i += ts) {
        float o = 0.0;
        for (uint w = 0u; w < nwg; w++) { o += ph[w*(hd + 2u) + i]*exp(ph[w*(hd + 2u) + hd] - gm); }
        out[head*hd + i] = o/gl;
    }
}

// ---- MMA prefill flash-attention (kernel_flash_attn_ext design, widened to BQ=32):
// 32 queries per threadgroup as four 8-row blocks, 8 simdgroups, causal, online
// softmax. KV traffic scales as 1/BQ (each query tile walks the whole cache), so the
// wide tile is what keeps 32k prefill off the memory wall; each K/V 8x8 tile is loaded
// once and reused by all four row-blocks (4 MMAs per load). K/V tiles come
// transposed/straight from the device f16 cache, unstaged. Scores round-trip
// threadgroup memory for masking and streaming softmax (four rows per simdgroup), then
// P(32x64)*V accumulates into per-simdgroup 8x8 output tiles, each simdgroup owning
// hd/8 output columns. Rescaling uses diag(factor)*O MMAs. Full 8-row blocks store
// simdgroup-direct to device; tail rows stage through the score buffer, dead after the
// block loop. hd <= 256 and hd % 64 == 0 (dispatch guards; attention_m covers the
// rest). Registered both generic and hd-specialized (see attn_mma_hd_src).
kernel void attention_m_mma(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kvdim [[buffer(5)]], constant uint& base_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]], constant float& scale [[buffer(8)]], constant uint& n_head [[buffer(9)]],
    constant uint& mtok [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint ts = 256u;                          // 8 simdgroups
    const uint C = 64u;                            // KV positions per block
    const uint SQ = 264u;                          // staged-Q stride (hd <= 256, padded)
    const uint SP = 72u;                           // staged-P stride (C + pad)
    threadgroup half  sq[32*264];                  // Q tile, half (16.9KB)
    threadgroup float ss[32*64];                   // block scores; reused as the
                                                   // tail output stage (8KB)
    threadgroup half  sp[32*72];                   // exp'd probabilities (4.6KB)
    threadgroup float sdg[256];                    // four 8x8 diag(factor) matrices
    threadgroup float srow[96];                    // m[32], l[32], factor[32]
    uint head = tg.x; uint qt = tg.y;
    uint kvh = head/group;
    uint R = n_head*hd;
    uint q0 = qt*32u;
    uint nq = min(32u, mtok - q0);
    // stage Q as half (pad rows zero), init row stats
    for (uint e = lid; e < 32u*hd; e += ts) {
        uint r = e/hd, i = e%hd;
        sq[r*SQ + i] = (r < nq) ? half(q[(ulong)(q0 + r)*(ulong)R + head*hd + i]) : half(0.0);
    }
    if (lid < 32u) { srow[lid] = -1e30; srow[32u + lid] = 0.0; }
    // per-simdgroup output tiles: sg owns hd/8 columns starting at sg*(hd/8)
    uint ncol = hd/8u;                             // columns per simdgroup
    uint nt = ncol/8u;                             // 8x8 output tiles per simdgroup (<= 4)
    simdgroup_float8x8 lo[4][4];
    for (uint rb = 0u; rb < 4u; rb++) {
        for (uint i = 0u; i < nt; i++) { lo[rb][i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0); }
    }
    uint maxseq = base_pos + q0 + nq;              // longest causal row in this tile
    for (uint c0 = 0u; c0 < maxseq; c0 += C) {
        // ---- scores: this sg's 8-position sub-block, K tile shared by 4 row-blocks
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint p0 = c0 + sg*8u;
        simdgroup_float8x8 mqk[4];
        for (uint rb = 0u; rb < 4u; rb++) { mqk[rb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0); }
        if (p0 < maxseq) {
            device const half* pk = kc + (ulong)p0*(ulong)kvdim + kvh*hd;
            simdgroup_half8x8 mq, mk;
            for (uint dd = 0u; dd < hd; dd += 8u) {
                simdgroup_load(mk, pk + dd, kvdim, 0, true);   // transposed: (d x pos)
                for (uint rb = 0u; rb < 4u; rb++) {
                    simdgroup_load(mq, sq + rb*8u*SQ + dd, SQ);
                    simdgroup_multiply_accumulate(mqk[rb], mq, mk, mqk[rb]);
                }
            }
        }
        for (uint rb = 0u; rb < 4u; rb++) { simdgroup_store(mqk[rb], ss + rb*8u*64u + sg*8u, 64u); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- streaming softmax: four rows per sg, lanes cover the 64 block columns
        for (uint rr = 0u; rr < 4u; rr++) {
            uint r = sg*4u + rr;
            uint myseq = base_pos + q0 + r + 1u;   // causal bound (valid rows only)
            bool ok0 = (r < nq) && (c0 + lane < myseq);
            bool ok1 = (r < nq) && (c0 + lane + 32u < myseq);
            float s0 = ok0 ? ss[r*64u + lane]*scale : -1e30;
            float s1 = ok1 ? ss[r*64u + lane + 32u]*scale : -1e30;
            float bm = simd_max(max(s0, s1));
            float m0 = srow[r];
            float nm = max(m0, bm);
            float pw0 = ok0 ? exp(s0 - nm) : 0.0;
            float pw1 = ok1 ? exp(s1 - nm) : 0.0;
            float rs = simd_sum(pw0 + pw1);
            if (lane == 0u) {
                float f = exp(m0 - nm);
                srow[r] = nm; srow[32u + r] = srow[32u + r]*f + rs;
                srow[64u + r] = f;                 // rescale factor, shared by all sgs
            }
            sp[r*SP + lane] = half(pw0);
            sp[r*SP + lane + 32u] = half(pw1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint rb = lid/64u, ix = lid%64u;       // 256 threads build 4 diag matrices
            sdg[lid] = (ix % 9u == 0u) ? srow[64u + rb*8u + ix/9u] : 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- rescale O by diag(factor), then O += P(32xC) * V(Cxncol), V shared
        simdgroup_float8x8 mdg;
        for (uint rb = 0u; rb < 4u; rb++) {
            simdgroup_load(mdg, sdg + rb*64u, 8u);
            for (uint i = 0u; i < nt; i++) { simdgroup_multiply(lo[rb][i], mdg, lo[rb][i]); }
        }
        uint cn = min(C, maxseq - c0);
        for (uint cc = 0u; cc < cn; cc += 8u) {
            simdgroup_half8x8 ms[4], mv;
            for (uint rb = 0u; rb < 4u; rb++) { simdgroup_load(ms[rb], sp + rb*8u*SP + cc, SP); }
            device const half* pv = vc + (ulong)(c0 + cc)*(ulong)kvdim + kvh*hd + sg*ncol;
            for (uint i = 0u; i < nt; i++) {
                simdgroup_load(mv, pv + i*8u, kvdim);
                for (uint rb = 0u; rb < 4u; rb++) {
                    simdgroup_multiply_accumulate(lo[rb][i], ms[rb], mv, lo[rb][i]);
                }
            }
        }
    }
    // ---- normalize rows by 1/l and store, one 8-row block at a time
    threadgroup float* sof = ss;                   // score buffer is dead now
    for (uint rb = 0u; rb < 4u; rb++) {
        uint rv = (nq > rb*8u) ? min(nq - rb*8u, 8u) : 0u;
        if (rv == 0u) { break; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u && lane < 8u) {
            float l = srow[32u + rb*8u + lane];
            for (uint c = 0u; c < 8u; c++) { sdg[lane*8u + c] = 0.0; }
            sdg[lane*9u] = (lane < rv && l > 0.0) ? 1.0/l : 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 mdv;
        simdgroup_load(mdv, sdg, 8u);
        if (rv == 8u) {
            // full block: simdgroup-direct store to device (row stride R)
            device float* po = out + (ulong)(q0 + rb*8u)*(ulong)R + head*hd + sg*ncol;
            for (uint i = 0u; i < nt; i++) {
                simdgroup_multiply(lo[rb][i], mdv, lo[rb][i]);
                simdgroup_store(lo[rb][i], po + i*8u, R);
            }
        } else {
            for (uint i = 0u; i < nt; i++) {
                simdgroup_multiply(lo[rb][i], mdv, lo[rb][i]);
                simdgroup_store(lo[rb][i], sof + sg*ncol + i*8u, hd);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lid; e < rv*hd; e += ts) {
                uint r = e/hd, i = e%hd;
                out[(ulong)(q0 + rb*8u + r)*(ulong)R + head*hd + i] = sof[r*hd + i];
            }
        }
    }
}

// ==================== Quest-style page-sparse decode ====================
// (arXiv:2406.10774) Per 64-position page, per kv-head, keep elementwise min/max
// of K. At decode each q-head upper-bounds every page's attention contribution
// with sum_i max(q_i*min_i, q_i*max_i), keeps the top-K pages (sinks + recent
// window always kept), and runs flash-decoding over only those pages. Exact
// softmax over the selected set; approximation is only in which pages are kept.

// Recompute min/max for pages [p0, p0+grid.x); rows scanned straight from the
// K cache (pages fully rebuilt each touch — no incremental state to corrupt).
// pmeta layout: [page][0=min,1=max][kvdim] half.

kernel void attention_m_mma_dq(device const half* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kvdim [[buffer(5)]], constant uint& base_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]], constant float& scale [[buffer(8)]], constant uint& n_head [[buffer(9)]],
    constant uint& mtok [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint ts = 256u;                          // 8 simdgroups
    const uint C = 64u;                            // KV positions per block
    const uint SP = 72u;                           // staged-P stride (C + pad)
    threadgroup float ss[32*64];                   // block scores; reused as the
                                                   // tail output stage (8KB)
    threadgroup half  sp[32*72];                   // exp'd probabilities (4.6KB)
    threadgroup float sdg[256];                    // four 8x8 diag(factor) matrices
    threadgroup float srow[96];                    // m[32], l[32], factor[32]
    uint head = tg.x; uint qt = tg.y;
    uint kvh = head/group;
    uint R = n_head*hd;
    uint q0 = qt*32u;
    uint nq = min(32u, mtok - q0);
    // Q fragments come straight from device (q_to_half wrote it as half, zeroed past
    // the token count). Staging cost occupancy rather than work — each threadgroup
    // staged only its own 32 rows — but sq was 16.9 KB of this kernel's ~31 KB, at
    // which size only two threadgroups fit per core. Without it, ~14 KB.
    device const half* qbase = q + (ulong)q0*(ulong)R + head*hd;
    if (lid < 32u) { srow[lid] = -1e30; srow[32u + lid] = 0.0; }
    // per-simdgroup output tiles: sg owns hd/8 columns starting at sg*(hd/8)
    uint ncol = hd/8u;                             // columns per simdgroup
    uint nt = ncol/8u;                             // 8x8 output tiles per simdgroup (<= 4)
    simdgroup_float8x8 lo[4][4];
    for (uint rb = 0u; rb < 4u; rb++) {
        for (uint i = 0u; i < nt; i++) { lo[rb][i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0); }
    }
    uint maxseq = base_pos + q0 + nq;              // longest causal row in this tile
    for (uint c0 = 0u; c0 < maxseq; c0 += C) {
        // ---- scores: this sg's 8-position sub-block, K tile shared by 4 row-blocks
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint p0 = c0 + sg*8u;
        simdgroup_float8x8 mqk[4];
        for (uint rb = 0u; rb < 4u; rb++) { mqk[rb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0); }
        if (p0 < maxseq) {
            device const half* pk = kc + (ulong)p0*(ulong)kvdim + kvh*hd;
            simdgroup_half8x8 mq, mk;
            for (uint dd = 0u; dd < hd; dd += 8u) {
                simdgroup_load(mk, pk + dd, kvdim, 0, true);   // transposed: (d x pos)
                for (uint rb = 0u; rb < 4u; rb++) {
                    simdgroup_load(mq, qbase + (ulong)rb*8u*(ulong)R + dd, R);
                    simdgroup_multiply_accumulate(mqk[rb], mq, mk, mqk[rb]);
                }
            }
        }
        for (uint rb = 0u; rb < 4u; rb++) { simdgroup_store(mqk[rb], ss + rb*8u*64u + sg*8u, 64u); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- streaming softmax: four rows per sg, lanes cover the 64 block columns
        for (uint rr = 0u; rr < 4u; rr++) {
            uint r = sg*4u + rr;
            uint myseq = base_pos + q0 + r + 1u;   // causal bound (valid rows only)
            bool ok0 = (r < nq) && (c0 + lane < myseq);
            bool ok1 = (r < nq) && (c0 + lane + 32u < myseq);
            float s0 = ok0 ? ss[r*64u + lane]*scale : -1e30;
            float s1 = ok1 ? ss[r*64u + lane + 32u]*scale : -1e30;
            float bm = simd_max(max(s0, s1));
            float m0 = srow[r];
            float nm = max(m0, bm);
            float pw0 = ok0 ? exp(s0 - nm) : 0.0;
            float pw1 = ok1 ? exp(s1 - nm) : 0.0;
            float rs = simd_sum(pw0 + pw1);
            if (lane == 0u) {
                float f = exp(m0 - nm);
                srow[r] = nm; srow[32u + r] = srow[32u + r]*f + rs;
                srow[64u + r] = f;                 // rescale factor, shared by all sgs
            }
            sp[r*SP + lane] = half(pw0);
            sp[r*SP + lane + 32u] = half(pw1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint rb = lid/64u, ix = lid%64u;       // 256 threads build 4 diag matrices
            sdg[lid] = (ix % 9u == 0u) ? srow[64u + rb*8u + ix/9u] : 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- rescale O by diag(factor), then O += P(32xC) * V(Cxncol), V shared
        simdgroup_float8x8 mdg;
        for (uint rb = 0u; rb < 4u; rb++) {
            simdgroup_load(mdg, sdg + rb*64u, 8u);
            for (uint i = 0u; i < nt; i++) { simdgroup_multiply(lo[rb][i], mdg, lo[rb][i]); }
        }
        uint cn = min(C, maxseq - c0);
        for (uint cc = 0u; cc < cn; cc += 8u) {
            simdgroup_half8x8 ms[4], mv;
            for (uint rb = 0u; rb < 4u; rb++) { simdgroup_load(ms[rb], sp + rb*8u*SP + cc, SP); }
            device const half* pv = vc + (ulong)(c0 + cc)*(ulong)kvdim + kvh*hd + sg*ncol;
            for (uint i = 0u; i < nt; i++) {
                simdgroup_load(mv, pv + i*8u, kvdim);
                for (uint rb = 0u; rb < 4u; rb++) {
                    simdgroup_multiply_accumulate(lo[rb][i], ms[rb], mv, lo[rb][i]);
                }
            }
        }
    }
    // ---- normalize rows by 1/l and store, one 8-row block at a time
    threadgroup float* sof = ss;                   // score buffer is dead now
    for (uint rb = 0u; rb < 4u; rb++) {
        uint rv = (nq > rb*8u) ? min(nq - rb*8u, 8u) : 0u;
        if (rv == 0u) { break; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u && lane < 8u) {
            float l = srow[32u + rb*8u + lane];
            for (uint c = 0u; c < 8u; c++) { sdg[lane*8u + c] = 0.0; }
            sdg[lane*9u] = (lane < rv && l > 0.0) ? 1.0/l : 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 mdv;
        simdgroup_load(mdv, sdg, 8u);
        if (rv == 8u) {
            // full block: simdgroup-direct store to device (row stride R)
            device float* po = out + (ulong)(q0 + rb*8u)*(ulong)R + head*hd + sg*ncol;
            for (uint i = 0u; i < nt; i++) {
                simdgroup_multiply(lo[rb][i], mdv, lo[rb][i]);
                simdgroup_store(lo[rb][i], po + i*8u, R);
            }
        } else {
            for (uint i = 0u; i < nt; i++) {
                simdgroup_multiply(lo[rb][i], mdv, lo[rb][i]);
                simdgroup_store(lo[rb][i], sof + sg*ncol + i*8u, hd);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lid; e < rv*hd; e += ts) {
                uint r = e/hd, i = e%hd;
                out[(ulong)(q0 + rb*8u + r)*(ulong)R + head*hd + i] = sof[r*hd + i];
            }
        }
    }
}

// ==================== Quest-style page-sparse decode ====================
// (arXiv:2406.10774) Per 64-position page, per kv-head, keep elementwise min/max
// of K. At decode each q-head upper-bounds every page's attention contribution
// with sum_i max(q_i*min_i, q_i*max_i), keeps the top-K pages (sinks + recent
// window always kept), and runs flash-decoding over only those pages. Exact
// softmax over the selected set; approximation is only in which pages are kept.

// Recompute min/max for pages [p0, p0+grid.x); rows scanned straight from the
// K cache (pages fully rebuilt each touch — no incremental state to corrupt).
// pmeta layout: [page][0=min,1=max][kvdim] half.

kernel void page_minmax(device const half* kc [[buffer(0)]], device half* pmeta [[buffer(1)]],
    constant uint& kvdim [[buffer(2)]], constant uint& p0 [[buffer(3)]], constant uint& endpos [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint ts [[threads_per_threadgroup]]) {
    uint page = p0 + tgid;
    uint start = page*64u;
    uint stop = min(start + 64u, endpos);
    if (start >= stop) { return; }
    for (uint i = lid; i < kvdim; i += ts) {
        float mn = 1e30, mx = -1e30;
        for (uint r = start; r < stop; r++) {
            float v = float(kc[(ulong)r*(ulong)kvdim + i]);
            mn = min(mn, v); mx = max(mx, v);
        }
        pmeta[((ulong)page*2u)*(ulong)kvdim + i] = half(mn);
        pmeta[((ulong)page*2u + 1u)*(ulong)kvdim + i] = half(mx);
    }
}

// Per q-head page scoring + iterative top-K into plist[head][MAXSEL].
// One threadgroup per head; scores live in threadgroup memory (max 4096 pages
// = 262k context). Pages 0-1 (attention sinks) and the last two pages (recent
// window) are always kept.
kernel void page_select(device const float* q [[buffer(0)]], device const half* pmeta [[buffer(1)]],
    device uint* plist [[buffer(2)]],
    constant uint& hd [[buffer(3)]], constant uint& kvdim [[buffer(4)]], constant uint& seq [[buffer(5)]],
    constant uint& group [[buffer(6)]], constant uint& ksel [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint ts = 256u;
    threadgroup float sq[512];
    threadgroup float sc[4096];
    threadgroup float red[8];
    threadgroup uint  ridx[8];
    uint kvh = head/group;
    uint npages = (seq + 63u)/64u;
    for (uint i = lid; i < hd; i += ts) { sq[i] = q[head*hd + i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pg = lid; pg < npages; pg += ts) {
        device const half* mn = pmeta + ((ulong)pg*2u)*(ulong)kvdim + kvh*hd;
        device const half* mx = mn + kvdim;
        float s = 0.0;
        for (uint i = 0u; i < hd; i++) {
            float qv = sq[i];
            s += max(qv*float(mn[i]), qv*float(mx[i]));
        }
        if (pg < 2u || pg + 2u >= npages) { s = 3.4e38; }  // sinks + recent window
        sc[pg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint k = min(ksel, npages);
    for (uint j = 0u; j < k; j++) {
        float b = -3.4e38; uint bi = 0u;
        for (uint pg = lid; pg < npages; pg += ts) {
            if (sc[pg] > b) { b = sc[pg]; bi = pg; }
        }
        for (uint off = 16u; off > 0u; off >>= 1u) {
            float ob = simd_shuffle_down(b, off);
            uint oi = simd_shuffle_down(bi, off);
            if (ob > b) { b = ob; bi = oi; }
        }
        if (lane == 0u) { red[sg] = b; ridx[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0u) {
            float bb = red[0]; uint bbi = ridx[0];
            for (uint s2 = 1u; s2 < 8u; s2++) {
                if (red[s2] > bb) { bb = red[s2]; bbi = ridx[s2]; }
            }
            plist[head*256u + j] = bbi;
            sc[bbi] = -3.4e38;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Flash-decoding over the selected pages only. Identical math to attention_part;
// global position index space [0, npsel*64) maps through plist to cache rows.
// Positions past seq (partial last page) contribute nothing.
kernel void attention_part_sparse(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* part [[buffer(3)]],
    device const uint* plist [[buffer(4)]],
    constant uint& hd [[buffer(5)]], constant uint& kvdim [[buffer(6)]], constant uint& seq [[buffer(7)]],
    constant uint& group [[buffer(8)]], constant float& scale [[buffer(9)]], constant uint& nwg [[buffer(10)]],
    constant uint& npsel [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint ts = 256u;
    threadgroup float tacc[8*512];
    threadgroup float tm[8];
    threadgroup float tl[8];
    uint head = tg.x; uint wg = tg.y;
    uint kvh = head/group;
    uint total = npsel*64u;
    uint chunk = (total + nwg - 1u) / nwg;
    uint t0 = wg * chunk;
    uint t1 = min(t0 + chunk, total);
    uint nsg = ts / 32u;
    uint nc4 = (hd + 127u)/128u;
    device const float4* qh4 = (device const float4*)(q + head*hd);
    device const uint* pl = plist + head*256u;
    float4 qr[4];
    for (uint c = 0u; c < nc4; c++) {
        uint e = c*32u + lane;
        qr[c] = (e*4u < hd) ? qh4[e] : float4(0.0);
    }
    float mi = -1e30, li = 0.0;
    float4 acc[4] = {float4(0.0), float4(0.0), float4(0.0), float4(0.0)};
    ulong kv4 = (ulong)kvdim/4u;
    ulong base4 = (ulong)(kvh*hd)/4u;
    for (uint gi = t0 + sg*4u; gi < t1; gi += nsg*4u) {
        float sv[4]; uint tj[4]; bool vj[4];
        for (uint j = 0u; j < 4u; j++) {
            uint g = gi + j;
            float s = 0.0;
            vj[j] = g < t1;
            if (vj[j]) {
                uint t = pl[g/64u]*64u + (g & 63u);
                tj[j] = t;
                vj[j] = t < seq;
                if (vj[j]) {
                    device const half4* kt4 = (device const half4*)kc + (ulong)t*kv4 + base4;
                    for (uint c = 0u; c < nc4; c++) {
                        uint e = c*32u + lane;
                        if (e*4u < hd) { s += dot(float4(kt4[e]), qr[c]); }
                    }
                }
            }
            sv[j] = simd_sum(s)*scale;
        }
        for (uint j = 0u; j < 4u; j++) {
            if (!vj[j]) { continue; }              // holes possible mid-list (partial page)
            float mn = max(mi, sv[j]);
            float corr = exp(mi - mn); float pw = exp(sv[j] - mn);
            li = li*corr + pw;
            device const half4* vt4 = (device const half4*)vc + (ulong)tj[j]*kv4 + base4;
            for (uint c = 0u; c < nc4; c++) {
                uint e = c*32u + lane;
                float4 v = (e*4u < hd) ? float4(vt4[e]) : float4(0.0);
                acc[c] = acc[c]*corr + pw*v;
            }
            mi = mn;
        }
    }
    if (lane == 0u) { tm[sg] = mi; tl[sg] = li; }
    threadgroup float4* tacc4 = (threadgroup float4*)tacc;
    for (uint c = 0u; c < nc4; c++) {
        uint e = c*32u + lane;
        if (e*4u < hd) { tacc4[(sg*hd)/4u + e] = acc[c]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = -1e30; for (uint j = 0u; j < nsg; j++) { gm = max(gm, tm[j]); }
    float gl = 0.0; for (uint j = 0u; j < nsg; j++) { gl += tl[j]*exp(tm[j] - gm); }
    device float* po = part + (ulong)(head*nwg + wg)*(ulong)(hd + 2u);
    for (uint i = lid; i < hd; i += ts) {
        float o = 0.0;
        for (uint j = 0u; j < nsg; j++) { o += tacc[j*hd + i]*exp(tm[j] - gm); }
        po[i] = o;
    }
    if (lid == 0u) { po[hd] = gm; po[hd + 1u] = gl; }
}
"#;
