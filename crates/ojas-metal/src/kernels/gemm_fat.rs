//! Fat-tile GEMM on explicit register fragment storage — the prefill-2000 kernel.
//!
//! Through MSL's opaque `simdgroup_matrix` API, holding more than ~8 accumulator
//! fragments per simdgroup spills: a 32-accumulator tile measured 35 ms/call against
//! the base kernel's 2.6. So the accumulator tile lives in explicit per-lane register
//! storage instead. With `#pragma METAL internals : enable`, a `vec<T,64>` is the
//! same register-backed storage the builtin matrix uses, and reinterpreting it as a
//! `vec<T,2>` exposes the two elements each lane owns of an 8x8 fragment. Loads and
//! stores then become plain per-lane vec2 memory ops with statically-known addresses,
//! so the compiler register-allocates dozens of fragments — to it they are just
//! vec2s. The MMA still goes through the
//! `__metal_simdgroup_matrix_8x8_multiply_accumulate` builtin that
//! `simdgroup_multiply_accumulate` lowers to.
//!
//! Each lane owns two horizontally-adjacent elements of the 8x8 — (row, col) and
//! (row, col+1) — distributed across the 32-lane simd by the hardware's fixed
//! layout, which `morton_order` reproduces from the lane id (map below shows which
//! lane owns each cell of the 8x8):
//!
//!    0  0  1  1  8  8  9  9
//!    2  2  3  3 10 10 11 11
//!    4  4  5  5 12 12 13 13
//!    6  6  7  7 14 14 15 15
//!   16 16 17 17 24 24 25 25
//!   18 18 19 19 26 26 27 27
//!   20 20 21 21 28 28 29 29
//!   22 22 23 23 30 30 31 31
//!
//! Tile config: 64 tokens x 64 rows per threadgroup, 128 threads (four simdgroups),
//! each simdgroup owning 32 output rows x its 32 tokens as a 4x4 = 16 fragment grid.
//! Four simdgroups measured faster here than two (193 vs 233 ms); see the per-kernel
//! note below.
//!
//! Compiled fallibly like the async-copy family: `#pragma METAL internals` is
//! semi-internal, so a toolchain that rejects it degrades to the staged GEMM.

/// Split-K partitions for `gemm_mm_f16_fat` at an M x N x K product: enough to bring
/// the 64x64 tile grid to about a thousand threadgroups (the target the Q4L split-K
/// path measured), at most 8, each partition at least 64 deep, and the partials
/// (`splits * m * n` floats) within `scratch_floats`. 1 means no split.
///
/// Only narrow outputs split. On an M2 Max (`examples/gemm_f16_bench.rs`, best of five)
/// splitting the N = 1024 projections measured 14-30% faster at 115 and at 731 rows,
/// while the 3072- and 5248-wide ones gained nothing or lost.
pub fn f16_fat_splits(m: u32, n: u32, k: u32, scratch_floats: u32) -> u32 {
    if n > 2048 { return 1; }
    let tiles = m.div_ceil(64) * (n / 64);
    let mut s = (1024 / tiles.max(1)).clamp(1, 8);
    while s > 1 && (k / s < 64 || (s as u64) * (m as u64) * (n as u64) > scratch_floats as u64) { s -= 1; }
    s
}

pub const GEMM_FAT_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---- fragment storage: explicit per-lane register tile ----
// Lane -> (col, row) of the 8x8 element this lane owns, read straight off the
// hardware's fixed distribution (the map in the header). Rows walk in half-steps
// with a +4 jump past lane 16; columns pick the 0/2 pair, +4 for the upper half.
METAL_FUNC static ushort2 morton_order(ushort lane_id) {
  ushort row = (lane_id / 16) * 4 + (lane_id / 2) % 4;
  ushort col = ((lane_id / 8) & 1) * 4 + (lane_id & 1) * 2;
  return ushort2(col, row);
}

#pragma METAL internals : enable
namespace metal {
  template <typename T>
  struct simdgroup_matrix_storage {
    typedef vec<T, 64> storage_type;
    storage_type t;

    METAL_FUNC thread vec<T, 2>* thread_elements() thread {
      return reinterpret_cast<thread vec<T, 2>*>(&t);
    }
    METAL_FUNC simdgroup_matrix_storage() thread = default;
    METAL_FUNC simdgroup_matrix_storage(vec<T, 2> te) thread {
      *(this->thread_elements()) = te;
    }
    // The morton part of the address is applied once via apply_offset; load/store
    // origins are then fragment corners (multiples of 8), statically known.
    METAL_FUNC static threadgroup T* apply_offset(threadgroup T *src, ushort ld, ushort2 origin) {
      return src + origin.y * ld + origin.x;
    }
    METAL_FUNC static device T* apply_offset(device T *src, uint ld, uint2 origin) {
      return src + ulong(origin.y * ld) + origin.x;
    }
    template <typename U>
    METAL_FUNC void load(const threadgroup U *src, ushort ld, ushort2 origin) {
      // even ld only (all our tiles pad to even leading dims)
      ushort address = ushort(origin.y) * ld + ushort(origin.x);
      vec<U, 2> m = *(const threadgroup vec<U, 2>*)(src + address);
      *(thread_elements()) = vec<T, 2>(m);
    }
    template <typename U>
    METAL_FUNC void load(const device U *src, uint ld, uint2 origin) {
      ulong address = ulong(origin.y) * ld + ulong(origin.x);
      vec<U, 2> m = *(const device vec<U, 2>*)(src + address);
      *(thread_elements()) = vec<T, 2>(m);
    }
    template <typename U>
    METAL_FUNC void store(device U *dst, uint ld, uint2 origin) {
      ulong address = ulong(origin.y) * ld + ulong(origin.x);
      vec<T, 2> r = *(thread_elements());
      *(device vec<U, 2>*)(dst + address) = vec<U, 2>(r);
    }
    template <typename U, typename V>
    METAL_FUNC void multiply(simdgroup_matrix_storage<U> a, simdgroup_matrix_storage<V> b, bool accumulate = true) {
      if (!accumulate) { *(thread_elements()) = vec<T, 2>(0); }
      t = __metal_simdgroup_matrix_8x8_multiply_accumulate(a.t, b.t, t, typename simdgroup_matrix_storage<T>::storage_type());
    }
  };
}
#pragma METAL internals : disable

// ---- the kernel ----
//
// C[token][row] = X[token][k] . Wt[k][row], X f32 in device, W Q4L-quantized.
//   sa: weights dequantised to half, [k][row] row-major, ld 72 (64 + 8 pad)
//   sb: activations as half,        [token][k] row-major, ld 40 (32 + 8 pad)
// K-slab 32 = one whole Q4L block per row (single scale pair per fill).
kernel void gemm_mm_q4l_fat(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    // 64x64 tile, four simdgroups, 16 fragments each (sg = 32 rows x 32 tokens).
    // Two simdgroups measured slower (233 vs 193 ms floor): the A-side is a dequant,
    // not a copy, and qkv's 512-row matrices starve a 64-thread tile. The opaque
    // simdgroup_matrix API cannot hold 16 fragments per simdgroup (35 ms/call spill);
    // custom storage keeps 32 floats/lane of accumulator in registers.
    threadgroup half sa[32*72];
    threadgroup half sb[64*40];
    const uint r0 = tgpig.y*64u;   // output rows
    const uint t0 = tgpig.x*64u;   // tokens
    uint nblk = K/32u;
    // A-fill: 128 threads, thread pair per weight row (16 k each, same block/scale).
    uint lr = tiitg/2u;
    uint il = tiitg%2u;
    device const uchar* arow = w4 + (ulong)(r0+lr)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr)*(ulong)nblk;
    // B-fill: thread pair per token (16 k each).
    device const float* xrow = x + (ulong)(t0+lr)*(ulong)K;
    bool okb = t0 + lr < M;

    ushort2 mo = morton_order(lane);
    uint row0 = (uint(sgitg) % 2u) * 32u;   // this sg's output-row half
    uint tok0 = (uint(sgitg) / 2u) * 32u;   // this sg's token half

    simdgroup_matrix_storage<float> c[4][4];
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) { c[t][r] = simdgroup_matrix_storage<float>(float2(0)); }
    }

    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A: dequant 16 k of row lr into sa[k][row] (stride-72 stores).
            half A = arow_a[lk/32u];
            half B = arow_b[lk/32u];
            device const uchar4* bp4 = (device const uchar4*)(arow + lk/2u) + il*2u;
            threadgroup half* dst = sa + il*16u*72u + lr;
#pragma clang loop unroll(full)
            for (short q = 0; q < 2; q++) {
                uchar4 b4 = bp4[q];
                dst[(q*8+0)*72] = A*half(b4.x & 0x0Fu) + B;  dst[(q*8+1)*72] = A*half(b4.x >> 4) + B;
                dst[(q*8+2)*72] = A*half(b4.y & 0x0Fu) + B;  dst[(q*8+3)*72] = A*half(b4.y >> 4) + B;
                dst[(q*8+4)*72] = A*half(b4.z & 0x0Fu) + B;  dst[(q*8+5)*72] = A*half(b4.z >> 4) + B;
                dst[(q*8+6)*72] = A*half(b4.w & 0x0Fu) + B;  dst[(q*8+7)*72] = A*half(b4.w >> 4) + B;
            }
        }
        {   // B: 16 k of token lr, f32 -> half, contiguous stores.
            device const float4* xr = (device const float4*)(xrow + lk) + il*4u;
            threadgroup half4* dst = (threadgroup half4*)(sb + lr*40u) + il*4u;
#pragma clang loop unroll(full)
            for (short q = 0; q < 4; q++) { dst[q] = okb ? half4(xr[q]) : half4(0.0); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup half* sb_m = simdgroup_matrix_storage<half>::apply_offset(sb, 40, ushort2(mo.x, mo.y));
        const threadgroup half* sa_m = simdgroup_matrix_storage<half>::apply_offset(sa, 72, ushort2(mo.x, mo.y));
#pragma clang loop unroll(full)
        for (short ko = 0; ko < 4; ko++) {
            simdgroup_matrix_storage<half> af[4];
            simdgroup_matrix_storage<half> bf[4];
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) { af[t].load(sb_m, 40, ushort2(ko*8, ushort(tok0) + t*8)); }
#pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) { bf[r].load(sa_m, 72, ushort2(ushort(row0) + r*8, ko*8)); }
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
                for (short r = 0; r < 4; r++) { c[t][r].multiply(af[t], bf[r]); }
            }
        }
    }
    device float* ybase = simdgroup_matrix_storage<float>::apply_offset(y, N, uint2(r0 + row0 + mo.x, mo.y));
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
        uint tok = t0 + tok0 + uint(t)*8u + uint(mo.y);
        if (tok >= M) { continue; }
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) {
            uint2 origin = uint2(uint(r)*8u, t0 + tok0 + uint(t)*8u);
            if (accum != 0u) {
                simdgroup_matrix_storage<float> prev;
                prev.load(ybase, N, origin);
                *(c[t][r].thread_elements()) += *(prev.thread_elements());
            }
            c[t][r].store(ybase, N, origin);
        }
    }
}






// Fat GEMM over weights read in place: y[tok][row] = x[tok][k] . W[row][k].
//
// The Q4L split-K fat kernel's structure with the weight format factored out: 64x64
// tile, four simdgroups of 16 register fragments, the weight tile double-buffered
// in threadgroup memory (one barrier per K slab), and activation fragments loaded
// straight from device, where the slab stays L2-resident across N-tiles. Each lane's
// token row is clamped to M-1, so a partial tile reads valid rows whose results the
// store guard discards, and `x` needs no row padding.
//
// `Fill` is the only per-format part: `Fill::fill(dst, w, row, K, k0)` writes the 16
// weights W[row][k0..k0+16] as half to dst[0], dst[72], ..., dst[15*72]. k0 is a
// multiple of 16, so a run never crosses a 256-weight K-quant super-block.
//
// Split-K on grid.z (`nsplit` > 1, 32-aligned partitions): plain fp32 partials at
// y + z*M*N, reduced by `splitk_accum`, which owns `accum`. It supplies threadgroups
// a short request cannot: at M=115, N=1024 the tile grid is 32 threadgroups for 38
// cores. N % 64 == 0, K % 32 == 0 (K % 256 == 0 for the K-quant fills).
template <typename Fill>
METAL_FUNC void gemm_fat_body(device const float* x, device const uchar* w, device float* y,
    uint K, uint N, uint accum, uint M, uint nsplit,
    uint3 tgpig, ushort tiitg, ushort sgitg, ushort lane, threadgroup half (*sa)[32*72]) {
    const uint r0 = tgpig.y*64u;   // output rows
    const uint t0 = tgpig.x*64u;   // tokens
    uint kper = ((K / nsplit) / 32u) * 32u;
    uint kbeg = tgpig.z * kper;
    uint kend = (tgpig.z + 1u == nsplit) ? K : kbeg + kper;
    device float* yz = y + (ulong)tgpig.z * (ulong)M * (ulong)N;
    // Weight fill: 128 threads, a thread pair per weight row, 16 k each.
    uint lr = tiitg/2u;
    uint il = tiitg%2u;

    ushort2 mo = morton_order(lane);
    uint row0 = (uint(sgitg) % 2u) * 32u;
    uint tok0 = (uint(sgitg) / 2u) * 32u;
    // Per-lane activation rows for the four token fragments, clamped into [0, M).
    device const float* xr[4];
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
        uint tok = min(t0 + tok0 + uint(t)*8u + uint(mo.y), M - 1u);
        xr[t] = x + (ulong)tok*(ulong)K + mo.x;
    }

    simdgroup_matrix_storage<float> c[4][4];
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) { c[t][r] = simdgroup_matrix_storage<float>(float2(0)); }
    }

    Fill::fill(sa[0] + il*16u*72u + lr, w, r0 + lr, K, kbeg + il*16u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint lk = kbeg; lk < kend; lk += 32u) {
        uint cur = ((lk - kbeg)/32u) & 1u;
        if (lk + 32u < kend) { Fill::fill(sa[1u-cur] + il*16u*72u + lr, w, r0 + lr, K, lk + 32u + il*16u); }
        const threadgroup half* sa_m = simdgroup_matrix_storage<half>::apply_offset(sa[cur], 72, ushort2(mo.x, mo.y));
#pragma clang loop unroll(full)
        for (short ko = 0; ko < 4; ko++) {
            simdgroup_matrix_storage<half> af[4];
            simdgroup_matrix_storage<half> bf[4];
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) {
                *(af[t].thread_elements()) = half2(*(device const float2*)(xr[t] + lk + uint(ko)*8u));
            }
#pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) { bf[r].load(sa_m, 72, ushort2(ushort(row0) + r*8, ko*8)); }
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
                for (short r = 0; r < 4; r++) { c[t][r].multiply(af[t], bf[r]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    bool add = accum != 0u && nsplit == 1u;
    device float* ybase = simdgroup_matrix_storage<float>::apply_offset(yz, N, uint2(r0 + row0 + mo.x, mo.y));
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
        uint tok = t0 + tok0 + uint(t)*8u + uint(mo.y);
        if (tok >= M) { continue; }
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) {
            uint2 origin = uint2(uint(r)*8u, t0 + tok0 + uint(t)*8u);
            if (add) {
                simdgroup_matrix_storage<float> prev;
                prev.load(ybase, N, origin);
                *(c[t][r].thread_elements()) += *(prev.thread_elements());
            }
            c[t][r].store(ybase, N, origin);
        }
    }
}

// Raw half [N][K] row-major: the encoders' projection (`projm` at f16).
struct FatFillF16 {
    static METAL_FUNC void fill(threadgroup half* dst, device const uchar* w, uint row, uint K, uint k0) {
        device const half4* ap = (device const half4*)((device const half*)w + (ulong)row*(ulong)K + k0);
        for (short q = 0; q < 4; q++) {
            half4 a = ap[q];
            dst[(q*4+0)*72] = a.x; dst[(q*4+1)*72] = a.y;
            dst[(q*4+2)*72] = a.z; dst[(q*4+3)*72] = a.w;
        }
    }
};

// GGUF Q4_K rows as stored: 144-byte super-blocks of 256 weights, {d, dmin} half,
// 12 bytes of packed 6-bit scales and mins, 128 bytes of nibbles. Sub-block j (32
// weights) is the low (j even) or high (j odd) nibble of qs[32*(j/2) ..], valued
// d*sc_j*q - dmin*m_j.
struct FatFillQ4K {
    static METAL_FUNC void fill(threadgroup half* dst, device const uchar* w, uint row, uint K, uint k0) {
        device const uchar* b = w + ((ulong)row*(ulong)(K/256u) + k0/256u)*144ul;
        uint io = k0 % 256u, j = io / 32u, l0 = io % 32u;
        device const uchar* s = b + 4u;
        uint sc, mn;
        if (j < 4u) { sc = s[j] & 63u; mn = s[j+4u] & 63u; }
        else { sc = (s[j+4u] & 0xFu) | ((s[j-4u] >> 6u) << 4u); mn = (s[j+4u] >> 4u) | ((s[j] >> 6u) << 4u); }
        float d = float(*(device const half*)b) * float(sc);
        float m = float(*(device const half*)(b + 2u)) * float(mn);
        device const uchar* q = b + 16u + 32u*(j/2u) + l0;
        uint sh = (j & 1u) * 4u;
        for (short i = 0; i < 16; i++) { dst[i*72] = half(d * float((q[i] >> sh) & 0xFu) - m); }
    }
};

// GGUF Q6_K rows as stored: 210-byte super-blocks of 256 weights, ql[128], qh[64],
// 16 int8 scales, d half. The weight at offset io sits in half h = io/128 at
// quarter q = (io%128)/32, its low nibble in ql[64h + 32(q&1) + l] (upper half
// for q >= 2) and its two high bits at 2q in qh[32h + l], scaled by
// scales[8h + l/16 + 2q].
struct FatFillQ6K {
    static METAL_FUNC void fill(threadgroup half* dst, device const uchar* w, uint row, uint K, uint k0) {
        device const uchar* b = w + ((ulong)row*(ulong)(K/256u) + k0/256u)*210ul;
        uint io = k0 % 256u, h = io / 128u, r = io % 128u, q = r / 32u, l0 = r % 32u;
        float sc = float(*(device const half*)(b + 208u)) * float(((device const char*)(b + 192u))[h*8u + l0/16u + 2u*q]);
        device const uchar* ql = b + h*64u + (q & 1u)*32u + l0;
        device const uchar* qh = b + 128u + h*32u + l0;
        uint shl = (q >= 2u) ? 4u : 0u, shh = 2u*q;
        for (short i = 0; i < 16; i++) {
            int v = int(((ql[i] >> shl) & 0xFu) | (((qh[i] >> shh) & 3u) << 4u)) - 32;
            dst[i*72] = half(sc * float(v));
        }
    }
};

#define GEMM_FAT_ENTRY(NAME, FILL) \
kernel void NAME(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]], \
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], \
    constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]], \
    constant uint& nsplit [[buffer(9)]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    ushort tiitg [[thread_index_in_threadgroup]], \
    ushort sgitg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]]) { \
    threadgroup half sa[2][32*72]; \
    gemm_fat_body<FILL>(x, w, y, K, N, accum, M, nsplit, tgpig, tiitg, sgitg, lane, sa); \
}
GEMM_FAT_ENTRY(gemm_mm_f16_fat, FatFillF16)
GEMM_FAT_ENTRY(gemm_mm_q4k_fat, FatFillQ4K)
GEMM_FAT_ENTRY(gemm_mm_q6k_fat, FatFillQ6K)
#undef GEMM_FAT_ENTRY


kernel void gemm_mm_q4l_fat8(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    // 128 tok x 64 row tile, eight simdgroups (256 threads), 16 fragments each
    // (sg = 32 rows x 32 tokens). The 2-simdgroup config measured slower here (233
    // vs 193 ms floor): the A-side is a dequant, not a copy, and qkv's 512-row
    // matrices starve a 64-thread tile. 16 fragments per simdgroup is the config the
    // opaque simdgroup_matrix API cannot express (35 ms/call spill); on custom
    // storage it holds 32 floats/lane and stays in registers.
    threadgroup half sa[32*72];
    threadgroup half sb[128*40];
    const uint r0 = tgpig.y*64u;   // output rows
    const uint t0 = tgpig.x*128u;  // tokens
    uint nblk = K/32u;
    // A-fill: 256 threads, thread QUAD per weight row (8 k each, same block/scale).
    uint lr = tiitg/4u;
    uint il = tiitg%4u;
    // B-fill: thread pair per token (16 k each), 128 tokens.
    uint tb = tiitg/2u;
    uint il2 = tiitg%2u;
    device const uchar* arow = w4 + (ulong)(r0+lr)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr)*(ulong)nblk;
    device const float* xrow = x + (ulong)(t0+tb)*(ulong)K;
    bool okb = t0 + tb < M;

    ushort2 mo = morton_order(lane);
    uint row0 = (uint(sgitg) % 2u) * 32u;   // this sg's output-row half
    uint tok0 = (uint(sgitg) / 2u) * 32u;   // this sg's token QUARTER (0..3 of 128)

    simdgroup_matrix_storage<float> c[4][4];
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) { c[t][r] = simdgroup_matrix_storage<float>(float2(0)); }
    }

    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A: dequant 8 k of row lr into sa[k][row] (stride-72 stores).
            half A = arow_a[lk/32u];
            half B = arow_b[lk/32u];
            uchar4 b4 = ((device const uchar4*)(arow + lk/2u))[il];
            threadgroup half* dst = sa + il*8u*72u + lr;
            dst[0*72] = A*half(b4.x & 0x0Fu) + B;  dst[1*72] = A*half(b4.x >> 4) + B;
            dst[2*72] = A*half(b4.y & 0x0Fu) + B;  dst[3*72] = A*half(b4.y >> 4) + B;
            dst[4*72] = A*half(b4.z & 0x0Fu) + B;  dst[5*72] = A*half(b4.z >> 4) + B;
            dst[6*72] = A*half(b4.w & 0x0Fu) + B;  dst[7*72] = A*half(b4.w >> 4) + B;
        }
        {   // B: 16 k of token tb, f32 -> half, contiguous stores.
            device const float4* xr = (device const float4*)(xrow + lk) + il2*4u;
            threadgroup half4* dst = (threadgroup half4*)(sb + tb*40u) + il2*4u;
#pragma clang loop unroll(full)
            for (short q = 0; q < 4; q++) { dst[q] = okb ? half4(xr[q]) : half4(0.0); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup half* sb_m = simdgroup_matrix_storage<half>::apply_offset(sb, 40, ushort2(mo.x, mo.y));
        const threadgroup half* sa_m = simdgroup_matrix_storage<half>::apply_offset(sa, 72, ushort2(mo.x, mo.y));
#pragma clang loop unroll(full)
        for (short ko = 0; ko < 4; ko++) {
            simdgroup_matrix_storage<half> af[4];
            simdgroup_matrix_storage<half> bf[4];
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) { af[t].load(sb_m, 40, ushort2(ko*8, ushort(tok0) + t*8)); }
#pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) { bf[r].load(sa_m, 72, ushort2(ushort(row0) + r*8, ko*8)); }
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
                for (short r = 0; r < 4; r++) { c[t][r].multiply(af[t], bf[r]); }
            }
        }
    }
    device float* ybase = simdgroup_matrix_storage<float>::apply_offset(y, N, uint2(r0 + row0 + mo.x, mo.y));
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
        uint tok = t0 + tok0 + uint(t)*8u + uint(mo.y);
        if (tok >= M) { continue; }
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) {
            uint2 origin = uint2(uint(r)*8u, t0 + tok0 + uint(t)*8u);
            if (accum != 0u) {
                simdgroup_matrix_storage<float> prev;
                prev.load(ybase, N, origin);
                *(c[t][r].thread_elements()) += *(prev.thread_elements());
            }
            c[t][r].store(ybase, N, origin);
        }
    }
}



kernel void gemm_mm_q4l_fatx(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    // 64x64 tile, four simdgroups, 16 fragments each (sg = 32 rows x 32 tokens).
    // Two simdgroups measured slower (233 vs 193 ms floor): the A-side is a dequant,
    // not a copy, and qkv's 512-row matrices starve a 64-thread tile. The opaque
    // simdgroup_matrix API cannot hold 16 fragments per simdgroup (35 ms/call spill);
    // custom storage keeps 32 floats/lane of accumulator in registers.
    threadgroup half sa[32*72];   // no sb: activation fragments load straight from device
    const uint r0 = tgpig.y*64u;   // output rows
    const uint t0 = tgpig.x*64u;   // tokens
    uint nblk = K/32u;
    // A-fill: 128 threads, thread pair per weight row (16 k each, same block/scale).
    uint lr = tiitg/2u;
    uint il = tiitg%2u;
    device const uchar* arow = w4 + (ulong)(r0+lr)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr)*(ulong)nblk;

    ushort2 mo = morton_order(lane);
    // Activation fragments come straight from device: the x slab is L2-resident and
    // shared by every N-tile, so threadgroup staging of it (fill + f32->half convert
    // + a barrier participant + 5 KB of occupancy) bought nothing. Tokens past M read
    // in-buffer garbage whose fragments the store guard discards.
    device const float* xb_m = simdgroup_matrix_storage<float>::apply_offset((device float*)x, K, uint2(mo.x, mo.y));
    uint row0 = (uint(sgitg) % 2u) * 32u;   // this sg's output-row half
    uint tok0 = (uint(sgitg) / 2u) * 32u;   // this sg's token half

    simdgroup_matrix_storage<float> c[4][4];
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) { c[t][r] = simdgroup_matrix_storage<float>(float2(0)); }
    }

    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A: dequant 16 k of row lr into sa[k][row] (stride-72 stores).
            half A = arow_a[lk/32u];
            half B = arow_b[lk/32u];
            device const uchar4* bp4 = (device const uchar4*)(arow + lk/2u) + il*2u;
            threadgroup half* dst = sa + il*16u*72u + lr;
#pragma clang loop unroll(full)
            for (short q = 0; q < 2; q++) {
                uchar4 b4 = bp4[q];
                dst[(q*8+0)*72] = A*half(b4.x & 0x0Fu) + B;  dst[(q*8+1)*72] = A*half(b4.x >> 4) + B;
                dst[(q*8+2)*72] = A*half(b4.y & 0x0Fu) + B;  dst[(q*8+3)*72] = A*half(b4.y >> 4) + B;
                dst[(q*8+4)*72] = A*half(b4.z & 0x0Fu) + B;  dst[(q*8+5)*72] = A*half(b4.z >> 4) + B;
                dst[(q*8+6)*72] = A*half(b4.w & 0x0Fu) + B;  dst[(q*8+7)*72] = A*half(b4.w >> 4) + B;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup half* sa_m = simdgroup_matrix_storage<half>::apply_offset(sa, 72, ushort2(mo.x, mo.y));
#pragma clang loop unroll(full)
        for (short ko = 0; ko < 4; ko++) {
            simdgroup_matrix_storage<half> af[4];
            simdgroup_matrix_storage<half> bf[4];
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) { af[t].load(xb_m, K, uint2(lk + uint(ko)*8u, t0 + tok0 + uint(t)*8u)); }
#pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) { bf[r].load(sa_m, 72, ushort2(ushort(row0) + r*8, ko*8)); }
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
                for (short r = 0; r < 4; r++) { c[t][r].multiply(af[t], bf[r]); }
            }
        }
    }
    device float* ybase = simdgroup_matrix_storage<float>::apply_offset(y, N, uint2(r0 + row0 + mo.x, mo.y));
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
        uint tok = t0 + tok0 + uint(t)*8u + uint(mo.y);
        if (tok >= M) { continue; }
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) {
            uint2 origin = uint2(uint(r)*8u, t0 + tok0 + uint(t)*8u);
            if (accum != 0u) {
                simdgroup_matrix_storage<float> prev;
                prev.load(ybase, N, origin);
                *(c[t][r].thread_elements()) += *(prev.thread_elements());
            }
            c[t][r].store(ybase, N, origin);
        }
    }
}



// Double-buffered fat GEMM: gemm_mm_q4l_fatx with a two-slab A tile, so the fill
// targets the buffer nobody is reading and one of the two per-K-slab barriers
// disappears. Bit-identical to fatx, +1% end-to-end. Full rationale inline below.
kernel void gemm_mm_q4l_fatx2(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    // 64x64 tile, four simdgroups, 16 fragments each (sg = 32 rows x 32 tokens).
    // Two simdgroups measured slower (233 vs 193 ms floor): the A-side is a dequant,
    // not a copy, and qkv's 512-row matrices starve a 64-thread tile. The opaque
    // simdgroup_matrix API cannot hold 16 fragments per simdgroup (35 ms/call spill);
    // custom storage keeps 32 floats/lane of accumulator in registers.
    // Double-buffered A tile: one fewer threadgroup barrier per K-slab (fatx pays
    // two, so 128 barriers at K=2048), because the fill targets the slab nobody is
    // reading. Costs 4.6 KB more threadgroup memory, which fatx has spare.
    // Double-buffering into registers instead lost (201 vs 193.6 ms): registers are
    // this kernel's scarce resource, threadgroup memory is not.
    threadgroup half sa[2][32*72];
    const uint r0 = tgpig.y*64u;   // output rows
    const uint t0 = tgpig.x*64u;   // tokens
    uint nblk = K/32u;
    // A-fill: 128 threads, thread pair per weight row (16 k each, same block/scale).
    uint lr = tiitg/2u;
    uint il = tiitg%2u;
    device const uchar* arow = w4 + (ulong)(r0+lr)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr)*(ulong)nblk;

    ushort2 mo = morton_order(lane);
    // Activation fragments come straight from device: the x slab is L2-resident and
    // shared by every N-tile, so threadgroup staging of it (fill + f32->half convert
    // + a barrier participant + 5 KB of occupancy) bought nothing. Tokens past M read
    // in-buffer garbage whose fragments the store guard discards.
    device const float* xb_m = simdgroup_matrix_storage<float>::apply_offset((device float*)x, K, uint2(mo.x, mo.y));
    uint row0 = (uint(sgitg) % 2u) * 32u;   // this sg's output-row half
    uint tok0 = (uint(sgitg) / 2u) * 32u;   // this sg's token half

    simdgroup_matrix_storage<float> c[4][4];
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) { c[t][r] = simdgroup_matrix_storage<float>(float2(0)); }
    }

#define FATX2_FILL(BUF, LK) { \
            half A = arow_a[(LK)/32u]; \
            half B = arow_b[(LK)/32u]; \
            device const uchar4* bp4 = (device const uchar4*)(arow + (LK)/2u) + il*2u; \
            threadgroup half* dst = (BUF) + il*16u*72u + lr; \
            for (short q = 0; q < 2; q++) { \
                uchar4 b4 = bp4[q]; \
                dst[(q*8+0)*72] = A*half(b4.x & 0x0Fu) + B;  dst[(q*8+1)*72] = A*half(b4.x >> 4) + B; \
                dst[(q*8+2)*72] = A*half(b4.y & 0x0Fu) + B;  dst[(q*8+3)*72] = A*half(b4.y >> 4) + B; \
                dst[(q*8+4)*72] = A*half(b4.z & 0x0Fu) + B;  dst[(q*8+5)*72] = A*half(b4.z >> 4) + B; \
                dst[(q*8+6)*72] = A*half(b4.w & 0x0Fu) + B;  dst[(q*8+7)*72] = A*half(b4.w >> 4) + B; } }

    FATX2_FILL(sa[0], 0u)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint lk = 0u; lk < K; lk += 32u) {
        uint cur = (lk/32u) & 1u;
        if (lk + 32u < K) { FATX2_FILL(sa[1u-cur], lk + 32u) }
        {
        }
        const threadgroup half* sa_m = simdgroup_matrix_storage<half>::apply_offset(sa[cur], 72, ushort2(mo.x, mo.y));
#pragma clang loop unroll(full)
        for (short ko = 0; ko < 4; ko++) {
            simdgroup_matrix_storage<half> af[4];
            simdgroup_matrix_storage<half> bf[4];
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) { af[t].load(xb_m, K, uint2(lk + uint(ko)*8u, t0 + tok0 + uint(t)*8u)); }
#pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) { bf[r].load(sa_m, 72, ushort2(ushort(row0) + r*8, ko*8)); }
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
                for (short r = 0; r < 4; r++) { c[t][r].multiply(af[t], bf[r]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#undef FATX2_FILL
    device float* ybase = simdgroup_matrix_storage<float>::apply_offset(y, N, uint2(r0 + row0 + mo.x, mo.y));
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
        uint tok = t0 + tok0 + uint(t)*8u + uint(mo.y);
        if (tok >= M) { continue; }
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) {
            uint2 origin = uint2(uint(r)*8u, t0 + tok0 + uint(t)*8u);
            if (accum != 0u) {
                simdgroup_matrix_storage<float> prev;
                prev.load(ybase, N, origin);
                *(c[t][r].thread_elements()) += *(prev.thread_elements());
            }
            c[t][r].store(ybase, N, origin);
        }
    }
}



// SwiGLU up-GEMM on the double-buffered fat tile: the `up` projection with the
// silu(gate)*up (or gelu, or identity) epilogue folded into the store, so the
// elementwise pass needs no dispatch of its own. `gate` is precomputed and read
// by explicit address in the epilogue. Full rationale inline below.
kernel void gemm_mm_q4l_fatx2_silu(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    device const float* gate [[buffer(9)]], constant uint& act [[buffer(10)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    // 64x64 tile, four simdgroups, 16 fragments each (sg = 32 rows x 32 tokens).
    // Two simdgroups measured slower (233 vs 193 ms floor): the A-side is a dequant,
    // not a copy, and qkv's 512-row matrices starve a 64-thread tile. The opaque
    // simdgroup_matrix API cannot hold 16 fragments per simdgroup (35 ms/call spill);
    // custom storage keeps 32 floats/lane of accumulator in registers.
    // Double-buffered A tile: one fewer threadgroup barrier per K-slab (fatx pays
    // two, so 128 barriers at K=2048), because the fill targets the slab nobody is
    // reading. Costs 4.6 KB more threadgroup memory, which fatx has spare.
    // Double-buffering into registers instead lost (201 vs 193.6 ms): registers are
    // this kernel's scarce resource, threadgroup memory is not.
    threadgroup half sa[2][32*72];
    const uint r0 = tgpig.y*64u;   // output rows
    const uint t0 = tgpig.x*64u;   // tokens
    uint nblk = K/32u;
    // A-fill: 128 threads, thread pair per weight row (16 k each, same block/scale).
    uint lr = tiitg/2u;
    uint il = tiitg%2u;
    device const uchar* arow = w4 + (ulong)(r0+lr)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr)*(ulong)nblk;

    ushort2 mo = morton_order(lane);
    // Activation fragments come straight from device: the x slab is L2-resident and
    // shared by every N-tile, so threadgroup staging of it (fill + f32->half convert
    // + a barrier participant + 5 KB of occupancy) bought nothing. Tokens past M read
    // in-buffer garbage whose fragments the store guard discards.
    device const float* xb_m = simdgroup_matrix_storage<float>::apply_offset((device float*)x, K, uint2(mo.x, mo.y));
    uint row0 = (uint(sgitg) % 2u) * 32u;   // this sg's output-row half
    uint tok0 = (uint(sgitg) / 2u) * 32u;   // this sg's token half

    simdgroup_matrix_storage<float> c[4][4];
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) { c[t][r] = simdgroup_matrix_storage<float>(float2(0)); }
    }

#define FATX2_FILL(BUF, LK) { \
            half A = arow_a[(LK)/32u]; \
            half B = arow_b[(LK)/32u]; \
            device const uchar4* bp4 = (device const uchar4*)(arow + (LK)/2u) + il*2u; \
            threadgroup half* dst = (BUF) + il*16u*72u + lr; \
            for (short q = 0; q < 2; q++) { \
                uchar4 b4 = bp4[q]; \
                dst[(q*8+0)*72] = A*half(b4.x & 0x0Fu) + B;  dst[(q*8+1)*72] = A*half(b4.x >> 4) + B; \
                dst[(q*8+2)*72] = A*half(b4.y & 0x0Fu) + B;  dst[(q*8+3)*72] = A*half(b4.y >> 4) + B; \
                dst[(q*8+4)*72] = A*half(b4.z & 0x0Fu) + B;  dst[(q*8+5)*72] = A*half(b4.z >> 4) + B; \
                dst[(q*8+6)*72] = A*half(b4.w & 0x0Fu) + B;  dst[(q*8+7)*72] = A*half(b4.w >> 4) + B; } }

    FATX2_FILL(sa[0], 0u)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint lk = 0u; lk < K; lk += 32u) {
        uint cur = (lk/32u) & 1u;
        if (lk + 32u < K) { FATX2_FILL(sa[1u-cur], lk + 32u) }
        {
        }
        const threadgroup half* sa_m = simdgroup_matrix_storage<half>::apply_offset(sa[cur], 72, ushort2(mo.x, mo.y));
#pragma clang loop unroll(full)
        for (short ko = 0; ko < 4; ko++) {
            simdgroup_matrix_storage<half> af[4];
            simdgroup_matrix_storage<half> bf[4];
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) { af[t].load(xb_m, K, uint2(lk + uint(ko)*8u, t0 + tok0 + uint(t)*8u)); }
#pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) { bf[r].load(sa_m, 72, ushort2(ushort(row0) + r*8, ko*8)); }
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
                for (short r = 0; r < 4; r++) { c[t][r].multiply(af[t], bf[r]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#undef FATX2_FILL
    device float* ybase = simdgroup_matrix_storage<float>::apply_offset(y, N, uint2(r0 + row0 + mo.x, mo.y));
    device const float* gbase = simdgroup_matrix_storage<float>::apply_offset((device float*)gate, N, uint2(r0 + row0 + mo.x, mo.y));
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
        uint tok = t0 + tok0 + uint(t)*8u + uint(mo.y);
        if (tok >= M) { continue; }
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) {
            uint2 origin = uint2(uint(r)*8u, t0 + tok0 + uint(t)*8u);
            // Fused SwiGLU epilogue: this is the `up` GEMM and `gate` is the
            // matching output of the one before it, so the elementwise
            // silu(gate)*up happens here instead of in its own dispatch. That
            // dispatch cost 5.3 ms of a 344 ms prefill on the critical path
            // (measured by removing it from the real concurrent pass) against 1.8
            // ms measured serially: it sits between two barriers and it re-reads
            // `up` and re-writes `act`, 22.5 MB per layer this version never moves.
            //
            // `gate` is read by explicit address rather than a fragment load: the
            // lane's two accumulator elements are adjacent N columns of one token
            // row, spelled out here rather than relying on the fragment load
            // reproducing the store's addressing.
            uint grow = t0 + tok0 + uint(t)*8u + uint(mo.y);
            uint gcol = r0 + row0 + uint(r)*8u + uint(mo.x);
            device const float* gp = gate + (ulong)grow*(ulong)N + (ulong)gcol;
            float2 gv = float2(gp[0], gp[1]);
            float2 cv = *(c[t][r].thread_elements());
            // Same formulation as the silu_mul kernel this replaces, fast-math
            // ops included. `precise::exp` is more accurate and therefore wrong
            // here: it moved the logits digest 0.20% and made a fused-vs-split A/B
            // unreadable. A fusion must change where the arithmetic happens, not
            // the arithmetic.
            float2 sg;
            if (act == 2u) {                      // BISECT: identity epilogue
                sg = float2(1.0);
            } else if (act == 1u) {               // GeLU tanh-approx (Gemma)
                float2 inner = 0.7978845608f*(gv + 0.044715f*gv*gv*gv);
                inner = clamp(inner, -30.0f, 30.0f);
                sg = float2(0.5f*gv.x*(1.0f + tanh(inner.x)), 0.5f*gv.y*(1.0f + tanh(inner.y)));
            } else {                              // SiLU
                // Componentwise, not float2: under fast-math the vector exp lowers
                // to a different approximation than the scalar one, and silu_mul
                // computes this per element. exp(float2) moved the logits checksum
                // 0.20% for algebraically identical arithmetic; a bisect (identity
                // epilogue into `up`, silu_mul left in place) showed the GEMM half
                // was already bit-exact, so the drift was this one call.
                sg = float2(gv.x/(1.0f + exp(-gv.x)), gv.y/(1.0f + exp(-gv.y)));
            }
            *(c[t][r].thread_elements()) = sg*cv;
            c[t][r].store(ybase, N, origin);
            (void)accum;
        }
    }
}



// Fat tile + split-K, for ffn_down.
//
// ffn_down (K=11008, N=2048, accum) runs at 8.27 TFLOP/s against ffn_gu's 9.20 and
// fails the fat gate `n >= 4096 && !accum` on both counts. The fat tile alone loses
// on this shape (1.647 vs 1.570 ms/call, and 55.8 vs 50.3 ms without split-K); the
// two mechanisms are orthogonal, split-K supplying the threadgroups the shape cannot
// expose and the fat tile the arithmetic intensity per staged fragment, so only the
// pair can win here.

kernel void gemm_mm_q4l_skfat(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    constant uint& nsplit [[buffer(9)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    // 64x64 tile, four simdgroups, 16 fragments each (sg = 32 rows x 32 tokens).
    // Two simdgroups measured slower (233 vs 193 ms floor): the A-side is a dequant,
    // not a copy, and qkv's 512-row matrices starve a 64-thread tile. The opaque
    // simdgroup_matrix API cannot hold 16 fragments per simdgroup (35 ms/call spill);
    // custom storage keeps 32 floats/lane of accumulator in registers.
    // Double-buffered A tile: one fewer threadgroup barrier per K-slab (fatx pays
    // two, so 128 barriers at K=2048), because the fill targets the slab nobody is
    // reading. Costs 4.6 KB more threadgroup memory, which fatx has spare.
    // Double-buffering into registers instead lost (201 vs 193.6 ms): registers are
    // this kernel's scarce resource, threadgroup memory is not.
    threadgroup half sa[2][32*72];
    const uint r0 = tgpig.y*64u;   // output rows
    const uint t0 = tgpig.x*64u;   // tokens
    // Split-K on grid.z, same contract as gemm_mm_q4l_sk: 32-aligned partitions
    // so a Q4L block never straddles one, plain fp32 partials at y + z*M*N, and
    // `accum` is ignored here because splitk_accum owns the residual add.
    uint kper = ((K / nsplit) / 32u) * 32u;
    uint kbeg = tgpig.z * kper;
    uint kend = (tgpig.z + 1u == nsplit) ? K : kbeg + kper;
    device float* yz = y + (ulong)tgpig.z * (ulong)M * (ulong)N;
    uint nblk = K/32u;
    // A-fill: 128 threads, thread pair per weight row (16 k each, same block/scale).
    uint lr = tiitg/2u;
    uint il = tiitg%2u;
    device const uchar* arow = w4 + (ulong)(r0+lr)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr)*(ulong)nblk;

    ushort2 mo = morton_order(lane);
    // Activation fragments come straight from device: the x slab is L2-resident and
    // shared by every N-tile, so threadgroup staging of it (fill + f32->half convert
    // + a barrier participant + 5 KB of occupancy) bought nothing. Tokens past M read
    // in-buffer garbage whose fragments the store guard discards.
    device const float* xb_m = simdgroup_matrix_storage<float>::apply_offset((device float*)x, K, uint2(mo.x, mo.y));
    uint row0 = (uint(sgitg) % 2u) * 32u;   // this sg's output-row half
    uint tok0 = (uint(sgitg) / 2u) * 32u;   // this sg's token half

    simdgroup_matrix_storage<float> c[4][4];
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) { c[t][r] = simdgroup_matrix_storage<float>(float2(0)); }
    }

#define SKFAT_FILL(BUF, LK) { \
            half A = arow_a[(LK)/32u]; \
            half B = arow_b[(LK)/32u]; \
            device const uchar4* bp4 = (device const uchar4*)(arow + (LK)/2u) + il*2u; \
            threadgroup half* dst = (BUF) + il*16u*72u + lr; \
            for (short q = 0; q < 2; q++) { \
                uchar4 b4 = bp4[q]; \
                dst[(q*8+0)*72] = A*half(b4.x & 0x0Fu) + B;  dst[(q*8+1)*72] = A*half(b4.x >> 4) + B; \
                dst[(q*8+2)*72] = A*half(b4.y & 0x0Fu) + B;  dst[(q*8+3)*72] = A*half(b4.y >> 4) + B; \
                dst[(q*8+4)*72] = A*half(b4.z & 0x0Fu) + B;  dst[(q*8+5)*72] = A*half(b4.z >> 4) + B; \
                dst[(q*8+6)*72] = A*half(b4.w & 0x0Fu) + B;  dst[(q*8+7)*72] = A*half(b4.w >> 4) + B; } }

    SKFAT_FILL(sa[0], kbeg)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint lk = kbeg; lk < kend; lk += 32u) {
        uint cur = ((lk - kbeg)/32u) & 1u;
        if (lk + 32u < kend) { SKFAT_FILL(sa[1u-cur], lk + 32u) }
        {
        }
        const threadgroup half* sa_m = simdgroup_matrix_storage<half>::apply_offset(sa[cur], 72, ushort2(mo.x, mo.y));
#pragma clang loop unroll(full)
        for (short ko = 0; ko < 4; ko++) {
            simdgroup_matrix_storage<half> af[4];
            simdgroup_matrix_storage<half> bf[4];
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) { af[t].load(xb_m, K, uint2(lk + uint(ko)*8u, t0 + tok0 + uint(t)*8u)); }
#pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) { bf[r].load(sa_m, 72, ushort2(ushort(row0) + r*8, ko*8)); }
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
                for (short r = 0; r < 4; r++) { c[t][r].multiply(af[t], bf[r]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#undef SKFAT_FILL
    device float* ybase = simdgroup_matrix_storage<float>::apply_offset(yz, N, uint2(r0 + row0 + mo.x, mo.y));
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
        uint tok = t0 + tok0 + uint(t)*8u + uint(mo.y);
        if (tok >= M) { continue; }
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) {
            uint2 origin = uint2(uint(r)*8u, t0 + tok0 + uint(t)*8u);
            c[t][r].store(ybase, N, origin);   // partial; splitk_accum reduces
            (void)accum;
        }
    }
}



// f16-accumulator twin of gemm_mm_q4l_fatx: 16 fragments of vec<half,64> is 16
// registers per lane against 32, and register pressure decides every experiment in
// this file. The risk is precision — K=2048 of f16 accumulation — so it ships only if
// the per-layer parity gate against HF holds, not merely if decode stays
// token-identical.


kernel void gemm_mm_q4l_fatxh(device const float* x [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    // 64x64 tile, four simdgroups, 16 fragments each (sg = 32 rows x 32 tokens).
    // Two simdgroups measured slower (233 vs 193 ms floor): the A-side is a dequant,
    // not a copy, and qkv's 512-row matrices starve a 64-thread tile. The opaque
    // simdgroup_matrix API cannot hold 16 fragments per simdgroup (35 ms/call spill);
    // custom storage keeps 32 floats/lane of accumulator in registers.
    threadgroup half sa[32*72];   // no sb: activation fragments load straight from device
    const uint r0 = tgpig.y*64u;   // output rows
    const uint t0 = tgpig.x*64u;   // tokens
    uint nblk = K/32u;
    // A-fill: 128 threads, thread pair per weight row (16 k each, same block/scale).
    uint lr = tiitg/2u;
    uint il = tiitg%2u;
    device const uchar* arow = w4 + (ulong)(r0+lr)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr)*(ulong)nblk;

    ushort2 mo = morton_order(lane);
    // Activation fragments come straight from device: the x slab is L2-resident and
    // shared by every N-tile, so threadgroup staging of it (fill + f32->half convert
    // + a barrier participant + 5 KB of occupancy) bought nothing. Tokens past M read
    // in-buffer garbage whose fragments the store guard discards.
    device const float* xb_m = simdgroup_matrix_storage<float>::apply_offset((device float*)x, K, uint2(mo.x, mo.y));
    uint row0 = (uint(sgitg) % 2u) * 32u;   // this sg's output-row half
    uint tok0 = (uint(sgitg) / 2u) * 32u;   // this sg's token half

    simdgroup_matrix_storage<half> c[4][4];
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) { c[t][r] = simdgroup_matrix_storage<half>(half2(0)); }
    }

    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // A: dequant 16 k of row lr into sa[k][row] (stride-72 stores).
            half A = arow_a[lk/32u];
            half B = arow_b[lk/32u];
            device const uchar4* bp4 = (device const uchar4*)(arow + lk/2u) + il*2u;
            threadgroup half* dst = sa + il*16u*72u + lr;
#pragma clang loop unroll(full)
            for (short q = 0; q < 2; q++) {
                uchar4 b4 = bp4[q];
                dst[(q*8+0)*72] = A*half(b4.x & 0x0Fu) + B;  dst[(q*8+1)*72] = A*half(b4.x >> 4) + B;
                dst[(q*8+2)*72] = A*half(b4.y & 0x0Fu) + B;  dst[(q*8+3)*72] = A*half(b4.y >> 4) + B;
                dst[(q*8+4)*72] = A*half(b4.z & 0x0Fu) + B;  dst[(q*8+5)*72] = A*half(b4.z >> 4) + B;
                dst[(q*8+6)*72] = A*half(b4.w & 0x0Fu) + B;  dst[(q*8+7)*72] = A*half(b4.w >> 4) + B;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup half* sa_m = simdgroup_matrix_storage<half>::apply_offset(sa, 72, ushort2(mo.x, mo.y));
#pragma clang loop unroll(full)
        for (short ko = 0; ko < 4; ko++) {
            simdgroup_matrix_storage<half> af[4];
            simdgroup_matrix_storage<half> bf[4];
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) { af[t].load(xb_m, K, uint2(lk + uint(ko)*8u, t0 + tok0 + uint(t)*8u)); }
#pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) { bf[r].load(sa_m, 72, ushort2(ushort(row0) + r*8, ko*8)); }
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
                for (short r = 0; r < 4; r++) { c[t][r].multiply(af[t], bf[r]); }
            }
        }
    }
    device float* ybase = simdgroup_matrix_storage<float>::apply_offset(y, N, uint2(r0 + row0 + mo.x, mo.y));
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
        uint tok = t0 + tok0 + uint(t)*8u + uint(mo.y);
        if (tok >= M) { continue; }
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) {
            uint2 origin = uint2(uint(r)*8u, t0 + tok0 + uint(t)*8u);
            if (accum != 0u) {
                simdgroup_matrix_storage<float> prev;
                prev.load(ybase, N, origin);
                float2 v = float2(*(c[t][r].thread_elements())) + *(prev.thread_elements());
                simdgroup_matrix_storage<float> o(v);
                o.store(ybase, N, origin);
            } else {
                simdgroup_matrix_storage<float> o(float2(*(c[t][r].thread_elements())));
                o.store(ybase, N, origin);
            }
        }
    }
}



// Fused gate+up fat GEMM: both FFN projections in one dispatch, activation fragments
// loaded once and multiplied against both weight tiles.
//
// The FFN's gate and up matmuls read the same activations. Two separate calls load 4
// activation + 4 weight fragments per simdgroup per K-slab, twice; fused it is 4 + 8,
// a quarter of the fragment loads gone, plus one dispatch and one set of barriers
// instead of two. 32 accumulator fragments per simdgroup, which custom storage holds
// without spilling.
kernel void ffn_gu_fat(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* yg [[buffer(3)]], device float* yu [[buffer(4)]],
    constant uint& K [[buffer(5)]], constant uint& N [[buffer(6)]],
    device const half* qag [[buffer(7)]], device const half* qbg [[buffer(8)]],
    device const half* qau [[buffer(9)]], device const half* qbu [[buffer(10)]],
    constant uint& M [[buffer(11)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half sag[32*72];
    threadgroup half sau[32*72];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*64u;
    uint nblk = K/32u;
    uint lr = tiitg/2u;
    uint il = tiitg%2u;
    ulong roff = (ulong)(r0+lr)*(ulong)(K/2u);
    ulong soff = (ulong)(r0+lr)*(ulong)nblk;
    ushort2 mo = morton_order(lane);
    uint row0 = (uint(sgitg) % 2u) * 32u;
    uint tok0 = (uint(sgitg) / 2u) * 32u;
    device const float* xb_m = simdgroup_matrix_storage<float>::apply_offset((device float*)x, K, uint2(mo.x, mo.y));

    simdgroup_matrix_storage<float> cg[4][4];
    simdgroup_matrix_storage<float> cu[4][4];
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) {
            cg[t][r] = simdgroup_matrix_storage<float>(float2(0));
            cu[t][r] = simdgroup_matrix_storage<float>(float2(0));
        }
    }

    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // dequant 16 k of row lr for BOTH matrices
            half Ag = qag[soff + lk/32u], Bg = qbg[soff + lk/32u];
            half Au = qau[soff + lk/32u], Bu = qbu[soff + lk/32u];
            device const uchar4* gp = (device const uchar4*)(wg + roff + lk/2u) + il*2u;
            device const uchar4* up = (device const uchar4*)(wu + roff + lk/2u) + il*2u;
            threadgroup half* dg = sag + il*16u*72u + lr;
            threadgroup half* du = sau + il*16u*72u + lr;
#pragma clang loop unroll(full)
            for (short q = 0; q < 2; q++) {
                uchar4 g4 = gp[q], u4 = up[q];
                dg[(q*8+0)*72] = Ag*half(g4.x & 0x0Fu) + Bg;  dg[(q*8+1)*72] = Ag*half(g4.x >> 4) + Bg;
                dg[(q*8+2)*72] = Ag*half(g4.y & 0x0Fu) + Bg;  dg[(q*8+3)*72] = Ag*half(g4.y >> 4) + Bg;
                dg[(q*8+4)*72] = Ag*half(g4.z & 0x0Fu) + Bg;  dg[(q*8+5)*72] = Ag*half(g4.z >> 4) + Bg;
                dg[(q*8+6)*72] = Ag*half(g4.w & 0x0Fu) + Bg;  dg[(q*8+7)*72] = Ag*half(g4.w >> 4) + Bg;
                du[(q*8+0)*72] = Au*half(u4.x & 0x0Fu) + Bu;  du[(q*8+1)*72] = Au*half(u4.x >> 4) + Bu;
                du[(q*8+2)*72] = Au*half(u4.y & 0x0Fu) + Bu;  du[(q*8+3)*72] = Au*half(u4.y >> 4) + Bu;
                du[(q*8+4)*72] = Au*half(u4.z & 0x0Fu) + Bu;  du[(q*8+5)*72] = Au*half(u4.z >> 4) + Bu;
                du[(q*8+6)*72] = Au*half(u4.w & 0x0Fu) + Bu;  du[(q*8+7)*72] = Au*half(u4.w >> 4) + Bu;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup half* sg_m = simdgroup_matrix_storage<half>::apply_offset(sag, 72, ushort2(mo.x, mo.y));
        const threadgroup half* su_m = simdgroup_matrix_storage<half>::apply_offset(sau, 72, ushort2(mo.x, mo.y));
#pragma clang loop unroll(full)
        for (short ko = 0; ko < 4; ko++) {
            simdgroup_matrix_storage<half> af[4];
            simdgroup_matrix_storage<half> bg[4];
            simdgroup_matrix_storage<half> bu[4];
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) { af[t].load(xb_m, K, uint2(lk + uint(ko)*8u, t0 + tok0 + uint(t)*8u)); }
#pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) {
                bg[r].load(sg_m, 72, ushort2(ushort(row0) + r*8, ko*8));
                bu[r].load(su_m, 72, ushort2(ushort(row0) + r*8, ko*8));
            }
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) {
#pragma clang loop unroll(full)
                for (short r = 0; r < 4; r++) {
                    cg[t][r].multiply(af[t], bg[r]);
                    cu[t][r].multiply(af[t], bu[r]);
                }
            }
        }
    }
    device float* ygb = simdgroup_matrix_storage<float>::apply_offset(yg, N, uint2(r0 + row0 + mo.x, mo.y));
    device float* yub = simdgroup_matrix_storage<float>::apply_offset(yu, N, uint2(r0 + row0 + mo.x, mo.y));
#pragma clang loop unroll(full)
    for (short t = 0; t < 4; t++) {
        uint tok = t0 + tok0 + uint(t)*8u + uint(mo.y);
        if (tok >= M) { continue; }
#pragma clang loop unroll(full)
        for (short r = 0; r < 4; r++) {
            uint2 origin = uint2(uint(r)*8u, t0 + tok0 + uint(t)*8u);
            cg[t][r].store(ygb, N, origin);
            cu[t][r].store(yub, N, origin);
        }
    }
}


// Flash-attention prefill on custom fragment storage.
//
// attention_m_mma measured 3.65 TFLOP/s, the least efficient kernel in prefill, with
// two structural faults: transposed simdgroup_loads straight out of the device KV
// cache (each lane's reads scatter across rows ~2 KB apart, repeated per 8-wide
// d-slice per block), and Q re-loaded from threadgroup memory for every (d-slice,
// row-block) pair.
//
// This kernel instead runs 32 queries/threadgroup, 4 simdgroups x 8 queries each,
// with Q and O in fragments (Q half, O float), K staged transposed into threadgroup
// memory (scatter is cheap there), V natural, one shared 16 KB K/V tile used
// sequentially with barriers, and exp2 softmax whose row stats are kept per-lane and
// reduced over the 4 row-mate lanes by shuffle-xor {1, 8}.
kernel void attn_prefill_fat(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& hd_rt [[buffer(4)]], constant uint& kvdim [[buffer(5)]], constant uint& base_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]], constant float& scale [[buffer(8)]], constant uint& n_head [[buffer(9)]],
    constant uint& M [[buffer(10)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint hd = 128u;            // specialized; the dispatch gates on this
    (void)hd_rt;
    const uint C = 64u;              // KV per block
    // One shared tile, two layouts in sequence: K transposed as [d=128][kv=64] with
    // ld 72 (18.4 KB), then V natural as [kv=64][d=128] with ld 136 (17.4 KB, fits
    // inside). Size the array for the larger of the two layouts: sizing it for V's
    // shape alone overflows on the K layout, a silent threadgroup-memory
    // out-of-bounds that produces coherent-but-wrong generations.
    threadgroup half kv[128*72];
    uint head = tgpig.x;
    uint q0 = tgpig.y * 32u;         // this tg's first query
    uint kvh = head / group;
    ushort2 mo = morton_order(lane);
    uint qrow = uint(sgitg) * 8u;    // this sg's 8 queries within the tile

    // Q -> registers, half, pre-scaled (folds the softmax scale and log2(e) so
    // the running-max math can use exp2). mq[dd] is the [q8 x d8] fragment.
    simdgroup_matrix_storage<half> mq[16];
    {
        device const float* qb = q + (ulong)(q0 + qrow)*(ulong)(n_head*hd) + head*hd;
        float sc2 = scale * 1.442695041f;  // log2(e)
        uint qi = qrow + mo.y;             // query row this lane reads
        bool okq = q0 + qi < M;
        device const float* qr = q + (ulong)(q0 + qi)*(ulong)(n_head*hd) + head*hd;
        (void)qb;
#pragma clang loop unroll(full)
        for (short dd = 0; dd < 16; dd++) {
            float2 v = okq ? float2(qr[dd*8 + mo.x], qr[dd*8 + mo.x + 1]) : float2(0.0);
            mq[dd] = simdgroup_matrix_storage<half>(half2(v.x * sc2, v.y * sc2));
        }
    }

    simdgroup_matrix_storage<float> o[16];
#pragma clang loop unroll(full)
    for (short dd = 0; dd < 16; dd++) { o[dd] = simdgroup_matrix_storage<float>(float2(0)); }
    float row_m = -1e30f;
    float row_l = 0.0f;

    uint mrem = (q0 < M) ? (M - q0) : 0u;
    uint nq = min(mrem, 32u);
    uint maxseq = base_pos + q0 + nq;     // longest causal row in this tile
    for (uint c0 = 0u; c0 < maxseq; c0 += C) {
        uint cn = min(C, maxseq - c0);
        // ---- K tile, TRANSPOSED into smem: kv[d][kvpos], ld 72. 256 threads
        // stage 64x128 halfs; scatter in threadgroup memory is cheap.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tiitg; e < C*32u; e += 256u) {
            uint kp = e / 32u;             // kv position 0..63
            uint dq = e % 32u;             // which 4-wide d chunk
            device const half* src = kc + (ulong)(c0+kp)*(ulong)kvdim + kvh*hd + dq*4u;
            bool ok = kp < cn;
            half4 v4 = ok ? *(device const half4*)src : half4(0.0);
            kv[(dq*4u+0u)*72u + kp] = v4.x;
            kv[(dq*4u+1u)*72u + kp] = v4.y;
            kv[(dq*4u+2u)*72u + kp] = v4.z;
            kv[(dq*4u+3u)*72u + kp] = v4.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- S = Q.K^T for this sg's 8 queries x 64 kv: 8 fragments.
        simdgroup_matrix_storage<float> sfr[8];
        const threadgroup half* kt_m = kv + ushort(mo.y)*72u + ushort(mo.x);
#pragma clang loop unroll(full)
        for (short kb = 0; kb < 8; kb++) { sfr[kb] = simdgroup_matrix_storage<float>(float2(0)); }
#pragma clang loop unroll(full)
        for (short dd = 0; dd < 16; dd++) {
            simdgroup_matrix_storage<half> kf[8];
#pragma clang loop unroll(full)
            for (short kb = 0; kb < 8; kb++) { kf[kb].load(kt_m, 72, ushort2(kb*8, dd*8)); }
#pragma clang loop unroll(full)
            for (short kb = 0; kb < 8; kb++) { sfr[kb].multiply(mq[dd], kf[kb]); }
        }
        // ---- causal mask + online softmax (base-2). Lane owns (q=qrow+mo.y,
        // kv = c0 + kb*8 + mo.x, +1) per fragment.
        uint qabs = base_pos + q0 + qrow + uint(mo.y);   // absolute position of this row
        float bmax = row_m;
#pragma clang loop unroll(full)
        for (short kb = 0; kb < 8; kb++) {
            thread float2* e = (thread float2*)sfr[kb].thread_elements();
            uint kp0 = c0 + uint(kb)*8u + uint(mo.x);
            e->x = (kp0     <= qabs && kp0     < maxseq) ? e->x : -1e30f;
            e->y = (kp0+1u  <= qabs && kp0+1u  < maxseq) ? e->y : -1e30f;
            bmax = max(bmax, max(e->x, e->y));
        }
        // row max across the 4 lanes sharing this query row (xor 1 flips kv-pair,
        // xor 8 flips the kv-quadrant; y is unchanged by both)
        bmax = max(bmax, simd_shuffle_xor(bmax, 1));
        bmax = max(bmax, simd_shuffle_xor(bmax, 8));
        float corr = (row_m <= -1e29f) ? 0.0f : exp2(row_m - bmax);
        float lsum = 0.0f;
#pragma clang loop unroll(full)
        for (short kb = 0; kb < 8; kb++) {
            thread float2* e = (thread float2*)sfr[kb].thread_elements();
            e->x = (e->x <= -1e29f) ? 0.0f : exp2(e->x - bmax);
            e->y = (e->y <= -1e29f) ? 0.0f : exp2(e->y - bmax);
            lsum += e->x + e->y;
        }
        lsum += simd_shuffle_xor(lsum, 1);
        lsum += simd_shuffle_xor(lsum, 8);
        row_l = row_l * corr + lsum;
        row_m = bmax;
        // rescale O by corr (per-lane: its rows all share qrow+mo.y)
#pragma clang loop unroll(full)
        for (short dd = 0; dd < 16; dd++) {
            thread float2* e = (thread float2*)o[dd].thread_elements();
            e->x *= corr; e->y *= corr;
        }
        // ---- V tile, natural layout kv[kvpos][d], ld 136 (reuse the same smem)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tiitg; e < C*32u; e += 256u) {
            uint kp = e / 32u;
            uint dq = e % 32u;
            device const half* src = vc + (ulong)(c0+kp)*(ulong)kvdim + kvh*hd + dq*4u;
            bool ok = kp < cn;
            *(threadgroup half4*)(kv + kp*136u + dq*4u) = ok ? *(device const half4*)src : half4(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- O += P.V  (P half fragments from sfr)
        simdgroup_matrix_storage<half> pf[8];
#pragma clang loop unroll(full)
        for (short kb = 0; kb < 8; kb++) {
            thread float2* e = (thread float2*)sfr[kb].thread_elements();
            pf[kb] = simdgroup_matrix_storage<half>(half2(e->x, e->y));
        }
        const threadgroup half* v_m = kv + ushort(mo.y)*136u + ushort(mo.x);
#pragma clang loop unroll(full)
        for (short dd = 0; dd < 16; dd++) {
            simdgroup_matrix_storage<half> vf[8];
#pragma clang loop unroll(full)
            for (short kb = 0; kb < 8; kb++) { vf[kb].load(v_m, 136, ushort2(dd*8, kb*8)); }
#pragma clang loop unroll(full)
            for (short kb = 0; kb < 8; kb++) { o[dd].multiply(pf[kb], vf[kb]); }
        }
    }
    // ---- normalize and store: out[query][head*hd + d]. row_l is already
    // replicated-correct across the row-mate lanes (the lsum reduction ran
    // inside the loop), so no further reduction here.
    float inv = (row_l > 0.0f) ? 1.0f / row_l : 0.0f;
    uint qi = q0 + qrow + uint(mo.y);
    if (qi < M) {
        device float* ob = out + (ulong)qi*(ulong)(n_head*hd) + head*hd;
#pragma clang loop unroll(full)
        for (short dd = 0; dd < 16; dd++) {
            thread float2* e = (thread float2*)o[dd].thread_elements();
            ob[dd*8 + mo.x]      = e->x * inv;
            ob[dd*8 + mo.x + 1u] = e->y * inv;
        }
    }
}

"#;
