//! Async-copy GEMM — the undocumented Apple7/8 DMA staging path.
//!
//! `simdgroup_async_copy_2d` is the hardware 2D DMA that MPS's own matmul uses
//! (reverse-engineered in dougallj/applegpu#28) and the main M1/M2-generation GEMM
//! lever: one simdgroup issues the copy in ~2 instructions and dedicated hardware moves
//! the tile while the other simdgroups keep computing, where cooperative staging burns
//! hundreds of ALU issue slots on loads and address math. Published M1 Max numbers with
//! it reach ~90% ALU against this engine's 59%.
//!
//! It is undocumented ABI, so this family is compiled fallibly: if the `__asm`
//! declarations ever stop linking on a future macOS, the loader logs and skips it,
//! dispatch never finds the pipeline, and the plain `gemm_mm_q4l` carries on.
//!
//! Hazards, per the reference headers and metal-benchmarks: (1) an M1-generation hang if
//! the copied region is never dereferenced before kernel end, which a GEMM loop always
//! does; (2) issuance must stay confined to one simdgroup, with a barrier afterwards;
//! (3) on Apple9+ (M3/M4) async copies are a slowdown, so the dispatch site must gate on
//! device family, not just pipeline presence.
//!
//! The kernel is gemm_mm_q4l restructured around row-major tiles, which is what the DMA
//! writes: A (weights) is still dequantised cooperatively, since nibbles plus scales
//! cannot be DMA'd into halves, but into contiguous stores; B (activations,
//! pre-converted to half once per chunk by copy_f32_half) is staged entirely by the copy
//! engine, including zero-fill of the M-tail.

/// Self-contained MSL source (compiled without the shared prelude — the async
/// intrinsics must be declared before use and nothing else here needs Q4 macros).
pub const GEMM_AC_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---- undocumented async-copy ABI ----
struct _simdgroup_event_t;
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
    ulong, ulong,
    threadgroup void *, ulong, ulong, ulong2,
    const device void *, ulong, ulong, ulong2,
    long2, int)
    __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");
void __metal_wait_simdgroup_events(int, thread _simdgroup_event_t**)
    __asm("air.wait_simdgroup_events");

// device -> threadgroup, half elements, zero-clamped edges.
METAL_FUNC thread _simdgroup_event_t* ac_copy_half(
    threadgroup half* dst, ulong dst_ld, ulong2 dst_tile,
    device const half* src, ulong src_ld, ulong2 src_tile) {
    return __metal_simdgroup_async_copy_2d(
        2, 2,
        (threadgroup void*)dst, dst_ld, 1, dst_tile,
        (const device void*)src, src_ld, 1, src_tile,
        long2(0, 0), 0 /* clamp_to_zero */);
}

// 64(N) x 32(M) x 32(K) tile, 128 threads, mc[8] — gemm_mm_q4l's shape with the
// activation tile staged by the copy engine. Row-major tiles, +8-half row padding
// (bank-conflict pad).
kernel void gemm_mm_q4l_ac(device const half* xh [[buffer(0)]], device const uchar* w4 [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const half* qa_ [[buffer(5)]], constant uint& accum [[buffer(6)]], constant uint& M [[buffer(7)]],
    device const half* qb_ [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[64*32];       // blocked 8x8, as the staged kernel
    threadgroup half sb[32*40];       // [token][k], ld 40
    threadgroup float idm[64];
    const uint r0 = tgpig.y*64u;
    const uint t0 = tgpig.x*32u;
    uint lr0 = tiitg/2u;              // A: row within tile (0..63)
    uint il0 = tiitg%2u;              // A: which 16-k half of the 32-k slab
    uint nblk = K/32u;
    device const uchar* arow = w4 + (ulong)(r0+lr0)*(ulong)(K/2u);
    device const half* arow_a = qa_ + (ulong)(r0+lr0)*(ulong)nblk;
    device const half* arow_b = qb_ + (ulong)(r0+lr0)*(ulong)nblk;
    if (tiitg < 64u) { idm[tiitg] = (tiitg/8u == tiitg%8u) ? 1.0 : 0.0; }
    uint mrem = (t0 < M) ? (M - t0) : 0u;
    ulong2 btile = ulong2(32, 32);
    ulong2 btile_src = ulong2(32, min(mrem, 32u));
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) { mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }
    for (uint lk = 0u; lk < K; lk += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // B: one simdgroup kicks the DMA and waits; A: everyone dequants meanwhile.
        if (sgitg == 0) {
            thread _simdgroup_event_t* ev[1];
            ev[0] = ac_copy_half(sb, 40, btile, xh + (ulong)t0*(ulong)K + lk, K, btile_src);
            __metal_wait_simdgroup_events(1, ev);
        }
        {   // A: identical blocked fill to the staged kernel (its layout measured
            // faster than row-major + transposed MMA loads — see file history).
            uint k0 = lk + il0*16u;
            half A = arow_a[k0/32u];
            half B = arow_b[k0/32u];
            uint sy = lr0/8u, lx = lr0%8u;
            device const uchar4* bp4 = (device const uchar4*)(arow + k0/2u);
            uchar4 b0 = bp4[0], b1 = bp4[1];
            threadgroup half* dst = sa + 64u*(16u*il0 + sy) + lx;
            dst[  0] = A*half(b0.x & 0x0Fu) + B;  dst[  8] = A*half(b0.x >> 4) + B;
            dst[ 16] = A*half(b0.y & 0x0Fu) + B;  dst[ 24] = A*half(b0.y >> 4) + B;
            dst[ 32] = A*half(b0.z & 0x0Fu) + B;  dst[ 40] = A*half(b0.z >> 4) + B;
            dst[ 48] = A*half(b0.w & 0x0Fu) + B;  dst[ 56] = A*half(b0.w >> 4) + B;
            dst[512] = A*half(b1.x & 0x0Fu) + B;  dst[520] = A*half(b1.x >> 4) + B;
            dst[528] = A*half(b1.y & 0x0Fu) + B;  dst[536] = A*half(b1.y >> 4) + B;
            dst[544] = A*half(b1.z & 0x0Fu) + B;  dst[552] = A*half(b1.z >> 4) + B;
            dst[560] = A*half(b1.w & 0x0Fu) + B;  dst[568] = A*half(b1.w >> 4) + B;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // MMA: blocked lsma exactly as the staged kernel; mb from the row-major
        // DMA'd tile with ld 40.
        uint tok0 = (uint(sgitg) / 2u) * 16u;
        threadgroup const half* lsma = sa + 4u*64u*(uint(sgitg)%2u);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) { simdgroup_load(ma[i], lsma + 64*i, 8, 0, false); }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], sb + (tok0 + uint(i)*8u)*40u + uint(ik)*8u, 40, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) { simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]); }
            lsma += 8*64;
        }
    }
    device float* C = y + (r0 + 32u*(uint(sgitg) & 1u)) + (ulong)(t0 + 16u*(uint(sgitg) >> 1u))*(ulong)N;
    if (accum != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 mid, mo;
        simdgroup_load(mid, idm, 8, 0, false);
        for (short i = 0; i < 8; i++) {
            device float* Ci = C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N;
            simdgroup_load(mo, Ci, N, 0, false);
            simdgroup_multiply_accumulate(mc[i], mc[i], mid, mo);
            simdgroup_store(mc[i], Ci, N, 0, false);
        }
    } else {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (ulong)(8*(i/4))*(ulong)N, N, 0, false);
        }
    }
}
"#;
