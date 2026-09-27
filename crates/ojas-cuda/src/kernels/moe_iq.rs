//! IQ-quantized expert kernels: the format the streamed models ship in.
//!
//! Flash Next's routed experts are `IQ3_S`/`IQ4_XS` gate-up with `IQ4_NL` down, so a CUDA
//! streaming path that only reads `Q8_0` cannot run the model the memory figure was measured on.
//! These are the twins of `ojas-metal`'s `moe_iq.rs` entries for that combination.
//!
//! Both formats share the IQ4_NL codebook: a 4-bit index into 16 fixed levels, which is why a
//! 4-bit IQ weight is closer to Q8 accuracy than a linear Q4 would be. The block layouts differ:
//!
//! * `IQ4_NL` — 32 values per block, 18 bytes: `f16 d`, then 16 bytes of packed nibbles.
//! * `IQ4_XS` — 256 values per block, 136 bytes: `f16 d`, `u16 scales_h`, `u8 scales_l[4]`,
//!   then 128 bytes of nibbles. Eight sub-blocks of 32 share `d`, each with its own 6-bit
//!   scale split 4 low bits in `scales_l` (nibble-packed) and 2 high bits in `scales_h`,
//!   biased by 32 as the K-quants are.
//!
//! The nibble order is easy to get subtly wrong: within a 16-byte group the low nibbles are the
//! first 16 values and the high nibbles the next 16, not interleaved. The reference is
//! `ojas-formats/src/gguf.rs`, and the conformance test compares against it.

pub const BODY: &str = r#"
__constant__ char KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// One IQ4_XS block (136 bytes, 256 values) dotted against 256 floats.
__device__ __forceinline__ float iq4xs_block_dot(const unsigned char* b, const float* x) {
    float d = __half2float(*(const __half*)b);
    unsigned short sh = (unsigned short)b[2] | ((unsigned short)b[3] << 8);
    const unsigned char* sl = b + 4;
    const unsigned char* qs = b + 8;
    float acc = 0.0f;
    #pragma unroll
    for (int ib = 0; ib < 8; ib++) {
        unsigned int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xF)
                        | ((unsigned int)((sh >> (2 * ib)) & 3) << 4);
        float dl = d * (float)((int)ls - 32);
        const unsigned char* q = qs + ib * 16;
        const float* xb = x + ib * 32;
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            s += (float)KVALUES_IQ4NL[q[j] & 0xF] * xb[j];          // low nibbles: first 16
            s += (float)KVALUES_IQ4NL[q[j] >> 4] * xb[j + 16];      // high nibbles: next 16
        }
        acc += dl * s;
    }
    return acc;
}

// One IQ4_NL block (18 bytes, 32 values).
__device__ __forceinline__ float iq4nl_block_dot(const unsigned char* b, const float* x) {
    float d = __half2float(*(const __half*)b);
    const unsigned char* q = b + 2;
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        s += (float)KVALUES_IQ4NL[q[j] & 0xF] * x[j];
        s += (float)KVALUES_IQ4NL[q[j] >> 4] * x[j + 16];
    }
    return d * s;
}

// act[j][n] = silu(gate) * up, experts stored IQ4_XS. grid.y selects the expert slot.
extern "C" __global__ void moe_gu_iq4xs(const float* x, const unsigned char* wg,
                                        const unsigned char* wu, float* act,
                                        const unsigned int* idx, int K, int N) {
    int j = blockIdx.y;
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int out_row = blockIdx.x * warps + warp;
    if (out_row >= N) return;
    int nblk = K >> 8;                       // 256 values per IQ4_XS block
    long rowbytes = (long)nblk * 136;
    long row = (long)idx[j] * (long)N + (long)out_row;
    const unsigned char* gr = wg + row * rowbytes;
    const unsigned char* ur = wu + row * rowbytes;
    float gs = 0.0f, us = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const float* xb = x + blk * 256;
        gs += iq4xs_block_dot(gr + (long)blk * 136, xb);
        us += iq4xs_block_dot(ur + (long)blk * 136, xb);
    }
    gs = warp_sum(gs);
    us = warp_sum(us);
    if (lane == 0) act[(long)j * (long)N + out_row] = (gs / (1.0f + __expf(-gs))) * us;
}

// x[n] += Σ_j wgt[j] · down(expert idx[j])[n], down stored IQ4_NL.
extern "C" __global__ void moe_down_iq4nl(const float* act, const unsigned char* wd, float* x,
                                          const unsigned int* idx, const float* wgt,
                                          int K, int N, int KSEL) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int out_row = blockIdx.x * warps + warp;
    if (out_row >= N) return;
    int nblk = K >> 5;                       // 32 values per IQ4_NL block
    long rowbytes = (long)nblk * 18;
    float res = 0.0f;
    for (int j = 0; j < KSEL; j++) {
        const float* xr = act + (long)j * (long)K;
        long row = (long)idx[j] * (long)N + (long)out_row;
        const unsigned char* wr = wd + row * rowbytes;
        float acc = 0.0f;
        for (int blk = lane; blk < nblk; blk += 32) acc += iq4nl_block_dot(wr + (long)blk * 18, xr + blk * 32);
        acc = warp_sum(acc);
        if (lane == 0) res += wgt[j] * acc;
    }
    if (lane == 0) x[out_row] += res;
}

// ---- IQ3_S expert gate/up -------------------------------------------------------------------
//
// The format Flash Next's routed experts ship in: IQ3_S gate-up with IQ4_NL down.
//
// IQ3_S block: 110 bytes per 256 values — `f16 d`, 64 B of low grid indices, 8 B of high bits
// (one per 32-value group), 32 B of sign bytes, 4 B of 4-bit scales. Each of the 8 groups reads
// eight 9-bit indices into `iq3s_grid`, where each entry packs four 8-bit magnitudes, and a sign
// byte whose bits are applied through `kmask_iq2xs`.
//
// Four rows per warp, as Metal does. The activation group is loaded into registers once and
// reused across all four gate/up row pairs; one row per warp would re-read those 32 floats eight
// times per block. The reference is `ojas-formats`, and the conformance test compares to it.
__device__ __forceinline__ float iq3s_group(const unsigned char* blk, unsigned int g,
                                            const float* xg) {
    const unsigned char* qs = blk + 2;
    const unsigned char* qh = blk + 66;
    const unsigned char* sg = blk + 74;
    const unsigned char* sc = blk + 106;
    unsigned int h = qh[g], qo = 8u * g, so = 4u * g;
    float gsum = 0.0f;
    #pragma unroll
    for (unsigned int l = 0; l < 4u; l++) {
        unsigned int i1 = (unsigned int)qs[qo + 2u * l]      | ((h << (8u - 2u * l)) & 256u);
        unsigned int i2 = (unsigned int)qs[qo + 2u * l + 1u] | ((h << (7u - 2u * l)) & 256u);
        unsigned int g1 = iq3s_grid[i1], g2 = iq3s_grid[i2];
        unsigned int sgn = sg[so + l];
        #pragma unroll
        for (unsigned int j = 0; j < 4u; j++) {
            float v1 = (float)((g1 >> (8u * j)) & 255u);
            float v2 = (float)((g2 >> (8u * j)) & 255u);
            gsum += v1 * ((sgn & kmask_iq2xs[j])      ? -xg[8u * l + j]      : xg[8u * l + j]);
            gsum += v2 * ((sgn & kmask_iq2xs[j + 4u]) ? -xg[8u * l + j + 4u] : xg[8u * l + j + 4u]);
        }
    }
    return (1.0f + 2.0f * (float)((sc[g >> 1u] >> (4u * (g & 1u))) & 15u)) * gsum;
}

/// Shared body: `gr`/`ur` are this output row's gate and up rows, already addressed.
#define IQ3S_GU_ROWS(GBASE, UBASE)                                                            \
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;            \
    unsigned int row0 = blockIdx.x * (warps * 4) + warp * 4;                                  \
    if (row0 >= (unsigned int)N) return;                                                      \
    unsigned int nb = (unsigned int)K / 256u, rb = nb * 110u;                                 \
    float gs[4] = {0.f, 0.f, 0.f, 0.f}, us[4] = {0.f, 0.f, 0.f, 0.f};                         \
    for (unsigned int u = lane; u < nb * 8u; u += 32u) {                                      \
        unsigned int b = u >> 3, g = u & 7u;                                                  \
        float xv[32];                                                                          \
        _Pragma("unroll")                                                                      \
        for (unsigned int z = 0; z < 32u; ++z) xv[z] = x[b * 256u + g * 32u + z];              \
        for (unsigned int r = 0; r < 4u && row0 + r < (unsigned int)N; ++r) {                  \
            const unsigned char* gb = (GBASE) + (long)(row0 + r) * rb + (long)b * 110;         \
            const unsigned char* ub = (UBASE) + (long)(row0 + r) * rb + (long)b * 110;         \
            gs[r] += __half2float(*(const __half*)gb) * iq3s_group(gb, g, xv);                 \
            us[r] += __half2float(*(const __half*)ub) * iq3s_group(ub, g, xv);                 \
        }                                                                                      \
    }                                                                                          \
    for (unsigned int r = 0; r < 4u; ++r) { gs[r] = warp_sum(gs[r]); us[r] = warp_sum(us[r]); }\
    if (lane == 0) {                                                                           \
        for (unsigned int r = 0; r < 4u && row0 + r < (unsigned int)N; ++r)                    \
            act[(long)j * (long)N + row0 + r] =                                                \
                (gs[r] / (1.0f + __expf(-gs[r]))) * us[r];                                     \
    }

// Contiguous expert tensor, indexed by router id.
extern "C" __global__ void moe_gu_iq3s(const float* x, const unsigned char* wg,
                                       const unsigned char* wu, float* act,
                                       const unsigned int* idx, int K, int N) {
    unsigned int j = blockIdx.y, e = idx[j];
    unsigned int nb_ = (unsigned int)K / 256u;
    long eoff = (long)e * (long)N * (long)(nb_ * 110u);
    IQ3S_GU_ROWS(wg + eoff, wu + eoff)
}

// Streamed: the expert lives at an arbitrary byte offset in a cache arena, so the host passes an
// offset per selected expert instead of an id. Same shape as `moe_gu_q80_slots`.
extern "C" __global__ void moe_gu_iq3s_slots(const float* x, const unsigned char* arena_g,
                                             const unsigned char* arena_u, float* act,
                                             const unsigned int* off_g, const unsigned int* off_u,
                                             int K, int N) {
    unsigned int j = blockIdx.y;
    IQ3S_GU_ROWS(arena_g + (long)off_g[j], arena_u + (long)off_u[j])
}
"#;

/// Canonical entry names registered from this family.
pub const NAMES: &[&str] = &["moe_gu_iq4xs", "moe_down_iq4nl",
                             "moe_gu_iq3s", "moe_gu_iq3s_slots"];
