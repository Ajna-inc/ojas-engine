//! The MoE family: CUDA twins of `ojas-metal`'s `moe.rs`.
//!
//! A routed MoE layer touches only `KSEL` of `E` experts per token, so the weights need not all
//! be resident: the host gathers the selected experts (from cache, or from disk) into a packed
//! buffer and these kernels index it by row. Nothing here assumes the whole expert tensor is
//! present.
//!
//! Weights are read as GGUF stores them (`block_q8_0` is 34 bytes per 32 values: an f16 scale
//! then 32 int8) rather than repacked into the row-Q8 layout the dense path uses. That avoids a
//! requant pass on every gather of a streamed expert, and the per-32 scale is more accurate than
//! one scale per row.
//!
//! Entry names, argument order and arithmetic follow Metal; `simd_sum` becomes `warp_sum`,
//! `simd_max`/`simd_min` become warp shuffles, and the threadgroup barrier becomes
//! `__syncthreads()`.

pub const BODY: &str = r#"
// ---- helpers ---------------------------------------------------------------------------------

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ unsigned int warp_min_u32(unsigned int v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// One block_q8_0: f16 scale then 32 int8 values, 34 bytes, no padding.
__device__ __forceinline__ float q80_block_dot(const unsigned char* b, const float* x) {
    float d = __half2float(*(const __half*)b);
    const char* q = (const char*)(b + 2);
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) s += (float)q[i] * x[i];
    return d * s;
}

// ---- router ----------------------------------------------------------------------------------

// Softmax over E expert logits, then KSEL passes each taking the largest remaining, with the
// selected weights renormalised to sum to 1. One warp; E is bounded by the shared table.
extern "C" __global__ void moe_topk(const float* lg, unsigned int* idx, float* wgt,
                                    int E, int KSEL) {
    __shared__ float pr[1024];
    __shared__ float tw[32];
    int lane = threadIdx.x & 31;
    float mx = -1e30f;
    for (int e = lane; e < E; e += 32) mx = fmaxf(mx, lg[e]);
    mx = warp_max(mx);
    float sum = 0.0f;
    for (int e = lane; e < E; e += 32) { float v = __expf(lg[e] - mx); pr[e] = v; sum += v; }
    sum = warp_all_sum(sum);
    __syncwarp();
    float wsum = 0.0f;
    for (int j = 0; j < KSEL; j++) {
        float bv = -1.0f; unsigned int bi = 0u;
        for (int e = lane; e < E; e += 32) { if (pr[e] > bv) { bv = pr[e]; bi = (unsigned int)e; } }
        float m2 = warp_max(bv);
        // ties resolve to the lowest index, exactly as Metal's simd_min does
        unsigned int cand = (bv == m2) ? bi : 0xffffffffu;
        cand = warp_min_u32(cand);
        if (lane == 0) { idx[j] = cand; tw[j] = m2 / sum; pr[cand] = -1.0f; }
        wsum += m2 / sum;
        __syncwarp();
    }
    if (lane < KSEL) wgt[lane] = tw[lane] / wsum;
}

// ---- expert FFN ------------------------------------------------------------------------------

// act[j][n] = silu(gate) * up for expert idx[j]; grid.y selects the expert slot.
extern "C" __global__ void moe_gu_q80(const float* x, const unsigned char* wg,
                                      const unsigned char* wu, float* act,
                                      const unsigned int* idx, int K, int N) {
    int j = blockIdx.y;
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int out_row = blockIdx.x * warps + warp;
    if (out_row >= N) return;
    int nblk = K >> 5;
    long rowbytes = (long)nblk * 34;
    long row = (long)idx[j] * (long)N + (long)out_row;
    const unsigned char* gr = wg + row * rowbytes;
    const unsigned char* ur = wu + row * rowbytes;
    float gs = 0.0f, us = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const float* xb = x + blk * 32;
        gs += q80_block_dot(gr + (long)blk * 34, xb);
        us += q80_block_dot(ur + (long)blk * 34, xb);
    }
    gs = warp_sum(gs);
    us = warp_sum(us);
    if (lane == 0) act[(long)j * (long)N + out_row] = (gs / (1.0f + __expf(-gs))) * us;
}

// x[n] += Σ_j wgt[j] · down(expert idx[j])[n] — the router weights are applied here, so the
// caller never materialises a per-expert output. Four output rows per warp, as Metal does.
extern "C" __global__ void moe_down_q80(const float* act, const unsigned char* wd, float* x,
                                        const unsigned int* idx, const float* wgt,
                                        int K, int N, int KSEL) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int out_row = (blockIdx.x * warps + warp) * 4;
    if (out_row >= N) return;
    int nblk = K >> 5;
    long rowbytes = (long)nblk * 34;
    float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < KSEL; j++) {
        float w = wgt[j];
        const float* xr = act + (long)j * (long)K;
        for (int r = 0; r < 4; r++) {
            if (out_row + r >= N) break;
            long row = (long)idx[j] * (long)N + (long)(out_row + r);
            const unsigned char* wr = wd + row * rowbytes;
            float acc = 0.0f;
            for (int blk = lane; blk < nblk; blk += 32) acc += q80_block_dot(wr + (long)blk * 34, xr + blk * 32);
            acc = warp_sum(acc);
            if (lane == 0) res[r] += w * acc;
        }
    }
    if (lane == 0) {
        for (int r = 0; r < 4 && out_row + r < N; r++) x[out_row + r] += res[r];
    }
}
// ---- streamed variants: experts live at arbitrary slots in a cache arena ---------------------
//
// The kernels above index a contiguous expert tensor (`idx[j] * N + row`), which is right when
// the whole tensor is resident. A streamed layer cannot: the cache holds the selected experts at
// whatever slots were free, so the host passes a byte offset per selected expert, the same shape
// as Metal's direct-cached-expert tables, where a hit is an address rather than a copy.

extern "C" __global__ void moe_gu_q80_slots(const float* x, const unsigned char* arena_g,
                                            const unsigned char* arena_u, float* act,
                                            const unsigned int* off_g, const unsigned int* off_u,
                                            int K, int N) {
    int j = blockIdx.y;
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int out_row = blockIdx.x * warps + warp;
    if (out_row >= N) return;
    int nblk = K >> 5;
    long rowbytes = (long)nblk * 34;
    const unsigned char* gr = arena_g + (long)off_g[j] + (long)out_row * rowbytes;
    const unsigned char* ur = arena_u + (long)off_u[j] + (long)out_row * rowbytes;
    float gs = 0.0f, us = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const float* xb = x + blk * 32;
        gs += q80_block_dot(gr + (long)blk * 34, xb);
        us += q80_block_dot(ur + (long)blk * 34, xb);
    }
    gs = warp_sum(gs);
    us = warp_sum(us);
    if (lane == 0) act[(long)j * (long)N + out_row] = (gs / (1.0f + __expf(-gs))) * us;
}

extern "C" __global__ void moe_down_q80_slots(const float* act, const unsigned char* arena,
                                              float* x, const unsigned int* off_d,
                                              const float* wgt, int K, int N, int KSEL) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int out_row = blockIdx.x * warps + warp;
    if (out_row >= N) return;
    int nblk = K >> 5;
    long rowbytes = (long)nblk * 34;
    float res = 0.0f;
    for (int j = 0; j < KSEL; j++) {
        const float* xr = act + (long)j * (long)K;
        const unsigned char* wr = arena + (long)off_d[j] + (long)out_row * rowbytes;
        float acc = 0.0f;
        for (int blk = lane; blk < nblk; blk += 32) acc += q80_block_dot(wr + (long)blk * 34, xr + blk * 32);
        acc = warp_sum(acc);
        if (lane == 0) res += wgt[j] * acc;
    }
    if (lane == 0) x[out_row] += res;
}
"#;

/// Canonical entry names registered from this family.
pub const NAMES: &[&str] = &["moe_topk", "moe_gu_q80", "moe_down_q80",
                             "moe_gu_q80_slots", "moe_down_q80_slots"];
