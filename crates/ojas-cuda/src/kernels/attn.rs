// Kernel bodies are byte-identical to the reference CUDA backend; entry names are
// canonicalized to the Metal-canonical ones.
// Compiles against kernels::PRELUDE (cuda_fp16 + warp reduction helpers).
pub const BODY: &str = r#"
// Flash-decoding. attention_short_g uses one block per head (n_head=16 blocks) and
// scans the whole KV serially, which starves the GPU at long context. Flash-decoding
// adds a KV-split axis: the sequence is cut into NSPLIT chunks, each (head, split)
// block computes a partial softmax+PV in parallel, then a reduce kernel combines them
// via log-sum-exp. grid=(n_head, NSPLIT) → 128 blocks, filling the SMs. Up to ~8× on
// long-context decode (Together.ai/Stanford). seq comes from ctl, so this is
// graph-capturable.
#define NSPLIT 8

extern "C" __global__ void attention_part_g(const float* q, const __half* kc,
    const __half* vc, float* po, float* pm, float* pl, const int* ctl,
    int hd, int kvdim, int group, float scale) {
    __shared__ float qsh[256];
    __shared__ float sc[512];   // max chunk = SEQ_MAX(4096)/NSPLIT
    __shared__ float part[32];
    int head = blockIdx.x, split = blockIdx.y, lid = threadIdx.x, ts = blockDim.x;
    int lane = lid & 31, sg = lid >> 5, nsg = ts >> 5;
    int seq = ctl[1] + 1;
    int chunk = (seq + NSPLIT - 1) / NSPLIT;
    int s0 = split * chunk, s1 = min(s0 + chunk, seq), idx = head * NSPLIT + split;
    if (s0 >= seq) { // empty split → contributes nothing in the reduce
        if (lid == 0) { pm[idx] = -1e30f; pl[idx] = 0.f; }
        for (int i = lid; i < hd; i += ts) po[(long)idx * hd + i] = 0.f;
        return;
    }
    int kvh = head / group, nc = s1 - s0;
    const float* qh = q + head * hd;
    for (int i = lid; i < hd; i += ts) qsh[i] = qh[i];
    __syncthreads();
    for (int tt = sg; tt < nc; tt += nsg) {
        const __half* kt = kc + (long)(s0 + tt) * kvdim + kvh * hd;
        float sv = 0.0f;
        for (int i = lane; i < hd; i += 32) sv += qsh[i] * __half2float(kt[i]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sv += __shfl_down_sync(0xffffffffu, sv, o);
        if (lane == 0) sc[tt] = sv * scale;
    }
    __syncthreads();
    float lmax = -1e30f;
    for (int tt = lid; tt < nc; tt += ts) lmax = fmaxf(lmax, sc[tt]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
    if (lane == 0) part[sg] = lmax;
    __syncthreads();
    float mx = -1e30f;
    for (int j = 0; j < nsg; j++) mx = fmaxf(mx, part[j]);
    float lsum = 0.0f;
    for (int tt = lid; tt < nc; tt += ts) { float e = expf(sc[tt] - mx); sc[tt] = e; lsum += e; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
    if (lane == 0) part[sg] = lsum;
    __syncthreads();
    float sum = 0.0f;
    for (int j = 0; j < nsg; j++) sum += part[j];
    for (int i = lid; i < hd; i += ts) {
        float acc = 0.0f;
        for (int tt = 0; tt < nc; tt++) acc += sc[tt] * __half2float(vc[(long)(s0 + tt) * kvdim + kvh * hd + i]);
        po[(long)idx * hd + i] = acc;   // unnormalized Σ exp(s-mx)·V
    }
    if (lid == 0) { pm[idx] = mx; pl[idx] = sum; }
}

// Reduce the NSPLIT partials per head via log-sum-exp. grid=(n_head).
extern "C" __global__ void attention_merge(const float* po, const float* pm,
    const float* pl, float* out, int hd) {
    int head = blockIdx.x, lid = threadIdx.x, ts = blockDim.x, base = head * NSPLIT;
    float gm = -1e30f;
    #pragma unroll
    for (int s = 0; s < NSPLIT; s++) gm = fmaxf(gm, pm[base + s]);
    float denom = 0.0f;
    #pragma unroll
    for (int s = 0; s < NSPLIT; s++) denom += pl[base + s] * expf(pm[base + s] - gm);
    float inv = 1.0f / denom;
    for (int i = lid; i < hd; i += ts) {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < NSPLIT; s++) acc += po[(long)(base + s) * hd + i] * expf(pm[base + s] - gm);
        out[head * hd + i] = acc * inv;
    }
}

// Tensor-core flash attention: f16 mma computes the QK^T scores (no per-score warp
// reduction), then CUDA-core online-softmax + PV (small O accumulator). 16 queries/
// block, KT=32 keys/tile, 8 warps (warps 0-3 do the 4 score n-tiles). Q/K(transposed
// hd-major)/V staged f16. grid=(n_head, ceil(M/16)). mma layout validated by probe.
#define TKT 32

// hd-aware (hd ∈ {128,256}, multiple of 16). Shared arrays keep a physical row
// stride of 256 (fits hd≤256); all loops iterate only over the real hd, so a
// hd=128 model (Qwen3) does half the work and writes exactly hd per query (no
// out-of-bounds into the next head).
__device__ __forceinline__ void flash_tc_core(const float* q, const __half* kc,
    const __half* vc, float* out, int hd, int kvdim, int qdim, int pos_base, int M,
    int group, float scale) {
    int head = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    int qbase = blockIdx.y * 16, kvh = head / group;
    int seqmax = pos_base + min(M - 1, qbase + 15) + 1;
    int hj = hd >> 5;                    // O-accumulator floats/lane (hd/32): 4 or 8
    __shared__ __half sQ[16 * 256];      // queries (row-major [q][256-stride])
    __shared__ __half sKt[256 * TKT];    // keys TRANSPOSED [hd][key] for mma B
    __shared__ __half sV[TKT * 256];     // values [key][256-stride]
    __shared__ float sS[16 * TKT];       // scores [q][key]
    // load Q (16 queries) → sQ f16 (only hd cols; row stride stays 256)
    for (int idx = tid; idx < 16 * hd; idx += 256) {
        int qq = idx / hd, ii = idx % hd;
        int qi = qbase + qq;
        sQ[qq * 256 + ii] = __float2half((qi < M) ? q[(long)qi * qdim + head * hd + ii] : 0.f);
    }
    // per-warp: 2 queries (n=0,1 → query = qbase + warp + 8*n)
    int qi[2], pos[2];
    #pragma unroll
    for (int n = 0; n < 2; n++) { qi[n] = qbase + warp + 8 * n; pos[n] = pos_base + qi[n]; }
    float m_i[2] = {-1e30f, -1e30f}, l_i[2] = {0.f, 0.f}, acc[2][8];
    #pragma unroll
    for (int n = 0; n < 2; n++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[n][j] = 0.f;
    __syncthreads();

    for (int t0 = 0; t0 < seqmax; t0 += TKT) {
        // stage K (transposed → sKt[hd][key]) and V (sV[key][hd]) as f16
        for (int idx = tid; idx < TKT * hd; idx += 256) {
            int kk = idx / hd, ii = idx % hd, t = t0 + kk;
            __half kv = __float2half(0.f), vv = __float2half(0.f);
            if (t < seqmax) { kv = kc[(long)t * kvdim + kvh * hd + ii]; vv = vc[(long)t * kvdim + kvh * hd + ii]; }
            sKt[ii * TKT + kk] = kv;   // f16 cache → direct (no conversion)
            sV[kk * 256 + ii] = vv;
        }
        __syncthreads();
        // scores via mma: warp w (<4) computes n-tile w = keys [w*8, w*8+8) over all 16 queries
        if (warp < 4) {
            int n0 = warp * 8;
            float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (int k0 = 0; k0 < hd; k0 += 16) {
                unsigned a0, a1, a2, a3, b0, b1;
                const __half* aptr = &sQ[(lane % 16) * 256 + k0 + (lane / 16) * 8];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(aptr));
                const __half* bptr = &sKt[(k0 + (lane % 16)) * TKT + n0];
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];"
                    : "=r"(b0), "=r"(b1) : "l"(bptr));
                asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            // C fragment (m16n8): d_l → row=(l/2)*8+lane/4, col=(lane%4)*2+l%2
            int gi0 = lane / 4, gi1 = 8 + lane / 4, gj0 = (lane % 4) * 2, gj1 = gj0 + 1;
            sS[gi0 * TKT + n0 + gj0] = c0; sS[gi0 * TKT + n0 + gj1] = c1;
            sS[gi1 * TKT + n0 + gj0] = c2; sS[gi1 * TKT + n0 + gj1] = c3;
        }
        __syncthreads();
        // CUDA-core online softmax + PV (2 queries/warp), scores from sS
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            int ql = warp + 8 * n;   // local query row (0..15)
            if (qi[n] >= M) continue;
            for (int kk = 0; kk < TKT; kk++) {
                int t = t0 + kk;
                if (t > pos[n]) break;
                float s = sS[ql * TKT + kk] * scale;
                float m_new = fmaxf(m_i[n], s);
                float corr = __expf(m_i[n] - m_new);
                float p = __expf(s - m_new);
                l_i[n] = l_i[n] * corr + p;
                // fixed-8 unroll keeps acc in registers; hj gates the real work.
                #pragma unroll
                for (int j = 0; j < 8; j++) if (j < hj) acc[n][j] = acc[n][j] * corr + p * __half2float(sV[kk * 256 + lane + j * 32]);
                m_i[n] = m_new;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        if (qi[n] >= M) continue;
        float inv = 1.0f / l_i[n];
        #pragma unroll
        for (int j = 0; j < 8; j++) if (j < hj) out[(long)qi[n] * qdim + head * hd + lane + j * 32] = acc[n][j] * inv;
    }
}

// pos_base as a host int (eager path — qwen35 + qwen3 eager).
extern "C" __global__ void attention_flash_tc(const float* q, const __half* kc,
    const __half* vc, float* out, int hd, int kvdim, int qdim, int pos_base, int M,
    int group, float scale) {
    flash_tc_core(q, kc, vc, out, hd, kvdim, qdim, pos_base, M, group, scale);
}

// pos_base from device memory (pctl[0]) — CUDA-graph-capturable chunked prefill.
extern "C" __global__ void attention_flash_tc_g(const float* q, const __half* kc,
    const __half* vc, float* out, const int* pctl, int hd, int kvdim, int qdim, int M,
    int group, float scale) {
    flash_tc_core(q, kc, vc, out, hd, kvdim, qdim, pctl[0], M, group, scale);
}

// Flash-attention v2 style: both QK and PV on tensor cores (flash_tc does PV on
// CUDA cores with 4 idle warps). Per K/V tile: mma QK → shared scores;
// shared-memory online softmax (warp w owns rows w, w+8; lane = key col) writes P
// (f16) + a per-row rescale; then mma P·V into an O accumulator held in registers
// across the whole K-loop (rescaled each tile), with the hd n-tiles split across
// all 8 warps. hd∈{128,256}, 16 queries/block, grid=(n_head, ceil(M/16)).
__device__ __forceinline__ void flash_tc2_core(const float* q, const __half* kc,
    const __half* vc, float* out, int hd, int kvdim, int qdim, int pos_base, int M,
    int group, float scale) {
    int head = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    int qbase = blockIdx.y * 16, kvh = head / group;
    int seqmax = pos_base + min(M - 1, qbase + 15) + 1;
    int nt_all = hd >> 3;                // n-tiles across hd (hd/8), ≤16 for hd≤128
    // HS = shared row stride = hd (128 for Qwen3). tc2 is the hd≤128 path; using hd
    // rather than 256 halves shared memory → 4 blocks/SM instead of 2, overlapping one
    // block's mma with another's softmax (the attention is occupancy/overhead-bound).
    // HS/KTS: padded shared strides (+8 f16). A power-of-2 row stride (128 / TKT=32)
    // maps consecutive rows to the same shared banks → severe conflicts on the
    // ldmatrix reads (sQ/sV) and the transpose staging write (sKt). Padding to
    // 136 / 40 (still 8-f16 aligned for ldmatrix) spreads rows across banks.
    const int HS = 136;
    const int KTS = TKT + 8;
    __shared__ __half sQ[16 * HS];       // queries [q][HS-stride]
    __shared__ __half sKt[HS * (TKT + 8)]; // K transposed [hd][key], padded stride
    __shared__ __half sV[TKT * HS];      // V [key][HS-stride]
    __shared__ __half sP[16 * (TKT + 8)]; // probabilities [q][key], padded stride (KTS)
    __shared__ float sS[16 * TKT];       // scores [q][key]
    __shared__ float sM[16], sL[16], sCorr[16]; // running max, denom, per-tile rescale
    for (int idx = tid; idx < 16 * hd; idx += 256) {
        int qq = idx / hd, ii = idx % hd, qi = qbase + qq;
        sQ[qq * HS + ii] = __float2half((qi < M) ? q[(long)qi * qdim + head * hd + ii] : 0.f);
    }
    if (tid < 16) { sM[tid] = -1e30f; sL[tid] = 0.f; }
    float O[4][4];                        // ≤4 n-tiles/warp (hd=256), each m16n8 = 4 floats/lane
    #pragma unroll
    for (int a = 0; a < 4; a++) { O[a][0] = O[a][1] = O[a][2] = O[a][3] = 0.f; }
    __syncthreads();

    for (int t0 = 0; t0 < seqmax; t0 += TKT) {
        // stage K^T and V (f16 cache → direct)
        for (int idx = tid; idx < TKT * hd; idx += 256) {
            int kk = idx / hd, ii = idx % hd, t = t0 + kk;
            __half kv = __float2half(0.f), vv = __float2half(0.f);
            if (t < seqmax) { kv = kc[(long)t * kvdim + kvh * hd + ii]; vv = vc[(long)t * kvdim + kvh * hd + ii]; }
            sKt[ii * KTS + kk] = kv;
            sV[kk * HS + ii] = vv;
        }
        __syncthreads();
        // QK scores: warps 0-3, one 8-key n-tile each. Splitting across all 8 warps is
        // slower: it halves the per-warp mma count and hurts mma pipelining.
        if (warp < 4) {
            int n0 = warp * 8;
            float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (int k0 = 0; k0 < hd; k0 += 16) {
                unsigned a0, a1, a2, a3, b0, b1;
                const __half* aptr = &sQ[(lane % 16) * HS + k0 + (lane / 16) * 8];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(aptr));
                const __half* bptr = &sKt[(k0 + (lane % 16)) * KTS + n0];
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "l"(bptr));
                asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            int gi0 = lane / 4, gi1 = 8 + lane / 4, gj0 = (lane % 4) * 2, gj1 = gj0 + 1;
            sS[gi0 * TKT + n0 + gj0] = c0; sS[gi0 * TKT + n0 + gj1] = c1;
            sS[gi1 * TKT + n0 + gj0] = c2; sS[gi1 * TKT + n0 + gj1] = c3;
        }
        __syncthreads();
        // online softmax: warp w → rows w and w+8; lane = key column (TKT==32==warpSize)
        #pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            int row = warp + rr * 8;
            int qpos = pos_base + qbase + row;
            int t = t0 + lane;
            float s = sS[row * TKT + lane] * scale;
            if (t > qpos || t >= seqmax) s = -1e30f;   // causal + bounds
            float rmax = s;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
            float m_old = sM[row], m_new = fmaxf(m_old, rmax);
            float p = __expf(s - m_new);
            float psum = p;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
            sP[row * KTS + lane] = __float2half(p);
            if (lane == 0) { float corr = __expf(m_old - m_new); sCorr[row] = corr; sL[row] = sL[row] * corr + psum; sM[row] = m_new; }
        }
        __syncthreads();
        // PV: O[q][hd] += P·V. warp owns n-tiles {warp, warp+8, ...}
        int oi = 0;
        for (int nt = warp; nt < nt_all; nt += 8, oi++) {
            int n0 = nt * 8;
            float cr0 = sCorr[lane / 4], cr1 = sCorr[8 + lane / 4];
            O[oi][0] *= cr0; O[oi][1] *= cr0; O[oi][2] *= cr1; O[oi][3] *= cr1;
            for (int k0 = 0; k0 < TKT; k0 += 16) {
                unsigned a0, a1, a2, a3, b0, b1;
                const __half* aptr = &sP[(lane % 16) * KTS + k0 + (lane / 16) * 8];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(aptr));
                const __half* bptr = &sV[(k0 + (lane % 16)) * HS + n0];
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "l"(bptr));
                asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(O[oi][0]), "+f"(O[oi][1]), "+f"(O[oi][2]), "+f"(O[oi][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }
    // epilogue: O[q][hd] / l[q]
    int oi = 0;
    for (int nt = warp; nt < nt_all; nt += 8, oi++) {
        int n0 = nt * 8, r0 = lane / 4, r1 = 8 + lane / 4, cc0 = (lane % 4) * 2, cc1 = cc0 + 1;
        int q0 = qbase + r0, q1 = qbase + r1;
        if (q0 < M) { float inv = 1.f / sL[r0]; out[(long)q0 * qdim + head * hd + n0 + cc0] = O[oi][0] * inv; out[(long)q0 * qdim + head * hd + n0 + cc1] = O[oi][1] * inv; }
        if (q1 < M) { float inv = 1.f / sL[r1]; out[(long)q1 * qdim + head * hd + n0 + cc0] = O[oi][2] * inv; out[(long)q1 * qdim + head * hd + n0 + cc1] = O[oi][3] * inv; }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) attention_flash_tc2(const float* q, const __half* kc,
    const __half* vc, float* out, int hd, int kvdim, int qdim, int pos_base, int M,
    int group, float scale) {
    flash_tc2_core(q, kc, vc, out, hd, kvdim, qdim, pos_base, M, group, scale);
}

// Flash-attention v3: 64 queries per block (tc2 uses 16), so each staged K/V tile is
// reused across 4× as many queries → 4× less KV global traffic, which is the win for
// large prompts where attention is O(M²) and outweighs the GEMMs. 4 warps, each owning
// 16 queries and running a self-contained QK + shared-softmax + register-O PV against
// the shared K/V tile. hd≤128 (O[16][4] register tile). Opt-in shared
// ~77KB (dynamic). grid=(n_head, ceil(M/64)), block=128.
#define QB3 64

__device__ __forceinline__ void flash_tc3_core(const float* q, const __half* kc,
    const __half* vc, float* out, int hd, int kvdim, int qdim, int pos_base, int M,
    int group, float scale) {
    int head = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    int qbase = blockIdx.y * QB3, wq0 = warp * 16, kvh = head / group;
    int seqmax = pos_base + min(M - 1, qbase + QB3 - 1) + 1;
    int nt_all = hd >> 3;
    extern __shared__ __half s3[];
    __half* sQ = s3;                          // QB3*256
    __half* sKt = sQ + QB3 * 256;             // 256*TKT
    __half* sV = sKt + 256 * TKT;             // TKT*256
    __half* sP = sV + TKT * 256;              // QB3*TKT
    float* sS = (float*)(sP + QB3 * TKT);     // QB3*TKT
    float* sM = sS + QB3 * TKT;               // QB3
    float* sL = sM + QB3;                     // QB3
    float* sCorr = sL + QB3;                  // QB3
    for (int idx = tid; idx < QB3 * hd; idx += 128) {
        int qq = idx / hd, ii = idx % hd, qi = qbase + qq;
        sQ[qq * 256 + ii] = __float2half((qi < M) ? q[(long)qi * qdim + head * hd + ii] : 0.f);
    }
    for (int i = tid; i < QB3; i += 128) { sM[i] = -1e30f; sL[i] = 0.f; }
    float O[16][4];
    #pragma unroll
    for (int a = 0; a < 16; a++) { O[a][0] = O[a][1] = O[a][2] = O[a][3] = 0.f; }
    __syncthreads();

    for (int t0 = 0; t0 < seqmax; t0 += TKT) {
        for (int idx = tid; idx < TKT * hd; idx += 128) {
            int kk = idx / hd, ii = idx % hd, t = t0 + kk;
            __half kv = __float2half(0.f), vv = __float2half(0.f);
            if (t < seqmax) { kv = kc[(long)t * kvdim + kvh * hd + ii]; vv = vc[(long)t * kvdim + kvh * hd + ii]; }
            sKt[ii * TKT + kk] = kv;
            sV[kk * 256 + ii] = vv;
        }
        __syncthreads();
        // QK: each warp scores its own 16 queries over all TKT keys → sS
        for (int nk = 0; nk < TKT; nk += 8) {
            float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (int k0 = 0; k0 < hd; k0 += 16) {
                unsigned a0, a1, a2, a3, b0, b1;
                const __half* aptr = &sQ[(wq0 + lane % 16) * 256 + k0 + (lane / 16) * 8];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(aptr));
                const __half* bptr = &sKt[(k0 + lane % 16) * TKT + nk];
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "l"(bptr));
                asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            int gi0 = wq0 + lane / 4, gi1 = wq0 + 8 + lane / 4, gj0 = (lane % 4) * 2, gj1 = gj0 + 1;
            sS[gi0 * TKT + nk + gj0] = c0; sS[gi0 * TKT + nk + gj1] = c1;
            sS[gi1 * TKT + nk + gj0] = c2; sS[gi1 * TKT + nk + gj1] = c3;
        }
        __syncthreads();
        // softmax: each warp handles its 16 rows (lane = key column)
        for (int rr = 0; rr < 16; rr++) {
            int row = wq0 + rr, qpos = pos_base + qbase + row, t = t0 + lane;
            float s = sS[row * TKT + lane] * scale;
            if (t > qpos || t >= seqmax) s = -1e30f;
            float rmax = s;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
            float m_old = sM[row], m_new = fmaxf(m_old, rmax);
            float p = __expf(s - m_new);
            float psum = p;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
            sP[row * TKT + lane] = __float2half(p);
            if (lane == 0) { float corr = __expf(m_old - m_new); sCorr[row] = corr; sL[row] = sL[row] * corr + psum; sM[row] = m_new; }
        }
        __syncthreads();
        // PV: each warp accumulates its 16 queries' O over hd n-tiles
        for (int nt = 0; nt < nt_all; nt++) {
            int n0 = nt * 8;
            float cr0 = sCorr[wq0 + lane / 4], cr1 = sCorr[wq0 + 8 + lane / 4];
            O[nt][0] *= cr0; O[nt][1] *= cr0; O[nt][2] *= cr1; O[nt][3] *= cr1;
            for (int k0 = 0; k0 < TKT; k0 += 16) {
                unsigned a0, a1, a2, a3, b0, b1;
                const __half* aptr = &sP[(wq0 + lane % 16) * TKT + k0 + (lane / 16) * 8];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(aptr));
                const __half* bptr = &sV[(k0 + lane % 16) * 256 + n0];
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "l"(bptr));
                asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(O[nt][0]), "+f"(O[nt][1]), "+f"(O[nt][2]), "+f"(O[nt][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        if (nt >= nt_all) break;
        int n0 = nt * 8, r0 = lane / 4, r1 = 8 + lane / 4, cc0 = (lane % 4) * 2, cc1 = cc0 + 1;
        int q0 = qbase + wq0 + r0, q1 = qbase + wq0 + r1;
        if (q0 < M) { float inv = 1.f / sL[wq0 + r0]; out[(long)q0 * qdim + head * hd + n0 + cc0] = O[nt][0] * inv; out[(long)q0 * qdim + head * hd + n0 + cc1] = O[nt][1] * inv; }
        if (q1 < M) { float inv = 1.f / sL[wq0 + r1]; out[(long)q1 * qdim + head * hd + n0 + cc0] = O[nt][2] * inv; out[(long)q1 * qdim + head * hd + n0 + cc1] = O[nt][3] * inv; }
    }
}

extern "C" __global__ void __launch_bounds__(128, 1) attention_flash_tc3(const float* q, const __half* kc,
    const __half* vc, float* out, int hd, int kvdim, int qdim, int pos_base, int M,
    int group, float scale) {
    flash_tc3_core(q, kc, vc, out, hd, kvdim, qdim, pos_base, M, group, scale);
}
"#;

/// Canonical entry names registered from this family.
pub const NAMES: &[&str] = &[
    "attention_part_g",
    "attention_merge",
    "attention_flash_tc",
    "attention_flash_tc_g",
    "attention_flash_tc2",
    "attention_flash_tc3",
];
