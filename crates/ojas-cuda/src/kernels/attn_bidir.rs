//! Bidirectional (non-causal) attention for the vision tower (surya-2 ViT: up to 16,384
//! patches, 12 heads, head dim 64 — 78% of the tower's FLOPs).
//!
//! Same entry names, argument order and meaning as Metal's `attention_m_bidir`
//! (`ojas-metal/src/kernels/attn_core.rs`) and the generated `attention_m_mma_bidir_<hd>`
//! (`ojas-metal/src/kernels/attn.rs`, `attn_mma_bidir_src`), so the tower's dispatch in
//! `ojas-models/src/decoder/vision.rs` maps one-to-one:
//!
//! ```text
//! q    f32  [mtok,  n_head*hd]   row m, head h at q[m*n_head*hd + h*hd]
//! kc   f16  [total, kvdim]       row t, kv-head kvh at kc[t*kvdim + kvh*hd]
//! vc   f16  [total, kvdim]       (K/V go f32 -> f16 in a copy before, as on Metal)
//! out  f32  [mtok,  n_head*hd]
//! hd, kvdim, total (the whole KV length — slot 6, where the causal kernels take base_pos),
//! group (n_head / n_kv_head), scale (f32 bits), n_head, mtok (query rows; MMA only)
//! ```
//!
//! out[m, h] = softmax_t(q[m,h] · k[t, h/group] * scale) · v[t, h/group], t over all of
//! [0, total) — no mask, no causality.
//!
//! `attention_m_bidir` — the streaming reference, a line-for-line port of Metal's: one
//! 256-thread block per (query, head), each warp walking every 8th key with an f32 online
//! softmax, then the eight partials merged. Q stays f32; any hd a multiple of 32 up to 512.
//! grid = [mtok*n_head, 1, 1], block = [256, 1, 1]. Every query streams the head's whole K and
//! V, ~824 GB per layer at 16k patches: the oracle and the fallback, not the production path.
//!
//! `attention_m_mma_bidir_{64,128}` — FlashAttention-2 on `mma.sync.m16n8k16` (f16 in, f32
//! accumulate). Each warp owns 16 query rows; Q (f32, rounded to f16 on load as Metal's
//! staged-Q tile does) lives in registers as A fragments, S = QKᵀ and O stay in registers,
//! P is re-packed from the S accumulators straight into A fragments (no shared round-trip),
//! and the online softmax runs in the exp2 domain, row max/sum reduced across the four lanes
//! that share a row. K/V tiles are double-buffered in shared memory with `cp.async`
//! (zero-filled past `total`), rows padded so `ldmatrix` is conflict-free; one K/V pass serves
//! BQ query rows (the 1/BQ traffic scaling Metal gets from its 32-row tile). hd is
//! compile-time, the `hd` argument accepted for signature parity and ignored as in Metal's
//! `hd_rt`. Requires sm_80+, kvdim % 8 == 0 and 16-byte aligned K/V bases (cp.async 16 B
//! chunks). grid = [ceil(mtok/BQ), n_head, 1] (q-tile fastest, so co-resident blocks walk the
//! same head's K/V and share it in L2), block = [32*warps, 1, 1]; see [`mma_bidir_launch`].
//! Static shared memory only (< 48 KB), so the plain `KernelRuntime::dispatch` launches it.
//!
//! Compiles against kernels::PRELUDE.
pub const BODY: &str = r#"
// ---- streaming bidirectional reference (Metal attention_m_bidir, attn_core.rs) ----
extern "C" __global__ void attention_m_bidir(const float* q, const __half* kc,
    const __half* vc, float* out, unsigned hd, unsigned kvdim, unsigned total,
    unsigned group, float scale, unsigned n_head) {
    __shared__ float qsh[512];
    __shared__ float tacc[8 * 512];
    __shared__ float tm[8];
    __shared__ float tl[8];
    unsigned tg = blockIdx.x, lid = threadIdx.x, ts = blockDim.x;
    unsigned m = tg / n_head, head = tg % n_head, kvh = head / group;
    unsigned seq = total;                  // bidirectional: attend to every position
    unsigned R = n_head * hd;
    unsigned sgid = lid / 32u, lane = lid % 32u, nsg = ts / 32u;
    const float* qh = q + (unsigned long long)m * R + head * hd;
    for (unsigned i = lid; i < hd; i += ts) qsh[i] = qh[i];
    __syncthreads();
    float mi = -1e30f, li = 0.0f;
    float acc[16];
    #pragma unroll
    for (int c = 0; c < 16; c++) acc[c] = 0.0f;
    unsigned nch = hd / 32u;
    for (unsigned t = sgid; t < seq; t += nsg) {
        const __half* kt = kc + (unsigned long long)t * kvdim + kvh * hd;
        float sv = 0.0f;
        for (unsigned i = lane; i < hd; i += 32u) sv += qsh[i] * __half2float(kt[i]);
        sv = warp_all_sum(sv) * scale;
        float mn = fmaxf(mi, sv); float corr = expf(mi - mn); float pw = expf(sv - mn);
        li = li * corr + pw;
        const __half* vt = vc + (unsigned long long)t * kvdim + kvh * hd;
        #pragma unroll
        for (unsigned c = 0u; c < 16u; c++)
            if (c < nch) acc[c] = acc[c] * corr + pw * __half2float(vt[lane + 32u * c]);
        mi = mn;
    }
    if (lane == 0u) { tm[sgid] = mi; tl[sgid] = li; }
    #pragma unroll
    for (unsigned c = 0u; c < 16u; c++) if (c < nch) tacc[sgid * hd + lane + 32u * c] = acc[c];
    __syncthreads();
    float gm = -1e30f; for (unsigned j = 0u; j < nsg; j++) gm = fmaxf(gm, tm[j]);
    float gl = 0.0f; for (unsigned j = 0u; j < nsg; j++) gl += tl[j] * expf(tm[j] - gm);
    for (unsigned i = lid; i < hd; i += ts) {
        float o = 0.0f;
        for (unsigned j = 0u; j < nsg; j++) o += tacc[j * hd + i] * expf(tm[j] - gm);
        out[(unsigned long long)m * R + head * hd + i] = o / gl;
    }
}

// ---- tensor-core flash attention, bidirectional (FA2, register-resident S/P/O) ----
__device__ __forceinline__ unsigned bidir_pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}

__device__ __forceinline__ void bidir_cp16(unsigned dst, const void* src, bool valid) {
    // src-size 0 -> the 16 destination bytes are zero-filled (rows past `total`)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "r"(valid ? 16 : 0));
}

// HD: head dim (compile-time). BKV: keys per K/V tile. NW: warps per block (16 rows each).
template <int HD, int BKV, int NW>
__device__ __forceinline__ void mma_bidir_core(const float* __restrict__ q,
    const __half* __restrict__ kc, const __half* __restrict__ vc, float* __restrict__ out,
    unsigned kvdim, unsigned total, unsigned group, float scale, unsigned n_head,
    unsigned mtok) {
    constexpr int ST = HD + 8;            // padded smem row stride (halves): 16 B aligned,
                                          // 8 consecutive rows hit disjoint bank groups
    constexpr int KC = HD / 16;           // k-chunks of QK^T
    constexpr int NS = BKV / 8;           // S n-tiles per warp (keys)
    constexpr int NO = HD / 8;            // O n-tiles per warp (head dims)
    constexpr int CH = BKV * HD / 8;      // 16-byte chunks per K (or V) tile
    constexpr int NT = NW * 32;
    __shared__ __align__(16) __half sK[2][BKV * ST];
    __shared__ __align__(16) __half sV[2][BKV * ST];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, c = lane & 3;          // mma fragment row group / column pair
    const unsigned head = blockIdx.y, kvh = head / group;
    const unsigned R = n_head * HD;
    const unsigned qrow0 = blockIdx.x * (NW * 16) + warp * 16;
    const float sl2 = scale * 1.4426950408889634f;  // softmax in the exp2 domain

    const __half* kbase = kc + kvh * HD;
    const __half* vbase = vc + kvh * HD;
    auto load_tile = [&](int buf, unsigned t0) {
        #pragma unroll
        for (int i = tid; i < CH; i += NT) {
            int r = i / (HD / 8), cc = (i % (HD / 8)) * 8;
            unsigned t = t0 + r;
            bool ok = t < total;
            unsigned long long off = (unsigned long long)(ok ? t : 0u) * kvdim + cc;
            bidir_cp16((unsigned)__cvta_generic_to_shared(&sK[buf][r * ST + cc]), kbase + off, ok);
            bidir_cp16((unsigned)__cvta_generic_to_shared(&sV[buf][r * ST + cc]), vbase + off, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const unsigned ntile = (total + BKV - 1) / BKV;
    load_tile(0, 0u);

    // Q -> f16 A fragments, straight from global (rows past mtok are zero).
    unsigned qa[KC][4];
    {
        const unsigned r0 = qrow0 + g, r1 = qrow0 + g + 8;
        const float* q0 = q + (unsigned long long)r0 * R + head * HD;
        const float* q1 = q + (unsigned long long)r1 * R + head * HD;
        #pragma unroll
        for (int kk = 0; kk < KC; kk++) {
            int col = kk * 16 + c * 2;
            float2 x00 = r0 < mtok ? *reinterpret_cast<const float2*>(q0 + col) : make_float2(0.f, 0.f);
            float2 x10 = r1 < mtok ? *reinterpret_cast<const float2*>(q1 + col) : make_float2(0.f, 0.f);
            float2 x01 = r0 < mtok ? *reinterpret_cast<const float2*>(q0 + col + 8) : make_float2(0.f, 0.f);
            float2 x11 = r1 < mtok ? *reinterpret_cast<const float2*>(q1 + col + 8) : make_float2(0.f, 0.f);
            qa[kk][0] = bidir_pack_h2(x00.x, x00.y);
            qa[kk][1] = bidir_pack_h2(x10.x, x10.y);
            qa[kk][2] = bidir_pack_h2(x01.x, x01.y);
            qa[kk][3] = bidir_pack_h2(x11.x, x11.y);
        }
    }

    float o[NO][4];
    #pragma unroll
    for (int n = 0; n < NO; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m0 = -1e30f, m1 = -1e30f;       // running max (rows g, g+8), log2 units
    float l0 = 0.f, l1 = 0.f;             // running denominators, THIS lane's columns only

    for (unsigned j = 0; j < ntile; j++) {
        const int buf = j & 1;
        if (j + 1 < ntile) {
            load_tile(buf ^ 1, (j + 1) * BKV);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const __half* K = sK[buf];
        const __half* V = sV[buf];

        // S = Q K^T  (16 x BKV per warp)
        float s[NS][4];
        #pragma unroll
        for (int n = 0; n < NS; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
        #pragma unroll
        for (int kk = 0; kk < KC; kk++) {
            #pragma unroll
            for (int n = 0; n < NS; n += 2) {
                // x4: [keys n*8..+8 | k lo], [.. | k hi], [keys n*8+8..+16 | k lo], [.. | k hi]
                int key = n * 8 + (lane & 7) + ((lane >> 4) << 3);
                int col = kk * 16 + ((lane >> 3) & 1) * 8;
                unsigned b0, b1, b2, b3;
                unsigned addr = (unsigned)__cvta_generic_to_shared(&K[key * ST + col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(s[n][0]), "+f"(s[n][1]), "+f"(s[n][2]), "+f"(s[n][3])
                    : "r"(qa[kk][0]), "r"(qa[kk][1]), "r"(qa[kk][2]), "r"(qa[kk][3]), "r"(b0), "r"(b1));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(s[n + 1][0]), "+f"(s[n + 1][1]), "+f"(s[n + 1][2]), "+f"(s[n + 1][3])
                    : "r"(qa[kk][0]), "r"(qa[kk][1]), "r"(qa[kk][2]), "r"(qa[kk][3]), "r"(b2), "r"(b3));
            }
        }

        // online softmax (exp2 domain). Only the last tile can hold keys >= total.
        const unsigned t0 = j * BKV;
        const bool tail = t0 + BKV > total;
        float mx0 = -1e30f, mx1 = -1e30f;
        #pragma unroll
        for (int n = 0; n < NS; n++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                float v = s[n][e] * sl2;
                if (tail && t0 + n * 8 + c * 2 + (e & 1) >= total) v = -__int_as_float(0x7f800000);
                s[n][e] = v;
            }
            mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
            mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float cr0 = exp2f(m0 - mn0), cr1 = exp2f(m1 - mn1);
        m0 = mn0; m1 = mn1;
        float ps0 = 0.f, ps1 = 0.f;
        #pragma unroll
        for (int n = 0; n < NS; n++) {
            s[n][0] = exp2f(s[n][0] - mn0); s[n][1] = exp2f(s[n][1] - mn0);
            s[n][2] = exp2f(s[n][2] - mn1); s[n][3] = exp2f(s[n][3] - mn1);
            ps0 += s[n][0] + s[n][1];
            ps1 += s[n][2] + s[n][3];
        }
        l0 = l0 * cr0 + ps0;
        l1 = l1 * cr1 + ps1;
        #pragma unroll
        for (int n = 0; n < NO; n++) { o[n][0] *= cr0; o[n][1] *= cr0; o[n][2] *= cr1; o[n][3] *= cr1; }

        // O += P V. The S accumulators of n-tiles (2k, 2k+1) are the A fragment of
        // k-chunk k (rows g/g+8, keys 16k + 2c.. and 16k + 8 + 2c..).
        #pragma unroll
        for (int kk = 0; kk < BKV / 16; kk++) {
            unsigned a0 = bidir_pack_h2(s[2 * kk][0], s[2 * kk][1]);
            unsigned a1 = bidir_pack_h2(s[2 * kk][2], s[2 * kk][3]);
            unsigned a2 = bidir_pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            unsigned a3 = bidir_pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
            #pragma unroll
            for (int n = 0; n < NO; n += 2) {
                // x4.trans: [keys lo | dims n*8], [keys hi | n*8], [keys lo | n*8+8], [keys hi | n*8+8]
                int key = kk * 16 + (lane & 15);
                int col = n * 8 + ((lane >> 4) << 3);
                unsigned b0, b1, b2, b3;
                unsigned addr = (unsigned)__cvta_generic_to_shared(&V[key * ST + col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(o[n][0]), "+f"(o[n][1]), "+f"(o[n][2]), "+f"(o[n][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(o[n + 1][0]), "+f"(o[n + 1][1]), "+f"(o[n + 1][2]), "+f"(o[n + 1][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));
            }
        }
        __syncthreads();                  // this buffer is refilled by the next prefetch
    }

    // epilogue: full row sums across the quad, then O / l
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = 1.f / l0, i1 = 1.f / l1;
    const unsigned r0 = qrow0 + g, r1 = qrow0 + g + 8;
    float* o0 = out + (unsigned long long)r0 * R + head * HD + c * 2;
    float* o1 = out + (unsigned long long)r1 * R + head * HD + c * 2;
    #pragma unroll
    for (int n = 0; n < NO; n++) {
        if (r0 < mtok) *reinterpret_cast<float2*>(o0 + n * 8) = make_float2(o[n][0] * i0, o[n][1] * i0);
        if (r1 < mtok) *reinterpret_cast<float2*>(o1 + n * 8) = make_float2(o[n][2] * i1, o[n][3] * i1);
    }
}

// hd 64: 64-key tiles, 4 warps (64 query rows / block), 36.9 KB static shared.
extern "C" __global__ void __launch_bounds__(128) attention_m_mma_bidir_64(const float* q,
    const __half* kc, const __half* vc, float* out, unsigned hd, unsigned kvdim,
    unsigned total, unsigned group, float scale, unsigned n_head, unsigned mtok) {
    (void)hd;
    mma_bidir_core<64, 64, 4>(q, kc, vc, out, kvdim, total, group, scale, n_head, mtok);
}

// hd 128: 32-key tiles, 4 warps (64 query rows / block), 34.8 KB static shared.
extern "C" __global__ void __launch_bounds__(128) attention_m_mma_bidir_128(const float* q,
    const __half* kc, const __half* vc, float* out, unsigned hd, unsigned kvdim,
    unsigned total, unsigned group, float scale, unsigned n_head, unsigned mtok) {
    (void)hd;
    mma_bidir_core<128, 32, 4>(q, kc, vc, out, kvdim, total, group, scale, n_head, mtok);
}
"#;

/// Canonical entry names registered from this family.
pub const NAMES: &[&str] = &[
    "attention_m_bidir",
    "attention_m_mma_bidir_64",
    "attention_m_mma_bidir_128",
];

/// Head dims with an `attention_m_mma_bidir_<hd>` entry (Metal: `ATTN_BIDIR_HD`).
pub const MMA_BIDIR_HD: &[u32] = &[64, 128];

/// Query rows per block of every `attention_m_mma_bidir_<hd>` entry.
pub const MMA_BIDIR_BQ: u32 = 64;

/// `(grid, block)` for `attention_m_mma_bidir_<hd>` over `mtok` query rows and `n_head` heads.
pub fn mma_bidir_launch(n_head: u32, mtok: u32) -> ([u32; 3], [u32; 3]) {
    ([mtok.div_ceil(MMA_BIDIR_BQ), n_head, 1], [128, 1, 1])
}

/// `(grid, block)` for `attention_m_bidir` (one 256-thread block per query row and head).
pub fn bidir_launch(n_head: u32, mtok: u32) -> ([u32; 3], [u32; 3]) {
    ([mtok * n_head, 1, 1], [256, 1, 1])
}
