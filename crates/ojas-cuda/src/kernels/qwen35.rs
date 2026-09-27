//! Model glue for the CUDA qwen35 decoder (surya-2): the pieces `crate::qwen35::CudaSsm`
//! dispatches that no other family has in the shape it needs. Everything else the model takes
//! from an existing family (`gemm_f16`, `ssm`, `ops`, `vision`, `attn_bidir`). These are
//! CUDA-only names (`q35_*`), so they do not count in the Metal parity ledger.
//!
//! Where the CPU oracle (`ojas_cpu::cpu_ssm`, exact mode) and the Metal kernels differ, the
//! numerics follow the oracle, which is what this port is accepted against:
//!
//! * the KV cache is f32 (Metal and llama.cpp store f16), so attention sees the same K/V
//!   values the oracle does;
//! * QK-RMSNorm multiplies by `inv * w` in that association (`seg[i] *= inv * q_norm[i]`);
//! * the rope angle is `pos * (1 / powf(base, 2j/rd))`, with the sectioned stream selection a
//!   transcription of `cpu_ssm::mrope_sel` (itself Metal's `mrope_sel`);
//! * attention accumulates with unnormalized weights and divides once at the end.
//!
//! | entry           | buffers (in order)                                   | scalars (in order)                                                  | launch |
//! |-----------------|------------------------------------------------------|---------------------------------------------------------------------|--------|
//! | q35_embed_f16   | emb(half), ids(u32), x                                | d, M                                                                | 1-D over M*d |
//! | q35_split_lo    | x, lo                                                 | n                                                                   | 1-D over n |
//! | q35_qk_prep     | qfull, kin, vin, qw, kw, mpos(u32), q, kc, vc, ctl(u32) | hd, nh, nkv, rd, M, s0, s1, s2, s3, mode, theta(f32), eps(f32)      | grid ceil(M*(nh+2nkv)/4), block 128 |
//! | q35_attn_256    | q, kc, vc, out, po, pml, ctl(u32)                     | kvdim, M, group, n_head, chunk, causal, scale(f32)                  | grid [ceil(M/(32/group)), n_kv, nsplit], block 128 |
//! | q35_attn_64     | (same)                                                | (same)                                                              | (same) |
//! | q35_attn_merge  | po, pml, out                                          | hd, n_head, M, nsplit                                               | grid M*n_head, block 128 |
//! | q35_unpack4     | src, d0, d1, d2, d3                                   | n0, n1, n2, n3, M                                                   | 1-D over M*(n0+n1+n2+n3) |
//! | q35_swiglu_rows | gu, out                                               | ffn, M                                                              | 1-D over M*ffn |
//! | q35_split_half  | x, hi(half), lo(half)                                 | n                                                                   | 1-D over n |
//! | q35_vattn_x3_64 | q, khi, klo, vhi, vlo, out                            | kvdim, total, group, scale(f32), n_head, mtok                       | grid [ceil(mtok/64), n_head], block 128 |
//!
//! ## q35_attn_<hd>
//!
//! Causal (or bidirectional) GQA attention over an f32 KV cache, FlashAttention-style
//! (online softmax, K/V tiles in shared memory), on CUDA cores so no operand is rounded.
//! A block owns one KV head and 32 (query row, q head) pairs — `32/group` query rows times the
//! `group` q heads that share the KV head — so one K/V tile load feeds all of them. Each warp
//! holds 8 pairs with the head vector split across lanes (`hd/32` dims per lane), computes a
//! 4-key x 8-pair block of partial dots and folds the 32 partials across the warp with a
//! transpose-reduce (31 shuffles for 32 sums), after which lane `L` owns the score of pair
//! `L/4`, key `L%4`. `gridDim.z > 1` splits the key range (flash-decoding): every split
//! writes an unnormalized partial (`po`) plus its running max and denominator (`pml`), and
//! `q35_attn_merge` combines them. With one split the block writes `out` directly.
//!
//! `ctl = [base, total]` lives in device memory: `base` is the cache row of query 0, `total`
//! the number of valid cache rows; causal query `m` attends rows `[0, base + m]`,
//! bidirectional queries attend `[0, total)`. Reading them from the device is what lets a
//! decode step be recorded once as a CUDA graph and replayed at every position (with the
//! split count fixed at its maximum; splits past `total` exit writing an empty partial).
pub const BODY: &str = r#"
extern "C" __global__ void q35_embed_f16(const __half* emb, const unsigned* ids, float* x,
    unsigned d, unsigned M) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (unsigned long long)M * d) return;
    unsigned m = (unsigned)(g / d), i = (unsigned)(g % d);
    x[g] = __half2float(emb[(unsigned long long)ids[m] * d + i]);
}

// lo = x - f16(x): the residual the f16 tensor-core tile drops. gemm(x) + gemm(lo) carries
// ~22 bits of every activation instead of 11.
extern "C" __global__ void q35_split_lo(const float* x, float* lo, unsigned n) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    float v = x[g];
    lo[g] = v - __half2float(__float2half_rn(v));
}

// cpu_ssm::mrope_sel. Returns (position, exponent index) for rotary pair j.
__device__ __forceinline__ void q35_sel(const unsigned* p4, unsigned s0, unsigned s1,
    unsigned s2, unsigned s3, unsigned mode, unsigned j, unsigned* pos, unsigned* je) {
    unsigned sect = s0 + s1 + s2 + s3;
    if (mode == 0u || sect == 0u) { *pos = p4[0]; *je = j; return; }
    unsigned sector = j % sect, sel, start;
    if (mode == 2u) {
        unsigned r = sector % 3u;
        if      (r == 1u && sector < 3u * s1) sel = 1u;
        else if (r == 2u && sector < 3u * s2) sel = 2u;
        else if (r == 0u && sector < 3u * s0) sel = 0u;
        else sel = 3u;
        start = 0u;
    } else if (sector < s0)           { sel = 0u; start = 0u; }
    else if (sector < s0 + s1)        { sel = 1u; start = s0; }
    else if (sector < s0 + s1 + s2)   { sel = 2u; start = s0 + s1; }
    else                              { sel = 3u; start = s0 + s1 + s2; }
    *pos = p4[sel];
    *je = (mode == 3u) ? (sector - start) : j;
}

// Gated-attention prologue for M rows, one warp per (row, slot):
//   slots [0, nh)          q head: split out of qfull's per-head [q | gate], RMSNorm(qw), rope -> q
//   slots [nh, nh+nkv)     k head: RMSNorm(kw), rope -> kc row (base + m)
//   slots [nh+nkv, +nkv)   v head: copy -> vc row (base + m)
// Partial NEOX rope over the first rd dims (pairs (j, j + rd/2)); mpos is (t,h,w,e) per row.
extern "C" __global__ void q35_qk_prep(const float* qfull, const float* kin, const float* vin,
    const float* qw, const float* kw, const unsigned* mpos, float* q, float* kc, float* vc,
    const unsigned* ctl, unsigned hd, unsigned nh, unsigned nkv, unsigned rd, unsigned M,
    unsigned s0, unsigned s1, unsigned s2, unsigned s3, unsigned mode, float theta, float eps) {
    const unsigned base = ctl[0];
    __shared__ float sh[4][512];
    unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    unsigned slots = nh + 2u * nkv;
    unsigned long long gw = (unsigned long long)blockIdx.x * 4u + warp;
    if (gw >= (unsigned long long)M * slots) return;
    unsigned m = (unsigned)(gw / slots), s = (unsigned)(gw % slots);
    unsigned kvdim = nkv * hd;
    unsigned long long crow = (unsigned long long)(base + m) * kvdim;
    if (s >= nh + nkv) {
        unsigned h = s - nh - nkv;
        const float* src = vin + (unsigned long long)m * kvdim + h * hd;
        for (unsigned i = lane; i < hd; i += 32u) vc[crow + h * hd + i] = src[i];
        return;
    }
    const float* src; const float* w; float* dst;
    if (s < nh) {
        src = qfull + (unsigned long long)m * (2u * nh * hd) + s * 2u * hd;
        w = qw;
        dst = q + (unsigned long long)m * nh * hd + s * hd;
    } else {
        unsigned h = s - nh;
        src = kin + (unsigned long long)m * kvdim + h * hd;
        w = kw;
        dst = kc + crow + h * hd;
    }
    float* v = sh[warp];
    float ss = 0.0f;
    for (unsigned i = lane; i < hd; i += 32u) { float a = src[i]; v[i] = a; ss += a * a; }
    ss = warp_all_sum(ss);
    float inv = 1.0f / sqrtf(ss / (float)hd + eps);
    for (unsigned i = lane; i < hd; i += 32u) v[i] = v[i] * (inv * w[i]);
    __syncwarp();
    unsigned rf = rd / 2u;
    const unsigned* p4 = mpos + 4u * m;
    for (unsigned j = lane; j < rf; j += 32u) {
        unsigned p, je;
        q35_sel(p4, s0, s1, s2, s3, mode, j, &p, &je);
        float freq = 1.0f / powf(theta, 2.0f * (float)je / (float)rd);
        float ang = (float)p * freq;
        float sn, cs;
        sincosf(ang, &sn, &cs);
        float x0 = v[j], x1 = v[j + rf];
        v[j] = x0 * cs - x1 * sn;
        v[j + rf] = x0 * sn + x1 * cs;
    }
    __syncwarp();
    for (unsigned i = lane; i < hd; i += 32u) dst[i] = v[i];
}

// Split fused-projection rows [a | b | c | d] (row stride n0+n1+n2+n3) into four contiguous
// row-major buffers. The decoder's projections are one matmul each (SSM: qkv|z|alpha|beta,
// attention: q|k|v); a single-token step reads the parts in place, a multi-row chunk unpacks.
// n3 may be 0 (dst3 is then never written). total = M*(n0+n1+n2+n3).
extern "C" __global__ void q35_unpack4(const float* src, float* d0, float* d1, float* d2, float* d3,
    unsigned n0, unsigned n1, unsigned n2, unsigned n3, unsigned M) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned rs = n0 + n1 + n2 + n3;
    if (g >= (unsigned long long)M * rs) return;
    unsigned m = (unsigned)(g / rs), c = (unsigned)(g % rs);
    float v = src[g];
    if (c < n0) d0[(unsigned long long)m * n0 + c] = v;
    else if (c < n0 + n1) d1[(unsigned long long)m * n1 + (c - n0)] = v;
    else if (c < n0 + n1 + n2) d2[(unsigned long long)m * n2 + (c - n0 - n1)] = v;
    else d3[(unsigned long long)m * n3 + (c - n0 - n1 - n2)] = v;
}

// SwiGLU over fused [gate | up] rows (stride 2*ffn): out[m, i] = silu(gate) * up.
extern "C" __global__ void q35_swiglu_rows(const float* gu, float* out, unsigned ffn, unsigned M) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (unsigned long long)M * ffn) return;
    unsigned m = (unsigned)(g / ffn), i = (unsigned)(g % ffn);
    const float* r = gu + (unsigned long long)m * 2u * ffn;
    float v = r[i];
    out[g] = (v / (1.0f + expf(-v))) * r[ffn + i];
}

// Fold 32 per-lane partials across the warp: lane L returns the full sum of index L.
__device__ __forceinline__ float q35_treduce32(float (&v)[32], unsigned lane) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        bool up = (lane & 16u) != 0u;
        float send = up ? v[i] : v[i + 16];
        float keep = up ? v[i + 16] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bool up = (lane & 8u) != 0u;
        float send = up ? v[i] : v[i + 8];
        float keep = up ? v[i + 8] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        bool up = (lane & 4u) != 0u;
        float send = up ? v[i] : v[i + 4];
        float keep = up ? v[i + 4] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
    }
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        bool up = (lane & 2u) != 0u;
        float send = up ? v[i] : v[i + 2];
        float keep = up ? v[i + 2] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
    }
    {
        bool up = (lane & 1u) != 0u;
        float send = up ? v[0] : v[1];
        float keep = up ? v[1] : v[0];
        v[0] = keep + __shfl_xor_sync(0xffffffffu, send, 1);
    }
    return v[0];
}

#define QA_NW 4
#define QA_PPW 8
#define QA_BK 32
#define QA_NEG -1e30f

template <int HD>
__device__ __forceinline__ void q35_attn_core(const float* q, const float* kc, const float* vc,
    float* out, float* po, float* pml, const unsigned* ctl, unsigned kvdim, unsigned M,
    unsigned group, unsigned n_head, unsigned chunk, unsigned causal, float scale) {
    const unsigned base = ctl[0], total = ctl[1];
    constexpr int DPL = HD / 32;
    constexpr int SROW = HD + 4;
    __shared__ __align__(16) float sm[QA_BK * SROW];
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned kvh = blockIdx.y, z = blockIdx.z, nsplit = gridDim.z;
    const unsigned rpt = (QA_NW * QA_PPW) / group;          // query rows per block
    const unsigned q0 = blockIdx.x * rpt;
    const unsigned qdim = n_head * HD;

    float qr[QA_PPW][DPL], acc[QA_PPW][DPL];
    #pragma unroll
    for (int j = 0; j < QA_PPW; j++) {
        unsigned p = j * QA_NW + warp, r = q0 + p / group, h = kvh * group + p % group;
        bool ok = r < M;
        const float* qp = q + (unsigned long long)(ok ? r : 0u) * qdim + h * HD;
        #pragma unroll
        for (int i = 0; i < DPL; i++) { qr[j][i] = ok ? qp[lane + 32 * i] : 0.0f; acc[j][i] = 0.0f; }
    }
    // softmax state of pair (lane / 4), replicated across its four lanes
    const unsigned jl = lane >> 2;
    const unsigned pl = jl * QA_NW + warp;
    const unsigned rl = q0 + pl / group;                     // this lane's query row
    const unsigned lim = causal ? base + rl + 1u : total;    // keys [0, lim) visible to it
    float mrun = QA_NEG, lrun = 0.0f;

    const unsigned rlast = min(q0 + rpt, M);
    const unsigned kend = causal ? min(total, base + rlast) : total;
    const unsigned klo = z * chunk;
    const unsigned khi = min(klo + chunk, kend);
    const unsigned kv_off = kvh * HD;

    for (unsigned t0 = klo; t0 < khi; t0 += QA_BK) {
        __syncthreads();
        // K tile -> sm (float4, zero past khi)
        for (unsigned e = threadIdx.x; e < QA_BK * (HD / 4); e += blockDim.x) {
            unsigned kk = e / (HD / 4), c = (e % (HD / 4)) * 4u, t = t0 + kk;
            float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < khi) v4 = *(const float4*)(kc + (unsigned long long)t * kvdim + kv_off + c);
            *(float4*)(sm + kk * SROW + c) = v4;
        }
        __syncthreads();
        float sc[8];
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            float v[32];
            #pragma unroll
            for (int a = 0; a < 32; a++) v[a] = 0.0f;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                #pragma unroll
                for (int i = 0; i < DPL; i++) {
                    float kv = sm[(4 * g + k) * SROW + lane + 32 * i];
                    #pragma unroll
                    for (int j = 0; j < QA_PPW; j++) v[j * 4 + k] = fmaf(qr[j][i], kv, v[j * 4 + k]);
                }
            }
            float s = q35_treduce32(v, lane) * scale;
            unsigned t = t0 + 4u * g + (lane & 3u);
            sc[g] = (t < khi && t < lim) ? s : QA_NEG;
        }
        float tmax = sc[0];
        #pragma unroll
        for (int g = 1; g < 8; g++) tmax = fmaxf(tmax, sc[g]);
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
        float mnew = fmaxf(mrun, tmax);
        float corr = expf(mrun - mnew);
        float pr[8], psum = 0.0f;
        #pragma unroll
        for (int g = 0; g < 8; g++) { pr[g] = (sc[g] <= QA_NEG) ? 0.0f : expf(sc[g] - mnew); psum += pr[g]; }
        psum += __shfl_xor_sync(0xffffffffu, psum, 1);
        psum += __shfl_xor_sync(0xffffffffu, psum, 2);
        lrun = lrun * corr + psum;
        mrun = mnew;
        #pragma unroll
        for (int j = 0; j < QA_PPW; j++) {
            float cj = __shfl_sync(0xffffffffu, corr, 4 * j);
            #pragma unroll
            for (int i = 0; i < DPL; i++) acc[j][i] *= cj;
        }
        __syncthreads();
        // V tile -> sm
        for (unsigned e = threadIdx.x; e < QA_BK * (HD / 4); e += blockDim.x) {
            unsigned kk = e / (HD / 4), c = (e % (HD / 4)) * 4u, t = t0 + kk;
            float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < khi) v4 = *(const float4*)(vc + (unsigned long long)t * kvdim + kv_off + c);
            *(float4*)(sm + kk * SROW + c) = v4;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < QA_BK; kk++) {
            float vv[DPL];
            #pragma unroll
            for (int i = 0; i < DPL; i++) vv[i] = sm[kk * SROW + lane + 32 * i];
            #pragma unroll
            for (int j = 0; j < QA_PPW; j++) {
                float pj = __shfl_sync(0xffffffffu, pr[kk >> 2], 4 * j + (kk & 3));
                #pragma unroll
                for (int i = 0; i < DPL; i++) acc[j][i] = fmaf(pj, vv[i], acc[j][i]);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < QA_PPW; j++) {
        float lj = __shfl_sync(0xffffffffu, lrun, 4 * j);
        float mj = __shfl_sync(0xffffffffu, mrun, 4 * j);
        unsigned p = j * QA_NW + warp, r = q0 + p / group, h = kvh * group + p % group;
        if (r >= M) continue;
        if (nsplit == 1u) {
            float* op = out + (unsigned long long)r * qdim + h * HD;
            #pragma unroll
            for (int i = 0; i < DPL; i++) op[lane + 32 * i] = lj > 0.0f ? acc[j][i] / lj : 0.0f;
        } else {
            unsigned long long pi = ((unsigned long long)z * M + r) * n_head + h;
            float* op = po + pi * HD;
            #pragma unroll
            for (int i = 0; i < DPL; i++) op[lane + 32 * i] = acc[j][i];
            if (lane == 0u) { pml[2 * pi] = mj; pml[2 * pi + 1] = lj; }
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) q35_attn_256(const float* q, const float* kc,
    const float* vc, float* out, float* po, float* pml, const unsigned* ctl, unsigned kvdim,
    unsigned M, unsigned group, unsigned n_head, unsigned chunk,
    unsigned causal, float scale) {
    q35_attn_core<256>(q, kc, vc, out, po, pml, ctl, kvdim, M, group, n_head, chunk,
                       causal, scale);
}

extern "C" __global__ void __launch_bounds__(128) q35_attn_64(const float* q, const float* kc,
    const float* vc, float* out, float* po, float* pml, const unsigned* ctl, unsigned kvdim,
    unsigned M, unsigned group, unsigned n_head, unsigned chunk,
    unsigned causal, float scale) {
    q35_attn_core<64>(q, kc, vc, out, po, pml, ctl, kvdim, M, group, n_head, chunk,
                      causal, scale);
}

// ---- vision tower attention, split-f16 tensor cores ("x3") ----
//
// f32 in, f32 out, bidirectional. Every f16 operand of attention_m_mma_bidir_<hd> is carried
// as hi + lo (hi = f16(x), lo = f16(x - hi)), and each product keeps the three terms that
// matter: S = qh kh + ql kh + qh kl, O += ph vh + pl vh + ph vl — ~22 bits per operand
// instead of 11. That removes the f16 kernel's page-scale drift against the f32 oracle (row
// errors of several % at 16k patches) for ~3x its MMA work, still a fraction of the f32
// CUDA-core kernel's cost. K/V arrive pre-split by q35_split_half. Layout and launch are
// those of attention_m_mma_bidir_64 (grid [ceil(mtok/64), n_head], block 128), 32-key tiles.
extern "C" __global__ void q35_split_half(const float* x, __half* hi, __half* lo, unsigned n) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    float v = x[g];
    __half h = __float2half_rn(v);
    hi[g] = h;
    lo[g] = __float2half_rn(v - __half2float(h));
}

__device__ __forceinline__ unsigned q35_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<unsigned*>(&h);
}
__device__ __forceinline__ unsigned q35_h2_lo(float a, float b) {
    float ha = __half2float(__float2half_rn(a)), hb = __half2float(__float2half_rn(b));
    __half2 h = __floats2half2_rn(a - ha, b - hb);
    return *reinterpret_cast<unsigned*>(&h);
}
__device__ __forceinline__ void q35_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void q35_cp16(unsigned dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(valid ? 16 : 0));
}

extern "C" __global__ void __launch_bounds__(128) q35_vattn_x3_64(const float* __restrict__ q,
    const __half* __restrict__ khi, const __half* __restrict__ klo,
    const __half* __restrict__ vhi, const __half* __restrict__ vlo, float* __restrict__ out,
    unsigned kvdim, unsigned total, unsigned group, float scale, unsigned n_head, unsigned mtok) {
    constexpr int HD = 64, BKV = 32, NW = 4;
    constexpr int ST = HD + 8, KC = HD / 16, NS = BKV / 8, NO = HD / 8, CH = BKV * HD / 8, NT = NW * 32;
    __shared__ __align__(16) __half sKh[2][BKV * ST];
    __shared__ __align__(16) __half sKl[2][BKV * ST];
    __shared__ __align__(16) __half sVh[2][BKV * ST];
    __shared__ __align__(16) __half sVl[2][BKV * ST];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, c = lane & 3;
    const unsigned head = blockIdx.y, kvh = head / group;
    const unsigned R = n_head * HD;
    const unsigned qrow0 = blockIdx.x * (NW * 16) + warp * 16;
    const float sl2 = scale * 1.4426950408889634f;
    const unsigned long long hoff = (unsigned long long)kvh * HD;
    auto load_tile = [&](int buf, unsigned t0) {
        #pragma unroll
        for (int i = tid; i < CH; i += NT) {
            int r = i / (HD / 8), cc = (i % (HD / 8)) * 8;
            unsigned t = t0 + r;
            bool ok = t < total;
            unsigned long long off = (unsigned long long)(ok ? t : 0u) * kvdim + hoff + cc;
            q35_cp16((unsigned)__cvta_generic_to_shared(&sKh[buf][r * ST + cc]), khi + off, ok);
            q35_cp16((unsigned)__cvta_generic_to_shared(&sKl[buf][r * ST + cc]), klo + off, ok);
            q35_cp16((unsigned)__cvta_generic_to_shared(&sVh[buf][r * ST + cc]), vhi + off, ok);
            q35_cp16((unsigned)__cvta_generic_to_shared(&sVl[buf][r * ST + cc]), vlo + off, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    const unsigned ntile = (total + BKV - 1) / BKV;
    load_tile(0, 0u);

    unsigned qh[KC][4], ql[KC][4];
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
            qh[kk][0] = q35_h2(x00.x, x00.y); ql[kk][0] = q35_h2_lo(x00.x, x00.y);
            qh[kk][1] = q35_h2(x10.x, x10.y); ql[kk][1] = q35_h2_lo(x10.x, x10.y);
            qh[kk][2] = q35_h2(x01.x, x01.y); ql[kk][2] = q35_h2_lo(x01.x, x01.y);
            qh[kk][3] = q35_h2(x11.x, x11.y); ql[kk][3] = q35_h2_lo(x11.x, x11.y);
        }
    }
    float o[NO][4];
    #pragma unroll
    for (int n = 0; n < NO; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;

    for (unsigned j = 0; j < ntile; j++) {
        const int buf = j & 1;
        if (j + 1 < ntile) {
            load_tile(buf ^ 1, (j + 1) * BKV);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        float s[NS][4];
        #pragma unroll
        for (int n = 0; n < NS; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
        #pragma unroll
        for (int kk = 0; kk < KC; kk++) {
            #pragma unroll
            for (int n = 0; n < NS; n += 2) {
                int key = n * 8 + (lane & 7) + ((lane >> 4) << 3);
                int col = kk * 16 + ((lane >> 3) & 1) * 8;
                unsigned b0, b1, b2, b3, e0, e1, e2, e3;
                unsigned ah = (unsigned)__cvta_generic_to_shared(&sKh[buf][key * ST + col]);
                unsigned al = (unsigned)__cvta_generic_to_shared(&sKl[buf][key * ST + col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(ah));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(e0), "=r"(e1), "=r"(e2), "=r"(e3) : "r"(al));
                q35_mma(s[n], ql[kk], b0, b1);
                q35_mma(s[n], qh[kk], e0, e1);
                q35_mma(s[n], qh[kk], b0, b1);
                q35_mma(s[n + 1], ql[kk], b2, b3);
                q35_mma(s[n + 1], qh[kk], e2, e3);
                q35_mma(s[n + 1], qh[kk], b2, b3);
            }
        }
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
        #pragma unroll
        for (int kk = 0; kk < BKV / 16; kk++) {
            unsigned ph[4] = { q35_h2(s[2 * kk][0], s[2 * kk][1]), q35_h2(s[2 * kk][2], s[2 * kk][3]),
                               q35_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]), q35_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]) };
            unsigned pl[4] = { q35_h2_lo(s[2 * kk][0], s[2 * kk][1]), q35_h2_lo(s[2 * kk][2], s[2 * kk][3]),
                               q35_h2_lo(s[2 * kk + 1][0], s[2 * kk + 1][1]), q35_h2_lo(s[2 * kk + 1][2], s[2 * kk + 1][3]) };
            #pragma unroll
            for (int n = 0; n < NO; n += 2) {
                int key = kk * 16 + (lane & 15);
                int col = n * 8 + ((lane >> 4) << 3);
                unsigned b0, b1, b2, b3, e0, e1, e2, e3;
                unsigned ah = (unsigned)__cvta_generic_to_shared(&sVh[buf][key * ST + col]);
                unsigned al = (unsigned)__cvta_generic_to_shared(&sVl[buf][key * ST + col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(ah));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(e0), "=r"(e1), "=r"(e2), "=r"(e3) : "r"(al));
                q35_mma(o[n], pl, b0, b1);
                q35_mma(o[n], ph, e0, e1);
                q35_mma(o[n], ph, b0, b1);
                q35_mma(o[n + 1], pl, b2, b3);
                q35_mma(o[n + 1], ph, e2, e3);
                q35_mma(o[n + 1], ph, b2, b3);
            }
        }
        __syncthreads();
    }
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

// Combine nsplit partials: out = sum_z po_z e^(m_z - M) / sum_z l_z e^(m_z - M).
extern "C" __global__ void q35_attn_merge(const float* po, const float* pml, float* out,
    unsigned hd, unsigned n_head, unsigned M, unsigned nsplit) {
    unsigned mh = blockIdx.x, m = mh / n_head, h = mh % n_head;
    float mx = QA_NEG;
    for (unsigned z = 0; z < nsplit; z++) {
        unsigned long long pi = ((unsigned long long)z * M + m) * n_head + h;
        mx = fmaxf(mx, pml[2 * pi]);
    }
    float den = 0.0f;
    for (unsigned z = 0; z < nsplit; z++) {
        unsigned long long pi = ((unsigned long long)z * M + m) * n_head + h;
        den += pml[2 * pi + 1] * expf(pml[2 * pi] - mx);
    }
    for (unsigned d = threadIdx.x; d < hd; d += blockDim.x) {
        float num = 0.0f;
        for (unsigned z = 0; z < nsplit; z++) {
            unsigned long long pi = ((unsigned long long)z * M + m) * n_head + h;
            num += po[pi * hd + d] * expf(pml[2 * pi] - mx);
        }
        out[(unsigned long long)m * n_head * hd + h * hd + d] = den > 0.0f ? num / den : 0.0f;
    }
}
"#;

pub const NAMES: &[&str] = &[
    "q35_embed_f16",
    "q35_split_lo",
    "q35_qk_prep",
    "q35_attn_256",
    "q35_attn_64",
    "q35_attn_merge",
    "q35_split_half",
    "q35_vattn_x3_64",
    "q35_unpack4",
    "q35_swiglu_rows",
];

/// Query rows per `q35_attn_<hd>` block for a GQA group size (32 pairs per block).
pub fn attn_rows_per_block(group: u32) -> u32 {
    32 / group
}
