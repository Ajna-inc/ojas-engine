//! CNN training kernels (`cnn_train` family): the Metal twin of
//! `ojas-cuda/src/kernels/learn.rs`, ported primitive for primitive so
//! `ojas-learn`'s CPU reference gates this backend exactly as it gates CUDA.
//! Contiguous row-major f32 everywhere (not the padded NHWC inference layout).
//!
//! Convention: buffers first at `[[buffer(0..)]]`, then scalar constants each
//! at their own following buffer index (`set_bytes` per constant — the same
//! `KernelRuntime::dispatch(bufs, consts, grid, block)` contract CUDA and
//! Vulkan use), float constants declared `constant float&` even though they
//! cross as `f32::to_bits()` u32 words: `set_bytes` copies raw bytes, so the
//! MSL parameter type is what actually reinterprets them.
//!
//! `accum` = 1 adds into the output instead of overwriting it. `grid`/`block`
//! below are `dispatch_thread_groups` arguments: grid = threadgroup count,
//! block = threads per threadgroup — CUDA's blockIdx/blockDim, not a global
//! thread id, so every kernel keeps CUDA's blockIdx-style indexing exactly.
//!
//! GEMM is the CPU-parity naive tile only, with no simdgroup_matrix tensor-core path,
//! mirroring the CUDA side's history: kernels first (`f498470`/`fa648fd`), tensor cores
//! as a separate pass (`937063d`). `reduce_to` is one general kernel with no row-split
//! fast path, favouring simplicity over CUDA's two-kernel split.

pub const BODY: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint LEARN_MAXR = 6;

// MSL has no builtin erf. Abramowitz & Stegun 7.1.26 (max abs error 1.5e-7,
// well under the 1e-4 relative gradcheck tolerance this backend is held to).
static inline float learn_erf(float x) {
    float s = x < 0.0f ? -1.0f : 1.0f;
    float ax = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return s * y;
}

// unary codes: 0 relu | 1 sigmoid | 2 silu | 3 tanh | 4 gelu (erf) | 5 exp | 6 log | 7 neg | 8 sqrt
static inline float learn_unary_f(float x, uint op) {
    switch (op) {
    case 0: return x > 0.0f ? x : 0.0f;
    case 1: return 1.0f / (1.0f + exp(-x));
    case 2: return x / (1.0f + exp(-x));
    case 3: return tanh(x);
    case 4: return 0.5f * x * (1.0f + learn_erf(x * 0.70710678118654752f));
    case 5: return exp(x);
    case 6: return log(x);
    case 7: return -x;
    default: return sqrt(x);
    }
}

// d f / d x, from the input x and the forward output y
static inline float learn_unary_d(float x, float y, uint op) {
    switch (op) {
    case 0: return x > 0.0f ? 1.0f : 0.0f;
    case 1: return y * (1.0f - y);
    case 2: { float s = 1.0f / (1.0f + exp(-x)); return s * (1.0f + x * (1.0f - s)); }
    case 3: return 1.0f - y * y;
    case 4: return 0.5f * (1.0f + learn_erf(x * 0.70710678118654752f)) + x * 0.3989422804014327f * exp(-0.5f * x * x);
    case 5: return y;
    case 6: return 1.0f / x;
    case 7: return -1.0f;
    default: return 0.5f / y;
    }
}

// binary codes: 0 add | 1 sub | 2 mul | 3 div | 4 max | 5 min
static inline float learn_binary_f(float a, float b, uint op) {
    switch (op) {
    case 0: return a + b;
    case 1: return a - b;
    case 2: return a * b;
    case 3: return a / b;
    case 4: return a >= b ? a : b;
    default: return a <= b ? a : b;
    }
}

// d(a op b)/da (which=0) or /db (which=1); ties of max/min go to a
static inline float learn_binary_d(float a, float b, uint op, uint which) {
    switch (op) {
    case 0: return 1.0f;
    case 1: return which ? -1.0f : 1.0f;
    case 2: return which ? a : b;
    case 3: return which ? -a / (b * b) : 1.0f / b;
    case 4: return which ? (b > a ? 1.0f : 0.0f) : (a >= b ? 1.0f : 0.0f);
    default: return which ? (b < a ? 1.0f : 0.0f) : (a <= b ? 1.0f : 0.0f);
    }
}

static inline void learn_store(device float* y, long i, float v, uint accum) {
    if (accum) y[i] += v; else y[i] = v;
}

kernel void learn_fill(device float* y [[buffer(0)]], constant uint& n [[buffer(1)]], constant float& v [[buffer(2)]],
                        uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i < n) y[i] = v;
}

kernel void learn_axpby(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                         constant float& a [[buffer(3)]], constant float& b [[buffer(4)]],
                         uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i < n) y[i] = a * x[i] + b * y[i];
}

kernel void learn_unary(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]], constant uint& op [[buffer(3)]],
                         uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i < n) y[i] = learn_unary_f(x[i], op);
}

kernel void learn_unary_bwd(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const float* dy [[buffer(2)]], device float* dx [[buffer(3)]],
                             constant uint& n [[buffer(4)]], constant uint& op [[buffer(5)]], constant uint& accum [[buffer(6)]],
                             uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i < n) learn_store(dx, i, learn_unary_d(x[i], y[i], op) * dy[i], accum);
}

// Broadcast binary: y (contiguous, dims) = a op b, a/b read through strides
// over the output dims (stride 0 = broadcast axis). Rank <= 6, right-aligned.
#define LEARN_BCAST_PARAMS constant uint& n [[buffer(NB+0)]], constant uint& op [[buffer(NB+1)]], \
    constant uint& d0 [[buffer(NB+2)]], constant uint& d1 [[buffer(NB+3)]], constant uint& d2 [[buffer(NB+4)]], \
    constant uint& d3 [[buffer(NB+5)]], constant uint& d4 [[buffer(NB+6)]], constant uint& d5 [[buffer(NB+7)]], \
    constant uint& sa0 [[buffer(NB+8)]], constant uint& sa1 [[buffer(NB+9)]], constant uint& sa2 [[buffer(NB+10)]], \
    constant uint& sa3 [[buffer(NB+11)]], constant uint& sa4 [[buffer(NB+12)]], constant uint& sa5 [[buffer(NB+13)]], \
    constant uint& sb0 [[buffer(NB+14)]], constant uint& sb1 [[buffer(NB+15)]], constant uint& sb2 [[buffer(NB+16)]], \
    constant uint& sb3 [[buffer(NB+17)]], constant uint& sb4 [[buffer(NB+18)]], constant uint& sb5 [[buffer(NB+19)]]
#define LEARN_BCAST_OFFSETS \
    uint dims[6] = {d0, d1, d2, d3, d4, d5}; \
    uint sa[6] = {sa0, sa1, sa2, sa3, sa4, sa5}; \
    uint sb[6] = {sb0, sb1, sb2, sb3, sb4, sb5}; \
    long oa = 0, ob = 0; uint rem = i; \
    for (int d = 5; d >= 0; d--) { uint c = rem % dims[d]; rem /= dims[d]; oa += (long)c * sa[d]; ob += (long)c * sb[d]; }

#define NB 3
kernel void learn_binary(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* y [[buffer(2)]],
                          LEARN_BCAST_PARAMS,
                          uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i >= n) return;
    LEARN_BCAST_OFFSETS
    y[i] = learn_binary_f(a[oa], b[ob], op);
}
#undef NB

#define NB 5
kernel void learn_binary_grad(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const float* dy [[buffer(2)]], device float* t [[buffer(3)]],
                               constant uint& which [[buffer(4)]], LEARN_BCAST_PARAMS,
                               uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i >= n) return;
    LEARN_BCAST_OFFSETS
    t[i] = dy[i] * learn_binary_d(a[oa], b[ob], op, which);
}
#undef NB

// Sum t (full dims, contiguous) over axes flagged in `red` (bit d = axis d of
// the 6 right-aligned dims) into g (kept axes, contiguous). One thread per
// output element, reduced in index order -> deterministic.
kernel void learn_reduce_to(device const float* t [[buffer(0)]], device float* g [[buffer(1)]],
                             constant uint& m [[buffer(2)]], constant uint& accum [[buffer(3)]], constant uint& red [[buffer(4)]],
                             constant uint& d0 [[buffer(5)]], constant uint& d1 [[buffer(6)]], constant uint& d2 [[buffer(7)]],
                             constant uint& d3 [[buffer(8)]], constant uint& d4 [[buffer(9)]], constant uint& d5 [[buffer(10)]],
                             uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint o = gid.x * bs.x + tid.x;
    if (o >= m) return;
    uint dims[6] = {d0, d1, d2, d3, d4, d5};
    long fs[6]; long s = 1;
    for (int d = 5; d >= 0; d--) { fs[d] = s; s *= dims[d]; }
    long base = 0; uint rem = o; long R = 1;
    for (int d = 5; d >= 0; d--) {
        if (red >> d & 1) { R *= dims[d]; continue; }
        uint c = rem % dims[d]; rem /= dims[d]; base += (long)c * fs[d];
    }
    float acc = 0.0f;
    for (long r = 0; r < R; r++) {
        long rr = r, off = base;
        for (int d = 5; d >= 0; d--) {
            if (!(red >> d & 1)) continue;
            uint c = (uint)(rr % dims[d]); rr /= dims[d]; off += (long)c * fs[d];
        }
        acc += t[off];
    }
    learn_store(g, o, acc, accum);
}

// y[yoff + Sum c*ys] (+)= x[xoff + Sum c*xs] over the index space `dims`
// (rank <= 6, right-aligned). Permute, slice, concat and their backwards.
kernel void learn_copy_strided(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                 constant uint& n [[buffer(2)]], constant uint& accum [[buffer(3)]],
                                 constant uint& d0 [[buffer(4)]], constant uint& d1 [[buffer(5)]], constant uint& d2 [[buffer(6)]],
                                 constant uint& d3 [[buffer(7)]], constant uint& d4 [[buffer(8)]], constant uint& d5 [[buffer(9)]],
                                 constant uint& xs0 [[buffer(10)]], constant uint& xs1 [[buffer(11)]], constant uint& xs2 [[buffer(12)]],
                                 constant uint& xs3 [[buffer(13)]], constant uint& xs4 [[buffer(14)]], constant uint& xs5 [[buffer(15)]],
                                 constant uint& ys0 [[buffer(16)]], constant uint& ys1 [[buffer(17)]], constant uint& ys2 [[buffer(18)]],
                                 constant uint& ys3 [[buffer(19)]], constant uint& ys4 [[buffer(20)]], constant uint& ys5 [[buffer(21)]],
                                 constant uint& xoff [[buffer(22)]], constant uint& yoff [[buffer(23)]],
                                 uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i >= n) return;
    uint dims[6] = {d0, d1, d2, d3, d4, d5};
    uint xs[6] = {xs0, xs1, xs2, xs3, xs4, xs5};
    uint ys[6] = {ys0, ys1, ys2, ys3, ys4, ys5};
    long ox = xoff, oy = yoff; uint rem = i;
    for (int d = 5; d >= 0; d--) { uint c = rem % dims[d]; rem /= dims[d]; ox += (long)c * xs[d]; oy += (long)c * ys[d]; }
    learn_store(y, oy, x[ox], accum);
}

// Batched row-major GEMM: C = alpha*op(A)*op(B) + beta*C, op(A) MxK, op(B) KxN.
// ta: A stored KxM; tb: B stored NxK. Batch z: A += z*sa, B += z*sb, C += z*sc.
// 64x64 tile, k step 16, 256 threads x 4x4 outputs -- the CPU-parity oracle,
// not the fast path; simdgroup_matrix tensor cores are a follow-up.
kernel void learn_gemm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
                        constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
                        constant uint& ta [[buffer(6)]], constant uint& tb [[buffer(7)]],
                        constant uint& sa [[buffer(8)]], constant uint& sb [[buffer(9)]], constant uint& sc [[buffer(10)]],
                        constant float& alpha [[buffer(11)]], constant float& beta [[buffer(12)]],
                        uint3 gridp [[threadgroup_position_in_grid]], uint3 tidp [[thread_position_in_threadgroup]]) {
    threadgroup float As[16][68];
    threadgroup float Bs[16][68];
    uint z = gridp.z;
    device const float* Az = A + (long)z * sa;
    device const float* Bz = B + (long)z * sb;
    device float* Cz = C + (long)z * sc;
    uint tidx = tidp.y * 16 + tidp.x; // linear 0..255 over a 16x16 threadgroup
    uint tx = tidx & 15, ty = tidx >> 4;
    uint row0 = gridp.y * 64, col0 = gridp.x * 64;
    float acc[4][4];
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += 16) {
        for (uint i = 0; i < 4; i++) {
            uint e = tidx + i * 256;
            uint r, kk;
            if (ta) { kk = e >> 6; r = e & 63; } else { r = e >> 4; kk = e & 15; }
            uint gr = row0 + r, gk = k0 + kk;
            float v = 0.0f;
            if (gr < M && gk < K) v = ta ? Az[(long)gk * M + gr] : Az[(long)gr * K + gk];
            As[kk][r] = v;
        }
        for (uint i = 0; i < 4; i++) {
            uint e = tidx + i * 256;
            uint c, kk;
            if (tb) { c = e >> 4; kk = e & 15; } else { kk = e >> 6; c = e & 63; }
            uint gc = col0 + c, gk = k0 + kk;
            float v = 0.0f;
            if (gc < N && gk < K) v = tb ? Bz[(long)gc * K + gk] : Bz[(long)gk * N + gc];
            Bs[kk][c] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 16; kk++) {
            float a[4], b[4];
            for (uint i = 0; i < 4; i++) a[i] = As[kk][ty * 4 + i];
            for (uint j = 0; j < 4; j++) b[j] = Bs[kk][tx * 4 + j];
            for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; i++) {
        uint r = row0 + ty * 4 + i;
        if (r >= M) continue;
        for (uint j = 0; j < 4; j++) {
            uint c = col0 + tx * 4 + j;
            if (c >= N) continue;
            long o = (long)r * N + c;
            float v = alpha * acc[i][j];
            Cz[o] = beta == 0.0f ? v : v + beta * Cz[o];
        }
    }
}

// Block-wide reduce: every thread's value into threadgroup memory, then a
// halving tree (bs must be a power of two -- every dispatch below uses 256
// or 1024). Simple and correct over a simdgroup-shuffle fast path.
//
// The closing barrier matters: several callers (layernorm_bwd, bn_wgrad) call this
// twice back to back over the same `sh`. Without it a thread that finishes early can
// start the second call's `sh[tid] = v` before a slower thread has read `sh[0]` for
// the first call's result, racing a write against a read of the same buffer.
static inline float learn_block_sum(float v, threadgroup float* sh, uint tid, uint bs) {
    sh[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline float learn_block_max(float v, threadgroup float* sh, uint tid, uint bs) {
    sh[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = max(sh[tid], sh[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// Row softmax over the last axis. One threadgroup per row, 256 threads.
kernel void learn_softmax(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                            constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                            uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint bs [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    long r = gid;
    device const float* xr = x + r * cols;
    device float* yr = y + r * cols;
    float m = -3.0e38f;
    for (uint c = tid; c < cols; c += bs) m = max(m, xr[c]);
    m = learn_block_max(m, sh, tid, bs);
    float s = 0.0f;
    for (uint c = tid; c < cols; c += bs) s += exp(xr[c] - m);
    s = learn_block_sum(s, sh, tid, bs);
    float inv = 1.0f / s;
    for (uint c = tid; c < cols; c += bs) yr[c] = exp(xr[c] - m) * inv;
}

// dx (+)= y (dy - Sum y*dy). One threadgroup per row.
kernel void learn_softmax_bwd(device const float* y [[buffer(0)]], device const float* dy [[buffer(1)]], device float* dx [[buffer(2)]],
                                constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]], constant uint& accum [[buffer(5)]],
                                uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint bs [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    long r = gid;
    device const float* yr = y + r * cols;
    device const float* dr = dy + r * cols;
    float s = 0.0f;
    for (uint c = tid; c < cols; c += bs) s += yr[c] * dr[c];
    s = learn_block_sum(s, sh, tid, bs);
    for (uint c = tid; c < cols; c += bs) learn_store(dx, r * cols + c, yr[c] * (dr[c] - s), accum);
}

// Last-axis LayerNorm; saves mean and rstd per row for the backward.
kernel void learn_layernorm(device const float* x [[buffer(0)]], device const float* g [[buffer(1)]], device const float* b [[buffer(2)]],
                             device float* y [[buffer(3)]], device float* mean [[buffer(4)]], device float* rstd [[buffer(5)]],
                             constant uint& rows [[buffer(6)]], constant uint& cols [[buffer(7)]], constant float& eps [[buffer(8)]],
                             uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint bs [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    long r = gid;
    device const float* xr = x + r * cols;
    float s = 0.0f;
    for (uint c = tid; c < cols; c += bs) s += xr[c];
    float mu = learn_block_sum(s, sh, tid, bs) / cols;
    float v = 0.0f;
    for (uint c = tid; c < cols; c += bs) { float d = xr[c] - mu; v += d * d; }
    float rs = rsqrt(learn_block_sum(v, sh, tid, bs) / cols + eps);
    for (uint c = tid; c < cols; c += bs) y[r * cols + c] = (xr[c] - mu) * rs * g[c] + b[c];
    if (tid == 0) { mean[r] = mu; rstd[r] = rs; }
}

kernel void learn_layernorm_bwd(device const float* x [[buffer(0)]], device const float* g [[buffer(1)]], device const float* mean [[buffer(2)]],
                                  device const float* rstd [[buffer(3)]], device const float* dy [[buffer(4)]], device float* dx [[buffer(5)]],
                                  constant uint& rows [[buffer(6)]], constant uint& cols [[buffer(7)]], constant uint& accum [[buffer(8)]],
                                  uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint bs [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    long r = gid;
    device const float* xr = x + r * cols;
    device const float* dr = dy + r * cols;
    float mu = mean[r], rs = rstd[r];
    float s1 = 0.0f, s2 = 0.0f;
    for (uint c = tid; c < cols; c += bs) {
        float gh = dr[c] * g[c], xh = (xr[c] - mu) * rs;
        s1 += gh; s2 += gh * xh;
    }
    s1 = learn_block_sum(s1, sh, tid, bs) / cols;
    s2 = learn_block_sum(s2, sh, tid, bs) / cols;
    for (uint c = tid; c < cols; c += bs) {
        float gh = dr[c] * g[c], xh = (xr[c] - mu) * rs;
        learn_store(dx, r * cols + c, rs * (gh - s1 - xh * s2), accum);
    }
}

// dg[c] (+)= Sum_r dy*xhat, db[c] (+)= Sum_r dy. One thread per column.
kernel void learn_layernorm_wgrad(device const float* x [[buffer(0)]], device const float* mean [[buffer(1)]], device const float* rstd [[buffer(2)]],
                                    device const float* dy [[buffer(3)]], device float* dg [[buffer(4)]], device float* db [[buffer(5)]],
                                    constant uint& rows [[buffer(6)]], constant uint& cols [[buffer(7)]], constant uint& accum [[buffer(8)]],
                                    uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint c = gid.x * bs.x + tid.x;
    if (c >= cols) return;
    float sg = 0.0f, sb = 0.0f;
    for (long r = 0; r < rows; r++) {
        float d = dy[r * cols + c];
        sg += d * (x[r * cols + c] - mean[r]) * rstd[r];
        sb += d;
    }
    learn_store(dg, c, sg, accum);
    learn_store(db, c, sb, accum);
}

// y[0] (+)= scale * Sum x. One threadgroup, 1024 threads.
kernel void learn_sum(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                        constant uint& n [[buffer(2)]], constant float& scale [[buffer(3)]], constant uint& accum [[buffer(4)]],
                        uint tid [[thread_position_in_threadgroup]], uint bs [[threads_per_threadgroup]]) {
    threadgroup float sh[1024];
    float s = 0.0f;
    for (uint i = tid; i < n; i += bs) s += x[i];
    float t = learn_block_sum(s, sh, tid, bs);
    if (tid == 0) learn_store(y, 0, t * scale, accum);
}

kernel void learn_sumsq(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], constant uint& accum [[buffer(3)]],
                          uint tid [[thread_position_in_threadgroup]], uint bs [[threads_per_threadgroup]]) {
    threadgroup float sh[1024];
    float s = 0.0f;
    for (uint i = tid; i < n; i += bs) s += x[i] * x[i];
    float t = learn_block_sum(s, sh, tid, bs);
    if (tid == 0) learn_store(y, 0, t, accum);
}

kernel void learn_scale(device float* y [[buffer(0)]], constant uint& n [[buffer(1)]], constant float& s [[buffer(2)]],
                          uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i < n) y[i] *= s;
}

kernel void learn_bcast_scalar(device const float* dy [[buffer(0)]], device float* dx [[buffer(1)]],
                                 constant uint& n [[buffer(2)]], constant float& scale [[buffer(3)]], constant uint& accum [[buffer(4)]],
                                 uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i < n) learn_store(dx, i, scale * dy[0], accum);
}

// AdamW (decoupled weight decay, PyTorch order).
kernel void learn_adamw(device float* p [[buffer(0)]], device const float* g [[buffer(1)]], device float* m [[buffer(2)]], device float* v [[buffer(3)]],
                          constant uint& n [[buffer(4)]], constant float& lr [[buffer(5)]], constant float& b1 [[buffer(6)]],
                          constant float& b2 [[buffer(7)]], constant float& eps [[buffer(8)]], constant float& wd [[buffer(9)]],
                          constant float& bc1 [[buffer(10)]], constant float& bc2 [[buffer(11)]],
                          uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i >= n) return;
    float gi = g[i];
    float pi = p[i] * (1.0f - lr * wd);
    float mi = b1 * m[i] + (1.0f - b1) * gi;
    float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    p[i] = pi - lr * (mi / bc1) / (sqrt(vi / bc2) + eps);
}

// ---------------------------------------------------------------- vision ---
// NCHW f32. Conv is im2col + learn_gemm per image.

kernel void learn_im2col(device const float* x [[buffer(0)]], device float* col [[buffer(1)]],
                           constant uint& C [[buffer(2)]], constant uint& H [[buffer(3)]], constant uint& W [[buffer(4)]],
                           constant uint& kh [[buffer(5)]], constant uint& kw [[buffer(6)]], constant uint& sh [[buffer(7)]], constant uint& sw [[buffer(8)]],
                           constant uint& pt [[buffer(9)]], constant uint& pl [[buffer(10)]], constant uint& OH [[buffer(11)]], constant uint& OW [[buffer(12)]],
                           uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    long ohw = (long)OH * OW, n = (long)C * kh * kw * ohw;
    if (i >= n) return;
    int p = (int)(i % ohw);
    long row = i / ohw;
    int kx = (int)(row % kw), ky = (int)(row / kw % kh), c = (int)(row / ((long)kw * kh));
    int oy = p / (int)OW, ox = p % (int)OW;
    int iy = oy * (int)sh - (int)pt + ky, ix = ox * (int)sw - (int)pl + kx;
    col[i] = (iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W) ? x[((long)c * H + iy) * W + ix] : 0.0f;
}

kernel void learn_col2im(device const float* col [[buffer(0)]], device float* dx [[buffer(1)]],
                           constant uint& C [[buffer(2)]], constant uint& H [[buffer(3)]], constant uint& W [[buffer(4)]],
                           constant uint& kh [[buffer(5)]], constant uint& kw [[buffer(6)]], constant uint& sh [[buffer(7)]], constant uint& sw [[buffer(8)]],
                           constant uint& pt [[buffer(9)]], constant uint& pl [[buffer(10)]], constant uint& OH [[buffer(11)]], constant uint& OW [[buffer(12)]],
                           constant uint& accum [[buffer(13)]],
                           uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    if (i >= (long)C * H * W) return;
    int ix = (int)(i % W), iy = (int)(i / W % H), c = (int)(i / ((long)W * H));
    long ohw = (long)OH * OW;
    float acc = 0.0f;
    for (int ky = 0; ky < (int)kh; ky++) {
        int ty = iy + (int)pt - ky;
        if (ty < 0 || ty % (int)sh) continue;
        int oy = ty / (int)sh;
        if (oy >= (int)OH) continue;
        for (int kx = 0; kx < (int)kw; kx++) {
            int tx = ix + (int)pl - kx;
            if (tx < 0 || tx % (int)sw) continue;
            int ox = tx / (int)sw;
            if (ox >= (int)OW) continue;
            acc += col[(((long)c * kh + ky) * kw + kx) * ohw + (long)oy * OW + ox];
        }
    }
    learn_store(dx, i, acc, accum);
}

// db[c] (+)= Sum_{n,p} dy[n][c][p]. One threadgroup per channel.
kernel void learn_channel_sum(device const float* dy [[buffer(0)]], device float* db [[buffer(1)]],
                                constant uint& N [[buffer(2)]], constant uint& C [[buffer(3)]], constant uint& HW [[buffer(4)]], constant uint& accum [[buffer(5)]],
                                uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint bs [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    uint c = gid;
    float s = 0.0f;
    for (uint n = 0; n < N; n++) {
        device const float* p = dy + ((long)n * C + c) * HW;
        for (uint i = tid; i < HW; i += bs) s += p[i];
    }
    s = learn_block_sum(s, sh, tid, bs);
    if (tid == 0) learn_store(db, c, s, accum);
}

kernel void learn_bn_stats(device const float* x [[buffer(0)]], device float* mean [[buffer(1)]], device float* rstd [[buffer(2)]],
                             constant uint& N [[buffer(3)]], constant uint& C [[buffer(4)]], constant uint& HW [[buffer(5)]], constant float& eps [[buffer(6)]],
                             uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint bs [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    uint c = gid;
    float m = (float)N * HW;
    float s = 0.0f;
    for (uint n = 0; n < N; n++) {
        device const float* p = x + ((long)n * C + c) * HW;
        for (uint i = tid; i < HW; i += bs) s += p[i];
    }
    float mu = learn_block_sum(s, sh, tid, bs) / m;
    float v = 0.0f;
    for (uint n = 0; n < N; n++) {
        device const float* p = x + ((long)n * C + c) * HW;
        for (uint i = tid; i < HW; i += bs) { float d = p[i] - mu; v += d * d; }
    }
    v = learn_block_sum(v, sh, tid, bs) / m;
    if (tid == 0) { mean[c] = mu; rstd[c] = rsqrt(v + eps); }
}

kernel void learn_bn_apply(device const float* x [[buffer(0)]], device const float* mean [[buffer(1)]], device const float* rstd [[buffer(2)]],
                             device const float* g [[buffer(3)]], device const float* b [[buffer(4)]], device float* y [[buffer(5)]],
                             constant uint& n [[buffer(6)]], constant uint& C [[buffer(7)]], constant uint& HW [[buffer(8)]],
                             uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i >= n) return;
    uint c = i / HW % C;
    y[i] = (x[i] - mean[c]) * rstd[c] * g[c] + b[c];
}

kernel void learn_bn_wgrad(device const float* x [[buffer(0)]], device const float* mean [[buffer(1)]], device const float* rstd [[buffer(2)]],
                             device const float* dy [[buffer(3)]], device float* dg [[buffer(4)]], device float* db [[buffer(5)]],
                             constant uint& N [[buffer(6)]], constant uint& C [[buffer(7)]], constant uint& HW [[buffer(8)]],
                             uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint bs [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    uint c = gid;
    float mu = mean[c], rs = rstd[c];
    float sg = 0.0f, sb = 0.0f;
    for (uint n = 0; n < N; n++) {
        long o = ((long)n * C + c) * HW;
        for (uint i = tid; i < HW; i += bs) {
            float d = dy[o + i];
            sg += d * (x[o + i] - mu) * rs;
            sb += d;
        }
    }
    sg = learn_block_sum(sg, sh, tid, bs);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sb = learn_block_sum(sb, sh, tid, bs);
    if (tid == 0) { dg[c] = sg; db[c] = sb; }
}

kernel void learn_bn_bwd(device const float* x [[buffer(0)]], device const float* mean [[buffer(1)]], device const float* rstd [[buffer(2)]],
                           device const float* g [[buffer(3)]], device const float* dg [[buffer(4)]], device const float* db [[buffer(5)]],
                           device const float* dy [[buffer(6)]], device float* dx [[buffer(7)]],
                           constant uint& n [[buffer(8)]], constant uint& C [[buffer(9)]], constant uint& HW [[buffer(10)]],
                           constant float& inv_m [[buffer(11)]], constant uint& accum [[buffer(12)]],
                           uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint i = gid.x * bs.x + tid.x;
    if (i >= n) return;
    uint c = i / HW % C;
    float xh = (x[i] - mean[c]) * rstd[c];
    learn_store(dx, i, g[c] * rstd[c] * (dy[i] - db[c] * inv_m - xh * dg[c] * inv_m), accum);
}

kernel void learn_bn_running(device const float* mean [[buffer(0)]], device const float* rstd [[buffer(1)]], device float* rm [[buffer(2)]], device float* rv [[buffer(3)]],
                               constant uint& C [[buffer(4)]], constant float& mom [[buffer(5)]], constant float& eps [[buffer(6)]], constant float& unbias [[buffer(7)]],
                               uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    uint c = gid.x * bs.x + tid.x;
    if (c >= C) return;
    float var = 1.0f / (rstd[c] * rstd[c]) - eps;
    rm[c] = (1.0f - mom) * rm[c] + mom * mean[c];
    rv[c] = (1.0f - mom) * rv[c] + mom * var * unbias;
}

kernel void learn_maxpool(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], device float* idx [[buffer(2)]],
                            constant uint& planes [[buffer(3)]], constant uint& H [[buffer(4)]], constant uint& W [[buffer(5)]],
                            constant uint& OH [[buffer(6)]], constant uint& OW [[buffer(7)]], constant uint& kh [[buffer(8)]], constant uint& kw [[buffer(9)]],
                            constant uint& sh [[buffer(10)]], constant uint& sw [[buffer(11)]], constant uint& pt [[buffer(12)]], constant uint& pl [[buffer(13)]],
                            uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    if (i >= (long)planes * OH * OW) return;
    int ox = (int)(i % OW), oy = (int)(i / OW % OH);
    long pbase = i / ((long)OW * OH) * H * W;
    float best = -3.0e38f;
    int bi = -1;
    for (int ky = 0; ky < (int)kh; ky++) {
        int iy = oy * (int)sh - (int)pt + ky;
        if (iy < 0 || iy >= (int)H) continue;
        for (int kx = 0; kx < (int)kw; kx++) {
            int ix = ox * (int)sw - (int)pl + kx;
            if (ix < 0 || ix >= (int)W) continue;
            float v = x[pbase + (long)iy * W + ix];
            if (v > best || bi < 0) { best = v; bi = iy * (int)W + ix; }
        }
    }
    y[i] = best;
    idx[i] = (float)bi;
}

kernel void learn_maxpool_bwd(device const float* dy [[buffer(0)]], device const float* idx [[buffer(1)]], device float* dx [[buffer(2)]],
                                constant uint& planes [[buffer(3)]], constant uint& H [[buffer(4)]], constant uint& W [[buffer(5)]],
                                constant uint& OH [[buffer(6)]], constant uint& OW [[buffer(7)]], constant uint& kh [[buffer(8)]], constant uint& kw [[buffer(9)]],
                                constant uint& sh [[buffer(10)]], constant uint& sw [[buffer(11)]], constant uint& pt [[buffer(12)]], constant uint& pl [[buffer(13)]],
                                constant uint& accum [[buffer(14)]],
                                uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    if (i >= (long)planes * H * W) return;
    int ix = (int)(i % W), iy = (int)(i / W % H);
    long plane = i / ((long)W * H), obase = plane * OH * OW;
    int q = iy * (int)W + ix;
    float acc = 0.0f;
    for (int ky = 0; ky < (int)kh; ky++) {
        int ty = iy + (int)pt - ky;
        if (ty < 0 || ty % (int)sh) continue;
        int oy = ty / (int)sh;
        if (oy >= (int)OH) continue;
        for (int kx = 0; kx < (int)kw; kx++) {
            int tx = ix + (int)pl - kx;
            if (tx < 0 || tx % (int)sw) continue;
            int ox = tx / (int)sw;
            if (ox >= (int)OW) continue;
            long o = obase + (long)oy * OW + ox;
            if ((int)idx[o] == q) acc += dy[o];
        }
    }
    learn_store(dx, i, acc, accum);
}

static inline int learn_avg_div(int oy, int ox, int H, int W, int kh, int kw, int sh, int sw, int pt, int pl, uint cip) {
    int y0 = oy * sh - pt, x0 = ox * sw - pl;
    int y1 = y0 + kh, x1 = x0 + kw;
    if (cip) {
        y1 = min(y1, H + pt); x1 = min(x1, W + pl);
        return (y1 - y0) * (x1 - x0);
    }
    y0 = max(y0, 0); x0 = max(x0, 0); y1 = min(y1, H); x1 = min(x1, W);
    return (y1 - y0) * (x1 - x0);
}

kernel void learn_avgpool(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                            constant uint& planes [[buffer(2)]], constant uint& H [[buffer(3)]], constant uint& W [[buffer(4)]],
                            constant uint& OH [[buffer(5)]], constant uint& OW [[buffer(6)]], constant uint& kh [[buffer(7)]], constant uint& kw [[buffer(8)]],
                            constant uint& sh [[buffer(9)]], constant uint& sw [[buffer(10)]], constant uint& pt [[buffer(11)]], constant uint& pl [[buffer(12)]],
                            constant uint& cip [[buffer(13)]],
                            uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    if (i >= (long)planes * OH * OW) return;
    int ox = (int)(i % OW), oy = (int)(i / OW % OH);
    long pbase = i / ((long)OW * OH) * H * W;
    float s = 0.0f;
    for (int ky = 0; ky < (int)kh; ky++) {
        int iy = oy * (int)sh - (int)pt + ky;
        if (iy < 0 || iy >= (int)H) continue;
        for (int kx = 0; kx < (int)kw; kx++) {
            int ix = ox * (int)sw - (int)pl + kx;
            if (ix >= 0 && ix < (int)W) s += x[pbase + (long)iy * W + ix];
        }
    }
    y[i] = s / (float)learn_avg_div(oy, ox, (int)H, (int)W, (int)kh, (int)kw, (int)sh, (int)sw, (int)pt, (int)pl, cip);
}

kernel void learn_avgpool_bwd(device const float* dy [[buffer(0)]], device float* dx [[buffer(1)]],
                                constant uint& planes [[buffer(2)]], constant uint& H [[buffer(3)]], constant uint& W [[buffer(4)]],
                                constant uint& OH [[buffer(5)]], constant uint& OW [[buffer(6)]], constant uint& kh [[buffer(7)]], constant uint& kw [[buffer(8)]],
                                constant uint& sh [[buffer(9)]], constant uint& sw [[buffer(10)]], constant uint& pt [[buffer(11)]], constant uint& pl [[buffer(12)]],
                                constant uint& cip [[buffer(13)]], constant uint& accum [[buffer(14)]],
                                uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    if (i >= (long)planes * H * W) return;
    int ix = (int)(i % W), iy = (int)(i / W % H);
    long obase = i / ((long)W * H) * OH * OW;
    float acc = 0.0f;
    for (int ky = 0; ky < (int)kh; ky++) {
        int ty = iy + (int)pt - ky;
        if (ty < 0 || ty % (int)sh) continue;
        int oy = ty / (int)sh;
        if (oy >= (int)OH) continue;
        for (int kx = 0; kx < (int)kw; kx++) {
            int tx = ix + (int)pl - kx;
            if (tx < 0 || tx % (int)sw) continue;
            int ox = tx / (int)sw;
            if (ox >= (int)OW) continue;
            acc += dy[obase + (long)oy * OW + ox] / (float)learn_avg_div(oy, ox, (int)H, (int)W, (int)kh, (int)kw, (int)sh, (int)sw, (int)pt, (int)pl, cip);
        }
    }
    learn_store(dx, i, acc, accum);
}

kernel void learn_upsample(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                             constant uint& planes [[buffer(2)]], constant uint& H [[buffer(3)]], constant uint& W [[buffer(4)]],
                             constant uint& fy [[buffer(5)]], constant uint& fx [[buffer(6)]],
                             uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    uint OH = H * fy, OW = W * fx;
    if (i >= (long)planes * OH * OW) return;
    uint ox = (uint)(i % OW), oy = (uint)(i / OW % OH);
    long p = i / ((long)OW * OH);
    y[i] = x[(p * H + oy / fy) * W + ox / fx];
}

kernel void learn_upsample_bwd(device const float* dy [[buffer(0)]], device float* dx [[buffer(1)]],
                                 constant uint& planes [[buffer(2)]], constant uint& H [[buffer(3)]], constant uint& W [[buffer(4)]],
                                 constant uint& fy [[buffer(5)]], constant uint& fx [[buffer(6)]], constant uint& accum [[buffer(7)]],
                                 uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    if (i >= (long)planes * H * W) return;
    uint ix = (uint)(i % W), iy = (uint)(i / W % H);
    long p = i / ((long)W * H);
    uint OW = W * fx;
    device const float* d = dy + p * H * fy * OW;
    float acc = 0.0f;
    for (uint a = 0; a < fy; a++)
        for (uint b = 0; b < fx; b++) acc += d[(long)(iy * fy + a) * OW + ix * fx + b];
    learn_store(dx, i, acc, accum);
}

// GridSample, bilinear, zeros padding, align_corners = false (PyTorch grid_sampler_2d).
kernel void learn_grid_sample(device const float* x [[buffer(0)]], device const float* grid [[buffer(1)]], device float* y [[buffer(2)]],
                                constant uint& N [[buffer(3)]], constant uint& C [[buffer(4)]], constant uint& H [[buffer(5)]], constant uint& W [[buffer(6)]],
                                constant uint& Ho [[buffer(7)]], constant uint& Wo [[buffer(8)]],
                                uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    if (i >= (long)N * Ho * Wo) return;
    int n = (int)(i / ((long)Ho * Wo));
    long pix = i % ((long)Ho * Wo);
    float ix = ((grid[i * 2] + 1.0f) * W - 1.0f) * 0.5f;
    float iy = ((grid[i * 2 + 1] + 1.0f) * H - 1.0f) * 0.5f;
    int x0 = (int)floor(ix), y0 = (int)floor(iy);
    float fx = ix - x0, fy = iy - y0;
    float wnw = (1 - fx) * (1 - fy), wne = fx * (1 - fy), wsw = (1 - fx) * fy, wse = fx * fy;
    bool in_nw = x0 >= 0 && x0 < (int)W && y0 >= 0 && y0 < (int)H;
    bool in_ne = x0 + 1 >= 0 && x0 + 1 < (int)W && y0 >= 0 && y0 < (int)H;
    bool in_sw = x0 >= 0 && x0 < (int)W && y0 + 1 >= 0 && y0 + 1 < (int)H;
    bool in_se = x0 + 1 >= 0 && x0 + 1 < (int)W && y0 + 1 >= 0 && y0 + 1 < (int)H;
    for (uint c = 0; c < C; c++) {
        device const float* p = x + ((long)n * C + c) * H * W;
        float acc = 0.0f;
        if (in_nw) acc += p[(long)y0 * W + x0] * wnw;
        if (in_ne) acc += p[(long)y0 * W + x0 + 1] * wne;
        if (in_sw) acc += p[(long)(y0 + 1) * W + x0] * wsw;
        if (in_se) acc += p[(long)(y0 + 1) * W + x0 + 1] * wse;
        y[((long)n * C + c) * Ho * Wo + pix] = acc;
    }
}

// Backward: dx via atomic float add (a sample point can feed pixels another
// thread also writes), dgrid (+)= Sum_c over the corners.
kernel void learn_grid_sample_bwd(device const float* x [[buffer(0)]], device const float* grid [[buffer(1)]], device const float* dy [[buffer(2)]],
                                    device atomic_float* dx [[buffer(3)]], device float* dgrid [[buffer(4)]],
                                    constant uint& N [[buffer(5)]], constant uint& C [[buffer(6)]], constant uint& H [[buffer(7)]], constant uint& W [[buffer(8)]],
                                    constant uint& Ho [[buffer(9)]], constant uint& Wo [[buffer(10)]],
                                    constant uint& want_dx [[buffer(11)]], constant uint& want_dgrid [[buffer(12)]], constant uint& accum_dgrid [[buffer(13)]],
                                    uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    if (i >= (long)N * Ho * Wo) return;
    int n = (int)(i / ((long)Ho * Wo));
    long pix = i % ((long)Ho * Wo);
    float ix = ((grid[i * 2] + 1.0f) * W - 1.0f) * 0.5f;
    float iy = ((grid[i * 2 + 1] + 1.0f) * H - 1.0f) * 0.5f;
    int x0 = (int)floor(ix), y0 = (int)floor(iy);
    float fx = ix - x0, fy = iy - y0;
    float wnw = (1 - fx) * (1 - fy), wne = fx * (1 - fy), wsw = (1 - fx) * fy, wse = fx * fy;
    bool in_nw = x0 >= 0 && x0 < (int)W && y0 >= 0 && y0 < (int)H;
    bool in_ne = x0 + 1 >= 0 && x0 + 1 < (int)W && y0 >= 0 && y0 < (int)H;
    bool in_sw = x0 >= 0 && x0 < (int)W && y0 + 1 >= 0 && y0 + 1 < (int)H;
    bool in_se = x0 + 1 >= 0 && x0 + 1 < (int)W && y0 + 1 >= 0 && y0 + 1 < (int)H;
    float gix = 0.0f, giy = 0.0f;
    for (uint c = 0; c < C; c++) {
        long pb = ((long)n * C + c) * H * W;
        float g = dy[((long)n * C + c) * Ho * Wo + pix];
        float vnw = in_nw ? x[pb + (long)y0 * W + x0] : 0.0f;
        float vne = in_ne ? x[pb + (long)y0 * W + x0 + 1] : 0.0f;
        float vsw = in_sw ? x[pb + (long)(y0 + 1) * W + x0] : 0.0f;
        float vse = in_se ? x[pb + (long)(y0 + 1) * W + x0 + 1] : 0.0f;
        gix += g * ((vne - vnw) * (1 - fy) + (vse - vsw) * fy);
        giy += g * ((vsw - vnw) * (1 - fx) + (vse - vne) * fx);
        if (want_dx) {
            if (in_nw) atomic_fetch_add_explicit(dx + pb + (long)y0 * W + x0, g * wnw, memory_order_relaxed);
            if (in_ne) atomic_fetch_add_explicit(dx + pb + (long)y0 * W + x0 + 1, g * wne, memory_order_relaxed);
            if (in_sw) atomic_fetch_add_explicit(dx + pb + (long)(y0 + 1) * W + x0, g * wsw, memory_order_relaxed);
            if (in_se) atomic_fetch_add_explicit(dx + pb + (long)(y0 + 1) * W + x0 + 1, g * wse, memory_order_relaxed);
        }
    }
    if (want_dgrid) {
        learn_store(dgrid, i * 2, gix * W * 0.5f, accum_dgrid);
        learn_store(dgrid, i * 2 + 1, giy * H * 0.5f, accum_dgrid);
    }
}

kernel void learn_gather_rows(device const float* x [[buffer(0)]], device const float* idx [[buffer(1)]], device float* y [[buffer(2)]],
                                constant uint& B [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]], constant uint& C [[buffer(6)]],
                                uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    if (i >= (long)B * K * C) return;
    uint c = (uint)(i % C);
    long bk = i / C;
    uint b = (uint)(bk / K);
    long r = (long)idx[bk];
    y[i] = x[((long)b * N + r) * C + c];
}

kernel void learn_gather_rows_bwd(device const float* dy [[buffer(0)]], device const float* idx [[buffer(1)]], device atomic_float* dx [[buffer(2)]],
                                    constant uint& B [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]], constant uint& C [[buffer(6)]],
                                    uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 bs [[threads_per_threadgroup]]) {
    long i = (long)gid.x * bs.x + tid.x;
    if (i >= (long)B * K * C) return;
    uint c = (uint)(i % C);
    long bk = i / C;
    uint b = (uint)(bk / K);
    long r = (long)idx[bk];
    atomic_fetch_add_explicit(dx + ((long)b * N + r) * C + c, dy[i], memory_order_relaxed);
}
"#;

pub const NAMES: &[&str] = &[
    "learn_fill", "learn_axpby", "learn_unary", "learn_unary_bwd",
    "learn_binary", "learn_binary_grad", "learn_reduce_to", "learn_copy_strided",
    "learn_gemm", "learn_softmax", "learn_softmax_bwd",
    "learn_layernorm", "learn_layernorm_bwd", "learn_layernorm_wgrad",
    "learn_sum", "learn_sumsq", "learn_scale", "learn_bcast_scalar", "learn_adamw",
    "learn_im2col", "learn_col2im", "learn_channel_sum",
    "learn_bn_stats", "learn_bn_apply", "learn_bn_wgrad", "learn_bn_bwd", "learn_bn_running",
    "learn_maxpool", "learn_maxpool_bwd", "learn_avgpool", "learn_avgpool_bwd",
    "learn_upsample", "learn_upsample_bwd",
    "learn_grid_sample", "learn_grid_sample_bwd",
    "learn_gather_rows", "learn_gather_rows_bwd",
];
