//! Training kernels (`learn` family): the f32 forward + backward primitives the
//! `ojas-learn` tape is built from. Contiguous row-major f32 everywhere (no
//! padded NHWC storage — that is the inference layout). Each kernel has a plain
//! Rust twin in `ojas-learn`'s CPU backend, which is the reference it is tested
//! against.
//!
//! Convention (shared with every family): buffers first, then `int` / `float`
//! constants (floats cross as `f32::to_bits()`). Unless noted: 1-D launch,
//! one thread per output element, `grid = ceil(n / 256)`.
//! `accum` = 1 adds into the output instead of overwriting it (gradients of a
//! tensor with several readers accumulate).

pub const BODY: &str = r#"
#define LEARN_MAXR 6

// unary codes: 0 relu | 1 sigmoid | 2 silu | 3 tanh | 4 gelu (erf) | 5 exp | 6 log
//              7 neg | 8 sqrt
__device__ __forceinline__ float learn_unary_f(float x, int op) {
    switch (op) {
    case 0: return x > 0.0f ? x : 0.0f;
    case 1: return 1.0f / (1.0f + expf(-x));
    case 2: return x / (1.0f + expf(-x));
    case 3: return tanhf(x);
    case 4: return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
    case 5: return expf(x);
    case 6: return logf(x);
    case 7: return -x;
    default: return sqrtf(x);
    }
}

// d f / d x, from the input x and the forward output y
__device__ __forceinline__ float learn_unary_d(float x, float y, int op) {
    switch (op) {
    case 0: return x > 0.0f ? 1.0f : 0.0f;
    case 1: return y * (1.0f - y);
    case 2: { float s = 1.0f / (1.0f + expf(-x)); return s * (1.0f + x * (1.0f - s)); }
    case 3: return 1.0f - y * y;
    case 4: return 0.5f * (1.0f + erff(x * 0.70710678118654752f))
                 + x * 0.3989422804014327f * expf(-0.5f * x * x);
    case 5: return y;
    case 6: return 1.0f / x;
    case 7: return -1.0f;
    default: return 0.5f / y;
    }
}

// binary codes: 0 add | 1 sub | 2 mul | 3 div | 4 max | 5 min
__device__ __forceinline__ float learn_binary_f(float a, float b, int op) {
    switch (op) {
    case 0: return a + b;
    case 1: return a - b;
    case 2: return a * b;
    case 3: return a / b;
    case 4: return a >= b ? a : b;
    default: return a <= b ? a : b;
    }
}

// d (a op b) / d a (which = 0) or / d b (which = 1); ties of max/min go to a
__device__ __forceinline__ float learn_binary_d(float a, float b, int op, int which) {
    switch (op) {
    case 0: return 1.0f;
    case 1: return which ? -1.0f : 1.0f;
    case 2: return which ? a : b;
    case 3: return which ? -a / (b * b) : 1.0f / b;
    case 4: return which ? (b > a ? 1.0f : 0.0f) : (a >= b ? 1.0f : 0.0f);
    default: return which ? (b < a ? 1.0f : 0.0f) : (a <= b ? 1.0f : 0.0f);
    }
}

__device__ __forceinline__ void learn_store(float* y, long i, float v, int accum) {
    if (accum) y[i] += v; else y[i] = v;
}

extern "C" __global__ void learn_fill(float* y, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = v;
}

// y = a·x + b·y
extern "C" __global__ void learn_axpby(const float* x, float* y, int n, float a, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b * y[i];
}

extern "C" __global__ void learn_unary(const float* x, float* y, int n, int op) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = learn_unary_f(x[i], op);
}

// dx (+)= f'(x) · dy
extern "C" __global__ void learn_unary_bwd(const float* x, const float* y, const float* dy, float* dx,
                                           int n, int op, int accum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) learn_store(dx, i, learn_unary_d(x[i], y[i], op) * dy[i], accum);
}

// Broadcast binary: y (contiguous, dims) = a op b, where a / b are read through
// strides over the output dims (stride 0 = broadcast axis). Rank ≤ 6, dims
// right-aligned and padded with 1.
#define LEARN_BCAST_ARGS int n, int op, int d0, int d1, int d2, int d3, int d4, int d5, \
    int sa0, int sa1, int sa2, int sa3, int sa4, int sa5, \
    int sb0, int sb1, int sb2, int sb3, int sb4, int sb5
#define LEARN_BCAST_OFFSETS \
    int dims[6] = {d0, d1, d2, d3, d4, d5}; \
    int sa[6] = {sa0, sa1, sa2, sa3, sa4, sa5}; \
    int sb[6] = {sb0, sb1, sb2, sb3, sb4, sb5}; \
    long oa = 0, ob = 0; int rem = i; \
    for (int d = 5; d >= 0; d--) { int c = rem % dims[d]; rem /= dims[d]; oa += (long)c * sa[d]; ob += (long)c * sb[d]; }

extern "C" __global__ void learn_binary(const float* a, const float* b, float* y, LEARN_BCAST_ARGS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    LEARN_BCAST_OFFSETS
    y[i] = learn_binary_f(a[oa], b[ob], op);
}

// t[i] = dy[i] · d(a op b)/d(which) at output position i (full output shape);
// the caller then sums t down to the operand's shape with learn_reduce_to.
extern "C" __global__ void learn_binary_grad(const float* a, const float* b, const float* dy, float* t,
                                             int which, LEARN_BCAST_ARGS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    LEARN_BCAST_OFFSETS
    t[i] = dy[i] * learn_binary_d(a[oa], b[ob], op, which);
}

// Sum t (full dims, contiguous) over the axes flagged in `red` (bit d = axis d
// of the 6 right-aligned dims) into g (the kept axes, contiguous). One thread per
// output element, in index order — deterministic.
extern "C" __global__ void learn_reduce_to(const float* t, float* g, int m, int accum, int red,
                                           int d0, int d1, int d2, int d3, int d4, int d5) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= m) return;
    int dims[6] = {d0, d1, d2, d3, d4, d5};
    long fs[6];
    long s = 1;
    for (int d = 5; d >= 0; d--) { fs[d] = s; s *= dims[d]; }
    long base = 0; int rem = o; long R = 1;
    for (int d = 5; d >= 0; d--) {
        if (red >> d & 1) { R *= dims[d]; continue; }
        int c = rem % dims[d]; rem /= dims[d]; base += (long)c * fs[d];
    }
    float acc = 0.0f;
    for (long r = 0; r < R; r++) {
        long rr = r, off = base;
        for (int d = 5; d >= 0; d--) {
            if (!(red >> d & 1)) continue;
            int c = (int)(rr % dims[d]); rr /= dims[d]; off += (long)c * fs[d];
        }
        acc += t[off];
    }
    learn_store(g, o, acc, accum);
}

// Fast path of learn_reduce_to when the reduced axes all precede the kept ones:
// t viewed as [R, C] → partial[s][c] = Σ over rows [s·chunk, (s+1)·chunk).
// grid [ceil(C/256), splits], block 256.
extern "C" __global__ void learn_reduce_rows_partial(const float* t, float* part, int R, int C, int chunk) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    int s = blockIdx.y;
    int r0 = s * chunk, r1 = min(R, r0 + chunk);
    float acc = 0.0f;
    for (int r = r0; r < r1; r++) acc += t[(long)r * C + c];
    part[(long)s * C + c] = acc;
}

// g[c] (+)= Σ_s part[s][c], splits in order (deterministic)
extern "C" __global__ void learn_reduce_rows_final(const float* part, float* g, int C, int splits, int accum) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float acc = 0.0f;
    for (int s = 0; s < splits; s++) acc += part[(long)s * C + c];
    learn_store(g, c, acc, accum);
}

// y[yoff + Σ c·ys] (+)= x[xoff + Σ c·xs] over the index space dims (rank ≤ 6,
// right-aligned). Permute, slice, concat and their backwards are all this.
extern "C" __global__ void learn_copy_strided(const float* x, float* y, int n, int accum,
                                              int d0, int d1, int d2, int d3, int d4, int d5,
                                              int xs0, int xs1, int xs2, int xs3, int xs4, int xs5,
                                              int ys0, int ys1, int ys2, int ys3, int ys4, int ys5,
                                              int xoff, int yoff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int dims[6] = {d0, d1, d2, d3, d4, d5};
    int xs[6] = {xs0, xs1, xs2, xs3, xs4, xs5};
    int ys[6] = {ys0, ys1, ys2, ys3, ys4, ys5};
    long ox = xoff, oy = yoff; int rem = i;
    for (int d = 5; d >= 0; d--) { int c = rem % dims[d]; rem /= dims[d]; ox += (long)c * xs[d]; oy += (long)c * ys[d]; }
    learn_store(y, oy, x[ox], accum);
}

// Batched row-major GEMM: C = alpha·op(A)·op(B) + beta·C, op(A) M×K, op(B) K×N.
// ta: A stored K×M; tb: B stored N×K. Batch z: A += z·sa, B += z·sb, C += z·sc.
// 64×64 tile, k step 16, 256 threads × 4×4 outputs. grid [ceil(N/64), ceil(M/64), batch], block 256.
extern "C" __global__ void __launch_bounds__(256) learn_gemm(const float* A, const float* B, float* C,
        int M, int N, int K, int ta, int tb, int sa, int sb, int sc, float alpha, float beta) {
    __shared__ float As[16][68];
    __shared__ float Bs[16][68];
    int z = blockIdx.z;
    A += (long)z * sa; B += (long)z * sb; C += (long)z * sc;
    int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    int row0 = blockIdx.y * 64, col0 = blockIdx.x * 64;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int e = tid + i * 256;
            int r, kk;
            if (ta) { kk = e >> 6; r = e & 63; } else { r = e >> 4; kk = e & 15; }
            int gr = row0 + r, gk = k0 + kk;
            float v = 0.0f;
            if (gr < M && gk < K) v = ta ? A[(long)gk * M + gr] : A[(long)gr * K + gk];
            As[kk][r] = v;
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int e = tid + i * 256;
            int c, kk;
            if (tb) { c = e >> 4; kk = e & 15; } else { kk = e >> 6; c = e & 63; }
            int gc = col0 + c, gk = k0 + kk;
            float v = 0.0f;
            if (gc < N && gk < K) v = tb ? B[(long)gc * K + gk] : B[(long)gk * N + gc];
            Bs[kk][c] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[kk][ty * 4 + i];
            #pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[kk][tx * 4 + j];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row0 + ty * 4 + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int c = col0 + tx * 4 + j;
            if (c >= N) continue;
            long o = (long)r * N + c;
            float v = alpha * acc[i][j];
            C[o] = beta == 0.0f ? v : v + beta * C[o];
        }
    }
}

// block-wide sum over 256 threads (all threads get the result)
__device__ float learn_block_sum(float v, float* sh) {
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float t = l < (blockDim.x >> 5) ? sh[l] : 0.0f;
    for (int m = 16; m; m >>= 1) t += __shfl_xor_sync(0xffffffffu, t, m);
    return t;
}

__device__ float learn_block_max(float v, float* sh) {
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float t = l < (blockDim.x >> 5) ? sh[l] : __int_as_float(0xff800000);
    for (int m = 16; m; m >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, m));
    return t;
}

// Row softmax over the last axis. grid rows, block 256.
extern "C" __global__ void learn_softmax(const float* x, float* y, int rows, int cols) {
    __shared__ float sh[32];
    long r = blockIdx.x;
    const float* xr = x + r * cols;
    float* yr = y + r * cols;
    float m = __int_as_float(0xff800000);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, xr[c]);
    m = learn_block_max(m, sh);
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += expf(xr[c] - m);
    s = learn_block_sum(s, sh);
    float inv = 1.0f / s;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = expf(xr[c] - m) * inv;
}

// dx (+)= y ⊙ (dy − Σ y⊙dy). grid rows, block 256.
extern "C" __global__ void learn_softmax_bwd(const float* y, const float* dy, float* dx, int rows,
                                             int cols, int accum) {
    __shared__ float sh[32];
    long r = blockIdx.x;
    const float* yr = y + r * cols;
    const float* dr = dy + r * cols;
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += yr[c] * dr[c];
    s = learn_block_sum(s, sh);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) learn_store(dx, r * cols + c, yr[c] * (dr[c] - s), accum);
}

// Last-axis LayerNorm; saves mean and rstd per row for the backward. grid rows, block 256.
extern "C" __global__ void learn_layernorm(const float* x, const float* g, const float* b, float* y,
                                           float* mean, float* rstd, int rows, int cols, float eps) {
    __shared__ float sh[32];
    long r = blockIdx.x;
    const float* xr = x + r * cols;
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += xr[c];
    float mu = learn_block_sum(s, sh) / cols;
    float v = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) { float d = xr[c] - mu; v += d * d; }
    float rs = rsqrtf(learn_block_sum(v, sh) / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[r * cols + c] = (xr[c] - mu) * rs * g[c] + b[c];
    if (threadIdx.x == 0) { mean[r] = mu; rstd[r] = rs; }
}

// dx (+)= rstd·(ĝ − mean(ĝ) − x̂·mean(ĝ⊙x̂)), ĝ = dy⊙g. grid rows, block 256.
extern "C" __global__ void learn_layernorm_bwd(const float* x, const float* g, const float* mean,
                                               const float* rstd, const float* dy, float* dx,
                                               int rows, int cols, int accum) {
    __shared__ float sh[32];
    long r = blockIdx.x;
    const float* xr = x + r * cols;
    const float* dr = dy + r * cols;
    float mu = mean[r], rs = rstd[r];
    float s1 = 0.0f, s2 = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float gh = dr[c] * g[c], xh = (xr[c] - mu) * rs;
        s1 += gh; s2 += gh * xh;
    }
    s1 = learn_block_sum(s1, sh) / cols;
    s2 = learn_block_sum(s2, sh) / cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float gh = dr[c] * g[c], xh = (xr[c] - mu) * rs;
        learn_store(dx, r * cols + c, rs * (gh - s1 - xh * s2), accum);
    }
}

// dg[c] (+)= Σ_r dy·x̂, db[c] (+)= Σ_r dy. One thread per column, rows in order.
extern "C" __global__ void learn_layernorm_wgrad(const float* x, const float* mean, const float* rstd,
                                                 const float* dy, float* dg, float* db, int rows,
                                                 int cols, int accum) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
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

// y[0] (+)= scale · Σ x. One block of 1024 threads, grid 1.
extern "C" __global__ void learn_sum(const float* x, float* y, int n, float scale, int accum) {
    __shared__ float sh[32];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i];
    s = learn_block_sum(s, sh);
    if (threadIdx.x == 0) learn_store(y, 0, s * scale, accum);
}

// y[0] (+)= Σ x²  (gradient norms). One block of 1024 threads, grid 1.
extern "C" __global__ void learn_sumsq(const float* x, float* y, int n, int accum) {
    __shared__ float sh[32];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
    s = learn_block_sum(s, sh);
    if (threadIdx.x == 0) learn_store(y, 0, s, accum);
}

// y *= s
extern "C" __global__ void learn_scale(float* y, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= s;
}

// dx[i] (+)= scale · dy[0]  (the backward of learn_sum)
extern "C" __global__ void learn_bcast_scalar(const float* dy, float* dx, int n, float scale, int accum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) learn_store(dx, i, scale * dy[0], accum);
}

// AdamW (decoupled weight decay, PyTorch torch.optim.AdamW order):
// p ← p − lr·wd·p; m ← b1·m + (1−b1)·g; v ← b2·v + (1−b2)·g²;
// p ← p − lr·(m/bc1) / (sqrt(v/bc2) + eps)
extern "C" __global__ void learn_adamw(float* p, const float* g, float* m, float* v, int n, float lr,
                                       float b1, float b2, float eps, float wd, float bc1, float bc2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float pi = p[i] * (1.0f - lr * wd);
    float mi = b1 * m[i] + (1.0f - b1) * gi;
    float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    p[i] = pi - lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
}

// ---------------------------------------------------------------- vision ---
// NCHW f32. Conv is im2col + learn_gemm per image; col rows are
// (c, ky, kx) with c major, columns (oy, ox): the rows of group g are contiguous,
// so a grouped conv is one batched GEMM over groups.

// col[(c·kh + ky)·kw + kx][oy·OW + ox] = x[c][oy·sh − pt + ky][ox·sw − pl + kx] (0 outside)
extern "C" __global__ void learn_im2col(const float* x, float* col, int C, int H, int W, int kh, int kw,
                                        int sh, int sw, int pt, int pl, int OH, int OW) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long ohw = (long)OH * OW, n = (long)C * kh * kw * ohw;
    if (i >= n) return;
    int p = (int)(i % ohw);
    long row = i / ohw;
    int kx = (int)(row % kw), ky = (int)(row / kw % kh), c = (int)(row / ((long)kw * kh));
    int oy = p / OW, ox = p % OW;
    int iy = oy * sh - pt + ky, ix = ox * sw - pl + kx;
    col[i] = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? x[((long)c * H + iy) * W + ix] : 0.0f;
}

// dx[c][iy][ix] (+)= Σ col entries that read it (gather: deterministic)
extern "C" __global__ void learn_col2im(const float* col, float* dx, int C, int H, int W, int kh, int kw,
                                        int sh, int sw, int pt, int pl, int OH, int OW, int accum) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)C * H * W) return;
    int ix = (int)(i % W), iy = (int)(i / W % H), c = (int)(i / ((long)W * H));
    long ohw = (long)OH * OW;
    float acc = 0.0f;
    for (int ky = 0; ky < kh; ky++) {
        int ty = iy + pt - ky;
        if (ty < 0 || ty % sh) continue;
        int oy = ty / sh;
        if (oy >= OH) continue;
        for (int kx = 0; kx < kw; kx++) {
            int tx = ix + pl - kx;
            if (tx < 0 || tx % sw) continue;
            int ox = tx / sw;
            if (ox >= OW) continue;
            acc += col[(((long)c * kh + ky) * kw + kx) * ohw + (long)oy * OW + ox];
        }
    }
    learn_store(dx, i, acc, accum);
}

// db[c] (+)= Σ_{n, p} dy[n][c][p]. grid C, block 256.
extern "C" __global__ void learn_channel_sum(const float* dy, float* db, int N, int C, int HW, int accum) {
    __shared__ float sh[32];
    int c = blockIdx.x;
    float s = 0.0f;
    for (int n = 0; n < N; n++) {
        const float* p = dy + ((long)n * C + c) * HW;
        for (int i = threadIdx.x; i < HW; i += blockDim.x) s += p[i];
    }
    s = learn_block_sum(s, sh);
    if (threadIdx.x == 0) learn_store(db, c, s, accum);
}

// BatchNorm batch statistics: mean and 1/sqrt(biased var + eps) per channel. grid C, block 256.
extern "C" __global__ void learn_bn_stats(const float* x, float* mean, float* rstd, int N, int C, int HW, float eps) {
    __shared__ float sh[32];
    int c = blockIdx.x;
    float m = (float)N * HW;
    float s = 0.0f;
    for (int n = 0; n < N; n++) {
        const float* p = x + ((long)n * C + c) * HW;
        for (int i = threadIdx.x; i < HW; i += blockDim.x) s += p[i];
    }
    float mu = learn_block_sum(s, sh) / m;
    float v = 0.0f;
    for (int n = 0; n < N; n++) {
        const float* p = x + ((long)n * C + c) * HW;
        for (int i = threadIdx.x; i < HW; i += blockDim.x) { float d = p[i] - mu; v += d * d; }
    }
    v = learn_block_sum(v, sh) / m;
    if (threadIdx.x == 0) { mean[c] = mu; rstd[c] = rsqrtf(v + eps); }
}

// y = (x − mean)·rstd·g + b per channel
extern "C" __global__ void learn_bn_apply(const float* x, const float* mean, const float* rstd, const float* g,
                                          const float* b, float* y, int n, int C, int HW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i / HW % C;
    y[i] = (x[i] - mean[c]) * rstd[c] * g[c] + b[c];
}

// dg[c] = Σ dy·x̂, db[c] = Σ dy (overwrite). grid C, block 256.
extern "C" __global__ void learn_bn_wgrad(const float* x, const float* mean, const float* rstd, const float* dy,
                                          float* dg, float* db, int N, int C, int HW) {
    __shared__ float sh[32];
    int c = blockIdx.x;
    float mu = mean[c], rs = rstd[c];
    float sg = 0.0f, sb = 0.0f;
    for (int n = 0; n < N; n++) {
        long o = ((long)n * C + c) * HW;
        for (int i = threadIdx.x; i < HW; i += blockDim.x) {
            float d = dy[o + i];
            sg += d * (x[o + i] - mu) * rs;
            sb += d;
        }
    }
    sg = learn_block_sum(sg, sh);
    sb = learn_block_sum(sb, sh);
    if (threadIdx.x == 0) { dg[c] = sg; db[c] = sb; }
}

// dx (+)= g·rstd·(dy − db/M − x̂·dg/M), M = N·HW (training-mode BatchNorm)
extern "C" __global__ void learn_bn_bwd(const float* x, const float* mean, const float* rstd, const float* g,
                                        const float* dg, const float* db, const float* dy, float* dx,
                                        int n, int C, int HW, float inv_m, int accum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i / HW % C;
    float xh = (x[i] - mean[c]) * rstd[c];
    learn_store(dx, i, g[c] * rstd[c] * (dy[i] - db[c] * inv_m - xh * dg[c] * inv_m), accum);
}

// running stats (PyTorch): rm ← (1−mom)·rm + mom·mean; rv ← (1−mom)·rv + mom·var·M/(M−1)
extern "C" __global__ void learn_bn_running(const float* mean, const float* rstd, float* rm, float* rv, int C,
                                            float mom, float eps, float unbias) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float var = 1.0f / (rstd[c] * rstd[c]) - eps;
    rm[c] = (1.0f - mom) * rm[c] + mom * mean[c];
    rv[c] = (1.0f - mom) * rv[c] + mom * var * unbias;
}

// Max pool over planes (N·C of them), windows bounds-checked (padding never wins);
// idx = the argmax's in-plane index, as float (exact below 2^24).
extern "C" __global__ void learn_maxpool(const float* x, float* y, float* idx, int planes, int H, int W,
                                         int OH, int OW, int kh, int kw, int sh, int sw, int pt, int pl) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)planes * OH * OW) return;
    int ox = (int)(i % OW), oy = (int)(i / OW % OH);
    long pbase = i / ((long)OW * OH) * H * W;
    float best = __int_as_float(0xff800000);
    int bi = -1;
    for (int ky = 0; ky < kh; ky++) {
        int iy = oy * sh - pt + ky;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < kw; kx++) {
            int ix = ox * sw - pl + kx;
            if (ix < 0 || ix >= W) continue;
            float v = x[pbase + (long)iy * W + ix];
            if (v > best || bi < 0) { best = v; bi = iy * W + ix; }
        }
    }
    y[i] = best;
    idx[i] = (float)bi;
}

// dx[p][q] (+)= Σ dy over the output windows whose argmax is q (gather)
extern "C" __global__ void learn_maxpool_bwd(const float* dy, const float* idx, float* dx, int planes, int H, int W,
                                             int OH, int OW, int kh, int kw, int sh, int sw, int pt, int pl, int accum) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)planes * H * W) return;
    int ix = (int)(i % W), iy = (int)(i / W % H);
    long plane = i / ((long)W * H), obase = plane * OH * OW;
    int q = iy * W + ix;
    float acc = 0.0f;
    for (int ky = 0; ky < kh; ky++) {
        int ty = iy + pt - ky;
        if (ty < 0 || ty % sh) continue;
        int oy = ty / sh;
        if (oy >= OH) continue;
        for (int kx = 0; kx < kw; kx++) {
            int tx = ix + pl - kx;
            if (tx < 0 || tx % sw) continue;
            int ox = tx / sw;
            if (ox >= OW) continue;
            long o = obase + (long)oy * OW + ox;
            if ((int)idx[o] == q) acc += dy[o];
        }
    }
    learn_store(dx, i, acc, accum);
}

// Average pool; divisor = the window clipped to the image (count_include_pad = 0)
// or clipped to the padded image (1), as PyTorch AvgPool2d.
__device__ __forceinline__ int learn_avg_div(int oy, int ox, int H, int W, int kh, int kw, int sh, int sw,
                                              int pt, int pl, int cip) {
    int y0 = oy * sh - pt, x0 = ox * sw - pl;
    int y1 = y0 + kh, x1 = x0 + kw;
    if (cip) {
        y1 = min(y1, H + pt); x1 = min(x1, W + pl);
        return (y1 - y0) * (x1 - x0);
    }
    y0 = max(y0, 0); x0 = max(x0, 0); y1 = min(y1, H); x1 = min(x1, W);
    return (y1 - y0) * (x1 - x0);
}

extern "C" __global__ void learn_avgpool(const float* x, float* y, int planes, int H, int W, int OH, int OW,
                                         int kh, int kw, int sh, int sw, int pt, int pl, int cip) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)planes * OH * OW) return;
    int ox = (int)(i % OW), oy = (int)(i / OW % OH);
    long pbase = i / ((long)OW * OH) * H * W;
    float s = 0.0f;
    for (int ky = 0; ky < kh; ky++) {
        int iy = oy * sh - pt + ky;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < kw; kx++) {
            int ix = ox * sw - pl + kx;
            if (ix >= 0 && ix < W) s += x[pbase + (long)iy * W + ix];
        }
    }
    y[i] = s / (float)learn_avg_div(oy, ox, H, W, kh, kw, sh, sw, pt, pl, cip);
}

extern "C" __global__ void learn_avgpool_bwd(const float* dy, float* dx, int planes, int H, int W, int OH, int OW,
                                             int kh, int kw, int sh, int sw, int pt, int pl, int cip, int accum) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)planes * H * W) return;
    int ix = (int)(i % W), iy = (int)(i / W % H);
    long obase = i / ((long)W * H) * OH * OW;
    float acc = 0.0f;
    for (int ky = 0; ky < kh; ky++) {
        int ty = iy + pt - ky;
        if (ty < 0 || ty % sh) continue;
        int oy = ty / sh;
        if (oy >= OH) continue;
        for (int kx = 0; kx < kw; kx++) {
            int tx = ix + pl - kx;
            if (tx < 0 || tx % sw) continue;
            int ox = tx / sw;
            if (ox >= OW) continue;
            acc += dy[obase + (long)oy * OW + ox] / (float)learn_avg_div(oy, ox, H, W, kh, kw, sh, sw, pt, pl, cip);
        }
    }
    learn_store(dx, i, acc, accum);
}

// Nearest upsample by integer factors: y[p][oy][ox] = x[p][oy/fy][ox/fx]
extern "C" __global__ void learn_upsample(const float* x, float* y, int planes, int H, int W, int fy, int fx) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int OH = H * fy, OW = W * fx;
    if (i >= (long)planes * OH * OW) return;
    int ox = (int)(i % OW), oy = (int)(i / OW % OH);
    long p = i / ((long)OW * OH);
    y[i] = x[(p * H + oy / fy) * W + ox / fx];
}

extern "C" __global__ void learn_upsample_bwd(const float* dy, float* dx, int planes, int H, int W, int fy, int fx,
                                              int accum) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)planes * H * W) return;
    int ix = (int)(i % W), iy = (int)(i / W % H);
    long p = i / ((long)W * H);
    int OW = W * fx;
    const float* d = dy + p * H * fy * OW;
    float acc = 0.0f;
    for (int a = 0; a < fy; a++)
        for (int b = 0; b < fx; b++) acc += d[(long)(iy * fy + a) * OW + ix * fx + b];
    learn_store(dx, i, acc, accum);
}

// GridSample, bilinear, zeros padding, align_corners = false — PyTorch
// grid_sampler_2d (the forward in ojas-vision's detr_ops is the same formula).
// x [N,C,H,W], grid [N,Ho,Wo,2] (x, y in [-1, 1]) → y [N,C,Ho,Wo]
extern "C" __global__ void learn_grid_sample(const float* x, const float* grid, float* y, int N, int C, int H,
                                             int W, int Ho, int Wo) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * Ho * Wo) return;
    int n = (int)(i / ((long)Ho * Wo));
    long pix = i % ((long)Ho * Wo);
    float ix = ((grid[i * 2] + 1.0f) * W - 1.0f) * 0.5f;
    float iy = ((grid[i * 2 + 1] + 1.0f) * H - 1.0f) * 0.5f;
    int x0 = (int)floorf(ix), y0 = (int)floorf(iy);
    float fx = ix - x0, fy = iy - y0;
    float wnw = (1 - fx) * (1 - fy), wne = fx * (1 - fy), wsw = (1 - fx) * fy, wse = fx * fy;
    bool in_nw = x0 >= 0 && x0 < W && y0 >= 0 && y0 < H;
    bool in_ne = x0 + 1 >= 0 && x0 + 1 < W && y0 >= 0 && y0 < H;
    bool in_sw = x0 >= 0 && x0 < W && y0 + 1 >= 0 && y0 + 1 < H;
    bool in_se = x0 + 1 >= 0 && x0 + 1 < W && y0 + 1 >= 0 && y0 + 1 < H;
    for (int c = 0; c < C; c++) {
        const float* p = x + ((long)n * C + c) * H * W;
        float acc = 0.0f;
        if (in_nw) acc += p[(long)y0 * W + x0] * wnw;
        if (in_ne) acc += p[(long)y0 * W + x0 + 1] * wne;
        if (in_sw) acc += p[(long)(y0 + 1) * W + x0] * wsw;
        if (in_se) acc += p[(long)(y0 + 1) * W + x0 + 1] * wse;
        y[((long)n * C + c) * Ho * Wo + pix] = acc;
    }
}

// Backward: dx via atomicAdd (a sample point feeds 4 arbitrary pixels; PyTorch
// also scatters), dgrid (+)= Σ_c over the corners (one thread per sample point).
extern "C" __global__ void learn_grid_sample_bwd(const float* x, const float* grid, const float* dy, float* dx,
                                                 float* dgrid, int N, int C, int H, int W, int Ho, int Wo,
                                                 int want_dx, int want_dgrid, int accum_dgrid) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * Ho * Wo) return;
    int n = (int)(i / ((long)Ho * Wo));
    long pix = i % ((long)Ho * Wo);
    float ix = ((grid[i * 2] + 1.0f) * W - 1.0f) * 0.5f;
    float iy = ((grid[i * 2 + 1] + 1.0f) * H - 1.0f) * 0.5f;
    int x0 = (int)floorf(ix), y0 = (int)floorf(iy);
    float fx = ix - x0, fy = iy - y0;
    float wnw = (1 - fx) * (1 - fy), wne = fx * (1 - fy), wsw = (1 - fx) * fy, wse = fx * fy;
    bool in_nw = x0 >= 0 && x0 < W && y0 >= 0 && y0 < H;
    bool in_ne = x0 + 1 >= 0 && x0 + 1 < W && y0 >= 0 && y0 < H;
    bool in_sw = x0 >= 0 && x0 < W && y0 + 1 >= 0 && y0 + 1 < H;
    bool in_se = x0 + 1 >= 0 && x0 + 1 < W && y0 + 1 >= 0 && y0 + 1 < H;
    float gix = 0.0f, giy = 0.0f;
    for (int c = 0; c < C; c++) {
        long pb = ((long)n * C + c) * H * W;
        float g = dy[((long)n * C + c) * Ho * Wo + pix];
        float vnw = in_nw ? x[pb + (long)y0 * W + x0] : 0.0f;
        float vne = in_ne ? x[pb + (long)y0 * W + x0 + 1] : 0.0f;
        float vsw = in_sw ? x[pb + (long)(y0 + 1) * W + x0] : 0.0f;
        float vse = in_se ? x[pb + (long)(y0 + 1) * W + x0 + 1] : 0.0f;
        gix += g * ((vne - vnw) * (1 - fy) + (vse - vsw) * fy);
        giy += g * ((vsw - vnw) * (1 - fx) + (vse - vne) * fx);
        if (want_dx) {
            if (in_nw) atomicAdd(dx + pb + (long)y0 * W + x0, g * wnw);
            if (in_ne) atomicAdd(dx + pb + (long)y0 * W + x0 + 1, g * wne);
            if (in_sw) atomicAdd(dx + pb + (long)(y0 + 1) * W + x0, g * wsw);
            if (in_se) atomicAdd(dx + pb + (long)(y0 + 1) * W + x0 + 1, g * wse);
        }
    }
    if (want_dgrid) {
        learn_store(dgrid, i * 2, gix * W * 0.5f, accum_dgrid);
        learn_store(dgrid, i * 2 + 1, giy * H * 0.5f, accum_dgrid);
    }
}

// y[b][k][:] = x[b][idx[b][k]][:]  (idx as float, exact below 2^24)
extern "C" __global__ void learn_gather_rows(const float* x, const float* idx, float* y, int B, int N, int K, int C) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)B * K * C) return;
    int c = (int)(i % C);
    long bk = i / C;
    int b = (int)(bk / K);
    long r = (long)idx[bk];
    y[i] = x[((long)b * N + r) * C + c];
}

// dx[b][idx[b][k]][:] += dy[b][k][:] (atomic: indices may repeat)
extern "C" __global__ void learn_gather_rows_bwd(const float* dy, const float* idx, float* dx, int B, int N, int K,
                                                 int C) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)B * K * C) return;
    int c = (int)(i % C);
    long bk = i / C;
    int b = (int)(bk / K);
    long r = (long)idx[bk];
    atomicAdd(dx + ((long)b * N + r) * C + c, dy[i]);
}

// ------------------------------------------------------- tensor cores ---
// learn_gemm on TF32 tensor cores (mma.sync m16n8k8, f32 accumulate) — the
// precision PyTorch uses for convolutions by default. 64×64 block tile, k step
// 16, 4 warps × 32×32. Split-K: blockIdx.z = batch·splits + split; with
// splits > 1 each split writes alpha·partial to the workspace C [batch][splits][M][N]
// and learn_splitk_reduce adds them in split order (deterministic).
// bcol = 1: B is not stored — it is the im2col matrix of the image at B
// (channels cC, cH×cW, kernel ckh×ckw, stride, top/left pad, output cOH×cOW),
// so a convolution never materialises its unfolded input.
// grid [ceil(N/64), ceil(M/64), batch·splits], block 128.
__device__ __forceinline__ unsigned learn_tf32(float x) {
    unsigned u;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
    return u;
}

extern "C" __global__ void __launch_bounds__(128) learn_gemm_tc(const float* A, const float* B, float* C,
        int M, int N, int K, int ta, int tb, int sa, int sb, int sc, float alpha, float beta,
        int kchunk, int splits, int bcol, int cC, int cH, int cW, int ckh, int ckw, int csh, int csw,
        int cpt, int cpl, int cOH, int cOW) {
    __shared__ float As[16][72];
    __shared__ float Bs[16][72];
    int z = blockIdx.z, bz = z / splits, sp = z % splits;
    A += (long)bz * sa; B += (long)bz * sb;
    int k_begin = sp * kchunk, k_end = min(K, k_begin + kchunk);
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int wm = warp >> 1, wn = warp & 1, g = lane >> 2, t4 = lane & 3;
    int row0 = blockIdx.y * 64, col0 = blockIdx.x * 64;
    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int q = 0; q < 4; q++) acc[i][j][q] = 0.0f;
    for (int k0 = k_begin; k0 < k_end; k0 += 16) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int e = tid + i * 128, r, kk;
            if (ta) { kk = e >> 6; r = e & 63; } else { r = e >> 4; kk = e & 15; }
            int gr = row0 + r, gk = k0 + kk;
            float v = 0.0f;
            if (gr < M && gk < k_end) v = ta ? A[(long)gk * M + gr] : A[(long)gr * K + gk];
            As[kk][r] = v;
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int e = tid + i * 128, c, kk;
            if (tb) { c = e >> 4; kk = e & 15; } else { kk = e >> 6; c = e & 63; }
            int gc = col0 + c, gk = k0 + kk;
            float v = 0.0f;
            if (gc < N && gk < k_end) {
                if (bcol) {
                    // implicit im2col: B is col(row, pixel) of the image at B (tb: B(k, n) = col(n, k))
                    int row = tb ? gc : gk, pix = tb ? gk : gc;
                    int kx = row % ckw, t = row / ckw, ky = t % ckh, ch = t / ckh;
                    int iy = (pix / cOW) * csh - cpt + ky, ix = (pix % cOW) * csw - cpl + kx;
                    if (iy >= 0 && iy < cH && ix >= 0 && ix < cW) v = B[((long)ch * cH + iy) * cW + ix];
                } else {
                    v = tb ? B[(long)gc * K + gk] : B[(long)gk * N + gc];
                }
            }
            Bs[kk][c] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < 16; ks += 8) {
            unsigned a[2][4], b[4][2];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int rb = wm * 32 + mi * 16;
                a[mi][0] = learn_tf32(As[ks + t4][rb + g]);
                a[mi][1] = learn_tf32(As[ks + t4][rb + g + 8]);
                a[mi][2] = learn_tf32(As[ks + t4 + 4][rb + g]);
                a[mi][3] = learn_tf32(As[ks + t4 + 4][rb + g + 8]);
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int cb = wn * 32 + ni * 8;
                b[ni][0] = learn_tf32(Bs[ks + t4][cb + g]);
                b[ni][1] = learn_tf32(Bs[ks + t4 + 4][cb + g]);
            }
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]), "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[ni][0]), "r"(b[ni][1]));
        }
        __syncthreads();
    }
    float* Cz = splits > 1 ? C + (long)z * M * N : C + (long)bz * sc;
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int r = row0 + wm * 32 + mi * 16 + g + (q >> 1) * 8;
                int c = col0 + wn * 32 + ni * 8 + t4 * 2 + (q & 1);
                if (r >= M || c >= N) continue;
                long o = (long)r * N + c;
                float v = alpha * acc[mi][ni][q];
                if (splits > 1) Cz[o] = v;
                else Cz[o] = beta == 0.0f ? v : v + beta * Cz[o];
            }
}

// ------------------------------------------------ bf16 tensor cores ---
// Mixed precision: f32 in memory, rounded to bf16 as tiles land in shared
// memory, mma.sync m16n8k16 bf16 with f32 accumulation (PyTorch autocast's
// bf16 GEMMs). bf16 keeps f32's exponent range: no loss scaling. Block 128×128,
// k step 32, 8 warps × 64×32; the next k tile is prefetched into registers
// while the current one is multiplied. Same arguments, split-K and implicit
// im2col (bcol) as learn_gemm_tc, specialised per layout at compile time.
__device__ __forceinline__ unsigned short learn_bf16(float x) {
    unsigned u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u); // round to nearest even (inputs are finite)
    return (unsigned short)(u >> 16);
}

#define LBF_SK 40    // [128][32] tile row stride in bf16: 32 + 8 (16-byte rows, conflict-free ldmatrix)
#define LBF_SMN 136  // [32][128] tile row stride: 128 + 8
#define LBF_TILE (128 * LBF_SK > 32 * LBF_SMN ? 128 * LBF_SK : 32 * LBF_SMN)

__device__ __forceinline__ unsigned learn_bf16x2(float lo, float hi) {
    return (unsigned)learn_bf16(lo) | ((unsigned)learn_bf16(hi) << 16);
}

// One A or B element of the implicit im2col: col(row, pix) of the image at X.
__device__ __forceinline__ float learn_col_at(const float* X, int row, int pix, int cH, int cW, int ckh, int ckw,
                                              int csh, int csw, int cpt, int cpl, int cOW) {
    int kx = row % ckw, t = row / ckw, ky = t % ckh, ch = t / ckh;
    int iy = (pix / cOW) * csh - cpt + ky, ix = (pix % cOW) * csw - cpl + kx;
    return (iy >= 0 && iy < cH && ix >= 0 && ix < cW) ? X[((long)ch * cH + iy) * cW + ix] : 0.0f;
}

// TA/TB: operand stored transposed; BCOL: B is the implicit im2col of X.
// VA/VB (runtime, uniform): the operand's contiguous axis is 16-byte aligned →
// float4 loads. A tile: 128 rows × 32 k; B tile: 32 k × 128 cols; per thread
// 4 groups of 4 elements each along the contiguous axis.
template <int TA, int TB, int BCOL>
__device__ __forceinline__ void lbf_body(const float* A, const float* B, float* C, int M, int N, int K,
        int sa, int sb, int sc, float alpha, float beta, int kchunk, int splits, int va, int vb,
        int cH, int cW, int ckh, int ckw, int csh, int csw, int cpt, int cpl, int cOW) {
    // [m|n][k] tiles (k contiguous, row stride LBF_SK) or [k][m|n] tiles (row stride
    // LBF_SMN): each operand keeps its global contiguous axis, so the stores are
    // always vectors; ldmatrix.trans turns [k][·] tiles into fragments.
    __shared__ __align__(16) unsigned short As[2][LBF_TILE];
    __shared__ __align__(16) unsigned short Bs[2][LBF_TILE];
    int z = blockIdx.z, bz = z / splits, sp = z % splits;
    A += (long)bz * sa; B += (long)bz * sb;
    int k_begin = sp * kchunk, k_end = min(K, k_begin + kchunk);
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int wm = warp >> 2, wn = warp & 3;
    int row0 = blockIdx.y * 128, col0 = blockIdx.x * 128;
    float ra[16], rb[16];
    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int q = 0; q < 4; q++) acc[i][j][q] = 0.0f;

    // group i of thread: 4 consecutive elements along the operand's contiguous axis.
    // A, !TA (k contiguous): r = g4 >> 3, k = (g4 & 7)·4; TA (m contiguous): k = g4 >> 5, r = (g4 & 31)·4
    // B, TB (k contiguous): c = g4 >> 3, k = (g4 & 7)·4;  !TB (n contiguous): k = g4 >> 5, c = (g4 & 31)·4
    auto load = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int g4 = tid + i * 256;
            int r = TA ? (g4 & 31) * 4 : g4 >> 3, kk = TA ? g4 >> 5 : (g4 & 7) * 4;
            int gr = row0 + r, gk = k0 + kk;
            bool full = TA ? (gk < k_end && gr + 3 < M) : (gr < M && gk + 3 < k_end);
            if (va && full) {
                float4 v = *(const float4*)(TA ? A + (long)gk * M + gr : A + (long)gr * K + gk);
                ra[i * 4] = v.x; ra[i * 4 + 1] = v.y; ra[i * 4 + 2] = v.z; ra[i * 4 + 3] = v.w;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int rr = TA ? gr + j : gr, kj = TA ? gk : gk + j;
                    ra[i * 4 + j] = (rr < M && kj < k_end) ? (TA ? A[(long)kj * M + rr] : A[(long)rr * K + kj]) : 0.0f;
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int g4 = tid + i * 256;
            int c = TB ? g4 >> 3 : (g4 & 31) * 4, kk = TB ? (g4 & 7) * 4 : g4 >> 5;
            int gc = col0 + c, gk = k0 + kk;
            if (BCOL) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int cj = TB ? gc : gc + j, kj = TB ? gk + j : gk;
                    rb[i * 4 + j] = (cj < N && kj < k_end)
                        ? learn_col_at(B, TB ? cj : kj, TB ? kj : cj, cH, cW, ckh, ckw, csh, csw, cpt, cpl, cOW) : 0.0f;
                }
            } else {
                bool full = TB ? (gc < N && gk + 3 < k_end) : (gk < k_end && gc + 3 < N);
                if (vb && full) {
                    float4 v = *(const float4*)(TB ? B + (long)gc * K + gk : B + (long)gk * N + gc);
                    rb[i * 4] = v.x; rb[i * 4 + 1] = v.y; rb[i * 4 + 2] = v.z; rb[i * 4 + 3] = v.w;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int cj = TB ? gc : gc + j, kj = TB ? gk + j : gk;
                        rb[i * 4 + j] = (cj < N && kj < k_end) ? (TB ? B[(long)cj * K + kj] : B[(long)kj * N + cj]) : 0.0f;
                    }
                }
            }
        }
    };
    // shared tiles: As[m][k], Bs[n][k] (k contiguous) as bf16
    auto store = [&](int buf) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int g4 = tid + i * 256;
            unsigned* d;
            if (TA) {  // [k][m]
                int kk = g4 >> 5, r = (g4 & 31) * 4;
                d = (unsigned*)(As[buf] + kk * LBF_SMN + r);
            } else {   // [m][k]
                int r = g4 >> 3, kk = (g4 & 7) * 4;
                d = (unsigned*)(As[buf] + r * LBF_SK + kk);
            }
            d[0] = learn_bf16x2(ra[i * 4], ra[i * 4 + 1]);
            d[1] = learn_bf16x2(ra[i * 4 + 2], ra[i * 4 + 3]);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int g4 = tid + i * 256;
            unsigned* d;
            if (TB) {  // [n][k]
                int c = g4 >> 3, kk = (g4 & 7) * 4;
                d = (unsigned*)(Bs[buf] + c * LBF_SK + kk);
            } else {   // [k][n]
                int kk = g4 >> 5, c = (g4 & 31) * 4;
                d = (unsigned*)(Bs[buf] + kk * LBF_SMN + c);
            }
            d[0] = learn_bf16x2(rb[i * 4], rb[i * 4 + 1]);
            d[1] = learn_bf16x2(rb[i * 4 + 2], rb[i * 4 + 3]);
        }
    };

    int buf = 0;
    if (k_begin < k_end) {
        load(k_begin);
        store(0);
    }
    __syncthreads();
    for (int k0 = k_begin; k0 < k_end; k0 += 32) {
        bool more = k0 + 32 < k_end;
        if (more) load(k0 + 32);
        const unsigned short* as = As[buf];
        const unsigned short* bs = Bs[buf];
        #pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            unsigned a[4][4], b[4][2];
            #pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                int m0 = wm * 64 + mi * 16, mat = lane >> 3;
                const unsigned short* ap = TA
                    // [k][m]: matrices (m0,k0) (m0+8,k0) (m0,k8) (m0+8,k8), each read transposed
                    ? as + (ks + (mat >> 1) * 8 + (lane & 7)) * LBF_SMN + m0 + (mat & 1) * 8
                    : as + (m0 + (lane & 15)) * LBF_SK + ks + (lane >> 4) * 8;
                unsigned addr = (unsigned)__cvta_generic_to_shared(ap);
                if (TA)
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                        : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3]) : "r"(addr));
                else
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                        : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3]) : "r"(addr));
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ni += 2) {
                int n0 = wn * 32 + ni * 8, mat = lane >> 3;
                const unsigned short* bp = TB
                    ? bs + (n0 + (lane >> 4) * 8 + (lane & 7)) * LBF_SK + ks + ((lane >> 3) & 1) * 8
                    // [k][n]: matrices (k0,n0) (k8,n0) (k0,n0+8) (k8,n0+8), each read transposed
                    : bs + (ks + (mat & 1) * 8 + (lane & 7)) * LBF_SMN + n0 + (mat >> 1) * 8;
                unsigned addr = (unsigned)__cvta_generic_to_shared(bp);
                if (TB)
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                        : "=r"(b[ni][0]), "=r"(b[ni][1]), "=r"(b[ni + 1][0]), "=r"(b[ni + 1][1]) : "r"(addr));
                else
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                        : "=r"(b[ni][0]), "=r"(b[ni][1]), "=r"(b[ni + 1][0]), "=r"(b[ni + 1][1]) : "r"(addr));
            }
            #pragma unroll
            for (int mi = 0; mi < 4; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]), "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[ni][0]), "r"(b[ni][1]));
        }
        if (more) store(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
    int g = lane >> 2, t4 = lane & 3;
    float* Cz = splits > 1 ? C + (long)z * M * N : C + (long)bz * sc;
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int r = row0 + wm * 64 + mi * 16 + g + (q >> 1) * 8;
                int c = col0 + wn * 32 + ni * 8 + t4 * 2 + (q & 1);
                if (r >= M || c >= N) continue;
                long o = (long)r * N + c;
                float v = alpha * acc[mi][ni][q];
                if (splits > 1) Cz[o] = v;
                else Cz[o] = beta == 0.0f ? v : v + beta * Cz[o];
            }
}

// learn_gemm_bf16_<ta><tb><bcol>: grid [ceil(N/128), ceil(M/128), batch·splits], block 256.
// Arguments as learn_gemm_tc plus va, vb (float4-aligned operands).
#define LBF_ENTRY(TA, TB, BC)                                                                          \
extern "C" __global__ void __launch_bounds__(256) learn_gemm_bf16_##TA##TB##BC(const float* A, const float* B, \
        float* C, int M, int N, int K, int ta, int tb, int sa, int sb, int sc, float alpha, float beta,     \
        int kchunk, int splits, int bcol, int cC, int cH, int cW, int ckh, int ckw, int csh, int csw,       \
        int cpt, int cpl, int cOH, int cOW, int va, int vb) {                                              \
    lbf_body<TA, TB, BC>(A, B, C, M, N, K, sa, sb, sc, alpha, beta, kchunk, splits, va, vb, cH, cW, ckh,   \
                         ckw, csh, csw, cpt, cpl, cOW);                                                    \
}
LBF_ENTRY(0, 0, 0)
LBF_ENTRY(1, 0, 0)
LBF_ENTRY(0, 1, 0)
LBF_ENTRY(1, 1, 0)
LBF_ENTRY(0, 0, 1)
LBF_ENTRY(0, 1, 1)

// C[b][j] = Σ_s ws[b][s][j] + beta·C[b][j] (splits added in order)
extern "C" __global__ void learn_splitk_reduce(const float* ws, float* C, int mn, int batch, int splits, int sc,
                                               float beta) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)batch * mn) return;
    int b = (int)(i / mn), j = (int)(i % mn);
    float s = 0.0f;
    for (int k = 0; k < splits; k++) s += ws[((long)b * splits + k) * mn + j];
    long o = (long)b * sc + j;
    C[o] = beta == 0.0f ? s : s + beta * C[o];
}
"#;

pub const NAMES: &[&str] = &[
    "learn_fill",
    "learn_axpby",
    "learn_unary",
    "learn_unary_bwd",
    "learn_binary",
    "learn_binary_grad",
    "learn_reduce_to",
    "learn_reduce_rows_partial",
    "learn_reduce_rows_final",
    "learn_copy_strided",
    "learn_gemm",
    "learn_gemm_tc",
    "learn_gemm_bf16_000",
    "learn_gemm_bf16_100",
    "learn_gemm_bf16_010",
    "learn_gemm_bf16_110",
    "learn_gemm_bf16_001",
    "learn_gemm_bf16_011",
    "learn_splitk_reduce",
    "learn_softmax",
    "learn_softmax_bwd",
    "learn_layernorm",
    "learn_layernorm_bwd",
    "learn_layernorm_wgrad",
    "learn_sum",
    "learn_bcast_scalar",
    "learn_sumsq",
    "learn_scale",
    "learn_adamw",
    "learn_im2col",
    "learn_col2im",
    "learn_channel_sum",
    "learn_bn_stats",
    "learn_bn_apply",
    "learn_bn_wgrad",
    "learn_bn_bwd",
    "learn_bn_running",
    "learn_maxpool",
    "learn_maxpool_bwd",
    "learn_avgpool",
    "learn_avgpool_bwd",
    "learn_upsample",
    "learn_upsample_bwd",
    "learn_grid_sample",
    "learn_grid_sample_bwd",
    "learn_gather_rows",
    "learn_gather_rows_bwd",
];
