// Kernel bodies are byte-identical to the reference CUDA backend; entry names are
// canonicalized to the Metal-canonical ones.
// Compiles against kernels::PRELUDE (cuda_fp16 + warp reduction helpers).
pub const BODY: &str = r#"
// alpha/beta post-processing per (token, v-head): gate = softplus(alpha+dt)*a,
// beta = sigmoid(beta). n = M*hv total elems; dt/a are [hv], indexed modulo.
extern "C" __global__ void ssm_ab(float* alpha, float* beta, const float* dt,
                                  const float* a, int n, int hv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int w = i % hv;
    float x = alpha[i] + dt[w];
    float sp = x > 20.0f ? x : logf(1.0f + expf(x));   // softplus (clamped)
    alpha[i] = sp * a[w];
    beta[i] = 1.0f / (1.0f + expf(-beta[i]));
}

// Causal depthwise conv1d for one token + SILU + rolling conv-state update.
// cw layout [K taps per channel]; cstate holds the last K-1 inputs [(K-1)*n_ch].
extern "C" __global__ void conv1d_decode(float* qkv, float* cstate, const float* cw,
                                         int n_ch, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;
    float xin = qkv[c];
    float acc = cw[c * K + (K - 1)] * xin;
    for (int j = 0; j < K - 1; j++) acc += cw[c * K + j] * cstate[j * n_ch + c];
    for (int j = 0; j + 2 < K; j++) cstate[j * n_ch + c] = cstate[(j + 1) * n_ch + c];
    cstate[(K - 2) * n_ch + c] = xin;
    qkv[c] = acc / (1.0f + expf(-acc));   // SILU
}

// Gated-DeltaNet recurrence: one warp per four consecutive state columns, the 4 x 128
// column values in registers (16/lane). q/k L2-norm fused in; 1/sqrt(S) folds into q's
// normalizer. Token-serial (the recurrence is sequential), so a step is a latency chain:
// the next token's q/k/v/gate/beta are loaded while this token is computed, and the four
// columns' reductions are independent and overlap. Four columns a warp puts a 32-head
// model's 256 blocks on the card at once (one wave, no tail). q/k/v are slices of the conv
// output row [conv_ch]; gate/beta rows [H_v]; out rows [H_v*S]. Assumes S==128.
// Several sequences at once: segment z of `segs` ([base, base+rows, slot, r0] u32 each)
// takes rows r0.. of the chunk and slot `slot` of the state (`slot_floats` apart).
// grid = (S/16, H_v, segments); block = 128 threads (4 warps).
extern "C" __global__ void deltanet_fused(float* state, const float* qkv,
                                          const float* gate, const float* beta,
                                          float* out, const unsigned* segs, int S, int H_k, int H_v,
                                          int conv_ch, int slot_floats, float eps) {
    const unsigned* seg = segs + 4 * blockIdx.z;
    const int M = (int)(seg[1] - seg[0]), r0 = (int)seg[3];
    state += (long)seg[2] * slot_floats;
    qkv += (long)r0 * conv_ch;
    gate += (long)r0 * H_v;
    beta += (long)r0 * H_v;
    out += (long)r0 * H_v * S;
    int h = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int col0 = (blockIdx.x * 4 + warp) * 4;
    if (col0 >= S) return;
    int hk = h % H_k;
    float scale = rsqrtf((float)S);
    float* sb = state + (long)(h * S + col0) * S;
    float ls[4][4];
    #pragma unroll
    for (int c = 0; c < 4; c++)
        #pragma unroll
        for (int j = 0; j < 4; j++) ls[c][j] = sb[c * S + j * 32 + lane];   // coalesced state layout
    const float* vcol = qkv + 2 * H_k * S + h * S + col0;
    auto fetch = [&](int t, float* qv, float* kv, float4* v4, float* g, float* b) {
        const float* row = qkv + (long)t * conv_ch;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int is = j * 32 + lane;
            qv[j] = row[hk * S + is];
            kv[j] = row[H_k * S + hk * S + is];
        }
        *v4 = *(const float4*)(vcol + (long)t * conv_ch);
        *g = gate[t * H_v + h];
        *b = beta[t * H_v + h];
    };
    float qv[4], kv[4], g, b;
    float4 v4;
    if (M > 0) fetch(0, qv, kv, &v4, &g, &b);
    for (int t = 0; t < M; t++) {
        float nq[4], nk[4], ng, nb;
        float4 nv;
        if (t + 1 < M) fetch(t + 1, nq, nk, &nv, &ng, &nb);
        float sq = 0.0f, s2 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) { sq += qv[j] * qv[j]; s2 += kv[j] * kv[j]; }
        sq = warp_all_sum(sq); s2 = warp_all_sum(s2);
        float qn = rsqrtf(sq + eps) * scale;
        float kn = rsqrtf(s2 + eps);
        float gexp = expf(g);
        float sk[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            sk[c] = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; j++) { ls[c][j] *= gexp; sk[c] += ls[c][j] * kv[j]; }
        }
        #pragma unroll
        for (int c = 0; c < 4; c++) sk[c] = warp_all_sum(sk[c]) * kn;
        const float vv[4] = {v4.x, v4.y, v4.z, v4.w};
        float y[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float d = (vv[c] - sk[c]) * b;
            y[c] = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; j++) { ls[c][j] += kv[j] * kn * d; y[c] += ls[c][j] * qv[j]; }
        }
        #pragma unroll
        for (int c = 0; c < 4; c++) y[c] = warp_all_sum(y[c]) * qn;
        if (lane == 0) *(float4*)(out + (long)t * (H_v * S) + h * S + col0) = make_float4(y[0], y[1], y[2], y[3]);
        if (t + 1 < M) {
            #pragma unroll
            for (int j = 0; j < 4; j++) { qv[j] = nq[j]; kv[j] = nk[j]; }
            v4 = nv; g = ng; b = nb;
        }
    }
    #pragma unroll
    for (int c = 0; c < 4; c++)
        #pragma unroll
        for (int j = 0; j < 4; j++) sb[c * S + j * 32 + lane] = ls[c][j];
}

// Causal depthwise conv1d over a segment's prompt tokens + SILU + rolling state update.
// qkv[*, n_ch] token-major; cstate holds the last K-1 inputs per slot (`slot_floats`
// apart). One thread per channel, sequential over tokens (mirrors conv1d_decode called M
// times); segment y of `segs` ([base, base+rows, slot, r0]) as in deltanet_fused.
extern "C" __global__ void conv1d_prefill(float* qkv, float* cstate, const float* cw,
    const unsigned* segs, int n_ch, int K, int slot_floats) {
    const unsigned* seg = segs + 4 * blockIdx.y;
    const int M = (int)(seg[1] - seg[0]);
    qkv += (long)seg[3] * n_ch;
    cstate += (long)seg[2] * slot_floats;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;
    float ring[8];
    for (int j = 0; j < K - 1; j++) ring[j] = cstate[j * n_ch + c];
    float wl = cw[c * K + (K - 1)];
    for (int t = 0; t < M; t++) {
        float xin = qkv[(long)t * n_ch + c];
        float acc = wl * xin;
        for (int j = 0; j < K - 1; j++) acc += cw[c * K + j] * ring[j];
        for (int j = 0; j + 2 < K; j++) ring[j] = ring[j + 1];
        ring[K - 2] = xin;
        qkv[(long)t * n_ch + c] = acc / (1.0f + expf(-acc));
    }
    for (int j = 0; j < K - 1; j++) cstate[j * n_ch + c] = ring[j];
}

// SSM prelude: L2-normalize q,k in-place in the conv output (per token, per
// k-head, over S). Folds q's 1/sqrt(S) scale in. Parallel over tokens → removes
// the sq/s2/qn/kn reductions from the sequential deltanet loop. One warp per
// (token, k-head). grid over ceil(M*Hk / (block/32)).
extern "C" __global__ void deltanet_prenorm(float* qkv, float* qkdot, int Hk, int S, int conv_ch, int M, float eps) {
    int warpid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int t = warpid / Hk, hk = warpid % Hk;
    if (t >= M) return;
    float* row = qkv + (long)t * conv_ch;
    float sc = rsqrtf((float)S);
    float qv[4], kv[4], sq = 0.f, s2 = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int is = lane * 4 + j;
        qv[j] = row[hk * S + is]; kv[j] = row[Hk * S + hk * S + is];
        sq += qv[j] * qv[j]; s2 += kv[j] * kv[j];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { sq += __shfl_xor_sync(0xffffffffu, sq, o); s2 += __shfl_xor_sync(0xffffffffu, s2, o); }
    float qn = rsqrtf(sq + eps) * sc, kn = rsqrtf(s2 + eps);
    float qk = 0.f;   // (q̂·k̂), col-independent → fuses the y-reduction in the scan
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int is = lane * 4 + j;
        float qh = qv[j] * qn, kh = kv[j] * kn;
        row[hk * S + is] = qh; row[Hk * S + hk * S + is] = kh;
        qk += qh * kh;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) qk += __shfl_xor_sync(0xffffffffu, qk, o);
    if (lane == 0) qkdot[t * Hk + hk] = qk;
}

// Gated-DeltaNet scan for pre-normalized q,k. Each warp processes COLS=4 state
// columns: q̂/k̂ are indexed by state row (column-independent) → loaded once and reused
// across the 4 columns (4× fewer q/k reads), and the 4 columns' recurrences are
// independent → their reductions overlap (ILP hides the sequential latency).
// grid=(S/(4*COLS), Hv), block=128.
#define DCOLS 4

extern "C" __global__ void deltanet_scan(float* state, const float* qkv, const float* qkdot,
    const float* gate, const float* beta, float* out, int S, int Hk, int Hv, int conv_ch, int M) {
    int h = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int col0 = blockIdx.x * (4 * DCOLS) + warp * DCOLS;   // first of this warp's DCOLS cols
    if (col0 >= S) return;
    int hk = h % Hk;
    float ls[DCOLS][4];
    #pragma unroll
    for (int c = 0; c < DCOLS; c++) {
        float* sb = state + (long)(h * S + col0 + c) * S;
        #pragma unroll
        for (int j = 0; j < 4; j++) ls[c][j] = sb[j * 32 + lane];
    }
    // Fused y (qk=q̂·k̂ precomputed, column-independent): y = q̂ᵀS_decayed + qk·d, so sk[c]
    // and yp[c] reduce together and y needs no post-update reduction: one pass.
    #define LOAD_TOK(tt, QV, KV, VC, GR, BT, QK) do { \
        const float* row = qkv + (long)(tt) * conv_ch; \
        _Pragma("unroll") for (int j = 0; j < 4; j++) { int is = j * 32 + lane; QV[j] = row[hk * S + is]; KV[j] = row[Hk * S + hk * S + is]; } \
        _Pragma("unroll") for (int c = 0; c < DCOLS; c++) VC[c] = row[2 * Hk * S + h * S + col0 + c]; \
        GR = gate[(tt) * Hv + h]; BT = beta[(tt) * Hv + h]; QK = qkdot[(tt) * Hk + hk]; \
    } while (0)
    float qv[4], kv[4], vc[DCOLS], gr, bt, qk;
    LOAD_TOK(0, qv, kv, vc, gr, bt, qk);
    for (int t = 0; t < M; t++) {
        float nqv[4], nkv[4], nvc[DCOLS], ngr, nbt, nqk;
        if (t + 1 < M) LOAD_TOK(t + 1, nqv, nkv, nvc, ngr, nbt, nqk);
        float g = __expf(gr);
        float sk[DCOLS], yp[DCOLS];
        #pragma unroll
        for (int c = 0; c < DCOLS; c++) { sk[c] = 0.f; yp[c] = 0.f; _Pragma("unroll") for (int j = 0; j < 4; j++) { ls[c][j] *= g; sk[c] += ls[c][j] * kv[j]; yp[c] += ls[c][j] * qv[j]; } }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            #pragma unroll
            for (int c = 0; c < DCOLS; c++) { sk[c] += __shfl_xor_sync(0xffffffffu, sk[c], o); yp[c] += __shfl_xor_sync(0xffffffffu, yp[c], o); }
        if (lane == 0) {
            #pragma unroll
            for (int c = 0; c < DCOLS; c++) { float d = (vc[c] - sk[c]) * bt; out[(long)t * (Hv * S) + h * S + col0 + c] = yp[c] + qk * d; }
        }
        #pragma unroll
        for (int c = 0; c < DCOLS; c++) { float d = (vc[c] - sk[c]) * bt; _Pragma("unroll") for (int j = 0; j < 4; j++) ls[c][j] += kv[j] * d; }
        #pragma unroll
        for (int j = 0; j < 4; j++) { qv[j] = nqv[j]; kv[j] = nkv[j]; }
        #pragma unroll
        for (int c = 0; c < DCOLS; c++) vc[c] = nvc[c];
        gr = ngr; bt = nbt; qk = nqk;
    }
    #undef LOAD_TOK
    #pragma unroll
    for (int c = 0; c < DCOLS; c++) {
        float* sb = state + (long)(h * S + col0 + c) * S;
        #pragma unroll
        for (int j = 0; j < 4; j++) sb[j * 32 + lane] = ls[c][j];
    }
}
"#;

/// Canonical entry names registered from this family.
pub const NAMES: &[&str] = &[
    "ssm_ab",
    "conv1d_decode",
    "deltanet_fused",
    "conv1d_prefill",
    "deltanet_prenorm",
    "deltanet_scan",
];
