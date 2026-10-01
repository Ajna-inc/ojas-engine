// Bodies compile against kernels::PRELUDE (shared defines + helpers).
pub const BODY: &str = r#"
// ===================== qwen35 Gated-DeltaNet (SSM) kernels =====================
// alpha/beta post-processing (per dt_rank head): gate = softplus(alpha+dt)*a; beta = sigmoid(beta).
// n = total elements (M tokens × hv rows); dt/a weights [hv] indexed modulo.
kernel void ssm_ab(device float* alpha [[buffer(0)]], device float* beta [[buffer(1)]],
    device const float* dt [[buffer(2)]], device const float* a [[buffer(3)]], constant uint& n [[buffer(4)]],
    constant uint& hv [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= n) { return; }
    uint w = i % hv;
    float x = alpha[i] + dt[w];
    float sp = x > 20.0 ? x : log(1.0 + exp(x));   // softplus (clamped)
    alpha[i] = sp * a[w];                          // gate (decay, pre-exp)
    beta[i] = 1.0/(1.0 + exp(-beta[i]));           // sigmoid
}

// Scalar GDN specialization for F32 alpha/beta matrices. Sharing the activation
// walk and applying ssm_ab's epilogue here replaces three dispatches while
// retaining the same per-row SIMD reduction order. K4 is hidden_size/4.
kernel void gdn_ab_fused(device const float4* x [[buffer(0)]],
    device const float4* wa [[buffer(1)]], device const float4* wb [[buffer(2)]],
    device float* alpha [[buffer(3)]], device float* beta [[buffer(4)]],
    device const float* dt [[buffer(5)]], device const float* a [[buffer(6)]],
    constant uint& K4 [[buffer(7)]], constant uint& HV [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint row = tg*(ts/32u) + sg; if (row >= HV) { return; }
    float pa = 0.0f, pb = 0.0f;
    for (uint k = lane; k < K4; k += 32u) {
        float4 v = x[k];
        pa += dot(wa[row*K4 + k], v);
        pb += dot(wb[row*K4 + k], v);
    }
    pa = simd_sum(pa); pb = simd_sum(pb);
    if (lane == 0u) {
        float z = pa + dt[row];
        float sp = z > 20.0f ? z : log(1.0f + exp(z));
        alpha[row] = sp * a[row];
        beta[row] = 1.0f/(1.0f + exp(-pb));
    }
}

// Causal depthwise conv1d for one token + SILU + rolling conv-state update.
// conv_w layout: [K taps contiguous per channel] (ssm_conv, ne[0]=K).
kernel void conv1d_decode(device float* qkv [[buffer(0)]], device float* cstate [[buffer(1)]],
    device const float* cw [[buffer(2)]], constant uint& n_ch [[buffer(3)]], constant uint& K [[buffer(4)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= n_ch) { return; }
    float acc = cw[c*K + (K-1u)] * qkv[c];                     // newest tap × current input
    for (uint j = 0u; j < K-1u; j++) { acc += cw[c*K + j] * cstate[j*n_ch + c]; }
    for (uint j = 0u; j + 2u < K; j++) { cstate[j*n_ch + c] = cstate[(j+1u)*n_ch + c]; } // shift
    cstate[(K-2u)*n_ch + c] = qkv[c];                          // append current
    qkv[c] = acc / (1.0 + exp(-acc));                          // SILU
}

// Causal depthwise conv1d over M tokens + SILU + conv-state update. One thread per
// channel; the K-1 tap history lives in registers across the token loop, so the
// device conv_state is read once and written once per chunk.
kernel void conv1d_prefill(device float* qkv [[buffer(0)]], device float* cstate [[buffer(1)]],
    device const float* cw [[buffer(2)]], constant uint& n_ch [[buffer(3)]], constant uint& K [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    device float* snap [[buffer(6)]], constant uint& snap_t [[buffer(7)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= n_ch) { return; }
    float st[8]; float wk[8];
    for (uint j = 0u; j < K-1u; j++) { st[j] = cstate[j*n_ch + c]; }
    for (uint j = 0u; j < K; j++)    { wk[j] = cw[c*K + j]; }
    for (uint t = 0u; t < M; t++) {
        float xin = qkv[(ulong)t*(ulong)n_ch + c];
        float acc = wk[K-1u]*xin;
        for (uint j = 0u; j < K-1u; j++) { acc += wk[j]*st[j]; }
        for (uint j = 0u; j + 2u < K; j++) { st[j] = st[j+1u]; }
        st[K-2u] = xin;
        qkv[(ulong)t*(ulong)n_ch + c] = acc / (1.0 + exp(-acc));
        // UINT_MAX disables snapshots. Otherwise high bit requests every row;
        // low 31 bits are the full conv-state stride, including appended PLE.
        bool all_snap = snap_t != 0xffffffffu && (snap_t & 0x80000000u) != 0u;
        if (t == snap_t || all_snap) {
            ulong off = all_snap ? (ulong)t * (snap_t & 0x7fffffffu) : 0ul;
            for (uint j = 0u; j < K-1u; j++) { snap[off + j*n_ch + c] = st[j]; }
        }
    }
    for (uint j = 0u; j < K-1u; j++) { cstate[j*n_ch + c] = st[j]; }
}

// L2-normalize each q and k head of the conv output in place, q also scaled by
// 1/sqrt(S): the normalization `deltanet_fused` would otherwise repeat in every one
// of a head's S columns. One simdgroup per (head, token); heads 0..H_k-1 are q,
// H_k..2H_k-1 are k, contiguous at the start of each conv_ch row. Same arithmetic
// as `deltanet_fused`'s l2_mode 0 and 1. Assumes S == 128.
kernel void qk_l2norm_heads(device float* qkv [[buffer(0)]], constant uint& S [[buffer(1)]],
    constant uint& H_k [[buffer(2)]], constant uint& conv_ch [[buffer(3)]], constant float& eps [[buffer(4)]],
    constant uint& clamp_l2 [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    device float* v = qkv + (ulong)tg.y*(ulong)conv_ch + tg.x*S + lane*4u;
    float x[4]; float ss = 0.0;
    _Pragma("unroll")
    for (short j = 0; j < 4; j++) { x[j] = v[j]; ss += x[j]*x[j]; }
    ss = simd_sum(ss);
    float n = clamp_l2 ? 1.0/max(sqrt(ss), eps) : rsqrt(ss + eps);
    if (tg.x < H_k) { n *= 1.0/sqrt(float(S)); }
    _Pragma("unroll")
    for (short j = 0; j < 4; j++) { v[j] = x[j]*n; }
}

// Gated-DeltaNet recurrence over M tokens, gated-delta-net kernel style: one simdgroup
// per state column, the 128 column values living in registers (4 per lane) and
// reductions via simd_sum — no threadgroup memory, no barriers, and S_v×H_v simdgroups
// per layer, 8× the parallelism of a threadgroup-resident version. Decode fuses the
// per-head q/k L2-norm in, with the 1/sqrt(S) scale folded into the q normalizer;
// prefill normalizes beforehand (`l2_mode` 2). Token-serial in-kernel, the
// recurrence being inherently sequential. Used by decode (M=1) and prefill.
// q/k/v are slices of the RAW conv output rows ([M, conv_ch]); gate/beta rows
// [M, H_v]; out rows [M, H_v*S]. Assumes S == 128 (4 regs × 32 lanes).
// `l2_mode`: 0 normalizes q/k with eps inside the square root, 1 with eps as a
// floor (diagnostics), 2 takes them already normalized and q pre-scaled by
// 1/sqrt(S) (`qk_l2norm_heads`), which saves every column of a head two of the
// four reductions per token.
kernel void deltanet_fused(device float* state [[buffer(0)]], device const float* qkv [[buffer(1)]],
    device const float* gate [[buffer(2)]], device const float* beta [[buffer(3)]], device float* out [[buffer(4)]],
    constant uint& S [[buffer(5)]], constant uint& H_k [[buffer(6)]], constant uint& H_v [[buffer(7)]],
    constant uint& conv_ch [[buffer(8)]], constant uint& M [[buffer(9)]], constant float& eps [[buffer(10)]],
    device float* snap [[buffer(11)]], constant uint& snap_t [[buffer(12)]],
    constant uint& kmap_div [[buffer(13)]],
    constant uint& l2_mode [[buffer(14)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    uint h = tg.y;
    uint col = tg.x*4u + sgid;               // 4 simdgroups/tg, one column each
    // Value head -> key head. Both layouts occur in the wild.
    //
    // HF stores V heads grouped by key head: [G0_v0..v{r-1}, G1_v0..v{r-1}, ...], i.e.
    // h / (H_v/H_k). The llama.cpp converter's _LinearAttentionVReorderBase permutes
    // them to tiled order [G0_v0, G1_v0, ..., G0_v1, ...] so ggml_repeat can broadcast
    // instead of doing an interleaved repeat, and tiled is h % H_k.
    // Qwen3.5 (and so Ornith) inherits that reorder; Qwen3Next does not. That is why
    // cpu_ssm builds kmap two different ways — `%` off GGUF, `/` off safetensors —
    // and both are correct for their source.
    //
    // The two agree only when H_v == H_k, so any model with a different H_v/H_k ratio
    // must state which layout it has rather than inherit the ratio-2 answer.
    uint hk = kmap_div ? (h / max(H_v / max(H_k, 1u), 1u)) : (h % H_k);
    float scale = 1.0/sqrt(float(S));
    device float* sb = state + (ulong)(h*S + col)*(ulong)S;
    float ls[4];
    _Pragma("unroll")
    for (short j = 0; j < 4; j++) { ls[j] = sb[lane*4u + j]; }
    for (uint t = 0u; t < M; t++) {
        device const float* row = qkv + (ulong)t*(ulong)conv_ch;
        float qv[4], kv[4];
        _Pragma("unroll")
        for (short j = 0; j < 4; j++) {
            uint is = lane*4u + j;
            qv[j] = row[hk*S + is];
            kv[j] = row[H_k*S + hk*S + is];
        }
        float qn = 1.0, kn = 1.0;
        if (l2_mode != 2u) {
            float sq = 0.0, s2 = 0.0;
            _Pragma("unroll")
            for (short j = 0; j < 4; j++) { sq += qv[j]*qv[j]; s2 += kv[j]*kv[j]; }
            sq = simd_sum(sq); s2 = simd_sum(s2);
            // GDN (including Flash) uses epsilon inside the squared norm.
            // The clamp mode is retained for explicit normalization diagnostics.
            qn = l2_mode == 1u ? scale/max(sqrt(sq), eps) : rsqrt(sq + eps)*scale;
            kn = l2_mode == 1u ? 1.0/max(sqrt(s2), eps) : rsqrt(s2 + eps);
        }
        float g = exp(gate[t*H_v + h]); float bet = beta[t*H_v + h];
        float sk = 0.0;
        _Pragma("unroll")
        for (short j = 0; j < 4; j++) { ls[j] *= g; sk += ls[j]*kv[j]; }
        sk = simd_sum(sk) * kn;
        float d = (row[2u*H_k*S + h*S + col] - sk) * bet;
        float y = 0.0;
        _Pragma("unroll")
        for (short j = 0; j < 4; j++) { ls[j] += kv[j]*kn*d; y += ls[j]*qv[j]; }
        y = simd_sum(y) * qn;
        if (lane == 0) { out[(ulong)t*(ulong)(H_v*S) + h*S + col] = y; }
        bool all_snap = snap_t != 0xffffffffu && (snap_t & 0x80000000u) != 0u;
        if (t == snap_t || all_snap) {
            ulong off = all_snap ? (ulong)t * H_v * S * S : 0ul;
            device float* sn = snap + off + (ulong)(h*S + col)*(ulong)S;
            _Pragma("unroll")
            for (short j = 0; j < 4; j++) { sn[lane*4u + j] = ls[j]; }
        }
    }
    _Pragma("unroll")
    for (short j = 0; j < 4; j++) { sb[lane*4u + j] = ls[j]; }
}
"#;
