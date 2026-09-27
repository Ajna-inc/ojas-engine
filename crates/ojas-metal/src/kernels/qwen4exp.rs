// Hyper-connection mixer primitives. The residual state is `hc` parallel copies
// of the d-wide hidden state, laid out [d, hc] per token. These kernels implement
// the per-stream norm, the low-rank gate's activations, the mean collapse to a
// single d-wide view, and the weighted scatter back into the streams. The two
// low-rank projections are ordinary mat-vecs handled by the shared GEMM path.
pub const BODY: &str = r#"

// NextN input layout is [token][stream][embedding || hidden]. The normalized
// embedding is shared; each normalized hidden stream must remain distinct.
kernel void nextn_concat_streams(device const float* emb [[buffer(0)]],
    device const float* hidden [[buffer(1)]], device float* out [[buffer(2)]],
    constant uint& d [[buffer(3)]], constant uint& hc [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x, c = gid.y, t = gid.z;
    if (i >= d) return;
    ulong row = ulong(t) * hc + c;
    out[row * 2u * d + i] = emb[ulong(t) * d + i];
    out[row * 2u * d + d + i] = hidden[row * d + i];
}

// Per-stream RMSNorm followed by the [d*hc] gamma. x, out: [d, hc]; one
// threadgroup per stream reduces over the d elements it owns.
// Buffers are token-major: hc_res/hc_xn/hc_gated are [T][hc][d], hc_mixed [T][d],
// hc_inject [T][hc], hc_lo [T][hc_lr]. At T=1 that is the old layout exactly, so
// the decode path is unchanged.
kernel void hc_rmsnorm(device const float* x [[buffer(0)]], device const float* gamma [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& d [[buffer(3)]], constant float& eps [[buffer(4)]],
    constant uint& hc [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid2 [[thread_position_in_threadgroup]],
    uint2 ts2 [[threads_per_threadgroup]]) {
    threadgroup float part[8];
    // MSL wants every position attribute scalar or every one a vector of the same
    // width, so these come in as uint2 and the unused axis is dropped here.
    uint lid = lid2.x, ts = ts2.x;
    uint c = tg.x, t = tg.y;
    ulong base = (ulong)(t * hc + c) * (ulong)d;
    device const float* xc = x + base;
    float acc = 0.0;
    for (uint i = lid; i < d; i += ts) { float v = xc[i]; acc += v * v; }
    acc = simd_sum(acc);
    uint sg = lid / 32u, lane = lid % 32u;
    if (lane == 0u) { part[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0;
    for (uint j = 0u; j < ts / 32u; j++) { total += part[j]; }
    float sc = rsqrt(total / float(d) + eps);
    for (uint i = lid; i < d; i += ts) { out[base + i] = xc[i] * sc * gamma[c * d + i]; }
}

// lo = silu(lo_raw / hc), applied in place over the low-rank vector.
kernel void hc_silu_scale(device float* lo [[buffer(0)]], constant uint& n [[buffer(1)]],
    constant uint& hc [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = lo[i] / float(hc);
    lo[i] = v / (1.0 + exp(-v));
}

// gated = xn * sigmoid(gate_raw), over the full [d*hc] stream vector.
kernel void hc_gate(device const float* xn [[buffer(0)]], device const float* gate_raw [[buffer(1)]],
    device float* gated [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    gated[i] = xn[i] * (1.0 / (1.0 + exp(-gate_raw[i])));
}

// out[i] = mean over the hc streams of gated[c*d + i].
kernel void hc_collapse(device const float* gated [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& d [[buffer(2)]], constant uint& hc [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    if (i >= d) return;
    ulong base = (ulong)t * (ulong)hc * (ulong)d;
    float acc = 0.0;
    for (uint c = 0u; c < hc; c++) { acc += gated[base + (ulong)c * (ulong)d + i]; }
    out[(ulong)t * (ulong)d + i] = acc / float(hc);
}

// Scalar qwen4exp HC fast path. The host admits only d=2560, hc=4, lr=320,
// native Q8_0 down/up and F32 injection tensors. Keeping this specialization
// explicit prevents an unmeasured geometry from silently taking it.
inline float hc_q8(device const uchar* w, uint row, uint k, uint K) {
    ulong b = ((ulong)row * (K / 32u) + k / 32u) * 34u;
    ushort bits = (ushort)w[b] | ((ushort)w[b + 1u] << 8);
    return float(as_type<half>(bits)) * float(as_type<char>(w[b + 2u + k % 32u]));
}

kernel void hc_down_inject_fused(device const float* xn [[buffer(0)]],
    device const uchar* down [[buffer(1)]], device const float* inject_w [[buffer(2)]],
    device float* lo [[buffer(3)]], device float* inject [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4];
    float acc = 0.0f;
    if (row < 320u) {
        for (uint i = tid; i < 10240u; i += 128u) acc += xn[i] * hc_q8(down, row, i, 10240u);
    } else {
        uint r = row - 320u;
        for (uint i = tid; i < 10240u; i += 128u) acc += xn[i] * inject_w[r * 10240u + i];
    }
    acc = simd_sum(acc);
    if (lane == 0u) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float z = part[0] + part[1] + part[2] + part[3];
        if (row < 320u) {
            z *= 0.25f;
            lo[row] = z / (1.0f + exp(-z));
        } else {
            inject[row - 320u] = z;
        }
    }
}

kernel void hc_up_gate_collapse_fused(device const float* lo [[buffer(0)]],
    device const uchar* up [[buffer(1)]], device const float* xn [[buffer(2)]],
    device float* mixed [[buffer(3)]],
    uint i [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4][4];
    float4 acc = 0.0f;
    for (uint j = tid; j < 320u; j += 128u) {
        float x = lo[j];
        acc.x += x * hc_q8(up, i,          j, 320u);
        acc.y += x * hc_q8(up, 2560u + i,  j, 320u);
        acc.z += x * hc_q8(up, 5120u + i,  j, 320u);
        acc.w += x * hc_q8(up, 7680u + i,  j, 320u);
    }
    acc.x = simd_sum(acc.x); acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z); acc.w = simd_sum(acc.w);
    if (lane == 0u) { part[sg][0] = acc.x; part[sg][1] = acc.y; part[sg][2] = acc.z; part[sg][3] = acc.w; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float out = 0.0f;
        for (uint c = 0u; c < 4u; ++c) {
            float z = part[0][c] + part[1][c] + part[2][c] + part[3][c];
            out += xn[c * 2560u + i] / (1.0f + exp(-z));
        }
        mixed[i] = out * 0.25f;
    }
}

// Scatter the block output back into every stream, weighted per stream by
// 2*sigmoid(inject[c]/hc) (centred on 1 so an untrained injection is a plain add).
kernel void hc_combine(device float* res [[buffer(0)]], device const float* block [[buffer(1)]],
    device const float* inject [[buffer(2)]], constant uint& d [[buffer(3)]], constant uint& hc [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x, c = gid.y, t = gid.z;
    if (i >= d || c >= hc) return;
    float w = 2.0 * (1.0 / (1.0 + exp(-inject[t * hc + c] / float(hc))));
    res[(ulong)(t * hc + c) * (ulong)d + i] += block[(ulong)t * (ulong)d + i] * w;
}

// PLE n-gram gather: emb[h*hd + i] = table[rows[h]*hd + i], flattening the head
// axis so the result is one d-wide vector (d == hd * n_heads).
kernel void ple_gather(device const float* table [[buffer(0)]], device const int* rows [[buffer(1)]],
    device float* emb [[buffer(2)]], constant uint& hd [[buffer(3)]], constant uint& n_heads [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= hd * n_heads) return;
    uint h = gid / hd, i = gid % hd;
    emb[gid] = table[(uint)rows[h] * hd + i];
}

// Per-stream indexer gate: s[c] = (Σ_i keyn[c*d+i]·queryn[c*d+i]) / sqrt(d), then a
// signed square root before the sigmoid, giving one gate scalar per stream.
// Same gather over an IQ4_NL-packed table: one row is `hd` weights = hd/32 blocks of 18
// bytes. Only `n_heads` rows are read per token, which is why the table stays mmap'd and
// cold: at 51.2B parameters, materializing it as f16 to use the f32 gather above would
// cost ~100 GB to serve ~1.4 KB of reads.
// `hd` must be a multiple of 32; the host asserts it.
kernel void ple_gather_iq4nl(device const uchar* table [[buffer(0)]], device const int* rows [[buffer(1)]],
    device float* emb [[buffer(2)]], constant uint& hd [[buffer(3)]], constant uint& n_heads [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= hd * n_heads) return;
    uint h = gid / hd, i = gid % hd;
    uint rb = hd / 32u * 18u;
    device const uchar* blk = table + (ulong)((uint)rows[h]) * (ulong)rb + (ulong)(i / 32u) * 18u;
    ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);
    float d = (float)as_type<half>(dbits);
    uint j = i % 32u;                    // low nibbles are outputs 0..15, high 16..31
    uint q = (uint)blk[2u + (j & 15u)];
    emb[gid] = d * (float)kvalues_iq4nl_p[(j < 16u) ? (q & 15u) : (q >> 4)];
}

kernel void ple_gate(device const float* keyn [[buffer(0)]], device const float* queryn [[buffer(1)]],
    device float* gate [[buffer(2)]], constant uint& d [[buffer(3)]],
    uint c [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    threadgroup float red[256];
    float acc = 0.0;
    for (uint i = lid; i < d; i += ts) { acc += keyn[c * d + i] * queryn[c * d + i]; }
    red[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = ts / 2u; k > 0u; k >>= 1u) { if (lid < k) red[lid] += red[lid + k]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (lid == 0u) {
        float s = red[0] / sqrt(float(d));
        float mag = sqrt(clamp(fabs(s), 1e-6, 1e30));
        float sgn = s > 0.0 ? 1.0 : (s < 0.0 ? -1.0 : 0.0);
        gate[c] = 1.0 / (1.0 + exp(-(sgn * mag)));
    }
}

// gated[c*d+i] = value[i] * gate[c] (value broadcast across the streams).
kernel void ple_bmul(device const float* value [[buffer(0)]], device const float* gate [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& d [[buffer(3)]], constant uint& hc [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, c = gid.y;
    if (i >= d || c >= hc) return;
    out[c * d + i] = value[i] * gate[c];
}

// Dilated causal convolution. GGUF weights are [kernel, channels], with
// the kernel axis contiguous. History is channel-major, oldest sample first.
// Each thread owns one channel, so shifts need no cross-thread synchronization.
kernel void ple_finish(device float* res [[buffer(0)]], device const float* value [[buffer(1)]],
    device const float* gate [[buffer(2)]], device const float* norm [[buffer(3)]],
    device const float* weights [[buffer(4)]], constant uint& d [[buffer(5)]], constant uint& hc [[buffer(6)]],
    device float* history [[buffer(7)]], constant uint& kern [[buffer(8)]], constant uint& dilation [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, c = gid.y;
    if (i >= d || c >= hc) return;
    uint ch = c * d + i;
    uint hist = (kern - 1) * dilation;
    float cv = weights[ch * kern + kern - 1] * norm[ch];
    for (uint k = 0; k + 1 < kern; ++k)
        cv += weights[ch * kern + k] * history[ch * hist + k * dilation];
    for (uint t = 0; t + 1 < hist; ++t)
        history[ch * hist + t] = history[ch * hist + t + 1];
    if (hist) history[ch * hist + hist - 1] = norm[ch];
    res[ch] += value[i] * gate[c] + cv / (1.0 + exp(-cv));
}

// Zero a d-wide buffer (used to reset a block output before an accumulating pass).
kernel void hc_zero(device float* x [[buffer(0)]], constant uint& d [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i < d) x[i] = 0.0;
}

// Broadcast a d-wide vector into hc identical streams (residual initialisation).
kernel void hc_broadcast(device const float* x [[buffer(0)]], device float* res [[buffer(1)]],
    constant uint& d [[buffer(2)]], constant uint& hc [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x, c = gid.y, t = gid.z;
    if (i >= d || c >= hc) return;
    res[(ulong)(t * hc + c) * (ulong)d + i] = x[(ulong)t * (ulong)d + i];
}
"#;
