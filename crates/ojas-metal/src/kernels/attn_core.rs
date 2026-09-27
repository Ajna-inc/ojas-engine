// Bodies compile against kernels::PRELUDE (shared defines + helpers).
pub const BODY: &str = r#"
// copy src[0..n] -> dst[off..off+n]
kernel void store_kv(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant uint& off [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) { dst[off+gid] = src[gid]; }
}

// Fused store of k and v into their caches in one dispatch.
kernel void store_kv2(device float* kdst [[buffer(0)]], device const float* ksrc [[buffer(1)]],
    device float* vdst [[buffer(2)]], device const float* vsrc [[buffer(3)]],
    constant uint& n [[buffer(4)]], constant uint& off [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n)        { kdst[off+gid] = ksrc[gid]; }
    else if (gid < 2u*n){ uint j = gid-n; vdst[off+j] = vsrc[j]; }
}

// Store M rows of k/v into cache at positions base_pos..base_pos+M.
kernel void store_kv_m(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
    constant uint& kvdim [[buffer(2)]], constant uint& base_pos [[buffer(3)]], constant uint& M [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    uint total=M*kvdim; if(gid>=total) { return; }
    uint m=gid/kvdim; uint e=gid%kvdim;
    dst[(ulong)(base_pos+m)*(ulong)kvdim + e] = src[gid];
}

// Short-context fast paths (seq ≤ 2048): scores materialized in threadgroup
// memory — measurably faster than the streaming kernels below at short sequences
// (no per-position online-softmax bookkeeping). The dispatch picks by seq.
kernel void attention_m_short(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kvdim [[buffer(5)]], constant uint& base_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]], constant float& scale [[buffer(8)]], constant uint& n_head [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    threadgroup float sc[4096];   // seq cap (16KB threadgroup); host gates batched prefill so seq stays ≤ this
    threadgroup float part[32];
    uint m = tg / n_head; uint head = tg % n_head; uint kvh = head/group;
    uint seq = base_pos + m + 1u;
    uint R = n_head*hd;
    device const float* qh = q + (ulong)m*(ulong)R + head*hd;
    uint sgid = lid / 32u; uint lane = lid % 32u; uint nsg = ts / 32u;
    for (uint t = sgid; t < seq; t += nsg) {
        device const half* kt = kc + (ulong)t*(ulong)kvdim + kvh*hd;
        float sv = 0.0;
        for (uint i = lane; i < hd; i += 32u) { sv += qh[i]*float(kt[i]); }
        sv = simd_sum(sv);
        if (lane == 0u) { sc[t] = sv*scale; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lmax = -1e30;
    for (uint t = lid; t < seq; t += ts) { lmax = max(lmax, sc[t]); }
    lmax = simd_max(lmax);
    if (lane == 0u) { part[sgid] = lmax; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -1e30; for (uint j = 0u; j < nsg; j++) { mx = max(mx, part[j]); }
    float lsum = 0.0;
    for (uint t = lid; t < seq; t += ts) { float e = exp(sc[t]-mx); sc[t] = e; lsum += e; }
    lsum = simd_sum(lsum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) { part[sgid] = lsum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0; for (uint j = 0u; j < nsg; j++) { sum += part[j]; }
    for (uint i = lid; i < hd; i += ts) {
        float acc = 0.0;
        for (uint t = 0; t < seq; t++) { acc += sc[t]*float(vc[(ulong)t*(ulong)kvdim + kvh*hd + i]); }
        out[(ulong)m*(ulong)R + head*hd + i] = acc/sum;
    }
}

// BIDIRECTIONAL twin of attention_m_short for masked-diffusion (Dream/DiffuCoder):
// every query row m attends to ALL `total` positions (no causal cap), single block,
// no KV-cache reuse. Identical math otherwise. buffer(6) carries `total` (the full
// token count) instead of base_pos. seq ≤ 4096 (sc[] threadgroup array).
kernel void attention_m_short_bidir(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kvdim [[buffer(5)]], constant uint& total [[buffer(6)]],
    constant uint& group [[buffer(7)]], constant float& scale [[buffer(8)]], constant uint& n_head [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    threadgroup float sc[4096];   // key cap (16KB threadgroup); >4096 needs streaming attn
    threadgroup float part[32];
    uint m = tg / n_head; uint head = tg % n_head; uint kvh = head/group;
    uint seq = total;                       // BIDIRECTIONAL: attend to every position
    uint R = n_head*hd;
    device const float* qh = q + (ulong)m*(ulong)R + head*hd;
    uint sgid = lid / 32u; uint lane = lid % 32u; uint nsg = ts / 32u;
    for (uint t = sgid; t < seq; t += nsg) {
        device const half* kt = kc + (ulong)t*(ulong)kvdim + kvh*hd;
        float sv = 0.0;
        for (uint i = lane; i < hd; i += 32u) { sv += qh[i]*float(kt[i]); }
        sv = simd_sum(sv);
        if (lane == 0u) { sc[t] = sv*scale; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lmax = -1e30;
    for (uint t = lid; t < seq; t += ts) { lmax = max(lmax, sc[t]); }
    lmax = simd_max(lmax);
    if (lane == 0u) { part[sgid] = lmax; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -1e30; for (uint j = 0u; j < nsg; j++) { mx = max(mx, part[j]); }
    float lsum = 0.0;
    for (uint t = lid; t < seq; t += ts) { float e = exp(sc[t]-mx); sc[t] = e; lsum += e; }
    lsum = simd_sum(lsum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) { part[sgid] = lsum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0; for (uint j = 0u; j < nsg; j++) { sum += part[j]; }
    for (uint i = lid; i < hd; i += ts) {
        float acc = 0.0;
        for (uint t = 0; t < seq; t++) { acc += sc[t]*float(vc[(ulong)t*(ulong)kvdim + kvh*hd + i]); }
        out[(ulong)m*(ulong)R + head*hd + i] = acc/sum;
    }
}

kernel void attention_short(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kvdim [[buffer(5)]], constant uint& seq [[buffer(6)]],
    constant uint& group [[buffer(7)]], constant float& scale [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint ts [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sc[2048];
    threadgroup float part[32];
    threadgroup float qsh[512];
    uint nsg = ts / 32u;
    uint kvh = head/group;
    device const float* qh = q + head*hd;
    for (uint i = lid; i < hd; i += ts) { qsh[i] = qh[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = sg; t < seq; t += nsg) {
        device const half* kt = kc + (ulong)t*(ulong)kvdim + kvh*hd;
        float sv = 0.0;
        for (uint i = lane; i < hd; i += 32u) { sv += qsh[i]*float(kt[i]); }
        sv = simd_sum(sv);
        if (lane == 0u) { sc[t] = sv*scale; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lmax = -1e30;
    for (uint t = lid; t < seq; t += ts) { lmax = max(lmax, sc[t]); }
    lmax = simd_max(lmax);
    if (lane == 0u) { part[sg] = lmax; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -1e30; for (uint j = 0u; j < nsg; j++) { mx = max(mx, part[j]); }
    float lsum = 0.0;
    for (uint t = lid; t < seq; t += ts) { float e = exp(sc[t]-mx); sc[t] = e; lsum += e; }
    lsum = simd_sum(lsum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) { part[sg] = lsum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0; for (uint j = 0u; j < nsg; j++) { sum += part[j]; }
    for (uint i = lid; i < hd; i += ts) {
        float acc = 0.0;
        for (uint t = 0; t < seq; t++) { acc += sc[t]*float(vc[(ulong)t*(ulong)kvdim + kvh*hd + i]); }
        out[head*hd + i] = acc/sum;
    }
}

// GPT-OSS decode attention with a per-head attention sink: attention_short, but a
// learned per-head sink logit sinks[head] is folded into the softmax denominator only
// and contributes no value, letting a head attend to nothing.
// den = Σ_t exp(s_t−m) + exp(sink−m), m = max(max_t s_t, sink).
// Streaming online softmax, so no score array and no context-length cap. Each simdgroup
// owns a strided subset of keys with a running (max, sum, acc) in registers, merged
// through threadgroup memory. A sliding window (gpt-oss even layers, n_swa=128)
// restricts keys to [seq-window, seq).
kernel void attention_short_sink(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kvdim [[buffer(5)]], constant uint& seq [[buffer(6)]],
    constant uint& group [[buffer(7)]], constant float& scale [[buffer(8)]],
    device const float* sinks [[buffer(9)]], constant uint& window [[buffer(10)]],
    uint head [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint ts [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qsh[512];
    threadgroup float tacc[8*512];
    threadgroup float tm[8];
    threadgroup float tl[8];
    uint nsg = ts / 32u;
    uint kvh = head/group;
    uint lo = (window > 0u && seq > window) ? seq - window : 0u;
    device const float* qh = q + head*hd;
    for (uint i = lid; i < hd; i += ts) { qsh[i] = qh[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mi = -1e30, li = 0.0;
    float acc[16] = {0.0};
    uint nch = hd/32u;
    for (uint t = lo + sg; t < seq; t += nsg) {
        device const half* kt = kc + (ulong)t*(ulong)kvdim + kvh*hd;
        float sv = 0.0;
        for (uint i = lane; i < hd; i += 32u) { sv += qsh[i]*float(kt[i]); }
        sv = simd_sum(sv)*scale;
        float mn = max(mi, sv);
        float corr = exp(mi - mn), pw = exp(sv - mn);
        li = li*corr + pw;
        device const half* vt = vc + (ulong)t*(ulong)kvdim + kvh*hd;
        for (uint c = 0u; c < nch; c++) { acc[c] = acc[c]*corr + pw*float(vt[lane + 32u*c]); }
        mi = mn;
    }
    if (lane == 0u) { tm[sg] = mi; tl[sg] = li; }
    for (uint c = 0u; c < nch; c++) { tacc[sg*hd + lane + 32u*c] = acc[c]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sink_h = sinks[head];
    float gm = sink_h; for (uint j = 0u; j < nsg; j++) { gm = max(gm, tm[j]); }
    float gl = exp(sink_h - gm); for (uint j = 0u; j < nsg; j++) { gl += tl[j]*exp(tm[j] - gm); }
    for (uint i = lid; i < hd; i += ts) {
        float o = 0.0;
        for (uint j = 0u; j < nsg; j++) { o += tacc[j*hd + i]*exp(tm[j] - gm); }
        out[head*hd + i] = o/gl;
    }
}

// GQA attention for M query rows (causal: query m attends to cache[0..base_pos+m]).
// Streaming softmax, flash-decoding style: no score array, so no context-length cap and
// constant threadgroup memory on every Apple GPU. Each simdgroup owns a strided subset
// of positions with a running (max, sum, acc) in registers; partials merge through
// threadgroup memory. hd <= 512, threads = 256 (8 simdgroups).
kernel void attention_m(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kvdim [[buffer(5)]], constant uint& base_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]], constant float& scale [[buffer(8)]], constant uint& n_head [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    threadgroup float qsh[512];
    threadgroup float tacc[8*512];
    threadgroup float tm[8];
    threadgroup float tl[8];
    uint m = tg / n_head; uint head = tg % n_head; uint kvh = head/group;
    uint seq = base_pos + m + 1u;
    uint R = n_head*hd;
    uint sgid = lid / 32u; uint lane = lid % 32u; uint nsg = ts / 32u;
    device const float* qh = q + (ulong)m*(ulong)R + head*hd;
    for (uint i = lid; i < hd; i += ts) { qsh[i] = qh[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mi = -1e30, li = 0.0;
    float acc[16] = {0.0};
    uint nch = hd/32u;
    for (uint t = sgid; t < seq; t += nsg) {
        device const half* kt = kc + (ulong)t*(ulong)kvdim + kvh*hd;
        float sv = 0.0;
        for (uint i = lane; i < hd; i += 32u) { sv += qsh[i]*float(kt[i]); }
        sv = simd_sum(sv)*scale;
        float mn = max(mi, sv);
        float corr = exp(mi - mn); float pw = exp(sv - mn);
        li = li*corr + pw;
        device const half* vt = vc + (ulong)t*(ulong)kvdim + kvh*hd;
        for (uint c = 0u; c < nch; c++) { acc[c] = acc[c]*corr + pw*float(vt[lane + 32u*c]); }
        mi = mn;
    }
    // merge simdgroup partials
    if (lane == 0u) { tm[sgid] = mi; tl[sgid] = li; }
    for (uint c = 0u; c < nch; c++) { tacc[sgid*hd + lane + 32u*c] = acc[c]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = -1e30; for (uint j = 0u; j < nsg; j++) { gm = max(gm, tm[j]); }
    float gl = 0.0; for (uint j = 0u; j < nsg; j++) { gl += tl[j]*exp(tm[j] - gm); }
    for (uint i = lid; i < hd; i += ts) {
        float o = 0.0;
        for (uint j = 0u; j < nsg; j++) { o += tacc[j*hd + i]*exp(tm[j] - gm); }
        out[(ulong)m*(ulong)R + head*hd + i] = o/gl;
    }
}

// STREAMING bidirectional twin of attention_m for masked diffusion: every query row m
// attends to ALL `total` positions (online-softmax, no score array → UNBOUNDED context,
// unlike attention_m_short_bidir's sc[4096]). buffer(6) = total. hd ≤ 512.
kernel void attention_m_bidir(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kvdim [[buffer(5)]], constant uint& total [[buffer(6)]],
    constant uint& group [[buffer(7)]], constant float& scale [[buffer(8)]], constant uint& n_head [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    threadgroup float qsh[512];
    threadgroup float tacc[8*512];
    threadgroup float tm[8];
    threadgroup float tl[8];
    uint m = tg / n_head; uint head = tg % n_head; uint kvh = head/group;
    uint seq = total;                      // BIDIRECTIONAL: attend to every position
    uint R = n_head*hd;
    uint sgid = lid / 32u; uint lane = lid % 32u; uint nsg = ts / 32u;
    device const float* qh = q + (ulong)m*(ulong)R + head*hd;
    for (uint i = lid; i < hd; i += ts) { qsh[i] = qh[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mi = -1e30, li = 0.0;
    float acc[16] = {0.0};
    uint nch = hd/32u;
    for (uint t = sgid; t < seq; t += nsg) {
        device const half* kt = kc + (ulong)t*(ulong)kvdim + kvh*hd;
        float sv = 0.0;
        for (uint i = lane; i < hd; i += 32u) { sv += qsh[i]*float(kt[i]); }
        sv = simd_sum(sv)*scale;
        float mn = max(mi, sv); float corr = exp(mi - mn); float pw = exp(sv - mn);
        li = li*corr + pw;
        device const half* vt = vc + (ulong)t*(ulong)kvdim + kvh*hd;
        for (uint c = 0u; c < nch; c++) { acc[c] = acc[c]*corr + pw*float(vt[lane + 32u*c]); }
        mi = mn;
    }
    if (lane == 0u) { tm[sgid] = mi; tl[sgid] = li; }
    for (uint c = 0u; c < nch; c++) { tacc[sgid*hd + lane + 32u*c] = acc[c]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = -1e30; for (uint j = 0u; j < nsg; j++) { gm = max(gm, tm[j]); }
    float gl = 0.0; for (uint j = 0u; j < nsg; j++) { gl += tl[j]*exp(tm[j] - gm); }
    for (uint i = lid; i < hd; i += ts) {
        float o = 0.0;
        for (uint j = 0u; j < nsg; j++) { o += tacc[j*hd + i]*exp(tm[j] - gm); }
        out[(ulong)m*(ulong)R + head*hd + i] = o/gl;
    }
}

// GPT-OSS batched (M-query) causal attention with a per-head sink: attention_m plus the
// sink logit sinks[head] folded into the global softmax denominator, contributing no
// value. Two-pass, with scores materialized in sc[], to keep byte-for-byte the same
// summation order as the decode attention_short_sink, so batched-prefill KV gives
// identical logits — an online-flash variant drifted argmax on borderline prompts.
// Caps seq <= 4096; the host guards gpt-oss prefill chunks so base_pos+m stays within.
kernel void attention_m_sink(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kvdim [[buffer(5)]], constant uint& base_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]], constant float& scale [[buffer(8)]], constant uint& n_head [[buffer(9)]],
    device const float* sinks [[buffer(10)]], constant uint& window [[buffer(11)]],
    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    threadgroup float qsh[512];
    threadgroup float tacc[8*512];
    threadgroup float tm[8];
    threadgroup float tl[8];
    uint m = tg / n_head; uint head = tg % n_head; uint kvh = head/group;
    uint seq = base_pos + m + 1u;
    uint lo = (window > 0u && seq > window) ? seq - window : 0u;
    uint R = n_head*hd;
    uint sgid = lid / 32u; uint lane = lid % 32u; uint nsg = ts / 32u;
    device const float* qh = q + (ulong)m*(ulong)R + head*hd;
    for (uint i = lid; i < hd; i += ts) { qsh[i] = qh[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mi = -1e30, li = 0.0;
    float acc[16] = {0.0};
    uint nch = hd/32u;
    for (uint t = lo + sgid; t < seq; t += nsg) {
        device const half* kt = kc + (ulong)t*(ulong)kvdim + kvh*hd;
        float sv = 0.0;
        for (uint i = lane; i < hd; i += 32u) { sv += qsh[i]*float(kt[i]); }
        sv = simd_sum(sv)*scale;
        float mn = max(mi, sv);
        float corr = exp(mi - mn), pw = exp(sv - mn);
        li = li*corr + pw;
        device const half* vt = vc + (ulong)t*(ulong)kvdim + kvh*hd;
        for (uint c = 0u; c < nch; c++) { acc[c] = acc[c]*corr + pw*float(vt[lane + 32u*c]); }
        mi = mn;
    }
    if (lane == 0u) { tm[sgid] = mi; tl[sgid] = li; }
    for (uint c = 0u; c < nch; c++) { tacc[sgid*hd + lane + 32u*c] = acc[c]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sink_h = sinks[head];
    float gm = sink_h; for (uint j = 0u; j < nsg; j++) { gm = max(gm, tm[j]); }
    float gl = exp(sink_h - gm); for (uint j = 0u; j < nsg; j++) { gl += tl[j]*exp(tm[j] - gm); }
    for (uint i = lid; i < hd; i += ts) {
        float o = 0.0;
        for (uint j = 0u; j < nsg; j++) { o += tacc[j*hd + i]*exp(tm[j] - gm); }
        out[(ulong)m*(ulong)R + head*hd + i] = o/gl;
    }
}

"#;
