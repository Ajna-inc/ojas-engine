// Bodies compile against kernels::PRELUDE (shared defines + helpers).
pub const BODY: &str = r#"
// Two identical trivial kernels used only to measure pure pipeline-switch cost:
// dispatching probe0 repeatedly (no switch) vs alternating probe0/probe1 (switch
// every dispatch) isolates the setComputePipelineState overhead.
// f32 -> f16 copy with zero padding past `valid`. Feeds attention_m_mma_dq: the MMA
// needs Q as half fragments, and rows past the token count must read as zero rather
// than whatever was last in the buffer — a stale NaN would survive the causal mask,
// which forces the score to -inf but not the operand.
kernel void q_to_half(device const float* q [[buffer(0)]], device half* qh [[buffer(1)]],
    constant uint& valid [[buffer(2)]], constant uint& total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) { return; }
    qh[gid] = (gid < valid) ? half(q[gid]) : half(0.0);
}

kernel void probe0(device float* y [[buffer(0)]], uint gid [[thread_position_in_grid]]) { if (gid==0u) y[0]+=1.0; }

// GPU argmax over the logits: one threadgroup scans all `n` logits and writes the max
// index to out[0]. Avoids copying 600 KB of logits to the CPU and scanning there every
// token, which was ~15% of per-token wall time.
// Row-wise argmax for a batched forward: one threadgroup per row, out[row] = id.
// The speculative verify needs only each row's argmax, and returning logits means
// copying vocab*M floats to the host — 4.9 MB for M=8 at a 152k vocab, the dominant
// cost of the verify pass (M=1 batched: 19.9 ms against a single forward's 8.24 ms).
// This reads back 4 bytes per row instead.
// Row-wise top-K (K <= 8) token ids for a batched forward — the Token Recycling
// primitive (Luo et al., ACL 2025). The verify pass already computes full logits for
// every position; the top-8 at each position are the model's own guesses about its
// next token, and recycling them into an adjacency table gives a drafter that works
// on novel text, where prompt-lookup cannot. Each thread keeps a local top-K over a
// strided slice (the global top-K is a subset of the union), then lane 0 merges the
// threadgroup.
kernel void topk_m(device const float* logits [[buffer(0)]], device uint* out [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant uint& m [[buffer(3)]], constant uint& kk [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint ts [[threads_per_threadgroup]]) {
    if (tg >= m) { return; }
    threadgroup float tv[256*8];
    threadgroup uint  ti[256*8];
    device const float* row = logits + (ulong)tg*(ulong)n;
    float lv[8]; uint li[8];
    for (uint j = 0u; j < 8u; j++) { lv[j] = -1e30; li[j] = 0u; }
    for (uint i = lid; i < n; i += ts) {
        float x = row[i];
        if (x > lv[7]) {
            uint j = 7u;
            while (j > 0u && x > lv[j-1u]) { lv[j] = lv[j-1u]; li[j] = li[j-1u]; j--; }
            lv[j] = x; li[j] = i;
        }
    }
    for (uint j = 0u; j < 8u; j++) { tv[lid*8u+j] = lv[j]; ti[lid*8u+j] = li[j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        // 8 selection passes over the 256 per-thread heads
        uint head[256];
        for (uint t = 0u; t < ts; t++) { head[t] = 0u; }
        for (uint j = 0u; j < kk; j++) {
            float best = -1e30; uint bt = 0u;
            for (uint t = 0u; t < ts; t++) {
                if (head[t] < 8u && tv[t*8u+head[t]] > best) { best = tv[t*8u+head[t]]; bt = t; }
            }
            out[tg*kk + j] = ti[bt*8u + head[bt]];
            head[bt]++;
        }
    }
}

kernel void argmax_m(device const float* logits [[buffer(0)]], device uint* out [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant uint& m [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint ts [[threads_per_threadgroup]]) {
    if (tg >= m) { return; }
    threadgroup float vals[1024];
    threadgroup uint idxs[1024];
    device const float* row = logits + (ulong)tg*(ulong)n;
    float bv = -1e30; uint bi = 0u;
    for (uint i = lid; i < n; i += ts) { float x = row[i]; if (x > bv) { bv = x; bi = i; } }
    vals[lid] = bv; idxs[lid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = ts/2u; off > 0u; off >>= 1u) {
        if (lid < off) { if (vals[lid+off] > vals[lid]) { vals[lid] = vals[lid+off]; idxs[lid] = idxs[lid+off]; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) { out[tg] = idxs[0]; }
}

kernel void argmax(device const float* logits [[buffer(0)]], device uint* out [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    threadgroup float vals[1024];
    threadgroup uint idxs[1024];
    float bv = -1e30; uint bi = 0u;
    for (uint i = lid; i < n; i += ts) { float x = logits[i]; if (x > bv) { bv = x; bi = i; } }
    vals[lid] = bv; idxs[lid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = ts/2u; off > 0u; off >>= 1u) {
        if (lid < off) { if (vals[lid+off] > vals[lid]) { vals[lid] = vals[lid+off]; idxs[lid] = idxs[lid+off]; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) { out[0] = idxs[0]; }
}

kernel void probe1(device float* y [[buffer(0)]], uint gid [[thread_position_in_grid]]) { if (gid==0u) y[0]+=1.0; }

// RMSNorm on one threadgroup. Reduction via simd_sum (1 barrier) instead of an
// 8-step shared-memory tree (8 barriers) — this kernel runs 2×/layer and its
// latency (single threadgroup, barrier-bound) is a real slice of per-token time.
kernel void rmsnorm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& d [[buffer(3)]], constant float& eps [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];   // one partial per simdgroup (256 threads = 8 sg)
    float s = 0.0;
    for (uint i = lid; i < d; i += ts) { s += x[i]*x[i]; }
    s = simd_sum(s);
    if (lane == 0u) { part[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = ts / 32u;
    float tot = 0.0;
    for (uint j = 0u; j < nsg; j++) { tot += part[j]; }
    float inv = rsqrt(tot/float(d) + eps);
    for (uint i = lid; i < d; i += ts) { out[i] = x[i]*inv*w[i]; }
}

// Fused Gemma sandwich residual: x[i] = (x[i] + rmsnorm(tmp,w)[i]) * oscale — replaces
// rmsnorm + add_inplace (+ optional out_scale mul) with one dispatch. Fewer dispatches
// = less per-layer overhead (gemma-4 has 4 of these + out_scale per layer).
kernel void rmsnorm_add(device float* x [[buffer(0)]], device const float* tmp [[buffer(1)]],
    device const float* w [[buffer(2)]], constant uint& d [[buffer(3)]], constant float& eps [[buffer(4)]],
    constant float& oscale [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    float s = 0.0;
    for (uint i = lid; i < d; i += ts) { float v = tmp[i]; s += v*v; }
    s = simd_sum(s);
    if (lane == 0u) { part[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = ts / 32u; float tot = 0.0;
    for (uint j = 0u; j < nsg; j++) { tot += part[j]; }
    float inv = rsqrt(tot/float(d) + eps);
    for (uint i = lid; i < d; i += ts) { x[i] = (x[i] + tmp[i]*inv*w[i]) * oscale; }
}

kernel void rope(device float* v [[buffer(0)]], constant uint& hd [[buffer(1)]],
    constant uint& pos [[buffer(2)]], constant float& base [[buffer(3)]],
    constant uint& total [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= total) { return; }
    uint hf = hd/2u; uint head = gid/hf; uint i = gid%hf; uint b = head*hd;
    float freq = 1.0/pow(base, 2.0*float(i)/float(hd));
    float ang = float(pos)*freq; float s = sin(ang), c = cos(ang);
    float x0 = v[b+i], x1 = v[b+hf+i];
    v[b+i] = x0*c - x1*s; v[b+hf+i] = x0*s + x1*c;
}

kernel void add_inplace(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) { x[gid] += y[gid]; }
}

// Gate/up activation for weights projected separately. The fused ffn_gu_* kernels do
// the two matvecs and this step in one dispatch, but each walks one hard-coded block
// layout, so the native-quant path — which keeps whatever format the file used —
// projects gate and up on their own and combines here. `act` is the same activation
// selector ffn_act takes.
kernel void ffn_gu_split(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
    constant uint& act [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) { out[gid] = ffn_act(gate[gid], act) * up[gid]; }
}

kernel void swiglu(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) { float g = gate[gid]; out[gid] = (g/(1.0+exp(-g)))*up[gid]; }
}

kernel void embed(device const half* emb [[buffer(0)]], device float* x [[buffer(1)]],
    constant uint& d [[buffer(2)]], constant uint& token [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid < d) { x[gid] = float(emb[token*d + gid]); }
}

// scale x[0..n) in place (Gemma embedding normalizer: ×sqrt(d); or layer_output_scale).
kernel void mul_scalar(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]],
    constant float& s [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) { x[gid] *= s; }
}

// copy src[0..n) -> dst (gemma-4 global layers: V = K projection).
kernel void copy_buf(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) { dst[gid] = src[gid]; }
}

// Q6_K embedding gather. token_embd ships as Q6_K in most K-quant GGUFs; we used
// to requantize it to Q8 purely because the gather kernel only spoke Q8, which cost
// 1.0625 B/weight instead of 0.8203 — 331 MB/token against 255 on a 152k vocab, and
// the tensor is read in full every token by the tied lm_head.
kernel void embed_q6k(device const uchar* emb [[buffer(0)]], device float* x [[buffer(1)]],
    constant uint& d [[buffer(2)]], constant uint& token [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= d) { return; }
    uint nsb = d / 256u;
    device const uchar* row = emb + (ulong)token*(ulong)nsb*210ul;
    float v; Q6K_AT(row, gid, v)
    x[gid] = v;
}

kernel void embed_m_q6k(device const uchar* emb [[buffer(0)]], device float* x [[buffer(1)]],
    constant uint& d [[buffer(2)]], device const uint* tokens [[buffer(3)]],
    constant uint& M [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    uint total = M*d; if (gid >= total) { return; }
    uint m = gid/d, i = gid%d;
    uint nsb = d / 256u;
    device const uchar* row = emb + (ulong)tokens[m]*(ulong)nsb*210ul;
    float v; Q6K_AT(row, i, v)
    x[gid] = v;
}

// Original GGUF Q8_0 blocks; retain the F32 scale*integer product instead of
// rounding a dequantized embedding table to F16 at load time.
kernel void embed_gguf_q8_0(device const uchar* emb [[buffer(0)]], device float* x [[buffer(1)]],
    constant uint& d [[buffer(2)]], constant uint& token [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= d) { return; }
    device const uchar* block = emb + (ulong(token)*(d/32u) + gid/32u)*34ul;
    x[gid] = float(*(device const half*)block) * float(*(device const char*)(block+2u+gid%32u));
}

kernel void embed_q8(device const char* emb [[buffer(0)]], device float* x [[buffer(1)]],
    constant uint& d [[buffer(2)]], constant uint& token [[buffer(3)]], device const float* scale [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < d) { x[gid] = float(emb[token*d + gid]) * scale[token]; }
}

// Q4 embedding lookup: dequant the token's row (nibbles + per-32-block scale/min).
kernel void embed_q4(device const uchar* emb [[buffer(0)]], device float* x [[buffer(1)]],
    constant uint& d [[buffer(2)]], constant uint& token [[buffer(3)]],
    device const half* scale [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= d) { return; }
    uint nblk = d/32u;
    device const uchar* row = emb + (ulong)token*(ulong)(d/2u);
    uint b = gid/32u;                         // which 32-block
    float sc = float((scale + (ulong)token*(ulong)nblk)[b]);
    uint within = gid % 32u; uint byte = within/2u;   // byte j = elem 2j lo | 2j+1 hi
    uchar by = row[b*16u + byte];
    uint nib = (within & 1u) ? uint(by >> 4u) : uint(by & 0xFu);
    x[gid] = sc*(float(nib) - 8.0);           // symmetric: w = scale*(nib-8)
}

// embed_q4 variant reading the token id from a device buffer (draft→verify GPU
// chaining: the verify's second row embeds the draft token without a CPU round-trip).
kernel void embed_q4_id(device const uchar* emb [[buffer(0)]], device float* x [[buffer(1)]],
    constant uint& d [[buffer(2)]], device const uint* ids [[buffer(3)]],
    device const half* scale [[buffer(4)]], constant uint& slot [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= d) { return; }
    uint token = ids[slot];
    uint nblk = d/32u;
    device const uchar* row = emb + (ulong)token*(ulong)(d/2u);
    uint b = gid/32u;
    float sc = float((scale + (ulong)token*(ulong)nblk)[b]);
    uint within = gid % 32u; uint byte = within/2u;
    uchar by = row[b*16u + byte];
    uint nib = (within & 1u) ? uint(by >> 4u) : uint(by & 0xFu);
    x[gid] = sc*(float(nib) - 8.0);
}

// Fused RoPE for q and k in ONE dispatch. gid < totq rotates q; else rotates k.
kernel void rope_qk(device float* vq [[buffer(0)]], device float* vk [[buffer(1)]],
    constant uint& hd [[buffer(2)]], constant uint& pos [[buffer(3)]], constant float& base [[buffer(4)]],
    constant uint& totq [[buffer(5)]], constant uint& totk [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    device float* v; uint g;
    if (gid < totq)            { v = vq; g = gid; }
    else if (gid < totq+totk)  { v = vk; g = gid - totq; }
    else return;
    uint hf = hd/2u; uint head = g/hf; uint i = g%hf; uint b = head*hd;
    float freq = 1.0/pow(base, 2.0*float(i)/float(hd));
    float ang = float(pos)*freq; float s = sin(ang), c = cos(ang);
    float x0 = v[b+i], x1 = v[b+hf+i];
    v[b+i] = x0*c - x1*s; v[b+hf+i] = x0*s + x1*c;
}

// Qwen3/Gemma QK-norm: per-head RMSNorm over head_dim on q and k, BEFORE RoPE.
// One simdgroup (32 threads) per head; q heads [0,nq) use qw, k heads [nq,nq+nk) use kw.
// Handles M tokens: grid.y = token index (q/k are [tok][head][hd] contiguous).
kernel void qk_rmsnorm(device float* vq [[buffer(0)]], device float* vk [[buffer(1)]],
    device const float* qw [[buffer(2)]], device const float* kw [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& nq [[buffer(5)]],
    constant uint& nk [[buffer(6)]], constant float& eps [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint head = gid.x; uint tok = gid.y;
    bool isq = head < nq;
    device float* v = isq ? vq : vk;
    device const float* w = isq ? qw : kw;
    uint h = isq ? head : (head - nq);
    uint rowdim = isq ? nq*hd : nk*hd;
    uint base = tok*rowdim + h*hd;
    float ss = 0.0;
    for (uint i = lane; i < hd; i += 32u) { float x = v[base+i]; ss += x*x; }
    ss = simd_sum(ss);
    float inv = rsqrt(ss/float(hd) + eps);
    for (uint i = lane; i < hd; i += 32u) { v[base+i] = v[base+i]*inv*w[i]; }
}

// Fused RoPE(q,k) + store(k,v) in one dispatch (replaces rope_qk + store_kv2). gid
// partitions into: q rope pairs | k rope pairs (written straight to kcache) | v
// copies (to vcache). Rotated k goes directly to the cache, with no k-scratch
// round-trip.
// neox=1: split-half pairing (i, i+rd/2) — Qwen/GPT-NeoX. neox=0: interleaved
// pairing (2i, 2i+1) — LLaMA GGUF (q/k weights are permuted for this convention).
// rd = rotary dims per head (== hd for full rope). Partial rope (qwen35: rd=64,
// hd=256): pair index j < rd/2 rotates within the first rd dims (freq uses rd);
// leftover pair indices map to plain copies of dims [rd, hd) — k must still reach
// the cache, while q is in place so untouched dims need no work.
kernel void rope_qk_store(device float* vq [[buffer(0)]], device float* vk [[buffer(1)]],
    device const float* vv [[buffer(2)]], device half* kc [[buffer(3)]], device half* vc [[buffer(4)]],
    constant uint& hd [[buffer(5)]], constant uint& pos [[buffer(6)]], constant float& base [[buffer(7)]],
    constant uint& totq [[buffer(8)]], constant uint& totk [[buffer(9)]],
    constant uint& kvdim [[buffer(10)]], constant uint& off [[buffer(11)]],
    constant uint& neox [[buffer(12)]], constant uint& rd [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
    uint hf = hd/2u; uint rf = rd/2u;
    if (gid < totq) {
        uint g = gid; uint head = g/hf; uint j = g%hf; uint b = head*hd;
        if (j >= rf) { return; }                  // partial rope: q dims >= rd stay as-is
        uint a0 = neox ? b+j : b+2u*j; uint a1 = neox ? b+rf+j : b+2u*j+1u;
        float freq = 1.0/pow(base, 2.0*float(j)/float(rd));
        float ang = float(pos)*freq; float s = sin(ang), c = cos(ang);
        float x0 = vq[a0], x1 = vq[a1];
        vq[a0] = x0*c - x1*s; vq[a1] = x0*s + x1*c;
    } else if (gid < totq+totk) {
        uint g = gid-totq; uint head = g/hf; uint j = g%hf; uint b = head*hd;
        if (j < rf) {
            uint a0 = neox ? b+j : b+2u*j; uint a1 = neox ? b+rf+j : b+2u*j+1u;
            float freq = 1.0/pow(base, 2.0*float(j)/float(rd));
            float ang = float(pos)*freq; float s = sin(ang), c = cos(ang);
            float x0 = vk[a0], x1 = vk[a1];
            kc[off+a0] = half(x0*c - x1*s); kc[off+a1] = half(x0*s + x1*c);   // rotated k straight to cache (f16)
        } else {
            uint e0 = b + rd + 2u*(j-rf);         // unrotated k dims: plain copy to cache
            kc[off+e0] = half(vk[e0]); kc[off+e0+1u] = half(vk[e0+1u]);
        }
    } else if (gid < totq+totk+kvdim) {
        uint j = gid-totq-totk;
        vc[off+j] = half(vv[j]);                  // v to cache (not rotated, f16)
    }
}

kernel void embed_m_q8(device const char* emb [[buffer(0)]], device float* x [[buffer(1)]],
    constant uint& d [[buffer(2)]], device const uint* tokens [[buffer(3)]], device const float* scale [[buffer(4)]],
    constant uint& M [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    uint total = M*d; if (gid>=total) { return; }
    uint m=gid/d, i=gid%d; uint tok=tokens[m];
    x[gid] = float(emb[tok*d+i]) * scale[tok];
}

kernel void rmsnorm_m(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& d [[buffer(3)]], constant float& eps [[buffer(4)]],
    uint m [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    device const float* xm = x + (ulong)m*(ulong)d; device float* om = out + (ulong)m*(ulong)d;
    threadgroup float part[256]; float s=0.0;
    for (uint i=lid;i<d;i+=ts) s+=xm[i]*xm[i];
    part[lid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off=ts/2u;off>0u;off>>=1u){ if(lid<off) part[lid]+=part[lid+off]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv=rsqrt(part[0]/float(d)+eps);
    for (uint i=lid;i<d;i+=ts) om[i]=xm[i]*inv*w[i];
}

// RoPE for M rows. R = elements per row (n_head*hd for q, n_kv*hd for k).
kernel void rope_m(device float* v [[buffer(0)]], constant uint& hd [[buffer(1)]],
    constant uint& base_pos [[buffer(2)]], constant float& base [[buffer(3)]],
    constant uint& R [[buffer(4)]], constant uint& M [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    uint hf = hd/2u; uint halfPerRow = R/2u; uint total = M*halfPerRow; if (gid>=total) { return; }
    uint m = gid/halfPerRow; uint rem = gid%halfPerRow; uint head = rem/hf; uint i = rem%hf;
    uint b = m*R + head*hd; uint pos = base_pos + m;
    float freq=1.0/pow(base,2.0*float(i)/float(hd)); float ang=float(pos)*freq; float s=sin(ang),c=cos(ang);
    float x0=v[b+i], x1=v[b+hf+i]; v[b+i]=x0*c-x1*s; v[b+hf+i]=x0*s+x1*c;
}

#define MROPE_SHIFT 8u

// Sectioned M-RoPE: which position stream drives each rotary pair.
//
// Plain rope drives every pair from one scalar position. M-RoPE splits the rd/2
// cos/sin pairs into four sections, each with its own position stream (t, h, w, e),
// so an image patch can carry the (row, column) identity a scalar token index cannot
// express. Section sizes are the GGUF `{arch}.rope.dimension_sections` counts,
// measured in pairs and summing to rd/2 (qwen35: [11,11,10,0] with rd = 64).
//
// mode:
//   0 OFF     `mpos` is never dereferenced and every expression below is the scalar
//             one, so the result is bit-identical to the pre-M-RoPE kernel.
//   1 MROPE   contiguous sections  [t t t t | h h | w w]      (qwen2-vl, glm4v)
//   2 IMROPE  interleaved sections [t h w t h w ...], the qwen35 mode: llama.cpp maps
//             LLM_ARCH_QWEN35 to LLAMA_ROPE_TYPE_IMROPE (llama-model.cpp:3034), and
//             [11,11,10,0] is the count of sectors {0,3..30} / {1,4..31} / {2,5..29}
//             over j in [0,32).
//   3 VISION  contiguous sections, but theta restarts at each section boundary
//             (ggml.h:1934, `indep_sects`). ViT parameterization (4 x d_head/4,
//             freq_base 10000); not degenerate with plain rope, by construction.
//
// Modes 1 and 2 keep the angle exponent at the plain pair index j (ggml.h:1927), so
// t == h == w == e reproduces plain rope bit for bit: only which position scales the
// angle changes, and they are all equal. Every model shipping today relies on that
// equivalence, and tests/rope_sections.rs asserts it.
//
// `mpos` is one buffer carrying both the sections and the positions:
//     [0..4)            section sizes s0,s1,s2,s3 (in pairs)
//     [4 + 4*m + s]     position of stream s (0=t,1=h,2=w,3=e) for token m
// Sections travel in that buffer rather than in scalar slots of their own so a call
// site which never enables M-RoPE binds nothing new (see the kernel comment).
// Returns (position for the angle, exponent index).
static inline uint2 mrope_sel(device const uint* mpos, uint mode, uint m, uint j) {
    uint s0=mpos[0], s1=mpos[1], s2=mpos[2], s3=mpos[3];
    uint sect = s0+s1+s2+s3;
    if (sect == 0u) { return uint2(mpos[4u + 4u*m], j); }   // no sections declared: t stream, plain theta
    uint sector = j % sect;
    uint sel = 0u, start = 0u;
    if (mode == 2u) {                                       // IMROPE: t h w t h w ...
        uint r = sector % 3u;
        if      (r == 1u && sector < 3u*s1) { sel = 1u; }
        else if (r == 2u && sector < 3u*s2) { sel = 2u; }
        else if (r == 0u && sector < 3u*s0) { sel = 0u; }
        else                                { sel = 3u; }
    } else {                                                // MROPE / VISION: contiguous
        if      (sector < s0)       { sel = 0u; start = 0u; }
        else if (sector < s0+s1)    { sel = 1u; start = s0; }
        else if (sector < s0+s1+s2) { sel = 2u; start = s0+s1; }
        else                        { sel = 3u; start = s0+s1+s2; }
    }
    return uint2(mpos[4u + 4u*m + sel], (mode == 3u) ? (sector - start) : j);
}

// Fused RoPE(q,k) + store(k,v) for M tokens in one dispatch (multi-token analog of
// rope_qk_store). Per token, work partitions into: q rope pairs (Aq) | k rope pairs
// (Ak, rotated straight into kcache) | v copies (kvdim). Aq = n_head*hd/2, so the q
// row stride is 2*Aq and the k/v row stride is kvdim.
//
// `base_pos + m` serves two purposes and sectioned M-RoPE moves only one: the rope
// angle takes the section's position stream, while the KV cache row stays
// `base_pos + m` so cache slots stay contiguous however the image positions are
// numbered. Conflating the two scatters an image span across the cache.
//
// The mode rides in the upper bits of `neox` (buffer 12): `(mode << MROPE_SHIFT) |
// pairing`. A fresh `constant uint&` slot would still be read on every call site not
// yet wired, and an unset argument slot holds whatever the previous dispatch in the
// same encoder left there — the qkv GEMM a few dispatches earlier binds attn_v.bias
// at index 15 (dispatch.rs:611) — so a new scalar slot would read a weight as a mode.
// `neox` is a literal 0/1 at all five call sites, so its upper bits are provably zero
// and mode 0 is what every model shipping today gets. Buffer 14 is likewise never
// dereferenced at mode 0, which makes leaving it unbound safe for the shader; but
// Metal API Validation (MTL_DEBUG_LAYER=1) asserts on a declared-and-unbound buffer
// whether or not the shader reads it, so a call site that never enables M-RoPE should
// still bind some live buffer at 14, the way graph_chunk.rs:165 binds `ssm_state` at
// the disabled snapshot slot. tests/rope_sections.rs covers both.
kernel void rope_qk_store_m(device float* vq [[buffer(0)]], device float* vk [[buffer(1)]],
    device const float* vv [[buffer(2)]], device half* kc [[buffer(3)]], device half* vc [[buffer(4)]],
    constant uint& hd [[buffer(5)]], constant uint& base_pos [[buffer(6)]], constant float& base [[buffer(7)]],
    constant uint& Aq [[buffer(8)]], constant uint& Ak [[buffer(9)]], constant uint& kvdim [[buffer(10)]],
    constant uint& M [[buffer(11)]], constant uint& neox [[buffer(12)]], constant uint& rd [[buffer(13)]],
    device const uint* mpos [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
    uint perTok = Aq + Ak + kvdim; uint total = M*perTok; if (gid >= total) { return; }
    uint m = gid/perTok; uint w = gid%perTok; uint hf = hd/2u; uint rf = rd/2u; uint pos = base_pos + m;
    uint nx = neox & 1u; uint mrope = neox >> MROPE_SHIFT;
    if (w < Aq) {
        uint head=w/hf; uint j=w%hf; uint b = m*(2u*Aq) + head*hd;
        if (j >= rf) { return; }                  // partial rope: q dims >= rd untouched
        uint a0 = nx ? b+j : b+2u*j; uint a1 = nx ? b+rf+j : b+2u*j+1u;
        uint2 pj = uint2(pos, j);
        if (mrope != 0u) { pj = mrope_sel(mpos, mrope, m, j); }   // ANGLE only
        float freq=1.0/pow(base,2.0*float(pj.y)/float(rd)); float ang=float(pj.x)*freq; float s=sin(ang),c=cos(ang);
        float x0=vq[a0], x1=vq[a1]; vq[a0]=x0*c-x1*s; vq[a1]=x0*s+x1*c;
    } else if (w < Aq+Ak) {
        uint wk=w-Aq; uint head=wk/hf; uint j=wk%hf; uint b=m*kvdim + head*hd;
        if (j >= rf) {                            // unrotated k dims: plain copy to cache
            uint e0 = head*hd + rd + 2u*(j-rf);
            ulong cb=(ulong)(base_pos+m)*(ulong)kvdim;            // cache row: ALWAYS base_pos+m
            kc[cb+e0] = half(vk[m*kvdim + e0]); kc[cb+e0+1u] = half(vk[m*kvdim + e0+1u]);
            return;
        }
        uint a0 = nx ? b+j : b+2u*j; uint a1 = nx ? b+rf+j : b+2u*j+1u;
        uint o0 = nx ? j : 2u*j; uint o1 = nx ? rf+j : 2u*j+1u;
        uint2 pj = uint2(pos, j);
        if (mrope != 0u) { pj = mrope_sel(mpos, mrope, m, j); }   // ANGLE only
        float freq=1.0/pow(base,2.0*float(pj.y)/float(rd)); float ang=float(pj.x)*freq; float s=sin(ang),c=cos(ang);
        float x0=vk[a0], x1=vk[a1]; float n0=x0*c-x1*s, n1=x0*s+x1*c;
        ulong cb=(ulong)(base_pos+m)*(ulong)kvdim + (ulong)(head*hd);   // cache row: ALWAYS base_pos+m
        kc[cb+o0]=half(n0); kc[cb+o1]=half(n1);
    } else {
        uint e=w-Aq-Ak; vc[(ulong)(base_pos+m)*(ulong)kvdim + e] = half(vv[m*kvdim + e]);
    }
}

// batched f16 embed: x[m,d] = half(w16_embd[token[m]*d + i]) (no scale). prec=0 diffusion.
kernel void embed_m_f16(device const half* emb [[buffer(0)]], device float* x [[buffer(1)]],
    constant uint& d [[buffer(2)]], device const uint* tokens [[buffer(3)]],
    constant uint& M [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    uint total = M*d; if (gid>=total) { return; }
    uint m=gid/d, i=gid%d; uint tok=tokens[m];
    x[gid] = float(emb[(ulong)tok*(ulong)d + i]);
}

// act = silu(gate) * up, elementwise over M*ffn (prefill SwiGLU second half — the
// gate/up GEMMs run as two gemv_m_q4 passes to keep register pressure sane).
kernel void silu_mul(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
    device float* act [[buffer(2)]], constant uint& total [[buffer(3)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= total) { return; }
    float v = gate[g];
    act[g] = (v/(1.0 + exp(-v))) * up[g];
}

// GPT-OSS SwiGLU-OAI (swiglu_oai): clamped gate with an (up+1) bias.
//   x = min(gate, limit);  y = clamp(up, -limit, limit)
//   act = (x * sigmoid(alpha*x)) * (y + 1)      [alpha=1.702, limit=7.0]
// gate/up already carry their expert biases (added before this pass).
kernel void swiglu_oai(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
    device float* act [[buffer(2)]], constant uint& total [[buffer(3)]],
    constant float& alpha [[buffer(4)]], constant float& limit [[buffer(5)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= total) { return; }
    float x = min(gate[g], limit);
    float y = clamp(up[g], -limit, limit);
    float glu = x / (1.0 + exp(-alpha * x));
    act[g] = glu * (y + 1.0);
}

// qwen35 gated attention: attn_q projects to per-head [q(hd) | gate(hd)] chunks
// (stride 2*hd). Split contiguous q out of qfull for the qk-norm/rope/attention path.
// M tokens: qfull rows [M, 2*qdim], q rows [M, qdim].
kernel void qgate_split(device const float* qfull [[buffer(0)]], device float* q [[buffer(1)]],
    constant uint& hd [[buffer(2)]], constant uint& qdim [[buffer(3)]], constant uint& M [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= M*qdim) { return; }
    uint m = g / qdim, r = g % qdim;
    uint h = r / hd, i = r % hd;
    q[g] = qfull[(ulong)m*(ulong)(2u*qdim) + h*2u*hd + i];
}

// attn[g] *= sigmoid(gate), gate = second hd of each 2*hd chunk of qfull.
// M tokens: attn rows [M, qdim], qfull rows [M, 2*qdim].
kernel void gate_mul_sigmoid(device float* attn [[buffer(0)]], device const float* qfull [[buffer(1)]],
    constant uint& hd [[buffer(2)]], constant uint& qdim [[buffer(3)]], constant uint& M [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
    if (g >= M*qdim) { return; }
    uint m = g / qdim, r = g % qdim;
    uint h = r / hd, i = r % hd;
    float z = qfull[(ulong)m*(ulong)(2u*qdim) + h*2u*hd + hd + i];
    attn[g] *= 1.0/(1.0 + exp(-z));
}

// Gated RMSNorm: per head, x = (rmsnorm(x)·w) · silu(z). w = ssm_norm[hd] (shared per head).
// Multi-token: gid.y = token, x/z rows are `rs` floats apart (decode: 1 tg in y).
kernel void gated_rmsnorm(device float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device const float* z [[buffer(2)]], constant uint& hd [[buffer(3)]], constant float& eps [[buffer(4)]],
    constant uint& rs [[buffer(5)]], constant uint& sig_gate [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    ulong base = (ulong)gid.y*(ulong)rs + gid.x*hd; float ss = 0.0;
    for (uint i = lane; i < hd; i += 32u) { float v = x[base+i]; ss += v*v; }
    ss = simd_sum(ss);
    float inv = rsqrt(ss/float(hd) + eps);
    for (uint i = lane; i < hd; i += 32u) {
        // Output gate: SILU for qwen35's gated DeltaNet, plain SIGMOID for qwen4exp.
        // Per the reference, that is the one numerical difference between the two
        // architectures' GDN, and the model declares it as output_gate_type.
        float zz = z[base+i];
        float sg = 1.0/(1.0 + exp(-zz));
        float sz = sig_gate ? sg : zz*sg;
        x[base+i] = x[base+i]*inv*w[i]*sz;
    }
}

kernel void set_const(device float* b [[buffer(0)]], constant float& v [[buffer(1)]], uint gid [[thread_position_in_grid]]) { b[gid] = v; }

// ===================== MLA absorption (tiny latent KV, decode-optimal) =====================
// The absorbed path folds kv_b into the query so the KV cache holds only the compressed
// latent [Lc(kv_lora) | Rc(rope)] (576 f16/token) — mathematically identical to the naive
// reconstruct path (validated token-identical). q_nope·k_nope = (q_nope·KVB_nope)·Lc = qabs·Lc.
kernel void copy_f32_half(device const float* src [[buffer(0)]], device half* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint g [[thread_position_in_grid]]) { if (g < n) dst[g] = half(src[g]); }
"#;

// ===================== Sectioned M-RoPE host-side contract =====================
// `rope_qk_store_m` takes its M-RoPE mode in the upper bits of the `neox` argument
// (buffer 12) and its sections + positions in one optional device buffer (buffer 14).
// Both ride on arguments the existing call sites already bind or never read, because
// a fresh scalar slot would be read from a stale argument-table entry on every call
// site not yet wired. See the kernel comment.

/// Bit position of the M-RoPE mode inside the `neox` argument.
pub const MROPE_SHIFT: u32 = 8;
/// Scalar rope. `mpos` is never dereferenced; output is bit-identical to the
/// kernel as it stood before sections existed.
pub const MROPE_OFF: u32 = 0;
/// Contiguous sections `[t t t t | h h | w w]` — qwen2-vl, glm4v (GGML_ROPE_TYPE_MROPE).
pub const MROPE_SECTIONS: u32 = 1;
/// Interleaved sections `[t h w t h w …]` — qwen3-vl and qwen35
/// (GGML_ROPE_TYPE_IMROPE; llama.cpp `llama-model.cpp:3034`).
pub const MROPE_INTERLEAVED: u32 = 2;
/// Contiguous sections with theta restarting per section — the ViT parameterization
/// (GGML_ROPE_TYPE_VISION; `ggml.h:1934`). Not degenerate with plain rope.
pub const MROPE_VISION: u32 = 3;

/// Pack the pairing convention and the M-RoPE mode into the kernel's `neox`
/// argument. `rope_mode(neox, MROPE_OFF) == neox as u32`, what every call site passes
/// today.
#[inline]
pub fn rope_mode(neox: bool, mode: u32) -> u32 { (mode << MROPE_SHIFT) | neox as u32 }

/// Build the buffer-14 descriptor: four section sizes (in cos/sin PAIRS, from
/// `{arch}.rope.dimension_sections`, summing to `n_rot/2`) followed by four
/// positions `(t,h,w,e)` per token. A text token sets all four equal to its
/// sequence position, which reproduces plain rope bit-for-bit.
pub fn mrope_desc(sections: [u32; 4], pos: &[[u32; 4]]) -> Vec<u32> {
    let mut v = Vec::with_capacity(4 + 4 * pos.len());
    v.extend_from_slice(&sections);
    for p in pos { v.extend_from_slice(p); }
    v
}
