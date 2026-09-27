// Bodies compile against kernels::PRELUDE (shared defines + helpers).

/// `mla_attn_abs` score buffer `sc[SC_CAP]` size; host caps absorbed-MLA context here.
pub const SC_CAP: usize = 4096;

pub const BODY: &str = r#"
// ===================== deepseek2 MLA (latent attention) kernels =====================
// rmsnorm kv_cmpr[0..kvlora] with kv_a_norm → out[0..kvlora]; NEOX-rope k_pe (kvc[kvlora..])
// → out[kvlora..kvlora+rope]. One threadgroup, 256 threads.
kernel void mla_kvnorm(device const float* kvc [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& kvlora [[buffer(3)]], constant uint& rope [[buffer(4)]],
    constant uint& pos [[buffer(5)]], constant float& base [[buffer(6)]], constant float& eps [[buffer(7)]],
    constant float& fs [[buffer(8)]], constant float& ext [[buffer(9)]], constant float& clow [[buffer(10)]], constant float& chigh [[buffer(11)]], constant uint& il [[buffer(12)]],
    uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    threadgroup float red[256];
    float acc = 0.0;
    for (uint i = lid; i < kvlora; i += ts) { float v = kvc[i]; acc += v*v; }
    red[lid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = ts/2u; k > 0u; k >>= 1u) { if (lid < k) red[lid] += red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sc = rsqrt(red[0]/float(kvlora) + eps);
    for (uint i = lid; i < kvlora; i += ts) { out[i] = kvc[i]*sc*w[i]; }
    uint hlf = rope/2u;
    for (uint j = lid; j < hlf; j += ts) {
        float fr = pow(base, -2.0*float(j)/float(rope));
        float te = float(pos)*fr; float ti = fs*te;
        float y = (float(j) - clow)/max(0.001, chigh-clow);
        float ramp = (1.0 - min(1.0, max(0.0, y)))*ext;
        float th = ti*(1.0-ramp) + te*ramp; float c = cos(th), sn = sin(th);
        float x0 = il ? kvc[kvlora + 2u*j] : kvc[kvlora + j];
        float x1 = il ? kvc[kvlora + 2u*j + 1u] : kvc[kvlora + j + hlf];
        out[kvlora + j] = x0*c - x1*sn; out[kvlora + j + hlf] = x0*sn + x1*c;
    }
}

// NEOX-rope the pe slice [nope..nope+rope) of each head in q, in place. grid=(n_head,).
kernel void mla_qrope(device float* q [[buffer(0)]], constant uint& kmla [[buffer(1)]],
    constant uint& nope [[buffer(2)]], constant uint& rope [[buffer(3)]], constant uint& pos [[buffer(4)]],
    constant float& base [[buffer(5)]], constant float& fs [[buffer(6)]], constant float& ext [[buffer(7)]], constant float& clow [[buffer(8)]], constant float& chigh [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    uint h = tg; uint hlf = rope/2u; device float* qh = q + h*kmla + nope;
    for (uint j = lid; j < hlf; j += ts) {
        float fr = pow(base, -2.0*float(j)/float(rope));
        float te = float(pos)*fr; float ti = fs*te;
        float y = (float(j) - clow)/max(0.001, chigh-clow);
        float ramp = (1.0 - min(1.0, max(0.0, y)))*ext;
        float th = ti*(1.0-ramp) + te*ramp; float c = cos(th), sn = sin(th);
        float x0 = qh[j], x1 = qh[j+hlf];
        qh[j] = x0*c - x1*sn; qh[j+hlf] = x0*sn + x1*c;
    }
}

// Assemble one KV cache row at pos. kv=[n_head*(nope+vmla)] (per head k_nope|v); kpe=[rope]
// shared. Kcache[pos][h]=[k_nope|k_pe]; Vcache[pos][h]=[v|0-pad to kmla]. grid=n_head*kmla.
kernel void mla_kvwrite(device const float* kv [[buffer(0)]], device const float* kpe [[buffer(1)]],
    device half* kc [[buffer(2)]], device half* vc [[buffer(3)]], constant uint& nhead [[buffer(4)]],
    constant uint& nope [[buffer(5)]], constant uint& vmla [[buffer(6)]], constant uint& kmla [[buffer(7)]],
    constant uint& pos [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    uint total = nhead*kmla; if (gid >= total) return;
    uint h = gid / kmla, i = gid % kmla;
    ulong off = (ulong)pos*(ulong)total + (ulong)h*(ulong)kmla + (ulong)i;
    kc[off] = half((i < nope) ? kv[h*(nope+vmla) + i] : kpe[i - nope]);
    vc[off] = half((i < vmla) ? kv[h*(nope+vmla) + nope + i] : 0.0);
}

// Gather attn[n_head*kmla] → out[n_head*vmla] (drop padded [vmla..kmla) per head).
kernel void mla_gather(device const float* attn [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& nhead [[buffer(2)]], constant uint& vmla [[buffer(3)]], constant uint& kmla [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = nhead*vmla; if (gid >= total) return;
    uint h = gid / vmla, i = gid % vmla;
    out[gid] = attn[h*kmla + i];
}

// qabs[h][k] = Σ_d q[h*kmla+d]·kvb_f16[(h*kvpair+d)*kvlora + k]  (absorb kv_b's nope rows into q);
// append q_pe → qa[h] = [qabs(kvlora) | q_pe(rope)]. grid (kvlora+rope, n_head).
kernel void mla_qabsorb(device const float* q [[buffer(0)]], device const half* kvb [[buffer(1)]],
    device float* qa [[buffer(2)]], constant uint& kmla [[buffer(3)]], constant uint& nope [[buffer(4)]],
    constant uint& rope [[buffer(5)]], constant uint& kvlora [[buffer(6)]], constant uint& kvpair [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint h = gid.y, idx = gid.x; uint hdk = kvlora + rope; if (idx >= hdk) return;
    if (idx < kvlora) {
        float acc = 0.0;
        for (uint dd = 0u; dd < nope; dd++) acc += q[h*kmla + dd] * float(kvb[((ulong)(h*kvpair + dd))*kvlora + idx]);
        qa[h*hdk + idx] = acc;
    } else {
        qa[h*hdk + idx] = q[h*kmla + nope + (idx - kvlora)];
    }
}

// MQA over the shared latent: score[t]=qa[h]·lat[t] (hdk=kvlora+rope), softmax, out[h][k]=Σ sm[t]·lat[t][k]
// (k<kvlora; V = latent's first kvlora dims = Lc). One threadgroup per head.
kernel void mla_attn_abs(device const float* qa [[buffer(0)]], device const half* lat [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& hdk [[buffer(3)]], constant uint& hdv [[buffer(4)]],
    constant uint& lstride [[buffer(5)]], constant uint& seq [[buffer(6)]], constant float& scale [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint ts [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sc[4096]; threadgroup float part[32];   // sc size == host SC_CAP; seq>SC_CAP uses the naive path
    device const float* qh = qa + head*hdk; uint nsg = ts/32u;
    for (uint t = sg; t < seq; t += nsg) {
        device const half* kt = lat + (ulong)t*(ulong)lstride;
        float sv = 0.0; for (uint i = lane; i < hdk; i += 32u) sv += qh[i]*float(kt[i]);
        sv = simd_sum(sv); if (lane==0u) sc[t] = sv*scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lmax = -1e30; for (uint t = lid; t < seq; t += ts) lmax = max(lmax, sc[t]);
    lmax = simd_max(lmax); if (lane==0u) part[sg]=lmax; threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=-1e30; for (uint j=0u;j<nsg;j++) mx=max(mx,part[j]);
    float lsum=0.0; for (uint t=lid;t<seq;t+=ts){ float e=exp(sc[t]-mx); sc[t]=e; lsum+=e; }
    lsum=simd_sum(lsum); threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane==0u) part[sg]=lsum; threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0.0; for (uint j=0u;j<nsg;j++) sum+=part[j];
    for (uint k = lid; k < hdv; k += ts) {
        float acc=0.0; for (uint t=0u;t<seq;t++) acc += sc[t]*float(lat[(ulong)t*(ulong)lstride + k]);
        out[head*hdv + k] = acc/sum;
    }
}

// ctx[h][i] = Σ_k clat[h][k]·kvb_f16[(h*kvpair+nope+i)*kvlora + k]  (decompress latent → v). grid (vmla, n_head).
kernel void mla_ctx(device const float* clat [[buffer(0)]], device const half* kvb [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& kvlora [[buffer(3)]], constant uint& vmla [[buffer(4)]],
    constant uint& nope [[buffer(5)]], constant uint& kvpair [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
    uint h = gid.y, i = gid.x; if (i >= vmla) return;
    float acc = 0.0;
    for (uint k = 0u; k < kvlora; k++) acc += clat[h*kvlora + k] * float(kvb[((ulong)(h*kvpair + nope + i))*kvlora + k]);
    out[h*vmla + i] = acc;
}

// Interleaved-pair RoPE for q_pe (GLM-5.2): input pairs (2j,2j+1)→out(j,hlf+j). Staged in
// threadgroup memory to avoid the in-place read/write hazard. grid=(n_head,).
kernel void mla_qrope_il(device float* q [[buffer(0)]], constant uint& kmla [[buffer(1)]],
    constant uint& nope [[buffer(2)]], constant uint& rope [[buffer(3)]], constant uint& pos [[buffer(4)]],
    constant float& base [[buffer(5)]], constant float& fs [[buffer(6)]], constant float& ext [[buffer(7)]],
    constant float& clow [[buffer(8)]], constant float& chigh [[buffer(9)]], constant float& mrope [[buffer(10)]],
    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint ts [[threads_per_threadgroup]]) {
    threadgroup float buf[256];
    uint h = tg; uint hlf = rope/2u; device float* qh = q + h*kmla + nope;
    for (uint i = lid; i < rope; i += ts) buf[i] = qh[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = lid; j < hlf; j += ts) {
        float fr = pow(base, -2.0*float(j)/float(rope));
        float te = float(pos)*fr; float ti = fs*te;
        float y = (float(j) - clow)/max(0.001, chigh-clow);
        float ramp = (1.0 - min(1.0, max(0.0, y)))*ext;
        float th = ti*(1.0-ramp) + te*ramp; float c = cos(th)*mrope, sn = sin(th)*mrope;
        float a = buf[2u*j], b = buf[2u*j + 1u];
        qh[j] = a*c - b*sn; qh[hlf + j] = b*c + a*sn;
    }
}
"#;
