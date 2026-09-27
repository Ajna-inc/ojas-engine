//! GENERATED (tools: this file's header) — Metal IQ codebook GEMV kernels
//! for the streamed/resident MoE expert path (pack + UD-IQ models on GPU).
//! Decode mirrors cpu_math dot_iq2xxs/dot_iq3xxs/dot_iq2s/dot_iq4xs
//! (byte-exact vs gguf-py).

pub const MOE_IQ_KERNEL_NAMES: &[&str] = &[
    "moe_gu_iq2xxs", "moe_down_iq3xxs", "moe_down_iq2xxs", "moe_gu_iq2s", "moe_down_iq4xs",
    "moe_down_iq4nl", "moe_down_iq4nl_parallel", "moe_down_iq4nl_parallel_w8", "moe_down_iq4nl_finish", "moe_gu_iq3s", "moe_gu_iq3s_table", "moe_gu_iq4xs",
    "moe_gu_iq3s_m", "moe_gu_iq4xs_m", "moe_down_iq4nl_m", "moe_gu_iq4xs_direct", "moe_gu_iq4xs_m_direct",
    "moe_down_iq4nl_direct", "moe_down_iq4nl_m_direct", "moe_gu_iq3s_direct", "moe_gu_iq3s_m_direct"];

/// Kernel bodies only. The IQ codebooks these index (iq2xxs/iq3xxs/iq2s/iq3s
/// grids and ksigns) used to be pasted in here verbatim, a second copy of tables
/// `iq_grids` already generates; the family is now assembled with those instead
/// (see kernels/mod.rs), which is also what makes `iq3s_grid` reachable from here.
pub const MOE_IQ_KERNELS: &str = r#"
constant int kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// one 32-weight group of an IQ2_XXS block: dot against x, pre-scale (d applied by caller)
inline float iq2xxs_group(device const uchar* qs8, uint g, device const float* xg) {
    uint a0 = ((uint)qs8[8*g]) | ((uint)qs8[8*g+1]<<8) | ((uint)qs8[8*g+2]<<16) | ((uint)qs8[8*g+3]<<24);
    uint a1 = ((uint)qs8[8*g+4]) | ((uint)qs8[8*g+5]<<8) | ((uint)qs8[8*g+6]<<16) | ((uint)qs8[8*g+7]<<24);
    float db = (0.5f + (float)(a1 >> 28)) * 0.25f;
    float gsum = 0.0f;
    for (uint l = 0u; l < 4u; l++) {
        uint64_t grid = iq2xxs_grid[(a0 >> (8u*l)) & 255u];
        uchar signs = ksigns_iq2xs[(a1 >> (7u*l)) & 127u];
        for (uint j = 0u; j < 8u; j++) {
            float m = (float)((grid >> (8u*j)) & 255u);
            gsum += ((signs >> j) & 1u ? -m : m) * xg[l*8u + j];
        }
    }
    return db * gsum;
}

inline float iq3xxs_group(device const uchar* qs, device const uchar* ss, uint g, device const float* xg) {
    uint aux = ((uint)ss[4*g]) | ((uint)ss[4*g+1]<<8) | ((uint)ss[4*g+2]<<16) | ((uint)ss[4*g+3]<<24);
    float db = (0.5f + (float)(aux >> 28)) * 0.5f;
    float gsum = 0.0f;
    for (uint l = 0u; l < 4u; l++) {
        uchar signs = ksigns_iq2xs[(aux >> (7u*l)) & 127u];
        uint g1 = iq3xxs_grid[qs[8*g + 2*l]];
        uint g2v = iq3xxs_grid[qs[8*g + 2*l + 1]];
        for (uint j = 0u; j < 4u; j++) {
            float m1 = (float)((g1 >> (8u*j)) & 255u);
            float m2 = (float)((g2v >> (8u*j)) & 255u);
            gsum += ((signs >> j) & 1u ? -m1 : m1) * xg[l*8u + j];
            gsum += ((signs >> (j+4u)) & 1u ? -m2 : m2) * xg[l*8u + j + 4u];
        }
    }
    return db * gsum;
}

// one 32-weight group of an IQ2_S block (82 B/256: d f16, qs[32] grid-low,
// signs[32] raw, qh[8] 2 high grid bits per entry, scales[8] two 4-bit halves):
// dot against x, pre-scale (d applied by caller). blk points at the block start.
inline float iq2s_group(device const uchar* blk, uint g, device const float* xg) {
    device const uchar* qs = blk + 2u;
    device const uchar* sgn = blk + 34u;
    device const uchar* qh = blk + 66u;
    device const uchar* sc = blk + 74u;
    float db0 = (0.5f + (float)(sc[g] & 15u)) * 0.25f;
    float db1 = (0.5f + (float)(sc[g] >> 4)) * 0.25f;
    float sum = 0.0f;
    for (uint l = 0u; l < 4u; l++) {
        uint gi = (uint)qs[4u*g + l] | (((uint)qh[g] << (8u - 2u*l)) & 0x300u);
        uint64_t grid = iq2s_grid[gi];
        uchar sb = sgn[4u*g + l];
        float gsum = 0.0f;
        for (uint j = 0u; j < 8u; j++) {
            float m = (float)((grid >> (8u*j)) & 255u);
            gsum += ((sb >> j) & 1u ? -m : m) * xg[l*8u + j];
        }
        sum += (l < 2u ? db0 : db1) * gsum;
    }
    return sum;
}

// one 32-weight group of an IQ4_XS block (136 B/256: d f16, scales_h u16,
// scales_l[4], qs[128]; 6-bit scale - 32, non-linear LUT, nibbles -> lanes j
// and j+16). pre-scale (d applied by caller). blk points at the block start.
inline float iq4xs_group(device const uchar* blk, uint g, device const float* xg) {
    uint sh = (uint)blk[2] | ((uint)blk[3] << 8);
    device const uchar* sl = blk + 4u;
    device const uchar* qs = blk + 8u;
    int ls = (int)((sl[g/2u] >> (4u*(g & 1u))) & 15u) | (int)(((sh >> (2u*g)) & 3u) << 4);
    float gsum = 0.0f;
    for (uint j = 0u; j < 16u; j++) {
        uint q = (uint)qs[16u*g + j];
        gsum += (float)kvalues_iq4nl[q & 15u] * xg[j];
        gsum += (float)kvalues_iq4nl[q >> 4] * xg[j + 16u];
    }
    return (float)(ls - 32) * gsum;
}

// fused gate+up GEMV over raw IQ2_XXS rows (signature = moe_gu_q4k):
// act[j][row] = silu(gate_row . x) * (up_row . x); grid (ceil(N/rows_per_tg), KSEL)
kernel void moe_gu_iq2xxs(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], device const uint* idx [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N) { return; }
    uint e = idx[j];
    uint rb = K/256u*66u;
    device const uchar* gr = wg + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
    device const uchar* ur = wu + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
    uint nb = K/256u;
    float gs = 0.0f, us = 0.0f;
    // lane u handles unit = block*8+group, strided by 32
    for (uint u = lane; u < nb*8u; u += 32u) {
        uint b = u / 8u, g = u % 8u;
        device const float* xg = x + b*256u + g*32u;
        device const uchar* gb = gr + b*66u;
        device const uchar* ub = ur + b*66u;
        float gd = (float)(*(device const half*)gb);
        float ud = (float)(*(device const half*)ub);
        gs += gd * iq2xxs_group(gb + 2u, g, xg);
        us += ud * iq2xxs_group(ub + 2u, g, xg);
    }
    gs = simd_sum(gs); us = simd_sum(us);
    if (lane == 0u) {
        float sg = gs / (1.0f + exp(-gs));
        act[j*N + out_row] = sg * us;
    }
}

// down GEMV over raw IQ3_XXS rows, accumulating all KSEL experts + shared
// (signature = moe_down_q80): x[row] += sum_j wgt[j]*(down_j_row . act_j) + shx*sigmoid(shg)
kernel void moe_down_iq3xxs(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u) + sgid; if (out_row >= N) { return; }
    uint rb = K/256u*98u;
    uint nb = K/256u;
    float total = 0.0f;
    for (uint j = 0u; j < KSEL; j++) {
        uint e = idx[j];
        device const uchar* dr = wd + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
        device const float* aj = act + j*K;
        float s = 0.0f;
        for (uint u = lane; u < nb*8u; u += 32u) {
            uint b = u / 8u, g = u % 8u;
            device const uchar* blk = dr + b*98u;
            float d = (float)(*(device const half*)blk);
            s += d * iq3xxs_group(blk + 2u, blk + 66u, g, aj + b*256u + g*32u);
        }
        s = simd_sum(s);
        total += wgt[j] * s;
    }
    if (lane == 0u) {
        float sh = shx[out_row] / (1.0f + exp(-shg[0]));
        x[out_row] += total + sh;
    }
}

// down variant for IQ2_XXS-typed down tensors
kernel void moe_down_iq2xxs(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u) + sgid; if (out_row >= N) { return; }
    uint rb = K/256u*66u;
    uint nb = K/256u;
    float total = 0.0f;
    for (uint j = 0u; j < KSEL; j++) {
        uint e = idx[j];
        device const uchar* dr = wd + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
        device const float* aj = act + j*K;
        float s = 0.0f;
        for (uint u = lane; u < nb*8u; u += 32u) {
            uint b = u / 8u, g = u % 8u;
            device const uchar* blk = dr + b*66u;
            float d = (float)(*(device const half*)blk);
            s += d * iq2xxs_group(blk + 2u, g, aj + b*256u + g*32u);
        }
        s = simd_sum(s);
        total += wgt[j] * s;
    }
    if (lane == 0u) {
        float sh = shx[out_row] / (1.0f + exp(-shg[0]));
        x[out_row] += total + sh;
    }
}

// fused gate+up GEMV over raw IQ2_S rows (signature = moe_gu_q4k):
// act[j][row] = silu(gate_row . x) * (up_row . x); grid (ceil(N/rows_per_tg), KSEL)
kernel void moe_gu_iq2s(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], device const uint* idx [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N) { return; }
    uint e = idx[j];
    uint rb = K/256u*82u;
    device const uchar* gr = wg + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
    device const uchar* ur = wu + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
    uint nb = K/256u;
    float gs = 0.0f, us = 0.0f;
    // lane u handles unit = block*8+group, strided by 32
    for (uint u = lane; u < nb*8u; u += 32u) {
        uint b = u / 8u, g = u % 8u;
        device const float* xg = x + b*256u + g*32u;
        device const uchar* gb = gr + b*82u;
        device const uchar* ub = ur + b*82u;
        float gd = (float)(*(device const half*)gb);
        float ud = (float)(*(device const half*)ub);
        gs += gd * iq2s_group(gb, g, xg);
        us += ud * iq2s_group(ub, g, xg);
    }
    gs = simd_sum(gs); us = simd_sum(us);
    if (lane == 0u) {
        float sg = gs / (1.0f + exp(-gs));
        act[j*N + out_row] = sg * us;
    }
}

// down variant for IQ4_XS-typed down tensors (signature = moe_down_q80):
// x[row] += sum_j wgt[j]*(down_j_row . act_j) + shx*sigmoid(shg)
kernel void moe_down_iq4xs(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u) + sgid; if (out_row >= N) { return; }
    uint rb = K/256u*136u;
    uint nb = K/256u;
    float total = 0.0f;
    for (uint j = 0u; j < KSEL; j++) {
        uint e = idx[j];
        device const uchar* dr = wd + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
        device const float* aj = act + j*K;
        float s = 0.0f;
        for (uint u = lane; u < nb*8u; u += 32u) {
            uint b = u / 8u, g = u % 8u;
            device const uchar* blk = dr + b*136u;
            float d = (float)(*(device const half*)blk);
            s += d * iq4xs_group(blk, g, aj + b*256u + g*32u);
        }
        s = simd_sum(s);
        total += wgt[j] * s;
    }
    if (lane == 0u) {
        float sh = shx[out_row] / (1.0f + exp(-shg[0]));
        x[out_row] += total + sh;
    }
}

// ---- IQ4_NL --------------------------------------------------------------
// 32 weights per 18-byte block { half d; uchar qs[16] }. No superblock and no
// per-group sub-scale, so the nibble indexes the codebook directly — the cheapest IQ
// decode here. `iq4xs_group` above is the same codebook wrapped in a 256-weight
// superblock with 6-bit group scales.
//
// The 16 low nibbles are outputs 0..16 and the 16 high nibbles outputs 16..32: split,
// not interleaved. Mirrors cpu_math::dot_iq4nl, which is pinned against
// gguf::dequant_to_f16 (checked value-for-value against gguf-py).
inline float iq4nl_block(device const uchar* blk, device const float* xb) {
    device const uchar* qs = blk + 2u;
    float s = 0.0f;
    for (uint j = 0u; j < 16u; j++) {
        uint q = (uint)qs[j];
        s += (float)kvalues_iq4nl[q & 15u] * xb[j];
        s += (float)kvalues_iq4nl[q >> 4] * xb[j + 16u];
    }
    return s;
}

kernel void moe_down_iq4nl(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint ts [[threads_per_threadgroup]]) {
    uint out_row = tgid*(ts/32u) + sgid; if (out_row >= N) { return; }
    uint nb = K/32u;      // 32-weight blocks per row
    uint rb = nb*18u;     // bytes per row
    float total = 0.0f;
    for (uint j = 0u; j < KSEL; j++) {
        uint e = idx[j];
        device const uchar* dr = wd + (ulong)e*(ulong)N*(ulong)rb + (ulong)out_row*(ulong)rb;
        device const float* aj = act + j*K;
        float s = 0.0f;
        for (uint b = lane; b < nb; b += 32u) {
            device const uchar* blk = dr + b*18u;
            // Byte-assembled, not a `half*` cast: an 18-byte block stride only
            // guarantees 2-byte alignment, and this costs nothing.
            ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);
            float d = (float)as_type<half>(dbits);
            s += d * iq4nl_block(blk, aj + b*32u);
        }
        s = simd_sum(s);
        total += wgt[j] * s;
    }
    if (lane == 0u) {
        float sh = shx[out_row] / (1.0f + exp(-shg[0]));
        x[out_row] += total + sh;
    }
}

// Give every selected expert its own grid row. Four lanes cooperate on each
// 32-weight block, exposing 10x more independent work than the serial expert
// loop above. The partial buffer is [selected expert][output row]; a small
// second dispatch preserves the original weighted sum and shared-expert term.
kernel void moe_down_iq4nl_parallel(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* partial [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N || tg.y >= KSEL) { return; }
    uint j = tg.y;
    uint nb = K/32u, rb = nb*18u;
    uint e = idx[j];
    device const uchar* dr = wd + (ulong)e*(ulong)N*(ulong)rb + (ulong)out_row*(ulong)rb;
    device const float* aj = act + j*K;
    float s = 0.0f;
    for (uint b = lane/4u; b < nb; b += 8u) {
        device const uchar* blk = dr + b*18u;
        ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);
        float d = (float)as_type<half>(dbits);
        float dot = 0.0f;
        for (uint z = 0u; z < 8u; ++z) {
            uint i = (lane & 3u)*8u + z;
            uint packed = (uint)blk[2u + (i & 15u)];
            uint q = i < 16u ? (packed & 15u) : (packed >> 4u);
            dot += (float)kvalues_iq4nl[q] * aj[b*32u + i];
        }
        s += d * dot;
    }
    s = simd_sum(s);
    if (lane == 0u) { partial[j*N + out_row] = wgt[j] * s; }
}

// Eight lanes per block halves each lane's dependent codebook/FMA chain. The
// wider subgroup still sums the same 32 values inside one SIMDgroup and keeps
// the same expert-parallel partial/finish contract as the four-lane kernel.
kernel void moe_down_iq4nl_parallel_w8(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* partial [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N || tg.y >= KSEL) { return; }
    uint j = tg.y;
    uint nb = K/32u, rb = nb*18u;
    uint e = idx[j];
    device const uchar* dr = wd + (ulong)e*(ulong)N*(ulong)rb + (ulong)out_row*(ulong)rb;
    device const float* aj = act + j*K;
    float s = 0.0f;
    for (uint b = lane/8u; b < nb; b += 4u) {
        device const uchar* blk = dr + b*18u;
        ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);
        float d = (float)as_type<half>(dbits);
        float dot = 0.0f;
        for (uint z = 0u; z < 4u; ++z) {
            uint i = (lane & 7u)*4u + z;
            uint packed = (uint)blk[2u + (i & 15u)];
            uint q = i < 16u ? (packed & 15u) : (packed >> 4u);
            dot += (float)kvalues_iq4nl[q] * aj[b*32u + i];
        }
        s += d * dot;
    }
    s = simd_sum(s);
    if (lane == 0u) { partial[j*N + out_row] = wgt[j] * s; }
}

kernel void moe_down_iq4nl_finish(device const float* partial [[buffer(0)]], device float* x [[buffer(1)]],
    device const float* shx [[buffer(2)]], device const float* shg [[buffer(3)]],
    constant uint& N [[buffer(4)]], constant uint& KSEL [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= N) { return; }
    float total = 0.0f;
    for (uint j = 0u; j < KSEL; ++j) { total += partial[j*N + i]; }
    x[i] += total + shx[i] / (1.0f + exp(-shg[0]));
}

// ---- IQ3_S ---------------------------------------------------------------
// 256 weights per 110-byte block { half d; u8 qs[64]; u8 qh[8]; u8 signs[32];
// u8 scales[4] }. Three things differ from the other IQ types here, each a silent wrong
// answer if missed: the grid has 512 entries with the 9th index bit coming from `qh`;
// signs are raw bits via kmask_iq2xs, not a ksigns lookup; and the 4-bit scale is
// applied as (1 + 2s). Per 32-weight group `g` the offsets collapse to qs+8g, signs+4g,
// qh[g], scales[g/2]>>4(g&1).
// Mirrors cpu_math::dot_iq3s, which is pinned against gguf::dequant_to_f16.
inline float iq3s_group(device const uchar* blk, uint g, device const float* xg) {
    device const uchar* qs = blk + 2u;
    device const uchar* qh = blk + 66u;
    device const uchar* sg = blk + 74u;
    device const uchar* sc = blk + 106u;
    uint h = (uint)qh[g];
    uint qo = 8u*g, so = 4u*g;
    float gsum = 0.0f;
    for (uint l = 0u; l < 4u; l++) {
        uint i1 = (uint)qs[qo + 2u*l]      | ((h << (8u - 2u*l)) & 256u);
        uint i2 = (uint)qs[qo + 2u*l + 1u] | ((h << (7u - 2u*l)) & 256u);
        uint g1 = iq3s_grid[i1], g2 = iq3s_grid[i2];
        uint sgn = (uint)sg[so + l];
        for (uint j = 0u; j < 4u; j++) {
            float v1 = (float)((g1 >> (8u*j)) & 255u);
            float v2 = (float)((g2 >> (8u*j)) & 255u);
            gsum += v1 * ((sgn & kmask_iq2xs[j])      ? -xg[8u*l + j]      : xg[8u*l + j]);
            gsum += v2 * ((sgn & kmask_iq2xs[j + 4u]) ? -xg[8u*l + j + 4u] : xg[8u*l + j + 4u]);
        }
    }
    return (1.0f + 2.0f * (float)((sc[g >> 1u] >> (4u*(g & 1u))) & 15u)) * gsum;
}

// Scalar decode evaluates four neighbouring gate and up rows against the same 32
// activations. The reuse has to be explicit: with a device pointer the compiler is free
// to re-emit the loads for every row, while this thread-address-space helper keeps the
// cached values in registers. The batched kernel keeps the device-pointer helper above,
// its occupancy and row geometry being a separate tuning problem.
inline float iq3s_group_cached(device const uchar* blk, uint g, thread const float* xg) {
    device const uchar* qs = blk + 2u;
    device const uchar* qh = blk + 66u;
    device const uchar* sg = blk + 74u;
    device const uchar* sc = blk + 106u;
    uint h = (uint)qh[g];
    uint qo = 8u*g, so = 4u*g;
    float gsum = 0.0f;
    for (uint l = 0u; l < 4u; l++) {
        uint i1 = (uint)qs[qo + 2u*l]      | ((h << (8u - 2u*l)) & 256u);
        uint i2 = (uint)qs[qo + 2u*l + 1u] | ((h << (7u - 2u*l)) & 256u);
        uint g1 = iq3s_grid[i1], g2 = iq3s_grid[i2];
        uint sgn = (uint)sg[so + l];
        for (uint j = 0u; j < 4u; j++) {
            float v1 = (float)((g1 >> (8u*j)) & 255u);
            float v2 = (float)((g2 >> (8u*j)) & 255u);
            gsum += v1 * ((sgn & kmask_iq2xs[j])      ? -xg[8u*l + j]      : xg[8u*l + j]);
            gsum += v2 * ((sgn & kmask_iq2xs[j + 4u]) ? -xg[8u*l + j + 4u] : xg[8u*l + j + 4u]);
        }
    }
    return (1.0f + 2.0f * (float)((sc[g >> 1u] >> (4u*(g & 1u))) & 15u)) * gsum;
}

inline float iq3s_group_cached_table(device const uchar* blk, uint g,
    thread const float* xg, threadgroup const uint* grid) {
    device const uchar* qs = blk + 2u;
    device const uchar* qh = blk + 66u;
    device const uchar* sg = blk + 74u;
    device const uchar* sc = blk + 106u;
    uint h = (uint)qh[g];
    uint qo = 8u*g, so = 4u*g;
    float gsum = 0.0f;
    threadgroup const uchar* bytes = (threadgroup const uchar*)grid;
    for (uint l = 0u; l < 4u; l++) {
        uint i1 = (uint)qs[qo + 2u*l]      | ((h << (8u - 2u*l)) & 256u);
        uint i2 = (uint)qs[qo + 2u*l + 1u] | ((h << (7u - 2u*l)) & 256u);
        threadgroup const uchar* g1 = bytes + 4u*i1;
        threadgroup const uchar* g2 = bytes + 4u*i2;
        uint sgn = (uint)sg[so + l];
        for (uint j = 0u; j < 4u; j++) {
            float v1 = (float)g1[j], v2 = (float)g2[j];
            gsum += v1 * ((sgn & kmask_iq2xs[j])      ? -xg[8u*l + j]      : xg[8u*l + j]);
            gsum += v2 * ((sgn & kmask_iq2xs[j + 4u]) ? -xg[8u*l + j + 4u] : xg[8u*l + j + 4u]);
        }
    }
    return (1.0f + 2.0f * (float)((sc[g >> 1u] >> (4u*(g & 1u))) & 15u)) * gsum;
}

kernel void moe_gu_iq3s(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], device const uint* idx [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    // Match the reference's IQ3_S geometry: each simdgroup evaluates four
    // adjacent rows. The activation sub-block is then loaded once and reused
    // across all four gate/up row pairs instead of once per output row.
    uint row0 = tg.x*8u + sgid*4u; if (row0 >= N) { return; }
    uint e = idx[j];
    uint rb = K/256u*110u;
    uint nb = K/256u;
    float gs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float us[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint u = lane; u < nb*8u; u += 32u) {
        uint b = u / 8u, g = u % 8u;
        float xv[32];
        for (uint z = 0u; z < 32u; ++z) { xv[z] = x[b*256u + g*32u + z]; }
        for (uint r = 0u; r < 4u && row0 + r < N; ++r) {
            uint out_row = row0 + r;
            device const uchar* gr = wg + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
            device const uchar* ur = wu + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
            device const uchar* gb = gr + b*110u;
            device const uchar* ub = ur + b*110u;
            float gd = (float)(*(device const half*)gb);
            float ud = (float)(*(device const half*)ub);
            gs[r] += gd * iq3s_group_cached(gb, g, xv);
            us[r] += ud * iq3s_group_cached(ub, g, xv);
        }
    }
    for (uint r = 0u; r < 4u; ++r) {
        gs[r] = simd_sum(gs[r]);
        us[r] = simd_sum(us[r]);
    }
    if (lane == 0u) {
        for (uint r = 0u; r < 4u && row0 + r < N; ++r) {
            float sg2 = gs[r] / (1.0f + exp(-gs[r]));
            act[j*N + row0 + r] = sg2 * us[r];
        }
    }
}

kernel void moe_gu_iq3s_table(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], device const uint* idx [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    threadgroup uint grid[512];
    uint lid = sgid*32u + lane;
    for (uint z = lid; z < 512u; z += ts.x) { grid[z] = iq3s_grid[z]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint j = tg.y;
    uint row0 = tg.x*8u + sgid*4u; if (row0 >= N) { return; }
    uint e = idx[j];
    uint rb = K/256u*110u;
    uint nb = K/256u;
    float gs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float us[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint u = lane; u < nb*8u; u += 32u) {
        uint b = u / 8u, g = u % 8u;
        float xv[32];
        for (uint z = 0u; z < 32u; ++z) { xv[z] = x[b*256u + g*32u + z]; }
        for (uint r = 0u; r < 4u && row0 + r < N; ++r) {
            uint out_row = row0 + r;
            device const uchar* gr = wg + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
            device const uchar* ur = wu + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
            device const uchar* gb = gr + b*110u;
            device const uchar* ub = ur + b*110u;
            float gd = (float)(*(device const half*)gb);
            float ud = (float)(*(device const half*)ub);
            gs[r] += gd * iq3s_group_cached_table(gb, g, xv, grid);
            us[r] += ud * iq3s_group_cached_table(ub, g, xv, grid);
        }
    }
    for (uint r = 0u; r < 4u; ++r) {
        gs[r] = simd_sum(gs[r]); us[r] = simd_sum(us[r]);
    }
    if (lane == 0u) {
        for (uint r = 0u; r < 4u && row0 + r < N; ++r) {
            float sg2 = gs[r] / (1.0f + exp(-gs[r]));
            act[j*N + row0 + r] = sg2 * us[r];
        }
    }
}

// ---- IQ4_XS as gate/up ---------------------------------------------------
// Same superblock `iq4xs_group` the down kernel uses; only the surrounding
// gate+up SwiGLU shell differs.
kernel void moe_gu_iq4xs(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], device const uint* idx [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N) { return; }
    uint e = idx[j];
    uint rb = K/256u*136u;
    device const uchar* gr = wg + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
    device const uchar* ur = wu + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
    uint nb = K/256u;
    float gs = 0.0f, us = 0.0f;
    for (uint u = lane; u < nb*8u; u += 32u) {
        uint b = u / 8u, g = u % 8u;
        device const float* xg = x + b*256u + g*32u;
        device const uchar* gb = gr + b*136u;
        device const uchar* ub = ur + b*136u;
        float gd = (float)(*(device const half*)gb);
        float ud = (float)(*(device const half*)ub);
        gs += gd * iq4xs_group(gb, g, xg);
        us += ud * iq4xs_group(ub, g, xg);
    }
    gs = simd_sum(gs); us = simd_sum(us);
    if (lane == 0u) {
        float sg2 = gs / (1.0f + exp(-gs));
        act[j*N + out_row] = sg2 * us;
    }
}

// ---- batched (M>1) variants -----------------------------------------------
// `tg.y` is a flat (token, expert-slot) index: token = j/KSEL picks the activation row,
// idx[j] the expert. Same decode as the M=1 kernels above, only the addressing changes.
// Needed for prefill and for MTP verify, both of which run the expert FFN over several
// tokens in one pass.

kernel void moe_gu_iq3s_m(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], device const uint* idx [[buffer(8)]],
    constant uint& KSEL [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N) { return; }
    uint e = idx[j];
    uint rb = K/256u*110u;
    device const uchar* gr = wg + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
    device const uchar* ur = wu + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
    device const float* xt = x + (ulong)(j/KSEL)*(ulong)K;
    uint nb = K/256u;
    float gs = 0.0f, us = 0.0f;
    for (uint u = lane; u < nb*8u; u += 32u) {
        uint b = u / 8u, g = u % 8u;
        device const float* xg = xt + b*256u + g*32u;
        device const uchar* gb = gr + b*110u;
        device const uchar* ub = ur + b*110u;
        float gd = (float)(*(device const half*)gb);
        float ud = (float)(*(device const half*)ub);
        gs += gd * iq3s_group(gb, g, xg);
        us += ud * iq3s_group(ub, g, xg);
    }
    gs = simd_sum(gs); us = simd_sum(us);
    if (lane == 0u) {
        float sg2 = gs / (1.0f + exp(-gs));
        act[(ulong)j*(ulong)N + out_row] = sg2 * us;
    }
}

kernel void moe_gu_iq4xs_m(device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* act [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], device const uint* idx [[buffer(8)]],
    constant uint& KSEL [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint j = tg.y;
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N) { return; }
    uint e = idx[j];
    uint rb = K/256u*136u;
    device const uchar* gr = wg + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
    device const uchar* ur = wu + (ulong)e*(ulong)N*rb + (ulong)out_row*rb;
    device const float* xt = x + (ulong)(j/KSEL)*(ulong)K;
    uint nb = K/256u;
    float gs = 0.0f, us = 0.0f;
    for (uint u = lane; u < nb*8u; u += 32u) {
        uint b = u / 8u, g = u % 8u;
        device const float* xg = xt + b*256u + g*32u;
        device const uchar* gb = gr + b*136u;
        device const uchar* ub = ur + b*136u;
        float gd = (float)(*(device const half*)gb);
        float ud = (float)(*(device const half*)ub);
        gs += gd * iq4xs_group(gb, g, xg);
        us += ud * iq4xs_group(ub, g, xg);
    }
    gs = simd_sum(gs); us = simd_sum(us);
    if (lane == 0u) {
        float sg2 = gs / (1.0f + exp(-gs));
        act[(ulong)j*(ulong)N + out_row] = sg2 * us;
    }
}

kernel void moe_down_iq4nl_m(device const float* act [[buffer(0)]], device const uchar* wd [[buffer(1)]],
    device float* x [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    device const uint* idx [[buffer(6)]], device const float* wgt [[buffer(7)]], constant uint& KSEL [[buffer(8)]],
    device const float* shx [[buffer(9)]], device const float* shg [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 ts [[threads_per_threadgroup]]) {
    uint row = tg.y;
    uint out_row = tg.x*(ts.x/32u) + sgid; if (out_row >= N) { return; }
    uint nb = K/32u, rb = nb*18u;
    float total = 0.0f;
    for (uint j = 0u; j < KSEL; j++) {
        uint e = idx[row*KSEL + j];
        device const uchar* dr = wd + (ulong)e*(ulong)N*(ulong)rb + (ulong)out_row*(ulong)rb;
        device const float* aj = act + (ulong)(row*KSEL + j)*(ulong)K;
        float s = 0.0f;
        for (uint b = lane; b < nb; b += 32u) {
            device const uchar* blk = dr + b*18u;
            ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);
            float d = (float)as_type<half>(dbits);
            s += d * iq4nl_block(blk, aj + b*32u);
        }
        s = simd_sum(s);
        total += wgt[row*KSEL + j] * s;
    }
    if (lane == 0u) {
        ulong o = (ulong)row*(ulong)N + out_row;
        x[o] += total + shx[o]/(1.0f + exp(-shg[row]));
    }
}
"#;

/// Diagnostic: hash the expert bytes through the same address table the expert kernels
/// read, with the same slot indexing (`tbl[slot]`, as the direct kernels do
/// `((device const uchar*)wg[e])`).
///
/// It compares what the GPU sees at those addresses against what the host wrote there.
/// A CPU-side audit cannot: it reads `contents()`, a different view of the mapping,
/// which has measured correct while output was still wrong.
///
/// Samples rather than hashing every byte — an expert is ~2 MB and a layer holds ~160 of
/// them across three tensors, so a full hash would read ~1 GB per layer. `SAMPLES`
/// evenly spaced 16-byte chunks detect a wrong expert, stale content or a shifted base,
/// but not a single flipped byte between samples.
pub const EXPERT_HASH_KERNEL: &str = r#"
kernel void expert_table_hash(device const ulong* tbl [[buffer(0)]],
                              device uint*        out [[buffer(1)]],
                              constant uint&      stride [[buffer(2)]],
                              constant uint&      samples [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    device const uchar* p = (device const uchar*)tbl[gid];
    uint h = 2166136261u;                 // FNV-1a
    if (p == 0) { out[gid] = 0u; return; } // null entry: report distinctly
    uint span = stride > 16u ? stride - 16u : 0u;
    for (uint s = 0; s < samples; ++s) {
        uint base = samples > 1 ? (uint)(((ulong)span * (ulong)s) / (ulong)(samples - 1)) : 0u;
        for (uint i = 0; i < 16u; ++i) { h ^= (uint)p[base + i]; h *= 16777619u; }
    }
    out[gid] = h;
}
"#;

/// Reuse the tested arithmetic verbatim; change only expert-base addressing.
/// Tables contain gpuAddress values for retained buffers, declared with useResource.
pub fn direct_kernels() -> String {
    let mut out = String::new();
    for name in ["moe_gu_iq3s", "moe_gu_iq3s_m", "moe_gu_iq4xs", "moe_gu_iq4xs_m", "moe_down_iq4nl", "moe_down_iq4nl_m"] {
        let start = MOE_IQ_KERNELS.find(&format!("kernel void {name}(" )).unwrap();
        let rest = &MOE_IQ_KERNELS[start..];
        let end = rest.find("\n}").unwrap() + 2;
        let mut body = rest[..end].replace(&format!("void {name}("), &format!("void {name}_direct("));
        for w in if name.contains("_gu_") { vec!["wg", "wu"] } else { vec!["wd"] } {
            body = body.replace(&format!("device const uchar* {w} [["),
                &format!("device const ulong* {w} [["));
            let old = if w == "wd" { format!("{w} + (ulong)e*(ulong)N*(ulong)rb") }
                else { format!("{w} + (ulong)e*(ulong)N*rb") };
            assert!(body.contains(&old), "expert addressing changed in {name}");
            body = body.replace(&old, &format!("((device const uchar*){w}[e])"));
        }
        out.push_str(&body);
        out.push('\n');
    }
    out
}
