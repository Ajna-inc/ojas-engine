//! Sectioned M-RoPE in `rope_qk_store_m`: the guards that let one rope kernel serve
//! both the text decoder and an image span.
//!
//! Three failure modes, one test each:
//!
//! 1. It changes an existing model. Extending the kernel instead of adding one is
//!    only safe while every shipping model keeps running through it.
//!    `mode_off_is_bit_identical_to_the_unextended_kernel` compiles a verbatim copy of
//!    the pre-extension kernel and demands bit equality, and
//!    `degenerate_sections_reproduce_plain_rope_bit_for_bit` demands the same of the
//!    sectioned path whenever every position stream agrees — the case for every text
//!    token of every model in the tree, and the claim the comments at `mod.rs:28`,
//!    `graph_decode.rs:307` and `graph_qwen4exp.rs:359` rest on.
//! 2. The section mapping is wrong. `sectioned_angles_match_cpu_oracle` checks a
//!    genuine 2-D span (distinct t/h/w/e) against an independent scalar computation,
//!    then re-runs the oracle with two streams swapped to show the comparison can
//!    fail.
//! 3. The cache moves. `pos = base_pos + m` is both the angle and the KV cache row,
//!    and only the angle is sectioned. `image_positions_do_not_move_cache_rows` drives
//!    the kernel with image positions far from the cache rows and asserts the rows
//!    outside `[base_pos, base_pos+m)` still hold their guard value.
//!
//! Every buffer is allocated with guard regions on both sides and bound at an
//! offset, so an over-store fails rather than silently corrupting a neighbouring
//! allocation (the idiom from `dispatch.rs:216,235-238,261`). Shapes are ragged —
//! `m in {1,2,3,7,8}`, partial-rotary `rd` that is neither `hd` nor a power of two —
//! for the same reason `dispatch.rs` uses `n=13`.

// The Metal device is macOS-only (`ojas-metal/src/lib.rs`); only its `kernels` source table
// builds elsewhere. These tests drive a real `MetalGpu`, so they compile only where one exists.
#![cfg(target_os = "macos")]

use metal::{ComputePipelineState, MTLResourceOptions, MTLSize};
use ojas_core::Device as _;
use ojas_metal::MetalGpu;
use ojas_metal::kernels::ops::{MROPE_INTERLEAVED, MROPE_OFF, MROPE_SECTIONS, MROPE_VISION,
                               mrope_desc, rope_mode};

/// A verbatim copy of `rope_qk_store_m` as it stood before sections were added,
/// renamed. The mode-off path is compared against this, so "byte-identical to the
/// unmodified kernel" is an executed claim rather than a reading of the diff.
const REF_FAMILY: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void rope_qk_store_m_ref(device float* vq [[buffer(0)]], device float* vk [[buffer(1)]],
    device const float* vv [[buffer(2)]], device half* kc [[buffer(3)]], device half* vc [[buffer(4)]],
    constant uint& hd [[buffer(5)]], constant uint& base_pos [[buffer(6)]], constant float& base [[buffer(7)]],
    constant uint& Aq [[buffer(8)]], constant uint& Ak [[buffer(9)]], constant uint& kvdim [[buffer(10)]],
    constant uint& M [[buffer(11)]], constant uint& neox [[buffer(12)]], constant uint& rd [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
    uint perTok = Aq + Ak + kvdim; uint total = M*perTok; if (gid >= total) { return; }
    uint m = gid/perTok; uint w = gid%perTok; uint hf = hd/2u; uint rf = rd/2u; uint pos = base_pos + m;
    if (w < Aq) {
        uint head=w/hf; uint j=w%hf; uint b = m*(2u*Aq) + head*hd;
        if (j >= rf) { return; }                  // partial rope: q dims >= rd untouched
        uint a0 = neox ? b+j : b+2u*j; uint a1 = neox ? b+rf+j : b+2u*j+1u;
        float freq=1.0/pow(base,2.0*float(j)/float(rd)); float ang=float(pos)*freq; float s=sin(ang),c=cos(ang);
        float x0=vq[a0], x1=vq[a1]; vq[a0]=x0*c-x1*s; vq[a1]=x0*s+x1*c;
    } else if (w < Aq+Ak) {
        uint wk=w-Aq; uint head=wk/hf; uint j=wk%hf; uint b=m*kvdim + head*hd;
        if (j >= rf) {                            // unrotated k dims: plain copy to cache
            uint e0 = head*hd + rd + 2u*(j-rf);
            ulong cb=(ulong)(base_pos+m)*(ulong)kvdim;
            kc[cb+e0] = half(vk[m*kvdim + e0]); kc[cb+e0+1u] = half(vk[m*kvdim + e0+1u]);
            return;
        }
        uint a0 = neox ? b+j : b+2u*j; uint a1 = neox ? b+rf+j : b+2u*j+1u;
        uint o0 = neox ? j : 2u*j; uint o1 = neox ? rf+j : 2u*j+1u;
        float freq=1.0/pow(base,2.0*float(j)/float(rd)); float ang=float(pos)*freq; float s=sin(ang),c=cos(ang);
        float x0=vk[a0], x1=vk[a1]; float n0=x0*c-x1*s, n1=x0*s+x1*c;
        ulong cb=(ulong)(base_pos+m)*(ulong)kvdim + (ulong)(head*hd);
        kc[cb+o0]=half(n0); kc[cb+o1]=half(n1);
    } else {
        uint e=w-Aq-Ak; vc[(ulong)(base_pos+m)*(ulong)kvdim + e] = half(vv[m*kvdim + e]);
    }
}
"#;

// ---------------------------------------------------------------------------
// Geometry + buffers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Geo {
    hd: u32,        // head dim
    nq: u32,        // q heads
    nkv: u32,       // kv heads
    rd: u32,        // rotary dims per head (rd == hd is full rope)
    m: u32,         // tokens in this dispatch
    base_pos: u32,  // first KV cache row
    maxseq: u32,    // cache rows allocated
    base: f32,      // rope freq base
    sec: [u32; 4],  // M-RoPE sections, in cos/sin PAIRS; must sum to rd/2
}

impl Geo {
    fn kvdim(&self) -> u32 { self.nkv * self.hd }
    fn aq(&self) -> u32 { self.nq * self.hd / 2 }
    fn ak(&self) -> u32 { self.nkv * self.hd / 2 }
    fn qlen(&self) -> usize { (self.m * self.nq * self.hd) as usize }
    fn kvlen(&self) -> usize { (self.m * self.kvdim()) as usize }
    fn cachelen(&self) -> usize { (self.maxseq * self.kvdim()) as usize }
}

/// Ragged on purpose: `m` crosses the 64-thread threadgroup in awkward places,
/// `rd` is partial and not a power of two, and one row is the surya-2 geometry
/// (hd=256, rd=64, sections [11,11,10,0] summing to n_rot/2 = 32).
fn cases() -> Vec<Geo> {
    vec![
        Geo { hd: 32, nq: 2, nkv: 1, rd: 32, m: 1, base_pos: 0, maxseq: 6,  base: 10000.0, sec: [6, 5, 5, 0] },
        Geo { hd: 32, nq: 3, nkv: 1, rd: 22, m: 2, base_pos: 5, maxseq: 11, base: 10000.0, sec: [4, 4, 3, 0] },
        Geo { hd: 16, nq: 1, nkv: 1, rd: 6,  m: 3, base_pos: 1, maxseq: 9,  base: 10000.0, sec: [1, 1, 1, 0] },
        Geo { hd: 64, nq: 4, nkv: 2, rd: 10, m: 7, base_pos: 13, maxseq: 24, base: 10000.0, sec: [2, 2, 1, 0] },
        // exercises the FOURTH stream (e): interleaved leaves sectors 8 and 11 to it.
        Geo { hd: 24, nq: 2, nkv: 2, rd: 24, m: 7, base_pos: 2, maxseq: 13, base: 10000.0, sec: [4, 4, 2, 2] },
        // surya-2: qwen35, n_rot 64 of hd 256, freq_base 1e7, sections [11,11,10,0].
        Geo { hd: 256, nq: 2, nkv: 1, rd: 64, m: 8, base_pos: 3, maxseq: 14, base: 1.0e7, sec: [11, 11, 10, 0] },
    ]
}

const GUARD: usize = 16;        // f32 guard elements each side (64 B — keeps the bind offset aligned)
const GUARD_H: usize = 32;      // f16 guard elements each side (64 B)
const SENT_F32: f32 = 12345.0;
const SENT_F16: u16 = 0xDEAD;

fn det(i: usize, salt: usize) -> f32 {
    (((i * 2654435761 + salt * 40503) % 2048) as f32) / 1024.0 - 1.0
}

fn buf_f32(gpu: &MetalGpu, live: &[f32]) -> metal::Buffer {
    let mut v = vec![SENT_F32; GUARD];
    v.extend_from_slice(live);
    v.extend(std::iter::repeat(SENT_F32).take(GUARD));
    gpu.device.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64,
                                    MTLResourceOptions::StorageModeShared)
}

fn buf_f16(gpu: &MetalGpu, live: usize) -> metal::Buffer {
    let v = vec![SENT_F16; GUARD_H * 2 + live];
    gpu.device.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 2) as u64,
                                    MTLResourceOptions::StorageModeShared)
}

fn buf_u32(gpu: &MetalGpu, v: &[u32]) -> metal::Buffer {
    gpu.device.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64,
                                    MTLResourceOptions::StorageModeShared)
}

fn read_f32(b: &metal::Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) }.to_vec()
}
fn read_u16(b: &metal::Buffer, n: usize) -> Vec<u16> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const u16, n) }.to_vec()
}

/// Whole buffers INCLUDING their guard regions, so every assertion can check both
/// the payload and the over-store at once.
struct Run { q: Vec<f32>, kc: Vec<u16>, vc: Vec<u16> }

impl Run {
    fn qlive(&self) -> &[f32] { &self.q[GUARD..self.q.len() - GUARD] }
    fn kclive(&self) -> &[u16] { &self.kc[GUARD_H..self.kc.len() - GUARD_H] }
    fn vclive(&self) -> &[u16] { &self.vc[GUARD_H..self.vc.len() - GUARD_H] }
    fn check_guards(&self, what: &str) {
        let f = |v: &[f32], n: &str| {
            assert!(v[..GUARD].iter().all(|&x| x == SENT_F32) && v[v.len() - GUARD..].iter().all(|&x| x == SENT_F32),
                    "{what}: {n} guard region overwritten");
        };
        let h = |v: &[u16], n: &str| {
            assert!(v[..GUARD_H].iter().all(|&x| x == SENT_F16) && v[v.len() - GUARD_H..].iter().all(|&x| x == SENT_F16),
                    "{what}: {n} guard region overwritten");
        };
        f(&self.q, "q");
        h(&self.kc, "kcache");
        h(&self.vc, "vcache");
    }
}

/// One dispatch. `mpos` is bound at buffer 14 only when it is `Some` — the
/// reference kernel does not declare it, and the mode-off path must not read it.
#[allow(clippy::too_many_arguments)]
fn run(gpu: &MetalGpu, pipe: &ComputePipelineState, g: &Geo, neox_arg: u32,
       mpos: Option<&[u32]>, qin: &[f32], kin: &[f32], vin: &[f32]) -> Run {
    let (qb, kb, vb) = (buf_f32(gpu, qin), buf_f32(gpu, kin), buf_f32(gpu, vin));
    let (kcb, vcb) = (buf_f16(gpu, g.cachelen()), buf_f16(gpu, g.cachelen()));
    let posb = mpos.map(|p| buf_u32(gpu, p));

    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(&qb), (GUARD * 4) as u64);
    enc.set_buffer(1, Some(&kb), (GUARD * 4) as u64);
    enc.set_buffer(2, Some(&vb), (GUARD * 4) as u64);
    enc.set_buffer(3, Some(&kcb), (GUARD_H * 2) as u64);
    enc.set_buffer(4, Some(&vcb), (GUARD_H * 2) as u64);
    for (i, v) in [(5u64, g.hd), (6, g.base_pos), (8, g.aq()), (9, g.ak()),
                   (10, g.kvdim()), (11, g.m), (12, neox_arg), (13, g.rd)] {
        enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void);
    }
    enc.set_bytes(7, 4, &g.base as *const f32 as *const std::ffi::c_void);
    if let Some(p) = &posb { enc.set_buffer(14, Some(p), 0); }
    let total = g.m * (g.aq() + g.ak() + g.kvdim());
    enc.dispatch_thread_groups(MTLSize::new(total.div_ceil(64) as u64, 1, 1), MTLSize::new(64, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Run {
        q: read_f32(&qb, GUARD * 2 + qin.len()),
        kc: read_u16(&kcb, GUARD_H * 2 + g.cachelen()),
        vc: read_u16(&vcb, GUARD_H * 2 + g.cachelen()),
    }
}

fn inputs(g: &Geo) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    ((0..g.qlen()).map(|i| det(i, 1)).collect(),
     (0..g.kvlen()).map(|i| det(i, 2)).collect(),
     (0..g.kvlen()).map(|i| det(i, 3)).collect())
}

/// Text positions: every stream carries the sequence position, which is the state
/// every model in the tree is in today.
fn degenerate_pos(g: &Geo) -> Vec<[u32; 4]> {
    (0..g.m).map(|i| { let p = g.base_pos + i; [p, p, p, p] }).collect()
}

// ---------------------------------------------------------------------------
// Independent scalar CPU model of the kernel
// ---------------------------------------------------------------------------

/// (position driving the angle, exponent index). Written from the ggml contract
/// (`ggml.h:1922-1935`, `ggml-cpu/ops.cpp:ggml_mrope_cache_init`), not from the MSL.
fn sel(mode: u32, s: [u32; 4], pos: [u32; 4], j: u32) -> (u32, u32) {
    let sect: u32 = s.iter().sum();
    if sect == 0 { return (pos[0], j); }
    let sector = j % sect;
    let (idx, start) = if mode == MROPE_INTERLEAVED {
        // [t h w t h w ...]: stream = sector % 3, falling through to the 4th
        // stream once that stream's quota of sectors is used up.
        match sector % 3 {
            1 if sector < 3 * s[1] => (1, 0),
            2 if sector < 3 * s[2] => (2, 0),
            0 if sector < 3 * s[0] => (0, 0),
            _ => (3, 0),
        }
    } else if sector < s[0] { (0, 0) }
    else if sector < s[0] + s[1] { (1, s[0]) }
    else if sector < s[0] + s[1] + s[2] { (2, s[0] + s[1]) }
    else { (3, s[0] + s[1] + s[2]) };
    (pos[idx as usize], if mode == MROPE_VISION { sector - start } else { j })
}

/// Scalar f64 reimplementation of the whole kernel: q rotated in place, k rotated
/// straight into the cache, unrotated tail copied, v copied. Returns
/// (q, kcache, vcache) with the cache in f32 (the kernel stores f16).
fn oracle(g: &Geo, mode: u32, neox: bool, pos: &[[u32; 4]],
          qin: &[f32], kin: &[f32], vin: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (hd, rd) = (g.hd as usize, g.rd as usize);
    let (hf, rf) = (hd / 2, rd / 2);
    let kvdim = g.kvdim() as usize;
    let mut q = qin.to_vec();
    let mut kc = vec![f32::NAN; g.cachelen()];
    let mut vc = vec![f32::NAN; g.cachelen()];
    for t in 0..g.m as usize {
        let angle = |j: usize| -> (f64, f64) {
            let (p, jx) = if mode == MROPE_OFF { (g.base_pos as usize + t, j as u32) }
                          else { let (p, jx) = sel(mode, g.sec, pos[t], j as u32); (p as usize, jx) };
            let freq = (g.base as f64).powf(-2.0 * jx as f64 / rd as f64);
            let ang = p as f64 * freq;
            (ang.cos(), ang.sin())
        };
        for head in 0..g.nq as usize {
            let b = t * g.nq as usize * hd + head * hd;
            for j in 0..rf {
                let (a0, a1) = if neox { (b + j, b + rf + j) } else { (b + 2 * j, b + 2 * j + 1) };
                let (c, s) = angle(j);
                let (x0, x1) = (q[a0] as f64, q[a1] as f64);
                q[a0] = (x0 * c - x1 * s) as f32;
                q[a1] = (x0 * s + x1 * c) as f32;
            }
        }
        let row = (g.base_pos as usize + t) * kvdim;
        for head in 0..g.nkv as usize {
            let b = t * kvdim + head * hd;
            for j in 0..rf {
                let (a0, a1) = if neox { (b + j, b + rf + j) } else { (b + 2 * j, b + 2 * j + 1) };
                let (o0, o1) = if neox { (j, rf + j) } else { (2 * j, 2 * j + 1) };
                let (c, s) = angle(j);
                let (x0, x1) = (kin[a0] as f64, kin[a1] as f64);
                kc[row + head * hd + o0] = (x0 * c - x1 * s) as f32;
                kc[row + head * hd + o1] = (x0 * s + x1 * c) as f32;
            }
            for j in rf..hf {                       // unrotated tail: plain copy
                let e0 = head * hd + rd + 2 * (j - rf);
                kc[row + e0] = kin[t * kvdim + e0];
                kc[row + e0 + 1] = kin[t * kvdim + e0 + 1];
            }
        }
        for e in 0..kvdim { vc[row + e] = vin[t * kvdim + e]; }
    }
    (q, kc, vc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn gpu() -> Option<MetalGpu> {
    match MetalGpu::new() {
        Ok(g) => Some(g),
        Err(e) => { eprintln!("skipping: no Metal GPU ({e})"); None }
    }
}

fn pipes(gpu: &MetalGpu) -> (ComputePipelineState, ComputePipelineState) {
    let ops = ojas_metal::kernels::family_source("ops").expect("ops family");
    let live = gpu.pipeline(ops, "rope_qk_store_m").expect("rope_qk_store_m");
    let refk = gpu.pipeline(REF_FAMILY, "rope_qk_store_m_ref").expect("reference kernel");
    (live, refk)
}

fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }

/// The pair -> stream mapping, checked against a hand-written table rather than a
/// second copy of the branch tree. For [11,11,10,0] over pairs 0..32 the interleaved
/// form is exactly `j % 3`, which is why the sections are [11,11,10] and not [16,8,8];
/// implementing the contiguous form for a qwen35 model breaks that silently.
#[test]
fn section_mapping_matches_hand_table() {
    let sec = [11u32, 11, 10, 0];
    let pos = [100u32, 200, 300, 400];
    for j in 0..32u32 {
        let (p, jx) = sel(MROPE_INTERLEAVED, sec, pos, j);
        assert_eq!(jx, j, "interleaved must keep the plain exponent index");
        assert_eq!(p, pos[(j % 3) as usize], "interleaved j={j}");
    }
    for j in 0..32u32 {
        let (p, jx) = sel(MROPE_SECTIONS, sec, pos, j);
        assert_eq!(jx, j, "contiguous must keep the plain exponent index");
        let want = if j < 11 { 0 } else if j < 22 { 1 } else { 2 };
        assert_eq!(p, pos[want], "contiguous j={j}");
    }
    // Vision restarts theta at each boundary, the one mode that is not degenerate
    // with plain rope (ggml.h:1934).
    let vs = [4u32, 4, 4, 4];
    for j in 0..16u32 {
        let (p, jx) = sel(MROPE_VISION, vs, pos, j);
        assert_eq!(p, pos[(j / 4) as usize]);
        assert_eq!(jx, j % 4, "vision exponent must restart per section");
    }
    // Every stream equal => the mapping cannot matter, whatever the mode.
    for mode in [MROPE_SECTIONS, MROPE_INTERLEAVED] {
        for j in 0..32u32 {
            assert_eq!(sel(mode, sec, [7, 7, 7, 7], j), (7, j));
        }
    }
}

/// Mode 0 must be the kernel that was there before. Compiled reference, bit equality.
#[test]
fn mode_off_is_bit_identical_to_the_unextended_kernel() {
    let Some(gpu) = gpu() else { return };
    let (live, refk) = pipes(&gpu);
    for g in cases() {
        for neox in [false, true] {
            let (qi, ki, vi) = inputs(&g);
            // Bind a descriptor full of absurd positions: mode 0 must not read it.
            let junk = mrope_desc(g.sec, &(0..g.m).map(|i| [9_000 + i, 8_000, 7_000, 6_000]).collect::<Vec<_>>());
            let got = run(&gpu, &live, &g, rope_mode(neox, MROPE_OFF), Some(&junk), &qi, &ki, &vi);
            // And the configuration every existing call site is in: buffer 14 never
            // bound. The argument table still holds whatever the previous dispatch in
            // the encoder left at index 14 (the qkv GEMM binds attn_k.bias there,
            // dispatch.rs:611), so this is the case that must not dereference it. The
            // run is correct but not validation-clean: MTL_DEBUG_LAYER=1 asserts
            // `missing buffer binding at index 14` on a declared-and-unbound buffer
            // whether or not the shader reads it. Every call site should bind a live
            // buffer at 14 even at mode 0, as the `Some(&junk)` run above does.
            let unbound = run(&gpu, &live, &g, neox as u32, None, &qi, &ki, &vi);
            let want = run(&gpu, &refk, &g, neox as u32, None, &qi, &ki, &vi);
            got.check_guards("mode-off");
            unbound.check_guards("mode-off, buffer 14 unbound");
            want.check_guards("reference");
            assert_eq!(bits(&got.q), bits(&want.q), "q differs at mode 0, {g:?} neox={neox}");
            assert_eq!(got.kc, want.kc, "kcache differs at mode 0, {g:?} neox={neox}");
            assert_eq!(got.vc, want.vc, "vcache differs at mode 0, {g:?} neox={neox}");
            assert_eq!(bits(&unbound.q), bits(&want.q), "q differs with buffer 14 unbound, {g:?} neox={neox}");
            assert_eq!(unbound.kc, want.kc, "kcache differs with buffer 14 unbound, {g:?} neox={neox}");
            assert_eq!(unbound.vc, want.vc, "vcache differs with buffer 14 unbound, {g:?} neox={neox}");
        }
    }
}

/// The degeneracy guard. With t == h == w == e every section selects the same
/// position, so sectioned M-RoPE must reproduce the plain-rope kernel bit for bit.
/// Every text token of every model in the tree is in that state, which makes this the
/// regression guard for the whole shipping surface.
#[test]
fn degenerate_sections_reproduce_plain_rope_bit_for_bit() {
    let Some(gpu) = gpu() else { return };
    let (live, refk) = pipes(&gpu);
    let mut checked = 0usize;
    for g in cases() {
        assert_eq!(g.sec.iter().sum::<u32>(), g.rd / 2, "sections must sum to rd/2 for {g:?}");
        for neox in [false, true] {
            let (qi, ki, vi) = inputs(&g);
            let want = run(&gpu, &refk, &g, neox as u32, None, &qi, &ki, &vi);
            want.check_guards("reference");
            let desc = mrope_desc(g.sec, &degenerate_pos(&g));
            for mode in [MROPE_SECTIONS, MROPE_INTERLEAVED] {
                let got = run(&gpu, &live, &g, rope_mode(neox, mode), Some(&desc), &qi, &ki, &vi);
                got.check_guards("sectioned");
                assert_eq!(bits(&got.q), bits(&want.q),
                           "DEGENERACY BROKEN (q) mode={mode} neox={neox} {g:?}");
                assert_eq!(got.kc, want.kc, "DEGENERACY BROKEN (kcache) mode={mode} neox={neox} {g:?}");
                assert_eq!(got.vc, want.vc, "DEGENERACY BROKEN (vcache) mode={mode} neox={neox} {g:?}");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, cases().len() * 4);
    eprintln!("degeneracy: {checked} sectioned dispatches bit-identical to plain rope");
}

/// A real 2-D span: distinct t/h/w/e, checked against the scalar oracle. The last
/// block re-runs the oracle with h and w SWAPPED and requires a mismatch, so a
/// tolerance wide enough to hide a wrong section mapping fails the test.
#[test]
fn sectioned_angles_match_cpu_oracle() {
    let Some(gpu) = gpu() else { return };
    let (live, _) = pipes(&gpu);
    for g in cases() {
        let kvdim = g.kvdim() as usize;
        // Well separated so a mis-selected stream is an O(1) error, and small
        // enough that the angle stays in a range where f32 sin/cos is accurate.
        let pos: Vec<[u32; 4]> = (0..g.m).map(|i| [11 + i, 53 + 2 * i, 97 + 3 * i, 131 + i]).collect();
        let desc = mrope_desc(g.sec, &pos);
        for neox in [false, true] {
            for mode in [MROPE_SECTIONS, MROPE_INTERLEAVED, MROPE_VISION] {
                let (qi, ki, vi) = inputs(&g);
                let got = run(&gpu, &live, &g, rope_mode(neox, mode), Some(&desc), &qi, &ki, &vi);
                got.check_guards("oracle case");
                let (wq, wkc, wvc) = oracle(&g, mode, neox, &pos, &qi, &ki, &vi);
                // The oracle only defines the rows this dispatch writes; the rest of
                // the cache is asserted to be untouched guard values instead.
                let (lo, hi) = ((g.base_pos as usize) * kvdim, ((g.base_pos + g.m) as usize) * kvdim);
                cmp_f32(got.qlive(), &wq, 2e-3, &format!("q mode={mode} neox={neox} {g:?}"));
                cmp_f16(&got.kclive()[lo..hi], &wkc[lo..hi], 4e-3, &format!("kcache mode={mode} neox={neox} {g:?}"));
                cmp_f16(&got.vclive()[lo..hi], &wvc[lo..hi], 4e-3, &format!("vcache mode={mode} neox={neox} {g:?}"));
                assert!(got.kclive()[..lo].iter().chain(&got.kclive()[hi..]).all(|&x| x == SENT_F16)
                        && got.vclive()[..lo].iter().chain(&got.vclive()[hi..]).all(|&x| x == SENT_F16),
                        "cache written outside [{}, {}) mode={mode} {g:?}", g.base_pos, g.base_pos + g.m);

                // Mutation check: the same oracle with h/w swapped must not match.
                if g.sec[1] > 0 && g.sec[2] > 0 && pos.iter().any(|p| p[1] != p[2]) {
                    let swapped: Vec<[u32; 4]> = pos.iter().map(|p| [p[0], p[2], p[1], p[3]]).collect();
                    let (bq, _, _) = oracle(&g, mode, neox, &swapped, &qi, &ki, &vi);
                    let worst = got.qlive().iter().zip(&bq).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    assert!(worst > 2e-3,
                            "the oracle comparison has no power: swapping h/w changes q by only {worst} ({g:?})");
                }
            }
        }
    }
}

/// The angle is sectioned; the cache row is not. Image positions in the thousands
/// must still land in rows `[base_pos, base_pos+m)` and nowhere else — a cache
/// addressed by the image position would scatter the span and leave holes.
#[test]
fn image_positions_do_not_move_cache_rows() {
    let Some(gpu) = gpu() else { return };
    let (live, _) = pipes(&gpu);
    for g in cases() {
        let kvdim = g.kvdim() as usize;
        let pos: Vec<[u32; 4]> = (0..g.m).map(|i| [5000 + i, 37, 41 + i, 3]).collect();
        let desc = mrope_desc(g.sec, &pos);
        let (qi, ki, vi) = inputs(&g);
        let got = run(&gpu, &live, &g, rope_mode(true, MROPE_INTERLEAVED), Some(&desc), &qi, &ki, &vi);
        got.check_guards("cache rows");
        for r in 0..g.maxseq as usize {
            let live_row = r >= g.base_pos as usize && r < (g.base_pos + g.m) as usize;
            let kc = &got.kclive()[r * kvdim..(r + 1) * kvdim];
            let vc = &got.vclive()[r * kvdim..(r + 1) * kvdim];
            if live_row {
                assert!(kc.iter().any(|&x| x != SENT_F16) && vc.iter().all(|&x| x != SENT_F16),
                        "row {r} should have been written ({g:?})");
            } else {
                assert!(kc.iter().all(|&x| x == SENT_F16) && vc.iter().all(|&x| x == SENT_F16),
                        "row {r} was written but is outside [{}, {}) ({g:?})",
                        g.base_pos, g.base_pos + g.m);
            }
        }
        // and the rows that WERE written hold token r-base_pos, not token t%m.
        let (_, wkc, _) = oracle(&g, MROPE_INTERLEAVED, true, &pos, &qi, &ki, &vi);
        for r in g.base_pos as usize..(g.base_pos + g.m) as usize {
            cmp_f16(&got.kclive()[r * kvdim..(r + 1) * kvdim],
                    &wkc[r * kvdim..(r + 1) * kvdim], 4e-3, &format!("cache row {r} ({g:?})"));
        }
    }
}

fn cmp_f32(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() <= tol, "{what}: [{i}] {a} vs {b}");
    }
}

fn cmp_f16(got: &[u16], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        let a = half::f16::from_bits(a).to_f32();
        assert!((a - b).abs() <= tol, "{what}: [{i}] {a} vs {b}");
    }
}
