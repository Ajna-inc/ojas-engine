//! CUDA `rope_m` / `rope_qk_store_m` (kernels/rope_m.rs) against a CPU reference written
//! directly from the Metal kernel's maths (`ojas-metal/src/kernels/ops.rs`: `mrope_sel` and
//! the rotation), for surya-2's decoder shape: head dim 256, n_rot 64, base 1e7, 8 q heads /
//! 2 kv heads, sections [11,11,10,0].
//!
//! Covered: all four modes (OFF, MROPE, IMROPE, VISION), NeoX and adjacent pairing, a
//! text-only span (every M-RoPE mode must collapse to OFF bit-for-bit, and OFF must equal the
//! existing `rope_partial_m` bit-for-bit), and a text | 64x48 merged image grid | text span
//! with Qwen-VL 3-D positions. The KV cache row is base_pos + m whatever the positions are.
//!
//! ```text
//! SP=~/.local/lib/python3.10/site-packages/nvidia
//! LD_LIBRARY_PATH=$SP/cuda_nvrtc/lib:$SP/cublas/lib OJAS_CUDA_INCLUDE=$SP/cuda_runtime/include \
//!   cargo test --release -p ojas-cuda --test surya_mrope -- --ignored --nocapture
//! ```
use ojas_core::{Device, KernelRuntime};
use ojas_cuda::{CudaGpu, CuBuf};

const HD: usize = 256;
const NROT: usize = 64;
const NQ: usize = 8;
const NKV: usize = 2;
const BASE: f32 = 1.0e7;
const SECTIONS: [u32; 4] = [11, 11, 10, 0];
const SHIFT: u32 = 8;
const OFF: u32 = 0;
const MROPE: u32 = 1;
const IMROPE: u32 = 2;
const VISION: u32 = 3;

fn gpu() -> CudaGpu {
    let mut g = CudaGpu::new(0).expect("requires a CUDA box: cargo test -- --ignored");
    g.ensure_family("rope_m").expect("rope_m family compiles");
    g.ensure_family("ops").expect("ops family compiles");
    g
}

fn tvec(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n).map(|_| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((s >> 9) as f32 / (1 << 23) as f32) - 1.0
    }).collect()
}

/// Per-token (t,h,w,e). `text_pre` text tokens, a gh x gw image, `text_post` text tokens —
/// Qwen2-VL numbering: the image sits at t = start with h/w = start + row/col, and text
/// resumes at start + max(gh, gw).
fn positions(text_pre: u32, gh: u32, gw: u32, text_post: u32) -> Vec<[u32; 4]> {
    let mut p = vec![];
    for i in 0..text_pre { p.push([i, i, i, 0]); }
    let st = text_pre;
    for r in 0..gh { for c in 0..gw { p.push([st, st + r, st + c, 0]); } }
    let nx = st + gh.max(gw);
    for i in 0..text_post { let t = nx + i; p.push([t, t, t, 0]); }
    p
}

fn mpos_desc(sections: [u32; 4], pos: &[[u32; 4]]) -> Vec<u32> {
    let mut d = sections.to_vec();
    for p in pos { d.extend_from_slice(p); }
    d
}

// ---------------- CPU reference, Metal line for line ----------------

fn mrope_sel(mpos: &[u32], mode: u32, m: u32, j: u32) -> (u32, u32) {
    let (s0, s1, s2, s3) = (mpos[0], mpos[1], mpos[2], mpos[3]);
    let sect = s0 + s1 + s2 + s3;
    let at = |s: u32| mpos[(4 + 4 * m + s) as usize];
    if sect == 0 { return (at(0), j); }
    let sector = j % sect;
    let (sel, start);
    if mode == 2 {
        let r = sector % 3;
        start = 0;
        sel = if r == 1 && sector < 3 * s1 { 1 }
              else if r == 2 && sector < 3 * s2 { 2 }
              else if r == 0 && sector < 3 * s0 { 0 }
              else { 3 };
    } else if sector < s0 { sel = 0; start = 0; }
    else if sector < s0 + s1 { sel = 1; start = s0; }
    else if sector < s0 + s1 + s2 { sel = 2; start = s0 + s1; }
    else { sel = 3; start = s0 + s1 + s2; }
    (at(sel), if mode == 3 { sector - start } else { j })
}

fn rot(pos: u32, j: u32, rd: u32, x0: f32, x1: f32) -> (f32, f32) {
    let freq = 1.0f32 / BASE.powf(2.0 * j as f32 / rd as f32);
    let ang = pos as f32 * freq;
    let (s, c) = (ang.sin(), ang.cos());
    (x0 * c - x1 * s, x0 * s + x1 * c)
}

/// Returns (q rotated, kcache f32 view, vcache f32 view) for `rope_qk_store_m`.
#[allow(clippy::too_many_arguments)]
fn ref_qk_store(q: &[f32], k: &[f32], v: &[f32], base_pos: u32, m_tok: u32, neox: u32,
                rd: u32, mpos: &[u32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (hd, kvdim) = (HD as u32, (NKV * HD) as u32);
    let (nx, mode) = (neox & 1, neox >> SHIFT);
    let rf = rd / 2;
    let rows = (base_pos + m_tok) as usize * kvdim as usize;
    let (mut q, mut kc, mut vc) = (q.to_vec(), vec![0f32; rows], vec![0f32; rows]);
    for m in 0..m_tok {
        let pick = |j: u32| if mode != 0 { mrope_sel(mpos, mode, m, j) } else { (base_pos + m, j) };
        let pair = |b: u32, j: u32| if nx == 1 { (b + j, b + rf + j) } else { (b + 2 * j, b + 2 * j + 1) };
        for head in 0..NQ as u32 {
            let b = m * (NQ as u32 * hd) + head * hd;
            for j in 0..rf {
                let (a0, a1) = pair(b, j);
                let (p, e) = pick(j);
                let (n0, n1) = rot(p, e, rd, q[a0 as usize], q[a1 as usize]);
                q[a0 as usize] = n0; q[a1 as usize] = n1;
            }
        }
        let row = (base_pos + m) * kvdim;
        for head in 0..NKV as u32 {
            let b = m * kvdim + head * hd;
            for j in 0..rf {
                let (a0, a1) = pair(b, j);
                let (o0, o1) = pair(row + head * hd, j);
                let (p, e) = pick(j);
                let (n0, n1) = rot(p, e, rd, k[a0 as usize], k[a1 as usize]);
                kc[o0 as usize] = n0; kc[o1 as usize] = n1;
            }
            for e in (head * hd + rd)..(head * hd + hd) {
                kc[(row + e) as usize] = k[(m * kvdim + e) as usize];
            }
        }
        for e in 0..kvdim { vc[(row + e) as usize] = v[(m * kvdim + e) as usize]; }
    }
    // the cache is f16
    let h = |x: &mut Vec<f32>| x.iter_mut().for_each(|y| *y = half::f16::from_f32(*y).to_f32());
    h(&mut kc); h(&mut vc);
    (q, kc, vc)
}

// ---------------- GPU drivers ----------------

fn u32s(g: &CudaGpu, d: &[u32]) -> CuBuf {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    g.upload_bytes(&b).unwrap()
}

fn read_f16(g: &CudaGpu, b: &CuBuf, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 2];
    g.read_bytes(b, 0, &mut raw).unwrap();
    raw.chunks(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
}

#[allow(clippy::too_many_arguments)]
fn gpu_qk_store(g: &CudaGpu, q: &[f32], k: &[f32], v: &[f32], base_pos: u32, m_tok: u32,
                neox: u32, rd: u32, mpos: &[u32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let kvdim = (NKV * HD) as u32;
    let (aq, ak) = ((NQ * HD / 2) as u32, kvdim / 2);
    let rows = (base_pos + m_tok) as usize * kvdim as usize;
    let (qd, kd, vd) = (g.upload(q), g.upload(k), g.upload(v));
    let (kc, vc) = (g.alloc_bytes(rows * 2).unwrap(), g.alloc_bytes(rows * 2).unwrap());
    let md = u32s(g, mpos);
    let total = m_tok * (aq + ak + kvdim);
    let enc = g.begin();
    g.dispatch(&enc, "rope_qk_store_m",
               &[(&qd, 0), (&kd, 0), (&vd, 0), (&kc, 0), (&vc, 0), (&md, 0)],
               &[HD as u32, base_pos, BASE.to_bits(), aq, ak, kvdim, m_tok, neox, rd],
               [total.div_ceil(256), 1, 1], [256, 1, 1]).unwrap();
    g.submit(enc).unwrap();
    let mut qo = vec![0f32; q.len()];
    g.read(&qd, &mut qo);
    (qo, read_f16(g, &kc, rows), read_f16(g, &vc, rows))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn mode_name(m: u32) -> &'static str { ["OFF", "MROPE", "IMROPE", "VISION"][m as usize] }

// ---------------- tests ----------------

/// Text-only span: OFF equals `rope_partial_m` bit-for-bit (q and k), MROPE / IMROPE with
/// t==h==w==e equal OFF bit-for-bit, and everything matches the CPU reference.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn text_only_collapses_to_plain_rope() {
    let g = gpu();
    let (base_pos, m_tok) = (37u32, 19u32);
    let kvdim = NKV * HD;
    let q = tvec(m_tok as usize * NQ * HD, 1);
    let k = tvec(m_tok as usize * kvdim, 2);
    let v = tvec(m_tok as usize * kvdim, 3);
    let pos: Vec<[u32; 4]> = (0..m_tok).map(|m| [base_pos + m; 4]).collect();
    let mp = mpos_desc(SECTIONS, &pos);

    let (q_off, k_off, v_off) = gpu_qk_store(&g, &q, &k, &v, base_pos, m_tok, 1, NROT as u32, &[0]);

    // existing kernel: rope_partial_m(v, hd, n_rot, pos_base, base, n_heads, rowdim, M)
    let run_partial = |x: &[f32], heads: usize, rowdim: usize| {
        let d = g.upload(x);
        let total = m_tok * (heads * NROT / 2) as u32;
        let enc = g.begin();
        g.dispatch(&enc, "rope_partial_m", &[(&d, 0)],
                   &[HD as u32, NROT as u32, base_pos, BASE.to_bits(), heads as u32,
                     rowdim as u32, m_tok], [total.div_ceil(256), 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut o = vec![0f32; x.len()];
        g.read(&d, &mut o);
        o
    };
    let q_pm = run_partial(&q, NQ, NQ * HD);
    let k_pm: Vec<f32> = run_partial(&k, NKV, kvdim).iter()
        .map(|x| half::f16::from_f32(*x).to_f32()).collect();
    let k_off_rows = &k_off[base_pos as usize * kvdim..];
    let q_pm_ok = bits_eq(&q_off, &q_pm);
    let k_pm_ok = bits_eq(k_off_rows, &k_pm);
    println!("OFF vs rope_partial_m: q bit-identical {q_pm_ok} (max {:.3e}), k(f16) bit-identical {k_pm_ok} (max {:.3e})",
             max_abs(&q_off, &q_pm), max_abs(k_off_rows, &k_pm));
    assert!(q_pm_ok && k_pm_ok, "OFF must equal rope_partial_m bit-for-bit");

    for mode in [MROPE, IMROPE] {
        let (qm, km, vm) = gpu_qk_store(&g, &q, &k, &v, base_pos, m_tok, (mode << SHIFT) | 1, NROT as u32, &mp);
        let same = bits_eq(&qm, &q_off) && bits_eq(&km, &k_off) && bits_eq(&vm, &v_off);
        println!("text-only {:>6} == OFF bit-for-bit: {same}", mode_name(mode));
        assert!(same, "{} with t==h==w must equal OFF", mode_name(mode));
    }
    for mode in [OFF, MROPE, IMROPE, VISION] {
        for nx in [1u32, 0] {
            let neox = (mode << SHIFT) | nx;
            let mpos: &[u32] = if mode == OFF { &[0] } else { &mp };
            let (qg, kg, vg) = gpu_qk_store(&g, &q, &k, &v, base_pos, m_tok, neox, NROT as u32, mpos);
            let (qr, kr, vr) = ref_qk_store(&q, &k, &v, base_pos, m_tok, neox, NROT as u32, mpos);
            let (eq, ek, ev) = (max_abs(&qg, &qr), max_abs(&kg, &kr), max_abs(&vg, &vr));
            println!("text-only {:>6} neox={nx}: max|err| q {eq:.3e} kcache {ek:.3e} vcache {ev:.3e}", mode_name(mode));
            assert!(eq < 1e-5 && ek < 2e-3 && ev == 0.0);
        }
    }
}

/// text(5) | 64x48 merged image grid | text(6), 3-D positions, all four modes, both pairings.
/// Also asserts the image modes differ from OFF and that the cache rows start at base_pos.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn image_span_3d_positions_all_modes() {
    let g = gpu();
    let pos = positions(5, 64, 48, 6);
    let m_tok = pos.len() as u32;
    let base_pos = 11u32;
    let kvdim = NKV * HD;
    let q = tvec(m_tok as usize * NQ * HD, 4);
    let k = tvec(m_tok as usize * kvdim, 5);
    let v = tvec(m_tok as usize * kvdim, 6);
    let mp = mpos_desc(SECTIONS, &pos);
    let (q_off, _, _) = gpu_qk_store(&g, &q, &k, &v, base_pos, m_tok, 1, NROT as u32, &[0]);
    let mut worst = 0f32;
    for mode in [OFF, MROPE, IMROPE, VISION] {
        for nx in [1u32, 0] {
            let neox = (mode << SHIFT) | nx;
            let mpos: &[u32] = if mode == OFF { &[0] } else { &mp };
            let (qg, kg, vg) = gpu_qk_store(&g, &q, &k, &v, base_pos, m_tok, neox, NROT as u32, mpos);
            let (qr, kr, vr) = ref_qk_store(&q, &k, &v, base_pos, m_tok, neox, NROT as u32, mpos);
            let (eq, ek, ev) = (max_abs(&qg, &qr), max_abs(&kg, &kr), max_abs(&vg, &vr));
            println!("image M={m_tok} {:>6} neox={nx}: max|err| q {eq:.3e} kcache {ek:.3e} vcache {ev:.3e}",
                     mode_name(mode));
            worst = worst.max(eq);
            assert!(eq < 1e-4 && ek < 2e-3 && ev == 0.0);
            // rows below base_pos untouched
            assert!(kg[..base_pos as usize * kvdim].iter().all(|x| *x == 0.0));
            if mode != OFF && nx == 1 {
                assert!(max_abs(&qg, &q_off) > 1e-2, "{} image positions had no effect", mode_name(mode));
            }
        }
    }
    println!("worst q error over all modes: {worst:.3e}");
}

/// `rope_m` (plain NeoX, full head, exponent over hd) vs CPU, and vs `rope_partial_m` with
/// n_rot = hd (same maths: must be bit-identical).
#[test]
#[ignore = "requires NVIDIA GPU"]
fn rope_m_matches_reference() {
    let g = gpu();
    let (base_pos, m_tok) = (5u32, 23u32);
    for heads in [NQ, NKV] {
        let r = heads * HD;
        let x = tvec(m_tok as usize * r, 9 + heads as u32);
        let d = g.upload(&x);
        let total = m_tok * (r / 2) as u32;
        let enc = g.begin();
        g.dispatch(&enc, "rope_m", &[(&d, 0)],
                   &[HD as u32, base_pos, BASE.to_bits(), r as u32, m_tok],
                   [total.div_ceil(256), 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0f32; x.len()];
        g.read(&d, &mut got);
        let mut want = x.clone();
        let hf = HD / 2;
        for m in 0..m_tok as usize {
            for h in 0..heads {
                let b = m * r + h * HD;
                for i in 0..hf {
                    let (n0, n1) = rot(base_pos + m as u32, i as u32, HD as u32, x[b + i], x[b + hf + i]);
                    want[b + i] = n0; want[b + hf + i] = n1;
                }
            }
        }
        let d2 = g.upload(&x);
        let enc = g.begin();
        g.dispatch(&enc, "rope_partial_m", &[(&d2, 0)],
                   &[HD as u32, HD as u32, base_pos, BASE.to_bits(), heads as u32, r as u32, m_tok],
                   [total.div_ceil(256), 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut pm = vec![0f32; x.len()];
        g.read(&d2, &mut pm);
        let e = max_abs(&got, &want);
        println!("rope_m heads={heads}: max|err| vs CPU {e:.3e}, bit-identical to rope_partial_m(n_rot=hd): {}",
                 bits_eq(&got, &pm));
        assert!(e < 1e-5 && bits_eq(&got, &pm));
    }
}
