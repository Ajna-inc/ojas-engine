//! GPU check harnesses + demos (vs CPU references). Not part of the training path.
use ojas_metal::{MBuf, MetalGpu};
use ojas_core::Device as _;
use anyhow::Result;
use metal::MTLSize;


use crate::kit::{g1, g2, Kit};
use crate::moe_ffn::MoeFfn;
use std::collections::HashMap;

/// Gradcheck the MoE routing kernels (t_moe_gate_fwd/bwd) against a CPU reference of the same math.
pub fn moe_gate_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (t, ne, k) = (12usize, 8usize, 2usize);
    let mut rng = 0x1234_5678u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let logits: Vec<f32> = (0..t*ne).map(|_| nxt()).collect();
    let dgate: Vec<f32> = (0..t*ne).map(|_| nxt()).collect();
    let (b_logits, b_dgate) = (gpu.upload(&logits), gpu.upload(&dgate));
    let (b_gates, b_topk, b_dlog) = (gpu.alloc(t*ne), gpu.alloc(t*k), gpu.alloc(t*ne));
    let ic = [t as u32, ne as u32, k as u32];
    let tg = MTLSize::new(256, 1, 1);
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    kit.d(enc, "t_moe_gate_fwd", &[(&b_logits, 0), (&b_gates, 0), (&b_topk, 0)], &ic, g1(t.div_ceil(256)), tg);
    kit.d(enc, "t_moe_gate_bwd", &[(&b_dgate, 0), (&b_gates, 0), (&b_topk, 0), (&b_dlog, 0)], &ic, g1(t.div_ceil(256)), tg);
    enc.end_encoding(); cb.commit(); cb.wait_until_completed();
    let (gates, dlog) = (gpu.read(&b_gates), gpu.read(&b_dlog));
    // CPU reference
    let mut cg = vec![0f32; t*ne];
    let mut cd = vec![0f32; t*ne];
    for ti in 0..t {
        let lg = &logits[ti*ne..ti*ne+ne];
        let mut idx = Vec::new(); let mut taken = vec![false; ne];
        for _ in 0..k {
            let (mut best, mut bi) = (f32::MIN, 0usize);
            for e in 0..ne { if !taken[e] && lg[e] > best { best = lg[e]; bi = e; } }
            taken[bi] = true; idx.push(bi);
        }
        let mx = idx.iter().map(|&e| lg[e]).fold(f32::MIN, f32::max);
        let exps: Vec<f32> = idx.iter().map(|&e| (lg[e]-mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (j, &e) in idx.iter().enumerate() { cg[ti*ne+e] = exps[j]/sum; }
        let s: Vec<f32> = idx.iter().map(|&e| cg[ti*ne+e]).collect();
        let dg: Vec<f32> = idx.iter().map(|&e| dgate[ti*ne+e]).collect();
        let dot: f32 = s.iter().zip(&dg).map(|(a, b)| a*b).sum();
        for (j, &e) in idx.iter().enumerate() { cd[ti*ne+e] = s[j]*(dg[j]-dot); }
    }
    let err = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x-y).abs()).fold(0f32, f32::max);
    let (eg, ed) = (err(&gates, &cg), err(&dlog, &cd));
    tracing::debug!(target: "moe-gate", "T={t} NE={ne} K={k}  gate_fwd max_err {eg:.2e}  gate_bwd max_err {ed:.2e}");
    if eg < 1e-5 && ed < 1e-5 { tracing::info!(target: "ojas", "PASS ✅ MoE routing kernels match the oracle"); }
    else { anyhow::bail!("moe-gate gradcheck FAIL"); }
    Ok(())
}

/// Assemble the masked-dense MoE FFN forward from the SwiGLU/GEMM and routing kernels and check
/// `out` against a CPU f32 reference, covering the expert loop, gated accumulate and shared-expert
/// wiring (fwd_moe). f16 weights give a ~1e-2 tolerance.
pub fn moe_fwd_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (t, d, ne, k, e) = (32usize, 64usize, 8usize, 2usize, 128usize);   // MMA-tile-aligned
    let mut rng = 0xBEEF_1234u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let h2: Vec<f32> = (0..t*d).map(|_| nxt()).collect();
    let wr: Vec<f32> = (0..ne*d).map(|_| nxt()*0.3).collect();          // router (ne,d)
    let mk = |n: usize, s: f32, r: &mut dyn FnMut() -> f32| (0..n).map(|_| r()*s).collect::<Vec<f32>>();
    let wg: Vec<Vec<f32>> = (0..ne).map(|_| mk(e*d, (d as f32).powf(-0.5), &mut nxt)).collect();
    let wu: Vec<Vec<f32>> = (0..ne).map(|_| mk(e*d, (d as f32).powf(-0.5), &mut nxt)).collect();
    let wd: Vec<Vec<f32>> = (0..ne).map(|_| mk(d*e, (e as f32).powf(-0.5), &mut nxt)).collect();
    let (sg, su, sd) = (mk(e*d, (d as f32).powf(-0.5), &mut nxt), mk(e*d, (d as f32).powf(-0.5), &mut nxt), mk(d*e, (e as f32).powf(-0.5), &mut nxt));

    // upload
    let bh2 = gpu.upload(&h2);
    let bwr = gpu.upload(&wr);       // router f32 (small N -> t_gemm_xwT, not the N>=64 MMA)
    let bwg: Vec<MBuf> = wg.iter().map(|w| gpu.upload_f16(w)).collect();
    let bwu: Vec<MBuf> = wu.iter().map(|w| gpu.upload_f16(w)).collect();
    let bwd: Vec<MBuf> = wd.iter().map(|w| gpu.upload_f16(w)).collect();
    let (bsg, bsu, bsd) = (gpu.upload_f16(&sg), gpu.upload_f16(&su), gpu.upload_f16(&sd));
    let (blog, bgates, btopk) = (gpu.alloc(t*ne), gpu.alloc(t*ne), gpu.alloc(t*k));
    let (bgl, bul, bact) = (gpu.alloc(t*e), gpu.alloc(t*e), gpu.alloc(t*e));
    let (boe, bout) = (gpu.alloc(t*d), gpu.alloc(t*d));

    let (du, eu, neu, tu) = (d as u32, e as u32, ne as u32, t as u32);
    let gm = |o: usize| g2(t.div_ceil(32), o.div_ceil(64));
    let tg128 = MTLSize::new(128, 1, 1);
    let tg256 = MTLSize::new(256, 1, 1);
    let el = |n: usize| g1(n.div_ceil(256));
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    // router (small N=ne -> t_gemm_xwT handles N<64) + gate
    kit.d(enc, "t_gemm_xwT", &[(&bh2, 0), (&bwr, 0), (&blog, 0)], &[du, neu, tu], g2(ne.div_ceil(8), t.div_ceil(16)), tg256);
    kit.d(enc, "t_moe_gate_fwd", &[(&blog, 0), (&bgates, 0), (&btopk, 0)], &[tu, neu, k as u32], g1(t.div_ceil(256)), tg256);
    // shared expert -> out
    let swiglu = |enc: &metal::ComputeCommandEncoderRef, wg: &MBuf, wu: &MBuf, wd: &MBuf, o: &MBuf| {
        kit.d(enc, "t_mm_xwT_h", &[(&bh2, 0), (wg, 0), (&bgl, 0)], &[du, eu, tu], gm(e), tg128);
        kit.d(enc, "t_mm_xwT_h", &[(&bh2, 0), (wu, 0), (&bul, 0)], &[du, eu, tu], gm(e), tg128);
        kit.d(enc, "t_swiglu_fwd", &[(&bgl, 0), (&bul, 0), (&bact, 0)], &[(t*e) as u32], el(t*e), tg256);
        kit.d(enc, "t_mm_xwT_h", &[(&bact, 0), (wd, 0), (o, 0)], &[eu, du, tu], gm(d), tg128);
    };
    swiglu(enc, &bsg, &bsu, &bsd, &bout);       // out = shared
    for ei in 0..ne {
        swiglu(enc, &bwg[ei], &bwu[ei], &bwd[ei], &boe);
        kit.d(enc, "t_moe_acc", &[(&bout, 0), (&boe, 0), (&bgates, 0)], &[du, neu, ei as u32, (t*d) as u32], el(t*d), tg256);
    }
    enc.end_encoding(); cb.commit(); cb.wait_until_completed();
    let out = gpu.read(&bout);
    let (glog, ggat) = (gpu.read(&blog), gpu.read(&bgates));

    // ---- CPU f32 reference ----
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let swig = |x: &[f32], wg: &[f32], wu: &[f32], wd: &[f32]| -> Vec<f32> {
        let mut o = vec![0f32; t*d];
        for ti in 0..t {
            let mut act = vec![0f32; e];
            for c in 0..e {
                let (mut a, mut b) = (0f32, 0f32);
                for i in 0..d { a += x[ti*d+i]*wg[c*d+i]; b += x[ti*d+i]*wu[c*d+i]; }
                act[c] = silu(a)*b;
            }
            for j in 0..d { let mut s = 0f32; for c in 0..e { s += act[c]*wd[j*e+c]; } o[ti*d+j] = s; }
        }
        o
    };
    let oshared = swig(&h2, &sg, &su, &sd);
    let mut oref = oshared.clone();
    let (mut clog, mut cgat) = (vec![0f32; t*ne], vec![0f32; t*ne]);
    for ti in 0..t {
        let mut lg = vec![0f32; ne];
        for ee in 0..ne { let mut s = 0f32; for i in 0..d { s += h2[ti*d+i]*wr[ee*d+i]; } lg[ee] = s; clog[ti*ne+ee] = s; }
        let mut idx = Vec::new(); let mut taken = vec![false; ne];
        for _ in 0..k { let (mut bv, mut bi) = (f32::MIN, 0); for ee in 0..ne { if !taken[ee] && lg[ee] > bv { bv = lg[ee]; bi = ee; } } taken[bi] = true; idx.push(bi); }
        let mx = idx.iter().map(|&ee| lg[ee]).fold(f32::MIN, f32::max);
        let ex: Vec<f32> = idx.iter().map(|&ee| (lg[ee]-mx).exp()).collect();
        let sm: f32 = ex.iter().sum();
        for (jj, &ee) in idx.iter().enumerate() { cgat[ti*ne+ee] = ex[jj]/sm; }
    }
    for ti in 0..t {
        for ee in 0..ne {
            let g = cgat[ti*ne+ee];
            if g == 0.0 { continue; }
            let oe = swig(&h2, &wg[ee], &wu[ee], &wd[ee]);
            for j in 0..d { oref[ti*d+j] += g*oe[ti*d+j]; }
        }
    }
    let mxerr = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x-y).abs()).fold(0f32, f32::max);
    tracing::debug!(target: "dbg", "logits_err {:.2e}  gates_err {:.2e}", mxerr(&glog, &clog), mxerr(&ggat, &cgat));
    let err = out.iter().zip(&oref).map(|(a, b)| (a-b).abs()).fold(0f32, f32::max);
    let rel = err / oref.iter().map(|x| x.abs()).fold(0f32, f32::max);
    tracing::debug!(target: "moe-fwd", "T={t} d={d} ne={ne} k={k} E={e}  out max_err {err:.2e}  rel {rel:.2e}");
    if rel < 2e-2 { tracing::info!(target: "ojas", "PASS ✅ MoE forward (router+gate+experts+shared+accumulate) matches ref"); }
    else { anyhow::bail!("moe-fwd check FAIL"); }
    Ok(())
}

/// Assemble the masked-dense MoE FFN backward (expert SwiGLU bwd scaled by gate, shared expert,
/// router gate jacobian) and check d_h2 and d_Wr against a CPU reference.
pub fn moe_bwd_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (t, d, ne, k, e) = (32usize, 64usize, 8usize, 2usize, 128usize);
    let mut rng = 0xF00D_9911u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let mk = |n: usize, s: f32, r: &mut dyn FnMut() -> f32| (0..n).map(|_| r()*s).collect::<Vec<f32>>();
    let sc = (d as f32).powf(-0.5); let se = (e as f32).powf(-0.5);
    let h2 = mk(t*d, 1.0, &mut nxt);
    let dout = mk(t*d, 1.0, &mut nxt);
    let wr = mk(ne*d, 0.3, &mut nxt);
    let wg: Vec<Vec<f32>> = (0..ne).map(|_| mk(e*d, sc, &mut nxt)).collect();
    let wu: Vec<Vec<f32>> = (0..ne).map(|_| mk(e*d, sc, &mut nxt)).collect();
    let wd: Vec<Vec<f32>> = (0..ne).map(|_| mk(d*e, se, &mut nxt)).collect();
    let (sg, su, sd) = (mk(e*d, sc, &mut nxt), mk(e*d, sc, &mut nxt), mk(d*e, se, &mut nxt));

    let bh2 = gpu.upload(&h2); let bdout = gpu.upload(&dout);
    let bwr = gpu.upload(&wr);
    let bwg: Vec<MBuf> = wg.iter().map(|w| gpu.upload_f16(w)).collect();
    let bwu: Vec<MBuf> = wu.iter().map(|w| gpu.upload_f16(w)).collect();
    let bwd: Vec<MBuf> = wd.iter().map(|w| gpu.upload_f16(w)).collect();
    let (bsg, bsu, bsd) = (gpu.upload_f16(&sg), gpu.upload_f16(&su), gpu.upload_f16(&sd));
    let (blog, bgat, btk) = (gpu.alloc(t*ne), gpu.alloc(t*ne), gpu.alloc(t*k));
    // saved forward intermediates
    let gl: Vec<MBuf> = (0..ne).map(|_| gpu.alloc(t*e)).collect();
    let ul: Vec<MBuf> = (0..ne).map(|_| gpu.alloc(t*e)).collect();
    let oe: Vec<MBuf> = (0..ne).map(|_| gpu.alloc(t*d)).collect();
    let (gls, uls) = (gpu.alloc(t*e), gpu.alloc(t*e));
    let bact = gpu.alloc(t*e);
    // backward scratch
    let (dact, dgl, dul, dtmp) = (gpu.alloc(t*e), gpu.alloc(t*e), gpu.alloc(t*e), gpu.alloc(t*d));
    let (dh2, dgate, dlog, dwr) = (gpu.alloc(t*d), gpu.alloc(t*ne), gpu.alloc(t*ne), gpu.alloc(ne*d));

    let (du, eu, neu, tu, ku) = (d as u32, e as u32, ne as u32, t as u32, k as u32);
    let (tg128, tg256) = (MTLSize::new(128, 1, 1), MTLSize::new(256, 1, 1));
    let gf = |o: usize| g2(t.div_ceil(32), o.div_ceil(64));    // fwd/dx: (M/32, out_or_in/64)
    let el = |n: usize| g1(n.div_ceil(256));
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    // ---- forward (save gl/ul/oe) ----
    kit.d(enc, "t_gemm_xwT", &[(&bh2, 0), (&bwr, 0), (&blog, 0)], &[du, neu, tu], g2(ne.div_ceil(8), t.div_ceil(16)), tg256);
    kit.d(enc, "t_moe_gate_fwd", &[(&blog, 0), (&bgat, 0), (&btk, 0)], &[tu, neu, ku], g1(t.div_ceil(256)), tg256);
    let fwd = |enc: &metal::ComputeCommandEncoderRef, wg: &MBuf, wu: &MBuf, wd: &MBuf, gl: &MBuf, ul: &MBuf, o: &MBuf| {
        kit.d(enc, "t_mm_xwT_h", &[(&bh2, 0), (wg, 0), (gl, 0)], &[du, eu, tu], gf(e), tg128);
        kit.d(enc, "t_mm_xwT_h", &[(&bh2, 0), (wu, 0), (ul, 0)], &[du, eu, tu], gf(e), tg128);
        kit.d(enc, "t_swiglu_fwd", &[(gl, 0), (ul, 0), (&bact, 0)], &[(t*e) as u32], el(t*e), tg256);
        kit.d(enc, "t_mm_xwT_h", &[(&bact, 0), (wd, 0), (o, 0)], &[eu, du, tu], gf(d), tg128);
    };
    fwd(enc, &bsg, &bsu, &bsd, &gls, &uls, &dtmp);           // shared -> dtmp (o_shared, unused in bwd d_h2 aside from chain)
    for ei in 0..ne { fwd(enc, &bwg[ei], &bwu[ei], &bwd[ei], &gl[ei], &ul[ei], &oe[ei]); }
    // ---- backward ----
    kit.d(enc, "t_fill", &[(&dh2, 0)], &[(t*d) as u32], el(t*d), tg256);
    for ei in 0..ne { kit.d(enc, "t_moe_dgate", &[(&dgate, 0), (&oe[ei], 0), (&bdout, 0)], &[du, neu, ei as u32, tu], g1(t.div_ceil(256)), tg256); }
    // expert bwd: d_o_e = gate*dout; swiglu bwd; accumulate d_h2
    for ei in 0..ne {
        kit.d(enc, "t_moe_rowscale", &[(&dtmp, 0), (&bdout, 0), (&bgat, 0)], &[du, neu, ei as u32, (t*d) as u32], el(t*d), tg256);
        kit.d(enc, "t_mm_dx_h", &[(&dtmp, 0), (&bwd[ei], 0), (&dact, 0)], &[eu, du, 0u32, tu], gf(e), tg128);
        kit.d(enc, "t_swiglu_bwd", &[(&dact, 0), (&gl[ei], 0), (&ul[ei], 0), (&dgl, 0), (&dul, 0)], &[(t*e) as u32], el(t*e), tg256);
        kit.d(enc, "t_mm_dx_h", &[(&dgl, 0), (&bwg[ei], 0), (&dh2, 0)], &[du, eu, 1u32, tu], gf(d), tg128);
        kit.d(enc, "t_mm_dx_h", &[(&dul, 0), (&bwu[ei], 0), (&dh2, 0)], &[du, eu, 1u32, tu], gf(d), tg128);
    }
    // shared bwd (gate 1)
    kit.d(enc, "t_mm_dx_h", &[(&bdout, 0), (&bsd, 0), (&dact, 0)], &[eu, du, 0u32, tu], gf(e), tg128);
    kit.d(enc, "t_swiglu_bwd", &[(&dact, 0), (&gls, 0), (&uls, 0), (&dgl, 0), (&dul, 0)], &[(t*e) as u32], el(t*e), tg256);
    kit.d(enc, "t_mm_dx_h", &[(&dgl, 0), (&bsg, 0), (&dh2, 0)], &[du, eu, 1u32, tu], gf(d), tg128);
    kit.d(enc, "t_mm_dx_h", &[(&dul, 0), (&bsu, 0), (&dh2, 0)], &[du, eu, 1u32, tu], gf(d), tg128);
    // router bwd: gate jacobian -> d_logits -> d_Wr + d_h2
    kit.d(enc, "t_moe_gate_bwd", &[(&dgate, 0), (&bgat, 0), (&btk, 0), (&dlog, 0)], &[tu, neu, ku], g1(t.div_ceil(256)), tg256);
    kit.d(enc, "t_gemm_dw", &[(&dlog, 0), (&bh2, 0), (&dwr, 0)], &[du, neu, tu], g2(d.div_ceil(1024), ne.div_ceil(8)), tg256);
    kit.d(enc, "t_gemm_dx", &[(&dlog, 0), (&bwr, 0), (&dh2, 0)], &[du, neu, 1u32, tu], g2(d.div_ceil(1024), t.div_ceil(8)), tg256);
    enc.end_encoding(); cb.commit(); cb.wait_until_completed();
    let (gdh2, gdwr) = (gpu.read(&dh2), gpu.read(&dwr));

    // ---- CPU reference ----
    let sig = |x: f32| 1.0/(1.0+(-x).exp());
    let sw_bwd = |wg: &[f32], wu: &[f32], wd: &[f32], dob: &[f32], acc: &mut [f32]| {
        for ti in 0..t {
            let mut dgl = vec![0f32; e]; let mut dul = vec![0f32; e];
            for c in 0..e {
                let (mut a, mut b) = (0f32, 0f32);
                for i in 0..d { a += h2[ti*d+i]*wg[c*d+i]; b += h2[ti*d+i]*wu[c*d+i]; }
                let s = sig(a); let silu = a*s; let dsilu = s*(1.0+a*(1.0-s));
                let mut da = 0f32; for j in 0..d { da += dob[ti*d+j]*wd[j*e+c]; }
                dgl[c] = da*b*dsilu; dul[c] = da*silu;
            }
            for i in 0..d { let mut s = 0f32; for c in 0..e { s += dgl[c]*wg[c*d+i] + dul[c]*wu[c*d+i]; } acc[ti*d+i] += s; }
        }
    };
    let sw_fwd = |wg: &[f32], wu: &[f32], wd: &[f32]| -> Vec<f32> {
        let mut o = vec![0f32; t*d];
        for ti in 0..t { let mut act = vec![0f32; e];
            for c in 0..e { let (mut a, mut b) = (0f32, 0f32); for i in 0..d { a += h2[ti*d+i]*wg[c*d+i]; b += h2[ti*d+i]*wu[c*d+i]; } act[c] = (a*sig(a))*b; }
            for j in 0..d { let mut s = 0f32; for c in 0..e { s += act[c]*wd[j*e+c]; } o[ti*d+j] = s; } }
        o
    };
    let mut cdh2 = vec![0f32; t*d];
    let mut cgat = vec![0f32; t*ne]; let mut ctk: Vec<Vec<usize>> = vec![vec![]; t];
    for ti in 0..t {
        let mut lg = vec![0f32; ne];
        for ee in 0..ne { let mut s = 0f32; for i in 0..d { s += h2[ti*d+i]*wr[ee*d+i]; } lg[ee] = s; }
        let mut idx = Vec::new(); let mut taken = vec![false; ne];
        for _ in 0..k { let (mut bv, mut bi) = (f32::MIN, 0); for ee in 0..ne { if !taken[ee] && lg[ee] > bv { bv = lg[ee]; bi = ee; } } taken[bi] = true; idx.push(bi); }
        let mx = idx.iter().map(|&ee| lg[ee]).fold(f32::MIN, f32::max);
        let ex: Vec<f32> = idx.iter().map(|&ee| (lg[ee]-mx).exp()).collect(); let sm: f32 = ex.iter().sum();
        for (jj, &ee) in idx.iter().enumerate() { cgat[ti*ne+ee] = ex[jj]/sm; }
        ctk[ti] = idx;
    }
    // shared + experts d_h2
    sw_bwd(&sg, &su, &sd, &dout, &mut cdh2);
    for ee in 0..ne {
        let mut dob = vec![0f32; t*d];
        for ti in 0..t { let g = cgat[ti*ne+ee]; for j in 0..d { dob[ti*d+j] = g*dout[ti*d+j]; } }
        if cgat.iter().skip(ee).step_by(ne).any(|&g| g != 0.0) { sw_bwd(&wg[ee], &wu[ee], &wd[ee], &dob, &mut cdh2); }
    }
    // router: d_gate, gate jacobian, d_Wr, d_h2 += d_logits @ Wr
    let mut cdwr = vec![0f32; ne*d];
    for ti in 0..t {
        let mut dg = vec![0f32; ne];
        for ee in 0..ne { let oe = sw_fwd(&wg[ee], &wu[ee], &wd[ee]); let mut s = 0f32; for j in 0..d { s += oe[ti*d+j]*dout[ti*d+j]; } dg[ee] = s; }
        let idx = &ctk[ti];
        let s: Vec<f32> = idx.iter().map(|&ee| cgat[ti*ne+ee]).collect();
        let dgi: Vec<f32> = idx.iter().map(|&ee| dg[ee]).collect();
        let dot: f32 = s.iter().zip(&dgi).map(|(a, b)| a*b).sum();
        for (jj, &ee) in idx.iter().enumerate() {
            let dl = s[jj]*(dgi[jj]-dot);
            for i in 0..d { cdwr[ee*d+i] += dl*h2[ti*d+i]; cdh2[ti*d+i] += dl*wr[ee*d+i]; }
        }
    }
    let mxe = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x-y).abs()).fold(0f32, f32::max);
    let sc2 = |a: &[f32]| a.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let (rh, rw) = (mxe(&gdh2, &cdh2)/sc2(&cdh2), mxe(&gdwr, &cdwr)/sc2(&cdwr));
    tracing::debug!(target: "moe-bwd", "d_h2 rel {rh:.2e}  d_Wr rel {rw:.2e}");
    if rh < 3e-2 && rw < 3e-2 { tracing::info!(target: "ojas", "PASS ✅ MoE backward (d_h2, d_Wr) matches the oracle"); }
    else { anyhow::bail!("moe-bwd check FAIL"); }
    Ok(())
}

/// End-to-end test of the MoeFfn component: forward and backward through the struct, `out` and
/// d_h2 against a CPU reference, and every weight-grad buffer finite.
pub fn moe_layer_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (t, d, ne, k, e) = (32usize, 64usize, 8usize, 2usize, 128usize);
    let moe = MoeFfn::new(gpu, d, e, ne, k, t, 0x51A7_2024);
    let mut rng = 0x0C0F_FEE1u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let h2: Vec<f32> = (0..t*d).map(|_| nxt()).collect();
    let dout: Vec<f32> = (0..t*d).map(|_| nxt()).collect();
    let (bh2, bdout) = (gpu.upload(&h2), gpu.upload(&dout));
    let (bout, bdh2) = (gpu.alloc(t*d), gpu.alloc(t*d));
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    moe.forward(&kit, enc, &bh2, &bout, t);
    moe.backward(&kit, enc, &bh2, &bdout, &bdh2, t);
    enc.end_encoding(); cb.commit(); cb.wait_until_completed();
    let (out, dh2) = (gpu.read(&bout), gpu.read(&bdh2));

    // finite checks on all weight grads
    let finite = |b: &MBuf| gpu.read(b).iter().all(|x| x.is_finite()) && gpu.read(b).iter().any(|&x| x != 0.0);
    let gok = finite(&moe.gwr) && finite(&moe.gsg) && finite(&moe.gsd) && finite(&moe.gwg) && finite(&moe.gwd);

    // CPU reference for out and d_h2. The f32 weights are regenerated from the same seed and
    // order as MoeFfn::new, since the GPU stores the experts as f16.
    let sig = |x: f32| 1.0/(1.0+(-x).exp());
    let mut wrng = 0x51A7_2024u64;
    let mut wn = || { wrng = wrng.wrapping_mul(6364136223846793005).wrapping_add(1); ((wrng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let (scw, sew) = ((d as f32).powf(-0.5), (e as f32).powf(-0.5));
    let gen = |n: usize, s: f32, r: &mut dyn FnMut() -> f32| (0..n).map(|_| r()*s).collect::<Vec<f32>>();
    let wr = gen(ne*d, 0.3, &mut wn);
    let wg: Vec<Vec<f32>> = (0..ne).map(|_| gen(e*d, scw, &mut wn)).collect();
    let wu: Vec<Vec<f32>> = (0..ne).map(|_| gen(e*d, scw, &mut wn)).collect();
    let wd: Vec<Vec<f32>> = (0..ne).map(|_| gen(d*e, sew, &mut wn)).collect();
    let (sg, su, sd) = (gen(e*d, scw, &mut wn), gen(e*d, scw, &mut wn), gen(d*e, sew, &mut wn));
    let sw_fwd = |wg: &[f32], wu: &[f32], wd: &[f32]| -> Vec<f32> { let mut o = vec![0f32; t*d];
        for ti in 0..t { let mut ac = vec![0f32; e]; for c in 0..e { let (mut a, mut b) = (0f32, 0f32); for i in 0..d { a += h2[ti*d+i]*wg[c*d+i]; b += h2[ti*d+i]*wu[c*d+i]; } ac[c] = (a*sig(a))*b; }
            for j in 0..d { let mut s = 0f32; for c in 0..e { s += ac[c]*wd[j*e+c]; } o[ti*d+j] = s; } } o };
    let sw_bwd = |wg: &[f32], wu: &[f32], wd: &[f32], dob: &[f32], acc: &mut [f32]| {
        for ti in 0..t { let mut dgl = vec![0f32; e]; let mut dul = vec![0f32; e];
            for c in 0..e { let (mut a, mut b) = (0f32, 0f32); for i in 0..d { a += h2[ti*d+i]*wg[c*d+i]; b += h2[ti*d+i]*wu[c*d+i]; }
                let s = sig(a); let da = { let mut x = 0f32; for j in 0..d { x += dob[ti*d+j]*wd[j*e+c]; } x }; dgl[c] = da*b*(s*(1.0+a*(1.0-s))); dul[c] = da*(a*s); }
            for i in 0..d { let mut s = 0f32; for c in 0..e { s += dgl[c]*wg[c*d+i] + dul[c]*wu[c*d+i]; } acc[ti*d+i] += s; } } };
    let mut oref = sw_fwd(&sg, &su, &sd);
    let (mut cgat, mut ctk) = (vec![0f32; t*ne], vec![vec![]; t] as Vec<Vec<usize>>);
    for ti in 0..t { let mut lg = vec![0f32; ne]; for ee in 0..ne { let mut s = 0f32; for i in 0..d { s += h2[ti*d+i]*wr[ee*d+i]; } lg[ee] = s; }
        let mut idx = Vec::new(); let mut tk = vec![false; ne];
        for _ in 0..k { let (mut bv, mut bi) = (f32::MIN, 0); for ee in 0..ne { if !tk[ee] && lg[ee] > bv { bv = lg[ee]; bi = ee; } } tk[bi] = true; idx.push(bi); }
        let mx = idx.iter().map(|&ee| lg[ee]).fold(f32::MIN, f32::max); let ex: Vec<f32> = idx.iter().map(|&ee| (lg[ee]-mx).exp()).collect(); let sm: f32 = ex.iter().sum();
        for (jj, &ee) in idx.iter().enumerate() { cgat[ti*ne+ee] = ex[jj]/sm; } ctk[ti] = idx; }
    for ti in 0..t { for ee in 0..ne { let g = cgat[ti*ne+ee]; if g == 0.0 { continue; } let oe = sw_fwd(&wg[ee], &wu[ee], &wd[ee]); for j in 0..d { oref[ti*d+j] += g*oe[ti*d+j]; } } }
    let mut cdh2 = vec![0f32; t*d];
    sw_bwd(&sg, &su, &sd, &dout, &mut cdh2);
    for ee in 0..ne { let mut dob = vec![0f32; t*d]; let mut used = false; for ti in 0..t { let g = cgat[ti*ne+ee]; if g != 0.0 { used = true; } for j in 0..d { dob[ti*d+j] = g*dout[ti*d+j]; } } if used { sw_bwd(&wg[ee], &wu[ee], &wd[ee], &dob, &mut cdh2); } }
    for ti in 0..t { let mut dg = vec![0f32; ne]; for ee in 0..ne { let oe = sw_fwd(&wg[ee], &wu[ee], &wd[ee]); let mut s = 0f32; for j in 0..d { s += oe[ti*d+j]*dout[ti*d+j]; } dg[ee] = s; }
        let idx = &ctk[ti]; let s: Vec<f32> = idx.iter().map(|&ee| cgat[ti*ne+ee]).collect(); let dgi: Vec<f32> = idx.iter().map(|&ee| dg[ee]).collect(); let dot: f32 = s.iter().zip(&dgi).map(|(a, b)| a*b).sum();
        for (jj, &ee) in idx.iter().enumerate() { let dl = s[jj]*(dgi[jj]-dot); for i in 0..d { cdh2[ti*d+i] += dl*wr[ee*d+i]; } } }
    let mxe = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x-y).abs()).fold(0f32, f32::max);
    let sc2 = |a: &[f32]| a.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let (ro, rh) = (mxe(&out, &oref)/sc2(&oref), mxe(&dh2, &cdh2)/sc2(&cdh2));
    tracing::debug!(target: "moe-layer", "out rel {ro:.2e}  d_h2 rel {rh:.2e}  weight_grads_finite {gok}");
    if ro < 2e-2 && rh < 3e-2 && gok { tracing::info!(target: "ojas", "PASS ✅ MoeFfn component wired: fwd+bwd correct, all grads finite"); }
    else { anyhow::bail!("moe-layer check FAIL"); }
    Ok(())
}

/// Gather/scatter MoE forward: route each token to its top-k experts only (contiguous per-expert
/// groups), run each expert's GEMM on just its tokens, scatter back gate-weighted. Checked against
/// masked-dense for identical output, and both timed. Routing bookkeeping on CPU.
pub fn moe_gs_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (d, e, ne, k, tmax) = (2048usize, 1024usize, 32usize, 4usize, 2048usize);
    let moe = MoeFfn::new(gpu, d, e, ne, k, tmax, 42);
    let mut rng = 0xA5A5_1234u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let h2: Vec<f32> = (0..tmax*d).map(|_| nxt()).collect();
    let bh2 = gpu.upload(&h2);
    let (du, eu, neu) = (d as u32, e as u32, ne as u32);
    let (tg128, tg256) = (MTLSize::new(128, 1, 1), MTLSize::new(256, 1, 1));
    let el = |n: usize| g1(n.div_ceil(256));
    let gf = |o: usize, m: usize| g2(m.div_ceil(32), o.div_ceil(64));
    let (blog, bgat, btk) = (gpu.alloc(tmax*ne), gpu.alloc(tmax*ne), gpu.alloc(tmax*k));
    let (bgath, bgl, bul, bact, beo) = (gpu.alloc(tmax*k*d), gpu.alloc(tmax*k*e), gpu.alloc(tmax*k*e), gpu.alloc(tmax*k*e), gpu.alloc(tmax*k*d));
    let (bshared, bgs, bref) = (gpu.alloc(tmax*d), gpu.alloc(tmax*d), gpu.alloc(tmax*d));

    tracing::debug!(target: "moe-gs", "d={d} ne={ne} top-{k} E={e}  (masked-dense vs gather/scatter, fwd)");
    for &t in &[256usize, 1024, 2048] {
        let (tu, ku) = (t as u32, k as u32);
        // masked-dense ref
        { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder(); moe.forward(&kit, enc, &bh2, &bref, t); enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
        let out_ref = gpu.read(&bref)[..t*d].to_vec();
        // route + CPU bookkeeping
        { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
          kit.d(enc, "t_gemm_xwT", &[(&bh2, 0), (&moe.wr, 0), (&blog, 0)], &[du, neu, tu], g2(ne.div_ceil(8), t.div_ceil(16)), tg256);
          kit.d(enc, "t_moe_gate_fwd", &[(&blog, 0), (&bgat, 0), (&btk, 0)], &[tu, neu, ku], g1(t.div_ceil(256)), tg256);
          enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
        let topk: Vec<u32> = unsafe { std::slice::from_raw_parts(btk.buf.contents() as *const u32, t*k).to_vec() };
        let gates = gpu.read(&bgat);
        let mut count = vec![0usize; ne]; for a in 0..t*k { count[topk[a] as usize] += 1; }
        let mut offset = vec![0usize; ne+1]; for x in 0..ne { offset[x+1] = offset[x] + count[x]; }
        let mut fill = vec![0usize; ne];
        let (mut gidx, mut tok2, mut gg) = (vec![0u32; t*k], vec![0u32; t*k], vec![0f32; t*k]);
        for ti in 0..t { for j in 0..k { let eix = topk[ti*k+j] as usize; let pos = offset[eix]+fill[eix]; fill[eix]+=1;
            gidx[pos]=ti as u32; gg[pos]=gates[ti*ne+eix]; tok2[ti*k+j]=pos as u32; } }
        let (bgidx, btok2, bgg) = (gpu.upload_u32(&gidx), gpu.upload_u32(&tok2), gpu.upload(&gg));
        let nr = t*k;
        let gs_run = || {
            let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
            kit.d(enc, "t_mm_xwT_h", &[(&bh2, 0), (&moe.sg, 0), (&bgl, 0)], &[du, eu, tu], gf(e, t), tg128);
            kit.d(enc, "t_mm_xwT_h", &[(&bh2, 0), (&moe.su, 0), (&bul, 0)], &[du, eu, tu], gf(e, t), tg128);
            kit.d(enc, "t_swiglu_fwd", &[(&bgl, 0), (&bul, 0), (&bact, 0)], &[(t*e) as u32], el(t*e), tg256);
            kit.d(enc, "t_mm_xwT_h", &[(&bact, 0), (&moe.sd, 0), (&bshared, 0)], &[eu, du, tu], gf(d, t), tg128);
            kit.d(enc, "t_moe_gather", &[(&bgath, 0), (&bh2, 0), (&bgidx, 0)], &[du, (nr*d) as u32], el(nr*d), tg256);
            for eix in 0..ne { let m = count[eix]; if m == 0 { continue; }
                let (od, oe, wo) = ((offset[eix]*d*4) as u64, (offset[eix]*e*4) as u64, (eix*e*d*2) as u64);
                kit.d(enc, "t_mm_xwT_h", &[(&bgath, od), (&moe.wg, wo), (&bgl, oe)], &[du, eu, m as u32], gf(e, m), tg128);
                kit.d(enc, "t_mm_xwT_h", &[(&bgath, od), (&moe.wu, wo), (&bul, oe)], &[du, eu, m as u32], gf(e, m), tg128);
                kit.d(enc, "t_swiglu_fwd", &[(&bgl, oe), (&bul, oe), (&bact, oe)], &[(m*e) as u32], el(m*e), tg256);
                kit.d(enc, "t_mm_xwT_h", &[(&bact, oe), (&moe.wd, wo), (&beo, od)], &[eu, du, m as u32], gf(d, m), tg128); }
            kit.d(enc, "t_moe_scatter_k", &[(&bgs, 0), (&bshared, 0), (&beo, 0), (&btok2, 0), (&bgg, 0)], &[du, ku, (t*d) as u32], el(t*d), tg256);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        };
        gs_run();
        let out_gs = gpu.read(&bgs)[..t*d].to_vec();
        let mxe = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x-y).abs()).fold(0f32, f32::max);
        let rel = mxe(&out_gs, &out_ref) / out_ref.iter().map(|x| x.abs()).fold(1e-9, f32::max);
        let md_run = || { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder(); moe.forward(&kit, enc, &bh2, &bref, t); enc.end_encoding(); cb.commit(); cb.wait_until_completed(); };
        let iters = 8;
        md_run(); let t0 = std::time::Instant::now(); for _ in 0..iters { md_run(); } let md = t0.elapsed().as_secs_f32()/iters as f32;
        gs_run(); let t1 = std::time::Instant::now(); for _ in 0..iters { gs_run(); } let gsms = t1.elapsed().as_secs_f32()/iters as f32;
        if rel >= 2e-2 { anyhow::bail!("moe-gs FAIL at t={t}: rel {rel:.2e}"); }
        tracing::debug!(target: "ojas", "  t={t:5}  tok/expert={:4}  masked-dense {:5.0} ms  gather/scatter {:5.0} ms  ->  {:.2}x   (rel {rel:.0e})",
                  t*k/ne, md*1e3, gsms*1e3, md/gsms);
    }
    tracing::info!(target: "ojas", "PASS ✅ gather/scatter exact; speedup grows with tokens/expert (batch)");
    Ok(())
}

/// The whole gather/scatter path (MoeFfn::forward_gs/backward_gs, every expert weight grad
/// included) against the masked-dense forward/backward.
pub fn moe_gs_full_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (d, e, ne, k, t) = (512usize, 256usize, 8usize, 2usize, 257usize);   // pilot-scale dims
    let moe = MoeFfn::new(gpu, d, e, ne, k, t, 42);
    let mut rng = 0x9E37_11A3u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let (h2, dout) = ((0..t*d).map(|_| nxt()).collect::<Vec<f32>>(), (0..t*d).map(|_| nxt()).collect::<Vec<f32>>());
    let (bh2, bdout) = (gpu.upload(&h2), gpu.upload(&dout));
    let (bo_md, bdh2_md, bo_gs, bdh2_gs) = (gpu.alloc(t*d), gpu.alloc(t*d), gpu.alloc(t*d), gpu.alloc(t*d));
    // masked-dense reference
    { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
      moe.forward(&kit, enc, &bh2, &bo_md, t); moe.backward(&kit, enc, &bh2, &bdout, &bdh2_md, t);
      enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
    let (o_md, dh2_md) = (gpu.read(&bo_md), gpu.read(&bdh2_md));
    let (gwr_md, gsg_md, gwg0_md, gwu0_md, gwd0_md) = (gpu.read(&moe.gwr), gpu.read(&moe.gsg), gpu.read(&moe.gwg), gpu.read(&moe.gwu), gpu.read(&moe.gwd));   // whole contiguous expert grads
    // gather/scatter path
    moe.forward_gs(&kit, gpu, &bh2, &bo_gs, t);
    moe.backward_gs(&kit, gpu, &bh2, &bdout, &bdh2_gs, t);
    let (o_gs, dh2_gs) = (gpu.read(&bo_gs), gpu.read(&bdh2_gs));
    let (gwr_gs, gsg_gs, gwg0_gs, gwu0_gs, gwd0_gs) = (gpu.read(&moe.gwr), gpu.read(&moe.gsg), gpu.read(&moe.gwg), gpu.read(&moe.gwu), gpu.read(&moe.gwd));
    let rel = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x-y).abs()).fold(0f32, f32::max) / b.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    let (ro, rh) = (rel(&o_gs, &o_md), rel(&dh2_gs, &dh2_md));
    let (rr, rs, rg, ru, rd) = (rel(&gwr_gs, &gwr_md), rel(&gsg_gs, &gsg_md), rel(&gwg0_gs, &gwg0_md), rel(&gwu0_gs, &gwu0_md), rel(&gwd0_gs, &gwd0_md));
    tracing::debug!(target: "moe-gs-full", "d={d} e={e} ne={ne} top{k} t={t}  out {ro:.1e} dh2 {rh:.1e} | dWr {rr:.1e} dWshared {rs:.1e} dWg0 {rg:.1e} dWu0 {ru:.1e} dWd0 {rd:.1e}");
    let mx = [ro, rh, rr, rs, rg, ru, rd].iter().cloned().fold(0f32, f32::max);
    if mx < 2e-2 { tracing::info!(target: "ojas", "PASS ✅ forward_gs/backward_gs (incl. all expert weight grads) match masked-dense — training-ready top-k MoE"); }
    else { anyhow::bail!("moe-gs-full FAIL (max rel {mx:.2e})"); }
    Ok(())
}

/// Backward gather/scatter: mirror the forward permutation for the grad path so the whole step
/// gets the speedup. Checks d_h2 and d_Wr against the masked-dense backward and times both.
pub fn moe_gs_bwd_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (d, e, ne, k, t) = (2048usize, 1024usize, 32usize, 4usize, 512usize);
    let moe = MoeFfn::new(gpu, d, e, ne, k, t, 42);
    let mut rng = 0x7E57_1234u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let h2: Vec<f32> = (0..t*d).map(|_| nxt()).collect();
    let dout: Vec<f32> = (0..t*d).map(|_| nxt()).collect();
    let (bh2, bdout) = (gpu.upload(&h2), gpu.upload(&dout));
    let (du, eu, neu, tu, ku) = (d as u32, e as u32, ne as u32, t as u32, k as u32);
    let (tg128, tg256) = (MTLSize::new(128, 1, 1), MTLSize::new(256, 1, 1));
    let el = |n: usize| g1(n.div_ceil(256));
    let gf = |o: usize, m: usize| g2(m.div_ceil(32), o.div_ceil(64));
    let nr = t*k;

    // ---- masked-dense reference fwd+bwd ----
    let (bout_md, bdh2_md) = (gpu.alloc(t*d), gpu.alloc(t*d));
    { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
      moe.forward(&kit, enc, &bh2, &bout_md, t); moe.backward(&kit, enc, &bh2, &bdout, &bdh2_md, t);
      enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
    let (dh2_ref, dwr_ref) = (gpu.read(&bdh2_md), gpu.read(&moe.gwr));

    // ---- routing + CPU bookkeeping ----
    let (blog, bgat, btk) = (gpu.alloc(t*ne), gpu.alloc(t*ne), gpu.alloc(t*k));
    { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
      kit.d(enc, "t_gemm_xwT", &[(&bh2, 0), (&moe.wr, 0), (&blog, 0)], &[du, neu, tu], g2(ne.div_ceil(8), t.div_ceil(16)), tg256);
      kit.d(enc, "t_moe_gate_fwd", &[(&blog, 0), (&bgat, 0), (&btk, 0)], &[tu, neu, ku], g1(t.div_ceil(256)), tg256);
      enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
    let topk: Vec<u32> = unsafe { std::slice::from_raw_parts(btk.buf.contents() as *const u32, t*k).to_vec() };
    let gates = gpu.read(&bgat);
    let mut count = vec![0usize; ne]; for a in 0..t*k { count[topk[a] as usize] += 1; }
    let mut offset = vec![0usize; ne+1]; for x in 0..ne { offset[x+1] = offset[x] + count[x]; }
    let mut fill = vec![0usize; ne];
    let (mut gidx, mut tok2, mut gg, mut rexp) = (vec![0u32; nr], vec![0u32; nr], vec![0f32; nr], vec![0u32; nr]);
    for ti in 0..t { for j in 0..k { let ex = topk[ti*k+j] as usize; let pos = offset[ex]+fill[ex]; fill[ex]+=1;
        gidx[pos]=ti as u32; gg[pos]=gates[ti*ne+ex]; tok2[ti*k+j]=pos as u32; rexp[pos]=ex as u32; } }
    let (bgidx, btok2, bgg, brexp) = (gpu.upload_u32(&gidx), gpu.upload_u32(&tok2), gpu.upload(&gg), gpu.upload_u32(&rexp));

    // buffers (gathered layout)
    let (bgath, bgl, bul, beo) = (gpu.alloc(nr*d), gpu.alloc(nr*e), gpu.alloc(nr*e), gpu.alloc(nr*d));
    let (bgls, buls, bshared) = (gpu.alloc(t*e), gpu.alloc(t*e), gpu.alloc(t*d));
    let (bact, bdeo, bdgr) = (gpu.alloc(nr*e), gpu.alloc(nr*d), gpu.alloc(nr));
    let (bdact, bdgl, bdul, bdgath) = (gpu.alloc(nr*e), gpu.alloc(nr*e), gpu.alloc(nr*e), gpu.alloc(nr*d));
    let (bdgd, bdlog, bdh2) = (gpu.alloc(t*ne), gpu.alloc(t*ne), gpu.alloc(t*d));

    let gs_fwd = |enc: &metal::ComputeCommandEncoderRef| {
        kit.d(enc, "t_mm_xwT_h", &[(&bh2, 0), (&moe.sg, 0), (&bgls, 0)], &[du, eu, tu], gf(e, t), tg128);
        kit.d(enc, "t_mm_xwT_h", &[(&bh2, 0), (&moe.su, 0), (&buls, 0)], &[du, eu, tu], gf(e, t), tg128);
        kit.d(enc, "t_swiglu_fwd", &[(&bgls, 0), (&buls, 0), (&bact, 0)], &[(t*e) as u32], el(t*e), tg256);
        kit.d(enc, "t_mm_xwT_h", &[(&bact, 0), (&moe.sd, 0), (&bshared, 0)], &[eu, du, tu], gf(d, t), tg128);
        kit.d(enc, "t_moe_gather", &[(&bgath, 0), (&bh2, 0), (&bgidx, 0)], &[du, (nr*d) as u32], el(nr*d), tg256);
        for ex in 0..ne { let m = count[ex]; if m == 0 { continue; }
            let (od, oe, wo) = ((offset[ex]*d*4) as u64, (offset[ex]*e*4) as u64, (ex*e*d*2) as u64);
            kit.d(enc, "t_mm_xwT_h", &[(&bgath, od), (&moe.wg, wo), (&bgl, oe)], &[du, eu, m as u32], gf(e, m), tg128);
            kit.d(enc, "t_mm_xwT_h", &[(&bgath, od), (&moe.wu, wo), (&bul, oe)], &[du, eu, m as u32], gf(e, m), tg128);
            kit.d(enc, "t_swiglu_fwd", &[(&bgl, oe), (&bul, oe), (&bact, oe)], &[(m*e) as u32], el(m*e), tg256);
            kit.d(enc, "t_mm_xwT_h", &[(&bact, oe), (&moe.wd, wo), (&beo, od)], &[eu, du, m as u32], gf(d, m), tg128); }
    };
    let gs_bwd = |enc: &metal::ComputeCommandEncoderRef| {
        kit.d(enc, "t_moe_gather_scaled", &[(&bdeo, 0), (&bdout, 0), (&bgidx, 0), (&bgg, 0)], &[du, (nr*d) as u32], el(nr*d), tg256);
        kit.d(enc, "t_moe_dgate_gs", &[(&bdgr, 0), (&beo, 0), (&bdout, 0), (&bgidx, 0)], &[du, nr as u32], g1(nr.div_ceil(256)), tg256);
        kit.d(enc, "t_fill", &[(&bdh2, 0)], &[(t*d) as u32], el(t*d), tg256);
        // shared bwd (full T)
        kit.d(enc, "t_mm_dx_h", &[(&bdout, 0), (&moe.sd, 0), (&bdact, 0)], &[eu, du, 0u32, tu], gf(e, t), tg128);
        kit.d(enc, "t_swiglu_fwd", &[(&bgls, 0), (&buls, 0), (&bact, 0)], &[(t*e) as u32], el(t*e), tg256);
        kit.d(enc, "t_swiglu_bwd", &[(&bdact, 0), (&bgls, 0), (&buls, 0), (&bdgl, 0), (&bdul, 0)], &[(t*e) as u32], el(t*e), tg256);
        kit.d(enc, "t_mm_dx_h", &[(&bdgl, 0), (&moe.sg, 0), (&bdh2, 0)], &[du, eu, 1u32, tu], gf(d, t), tg128);
        kit.d(enc, "t_mm_dx_h", &[(&bdul, 0), (&moe.su, 0), (&bdh2, 0)], &[du, eu, 1u32, tu], gf(d, t), tg128);
        // expert bwd on gathered slices
        for ex in 0..ne { let m = count[ex]; if m == 0 { continue; }
            let (od, oe, wo) = ((offset[ex]*d*4) as u64, (offset[ex]*e*4) as u64, (ex*e*d*2) as u64);
            kit.d(enc, "t_mm_dx_h", &[(&bdeo, od), (&moe.wd, wo), (&bdact, oe)], &[eu, du, 0u32, m as u32], gf(e, m), tg128);
            kit.d(enc, "t_swiglu_bwd", &[(&bdact, oe), (&bgl, oe), (&bul, oe), (&bdgl, oe), (&bdul, oe)], &[(m*e) as u32], el(m*e), tg256);
            kit.d(enc, "t_mm_dx_h", &[(&bdgl, oe), (&moe.wg, wo), (&bdgath, od)], &[du, eu, 0u32, m as u32], gf(d, m), tg128);
            kit.d(enc, "t_mm_dx_h", &[(&bdul, oe), (&moe.wu, wo), (&bdgath, od)], &[du, eu, 1u32, m as u32], gf(d, m), tg128); }
        kit.d(enc, "t_moe_scatter_dh2", &[(&bdh2, 0), (&bdgath, 0), (&btok2, 0)], &[du, ku, (t*d) as u32], el(t*d), tg256);
        // router
        kit.d(enc, "t_fill", &[(&bdgd, 0)], &[(t*ne) as u32], el(t*ne), tg256);
        kit.d(enc, "t_moe_scatter_dgate", &[(&bdgd, 0), (&bdgr, 0), (&bgidx, 0), (&brexp, 0)], &[neu, nr as u32], g1(nr.div_ceil(256)), tg256);
        kit.d(enc, "t_moe_gate_bwd", &[(&bdgd, 0), (&bgat, 0), (&btk, 0), (&bdlog, 0)], &[tu, neu, ku], g1(t.div_ceil(256)), tg256);
        kit.d(enc, "t_gemm_dw", &[(&bdlog, 0), (&bh2, 0), (&moe.gwr, 0)], &[du, neu, tu], g2(d.div_ceil(1024), ne.div_ceil(8)), tg256);
        kit.d(enc, "t_gemm_dx", &[(&bdlog, 0), (&moe.wr, 0), (&bdh2, 0)], &[du, neu, 1u32, tu], g2(d.div_ceil(1024), t.div_ceil(8)), tg256);
    };
    { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder(); gs_fwd(enc); gs_bwd(enc); enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
    let (dh2_gs, dwr_gs) = (gpu.read(&bdh2), gpu.read(&moe.gwr));
    let mxe = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x-y).abs()).fold(0f32, f32::max);
    let sc2 = |a: &[f32]| a.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    let (rh, rw) = (mxe(&dh2_gs, &dh2_ref)/sc2(&dh2_ref), mxe(&dwr_gs, &dwr_ref)/sc2(&dwr_ref));

    // timing: masked-dense bwd vs gather/scatter bwd (forward already done once)
    let md_bwd = || { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder(); moe.forward(&kit, enc, &bh2, &bout_md, t); moe.backward(&kit, enc, &bh2, &bdout, &bdh2_md, t); enc.end_encoding(); cb.commit(); cb.wait_until_completed(); };
    let gs_step = || { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder(); gs_fwd(enc); gs_bwd(enc); enc.end_encoding(); cb.commit(); cb.wait_until_completed(); };
    let iters = 8;
    md_bwd(); let t0 = std::time::Instant::now(); for _ in 0..iters { md_bwd(); } let md = t0.elapsed().as_secs_f32()/iters as f32;
    gs_step(); let t1 = std::time::Instant::now(); for _ in 0..iters { gs_step(); } let gsms = t1.elapsed().as_secs_f32()/iters as f32;
    tracing::debug!(target: "moe-gs-bwd", "d={d} ne={ne} k={k} E={e} t={t} tok/expert={}  d_h2 rel {rh:.2e}  d_Wr rel {rw:.2e}", t*k/ne);
    tracing::debug!(target: "moe-gs-bwd", "full step (fwd+bwd): masked-dense {:.0} ms  gather/scatter {:.0} ms  ->  {:.2}x", md*1e3, gsms*1e3, md/gsms);
    if rh < 3e-2 && rw < 3e-2 { tracing::info!(target: "ojas", "PASS ✅ backward gather/scatter matches masked-dense; whole step {:.1}x faster", md/gsms); }
    else { anyhow::bail!("moe-gs-bwd FAIL (d_h2 {rh:.2e}, d_Wr {rw:.2e})"); }
    Ok(())
}

/// End-to-end training loop for a MoeFfn block: fit a random target (forward -> loss -> backward
/// -> opt_step), checking the MoE training path closes and the optimizer updates.
pub fn moe_train_demo(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (d, e, ne, k, t) = (512usize, 1024usize, 8usize, 2usize, 256usize);
    let moe = MoeFfn::new(gpu, d, e, ne, k, t, 1);
    let mut rng = 0xD00D_1234u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let h2: Vec<f32> = (0..t*d).map(|_| nxt()).collect();
    let tgt: Vec<f32> = (0..t*d).map(|_| nxt()*0.3).collect();
    let bh2 = gpu.upload(&h2);
    let (bout, bdout, bdh2) = (gpu.alloc(t*d), gpu.alloc(t*d), gpu.alloc(t*d));
    let muon = ojas_core::config::var("OJAS_MUON").is_ok();
    tracing::info!(target: "moe-train", "fit a MoE block to a random target (fwd -> loss -> bwd -> opt)  d={d} ne={ne} top-{k} E={e}  optimizer={}", if muon { "MUON" } else { "AdamW" });
    let (mut loss0, mut lossn) = (0f32, 0f32);
    for step in 0..80u32 {
        { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder(); moe.forward(&kit, enc, &bh2, &bout, t); enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
        let out = gpu.read(&bout);
        let mut loss = 0f32; let mut dout = vec![0f32; t*d];
        let n = (t*d) as f32;
        for i in 0..t*d { let diff = out[i]-tgt[i]; loss += 0.5*diff*diff; dout[i] = diff/n; }
        loss /= n;
        unsafe { std::ptr::copy_nonoverlapping(dout.as_ptr(), bdout.buf.contents() as *mut f32, t*d); }
        { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
          moe.backward(&kit, enc, &bh2, &bdout, &bdh2, t);
          if muon { moe.opt_step_muon(&kit, enc, 2e-3, 0.0, 0.9, step); } else { moe.opt_step(&kit, enc, 1e-3, 0.0, step); }
          enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
        if step == 0 { loss0 = loss; }
        lossn = loss;
        if step % 10 == 0 || step == 79 { tracing::debug!(target: "ojas", "  step {step:3}  loss {loss:.5}"); }
    }
    tracing::info!(target: "moe-train", "loss {loss0:.5} -> {lossn:.5}  ({:.0}% reduction)", 100.0*(1.0-lossn/loss0));
    if lossn < loss0*0.5 { tracing::info!(target: "ojas", "PASS ✅ MoE trains end-to-end — fwd + bwd + 8-bit-AdamW opt_step all wired & working"); }
    else { anyhow::bail!("moe-train did not converge ({loss0:.4} -> {lossn:.4})"); }
    Ok(())
}

/// Fused RMSNorm->GEMM against the separate t_rms_fwd then t_mm_xwT_h: identical output, both
/// timed. The fused path never materializes h2, hiding the norm's memory IO.
pub fn rms_fused_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (m, k, n) = (512usize, 2048usize, 1024usize);
    let mut rng = 0x9E37_79B9u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let x: Vec<f32> = (0..m*k).map(|_| nxt()).collect();
    let nw: Vec<f32> = (0..k).map(|_| 1.0 + nxt()*0.1).collect();
    let w: Vec<f32> = (0..n*k).map(|_| nxt()*(k as f32).powf(-0.5)).collect();
    let (bx, bnw, bw) = (gpu.upload(&x), gpu.upload(&nw), gpu.upload_f16(&w));
    let (bh2, brp, brp2) = (gpu.alloc(m*k), gpu.alloc(m), gpu.alloc(m));
    let (byu, byf) = (gpu.alloc(m*n), gpu.alloc(m*n));
    let (ku, nu, mu) = (k as u32, n as u32, m as u32);
    let (tg128, tg256) = (MTLSize::new(128, 1, 1), MTLSize::new(256, 1, 1));
    let unfused = || { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
        kit.d(enc, "t_rms_fwd", &[(&bx, 0), (&bnw, 0), (&bh2, 0), (&brp, 0)], &[ku, mu], g1(m.div_ceil(8)), tg256);
        kit.d(enc, "t_mm_xwT_h", &[(&bh2, 0), (&bw, 0), (&byu, 0)], &[ku, nu, mu], g2(m.div_ceil(32), n.div_ceil(64)), tg128);
        enc.end_encoding(); cb.commit(); cb.wait_until_completed(); };
    let fused = || { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
        kit.d(enc, "t_rmsrp", &[(&bx, 0), (&brp2, 0)], &[ku, mu], g1(m.div_ceil(8)), tg256);
        kit.d(enc, "t_mm_rmsnorm_xwT", &[(&bx, 0), (&bw, 0), (&byf, 0), (&bnw, 0), (&brp2, 0)], &[ku, nu, mu], g2(m.div_ceil(32), n.div_ceil(64)), tg128);
        enc.end_encoding(); cb.commit(); cb.wait_until_completed(); };
    unfused(); fused();
    let (yu, yf) = (gpu.read(&byu), gpu.read(&byf));
    let rel = yu.iter().zip(&yf).map(|(a, b)| (a-b).abs()).fold(0f32, f32::max) / yu.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    let it = 30;
    unfused(); let t0 = std::time::Instant::now(); for _ in 0..it { unfused(); } let uf = t0.elapsed().as_secs_f32()/it as f32*1e3;
    fused(); let t1 = std::time::Instant::now(); for _ in 0..it { fused(); } let ff = t1.elapsed().as_secs_f32()/it as f32*1e3;
    tracing::debug!(target: "rms-fused", "M={m} K={k} N={n}  out rel {rel:.2e}  separate {uf:.3} ms  fused {ff:.3} ms  ->  {:.2}x", uf/ff);
    if rel < 1e-2 { tracing::info!(target: "ojas", "PASS ✅ fused RMSNorm->GEMM exact, {:.2}x faster (no h2 materialization)", uf/ff); }
    else { anyhow::bail!("rms-fused FAIL (rel {rel:.2e})"); }
    Ok(())
}

/// GPU-side MoE routing (histogram + prefix sum + atomic scatter), which removes the CPU roundtrip
/// in gather/scatter. Checks the GPU-built maps are self-consistent and the counts match the CPU's.
pub fn moe_route_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (d, ne, k, t) = (2048usize, 32usize, 4usize, 2048usize);
    let mut rng = 0x5207_E123u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let h2: Vec<f32> = (0..t*d).map(|_| nxt()).collect();
    let wr: Vec<f32> = (0..ne*d).map(|_| nxt()*0.3).collect();
    let (bh2, bwr) = (gpu.upload(&h2), gpu.upload(&wr));
    let (blog, bgat, btk) = (gpu.alloc(t*ne), gpu.alloc(t*ne), gpu.alloc(t*k));
    let na = t*k;
    let (bcount, boff, bfill) = (gpu.alloc(ne), gpu.alloc(ne+1), gpu.alloc(ne));
    let (bgidx, btok2, bgg, brexp) = (gpu.alloc(na), gpu.alloc(na), gpu.alloc(na), gpu.alloc(na));
    let (tg256, du, neu, ku, tu) = (MTLSize::new(256, 1, 1), d as u32, ne as u32, k as u32, t as u32);
    let el = |n: usize| g1(n.div_ceil(256));
    let routed = || {
        let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
        kit.d(enc, "t_gemm_xwT", &[(&bh2, 0), (&bwr, 0), (&blog, 0)], &[du, neu, tu], g2(ne.div_ceil(8), t.div_ceil(16)), tg256);
        kit.d(enc, "t_moe_gate_fwd", &[(&blog, 0), (&bgat, 0), (&btk, 0)], &[tu, neu, ku], g1(t.div_ceil(256)), tg256);
        kit.d(enc, "t_fill", &[(&bcount, 0)], &[neu], el(ne), tg256);
        kit.d(enc, "t_moe_route_count", &[(&bcount, 0), (&btk, 0)], &[na as u32], el(na), tg256);
        kit.d(enc, "t_moe_route_offset", &[(&bcount, 0), (&boff, 0)], &[neu], g1(1), tg256);
        kit.d(enc, "t_fill", &[(&bfill, 0)], &[neu], el(ne), tg256);
        kit.d(enc, "t_moe_route_scatter", &[(&btk, 0), (&bgat, 0), (&boff, 0), (&bfill, 0), (&bgidx, 0), (&btok2, 0), (&bgg, 0), (&brexp, 0)], &[neu, ku, tu], el(na), tg256);
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
    };
    routed();
    let ru = |b: &MBuf, n: usize| unsafe { std::slice::from_raw_parts(b.buf.contents() as *const u32, n).to_vec() };
    let (topk, gates) = (ru(&btk, na), gpu.read(&bgat));
    let (gidx, tok2, rexp, count) = (ru(&bgidx, na), ru(&btok2, na), ru(&brexp, na), ru(&bcount, ne));
    let gg = gpu.read(&bgg);
    // CPU counts
    let mut ccount = vec![0u32; ne]; for a in 0..na { ccount[topk[a] as usize] += 1; }
    let counts_ok = count == ccount;
    // self-consistency: every assignment maps back correctly
    let mut consistent = true;
    for a in 0..na {
        let (tt, e) = (a/k, topk[a]);
        let pos = tok2[a] as usize;
        if gidx[pos] as usize != tt || rexp[pos] != e || (gg[pos] - gates[tt*ne + e as usize]).abs() > 1e-5 { consistent = false; break; }
    }
    let it = 20;
    let t0 = std::time::Instant::now(); for _ in 0..it { routed(); } let ms = t0.elapsed().as_secs_f32()/it as f32*1e3;
    tracing::debug!(target: "moe-route", "T={t} ne={ne} k={k}  counts_match_CPU={counts_ok}  self_consistent={consistent}  {ms:.2} ms (all-GPU, no CPU roundtrip)");
    if counts_ok && consistent { tracing::info!(target: "ojas", "PASS ✅ GPU-side routing correct — CPU bookkeeping roundtrip eliminated"); }
    else { anyhow::bail!("moe-route FAIL"); }
    Ok(())
}

/// Flash Attention forward (online softmax) against naive softmax attention. It uses O(T)
/// attention memory instead of O(T^2), never materializing the T x T score matrix.
pub fn flash_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (t, hd, h) = (384usize, 128usize, 4usize);
    let mut rng = 0xF1A5_9911u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let mut mk = |n: usize| (0..n).map(|_| nxt()*0.3).collect::<Vec<f32>>();   // [T,H,HD]
    let (q, k, v) = (mk(t*h*hd), mk(t*h*hd), mk(t*h*hd));
    let (bq, bk, bv, bo) = (gpu.upload(&q), gpu.upload(&k), gpu.upload(&v), gpu.alloc(t*h*hd));
    let run = || { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
        kit.d(enc, "t_flash_attn_fwd", &[(&bq, 0), (&bk, 0), (&bv, 0), (&bo, 0)], &[t as u32, hd as u32, h as u32], g1((h*t).div_ceil(256)), MTLSize::new(256, 1, 1));
        enc.end_encoding(); cb.commit(); cb.wait_until_completed(); };
    run();
    let o = gpu.read(&bo);
    // CPU reference: naive softmax attention, bidirectional
    let scale = 1.0/(hd as f32).sqrt();
    let mut oref = vec![0f32; t*h*hd];
    for hh in 0..h { for qi in 0..t {
        let qb = &q[(qi*h+hh)*hd..][..hd];
        let mut s = vec![0f32; t]; let mut mx = f32::MIN;
        for kj in 0..t { let kb = &k[(kj*h+hh)*hd..][..hd]; let d: f32 = qb.iter().zip(kb).map(|(a, b)| a*b).sum(); s[kj] = d*scale; mx = mx.max(s[kj]); }
        let mut den = 0f32; for kj in 0..t { s[kj] = (s[kj]-mx).exp(); den += s[kj]; }
        for kj in 0..t { let vb = &v[(kj*h+hh)*hd..][..hd]; let p = s[kj]/den; for i in 0..hd { oref[(qi*h+hh)*hd+i] += p*vb[i]; } }
    } }
    let rel = o.iter().zip(&oref).map(|(a, b)| (a-b).abs()).fold(0f32, f32::max) / oref.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    let it = 20;
    run(); let t0 = std::time::Instant::now(); for _ in 0..it { run(); } let ms = t0.elapsed().as_secs_f32()/it as f32*1e3;
    tracing::debug!(target: "flash", "T={t} HD={hd} H={h}  rel err vs naive {rel:.2e}  {ms:.2} ms/call  (attn mem O(T)={} KB not O(T^2)={} KB)", t*hd*4/1024, t*t*4/1024);
    if rel < 1e-3 { tracing::info!(target: "ojas", "PASS ✅ Flash Attention forward correct (online softmax); O(T) memory, no T x T matrix"); }
    else { anyhow::bail!("flash-check FAIL (rel {rel:.2e})"); }
    Ok(())
}

/// MMA Flash Attention forward (simdgroup_matrix QK^T + P@V, online softmax) against both naive
/// attention and the single-thread flash path.
pub fn flash_mma_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (t, hd, h) = (256usize, 128usize, 4usize);
    let mut rng = 0x1DEA_7777u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let mut mk = |n: usize| (0..n).map(|_| nxt()*0.3).collect::<Vec<f32>>();
    let (q, k, v) = (mk(t*h*hd), mk(t*h*hd), mk(t*h*hd));
    let (bq, bk, bv, bo, bo2) = (gpu.upload(&q), gpu.upload(&k), gpu.upload(&v), gpu.alloc(t*h*hd), gpu.alloc(t*h*hd));
    let bl = gpu.alloc(t*h);
    let c = [t as u32, hd as u32, h as u32, 0u32];  // W=0 = full attention
    let mma = || { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
        kit.d(enc, "t_flash_mma_fwd", &[(&bq, 0), (&bk, 0), (&bv, 0), (&bo, 0), (&bl, 0)], &c, g2(t.div_ceil(8), h), MTLSize::new(32, 1, 1));
        enc.end_encoding(); cb.commit(); cb.wait_until_completed(); };
    let simple = || { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
        kit.d(enc, "t_flash_attn_fwd", &[(&bq, 0), (&bk, 0), (&bv, 0), (&bo2, 0)], &c, g1((h*t).div_ceil(256)), MTLSize::new(256, 1, 1));
        enc.end_encoding(); cb.commit(); cb.wait_until_completed(); };
    mma(); simple();
    let (o, o_ref) = (gpu.read(&bo), gpu.read(&bo2));   // simple flash already validated vs naive
    let rel = o.iter().zip(&o_ref).map(|(a, b)| (a-b).abs()).fold(0f32, f32::max) / o_ref.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    let it = 30;
    mma(); let t0 = std::time::Instant::now(); for _ in 0..it { mma(); } let mm = t0.elapsed().as_secs_f32()/it as f32*1e3;
    simple(); let t1 = std::time::Instant::now(); for _ in 0..it { simple(); } let sp = t1.elapsed().as_secs_f32()/it as f32*1e3;
    tracing::debug!(target: "flash-mma", "T={t} HD={hd} H={h}  rel vs simple {rel:.2e}  MMA {mm:.2} ms  simple {sp:.2} ms  ->  {:.2}x", sp/mm);
    if rel < 2e-3 { tracing::info!(target: "ojas", "PASS ✅ MMA Flash Attention forward correct, {:.2}x vs 1-thread flash", sp/mm); }
    else { anyhow::bail!("flash-mma FAIL (rel {rel:.2e})"); }
    Ok(())
}

/// MMA Flash Attention backward (dQ, dK, dV) against a naive backward.
pub fn flash_bwd_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (t, hd, h) = (256usize, 128usize, 4usize);
    let mut rng = 0xBAC4_2024u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let mut mk = |n: usize| (0..n).map(|_| nxt()*0.3).collect::<Vec<f32>>();
    let (q, k, v, dobuf) = (mk(t*h*hd), mk(t*h*hd), mk(t*h*hd), mk(t*h*hd));
    let (bq, bk, bv, bdo) = (gpu.upload(&q), gpu.upload(&k), gpu.upload(&v), gpu.upload(&dobuf));
    let (bo, bl, bd) = (gpu.alloc(t*h*hd), gpu.alloc(t*h), gpu.alloc(t*h));
    let (bdq, bdk, bdv) = (gpu.alloc(t*h*hd), gpu.alloc(t*h*hd), gpu.alloc(t*h*hd));
    let w = 64usize;   // LOCAL window (block-aligned): query-block attends key-blocks within W bytes
    let c = [t as u32, hd as u32, h as u32, w as u32];
    let att = |qi: usize, kj: usize| { let (a, b) = ((qi/8*8) as i64, (kj/8*8) as i64); (a-b).abs() as usize <= w };
    { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
      kit.d(enc, "t_flash_mma_fwd", &[(&bq, 0), (&bk, 0), (&bv, 0), (&bo, 0), (&bl, 0)], &c, g2(t.div_ceil(8), h), MTLSize::new(32, 1, 1));
      kit.d(enc, "t_flash_drow", &[(&bdo, 0), (&bo, 0), (&bd, 0)], &[hd as u32, (t*h) as u32], g1((t*h).div_ceil(256)), MTLSize::new(256, 1, 1));
      kit.d(enc, "t_flash_mma_dq", &[(&bq, 0), (&bk, 0), (&bv, 0), (&bdo, 0), (&bl, 0), (&bd, 0), (&bdq, 0)], &c, g2(t.div_ceil(8), h), MTLSize::new(32, 1, 1));
      kit.d(enc, "t_flash_mma_dkv", &[(&bq, 0), (&bk, 0), (&bv, 0), (&bdo, 0), (&bl, 0), (&bd, 0), (&bdk, 0), (&bdv, 0)], &c, g2(t.div_ceil(8), h), MTLSize::new(32, 1, 1));
      enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
    let (gdq, gdk, gdv) = (gpu.read(&bdq), gpu.read(&bdk), gpu.read(&bdv));
    // naive backward reference, windowed: softmax and grads only over key blocks in the window
    let scale = 1.0/(hd as f32).sqrt();
    let (mut rdq, mut rdk, mut rdv) = (vec![0f32; t*h*hd], vec![0f32; t*h*hd], vec![0f32; t*h*hd]);
    for hh in 0..h {
        let idx = |i: usize, d: usize| (i*h+hh)*hd + d;
        let mut p = vec![vec![0f32; t]; t];
        for qi in 0..t {
            let mut mx = f32::MIN; let mut s = vec![0f32; t];
            for kj in 0..t { if !att(qi,kj) { continue; } let dd: f32 = (0..hd).map(|d| q[idx(qi,d)]*k[idx(kj,d)]).sum(); s[kj] = dd*scale; mx = mx.max(s[kj]); }
            let mut den = 0f32; for kj in 0..t { if !att(qi,kj) { continue; } s[kj] = (s[kj]-mx).exp(); den += s[kj]; }
            for kj in 0..t { if att(qi,kj) { p[qi][kj] = s[kj]/den; } }
        }
        let mut dvec = vec![0f32; t];
        for qi in 0..t { for d in 0..hd { let mut o = 0f32; for kj in 0..t { o += p[qi][kj]*v[idx(kj,d)]; } dvec[qi] += dobuf[idx(qi,d)]*o; } }
        for kj in 0..t { for d in 0..hd { let mut s = 0f32; for qi in 0..t { s += p[qi][kj]*dobuf[idx(qi,d)]; } rdv[idx(kj,d)] = s; } }
        for qi in 0..t { for kj in 0..t {
            let dp: f32 = (0..hd).map(|d| dobuf[idx(qi,d)]*v[idx(kj,d)]).sum();
            let ds = p[qi][kj]*(dp - dvec[qi]);
            for d in 0..hd { rdq[idx(qi,d)] += scale*ds*k[idx(kj,d)]; rdk[idx(kj,d)] += scale*ds*q[idx(qi,d)]; }
        } }
    }
    let rel = |g: &[f32], r: &[f32]| g.iter().zip(r).map(|(a, b)| (a-b).abs()).fold(0f32, f32::max) / r.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    let (rq, rk, rv) = (rel(&gdq, &rdq), rel(&gdk, &rdk), rel(&gdv, &rdv));
    tracing::debug!(target: "flash-bwd", "T={t} HD={hd} H={h}  dQ rel {rq:.2e}  dK rel {rk:.2e}  dV rel {rv:.2e}");
    if rq < 3e-3 && rk < 3e-3 && rv < 3e-3 { tracing::info!(target: "ojas", "PASS ✅ MMA Flash Attention BACKWARD correct — full flash (fwd+bwd) done"); }
    else { anyhow::bail!("flash-bwd FAIL (dQ {rq:.2e} dK {rk:.2e} dV {rv:.2e})"); }
    Ok(())
}

/// Segment scatter-mean pool (t_pool_fwd/bwd) against a CPU scatter-mean reference. Random pid
/// with empty patches, to exercise the count = 0 path.
pub fn pool_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (t, d, kmax) = (64usize, 48usize, 20usize);
    let kp = kmax + 1;
    let mut rng = 0x9111_2233u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let z: Vec<f32> = (0..t*d).map(|_| nxt()).collect();
    let bos: Vec<f32> = (0..d).map(|_| nxt()).collect();
    let pid: Vec<u32> = (0..t).map(|_| 1 + ((((nxt()*0.5+0.5)*kmax as f32) as usize) % kmax) as u32).collect();
    let mut count = vec![0u32; kp];
    for &p in &pid { count[p as usize] += 1; }
    let (bz, bpid, bcount, bbos) = (gpu.upload(&z), gpu.upload_u32(&pid), gpu.upload_u32(&count), gpu.upload(&bos));
    let bhp = gpu.alloc(kp*d);
    let (du, tu, kpu) = (d as u32, t as u32, kp as u32);
    let tg = MTLSize::new(256, 1, 1);
    { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
      kit.d(enc, "t_pool_fwd", &[(&bhp, 0), (&bz, 0), (&bpid, 0), (&bcount, 0), (&bbos, 0)], &[du, tu, kpu], g1((kp*d).div_ceil(256)), tg);
      enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
    let ghp = gpu.read(&bhp);
    // CPU reference forward
    let mut rhp = vec![0f32; kp*d];
    for i in 0..d { rhp[i] = bos[i]; }
    for k in 1..kp { let c = (count[k].max(1)) as f32;
        for i in 0..d { let mut acc = 0f32; for tt in 0..t { if pid[tt] as usize == k { acc += z[tt*d+i]; } } rhp[k*d+i] = acc/c; } }
    let relf = ghp.iter().zip(&rhp).map(|(a, b)| (a-b).abs()).fold(0f32, f32::max) / rhp.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    // backward
    let dhp: Vec<f32> = (0..kp*d).map(|_| nxt()).collect();
    let bdhp = gpu.upload(&dhp); let bdz = gpu.alloc(t*d);
    { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
      kit.d(enc, "t_pool_bwd", &[(&bdz, 0), (&bdhp, 0), (&bpid, 0), (&bcount, 0)], &[du, tu], g1((t*d).div_ceil(256)), tg);
      enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
    let gdz = gpu.read(&bdz);
    let mut rdz = vec![0f32; t*d];
    for tt in 0..t { let k = pid[tt] as usize; let c = (count[k].max(1)) as f32; for i in 0..d { rdz[tt*d+i] = dhp[k*d+i]/c; } }
    let relb = gdz.iter().zip(&rdz).map(|(a, b)| (a-b).abs()).fold(0f32, f32::max) / rdz.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    let empty = count[1..].iter().filter(|&&c| c == 0).count();
    tracing::debug!(target: "pool", "T={t} D={d} Kmax={kmax}  fwd rel {relf:.2e}  bwd(dz) rel {relb:.2e}  (empty patches {empty})");
    if relf < 1e-3 && relb < 1e-3 { tracing::info!(target: "ojas", "PASS ✅ segment scatter-mean pool fwd+bwd exact vs pilot ref"); }
    else { anyhow::bail!("pool FAIL (fwd {relf:.2e} bwd {relb:.2e})"); }
    Ok(())
}

/// Byte->patch cross-attention (t_xattn_fwd + t_flash_drow + t_xattn_dq/dkv) against a CPU
/// masked-MHA reference (band mask: query t attends patch kp iff kp == 0 || kp < pid[t]).
pub fn xattn_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (t, hd, h, kmax) = (48usize, 32usize, 4usize, 16usize);
    let kp = kmax + 1;
    let mut rng = 0x5AFE_1CE7u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let mut mk = |n: usize| (0..n).map(|_| nxt()*0.3).collect::<Vec<f32>>();
    let (q, kb, vb, dob) = (mk(t*h*hd), mk(kp*h*hd), mk(kp*h*hd), mk(t*h*hd));
    let pid: Vec<u32> = (0..t).map(|_| 1 + ((((nxt()*0.5+0.5)*kmax as f32) as usize) % kmax) as u32).collect();
    let (bq, bk, bv, bdo, bpid) = (gpu.upload(&q), gpu.upload(&kb), gpu.upload(&vb), gpu.upload(&dob), gpu.upload_u32(&pid));
    let (bo, bl, bdr) = (gpu.alloc(t*h*hd), gpu.alloc(t*h), gpu.alloc(t*h));
    let (bdq, bdk, bdv) = (gpu.alloc(t*h*hd), gpu.alloc(kp*h*hd), gpu.alloc(kp*h*hd));
    let (tu, kpu, hdu, hu) = (t as u32, kp as u32, hd as u32, h as u32);
    let tg = MTLSize::new(256, 1, 1);
    { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
      kit.d(enc, "t_xattn_fwd", &[(&bq, 0), (&bk, 0), (&bv, 0), (&bo, 0), (&bl, 0), (&bpid, 0)], &[tu, kpu, hdu, hu], g1((h*t).div_ceil(256)), tg);
      kit.d(enc, "t_flash_drow", &[(&bdo, 0), (&bo, 0), (&bdr, 0)], &[hdu, (t*h) as u32], g1((t*h).div_ceil(256)), tg);
      kit.d(enc, "t_xattn_dq", &[(&bq, 0), (&bk, 0), (&bv, 0), (&bdo, 0), (&bl, 0), (&bdr, 0), (&bdq, 0), (&bpid, 0)], &[tu, kpu, hdu, hu], g1((h*t).div_ceil(256)), tg);
      kit.d(enc, "t_xattn_dkv", &[(&bq, 0), (&bk, 0), (&bv, 0), (&bdo, 0), (&bl, 0), (&bdr, 0), (&bdk, 0), (&bdv, 0), (&bpid, 0)], &[tu, kpu, hdu, hu], g1((h*kp).div_ceil(256)), tg);
      enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
    let (go, gdq, gdk, gdv) = (gpu.read(&bo), gpu.read(&bdq), gpu.read(&bdk), gpu.read(&bdv));
    // CPU reference (masked MHA fwd + bwd)
    let scale = 1.0/(hd as f32).sqrt();
    let (mut ro, mut rdq, mut rdk, mut rdv) = (vec![0f32; t*h*hd], vec![0f32; t*h*hd], vec![0f32; kp*h*hd], vec![0f32; kp*h*hd]);
    for head in 0..h {
        let iq = |qi: usize, dd: usize| (qi*h+head)*hd + dd;
        let ip = |kk: usize, dd: usize| (kk*h+head)*hd + dd;
        for qi in 0..t {
            let pt = pid[qi] as usize;
            let ks: Vec<usize> = (0..kp).filter(|&x| x == 0 || x < pt).collect();
            let mut sc = vec![0f32; ks.len()]; let mut mx = f32::MIN;
            for (j, &kk) in ks.iter().enumerate() { let mut dd = 0f32; for d in 0..hd { dd += q[iq(qi,d)]*kb[ip(kk,d)]; } sc[j] = dd*scale; mx = mx.max(sc[j]); }
            let mut den = 0f32; let mut p = vec![0f32; ks.len()];
            for j in 0..ks.len() { p[j] = (sc[j]-mx).exp(); den += p[j]; }
            for j in 0..ks.len() { p[j] /= den; }
            let mut ovec = vec![0f32; hd];
            for (j, &kk) in ks.iter().enumerate() { for d in 0..hd { ovec[d] += p[j]*vb[ip(kk,d)]; } }
            for d in 0..hd { ro[iq(qi,d)] = ovec[d]; }
            let mut dvec = 0f32; for d in 0..hd { dvec += dob[iq(qi,d)]*ovec[d]; }
            for (j, &kk) in ks.iter().enumerate() {
                let mut dp = 0f32; for d in 0..hd { dp += dob[iq(qi,d)]*vb[ip(kk,d)]; }
                let ds = p[j]*(dp - dvec);
                for d in 0..hd { rdv[ip(kk,d)] += p[j]*dob[iq(qi,d)]; rdq[iq(qi,d)] += scale*ds*kb[ip(kk,d)]; rdk[ip(kk,d)] += scale*ds*q[iq(qi,d)]; }
            }
        }
    }
    let rel = |g: &[f32], r: &[f32]| g.iter().zip(r).map(|(a, b)| (a-b).abs()).fold(0f32, f32::max) / r.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    let (ro_, rq, rk, rv) = (rel(&go, &ro), rel(&gdq, &rdq), rel(&gdk, &rdk), rel(&gdv, &rdv));
    tracing::debug!(target: "xattn", "T={t} KP={kp} HD={hd} H={h}  fwd(O) rel {ro_:.2e}  dQ {rq:.2e}  dK {rk:.2e}  dV {rv:.2e}");
    if ro_ < 1e-5 && rq < 3e-3 && rk < 3e-3 && rv < 3e-3 { tracing::info!(target: "ojas", "PASS ✅ byte->patch cross-attention fwd+bwd exact vs pilot ref"); }
    else { anyhow::bail!("xattn FAIL (O {ro_:.2e} dQ {rq:.2e} dK {rk:.2e} dV {rv:.2e})"); }
    Ok(())
}

/// Front-end topology: pool -> cross-attn chained, full forward and backward, checking the two
/// bridges compose with correct gradient flow. Covers the two shared-tensor accumulations the real
/// model has: z feeds both the cross-attn query and the pool input (dz = dQ + pool_bwd(dhp)), and
/// hp is both K and V (dhp = dK + dV). Gradchecked against a CPU reference.
pub fn frontend_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (t, hd, h, kmax) = (32usize, 16usize, 4usize, 12usize);
    let d = h*hd; let kp = kmax + 1;
    let mut rng = 0xF00D_5EEDu64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let z: Vec<f32> = (0..t*d).map(|_| nxt()*0.3).collect();       // byte-encoder output stand-in [T,D]
    let bos: Vec<f32> = (0..d).map(|_| nxt()*0.3).collect();
    let dobuf: Vec<f32> = (0..t*d).map(|_| nxt()*0.3).collect();    // upstream grad on O
    let pid: Vec<u32> = (0..t).map(|_| 1 + ((((nxt()*0.5+0.5)*kmax as f32) as usize) % kmax) as u32).collect();
    let mut count = vec![0u32; kp]; for &p in &pid { count[p as usize] += 1; }
    let (bz, bbos, bdo, bpid, bcount) = (gpu.upload(&z), gpu.upload(&bos), gpu.upload(&dobuf), gpu.upload_u32(&pid), gpu.upload_u32(&count));
    let (bhp, bo, bl, bdr) = (gpu.alloc(kp*d), gpu.alloc(t*d), gpu.alloc(t*h), gpu.alloc(t*h));
    let (bdq, bdk, bdv) = (gpu.alloc(t*d), gpu.alloc(kp*d), gpu.alloc(kp*d));
    let (bdhp, bdzp, bdz) = (gpu.alloc(kp*d), gpu.alloc(t*d), gpu.alloc(t*d));
    let (du, tu, kpu, hdu, hu) = (d as u32, t as u32, kp as u32, hd as u32, h as u32);
    let tg = MTLSize::new(256, 1, 1); let el = |n: usize| g1(n.div_ceil(256));
    { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
      // forward: hp = pool(z); O = xattn(Q=z, K=V=hp)
      kit.d(enc, "t_pool_fwd", &[(&bhp, 0), (&bz, 0), (&bpid, 0), (&bcount, 0), (&bbos, 0)], &[du, tu, kpu], el(kp*d), tg);
      kit.d(enc, "t_xattn_fwd", &[(&bz, 0), (&bhp, 0), (&bhp, 0), (&bo, 0), (&bl, 0), (&bpid, 0)], &[tu, kpu, hdu, hu], el(h*t), tg);
      // backward: dO -> dQ (=dz from query) + dK,dV (-> dhp = dK+dV) -> pool_bwd -> dz_pool; dz = dQ + dz_pool
      kit.d(enc, "t_flash_drow", &[(&bdo, 0), (&bo, 0), (&bdr, 0)], &[hdu, (t*h) as u32], el(t*h), tg);
      kit.d(enc, "t_xattn_dq", &[(&bz, 0), (&bhp, 0), (&bhp, 0), (&bdo, 0), (&bl, 0), (&bdr, 0), (&bdq, 0), (&bpid, 0)], &[tu, kpu, hdu, hu], el(h*t), tg);
      kit.d(enc, "t_xattn_dkv", &[(&bz, 0), (&bhp, 0), (&bhp, 0), (&bdo, 0), (&bl, 0), (&bdr, 0), (&bdk, 0), (&bdv, 0), (&bpid, 0)], &[tu, kpu, hdu, hu], el(h*kp), tg);
      kit.d(enc, "t_copy", &[(&bdhp, 0), (&bdk, 0)], &[(kp*d) as u32], el(kp*d), tg);
      kit.d(enc, "t_add", &[(&bdhp, 0), (&bdv, 0)], &[(kp*d) as u32], el(kp*d), tg);   // dhp = dK + dV
      kit.d(enc, "t_pool_bwd", &[(&bdzp, 0), (&bdhp, 0), (&bpid, 0), (&bcount, 0)], &[du, tu], el(t*d), tg);
      kit.d(enc, "t_copy", &[(&bdz, 0), (&bdq, 0)], &[(t*d) as u32], el(t*d), tg);
      kit.d(enc, "t_add", &[(&bdz, 0), (&bdzp, 0)], &[(t*d) as u32], el(t*d), tg);      // dz = dQ + dz_pool
      enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
    let (go, gdz) = (gpu.read(&bo), gpu.read(&bdz));
    // ---- CPU reference: same composed pipeline ----
    let scale = 1.0/(hd as f32).sqrt();
    let mut hp = vec![0f32; kp*d];
    for i in 0..d { hp[i] = bos[i]; }
    for k in 1..kp { let c = (count[k].max(1)) as f32; for i in 0..d { let mut acc = 0f32; for tt in 0..t { if pid[tt] as usize == k { acc += z[tt*d+i]; } } hp[k*d+i] = acc/c; } }
    let (mut ro, mut rdz) = (vec![0f32; t*d], vec![0f32; t*d]);
    let mut dhp = vec![0f32; kp*d];
    for head in 0..h {
        let iq = |qi: usize, dd: usize| (qi*h+head)*hd + dd;   // z/O row [T,H,HD]; d = head*hd+dd
        let ip = |kk: usize, dd: usize| (kk*h+head)*hd + dd;
        for qi in 0..t {
            let pt = pid[qi] as usize;
            let ks: Vec<usize> = (0..kp).filter(|&x| x == 0 || x < pt).collect();
            let mut sc = vec![0f32; ks.len()]; let mut mx = f32::MIN;
            for (j, &kk) in ks.iter().enumerate() { let mut dd = 0f32; for e in 0..hd { dd += z[iq(qi,e)]*hp[ip(kk,e)]; } sc[j] = dd*scale; mx = mx.max(sc[j]); }
            let mut den = 0f32; let mut p = vec![0f32; ks.len()];
            for j in 0..ks.len() { p[j] = (sc[j]-mx).exp(); den += p[j]; }
            for j in 0..ks.len() { p[j] /= den; }
            let mut ovec = vec![0f32; hd];
            for (j, &kk) in ks.iter().enumerate() { for e in 0..hd { ovec[e] += p[j]*hp[ip(kk,e)]; } }
            for e in 0..hd { ro[iq(qi,e)] = ovec[e]; }
            let mut dvec = 0f32; for e in 0..hd { dvec += dobuf[iq(qi,e)]*ovec[e]; }
            for (j, &kk) in ks.iter().enumerate() {
                let mut dp = 0f32; for e in 0..hd { dp += dobuf[iq(qi,e)]*hp[ip(kk,e)]; }
                let ds = p[j]*(dp - dvec);
                for e in 0..hd {
                    rdz[iq(qi,e)] += scale*ds*hp[ip(kk,e)];       // dQ contribution to dz
                    dhp[ip(kk,e)] += p[j]*dobuf[iq(qi,e)] + scale*ds*z[iq(qi,e)];   // dV + dK into dhp
                }
            }
        }
    }
    for tt in 0..t { let k = pid[tt] as usize; let c = (count[k].max(1)) as f32; for i in 0..d { rdz[tt*d+i] += dhp[k*d+i]/c; } }
    let rel = |g: &[f32], r: &[f32]| g.iter().zip(r).map(|(a, b)| (a-b).abs()).fold(0f32, f32::max) / r.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    let (ro_, rz) = (rel(&go, &ro), rel(&gdz, &rdz));
    tracing::debug!(target: "frontend", "T={t} D={d} KP={kp} H={h}  fwd(O) rel {ro_:.2e}  dz(composed) rel {rz:.2e}");
    if ro_ < 1e-5 && rz < 3e-3 { tracing::info!(target: "ojas", "PASS ✅ pool->xattn topology composes: correct fwd + gradient flow (dz=dQ+pool_bwd(dK+dV))"); }
    else { anyhow::bail!("frontend FAIL (O {ro_:.2e} dz {rz:.2e})"); }
    Ok(())
}

/// End-to-end byte block-diffusion training loop on the assembled tokenizer-free front-end
/// (byte-embed -> pool -> cross-attn -> byte-head): full forward, backward and AdamW on structured
/// text. The loss must drop.
pub fn frontend_train_demo(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (t, hd, h) = (128usize, 16usize, 8usize);
    let d = h*hd; let embv = 257usize; let v = 257usize; let mask_tok = 256u32;   // 0..255 bytes + MASK
    let kmax = t; let kp = kmax + 1;                                               // fixed KMAX arena (upper bound)
    // corpus: structured text (spaces and newlines drive patch boundaries), repeated so it is
    // learnable
    let base = b"the quick brown fox jumps over the lazy dog.\nfn main() { let x = compute(a, b); println!(\"{x}\"); }\nagentic diffusion over bytes: pool then cross attend patches back to bytes.\n";
    let mut corpus = Vec::new(); while corpus.len() < 8000 { corpus.extend_from_slice(base); }
    let mut rng = 0x1357_9BDFu64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); (rng >> 33) as f32 / (1u64 << 31) as f32 };  // U[0,1)
    let mut sn = || nxt()*2.0 - 1.0;
    // params
    let emb0: Vec<f32> = (0..embv*d).map(|_| sn()*0.02).collect();
    let head0: Vec<f32> = (0..v*d).map(|_| sn()*(d as f32).powf(-0.5)).collect();
    let bos0: Vec<f32> = (0..d).map(|_| sn()*0.02).collect();
    let (bemb, bhead, bbos) = (gpu.upload_f16(&emb0), gpu.upload(&head0), gpu.upload(&bos0));
    // opt states: emb f16 -> 8-bit; head/bos f32
    let ne_ = embv*d;
    let (emb_mh, emb_vq, emb_vs) = (gpu.alloc(ne_.div_ceil(2)), gpu.alloc(ne_.div_ceil(4)), gpu.alloc(ne_.div_ceil(256)));
    let (head_m, head_v) = (gpu.alloc(v*d), gpu.alloc(v*d));
    let (bos_m, bos_v) = (gpu.alloc(d), gpu.alloc(d));
    for z in [&emb_mh, &emb_vq, &emb_vs, &head_m, &head_v, &bos_m, &bos_v] {
        unsafe { std::ptr::write_bytes(z.buf.contents() as *mut u8, 0, z.len*4); }
    }
    // buffers
    let (btok, btgt) = (gpu.alloc(t), gpu.alloc(t));
    let (bz, bhp, bo, bl, bdr) = (gpu.alloc(t*d), gpu.alloc(kp*d), gpu.alloc(t*d), gpu.alloc(t*h), gpu.alloc(t*h));
    let blog = gpu.alloc(t*v);
    let (bdlog, bdo, bdq, bdk, bdv) = (gpu.alloc(t*v), gpu.alloc(t*d), gpu.alloc(t*d), gpu.alloc(kp*d), gpu.alloc(kp*d));
    let (bdhp, bdzp, bdz, bdemb, bdhead) = (gpu.alloc(kp*d), gpu.alloc(t*d), gpu.alloc(t*d), gpu.alloc(embv*d), gpu.alloc(v*d));
    let bpid = gpu.alloc(t); let bcount = gpu.alloc(kp);
    let (du, tu, kpu, vu, hdu, hu) = (d as u32, t as u32, kp as u32, v as u32, hd as u32, h as u32);
    let tg = MTLSize::new(256, 1, 1); let el = |n: usize| g1(n.div_ceil(256));
    let (b1, b2, wd, lr) = (0.9f32, 0.999f32, 0.0f32, 3e-3f32);
    let (mut first, mut last) = (0f32, 0f32);
    let steps = 400usize;
    for step in 0..steps {
        // sample window + build pid (boundary after space/newline), count, diffusion mask
        let off = (nxt()*(corpus.len()-t-1) as f32) as usize;
        let win = &corpus[off..off+t];
        let mut pid = vec![0u32; t]; let mut pc = 0u32;
        for i in 0..t { if i == 0 || win[i-1] == b' ' || win[i-1] == b'\n' { pc += 1; } pid[i] = pc.min(kmax as u32); }
        let mut count = vec![0u32; kp]; for &p in &pid { count[p as usize] += 1; }
        let frac = 0.15 + nxt()*0.85;
        let mut masked = vec![false; t]; let mut nmask = 0usize;
        for i in 0..t { if nxt() < frac { masked[i] = true; nmask += 1; } }
        if nmask == 0 { masked[0] = true; nmask = 1; }
        let tok: Vec<f32> = (0..t).map(|i| if masked[i] { mask_tok as f32 } else { win[i] as f32 }).collect();
        unsafe {
            let tp = btok.buf.contents() as *mut f32; let gp = btgt.buf.contents() as *mut f32;
            for i in 0..t { *tp.add(i) = tok[i]; *gp.add(i) = win[i] as f32; }
            std::ptr::copy_nonoverlapping(pid.as_ptr(), bpid.buf.contents() as *mut u32, t);
            std::ptr::copy_nonoverlapping(count.as_ptr(), bcount.buf.contents() as *mut u32, kp);
        }
        // ---------- forward ----------
        { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
          kit.d(enc, "t_embed_fwd", &[(&btok, 0), (&bemb, 0), (&bz, 0)], &[du, (t*d) as u32], el(t*d), tg);
          kit.d(enc, "t_pool_fwd", &[(&bhp, 0), (&bz, 0), (&bpid, 0), (&bcount, 0), (&bbos, 0)], &[du, tu, kpu], el(kp*d), tg);
          kit.d(enc, "t_xattn_fwd", &[(&bz, 0), (&bhp, 0), (&bhp, 0), (&bo, 0), (&bl, 0), (&bpid, 0)], &[tu, kpu, hdu, hu], el(h*t), tg);
          kit.d(enc, "t_gemm_xwT", &[(&bo, 0), (&bhead, 0), (&blog, 0)], &[du, vu, tu], g2(v.div_ceil(8), t.div_ceil(16)), tg);
          enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
        // ---------- CE + dlogits on masked positions (CPU) ----------
        let logits = gpu.read(&blog);
        let mut dlog = vec![0f32; t*v]; let mut loss = 0f32;
        for i in 0..t {
            if !masked[i] { continue; }
            let row = &logits[i*v..i*v+v];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let mut den = 0f32; let mut p = vec![0f32; v];
            for j in 0..v { p[j] = (row[j]-mx).exp(); den += p[j]; }
            for j in 0..v { p[j] /= den; }
            let y = win[i] as usize;
            loss += -(p[y].max(1e-9)).ln();
            for j in 0..v { dlog[i*v+j] = (p[j] - if j == y { 1.0 } else { 0.0 }) / nmask as f32; }
        }
        loss /= nmask as f32;
        if step == 0 { first = loss; } last = loss;
        unsafe { std::ptr::copy_nonoverlapping(dlog.as_ptr(), bdlog.buf.contents() as *mut u8 as *mut f32, t*v); }
        // ---------- backward + AdamW ----------
        let bc1 = 1.0/(1.0 - b1.powi(step as i32 + 1)); let bc2 = 1.0/(1.0 - b2.powi(step as i32 + 1));
        let fc = [lr, b1, b2, bc1, bc2, wd];
        { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
          // head: dO = dlog @ head ; dhead = dlog^T @ O
          kit.d(enc, "t_gemm_dx", &[(&bdlog, 0), (&bhead, 0), (&bdo, 0)], &[du, vu, 0, tu], g2(d.div_ceil(1024), t.div_ceil(8)), tg);
          kit.d(enc, "t_gemm_dw", &[(&bdlog, 0), (&bo, 0), (&bdhead, 0)], &[du, vu, tu], g2(d.div_ceil(1024), v.div_ceil(8)), tg);
          // cross-attn backward
          kit.d(enc, "t_flash_drow", &[(&bdo, 0), (&bo, 0), (&bdr, 0)], &[hdu, (t*h) as u32], el(t*h), tg);
          kit.d(enc, "t_xattn_dq", &[(&bz, 0), (&bhp, 0), (&bhp, 0), (&bdo, 0), (&bl, 0), (&bdr, 0), (&bdq, 0), (&bpid, 0)], &[tu, kpu, hdu, hu], el(h*t), tg);
          kit.d(enc, "t_xattn_dkv", &[(&bz, 0), (&bhp, 0), (&bhp, 0), (&bdo, 0), (&bl, 0), (&bdr, 0), (&bdk, 0), (&bdv, 0), (&bpid, 0)], &[tu, kpu, hdu, hu], el(h*kp), tg);
          kit.d(enc, "t_copy", &[(&bdhp, 0), (&bdk, 0)], &[(kp*d) as u32], el(kp*d), tg);
          kit.d(enc, "t_add", &[(&bdhp, 0), (&bdv, 0)], &[(kp*d) as u32], el(kp*d), tg);      // dhp = dK+dV
          kit.d(enc, "t_pool_bwd", &[(&bdzp, 0), (&bdhp, 0), (&bpid, 0), (&bcount, 0)], &[du, tu], el(t*d), tg);
          kit.d(enc, "t_copy", &[(&bdz, 0), (&bdq, 0)], &[(t*d) as u32], el(t*d), tg);
          kit.d(enc, "t_add", &[(&bdz, 0), (&bdzp, 0)], &[(t*d) as u32], el(t*d), tg);        // dz = dQ + dz_pool
          kit.d(enc, "t_embed_bwd", &[(&btok, 0), (&bdz, 0), (&bdemb, 0)], &[du, tu], g1(embv), tg);
          // AdamW updates: emb (8-bit), head (f32), bos (f32, grad = dhp[patch0])
          kit.df(enc, "t_adamw_8h", &[(&bemb, 0), (&bdemb, 0), (&emb_mh, 0), (&emb_vq, 0), (&emb_vs, 0)], &fc, &[(embv*d) as u32, step as u32], g1((embv*d).div_ceil(2048)), tg);
          kit.df(enc, "t_adamw", &[(&bhead, 0), (&bdhead, 0), (&head_m, 0), (&head_v, 0)], &fc, &[(v*d) as u32], g1((v*d).div_ceil(256)), tg);
          kit.df(enc, "t_adamw", &[(&bbos, 0), (&bdhp, 0), (&bos_m, 0), (&bos_v, 0)], &fc, &[d as u32], g1(d.div_ceil(256)), tg);
          enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
        if step % 50 == 0 || step == steps-1 { tracing::debug!(target: "ojas", "step {step:3}  diff_loss {loss:.4}"); }
    }
    tracing::info!(target: "frontend-train", "byte block-diffusion on assembled front-end: {first:.4} -> {last:.4}  ({:.0}% drop)", (1.0-last/first)*100.0);
    if last < first*0.6 { tracing::info!(target: "ojas", "PASS ✅ assembled tokenizer-free front-end TRAINS end-to-end in Metal (embed->pool->xattn->head, block-diffusion + AdamW)"); }
    else { anyhow::bail!("frontend-train FAIL (loss {first:.3} -> {last:.3}, insufficient drop)"); }
    Ok(())
}

/// Grouped MoE GEMM (t_mm_grp_xwT: one launch over an expert tile map) against the per-expert
/// GEMM loop: identical output, both timed.
pub fn moe_grouped_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (d, ee, ne) = (2048usize, 1024usize, 32usize);   // K=d, N=E
    let nr = 8192usize;                                   // gathered rows (~256/expert)
    let mut rng = 0xC0DE_2468u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let gath: Vec<f32> = (0..nr*d).map(|_| nxt()*0.1).collect();
    let wall: Vec<f32> = (0..ne*ee*d).map(|_| nxt()*(d as f32).powf(-0.5)).collect();
    let bgath = gpu.upload(&gath);
    let bwall = gpu.upload_f16(&wall);
    let (bref, bgrp) = (gpu.alloc(nr*ee), gpu.alloc(nr*ee));
    // random token->expert assignment -> counts, offsets, tile map
    let mut count = vec![0usize; ne];
    let asn: Vec<usize> = (0..nr).map(|_| ((nxt()*0.5+0.5)*ne as f32) as usize % ne).collect();
    for &a in &asn { count[a] += 1; }
    let mut offset = vec![0usize; ne+1]; for x in 0..ne { offset[x+1] = offset[x] + count[x]; }
    let (mut texp, mut trow0, mut tmend) = (Vec::new(), Vec::new(), Vec::new());
    for eix in 0..ne { let tiles = count[eix].div_ceil(32); for local in 0..tiles {
        texp.push(eix as u32); trow0.push((offset[eix] + local*32) as u32); tmend.push((offset[eix] + count[eix]) as u32); } }
    let nmt = texp.len();
    let (btexp, btrow0, btmend) = (gpu.upload_u32(&texp), gpu.upload_u32(&trow0), gpu.upload_u32(&tmend));
    let tg128 = MTLSize::new(128, 1, 1);

    let loop_run = || {   // per-expert GEMM loop (current gather/scatter)
        let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
        for eix in 0..ne { let m = count[eix]; if m == 0 { continue; }
            kit.d(enc, "t_mm_xwT_h", &[(&bgath, (offset[eix]*d*4) as u64), (&bwall, (eix*ee*d*2) as u64), (&bref, (offset[eix]*ee*4) as u64)],
                  &[d as u32, ee as u32, m as u32], g2(m.div_ceil(32), ee.div_ceil(64)), tg128); }
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
    };
    let grp_run = || {   // grouped: ONE launch
        let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
        kit.d(enc, "t_mm_grp_xwT", &[(&bgath, 0), (&bwall, 0), (&bgrp, 0), (&btexp, 0), (&btrow0, 0), (&btmend, 0)],
              &[d as u32, ee as u32], g2(nmt, ee.div_ceil(64)), tg128);
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
    };
    loop_run(); grp_run();
    let (rf, gr) = (gpu.read(&bref), gpu.read(&bgrp));
    let rel = rf.iter().zip(&gr).map(|(a, b)| (a-b).abs()).fold(0f32, f32::max) / rf.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    let it = 12;
    loop_run(); let t0 = std::time::Instant::now(); for _ in 0..it { loop_run(); } let lp = t0.elapsed().as_secs_f32()/it as f32;
    grp_run(); let t1 = std::time::Instant::now(); for _ in 0..it { grp_run(); } let gp = t1.elapsed().as_secs_f32()/it as f32;
    tracing::debug!(target: "moe-grouped", "nr={nr} experts={ne} K={d} N={ee}  {nmt} tiles  out rel {rel:.2e}");
    tracing::debug!(target: "moe-grouped", "per-expert loop ({ne} launches) {:.1} ms  ->  grouped (1 launch) {:.1} ms  ->  {:.2}x", lp*1e3, gp*1e3, lp/gp);
    if rel < 1e-2 { tracing::info!(target: "ojas", "PASS ✅ grouped GEMM exact, {:.2}x faster than the per-expert loop", lp/gp); }
    else { anyhow::bail!("moe-grouped FAIL (rel {rel:.2e})"); }
    Ok(())
}

/// Muon optimizer core: Newton-Schulz orthogonalization of a gradient matrix (5 iterations,
/// coefficients 3.4445 / -4.7750 / 2.0315) against a CPU reference. The momentum and scaled update
/// around it are standard; Muon takes ~2x fewer steps than AdamW.
pub fn muon_check(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (m, n, steps) = (256usize, 512usize, 5usize);   // wide (m<=n), MMA-aligned
    let (a, b, c) = (3.4445f32, -4.7750f32, 2.0315f32);
    let mut rng = 0x3141_5926u64;
    let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
    let g: Vec<f32> = (0..m*n).map(|_| nxt()).collect();

    // ---- CPU reference (no transpose opt; matches the Metal path) ----
    let fnorm = (g.iter().map(|x| x*x).sum::<f32>()).sqrt() + 1e-7;
    let mut xr: Vec<f32> = g.iter().map(|v| v/fnorm).collect();     // [m,n]
    let matmul = |p: &[f32], q: &[f32], pr: usize, pc: usize, qc: usize| -> Vec<f32> {   // p[pr,pc]@q[pc,qc]
        let mut o = vec![0f32; pr*qc];
        for i in 0..pr { for j in 0..qc { let mut s = 0f32; for k in 0..pc { s += p[i*pc+k]*q[k*qc+j]; } o[i*qc+j] = s; } } o };
    let transpose = |p: &[f32], r: usize, cc: usize| -> Vec<f32> { let mut o = vec![0f32; r*cc]; for i in 0..r { for j in 0..cc { o[j*r+i] = p[i*cc+j]; } } o };
    for _ in 0..steps {
        let xt = transpose(&xr, m, n);
        let aa = matmul(&xr, &xt, m, n, m);                        // A = X X^T [m,m]
        let a2 = matmul(&aa, &aa, m, m, m);                        // A@A
        let bb: Vec<f32> = (0..m*m).map(|i| b*aa[i] + c*a2[i]).collect();
        let bx = matmul(&bb, &xr, m, m, n);                        // B@X
        for i in 0..m*n { xr[i] = a*xr[i] + bx[i]; }
    }

    // ---- Metal ----
    let bg = gpu.upload(&g);
    let (bx, ba, ba2, bbb, bbx) = (gpu.alloc(m*n), gpu.alloc(m*m), gpu.alloc(m*m), gpu.alloc(m*m), gpu.alloc(m*n));
    let (mu, nu) = (m as u32, n as u32);
    let tg256 = MTLSize::new(256, 1, 1);
    let el = |x: usize| g1(x.div_ceil(256));
    let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
    kit.df(enc, "t_scale", &[(&bx, 0), (&bg, 0)], &[1.0/fnorm], &[(m*n) as u32], el(m*n), tg256);
    for _ in 0..steps {
        kit.d(enc, "t_gemm_xwT", &[(&bx, 0), (&bx, 0), (&ba, 0)], &[nu, mu, mu], g2(m.div_ceil(8), m.div_ceil(16)), tg256);       // A = X X^T
        kit.d(enc, "t_gemm_dx", &[(&ba, 0), (&ba, 0), (&ba2, 0)], &[mu, mu, 0u32, mu], g2(m.div_ceil(1024), m.div_ceil(8)), tg256); // A@A
        kit.df(enc, "t_lincomb2", &[(&bbb, 0), (&ba, 0), (&ba2, 0)], &[b, c], &[(m*m) as u32], el(m*m), tg256);                     // B = bA+cAA
        kit.d(enc, "t_gemm_dx", &[(&bbb, 0), (&bx, 0), (&bbx, 0)], &[nu, mu, 0u32, mu], g2(n.div_ceil(1024), m.div_ceil(8)), tg256);// B@X
        kit.df(enc, "t_lincomb2", &[(&bx, 0), (&bx, 0), (&bbx, 0)], &[a, 1.0], &[(m*n) as u32], el(m*n), tg256);                    // X = aX+BX
    }
    enc.end_encoding(); cb.commit(); cb.wait_until_completed();
    let o = gpu.read(&bx);

    let rel = o.iter().zip(&xr).map(|(x, y)| (x-y).abs()).fold(0f32, f32::max) / xr.iter().map(|x| x.abs()).fold(1e-9, f32::max);
    // orthogonality: singular values of an orthogonalized matrix are ~1, so ||O||_F^2 ~ min(m,n)
    let onorm2: f32 = o.iter().map(|x| x*x).sum();
    tracing::debug!(target: "muon", "Newton-Schulz {steps} iters, {m}x{n}  rel err vs ref {rel:.2e}  ||O||_F^2 {onorm2:.1} (~min(m,n)={})", m.min(n));
    if rel < 1e-3 { tracing::info!(target: "ojas", "PASS ✅ Muon orthogonalization matches reference — core of the ~2x-fewer-steps optimizer"); }
    else { anyhow::bail!("muon-check FAIL (rel {rel:.2e})"); }
    Ok(())
}

/// Forward and backward timing of one MoE FFN layer at full 6B dims (d = 2048, 32 experts,
/// E = 1024), extrapolated to the 28-layer 6B model step. Masked-dense path, so every expert is
/// computed.
pub fn moe_6b_perf(gpu: &MetalGpu) -> Result<()> {
    let kit = Kit::new(gpu)?;
    let (d, e, ne, k, t, layers) = (2048usize, 1024usize, 32usize, 4usize, 256usize, 28usize);
    tracing::info!(target: "6b", "building one MoE layer: d={d} experts={ne} top-{k} E={e} tokens={t} ...");
    let t0 = std::time::Instant::now();
    let moe = MoeFfn::new(gpu, d, e, ne, k, t, 42);
    let wbytes = (ne*3*e*d + 3*e*d) * 2;                       // f16 weights, one layer
    tracing::info!(target: "6b", "layer built in {:.1}s  (weights {:.0} MB/layer, {:.1} GB x{layers} layers)",
              t0.elapsed().as_secs_f32(), wbytes as f32/1e6, (wbytes*layers) as f32/1e9);
    let (bh2, bout, bdout, bdh2) = (gpu.alloc(t*d), gpu.alloc(t*d), gpu.alloc(t*d), gpu.alloc(t*d));
    let run = || {
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        moe.forward(&kit, enc, &bh2, &bout, t);
        moe.backward(&kit, enc, &bh2, &bdout, &bdh2, t);
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
    };
    run();                                                     // warm up
    let iters = 8;
    let t1 = std::time::Instant::now();
    for _ in 0..iters { run(); }
    let per_layer = t1.elapsed().as_secs_f32() / iters as f32;
    let per_step = per_layer * layers as f32;                  // MoE FFN across all layers
    tracing::debug!(target: "6b", "MoE fwd+bwd: {:.0} ms/layer  ->  {:.2} s/step (28 layers, MoE-only)", per_layer*1e3, per_step);
    tracing::debug!(target: "6b", "~{:.0} tok/s (MoE-only, masked-dense); attention/GDN adds ~10-25%", t as f32/per_step);
    let ratio = ne as f32 / (k as f32 + 1.0);
    tracing::debug!(target: "6b", "masked-dense computes all {ne} experts; gather/scatter (top-{k}+shared) would be ~{ratio:.0}x faster -> ~{:.2} s/step", per_step/ratio);
    Ok(())
}

/// Full GDN decoder-layer forward and backward on Metal (f32, HF layout), weights keyed by
/// stripped HF names. Returns "out" and the grads as host vectors for the gradcheck comparator.
pub fn gdn_layer_check(gpu: &MetalGpu, w: &HashMap<String, (Vec<usize>, Vec<f32>)>,
                       x: &[f32], dy: &[f32], t: usize) -> Result<HashMap<String, Vec<f32>>> {
    let (d, hk, hv, s, ffn) = (4096usize, 16usize, 32usize, 128usize, 12288usize);
    let (di, c) = (hv * s, 2 * hk * s + hv * s);
    let kit = Kit::new(gpu)?;
    let up = |n: &str| -> MBuf { gpu.upload(&w[n].1) };
    let up1p = |n: &str| -> MBuf {
        gpu.upload(&w[n].1.iter().map(|v| v + 1.0).collect::<Vec<f32>>())
    };
    let ln = up1p("input_layernorm.weight");
    let pln = up1p("post_attention_layernorm.weight");
    let wqkv = up("in_proj_qkv.weight");
    let wz = up("in_proj_z.weight");
    let wa = up("in_proj_a.weight");
    let wb = up("in_proj_b.weight");
    let dt = up("dt_bias");
    let alog = up("A_log");
    let cw = up("conv1d.weight");
    let nw = up("norm.weight");
    let wout = up("out_proj.weight");
    let wg = up("mlp.gate_proj.weight");
    let wu = up("mlp.up_proj.weight");
    let wd = up("mlp.down_proj.weight");

    let a = |n: usize| -> MBuf { gpu.alloc(n) };
    let xb = gpu.upload(x);
    let dyb = gpu.upload(dy);
    let (h, rln) = (a(t * d), a(t));
    let qkv = a(t * c);
    let (z, a_in, b_raw) = (a(t * di), a(t * hv), a(t * hv));
    let (bet, sp, gex) = (a(t * hv), a(t * hv), a(t * hv));
    let (acc, conv) = (a(t * c), a(t * c));
    let (o, st_hist) = (a(t * di), a((t + 1) * hv * s * s));
    let (sk_all, dlt_all) = (a(t * hv * s), a(t * hv * s));
    let (og, ron) = (a(t * di), a(t * hv));
    let mix = a(t * d);
    let x1 = a(t * d);
    let (h2, rpln) = (a(t * d), a(t));
    let (glin, ulin, act) = (a(t * ffn), a(t * ffn), a(t * ffn));
    let out = a(t * d);

    let (du32, c32, di32, hv32, hk32, s32, t32, ffn32) =
        (d as u32, c as u32, di as u32, hv as u32, hk as u32, s as u32, t as u32, ffn as u32);
    let tg256 = MTLSize::new(256, 1, 1);
    let tg128 = MTLSize::new(128, 1, 1);

    // ---------- forward ----------
    {
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        kit.d(enc, "t_rms_fwd", &[(&xb, 0), (&ln, 0), (&h, 0), (&rln, 0)], &[du32, t32], g1(t.div_ceil(8)), tg256);
        kit.d(enc, "t_mm_xwT", &[(&h, 0), (&wqkv, 0), (&qkv, 0)], &[du32, c32, t32], g2(t.div_ceil(32), (c).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_xwT", &[(&h, 0), (&wz, 0), (&z, 0)], &[du32, di32, t32], g2(t.div_ceil(32), (di).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_gemm_xwT", &[(&h, 0), (&wa, 0), (&a_in, 0)], &[du32, hv32, t32], g2(hv.div_ceil(8), t.div_ceil(16)), tg256);
        kit.d(enc, "t_gemm_xwT", &[(&h, 0), (&wb, 0), (&b_raw, 0)], &[du32, hv32, t32], g2(hv.div_ceil(8), t.div_ceil(16)), tg256);
        kit.d(enc, "t_gates_fwd", &[(&a_in, 0), (&b_raw, 0), (&dt, 0), (&alog, 0), (&bet, 0), (&sp, 0), (&gex, 0)],
              &[hv32, (t * hv) as u32], g1((t * hv).div_ceil(256)), tg256);
        kit.d(enc, "t_conv_fwd", &[(&qkv, 0), (&cw, 0), (&acc, 0), (&conv, 0)], &[c32, t32], g2(c.div_ceil(64), t), MTLSize::new(64, 1, 1));
        kit.d(enc, "t_dn_fwd", &[(&conv, 0), (&gex, 0), (&bet, 0), (&o, 0), (&st_hist, 0), (&sk_all, 0), (&dlt_all, 0)],
              &[s32, hk32, hv32, c32, t32], g2(s / 4, hv), tg128);
        kit.d(enc, "t_gnorm_fwd", &[(&o, 0), (&z, 0), (&nw, 0), (&og, 0), (&ron, 0)], &[s32, (t * hv) as u32], g1((t * hv).div_ceil(8)), tg256);
        kit.d(enc, "t_mm_xwT", &[(&og, 0), (&wout, 0), (&mix, 0)], &[di32, du32, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_copy", &[(&x1, 0), (&xb, 0)], &[(t * d) as u32], g1((t * d).div_ceil(256)), tg256);
        kit.d(enc, "t_add", &[(&x1, 0), (&mix, 0)], &[(t * d) as u32], g1((t * d).div_ceil(256)), tg256);
        kit.d(enc, "t_rms_fwd", &[(&x1, 0), (&pln, 0), (&h2, 0), (&rpln, 0)], &[du32, t32], g1(t.div_ceil(8)), tg256);
        kit.d(enc, "t_mm_xwT", &[(&h2, 0), (&wg, 0), (&glin, 0)], &[du32, ffn32, t32], g2(t.div_ceil(32), (ffn).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_xwT", &[(&h2, 0), (&wu, 0), (&ulin, 0)], &[du32, ffn32, t32], g2(t.div_ceil(32), (ffn).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_swiglu_fwd", &[(&glin, 0), (&ulin, 0), (&act, 0)], &[(t * ffn) as u32], g1((t * ffn).div_ceil(256)), tg256);
        kit.d(enc, "t_mm_xwT", &[(&act, 0), (&wd, 0), (&out, 0)], &[ffn32, du32, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_add", &[(&out, 0), (&x1, 0)], &[(t * d) as u32], g1((t * d).div_ceil(256)), tg256);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // ---------- backward ----------
    let dwd = a(d * ffn);
    let dact = a(t * ffn);
    let (dglin, dulin) = (a(t * ffn), a(t * ffn));
    let (dwg, dwu) = (a(ffn * d), a(ffn * d));
    let dh2 = a(t * d);
    let dpln = a(d);
    let dx1 = a(t * d);
    let dwout = a(d * di);
    let dog = a(t * di);
    let (d_o, dz, dnw) = (a(t * di), a(t * di), a(s));
    let ds_buf = a(hv * s * s);
    let dqk = a(t * hv * 16 * s);
    let (d_gexp, d_betp) = (a(t * hv * 4), a(t * hv * 4));
    let dconv = a(t * c);
    let (d_gex, d_bet) = (a(t * hv), a(t * hv));
    let dqkv = a(t * c);
    let dcw = a(c * 4);
    let (d_ain, d_braw) = (a(t * hv), a(t * hv));
    let (d_dt, d_alog) = (a(hv), a(hv));
    let (dwqkv, dwz, dwa, dwb) = (a(c * d), a(di * d), a(hv * d), a(hv * d));
    let dh = a(t * d);
    let dln = a(d);
    let dx = a(t * d);
    for b in [&ds_buf, &dconv] {
        unsafe { std::ptr::write_bytes(b.buf.contents() as *mut u8, 0, b.len * 4); }
    }
    {
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        // FFN backward
        kit.d(enc, "t_mm_dw", &[(&dyb, 0), (&act, 0), (&dwd, 0)], &[ffn32, du32, t32], g2((ffn).div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dyb, 0), (&wd, 0), (&dact, 0)], &[ffn32, du32, 0, t32], g2(t.div_ceil(32), (ffn).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_swiglu_bwd", &[(&dact, 0), (&glin, 0), (&ulin, 0), (&dglin, 0), (&dulin, 0)], &[(t * ffn) as u32], g1((t * ffn).div_ceil(256)), tg256);
        kit.d(enc, "t_mm_dw", &[(&dglin, 0), (&h2, 0), (&dwg, 0)], &[du32, ffn32, t32], g2((d).div_ceil(32), (ffn).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dglin, 0), (&wg, 0), (&dh2, 0)], &[du32, ffn32, 0, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dw", &[(&dulin, 0), (&h2, 0), (&dwu, 0)], &[du32, ffn32, t32], g2((d).div_ceil(32), (ffn).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dulin, 0), (&wu, 0), (&dh2, 0)], &[du32, ffn32, 1, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_copy", &[(&dx1, 0), (&dyb, 0)], &[(t * d) as u32], g1((t * d).div_ceil(256)), tg256);
        kit.d(enc, "t_rms_bwd", &[(&x1, 0), (&pln, 0), (&rpln, 0), (&dh2, 0), (&dx1, 0)], &[du32, t32, 1], g1(t.div_ceil(8)), tg256);
        kit.d(enc, "t_rms_dw", &[(&x1, 0), (&rpln, 0), (&dh2, 0), (&dpln, 0)], &[du32, t32], g1(d.div_ceil(256)), tg256);
        // mixer backward
        kit.d(enc, "t_mm_dw", &[(&dx1, 0), (&og, 0), (&dwout, 0)], &[di32, du32, t32], g2((di).div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dx1, 0), (&wout, 0), (&dog, 0)], &[di32, du32, 0, t32], g2(t.div_ceil(32), (di).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_gnorm_bwd", &[(&o, 0), (&z, 0), (&nw, 0), (&ron, 0), (&dog, 0), (&d_o, 0), (&dz, 0)], &[s32, (t * hv) as u32], g1((t * hv).div_ceil(8)), tg256);
        kit.d(enc, "t_gnorm_dnw", &[(&o, 0), (&z, 0), (&ron, 0), (&dog, 0), (&dnw, 0)], &[s32, (t * hv) as u32], g1(1), tg128);
        kit.d(enc, "t_dn_bwd", &[(&conv, 0), (&gex, 0), (&bet, 0), (&d_o, 0), (&st_hist, 0), (&sk_all, 0), (&dlt_all, 0),
              (&ds_buf, 0), (&dqk, 0), (&dconv, 0), (&d_gexp, 0), (&d_betp, 0)],
              &[s32, hk32, hv32, c32, t32], g2(4, hv), tg128);
        kit.d(enc, "t_dng_fold", &[(&d_gexp, 0), (&d_betp, 0), (&d_gex, 0), (&d_bet, 0)], &[(t * hv) as u32], g1((t * hv).div_ceil(256)), tg256);
        kit.d(enc, "t_dnqk_fold", &[(&dqk, 0), (&dconv, 0), (&conv, 0)], &[s32, hk32, hv32, c32, t32], g2(s.div_ceil(64), t * hk), MTLSize::new(64, 1, 1));
        kit.d(enc, "t_conv_bwd", &[(&dconv, 0), (&acc, 0), (&cw, 0), (&dqkv, 0)], &[c32, t32], g2(c.div_ceil(64), t), MTLSize::new(64, 1, 1));
        kit.d(enc, "t_conv_dw", &[(&dconv, 0), (&acc, 0), (&qkv, 0), (&dcw, 0)], &[c32, t32], g2(c.div_ceil(64), 4), MTLSize::new(64, 1, 1));
        kit.d(enc, "t_gates_bwd", &[(&d_gex, 0), (&d_bet, 0), (&gex, 0), (&bet, 0), (&a_in, 0), (&dt, 0), (&alog, 0), (&d_ain, 0), (&d_braw, 0)],
              &[hv32, (t * hv) as u32], g1((t * hv).div_ceil(256)), tg256);
        kit.d(enc, "t_gates_dtb", &[(&d_ain, 0), (&d_gex, 0), (&gex, 0), (&sp, 0), (&alog, 0), (&d_dt, 0), (&d_alog, 0)],
              &[hv32, t32], g1(1), tg128);
        kit.d(enc, "t_mm_dw", &[(&dqkv, 0), (&h, 0), (&dwqkv, 0)], &[du32, c32, t32], g2((d).div_ceil(32), (c).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dqkv, 0), (&wqkv, 0), (&dh, 0)], &[du32, c32, 0, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dw", &[(&dz, 0), (&h, 0), (&dwz, 0)], &[du32, di32, t32], g2((d).div_ceil(32), (di).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dz, 0), (&wz, 0), (&dh, 0)], &[du32, di32, 1, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_gemm_dw", &[(&d_ain, 0), (&h, 0), (&dwa, 0)], &[du32, hv32, t32], g2((d).div_ceil(1024), (hv).div_ceil(8)), tg256);
        kit.d(enc, "t_mm_dx", &[(&d_ain, 0), (&wa, 0), (&dh, 0)], &[du32, hv32, 1, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_gemm_dw", &[(&d_braw, 0), (&h, 0), (&dwb, 0)], &[du32, hv32, t32], g2((d).div_ceil(1024), (hv).div_ceil(8)), tg256);
        kit.d(enc, "t_mm_dx", &[(&d_braw, 0), (&wb, 0), (&dh, 0)], &[du32, hv32, 1, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_copy", &[(&dx, 0), (&dx1, 0)], &[(t * d) as u32], g1((t * d).div_ceil(256)), tg256);
        kit.d(enc, "t_rms_bwd", &[(&xb, 0), (&ln, 0), (&rln, 0), (&dh, 0), (&dx, 0)], &[du32, t32, 1], g1(t.div_ceil(8)), tg256);
        kit.d(enc, "t_rms_dw", &[(&xb, 0), (&rln, 0), (&dh, 0), (&dln, 0)], &[du32, t32], g1(d.div_ceil(256)), tg256);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let dl = |b: &MBuf| -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(b.buf.contents() as *const f32, b.len) }.to_vec()
    };
    let mut res = HashMap::new();
    res.insert("out".into(), dl(&out));
    res.insert("x".into(), dl(&dx));
    res.insert("input_layernorm.weight".into(), dl(&dln));
    res.insert("post_attention_layernorm.weight".into(), dl(&dpln));
    res.insert("in_proj_qkv.weight".into(), dl(&dwqkv));
    res.insert("in_proj_z.weight".into(), dl(&dwz));
    res.insert("in_proj_a.weight".into(), dl(&dwa));
    res.insert("in_proj_b.weight".into(), dl(&dwb));
    res.insert("dt_bias".into(), dl(&d_dt));
    res.insert("A_log".into(), dl(&d_alog));
    res.insert("conv1d.weight".into(), dl(&dcw));
    res.insert("norm.weight".into(), dl(&dnw));
    res.insert("out_proj.weight".into(), dl(&dwout));
    res.insert("mlp.gate_proj.weight".into(), dl(&dwg));
    res.insert("mlp.up_proj.weight".into(), dl(&dwu));
    res.insert("mlp.down_proj.weight".into(), dl(&dwd));
    Ok(res)
}

/// Full attention decoder-layer forward and backward on Metal (f32, HF layout). Same contract as
/// gdn_layer_check.
pub fn attn_layer_check(gpu: &MetalGpu, w: &HashMap<String, (Vec<usize>, Vec<f32>)>,
                        x: &[f32], dy: &[f32], t: usize) -> Result<HashMap<String, Vec<f32>>> {
    let (d, nh, nkv, hd, rot, ffn) = (4096usize, 16usize, 4usize, 256usize, 64usize, 12288usize);
    let (qdim, kvdim) = (nh * hd, nkv * hd);
    let kit = Kit::new(gpu)?;
    let up = |n: &str| -> MBuf { gpu.upload(&w[n].1) };
    let up1p = |n: &str| -> MBuf {
        gpu.upload(&w[n].1.iter().map(|v| v + 1.0).collect::<Vec<f32>>())
    };
    let ln = up1p("input_layernorm.weight");
    let pln = up1p("post_attention_layernorm.weight");
    let qnw = up1p("q_norm.weight");
    let knw = up1p("k_norm.weight");
    let wq = up("q_proj.weight");
    let wk = up("k_proj.weight");
    let wv = up("v_proj.weight");
    let wo = up("o_proj.weight");
    let wg = up("mlp.gate_proj.weight");
    let wu = up("mlp.up_proj.weight");
    let wd = up("mlp.down_proj.weight");

    let a = |n: usize| -> MBuf { gpu.alloc(n) };
    let xb = gpu.upload(x);
    let dyb = gpu.upload(dy);
    let (h, rln) = (a(t * d), a(t));
    let qfull = a(t * 2 * qdim);
    let (kfull, vfull) = (a(t * kvdim), a(t * kvdim));
    let (q0, q2) = (a(t * qdim), a(t * qdim));       // q2 = normed+roped (in place)
    let (rqn, rkn) = (a(t * nh), a(t * nkv));
    let k2 = a(t * kvdim);
    let p = a(nh * t * t);
    let (aog, ao) = (a(t * qdim), a(t * qdim));
    let mix = a(t * d);
    let x1 = a(t * d);
    let (h2, rpln) = (a(t * d), a(t));
    let (glin, ulin, act) = (a(t * ffn), a(t * ffn), a(t * ffn));
    let out = a(t * d);

    let (du32, t32, hd32, nh32, nkv32, rot32, ffn32) =
        (d as u32, t as u32, hd as u32, nh as u32, nkv as u32, rot as u32, ffn as u32);
    let (qd32, kvd32) = ((2 * qdim) as u32, kvdim as u32);
    let tg256 = MTLSize::new(256, 1, 1);
    let el = |n: usize| g1(n.div_ceil(256));

    // ---------- forward ----------
    {
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        kit.d(enc, "t_rms_fwd", &[(&xb, 0), (&ln, 0), (&h, 0), (&rln, 0)], &[du32, t32], g1(t.div_ceil(8)), tg256);
        kit.d(enc, "t_mm_xwT", &[(&h, 0), (&wq, 0), (&qfull, 0)], &[du32, qd32, t32], g2(t.div_ceil(32), ((2 * qdim)).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_xwT", &[(&h, 0), (&wk, 0), (&kfull, 0)], &[du32, kvd32, t32], g2(t.div_ceil(32), (kvdim).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_xwT", &[(&h, 0), (&wv, 0), (&vfull, 0)], &[du32, kvd32, t32], g2(t.div_ceil(32), (kvdim).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_qsplit", &[(&qfull, 0), (&q0, 0)], &[hd32, nh32, 0, (t * qdim) as u32], el(t * qdim), tg256);
        kit.d(enc, "t_rms_fwd", &[(&q0, 0), (&qnw, 0), (&q2, 0), (&rqn, 0)], &[hd32, (t * nh) as u32], g1((t * nh).div_ceil(8)), tg256);
        kit.d(enc, "t_rms_fwd", &[(&kfull, 0), (&knw, 0), (&k2, 0), (&rkn, 0)], &[hd32, (t * nkv) as u32], g1((t * nkv).div_ceil(8)), tg256);
        kit.d(enc, "t_rope", &[(&q2, 0)], &[hd32, nh32, rot32, 0, (t * nh * rot / 2) as u32], el(t * nh * rot / 2), tg256);
        kit.d(enc, "t_rope", &[(&k2, 0)], &[hd32, nkv32, rot32, 0, (t * nkv * rot / 2) as u32], el(t * nkv * rot / 2), tg256);
        kit.d(enc, "t_attn_fwd", &[(&q2, 0), (&k2, 0), (&vfull, 0), (&p, 0), (&aog, 0)],
              &[nh32, nkv32, hd32, t32], g2(nh.div_ceil(8), t.div_ceil(8)), MTLSize::new(8, 8, 1));
        kit.d(enc, "t_attngate_fwd", &[(&aog, 0), (&qfull, 0), (&ao, 0)], &[hd32, nh32, (t * qdim) as u32], el(t * qdim), tg256);
        kit.d(enc, "t_mm_xwT", &[(&ao, 0), (&wo, 0), (&mix, 0)], &[qdim as u32, du32, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_copy", &[(&x1, 0), (&xb, 0)], &[(t * d) as u32], el(t * d), tg256);
        kit.d(enc, "t_add", &[(&x1, 0), (&mix, 0)], &[(t * d) as u32], el(t * d), tg256);
        kit.d(enc, "t_rms_fwd", &[(&x1, 0), (&pln, 0), (&h2, 0), (&rpln, 0)], &[du32, t32], g1(t.div_ceil(8)), tg256);
        kit.d(enc, "t_mm_xwT", &[(&h2, 0), (&wg, 0), (&glin, 0)], &[du32, ffn32, t32], g2(t.div_ceil(32), (ffn).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_xwT", &[(&h2, 0), (&wu, 0), (&ulin, 0)], &[du32, ffn32, t32], g2(t.div_ceil(32), (ffn).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_swiglu_fwd", &[(&glin, 0), (&ulin, 0), (&act, 0)], &[(t * ffn) as u32], el(t * ffn), tg256);
        kit.d(enc, "t_mm_xwT", &[(&act, 0), (&wd, 0), (&out, 0)], &[ffn32, du32, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_add", &[(&out, 0), (&x1, 0)], &[(t * d) as u32], el(t * d), tg256);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // ---------- backward ----------
    let dwd = a(d * ffn);
    let dact = a(t * ffn);
    let (dglin, dulin) = (a(t * ffn), a(t * ffn));
    let (dwg, dwu) = (a(ffn * d), a(ffn * d));
    let dh2 = a(t * d);
    let dpln = a(d);
    let dx1 = a(t * d);
    let dwo = a(d * qdim);
    let dao = a(t * qdim);
    let (daog, dqfull) = (a(t * qdim), a(t * 2 * qdim));
    let ds = a(nh * t * t);
    let (dq2, dk2, dv) = (a(t * qdim), a(t * kvdim), a(t * kvdim));
    let (dq0, dk0) = (a(t * qdim), a(t * kvdim));
    let (dqnw, dknw) = (a(hd), a(hd));
    let (dwq, dwk, dwv) = (a(2 * qdim * d), a(kvdim * d), a(kvdim * d));
    let dh = a(t * d);
    let dln = a(d);
    let dx = a(t * d);
    {
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        kit.d(enc, "t_mm_dw", &[(&dyb, 0), (&act, 0), (&dwd, 0)], &[ffn32, du32, t32], g2((ffn).div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dyb, 0), (&wd, 0), (&dact, 0)], &[ffn32, du32, 0, t32], g2(t.div_ceil(32), (ffn).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_swiglu_bwd", &[(&dact, 0), (&glin, 0), (&ulin, 0), (&dglin, 0), (&dulin, 0)], &[(t * ffn) as u32], el(t * ffn), tg256);
        kit.d(enc, "t_mm_dw", &[(&dglin, 0), (&h2, 0), (&dwg, 0)], &[du32, ffn32, t32], g2((d).div_ceil(32), (ffn).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dglin, 0), (&wg, 0), (&dh2, 0)], &[du32, ffn32, 0, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dw", &[(&dulin, 0), (&h2, 0), (&dwu, 0)], &[du32, ffn32, t32], g2((d).div_ceil(32), (ffn).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dulin, 0), (&wu, 0), (&dh2, 0)], &[du32, ffn32, 1, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_copy", &[(&dx1, 0), (&dyb, 0)], &[(t * d) as u32], el(t * d), tg256);
        kit.d(enc, "t_rms_bwd", &[(&x1, 0), (&pln, 0), (&rpln, 0), (&dh2, 0), (&dx1, 0)], &[du32, t32, 1], g1(t.div_ceil(8)), tg256);
        kit.d(enc, "t_rms_dw", &[(&x1, 0), (&rpln, 0), (&dh2, 0), (&dpln, 0)], &[du32, t32], g1(d.div_ceil(256)), tg256);
        // attention mixer backward
        kit.d(enc, "t_mm_dw", &[(&dx1, 0), (&ao, 0), (&dwo, 0)], &[qdim as u32, du32, t32], g2((qdim).div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dx1, 0), (&wo, 0), (&dao, 0)], &[qdim as u32, du32, 0, t32], g2(t.div_ceil(32), (qdim).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_attngate_bwd", &[(&dao, 0), (&aog, 0), (&qfull, 0), (&daog, 0), (&dqfull, 0)], &[hd32, nh32, (t * qdim) as u32], el(t * qdim), tg256);
        kit.d(enc, "t_attn_dscore", &[(&q2, 0), (&k2, 0), (&vfull, 0), (&p, 0), (&daog, 0), (&ds, 0), (&dq2, 0)],
              &[nh32, nkv32, hd32, t32], g2(nh.div_ceil(8), t.div_ceil(8)), MTLSize::new(8, 8, 1));
        kit.d(enc, "t_attn_dkv", &[(&q2, 0), (&p, 0), (&ds, 0), (&daog, 0), (&dk2, 0), (&dv, 0)],
              &[nh32, nkv32, hd32, t32], g2(t, nkv), tg256);
        kit.d(enc, "t_rope", &[(&dq2, 0)], &[hd32, nh32, rot32, 1, (t * nh * rot / 2) as u32], el(t * nh * rot / 2), tg256);
        kit.d(enc, "t_rope", &[(&dk2, 0)], &[hd32, nkv32, rot32, 1, (t * nkv * rot / 2) as u32], el(t * nkv * rot / 2), tg256);
        kit.d(enc, "t_rms_bwd", &[(&q0, 0), (&qnw, 0), (&rqn, 0), (&dq2, 0), (&dq0, 0)], &[hd32, (t * nh) as u32, 0], g1((t * nh).div_ceil(8)), tg256);
        kit.d(enc, "t_rms_dw", &[(&q0, 0), (&rqn, 0), (&dq2, 0), (&dqnw, 0)], &[hd32, (t * nh) as u32], g1(1), tg256);
        kit.d(enc, "t_rms_bwd", &[(&kfull, 0), (&knw, 0), (&rkn, 0), (&dk2, 0), (&dk0, 0)], &[hd32, (t * nkv) as u32, 0], g1((t * nkv).div_ceil(8)), tg256);
        kit.d(enc, "t_rms_dw", &[(&kfull, 0), (&rkn, 0), (&dk2, 0), (&dknw, 0)], &[hd32, (t * nkv) as u32], g1(1), tg256);
        kit.d(enc, "t_qsplit", &[(&dqfull, 0), (&dq0, 0)], &[hd32, nh32, 1, (t * qdim) as u32], el(t * qdim), tg256);
        kit.d(enc, "t_mm_dw", &[(&dqfull, 0), (&h, 0), (&dwq, 0)], &[du32, qd32, t32], g2((d).div_ceil(32), (2 * qdim).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dqfull, 0), (&wq, 0), (&dh, 0)], &[du32, qd32, 0, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dw", &[(&dk0, 0), (&h, 0), (&dwk, 0)], &[du32, kvd32, t32], g2((d).div_ceil(32), (kvdim).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dk0, 0), (&wk, 0), (&dh, 0)], &[du32, kvd32, 1, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dw", &[(&dv, 0), (&h, 0), (&dwv, 0)], &[du32, kvd32, t32], g2((d).div_ceil(32), (kvdim).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_mm_dx", &[(&dv, 0), (&wv, 0), (&dh, 0)], &[du32, kvd32, 1, t32], g2(t.div_ceil(32), (d).div_ceil(64)), MTLSize::new(128, 1, 1));
        kit.d(enc, "t_copy", &[(&dx, 0), (&dx1, 0)], &[(t * d) as u32], el(t * d), tg256);
        kit.d(enc, "t_rms_bwd", &[(&xb, 0), (&ln, 0), (&rln, 0), (&dh, 0), (&dx, 0)], &[du32, t32, 1], g1(t.div_ceil(8)), tg256);
        kit.d(enc, "t_rms_dw", &[(&xb, 0), (&rln, 0), (&dh, 0), (&dln, 0)], &[du32, t32], g1(d.div_ceil(256)), tg256);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let dl = |b: &MBuf| -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(b.buf.contents() as *const f32, b.len) }.to_vec()
    };
    let mut res = HashMap::new();
    res.insert("out".into(), dl(&out));
    res.insert("x".into(), dl(&dx));
    res.insert("input_layernorm.weight".into(), dl(&dln));
    res.insert("post_attention_layernorm.weight".into(), dl(&dpln));
    res.insert("q_proj.weight".into(), dl(&dwq));
    res.insert("k_proj.weight".into(), dl(&dwk));
    res.insert("v_proj.weight".into(), dl(&dwv));
    res.insert("o_proj.weight".into(), dl(&dwo));
    res.insert("q_norm.weight".into(), dl(&dqnw));
    res.insert("k_norm.weight".into(), dl(&dknw));
    res.insert("mlp.gate_proj.weight".into(), dl(&dwg));
    res.insert("mlp.up_proj.weight".into(), dl(&dwu));
    res.insert("mlp.down_proj.weight".into(), dl(&dwd));
    Ok(res)
}

