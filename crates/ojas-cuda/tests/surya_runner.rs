//! The CUDA surya-2 runner (`ojas_cuda::CudaSsm`) against the CPU oracle
//! (`ojas_cpu::CpuSsm` in exact mode, `ojas_cpu::CpuVit`).
//!
//! GPU-only, so every test is `#[ignore]`:
//!
//! ```text
//! SP=~/.local/lib/python3.10/site-packages/nvidia
//! LD_LIBRARY_PATH=$SP/cuda_nvrtc/lib:$SP/cublas/lib OJAS_CUDA_INCLUDE=$SP/cuda_runtime/include \
//!   cargo test --release -p ojas-cuda --test surya_runner -- --ignored --nocapture --test-threads 1
//! ```
//!
//! The model tests find surya-2 as `ojas-cpu/tests/cpu_ssm_oracle.rs` does
//! (`OJAS_SURYA_GGUF` / `OJAS_SURYA_MMPROJ`, else an HF cache snapshot) and print `skip:`
//! without it. The page test reads `OJAS_OCR_PAGES` (default `/data/dev-cache/ocr-pages`).
//!
//! Compared:
//! * kernels (no model): the SSM family (`ssm_ab`, `conv1d_prefill`, `deltanet_fused`,
//!   `deltanet_prenorm` + `deltanet_scan`), `q35_qk_prep` and `q35_attn_256` / merge against
//!   transcriptions of the oracle's loops;
//! * per-layer hidden state: for every layer boundary, the max over rows of the normwise
//!   error `|cuda - cpu| / |cpu|`, for a text prompt and for an image prompt (identical
//!   injected rows on both sides);
//! * logits top-1 over a greedy page decode (CUDA teacher-forced along the oracle's tokens,
//!   and the first index where a free-running CUDA greedy decode leaves it);
//! * the vision tower's projector output (and per-block residual) against `CpuVit`.

use ojas_core::{KernelRuntime, Model};
use ojas_cpu::cpu_ssm::{image_pos3, text_pos3, CpuSsm, CpuSsmOpts, TraceCfg as CpuTrace};
use ojas_cuda::qwen35::{GemmMode, TraceCfg, VitAttn};
use ojas_cuda::{CuBuf, CudaGpu, CudaSsm, CudaSsmOpts};
use ojas_formats::gguf::Gguf;
use std::path::{Path, PathBuf};

// ================================ helpers ===================================

fn find_surya(file: &str, env: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env) {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(h) = std::env::var("HF_HUB_CACHE") { roots.push(h.into()); }
    if let Ok(h) = std::env::var("HF_HOME") { roots.push(Path::new(&h).join("hub")); }
    if let Ok(h) = std::env::var("HOME") { roots.push(Path::new(&h).join(".cache/huggingface/hub")); }
    roots.push("/data/dev-cache/hf/hub".into());
    for r in roots {
        let snaps = r.join("models--datalab-to--surya-ocr-2-gguf/snapshots");
        for e in std::fs::read_dir(&snaps).into_iter().flatten().flatten() {
            let p = e.path().join(file);
            if std::fs::metadata(&p).is_ok() { return Some(p); }
        }
    }
    None
}

fn surya() -> Option<(PathBuf, PathBuf)> {
    let m = find_surya("surya-2.gguf", "OJAS_SURYA_GGUF");
    let v = find_surya("surya-2-mmproj.gguf", "OJAS_SURYA_MMPROJ");
    match (m, v) {
        (Some(m), Some(v)) => Some((m, v)),
        _ => { eprintln!("skip: surya-2.gguf / surya-2-mmproj.gguf not found (set OJAS_SURYA_GGUF / OJAS_SURYA_MMPROJ)"); None }
    }
}

fn load_cpu(path: &Path) -> CpuSsm {
    let mut g = Gguf::open(path.to_str().unwrap()).unwrap();
    CpuSsm::load_with(&mut g, CpuSsmOpts { exact: true }).unwrap()
}

fn load_cuda(path: &Path, mmproj: Option<&Path>, gemm: GemmMode, context: usize, chunk: usize) -> CudaSsm {
    let mut g = Gguf::open(path.to_str().unwrap()).unwrap();
    let mut m = CudaSsm::load_with(&mut g, CudaSsmOpts { gemm, context, chunk, ..CudaSsmOpts::default() })
        .expect("these tests require a CUDA box: cargo test -- --ignored");
    if let Some(mm) = mmproj {
        let mut g = Gguf::open(mm.to_str().unwrap()).unwrap();
        m.attach_vit_gguf(&mut g).unwrap();
    }
    m
}

fn norm(v: &[f32]) -> f64 { v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt() }

/// `|got - want| / |want|`.
fn nerr(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let d: f64 = got.iter().zip(want).map(|(&a, &b)| { let e = a as f64 - b as f64; e * e }).sum::<f64>().sqrt();
    d / norm(want).max(1e-30)
}

fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for (i, &x) in v.iter().enumerate() { if x > v[b] { b = i; } }
    b
}

/// Gap between the best and second-best logit.
fn margin(v: &[f32]) -> f32 {
    let a = argmax(v);
    let second = v.iter().enumerate().filter(|&(i, _)| i != a).map(|(_, &x)| x).fold(f32::MIN, f32::max);
    v[a] - second
}

/// Per-boundary max-over-rows normwise error between two traces of the same rows. Boundary
/// 0 = input row, l+1 = after layer l, L+1 = post-final-norm. Prints the table and returns it.
fn layer_table(label: &str, cpu: &[Vec<Vec<f32>>], cuda: &[Vec<Vec<f32>>], n_layers: usize, attn_every: usize) -> Vec<f64> {
    assert_eq!(cpu.len(), cuda.len(), "{label}: traced row counts differ");
    let nb = n_layers + 2;
    let mut worst = vec![0f64; nb];
    let mut mean = vec![0f64; nb];
    for (c, g) in cpu.iter().zip(cuda) {
        assert_eq!(c.len(), nb);
        assert_eq!(g.len(), nb);
        for k in 0..nb {
            let e = nerr(&g[k], &c[k]);
            worst[k] = worst[k].max(e);
            mean[k] += e / cpu.len() as f64;
        }
    }
    eprintln!("\n{label}: {} rows, normwise |cuda-cpu|/|cpu| per layer boundary", cpu.len());
    eprintln!("  {:>9} {:>6} {:>11} {:>11}", "boundary", "kind", "max", "mean");
    for k in 0..nb {
        let (name, kind) = if k == 0 { ("input".to_string(), "-") }
            else if k == nb - 1 { ("final_norm".to_string(), "-") }
            else { (format!("L{}", k - 1), if k % attn_every == 0 { "attn" } else { "ssm" }) };
        eprintln!("  {name:>9} {kind:>6} {:>11.3e} {:>11.3e}", worst[k], mean[k]);
    }
    worst
}

fn gpu() -> CudaGpu {
    let mut g = CudaGpu::new(0).expect("these tests require a CUDA box: cargo test -- --ignored");
    for f in ["ssm", "qwen35", "ops"] { g.ensure_family(f).unwrap(); }
    g
}

fn up(g: &CudaGpu, v: &[f32]) -> CuBuf {
    g.upload_bytes(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap()
}

fn down(g: &CudaGpu, b: &CuBuf, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 4];
    g.read_bytes(b, 0, &mut raw).unwrap();
    raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn tvec(n: usize, seed: u32, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n).map(|_| { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (((s >> 9) as f32 / (1 << 23) as f32) - 1.0) * amp }).collect()
}

fn blocks(n: usize, b: usize) -> u32 { n.div_ceil(b) as u32 }

// ============================ kernel-level tests ============================

/// surya-2's GDN shape.
const S: usize = 128;
const HK: usize = 16;
const HV: usize = 16;
const DINNER: usize = 2048;
const CONV_CH: usize = 2 * HK * S + DINNER;
const KTAPS: usize = 4;
const EPS: f32 = 1e-6;

/// `cpu_ssm`'s SSM mixer body from the projections to the pre-norm output `o`, transcribed
/// loop for loop (conv1d + SiLU with rolling state, then the Gated DeltaNet recurrence).
#[allow(clippy::too_many_arguments)]
fn ssm_oracle(qkv_in: &[f32], alpha: &[f32], beta: &[f32], dt: &[f32], a: &[f32], conv_w: &[f32],
              conv_st: &mut [f32], state: &mut [f32], m: usize) -> Vec<f32> {
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let mut gate = vec![0f32; m * HV];
    let mut bet = vec![0f32; m * HV];
    for r in 0..m {
        for w in 0..HV {
            let xg = alpha[r * HV + w] + dt[w];
            let sp = if xg > 20.0 { xg } else { (1.0 + xg.exp()).ln() };
            gate[r * HV + w] = sp * a[w];
            bet[r * HV + w] = 1.0 / (1.0 + (-beta[r * HV + w]).exp());
        }
    }
    let mut qkv = qkv_in.to_vec();
    for q in qkv.chunks_exact_mut(CONV_CH) {
        for c in 0..CONV_CH {
            let mut acc = conv_w[c * KTAPS + (KTAPS - 1)] * q[c];
            for j in 0..KTAPS - 1 { acc += conv_w[c * KTAPS + j] * conv_st[j * CONV_CH + c]; }
            for j in 0..KTAPS - 2 { conv_st[j * CONV_CH + c] = conv_st[(j + 1) * CONV_CH + c]; }
            conv_st[(KTAPS - 2) * CONV_CH + c] = q[c];
            q[c] = silu(acc);
        }
    }
    let scale = 1.0 / (S as f32).sqrt();
    let mut o = vec![0f32; m * DINNER];
    for hh in 0..HV {
        let sb = &mut state[hh * S * S..(hh + 1) * S * S];
        let kh = hh % HK;
        for r in 0..m {
            let row = &qkv[r * CONV_CH..(r + 1) * CONV_CH];
            let q = &row[kh * S..(kh + 1) * S];
            let k = &row[HK * S + kh * S..HK * S + (kh + 1) * S];
            let v = &row[2 * HK * S + hh * S..2 * HK * S + (hh + 1) * S];
            let qn = 1.0 / (q.iter().map(|a| a * a).sum::<f32>() + EPS).sqrt() * scale;
            let kn = 1.0 / (k.iter().map(|a| a * a).sum::<f32>() + EPS).sqrt();
            let g = gate[r * HV + hh].exp();
            let b = bet[r * HV + hh];
            for col in 0..S {
                let srow = &mut sb[col * S..(col + 1) * S];
                let mut sk = 0.0;
                for j in 0..S { srow[j] *= g; sk += srow[j] * k[j]; }
                let dlt = (v[col] - sk * kn) * b;
                let mut y = 0.0;
                for j in 0..S { srow[j] += k[j] * kn * dlt; y += srow[j] * q[j]; }
                o[r * DINNER + hh * S + col] = y * qn;
            }
        }
    }
    o
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn ssm_kernels_match_oracle() {
    let g = gpu();
    let enc = g.begin();
    // several calls (a 37-row prefill, then short decode batches) so the carried conv and
    // recurrent state is checked, not just one call's output
    let dt = tvec(HV, 3, 1.0);
    let a: Vec<f32> = tvec(HV, 4, 1.0).iter().map(|v| -(v.abs() * 2.0 + 0.1)).collect();
    let conv_w = tvec(CONV_CH * KTAPS, 5, 0.6);
    let mut conv_ref = vec![0f32; (KTAPS - 1) * CONV_CH];
    let mut st_ref = vec![0f32; HV * S * S];
    let dtb = up(&g, &dt);
    let ab = up(&g, &a);
    let cwb = up(&g, &conv_w);
    for variant in ["fused", "scan"] {
        let conv_fused = up(&g, &vec![0f32; (KTAPS - 1) * CONV_CH]);
        let st_fused = up(&g, &vec![0f32; HV * S * S]);
        conv_ref.iter_mut().for_each(|v| *v = 0.0);
        st_ref.iter_mut().for_each(|v| *v = 0.0);
        let mut worst = 0f64;
        let mut worst_state = 0f64;
        for (call, m) in [(0u32, 37usize), (1, 1), (2, 5)] {
            let qkv = tvec(m * CONV_CH, 10 + call, 1.5);
            let alpha = tvec(m * HV, 20 + call, 2.0);
            let beta = tvec(m * HV, 30 + call, 2.0);
            let want = ssm_oracle(&qkv, &alpha, &beta, &dt, &a, &conv_w, &mut conv_ref, &mut st_ref, m);
            let qb = up(&g, &qkv);
            let alb = up(&g, &alpha);
            let beb = up(&g, &beta);
            let ob = up(&g, &vec![0f32; m * DINNER]);
            let nab = (m * HV) as u32;
            g.dispatch(&enc, "ssm_ab", &[(&alb, 0), (&beb, 0), (&dtb, 0), (&ab, 0)], &[nab, HV as u32], [blocks(m * HV, 256), 1, 1], [256, 1, 1]).unwrap();
            // one segment: rows 0..m of slot 0
            let seg = g.upload_bytes(&[0u32, m as u32, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            g.dispatch(&enc, "conv1d_prefill", &[(&qb, 0), (&conv_fused, 0), (&cwb, 0), (&seg, 0)], &[CONV_CH as u32, KTAPS as u32, 0],
                       [blocks(CONV_CH, 128), 1, 1], [128, 1, 1]).unwrap();
            if variant == "fused" {
                g.dispatch(&enc, "deltanet_fused", &[(&st_fused, 0), (&qb, 0), (&alb, 0), (&beb, 0), (&ob, 0), (&seg, 0)],
                           &[S as u32, HK as u32, HV as u32, CONV_CH as u32, 0, EPS.to_bits()], [(S / 16) as u32, HV as u32, 1], [128, 1, 1]).unwrap();
            } else {
                let qk = up(&g, &vec![0f32; m * HK]);
                let warps = m * HK;
                g.dispatch(&enc, "deltanet_prenorm", &[(&qb, 0), (&qk, 0)], &[HK as u32, S as u32, CONV_CH as u32, m as u32, EPS.to_bits()],
                           [blocks(warps * 32, 128), 1, 1], [128, 1, 1]).unwrap();
                g.dispatch(&enc, "deltanet_scan", &[(&st_fused, 0), (&qb, 0), (&qk, 0), (&alb, 0), (&beb, 0), (&ob, 0)],
                           &[S as u32, HK as u32, HV as u32, CONV_CH as u32, m as u32], [(S / 16) as u32, HV as u32, 1], [128, 1, 1]).unwrap();
            }
            let got = down(&g, &ob, m * DINNER);
            let e = nerr(&got, &want);
            worst = worst.max(e);
            let st_got = down(&g, &st_fused, HV * S * S);
            worst_state = worst_state.max(nerr(&st_got, &st_ref));
            let cv_got = down(&g, &conv_fused, (KTAPS - 1) * CONV_CH);
            assert_eq!(cv_got, conv_ref, "conv1d_prefill rolling state must be the inputs verbatim");
        }
        eprintln!("ssm {variant}: output normwise err {worst:.3e}, carried state {worst_state:.3e}");
        assert!(worst < 2e-5 && worst_state < 2e-5, "{variant}: ssm output {worst:.3e} / state {worst_state:.3e}");
    }
}

/// `q35_qk_prep` + `q35_attn_256` (+ split/merge) against the oracle's attention block:
/// QK-RMSNorm, sectioned partial rope, causal softmax over an f32 cache.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_matches_oracle() {
    use ojas_cpu::cpu_ssm::{rope_partial, rope_partial_m, RopeAt, MROPE_INTERLEAVED};
    let g = gpu();
    let enc = g.begin();
    let (nh, nkv, hd, rd) = (8usize, 2usize, 256usize, 64usize);
    let (qdim, kvdim) = (nh * hd, nkv * hd);
    let sections = [11u32, 11, 10, 0];
    let theta = 1e7f32;
    let qw: Vec<f32> = tvec(hd, 7, 0.3).iter().map(|v| 1.0 + v).collect();
    let kw: Vec<f32> = tvec(hd, 8, 0.3).iter().map(|v| 1.0 + v).collect();
    let ctx = 1400usize;
    let kc = up(&g, &vec![0f32; ctx * kvdim]);
    let vc = up(&g, &vec![0f32; ctx * kvdim]);
    let (qwb, kwb) = (up(&g, &qw), up(&g, &kw));
    let mut kref: Vec<f32> = Vec::new();
    let mut vref: Vec<f32> = Vec::new();
    let mut worst = 0f64;
    // calls: image-like sectioned span, a text chunk, single decode rows at long context
    let mut base = 0usize;
    for (call, (m, sect)) in [(300usize, true), (700, false), (1, false), (3, false), (1, true)].into_iter().enumerate() {
        let qfull = tvec(m * 2 * qdim, 100 + call as u32, 2.0);
        let kin = tvec(m * kvdim, 200 + call as u32, 2.0);
        let vin = tvec(m * kvdim, 300 + call as u32, 1.0);
        let rope: Vec<RopeAt> = (0..m).map(|i| if sect {
            RopeAt::Sect([base as u32 + 3, (base + 3 + i / 17) as u32, (base + 3 + i % 17) as u32, 0])
        } else { RopeAt::Scalar(base + i) }).collect();
        // oracle
        let mut q = vec![0f32; m * qdim];
        for r in 0..m {
            let rope_head = |seg: &mut [f32]| match rope[r] {
                RopeAt::Scalar(p) => rope_partial(seg, rd, p, theta),
                RopeAt::Sect(p4) => rope_partial_m(seg, rd, p4, sections, MROPE_INTERLEAVED, theta),
            };
            for h in 0..nh {
                let seg = &mut q[r * qdim + h * hd..r * qdim + (h + 1) * hd];
                seg.copy_from_slice(&qfull[r * 2 * qdim + h * 2 * hd..r * 2 * qdim + h * 2 * hd + hd]);
                let ss: f32 = seg.iter().map(|v| v * v).sum::<f32>() / hd as f32;
                let inv = 1.0 / (ss + EPS).sqrt();
                for i in 0..hd { seg[i] *= inv * qw[i]; }
                rope_head(seg);
            }
            let mut k = kin[r * kvdim..(r + 1) * kvdim].to_vec();
            for h in 0..nkv {
                let seg = &mut k[h * hd..(h + 1) * hd];
                let ss: f32 = seg.iter().map(|v| v * v).sum::<f32>() / hd as f32;
                let inv = 1.0 / (ss + EPS).sqrt();
                for i in 0..hd { seg[i] *= inv * kw[i]; }
                rope_head(seg);
            }
            kref.extend_from_slice(&k);
            vref.extend_from_slice(&vin[r * kvdim..(r + 1) * kvdim]);
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let mut want = vec![0f32; m * qdim];
        for r in 0..m {
            let seq = base + r + 1;
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                let qh = &q[r * qdim + h * hd..r * qdim + (h + 1) * hd];
                let mut sc: Vec<f32> = (0..seq).map(|t| {
                    qh.iter().zip(&kref[t * kvdim + kvh * hd..t * kvdim + kvh * hd + hd]).map(|(a, b)| a * b).sum::<f32>() * scale
                }).collect();
                let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                let mut den = 0.0;
                for s in sc.iter_mut() { *s = (*s - mx).exp(); den += *s; }
                for i in 0..hd {
                    let acc: f32 = (0..seq).map(|t| sc[t] * vref[t * kvdim + kvh * hd + i]).sum();
                    want[r * qdim + h * hd + i] = acc / den;
                }
            }
        }
        // cuda
        let mpos: Vec<u32> = rope.iter().flat_map(|r| r.as4()).collect();
        let mode = if sect { 2u32 } else { 0 };
        let mposb = g.upload_bytes(&mpos.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let (qfb, kib, vib) = (up(&g, &qfull), up(&g, &kin), up(&g, &vin));
        let qb = up(&g, &vec![0f32; m * qdim]);
        let ob = up(&g, &vec![0f32; m * qdim]);
        let warps = m * (nh + 2 * nkv);
        let total = base + m;
        let ctl = g.upload_bytes(&[base as u32, total as u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        g.dispatch(&enc, "q35_qk_prep", &[(&qfb, 0), (&kib, 0), (&vib, 0), (&qwb, 0), (&kwb, 0), (&mposb, 0), (&qb, 0), (&kc, 0), (&vc, 0), (&ctl, 0)],
                   &[hd as u32, nh as u32, nkv as u32, rd as u32, m as u32, sections[0], sections[1], sections[2], sections[3],
                     mode, theta.to_bits(), EPS.to_bits()], [blocks(warps, 4), 1, 1], [128, 1, 1]).unwrap();
        let qe = nerr(&down(&g, &qb, m * qdim), &q);
        let nsplit = if m <= 16 { total.div_ceil(256) } else { 1 };
        let chunk = if nsplit > 1 { 256 } else { total };
        let po = up(&g, &vec![0f32; nsplit * m * qdim]);
        let pml = up(&g, &vec![0f32; nsplit * m * nh * 2]);
        g.dispatch(&enc, "q35_attn_256", &[(&qb, 0), (&kc, 0), (&vc, 0), (&ob, 0), (&po, 0), (&pml, 0), (&ctl, 0)],
                   &[kvdim as u32, m as u32, (nh / nkv) as u32, nh as u32, chunk as u32, 1, scale.to_bits()],
                   [blocks(m, 32 / (nh / nkv)), nkv as u32, nsplit as u32], [128, 1, 1]).unwrap();
        if nsplit > 1 {
            g.dispatch(&enc, "q35_attn_merge", &[(&po, 0), (&pml, 0), (&ob, 0)], &[hd as u32, nh as u32, m as u32, nsplit as u32],
                       [(m * nh) as u32, 1, 1], [128, 1, 1]).unwrap();
        }
        let got = down(&g, &ob, m * qdim);
        let e = nerr(&got, &want);
        eprintln!("attention call {call}: m={m} base={base} sect={sect} splits={nsplit} | q prep err {qe:.3e} | out err {e:.3e}");
        worst = worst.max(e).max(qe);
        base += m;
    }
    let kc_got = down(&g, &kc, base * kvdim);
    let ke = nerr(&kc_got, &kref);
    let ve = nerr(&down(&g, &vc, base * kvdim), &vref);
    eprintln!("cache: K err {ke:.3e}, V err {ve:.3e}");
    assert!(worst < 1e-5 && ke < 1e-5 && ve == 0.0, "attention parity {worst:.3e} K {ke:.3e} V {ve:.3e}");
}

// ============================== model tests =================================

const TEXT: &str = "<div data-label=\"Text\">The quick brown fox jumps over the lazy dog. \
    Invoice 2026-09-27, total 1,234.50 INR.</div>";

fn cpu_rows(t: Vec<ojas_cpu::cpu_ssm::TraceRow>) -> (Vec<Vec<Vec<f32>>>, Vec<Option<Vec<f32>>>) {
    t.into_iter().map(|r| (r.hidden, r.logits)).unzip()
}

fn cuda_rows(t: Vec<ojas_cuda::qwen35::TraceRow>) -> (Vec<Vec<Vec<f32>>>, Vec<Option<Vec<f32>>>) {
    t.into_iter().map(|r| (r.hidden, r.logits)).unzip()
}

fn logits_report(label: &str, cpu: &[Option<Vec<f32>>], cuda: &[Option<Vec<f32>>]) -> (usize, usize, f64) {
    let mut agree = 0;
    let mut n = 0;
    let mut worst = 0f64;
    for (i, (c, g)) in cpu.iter().zip(cuda).enumerate() {
        let (Some(c), Some(g)) = (c, g) else { continue };
        n += 1;
        worst = worst.max(nerr(g, c));
        if argmax(c) == argmax(g) { agree += 1; } else {
            eprintln!("  {label} row {i}: cpu top1 {} (margin {:.3}) vs cuda top1 {}", argmax(c), margin(c), argmax(g));
        }
    }
    eprintln!("{label}: logits top-1 {agree}/{n} agree, max normwise logits err {worst:.3e}");
    (agree, n, worst)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn text_prompt_layer_parity() {
    let Some((path, _)) = surya() else { return };
    let g = Gguf::open(path.to_str().unwrap()).unwrap();
    let bpe = ojas_tokenize::Bpe::from_gguf(&g);
    let toks: Vec<u32> = bpe.encode(&ojas_tokenize::chat_template("qwen35", TEXT)).into_iter().map(|t| t as u32).collect();
    let n = toks.len();
    let cfg = || CpuTrace { hidden: true, logits: true, rows: None };

    let cpu = load_cpu(&path);
    cpu.trace_start(cfg());
    cpu.prefill(&toks[..n - 1], 0);
    let _ = cpu.forward_logits(toks[n - 1], n - 1).unwrap();
    let (ch, cl) = cpu_rows(cpu.trace_take());
    let n_layers = cpu.n_layers();
    drop(cpu);

    for mode in [GemmMode::Exact, GemmMode::Split, GemmMode::Fast] {
        // chunk 16 so the prompt crosses internal chunk boundaries (state carried in between)
        let cuda = load_cuda(&path, None, mode, 1024, 16);
        cuda.trace_start(TraceCfg { hidden: true, logits: true, rows: None });
        cuda.prefill(&toks[..n - 1], 0);
        let _ = cuda.forward_logits(toks[n - 1], n - 1).unwrap();
        let (gh, gl) = cuda_rows(cuda.trace_take());
        let worst = layer_table(&format!("text prompt ({n} rows), gemm {mode:?}"), &ch, &gh, n_layers, 4);
        let (agree, total, _) = logits_report(&format!("text {mode:?}"), &cl, &gl);
        if mode == GemmMode::Exact {
            assert!(worst.iter().all(|&e| e < 1e-3), "exact-mode hidden parity too loose: {worst:?}");
            assert_eq!(agree, total, "exact mode must agree on every top-1");
        }
    }
}

/// The ocr prompt for an `nx` x `ny` image: ChatML head (injected, shifted by `delta` as
/// `PosLayout::Anchored` does), image rows at M-RoPE coordinates, then the tail by id.
struct PagePrompt {
    head: Vec<u32>,
    tail: Vec<u32>,
    pad: u32,
    nx: usize,
    ny: usize,
}

impl PagePrompt {
    fn new(bpe: &ojas_tokenize::Bpe, nx: usize, ny: usize) -> PagePrompt {
        const MARK: &str = "\u{0}\u{0}span\u{0}\u{0}";
        let instr = "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).";
        let full = ojas_tokenize::chat_template("qwen35", &format!("<|vision_start|>{MARK}<|vision_end|>{instr}"));
        let (h, t) = full.split_once(MARK).unwrap();
        let enc = |s: &str| bpe.encode(s).into_iter().map(|v| v as u32).collect::<Vec<u32>>();
        let pad = enc("<|image_pad|>")[0];
        PagePrompt { head: enc(h), tail: enc(t), pad, nx, ny }
    }
    fn n_img(&self) -> usize { self.nx * self.ny }
    fn image_at(&self) -> usize { self.head.len() }
    fn len(&self) -> usize { self.head.len() + self.n_img() + self.tail.len() }
    fn delta(&self) -> u32 { (self.n_img() - self.nx.max(self.ny)) as u32 }
    /// Prefill everything but the last tail token; returns (last token, its cache row).
    fn prefill(&self, m: &dyn Model, head_rows: &[f32], img_rows: &[f32]) -> (u32, usize) {
        m.reset_session();
        let d = self.delta();
        assert!(m.prefill_embeds(&self.head, head_rows, 0, Some(&text_pos3(d, self.head.len()))));
        let at = self.image_at();
        assert!(m.prefill_embeds(&vec![self.pad; self.n_img()], img_rows, at, Some(&image_pos3(at as u32 + d, self.nx, self.ny))));
        let t0 = at + self.n_img();
        m.prefill(&self.tail[..self.tail.len() - 1], t0);
        (self.tail[self.tail.len() - 1], self.len() - 1)
    }
}

fn pages_dir() -> PathBuf {
    std::env::var("OJAS_OCR_PAGES").map(PathBuf::from)
        .unwrap_or_else(|_| "/data/dev-cache/ocr-pages".into())
}

/// Preprocessed page (planar CHW) at a merged-token budget.
fn page(name: &str, max_tokens: i32) -> Option<(usize, usize, Vec<f32>)> {
    let p = pages_dir().join(name);
    if !p.exists() { eprintln!("skip: {} not found (set OJAS_OCR_PAGES)", p.display()); return None; }
    let (w0, h0, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(&p).unwrap();
    let pre = ojas_cpu::VitPreproc::qwen3vl().with_token_budget(8, max_tokens);
    Some(pre.preprocess(&rgb, w0, h0).unwrap())
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn vision_tower_matches_cpu_vit() {
    let Some((path, mmproj)) = surya() else { return };
    // OJAS_VIT_PAGE / OJAS_VIT_BUDGET pick a larger case (the CPU tower is quadratic in patches)
    let name = std::env::var("OJAS_VIT_PAGE").unwrap_or_else(|_| "einstein_szilard_p1.jpg".into());
    let budget = std::env::var("OJAS_VIT_BUDGET").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let small = budget <= 256;
    let Some((w, h, img)) = page(&name, budget) else { return };
    let mut gv = Gguf::open(mmproj.to_str().unwrap()).unwrap();
    let vit = ojas_cpu::CpuVit::load(&mut gv).unwrap();
    let mut layers: Vec<Vec<f32>> = Vec::new();
    let t0 = std::time::Instant::now();
    let cpu = vit.forward_tapped(&img, w, h, &mut |name, _l, v| if name == "layer_out" { layers.push(v.to_vec()) }).unwrap();
    let cpu_s = t0.elapsed().as_secs_f64();
    let n_pos = (w / 16) * (h / 16);
    eprintln!("image {w}x{h}: {n_pos} patches -> {} rows (CPU ViT {cpu_s:.1}s)", cpu.out.len() / 1024);
    for (mode, attn) in [(GemmMode::Split, VitAttn::F32), (GemmMode::Split, VitAttn::X3), (GemmMode::Split, VitAttn::F16),
                         (GemmMode::Fast, VitAttn::F16)] {
        let mut m = load_cuda(&path, Some(&mmproj), mode, 64, 64);
        m.set_vit_attn(attn);
        let t0 = std::time::Instant::now();
        let got = m.encode_image_trace(&img, w, h, true).unwrap();
        let gs = t0.elapsed().as_secs_f64();
        let per_block: Vec<String> = got.layer_out.iter().zip(&layers).map(|(g, c)| format!("{:.1e}", nerr(g, c))).collect();
        let e_post = nerr(&got.post_ln, &cpu.post_ln);
        let e_out = nerr(&got.out, &cpu.out);
        let d = 1024;
        let row_worst = got.out.chunks(d).zip(cpu.out.chunks(d)).map(|(a, b)| nerr(a, b)).fold(0f64, f64::max);
        eprintln!("ViT gemm {mode:?} attn {attn:?}: out err {e_out:.3e} (worst row {row_worst:.3e}), post_ln {e_post:.3e}, {gs:.2}s\n  per block: {}",
                  per_block.join(" "));
        match (mode, attn) {
            // past a few hundred patches the tower amplifies last-bit differences
            // (vision_tower_page_scale); the tight bound holds only for the small fixture
            _ if !small => assert!(e_out < 5e-2, "{attn:?} tower drifted: {e_out:.3e}"),
            (GemmMode::Split, VitAttn::F32 | VitAttn::X3) =>
                assert!(e_out < 2e-4, "{attn:?}-attention tower should match the oracle to ~f32 rounding: {e_out:.3e}"),
            _ => assert!(e_out < 2e-2, "f16-attention tower drifted: {e_out:.3e}"),
        }
    }
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn image_prompt_layer_parity() {
    let Some((path, mmproj)) = surya() else { return };
    let Some((w, h, img)) = page("einstein_szilard_p1.jpg", 128) else { return };
    let mut gv = Gguf::open(mmproj.to_str().unwrap()).unwrap();
    let rows = ojas_cpu::CpuVit::load(&mut gv).unwrap().forward(&img, w, h).unwrap();
    let g = Gguf::open(path.to_str().unwrap()).unwrap();
    let bpe = ojas_tokenize::Bpe::from_gguf(&g);
    let pp = PagePrompt::new(&bpe, w / 32, h / 32);
    assert_eq!(rows.len(), pp.n_img() * 1024);
    let cpu = load_cpu(&path);
    let head_rows: Vec<f32> = pp.head.iter().flat_map(|&t| cpu.embed_row(t as usize)).collect();
    cpu.trace_start(CpuTrace { hidden: true, logits: true, rows: None });
    let (last, pos) = pp.prefill(&cpu, &head_rows, &rows);
    let _ = cpu.forward_logits(last, pos).unwrap();
    let (ch, cl) = cpu_rows(cpu.trace_take());
    let n_layers = cpu.n_layers();
    drop(cpu);
    for mode in [GemmMode::Exact, GemmMode::Split, GemmMode::Fast] {
        let cuda = load_cuda(&path, None, mode, 2048, 64);
        cuda.trace_start(TraceCfg { hidden: true, logits: true, rows: None });
        let (last, pos) = pp.prefill(&cuda, &head_rows, &rows);
        let _ = cuda.forward_logits(last, pos).unwrap();
        let (gh, gl) = cuda_rows(cuda.trace_take());
        let worst = layer_table(&format!("image prompt ({} rows: {} image {}x{}), gemm {mode:?}", pp.len(), pp.n_img(), pp.nx, pp.ny),
                                &ch, &gh, n_layers, 4);
        let (agree, total, _) = logits_report(&format!("image {mode:?}"), &cl, &gl);
        if mode == GemmMode::Exact {
            assert!(worst.iter().all(|&e| e < 1e-3), "exact-mode hidden parity too loose: {worst:?}");
            assert!(agree + 2 >= total, "exact mode top-1 agreement {agree}/{total}");
        }
    }
}

/// Greedy decode of a page: the oracle generates, CUDA is teacher-forced along its tokens
/// (top-1 agreement per step), then CUDA decodes freely and the first divergence is reported.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn greedy_page_decode_top1() {
    let Some((path, mmproj)) = surya() else { return };
    let budget: i32 = std::env::var("OJAS_GATE_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let steps: usize = std::env::var("OJAS_GATE_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(192);
    let Some((w, h, img)) = page("einstein_szilard_p1.jpg", budget) else { return };
    let mut gv = Gguf::open(mmproj.to_str().unwrap()).unwrap();
    let t0 = std::time::Instant::now();
    let rows = ojas_cpu::CpuVit::load(&mut gv).unwrap().forward(&img, w, h).unwrap();
    eprintln!("page {w}x{h}: CPU ViT {:.1}s", t0.elapsed().as_secs_f64());
    let g = Gguf::open(path.to_str().unwrap()).unwrap();
    let bpe = ojas_tokenize::Bpe::from_gguf(&g);
    let eog = ojas_tokenize::eog_token_ids(&g, "qwen35");
    let pp = PagePrompt::new(&bpe, w / 32, h / 32);
    let cpu = load_cpu(&path);
    let head_rows: Vec<f32> = pp.head.iter().flat_map(|&t| cpu.embed_row(t as usize)).collect();
    let t0 = std::time::Instant::now();
    let (mut cur, mut pos) = pp.prefill(&cpu, &head_rows, &rows);
    let pre_s = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let mut cpu_logits = Vec::new();
    let mut cpu_ids = Vec::new();
    for _ in 0..steps {
        let l = cpu.forward_logits(cur, pos).unwrap();
        let id = argmax(&l) as u32;
        cpu_logits.push(l);
        cpu_ids.push(id);
        if eog.contains(&id) { break; }
        cur = id;
        pos += 1;
    }
    eprintln!("oracle: prefill {} rows {pre_s:.1}s, {} tokens {:.1}s", pp.len() - 1, cpu_ids.len(), t0.elapsed().as_secs_f64());
    drop(cpu);
    let text = |ids: &[u32]| ids.iter().map(|&i| bpe.decode(i as usize)).collect::<String>();
    eprintln!("oracle text: {}", text(&cpu_ids));
    for mode in [GemmMode::Exact, GemmMode::Split, GemmMode::Fast] {
        let cuda = load_cuda(&path, None, mode, 4096, 256);
        // teacher-forced
        let (first, p0) = pp.prefill(&cuda, &head_rows, &rows);
        let (mut cur, mut pos) = (first, p0);
        let mut agree = 0;
        let mut worst = 0f64;
        let mut flips = Vec::new();
        for (i, want) in cpu_logits.iter().enumerate() {
            let l = cuda.forward_logits(cur, pos).unwrap();
            worst = worst.max(nerr(&l, want));
            if argmax(&l) == argmax(want) { agree += 1; } else { flips.push((i, margin(want))); }
            cur = cpu_ids[i];
            pos += 1;
        }
        // free-running greedy
        let (mut cur, mut pos) = pp.prefill(&cuda, &head_rows, &rows);
        let t0 = std::time::Instant::now();
        let mut ids = Vec::new();
        for _ in 0..cpu_ids.len() {
            let id = cuda.forward_id(cur, pos);
            ids.push(id);
            if eog.contains(&id) { break; }
            cur = id;
            pos += 1;
        }
        let ds = t0.elapsed().as_secs_f64();
        if mode == GemmMode::Split {
            // the recorded decode graph against the same step launched kernel by kernel
            std::env::set_var("OJAS_CUDA_GRAPH", "0");
            let (mut c2, mut p2) = pp.prefill(&cuda, &head_rows, &rows);
            let mut ids2 = Vec::new();
            for _ in 0..ids.len() {
                let id = cuda.forward_id(c2, p2);
                ids2.push(id);
                c2 = id;
                p2 += 1;
            }
            std::env::remove_var("OJAS_CUDA_GRAPH");
            assert_eq!(ids, ids2, "graph and eager decode must produce the same tokens");
        }
        let div = ids.iter().zip(&cpu_ids).position(|(a, b)| a != b);
        eprintln!("greedy {mode:?}: teacher-forced top-1 {agree}/{} (flips at {flips:?}), max logits err {worst:.3e}; free-run first divergence {:?} of {}; decode {:.1} tok/s",
                  cpu_logits.len(), div, cpu_ids.len(), ids.len() as f64 / ds);
        if div.is_some() { eprintln!("  cuda text: {}", text(&ids)); }
        if mode == GemmMode::Exact {
            assert!(agree + 1 >= cpu_logits.len(), "exact mode teacher-forced top-1 {agree}/{}", cpu_logits.len());
        }
    }
}

/// Page scale (4 047 tokens, 16 188 patches), where the CPU tower would take ~10 minutes: the
/// two tensor-core attentions against the f32 CUDA-core one, which matches `CpuVit` to ~6e-5
/// at small scale (`vision_tower_matches_cpu_vit`). Also times the tower per mode.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn vision_tower_page_scale() {
    let Some((path, mmproj)) = surya() else { return };
    let Some((w, h, img)) = page("blisscopy_1.jpg", 4096) else { return };
    let mut m = load_cuda(&path, Some(&mmproj), GemmMode::Split, 64, 64);
    let mut outs = Vec::new();
    for attn in [VitAttn::F32, VitAttn::X3, VitAttn::F16] {
        m.set_vit_attn(attn);
        let _ = m.encode_image_trace(&img, w, h, false).unwrap(); // warm
        let t0 = std::time::Instant::now();
        let o = m.encode_image_trace(&img, w, h, false).unwrap();
        eprintln!("ViT {attn:?} at {}x{} ({} patches): {:.3}s", w, h, (w / 16) * (h / 16), t0.elapsed().as_secs_f64());
        outs.push((attn, o.out));
    }
    let d = 1024;
    for (attn, o) in &outs[1..] {
        let e = nerr(o, &outs[0].1);
        let worst = o.chunks(d).zip(outs[0].1.chunks(d)).map(|(a, b)| nerr(a, b)).fold(0f64, f64::max);
        eprintln!("  {attn:?} vs F32: {e:.3e} (worst row {worst:.3e})");
        if *attn == VitAttn::X3 { assert!(e < 1e-2, "x3 drifted from f32 at page scale: {e:.3e}"); }
    }
}

/// Decode throughput on a page prompt (CUDA tower, no oracle): the recorded decode graph
/// against eager launches, at the page's context.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn decode_speed() {
    let Some((path, mmproj)) = surya() else { return };
    let name = std::env::var("OJAS_VIT_PAGE").unwrap_or_else(|_| "einstein_szilard_p1.jpg".into());
    let Some((w, h, img)) = page(&name, 4096) else { return };
    let steps: usize = std::env::var("OJAS_GATE_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let g = Gguf::open(path.to_str().unwrap()).unwrap();
    let bpe = ojas_tokenize::Bpe::from_gguf(&g);
    let cuda = load_cuda(&path, Some(&mmproj), GemmMode::Split, 16384, 256);
    let pp = PagePrompt::new(&bpe, w / 32, h / 32);
    let (dims, _, emb) = { let mut g = g; g.read_tensor("token_embd.weight").unwrap() };
    let d = dims[0] as usize;
    let head_rows: Vec<f32> = pp.head.iter().flat_map(|&t| emb[t as usize * d * 2..(t as usize + 1) * d * 2].chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect::<Vec<_>>()).collect();
    let t0 = std::time::Instant::now();
    let rows = cuda.encode_image(&img, w, h).unwrap().unwrap();
    let vit_s = t0.elapsed().as_secs_f64();
    for graph in [true, false] {
        if !graph { std::env::set_var("OJAS_CUDA_GRAPH", "0"); }
        let t0 = std::time::Instant::now();
        let (mut cur, mut pos) = pp.prefill(&cuda, &head_rows, &rows);
        let pre_s = t0.elapsed().as_secs_f64();
        let _ = cuda.forward_id(cur, pos); // warm (and capture)
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            cur = cuda.forward_id(cur, pos);
            pos += 1;
        }
        let s = t0.elapsed().as_secs_f64();
        eprintln!("{name}: ViT {vit_s:.2}s, prefill {} rows {pre_s:.2}s | decode ({}) {steps} tokens from row {}: {:.2} ms/token = {:.1} tok/s",
                  pp.len() - 1, if graph { "graph" } else { "eager" }, pp.len(), 1e3 * s / steps as f64, steps as f64 / s);
        std::env::remove_var("OJAS_CUDA_GRAPH");
    }
}
