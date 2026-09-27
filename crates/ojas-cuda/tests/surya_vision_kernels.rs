//! The CUDA `vision` family (`src/kernels/vision.rs`) against the CPU ViT oracle.
//!
//! The reference is the tower itself, not a hand-written expectation:
//! `ojas_cpu::cpu_math::{layernorm, gelu}` and `ojas_cpu::cpu_vit::{patchify,
//! merge_permutation, mrope_positions, vision_rope, resize_position_embeddings}` are the
//! functions the CPU tower runs, so a disagreement here is a real GPU/CPU divergence. Shapes
//! are surya-2's: width 768, 12 heads x 64, patch 16, 2x2 merge, a 64x48-patch page (1024x768
//! px, 3072 tokens), plus one odd size per kernel. Activations are scaled into the 1500-3000
//! range the real residual stream reaches.
//!
//! Every output buffer carries guard regions on both sides, so an over-store is
//! caught even when the numbers agree.
//!
//! ```text
//! SP=~/.local/lib/python3.10/site-packages/nvidia
//! LD_LIBRARY_PATH=$SP/cuda_nvrtc/lib:$SP/cublas/lib OJAS_CUDA_INCLUDE=$SP/cuda_runtime/include \
//!   cargo test --release -p ojas-cuda --test surya_vision_kernels -- --ignored --nocapture
//! ```

use ojas_cuda::kernels;

const VISION: &[&str] = &[
    "vit_layernorm_m", "vit_gelu", "vit_patchify", "vit_rope", "vit_qkv_split",
    "vit_merge_permute", "add_rowbias_m", "copy_f32_half",
];

/// Host-only: every vision entry is claimed by CUDA, lives in the `vision` family, and has a
/// Metal twin of the same name, so it counts in the parity ledger (`examples/parity.rs`).
#[test]
fn vision_names_in_parity_ledger() {
    let metal: std::collections::BTreeSet<&str> =
        ojas_metal::kernels::all_names().into_iter().collect();
    for n in VISION {
        assert!(kernels::all_names().any(|k| k == *n), "'{n}' missing from CUDA all_names()");
        assert_eq!(kernels::family_of(n), Some("vision"), "'{n}' not in the vision family");
        assert!(metal.contains(n), "'{n}' has no Metal twin — it would not count as shared");
    }
}

#[cfg(test)]
mod gpu {
    use ojas_core::{Device, KernelRuntime};
    use ojas_cpu::{cpu_math, cpu_vit};
    use ojas_cuda::{CuBuf, CudaGpu};

    // surya-2 tower shape
    const D: usize = 768;
    const HD: usize = 64;
    const NH: usize = 12;
    const P: usize = 16;
    const EPS: f32 = 1e-6;
    const BASE: f32 = 10000.0;
    // the page: 64 x 48 patches
    const PW: usize = 64;
    const PH: usize = 48;

    const GUARD: usize = 64;
    const GUARD_FILL: f32 = 12345.0;

    fn gpu() -> CudaGpu {
        let mut g = CudaGpu::new(0).expect("these tests require a CUDA box: cargo test -- --ignored");
        g.ensure_family("vision").expect("vision family NVRTC");
        g
    }

    /// Deterministic values in [-1, 1). conformance.rs's `tvec` (`>> 9` over 2^23) only ever
    /// yields [-1, 0), which would leave GELU's positive half and sign-dependent bugs untested.
    fn tvec(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// An f32 buffer of `n` values (initialised to `init`, or the guard fill) flanked by guards.
    /// Bind at byte offset `GUARD*4`.
    fn guarded(g: &CudaGpu, init: Option<&[f32]>, n: usize) -> CuBuf {
        let mut v = vec![GUARD_FILL; n + 2 * GUARD];
        if let Some(src) = init {
            v[GUARD..GUARD + n].copy_from_slice(src);
        }
        g.upload(&v)
    }
    const OFF: u64 = (GUARD * 4) as u64;

    fn read_guarded(g: &CudaGpu, label: &str, b: &CuBuf, n: usize) -> Vec<f32> {
        let mut all = vec![0f32; n + 2 * GUARD];
        g.read(b, &mut all);
        assert!(all[..GUARD].iter().all(|&v| v == GUARD_FILL), "{label}: stored BEFORE the output");
        assert!(all[GUARD + n..].iter().all(|&v| v == GUARD_FILL), "{label}: stored PAST the output");
        all[GUARD..GUARD + n].to_vec()
    }

    /// (max abs err, max elementwise rel err, normwise rel err = max abs err / max |want|).
    /// The elementwise error is floored at 1e-3 of max |want| so values crossing zero do not
    /// divide by ~0; it is an upper bound dominated by the smallest outputs, so read normwise.
    fn errs(got: &[f32], want: &[f32]) -> (f32, f32, f32) {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
        let floor = 1e-3 * scale;
        let (mut a, mut r) = (0f32, 0f32);
        for (g, w) in got.iter().zip(want) {
            assert!(g.is_finite(), "non-finite output {g} (want {w})");
            let e = (g - w).abs();
            a = a.max(e);
            r = r.max(e / w.abs().max(floor));
        }
        (a, r, a / scale)
    }

    fn blocks(n: usize, b: usize) -> u32 { n.div_ceil(b) as u32 }

    fn run_layernorm(g: &CudaGpu, x: &[f32], w: &[f32], b: &[f32], m: usize, inplace: bool) -> Vec<f32> {
        let (xd, wd, bd) = (guarded(g, Some(x), m * D), g.upload(w), g.upload(b));
        let od = if inplace { None } else { Some(guarded(g, None, m * D)) };
        let out = od.as_ref().unwrap_or(&xd);
        let enc = g.begin();
        g.dispatch(&enc, "vit_layernorm_m", &[(&xd, OFF), (&wd, 0), (out, OFF), (&bd, 0)],
                   &[D as u32, EPS.to_bits()], [m as u32, 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        read_guarded(g, "vit_layernorm_m", out, m * D)
    }

    fn ln_oracle(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
        x.chunks(D).flat_map(|r| cpu_math::layernorm(r, w, b, EPS)).collect()
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn layernorm_matches_oracle() {
        let g = gpu();
        let w: Vec<f32> = tvec(D, 2).iter().map(|v| 1.0 + 0.5 * v).collect();
        let b: Vec<f32> = tvec(D, 3).iter().map(|v| 0.2 * v).collect();
        for (label, m) in [("64x48 page", PW * PH), ("odd 37 rows", 37)] {
            // residual-stream magnitudes: a per-row offset up to ±2000 plus spread up to ±1000
            let mut x = tvec(m * D, 1 + m as u32);
            for (r, row) in x.chunks_mut(D).enumerate() {
                let off = 2000.0 * tvec(1, 100 + r as u32)[0];
                for v in row.iter_mut() { *v = off + 1000.0 * *v; }
            }
            // row 0: near-constant at a large offset — the eps-inside-sqrt / two-pass row
            for (i, v) in x[..D].iter_mut().enumerate() { *v = 2500.0 + 1e-3 * ((i % 7) as f32 - 3.0); }
            let want = ln_oracle(&x, &w, &b);
            let got = run_layernorm(&g, &x, &w, &b, m, false);
            let (a, r, nw) = errs(&got, &want);
            println!("vit_layernorm_m [{label}, {m}x{D}]: max abs {a:.3e}, max rel {r:.3e}, normwise {nw:.3e}");
            assert!(nw < 1e-5 && r < 1e-2, "vit_layernorm_m {label}: normwise {nw}, rel {r}");
            // near-constant row on its own (its outputs are O(1) too)
            let (a0, r0, nw0) = errs(&got[..D], &want[..D]);
            println!("vit_layernorm_m [{label}, near-constant row]: max abs {a0:.3e}, max rel {r0:.3e}, normwise {nw0:.3e}");
            assert!(nw0 < 1e-4, "near-constant row normwise {nw0}");
            // in place (post_ln runs with out == x)
            let got_ip = run_layernorm(&g, &x, &w, &b, m, true);
            assert_eq!(got_ip, got, "in-place layernorm differs from out-of-place");
        }
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn gelu_matches_oracle() {
        let g = gpu();
        // ffn width 3072 at the page's row count, plus an odd count; values span ±12 so
        // both tails and the clamp region are exercised
        for (label, n) in [("64x48 page x 3072", PW * PH * 3072), ("odd", 100_003)] {
            let x: Vec<f32> = tvec(n, 7 + n as u32).iter().map(|v| 12.0 * v).collect();
            let want: Vec<f32> = x.iter().map(|&v| cpu_math::gelu(v)).collect();
            let (xd, od) = (g.upload(&x), guarded(&g, None, n));
            let enc = g.begin();
            g.dispatch(&enc, "vit_gelu", &[(&xd, 0), (&od, OFF)], &[n as u32],
                       [blocks(n, 256), 1, 1], [256, 1, 1]).unwrap();
            g.submit(enc).unwrap();
            let got = read_guarded(&g, "vit_gelu", &od, n);
            let (a, r, nw) = errs(&got, &want);
            println!("vit_gelu [{label}, n={n}]: max abs {a:.3e}, max rel {r:.3e}, normwise {nw:.3e}");
            assert!(r < 5e-5 && nw < 1e-6, "vit_gelu {label}: rel {r}, normwise {nw}");
        }
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn patchify_matches_oracle() {
        let g = gpu();
        // 1024x768 page; odd: 597x377 (37x23 patches, partial trailing patches dropped)
        for (label, w, h) in [("64x48 page", PW * P, PH * P), ("odd 597x377", 597, 377)] {
            let c = 3;
            // value encodes (c, y, x) exactly, so a wrong nesting is caught positionally
            let img: Vec<f32> = (0..c * h * w)
                .map(|i| { let (ci, r) = (i / (h * w), i % (h * w)); (ci * 1_000_000 + (r / w) * 1000 + r % w) as f32 })
                .collect();
            let want = cpu_vit::patchify(&img, w, h, c, P);
            let n = want.len();
            let (id, od) = (g.upload(&img), guarded(&g, None, n));
            let enc = g.begin();
            g.dispatch(&enc, "vit_patchify", &[(&id, 0), (&od, OFF)],
                       &[w as u32, h as u32, c as u32, P as u32, n as u32],
                       [blocks(n, 256), 1, 1], [256, 1, 1]).unwrap();
            g.submit(enc).unwrap();
            let got = read_guarded(&g, "vit_patchify", &od, n);
            let bad = got.iter().zip(&want).position(|(a, b)| a != b);
            println!("vit_patchify [{label}, {}x{}]: max rel {:.3e} (bit-exact: {})",
                     n / (c * P * P), c * P * P, errs(&got, &want).1, bad.is_none());
            assert!(bad.is_none(), "patchify {label}: first mismatch at {bad:?}");
        }
    }

    fn mpos_desc(pw: usize, ph: usize) -> Vec<u8> {
        let s = (HD / 4) as u32;
        let mut v = vec![s, s, s, s];
        for p in cpu_vit::mrope_positions(pw, ph) {
            v.extend(p.iter().map(|&x| x as u32));
        }
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn rope_matches_oracle() {
        let g = gpu();
        // odd: 38x26 patches (the merge needs even sides; otherwise nothing is a power of two)
        for (label, pw, ph) in [("64x48 page", PW, PH), ("odd 38x26", 38, 26)] {
            let m = pw * ph;
            let x: Vec<f32> = tvec(m * D, 11 + m as u32).iter().map(|v| 40.0 * v).collect();
            let pos = cpu_vit::mrope_positions(pw, ph);
            let mut want = x.clone();
            for (t, row) in want.chunks_mut(D).enumerate() {
                for head in row.chunks_mut(HD) {
                    cpu_vit::vision_rope(head, HD, pos[t], BASE);
                }
            }
            let (vd, md) = (guarded(&g, Some(&x), m * D), g.upload_bytes(&mpos_desc(pw, ph)).unwrap());
            let pairs = m * NH * (HD / 2);
            let enc = g.begin();
            g.dispatch(&enc, "vit_rope", &[(&vd, OFF), (&md, 0)],
                       &[HD as u32, BASE.to_bits(), D as u32, m as u32],
                       [blocks(pairs, 64), 1, 1], [64, 1, 1]).unwrap();
            g.submit(enc).unwrap();
            let got = read_guarded(&g, "vit_rope", &vd, m * D);
            let (a, r, nw) = errs(&got, &want);
            println!("vit_rope [{label}, {m} tok x {NH}x{HD}]: max abs {a:.3e}, max rel {r:.3e}, normwise {nw:.3e}");
            assert!(r < 5e-4 && nw < 1e-6, "vit_rope {label}: rel {r}, normwise {nw}");
        }
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn qkv_split_matches_oracle() {
        let g = gpu();
        for (label, m) in [("64x48 page", PW * PH), ("odd 37 rows", 37)] {
            let qkv: Vec<f32> = tvec(m * 3 * D, 21 + m as u32).iter().map(|v| 3000.0 * v).collect();
            let n = m * D;
            let (qd, a, b, c) = (g.upload(&qkv), guarded(&g, None, n), guarded(&g, None, n), guarded(&g, None, n));
            let enc = g.begin();
            g.dispatch(&enc, "vit_qkv_split", &[(&qd, 0), (&a, OFF), (&b, OFF), (&c, OFF)],
                       &[D as u32, n as u32], [blocks(n, 64), 1, 1], [64, 1, 1]).unwrap();
            g.submit(enc).unwrap();
            for (which, buf) in [(0usize, &a), (1, &b), (2, &c)] {
                let got = read_guarded(&g, "vit_qkv_split", buf, n);
                let want: Vec<f32> = qkv.chunks(3 * D).flat_map(|r| r[which * D..(which + 1) * D].to_vec()).collect();
                assert_eq!(got, want, "qkv_split {label} stream {which}");
            }
            println!("vit_qkv_split [{label}, {m}x3x{D}]: max rel 0 (bit-exact, q/k/v)");
        }
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn merge_permute_matches_oracle() {
        let g = gpu();
        for (label, pw, ph) in [("64x48 page", PW, PH), ("odd 38x26", 38, 26)] {
            let t = pw * ph;
            let src: Vec<f32> = tvec(t * D, 31 + t as u32).iter().map(|v| 3000.0 * v).collect();
            let perm = cpu_vit::merge_permutation(pw, ph);
            let want: Vec<f32> = perm.iter().flat_map(|&s| src[s * D..(s + 1) * D].to_vec()).collect();
            let (sd, dd) = (g.upload(&src), guarded(&g, None, t * D));
            let enc = g.begin();
            g.dispatch(&enc, "vit_merge_permute", &[(&sd, 0), (&dd, OFF)],
                       &[D as u32, pw as u32, (t * D) as u32], [blocks(t * D, 64), 1, 1], [64, 1, 1]).unwrap();
            g.submit(enc).unwrap();
            let got = read_guarded(&g, "vit_merge_permute", &dd, t * D);
            assert_eq!(got, want, "merge_permute {label}");
            println!("vit_merge_permute [{label}, {t}x{D}]: max rel 0 (bit-exact)");
        }
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn add_rowbias_matches_oracle() {
        let g = gpu();
        for (label, m) in [("64x48 page", PW * PH), ("odd 37 rows", 37)] {
            let total = m * D;
            let x: Vec<f32> = tvec(total, 41 + m as u32).iter().map(|v| 2000.0 * v).collect();
            // per-row bias (N = d) and full-tensor add (N = total, the position-embedding use)
            for (mode, bias) in [("N=d", tvec(D, 42)), ("N=total", tvec(total, 43))] {
                let nb = bias.len();
                let want: Vec<f32> = x.iter().enumerate().map(|(i, v)| v + bias[i % nb]).collect();
                let (xd, bd) = (guarded(&g, Some(&x), total), g.upload(&bias));
                let enc = g.begin();
                g.dispatch(&enc, "add_rowbias_m", &[(&xd, OFF), (&bd, 0)],
                           &[nb as u32, total as u32], [blocks(total, 64), 1, 1], [64, 1, 1]).unwrap();
                g.submit(enc).unwrap();
                let got = read_guarded(&g, "add_rowbias_m", &xd, total);
                assert_eq!(got, want, "add_rowbias_m {label} {mode}");
                println!("add_rowbias_m [{label}, {mode}]: max rel 0 (bit-exact)");
            }
        }
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn copy_f32_half_matches_oracle() {
        let g = gpu();
        for (label, n) in [("64x48 page K", PW * PH * D), ("odd", 1001)] {
            let mut x: Vec<f32> = tvec(n, 51 + n as u32).iter().map(|v| 3000.0 * v).collect();
            // ties, subnormals, and the f16 max neighbourhood
            let special = [0.0, -0.0, 1.0 + 1.0 / 2048.0, 1.0 + 3.0 / 2048.0, 6e-8, 3e-5, 65504.0, -65519.0];
            x[..special.len()].copy_from_slice(&special);
            let want: Vec<u16> = x.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
            let xd = g.upload(&x);
            let od = g.alloc_bytes(n * 2 + 2 * GUARD * 4).unwrap();
            let enc = g.begin();
            g.dispatch(&enc, "copy_f32_half", &[(&xd, 0), (&od, OFF)], &[n as u32],
                       [blocks(n, 256), 1, 1], [256, 1, 1]).unwrap();
            g.submit(enc).unwrap();
            let mut raw = vec![0u8; n * 2 + 2 * GUARD * 4];
            g.read_bytes(&od, 0, &mut raw).unwrap();
            assert!(raw[..GUARD * 4].iter().all(|&b| b == 0), "copy_f32_half stored BEFORE");
            assert!(raw[GUARD * 4 + n * 2..].iter().all(|&b| b == 0), "copy_f32_half stored PAST");
            let got: Vec<u16> = raw[GUARD * 4..GUARD * 4 + n * 2]
                .chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let bad = got.iter().zip(&want).position(|(a, b)| a != b);
            assert!(bad.is_none(), "copy_f32_half {label}: first mismatch at {bad:?}");
            println!("copy_f32_half [{label}, n={n}]: bit-exact vs half::f16::from_f32 (RNE)");
        }
    }

    /// The tower's pre-block-0 chain at the page shape: merge_permute -> + patch bias ->
    /// + permuted, resized position embedding -> ln1, against the same sequence on the CPU
    /// (`resize_position_embeddings` from the learned 48x48 grid to 64x48, then permuted).
    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn stem_chain_matches_oracle() {
        let g = gpu();
        let (pw, ph) = (PW, PH);
        let t = pw * ph;
        let h: Vec<f32> = tvec(t * D, 61).iter().map(|v| 1500.0 * v).collect();
        let pbias: Vec<f32> = tvec(D, 62).iter().map(|v| 50.0 * v).collect();
        let pe48: Vec<f32> = tvec(48 * 48 * D, 63).iter().map(|v| 20.0 * v).collect();
        let pe = cpu_vit::resize_position_embeddings(&pe48, D, 48, pw, ph);
        let perm = cpu_vit::merge_permutation(pw, ph);
        let pe_perm: Vec<f32> = perm.iter().flat_map(|&s| pe[s * D..(s + 1) * D].to_vec()).collect();
        let w: Vec<f32> = tvec(D, 64).iter().map(|v| 1.0 + 0.5 * v).collect();
        let b: Vec<f32> = tvec(D, 65).iter().map(|v| 0.2 * v).collect();
        // oracle
        let mut x: Vec<f32> = perm.iter().flat_map(|&s| h[s * D..(s + 1) * D].to_vec()).collect();
        for (i, v) in x.iter_mut().enumerate() { *v += pbias[i % D]; }
        for (i, v) in x.iter_mut().enumerate() { *v += pe_perm[i]; }
        let want = ln_oracle(&x, &w, &b);
        // gpu
        let n = t * D;
        let (hd_, xd, pbd, ped, wd, bd, od) = (g.upload(&h), g.alloc(n), g.upload(&pbias),
            g.upload(&pe_perm), g.upload(&w), g.upload(&b), guarded(&g, None, n));
        let enc = g.begin();
        g.dispatch(&enc, "vit_merge_permute", &[(&hd_, 0), (&xd, 0)],
                   &[D as u32, pw as u32, n as u32], [blocks(n, 64), 1, 1], [64, 1, 1]).unwrap();
        g.dispatch(&enc, "add_rowbias_m", &[(&xd, 0), (&pbd, 0)], &[D as u32, n as u32],
                   [blocks(n, 64), 1, 1], [64, 1, 1]).unwrap();
        g.dispatch(&enc, "add_rowbias_m", &[(&xd, 0), (&ped, 0)], &[n as u32, n as u32],
                   [blocks(n, 64), 1, 1], [64, 1, 1]).unwrap();
        g.dispatch(&enc, "vit_layernorm_m", &[(&xd, 0), (&wd, 0), (&od, OFF), (&bd, 0)],
                   &[D as u32, EPS.to_bits()], [t as u32, 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let got = read_guarded(&g, "stem chain", &od, n);
        let (a, r, nw) = errs(&got, &want);
        println!("stem chain [merge+bias+pe(resized 48->64x48)+ln1, {t}x{D}]: max abs {a:.3e}, max rel {r:.3e}, normwise {nw:.3e}");
        assert!(nw < 1e-5 && r < 1e-2, "stem chain normwise {nw}, rel {r}");
    }
}
