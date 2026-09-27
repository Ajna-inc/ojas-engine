//! Conformance scaffolding for the CUDA backend.
//!
//! Two layers:
//! 1. Host-only manifest guards (run everywhere): name integrity inside the
//!    crate, retired-kernel exclusion, and the cross-backend canonical-name
//!    alignment check against ojas-metal.
//! 2. `#[ignore]`d GPU tests: on a CUDA host (`cargo test -p ojas-cuda --
//!    --ignored`) they NVRTC-compile every family and diff kernel outputs
//!    against CPU oracle values. They compile everywhere but touch a GPU only
//!    when requested.

use ojas_cuda::kernels;

/// Entry names a family's source declares, including the ones a macro generates.
///
/// `learn_gemm_bf16_<ta><tb><bcol>` exists only after the preprocessor expands `LBF_ENTRY`, so
/// a plain text scan of `__global__ void` misses six real kernels. The macro's instantiation
/// list is scanned too.
fn entries_in(src: &str) -> Vec<String> {
    // "__global__ void [__launch_bounds__(..)] name("
    let mut out = vec![];
    for line in src.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("LBF_ENTRY(") else { continue };
        let Some(args) = rest.split(')').next() else { continue };
        let digits: String = args.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() == 3 {
            out.push(format!("learn_gemm_bf16_{digits}"));
        }
    }
    for part in src.split("__global__").skip(1) {
        let part = part.trim_start();
        let part = match part.strip_prefix("void") {
            Some(p) => p.trim_start(),
            None => continue,
        };
        let part = if let Some(p) = part.strip_prefix("__launch_bounds__") {
            match p.find(')') {
                Some(i) => p[i + 1..].trim_start(),
                None => continue,
            }
        } else {
            part
        };
        if let Some(i) = part.find('(') {
            let name = part[..i].trim();
            // the macro's own template line ("learn_gemm_bf16_##TA##TB##BC") is not an entry;
            // its instantiations were collected above
            if !name.contains("##") {
                out.push(name.to_string());
            }
        }
    }
    out
}

// ---------- layer 1: manifest guards (always run) ----------

/// Parameter lists of every `__global__` entry in a family's source, as `(name, params)`.
fn signatures_in(src: &str) -> Vec<(String, String)> {
    let mut out = vec![];
    for part in src.split("__global__").skip(1) {
        let part = part.trim_start();
        let Some(part) = part.strip_prefix("void").map(str::trim_start) else { continue };
        let part = if let Some(p) = part.strip_prefix("__launch_bounds__") {
            match p.find(')') { Some(i) => p[i + 1..].trim_start(), None => continue }
        } else { part };
        let (Some(open), Some(close)) = (part.find('('), part.find(") {")) else { continue };
        if close < open { continue }
        let name = part[..open].trim().to_string();
        if name.contains("##") { continue }          // the LBF_ENTRY template, not an entry
        out.push((name, part[open + 1..close].to_string()));
    }
    out
}

/// Every pointer parameter must come before every scalar parameter.
///
/// `KernelRuntime::dispatch` takes `bufs` then `consts` and the CUDA launcher appends them in
/// that order, so a kernel declaring a pointer after an `int` cannot be dispatched through the
/// seam: the buffer lands in the scalar's slot and the launch reads garbage. The device-control
/// entries (`store_kv_g`, `attention_short_g`, `attention_part_g`, `attention_flash_tc_g`,
/// `store_kv_m_g`, `embed_q4_g`, `rope_partial_g`, `rope_partial_m_g`) are the ones at risk,
/// since their `ctl`/`pctl` pointer reads naturally as a trailing argument.
#[test]
fn pointers_precede_scalars_in_every_signature() {
    let mut bad = vec![];
    for (family, src) in kernels::families() {
        for (name, params) in signatures_in(src) {
            let mut seen_scalar: Option<String> = None;
            for p in params.split(',') {
                let p = p.trim();
                if p.is_empty() { continue }
                if p.contains('*') {
                    if let Some(scalar) = &seen_scalar {
                        bad.push(format!(
                            "{family}::{name}: pointer '{p}' follows scalar '{scalar}'"));
                        break;
                    }
                } else {
                    seen_scalar = Some(p.to_string());
                }
            }
        }
    }
    assert!(bad.is_empty(),
            "{} entries cannot be dispatched through KernelRuntime:\n  {}",
            bad.len(), bad.join("\n  "));
}

#[test]
fn every_claimed_name_exists_exactly_once() {
    let mut all: Vec<String> = vec![];
    for (family, src) in kernels::families() {
        let found = entries_in(src);
        for name in kernels::all_names().filter(|n| kernels::family_of(n) == Some(family)) {
            let n = found.iter().filter(|f| f.as_str() == name).count();
            assert_eq!(n, 1, "'{name}' must appear exactly once in family '{family}' (found {n})");
        }
        all.extend(found);
    }
    // no duplicate entry names across families (one NVRTC unit per family, but
    // the canonical namespace is global)
    let mut seen = std::collections::HashSet::new();
    for n in &all {
        assert!(seen.insert(n.clone()), "duplicate kernel entry across families: {n}");
    }
    // and nothing unlisted hides in the sources
    for n in &all {
        assert!(
            kernels::all_names().any(|k| k == n),
            "kernel '{n}' present in source but missing from NAMES"
        );
    }
}

#[test]
fn retired_kernels_stay_retired() {
    for (_, src) in kernels::families() {
        for dead in kernels::RETIRED {
            assert!(
                !entries_in(src).iter().any(|n| n == dead),
                "retired kernel '{dead}' re-appeared"
            );
        }
    }
}

/// Cross-backend canonical-name guard: every ojas-cuda entry that claims to be Metal-aligned
/// must resolve in the Metal kernel table too. CUDA-only entries (graphs/_g, W4A8 pipeline,
/// kv-shift) are exempt.
///
/// This runs on every host: `ojas-metal`'s `kernels` module compiles everywhere, only its device
/// half is macOS-only.
#[test]
fn aligned_names_resolve_in_metal() {
    // aligned = same-name kept; renamed-onto-metal = rename targets.
    const ALIGNED: &[&str] = &[
        "rmsnorm", "silu_mul", "swiglu", "rope", "add_inplace", "mul_scalar",
        "copy_buf", "store_kv", "argmax", "embed_q4", "qgate_split",
        "gate_mul_sigmoid", "qk_rmsnorm", "gated_rmsnorm", "ssm_ab",
        "conv1d_decode", "conv1d_prefill", "deltanet_fused", "attention_short",
        "gemv_q4",
        // renamed onto existing Metal entries:
        "rmsnorm_m", "store_kv_m", "attention_m_short", "attention_merge",
    ];
    for name in ALIGNED {
        assert!(
            kernels::all_names().any(|n| n == *name),
            "'{name}' missing from the CUDA manifest"
        );
        assert!(
            ojas_metal::kernels::source_of(name).is_some(),
            "'{name}' claimed Metal-aligned but not found in ojas-metal kernels"
        );
    }
}


/// Parity ratchet: the number of canonical entries present on both backends may go up, never
/// down. `cargo run -p ojas-cuda --example parity` prints the full ledger when this fails; the
/// floor here catches a CUDA entry renamed away from its Metal twin, which splits the namespace.
///
/// Raise the floor when kernels are ported; lowering it needs a reason in the commit message.
#[test]
fn cross_backend_parity_does_not_regress() {
    const FLOOR: usize = 176;
    let metal: std::collections::BTreeSet<&str> =
        ojas_metal::kernels::all_names().into_iter().collect();
    let shared = kernels::all_names().filter(|n| metal.contains(n)).count();
    assert!(
        shared >= FLOOR,
        "entries shared with Metal fell to {shared} (floor {FLOOR}); run\n    \
         cargo run -p ojas-cuda --example parity\n\
         to see which ones went missing"
    );
    // a floor far below the truth stops ratcheting
    assert!(
        shared < FLOOR + 25,
        "parity is now {shared}, well past the floor of {FLOOR} — raise FLOOR in this test"
    );
}

// ---------- layer 2: GPU conformance (ignored; CUDA box only) ----------

#[cfg(test)]
mod gpu {
    use ojas_core::{Device, KernelRuntime};
    use ojas_cuda::CudaGpu;

    fn gpu() -> CudaGpu {
        CudaGpu::new(0).expect("these tests require a CUDA box: cargo test -- --ignored")
    }

    fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs() / w.abs().max(1e-6))
            .fold(0.0, f32::max)
    }

    /// Deterministic pseudo-random test vector (no rand dep).
    fn tvec(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 9) as f32 / (1 << 23) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn families_compile() {
        let mut g = gpu();
        // every family the crate ships: a hardcoded list would stop covering new ones
        let families: Vec<&str> = ojas_cuda::kernels::families().map(|(f, _)| f).collect();
        for fam in families {
            g.ensure_family(fam).unwrap_or_else(|e| panic!("family '{fam}' failed NVRTC: {e:?}"));
        }
        for name in ojas_cuda::kernels::all_names() {
            assert!(g.has_kernel(name), "'{name}' did not load");
        }
    }

    /// int8 weights and a per-row scale, the layout the Q8 family reads.
    fn q8_rows(rows: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let mut w = Vec::with_capacity(rows * k);
        for i in 0..rows * k {
            let v = (tvec(1, seed.wrapping_add(i as u32))[0] * 127.0).round().clamp(-127.0, 127.0);
            w.push(v as i8 as u8);
        }
        let scale: Vec<f32> =
            (0..rows).map(|r| 0.01 + 0.001 * tvec(1, seed ^ (r as u32 + 7)) [0].abs()).collect();
        (w, scale)
    }

    fn q8_ref(w: &[u8], scale: &[f32], x: &[f32], k: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|r| {
                let acc: f32 =
                    (0..k).map(|i| (w[r * k + i] as i8) as f32 * x[i]).sum();
                acc * scale[r]
            })
            .collect()
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn gemv_q8_matches_oracle() {
        let mut g = gpu();
        g.ensure_family("gemv_q8").unwrap();
        let (k, n) = (512usize, 256usize);
        let x = tvec(k, 11);
        let (w, scale) = q8_rows(n, k, 12);
        let want = q8_ref(&w, &scale, &x, k, n);
        let (xd, wd, sd, yd) =
            (g.upload(&x), g.upload_bytes(&w).unwrap(), g.upload(&scale), g.alloc(n));
        let enc = g.begin();
        g.dispatch(&enc, "gemv_q8", &[(&xd, 0), (&wd, 0), (&yd, 0), (&sd, 0)],
                   &[k as u32, n as u32], [(n as u32).div_ceil(8), 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; n];
        g.read(&yd, &mut got);
        assert!(max_rel_err(&got, &want) < 1e-4, "gemv_q8 err {}", max_rel_err(&got, &want));
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn gemv_q8_ksplit_matches_plain() {
        let mut g = gpu();
        g.ensure_family("gemv_q8").unwrap();
        // the shape split-K exists for: long K, few rows
        let (k, n) = (4864usize, 96usize);
        let x = tvec(k, 21);
        let (w, scale) = q8_rows(n, k, 22);
        let want = q8_ref(&w, &scale, &x, k, n);
        let (xd, wd, sd, yd) =
            (g.upload(&x), g.upload_bytes(&w).unwrap(), g.upload(&scale), g.alloc(n));
        let enc = g.begin();
        // one block per row, 8 warps, shared memory for the per-warp partials
        g.dispatch(&enc, "gemv_q8_ksplit", &[(&xd, 0), (&wd, 0), (&yd, 0), (&sd, 0)],
                   &[k as u32, n as u32], [n as u32, 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; n];
        g.read(&yd, &mut got);
        assert!(max_rel_err(&got, &want) < 1e-3, "ksplit err {}", max_rel_err(&got, &want));
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn ffn_gu_q8_matches_oracle() {
        let mut g = gpu();
        g.ensure_family("gemv_q8").unwrap();
        let (k, n) = (512usize, 128usize);
        let x = tvec(k, 31);
        let (wg, sg) = q8_rows(n, k, 32);
        let (wu, su) = q8_rows(n, k, 33);
        let gate = q8_ref(&wg, &sg, &x, k, n);
        let up = q8_ref(&wu, &su, &x, k, n);
        // act 0 = SiLU
        let want: Vec<f32> =
            gate.iter().zip(&up).map(|(gv, uv)| gv / (1.0 + (-gv).exp()) * uv).collect();
        let (xd, gd, ud, od) = (g.upload(&x), g.upload_bytes(&wg).unwrap(),
                                g.upload_bytes(&wu).unwrap(), g.alloc(n));
        let (sgd, sud) = (g.upload(&sg), g.upload(&su));
        let enc = g.begin();
        g.dispatch(&enc, "ffn_gu_q8",
                   &[(&xd, 0), (&gd, 0), (&ud, 0), (&od, 0), (&sgd, 0), (&sud, 0)],
                   &[k as u32, n as u32, 0], [(n as u32).div_ceil(8), 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; n];
        g.read(&od, &mut got);
        assert!(max_rel_err(&got, &want) < 1e-3, "ffn_gu_q8 err {}", max_rel_err(&got, &want));
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn embed_q8_matches_oracle() {
        let mut g = gpu();
        g.ensure_family("gemv_q8").unwrap();
        let (vocab, d, token) = (64usize, 256usize, 37usize);
        let (table, scale) = q8_rows(vocab, d, 41);
        let want: Vec<f32> =
            (0..d).map(|i| (table[token * d + i] as i8) as f32 * scale[token]).collect();
        let (td, sd, od) = (g.upload_bytes(&table).unwrap(), g.upload(&scale), g.alloc(d));
        let enc = g.begin();
        g.dispatch(&enc, "embed_q8", &[(&td, 0), (&sd, 0), (&od, 0)],
                   &[token as u32, d as u32], [4, 1, 1], [128, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; d];
        g.read(&od, &mut got);
        assert!(max_rel_err(&got, &want) < 1e-5, "embed_q8 err {}", max_rel_err(&got, &want));
    }

    /// GGUF block_q8_0 rows: 34 bytes per 32 values (f16 scale, then 32 int8).
    fn q80_rows(rows: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let nblk = k / 32;
        let mut bytes = vec![0u8; rows * nblk * 34];
        let mut plain = vec![0f32; rows * k];
        for r in 0..rows {
            for b in 0..nblk {
                let d = 0.002 + 0.001 * tvec(1, seed ^ ((r * nblk + b) as u32))[0].abs();
                let off = (r * nblk + b) * 34;
                bytes[off..off + 2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                for i in 0..32 {
                    let q = (tvec(1, seed.wrapping_add((r * k + b * 32 + i) as u32))[0] * 127.0)
                        .round().clamp(-127.0, 127.0);
                    bytes[off + 2 + i] = (q as i8) as u8;
                    plain[r * k + b * 32 + i] = q * half::f16::from_f32(d).to_f32();
                }
            }
        }
        (bytes, plain)
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn moe_topk_matches_oracle() {
        let mut g = gpu();
        g.ensure_family("moe").unwrap();
        let (e, ksel) = (64usize, 8usize);
        let lg = tvec(e, 51);
        // oracle: softmax, then ksel largest, renormalised over the chosen ones
        let mx = lg.iter().cloned().fold(f32::MIN, f32::max);
        let ex: Vec<f32> = lg.iter().map(|v| (v - mx).exp()).collect();
        let sum: f32 = ex.iter().sum();
        let mut probs: Vec<(usize, f32)> = ex.iter().enumerate().map(|(i, v)| (i, v / sum)).collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let chosen: Vec<(usize, f32)> = probs[..ksel].to_vec();
        let wsum: f32 = chosen.iter().map(|c| c.1).sum();
        let (lgd, idxd, wd) = (g.upload(&lg), g.alloc(ksel), g.alloc(ksel));
        let enc = g.begin();
        g.dispatch(&enc, "moe_topk", &[(&lgd, 0), (&idxd, 0), (&wd, 0)],
                   &[e as u32, ksel as u32], [1, 1, 1], [32, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut gw = vec![0.0; ksel];
        g.read(&wd, &mut gw);
        let mut raw = vec![0u8; ksel * 4];
        g.read_bytes(&idxd, 0, &mut raw).unwrap();
        let gi: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        for j in 0..ksel {
            assert_eq!(gi[j] as usize, chosen[j].0, "expert {j}: gpu {} want {}", gi[j], chosen[j].0);
            let want = chosen[j].1 / wsum;
            assert!((gw[j] - want).abs() < 1e-5, "weight {j}: gpu {} want {want}", gw[j]);
        }
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn moe_expert_ffn_matches_oracle() {
        let mut g = gpu();
        g.ensure_family("moe").unwrap();
        let (k, n, experts, ksel) = (256usize, 64usize, 8usize, 2usize);
        let x = tvec(k, 61);
        let (gbytes, gplain) = q80_rows(experts * n, k, 62);
        let (ubytes, uplain) = q80_rows(experts * n, k, 63);
        let (dbytes, dplain) = q80_rows(experts * k, n, 64);
        let idx: Vec<u32> = vec![3, 6];
        let wgt = vec![0.7f32, 0.3f32];

        // oracle: per selected expert, silu(gate)*up, then the weighted down-projection
        let mut want = vec![0f32; k];
        for (j, e) in idx.iter().enumerate() {
            let base = (*e as usize) * n;
            let act: Vec<f32> = (0..n).map(|r| {
                let gv: f32 = (0..k).map(|i| gplain[(base + r) * k + i] * x[i]).sum();
                let uv: f32 = (0..k).map(|i| uplain[(base + r) * k + i] * x[i]).sum();
                gv / (1.0 + (-gv).exp()) * uv
            }).collect();
            let dbase = (*e as usize) * k;
            for (r, w) in want.iter_mut().enumerate() {
                let dv: f32 = (0..n).map(|i| dplain[(dbase + r) * n + i] * act[i]).sum();
                *w += wgt[j] * dv;
            }
        }

        let idx_bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (xd, gd, ud, dd) = (g.upload(&x), g.upload_bytes(&gbytes).unwrap(),
                                g.upload_bytes(&ubytes).unwrap(), g.upload_bytes(&dbytes).unwrap());
        let (idxd, wgtd) = (g.upload_bytes(&idx_bytes).unwrap(), g.upload(&wgt));
        let actd = g.alloc(ksel * n);
        let outd = g.upload(&vec![0f32; k]);
        let enc = g.begin();
        g.dispatch(&enc, "moe_gu_q80", &[(&xd, 0), (&gd, 0), (&ud, 0), (&actd, 0), (&idxd, 0)],
                   &[k as u32, n as u32], [(n as u32).div_ceil(8), ksel as u32, 1], [256, 1, 1]).unwrap();
        g.dispatch(&enc, "moe_down_q80",
                   &[(&actd, 0), (&dd, 0), (&outd, 0), (&idxd, 0), (&wgtd, 0)],
                   &[n as u32, k as u32, ksel as u32],
                   [(k as u32).div_ceil(8 * 4), 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; k];
        g.read(&outd, &mut got);
        let err = max_rel_err(&got, &want);
        assert!(err < 2e-3, "moe expert ffn err {err}");
    }

    /// IQ4_XS rows (136 bytes per 256 values) plus the values this repo's own reader decodes
    /// them to, so the GPU is checked against `ojas-formats` rather than a hand reading of the
    /// format. The nibble order is the part that is easy to get subtly wrong.
    fn iq4xs_rows(rows: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        assert!(k % 256 == 0);
        let nblk = k / 256;
        let mut bytes = vec![0u8; rows * nblk * 136];
        for i in 0..bytes.len() {
            bytes[i] = (tvec(1, seed.wrapping_add(i as u32))[0].abs() * 255.0) as u8;
        }
        // keep d small and positive so the reference and the kernel stay in a sane range
        for r in 0..rows {
            for b in 0..nblk {
                let off = (r * nblk + b) * 136;
                let d = half::f16::from_f32(0.01 + 0.002 * ((r + b) % 7) as f32);
                bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
            }
        }
        let plain_bytes = ojas_formats::gguf::dequant_to_f16(&bytes, 23, rows * k);
        let plain: Vec<f32> = plain_bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        (bytes, plain)
    }

    /// IQ4_NL rows (18 bytes per 32 values), decoded by the same reader.
    fn iq4nl_rows(rows: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        assert!(k % 32 == 0);
        let nblk = k / 32;
        let mut bytes = vec![0u8; rows * nblk * 18];
        for i in 0..bytes.len() {
            bytes[i] = (tvec(1, seed.wrapping_add(i as u32))[0].abs() * 255.0) as u8;
        }
        for r in 0..rows {
            for b in 0..nblk {
                let off = (r * nblk + b) * 18;
                let d = half::f16::from_f32(0.01 + 0.002 * ((r + b) % 5) as f32);
                bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
            }
        }
        let plain_bytes = ojas_formats::gguf::dequant_to_f16(&bytes, 20, rows * k);
        let plain: Vec<f32> = plain_bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        (bytes, plain)
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn moe_iq_expert_ffn_matches_reader() {
        let mut g = gpu();
        g.ensure_family("moe_iq").unwrap();
        let (k, n, experts, ksel) = (512usize, 256usize, 8usize, 2usize);
        let x = tvec(k, 71);
        let (gbytes, gplain) = iq4xs_rows(experts * n, k, 72);
        let (ubytes, uplain) = iq4xs_rows(experts * n, k, 73);
        let (dbytes, dplain) = iq4nl_rows(experts * k, n, 74);
        let idx: Vec<u32> = vec![1, 5];
        let wgt = vec![0.6f32, 0.4f32];

        let mut want = vec![0f32; k];
        for (j, e) in idx.iter().enumerate() {
            let base = (*e as usize) * n;
            let act: Vec<f32> = (0..n).map(|r| {
                let gv: f32 = (0..k).map(|i| gplain[(base + r) * k + i] * x[i]).sum();
                let uv: f32 = (0..k).map(|i| uplain[(base + r) * k + i] * x[i]).sum();
                gv / (1.0 + (-gv).exp()) * uv
            }).collect();
            let dbase = (*e as usize) * k;
            for (r, w) in want.iter_mut().enumerate() {
                let dv: f32 = (0..n).map(|i| dplain[(dbase + r) * n + i] * act[i]).sum();
                *w += wgt[j] * dv;
            }
        }

        let idx_bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (xd, gd, ud, dd) = (g.upload(&x), g.upload_bytes(&gbytes).unwrap(),
                                g.upload_bytes(&ubytes).unwrap(), g.upload_bytes(&dbytes).unwrap());
        let (idxd, wgtd) = (g.upload_bytes(&idx_bytes).unwrap(), g.upload(&wgt));
        let actd = g.alloc(ksel * n);
        let outd = g.upload(&vec![0f32; k]);
        let enc = g.begin();
        g.dispatch(&enc, "moe_gu_iq4xs", &[(&xd, 0), (&gd, 0), (&ud, 0), (&actd, 0), (&idxd, 0)],
                   &[k as u32, n as u32], [(n as u32).div_ceil(8), ksel as u32, 1], [256, 1, 1]).unwrap();
        g.dispatch(&enc, "moe_down_iq4nl",
                   &[(&actd, 0), (&dd, 0), (&outd, 0), (&idxd, 0), (&wgtd, 0)],
                   &[n as u32, k as u32, ksel as u32],
                   [(k as u32).div_ceil(8), 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; k];
        g.read(&outd, &mut got);
        // The reference rounds every weight through f16 (`dequant_to_f16`) while the kernel
        // keeps f32, so per-element relative error explodes wherever a dot product lands near
        // zero. Compare the largest absolute deviation against the largest value in the vector.
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let max_abs = got.iter().zip(&want).fold(0f32, |m, (g, w)| m.max((g - w).abs()));
        let err = max_abs / scale;
        assert!(err < 5e-3, "moe iq expert ffn: max |Δ| {max_abs} over peak {scale} = {err}");
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn gemv_q80_dp4a_matches_reader() {
        let mut g = gpu();
        g.ensure_family("gemv_q8").unwrap();
        g.ensure_family("gemv").unwrap();
        let (k, n) = (1024usize, 128usize);
        let x = tvec(k, 81);
        // block_q8_0 weights, and the values ojas-formats decodes them to
        let nblk = k / 32;
        let mut wbytes = vec![0u8; n * nblk * 34];
        for r in 0..n {
            for b in 0..nblk {
                let off = (r * nblk + b) * 34;
                let d = half::f16::from_f32(0.004 + 0.001 * ((r + b) % 5) as f32);
                wbytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
                for i in 0..32 {
                    let q = (tvec(1, 82u32.wrapping_add((r * k + b * 32 + i) as u32))[0] * 127.0)
                        .round().clamp(-127.0, 127.0);
                    wbytes[off + 2 + i] = (q as i8) as u8;
                }
            }
        }
        let plain_bytes = ojas_formats::gguf::dequant_to_f16(&wbytes, 8, n * k);
        let plain: Vec<f32> = plain_bytes.chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
        let want: Vec<f32> = (0..n)
            .map(|r| (0..k).map(|i| plain[r * k + i] * x[i]).sum())
            .collect();

        let (xd, wd, yd) = (g.upload(&x), g.upload_bytes(&wbytes).unwrap(), g.alloc(n));
        let (q8d, d8d, d8sumd) = (g.alloc_bytes(k).unwrap(), g.alloc(nblk), g.alloc(nblk));
        let enc = g.begin();
        g.dispatch(&enc, "quantize_q8_1", &[(&xd, 0), (&q8d, 0), (&d8d, 0), (&d8sumd, 0)],
                   &[k as u32], [nblk as u32, 1, 1], [32, 1, 1]).unwrap();
        g.dispatch(&enc, "gemv_q80_dp4a", &[(&wd, 0), (&q8d, 0), (&d8d, 0), (&yd, 0)],
                   &[k as u32, n as u32], [(n as u32).div_ceil(8), 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; n];
        g.read(&yd, &mut got);
        // activations are int8-quantised per 32 values, so this is a quantised dot product
        // against an f32 reference: judge it on the vector peak, not per element
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let max_abs = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(max_abs / scale < 1e-2, "gemv_q80_dp4a: {max_abs} over peak {scale}");
    }

    /// Build `n * k` values as GGUF `block_q8_0` and return (bytes, the f32 values a reader
    /// decodes them to). Every dp4a test below compares against the reader rather than the
    /// pre-quantisation floats, so a disagreement means the kernel misread the format.
    fn q80_weights(n: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let nblk = k / 32;
        let mut bytes = vec![0u8; n * nblk * 34];
        for r in 0..n {
            for b in 0..nblk {
                let off = (r * nblk + b) * 34;
                let d = half::f16::from_f32(0.004 + 0.001 * ((r + b) % 5) as f32);
                bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
                for i in 0..32 {
                    let q = (tvec(1, seed.wrapping_add((r * k + b * 32 + i) as u32))[0] * 127.0)
                        .round().clamp(-127.0, 127.0);
                    bytes[off + 2 + i] = (q as i8) as u8;
                }
            }
        }
        let f16 = ojas_formats::gguf::dequant_to_f16(&bytes, 8, n * k);
        let vals = f16.chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
        (bytes, vals)
    }

    /// Peak-normalised deviation. The activation is int8-quantised per 32 values, so a
    /// per-element relative error explodes wherever a dot lands near zero; the vector peak is
    /// what the next layer sees.
    fn peak_err(got: &[f32], want: &[f32]) -> f32 {
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale
    }

    /// bias, accumulate, fused qkv and fused gate/up: the four dp4a entries a dense decode step
    /// needs beyond the plain matvec. One test, because they share the weight set.
    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn gemv_q80_dp4a_variants_match_reader() {
        let mut g = gpu();
        g.ensure_family("gemv_q8").unwrap();
        g.ensure_family("gemv").unwrap();
        let (k, n) = (1024usize, 128usize);
        let nblk = k / 32;
        let x = tvec(k, 81);
        let (wb, wv) = q80_weights(n, k, 82);
        let dot = |r: usize| -> f32 { (0..k).map(|i| wv[r * k + i] * x[i]).sum() };

        let xd = g.upload(&x);
        let wd = g.upload_bytes(&wb).unwrap();
        let (q8d, d8d, d8sumd) = (g.alloc_bytes(k).unwrap(), g.alloc(nblk), g.alloc(nblk));
        let quant = |g: &CudaGpu, enc: &<CudaGpu as ojas_core::Device>::Enc| {
            g.dispatch(enc, "quantize_q8_1", &[(&xd, 0), (&q8d, 0), (&d8d, 0), (&d8sumd, 0)],
                       &[k as u32], [nblk as u32, 1, 1], [32, 1, 1]).unwrap();
        };
        let grid = [(n as u32).div_ceil(8), 1, 1];

        // ---- bias
        let bias = tvec(n, 83);
        let want: Vec<f32> = (0..n).map(|r| dot(r) + bias[r]).collect();
        let (bd, yd) = (g.upload(&bias), g.alloc(n));
        let enc = g.begin();
        quant(&g, &enc);
        g.dispatch(&enc, "gemv_q80_dp4a_bias",
                   &[(&wd, 0), (&q8d, 0), (&d8d, 0), (&yd, 0), (&bd, 0)],
                   &[k as u32, n as u32], grid, [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; n];
        g.read(&yd, &mut got);
        assert!(peak_err(&got, &want) < 1e-2, "bias: {}", peak_err(&got, &want));

        // ---- accumulate: y must be the prior contents plus the dot, not overwritten
        let seed_y = tvec(n, 84);
        let want: Vec<f32> = (0..n).map(|r| seed_y[r] + dot(r)).collect();
        let yd = g.upload(&seed_y);
        let enc = g.begin();
        quant(&g, &enc);
        g.dispatch(&enc, "gemv_q80_dp4a_accum", &[(&wd, 0), (&q8d, 0), (&d8d, 0), (&yd, 0)],
                   &[k as u32, n as u32], grid, [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; n];
        g.read(&yd, &mut got);
        assert!(peak_err(&got, &want) < 1e-2, "accum: {}", peak_err(&got, &want));

        // ---- fused qkv, GQA-shaped (Nkv < Nq) and with bias, against three separate matrices
        let (nq, nkv) = (128usize, 32usize);
        let (kb, kv_vals) = q80_weights(nkv, k, 85);
        let (vb, vv_vals) = q80_weights(nkv, k, 86);
        let qkv_bias = tvec(nq + 2 * nkv, 87);
        let (kbd, vbd, qbd) = (g.upload_bytes(&kb).unwrap(), g.upload_bytes(&vb).unwrap(),
                               g.upload(&qkv_bias));
        let (qo, ko, vo) = (g.alloc(nq), g.alloc(nkv), g.alloc(nkv));
        let enc = g.begin();
        quant(&g, &enc);
        g.dispatch(&enc, "qkv_q80_dp4a",
                   &[(&wd, 0), (&kbd, 0), (&vbd, 0), (&q8d, 0), (&d8d, 0),
                     (&qo, 0), (&ko, 0), (&vo, 0), (&qbd, 0)],
                   &[k as u32, nq as u32, nkv as u32],
                   [((nq + 2 * nkv) as u32).div_ceil(8), 1, 1], [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        for (label, buf, vals, rows, boff) in [
            ("q", &qo, &wv, nq, 0usize),
            ("k", &ko, &kv_vals, nkv, nq),
            ("v", &vo, &vv_vals, nkv, nq + nkv),
        ] {
            let want: Vec<f32> = (0..rows)
                .map(|r| (0..k).map(|i| vals[r * k + i] * x[i]).sum::<f32>() + qkv_bias[boff + r])
                .collect();
            let mut got = vec![0.0; rows];
            g.read(buf, &mut got);
            assert!(peak_err(&got, &want) < 1e-2, "qkv {label}: {}", peak_err(&got, &want));
        }

        // ---- fused gate/up SwiGLU
        let (ub, uv) = q80_weights(n, k, 88);
        let want: Vec<f32> = (0..n).map(|r| {
            let gt = dot(r);
            let up: f32 = (0..k).map(|i| uv[r * k + i] * x[i]).sum();
            gt / (1.0 + (-gt).exp()) * up
        }).collect();
        let (ud, od) = (g.upload_bytes(&ub).unwrap(), g.alloc(n));
        let enc = g.begin();
        quant(&g, &enc);
        g.dispatch(&enc, "ffn_gu_q80_dp4a",
                   &[(&wd, 0), (&ud, 0), (&q8d, 0), (&d8d, 0), (&od, 0)],
                   &[k as u32, n as u32, 0], grid, [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; n];
        g.read(&od, &mut got);
        assert!(peak_err(&got, &want) < 2e-2, "ffn_gu: {}", peak_err(&got, &want));
    }

    /// `embed_q80` gathers a row out of `block_q8_0` storage.
    ///
    /// Not bit-for-bit against the reader: `dequant_to_f16` rounds each `d * q` to f16 while the
    /// kernel keeps the f32 product, so the kernel is the more accurate of the two. The check is
    /// that rounding the kernel's output to f16 reproduces the reader exactly, which still
    /// catches a misread block layout (wrong scale, wrong stride, swapped nibble halves) without
    /// requiring the kernel to discard precision.
    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn embed_q80_matches_reader() {
        let mut g = gpu();
        g.ensure_family("gemv_q8").unwrap();
        let (vocab, d) = (64usize, 512usize);
        let (bytes, vals) = q80_weights(vocab, d, 90);
        let (td, od) = (g.upload_bytes(&bytes).unwrap(), g.alloc(d));
        for token in [0usize, 7, 63] {
            let enc = g.begin();
            g.dispatch(&enc, "embed_q80", &[(&td, 0), (&od, 0)], &[token as u32, d as u32],
                       [8, 1, 1], [128, 1, 1]).unwrap();
            g.submit(enc).unwrap();
            let mut got = vec![0.0; d];
            g.read(&od, &mut got);
            let want = &vals[token * d..(token + 1) * d];
            let got16: Vec<f32> = got.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
            assert_eq!(got16.as_slice(), want, "embed_q80 token {token}");
        }
    }

    /// The seven IQ codebook formats, against `ojas-formats`' own dequantiser.
    ///
    /// Flash Next's streamed experts ship as IQ3_S, so the model the memory figure was measured
    /// on depends on these.
    ///
    /// `synth::blocks` is the right input: for these formats every bit pattern is a legal block
    /// (grid indices are masked into range, sign indices are 7 bits into a 128-entry table), so
    /// random data sweeps the entire codebook, while a real tensor touches only the entries it
    /// happens to need. A transposed grid or an off-by-one in the sign table hides behind real
    /// weights and cannot hide behind these.
    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn gemv_nat_iq_matches_reader() {
        let mut g = gpu();
        g.ensure_family("gemv_iq").unwrap();
        // (GGUF type, tag) for every format whose decoder indexes a codebook
        let cases: &[(u32, &str)] = &[
            (16, "iq2xxs"), (17, "iq2xs"), (22, "iq2s"),
            (18, "iq3xxs"), (21, "iq3s"),
            (19, "iq1s"), (29, "iq1m"),
        ];
        let (k, n) = (512usize, 32usize);
        let x = tvec(k, 4242);
        let xd = g.upload(&x);

        for (ty, tag) in cases {
            let (bb, wpb) = ojas_formats::synth::block_shape(*ty).unwrap();
            let nblk = k / wpb;
            let bytes = ojas_formats::synth::blocks(*ty, n * nblk);
            assert_eq!(bytes.len(), n * nblk * bb);

            // the oracle: the same bytes through the reader, then a plain f32 dot
            let f16 = ojas_formats::gguf::dequant_to_f16(&bytes, *ty, n * k);
            let w: Vec<f32> = f16.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
            let want: Vec<f32> = (0..n)
                .map(|r| (0..k).map(|i| w[r * k + i] * x[i]).sum())
                .collect();
            assert!(want.iter().any(|v| v.abs() > 1e-3),
                    "{tag}: oracle is all zeros — the synthetic blocks decoded to nothing");

            let wd = g.upload_bytes(&bytes).unwrap();
            let yd = g.alloc(n);
            let enc = g.begin();
            g.dispatch(&enc, &format!("gemv_nat_{tag}"), &[(&xd, 0), (&wd, 0), (&yd, 0)],
                       &[k as u32, n as u32], [(n as u32).div_ceil(4), 1, 1], [128, 1, 1])
                .unwrap_or_else(|e| panic!("{tag}: {e}"));
            g.submit(enc).unwrap();
            let mut got = vec![0.0; n];
            g.read(&yd, &mut got);

            // Peak-normalised: the weights are f16 on the oracle side and f32 in the kernel, so a
            // per-element relative error blows up wherever a dot lands near zero.
            let peak = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let dev = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(dev / peak < 2e-3,
                    "gemv_nat_{tag}: deviation {dev:.5} over peak {peak:.3} = {:.3} %",
                    100.0 * dev / peak);
        }
    }

    /// `moe_gu_iq3s`, the gate-up half of Flash Next's expert FFN (IQ3_S gate-up with IQ4_NL
    /// down; `moe_down_iq4nl` is the other half).
    ///
    /// Checked against `ojas-formats`' dequantiser in both addressing modes: by router id into a
    /// contiguous expert tensor, and by byte offset into a streaming cache arena. The streamed
    /// path is the one the memory figure uses, and it is easy to get right for the first expert
    /// and wrong for the rest.
    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn moe_gu_iq3s_matches_reader() {
        let mut g = gpu();
        g.ensure_family("moe_iq").unwrap();
        let (k, n, experts, ksel) = (512usize, 16usize, 4usize, 2usize);
        let nblk = k / 256;
        let rowbytes = nblk * 110;
        let ebytes = n * rowbytes;

        let x = tvec(k, 31);
        // one contiguous IQ3_S tensor per projection: [experts][n rows][k]
        let gb = ojas_formats::synth::blocks(21, experts * n * nblk);
        let ub = ojas_formats::synth::blocks(21, experts * n * nblk);
        assert_eq!(gb.len(), experts * ebytes);

        let deq = |bytes: &[u8]| -> Vec<f32> {
            ojas_formats::gguf::dequant_to_f16(bytes, 21, experts * n * k)
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        let (gw, uw) = (deq(&gb), deq(&ub));
        let ids: Vec<u32> = (0..ksel).map(|j| ((j * 3 + 1) % experts) as u32).collect();

        let mut want: Vec<f32> = Vec::with_capacity(ksel * n);
        for j in 0..ksel {
            let e = ids[j] as usize;
            for r in 0..n {
                let base = (e * n + r) * k;
                let gt: f32 = (0..k).map(|i| gw[base + i] * x[i]).sum();
                let up: f32 = (0..k).map(|i| uw[base + i] * x[i]).sum();
                want.push(gt / (1.0 + (-gt).exp()) * up);
            }
        }
        assert!(want.iter().any(|v| v.abs() > 1e-4),
                "oracle is all zeros — the synthetic IQ3_S blocks decoded to nothing");

        let xd = g.upload(&x);
        let gd = g.upload_bytes(&gb).unwrap();
        let ud = g.upload_bytes(&ub).unwrap();
        let idxd = g.upload_bytes(
            &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        // 4 rows per warp, 8 warps per block
        let grid = [(n as u32).div_ceil(32), ksel as u32, 1];

        let act = g.alloc(ksel * n);
        let enc = g.begin();
        g.dispatch(&enc, "moe_gu_iq3s",
                   &[(&xd, 0), (&gd, 0), (&ud, 0), (&act, 0), (&idxd, 0)],
                   &[k as u32, n as u32], grid, [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; ksel * n];
        g.read(&act, &mut got);

        let peak = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let dev = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(dev / peak < 5e-3,
                "moe_gu_iq3s: deviation {dev:.5} over peak {peak:.3} = {:.3} %",
                100.0 * dev / peak);

        // ---- the same answer addressed by byte offset, as the cache arena does. The offsets
        // do not follow the id order: a kernel that ignored them and used blockIdx.y as an index
        // would still pass a test where slot j held expert j.
        let offs: Vec<u32> = ids.iter().map(|e| (*e as usize * ebytes) as u32).collect();
        let offd = g.upload_bytes(
            &offs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let act2 = g.alloc(ksel * n);
        let enc = g.begin();
        g.dispatch(&enc, "moe_gu_iq3s_slots",
                   &[(&xd, 0), (&gd, 0), (&ud, 0), (&act2, 0), (&offd, 0), (&offd, 0)],
                   &[k as u32, n as u32], grid, [256, 1, 1]).unwrap();
        g.submit(enc).unwrap();
        let mut slots = vec![0.0; ksel * n];
        g.read(&act2, &mut slots);
        assert_eq!(slots, got, "moe_gu_iq3s_slots disagrees with the id-addressed form");
    }

    /// `_accum` adds into `y` and `_bias` adds a vector; both are what the decode graph
    /// dispatches. A copy-paste slip between the three bodies would leave the plain form correct
    /// and these wrong.
    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn gemv_nat_iq_accum_and_bias_match_plain() {
        let mut g = gpu();
        g.ensure_family("gemv_iq").unwrap();
        let (k, n) = (512usize, 32usize);
        let x = tvec(k, 77);
        let xd = g.upload(&x);
        for (ty, tag) in [(21u32, "iq3s"), (16, "iq2xxs")] {
            let (_, wpb) = ojas_formats::synth::block_shape(ty).unwrap();
            let bytes = ojas_formats::synth::blocks(ty, n * (k / wpb));
            let wd = g.upload_bytes(&bytes).unwrap();
            let grid = [(n as u32).div_ceil(4), 1, 1];

            let plain = g.alloc(n);
            let enc = g.begin();
            g.dispatch(&enc, &format!("gemv_nat_{tag}"), &[(&xd, 0), (&wd, 0), (&plain, 0)],
                       &[k as u32, n as u32], grid, [128, 1, 1]).unwrap();
            g.submit(enc).unwrap();
            let mut base = vec![0.0; n];
            g.read(&plain, &mut base);

            let seed = tvec(n, 78);
            let acc = g.upload(&seed);
            let bias = tvec(n, 79);
            let bd = g.upload(&bias);
            let biased = g.alloc(n);
            let enc = g.begin();
            g.dispatch(&enc, &format!("gemv_nat_{tag}_accum"), &[(&xd, 0), (&wd, 0), (&acc, 0)],
                       &[k as u32, n as u32], grid, [128, 1, 1]).unwrap();
            g.dispatch(&enc, &format!("gemv_nat_{tag}_bias"),
                       &[(&xd, 0), (&wd, 0), (&biased, 0), (&bd, 0)],
                       &[k as u32, n as u32], grid, [128, 1, 1]).unwrap();
            g.submit(enc).unwrap();
            let (mut ga, mut gb) = (vec![0.0; n], vec![0.0; n]);
            g.read(&acc, &mut ga);
            g.read(&biased, &mut gb);

            let peak = base.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            for i in 0..n {
                assert!((ga[i] - (seed[i] + base[i])).abs() / peak < 1e-5,
                        "{tag}_accum[{i}]: {} != {} + {}", ga[i], seed[i], base[i]);
                assert!((gb[i] - (base[i] + bias[i])).abs() / peak < 1e-5,
                        "{tag}_bias[{i}]: {} != {} + {}", gb[i], base[i], bias[i]);
            }
        }
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn rmsnorm_matches_oracle() {
        let mut g = gpu();
        g.ensure_family("ops").unwrap();
        let d = 1024usize;
        let eps = 1e-6f32;
        let x = tvec(d, 1);
        let w = tvec(d, 2);
        // CPU oracle
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        let want: Vec<f32> = x.iter().zip(&w).map(|(xv, wv)| xv * inv * wv).collect();
        let (xd, wd, outd) = (g.upload(&x), g.upload(&w), g.alloc(d));
        let enc = g.begin();
        g.dispatch(
            &enc,
            "rmsnorm",
            &[(&xd, 0), (&wd, 0), (&outd, 0)],
            &[d as u32, eps.to_bits()],
            [1, 1, 1],
            [256, 1, 1],
        )
        .unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; d];
        g.read(&outd, &mut got);
        assert!(max_rel_err(&got, &want) < 1e-4);
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn silu_mul_matches_oracle() {
        let mut g = gpu();
        g.ensure_family("ops").unwrap();
        let n = 4096usize;
        let gate = tvec(n, 3);
        let up = tvec(n, 4);
        let want: Vec<f32> =
            gate.iter().zip(&up).map(|(gv, uv)| gv / (1.0 + (-gv).exp()) * uv).collect();
        let (gd, ud, od) = (g.upload(&gate), g.upload(&up), g.alloc(n));
        let enc = g.begin();
        g.dispatch(
            &enc,
            "silu_mul",
            &[(&gd, 0), (&ud, 0), (&od, 0)],
            &[n as u32],
            [(n as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
        )
        .unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; n];
        g.read(&od, &mut got);
        assert!(max_rel_err(&got, &want) < 1e-5);
    }

    /// gemv_q4 uses the q4_ggml (block_q4_0 split-half) packing, the CUDA-side format, packed by
    /// the shared ojas-formats implementation.
    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn gemv_q4_matches_oracle_q4_ggml() {
        let mut g = gpu();
        g.ensure_family("gemv").unwrap();
        let (k, n) = (2048usize, 512usize);
        let w = tvec(n * k, 5);
        let x = tvec(k, 6);
        let (w4, scales) = ojas_formats::quant::q4_ggml_from_f32(&w, n, k);
        // CPU oracle on the dequantized weights (same rounding as the kernel sees)
        let nblk = k / 32;
        let mut want = vec![0.0f32; n];
        for row in 0..n {
            let mut acc = 0.0f32;
            for b in 0..nblk {
                let sc = scales[row * nblk + b].to_f32();
                for j in 0..16 {
                    let by = w4[row * k / 2 + b * 16 + j];
                    let lo = (by & 0xF) as f32 - 8.0;
                    let hi = (by >> 4) as f32 - 8.0;
                    acc += sc * (lo * x[b * 32 + j] + hi * x[b * 32 + 16 + j]);
                }
            }
            want[row] = acc;
        }
        let xd = g.upload(&x);
        // raw byte buffers uploaded via f32 reinterpretation (untyped on device)
        let w4d = {
            let as_f32: Vec<f32> = w4
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            g.upload(&as_f32)
        };
        let scd = {
            let bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_bits().to_le_bytes()).collect();
            let as_f32: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            g.upload(&as_f32)
        };
        let yd = g.alloc(n);
        let enc = g.begin();
        g.dispatch(
            &enc,
            "gemv_q4",
            &[(&xd, 0), (&w4d, 0), (&yd, 0), (&scd, 0)],
            &[k as u32, n as u32],
            [(n as u32).div_ceil(4), 1, 1],
            [128, 1, 1],
        )
        .unwrap();
        g.submit(enc).unwrap();
        let mut got = vec![0.0; n];
        g.read(&yd, &mut got);
        assert!(max_rel_err(&got, &want) < 1e-3);
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn ssm_ab_matches_oracle() {
        let mut g = gpu();
        g.ensure_family("ssm").unwrap();
        let hv = 8usize;
        let m = 4usize;
        let n = m * hv;
        let alpha0 = tvec(n, 7);
        let beta0 = tvec(n, 8);
        let dt = tvec(hv, 9);
        let a = tvec(hv, 10);
        let mut wa = alpha0.clone();
        let mut wb = beta0.clone();
        for i in 0..n {
            let x = alpha0[i] + dt[i % hv];
            let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            wa[i] = sp * a[i % hv];
            wb[i] = 1.0 / (1.0 + (-beta0[i]).exp());
        }
        let (ad, bd, dtd, aad) = (g.upload(&alpha0), g.upload(&beta0), g.upload(&dt), g.upload(&a));
        let enc = g.begin();
        g.dispatch(
            &enc,
            "ssm_ab",
            &[(&ad, 0), (&bd, 0), (&dtd, 0), (&aad, 0)],
            &[n as u32, hv as u32],
            [(n as u32).div_ceil(64), 1, 1],
            [64, 1, 1],
        )
        .unwrap();
        g.submit(enc).unwrap();
        let (mut ga, mut gb) = (vec![0.0; n], vec![0.0; n]);
        g.read(&ad, &mut ga);
        g.read(&bd, &mut gb);
        assert!(max_rel_err(&ga, &wa) < 1e-4 && max_rel_err(&gb, &wb) < 1e-5);
    }

    // ---------- cnn family vs PyTorch, bit for bit ----------

    use ojas_cuda::CuBuf;
    use serde_json::Value;
    use std::path::{Path, PathBuf};

    struct Golden {
        dir: PathBuf,
        f16: bool,
    }

    impl Golden {
        fn tensor(&self, case: &Value, key: &str) -> (Vec<f32>, Vec<usize>) {
            let t = &case["tensors"][key];
            let bytes = std::fs::read(self.dir.join(t["file"].as_str().unwrap())).unwrap();
            let data = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let shape = t["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
            (data, shape)
        }
        fn has(&self, case: &Value, key: &str) -> bool {
            !case["tensors"][key].is_null()
        }
        fn up(&self, g: &CudaGpu, data: &[f32]) -> CuBuf {
            if self.f16 { g.upload_f16(data) } else { g.upload(data) }
        }
        fn zeros(&self, g: &CudaGpu, n: usize) -> CuBuf {
            self.up(g, &vec![0.0; n])
        }
    }

    fn launch(g: &CudaGpu, enc: &<CudaGpu as ojas_core::Device>::Enc, name: &str,
              bufs: &[&CuBuf], consts: &[u32], n: usize) {
        let bufs: Vec<(&CuBuf, u64)> = bufs.iter().map(|b| (*b, 0)).collect();
        g.dispatch(enc, name, &bufs, consts, [(n as u32).div_ceil(256), 1, 1], [256, 1, 1]).unwrap();
    }

    /// Bitwise equality; two NaNs compare equal (payloads are not specified upstream).
    fn bits_eq(got: &[f32], want: &[f32]) -> Result<(), String> {
        if got.len() != want.len() {
            return Err(format!("length {} != {}", got.len(), want.len()));
        }
        let bad: Vec<usize> = (0..got.len())
            .filter(|&i| got[i].to_bits() != want[i].to_bits() && !(got[i].is_nan() && want[i].is_nan()))
            .collect();
        match bad.first() {
            None => Ok(()),
            Some(&i) => Err(format!("{} of {} differ; first at {i}: got {:e} want {:e}",
                                    bad.len(), got.len(), got[i], want[i])),
        }
    }

    fn run_case(g: &CudaGpu, gd: &Golden, case: &Value) -> Result<(), String> {
        let sfx = if gd.f16 { "_f16" } else { "" };
        let k = |base: &str| format!("{base}{sfx}");
        let p = |key: &str| case[key].as_u64().unwrap() as u32;
        let enc = g.begin();
        let (out, want): (Vec<(CuBuf, usize)>, Vec<Vec<f32>>) = match case["kernel"].as_str().unwrap() {
            "cnn_bias_act" => {
                let (x, _) = gd.tensor(case, "x");
                let (b, _) = gd.tensor(case, "bias");
                let (xd, bd) = (gd.up(g, &x), gd.up(g, &b));
                launch(g, &enc, &k("cnn_bias_act"), &[&xd, &bd],
                       &[x.len() as u32, p("plane"), p("ch"), p("act")], x.len());
                (vec![(xd, x.len())], vec![gd.tensor(case, "y").0])
            }
            "cnn_act" => {
                let (x, _) = gd.tensor(case, "x");
                let (xd, yd) = (gd.up(g, &x), gd.zeros(g, x.len()));
                launch(g, &enc, &k("cnn_act"), &[&xd, &yd], &[x.len() as u32, p("act")], x.len());
                (vec![(yd, x.len())], vec![gd.tensor(case, "y").0])
            }
            "cnn_add" => {
                let (a, _) = gd.tensor(case, "a");
                let (b, _) = gd.tensor(case, "b");
                let (ad, bd, yd) = (gd.up(g, &a), gd.up(g, &b), gd.zeros(g, a.len()));
                launch(g, &enc, &k("cnn_add"), &[&ad, &bd, &yd], &[a.len() as u32], a.len());
                (vec![(yd, a.len())], vec![gd.tensor(case, "y").0])
            }
            kind @ ("cnn_concat" | "cnn_split") => {
                let lens: Vec<u32> = case["axis_lens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
                let total: u32 = lens.iter().sum();
                let inner = p("inner");
                let concat = kind == "cnn_concat";
                let (whole, wshape) = gd.tensor(case, if concat { "y" } else { "x" });
                let outer = wshape[0] as u32;
                let wd = if concat { gd.zeros(g, whole.len()) } else { gd.up(g, &whole) };
                let mut parts = vec![];
                let mut off = 0u32;
                for (i, &len) in lens.iter().enumerate() {
                    let n = (outer * len * inner) as usize;
                    if concat {
                        let src = gd.up(g, &gd.tensor(case, &format!("in{i}")).0);
                        launch(g, &enc, &k("cnn_axis_copy"), &[&src, &wd],
                               &[n as u32, len, inner, len, 0, total, off], n);
                        parts.push(src);
                    } else {
                        let dst = gd.zeros(g, n);
                        launch(g, &enc, &k("cnn_axis_copy"), &[&wd, &dst],
                               &[n as u32, len, inner, total, off, len, 0], n);
                        parts.push(dst);
                    }
                    off += len;
                }
                if concat {
                    (vec![(wd, whole.len())], vec![whole])
                } else {
                    let want = (0..lens.len()).map(|i| gd.tensor(case, &format!("out{i}")).0).collect::<Vec<_>>();
                    let out = parts.into_iter().zip(&want).map(|(b, w)| (b, w.len())).collect();
                    (out, want)
                }
            }
            "cnn_upsample_nearest" => {
                let (x, xs) = gd.tensor(case, "x");
                let (y, ys) = gd.tensor(case, "y");
                let scale = (1.0f64 / p("scale") as f64) as f32;
                let (xd, yd) = (gd.up(g, &x), gd.zeros(g, y.len()));
                launch(g, &enc, &k("cnn_upsample_nearest"), &[&xd, &yd],
                       &[y.len() as u32, xs[2] as u32, xs[3] as u32, ys[2] as u32, ys[3] as u32,
                         scale.to_bits(), scale.to_bits()], y.len());
                (vec![(yd, y.len())], vec![y])
            }
            "cnn_maxpool" => {
                let (x, xs) = gd.tensor(case, "x");
                let (y, ys) = gd.tensor(case, "y");
                let (xd, yd) = (gd.up(g, &x), gd.zeros(g, y.len()));
                let (kk, s, pad, dil) = (p("k"), p("s"), p("pad"), p("dil"));
                launch(g, &enc, &k("cnn_maxpool"), &[&xd, &yd],
                       &[y.len() as u32, xs[1] as u32, xs[2] as u32, xs[3] as u32, ys[2] as u32,
                         ys[3] as u32, kk, kk, s, s, pad, pad, dil, dil], y.len());
                (vec![(yd, y.len())], vec![y])
            }
            "cnn_softmax" => {
                let (x, _) = gd.tensor(case, "x");
                let (dim, inner) = (p("dim"), p("inner"));
                let slices = x.len() / dim as usize;
                let (xd, yd) = (gd.up(g, &x), gd.zeros(g, x.len()));
                launch(g, &enc, &k("cnn_softmax"), &[&xd, &yd], &[slices as u32, dim, inner], slices);
                (vec![(yd, x.len())], vec![gd.tensor(case, "y").0])
            }
            "cnn_dwconv" => {
                let (x, xs) = gd.tensor(case, "x");
                let (w, ws) = gd.tensor(case, "w");
                let (y, ys) = gd.tensor(case, "y");
                let has_bias = gd.has(case, "b");
                let (xd, wd, yd) = (gd.up(g, &x), gd.up(g, &w), gd.zeros(g, y.len()));
                let bd = if has_bias { gd.up(g, &gd.tensor(case, "b").0) } else { gd.zeros(g, 1) };
                let (kk, s, pad, dil) = (p("k"), p("s"), p("pad"), p("dil"));
                let out_ch = ws[0] as u32;
                launch(g, &enc, &k("cnn_dwconv"), &[&xd, &wd, &bd, &yd],
                       &[y.len() as u32, out_ch, out_ch / xs[1] as u32, xs[3] as u32, xs[2] as u32,
                         ys[3] as u32, ys[2] as u32, kk, kk, s, s, pad, pad, dil, dil, has_bias as u32],
                       y.len());
                (vec![(yd, y.len())], vec![y])
            }
            other => return Err(format!("unknown kernel {other}")),
        };
        g.submit(enc).map_err(|e| e.to_string())?;
        for ((buf, n), want) in out.iter().zip(&want) {
            let mut got = vec![0.0; *n];
            g.read(buf, &mut got);
            bits_eq(&got, want)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires NVIDIA GPU and scripts/release/torch_kernel_golden.py output"]
    fn cnn_matches_torch_bitwise() {
        let dir = std::env::var("OJAS_TORCH_GOLDEN").map(PathBuf::from).unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/torch_golden")
        });
        let manifest = std::fs::read_to_string(dir.join("manifest.json"))
            .unwrap_or_else(|e| panic!("{}: {e}; run scripts/release/torch_kernel_golden.py", dir.display()));
        let manifest: Value = serde_json::from_str(&manifest).unwrap();
        let mut g = gpu();
        g.ensure_family("cnn").unwrap();
        let cases = manifest["cases"].as_array().unwrap();
        let mut failures = vec![];
        for case in cases {
            let gd = Golden { dir: dir.clone(), f16: case["dtype"] == "f16" };
            if let Err(e) = run_case(&g, &gd, case) {
                failures.push(format!("{}: {e}", case["name"].as_str().unwrap()));
            }
        }
        assert!(failures.is_empty(), "{} of {} cases differ from torch {}:\n{}",
                failures.len(), cases.len(), manifest["torch"], failures.join("\n"));
    }
}
