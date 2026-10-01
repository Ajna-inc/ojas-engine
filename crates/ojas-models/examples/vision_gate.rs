//! Gate: VISION — the Metal ViT tower must reproduce the CPU oracle.
//!
//! `DecoderGpu::encode_image` is 215 Metal dispatches (17 per block x 12, plus a 5-dispatch
//! patch-embed prologue and a 6-dispatch projector) whose output nothing downstream can
//! sanity-check: a transposed patch row, a 2×2 merge that drifted out of step with the M-RoPE
//! position table, sections 2 and 3 of the rope wired as if they were reachable, or `eps` moved
//! outside a LayerNorm's sqrt all produce finite, plausible embeddings, and the page then
//! transcribes as fluent nonsense or as almost the right text.
//!
//! The reference is `ojas_cpu::cpu_vit::CpuVit`, a literal transcription of
//! `tools/mtmd/models/qwen3vl.cpp` with a unit test for each of the five things that are not
//! what you would write from the config. Both sides are handed the same preprocessed pixels,
//! produced once by `ojas_cpu::VitPreproc`; the resampler is out of scope and has its own gate.
//!
//! The two sides are not bit-identical by construction: `gemm_mm_f16` (`gemv.rs:1820`) rounds
//! activations to f16 into the MMA tile and the attention kernels read K and V as `half` (as
//! does `ggml_flash_attn_ext`, the reference here), while the oracle keeps activations and K/V
//! in f32 with f16 weights — ~5e-4 of relative rounding per matmul that the oracle does not
//! carry. The tower is also chaotically ill-conditioned: last-layer activations reach 1500–3000
//! and `v.post_ln` divides by a std those channels dominate, so a 1.14e-4 relative perturbation
//! costs post_ln 0.9972 at 2304 patches on its own. The per-layer cosines printed below separate
//! accumulated f16 rounding (smooth decay from layer 0) from a wiring bug (a cliff at one layer,
//! or layer 0 already wrong).
//!
//! Conditioning probe: each size is encoded again with the input pixels perturbed by one f16
//! rounding's worth of relative error (2.4e-4, alternating sign — the size of one `gemm_mm_f16`
//! activation round). GPU-vs-GPU-epsilon is the tower's own amplification at this patch count
//! with every index and weight identical, and a lower bound on the expected disagreement, since
//! the real GPU path carries one such perturbation per matmul rather than one at the input. A
//! GPU-vs-oracle gap within a small factor of it is arithmetic, not wiring. Both numbers are
//! always printed.
//!
//! Negative control: agreement alone would also pass if both sides ignored most of the image, so
//! each size is re-encoded with one 16×16 patch perturbed — the last patch, which a causal mask
//! or a truncated attention bound would drop — and the perturbed-vs-oracle cosine must be far
//! worse than the clean one. Attention is bidirectional, so that patch must move every merged
//! token, not just its own; the control reports how many moved.
//!
//! Sizes: three branches of the geometry, all pure index arithmetic:
//!   * a small square grid — cheap, and the size at which the numerics are tight enough that
//!     any disagreement is structural;
//!   * 768×768 — a 48×48 patch grid, which is `image_size / patch_size` exactly and therefore
//!     takes `resize_position_embeddings`' early return (the position embedding must be used
//!     bit-unmodified, not resized with sf == 1);
//!   * a non-square grid — `sf0` and `sf1` are computed independently, and a transposed resize
//!     or a `pw`/`ph` swap in the merge nest survives every square test.
//!
//! Thresholds on the projector output (what the decoder consumes): cosine ≥ 0.999 and RMSE ≤
//! 0.05 on the projector row, per token, the convention `scripts/release/compare_logits.py`
//! enforces (every row must pass, not the pooled buffer). A worst-token cosine below the bar
//! fails unless it is within 2x, in (1 - cos), of the conditioning probe above, where the tower
//! cannot distinguish our arithmetic from its own. OJAS_VIT_MIN_COS / OJAS_VIT_MAX_RMSE
//! override the bar; the gate prints what it measured either way.
//!
//! usage: vision_gate <gguf> [prec] [sizes]
//!        sizes is a comma-separated WxH list of source images (default
//!        96x96,768x768,160x96); prec defaults to 4, the CLI default, which is the
//!        precision that keeps the mmproj in f16.

use anyhow::{Context, Result};
use ojas_cpu::{cpu_vit::CpuVit, VitPreproc};

/// Cosine similarity and RMSE of two equal-length vectors.
fn cos_rmse(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut se) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        se += (x - y) * (x - y);
    }
    let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 1.0 };
    (cos, (se / a.len().max(1) as f64).sqrt())
}

/// The single worst row of two `[rows][row]` buffers: `(min cosine, that row's RMSE, that
/// row's index, max RMSE over all rows)`.
///
/// Per row, not pooled: the projector output is what the decoder reads one token at a time, and
/// a whole-buffer cosine is dominated by the high-norm tokens and would hide one badly wrong
/// embedding among 576 good ones.
fn worst_row(a: &[f32], b: &[f32], row: usize) -> (f64, f64, usize, f64) {
    let rows = a.len() / row;
    let (mut cmin, mut crm, mut at, mut rmax) = (f64::INFINITY, 0.0f64, 0usize, 0.0f64);
    for t in 0..rows {
        let (c, r) = cos_rmse(&a[t * row..(t + 1) * row], &b[t * row..(t + 1) * row]);
        if c < cmin { cmin = c; crm = r; at = t; }
        rmax = rmax.max(r);
    }
    (if cmin.is_finite() { cmin } else { 1.0 }, crm, at, rmax)
}

/// A deterministic, structured RGB test image. Structure matters: a flat or uniformly random
/// field makes a merge-order or position-embedding bug invisible, because every patch looks
/// like every other patch. This has a diagonal low-frequency gradient (so `pos` and `merge`
/// order are observable), a per-channel offset (so a channel-major/row-major patch row
/// transpose shows up) and a high-frequency checker (so a patch that reads its neighbour's
/// pixels does not average out). Pure LCG, no seed crate.
fn synth_rgb(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = ((s >> 56) as i32) - 128;
            for c in 0..3 {
                let grad = (x * 180 / w.max(1) + y * 60 / h.max(1)) as i32;
                let checker = if ((x / 7) + (y / 5)) % 2 == 0 { 34 } else { -34 };
                let v = grad + checker + noise / 3 + (c as i32) * 21;
                out[(y * w + x) * 3 + c] = v.clamp(0, 255) as u8;
            }
        }
    }
    out
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .or_else(|| std::env::var("OJAS_TEST_GGUF").ok()
            .and_then(|l| l.split(':').map(str::to_string).find(|p| std::path::Path::new(p).exists())))
        .expect("usage: vision_gate <gguf> [prec] [sizes]  (or set OJAS_TEST_GGUF)");
    let prec: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let sizes: Vec<(usize, usize)> = std::env::args().nth(3)
        .unwrap_or_else(|| "96x96,768x768,160x96".into())
        .split(',')
        .map(|s| {
            let (w, h) = s.trim().split_once('x').expect("sizes look like 768x768,160x96");
            (w.parse().expect("width"), h.parse().expect("height"))
        })
        .collect();
    let min_cos: f64 = std::env::var("OJAS_VIT_MIN_COS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.999);
    let max_rmse: f64 = std::env::var("OJAS_VIT_MAX_RMSE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);

    ojas_core::logging::init();
    let timings_trustworthy = ojas_models::bench::report_load();

    // The oracle reads the mmproj directly; the decoder discovers the same sidecar
    // through `mmproj::discover`, so both sides are looking at one file.
    let side = ojas_formats::mmproj::discover(std::path::Path::new(&path), std::env::var("OJAS_MMPROJ").ok().as_deref())?
        .context("no mmproj sidecar found next to the model (and OJAS_MMPROJ unset) — \
                  there is no vision tower to gate")?;
    println!("  model   {path}");
    println!("  mmproj  {}", side.display());

    let mut mg = ojas_formats::gguf::Gguf::open(side.to_str().context("non-UTF8 mmproj path")?)?;
    let t_cpu_load = std::time::Instant::now();
    let oracle = CpuVit::load(&mut mg).context("loading the CPU ViT oracle")?;
    let cpu_load_s = t_cpu_load.elapsed().as_secs_f64();

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&path)?;
    let t_load = std::time::Instant::now();
    // A small ctx: this gate never prefills, and the KV cache is pure overhead here.
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 512, prec, None, None)?;
    let load_s = t_load.elapsed().as_secs_f64();
    anyhow::ensure!(m.has_vision(), "the decoder loaded without a vision tower — the mmproj was not attached");
    println!("  loaded  cpu oracle {cpu_load_s:.1}s / gpu decoder {load_s:.1}s (prec {prec})");
    // A requirement, for the same reason `embed_inject_gate` has one: the oracle reads the
    // mmproj's f16 straight out of the GGUF, so a requantized tower would make the two sides
    // run different weights, and the number this gate prints would be the requantizer's
    // error, not the encoder's (see `vision_weights_f16`).
    if !m.vision_weights_f16() {
        println!("\n  The vision weights were not loaded in f16 at prec {prec}, so they are not the \
                  weights the CPU oracle reads. `repr_gate` prints where each tensor landed.");
        println!("\nGATE: VISION FAIL (tower not in f16)");
        std::process::exit(1);
    }

    // One preprocessing config for both sides. The token budget is widened so the size policy
    // does not resize the synthetic images out from under the test: this gate is about the
    // encoder, and the resampler has its own.
    let pre = VitPreproc::qwen3vl().with_token_budget(1, 1 << 16);
    let pd = m.vision_proj_dim().context("no projection dim")?;

    let mut bad = false;
    let mut at_floor = 0usize;
    let mut rows: Vec<String> = Vec::new();
    for &(sw, sh) in &sizes {
        let rgb = synth_rgb(sw, sh, 0x51_7a_2b_09 ^ (sw as u64) << 20 ^ sh as u64);
        let (w, h, planar) = pre.preprocess(&rgb, sw, sh)?;
        let patches = (w / oracle.patch) * (h / oracle.patch);
        let n_mm = m.n_merged_tokens(w, h).context("merged token count")?;
        let early = (w / oracle.patch) == oracle.pos_side && (h / oracle.patch) == oracle.pos_side;
        println!();
        println!("  ---- {sw}x{sh} -> {w}x{h} px, {} patch grid, {patches} patches, {n_mm} merged tokens{}",
            format!("{}x{}", w / oracle.patch, h / oracle.patch),
            if early { "  [position-embd EARLY RETURN branch]" } else { "  [position-embd bilinear resize]" });

        // One oracle pass for everything: `forward_tapped` returns the same `VitTrace`
        // `forward_trace` does and fires the reference's `cb()` points on the way, so the
        // per-layer residual costs a memcpy rather than a second 12-layer forward (which at
        // 2304 patches is minutes of CPU).
        let mut cpu_layers: Vec<Vec<f32>> = Vec::new();
        let t = std::time::Instant::now();
        let cpu = oracle.forward_tapped(&planar, w, h, &mut |name, _l, vals| {
            if name == "layer_out" { cpu_layers.push(vals.to_vec()); }
        })?;
        let cpu_s = t.elapsed().as_secs_f64();

        // Per-layer residual, for localization. Same arithmetic as the one-command-buffer
        // path; only the commit boundaries move.
        let t = std::time::Instant::now();
        let g_tr = m.encode_image_layer_trace(&planar, w, h)?;
        let gpu_s = t.elapsed().as_secs_f64();

        anyhow::ensure!(g_tr.grid == cpu.grid, "grid disagreement: gpu {:?} vs cpu {:?}", g_tr.grid, cpu.grid);
        anyhow::ensure!(g_tr.out.len() == cpu.out.len(),
            "projector output length: gpu {} vs cpu {}", g_tr.out.len(), cpu.out.len());

        if !cpu_layers.is_empty() && cpu_layers.len() == g_tr.layer_out.len() {
            let mut line = String::from("    per-layer residual cosine ");
            for (i, (a, b)) in g_tr.layer_out.iter().zip(&cpu_layers).enumerate() {
                let (c, _) = cos_rmse(a, b);
                line.push_str(&format!("{}:{:.6} ", i, c));
            }
            println!("{}", line.trim_end());
        }

        let (c_pln, r_pln) = cos_rmse(&g_tr.post_ln, &cpu.post_ln);
        let (c_mm0, r_mm0) = cos_rmse(&g_tr.mm0, &cpu.mm0);
        let (c_all, r_all) = cos_rmse(&g_tr.out, &cpu.out);
        let (wc, wr, wt, rmax) = worst_row(&g_tr.out, &cpu.out, pd);
        println!("    post_ln  cos {c_pln:.6}  rmse {r_pln:.4}   (|x| up to {:.0})",
            cpu.post_ln.iter().fold(0f32, |a, &v| a.max(v.abs())));
        println!("    mm.0     cos {c_mm0:.6}  rmse {r_mm0:.4}");
        println!("    out      cos {c_all:.6}  rmse {r_all:.4}   worst token {wt}: cos {wc:.6} \
                  rmse {wr:.4}   max per-token rmse {rmax:.4}");

        // ---- how many tokens miss the bar, and which ----
        // One token of 576 is noise. A dozen at a stride of 24 is a merge-order or
        // position-embedding bug, and the indices say which.
        let below: Vec<usize> = (0..n_mm).filter(|&t| {
            cos_rmse(&g_tr.out[t * pd..(t + 1) * pd], &cpu.out[t * pd..(t + 1) * pd]).0 < min_cos
        }).collect();

        // ---- conditioning probe: the tower's own amplification of one f16 round --
        // Same weights, same indices, same kernels; only the input moves, by the relative size
        // of a single `gemm_mm_f16` activation rounding. Whatever this costs is a floor no
        // implementation of this tower can beat.
        const F16_ULP_REL: f32 = 2.4e-4;
        let eps_in: Vec<f32> = planar.iter().enumerate()
            .map(|(i, &v)| v * (1.0 + if i % 2 == 0 { F16_ULP_REL } else { -F16_ULP_REL })).collect();
        // Also the only call in the gate that runs the production path — one command buffer
        // for the whole tower — so it is what the timing line reports.
        // `encode_image_layer_trace` above splits the same dispatches across 12 command buffers
        // to read the residual out between blocks, which is an upper bound on the real cost.
        let t = std::time::Instant::now();
        let eps_out = m.encode_image(&eps_in, w, h)?;
        let one_cb_s = t.elapsed().as_secs_f64();
        let (ec, _, et, _) = worst_row(&eps_out, &g_tr.out, pd);
        println!("    cond     input perturbed by one f16 ulp ({F16_ULP_REL:.1e} rel) -> worst token \
                  {et}: cos {ec:.6}   [the tower's own floor at this size]");

        // ---- negative control: one perturbed patch must move the answer ----
        // The last patch, and by a realistic amount (a fifth of the normalized range) rather
        // than a spike. A causal mask, a KV bound stuck at the tile count, or a merge that
        // dropped the trailing block all leave this patch out of some token's attention.
        let mut pert = planar.clone();
        let (pw, ph) = (w / oracle.patch, h / oracle.patch);
        let (px0, py0) = ((pw - 1) * oracle.patch, (ph - 1) * oracle.patch);
        for c in 0..oracle.channels {
            for yy in py0..py0 + oracle.patch {
                for xx in px0..px0 + oracle.patch {
                    pert[c * h * w + yy * w + xx] += 0.4;
                }
            }
        }
        let pert_out = m.encode_image(&pert, w, h)?;
        let (c_pert, r_pert) = cos_rmse(&pert_out, &cpu.out);
        let moved = (0..n_mm).filter(|&t| {
            let a = &pert_out[t * pd..(t + 1) * pd];
            let b = &g_tr.out[t * pd..(t + 1) * pd];
            a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-3)
        }).count();
        println!("    control  one patch perturbed -> cos {c_pert:.6} rmse {r_pert:.4}, \
                  {moved}/{n_mm} merged tokens moved");

        println!("    timing   encode {:.0} ms wall (1 command buffer)   {:.0} ms wall / {:.0} ms \
                  device (12 cb, traced)   cpu oracle {:.0} ms{}",
            one_cb_s * 1e3, gpu_s * 1e3, g_tr.gpu_s * 1e3, cpu_s * 1e3,
            if timings_trustworthy { "" } else { "  [PROVISIONAL — machine busy]" });

        if !g_tr.out.iter().all(|v| v.is_finite()) {
            println!("    FAIL: the GPU projector output is not finite");
            bad = true;
        }
        if rmax > max_rmse {
            println!("    FAIL: max per-token rmse {rmax:.4} exceeds {max_rmse}");
            bad = true;
        }
        if c_all < min_cos {
            println!("    FAIL: pooled projector cosine {c_all:.6} misses {min_cos}");
            bad = true;
        }
        if wc < min_cos {
            // Judged against the conditioning floor, not waved through: the gap has to be
            // within 2x of what the same kernels cost themselves on a one-f16-ulp input
            // change.
            let slack = (1.0 - wc) / (1.0 - ec).max(1e-12);
            if slack > 2.0 {
                println!("    FAIL: worst token {wt} at cos {wc:.6} misses {min_cos}, and it is \
                          {slack:.1}x the tower's own conditioning floor ({ec:.6}) — that is a \
                          wiring error, not arithmetic. The per-layer cosines above localize it.");
                bad = true;
            } else {
                at_floor += 1;
                println!("    NOTE: worst token {wt} at cos {wc:.6} misses the {min_cos} bar, but it \
                          is only {slack:.2}x the conditioning floor ({ec:.6}) — at {patches} \
                          patches this tower amplifies one f16 rounding by that much on its own, so \
                          the bar is not reachable here by any implementation that rounds \
                          activations to f16 (gemv.rs:1820 does). Accepted; recorded.");
            }
        }
        if !below.is_empty() {
            println!("    tokens below cos {min_cos}: {}/{n_mm} at {:?}{}", below.len(), 
                &below[..below.len().min(6)], if below.len() > 6 { " ..." } else { "" });
        }
        // The control must be worse than the clean run by a clear margin, and it must move
        // most of the sequence (attention is bidirectional).
        if c_pert >= wc || (1.0 - c_pert) < 10.0 * (1.0 - c_all).max(1e-9) {
            println!("    FAIL: perturbing one patch barely changed the output (clean cos \
                      {c_all:.6}, perturbed {c_pert:.6}) — the gate has no teeth at this size");
            bad = true;
        }
        if moved * 4 < n_mm * 3 {
            println!("    FAIL: only {moved}/{n_mm} tokens saw the perturbed patch — attention \
                      is not bidirectional (a causal mask or a truncated KV bound)");
            bad = true;
        }
        rows.push(format!("{w}x{h} ({patches} patches): out cos {c_all:.6} worst-token {wc:.6} \
                           (floor {ec:.6}, {}/{n_mm} under bar) rmse<={rmax:.4} | post_ln {c_pln:.6} \
                           | mm.0 {c_mm0:.6} | control {c_pert:.6}", below.len()));
    }

    println!();
    for r in &rows { println!("  {r}"); }
    if bad { println!("\nGATE: VISION FAIL"); std::process::exit(1); }
    println!("\nGATE: VISION PASS ({} sizes against ojas_cpu::CpuVit; pooled cos >= {min_cos}, \
              per-token rmse <= {max_rmse}{})", sizes.len(),
        if at_floor == 0 { ", every token's cos >= the bar".to_string() }
        else { format!(", {at_floor} size(s) with a token at the tower's conditioning floor — see the NOTEs") });
    Ok(())
}
