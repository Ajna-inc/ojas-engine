//! Gate: EMBED_INJECT — a prefill from precomputed residual rows must be indistinguishable
//! from the same prefill done from token ids.
//!
//! `Model::prefill_embeds` is the seam a vision encoder enters the decoder through: it hands the
//! decoder ~4096 rows of `[hidden_dim]` f32 and the decoder runs its stack over them without
//! ever touching `token_embd`. Nothing downstream can tell you the seam is wrong — a mis-strided
//! memcpy, a chunk offset that forgot `ci * chunk_sz * d`, or an ignored `do_embed` flag all
//! produce plausible logits and plausible text, and you find out when a page transcribes as
//! fluent nonsense.
//!
//! So the gate removes the vision model from the question: the rows it injects are the rows the
//! decoder would have gathered anyway, `token_embd.weight` read host-side straight out of the
//! GGUF and widened to f32. Two prefills of the same prompt — one by id, one by row — must land
//! on the same logits, and on an F16 table with no requantization in the way, "same" means
//! bit-identical rather than close. Any drift is a bug in the injection, because nothing else
//! differs.
//!
//! That identity would also pass if the injected rows were ignored and the gather ran anyway, so
//! there is a negative control: re-inject with one row (in the second chunk, where the
//! chunk-offset arithmetic lives) swapped for a different token's embedding, and require the
//! logits to move.
//!
//! Last, `reuse_prefix_len` must return 0 after an injected prefill. Injected spans carry
//! placeholder ids — every image is the same `<|image_pad|>` run — so a longest-common-prefix
//! over raw ids would match one page's KV cache against another page's ids and transcribe the
//! wrong document. The check has teeth because the gate first shows the same prompt is reusable
//! after a normal id prefill.
//!
//! Requirements: the rows are reconstructed host-side, so they only match what the decoder would
//! gather if the decoder kept `token_embd` in the file's own F16, i.e. a prec that does not
//! requantize the embedding table (prec 0 on an F16 file; `repr_gate` prints where the table
//! landed). At a requantizing prec the host rows are the pre-quantization values and the
//! comparison would measure the requantizer, not the injection, so the gate fails instead.
//!
//! usage: embed_inject_gate <gguf> [prec] [n_prompt_tokens]

use anyhow::Result;
use ojas_core::Model;

/// The reuse floor in `prefill_session.rs` (REUSE_MIN_LCP). A prompt shorter than this can
/// never be reused, so the positive control would be vacuous.
const REUSE_MIN_LCP: usize = 256;

fn main() -> Result<()> {
    // Positional path first, then OJAS_TEST_GGUF (colon-separated, as the
    // model-dependent tests in ojas-formats / ojas-tokenize use it).
    let gguf = std::env::args().nth(1)
        .or_else(|| std::env::var("OJAS_TEST_GGUF").ok()
            .and_then(|l| l.split(':').map(str::to_string).find(|p| std::path::Path::new(p).exists())))
        .expect("usage: embed_inject_gate <gguf> [prec] [n_prompt_tokens]  (or set OJAS_TEST_GGUF)");
    let prec: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let want_n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(320);

    ojas_core::logging::init();

    // A real Qwen2.5 chat prompt, tiled up to `want_n` ids. The tiling exists for the reuse
    // control: the reuse floor is 256 tokens, so a 10-token prompt could not be reused even by a
    // model that wanted to, and "reuse returned 0" would prove nothing. OJAS_PROMPT overrides it
    // verbatim, with no tiling — a prompt is only meaningful against its own vocabulary.
    let base: Vec<u32> = match std::env::var("OJAS_PROMPT") {
        Ok(v) => v.split(',').map(|t| t.trim().parse()).collect::<std::result::Result<Vec<_>, _>>()?,
        Err(_) => {
            let chat = [151644u32, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198];
            (0..want_n.max(2)).map(|i| chat[i % chat.len()]).collect()
        }
    };
    anyhow::ensure!(base.len() >= 2, "need at least 2 prompt tokens");
    // Prefill everything but the last id; the last id drives the forward whose logits we
    // compare, so the comparison reads the state the prefill left.
    let (span, probe) = (&base[..base.len() - 1], base[base.len() - 1]);
    let n = span.len();

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;

    // ---- host-side rows -----------------------------------------------------
    // read_tensor hands back F16 bytes for an F16 table (and F32 for an F32 one); widening
    // f16 -> f32 is exact, which is what lets the comparison below be an equality rather than a
    // tolerance.
    let (dims, ty, bytes) = g.read_tensor("token_embd.weight")?;
    anyhow::ensure!(dims.len() == 2, "token_embd.weight is not 2-D: {dims:?}");
    let d = dims[0] as usize;                       // ne[0] = embedding length
    let vocab = dims[1] as usize;
    println!("  token_embd.weight: [{d} x {vocab}] ggml_type {ty} ({} MB)", bytes.len() / (1 << 20));

    // Read the M-RoPE section split before `load`, since `int_arr` borrows the Gguf; whether
    // sections are declared decides whether the position cases below can run at all. Sizes are
    // in cos/sin pairs and sum to n_rot/2; surya-2 declares [11,11,10,0].
    let arch = g.arch();
    let sections: [u32; 4] = g.int_arr(&format!("{arch}.rope.dimension_sections"))
        .map(|a| { let mut v = [0u32; 4]; for (i, x) in a.iter().take(4).enumerate() { v[i] = (*x).max(0) as u32; } v })
        .unwrap_or([0; 4]);
    let has_sections = sections.iter().any(|&s| s != 0);

    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, (base.len() + 64).max(2048), prec, None, None)?;
    anyhow::ensure!(m.d == d, "hidden_dim mismatch: model {} vs table {d}", m.d);
    anyhow::ensure!(span.iter().all(|&t| (t as usize) < vocab), "prompt id outside the vocab");

    // The rows must be the ones the decoder would have gathered. If the loader requantized the
    // table, they are not, and a difference downstream would be the requantizer's rather than
    // the injection's.
    let want_repr = if ty == 1 { "F16" } else { "F32" };
    let embed_repr_ok = m.repr_names(want_repr).iter().any(|s| s == "token_embd.weight");
    if !embed_repr_ok {
        for r in ["Native", "Q4L", "Q4K", "Q6K", "Q20", "Q8", "Q4", "F16", "F32"] {
            if m.repr_names(r).iter().any(|s| s == "token_embd.weight") {
                println!("  token_embd.weight loaded as {r}, not {want_repr}");
            }
        }
        println!("GATE: EMBED_INJECT FAIL (prec {prec} requantized the embedding table, so \
                  host-read rows are not the rows the gather would produce — \
                  re-run at a prec that keeps token_embd in the file's own format, e.g. prec 0 on an F16 file)");
        std::process::exit(1);
    }

    let row = |t: u32| -> Vec<f32> {
        let o = t as usize * d;
        match ty {
            1 => bytes[o * 2..(o + d) * 2].chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            _ => bytes[o * 4..(o + d) * 4].chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        }
    };
    let mut rows = Vec::with_capacity(n * d);
    for &t in span { rows.extend_from_slice(&row(t)); }
    println!("  injecting {n} rows x {d} = {} f32 ({:.1} MB)", rows.len(), rows.len() as f64 * 4.0 / 1e6);

    // Drive everything through the trait, not the inherent methods: that is the surface a
    // vision host holds, and where a missed forward in `impl Model for Box<T>` / `&T` would
    // show up as "unsupported".
    let md: &dyn Model = &m;

    // ---- path A: prefill by token id ---------------------------------------
    md.reset_session();
    md.prefill(span, 0);
    // Logits first, reuse probe second, and it has to be this order. On a recurrent arch
    // `reuse_prefix_len` is not a pure query: once the shared prefix reaches its 256-token floor
    // it restores the nearest snapshot at or before the divergence, and on a freshly prefilled
    // sequence the only snapshot is the zeroed anchor `prefill` captured at position 0. So
    // asking "how much is reusable" wipes the SSM state and only then answers "none". Probing
    // first makes a 199-row span pass and a 256-row span fail (the LCP floor is the threshold,
    // not anything about chunking), on the recurrent model only, which reads exactly like a
    // multi-chunk injection bug.
    let a = md.forward_logits(probe, n).expect("forward_logits");
    let reuse_a = md.reuse_prefix_len(&base);

    // ---- path B: prefill by injected row ------------------------------------
    md.reset_session();
    let ok = md.prefill_embeds(span, &rows, 0, None);
    if !ok {
        println!("GATE: EMBED_INJECT FAIL (prefill_embeds returned false — this architecture has no injection path)");
        std::process::exit(1);
    }
    let reuse_b = md.reuse_prefix_len(&base);
    let b = md.forward_logits(probe, n).expect("forward_logits");

    // ---- path C: negative control -------------------------------------------
    // One row replaced by a different token's embedding, placed past the 256-token chunk
    // boundary so it also exercises the per-chunk source offset. If the injected rows were being
    // ignored (gather still running), C == B here and the identity above would be meaningless.
    let hurt_at = n - 1;
    let other = span.iter().copied().find(|&t| t != span[hurt_at]).unwrap_or((span[hurt_at] + 1) % vocab as u32);
    let mut rows_c = rows.clone();
    rows_c[hurt_at * d..(hurt_at + 1) * d].copy_from_slice(&row(other));
    md.reset_session();
    assert!(md.prefill_embeds(span, &rows_c, 0, None));
    let c = md.forward_logits(probe, n).expect("forward_logits");

    // ---- paths N/D/E: sectioned M-RoPE --------------------------------------
    // Three runs over the same injected rows, differing only in `pos3`:
    //
    //   N  pos3 = None         the baseline B above, re-run so the fingerprints come from a
    //                          cache the negative control C has not touched.
    //   D  t == h == w, e = 0  what llama.cpp writes for a text token under M-RoPE
    //                          (`llama-graph.cpp`, `llm_graph_input_pos::set_input`: "the 3
    //                          first dims are the same, and 4th dim is all 0"). Every section
    //                          then reads the same number, so the sectioned kernel has to
    //                          reproduce plain rope bit-for-bit — the regression guard for
    //                          every model already shipped. Not vacuous: the decoder enables
    //                          sectioned rope on `pos3.is_some()` and a declared section split,
    //                          nothing about the values, so D runs the sectioned kernel, and E,
    //                          which differs from D only in the numbers, proves the mode was on.
    //   E  a real patch grid   (pos_0, pos_0 + row, pos_0 + col, 0) per
    //                          `mtmd_image_tokens_get_decoder_pos` (MTMD_POS_TYPE_MROPE). Must
    //                          move the logits, or the coordinates are being ignored and §3.2
    //                          is not wired at all.
    //
    // D and E are compared at the KV cache as well as at the logits, because that is where the
    // two things sectioned rope must keep separate become visible: the rope angle may move, the
    // cache row may not. K is rotated, V is not, and the first roping layer's input is
    // rope-independent (on a hybrid the layers before it are recurrent), so at that layer V must
    // be bit-identical however the positions are numbered, and any V drift means rows landed in
    // the wrong slots.
    let (mut pos_ndiff, mut pos_bad, mut pos_note) = (usize::MAX, false, String::new());
    let mut pos_report: Vec<String> = Vec::new();
    if !has_sections {
        pos_note = format!("{arch} declares no rope.dimension_sections, so there is nothing to                             section — the M-RoPE cases cannot be run on this model");
    } else {
        let nl = md.n_layers();
        let spare = (m.max_seq() - n).min(64);
        anyhow::ensure!(spare >= 2, "need a few spare context rows past the span for the                                      out-of-range cache check (ctx {} , span {n})", m.max_seq());
        let fnv = |v: &[f32]| -> u64 {
            v.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3))
        };
        // Fingerprint of everything past the span, every layer, K and V. The rope kernel
        // addresses the cache as base_pos+m; if a position leaked into that address instead of
        // only into the angle, an image coordinate larger than the span would land here.
        let tail_fp = || -> u64 {
            (0..nl).fold(0u64, |h, l| {
                let (k, v) = m.dump_kv(l, n, spare);
                h.wrapping_mul(31).wrapping_add(fnv(&k)).wrapping_mul(31).wrapping_add(fnv(&v))
            })
        };

        // --- N: baseline, and find the first layer that ropes at all -----------
        md.reset_session();
        anyhow::ensure!(md.prefill_embeds(span, &rows, 0, None), "baseline re-run refused");
        // On a hybrid only 1 in `full_attention_interval` layers has a KV cache at all; the
        // recurrent ones never write theirs, so it stays zero. That makes "first non-zero K
        // cache" the first roping layer, with no arch table.
        let l0 = (0..nl).find(|&l| m.dump_kv(l, 0, n).0.iter().any(|&x| x != 0.0));
        let Some(l0) = l0 else {
            anyhow::bail!("no layer wrote a K cache — this model does not rope, so the                            M-RoPE cases cannot be checked here");
        };
        let (kn, vn) = m.dump_kv(l0, 0, n);
        let nlog = md.forward_logits(probe, n).expect("forward_logits");
        anyhow::ensure!(nlog.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "the pos3=None baseline did not reproduce path B — state is leaking between runs");

        // --- D: degeneracy guard ------------------------------------------------
        let deg: Vec<[u32; 4]> = (0..n).map(|i| [i as u32, i as u32, i as u32, 0]).collect();
        md.reset_session();
        anyhow::ensure!(md.prefill_embeds(span, &rows, 0, Some(&deg)),
            "prefill_embeds refused degenerate pos3 on a model that declares sections {sections:?}");
        let (kd, vd) = m.dump_kv(l0, 0, n);
        let dlog = md.forward_logits(probe, n).expect("forward_logits");
        let dbits = dlog.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let kd_diff = kd.iter().zip(&kn).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let vd_diff = vd.iter().zip(&vn).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        // After D's forward, so the row the probe writes at position n is already present in
        // the baseline E is compared against.
        let tail_ref = tail_fp();

        // --- E: a genuine patch grid ---------------------------------------------
        // nx columns x ny rows over the span. The largest divisor of n at or below sqrt(n)
        // gives an exact grid where one exists (319 = 11 x 29); a prime span falls back to nx=2
        // and a partial last row, which the per-token formula (row = i/nx, col = i%nx) handles
        // either way.
        let nx = (1..=n).rev().find(|c| c * c <= n && n % c == 0).unwrap_or(1).clamp(2, n);
        let ny = n.div_ceil(nx);
        let img: Vec<[u32; 4]> = (0..n).map(|i| [0, (i / nx) as u32, (i % nx) as u32, 0]).collect();
        md.reset_session();
        anyhow::ensure!(md.prefill_embeds(span, &rows, 0, Some(&img)),
            "prefill_embeds refused 2-D pos3 on a model that declares sections {sections:?}");
        let (k2, v2) = m.dump_kv(l0, 0, n);
        let tail_2d = tail_fp();
        let elog = md.forward_logits(probe, n).expect("forward_logits");
        let ebits = elog.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        pos_ndiff = ebits;
        let kvd = kn.len() / n;                       // kvdim, from the dump itself
        let k2_diff = k2.iter().zip(&kn).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let v2_diff = v2.iter().zip(&vn).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        // Per row: row 0's grid coordinate is (0,0,0), which is its own scalar position, so row
        // 0 alone must be untouched and every other row must move. A descriptor read at a fixed
        // row, or off by one, fails exactly here.
        let rows_moved: Vec<usize> = (0..n)
            .filter(|&r| k2[r * kvd..(r + 1) * kvd].iter().zip(&kn[r * kvd..(r + 1) * kvd])
                .any(|(x, y)| x.to_bits() != y.to_bits()))
            .collect();

        pos_report.push(format!("  sections            {sections:?}  mode IMROPE (interleaved t h w)  first roping layer {l0} (kvdim {kvd})"));
        pos_report.push(format!("  D degenerate t=h=w  argmax {}  logit bitdiff {dbits}/{}  K bitdiff {kd_diff}  V bitdiff {vd_diff}", argmax(&dlog), b.len()));
        pos_report.push(format!("  E grid {nx}x{ny}         argmax {}  logit bitdiff {ebits}/{}  K bitdiff {k2_diff}  V bitdiff {v2_diff}", argmax(&elog), b.len()));
        pos_report.push(format!("  E rows with moved K  {} of {n} (expect n-1: only row 0's (t,h,w)=(0,0,0) equals its scalar position)", rows_moved.len()));
        pos_report.push(format!("  cache past the span  fingerprint {} (rows [{n}, {}))",
            if tail_2d == tail_ref { "unchanged" } else { "CHANGED" }, n + spare));

        if dbits != 0 || kd_diff != 0 || vd_diff != 0 {
            pos_report.push(format!("  FAIL: t == h == w is NOT bit-identical to pos3=None                 ({dbits} logits, {kd_diff} K, {vd_diff} V) — sectioned rope has changed the                 answer for every text model in the tree, not just for images"));
            pos_bad = true;
        }
        if ebits == 0 || k2_diff == 0 {
            pos_report.push("  FAIL: 2-D positions changed nothing — pos3 is being IGNORED                 (plumbed but not reaching the rope dispatch)".into());
            pos_bad = true;
        }
        if v2_diff != 0 {
            pos_report.push(format!("  FAIL: {v2_diff} V elements moved at layer {l0}. V is not                 rotated and this layer's input does not depend on rope, so its rows can only                 change by being WRITTEN SOMEWHERE ELSE — a position has leaked into the cache                 row address, scattering the span"));
            pos_bad = true;
        }
        if tail_2d != tail_ref {
            pos_report.push(format!("  FAIL: the cache changed outside [0, {n}) — an image                 coordinate became a cache row"));
            pos_bad = true;
        }
        if rows_moved.len() != n - 1 || rows_moved.first() == Some(&0) {
            pos_report.push(format!("  FAIL: {} rows moved (first {:?}); expected exactly rows                 1..{n} — the per-row descriptor index is wrong", rows_moved.len(), rows_moved.first()));
            pos_bad = true;
        }
    }

    // ---- verdict -------------------------------------------------------------
    anyhow::ensure!(a.len() == b.len() && a.len() == c.len(), "logit lengths differ");
    let (mut ndiff, mut maxabs, mut at) = (0usize, 0f32, usize::MAX);
    for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
        if x.to_bits() != y.to_bits() {
            ndiff += 1;
            if at == usize::MAX { at = i; }
            maxabs = maxabs.max((x - y).abs());
        }
    }
    let cdiff = a.iter().zip(&c).filter(|(&x, &y)| x.to_bits() != y.to_bits()).count();
    let cmax = a.iter().zip(&c).map(|(&x, &y)| (x - y).abs()).fold(0f32, f32::max);

    println!();
    println!("  prompt              {} ids ({n} prefilled, probe id {probe} at pos {n})", base.len());
    println!("  A id-prefill        argmax {}", argmax(&a));
    println!("  B row-injection     argmax {}  bitdiff {ndiff}/{}  maxabs {maxabs:e}  first {}",
        argmax(&b), a.len(), if at == usize::MAX { "-".into() } else { at.to_string() });
    println!("  C perturbed row {hurt_at}  argmax {}  bitdiff {cdiff}/{}  maxabs {cmax:e}", argmax(&c), a.len());
    println!("  reuse_prefix_len    after A = {reuse_a}   after B = {reuse_b}   (floor {REUSE_MIN_LCP})");
    if pos_report.is_empty() { println!("  NOTE: {pos_note}"); }
    for l in &pos_report { println!("{l}"); }

    let mut bad = pos_bad;
    if ndiff != 0 {
        println!("  FAIL: injected prefill is not bit-identical to the id prefill \
                  ({ndiff} logits differ, max |Δ| {maxabs:e})");
        bad = true;
    }
    if cdiff == 0 {
        println!("  FAIL: perturbing an injected row changed nothing — the rows are being \
                  IGNORED and the embedding gather is still running");
        bad = true;
    }
    if reuse_b != 0 {
        println!("  FAIL: reuse_prefix_len returned {reuse_b} after an injected prefill — a second \
                  image with the same placeholder ids would be served this one's KV cache");
        bad = true;
    }
    if n >= REUSE_MIN_LCP && reuse_a == 0 {
        println!("  NOTE: reuse returned 0 after the id prefill too, so the reuse check above is \
                  vacuous on this model/config (reuse may be disabled or unsupported for this arch)");
    }

    if bad { println!("\nGATE: EMBED_INJECT FAIL"); std::process::exit(1); }
    let mrope = if pos_ndiff == usize::MAX { "M-RoPE not applicable".to_string() }
                else { format!("M-RoPE degenerate == plain, 2-D moved {pos_ndiff} logits") };
    println!("\nGATE: EMBED_INJECT PASS (bit-identical over {} logits, control moved {cdiff}, span non-reusable, {mrope})", a.len());
    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0usize, f32::MIN), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
}
