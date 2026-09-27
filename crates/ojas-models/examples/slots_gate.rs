//! Gate: decoding B independent sequences per step must not change any of them.
//!
//! Slot batching exists for throughput — one weight read serving B tokens instead of one
//! (measured 1.84x on the weight-read term at B=4, ~2.0x whole-token). Its risk is cross-slot
//! state leakage: the B sequences share one set of scratch buffers, one command buffer and one
//! set of weights, and only the slot offset on each state binding keeps them apart. A wrong
//! offset — conv state, delta-net matrix state, a KV base, the flash-decoding partials — still
//! decodes fluent tokens, just the wrong ones, with no crash to find it by.
//!
//! The checks:
//!
//! 1. B=1 through the slot graph is token-identical to `forward_id`. Not a tautology: the slot
//!    graph routes through the M-row kernel family, the scalar graph through GEMVs.
//!    Token-identical rather than bit-identical, since different kernels legitimately differ
//!    in the last ulp and decoding uses the argmax.
//! 2. B=2..4 over prompts of different lengths: each slot gets exactly the tokens it gets
//!    alone. The differing lengths put the slots at different positions, so the per-slot rope
//!    angle and KV base are exercised rather than degenerate.
//! 2b. Companion invariance, the numerics-immune form of check 2. B is held fixed at 4, so both
//!    runs use identical kernels and reduction order, and only the sequences occupying the
//!    other slots change; comparing against a solo run instead would conflate slot
//!    independence with two different kernels agreeing to the last ulp. If the victim's stream
//!    moves, state crossed between slots. This is the check that catches a wrong slot offset.
//! 3. A negative control. Equality checks alone also pass when the slots are secretly aliased
//!    onto one sequence, so one slot's recurrent state is destroyed mid-decode: that slot's
//!    output must move, and the other three must not move at all.
//! 4. The speedup, interleaved with the correctness checks.
//!
//! Grading rule for checks 1-3: a divergence from the solo run fails when it lands strictly
//! before the model ends its turn, and is reported as post-turn drift at or after the
//! turn-ending token. Past that token the model has said it is
//! done and the decoder is sampling a region no training signal covers; `decode_gate` refuses
//! to grade there at all, having recorded three Qwen2.5-0.5B quantizations "failing" purely
//! past it while agreeing on every real token. `dispatch.rs` ships its split-K GEMM on the same
//! footing ("different partition boundaries reassociate the fp32 sum"), validated on whether
//! tokens move rather than on bit-identity. Check 2b is exempt and strictly exact, because it
//! holds the arithmetic fixed.
//!
//! Self-check first: the four prompts must decode to different token streams when run alone,
//! or checks 2 and 3 compare constants and would pass over a broken implementation.
//!
//! Measurement: configurations are interleaved across rounds and the per-B minimum is kept.
//! Running one B to completion and then the next records how machine load drifted rather than
//! what B did — a build starting mid-sweep once charged the later configuration ~2x and
//! inverted the result. Contention only ever makes a round slower, so min-over-rounds recovers
//! each configuration's uncontended cost from one window. Correctness does not depend on the
//! clock and is reported as PASS/FAIL regardless of load; timings above load average 4 are
//! labelled provisional.
//!
//! usage: slots_gate <gguf> [n_tokens] [prec] [rounds]

use anyhow::Result;
use ojas_core::Model;
use std::time::Instant;

/// Four prompts that must decode to four different token streams on the model under test.
/// The lengths differ as well as the content: equal lengths would put every slot at the same
/// position and leave the per-slot rope angle and KV base untested, which is one of the two
/// places a slot offset can be wrong.
///
/// Low ids only. The Qwen2.5 chat ids used elsewhere (77091 "assistant") exceed surya-2's
/// 65425 vocabulary and would trip the prefill bounds check.
const PROMPTS: [&[u32]; 4] = [
    &[14, 15, 16, 17, 18, 19],
    &[220, 33, 44, 55, 66, 77, 88, 99],
    &[16, 1054, 2020, 311, 279],
    &[9, 8, 7, 6, 5, 4, 3, 2, 1],
];

/// Context length at which the gate re-runs the independence check.
///
/// Below 512 positions the decode graph takes `attention_short`, which reads the KV cache
/// directly; above it takes flash-decoding, which stages per-head partials through
/// `st.attn_part` — the one buffer in the whole arena that slot batching had to widen. A
/// short-only gate would leave that stride, and the long-context KV base, untested.
const LONG: usize = 700;

/// A deterministic pseudo-random prompt of `n` ids for slot `s`, all in range.
///
/// Distinct per slot so the slots diverge, and generated rather than literal because it is
/// 700 ids x 4 slots. The LCG keeps it reproducible across runs and machines.
fn long_prompt(s: usize, n: usize, vocab: usize) -> Vec<u32> {
    let mut x = 0x2545_F491_4F6C_DD1Du64 ^ ((s as u64 + 1) * 0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        // Keep clear of the special-token band at the bottom of the vocab: an EOS mid-prompt
        // is legal but makes the reference stream depend on where it landed, not on the slot.
        (32 + (x % (vocab as u64 - 64))) as u32
    }).collect()
}

/// First index at which `a` and `b` differ, or `None` if the compared prefix agrees.
fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
        .or(if a.len() == b.len() { None } else { Some(a.len().min(b.len())) })
}

/// Index of the token that ends the model's turn, if it emitted one.
///
/// Everything from here on is out of distribution: the model has said it is done, and a decoder
/// driven past that point samples a region no training signal covers. `decode_gate` stops
/// generating here for the same reason, and records that three quantizations of Qwen2.5-0.5B
/// once "failed" purely past it while agreeing on every real token.
fn turn_end(v: &[u32], eos: Option<u32>) -> usize {
    eos.and_then(|e| v.iter().position(|&t| t == e)).unwrap_or(v.len())
}

/// Decode `n` tokens from `prompt` the ordinary single-sequence way: the reference every slot
/// answer is compared against, over the public path a normal caller uses (`prefill` +
/// `forward_id`) rather than an internal one.
fn solo(m: &ojas_models::decoder::DecoderGpu, prompt: &[u32], n: usize) -> Vec<u32> {
    m.reset_session();
    m.prefill(&prompt[..prompt.len() - 1], 0);
    let mut out = Vec::with_capacity(n);
    let mut t = prompt[prompt.len() - 1];
    for i in 0..n {
        t = m.forward_id(t, prompt.len() - 1 + i);
        out.push(t);
    }
    out
}

/// Prefill `k` slots, one prompt each, and decode `n` steps with all of them in flight.
/// `corrupt` is `(step, slot)`: that slot's recurrent state is destroyed after that many
/// steps — the negative control.
///
/// Returns one token stream per slot, in slot order.
fn batched(m: &ojas_models::decoder::DecoderGpu, prompts: &[&[u32]], n: usize,
           corrupt: Option<(usize, usize)>) -> Vec<Vec<u32>> {
    let k = prompts.len();
    for (s, p) in prompts.iter().enumerate() {
        // Reset first: recurrent state is a running accumulation and does not self-heal, so a
        // slot reused without this answers from the middle of whatever ran in it before.
        Model::reset_slot(m, s);
        assert!(Model::prefill_slot(m, s, &p[..p.len() - 1], 0), "prefill_slot({s}) refused");
    }
    let mut cur: Vec<u32> = prompts.iter().map(|p| p[p.len() - 1]).collect();
    let mut pos: Vec<usize> = prompts.iter().map(|p| p.len() - 1).collect();
    let mut out = vec![Vec::with_capacity(n); k];
    for i in 0..n {
        if let Some((at, slot)) = corrupt {
            if i == at { Model::reset_slot(m, slot); }
        }
        let steps: Vec<(usize, u32, usize)> =
            (0..k).map(|s| (s, cur[s], pos[s])).collect();
        let ids = Model::decode_slots(m, &steps).expect("decode_slots refused mid-run");
        assert_eq!(ids.len(), k, "decode_slots must answer one id per step");
        for s in 0..k {
            out[s].push(ids[s]);
            cur[s] = ids[s];
            pos[s] += 1;
        }
    }
    out
}

fn show(v: &[u32]) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
}

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: slots_gate <gguf> [n] [prec] [rounds]");
    let n_tok: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let prec: u8 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    // 9 rounds is the floor, not the target: a 3-pair A/B here once gave the opposite sign
    // from a 9-pair one.
    let rounds: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(9);
    anyhow::ensure!(n_tok > 0 && rounds > 0, "token count and rounds must be nonzero");
    ojas_core::logging::init();

    // The gate sets its own slot count rather than requiring the caller to export it: a run
    // that silently fell back to OJAS_SLOTS=1 would report PASS having tested nothing. Safe on
    // edition 2021; read once inside `DecoderGpu::load`, so it must be set before the load.
    std::env::set_var("OJAS_SLOTS", "4");

    let clock_ok = ojas_models::bench::report_load();
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    // `token_embd`'s row count is what `arch.vocab` derives from, so this is the same bound
    // the prefill assert uses.
    let vocab = g.tensors.get("token_embd.weight")
        .and_then(|t| t.dims.get(1).copied()).unwrap_or(0) as usize;
    // Absent metadata leaves this None, which makes `turn_end` return the full length and so
    // grades every token strictly — the conservative direction.
    let eos = g.meta_u32("tokenizer.ggml.eos_token_id");
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 2048, prec, None, None)?;

    let slots = Model::max_slots(&m);
    if slots < 4 {
        // A capability refusal, not a numerical failure: name it and stop, rather than
        // reporting PASS over an untested path.
        println!("GATE: SLOTS FAIL (this model reports max_slots()={slots}; the slot graph \
                  covers the qwen35 hybrid with a dense FFN, no MoE/MLA/qwen4exp/MTP, \
                  no streamed experts and no page-sparse decode)");
        std::process::exit(1);
    }
    for p in PROMPTS {
        anyhow::ensure!(p.iter().all(|&t| (t as usize) < vocab),
            "a gate prompt has ids outside this model's vocabulary");
    }

    // ---- references + the self-check that the prompts discriminate --------------
    let refs: Vec<Vec<u32>> = PROMPTS.iter().map(|p| solo(&m, p, n_tok)).collect();
    for (i, r) in refs.iter().enumerate() {
        println!("solo[{i}] (len {:>2} prompt): {}", PROMPTS[i].len(), show(r));
    }
    let distinct = refs.iter().collect::<std::collections::HashSet<_>>().len();
    if distinct < 2 {
        println!("GATE: SLOTS FAIL (all {} prompts decode to the SAME tokens, so the \
                  cross-slot checks below would compare constants and could not \
                  detect aliased slots — pick prompts this model distinguishes)",
                 PROMPTS.len());
        std::process::exit(1);
    }
    println!("prompts discriminate: {distinct} distinct streams of {}", PROMPTS.len());

    let mut fails: Vec<String> = Vec::new();

    // ---- 1. B=1 through the slot graph == forward_id ----------------------------
    // Not a tautology: the slot graph reaches the M-row kernel family and the head through
    // `chunk_gemm`, where `forward_id` reaches the GEMV family through `mm()`.
    for (i, p) in PROMPTS.iter().enumerate() {
        let got = batched(&m, &[p], n_tok, None).remove(0);
        if got != refs[i] {
            fails.push(format!("B=1 slot graph diverged from forward_id on prompt {i}\n  \
                                want: {}\n  got:  {}", show(&refs[i]), show(&got)));
        }
    }
    println!("B=1 vs forward_id: {}", if fails.is_empty() { "identical on all 4 prompts" } else { "MISMATCH" });

    // ---- 2. every batch width: each slot == its solo run ------------------------
    // Sequences of different lengths share one pass; each must be unable to tell.
    //
    // Every B from 2 to 4, not just the widest, because B is a live index into kernel
    // selection: a projection at B=2 can take a different kernel from the same projection at
    // B=3 (plain Q4 has a tuned two-row kernel and nothing tuned between there and M>=8), and
    // an odd B is the case where a pair-tiled dispatch computes a phantom trailing row.
    //
    // Divergence before the turn ends is a failure, divergence at or after it is drift, per the
    // grading rule in the module header: B>1 cannot be bit-identical to B=1 even in principle,
    // since a batched projection reduces its dot products in a different order and fp32
    // addition is not associative.
    let mut widths: Vec<(usize, Vec<Vec<u32>>)> = Vec::new();
    let mut indep_ok = true;
    let mut drift: Vec<String> = Vec::new();
    for b in 2..=4usize {
        let got = batched(&m, &PROMPTS[..b], n_tok, None);
        for s in 0..b {
            if let Some(i) = first_diff(&refs[s], &got[s]) {
                let end = turn_end(&refs[s], eos);
                if i < end {
                    indep_ok = false;
                    fails.push(format!("B={b} slot {s} diverged from its solo run at token {i}, \
                                        BEFORE its turn ended at {end}\n  want: {}\n  got:  {}",
                                       show(&refs[s]), show(&got[s])));
                } else {
                    drift.push(format!("B={b} slot {s}: agrees for {i} tokens, then drifts at the \
                                        turn boundary (solo ends its turn at token {end})"));
                }
            }
        }
        widths.push((b, got));
    }
    println!("vs solo at B=2,3,4: {}",
             if indep_ok { "every slot agrees on every token up to the end of its turn" } else { "MISMATCH" });
    for d in &drift { println!("  drift (allowed, post-turn): {d}"); }
    // The B=4 streams are the baseline the negative control perturbs.
    let four = widths.pop().expect("B=4 row").1;

    // ---- 2b. companion invariance: the numerics-immune independence proof -------
    // B is held fixed at 4, so every slot takes the same kernels and the same reduction order
    // in both runs and the only change is which sequences occupy the other three slots.
    // Comparing against a solo run instead would conflate slot independence with two different
    // kernels agreeing to the last ulp. If the victim's stream moves, state crossed between
    // slots, and fp32 associativity cannot explain it.
    let victim_p = PROMPTS[3];
    let alt: Vec<Vec<u32>> = (0..3).map(|s| long_prompt(s + 11, 5 + s, vocab)).collect();
    let with_a: Vec<&[u32]> = vec![PROMPTS[0], PROMPTS[1], PROMPTS[2], victim_p];
    let with_b: Vec<&[u32]> = vec![alt[0].as_slice(), alt[1].as_slice(), alt[2].as_slice(), victim_p];
    let run_a = batched(&m, &with_a, n_tok, None);
    let run_b = batched(&m, &with_b, n_tok, None);
    let companion_ok = run_a[3] == run_b[3];
    if !companion_ok {
        fails.push(format!("slot 3's output CHANGED when only its companion slots changed — \
                            state is crossing between slots (B=4 in both runs, so the kernels \
                            and reduction order are identical)\n  with A: {}\n  with B: {}",
                           show(&run_a[3]), show(&run_b[3])));
    }
    // Only meaningful if changing the companions changed something: if slots 0..2 produced the
    // same streams either way, the invariance is vacuous.
    let companions_differed = (0..3).any(|s| run_a[s] != run_b[s]);
    if !companions_differed {
        fails.push("companion-invariance check is vacuous: swapping the companion prompts \
                    changed none of slots 0..2, so slot 3 had nothing to be perturbed by".into());
    }
    println!("companion invariance (slot 3 held, slots 0-2 swapped, B=4 both): {} \
              [companions did change: {companions_differed}]",
             if companion_ok { "slot 3 BYTE-IDENTICAL — no state crosses slots" } else { "CHANGED — LEAKAGE" });

    // ---- 3. negative control ----------------------------------------------------
    // Destroy one slot's recurrent state partway through and re-run the same batch. Two things
    // must both hold, and they fail in opposite directions:
    //   * the corrupted slot's tail must move — otherwise the slots are not carrying separate
    //     state and check 2 proved nothing;
    //   * every other slot's stream must be byte-identical to the clean run, which says the
    //     damage stayed inside its slot.
    let at = n_tok / 2;
    let victim = 2usize;
    let hurt = batched(&m, &PROMPTS, n_tok, Some((at, victim)));
    let moved = hurt[victim] != four[victim];
    let bystanders: Vec<usize> = (0..4).filter(|&s| s != victim && hurt[s] != four[s]).collect();
    if !moved {
        fails.push(format!(
            "negative control: zeroing slot {victim}'s recurrent state at step {at} changed \
             NOTHING — the slots are not carrying independent state, so the equality \
             checks above cannot be trusted\n  {}", show(&hurt[victim])));
    }
    if !bystanders.is_empty() {
        fails.push(format!(
            "negative control: corrupting slot {victim} also moved slot(s) {bystanders:?} — \
             state is LEAKING ACROSS SLOTS\n  slot {}: clean {}\n  slot {}: hurt  {}",
            bystanders[0], show(&four[bystanders[0]]),
            bystanders[0], show(&hurt[bystanders[0]])));
    }
    println!("negative control (zero slot {victim} at step {at}): slot {victim} {} | \
              other slots {}",
             if moved { "MOVED (as it must)" } else { "did NOT move" },
             if bystanders.is_empty() { "unchanged (no leakage)" } else { "CHANGED — leakage" });
    if moved {
        println!("  slot {victim} clean: {}", show(&four[victim]));
        println!("  slot {victim} hurt:  {}", show(&hurt[victim]));
    }

    // ---- 3b. the same independence check above the flash-decoding threshold ----
    // Everything up to here ran at positions < 512, where attention reads the KV cache
    // directly. This run crosses into flash-decoding: a different kernel pair
    // (`attention_part` + `attention_merge`) staging through `st.attn_part`, the only arena
    // buffer slots had to widen and so the only per-slot stride nothing above this touches.
    let longs: Vec<Vec<u32>> = (0..4).map(|s| long_prompt(s, LONG, vocab)).collect();
    let long_refs: Vec<Vec<u32>> = longs.iter().map(|p| solo(&m, p, n_tok)).collect();
    let long_ps: Vec<&[u32]> = longs.iter().map(|p| p.as_slice()).collect();
    let mut long_ok = true;
    let mut long_got = Vec::new();
    for b in 2..=4usize {
        let got = batched(&m, &long_ps[..b], n_tok, None);
        for s in 0..b {
            if let Some(i) = first_diff(&long_refs[s], &got[s]) {
                let end = turn_end(&long_refs[s], eos);
                if i < end {
                    long_ok = false;
                    fails.push(format!("B={b} slot {s} diverged from its solo run at token {i} at \
                                        pos {LONG} (flash-decoding path), BEFORE its turn ended at {end}\
                                        \n  want: {}\n  got:  {}", show(&long_refs[s]), show(&got[s])));
                } else {
                    drift.push(format!("B={b} slot {s} at pos {LONG}: agrees for {i} tokens, then \
                                        drifts at the turn boundary"));
                }
            }
        }
        long_got = got;
    }
    let _ = &long_got;
    let long_distinct = long_refs.iter().collect::<std::collections::HashSet<_>>().len();
    if long_distinct < 2 {
        fails.push(format!("the {LONG}-token prompts all decode alike ({long_distinct} distinct \
                            stream(s)) — the flash-path check cannot detect aliased slots"));
    }
    println!("B=2,3,4 at pos {LONG} (flash-decoding, per-slot attn_part): {} [{long_distinct} distinct streams]",
             if long_ok { "every slot token-identical to its solo run" } else { "MISMATCH" });

    // ---- 4. throughput, interleaved --------------------------------------------
    // One timed unit = `n_tok` decode steps with B slots in flight, so it produces B*n_tok
    // tokens. Prefill is redone before each unit and excluded from the timing; only the decode
    // loop is clocked.
    //
    // GPU-busy time is the primary metric, wall clock the secondary one: this box runs a
    // windowing server and editor renderers that contend for CPU and compositor time, and wall
    // clock charges that to whichever configuration was unlucky — which is how a B=2 step once
    // measured faster than a B=1 step, impossible since B=2 does strictly more work per step.
    // `GPUStartTime`/`GPUEndTime` measure the command buffers themselves and are what the
    // throughput measurements use throughout.
    let steps_n = n_tok.max(64);
    let (mut best_gpu, mut best_wall) = (vec![f64::MAX; 5], vec![f64::MAX; 5]);
    for _ in 0..rounds {
        for b in 1..=4usize {
            // Timed at LONG, not at the 5-9 token prompts: a page is thousands of tokens, so
            // the realistic decode step carries a real KV read and takes the flash path.
            // Measuring at pos ~20 would report a cost the OCR path never pays.
            let ps: Vec<&[u32]> = long_ps[..b].to_vec();
            for (s, p) in ps.iter().enumerate() {
                Model::reset_slot(&m, s);
                assert!(Model::prefill_slot(&m, s, &p[..p.len() - 1], 0), "prefill_slot({s}) refused");
            }
            let mut cur: Vec<u32> = ps.iter().map(|p| p[p.len() - 1]).collect();
            let mut pos: Vec<usize> = ps.iter().map(|p| p.len() - 1).collect();
            let g0 = m.gpu_seconds();
            let t0 = Instant::now();
            for _ in 0..steps_n {
                let steps: Vec<(usize, u32, usize)> = (0..b).map(|s| (s, cur[s], pos[s])).collect();
                let ids = Model::decode_slots(&m, &steps).expect("decode_slots refused while timing");
                for s in 0..b { cur[s] = ids[s]; pos[s] += 1; }
            }
            let (wall, gpu) = (t0.elapsed().as_secs_f64(), m.gpu_seconds() - g0);
            // Contention can only ever make a round slower, so the minimum over rounds is
            // this configuration's uncontended cost.
            if gpu < best_gpu[b] { best_gpu[b] = gpu; }
            if wall < best_wall[b] { best_wall[b] = wall; }
        }
    }
    println!("\nthroughput ({rounds} interleaved rounds, per-B minimum, {steps_n} steps each){}",
             if clock_ok { "" } else { "  [PROVISIONAL: machine was loaded]" });
    println!("  {:>2}  {:>9}  {:>9}  {:>8}   {:>9}  {:>9}",
             "B", "gpu ms", "gpu tok/s", "speedup", "wall ms", "wall tok/s");
    let (bg, bw) = (best_gpu[1] / steps_n as f64, best_wall[1] / steps_n as f64);
    for b in 1..=4usize {
        let (pg, pw) = (best_gpu[b] / steps_n as f64, best_wall[b] / steps_n as f64);
        println!("  {:>2}  {:>9.3}  {:>9.1}  {:>7.2}x   {:>9.3}  {:>9.1}",
                 b, pg * 1e3, b as f64 / pg, bg * b as f64 / pg, pw * 1e3, b as f64 / pw);
    }
    let _ = bw;
    if let Some(l) = ojas_models::bench::load_average() {
        println!("  load average at finish: {l:.2}");
    }

    println!();
    if fails.is_empty() {
        println!("GATE: SLOTS PASS (B=1 matches forward_id; B=2,3,4 over prompts of \
                  lengths {:?} token-identical per slot, short AND at pos {LONG}; \
                  corrupting slot {victim} moved only slot {victim})",
                 PROMPTS.map(|p| p.len()));
        Ok(())
    } else {
        for f in &fails { println!("-- {f}"); }
        println!("GATE: SLOTS FAIL ({} check(s) failed)", fails.len());
        std::process::exit(1);
    }
}
