//! Gate: cross-turn KV-prefix reuse must be token-identical to a full re-prefill.
//!
//! Runs a multi-turn greedy conversation two ways on one resident model and asserts the
//! generated token ids match byte-for-byte per turn:
//!
//!   * Reference — `reset_session()` before every turn, then a full prefill from
//!     position 0. This is what `OJAS_NO_PREFIX_REUSE=1` produces (LCP forced to 0),
//!     so it is the A-B control.
//!   * Reuse — a single continuous conversation: the KV cache and token log carry
//!     across turns and `reuse_prefix_len` skips the shared prefix, re-prefilling
//!     only the changed suffix (plus a small template-divergence slack).
//!
//! Both passes are driven with the same per-turn prompts (built from the reference
//! outputs), so a divergence is caught at the first differing token rather than masked
//! by a drifting conversation.
//!
//! Reuse is bit-exact by construction: dense KV is positional and persistent, so a
//! reused row for a shared token is identical to what a fresh prefill of that token at
//! that position would write. For per-token-prefill archs (qk-norm models like Qwen3)
//! prefill and decode share the `forward_id` kernel, so every position is reproducible;
//! for batched-prefill archs reuse is capped at the prefilled prefix (see
//! `dense_reuse_start`), which this gate also honors.
//!
//! usage: prefix_reuse_gate <gguf> [prec]   (prec default 1 = dequant→Q8, safe)

use anyhow::{bail, Result};
use ojas_core::Model;

const N_TURNS: usize = 3;
const DECODE_N: usize = 24; // tokens generated per turn (fixed; eos ignored)

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: prefix_reuse_gate <gguf> [prec]");
    let prec: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;

    // A shared system prefix longer than the reuse floor (REUSE_MIN_LCP = 256) so reuse
    // fires on turns >= 2. Token ids stay in a mid-vocab band to avoid special/control
    // tokens; this is a token-identity test, so the content only has to be valid.
    let sys: Vec<u32> = (0..300).map(|i| 1000 + (i as u32 * 7) % 4000).collect();
    let users: Vec<Vec<u32>> = (0..N_TURNS)
        .map(|t| (0..10).map(|i| 5000 + (t as u32 * 97 + i as u32 * 13) % 3000).collect())
        .collect();

    // ---- Reference pass: fresh (reset) every turn = full re-prefill from 0 ----
    let mut history: Vec<u32> = Vec::new();
    let mut ref_turns: Vec<Vec<u32>> = Vec::new();
    for user in &users {
        let full = build_full(&sys, &history, user);
        let pre = &full[..full.len() - 1];
        let last = *full.last().unwrap();
        m.reset_session(); // fresh sequence — no reuse of any prior turn
        m.prefill(pre, 0);
        let gen = decode(&m, last, pre.len(), DECODE_N);
        history.extend_from_slice(user);
        history.extend_from_slice(&gen);
        ref_turns.push(gen);
    }

    // ---- Reference for conversation B (the switch scenario): fresh every turn ----
    // B shares only the system prefix with A and is shorter, so switching to it on a
    // warm A cache leaves the prefill high-water mark stale-high — the condition under
    // which a stale HWM failed to cap B-turn-2's reuse against B-turn-1's decode rows.
    let usersB: Vec<Vec<u32>> = (0..2)
        .map(|t| (0..10).map(|i| 8000 + (t as u32 * 131 + i as u32 * 17) % 3000).collect())
        .collect();
    let mut histB: Vec<u32> = Vec::new();
    let mut ref_b: Vec<Vec<u32>> = Vec::new();
    for user in &usersB {
        let full = build_full(&sys, &histB, user);
        let pre = &full[..full.len() - 1];
        let last = *full.last().unwrap();
        m.reset_session();
        m.prefill(pre, 0);
        let gen = decode(&m, last, pre.len(), DECODE_N);
        histB.extend_from_slice(user);
        histB.extend_from_slice(&gen);
        ref_b.push(gen);
    }

    // ---- Reuse pass: one continuous conversation, reuse the shared prefix ----
    m.reset_session(); // clean start for the reuse conversation
    let mut history2: Vec<u32> = Vec::new(); // built from the reference gens (same prompts)
    let mut ok = true;
    let mut actual_reuse = false;
    for (t, user) in users.iter().enumerate() {
        let full = build_full(&sys, &history2, user);
        let pre = &full[..full.len() - 1];
        let last = *full.last().unwrap();
        let start = m.reuse_prefix_len(pre);
        if start < pre.len() {
            m.prefill(&pre[start..], start);
        }
        let gen = decode(&m, last, pre.len(), DECODE_N);
        let reused = start.max(m.last_prefill_reused());
        actual_reuse |= reused > 0;
        let identical = gen == ref_turns[t];
        println!(
            "turn {t}: prompt={} reused={reused} reprefilled={} -> {}",
            pre.len(),
            pre.len().saturating_sub(reused),
            if identical { "TOKEN-IDENTICAL" } else { "MISMATCH" },
        );
        if !identical {
            ok = false;
            let at = ref_turns[t].iter().zip(&gen).position(|(a, b)| a != b);
            println!("  first diff at {at:?}\n  want: {:?}\n  got:  {:?}", ref_turns[t], gen);
        }
        // Advance the conversation using the reference output so both passes see
        // byte-identical prompts (a bug shows as a token mismatch, not drift).
        history2.extend_from_slice(user);
        history2.extend_from_slice(&ref_turns[t]);
    }

    // ---- Switch pass: continue on conversation A's warm cache (no reset) with the
    //      shorter conversation B — the shrink/diverge case. With a stale-high HWM,
    //      turn 1 of B reuses less than the HWM (fine) but turn 2 reuses B-turn-1's
    //      decode-written rows the stale mark failed to cap, a token mismatch on
    //      batched archs. Correct behavior is byte-identical to the fresh B reference. --
    let mut histB2: Vec<u32> = Vec::new();
    for (t, user) in usersB.iter().enumerate() {
        let full = build_full(&sys, &histB2, user);
        let pre = &full[..full.len() - 1];
        let last = *full.last().unwrap();
        let start = m.reuse_prefix_len(pre);
        if start < pre.len() {
            m.prefill(&pre[start..], start);
        }
        let gen = decode(&m, last, pre.len(), DECODE_N);
        let identical = gen == ref_b[t];
        println!(
            "switch-B turn {t}: prompt={} reused={start} -> {}",
            pre.len(),
            if identical { "TOKEN-IDENTICAL" } else { "MISMATCH" },
        );
        if !identical {
            ok = false;
            let at = ref_b[t].iter().zip(&gen).position(|(a, b)| a != b);
            println!("  first diff at {at:?}\n  want: {:?}\n  got:  {:?}", ref_b[t], gen);
        }
        histB2.extend_from_slice(user);
        histB2.extend_from_slice(&ref_b[t]);
    }

    // The feature must actually have engaged, or the gate is vacuous.
    let any_reuse = {
        // recompute the reuse the last turn would see against a warm cache
        let full = build_full(&sys, &history2, &users[0]);
        m.reuse_prefix_len(&full[..full.len() - 1]) > 0
    };

    if !ok {
        bail!("GATE: TOKEN MISMATCH — reuse is NOT byte-identical to full re-prefill");
    }
    if !any_reuse && !actual_reuse {
        bail!("GATE: reuse never fired — test is vacuous (check REUSE_MIN_LCP vs prompt length)");
    }
    println!("GATE: PREFIX-REUSE TOKEN-IDENTICAL PASS ({N_TURNS} turns, {DECODE_N} tok/turn)");
    Ok(())
}

/// Full transcript token stream for a turn: shared system prefix, prior turns, new user.
fn build_full(sys: &[u32], history: &[u32], user: &[u32]) -> Vec<u32> {
    let mut v = Vec::with_capacity(sys.len() + history.len() + user.len());
    v.extend_from_slice(sys);
    v.extend_from_slice(history);
    v.extend_from_slice(user);
    v
}

/// Greedy decode `n` tokens starting from `cur` at `pos` (eos ignored — fixed length).
fn decode(m: &ojas_models::decoder::DecoderGpu, mut cur: u32, mut pos: usize, n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        cur = m.forward_id(cur, pos);
        pos += 1;
        out.push(cur);
    }
    out
}
