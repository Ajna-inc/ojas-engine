//! ojas-infer — the serving frontend. `EngineCore` drives any `ojas_core::Model`
//! (prefill → decode loop, greedy or sampled). Sessions/spec/diffusion entries
//! remain on the concrete model types.

use ojas_core::cancel::STREAM_CANCEL;
use ojas_core::Model;
use std::sync::atomic::Ordering;

/// Sampling controls for [`EngineCore::generate_with`]. Deterministic for a fixed
/// seed: sampling uses a local xorshift RNG, not global randomness.
#[derive(Debug, Clone)]
pub struct SampleOpts {
    pub temperature: f32,     // ≤0 → greedy
    pub top_p: f32,           // nucleus mass (1.0 = off)
    pub top_k: usize,         // keep only k best candidates (0 = off)
    pub repeat_penalty: f32,  // 1.0 = off (divides logits of recent tokens)
    pub repeat_window: usize, // how far back the penalty looks
    pub seed: u64,
}

impl Default for SampleOpts {
    fn default() -> Self {
        SampleOpts { temperature: 0.8, top_p: 0.95, top_k: 0, repeat_penalty: 1.1, repeat_window: 64, seed: 42 }
    }
}

/// xorshift64* — tiny deterministic RNG (no dependency, stable across builds).
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Temperature → repetition penalty → top-p nucleus sample over `logits`.
fn sample_logits(logits: &mut [f32], recent: &[u32], o: &SampleOpts, rng: &mut Rng) -> usize {
    if o.repeat_penalty > 1.0 {
        for &t in recent {
            let l = &mut logits[t as usize];
            *l = if *l > 0.0 { *l / o.repeat_penalty } else { *l * o.repeat_penalty };
        }
    }
    let inv_t = 1.0 / o.temperature.max(1e-4);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    // partial top-256 selection is plenty for nucleus sampling
    let k = if o.top_k > 0 { o.top_k.min(idx.len()) } else { 256.min(idx.len()) };
    idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
    idx.truncate(k);
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let mx = logits[idx[0]];
    let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i] - mx) * inv_t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() { *p /= sum; }
    let mut mass = 0.0;
    let mut keep = probs.len();
    for (i, &p) in probs.iter().enumerate() {
        mass += p;
        if mass >= o.top_p { keep = i + 1; break; }
    }
    let total: f32 = probs[..keep].iter().sum();
    let mut r = rng.next_f32() * total;
    for i in 0..keep {
        r -= probs[i];
        if r <= 0.0 { return idx[i]; }
    }
    idx[keep - 1]
}

/// The frontend decode/prefill host loop, generic over the model contract.
/// open GGUF → build model → `EngineCore::new(model).generate(ids, n, greedy)`.
pub struct EngineCore<M: Model> {
    model: M,
    /// stop generation when this token is produced (None = run to max_tokens)
    pub eos: Option<u32>,
    /// Full end-of-generation set. A checkpoint can have several stop tokens — a
    /// Qwen-family model reaches `<|endoftext|>` before the `<|im_end|>` named in
    /// `tokenizer.ggml.eos_token_id` — so watching `eos` alone lets generation emit
    /// its own terminator and run on into degeneration. Empty = fall back to `eos`.
    pub eog: Vec<u32>,
    /// Ids that may never be selected, the contract llama.cpp's `ignore_eos`
    /// implements. Distinct from declining to stop, which still emits the
    /// terminator and everything after it. Non-empty forces the logits path,
    /// because an on-device argmax cannot exclude an id.
    pub banned: Vec<u32>,
    /// Token Recycling adjacency (Luo et al., ACL 2025): token -> the model's top-8
    /// next-token guesses, harvested from logits every forward already computes.
    /// Drafts drawn from here fire on novel text, where prompt-lookup cannot (it
    /// only repeats text it has seen). Persists across `generate` calls.
    adj: std::cell::RefCell<std::collections::HashMap<u32, [u32; 8]>>,
}

/// OJAS_ADJ_DRAFT=1 enables Token Recycling adjacency drafts (Luo et al., ACL
/// 2025) alongside prompt-lookup. Off by default: with this engine's verify cost
/// (M=4 verify ~2.4x a single forward) chain-only adjacency drafts measured
/// 195->157 tok/s repetitive and 157->90 novel. The paper's 2.17x on novel text
/// comes from an ~80-node draft tree verified nearly for free on server GPUs; a
/// chain is all this engine's verify cost supports, and a chain accepts at
/// break-even. The top-8 harvest stays wired up because a cheap wide-batch verify
/// or Medusa heads would make the tree viable.
fn adj_draft_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| ojas_core::config::EngineConfig::current().adj_draft)
}

/// OJAS_NO_SPEC=1 turns speculative decoding off — the A/B control for spec_gate,
/// which asserts the two paths emit identical tokens.
fn spec_disabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| ojas_core::config::EngineConfig::current().no_spec)
}

/// Prompt-lookup draft: propose up to `k` tokens that followed the most recent
/// earlier occurrence of the current suffix. No draft model, no training.
///
/// Tries the longest n-gram first, scanning backward from the most recent
/// occurrence: a longer context match is a higher-confidence draft and raises the
/// accept rate.
///
/// The >=3-gram floor matters. Common bigrams ("of the", "in a") match constantly
/// on novel text but their continuations differ, so the batched verify is wasted
/// work; requiring 3 makes a match mean the exact phrase occurred before, which is
/// what keeps the drafter net-positive.
pub fn draft_lookup(gen: &[u32], k: usize, max_ngram: usize) -> Vec<u32> {
    for ngram in (3..=max_ngram).rev() {
        if gen.len() < ngram + 1 { continue; }
        let suf = &gen[gen.len() - ngram..];
        let hi = gen.len() - ngram;
        for start in (0..hi).rev() {
            if &gen[start..start + ngram] == suf {
                let fs = start + ngram;
                let end = (fs + k).min(gen.len());
                if end > fs { return gen[fs..end].to_vec(); }
            }
        }
    }
    Vec::new()
}

/// Draft tokens proposed per speculative step (OJAS_SPEC_DRAFT, default 3), and
/// the longest n-gram tried. batch = current token + draft, and the batched path is
/// cheapest at M <= 4, where gemv_m4_q4l keeps the weights in registers across
/// tokens (M=4 verify 24.9 ms vs a single forward's 8.4 ms, so break-even is ~3
/// accepted). Above that the padded GEMM takes over and the cost more than doubles.
fn spec_draft() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OJAS_SPEC_DRAFT").ok().and_then(|v| v.parse().ok()).unwrap_or(3))
}
const SPEC_NGRAM: usize = 4;

impl<M: Model> EngineCore<M> {
    pub fn new(model: M) -> Self {
        EngineCore { model, eos: None, eog: Vec::new(), banned: Vec::new(), adj: Default::default() }
    }

    pub fn model(&self) -> &M { &self.model }

    /// Whether `t` ends generation: the full EOG set, or `eos` when none is set.
    fn is_stop(&self, t: u32) -> bool {
        if self.eog.is_empty() { Some(t) == self.eos } else { self.eog.contains(&t) }
    }

    /// Argmax over `logits` excluding the banned ids.
    fn best_allowed(&self, logits: &[f32]) -> u32 {
        let mut best = (f32::NEG_INFINITY, 0u32);
        for (i, &v) in logits.iter().enumerate() {
            let id = i as u32;
            if self.banned.contains(&id) { continue; }
            if v > best.0 { best = (v, id); }
        }
        best.1
    }

    /// Generate up to `max_tokens` ids after `prompt_ids` (greedy or default-sampled).


    pub fn generate(&self, prompt_ids: &[u32], max_tokens: usize, greedy: bool) -> Vec<u32> {
        let opts = if greedy { None } else { Some(SampleOpts::default()) };
        self.generate_with(prompt_ids, max_tokens, opts.as_ref(), &mut |_, _| {}, &mut |_| true)
    }

    /// [`EngineCore::generate_with`], but refusing to run — or to return tokens —
    /// when a device fault is latched.
    ///
    /// A latched fault means a command buffer failed and its output buffers are
    /// undefined; the recurrent state, KV cache and expert tables may be
    /// half-written, so continuing would emit plausible wrong text. Recovery is
    /// reinitialisation, not retry.
    pub fn try_generate_with(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        opts: Option<&SampleOpts>,
        on_prefill: &mut dyn FnMut(usize, usize),
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, ojas_core::device_fault::DeviceError> {
        if let Some(err) = ojas_core::device_fault::peek() {
            return Err(err);
        }
        let out = self.generate_with(prompt_ids, max_tokens, opts, on_prefill, on_token);
        // A fault raised during generation poisons the result: the tokens after the
        // failing dispatch were read out of undefined buffers.
        match ojas_core::device_fault::peek() {
            Some(err) => Err(err),
            None => Ok(out),
        }
    }

    /// Full loop: prefill all but the last prompt token, then decode step by step.
    /// `opts` None = greedy argmax on-device (`forward_id`); Some = sampled via
    /// `forward_logits` when the model exposes logits (greedy fallback otherwise).
    /// `on_token` returns false to stop early. u32::MAX from the model = cancelled.
    pub fn generate_with(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        opts: Option<&SampleOpts>,
        // Prefill progress as (tokens_processed, total_prompt_tokens) after each
        // chunk, so a caller can report progress before the first token.
        on_prefill: &mut dyn FnMut(usize, usize),
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Vec<u32> {
        if prompt_ids.is_empty() || max_tokens == 0 { return Vec::new(); }
        let capacity = self.model.context_capacity();
        assert!(prompt_ids.len() <= capacity, "prompt exceeds allocated model context");
        let max_tokens = max_tokens.min(capacity.saturating_sub(prompt_ids.len()).saturating_add(1));
        let mut out = Vec::with_capacity(max_tokens.min(4096));
        STREAM_CANCEL.store(false, Ordering::Relaxed);

        let mut rng = Rng(opts.map(|o| o.seed).unwrap_or(42) | 1);
        let mut recent: Vec<u32> = opts.map(|o| {
            prompt_ids.iter().rev().take(o.repeat_window).copied().collect()
        }).unwrap_or_default();

        // Cross-turn KV-prefix reuse: skip the leading tokens the cache already holds
        // from the previous turn (dense KV is positional and persistent). Computed on
        // `pre` (all but the last token) unconditionally — for a 1-token prompt `pre`
        // is empty, which resets the session log so decode tracks from pos 0 instead
        // of desyncing against a stale previous sequence. `start` is 0 for a fresh or
        // diverged sequence, and when reuse is disabled, giving a full prefill. The
        // model guarantees the reused rows are bit-identical to a fresh prefill.
        let pre = &prompt_ids[..prompt_ids.len() - 1];
        let start = self.model.reuse_prefix_len(pre).min(pre.len());

        let mut pos = 0usize;
        if prompt_ids.len() == 1 { self.model.prefill(&[], 0); }
        if prompt_ids.len() > 1 {
            // Prefill in progress-reporting chunks. PREFILL_STEP == MAXM, the model's
            // own batch granularity, so chunking adds no GPU passes; it only allows
            // progress reports and a cancel mid-prefill.
            const PREFILL_STEP: usize = 256;
            let total = pre.len();
            on_prefill(start, total);
            let mut done = start;
            while done < total {
                if STREAM_CANCEL.load(Ordering::Relaxed) {
                    return out;
                }
                let end = (done + PREFILL_STEP).min(total);
                self.model.prefill(&pre[done..end], done);
                done = end;
                on_prefill(done, total);
            }
            pos = prompt_ids.len() - 1;
        }
        let mut cur = *prompt_ids.last().unwrap();

        // Speculative decoding via prompt-lookup, greedy only.
        //
        // A batched forward reads each weight once and applies it to all M tokens, so
        // verifying a draft costs about one forward's bandwidth rather than M. A right
        // draft emits several tokens per GPU pass; a wrong one costs the single
        // forward it would have cost anyway. Output is exact: a drafted token is kept
        // only where it equals the model's argmax.
        //
        // Sampled requests take the single-token path below. Exact speculative
        // sampling would need modified rejection sampling against the draft
        // distribution, which is not implemented.
        let greedy = opts.map(|o| o.temperature <= 0.0).unwrap_or(true);
        // Suppression and speculation are incompatible as built: prompt-lookup,
        // adjacency and MTP all commit ids the model chose, and none can be told an
        // id is forbidden. Verifying against a banned argmax would either emit the
        // banned token or change what "verified" means, so speculation is disabled
        // rather than given different semantics from the single-token path.
        let suppressing = !self.banned.is_empty();
        let mut spec_on = greedy && !spec_disabled() && !suppressing;
        // Full token history (prompt + output) is what the drafter searches.
        let mut hist: Vec<u32> = if spec_on { prompt_ids.to_vec() } else { Vec::new() };

        // Self-calibrating gate. A batched verify is not free: on this engine a single
        // forward is ~8.4 ms while an M=8 verify is ~55.8 ms, because the batched
        // kernels are tuned for M=256 prefill, not M=8 (the tile is 64x32, so a small
        // M still dispatches one M-tile and the grid collapses). Break-even is
        // therefore ~6.6 accepted tokens, not ~1.5, and a fixed "draft whenever a
        // 3-gram matches" rule measured 67 vs 108 tok/s on novel text.
        //
        // So instead of a hardcoded threshold, which would be wrong on other models
        // or hardware, both paths are timed and drafting continues only while the
        // measured accept rate pays for the measured verify cost.
        let (mut t_single, mut n_single) = (0.0f64, 0u32);
        let (mut t_verify, mut n_verify) = (0.0f64, 0u32);
        let mut acc_sum = 0.0f64;
        // --- instrumentation (OJAS_SPEC_STATS) ---
        let stats_on = std::env::var("OJAS_SPEC_STATS").is_ok();
        let (mut n_calls, mut n_hit, mut t_search) = (0u64, 0u64, 0.0f64);
        let mut acc_hist = [0u64; 16];
        let mut acc_all = 0.0f64;
        let mut spec_off_at: Option<(usize, &'static str)> = None;
        let mut n_gate_open = 0u64;
        // Adjacency (Token Recycling) drafts are gated separately: their accept rate
        // is lower than lookup's, so a shared gate would either poison the lookup path
        // or let a losing drafter keep firing.
        let (mut adj_t, mut adj_n, mut adj_acc) = (0.0f64, 0u32, 0.0f64);
        // The model's own NextN head, gated like the others. It drafts one token from
        // a block that shares the backbone, so it fires on novel text where lookup
        // finds nothing, but it costs a real forward.
        let (mut mtp_t, mut mtp_n, mut mtp_acc) = (0.0f64, 0u32, 0.0f64);
        let mtp_on = greedy && !spec_disabled() && self.model.has_mtp();
        // Whether the measured accept rate pays for the measured verify cost.
        //
        // Returns false, not true, when there is nothing to compare (`n_single == 0`,
        // no ordinary step ever timed). A drafter that hits on every iteration never
        // lets the ordinary path run, so `n_single` would stay zero and the gate could
        // never switch an unproductive drafter off; repetitive output has exactly that
        // shape (markup, a tool loop, a model quoting itself). Bootstrapping therefore
        // belongs at the call sites (`n_verify < 4`, `adj_n < 8`, `mtp_n < 4`).
        let worth_it = |t_single: f64, n_single: u32, t_verify: f64, n_verify: u32, acc_sum: f64| -> bool {
            if n_verify < 2 || n_single == 0 { return false; }  // nothing to compare
            let avg_acc = acc_sum / n_verify as f64;
            let ts = t_single / n_single as f64;
            let tv = t_verify / n_verify as f64;
            avg_acc * ts > tv * 1.05                            // 5% margin
        };

        while out.len() < max_tokens {
            // Only pay for a batched verify when there is a confident (>=3-gram)
            // match. On novel text the drafter returns nothing and costs one array
            // scan, so spec mode is never much slower than single stream. Re-probe
            // every 64 tokens: a request can start novel and turn repetitive (a model
            // quoting its own earlier output, a tool loop).
            let probe = out.len() % 64 == 0 && n_single > 0;
            // `n_verify < 4` is the bootstrap; `n_verify - 1` skips the cold start,
            // because the first verify of a session compiles pipelines and faults
            // buffers in.
            let lookup_ok = n_verify < 4
                || worth_it(t_single, n_single, t_verify, n_verify - 1, acc_sum);
            if spec_on && (probe || lookup_ok) {
                n_gate_open += 1;
                // Timed from here, not from after the draft is built: the n-gram scan
                // is only paid because drafting is enabled, so it belongs to the
                // verify's cost.
                let tv0 = std::time::Instant::now();
                let mut draft = draft_lookup(&hist, spec_draft(), SPEC_NGRAM);
                if stats_on {
                    t_search += tv0.elapsed().as_secs_f64();
                    n_calls += 1;
                    if !draft.is_empty() { n_hit += 1; }
                }
                let mut from_adj = false;
                if draft.is_empty() && adj_draft_enabled()
                    && (adj_n < 8 || worth_it(t_single, n_single, adj_t, adj_n - 1, adj_acc))
                {
                    // Token Recycling chain: walk the adjacency table greedily from
                    // the current token. Fires on novel text where lookup missed.
                    let adj = self.adj.borrow();
                    let mut t = cur;
                    for _ in 0..spec_draft() {
                        match adj.get(&t) {
                            Some(row) => { t = row[0]; draft.push(t); }
                            None => break,
                        }
                    }
                    from_adj = !draft.is_empty();
                }
                draft.truncate(spec_draft().min(capacity.saturating_sub(pos).saturating_sub(1)));
                if !draft.is_empty() {
                    let mut batch = Vec::with_capacity(draft.len() + 1);
                    batch.push(cur);
                    batch.extend_from_slice(&draft);
                    // With the adjacency drafter on, use the top-k variant: same
                    // verify, plus each position's top-8 recycled. Otherwise skip the
                    // harvest, which would have no consumer.
                    let got = match if adj_draft_enabled() { self.model.forward_batch_topk(&batch, pos) } else { None } {
                        Some((ids, tk)) => {
                            let mut adj = self.adj.borrow_mut();
                            for (j, &t) in batch.iter().enumerate() {
                                let mut row = [0u32; 8];
                                row.copy_from_slice(&tk[j * 8..(j + 1) * 8]);
                                adj.insert(t, row);
                            }
                            Some(ids)
                        }
                        None => self.model.forward_batch_ids(&batch, pos),
                    };
                    match got {
                        None => { spec_on = false; spec_off_at = Some((out.len(), "no batched verify")); }  // model can't verify; stop trying
                        Some(ids) if ids.len() == batch.len() => {
                            // ids[i] is the model's argmax after batch[i], i.e. the true
                            // next token at that position. Keep drafts while they match.
                            let mut accepted = vec![ids[0]];
                            let mut i = 0usize;
                            while i < draft.len() && draft[i] == accepted[i] {
                                accepted.push(ids[i + 1]);
                                i += 1;
                            }
                            if from_adj {
                                adj_t += tv0.elapsed().as_secs_f64();
                                adj_n += 1;
                                adj_acc += accepted.len() as f64;
                            } else {
                                // Steady state only: charging the first verify's
                                // one-off warm-up would disable a fast drafter.
                                if n_verify > 0 {
                                    t_verify += tv0.elapsed().as_secs_f64();
                                    acc_sum += accepted.len() as f64;
                                }
                                n_verify += 1;
                                acc_all += accepted.len() as f64;
                                acc_hist[accepted.len().min(15)] += 1;
                            }
                            let mut stop = false;
                            if ojas_core::device_fault::is_faulted() { break; }
                            for t in accepted {
                                pos += 1;
                                out.push(t);
                                hist.push(t);
                                cur = t;
                                if self.is_stop(t) { stop = true; break; }
                                if !on_token(t) { stop = true; break; }
                                if out.len() >= max_tokens { stop = true; break; }
                            }
                            if stop { break; }
                            continue;
                        }
                        Some(_) => { spec_on = false; spec_off_at = Some((out.len(), "verify length mismatch")); }
                    }
                }
            }

            // NextN draft: tried after lookup and adjacency, which cost one array scan
            // each while this costs a forward through the draft block. After the
            // initial probes at least one ordinary step must be measured, or MTP
            // consumes every iteration, `n_single` stays zero, and `worth_it`'s
            // bootstrap keeps a losing drafter on forever.
            let mtp_probe = mtp_n < 4 || (probe && n_single > 0);
            let mtp_worthwhile = n_single > 0 && worth_it(t_single, n_single, mtp_t, mtp_n.saturating_sub(1), mtp_acc);
            if mtp_on && self.model.mtp_verify_width() <= capacity.saturating_sub(pos) && (mtp_probe || mtp_worthwhile) {
                let tm0 = std::time::Instant::now();
                if let Some(committed) = self.model.mtp_step_committed(cur, pos) {
                    if STREAM_CANCEL.load(Ordering::Relaxed) { return out; }
                    // The first verification faults GPU resources into memory. Keep its
                    // output but exclude its cost, comparing steady state over the
                    // remaining probes, so startup does not disable useful MTP.
                    if mtp_n > 0 {
                        mtp_t += tm0.elapsed().as_secs_f64();
                        mtp_acc += committed.len() as f64;
                    }
                    mtp_n += 1;
                    let mut stop = false;
                    for t in committed {
                        pos += 1;
                        out.push(t);
                        if spec_on { hist.push(t); }
                        cur = t;
                        if self.is_stop(t) { stop = true; break; }
                        if !on_token(t) { stop = true; break; }
                        if out.len() >= max_tokens { stop = true; break; }
                    }
                    if stop { break; }
                    continue;
                }
            }

            let ts0 = std::time::Instant::now();
            // Suppression needs the distribution: an on-device argmax cannot skip an id.
            if suppressing {
                let logits = self.model.forward_logits(cur, pos);
                cur = match logits {
                    Some(l) => self.best_allowed(&l),
                    // A model with no logits path cannot honour a ban, so stop rather
                    // than emit a token the caller excluded.
                    None => break,
                };
                if cur == u32::MAX { break; }
                pos += 1;
                out.push(cur);
                if self.is_stop(cur) { break; }
                if !on_token(cur) { break; }
                if out.len() >= max_tokens { break; }
                continue;
            }
            let sampled = opts.filter(|o| o.temperature > 0.0).and_then(|o| {
                self.model.forward_logits(cur, pos)
                    .map(|mut lg| sample_logits(&mut lg, &recent, o, &mut rng) as u32)
            });
            cur = match sampled {
                Some(t) => t,
                None => {
                    if spec_on && adj_draft_enabled() {
                        // Recycle the single-stream steps too: these are the novel-text
                        // stretches where the adjacency table has to learn.
                        match self.model.forward_id_topk(cur, pos) {
                            Some((id, tk)) => { self.adj.borrow_mut().insert(cur, tk); id }
                            None => self.model.forward_id(cur, pos),
                        }
                    } else {
                        self.model.forward_id(cur, pos)
                    }
                }
            };
            t_single += ts0.elapsed().as_secs_f64();
            n_single += 1;
            if cur == u32::MAX { break; } // cancelled mid-forward
            // A failed command buffer leaves the output buffers undefined, so this
            // token is meaningless. Stop before emitting it.
            if ojas_core::device_fault::is_faulted() { break; }
            pos += 1;
            out.push(cur);
            if spec_on { hist.push(cur); }
            if self.is_stop(cur) { break; }
            if !on_token(cur) { break; }
            if let Some(o) = opts {
                recent.push(cur);
                if recent.len() > o.repeat_window { recent.remove(0); }
            }
        }
        if stats_on {
            let ts = if n_single > 0 { t_single / n_single as f64 * 1e3 } else { 0.0 };
            let tv = if n_verify > 1 { t_verify / (n_verify - 1) as f64 * 1e3 } else { 0.0 };
            let avg = if n_verify > 0 { acc_all / n_verify as f64 } else { 0.0 };
            eprintln!(
                "\n[spec] out={} hist={} gate_open={n_gate_open} lookup_calls={n_calls} hits={n_hit} ({:.1}%) \
                 verifies={n_verify} accepted={:.0} avg_accept={avg:.3} \
                 t_verify={tv:.3}ms t_single={ts:.3}ms t_search_total={:.1}ms singles={n_single} \
                 spec_off_at={:?} worth={} hist_accept={:?}",
                out.len(), hist.len(),
                if n_calls > 0 { n_hit as f64 * 100.0 / n_calls as f64 } else { 0.0 },
                acc_all,
                t_search * 1e3,
                spec_off_at,
                worth_it(t_single, n_single, t_verify, n_verify.saturating_sub(1), acc_sum),
                &acc_hist[..6],
            );
            if mtp_n > 0 || self.model.has_mtp() {
                eprintln!("[spec] mtp_calls={mtp_n} mtp_acc={mtp_acc:.0} mtp_t={:.3}ms", if mtp_n > 1 { mtp_t / (mtp_n - 1) as f64 * 1e3 } else { 0.0 });
            }
        }
        out
    }
}

#[cfg(test)]
mod device_fault_tests {
    use super::*;
    use ojas_core::Model;
    use std::cell::Cell;

    /// Fails on the Nth decode step the way a real command-buffer failure does:
    /// latch the fault, then return a value read from undefined buffers.
    struct FailsAtStep {
        step: Cell<usize>,
        fail_at: usize,
        calls: Cell<usize>,
    }
    impl Model for FailsAtStep {
        fn n_layers(&self) -> usize { 1 }
        fn hidden_dim(&self) -> usize { 4 }
        fn context_capacity(&self) -> usize { 256 }
        fn prefill(&self, _t: &[u32], _p: usize) {}
        fn forward_id(&self, _t: u32, _p: usize) -> u32 {
            self.calls.set(self.calls.get() + 1);
            let n = self.step.get();
            self.step.set(n + 1);
            if n == self.fail_at {
                ojas_core::device_fault::inject_for_test("test decode step");
            }
            // Whatever a failed dispatch left behind. The engine must not emit it.
             7
        }
    }

    fn fresh() -> EngineCore<FailsAtStep> {
        let _ = ojas_core::device_fault::take();
        EngineCore::new(FailsAtStep { step: Cell::new(0), fail_at: 2, calls: Cell::new(0) })
    }

    /// A failed dispatch leaves stale buffers in place; unguarded, decoding carries
    /// on and turns a device error into fluent wrong text.
    #[test]
    fn a_fault_stops_generation_and_is_reported() {
        let core = fresh();
        let mut emitted = Vec::new();
        let err = core
            .try_generate_with(&[1, 2], 16, None, &mut |_, _| {}, &mut |t| { emitted.push(t); true })
            .expect_err("a faulted run must not report success");
        assert_eq!(err.domain, "TestInjected");
        // Steps 0 and 1 succeeded; step 2 failed and must not be emitted.
        assert_eq!(emitted.len(), 2, "no token may be emitted from failed work: {emitted:?}");
        let _ = ojas_core::device_fault::take();
    }

    /// A failed command buffer may have half-written recurrent state or caches, so
    /// the session stays refused until it is rebuilt — retrying is not recovery.
    #[test]
    fn a_poisoned_session_refuses_further_generation() {
        let core = fresh();
        let _ = core.try_generate_with(&[1, 2], 8, None, &mut |_, _| {}, &mut |_| true);
        assert!(ojas_core::device_fault::is_faulted());
        let mut emitted = Vec::new();
        let err = core
            .try_generate_with(&[1, 2], 8, None, &mut |_, _| {}, &mut |t| { emitted.push(t); true })
            .expect_err("reuse after a fault must be refused");
        assert_eq!(err.domain, "TestInjected");
        assert!(emitted.is_empty(), "a refused request must not decode at all");
        let _ = ojas_core::device_fault::take();
    }

    #[test]
    fn a_clean_session_still_generates() {
        let _ = ojas_core::device_fault::take();
        let core = EngineCore::new(FailsAtStep { step: Cell::new(0), fail_at: usize::MAX, calls: Cell::new(0) });
        let out = core
            .try_generate_with(&[1, 2], 5, None, &mut |_, _| {}, &mut |_| true)
            .expect("no fault, so no error");
        assert_eq!(out.len(), 5, "the guard must not break ordinary generation");
    }
}

#[cfg(test)]
mod context_tests {
    use super::*;
    use std::cell::RefCell;
    static GENERATION_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
    #[derive(Default)]
    struct Bounded { calls: RefCell<Vec<(usize, usize)>>, cancel: bool }
    impl Model for Bounded {
        fn context_capacity(&self) -> usize { 8 }
        fn mtp_verify_width(&self) -> usize { 3 }
        fn n_layers(&self) -> usize { 1 }
        fn hidden_dim(&self) -> usize { 1 }
        fn prefill(&self, t: &[u32], p: usize) {
            assert!(p+t.len()<=8); self.calls.borrow_mut().push((p,t.len()));
        }
        fn forward_id(&self, _: u32, p: usize) -> u32 {
            assert!(p<8); self.calls.borrow_mut().push((p,1)); 1
        }
        fn has_mtp(&self) -> bool { true }
        fn mtp_step_committed(&self, _: u32, p: usize) -> Option<Vec<u32>> {
            if self.cancel { STREAM_CANCEL.store(true, Ordering::Relaxed); }
            assert!(p+3<=8); self.calls.borrow_mut().push((p,3)); Some(vec![1,1,1])
        }
    }
    #[test]
    fn generation_and_speculation_stay_inside_context() {
        let _guard = GENERATION_TEST.lock().unwrap();
        for n in [1, 5, 7, 8] {
            let core=EngineCore::new(Bounded::default());
            let out=core.generate_with(&vec![1;n],100,None,&mut |_,_| {},&mut |_| true);
            assert_eq!(out.len(),9-n);
            assert!(core.model.calls.borrow().iter().all(|(p,n)| p+n<=8));
            if n==1 { assert_eq!(core.model.calls.borrow()[0],(0,0)); }
        }
    }
    #[test]
    fn cancellation_during_mtp_emits_no_committed_batch() {
        let _guard = GENERATION_TEST.lock().unwrap();
        let core=EngineCore::new(Bounded{cancel:true,..Default::default()});
        let mut callbacks=0;
        let out=core.generate_with(&[1],8,None,&mut |_,_| {},&mut |_| {callbacks+=1;true});
        assert!(out.is_empty());
        assert_eq!(callbacks,0);
    }
    #[test]
    fn slow_mtp_is_measured_against_scalar_decode_and_disabled() {
        let _guard = GENERATION_TEST.lock().unwrap();
        struct SlowDraft { drafts: std::cell::Cell<usize>, singles: std::cell::Cell<usize> }
        impl Model for SlowDraft {
            fn n_layers(&self) -> usize { 1 }
            fn hidden_dim(&self) -> usize { 1 }
            fn has_mtp(&self) -> bool { true }
            fn prefill(&self, _: &[u32], _: usize) {}
            fn forward_id(&self, _: u32, _: usize) -> u32 { self.singles.set(self.singles.get()+1); 1 }
            fn mtp_step_committed(&self, _: u32, _: usize) -> Option<Vec<u32>> {
                self.drafts.set(self.drafts.get()+1);
                std::thread::sleep(std::time::Duration::from_millis(5));
                Some(vec![1,1])
            }
        }
        let core=EngineCore::new(SlowDraft{drafts:std::cell::Cell::new(0),singles:std::cell::Cell::new(0)});
        assert_eq!(core.generate(&[1],32,true),vec![1;32]);
        assert_eq!(core.model.drafts.get(),4,"a losing draft must stop after the initial probes");
        assert_eq!(core.model.singles.get(),24);
    }
    /// The prompt-lookup twin of `slow_mtp_...`. On repetitive output the n-gram
    /// drafter hits on every iteration, so without a call-site probe the ordinary
    /// single-token path is never reached, `n_single` stays zero, and `worth_it`'s
    /// bootstrap clause keeps a losing drafter on forever.
    #[test]
    fn slow_lookup_verify_is_measured_against_scalar_decode_and_disabled() {
        let _guard = GENERATION_TEST.lock().unwrap();
        struct SlowVerify {
            verifies: std::cell::Cell<usize>,
            singles: std::cell::Cell<usize>,
        }
        impl Model for SlowVerify {
            fn n_layers(&self) -> usize { 1 }
            fn hidden_dim(&self) -> usize { 1 }
            fn prefill(&self, _: &[u32], _: usize) {}
            fn forward_id(&self, _: u32, _: usize) -> u32 {
                self.singles.set(self.singles.get() + 1);
                1
            }
            fn forward_batch_ids(&self, t: &[u32], _: usize) -> Option<Vec<u32>> {
                self.verifies.set(self.verifies.get() + 1);
                std::thread::sleep(std::time::Duration::from_millis(5));
                Some(vec![1; t.len()])   // every draft accepted, and still not worth it
            }
        }
        let core = EngineCore::new(SlowVerify {
            verifies: std::cell::Cell::new(0),
            singles: std::cell::Cell::new(0),
        });
        assert_eq!(core.generate(&[1; 8], 64, true), vec![1; 64]);
        assert_eq!(core.model.verifies.get(), 4, "a losing lookup drafter must stop after the initial probes");
        assert!(core.model.singles.get() >= 40, "ordinary decode must take over: {}", core.model.singles.get());
    }

    /// A verify that is genuinely cheap per emitted token must keep running, and
    /// must still take its one calibration sample.
    #[test]
    fn fast_lookup_verify_keeps_drafting() {
        let _guard = GENERATION_TEST.lock().unwrap();
        struct FastVerify {
            verifies: std::cell::Cell<usize>,
            singles: std::cell::Cell<usize>,
        }
        impl Model for FastVerify {
            fn n_layers(&self) -> usize { 1 }
            fn hidden_dim(&self) -> usize { 1 }
            fn prefill(&self, _: &[u32], _: usize) {}
            fn forward_id(&self, _: u32, _: usize) -> u32 {
                self.singles.set(self.singles.get() + 1);
                std::thread::sleep(std::time::Duration::from_millis(4));
                1
            }
            fn forward_batch_ids(&self, t: &[u32], _: usize) -> Option<Vec<u32>> {
                self.verifies.set(self.verifies.get() + 1);
                std::thread::sleep(std::time::Duration::from_millis(5));
                Some(vec![1; t.len()])   // 4 tokens for ~1.25 singles: clearly worth it
            }
        }
        let core = EngineCore::new(FastVerify {
            verifies: std::cell::Cell::new(0),
            singles: std::cell::Cell::new(0),
        });
        assert_eq!(core.generate(&[1; 8], 64, true), vec![1; 64]);
        assert!(core.model.verifies.get() > 8, "a winning lookup drafter must keep drafting: {}", core.model.verifies.get());
        assert!(core.model.singles.get() >= 1, "the gate must still take a scalar calibration sample");
    }

    #[test]
    fn mtp_startup_cost_does_not_disable_fast_steady_state() {
        let _guard = GENERATION_TEST.lock().unwrap();
        struct ColdDraft { drafts: std::cell::Cell<usize>, singles: std::cell::Cell<usize> }
        impl Model for ColdDraft {
            fn n_layers(&self) -> usize { 1 }
            fn hidden_dim(&self) -> usize { 1 }
            fn has_mtp(&self) -> bool { true }
            fn prefill(&self, _: &[u32], _: usize) {}
            fn forward_id(&self, _: u32, _: usize) -> u32 {
                self.singles.set(self.singles.get()+1);
                std::thread::sleep(std::time::Duration::from_millis(2));
                1
            }
            fn mtp_step_committed(&self, _: u32, _: usize) -> Option<Vec<u32>> {
                if self.drafts.get()==0 { std::thread::sleep(std::time::Duration::from_millis(200)); }
                self.drafts.set(self.drafts.get()+1);
                Some(vec![1,1])
            }
        }
        let core=EngineCore::new(ColdDraft{drafts:std::cell::Cell::new(0),singles:std::cell::Cell::new(0)});
        assert_eq!(core.generate(&[1],32,true),vec![1;32]);
        assert!(core.model.drafts.get()>4,"fast MTP must continue beyond startup probes");
        assert_eq!(core.model.singles.get(),1,"scalar calibration must still run");
    }
    #[test]
    #[should_panic(expected="prompt exceeds allocated model context")]
    fn oversized_prompt_is_rejected_before_dispatch() {
        let core=EngineCore::new(Bounded::default());
        core.generate_with(&[1;9],1,None,&mut |_,_| {},&mut |_| true);
    }
    #[test]
    fn empty_request_does_not_allocate_requested_output_capacity() {
        let core=EngineCore::new(Bounded::default());
        assert!(core.generate_with(&[],usize::MAX,None,&mut |_,_| {},&mut |_| true).is_empty());
        assert!(core.model.calls.borrow().is_empty());
    }
}
