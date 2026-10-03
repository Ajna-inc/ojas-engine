#![allow(clippy::too_many_arguments)]
use super::*;
use super::prefix_cache::{self, Plan, BLOCK};
use super::prefix_disk;
 // re-export

/// Cross-turn dense KV-prefix reuse gate: a shared prefix shorter than this
/// isn't worth the bookkeeping (the skip is tiny). Matches the SSM path's floor.
const REUSE_MIN_LCP: usize = 256;
/// Template-divergence slack. When a turn is re-rendered the transcript inserts role
/// headers / `<|im_end|>` before content resumes, so the reusable prefix ends a little
/// before the raw tokens diverge. Re-prefilling this many tail tokens keeps a stray
/// close/header token from reusing a KV row that should have moved.
const REUSE_SLACK: usize = 16;

thread_local! {
    /// High-water mark (exclusive) of cache positions written by the prefill path for
    /// the resident sequence. Batched-prefill archs write KV with a different kernel
    /// than per-token decode, so a reused row is bit-identical to a fresh prefill only
    /// if prefill wrote it; this caps batched-arch reuse to the prefilled prefix,
    /// excluding the decoded tail. One decoder is resident per engine thread, and a
    /// fresh sequence (base_pos==0 prefill) resets it, so a model swap cannot leak a
    /// stale mark. It is shared by all slots, which only recurrent models have, and
    /// those never consult it.
    static PREFILLED_HI: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl<'a> DecoderGpu<'a> {
    /// Cross-turn KV-prefix reuse decision. Returns how many leading tokens of
    /// `full` the sequence already holds, restored from the prompt-prefix cache or
    /// left by the previous turn; the caller prefills only `[start, len)`. With the
    /// document cache, part of that may be approximate (`place_docs`). Leaves `session_tokens` holding exactly the
    /// reused prefix so the subsequent prefill/decode appends extend it cleanly.
    ///
    /// Recurrent decoders restore their snapshot and draft carry here, before the
    /// caller divides the remaining suffix into progress chunks.
    pub(crate) fn dense_reuse_start(&self, full: &[u32]) -> usize {
        // EngineCore passes the full prefix here before splitting progress into
        // 256-token chunks. Decide recurrent reuse now; a chunk alone is too short
        // to discover the useful shared prefix and would reset the state.
        self.sess.cache_last.set(ojas_core::PrefixRestore::default());
        if self.arch.ssm.is_some() && full.len() > 1 {
            if self.uses_prefix_cache() { return self.cache_resume(&full[..full.len() - 1]); }
            let n = self.reuse_prefix(&full[..full.len()-1]);
            if n > 0 { self.seq().session_tokens.borrow_mut().truncate(n); }
            return n;
        }
        self.seq().cache_marks.take();
        self.seq().cache_docs.take();
        if self.arch.ssm.is_some() { return 0; }
        // gpt-oss prefills batched but decodes per-token (two kernels), and is not
        // `batched_dense_ok()`, so it would take the no-cap branch below and could reuse
        // a decode-written row. It also never records a prefill high-water mark, so the
        // cap could not protect it — opt it out of reuse entirely.
        if self.arch.gpt_oss { return 0; }
        if self.cfg.no_prefix_reuse {
            // Kill-switch / A-B control: force a full re-prefill from 0. Clear the
            // token log so the fresh sequence tracks cleanly from position 0.
            self.seq().session_tokens.borrow_mut().clear();
            PREFILLED_HI.with(|c| c.set(0));
            return 0;
        }
        let raw_start = {
            let prev = self.seq().session_tokens.borrow();
            let lcp = prev.iter().zip(full.iter()).take_while(|(a, b)| a == b).count();
            if lcp < REUSE_MIN_LCP { 0 } else { lcp - REUSE_SLACK }
        };
        // Batched-prefill archs: cap at the prefilled high-water mark so a KV row the
        // decode kernel wrote is never reused (its low bits can differ from the batched
        // prefill a no-reuse turn would run, which breaks token equivalence).
        // Per-token-prefill archs use forward_id for both prefill and decode, so every
        // row is reproducible and no cap is needed.
        let start = if self.batched_dense_ok() {
            raw_start.min(PREFILLED_HI.with(|c| c.get()))
        } else {
            raw_start
        };
        self.seq().session_tokens.borrow_mut().truncate(start);
        // Lower the high-water mark to the reused extent. Without this, a shorter or
        // diverged re-prefix leaves the mark stale-high from an earlier, longer
        // sequence, and the cap above then fails to exclude this sequence's
        // decode-written rows on the next turn — a byte-identity break on batched archs
        // (the multi-conversation, shared-system-prompt case). `record_prefilled`
        // re-raises it to this turn's true prefill end.
        PREFILLED_HI.with(|c| c.set(c.get().min(start)));
        start
    }

    /// Record a dense prefill span for cross-turn reuse: append its tokens to the
    /// session log (when it contiguously extends what the cache holds) and advance
    /// the prefilled high-water mark. Called by the batched dense prefill path;
    /// the per-token path records via `forward_id`'s own append.
    fn record_prefilled(&self, base_pos: usize, toks: &[u32]) {
        {
            let mut st = self.seq().session_tokens.borrow_mut();
            if base_pos == 0 {
                st.clear();
                st.extend_from_slice(toks);
            } else if base_pos == st.len() {
                st.extend_from_slice(toks);
            }
            // else: non-contiguous span (not produced by the normal flow) — leave
            // the log alone rather than record a sequence the cache doesn't hold.
        }
        let hi = base_pos + toks.len();
        PREFILLED_HI.with(|c| {
            // Track this sequence's contiguous batched-prefill end, not a monotonic max
            // across turns: chunks arrive contiguous and increasing from the single
            // caller (`generate_with`), so `set(hi)` grows within a turn and also drops
            // a stale-high mark left by a longer earlier sequence (see
            // `dense_reuse_start`). A `max` here would keep the stale mark.
            if base_pos == 0 || base_pos <= c.get() { c.set(hi); }
        });
    }

    /// Drop all cross-turn reuse bookkeeping for the dense path: a fresh
    /// conversation must not reuse the previous one's KV. (SSM state is zeroed by
    /// `reset_state`; this clears the dense token log + high-water mark.)
    pub(crate) fn reset_dense_reuse(&self) {
        self.seq().session_tokens.borrow_mut().clear();
        PREFILLED_HI.with(|c| c.set(0));
    }

    /// Number of leading tokens restored by the most recent recurrent prefill.
    pub fn last_prefill_reused(&self) -> usize { self.seq().last_prefill_reused.get() }

    pub fn prefill(&self, tokens: &[u32], base_pos: usize) {
        assert!(base_pos.checked_add(tokens.len()).is_some_and(|n| n <= self.st.max_seq)
            && tokens.iter().all(|&t| (t as usize) < self.arch.vocab), "prefill exceeds model bounds");
        // The prefix-cache path sets this in `cache_resume`, before the prefills
        // that continue the same prompt.
        if !self.uses_prefix_cache() { self.seq().last_prefill_reused.set(0); }
        if base_pos == 0 && self.arch.ssm.is_some() && !self.uses_prefix_cache()
            && (tokens.len() <= 1 || (self.arch.qwen4exp.is_none() && (!self.wt.q4 || self.cfg.no_prefill))) {
            // These paths do not restore a recurrent snapshot. A new prompt
            // must start from zero rather than inherit the previous request.
            // The prefix-cache path resets in `cache_resume`, after its lookup.
            self.reset_session();
        }
        // gpt-oss: batched Q8 chunked prefill (MoE + sinks + SwiGLU-OAI) — no SSM,
        // so it can't use forward_chunk; forward_batch_impl populates the KV cache.
        if self.arch.gpt_oss && !self.cfg.no_prefill {
            // Batched gpt-oss prefill: dense projections use the p[32] qkv_mg_q8 /
            // gemv_mg_q8_bias kernels (M ≤ 32) and attention_m_sink caps seq at 4096.
            // Bigger M = fewer MoE-expert-streaming passes, so push to 32.
            let chunk_sz = self.cfg.prefill_m.clamp(1, 32);
            let mut pos = base_pos;
            for chunk in tokens.chunks(chunk_sz) {
                self.forward_batch_impl(chunk, pos, LogitsOut::None, false);
                pos += chunk.len();
            }
            return;
        }
        // Dense archs: batched prefill through forward_batch_impl. Without it, prompt
        // processing runs one token at a time and is bandwidth-bound like decode —
        // measured 90 tok/s against the reference's 1018 on the same model. Batching
        // reads each weight once for M tokens instead of M times.
        //
        // Requires the Q8 maps the M-row kernels index (embed_m_q8, qkv_m_q8,
        // ffn_gu_m_q8, gemv_m_q8); a model whose FFN landed in another format falls
        // through to the per-token path below.
        if self.batched_dense_ok() && !self.cfg.no_prefill && tokens.len() > 1 {
            // Chunk as large as the caller asks, up to MAXM. Nothing here requires 32:
            // `gemm_mm_q4l`/`gemm8` tile over M and zero-pad the tail, the Q4L M-row
            // fallback strides in groups of 8, and the elementwise kernels (rmsnorm_m,
            // rope_qk_store_m, silu_mul, add_rowbias_m) index a thread per (token,
            // feature). The buffers are already sized for MAXM.
            //
            // Per-chunk cost is dominated by a fixed term — reading every weight once —
            // so time is roughly (n_chunks * fixed) + (tokens * marginal). Measured at
            // 512 tokens the fixed term is ~58 ms against a ~0.7 ms/token marginal, so
            // throughput scales almost linearly with the chunk size (8 -> 123 tok/s,
            // 16 -> 226, 32 -> 394). Fewer, bigger chunks pay the fixed cost fewer times.
            let chunk_sz = self.cfg.prefill_m.clamp(1, MAXM);
            let mut pos = base_pos;
            for chunk in tokens.chunks(chunk_sz) {
                self.forward_batch_impl(chunk, pos, LogitsOut::None, false);
                pos += chunk.len();
            }
            // Track the prefilled span so the next turn's dense_reuse_start can skip
            // this shared prefix (positional KV survives — reset_state zeroes only SSM).
            // The per-token path below records via forward_id instead.
            self.record_prefilled(base_pos, tokens);
            return;
        }
        // qwen4exp: hyper-connections, PLE and gated DeltaNet, none of which the
        // qwen35 chunk graph encodes — it has its own batched graph. Streamed models
        // cap the chunk at `--ubatch-size`: M tokens route up to M*n_used distinct
        // experts and the packed gather scratch is sized for that union.
        if self.arch.qwen4exp.is_some() && !self.cfg.no_prefill && tokens.len() > 1 {
            let reused = if base_pos == 0 && !self.cfg.no_prefix_reuse {
                // Recompute at least the final token to refresh the MTP hidden
                // row as well as its KV cache after restoring the target state.
                self.reuse_prefix(&tokens[..tokens.len()-1])
            } else { 0 };
            if base_pos == 0 && reused == 0 {
                self.reset_state();
                self.seq().snap_pos.borrow_mut().clear();
                self.seq().snap_buf.borrow_mut().clear();
            }
            self.seq().last_prefill_reused.set(reused);
            let cap = if self.strm.stream { self.strm.ubatch } else { MAXM };
            let chunk_sz = self.cfg.prefill_m.clamp(1, cap);
            // Before the loop, not after: the PLE layer hashes an n-gram of the token
            // history out of `session_tokens`, so a chunk whose predecessors are not
            // in the log gathers the wrong embedding rows.
            self.record_prefilled(base_pos, tokens);
            let mut pos = base_pos + reused;
            for chunk in tokens[reused..].chunks(chunk_sz) {
                self.forward_chunk_qwen4exp(chunk, pos, false);
                // Keep the NextN block's own KV cache alongside the target's; without
                // it the draft attends over an empty cache and echoes its input.
                self.qwen4exp_mtp_catchup(chunk, pos);
                // The first draft conditions on the last prefilled position's hidden.
                self.sp.hrow.set(chunk.len() - 1);
                pos += chunk.len();
                if !self.cfg.no_prefix_reuse { self.maybe_snapshot(pos); }
            }
            return;
        }
        // `wt.q4` is true only at prec 2, so gating the chunk graph on it alone sent
        // every other precision to N sequential forward_id calls — scalar decode in
        // place of prefill (measured 165 tok/s against 4664 for the chunk graph on a
        // 4000-token prompt). The chunk graph's `gemm` closure covers Native/F32/F16/Q4,
        // so the only real dependency is the MoE expert path at graph_chunk.rs:314/327,
        // which indexes wt.w4 unconditionally.
        if self.arch.ssm.is_none() || (self.arch.moe.is_some() && !self.wt.q4) || self.cfg.no_prefill {
            for (i, &t) in tokens.iter().enumerate() { self.forward_id(t, base_pos + i); }
            return;
        }
        let full = tokens;
        let input_base = base_pos;
        let (mut tokens, mut base_pos) = (tokens, base_pos);
        let cached = self.uses_prefix_cache();
        if cached {
            if base_pos == 0 {
                let n = self.cache_start(full);
                tokens = &full[n..];
                base_pos = n;
            } else {
                self.cache_continue(base_pos, tokens);
            }
        } else if base_pos == 0 {
            // In-memory prefix reuse: if the new prompt shares a leading run with
            // the sequence already in the KV cache, roll the SSM state back to the
            // nearest snapshot at/before the divergence and prefill only the tail.
            let reusable = &full[..full.len().saturating_sub(1)];
            let reused = self.reuse_prefix(reusable);
            if reused > 0 {
                tracing::debug!(target: "prefix", "reused {reused} of {} tokens — prefilling {} new",
                    full.len(), full.len() - reused);
                tokens = &full[reused..];
                base_pos = reused;
            } else {
                // no in-memory hit → try the on-disk session cache (cross-restart)
                let n = self.try_restore_session(reusable);
                if n == 0 {
                    self.reset_state();
                    self.seq().snap_pos.borrow_mut().clear();
                    self.seq().snap_buf.borrow_mut().clear();
                }
                if n > 0 {
                    tracing::info!(target: "session", "restored {n} tokens — prefilling {} new", full.len() - n);
                    tokens = &full[n..];
                    base_pos = n;
                    // seed the in-memory snapshot ladder from the restored prefix
                    self.seq().snap_pos.borrow_mut().clear();
                    self.seq().snap_buf.borrow_mut().clear();
                    self.capture_snapshot(n);
                }
            }
        }
        let mut pos = base_pos;
        if !cached && pos == 0 && !self.seq().snap_pos.borrow().contains(&0) {
            self.seq().snap_pos.borrow_mut().clear();
            self.seq().snap_buf.borrow_mut().clear();
            self.capture_snapshot(0); // state at position 0 is the zeroed init
        }
        let chunk_sz: usize = self.cfg.prefill_m.min(MAXM);
        for chunk in tokens.chunks(chunk_sz) {
            self.forward_chunk(chunk, pos, false);
            if ojas_core::device_fault::is_faulted() {
                // A failed chunk leaves the cache and recurrent state undefined: stop
                // here, and record nothing a later request could reuse or restore.
                self.mark_span_unreusable();
                return;
            }
            self.qwen35_mtp_catchup(chunk, pos);
            self.sp.hrow.set(chunk.len()-1);
            pos += chunk.len();
            if cached {
                self.cache_capture(pos);
                self.doc_capture(pos);
            } else {
                self.maybe_snapshot(pos);
            }
        }
        if cached { self.sess.cache.borrow_mut().sync(); }
        // record the exact token sequence now represented by the KV cache + state
        self.record_prefilled(input_base, full);
        if !tokens.is_empty() && !cached {
            self.save_session(pos);
        }
    }

    /// Make the resident sequence unmatchable by both prefix-reuse paths.
    ///
    /// `dense_reuse_start` and `reuse_prefix` both decide by a longest-common-prefix
    /// over the raw token ids in `session_tokens`. An injected span's ids are
    /// placeholders — the same `<|image_pad|>` repeated for every image — so two
    /// different pages produce byte-identical id sequences, and the LCP (floor 256,
    /// against a page of thousands of rows) would match one against the other and serve
    /// page A's KV for page B. The failure is silent and looks like the model
    /// hallucinating a previous document.
    ///
    /// The ids carry nothing that could distinguish them, so the span stops claiming to
    /// be reusable at all: an empty log has LCP 0 with every prompt, and the dropped
    /// high-water mark keeps the dense cap from vouching for rows behind it.
    /// `save_session` and `try_restore_session` fail closed on the same emptiness
    /// (`toks.len() != n` and `n < 64`), so the on-disk cache cannot resurrect it
    /// across a restart either.
    fn mark_span_unreusable(&self) {
        self.reset_dense_reuse();     // session_tokens.clear() + PREFILLED_HI = 0
        self.seq().last_prefill_reused.set(0);
        // Nor may the prefix cache store blocks under these ids.
        *self.seq().cache_plan.borrow_mut() = None;
    }

    /// Prefill a span from precomputed residual rows instead of token ids — the seam a
    /// vision encoder's output enters the decoder through. See
    /// [`ojas_core::Model::prefill_embeds`] for the contract; this is its
    /// implementation. Returns false when the architecture has no injection path.
    ///
    /// `x` is `tokens.len() * self.d` f32, row-major: row i is position `base_pos + i`'s
    /// residual stream, used verbatim. `arch.embed_scale` (Gemma's sqrt(d)) is not
    /// applied — a caller producing rows for such a model applies it itself, since the
    /// rows need not be embeddings at all.
    ///
    /// Two branches, matching what `prefill` would do with the same ids:
    ///
    /// * chunked (qwen35 / surya) — `MAXM`-sized chunks memcpy'd into `st.x` and run
    ///   through the existing `forward_chunk_enc_embed` with `do_embed = false`. Same
    ///   graph, kernels and chunk size as a token prefill; only the gather is gone.
    /// * per-token fallback — one full-stack forward per row with `do_embed = false`
    ///   (the swarm middle-stage primitive), for every arch whose `prefill` also falls
    ///   through to the per-token loop, and for dense archs whose batched prefill kernel
    ///   embeds unconditionally. Correct, but one command buffer per row: a 4096-row
    ///   page is 4096 submits, so a dense vision path wants `batch.rs` taught the flag
    ///   first.
    ///
    /// `pos3`, when present, is the per-row sectioned-M-RoPE coordinate `(t,h,w,e)`. It
    /// is sliced per chunk and handed to `rope_qk_store_m`, which applies it to the rope
    /// angle only: the KV cache row stays `base_pos + i`, so an image span occupies
    /// contiguous slots however its coordinates are numbered. Only the 6 attention
    /// layers rope at all on surya-2 (3, 7, 11, 15, 19, 23); the 18 GDN layers are
    /// order-dependent and take no position.
    ///
    /// Unsupported (returns false): disk-streamed MoE (its driver owns the residual
    /// across command buffers), qwen4exp (hyper-connections + PLE mean `x` is not the
    /// whole state, and its batched graph embeds unconditionally), and a `Some(pos3)`
    /// for an architecture that declares sections but whose prefill falls through to
    /// the per-token loop below — that loop runs the scalar `rope_qk_store`, which has
    /// no `mpos` argument, so the coordinates cannot be honoured there. Ignoring them
    /// would return true and read as a vision-encoder quality problem.
    pub fn prefill_embeds(&self, tokens: &[u32], x: &[f32], base_pos: usize,
                          pos3: Option<&[[u32; 4]]>) -> bool {
        if let Some(p3) = pos3 {
            assert_eq!(p3.len(), tokens.len(), "pos3 must carry one (t,h,w,e) per row");
        }
        if tokens.is_empty() { return true; }
        assert_eq!(x.len(), tokens.len() * self.d,
            "prefill_embeds: x must be tokens.len() * hidden_dim f32 row-major");
        assert!(base_pos.checked_add(tokens.len()).is_some_and(|n| n <= self.st.max_seq)
            && tokens.iter().all(|&t| (t as usize) < self.arch.vocab),
            "prefill_embeds exceeds model bounds");
        // Streamed MoE drives the residual through its own per-layer command-buffer
        // chunking; qwen4exp carries hyper-connection lanes and PLE n-gram rows that `x`
        // alone does not describe. Refuse rather than inject half a state.
        //
        // `strm.stream` is set from `prec == 4` unconditionally (load.rs:59), so testing
        // it alone refuses every model at the default precision, including dense ones
        // with no experts to stream. Test the actual capability.
        if (self.strm.stream && self.arch.moe.is_some()) || self.arch.qwen4exp.is_some() { return false; }

        // The qwen35 chunk graph, reached on exactly the condition `prefill` uses
        // to reach it (gpt-oss / batched-dense / qwen4exp were all excluded above
        // or are excluded by `ssm.is_some()`).
        let chunked = self.arch.ssm.is_some() && !(self.arch.moe.is_some() && !self.wt.q4) && !self.cfg.no_prefill;
        // Sectioned rope lives in `rope_qk_store_m`, which only the chunk graph
        // dispatches. Decide before touching any state: a refusal has to leave the
        // session exactly as it found it.
        let sections = self.arch.ssm.map_or([0u32; 4], |c| c.mrope_sections);
        if pos3.is_some() && sections.iter().any(|&s| s != 0) && !chunked { return false; }

        let d = self.d;
        // A fresh sequence must not inherit the previous one's recurrent state — the
        // same reason `prefill` resets at base_pos 0.
        if base_pos == 0 {
            self.reset_session();
        }
        // Before the forwards, not after: the span is unreusable the moment its
        // rows enter the cache, and an early return or a panic mid-span must not
        // leave a reusable-looking log behind.
        self.mark_span_unreusable();

        if chunked {
            let chunk_sz: usize = self.cfg.prefill_m.clamp(1, MAXM);
            let mut pos = base_pos;
            for (ci, chunk) in tokens.chunks(chunk_sz).enumerate() {
                // st.x is StorageModeShared, so this host store is visible to the
                // command buffer the next line encodes (span.rs:17 / trainer.rs:639).
                let src = &x[ci * chunk_sz * d..][..chunk.len() * d];
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), self.st.x.contents() as *mut f32, src.len());
                }
                // Same slice arithmetic as the rows, one coordinate per row instead
                // of d floats. The chunk encoder indexes it by m, not by pos.
                let p3 = pos3.map(|p| &p[ci * chunk_sz..][..chunk.len()]);
                self.forward_chunk_enc_embed(None, chunk, pos, false, false, false, p3);
                // Keep the draft block's KV positionally aligned with the target's. It
                // conditions on the placeholder ids for these rows, so its drafts over
                // an injected span are worthless — but they are verified, so that costs
                // speed, not correctness.
                self.qwen35_mtp_catchup(chunk, pos);
                self.sp.hrow.set(chunk.len() - 1);
                pos += chunk.len();
            }
            // Deliberately no maybe_snapshot / save_session: a snapshot is only ever
            // reached through an id LCP, and this span has disowned its ids.
            self.mark_span_unreusable();
            return true;
        }

        // Per-token fallback: write row i, then run the whole stack with do_embed
        // false. `forward_span` is the swarm's middle-stage entry and already means
        // this: st.x already holds the incoming activation.
        let nl = self.arch.n_layers;
        let last = tokens.len() - 1;
        for (i, &t) in tokens.iter().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(x[i * d..].as_ptr(), self.st.x.contents() as *mut f32, d);
            }
            // do_head only on the final row: the head does not touch the KV cache or
            // the recurrent state, so running it per row would buy nothing but an
            // lm_head per image patch. The last row's logits stay available for a
            // caller that wants to sample straight off the span.
            let _ = self.forward_span(t, base_pos + i, 0, nl, false, i == last);
        }
        self.mark_span_unreusable();  // forward_span does not append, but be explicit
        true
    }

    /// Whether prefill goes through the prompt-prefix cache: the qwen35 chunk graph,
    /// whose chunks tile a cache block exactly, with a cache budget set.
    pub(crate) fn uses_prefix_cache(&self) -> bool {
        self.arch.ssm.is_some() && self.arch.qwen4exp.is_none()
            && !(self.arch.moe.is_some() && !self.wt.q4) && !self.cfg.no_prefill && !self.cfg.no_prefix_reuse
            && BLOCK.is_multiple_of(self.cfg.prefill_m.clamp(1, MAXM))
            && self.sess.cache.borrow().enabled()
    }

    /// The current slot's KV rows for positions `[p0, p0 + BLOCK)` of every
    /// attention layer, the draft block's included: K then V per layer.
    fn kv_block_parts(&self, p0: usize) -> Vec<(*mut u8, usize)> { self.kv_rows_parts(p0, BLOCK) }

    /// Every layer with an attention KV cache, the draft block's included.
    fn kv_layers(&self) -> Vec<usize> {
        let draft = self.sp.mtp.map(|m| m.layer);
        (0..self.arch.n_layers).filter(|&l| !self.arch.layers[l].is_ssm).chain(draft).collect()
    }

    /// The current slot's KV rows for positions `[p0, p0 + n)`: K then V for each of
    /// `kv_layers`.
    fn kv_rows_parts(&self, p0: usize, n: usize) -> Vec<(*mut u8, usize)> {
        let mut parts = Vec::new();
        for l in self.kv_layers() {
            let row = self.arch.layers[l].kvdim as usize * 2;
            for cache in [&self.st.kcache[l], &self.st.vcache[l]] {
                parts.push((unsafe { self.kv_ptr(cache, l).add(p0 * row) }, n * row));
            }
        }
        parts
    }

    /// Start a new sequence from the longest cached prefix of `prompt` (every token
    /// before the one whose logits are wanted) and plan what this prefill adds to
    /// the cache. With reuse forbidden nothing is restored, but the lookup still
    /// tells the plan where the prompt branches. Returns the position processing
    /// resumes at: the positions restored, a multiple of `BLOCK`, or the end of the
    /// last document `place_docs` served.
    pub(crate) fn cache_resume(&self, prompt: &[u32]) -> usize {
        let t0 = std::time::Instant::now();
        let reads = self.sess.cache.borrow().stats().disk_reads;
        let keys = prefix_cache::block_keys(prompt);
        let reuse = self.seq().cache_reuse.get();
        // A block that fails to load from the cache directory is dropped, so the
        // next lookup stops short of it.
        let (hit, used, (kv, snapshot)) = loop {
            let mut cache = self.sess.cache.borrow_mut();
            let mut hit = cache.lookup(prompt, &keys, true);
            let used = if reuse { hit.blocks } else { 0 };
            hit.resume = hit.resume.min(used);
            if let Some(payloads) = cache.load(&keys[..hit.resume]) { break (hit, used, payloads); }
        };
        self.reset_session();
        let n = hit.resume * BLOCK;
        self.sess.cache.borrow_mut().record(&keys[..used], hit.resume);
        if n > 0 {
            for (b, bytes) in kv.iter().enumerate() {
                write_parts(&self.kv_block_parts(b * BLOCK), bytes);
            }
            self.set_state_bytes(&snapshot.expect("a resume block carries a snapshot"));
            self.rebuild_page_meta(n);
            self.seq().session_tokens.borrow_mut().extend_from_slice(&prompt[..n]);
        }
        let disk_payloads = self.sess.cache.borrow().stats().disk_reads - reads;
        let restore_us = t0.elapsed().as_micros() as u64;
        self.seq().last_prefill_reused.set(n);
        let mut plan = Plan::new(prompt.to_vec(), keys, hit, &self.seq().cache_marks.take());
        // KV restored from a Q8_0 directory is close to, not identical with, what
        // processing computes, so nothing built on it is cached as exact.
        if disk_payloads > 0 && !self.sess.cache.borrow().disk_kv_exact() { plan.exact = plan.exact.min(n); }
        *self.seq().cache_plan.borrow_mut() = Some(plan);
        let (resumed, doc_reused_tokens) = self.place_docs(prompt, n);
        self.sess.cache_last.set(ojas_core::PrefixRestore {
            matched_tokens: hit.blocks * BLOCK, reused_tokens: n, doc_reused_tokens, disk_payloads, restore_us,
        });
        resumed
    }

    /// Serve the requested document spans of `prompt` from the document cache, past
    /// the `n` tokens restored exactly. A span found there is placed at its new
    /// position: the text before it and its first `doc_recompute` tokens are
    /// processed, its middle comes from the cache with keys rotated, and its last
    /// `doc_tail` share is processed again to rebuild the recurrent state. Spans not
    /// found, or not to be served, are added once processed. Returns the position
    /// processing resumes at, and the tokens served from the cache.
    fn place_docs(&self, prompt: &[u32], n: usize) -> (usize, usize) {
        let mut spans: Vec<(usize, usize)> = self.seq().cache_docs.take().into_iter()
            .map(|(s, e)| (s, e.min(prompt.len()))).filter(|&(s, e)| s < e).collect();
        spans.sort_unstable();
        self.seq().docs_pending.borrow_mut().clear();
        if !self.sess.docs.borrow().enabled() { return (n, 0); }
        let serve = self.seq().docs_serve.get();
        let (mut pos, mut reused) = (n, 0);
        for (s, e) in spans {
            if s < pos {
                // Already restored exactly: cache it from the rows now in place.
                if e <= n && !self.sess.docs.borrow().contains(&prompt[s..e]) {
                    self.seq().docs_pending.borrow_mut().push((s, e));
                }
                continue;
            }
            let len = e - s;
            let recompute = self.cfg.doc_recompute.min(len);
            let tail = ((len as f64 * self.cfg.doc_tail).ceil() as usize).max(1);
            let found = if serve { self.sess.docs.borrow_mut().get(&prompt[s..e]) } else { None };
            let Some((kv, base)) = found.filter(|_| recompute + tail < len) else {
                self.seq().docs_pending.borrow_mut().push((s, e));
                continue;
            };
            let (head, body_end) = (s + recompute, e - tail);
            if head > pos { self.prefill(&prompt[pos..head], pos); }
            self.place_rows(&kv, len, recompute, len - tail, s as i64 - base as i64, head);
            self.seq().session_tokens.borrow_mut().extend_from_slice(&prompt[head..body_end]);
            if let Some(plan) = self.seq().cache_plan.borrow_mut().as_mut() { plan.exact = plan.exact.min(head); }
            self.prefill(&prompt[body_end..e], body_end);
            reused += body_end - head;
            pos = e;
        }
        if reused > 0 { self.sess.docs.borrow_mut().record(reused); }
        self.doc_capture(n);
        (pos, reused)
    }

    /// Write rows `[from, to)` of a cached span of `len` tokens to positions starting
    /// at `at`, rotating keys by `delta` positions.
    fn place_rows(&self, kv: &[u8], len: usize, from: usize, to: usize, delta: i64, at: usize) {
        let rd = self.arch.ssm.as_ref().map_or(self.arch.hd, |s| s.n_rot as usize);
        let mut off = 0;
        for l in self.kv_layers() {
            let lp = self.arch.layers[l];
            let row = lp.kvdim as usize * 2;
            let rope = super::doc_cache::Rope { hd: lp.head_dim as usize, rd, base: lp.rope_base };
            let parts = [(&self.st.kcache[l], true), (&self.st.vcache[l], false)];
            for (cache, is_key) in parts {
                let mut rows = kv[off + from * row..off + to * row].to_vec();
                if is_key { super::doc_cache::rotate_keys(&mut rows, lp.kvdim as usize, rope, delta); }
                write_parts(&[(unsafe { self.kv_ptr(cache, l).add(at * row) }, rows.len())], &rows);
                off += len * row;
            }
        }
        self.rebuild_page_meta(at + (to - from));
    }

    /// After processing reached `pos`: add the pending document spans that end by
    /// then, while the sequence is still exact, to the document cache.
    fn doc_capture(&self, pos: usize) {
        let ready: Vec<(usize, usize)> = self.seq().docs_pending.borrow().iter().copied().filter(|&(_, e)| e <= pos).collect();
        if ready.is_empty() { return; }
        self.seq().docs_pending.borrow_mut().retain(|&(_, e)| e > pos);
        let plan = self.seq().cache_plan.borrow();
        let Some(plan) = plan.as_ref() else { return };
        for (s, e) in ready.into_iter().filter(|&(_, e)| e <= plan.exact) {
            let kv = read_parts(&self.kv_rows_parts(s, e - s));
            self.sess.docs.borrow_mut().insert(&plan.tokens[s..e], s, kv);
        }
    }

    /// Where a prefill at position 0 starts. A plan `reuse_prefix_len` made for this
    /// prompt and nothing has started yet already ran the lookup and reset the
    /// state; anything else looks the prompt up now.
    fn cache_start(&self, tokens: &[u32]) -> usize {
        let fresh = self.seq().cache_plan.borrow_mut().as_mut().is_some_and(|p| {
            let k = p.tokens.len().min(tokens.len());
            let ok = !p.started && p.resume == 0 && p.tokens[..k] == tokens[..k];
            p.started |= ok;
            ok
        });
        if fresh { return 0; }
        let n = self.cache_resume(&tokens[..tokens.len().saturating_sub(1)]);
        if let Some(p) = self.seq().cache_plan.borrow_mut().as_mut() { p.started = true; }
        n
    }

    /// A prefill continuing at `base`: keep the plan only if it describes these tokens.
    fn cache_continue(&self, base: usize, tokens: &[u32]) {
        let mut plan = self.seq().cache_plan.borrow_mut();
        let keep = plan.as_ref().is_some_and(|p| {
            let end = (base + tokens.len()).min(p.tokens.len());
            base <= end && p.tokens[base..end] == tokens[..end - base]
        });
        match plan.as_mut() {
            Some(p) if keep => {
                p.started = true;
                // Chunks that start off the chunk grid run through different kernel
                // shapes than a fresh prefill, so nothing after them is exact.
                if !base.is_multiple_of(self.cfg.prefill_m.clamp(1, MAXM)) { p.exact = p.exact.min(base); }
            }
            _ => *plan = None,
        }
    }

    /// After the prefill chunk that ended at `pos`: cache the block that ended
    /// there, with a snapshot where the plan wants one.
    fn cache_capture(&self, pos: usize) {
        let mut plan = self.seq().cache_plan.borrow_mut();
        let Some(plan) = plan.as_mut() else { return };
        let b = pos / BLOCK;
        if !pos.is_multiple_of(BLOCK) || b <= plan.resume || b > plan.keys.len() || pos > plan.exact { return; }
        let key = plan.keys[b - 1];
        let parent = if b > 1 { plan.keys[b - 2] } else { prefix_cache::ROOT };
        let mut cache = self.sess.cache.borrow_mut();
        let tokens = &plan.tokens[pos - BLOCK..pos];
        // A key held by another block (a hash collision) is left alone: neither its
        // KV nor its snapshot may come from this prompt.
        if !cache.matches(key, parent, tokens) {
            if cache.contains(key) { return; }
            let kv = read_parts(&self.kv_block_parts(pos - BLOCK));
            if !cache.insert(parent, tokens, kv) { return; }
        }
        if plan.wants_snapshot(b, self.snap_interval()) && !cache.has_snapshot(key)
            && cache.attach_snapshot(key, self.state_bytes()) {
            plan.took_snapshot(b);
        }
    }

    /// Occupancy and hit counts of the prompt-prefix cache.
    pub(crate) fn prefix_cache_stats(&self) -> ojas_core::PrefixCacheStats {
        let s = self.sess.cache.borrow().stats();
        ojas_core::PrefixCacheStats {
            blocks: s.blocks, snapshots: s.snapshots, bytes: s.bytes, budget: s.budget,
            lookups: s.lookups, hits: s.hits, reused_tokens: s.reused_tokens, block_tokens: BLOCK,
            directory: s.disk_budget > 0 || s.disk_blocks > 0, disk_blocks: s.disk_blocks,
            disk_snapshots: s.disk_snapshots, disk_bytes: s.disk_bytes, disk_budget: s.disk_budget,
            disk_reads: s.disk_reads, pinned_blocks: s.pinned, evictions: s.evictions, last: self.sess.cache_last.get(),
            docs: { let d = self.sess.docs.borrow().stats(); ojas_core::DocCacheStats {
                docs: d.docs, bytes: d.bytes, budget: d.budget, hits: d.hits, reused_tokens: d.reused_tokens } },
        }
    }

    /// Process `tokens` into the prefix cache without generating: every full block
    /// is cached, with a snapshot after the last, and pinned when `pin` is set.
    /// Returns the tokens now cached.
    pub(crate) fn warm_prefix(&self, tokens: &[u32], pin: bool) -> usize {
        let prompt = &tokens[..tokens.len() / BLOCK * BLOCK];
        self.seq().cache_marks.take();
        self.seq().cache_docs.take();
        if prompt.is_empty() || !self.uses_prefix_cache() { return 0; }
        let reuse = self.seq().cache_reuse.replace(true);
        let start = self.cache_resume(prompt);
        if start < prompt.len() { self.prefill(&prompt[start..], start); }
        self.seq().cache_reuse.set(reuse);
        let keys = prefix_cache::block_keys(prompt);
        let mut cache = self.sess.cache.borrow_mut();
        if pin { cache.pin(&keys); }
        cache.sync();
        cache.lookup(prompt, &keys, true).resume * BLOCK
    }

    /// Everything besides the weights, precision and state layout that decides the
    /// numbers a prefill computes: the engine version, every kernel's source, the
    /// GPU, and the engine settings other than the cache and serving options.
    fn numerics_identity(&self) -> String {
        let mut cfg = self.cfg.clone();
        (cfg.prefix_cache_gb, cfg.prefix_cache_dir, cfg.prefix_cache_readonly) = (None, None, false);
        (cfg.prefix_cache_disk_gb, cfg.prefix_cache_reserve_gb) = (None, 0.0);
        (cfg.prefix_cache_save, cfg.prefix_cache_disk_int8, cfg.prefix_cache_pin) = (Default::default(), false, None);
        (cfg.parallel, cfg.doc_cache_gb, cfg.doc_recompute, cfg.doc_tail) = (None, 0.0, 0, 0.0);
        let mut kernels: Vec<(&str, &str)> = ojas_metal::kernels::families().collect();
        kernels.sort_unstable();
        let source: Vec<u8> = kernels.iter().flat_map(|(name, src)| name.bytes().chain(src.bytes())).collect();
        format!("{} {:016x} {} {cfg:?}", env!("CARGO_PKG_VERSION"), prefix_disk::checksum(&source), self.gpu.device.name())
    }

    /// Back the prefix cache with `prefix_cache_dir`, when set. Blocks saved there by
    /// any earlier run of this model build are reusable at once. The subdirectory is
    /// chosen by the model's weights, `prec` and the state layout, so a different
    /// build never reads them. A directory that cannot be opened leaves the cache in
    /// RAM only.
    pub(crate) fn open_prefix_dir(&self, model_files: &[std::path::PathBuf], prec: u8) {
        let Some(root) = self.cfg.prefix_cache_dir.as_deref() else { return };
        if !self.uses_prefix_cache() {
            tracing::warn!(target: "prefix", "this model and configuration keep no prompt-prefix cache; {} is not used",
                root.display());
            return;
        }
        if model_files.is_empty() {
            tracing::warn!(target: "prefix", "the model was not opened from files; caching prompts in RAM only");
            return;
        }
        let gb = |v: f64| (v.max(0.0) * 1e9) as u64;
        let kv_len: usize = self.kv_block_parts(0).iter().map(|&(_, len)| len).sum();
        let snapshot_len: usize = self.state_parts().iter().map(|&(_, len)| len).sum();
        let kv = if self.cfg.prefix_cache_disk_int8 { prefix_disk::KvFormat::Q8 } else { prefix_disk::KvFormat::F16 };
        let opened = prefix_disk::fingerprint(model_files).and_then(|weights| {
            let build = [weights, u64::from(prec), self.cfg.prefill_m.clamp(1, MAXM) as u64, kv_len as u64,
                snapshot_len as u64, BLOCK as u64, kv as u64];
            let mut salt = build.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
            salt.extend_from_slice(self.numerics_identity().as_bytes());
            let salt = prefix_disk::checksum(&salt);
            let format = if kv == prefix_disk::KvFormat::Q8 { ", int8 KV" } else { "" };
            let about = format!("{} at precision {prec}{format}\n", self.sess.model_name);
            prefix_disk::Disk::open(root, salt, &about, prefix_disk::DiskOptions {
                readonly: self.cfg.prefix_cache_readonly, reserve: gb(self.cfg.prefix_cache_reserve_gb),
                block_tokens: BLOCK, kv,
            })
        });
        let opened = match opened {
            Ok(opened) => opened,
            Err(e) => {
                tracing::warn!(target: "prefix", "cache directory {}: {e}; caching prompts in RAM only", root.display());
                return;
            }
        };
        let budget = match self.cfg.prefix_cache_disk_gb {
            Some(v) => gb(v),
            None => gb(20.0).min(prefix_disk::free_space(opened.disk.dir()) / 4),
        };
        let (dir, blocks, writable) = (opened.disk.dir().display().to_string(), opened.records.len(), opened.disk.writable());
        self.sess.cache.borrow_mut().attach_disk(opened, budget as usize, self.cfg.prefix_cache_save, kv_len, snapshot_len);
        if writable {
            tracing::info!(target: "prefix", "cache directory {dir}: {blocks} blocks, cap {:.1} GB", budget as f64 / 1e9);
        } else {
            tracing::info!(target: "prefix", "cache directory {dir}: {blocks} blocks, read-only");
        }
    }

    /// Snapshot interval in tokens (env OJAS_SNAP; default 2048). Smaller =
    /// less redundant re-prefill on a hit, more memory.
    pub(crate) fn snap_interval(&self) -> usize {
        self.cfg.snap_interval
    }

    /// The current slot's recurrent state as bytes: for each SSM layer conv then
    /// SSM state, then the draft carry when MTP is loaded. Attention layers keep
    /// their state in addressable KV rows instead.
    pub(crate) fn state_bytes(&self) -> Vec<u8> { read_parts(&self.state_parts()) }

    /// Overwrite the current slot's recurrent state with bytes from `state_bytes`.
    pub(crate) fn set_state_bytes(&self, bytes: &[u8]) { write_parts(&self.state_parts(), bytes) }

    fn state_parts(&self) -> Vec<(*mut u8, usize)> {
        let mut parts = Vec::new();
        for l in 0..self.arch.n_layers {
            if self.arch.layers[l].is_ssm {
                parts.extend([self.conv_region(l), self.ssm_region(l)]);
            }
        }
        if let Some(m) = self.sp.mtp {
            let bytes = m.hnorm_len * 4;
            parts.push((unsafe { (self.sp.mtp_hprev.contents() as *mut u8).add(MAXM * bytes) }, bytes));
        }
        parts
    }

    /// Copy the current SSM/conv state into a host snapshot tagged with `pos`.
    pub(crate) fn capture_snapshot(&self, pos: usize) {
        if self.arch.ssm.is_none() { return; }
        let buf = self.state_bytes();
        self.seq().snap_pos.borrow_mut().push(pos);
        self.seq().snap_buf.borrow_mut().push(buf);
        // bound memory: keep position 0 (anchor) + the most recent snapshots
        let cap = 24usize;
        let mut sp = self.seq().snap_pos.borrow_mut();
        let mut sb = self.seq().snap_buf.borrow_mut();
        while sp.len() > cap {
            let drop = if sp[0] == 0 { 1 } else { 0 }; // never drop the pos-0 anchor
            sp.remove(drop);
            sb.remove(drop);
        }
    }

    /// Snapshot at chunk boundaries that cross an interval multiple.
    pub(crate) fn maybe_snapshot(&self, pos: usize) {
        let iv = self.snap_interval();
        let last = self.seq().snap_pos.borrow().last().copied().unwrap_or(0);
        if pos >= last + iv {
            self.capture_snapshot(pos);
        }
    }

    /// Restore the SSM/conv state from snapshot index `idx`.
    pub(crate) fn restore_snapshot(&self, idx: usize) {
        self.set_state_bytes(&self.seq().snap_buf.borrow()[idx]);
    }

    /// If `tokens` shares a leading run with the sequence already in the KV cache,
    /// roll the SSM state back to the nearest snapshot ≤ the divergence point and
    /// return that position (KV for [0..pos] is reused in place). 0 = no reuse.
    pub(crate) fn reuse_prefix(&self, tokens: &[u32]) -> usize {
        if self.arch.ssm.is_none() || self.cfg.no_prefix_reuse { return 0; }
        let prev = self.seq().session_tokens.borrow();
        // longest common prefix with what the cache currently holds
        let lcp = prev.iter().zip(tokens.iter()).take_while(|(a, b)| a == b).count();
        if lcp < 256 { return 0; } // not worth the snapshot restore + bookkeeping
        // largest snapshot position ≤ lcp
        let sp = self.seq().snap_pos.borrow();
        let Some((idx, &pos)) = sp.iter().enumerate().filter(|(_, &p)| p <= lcp).max_by_key(|(_, &p)| p)
        else { return 0 };
        drop(sp);
        // Nothing to restore to: the pos-0 snapshot is the zeroed anchor, so restoring
        // it reuses no tokens (this function returns 0 anyway) while wiping the live
        // recurrent state and popping the whole snapshot ladder. Return early instead,
        // so probing reuse stays non-destructive: `reuse_prefix_len` is documented as a
        // query and `Model::reuse_prefix_len` is public API, and a caller that probes
        // before deciding what to prefill must not lose the state it just built. Every
        // in-tree caller follows a 0 with its own `reset_state`/`try_restore_session`,
        // so the wipe is not load-bearing.
        if pos == 0 { return 0; }
        self.restore_snapshot(idx);
        // drop any snapshots after the reuse point — they belong to a diverged tail
        let mut spm = self.seq().snap_pos.borrow_mut();
        let mut sbm = self.seq().snap_buf.borrow_mut();
        while spm.last().map_or(false, |&p| p > pos) { spm.pop(); sbm.pop(); }
        pos
    }

    /// Recompute page-sparse attention metadata for the first `n` cache rows after
    /// they were restored rather than prefilled. A no-op without a sparse budget.
    fn rebuild_page_meta(&self, n: usize) {
        if self.arch.sparse_budget.is_none() || n == 0 { return; }
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        let npg = n.div_ceil(ojas_metal::kernels::attn::PAGE) as u32;
        for l in 0..self.arch.n_layers {
            let p = self.arch.layers[l];
            if p.is_ssm { continue; }
            self.enc_reduce(enc, "page_minmax",
                &[(&self.st.kcache[l], 0), (&self.st.pmeta[l], 1)],
                &[(2, p.kvdim), (3, 0), (4, n as u32)], &[], npg as u64, 256);
        }
        enc.end_encoding();
        let _ = ojas_metal::commit_and_wait_checked(cb, "prefill");
    }

    /// Sequence-state parts in a fixed layer order (session save/restore contract):
    /// SSM layers contribute (conv_state, ssm_state) in full; attention layers
    /// contribute the first `n` KV rows of (kcache, vcache).
    pub(crate) fn session_parts(&self, n: usize) -> Vec<(*mut u8, usize)> {
        let mut parts = Vec::new();
        for l in 0..self.arch.n_layers {
            let p = self.arch.layers[l];
            if p.is_ssm {
                parts.extend([self.conv_region(l), self.ssm_region(l)]);
            } else {
                let bytes = n * p.kvdim as usize * 2; // f16 rows
                parts.push((self.kv_ptr(&self.st.kcache[l], l), bytes));
                parts.push((self.kv_ptr(&self.st.vcache[l], l), bytes));
            }
        }
        if let Some(mc) = self.sp.mtp {
            let bytes = n * self.arch.layers[mc.layer].kvdim as usize * 2;
            parts.push((self.kv_ptr(&self.st.kcache[mc.layer], mc.layer), bytes));
            parts.push((self.kv_ptr(&self.st.vcache[mc.layer], mc.layer), bytes));
            let row = self.sp.mtp.map(|m| m.hnorm_len * 4).unwrap_or(0);
            parts.push((unsafe { (self.sp.mtp_hprev.contents() as *mut u8).add(MAXM*row) }, row));
        }
        parts
    }

    pub(crate) fn session_path_key(&self) -> Option<(std::path::PathBuf, u64)> {
        let dir = crate::session::dir()?;
        let key = crate::session::model_key(&self.sess.model_name, self.arch.n_layers, self.d);
        Some((dir.join(format!("{}.ajs", self.sess.model_name)), key))
    }

    /// Save the current sequence state (position `n`, tokens recorded by prefill).
    pub(crate) fn save_session(&self, n: usize) {
        let Some((path, key)) = self.session_path_key() else { return };
        let toks = self.seq().session_tokens.borrow();
        if toks.len() != n { return; } // only save states we fully tracked
        let parts = self.session_parts(n);
        let slices: Vec<&[u8]> = parts.iter()
            .map(|&(p, len)| unsafe { std::slice::from_raw_parts(p as *const u8, len) })
            .collect();
        let t0 = std::time::Instant::now();
        match crate::session::save(&path, key, &toks, &slices) {
            Ok(()) => tracing::info!(target: "session", "saved {n} tokens in {:.2}s", t0.elapsed().as_secs_f32()),
            Err(e) => tracing::error!(target: "session", "save failed: {e}"),
        }
    }

    /// If a saved session's tokens are a prefix of `tokens`, restore its state
    /// and return the restored position (0 = no usable session).
    pub(crate) fn try_restore_session(&self, tokens: &[u32]) -> usize {
        self.seq().session_tokens.borrow_mut().clear();
        self.seq().session_tokens.borrow_mut().extend_from_slice(tokens);
        let Some((path, key)) = self.session_path_key() else { return 0 };
        let Some(mut loaded) = crate::session::open(&path, key) else { return 0 };
        let n = loaded.tokens.len();
        if n < 64 || n > tokens.len() || loaded.tokens != tokens[..n] { return 0; }
        let t0 = std::time::Instant::now();
        for (ptr, len) in self.session_parts(n) {
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            if !loaded.next_part(dst) {
                tracing::warn!(target: "session", "stale/mismatched file — ignoring");
                return 0; // states are zero-init; partial copy is harmless pre-prefill
            }
        }
        self.rebuild_page_meta(n);
        tracing::info!(target: "session", "state restored in {:.2}s", t0.elapsed().as_secs_f32());
        n
    }

}

/// Concatenate the bytes behind `parts`.
fn read_parts(parts: &[(*mut u8, usize)]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(parts.iter().map(|&(_, len)| len).sum());
    for &(ptr, len) in parts {
        buf.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr as *const u8, len) });
    }
    buf
}

/// Scatter `bytes` back over `parts`, in order; the inverse of `read_parts`.
fn write_parts(parts: &[(*mut u8, usize)], bytes: &[u8]) {
    let mut off = 0;
    for &(ptr, len) in parts {
        unsafe { std::ptr::copy_nonoverlapping(bytes[off..off + len].as_ptr(), ptr, len); }
        off += len;
    }
}
