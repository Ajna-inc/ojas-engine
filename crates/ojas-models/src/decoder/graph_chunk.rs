#![allow(clippy::too_many_arguments)]
use super::*;
use metal::MTLSize;
use std::ffi::c_void;
 // re-export

/// One prompt's share of [`DecoderGpu::prefill_hidden_slots`]: rows `span` of
/// `prompt`, prefilled into `slot` at cache rows `span`, reading the hidden states
/// of the rows `read`.
pub(crate) struct SlotPrefill<'p> {
    pub(crate) prompt: &'p PromptRows<'p>,
    pub(crate) span: std::ops::Range<usize>,
    pub(crate) slot: usize,
    pub(crate) read: &'p [usize],
}

/// One sequence's run of consecutive rows in a multi-sequence chunk
/// ([`DecoderGpu::forward_segments_enc`]).
#[derive(Clone, Debug)]
pub(crate) struct ChunkSegment {
    /// The rows of the chunk (and of its activation buffers) this sequence holds.
    pub(crate) rows: std::ops::Range<usize>,
    /// The slot whose recurrent state and KV rows the rows continue.
    pub(crate) slot: usize,
    /// The cache row, within the slot, of the segment's first row.
    pub(crate) base_pos: usize,
}

/// A prompt for [`DecoderGpu::prefill_hidden`]: token ids, some runs of which are
/// given as embedding rows (an image), and optionally a rotary coordinate per row.
pub(crate) struct PromptRows<'p> {
    /// One id per row. An embedded row's id is a placeholder and is never gathered.
    pub(crate) ids: &'p [u32],
    /// `(first row, rows)`: runs of rows given directly, `d` f32 per row, in
    /// ascending order and not overlapping.
    pub(crate) embedded: &'p [(usize, &'p [f32])],
    /// `(t, h, w, e)` per row; `None` ropes every row at its cache row.
    pub(crate) positions: Option<&'p [[u32; 4]]>,
}

impl<'a> DecoderGpu<'a> {
    /// The M-row projection dispatch the chunk graph routes through: `y[M,n] =
    /// x[M,k] @ W^T`, choosing among the native / F32 / F16 / m==2 / MMA-tile / M-row
    /// Q4 kernels by weight representation and shape.
    ///
    /// Shared with the slot-decode graph: the M dimension of these kernels indexes
    /// `x[m*K + k]` and reads nothing else about `m`, so the M rows can be M tokens
    /// of one sequence (prefill) or one token of each of M sequences (slots).
    pub(crate) fn chunk_gemm(&self, enc: &metal::ComputeCommandEncoderRef, m: u32, accum: bool,
                             wname: &str, x: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32) {
        let ints = |enc: &metal::ComputeCommandEncoderRef, vals: &[(u64, u32)]| {
            for (idx, v) in vals { enc.set_bytes(*idx, 4, v as *const u32 as *const c_void); }
        };

            self.check_shape(wname, k, n); // qwen35 chunk projections bypass gemm_named
            // Native quant: the code below indexes w4/scale4 directly, so a tensor
            // kept in its own format has to be served first. The batched native
            // kernel covers every M, including the m==2 MTP case.
            if self.wt.repr(wname) == Repr::Native {
                self.nat_batched(enc, wname, x, y, k, n, m, accum);
                return;
            }
            // F32-kept tensors (dynamic-quant GGUFs: ssm_alpha/beta): per-row f32 GEMV fallback.
            if !self.wt.w4.contains_key(wname) {
                if let Some(w) = self.wt.w32.get(wname) {
                    for row in 0..m as u64 {
                        enc.set_compute_pipeline_state(&self.p[if accum { "gemv_w32_accum" } else { "gemv_w32" }]);
                        enc.set_buffer(0, Some(x), row * (k as u64) * 4);
                        enc.set_buffer(1, Some(w), 0);
                        enc.set_buffer(2, Some(y), row * (n as u64) * 4);
                        ints(enc, &[(3, k), (4, n)]);
                        enc.dispatch_thread_groups(MTLSize::new(((n + 7)/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
                    }
                    return;
                }
                if let Some(w) = self.wt.w16.get(wname) {
                    // Tiled f16 GEMM where the shape allows it; the per-row GEMV loop
                    // below is 4.5x slower (1022 -> 4664 tok/s on a 4000-token prefill).
                    // ssm_alpha/ssm_beta (N=16) fail n%64 and keep the loop.
                    if n % 64 == 0 && k % 32 == 0 && m >= 8 && self.gpu.native_reduce
                        && self.p.contains_key("gemm_mm_f16") {
                        self.gemm16_off(enc, x, 0, w, y, k, n, m, accum);
                        return;
                    }
                    for row in 0..m as u64 {
                        enc.set_compute_pipeline_state(&self.p[if accum { "gemv_accum" } else { "gemv_f16" }]);
                        enc.set_buffer(0, Some(x), row*k as u64*4);
                        enc.set_buffer(1, Some(w), 0);
                        enc.set_buffer(2, Some(y), row*n as u64*4);
                        ints(enc, &[(3,k),(4,n)]);
                        enc.dispatch_thread_groups(MTLSize::new(((n+7)/8) as u64,1,1),MTLSize::new(256,1,1));
                    }
                    return;
                }
                assert!(self.gemm_named(enc,wname,x,y,k,n,m,accum),"missing MTP matrix dispatch: {wname}");
                return;
            }
            // m==2 (MTP verify): compile-time two-token GEMV — weight-reuse with full
            // gemv parallelism (the MMA tile kernel is 4-5× slower at M=2).
            if m == 2 {
                enc.set_compute_pipeline_state(&self.p[if accum { "gemv_m2_q4_accum" } else { "gemv_m2_q4" }]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(&self.wt.w4[wname]), 0);
                enc.set_buffer(2, Some(y), 0);
                ints(enc, &[(3, k), (4, n)]);
                enc.set_buffer(5, Some(&self.wt.scale4[wname]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((n + 7)/8) as u64, 1, 1), MTLSize::new(64, 1, 1));
                return;
            }
            // MMA (simdgroup-matrix) GEMM — the reference kernel_mul_mm style. Handles
            // any m ≤ 32 (token rows ≥ m are zero-padded). Needs N%64==0; the tiny
            // N=32 alpha/beta projections fall through to the scalar kernels below.
            if n % 64 == 0 && k % 32 == 0 && m >= 8 && self.gpu.native_reduce {
                enc.set_compute_pipeline_state(&self.p["gemm_mm_q4"]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(&self.wt.w4[wname]), 0);
                enc.set_buffer(2, Some(y), 0);
                ints(enc, &[(3, k), (4, n), (6, accum as u32), (7, m)]);
                enc.set_buffer(5, Some(&self.wt.scale4[wname]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((m+31)/32) as u64, (n/64) as u64, 1), MTLSize::new(128, 1, 1));
                return;
            } else {
                enc.set_compute_pipeline_state(&self.p[if accum { "gemv_m_q4_accum" } else { "gemv_m_q4" }]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(&self.wt.w4[wname]), 0);
                enc.set_buffer(2, Some(y), 0);
                ints(enc, &[(3, k), (4, n), (6, m)]);
                enc.set_buffer(5, Some(&self.wt.scale4[wname]), 0);
            }
            enc.dispatch_thread_groups(MTLSize::new(((n + 7)/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Row-wise RMSNorm over M rows of `st.x` into `st.h`. Shared with the slot graph.
    pub(crate) fn rmsnorm_rows(&self, enc: &metal::ComputeCommandEncoderRef, m: u32, w: &metal::Buffer) {
        let d = self.d as u32;
        let ints = |enc: &metal::ComputeCommandEncoderRef, vals: &[(u64, u32)]| {
            for (idx, v) in vals { enc.set_bytes(*idx, 4, v as *const u32 as *const c_void); }
        };

            enc.set_compute_pipeline_state(&self.p["rmsnorm_m"]);
            enc.set_buffer(0, Some(&self.st.x), 0);
            enc.set_buffer(1, Some(w), 0);
            enc.set_buffer(2, Some(&self.st.h), 0);
            ints(enc, &[(3, d)]);
            enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Whether the model keeps a recurrent state, which `prefill_hidden` needs.
    pub(crate) fn is_recurrent(&self) -> bool { self.arch.ssm.is_some() }

    /// Prefill rows `span` of `prompt` into cache rows `span` of the current slot,
    /// and return the final hidden state (after `output_norm`) of the rows `read`
    /// names: ascending prompt indices inside `span`. Recurrent (qwen35) models only.
    pub(crate) fn prefill_hidden(&self, prompt: &PromptRows, span: std::ops::Range<usize>, read: &[usize]) -> Vec<Vec<f32>> {
        let job = SlotPrefill { prompt, span, slot: self.cur_slot.get(), read };
        self.prefill_hidden_slots(std::slice::from_ref(&job)).pop().unwrap()
    }

    /// [`DecoderGpu::prefill_hidden`] for several prompts at once, each in its own
    /// slot: every pass holds rows of as many prompts as fit in a chunk, one segment
    /// each ([`DecoderGpu::forward_segments_enc`]), so the weights are read once for
    /// all of them.
    ///
    /// Rows the prompt gives as embeddings enter the residual stream directly, and
    /// rows with explicit rotary coordinates are roped at those rather than at their
    /// cache row; a prompt without coordinates, beside one with, is roped at its cache
    /// rows on every axis, which is the same rotation.
    pub(crate) fn prefill_hidden_slots(&self, jobs: &[SlotPrefill]) -> Vec<Vec<Vec<f32>>> {
        assert!(self.arch.ssm.is_some(), "prefill_hidden runs the recurrent chunk graph");
        assert!(jobs.len() <= self.st.slots
            && jobs.iter().enumerate().all(|(i, j)| jobs[..i].iter().all(|k| k.slot != j.slot)),
            "prefill_hidden_slots: one slot per prompt, at most {} slots", self.st.slots);
        for j in jobs {
            assert!(j.span.end <= j.prompt.ids.len() && j.prompt.positions.is_none_or(|p| p.len() == j.prompt.ids.len()),
                "prefill_hidden: the span or the positions do not fit the prompt");
            assert!(j.read.windows(2).all(|w| w[0] < w[1]) && j.read.iter().all(|r| j.span.contains(r)),
                "prefill_hidden: rows to read must be ascending and inside the span");
        }
        let d = self.d;
        let norm = &self.wt.w32["output_norm.weight"];
        let chunk_sz = self.cfg.prefill_m.clamp(1, MAXM);
        let positioned = jobs.iter().any(|j| j.prompt.positions.is_some());
        let mut next: Vec<usize> = jobs.iter().map(|j| j.span.start).collect();
        let mut out: Vec<Vec<Vec<f32>>> = jobs.iter().map(|j| Vec::with_capacity(j.read.len())).collect();
        loop {
            let (mut tokens, mut segments, mut given, mut positions) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (i, j) in jobs.iter().enumerate() {
                let room = chunk_sz - tokens.len();
                if next[i] == j.span.end || room == 0 { continue; }
                let take = (j.span.end - next[i]).min(room);
                let (from, r0) = (next[i], tokens.len());
                tokens.extend_from_slice(&j.prompt.ids[from..from + take]);
                for p in from..from + take {
                    positions.push(j.prompt.positions.map_or([p as u32, p as u32, p as u32, 0], |pos| pos[p]));
                }
                // Embedded rows inside this run go straight into st.x.
                for &(first, rows) in j.prompt.embedded {
                    let (lo, hi) = (first.max(from), (first + rows.len() / d).min(from + take));
                    if lo >= hi { continue; }
                    let src = &rows[(lo - first) * d..(hi - first) * d];
                    unsafe {
                        let dst = (self.st.x.contents() as *mut f32).add((r0 + lo - from) * d);
                        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
                    }
                    given.push(r0 + lo - from..r0 + hi - from);
                }
                segments.push((i, ChunkSegment { rows: r0..r0 + take, slot: j.slot, base_pos: from }));
                next[i] += take;
            }
            if tokens.is_empty() { break; }
            let chunk: Vec<ChunkSegment> = segments.iter().map(|(_, g)| g.clone()).collect();
            self.forward_segments_enc(None, &tokens, &chunk, false, false, &given, positioned.then_some(positions.as_slice()));
            let wanted = segments.iter().any(|(i, g)| jobs[*i].read.iter().any(|&r| (g.base_pos..g.base_pos + g.rows.len()).contains(&r)));
            if !wanted { continue; }
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.rmsnorm_rows(enc, tokens.len() as u32, norm);
            enc.end_encoding();
            let _ = ojas_metal::commit_and_wait_checked(cb, "prefill hidden norm");
            let h = unsafe { std::slice::from_raw_parts(self.st.h.contents() as *const f32, tokens.len() * d) };
            for (i, g) in &segments {
                for &r in jobs[*i].read.iter().filter(|&&r| (g.base_pos..g.base_pos + g.rows.len()).contains(&r)) {
                    let row = g.rows.start + r - g.base_pos;
                    out[*i].push(h[row * d..(row + 1) * d].to_vec());
                }
            }
        }
        out
    }

    /// Copy slot `from`'s recurrent state and its first `rows` cache rows into slot
    /// `to`, on the GPU, so `to` continues the same prefix.
    pub(crate) fn copy_slot_prefix(&self, from: usize, to: usize, rows: usize) {
        assert!(from < self.slots() && to < self.slots() && rows <= self.st.max_seq, "copy_slot_prefix: slot or rows out of range");
        if from == to { return; }
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        let copy = |buf: &metal::Buffer, from_off: u64, to_off: u64, bytes: u64| {
            let n = (bytes / 4) as u32;
            if n == 0 { return; }
            self.enc_reduce_off(enc, "copy_buf", &[(buf, 0, to_off), (buf, 1, from_off)], &[(2, n)], &[], n.div_ceil(256) as u64, 256);
        };
        for l in 0..self.arch.n_layers {
            let lp = self.arch.layers[l];
            if lp.is_ssm {
                copy(&self.st.conv_state[l], self.conv_slot_off(l, from), self.conv_slot_off(l, to), self.st.conv_state[l].length() / self.slots() as u64);
                copy(&self.st.ssm_state[l], self.ssm_slot_off(l, from), self.ssm_slot_off(l, to), self.st.ssm_state[l].length() / self.slots() as u64);
            } else {
                // Two bytes per cached value.
                let bytes = rows as u64 * lp.kvdim as u64 * 2;
                copy(&self.st.kcache[l], self.kv_slot_off(l, from), self.kv_slot_off(l, to), bytes);
                copy(&self.st.vcache[l], self.kv_slot_off(l, from), self.kv_slot_off(l, to), bytes);
            }
        }
        enc.end_encoding();
        let _ = ojas_metal::commit_and_wait_checked(cb, "copy slot prefix");
    }

    /// One qwen35 prefill chunk (M ≤ MAXM tokens), split across command buffers every
    /// `prefill_cb_layers` layers.
    /// Mirrors the qwen35 branch of encode_forward with M-token batched kernels;
    /// activations are [M, dim] row-major throughout.
    pub(crate) fn forward_chunk(&self, tokens: &[u32], base_pos: usize, verify: bool) {
        self.forward_chunk_enc(None, tokens, base_pos, verify, false);
    }

    /// Core chunk encoder. `ext_enc`: encode into an existing encoder/cb (the
    /// draft+verify fused step); `row1_gpu`: row 1 embeds the token id from tmp[2]
    /// (the draft's argmax) instead of tokens[1], with no CPU round-trip.
    ///
    /// Embeds `tokens` itself. The `do_embed` form below is the same encoder with the
    /// embedding gather skipped; this wrapper keeps the existing call sites
    /// (`forward_chunk`, `spec.rs`'s fused draft+verify) on the signature they
    /// already pass.
    pub(crate) fn forward_chunk_enc(&self, ext: Option<&metal::ComputeCommandEncoderRef>, tokens: &[u32], base_pos: usize, verify: bool, row1_gpu: bool) {
        self.forward_chunk_enc_embed(ext, tokens, base_pos, verify, row1_gpu, true, None)
    }

    /// `forward_chunk_enc` plus `do_embed`, the sixth flag.
    ///
    /// `do_embed = false` means `self.st.x` already holds the chunk's `[M, d]`
    /// row-major residual rows, written by the host — legal because `st.x` is
    /// `StorageModeShared` (`load.rs`), so a host store is visible to the next
    /// command buffer. That is the seam a vision encoder's output enters through
    /// (`prefill_embeds`); `tokens` is then only the placeholder ids, kept for the
    /// bounds assert and the session bookkeeping, never gathered. Everything
    /// downstream of the gather is byte-for-byte the same graph, so injected rows and
    /// gathered rows are indistinguishable to the 32 layers that follow.
    ///
    /// `pos3` is the sectioned-M-RoPE argument: `Some` carries one `(t,h,w,e)`
    /// coordinate per row of this chunk, already sliced by the caller (indexed by
    /// `m`, not by `base_pos + m`). `None`, or an architecture that declares no rope
    /// sections, leaves the rope dispatch on the scalar `base_pos + m`. See
    /// [`ojas_core::Model::prefill_embeds`] for the coordinate convention.
    pub(crate) fn forward_chunk_enc_embed(&self, ext: Option<&metal::ComputeCommandEncoderRef>, tokens: &[u32], base_pos: usize, verify: bool, row1_gpu: bool, do_embed: bool, pos3: Option<&[[u32; 4]]>) {
        let one = [ChunkSegment { rows: 0..tokens.len(), slot: self.cur_slot.get(), base_pos }];
        let given = if do_embed { Vec::new() } else { vec![0..tokens.len()] };
        self.forward_segments_enc(ext, tokens, &one, verify, row1_gpu, &given, pos3)
    }

    /// [`DecoderGpu::forward_chunk_enc_embed`] over several sequences at once: each
    /// segment is a run of consecutive rows continuing its own slot. The projections
    /// and every row-wise kernel run once over all rows, so the weights are read once
    /// for the whole chunk; the work that carries a sequence's state (convolution,
    /// recurrence, rope and KV store, attention) is dispatched per segment, at the
    /// segment's rows and its slot's state, as [`DecoderGpu::encode_slots`] does for
    /// single tokens. With one segment at the current slot this is the
    /// single-sequence chunk, dispatch for dispatch.
    ///
    /// `given` lists the rows the caller has already written to `st.x` (an image's
    /// rows, say); every other row is gathered from the token embeddings.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_segments_enc(&self, ext: Option<&metal::ComputeCommandEncoderRef>, tokens: &[u32], segments: &[ChunkSegment], verify: bool, row1_gpu: bool, given: &[std::ops::Range<usize>], pos3: Option<&[[u32; 4]]>) {
        let is_given = |i: usize| given.iter().any(|r| r.contains(&i));
        debug_assert!(!row1_gpu || !is_given(1), "row1_gpu embeds row 1 — it needs gathering");
        assert!(!tokens.is_empty() && tokens.len() <= MAXM
            && tokens.iter().all(|&t| (t as usize) < self.arch.vocab), "chunk exceeds model bounds");
        assert!(!segments.is_empty() && segments.len() <= self.st.slots
            && segments.windows(2).all(|w| w[0].rows.end == w[1].rows.start)
            && segments[0].rows.start == 0 && segments.last().unwrap().rows.end == tokens.len()
            && segments.iter().all(|g| !g.rows.is_empty() && g.slot < self.st.slots
                && g.base_pos.checked_add(g.rows.len()).is_some_and(|n| n <= self.st.max_seq)),
            "chunk segments must tile the chunk, one slot each, inside the context");
        let sc = self.arch.ssm.unwrap();
        let m = tokens.len() as u32;
        let d = self.d as u32;
        // The MTP rollback snapshots are sized by `snapshot_rows`, not by slots — they
        // belong to the speculative protocol, which runs on one sequence. Prefilling a
        // verify chunk into a non-zero slot would write another slot's snapshot.
        debug_assert!(!verify || (segments.len() == 1 && segments[0].slot == 0),
            "MTP verify snapshots have no slot dimension — prefill slot 0 or disable MTP");
        // A segment's rows within the chunk's activation buffers, `width` f32 each.
        let at = |g: &ChunkSegment, width: u32| g.rows.start as u64 * width as u64 * 4;
        let rows = |g: &ChunkSegment| g.rows.len() as u32;
        let f4 = std::mem::size_of::<f32>() as u64;
        // ---- sectioned M-RoPE ---------------------------------------------------
        // Two conditions, both necessary: the caller has to have coordinates, and the
        // architecture has to declare the section split that says what to do with
        // them. surya-2/qwen35 declares [11,11,10,0] against n_rot 64 and is IMROPE
        // (interleaved t h w t h w …, llama.cpp LLM_ARCH_QWEN35 ->
        // LLAMA_ROPE_TYPE_IMROPE); the contiguous layout would be silently wrong and
        // would still pass a t==h==w degeneracy test, so the mode is named rather
        // than assumed. With no sections declared the coordinates are ignored.
        let mrope = match pos3 {
            Some(p3) => {
                assert_eq!(p3.len(), tokens.len(), "pos3 must carry one (t,h,w,e) per chunk row");
                sc.mrope_sections.iter().any(|&s| s != 0)
            }
            None => false,
        };
        let neox_arg = if mrope {
            // The mode rides in the upper bits of `neox` (see the kernel comment in
            // ops.rs): a fresh scalar slot would be read from a stale argument-table
            // entry at every call site that never enables M-RoPE.
            ojas_metal::kernels::ops::rope_mode(true, ojas_metal::kernels::ops::MROPE_INTERLEAVED)
        } else {
            ojas_metal::kernels::ops::rope_mode(true, ojas_metal::kernels::ops::MROPE_OFF)  // == 1, what every call site passes
        };
        // Each segment's descriptor starts at its own offset in st.mpos: the kernel
        // reads a header then one coordinate per row, counting rows from its dispatch.
        let mpos_at: Vec<u64> = segments.iter().scan(0u64, |at, g| {
            let here = *at;
            *at += (4 + 4 * g.rows.len() as u64) * 4;
            Some(here)
        }).collect();
        if mrope {
            // [s0,s1,s2,s3] then (t,h,w,e) per row. st.mpos is StorageModeShared and
            // sized for a header per slot plus 4*MAXM u32 (load.rs), so this host store
            // is visible to the command buffer encoded below, the same seam st.x uses.
            // It is preallocated rather than made per dispatch because an external
            // encoder's command buffer is committed elsewhere, and
            // set_unretained_command_buffers(true) would not keep a temporary alive.
            for (g, &off) in segments.iter().zip(&mpos_at) {
                let desc = ojas_metal::kernels::ops::mrope_desc(sc.mrope_sections, &pos3.unwrap()[g.rows.clone()]);
                debug_assert!(off as usize + desc.len() * 4 <= self.st.mpos.length() as usize, "mpos descriptor overruns its buffer");
                unsafe { std::ptr::copy_nonoverlapping(desc.as_ptr(), (self.st.mpos.contents() as *mut u8).add(off as usize) as *mut u32, desc.len()); }
            }
        }
        // concurrent dispatch (reference-style): independent kernels within a stage
        // overlap; bar() marks the real data dependencies (this path is qwen35-only,
        // so bar() is always active here).
        // Without an external encoder the chunk owns its command buffers, split by
        // layer (`pass.rs`).
        let mut pass = ext.is_none().then(|| super::pass::SplitPass::new(self.gpu, !self.cfg.serial, self.cfg.prefill_cb_layers, m));
        let mut enc = match ext {
            Some(e) => e.to_owned(),
            None => pass.as_mut().unwrap().open(),
        };
        // Named stage boundaries for `OJAS_PREFILL_PROFILE`; free otherwise.
        macro_rules! stage { ($label:expr) => { if let Some(p) = pass.as_mut() { p.stage($label, &mut enc); } } }
        let ints = |enc: &metal::ComputeCommandEncoderRef, vals: &[(u64, u32)]| {
            for (idx, v) in vals { enc.set_bytes(*idx, 4, v as *const u32 as *const c_void); }
        };
        // embed each token into its x row (per-token kernel + row byte-offset).
        // Skipped entirely when the caller supplied the rows: x already holds them.
        for (i, &t) in tokens.iter().enumerate() {
            if is_given(i) { continue; }
            if row1_gpu && i == 1 {
                // draft-chained verify: token id comes from tmp[2] on the GPU
                enc.set_compute_pipeline_state(&self.p["embed_q4_id"]);
                enc.set_buffer(0, Some(&self.wt.w4["token_embd.weight"]), 0);
                enc.set_buffer(1, Some(&self.st.x), (i as u64) * (d as u64) * f4);
                ints(&enc, &[(2, d), (5, 0)]);
                enc.set_buffer(3, Some(&self.sp.mtp_tok), 0);
                enc.set_buffer(4, Some(&self.wt.scale4["token_embd.weight"]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((d + 63)/64) as u64, 1, 1), MTLSize::new(64, 1, 1));
                continue;
            }
            self.embed_named_off(&enc, "token_embd.weight", t, d, (i as u64)*(d as u64)*f4);
        }
        self.bar(&enc); // x (embeddings) ready
        // The M-row projection and norm dispatches live as methods (`chunk_gemm`,
        // `rmsnorm_rows`) so the slot-decode graph reuses the same routing rather than
        // growing a second copy that drifts. These closures only bind `m`.
        let gemm = |enc: &metal::ComputeCommandEncoderRef, accum: bool, wname: &str, x: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32| {
            self.chunk_gemm(enc, m, accum, wname, x, y, k, n)
        };
        let rmsnorm_m = |enc: &metal::ComputeCommandEncoderRef, w: &metal::Buffer| {
            self.rmsnorm_rows(enc, m, w)
        };
        for l in 0..self.arch.n_layers {
            if let Some(pass) = pass.as_mut() { pass.layer(l, &mut enc); }
            let p = |s: &str| format!("blk.{l}.{s}");
            let lp = self.arch.layers[l];
            stage!("attn_norm");
            rmsnorm_m(&enc, &self.wt.w32[&p("attn_norm.weight")]);
            self.bar(&enc); // h ready for the projections
            if lp.is_ssm {
                stage!("ssm_proj");
                let (s_st, hk, hv) = (sc.d_state, sc.n_group, sc.dt_rank);
                let d_inner = sc.d_inner; let conv_ch = d_inner + 2*hk*s_st; let conv_k = sc.conv_kernel;
                let head_v = d_inner / hv;
                gemm(&enc, false, &p("attn_qkv.weight"), &self.st.h, &self.st.ssm_qkv, d, conv_ch);
                gemm(&enc, false, &p("attn_gate.weight"), &self.st.h, &self.st.ssm_z, d, d_inner);
                gemm(&enc, false, &p("ssm_alpha.weight"), &self.st.h, &self.st.ssm_gate, d, hv);
                gemm(&enc, false, &p("ssm_beta.weight"), &self.st.h, &self.st.ssm_beta, d, hv);
                self.bar(&enc); // qkv/z/alpha/beta projections done (ran concurrently)
                stage!("ssm_conv");
                self.enc_reduce(&enc, "ssm_ab", &[(&self.st.ssm_gate, 0), (&self.st.ssm_beta, 1), (&self.wt.w32[&p("ssm_dt.bias")], 2), (&self.wt.w32[&p("ssm_a")], 3)], &[(4, m*hv), (5, hv)], &[], ((m*hv + 63)/64) as u64, 64);
                for g in segments {
                    // Tokens in parallel (`conv1d_prefill_tiled`): 1.5 against 3.3 ms per
                    // 256-token chunk of Qwen3.5 4B for the token-serial kernel.
                    let conv_o = self.conv_slot_off(l, g.slot);
                    enc.set_compute_pipeline_state(&self.p["conv1d_prefill_tiled"]);
                    enc.set_buffer(0, Some(&self.st.ssm_qkv), at(g, conv_ch));
                    enc.set_buffer(1, Some(&self.st.conv_state[l]), conv_o);
                    enc.set_buffer(2, Some(&self.wt.w32[&p("ssm_conv1d.weight")]), 0);
                    ints(&enc, &[(3, conv_ch), (4, conv_k), (5, rows(g))]);
                    enc.set_buffer(6, Some(if verify { &self.sp.conv_snap[l] } else { &self.st.conv_state[l] }),
                                   if verify { 0 } else { conv_o });
                    ints(&enc, &[(7, if verify { 0 } else { u32::MAX })]);
                    // 16 channels per threadgroup: the kernel's CONV_TILE_C.
                    enc.dispatch_thread_groups(MTLSize::new(conv_ch.div_ceil(16) as u64, 1, 1), MTLSize::new(256, 1, 1));
                }
                self.bar(&enc); // ssm_ab + conv done (ran concurrently)
                stage!("deltanet");
                // q/k normalized once per (head, token) rather than in each of a head's
                // S state columns; the recurrence then reads them as given (l2_mode 2).
                enc.set_compute_pipeline_state(&self.p["qk_l2norm_heads"]);
                enc.set_buffer(0, Some(&self.st.ssm_qkv), 0);
                ints(&enc, &[(1, s_st), (2, hk), (3, conv_ch), (5, 0)]);
                enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new((2*hk) as u64, m as u64, 1), MTLSize::new(32, 1, 1));
                self.bar(&enc); // q/k normalized
                for g in segments {
                    let ssm_o = self.ssm_slot_off(l, g.slot);
                    enc.set_compute_pipeline_state(&self.p["deltanet_fused"]);
                    enc.set_buffer(0, Some(&self.st.ssm_state[l]), ssm_o);
                    enc.set_buffer(1, Some(&self.st.ssm_qkv), at(g, conv_ch));
                    enc.set_buffer(2, Some(&self.st.ssm_gate), at(g, hv));
                    enc.set_buffer(3, Some(&self.st.ssm_beta), at(g, hv));
                    enc.set_buffer(4, Some(&self.st.ssm_o), at(g, d_inner));
                    ints(&enc, &[(5, s_st), (6, hk), (7, hv), (8, conv_ch), (9, rows(g))]);
                    enc.set_bytes(10, 4, &self.arch.eps as *const f32 as *const c_void);
                    enc.set_buffer(11, Some(if verify { &self.sp.ssm_snap[l] } else { &self.st.ssm_state[l] }),
                                    if verify { 0 } else { ssm_o });
                    // OJAS_KMAP_DIV=1 selects the grouped value->key head mapping.
                    ints(&enc, &[(12, if verify { 0 } else { u32::MAX }),
                                 (13, self.cfg.moe_kmap_div as u32), (14, 2)]);
                    enc.dispatch_thread_groups(MTLSize::new((s_st/4) as u64, hv as u64, 1), MTLSize::new(128, 1, 1));
                }
                self.bar(&enc); // deltanet done
                stage!("ssm_norm");
                enc.set_compute_pipeline_state(&self.p["gated_rmsnorm"]);
                enc.set_buffer(0, Some(&self.st.ssm_o), 0);
                enc.set_buffer(1, Some(&self.wt.w32[&p("ssm_norm.weight")]), 0);
                enc.set_buffer(2, Some(&self.st.ssm_z), 0);
                ints(&enc, &[(3, head_v), (5, d_inner), (6, 0)]);   // qwen35: silu gate
                enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(hv as u64, m as u64, 1), MTLSize::new(32, 1, 1));
                self.bar(&enc); // gated norm done
                stage!("ssm_out");
                gemm(&enc, true, &p("ssm_out.weight"), &self.st.ssm_o, &self.st.x, d_inner, d);
            } else {
                let (hd, kvdim, qdim) = (lp.head_dim, lp.kvdim, lp.qdim);
                let group = lp.n_head / lp.n_kv.max(1);
                stage!("attn_qkv");
                gemm(&enc, false, &p("attn_q.weight"), &self.st.h, &self.st.ssm_qkv, d, 2*qdim);
                gemm(&enc, false, &p("attn_k.weight"), &self.st.h, &self.st.k, d, kvdim);
                gemm(&enc, false, &p("attn_v.weight"), &self.st.h, &self.st.v, d, kvdim);
                self.bar(&enc); // q/k/v projections done (ran concurrently)
                stage!("attn_rope");
                self.enc_reduce(&enc, "qgate_split", &[(&self.st.ssm_qkv, 0), (&self.st.q, 1)], &[(2, hd), (3, qdim), (4, m)], &[], ((m*qdim + 63)/64) as u64, 64);
                let (nq, nk) = (lp.n_head, lp.n_kv);
                self.bar(&enc); // q split done
                enc.set_compute_pipeline_state(&self.p["qk_rmsnorm"]);
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.st.k), 0);
                enc.set_buffer(2, Some(&self.wt.w32[&p("attn_q_norm.weight")]), 0);
                enc.set_buffer(3, Some(&self.wt.w32[&p("attn_k_norm.weight")]), 0);
                ints(&enc, &[(4, hd), (5, nq), (6, nk)]);
                enc.set_bytes(7, 4, &self.arch.eps as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new((nq + nk) as u64, m as u64, 1), MTLSize::new(32, 1, 1));
                self.bar(&enc); // qk-norm done
                let (aq, ak) = (qdim/2, kvdim/2);
                // Buffer 14 is `st.mpos` whether or not M-RoPE is on: at mode 0 the
                // kernel never dereferences it, but Metal API Validation asserts on a
                // declared-and-unbound buffer regardless, so the real descriptor
                // buffer is bound rather than a dummy.
                //
                // `rope_qk_store_m` writes the cache at `(base_pos + m) * kvdim`
                // relative to the bound base, so a sequence slot is a buffer offset
                // and the kernel needs no slot argument of its own.
                for (g, &mp) in segments.iter().zip(&mpos_at) {
                    let kv_o = self.kv_slot_off(l, g.slot);
                    self.enc_reduce_off(&enc, "rope_qk_store_m",
                        &[(&self.st.q, 0, at(g, qdim)), (&self.st.k, 1, at(g, kvdim)), (&self.st.v, 2, at(g, kvdim)),
                          (&self.st.kcache[l], 3, kv_o), (&self.st.vcache[l], 4, kv_o), (&self.st.mpos, 14, mp)],
                        &[(5, hd), (6, g.base_pos as u32), (8, aq), (9, ak), (10, kvdim), (11, rows(g)), (12, neox_arg), (13, sc.n_rot)], &[(7, lp.rope_base)],
                        ((rows(g)*(aq + ak + kvdim) + 63)/64) as u64, 64);
                }
                self.bar(&enc); // rope + cache store done
                stage!("attention");
                if self.arch.sparse_budget.is_some() {
                    // keep page min/max metadata current for the pages this chunk touched
                    for g in segments {
                        let (base_pos, m) = (g.base_pos, rows(g));
                        let pg0 = (base_pos / ojas_metal::kernels::attn::PAGE) as u32;
                        let npg = ((base_pos + m as usize + ojas_metal::kernels::attn::PAGE - 1) / ojas_metal::kernels::attn::PAGE) as u32 - pg0;
                        self.enc_reduce_off(&enc, "page_minmax",
                            &[(&self.st.kcache[l], 0, self.kv_slot_off(l, g.slot)), (&self.st.pmeta[l], 1, 0)],
                            &[(2, kvdim), (3, pg0), (4, (base_pos + m as usize) as u32)], &[],
                            npg as u64, 256);
                    }
                    self.bar(&enc); // metadata current before attention reads scores
                }
                // The MMA kernel serves every context length; the scalar
                // `attention_m_short` serves only the head dims it cannot tile. At hd 256 the scalar
                // kernel runs at 0.35 TFLOPS, 48.6 of 523 ms in a 512-token chunk of
                // Qwen3.5 4B, and the MMA kernel takes pp512 from 946 to 1033 tok/s (M2 Max).
                for g in segments {
                    let (kv_o, base_pos, m) = (self.kv_slot_off(l, g.slot), g.base_pos as u32, rows(g));
                    let (q_at, a_at) = (at(g, qdim), at(g, qdim));
                    if hd <= 256 && hd % 64 != 0 {
                        self.enc_reduce_off(&enc, "attention_m_short",
                            &[(&self.st.q, 0, q_at), (&self.st.kcache[l], 1, kv_o), (&self.st.vcache[l], 2, kv_o), (&self.st.attn, 3, a_at)],
                            &[(4, hd), (5, kvdim), (6, base_pos), (7, group), (9, lp.n_head)], &[(8, lp.scale)],
                            (m * lp.n_head) as u64, 256);
                    } else if hd > 256 || !self.gpu.native_reduce {
                        self.enc_reduce_off(&enc, "attention_m",
                            &[(&self.st.q, 0, q_at), (&self.st.kcache[l], 1, kv_o), (&self.st.vcache[l], 2, kv_o), (&self.st.attn, 3, a_at)],
                            &[(4, hd), (5, kvdim), (6, base_pos), (7, group), (9, lp.n_head)], &[(8, lp.scale)],
                            (m * lp.n_head) as u64, 256);
                    } else {
                        // MMA flash-attention: 32 queries/tg, simdgroup-matrix Q·K^T and P·V
                        // (hd-specialized pipeline when available — loops fully unrolled)
                        let kname = if ojas_metal::kernels::attn::ATTN_HD_SPECIAL.contains(&hd) {
                            format!("attention_m_mma_{hd}")
                        } else { "attention_m_mma".to_string() };
                        enc.set_compute_pipeline_state(&self.p[&kname]);
                        enc.set_buffer(0, Some(&self.st.q), q_at);
                        enc.set_buffer(1, Some(&self.st.kcache[l]), kv_o);
                        enc.set_buffer(2, Some(&self.st.vcache[l]), kv_o);
                        enc.set_buffer(3, Some(&self.st.attn), a_at);
                        ints(&enc, &[(4, hd), (5, kvdim), (6, base_pos), (7, group), (9, lp.n_head), (10, m)]);
                        enc.set_bytes(8, 4, &lp.scale as *const f32 as *const c_void);
                        enc.dispatch_thread_groups(MTLSize::new(lp.n_head as u64, ((m + 31)/32) as u64, 1), MTLSize::new(256, 1, 1));
                    }
                }
                self.bar(&enc); // attention done
                self.enc_reduce(&enc, "gate_mul_sigmoid", &[(&self.st.attn, 0), (&self.st.ssm_qkv, 1)], &[(2, hd), (3, qdim), (4, m)], &[], ((m*qdim + 63)/64) as u64, 64);
                self.bar(&enc); // gate applied
                stage!("attn_out");
                gemm(&enc, true, &p("attn_output.weight"), &self.st.attn, &self.st.x, qdim, d);
            }
            // FFN: post_attention_norm pre-norm, SwiGLU via two GEMMs + silu_mul
            self.bar(&enc); // mixer residual in x
            stage!("ffn_norm");
            rmsnorm_m(&enc, &self.wt.w32[&p("post_attention_norm.weight")]);
            self.bar(&enc); // h ready for FFN
            if let Some(mo) = self.arch.moe {
                // Batched MoE FFN (the reference mul_mm_id shape): every row's router,
                // expert SwiGLU and down-proj is one grid-wide dispatch, so
                // threadgroups hitting the same expert run concurrently and its weight
                // stream is read once through the cache instead of once per token.
                // 4 barriers per layer per chunk, not 4 per row.
                let (ne, nu, fe, fs) = (mo.n_expert, mo.n_used, mo.ffn_exp, mo.ffn_shexp);
                self.check_moe(&p, d, fe);
                let ib = |enc: &metal::ComputeCommandEncoderRef, idx: u64, v: u32| {
                    enc.set_bytes(idx, 4, &v as *const u32 as *const c_void);
                };
                // stage 1 (concurrent): router logits + shexp scalar gates + shexp gate/up
                enc.set_compute_pipeline_state(&self.p["gemv_w32_m"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w32[&p("ffn_gate_inp.weight")]), 0);
                enc.set_buffer(2, Some(&self.ms.moe_blg), 0);
                ib(&enc, 3, d); ib(&enc, 4, ne);
                enc.dispatch_thread_groups(MTLSize::new(((ne + 7)/8) as u64, m as u64, 1), MTLSize::new(256, 1, 1));
                enc.set_compute_pipeline_state(&self.p["gemv_w32_m"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w32[&p("ffn_gate_inp_shexp.weight")]), 0);
                enc.set_buffer(2, Some(&self.ms.moe_bsh), 0);
                ib(&enc, 3, d); ib(&enc, 4, 1);
                enc.dispatch_thread_groups(MTLSize::new(1, m as u64, 1), MTLSize::new(32, 1, 1));
                gemm(&enc, false, &p("ffn_gate_shexp.weight"), &self.st.h, &self.ms.moe_bg, d, fs);
                gemm(&enc, false, &p("ffn_up_shexp.weight"), &self.st.h, &self.ms.moe_bu, d, fs);
                self.bar(&enc); // logits + shexp projections done
                // stage 2 (concurrent): per-row top-k + shexp SwiGLU activation
                enc.set_compute_pipeline_state(&self.p["moe_topk_m"]);
                enc.set_buffer(0, Some(&self.ms.moe_blg), 0);
                enc.set_buffer(1, Some(&self.ms.moe_bidx), 0);
                enc.set_buffer(2, Some(&self.ms.moe_bwgt), 0);
                ib(&enc, 3, ne); ib(&enc, 4, nu);
                enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(32, 1, 1));
                self.enc_reduce(&enc, "silu_mul", &[(&self.ms.moe_bg, 0), (&self.ms.moe_bu, 1), (&self.ms.moe_bg, 2)],
                    &[(3, m*fs)], &[], ((m*fs + 63)/64) as u64, 64);
                self.bar(&enc); // routing tables + shexp act ready
                // stage 3 (concurrent): shexp down + all rows' routed expert SwiGLUs
                gemm(&enc, false, &p("ffn_down_shexp.weight"), &self.ms.moe_bg, &self.ms.moe_btmp, fs, d);
                enc.set_compute_pipeline_state(&self.p["moe_gu_q4_m"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_gate_exps.weight")]), 0);
                enc.set_buffer(2, Some(&self.wt.w4[&p("ffn_up_exps.weight")]), 0);
                enc.set_buffer(3, Some(&self.ms.moe_bact), 0);
                ib(&enc, 4, d); ib(&enc, 5, fe);
                enc.set_buffer(6, Some(&self.wt.scale4[&p("ffn_gate_exps.weight")]), 0);
                enc.set_buffer(7, Some(&self.wt.scale4[&p("ffn_up_exps.weight")]), 0);
                enc.set_buffer(8, Some(&self.ms.moe_bidx), 0);
                ib(&enc, 9, nu);
                enc.dispatch_thread_groups(MTLSize::new(((fe + 7)/8) as u64, (m*nu) as u64, 1), MTLSize::new(64, 1, 1));
                self.bar(&enc); // expert activations + shexp down done
                // stage 4: weighted expert down + shexp add into x rows
                enc.set_compute_pipeline_state(&self.p["moe_down_q4_m"]);
                enc.set_buffer(0, Some(&self.ms.moe_bact), 0);
                enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_down_exps.weight")]), 0);
                enc.set_buffer(2, Some(&self.st.x), 0);
                ib(&enc, 3, fe); ib(&enc, 4, d);
                enc.set_buffer(5, Some(&self.wt.scale4[&p("ffn_down_exps.weight")]), 0);
                enc.set_buffer(6, Some(&self.ms.moe_bidx), 0);
                enc.set_buffer(7, Some(&self.ms.moe_bwgt), 0);
                ib(&enc, 8, nu);
                enc.set_buffer(9, Some(&self.ms.moe_btmp), 0);
                enc.set_buffer(10, Some(&self.ms.moe_bsh), 0);
                enc.dispatch_thread_groups(MTLSize::new(((d + 7)/8) as u64, m as u64, 1), MTLSize::new(64, 1, 1));
                self.bar(&enc); // layer output in x
            } else {
            let nffn = self.arch.ffn as u32;
            // SwiGLU tile fusion loses to this concurrent 2-GEMM + silu_mul pipeline:
            // a silu-in-up epilogue and a dual-stream gate+up MMA kernel measured
            // 320/322 against 343 tok/s, because the fusions either serialize the
            // projections or double accumulator pressure.
            stage!("ffn_gate_up");
            gemm(&enc, false, &p("ffn_gate.weight"), &self.st.h, &self.st.gate, d, nffn);
            gemm(&enc, false, &p("ffn_up.weight"), &self.st.h, &self.st.up, d, nffn);
            self.bar(&enc); // gate + up done (ran concurrently)
            stage!("ffn_act");
            self.enc_reduce(&enc, "silu_mul", &[(&self.st.gate, 0), (&self.st.up, 1), (&self.st.act, 2)], &[(3, m*nffn)], &[], ((m*nffn + 63)/64) as u64, 64);
            self.bar(&enc); // SwiGLU act ready
            stage!("ffn_down");
            gemm(&enc, true, &p("ffn_down.weight"), &self.st.act, &self.st.x, nffn, d);
            self.bar(&enc); // layer output in x
            }
        }
        if self.sp.mtp.is_some() || verify { stage!("tail"); }
        if self.sp.mtp.is_some() {
            self.bar(&enc);
            self.enc_reduce(&enc, "copy_buf", &[(&self.sp.mtp_h, 0), (&self.st.x, 1)], &[(2, m*d)], &[], ((m*d + 63)/64) as u64, 64);
        }
        if verify {
            // MTP verify tail: capture pre-output-norm hidden rows (next draft input),
            // then logits + argmax per row (tmp[row] = predicted token id).
            self.bar(&enc);
            rmsnorm_m(&enc, &self.wt.w32["output_norm.weight"]);
            self.bar(&enc);
            let lm = self.arch.lm_head.clone();
            gemm(&enc, false, &lm, &self.st.h, &self.st.logits, d, self.arch.vocab as u32);
            self.bar(&enc);
            for row in 0..m as u64 {
                enc.set_compute_pipeline_state(&self.p["argmax"]);
                enc.set_buffer(0, Some(&self.st.logits), row * (self.arch.vocab as u64) * 4);
                enc.set_buffer(1, Some(&self.st.tmp), row * 4);
                enc.set_bytes(2, 4, &(self.arch.vocab as u32) as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(self.tune.max_tg.min(1024), 1, 1));
            }
        }
        if let Some(pass) = pass {
            let cbs = pass.command_buffers();
            let gpu = pass.finish(&enc, "chunked prefill");
            self.gpu_s.set(self.gpu_s.get() + gpu);
            if self.cfg.prefill_dbg {
                tracing::trace!(target: "prefill", "chunk M={} command buffers={} gpu={:.1}ms", m, cbs, gpu * 1e3);
            }
        }
    }

}
