#![allow(clippy::too_many_arguments)]
use super::*;
use objc::{msg_send, sel, sel_impl};
 // re-export

/// Layer index at which the per-token forward is split into a second command
/// buffer. 0 disables the split. Default: a quarter of the stack — enough work in
/// the first buffer to cover the CPU encode of the second, without paying an extra
/// submit for a trivial amount of GPU work.
fn cb_split(n_layers: usize) -> usize {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let v = *V.get_or_init(|| std::env::var("OJAS_CB_SPLIT").ok().and_then(|s| s.parse().ok()));
    v.unwrap_or(n_layers / 4)
}

impl<'a> DecoderGpu<'a> {
    pub fn forward(&self, token: u32, pos: usize) -> Vec<f32> {
        // Use the same dispatch and state bookkeeping as greedy decode. In
        // particular, prec=4 on a dense model has no MoE Route/Experts split.
        self.forward_id(token, pos);
        let ptr = self.st.logits.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.arch.vocab) }.to_vec()
    }

    /// Last-token text embedding: run the ids through the decode path (fresh
    /// context at pos 0) and read back the post-output_norm hidden state of the
    /// final token instead of logits. This is the pooling Qwen3-Embedding-style
    /// decoder embedders specify (last token; normalization is the caller's step).
    /// Streamed-MoE models are refused.
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if self.strm.stream {
            return Err("embedding unsupported for disk-streamed models".into());
        }
        if ids.is_empty() {
            return Err("embed: empty token sequence".into());
        }
        if ids.len() > self.st.max_seq {
            return Err(format!("embed: {} tokens exceeds ctx {}", ids.len(), self.st.max_seq));
        }
        let d32 = self.d as u32;
        let hd = self.arch.hd as u32;
        let kvdim = (self.arch.n_kv * self.arch.hd) as u32;
        let group = (self.arch.n_head / self.arch.n_kv) as u32;
        let scale = 1.0 / (self.arch.hd as f32).sqrt();
        for (pos, &t) in ids.iter().enumerate() {
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_forward(&enc, t, pos, d32, hd, kvdim, group, scale, (pos + 1) as u32);
            enc.end_encoding();
            let _ = ojas_metal::commit_and_wait_checked(cb, "scalar decode");
        }
        let ptr = self.st.h.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.d) }.to_vec())
    }

    /// Greedy decode step returning only the argmax token id. Runs the full forward
    /// plus a GPU argmax in one command buffer and reads back 4 bytes — avoiding the
    /// per-token 600KB logits copy + CPU scan that `forward()` incurs.
    pub fn forward_id(&self, token: u32, pos: usize) -> u32 {
        assert!(pos < self.st.max_seq && (token as usize) < self.arch.vocab, "decode token or position exceeds model bounds");
        // Streaming (GLM/DeepSeek disk-streamed MoE): the whole-token forward binds
        // every layer's expert buffers, so a single command buffer would make all
        // ~440GB of no-copy mmap'd experts resident at commit → OOM. Chunk the forward
        // into small layer ranges, each its own command buffer, so only
        // OJAS_STREAM_CHUNK layers' experts are wired at a time (the rest stay
        // reclaimable page cache). With resident experts routing never needs the CPU,
        // so the whole token is encoded at once like a dense model.
        if self.strm.stream && self.phase_split() {
            let trace = self.flash_trace_start();
            let next = if self.resident_any() {
                self.forward_id_partial(token, pos)
            } else {
                self.forward_id_streamed(token, pos)
            };
            let mut history = self.sess.session_tokens.borrow_mut();
            if pos <= history.len() { history.truncate(pos); history.push(token); }
            drop(history);
            if next != u32::MAX { self.sync_mtp_single(token, pos); }
            self.flash_trace_finish(trace, "scalar", pos, 1, FlashTargetTiming::default());
            return next;
        }
        let d = self.d as u32;
        let hd = self.arch.hd as u32;
        let kvdim = (self.arch.n_kv * self.arch.hd) as u32;
        let group = (self.arch.n_head / self.arch.n_kv) as u32;
        let scale = 1.0 / (self.arch.hd as f32).sqrt();
        let seq = (pos + 1) as u32;
        // Split submit. Encoding a token is ~290 dispatches and ~0.8 ms of CPU; with
        // one command buffer the GPU sits idle for all of it, since nothing is
        // submitted until the last dispatch is encoded. Splitting the layer stack
        // across two command buffers and committing the first immediately lets the GPU
        // start on the early layers while the CPU encodes the late ones. The reference
        // Metal backend does the same: the first max(64, 0.1*n_nodes) nodes on the
        // calling thread, submitted, then the rest on n_cb further threads with their
        // own command buffers. `enqueue` reserves each buffer's slot in the queue up
        // front, so execution order is fixed regardless of when encoding finishes.
        let nl = self.arch.n_layers;
        let split = cb_split(nl);
        // qwen4exp's scalar graph assumes serial encoders (its streamed path always
        // ran them), so it stays serial under resident mode too: it relies on
        // implicit hazard ordering rather than on explicit barriers.
        let conc = self.arch.ssm.is_some() && !self.cfg.serial && self.arch.qwen4exp.is_none();
        let (cb, cb_first) = if split > 0 && split < nl {
            let cb1 = self.gpu.command_buffer();
            let cb2 = self.gpu.command_buffer();
            cb1.enqueue();
            cb2.enqueue();                     // order reserved before either is encoded
            let e1 = if conc { cb1.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent) } else { cb1.new_compute_command_encoder() };
            self.encode_forward_span(&e1, token, pos, d, hd, kvdim, group, scale, seq, 0, split, true, false);
            e1.end_encoding();
            cb1.commit();                      // GPU starts here, CPU keeps encoding
            let e2 = if conc { cb2.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent) } else { cb2.new_compute_command_encoder() };
            self.encode_forward_span(&e2, token, pos, d, hd, kvdim, group, scale, seq, split, nl, false, true);
            self.bar(&e2);
            self.enc_reduce(&e2, "argmax", &[(&self.st.logits, 0), (&self.st.tmp, 1)], &[(2, self.arch.vocab as u32)], &[], 1, self.tune.max_tg.min(1024));
            if self.want_topk.get() { self.enc_topk1(&e2); }
            e2.end_encoding();
            let _ = ojas_metal::commit_and_wait_checked(cb2, "scalar decode");
            (cb2, Some(cb1))
        } else {
            let cb = self.gpu.command_buffer();
            let enc = if conc { cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent) } else { cb.new_compute_command_encoder() };
            self.encode_forward(&enc, token, pos, d, hd, kvdim, group, scale, seq);
            self.bar(&enc);
            // GPU argmax over logits -> tmp[0] (as u32). Threads clamped to device max.
            self.enc_reduce(&enc, "argmax", &[(&self.st.logits, 0), (&self.st.tmp, 1)], &[(2, self.arch.vocab as u32)], &[], 1, self.tune.max_tg.min(1024));
            if self.want_topk.get() { self.enc_topk1(&enc); }
            enc.end_encoding();
            let _ = ojas_metal::commit_and_wait_checked(cb, "scalar decode");
            (cb, None)
        };
        if self.cfg.glm_dbg {
            let lg = unsafe { std::slice::from_raw_parts(self.st.logits.contents() as *const f32, self.arch.vocab) };
            let (mut mx, mut mn, mut ai) = (f32::MIN, f32::MAX, 0usize);
            for (i,&v) in lg.iter().enumerate() { if v>mx {mx=v; ai=i;} if v<mn {mn=v;} }
            let nan = lg.iter().filter(|x| x.is_nan()).count();
            let xb = unsafe { std::slice::from_raw_parts(self.st.x.contents() as *const f32, self.d) };
            let xabs: f32 = xb.iter().map(|v| v.abs()).sum();
            tracing::trace!(target: "glm-dbg", "pos{} sum|x_final|={:.3} logits max={:.4} min={:.4} nan={} argmax={}", pos, xabs, mx, mn, nan, ai);
        }
        // GPU busy time spans both buffers when the forward was split.
        let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        let gs = match &cb_first {
            Some(c) => { let s0: f64 = unsafe { msg_send![*c, GPUStartTime] }; s0.min(gs) }
            None => gs,
        };
        self.gpu_s.set(self.gpu_s.get() + (ge - gs));
        self.sync_mtp_single(token, pos);
        // Track the sequence for prefix reuse: if this token contiguously extends
        // what the cache holds (decode after prefill, or the per-token prefill path),
        // record it so the next turn can reuse this response, not just the prompt.
        // Dense KV is positional and persistent, so this runs for every arch, not
        // just SSM. Snapshots stay SSM-only (dense has none).
        if pos == self.sess.session_tokens.borrow().len() {
            self.sess.session_tokens.borrow_mut().push(token);
            if self.arch.ssm.is_some() {
                self.maybe_snapshot(pos + 1);
            }
        }
        // temporal prefetch: pre-fault the experts this token used (strong temporal
        // locality → they're the best prediction for the next token).
        if let Some(mc) = self.arch.moe {
            if self.strm.prefetch.is_some() || self.strm.stream_prefetch.is_some() {
                let nu = mc.n_used as usize;
                let ids = unsafe { std::slice::from_raw_parts(self.ms.moe_idx.contents() as *const u32, self.arch.n_layers * MAXM * nu) };
                let mut used = Vec::with_capacity(self.arch.n_layers * nu);
                for l in 0..self.arch.n_layers {
                    for j in 0..nu {
                        let e = ids[l * MAXM * nu + j];
                        if e < mc.n_expert { used.push((l as u32, e)); }
                    }
                }
                if let Some(pf) = &self.strm.prefetch { pf.note(used.clone()); }
                if let Some(sp) = &self.strm.stream_prefetch { sp.note(used); }
            }
        }
        if self.cfg.ssm_dbg && pos == 0 {
            // Layer-0 captures only (see the dbg copies in encode_forward): ssm scratch
            // is reused per layer, so a direct readback would show the last layer.
            let f3 = |b: &metal::Buffer| { let s = unsafe { std::slice::from_raw_parts(b.contents() as *const f32, 3) }; format!("[{:.4} {:.4} {:.4}]", s[0], s[1], s[2]) };
            tracing::trace!(target: "ssm", "L0 z={} (llama z-0=[-0.838 -1.200 -1.125])  l_out={} (llama l_out-0=[-0.0527 0.0667 0.0304])", f3(&self.st.gate), f3(&self.st.up));
        }
        unsafe { *(self.st.tmp.contents() as *const u32) }
    }


    /// Whether this model can decode several independent sequences per step.
    ///
    /// Each clause is a capability the slot graph (`encode_slots`) does not encode,
    /// and refuses rather than mis-decodes:
    ///
    /// * qwen35 hybrid only: `encode_slots` is the qwen35 graph at M=B, and a dense
    ///   or MLA arch would need its own.
    /// * No qwen4exp: hyper-connection lanes and the PLE n-gram rows are extra
    ///   per-sequence state with no slot dimension allocated.
    /// * No MoE: the batched expert path routes through `moe_b*` scratch laid out per
    ///   token of one sequence; sharing it across slots needs a pass the slot graph
    ///   does not make. surya-2 is dense, so the OCR path is unaffected.
    /// * No streamed experts: streaming chunks the forward per layer range for
    ///   residency reasons that conflict with a single batched pass. The test is
    ///   `strm.stream && moe.is_some()`, not `strm.stream`: `stream` is set from
    ///   `prec == 4` unconditionally (`load.rs:127`), so testing it alone refuses
    ///   every model at the default precision, including dense ones with no experts
    ///   to stream. `moe.is_none()` above already implies it; it is spelled out so
    ///   that relaxing the MoE restriction later does not re-introduce the trap.
    /// * No page-sparse decode: `pmeta`/`plist` have no slot dimension, so two slots
    ///   would select pages against each other's metadata.
    /// * No MTP: the draft carry (`mtp_h`/`mtp_hprev`) and the rollback snapshots are
    ///   single-sequence by construction.
    pub fn slots_ok(&self) -> bool {
        self.st.slots > 1
            && self.arch.ssm.is_some()
            && self.arch.qwen4exp.is_none()
            && self.arch.moe.is_none()
            && self.arch.mla.is_none()
            && !self.arch.gpt_oss
            && !(self.strm.stream && self.arch.moe.is_some())
            && self.arch.sparse_budget.is_none()
            && self.sp.mtp.is_none()
    }

    /// Decode one token for each of B independent sequences in one step.
    ///
    /// Returns the argmax per entry of `steps`, in the same order. This is the driver
    /// around [`DecoderGpu::encode_slots`]: bounds, the command-buffer split, and the
    /// readback.
    pub fn decode_slots(&self, steps: &[(usize, u32, usize)]) -> Vec<u32> {
        assert!(!steps.is_empty() && steps.len() <= self.st.slots,
            "decode_slots: {} steps against {} allocated slots", steps.len(), self.st.slots);
        for &(slot, token, pos) in steps {
            assert!(slot < self.st.slots && pos < self.st.max_seq && (token as usize) < self.arch.vocab,
                "decode_slots: (slot {slot}, token {token}, pos {pos}) exceeds model bounds");
        }
        // Two slots naming the same state in one pass would both read the same
        // recurrent state and then race to write it, and the survivor's tokens would
        // look plausible, so it is checked rather than assumed.
        assert!((1..steps.len()).all(|i| steps[..i].iter().all(|s| s.0 != steps[i].0)),
            "decode_slots: a slot appears twice in one step");
        let nl = self.arch.n_layers;
        let split = cb_split(nl);
        // qwen35 under concurrent dispatch, same condition the scalar decode uses.
        let conc = !self.cfg.serial;
        let (cb, cb_first) = if split > 0 && split < nl {
            let cb1 = self.gpu.command_buffer();
            let cb2 = self.gpu.command_buffer();
            cb1.enqueue();
            cb2.enqueue();                     // order reserved before either is encoded
            let e1 = if conc { cb1.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent) } else { cb1.new_compute_command_encoder() };
            self.encode_slots(&e1, steps, 0, split, true, false);
            e1.end_encoding();
            cb1.commit();                      // GPU starts here, CPU keeps encoding
            let e2 = if conc { cb2.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent) } else { cb2.new_compute_command_encoder() };
            self.encode_slots(&e2, steps, split, nl, false, true);
            e2.end_encoding();
            let _ = ojas_metal::commit_and_wait_checked(cb2, "slot decode");
            (cb2, Some(cb1))
        } else {
            let cb = self.gpu.command_buffer();
            let enc = if conc { cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent) } else { cb.new_compute_command_encoder() };
            self.encode_slots(&enc, steps, 0, nl, true, true);
            enc.end_encoding();
            let _ = ojas_metal::commit_and_wait_checked(cb, "slot decode");
            (cb, None)
        };
        let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        let gs = match &cb_first {
            Some(c) => { let s0: f64 = unsafe { msg_send![*c, GPUStartTime] }; s0.min(gs) }
            None => gs,
        };
        self.gpu_s.set(self.gpu_s.get() + (ge - gs));
        // `st.tmp` is `max(d, vocab)` floats, so B <= MAX_SLOTS ids fit. The ids are
        // laid out by position in `steps`, matching the return order.
        let p = self.st.tmp.contents() as *const u32;
        (0..steps.len()).map(|i| unsafe { *p.add(i) }).collect()
    }

    /// Prefill `tokens` into slot `s` — [`ojas_core::Model::prefill_slot`].
    ///
    /// Slot 0 is `prefill` verbatim, cross-turn reuse and session bookkeeping
    /// included: slot 0 is the sequence every existing caller means.
    ///
    /// A non-zero slot takes a narrower path — the qwen35 chunk graph pointed at
    /// that slot's state, and nothing else. The reuse machinery (`session_tokens`,
    /// the snapshot ladder, the on-disk session cache) is keyed to one resident
    /// sequence: `reuse_prefix` would match slot 2's prompt against slot 0's cached
    /// ids and restore a snapshot belonging to neither. Skipping it costs only a
    /// warm start.
    pub fn prefill_slot(&self, s: usize, tokens: &[u32], base_pos: usize) -> bool {
        if s == 0 { self.prefill(tokens, base_pos); return true; }
        if !self.slots_ok() || s >= self.st.slots || self.cfg.no_prefill { return false; }
        assert!(base_pos.checked_add(tokens.len()).is_some_and(|n| n <= self.st.max_seq)
            && tokens.iter().all(|&t| (t as usize) < self.arch.vocab), "prefill_slot exceeds model bounds");
        if tokens.is_empty() { return true; }
        self.with_slot(s, || {
            if base_pos == 0 { self.reset_state(); }
            let chunk_sz: usize = self.cfg.prefill_m.clamp(1, MAXM);
            let mut pos = base_pos;
            for chunk in tokens.chunks(chunk_sz) {
                self.forward_chunk(chunk, pos, false);
                pos += chunk.len();
            }
        });
        true
    }

    /// [`ojas_core::Model::prefill_embeds_slot`] — precomputed residual rows into one
    /// slot. Same slot-0/slot-s split, and the same reason for it.
    pub fn prefill_embeds_slot(&self, s: usize, tokens: &[u32], x: &[f32], base_pos: usize,
                               pos3: Option<&[[u32; 4]]>) -> bool {
        if s == 0 { return self.prefill_embeds(tokens, x, base_pos, pos3); }
        if !self.slots_ok() || s >= self.st.slots || self.cfg.no_prefill { return false; }
        if let Some(p3) = pos3 {
            assert_eq!(p3.len(), tokens.len(), "pos3 must carry one (t,h,w,e) per row");
            // Sectioned rope lives in `rope_qk_store_m`, which only the chunk graph
            // dispatches, and that is the path here, so the coordinates are honoured.
            // Refuse if the architecture declares no sections, matching
            // `prefill_embeds`: silently dropping them reads as a vision-quality bug.
            if !self.arch.ssm.map_or(false, |c| c.mrope_sections.iter().any(|&v| v != 0)) { return false; }
        }
        if tokens.is_empty() { return true; }
        assert_eq!(x.len(), tokens.len() * self.d,
            "prefill_embeds_slot: x must be tokens.len() * hidden_dim f32 row-major");
        assert!(base_pos.checked_add(tokens.len()).is_some_and(|n| n <= self.st.max_seq)
            && tokens.iter().all(|&t| (t as usize) < self.arch.vocab),
            "prefill_embeds_slot exceeds model bounds");
        let d = self.d;
        self.with_slot(s, || {
            if base_pos == 0 { self.reset_state(); }
            let chunk_sz: usize = self.cfg.prefill_m.clamp(1, MAXM);
            let mut pos = base_pos;
            for (ci, chunk) in tokens.chunks(chunk_sz).enumerate() {
                // st.x is StorageModeShared, so this host store is visible to the
                // command buffer the next line encodes.
                let src = &x[ci * chunk_sz * d..][..chunk.len() * d];
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), self.st.x.contents() as *mut f32, src.len());
                }
                let p3 = pos3.map(|p| &p[ci * chunk_sz..][..chunk.len()]);
                self.forward_chunk_enc_embed(None, chunk, pos, false, false, false, p3);
                pos += chunk.len();
            }
        });
        true
    }

    /// Total transformer layers (for pipeline-stage layer-range assignment).
    pub fn n_layers(&self) -> usize {
        self.arch.n_layers
    }

    /// Whether a loaded NextN/MTP block has a supported execution path.
    ///
    /// This is capability detection. EngineCore separately gates use on greedy
    /// generation, speculation settings, and measured draft/verify benefit.
    /// Qwen hyper-connection models preserve the full residual for nextn.hnorm
    /// and warm the draft KV cache through qwen4exp_mtp_catchup.
    pub fn has_mtp(&self) -> bool {
        self.sp.mtp.is_some() && self.arch.ssm.is_some()
            && (self.arch.moe.is_none() || self.wt.q4 || self.arch.qwen4exp.is_some())
            && (!self.strm.stream || self.strm.ubatch >= 2)
    }

    /// Whether the architecture contains a mixture-of-experts configuration.
    pub fn is_moe(&self) -> bool {
        self.arch.moe.is_some()
    }

}

impl<'a> ojas_core::Model for DecoderGpu<'a> {
    fn context_capacity(&self) -> usize { self.st.max_seq }
    fn mtp_verify_width(&self) -> usize { self.mtp_width() }
    fn n_layers(&self) -> usize { self.arch.n_layers }
    fn hidden_dim(&self) -> usize { self.d }
    fn prefill(&self, tokens: &[u32], base_pos: usize) { DecoderGpu::prefill(self, tokens, base_pos) }
    fn prefill_embeds(&self, tokens: &[u32], x: &[f32], base_pos: usize, pos3: Option<&[[u32; 4]]>) -> bool {
        DecoderGpu::prefill_embeds(self, tokens, x, base_pos, pos3)
    }
    // ---- vision tower ------------------------------------------------------
    // The tower is a method on this object (`decoder/vision.rs` is an
    // `impl DecoderGpu`) because it shares the weight arena, the `projm` precision
    // flag and the state arena with the decoder. These three are how a host that
    // only sees `&dyn Model` — everything above `backend::with_model` — reaches it.
    // All three answer `None` without an mmproj, the same question `has_vision()`
    // asks, so there is one definition of "no tower".
    fn vision_width(&self) -> Option<usize> { DecoderGpu::vision_proj_dim(self) }
    fn vision_tokens(&self, w: usize, h: usize) -> Option<usize> {
        DecoderGpu::n_merged_tokens(self, w, h)
    }
    fn encode_image(&self, img: &[f32], w: usize, h: usize) -> Option<anyhow::Result<Vec<f32>>> {
        // `None` = no tower (fall back to a portable encoder); `Some(Err(_))` = a
        // tower that refused this image. `encode_image` errors without a tower, so
        // the probe has to come first, or a missing mmproj surfaces as a failure
        // rather than as an absence.
        if !self.has_vision() { return None; }
        Some(DecoderGpu::encode_image(self, img, w, h))
    }
    fn reuse_prefix_len(&self, full_prompt: &[u32]) -> usize { self.dense_reuse_start(full_prompt) }
    fn forward_id(&self, token: u32, pos: usize) -> u32 { DecoderGpu::forward_id(self, token, pos) }
    fn reset_session(&self) { DecoderGpu::reset_session(self) }
    fn has_mtp(&self) -> bool { DecoderGpu::has_mtp(self) }
    fn mtp_step_committed(&self, cur: u32, pos: usize) -> Option<Vec<u32>> {
        if !DecoderGpu::has_mtp(self) { return None; }
        Some(self.mtp_generate_step(cur, pos))
    }
    fn forward_batch_topk(&self, tokens: &[u32], base_pos: usize) -> Option<(Vec<u32>, Vec<u32>)> {
        if !self.batched_dense_ok() { return None; }
        if tokens.is_empty() || tokens.len() > MAXM { return None; }
        Some(DecoderGpu::forward_batch_ids_topk(self, tokens, base_pos))
    }

    fn forward_id_topk(&self, token: u32, pos: usize) -> Option<(u32, [u32; 8])> {
        if !self.batched_dense_ok() { return None; }
        // Same command buffer as the forward: the flag makes forward_id encode the
        // top-8 selection right after its argmax, and the ids land in tmp[1..9].
        self.want_topk.set(true);
        let id = DecoderGpu::forward_id(self, token, pos);
        self.want_topk.set(false);
        if id == u32::MAX { return None; }
        let p = self.st.tmp.contents() as *const u32;
        let mut tk = [0u32; 8];
        for (j, t) in tk.iter_mut().enumerate() { *t = unsafe { *p.add(1 + j) }; }
        Some((id, tk))
    }

    fn forward_batch_ids(&self, tokens: &[u32], base_pos: usize) -> Option<Vec<u32>> {
        if !self.batched_dense_ok() { return None; }
        if tokens.is_empty() || tokens.len() > MAXM { return None; }
        Some(DecoderGpu::forward_batch_ids(self, tokens, base_pos))
    }

    fn forward_batch_logits(&self, tokens: &[u32], base_pos: usize) -> Option<Vec<Vec<f32>>> {
        // Recurrent architectures carry state that a rejected draft would have to
        // roll back (that is what spec.rs's mtp_rollback exists for); the plain
        // lookup drafter has no snapshot, so it stays on the dense path only.
        if !self.batched_dense_ok() { return None; }
        if tokens.is_empty() || tokens.len() > MAXM { return None; }
        Some(self.forward_batch_impl(tokens, base_pos, LogitsOut::Host, false))
    }

    fn forward_logits(&self, token: u32, pos: usize) -> Option<Vec<f32>> {
        Some(self.forward(token, pos))
    }

    // ---- sequence slots ----------------------------------------------------
    // `max_slots` reports 1 whenever the slot graph cannot serve this model, not
    // just when `OJAS_SLOTS` is unset: a scheduler branches on it, so answering 4
    // for an architecture `decode_slots` then refuses would have it build batches
    // nothing can run.
    fn max_slots(&self) -> usize { if self.slots_ok() { self.st.slots } else { 1 } }
    fn decode_slots(&self, steps: &[(usize, u32, usize)]) -> Option<Vec<u32>> {
        if !self.slots_ok() || steps.is_empty() || steps.len() > self.st.slots { return None; }
        Some(DecoderGpu::decode_slots(self, steps))
    }
    fn reset_slot(&self, s: usize) { DecoderGpu::reset_slot(self, s) }
    fn prefill_slot(&self, s: usize, tokens: &[u32], base_pos: usize) -> bool {
        DecoderGpu::prefill_slot(self, s, tokens, base_pos)
    }
    fn prefill_embeds_slot(&self, s: usize, tokens: &[u32], x: &[f32], base_pos: usize,
                           pos3: Option<&[[u32; 4]]>) -> bool {
        DecoderGpu::prefill_embeds_slot(self, s, tokens, x, base_pos, pos3)
    }
}
