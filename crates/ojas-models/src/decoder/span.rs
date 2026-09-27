#![allow(clippy::too_many_arguments)]
use super::*;
 // re-export

impl<'a> DecoderGpu<'a> {
    /// Read the current residual-stream hidden state (self.st.x, `d` floats) to host —
    /// the activation a pipeline stage forwards to the next worker.
    pub fn read_hidden(&self) -> Vec<f32> {
        let ptr = self.st.x.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.d) }.to_vec()
    }

    /// Write an incoming hidden state into self.st.x — the activation received from
    /// the previous pipeline stage (host copy is visible to the next command buffer).
    pub fn write_hidden(&self, h: &[f32]) {
        let n = h.len().min(self.d);
        unsafe { std::ptr::copy_nonoverlapping(h.as_ptr(), self.st.x.contents() as *mut f32, n); }
    }

    /// Run one pipeline stage: layers [l_start, l_end) at position `pos`.
    /// `do_embed` embeds `token` into self.st.x (stage 0); otherwise self.st.x must
    /// already hold the incoming activation (see write_hidden). `do_head` runs
    /// output_norm + lm_head and returns the argmax token id (last stage);
    /// middle stages return 0 and leave the hidden state in self.st.x (read_hidden).
    pub fn forward_span(&self, token: u32, pos: usize, l_start: usize, l_end: usize,
                        do_embed: bool, do_head: bool) -> u32 {
        let d = self.d as u32;
        let hd = self.arch.hd as u32;
        let kvdim = (self.arch.n_kv * self.arch.hd) as u32;
        let group = (self.arch.n_head / self.arch.n_kv) as u32;
        let scale = 1.0 / (self.arch.hd as f32).sqrt();
        let seq = (pos + 1) as u32;
        let cb = self.gpu.command_buffer();
        let enc = if self.arch.ssm.is_some() && !self.cfg.serial {
            cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent)
        } else { cb.new_compute_command_encoder() };
        self.encode_forward_span(&enc, token, pos, d, hd, kvdim, group, scale, seq,
                                 l_start, l_end, do_embed, do_head);
        let mut out = 0u32;
        if do_head {
            self.bar(&enc);
            self.enc_reduce(&enc, "argmax", &[(&self.st.logits, 0), (&self.st.tmp, 1)], &[(2, self.arch.vocab as u32)], &[], 1, self.tune.max_tg.min(1024));
        }
        enc.end_encoding();
        let _ = ojas_metal::commit_and_wait_checked(cb, "pipeline span");
        if do_head {
            out = unsafe { *(self.st.tmp.contents() as *const u32) };
        }
        out
    }

    /// Zero the recurrent state (SSM conv window + matrix state).
    ///
    /// KV attention is positional and self-heals: a new sequence overwrites entry
    /// `pos` as it advances. The SSM state does not — it is a running accumulation,
    /// so a fresh sequence started without clearing it carries the previous
    /// sequence's tail into the new one. On a hybrid like qwen35 (24 of 32 layers
    /// recurrent) the output is coherent but subtly wrong, and worse the longer the
    /// previous sequence was.
    ///
    /// Scoped to one slot: whichever the single-sequence paths are pointed at (slot 0
    /// unless a `with_slot` is in scope). Slots hold unrelated sequences, so this must
    /// not touch the others; [`DecoderGpu::reset_slot`] names one directly.
    pub fn reset_state(&self) {
        for l in 0..self.st.conv_state.len() {
            for (ptr, len) in [self.conv_region(l), self.ssm_region(l)] {
                if len > 0 { unsafe { std::ptr::write_bytes(ptr, 0, len) }; }
            }
        }
    }

    /// Zero slot `s`'s recurrent state — [`ojas_core::Model::reset_slot`].
    ///
    /// KV is left alone deliberately: it is addressed by position, so a sequence
    /// restarting in this slot overwrites row `pos` as it advances and never reads a
    /// row it has not written. The recurrent state is the half that does not
    /// self-heal — it is a running accumulation, so a slot reused without this
    /// answers the next request from the middle of the previous one.
    pub fn reset_slot(&self, s: usize) {
        assert!(s < self.st.slots, "slot {s} exceeds the {} allocated", self.st.slots);
        self.with_slot(s, || self.reset_state());
    }

    /// Start a fresh sequence: zero the SSM state and drop the dense cross-turn-reuse
    /// bookkeeping (token log + prefilled high-water mark), so the next prompt
    /// re-prefills from position 0 with no reuse of the previous conversation.
    /// `reset_state` alone only clears SSM; dense KV is positional and self-heals
    /// under a full re-prefill, but the reuse LCP must not see stale tokens. Use this
    /// when switching conversations, not between turns of one.
    pub fn reset_session(&self) {
        self.reset_state();
        self.sp.hrow.set(0);
        self.sp.verified_rows.set(0);
        if self.sp.mtp.is_some() {
            for b in [&self.sp.mtp_h, &self.sp.mtp_hprev] {
                unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, b.length() as usize) };
            }
        }
        self.reset_dense_reuse();
        self.sess.snap_pos.borrow_mut().clear();
        self.sess.snap_buf.borrow_mut().clear();
        self.sess.last_prefill_reused.set(0);
    }

    /// Read back the logits buffer (valid after a do_head forward/span).
    pub fn logits_vec(&self) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(self.st.logits.contents() as *const f32, self.arch.vocab) }.to_vec()
    }

}
