#![allow(clippy::too_many_arguments)]
use super::*;
use objc::{msg_send, sel, sel_impl};
use metal::MTLSize;
use std::ffi::c_void;
 // re-export

impl<'a> DecoderGpu<'a> {
    pub fn mtp_draft(&self, token: u32, pos: usize, hrow: usize, head: bool) -> u32 {
        if self.arch.qwen4exp.is_some() { return self.mtp_draft_qwen4exp(token, pos, hrow, head, false); }
        let cb = self.gpu.command_buffer();
        let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        self.mtp_draft_encode(&enc, token, pos, hrow, head);
        enc.end_encoding();
        let _ = ojas_metal::commit_and_wait_checked(cb, "speculative draft/verify");
        let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        self.gpu_s.set(self.gpu_s.get() + (ge - gs));
        unsafe { *(self.sp.mtp_tok.contents() as *const u32) }
    }

    /// Encode the draft block into an existing encoder (argmax → tmp[2] so the
    /// verify pass can chain off it GPU-side).
    pub(crate) fn mtp_draft_encode(&self, enc: &metal::ComputeCommandEncoderRef, token: u32, pos: usize, hrow: usize, head: bool) {
        self.mtp_draft_encode_from(enc, token, pos, hrow, head, &self.sp.mtp_h);
    }

    fn mtp_draft_encode_from(&self, enc: &metal::ComputeCommandEncoderRef, token: u32, pos: usize, hrow: usize, head: bool, history: &metal::Buffer) {
        let mc = self.sp.mtp.unwrap();
        let l = mc.layer;
        let p = |s: &str| format!("blk.{l}.{s}");
        let d = self.d as u32;
        let lp = self.arch.layers[l];
        let ib = |enc: &metal::ComputeCommandEncoderRef, idx: u64, v: u32| {
            enc.set_bytes(idx, 4, &v as *const u32 as *const c_void);
        };
        // embed(token) → x  (nextn.embed_tokens if present, else main embeddings)
        let embw = if mc.has_embed { p("nextn.embed_tokens.weight") } else { "token_embd.weight".to_string() };
        self.embed_named_off(enc,&embw,token,d,0);
        self.bar(&enc);
        // combiner: mtp_cat = [rmsnorm(emb)·enorm ‖ rmsnorm(h)·hnorm] → eh_proj → x
        let rms_off = |enc: &metal::ComputeCommandEncoderRef, src: &metal::Buffer, soff: u64, w: &metal::Buffer, dst: &metal::Buffer, doff: u64| {
            enc.set_compute_pipeline_state(&self.p["rmsnorm"]);
            enc.set_buffer(0, Some(src), soff);
            enc.set_buffer(1, Some(w), 0);
            enc.set_buffer(2, Some(dst), doff);
            ib(enc, 3, d);
            enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        };
        rms_off(&enc, &self.st.x, 0, &self.wt.w32[&p("nextn.enorm.weight")], &self.sp.mtp_cat, 0);
        rms_off(&enc, history, (hrow as u64) * (d as u64) * 4, &self.wt.w32[&p("nextn.hnorm.weight")], &self.sp.mtp_cat, (d as u64) * 4);
        self.bar(&enc);
        self.mm(&enc, "plain", &p("nextn.eh_proj.weight"), &self.sp.mtp_cat, &self.st.x, 2*d, d, None);
        self.bar(&enc);
        // gated attention block at position `pos` over the draft block's own KV cache
        let (hd, kvdim, qdim) = (lp.head_dim, lp.kvdim, lp.qdim);
        let group = lp.n_head / lp.n_kv.max(1);
        let sc = self.arch.ssm.unwrap();
        self.enc_reduce(&enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32[&p("attn_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
        self.bar(&enc);
        self.mm(&enc, "plain", &p("attn_q.weight"), &self.st.h, &self.st.ssm_qkv, d, 2*qdim, None);
        self.mm(&enc, "plain", &p("attn_k.weight"), &self.st.h, &self.st.k, d, kvdim, None);
        self.mm(&enc, "plain", &p("attn_v.weight"), &self.st.h, &self.st.v, d, kvdim, None);
        self.bar(&enc);
        self.enc_reduce(&enc, "qgate_split", &[(&self.st.ssm_qkv, 0), (&self.st.q, 1)], &[(2, hd), (3, qdim), (4, 1)], &[], ((qdim+63)/64) as u64, 64);
        self.bar(&enc);
        let (nq, nk) = (lp.n_head, lp.n_kv);
        enc.set_compute_pipeline_state(&self.p["qk_rmsnorm"]);
        enc.set_buffer(0, Some(&self.st.q), 0);
        enc.set_buffer(1, Some(&self.st.k), 0);
        enc.set_buffer(2, Some(&self.wt.w32[&p("attn_q_norm.weight")]), 0);
        enc.set_buffer(3, Some(&self.wt.w32[&p("attn_k_norm.weight")]), 0);
        ib(&enc, 4, hd); ib(&enc, 5, nq); ib(&enc, 6, nk);
        enc.set_bytes(7, 4, &self.arch.eps as *const f32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new((nq + nk) as u64, 1, 1), MTLSize::new(32, 1, 1));
        self.bar(&enc);
        let (totq, totk) = (qdim/2, kvdim/2);
        let off = pos as u32 * kvdim;
        self.enc_reduce(&enc, "rope_qk_store",
            &[(&self.st.q, 0), (&self.st.k, 1), (&self.st.v, 2), (&self.st.kcache[l], 3), (&self.st.vcache[l], 4)],
            &[(5, hd), (6, pos as u32), (8, totq), (9, totk), (10, kvdim), (11, off), (12, 1), (13, sc.n_rot)], &[(7, lp.rope_base)],
            (((totq + totk + kvdim) + 63) / 64) as u64, 64);
        self.bar(&enc);
        let attn_t = self.tune.gemv_plan.get(&(0, 2)).map(|pl| pl.threads).unwrap_or(64).min(256); // streaming kernel: nsg ≤ 8
        if pos + 1 <= 512 {
            self.enc_reduce(&enc, "attention_short",
                &[(&self.st.q, 0), (&self.st.kcache[l], 1), (&self.st.vcache[l], 2), (&self.st.attn, 3)],
                &[(4, hd), (5, kvdim), (6, pos as u32 + 1), (7, group)], &[(8, lp.scale)],
                lp.n_head as u64, attn_t);
        } else {
            self.attn_flash(&enc, lp.n_head, l, l, hd, kvdim, pos as u32 + 1, group, lp.scale);
        }
        self.bar(&enc);
        self.enc_reduce(&enc, "gate_mul_sigmoid", &[(&self.st.attn, 0), (&self.st.ssm_qkv, 1)], &[(2, hd), (3, qdim), (4, 1)], &[], ((qdim+63)/64) as u64, 64);
        self.bar(&enc);
        self.mm(&enc, "accum", &p("attn_output.weight"), &self.st.attn, &self.st.x, qdim, d, None);
        self.bar(&enc);
        // FFN (dense or MoE — mirrors the main layer FFN)
        self.enc_reduce(&enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32[&p("post_attention_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
        self.bar(&enc);
        if let Some(m) = self.arch.moe {
            let (ne, nu, fe, fs) = (m.n_expert, m.n_used, m.ffn_exp, m.ffn_shexp);
            self.check_moe(&p, d, fe);
            self.enc_reduce(&enc, "gemv_w32", &[(&self.st.h, 0), (&self.wt.w32[&p("ffn_gate_inp.weight")], 1), (&self.ms.moe_lg, 2)], &[(3, d), (4, ne)], &[], ((ne + 7)/8) as u64, 256);
            let act0 = 0u32;
            enc.set_compute_pipeline_state(&self.p["ffn_gu_q4"]);
            enc.set_buffer(0, Some(&self.st.h), 0);
            enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_gate_shexp.weight")]), 0);
            enc.set_buffer(2, Some(&self.wt.w4[&p("ffn_up_shexp.weight")]), 0);
            enc.set_buffer(3, Some(&self.st.act), 0);
            ib(&enc, 4, d); ib(&enc, 5, fs);
            enc.set_buffer(6, Some(&self.wt.scale4[&p("ffn_gate_shexp.weight")]), 0);
            enc.set_buffer(7, Some(&self.wt.scale4[&p("ffn_up_shexp.weight")]), 0);
            ib(&enc, 8, act0);
            enc.dispatch_thread_groups(MTLSize::new((fs / 8) as u64, 1, 1), MTLSize::new(64, 1, 1));
            self.enc_reduce(&enc, "gemv_w32", &[(&self.st.h, 0), (&self.wt.w32[&p("ffn_gate_inp_shexp.weight")], 1), (&self.ms.moe_sh, 2)], &[(3, d), (4, 1)], &[], 1, 32);
            self.bar(&enc);
            if let Some(rl) = &self.ms.route_lg {
                enc.set_compute_pipeline_state(&self.p["lg_copy"]);
                enc.set_buffer(0, Some(&self.ms.moe_lg), 0);
                enc.set_buffer(1, Some(rl), 0);
                let roff = (l as usize * ne as usize) as u32;
                enc.set_bytes(2, 4, &roff as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(ne as u64, 1, 1));
            }
            enc.set_compute_pipeline_state(&self.p["moe_topk"]);
            enc.set_buffer(0, Some(&self.ms.moe_lg), 0);
            enc.set_buffer(1, Some(&self.ms.moe_idx), (l as u64) * (MAXM as u64) * (nu as u64) * 4);
            enc.set_buffer(2, Some(&self.ms.moe_wgt), 0);
            ib(&enc, 3, ne); ib(&enc, 4, nu);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
            self.mm(&enc, "plain", &p("ffn_down_shexp.weight"), &self.st.act, &self.st.tmp, fs, d, None);
            self.bar(&enc);
            enc.set_compute_pipeline_state(&self.p["moe_gu_q4"]);
            enc.set_buffer(0, Some(&self.st.h), 0);
            enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_gate_exps.weight")]), 0);
            enc.set_buffer(2, Some(&self.wt.w4[&p("ffn_up_exps.weight")]), 0);
            enc.set_buffer(3, Some(&self.ms.moe_act), 0);
            ib(&enc, 4, d); ib(&enc, 5, fe);
            enc.set_buffer(6, Some(&self.wt.scale4[&p("ffn_gate_exps.weight")]), 0);
            enc.set_buffer(7, Some(&self.wt.scale4[&p("ffn_up_exps.weight")]), 0);
            enc.set_buffer(8, Some(&self.ms.moe_idx), (l as u64) * (MAXM as u64) * (nu as u64) * 4);
            enc.dispatch_thread_groups(MTLSize::new(((fe + 7)/8) as u64, nu as u64, 1), MTLSize::new(64, 1, 1));
            self.bar(&enc);
            enc.set_compute_pipeline_state(&self.p["moe_down_q4"]);
            enc.set_buffer(0, Some(&self.ms.moe_act), 0);
            enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_down_exps.weight")]), 0);
            enc.set_buffer(2, Some(&self.st.x), 0);
            ib(&enc, 3, fe); ib(&enc, 4, d);
            enc.set_buffer(5, Some(&self.wt.scale4[&p("ffn_down_exps.weight")]), 0);
            enc.set_buffer(6, Some(&self.ms.moe_idx), (l as u64) * (MAXM as u64) * (nu as u64) * 4);
            enc.set_buffer(7, Some(&self.ms.moe_wgt), 0);
            ib(&enc, 8, self.arch.moe.unwrap().n_used);
            enc.set_buffer(9, Some(&self.st.tmp), 0);
            enc.set_buffer(10, Some(&self.ms.moe_sh), 0);
            enc.dispatch_thread_groups(MTLSize::new(((d + 7)/8) as u64, 1, 1), MTLSize::new(64, 1, 1));
        } else {
            let nffn = self.arch.ffn as u32; let act0 = 0u32;
            if !self.wt.w4.contains_key(&p("ffn_gate.weight")) || !self.wt.w4.contains_key(&p("ffn_up.weight")) {
                self.mm(enc,"plain",&p("ffn_gate.weight"),&self.st.h,&self.st.gate,d,nffn,None);
                self.mm(enc,"plain",&p("ffn_up.weight"),&self.st.h,&self.st.up,d,nffn,None);
                self.bar(enc);
                self.enc_reduce(enc,"silu_mul",&[(&self.st.gate,0),(&self.st.up,1),(&self.st.act,2)],&[(3,nffn)],&[],((nffn+63)/64) as u64,64);
            } else {
            enc.set_compute_pipeline_state(&self.p["ffn_gu_q4"]);
            enc.set_buffer(0, Some(&self.st.h), 0);
            enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_gate.weight")]), 0);
            enc.set_buffer(2, Some(&self.wt.w4[&p("ffn_up.weight")]), 0);
            enc.set_buffer(3, Some(&self.st.act), 0);
            ib(&enc, 4, d); ib(&enc, 5, nffn);
            enc.set_buffer(6, Some(&self.wt.scale4[&p("ffn_gate.weight")]), 0);
            enc.set_buffer(7, Some(&self.wt.scale4[&p("ffn_up.weight")]), 0);
            ib(&enc, 8, act0);
            enc.dispatch_thread_groups(MTLSize::new((nffn / 8) as u64, 1, 1), MTLSize::new(64, 1, 1));
            }
            self.bar(&enc);
            self.mm(&enc, "accum", &p("ffn_down.weight"), &self.st.act, &self.st.x, nffn, d, None);
        }
        self.bar(&enc);
        // head: shared_head_norm (else output_norm) → shared_head_head (else lm_head)
        // → argmax. Skipped for KV-hole-filling drafts (prediction unused).
        if head {
            let hnw = if mc.has_head_norm { p("nextn.shared_head_norm.weight") } else { "output_norm.weight".to_string() };
            self.enc_reduce(&enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32[&hnw], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
            self.bar(&enc);
            let hw = if mc.has_head { p("nextn.shared_head_head.weight") } else { self.arch.lm_head.clone() };
            self.mm(&enc, "plain", &hw, &self.st.h, &self.st.logits, d, self.arch.vocab as u32, None);
            self.bar(&enc);
            enc.set_compute_pipeline_state(&self.p["argmax"]);
            enc.set_buffer(0, Some(&self.st.logits), 0);
            enc.set_buffer(1, Some(&self.sp.mtp_tok), 0);   // draft token id (own buffer —
            enc.set_bytes(2, 4, &(self.arch.vocab as u32) as *const u32 as *const c_void); // tmp is layer scratch)
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(self.tune.max_tg.min(1024), 1, 1));
        }
    }

    /// Fused MTP step in one command buffer: draft block (argmax → tmp[2]) then the
    /// two-token verify whose second row embeds tmp[2] GPU-side. Returns
    /// (a0, a1, draft): the model's argmax after each position + the draft itself.
    pub fn mtp_step(&self, cur: u32, pos: usize, hrow: usize) -> (u32, u32, u32) {
        // qwen4exp cannot fuse the two: both halves are multi-command-buffer when
        // streamed (each layer breaks for the expert gather), and the draft's token
        // has to reach the CPU before verify can embed it. Run them sequentially.
        if self.arch.qwen4exp.is_some() {
            let draft = self.mtp_draft_qwen4exp(cur, pos, hrow, true, false);
            let (a0, a1) = self.mtp_verify(cur, draft, pos);
            return (a0, a1, draft);
        }
        if !self.wt.q4 {
            let draft=self.mtp_draft(cur,pos,hrow,true);
            let (a0,a1)=self.mtp_verify(cur,draft,pos);
            return (a0,a1,draft);
        }
        let cb = self.gpu.command_buffer();
        let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        self.mtp_draft_encode(&enc, cur, pos, hrow, true);
        self.bar(&enc); // draft token in tmp[2]
        self.forward_chunk_enc(Some(&enc), &[cur, 0], pos, true, true);
        enc.end_encoding();
        let _ = ojas_metal::commit_and_wait_checked(cb, "speculative draft/verify");
        let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        self.gpu_s.set(self.gpu_s.get() + (ge - gs));
        if let (Some(pf), Some(mc)) = (&self.strm.prefetch, self.arch.moe) {
            let nu = mc.n_used as usize;
            let ids = unsafe { std::slice::from_raw_parts(self.ms.moe_idx.contents() as *const u32, self.arch.n_layers * MAXM * nu) };
            let mut used = Vec::with_capacity(self.arch.n_layers * 2 * nu);
            for l in 0..self.arch.n_layers {
                for j in 0..2 * nu {
                    let e = ids[l * MAXM * nu + j];
                    if e < mc.n_expert { used.push((l as u32, e)); }
                }
            }
            pf.note(used);
        }
        let t = unsafe { std::slice::from_raw_parts(self.st.tmp.contents() as *const u32, 2) };
        let draft = unsafe { *(self.sp.mtp_tok.contents() as *const u32) };
        let result = (t[0], t[1], draft);
        self.qwen35_mtp_catchup(&[cur, draft], pos);
        result
    }

    /// MTP speculative verify: run [t0@base_pos, t1@base_pos+1] through the main
    /// stack (weights read once) with state snapshots after t0. Returns the model's
    /// argmax after each position. Feeds the expert prefetcher as a side effect.
    pub fn mtp_verify(&self, t0: u32, t1: u32, base_pos: usize) -> (u32, u32) {
        if self.arch.qwen4exp.is_some() {
            self.forward_chunk_qwen4exp(&[t0, t1], base_pos, true);
            // Same catch-up prefill runs, over the two positions the target just
            // committed. Without it an accepted draft advances two positions while
            // the block ran at one, and the hole it leaves is attended over by every
            // later draft. llama.cpp keeps the equivalent invariant by truncating the
            // draft context at the batch start and re-decoding it.
            self.qwen4exp_mtp_catchup(&[t0, t1], base_pos);
        } else {
            self.forward_chunk(&[t0, t1], base_pos, true);
            self.qwen35_mtp_catchup(&[t0, t1], base_pos);
        }
        if let (Some(pf), Some(mc)) = (&self.strm.prefetch, self.arch.moe) {
            let nu = mc.n_used as usize;
            let ids = unsafe { std::slice::from_raw_parts(self.ms.moe_idx.contents() as *const u32, self.arch.n_layers * MAXM * nu) };
            let mut used = Vec::with_capacity(self.arch.n_layers * 2 * nu);
            for l in 0..self.arch.n_layers {
                for j in 0..2 * nu {
                    let e = ids[l * MAXM * nu + j];
                    if e < mc.n_expert { used.push((l as u32, e)); }
                }
            }
            pf.note(used);
        }
        let t = unsafe { std::slice::from_raw_parts(self.st.tmp.contents() as *const u32, 2) };
        (t[0], t[1])
    }

    /// Rejected draft: restore the SSM/conv states snapshotted after verify token 0.
    /// (KV caches need no rollback — the next verify overwrites the same slots.)
    pub fn mtp_rollback(&self) {
        self.mtp_rollback_to(0);
    }

    /// Restore the state after verified input row `row`. Nonzero rows require
    /// Flash's opt-in per-row snapshots and a completed verification.
    pub fn mtp_rollback_to(&self, row: usize) {
        let trace = self.flash_trace_start();
        assert!(row < self.sp.snapshot_rows && (row == 0 || self.cfg.mtp_prefix && row < self.sp.verified_rows.get()),
            "rollback row has no snapshot");
        let cb = self.gpu.command_buffer();
        let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        let sc = self.arch.ssm.unwrap();
        let state_n = sc.d_state * sc.d_state * sc.dt_rank;
        for l in 0..self.arch.n_layers {
            if !self.arch.layers[l].is_ssm { continue; }
            enc.set_compute_pipeline_state(&self.p["copy_buf"]);
            enc.set_buffer(0, Some(&self.st.ssm_state[l]), 0);
            enc.set_buffer(1, Some(&self.sp.ssm_snap[l]), row as u64 * self.st.ssm_state[l].length());
            enc.set_bytes(2, 4, &state_n as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(state_n.div_ceil(64) as u64, 1, 1), MTLSize::new(64, 1, 1));
            let conv_n = (self.st.conv_state[l].length() / 4) as u32;
            enc.set_buffer(0, Some(&self.st.conv_state[l]), 0);
            enc.set_buffer(1, Some(&self.sp.conv_snap[l]), row as u64 * self.st.conv_state[l].length());
            enc.set_bytes(2, 4, &conv_n as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(conv_n.div_ceil(64) as u64, 1, 1), MTLSize::new(64, 1, 1));
        }
        enc.end_encoding();
        let _ = ojas_metal::commit_and_wait_checked(cb, "speculative draft/verify");
        // Catch-up advanced through rejected rows. Restore its carry to the last
        // committed target hidden; later draft KV positions are causally masked.
        if self.sp.mtp.is_some() {
            let bytes = self.sp.mtp.map(|m| m.hnorm_len * 4).unwrap_or(0);
            unsafe { std::ptr::copy_nonoverlapping((self.sp.mtp_h.contents() as *const u8).add(row * bytes),
                (self.sp.mtp_hprev.contents() as *mut u8).add(MAXM * bytes), bytes); }
        }
        self.sp.hrow.set(row);
        self.flash_trace_finish(trace, "rollback", row, 1, FlashTargetTiming::default());
    }


    /// One self-speculative step, committing between one and `draft_depth`+1 tokens.
    ///
    /// The head is trained to predict one token, so deeper drafts come from chaining
    /// it on its own hidden, as the reference drafter does. Measured here, an M=3
    /// verify costs 1.54x a single forward while carrying up to three tokens.
    ///
    /// By default rejection restores row zero. OJAS_MTP_PREFIX saves each row's
    /// recurrent and PLE state and retains the longest correct draft prefix.
    ///
    /// Which verify row the next draft conditions on, and the rollback a rejected
    /// draft needs, both live here, so callers get tokens rather than a draft/verify
    /// protocol. A wrong draft cannot corrupt output: verify re-runs the real stack
    /// and the draft is kept only when it agrees.
    pub fn mtp_generate_step(&self, cur: u32, pos: usize) -> Vec<u32> {
        let trace = self.flash_trace_start();
        let out = self.mtp_generate_step_inner(cur, pos);
        self.flash_trace_finish(trace, "mtp", pos, out.len(), FlashTargetTiming::default());
        out
    }

    fn mtp_generate_step_inner(&self, cur: u32, pos: usize) -> Vec<u32> {
        assert!(pos < self.st.max_seq && (cur as usize) < self.arch.vocab,
            "MTP input exceeds model bounds");
        let width = self.mtp_width();
        if self.cfg.no_spec || !self.has_mtp() || self.st.max_seq - pos < width {
            self.sp.hrow.set(0);
            return vec![self.forward_id(cur, pos)];
        }
        let n = width - 1;
        if n == 1 || self.arch.qwen4exp.is_none() {
            let (a0, a1, draft) = self.mtp_step(cur, pos, self.sp.hrow.get());
            if draft == a0 { self.sp.hrow.set(1); return vec![a0, a1]; }
            self.mtp_rollback();
            self.sp.hrow.set(0);
            return vec![a0];
        }
        let hrow = self.sp.hrow.get();
        let mut drafts: Vec<u32> = Vec::with_capacity(n);
        let mut tok = cur;
        for i in 0..n {
            let d = self.mtp_draft_qwen4exp(tok, pos + i, hrow, true, i > 0);
            if d == u32::MAX { break; }
            drafts.push(d);
            tok = d;
        }
        if drafts.is_empty() {
            self.sp.hrow.set(0);
            return vec![self.forward_id(cur, pos)];
        }
        let mut batch = Vec::with_capacity(drafts.len() + 1);
        batch.push(cur);
        batch.extend_from_slice(&drafts);
        let got = self.mtp_verify_n(&batch, pos);
        if got.len() == batch.len() && drafts.iter().zip(&got).all(|(d, g)| d == g) {
            self.sp.hrow.set(drafts.len());
            return got;                       // every draft held: commit all of them
        }
        let row = if self.cfg.mtp_prefix {
            drafts.iter().zip(&got).take_while(|(d, g)| d == g).count()
        } else { 0 };
        self.mtp_rollback_to(row);
        got[..=row].to_vec()
    }

    /// Verify M tokens, returning the model's own argmax after each position.
    pub fn mtp_verify_n(&self, tokens: &[u32], base_pos: usize) -> Vec<u32> {
        assert!(!tokens.is_empty() && tokens.len() <= MAXM,
            "MTP verification requires 1..=MAXM tokens");
        if self.arch.qwen4exp.is_some() {
            self.forward_chunk_qwen4exp(tokens, base_pos, true);
            self.qwen4exp_mtp_catchup(tokens, base_pos);
        } else {
            self.forward_chunk(tokens, base_pos, true);
            self.qwen35_mtp_catchup(tokens, base_pos);
        }
        let t = unsafe { std::slice::from_raw_parts(self.st.tmp.contents() as *const u32, tokens.len()) };
        t.to_vec()
    }

    /// Width supported by both the draft chain and the allocated verify scratch.
    pub(crate) fn mtp_width(&self) -> usize {
        if self.arch.qwen4exp.is_none() { return 2; }
        let cap = if self.strm.stream { self.strm.ubatch.min(MAXM) } else { MAXM };
        self.cfg.mtp_draft.max(1).min(cap.saturating_sub(1)).saturating_add(1)
    }

    /// Keep draft context synchronized when serving switches back to scalar
    /// decode (timing gate, sampling, or the end of the allocated context).
    pub(crate) fn sync_mtp_single(&self, token: u32, pos: usize) {
        self.sp.hrow.set(0);
        if self.cfg.no_spec || !self.has_mtp() { return; }
        if self.arch.qwen4exp.is_some() {
            self.qwen4exp_mtp_catchup(&[token], pos);
        } else {
            self.qwen35_mtp_catchup(&[token], pos);
        }
    }

    /// Populate the dense Qwen3.5 draft cache from the target's shifted hiddens.
    pub(crate) fn qwen35_mtp_catchup(&self, tokens: &[u32], base_pos: usize) {
        if self.cfg.no_spec || !self.has_mtp() || self.arch.qwen4exp.is_some() || tokens.is_empty() { return; }
        let row = self.d * 4;
        unsafe {
            let prev = self.sp.mtp_hprev.contents() as *mut u8;
            if base_pos == 0 { std::ptr::write_bytes(prev.add(MAXM * row), 0, row); }
            std::ptr::copy_nonoverlapping(prev.add(MAXM * row), prev, row);
            if tokens.len()>1 { std::ptr::copy_nonoverlapping(self.sp.mtp_h.contents() as *const u8, prev.add(row), (tokens.len()-1)*row); }
            std::ptr::copy_nonoverlapping((self.sp.mtp_h.contents() as *const u8).add((tokens.len()-1)*row), prev.add(MAXM*row), row);
        }
        let cb=self.gpu.command_buffer();
        let enc=cb.new_compute_command_encoder();
        for (i,&token) in tokens.iter().enumerate() {
            self.mtp_draft_encode_from(&enc,token,base_pos+i,i,false,&self.sp.mtp_hprev);
        }
        enc.end_encoding();let _ = ojas_metal::commit_and_wait_checked(cb, "speculative draft/verify");
    }

    /// Time an M-token verify without the speculation bookkeeping around it.
    pub fn verify_cost_probe(&self, tokens: &[u32], base_pos: usize) {
        if self.arch.qwen4exp.is_some() {
            self.forward_chunk_qwen4exp(tokens, base_pos, true);
        } else {
            self.forward_chunk(tokens, base_pos, true);
        }
    }
}
