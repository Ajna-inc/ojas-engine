#![allow(clippy::too_many_arguments)]
use super::*;
use metal::MTLSize;
use std::ffi::c_void;
 // re-export

impl<'a> DecoderGpu<'a> {
    /// deepseek2/glm-dsa MLA forward. Two paths, chosen at load by `m.absorb`:
    ///  * absorb (default): attend over the compressed latent (`mla_lat`) via
    ///    `mla_attn_abs`; cheap KV, context capped at SC_CAP.
    ///  * naive (OJAS_MLA_NAIVE): decompress per-head K/V into a full cache and run
    ///    standard MHA (group=1, hd=k_mla, scale=mscale²/√k_mla) with streaming attn_flash.
    /// Both gather v-dims and apply wo. FFN = dense (leading_dense) or MoE.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_mla(&self, enc: &metal::ComputeCommandEncoderRef, token: u32, pos: usize, seq: u32,
                  l_start: usize, l_end: usize, do_embed: bool, do_head: bool, phase: MoePhase) {
        let d = self.d as u32;
        let m = self.arch.mla.as_ref().unwrap();
        let nh = self.arch.n_head as u32;
        let (kmla, rope) = (m.k_mla, m.qk_rope);
        let nope = kmla - rope;
        let (vmla, kvlora) = (m.v_mla, m.kv_lora);
        let qdim = nh * kmla;
        let kvpair = nope + vmla;      // attn_kv_b per-head output width
        let kvdim = nh * kmla;         // K/V cache head stride = kmla (v zero-padded)
        // ---- embed (stage 0) ----
        if do_embed {
            if self.wt.q4 {
                enc.set_compute_pipeline_state(&self.p["embed_q4"]);
                enc.set_buffer(0, Some(&self.wt.w4["token_embd.weight"]), 0);
                enc.set_buffer(1, Some(&self.st.x), 0);
                enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(3, 4, &token as *const u32 as *const c_void);
                enc.set_buffer(4, Some(&self.wt.scale4["token_embd.weight"]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((d + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
            } else if self.wt.q8 || self.wt.w8.contains_key("token_embd.weight") {
                enc.set_compute_pipeline_state(&self.p["embed_q8"]);
                enc.set_buffer(0, Some(&self.wt.w8["token_embd.weight"]), 0);
                enc.set_buffer(1, Some(&self.st.x), 0);
                enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(3, 4, &token as *const u32 as *const c_void);
                enc.set_buffer(4, Some(&self.wt.scale8["token_embd.weight"]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((d + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
            } else {
                self.enc1d(enc, "embed", &[(&self.wt.w16["token_embd.weight"], 0), (&self.st.x, 1)], &[(2, d), (3, token)], &[], d as u64);
            }
        }
        self.bar(enc);
        for l in l_start..l_end {
            let p = |s: &str| format!("blk.{l}.{s}");
            let lp = self.arch.layers[l];
            // Experts phase (gather path): attention + router already ran in the Route phase;
            // only run this layer's routed experts (over the packed gather scratch).
            if matches!(phase, MoePhase::Experts) {
                self.encode_mla_moe(enc, l, m, d, phase);
                self.bar(enc);
                continue;
            }
            // YaRN rope params (deepseek2 V2-Lite: factor 40, orig ctx 4096, beta 32/1).
            // Interpolate low-freq dims by freq_scale; ramp between corr dims. mscale on
            // cos/sin is folded into kq_scale (lp.scale) so ext-side mscale stays 1.
            // GLM (interleaved): plain rope, no YaRN. deepseek2: YaRN factor 40.
            let il = if m.interleaved { 1u32 } else { 0u32 };
            let ys = if m.interleaved { 1.0f32 } else { 1.0f32 / 40.0 };   // freq_scale
            let yext = if m.interleaved { 0.0f32 } else { 1.0f32 };        // yarn ext_factor
            let nctx = 4096.0f32; let (bf, bs) = (32.0f32, 1.0f32);
            let corr = |nr: f32| (rope as f32) * (nctx / (nr * 2.0 * std::f32::consts::PI)).ln() / (2.0 * lp.rope_base.ln());
            let yclow = corr(bf).floor().max(0.0);
            let ychigh = corr(bs).ceil().min(rope as f32 - 1.0);
            // h = rmsnorm(x, attn_norm)
            self.enc_reduce(enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32[&p("attn_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
            self.bar(enc);
            // q → self.st.q [nh*kmla]: GLM (q_lora>0) uses q_a→rmsnorm(q_a_norm)→q_b; deepseek-lite direct.
            if m.q_lora > 0 {
                let ql = m.q_lora;
                self.mm(enc, "plain", &p("attn_q_a.weight"), &self.st.h, &self.st.k, d, ql, None);       // qa (latent) → self.st.k
                self.bar(enc);
                self.enc_reduce(enc, "rmsnorm", &[(&self.st.k, 0), (&self.wt.w32[&p("attn_q_a_norm.weight")], 1), (&self.st.k, 2)], &[(3, ql)], &[(4, self.arch.eps)], 1, 256);
                self.bar(enc);
                self.mm(enc, "plain", &p("attn_q_b.weight"), &self.st.k, &self.st.q, ql, qdim, None);      // q = q_b(qa)
            } else {
                self.mm(enc, "plain", &p("attn_q.weight"), &self.st.h, &self.st.q, d, qdim, None);
            }
            self.bar(enc);
            // kvc = attn_kv_a_mqa · h → self.st.k [kvlora+rope]  (overwrites the q_a latent, q already in self.st.q)
            self.mm(enc, "plain", &p("attn_kv_a_mqa.weight"), &self.st.h, &self.st.k, d, kvlora + rope, None);
            self.bar(enc);
            // mla_kvnorm: self.st.k → self.st.gate = [rmsnorm(kv_cmpr)[kvlora] | rope(k_pe)[rope]]
            self.enc_reduce(enc, "mla_kvnorm",
                &[(&self.st.k, 0), (&self.wt.w32[&p("attn_kv_a_norm.weight")], 1), (&self.st.gate, 2)],
                &[(3, kvlora), (4, rope), (5, pos as u32), (12, il)],
                &[(6, lp.rope_base), (7, self.arch.eps), (8, ys), (9, yext), (10, yclow), (11, ychigh)], 1, 256);
            // mla_qrope: rope the pe slice of each head in self.st.q (in place; interleaved for GLM)
            enc.set_compute_pipeline_state(&self.p[if m.interleaved { "mla_qrope_il" } else { "mla_qrope" }]);
            enc.set_buffer(0, Some(&self.st.q), 0);
            enc.set_bytes(1, 4, &kmla as *const u32 as *const c_void);
            enc.set_bytes(2, 4, &nope as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &rope as *const u32 as *const c_void);
            let posu = pos as u32;
            enc.set_bytes(4, 4, &posu as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &lp.rope_base as *const f32 as *const c_void);
            enc.set_bytes(6, 4, &ys as *const f32 as *const c_void);
            enc.set_bytes(7, 4, &yext as *const f32 as *const c_void);
            enc.set_bytes(8, 4, &yclow as *const f32 as *const c_void);
            enc.set_bytes(9, 4, &ychigh as *const f32 as *const c_void);
            if m.interleaved {
                // mla_qrope_il's mscale factor (buffer 10) must be bound: an unset constant
                // buffer reads stale encoder state, which zeroed q_pe (invisible at pos 0,
                // where softmax over one score is weight-1 regardless). GLM: no YaRN → 1.0.
                let mrope = 1.0f32;
                enc.set_bytes(10, 4, &mrope as *const f32 as *const c_void);
            }
            enc.dispatch_thread_groups(MTLSize::new(nh as u64, 1, 1), MTLSize::new(64, 1, 1));
            self.bar(enc);
            let absorb = m.absorb;  // load-time decision (see MlaConfig::absorb); naive when seq can exceed SC_CAP
            let hdk = kvlora + rope;
            if absorb {
                // store latent [Lc|Rc] = gate[0..hdk] → mla_lat[l] at pos (f32→half)
                enc.set_compute_pipeline_state(&self.p["copy_f32_half"]);
                enc.set_buffer(0, Some(&self.st.gate), 0);
                enc.set_buffer(1, Some(&self.st.mla_lat[l]), (pos as u64) * (hdk as u64) * 2);
                enc.set_bytes(2, 4, &hdk as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((hdk + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
                self.bar(enc);
                // qabs = q_nope·kv_b_nope, append q_pe → self.st.up [nh*hdk]
                enc.set_compute_pipeline_state(&self.p["mla_qabsorb"]);
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.wt.w16[&p("attn_kv_b.weight")]), 0);
                enc.set_buffer(2, Some(&self.st.up), 0);
                enc.set_bytes(3, 4, &kmla as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &nope as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &rope as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &kvlora as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &kvpair as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((hdk + 63) / 64) as u64, nh as u64, 1), MTLSize::new(64, 1, 1));
                self.bar(enc);
                // MQA attention over the shared latent → clat self.st.act [nh*kvlora]
                enc.set_compute_pipeline_state(&self.p["mla_attn_abs"]);
                enc.set_buffer(0, Some(&self.st.up), 0);
                enc.set_buffer(1, Some(&self.st.mla_lat[l]), 0);
                enc.set_buffer(2, Some(&self.st.act), 0);
                enc.set_bytes(3, 4, &hdk as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &kvlora as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &hdk as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &seq as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &lp.scale as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, 1, 1), MTLSize::new(256, 1, 1));
                self.bar(enc);
                // ctx = clat·kv_b_v → self.st.v [nh*vmla]
                enc.set_compute_pipeline_state(&self.p["mla_ctx"]);
                enc.set_buffer(0, Some(&self.st.act), 0);
                enc.set_buffer(1, Some(&self.wt.w16[&p("attn_kv_b.weight")]), 0);
                enc.set_buffer(2, Some(&self.st.v), 0);
                enc.set_bytes(3, 4, &kvlora as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &vmla as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &nope as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &kvpair as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((vmla + 63) / 64) as u64, nh as u64, 1), MTLSize::new(64, 1, 1));
                self.bar(enc);
            } else {
            // kv = attn_kv_b · gate[0..kvlora] → self.st.up [nh*(nope+vmla)]
            self.mm(enc, "plain", &p("attn_kv_b.weight"), &self.st.gate, &self.st.up, kvlora, nh * kvpair, None);
            self.bar(enc);
            // assemble K=[k_nope|k_pe] / V=[v|0pad] into the cache at pos
            enc.set_compute_pipeline_state(&self.p["mla_kvwrite"]);
            enc.set_buffer(0, Some(&self.st.up), 0);
            enc.set_buffer(1, Some(&self.st.gate), (kvlora as u64) * 4); // k_pe lives at gate[kvlora..]
            enc.set_buffer(2, Some(&self.st.kcache[l]), 0);
            enc.set_buffer(3, Some(&self.st.vcache[l]), 0);
            enc.set_bytes(4, 4, &nh as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &nope as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &vmla as *const u32 as *const c_void);
            enc.set_bytes(7, 4, &kmla as *const u32 as *const c_void);
            enc.set_bytes(8, 4, &posu as *const u32 as *const c_void);
            let ktot = nh * kmla;
            enc.dispatch_thread_groups(MTLSize::new(((ktot + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
            self.bar(enc);
            // MHA over the reconstructed cache (group=1, hd=kmla)
            if seq <= 512 {
                self.enc_reduce(enc, "attention_short",
                    &[(&self.st.q, 0), (&self.st.kcache[l], 1), (&self.st.vcache[l], 2), (&self.st.attn, 3)],
                    &[(4, kmla), (5, kvdim), (6, seq), (7, 1)], &[(8, lp.scale)], nh as u64, 256);
            } else {
                self.attn_flash(enc, nh, l, l, kmla, kvdim, seq, 1, lp.scale);
            }
            self.bar(enc);
            // gather v-dims: attn[nh*kmla] → self.st.v [nh*vmla]
            enc.set_compute_pipeline_state(&self.p["mla_gather"]);
            enc.set_buffer(0, Some(&self.st.attn), 0);
            enc.set_buffer(1, Some(&self.st.v), 0);
            enc.set_bytes(2, 4, &nh as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &vmla as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &kmla as *const u32 as *const c_void);
            let vtot = nh * vmla;
            enc.dispatch_thread_groups(MTLSize::new(((vtot + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
            self.bar(enc);
            }
            // x += attn_output · v
            self.mm(enc, "accum", &p("attn_output.weight"), &self.st.v, &self.st.x, nh * vmla, d, None);
            self.bar(enc);
            // FFN pre-norm
            self.enc_reduce(enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32[&p("ffn_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
            self.bar(enc);
            if (l as u32) < m.leading_dense {
                // Dense SwiGLU FFN via mm+silu_mul (precision-agnostic: q4/q8/f16).
                let nffn = self.arch.ffn as u32;
                self.mm(enc, "plain", &p("ffn_gate.weight"), &self.st.h, &self.st.gate, d, nffn, None);
                self.mm(enc, "plain", &p("ffn_up.weight"), &self.st.h, &self.st.up, d, nffn, None);
                self.bar(enc);
                self.enc_reduce(enc, "silu_mul", &[(&self.st.gate, 0), (&self.st.up, 1), (&self.st.act, 2)], &[(3, nffn)], &[], ((nffn + 63) / 64) as u64, 64);
                self.bar(enc);
                self.mm(enc, "accum", &p("ffn_down.weight"), &self.st.act, &self.st.x, nffn, d, None);
            } else {
                self.encode_mla_moe(enc, l, m, d, phase);
            }
            self.bar(enc);
        }
        if !do_head { return; }
        self.enc_reduce(enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32["output_norm.weight"], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
        self.bar(enc);
        let lm = self.arch.lm_head.clone();
        self.mm(enc, "plain", &lm, &self.st.h, &self.st.logits, d, self.arch.vocab as u32, None);
    }

    /// deepseek2 MoE FFN: softmax router → top-k routed experts + always-on shared
    /// expert(s). Reuses the qwen35moe batched-M=1 kernels; the shared expert has no
    /// per-token gate here (DeepSeek adds it directly), so moe_sh is pinned so
    /// sigmoid(moe_sh)~=1 in moe_down_q4.
    pub(crate) fn encode_mla_moe(&self, enc: &metal::ComputeCommandEncoderRef, l: usize, m: &MlaConfig, d: u32, phase: MoePhase) {
        let p = |s: &str| format!("blk.{l}.{s}");
        let (ne, nu, fe) = (m.n_expert, m.n_used, m.ffn_exp);
        self.check_moe(&p, d, fe);
        let fs = m.ffn_exp * m.n_shared;
        let off = (l as u64) * (MAXM as u64) * (nu as u64) * 4;   // this layer's moe_idx slot
        // router + shared expert: only in Full/Route (in the Experts phase they already ran,
        // and moe_idx / tmp / moe_sh persist across the split command buffers).
        if !matches!(phase, MoePhase::Experts) {
        // router logits (ffn_gate_inp is f32)
        self.enc_reduce(enc, "gemv_w32", &[(&self.st.h, 0), (&self.wt.w32[&p("ffn_gate_inp.weight")], 1), (&self.ms.moe_lg, 2)], &[(3, d), (4, ne)], &[], ((ne + 7) / 8) as u64, 256);
        // routing telemetry: snapshot this layer's logits (same read-ordering as topk)
        if let Some(rl) = &self.ms.route_lg {
            enc.set_compute_pipeline_state(&self.p["lg_copy"]);
            enc.set_buffer(0, Some(&self.ms.moe_lg), 0);
            enc.set_buffer(1, Some(rl), 0);
            let roff = (l * ne as usize) as u32;
            enc.set_bytes(2, 4, &roff as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(ne as u64, 1, 1));
        }
        // router top-k: GLM/DeepSeek-V3 sigmoid+bias (moe_topk_v3) or deepseek2 softmax-no-renorm.
        if m.sigmoid_router {
            let norm = 1u32; let rscale = m.routed_scale;
            enc.set_compute_pipeline_state(&self.p["moe_topk_v3"]);
            enc.set_buffer(0, Some(&self.ms.moe_lg), 0);
            enc.set_buffer(1, Some(&self.wt.w32[&p("exp_probs_b.bias")]), 0);
            enc.set_buffer(2, Some(&self.ms.moe_idx), off);
            enc.set_buffer(3, Some(&self.ms.moe_wgt), 0);
            enc.set_bytes(4, 4, &ne as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &nu as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &rscale as *const f32 as *const c_void);
            enc.set_bytes(7, 4, &norm as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
        } else {
            enc.set_compute_pipeline_state(&self.p["moe_topk_nonorm"]);
            enc.set_buffer(0, Some(&self.ms.moe_lg), 0);
            enc.set_buffer(1, Some(&self.ms.moe_idx), off);
            enc.set_buffer(2, Some(&self.ms.moe_wgt), 0);
            enc.set_bytes(3, 4, &ne as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
        }
        // shared-expert SwiGLU via mm+silu_mul (precision-agnostic) → tmp (down output)
        self.mm(enc, "plain", &p("ffn_gate_shexp.weight"), &self.st.h, &self.st.gate, d, fs, None);
        self.mm(enc, "plain", &p("ffn_up_shexp.weight"), &self.st.h, &self.st.up, d, fs, None);
        self.bar(enc);
        self.enc_reduce(enc, "silu_mul", &[(&self.st.gate, 0), (&self.st.up, 1), (&self.st.act, 2)], &[(3, fs)], &[], ((fs + 63) / 64) as u64, 64);
        self.bar(enc);
        self.mm(enc, "plain", &p("ffn_down_shexp.weight"), &self.st.act, &self.st.tmp, fs, d, None);
        // pin moe_sh so sigmoid(moe_sh) ~= 1 (shared expert added with weight 1)
        enc.set_compute_pipeline_state(&self.p["set_const"]);
        enc.set_buffer(0, Some(&self.ms.moe_sh), 0);
        let big = 30.0f32;
        enc.set_bytes(1, 4, &big as *const f32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        self.bar(enc);
        if matches!(phase, MoePhase::Route) { return; }
        }
        // routed experts SwiGLU → moe_act, then weighted down + shared accumulate into x.
        // Q4_K (native gate/up + Q8 down) / Q8 / Q4.
        if self.wt.q4k {
            // gate/up + idx: streaming uses the packed gather scratch (moe_gs/us/ds, routed
            // experts in slots 0..nu-1) with the identity index, so Metal wires only the small
            // scratch, not all 256 experts; else the resident cache tensors indexed by moe_idx.
            let (gb, ub, idxb, idx_off) = if self.strm.stream {
                (&self.strm.moe_gs, &self.strm.moe_us, &self.strm.moe_slot, 0u64)
            } else {
                (&self.wt.w4k[&p("ffn_gate_exps.weight")], &self.wt.w4k[&p("ffn_up_exps.weight")], &self.ms.moe_idx, off)
            };
            let dn = p("ffn_down_exps.weight");
            // Per-layer gate/up type picks the kernel (UD-IQ2 GLM: mostly IQ2_XXS,
            // blk.8 IQ2_S). `w4k` holds Q4_K only and requantized tensors carry no
            // `w_qtype`, so an absent entry means Q4_K. The loader validated every
            // expert tensor against this same table before upload, so a miss here
            // panics instead of falling back: reading the weights with another
            // format's walker corrupts them silently.
            let gname = p("ffn_gate_exps.weight");
            let gty = self.wt.w_qtype.get(&gname).copied().unwrap_or(12);
            let gk = ojas_core::quant_src::moe_kernel(gty, ojas_core::quant_src::MoeRole::GateUp)
                .unwrap_or_else(|| panic!("no MoE gate/up kernel for GGUF type {gty} ({gname})"));
            enc.set_compute_pipeline_state(&self.p[gk.entry]);
            enc.set_buffer(0, Some(&self.st.h), 0);
            enc.set_buffer(1, Some(gb), 0);
            enc.set_buffer(2, Some(ub), 0);
            enc.set_buffer(3, Some(&self.ms.moe_act), 0);
            enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &fe as *const u32 as *const c_void);
            enc.set_buffer(8, Some(idxb), idx_off);
            let (gt, gr) = gk.launch;
            enc.dispatch_thread_groups(MTLSize::new(((fe + gr - 1) / gr) as u64, nu as u64, 1), MTLSize::new(gt as u64, 1, 1));
            self.bar(enc);
            if self.strm.stream {
                // packed gather scratch down + identity idx; per-layer type picks the kernel.
                let dty = self.wt.w_qtype.get(&dn).copied().unwrap_or(8);
                let dk = ojas_core::quant_src::moe_kernel(dty, ojas_core::quant_src::MoeRole::Down)
                    .unwrap_or_else(|| panic!("no MoE down kernel for GGUF type {dty} ({dn})"));
                enc.set_compute_pipeline_state(&self.p[dk.entry]);
                enc.set_buffer(0, Some(&self.ms.moe_act), 0);
                enc.set_buffer(1, Some(&self.strm.moe_ds), 0);
                enc.set_buffer(2, Some(&self.st.x), 0);
                enc.set_bytes(3, 4, &fe as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                enc.set_buffer(6, Some(&self.strm.moe_slot), 0);
                enc.set_buffer(7, Some(&self.ms.moe_wgt), 0);
                enc.set_bytes(8, 4, &nu as *const u32 as *const c_void);
                enc.set_buffer(9, Some(&self.st.tmp), 0);
                enc.set_buffer(10, Some(&self.ms.moe_sh), 0);
                // Geometry comes from the table, not from the kernel name: matching on a
                // name prefix silently mis-dispatches any kernel whose row count does not
                // match that prefix.
                let (dt, dr) = dk.launch;
                enc.dispatch_thread_groups(MTLSize::new(((d + dr - 1) / dr) as u64, 1, 1), MTLSize::new(dt as u64, 1, 1));
            } else {
                enc.set_compute_pipeline_state(&self.p["moe_down_q8"]);
                enc.set_buffer(0, Some(&self.ms.moe_act), 0);
                enc.set_buffer(1, Some(&self.wt.w8[&dn]), 0);
                enc.set_buffer(2, Some(&self.st.x), 0);
                enc.set_bytes(3, 4, &fe as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                enc.set_buffer(5, Some(&self.wt.scale8[&dn]), 0);
                enc.set_buffer(6, Some(&self.ms.moe_idx), off);
                enc.set_buffer(7, Some(&self.ms.moe_wgt), 0);
                enc.set_bytes(8, 4, &nu as *const u32 as *const c_void);
                enc.set_buffer(9, Some(&self.st.tmp), 0);
                enc.set_buffer(10, Some(&self.ms.moe_sh), 0);
                enc.dispatch_thread_groups(MTLSize::new(((d + 7) / 8) as u64, 1, 1), MTLSize::new(64, 1, 1));
            }
        } else if self.wt.q8 {
            enc.set_compute_pipeline_state(&self.p["moe_gu_q8"]);
            enc.set_buffer(0, Some(&self.st.h), 0);
            enc.set_buffer(1, Some(&self.wt.w8[&p("ffn_gate_exps.weight")]), 0);
            enc.set_buffer(2, Some(&self.wt.w8[&p("ffn_up_exps.weight")]), 0);
            enc.set_buffer(3, Some(&self.ms.moe_act), 0);
            enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &fe as *const u32 as *const c_void);
            enc.set_buffer(6, Some(&self.wt.scale8[&p("ffn_gate_exps.weight")]), 0);
            enc.set_buffer(7, Some(&self.wt.scale8[&p("ffn_up_exps.weight")]), 0);
            enc.set_buffer(8, Some(&self.ms.moe_idx), off);
            enc.dispatch_thread_groups(MTLSize::new(((fe + 7) / 8) as u64, nu as u64, 1), MTLSize::new(64, 1, 1));
            self.bar(enc);
            enc.set_compute_pipeline_state(&self.p["moe_down_q8"]);
            enc.set_buffer(0, Some(&self.ms.moe_act), 0);
            enc.set_buffer(1, Some(&self.wt.w8[&p("ffn_down_exps.weight")]), 0);
            enc.set_buffer(2, Some(&self.st.x), 0);
            enc.set_bytes(3, 4, &fe as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
            enc.set_buffer(5, Some(&self.wt.scale8[&p("ffn_down_exps.weight")]), 0);
            enc.set_buffer(6, Some(&self.ms.moe_idx), off);
            enc.set_buffer(7, Some(&self.ms.moe_wgt), 0);
            enc.set_bytes(8, 4, &nu as *const u32 as *const c_void);
            enc.set_buffer(9, Some(&self.st.tmp), 0);
            enc.set_buffer(10, Some(&self.ms.moe_sh), 0);
            enc.dispatch_thread_groups(MTLSize::new(((d + 7) / 8) as u64, 1, 1), MTLSize::new(64, 1, 1));
        } else {
            enc.set_compute_pipeline_state(&self.p["moe_gu_q4"]);
            enc.set_buffer(0, Some(&self.st.h), 0);
            enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_gate_exps.weight")]), 0);
            enc.set_buffer(2, Some(&self.wt.w4[&p("ffn_up_exps.weight")]), 0);
            enc.set_buffer(3, Some(&self.ms.moe_act), 0);
            enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &fe as *const u32 as *const c_void);
            enc.set_buffer(6, Some(&self.wt.scale4[&p("ffn_gate_exps.weight")]), 0);
            enc.set_buffer(7, Some(&self.wt.scale4[&p("ffn_up_exps.weight")]), 0);
            enc.set_buffer(8, Some(&self.ms.moe_idx), off);
            enc.dispatch_thread_groups(MTLSize::new(((fe + 7) / 8) as u64, nu as u64, 1), MTLSize::new(64, 1, 1));
            self.bar(enc);
            enc.set_compute_pipeline_state(&self.p["moe_down_q4"]);
            enc.set_buffer(0, Some(&self.ms.moe_act), 0);
            enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_down_exps.weight")]), 0);
            enc.set_buffer(2, Some(&self.st.x), 0);
            enc.set_bytes(3, 4, &fe as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
            enc.set_buffer(5, Some(&self.wt.scale4[&p("ffn_down_exps.weight")]), 0);
            enc.set_buffer(6, Some(&self.ms.moe_idx), off);
            enc.set_buffer(7, Some(&self.ms.moe_wgt), 0);
            enc.set_bytes(8, 4, &nu as *const u32 as *const c_void);
            enc.set_buffer(9, Some(&self.st.tmp), 0);
            enc.set_buffer(10, Some(&self.ms.moe_sh), 0);
            enc.dispatch_thread_groups(MTLSize::new(((d + 7) / 8) as u64, 1, 1), MTLSize::new(64, 1, 1));
        }
    }
}
