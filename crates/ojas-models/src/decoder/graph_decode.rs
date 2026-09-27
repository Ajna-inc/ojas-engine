#![allow(clippy::too_many_arguments)]
use super::*;
use metal::MTLSize;
use std::ffi::c_void;
 // re-export

/// Route a 3- or 4-slot plain-Q4 projection through `ceil(b/2)` dispatches of the
/// tuned two-row `gemv_m2_q4` instead of one dispatch of the generic `gemv_m_q4`.
///
/// Off by default: it is not reproducible. Three identical `ojas ocr` runs over the
/// same four pages at B=4 gave three different transcriptions (27319 / 26478 / 26542
/// bytes); with it off, three runs are byte-identical. B=3 varies too; B=2 never
/// enters this path and is reproducible either way. Each run reproduces the whole
/// sequential transcription before diverging, so the divergence is in the stop
/// decision, not the text — suspects are the phantom trailing row the pair tiling
/// writes when `b` is odd, and a partial-row reduction read before it is written.
/// `OJAS_SLOT_Q4PAIR=1` re-enables it for debugging; fixing it is worth ~1.9x at B=4
/// over the generic dispatch. With it off, B=4 falls back to the generic `gemv_m_q4`
/// and B=2 is the fastest reproducible configuration, which is why
/// `ocr::DEFAULT_SLOTS` is 2.
///
/// It addresses the B=2 -> B=3 throughput cliff: plain Q4 (`wt.w4`) has exactly three
/// M-row kernels — `gemv_m2_q4` (M==2 only), `gemm_mm_q4` (M>=8), and the generic
/// `gemv_m_q4` — so B=3 and B=4 fall off the tuned one onto the generic one and lose
/// to single-sequence decode. The tuned 4-row family measured at 1.84x/M=4
/// (`gemv_m4_q4l`, `gemv_m8_q4l`, `gemv_x{4,8}_{1..8}_q4l`) exists only for Q4L, which
/// surya-2 never reaches: `w4l` is populated only from GGUF type 12 (Q4_K) source
/// tensors (`load.rs`), and surya-2 ships F16, so its weights are requantized into
/// `w4` (`repr_gate` reports 188 Q4, zero Q4L).
///
/// Pairing trades half the weight-read amortization for the tuned kernel — two reads
/// for four rows instead of one — and still wins, because the per-step fixed costs
/// (command-buffer submit, barrier serialization, ~0.8 ms of CPU encode) are paid
/// once per step regardless of B, so four tokens per step halve them against two.
/// Measured on surya-2/prec 2, three interleaved process-level rounds (same sign in
/// every round), per-arm minimum, speedup against the same process's own B=1:
///
/// ```text
///   B    generic          paired
///   3    0.59-1.03x       1.45-1.62x
///   4    0.68-1.09x       1.66-2.06x
/// ```
///
/// A `gemv_m4_q4` kernel mirroring `gemv_m4_q4l` — one weight read for four rows —
/// would lift B=4 toward the 1.84x on the read term; that is an `ojas-metal` addition.
fn q4_pair() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OJAS_SLOT_Q4PAIR").as_deref() == Ok("1"))
}

fn attn_force_nwg() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("OJAS_ATTN_NWG").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

impl<'a> DecoderGpu<'a> {
    pub(crate) fn encode_forward(&self, enc: &metal::ComputeCommandEncoderRef, token: u32, pos: usize,
                      d: u32, hd: u32, kvdim: u32, group: u32, scale: f32, seq: u32) {
        self.encode_forward_span(enc, token, pos, d, hd, kvdim, group, scale, seq,
                                 0, self.arch.n_layers, true, true);
    }

    /// One GPT-OSS decoder layer: GQA + per-head attention sinks + biased SwiGLU-OAI
    /// MoE. `x` is the residual stream. Short-context path (YaRN≈1), following the
    /// LLM_ARCH_OPENAI_MOE reference. Not yet validated on hardware.
    pub(crate) fn encode_gptoss_layer(&self, enc: &metal::ComputeCommandEncoderRef, l: usize, pos: usize, seq: u32) {
        let p = |s: &str| format!("blk.{l}.{s}");
        let lp = self.arch.layers[l];
        let d = self.d as u32; let hd = self.arch.hd as u32;
        let (n_head, n_kv) = (lp.n_head, lp.n_kv);
        let qdim = n_head * hd; let kvdim = n_kv * hd; let group = n_head / n_kv;
        let m = self.arch.moe.unwrap();
        let (ne, nu, fe) = (m.n_expert, m.n_used, m.ffn_exp);
        self.check_moe(&p, d, fe);
        let off = (l as u64) * (MAXM as u64) * (nu as u64) * 4;   // this layer's moe_idx slot
        let (alpha, limit) = (1.702f32, 7.0f32);
        // --- attention (pre-norm) ---
        self.enc_reduce(enc, "rmsnorm", &[(&self.st.x,0),(&self.wt.w32[&p("attn_norm.weight")],1),(&self.st.h,2)], &[(3,d)], &[(4,self.arch.eps)], 1, 256);
        self.bar(enc);
        // biased q/k/v projections — kind="bias" is required; the q8 "plain" kernel
        // ignores the bias buffer.
        self.mm(enc, "bias", &p("attn_q.weight"), &self.st.h, &self.st.q, d, qdim, Some(&self.wt.w32[&p("attn_q.bias")]));
        self.mm(enc, "bias", &p("attn_k.weight"), &self.st.h, &self.st.k, d, kvdim, Some(&self.wt.w32[&p("attn_k.bias")]));
        self.mm(enc, "bias", &p("attn_v.weight"), &self.st.h, &self.st.v, d, kvdim, Some(&self.wt.w32[&p("attn_v.bias")]));
        self.bar(enc);
        // full-head NEOX rope (n_rot=hd) + store k,v to cache
        let (totq, totk) = (qdim/2, kvdim/2); let off_kv = pos as u32 * kvdim;
        self.enc_reduce(enc, "rope_qk_store",
            &[(&self.st.q,0),(&self.st.k,1),(&self.st.v,2),(&self.st.kcache[l],3),(&self.st.vcache[l],4)],
            &[(5,hd),(6,pos as u32),(8,totq),(9,totk),(10,kvdim),(11,off_kv),(12,1),(13,hd)], &[(7,lp.rope_base)],
            (((totq+totk+kvdim)+63)/64) as u64, 64);
        self.bar(enc);
        // GQA attention with per-head sinks (short-context path; buffer 9 = sinks[n_head])
        let attn_t = self.tune.gemv_plan.get(&(0,2)).map(|pp| pp.threads).unwrap_or(64).min(256);
        // gpt-oss alternates sliding-window (n_swa=128) on even layers, full on odd.
        let win: u32 = if l % 2 == 0 { 128 } else { 0 };
        self.enc_reduce(enc, "attention_short_sink",
            &[(&self.st.q,0),(&self.st.kcache[l],1),(&self.st.vcache[l],2),(&self.st.attn,3),(&self.wt.w32[&p("attn_sinks.weight")],9)],
            &[(4,hd),(5,kvdim),(6,seq),(7,group),(10,win)], &[(8,lp.scale)], n_head as u64, attn_t);
        self.bar(enc);
        // biased output proj → gate scratch, then x += (accum kernels lack bias)
        self.mm(enc, "bias", &p("attn_output.weight"), &self.st.attn, &self.st.gate, qdim, d, Some(&self.wt.w32[&p("attn_output.bias")]));
        self.bar(enc);
        self.enc_reduce(enc, "add_inplace", &[(&self.st.x,0),(&self.st.gate,1)], &[(2,d)], &[], ((d+63)/64) as u64, 64);
        self.bar(enc);
        // --- MoE FFN (post_attention_norm = pre-FFN norm) ---
        self.enc_reduce(enc, "rmsnorm", &[(&self.st.x,0),(&self.wt.w32[&p("post_attention_norm.weight")],1),(&self.st.h,2)], &[(3,d)], &[(4,self.arch.eps)], 1, 256);
        self.bar(enc);
        // router logits (f32) + router bias, then softmax top-k
        self.enc_reduce(enc, "gemv_w32", &[(&self.st.h,0),(&self.wt.w32[&p("ffn_gate_inp.weight")],1),(&self.ms.moe_lg,2)], &[(3,d),(4,ne)], &[], ((ne+7)/8) as u64, 256);
        self.bar(enc);
        self.enc_reduce(enc, "add_inplace", &[(&self.ms.moe_lg,0),(&self.wt.w32[&p("ffn_gate_inp.bias")],1)], &[(2,ne)], &[], ((ne+63)/64) as u64, 64);
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p["moe_topk"]);
        enc.set_buffer(0, Some(&self.ms.moe_lg), 0);
        enc.set_buffer(1, Some(&self.ms.moe_idx), off);
        enc.set_buffer(2, Some(&self.ms.moe_wgt), 0);
        enc.set_bytes(3, 4, &ne as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(1,1,1), MTLSize::new(32,1,1));
        self.bar(enc);
        // biased SwiGLU-OAI gate/up → moe_act [nu, fe]
        enc.set_compute_pipeline_state(&self.p["moe_gu_q8_oai"]);
        enc.set_buffer(0, Some(&self.st.h), 0);
        enc.set_buffer(1, Some(&self.wt.w8[&p("ffn_gate_exps.weight")]), 0);
        enc.set_buffer(2, Some(&self.wt.w8[&p("ffn_up_exps.weight")]), 0);
        enc.set_buffer(3, Some(&self.ms.moe_act), 0);
        enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &fe as *const u32 as *const c_void);
        enc.set_buffer(6, Some(&self.wt.scale8[&p("ffn_gate_exps.weight")]), 0);
        enc.set_buffer(7, Some(&self.wt.scale8[&p("ffn_up_exps.weight")]), 0);
        enc.set_buffer(8, Some(&self.ms.moe_idx), off);
        enc.set_buffer(9, Some(&self.wt.w32[&p("ffn_gate_exps.bias")]), 0);
        enc.set_buffer(10, Some(&self.wt.w32[&p("ffn_up_exps.bias")]), 0);
        enc.set_bytes(11, 4, &alpha as *const f32 as *const c_void);
        enc.set_bytes(12, 4, &limit as *const f32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(((fe+7)/8) as u64, nu as u64, 1), MTLSize::new(64,1,1));
        self.bar(enc);
        // weighted biased down-proj accumulate into x (no shared expert)
        enc.set_compute_pipeline_state(&self.p["moe_down_q8_oai"]);
        enc.set_buffer(0, Some(&self.ms.moe_act), 0);
        enc.set_buffer(1, Some(&self.wt.w8[&p("ffn_down_exps.weight")]), 0);
        enc.set_buffer(2, Some(&self.st.x), 0);
        enc.set_bytes(3, 4, &fe as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
        enc.set_buffer(5, Some(&self.wt.scale8[&p("ffn_down_exps.weight")]), 0);
        enc.set_buffer(6, Some(&self.ms.moe_idx), off);
        enc.set_buffer(7, Some(&self.ms.moe_wgt), 0);
        enc.set_bytes(8, 4, &nu as *const u32 as *const c_void);
        enc.set_buffer(9, Some(&self.wt.w32[&p("ffn_down_exps.bias")]), 0);
        enc.dispatch_thread_groups(MTLSize::new(((d+7)/8) as u64, 1, 1), MTLSize::new(64,1,1));
        self.bar(enc);
    }

    /// Token id → `st.x`, using whichever representation `token_embd` loaded in.
    ///
    /// Decentralized-worker primitive: run layer range [l_start, l_end) on the residual
    /// stream in `self.st.x`. `do_embed` = stage 0 (embed the token id; otherwise
    /// `self.st.x` already holds the incoming hidden state, written by the host from the
    /// previous stage's activation). `do_head` = final stage (output_norm + lm_head →
    /// logits). Middle stages leave the hidden state in `self.st.x` for the host to read
    /// out and forward.
    ///
    /// Keep this the only copy: a separate three-way version in `graph_qwen4exp` had no
    /// Q6_K branch, so an untied Q6_K embedding panicked on that architecture alone.
    pub(crate) fn embed_token(&self, enc: &metal::ComputeCommandEncoderRef, token: u32, d: u32) {
        self.embed_named(enc, "token_embd.weight", token, d)
    }

    /// The same gather from a named table — the MTP draft block may ship its own
    /// `nextn.embed_tokens.weight`.
    pub(crate) fn embed_named(&self, enc: &metal::ComputeCommandEncoderRef, tbl: &str, token: u32, d: u32) {
        self.embed_named_off(enc, tbl, token, d, 0)
    }

    /// The same gather into a row of `x`, for batched passes that embed M tokens.
    pub(crate) fn embed_named_off(&self, enc: &metal::ComputeCommandEncoderRef, tbl: &str, token: u32, d: u32, dst_off: u64) {
    if let Some(w) = self.wt.w32.get(tbl) {
        self.enc_reduce_off(enc, "copy_buf", &[(&self.st.x, 0, dst_off), (w, 1, token as u64 * d as u64 * 4)],
                            &[(2, d)], &[], d.div_ceil(64) as u64, 64);
    } else if let Some(w) = self.wt.wq.get(tbl) {
        assert_eq!(self.wt.w_qtype.get(tbl), Some(&8), "native embedding requires GGUF Q8_0");
        assert_eq!(d % 32, 0, "Q8_0 embedding width must be block aligned");
        enc.set_compute_pipeline_state(&self.p["embed_gguf_q8_0"]);
        enc.set_buffer(0, Some(w), self.wt.w_off.get(tbl).copied().unwrap_or(0));
        enc.set_buffer(1, Some(&self.st.x), dst_off);
        enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &token as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(d.div_ceil(64) as u64, 1, 1), MTLSize::new(64, 1, 1));
    } else if let Some(w) = self.wt.w6k.get(tbl) {
        enc.set_compute_pipeline_state(&self.p["embed_q6k"]);
        enc.set_buffer(0, Some(w), self.wt.w_off.get(tbl).copied().unwrap_or(0));
        enc.set_buffer(1, Some(&self.st.x), dst_off);
        enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &token as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(((d + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
    } else if self.wt.q4 {
        enc.set_compute_pipeline_state(&self.p["embed_q4"]);
        enc.set_buffer(0, Some(&self.wt.w4[tbl]), 0);
        enc.set_buffer(1, Some(&self.st.x), dst_off);
        enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &token as *const u32 as *const c_void);
        enc.set_buffer(4, Some(&self.wt.scale4[tbl]), 0);
        enc.dispatch_thread_groups(MTLSize::new(((d + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
    } else if self.wt.q8 {
        enc.set_compute_pipeline_state(&self.p["embed_q8"]);
        enc.set_buffer(0, Some(&self.wt.w8[tbl]), 0);
        enc.set_buffer(1, Some(&self.st.x), dst_off);
        enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &token as *const u32 as *const c_void);
        enc.set_buffer(4, Some(&self.wt.scale8[tbl]), 0);
        enc.dispatch_thread_groups(MTLSize::new(((d + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
    } else {
        self.enc_reduce_off(enc, "embed", &[(&self.wt.w16[tbl], 0, 0), (&self.st.x, 1, dst_off)],
                            &[(2, d), (3, token)], &[], ((d + 63) / 64) as u64, 64);
    }
    // Gemma: scale the input embedding by sqrt(d). The following rmsnorm would cancel
    // a uniform scale, but the residual stream keeps it.
    if self.arch.embed_scale != 1.0 {
        enc.set_compute_pipeline_state(&self.p["mul_scalar"]);
        enc.set_buffer(0, Some(&self.st.x), dst_off);
        enc.set_bytes(1, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(2, 4, &self.arch.embed_scale as *const f32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(((d + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
    }
    }

    pub(crate) fn encode_forward_span(&self, enc: &metal::ComputeCommandEncoderRef, token: u32, pos: usize,
                      d: u32, hd: u32, kvdim: u32, group: u32, scale: f32, seq: u32,
                      l_start: usize, l_end: usize, do_embed: bool, do_head: bool) {
        self.encode_span_phase(enc, token, pos, d, hd, kvdim, group, scale, seq,
                               l_start, l_end, do_embed, do_head, MoePhase::Full);
    }

    /// Geometry-deriving entry for the streamed driver: same as
    /// `encode_span_phase` but reads the attention shape off `self.arch` so the
    /// caller does not restate it.
    pub(crate) fn encode_phase(&self, enc: &metal::ComputeCommandEncoderRef, token: u32, pos: usize, seq: u32,
                      l_start: usize, l_end: usize, do_embed: bool, do_head: bool, phase: MoePhase) {
        let d = self.d as u32;
        let hd = self.arch.hd as u32;
        let kvdim = (self.arch.n_kv * self.arch.hd) as u32;
        let group = (self.arch.n_head / self.arch.n_kv.max(1)) as u32;
        let scale = 1.0 / (self.arch.hd as f32).sqrt();
        self.encode_span_phase(enc, token, pos, d, hd, kvdim, group, scale, seq,
                               l_start, l_end, do_embed, do_head, phase);
    }

    /// `encode_forward_span` for one half of a streamed MoE layer.
    ///
    /// The streamed driver calls this rather than `encode_mla` directly: going straight
    /// to `encode_mla` made disk-streaming DeepSeek-only, since any other streamed arch
    /// reached `forward_id_streamed` and then ran the MLA graph over tensors it does not
    /// have. Adding a stream-capable arch is one branch here. Architectures with no
    /// phase split accept only `Full`.
    pub(crate) fn encode_span_phase(&self, enc: &metal::ComputeCommandEncoderRef, token: u32, pos: usize,
                      d: u32, hd: u32, kvdim: u32, group: u32, scale: f32, seq: u32,
                      l_start: usize, l_end: usize, do_embed: bool, do_head: bool, phase: MoePhase) {
        if self.arch.mla.is_some() {
            self.encode_mla(enc, token, pos, seq, l_start, l_end, do_embed, do_head, phase);
            return;
        }
        if self.arch.qwen4exp.is_some() {
            self.encode_qwen4exp(enc, token, pos, d, seq, l_start, l_end, do_embed, do_head, phase);
            return;
        }
        assert!(matches!(phase, MoePhase::Full),
            "this architecture has no streamed Route/Experts split — see encode_span_phase");
        // embed token -> x (stage 0 only; later stages receive x as an activation)
        if do_embed { self.embed_token(enc, token, d); }

        self.bar(enc); // x (embedding or incoming activation) ready

        let _ = (hd, kvdim, group, scale); // per-layer values now come from self.arch.layers[l]
        for l in l_start..l_end {
            let p = |s: &str| format!("blk.{l}.{s}");
            let lp = self.arch.layers[l];           // per-layer plan (geometry, RoPE, KV source)
            if self.arch.gpt_oss { self.encode_gptoss_layer(enc, l, pos, seq); continue; }
            // qwen35 Gated-DeltaNet forward.
            // Structure: x += mixer(attn_norm(x)); x += FFN(post_norm(x)).
            if let Some(sc) = self.arch.ssm {
                if lp.is_ssm {
                    let (s_st, hk, hv) = (sc.d_state, sc.n_group, sc.dt_rank);
                    let d_inner = sc.d_inner; let conv_ch = d_inner + 2*hk*s_st; let conv_k = sc.conv_kernel;
                    let head_v = d_inner / hv;
                    // h = attn_norm(x)
                    self.enc_reduce(enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32[&p("attn_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
                    self.bar(enc); // h ready for the projections
                    // Activation-diff debugging: ssm_qkv/ssm_z/ssm_o/x are scratch reused by
                    // every SSM layer, so reading them back after the forward gives the last
                    // layer's values. Layer-0 checks must copy into gate/up (unused in the
                    // qwen35 path) at l==0, like the two dbg captures below.
                    let dbg = l == 0 && self.cfg.ssm_dbg;
                    // qkv_mixed = attn_qkv @ h  [conv_ch];  z = attn_gate @ h  [d_inner]
                    self.mm(enc, "plain", &p("attn_qkv.weight"), &self.st.h, &self.st.ssm_qkv, d, conv_ch, None);
                    self.mm(enc, "plain", &p("attn_gate.weight"), &self.st.h, &self.st.ssm_z, d, d_inner, None);
                    if dbg { self.enc_reduce(enc, "copy_buf", &[(&self.st.gate, 0), (&self.st.ssm_z, 1)], &[(2, d_inner)], &[], ((d_inner+63)/64) as u64, 64); }
                    // When the loader retains alpha/beta as F32 narrow matrices,
                    // fuse their shared activation read and epilogue when requested;
                    // quantized/non-vector geometry keeps the general path.
                    let alpha_name = p("ssm_alpha.weight");
                    let beta_name = p("ssm_beta.weight");
                    if self.cfg.flash_gdn_ab_fused && d % 4 == 0
                        && self.wt.repr(&alpha_name) == Repr::F32
                        && self.wt.repr(&beta_name) == Repr::F32
                    {
                        enc.set_compute_pipeline_state(&self.p["gdn_ab_fused"]);
                        enc.set_buffer(0, Some(&self.st.h), 0);
                        enc.set_buffer(1, Some(&self.wt.w32[&alpha_name]), 0);
                        enc.set_buffer(2, Some(&self.wt.w32[&beta_name]), 0);
                        enc.set_buffer(3, Some(&self.st.ssm_gate), 0);
                        enc.set_buffer(4, Some(&self.st.ssm_beta), 0);
                        enc.set_buffer(5, Some(&self.wt.w32[&p("ssm_dt.bias")]), 0);
                        enc.set_buffer(6, Some(&self.wt.w32[&p("ssm_a")]), 0);
                        let k4 = d / 4;
                        enc.set_bytes(7, 4, &k4 as *const u32 as *const c_void);
                        enc.set_bytes(8, 4, &hv as *const u32 as *const c_void);
                        enc.dispatch_thread_groups(MTLSize::new(((hv + 3) / 4) as u64, 1, 1), MTLSize::new(128, 1, 1));
                    } else {
                        self.mm(enc, "plain", &alpha_name, &self.st.h, &self.st.ssm_gate, d, hv, None);
                        self.mm(enc, "plain", &beta_name, &self.st.h, &self.st.ssm_beta, d, hv, None);
                        self.bar(enc); // projections done (qkv/z/alpha/beta all read h, ran concurrently)
                        self.enc_reduce(enc, "ssm_ab", &[(&self.st.ssm_gate, 0), (&self.st.ssm_beta, 1), (&self.wt.w32[&p("ssm_dt.bias")], 2), (&self.wt.w32[&p("ssm_a")], 3)], &[(4, hv), (5, hv)], &[], ((hv + 63)/64) as u64, 64);
                    }
                    // causal conv1d + SILU (+ rolling conv state)
                    enc.set_compute_pipeline_state(&self.p["conv1d_decode"]);
                    enc.set_buffer(0, Some(&self.st.ssm_qkv), 0);
                    enc.set_buffer(1, Some(&self.st.conv_state[l]), 0);
                    enc.set_buffer(2, Some(&self.wt.w32[&p("ssm_conv1d.weight")]), 0);
                    enc.set_bytes(3, 4, &conv_ch as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &conv_k as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(((conv_ch + 63)/64) as u64, 1, 1), MTLSize::new(64, 1, 1));
                    self.bar(enc); // conv + ssm_ab done
                    // delta-net recurrence with FUSED per-head q/k L2-norm (M=1),
                    // reference kernel_gated_delta_net style (register-resident columns).
                    enc.set_compute_pipeline_state(&self.p["deltanet_fused"]);
                    enc.set_buffer(0, Some(&self.st.ssm_state[l]), 0);
                    enc.set_buffer(1, Some(&self.st.ssm_qkv), 0);
                    enc.set_buffer(2, Some(&self.st.ssm_gate), 0);
                    enc.set_buffer(3, Some(&self.st.ssm_beta), 0);
                    enc.set_buffer(4, Some(&self.st.ssm_o), 0);
                    enc.set_bytes(5, 4, &s_st as *const u32 as *const c_void);
                    enc.set_bytes(6, 4, &hk as *const u32 as *const c_void);
                    enc.set_bytes(7, 4, &hv as *const u32 as *const c_void);
                    enc.set_bytes(8, 4, &conv_ch as *const u32 as *const c_void);
                    enc.set_bytes(9, 4, &1u32 as *const u32 as *const c_void);
                    enc.set_bytes(10, 4, &self.arch.eps as *const f32 as *const c_void);
                    enc.set_buffer(11, Some(&self.st.ssm_state[l]), 0);   // snapshot disabled
                    enc.set_bytes(12, 4, &u32::MAX as *const u32 as *const c_void);
                    // OJAS_KMAP_DIV=1 selects the grouped value->key head mapping.
                    let kmap_div: u32 = self.cfg.moe_kmap_div as u32;
                    enc.set_bytes(13, 4, &kmap_div as *const u32 as *const c_void);
                    enc.set_bytes(14, 4, &0u32 as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new((s_st/4) as u64, hv as u64, 1), MTLSize::new(128, 1, 1));
                    self.bar(enc); // deltanet done
                    // gated RMSNorm: ssm_o = (rmsnorm(ssm_o)·ssm_norm)·silu(z), per v-head (head_v dims)
                    enc.set_compute_pipeline_state(&self.p["gated_rmsnorm"]);
                    enc.set_buffer(0, Some(&self.st.ssm_o), 0);
                    enc.set_buffer(1, Some(&self.wt.w32[&p("ssm_norm.weight")]), 0);
                    enc.set_buffer(2, Some(&self.st.ssm_z), 0);
                    enc.set_bytes(3, 4, &head_v as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
                    enc.set_bytes(5, 4, &d_inner as *const u32 as *const c_void);
                    enc.set_bytes(6, 4, &0u32 as *const u32 as *const c_void);   // qwen35: silu gate
                    enc.dispatch_thread_groups(MTLSize::new(hv as u64, 1, 1), MTLSize::new(32, 1, 1));
                    self.bar(enc); // gated norm done
                    // mixer_out = ssm_out @ ssm_o;  x += mixer_out
                    self.mm(enc, "accum", &p("ssm_out.weight"), &self.st.ssm_o, &self.st.x, d_inner, d, None);
                } else {
                    // Gated attention layer (Qwen3-Next, ref qwen35.cpp build_layer_attn):
                    // attn_q projects to per-head [q(hd)|gate(hd)] (2*qdim); per-head QK-norm;
                    // partial NEOX rope (n_rot=64 of hd=256; M-RoPE sections reduce to this
                    // for text); standard GQA attention; attn *= sigmoid(gate); then wo.
                    let (hd, kvdim, qdim) = (lp.head_dim, lp.kvdim, lp.qdim);
                    let group = lp.n_head / lp.n_kv.max(1);
                    // h = attn_norm(x)
                    self.enc_reduce(enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32[&p("attn_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
                    self.bar(enc); // h ready for the projections
                    // qfull = attn_q @ h [2*qdim] (→ ssm_qkv scratch, 8192 = fits); k,v [kvdim]
                    self.mm(enc, "plain", &p("attn_q.weight"), &self.st.h, &self.st.ssm_qkv, d, 2*qdim, None);
                    self.mm(enc, "plain", &p("attn_k.weight"), &self.st.h, &self.st.k, d, kvdim, None);
                    self.mm(enc, "plain", &p("attn_v.weight"), &self.st.h, &self.st.v, d, kvdim, None);
                    // split contiguous q out of qfull (gate stays in qfull for the multiply)
                    self.bar(enc); // q/k/v projections done (ran concurrently)
                    self.enc_reduce(enc, "qgate_split", &[(&self.st.ssm_qkv, 0), (&self.st.q, 1)], &[(2, hd), (3, qdim), (4, 1)], &[], ((qdim+63)/64) as u64, 64);
                    // per-head QK-norm
                    let (nq, nk) = (lp.n_head, lp.n_kv);
                    self.bar(enc); // q split done
                    enc.set_compute_pipeline_state(&self.p["qk_rmsnorm"]);
                    enc.set_buffer(0, Some(&self.st.q), 0);
                    enc.set_buffer(1, Some(&self.st.k), 0);
                    enc.set_buffer(2, Some(&self.wt.w32[&p("attn_q_norm.weight")]), 0);
                    enc.set_buffer(3, Some(&self.wt.w32[&p("attn_k_norm.weight")]), 0);
                    enc.set_bytes(4, 4, &hd as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &nq as *const u32 as *const c_void);
                    enc.set_bytes(6, 4, &nk as *const u32 as *const c_void);
                    enc.set_bytes(7, 4, &self.arch.eps as *const f32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new((nq + nk) as u64, 1, 1), MTLSize::new(32, 1, 1));
                    self.bar(enc); // qk-norm done
                    // partial rope (rd=n_rot) + store k,v to cache
                    let (totq, totk) = (qdim/2, kvdim/2);
                    let off = pos as u32 * kvdim;
                    self.enc_reduce(enc, "rope_qk_store",
                        &[(&self.st.q, 0), (&self.st.k, 1), (&self.st.v, 2), (&self.st.kcache[l], 3), (&self.st.vcache[l], 4)],
                        &[(5, hd), (6, pos as u32), (8, totq), (9, totk), (10, kvdim), (11, off), (12, 1), (13, sc.n_rot)], &[(7, lp.rope_base)],
                        (((totq + totk + kvdim) + 63) / 64) as u64, 64);
                    self.bar(enc); // rope + cache store done
                    // GQA attention: score-array kernel ≤512; page-sparse (OJAS_SPARSE)
                    // when the history is at least 2× the budget; flash-decoding else
                    let attn_t = self.tune.gemv_plan.get(&(0, 2)).map(|p| p.threads).unwrap_or(64).min(256);
                    let sparse = self.arch.sparse_budget.filter(|&b| seq > 2*b.max(256));
                    if let Some(b) = sparse {
                        self.sparse_attn(enc, lp.n_head, l, hd, kvdim, seq, group, lp.scale, b);
                    } else if seq <= 512 {
                        super::attn_log("decode", "attention_short", seq);
                        self.enc_reduce(enc, "attention_short",
                            &[(&self.st.q, 0), (&self.st.kcache[l], 1), (&self.st.vcache[l], 2), (&self.st.attn, 3)],
                            &[(4, hd), (5, kvdim), (6, seq), (7, group)], &[(8, lp.scale)],
                            lp.n_head as u64, attn_t);
                    } else {
                        super::attn_log("decode", "attn_flash (flash-decoding)", seq);
                        self.attn_flash(enc, lp.n_head, l, l, hd, kvdim, seq, group, lp.scale);
                    }
                    // attn *= sigmoid(gate)
                    self.bar(enc); // attention done
                    self.enc_reduce(enc, "gate_mul_sigmoid", &[(&self.st.attn, 0), (&self.st.ssm_qkv, 1)], &[(2, hd), (3, qdim), (4, 1)], &[], ((qdim+63)/64) as u64, 64);
                    self.bar(enc); // gate applied
                    // x += wo @ attn
                    self.mm(enc, "accum", &p("attn_output.weight"), &self.st.attn, &self.st.x, qdim, d, None);
                }
                // FFN with post_attention_norm as the pre-FFN norm, shared by SSM and
                // attention layers. On qwen35 it is the FFN pre-norm, not a gemma-style
                // post-norm, so the sandwich path does not apply.
                self.bar(enc); // mixer residual in x
                self.enc_reduce(enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32[&p("post_attention_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
                self.bar(enc); // h ready for FFN
                if let Some(mc) = self.arch.moe {
                    // MoE FFN (qwen35moe.cpp build_layer_ffn): softmax router → top-k
                    // expert SwiGLUs (weights renormed over the k) + shared expert
                    // scaled by sigmoid(gate_inp_shexp·h).
                    let (ne, nu, fe, fs) = (mc.n_expert, mc.n_used, mc.ffn_exp, mc.ffn_shexp);
                    self.check_moe(&p, d, fe);
                    // stage 1 (concurrent): router logits + shared-expert SwiGLU +
                    // shared-expert scalar gate — all three only read h.
                    self.enc_reduce(enc, "gemv_w32", &[(&self.st.h, 0), (&self.wt.w32[&p("ffn_gate_inp.weight")], 1), (&self.ms.moe_lg, 2)], &[(3, d), (4, ne)], &[], ((ne + 7)/8) as u64, 256);
                    let act0 = 0u32;
                    enc.set_compute_pipeline_state(&self.p["ffn_gu_q4"]);
                    enc.set_buffer(0, Some(&self.st.h), 0);
                    enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_gate_shexp.weight")]), 0);
                    enc.set_buffer(2, Some(&self.wt.w4[&p("ffn_up_shexp.weight")]), 0);
                    enc.set_buffer(3, Some(&self.st.act), 0);
                    enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &fs as *const u32 as *const c_void);
                    enc.set_buffer(6, Some(&self.wt.scale4[&p("ffn_gate_shexp.weight")]), 0);
                    enc.set_buffer(7, Some(&self.wt.scale4[&p("ffn_up_shexp.weight")]), 0);
                    enc.set_bytes(8, 4, &act0 as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new((fs / 8) as u64, 1, 1), MTLSize::new(64, 1, 1));
                    self.enc_reduce(enc, "gemv_w32", &[(&self.st.h, 0), (&self.wt.w32[&p("ffn_gate_inp_shexp.weight")], 1), (&self.ms.moe_sh, 2)], &[(3, d), (4, 1)], &[], 1, 32);
                    self.bar(enc);
                    // stage 2 (concurrent): top-k on router logits + shared-expert down-proj
                    {   // top-k writes this layer's slot of moe_idx (read back post-forward
                        // by the expert prefetcher)
                        enc.set_compute_pipeline_state(&self.p["moe_topk"]);
                        enc.set_buffer(0, Some(&self.ms.moe_lg), 0);
                        enc.set_buffer(1, Some(&self.ms.moe_idx), (l as u64) * (MAXM as u64) * (nu as u64) * 4);
                        enc.set_buffer(2, Some(&self.ms.moe_wgt), 0);
                        enc.set_bytes(3, 4, &ne as *const u32 as *const c_void);
                        enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
                        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
                    }
                    self.mm(enc, "plain", &p("ffn_down_shexp.weight"), &self.st.act, &self.st.tmp, fs, d, None);
                    self.bar(enc);
                    // stage 3: per-expert SwiGLU → moe_act [nu, fe]
                    enc.set_compute_pipeline_state(&self.p["moe_gu_q4"]);
                    enc.set_buffer(0, Some(&self.st.h), 0);
                    enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_gate_exps.weight")]), 0);
                    enc.set_buffer(2, Some(&self.wt.w4[&p("ffn_up_exps.weight")]), 0);
                    enc.set_buffer(3, Some(&self.ms.moe_act), 0);
                    enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &fe as *const u32 as *const c_void);
                    enc.set_buffer(6, Some(&self.wt.scale4[&p("ffn_gate_exps.weight")]), 0);
                    enc.set_buffer(7, Some(&self.wt.scale4[&p("ffn_up_exps.weight")]), 0);
                    enc.set_buffer(8, Some(&self.ms.moe_idx), (l as u64) * (MAXM as u64) * (nu as u64) * 4);
                    enc.dispatch_thread_groups(MTLSize::new(((fe + 7)/8) as u64, nu as u64, 1), MTLSize::new(64, 1, 1));
                    self.bar(enc); // expert activations ready
                    // weighted expert down-proj accumulate into x
                    enc.set_compute_pipeline_state(&self.p["moe_down_q4"]);
                    enc.set_buffer(0, Some(&self.ms.moe_act), 0);
                    enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_down_exps.weight")]), 0);
                    enc.set_buffer(2, Some(&self.st.x), 0);
                    enc.set_bytes(3, 4, &fe as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                    enc.set_buffer(5, Some(&self.wt.scale4[&p("ffn_down_exps.weight")]), 0);
                    enc.set_buffer(6, Some(&self.ms.moe_idx), (l as u64) * (MAXM as u64) * (nu as u64) * 4);
                    enc.set_buffer(7, Some(&self.ms.moe_wgt), 0);
                    enc.set_bytes(8, 4, &nu as *const u32 as *const c_void);
                    enc.set_buffer(9, Some(&self.st.tmp), 0);       // shared-expert down output
                    enc.set_buffer(10, Some(&self.ms.moe_sh), 0);   // scalar gate (pre-sigmoid)
                    enc.dispatch_thread_groups(MTLSize::new(((d + 7)/8) as u64, 1, 1), MTLSize::new(64, 1, 1));
                } else {
                let nffn = self.arch.ffn as u32; let act = 0u32; // SiLU
                // Q4L first — same values as Q4_K, tuned layout. This is the dense-FFN
                // arm of the hybrid path, a separate site from the one below.
                if !matches!(self.wt.fused_repr_ffn(&p), Some(Repr::Q4L) | Some(Repr::Q4) | Some(Repr::Q8) | Some(Repr::Q20) | Some(Repr::F16) | Some(Repr::Q4K) | Some(Repr::Q6K)) {
                    // Native quant: project gate/up separately, then combine. Same
                    // reason as the dense site below — no fused kernel reads `wq`.
                    let (gn, un) = (p("ffn_gate.weight"), p("ffn_up.weight"));
                    self.mm(enc, "plain", &gn, &self.st.h, &self.st.gate, d, nffn, None);
                    self.mm(enc, "plain", &un, &self.st.h, &self.st.up, d, nffn, None);
                    self.bar(enc);
                    enc.set_compute_pipeline_state(&self.p["ffn_gu_split"]);
                    enc.set_buffer(0, Some(&self.st.gate), 0);
                    enc.set_buffer(1, Some(&self.st.up), 0);
                    enc.set_buffer(2, Some(&self.st.act), 0);
                    enc.set_bytes(3, 4, &nffn as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &act as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(((nffn + 255) / 256) as u64, 1, 1),
                                               MTLSize::new(256, 1, 1));
                } else if self.wt.w4l.contains_key(&p("ffn_gate.weight")) && self.wt.w4l.contains_key(&p("ffn_up.weight")) {
                    let (gn, un) = (p("ffn_gate.weight"), p("ffn_up.weight"));
                    enc.set_compute_pipeline_state(&self.p["ffn_gu_q4l"]);
                    enc.set_buffer(0, Some(&self.st.h), 0);
                    enc.set_buffer(1, Some(&self.wt.w4l[&gn]), 0);
                    enc.set_buffer(2, Some(&self.wt.w4l[&un]), 0);
                    enc.set_buffer(3, Some(&self.st.act), 0);
                    enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                    enc.set_buffer(6, Some(&self.wt.q4l_a[&gn]), 0);
                    enc.set_buffer(7, Some(&self.wt.q4l_b[&gn]), 0);
                    enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
                    enc.set_buffer(9, Some(&self.wt.q4l_a[&un]), 0);
                    enc.set_buffer(10, Some(&self.wt.q4l_b[&un]), 0);
                    let ft = self.q4l_threads(0, 3);   // pseudo-key: fused ffn_gu_q4l
                    let frows = ft / 32 * 4;
                    enc.dispatch_thread_groups(MTLSize::new(((nffn + frows - 1) / frows) as u64, 1, 1), MTLSize::new(ft as u64, 1, 1));
                } else if self.wt.q4 {
                    enc.set_compute_pipeline_state(&self.p["ffn_gu_q4"]);
                    enc.set_buffer(0, Some(&self.st.h), 0);
                    enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_gate.weight")]), 0);
                    enc.set_buffer(2, Some(&self.wt.w4[&p("ffn_up.weight")]), 0);
                    enc.set_buffer(3, Some(&self.st.act), 0);
                    enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                    enc.set_buffer(6, Some(&self.wt.scale4[&p("ffn_gate.weight")]), 0);
                    enc.set_buffer(7, Some(&self.wt.scale4[&p("ffn_up.weight")]), 0);
                    enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new((nffn / 8) as u64, 1, 1), MTLSize::new(64, 1, 1));
                } else {
                    // f16/q8 (training-forward path: f16 GGUF)
                    enc.set_compute_pipeline_state(&self.p[if self.wt.q8 { "ffn_gu_q8" } else { "ffn_gu_f16" }]);
                    enc.set_buffer(0, Some(&self.st.h), 0);
                    enc.set_buffer(1, Some(if self.wt.q8 { &self.wt.w8[&p("ffn_gate.weight")] } else { &self.wt.w16[&p("ffn_gate.weight")] }), 0);
                    enc.set_buffer(2, Some(if self.wt.q8 { &self.wt.w8[&p("ffn_up.weight")] } else { &self.wt.w16[&p("ffn_up.weight")] }), 0);
                    enc.set_buffer(3, Some(&self.st.act), 0);
                    enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                    if self.wt.q8 {
                        enc.set_buffer(6, Some(&self.wt.scale8[&p("ffn_gate.weight")]), 0);
                        enc.set_buffer(7, Some(&self.wt.scale8[&p("ffn_up.weight")]), 0);
                        enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
                    } else {
                        enc.set_bytes(6, 4, &act as *const u32 as *const c_void);
                    }
                    let ft = if self.wt.q8 { self.tune.gemv_plan.get(&(0, 1)).map(|p| p.threads).unwrap_or(256) } else { 256 };
                    let frows = (ft / 32).max(1) as u32;
                    enc.dispatch_thread_groups(MTLSize::new(((nffn + frows - 1) / frows) as u64, 1, 1), MTLSize::new(ft, 1, 1));
                }
                self.bar(enc); // SwiGLU act ready
                self.mm(enc, "accum", &p("ffn_down.weight"), &self.st.act, &self.st.x, self.arch.ffn as u32, d, None);
                }
                self.bar(enc); // layer output complete in x (next layer's norm reads it)
                let dbg0 = l == 0 && self.cfg.ssm_dbg;
                if dbg0 { self.enc_reduce(enc, "copy_buf", &[(&self.st.up, 0), (&self.st.x, 1)], &[(2, d)], &[], ((d+63)/64) as u64, 64); }
                continue;
            }
            let (hd, kvdim, qdim) = (lp.head_dim, lp.kvdim, lp.qdim);
            let group = lp.n_head / lp.n_kv.max(1);
            // attn norm
            self.bar(enc); // previous layer's output complete in x
            self.enc_reduce(enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32[&p("attn_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
                    self.bar(enc); // h ready for the projections
            // q,k,v = W@h + bias, fused into one dispatch (q4/q8/f16). qdim = n_head*head_dim.
            // Shared-KV layers still project their own Q,K; V comes from a zero dummy and
            // attention reads vcache[kv_source] instead.
            //
            // Native-quant weights live in `wq`, which no fused qkv kernel can read (each
            // walks one hard-coded block layout), so those take three separate matvecs.
            // The fused form is a dispatch-count optimization, not a correctness
            // requirement.
            let qkv_repr = self.wt.fused_repr(
                &[&p("attn_q.weight"), &p("attn_k.weight"), &p("attn_v.weight")]);
            // The query must cover all three weights: probing only attn_q and indexing
            // the same map for k and v panics on a mixed-type file, and those are real —
            // one 0.5B ships attn_q as IQ4_NL and attn_k as Q5_1, so `fused_repr` returns
            // None and no fused kernel can serve the trio.
            //
            // Separate matvecs are the universal fallback — `mm()` resolves each weight
            // independently, so they work for any mix. Use them whenever no fused kernel
            // applies, plus the two cases where three GEMVs measured faster than the
            // fused block-per-lane kernel.
            let qkv_sep = match qkv_repr {
                None | Some(Repr::Native) => true,
                Some(Repr::Q4) | Some(Repr::Q20) => !self.arch.qkv_bias,
                _ => false,
            };
            if qkv_sep {
                // qkv_bias archs (qwen2) keep their bias here; dropping it produces
                // fluent garbage rather than a crash.
                let bias = |t: &str| if self.arch.qkv_bias { self.wt.w32.get(&p(t)) } else { None };
                let kind = if self.arch.qkv_bias { "bias" } else { "plain" };
                self.mm(enc, kind, &p("attn_q.weight"), &self.st.h, &self.st.q, d, qdim, bias("attn_q.bias"));
                self.mm(enc, kind, &p("attn_k.weight"), &self.st.h, &self.st.k, d, kvdim, bias("attn_k.bias"));
                if lp.has_v { self.mm(enc, kind, &p("attn_v.weight"), &self.st.h, &self.st.v, d, kvdim, bias("attn_v.bias")); }
            } else if qkv_repr == Some(Repr::Q4L) && qdim % 4 == 0 && kvdim % 4 == 0 {
                self.qkv_q4l(enc, &p, &self.st.h, d, qdim, kvdim);
            } else if self.wt.q4 {
                self.qkv_q4(enc, &p, &self.st.h, d, qdim, kvdim);
            } else if self.wt.q8 {
                self.qkv(enc, &p, &self.st.h, d, qdim, kvdim);
            } else {
                self.qkv_f16(enc, &p, &self.st.h, d, qdim, kvdim);
            }
            // Gemma4 (v_rmsnorm): V=K copy for KV-shared layers, then q-norm + k-norm +
            // weightless V-norm in one fused qkv_rmsnorm dispatch. Other qk-norm archs
            // (Qwen3/Gemma3) use qk_rmsnorm (q,k only).
            if self.arch.v_rmsnorm {
                if !lp.has_v {
                    enc.set_compute_pipeline_state(&self.p["copy_buf"]);
                    enc.set_buffer(0, Some(&self.st.v), 0);
                    enc.set_buffer(1, Some(&self.st.k), 0);
                    enc.set_bytes(2, 4, &kvdim as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(((kvdim + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
                }
                let (nq, nk) = (lp.n_head, lp.n_kv);
                enc.set_compute_pipeline_state(&self.p["qkv_rmsnorm"]);
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.st.k), 0);
                enc.set_buffer(2, Some(&self.st.v), 0);
                enc.set_buffer(3, Some(&self.wt.w32[&p("attn_q_norm.weight")]), 0);
                enc.set_buffer(4, Some(&self.wt.w32[&p("attn_k_norm.weight")]), 0);
                enc.set_buffer(5, Some(&self.st.ones), 0); // weightless V norm
                enc.set_bytes(6, 4, &hd as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &nq as *const u32 as *const c_void);
                enc.set_bytes(8, 4, &nk as *const u32 as *const c_void);
                enc.set_bytes(9, 4, &nk as *const u32 as *const c_void); // nv = n_kv
                enc.set_bytes(10, 4, &self.arch.eps as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new((nq + nk + nk) as u64, 1, 1), MTLSize::new(32, 1, 1));
            } else if self.arch.qk_norm {
                let (nq, nk) = (lp.n_head, lp.n_kv);
                self.bar(&enc); // q split done
                enc.set_compute_pipeline_state(&self.p["qk_rmsnorm"]);
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.st.k), 0);
                enc.set_buffer(2, Some(&self.wt.w32[&p("attn_q_norm.weight")]), 0);
                enc.set_buffer(3, Some(&self.wt.w32[&p("attn_k_norm.weight")]), 0);
                enc.set_bytes(4, 4, &hd as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &nq as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &nk as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &self.arch.eps as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new((nq + nk) as u64, 1, 1), MTLSize::new(32, 1, 1));
            }
            // rope(q,k) + store(k,v) fused into one dispatch
            self.bar(enc); // q,k,v (and any qk-norm) complete before rope reads them
            let pos_u = pos as u32;
            let totq = qdim / 2;
            let totk = kvdim / 2;
            let off = pos as u32 * kvdim;
            let neox = self.arch.rope_neox as u32;
            self.enc_reduce(enc, "rope_qk_store",
                &[(&self.st.q, 0), (&self.st.k, 1), (&self.st.v, 2), (&self.st.kcache[l], 3), (&self.st.vcache[l], 4)],
                &[(5, hd), (6, pos_u), (8, totq), (9, totk), (10, kvdim), (11, off), (12, neox), (13, hd)], &[(7, lp.rope_base)],
                (((totq + totk + kvdim) + 63) / 64) as u64, 64);
            self.bar(enc); // k,v are in the cache before attention reads it
            // attention (threads autotuned). V comes from kv_source (self, unless KV-shared).
            let attn_t = self.tune.gemv_plan.get(&(0, 2)).map(|p| p.threads).unwrap_or(64).min(256); // streaming kernel: nsg ≤ 8
            // Split the KV across workgroups at almost any depth. attention_short gives
            // the whole dispatch n_head threadgroups — 16 on qwen2-3B, about 10% of this
            // GPU — so it cost 1.195 ms per token (15% of all decode kernel time) while
            // moving under 4 MB. Splitting KV costs one extra merge dispatch and buys
            // real occupancy:
            //
            //   seq ~100   nwg 0 -> 101.3 tok/s,  nwg 4 -> 115.1
            //   seq ~290   nwg 0 ->  76.0,        nwg 4 -> 112.6
            //
            // A threshold that engages only above seq 512, or splits just 2 ways, misses
            // the depths where attention_short is worst. Target ~64 threadgroups total,
            // keep >=16 keys per workgroup, and keep the deep-context term (seq/1024) so
            // long contexts still widen.
            let force_nwg: u32 = attn_force_nwg();
            let nwg_auto: u32 = if seq < 32 { 0 } else {
                let base = (64 / lp.n_head.max(1)).max(seq / 1024);
                base.clamp(2, ojas_metal::kernels::attn::ATTN_NWG as u32).min((seq / 16).max(2))
            };
            // OJAS_ATTN_NWG=1 forces the single-dispatch kernel, for A/B measurement.
            if force_nwg == 1 || (force_nwg == 0 && nwg_auto == 0) {
                super::attn_log("decode(dense)", "attention_short (single-dispatch)", seq);
                self.enc_reduce(enc, "attention_short",
                    &[(&self.st.q, 0), (&self.st.kcache[l], 1), (&self.st.vcache[lp.kv_source], 2), (&self.st.attn, 3)],
                    &[(4, hd), (5, kvdim), (6, seq), (7, group)], &[(8, lp.scale)],
                    lp.n_head as u64, attn_t);
            } else {
                super::attn_log("decode(dense)", "flash-decoding (attention_part+merge)", seq);
                // flash-decoding: split KV across workgroups (the reference nwg pattern),
                // partials merged exactly — keeps the GPU busy at any context depth.
                let nwg = if force_nwg > 0 { force_nwg.min(ojas_metal::kernels::attn::ATTN_NWG as u32).min(seq.max(1)) } else { nwg_auto };
                enc.set_compute_pipeline_state(&self.p["attention_part"]);
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.st.kcache[l]), 0);
                enc.set_buffer(2, Some(&self.st.vcache[lp.kv_source]), 0);
                enc.set_buffer(3, Some(&self.st.attn_part), 0);
                enc.set_bytes(4, 4, &hd as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &kvdim as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &seq as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &group as *const u32 as *const c_void);
                enc.set_bytes(8, 4, &lp.scale as *const f32 as *const c_void);
                enc.set_bytes(9, 4, &nwg as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(lp.n_head as u64, nwg as u64, 1), MTLSize::new(256, 1, 1));
                self.bar(enc);
                enc.set_compute_pipeline_state(&self.p["attention_merge"]);
                enc.set_buffer(0, Some(&self.st.attn_part), 0);
                enc.set_buffer(1, Some(&self.st.attn), 0);
                enc.set_bytes(2, 4, &hd as *const u32 as *const c_void);
                enc.set_bytes(3, 4, &nwg as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(lp.n_head as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            self.bar(enc); // attention output complete
            // o_proj. Gemma (sandwich): tmp = Wo@attn; h = post_attention_norm(tmp); x += h.
            // Others: x += Wo@attn (residual fused into the GEMV). K = n_head*head_dim (=qdim).
            if self.arch.sandwich {
                // tmp = Wo@attn; x = x + post_attention_norm(tmp)  (fused, oscale=1).
                self.mm(enc, "plain", &p("attn_output.weight"), &self.st.attn, &self.st.tmp, qdim, d, None);
                self.enc_reduce(enc, "rmsnorm_add", &[(&self.st.x, 0), (&self.st.tmp, 1), (&self.wt.w32[&p("post_attention_norm.weight")], 2)], &[(3, d)], &[(4, self.arch.eps), (5, 1.0)], 1, 256);
            } else {
                self.mm(enc, "accum", &p("attn_output.weight"), &self.st.attn, &self.st.x, qdim, d, None);
            }
            // ffn norm. Do not fold this into ffn_gu_q4l: measured 8.19 -> 8.59 ms of
            // GPU time per token, consistently across an interleaved A/B. Folding
            // removes 36 of ~290 dispatches, but each of the 344 threadgroups in the
            // SwiGLU dispatch then redoes the reduction and re-reads x and
            // ffn_norm.weight — ~198 MB/token of extra traffic, more than the sync it
            // saves. A separate 5 us dispatch is cheaper.
            self.bar(enc); // attention residual landed in x
            self.enc_reduce(enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32[&p("ffn_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
            self.bar(enc); // h ready for the SwiGLU
            // SwiGLU first half fused: act = silu(Wg·h)·(Wu·h) in one dispatch (q4/q8/f16).
            {
                let nffn = self.arch.ffn as u32;
                let act = self.arch.gelu as u32; // 0=silu (Qwen/Llama), 1=gelu (Gemma)
                let mut return_early_ffn = false;
                let gate_n = p("ffn_gate.weight");
                let up_n = p("ffn_up.weight");
                if self.wt.w4l.contains_key(&gate_n) && self.wt.w4l.contains_key(&up_n) {
                    enc.set_compute_pipeline_state(&self.p["ffn_gu_q4l"]);
                    enc.set_buffer(0, Some(&self.st.h), 0);
                    enc.set_buffer(1, Some(&self.wt.w4l[&gate_n]), 0);
                    enc.set_buffer(2, Some(&self.wt.w4l[&up_n]), 0);
                    enc.set_buffer(3, Some(&self.st.act), 0);
                    enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                    enc.set_buffer(6, Some(&self.wt.q4l_a[&gate_n]), 0);
                    enc.set_buffer(7, Some(&self.wt.q4l_b[&gate_n]), 0);
                    enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
                    enc.set_buffer(9, Some(&self.wt.q4l_a[&up_n]), 0);
                    enc.set_buffer(10, Some(&self.wt.q4l_b[&up_n]), 0);
                    enc.dispatch_thread_groups(MTLSize::new(((nffn + 31) / 32) as u64, 1, 1), MTLSize::new(256, 1, 1));
                    return_early_ffn = true;
                }
                let native_k = if self.wt.w4k.contains_key(&gate_n) && self.wt.w4k.contains_key(&up_n) {
                    Some(("ffn_gu_q4k", &self.wt.w4k))
                } else if self.wt.w6k.contains_key(&gate_n) && self.wt.w6k.contains_key(&up_n) {
                    Some(("ffn_gu_q6k", &self.wt.w6k))
                } else {
                    None
                };
                if return_early_ffn {
                    // already dispatched above
                } else if !matches!(self.wt.fused_repr(&[&gate_n, &up_n]), Some(Repr::Q4L) | Some(Repr::Q4) | Some(Repr::Q8) | Some(Repr::Q20) | Some(Repr::F16) | Some(Repr::Q4K) | Some(Repr::Q6K)) {
                    // Native quant: no fused gate/up kernel can read `wq` (each one
                    // walks a single hard-coded block layout), so project separately
                    // and combine. Three dispatches instead of one; the fused form is
                    // a dispatch-count optimization, not a correctness requirement.
                    self.mm(enc, "plain", &gate_n, &self.st.h, &self.st.gate, d, nffn, None);
                    self.mm(enc, "plain", &up_n, &self.st.h, &self.st.up, d, nffn, None);
                    self.bar(enc);
                    enc.set_compute_pipeline_state(&self.p["ffn_gu_split"]);
                    enc.set_buffer(0, Some(&self.st.gate), 0);
                    enc.set_buffer(1, Some(&self.st.up), 0);
                    enc.set_buffer(2, Some(&self.st.act), 0);
                    enc.set_bytes(3, 4, &nffn as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &act as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(((nffn + 255) / 256) as u64, 1, 1),
                                               MTLSize::new(256, 1, 1));
                } else if let Some((kernel, map)) = native_k {
                    enc.set_compute_pipeline_state(&self.p[kernel]);
                    enc.set_buffer(0, Some(&self.st.h), 0);
                    enc.set_buffer(1, Some(&map[&gate_n]), self.wt.w_off.get(&gate_n).copied().unwrap_or(0));
                    enc.set_buffer(2, Some(&map[&up_n]), self.wt.w_off.get(&up_n).copied().unwrap_or(0));
                    enc.set_buffer(3, Some(&self.st.act), 0);
                    enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                    enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(((nffn + 1) / 2) as u64, 1, 1), MTLSize::new(64, 1, 1));
                } else if self.wt.q20 {
                    enc.set_compute_pipeline_state(&self.p["ffn_gu_q20"]);
                    enc.set_buffer(0, Some(&self.st.h), 0);
                    enc.set_buffer(1, Some(&self.wt.w20[&p("ffn_gate.weight")]), 0);
                    enc.set_buffer(2, Some(&self.wt.w20[&p("ffn_up.weight")]), 0);
                    enc.set_buffer(3, Some(&self.st.act), 0);
                    enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                    enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
                    enc.set_buffer(6, Some(&self.wt.s20[&p("ffn_gate.weight")]), 0);
                    enc.set_buffer(7, Some(&self.wt.s20[&p("ffn_up.weight")]), 0);
                    enc.dispatch_thread_groups(MTLSize::new(((nffn + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
                } else if self.wt.q4 {
                    enc.set_compute_pipeline_state(&self.p["ffn_gu_q4"]);
                    enc.set_buffer(0, Some(&self.st.h), 0);
                    enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_gate.weight")]), 0);
                    enc.set_buffer(2, Some(&self.wt.w4[&p("ffn_up.weight")]), 0);
                    enc.set_buffer(3, Some(&self.st.act), 0);
                    enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                    enc.set_buffer(6, Some(&self.wt.scale4[&p("ffn_gate.weight")]), 0);
                    enc.set_buffer(7, Some(&self.wt.scale4[&p("ffn_up.weight")]), 0);
                    enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
                    // qmv_fast style: 8 rows/tg, 64 threads (fixed — see mm() note).
                    enc.dispatch_thread_groups(MTLSize::new((nffn / 8) as u64, 1, 1), MTLSize::new(64, 1, 1));
                } else {
                    enc.set_compute_pipeline_state(&self.p[if self.wt.q8 { "ffn_gu_q8" } else { "ffn_gu_f16" }]);
                    enc.set_buffer(0, Some(&self.st.h), 0);
                    enc.set_buffer(1, Some(if self.wt.q8 { &self.wt.w8[&p("ffn_gate.weight")] } else { &self.wt.w16[&p("ffn_gate.weight")] }), 0);
                    enc.set_buffer(2, Some(if self.wt.q8 { &self.wt.w8[&p("ffn_up.weight")] } else { &self.wt.w16[&p("ffn_up.weight")] }), 0);
                    enc.set_buffer(3, Some(&self.st.act), 0);
                    enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                    if self.wt.q8 {
                        enc.set_buffer(6, Some(&self.wt.scale8[&p("ffn_gate.weight")]), 0);
                        enc.set_buffer(7, Some(&self.wt.scale8[&p("ffn_up.weight")]), 0);
                        enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
                    } else {
                        enc.set_bytes(6, 4, &act as *const u32 as *const c_void);
                    }
                    let ft = if self.wt.q8 { self.tune.gemv_plan.get(&(0, 1)).map(|p| p.threads).unwrap_or(256) } else { 256 };
                    let frows = (ft / 32).max(1) as u32;
                    enc.dispatch_thread_groups(MTLSize::new(((nffn + frows - 1) / frows) as u64, 1, 1), MTLSize::new(ft, 1, 1));
                }
            }
            self.bar(enc); // SwiGLU activation ready
            // ffn_down. Gemma (sandwich): tmp = Wdown@act; x = (x + post_ffw_norm(tmp)) * out_scale
            // — the per-layer output scale (gemma4) folds into the fused rmsnorm_add.
            if self.arch.sandwich {
                let oscale = self.arch.out_scale.get(l).copied().unwrap_or(1.0);
                self.mm(enc, "plain", &p("ffn_down.weight"), &self.st.act, &self.st.tmp, self.arch.ffn as u32, d, None);
                self.enc_reduce(enc, "rmsnorm_add", &[(&self.st.x, 0), (&self.st.tmp, 1), (&self.wt.w32[&p("post_ffw_norm.weight")], 2)], &[(3, d)], &[(4, self.arch.eps), (5, oscale)], 1, 256);
            } else {
                self.mm(enc, "accum", &p("ffn_down.weight"), &self.st.act, &self.st.x, self.arch.ffn as u32, d, None);
            }
        }
        // final norm + lm head — last pipeline stage only (middle stages leave the
        // hidden state in self.st.x for the host to read out and forward downstream)
        self.bar(enc); // last layer's x complete
        if !do_head { return; }
        if self.sp.mtp.is_some() {
            // capture the pre-output-norm hidden for the NextN/MTP draft block
            self.enc_reduce(enc, "copy_buf", &[(&self.sp.mtp_h, 0), (&self.st.x, 1)], &[(2, d)], &[], ((d + 63)/64) as u64, 64);
        }
        self.enc_reduce(enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32["output_norm.weight"], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
        self.bar(enc);
        let lm = self.arch.lm_head.clone();
        self.mm(enc, "plain", &lm, &self.st.h, &self.st.logits, d, self.arch.vocab as u32, None);
    }



    /// The projection dispatch for a B-row slot batch.
    ///
    /// The right kernel for B independent sequences is not the right kernel for a
    /// B-token prefill chunk, even though the arithmetic is identical. `chunk_gemm` is
    /// tuned for the chunk sizes prefill uses (M = 256), and two of its arms are poor
    /// at the M = 1..4 a slot batch lives in:
    ///
    /// * M = 1. Every M-row kernel carries per-batch machinery — `float p[8]`
    ///   accumulator arrays, row loops, tile padding — and at one row pays for all of
    ///   it while using a quarter. Measured on surya-2/Q4, the slot graph at B=1 through
    ///   `chunk_gemm` was slower than at B=2. B=1 therefore goes through `mm()`, the
    ///   single-row path `forward_id` itself uses, which keeps the B=1 row of the gate's
    ///   table an honest decode baseline.
    /// * F16 below M=8. `chunk_gemm`'s F16 arm takes its tile GEMM only at M >= 8 and
    ///   otherwise loops a per-row `gemv_f16`, which re-reads the entire matrix once per
    ///   row — the cost slot batching exists to remove, so routing through it makes F16
    ///   slots ~1.0x, and F16 is the `ocr` default precision (`flags.rs`). `gemv_m_f16`
    ///   reads each weight row once for all M rows and measured 2.24x/M=4.
    ///
    /// Everything else defers to `chunk_gemm` unchanged, so Q4/Q4L/Q8/native keep the
    /// routing tuned for them.
    fn slot_gemm(&self, enc: &metal::ComputeCommandEncoderRef, b: u32, accum: bool,
                 wname: &str, x: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32) {
        if b == 1 {
            self.mm(enc, if accum { "accum" } else { "plain" }, wname, x, y, k, n, None);
            return;
        }
        // `gemv_m_f16` holds `float p[8]`, hence M <= 8, and loads `float4`/`half4`,
        // hence K % 4. Both limits are the kernel's.
        if b <= 8 && k % 4 == 0 && self.wt.repr(wname) == Repr::F16 {
            if let Some(w) = self.wt.w16.get(wname) {
                // There is no `gemv_m_f16_accum`, so the three accumulating projections
                // (attn_output, ssm_out, ffn_down — the widest weights in the layer, and
                // so the ones that most want a single read) land in scratch and are
                // added rather than falling back to the per-row loop. Two dispatches
                // over b*d floats is far less traffic than re-reading a [k, n] matrix
                // b times.
                //
                // `st.gate` is that scratch: `ffn * MAXM` floats, so it holds b*n for
                // every accumulating shape here (all have n = d), and it is dead at each
                // of the three sites — the mixer runs before the FFN writes it, and
                // `silu_mul` has consumed it into `st.act` by the time `ffn_down` runs.
                let dst = if accum { &self.st.gate } else { y };
                enc.set_compute_pipeline_state(&self.p["gemv_m_f16"]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(dst), 0);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &b as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((n + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
                if accum {
                    debug_assert!((b * n) as u64 * 4 <= self.st.gate.length(),
                        "slot_gemm f16 accum scratch overruns st.gate");
                    self.bar(enc);   // the add reads what the GEMV just wrote
                    self.enc_reduce(enc, "add_inplace", &[(y, 0), (&self.st.gate, 1)],
                        &[(2, b * n)], &[], ((b * n + 63) / 64) as u64, 64);
                }
                return;
            }
        }
        // Plain Q4 (`w4`) has a tuned two-row kernel and then nothing until the M>=8
        // tile, so M=3 and M=4 drop to the generic `gemv_m_q4`, where the slot speedup
        // collapses (B=3 measured 0.59-1.03x, slower than decoding the sequences one at
        // a time). `ceil(b/2)` two-row dispatches re-read the weight twice and still win
        // by a wide margin; see `q4_pair` for the numbers.
        if (3..=4).contains(&b) && q4_pair() && self.wt.repr(wname) == Repr::Q4 {
            if let (Some(w), Some(sc)) = (self.wt.w4.get(wname), self.wt.scale4.get(wname)) {
                // The kernel always writes two rows. At b=3 the second row of the last
                // pair is a phantom: it reads row 3 of the activation scratch and writes
                // row 3 of the output, both inside buffers sized for MAXM=256 rows and
                // neither read back.
                for pair in 0..b.div_ceil(2) {
                    let r0 = 2 * pair;
                    enc.set_compute_pipeline_state(&self.p[if accum { "gemv_m2_q4_accum" } else { "gemv_m2_q4" }]);
                    enc.set_buffer(0, Some(x), (r0 as u64) * (k as u64) * 4);
                    enc.set_buffer(1, Some(w), 0);
                    enc.set_buffer(2, Some(y), (r0 as u64) * (n as u64) * 4);
                    enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                    enc.set_buffer(5, Some(sc), 0);
                    enc.dispatch_thread_groups(MTLSize::new(((n + 7) / 8) as u64, 1, 1), MTLSize::new(64, 1, 1));
                }
                return;
            }
        }
        self.chunk_gemm(enc, b, accum, wname, x, y, k, n);
    }

    /// Decode one token for each of B independent sequences in a single pass — the
    /// encode half of [`ojas_core::Model::decode_slots`].
    ///
    /// `steps` is `(slot, token, pos)` per sequence; row `i` of every activation
    /// buffer belongs to `steps[i]`, and `st.tmp[i]` receives its argmax.
    ///
    /// A decode step reads all ~353 MB of Q4L weights (surya-2) to produce one token.
    /// At B slots it reads them once and produces B, because every projection goes
    /// through the M-row kernel family with M = B. Measured on an M2 Max: 1.84x on the
    /// weight-read term at B=4, ~2.0x on the whole token. Dispatch count is the other
    /// half of the win — a Q4 `ffn_gu` does 10.8 us of work in a 16.8 us dispatch, so
    /// ~5-6 us of each of the ~290 dispatches per token is launch and ramp. Batching
    /// multiplies the work per dispatch without multiplying the weight reads or the
    /// dispatch count, so surya-2 at Q4 gains even though its GEMVs already run at
    /// 382-387 GB/s against a measured 379.8 GB/s roof.
    ///
    /// Three classes of work appear here:
    ///
    /// * Weight projections — one dispatch at M=B. This is the part that amortizes,
    ///   and it needs no new kernel: the M-row family indexes `x[m*K + k]` and is
    ///   agnostic to whether the M rows are M tokens of one sequence or one token of
    ///   each of M sequences.
    /// * Row-wise elementwise work (`rmsnorm_m`, `ssm_ab`, `gated_rmsnorm`,
    ///   `qgate_split`, `qk_rmsnorm`, `silu_mul`) — one dispatch, B rows, since these
    ///   already take a row count and the rows are contiguous.
    /// * Per-sequence state work (`conv1d_decode`, `deltanet_fused`, `rope_qk_store`,
    ///   attention) — B dispatches with buffer offsets. These cannot fuse: each
    ///   carries a recurrent state or a KV history private to its sequence, and the
    ///   attention kernels take `seq` as a scalar `constant uint&`, so two slots at
    ///   different positions cannot share a launch. B extra dispatches on 18 + 6
    ///   layers is ~90 of ~290, and they carry no weight traffic.
    ///
    /// None of this needs a Metal kernel edit: slots are expressed entirely in buffer
    /// offsets and row counts the existing kernels already read.
    ///
    /// `l_start`/`l_end`/`do_embed`/`do_head` split the pass across two command buffers
    /// as `encode_forward_span` does for one sequence: the first is committed while the
    /// CPU still encodes the second, so the GPU is not idle for the ~0.8 ms of encode.
    /// Slots make that split more valuable, since the per-sequence dispatches add ~90
    /// to the ~290 a single token issues.
    pub(crate) fn encode_slots(&self, enc: &metal::ComputeCommandEncoderRef, steps: &[(usize, u32, usize)],
                               l_start: usize, l_end: usize, do_embed: bool, do_head: bool) {
        let sc = self.arch.ssm.expect("encode_slots is the qwen35 hybrid graph");
        let b = steps.len() as u32;
        let d = self.d as u32;
        let f4 = 4u64;
        let eps = self.arch.eps;
        // Row i of every activation buffer is steps[i]. `slot` addresses state
        // (recurrent + KV, allocated per slot at load); `i` addresses the activation
        // row (scratch, allocated per MAXM and so already wide enough). The two indices
        // are not interchangeable: a caller may pass slots out of order or pass a
        // subset, and conflating them cross-wires the batch while still producing
        // plausible tokens.
        let row = |i: usize, width: u32| (i as u64) * (width as u64) * f4;

        // ---- embed: one gather per row into its own row of x ----
        if do_embed {
            for (i, &(_, tok, _)) in steps.iter().enumerate() {
                self.embed_named_off(enc, "token_embd.weight", tok, d, row(i, d));
            }
        }
        self.bar(enc); // x rows ready (gathered, or carried from the first buffer)

        for l in l_start..l_end {
            let p = |s: &str| format!("blk.{l}.{s}");
            let lp = self.arch.layers[l];
            self.rmsnorm_rows(enc, b, &self.wt.w32[&p("attn_norm.weight")]);
            self.bar(enc); // h ready for the projections
            if lp.is_ssm {
                let (s_st, hk, hv) = (sc.d_state, sc.n_group, sc.dt_rank);
                let d_inner = sc.d_inner;
                let conv_ch = d_inner + 2 * hk * s_st;
                let conv_k = sc.conv_kernel;
                let head_v = d_inner / hv;
                // The four projections: one dispatch each, B rows. This is where the
                // weight read amortizes.
                self.slot_gemm(enc, b, false, &p("attn_qkv.weight"), &self.st.h, &self.st.ssm_qkv, d, conv_ch);
                self.slot_gemm(enc, b, false, &p("attn_gate.weight"), &self.st.h, &self.st.ssm_z, d, d_inner);
                self.slot_gemm(enc, b, false, &p("ssm_alpha.weight"), &self.st.h, &self.st.ssm_gate, d, hv);
                self.slot_gemm(enc, b, false, &p("ssm_beta.weight"), &self.st.h, &self.st.ssm_beta, d, hv);
                self.bar(enc); // projections done (all four read h, ran concurrently)
                // `ssm_ab` is elementwise over [rows, hv] and indexes its dt/a weights
                // modulo hv, so B rows are one dispatch with n = B*hv — exactly what
                // the chunk graph passes as m*hv.
                self.enc_reduce(enc, "ssm_ab",
                    &[(&self.st.ssm_gate, 0), (&self.st.ssm_beta, 1),
                      (&self.wt.w32[&p("ssm_dt.bias")], 2), (&self.wt.w32[&p("ssm_a")], 3)],
                    &[(4, b * hv), (5, hv)], &[], ((b * hv + 63) / 64) as u64, 64);
                // Per-slot from here to the gated norm: both kernels carry state.
                for (i, &(slot, _, _)) in steps.iter().enumerate() {
                    enc.set_compute_pipeline_state(&self.p["conv1d_decode"]);
                    enc.set_buffer(0, Some(&self.st.ssm_qkv), row(i, conv_ch));
                    enc.set_buffer(1, Some(&self.st.conv_state[l]), self.conv_slot_off(l, slot));
                    enc.set_buffer(2, Some(&self.wt.w32[&p("ssm_conv1d.weight")]), 0);
                    enc.set_bytes(3, 4, &conv_ch as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &conv_k as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(((conv_ch + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
                }
                self.bar(enc); // conv + ssm_ab done
                for (i, &(slot, _, _)) in steps.iter().enumerate() {
                    // M=1 per slot. The recurrence is token-serial inside the kernel, so
                    // B slots cannot become one M=B launch the way a chunk of B
                    // consecutive tokens can — those share a state, these do not.
                    let soff = self.ssm_slot_off(l, slot);
                    enc.set_compute_pipeline_state(&self.p["deltanet_fused"]);
                    enc.set_buffer(0, Some(&self.st.ssm_state[l]), soff);
                    enc.set_buffer(1, Some(&self.st.ssm_qkv), row(i, conv_ch));
                    enc.set_buffer(2, Some(&self.st.ssm_gate), row(i, hv));
                    enc.set_buffer(3, Some(&self.st.ssm_beta), row(i, hv));
                    enc.set_buffer(4, Some(&self.st.ssm_o), row(i, d_inner));
                    enc.set_bytes(5, 4, &s_st as *const u32 as *const c_void);
                    enc.set_bytes(6, 4, &hk as *const u32 as *const c_void);
                    enc.set_bytes(7, 4, &hv as *const u32 as *const c_void);
                    enc.set_bytes(8, 4, &conv_ch as *const u32 as *const c_void);
                    enc.set_bytes(9, 4, &1u32 as *const u32 as *const c_void);
                    enc.set_bytes(10, 4, &eps as *const f32 as *const c_void);
                    // Snapshots disabled (snap_t = UINT_MAX), so buffer 11 is never
                    // dereferenced, but it must still be bound; binding it to the state
                    // at the same offset keeps a stray write inside this slot.
                    enc.set_buffer(11, Some(&self.st.ssm_state[l]), soff);
                    enc.set_bytes(12, 4, &u32::MAX as *const u32 as *const c_void);
                    let kmap_div: u32 = self.cfg.moe_kmap_div as u32;
                    enc.set_bytes(13, 4, &kmap_div as *const u32 as *const c_void);
                    enc.set_bytes(14, 4, &0u32 as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new((s_st / 4) as u64, hv as u64, 1), MTLSize::new(128, 1, 1));
                }
                self.bar(enc); // deltanet done
                // Row-wise again: gid.y is the row and `rs` is the row stride.
                enc.set_compute_pipeline_state(&self.p["gated_rmsnorm"]);
                enc.set_buffer(0, Some(&self.st.ssm_o), 0);
                enc.set_buffer(1, Some(&self.wt.w32[&p("ssm_norm.weight")]), 0);
                enc.set_buffer(2, Some(&self.st.ssm_z), 0);
                enc.set_bytes(3, 4, &head_v as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
                enc.set_bytes(5, 4, &d_inner as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &0u32 as *const u32 as *const c_void);   // qwen35: silu gate
                enc.dispatch_thread_groups(MTLSize::new(hv as u64, b as u64, 1), MTLSize::new(32, 1, 1));
                self.bar(enc); // gated norm done
                self.slot_gemm(enc, b, true, &p("ssm_out.weight"), &self.st.ssm_o, &self.st.x, d_inner, d);
            } else {
                let (hd, kvdim, qdim) = (lp.head_dim, lp.kvdim, lp.qdim);
                let group = lp.n_head / lp.n_kv.max(1);
                let (nq, nk) = (lp.n_head, lp.n_kv);
                self.slot_gemm(enc, b, false, &p("attn_q.weight"), &self.st.h, &self.st.ssm_qkv, d, 2 * qdim);
                self.slot_gemm(enc, b, false, &p("attn_k.weight"), &self.st.h, &self.st.k, d, kvdim);
                self.slot_gemm(enc, b, false, &p("attn_v.weight"), &self.st.h, &self.st.v, d, kvdim);
                self.bar(enc); // q/k/v projections done
                self.enc_reduce(enc, "qgate_split", &[(&self.st.ssm_qkv, 0), (&self.st.q, 1)],
                    &[(2, hd), (3, qdim), (4, b)], &[], ((b * qdim + 63) / 64) as u64, 64);
                self.bar(enc); // q split done
                enc.set_compute_pipeline_state(&self.p["qk_rmsnorm"]);
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.st.k), 0);
                enc.set_buffer(2, Some(&self.wt.w32[&p("attn_q_norm.weight")]), 0);
                enc.set_buffer(3, Some(&self.wt.w32[&p("attn_k_norm.weight")]), 0);
                enc.set_bytes(4, 4, &hd as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &nq as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &nk as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &eps as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new((nq + nk) as u64, b as u64, 1), MTLSize::new(32, 1, 1));
                self.bar(enc); // qk-norm done
                // Rope + KV store, per slot. Two things are per-slot: the position (each
                // sequence is at its own `pos`, so the rope angle differs) and the
                // destination (each sequence owns its own KV region). The destination is
                // a buffer offset because the kernel writes `off + ...` relative to the
                // bound base; `off` stays the within-slot row `pos*kvdim`, keeping it far
                // from overflowing its u32.
                let (totq, totk) = (qdim / 2, kvdim / 2);
                for (i, &(slot, _, pos)) in steps.iter().enumerate() {
                    let kvo = self.kv_slot_off(l, slot);
                    self.enc_reduce_off(enc, "rope_qk_store",
                        &[(&self.st.q, 0, row(i, qdim)), (&self.st.k, 1, row(i, kvdim)), (&self.st.v, 2, row(i, kvdim)),
                          (&self.st.kcache[l], 3, kvo), (&self.st.vcache[l], 4, kvo)],
                        &[(5, hd), (6, pos as u32), (8, totq), (9, totk), (10, kvdim),
                          (11, pos as u32 * kvdim), (12, 1), (13, sc.n_rot)],
                        &[(7, lp.rope_base)],
                        (((totq + totk + kvdim) + 63) / 64) as u64, 64);
                }
                self.bar(enc); // rope + cache store done
                // Attention, per slot: `seq` is a scalar `constant uint&`, so slots at
                // different positions are structurally different launches.
                let attn_t = self.tune.gemv_plan.get(&(0, 2)).map(|pp| pp.threads).unwrap_or(64).min(256);
                let short: Vec<bool> = steps.iter().map(|&(_, _, pos)| (pos as u32 + 1) <= 512).collect();
                for (i, &(slot, _, pos)) in steps.iter().enumerate() {
                    if !short[i] { continue; }
                    let kvo = self.kv_slot_off(l, slot);
                    super::attn_log("slots", "attention_short", pos as u32 + 1);
                    self.enc_reduce_off(enc, "attention_short",
                        &[(&self.st.q, 0, row(i, qdim)), (&self.st.kcache[l], 1, kvo),
                          (&self.st.vcache[l], 2, kvo), (&self.st.attn, 3, row(i, qdim))],
                        &[(4, hd), (5, kvdim), (6, pos as u32 + 1), (7, group)], &[(8, lp.scale)],
                        lp.n_head as u64, attn_t);
                }
                if short.iter().any(|s| !s) {
                    // Flash-decoding for the long slots. Split into all-parts, barrier,
                    // all-merges so the B partial passes overlap instead of serializing
                    // behind one another's merge — each writes its own `attn_part`
                    // region, which is the one arena buffer slots had to widen.
                    let mut nwgs = vec![0u32; steps.len()];
                    for (i, &(slot, _, pos)) in steps.iter().enumerate() {
                        if short[i] { continue; }
                        let seq = pos as u32 + 1;
                        let nwg = ((seq as usize + 1023) / 1024).clamp(2, ojas_metal::kernels::attn::ATTN_NWG) as u32;
                        nwgs[i] = nwg;
                        let kvo = self.kv_slot_off(l, slot);
                        super::attn_log("slots", "attn_flash (flash-decoding)", seq);
                        enc.set_compute_pipeline_state(&self.p["attention_part"]);
                        enc.set_buffer(0, Some(&self.st.q), row(i, qdim));
                        enc.set_buffer(1, Some(&self.st.kcache[l]), kvo);
                        enc.set_buffer(2, Some(&self.st.vcache[l]), kvo);
                        enc.set_buffer(3, Some(&self.st.attn_part), self.part_slot_off(slot));
                        enc.set_bytes(4, 4, &hd as *const u32 as *const c_void);
                        enc.set_bytes(5, 4, &kvdim as *const u32 as *const c_void);
                        enc.set_bytes(6, 4, &seq as *const u32 as *const c_void);
                        enc.set_bytes(7, 4, &group as *const u32 as *const c_void);
                        enc.set_bytes(8, 4, &lp.scale as *const f32 as *const c_void);
                        enc.set_bytes(9, 4, &nwg as *const u32 as *const c_void);
                        enc.dispatch_thread_groups(MTLSize::new(lp.n_head as u64, nwg as u64, 1), MTLSize::new(256, 1, 1));
                    }
                    self.bar(enc); // partials complete
                    for (i, &(slot, _, _)) in steps.iter().enumerate() {
                        if short[i] { continue; }
                        enc.set_compute_pipeline_state(&self.p["attention_merge"]);
                        enc.set_buffer(0, Some(&self.st.attn_part), self.part_slot_off(slot));
                        enc.set_buffer(1, Some(&self.st.attn), row(i, qdim));
                        enc.set_bytes(2, 4, &hd as *const u32 as *const c_void);
                        enc.set_bytes(3, 4, &nwgs[i] as *const u32 as *const c_void);
                        enc.dispatch_thread_groups(MTLSize::new(lp.n_head as u64, 1, 1), MTLSize::new(256, 1, 1));
                    }
                }
                self.bar(enc); // attention done
                self.enc_reduce(enc, "gate_mul_sigmoid", &[(&self.st.attn, 0), (&self.st.ssm_qkv, 1)],
                    &[(2, hd), (3, qdim), (4, b)], &[], ((b * qdim + 63) / 64) as u64, 64);
                self.bar(enc); // gate applied
                self.slot_gemm(enc, b, true, &p("attn_output.weight"), &self.st.attn, &self.st.x, qdim, d);
            }
            // FFN — post_attention_norm is the pre-FFN norm on qwen35, shared by the
            // recurrent and attention layers. Two projections + silu_mul + down, the
            // same concurrent pipeline the chunk graph uses; both tile-fusions tried
            // there measured slower.
            self.bar(enc); // mixer residual in x
            self.rmsnorm_rows(enc, b, &self.wt.w32[&p("post_attention_norm.weight")]);
            self.bar(enc); // h ready for FFN
            let nffn = self.arch.ffn as u32;
            self.slot_gemm(enc, b, false, &p("ffn_gate.weight"), &self.st.h, &self.st.gate, d, nffn);
            self.slot_gemm(enc, b, false, &p("ffn_up.weight"), &self.st.h, &self.st.up, d, nffn);
            self.bar(enc); // gate + up done
            self.enc_reduce(enc, "silu_mul", &[(&self.st.gate, 0), (&self.st.up, 1), (&self.st.act, 2)],
                &[(3, b * nffn)], &[], ((b * nffn + 63) / 64) as u64, 64);
            self.bar(enc); // SwiGLU act ready
            self.slot_gemm(enc, b, true, &p("ffn_down.weight"), &self.st.act, &self.st.x, nffn, d);
            self.bar(enc); // layer output in x
        }
        // ---- head: one norm, one lm_head at M=B, then a per-row argmax ----
        if !do_head { return; }
        self.rmsnorm_rows(enc, b, &self.wt.w32["output_norm.weight"]);
        self.bar(enc);
        let lm = self.arch.lm_head.clone();
        self.slot_gemm(enc, b, false, &lm, &self.st.h, &self.st.logits, d, self.arch.vocab as u32);
        self.bar(enc);
        // tmp[i] = row i's argmax, indexed by position in `steps`, not by slot.
        for i in 0..steps.len() {
            self.enc_reduce_off(enc, "argmax",
                &[(&self.st.logits, 0, row(i, self.arch.vocab as u32)), (&self.st.tmp, 1, (i as u64) * 4)],
                &[(2, self.arch.vocab as u32)], &[], 1, self.tune.max_tg.min(1024));
        }
    }

}
