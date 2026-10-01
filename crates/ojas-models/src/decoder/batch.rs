#![allow(clippy::too_many_arguments)]
use super::*;
use objc::{msg_send, sel, sel_impl};
use metal::{MTLResourceOptions, MTLSize};
use std::ffi::c_void;
 // re-export

impl<'a> DecoderGpu<'a> {
    /// Multi-token forward: process `tokens` (≤ MAXM) at positions
    /// base_pos..base_pos+M in one pass. Returns M logit vectors. Q8 only.
    /// KV cache is written for all M positions. This is the speculative-verify
    /// forward, near the cost of a single-token forward on a latency-bound GPU.
    pub fn forward_batch(&self, tokens: &[u32], base_pos: usize) -> Vec<Vec<f32>> {
        self.forward_batch_impl(tokens, base_pos, LogitsOut::Host, false)
    }

    /// Masked-diffusion forward: full bidirectional pass, all per-position logits (for
    /// validation). The decode loop uses forward_diffusion_range for windowed logits.
    pub fn forward_diffusion(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        self.forward_diffusion_range(tokens, 0, 0)
    }
    /// Fast-dLLM KV cache: recompute only positions `[prefix..end]`, reuse cached prefix K/V.
    pub fn forward_diffusion_suffix(&self, tokens: &[u32], prefix: usize) -> Vec<Vec<f32>> {
        self.forward_diffusion_range(tokens, prefix, prefix)
    }

    /// Core masked-diffusion forward. Recompute positions `[compute_from..end]` (0 = full;
    /// >0 reuses the cached prefix K/V for `[0..compute_from)`). Attention keys span the full
    /// length so suffix queries still attend to the cached prefix. Emit logits only for
    /// `[logit_from..end]` (keeps logits small for long prompts). Working buffers are allocated
    /// on demand when `m > MAXM`, so this handles long prompts (up to ~4096 total tokens — the
    /// attention_m_short_bidir key cap; >4096 needs a streaming bidir attention). Dense qwen2.
    pub fn forward_diffusion_range(&self, tokens: &[u32], compute_from: usize, logit_from: usize) -> Vec<Vec<f32>> {
        let total = tokens.len() as u32;
        let m = (tokens.len() - compute_from) as u32;
        let lf = logit_from.max(compute_from);
        let ln = (tokens.len() - lf) as u32;               // logit rows
        let d = self.d as u32; let hd = self.arch.hd as u32;
        let kvdim = (self.arch.n_kv * self.arch.hd) as u32;
        let group = (self.arch.n_head / self.arch.n_kv) as u32;
        let scale = 1.0 / (self.arch.hd as f32).sqrt();
        let nh = self.arch.n_head as u32; let qdim = nh * hd; let ff = self.arch.ffn as u32;
        let vocab = self.arch.vocab as u32;
        // working buffers: reuse the MAXM-sized instance buffers when they fit, else allocate
        // M-sized temporaries (one-time for a long-prompt prefill; suffix steps stay small).
        let big = m as usize > MAXM;
        // GEMM outputs are stored in full 32-token tiles (cooperative simdgroup_store
        // can't skip sub-tile rows), so any M-sized temporary must have its row count
        // padded to a multiple of 32 or the store overruns. The MAXM base buffers are
        // already 256-row (multiple of 32); only the on-demand big-M ones need padding.
        let pad32 = |r: usize| (r + 31) / 32 * 32;
        let mrows = if big { pad32(m as usize) } else { m as usize };
        let mk = |sz: usize, base: &metal::Buffer| if big { buf(self.gpu, sz) } else { base.clone() };
        let (x, h) = (mk(mrows * self.d, &self.st.x), mk(mrows * self.d, &self.st.h));
        let q = mk(mrows * qdim as usize, &self.st.q);
        let k = mk(mrows * kvdim as usize, &self.st.k);
        let v = mk(mrows * kvdim as usize, &self.st.v);
        let attn = mk(mrows * qdim as usize, &self.st.attn);
        let gate = mk(mrows * self.arch.ffn, &self.st.gate);
        let up = mk(mrows * self.arch.ffn, &self.st.up);
        let act = mk(mrows * self.arch.ffn, &self.st.act);
        let logits = if ln as usize > MAXM { buf(self.gpu, pad32(ln as usize) * self.arch.vocab) } else { self.st.logits.clone() };

        let suffix = &tokens[compute_from..];
        let tokbuf = self.gpu.device.new_buffer_with_data(
            suffix.as_ptr() as *const c_void, (suffix.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        if self.wt.q8 {
            self.enc_reduce(&enc, "embed_m_q8",
                &[(&self.wt.w8["token_embd.weight"], 0), (&x, 1), (&tokbuf, 3), (&self.wt.scale8["token_embd.weight"], 4)],
                &[(2, d), (5, m)], &[], (((m * d) + 63) / 64) as u64, 64);
        } else {
            self.enc_reduce(&enc, "embed_m_f16",
                &[(&self.wt.w16["token_embd.weight"], 0), (&x, 1), (&tokbuf, 3)],
                &[(2, d), (5, m)], &[], (((m * d) + 63) / 64) as u64, 64);
        }
        for l in 0..self.arch.n_layers {
            let p = |s: &str| format!("blk.{l}.{s}");
            self.enc_reduce(&enc, "rmsnorm_m", &[(&x, 0), (&self.wt.w32[&p("attn_norm.weight")], 1), (&h, 2)], &[(3, d)], &[(4, self.arch.eps)], m as u64, 256);
            self.projm(&enc, &h, 0, &p("attn_q.weight"), &q, d, qdim, m, false);
            self.projm(&enc, &h, 0, &p("attn_k.weight"), &k, d, kvdim, m, false);
            self.projm(&enc, &h, 0, &p("attn_v.weight"), &v, d, kvdim, m, false);
            self.enc_reduce(&enc, "add_rowbias_m", &[(&q,0),(&self.wt.w32[&p("attn_q.bias")],1)], &[(2,qdim),(3,m*qdim)], &[], ((m*qdim+63)/64) as u64, 64);
            self.enc_reduce(&enc, "add_rowbias_m", &[(&k,0),(&self.wt.w32[&p("attn_k.bias")],1)], &[(2,kvdim),(3,m*kvdim)], &[], ((m*kvdim+63)/64) as u64, 64);
            self.enc_reduce(&enc, "add_rowbias_m", &[(&v,0),(&self.wt.w32[&p("attn_v.bias")],1)], &[(2,kvdim),(3,m*kvdim)], &[], ((m*kvdim+63)/64) as u64, 64);
            let aq = nh * hd / 2; let ak = kvdim / 2;
            self.enc_reduce(&enc, "rope_qk_store_m",
                &[(&q,0),(&k,1),(&v,2),(&self.st.kcache[l],3),(&self.st.vcache[l],4),(&self.st.k,14)],
                &[(5,hd),(6,compute_from as u32),(8,aq),(9,ak),(10,kvdim),(11,m),(12,self.arch.rope_neox as u32),(13,hd)], &[(7,self.arch.rope_base)],
                (((m*(aq+ak+kvdim))+63)/64) as u64, 64);
            // ≤4096 keys: two-pass short kernel (sc[]); beyond: streaming (unbounded ctx).
            // OJAS_DIFFUSION_STREAM forces streaming (for validation vs the short kernel).
            let attn_kern = if total > 4096 || ojas_core::config::var("OJAS_DIFFUSION_STREAM").is_ok()
                { "attention_m_bidir" } else { "attention_m_short_bidir" };
            self.enc_reduce(&enc, attn_kern,
                &[(&q,0),(&self.st.kcache[l],1),(&self.st.vcache[l],2),(&attn,3)],
                &[(4,hd),(5,kvdim),(6,total),(7,group),(9,nh)], &[(8,scale)], (m*nh) as u64, 256);
            // K is qdim (n_head*hd), not d. They coincide on Qwen2 and diverge on
            // Qwen3 (qdim 2048 vs d 1024), where passing d makes o_proj consume
            // half the attention output.
            self.projm(&enc, &attn, 0, &p("attn_output.weight"), &x, qdim, d, m, true);
            self.enc_reduce(&enc, "rmsnorm_m", &[(&x, 0), (&self.wt.w32[&p("ffn_norm.weight")], 1), (&h, 2)], &[(3, d)], &[(4, self.arch.eps)], m as u64, 256);
            self.projm(&enc, &h, 0, &p("ffn_gate.weight"), &gate, d, ff, m, false);
            self.projm(&enc, &h, 0, &p("ffn_up.weight"), &up, d, ff, m, false);
            self.enc_reduce(&enc, "silu_mul", &[(&gate, 0), (&up, 1), (&act, 2)], &[(3, m*ff)], &[], ((m*ff + 63)/64) as u64, 64);
            self.projm(&enc, &act, 0, &p("ffn_down.weight"), &x, ff, d, m, true);
        }
        self.enc_reduce(&enc, "rmsnorm_m", &[(&x, 0), (&self.wt.w32["output_norm.weight"], 1), (&h, 2)], &[(3, d)], &[(4, self.arch.eps)], m as u64, 256);
        // lm_head over only the logit window [lf..end]: h rows [lf-compute_from ..], ln rows.
        let off = ((lf - compute_from) as u64) * (d as u64) * 4;
        let lmh = self.arch.lm_head.clone();
        self.projm(&enc, &h, off, &lmh, &logits, d, vocab, ln, false);
        enc.end_encoding();
        let _ = ojas_metal::commit_and_wait_checked(cb, "batched forward");
        let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        self.gpu_s.set(self.gpu_s.get() + (ge - gs));
        let ptr = logits.contents() as *const f32;
        let all = unsafe { std::slice::from_raw_parts(ptr, self.arch.vocab * ln as usize) };
        (0..ln as usize).map(|i| all[i * self.arch.vocab..(i + 1) * self.arch.vocab].to_vec()).collect()
    }

    /// Batched forward; `want_logits=false` skips the lm_head + readback (prefill,
    /// where only the KV cache matters). gpt-oss (MoE + sinks + SwiGLU-OAI) is
    /// handled inline via the batched OAI kernels.
    pub(crate) fn forward_batch_impl(&self, tokens: &[u32], base_pos: usize, logits: LogitsOut, bidir: bool) -> Vec<Vec<f32>> {
        assert!(base_pos.checked_add(tokens.len()).is_some_and(|n| n <= self.st.max_seq)
            && tokens.iter().all(|&t| (t as usize) < self.arch.vocab), "batch exceeds model bounds");
        // OJAS_SKIP=<cat>[,<cat>] drops a category from the real concurrent pass.
        // Numerically wrong on purpose: the serial per-category profiler cannot
        // price a category inside a concurrent encoder, so the only way to learn its
        // critical-path cost is to remove it and watch the wall.
        // cats: attn qkv_bias silu ffn_gu ffn_down o_proj
        let skip = std::env::var("OJAS_SKIP").unwrap_or_default();
        let skip_attn = skip.contains("attn");
        let skip_bias = skip.contains("qkv_bias");
        let skip_silu = skip.contains("silu");
        let skip_ffn_gu = skip.contains("ffn_gu");
        let skip_ffn_down = skip.contains("ffn_down");
        let skip_oproj = skip.contains("o_proj");
        let want_logits = logits != LogitsOut::None;
        let m = tokens.len() as u32;
        let d = self.d as u32;
        let hd = self.arch.hd as u32;
        let kvdim = (self.arch.n_kv * self.arch.hd) as u32;
        let group = (self.arch.n_head / self.arch.n_kv) as u32;
        let scale = 1.0 / (self.arch.hd as f32).sqrt();
        let nh = self.arch.n_head as u32;
        let qdim = nh * hd;   // gpt-oss: q output dim (n_head*hd) != d; for qwen qdim==d
        // Reuse the pooled ids buffer (shared storage, so a host write is visible to
        // the next command buffer) instead of allocating one per call.
        let tokbuf = &self.st.tokbuf;
        unsafe {
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), tokbuf.contents() as *mut u32, tokens.len());
        }

        // Concurrent dispatch for the dense batch path, barriers only at true
        // dependency edges. Measured by ablation on the reference binary: disabling
        // concurrency alone drops its pp512 from 1429 to 1176 (-18%), and with fusion
        // and graph-optimize also off it lands at 1085-1176, below the 1320 a serial
        // encoder reaches here. Decode does not benefit (tiny ops, barrier cost >
        // overlap), but prefill ops are 0.1-2.7 ms, so barriers are noise and the
        // overlaps — q/k/v (three independent GEMMs), gate/up, and each op's tail
        // waves — are free throughput. gpt-oss stays serial: its MoE helpers assume
        // ordering.
        let conc = !self.arch.gpt_oss;
        // Split by layer across command buffers (`pass.rs`).
        let mut pass = super::pass::SplitPass::new(self.gpu, conc, self.cfg.prefill_cb_layers, m);
        let mut enc = pass.open();
        let eb = |e: &metal::ComputeCommandEncoderRef| { if conc { self.barc(e); } };
        // embed
        if let Some(w) = self.wt.w6k.get("token_embd.weight") {
            enc.set_compute_pipeline_state(&self.p["embed_m_q6k"]);
            enc.set_buffer(0, Some(w), self.wt.w_off.get("token_embd.weight").copied().unwrap_or(0));
            enc.set_buffer(1, Some(&self.st.x), 0);
            enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
            enc.set_buffer(3, Some(tokbuf), 0);
            enc.set_bytes(5, 4, &m as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new((((m * d) + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
        } else {
            self.enc_reduce(&enc, "embed_m_q8",
                &[(&self.wt.w8["token_embd.weight"], 0), (&self.st.x, 1), (&tokbuf, 3), (&self.wt.scale8["token_embd.weight"], 4)],
                &[(2, d), (5, m)], &[], (((m * d) + 63) / 64) as u64, 64);
        }

        for l in 0..self.arch.n_layers {
            pass.layer(l, &mut enc);
            let p = |s: &str| format!("blk.{l}.{s}");
            eb(&enc); // x complete (embed, or the previous layer's ffn_down)
            self.enc_reduce(&enc, "rmsnorm_m", &[(&self.st.x, 0), (&self.wt.w32[&p("attn_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], m as u64, 256);
            eb(&enc); // h ready for the projections
            // q,k,v with bias — fused into one dispatch (all read h)
            let total = qdim + 2 * kvdim;
            if self.arch.gpt_oss {
                self.check_qkv(&p, d, qdim, kvdim);
                let e = &enc; e.set_compute_pipeline_state(&self.p["qkv_mg_q8"]);
                e.set_buffer(0, Some(&self.st.h), 0);
                e.set_buffer(1, Some(&self.wt.w8[&p("attn_q.weight")]), 0);
                e.set_buffer(2, Some(&self.wt.w8[&p("attn_k.weight")]), 0);
                e.set_buffer(3, Some(&self.wt.w8[&p("attn_v.weight")]), 0);
                e.set_buffer(4, Some(&self.st.q), 0); e.set_buffer(5, Some(&self.st.k), 0); e.set_buffer(6, Some(&self.st.v), 0);
                e.set_bytes(7, 4, &d as *const u32 as *const c_void);
                e.set_bytes(8, 4, &qdim as *const u32 as *const c_void);
                e.set_bytes(9, 4, &kvdim as *const u32 as *const c_void);
                e.set_buffer(10, Some(&self.wt.scale8[&p("attn_q.weight")]), 0);
                e.set_buffer(11, Some(&self.wt.scale8[&p("attn_k.weight")]), 0);
                e.set_buffer(12, Some(&self.wt.scale8[&p("attn_v.weight")]), 0);
                e.set_buffer(13, Some(&self.wt.w32[&p("attn_q.bias")]), 0);
                e.set_buffer(14, Some(&self.wt.w32[&p("attn_k.bias")]), 0);
                e.set_buffer(15, Some(&self.wt.w32[&p("attn_v.bias")]), 0);
                e.set_bytes(16, 4, &m as *const u32 as *const c_void);
                e.dispatch_thread_groups(MTLSize::new(((total + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            } else {
                // dense qwen2: q/k/v via MMA GEMM (weights streamed once) + broadcast bias.
                let _ = total;
                // The three projections are mutually independent and overlap; one
                // barrier, then the three bias adds overlap too.
                let _ = self.gemm_named(&enc, &p("attn_q.weight"), &self.st.h, &self.st.q, d, qdim, m, false);
                let _ = self.gemm_named(&enc, &p("attn_k.weight"), &self.st.h, &self.st.k, d, kvdim, m, false);
                let _ = self.gemm_named(&enc, &p("attn_v.weight"), &self.st.h, &self.st.v, d, kvdim, m, false);
                eb(&enc);
                if !skip_bias {
                self.enc_reduce(&enc, "add_rowbias_m", &[(&self.st.q,0),(&self.wt.w32[&p("attn_q.bias")],1)], &[(2,qdim),(3,m*qdim)], &[], ((m*qdim+63)/64) as u64, 64);
                self.enc_reduce(&enc, "add_rowbias_m", &[(&self.st.k,0),(&self.wt.w32[&p("attn_k.bias")],1)], &[(2,kvdim),(3,m*kvdim)], &[], ((m*kvdim+63)/64) as u64, 64);
                self.enc_reduce(&enc, "add_rowbias_m", &[(&self.st.v,0),(&self.wt.w32[&p("attn_v.bias")],1)], &[(2,kvdim),(3,m*kvdim)], &[], ((m*kvdim+63)/64) as u64, 64);
                }
            }
            // rope(q,k) + store(k,v) — fused into one dispatch
            let bp = base_pos as u32;
            let aq = nh * hd / 2;
            let ak = kvdim / 2;
            eb(&enc); // q,k,v (and biases) complete before rope reads them
            // qk-norm (Qwen3/Gemma3): per-(token,head) RMSNorm on Q and K before
            // rope, mirroring the decode path. The kernel is M-aware (gid.y = token);
            // decode dispatches it with m=1. Required for a batched qwen3 forward but
            // not sufficient on its own — see batched_dense_ok, which still routes
            // qk-norm archs to the per-token prefill. It also runs on the spec-decode
            // batched forward (forward_batch_ids), which reaches here regardless of
            // the prefill gate. v_rmsnorm (Gemma4) is handled elsewhere.
            if self.arch.qk_norm {
                let nq = nh as u32;
                let nk = (kvdim / hd) as u32;
                enc.set_compute_pipeline_state(&self.p["qk_rmsnorm"]);
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.st.k), 0);
                enc.set_buffer(2, Some(&self.wt.w32[&p("attn_q_norm.weight")]), 0);
                enc.set_buffer(3, Some(&self.wt.w32[&p("attn_k_norm.weight")]), 0);
                enc.set_bytes(4, 4, &hd as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &nq as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &nk as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &self.arch.eps as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new((nq + nk) as u64, m as u64, 1), MTLSize::new(32, 1, 1));
                eb(&enc); // qk-norm complete before rope reads q,k
            }
            self.enc_reduce(&enc, "rope_qk_store_m",
                &[(&self.st.q,0),(&self.st.k,1),(&self.st.v,2),(&self.st.kcache[l],3),(&self.st.vcache[l],4),(&self.st.k,14)],
                &[(5,hd),(6,bp),(8,aq),(9,ak),(10,kvdim),(11,m),(12,self.arch.rope_neox as u32),(13,hd)], &[(7,self.arch.rope_base)],
                (((m*(aq+ak+kvdim))+63)/64) as u64, 64);
            // attention (M*n_head threadgroups) — gpt-oss uses the per-head sink variant
            eb(&enc); // k,v in the cache before attention reads it
            if skip_attn { /* ablation: leave st.attn stale */ } else if self.arch.gpt_oss {
                let win: u32 = if l % 2 == 0 { 128 } else { 0 }; // even layers = sliding-window-128
                self.enc_reduce(&enc, "attention_m_sink",
                    &[(&self.st.q,0),(&self.st.kcache[l],1),(&self.st.vcache[l],2),(&self.st.attn,3),(&self.wt.w32[&p("attn_sinks.weight")],10)],
                    &[(4,hd),(5,kvdim),(6,bp),(7,group),(9,nh),(11,win)], &[(8,scale)], (m*nh) as u64, 256);
            } else if bidir {
                // masked-diffusion: every query attends to all `m` positions (buffer 6 = total)
                self.enc_reduce(&enc, "attention_m_short_bidir",
                    &[(&self.st.q,0),(&self.st.kcache[l],1),(&self.st.vcache[l],2),(&self.st.attn,3)],
                    &[(4,hd),(5,kvdim),(6,m),(7,group),(9,nh)], &[(8,scale)], (m*nh) as u64, 256);
            } else if self.gpu.native_reduce && hd == 128 && m >= 8
                && self.p.contains_key("attn_prefill_fat")
                // Opt-in: correct (attn_gate cos=1.000000) but slower than
                // attention_m_mma (16.8 vs 10.6 ms at M=256): register-resident Q
                // (16 half frags) + O (32 f32/lane) caps occupancy harder than the
                // old kernel's transposed device loads cost it. C=32 and C=64
                // measured within 0.3% of each other, so the block size is not the
                // issue. Kept behind its numeric gate for a config that spends
                // fewer registers.
                && std::env::var("OJAS_ATTN_FAT").is_ok() {
                // Register-resident flash attention: Q and O live in fragments, K
                // staged transposed through threadgroup memory. Replaces
                // attention_m_mma's transposed device loads, which held it at
                // 3.65 TFLOP/s. OJAS_NO_ATTN_FAT falls back for A/B.
                enc.set_compute_pipeline_state(&self.p["attn_prefill_fat"]);
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.st.kcache[l]), 0);
                enc.set_buffer(2, Some(&self.st.vcache[l]), 0);
                enc.set_buffer(3, Some(&self.st.attn), 0);
                for (i, v) in [(4u32, hd), (5, kvdim), (6, bp), (7, group), (9, nh), (10, m)] {
                    enc.set_bytes(i as u64, 4, &v as *const u32 as *const c_void);
                }
                enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, ((m + 31) / 32) as u64, 1), MTLSize::new(256, 1, 1));
            } else if self.gpu.native_reduce && hd <= 256 && hd % 64 == 0 && m >= 2 {
                // MMA flash-attention: 32 queries per threadgroup, simdgroup-matrix
                // Q·K^T and P·V. The fallback, attention_m_short, gives each
                // (query, head) its own threadgroup, so every query streams the whole
                // KV cache — M-fold redundant traffic. Profiled at M=256 that put
                // attention at 282 ms of a 480 ms pass (59%) against 195 ms for every
                // GEMM combined.
                //
                // Device-Q twin: Q converted to f16 once (q_to_half) and read as
                // fragments straight from device, instead of staged into a 16.9 KB
                // threadgroup array. The stage cost occupancy, and this kernel sits
                // alone between two barriers with nothing to overlap with, so it pays
                // that starvation in full. OJAS_NO_ATTN_DQ falls back for A/B.
                let dq = std::env::var("OJAS_ATTN_DQ").is_ok()
                    && self.p.contains_key("attention_m_mma_dq");
                if dq {
                    let total = ((m + 31) / 32) * 32 * qdim;
                    let valid = m * qdim;
                    self.enc_reduce(&enc, "q_to_half", &[(&self.st.q, 0), (&self.st.qh, 1)],
                        &[(2, valid), (3, total)], &[], ((total + 63) / 64) as u64, 64);
                    eb(&enc);   // qh complete before the MMA reads it
                }
                let kname = if ojas_metal::kernels::attn::ATTN_HD_SPECIAL.contains(&hd) {
                    format!("attention_m_mma{}_{hd}", if dq { "_dq" } else { "" })
                } else if dq { "attention_m_mma_dq".to_string() } else { "attention_m_mma".to_string() };
                super::attn_log("prefill", &format!("{kname} (MMA flash)"), m);
                enc.set_compute_pipeline_state(&self.p[&kname]);
                enc.set_buffer(0, Some(if dq { &self.st.qh } else { &self.st.q }), 0);
                enc.set_buffer(1, Some(&self.st.kcache[l]), 0);
                enc.set_buffer(2, Some(&self.st.vcache[l]), 0);
                enc.set_buffer(3, Some(&self.st.attn), 0);
                for (i, v) in [(4u32, hd), (5, kvdim), (6, bp), (7, group), (9, nh), (10, m)] {
                    enc.set_bytes(i as u64, 4, &v as *const u32 as *const c_void);
                }
                enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, ((m + 31) / 32) as u64, 1), MTLSize::new(256, 1, 1));
            } else {
                super::attn_log("prefill", "attention_m_short (score-array)", m);
                self.enc_reduce(&enc, "attention_m_short",
                    &[(&self.st.q,0),(&self.st.kcache[l],1),(&self.st.vcache[l],2),(&self.st.attn,3)],
                    &[(4,hd),(5,kvdim),(6,bp),(7,group),(9,nh)], &[(8,scale)], (m*nh) as u64, 64);
            }
            if self.arch.gpt_oss {
                // biased output proj → self.st.h scratch, then residual x += h (accum kernel has no bias).
                // gpt-oss qdim (n_head*hd=4096) != d (2880), so o_proj's K is qdim, not d.
                let qdim = nh * hd;
                self.enc_reduce(&enc, "gemv_mg_q8_bias", &[(&self.st.attn,0),(&self.wt.w8[&p("attn_output.weight")],1),(&self.st.h,2),(&self.wt.scale8[&p("attn_output.weight")],5),(&self.wt.w32[&p("attn_output.bias")],7)], &[(3,qdim),(4,d),(6,m)], &[], ((d+7)/8) as u64, 256);
                self.enc_reduce(&enc, "add_inplace", &[(&self.st.x,0),(&self.st.h,1)], &[(2,m*d)], &[], ((m*d+63)/64) as u64, 64);
            } else {
                // o_proj accum via MMA GEMM (K=qdim, N=d, no bias).
                //
                // K must be qdim = n_head*hd, not d. The two coincide on Qwen2
                // (16*128 = 2048 = d) and differ on Qwen3-0.6B (qdim 2048, d 1024),
                // where passing d makes o_proj consume half the attention output.
                // Because the projection is the last step before the residual add,
                // that presents as layer 0's KV cache being correct and layer 1's
                // input garbage — it looks like a broken batched cache write.
                eb(&enc); // attention output complete
                if !skip_oproj { let _ = self.gemm_named(&enc, &p("attn_output.weight"), &self.st.attn, &self.st.x, qdim, d, m, true); }
            }
            if self.arch.gpt_oss {
                self.batched_moe_oai(&enc, l, m);
            } else {
            // ffn: rmsnorm → gate/up GEMM → silu_mul → down GEMM(accum). Weights streamed
            // once across all M tokens (MMA), vs the p[8] tiled GEMV re-reading per 8 rows.
            eb(&enc); // attention residual landed in x
            self.enc_reduce(&enc, "rmsnorm_m", &[(&self.st.x, 0), (&self.wt.w32[&p("ffn_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], m as u64, 256);
            eb(&enc); // h ready for gate/up (which overlap)
            let ff = self.arch.ffn as u32;
            let gn = p("ffn_gate.weight");
            let un = p("ffn_up.weight");
            // Fused fat GEMM for gate+up: activation fragments loaded once for both.
            let mut fused_silu = false;
            if self.gpu.native_reduce && ff % 64 == 0 && d % 32 == 0 && m >= 8
                && self.p.contains_key("ffn_gu_fat")
                && self.wt.w4l.contains_key(&gn) && self.wt.w4l.contains_key(&un)
                // Opt-in: fused measured 99.8 vs 92.0 ms — the 25% fragment-load
                // saving costs more in registers (64 accumulator floats/lane) and
                // dual-tile shmem than it returns. Kept for a config that frees
                // registers elsewhere.
                && std::env::var("OJAS_FFN_FAT").is_ok()
            {
                enc.set_compute_pipeline_state(&self.p["ffn_gu_fat"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w4l[&gn]), 0);
                enc.set_buffer(2, Some(&self.wt.w4l[&un]), 0);
                enc.set_buffer(3, Some(&self.st.gate), 0);
                enc.set_buffer(4, Some(&self.st.up), 0);
                enc.set_bytes(5, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &ff as *const u32 as *const c_void);
                enc.set_buffer(7, Some(&self.wt.q4l_a[&gn]), 0);
                enc.set_buffer(8, Some(&self.wt.q4l_b[&gn]), 0);
                enc.set_buffer(9, Some(&self.wt.q4l_a[&un]), 0);
                enc.set_buffer(10, Some(&self.wt.q4l_b[&un]), 0);
                enc.set_bytes(11, 4, &m as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((m + 63) / 64) as u64, (ff / 64) as u64, 1), MTLSize::new(128, 1, 1));
            } else {
                if !skip_ffn_gu {
                    let _ = self.gemm_named(&enc, &gn, &self.st.h, &self.st.gate, d, ff, m, false);
                    // gate must be complete before the up GEMM's epilogue reads it.
                    // OJAS_FUSED_BISECT: run the fused kernel with an identity
                    // epilogue into `up` and leave silu_mul in place. If the digest
                    // then matches the split control exactly, the fused kernel's GEMM
                    // half is faithful and any drift is the epilogue.
                    let bisect = std::env::var("OJAS_FUSED_BISECT").is_ok();
                    let acti = if bisect { 2u32 } else { self.arch.gelu as u32 };
                    let dst = if bisect { &self.st.up } else { &self.st.act };
                    let did = { eb(&enc);
                        self.gemm_up_silu(&enc, &un, &self.st.h, &self.st.gate, dst, d, ff, m, acti) };
                    fused_silu = did && !bisect;
                    if !did {
                        let _ = self.gemm_named(&enc, &un, &self.st.h, &self.st.up, d, ff, m, false);
                    }
                }
            }
            eb(&enc); // gate and up complete
            if !skip_silu && !fused_silu { self.enc_reduce(&enc, "silu_mul", &[(&self.st.gate, 0), (&self.st.up, 1), (&self.st.act, 2)], &[(3, m*ff)], &[], ((m*ff + 63)/64) as u64, 64); }
            eb(&enc); // activation ready
            if !skip_ffn_down { let _ = self.gemm_named(&enc, &p("ffn_down.weight"), &self.st.act, &self.st.x, ff, d, m, true); }
            }
        }
        // prefill only needs the KV cache populated — skip the (expensive, ~200k-vocab) lm_head
        if want_logits {
            eb(&enc); // last layer's x complete
            self.enc_reduce(&enc, "rmsnorm_m", &[(&self.st.x, 0), (&self.wt.w32["output_norm.weight"], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], m as u64, 256);
            eb(&enc); // h ready for the head
            // untied models (DiffuCoder/Dream: tie_word_embeddings=false) have a
            // separate lm_head (output.weight) — using token_embd here gives garbage.
            // MMA GEMM (vocab 152064 %64==0).
            let vocab = self.arch.vocab as u32;
            if let Some(w) = self.wt.w6k.get(&self.arch.lm_head) {
                // Native Q6_K head: the fat GEMM at prefill widths, else the 32-row tile.
                // Both need a full 64-wide N tile and 256-aligned K (one super-block);
                // every real vocab/hidden pair satisfies both.
                let off = self.wt.w_off.get(&self.arch.lm_head).copied().unwrap_or(0);
                if !self.kquant_fat(14, &enc, &self.st.h, w, off, &self.st.logits, d, vocab, m, false) {
                    enc.set_compute_pipeline_state(&self.p["gemm_mm_q6k"]);
                    enc.set_buffer(0, Some(&self.st.h), 0);
                    enc.set_buffer(1, Some(w), off);
                    enc.set_buffer(2, Some(&self.st.logits), 0);
                    enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &vocab as *const u32 as *const c_void);
                    let zero = 0u32;
                    enc.set_bytes(6, 4, &zero as *const u32 as *const c_void);
                    enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(((m + 31) / 32) as u64, (vocab / 64) as u64, 1), MTLSize::new(128, 1, 1));
                }
            } else {
                self.gemm8(&enc, &self.st.h, &self.wt.w8[&self.arch.lm_head], &self.wt.scale8[&self.arch.lm_head],
                    &self.st.logits, d, vocab, m, false);
            }
            // Row-wise argmax on-device, in the same command buffer. Callers that
            // only need the ids (speculative verify) read 4 bytes per row from st.tmp
            // and never touch the vocab*M logit block.
            eb(&enc);
            let vocab = self.arch.vocab as u32;
            self.enc_reduce(&enc, "argmax_m", &[(&self.st.logits, 0), (&self.st.tmp, 1)],
                &[(2, vocab), (3, m)], &[], m as u64, self.tune.max_tg.min(1024));
        }
        let gpu = pass.finish(&enc, "batched forward");
        self.gpu_s.set(self.gpu_s.get() + gpu);

        // Ids-only callers stop here: the argmax is already in st.tmp and the
        // vocab*M logit block never crosses the bus.
        if logits != LogitsOut::Host { return Vec::new(); }
        let ptr = self.st.logits.contents() as *const f32;
        let all = unsafe { std::slice::from_raw_parts(ptr, self.arch.vocab * tokens.len()) };
        (0..tokens.len()).map(|i| all[i * self.arch.vocab..(i + 1) * self.arch.vocab].to_vec()).collect()
    }

    /// Batched forward returning only each row's argmax — the speculative-verify
    /// primitive. Same graph as `forward_batch_impl(.., true, ..)`, but the argmax
    /// runs on-device and only 4 bytes per row cross the bus. Returning logits
    /// instead copies vocab*M floats to the host (4.9 MB at M=8, 152k vocab) and
    /// allocates M Vecs for them, which measured an M=1 batched forward at 19.9 ms
    /// against a single forward's 8.24 ms.
    ///
    /// Whether this model can run the batched dense graph: `forward_batch_impl`
    /// indexes w8/w4l directly for the embedding and the FFN, so a model whose
    /// weights landed elsewhere (native ternary keeps them in w20) panics with
    /// "no entry found for key" rather than falling back. One predicate, shared by
    /// prefill and speculative verify.
    ///
    /// Test hook: the predicate fails closed, so it needs to be observable.
    pub fn batched_dense_ok_pub(&self) -> bool { self.batched_dense_ok() }

    /// Names resolving to a given representation — for diagnosing why a weight
    /// did not land where a mode intended.
    pub fn repr_names(&self, want: &str) -> Vec<String> {
        let mut v: Vec<String> = self.wt.wshape.keys()
            .filter(|n| format!("{:?}", self.wt.repr(n)) == want)
            .cloned().collect();
        v.sort();
        v
    }

    /// Test hook for `repr_gate`: histogram + ambiguity report over the model's
    /// 2-D weights.
    pub fn repr_audit(&self) -> (Vec<(String, usize)>, Vec<String>, Vec<String>) {
        let names: Vec<String> = self.wt.wshape.keys().cloned().collect();
        self.wt.audit(&names)
    }

    pub(crate) fn batched_dense_ok(&self) -> bool {
        self.arch.ssm.is_none()
            && !self.arch.gpt_oss
            // qk-norm archs (Qwen3/Qwen3.5/Gemma3) are no longer excluded here. The
            // garbage they produced came from o_proj in the batched path being
            // dispatched with K = d instead of K = qdim (n_head*hd) — equal on Qwen2
            // (16*128 = 2048 = d), different on Qwen3-0.6B (qdim 2048, d 1024).
            // Fixed at both batched call sites; verified by kv_diff (layer 1 maxdiff
            // 159.6 -> 0.0625, layer 27 7.66 -> 0.0078, i.e. f16 noise) and by
            // decode_gate being token-identical to the per-token path on Qwen3.
            && !self.arch.v_rmsnorm
            // MLA (DeepSeek-lite / GLM-DSA) has no attn_k/attn_v.weight — the dense
            // batched graph would call gemm_named on tensors it lacks, get a false
            // return that the caller ignores, and read a stale k/v buffer. The dense
            // graph is only valid for standard MHA/GQA attention.
            && self.arch.mla.is_none()
            && (self.wt.w8.contains_key("token_embd.weight")
                || self.wt.w6k.contains_key("token_embd.weight"))
            // Asked through `repr()` rather than by probing maps: a map-probing
            // version silently returns false when a newly added representation's map
            // is missed, disabling batched prefill (a ~4x path) with no error.
            && self.wt.repr("blk.0.ffn_gate.weight").present()
            && self.wt.repr("blk.0.ffn_down.weight").present()
            // The score-array prefill attention (attention_m_short) buffers scores in
            // sc[4096]; it's the fallback when the MMA path can't run (hd not a multiple
            // of 64, hd>256, or no native reduce). Such a model must not batch-prefill
            // past 4096 or it overflows — route it to per-token forward_id (decode's
            // streaming attn is depth-safe) instead.
            && {
                let mma = self.gpu.native_reduce && self.arch.hd <= 256 && self.arch.hd % 64 == 0;
                mma || self.max_seq() <= 4096
            }
    }

    pub(crate) fn forward_batch_ids(&self, tokens: &[u32], base_pos: usize) -> Vec<u32> {
        if tokens.is_empty() { return Vec::new(); }
        // Ids, not Host: Host pays the vocab*M logits readback plus M Vec
        // allocations on every speculative verify, which is the cost Ids removes.
        self.forward_batch_impl(tokens, base_pos, LogitsOut::Ids, false);
        let p = self.st.tmp.contents() as *const u32;
        (0..tokens.len()).map(|i| unsafe { *p.add(i) }).collect()
    }

    /// Like `forward_batch_ids`, but also returns each position's top-8 token ids
    /// (m*8, row-major). The Token Recycling drafter feeds on these: the model's own
    /// next-token guesses, computed anyway.
    pub(crate) fn forward_batch_ids_topk(&self, tokens: &[u32], base_pos: usize) -> (Vec<u32>, Vec<u32>) {
        if tokens.is_empty() { return (Vec::new(), Vec::new()); }
        self.forward_batch_impl(tokens, base_pos, LogitsOut::IdsTopK, false);
        let m = tokens.len();
        let p = self.st.tmp.contents() as *const u32;
        let ids = (0..m).map(|i| unsafe { *p.add(i) }).collect();
        let tk = (0..m * 8).map(|i| unsafe { *p.add(m + i) }).collect();
        (ids, tk)
    }

    /// Batched gpt-oss MoE FFN for M tokens in one chunk: post-attn rmsnorm →
    /// router (f32) + bias → softmax top-k → biased SwiGLU-OAI experts → weighted
    /// biased down-proj accumulated into x. Uses the M-sized batched OAI kernels
    /// and the shared batched-MoE scratch (moe_blg/bidx/bwgt/bact).
    pub(crate) fn batched_moe_oai(&self, enc: &metal::ComputeCommandEncoderRef, l: usize, m: u32) {
        let p = |s: &str| format!("blk.{l}.{s}");
        let d = self.d as u32;
        let mo = self.arch.moe.unwrap();
        let (ne, nu, fe) = (mo.n_expert, mo.n_used, mo.ffn_exp);
        self.check_moe(&p, d, fe);
        let (alpha, limit) = (1.702f32, 7.0f32);
        let ib = |enc: &metal::ComputeCommandEncoderRef, idx: u64, v: u32| {
            enc.set_bytes(idx, 4, &v as *const u32 as *const c_void);
        };
        // pre-FFN norm (post_attention_norm) → h [m,d]
        self.enc_reduce(enc, "rmsnorm_m", &[(&self.st.x,0),(&self.wt.w32[&p("post_attention_norm.weight")],1),(&self.st.h,2)], &[(3,d)], &[(4,self.arch.eps)], m as u64, 256);
        // router logits [m,ne] + broadcast router bias
        enc.set_compute_pipeline_state(&self.p["gemv_w32_m"]);
        enc.set_buffer(0, Some(&self.st.h), 0);
        enc.set_buffer(1, Some(&self.wt.w32[&p("ffn_gate_inp.weight")]), 0);
        enc.set_buffer(2, Some(&self.ms.moe_blg), 0);
        ib(enc, 3, d); ib(enc, 4, ne);
        enc.dispatch_thread_groups(MTLSize::new(((ne+7)/8) as u64, m as u64, 1), MTLSize::new(256,1,1));
        self.enc_reduce(enc, "add_rowbias_m", &[(&self.ms.moe_blg,0),(&self.wt.w32[&p("ffn_gate_inp.bias")],1)], &[(2,ne),(3,m*ne)], &[], ((m*ne+63)/64) as u64, 64);
        // per-row softmax top-k → moe_bidx [m,nu], moe_bwgt [m,nu]
        enc.set_compute_pipeline_state(&self.p["moe_topk_m"]);
        enc.set_buffer(0, Some(&self.ms.moe_blg), 0);
        enc.set_buffer(1, Some(&self.ms.moe_bidx), 0);
        enc.set_buffer(2, Some(&self.ms.moe_bwgt), 0);
        ib(enc, 3, ne); ib(enc, 4, nu);
        enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(32,1,1));
        self.bar(enc); // routing tables ready
        // biased SwiGLU-OAI gate/up → moe_bact [m*nu, fe]
        enc.set_compute_pipeline_state(&self.p["moe_gu_q8_oai_m"]);
        enc.set_buffer(0, Some(&self.st.h), 0);
        enc.set_buffer(1, Some(&self.wt.w8[&p("ffn_gate_exps.weight")]), 0);
        enc.set_buffer(2, Some(&self.wt.w8[&p("ffn_up_exps.weight")]), 0);
        enc.set_buffer(3, Some(&self.ms.moe_bact), 0);
        ib(enc, 4, d); ib(enc, 5, fe);
        enc.set_buffer(6, Some(&self.wt.scale8[&p("ffn_gate_exps.weight")]), 0);
        enc.set_buffer(7, Some(&self.wt.scale8[&p("ffn_up_exps.weight")]), 0);
        enc.set_buffer(8, Some(&self.ms.moe_bidx), 0);
        ib(enc, 9, nu);
        enc.set_buffer(10, Some(&self.wt.w32[&p("ffn_gate_exps.bias")]), 0);
        enc.set_buffer(11, Some(&self.wt.w32[&p("ffn_up_exps.bias")]), 0);
        enc.set_bytes(12, 4, &alpha as *const f32 as *const c_void);
        enc.set_bytes(13, 4, &limit as *const f32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(((fe+31)/32) as u64, (m*nu) as u64, 1), MTLSize::new(256,1,1));
        self.bar(enc); // expert activations done
        // weighted biased down-proj accumulate into x [m,d]
        enc.set_compute_pipeline_state(&self.p["moe_down_q8_oai_m"]);
        enc.set_buffer(0, Some(&self.ms.moe_bact), 0);
        enc.set_buffer(1, Some(&self.wt.w8[&p("ffn_down_exps.weight")]), 0);
        enc.set_buffer(2, Some(&self.st.x), 0);
        ib(enc, 3, fe); ib(enc, 4, d);
        enc.set_buffer(5, Some(&self.wt.scale8[&p("ffn_down_exps.weight")]), 0);
        enc.set_buffer(6, Some(&self.ms.moe_bidx), 0);
        enc.set_buffer(7, Some(&self.ms.moe_bwgt), 0);
        ib(enc, 8, nu);
        enc.set_buffer(9, Some(&self.wt.w32[&p("ffn_down_exps.bias")]), 0);
        enc.dispatch_thread_groups(MTLSize::new(((d+31)/32) as u64, m as u64, 1), MTLSize::new(256,1,1));
    }

}
