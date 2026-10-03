#![allow(clippy::too_many_arguments)]
//! Batched (M>1) qwen4exp forward — prefill chunks and speculative verify.
//!
//! Same graph as `encode_qwen4exp`, with every buffer token-major: hc_res/hc_xn/
//! hc_gated are [T][hc][d], hc_mixed [T][d], hc_inject [T][hc], h [T][d]. T=1 is
//! the decode layout, so an M=1 run here checks this graph against decode.
//!
//! The PLE layer loops tokens rather than batching: its row indices come from a
//! host-side n-gram hash of the token history, so the gather has nothing to
//! batch, and it covers one layer of 48.
use super::*;
use crate::decoder::dispatch::mrow_max;
use metal::MTLSize;
use objc::{sel, sel_impl};
use std::ffi::c_void;

impl<'a> DecoderGpu<'a> {
    /// Batched matmul for the chunk path: y[T][n] = w · x[T][k].
    ///
    /// Kernel choice by representation, and by shape for f16: the MMA GEMM needs
    /// N%64 and K%32, which the projections satisfy but `hc_*_inject` (N = hc = 4)
    /// and `ssm_alpha/beta` (N = 48) do not, so those fall back to a per-row GEMV.
    fn gemm_m(&self, enc: &metal::ComputeCommandEncoderRef, wname: &str,
              x: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32, m: u32) {
        self.check_shape(wname, k, n);
        let f4 = std::mem::size_of::<f32>() as u64;
        if self.wt.wq.contains_key(wname) {
            self.nat_batched(enc, wname, x, y, k, n, m, false);
            return;
        }
        if let Some(w) = self.wt.w32.get(wname) {
            for row in 0..m as u64 {
                enc.set_compute_pipeline_state(&self.p["gemv_w32"]);
                enc.set_buffer(0, Some(x), row * (k as u64) * f4);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(y), row * (n as u64) * f4);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((n + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            return;
        }
        if let Some(w) = self.wt.w16.get(wname) {
            // The MMA tile is 32 tokens wide and zero-pads the rest, so a 2-token
            // batch does 16x the arithmetic of two GEMVs for the same answer. Reads
            // the same crossover knob the dense path measured for its m-row kernels.
            // Keeping m=1 on the GEMV also makes it bit-identical to decode.
            if m > mrow_max() && n % 64 == 0 && k % 32 == 0 && self.gpu.native_reduce {
                self.gemm16_off(enc, x, 0, w, y, k, n, m, false);
                return;
            }
            // Below the MMA crossover, the m-row kernel reads each weight row once
            // for all M tokens; looping gemv_f16 re-reads the whole matrix per token.
            // M <= 8 is the kernel's accumulator array; K % 4 its vector load.
            if m > 1 && m <= 8 && k % 4 == 0 {
                enc.set_compute_pipeline_state(&self.p["gemv_m_f16"]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(y), 0);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &m as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((n + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
                return;
            }
            for row in 0..m as u64 {
                enc.set_compute_pipeline_state(&self.p["gemv_f16"]);
                enc.set_buffer(0, Some(x), row * (k as u64) * f4);
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(y), row * (n as u64) * f4);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((n + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            return;
        }
        panic!("{wname} is in no batched-capable map ({:?})", self.wt.repr(wname));
    }

    /// One hyper-connection mixer over M tokens. Mirrors `hc_mix`.
    fn hc_mix_m(&self, enc: &metal::ComputeCommandEncoderRef, base: &str,
                d: u32, hc: u32, hc_lr: u32, m: u32, with_inject: bool) {
        let w = |s: &str| format!("{base}{s}");
        let hc_dim = d * hc;
        enc.set_compute_pipeline_state(&self.p["hc_rmsnorm"]);
        enc.set_buffer(0, Some(&self.st.hc_res), 0);
        enc.set_buffer(1, Some(&self.wt.w32[&w("_norm.weight")]), 0);
        enc.set_buffer(2, Some(&self.st.hc_xn), 0);
        enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
        enc.set_bytes(5, 4, &hc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(hc as u64, m as u64, 1), MTLSize::new(256, 1, 1));
        self.bar(enc);
        self.gemm_m(enc, &w("_down.weight"), &self.st.hc_xn, &self.st.hc_lo, hc_dim, hc_lr, m);
        self.bar(enc);
        // elementwise over the whole [T][hc_lr] buffer — a batch is just a longer grid
        let n_lo = hc_lr * m;
        enc.set_compute_pipeline_state(&self.p["hc_silu_scale"]);
        enc.set_buffer(0, Some(&self.st.hc_lo), 0);
        enc.set_bytes(1, 4, &n_lo as *const u32 as *const c_void);
        enc.set_bytes(2, 4, &hc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(((n_lo + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
        self.bar(enc);
        self.gemm_m(enc, &w("_up.weight"), &self.st.hc_lo, &self.st.hc_graw, hc_lr, hc_dim, m);
        self.bar(enc);
        let n_g = hc_dim * m;
        enc.set_compute_pipeline_state(&self.p["hc_gate"]);
        enc.set_buffer(0, Some(&self.st.hc_xn), 0);
        enc.set_buffer(1, Some(&self.st.hc_graw), 0);
        enc.set_buffer(2, Some(&self.st.hc_gated), 0);
        enc.set_bytes(3, 4, &n_g as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(((n_g + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p["hc_collapse"]);
        enc.set_buffer(0, Some(&self.st.hc_gated), 0);
        enc.set_buffer(1, Some(&self.st.hc_mixed), 0);
        enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &hc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(((d + 63) / 64) as u64, m as u64, 1), MTLSize::new(64, 1, 1));
        if with_inject {
            self.gemm_m(enc, &w("_inject.weight"), &self.st.hc_xn, &self.st.hc_inject, hc_dim, hc, m);
        }
    }

    fn hc_combine_m(&self, enc: &metal::ComputeCommandEncoderRef, block: &metal::Buffer,
                    d: u32, hc: u32, m: u32) {
        enc.set_compute_pipeline_state(&self.p["hc_combine"]);
        enc.set_buffer(0, Some(&self.st.hc_res), 0);
        enc.set_buffer(1, Some(block), 0);
        enc.set_buffer(2, Some(&self.st.hc_inject), 0);
        enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &hc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(((d + 63) / 64) as u64, hc as u64, m as u64),
                                   MTLSize::new(64, 1, 1));
    }

    /// Shared + routed MoE over M tokens, accumulating into `h` (pre-zeroed).
    /// Kernels come from `MOE_FORMATS`' batched column; a format without one would
    /// have to run per token, so this refuses rather than substituting another.
    fn qwen4exp_moe_m_route(&self, enc: &metal::ComputeCommandEncoderRef, l: usize, d: u32, m: u32) {
        let p = |s: &str| format!("blk.{l}.{s}");
        let mc = self.arch.moe.unwrap();
        let (ne, nu, fe, fs) = (mc.n_expert, mc.n_used, mc.ffn_exp, mc.ffn_shexp);
        self.check_moe(&p, d, fe);
        let inp = &self.st.hc_mixed;
        let off = (l as u64) * (MAXM as u64) * (nu as u64) * 4;
        let f4 = std::mem::size_of::<f32>() as u64;

        // Router logits + shared expert, per token, through the M-strided router
        // scratch (moe_b*) rather than the decode buffers: moe_lg holds one token's
        // n_expert logits, moe_wgt one token's top-k weights and moe_sh a single
        // scalar, so writing token t at stride t would run off the end of all three.
        for t in 0..m as u64 {
            self.enc_reduce_off(enc, "gemv_w32",
                &[(inp, 0, t * (d as u64) * f4), (&self.wt.w32[&p("ffn_gate_inp.weight")], 1, 0),
                  (&self.ms.moe_blg, 2, t * (ne as u64) * f4)],
                &[(3, d), (4, ne)], &[], ((ne + 7) / 8) as u64, 256);
        }
        self.gemm_m(enc, &p("ffn_gate_shexp.weight"), inp, &self.st.gate, d, fs, m);
        self.gemm_m(enc, &p("ffn_up_shexp.weight"), inp, &self.st.up, d, fs, m);
        self.bar(enc);
        let n_sm = fs * m;
        self.enc_reduce(enc, "silu_mul", &[(&self.st.gate, 0), (&self.st.up, 1), (&self.st.act, 2)],
                        &[(3, n_sm)], &[], ((n_sm + 63) / 64) as u64, 64);
        self.bar(enc);
        self.gemm_m(enc, &p("ffn_down_shexp.weight"), &self.st.act, &self.ms.moe_btmp, fs, d, m);
        for t in 0..m as u64 {
            self.enc_reduce_off(enc, "gemv_w32",
                &[(inp, 0, t * (d as u64) * f4), (&self.wt.w32[&p("ffn_gate_inp_shexp.weight")], 1, 0),
                  (&self.ms.moe_bsh, 2, t * f4)],
                &[(3, d), (4, 1)], &[], 1, 32);
            enc.set_compute_pipeline_state(&self.p["moe_topk"]);
            enc.set_buffer(0, Some(&self.ms.moe_blg), t * (ne as u64) * f4);
            enc.set_buffer(1, Some(&self.ms.moe_idx), off + t * (nu as u64) * 4);
            enc.set_buffer(2, Some(&self.ms.moe_bwgt), t * (nu as u64) * f4);
            enc.set_bytes(3, 4, &ne as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
        }
    }

    /// Routed-expert half. Split from the router so a streamed model can gather
    /// between the two: `moe_idx` has to be readable on the CPU before the expert
    /// weights it names can be packed into scratch.
    fn qwen4exp_moe_m_experts(&self, enc: &metal::ComputeCommandEncoderRef, l: usize, d: u32, m: u32) {
        let p = |s: &str| format!("blk.{l}.{s}");
        let mc = self.arch.moe.unwrap();
        let (nu, fe) = (mc.n_used, mc.ffn_exp);
        let inp = &self.st.hc_mixed;
        let off = (l as u64) * (MAXM as u64) * (nu as u64) * 4;
        let gname = p("ffn_gate_exps.weight");
        let dname = p("ffn_down_exps.weight");
        let gty = self.wt.w_qtype.get(&gname).copied().unwrap_or(12);
        let dty = self.wt.w_qtype.get(&dname).copied().unwrap_or(8);
        let gk = ojas_core::quant_src::moe_kernel_m(gty, ojas_core::quant_src::MoeRole::GateUp)
            .unwrap_or_else(|| panic!("no batched MoE gate/up kernel for GGUF type {gty} ({gname})"));
        let dk = ojas_core::quant_src::moe_kernel_m(dty, ojas_core::quant_src::MoeRole::Down)
            .unwrap_or_else(|| panic!("no batched MoE down kernel for GGUF type {dty} ({dname})"));

        let direct = self.strm.direct_layer.get() == Some(l);
        if direct { self.direct_expert_resources(enc); }
        let uname = p("ffn_up_exps.weight");
        // (buffer, byte offset). Zero-copy mmap buffers carry a page-alignment
        // offset (`w_off`); scratch and requant buffers start at zero.
        let woff = |n: &str| self.wt.w_off.get(n).copied().unwrap_or(0);
        let (gb, ub, db, goff, uoff, doff, idxb, idx_off) = if self.is_resident(l) {
            // Full native tensors, indexed by the router's own moe_idx — the same
            // per-expert stride arithmetic the scratch kernels use, with `e` the
            // true expert id instead of a union slot.
            (&self.wt.wq[&gname], &self.wt.wq[&uname], &self.wt.wq[&dname],
             woff(&gname), woff(&uname), woff(&dname), &self.ms.moe_idx, off)
        } else if direct {
            (&self.strm.direct_tables[0], &self.strm.direct_tables[1], &self.strm.direct_tables[2],
             0, 0, 0, &self.strm.moe_slot, 0u64)
        } else if self.strm.stream {
            (&self.strm.moe_gs, &self.strm.moe_us, &self.strm.moe_ds,
             0, 0, 0, &self.strm.moe_slot, 0u64)
        } else {
            (&self.wt.w4k[&gname], &self.wt.w4k[&uname], &self.wt.w4k[&dname],
             0, 0, 0, &self.ms.moe_idx, off)
        };
        enc.set_compute_pipeline_state(&self.p[if direct { if gty == 21 { "moe_gu_iq3s_m_direct" } else { "moe_gu_iq4xs_m_direct" } } else { gk.entry }]);
        enc.set_buffer(0, Some(inp), 0);
        enc.set_buffer(1, Some(gb), goff);
        enc.set_buffer(2, Some(ub), uoff);
        enc.set_buffer(3, Some(&self.ms.moe_bact), 0);
        enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &fe as *const u32 as *const c_void);
        enc.set_buffer(8, Some(idxb), idx_off);
        enc.set_bytes(9, 4, &nu as *const u32 as *const c_void);
        let (gt, gr) = gk.launch;
        enc.dispatch_thread_groups(MTLSize::new(((fe + gr - 1) / gr) as u64, (m * nu) as u64, 1),
                                   MTLSize::new(gt as u64, 1, 1));
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p[if direct { "moe_down_iq4nl_m_direct" } else { dk.entry }]);
        enc.set_buffer(0, Some(&self.ms.moe_bact), 0);
        enc.set_buffer(1, Some(db), doff);
        enc.set_buffer(2, Some(&self.st.h), 0);
        enc.set_bytes(3, 4, &fe as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
        enc.set_buffer(6, Some(idxb), idx_off);
        enc.set_buffer(7, Some(&self.ms.moe_bwgt), 0);
        enc.set_bytes(8, 4, &nu as *const u32 as *const c_void);
        enc.set_buffer(9, Some(&self.ms.moe_btmp), 0);
        enc.set_buffer(10, Some(&self.ms.moe_bsh), 0);
        let (dt, dr) = dk.launch;
        enc.dispatch_thread_groups(MTLSize::new(((d + dr - 1) / dr) as u64, m as u64, 1),
                                   MTLSize::new(dt as u64, 1, 1));
    }

    /// Gated DeltaNet over M tokens. Same kernels as the decode path; the M-aware
    /// conv and recurrence are the ones graph_chunk already uses for qwen35.
    fn qwen4exp_gdn_m(&self, enc: &metal::ComputeCommandEncoderRef, l: usize, d: u32, m: u32, verify: bool) {
        let p = |s: &str| format!("blk.{l}.{s}");
        let sc = self.arch.ssm.unwrap();
        let (s_st, hk, hv) = (sc.d_state, sc.n_group, sc.dt_rank);
        let d_inner = sc.d_inner;
        let conv_ch = d_inner + 2 * hk * s_st;
        let conv_k = sc.conv_kernel;
        let head_v = d_inner / hv;
        let inp = &self.st.hc_mixed;
        self.gemm_m(enc, &p("attn_qkv.weight"), inp, &self.st.ssm_qkv, d, conv_ch, m);
        self.gemm_m(enc, &p("attn_gate.weight"), inp, &self.st.ssm_z, d, d_inner, m);
        self.gemm_m(enc, &p("ssm_alpha.weight"), inp, &self.st.ssm_gate, d, hv, m);
        self.gemm_m(enc, &p("ssm_beta.weight"), inp, &self.st.ssm_beta, d, hv, m);
        self.bar(enc);
        self.enc_reduce(enc, "ssm_ab",
            &[(&self.st.ssm_gate, 0), (&self.st.ssm_beta, 1), (&self.wt.w32[&p("ssm_dt.bias")], 2),
              (&self.wt.w32[&p("ssm_a")], 3)], &[(4, m * hv), (5, hv)], &[], ((m * hv + 63) / 64) as u64, 64);
        enc.set_compute_pipeline_state(&self.p["conv1d_prefill"]);
        enc.set_buffer(0, Some(&self.st.ssm_qkv), 0);
        enc.set_buffer(1, Some(&self.st.conv_state[l]), 0);
        enc.set_buffer(2, Some(&self.wt.w32[&p("ssm_conv1d.weight")]), 0);
        for (i, v) in [(3u64, conv_ch), (4, conv_k), (5, m)] { enc.set_bytes(i, 4, &v as *const u32 as *const c_void); }
        // Verify saves row zero, or every row when prefix acceptance is enabled.
        // UINT_MAX disables writes; high bit plus full conv stride selects rows.
        enc.set_buffer(6, Some(if verify { &self.sp.conv_snap[l] } else { &self.st.conv_state[l] }), 0);
        let snap_at = if verify && self.cfg.mtp_prefix && self.sp.snapshot_rows > 1 {
            let stride = self.st.conv_state[l].length() / 4;
            assert!(stride < 0x7fff_ffff);
            0x8000_0000 | stride as u32
        } else if verify { 0u32 } else { u32::MAX };
        enc.set_bytes(7, 4, &snap_at as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(((conv_ch + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p["deltanet_fused"]);
        enc.set_buffer(0, Some(&self.st.ssm_state[l]), 0);
        enc.set_buffer(1, Some(&self.st.ssm_qkv), 0);
        enc.set_buffer(2, Some(&self.st.ssm_gate), 0);
        enc.set_buffer(3, Some(&self.st.ssm_beta), 0);
        enc.set_buffer(4, Some(&self.st.ssm_o), 0);
        for (i, v) in [(5u64, s_st), (6, hk), (7, hv), (8, conv_ch), (9, m)] { enc.set_bytes(i, 4, &v as *const u32 as *const c_void); }
        enc.set_bytes(10, 4, &self.arch.eps as *const f32 as *const c_void);
        enc.set_buffer(11, Some(if verify { &self.sp.ssm_snap[l] } else { &self.st.ssm_state[l] }), 0);
        enc.set_bytes(12, 4, &snap_at as *const u32 as *const c_void);
        enc.set_bytes(13, 4, &(self.cfg.moe_kmap_div as u32) as *const u32 as *const c_void);
        // Match scalar decode and FLA GDN normalization during verification.
        enc.set_bytes(14, 4, &0u32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new((s_st / 4) as u64, hv as u64, 1), MTLSize::new(128, 1, 1));
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p["gated_rmsnorm"]);
        enc.set_buffer(0, Some(&self.st.ssm_o), 0);
        enc.set_buffer(1, Some(&self.wt.w32[&p("ssm_norm.weight")]), 0);
        enc.set_buffer(2, Some(&self.st.ssm_z), 0);
        for (i, v) in [(3u64, head_v), (5, d_inner)] { enc.set_bytes(i, 4, &v as *const u32 as *const c_void); }
        enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
        enc.set_bytes(6, 4, &1u32 as *const u32 as *const c_void);   // qwen4exp: sigmoid gate
        enc.dispatch_thread_groups(MTLSize::new(hv as u64, m as u64, 1), MTLSize::new(32, 1, 1));
        self.bar(enc);
        self.gemm_m(enc, &p("ssm_out.weight"), &self.st.ssm_o, &self.st.h, d_inner, d, m);
    }

    /// Gated GQA over M tokens. Dense, as in decode — the learned indexer that
    /// prunes KV for long contexts is not applied on either path yet.
    fn qwen4exp_qsa_m(&self, enc: &metal::ComputeCommandEncoderRef, l: usize, base_pos: usize,
                      d: u32, m: u32) {
        let p = |s: &str| format!("blk.{l}.{s}");
        let lp = self.arch.layers[l];
        let sc = self.arch.ssm.unwrap();
        let (hd, kvdim, qdim) = (lp.head_dim, lp.kvdim, lp.qdim);
        let group = lp.n_head / lp.n_kv.max(1);
        let inp = &self.st.hc_mixed;
        self.gemm_m(enc, &p("attn_q.weight"), inp, &self.st.ssm_qkv, d, 2 * qdim, m);
        self.gemm_m(enc, &p("attn_k.weight"), inp, &self.st.k, d, kvdim, m);
        self.gemm_m(enc, &p("attn_v.weight"), inp, &self.st.v, d, kvdim, m);
        self.bar(enc);
        self.enc_reduce(enc, "qgate_split", &[(&self.st.ssm_qkv, 0), (&self.st.q, 1)],
                        &[(2, hd), (3, qdim), (4, m)], &[], ((m * qdim + 63) / 64) as u64, 64);
        let (nq, nk) = (lp.n_head, lp.n_kv);
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p["qk_rmsnorm"]);
        enc.set_buffer(0, Some(&self.st.q), 0);
        enc.set_buffer(1, Some(&self.st.k), 0);
        enc.set_buffer(2, Some(&self.wt.w32[&p("attn_q_norm.weight")]), 0);
        enc.set_buffer(3, Some(&self.wt.w32[&p("attn_k_norm.weight")]), 0);
        for (i, v) in [(4u64, hd), (5, nq), (6, nk)] { enc.set_bytes(i, 4, &v as *const u32 as *const c_void); }
        enc.set_bytes(7, 4, &self.arch.eps as *const f32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new((nq + nk) as u64, m as u64, 1), MTLSize::new(32, 1, 1));
        self.bar(enc);
        let (aq, ak) = (qdim / 2, kvdim / 2);
        self.enc_reduce(enc, "rope_qk_store_m",
            &[(&self.st.q, 0), (&self.st.k, 1), (&self.st.v, 2), (&self.st.kcache[l], 3), (&self.st.vcache[l], 4), (&self.st.k, 14)],
            &[(5, hd), (6, base_pos as u32), (8, aq), (9, ak), (10, kvdim), (11, m), (12, 1), (13, sc.n_rot)],
            &[(7, lp.rope_base)], ((m * (aq + ak + kvdim) + 63) / 64) as u64, 64);
        self.bar(enc);
        // Short-context kernel below the streaming-softmax crossover, the general
        // M-kernel above it. graph_chunk's MMA flash variant is not wired here: a
        // streamed chunk is at most GATHER_M tokens and the time goes to the expert
        // reads, not to attention.
        let kern = if base_pos + m as usize <= 512 { "attention_m_short" } else { "attention_m" };
        self.enc_reduce(enc, kern,
            &[(&self.st.q, 0), (&self.st.kcache[l], 1), (&self.st.vcache[l], 2), (&self.st.attn, 3)],
            &[(4, hd), (5, kvdim), (6, base_pos as u32), (7, group), (9, lp.n_head)],
            &[(8, lp.scale)], (m * lp.n_head) as u64, 256);
        self.bar(enc);
        self.enc_reduce(enc, "gate_mul_sigmoid", &[(&self.st.attn, 0), (&self.st.ssm_qkv, 1)],
                        &[(2, hd), (3, qdim), (4, m)], &[], ((m * qdim + 63) / 64) as u64, 64);
        self.bar(enc);
        self.gemm_m(enc, &p("attn_output.weight"), &self.st.attn, &self.st.h, qdim, d, m);
    }

    /// Everything in layer `l` up to and including the router: hc mix, the token
    /// mixer (DeltaNet or gated attention), the combine, the FFN mix, and the
    /// shared expert + top-k. Leaves `moe_idx` written and `h` holding the shared
    /// expert's output for the routed half to accumulate onto.
    fn qwen4exp_layer_route(&self, enc: &metal::ComputeCommandEncoderRef, l: usize,
                            tokens: &[u32], base_pos: usize, d: u32, hc: u32, hc_lr: u32, verify: bool) {
        let m = tokens.len() as u32;
        let q = self.arch.qwen4exp.as_ref().unwrap();
        // PLE loops tokens: its rows come from a host-side n-gram hash of the token
        // history, so the gather has nothing to batch. Each pass takes its own row
        // of hc_res and of the packed PLE staging.
        if q.ple_layers.contains(&(l as u32)) && !self.cfg.no_ple {
            for (i, &t) in tokens.iter().enumerate() {
                self.qwen4exp_ple(enc, l, t, base_pos + i, d, i);
                self.bar(enc);
                if verify && (i == 0 || self.cfg.mtp_prefix && self.sp.snapshot_rows > 1) {
                    // DeltaNet snapshots its own prefix later. Save only PLE's
                    // appended history here, immediately after this input row.
                    let sc = self.arch.ssm.unwrap();
                    let off = ((sc.conv_kernel - 1) * (sc.d_inner + 2 * sc.n_group * sc.d_state)) as u64 * 4;
                    let n = (self.st.conv_state[l].length() - off) / 4;
                    enc.set_compute_pipeline_state(&self.p["copy_buf"]);
                    enc.set_buffer(0, Some(&self.sp.conv_snap[l]), off + i as u64 * self.st.conv_state[l].length());
                    enc.set_buffer(1, Some(&self.st.conv_state[l]), off);
                    let n = n as u32;
                    enc.set_bytes(2, 4, &n as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(64) as u64, 1, 1), MTLSize::new(64, 1, 1));
                    self.bar(enc);
                }
            }
        }
        self.hc_mix_m(enc, &format!("blk.{l}.hc_attn"), d, hc, hc_lr, m, true);
        self.bar(enc);
        if self.arch.layers[l].is_ssm {
            self.qwen4exp_gdn_m(enc, l, d, m, verify);
        } else {
            self.qwen4exp_qsa_m(enc, l, base_pos, d, m);
        }
        self.bar(enc);
        self.hc_combine_m(enc, &self.st.h, d, hc, m);
        self.bar(enc);
        self.hc_mix_m(enc, &format!("blk.{l}.hc_ffn"), d, hc, hc_lr, m, true);
        self.bar(enc);
        self.zero_buf(enc, &self.st.h, d * m);
        self.bar(enc);
        self.qwen4exp_moe_m_route(enc, l, d, m);
        self.bar(enc);
    }

    fn qwen4exp_layer_experts(&self, enc: &metal::ComputeCommandEncoderRef, l: usize,
                              d: u32, hc: u32, m: u32) {
        // The diagnostic hash runs in this encoder, immediately before the experts
        // read the same table, so it sees the same addresses under the same
        // residency and pinning as the consumer.
        if super::audit::instrumented_layer() == Some(l) {
            self.encode_expert_table_hash(enc, l);
            self.bar(enc);
        }
        self.qwen4exp_moe_m_experts(enc, l, d, m);
        self.bar(enc);
        self.hc_combine_m(enc, &self.st.h, d, hc, m);
        self.bar(enc);
    }

    /// Dispatch the diagnostic hash over the expert address table.
    ///
    /// Indexes `direct_tables[kind][slot]` — the same table and the same slot
    /// indexing the direct expert kernels use (`((device const uchar*)wg[e])`).
    fn encode_expert_table_hash(&self, enc: &metal::ComputeCommandEncoderRef, l: usize) {
        let Some(moe) = self.arch.moe.as_ref() else { return };
        let ne = moe.n_expert as u64;
        let records = self.strm.expert_hash_records.borrow();
        let names = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];
        for (kind, name) in names.iter().enumerate() {
            let Some(&(_, _, rawlen, _)) = self.strm.stream_meta.get(&format!("blk.{l}.{name}")) else { continue };
            let stride = (rawlen / ne) as u32;
            let count = records.iter().filter(|r| r.kind == kind as u8).count() as u64;
            if count == 0 { continue; }
            enc.set_compute_pipeline_state(&self.p["expert_table_hash"]);
            // The effective addresses the consumer will use, resolved by
            // `record_expert_table` the same way the expert kernel resolves them.
            enc.set_buffer(0, Some(&self.st.expert_hash_addr), (kind as u64) * 1024 * 8);
            // One u32 per slot per kind, kinds laid out back to back.
            enc.set_buffer(1, Some(&self.st.expert_hash_out), (kind as u64) * 1024 * 4);
            enc.set_bytes(2, 4, &stride as *const u32 as *const c_void);
            let samples = super::audit::HASH_SAMPLES;
            enc.set_bytes(3, 4, &samples as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(count, 1, 1), MTLSize::new(1, 1, 1));
        }
    }

    /// Compare the GPU's view of the expert table against the intended experts.
    ///
    /// Indexes the GPU output by `SlotRecord::slot`, not by position in the record
    /// list: a sequential walk silently compares slot N against slot M's hash the
    /// moment any slot is missing.
    ///
    /// Returns (mismatches, unresolved, compared). Unresolved slots are missing
    /// evidence, not agreement.
    pub(crate) fn check_expert_table_hash(&self)
        -> (Vec<(super::audit::SlotRecord, u32)>, usize, usize) {
        let records = self.strm.expert_hash_records.borrow();
        let out = self.st.expert_hash_out.contents() as *const u32;
        let (mut bad, mut unresolved, mut compared) = (Vec::new(), 0usize, 0usize);
        for r in records.iter() {
            if !r.resolved { unresolved += 1; continue; }
            let slot = r.slot as usize;
            if slot >= 1024 { unresolved += 1; continue; }
            let gpu = unsafe { *out.add(r.kind as usize * 1024 + slot) };
            compared += 1;
            if gpu != r.expected_hash { bad.push((r.clone(), gpu)); }
        }
        (bad, unresolved, compared)
    }

    /// Batched qwen4exp forward over a chunk of tokens: fills the KV cache and
    /// advances the DeltaNet conv/recurrent state.
    ///
    /// `verify` is the MTP path: it snapshots the recurrent state as of token 0 so a
    /// rejected draft can be rolled back, and runs the terminal mixer + head to leave
    /// each row's argmax in `tmp[row]`. Prefill skips both.
    ///
    /// Streamed models split each layer into two command buffers around the expert
    /// gather, exactly as `forward_id_streamed` does and for the same reason: a
    /// single buffer binding all 48 layers' mmap'd experts would make the whole
    /// model resident at commit. In-RAM models run the chunk in one buffer.
    pub(crate) fn forward_chunk_qwen4exp(&self, tokens: &[u32], base_pos: usize, verify: bool) {
        let start = self.flash_trace_start();
        let stats = std::cell::RefCell::new(FlashTargetTiming::default());
        self.forward_chunk_qwen4exp_timed(tokens, base_pos, verify, if start.is_some() { Some(&stats) } else { None });
        self.flash_trace_finish(start, if verify { "verify" } else { "prefill_target" }, base_pos, tokens.len(), stats.into_inner());
    }

    pub(crate) fn forward_chunk_qwen4exp_timed(&self, tokens: &[u32], base_pos: usize, verify: bool,
        timing: Option<&std::cell::RefCell<FlashTargetTiming>>) {
        assert!(base_pos.checked_add(tokens.len()).is_some_and(|n| n <= self.st.max_seq)
            && tokens.iter().all(|&t| (t as usize) < self.arch.vocab), "batch exceeds model bounds");
        let q = self.arch.qwen4exp.as_ref().unwrap();
        let (hc, hc_lr) = (q.hc_mult, q.hc_low_rank);
        let m = tokens.len() as u32;
        let d = self.d as u32;
        let f4 = std::mem::size_of::<f32>() as u64;
        let n = self.arch.n_layers;
        assert!(tokens.len() <= MAXM, "chunk of {} exceeds MAXM {MAXM}", tokens.len());
        if verify {
            assert!(self.sp.snapshot_rows == 1 || tokens.len() <= self.sp.snapshot_rows,
                "prefix verification exceeds allocated snapshot rows; raise OJAS_MTP_DRAFT");
            self.sp.verified_rows.set(tokens.len());
        }
        assert!(!self.strm.stream || self.resident_any() || tokens.len() <= self.strm.ubatch,
            "streamed chunk of {} exceeds --ubatch-size {}", tokens.len(), self.strm.ubatch);
        // PLE hashes preceding token IDs on the host. Verification must expose
        // its own input rows before encoding: row i depends on rows i-1/i-2,
        // including drafts that have not yet been committed by the frontend.
        {
            let mut history = self.seq().session_tokens.borrow_mut();
            if base_pos <= history.len() {
                history.truncate(base_pos);
                history.extend_from_slice(tokens);
            }
        }

        let finish = |cb: &metal::CommandBufferRef, encoded: Option<std::time::Instant>| {
            let encode_s = encoded.map(|t| t.elapsed().as_secs_f64());
            let submitted = timing.map(|_| std::time::Instant::now());
            let _ = ojas_metal::commit_and_wait_checked(cb, "qwen4exp batched graph");
            if let Some(stats) = timing {
                let wait_s = submitted.unwrap().elapsed().as_secs_f64();
                let (gs, ge): (f64, f64) = unsafe {
                    (objc::msg_send![cb, GPUStartTime], objc::msg_send![cb, GPUEndTime])
                };
                let mut s = stats.borrow_mut();
                s.encode_s += encode_s.unwrap(); s.submit_wait_s += wait_s; s.commands += 1;
                if gs > 0.0 && ge >= gs && gs.is_finite() && ge.is_finite() { s.gpu_s += ge-gs; }
                else { s.invalid_gpu_timestamps += 1; }
            }
        };
        let cb_run = |f: &dyn Fn(&metal::ComputeCommandEncoderRef)| {
            objc::rc::autoreleasepool(|| {
                let encoded = timing.map(|_| std::time::Instant::now());
                let cb = self.gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                f(enc);
                enc.end_encoding();
                finish(cb, encoded);
            });
        };
        // prologue: embed each token into its row of x, then fan out to hc streams
        let prologue = |enc: &metal::ComputeCommandEncoderRef| {
            for (i, &t) in tokens.iter().enumerate() {
                self.embed_named_off(enc, "token_embd.weight", t, d, (i as u64) * (d as u64) * f4);
            }
            self.bar(enc);
            enc.set_compute_pipeline_state(&self.p["hc_broadcast"]);
            enc.set_buffer(0, Some(&self.st.x), 0);
            enc.set_buffer(1, Some(&self.st.hc_res), 0);
            enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &hc as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(((d + 63) / 64) as u64, hc as u64, m as u64),
                                       MTLSize::new(64, 1, 1));
            self.bar(enc);
        };

        if self.resident_any() {
            if let Some(rs) = &self.strm.expert_residency { rs.renew(); }
            // Partial/full residency. A run of consecutive resident layers shares
            // one command buffer (their experts are wired, so the router's moe_idx
            // feeds the expert kernels on-GPU with no host gather). A streamed layer
            // keeps the route -> CPU gather -> experts split. `group` bounds
            // per-buffer residency so a run never wires more than it should.
            let group = self.cfg.flash_resident_group.max(1);
            let mut l = 0usize;
            while l < n {
                let first = l == 0;
                if self.is_resident(l) {
                    let mut hi = l;
                    while hi < n && self.is_resident(hi) && (hi - l) < group { hi += 1; }
                    let last = hi == n;
                    cb_run(&|enc| {
                        if first { prologue(enc); }
                        for ll in l..hi {
                            self.qwen4exp_layer_route(enc, ll, tokens, base_pos, d, hc, hc_lr, verify);
                            self.qwen4exp_layer_experts(enc, ll, d, hc, m);
                        }
                        if last && verify { self.qwen4exp_verify_tail(enc, d, hc, hc_lr, m); }
                    });
                    l = hi;
                } else {
                    // Streamed layer: route (writes moe_idx), CPU gather, experts.
                    cb_run(&|enc| {
                        if first { prologue(enc); }
                        self.qwen4exp_layer_route(enc, l, tokens, base_pos, d, hc, hc_lr, verify);
                    });
                    self.gather_experts_m(l, tokens.len());
                    let last = l + 1 == n;
                    cb_run(&|enc| {
                        self.qwen4exp_layer_experts(enc, l, d, hc, m);
                        if last && verify { self.qwen4exp_verify_tail(enc, d, hc, hc_lr, m); }
                    });
                    l += 1;
                }
            }
            return;
        }
        if !self.strm.stream {
            cb_run(&|enc| {
                prologue(enc);
                for l in 0..n {
                    self.qwen4exp_layer_route(enc, l, tokens, base_pos, d, hc, hc_lr, verify);
                    self.qwen4exp_layer_experts(enc, l, d, hc, m);
                }
                if verify { self.qwen4exp_verify_tail(enc, d, hc, hc_lr, m); }
            });
            return;
        }
        if let Some(stats) = timing.filter(|s| s.borrow().split_stages) {
            assert!(verify, "stage profiling requires the verification tail");
            let stage = |bucket: fn(&mut FlashTargetTiming) -> &mut f64,
                         f: &dyn Fn(&metal::ComputeCommandEncoderRef)| {
                let before = stats.borrow().gpu_s;
                cb_run(f);
                let elapsed = stats.borrow().gpu_s - before;
                *bucket(&mut stats.borrow_mut()) += elapsed;
            };
            stage(|s| &mut s.route_gpu_s, &|enc| {
                prologue(enc);
                self.qwen4exp_layer_route(enc, 0, tokens, base_pos, d, hc, hc_lr, verify);
            });
            for l in 0..n {
                let start = std::time::Instant::now();
                self.gather_experts_m_timed(l, tokens.len(), timing);
                stats.borrow_mut().gather_s += start.elapsed().as_secs_f64();
                stage(|s| &mut s.expert_gpu_s, &|enc| self.qwen4exp_layer_experts(enc, l, d, hc, m));
                if l + 1 < n {
                    stage(|s| &mut s.route_gpu_s, &|enc| self.qwen4exp_layer_route(enc, l+1, tokens, base_pos, d, hc, hc_lr, verify));
                } else if verify {
                    stage(|s| &mut s.tail_gpu_s, &|enc| self.qwen4exp_verify_tail(enc, d, hc, hc_lr, m));
                }
            }
            return;
        }
        // One command buffer per layer, not two. The gather has to sit between a
        // layer's router and its experts, but layer L's experts and layer L+1's
        // router have no gather between them, so they share a buffer: the sequence
        // is [prologue, route 0] gather [experts 0, route 1] gather [experts 1, ...].
        // A serial encoder is memory-coherent between commands, so route L+1 reading
        // the hc_res that experts L just wrote needs no barrier beyond that.
        // Halves the commit+wait count, ~0.5 ms each on a 48-layer model.
        cb_run(&|enc| {
            prologue(enc);
            self.qwen4exp_layer_route(enc, 0, tokens, base_pos, d, hc, hc_lr, verify);
        });
        for l in 0..n {
            let gathered = timing.map(|_| std::time::Instant::now());
            self.gather_experts_m_timed(l, tokens.len(), timing);
            if let Some(stats) = timing { stats.borrow_mut().gather_s += gathered.unwrap().elapsed().as_secs_f64(); }
            objc::rc::autoreleasepool(|| {
                let encoded = timing.map(|_| std::time::Instant::now());
                let cb = self.gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                self.qwen4exp_layer_experts(enc, l, d, hc, m);
                if l + 1 < n {
                    self.qwen4exp_layer_route(enc, l + 1, tokens, base_pos, d, hc, hc_lr, verify);
                } else if verify {
                    self.qwen4exp_verify_tail(enc, d, hc, hc_lr, m);
                }
                enc.end_encoding();
                let instrumented = super::audit::instrumented_layer() == Some(l);
                // Prefetch before commit. dbuf_prefetch reads moe_idx synchronously to
                // build its candidate list, and this buffer is about to overwrite layer
                // L+1's slot with the real routing, so reading it after commit would
                // race the GPU. Before commit it reads the previous token's routing,
                // the intended ~90%-accurate predictor.
                if self.cfg.moe_dbuf && l + 1 < n { self.dbuf_prefetch_m(l + 1, tokens.len()); }
                finish(cb, encoded);
                // `finish` waited for completion, so the GPU's hashes are readable and
                // the allocations were pinned for the whole dispatch.
                if instrumented {
                    let (bad, unresolved, compared) = self.check_expert_table_hash();
                    let total = self.strm.expert_hash_records.borrow().len();
                    if total == 0 {
                        tracing::error!(target: "audit",
                            "layer {l}: instrumentation produced NO records — nothing was verified");
                    } else if unresolved > 0 {
                        tracing::error!(target: "audit",
                            "layer {l}: INCOMPLETE — {unresolved}/{total} slots unresolved \
                             (address unmappable or hashed range outside its allocation); \
                             {} of {compared} compared slots mismatched", bad.len());
                    } else if bad.is_empty() {
                        // Also report which experts were selected: correct expert bytes
                        // with different routing produce coherent-but-different output,
                        // indistinguishable from a weights fault at the token level.
                        let recs = self.strm.expert_hash_records.borrow();
                        let experts: Vec<u32> = recs.iter().filter(|r| r.kind == 0).map(|r| r.expert).collect();
                        let sig = experts.iter().fold(2166136261u32, |h, &e| (h ^ e).wrapping_mul(16777619));
                        tracing::info!(target: "audit",
                            "layer {l}: all {compared} slots verified against the SOURCE bytes of \
                             the routed expert | path={} | routed {} experts, routing_sig=0x{sig:08x}, first={:?}",
                            if recs.first().map(|r| r.direct).unwrap_or(false) { "direct" } else { "scratch" },
                            experts.len(), &experts[..experts.len().min(10)]);
                    } else {
                        tracing::error!(target: "audit",
                            "layer {l}: {}/{compared} slots DISAGREE — GPU read bytes that are not \
                             the routed expert's (path={})", bad.len(),
                            if bad.first().map(|(r, _)| r.direct).unwrap_or(false) { "direct" } else { "scratch" });
                        for (r, gpu) in bad.iter().take(6) {
                            tracing::error!(target: "audit",
                                "  kind={} slot={} expert={} addr=0x{:x} off={} len={} host=0x{:08x} gpu=0x{:08x}",
                                r.kind, r.slot, r.expert, r.address, r.offset_in_tensor, r.len, r.expected_hash, gpu);
                            tracing::error!(target: "audit", "    path={} ggml_type={}", if r.direct { "direct" } else { "scratch" }, r.ggml_type);
                        }
                    }
                }
            });
        }
        if !verify && self.sp.mtp.map(|mc| mc.hnorm_len == (d * hc) as usize).unwrap_or(false) {
            // The catch-up needs one hidden per position and prefill runs no head.
            let n_el = m * d * hc;
            cb_run(&|enc| self.enc_reduce(enc, "copy_buf", &[(&self.sp.mtp_h, 0), (&self.st.hc_res, 1)],
                                          &[(2, n_el)], &[], ((n_el + 63) / 64) as u64, 64));
        }
    }

    /// MTP verify tail: terminal hc mixer, then the hidden rows the NEXT draft
    /// consumes, then logits and a per-row argmax into `tmp[row]`.
    fn qwen4exp_verify_tail(&self, enc: &metal::ComputeCommandEncoderRef,
                            d: u32, hc: u32, hc_lr: u32, m: u32) {
        let vocab = self.arch.vocab as u32;
        let grouped = self.sp.mtp.map(|mc| mc.hnorm_len == (d * hc) as usize).unwrap_or(false);
        if grouped {
            let n = m * d * hc;
            self.enc_reduce(enc, "copy_buf", &[(&self.sp.mtp_h, 0), (&self.st.hc_res, 1)],
                            &[(2, n)], &[], ((n + 63) / 64) as u64, 64);
            self.bar(enc);
        }
        self.hc_mix_m(enc, "output_hc", d, hc, hc_lr, m, false);
        self.bar(enc);
        if !grouped {
            self.enc_reduce(enc, "copy_buf", &[(&self.sp.mtp_h, 0), (&self.st.hc_mixed, 1)],
                            &[(2, m * d)], &[], ((m * d + 63) / 64) as u64, 64);
            self.bar(enc);
        }
        let lm = self.arch.lm_head.clone();
        self.gemm_m(enc, &lm, &self.st.hc_mixed, &self.st.logits, d, vocab, m);
        self.bar(enc);
        for row in 0..m as u64 {
            enc.set_compute_pipeline_state(&self.p["argmax"]);
            enc.set_buffer(0, Some(&self.st.logits), row * (vocab as u64) * 4);
            enc.set_buffer(1, Some(&self.st.tmp), row * 4);
            enc.set_bytes(2, 4, &vocab as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(self.tune.max_tg.min(1024), 1, 1));
        }
    }

    /// Run the NextN draft block over a batch of already-processed tokens purely to
    /// fill its KV cache. llama.cpp's driver calls this a catch-up decode and runs
    /// it after every target batch; without it the block attends over an empty
    /// cache and falls back to echoing its own input.
    ///
    /// `mtp_h` holds the target's hidden for each row of the batch just processed.
    /// The draft at row i needs the hidden one position earlier, so the rows are
    /// shifted into `mtp_hprev` with row MAXM carrying the last one across batches.
    /// NextN projects [normalized embedding, normalized hidden] independently
    /// for every hyper-connection stream. Pooling before projection loses the
    /// residual differences that the subsequent nonlinear HC mixer needs.
    pub(crate) fn qwen4exp_mtp_project(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        tokens: &[u32],
        hidden: &metal::Buffer,
        hidden_offset: u64,
    ) {
        let mc = self.sp.mtp.expect("MTP projection requires draft weights");
        let q = self.arch.qwen4exp.as_ref().unwrap();
        let (d, hc, m) = (self.d as u32, q.hc_mult, tokens.len() as u32);
        let grouped = mc.hnorm_len == (d * hc) as usize;
        let hrow = if grouped { d * hc } else { d } as u64;
        let f4 = 4u64;
        let p = |s: &str| format!("blk.{}.{}", mc.layer, s);
        let embw = if mc.has_embed {
            p("nextn.embed_tokens.weight")
        } else {
            "token_embd.weight".to_string()
        };
        for (i, &t) in tokens.iter().enumerate() {
            let r = i as u64;
            self.embed_named_off(enc, &embw, t, d, r * (d as u64) * f4);
        }
        self.bar(enc);
        for i in 0..m as u64 {
            enc.set_compute_pipeline_state(&self.p["rmsnorm"]);
            enc.set_buffer(0, Some(&self.st.x), i * (d as u64) * f4);
            enc.set_buffer(1, Some(&self.wt.w32[&p("nextn.enorm.weight")]), 0);
            enc.set_buffer(
                2,
                Some(if grouped {
                    &self.st.h
                } else {
                    &self.sp.mtp_cat
                }),
                if grouped {
                    i * (d as u64) * f4
                } else {
                    i * 2 * (d as u64) * f4
                },
            );
            enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            if grouped {
                self.hc_gnorm_off(
                    enc,
                    hidden,
                    hidden_offset + i * hrow * f4,
                    &p("nextn.hnorm.weight"),
                    &self.st.hc_gated,
                    i * hrow * f4,
                    d,
                    hc,
                );
            } else {
                enc.set_compute_pipeline_state(&self.p["rmsnorm"]);
                enc.set_buffer(0, Some(hidden), hidden_offset + i * hrow * f4);
                enc.set_buffer(1, Some(&self.wt.w32[&p("nextn.hnorm.weight")]), 0);
                enc.set_buffer(2, Some(&self.sp.mtp_cat), (i * 2 + 1) * (d as u64) * f4);
                enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            }
            self.bar(enc);
        }
        if grouped {
            enc.set_compute_pipeline_state(&self.p["nextn_concat_streams"]);
            enc.set_buffer(0, Some(&self.st.h), 0);
            enc.set_buffer(1, Some(&self.st.hc_gated), 0);
            enc.set_buffer(2, Some(&self.sp.mtp_cat), 0);
            enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &hc as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(d.div_ceil(64) as u64, hc as u64, m as u64),
                MTLSize::new(64, 1, 1),
            );
            self.bar(enc);
            self.gemm_m(
                enc,
                &p("nextn.eh_proj.weight"),
                &self.sp.mtp_cat,
                &self.st.hc_res,
                2 * d,
                d,
                m * hc,
            );
            self.bar(enc);
        } else {
            self.gemm_m(
                enc,
                &p("nextn.eh_proj.weight"),
                &self.sp.mtp_cat,
                &self.st.x,
                2 * d,
                d,
                m,
            );
            self.bar(enc);
            enc.set_compute_pipeline_state(&self.p["hc_broadcast"]);
            enc.set_buffer(0, Some(&self.st.x), 0);
            enc.set_buffer(1, Some(&self.st.hc_res), 0);
            enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &hc as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((d + 63) / 64) as u64, hc as u64, m as u64),
                MTLSize::new(64, 1, 1),
            );
            self.bar(enc);
        }
    }

    pub(crate) fn qwen4exp_mtp_catchup(&self, tokens: &[u32], base_pos: usize) {
        if self.cfg.no_spec { return; }
        let Some(mc) = self.sp.mtp else { return };
        if tokens.is_empty() { return }
        let trace = self.flash_trace_start();
        let q = self.arch.qwen4exp.as_ref().unwrap();
        let (hc, hc_lr) = (q.hc_mult, q.hc_low_rank);
        let d = self.d as u32;
        let m = tokens.len() as u32;
        let l = mc.layer;
        let p = |s: &str| format!("blk.{l}.{s}");
        let grouped = mc.hnorm_len == (d * hc) as usize;
        let hrow = if grouped { d * hc } else { d } as u64;   // elements per stored hidden
        let f4 = 4u64;
        let carry = (MAXM as u64) * hrow * f4;
        if base_pos == 0 {
            unsafe { std::ptr::write_bytes((self.sp.mtp_hprev.contents() as *mut u8).add(carry as usize), 0, (hrow*f4) as usize); }
        }

        let shift = |enc: &metal::ComputeCommandEncoderRef| {
            self.enc_reduce_off(enc, "copy_buf",
                &[(&self.sp.mtp_hprev, 0, 0), (&self.sp.mtp_hprev, 1, carry)],
                &[(2, hrow as u32)], &[], ((hrow as u32 + 63) / 64) as u64, 64);
            if m > 1 {
                self.enc_reduce_off(enc, "copy_buf",
                    &[(&self.sp.mtp_hprev, 0, hrow * f4), (&self.sp.mtp_h, 1, 0)],
                    &[(2, (m - 1) * hrow as u32)], &[], (((m - 1) * hrow as u32 + 63) / 64) as u64, 64);
            }
            self.enc_reduce_off(enc, "copy_buf",
                &[(&self.sp.mtp_hprev, 0, carry), (&self.sp.mtp_h, 1, (m as u64 - 1) * hrow * f4)],
                &[(2, hrow as u32)], &[], ((hrow as u32 + 63) / 64) as u64, 64);
            self.bar(enc);
        };

        let combiner = |enc: &metal::ComputeCommandEncoderRef| {
            self.qwen4exp_mtp_project(enc, tokens, &self.sp.mtp_hprev, 0);
            self.hc_mix_m(enc, &p("hc_attn"), d, hc, hc_lr, m, true);
            self.bar(enc);
            self.qwen4exp_qsa_m(enc, l, base_pos, d, m);
            // Catchup persists only K/V and the shifted target-hidden carry.
            // The next draft projects target hiddens afresh; no FFN output from
            // this pass is consumed. Avoid routing/gathering experts merely to
            // overwrite scratch, which also needlessly evicts target experts.
        };

        let run = |f: &dyn Fn(&metal::ComputeCommandEncoderRef)| {
            objc::rc::autoreleasepool(|| {
                let cb = self.gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                f(enc);
                enc.end_encoding(); let _ = ojas_metal::commit_and_wait_checked(cb, "qwen4exp batched graph");
            });
        };
        run(&|enc| { shift(enc); combiner(enc); });
        self.flash_trace_finish(trace, "catchup", base_pos, tokens.len(), FlashTargetTiming::default());
    }
}
