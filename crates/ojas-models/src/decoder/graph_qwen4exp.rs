#![allow(clippy::too_many_arguments)]
use super::*;
use metal::MTLSize;
use objc::{msg_send, sel, sel_impl};
use std::ffi::c_void;

impl<'a> DecoderGpu<'a> {
    /// Diagnostic decomposition of one real resident scalar token. Command
    /// boundaries are added between dependency groups, so the category sum is a
    /// GPU-compute attribution rather than a production wall-time measurement.
    pub fn profile_qwen4exp_resident_token(
        &self,
        token: u32,
        pos: usize,
    ) -> Vec<(&'static str, f64)> {
        assert!(self.arch.qwen4exp.is_some() && self.strm.resident_layers == self.arch.n_layers);
        let q = self.arch.qwen4exp.as_ref().unwrap();
        let (d, hc, hc_lr) = (self.d as u32, q.hc_mult, q.hc_low_rank);
        let mut totals = std::collections::HashMap::<&'static str, f64>::new();
        let mut stage = |name: &'static str, f: &dyn Fn(&metal::ComputeCommandEncoderRef)| {
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            f(enc);
            enc.end_encoding();
            let _ = ojas_metal::commit_and_wait_checked(cb, "qwen4exp resident profile");
            let (start, end): (f64, f64) =
                unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
            *totals.entry(name).or_default() += (end - start) * 1e3;
        };

        stage("embed", &|enc| {
            self.encode_qwen4exp(
                enc,
                token,
                pos,
                d,
                (pos + 1) as u32,
                0,
                0,
                true,
                false,
                MoePhase::Full,
            )
        });
        for l in 0..self.arch.n_layers {
            if q.ple_layers.contains(&(l as u32)) && !self.cfg.no_ple {
                // qwen4exp_ple performs its bounded CPU row copy before encoding.
                stage("ple", &|enc| self.qwen4exp_ple(enc, l, token, pos, d, 0));
            }
            stage("hc_attn", &|enc| {
                self.hc_mix(enc, &format!("blk.{l}.hc_attn"), d, hc, hc_lr, true)
            });
            let is_ssm = self.arch.layers[l].is_ssm;
            stage(if is_ssm { "gdn" } else { "attention" }, &|enc| {
                if is_ssm {
                    self.qwen4exp_gdn(enc, l, d);
                } else {
                    self.qwen4exp_qsa(enc, l, pos, (pos + 1) as u32, d);
                }
            });
            stage("hc_ffn", &|enc| {
                self.hc_combine(enc, &self.st.h, d, hc);
                self.hc_mix(enc, &format!("blk.{l}.hc_ffn"), d, hc, hc_lr, true);
            });
            stage("shared_route", &|enc| {
                self.qwen4exp_moe(enc, l, d, MoePhase::Route)
            });
            stage("routed_experts", &|enc| {
                self.qwen4exp_moe(enc, l, d, MoePhase::Experts);
                self.hc_combine(enc, &self.st.h, d, hc);
            });
        }
        stage("head", &|enc| {
            self.hc_mix(enc, "output_hc", d, hc, hc_lr, false);
            self.mm(
                enc,
                "plain",
                &self.arch.lm_head,
                &self.st.hc_mixed,
                &self.st.logits,
                d,
                self.arch.vocab as u32,
                None,
            );
        });
        let order = [
            "embed",
            "ple",
            "hc_attn",
            "gdn",
            "attention",
            "hc_ffn",
            "shared_route",
            "routed_experts",
            "head",
        ];
        order
            .into_iter()
            .map(|name| (name, totals.remove(name).unwrap_or(0.0)))
            .collect()
    }

    /// qwen4exp decode (single token). The residual is `hc` parallel d-wide
    /// streams; each block reads one hyper-connection-mixed d-wide view and
    /// scatters its output back through per-stream injection weights. Recurrent
    /// layers run the gated delta-net mixer, full-attention layers the sparse
    /// attention, and every layer's FFN is a shared + routed mixture of experts.
    /// The terminal mixer serves as the output norm.
    ///
    /// `phase` splits the layer for disk-streamed experts: `Route` runs everything
    /// up to and including the router (leaving `moe_idx` for the CPU gather),
    /// `Experts` runs the routed GEMMs and the FFN hyper-connection scatter, and
    /// `Full` runs both in one command buffer. The state that has to survive the
    /// gap between the two command buffers — `hc_res`, `hc_inject`, `h`, `moe_idx`,
    /// `tmp`, `moe_sh` — all lives in persistent GPU buffers.
    pub(crate) fn encode_qwen4exp(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        token: u32,
        pos: usize,
        d: u32,
        seq: u32,
        l_start: usize,
        l_end: usize,
        do_embed: bool,
        do_head: bool,
        phase: MoePhase,
    ) {
        let q = self.arch.qwen4exp.as_ref().unwrap();
        let (hc, hc_lr) = (q.hc_mult, q.hc_low_rank);

        if do_embed {
            self.embed_token(enc, token, d);
            self.bar(enc);
            enc.set_compute_pipeline_state(&self.p["hc_broadcast"]);
            enc.set_buffer(0, Some(&self.st.x), 0);
            enc.set_buffer(1, Some(&self.st.hc_res), 0);
            enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &hc as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((d + 63) / 64) as u64, hc as u64, 1),
                MTLSize::new(64, 1, 1),
            );
            self.bar(enc);
        }

        for l in l_start..l_end {
            let is_recr = self.arch.layers[l].is_ssm;
            if !matches!(phase, MoePhase::Experts) {
                // Diagnostic ablation only. Normal inference includes PLE,
                // matching llama.cpp's n-gram hash and selected-row gather.
                if q.ple_layers.contains(&(l as u32)) && !self.cfg.no_ple {
                    self.qwen4exp_ple(enc, l, token, pos, d, 0);
                    self.bar(enc);
                }

                // token-mixer hyper-connection → hc_mixed, hc_inject
                self.hc_mix(enc, &format!("blk.{l}.hc_attn"), d, hc, hc_lr, true);
                self.bar(enc);
                if is_recr {
                    self.qwen4exp_gdn(enc, l, d);
                } else {
                    self.qwen4exp_qsa(enc, l, pos, seq, d);
                }
                self.bar(enc);
                self.hc_combine(enc, &self.st.h, d, hc);
                self.bar(enc);

                // FFN hyper-connection → hc_mixed, hc_inject
                self.hc_mix(enc, &format!("blk.{l}.hc_ffn"), d, hc, hc_lr, true);
                self.bar(enc);
            }
            self.qwen4exp_moe(enc, l, d, phase);
            if matches!(phase, MoePhase::Route) {
                continue;
            }
            self.bar(enc);
            self.hc_combine(enc, &self.st.h, d, hc);
            self.bar(enc);
        }

        if do_head {
            if let Some(mc) = self.sp.mtp {
                // llama.cpp hands the MTP block the "hidden state before final output
                // norm". Which tensor that is depends on the width of nextn.hnorm: a
                // per-stream gamma wants the residual streams themselves, a d-wide one
                // wants the collapse the mixer produces.
                let grouped = mc.hnorm_len == (d * hc) as usize;
                let n = if grouped { d * hc } else { d };
                let src = if grouped {
                    &self.st.hc_res
                } else {
                    &self.st.hc_mixed
                };
                if grouped {
                    self.enc_reduce(
                        enc,
                        "copy_buf",
                        &[(&self.sp.mtp_h, 0), (src, 1)],
                        &[(2, n)],
                        &[],
                        ((n + 63) / 64) as u64,
                        64,
                    );
                    self.bar(enc);
                }
            }
            self.hc_mix(enc, "output_hc", d, hc, hc_lr, false);
            self.bar(enc);
            let lm = self.arch.lm_head.clone();
            self.mm(
                enc,
                "plain",
                &lm,
                &self.st.hc_mixed,
                &self.st.logits,
                d,
                self.arch.vocab as u32,
                None,
            );
        }
    }

    /// One hyper-connection mixer: per-stream RMSNorm, a low-rank down/silu/up
    /// gate, and the mean collapse to `hc_mixed`. When `with_inject`, also produce
    /// the per-stream injection weights in `hc_inject`. `base` is the tensor-name
    /// stem, e.g. "blk.5.hc_attn" or "output_hc".
    fn hc_mix(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        base: &str,
        d: u32,
        hc: u32,
        hc_lr: u32,
        with_inject: bool,
    ) {
        let w = |s: &str| format!("{base}{s}");
        let hc_dim = d * hc;
        // xn = per-stream rmsnorm(hc_res) * gamma
        enc.set_compute_pipeline_state(&self.p["hc_rmsnorm"]);
        enc.set_buffer(0, Some(&self.st.hc_res), 0);
        enc.set_buffer(1, Some(&self.wt.w32[&w("_norm.weight")]), 0);
        enc.set_buffer(2, Some(&self.st.hc_xn), 0);
        enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
        enc.set_bytes(5, 4, &hc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(hc as u64, 1, 1), MTLSize::new(256, 1, 1));
        self.bar(enc);
        let down = w("_down.weight");
        let up = w("_up.weight");
        let inject = w("_inject.weight");
        let fused = self.cfg.flash_hc_fused
            && with_inject
            && d == 2560
            && hc == 4
            && hc_lr == 320
            && self.wt.w_qtype.get(&down) == Some(&8)
            && self.wt.w_qtype.get(&up) == Some(&8)
            && self.wt.repr(&inject) == Repr::F32;
        if fused {
            enc.set_compute_pipeline_state(&self.p["hc_down_inject_fused"]);
            enc.set_buffer(0, Some(&self.st.hc_xn), 0);
            enc.set_buffer(
                1,
                Some(&self.wt.wq[&down]),
                self.wt.w_off.get(&down).copied().unwrap_or(0),
            );
            enc.set_buffer(2, Some(&self.wt.w32[&inject]), 0);
            enc.set_buffer(3, Some(&self.st.hc_lo), 0);
            enc.set_buffer(4, Some(&self.st.hc_inject), 0);
            enc.dispatch_thread_groups(
                MTLSize::new((hc_lr + hc) as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            self.bar(enc);
            enc.set_compute_pipeline_state(&self.p["hc_up_gate_collapse_fused"]);
            enc.set_buffer(0, Some(&self.st.hc_lo), 0);
            enc.set_buffer(
                1,
                Some(&self.wt.wq[&up]),
                self.wt.w_off.get(&up).copied().unwrap_or(0),
            );
            enc.set_buffer(2, Some(&self.st.hc_xn), 0);
            enc.set_buffer(3, Some(&self.st.hc_mixed), 0);
            enc.dispatch_thread_groups(MTLSize::new(d as u64, 1, 1), MTLSize::new(128, 1, 1));
            return;
        }
        // lo = silu((down @ xn) / hc)
        self.mm(
            enc,
            "plain",
            &down,
            &self.st.hc_xn,
            &self.st.hc_lo,
            hc_dim,
            hc_lr,
            None,
        );
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p["hc_silu_scale"]);
        enc.set_buffer(0, Some(&self.st.hc_lo), 0);
        enc.set_bytes(1, 4, &hc_lr as *const u32 as *const c_void);
        enc.set_bytes(2, 4, &hc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((hc_lr + 63) / 64) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        self.bar(enc);
        // gated = xn * sigmoid(up @ lo)
        self.mm(
            enc,
            "plain",
            &w("_up.weight"),
            &self.st.hc_lo,
            &self.st.hc_graw,
            hc_lr,
            hc_dim,
            None,
        );
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p["hc_gate"]);
        enc.set_buffer(0, Some(&self.st.hc_xn), 0);
        enc.set_buffer(1, Some(&self.st.hc_graw), 0);
        enc.set_buffer(2, Some(&self.st.hc_gated), 0);
        enc.set_bytes(3, 4, &hc_dim as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((hc_dim + 63) / 64) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        self.bar(enc);
        // mixed = mean over streams
        enc.set_compute_pipeline_state(&self.p["hc_collapse"]);
        enc.set_buffer(0, Some(&self.st.hc_gated), 0);
        enc.set_buffer(1, Some(&self.st.hc_mixed), 0);
        enc.set_bytes(2, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &hc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((d + 63) / 64) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        if with_inject {
            self.mm(
                enc,
                "plain",
                &w("_inject.weight"),
                &self.st.hc_xn,
                &self.st.hc_inject,
                hc_dim,
                hc,
                None,
            );
        }
    }

    /// Scatter a block's d-wide output back into every residual stream, weighted
    /// per stream by the injection vector.
    fn hc_combine(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        block: &metal::Buffer,
        d: u32,
        hc: u32,
    ) {
        enc.set_compute_pipeline_state(&self.p["hc_combine"]);
        enc.set_buffer(0, Some(&self.st.hc_res), 0);
        enc.set_buffer(1, Some(block), 0);
        enc.set_buffer(2, Some(&self.st.hc_inject), 0);
        enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &hc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((d + 63) / 64) as u64, hc as u64, 1),
            MTLSize::new(64, 1, 1),
        );
    }

    /// Grouped RMSNorm: per-stream RMSNorm over d, then the [d*hc] affine weight.
    pub(crate) fn hc_gnorm_off(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x: &metal::Buffer,
        x_off: u64,
        gamma: &str,
        out: &metal::Buffer,
        out_off: u64,
        d: u32,
        hc: u32,
    ) {
        self.hc_gnorm_inner(enc, x, x_off, gamma, out, out_off, d, hc)
    }

    fn hc_gnorm(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x: &metal::Buffer,
        x_off: u64,
        gamma: &str,
        out: &metal::Buffer,
        d: u32,
        hc: u32,
    ) {
        self.hc_gnorm_inner(enc, x, x_off, gamma, out, 0, d, hc)
    }

    fn hc_gnorm_inner(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x: &metal::Buffer,
        x_off: u64,
        gamma: &str,
        out: &metal::Buffer,
        out_off: u64,
        d: u32,
        hc: u32,
    ) {
        enc.set_compute_pipeline_state(&self.p["hc_rmsnorm"]);
        enc.set_buffer(0, Some(x), x_off);
        enc.set_buffer(1, Some(&self.wt.w32[gamma]), 0);
        enc.set_buffer(2, Some(out), out_off);
        enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
        enc.set_bytes(5, 4, &hc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(hc as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// CPU-gather the selected packed PLE rows, then decode with the GPU kernel.
    /// Each PLE layer and batch row has disjoint staging storage, so host writes
    /// during encoding cannot overwrite an earlier dispatch's inputs.
    pub(crate) fn qwen4exp_ple(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        l: usize,
        token: u32,
        pos: usize,
        d: u32,
        row: usize,
    ) {
        let q = self.arch.qwen4exp.as_ref().unwrap();
        let (hc, nh, hd) = (q.hc_mult, q.ple_n_heads, q.ple_head_dim);
        let table = self.wt.ple.as_ref().expect("PLE CPU table was not loaded");
        let layer = q
            .ple_layers
            .iter()
            .position(|&v| v as usize == l)
            .expect("not a PLE layer");
        assert!(row < MAXM, "PLE batch exceeds staging capacity");
        let stride = table.row_bytes * nh as usize;
        let offset = (layer * MAXM + row) * stride.div_ceil(4) * 4;
        assert!(offset + stride <= self.st.ple_rows.length() as usize);
        let rows = q.ple_rows(token, pos, &self.seq().session_tokens.borrow());
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                (self.st.ple_rows.contents() as *mut u8).add(offset),
                stride,
            )
        };
        table
            .copy_rows(&rows, dst)
            .expect("validated PLE row gather failed");
        let p = |s: &str| format!("blk.{l}.{s}");
        let kernel = if table.format == 0 {
            "ple_gather"
        } else {
            "ple_gather_iq4nl"
        };
        enc.set_compute_pipeline_state(&self.p[kernel]);
        enc.set_buffer(0, Some(&self.st.ple_rows), offset as u64);
        enc.set_buffer(1, Some(&self.st.ple_idx), 0);
        enc.set_buffer(2, Some(&self.st.ssm_qkv), 0);
        enc.set_bytes(3, 4, &hd as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nh as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((d + 63) / 64) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        self.bar(enc);
        self.mm(
            enc,
            "plain",
            &p("ple_key.weight"),
            &self.st.ssm_qkv,
            &self.st.hc_gated,
            d,
            d * hc,
            None,
        );
        self.mm(
            enc,
            "plain",
            &p("ple_value.weight"),
            &self.st.ssm_qkv,
            &self.st.hc_mixed,
            d,
            d,
            None,
        );
        self.bar(enc);
        let res_off = (row * (d * hc) as usize * 4) as u64;
        self.hc_gnorm(
            enc,
            &self.st.hc_gated,
            0,
            &p("ple_norm_key.weight"),
            &self.st.hc_xn,
            d,
            hc,
        );
        self.hc_gnorm(
            enc,
            &self.st.hc_res,
            res_off,
            &p("ple_norm_query.weight"),
            &self.st.hc_gated,
            d,
            hc,
        );
        self.bar(enc);
        // per-stream indexer gate
        enc.set_compute_pipeline_state(&self.p["ple_gate"]);
        enc.set_buffer(0, Some(&self.st.hc_xn), 0);
        enc.set_buffer(1, Some(&self.st.hc_gated), 0);
        enc.set_buffer(2, Some(&self.st.hc_inject), 0);
        enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(hc as u64, 1, 1), MTLSize::new(256, 1, 1));
        self.bar(enc);
        // gated value → hc_xn, then its grouped norm → hc_graw
        enc.set_compute_pipeline_state(&self.p["ple_bmul"]);
        enc.set_buffer(0, Some(&self.st.hc_mixed), 0);
        enc.set_buffer(1, Some(&self.st.hc_inject), 0);
        enc.set_buffer(2, Some(&self.st.hc_xn), 0);
        enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &hc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((d + 63) / 64) as u64, hc as u64, 1),
            MTLSize::new(64, 1, 1),
        );
        self.bar(enc);
        self.hc_gnorm(
            enc,
            &self.st.hc_xn,
            0,
            &p("ple_norm_conv.weight"),
            &self.st.hc_graw,
            d,
            hc,
        );
        self.bar(enc);
        // res += value*gate + silu(dilated causal conv).
        let kern = q.ple_conv_kernel;
        let dilation = q.ple_ngram_size;
        let sc = self.arch.ssm.unwrap();
        let offset = ((sc.conv_kernel - 1) * (sc.d_inner + 2 * sc.n_group * sc.d_state)) as u64 * 4;
        enc.set_compute_pipeline_state(&self.p["ple_finish"]);
        enc.set_buffer(0, Some(&self.st.hc_res), res_off);
        enc.set_buffer(1, Some(&self.st.hc_mixed), 0);
        enc.set_buffer(2, Some(&self.st.hc_inject), 0);
        enc.set_buffer(3, Some(&self.st.hc_graw), 0);
        enc.set_buffer(4, Some(&self.wt.w32[&p("ple_conv1d.weight")]), 0);
        enc.set_bytes(5, 4, &d as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &hc as *const u32 as *const c_void);
        enc.set_buffer(7, Some(&self.st.conv_state[l]), offset);
        enc.set_bytes(8, 4, &kern as *const u32 as *const c_void);
        enc.set_bytes(9, 4, &dilation as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((d + 63) / 64) as u64, hc as u64, 1),
            MTLSize::new(64, 1, 1),
        );
    }

    pub(crate) fn zero_buf(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        buf: &metal::Buffer,
        d: u32,
    ) {
        enc.set_compute_pipeline_state(&self.p["hc_zero"]);
        enc.set_buffer(0, Some(buf), 0);
        enc.set_bytes(1, 4, &d as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((d + 63) / 64) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }

    /// Gated delta-net token mixer over the HC-mixed view (`hc_mixed`), writing the
    /// block output to `h`. The input is already normalized by the hyper-connection
    /// mixer, so there is no leading RMSNorm.
    fn qwen4exp_gdn(&self, enc: &metal::ComputeCommandEncoderRef, l: usize, d: u32) {
        let p = |s: &str| format!("blk.{l}.{s}");
        let sc = self.arch.ssm.unwrap();
        let (s_st, hk, hv) = (sc.d_state, sc.n_group, sc.dt_rank);
        let d_inner = sc.d_inner;
        let conv_ch = d_inner + 2 * hk * s_st;
        let conv_k = sc.conv_kernel;
        let head_v = d_inner / hv;
        self.mm(
            enc,
            "plain",
            &p("attn_qkv.weight"),
            &self.st.hc_mixed,
            &self.st.ssm_qkv,
            d,
            conv_ch,
            None,
        );
        self.mm(
            enc,
            "plain",
            &p("attn_gate.weight"),
            &self.st.hc_mixed,
            &self.st.ssm_z,
            d,
            d_inner,
            None,
        );
        let alpha_name = p("ssm_alpha.weight");
        let beta_name = p("ssm_beta.weight");
        if self.cfg.flash_gdn_ab_fused
            && d % 4 == 0
            && self.wt.repr(&alpha_name) == Repr::F32
            && self.wt.repr(&beta_name) == Repr::F32
        {
            enc.set_compute_pipeline_state(&self.p["gdn_ab_fused"]);
            enc.set_buffer(0, Some(&self.st.hc_mixed), 0);
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
            self.mm(
                enc,
                "plain",
                &alpha_name,
                &self.st.hc_mixed,
                &self.st.ssm_gate,
                d,
                hv,
                None,
            );
            self.mm(
                enc,
                "plain",
                &beta_name,
                &self.st.hc_mixed,
                &self.st.ssm_beta,
                d,
                hv,
                None,
            );
            self.bar(enc);
            self.enc_reduce(
                enc,
                "ssm_ab",
                &[
                    (&self.st.ssm_gate, 0),
                    (&self.st.ssm_beta, 1),
                    (&self.wt.w32[&p("ssm_dt.bias")], 2),
                    (&self.wt.w32[&p("ssm_a")], 3),
                ],
                &[(4, hv), (5, hv)],
                &[],
                ((hv + 63) / 64) as u64,
                64,
            );
        }
        enc.set_compute_pipeline_state(&self.p["conv1d_decode"]);
        enc.set_buffer(0, Some(&self.st.ssm_qkv), 0);
        enc.set_buffer(1, Some(&self.st.conv_state[l]), 0);
        enc.set_buffer(2, Some(&self.wt.w32[&p("ssm_conv1d.weight")]), 0);
        enc.set_bytes(3, 4, &conv_ch as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &conv_k as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((conv_ch + 63) / 64) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        self.bar(enc);
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
        enc.set_buffer(11, Some(&self.st.ssm_state[l]), 0);
        enc.set_bytes(12, 4, &u32::MAX as *const u32 as *const c_void);
        // OJAS_KMAP_DIV=1 selects the grouped value->key head mapping.
        let kmap_div: u32 = self.cfg.moe_kmap_div as u32;
        enc.set_bytes(13, 4, &kmap_div as *const u32 as *const c_void);
        // FLA GDN uses rsqrt(sum(x*x) + eps), including for Flash.
        enc.set_bytes(14, 4, &0u32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((s_st / 4) as u64, hv as u64, 1),
            MTLSize::new(128, 1, 1),
        );
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p["gated_rmsnorm"]);
        enc.set_buffer(0, Some(&self.st.ssm_o), 0);
        enc.set_buffer(1, Some(&self.wt.w32[&p("ssm_norm.weight")]), 0);
        enc.set_buffer(2, Some(&self.st.ssm_z), 0);
        enc.set_bytes(3, 4, &head_v as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
        enc.set_bytes(5, 4, &d_inner as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &1u32 as *const u32 as *const c_void); // qwen4exp: sigmoid gate
        enc.dispatch_thread_groups(MTLSize::new(hv as u64, 1, 1), MTLSize::new(32, 1, 1));
        self.bar(enc);
        self.mm(
            enc,
            "plain",
            &p("ssm_out.weight"),
            &self.st.ssm_o,
            &self.st.h,
            d_inner,
            d,
            None,
        );
    }

    /// Sparse (indexer-selected) attention over the HC-mixed view, writing the block
    /// output to `h`. Within the indexer budget this is bit-identical to dense gated
    /// GQA, which is what runs here: q projects to per-head [q|gate], per-head
    /// QK-norm, partial NEOX RoPE (the M-RoPE sections this arch declares reduce to
    /// plain rope for text-only input — Ornith carries the same
    /// `rope.dimension_sections` and is correct on this path), GQA over the cache,
    /// attn *= sigmoid(gate), output projection. The learned indexer that prunes KV
    /// for long contexts is not yet applied, so long-context runs are dense;
    /// equivalence to the trained sparse selection is not yet validated against the
    /// reference.
    fn qwen4exp_qsa(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        l: usize,
        pos: usize,
        seq: u32,
        d: u32,
    ) {
        let p = |s: &str| format!("blk.{l}.{s}");
        let lp = self.arch.layers[l];
        let sc = self.arch.ssm.unwrap();
        let (hd, kvdim, qdim) = (lp.head_dim, lp.kvdim, lp.qdim);
        let group = lp.n_head / lp.n_kv.max(1);
        let inp = &self.st.hc_mixed;
        self.mm(
            enc,
            "plain",
            &p("attn_q.weight"),
            inp,
            &self.st.ssm_qkv,
            d,
            2 * qdim,
            None,
        );
        self.mm(
            enc,
            "plain",
            &p("attn_k.weight"),
            inp,
            &self.st.k,
            d,
            kvdim,
            None,
        );
        self.mm(
            enc,
            "plain",
            &p("attn_v.weight"),
            inp,
            &self.st.v,
            d,
            kvdim,
            None,
        );
        self.bar(enc);
        self.enc_reduce(
            enc,
            "qgate_split",
            &[(&self.st.ssm_qkv, 0), (&self.st.q, 1)],
            &[(2, hd), (3, qdim), (4, 1)],
            &[],
            ((qdim + 63) / 64) as u64,
            64,
        );
        let (nq, nk) = (lp.n_head, lp.n_kv);
        self.bar(enc);
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
        self.bar(enc);
        let (totq, totk) = (qdim / 2, kvdim / 2);
        let off = pos as u32 * kvdim;
        self.enc_reduce(
            enc,
            "rope_qk_store",
            &[
                (&self.st.q, 0),
                (&self.st.k, 1),
                (&self.st.v, 2),
                (&self.st.kcache[l], 3),
                (&self.st.vcache[l], 4),
            ],
            &[
                (5, hd),
                (6, pos as u32),
                (8, totq),
                (9, totk),
                (10, kvdim),
                (11, off),
                (12, 1),
                (13, sc.n_rot),
            ],
            &[(7, lp.rope_base)],
            (((totq + totk + kvdim) + 63) / 64) as u64,
            64,
        );
        self.bar(enc);
        if seq <= 512 {
            super::attn_log("qwen4exp", "attention_short", seq);
            self.enc_reduce(
                enc,
                "attention_short",
                &[
                    (&self.st.q, 0),
                    (&self.st.kcache[l], 1),
                    (&self.st.vcache[l], 2),
                    (&self.st.attn, 3),
                ],
                &[(4, hd), (5, kvdim), (6, seq), (7, group)],
                &[(8, lp.scale)],
                lp.n_head as u64,
                64,
            );
        } else {
            super::attn_log("qwen4exp", "attn_flash", seq);
            self.attn_flash(enc, lp.n_head, l, l, hd, kvdim, seq, group, lp.scale);
        }
        self.bar(enc);
        self.enc_reduce(
            enc,
            "gate_mul_sigmoid",
            &[(&self.st.attn, 0), (&self.st.ssm_qkv, 1)],
            &[(2, hd), (3, qdim), (4, 1)],
            &[],
            ((qdim + 63) / 64) as u64,
            64,
        );
        self.bar(enc);
        self.mm(
            enc,
            "plain",
            &p("attn_output.weight"),
            &self.st.attn,
            &self.st.h,
            qdim,
            d,
            None,
        );
    }

    /// Shared + routed mixture of experts over the HC-mixed view, accumulating the
    /// block output into `h` (pre-zeroed).
    ///
    /// Representation-agnostic on both halves. The shared expert runs as mm +
    /// silu_mul rather than the fused `ffn_gu_q4`, so it works whichever map the
    /// weights landed in, and the routed experts pick their kernel from
    /// `quant_src::MOE_FORMATS` by the tensor's own GGUF type. Indexing `wt.w4`/
    /// `scale4` and naming `moe_gu_q4`/`moe_down_q4` outright would pin the
    /// architecture to prec=2, which cannot run a model too large to requantize.
    fn qwen4exp_moe(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        l: usize,
        d: u32,
        phase: MoePhase,
    ) {
        let p = |s: &str| format!("blk.{l}.{s}");
        let mc = self.arch.moe.unwrap();
        let (ne, nu, fe, fs) = (mc.n_expert, mc.n_used, mc.ffn_exp, mc.ffn_shexp);
        self.check_moe(&p, d, fe);
        let inp = &self.st.hc_mixed;
        let off = (l as u64) * (MAXM as u64) * (nu as u64) * 4; // this layer's moe_idx slot

        if !matches!(phase, MoePhase::Experts) {
            // router logits (ffn_gate_inp is f32)
            self.enc_reduce(
                enc,
                "gemv_w32",
                &[
                    (inp, 0),
                    (&self.wt.w32[&p("ffn_gate_inp.weight")], 1),
                    (&self.ms.moe_lg, 2),
                ],
                &[(3, d), (4, ne)],
                &[],
                ((ne + 7) / 8) as u64,
                256,
            );
            // Shared-expert SwiGLU → tmp. mm + silu_mul, not the fused ffn_gu_*:
            // `mm` resolves the representation per tensor, which the fused kernel
            // cannot. Same shape graph_mla uses, and for the same reason.
            self.mm(
                enc,
                "plain",
                &p("ffn_gate_shexp.weight"),
                inp,
                &self.st.gate,
                d,
                fs,
                None,
            );
            self.mm(
                enc,
                "plain",
                &p("ffn_up_shexp.weight"),
                inp,
                &self.st.up,
                d,
                fs,
                None,
            );
            self.bar(enc);
            self.enc_reduce(
                enc,
                "silu_mul",
                &[(&self.st.gate, 0), (&self.st.up, 1), (&self.st.act, 2)],
                &[(3, fs)],
                &[],
                ((fs + 63) / 64) as u64,
                64,
            );
            self.bar(enc);
            self.mm(
                enc,
                "plain",
                &p("ffn_down_shexp.weight"),
                &self.st.act,
                &self.st.tmp,
                fs,
                d,
                None,
            );
            // shared-expert gate logit; the down kernel applies the sigmoid
            self.enc_reduce(
                enc,
                "gemv_w32",
                &[
                    (inp, 0),
                    (&self.wt.w32[&p("ffn_gate_inp_shexp.weight")], 1),
                    (&self.ms.moe_sh, 2),
                ],
                &[(3, d), (4, 1)],
                &[],
                1,
                32,
            );
            self.bar(enc);
            // router top-k → this layer's moe_idx slot + moe_wgt
            enc.set_compute_pipeline_state(&self.p["moe_topk"]);
            enc.set_buffer(0, Some(&self.ms.moe_lg), 0);
            enc.set_buffer(1, Some(&self.ms.moe_idx), off);
            enc.set_buffer(2, Some(&self.ms.moe_wgt), 0);
            enc.set_bytes(3, 4, &ne as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
            // Route ends here — the CPU now gathers exactly the experts moe_idx names.
            if matches!(phase, MoePhase::Route) {
                return;
            }
            self.bar(enc);
        }
        // `h` accumulates the routed-expert outputs, so clear it immediately before
        // that accumulation. (It still holds the token-mixer output until this point,
        // which the hc trace reads to price each mixer.)
        self.zero_buf(enc, &self.st.h, d);
        self.bar(enc);

        if self.is_resident(l) {
            // Resident: full expert tensors, indexed by the router's own moe_idx.
            // No gather, no slot map: `moe_idx` is the per-expert index the kernel
            // multiplies by the tensor stride. Kernel `entry` (not `_direct`) is the
            // native full-tensor walker the non-streamed path uses.
            let gname = p("ffn_gate_exps.weight");
            let uname = p("ffn_up_exps.weight");
            let dname = p("ffn_down_exps.weight");
            let woff = |n: &str| self.wt.w_off.get(n).copied().unwrap_or(0);
            let gty = self.wt.w_qtype.get(&gname).copied().unwrap_or(12);
            let dty = self.wt.w_qtype.get(&dname).copied().unwrap_or(8);
            let gk = ojas_core::quant_src::moe_kernel(gty, ojas_core::quant_src::MoeRole::GateUp)
                .unwrap_or_else(|| panic!("no MoE gate/up kernel for GGUF type {gty} ({gname})"));
            let dk = ojas_core::quant_src::moe_kernel(dty, ojas_core::quant_src::MoeRole::Down)
                .unwrap_or_else(|| panic!("no MoE down kernel for GGUF type {dty} ({dname})"));
            enc.set_compute_pipeline_state(
                &self.p[if gty == 21 && self.cfg.flash_iq3s_table {
                    "moe_gu_iq3s_table"
                } else {
                    gk.entry
                }],
            );
            enc.set_buffer(0, Some(inp), 0);
            enc.set_buffer(1, Some(&self.wt.wq[&gname]), woff(&gname));
            enc.set_buffer(2, Some(&self.wt.wq[&uname]), woff(&uname));
            enc.set_buffer(3, Some(&self.ms.moe_act), 0);
            enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &fe as *const u32 as *const c_void);
            enc.set_buffer(8, Some(&self.ms.moe_idx), off);
            let (gt, gr) = gk.launch;
            enc.dispatch_thread_groups(
                MTLSize::new(((fe + gr - 1) / gr) as u64, nu as u64, 1),
                MTLSize::new(gt as u64, 1, 1),
            );
            self.bar(enc);
            if dty == 20 {
                enc.set_compute_pipeline_state(
                    &self.p[if self.cfg.flash_iq4nl_wide {
                        "moe_down_iq4nl_parallel_w8"
                    } else {
                        "moe_down_iq4nl_parallel"
                    }],
                );
                enc.set_buffer(0, Some(&self.ms.moe_act), 0);
                enc.set_buffer(1, Some(&self.wt.wq[&dname]), woff(&dname));
                enc.set_buffer(2, Some(&self.st.skbuf), 0);
                enc.set_bytes(3, 4, &fe as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                enc.set_buffer(6, Some(&self.ms.moe_idx), off);
                enc.set_buffer(7, Some(&self.ms.moe_wgt), 0);
                enc.set_bytes(8, 4, &nu as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(((d + 3) / 4) as u64, nu as u64, 1),
                    MTLSize::new(128, 1, 1),
                );
                self.bar(enc);
                enc.set_compute_pipeline_state(&self.p["moe_down_iq4nl_finish"]);
                enc.set_buffer(0, Some(&self.st.skbuf), 0);
                enc.set_buffer(1, Some(&self.st.h), 0);
                enc.set_buffer(2, Some(&self.st.tmp), 0);
                enc.set_buffer(3, Some(&self.ms.moe_sh), 0);
                enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &nu as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(((d + 63) / 64) as u64, 1, 1),
                    MTLSize::new(64, 1, 1),
                );
            } else {
                enc.set_compute_pipeline_state(&self.p[dk.entry]);
                enc.set_buffer(0, Some(&self.ms.moe_act), 0);
                enc.set_buffer(1, Some(&self.wt.wq[&dname]), woff(&dname));
                enc.set_buffer(2, Some(&self.st.h), 0);
                enc.set_bytes(3, 4, &fe as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                enc.set_buffer(6, Some(&self.ms.moe_idx), off);
                enc.set_buffer(7, Some(&self.ms.moe_wgt), 0);
                enc.set_bytes(8, 4, &nu as *const u32 as *const c_void);
                enc.set_buffer(9, Some(&self.st.tmp), 0);
                enc.set_buffer(10, Some(&self.ms.moe_sh), 0);
                let (dt, dr) = dk.launch;
                enc.dispatch_thread_groups(
                    MTLSize::new(((d + dr - 1) / dr) as u64, 1, 1),
                    MTLSize::new(dt as u64, 1, 1),
                );
            }
        } else if self.strm.stream {
            // Streamed: the gather packed the routed experts into moe_gs/us/ds and
            // recorded where each pick landed in moe_slot, so the kernels walk that
            // scratch instead of all `ne` experts. The map is the identity whenever
            // the top-k is distinct and in range, the normal case; it is read rather
            // than assumed so that the gather can dedupe, as it must once M > 1.
            let gname = p("ffn_gate_exps.weight");
            let dname = p("ffn_down_exps.weight");
            // The loader validated both against this same table, so a miss is an
            // invariant break, not a case to paper over with another format's kernel.
            let gty = self.wt.w_qtype.get(&gname).copied().unwrap_or(12);
            let dty = self.wt.w_qtype.get(&dname).copied().unwrap_or(8);
            let gk = ojas_core::quant_src::moe_kernel(gty, ojas_core::quant_src::MoeRole::GateUp)
                .unwrap_or_else(|| panic!("no MoE gate/up kernel for GGUF type {gty} ({gname})"));
            let dk = ojas_core::quant_src::moe_kernel(dty, ojas_core::quant_src::MoeRole::Down)
                .unwrap_or_else(|| panic!("no MoE down kernel for GGUF type {dty} ({dname})"));
            let direct = self.strm.direct_layer.get() == Some(l);
            if direct {
                self.direct_expert_resources(enc);
            }
            enc.set_compute_pipeline_state(
                &self.p[if direct {
                    if gty == 21 {
                        "moe_gu_iq3s_direct"
                    } else {
                        "moe_gu_iq4xs_direct"
                    }
                } else {
                    gk.entry
                }],
            );
            enc.set_buffer(0, Some(inp), 0);
            enc.set_buffer(
                1,
                Some(if direct {
                    &self.strm.direct_tables[0]
                } else {
                    &self.strm.moe_gs
                }),
                0,
            );
            enc.set_buffer(
                2,
                Some(if direct {
                    &self.strm.direct_tables[1]
                } else {
                    &self.strm.moe_us
                }),
                0,
            );
            enc.set_buffer(3, Some(&self.ms.moe_act), 0);
            enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &fe as *const u32 as *const c_void);
            enc.set_buffer(8, Some(&self.strm.moe_slot), 0);
            let (gt, gr) = gk.launch;
            enc.dispatch_thread_groups(
                MTLSize::new(((fe + gr - 1) / gr) as u64, nu as u64, 1),
                MTLSize::new(gt as u64, 1, 1),
            );
            self.bar(enc);
            enc.set_compute_pipeline_state(
                &self.p[if direct {
                    "moe_down_iq4nl_direct"
                } else {
                    dk.entry
                }],
            );
            enc.set_buffer(0, Some(&self.ms.moe_act), 0);
            enc.set_buffer(
                1,
                Some(if direct {
                    &self.strm.direct_tables[2]
                } else {
                    &self.strm.moe_ds
                }),
                0,
            );
            enc.set_buffer(2, Some(&self.st.h), 0);
            enc.set_bytes(3, 4, &fe as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
            enc.set_buffer(6, Some(&self.strm.moe_slot), 0);
            enc.set_buffer(7, Some(&self.ms.moe_wgt), 0);
            enc.set_bytes(8, 4, &nu as *const u32 as *const c_void);
            enc.set_buffer(9, Some(&self.st.tmp), 0);
            enc.set_buffer(10, Some(&self.ms.moe_sh), 0);
            let (dt, dr) = dk.launch;
            enc.dispatch_thread_groups(
                MTLSize::new(((d + dr - 1) / dr) as u64, 1, 1),
                MTLSize::new(dt as u64, 1, 1),
            );
        } else {
            // Resident Q4L (prec=2): the tuned requantized path, indexed by moe_idx.
            enc.set_compute_pipeline_state(&self.p["moe_gu_q4"]);
            enc.set_buffer(0, Some(inp), 0);
            enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_gate_exps.weight")]), 0);
            enc.set_buffer(2, Some(&self.wt.w4[&p("ffn_up_exps.weight")]), 0);
            enc.set_buffer(3, Some(&self.ms.moe_act), 0);
            enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &fe as *const u32 as *const c_void);
            enc.set_buffer(6, Some(&self.wt.scale4[&p("ffn_gate_exps.weight")]), 0);
            enc.set_buffer(7, Some(&self.wt.scale4[&p("ffn_up_exps.weight")]), 0);
            enc.set_buffer(
                8,
                Some(&self.ms.moe_idx),
                (l as u64) * (MAXM as u64) * (nu as u64) * 4,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(((fe + 7) / 8) as u64, nu as u64, 1),
                MTLSize::new(64, 1, 1),
            );
            self.bar(enc);
            enc.set_compute_pipeline_state(&self.p["moe_down_q4"]);
            enc.set_buffer(0, Some(&self.ms.moe_act), 0);
            enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_down_exps.weight")]), 0);
            enc.set_buffer(2, Some(&self.st.h), 0);
            enc.set_bytes(3, 4, &fe as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
            enc.set_buffer(5, Some(&self.wt.scale4[&p("ffn_down_exps.weight")]), 0);
            enc.set_buffer(
                6,
                Some(&self.ms.moe_idx),
                (l as u64) * (MAXM as u64) * (nu as u64) * 4,
            );
            enc.set_buffer(7, Some(&self.ms.moe_wgt), 0);
            enc.set_bytes(8, 4, &nu as *const u32 as *const c_void);
            enc.set_buffer(9, Some(&self.st.tmp), 0);
            enc.set_buffer(10, Some(&self.ms.moe_sh), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(((d + 7) / 8) as u64, 1, 1),
                MTLSize::new(64, 1, 1),
            );
        }
    }

    /// NextN/MTP draft block for qwen4exp (blk.n_layers).
    ///
    /// Unlike qwen35's dense draft block this is a full qwen4exp layer — hyper
    /// connections, gated attention over its own KV cache, its own 512-expert MoE —
    /// preceded by the NextN combiner and followed by its own terminal hc mixer
    /// (`nextn.hc_head_*`) in place of the model's `output_hc`.
    ///
    /// The combiner preserves each hyper-connection stream through eh_proj, following
    /// the NextN graph rather than pooling and broadcasting hidden state. Output
    /// fidelity against the reference draft graph is not covered by a test here.
    ///
    /// `phase` splits it the same way the main layers split, so a streamed model can
    /// gather blk.n_layers' routed experts between the two halves.
    pub(crate) fn mtp_draft_encode_qwen4exp(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        token: u32,
        pos: usize,
        hrow: usize,
        head: bool,
        chain: bool,
        phase: MoePhase,
    ) {
        let mc = self
            .sp
            .mtp
            .expect("mtp_draft_encode_qwen4exp without a draft block");
        let l = mc.layer;
        let q = self.arch.qwen4exp.as_ref().unwrap();
        let (hc, hc_lr) = (q.hc_mult, q.hc_low_rank);
        let d = self.d as u32;
        let p = |s: &str| format!("blk.{l}.{s}");

        if !matches!(phase, MoePhase::Experts) {
            let (hsrc, hoff) = if chain {
                (&self.sp.mtp_chain, 0u64)
            } else {
                (&self.sp.mtp_h, (hrow as u64) * (d as u64) * (hc as u64) * 4)
            };
            self.qwen4exp_mtp_project(enc, &[token], hsrc, hoff);
            self.hc_mix(enc, &p("hc_attn"), d, hc, hc_lr, true);
            self.bar(enc);
            self.qwen4exp_qsa(enc, l, pos, (pos + 1) as u32, d);
            self.bar(enc);
            self.hc_combine(enc, &self.st.h, d, hc);
            self.bar(enc);
            self.hc_mix(enc, &p("hc_ffn"), d, hc, hc_lr, true);
            self.bar(enc);
        }
        self.qwen4exp_moe(enc, l, d, phase);
        if matches!(phase, MoePhase::Route) {
            return;
        }
        self.bar(enc);
        self.hc_combine(enc, &self.st.h, d, hc);
        self.bar(enc);
        // Skipped for KV-hole-filling drafts, whose prediction is discarded.
        if !head {
            return;
        }
        // Chaining reads this back as the next step's hidden, on the same "before the
        // final norm" rule the target's capture uses.
        if mc.hnorm_len == (d * hc) as usize {
            self.enc_reduce(
                enc,
                "copy_buf",
                &[(&self.sp.mtp_chain, 0), (&self.st.hc_res, 1)],
                &[(2, d * hc)],
                &[],
                ((d * hc + 63) / 64) as u64,
                64,
            );
            self.bar(enc);
        }
        self.hc_mix(enc, &p("nextn.hc_head"), d, hc, hc_lr, false);
        self.bar(enc);
        let hw = if mc.has_head {
            p("nextn.shared_head_head.weight")
        } else {
            self.arch.lm_head.clone()
        };
        self.mm(
            enc,
            "plain",
            &hw,
            &self.st.hc_mixed,
            &self.st.logits,
            d,
            self.arch.vocab as u32,
            None,
        );
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p["argmax"]);
        enc.set_buffer(0, Some(&self.st.logits), 0);
        enc.set_buffer(1, Some(&self.sp.mtp_tok), 0);
        enc.set_bytes(
            2,
            4,
            &(self.arch.vocab as u32) as *const u32 as *const c_void,
        );
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(self.tune.max_tg.min(1024), 1, 1),
        );
    }

    /// Run the qwen4exp draft block. Streamed models split it around the expert
    /// gather for blk.n_layers, exactly as the main stack's layers are split.
    pub(crate) fn mtp_draft_qwen4exp(
        &self,
        token: u32,
        pos: usize,
        hrow: usize,
        head: bool,
        chain: bool,
    ) -> u32 {
        let trace = self.flash_trace_start();
        let l = self
            .sp
            .mtp
            .expect("mtp_draft_qwen4exp without a draft block")
            .layer;
        let run = |phase: MoePhase| {
            objc::rc::autoreleasepool(|| {
                let cb = self.gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                self.mtp_draft_encode_qwen4exp(enc, token, pos, hrow, head, chain, phase);
                enc.end_encoding();
                let _ = ojas_metal::commit_and_wait_checked(cb, "qwen4exp scalar graph");
            });
        };
        if self.strm.stream && !self.is_resident(l) {
            run(MoePhase::Route);
            self.gather_experts_m(l, 1);
            run(MoePhase::Experts);
        } else {
            // Resident (and in-RAM) models keep every draft-block expert addressable,
            // so route and experts run in one command buffer with no host gather.
            run(MoePhase::Full);
        }
        self.flash_trace_finish(trace, "draft", pos, 1, FlashTargetTiming::default());
        unsafe { *(self.sp.mtp_tok.contents() as *const u32) }
    }
}
