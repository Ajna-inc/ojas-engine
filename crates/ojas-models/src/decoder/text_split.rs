//! The split-layout text encoder on Metal: a decoder-style block stack (separate Q/K/V
//! projections over grouped-query heads, RMSNorm, sandwich norms, per-layer inputs,
//! a learned scale per block) run bidirectionally over packed sequences.
//!
//! It shares the request machinery of the fused layout (`text_encoder.rs`): the
//! packed rows, the per-row key spans (each sequence alone, and on sliding layers
//! `|i - j| <= window` inside it), the position table restarting at 0 per sequence,
//! the attention kernels and the pooling tail. What differs is the block, written out
//! here rather than through `encoder.rs`'s LayerNorm block:
//!
//! ```text
//! x   = embed(tokens) * embed_scale
//! ple = rms(per_layer_model_proj(x) / sqrt(d))             one slice per layer
//! per layer l:
//!   q, k, v = attn_q(rms(x)), attn_k(..), attn_v(..)
//!   q, k = rope(rms_head(q) * q_norm, rms_head(k) * k_norm);  v = rms_head(v)
//!   x += post_attention_norm(attn_output(attention(q, k, v)))
//!   x += post_ffw_norm(ffn_down(act(ffn_gate(rms(x))) * ffn_up(rms(x))))
//!   x  = (x + post_norm(proj(act(inp_gate(x)) * ple[l]))) * layer_scale[l]
//! x = output_norm(x)
//! ```
//!
//! Each optional piece follows the spec ([`SplitStack`]): without sandwich norms the
//! projections accumulate straight into the residual, without per-layer inputs the
//! layer scale is a plain multiply. Attention is the span-tiled MMA kernel that reads
//! Q as half from device (`attn_mma_dq_span_src`, heads up to 512 wide), or the
//! per-query span kernel where that is unavailable. `tests/embedding_parity.rs`
//! compares the result with the reference model's sentence embeddings.

use super::*;
use super::encoder::Heads;
use super::text_encoder::TextBuffers;
use anyhow::Result;
use ojas_arch::text_encoder::SplitStack;
use objc::{msg_send, sel, sel_impl};

/// Working rows the split layout needs beyond the shared [`TextBuffers`] (which hold
/// the residual `x`, the norm output `h`, `q`, the half K/V and the MLP gate in `ffn`).
pub(crate) struct SplitRows {
    /// K and V projections, f32, `[rows, max n_kv*hd]`.
    k: metal::Buffer,
    v: metal::Buffer,
    /// Attention output `[rows, max n_head*hd]`; the per-layer input gate reuses it.
    ao: metal::Buffer,
    /// Q as half for the tiled kernel, with the 32 rows of slack it reads past the end.
    qh: metal::Buffer,
    /// MLP up projection `[rows, ffn]`.
    up: metal::Buffer,
    /// Every layer's per-layer input, side by side: `[rows, layers * ple_width]`.
    ple: metal::Buffer,
}

impl SplitRows {
    pub(crate) fn alloc(gpu: &MetalGpu, te: &TextEncoderConfig, sp: &SplitStack, cap: usize,
                        q_w: usize, kv_w: usize) -> SplitRows {
        let ple_w = sp.ple_width.map_or(0, |w| w as usize);
        SplitRows {
            k: buf(gpu, cap * kv_w), v: buf(gpu, cap * kv_w),
            ao: buf(gpu, cap * q_w.max(ple_w)),
            qh: buf_zeroed(gpu, ((cap + 32) * q_w).div_ceil(2)),
            up: buf(gpu, cap * te.ffn as usize),
            ple: buf(gpu, (cap * te.layers as usize * ple_w).max(1)),
        }
    }
}

impl<'a> DecoderGpu<'a> {
    /// The split stack over `seqs` into `b.x` (final-normed), then `tail` on the same
    /// encoder; one command buffer, waited on. Returns its GPU time in seconds.
    pub(crate) fn run_split(&self, te: &TextEncoderConfig, sp: &SplitStack, b: &TextBuffers, seqs: &[Vec<u32>],
                            tail: impl FnOnce(&metal::ComputeCommandEncoderRef, usize) -> Result<()>) -> Result<f64> {
        let r = b.split.as_ref().ok_or_else(|| anyhow::anyhow!("text buffers were sized for the fused layout"))?;
        let m: usize = seqs.iter().map(Vec::len).sum();
        let (d, m32, eps) = (te.d, m as u32, te.eps);
        let act = sp.act.kernel_code();
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();

        let scale = |x: &metal::Buffer, n: u32, s: f32| {
            self.enc_reduce(enc, "mul_scalar", &[(x, 0)], &[(1, n)], &[(2, s)], n.div_ceil(256) as u64, 256);
        };
        let rms = |src: &metal::Buffer, dst: &metal::Buffer, w: &str, width: u32, rows: u32| {
            self.enc_reduce(enc, "rmsnorm_m", &[(src, 0), (&self.wt.w32[w], 1), (dst, 2)],
                &[(3, width)], &[(4, eps)], rows as u64, 256);
        };
        // x = (x + rms(t) * w) * oscale
        let rms_add = |t: &metal::Buffer, w: &str, oscale: f32| {
            self.enc_reduce(enc, "enc_rms_add_m", &[(&b.x, 0), (t, 1), (&self.wt.w32[w], 2)],
                &[(3, d)], &[(4, eps), (5, oscale)], m32 as u64, 256);
        };

        self.encode_text_embed(enc, b, d, m32)?;
        if sp.embed_scale != 1.0 { scale(&b.x, m32 * d, sp.embed_scale); }
        if let Some(pw) = sp.ple_width {
            let n = te.layers * pw;
            self.projm(enc, &b.x, 0, "per_layer_model_proj.weight", &r.ple, d, n, m32, false);
            scale(&r.ple, m32 * n, 1.0 / (d as f32).sqrt());
            rms(&r.ple, &r.ple, "per_layer_proj_norm.weight", pw, m32 * te.layers);
        }

        for (l, ly) in sp.layers.iter().enumerate() {
            let p = |t: &str| format!("blk.{l}.{t}.weight");
            let (nh, nkv, hd) = (te.n_head, ly.n_kv, ly.hd);
            let (qd, kvd) = (nh * hd, nkv * hd);
            let layer_scale = if sp.layer_scale { self.arch.out_scale.get(l).copied().unwrap_or(1.0) } else { 1.0 };

            // Attention.
            rms(&b.x, &b.h, &p("attn_norm"), d, m32);
            self.projm(enc, &b.h, 0, &p("attn_q"), &b.q, d, qd, m32, false);
            self.projm(enc, &b.h, 0, &p("attn_k"), &r.k, d, kvd, m32, false);
            self.projm(enc, &b.h, 0, &p("attn_v"), &r.v, d, kvd, m32, false);
            // Without Q/K norm weights the kernel reads none; any buffer fills the slots.
            let (qn, kn) = if sp.qk_norm { (p("attn_q_norm"), p("attn_k_norm")) } else { (p("attn_norm"), p("attn_norm")) };
            let (span, tiles) = b.span_and_tiles(ly.local);
            let tiled = tiles.filter(|_| self.p.contains_key(&ojas_metal::kernels::attn::attn_mma_dq_span_name(hd)));
            let flags = sp.qk_norm as u32 | (sp.v_norm as u32) << 1 | (tiled.is_some() as u32) << 2;
            self.enc_reduce(enc, "enc_head_prep",
                &[(&b.q, 0), (&r.k, 1), (&r.v, 2), (&b.kh, 3), (&b.vh, 4),
                  (&self.wt.w32[&qn], 5), (&self.wt.w32[&kn], 6), (&b.pos, 7), (&r.qh, 14)],
                &[(8, hd), (9, nh), (10, nkv), (13, flags)], &[(11, eps), (12, ly.rope_base)],
                m32 as u64 * (nh + 2 * nkv) as u64, 32);
            let heads = Heads { n_head: nh as usize, n_kv: nkv as usize, hd: hd as usize,
                                scale: if sp.unit_scale { 1.0 } else { 1.0 / (hd as f32).sqrt() } };
            match tiled {
                Some(tiles) => self.encode_attention_dq_span(enc, heads, &r.qh, &b.kh, &b.vh, &r.ao, span, tiles, b.n_tiles, m),
                None => self.encode_attention_heads(enc, heads, &b.q, &b.kh, &b.vh, &r.ao, b.keys_tiled(ly.local, false), m),
            }
            if sp.sandwich {
                self.projm(enc, &r.ao, 0, &p("attn_output"), &b.h, qd, d, m32, false);
                rms_add(&b.h, &p("post_attention_norm"), 1.0);
            } else {
                self.projm(enc, &r.ao, 0, &p("attn_output"), &b.x, qd, d, m32, true);
            }

            // MLP.
            rms(&b.x, &b.h, &p("ffn_norm"), d, m32);
            self.projm(enc, &b.h, 0, &p("ffn_gate"), &b.ffn, d, te.ffn, m32, false);
            self.projm(enc, &b.h, 0, &p("ffn_up"), &r.up, d, te.ffn, m32, false);
            let n = te.ffn * m32;
            self.enc_reduce(enc, "ffn_gu_split", &[(&b.ffn, 0), (&r.up, 1), (&b.ffn, 2)],
                &[(3, n), (4, act)], &[], n.div_ceil(256) as u64, 256);
            let last_scale = if sp.ple_width.is_some() { 1.0 } else { layer_scale };
            if sp.sandwich {
                self.projm(enc, &b.ffn, 0, &p("ffn_down"), &b.h, te.ffn, d, m32, false);
                rms_add(&b.h, &p("post_ffw_norm"), last_scale);
            } else {
                self.projm(enc, &b.ffn, 0, &p("ffn_down"), &b.x, te.ffn, d, m32, true);
                if last_scale != 1.0 { scale(&b.x, m32 * d, last_scale); }
            }

            // Per-layer input, then the layer scale.
            if let Some(pw) = sp.ple_width {
                self.projm(enc, &b.x, 0, &p("inp_gate"), &r.ao, d, pw, m32, false);
                let n = pw * m32;
                self.enc_reduce(enc, "enc_ple_gate_m", &[(&r.ao, 0), (&r.ple, 1)],
                    &[(2, pw), (3, n), (4, te.layers * pw), (5, l as u32 * pw), (6, act)], &[],
                    n.div_ceil(256) as u64, 256);
                self.projm(enc, &r.ao, 0, &p("proj"), &b.h, pw, d, m32, false);
                rms_add(&b.h, &p("post_norm"), layer_scale);
            }
        }
        rms(&b.x, &b.x, "output_norm.weight", d, m32);
        tail(enc, m)?;
        enc.end_encoding();
        ojas_metal::commit_and_wait_checked(cb, "text encode").map_err(|e| anyhow::anyhow!("text encode: {e}"))?;
        let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        Ok((ge - gs).max(0.0))
    }
}
