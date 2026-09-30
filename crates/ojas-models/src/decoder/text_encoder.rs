//! The ModernBERT text encoder and the Laya decision head on Metal.
//!
//! A request is a batch of independent token sequences packed into one row buffer
//! with no padding between them. Every block is the shared encoder block
//! (`encoder.rs`); what is specific to this model is expressed as data:
//!
//! * per-row key spans restrict attention to the row's own sequence, and on the
//!   sliding-window layers to `|i - j| <= window` inside it;
//! * the RoPE position table restarts at 0 for every sequence;
//! * layer `l` uses the local RoPE base and window when `l % swa_pattern != 0`;
//! * layer 0 has no attention norm (ModernBERT feeds the embedding LayerNorm's
//!   output straight to its QKV projection);
//! * the MLP is gated over one fused `ffn_up` (first half activated), erf GELU.
//!
//! When the file carries a Laya head, [`DecoderGpu::laya_forward`] continues on the
//! same rows: the question-type embedding is added per sequence and the head's
//! transformer blocks run with sequence-wide spans. Only each sequence's first row and
//! its option-marker rows are read afterwards, so the last head block finishes, and
//! the scorer's LayerNorm, first linear and GELU run, on those rows alone. The
//! scorer's final `d -> 1` projection and the act head are small enough to finish on
//! the host (`crate::laya`).

use super::encoder::{Act, Block, Geom, Keys, Mlp, Norm, Rope, Scratch};
use super::*;
use anyhow::{ensure, Result};
use objc::{msg_send, sel, sel_impl};

/// Per-sequence results of [`DecoderGpu::laya_forward`].
pub struct LayaGpuOut {
    /// For each sequence, for each marker position asked for: the scorer's hidden
    /// vector after its GELU, `d` floats.
    pub scorer_hidden: Vec<Vec<Vec<f32>>>,
    /// For each sequence, the head's output at the sequence's first position
    /// (`[CLS]`), `d` floats: the act head's pooled input.
    pub pooled: Vec<Vec<f32>>,
    /// GPU execution time, from the command buffer's timestamps.
    pub gpu_s: f64,
}

/// Row buffers for packed requests, kept on the decoder and reused. Allocating them per
/// request (up to 15 MB each, two of them zero-filled) put about 9 ms of host time on
/// a seven-question request. They grow when a request needs more rows.
pub(crate) struct TextBuffers {
    /// Rows every buffer holds (a multiple of 32).
    cap: usize,
    x: metal::Buffer,
    h: metal::Buffer,
    qkv: metal::Buffer,
    q: metal::Buffer,
    kh: metal::Buffer,
    vh: metal::Buffer,
    ffn: metal::Buffer,
    ffn_wide: metal::Buffer,
    pos: metal::Buffer,
    span_global: metal::Buffer,
    span_local: metal::Buffer,
    /// MMA query-tile descriptors for the global and local spans and how many the
    /// current request uses; `None` where the tiled kernel is not available.
    tiles_global: Option<metal::Buffer>,
    tiles_local: Option<metal::Buffer>,
    n_tiles: u32,
    tokens: metal::Buffer,
    /// Rows the Laya head's last block keeps (`int`), in compact order.
    keep: metal::Buffer,
}

impl TextBuffers {
    /// The key spans for a global (`local == false`) or sliding-window layer.
    fn keys(&self, local: bool) -> Keys<'_> {
        let (span, tiles) = if local { (&self.span_local, &self.tiles_local) } else { (&self.span_global, &self.tiles_global) };
        Keys::Spans { span, tiles: tiles.as_ref().map(|b| (b, self.n_tiles)) }
    }

    fn scratch(&self) -> Scratch<'_> {
        Scratch {
            x: &self.x, h: &self.h, qkv: &self.qkv, q: &self.q,
            kh: &self.kh, vh: &self.vh, ffn: &self.ffn, ffn_wide: &self.ffn_wide, pos: &self.pos,
        }
    }
}

/// Copy `v` to the start of a shared buffer.
///
/// SAFETY: the buffer is `StorageModeShared`, holds at least `v.len()` u32s, and no
/// command buffer using it is in flight (every text pass waits for completion).
fn write_u32(b: &metal::Buffer, v: &[u32]) {
    debug_assert!(b.length() as usize >= v.len() * 4);
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), b.contents() as *mut u32, v.len()) };
}

/// Host view of row `r` of a `[rows, d]` shared buffer.
///
/// SAFETY: the buffer is `StorageModeShared` and the command buffer that wrote it
/// has completed.
fn row(b: &metal::Buffer, r: usize, d: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts((b.contents() as *const f32).add(r * d), d) }.to_vec()
}

impl<'a> DecoderGpu<'a> {
    /// True when this file is a text encoder rather than a decoder.
    pub fn has_text_encoder(&self) -> bool { self.arch.text_encoder.is_some() }

    /// True when the text encoder carries a Laya decision head.
    pub fn has_laya_head(&self) -> bool {
        self.arch.text_encoder.as_ref().is_some_and(|t| t.laya.is_some())
    }

    /// Hidden width of the text encoder.
    pub fn text_encoder_width(&self) -> Option<usize> {
        self.arch.text_encoder.as_ref().map(|t| t.d as usize)
    }

    /// Longest sequence the text encoder's position encoding was trained for.
    pub fn text_encoder_max_positions(&self) -> Option<usize> {
        self.arch.text_encoder.as_ref().map(|t| t.max_positions as usize)
    }

    /// Final hidden states of the text encoder for each sequence, `[len * d]`
    /// row-major per sequence (after the final LayerNorm, before any head).
    pub fn encode_text(&self, seqs: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let te = self.text_config()?;
        let d = te.d as usize;
        let b = self.text_buffers(te, seqs)?;
        let gpu_s = self.run_text(te, &b, seqs, |_, _| Ok(()))?;
        self.gpu_s.set(self.gpu_s.get() + gpu_s);
        let mut out = Vec::with_capacity(seqs.len());
        let mut start = 0usize;
        for s in seqs {
            let mut v = Vec::with_capacity(s.len() * d);
            for r in start..start + s.len() { v.extend_from_slice(&row(&b.x, r, d)); }
            out.push(v);
            start += s.len();
        }
        Ok(out)
    }

    /// The encoder, then the Laya head, over `seqs`. `qtypes[i]` selects the row of
    /// `laya.type_emb` added to sequence `i`; `markers[i]` lists the positions in
    /// sequence `i` whose scorer hidden vector is returned.
    pub fn laya_forward(&self, seqs: &[Vec<u32>], qtypes: &[u32], markers: &[Vec<usize>]) -> Result<LayaGpuOut> {
        let te = self.text_config()?;
        let head = te.laya.as_ref().ok_or_else(|| anyhow::anyhow!("this text encoder has no Laya head"))?;
        ensure!(qtypes.len() == seqs.len() && markers.len() == seqs.len(),
            "laya_forward: {} sequences, {} question types, {} marker lists", seqs.len(), qtypes.len(), markers.len());
        let n_types = self.wt.wshape.get("laya.type_emb.weight").map(|&(_, n)| n).unwrap_or(0);
        for (i, (&q, mk)) in qtypes.iter().zip(markers).enumerate() {
            ensure!(q < n_types, "sequence {i}: question type {q} out of range ({n_types} types)");
            ensure!(mk.iter().all(|&p| p < seqs[i].len()), "sequence {i}: a marker lies past its end");
        }
        let d = te.d as usize;
        let b = self.text_buffers(te, seqs)?;
        // The head's reads, in compact order: each sequence's first row ([CLS], the
        // act head's input), then its option markers.
        let mut keep: Vec<u32> = Vec::new();
        let mut start = 0usize;
        for (s, mk) in seqs.iter().zip(markers) {
            keep.push(start as u32);
            keep.extend(mk.iter().map(|&p| (start + p) as u32));
            start += s.len();
        }
        write_u32(&b.keep, &keep);
        let gpu_s = self.run_text(te, &b, seqs, |enc, m| {
            self.encode_laya_head(enc, te, head, &b, seqs, qtypes, keep.len(), m);
            Ok(())
        })?;
        self.gpu_s.set(self.gpu_s.get() + gpu_s);
        let (xc, scorer) = (&b.qkv, &b.ffn);
        let mut scorer_hidden = Vec::with_capacity(seqs.len());
        let mut pooled = Vec::with_capacity(seqs.len());
        let mut r = 0usize;
        for mk in markers {
            pooled.push(row(xc, r, d));
            scorer_hidden.push((1..=mk.len()).map(|i| row(scorer, r + i, d)).collect());
            r += 1 + mk.len();
        }
        Ok(LayaGpuOut { scorer_hidden, pooled, gpu_s })
    }

    /// Per-category GPU time of one text-encoder pass over `seqs`, in the manner of
    /// `profile_batch`: each category is dispatched once per layer, cycling the real
    /// layer weights, in its own command buffer; after two warm-up runs the best of
    /// five is kept, so a busy GPU inflates the numbers less than a whole-pass timing.
    /// Returns `(category, milliseconds)` for the encoder's layers and their sum, the
    /// pass's compute floor, then the Laya head's cost measured against a whole pass.
    pub fn profile_text(&self, seqs: &[Vec<u32>]) -> Result<Vec<(String, f64)>> {
        let te = self.text_config()?;
        let b = self.text_buffers(te, seqs)?;
        let m: usize = seqs.iter().map(Vec::len).sum();
        let (d, ffn, m32, nl) = (te.d, te.ffn, m as u32, te.layers as usize);
        let geom = Geom { d: te.d as usize, n_head: te.n_head as usize, hd: te.hd as usize,
                          ffn: te.ffn as usize, eps: te.eps };
        // Fill the working buffers once, so every category reads finite data.
        self.run_text(te, &b, seqs, |_, _| Ok(()))?;
        let time = |f: &dyn Fn(&metal::ComputeCommandEncoderRef, usize)| -> Result<f64> {
            let mut best = f64::INFINITY;
            for rep in 0..7 {
                let cb = self.gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                for l in 0..nl { f(enc, l); }
                enc.end_encoding();
                ojas_metal::commit_and_wait_checked(cb, "text profile").map_err(|e| anyhow::anyhow!("{e}"))?;
                let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
                if rep >= 2 { best = best.min((ge - gs) * 1e3); }
            }
            Ok(best)
        };
        let s = b.scratch();
        // Every norm in the stack costs the same; one weight stands in for all of them.
        let norm = Norm { weight: "output_norm.weight".into(), bias: None };
        let p = |l: usize, t: &str| format!("blk.{l}.{t}");
        let block = |l: usize| Block {
            attn_norm: None, qkv: p(l, "attn_qkv.weight"), qkv_bias: None, out: p(l, "attn_output.weight"),
            out_bias: None, ffn_norm: norm.clone(), up: p(l, "ffn_up.weight"), up_bias: None,
            down: p(l, "ffn_down.weight"), down_bias: None, mlp: Mlp::Gated, act: Act::GeluErf,
            rope: Some(Rope { base: te.rope_base, freq_dims: te.hd }),
        };
        let mut out = vec![
            ("layernorm x2".to_string(), time(&|enc, _| {
                self.enc_layernorm(enc, &b.x, &b.h, &norm, d, te.eps, m32);
                self.enc_layernorm(enc, &b.x, &b.h, &norm, d, te.eps, m32);
            })?),
            ("gemm qkv".into(), time(&|enc, l| self.projm(enc, &b.h, 0, &p(l, "attn_qkv.weight"), &b.qkv, d, 3 * d, m32, false))?),
            ("qkv prep".into(), time(&|enc, l| self.encode_qkv_prep(enc, &block(l), &geom, &s, m))?),
            ("attention".into(), time(&|enc, l| self.encode_attention(enc, &geom, &s, b.keys(te.is_local(l)), m))?),
            ("gemm out".into(), time(&|enc, l| self.projm(enc, &b.h, 0, &p(l, "attn_output.weight"), &b.x, d, d, m32, true))?),
            ("gemm up".into(), time(&|enc, l| self.projm(enc, &b.h, 0, &p(l, "ffn_up.weight"), &b.ffn_wide, d, 2 * ffn, m32, false))?),
            ("glu".into(), time(&|enc, _| {
                let n = ffn * m32;
                self.enc_reduce(enc, "ffn_gu_rows", &[(&b.ffn_wide, 0), (&b.ffn, 1)],
                    &[(2, ffn), (3, n), (4, Act::GeluErf.code())], &[], n.div_ceil(256) as u64, 256);
            })?),
            ("gemm down".into(), time(&|enc, l| self.projm(enc, &b.ffn, 0, &p(l, "ffn_down.weight"), &b.x, ffn, d, m32, true))?),
        ];
        let total: f64 = out.iter().map(|(_, t)| t).sum();
        out.push(("sum (encoder)".into(), total));
        // The Laya head, once (not per layer): its blocks and the scorer, timed as a
        // whole against the encoder pass it follows.
        if let Some(head) = &te.laya {
            let qtypes = vec![0u32; seqs.len()];
            let firsts: Vec<u32> = seqs.iter().scan(0u32, |at, s| { let r = *at; *at += s.len() as u32; Some(r) }).collect();
            write_u32(&b.keep, &firsts);
            let enc_only = self.run_text(te, &b, seqs, |_, _| Ok(()))?;
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let with_head = self.run_text(te, &b, seqs, |enc, m| {
                    self.encode_laya_head(enc, te, head, &b, seqs, &qtypes, firsts.len(), m);
                    Ok(())
                })?;
                best = best.min((with_head - enc_only) * 1e3);
            }
            out.push(("laya head (approx.)".into(), best.max(0.0)));
        }
        Ok(out)
    }

    fn text_config(&self) -> Result<&TextEncoderConfig> {
        self.arch.text_encoder.as_ref().ok_or_else(|| anyhow::anyhow!("this model is not a text encoder"))
    }

    /// The arena's row buffers, grown if needed, with this request's tokens, positions,
    /// key spans and tile descriptors written into them.
    fn text_buffers(&self, te: &TextEncoderConfig, seqs: &[Vec<u32>]) -> Result<std::cell::RefMut<'_, TextBuffers>> {
        ensure!(!seqs.is_empty() && seqs.iter().all(|s| !s.is_empty()), "every sequence needs at least one token");
        let vocab = self.wt.wshape.get("token_embd.weight").map(|&(_, n)| n).unwrap_or(0);
        for (i, s) in seqs.iter().enumerate() {
            ensure!(s.len() <= te.max_positions as usize,
                "sequence {i} has {} tokens, past the encoder's {} positions", s.len(), te.max_positions);
            ensure!(s.iter().all(|&t| t < vocab), "sequence {i} holds a token id outside the {vocab}-token vocabulary");
        }
        let m: usize = seqs.iter().map(Vec::len).sum();
        let mut arena = self.text_arena.borrow_mut();
        if arena.as_ref().is_none_or(|a| a.cap < m) {
            // Grow geometrically, so a stream of slightly larger requests reallocates
            // a handful of times rather than on every call.
            let cap = m.max(arena.as_ref().map_or(0, |a| a.cap * 2)).div_ceil(32) * 32;
            *arena = None;
            *arena = Some(self.alloc_text_buffers(te, cap));
        }
        let b = std::cell::RefMut::map(arena, |a| a.as_mut().expect("allocated above"));

        let (mut pos, mut global, mut local, mut tokens) =
            (vec![0u32; 4 + 4 * m], Vec::with_capacity(2 * m), Vec::with_capacity(2 * m), Vec::with_capacity(m));
        let w = te.window as usize;
        let mut start = 0usize;
        for s in seqs {
            let end = start + s.len();
            for (j, &t) in s.iter().enumerate() {
                let r = start + j;
                pos[4 + 4 * r] = j as u32;
                global.extend_from_slice(&[start as u32, end as u32]);
                local.extend_from_slice(&[r.saturating_sub(w).max(start) as u32, (r + w + 1).min(end) as u32]);
                tokens.push(t);
            }
            start = end;
        }
        write_u32(&b.pos, &pos);
        write_u32(&b.span_global, &global);
        write_u32(&b.span_local, &local);
        write_u32(&b.tokens, &tokens);
        // One MMA tile per 32 query rows of a sequence; its key range is the union of
        // its rows' spans, which are contiguous and monotone within a sequence.
        let tiles = |span: &[u32]| -> Vec<u32> {
            let mut t = Vec::new();
            let mut start = 0usize;
            for s in seqs {
                for q0 in (start..start + s.len()).step_by(32) {
                    let nq = 32.min(start + s.len() - q0);
                    t.extend_from_slice(&[q0 as u32, nq as u32, span[2 * q0], span[2 * (q0 + nq - 1) + 1]]);
                }
                start += s.len();
            }
            t
        };
        let mut b = b;
        if let (Some(tg), Some(tl)) = (&b.tiles_global, &b.tiles_local) {
            let (g, l) = (tiles(&global), tiles(&local));
            write_u32(tg, &g);
            write_u32(tl, &l);
            b.n_tiles = (g.len() / 4) as u32;
        }
        Ok(b)
    }

    fn alloc_text_buffers(&self, te: &TextEncoderConfig, cap: usize) -> TextBuffers {
        let d = te.d as usize;
        let ffn = te.laya.as_ref().map_or(te.ffn, |h| h.ffn.max(te.ffn)).max(te.d) as usize;
        let mma = self.p.contains_key(&ojas_metal::kernels::attn::attn_mma_span_name(te.hd));
        let gpu = self.gpu;
        // Tiles: at most one per 32 rows of each sequence, so cap/32 + one per sequence.
        let max_tiles = cap / 32 + cap;
        TextBuffers {
            cap,
            x: buf(gpu, cap * d), h: buf(gpu, cap * d), qkv: buf(gpu, cap * 3 * d),
            q: buf(gpu, cap * d),
            // Zeroed with 64 spare rows: the tiled attention reads 8-row K/V blocks past
            // a tile's key range and masks them, which needs finite values there. Rows
            // a later, shorter request leaves behind hold finite K/V, which is as good.
            kh: buf_zeroed(gpu, ((cap + 64) * d).div_ceil(2)), vh: buf_zeroed(gpu, ((cap + 64) * d).div_ceil(2)),
            ffn: buf(gpu, cap * ffn), ffn_wide: buf(gpu, cap * 2 * te.ffn as usize),
            pos: buf(gpu, 4 + 4 * cap), span_global: buf(gpu, 2 * cap), span_local: buf(gpu, 2 * cap),
            tokens: buf(gpu, cap), keep: buf(gpu, cap),
            tiles_global: mma.then(|| buf(gpu, 4 * max_tiles)), tiles_local: mma.then(|| buf(gpu, 4 * max_tiles)),
            n_tiles: 0,
        }
    }

    /// Embed, run every encoder block and the final norm, then `tail` on the same
    /// encoder; one command buffer, waited on. Returns its GPU time in seconds.
    fn run_text(&self, te: &TextEncoderConfig, b: &TextBuffers, seqs: &[Vec<u32>],
                tail: impl FnOnce(&metal::ComputeCommandEncoderRef, usize) -> Result<()>) -> Result<f64> {
        let m: usize = seqs.iter().map(Vec::len).sum();
        let (d, m32) = (te.d, m as u32);
        let geom = Geom { d: te.d as usize, n_head: te.n_head as usize, hd: te.hd as usize,
                          ffn: te.ffn as usize, eps: te.eps };
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();

        let emb = "token_embd.weight";
        match self.wt.repr(emb) {
            Repr::F16 => self.enc_reduce(enc, "embed_m_f16",
                &[(&self.wt.w16[emb], 0), (&b.x, 1), (&b.tokens, 3)],
                &[(2, d), (5, m32)], &[], (m32 * d).div_ceil(64) as u64, 64),
            Repr::Q8 => self.enc_reduce(enc, "embed_m_q8",
                &[(&self.wt.w8[emb], 0), (&b.x, 1), (&b.tokens, 3), (&self.wt.scale8[emb], 4)],
                &[(2, d), (5, m32)], &[], (m32 * d).div_ceil(64) as u64, 64),
            r => anyhow::bail!("{emb} resolved to {r:?}; the text encoder embeds from f16 or q8 only"),
        }
        let no_bias = |w: String| Norm { weight: w, bias: None };
        self.enc_layernorm(enc, &b.x, &b.x, &no_bias("token_embd_norm.weight".into()), d, te.eps, m32);

        let s = b.scratch();
        for l in 0..te.layers as usize {
            let p = |t: &str| format!("blk.{l}.{t}");
            let local = te.is_local(l);
            let block = Block {
                attn_norm: self.wt.w32.contains_key(&p("attn_norm.weight")).then(|| no_bias(p("attn_norm.weight"))),
                qkv: p("attn_qkv.weight"), qkv_bias: None,
                out: p("attn_output.weight"), out_bias: None,
                ffn_norm: no_bias(p("ffn_norm.weight")),
                up: p("ffn_up.weight"), up_bias: None,
                down: p("ffn_down.weight"), down_bias: None,
                mlp: Mlp::Gated,
                act: Act::GeluErf,
                rope: Some(Rope { base: if local { te.rope_base_local } else { te.rope_base }, freq_dims: te.hd }),
            };
            self.encode_block(enc, &block, &geom, &s, b.keys(local), m);
        }
        self.enc_layernorm(enc, &b.x, &b.x, &no_bias("output_norm.weight".into()), d, te.eps, m32);
        tail(enc, m)?;
        enc.end_encoding();
        ojas_metal::commit_and_wait_checked(cb, "text encode").map_err(|e| anyhow::anyhow!("text encode: {e}"))?;
        let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        Ok((ge - gs).max(0.0))
    }

    /// The Laya head on the encoder output in `b.x`: the question-type embedding, the
    /// head's blocks, then the scorer's LayerNorm, first linear and GELU.
    ///
    /// Only `n_keep` rows are read afterwards (`b.keep`: each sequence's first row and
    /// its option markers), so the last block runs its attention over every row (all
    /// rows are keys) and everything after it on the kept rows alone: they are
    /// gathered into `b.qkv` (residual) and `b.q` (attention output), both free by
    /// then, and the scorer's hidden rows land in `b.ffn`, in `b.keep` order.
    #[allow(clippy::too_many_arguments)]
    fn encode_laya_head(&self, enc: &metal::ComputeCommandEncoderRef, te: &TextEncoderConfig,
                        head: &LayaHeadConfig, b: &TextBuffers, seqs: &[Vec<u32>], qtypes: &[u32],
                        n_keep: usize, m: usize) {
        let d = te.d;
        let row_bytes = te.d as u64 * 4;
        // `add_rowbias_m` over one sequence's rows, with the type embedding's row as
        // the bias: `x[r] += type_emb[qtype]`.
        let mut start = 0u64;
        for (s, &q) in seqs.iter().zip(qtypes) {
            let total = s.len() as u32 * d;
            self.enc_reduce_off(enc, "add_rowbias_m",
                &[(&b.x, 0, start * row_bytes), (&self.wt.w32["laya.type_emb.weight"], 1, q as u64 * row_bytes)],
                &[(2, d), (3, total)], &[], total.div_ceil(64) as u64, 64);
            start += s.len() as u64;
        }
        let geom = Geom { d: te.d as usize, n_head: head.n_head as usize, hd: te.hd as usize,
                          ffn: head.ffn as usize, eps: head.eps };
        let s = b.scratch();
        let block = |i: u32| {
            let p = |t: &str| format!("laya.blk.{i}.{t}");
            let norm = |n: &str| Norm { weight: p(&format!("{n}.weight")), bias: Some(p(&format!("{n}.bias"))) };
            Block {
                attn_norm: Some(norm("attn_norm")),
                qkv: p("attn_qkv.weight"), qkv_bias: Some(p("attn_qkv.bias")),
                out: p("attn_output.weight"), out_bias: Some(p("attn_output.bias")),
                ffn_norm: norm("ffn_norm"),
                up: p("ffn_up.weight"), up_bias: Some(p("ffn_up.bias")),
                down: p("ffn_down.weight"), down_bias: Some(p("ffn_down.bias")),
                mlp: Mlp::Plain,
                act: Act::Relu,
                rope: None,
            }
        };
        let last = head.blocks - 1;
        for i in 0..last { self.encode_block(enc, &block(i), &geom, &s, b.keys(false), m); }
        self.encode_attention_half(enc, &block(last), &geom, &s, b.keys(false), m);
        let (xc, hc, k32) = (&b.qkv, &b.q, n_keep as u32);
        for (src, dst) in [(&b.x, xc), (&b.h, hc)] {
            self.enc_reduce(enc, "ple_gather", &[(src, 0), (&b.keep, 1), (dst, 2)],
                &[(3, d), (4, k32)], &[], (d * k32).div_ceil(256) as u64, 256);
        }
        self.encode_mlp_half(enc, &block(last), &geom, xc, hc, &b.ffn, &b.ffn_wide, n_keep);

        let scorer_norm = Norm { weight: "laya.scorer_norm.weight".into(), bias: Some("laya.scorer_norm.bias".into()) };
        self.enc_layernorm(enc, xc, hc, &scorer_norm, d, head.eps, k32);
        self.projm(enc, hc, 0, "laya.scorer_fc.weight", &b.ffn, d, d, k32, false);
        self.enc_bias(enc, &b.ffn, "laya.scorer_fc.bias", d, k32);
        self.enc_act(enc, &b.ffn, &b.ffn, Act::GeluErf, d * k32);
    }
}
