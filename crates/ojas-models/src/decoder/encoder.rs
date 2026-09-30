//! The bidirectional encoder block, shared by every encoder tower on Metal.
//!
//! Three stacks run through [`DecoderGpu::encode_block`]: the qwen3vl ViT tower
//! (`vision.rs`), the ModernBERT text encoder and the Laya decision head
//! (`text_encoder.rs`). Each is a pre-norm residual block over `M` packed rows:
//!
//! ```text
//! h = norm1(x);  x += out_proj(attention(rope(split(qkv_proj(h)))))
//! h = norm2(x);  x += down_proj(mlp(up_proj(h)))
//! ```
//!
//! and they differ only in what [`Block`] describes: LayerNorm with or without bias
//! (or no first norm at all, ModernBERT's layer 0), projection biases, a plain or a
//! gated MLP and its activation, the RoPE base (or none), and which keys each query
//! row may see.
//!
//! Moving the ViT tower onto this block changed none of its output (compared bit for
//! bit), and `vit_qkv_prep` is held bit-identical to the split, rotation and half
//! conversion it replaced (`ojas-metal/tests/vision_kernels.rs`). `examples/vision_gate.rs`
//! checks the tower against its CPU oracle and `examples/laya_gate.rs` the text encoder
//! against the PyTorch reference.

use super::*;
use metal::MTLSize;
use std::ffi::c_void;

/// Activation selectors understood by `ffn_act` in the Metal prelude.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Act {
    /// Tanh-approximated GELU: the qwen3vl ViT.
    GeluTanh,
    /// Exact erf GELU: PyTorch `nn.GELU()`, ModernBERT.
    GeluErf,
    Relu,
}

impl Act {
    pub(crate) fn code(self) -> u32 {
        match self { Act::GeluTanh => 1, Act::GeluErf => 3, Act::Relu => 4 }
    }
}

/// A LayerNorm's tensors. Encoders here all use mean-subtracting LayerNorm; the
/// bias is absent in ModernBERT.
#[derive(Clone, Debug)]
pub(crate) struct Norm {
    pub(crate) weight: String,
    pub(crate) bias: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mlp {
    /// `down(act(up(h)))`, `up` producing `ffn` columns.
    Plain,
    /// `down(act(g) * u)` with `[g | u] = up(h)`, `up` producing `2 * ffn` columns
    /// and the first half activated (ModernBERT's `Wi`).
    Gated,
}

/// Which keys each query row attends to.
#[derive(Clone, Copy)]
pub(crate) enum Keys<'b> {
    /// All `M` rows are one sequence and every row sees every row. Uses the MMA
    /// kernel when `mma` is set (the caller has probed the pipeline).
    All { mma: bool },
    /// Row `m` sees keys `[span[m].0, span[m].1)`: packed sequences, optionally
    /// windowed. `span` holds one `uint2` per row. With `tiles` (the MMA kernel's
    /// `(q0, nq, klo, khi)` descriptors, one per 32-row query tile inside a sequence,
    /// and their count) the tiled kernel runs; it reads K and V in 8-row blocks past
    /// a tile's key range and masks them, so the 64 rows past the last token must hold
    /// finite values (zero on allocation).
    Spans { span: &'b metal::Buffer, tiles: Option<(&'b metal::Buffer, u32)> },
}

/// NEOX rotary embedding of Q and K, driven by the position table in
/// [`Scratch::pos`]: `theta_j = pos * base^(-2j / freq_dims)` for pair
/// `(j, j + hd/2)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rope {
    pub(crate) base: f32,
    /// `hd` for a text rope; `hd/2` for ggml's vision rope.
    pub(crate) freq_dims: u32,
}

/// One encoder block, by tensor name.
#[derive(Clone, Debug)]
pub(crate) struct Block {
    /// `None` feeds `x` to the QKV projection unnormalized (ModernBERT layer 0).
    pub(crate) attn_norm: Option<Norm>,
    pub(crate) qkv: String,
    pub(crate) qkv_bias: Option<String>,
    pub(crate) out: String,
    pub(crate) out_bias: Option<String>,
    pub(crate) ffn_norm: Norm,
    pub(crate) up: String,
    pub(crate) up_bias: Option<String>,
    pub(crate) down: String,
    pub(crate) down_bias: Option<String>,
    pub(crate) mlp: Mlp,
    pub(crate) act: Act,
    /// `None` for a block without positional rotation.
    pub(crate) rope: Option<Rope>,
}

/// Shape shared by every block of one stack.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Geom {
    pub(crate) d: usize,
    pub(crate) n_head: usize,
    pub(crate) hd: usize,
    /// MLP hidden width (per half, for a gated MLP).
    pub(crate) ffn: usize,
    pub(crate) eps: f32,
}

/// Working buffers for one encode, all f32 unless noted. Row counts must be padded
/// to a multiple of 32: `projm`'s staged GEMMs (Q8, and f16 without simdgroup
/// matrices) store whole 32-row tiles.
pub(crate) struct Scratch<'b> {
    /// Residual stream `[rows, d]`.
    pub(crate) x: &'b metal::Buffer,
    /// Norm output and attention output `[rows, d]`.
    pub(crate) h: &'b metal::Buffer,
    pub(crate) qkv: &'b metal::Buffer,
    pub(crate) q: &'b metal::Buffer,
    /// K and V as f16, written by `vit_qkv_prep`; see [`Keys`] for the rows past `M`
    /// each attention kernel needs kept finite.
    pub(crate) kh: &'b metal::Buffer,
    pub(crate) vh: &'b metal::Buffer,
    /// MLP hidden `[rows, ffn]`.
    pub(crate) ffn: &'b metal::Buffer,
    /// Gated MLP pre-activation `[rows, 2*ffn]`; unused by a plain MLP.
    pub(crate) ffn_wide: &'b metal::Buffer,
    /// `vit_rope` position table: four section sizes, then four streams per row.
    pub(crate) pos: &'b metal::Buffer,
}

impl<'a> DecoderGpu<'a> {
    /// LayerNorm over `m` rows of `src` into `dst` (which may alias `src`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enc_layernorm(&self, enc: &metal::ComputeCommandEncoderRef, src: &metal::Buffer,
                                dst: &metal::Buffer, n: &Norm, d: u32, eps: f32, m: u32) {
        let w = &self.wt.w32[&n.weight];
        // With no bias the kernel does not read slot 5; bind the weight there.
        let (b, has_bias) = match &n.bias { Some(b) => (&self.wt.w32[b], 1u32), None => (w, 0u32) };
        self.enc_reduce(enc, "vit_layernorm_m", &[(src, 0), (w, 1), (dst, 2), (b, 5)],
            &[(3, d), (6, has_bias)], &[(4, eps)], m as u64, 256);
    }

    /// `y[r, :] += bias` over `m` rows of width `n`.
    pub(crate) fn enc_bias(&self, enc: &metal::ComputeCommandEncoderRef, y: &metal::Buffer,
                           bias: &str, n: u32, m: u32) {
        let total = n * m;
        self.enc_reduce(enc, "add_rowbias_m", &[(y, 0), (&self.wt.w32[bias], 1)],
            &[(2, n), (3, total)], &[], total.div_ceil(64) as u64, 64);
    }

    /// Elementwise `ffn_act` over `n` values, `out` may alias `x`.
    pub(crate) fn enc_act(&self, enc: &metal::ComputeCommandEncoderRef, x: &metal::Buffer,
                          out: &metal::Buffer, act: Act, n: u32) {
        self.enc_reduce(enc, "act_m", &[(x, 0), (out, 1)], &[(2, n), (3, act.code())], &[],
            n.div_ceil(256) as u64, 256);
    }

    /// Encode one block over the first `m` rows of `s.x`, in place.
    pub(crate) fn encode_block(&self, enc: &metal::ComputeCommandEncoderRef, b: &Block, g: &Geom,
                               s: &Scratch, keys: Keys, m: usize) {
        self.encode_attention_half(enc, b, g, s, keys, m);
        self.encode_mlp_half(enc, b, g, s.x, s.h, s.ffn, s.ffn_wide, m);
    }

    /// The attention sub-block up to, not including, its out-projection: norm, QKV,
    /// rotation and attention over the first `m` rows, leaving the attention output
    /// in `s.h`. [`DecoderGpu::encode_mlp_half`] finishes the block, and may do so on
    /// a subset of the rows (the Laya head's last block keeps only the rows it reads).
    pub(crate) fn encode_attention_half(&self, enc: &metal::ComputeCommandEncoderRef, b: &Block, g: &Geom,
                                        s: &Scratch, keys: Keys, m: usize) {
        let d = g.d as u32;
        let m32 = m as u32;
        let qkv_in = match &b.attn_norm {
            Some(n) => { self.enc_layernorm(enc, s.x, s.h, n, d, g.eps, m32); s.h }
            None => s.x,
        };
        self.projm(enc, qkv_in, 0, &b.qkv, s.qkv, d, 3 * d, m32, false);
        if let Some(bias) = &b.qkv_bias { self.enc_bias(enc, s.qkv, bias, 3 * d, m32); }
        self.encode_qkv_prep(enc, b, g, s, m);
        self.encode_attention(enc, g, s, keys, m);
    }

    /// The rest of the block over `m` rows: the attention out-projection from `h`
    /// into the residual `x`, then the MLP sub-block. `h` is overwritten by the
    /// second norm; `ffn`/`ffn_wide` are the MLP's working rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_mlp_half(&self, enc: &metal::ComputeCommandEncoderRef, b: &Block, g: &Geom,
                                  x: &metal::Buffer, h: &metal::Buffer, ffn_buf: &metal::Buffer,
                                  ffn_wide: &metal::Buffer, m: usize) {
        let (d, ffn) = (g.d as u32, g.ffn as u32);
        let m32 = m as u32;
        // Out-projection straight into the residual: `projm(accum)` gives `x += o`,
        // then the bias rides the same buffer, one buffer and one dispatch fewer than
        // staging `o`. The bias therefore enters the f32 sum last here and first in
        // the CPU oracle (`cpu_math::matmul` seeds each dot with it): a last-bits f32
        // difference, orders of magnitude under the f16 activation rounding.
        self.projm(enc, h, 0, &b.out, x, d, d, m32, true);
        if let Some(bias) = &b.out_bias { self.enc_bias(enc, x, bias, d, m32); }

        self.enc_layernorm(enc, x, h, &b.ffn_norm, d, g.eps, m32);
        match b.mlp {
            Mlp::Plain => {
                self.projm(enc, h, 0, &b.up, ffn_buf, d, ffn, m32, false);
                if let Some(bias) = &b.up_bias { self.enc_bias(enc, ffn_buf, bias, ffn, m32); }
                self.enc_act(enc, ffn_buf, ffn_buf, b.act, ffn * m32);
            }
            Mlp::Gated => {
                self.projm(enc, h, 0, &b.up, ffn_wide, d, 2 * ffn, m32, false);
                if let Some(bias) = &b.up_bias { self.enc_bias(enc, ffn_wide, bias, 2 * ffn, m32); }
                let n = ffn * m32;
                self.enc_reduce(enc, "ffn_gu_rows", &[(ffn_wide, 0), (ffn_buf, 1)],
                    &[(2, ffn), (3, n), (4, b.act.code())], &[], n.div_ceil(256) as u64, 256);
            }
        }
        self.projm(enc, ffn_buf, 0, &b.down, x, ffn, d, m32, true);
        if let Some(bias) = &b.down_bias { self.enc_bias(enc, x, bias, d, m32); }
    }

    /// Split the fused QKV projection, rotate Q and K, and write K and V as half,
    /// in one pass (`vit_qkv_prep`).
    pub(crate) fn encode_qkv_prep(&self, enc: &metal::ComputeCommandEncoderRef, b: &Block, g: &Geom,
                                  s: &Scratch, m: usize) {
        let (d, hd) = (g.d as u32, g.hd as u32);
        let (freq_dims, base) = b.rope.map_or((0, 0.0), |r| (r.freq_dims, r.base));
        let threads = m as u32 * (d / 2);
        self.enc_reduce(enc, "vit_qkv_prep", &[(s.qkv, 0), (s.q, 1), (s.kh, 2), (s.vh, 3), (s.pos, 5)],
            &[(4, d), (6, hd), (7, m as u32), (8, freq_dims)], &[(9, base)], threads.div_ceil(64) as u64, 64);
    }

    /// Bidirectional attention of `s.q` over `s.kh`/`s.vh` into `s.h`.
    pub(crate) fn encode_attention(&self, enc: &metal::ComputeCommandEncoderRef, g: &Geom, s: &Scratch,
                                   keys: Keys, m: usize) {
        let (d, nh, hd) = (g.d as u32, g.n_head as u32, g.hd as u32);
        let m32 = m as u32;
        let scale = 1.0 / (hd as f32).sqrt();
        match keys {
            Keys::All { mma: true } => {
                enc.set_compute_pipeline_state(&self.p[&format!("attention_m_mma_bidir_{hd}")]);
                enc.set_buffer(0, Some(s.q), 0);
                enc.set_buffer(1, Some(s.kh), 0);
                enc.set_buffer(2, Some(s.vh), 0);
                enc.set_buffer(3, Some(s.h), 0);
                for (i, val) in [(4u32, hd), (5, d), (6, m32), (7, 1u32), (9, nh), (10, m32)] {
                    enc.set_bytes(i as u64, 4, &val as *const u32 as *const c_void);
                }
                enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, m.div_ceil(32) as u64, 1),
                                           MTLSize::new(256, 1, 1));
            }
            Keys::All { mma: false } => {
                self.enc_reduce(enc, "attention_m_bidir", &[(s.q, 0), (s.kh, 1), (s.vh, 2), (s.h, 3)],
                    &[(4, hd), (5, d), (6, m32), (7, 1), (9, nh)], &[(8, scale)], (m32 * nh) as u64, 256);
            }
            Keys::Spans { span, tiles: Some((tiles, n_tiles)) } => {
                enc.set_compute_pipeline_state(&self.p[&ojas_metal::kernels::attn::attn_mma_span_name(hd)]);
                for (i, b) in [(0u64, s.q), (1, s.kh), (2, s.vh), (3, s.h), (11, tiles), (12, span)] {
                    enc.set_buffer(i, Some(b), 0);
                }
                for (i, val) in [(4u64, hd), (5, d), (6, 0u32), (7, 1u32), (9, nh), (10, m32)] {
                    enc.set_bytes(i, 4, &val as *const u32 as *const c_void);
                }
                enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, n_tiles as u64, 1), MTLSize::new(256, 1, 1));
            }
            Keys::Spans { span, tiles: None } => {
                self.enc_reduce(enc, ojas_metal::kernels::attn::ATTN_BIDIR_SPAN,
                    &[(s.q, 0), (s.kh, 1), (s.vh, 2), (s.h, 3), (span, 10)],
                    &[(4, hd), (5, d), (6, m32), (7, 1), (9, nh)], &[(8, scale)], (m32 * nh) as u64, 256);
            }
        }
    }
}
