//! The ModernBERT text encoder and a decision head read at marker tokens, on CUDA
//! (`ojas-models/src/decoder/text_encoder.rs` is the Metal twin; the shape is
//! [`ojas_arch::text_encoder::TextEncoderSpec`]).
//!
//! A request is a batch of independent token sequences packed into one row buffer
//! with no padding between them. Every block is the same pre-norm residual block over
//! the `M` packed rows, and what is specific to the model is data:
//!
//! * per-row key spans restrict attention to the row's own sequence, and on the
//!   sliding-window layers to `|i - j| <= window` inside it (`attention_m_bidir_span`);
//! * the rope position restarts at 0 for every sequence;
//! * layer `l` uses the local rope base and window when `l % swa_pattern != 0`;
//! * layer 0 has no attention norm (ModernBERT feeds the embedding LayerNorm's output
//!   straight to its QKV projection);
//! * the MLP is gated over one fused `ffn_up` (first half activated), erf GELU.
//!
//! When the file carries a marker head, [`CudaBert::marker_head_forward`] continues on
//! the same rows: the question-type embedding is added per sequence and the head's
//! `nn.TransformerEncoderLayer` blocks (biased, ReLU, no rope) run with sequence-wide
//! spans. Only the option-marker rows are read afterwards, so the last head block
//! finishes, and the scorer's LayerNorm, first linear and GELU run, on those rows alone.
//! The scorer's final `d -> 1` projection finishes on the host (`ojas_decision`).
//!
//! Weights are the file's own f16 (quantized tensors are dequantized to f16 on load):
//! every token goes through the batched `gemm_mm_f16`, one tensor-core pass unless
//! `OJAS_CUDA_GEMM` asks for another precision.

use crate::kernels::attn_bidir::{mma_span_tiles, MMA_SPAN_BQ};
use crate::qwen35::{blocks, bytes_f32, f32_bytes, read_f32_tensor, u32s_bytes, GemmMode, Lin, Repr};
use crate::{CuBuf, CudaGpu};
use anyhow::{bail, ensure, Context, Result};
use ojas_arch::text_encoder::{MarkerHeadSpec, TextEncoderSpec};
use ojas_core::KernelRuntime;
use ojas_decision::MarkerHeadOut;
use ojas_formats::gguf::Gguf;
use std::cell::RefCell;

/// `ffn_act` codes.
const ACT_GELU_ERF: u32 = 3;
const ACT_RELU: u32 = 4;

/// A LayerNorm's weight and, for the head's norms, bias.
struct Norm {
    w: CuBuf,
    b: Option<CuBuf>,
}

/// One block, by tensor; the encoder's have no biases and no first norm on block 0.
struct Block {
    attn_norm: Option<Norm>,
    qkv: Lin,
    qkv_b: Option<CuBuf>,
    out: Lin,
    out_b: Option<CuBuf>,
    ffn_norm: Norm,
    up: Lin,
    up_b: Option<CuBuf>,
    down: Lin,
    down_b: Option<CuBuf>,
}

struct Head {
    spec: MarkerHeadSpec,
    blocks: Vec<Block>,
    /// `[n_types, d]` f32.
    token_types: CuBuf,
    n_types: usize,
    cls_norm: Norm,
    cls: Lin,
    cls_b: CuBuf,
}

/// Row buffers for packed requests, kept on the encoder and grown when a request needs
/// more rows.
struct Arena {
    cap: usize,
    x: CuBuf,
    h: CuBuf,
    qkv: CuBuf,
    q: CuBuf,
    kb: CuBuf,
    vb: CuBuf,
    /// K and V as f16, `cap + 64` rows: the span kernel reads only inside each span, the
    /// spare rows keep the layout shared with the tiled kernels.
    kh: CuBuf,
    vh: CuBuf,
    ffn: CuBuf,
    ffn_wide: CuBuf,
    /// Split-precision scratch (`GemmMode::Split`), the widest `[cap, k]` input.
    lo: Option<CuBuf>,
    /// GEMV staging (`GemmMode::Exact`), `[cap, max(d, ffn)]`.
    tmp: Option<CuBuf>,
    pos: CuBuf,
    span_global: CuBuf,
    span_local: CuBuf,
    /// `attention_m_mma_span` tiles for the global and the local spans, and their count;
    /// `None` where the head width has no tiled kernel.
    tiles_global: Option<CuBuf>,
    tiles_local: Option<CuBuf>,
    n_tiles: usize,
    tokens: CuBuf,
    keep: CuBuf,
}

pub struct CudaBert {
    gpu: CudaGpu,
    spec: TextEncoderSpec,
    gemm: GemmMode,
    vocab: usize,
    /// `[vocab, d]` f16.
    embd: CuBuf,
    embd_norm: CuBuf,
    out_norm: CuBuf,
    blocks: Vec<Block>,
    head: Option<Head>,
    /// `d` zeros: the bias slot of a LayerNorm without one.
    zero: CuBuf,
    arena: RefCell<Option<Arena>>,
    /// Wall time of the passes so far, seconds.
    gpu_s: std::cell::Cell<f64>,
    profile: crate::KernelProfile,
}

impl CudaBert {
    pub fn load(ordinal: usize, g: &mut Gguf) -> Result<CudaBert> {
        ensure!(g.arch() == "modern-bert", "CudaBert runs modern-bert; got {}", g.arch());
        let spec = TextEncoderSpec::from_gguf(g)?;
        let mut gpu = CudaGpu::new(ordinal)?;
        // One tensor-core pass per matmul: with Q carried at f32 precision through the
        // attention the encoders hold parity with it (Julia-1 within 8e-3), and the
        // split pass would double every GEMM. `OJAS_CUDA_GEMM` still overrides.
        let gemm = if std::env::var("OJAS_CUDA_GEMM").is_ok() { GemmMode::from_env() } else { GemmMode::Fast };
        // `q35_split_lo` (the split-precision residual) lives in the qwen35 family.
        let families: &[&str] = if gemm == GemmMode::Split { &["ops", "gemm_f16", "vision", "bert", "attn_bidir", "qwen35"] } else { &["ops", "gemm_f16", "vision", "bert", "attn_bidir"] };
        for fam in families {
            gpu.ensure_family(fam).with_context(|| format!("compiling CUDA family {fam}"))?;
        }
        let t0 = std::time::Instant::now();
        let lin = |g: &mut Gguf, name: &str| -> Result<Lin> {
            let (dims, ty, bytes) = g.read_tensor(name).with_context(|| format!("reading {name}"))?;
            let k = dims.first().copied().unwrap_or(1) as usize;
            let b = match ty {
                1 => bytes,
                0 => bytes_f32(&bytes).iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect(),
                t => bail!("{name}: unsupported GGUF type {t}"),
            };
            let n = b.len() / 2 / k.max(1);
            Lin::f16(&gpu, &b, n, k)
        };
        let vf = |g: &mut Gguf, name: &str| -> Result<CuBuf> { gpu.upload_bytes(&f32_bytes(&read_f32_tensor(g, name)?)) };
        let opt = |g: &mut Gguf, name: &str| -> Result<Option<CuBuf>> {
            if g.tensors.contains_key(name) { Ok(Some(vf(g, name)?)) } else { Ok(None) }
        };
        let norm = |g: &mut Gguf, name: &str| -> Result<Norm> {
            Ok(Norm { w: vf(g, &format!("{name}.weight"))?, b: opt(g, &format!("{name}.bias"))? })
        };
        let block = |g: &mut Gguf, i: u32, first_norm: bool| -> Result<Block> {
            let p = |t: &str| format!("blk.{i}.{t}");
            Ok(Block {
                attn_norm: if first_norm { Some(norm(g, &p("attn_norm"))?) } else { None },
                qkv: lin(g, &p("attn_qkv.weight"))?, qkv_b: opt(g, &p("attn_qkv.bias"))?,
                out: lin(g, &p("attn_output.weight"))?, out_b: opt(g, &p("attn_output.bias"))?,
                ffn_norm: norm(g, &p("ffn_norm"))?,
                up: lin(g, &p("ffn_up.weight"))?, up_b: opt(g, &p("ffn_up.bias"))?,
                down: lin(g, &p("ffn_down.weight"))?, down_b: opt(g, &p("ffn_down.bias"))?,
            })
        };
        let d = spec.d as usize;
        let mut blocks_v = Vec::with_capacity(spec.layers as usize);
        for i in 0..spec.layers {
            blocks_v.push(block(g, i, g.tensors.contains_key(&format!("blk.{i}.attn_norm.weight")))?);
        }
        let head = match &spec.marker_head {
            None => None,
            Some(h) => {
                let mut hb = Vec::with_capacity(h.blocks as usize);
                for i in h.first..h.first + h.blocks { hb.push(block(g, i, true)?); }
                let types = read_f32_tensor(g, "token_types.weight")?;
                ensure!(types.len() % d == 0, "token_types.weight is not [*, {d}]");
                Some(Head {
                    spec: h.clone(), blocks: hb, n_types: types.len() / d, token_types: gpu.upload_bytes(&f32_bytes(&types))?,
                    cls_norm: norm(g, "cls.norm")?, cls: lin(g, "cls.weight")?, cls_b: vf(g, "cls.bias")?,
                })
            }
        };
        let embd = lin(g, "token_embd.weight")?;
        ensure!(embd.k == d, "token_embd.weight is [{}, {}], expected [*, {d}]", embd.n, embd.k);
        let Repr::F16(table) = embd.w else { unreachable!("lin uploads f16") };
        let bert = CudaBert {
            vocab: embd.n, embd: table,
            embd_norm: vf(g, "token_embd_norm.weight")?, out_norm: vf(g, "output_norm.weight")?,
            zero: gpu.upload_bytes(&vec![0u8; d * 4])?,
            gemm, blocks: blocks_v, head, spec, gpu,
            arena: RefCell::new(None), gpu_s: std::cell::Cell::new(0.0), profile: crate::KernelProfile::from_env(),
        };
        bert.gpu.sync()?;
        tracing::info!(target: "cuda:bert", "loaded {} encoder blocks{} in {:.1}s | d={} heads={} ffn={} window={} gemm={:?}",
            bert.spec.layers, bert.head.as_ref().map_or(String::new(), |h| format!(" + {} head blocks", h.spec.blocks)),
            t0.elapsed().as_secs_f64(), bert.spec.d, bert.spec.n_head, bert.spec.ffn, bert.spec.window, bert.gemm);
        Ok(bert)
    }

    pub fn has_marker_head(&self) -> bool { self.head.is_some() }
    pub fn width(&self) -> usize { self.spec.d as usize }
    pub fn max_positions(&self) -> usize { self.spec.max_positions as usize }
    /// Wall time of the passes so far, seconds.
    pub fn gpu_seconds(&self) -> f64 { self.gpu_s.get() }

    fn k(&self, name: &str, bufs: &[(&CuBuf, u64)], consts: &[u32], grid: [u32; 3], block: [u32; 3]) -> Result<()> {
        self.gpu.dispatch_profiled(&self.profile, name, bufs, consts, grid, block).with_context(|| format!("launching {name}"))
    }

    /// `y[rows, n] (+)= x[rows, :] · W[n, :]`.
    fn mm(&self, a: &Arena, x: (&CuBuf, u64), w: &Lin, y: (&CuBuf, u64), rows: usize, accum: bool) -> Result<()> {
        let (k, n) = (w.k as u32, w.n as u32);
        let el = |t: usize| [blocks(t, 256), 1, 1];
        let Repr::F16(wb) = &w.w else { bail!("the encoder's weights are held in f16") };
        match self.gemm {
            GemmMode::Exact => {
                let dst = if accum { (a.tmp.as_ref().expect("exact-mode staging"), 0) } else { y };
                self.k("gemv_m_f16", &[x, (wb, 0), dst], &[k, n, rows as u32], [blocks(w.n, 8), 1, 1], [256, 1, 1])?;
                if accum {
                    let t = rows * w.n;
                    self.k("add_inplace", &[y, dst], &[t as u32], el(t), [256, 1, 1])?;
                }
            }
            mode => {
                let grid = [blocks(w.n, 128), blocks(rows, 128), 1];
                self.k("gemm_mm_f16", &[x, (wb, 0), y], &[k, n, accum as u32, rows as u32], grid, [256, 1, 1])?;
                if mode == GemmMode::Split {
                    let lo = a.lo.as_ref().expect("split-mode scratch");
                    let t = rows * w.k;
                    self.k("q35_split_lo", &[x, (lo, 0)], &[t as u32], el(t), [256, 1, 1])?;
                    self.k("gemm_mm_f16", &[(lo, 0), (wb, 0), y], &[k, n, 1, rows as u32], grid, [256, 1, 1])?;
                }
            }
        }
        Ok(())
    }

    fn layernorm(&self, src: (&CuBuf, u64), dst: (&CuBuf, u64), w: &CuBuf, b: Option<&CuBuf>, eps: f32, rows: usize) -> Result<()> {
        let (b, has) = match b { Some(b) => (b, 1u32), None => (&self.zero, 0u32) };
        self.k("bert_layernorm_m", &[src, (w, 0), dst, (b, 0)], &[self.spec.d, eps.to_bits(), has], [rows as u32, 1, 1], [256, 1, 1])
    }

    fn bias(&self, y: (&CuBuf, u64), b: Option<&CuBuf>, n: usize, rows: usize) -> Result<()> {
        let Some(b) = b else { return Ok(()) };
        let t = n * rows;
        self.k("add_rowbias_m", &[y, (b, 0)], &[n as u32, t as u32], [blocks(t, 256), 1, 1], [256, 1, 1])
    }

    /// The attention sub-block up to, not including, its out-projection: norm, QKV,
    /// rotation and attention over `m` rows, leaving the attention output in `a.h`.
    #[allow(clippy::too_many_arguments)]
    fn attention_half(&self, a: &Arena, b: &Block, n_head: usize, eps: f32, rope_base: Option<f32>, local: bool, m: usize) -> Result<()> {
        let d = self.spec.d as usize;
        let (hd, d32, m32) = (d / n_head, d as u32, m as u32);
        let el = |t: usize| [blocks(t, 256), 1, 1];
        let qkv_in = match &b.attn_norm {
            Some(n) => { self.layernorm((&a.x, 0), (&a.h, 0), &n.w, n.b.as_ref(), eps, m)?; &a.h }
            None => &a.x,
        };
        self.mm(a, (qkv_in, 0), &b.qkv, (&a.qkv, 0), m, false)?;
        self.bias((&a.qkv, 0), b.qkv_b.as_ref(), 3 * d, m)?;
        self.k("vit_qkv_split", &[(&a.qkv, 0), (&a.q, 0), (&a.kb, 0), (&a.vb, 0)], &[d32, (m * d) as u32], el(m * d), [256, 1, 1])?;
        if let Some(base) = rope_base {
            let pairs = m * n_head * (hd / 2);
            for v in [&a.q, &a.kb] {
                self.k("bert_rope_m", &[(v, 0), (&a.pos, 0)], &[hd as u32, base.to_bits(), d32, m32], el(pairs), [256, 1, 1])?;
            }
        }
        self.k("copy_f32_half", &[(&a.kb, 0), (&a.kh, 0)], &[(m * d) as u32], el(m * d), [256, 1, 1])?;
        self.k("copy_f32_half", &[(&a.vb, 0), (&a.vh, 0)], &[(m * d) as u32], el(m * d), [256, 1, 1])?;
        let scale = 1.0 / (hd as f32).sqrt();
        let (span, tiles) = if local { (&a.span_local, &a.tiles_local) } else { (&a.span_global, &a.tiles_global) };
        match tiles {
            Some(tiles) => self.k(&format!("attention_m_mma_span_{hd}"), &[(&a.q, 0), (&a.kh, 0), (&a.vh, 0), (&a.h, 0), (tiles, 0), (span, 0)],
                &[hd as u32, d32, m32, 1, scale.to_bits(), n_head as u32, m32], [a.n_tiles as u32, n_head as u32, 1], [128, 1, 1]),
            None => self.k("attention_m_bidir_span", &[(&a.q, 0), (&a.kh, 0), (&a.vh, 0), (&a.h, 0), (span, 0)],
                &[hd as u32, d32, m32, 1, scale.to_bits(), n_head as u32], [(m * n_head) as u32, 1, 1], [256, 1, 1]),
        }
    }

    /// The rest of a block over `rows` rows: the out-projection from `h` into the residual
    /// `x`, then the MLP, gated (erf GELU) or plain (`act`).
    #[allow(clippy::too_many_arguments)]
    fn mlp_half(&self, a: &Arena, b: &Block, x: &CuBuf, h: &CuBuf, eps: f32, ffn: usize, gated: bool, act: u32, rows: usize) -> Result<()> {
        let d = self.spec.d as usize;
        let el = |t: usize| [blocks(t, 256), 1, 1];
        self.mm(a, (h, 0), &b.out, (x, 0), rows, true)?;
        self.bias((x, 0), b.out_b.as_ref(), d, rows)?;
        self.layernorm((x, 0), (h, 0), &b.ffn_norm.w, b.ffn_norm.b.as_ref(), eps, rows)?;
        if gated {
            self.mm(a, (h, 0), &b.up, (&a.ffn_wide, 0), rows, false)?;
            self.bias((&a.ffn_wide, 0), b.up_b.as_ref(), 2 * ffn, rows)?;
            let t = rows * ffn;
            self.k("ffn_gu_rows", &[(&a.ffn_wide, 0), (&a.ffn, 0)], &[ffn as u32, t as u32, act], el(t), [256, 1, 1])?;
        } else {
            self.mm(a, (h, 0), &b.up, (&a.ffn, 0), rows, false)?;
            self.bias((&a.ffn, 0), b.up_b.as_ref(), ffn, rows)?;
            let t = rows * ffn;
            self.k("act_m", &[(&a.ffn, 0), (&a.ffn, 0)], &[t as u32, act], el(t), [256, 1, 1])?;
        }
        self.mm(a, (&a.ffn, 0), &b.down, (x, 0), rows, true)?;
        self.bias((x, 0), b.down_b.as_ref(), d, rows)
    }

    /// The arena, grown to hold `m` rows, with the request's tokens, positions and key
    /// spans written into it.
    fn arena(&self, seqs: &[Vec<u32>]) -> Result<std::cell::RefMut<'_, Arena>> {
        let te = &self.spec;
        ensure!(!seqs.is_empty() && seqs.iter().all(|s| !s.is_empty()), "every sequence needs at least one token");
        for (i, s) in seqs.iter().enumerate() {
            ensure!(s.len() <= te.max_positions as usize, "sequence {i} has {} tokens, past the encoder's {} positions", s.len(), te.max_positions);
            ensure!(s.iter().all(|&t| (t as usize) < self.vocab), "sequence {i} holds a token id outside the {}-token vocabulary", self.vocab);
        }
        let m: usize = seqs.iter().map(Vec::len).sum();
        let mut arena = self.arena.borrow_mut();
        if arena.as_ref().is_none_or(|a| a.cap < m) {
            let cap = m.max(arena.as_ref().map_or(0, |a| a.cap * 2)).div_ceil(32) * 32;
            *arena = None;
            *arena = Some(self.alloc_arena(cap)?);
        }
        let mut a = std::cell::RefMut::map(arena, |a| a.as_mut().expect("allocated above"));
        let (mut pos, mut global, mut local, mut tokens) = (Vec::with_capacity(m), Vec::with_capacity(2 * m), Vec::with_capacity(2 * m), Vec::with_capacity(m));
        let w = te.window as usize;
        let mut start = 0usize;
        for s in seqs {
            let end = start + s.len();
            for (j, &t) in s.iter().enumerate() {
                let r = start + j;
                pos.push(j as u32);
                global.extend_from_slice(&[start as u32, end as u32]);
                local.extend_from_slice(&[r.saturating_sub(w).max(start) as u32, (r + w + 1).min(end) as u32]);
                tokens.push(t);
            }
            start = end;
        }
        self.gpu.write_bytes(&mut a.pos, 0, &u32s_bytes(&pos))?;
        self.gpu.write_bytes(&mut a.span_global, 0, &u32s_bytes(&global))?;
        self.gpu.write_bytes(&mut a.span_local, 0, &u32s_bytes(&local))?;
        self.gpu.write_bytes(&mut a.tokens, 0, &u32s_bytes(&tokens))?;
        if a.tiles_global.is_some() {
            let lens: Vec<usize> = seqs.iter().map(Vec::len).collect();
            let (tg, tl) = (mma_span_tiles(&lens, &global), mma_span_tiles(&lens, &local));
            a.n_tiles = tg.len() / 4;
            self.gpu.write_bytes(a.tiles_global.as_mut().unwrap(), 0, &u32s_bytes(&tg))?;
            self.gpu.write_bytes(a.tiles_local.as_mut().unwrap(), 0, &u32s_bytes(&tl))?;
        }
        Ok(a)
    }

    fn alloc_arena(&self, cap: usize) -> Result<Arena> {
        let te = &self.spec;
        let d = te.d as usize;
        let ffn = self.head.as_ref().map_or(te.ffn, |h| h.spec.ffn.max(te.ffn)).max(te.d) as usize;
        let f = |n: usize| self.gpu.alloc_bytes(n * 4);
        // tiles: at most one per MMA_SPAN_BQ rows of each sequence, so cap/BQ + one per sequence
        let tiled = self.gpu.has_kernel(&format!("attention_m_mma_span_{}", te.hd));
        let max_tiles = cap / MMA_SPAN_BQ + cap;
        Ok(Arena {
            cap,
            tiles_global: if tiled { Some(f(4 * max_tiles)?) } else { None },
            tiles_local: if tiled { Some(f(4 * max_tiles)?) } else { None },
            n_tiles: 0,
            x: f(cap * d)?, h: f(cap * d)?, qkv: f(cap * 3 * d)?, q: f(cap * d)?, kb: f(cap * d)?, vb: f(cap * d)?,
            kh: self.gpu.alloc_bytes((cap + 64) * d * 2)?, vh: self.gpu.alloc_bytes((cap + 64) * d * 2)?,
            ffn: f(cap * ffn)?, ffn_wide: f(cap * 2 * te.ffn as usize)?,
            lo: if self.gemm == GemmMode::Split { Some(f(cap * ffn)?) } else { None },
            tmp: if self.gemm == GemmMode::Exact { Some(f(cap * ffn)?) } else { None },
            pos: f(cap)?, span_global: f(2 * cap)?, span_local: f(2 * cap)?, tokens: f(cap)?, keep: f(cap)?,
        })
    }

    /// Embed, run every encoder block and the final norm, then `tail`; one synchronized
    /// pass whose wall time is returned.
    fn run_text(&self, a: &Arena, seqs: &[Vec<u32>], tail: impl FnOnce(&Arena, usize) -> Result<()>) -> Result<f64> {
        let te = &self.spec;
        let d = te.d as usize;
        let m: usize = seqs.iter().map(Vec::len).sum();
        let t0 = std::time::Instant::now();
        self.k("bert_embed_f16", &[(&self.embd, 0), (&a.tokens, 0), (&a.x, 0)], &[te.d, m as u32], [blocks(m * d, 256), 1, 1], [256, 1, 1])?;
        self.layernorm((&a.x, 0), (&a.x, 0), &self.embd_norm, None, te.eps, m)?;
        for (l, b) in self.blocks.iter().enumerate() {
            let local = te.is_local(l);
            let base = if local { te.rope_base_local } else { te.rope_base };
            self.attention_half(a, b, te.n_head as usize, te.eps, Some(base), local, m)?;
            self.mlp_half(a, b, &a.x, &a.h, te.eps, te.ffn as usize, true, ACT_GELU_ERF, m)?;
        }
        self.layernorm((&a.x, 0), (&a.x, 0), &self.out_norm, None, te.eps, m)?;
        tail(a, m)?;
        self.gpu.sync()?;
        let s = t0.elapsed().as_secs_f64();
        self.gpu_s.set(self.gpu_s.get() + s);
        self.profile.report(&self.gpu, &format!("text pass, {m} rows in {:.1} ms", s * 1e3));
        Ok(s)
    }

    /// Final hidden states of the encoder for each sequence, `[len * d]` row-major per
    /// sequence (after the final LayerNorm, before any head).
    pub fn encode_text(&self, seqs: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let d = self.spec.d as usize;
        let a = self.arena(seqs)?;
        self.run_text(&a, seqs, |_, _| Ok(()))?;
        let mut out = Vec::with_capacity(seqs.len());
        let mut start = 0usize;
        for s in seqs {
            let mut b = vec![0u8; s.len() * d * 4];
            self.gpu.read_bytes(&a.x, start * d * 4, &mut b)?;
            out.push(bytes_f32(&b));
            start += s.len();
        }
        Ok(out)
    }

    /// The encoder, then the marker head, over `seqs`. `qtypes[i]` selects the row of
    /// `token_types` added to sequence `i`; `markers[i]` lists the positions in
    /// sequence `i` whose scorer hidden vector is returned.
    pub fn marker_head_forward(&self, seqs: &[Vec<u32>], qtypes: &[u32], markers: &[Vec<usize>]) -> Result<MarkerHeadOut> {
        let head = self.head.as_ref().ok_or_else(|| anyhow::anyhow!("this text encoder has no marker head"))?;
        ensure!(qtypes.len() == seqs.len() && markers.len() == seqs.len(),
            "marker_head_forward: {} sequences, {} question types, {} marker lists", seqs.len(), qtypes.len(), markers.len());
        for (i, (&q, mk)) in qtypes.iter().zip(markers).enumerate() {
            ensure!((q as usize) < head.n_types, "sequence {i}: question type {q} out of range ({} types)", head.n_types);
            ensure!(mk.iter().all(|&p| p < seqs[i].len()), "sequence {i}: a marker lies past its end");
        }
        let d = self.spec.d as usize;
        let mut a = self.arena(seqs)?;
        let mut keep: Vec<u32> = Vec::new();
        let mut start = 0usize;
        for (s, mk) in seqs.iter().zip(markers) {
            keep.extend(mk.iter().map(|&p| (start + p) as u32));
            start += s.len();
        }
        self.gpu.write_bytes(&mut a.keep, 0, &u32s_bytes(&keep))?;
        let gpu_s = self.run_text(&a, seqs, |a, m| self.encode_marker_head(a, head, seqs, qtypes, keep.len(), m))?;
        let mut scorer_hidden = Vec::with_capacity(seqs.len());
        let mut r = 0usize;
        for mk in markers {
            let mut rows = Vec::with_capacity(mk.len());
            for i in r..r + mk.len() {
                let mut b = vec![0u8; d * 4];
                self.gpu.read_bytes(&a.ffn, i * d * 4, &mut b)?;
                rows.push(bytes_f32(&b));
            }
            scorer_hidden.push(rows);
            r += mk.len();
        }
        Ok(MarkerHeadOut { scorer_hidden, gpu_s })
    }

    /// The marker head on the encoder output in `a.x`: the question-type embedding, the
    /// head's blocks, then the scorer's LayerNorm, first linear and GELU. Only `n_keep`
    /// rows are read afterwards (`a.keep`), so the last block runs its attention over
    /// every row and everything after it on the kept rows alone: they are gathered into
    /// `a.qkv` (residual) and `a.q` (attention output), both free by then, and the
    /// scorer's hidden rows land in `a.ffn`, in `a.keep` order.
    fn encode_marker_head(&self, a: &Arena, head: &Head, seqs: &[Vec<u32>], qtypes: &[u32], n_keep: usize, m: usize) -> Result<()> {
        let d = self.spec.d as usize;
        let mut start = 0usize;
        for (s, &q) in seqs.iter().zip(qtypes) {
            let t = s.len() * d;
            self.k("add_rowbias_m", &[(&a.x, (start * d * 4) as u64), (&head.token_types, (q as usize * d * 4) as u64)],
                   &[d as u32, t as u32], [blocks(t, 256), 1, 1], [256, 1, 1])?;
            start += s.len();
        }
        let hs = &head.spec;
        let (nh, eps, ffn) = (hs.n_head as usize, hs.eps, hs.ffn as usize);
        let last = head.blocks.len() - 1;
        for b in &head.blocks[..last] {
            self.attention_half(a, b, nh, eps, None, false, m)?;
            self.mlp_half(a, b, &a.x, &a.h, eps, ffn, false, ACT_RELU, m)?;
        }
        let b = &head.blocks[last];
        self.attention_half(a, b, nh, eps, None, false, m)?;
        let (xc, hc, k32) = (&a.qkv, &a.q, n_keep as u32);
        for (src, dst) in [(&a.x, xc), (&a.h, hc)] {
            self.k("ple_gather", &[(src, 0), (&a.keep, 0), (dst, 0)], &[d as u32, k32], [blocks(d * n_keep, 256), 1, 1], [256, 1, 1])?;
        }
        self.mlp_half(a, b, xc, hc, eps, ffn, false, ACT_RELU, n_keep)?;
        self.layernorm((xc, 0), (hc, 0), &head.cls_norm.w, head.cls_norm.b.as_ref(), eps, n_keep)?;
        self.mm(a, (hc, 0), &head.cls, (&a.ffn, 0), n_keep, false)?;
        self.bias((&a.ffn, 0), Some(&head.cls_b), d, n_keep)?;
        let t = d * n_keep;
        self.k("act_m", &[(&a.ffn, 0), (&a.ffn, 0)], &[t as u32, ACT_GELU_ERF], [blocks(t, 256), 1, 1], [256, 1, 1])
    }
}
