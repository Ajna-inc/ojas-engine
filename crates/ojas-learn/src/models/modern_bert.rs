//! The ModernBERT text encoder with its marker decision head (Laya), on the tape.
//!
//! The same model the decision server runs (`ojas-models/src/decoder/text_encoder.rs`
//! on Metal, `ojas-cuda/src/bert.rs` on CUDA), built from the tape's primitives so
//! every weight receives a gradient. The shape is
//! [`ojas_arch::text_encoder::TextEncoderSpec`]; weights load from the GGUF the server
//! reads, as f32, keeping the file's tensor names:
//!
//! * the encoder: token embedding and its LayerNorm, then pre-norm blocks (layer 0 has
//!   no attention norm), NeoX RoPE restarting at 0 per sequence, a symmetric window on
//!   the local layers (`|i - j| <= window`), a gated erf-GELU MLP over one fused
//!   `ffn_up` (first half activated), and the final LayerNorm. No biases.
//! * the head: the question type's row of `token_types` added to every row, then
//!   PyTorch `nn.TransformerEncoderLayer` blocks (pre-norm, biased, ReLU, no RoPE), and
//!   the scorer at the option markers: LayerNorm, `cls.weight`, erf GELU, `cls.output`.
//!
//! [`LearnDecision`] serves the model through `ojas_decision`, so the decision
//! server's prompts, calibration and parity suite run on it unchanged.

use crate::backend::Backend;
use crate::tape::{Param, Tape, Var};
use anyhow::{anyhow, bail, ensure, Context, Result};
use ojas_arch::text_encoder::TextEncoderSpec;
use ojas_decision::{CausalBackend, CausalLoad, DecisionGpu, MarkerBackend, MarkerHeadOut, SlotPrefill};
use ojas_formats::gguf::Gguf;
use std::collections::HashMap;

/// Added to the score of a key outside a local layer's window: zero after the softmax.
const MASKED: f32 = -1e9;

/// One sequence for [`ModernBert::scores`]: its tokens, its question type (the row of
/// `token_types`) and the positions of its option markers.
#[derive(Clone, Debug)]
pub struct MarkerSeq {
    pub ids: Vec<u32>,
    pub qtype: u32,
    pub markers: Vec<usize>,
}

pub struct ModernBert<B: Backend> {
    pub spec: TextEncoderSpec,
    params: Vec<Param<B>>,
    index: HashMap<String, usize>,
}

impl<B: Backend> ModernBert<B> {
    /// Every tensor of a ModernBERT GGUF with a marker head, as f32 parameters.
    pub fn from_gguf(be: &B, g: &mut Gguf) -> Result<Self> {
        let spec = TextEncoderSpec::from_gguf(g)?;
        ensure!(spec.marker_head.is_some(), "{}: no decision head (decision.block_count)", g.arch());
        let mut names: Vec<String> = g.tensors.keys().cloned().collect();
        names.sort();
        let (mut params, mut index) = (Vec::with_capacity(names.len()), HashMap::new());
        for name in names {
            let (dims, ty, bytes) = g.read_tensor(&name)?;
            let data: Vec<f32> = match ty {
                0 => bytes.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
                1 => bytes.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c).to_f32()).collect(),
                t => bail!("{name}: tensor type {t} did not decode to f16 or f32"),
            };
            // GGUF lists the fastest axis first; the tape is row-major, slowest first,
            // so a [in, out] matrix becomes a PyTorch Linear weight [out, in].
            let shape: Vec<usize> = dims.iter().rev().map(|&d| d as usize).collect();
            index.insert(name.clone(), params.len());
            params.push(Param::new(be, name, &shape, &data));
        }
        Ok(ModernBert { spec, params, index })
    }

    pub fn params(&self) -> &[Param<B>] { &self.params }

    /// Add `n` encoder blocks after the last one, before the decision head, and return
    /// their name prefixes. Each new block starts as a copy of the nearest existing
    /// block with the same local or global attention (which the block's index fixes),
    /// with its attention output and down projections zeroed, so the residual stream
    /// passes through unchanged and the grown model answers exactly as before. The
    /// head's blocks are renumbered after the new ones.
    pub fn grow(&mut self, be: &B, n: usize) -> Result<Vec<String>> {
        let layers = self.spec.layers as usize;
        let head = self.spec.marker_head.clone().context("no decision head")?;
        let rename = |name: &str| -> String {
            for i in head.first as usize..(head.first + head.blocks) as usize {
                if let Some(rest) = name.strip_prefix(&format!("blk.{i}.")) {
                    return format!("blk.{}.{rest}", i + n);
                }
            }
            name.to_string()
        };
        for p in &mut self.params { p.name = rename(&p.name); }
        let mut added = Vec::with_capacity(n);
        for j in 0..n {
            let local = self.spec.is_local(layers + j);
            let source = (0..layers).rev().find(|&l| self.spec.is_local(l) == local).unwrap_or(layers - 1);
            let src = format!("blk.{source}.");
            let dst = format!("blk.{}.", layers + j);
            let copies: Vec<(String, Vec<usize>, Vec<f32>)> = self.params.iter().filter(|p| p.name.starts_with(&src)).map(|p| {
                let name = format!("{dst}{}", &p.name[src.len()..]);
                let zero = name.ends_with("attn_output.weight") || name.ends_with("ffn_down.weight");
                let data = if zero { vec![0.0; p.shape.iter().product()] } else { be.download(&p.val) };
                (name, p.shape.clone(), data)
            }).collect();
            ensure!(!copies.is_empty(), "no tensors under {src}");
            for (name, shape, data) in copies { self.params.push(Param::new(be, name, &shape, &data)); }
            added.push(dst);
        }
        self.spec.layers += n as u32;
        if let Some(h) = self.spec.marker_head.as_mut() { h.first += n as u32; }
        self.index = self.params.iter().enumerate().map(|(i, p)| (p.name.clone(), i)).collect();
        Ok(added)
    }

    pub fn param(&self, name: &str) -> Result<&Param<B>> {
        self.index.get(name).map(|&i| &self.params[i]).ok_or_else(|| anyhow!("no tensor {name}"))
    }

    fn has(&self, name: &str) -> bool { self.index.contains_key(name) }

    fn p(&self, t: &mut Tape<'_, B>, name: &str) -> Result<Var> { Ok(t.param(self.param(name)?)) }

    /// The scorer's hidden vector (after its GELU) at every marker of every sequence,
    /// `[markers, d]` in sequence order.
    pub fn scorer_hidden(&self, t: &mut Tape<'_, B>, seqs: &[MarkerSeq]) -> Result<Var> {
        let mut rows = Vec::with_capacity(seqs.len());
        for s in seqs {
            ensure!(!s.ids.is_empty() && !s.markers.is_empty(), "a sequence needs tokens and markers");
            ensure!(s.markers.iter().all(|&m| m < s.ids.len()), "a marker lies past its sequence");
            let x = self.encode(t, &s.ids)?;
            let x = self.head(t, x, s.qtype)?;
            rows.push(self.scorer(t, x, &s.markers)?);
        }
        if rows.len() == 1 { Ok(rows[0]) } else { t.concat(&rows, 0) }
    }

    /// Each marker's score, `[markers]`: `cls.output` over [`Self::scorer_hidden`].
    pub fn scores(&self, t: &mut Tape<'_, B>, seqs: &[MarkerSeq]) -> Result<Var> {
        let h = self.scorer_hidden(t, seqs)?;
        let w = self.p(t, "cls.output.weight")?;
        let w = t.reshape(w, &[1, self.spec.d as usize])?;
        let b = self.p(t, "cls.output.bias")?;
        let y = t.linear(h, w, Some(b))?;
        let n = t.shape(y)[0];
        t.reshape(y, &[n])
    }

    /// The encoder over one sequence: `[T, d]` after the final LayerNorm.
    fn encode(&self, t: &mut Tape<'_, B>, ids: &[u32]) -> Result<Var> {
        let sp = &self.spec;
        let (n, d) = (ids.len(), sp.d as usize);
        let emb = self.p(t, "token_embd.weight")?;
        let vocab = t.shape(emb)[0];
        ensure!(ids.iter().all(|&i| (i as usize) < vocab), "a token id is outside the {vocab}-token vocabulary");
        let emb = t.reshape(emb, &[1, vocab, d])?;
        let idx: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
        let x = t.gather_rows(emb, &idx, n)?;
        let x = t.reshape(x, &[n, d])?;
        let mut x = self.norm(t, x, "token_embd_norm", false)?;

        let tables = [false, true].map(|local| {
            let base = if local { sp.rope_base_local } else { sp.rope_base };
            rope_tables(n, sp.hd as usize, base)
        });
        let window = local_mask(n, sp.window as usize);
        for l in 0..sp.layers as usize {
            let local = sp.is_local(l);
            let pre = format!("blk.{l}");
            let a = if self.has(&format!("{pre}.attn_norm.weight")) {
                self.norm(t, x, &format!("{pre}.attn_norm"), false)?
            } else {
                x
            };
            let (cos, sin) = &tables[local as usize];
            let rope = Some((t.input(cos, &[n, sp.hd as usize]), t.input(sin, &[n, sp.hd as usize])));
            let mask = if local { Some(t.input(&window, &[n, n])) } else { None };
            let o = self.attention(t, a, &pre, sp.n_head as usize, false, rope, mask)?;
            x = t.add(x, o)?;
            let f = self.norm(t, x, &format!("{pre}.ffn_norm"), false)?;
            let up = self.linear(t, f, &format!("{pre}.ffn_up"), false)?;
            let ffn = sp.ffn as usize;
            let (g, u) = (t.slice(up, 1, 0, ffn)?, t.slice(up, 1, ffn, 2 * ffn)?);
            let g = t.gelu(g);
            let h = t.mul(g, u)?;
            let down = self.linear(t, h, &format!("{pre}.ffn_down"), false)?;
            x = t.add(x, down)?;
        }
        self.norm(t, x, "output_norm", false)
    }

    /// The question type's embedding and the head's blocks over one sequence.
    fn head(&self, t: &mut Tape<'_, B>, x: Var, qtype: u32) -> Result<Var> {
        let head = self.spec.marker_head.as_ref().expect("checked at load");
        let types = self.p(t, "token_types.weight")?;
        let n_types = t.shape(types)[0];
        ensure!((qtype as usize) < n_types, "question type {qtype} past the {n_types} type embeddings");
        let row = t.slice(types, 0, qtype as usize, qtype as usize + 1)?;
        let mut x = t.add(x, row)?;
        for i in head.first..head.first + head.blocks {
            let pre = format!("blk.{i}");
            let a = self.norm(t, x, &format!("{pre}.attn_norm"), true)?;
            let o = self.attention(t, a, &pre, head.n_head as usize, true, None, None)?;
            x = t.add(x, o)?;
            let f = self.norm(t, x, &format!("{pre}.ffn_norm"), true)?;
            let up = self.linear(t, f, &format!("{pre}.ffn_up"), true)?;
            let up = t.relu(up);
            let down = self.linear(t, up, &format!("{pre}.ffn_down"), true)?;
            x = t.add(x, down)?;
        }
        Ok(x)
    }

    /// The scorer's LayerNorm, first linear and GELU at the marker rows: `[markers, d]`.
    fn scorer(&self, t: &mut Tape<'_, B>, x: Var, markers: &[usize]) -> Result<Var> {
        let s = t.shape(x).to_vec();
        let x = t.reshape(x, &[1, s[0], s[1]])?;
        let h = t.gather_rows(x, markers, markers.len())?;
        let h = t.reshape(h, &[markers.len(), s[1]])?;
        let h = self.norm(t, h, "cls.norm", true)?;
        let w = self.p(t, "cls.weight")?;
        let b = self.p(t, "cls.bias")?;
        let h = t.linear(h, w, Some(b))?;
        Ok(t.gelu(h))
    }

    /// Bidirectional multi-head attention over `x` `[T, d]` through the block's fused
    /// QKV and output projections: `[T, d]`, before the residual add.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, t: &mut Tape<'_, B>, x: Var, pre: &str, n_head: usize, bias: bool,
                 rope: Option<(Var, Var)>, mask: Option<Var>) -> Result<Var> {
        let (n, d) = (t.shape(x)[0], self.spec.d as usize);
        let hd = d / n_head;
        let qkv = self.linear(t, x, &format!("{pre}.attn_qkv"), bias)?;
        let mut heads = Vec::with_capacity(3);
        for i in 0..3 {
            let part = t.slice(qkv, 1, i * d, (i + 1) * d)?;
            let part = t.reshape(part, &[n, n_head, hd])?;
            heads.push(t.permute(part, &[1, 0, 2])?);
        }
        let (mut q, mut k, v) = (heads[0], heads[1], heads[2]);
        if let Some((cos, sin)) = rope {
            q = rotate(t, q, cos, sin)?;
            k = rotate(t, k, cos, sin)?;
        }
        let s = t.matmul_opts(q, k, true)?;
        let mut s = t.scale(s, 1.0 / (hd as f32).sqrt())?;
        if let Some(m) = mask {
            s = t.add(s, m)?;
        }
        let p = t.softmax(s);
        let o = t.matmul(p, v)?;
        let o = t.permute(o, &[1, 0, 2])?;
        let o = t.reshape(o, &[n, d])?;
        self.linear(t, o, &format!("{pre}.attn_output"), bias)
    }

    fn linear(&self, t: &mut Tape<'_, B>, x: Var, name: &str, bias: bool) -> Result<Var> {
        let w = self.p(t, &format!("{name}.weight"))?;
        let b = if bias { Some(self.p(t, &format!("{name}.bias"))?) } else { None };
        t.linear(x, w, b)
    }

    /// LayerNorm over the last axis; ModernBERT's own norms carry no bias.
    fn norm(&self, t: &mut Tape<'_, B>, x: Var, name: &str, bias: bool) -> Result<Var> {
        let g = self.p(t, &format!("{name}.weight"))?;
        let b = if bias {
            self.p(t, &format!("{name}.bias"))?
        } else {
            let cols = *t.shape(x).last().unwrap();
            t.input(&vec![0.0; cols], &[cols])
        };
        let eps = self.spec.marker_head.as_ref().filter(|_| bias).map_or(self.spec.eps, |h| h.eps);
        Ok(t.layer_norm(x, g, b, eps))
    }
}

/// NeoX rotation of `x` `[H, T, hd]`: pair `(j, j + hd/2)` of every head turns by the
/// angle in `cos` / `sin` `[T, hd]`, whose columns `j` and `j + hd/2` hold the same angle.
fn rotate<B: Backend>(t: &mut Tape<'_, B>, x: Var, cos: Var, sin: Var) -> Result<Var> {
    let hd = *t.shape(x).last().unwrap();
    let (lo, hi) = (t.slice(x, 2, 0, hd / 2)?, t.slice(x, 2, hd / 2, hd)?);
    let neg = t.scale(hi, -1.0)?;
    let turned = t.concat(&[neg, lo], 2)?;
    let a = t.mul(x, cos)?;
    let b = t.mul(turned, sin)?;
    t.add(a, b)
}

/// `cos` and `sin` of every position's angles, `[T, hd]` each: column `j` and
/// `j + hd/2` hold `pos * base^(-2j/hd)`; positions start at 0.
fn rope_tables(n: usize, hd: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let (mut cos, mut sin) = (vec![0.0; n * hd], vec![0.0; n * hd]);
    for pos in 0..n {
        for j in 0..half {
            let theta = pos as f64 * (base as f64).powf(-2.0 * j as f64 / hd as f64);
            let (s, c) = theta.sin_cos();
            for col in [j, j + half] {
                cos[pos * hd + col] = c as f32;
                sin[pos * hd + col] = s as f32;
            }
        }
    }
    (cos, sin)
}

/// The score bias of a local layer, `[T, T]`: 0 where `|i - j| <= window`, else masked.
fn local_mask(n: usize, window: usize) -> Vec<f32> {
    let mut m = vec![MASKED; n * n];
    for i in 0..n {
        for j in i.saturating_sub(window)..(i + window + 1).min(n) {
            m[i * n + j] = 0.0;
        }
    }
    m
}

/// A training backend as a decision GPU: marker models (Laya) load as [`ModernBert`]
/// and answer through `ojas_decision`'s own prompts and calibration, optionally grown
/// by encoder blocks as they load ([`ModernBert::grow`]). Causal models are not trained
/// here and are refused.
pub struct LearnDecision<'b, B: Backend> {
    be: &'b B,
    grow: usize,
}

impl<'b, B: Backend> LearnDecision<'b, B> {
    pub fn new(be: &'b B) -> Self { LearnDecision { be, grow: 0 } }

    /// Every marker model loaded gains `blocks` encoder blocks.
    pub fn grown(be: &'b B, blocks: usize) -> Self { LearnDecision { be, grow: blocks } }
}

/// A loaded [`ModernBert`] answering on its backend, one tape per pass.
pub struct LearnMarker<'b, B: Backend> {
    be: &'b B,
    pub model: ModernBert<B>,
}

impl<B: Backend> MarkerBackend for LearnMarker<'_, B> {
    fn width(&self) -> usize { self.model.spec.d as usize }

    fn max_positions(&self) -> usize { self.model.spec.max_positions as usize }

    fn marker_head_forward(&self, seqs: &[Vec<u32>], qtypes: &[u32], markers: &[Vec<usize>]) -> Result<MarkerHeadOut> {
        let start = std::time::Instant::now();
        let d = self.width();
        let mut scorer_hidden = Vec::with_capacity(seqs.len());
        for ((ids, &qtype), markers) in seqs.iter().zip(qtypes).zip(markers) {
            let mut t = Tape::new(self.be);
            let seq = MarkerSeq { ids: ids.clone(), qtype, markers: markers.clone() };
            let h = self.model.scorer_hidden(&mut t, std::slice::from_ref(&seq))?;
            scorer_hidden.push(t.value(h).chunks(d).map(<[f32]>::to_vec).collect());
        }
        Ok(MarkerHeadOut { scorer_hidden, gpu_s: start.elapsed().as_secs_f64() })
    }
}

/// No causal model loads on a training backend.
pub enum NoCausal {}

impl CausalBackend for NoCausal {
    fn width(&self) -> usize { match *self {} }
    fn slots(&self) -> usize { match *self {} }
    fn vision_patch_merge(&self) -> Option<(usize, usize)> { match *self {} }
    fn vision_grid(&self, _: usize, _: usize) -> Option<(usize, usize)> { match *self {} }
    fn encode_image(&self, _: &[f32], _: usize, _: usize) -> Result<Vec<f32>> { match *self {} }
    fn reset_session(&self) { match *self {} }
    fn reset_slot(&self, _: usize) { match *self {} }
    fn save_state(&self) { match *self {} }
    fn restore_state(&self) { match *self {} }
    fn copy_slot_prefix(&self, _: usize, _: usize, _: usize) { match *self {} }
    fn prefill_hidden_slots(&self, _: &[SlotPrefill]) -> Vec<Vec<Vec<f32>>> { match *self {} }
    fn gpu_seconds(&self) -> f64 { match *self {} }
}

impl<'b, B: Backend> DecisionGpu for LearnDecision<'b, B> {
    type Marker<'a> = LearnMarker<'b, B> where Self: 'a;
    type Causal<'a> = NoCausal where Self: 'a;
    const NAME: &'static str = "learn";

    fn load_marker<'a>(&'a self, g: &mut Gguf) -> Result<Self::Marker<'a>> {
        let mut model = ModernBert::from_gguf(self.be, g)?;
        if self.grow > 0 { model.grow(self.be, self.grow)?; }
        Ok(LearnMarker { be: self.be, model })
    }

    fn load_causal<'a>(&'a self, g: &mut Gguf, _: CausalLoad) -> Result<Self::Causal<'a>> {
        bail!("{}: causal decision models are not trained on this backend", g.arch())
    }
}
