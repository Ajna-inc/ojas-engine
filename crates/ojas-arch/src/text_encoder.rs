//! Bidirectional text encoders and the ModernBERT marker decision head (Laya), as data.
//!
//! Two block layouts share one description. [`Layout::Fused`] is ModernBERT's;
//! [`Layout::Split`] is a decoder-style stack (separate Q/K/V projections with
//! grouped-query heads, RMSNorm, sandwich norms, per-layer inputs) whose file declares
//! `<arch>.attention.causal = false`, such as a Gemma 4 stack trained as an embedder.
//! Which one a file is follows from its tensors, not its name.
//!
//! Shapes are checked against the tensors here, so a truncated or foreign file fails
//! with a key name rather than inside graph construction.

use anyhow::{anyhow, bail, ensure, Result};
use ojas_formats::gguf::{Gguf, Meta};

/// A bidirectional text encoder, read from the same GGUF keys the reference reads. It
/// runs through its own entry on each backend (`ojas-models/src/decoder/text_encoder.rs`,
/// and for the fused layout `ojas-cuda/src/bert.rs`), not the decoder graph, because it
/// has no KV cache, no causal mask and no LM head.
#[derive(Clone, Debug)]
pub struct TextEncoderSpec {
    pub d: u32,
    pub layers: u32,
    pub n_head: u32,
    /// Head width; for a split layout whose layers differ, the widest.
    pub hd: u32,
    /// Per half of the gated MLP: `ffn_up` produces `2 * ffn` columns.
    pub ffn: u32,
    pub eps: f32,
    /// RoPE base of the global-attention layers.
    pub rope_base: f32,
    /// RoPE base of the sliding-window layers.
    pub rope_base_local: f32,
    /// Keys a local layer's query sees on each side: `|i - j| <= window`.
    /// `attention.sliding_window` is the full width (128), this is half of it.
    pub window: u32,
    /// Layer `l` is local when `l % swa_pattern != 0` (the reference's dense-first
    /// rule); 0 means every layer is global.
    pub swa_pattern: u32,
    pub max_positions: u32,
    /// The decision head read at marker tokens, when the file carries one.
    pub marker_head: Option<MarkerHeadSpec>,
    /// How per-token states become one vector per sequence (`<arch>.pooling_type`).
    pub pooling: Pooling,
    /// Linear maps applied to the pooled vector, in order (`dense_2`, then `dense_3`).
    pub dense: Vec<Dense>,
    pub layout: Layout,
}

impl TextEncoderSpec {
    pub fn is_local(&self, layer: usize) -> bool {
        match &self.layout {
            Layout::Split(s) => self.window > 0 && s.layers.get(layer).is_some_and(|l| l.local),
            Layout::Fused => self.swa_pattern > 0 && self.window > 0 && !(layer as u32).is_multiple_of(self.swa_pattern),
        }
    }

    /// Every head width the layers use, ascending and without repeats.
    pub fn head_dims(&self) -> Vec<u32> {
        let mut v: Vec<u32> = match &self.layout {
            Layout::Split(s) => s.layers.iter().map(|l| l.hd).collect(),
            Layout::Fused => vec![self.hd],
        };
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Width of the sentence vector: the last dense map's output, else the hidden width.
    pub fn output_width(&self) -> u32 {
        self.dense.last().map_or(self.d, |m| m.n_out)
    }
}

/// The reference's pooling enum (`<arch>.pooling_type`): 0 none, 1 mean, 2 first token
/// (CLS), 3 last token. Rank pooling (4) belongs to rerankers and is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// The file declares none; per-token states are the output.
    None,
    Mean,
    Cls,
    Last,
}

/// One linear map over the pooled vector, `[n_in] -> [n_out]`, without bias.
#[derive(Clone, Debug)]
pub struct Dense {
    pub tensor: String,
    pub n_in: u32,
    pub n_out: u32,
}

/// The block layout of the encoder.
#[derive(Clone, Debug)]
pub enum Layout {
    /// ModernBERT: one fused `attn_qkv` projection, LayerNorm without bias (none before
    /// layer 0's attention), a gated MLP over one fused `ffn_up` with the first half
    /// activated by erf GELU, uniform heads, and layer `l` local when
    /// `l % swa_pattern != 0`.
    Fused,
    Split(SplitStack),
}

/// A decoder-style stack run bidirectionally: per block
///
/// ```text
/// h = rms(x);  x += post_attn_norm(o_proj(attention(rope(qk_norm(q, k)), v_norm(v))))
/// h = rms(x);  x += post_ffw_norm(down(act(gate(h)) * up(h)))
/// x += post_norm(proj(act(inp_gate(x)) * ple[l]))        (with per-layer inputs)
/// x *= layer_scale[l]
/// ```
///
/// with the embedding scaled by `embed_scale` and, when per-layer inputs are present,
/// `ple = rms(per_layer_model_proj(x0) / sqrt(d))` split into one `ple_width` slice
/// per layer.
#[derive(Clone, Debug)]
pub struct SplitStack {
    pub layers: Vec<SplitLayer>,
    pub embed_scale: f32,
    /// Activation of the MLP gate and of the per-layer input gate.
    pub act: crate::Act,
    /// Weighted per-head RMSNorm on Q and K (`attn_q_norm`, `attn_k_norm`).
    pub qk_norm: bool,
    /// Per-head RMSNorm without a weight on V.
    pub v_norm: bool,
    /// Attention logits are not scaled (Q and K are RMS-normed); otherwise `1/sqrt(hd)`.
    pub unit_scale: bool,
    /// RMSNorm on each sub-block's output before its residual add (`post_attention_norm`,
    /// `post_ffw_norm`).
    pub sandwich: bool,
    /// Width of each layer's per-layer input when the file carries them.
    pub ple_width: Option<u32>,
    /// Each block's output is multiplied by `blk.N.layer_output_scale`.
    pub layer_scale: bool,
}

/// Attention geometry of one split-layout layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitLayer {
    pub n_kv: u32,
    pub hd: u32,
    /// Sliding-window layer: keys within `window` of the query on each side.
    pub local: bool,
    pub rope_base: f32,
}

/// Geometry of a decision head read at marker tokens: the encoder's last `blocks`
/// blocks are PyTorch `nn.TransformerEncoderLayer` blocks (pre-norm, ReLU, biased,
/// no RoPE) over the encoder output plus a per-question-type embedding
/// (`token_types`), followed by a scorer (`cls.*`).
#[derive(Clone, Debug)]
pub struct MarkerHeadSpec {
    /// Index of the first head block; the encoder blocks are the ones before it.
    pub first: u32,
    pub blocks: u32,
    pub n_head: u32,
    pub ffn: u32,
    pub eps: f32,
}

impl TextEncoderSpec {
    /// The `<arch>.*` encoder keys, and the `<arch>.decision.*` keys of a decision
    /// head over the last blocks. Shapes are checked against the tensors here, so a
    /// truncated or foreign file fails with a key name rather than inside graph
    /// construction.
    pub fn from_gguf(g: &Gguf) -> Result<TextEncoderSpec> {
        if g.tensors.contains_key("blk.0.attn_q.weight") && !g.tensors.contains_key("blk.0.attn_qkv.weight") {
            return Self::split_from_gguf(g);
        }
        let arch = g.arch();
        let mu = |k: &str| g.meta_u32(&format!("{arch}.{k}"));
        let total = mu("block_count").unwrap_or(0);
        let head_blocks = mu("decision.block_count").unwrap_or(0);
        ensure!(head_blocks < total, "{arch}: decision.block_count {head_blocks} leaves no encoder blocks");
        let layers = total - head_blocks;
        // One width for every encoder block; the head's blocks have their own.
        let ffn = match g.meta.get(&format!("{arch}.feed_forward_length")) {
            Some(Meta::IntArr(per_block)) => {
                let encoder = &per_block[..(layers as usize).min(per_block.len())];
                ensure!(!encoder.is_empty() && encoder.iter().all(|&f| f == encoder[0]),
                    "{arch}: the encoder blocks do not share one feed_forward_length");
                encoder[0] as u32
            }
            _ => mu("feed_forward_length").unwrap_or(0),
        };
        let (d, n_head) = (mu("embedding_length").unwrap_or(0), mu("attention.head_count").unwrap_or(0));
        ensure!(d > 0 && layers > 0 && n_head > 0 && ffn > 0 && d % n_head == 0,
            "{arch}: incomplete encoder metadata (embd={d} blocks={layers} heads={n_head} ffn={ffn})");
        let hd = d / n_head;
        // The bidirectional attention kernels keep a head in 16 accumulators per lane
        // across a 32-lane simdgroup: hd % 32 == 0 and hd <= 512.
        ensure!(hd % 32 == 0 && hd <= 512,
            "{arch}: head_dim {hd} is not supported by the bidirectional attention kernels");
        let act = match g.meta.get(&format!("{arch}.hidden_activation")) {
            Some(Meta::Str(s)) => s.clone(),
            _ => "gelu".to_string(),
        };
        ensure!(act == "gelu", "{arch}: hidden_activation {act:?} is not implemented (only \"gelu\", the erf form)");
        let eps = g.meta_f32(&format!("{arch}.attention.layer_norm_epsilon")).unwrap_or(1e-5);
        let rope_base = g.meta_f32(&format!("{arch}.rope.freq_base")).unwrap_or(160000.0);
        let rope_base_local = g.meta_f32(&format!("{arch}.rope.freq_base_swa")).unwrap_or(rope_base);
        let sliding = mu("attention.sliding_window").unwrap_or(0);
        let swa_pattern = if sliding > 0 { mu("attention.sliding_window_pattern").unwrap_or(3) } else { 0 };
        let max_positions = mu("context_length").unwrap_or(8192);
        for (name, want) in [("blk.0.attn_qkv.weight", [d as u64, 3 * d as u64]),
                             ("blk.0.ffn_up.weight", [d as u64, 2 * ffn as u64]),
                             ("blk.0.ffn_down.weight", [ffn as u64, d as u64])] {
            let t = g.tensors.get(name).ok_or_else(|| anyhow!("{arch}: missing {name}"))?;
            ensure!(t.dims == want, "{arch}: {name} is {:?}, expected {want:?}", t.dims);
        }
        let marker_head = if head_blocks == 0 {
            None
        } else {
            let up = format!("blk.{layers}.ffn_up.weight");
            let t = g.tensors.get(&up).ok_or_else(|| anyhow!("decision head: missing {up}"))?;
            ensure!(t.dims.len() == 2 && t.dims[0] == d as u64, "decision head: {up} is {:?}", t.dims);
            let head = MarkerHeadSpec { first: layers, blocks: head_blocks, n_head, ffn: t.dims[1] as u32, eps };
            for i in layers..total {
                let name = format!("blk.{i}.ffn_up.weight");
                let t = g.tensors.get(&name).ok_or_else(|| anyhow!("decision head: missing {name}"))?;
                ensure!(t.dims == [d as u64, head.ffn as u64], "decision head: {name} is {:?}", t.dims);
            }
            for name in ["token_types.weight", "cls.norm.weight", "cls.weight", "cls.output.weight"] {
                ensure!(g.tensors.contains_key(name), "decision head: missing {name}");
            }
            Some(head)
        };
        Ok(TextEncoderSpec {
            d, layers, n_head, hd, ffn, eps, rope_base, rope_base_local,
            window: sliding / 2, swa_pattern, max_positions, marker_head,
            pooling: pooling(g)?, dense: dense(g, d)?, layout: Layout::Fused,
        })
    }

    /// The split layout. Per-layer geometry uses the Gemma 4 keys: `head_count_kv` and
    /// `sliding_window_pattern` per layer (an array, or one value for every layer),
    /// `key_length` / `key_length_swa` and `rope.freq_base` / `rope.freq_base_swa` for
    /// the global and the sliding layers. `attention.sliding_window` is the full
    /// symmetric width of a local layer, as for ModernBERT: the reach on each side is
    /// half of it.
    fn split_from_gguf(g: &Gguf) -> Result<TextEncoderSpec> {
        let arch = g.arch();
        let key = |k: &str| format!("{arch}.{k}");
        let mu = |k: &str| g.meta_u32(&key(k));
        let has = |t: &str| g.tensors.contains_key(t);
        let dims = |t: &str| g.tensors.get(t).map(|i| i.dims.clone()).ok_or_else(|| anyhow!("{arch}: missing {t}"));
        let layers = mu("block_count").unwrap_or(0);
        let (d, n_head) = (mu("embedding_length").unwrap_or(0), mu("attention.head_count").unwrap_or(0));
        let ffn = match g.meta.get(&key("feed_forward_length")) {
            Some(Meta::IntArr(per)) => {
                ensure!(!per.is_empty() && per.iter().all(|&f| f == per[0]), "{arch}: blocks of different MLP widths are not implemented");
                per[0] as u32
            }
            _ => mu("feed_forward_length").unwrap_or(0),
        };
        ensure!(d > 0 && layers > 0 && n_head > 0 && ffn > 0,
            "{arch}: incomplete encoder metadata (embd={d} blocks={layers} heads={n_head} ffn={ffn})");
        // Features of a decoder stack the encoder does not implement; each would be
        // silently wrong if ignored.
        ensure!(mu("expert_count").unwrap_or(0) == 0, "{arch}: a mixture-of-experts encoder is not implemented");
        ensure!(mu("attention.shared_kv_layers").unwrap_or(0) == 0, "{arch}: shared KV layers are not implemented in the encoder");
        ensure!(g.meta_f32(&key("attn_logit_softcapping")).is_none(), "{arch}: attention logit soft-capping is not implemented");
        for t in ["rope_freqs.weight", "per_layer_token_embd.weight", "blk.0.laurel_l.weight"] {
            ensure!(!has(t), "{arch}: {t} is not implemented in the encoder");
        }
        let per_layer = |k: &str, default: u32| -> Vec<u32> {
            match g.meta.get(&key(k)) {
                Some(Meta::IntArr(a)) => (0..layers as usize).map(|l| a.get(l).map_or(default, |&v| v as u32)).collect(),
                Some(_) => vec![mu(k).unwrap_or(default); layers as usize],
                None => vec![default; layers as usize],
            }
        };
        let n_kv = per_layer("attention.head_count_kv", n_head);
        let sliding = mu("attention.sliding_window").unwrap_or(0);
        // A per-layer array marks the sliding layers; without one, no layer slides.
        let local: Vec<bool> = match g.meta.get(&key("attention.sliding_window_pattern")) {
            Some(Meta::IntArr(a)) => (0..layers as usize).map(|l| a.get(l).is_some_and(|&v| v != 0) && sliding > 0).collect(),
            _ => vec![false; layers as usize],
        };
        let hd_full = mu("attention.key_length").unwrap_or(d / n_head);
        let hd_swa = mu("attention.key_length_swa").unwrap_or(hd_full);
        let rope_full = g.meta_f32(&key("rope.freq_base")).unwrap_or(10000.0);
        let rope_swa = g.meta_f32(&key("rope.freq_base_swa")).unwrap_or(rope_full);
        let rot_full = mu("rope.dimension_count").unwrap_or(hd_full);
        let rot_swa = mu("rope.dimension_count_swa").unwrap_or(hd_swa);
        ensure!(rot_full == hd_full && rot_swa == hd_swa,
            "{arch}: partial rotary embedding ({rot_full}/{hd_full}, {rot_swa}/{hd_swa}) is not implemented in the encoder");
        let eps = g.meta_f32(&key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
        let ple = mu("embedding_length_per_layer_input").unwrap_or(0);
        let ple_width = (ple > 0 && has("per_layer_model_proj.weight")).then_some(ple);
        let spec_layers: Vec<SplitLayer> = (0..layers as usize).map(|l| {
            let hd = if local[l] { hd_swa } else { hd_full };
            SplitLayer { n_kv: n_kv[l], hd, local: local[l], rope_base: if local[l] { rope_swa } else { rope_full } }
        }).collect();
        let sandwich = has("blk.0.post_attention_norm.weight") && has("blk.0.post_ffw_norm.weight");
        let qk_norm = has("blk.0.attn_q_norm.weight");
        for (l, sl) in spec_layers.iter().enumerate() {
            let (nkv, hd) = (sl.n_kv as u64, sl.hd as u64);
            // The bidirectional attention kernels keep a head in 16 accumulators per lane
            // across a 32-lane simdgroup.
            ensure!(hd % 32 == 0 && hd <= 512, "{arch}: layer {l} head_dim {hd} is not supported by the attention kernels");
            ensure!(nkv > 0 && (n_head as u64).is_multiple_of(nkv), "{arch}: layer {l} has {n_head} heads over {nkv} KV heads");
            let p = |t: &str| format!("blk.{l}.{t}.weight");
            let (d, nh, f) = (d as u64, n_head as u64, ffn as u64);
            for (t, want) in [("attn_q", [d, nh * hd]), ("attn_k", [d, nkv * hd]), ("attn_v", [d, nkv * hd]),
                              ("attn_output", [nh * hd, d]), ("ffn_gate", [d, f]), ("ffn_up", [d, f]), ("ffn_down", [f, d])] {
                let got = dims(&p(t))?;
                ensure!(got == want, "{arch}: {} is {got:?}, expected {want:?}", p(t));
            }
            let mut norms = vec![p("attn_norm"), p("ffn_norm")];
            if qk_norm { norms.extend([p("attn_q_norm"), p("attn_k_norm")]); }
            if sandwich { norms.extend([p("post_attention_norm"), p("post_ffw_norm")]); }
            if let Some(w) = ple_width {
                ensure!(dims(&p("inp_gate"))? == [d, w as u64] && dims(&p("proj"))? == [w as u64, d],
                    "{arch}: layer {l} per-layer input projections do not match width {w}");
                norms.push(p("post_norm"));
            }
            for n in norms { dims(&n)?; }
        }
        if let Some(w) = ple_width {
            let want = [d as u64, layers as u64 * w as u64];
            let got = dims("per_layer_model_proj.weight")?;
            ensure!(got == want, "{arch}: per_layer_model_proj.weight is {got:?}, expected {want:?}");
            dims("per_layer_proj_norm.weight")?;
        }
        dims("output_norm.weight")?;
        // Gemma 4 normalizes V without a weight and attends at scale 1 (its Q and K are
        // RMS-normed); no tensor carries either, so they follow from the architecture,
        // as they do for the decoder (`ojas-models/src/decoder/load.rs`).
        let gemma4 = arch == "gemma4";
        let family = crate::ArchSpec::from_gguf(g)?;
        let hd = spec_layers.iter().map(|l| l.hd).max().unwrap_or(hd_full);
        Ok(TextEncoderSpec {
            d, layers, n_head, hd, ffn, eps, rope_base: rope_full, rope_base_local: rope_swa,
            window: sliding / 2, swa_pattern: 0, max_positions: mu("context_length").unwrap_or(8192),
            marker_head: None, pooling: pooling(g)?, dense: dense(g, d)?,
            layout: Layout::Split(SplitStack {
                layers: spec_layers, embed_scale: family.embed_scale, act: family.act, qk_norm,
                v_norm: gemma4, unit_scale: gemma4, sandwich, ple_width,
                layer_scale: has("blk.0.layer_output_scale.weight"),
            }),
        })
    }
}

/// `<arch>.pooling_type`, absent meaning none.
fn pooling(g: &Gguf) -> Result<Pooling> {
    let arch = g.arch();
    let value = match g.meta.get(&format!("{arch}.pooling_type")) {
        None => return Ok(Pooling::None),
        Some(Meta::U32(v)) => *v as i64,
        Some(Meta::I32(v)) => *v as i64,
        Some(other) => bail!("{arch}: pooling_type {other:?} is not an integer"),
    };
    Ok(match value {
        -1 | 0 => Pooling::None,
        1 => Pooling::Mean,
        2 => Pooling::Cls,
        3 => Pooling::Last,
        other => bail!("{arch}: pooling_type {other} is not implemented"),
    })
}

/// The sentence-transformers dense maps (`dense_2`, then `dense_3`), each checked to
/// take the previous width.
fn dense(g: &Gguf, d: u32) -> Result<Vec<Dense>> {
    let mut maps = Vec::new();
    let mut width = d;
    for name in ["dense_2.weight", "dense_3.weight"] {
        let Some(t) = g.tensors.get(name) else { continue };
        ensure!(t.dims.len() == 2 && t.dims[0] == width as u64, "{name} is {:?}, expected [{width}, n]", t.dims);
        maps.push(Dense { tensor: name.to_string(), n_in: width, n_out: t.dims[1] as u32 });
        width = t.dims[1] as u32;
    }
    Ok(maps)
}
