//! The ModernBERT text encoder and its marker decision head (Laya), as data.
//!
//! Shapes are checked against the tensors here, so a truncated or foreign file fails
//! with a key name rather than inside graph construction.

use anyhow::{anyhow, ensure, Result};
use ojas_formats::gguf::{Gguf, Meta};

/// A ModernBERT text encoder (`general.architecture = "modern-bert"`), read from the
/// same GGUF keys the reference reads. It runs through its own entry on each backend
/// (`ojas-models/src/decoder/text_encoder.rs`, `ojas-cuda/src/bert.rs`), not the decoder
/// graph, because it has no KV cache, no causal mask and no LM head.
#[derive(Clone, Debug)]
pub struct TextEncoderSpec {
    pub d: u32,
    pub layers: u32,
    pub n_head: u32,
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
}

impl TextEncoderSpec {
    pub fn is_local(&self, layer: usize) -> bool {
        self.swa_pattern > 0 && self.window > 0 && !(layer as u32).is_multiple_of(self.swa_pattern)
    }
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
        })
    }
}
