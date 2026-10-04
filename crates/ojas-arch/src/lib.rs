//! What a model *is*, separately from what runs it.
//!
//! `ojas-models` holds the same description as pure data (`decoder::Arch`, `LayerPlan`,
//! `MoeConfig`), but that crate calls `metal::` in 21 files, so it only builds on macOS
//! and the CUDA side re-derived the same metadata keys by hand in `decode_fast`. This
//! crate builds everywhere and knows about no device, encoder or buffer: it reads GGUF
//! metadata and tensor names and answers what shape a model is and which features its
//! graph needs.
//!
//! It does not plan execution — no kernel names, threadgroup sizes or weight layout —
//! and does not yet cover MoE, SSM, MLA or the vision tower. Those configs live in
//! `ojas-models` and move here as each gains a second backend; the ModernBERT text
//! encoder ([`text_encoder`]) has.
//!
//! The semantics mirror `ojas-models/src/decoder/load.rs` key for key: two readers of the
//! same file that disagree are worse than one reader in the wrong crate.

pub mod text_encoder;

use anyhow::{anyhow, bail, Result};
use ojas_formats::gguf::Gguf;

/// Per-layer shape. Most models repeat one of these; it is per-layer because of Gemma's
/// dual RoPE and KV sharing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerSpec {
    pub n_head: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    /// `n_head * head_dim`
    pub qdim: u32,
    /// `n_kv * head_dim`
    pub kvdim: u32,
    /// Per-layer, because Gemma 3 alternates a local and a global base.
    pub rope_base: f32,
    /// `1 / sqrt(head_dim)`
    pub scale: f32,
}

/// Which activation the FFN uses. Both are gated (gate ⊙ up); only the nonlinearity differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Silu,
    Gelu,
}

impl Act {
    /// The `act` constant the `ffn_gu_*` kernels take — 1 is GeLU on both backends.
    pub fn kernel_code(self) -> u32 {
        match self {
            Act::Silu => 0,
            Act::Gelu => 1,
        }
    }
}

/// A dense decoder's shape and the optional features its graph has to honour.
///
/// Every flag here changes the arithmetic. A backend that cannot implement one must refuse
/// rather than ignore it: skipping Gemma's embedding scale or Qwen3's q/k norm produces
/// fluent-looking output from the wrong model, with no error to notice.
#[derive(Clone, Debug, PartialEq)]
pub struct ArchSpec {
    /// The GGUF `general.architecture` string, verbatim.
    pub arch: String,
    /// Main decoder blocks: `block_count` minus any NextN/MTP prediction layers. A Qwen3.5
    /// MTP GGUF reports 33 = 32 main + 1 draft at `blk.32`, and running 33 would evaluate
    /// the draft head as part of the stack.
    pub n_layers: usize,
    /// `block_count` verbatim, before the MTP subtraction — the loader that attaches a draft
    /// sidecar needs the raw value.
    pub block_count: usize,
    /// `nextn_predict_layers`: how many of `block_count` are MTP draft blocks.
    pub n_nextn: usize,
    /// Hidden size (`embedding_length`).
    pub d: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub eps: f32,
    pub layers: Vec<LayerSpec>,
    /// `token_embd.weight` when the head is tied, `output.weight` when it is not.
    pub lm_head: String,
    /// Qwen2 ships q/k/v bias; Llama, Qwen3 and Gemma do not.
    pub qkv_bias: bool,
    /// Qwen3 / Gemma: per-head RMSNorm on q and k before RoPE.
    pub qk_norm: bool,
    /// true = NeoX split-half rotation; false = interleaved pairs (Llama's permuted weights).
    pub rope_neox: bool,
    /// Gemma multiplies the gathered embedding by sqrt(d); everyone else by 1.
    pub embed_scale: f32,
    /// Gemma: an extra RMSNorm on the attention and FFN outputs, before the residual add.
    pub sandwich: bool,
    pub act: Act,
    /// Set when the file declares experts. MoE itself is not described here yet; this flag
    /// lets a dense-only backend refuse instead of running the shared FFN as if it were the
    /// whole model.
    pub n_experts: u32,
    /// Set for architectures whose graph has recurrent layers, MLA, or a vision tower, so a
    /// dense runner knows to refuse.
    pub non_dense: Option<&'static str>,
}

impl ArchSpec {
    /// Read the architecture from an open GGUF.
    ///
    /// Optional features are detected from tensor presence rather than the architecture
    /// string, so one code path serves Qwen2/2.5/3, Llama and Gemma without a table of model
    /// names. `load.rs` does the same.
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.arch();
        let key = |k: &str| format!("{arch}.{k}");
        let block_count = g
            .meta_u32(&key("block_count"))
            .ok_or_else(|| anyhow!("{arch}: no block_count"))? as usize;
        let n_nextn = g.meta_u32(&key("nextn_predict_layers")).unwrap_or(0) as usize;
        let n_layers = block_count.saturating_sub(n_nextn);
        let d = g
            .meta_u32(&key("embedding_length"))
            .ok_or_else(|| anyhow!("{arch}: no embedding_length"))? as usize;
        let n_head = g.meta_u32(&key("attention.head_count")).unwrap_or(1);
        // `head_count_kv` may be a per-layer array (Gemma 4). The model-level number is the
        // max, which is what scratch and bias sizing need; a per-layer graph reads the array.
        let kv_key = key("attention.head_count_kv");
        let n_kv = g
            .meta_u32(&kv_key)
            .or_else(|| g.int_arr(&kv_key).and_then(|a| a.iter().copied().max()).map(|v| v as u32))
            .unwrap_or(n_head);
        if n_head == 0 || n_kv == 0 || n_kv > n_head || n_head % n_kv != 0 {
            bail!("{arch}: head_count {n_head} / head_count_kv {n_kv} is not a valid GQA grouping");
        }
        // `attention.key_length` wins when present: Gemma's head_dim is not d/n_head.
        let head_dim = g.meta_u32(&key("attention.key_length")).unwrap_or_else(|| {
            (d / n_head as usize) as u32
        });
        // MoE files put the dense width under a different key. 0 when neither is present, as
        // `load.rs` has it: a non-zero default would size buffers for a model that is not
        // there, while 0 fails the caller's own shape check immediately.
        let ffn = g
            .meta_u32(&key("feed_forward_length"))
            .or_else(|| g.meta_u32(&key("expert_shared_feed_forward_length")))
            .unwrap_or(0) as usize;
        let eps = g.meta_f32(&key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
        // 1e6, not 10000: every architecture this engine loads that omits the key is a
        // long-context one that uses 1e6, and `load.rs` has defaulted this way since Qwen2.
        // Guessing 10000 on such a file rotates every position by the wrong angle.
        let rope_base = g.meta_f32(&key("rope.freq_base")).unwrap_or(1e6);

        // The tokenizer's own list is the authority; `vocab_size` disagrees with it in real files.
        let vocab = g
            .str_arr("tokenizer.ggml.tokens")
            .map(|v| v.len())
            .filter(|&n| n > 0)
            .or_else(|| g.meta_u32(&key("vocab_size")).map(|v| v as usize))
            .unwrap_or(0);

        let has = |t: &str| g.tensors.contains_key(t);
        let is_gemma = arch.starts_with("gemma");

        let layer = LayerSpec {
            n_head,
            n_kv,
            head_dim,
            qdim: n_head * head_dim,
            kvdim: n_kv * head_dim,
            rope_base,
            scale: 1.0 / (head_dim as f32).sqrt(),
        };

        // Recurrent / latent-attention / vision architectures need graph pieces a dense
        // runner does not have. Named rather than boolean so the refusal can say which.
        let non_dense = match arch.as_str() {
            "qwen35" | "qwen35moe" => Some("Gated-DeltaNet recurrent layers"),
            "qwen4exp" => Some("hyper-connections, sparse indexer and recurrent MoE"),
            "deepseek2" | "glm-dsa" => Some("multi-head latent attention"),
            "gpt-oss" => Some("per-head attention sinks and biased SwiGLU-OAI MoE"),
            _ => None,
        };

        Ok(Self {
            n_layers,
            block_count,
            n_nextn,
            d,
            ffn,
            vocab,
            eps,
            layers: vec![layer; n_layers],
            lm_head: if has("output.weight") { "output.weight" } else { "token_embd.weight" }
                .to_string(),
            qkv_bias: has("blk.0.attn_q.bias"),
            qk_norm: has("blk.0.attn_q_norm.weight"),
            // Llama's GGUF weights are permuted so that interleaved-pair rotation reproduces the
            // reference; every other family here is NeoX split-half.
            rope_neox: arch != "llama",
            embed_scale: if is_gemma { (d as f32).sqrt() } else { 1.0 },
            // gpt-oss also ships `post_attention_norm`, but as the ordinary pre-FFN norm
            // rather than a sandwich norm, so `load.rs` excludes it by name.
            sandwich: has("blk.0.post_attention_norm.weight") && arch != "gpt-oss",
            act: if is_gemma { Act::Gelu } else { Act::Silu },
            n_experts: g.meta_u32(&key("expert_count")).unwrap_or(0),
            non_dense,
            arch,
        })
    }

    /// Head dim of layer 0 — the common case, for callers that do not vary per layer.
    pub fn head_dim(&self) -> usize {
        self.layers.first().map_or(0, |l| l.head_dim as usize)
    }

    /// Whether the lm_head shares storage with the embedding table.
    pub fn tied_head(&self) -> bool {
        self.lm_head == "token_embd.weight"
    }

    /// Reject what a plain dense graph would get wrong, naming the feature rather than the model.
    ///
    /// `supported` lists the optional dense features the caller has implemented: pass
    /// `&["qkv_bias"]` to accept Qwen2 and refuse Qwen3's q/k norm. Structural features (MoE,
    /// recurrent layers, MLA) are refused regardless, since no flag lets a dense runner
    /// execute them.
    pub fn require_dense(&self, supported: &[&str]) -> Result<()> {
        if let Some(what) = self.non_dense {
            bail!("{}: needs {what}, which this runner does not implement", self.arch);
        }
        if self.n_experts > 0 {
            bail!("{}: declares {} experts; a dense runner would silently evaluate the shared \
                   FFN and produce a different model", self.arch, self.n_experts);
        }
        for (flag, on) in [
            ("qkv_bias", self.qkv_bias),
            ("qk_norm", self.qk_norm),
            ("sandwich", self.sandwich),
            ("embed_scale", self.embed_scale != 1.0),
            ("gelu", self.act == Act::Gelu),
            ("interleaved_rope", !self.rope_neox),
        ] {
            if on && !supported.contains(&flag) {
                bail!("{}: needs '{flag}', which this runner does not implement — running \
                       without it produces fluent output from the wrong model", self.arch);
            }
        }
        Ok(())
    }
}
