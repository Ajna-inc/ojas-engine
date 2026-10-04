//! What a decision model asks of a GPU backend.
//!
//! The readouts (`marker.rs`, `causal.rs`) never touch a device directly: a marker
//! readout runs an encoder pass ([`MarkerBackend`]), a causal readout prefills prompts
//! into sequence slots and reads hidden states ([`CausalBackend`]), and a backend
//! builds either from a GGUF ([`DecisionGpu`]). `ojas-models` implements the three on
//! Metal (`DecoderGpu`) and `ojas-cuda` on CUDA (`CudaSsm`, `CudaBert`), so the
//! decision logic, the tests and the server are the same code on both.

use anyhow::{bail, Result};
use ojas_formats::gguf::Gguf;

/// Most sequence slots a causal decision model asks for: the number of prompts of one
/// request run at once. Metal's arena holds four; a backend may hold fewer.
pub const MAX_SLOTS: usize = 4;

/// A prompt for [`CausalBackend::prefill_hidden_slots`]: token ids, some runs of
/// which are given as embedding rows (an image), and optionally a rotary coordinate
/// per row.
pub struct PromptRows<'p> {
    /// One id per row. An embedded row's id is a placeholder and is never gathered.
    pub ids: &'p [u32],
    /// `(first row, rows)`: runs of rows given directly, `d` f32 per row, in
    /// ascending order and not overlapping.
    pub embedded: &'p [(usize, &'p [f32])],
    /// `(t, h, w, e)` per row; `None` ropes every row at its cache row.
    pub positions: Option<&'p [[u32; 4]]>,
}

/// One prompt's share of [`CausalBackend::prefill_hidden_slots`]: rows `span` of
/// `prompt`, prefilled into `slot` at cache rows `span`, reading the hidden states
/// of the rows `read`.
pub struct SlotPrefill<'p> {
    pub prompt: &'p PromptRows<'p>,
    pub span: std::ops::Range<usize>,
    pub slot: usize,
    /// Ascending prompt indices inside `span`.
    pub read: &'p [usize],
}

/// Per-sequence results of [`MarkerBackend::marker_head_forward`].
pub struct MarkerHeadOut {
    /// For each sequence, for each marker position asked for: the scorer's hidden
    /// vector after its GELU, `d` floats.
    pub scorer_hidden: Vec<Vec<Vec<f32>>>,
    /// GPU execution time, seconds.
    pub gpu_s: f64,
}

/// A loaded text encoder with a decision head read at marker tokens (Laya).
pub trait MarkerBackend {
    /// Hidden width of the encoder, the length of each scorer hidden vector.
    fn width(&self) -> usize;
    /// Longest sequence the encoder's position encoding was trained for.
    fn max_positions(&self) -> usize;
    /// The encoder, then the marker head, over `seqs` (independent token sequences).
    /// `qtypes[i]` selects the question-type embedding added to sequence `i`;
    /// `markers[i]` lists the positions in sequence `i` whose scorer hidden vector is
    /// returned.
    fn marker_head_forward(&self, seqs: &[Vec<u32>], qtypes: &[u32], markers: &[Vec<usize>]) -> Result<MarkerHeadOut>;
    /// Per-category GPU time of one encoder pass over `seqs`, `(category, ms)`.
    fn profile_text(&self, _seqs: &[Vec<u32>]) -> Result<Vec<(String, f64)>> {
        bail!("GPU profiling is not available on this backend")
    }
}

/// A loaded causal language model with a recurrent state, sequence slots and,
/// optionally, its vision tower (OpenJev, Lev, Kev).
pub trait CausalBackend {
    /// Hidden width: the length of each hidden state read, and of each embedded row.
    fn width(&self) -> usize;
    /// Sequence slots: prompts prefilled at once, one slot each.
    fn slots(&self) -> usize;
    /// The vision tower's patch side and spatial merge, when a tower is attached.
    fn vision_patch_merge(&self) -> Option<(usize, usize)>;
    /// The merged token grid `(columns, rows)` an image of this size encodes to.
    fn vision_grid(&self, width: usize, height: usize) -> Option<(usize, usize)>;
    /// One image through the tower: `img` is planar CHW f32 as
    /// `ojas_cpu::VitPreproc::preprocess` returns it; the result is
    /// `columns * rows` embedded rows of `width()` floats, in raster order.
    fn encode_image(&self, img: &[f32], width: usize, height: usize) -> Result<Vec<f32>>;
    /// Clear every slot's recurrent state.
    fn reset_session(&self);
    /// Clear one slot's recurrent state.
    fn reset_slot(&self, slot: usize);
    /// Keep slot 0's recurrent state as the saved state. One is kept: the next save
    /// replaces it. (It stays on the device: a 4B model's is 50 MB a request.)
    fn save_state(&self);
    /// Restore slot 0's recurrent state from the saved one.
    fn restore_state(&self);
    /// Copy slot `from`'s recurrent state and its first `rows` cache rows into slot
    /// `to`, so `to` continues the same prefix.
    fn copy_slot_prefix(&self, from: usize, to: usize, rows: usize);
    /// Prefill several prompts at once, each in its own slot, and return the final
    /// hidden state (after the output norm) of each prompt's `read` rows.
    fn prefill_hidden_slots(&self, jobs: &[SlotPrefill]) -> Vec<Vec<Vec<f32>>>;
    /// GPU time spent so far, seconds; a request's cost is the difference.
    fn gpu_seconds(&self) -> f64;
}

/// How a causal decision model is loaded.
#[derive(Clone, Copy, Debug)]
pub struct CausalLoad {
    /// Longest prompt, in tokens: the cache rows a slot needs.
    pub max_seq: usize,
    /// Sequence slots asked for (the host's `--parallel`, else [`MAX_SLOTS`]); a backend
    /// may provide fewer.
    pub slots: usize,
    /// Whether the model's vision projector is to be loaded from beside the file.
    pub images: bool,
}

/// A GPU that loads decision models.
pub trait DecisionGpu {
    type Marker<'a>: MarkerBackend where Self: 'a;
    type Causal<'a>: CausalBackend where Self: 'a;
    /// The backend's name, for banners: `metal`, `cuda`.
    const NAME: &'static str;
    /// Load a text encoder with a marker head. The file's weights are held in f16:
    /// every token goes through batched GEMMs.
    fn load_marker<'a>(&'a self, g: &mut Gguf) -> Result<Self::Marker<'a>>;
    /// Load a recurrent causal model with `load.slots` slots of `load.max_seq` rows.
    fn load_causal<'a>(&'a self, g: &mut Gguf, load: CausalLoad) -> Result<Self::Causal<'a>>;
}
