//! The decision models' backend traits on Metal: `ojas_decision` loads and drives a
//! [`DecoderGpu`] through them. The ModernBERT encoder with its marker head is
//! `decoder/text_encoder.rs`; the recurrent causal models use the multi-slot chunk
//! graph (`decoder/graph_chunk.rs`) and the vision tower (`decoder/vision.rs`).

use crate::decoder::{DecoderGpu, MAX_SLOTS, PRECISION_AUTO};
use anyhow::{ensure, Result};
use ojas_decision::{CausalBackend, CausalLoad, DecisionGpu, MarkerBackend, MarkerHeadOut, SlotPrefill};
use ojas_formats::gguf::Gguf;
use ojas_metal::MetalGpu;

/// A Metal device as a decision backend: `DecisionModel::load(&MetalDecision(&gpu), path)`.
/// (A newtype, since the trait and the device are both foreign to this crate.)
pub struct MetalDecision<'g>(pub &'g MetalGpu);

impl<'g> DecisionGpu for MetalDecision<'g> {
    type Marker<'a> = DecoderGpu<'g> where Self: 'a;
    type Causal<'a> = DecoderGpu<'g> where Self: 'a;
    const NAME: &'static str = "metal";

    fn load_marker<'a>(&'a self, g: &mut Gguf) -> Result<DecoderGpu<'g>> {
        // The encoder keeps no KV cache, so the decoder context is minimal. Weights are
        // held in f16 (precision 0): every token goes through batched GEMMs, which the
        // file's quantized blocks would serve one row at a time, and requantizing them
        // per row moves the probabilities further from the file's own numbers.
        // An encoder takes no projector or draft head; files beside it belong to
        // other models.
        g.sidecars = false;
        let dec = DecoderGpu::load(self.0, g, 64, 0, None, None)?;
        ensure!(dec.has_marker_head(), "the encoder loaded without its decision head");
        Ok(dec)
    }

    fn load_causal<'a>(&'a self, g: &mut Gguf, load: CausalLoad) -> Result<DecoderGpu<'g>> {
        g.sidecars = load.images;
        let mut cfg = ojas_core::config::EngineConfig::current();
        cfg.parallel = Some(load.slots.clamp(1, MAX_SLOTS));
        let dec = DecoderGpu::load_with(self.0, g, load.max_seq, PRECISION_AUTO, None, None, cfg)?;
        ensure!(dec.is_recurrent(), "decision readouts of a causal model need a recurrent model");
        Ok(dec)
    }
}

impl MarkerBackend for DecoderGpu<'_> {
    fn width(&self) -> usize { self.text_encoder_width().expect("loaded as a text encoder") }
    fn max_positions(&self) -> usize { self.text_encoder_max_positions().unwrap_or(usize::MAX) }
    fn marker_head_forward(&self, seqs: &[Vec<u32>], qtypes: &[u32], markers: &[Vec<usize>]) -> Result<MarkerHeadOut> {
        DecoderGpu::marker_head_forward(self, seqs, qtypes, markers)
    }
    fn profile_text(&self, seqs: &[Vec<u32>]) -> Result<Vec<(String, f64)>> { DecoderGpu::profile_text(self, seqs) }
}

impl CausalBackend for DecoderGpu<'_> {
    fn width(&self) -> usize { self.d }
    fn slots(&self) -> usize { DecoderGpu::slots(self) }
    fn vision_patch_merge(&self) -> Option<(usize, usize)> { DecoderGpu::vision_patch_merge(self) }
    fn vision_grid(&self, width: usize, height: usize) -> Option<(usize, usize)> { DecoderGpu::vision_grid(self, width, height) }
    fn encode_image(&self, img: &[f32], width: usize, height: usize) -> Result<Vec<f32>> { DecoderGpu::encode_image(self, img, width, height) }
    fn reset_session(&self) { DecoderGpu::reset_session(self) }
    fn reset_slot(&self, slot: usize) { DecoderGpu::reset_slot(self, slot) }
    fn save_state(&self) { *self.saved_state.borrow_mut() = DecoderGpu::state_bytes(self) }
    fn restore_state(&self) { DecoderGpu::set_state_bytes(self, &self.saved_state.borrow()) }
    fn copy_slot_prefix(&self, from: usize, to: usize, rows: usize) { DecoderGpu::copy_slot_prefix(self, from, to, rows) }
    fn prefill_hidden_slots(&self, jobs: &[SlotPrefill]) -> Vec<Vec<Vec<f32>>> { DecoderGpu::prefill_hidden_slots(self, jobs) }
    fn gpu_seconds(&self) -> f64 { DecoderGpu::gpu_seconds(self) }
}
