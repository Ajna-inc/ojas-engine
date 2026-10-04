//! The decision models' backend traits on CUDA: `ojas_decision` loads and drives a
//! [`CudaBert`] (Laya's marker readout) or a [`CudaSsm`] (the Qwen3.5 readouts, with
//! its vision tower when a projector sits beside the file) through them.

use crate::{CudaBert, CudaGpu, CudaSsm, CudaSsmOpts, GemmMode};
use anyhow::{ensure, Context, Result};
use ojas_core::Model;
use ojas_decision::{CausalBackend, CausalLoad, DecisionGpu, MarkerBackend, MarkerHeadOut, SlotPrefill};
use ojas_formats::gguf::Gguf;

/// A CUDA device as a decision backend: `DecisionModel::load(&CudaDecision::new(0)?, path)`.
/// Each model it loads opens its own context on the device.
pub struct CudaDecision {
    ordinal: usize,
}

impl CudaDecision {
    pub fn new(ordinal: usize) -> Result<Self> {
        let n = CudaGpu::device_count()?;
        ensure!(ordinal < n, "CUDA device {ordinal} does not exist ({n} devices)");
        Ok(CudaDecision { ordinal })
    }
}

impl DecisionGpu for CudaDecision {
    type Marker<'a> = CudaBert;
    type Causal<'a> = CudaSsm;
    const NAME: &'static str = "cuda";

    fn load_marker(&self, g: &mut Gguf) -> Result<CudaBert> {
        let bert = CudaBert::load(self.ordinal, g)?;
        ensure!(bert.has_marker_head(), "the encoder loaded without its decision head");
        Ok(bert)
    }

    fn load_causal(&self, g: &mut Gguf, load: CausalLoad) -> Result<CudaSsm> {
        // One tensor-core pass per matmul (activations rounded to f16, as Metal): the causal
        // readouts hold parity with it, and the split pass would double every GEMM.
        // `OJAS_CUDA_GEMM` still overrides.
        let gemm = if std::env::var("OJAS_CUDA_GEMM").is_ok() { GemmMode::from_env() } else { GemmMode::Fast };
        // Up to eight slots (twice `MAX_SLOTS`, the portable default): a seven-question
        // request then runs its prompts in one shared pass rather than two, 10 % faster on
        // Kev-4B, at 540 MB of f32 KV cache per slot of 8192 rows.
        let opts = CudaSsmOpts { context: load.max_seq, slots: load.slots.clamp(1, 8), ordinal: self.ordinal, gemm, ..CudaSsmOpts::default() };
        let mut m = CudaSsm::load_with(g, opts)?;
        if load.images {
            let explicit = ojas_core::config::EngineConfig::current().mmproj;
            if let Some(mm) = ojas_formats::mmproj::discover(std::path::Path::new(&g.path), explicit.as_deref())? {
                let mut mg = Gguf::open(mm.to_string_lossy().as_ref()).with_context(|| format!("opening {}", mm.display()))?;
                ojas_formats::mmproj::validate(g, &mg)?;
                m.attach_vit_gguf(&mut mg).with_context(|| format!("attaching {}", mm.display()))?;
            }
        }
        Ok(m)
    }
}

impl MarkerBackend for CudaBert {
    fn width(&self) -> usize { self.width() }
    fn max_positions(&self) -> usize { self.max_positions() }
    fn marker_head_forward(&self, seqs: &[Vec<u32>], qtypes: &[u32], markers: &[Vec<usize>]) -> Result<MarkerHeadOut> {
        CudaBert::marker_head_forward(self, seqs, qtypes, markers)
    }
}

impl CausalBackend for CudaSsm {
    fn width(&self) -> usize { self.hidden_dim() }
    fn slots(&self) -> usize { CudaSsm::slots(self) }
    fn vision_patch_merge(&self) -> Option<(usize, usize)> { CudaSsm::vision_patch_merge(self) }
    fn vision_grid(&self, width: usize, height: usize) -> Option<(usize, usize)> { CudaSsm::vision_grid(self, width, height) }
    fn encode_image(&self, img: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        Ok(self.encode_image_trace(img, width, height, false)?.out)
    }
    fn reset_session(&self) { self.reset() }
    fn reset_slot(&self, slot: usize) { CudaSsm::reset_slot(self, slot) }
    fn save_state(&self) { CudaSsm::save_state(self).expect("CUDA save_state") }
    fn restore_state(&self) { CudaSsm::restore_state(self).expect("CUDA restore_state") }
    fn copy_slot_prefix(&self, from: usize, to: usize, rows: usize) {
        CudaSsm::copy_slot_prefix(self, from, to, rows).expect("CUDA copy_slot_prefix")
    }
    fn prefill_hidden_slots(&self, jobs: &[SlotPrefill]) -> Vec<Vec<Vec<f32>>> {
        CudaSsm::prefill_hidden_slots(self, jobs).expect("CUDA prefill_hidden_slots")
    }
    fn gpu_seconds(&self) -> f64 { CudaSsm::gpu_seconds(self) }
}
