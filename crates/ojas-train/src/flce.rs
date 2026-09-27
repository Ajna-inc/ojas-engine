//! Chunked fused linear-cross-entropy (FLCE) — loss + dHidden + dW for a 248k
//! vocab lm_head without materializing full-sequence logits.
use ojas_metal::{MBuf, MetalGpu};
use anyhow::Result;
use metal::ComputePipelineState;

use ojas_metal::kernels::train::TRAIN_KERNELS;


pub struct Flce {
    pub(crate) p_row: ComputePipelineState,
}

pub struct FlceOut {
    pub loss: Vec<f32>,
    pub dh: MBuf,
    pub dw: MBuf,
}

impl Flce {
    pub fn new(gpu: &MetalGpu) -> Result<Flce> {
        let mut ps: std::collections::HashMap<String, _> = gpu
            .compile_all(TRAIN_KERNELS, |n| n.starts_with("flce_"))?
            .into_iter().collect();
        Ok(Flce {
            p_row: ps.remove("flce_row").unwrap(),
        })
    }
}
