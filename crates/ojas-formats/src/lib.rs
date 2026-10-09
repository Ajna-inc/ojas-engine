//! ojas-formats — backend-free model IO: GGUF, safetensors, quantization, IQ codebooks.
pub mod gguf;
pub mod safetensors;
pub mod mxfp4;
pub mod quant;
/// The IQ codebooks moved to `ojas-core` so both GPU backends can render them into their own
/// dialect from one copy of the data (`ojas_core::iq_grids`). Re-exported here because every
/// caller — `ojas-cpu`'s dequantisers, the encoders — reaches them by this path.
pub use ojas_core::iq_tables;
pub mod synth;
pub mod iq_encode;

pub mod mtp;
pub mod mmproj;
pub mod onnx;
pub mod pth;
pub mod swarm_id;
