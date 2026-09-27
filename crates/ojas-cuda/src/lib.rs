//! ojas-cuda — CUDA Device backend.
//!
//! Kernel entry names are Metal-canonical (aligned / renamed / CUDA-only /
//! retired / Metal-only gaps).
//! The crate compiles on every host via cudarc dynamic-loading; GPU execution
//! requires an NVIDIA driver at runtime and is exercised only by the
//! `#[ignore]`d conformance tests.

pub mod conv;
pub mod device;
pub mod expert_cache;
pub mod kernels;
pub mod nvdec;
pub mod qwen35;

pub use device::{CuBuf, CudaGpu, Recorded};
pub use expert_cache::{ExpertCache, GatherStats};
pub use qwen35::{CudaSsm, CudaSsmOpts, CudaVit, GemmMode, VitAttn};
