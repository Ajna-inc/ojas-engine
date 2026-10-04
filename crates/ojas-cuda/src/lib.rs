//! ojas-cuda — CUDA Device backend.
//!
//! Kernel entry names are Metal-canonical (aligned / renamed / CUDA-only /
//! retired / Metal-only gaps).
//! The crate compiles on every host via cudarc dynamic-loading; GPU execution
//! requires an NVIDIA driver at runtime and is exercised only by the
//! `#[ignore]`d conformance tests.

pub mod bert;
pub mod conv;
pub mod decision;
pub mod device;
pub mod expert_cache;
pub mod kernels;
pub mod nvdec;
pub mod prefix;
pub mod qwen35;

pub use bert::CudaBert;
pub use decision::CudaDecision;
pub use device::{CuBuf, CudaGpu, KernelProfile, Recorded};
pub use expert_cache::{ExpertCache, GatherStats};
pub use prefix::PrefixOptions;
pub use qwen35::{CudaSsm, CudaSsmOpts, CudaVit, GemmMode, VitAttn};
