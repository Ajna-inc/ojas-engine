//! ojas-cpu — the pure-CPU inference tier and the conformance oracle for all
//! GPU backends; correctness over speed. Most models implement
//! [`ojas_core::Model`];
//! the encoders ([`CpuWhisper`], [`CpuVit`]) are plain structs with inherent methods.

pub mod cpu_cnn;
pub mod cpu_deepseek;
pub mod cpu_glm;
pub mod cpu_math;
pub mod cpu_qwen;
pub mod cpu_ssm;
pub mod cpu_vit;
pub mod cpu_whisper;
pub mod vit_preprocess;

pub use cpu_deepseek::CpuDeepseek;
pub use cpu_glm::CpuGlm;
pub use cpu_qwen::CpuQwen;
pub use cpu_ssm::CpuSsm;
pub use cpu_vit::CpuVit;
pub use cpu_whisper::CpuWhisper;
pub use vit_preprocess::VitPreproc;
