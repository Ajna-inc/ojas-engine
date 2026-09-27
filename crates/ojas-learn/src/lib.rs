//! ojas-learn — training for ojas models.
//!
//! A define-by-run tape ([`tape::Tape`]) over a small primitive set
//! ([`backend::Backend`]). The CPU backend ([`cpu::Cpu`]) is the reference:
//! gradients are checked against finite differences there, and every device
//! backend ([`cuda::Cuda`], [`metal::Metal`]) is checked against it, primitive
//! by primitive.

pub mod backend;
pub mod check;
pub mod cpu;
pub mod data;
pub mod eval;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "metal")]
pub mod metal;
pub mod models;
pub mod tape;
pub mod train;
pub mod passenger;

pub use backend::{Backend, Binary, Unary};
pub use tape::{AdamW, BnRunning, Param, Tape, Var};
