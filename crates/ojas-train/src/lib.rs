//! ojas-train — Metal training: Kit/Flce/MoeFfn + Trainer (ojas_core::Learner).
#![allow(unexpected_cfgs)] // objc msg_send! expands cfg(cargo-clippy)
pub mod checks;
pub mod flce;
pub mod kit;
pub mod moe_ffn;
pub mod trainer;

pub use flce::Flce;
pub use kit::Kit;
pub use ojas_metal::kernels::train::TRAIN_KERNELS;
pub use trainer::Trainer;

pub mod kernels {
    pub use crate::flce::{Flce, FlceOut};
    pub use crate::kit::Kit;
    pub(crate) use crate::kit::{g1, g2};
    pub use crate::moe_ffn::MoeFfn;
    pub use ojas_metal::kernels::train::TRAIN_KERNELS;
}
