//! ojas-models — model architectures on the Metal device.
#![allow(unexpected_cfgs)] // objc msg_send! expands cfg(cargo-clippy)
pub mod bench;
pub mod decoder;
pub mod laya;
pub mod session;
pub mod weights;
