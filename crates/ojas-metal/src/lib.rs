//! ojas-metal — Metal device backend.
//!
//! The crate is split so `kernels` builds on every platform while the device does not. The
//! canonical entry names are the cross-backend contract, and `kernels` is pure MSL source
//! text plus the `nat` generator with no Apple dependency, so `ojas-cuda`'s `parity`
//! example can read the real table from Linux rather than checking a hardcoded name list
//! on a Mac.
//!
//! Everything that touches a GPU — `MetalGpu`, the pipelines, the residency sets — lives
//! in `device` and is macOS-only.

pub mod kernels;

// `device` is always compiled on macOS. Off macOS it is compiled only when the `device`
// feature asks for it, and then for type-checking only: the `metal` crate is taken without
// its `link` feature, so every type and signature is present but a binary linking this
// fails on undefined Apple framework symbols — correct, since there is no Metal device
// there.
//
// That distinction is what lets both of these work on one Linux box:
//
//   cargo check -p ojas-models     # feature on — 15k lines of decoder, type-checked
//   cargo test  -p ojas-cuda       # feature off — needs only the kernel source table
//
// Cargo features are additive, so a single `cargo test --workspace` off macOS unifies them,
// turns `device` on for everyone, and the CUDA test binaries then fail to link.
// Per-package commands are the supported way to work off macOS.
#[cfg(any(target_os = "macos", feature = "device"))]
mod device;
#[cfg(any(target_os = "macos", feature = "device"))]
pub use device::*;
