//! Kernel contract layer: the pluggable-backend design.
//!
//! One canonical manifest of kernels (name, family, signature, tier requirement).
//! Each backend (metal / cuda / wgpu / cpu) ships the same entry names in its own
//! dialect and is dispatched through `KernelRuntime`. A kernel counts as supported
//! on a backend only when its conformance test passes there against the CPU
//! oracle — one CI-checked result per (kernel x backend) cell.

use anyhow::Result;
use crate::Device;

/// Hardware capability tier — generalizes the Apple-family tiering
/// (7+ native MMA / 6 shuffle-shim / <=5 CPU floor) across all platforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Scalar/vector CPU: NEON + SDOT, AVX2. Always available; the conformance oracle.
    C,
    /// Subgroup/simd ops, no matrix units: Metal family 6, Vulkan subgroups, WGSL.
    B,
    /// Matrix units: Metal family 7+ simdgroup MMA, CUDA sm70+ wmma/tensor cores.
    A,
}

/// What a device can do. Host graphs pick kernel variants from this rather than
/// assuming a lowest common denominator.
#[derive(Clone, Debug)]
pub struct Caps {
    pub tier: Tier,
    pub f16_compute: bool,
    pub simd_width: u32,
    pub unified_memory: bool,
    /// Max single-buffer bytes (mobile jetsam / maxBufferLength constraints).
    pub max_buffer: usize,
    /// Backend-specific extras, capability-gated by name (e.g. "cuda_graphs",
    /// "kv_context_shift", "wgpu_subgroups").
    pub features: Vec<&'static str>,
}

impl Caps {
    pub fn has(&self, feature: &str) -> bool { self.features.contains(&feature) }
}

/// Manifest entry: the platform-neutral identity of one kernel.
#[derive(Clone, Debug)]
pub struct KernelSpec {
    /// Canonical entry name, identical in every dialect (msl/cu/wgsl/cpu).
    pub name: &'static str,
    /// Family key, one compilation unit per backend ("glue", "gemv", "q4",
    /// "batched", "ssm", "mla", "moe", "attn", "train_flce", ...).
    pub family: &'static str,
    /// Positional buffer count the dispatch passes (validated at dispatch in debug).
    pub n_bufs: u8,
    /// Trailing u32 constant count.
    pub n_consts: u8,
    /// Minimum tier: dispatching on a lower-tier device is a bug, not a fallback —
    /// fallbacks are separate manifest entries selected by the host graph.
    pub min_tier: Tier,
}

/// Static manifest slice per family. In debug builds, backends verify at startup
/// that every entry name they claim resolves in their compiled family, which guards
/// against name drift.
pub type Manifest = &'static [KernelSpec];

/// The dispatch seam every backend implements. Grid/block triples are
/// threadgroups/threads-per-tg on Metal, blocks/threads-per-block on CUDA.
pub trait KernelRuntime: Device {
    /// Compile (or fetch cached) this family from the backend's own dialect.
    fn ensure_family(&mut self, family: &str) -> Result<()>;
    /// True if `name` resolved in a compiled family on this backend.
    fn has_kernel(&self, name: &str) -> bool;
    fn caps(&self) -> &Caps;
    /// Begin an encoder/stream for a chain of dispatches.
    fn begin(&self) -> Self::Enc;
    /// Dispatch by canonical name. Buffers positional, then u32 constants.
    fn dispatch(&self, enc: &Self::Enc, name: &str,
                bufs: &[(&Self::Buf, u64)], consts: &[u32],
                grid: [u32; 3], block: [u32; 3]) -> Result<()>;
    /// Submit the encoder and wait (sub-second chains — the watchdog rule).
    fn submit(&self, enc: Self::Enc) -> Result<()>;
}

/// Variant selection: given a base kernel and device caps, choose the best
/// manifest entry. Variants share the base name with a suffix ("attention_m" ->
/// "attention_m_mma_128"), and selection is explicit data rather than string
/// matching spread through the graphs.
pub struct VariantSet {
    pub base: &'static str,
    /// (required tier, required feature or "", entry name) — first match wins,
    /// ordered best-first.
    pub variants: &'static [(Tier, &'static str, &'static str)],
}

impl VariantSet {
    pub fn pick(&self, caps: &Caps) -> &'static str {
        for (tier, feat, name) in self.variants {
            if caps.tier >= *tier && (feat.is_empty() || caps.has(feat)) {
                return name;
            }
        }
        self.base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(tier: Tier, features: Vec<&'static str>) -> Caps {
        Caps { tier, f16_compute: true, simd_width: 32, unified_memory: true,
               max_buffer: 1 << 30, features }
    }

    #[test]
    fn tier_ordering() {
        assert!(Tier::A > Tier::B);
        assert!(Tier::B > Tier::C);
    }

    #[test]
    fn variant_pick_respects_tier_and_features() {
        static V: VariantSet = VariantSet {
            base: "attention_m",
            variants: &[
                (Tier::A, "", "attention_m_mma_128"),
                (Tier::B, "wgpu_subgroups", "attention_m_subgroup"),
                (Tier::C, "", "attention_m"),
            ],
        };
        assert_eq!(V.pick(&caps(Tier::A, vec![])), "attention_m_mma_128");
        assert_eq!(V.pick(&caps(Tier::B, vec!["wgpu_subgroups"])), "attention_m_subgroup");
        assert_eq!(V.pick(&caps(Tier::B, vec![])), "attention_m");     // no subgroups -> base
        assert_eq!(V.pick(&caps(Tier::C, vec![])), "attention_m");
    }

    #[test]
    fn platform_feature_gate() {
        let c = caps(Tier::A, vec!["cuda_graphs", "kv_context_shift"]);
        assert!(c.has("cuda_graphs") && !c.has("wgpu_subgroups"));
    }
}
