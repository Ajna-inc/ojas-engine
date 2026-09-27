//! Metal kernel families, split by function rather than by model. A model compiles only
//! the families its config needs; kernels are addressed by canonical name.
pub mod attn;
pub mod attn_core;
pub mod cnn_train;
pub mod gemm_ac;
pub mod gemm_fat;
pub mod gemv;
pub mod mla;
pub mod moe;
pub mod moe_iq;
pub mod ops;
pub mod iq_grids;
pub mod nat;
pub mod prelude;
pub mod qwen4exp;
pub mod requant_iq;
pub mod ssm;
pub mod train;
pub mod vision;

use std::collections::HashMap;
use std::sync::OnceLock;

/// (family, kernel bodies). `attn`/`moe`/`moe_iq` are self-contained sources; the
/// families listed here compile against the shared PRELUDE.
const SPLIT: &[(&str, &str)] = &[
    ("gemv", gemv::BODY),
    ("ops", ops::BODY),
    ("attn_core", attn_core::BODY),
    ("mla", mla::BODY),
    ("ssm", ssm::BODY),
    ("qwen4exp", qwen4exp::BODY),
    ("vision", vision::BODY),
];

fn table() -> &'static HashMap<&'static str, &'static str> {
    static T: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    T.get_or_init(|| {
        let mut m = HashMap::new();
        for (f, body) in SPLIT {
            // The native matvecs are instantiated only in `gemv`. The decoders
            // themselves are cheap macro text and go everywhere, since a family may
            // want to decode blocks for its own kernels, but emitting the entry points
            // in every family would define the same kernel name several times.
            // `tests/manifest.rs` rejects that: `source_of` resolves a name to one
            // family, so duplicates make the lookup arbitrary.
            let nat_entries = if *f == "gemv" {
                nat::instantiate(nat::plain_formats())
            } else {
                String::new()
            };
            let src: &'static str =
                Box::leak(format!("{}\n{}\n{}\n{}", prelude::PRELUDE, nat::metal_decoders(), body,
                                  nat_entries).into_boxed_str());
            m.insert(*f, src);
        }
        m.insert("attn", attn::ATTN_KERNELS);
        m.insert("moe", moe::MOE_KERNELS);
        // moe_iq carries kernel bodies only; the IQ codebooks come from the shared
        // generator rather than a second pasted copy. Listing the grids it indexes here
        // keeps the constant-address-space footprint to what it reads.
        let moe_iq_src: &'static str = Box::leak(
            format!("#include <metal_stdlib>\nusing namespace metal;\n{}\n{}",
                    iq_grids::grids(&["iq2xxs_grid", "iq3xxs_grid", "ksigns_iq2xs",
                                      "iq2s_grid", "iq3s_grid", "kmask_iq2xs"]),
                    format!("{}\n{}\n{}", moe_iq::MOE_IQ_KERNELS, moe_iq::direct_kernels(), moe_iq::EXPERT_HASH_KERNEL)).into_boxed_str());
        m.insert("moe_iq", moe_iq_src);
        // Self-contained like moe_iq, but assembled at first use: the ~15 KB of IQ
        // codebooks are shared with nothing else, so they stay out of the prelude every
        // other family compiles against, and only the codebooks this family indexes are
        // emitted. The IQ tables total ~41 KB and constant-address-space pressure is the
        // leading suspect for the IQ1_S specialization measuring slower than the generic
        // body, so `iq1s_grid_gpu` is emitted only when the specialization is on rather
        // than making the default path pay for a table it never reads.
        let mut needed = vec!["kmask_iq2xs", "ksigns_iq2xs", "iq2xxs_grid", "iq2xs_grid",
                              "iq2s_grid", "iq3xxs_grid", "iq3s_grid", "iq1s_grid"];
        // OJAS_GRIDS_ALL is a diagnostic: it emits the extra table while leaving the
        // kernel bodies alone, isolating constant-address-space footprint as a single
        // variable, where adding the table by changing the kernel moves two at once.
        // The specialization is on by default, so its codebook is too; emitting it costs
        // nothing measurable (33 KB vs 41 KB of constants, 0.998x), but a family should
        // still not carry a table it never reads.
        if std::env::var("OJAS_NO_FAST").is_err() || std::env::var("OJAS_GRIDS_ALL").is_ok() {
            needed.push("iq1s_grid_gpu");
        }
        let iq: &'static str = Box::leak(
            format!("#include <metal_stdlib>\nusing namespace metal;\n{}\n{}\n{}\n{}",
                    iq_grids::grids(&needed), nat::metal_decoders(), requant_iq::BODY,
                    nat::instantiate(nat::grid_formats())).into_boxed_str());
        m.insert("requant_iq", iq);
        m.insert("train", train::TRAIN_KERNELS);
        // ojas-learn's CNN training primitives (im2col conv, pool, BatchNorm,
        // GridSample, AdamW): self-contained, the Metal twin of ojas-cuda's `learn`
        // family. Nothing else reads these names.
        m.insert("cnn_train", cnn_train::BODY);
        m
    })
}

pub fn family_source(family: &str) -> Option<&'static str> {
    table().get(family).copied()
}

pub fn families() -> impl Iterator<Item = (&'static str, &'static str)> {
    table().iter().map(|(f, s)| (*f, *s))
}

/// Resolve a canonical kernel entry name to the source of the family holding it.
pub fn source_of(entry: &str) -> Option<&'static str> {
    static IDX: OnceLock<HashMap<String, &'static str>> = OnceLock::new();
    IDX.get_or_init(|| {
        let mut m = HashMap::new();
        for (_, src) in table() {
            for part in src.split("kernel void ").skip(1) {
                if let Some(p) = part.find('(') {
                    m.insert(part[..p].trim().to_string(), *src);
                }
            }
        }
        m
    })
    .get(entry)
    .copied()
}

pub fn family_of(entry: &str) -> Option<&'static str> {
    let needle = format!("kernel void {entry}(");
    table().iter().find(|(_, src)| src.contains(needle.as_str())).map(|(f, _)| *f)
}

pub fn all_names() -> Vec<&'static str> {
    let mut out = vec![];
    for (_, src) in table() {
        for part in src.split("kernel void ").skip(1) {
            if let Some(p) = part.find('(') {
                out.push(&part[..p]);
            }
        }
    }
    out
}
