//! Structural gates on the CUDA native-quant source.
//!
//! These run without a GPU, so nvcc is not the gate. They check what breaks when a shared
//! decoder is edited with only Metal in mind: a Metal-ism leaking into the CUDA dialect, an
//! entry point that no longer exists, or the two backends drifting apart in which formats they
//! claim. A genuine CUDA compile error needs hardware and is caught on the first build there.

use ojas_cuda::kernels;

#[test]
fn cuda_source_has_no_metal_isms() {
    let src = kernels::nat::cuda_decoders() + &kernels::nat::instantiate(kernels::nat::plain_formats());
    for bad in ["[[buffer(", "simd_sum", "device const", "threadgroup ",
                "constant int", "as_type<"] {
        assert!(!src.contains(bad), "Metal-only construct {bad:?} leaked into CUDA source");
    }
    // and the CUDA prologue really is in effect
    assert!(src.contains("__half2float"), "CUDA f16 load missing");
    assert!(src.contains("warp_sum"), "CUDA warp reduction missing");
}

#[test]
fn every_declared_entry_is_defined() {
    // both sets: the grid formats are instantiated into `gemv_iq`, so checking only
    // `plain_formats` would leave seven of them uncovered
    let src = kernels::nat::instantiate(kernels::nat::plain_formats())
        + &kernels::nat::instantiate(kernels::nat::grid_formats());
    for name in kernels::nat::names() {
        assert!(src.contains(&format!("void {name}(")),
            "{name} is declared by names() but never defined");
    }
}

/// A native matvec must resolve to the family that actually carries its codebook requirement.
///
/// The grid formats are instantiated into `gemv_iq`, the only family emitting the IQ tables; the
/// rest go in `gemv`. `family_of` decides which module is compiled and searched for the entry, so
/// a misrouted name is a "named symbol not found" at first dispatch.
#[test]
fn entries_resolve_to_the_family_carrying_their_codebooks() {
    let iq: Vec<&str> = kernels::nat::grid_formats().map(|f| f.tag).collect();
    for name in kernels::nat::names() {
        let tag = name.strip_prefix("gemv_nat_").expect("nat entry name");
        let needs_grid = iq.iter().any(|t| tag == *t || tag.starts_with(&format!("{t}_")));
        let want = if needs_grid { "gemv_iq" } else { "gemv" };
        assert_eq!(kernels::family_of(&name), Some(want), "{name} resolves elsewhere");
        let src = kernels::source_of(&name).expect("no source for {name}");
        assert!(src.contains(&format!("void {name}(")), "{name} missing from its own family");
    }
}

#[test]
fn four_forms_per_supported_format() {
    let n = ojas_core::quant_src::FORMATS.len();
    assert_eq!(kernels::nat::names().len(), n * 4,
        "expected plain/accum/bias/m for each of {n} formats");
}

/// Every format in the shared table now has a CUDA entry, codebooks included.
///
/// A `None` here would make a caller requantise an IQ3_S tensor instead of reading it natively,
/// which is the cost the native path exists to avoid. The check was inverted when the IQ
/// codebooks gained a CUDA rendering.
#[test]
fn every_format_claims_an_entry() {
    for f in ojas_core::quant_src::FORMATS.iter() {
        assert_eq!(kernels::nat::nat_entry(f.ty, ""), Some(format!("gemv_nat_{}", f.tag)),
            "{} has no CUDA native entry", f.tag);
        assert!(kernels::nat::nat_entry(f.ty, "_bias").is_some(), "{}: no bias form", f.tag);
    }
    // and a type nothing implements still reports None rather than inventing a name
    assert_eq!(kernels::nat::nat_entry(9999, ""), None);
}
