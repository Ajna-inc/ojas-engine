//! Canonical-name guards: every kernel entry lives in exactly one family.
//
// Unlike the other test files here, this one touches no device — it reads the kernel source
// table — so it runs everywhere now that `ojas-metal::kernels` builds off macOS. Worth keeping
// that way: a name-uniqueness guard that only ran on a Mac did not run during the CUDA port,
// when names drift.

use std::collections::HashMap;

#[test]
fn kernel_names_unique_across_families() {
    let mut seen: HashMap<&str, &str> = HashMap::new();
    for (fam, src) in ojas_metal::kernels::families() {
        for part in src.split("kernel void ").skip(1) {
            let name = part[..part.find('(').unwrap()].trim();
            if let Some(prev) = seen.insert(name, fam) {
                panic!("kernel {name} defined in both {prev} and {fam}");
            }
        }
    }
    assert!(seen.len() > 190, "expected the full kernel surface, got {}", seen.len());
}

#[test]
fn source_of_resolves_every_name() {
    for name in ojas_metal::kernels::all_names() {
        assert!(ojas_metal::kernels::source_of(name).is_some(), "{name} unresolvable");
        assert!(ojas_metal::kernels::family_of(name).is_some(), "{name} has no family");
    }
}

/// Every kernel named by `quant_src::MOE_FORMATS` must actually exist.
///
/// The table is the single source of truth for which kernel decodes a given GGUF type as
/// an expert, and both the loader's validation and the graph's dispatch read it. A row
/// naming a kernel nobody wrote passes the loader and panics at the pipeline lookup, on a
/// model large enough that reaching that point takes minutes. Adding a format is one table
/// row plus one kernel; this asserts the pair.
#[test]
fn moe_format_table_names_real_kernels() {
    use ojas_core::quant_src::{MOE_FORMATS, MoeRole, moe_kernel};
    let mut checked = 0;
    for f in MOE_FORMATS {
        for role in [MoeRole::GateUp, MoeRole::Down] {
            for k in [moe_kernel(f.ty, role), ojas_core::quant_src::moe_kernel_m(f.ty, role)] {
            let Some(k) = k else { continue };
            assert!(
                ojas_metal::kernels::source_of(k.entry).is_some(),
                "MOE_FORMATS ty={} {role:?} names kernel {} which does not exist",
                f.ty, k.entry,
            );
            let (threads, rows) = k.launch;
            assert!(threads > 0 && threads % 32 == 0, "{}: threads {threads} must be a positive multiple of 32", k.entry);
            assert!(rows > 0, "{}: rows/threadgroup must be > 0", k.entry);
            checked += 1;
            }
        }
    }
    assert!(checked >= 9, "expected the MoE expert surface, got {checked}");
}

/// The two expert roles have different kernel sets, and conflating them is a
/// silent-corruption bug rather than a crash: the old dispatcher fell back to
/// `moe_gu_q4k` / `moe_down_q80`, decoding the blocks with another format's
/// walker. These are the cases that used to slip through.
#[test]
fn moe_roles_are_not_interchangeable() {
    use ojas_core::quant_src::{MoeRole, moe_kernel};
    // Q4_K is a gate/up format only — there is no `moe_down_q4k`.
    assert!(moe_kernel(12, MoeRole::GateUp).is_some());
    assert!(moe_kernel(12, MoeRole::Down).is_none(), "Q4_K must not resolve as a down kernel");
    // Q8_0 has both: the qwen4exp MTP draft head ships Q8_0 experts.
    assert!(moe_kernel(8, MoeRole::Down).is_some());
    assert!(moe_kernel(8, MoeRole::GateUp).is_some());
    // IQ4_NL is a down format only — `moe_gu_iq4nl` is not written yet, and until
    // it is, a gate/up tensor of this type must be refused rather than decoded by
    // some other format's kernel.
    assert!(moe_kernel(20, MoeRole::Down).is_some());
    assert!(moe_kernel(20, MoeRole::GateUp).is_none(), "IQ4_NL has no gate/up kernel yet");
    // IQ3_S is a gate/up format only — `moe_down_iq3s` is not written.
    assert!(moe_kernel(21, MoeRole::GateUp).is_some());
    assert!(moe_kernel(21, MoeRole::Down).is_none(), "IQ3_S has no down kernel");
    // IQ4_XS is the one format with a kernel in both roles, and they are different
    // kernels over the same superblock, so the lookup is keyed on the role rather than on
    // the type alone.
    assert!(moe_kernel(23, MoeRole::GateUp).is_some());
    assert!(moe_kernel(23, MoeRole::Down).is_some());
}
