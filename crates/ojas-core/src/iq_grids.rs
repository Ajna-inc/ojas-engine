//! IQ codebooks as GPU source, rendered per dialect from the one copy of the data.
//!
//! The tables live in [`crate::iq_tables`] as plain Rust arrays, generated from a
//! supplied `ggml-common.h` by `scripts/gen_iq_tables.py`. Rendering the GPU
//! declarations from those arrays keeps one copy of the data and serves every
//! backend from it.
//!
//! A kernel family asks for only the codebooks it indexes. `iq1s_grid` alone is 2048 `uint64_t`
//! (16 KiB) and `iq2s_grid` another 8 KiB; emitting every table into every family would push
//! constant memory for no reason.

use crate::iq_tables as t;

/// Which GPU dialect to emit. The tables are identical; only the storage qualifier and the
/// 64-bit integer spelling differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Metal,
    Cuda,
}

impl Dialect {
    /// `constant` on Metal, `__constant__` on CUDA — the address space these belong in on both.
    fn qualifier(self) -> &'static str {
        match self {
            Dialect::Metal => "constant",
            Dialect::Cuda => "__constant__",
        }
    }

    /// MSL spells the 64-bit type `uint64_t` and CUDA `unsigned long long`; `uchar`/`uint` are
    /// Metal's and need their C spellings in CUDA.
    fn ty(self, rust_ty: &str) -> &'static str {
        match (self, rust_ty) {
            (_, "u8") => match self {
                Dialect::Metal => "uchar",
                Dialect::Cuda => "unsigned char",
            },
            (_, "u32") => match self {
                Dialect::Metal => "uint",
                Dialect::Cuda => "unsigned int",
            },
            (_, "u64") => match self {
                Dialect::Metal => "uint64_t",
                Dialect::Cuda => "unsigned long long",
            },
            (_, "i8") => "char",
            (_, other) => unreachable!("no dialect spelling for {other}"),
        }
    }

    /// 64-bit literals need a suffix in both dialects or they are truncated to `int` before the
    /// assignment — silently, and only for the values that do not fit.
    fn suffix(self, rust_ty: &str) -> &'static str {
        match rust_ty {
            "u64" => "ULL",
            _ => "",
        }
    }
}

/// One table: the name kernels index it by, its Rust element type, and its values as hex or
/// decimal text.
struct Table {
    name: &'static str,
    ty: &'static str,
    render: fn() -> Vec<String>,
}

fn hex64(v: &[u64]) -> Vec<String> {
    v.iter().map(|x| format!("0x{x:016x}")).collect()
}
fn hex32(v: &[u32]) -> Vec<String> {
    v.iter().map(|x| format!("0x{x:08x}")).collect()
}
fn dec8(v: &[u8]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

const TABLES: &[Table] = &[
    Table { name: "kmask_iq2xs", ty: "u8", render: || dec8(&t::KMASK_IQ2XS) },
    Table { name: "ksigns_iq2xs", ty: "u8", render: || dec8(&t::KSIGNS_IQ2XS) },
    Table { name: "iq2xxs_grid", ty: "u64", render: || hex64(&t::IQ2XXS_GRID) },
    Table { name: "iq2xs_grid", ty: "u64", render: || hex64(&t::IQ2XS_GRID) },
    Table { name: "iq2s_grid", ty: "u64", render: || hex64(&t::IQ2S_GRID) },
    Table { name: "iq3xxs_grid", ty: "u32", render: || hex32(&t::IQ3XXS_GRID) },
    Table { name: "iq3s_grid", ty: "u32", render: || hex32(&t::IQ3S_GRID) },
    Table { name: "iq1s_grid", ty: "u64", render: || hex64(&t::IQ1S_GRID) },
    Table { name: "iq1s_grid_gpu", ty: "u32", render: || hex32(&t::IQ1S_GRID_GPU) },
    Table {
        name: "kvalues_iq4nl",
        ty: "i8",
        render: || t::KVALUES_IQ4NL.iter().map(|x| x.to_string()).collect(),
    },
];

/// Declarations for the named codebooks, concatenated, in `dialect`.
///
/// An unknown name panics rather than emitting nothing: a typo would otherwise surface as an
/// undeclared-identifier error from the shader compiler, pointing at the kernel instead of at
/// the list that failed to include the table.
pub fn grids(dialect: Dialect, names: &[&str]) -> String {
    let mut out = String::new();
    for n in names {
        let t = TABLES
            .iter()
            .find(|t| t.name == *n)
            .unwrap_or_else(|| panic!("no IQ codebook named {n}"));
        let vals = (t.render)();
        out.push_str(&format!(
            "{} {} {}[{}] = {{\n",
            dialect.qualifier(),
            dialect.ty(t.ty),
            t.name,
            vals.len()
        ));
        let sfx = dialect.suffix(t.ty);
        for chunk in vals.chunks(8) {
            out.push_str("    ");
            for v in chunk {
                out.push_str(v);
                out.push_str(sfx);
                out.push_str(", ");
            }
            out.push('\n');
        }
        out.push_str("};\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rendered text must parse back to the arrays it came from, which catches a
    /// wrong element count, a truncated literal, or a dialect spelling that changes a
    /// value rather than just its type.
    #[test]
    fn rendered_tables_round_trip() {
        for d in [Dialect::Metal, Dialect::Cuda] {
            let src = grids(d, &["iq2xxs_grid", "iq3s_grid", "ksigns_iq2xs", "iq1s_grid"]);
            let nums: Vec<u64> = src
                .split('{')
                .nth(1)
                .unwrap()
                .split('}')
                .next()
                .unwrap()
                .split(',')
                .filter_map(|s| {
                    let s = s.trim().trim_end_matches("ULL");
                    s.strip_prefix("0x").map(|h| u64::from_str_radix(h, 16).unwrap())
                })
                .collect();
            assert_eq!(nums.len(), t::IQ2XXS_GRID.len(), "{d:?}: iq2xxs_grid length");
            assert_eq!(nums, t::IQ2XXS_GRID.to_vec(), "{d:?}: iq2xxs_grid values");
        }
    }

    #[test]
    fn dialects_differ_only_in_spelling() {
        let m = grids(Dialect::Metal, &["iq3s_grid"]);
        let c = grids(Dialect::Cuda, &["iq3s_grid"]);
        let strip = |s: &str| {
            s.replace("__constant__", "").replace("constant", "")
                .replace("unsigned int", "").replace("uint", "")
        };
        assert_eq!(strip(&m), strip(&c));
    }

    #[test]
    #[should_panic(expected = "no IQ codebook named")]
    fn unknown_name_panics() {
        grids(Dialect::Cuda, &["iq9_nonexistent"]);
    }
}
