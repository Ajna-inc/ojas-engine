//! IQ codebooks for the Metal families that index them.
//!
//! This was 902 lines of generated MSL holding the same numbers as
//! `ojas-core`'s `iq_tables` — two generated copies that had to be regenerated together, and
//! still gave CUDA nothing. The tables now live once in core and are rendered per dialect, so
//! this is the Metal spelling of that call and nothing else.

use ojas_core::iq_grids::{grids as render, Dialect};

/// Source for the named codebooks, concatenated. Unknown names panic — a typo would otherwise
/// emit nothing and surface as an undeclared-identifier error pointing at the kernel rather than
/// at the list that failed to include the table.
pub fn grids(names: &[&str]) -> String {
    render(Dialect::Metal, names)
}
