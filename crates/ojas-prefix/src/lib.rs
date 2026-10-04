//! The prompt-prefix cache and the document cache, shared by every GPU backend.
//!
//! A decoder keeps model state for prompt prefixes in fixed blocks (`prefix_cache`),
//! optionally backed by a directory that outlives the process (`prefix_disk`), and
//! spans of a prompt reusable at another position (`doc_cache`). These modules hold the
//! bookkeeping, the file format and the key rotation only; copying state in and out of
//! the model is each backend's job (`ojas-models` on Metal, `ojas-cuda` on CUDA).

pub mod doc_cache;
pub mod prefix_cache;
pub mod prefix_disk;

use ojas_core::config::EngineConfig;

/// Bytes for the prompt-prefix cache: `prefix_cache_gb` when set (0 disables it),
/// else a sixteenth of `ram_bytes`, at most 4 GB.
pub fn budget(cfg: &EngineConfig, ram_bytes: u64) -> usize {
    match cfg.prefix_cache_gb {
        Some(gb) => (gb.max(0.0) * 1e9) as usize,
        None => (ram_bytes / 16).min(4_000_000_000) as usize,
    }
}

/// The machine's physical memory; 64 GiB when the query fails.
#[cfg(target_os = "macos")]
pub fn physical_ram_bytes() -> u64 {
    let mut v: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = std::ffi::CString::new("hw.memsize").unwrap();
    let ok = unsafe {
        libc::sysctlbyname(name.as_ptr(), &mut v as *mut u64 as *mut libc::c_void, &mut len,
                           std::ptr::null_mut(), 0)
    };
    if ok == 0 && v > 0 { v } else { 64u64 << 30 }
}

/// `sysctlbyname("hw.memsize")` has no portable twin; `_SC_PHYS_PAGES` is the POSIX one.
/// Same 64 GiB fallback as the macOS path uses when the query fails.
#[cfg(not(target_os = "macos"))]
pub fn physical_ram_bytes() -> u64 {
    let (pages, page) = unsafe {
        (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE))
    };
    if pages > 0 && page > 0 { pages as u64 * page as u64 } else { 64u64 << 30 }
}
