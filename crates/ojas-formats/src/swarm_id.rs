//! A GGUF's swarm [`Identity`]: the layout (architecture, dimensions, every tensor's
//! name, storage type and shape) and the content (every tensor's bytes).
//!
//! Hashing reads the whole file, which for a large model is seconds to minutes, so
//! the result is cached under `cache_dir()/swarm-identity`, keyed by each part's
//! canonical path, size and modification time (and, on Unix, inode and ctime, which
//! nothing short of a write can set back). A reload of an unchanged file is instant;
//! any write to it is a miss.
//!
//! The dimensions are every scalar `{arch}.*` key in the metadata plus the vocabulary
//! size, taken as the file states them, so no architecture needs a list of its own.

use anyhow::{bail, Context, Result};
use crate::gguf::{gguf_type_name, Gguf, Meta};
use ojas_swarm_proto::{Identity, IdentityBuilder};
use std::path::{Path, PathBuf};

/// Progress is reported at most this often, in bytes hashed.
const PROGRESS_STEP: u64 = 256 << 20;

/// `progress(done, total)` in bytes. Returns the identity and the model's total size
/// on disk.
pub fn identity(path: &str, progress: &mut dyn FnMut(u64, u64)) -> Result<(Identity, u64)> {
    let g = Gguf::open(path).with_context(|| format!("opening {path}"))?;
    let parts = g.shard_paths();
    let key = cache_key(&parts)?;
    let bytes: u64 = key.iter().map(|p| p.size).sum();
    if let Some(id) = cached(&key) {
        return Ok((id, bytes));
    }
    let id = compute(&g, &parts, progress)?;
    store(&key, &id);
    Ok((id, bytes))
}

/// The hash itself, uncached.
pub fn compute(g: &Gguf, parts: &[PathBuf], progress: &mut dyn FnMut(u64, u64)) -> Result<Identity> {
    let arch = g.arch();
    let prefix = format!("{arch}.");
    let mut b = IdentityBuilder::new(&arch);
    for (k, v) in &g.meta {
        let Some(key) = k.strip_prefix(&prefix) else { continue };
        let v = match v {
            Meta::U32(x) => *x as u64,
            Meta::U64(x) => *x,
            Meta::I32(x) => *x as i64 as u64,
            // Floats (rope base, norm epsilon) change the function as surely as a
            // width does; their bits are the value.
            Meta::F32(x) => x.to_bits() as u64,
            Meta::Bool(x) => *x as u64,
            _ => continue,
        };
        b = b.dim(key, v);
    }
    if let Some(t) = g.str_arr("tokenizer.ggml.tokens") {
        b = b.dim("vocab", t.len() as u64);
    }

    let spans = spans(g, parts)?;
    let total: u64 = spans.iter().map(|s| s.len).sum();
    let mut done = 0u64;
    let mut reported = 0u64;
    progress(0, total);
    let mut files: Vec<Option<Source>> = (0..parts.len()).map(|_| None).collect();
    for s in &spans {
        if files[s.part].is_none() {
            files[s.part] = Some(Source::open(&parts[s.part])?);
        }
        let src = files[s.part].as_mut().unwrap();
        let bytes = src.slice(s.offset, s.len).with_context(|| format!("reading tensor {}", s.name))?;
        b.tensor(&s.name, &s.dtype, &s.shape, bytes);
        done += s.len;
        if done - reported >= PROGRESS_STEP {
            progress(done, total);
            reported = done;
        }
    }
    progress(total, total);
    Ok(b.finish())
}

struct Span {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    part: usize,
    offset: u64,
    len: u64,
}

/// Where each tensor's bytes are. A storage type the reader cannot size is hashed
/// over its extent up to the next tensor (or the end of the part): that includes
/// alignment padding, which is fixed by the file, so the id is still a function of
/// the file alone.
fn spans(g: &Gguf, parts: &[PathBuf]) -> Result<Vec<Span>> {
    let mut by_part: Vec<Vec<(u64, &str)>> = vec![Vec::new(); parts.len()];
    for (name, ti) in &g.tensors {
        by_part.get_mut(ti.part).context("tensor in an unknown part")?.push((ti.rel_offset, name));
    }
    by_part.iter_mut().for_each(|v| v.sort());
    let mut out = Vec::with_capacity(g.tensors.len());
    for (name, ti) in &g.tensors {
        let base = g.data_offsets[ti.part];
        let (offset, len) = match g.tensor_meta(name) {
            Some((_, off, len, _)) => (off, len),
            None => {
                let v = &by_part[ti.part];
                let i = v.iter().position(|(o, n)| *o == ti.rel_offset && *n == name.as_str()).unwrap();
                let end = match v.get(i + 1) {
                    Some((o, _)) => base + o,
                    None => std::fs::metadata(&parts[ti.part])?.len(),
                };
                (base + ti.rel_offset, end.saturating_sub(base + ti.rel_offset))
            }
        };
        let tn = gguf_type_name(ti.ggml_type);
        let dtype = if tn == "unknown" { format!("ggml{}", ti.ggml_type) } else { tn.to_ascii_lowercase() };
        out.push(Span { name: name.clone(), dtype, shape: ti.dims.clone(), part: ti.part, offset, len });
    }
    // File order: one sequential pass over each part.
    out.sort_by_key(|s| (s.part, s.offset));
    Ok(out)
}

/// A part's bytes. Mapped on Unix, so hashing a multi-GB tensor allocates nothing;
/// read into one reused buffer elsewhere.
enum Source {
    #[cfg(unix)]
    Map { ptr: *mut libc::c_void, len: usize },
    Read { f: std::fs::File, buf: Vec<u8> },
}

impl Source {
    fn open(p: &Path) -> Result<Source> {
        let f = std::fs::File::open(p).with_context(|| format!("opening {}", p.display()))?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let len = f.metadata()?.len() as usize;
            if len > 0 {
                // SAFETY: a read-only private mapping of a file we hold open; unmapped
                // in Drop and only read through `slice`, bounds-checked.
                let ptr = unsafe {
                    libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, f.as_raw_fd(), 0)
                };
                if ptr != libc::MAP_FAILED {
                    unsafe { libc::madvise(ptr, len, libc::MADV_SEQUENTIAL) };
                    return Ok(Source::Map { ptr, len });
                }
            }
        }
        Ok(Source::Read { f, buf: Vec::new() })
    }

    fn slice(&mut self, offset: u64, len: u64) -> Result<&[u8]> {
        match self {
            #[cfg(unix)]
            Source::Map { ptr, len: size } => {
                let end = offset.checked_add(len).filter(|&e| e <= *size as u64);
                let Some(_) = end else { bail!("tensor at {offset}+{len} is past the end of the file") };
                Ok(unsafe { std::slice::from_raw_parts((*ptr as *const u8).add(offset as usize), len as usize) })
            }
            Source::Read { f, buf } => {
                use std::io::{Read, Seek, SeekFrom};
                buf.resize(len as usize, 0);
                f.seek(SeekFrom::Start(offset))?;
                f.read_exact(buf)?;
                Ok(buf)
            }
        }
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Source::Map { ptr, len } = self {
            unsafe { libc::munmap(*ptr, *len) };
        }
    }
}

#[derive(PartialEq, Debug)]
struct PartKey {
    path: String,
    size: u64,
    mtime_ns: u128,
    /// Unix (dev, inode, ctime ns); zeros elsewhere.
    unix: (u64, u64, i128),
}

fn cache_key(parts: &[PathBuf]) -> Result<Vec<PartKey>> {
    parts
        .iter()
        .map(|p| {
            let canon = std::fs::canonicalize(p).with_context(|| format!("resolving {}", p.display()))?;
            let md = std::fs::metadata(&canon)?;
            let mtime_ns = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            #[cfg(unix)]
            let unix = {
                use std::os::unix::fs::MetadataExt;
                (md.dev(), md.ino(), md.ctime() as i128 * 1_000_000_000 + md.ctime_nsec() as i128)
            };
            #[cfg(not(unix))]
            let unix = (0, 0, 0);
            Ok(PartKey { path: canon.to_string_lossy().into_owned(), size: md.len(), mtime_ns, unix })
        })
        .collect()
}

fn key_json(key: &[PartKey]) -> serde_json::Value {
    serde_json::Value::Array(
        key.iter()
            .map(|k| {
                serde_json::json!({
                    "path": k.path, "size": k.size, "mtime_ns": k.mtime_ns.to_string(),
                    "dev": k.unix.0, "ino": k.unix.1, "ctime_ns": k.unix.2.to_string(),
                })
            })
            .collect(),
    )
}

/// The cache file for `key`. The name only spreads entries out; the full key is
/// stored inside and compared on read, so a name collision is a miss, not a wrong id.
fn cache_path(key: &[PartKey]) -> Option<PathBuf> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key_json(key).to_string().hash(&mut h);
    let dir = ojas_core::config::cache_dir()?.join("swarm-identity");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join(format!("{:016x}.json", h.finish())))
}

fn cached(key: &[PartKey]) -> Option<Identity> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(cache_path(key)?).ok()?).ok()?;
    if v.get("key")? != &key_json(key) {
        return None;
    }
    serde_json::from_value(v.get("identity")?.clone()).ok()
}

/// Best effort: a cache that cannot be written costs a re-hash next time, nothing more.
/// Written to a temporary name and renamed, so a concurrent reader never sees half.
fn store(key: &[PartKey], id: &Identity) {
    let Some(p) = cache_path(key) else { return };
    let body = serde_json::json!({ "key": key_json(key), "identity": id });
    let tmp = p.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, body.to_string()).is_ok() && std::fs::rename(&tmp, &p).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}
