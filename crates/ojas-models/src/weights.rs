//! Weight cache: requantized weights stored page-aligned on disk, mmap'd at load
//! and wrapped in zero-copy Metal buffers.
//!
//! The "LLM in a flash" approach, in three parts:
//! - Instant model loads (no per-launch requant) plus OS-managed residency:
//!   file-backed pages are evictable, so big models no longer pin their full size.
//! - The dense skeleton is mlock'd/prewarmed while MoE routed experts
//!   ("_exps." tensors, ~94% of a qwen35moe file) stay lazy — the page cache is the
//!   expert cache.
//! - A temporal expert prefetcher madvise(WILLNEED)s the experts selected
//!   for the previous token ("windowing": strong temporal locality), pre-faulting
//!   them for the next token while the GPU is busy.
//!
//! File format v1 ("OJASWC01", little-endian), keyed to the source GGUF by size +
//! mtime. Header (64 B): magic[8], src_size u64, src_mtime u64, index_off u64,
//! entry_count u64. Data entries start at 16 KB (Apple Silicon page size) aligned
//! offsets; the index (name_len u32, name, kind u8, offset u64, len u64 per entry)
//! sits after the data. Metal's newBufferWithBytesNoCopy requires page-aligned
//! pointers and page-multiple lengths, hence the padding.

use anyhow::{bail, Result};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use ojas_metal::MetalGpu;
use metal::MTLResourceOptions;

/// Apple Silicon page size (16 KB) — required alignment for no-copy Metal buffers.
pub const PAGE: u64 = 16384;

const MAGIC: &[u8; 8] = b"OJASWC01";

/// Buffer kind codes (which engine map an entry belongs to).
pub const K_W4: u8 = 0;
pub const K_SCALE4: u8 = 1;
pub const K_W16: u8 = 2;
pub const K_W32: u8 = 3;

pub struct Entry {
    pub name: String,
    pub kind: u8,
    pub offset: u64,
    pub len: u64,
}

fn round_page(n: u64) -> u64 {
    (n + PAGE - 1) / PAGE * PAGE
}

/// `$OJAS_WEIGHTS_DIR/<file>-<size>-q4wc1.awc`, else `<cache_dir>/weights/…`.
/// The env override lets sandboxed harnesses (which redirect $HOME) keep the big
/// requant cache at the real home so it is built once and reused; otherwise every
/// run rebuilds an ~18 GB cache inside its throwaway sandbox.
pub fn cache_path(gguf_path: &str, src_size: u64) -> PathBuf {
    let dir = ojas_core::config::var("OJAS_WEIGHTS_DIR").ok().map(PathBuf::from).unwrap_or_else(|| {
        ojas_core::config::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("weights")
    });
    let base = std::path::Path::new(gguf_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".into());
    dir.join(format!("{base}-{src_size}-q4wc1.awc"))
}

pub fn source_meta(gguf_path: &str) -> Result<(u64, u64)> {
    let md = fs::metadata(gguf_path)?;
    let mtime = md
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok((md.len(), mtime))
}

// ---------------------------------------------------------------- writer

pub struct CacheWriter {
    file: File,
    path: PathBuf,
    src_size: u64,
    src_mtime: u64,
    entries: Vec<Entry>,
    off: u64, // next data offset (page-aligned)
}

impl CacheWriter {
    pub fn create(path: PathBuf, src_size: u64, src_mtime: u64) -> Result<Self> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        // write to a temp name; rename on finish so partial files are never "valid"
        let tmp = path.with_extension("awc.tmp");
        let file = File::create(&tmp)?;
        Ok(Self { file, path, src_size, src_mtime, entries: Vec::new(), off: PAGE })
    }

    /// Append one tensor blob at the next page-aligned offset (write-through — the
    /// bytes are not kept in RAM).
    pub fn add(&mut self, name: &str, kind: u8, bytes: &[u8]) -> Result<()> {
        self.file.seek(SeekFrom::Start(self.off))?;
        self.file.write_all(bytes)?;
        self.entries.push(Entry {
            name: name.to_string(),
            kind,
            offset: self.off,
            len: bytes.len() as u64,
        });
        self.off = round_page(self.off + bytes.len() as u64);
        Ok(())
    }

    pub fn finish(mut self) -> Result<PathBuf> {
        // pad the data region to a full page (no-copy buffers read page-multiples)
        self.file.set_len(self.off)?;
        // index after data
        let index_off = self.off;
        self.file.seek(SeekFrom::Start(index_off))?;
        let mut idx: Vec<u8> = Vec::new();
        for e in &self.entries {
            idx.extend_from_slice(&(e.name.len() as u32).to_le_bytes());
            idx.extend_from_slice(e.name.as_bytes());
            idx.push(e.kind);
            idx.extend_from_slice(&e.offset.to_le_bytes());
            idx.extend_from_slice(&e.len.to_le_bytes());
        }
        self.file.write_all(&idx)?;
        // header
        self.file.seek(SeekFrom::Start(0))?;
        let mut hdr: Vec<u8> = Vec::new();
        hdr.extend_from_slice(MAGIC);
        hdr.extend_from_slice(&self.src_size.to_le_bytes());
        hdr.extend_from_slice(&self.src_mtime.to_le_bytes());
        hdr.extend_from_slice(&index_off.to_le_bytes());
        hdr.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        self.file.write_all(&hdr)?;
        self.file.sync_all()?;
        fs::rename(self.path.with_extension("awc.tmp"), &self.path)?;
        Ok(self.path)
    }
}

// ---------------------------------------------------------------- mapped cache

pub struct MappedCache {
    ptr: *mut libc::c_void,
    len: u64,
    pub entries: Vec<Entry>,
}

// The mapping is read-only and lives for the program's lifetime.
unsafe impl Send for MappedCache {}
unsafe impl Sync for MappedCache {}

impl MappedCache {
    /// Open + validate against the source GGUF; None = missing/stale (caller builds).
    pub fn open(path: &PathBuf, src_size: u64, src_mtime: u64) -> Result<Option<Self>> {
        let mut f = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };
        let mut hdr = [0u8; 40];
        if f.read_exact(&mut hdr).is_err() || &hdr[0..8] != MAGIC {
            return Ok(None);
        }
        let rd = |o: usize| u64::from_le_bytes(hdr[o..o + 8].try_into().unwrap());
        if rd(8) != src_size || rd(16) != src_mtime {
            return Ok(None); // stale — source GGUF changed
        }
        let index_off = rd(24);
        let count = rd(32) as usize;
        // read index
        f.seek(SeekFrom::Start(index_off))?;
        let mut idx = Vec::new();
        f.read_to_end(&mut idx)?;
        let mut entries = Vec::with_capacity(count);
        let mut p = 0usize;
        for _ in 0..count {
            let nl = u32::from_le_bytes(idx[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            let name = String::from_utf8_lossy(&idx[p..p + nl]).to_string();
            p += nl;
            let kind = idx[p];
            p += 1;
            let offset = u64::from_le_bytes(idx[p..p + 8].try_into().unwrap());
            p += 8;
            let len = u64::from_le_bytes(idx[p..p + 8].try_into().unwrap());
            p += 8;
            entries.push(Entry { name, kind, offset, len });
        }
        let file_len = f.metadata()?.len();
        let map_len = index_off; // only the (page-aligned) data region is mapped
        if map_len == 0 || map_len % PAGE != 0 || map_len > file_len {
            bail!("corrupt weight cache {path:?}");
        }
        use std::os::unix::io::AsRawFd;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len as usize,
                libc::PROT_READ,
                libc::MAP_SHARED,
                f.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            bail!("mmap failed for {path:?}");
        }
        Ok(Some(Self { ptr, len: map_len, entries }))
    }

    pub fn base(&self) -> u64 {
        self.ptr as u64
    }

    /// Zero-copy Metal buffer over one entry (length rounded up to the page the
    /// on-disk layout already reserves).
    pub fn buffer(&self, gpu: &MetalGpu, e: &Entry) -> metal::Buffer {
        let ptr = unsafe { (self.ptr as *const u8).add(e.offset as usize) };
        gpu.device.new_buffer_with_bytes_no_copy(
            ptr as *const std::ffi::c_void,
            round_page(e.len),
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }

    /// Best-effort residency hints: mlock + WILLNEED (skeleton tensors).
    pub fn pin(&self, e: &Entry) {
        unsafe {
            let p = (self.ptr as *mut u8).add(e.offset as usize) as *mut libc::c_void;
            let l = round_page(e.len) as usize;
            libc::madvise(p, l, libc::MADV_WILLNEED);
            libc::mlock(p, l); // may fail on large regions — fine, WILLNEED still holds
        }
    }

    pub fn willneed(&self, offset: u64, len: u64) {
        unsafe {
            let p = (self.ptr as *mut u8).add(offset as usize) as *mut libc::c_void;
            libc::madvise(p, len as usize, libc::MADV_WILLNEED);
        }
    }
}

impl Drop for MappedCache {
    fn drop(&mut self) {
        // Metal no-copy buffers reference this mapping; DecoderGpu owns both and drops
        // them together at program end, so unmapping here is safe in practice.
        unsafe { libc::munmap(self.ptr, self.len as usize) };
    }
}

// ---- Direct-GGUF mmap streaming (no requant cache) --------------------------
// Streams expert weights straight from the mmap'd GGUF shards: zero-copy, no second
// on-disk copy, faithful precision (kernels read the native Q4_K/Q8 blocks). GGUF
// tensors are only 32-byte aligned, so no-copy buffers are made over the page-aligned
// region containing the tensor and the byte offset is passed to Metal's set_buffer.
pub struct MappedGguf {
    shards: Vec<(*mut libc::c_void, u64)>, // (mmap ptr, mmap len) per shard part
    _files: Vec<File>,                     // kept open so fds stay valid for pread()
    fds: Vec<i32>,                         // raw fd per shard (positioned pread, no fault serialization)
    /// One Metal buffer over each whole shard mmap, created lazily and cached.
    /// llama.cpp's layout: every tensor is a `(shard_buffer, offset)` view rather than
    /// its own `newBufferWithBytesNoCopy`, so a resident model wires ~2 large buffers
    /// instead of ~150 small ones — the many-small-buffer set OOMs at the same bytes.
    shard_bufs: std::sync::Mutex<Vec<Option<metal::Buffer>>>,
}
unsafe impl Send for MappedGguf {}
unsafe impl Sync for MappedGguf {}

impl MappedGguf {
    /// mmap each shard file (whole file, read-only) and keep its fd open for pread.
    pub fn open(paths: &[std::path::PathBuf]) -> Result<Self> {
        Self::open_with(paths, true)
    }

    /// Same, but leaving the page cache enabled and asking the OS to read ahead.
    ///
    /// `open` sets `F_NOCACHE`, which is correct for the streaming-expert path: it
    /// keeps routed experts out of the page cache so they live only in the engine's
    /// own bounded cache. It is wrong when the whole model is about to be touched once,
    /// sequentially — every fault then goes to disk with no readahead.
    pub fn open_cached(paths: &[std::path::PathBuf]) -> Result<Self> {
        let mg = Self::open_with(paths, false)?;
        // One WILLNEED over each whole mapping turns 434 scattered faults into
        // sequential readahead.
        for (ptr, len) in &mg.shards {
            unsafe { libc::madvise(*ptr, *len as usize, libc::MADV_WILLNEED) };
        }
        Ok(mg)
    }

    /// Same as `open`, but over files somebody else opened.
    ///
    /// A sandboxed process cannot call `open()` on a model path: the macOS seatbelt
    /// profile for model execution grants Metal and the shader compiler but no
    /// filesystem. Everything after the open (mmap, pread, fcntl, madvise) acts on a
    /// descriptor and is permitted, so the host opens the shards and passes them in.
    pub fn from_files(files: Vec<File>) -> Result<Self> {
        Self::from_files_with(files, true)
    }

    /// `from_files`, leaving the page cache on and asking for readahead.
    pub fn from_files_cached(files: Vec<File>) -> Result<Self> {
        let mg = Self::from_files_with(files, false)?;
        for (ptr, len) in &mg.shards {
            unsafe { libc::madvise(*ptr, *len as usize, libc::MADV_WILLNEED) };
        }
        Ok(mg)
    }

    fn open_with(paths: &[std::path::PathBuf], nocache: bool) -> Result<Self> {
        let mut files = Vec::with_capacity(paths.len());
        for p in paths {
            files.push(File::open(p)?);
        }
        Self::from_files_with(files, nocache)
    }

    fn from_files_with(files: Vec<File>, nocache: bool) -> Result<Self> {
        use std::os::unix::io::AsRawFd;
        let mut shards = Vec::with_capacity(files.len());
        let mut fds = Vec::with_capacity(files.len());
        for f in &files {
            let len = f.metadata()?.len();
            let map_len = round_page(len);
            let ptr = unsafe {
                libc::mmap(std::ptr::null_mut(), map_len as usize, libc::PROT_READ,
                           libc::MAP_SHARED, f.as_raw_fd(), 0)
            };
            if ptr == libc::MAP_FAILED { bail!("mmap failed for shard fd {}", f.as_raw_fd()); }
            // F_NOCACHE: pread on these fds skips the OS page cache, so cached experts
            // live only in our userspace ExpertCache, never stored twice.
            if nocache {
                // F_NOCACHE is Darwin's "do not keep this in the unified buffer cache";
                // there is no portable equivalent, the Metal loader cannot run off macOS
                // anyway, and skipping it elsewhere costs only a copy nothing reads.
                #[cfg(target_os = "macos")]
                unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1); }
            }
            shards.push((ptr, map_len));
            fds.push(f.as_raw_fd());
        }
        let n = shards.len();
        Ok(Self { shards, _files: files, fds, shard_bufs: std::sync::Mutex::new(vec![None; n]) })
    }

    /// Raw fd for a shard (valid for the MappedGguf lifetime — `files` keeps them open).
    /// pread(fd, buf, off) bypasses the mmap fault path (which serializes on the VMA lock,
    /// capping parallel memcpy ~800MB/s) → parallel positioned reads saturate the SSD.
    pub fn fd(&self, part: usize) -> i32 { self.fds[part] }

    /// One Metal buffer over the whole shard (llama.cpp layout), created once and
    /// cached. A shard is up to `max_buffer_length` here (Flash's largest is
    /// 49.8 GB < 54 GB), so one buffer per shard suffices; a shard exceeding the
    /// cap would need chunking, which no supported model triggers.
    pub fn shard_buffer(&self, gpu: &MetalGpu, part: usize) -> metal::Buffer {
        let mut cache = self.shard_bufs.lock().unwrap();
        if let Some(b) = &cache[part] { return b.clone(); }
        let (ptr, len) = self.shards[part];
        assert!(len <= gpu.device.max_buffer_length(),
            "shard {part} is {len} bytes, over maxBufferLength {} — needs chunking",
            gpu.device.max_buffer_length());
        let buf = gpu.device.new_buffer_with_bytes_no_copy(
            ptr as *const std::ffi::c_void, len as u64,
            MTLResourceOptions::StorageModeShared, None);
        cache[part] = Some(buf.clone());
        buf
    }

    /// A tensor as a view into its whole-shard buffer: `(shard_buffer, abs_offset)`.
    /// The offset is the tensor's absolute file position, passed to set_buffer. The
    /// shard buffer starts at the (page-aligned) mmap base, so no per-tensor
    /// alignment slop is needed.
    pub fn view(&self, gpu: &MetalGpu, part: usize, abs_offset: u64) -> (metal::Buffer, u64) {
        (self.shard_buffer(gpu, part), abs_offset)
    }

    /// Distinct whole-shard buffers that currently exist, for a residency set.
    pub fn live_shard_buffers(&self) -> Vec<metal::Buffer> {
        self.shard_bufs.lock().unwrap().iter().filter_map(|b| b.clone()).collect()
    }

    /// Zero-copy Metal buffer over the tensor at (part, abs_offset, len).
    /// Returns (buffer, byte_offset_into_buffer) — pass the offset to set_buffer.
    pub fn buffer(&self, gpu: &MetalGpu, part: usize, abs_offset: u64, len: u64) -> (metal::Buffer, u64) {
        let length = self.buffer_length(part, abs_offset, len).expect("invalid mapped tensor extent");
        let base = self.shards[part].0 as usize;
        let addr = base + abs_offset as usize;
        let aligned = addr & !((PAGE as usize) - 1);
        let off_in = (addr - aligned) as u64;
        let buf = gpu.device.new_buffer_with_bytes_no_copy(
            aligned as *const std::ffi::c_void,
            length,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        (buf, off_in)
    }

    /// Exact allocation length without constructing a Metal buffer. Residency
    /// preflight must include page slack, not just the tensor's packed bytes.
    pub fn buffer_length(&self, part: usize, abs_offset: u64, len: u64) -> Result<u64> {
        let (_, mapped_len) = self.shards.get(part).ok_or_else(|| anyhow::anyhow!("missing shard {part}"))?;
        anyhow::ensure!(len > 0 && abs_offset.checked_add(len).is_some_and(|end| end <= *mapped_len),
            "tensor extent exceeds mapped shard {part}");
        let bytes = (abs_offset % PAGE).checked_add(len).ok_or_else(|| anyhow::anyhow!("tensor extent overflow"))?;
        anyhow::ensure!(bytes <= u64::MAX - (PAGE - 1), "page-rounded tensor length overflow");
        Ok(round_page(bytes))
    }

    /// Raw mmap base pointers per shard part (stable for the mmap lifetime). Used by
    /// the streaming expert prefetcher/pinner to madvise ranges from a background thread.
    pub fn shard_bases(&self) -> Vec<u64> { self.shards.iter().map(|(p, _)| *p as u64).collect() }

    /// Total bytes mapped across all shards (the whole model file). Full residency
    /// references whole-shard buffers, so this is what must fit the working set.
    pub fn total_mapped_bytes(&self) -> u64 { self.shards.iter().map(|(_, l)| *l).sum() }

    /// Raw pointer into a shard's mmap at absolute file offset `abs` (for CPU gather of
    /// routed experts). Reading it faults the underlying file pages in.
    pub fn ptr(&self, part: usize, abs: u64) -> *const u8 {
        (self.shards[part].0 as *const u8).wrapping_add(abs as usize)
    }

    /// Prefetch a tensor's pages into RAM (skeleton pin / expert readahead).
    pub fn willneed(&self, part: usize, abs_offset: u64, len: u64, lock: bool) {
        unsafe {
            let addr = (self.shards[part].0 as usize) + abs_offset as usize;
            let aligned = addr & !((PAGE as usize) - 1);
            let l = round_page((addr - aligned) as u64 + len) as usize;
            libc::madvise(aligned as *mut libc::c_void, l, libc::MADV_WILLNEED);
            if lock { libc::mlock(aligned as *mut libc::c_void, l); }
        }
    }
}

impl Drop for MappedGguf {
    fn drop(&mut self) {
        for (ptr, len) in &self.shards { unsafe { libc::munmap(*ptr, *len as usize); } }
    }
}

// ---------------------------------------------------------------- prefetcher

/// One routed expert's byte ranges within the mapped file (gate/up/down slices).
#[derive(Clone, Copy)]
pub struct ExpertRegion {
    pub gate_off: u64,
    pub up_off: u64,
    pub down_off: u64,
    pub gu_bytes: u64,   // per-expert gate/up slice (nibbles)
    pub down_bytes: u64, // per-expert down slice (nibbles)
}

/// Temporal expert prefetcher: after each token, the engine sends the
/// (layer, expert) pairs that were routed; a background thread madvise(WILLNEED)s
/// their weight ranges so the next token's likely experts are resident before the
/// GPU needs them (experts show strong temporal locality across tokens).
pub struct ExpertPrefetcher {
    tx: std::sync::mpsc::Sender<Vec<(u32, u32)>>,
}

impl ExpertPrefetcher {
    /// `regions[layer]` = per-layer expert layout (None for attention layers).
    pub fn new(base: u64, regions: Vec<Option<ExpertRegion>>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<(u32, u32)>>();
        std::thread::spawn(move || {
            for batch in rx {
                for (l, e) in batch {
                    if let Some(Some(r)) = regions.get(l as usize) {
                        let e = e as u64;
                        unsafe {
                            let ad = |off: u64, len: u64| {
                                let a0 = (base + off) / PAGE * PAGE;             // align down
                                let a1 = (base + off + len + PAGE - 1) / PAGE * PAGE;
                                libc::madvise(a0 as *mut libc::c_void, (a1 - a0) as usize, libc::MADV_WILLNEED);
                            };
                            ad(r.gate_off + e * r.gu_bytes, r.gu_bytes);
                            ad(r.up_off + e * r.gu_bytes, r.gu_bytes);
                            ad(r.down_off + e * r.down_bytes, r.down_bytes);
                        }
                    }
                }
            }
        });
        Self { tx }
    }

    pub fn note(&self, used: Vec<(u32, u32)>) {
        let _ = self.tx.send(used);
    }
}

// ---------------------------------------------------------------- streaming prefetch + pinning

/// One MoE layer's expert byte layout inside the mmap'd GGUF shards (streaming path).
/// Offsets are absolute file offsets (== offset into the shard mmap, mapped from 0).
#[derive(Clone, Copy)]
pub struct StreamRegion {
    pub part: usize,        // which shard mmap the experts live in
    pub gate_abs: u64,      // absolute offset of ffn_gate_exps.weight
    pub up_abs: u64,        // absolute offset of ffn_up_exps.weight
    pub down_abs: u64,      // absolute offset of ffn_down_exps.weight
    pub gu_stride: u64,     // per-expert bytes of gate (== up)
    pub down_stride: u64,   // per-expert bytes of down
}

impl StreamRegion {
    #[inline] fn bytes(&self) -> u64 { self.gu_stride * 2 + self.down_stride }
}

/// Streaming expert prefetcher + frequency pinner (GLM/DeepSeek disk-streamed MoE).
///
/// After each token the engine sends the routed (layer, expert) pairs. A background
/// thread then, for the whole token at once:
///   1. madvise(WILLNEED) every predicted expert's gate/up/down ranges across all
///      layers (temporal prediction: the experts a token used are ~90% of what the
///      next token uses), so the OS pulls them from disk while the CPU samples and
///      the GPU works through the early layers. Metal wires no-copy pages resident
///      at commit, so this shrinks that synchronous stall.
///   2. frequency-pins the hottest experts with mlock into a RAM budget (`pin_budget`),
///      evicting the coldest pinned expert when a hotter one appears. Pinned experts
///      never fault → 0 disk bytes on hit. Inert when pin_budget == 0.
pub struct StreamPrefetcher {
    tx: std::sync::mpsc::Sender<Vec<(u32, u32)>>,
}

impl StreamPrefetcher {
    pub fn new(bases: Vec<u64>, regions: Vec<Option<StreamRegion>>, pin_budget: u64) -> Self {
        use std::collections::HashMap;
        let (tx, rx) = std::sync::mpsc::channel::<Vec<(u32, u32)>>();
        std::thread::spawn(move || {
            // advise(base+off .. +len) with MADV_WILLNEED (prefetch) or mlock/munlock (pin).
            let advise = |base: u64, off: u64, len: u64, mode: u8| unsafe {
                let a0 = (base + off) / PAGE * PAGE;
                let a1 = (base + off + len + PAGE - 1) / PAGE * PAGE;
                let p = a0 as *mut libc::c_void;
                let l = (a1 - a0) as usize;
                match mode {
                    0 => { libc::madvise(p, l, libc::MADV_WILLNEED); }
                    1 => { libc::mlock(p, l); }
                    _ => { libc::munlock(p, l); }
                }
            };
            let touch = |r: &StreamRegion, e: u64, mode: u8| {
                let b = bases[r.part];
                advise(b, r.gate_abs + e * r.gu_stride, r.gu_stride, mode);
                advise(b, r.up_abs + e * r.gu_stride, r.gu_stride, mode);
                advise(b, r.down_abs + e * r.down_stride, r.down_stride, mode);
            };
            let mut freq: HashMap<(u32, u32), u64> = HashMap::new();
            let mut pinned: HashMap<(u32, u32), u64> = HashMap::new(); // (l,e) -> bytes
            let mut pinned_bytes: u64 = 0;
            // OJAS_EXPERT_STATS=<path>: dump the (layer,expert)->hits histogram every
            // 32 batches, for usage-aware expert packing.
            let stats_path = ojas_core::config::var("OJAS_EXPERT_STATS").ok();
            let mut batches: u64 = 0;
            for batch in rx {
                // 1. prefetch every predicted expert for the next token.
                for &(l, e) in &batch {
                    if let Some(Some(r)) = regions.get(l as usize) { touch(r, e as u64, 0); }
                }
                if let Some(sp) = &stats_path {
                    for &(l, e) in &batch { *freq.entry((l, e)).or_insert(0) += 1; }
                    batches += 1;
                    if batches % 32 == 0 {
                        let mut out = String::with_capacity(freq.len() * 16);
                        out.push_str(&format!("# batches {batches}\nlayer,expert,hits\n"));
                        let mut rows: Vec<_> = freq.iter().collect();
                        rows.sort();
                        for (&(l, e), &c) in rows { out.push_str(&format!("{l},{e},{c}\n")); }
                        let _ = std::fs::write(sp, out);
                    }
                }
                if pin_budget == 0 { continue; }
                // 2. frequency-pin the hottest experts within the RAM budget.
                for &(l, e) in &batch {
                    let key = (l, e);
                    if stats_path.is_none() { *freq.entry(key).or_insert(0) += 1; }
                    if pinned.contains_key(&key) { continue; }
                    let r = match regions.get(l as usize) { Some(Some(r)) => r, _ => continue };
                    let bytes = r.bytes();
                    if pinned_bytes + bytes <= pin_budget {
                        touch(r, e as u64, 1);
                        pinned.insert(key, bytes);
                        pinned_bytes += bytes;
                    } else {
                        // evict the coldest pinned expert if this candidate is hotter.
                        let cand_f = freq[&key];
                        if let Some((&vk, _)) = pinned.iter().min_by_key(|(k, _)| freq.get(*k).copied().unwrap_or(0)) {
                            if freq.get(&vk).copied().unwrap_or(0) < cand_f {
                                if let Some(Some(vr)) = regions.get(vk.0 as usize) {
                                    touch(vr, vk.1 as u64, 2);
                                    pinned_bytes -= pinned.remove(&vk).unwrap_or(0);
                                }
                                touch(r, e as u64, 1);
                                pinned.insert(key, bytes);
                                pinned_bytes += bytes;
                            }
                        }
                    }
                }
            }
        });
        Self { tx }
    }

    pub fn note(&self, used: Vec<(u32, u32)>) { let _ = self.tx.send(used); }
}
