//! The prompt-prefix cache's directory: one file per cached payload, an index that
//! lets startup rebuild the block tree without reading a payload, and a background
//! writer, so saving never delays a request.
//!
//! Each build of a model (weights, precision, state layout) gets its own
//! subdirectory, named by a salt, so one directory serves several models and a
//! block is never restored into a model that did not produce it:
//!
//! ```text
//! <dir>/<salt>/index       every block on disk: key, parent, tokens, use counts, pins, sizes
//! <dir>/<salt>/<key>.kv    a block's attention KV rows
//! <dir>/<salt>/<key>.snap  the recurrent-state snapshot after the block
//! <dir>/<salt>/model       which model the subdirectory belongs to
//! <dir>/<salt>/lock        held by the one process that writes
//! ```
//!
//! Payloads are 8 to 50 MB, so a file each keeps every read sequential and makes
//! eviction a single unlink. A payload file repeats its key, parent and tokens and
//! carries a checksum; all of it is verified before the payload is used. Payloads
//! and the index are written to a temporary file and renamed into place, so a crash
//! leaves either the old file or the new one.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

/// A payload shared between the cache and the writer.
pub(crate) type Bytes = Arc<Vec<u8>>;

const PAYLOAD_MAGIC: u64 = u64::from_le_bytes(*b"OJPCBLK1");
const INDEX_MAGIC: u64 = u64::from_le_bytes(*b"OJPCIDX1");

/// Payloads read at once when a restore needs several from disk. Reads and
/// checksums overlap, so a restore runs close to the drive's throughput.
const PARALLEL_READS: usize = 8;

/// Bytes of payload queued for writing at most; a write beyond it is skipped
/// rather than held in memory.
const MAX_PENDING: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Part {
    Kv,
    Snapshot,
}

impl Part {
    fn extension(self) -> &'static str {
        match self {
            Part::Kv => "kv",
            Part::Snapshot => "snap",
        }
    }
}

/// One block as the index stores it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Record {
    pub(crate) key: u64,
    pub(crate) parent: u64,
    pub(crate) tokens: Box<[u32]>,
    pub(crate) uses: u32,
    pub(crate) snapshot_uses: u32,
    pub(crate) last_used: u64,
    pub(crate) kv_len: u64,
    /// Zero when the block's snapshot is not on disk.
    pub(crate) snapshot_len: u64,
    pub(crate) pinned: bool,
}

/// How the directory stores KV payloads. Snapshots are always stored exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum KvFormat {
    /// The f16 rows as computed: a restore is bit-identical to processing.
    #[default]
    F16,
    /// Q8_0, about half the size; a restore is close to, not identical with,
    /// processing.
    Q8,
}

/// How a directory is opened.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DiskOptions {
    pub(crate) readonly: bool,
    /// Free space every write must leave on the volume.
    pub(crate) reserve: u64,
    /// The length every record's token list must have.
    pub(crate) block_tokens: usize,
    pub(crate) kv: KvFormat,
}

/// A payload to read back, and what it must match.
pub(crate) struct Wanted<'a> {
    pub(crate) key: u64,
    pub(crate) part: Part,
    pub(crate) parent: u64,
    pub(crate) tokens: &'a [u32],
    pub(crate) len: usize,
}

/// The outcome of a queued write, named by the number `Disk::write` gave it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Written {
    pub(crate) key: u64,
    pub(crate) part: Part,
    pub(crate) seq: u64,
    pub(crate) ok: bool,
}

enum Job {
    Put { key: u64, part: Part, seq: u64, parent: u64, tokens: Box<[u32]>, data: Bytes },
    Remove(PathBuf),
    Index,
    Flush(mpsc::Sender<()>),
}

/// State the cache and the writer thread share.
struct Shared {
    dir: PathBuf,
    reserve: u64,
    kv: KvFormat,
    pending: AtomicU64,
    /// Number of the last write queued.
    seq: AtomicU64,
    /// The newest index to write; older ones still queued are superseded.
    index: Mutex<Option<Vec<u8>>>,
    written: Mutex<Vec<Written>>,
}

/// An open cache directory.
pub(crate) struct Disk {
    shared: Arc<Shared>,
    /// The job queue and the writer thread; absent when read-only.
    writer: Option<(mpsc::Sender<Job>, JoinHandle<()>)>,
    _lock: Option<File>,
}

/// A directory as `Disk::open` found it.
pub(crate) struct Opened {
    pub(crate) disk: Disk,
    /// The index's blocks whose payload files are present, parents before children.
    pub(crate) records: Vec<Record>,
    pub(crate) clock: u64,
}

impl Disk {
    /// Open `root/<salt>`, creating it unless read-only. The directory is opened
    /// read-only when asked, or when another process already writes to it. In
    /// read-write mode, payload files the index does not reference are removed.
    pub(crate) fn open(root: &Path, salt: u64, about: &str, options: DiskOptions) -> io::Result<Opened> {
        let DiskOptions { readonly, reserve, block_tokens, kv } = options;
        let dir = root.join(format!("{salt:016x}"));
        let lock = if readonly {
            None
        } else {
            fs::create_dir_all(&dir)?;
            let lock = lock(&dir)?;
            if lock.is_none() {
                tracing::warn!(target: "prefix", "{} is in use by another process; opening it read-only", dir.display());
            }
            lock
        };
        let (records, clock) = fs::read(dir.join("index")).ok()
            .and_then(|bytes| decode_index(&bytes, block_tokens))
            .unwrap_or_default();
        let records = present(&dir, records, kv);
        let shared = Arc::new(Shared {
            dir, reserve, kv, pending: AtomicU64::new(0), seq: AtomicU64::new(0), index: Mutex::new(None), written: Mutex::new(Vec::new()),
        });
        let writer = match lock {
            Some(_) => {
                fs::write(shared.dir.join("model"), about)?;
                sweep(&shared.dir, &records)?;
                let (tx, rx) = mpsc::channel();
                let s = Arc::clone(&shared);
                let handle = std::thread::Builder::new().name("prefix-cache-writer".into())
                    .spawn(move || run_writer(&s, rx))?;
                Some((tx, handle))
            }
            None => None,
        };
        Ok(Opened { disk: Disk { shared, writer, _lock: lock }, records, clock })
    }

    pub(crate) fn writable(&self) -> bool { self.writer.is_some() }

    pub(crate) fn dir(&self) -> &Path { &self.shared.dir }

    /// Bytes a payload of `len` bytes takes in the directory.
    pub(crate) fn stored_len(&self, part: Part, len: usize) -> usize { stored_len(self.shared.kv, part, len) }

    /// Read a payload and verify it is the one the index describes.
    pub(crate) fn read(&self, key: u64, part: Part, parent: u64, tokens: &[u32], len: usize) -> io::Result<Vec<u8>> {
        let mut file = File::open(self.shared.path(key, part))?;
        bypass_page_cache(&file);
        let mut head = vec![0; header_len(tokens.len())];
        file.read_exact(&mut head)?;
        let mut data = vec![0; self.stored_len(part, len)];
        file.read_exact(&mut data)?;
        if head != header(key, part, parent, tokens, &data) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "payload does not match the index"));
        }
        Ok(match (part, self.shared.kv) {
            (Part::Kv, KvFormat::Q8) => q8_decode(&data),
            _ => data,
        })
    }

    /// Read several payloads at once; results are in the order of `wanted`.
    pub(crate) fn read_many(&self, wanted: &[Wanted<'_>]) -> Vec<io::Result<Vec<u8>>> {
        let mut results: Vec<io::Result<Vec<u8>>> = Vec::with_capacity(wanted.len());
        std::thread::scope(|scope| {
            let per = wanted.len().div_ceil(PARALLEL_READS).max(1);
            let handles: Vec<_> = wanted.chunks(per).map(|batch| scope.spawn(move || {
                batch.iter().map(|w| self.read(w.key, w.part, w.parent, w.tokens, w.len)).collect::<Vec<_>>()
            })).collect();
            for (h, batch) in handles.into_iter().zip(wanted.chunks(per)) {
                match h.join() {
                    Ok(read) => results.extend(read),
                    Err(_) => results.extend(batch.iter().map(|_| Err(io::Error::other("read thread failed")))),
                }
            }
        });
        results
    }

    /// Queue a payload for writing, returning the number its `Written` will carry.
    /// `None`, queueing nothing, when the directory is read-only or the queue is
    /// full.
    pub(crate) fn write(&self, key: u64, part: Part, parent: u64, tokens: &[u32], data: Bytes) -> Option<u64> {
        let (tx, _) = self.writer.as_ref()?;
        let len = data.len() as u64;
        if self.shared.pending.fetch_add(len, Ordering::Relaxed) + len > MAX_PENDING {
            self.shared.pending.fetch_sub(len, Ordering::Relaxed);
            return None;
        }
        let seq = self.shared.seq.fetch_add(1, Ordering::Relaxed) + 1;
        tx.send(Job::Put { key, part, seq, parent, tokens: tokens.into(), data }).ok().map(|_| seq)
    }

    /// Delete a payload, after any write of it already queued.
    pub(crate) fn remove(&self, key: u64, part: Part) {
        if let Some((tx, _)) = &self.writer {
            let _ = tx.send(Job::Remove(self.shared.path(key, part)));
        }
    }

    /// Replace the index, after the payload writes already queued.
    pub(crate) fn save_index(&self, records: &[Record], clock: u64) {
        let Some((tx, _)) = &self.writer else { return };
        *self.shared.index.lock().unwrap() = Some(encode_index(records, clock));
        let _ = tx.send(Job::Index);
    }

    /// Writes finished since the last call.
    pub(crate) fn written(&self) -> Vec<Written> { std::mem::take(&mut *self.shared.written.lock().unwrap()) }

    /// Wait until every job queued so far is done.
    pub(crate) fn flush(&self) {
        let Some((tx, _)) = &self.writer else { return };
        let (ack, done) = mpsc::channel();
        if tx.send(Job::Flush(ack)).is_ok() {
            let _ = done.recv();
        }
    }
}

impl Drop for Disk {
    fn drop(&mut self) {
        if let Some((tx, handle)) = self.writer.take() {
            drop(tx);
            let _ = handle.join();
        }
    }
}

impl Shared {
    fn path(&self, key: u64, part: Part) -> PathBuf {
        self.dir.join(format!("{key:016x}.{}", part.extension()))
    }

    fn put(&self, key: u64, part: Part, parent: u64, tokens: &[u32], data: &[u8]) -> io::Result<()> {
        let encoded;
        let data = match (part, self.kv) {
            (Part::Kv, KvFormat::Q8) => {
                encoded = q8_encode(data);
                &encoded[..]
            }
            _ => data,
        };
        let head = header(key, part, parent, tokens, data);
        if free_space(&self.dir) < self.reserve + (head.len() + data.len()) as u64 {
            return Err(io::Error::new(io::ErrorKind::StorageFull, "the volume is at its free-space reserve"));
        }
        replace(&self.path(key, part), &[&head, data])
    }
}

fn run_writer(shared: &Shared, jobs: mpsc::Receiver<Job>) {
    for job in jobs {
        match job {
            Job::Put { key, part, seq, parent, tokens, data } => {
                let result = shared.put(key, part, parent, &tokens, &data);
                if let Err(e) = &result {
                    tracing::warn!(target: "prefix", "could not save a cached block: {e}");
                }
                shared.pending.fetch_sub(data.len() as u64, Ordering::Relaxed);
                shared.written.lock().unwrap().push(Written { key, part, seq, ok: result.is_ok() });
            }
            Job::Remove(path) => {
                let _ = fs::remove_file(path);
            }
            Job::Index => {
                let latest = shared.index.lock().unwrap().take();
                if let Some(bytes) = latest {
                    if let Err(e) = replace(&shared.dir.join("index"), &[&bytes]) {
                        tracing::warn!(target: "prefix", "could not save the cache index: {e}");
                    }
                }
            }
            Job::Flush(ack) => {
                let _ = ack.send(());
            }
        }
    }
}

/// Write `parts` to `path` through a temporary file, so readers see the old file
/// or the complete new one.
fn replace(path: &Path, parts: &[&[u8]]) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut file = File::create(&tmp)?;
    for part in parts {
        file.write_all(part)?;
    }
    file.sync_data()?;
    fs::rename(&tmp, path)
}

fn header_len(tokens: usize) -> usize { 40 + 4 * tokens }

/// A payload file's header: what the payload is, and the checksum of its bytes.
fn header(key: u64, part: Part, parent: u64, tokens: &[u32], data: &[u8]) -> Vec<u8> {
    let mut h = Vec::with_capacity(header_len(tokens.len()));
    h.extend_from_slice(&PAYLOAD_MAGIC.to_le_bytes());
    h.extend_from_slice(&key.to_le_bytes());
    h.extend_from_slice(&parent.to_le_bytes());
    h.extend_from_slice(&(part as u32).to_le_bytes());
    h.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    h.extend_from_slice(&checksum(data).to_le_bytes());
    for t in tokens {
        h.extend_from_slice(&t.to_le_bytes());
    }
    h
}

fn encode_index(records: &[Record], clock: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for v in [INDEX_MAGIC, clock, records.len() as u64] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for r in records {
        for v in [r.key, r.parent, r.last_used, r.kv_len, r.snapshot_len] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in [r.uses, r.snapshot_uses, u32::from(r.pinned), r.tokens.len() as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for t in &r.tokens {
            out.extend_from_slice(&t.to_le_bytes());
        }
    }
    out.extend_from_slice(&checksum(&out).to_le_bytes());
    out
}

/// The records and clock of an index, or `None` if it is damaged or describes
/// blocks of another length.
fn decode_index(bytes: &[u8], block_tokens: usize) -> Option<(Vec<Record>, u64)> {
    let (body, sum) = bytes.split_at_checked(bytes.len().checked_sub(8)?)?;
    if checksum(body) != u64::from_le_bytes(sum.try_into().ok()?) { return None; }
    let mut r = Cursor(body);
    if r.u64()? != INDEX_MAGIC { return None; }
    let clock = r.u64()?;
    let n = r.u64()?;
    let mut records = Vec::new();
    for _ in 0..n {
        let (key, parent, last_used, kv_len, snapshot_len) = (r.u64()?, r.u64()?, r.u64()?, r.u64()?, r.u64()?);
        let (uses, snapshot_uses, pinned, ntok) = (r.u32()?, r.u32()?, r.u32()? != 0, r.u32()? as usize);
        if ntok != block_tokens { return None; }
        let tokens = (0..ntok).map(|_| r.u32()).collect::<Option<Box<[u32]>>>()?;
        records.push(Record { key, parent, tokens, uses, snapshot_uses, last_used, kv_len, snapshot_len, pinned });
    }
    r.0.is_empty().then_some((records, clock))
}

struct Cursor<'a>(&'a [u8]);

impl Cursor<'_> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (head, rest) = self.0.split_first_chunk::<N>()?;
        self.0 = rest;
        Some(*head)
    }
    fn u64(&mut self) -> Option<u64> { self.take().map(u64::from_le_bytes) }
    fn u32(&mut self) -> Option<u32> { self.take().map(u32::from_le_bytes) }
}

/// The records whose payload files exist at their recorded sizes and whose parent
/// survives too. A missing snapshot file drops only the snapshot.
fn present(dir: &Path, records: Vec<Record>, kv: KvFormat) -> Vec<Record> {
    let size = |r: &Record, part: Part, len: u64| {
        let path = dir.join(format!("{:016x}.{}", r.key, part.extension()));
        let expect = stored_len(kv, part, len as usize) + header_len(r.tokens.len());
        fs::metadata(path).is_ok_and(|m| m.len() == expect as u64)
    };
    let mut kept = std::collections::HashSet::from([super::prefix_cache::ROOT]);
    let mut out = Vec::with_capacity(records.len());
    for mut r in records {
        if !kept.contains(&r.parent) || !size(&r, Part::Kv, r.kv_len) { continue; }
        if r.snapshot_len > 0 && !size(&r, Part::Snapshot, r.snapshot_len) {
            r.snapshot_len = 0;
            r.snapshot_uses = 0;
        }
        kept.insert(r.key);
        out.push(r);
    }
    out
}

/// Remove payload and temporary files that `records` do not account for.
fn sweep(dir: &Path, records: &[Record]) -> io::Result<()> {
    let mut keep = std::collections::HashSet::new();
    for r in records {
        keep.insert(format!("{:016x}.kv", r.key));
        if r.snapshot_len > 0 { keep.insert(format!("{:016x}.snap", r.key)); }
    }
    for entry in fs::read_dir(dir)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        let payload = name.ends_with(".kv") || name.ends_with(".snap") || name.ends_with(".tmp");
        if payload && !keep.contains(&name) {
            let _ = fs::remove_file(dir.join(&name));
        }
    }
    Ok(())
}

/// Read a payload straight from the drive: the cache keeps its own RAM copy, so
/// the OS page cache would only hold a second one.
#[cfg(target_os = "macos")]
fn bypass_page_cache(file: &File) {
    use std::os::fd::AsRawFd;
    unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
}

#[cfg(not(target_os = "macos"))]
fn bypass_page_cache(_file: &File) {}

#[cfg(unix)]
fn lock(dir: &Path) -> io::Result<Option<File>> {
    use std::os::fd::AsRawFd;
    let file = File::options().create(true).truncate(false).write(true).open(dir.join("lock"))?;
    let held = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0;
    Ok(held.then_some(file))
}

/// Without `flock` the directory is not locked: two processes writing it at once
/// are not detected.
#[cfg(not(unix))]
fn lock(dir: &Path) -> io::Result<Option<File>> {
    File::options().create(true).truncate(false).write(true).open(dir.join("lock")).map(Some)
}

/// Bytes available to this process on the volume holding `path`.
#[cfg(unix)]
pub(crate) fn free_space(path: &Path) -> u64 {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) else { return 0 };
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 { return 0; }
    (s.f_bavail as u64).saturating_mul(s.f_frsize as u64)
}

#[cfg(not(unix))]
pub(crate) fn free_space(_path: &Path) -> u64 { u64::MAX }

fn stored_len(kv: KvFormat, part: Part, len: usize) -> usize {
    match (part, kv) {
        (Part::Kv, KvFormat::Q8) => len / 64 * 34,
        _ => len,
    }
}

/// Encode f16 values as Q8_0: per 32 values, an f16 scale and 32 signed bytes.
/// The length must be a multiple of 64 bytes, as KV rows are.
fn q8_encode(data: &[u8]) -> Vec<u8> {
    debug_assert!(data.len().is_multiple_of(64), "KV rows are whole groups of 32 halves");
    let mut out = Vec::with_capacity(data.len() / 64 * 34);
    for group in data.as_chunks::<64>().0 {
        let (halves, _) = group.as_chunks::<2>();
        let values = halves.iter().map(|&h| half::f16::from_le_bytes(h).to_f32());
        let amax = values.clone().fold(0f32, |m, v| m.max(v.abs()));
        let scale = half::f16::from_f32(amax / 127.0);
        let inv = if scale.to_f32() > 0.0 { 1.0 / scale.to_f32() } else { 0.0 };
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend(values.map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8 as u8));
    }
    out
}

/// The inverse of `q8_encode`, to within the quantisation step.
fn q8_decode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 34 * 64);
    for group in data.as_chunks::<34>().0 {
        let scale = half::f16::from_le_bytes([group[0], group[1]]).to_f32();
        for &q in &group[2..] {
            out.extend_from_slice(&half::f16::from_f32(q as i8 as f32 * scale).to_le_bytes());
        }
    }
    out
}

/// A 64-bit checksum, four lanes wide so it runs near memory speed. Each lane is
/// a bijection of its state and the next word, so any single changed word changes
/// the result.
pub(crate) fn checksum(data: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut lanes = [0x243f_6a88_85a3_08d3_u64, 0x1319_8a2e_0370_7344, 0xa409_3822_299f_31d0, 0x082e_fa98_ec4e_6c89];
    let (chunks, tail) = data.as_chunks::<32>();
    for chunk in chunks {
        let (words, _) = chunk.as_chunks::<8>();
        for (lane, word) in lanes.iter_mut().zip(words) {
            *lane = (*lane ^ u64::from_le_bytes(*word)).wrapping_mul(K).rotate_left(29);
        }
    }
    let mut h = data.len() as u64;
    for lane in lanes.into_iter().chain(tail.iter().map(|&b| b as u64)) {
        h = (h ^ lane).wrapping_mul(K).rotate_left(31);
    }
    h
}

/// Identity of a model's weight files: each file's size, its first 16 MiB (header,
/// metadata and vocabulary) and 64 pages sampled across the rest. Cheap at startup,
/// and different for any two models that differ in their weights.
pub(crate) fn fingerprint(paths: &[PathBuf]) -> io::Result<u64> {
    const HEAD: u64 = 16 << 20;
    const PAGE: u64 = 4096;
    let mut h = Vec::new();
    for path in paths {
        let mut file = File::open(path)?;
        let len = file.metadata()?.len();
        h.extend_from_slice(&len.to_le_bytes());
        let mut head = Vec::new();
        Read::take(&mut file, HEAD).read_to_end(&mut head)?;
        h.extend_from_slice(&checksum(&head).to_le_bytes());
        if len > HEAD + PAGE {
            let mut page = vec![0; PAGE as usize];
            for i in 0..64 {
                file.seek(SeekFrom::Start(HEAD + (len - HEAD - PAGE) / 63 * i))?;
                file.read_exact(&mut page)?;
                h.extend_from_slice(&checksum(&page).to_le_bytes());
            }
        }
    }
    Ok(checksum(&h))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::decoder::prefix_cache::{chain, BLOCK, ROOT};

    /// A fresh, empty directory under the system temporary directory.
    pub(crate) fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ojas-prefix-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn record(parent: u64, seed: u32, kv_len: u64, snapshot_len: u64) -> Record {
        let tokens: Box<[u32]> = (0..BLOCK as u32).map(|i| i * 7 + seed).collect();
        Record {
            key: chain(parent, &tokens), parent, tokens, uses: 3, snapshot_uses: 1, last_used: 9, kv_len, snapshot_len,
            pinned: seed == 1,
        }
    }

    fn options(readonly: bool, reserve: u64) -> DiskOptions {
        DiskOptions { readonly, reserve, block_tokens: BLOCK, kv: KvFormat::F16 }
    }

    fn open(root: &Path, readonly: bool) -> Opened { Disk::open(root, 42, "test model", options(readonly, 0)).unwrap() }

    #[test]
    fn checksum_sees_any_changed_byte() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31) as u8).collect();
        let base = checksum(&data);
        for i in [0, 7, 31, 32, 500, 999] {
            let mut d = data.clone();
            d[i] ^= 1;
            assert_ne!(checksum(&d), base, "byte {i}");
        }
        assert_ne!(checksum(&data[..999]), base);
    }

    #[test]
    fn index_round_trips_and_rejects_damage() {
        let a = record(ROOT, 1, 10, 20);
        let b = record(a.key, 2, 10, 0);
        let bytes = encode_index(&[a.clone(), b.clone()], 77);
        assert_eq!(decode_index(&bytes, BLOCK), Some((vec![a, b], 77)));
        let mut bad = bytes.clone();
        bad[30] ^= 1;
        assert_eq!(decode_index(&bad, BLOCK), None);
        assert_eq!(decode_index(&bytes[..bytes.len() - 1], BLOCK), None);
        assert_eq!(decode_index(&bytes, BLOCK + 1), None, "blocks of another length");
    }

    #[test]
    fn payloads_and_index_survive_reopening() {
        let root = scratch("reopen");
        let a = record(ROOT, 1, 64, 128);
        let b = record(a.key, 2, 64, 0);
        let (kv_a, snap_a, kv_b) = (Arc::new(vec![1u8; 64]), Arc::new(vec![2u8; 128]), Arc::new(vec![3u8; 64]));
        {
            let o = open(&root, false);
            assert!(o.records.is_empty() && o.disk.writable());
            assert!(o.disk.write(a.key, Part::Kv, ROOT, &a.tokens, kv_a.clone()).is_some());
            assert!(o.disk.write(a.key, Part::Snapshot, ROOT, &a.tokens, snap_a.clone()).is_some());
            assert!(o.disk.write(b.key, Part::Kv, a.key, &b.tokens, kv_b.clone()).is_some());
            o.disk.save_index(&[a.clone(), b.clone()], 5);
            o.disk.flush();
            assert_eq!(o.disk.written().len(), 3);
        }
        let o = open(&root, false);
        assert_eq!((o.records.clone(), o.clock), (vec![a.clone(), b.clone()], 5));
        assert_eq!(o.disk.read(a.key, Part::Snapshot, ROOT, &a.tokens, 128).unwrap(), *snap_a);
        assert_eq!(o.disk.read(b.key, Part::Kv, a.key, &b.tokens, 64).unwrap(), *kv_b);
        assert!(o.disk.read(b.key, Part::Kv, ROOT, &b.tokens, 64).is_err(), "wrong parent");
        assert!(o.disk.read(b.key, Part::Kv, a.key, &a.tokens, 64).is_err(), "wrong tokens");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_corrupt_payload_fails_verification() {
        let root = scratch("corrupt");
        let a = record(ROOT, 1, 64, 0);
        let o = open(&root, false);
        assert!(o.disk.write(a.key, Part::Kv, ROOT, &a.tokens, Arc::new(vec![5u8; 64])).is_some());
        o.disk.flush();
        let path = o.disk.dir().join(format!("{:016x}.kv", a.key));
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0xff;
        fs::write(&path, bytes).unwrap();
        assert_eq!(o.disk.read(a.key, Part::Kv, ROOT, &a.tokens, 64).unwrap_err().kind(), io::ErrorKind::InvalidData);
        drop(o);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn startup_drops_missing_files_orphans_and_strays() {
        let root = scratch("startup");
        let a = record(ROOT, 1, 64, 128);
        let b = record(a.key, 2, 64, 0);
        let c = record(b.key, 3, 64, 0);
        {
            let o = open(&root, false);
            for (r, parent) in [(&a, ROOT), (&b, a.key), (&c, b.key)] {
                assert!(o.disk.write(r.key, Part::Kv, parent, &r.tokens, Arc::new(vec![1; 64])).is_some());
            }
            assert!(o.disk.write(a.key, Part::Snapshot, ROOT, &a.tokens, Arc::new(vec![2; 128])).is_some());
            o.disk.save_index(&[a.clone(), b.clone(), c.clone()], 1);
            o.disk.flush();
            fs::remove_file(o.disk.dir().join(format!("{:016x}.kv", b.key))).unwrap();
            fs::remove_file(o.disk.dir().join(format!("{:016x}.snap", a.key))).unwrap();
            fs::write(o.disk.dir().join("00000000000000ff.kv"), b"stray").unwrap();
        }
        let o = open(&root, false);
        let mut expect = a.clone();
        (expect.snapshot_len, expect.snapshot_uses) = (0, 0);
        assert_eq!(o.records, vec![expect], "b is missing, so c is an orphan");
        let mut names: Vec<String> = fs::read_dir(o.disk.dir()).unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(names, vec![format!("{:016x}.kv", a.key), "index".into(), "lock".into(), "model".into()]);
        drop(o);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_second_writer_is_read_only_and_read_only_writes_nothing() {
        let root = scratch("lock");
        let first = open(&root, false);
        let second = open(&root, false);
        assert!(first.disk.writable() && !second.disk.writable());
        let a = record(ROOT, 1, 4, 0);
        assert!(second.disk.write(a.key, Part::Kv, ROOT, &a.tokens, Arc::new(vec![0; 4])).is_none());
        drop((first, second));
        let ro = Disk::open(&root.join("absent"), 42, "", options(true, 0)).unwrap();
        assert!(!ro.disk.writable() && ro.records.is_empty() && !root.join("absent").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn the_free_space_reserve_refuses_writes() {
        let root = scratch("reserve");
        let o = Disk::open(&root, 42, "", options(false, u64::MAX / 2)).unwrap();
        let a = record(ROOT, 1, 4, 0);
        assert!(o.disk.write(a.key, Part::Kv, ROOT, &a.tokens, Arc::new(vec![0; 4])).is_some());
        o.disk.flush();
        assert_eq!(o.disk.written(), vec![Written { key: a.key, part: Part::Kv, seq: 1, ok: false }]);
        drop(o);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn q8_round_trips_within_half_a_step_per_group() {
        let values: Vec<f32> = (0..256).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013 * (1 + i / 64) as f32).collect();
        let data: Vec<u8> = values.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
        let encoded = q8_encode(&data);
        assert_eq!(encoded.len(), data.len() / 64 * 34);
        let decoded = q8_decode(&encoded);
        let back: Vec<f32> = decoded.as_chunks::<2>().0.iter().map(|&h| half::f16::from_le_bytes(h).to_f32()).collect();
        for (group, out) in values.chunks(32).zip(back.chunks(32)) {
            let step = group.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
            for (v, o) in group.iter().zip(out) {
                assert!((v - o).abs() <= step * 0.51 + v.abs() * 2e-3, "{v} came back as {o}");
            }
        }
    }

    #[test]
    fn a_q8_directory_stores_kv_compactly_and_snapshots_exactly() {
        let root = scratch("q8");
        let a = record(ROOT, 1, 128, 64);
        let whole = |i: u16| if i.is_multiple_of(32) { 127.0 } else { (i % 32) as f32 - 16.0 };
        let kv: Vec<u8> = (0..64u16).flat_map(|i| half::f16::from_f32(whole(i)).to_le_bytes()).collect();
        let snapshot: Vec<u8> = (0..64u8).collect();
        let o = Disk::open(&root, 42, "", DiskOptions { kv: KvFormat::Q8, ..options(false, 0) }).unwrap();
        assert_eq!(o.disk.stored_len(Part::Kv, 128), 68);
        assert!(o.disk.write(a.key, Part::Kv, ROOT, &a.tokens, Arc::new(kv.clone())).is_some());
        assert!(o.disk.write(a.key, Part::Snapshot, ROOT, &a.tokens, Arc::new(snapshot.clone())).is_some());
        o.disk.flush();
        let file = o.disk.dir().join(format!("{:016x}.kv", a.key));
        assert_eq!(fs::metadata(file).unwrap().len(), (68 + header_len(BLOCK)) as u64);
        assert_eq!(o.disk.read(a.key, Part::Kv, ROOT, &a.tokens, 128).unwrap(), kv, "whole numbers with a group maximum of 127 are exact in Q8");
        assert_eq!(o.disk.read(a.key, Part::Snapshot, ROOT, &a.tokens, 64).unwrap(), snapshot);
        drop(o);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn fingerprints_differ_when_sampled_weights_differ() {
        let root = scratch("fingerprint");
        let path = root.join("model.gguf");
        let mut bytes = vec![7u8; (16 << 20) + (1 << 20)];
        fs::write(&path, &bytes).unwrap();
        let before = fingerprint(std::slice::from_ref(&path)).unwrap();
        bytes[16 << 20] ^= 1;
        fs::write(&path, &bytes).unwrap();
        assert_ne!(fingerprint(std::slice::from_ref(&path)).unwrap(), before);
        let _ = fs::remove_dir_all(root);
    }
}
