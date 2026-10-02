//! Prompt-prefix cache: model state for prompt prefixes, kept in fixed blocks so a
//! request reuses every leading block it shares with any earlier one.
//!
//! A block is `BLOCK` tokens. Its key chains its parent's key with its own tokens,
//! so equal keys name equal prefixes, and the blocks form a tree: one cached system
//! prompt is the root of every conversation that starts with it. A block holds the
//! attention KV rows of its own positions, and optionally a snapshot of the
//! recurrent state after its last token. Attention rows can be cut back to any
//! block boundary; recurrent state can only resume where a snapshot was taken, so
//! a prefix is resumable at the deepest block that has one.
//!
//! Payloads live in two tiers, each with its own byte budget: RAM, and optionally
//! a cache directory (`prefix_disk`) that outlives the process. A payload may have
//! a copy in either or both; one read from disk is kept in RAM while it fits.
//!
//! Eviction keeps what prompts use most, in both tiers. Every block and snapshot
//! counts the prompts that reused it, the counts halve every `HALF_LIFE` prompts so
//! that old favourites fade, and the entry with the lowest count goes first, the
//! least recently used among equals. A shared system prompt therefore outlives any
//! number of one-off prompts that arrived after it. A pinned prefix is never
//! evicted: it is saved to the directory, and RAM drops it only when it is there.
//!
//! This module is the bookkeeping only. Copying state in and out of the model is
//! the decoder's job, and file I/O is `prefix_disk`'s.

use super::prefix_disk::{Bytes, Disk, Opened, Part, Record, Wanted};
use ojas_core::config::PrefixCacheSave;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Tokens per block. A multiple of the prefill chunk, so a run resumed at a block
/// boundary prefills the rest in the same chunks a fresh run would.
pub(crate) const BLOCK: usize = 256;

/// Prompts after which every use count halves. A prefix reused every few prompts
/// stays ahead of new entries; one unused for a few hundred falls level with them.
const HALF_LIFE: u64 = 128;

/// Prompts between index writes that only update use counts. A change to what is
/// on disk writes the index after the prompt that made it.
const INDEX_EVERY: u64 = 32;

/// Key of the empty prefix, the parent of every first block.
pub(crate) const ROOT: u64 = 0xcbf2_9ce4_8422_2325;

/// Chain `parent` with one block of tokens (FNV-1a). A collision cannot serve the
/// wrong state: lookups also compare the stored tokens.
pub(crate) fn chain(parent: u64, tokens: &[u32]) -> u64 {
    let mut h = parent;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// The keys of every full block of `tokens`.
pub(crate) fn block_keys(tokens: &[u32]) -> Vec<u64> {
    let mut keys = Vec::with_capacity(tokens.len() / BLOCK);
    let mut parent = ROOT;
    for block in tokens.as_chunks::<BLOCK>().0 {
        parent = chain(parent, block);
        keys.push(parent);
    }
    keys
}

/// What a prefill in progress adds to the cache: its prompt, the prompt's block
/// keys, and where to keep recurrent snapshots.
pub(crate) struct Plan {
    pub(crate) tokens: Vec<u32>,
    pub(crate) keys: Vec<u64>,
    /// Leading blocks the prompt shares with the cache. When no snapshot exists
    /// there, the run re-prefills through this boundary and leaves one, so the next
    /// prompt that branches at the same place resumes there.
    pub(crate) branch: usize,
    /// Blocks restored from the cache; prefill starts after them.
    pub(crate) resume: usize,
    /// Whether a prefill has taken up this plan.
    pub(crate) started: bool,
    /// Block counts where the prompt has a boundary a later prompt may diverge at.
    marks: Vec<usize>,
    /// Snapshots taken at marks and intervals so far.
    extra: usize,
    /// Tokens through which the sequence is exact. A document reused from the
    /// document cache is approximate, and no block after it is cached.
    pub(crate) exact: usize,
}

/// Snapshots a prompt may keep besides the ones at its branch point and its end.
const EXTRA_SNAPSHOTS: usize = 6;

impl Plan {
    /// A plan for `tokens` (whose block keys are `keys`) after a lookup that found
    /// `hit`. `marks` are token positions of boundaries in the prompt; each keeps a
    /// snapshot at the block boundary at or before it.
    pub(crate) fn new(tokens: Vec<u32>, keys: Vec<u64>, hit: Hit, marks: &[usize]) -> Self {
        let mut marks: Vec<usize> = marks.iter().map(|&p| p / BLOCK).filter(|&b| b > 0).collect();
        marks.dedup();
        Plan { tokens, keys, branch: hit.blocks, resume: hit.resume, started: false, marks, extra: 0, exact: usize::MAX }
    }

    /// Whether to keep a snapshot after the first `blocks` blocks: always at the
    /// branch point and at the end of the prompt; at marks and every `every` tokens
    /// (rounded down to whole blocks) inside a long prompt, up to `EXTRA_SNAPSHOTS`
    /// of them.
    pub(crate) fn wants_snapshot(&self, blocks: usize, every: usize) -> bool {
        if blocks == self.branch || blocks == self.keys.len() { return true; }
        self.extra < EXTRA_SNAPSHOTS && (self.marks.contains(&blocks) || blocks.is_multiple_of((every / BLOCK).max(1)))
    }

    /// Count a snapshot taken after the first `blocks` blocks.
    pub(crate) fn took_snapshot(&mut self, blocks: usize) {
        if blocks != self.branch && blocks != self.keys.len() { self.extra += 1; }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tier {
    Ram,
    Disk,
}

/// Whether a payload is in the cache directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Saved {
    No,
    /// Handed to the writer as write number `n`; the writer holds the bytes until
    /// the file is complete.
    Queued(u64),
    Yes,
}

/// A block's KV rows or its snapshot, and where its copies are.
struct Payload {
    len: usize,
    ram: Option<Bytes>,
    disk: Saved,
}

impl Payload {
    fn in_ram(bytes: Vec<u8>) -> Self { Payload { len: bytes.len(), ram: Some(Arc::new(bytes)), disk: Saved::No } }

    fn on_disk(len: usize) -> Self { Payload { len, ram: None, disk: Saved::Yes } }

    fn held(&self, tier: Tier) -> bool {
        match tier {
            Tier::Ram => self.ram.is_some(),
            Tier::Disk => self.disk != Saved::No,
        }
    }
}

struct Block {
    parent: u64,
    tokens: Box<[u32]>,
    depth: u32,
    kv: Payload,
    snapshot: Option<Payload>,
    /// Child blocks, and those of them whose KV is in the directory.
    children: u32,
    disk_children: u32,
    last_used: u64,
    /// Prompts that contained this block, and prompts that resumed at its snapshot.
    uses: u32,
    snapshot_uses: u32,
    pinned: bool,
}

impl Block {
    fn part(&self, part: Part) -> Option<&Payload> {
        match part {
            Part::Kv => Some(&self.kv),
            Part::Snapshot => self.snapshot.as_ref(),
        }
    }

    fn part_mut(&mut self, part: Part) -> Option<&mut Payload> {
        match part {
            Part::Kv => Some(&mut self.kv),
            Part::Snapshot => self.snapshot.as_mut(),
        }
    }

    fn record(&self, key: u64) -> Record {
        let snapshot = self.snapshot.as_ref().filter(|s| s.disk != Saved::No);
        Record {
            key, parent: self.parent, tokens: self.tokens.clone(), uses: self.uses,
            snapshot_uses: if snapshot.is_some() { self.snapshot_uses } else { 0 },
            last_used: self.last_used, kv_len: self.kv.len as u64,
            snapshot_len: snapshot.map_or(0, |s| s.len as u64),
            pinned: self.pinned,
        }
    }
}

/// The longest cached prefix of a prompt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Hit {
    /// Leading blocks present (KV rows cached).
    pub(crate) blocks: usize,
    /// Leading blocks the run can resume after: the deepest block among them with
    /// a snapshot, or all of them when the model keeps no recurrent state.
    pub(crate) resume: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Stats {
    pub(crate) blocks: usize,
    pub(crate) snapshots: usize,
    pub(crate) bytes: usize,
    pub(crate) budget: usize,
    pub(crate) disk_blocks: usize,
    pub(crate) disk_snapshots: usize,
    pub(crate) disk_bytes: usize,
    pub(crate) disk_budget: usize,
    pub(crate) disk_reads: u64,
    pub(crate) pinned: usize,
    pub(crate) evictions: u64,
    pub(crate) lookups: u64,
    pub(crate) hits: u64,
    pub(crate) reused_tokens: u64,
}

/// A tier's byte budget and the bytes it holds.
#[derive(Clone, Copy, Debug, Default)]
struct Room {
    budget: usize,
    used: usize,
}

struct DiskTier {
    io: Disk,
    room: Room,
    save: PrefixCacheSave,
    reads: u64,
    /// Whether the saved index is behind what is on disk, or only behind its use
    /// counts, and the prompt count when it was last written.
    dirty: bool,
    counts_dirty: bool,
    indexed_at: u64,
}

pub(crate) struct PrefixCache {
    blocks: HashMap<u64, Block>,
    ram: Room,
    disk: Option<DiskTier>,
    clock: u64,
    stats: Stats,
}

impl PrefixCache {
    pub(crate) fn new(budget: usize) -> Self {
        PrefixCache { blocks: HashMap::new(), ram: Room { budget, used: 0 }, disk: None, clock: 0, stats: Stats::default() }
    }

    pub(crate) fn enabled(&self) -> bool { self.ram.budget > 0 }

    /// Back the cache with an opened directory holding at most `budget` bytes. Its
    /// blocks become cached prefixes at once; records whose payload sizes are not
    /// `kv_len` and `snapshot_len`, or whose key does not match their tokens, are
    /// left out of the next index, and their files are removed the next time the
    /// directory is opened.
    pub(crate) fn attach_disk(&mut self, opened: Opened, budget: usize, save: PrefixCacheSave, kv_len: usize,
                              snapshot_len: usize) {
        let Opened { disk, records, clock } = opened;
        let mut used = 0;
        for r in records {
            let parent = match r.parent {
                ROOT => Some(0),
                p => self.blocks.get(&p).map(|b| b.depth),
            };
            let Some(parent_depth) = parent else { continue };
            if r.kv_len as usize != kv_len || chain(r.parent, &r.tokens) != r.key || self.blocks.contains_key(&r.key) {
                continue;
            }
            let snapshot = (r.snapshot_len > 0 && r.snapshot_len as usize == snapshot_len).then(|| Payload::on_disk(snapshot_len));
            used += disk.stored_len(Part::Kv, kv_len) + snapshot.as_ref().map_or(0, |s| disk.stored_len(Part::Snapshot, s.len));
            if let Some(p) = self.blocks.get_mut(&r.parent) {
                p.children += 1;
                p.disk_children += 1;
            }
            self.blocks.insert(r.key, Block {
                parent: r.parent, tokens: r.tokens, depth: parent_depth + 1, kv: Payload::on_disk(kv_len),
                snapshot_uses: if snapshot.is_some() { r.snapshot_uses } else { 0 }, snapshot,
                children: 0, disk_children: 0, last_used: r.last_used, uses: r.uses, pinned: r.pinned,
            });
        }
        self.clock = self.clock.max(clock);
        let writable = disk.writable();
        self.disk = Some(DiskTier {
            io: disk, room: Room { budget, used }, save, reads: 0, dirty: writable, counts_dirty: false, indexed_at: 0,
        });
        if writable {
            self.make_room(Tier::Disk, 0, None);
        }
    }

    pub(crate) fn stats(&self) -> Stats {
        let mut s = Stats { blocks: self.blocks.len(), bytes: self.ram.used, budget: self.ram.budget, ..self.stats };
        for b in self.blocks.values() {
            s.snapshots += usize::from(b.snapshot.is_some());
            s.pinned += usize::from(b.pinned);
            s.disk_blocks += usize::from(b.kv.held(Tier::Disk));
            s.disk_snapshots += usize::from(b.snapshot.as_ref().is_some_and(|p| p.held(Tier::Disk)));
        }
        if let Some(d) = &self.disk {
            (s.disk_bytes, s.disk_budget, s.disk_reads) = (d.room.used, d.room.budget, d.reads);
        }
        s
    }

    fn matches(&self, key: u64, parent: u64, tokens: &[u32]) -> bool {
        self.blocks.get(&key).is_some_and(|b| b.parent == parent && *b.tokens == *tokens)
    }

    /// The longest cached prefix of `tokens` (full blocks only). `stateful` models
    /// can resume only after a block with a snapshot.
    pub(crate) fn lookup(&self, tokens: &[u32], keys: &[u64], stateful: bool) -> Hit {
        let mut hit = Hit::default();
        let mut parent = ROOT;
        for (i, (&key, block)) in keys.iter().zip(tokens.as_chunks::<BLOCK>().0).enumerate() {
            if !self.matches(key, parent, block) { break; }
            hit.blocks = i + 1;
            if !stateful || self.blocks[&key].snapshot.is_some() { hit.resume = i + 1; }
            parent = key;
        }
        hit
    }

    /// The payloads that restore the blocks `keys` (a prefix's leading blocks): the
    /// KV of each, and the snapshot after the last if it has one. `None` when a copy
    /// on disk is unreadable or damaged; that entry is dropped, so a new lookup finds
    /// what is still usable.
    pub(crate) fn load(&mut self, keys: &[u64]) -> Option<(Vec<Bytes>, Option<Bytes>)> {
        let Some(&last) = keys.last() else { return Some((Vec::new(), None)) };
        let mut wanted: Vec<(u64, Part)> = keys.iter().map(|&key| (key, Part::Kv)).collect();
        if self.blocks.get(&last)?.snapshot.is_some() { wanted.push((last, Part::Snapshot)); }
        // RAM copies are taken first: keeping a payload read from disk can evict a
        // later one's RAM copy, which would then be read again.
        let in_ram: Vec<Option<Bytes>> = wanted.iter()
            .map(|&(key, part)| self.blocks.get(&key).and_then(|b| b.part(part)).and_then(|p| p.ram.clone()))
            .collect();
        let mut read = self.read_from_disk(&wanted);
        let mut payloads = Vec::with_capacity(wanted.len());
        for ((key, part), ram) in wanted.into_iter().zip(in_ram) {
            payloads.push(match ram {
                Some(bytes) => bytes,
                None => self.fetch(key, part, last, read.remove(&(key, part)))?,
            });
        }
        let snapshot = (payloads.len() > keys.len()).then(|| payloads.pop().unwrap());
        Some((payloads, snapshot))
    }

    /// Read, all at once, the payloads among `wanted` that only the directory holds.
    fn read_from_disk(&self, wanted: &[(u64, Part)]) -> HashMap<(u64, Part), std::io::Result<Vec<u8>>> {
        let Some(d) = &self.disk else { return HashMap::new() };
        let mut queued = false;
        let mut list = Vec::new();
        for &(key, part) in wanted {
            let Some((b, p)) = self.blocks.get(&key).and_then(|b| Some((b, b.part(part)?))) else { continue };
            if p.ram.is_some() { continue; }
            queued |= matches!(p.disk, Saved::Queued(_));
            list.push(Wanted { key, part, parent: b.parent, tokens: &b.tokens, len: p.len });
        }
        if queued { d.io.flush(); }
        let results = d.io.read_many(&list);
        list.iter().map(|w| (w.key, w.part)).zip(results).collect()
    }

    /// Count a prompt that started a sequence. `used` are the cached blocks it
    /// matched, and it resumed after the first `resume` of them, at that block's
    /// snapshot; both are empty when reuse was not allowed. Blocks evicted since
    /// the lookup are skipped, with those after them. An entry is saved to the
    /// directory on its second use under the `Reused` policy, and under `Always`
    /// if an earlier save was skipped.
    pub(crate) fn record(&mut self, used: &[u64], resume: usize) {
        let used = &used[..used.iter().take_while(|key| self.blocks.contains_key(key)).count()];
        self.stats.lookups += 1;
        if self.stats.lookups.is_multiple_of(HALF_LIFE) {
            for b in self.blocks.values_mut() {
                b.uses /= 2;
                b.snapshot_uses /= 2;
            }
        }
        self.clock += 1;
        for (i, key) in used.iter().enumerate() {
            let b = self.blocks.get_mut(key).unwrap();
            b.uses += 1;
            b.snapshot_uses += u32::from(i + 1 == resume);
            b.last_used = self.clock;
        }
        if resume > 0 {
            self.stats.hits += 1;
            self.stats.reused_tokens += (resume * BLOCK) as u64;
        }
        let Some(d) = self.disk.as_mut() else { return };
        d.counts_dirty |= !used.is_empty();
        let always = d.save == PrefixCacheSave::Always;
        let Some(&chain_end) = used.last() else { return };
        for (i, &key) in used.iter().enumerate() {
            let Some(b) = self.blocks.get(&key) else { break };
            let (kv, snapshot) = (always || b.uses >= 2, i + 1 == resume && (always || b.snapshot_uses >= 2));
            if kv { self.save(key, Part::Kv, chain_end); }
            if snapshot { self.save(key, Part::Snapshot, chain_end); }
        }
    }

    pub(crate) fn contains(&self, key: u64) -> bool { self.blocks.contains_key(&key) }

    pub(crate) fn has_snapshot(&self, key: u64) -> bool {
        self.blocks.get(&key).is_some_and(|b| b.snapshot.is_some())
    }

    /// Admit a block whose parent is cached (or the root). Returns false, storing
    /// nothing, when the block cannot fit even after eviction.
    pub(crate) fn insert(&mut self, parent: u64, tokens: &[u32], kv: Vec<u8>) -> bool {
        let key = chain(parent, tokens);
        if self.blocks.contains_key(&key) || (parent != ROOT && !self.blocks.contains_key(&parent)) {
            return self.blocks.contains_key(&key);
        }
        if !self.make_room(Tier::Ram, kv.len(), Some(parent)) { return false; }
        self.ram.used += kv.len();
        self.clock += 1;
        let depth = match self.blocks.get_mut(&parent) {
            Some(p) => {
                p.children += 1;
                p.depth + 1
            }
            None => 1,
        };
        self.blocks.insert(key, Block {
            parent, tokens: tokens.into(), depth, kv: Payload::in_ram(kv), snapshot: None,
            children: 0, disk_children: 0, last_used: self.clock, uses: 1, snapshot_uses: 0, pinned: false,
        });
        if self.saves(PrefixCacheSave::Always) { self.save(key, Part::Kv, key); }
        true
    }

    /// Attach a recurrent-state snapshot to a cached block.
    pub(crate) fn attach_snapshot(&mut self, key: u64, snapshot: Vec<u8>) -> bool {
        if !self.blocks.contains_key(&key) || self.has_snapshot(key) { return self.has_snapshot(key); }
        if !self.make_room(Tier::Ram, snapshot.len(), Some(key)) { return false; }
        self.ram.used += snapshot.len();
        let b = self.blocks.get_mut(&key).unwrap();
        b.snapshot = Some(Payload::in_ram(snapshot));
        b.snapshot_uses = 1;
        if self.saves(PrefixCacheSave::Always) { self.save(key, Part::Snapshot, key); }
        true
    }

    /// Pin the cached leading blocks of `keys` and the snapshot after the last of
    /// them, and save them to the directory whatever the save policy. Returns the
    /// blocks pinned.
    pub(crate) fn pin(&mut self, keys: &[u64]) -> usize {
        let keys = &keys[..keys.iter().take_while(|key| self.blocks.contains_key(key)).count()];
        let Some(&last) = keys.last() else { return 0 };
        for &key in keys {
            self.blocks.get_mut(&key).unwrap().pinned = true;
            self.save(key, Part::Kv, last);
        }
        self.save(last, Part::Snapshot, last);
        if let Some(d) = self.disk.as_mut() { d.dirty = true; }
        keys.len()
    }

    /// Release every pin; the blocks stay cached and compete on use again.
    pub(crate) fn unpin_all(&mut self) {
        for b in self.blocks.values_mut() { b.pinned = false; }
        if let Some(d) = self.disk.as_mut() { d.dirty = true; }
    }

    /// Take in finished writes, and queue a new index if what is on disk changed,
    /// or if only use counts did and `INDEX_EVERY` prompts have passed.
    pub(crate) fn sync(&mut self) { self.sync_index(false); }

    fn sync_index(&mut self, always: bool) {
        let Some(d) = self.disk.as_mut() else { return };
        for w in d.io.written() {
            let this_write = self.blocks.get(&w.key).and_then(|b| b.part(w.part))
                .is_some_and(|p| p.disk == Saved::Queued(w.seq));
            if !this_write { continue; }
            match (w.ok, w.part) {
                (true, _) => self.set_saved(w.key, w.part, Saved::Yes),
                (false, Part::Kv) => self.unsave_tree(w.key),
                (false, Part::Snapshot) => self.drop_copy(w.key, Part::Snapshot, Tier::Disk),
            }
        }
        let lookups = self.stats.lookups;
        let Some(d) = self.disk.as_mut().filter(|d| d.io.writable()) else { return };
        let counts_due = d.counts_dirty && (always || lookups - d.indexed_at >= INDEX_EVERY);
        if !d.dirty && !counts_due { return; }
        let mut saved: Vec<(&u64, &Block)> = self.blocks.iter().filter(|(_, b)| b.kv.held(Tier::Disk)).collect();
        saved.sort_by_key(|(_, b)| b.depth);
        let records: Vec<Record> = saved.into_iter().map(|(&key, b)| b.record(key)).collect();
        d.io.save_index(&records, self.clock);
        (d.dirty, d.counts_dirty, d.indexed_at) = (false, false, lookups);
    }

    /// Wait for every queued write and save the index, so the directory holds
    /// everything cached so far. Two passes: the first queues an index and waits
    /// for the writes before it, the second takes in their results and writes the
    /// index that reflects them.
    pub(crate) fn flush(&mut self) {
        for _ in 0..2 {
            self.sync_index(true);
            if let Some(d) = &self.disk { d.io.flush(); }
        }
    }

    fn saves(&self, policy: PrefixCacheSave) -> bool { self.disk.as_ref().is_some_and(|d| d.save == policy) }

    /// Queue a RAM payload for the directory, when there is room for it there and
    /// the blocks it depends on are saved too. `keep` (the payload's block or one
    /// below it) and its ancestors are not evicted to make room.
    fn save(&mut self, key: u64, part: Part, keep: u64) {
        if !self.disk.as_ref().is_some_and(|d| d.io.writable()) { return; }
        let b = &self.blocks[&key];
        let depends_saved = match part {
            Part::Kv => b.parent == ROOT || self.blocks[&b.parent].kv.held(Tier::Disk),
            Part::Snapshot => b.kv.held(Tier::Disk),
        };
        let Some((len, Some(bytes))) = b.part(part).filter(|p| p.disk == Saved::No).map(|p| (p.len, p.ram.clone()))
        else { return };
        let stored = self.disk.as_ref().unwrap().io.stored_len(part, len);
        if !depends_saved || !self.make_room(Tier::Disk, stored, Some(keep)) { return; }
        let (b, d) = (&self.blocks[&key], self.disk.as_mut().unwrap());
        let Some(seq) = d.io.write(key, part, b.parent, &b.tokens, bytes) else { return };
        d.room.used += stored;
        self.set_saved(key, part, Saved::Queued(seq));
    }

    /// The bytes of a payload: RAM's copy, or else the directory's (`read`, when
    /// it was already read), kept in RAM if it fits there; `keep` and its ancestors
    /// are not evicted to make room. `None` when the directory's copy is unreadable
    /// or damaged; that copy is dropped.
    fn fetch(&mut self, key: u64, part: Part, keep: u64, read: Option<std::io::Result<Vec<u8>>>) -> Option<Bytes> {
        let p = self.blocks.get(&key)?.part(part)?;
        if let Some(bytes) = &p.ram { return Some(bytes.clone()); }
        let (len, d) = (p.len, self.disk.as_mut()?);
        if matches!(p.disk, Saved::Queued(_)) { d.io.flush(); }
        let b = &self.blocks[&key];
        match read.unwrap_or_else(|| d.io.read(key, part, b.parent, &b.tokens, len)) {
            Ok(data) => {
                d.reads += 1;
                let bytes = Arc::new(data);
                if self.make_room(Tier::Ram, len, Some(keep)) {
                    self.ram.used += len;
                    self.blocks.get_mut(&key).and_then(|b| b.part_mut(part)).unwrap().ram = Some(bytes.clone());
                }
                Some(bytes)
            }
            Err(e) => {
                tracing::warn!(target: "prefix", "dropping cached block {key:016x}: {e}");
                self.drop_copy(key, part, Tier::Disk);
                None
            }
        }
    }

    /// Record a payload's directory state, keeping the parent's count of children
    /// on disk and the index's dirty flag current.
    fn set_saved(&mut self, key: u64, part: Part, state: Saved) {
        let b = self.blocks.get_mut(&key).unwrap();
        let p = b.part_mut(part).unwrap();
        let (was, now) = (p.disk != Saved::No, state != Saved::No);
        p.disk = state;
        let parent = b.parent;
        if part == Part::Kv && was != now {
            if let Some(pb) = self.blocks.get_mut(&parent) {
                if now { pb.disk_children += 1 } else { pb.disk_children -= 1 }
            }
        }
        if let Some(d) = self.disk.as_mut() { d.dirty = true; }
    }

    /// Evict from `tier` until `need` more bytes fit, never touching `keep` or its
    /// ancestors (the block being extended or restored).
    fn make_room(&mut self, tier: Tier, need: usize, keep: Option<u64>) -> bool {
        let room = |c: &Self| match tier {
            Tier::Ram => Some(c.ram),
            Tier::Disk => c.disk.as_ref().map(|d| d.room),
        };
        let Some(budget) = room(self).map(|r| r.budget) else { return false };
        if need > budget { return false; }
        let protected: HashSet<u64> = std::iter::successors(keep.filter(|&k| k != ROOT), |k| {
            self.blocks.get(k).map(|b| b.parent).filter(|&p| p != ROOT)
        }).collect();
        while room(self).is_some_and(|r| r.used + need > budget) {
            let Some((key, part)) = self.victim(tier, &protected) else { return false };
            self.drop_copy(key, part, tier);
            self.stats.evictions += 1;
        }
        true
    }

    /// The next payload to evict from `tier`: least used, then least recently used.
    /// A snapshot goes before its block. A block's copy goes only once nothing in
    /// the same tier depends on it: in RAM, when it is saved or has no children; in
    /// the directory, when no child is saved there and it is either still in RAM or
    /// childless, since a block is useful only while the blocks before it are kept.
    /// Protected and pinned blocks keep their directory copies, and their RAM
    /// copies unless the directory holds them too.
    fn victim(&self, tier: Tier, protected: &HashSet<u64>) -> Option<(u64, Part)> {
        self.blocks.iter().filter_map(|(&key, b)| {
            let (rank, part) = if b.snapshot.as_ref().is_some_and(|s| s.held(tier)) {
                ((b.snapshot_uses, b.last_used), Part::Snapshot)
            } else {
                let droppable = b.kv.held(tier) && match tier {
                    Tier::Ram => b.kv.held(Tier::Disk) || b.children == 0,
                    Tier::Disk => b.disk_children == 0 && (b.kv.held(Tier::Ram) || b.children == 0),
                };
                if !droppable { return None; }
                ((b.uses, b.last_used), Part::Kv)
            };
            let saved = tier == Tier::Ram && b.part(part).is_some_and(|p| p.held(Tier::Disk));
            (saved || !(b.pinned || protected.contains(&key))).then_some((rank, key, part))
        }).min_by_key(|&(rank, ..)| rank).map(|(_, key, part)| (key, part))
    }

    /// Drop one tier's copy of a payload. A payload left with no copy is gone: a
    /// snapshot alone, or a block with every block below it.
    fn drop_copy(&mut self, key: u64, part: Part, tier: Tier) {
        let Some(p) = self.blocks.get(&key).and_then(|b| b.part(part)) else { return };
        let other = match tier {
            Tier::Ram => p.held(Tier::Disk),
            Tier::Disk => p.held(Tier::Ram),
        };
        if !other {
            match part {
                Part::Kv => self.remove_tree(key),
                Part::Snapshot => {
                    let s = self.blocks.get_mut(&key).unwrap().snapshot.take().unwrap();
                    self.blocks.get_mut(&key).unwrap().snapshot_uses = 0;
                    self.release(key, Part::Snapshot, &s);
                }
            }
            return;
        }
        let len = p.len;
        match tier {
            Tier::Ram => {
                self.blocks.get_mut(&key).and_then(|b| b.part_mut(part)).unwrap().ram = None;
                self.ram.used -= len;
            }
            Tier::Disk => {
                let d = self.disk.as_mut().unwrap();
                d.room.used -= d.io.stored_len(part, len);
                d.io.remove(key, part);
                self.set_saved(key, part, Saved::No);
            }
        }
    }

    /// `key` and every block below it, each before its children.
    fn subtree(&self, key: u64) -> Vec<u64> {
        let mut tree = vec![key];
        let mut i = 0;
        while i < tree.len() {
            let k = tree[i];
            if self.blocks.get(&k).is_some_and(|b| b.children > 0) {
                tree.extend(self.blocks.iter().filter(|(_, b)| b.parent == k).map(|(&c, _)| c));
            }
            i += 1;
        }
        tree
    }

    /// Drop the directory copies of a block, of its snapshot and of every block
    /// below it: on disk they are useless without it.
    fn unsave_tree(&mut self, key: u64) {
        for k in self.subtree(key).into_iter().rev() {
            for part in [Part::Snapshot, Part::Kv] {
                if self.blocks.get(&k).and_then(|b| b.part(part)).is_some_and(|p| p.held(Tier::Disk)) {
                    self.drop_copy(k, part, Tier::Disk);
                }
            }
        }
    }

    /// Remove a block and every block below it.
    fn remove_tree(&mut self, key: u64) {
        for k in self.subtree(key).into_iter().rev() {
            let b = self.blocks.remove(&k).unwrap();
            self.release(k, Part::Kv, &b.kv);
            if let Some(s) = &b.snapshot { self.release(k, Part::Snapshot, s); }
            if let Some(p) = self.blocks.get_mut(&b.parent) {
                p.children -= 1;
                p.disk_children -= u32::from(b.kv.held(Tier::Disk));
            }
        }
    }

    /// Return a removed payload's bytes to both tiers, deleting its file.
    fn release(&mut self, key: u64, part: Part, p: &Payload) {
        if p.held(Tier::Ram) { self.ram.used -= p.len; }
        if p.held(Tier::Disk) {
            let d = self.disk.as_mut().unwrap();
            d.room.used -= d.io.stored_len(part, p.len);
            d.io.remove(key, part);
            d.dirty = true;
        }
    }
}

impl Drop for PrefixCache {
    fn drop(&mut self) { self.flush(); }
}

#[cfg(test)]
mod tests {
    use super::super::prefix_disk::{tests::scratch, DiskOptions};
    use super::*;
    use std::path::Path;

    fn prompt(n: usize, seed: u32) -> Vec<u32> {
        (0..n as u32).map(|i| i.wrapping_mul(2_654_435_761).wrapping_add(seed) % 50_000).collect()
    }

    /// Cache every full block of `tokens`, with a snapshot after the blocks in `snaps`.
    fn fill(c: &mut PrefixCache, tokens: &[u32], kv: usize, snap: usize, snaps: &[usize]) {
        let mut parent = ROOT;
        for (i, block) in tokens.as_chunks::<BLOCK>().0.iter().enumerate() {
            assert!(c.insert(parent, block, vec![i as u8; kv]));
            parent = chain(parent, block);
            if snaps.contains(&(i + 1)) { assert!(c.attach_snapshot(parent, vec![0xAB; snap])); }
        }
    }

    #[test]
    fn keys_chain_so_equal_keys_mean_equal_prefixes() {
        let a = prompt(3 * BLOCK, 1);
        let mut b = a.clone();
        b[2 * BLOCK + 5] ^= 1;
        let (ka, kb) = (block_keys(&a), block_keys(&b));
        assert_eq!(ka.len(), 3);
        assert_eq!(ka[..2], kb[..2]);
        assert_ne!(ka[2], kb[2]);
        assert_eq!(block_keys(&a[..BLOCK + 10]).len(), 1, "partial blocks have no key");
    }

    #[test]
    fn resumes_at_the_deepest_snapshot_of_the_shared_prefix() {
        let mut c = PrefixCache::new(1 << 30);
        let a = prompt(4 * BLOCK, 7);
        fill(&mut c, &a, 16, 64, &[1, 3]);
        let keys = block_keys(&a);
        assert_eq!(c.lookup(&a, &keys, true), Hit { blocks: 4, resume: 3 });
        assert_eq!(c.lookup(&a, &keys, false), Hit { blocks: 4, resume: 4 });
        // A prompt that diverges in block 3 shares blocks 0..2 and resumes after 1.
        let mut b = a.clone();
        b[2 * BLOCK] ^= 1;
        assert_eq!(c.lookup(&b, &block_keys(&b), true), Hit { blocks: 2, resume: 1 });
        let other = prompt(4 * BLOCK, 99);
        assert_eq!(c.lookup(&other, &block_keys(&other), true), Hit::default());
    }

    #[test]
    fn branches_share_their_common_blocks() {
        let mut c = PrefixCache::new(1 << 30);
        let system = prompt(2 * BLOCK, 3);
        let page_a = [system.clone(), prompt(2 * BLOCK, 11)].concat();
        let page_b = [system.clone(), prompt(2 * BLOCK, 12)].concat();
        fill(&mut c, &page_a, 10, 20, &[2, 4]);
        fill(&mut c, &page_b, 10, 20, &[4]);
        assert_eq!(c.stats().blocks, 6, "the two system-prompt blocks are stored once");
        assert_eq!(c.lookup(&page_b, &block_keys(&page_b), true), Hit { blocks: 4, resume: 4 });
    }

    #[test]
    fn among_equally_used_entries_evicts_the_least_recent_from_the_leaves() {
        // Budget: four 10-byte blocks plus one 30-byte snapshot.
        let mut c = PrefixCache::new(70);
        let a = prompt(3 * BLOCK, 5);
        fill(&mut c, &a, 10, 30, &[3]);
        assert_eq!(c.stats().bytes, 60);
        let b = prompt(BLOCK, 6);
        fill(&mut c, &b, 10, 0, &[]);
        assert_eq!(c.stats().bytes, 70);
        // One more block must evict: the least recently used candidate is a's snapshot.
        let d = prompt(BLOCK, 8);
        fill(&mut c, &d, 10, 0, &[]);
        assert_eq!((c.stats().blocks, c.stats().snapshots), (5, 0));
        // Two more blocks fill the budget exactly; from here every block evicts one.
        fill(&mut c, &prompt(2 * BLOCK, 9), 10, 0, &[]);
        assert_eq!(c.stats().bytes, 70);
        // a's blocks are the oldest, but only its last one is a leaf: the chain is
        // eaten from the end, never from the middle.
        let ka = block_keys(&a);
        fill(&mut c, &prompt(BLOCK, 10), 10, 0, &[]);
        assert!(!c.contains(ka[2]) && c.contains(ka[1]) && c.contains(ka[0]));
        fill(&mut c, &prompt(BLOCK, 11), 10, 0, &[]);
        assert!(!c.contains(ka[1]) && c.contains(ka[0]));
        assert_eq!(c.stats().bytes, 70);
    }

    /// A system prompt of two blocks with a snapshot after it, reused `times` times.
    fn hot_system_prompt(c: &mut PrefixCache, times: usize) -> Vec<u32> {
        let system = prompt(2 * BLOCK, 21);
        fill(c, &system, 10, 30, &[2]);
        for _ in 0..times {
            let hit = c.lookup(&system, &block_keys(&system), true);
            c.record(&block_keys(&system)[..hit.blocks], hit.resume);
        }
        system
    }

    #[test]
    fn keeps_the_most_used_prefix_over_newer_one_off_prompts() {
        // Room for the system prompt and one one-off prompt of the same size.
        let mut c = PrefixCache::new(100);
        let system = hot_system_prompt(&mut c, 3);
        let one_offs: Vec<Vec<u32>> = (0..5).map(|i| prompt(2 * BLOCK, 100 + i)).collect();
        for p in &one_offs { fill(&mut c, p, 10, 30, &[2]); }
        assert_eq!(c.lookup(&system, &block_keys(&system), true), Hit { blocks: 2, resume: 2 });
        let last = &one_offs[4];
        assert_eq!(c.lookup(last, &block_keys(last), true), Hit { blocks: 2, resume: 2 });
        assert!(one_offs[..4].iter().all(|p| !c.contains(block_keys(p)[0])));
    }

    #[test]
    fn use_counts_fade_so_an_unused_favourite_gives_way() {
        let mut c = PrefixCache::new(100);
        let system = hot_system_prompt(&mut c, 3);
        for _ in 0..3 * HALF_LIFE { c.record(&[], 0); }
        let (a, b) = (prompt(2 * BLOCK, 31), prompt(2 * BLOCK, 32));
        fill(&mut c, &a, 10, 30, &[2]);
        fill(&mut c, &b, 10, 30, &[2]);
        assert_eq!(c.lookup(&system, &block_keys(&system), true), Hit::default());
        for p in [&a, &b] { assert_eq!(c.lookup(p, &block_keys(p), true), Hit { blocks: 2, resume: 2 }); }
    }

    #[test]
    fn a_pinned_prefix_is_kept_until_unpinned() {
        let mut c = PrefixCache::new(100);
        let system = prompt(2 * BLOCK, 41);
        fill(&mut c, &system, 10, 30, &[2]);
        assert_eq!(c.pin(&block_keys(&system)), 2);
        for i in 0..4 { fill(&mut c, &prompt(2 * BLOCK, 300 + i), 10, 30, &[2]); }
        assert_eq!(c.lookup(&system, &block_keys(&system), true), Hit { blocks: 2, resume: 2 });
        assert_eq!(c.stats().pinned, 2);
        c.unpin_all();
        fill(&mut c, &prompt(2 * BLOCK, 400), 10, 30, &[2]);
        assert_eq!(c.lookup(&system, &block_keys(&system), true), Hit::default(), "the oldest entry goes once unpinned");
    }

    #[test]
    fn refuses_what_cannot_fit_and_protects_the_block_being_extended() {
        let mut c = PrefixCache::new(25);
        assert!(!c.insert(ROOT, &prompt(BLOCK, 1), vec![0; 26]));
        let a = prompt(2 * BLOCK, 2);
        fill(&mut c, &a, 10, 0, &[]);
        // The parent chain is protected, so a snapshot larger than the room left fails.
        let keys = block_keys(&a);
        assert!(!c.attach_snapshot(keys[1], vec![0; 10]));
        assert!(c.contains(keys[0]) && c.contains(keys[1]));
    }

    #[test]
    fn snapshots_go_at_the_branch_the_end_and_every_interval() {
        let plan = Plan::new(vec![], vec![0; 13], Hit { blocks: 3, resume: 0 }, &[]);
        let want: Vec<usize> = (1..=13).filter(|&b| plan.wants_snapshot(b, 2048)).collect();
        assert_eq!(want, vec![3, 8, 13]);
    }

    #[test]
    fn marks_add_snapshots_up_to_a_cap() {
        let marks: Vec<usize> = (1..=20).map(|b| b * BLOCK + 7).collect();
        let mut plan = Plan::new(vec![], vec![0; 30], Hit { blocks: 25, resume: 0 }, &[100, 2 * BLOCK + 3]);
        let want: Vec<usize> = (1..=30).filter(|&b| plan.wants_snapshot(b, 1 << 20)).collect();
        assert_eq!(want, vec![2, 25, 30], "a mark inside the first block keeps nothing");
        plan = Plan::new(vec![], vec![0; 30], Hit::default(), &marks);
        let mut kept = Vec::new();
        for b in 1..=30 {
            if plan.wants_snapshot(b, 1 << 20) {
                plan.took_snapshot(b);
                kept.push(b);
            }
        }
        assert_eq!(kept, vec![1, 2, 3, 4, 5, 6, 30]);
    }

    #[test]
    fn a_disabled_cache_stores_nothing() {
        let mut c = PrefixCache::new(0);
        assert!(!c.enabled());
        assert!(!c.insert(ROOT, &prompt(BLOCK, 1), vec![0; 1]));
    }

    #[test]
    fn stats_count_prompts_hits_and_bytes() {
        let mut c = PrefixCache::new(1 << 30);
        let p = prompt(2 * BLOCK, 5);
        fill(&mut c, &p, 10, 30, &[2]);
        c.record(&[], 0);
        c.record(&block_keys(&p), 2);
        let s = c.stats();
        assert_eq!((s.blocks, s.snapshots, s.bytes), (2, 1, 50));
        assert_eq!((s.lookups, s.hits, s.reused_tokens), (2, 1, 2 * BLOCK as u64));
    }

    /// A cache backed by `root`, sized for the 10-byte blocks and 30-byte snapshots
    /// the tests store.
    fn open_dir(root: &Path, ram: usize, disk: usize, save: PrefixCacheSave, readonly: bool) -> PrefixCache {
        let mut c = PrefixCache::new(ram);
        let options = DiskOptions { readonly, reserve: 0, block_tokens: BLOCK, kv: Default::default() };
        c.attach_disk(Disk::open(root, 7, "test", options).unwrap(), disk, save, 10, 30);
        c
    }

    fn kv_tags(kv: &[Bytes]) -> Vec<u8> { kv.iter().map(|b| b[0]).collect() }

    #[test]
    fn saved_blocks_survive_a_restart_and_load_verbatim() {
        let root = scratch("restart");
        let p = prompt(3 * BLOCK, 1);
        let keys = block_keys(&p);
        {
            let mut c = open_dir(&root, 1 << 20, 1 << 20, PrefixCacheSave::Always, false);
            fill(&mut c, &p, 10, 30, &[3]);
            c.flush();
            assert_eq!((c.stats().disk_blocks, c.stats().disk_snapshots, c.stats().disk_bytes), (3, 1, 60));
        }
        let mut c = open_dir(&root, 1 << 20, 1 << 20, PrefixCacheSave::Always, false);
        assert_eq!((c.stats().blocks, c.stats().bytes), (3, 0), "startup reads no payload");
        assert_eq!(c.lookup(&p, &keys, true), Hit { blocks: 3, resume: 3 });
        let (kv, snapshot) = c.load(&keys).unwrap();
        assert_eq!((kv_tags(&kv), snapshot.unwrap().to_vec()), (vec![0, 1, 2], vec![0xAB; 30]));
        assert_eq!((c.stats().disk_reads, c.stats().bytes), (4, 60));
        c.load(&keys).unwrap();
        assert_eq!(c.stats().disk_reads, 4, "a second load is served from RAM");
        drop(c);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn blocks_evicted_from_ram_are_still_served_from_the_directory() {
        let root = scratch("spill");
        let mut c = open_dir(&root, 25, 1 << 20, PrefixCacheSave::Always, false);
        let p = prompt(3 * BLOCK, 2);
        let keys = block_keys(&p);
        fill(&mut c, &p, 10, 0, &[]);
        assert!(c.stats().bytes <= 25);
        assert_eq!(c.lookup(&p, &keys, false), Hit { blocks: 3, resume: 3 });
        assert_eq!(kv_tags(&c.load(&keys).unwrap().0), vec![0, 1, 2]);
        assert!(c.stats().disk_reads > 0);
        drop(c);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn the_reused_policy_saves_an_entry_on_its_second_use() {
        let root = scratch("reused");
        let mut c = open_dir(&root, 1 << 20, 1 << 20, PrefixCacheSave::Reused, false);
        let p = prompt(2 * BLOCK, 3);
        let keys = block_keys(&p);
        fill(&mut c, &p, 10, 30, &[2]);
        assert_eq!(c.stats().disk_blocks, 0);
        let hit = c.lookup(&p, &keys, true);
        c.record(&keys[..hit.blocks], hit.resume);
        c.flush();
        assert_eq!((c.stats().disk_blocks, c.stats().disk_snapshots), (2, 1));
        drop(c);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn the_directory_budget_keeps_the_most_used_prefix() {
        let root = scratch("disk-budget");
        let one_offs: Vec<Vec<u32>> = (0..5).map(|i| prompt(2 * BLOCK, 200 + i)).collect();
        let system = {
            let mut c = open_dir(&root, 1 << 20, 100, PrefixCacheSave::Always, false);
            let system = hot_system_prompt(&mut c, 3);
            for p in &one_offs { fill(&mut c, p, 10, 30, &[2]); }
            system
        };
        let c = open_dir(&root, 1 << 20, 100, PrefixCacheSave::Always, false);
        assert_eq!(c.lookup(&system, &block_keys(&system), true), Hit { blocks: 2, resume: 2 });
        let last = &one_offs[4];
        assert_eq!(c.lookup(last, &block_keys(last), true), Hit { blocks: 2, resume: 2 });
        assert!(one_offs[..4].iter().all(|p| !c.contains(block_keys(p)[0])));
        drop(c);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn pins_are_saved_whatever_the_policy_and_survive_a_restart() {
        let root = scratch("pins");
        let system = prompt(2 * BLOCK, 7);
        {
            let mut c = open_dir(&root, 1 << 20, 1 << 20, PrefixCacheSave::Reused, false);
            fill(&mut c, &system, 10, 30, &[2]);
            assert_eq!(c.stats().disk_blocks, 0);
            c.pin(&block_keys(&system));
        }
        let c = open_dir(&root, 1 << 20, 1 << 20, PrefixCacheSave::Reused, false);
        let s = c.stats();
        assert_eq!((s.pinned, s.disk_blocks, s.disk_snapshots), (2, 2, 1));
        assert_eq!(c.lookup(&system, &block_keys(&system), true), Hit { blocks: 2, resume: 2 });
        drop(c);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn saving_a_snapshot_never_evicts_the_rest_of_the_chain_being_recorded() {
        let root = scratch("record-chain");
        let mut c = open_dir(&root, 1 << 20, 40, PrefixCacheSave::Reused, false);
        let p = prompt(4 * BLOCK, 9);
        let keys = block_keys(&p);
        fill(&mut c, &p, 10, 30, &[2]);
        for _ in 0..2 {
            let hit = c.lookup(&p, &keys, true);
            assert_eq!(hit, Hit { blocks: 4, resume: 2 });
            c.record(&keys[..hit.blocks], hit.resume);
        }
        c.flush();
        let s = c.stats();
        assert_eq!((s.disk_blocks, s.disk_snapshots), (4, 0), "the chain stays saved; the snapshot does not fit");
        drop(c);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_failed_write_takes_back_everything_that_depends_on_it() {
        let root = scratch("failed-write");
        let mut c = PrefixCache::new(1 << 20);
        let options = DiskOptions { readonly: false, reserve: u64::MAX / 2, block_tokens: BLOCK, kv: Default::default() };
        c.attach_disk(Disk::open(&root, 7, "test", options).unwrap(), 1 << 20, PrefixCacheSave::Always, 10, 30);
        let p = prompt(3 * BLOCK, 1);
        fill(&mut c, &p, 10, 30, &[3]);
        c.flush();
        let s = c.stats();
        assert_eq!((s.disk_blocks, s.disk_snapshots, s.disk_bytes), (0, 0, 0));
        assert_eq!(c.lookup(&p, &block_keys(&p), true), Hit { blocks: 3, resume: 3 }, "RAM still holds it all");
        drop(c);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_damaged_file_is_dropped_and_the_prefix_shrinks_to_what_is_intact() {
        let root = scratch("damaged");
        let p = prompt(3 * BLOCK, 4);
        let keys = block_keys(&p);
        {
            let mut c = open_dir(&root, 1 << 20, 1 << 20, PrefixCacheSave::Always, false);
            fill(&mut c, &p, 10, 30, &[1, 3]);
        }
        let file = root.join(format!("{:016x}", 7)).join(format!("{:016x}.kv", keys[1]));
        let mut bytes = std::fs::read(&file).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&file, bytes).unwrap();
        let mut c = open_dir(&root, 1 << 20, 1 << 20, PrefixCacheSave::Always, false);
        assert_eq!(c.lookup(&p, &keys, true), Hit { blocks: 3, resume: 3 });
        assert!(c.load(&keys).is_none());
        assert_eq!(c.lookup(&p, &keys, true), Hit { blocks: 1, resume: 1 });
        assert_eq!(kv_tags(&c.load(&keys[..1]).unwrap().0), vec![0]);
        drop(c);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_read_only_directory_serves_blocks_but_never_changes() {
        let root = scratch("read-only");
        let p = prompt(2 * BLOCK, 5);
        {
            let mut c = open_dir(&root, 1 << 20, 1 << 20, PrefixCacheSave::Always, false);
            fill(&mut c, &p, 10, 30, &[2]);
        }
        let listing = || {
            let mut names: Vec<_> = std::fs::read_dir(root.join(format!("{:016x}", 7))).unwrap()
                .map(|e| e.unwrap().file_name()).collect();
            names.sort();
            names
        };
        let before = listing();
        {
            let mut c = open_dir(&root, 1 << 20, 10, PrefixCacheSave::Always, true);
            assert_eq!(c.lookup(&p, &block_keys(&p), true), Hit { blocks: 2, resume: 2 });
            assert!(c.load(&block_keys(&p)).is_some());
            fill(&mut c, &prompt(2 * BLOCK, 6), 10, 30, &[2]);
            assert_eq!(c.stats().disk_blocks, 2);
        }
        assert_eq!(listing(), before);
        let _ = std::fs::remove_dir_all(root);
    }
}
