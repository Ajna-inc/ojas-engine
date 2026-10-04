//! The prompt-prefix cache on CUDA.
//!
//! [`CudaSsm`] keeps slot 0's prompt prefixes as `ojas_prefix` blocks: a block holds the
//! attention KV rows of its [`BLOCK`] positions (K then V for each attention layer) and,
//! where the plan asks for one, a snapshot of the recurrent state after its last token.
//! Blocks live in host memory within the cache budget, and in the cache directory when
//! one is configured, where they outlive the process. Payloads cross PCIe: a block's rows
//! are read back after the chunk that completes it, and written back when a prompt
//! resumes from it.
//!
//! The bookkeeping ([`PrefixCache`], [`Plan`]) and the directory format are
//! `ojas_prefix`'s; this module copies state in and out of the device and decides what a
//! prefill caches, as the Metal decoder's prefill session does.

use crate::qwen35::{CudaSsm, Want};
use ojas_core::config::{EngineConfig, PrefixCacheSave};
use ojas_core::{PrefixCacheStats, PrefixRestore};
use ojas_prefix::prefix_cache::{self, Plan, PrefixCache, BLOCK};
use ojas_prefix::prefix_disk::{self, DiskOptions, KvFormat};
use std::path::PathBuf;

/// How a [`CudaSsm`] caches prompt prefixes ([`CudaSsmOpts::prefix`](crate::CudaSsmOpts)).
#[derive(Clone, Debug)]
pub struct PrefixOptions {
    /// Host memory for cached blocks, in bytes; 0 keeps no cache.
    pub budget: usize,
    /// Directory the blocks are also kept in, across runs.
    pub dir: Option<PathBuf>,
    /// Use the directory's blocks without changing it.
    pub readonly: bool,
    /// Bytes the directory may hold; `None` is 20 GB, at most a quarter of the free space.
    pub disk_budget: Option<u64>,
    /// Free space every write leaves on the volume.
    pub reserve: u64,
    /// When a block is written to the directory.
    pub save: PrefixCacheSave,
    /// Store KV rows in the directory as int8 (half the size; a restore is then close
    /// to, not identical with, processing).
    pub int8: bool,
    /// Tokens between the snapshots a long prompt keeps inside it.
    pub snap_interval: usize,
}

impl PrefixOptions {
    /// The engine's settings: `--prefix-cache-gb`, `--prefix-cache-dir` and the rest.
    pub fn from_config(cfg: &EngineConfig) -> Self {
        let gb = |v: f64| (v.max(0.0) * 1e9) as u64;
        PrefixOptions {
            budget: if cfg.no_prefix_reuse { 0 } else { ojas_prefix::budget(cfg, ojas_prefix::physical_ram_bytes()) },
            dir: cfg.prefix_cache_dir.clone(),
            readonly: cfg.prefix_cache_readonly,
            disk_budget: cfg.prefix_cache_disk_gb.map(gb),
            reserve: gb(cfg.prefix_cache_reserve_gb),
            save: cfg.prefix_cache_save,
            int8: cfg.prefix_cache_disk_int8,
            snap_interval: cfg.snap_interval,
        }
    }
}

/// The cache and the prefill in progress against it.
pub(crate) struct PrefixSession {
    cache: PrefixCache,
    plan: Option<Plan>,
    /// Message boundaries of the next prompt (`Model::set_prefix_marks`).
    marks: Vec<usize>,
    /// Whether the next lookups may restore (`Model::set_prefix_reuse`).
    reuse: bool,
    /// What the last lookup restored.
    last: PrefixRestore,
    snap_interval: usize,
}

impl PrefixSession {
    /// A session for `opts`; without options, or when the prefill chunk does not tile
    /// a block, one that caches nothing.
    pub(crate) fn new(opts: Option<&PrefixOptions>, chunk: usize) -> Self {
        let budget = match opts {
            Some(o) if BLOCK.is_multiple_of(chunk) => o.budget,
            _ => 0,
        };
        PrefixSession {
            cache: PrefixCache::new(budget),
            plan: None,
            marks: Vec::new(),
            reuse: true,
            last: PrefixRestore::default(),
            snap_interval: opts.map_or(2048, |o| o.snap_interval),
        }
    }
}

impl CudaSsm {
    /// Whether prefill goes through the prompt-prefix cache.
    pub fn uses_prefix_cache(&self) -> bool { self.prefix.borrow().cache.enabled() }

    pub(crate) fn set_prefix_reuse(&self, on: bool) { self.prefix.borrow_mut().reuse = on }

    pub(crate) fn set_prefix_marks(&self, positions: &[usize]) { self.prefix.borrow_mut().marks = positions.to_vec() }

    /// Layers with an attention KV cache.
    fn kv_layers(&self) -> Vec<usize> {
        (0..self.layers.len()).filter(|&l| self.layers[l].has_kv()).collect()
    }

    /// Bytes of one KV row.
    fn kv_row_bytes(&self) -> usize { self.n_kv * self.hd * self.kv_elem() }

    /// Slot 0's KV rows for positions `[p0, p0 + BLOCK)`: K then V for each attention layer.
    fn kv_block(&self, p0: usize) -> Vec<u8> {
        let row = self.kv_row_bytes();
        let st = self.st.borrow();
        let mut out = vec![0u8; self.kv_layers().len() * 2 * BLOCK * row];
        let mut at = 0;
        for l in self.kv_layers() {
            for cache in [st.kc[l].as_ref().unwrap(), st.vc[l].as_ref().unwrap()] {
                self.gpu.read_bytes(cache, p0 * row, &mut out[at..at + BLOCK * row]).expect("cuda read");
                at += BLOCK * row;
            }
        }
        out
    }

    /// The inverse of [`CudaSsm::kv_block`].
    fn write_kv_block(&self, p0: usize, bytes: &[u8]) {
        let row = self.kv_row_bytes();
        let st = &mut *self.st.borrow_mut();
        let mut at = 0;
        for l in self.kv_layers() {
            for cache in [st.kc[l].as_mut().unwrap(), st.vc[l].as_mut().unwrap()] {
                self.gpu.write_bytes(cache, p0 * row, &bytes[at..at + BLOCK * row]).expect("cuda write");
                at += BLOCK * row;
            }
        }
    }

    /// Slot 0's recurrent state: for each SSM layer its conv state then its SSM state.
    pub fn state_bytes(&self) -> Vec<u8> {
        let (cb, sb, _) = self.slot_bytes();
        let st = self.st.borrow();
        let mut out = Vec::new();
        for l in 0..self.layers.len() {
            if let (Some(c), Some(s)) = (&st.conv[l], &st.ssm[l]) {
                let at = out.len();
                out.resize(at + cb + sb, 0);
                self.gpu.read_bytes(c, 0, &mut out[at..at + cb]).expect("cuda read");
                self.gpu.read_bytes(s, 0, &mut out[at + cb..at + cb + sb]).expect("cuda read");
            }
        }
        out
    }

    /// Restore slot 0's recurrent state from [`CudaSsm::state_bytes`].
    pub fn set_state_bytes(&self, bytes: &[u8]) {
        let (cb, sb, _) = self.slot_bytes();
        let st = &mut *self.st.borrow_mut();
        let mut at = 0;
        for l in 0..self.layers.len() {
            if let (Some(c), Some(s)) = (&mut st.conv[l], &mut st.ssm[l]) {
                self.gpu.write_bytes(c, 0, &bytes[at..at + cb]).expect("cuda write");
                self.gpu.write_bytes(s, 0, &bytes[at + cb..at + cb + sb]).expect("cuda write");
                at += cb + sb;
            }
        }
        assert_eq!(at, bytes.len(), "a snapshot of another model");
    }

    /// Start a new sequence from the longest cached prefix of `prompt` and plan what
    /// this prefill adds to the cache. With reuse forbidden nothing is restored, but the
    /// lookup still tells the plan where the prompt branches. Returns the position
    /// processing resumes at, a multiple of `BLOCK`.
    pub(crate) fn cache_resume(&self, prompt: &[u32]) -> usize {
        let t0 = std::time::Instant::now();
        let keys = prefix_cache::block_keys(prompt);
        let mut px = self.prefix.borrow_mut();
        let reads = px.cache.stats().disk_reads;
        // A block that fails to load from the cache directory is dropped, so the next
        // lookup stops short of it.
        let (hit, used, (kv, snapshot)) = loop {
            let mut hit = px.cache.lookup(prompt, &keys, true);
            let used = if px.reuse { hit.blocks } else { 0 };
            hit.resume = hit.resume.min(used);
            if let Some(payloads) = px.cache.load(&keys[..hit.resume]) { break (hit, used, payloads); }
        };
        self.reset();
        let n = hit.resume * BLOCK;
        px.cache.record(&keys[..used], hit.resume);
        if n > 0 {
            for (b, bytes) in kv.iter().enumerate() {
                self.write_kv_block(b * BLOCK, bytes);
            }
            self.set_state_bytes(&snapshot.expect("a resume block carries a snapshot"));
        }
        let disk_payloads = px.cache.stats().disk_reads - reads;
        let marks = std::mem::take(&mut px.marks);
        let mut plan = Plan::new(prompt.to_vec(), keys, hit, &marks);
        // KV restored from an int8 directory is close to, not identical with, what
        // processing computes, so nothing built on it is cached as exact.
        if disk_payloads > 0 && !px.cache.disk_kv_exact() { plan.exact = plan.exact.min(n); }
        px.plan = Some(plan);
        px.last = PrefixRestore {
            matched_tokens: hit.blocks * BLOCK, reused_tokens: n, doc_reused_tokens: 0, disk_payloads,
            restore_us: t0.elapsed().as_micros() as u64,
        };
        n
    }

    /// Where a prefill at position 0 starts. A plan made for this prompt by
    /// `reuse_prefix_len` that nothing has started yet already ran the lookup and
    /// reset the state; anything else looks the prompt up now.
    fn cache_start(&self, tokens: &[u32]) -> usize {
        let fresh = self.prefix.borrow_mut().plan.as_mut().is_some_and(|p| {
            let k = p.tokens.len().min(tokens.len());
            let ok = !p.started && p.resume == 0 && p.tokens[..k] == tokens[..k];
            p.started |= ok;
            ok
        });
        if fresh { return 0; }
        let n = self.cache_resume(&tokens[..tokens.len().saturating_sub(1)]);
        if let Some(p) = self.prefix.borrow_mut().plan.as_mut() { p.started = true; }
        n
    }

    /// A prefill continuing at `base`: keep the plan only if it describes these tokens.
    fn cache_continue(&self, base: usize, tokens: &[u32]) {
        let mut px = self.prefix.borrow_mut();
        let keep = px.plan.as_ref().is_some_and(|p| {
            let end = (base + tokens.len()).min(p.tokens.len());
            base <= end && p.tokens[base..end] == tokens[..end - base]
        });
        match px.plan.as_mut() {
            Some(p) if keep => {
                p.started = true;
                // Chunks that start off the chunk grid run through different kernel
                // shapes than a fresh prefill, so nothing after them is exact.
                if !base.is_multiple_of(self.opts.chunk) { p.exact = p.exact.min(base); }
            }
            _ => px.plan = None,
        }
    }

    /// After the prefill chunk that ended at `pos`: cache the block that ended there,
    /// with a snapshot where the plan wants one.
    fn cache_capture(&self, pos: usize) {
        let b = pos / BLOCK;
        let (key, parent, tokens, wants) = {
            let px = self.prefix.borrow();
            let Some(plan) = px.plan.as_ref() else { return };
            if !pos.is_multiple_of(BLOCK) || b <= plan.resume || b > plan.keys.len() || pos > plan.exact { return; }
            let parent = if b > 1 { plan.keys[b - 2] } else { prefix_cache::ROOT };
            (plan.keys[b - 1], parent, plan.tokens[pos - BLOCK..pos].to_vec(), plan.wants_snapshot(b, px.snap_interval))
        };
        // A key held by another block (a hash collision) is left alone: neither its KV
        // nor its snapshot may come from this prompt.
        let present = {
            let px = self.prefix.borrow();
            if px.cache.matches(key, parent, &tokens) { true } else if px.cache.contains(key) { return } else { false }
        };
        if !present {
            let kv = self.kv_block(pos - BLOCK);
            if !self.prefix.borrow_mut().cache.insert(parent, &tokens, kv) { return; }
        }
        if wants && !self.prefix.borrow().cache.has_snapshot(key) {
            let snapshot = self.state_bytes();
            let mut px = self.prefix.borrow_mut();
            if px.cache.attach_snapshot(key, snapshot) {
                if let Some(plan) = px.plan.as_mut() { plan.took_snapshot(b); }
            }
        }
    }

    /// `Model::prefill` through the cache: a prompt starting at position 0 resumes from
    /// its longest cached prefix, and every block the prefill completes is cached.
    pub(crate) fn prefill_cached(&self, tokens: &[u32], base_pos: usize) {
        let (tokens, mut pos) = if base_pos == 0 {
            let n = self.cache_start(tokens);
            (&tokens[n..], n)
        } else {
            self.cache_continue(base_pos, tokens);
            (tokens, base_pos)
        };
        for chunk in tokens.chunks(self.opts.chunk) {
            self.forward(Some(chunk), None, chunk.len(), None, pos, Want::Nothing).expect("CUDA qwen35 prefill");
            pos += chunk.len();
            self.cache_capture(pos);
        }
        self.prefix.borrow_mut().cache.sync();
    }

    /// The longest cached prefix of `full_prompt` that a prefill at position 0 will
    /// resume from (`Model::reuse_prefix_len`): the lookup restores it now and leaves the
    /// plan for that prefill.
    pub(crate) fn reuse_prefix_len(&self, full_prompt: &[u32]) -> usize {
        self.prefix.borrow_mut().last = PrefixRestore::default();
        if !self.uses_prefix_cache() || full_prompt.len() < 2 { return 0; }
        self.cache_resume(&full_prompt[..full_prompt.len() - 1])
    }

    /// Process `tokens` into the prefix cache without generating: every full block is
    /// cached, with a snapshot after the last, and pinned when `pin` is set. Returns the
    /// tokens now cached.
    pub fn warm_prefix(&self, tokens: &[u32], pin: bool) -> usize {
        let prompt = &tokens[..tokens.len() / BLOCK * BLOCK];
        self.prefix.borrow_mut().marks.clear();
        if prompt.is_empty() || !self.uses_prefix_cache() { return 0; }
        let reuse = std::mem::replace(&mut self.prefix.borrow_mut().reuse, true);
        let start = self.cache_resume(prompt);
        if start < prompt.len() { self.prefill_cached(&prompt[start..], start); }
        let keys = prefix_cache::block_keys(prompt);
        let mut px = self.prefix.borrow_mut();
        px.reuse = reuse;
        if pin { px.cache.pin(&keys); }
        px.cache.sync();
        px.cache.lookup(prompt, &keys, true).resume * BLOCK
    }

    /// Occupancy and hit counts of the prompt-prefix cache.
    pub fn prefix_cache_stats(&self) -> PrefixCacheStats {
        let px = self.prefix.borrow();
        let s = px.cache.stats();
        PrefixCacheStats {
            blocks: s.blocks, snapshots: s.snapshots, bytes: s.bytes, budget: s.budget,
            lookups: s.lookups, hits: s.hits, reused_tokens: s.reused_tokens, block_tokens: BLOCK,
            directory: s.disk_budget > 0 || s.disk_blocks > 0, disk_blocks: s.disk_blocks,
            disk_snapshots: s.disk_snapshots, disk_bytes: s.disk_bytes, disk_budget: s.disk_budget,
            disk_reads: s.disk_reads, pinned_blocks: s.pinned, evictions: s.evictions, last: px.last,
            docs: Default::default(),
        }
    }

    /// Finish writing the cache directory.
    pub fn save_prefix_cache(&self) { self.prefix.borrow_mut().cache.flush() }

    /// Release every pinned prefix.
    pub fn unpin_prefix_cache(&self) { self.prefix.borrow_mut().cache.unpin_all() }

    /// Everything besides the weights and state layout that decides the numbers a prefill
    /// computes: the engine version, every kernel's source, the GPU and the GEMM mode.
    fn numerics_identity(&self) -> String {
        let mut kernels: Vec<(&str, &str)> = crate::kernels::families().collect();
        kernels.sort_unstable();
        let source: Vec<u8> = kernels.iter().flat_map(|(name, src)| name.bytes().chain(src.bytes())).collect();
        format!("{} {:016x} {} {:?}", env!("CARGO_PKG_VERSION"), prefix_disk::checksum(&source),
                self.gpu.device_name(), self.opts.gemm)
    }

    /// Back the prefix cache with `opts.dir`, when set. Blocks saved there by any earlier
    /// run of this model build are reusable at once. The subdirectory is chosen by the
    /// model's weights and the state layout, so a different build never reads them. A
    /// directory that cannot be opened leaves the cache in memory only.
    pub(crate) fn open_prefix_dir(&self, opts: &PrefixOptions, model_files: &[PathBuf]) {
        let Some(root) = opts.dir.as_deref() else { return };
        if !self.uses_prefix_cache() {
            tracing::warn!(target: "prefix", "this model and configuration keep no prompt-prefix cache; {} is not used",
                root.display());
            return;
        }
        if model_files.is_empty() {
            tracing::warn!(target: "prefix", "the model was not opened from files; caching prompts in memory only");
            return;
        }
        let kv_len = self.kv_layers().len() * 2 * BLOCK * self.kv_row_bytes();
        let snapshot_len = {
            let (cb, sb, _) = self.slot_bytes();
            self.layers.len().saturating_sub(self.kv_layers().len()) * (cb + sb)
        };
        // Int8 storage encodes f16 rows; the exact mode's f32 rows are stored as they are.
        let kv = if opts.int8 && self.kv_elem() == 2 { KvFormat::Q8 } else { KvFormat::F16 };
        let opened = prefix_disk::fingerprint(model_files).and_then(|weights| {
            let build = [weights, self.kv_elem() as u64, self.opts.chunk as u64, kv_len as u64, snapshot_len as u64,
                BLOCK as u64, kv as u64];
            let mut salt: Vec<u8> = build.iter().flat_map(|v| v.to_le_bytes()).collect();
            salt.extend_from_slice(self.numerics_identity().as_bytes());
            let salt = prefix_disk::checksum(&salt);
            let name = model_files[0].file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
            let format = if kv == KvFormat::Q8 { ", int8 KV" } else { "" };
            let about = format!("{name} on CUDA{format}\n");
            prefix_disk::Disk::open(root, salt, &about, DiskOptions {
                readonly: opts.readonly, reserve: opts.reserve, block_tokens: BLOCK, kv,
            })
        });
        let opened = match opened {
            Ok(opened) => opened,
            Err(e) => {
                tracing::warn!(target: "prefix", "cache directory {}: {e}; caching prompts in memory only", root.display());
                return;
            }
        };
        let budget = opts.disk_budget.unwrap_or_else(|| 20_000_000_000u64.min(prefix_disk::free_space(opened.disk.dir()) / 4));
        let (dir, blocks, writable) = (opened.disk.dir().display().to_string(), opened.records.len(), opened.disk.writable());
        self.prefix.borrow_mut().cache.attach_disk(opened, budget as usize, opts.save, kv_len, snapshot_len);
        if writable {
            tracing::info!(target: "prefix", "cache directory {dir}: {blocks} blocks, cap {:.1} GB", budget as f64 / 1e9);
        } else {
            tracing::info!(target: "prefix", "cache directory {dir}: {blocks} blocks, read-only");
        }
    }
}
