//! Resident expert cache: the CUDA half of what `moe_stream.rs` does on Metal.
//!
//! A routed MoE layer touches `KSEL` of `E` experts per token, so the model does not have to be
//! resident: keep a bounded arena of expert slots in VRAM, serve what is cached, and stream the
//! misses in. That is how a 27 GB model runs in a 3.18 GB resident set on Metal, and the
//! measured PCIe headroom indicates the same shape works here.
//!
//! Two differences from Metal follow from the hardware:
//!
//! * Apple's unified memory lets a cached expert be a CPU pointer the GPU can read. Here a miss
//!   must cross PCIe, so it lands in page-locked staging first: pinned is 1.58× pageable on the
//!   card this was measured on, and only pinned can overlap with compute.
//! * Eviction is LRU by slot, and a slot is reused only once the stream that read it has caught
//!   up. `ExpertPool` on Metal does the same bookkeeping through its `Ranges` type.
//!
//! The arena is one device allocation; kernels address a slot by byte offset, which is what
//! `KernelRuntime::dispatch` already takes alongside each buffer.

use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::{CuBuf, CudaGpu};

/// What a gather did, so a caller can report hit rate without threading counters through.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GatherStats {
    pub hits: usize,
    pub misses: usize,
    pub bytes_streamed: usize,
    pub evictions: usize,
}

/// Bounded VRAM arena of fixed-size expert slots with LRU eviction.
pub struct ExpertCache {
    arena: CuBuf,
    staging: cudarc::driver::PinnedHostSlice<u8>,
    expert_bytes: usize,
    slots: usize,
    /// expert id -> slot
    resident: HashMap<u32, usize>,
    /// slot -> (expert id, last use tick); `None` means free
    slot_of: Vec<Option<(u32, u64)>>,
    tick: u64,
    pub stats: GatherStats,
}

impl ExpertCache {
    /// `budget_bytes` is the VRAM ceiling; the arena holds as many whole experts as fit, and at
    /// least one. A budget too small for one token's working set is reported by `gather` rather
    /// than absorbed.
    pub fn new(gpu: &CudaGpu, budget_bytes: usize, expert_bytes: usize) -> Result<Self> {
        if expert_bytes == 0 {
            return Err(anyhow!("expert_bytes must be non-zero"));
        }
        let slots = (budget_bytes / expert_bytes).max(1);
        Ok(Self {
            arena: gpu.alloc_bytes(slots * expert_bytes)?,
            staging: gpu.alloc_pinned(expert_bytes)?,
            expert_bytes,
            slots,
            resident: HashMap::with_capacity(slots),
            slot_of: vec![None; slots],
            tick: 0,
            stats: GatherStats::default(),
        })
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Bytes of VRAM the arena holds; compare against the model size when quoting a
    /// resident-set figure.
    pub fn resident_bytes(&self) -> usize {
        self.slots * self.expert_bytes
    }

    /// The arena, for passing to a kernel alongside the offsets `gather` returned.
    pub fn arena(&self) -> &CuBuf {
        &self.arena
    }

    /// Make every requested expert resident and return each one's byte offset into the arena, in
    /// the order asked for.
    ///
    /// `load` fills the staging slice with one expert's bytes; it is where a real caller does its
    /// `pread` from the GGUF. Misses are uploaded on the compute stream, so a kernel queued after
    /// this call sees them without an explicit sync.
    pub fn gather<F>(&mut self, gpu: &CudaGpu, ids: &[u32], mut load: F) -> Result<Vec<u64>>
    where
        F: FnMut(u32, &mut [u8]) -> Result<()>,
    {
        // a token asking for more distinct experts than the arena holds would evict a slot the
        // same gather still needs, so refuse instead of thrashing
        let mut distinct: Vec<u32> = ids.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() > self.slots {
            return Err(anyhow!(
                "gather wants {} distinct experts but the arena holds {}; raise the budget",
                distinct.len(),
                self.slots
            ));
        }

        let mut offsets = Vec::with_capacity(ids.len());
        for &id in ids {
            self.tick += 1;
            if let Some(&slot) = self.resident.get(&id) {
                self.slot_of[slot] = Some((id, self.tick));
                self.stats.hits += 1;
                offsets.push((slot * self.expert_bytes) as u64);
                continue;
            }
            // miss: a free slot if there is one, else the least recently used. The
            // distinct-vs-slots check above guarantees the victim is not an expert this gather
            // still needs, since every id touched in this call has just been ticked.
            let slot = match self.slot_of.iter().position(|s| s.is_none()) {
                Some(free) => free,
                None => {
                    let (victim, &(vid, _)) = self
                        .slot_of
                        .iter()
                        .enumerate()
                        .filter_map(|(i, s)| s.as_ref().map(|e| (i, e)))
                        .min_by_key(|(_, (_, used))| *used)
                        .ok_or_else(|| anyhow!("no evictable slot"))?;
                    self.resident.remove(&vid);
                    self.stats.evictions += 1;
                    victim
                }
            };
            load(id, self.staging.as_mut_slice()?)?;
            gpu.upload_pinned_at(&self.staging, &mut self.arena, slot * self.expert_bytes)?;
            self.resident.insert(id, slot);
            self.slot_of[slot] = Some((id, self.tick));
            self.stats.misses += 1;
            self.stats.bytes_streamed += self.expert_bytes;
            offsets.push((slot * self.expert_bytes) as u64);
        }
        Ok(offsets)
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.hits + self.stats.misses;
        if total == 0 {
            0.0
        } else {
            self.stats.hits as f64 / total as f64
        }
    }
}
