//! Diagnostics for the streamed expert cache.
//!
//! The symptom it was written for: with a 48 GiB pooled expert cache the Flash
//! decoder emits different (coherent) tokens than the same build with a 16 or
//! 32 GiB cache, or with pooling off entirely; the scratch path agrees with
//! llama.cpp at every size, pooled-direct at 48 GiB does not. Output that is
//! coherent but different means slightly wrong expert bytes rather than a wild
//! pointer, so this checks:
//!
//!   1. every cached expert still equals the bytes on disk
//!   2. no two live pooled allocations overlap in GPU address space
//!
//! Neither check needs the GPU, so a failure localises the bug to
//! admission/lifetime rather than to addressing or synchronisation.

use super::DecoderGpu;

#[derive(Debug, Default)]
pub struct ExpertAudit {
    pub entries: usize,
    /// Cached entries whose bytes differ from the source file.
    pub corrupt: Vec<CorruptExpert>,
    /// Pairs of live allocations whose GPU address ranges intersect.
    pub overlaps: Vec<(u64, u64)>,
    /// Entries skipped because their tensor metadata was not found.
    pub unresolved: usize,
    pub bytes_checked: u64,
}

#[derive(Debug)]
pub struct CorruptExpert {
    pub layer: u32,
    pub expert: u32,
    pub kind: u8,
    pub len: usize,
    /// Index of the first differing byte.
    pub first_diff: usize,
    pub cached: u8,
    pub source: u8,
}

impl ExpertAudit {
    /// True only when nothing was found wrong and nothing was skipped.
    /// `unresolved` covers missing metadata and failed source reads; those are not
    /// evidence of agreement and do not count as a pass.
    pub fn ok(&self) -> bool {
        self.corrupt.is_empty() && self.overlaps.is_empty() && self.unresolved == 0
    }
}

impl DecoderGpu<'_> {
    /// Verify the expert cache against the model file.
    ///
    /// Reads each cached expert's source bytes with `pread` — the same fd and
    /// offset the gather's miss path uses — and compares. Also checks that live
    /// pooled allocations do not overlap.
    pub fn audit_expert_cache(&self) -> ExpertAudit {
        let mut audit = ExpertAudit::default();
        let Some(mapped) = self.wt.mapped.as_ref() else { return audit };
        let Some(moe) = self.arch.moe.as_ref() else { return audit };
        let n_expert = moe.n_expert as u64;

        let cache = self.strm.expert_cache.borrow();
        let mut ranges: Vec<(u64, u64, u64)> = Vec::new(); // (start, end, key)

        for (&key, (bytes, _, _)) in cache.map.iter() {
            audit.entries += 1;
            // key = (layer << 40) | (expert << 8) | kind
            let kind = (key & 0xff) as u8;
            let expert = ((key >> 8) & 0xffff_ffff) as u32;
            let layer = (key >> 40) as u32;
            let name = match kind {
                0 => "ffn_gate_exps.weight",
                1 => "ffn_up_exps.weight",
                2 => "ffn_down_exps.weight",
                _ => {
                    audit.unresolved += 1;
                    continue;
                }
            };
            let Some(&(part, abs, rawlen, _)) =
                self.strm.stream_meta.get(&format!("blk.{layer}.{name}"))
            else {
                audit.unresolved += 1;
                continue;
            };
            let stride = rawlen / n_expert;
            let len = bytes.len();
            if len as u64 != stride {
                audit.unresolved += 1;
                continue;
            }

            let mut source = vec![0u8; len];
            let off = (abs + expert as u64 * stride) as i64;
            let got = unsafe {
                libc::pread(
                    mapped.fd(part),
                    source.as_mut_ptr() as *mut std::ffi::c_void,
                    len,
                    off,
                )
            };
            if got != len as isize {
                audit.unresolved += 1;
                continue;
            }
            audit.bytes_checked += len as u64;

            let cached = unsafe { std::slice::from_raw_parts(bytes.as_ptr(), len) };
            if cached != source.as_slice() {
                let first_diff = cached
                    .iter()
                    .zip(&source)
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                audit.corrupt.push(CorruptExpert {
                    layer,
                    expert,
                    kind,
                    len,
                    first_diff,
                    cached: cached[first_diff],
                    source: source[first_diff],
                });
            }

            if let Some(buffer) = cache.metal_buffer(key) {
                let start = buffer.address();
                ranges.push((start, start + len as u64, key));
            }
        }

        // Two live allocations sharing GPU address space would let one expert's
        // admission silently rewrite another's weights — coherent but different output.
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            let (a_start, a_end, a_key) = pair[0];
            let (b_start, _, b_key) = pair[1];
            let _ = a_start;
            if b_start < a_end {
                audit.overlaps.push((a_key, b_key));
            }
        }
        audit
    }
}


/// One slot of the expert address table, as the expert kernel will index it.
#[derive(Debug, Clone)]
pub struct SlotRecord {
    pub layer: u32,
    pub kind: u8,
    pub slot: u32,
    pub expert: u32,
    pub address: u64,
    pub offset_in_tensor: u64,
    pub len: usize,
    /// FNV-1a over the same sampled ranges the GPU kernel hashes, computed from the
    /// source file rather than from the CPU view of whatever the table points at.
    /// Hashing the pointed-at bytes on both sides would agree even when the table
    /// holds the wrong expert's allocation; hashing the intended expert's bytes from
    /// disk makes the comparison test that the GPU reads the expert the router chose.
    pub expected_hash: u32,
    /// False when the address could not be mapped to a CPU view, or the hashed
    /// range did not fit inside the owning allocation. Such a slot is not evidence
    /// of agreement.
    pub resolved: bool,
    /// Which buffer the consumer reads for this layer. Reading the direct table on a
    /// scratch layer yields stale addresses from whichever direct layer last wrote
    /// them, interpreted with this layer's stride, and reports false mismatches.
    pub direct: bool,
    /// GGML type of this weight tensor. Q8_0 down weights disqualify a layer from
    /// direct addressing, so the type explains the path.
    pub ggml_type: u32,
}

/// The sampling the GPU kernel performs, reproduced exactly on the host.
/// Any change here must change `EXPERT_HASH_KERNEL` identically or every
/// comparison becomes a false positive.
pub const HASH_SAMPLES: u32 = 64;

pub fn sample_hash(base: *const u8, stride: usize, samples: u32) -> u32 {
    let span = stride.saturating_sub(16);
    let mut h: u32 = 2166136261;
    for s in 0..samples {
        let off = if samples > 1 { (span as u64 * s as u64 / (samples as u64 - 1)) as usize } else { 0 };
        for i in 0..16usize {
            let b = unsafe { *base.add(off + i) };
            h ^= b as u32;
            h = h.wrapping_mul(16777619);
        }
    }
    h
}

/// Which layer to instrument, from OJAS_EXPERT_HASH_LAYER. None = disabled.
pub fn instrumented_layer() -> Option<usize> {
    ojas_core::config::var("OJAS_EXPERT_HASH_LAYER").ok()?.parse().ok()
}
