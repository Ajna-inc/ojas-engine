//! Keeping a training run inside the machine's memory.
//!
//! A tape keeps every activation of a pass until its backward has run, and attention
//! holds a `[heads, T, T]` matrix per layer, so one long prompt can take tens of
//! gigabytes and a step of several of them takes the machine down. Three guards:
//! prompts past a length are not trained on; the items of a step are packed into
//! passes whose estimated activation memory fits a budget, with the gradients summed
//! across them; and the process's resident memory is read before every pass, so a
//! step stops early when the estimate was wrong, well before the system does.

use ojas_arch::text_encoder::TextEncoderSpec;

/// A memory budget for one pass, in bytes, and its share of the machine.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// Most estimated activation bytes one pass may take.
    pub pass: u64,
    /// Resident bytes past which no further pass starts in the step.
    pub resident: u64,
}

impl Budget {
    /// `gb` for activations, or when 0 a third of the machine's memory; the resident
    /// limit is that plus what the process holds when the run starts.
    pub fn new(gb: f64) -> Budget {
        let physical = physical_bytes();
        let pass = if gb > 0.0 { (gb * 1e9) as u64 } else { physical / 3 };
        let baseline = resident_bytes();
        Budget { pass, resident: (baseline + pass).min(physical * 9 / 10) }
    }
}

/// Estimated bytes a pass over sequences of these lengths holds: the tape keeps every
/// activation until the pass ends, with or without a backward, and the backward's
/// gradients come and go a node at a time on top of them.
pub fn activation_bytes(spec: &TextEncoderSpec, lengths: &[usize], train: bool) -> u64 {
    let (d, ffn, heads) = (spec.d as u64, spec.ffn as u64, spec.n_head as u64);
    let blocks = spec.layers as u64 + spec.marker_head.as_ref().map_or(0, |h| h.blocks as u64);
    let per_layer = |t: u64| {
        // scores, probabilities and their gradients; the rows of q, k, v, their
        // rotations and permutations, the residual and norms; the gated MLP's rows.
        let attention = 4 * heads * t * t;
        let rows = 40 * t * d + 8 * t * ffn;
        4 * (attention + rows)
    };
    let total: u64 = lengths.iter().map(|&t| blocks * per_layer(t as u64)).sum();
    if train { total * 5 / 4 } else { total }
}

/// Index ranges of `lengths` packed in order into passes under `budget`; an item
/// too long for a pass of its own is reported in the second list and left out.
pub fn passes(spec: &TextEncoderSpec, lengths: &[usize], budget: u64, train: bool) -> (Vec<std::ops::Range<usize>>, Vec<usize>) {
    let (mut out, mut dropped) = (Vec::new(), Vec::new());
    let mut start = 0;
    let mut used = 0u64;
    for (i, &len) in lengths.iter().enumerate() {
        let cost = activation_bytes(spec, &[len], train);
        if cost > budget {
            if start < i { out.push(start..i); }
            dropped.push(i);
            start = i + 1;
            used = 0;
            continue;
        }
        if used + cost > budget && start < i {
            out.push(start..i);
            start = i;
            used = 0;
        }
        used += cost;
    }
    if start < lengths.len() { out.push(start..lengths.len()); }
    (out, dropped)
}

/// Physical memory of the machine, in bytes.
pub fn physical_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut size: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = c"hw.memsize";
        let rc = unsafe { sysctlbyname(name.as_ptr(), &mut size as *mut u64 as *mut _, &mut len, std::ptr::null_mut(), 0) };
        if rc == 0 && size > 0 { return size; }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    if let Some(kb) = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok()) { return kb * 1024; }
                }
            }
        }
    }
    16 << 30
}

/// Memory this process holds, in bytes, as the system judges it for memory
/// pressure: on macOS the task's physical footprint, which counts the GPU buffers in
/// unified memory that its resident size leaves out.
pub fn resident_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        // `task_vm_info` at its first revision: 38 words, `phys_footprint` the last.
        const TASK_VM_INFO: u32 = 22;
        const WORDS: u32 = 38;
        let mut info = [0u32; WORDS as usize];
        let mut count = WORDS;
        let rc = unsafe { task_info(mach_task_self(), TASK_VM_INFO, info.as_mut_ptr() as *mut _, &mut count) };
        if rc == 0 && count >= WORDS {
            return u64::from(info[36]) | (u64::from(info[37]) << 32);
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(pages) = text.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()) { return pages * 4096; }
        }
    }
    0
}

#[cfg(target_os = "macos")]
extern "C" {
    fn mach_task_self() -> u32;
    fn task_info(task: u32, flavor: u32, info: *mut std::ffi::c_void, count: *mut u32) -> i32;
    fn sysctlbyname(name: *const std::ffi::c_char, old: *mut std::ffi::c_void, old_len: *mut usize, new: *mut std::ffi::c_void, new_len: usize) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> TextEncoderSpec {
        TextEncoderSpec { d: 1024, layers: 28, n_head: 16, hd: 64, ffn: 2624, eps: 1e-5, rope_base: 160000.0, rope_base_local: 10000.0,
                          window: 64, swa_pattern: 3, max_positions: 8192, marker_head: None }
    }

    #[test]
    fn long_prompts_cost_quadratically_and_are_packed_or_dropped() {
        let s = spec();
        let (short, long) = (activation_bytes(&s, &[128], true), activation_bytes(&s, &[2048], true));
        assert!(long > 30 * short, "{long} vs {short}");
        assert!(long > 20 << 30, "a 2k-token prompt keeps over 20 GB: {long}");
        let budget = activation_bytes(&s, &[256, 256], true) + 1;
        let (passes, dropped) = passes(&s, &[256, 256, 256, 4096, 128], budget, true);
        assert_eq!(passes, vec![0..2, 2..3, 4..5]);
        assert_eq!(dropped, vec![3]);
    }

    #[test]
    fn the_machine_and_the_process_are_measured() {
        assert!(physical_bytes() >= 4 << 30);
        assert!(resident_bytes() > 1 << 20);
        let b = Budget::new(0.0);
        assert!(b.pass >= physical_bytes() / 4 && b.resident > b.pass);
    }
}
