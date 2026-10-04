//! The prompt-prefix cache on CUDA, end to end on the Qwen3.5 test file: a repeated
//! prompt resumes from its cached blocks, a prompt that diverges inside a block leaves a
//! snapshot at the branch for the next one, and a cache directory restores blocks into a
//! freshly loaded model.
//!
//! GPU-only, and the model file is not in the repository: the tests run when
//! `OJAS_DECISION_MODELS` names the directory holding `tinyopenjev-Q8_0.gguf`:
//!
//! ```text
//! OJAS_DECISION_MODELS=~/models/decision cargo test --release -p ojas-cuda \
//!     --test prefix_cache -- --ignored
//! ```
use ojas_core::config::PrefixCacheSave;
use ojas_core::Model;
use ojas_cuda::{CudaSsm, CudaSsmOpts, PrefixOptions};
use ojas_formats::gguf::Gguf;
use std::path::{Path, PathBuf};

const MODEL: &str = "tinyopenjev-Q8_0.gguf";

fn model_path() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("OJAS_DECISION_MODELS").ok()?);
    let path = dir.join(MODEL);
    if path.exists() { Some(path) } else { eprintln!("skip: {} is not in {}", MODEL, dir.display()); None }
}

/// The test model with a prefix cache of `budget` bytes, backed by `dir` when given.
fn load(path: &Path, dir: Option<PathBuf>, budget: usize) -> CudaSsm { load_with(path, dir, budget, false) }

fn load_with(path: &Path, dir: Option<PathBuf>, budget: usize, int8: bool) -> CudaSsm {
    let mut g = Gguf::open(path.to_string_lossy().as_ref()).expect("open the model");
    let prefix = PrefixOptions {
        budget, dir, readonly: false, disk_budget: Some(1 << 30), reserve: 0, save: PrefixCacheSave::Always,
        int8, snap_interval: 512,
    };
    CudaSsm::load_with(&mut g, CudaSsmOpts { context: 2048, prefix: Some(prefix), ..CudaSsmOpts::default() })
        .expect("load the CUDA runner")
}

/// A fixed prompt of `n` tokens.
fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 10 + i.wrapping_mul(2_654_435_761) % 990).collect()
}

/// A fresh, empty directory under the system temporary directory.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ojas-cuda-prefix-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The logits of the prompt's last token, the way the engine asks for them: the reusable
/// prefix is looked up first, then the rest is prefilled from where it resumes.
fn last_logits(m: &CudaSsm, prompt: &[u32]) -> Vec<f32> {
    let last = prompt.len() - 1;
    let start = m.reuse_prefix_len(prompt);
    m.prefill(&prompt[start..last], start);
    m.forward_logits(prompt[last], last).expect("logits")
}

/// Largest difference between two logit vectors relative to their scale.
fn divergence(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale
}

/// Restored blocks are byte copies of what processing wrote; the tail recomputed after
/// them differs only by the order of the split-K sums.
const TOLERANCE: f32 = 1e-3;

#[test]
#[ignore = "needs the CUDA box and the model file; set OJAS_DECISION_MODELS"]
fn a_repeated_prompt_resumes_from_its_cached_blocks() {
    let Some(path) = model_path() else { return };
    let m = load(&path, None, 1 << 30);
    assert!(m.uses_prefix_cache());
    let p = prompt(700);

    let first = last_logits(&m, &p);
    let s = m.prefix_cache_stats();
    assert_eq!((s.lookups, s.hits, s.blocks), (1, 0, 2), "two full blocks of 700 tokens are cached");
    assert!(s.snapshots >= 1, "the prompt's end keeps a snapshot");
    assert_eq!(s.last.reused_tokens, 0);

    let again = last_logits(&m, &p);
    let s = m.prefix_cache_stats();
    assert_eq!((s.lookups, s.hits), (2, 1));
    assert_eq!(s.last.reused_tokens, 512, "the repeat resumes after the deepest snapshot");
    let d = divergence(&first, &again);
    assert!(d < TOLERANCE, "logits after a resume differ by {d:.2e}");

    // A prompt that diverges inside the second block shares only the first, which has
    // no snapshot yet: nothing is restored, and the run leaves one there.
    let mut q = p.clone();
    q[300] ^= 1;
    let fresh_q = last_logits(&m, &q);
    let s = m.prefix_cache_stats();
    assert_eq!((s.last.matched_tokens, s.last.reused_tokens), (256, 0));
    let again_q = last_logits(&m, &q);
    let s = m.prefix_cache_stats();
    assert_eq!(s.last.reused_tokens, 512, "the diverged prompt's own blocks are cached now");
    assert!(divergence(&fresh_q, &again_q) < TOLERANCE);
    let p_once_more = last_logits(&m, &p);
    assert_eq!(m.prefix_cache_stats().last.reused_tokens, 512);
    assert!(divergence(&first, &p_once_more) < TOLERANCE);
}

#[test]
#[ignore = "needs the CUDA box and the model file; set OJAS_DECISION_MODELS"]
fn the_cache_directory_restores_blocks_into_a_reloaded_model() {
    let Some(path) = model_path() else { return };
    let dir = scratch("reload");
    let p = prompt(700);
    let first = {
        let m = load(&path, Some(dir.clone()), 1 << 30);
        let logits = last_logits(&m, &p);
        m.save_prefix_cache();
        let s = m.prefix_cache_stats();
        assert!(s.directory, "the directory is attached");
        assert_eq!((s.disk_blocks, s.disk_snapshots), (2, 1), "both blocks and the end snapshot are on disk");
        logits
    };
    let m = load(&path, Some(dir.clone()), 1 << 30);
    let s = m.prefix_cache_stats();
    assert_eq!((s.blocks, s.disk_blocks, s.disk_reads), (2, 2, 0), "the index describes the blocks before any payload is read");
    let restored = last_logits(&m, &p);
    let s = m.prefix_cache_stats();
    assert_eq!(s.last.reused_tokens, 512);
    assert_eq!(s.last.disk_payloads, 3, "two KV blocks and one snapshot come from the directory");
    let d = divergence(&first, &restored);
    assert!(d < TOLERANCE, "logits after a restore from disk differ by {d:.2e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "needs the CUDA box and the model file; set OJAS_DECISION_MODELS"]
fn a_zero_budget_keeps_no_cache() {
    let Some(path) = model_path() else { return };
    let m = load(&path, None, 0);
    assert!(!m.uses_prefix_cache());
    assert!(Model::prefix_cache_stats(&m).is_none());
    let p = prompt(300);
    let a = last_logits(&m, &p);
    let b = last_logits(&m, &p);
    assert!(divergence(&a, &b) < TOLERANCE);
}

#[test]
#[ignore = "needs the CUDA box and the model file; set OJAS_DECISION_MODELS"]
fn an_int8_directory_restores_approximately_and_nothing_built_on_it_is_cached_as_exact() {
    let Some(path) = model_path() else { return };
    let dir = scratch("int8");
    let p = prompt(700);
    let first = {
        let m = load_with(&path, Some(dir.clone()), 1 << 30, true);
        let logits = last_logits(&m, &p);
        m.save_prefix_cache();
        assert_eq!(m.prefix_cache_stats().disk_blocks, 2);
        logits
    };
    let m = load_with(&path, Some(dir.clone()), 1 << 30, true);
    let restored = last_logits(&m, &p);
    let s = m.prefix_cache_stats();
    assert_eq!((s.last.reused_tokens, s.last.disk_payloads), (512, 3));
    let d = divergence(&first, &restored);
    assert!(d < 5e-2, "logits after an int8 restore differ by {d:.2e}");
    // A longer prompt restored from the int8 directory caches no block past the
    // restore: its third block would be built on rows that are not what processing
    // computed.
    let m = load_with(&path, Some(dir.clone()), 1 << 30, true);
    let mut q = p.clone();
    q.extend(prompt(300));
    let _ = last_logits(&m, &q);
    let s = m.prefix_cache_stats();
    assert_eq!((s.last.reused_tokens, s.last.disk_payloads), (512, 3));
    assert_eq!(s.blocks, 2, "no block is cached on top of an approximate restore");
    let _ = std::fs::remove_dir_all(&dir);
}
