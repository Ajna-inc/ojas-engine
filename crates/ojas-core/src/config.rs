//! `EngineConfig` — the single place engine env vars are read.
//!
//! Every knob is captured once at model-load time: env vars are load-time API and
//! nothing may re-read the environment in the token hot path. App-facing per-load
//! options (`kv_gb` / `expert_cache_gb`, the memory-split slider) arrive as
//! explicit `DecoderGpu::load` params rather than env mutation; the `OJAS_*` vars
//! `from_env` reads here are the power-user override behind them.
//!
//! Adding a knob is one field, one line in `from_env`, and a doc row here.
//!
//! Engine code reads [`EngineConfig::current`], not `from_env`, because a host may
//! [`EngineConfig::install`] a config instead of setting environment variables.
//! The CLI starts from this environment baseline, applies command-line flags, and
//! installs the resulting config before model load, so `OJAS_*` names stay
//! compatible while normal CLI use needs no environment variables.
//!
//! Names are frozen API. Read env vars through [`var`] / [`flag`], never
//! `std::env::var` directly, so the read stays in one place.

/// Set at most once, before any model is loaded. Read through
/// [`EngineConfig::current`].
static INSTALLED: std::sync::OnceLock<EngineConfig> = std::sync::OnceLock::new();

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// OJAS_TOPK — MoE adaptive routing: keep only top-K experts (renormalized).
    pub top_k: Option<u32>,
    /// OJAS_NO_MMAP — load requant cache into RAM instead of zero-copy mmap.
    pub no_mmap: bool,
    /// OJAS_PREWARM — touch skeleton pages at load (front-load page faults).
    pub prewarm: bool,
    /// OJAS_NO_SKEL_LOCK — skip mlock'ing the streaming skeleton.
    pub no_skel_lock: bool,
    /// OJAS_MLA_NAIVE — non-absorbed MLA path (testing; absorbed is default).
    pub mla_naive: bool,
    /// OJAS_KV_GB — KV-cache RAM ceiling in GB (None = arch default).
    pub kv_gb: Option<u64>,
    /// OJAS_CTX — hard context-length override.
    pub ctx: Option<usize>,
    /// OJAS_EXPERT_CACHE_GB — streamed-expert LRU cache budget (GB).
    pub expert_cache_gb: f64,
    /// OJAS_SPARSE — page-sparse decode dense-token budget (>0 enables).
    pub sparse: Option<u32>,
    /// OJAS_ADJ_DRAFT — adjacency-harvested draft chain.
    pub adj_draft: bool,
    /// OJAS_NO_SPEC — turn speculative decoding off (A/B control for spec_gate).
    pub no_spec: bool,
    /// OJAS_GEMV_R4 — 16-row-per-threadgroup GEMV variant.
    pub gemv_r4: bool,
    /// OJAS_NO_PREFETCH — disable expert/SSM prefetch threads.
    pub no_prefetch: bool,
    /// OJAS_EXPERT_STATS — CSV path for the (layer,expert) routing census.
    pub expert_stats: Option<String>,
    /// OJAS_PIN_GB — RAM budget for frequency-pinned hot experts, in BYTES
    /// (converted from GB; 0 = pinning off).
    pub pin_budget: u64,
    /// OJAS_SERIAL — force serial encoders (disable concurrent dispatch).
    pub serial: bool,
    /// OJAS_SSM_DBG — SSM layer debug prints.
    pub ssm_dbg: bool,
    /// OJAS_GATHER_THREADS — pread threads for expert-cache misses (≥1).
    pub gather_threads: usize,
    /// OJAS_EXPERT_COPY_THREADS — parallel cached-expert copies (1 = serial;
    /// experimental, at most 8). Independent of disk-read concurrency.
    pub expert_copy_threads: usize,
    /// Experimental direct Metal cache reads for Flash native-quant target experts.
    pub flash_direct_experts: bool,
    /// Pool direct cached experts into one Metal resource (experimental).
    pub flash_expert_pool: bool,
    /// OJAS_GLM_DBG — GLM forward debug prints.
    pub glm_dbg: bool,
    /// OJAS_NO_PREFILL — disable batched prefill (per-token fallback).
    pub no_prefill: bool,
    /// `-b`/`--batch-size` / OJAS_PREFILL_M — prefill chunk size in tokens.
    pub prefill_m: usize,
    /// OJAS_MTP_DRAFT — tokens the NextN head drafts per step by chaining on its
    /// own hidden. Default 1: deeper measured worse end to end (4.38 tok/s at 1,
    /// 2.74 at 2, 2.00 at 3) even though verify is cheap (M=3 costs 1.54x a single
    /// forward, M=2 1.47x). Acceptance decays down the chain — step 2 conditions on
    /// the draft block's own hidden, not the target's — and is all-or-nothing, so one
    /// bad chained draft discards the whole step. Longest-prefix acceptance with a
    /// confidence cutoff would fix both, but needs a recurrent-state snapshot per
    /// row, ~113 MB each on this model.
    pub mtp_draft: usize,
    /// OJAS_MTP_PREFIX — Flash per-row recurrent snapshots and longest-prefix
    /// acceptance. Experimental; allocates additional snapshot memory.
    pub mtp_prefix: bool,
    /// OJAS_FLASH_Q8_COOPERATIVE — opt-in native Q8_0 cooperative Metal
    /// projections for wide, 2–8-token Flash batches.
    pub flash_q8_cooperative: bool,
    /// OJAS_FLASH_HC_FUSED — opt in to the qwen4exp scalar HC three-pass path.
    /// Restricted at dispatch to the measured d=2560/hc=4/lr=320 Q8_0 layout.
    pub flash_hc_fused: bool,
    /// OJAS_FLASH_IQ4NL_WIDE — use eight cooperating lanes per routed-expert
    /// IQ4_NL down block. Restricted to qwen4exp scalar resident decode.
    pub flash_iq4nl_wide: bool,
    /// OJAS_GDN_AB_FUSED — fuse Qwen3.5/Flash's two narrow F32 alpha/beta
    /// projections with their scalar activations during one-token decode.
    /// OJAS_FLASH_GDN_AB_FUSED remains a compatibility alias.
    pub flash_gdn_ab_fused: bool,
    /// `--q4-head` / OJAS_Q4_HEAD — experimentally requantize an untied Q8_0
    /// output head to the tuned Q4 layout while leaving the rest unchanged.
    pub q4_head: bool,
    /// `--q4-ffn-down` / OJAS_Q4_FFN_DOWN — experimentally requantize dense
    /// Qwen3.5 Q8_0 FFN down projections to Q4.
    pub q4_ffn_down: bool,
    /// `--q4-ffn-down-last N` / OJAS_Q4_FFN_DOWN_LAST=N — convert only the last
    /// N dense layers. Ignored when all FFN-down projections are selected.
    pub q4_ffn_down_last: usize,
    /// OJAS_FLASH_IQ3S_TABLE — use llama.cpp-style threadgroup byte lookup for
    /// resident scalar IQ3_S expert gate/up projections.
    pub flash_iq3s_table: bool,
    /// `-ub`/`--ubatch-size` / OJAS_UBATCH — tokens per physical pass on a
    /// disk-streamed MoE model. M tokens route up to M*n_used distinct experts and
    /// the packed gather scratch must hold that union, so this sizes the scratch and
    /// caps a streamed prefill chunk. Raising it spends memory (n_used * ubatch
    /// experts of gate/up/down) to read each shared expert once for more tokens.
    /// Models held in RAM ignore it.
    ///
    /// Measured on a 93.7 GB, 512-expert top-10 MoE, warm prefill: 4 -> 6.8 tok/s,
    /// 8 -> 13.2, 16 -> 20.2. 16 costs ~400 MB of scratch there; a model with
    /// fatter experts pays proportionally more, and the load log prints the figure
    /// so it can be lowered.
    pub ubatch: usize,
    /// OJAS_SNAP — SSM snapshot ladder interval in tokens (≥256).
    pub snap_interval: usize,
    /// OJAS_PREFILL_DBG — prefill debug prints.
    pub prefill_dbg: bool,
    /// OJAS_NO_PREFIX_REUSE — disable cross-turn KV-prefix reuse (dense + SSM),
    /// forcing a full re-prefill from position 0 every request. Default off, so
    /// reuse is on; this is the kill switch and A/B control.
    pub no_prefix_reuse: bool,
    /// `--moe-dbuf` / OJAS_MOE_DBUF — double-buffered MoE expert gather (prefetch
    /// layer L+1's experts while computing layer L). Opt-in; the CLI flag wins over
    /// the env var.
    ///
    /// `moe_dbuf_gate` covers it against a real MoE model (qwen4exp, 512 experts,
    /// prec=4): the structural staging check passes and 24 tokens come out
    /// byte-identical between dbuf and serial. It defaults off because the measured
    /// win was 1.01x (serial 4.01 tok/s, dbuf 4.04) — and prefetching layer L+1 pays
    /// in proportion to how well the previous token predicts the next token's
    /// experts, which that model does poorly because its output is not yet correct.
    /// Re-measure on a model that generates coherently before changing the default.
    pub moe_dbuf: bool,
    /// Expert-memory strategy: Some(true) = keep every expert resident (llama-style,
    /// fastest), Some(false) = stream through a bounded cache (low memory), None =
    /// default (resident for models that fit, else streaming). `--resident` /
    /// `--stream` / OJAS_FLASH_RESIDENT set it.
    pub flash_resident: Option<bool>,
    /// OJAS_FLASH_RESIDENT_GROUP — layers whose experts are wired into one command
    /// buffer in resident mode, bounding per-command-buffer residency. At ~1.3 GB
    /// of experts per layer, group 8 wires ~10 GB, well under the device ceiling,
    /// and cuts the ~49 per-token command buffers to ~6. Full single-buffer
    /// residency (group >= n_layers) wires the whole 62 GB at once and runs a 96 GB
    /// machine out of memory, so the default is a safe group, not the whole stack.
    pub flash_resident_group: usize,
    /// OJAS_FLASH_RESIDENT_LAYERS — how many leading layers to hold fully resident
    /// (all experts wired) when OJAS_FLASH_RESIDENT is set. 0 = derive from the
    /// budget that fits; a very large value = all. Partial residency wires these
    /// layers gather-free and streams the rest, so a machine that cannot hold all
    /// 62 GB still removes the host gather for the wired fraction.
    pub flash_resident_layers: usize,
    /// OJAS_MOE_SKEW — emit routing-skew telemetry (oracle-hit-at-slots,
    /// working-set, entropy) to help size the expert cache.
    pub moe_skew: bool,
    /// OJAS_KMAP_DIV — use the grouped value->key head mapping in gated DeltaNet
    /// (h / (H_v/H_k)) instead of the tiled one (h % H_k). GGUF-converted qwen35 is
    /// tiled, which is the default; see the note in `ssm.rs`.
    pub moe_kmap_div: bool,
    /// OJAS_NO_PLE — skip the qwen4exp n-gram (per-layer) embedding block.
    pub no_ple: bool,
    /// OJAS_HC_TRACE — log per-stream hyper-connection residual norms per layer.
    pub hc_trace: bool,
    /// Optional per-layer Qwen4 residual capture directory for reference diagnostics.
    pub reference_trace: Option<String>,
    /// `-md`/`--spec-draft-model` / OJAS_MTP — path to the MTP/NextN draft-head
    /// GGUF. None = auto-discover a sibling `MTP/` directory.
    pub mtp: Option<String>,
    /// `--mmproj` / OJAS_MMPROJ — path to the vision-projector GGUF that ships
    /// alongside a multimodal decoder. None = auto-discover a sibling
    /// `*mmproj*.gguf`.
    ///
    /// This is an engine knob rather than a runtime one: the model loader is the
    /// only thing that can act on it and `RunOpts` is not visible there, so one
    /// flag serves every command that loads a multimodal model.
    pub mmproj: Option<String>,
}

/// Read an engine env var. The single choke point for `OJAS_*` reads.
pub fn var(key: &str) -> Result<String, std::env::VarError> {
    std::env::var(key)
}

/// Presence test for a boolean knob: present means true, except that an empty value
/// and the conventional negatives (`0`, `false`, `no`, `off`) are false. Under bare
/// presence alone, `OJAS_NO_SPEC=` and `OJAS_NO_SPEC=0` both disabled speculation,
/// so `OJAS_NO_SPEC=$MAYBE` with `MAYBE` unset silently selected the opposite arm.
pub fn flag(key: &str) -> bool {
    match var(key) {
        Ok(v) => {
            let v = v.trim();
            !(v.is_empty()
                || v.eq_ignore_ascii_case("0")
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("no")
                || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => false,
    }
}

impl EngineConfig {
    /// The config the engine should use: whatever was installed, else the
    /// environment.
    ///
    /// Engine code calls this rather than `from_env`. A host that is not a shell —
    /// a sandboxed utility process, say — has no meaningful environment to set, and
    /// env vars travel poorly across a process boundary, so it installs a config.
    pub fn current() -> Self {
        INSTALLED.get().cloned().unwrap_or_else(Self::from_env)
    }

    /// Installs the config the engine will use. The first call wins; a later call
    /// hands the config back as an error, because knobs are captured once at load
    /// and a second install would not apply.
    pub fn install(cfg: Self) -> Result<(), Self> {
        INSTALLED.set(cfg)
    }

    pub fn from_env() -> Self {
        EngineConfig {
            top_k: var("OJAS_TOPK").ok().and_then(|s| s.parse().ok()),
            no_mmap: flag("OJAS_NO_MMAP"),
            prewarm: flag("OJAS_PREWARM"),
            no_skel_lock: flag("OJAS_NO_SKEL_LOCK"),
            mla_naive: flag("OJAS_MLA_NAIVE"),
            kv_gb: var("OJAS_KV_GB").ok().and_then(|v| v.parse().ok()),
            ctx: var("OJAS_CTX").ok().and_then(|v| v.parse().ok()),
            expert_cache_gb: var("OJAS_EXPERT_CACHE_GB")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(32.0),
            sparse: var("OJAS_SPARSE")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&b| b > 0),
            adj_draft: flag("OJAS_ADJ_DRAFT"),
            no_spec: flag("OJAS_NO_SPEC"),
            gemv_r4: flag("OJAS_GEMV_R4"),
            no_prefetch: flag("OJAS_NO_PREFETCH"),
            expert_stats: var("OJAS_EXPERT_STATS").ok(),
            pin_budget: var("OJAS_PIN_GB")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .map(|gb| (gb * 1e9) as u64)
                .unwrap_or(0),
            serial: flag("OJAS_SERIAL"),
            ssm_dbg: flag("OJAS_SSM_DBG"),
            gather_threads: var("OJAS_GATHER_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8)
                .max(1),
            expert_copy_threads: var("OJAS_EXPERT_COPY_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1)
                .clamp(1, 8),
            flash_direct_experts: flag("OJAS_FLASH_DIRECT_EXPERTS"),
            flash_expert_pool: flag("OJAS_FLASH_EXPERT_POOL"),
            glm_dbg: flag("OJAS_GLM_DBG"),
            no_prefill: flag("OJAS_NO_PREFILL"),
            prefill_m: var("OJAS_PREFILL_M")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            mtp_draft: var("OJAS_MTP_DRAFT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1)
                .max(1),
            mtp_prefix: flag("OJAS_MTP_PREFIX"),
            flash_q8_cooperative: flag("OJAS_FLASH_Q8_COOPERATIVE"),
            flash_hc_fused: flag("OJAS_FLASH_HC_FUSED"),
            flash_iq4nl_wide: flag("OJAS_FLASH_IQ4NL_WIDE"),
            flash_gdn_ab_fused: flag("OJAS_GDN_AB_FUSED") || flag("OJAS_FLASH_GDN_AB_FUSED"),
            q4_head: flag("OJAS_Q4_HEAD"),
            q4_ffn_down: flag("OJAS_Q4_FFN_DOWN"),
            q4_ffn_down_last: var("OJAS_Q4_FFN_DOWN_LAST")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            flash_iq3s_table: flag("OJAS_FLASH_IQ3S_TABLE"),
            ubatch: var("OJAS_UBATCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(16)
                .max(1),
            snap_interval: var("OJAS_SNAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2048)
                .max(256),
            prefill_dbg: flag("OJAS_PREFILL_DBG"),
            no_prefix_reuse: flag("OJAS_NO_PREFIX_REUSE"),
            moe_dbuf: flag("OJAS_MOE_DBUF"),
            flash_resident: if flag("OJAS_FLASH_RESIDENT") {
                Some(true)
            } else if flag("OJAS_FLASH_STREAM") {
                Some(false)
            } else {
                None
            },
            flash_resident_group: std::env::var("OJAS_FLASH_RESIDENT_GROUP")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&g: &usize| g > 0)
                .unwrap_or(8),
            flash_resident_layers: std::env::var("OJAS_FLASH_RESIDENT_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            moe_skew: flag("OJAS_MOE_SKEW"),
            moe_kmap_div: flag("OJAS_KMAP_DIV"),
            no_ple: flag("OJAS_NO_PLE"),
            hc_trace: flag("OJAS_HC_TRACE"),
            reference_trace: var("OJAS_REFERENCE_TRACE").ok(),
            mtp: var("OJAS_MTP").ok(),
            mmproj: var("OJAS_MMPROJ").ok(),
        }
    }
}

/// Root of this engine's on-disk cache: `$OJAS_CACHE_DIR`, else
/// `$XDG_CACHE_HOME/ojas`, else `~/.cache/ojas`.
///
/// Everything the engine writes between runs — autotune plans, quantized weight
/// caches, sessions — lives under here. Nothing outside this function spells the
/// directory.
pub fn cache_dir() -> Option<std::path::PathBuf> {
    let dir = if let Ok(d) = std::env::var("OJAS_CACHE_DIR") {
        std::path::PathBuf::from(d)
    } else if let Ok(x) = std::env::var("XDG_CACHE_HOME") {
        std::path::Path::new(&x).join("ojas")
    } else if let Ok(local) = std::env::var("LOCALAPPDATA") {
        // Windows: per-user, non-roaming cache root.
        std::path::Path::new(&local).join("ojas").join("cache")
    } else {
        std::path::Path::new(&std::env::var("HOME").ok()?)
            .join(".cache")
            .join("ojas")
    };
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Path under [`cache_dir`].
pub fn cache_file(rel: &str) -> Option<std::path::PathBuf> {
    Some(cache_dir()?.join(rel))
}
