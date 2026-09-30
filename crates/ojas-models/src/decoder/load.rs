#![allow(clippy::too_many_arguments)]
use super::*;
use ojas_formats::gguf::{Gguf, Meta};
use ojas_metal::MetalGpu;
use anyhow::Result;
use metal::MTLResourceOptions;
use std::collections::HashMap;
use std::ffi::c_void;
use ojas_core::config::EngineConfig;

/// Minimum RAM left outside the GPU residency request. Three host crashes established
/// the 24 GiB floor; `OJAS_RAM_RESERVE_GB` may only raise it.
///
/// The request is planned from all allocated state/scratch plus page-rounded expert
/// views (PLE is mapped CPU-only, so file size is not the GPU request), and that exact
/// sum is verified again before residency is requested.
fn ram_reserve_bytes() -> u64 {
    const FLOOR: u64 = 24 << 30;
    std::env::var("OJAS_RAM_RESERVE_GB").ok().and_then(|s| s.parse::<u64>().ok())
        .map(|g| g.checked_mul(1 << 30).unwrap_or(u64::MAX).max(FLOOR)).unwrap_or(FLOOR)
}

#[derive(Debug)]
struct ResidentMemoryPlan {
    existing_bytes: u64,
    expert_bytes: u64,
    requested_bytes: u64,
    reserve_bytes: u64,
    ram_bytes: u64,
    working_set_bytes: u64,
}

impl ResidentMemoryPlan {
    fn fits(&self) -> bool {
        self.reserve_bytes >= 24 << 30
            && self.requested_bytes > 0
            && self.requested_bytes <= self.working_set_bytes
            && self.requested_bytes.checked_add(self.reserve_bytes).is_some_and(|b| b <= self.ram_bytes)
    }
}

fn expert_layer(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

// XNU's AVAILABLE_NON_COMPRESSED_MEMORY is active + inactive + free. Active
// includes reclaimable file-backed pages (notably a just-released model); leaving
// it out makes a warm cache look like unavailable RAM. The separate fixed 24 GiB
// physical reserve covers live application/kernel/compressor memory. Failure to
// query still means streaming.
#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn available_ram_bytes() -> Option<u64> {
    unsafe extern "C" {
        fn mach_port_deallocate(task: libc::mach_port_t, name: libc::mach_port_t) -> libc::kern_return_t;
    }
    unsafe {
        let host = libc::mach_host_self();
        let mut stats: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        let rc = libc::host_statistics64(host, libc::HOST_VM_INFO64,
            &mut stats as *mut _ as *mut libc::integer_t, &mut count);
        mach_port_deallocate(libc::mach_task_self(), host);
        let page = libc::sysconf(libc::_SC_PAGESIZE);
        if rc != libc::KERN_SUCCESS || page <= 0 { return None; }
        (stats.free_count as u64 + stats.inactive_count as u64 + stats.active_count as u64)
            .checked_mul(page as u64)
    }
}

/// The Mach call above is macOS-only. `None` means "could not query", which every caller
/// reads as stream — the safe default.
///
/// The Metal decoder cannot run off macOS; this exists so the crate type-checks on the
/// machine doing the CUDA port.
#[cfg(not(target_os = "macos"))]
fn available_ram_bytes() -> Option<u64> {
    None
}

fn live_residency_fits(requested: u64, available: Option<u64>) -> bool {
    requested.checked_add(8 << 30).zip(available).is_some_and(|(need, have)| need <= have)
}

fn checked_buffer_bytes(buffers: &[&metal::Buffer]) -> Result<u64> {
    buffers.iter().try_fold(0u64, |sum, b| {
        sum.checked_add(b.length()).ok_or_else(|| anyhow::anyhow!("GPU allocation size overflow"))
    })
}

impl DecoderGpu<'_> {
    /// Called at load, with the cache empty and before any resident expert view
    /// exists. All other GPU allocations already exist and are counted exactly.
    fn resident_memory_plan(&self) -> Result<ResidentMemoryPlan> {
        anyhow::ensure!(self.strm.expert_cache.borrow().bytes() == 0 && !self.cfg.flash_expert_pool,
            "resident planning requires an empty, unpooled expert cache");
        let existing_bytes = checked_buffer_bytes(&self.all_gpu_buffers())?;
        let mapped = self.wt.mapped.as_ref().ok_or_else(|| anyhow::anyhow!("missing expert mapping"))?;
        let mut expert_bytes = 0u64;
        for (name, &(part, offset, bytes, _)) in &self.strm.stream_meta {
            if expert_layer(name).is_some_and(|l| l < self.arch.n_layers) {
                expert_bytes = expert_bytes.checked_add(mapped.buffer_length(part, offset, bytes)?)
                    .ok_or_else(|| anyhow::anyhow!("resident expert size overflow"))?;
            }
        }
        let requested_bytes = existing_bytes.checked_add(expert_bytes)
            .ok_or_else(|| anyhow::anyhow!("resident request overflow"))?;
        Ok(ResidentMemoryPlan { existing_bytes, expert_bytes, requested_bytes,
            reserve_bytes: ram_reserve_bytes(), ram_bytes: physical_ram_bytes(),
            working_set_bytes: self.gpu.device.recommended_max_working_set_size() })
    }
}

#[cfg(target_os = "macos")]
fn physical_ram_bytes() -> u64 {
    let mut v: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = std::ffi::CString::new("hw.memsize").unwrap();
    let ok = unsafe {
        libc::sysctlbyname(name.as_ptr(), &mut v as *mut u64 as *mut c_void, &mut len,
                           std::ptr::null_mut(), 0)
    };
    if ok == 0 && v > 0 { v } else { 64u64 << 30 }
}

/// `sysctlbyname("hw.memsize")` has no portable twin; `_SC_PHYS_PAGES` is the POSIX one. Same
/// 64 GiB fallback as the macOS path uses when the query fails.
#[cfg(not(target_os = "macos"))]
fn physical_ram_bytes() -> u64 {
    let (pages, page) = unsafe {
        (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE))
    };
    if pages > 0 && page > 0 { pages as u64 * page as u64 } else { 64u64 << 30 }
}
 // re-export

impl IndexerConfig {
    /// Read `{arch}.attention.indexer.{head_count,key_length,top_k}`. Absent keys
    /// default to zero, which denotes "no indexer configured".
    fn from_gguf(g: &Gguf, arch: &str) -> Self {
        let u = |k: &str| g.meta_u32(&format!("{arch}.attention.indexer.{k}")).unwrap_or(0);
        IndexerConfig { n_head: u("head_count"), head_size: u("key_length"), top_k: u("top_k") }
    }
}

impl VisionConfig {
    /// Read the `clip.*` KV an mmproj contributed and check that the tower is one the
    /// engine can build, before anything is allocated. The checks mirror
    /// `ojas_cpu::cpu_vit::CpuVit::load` one for one, so the CPU oracle's geometry and
    /// the GPU's cannot diverge silently — a divergence a numeric gate cannot localize.
    ///
    /// `channels` and the patch-embed shape come from `v.patch_embd.weight`'s own dims
    /// rather than from the KV, which carries no channel count; `mmproj::validate` has
    /// already accepted either the 4-D `[kw,kh,ic,oc]` or the flattened `[kw*kh*ic, oc]`
    /// spelling.
    fn from_gguf(g: &Gguf) -> Result<VisionConfig> {
        let mu = |k: &str| g.meta_u32(&format!("clip.vision.{k}")).unwrap_or(0);
        let (d, layers, n_head) = (mu("embedding_length"), mu("block_count"), mu("attention.head_count"));
        let (ffn, patch, image_size) = (mu("feed_forward_length"), mu("patch_size"), mu("image_size"));
        let proj_dim = mu("projection_dim");
        let merge = mu("spatial_merge_size").max(1);
        let eps = g.meta_f32("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6);
        anyhow::ensure!(d > 0 && layers > 0 && n_head > 0 && patch > 0 && image_size > 0 && ffn > 0,
            "mmproj is missing clip.vision.* metadata (embd={d} blocks={layers} heads={n_head} ffn={ffn} patch={patch} image={image_size})");
        anyhow::ensure!(d % n_head == 0, "vision embedding_length {d} not divisible by head_count {n_head}");
        let hd = d / n_head;
        // The four M-RoPE sections are hd/4 each (qwen3vl.cpp:14).
        anyhow::ensure!(hd % 4 == 0, "vision M-RoPE needs head_dim {hd} divisible by 4");
        anyhow::ensure!(merge == 2, "only spatial_merge_size=2 is implemented (got {merge})");
        anyhow::ensure!(image_size % patch == 0,
            "image_size {image_size} is not a multiple of patch_size {patch}");
        // Fail loudly rather than silently dropping a feature: surya-2 ships twelve
        // `false`s and no v.deepstack.* tensors, but Qwen3.5-VL proper uses them.
        anyhow::ensure!(!g.int_arr("clip.vision.is_deepstack_layers").is_some_and(|v| v.iter().any(|&b| b != 0)),
            "this mmproj declares deepstack layers; the vision tower does not implement deepstack");
        let pe = g.tensors.get("v.patch_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("mmproj missing v.patch_embd.weight"))?;
        let (channels, oc) = match pe.dims.len() {
            4 => (pe.dims[2] as u32, pe.dims[3] as u32),
            2 => ((pe.dims[0] / (patch * patch) as u64) as u32, pe.dims[1] as u32),
            n => anyhow::bail!("v.patch_embd.weight has rank {n}, expected 4 ([kw,kh,ic,oc]) or 2"),
        };
        anyhow::ensure!(oc == d && channels > 0,
            "v.patch_embd.weight {:?} disagrees with embedding_length {d}", pe.dims);
        anyhow::ensure!(g.tensors.contains_key("v.patch_embd.weight.1"),
            "mmproj missing v.patch_embd.weight.1 — the temporal-merge fold needs both convs");
        // mm.0's output width is the projector's hidden size: read it rather than
        // compute it. The ensure below checks it against d*merge^2.
        let mm_hidden = g.tensors.get("mm.0.weight")
            .ok_or_else(|| anyhow::anyhow!("mmproj missing mm.0.weight"))?
            .dims.get(1).copied().unwrap_or(0) as u32;
        anyhow::ensure!(mm_hidden == d * merge * merge,
            "mm.0.weight output width {mm_hidden} != embedding_length*merge^2 = {}", d * merge * merge);
        let pos_side = image_size / patch;
        let pos = g.tensors.get("v.position_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("mmproj missing v.position_embd.weight"))?;
        anyhow::ensure!(pos.dims.iter().product::<u64>() == (pos_side * pos_side * d) as u64,
            "v.position_embd.weight {:?} is not [{d}, {pos_side}x{pos_side}]", pos.dims);
        // Arena capacity in patches. The default is the native position grid: exactly a
        // 768x768 page, the size the learned embedding was trained at and the one that
        // takes the resize early-return. Bigger images encode through on-demand
        // temporaries (`encode_image`).
        let max_patches = ojas_core::config::var("OJAS_VIT_MAX_PATCHES").ok()
            .and_then(|v| v.parse::<u32>().ok()).filter(|&n| n >= 4)
            .unwrap_or(pos_side * pos_side);
        Ok(VisionConfig {
            d, layers, n_head, hd, ffn, patch, channels, pos_side, merge, proj_dim, mm_hidden,
            eps, rope_base: 10000.0, max_patches,
        })
    }
}

impl TextEncoderConfig {
    /// The `modern-bert.*` keys (llama.cpp's names for the architecture) and the
    /// `laya.head.*` keys, both as `scripts/laya_convert.py` writes them. Shapes are checked against the tensors
    /// here, so a truncated or foreign file fails with a key name rather than inside
    /// graph construction.
    fn from_gguf(g: &Gguf) -> Result<TextEncoderConfig> {
        let arch = g.arch();
        let mu = |k: &str| g.meta_u32(&format!("{arch}.{k}"));
        let (d, layers, n_head, ffn) = (mu("embedding_length").unwrap_or(0), mu("block_count").unwrap_or(0),
            mu("attention.head_count").unwrap_or(0), mu("feed_forward_length").unwrap_or(0));
        anyhow::ensure!(d > 0 && layers > 0 && n_head > 0 && ffn > 0 && d % n_head == 0,
            "{arch}: incomplete encoder metadata (embd={d} blocks={layers} heads={n_head} ffn={ffn})");
        let hd = d / n_head;
        // The bidirectional attention kernels keep a head in 16 accumulators per lane
        // across a 32-lane simdgroup: hd % 32 == 0 and hd <= 512.
        anyhow::ensure!(hd % 32 == 0 && hd <= 512,
            "{arch}: head_dim {hd} is not supported by the bidirectional attention kernels");
        let act = match g.meta.get(&format!("{arch}.hidden_activation")) {
            Some(Meta::Str(s)) => s.clone(),
            _ => "gelu".to_string(),
        };
        anyhow::ensure!(act == "gelu", "{arch}: hidden_activation {act:?} is not implemented (only \"gelu\", the erf form)");
        let eps = g.meta_f32(&format!("{arch}.attention.layer_norm_epsilon")).unwrap_or(1e-5);
        let rope_base = g.meta_f32(&format!("{arch}.rope.freq_base")).unwrap_or(160000.0);
        let rope_base_local = g.meta_f32(&format!("{arch}.rope.freq_base_swa")).unwrap_or(rope_base);
        let sliding = mu("attention.sliding_window").unwrap_or(0);
        let swa_pattern = if sliding > 0 { mu("attention.sliding_window_pattern").unwrap_or(3) } else { 0 };
        let max_positions = mu("context_length").unwrap_or(8192);
        for (name, want) in [("blk.0.attn_qkv.weight", [d as u64, 3 * d as u64]),
                             ("blk.0.ffn_up.weight", [d as u64, 2 * ffn as u64]),
                             ("blk.0.ffn_down.weight", [ffn as u64, d as u64])] {
            let t = g.tensors.get(name).ok_or_else(|| anyhow::anyhow!("{arch}: missing {name}"))?;
            anyhow::ensure!(t.dims == want, "{arch}: {name} is {:?}, expected {want:?}", t.dims);
        }
        let laya = match g.meta_u32("laya.head.block_count") {
            None => None,
            Some(blocks) => {
                anyhow::ensure!(blocks >= 1, "laya head: laya.head.block_count is 0");
                let head = LayaHeadConfig {
                    blocks,
                    n_head: g.meta_u32("laya.head.head_count").unwrap_or(n_head),
                    ffn: g.meta_u32("laya.head.feed_forward_length").unwrap_or(4 * d),
                    eps: g.meta_f32("laya.head.layer_norm_epsilon").unwrap_or(1e-5),
                };
                anyhow::ensure!(head.n_head > 0 && d % head.n_head == 0 && d / head.n_head == hd,
                    "laya head: {} heads over d={d} gives a head_dim other than the encoder's {hd}", head.n_head);
                for i in 0..blocks {
                    let name = format!("laya.blk.{i}.ffn_up.weight");
                    let t = g.tensors.get(&name).ok_or_else(|| anyhow::anyhow!("laya head: missing {name}"))?;
                    anyhow::ensure!(t.dims == [d as u64, head.ffn as u64], "laya head: {name} is {:?}", t.dims);
                }
                Some(head)
            }
        };
        Ok(TextEncoderConfig {
            d, layers, n_head, hd, ffn, eps, rope_base, rope_base_local,
            window: sliding / 2, swa_pattern, max_positions, laya,
        })
    }

    /// Every tensor the GPU reads, in upload order.
    fn gpu_tensors(&self, g: &Gguf) -> Vec<String> {
        let mut names: Vec<String> = vec!["token_embd.weight".into(), "token_embd_norm.weight".into(),
                                          "output_norm.weight".into()];
        for i in 0..self.layers {
            if g.tensors.contains_key(&format!("blk.{i}.attn_norm.weight")) {
                names.push(format!("blk.{i}.attn_norm.weight"));
            }
            for s in ["attn_qkv.weight", "attn_output.weight", "ffn_norm.weight", "ffn_up.weight", "ffn_down.weight"] {
                names.push(format!("blk.{i}.{s}"));
            }
        }
        if let Some(h) = &self.laya {
            for i in 0..h.blocks {
                for s in ["attn_norm", "attn_qkv", "attn_output", "ffn_norm", "ffn_up", "ffn_down"] {
                    for kind in ["weight", "bias"] { names.push(format!("laya.blk.{i}.{s}.{kind}")); }
                }
            }
            for s in ["laya.type_emb.weight", "laya.scorer_norm.weight", "laya.scorer_norm.bias",
                      "laya.scorer_fc.weight", "laya.scorer_fc.bias"] {
                names.push(s.into());
            }
        }
        names
    }
}

/// The precision argument that asks [`DecoderGpu::load`] to choose the tier for the
/// file (see [`DecoderGpu::auto_precision`]).
pub const PRECISION_AUTO: u8 = u8::MAX;

impl<'a> DecoderGpu<'a> {
    /// The tier [`PRECISION_AUTO`] resolves to for this file, and why.
    ///
    /// Precision 4 streams mixture-of-experts weights from disk, and to do so keeps
    /// every other quantized tensor in the file's own format behind the native
    /// kernels. For a dense model that buys nothing, since all of it is resident
    /// anyway. M2 Max, cold prefill, precision 3 against 4:
    ///
    /// | file                       | decode tok/s  | first token      |
    /// |----------------------------|---------------|------------------|
    /// | Qwen3.5 4B Q4_K_M, 591 tok | 79.9 vs 84.8  | 0.67 s vs 0.78 s |
    /// | same, 2340 tok             | 74.5 vs 78.8  | 2.41 s vs 2.81 s |
    /// | Qwen3.5 4B F16, 591 tok    | 55.4 vs 33.1  | 0.72 s vs 0.69 s |
    /// | Qwen3 0.6B F16             | 202 vs 125    | 0.02 s vs 0.12 s |
    /// | Qwen2.5 0.5B Q8_0          | 249 vs 201    | 0.02 s vs 0.07 s |
    ///
    /// On a K-quant file the two are close: precision 4 reads the file's own Q4_K and
    /// Q6_K bytes, which are fewer than precision 3's Q8 copies of the Q6_K tensors, so
    /// it decodes ~6% faster and loads in 0.1 s against 1.1; precision 3's tuned Q4L
    /// tiles prefill ~15% faster. Prompts outweigh replies in the workloads this
    /// engine serves (page context, OCR), and on F16 and Q8_0 files precision 3 wins
    /// outright, so dense decoders take precision 3.
    ///
    /// Precision 3 stores F16 and Q8_0 matrices as Q8 and Q4_K as its Q4L relayout;
    /// `--precision 4` (or 0 for f16) keeps a file exact. It cannot overcommit
    /// memory: when its requantized weights would exceed the device budget, the loader
    /// keeps the file's format for every tensor instead (the fit check below). A
    /// vision tower keeps its f16 weights at every precision (`vision_weights_f16`).
    /// Precision 4 stays for models with experts, which is what streaming is for.
    pub fn auto_precision(g: &Gguf) -> Result<(u8, &'static str)> {
        let arch = g.arch();
        let experts = g.tensors.keys().any(|n| n.contains("_exps."))
            || g.meta_u32(&format!("{arch}.expert_count")).unwrap_or(0) > 0;
        if experts {
            return Ok((4, "mixture-of-experts weights stream from disk"));
        }
        Ok((3, "dense decoder: tuned kernels, weights requantized where they fit"))
    }

    /// `kv_gb` / `expert_cache_gb` are per-load overrides for the KV↔experts
    /// memory split (the app's slider). `None` falls back to the `OJAS_KV_GB` /
    /// `OJAS_EXPERT_CACHE_GB` env knobs read via `EngineConfig`. Both are GiB.
    pub fn load(
        gpu: &'a MetalGpu,
        g: &mut Gguf,
        max_seq: usize,
        prec: u8,
        kv_gb: Option<u64>,
        expert_cache_gb: Option<f64>,
    ) -> Result<Self> {
        let ecfg = EngineConfig::current(); // all OJAS_* knobs, read once
        let prec = if prec == PRECISION_AUTO {
            let (p, why) = Self::auto_precision(g)?;
            tracing::info!(target: "arch", "precision {p} ({why})");
            p
        } else { prec };
        let mut max_seq = max_seq;
        // gpt-oss: force Q8 requant — its huge Q5_1 experts can't dequant to f16 (~38GB),
        // and the moe_*_q8_oai gather kernels consume Q8 (w8/scale8). Biases/sinks are f32→w32.
        let is_gptoss = g.arch() == "gpt-oss";
        let quant = prec == 1 || prec == 3 || is_gptoss;
        let q4mode = prec == 2 && !is_gptoss;
        let q4k = prec == 3 || prec == 4;   // native Q4_K (faithful)
        // prec 5: native ternary Q2_0 (g128). Never requantize ternary — every non-zero
        // weight sits at +-amax, and quantize_row_q4 clamps +amax to nibble 15 (0.875x),
        // biasing every positive weight low.
        let q20 = prec == 5;
        let stream = prec == 4;   // direct-mmap streaming from the GGUF (no requant cache)
        // OJAS_NATIVE: keep weights in the GGUF's own quantization instead of
        // requantizing to Q8/Q4L.
        //   "1"/"all" — every format that has a native kernel
        //   "auto"    — only formats under 4 bits/weight
        //   "0"       — never
        //   unset     — fit-based (below)
        //
        // Measured on Qwen3.8-27B IQ2_XXS: requant 28.58 GB at 73.8 ms/token, native
        // 9.77 GB at 107.7 — requant is faster where there is room for it, and 28.58 GB
        // does not run on a 16 GB machine at all.
        //
        // Requant size estimate: prec 1 lands on Q8 (1 byte/weight + a f32 row scale),
        // prec 2 on Q4L (~5 bits/weight), compared against 80% of the recommended working
        // set to leave headroom for KV cache and activations.
        let native_mode = std::env::var("OJAS_NATIVE").unwrap_or_default();
        let native_forced_off = native_mode == "0";
        // Native is the default, with `native_wants` deciding per tensor: keep the file's
        // format when it is smaller than the requant target, requantize when it is not.
        // Requantizing whenever the model fits picked the slower path for low-bit models —
        // on the 27B IQ2_XXS, requant 75.6 ms/token against native 59.8, and 19 GB more
        // memory. The fit check below only escalates: it forces everything native when
        // requant would not fit at all.
        let mut native_quant = !native_forced_off;
        let mut native_all_fit = false;
        let native_all_env = native_mode == "1" || native_mode == "all";
        if native_mode.is_empty() {
            let quant_elems: u64 = g.tensors.values()
                .filter(|i| i.dims.len() == 2
                    && ojas_metal::kernels::nat::nat_wpb(i.ggml_type).is_some())
                .map(|i| i.dims.iter().product::<u64>())
                .sum();
            let est = if prec == 2 { quant_elems * 5 / 8 } else { quant_elems };
            // OJAS_VRAM_BUDGET_GB overrides the device budget, so the fit branch can be
            // exercised on a machine with room to spare, where it would never fire.
            let budget = match std::env::var("OJAS_VRAM_BUDGET_GB").ok().and_then(|v| v.parse::<f64>().ok()) {
                Some(gb) => (gb * 1e9) as u64,
                None => gpu.device.recommended_max_working_set_size() * 8 / 10,
            };
            if est > budget {
                // Over budget: take every byte available, including formats the
                // bpw rule would otherwise skip.
                native_all_fit = true;
                tracing::warn!(target: "native", "requantized weights would need {:.1} GB against a {:.1} GB \
                           budget — keeping the file's own format (OJAS_NATIVE=0 to override)",
                    est as f64 / 1e9, budget as f64 / 1e9);
                native_quant = true;
            }
        }
        // "auto" keeps the file's own format only under 4 bits/weight, where requantizing
        // to Q8 more than doubles the resident size. Derived from the shared format table
        // rather than a hand-listed set of type ids, which is easy to get wrong (IQ3_S is
        // the same 3.4 bits as IQ3_XXS).
        //
        // Q4_K (12) and Q6_K (14) have hand-tuned kernels — gemv_q4k, gemv_q6k and the Q4L
        // relayout — that the generic native matvec loses to: on the 27B Q6_K, generic
        // native 91.5 ms/token against 46.7 for the tuned path. The generic path is for
        // formats with no tuned kernel.
        let has_tuned_kernel = |ty: u32| matches!(ty, 12 | 14);
        // Native is only worth it when the file's own format is smaller than what
        // requantizing would produce. Requant targets Q8 (8.5 bits/weight with the row
        // scale) at prec=1 and Q4L (5.0) at prec=2, so at prec=2 a Q6_K or Q5_K tensor
        // requantized down reads fewer bytes and runs on the best-tuned kernel here.
        //
        // Measured on the 27B Q6_K file (43% Q6_K, 27% Q8_0, 18% Q5_K — all high bpw):
        // forced native 91.5 ms/token, requant-to-Q8 74.7, prec=2 tuned path 46.6.
        let target_bpw = if prec == 2 { 5.0 } else { 8.5 };
        let src_bpw = |ty: u32| ojas_core::quant_src::format_of(ty)
            .map(|f| f.block_bytes as f32 * 8.0 / f.weights as f32)
            .unwrap_or(f32::MAX);
        let native_wants = |ty: u32| -> bool {
            if has_tuned_kernel(ty) { return false; }
            if native_all_env || native_all_fit { return true; }
            if src_bpw(ty) >= target_bpw { return false; }
            ojas_core::quant_src::format_of(ty)
                .map(|f| (f.block_bytes as f32 * 8.0 / f.weights as f32) < 4.0)
                .unwrap_or(false)
        };
        // Tensors a specialized kernel claims. Those kernels read a specific buffer map
        // (w4/w8/moe_*) and index it directly, so a tensor that went native would leave
        // the lookup empty and panic. The list lives here once, instead of a guard at
        // every call site.
        //   token_embd / nextn.embed_tokens : row-gather `embed_*` kernels
        //   *_exps / *_shexp                : the MoE expert and shared-expert path
        //   ssm_conv1d                      : elementwise, never a gemv weight
        let is_qwen4exp = g.arch() == "qwen4exp";
        let native_eligible = |name: &str| -> bool {
            !name.contains("ssm_conv1d")
                && !name.contains("token_embd")
                && !name.contains("embed_tokens")
                && !name.contains("_exps.")
                && (!name.contains("_shexp") || is_qwen4exp)
        };

        let arch = g.arch();
        // ---- MTP/NextN sidecar -------------------------------------------------
        // Some publishers ship the draft head as its own GGUF (`MTP/mtp-<model>-Q4_0.gguf`;
        // the reference loads it as a separate draft context via --mtp). No second context
        // is needed here: the draft block's tensor names already match what
        // `mtp_draft_encode` looks up, so attaching the sidecar as an extra GGUF part
        // leaves the name list, weight maps and MtpConfig unchanged.
        //
        // Only the draft block is taken — the sidecar also carries its own
        // token_embd/output/output_norm. `attach` refuses to overwrite names the main model
        // already has; this filter states that explicitly rather than relying on it.
        //
        // Publishers disagree about where the draft block lives. Inline: block_count=65,
        // nextn_predict_layers=1, blk.64 present. Stripped: block_count=64,
        // nextn_predict_layers=0, and the block ships in a sidecar. So the guard is "the
        // tensor is missing and a sidecar has it", not "does the model claim a NextN layer"
        // — for the sidecar layout it claims zero.
        let nx = g.meta_u32(&format!("{arch}.nextn_predict_layers")).unwrap_or(0) as usize;
        let nl = (g.meta_u32(&format!("{arch}.block_count")).unwrap_or(0) as usize)
            .checked_sub(nx).ok_or_else(|| anyhow::anyhow!("NextN layer count exceeds block count"))?;
        let mut mtp_sidecar = false;
        if !g.tensors.contains_key(&format!("blk.{nl}.nextn.eh_proj.weight")) {
            let sidecar = ojas_formats::mtp::discover(std::path::Path::new(&g.path), ecfg.mtp.as_deref())?;
            if let Some(sc) = sidecar {
                let path = sc.to_str().ok_or_else(|| anyhow::anyhow!("non-UTF8 MTP path"))?;
                let draft = Gguf::open(path)?;
                ojas_formats::mtp::validate(g, &draft, nl)?;
                let pfx = format!("blk.{nl}.");
                let k = g.attach(path, |n| n.starts_with(&pfx))?;
                anyhow::ensure!(k > 0, "MTP sidecar had no draft tensors");
                mtp_sidecar = true;
                tracing::info!(target: "mtp", "attached {k} validated draft-block tensors");
            }
        }

        // Vision projector sidecar. Unlike MTP, the mmproj's KV *is* its configuration
        // (~20 clip.* keys describing the ViT), so this needs attach_with_meta — plain
        // attach merges tensors only and drops the incoming metadata. "Main model wins"
        // holds for KV as it does for tensor names.
        if let Some(sc) = ojas_formats::mmproj::discover(std::path::Path::new(&g.path), ecfg.mmproj.as_deref())? {
            let path = sc.to_str().ok_or_else(|| anyhow::anyhow!("non-UTF8 mmproj path"))?;
            let side = Gguf::open(path)?;
            ojas_formats::mmproj::validate(g, &side)?;
            let k = g.attach_with_meta(path, ojas_formats::mmproj::keep_tensor, ojas_formats::mmproj::keep_kv)?;
            anyhow::ensure!(k > 0, "mmproj had no v.*/mm.* tensors");
            tracing::info!(target: "mmproj", "attached {k} validated vision tensors");
        }

        let mut mla_cfg: Option<MlaConfig> = None;
        // MLA archs (deepseek2 / glm-dsa — DeepSeek-V2/V3, GLM-5.2): parse + report
        // the MLA/MoE(/DSA) geometry. deepseek2 (DeepSeek-V2-Lite) is the dev
        // model: lite (q_lora=0 → direct wq), no DSA indexer, key_length not _mla.
        if arch == "glm-dsa" || arch == "deepseek2" {
            let mut cfg = MlaConfig {
                q_lora:  g.meta_u32(&format!("{arch}.attention.q_lora_rank")).unwrap_or(0),
                kv_lora: g.meta_u32(&format!("{arch}.attention.kv_lora_rank")).unwrap_or(512),
                k_mla:   g.meta_u32(&format!("{arch}.attention.key_length_mla"))
                    .or_else(|| g.meta_u32(&format!("{arch}.attention.key_length"))).unwrap_or(576),
                v_mla:   g.meta_u32(&format!("{arch}.attention.value_length_mla"))
                    .or_else(|| g.meta_u32(&format!("{arch}.attention.value_length"))).unwrap_or(512),
                qk_rope: g.meta_u32(&format!("{arch}.rope.dimension_count")).unwrap_or(64),
                n_expert:   g.meta_u32(&format!("{arch}.expert_count")).unwrap_or(0),
                n_used:     g.meta_u32(&format!("{arch}.expert_used_count")).unwrap_or(8),
                n_group:    g.meta_u32(&format!("{arch}.expert_group_count")).unwrap_or(1),
                group_used: g.meta_u32(&format!("{arch}.expert_group_used_count")).unwrap_or(1),
                n_shared:   g.meta_u32(&format!("{arch}.expert_shared_count")).unwrap_or(1),
                ffn_exp:    g.meta_u32(&format!("{arch}.expert_feed_forward_length")).unwrap_or(0),
                leading_dense: g.meta_u32(&format!("{arch}.leading_dense_block_count")).unwrap_or(0),
                indexer: IndexerConfig::from_gguf(g, &arch),
                sigmoid_router: arch == "glm-dsa",   // GLM-5.2 = DeepSeek-V3 sigmoid router
                routed_scale: g.meta_f32(&format!("{arch}.expert_weights_scale")).unwrap_or(1.0),
                interleaved: arch == "glm-dsa",      // GLM uses interleaved-pair RoPE
                absorb: true,                        // finalized below once max_seq is known
            };
            // Adaptive top-k (AdapMoE): read/compute fewer experts/token → less gather.
            // The router still scores all experts but keeps only the top-K (renormalized).
            if let Some(k) = ecfg.top_k {
                if k > 0 && k < cfg.n_used { tracing::info!(target: "mla", "OJAS_TOPK: top-{k} of {} experts (fewer bytes/token)", cfg.n_used); cfg.n_used = k; }
            }
            let nl = g.meta_u32(&format!("{arch}.block_count")).unwrap_or(0);
            let nh = g.meta_u32(&format!("{arch}.attention.head_count")).unwrap_or(0);
            let dd = g.meta_u32(&format!("{arch}.embedding_length")).unwrap_or(0);
            let lite = if cfg.q_lora == 0 { " (lite)" } else { "" };
            tracing::info!(target: "mla", "{arch}: d={dd} L={nl} heads={nh}");
            tracing::info!(target: "mla", "{arch}: MLA{}: q_lora={} kv_lora={} k_mla={} (nope={} rope={}) v_mla={}  → KV/token={} B",
                lite, cfg.q_lora, cfg.kv_lora, cfg.k_mla, cfg.k_mla.saturating_sub(cfg.qk_rope), cfg.qk_rope, cfg.v_mla,
                (cfg.kv_lora + cfg.qk_rope) * 2);
            tracing::info!(target: "mla", "{arch}: MoE: {} experts (top-{}), {} shared, ffn_exp={}, {} leading-dense",
                cfg.n_expert, cfg.n_used, cfg.n_shared, cfg.ffn_exp, cfg.leading_dense);
            mla_cfg = Some(cfg);
            // glm-dsa (GLM-5.2): MLA absorption + q_lora + sigmoid router + interleaved rope.
            // DSA indexer + MTP are optional (dense attention is correct); untested pending a GLM GGUF.
            if arch == "glm-dsa" {
                tracing::warn!(target: "mla", "glm-dsa: q_lora path + DeepSeek-V3 sigmoid router (scale={}) + interleaved RoPE — UNTESTED (no GLM model yet)",
                    mla_cfg.as_ref().map(|c| c.routed_scale).unwrap_or(1.0));
            }
        }
        // ---- shape and optional features: read once, from `ojas-arch` ---------------------
        //
        // Nothing here re-derives these keys inline: the backend-neutral `ArchSpec` is the
        // single reader, because the CUDA side needs the same answers and a second reader of
        // the same file is how the two drift apart. Points where they had already drifted,
        // since corrected in `ojas-arch` to match this file: the MTP layer subtraction, the
        // per-layer `head_count_kv` array, the `expert_shared_feed_forward_length` fallback,
        // and a rope_base default of 10000 against this file's 1e6.
        let spec = ojas_arch::ArchSpec::from_gguf(g)?;
        let d = spec.d;
        // With a sidecar the main file reports nextn_predict_layers=0 and a block_count that
        // already excludes the draft block: the draft count is 1 whatever the metadata says,
        // the main stack is block_count as-is, and the draft sits at blk.{block_count}. For
        // inline MTP models `spec.n_layers` has already subtracted it.
        let n_nextn = if mtp_sidecar { 1 } else { spec.n_nextn };
        let n_layers = if mtp_sidecar { spec.block_count } else { spec.n_layers };
        let n_head = spec.layers[0].n_head as usize;
        let n_kv = spec.layers[0].n_kv as usize;
        let ffn = spec.ffn;
        let hd = spec.layers[0].head_dim as usize;
        let kvdim = n_kv * hd;
        let rope_base = spec.layers[0].rope_base;
        let eps = spec.eps;
        let vocab = spec.vocab;
        let qkv_bias = spec.qkv_bias;
        let qk_norm = spec.qk_norm;
        let tied_embed = spec.tied_head();
        let lm_head_name: String = spec.lm_head.clone();
        let is_gemma = spec.act == ojas_arch::Act::Gelu;
        let sandwich = spec.sandwich;
        let embed_scale = spec.embed_scale;
        tracing::info!(target: "arch", "{arch}: d={d} L={n_layers} heads={n_head}/{n_kv} hd={hd} ffn={ffn} \
                   | qkv_bias={qkv_bias} qk_norm={qk_norm} tied_lm_head={tied_embed} sandwich={sandwich} gelu={is_gemma}");
        // The note is about decoders. `modern-bert` is an encoder, gated by
        // `examples/laya_gate.rs`.
        if !matches!(arch.as_str(), "qwen2" | "qwen3" | "llama" | "gemma3" | "modern-bert") {
            tracing::warn!(target: "arch", "NOTE: arch '{arch}' partially supported; validation is model-, quantization-, and execution-path-specific");
        }

        if arch == "qwen4exp" {
            tracing::warn!(target: "arch", "qwen4exp: learned sparse attention is not applied; context is limited to the indexer's candidate budget");
        }

        // pipelines: compile the kernel families the config needs, one library each.
        // Families are functional (gemv/ops/attn_core/attn/mla/ssm/moe) — nothing
        // model-specific lives in the kernel layer.
        let has_mla = arch == "glm-dsa" || arch == "deepseek2";
        let has_ssm = arch == "qwen35" || arch == "qwen35moe" || arch == "qwen4exp";
        let has_moe = has_mla || arch == "gpt-oss"
            || g.meta_u32(&format!("{arch}.expert_count")).unwrap_or(0) > 0;
        let mut fams = vec!["ops", "gemv", "attn_core", "attn"];
        if has_mla { fams.push("mla"); }
        if has_ssm { fams.push("ssm"); }
        if arch == "qwen4exp" { fams.push("qwen4exp"); }
        // ViT tower: LayerNorm-with-bias, standalone GELU, patch-embed im2col. Everything
        // else the tower runs (GEMM, bias-add, bidirectional attention) is already in
        // ops/gemv/attn_core/attn. Gated on the mmproj actually being attached.
        let has_vision = g.tensors.keys().any(|n| n.starts_with("v.blk."));
        if has_vision { fams.push("vision"); }
        // Geometry first, allocation later: a malformed mmproj must fail with a key
        // name here, not with an unwrap inside graph construction.
        let vision: Option<VisionConfig> = if has_vision { Some(VisionConfig::from_gguf(g)?) } else { None };
        // Text encoder: LayerNorm and the QKV preparation (split, rotation, half K/V)
        // come from the same encoder family as the ViT tower.
        let text_encoder: Option<TextEncoderConfig> =
            if arch == "modern-bert" { Some(TextEncoderConfig::from_gguf(g)?) } else { None };
        if text_encoder.is_some() && !has_vision { fams.push("vision"); }
        if let Some(v) = &vision {
            tracing::info!(target: "vision", "vit d={} L={} heads={} hd={} ffn={} patch={} c={} \
                pos_grid={}x{} merge={} proj={} mm_hidden={} eps={:e} | arena {} patches",
                v.d, v.layers, v.n_head, v.hd, v.ffn, v.patch, v.channels, v.pos_side, v.pos_side,
                v.merge, v.proj_dim, v.mm_hidden, v.eps, v.max_patches);
        }
        let has_f32_matrix = g.tensors.values().any(|t| t.ggml_type == 0 && t.dims.len() >= 2);
        if has_moe || has_f32_matrix { fams.push("moe"); }
        if has_moe { fams.push("moe_iq"); }
        // The IQ1/IQ2/IQ3 native matvecs live with the codebooks in `requant_iq`
        // (~15 KB of constant tables kept out of the shared prelude), so that
        // family only compiles when something actually needs to decode them.
        if native_quant { fams.push("requant_iq"); }
        let mut p = HashMap::new();
        for f in fams {
            let src = ojas_metal::kernels::family_source(f)
                .ok_or_else(|| anyhow::anyhow!("unknown kernel family {f}"))?;
            // simdgroup_matrix (MMA) needs family 7+ — the shuffle shim can't
            // cover it; dispatch sites fall back to non-MMA kernels instead.
            let native = gpu.native_reduce;
            for (name, pipe) in gpu.compile_all(src, |n| {
                native || !(n.starts_with("gemm_mm") || n.contains("mma"))
            })? {
                p.insert(name, pipe);
            }
        }
        // Fat-tile GEMM family (custom fragment storage): `#pragma METAL
        // internals` is semi-internal, so compiled fallibly like the async family.
        if gpu.native_reduce {
            match gpu.compile_all(ojas_metal::kernels::gemm_fat::GEMM_FAT_KERNELS, |_| true) {
                Ok(list) => { for (name, pipe) in list { p.insert(name, pipe); } }
                Err(e) => tracing::warn!(target: "gpu", "fat-tile GEMM unavailable ({e}); using the staged GEMM"),
            }
        }
        // Async-copy GEMM family: undocumented ABI, compiled fallibly — a compile
        // failure logs and falls back to the plain GEMM. Only on Apple7/8: the reference
        // documents async copies as a slowdown on Apple9+ (M3 and later).
        if gpu.native_reduce && gpu.async_copy_ok() {
            match gpu.compile_all(ojas_metal::kernels::gemm_ac::GEMM_AC_KERNELS, |_| true) {
                Ok(list) => { for (name, pipe) in list { p.insert(name, pipe); } }
                Err(e) => tracing::warn!(target: "gpu", "async-copy GEMM unavailable ({e}); using the staged GEMM"),
            }
        }
        if gpu.native_reduce {
            for &hdv in ojas_metal::kernels::attn::ATTN_HD_SPECIAL {
                let name = format!("attention_m_mma_{hdv}");
                p.insert(name.clone(), gpu.pipeline(&ojas_metal::kernels::attn::attn_mma_hd_src(hdv), &name)?);
                let dqn = format!("attention_m_mma_dq_{hdv}");
                p.insert(dqn.clone(), gpu.pipeline(&ojas_metal::kernels::attn::attn_mma_dq_hd_src(hdv), &dqn)?);
            }
            // Bidirectional twins, for encoder towers where every query sees every key.
            // Same kernel with buffer(6) read as the whole KV length instead of base_pos.
            // MMA-only, so every dispatch site must probe `self.p.contains_key(..)` and
            // fall back to `attention_m_bidir` on non-Apple7.
            for &hdv in ojas_metal::kernels::attn::ATTN_BIDIR_HD {
                let name = format!("attention_m_mma_bidir_{hdv}");
                p.insert(name.clone(), gpu.pipeline(&ojas_metal::kernels::attn::attn_mma_bidir_src(hdv), &name)?);
                let dqn = format!("attention_m_mma_dq_bidir_{hdv}");
                p.insert(dqn.clone(), gpu.pipeline(&ojas_metal::kernels::attn::attn_mma_dq_bidir_src(hdv), &dqn)?);
            }
        }
        if let Some(te) = &text_encoder {
            let name = ojas_metal::kernels::attn::ATTN_BIDIR_SPAN;
            p.insert(name.to_string(), gpu.pipeline(&ojas_metal::kernels::attn::attn_bidir_span_src(), name)?);
            // The tiled twin needs simdgroup matrices; without them the text encoder
            // runs the per-query kernel above.
            if gpu.native_reduce {
                let name = ojas_metal::kernels::attn::attn_mma_span_name(te.hd);
                p.insert(name.clone(), gpu.pipeline(&ojas_metal::kernels::attn::attn_mma_span_src(te.hd), &name)?);
            }
        }


        // upload weights
        let mut w16 = HashMap::new();
        let mut w32 = HashMap::new();
        let mut w8 = HashMap::new();
        let mut scale8 = HashMap::new();
        let mut w4 = HashMap::new();
        let mut scale4 = HashMap::new();
        let mut w4k = HashMap::new();
        let mut w6k = HashMap::new();
        let mut w4l = HashMap::new();
        let mut q4l_a = HashMap::new();
        let mut q4l_b = HashMap::new();
        let mut w20 = HashMap::new();
        let mut s20 = HashMap::new();
        let mut w_off: HashMap<String, u64> = HashMap::new();
        let mut w_qtype: HashMap<String, u32> = HashMap::new();
        // Per-weight (K, N) straight from the GGUF header — the shape every GEMM must
        // match. For 2D weights that is the full shape; for 3D MoE expert stacks
        // ([K, N_per_expert, n_expert]) it is the per-expert (K, N), which is what the
        // moe_gu/moe_down kernels contract over. `check_shape` compares against this to
        // catch a d-vs-qdim-vs-kvdim (or expert K/N) dispatch mistake.
        let mut wshape: HashMap<String, (u32, u32)> = HashMap::new();
        for (nm, info) in g.tensors.iter() {
            // Rank 3 is the MoE expert stack, recorded per-expert as above. Rank 4 is the
            // vision patch-embed conv [kw,kh,ic,oc]: recording (dims[0], dims[1]) would
            // claim (16,16) and trip check_shape's debug_assert on every dispatch that
            // names it, so flatten instead.
            if info.dims.len() == 4 {
                let n: u64 = info.dims[1..].iter().product();
                wshape.insert(nm.clone(), (info.dims[0] as u32, n as u32));
            } else if info.dims.len() >= 2 {
                wshape.insert(nm.clone(), (info.dims[0] as u32, info.dims[1] as u32));
            }
        }
        let mut stream_meta: HashMap<String, (usize, u64, u64, u32)> = HashMap::new();
        let mut mapped_gguf: Option<crate::weights::MappedGguf> = None;
        let newbuf = |v: &[u8]| gpu.device.new_buffer_with_data(v.as_ptr() as *const c_void, v.len() as u64, MTLResourceOptions::StorageModeShared);
        // Q8 -> the engine's Q4, a second GPU pass. `w4`/`scale4` feed the tuned 4-bit family
        // (gemv_q4_fast / _ksplit / r4, autotuned threadgroups) which measured
        // 9.39 ms/token against 12.32 for Q8 and 16.79 for native Q4_K.
        let requant_q4 = |q8: &metal::Buffer, s8: &metal::Buffer, k: usize, n: usize|
         -> Option<(metal::Buffer, metal::Buffer)> {
            if k % 32 != 0 { return None; }
            let src = ojas_metal::kernels::source_of("requant_q8_to_q4")?;
            let pipe = gpu.pipeline(src, "requant_q8_to_q4").ok()?;
            let nib = gpu.alloc((n * (k / 2) + 3) / 4);
            let sc = gpu.alloc((n * (k / 32) + 1) / 2);
            let cb = gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(q8), 0);
            enc.set_buffer(1, Some(s8), 0);
            enc.set_buffer(2, Some(&nib.buf), 0);
            enc.set_buffer(3, Some(&sc.buf), 0);
            let (ku, nu) = (k as u32, n as u32);
            enc.set_bytes(4, 4, &ku as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &nu as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(((n + 7) / 8) as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            let _ = ojas_metal::commit_and_wait_checked(cb, "model load");
            Some((nib.buf, sc.buf))
        };
        let requant_q80_q4 = |src_buf: &metal::Buffer, src_off: u64, k: usize, n: usize|
         -> Option<(metal::Buffer, metal::Buffer)> {
            if k % 32 != 0 { return None; }
            let entry = "requant_q80_q4";
            let src = ojas_metal::kernels::source_of(entry)?;
            let pipe = gpu.pipeline(src, entry).ok()?;
            let nib = gpu.alloc((n.checked_mul(k / 2)? + 3) / 4);
            let sc = gpu.alloc((n.checked_mul(k / 32)?.checked_mul(2)? + 3) / 4);
            let cb = gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(src_buf), src_off);
            enc.set_buffer(1, Some(&nib.buf), 0);
            enc.set_buffer(2, Some(&sc.buf), 0);
            let (ku, nu) = (k as u32, n as u32);
            enc.set_bytes(3, 4, &ku as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(nu.div_ceil(8) as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            ojas_metal::commit_and_wait_checked(cb, "Q8_0 output-head requant").ok()?;
            Some((nib.buf, sc.buf))
        };
        // Q4_K -> Q4L: same values, layout the tuned kernel wants. One GPU pass.
        let relayout_q4l = |src_buf: &metal::Buffer, src_off: u64, k: usize, n: usize|
         -> Option<(metal::Buffer, metal::Buffer, metal::Buffer)> {
            if k % 256 != 0 { return None; }
            let src = ojas_metal::kernels::source_of("relayout_q4k_q4l")?;
            let pipe = gpu.pipeline(src, "relayout_q4k_q4l").ok()?;
            let nib = gpu.alloc((n * (k / 2) + 3) / 4);
            let qa = gpu.alloc((n * (k / 32) + 1) / 2);
            let qb = gpu.alloc((n * (k / 32) + 1) / 2);
            let cb = gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(src_buf), src_off);
            enc.set_buffer(1, Some(&nib.buf), 0);
            enc.set_buffer(2, Some(&qa.buf), 0);
            enc.set_buffer(3, Some(&qb.buf), 0);
            let (ku, nu) = (k as u32, n as u32);
            enc.set_bytes(4, 4, &ku as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &nu as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(((n + 7) / 8) as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            let _ = ojas_metal::commit_and_wait_checked(cb, "model load");
            Some((nib.buf, qa.buf, qb.buf))
        };
        // K-quant -> Q8 on the GPU. Decode wants Q8 (one int8*scale per weight) while
        // loading wants the K-quant blocks untouched; the same conversion on the CPU costs
        // minutes (dequantize every block to f16, then scan each row twice), so it runs
        // here, on hardware already holding the bytes.
        let requant_q8 = |src_buf: &metal::Buffer, src_off: u64, k: usize, n: usize, ty: u32| -> Option<(metal::Buffer, metal::Buffer)> {
            let (entry, group) = match ty {
                19 => ("requant_iq1s_q8", 256),
                29 => ("requant_iq1m_q8", 256),
                16 => ("requant_iq2xxs_q8", 256),
                17 => ("requant_iq2xs_q8", 256),
                18 => ("requant_iq3xxs_q8", 256),
                21 => ("requant_iq3s_q8", 256),
                22 => ("requant_iq2s_q8", 256),
                20 => ("requant_iq4nl_q8", 32),
                23 => ("requant_iq4xs_q8", 256),
                10 => ("requant_q2k_q8", 256),
                11 => ("requant_q3k_q8", 256),
                12 => ("requant_q4k_q8", 256),
                14 => ("requant_q6k_q8", 256),
                13 => ("requant_q5k_q8", 256),
                8 => ("requant_q80_q8", 32),
                _ => return None,
            };
            if k % group != 0 { return None; }
            let src = ojas_metal::kernels::source_of(entry)?;
            let pipe = gpu.pipeline(src, entry).ok()?;
            // int8 weights + one f32 scale per row. `alloc` sizes in f32 units.
            let qb = gpu.alloc((n * k + 3) / 4);
            let sb = gpu.alloc(n);
            let cb = gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(src_buf), src_off);
            enc.set_buffer(1, Some(&qb.buf), 0);
            enc.set_buffer(2, Some(&sb.buf), 0);
            let (ku, nu) = (k as u32, n as u32);
            enc.set_bytes(3, 4, &ku as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(((n + 7) / 8) as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            let _ = ojas_metal::commit_and_wait_checked(cb, "model load");
            Some((qb.buf, sb.buf))
        };
        let mut names: Vec<String> = vec!["token_embd.weight".into()];
        // Vision tensors ride the same upload loop as everything else; appended after
        // the arch chain below so they are never confused with a decoder tensor.
        let mut vision_names: Vec<String> =
            g.tensors.keys().filter(|n| ojas_formats::mmproj::keep_tensor(n)).cloned().collect();
        vision_names.sort();
        // qwen4exp has no final RMSNorm: the terminal hyper-connection head carries it.
        if arch != "qwen4exp" { names.push("output_norm.weight".into()); }
        // GLM (glm-dsa) splits the MLA up-proj into attn_k_b + attn_v_b (the reference MLA
        // naming); deepseek-lite ships a single attn_kv_b. Detect & synthesize below.
        let split_kvb = g.tensors.contains_key("blk.0.attn_k_b.weight")
            && !g.tensors.contains_key("blk.0.attn_kv_b.weight");
        if !tied_embed { names.push("output.weight".into()); }
        if let Some(te) = &text_encoder {
            names = te.gpu_tensors(g);
        } else if arch == "qwen35" || arch == "qwen35moe" {
            // Gated-DeltaNet hybrid: SSM layers vs attention layers have different tensors.
            let interval = g.meta_u32(&format!("{arch}.full_attention_interval")).unwrap_or(4);
            let moe_arch = arch == "qwen35moe";
            for i in 0..n_layers {
                let is_ssm = (i as u32 + 1) % interval != 0;
                for s in ["attn_norm.weight", "post_attention_norm.weight"] {
                    names.push(format!("blk.{i}.{s}"));
                }
                if moe_arch {
                    // MoE FFN: router (f32) + 3D expert tensors + always-on shared expert
                    for s in ["ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
                              "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                              "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"] {
                        names.push(format!("blk.{i}.{s}"));
                    }
                } else {
                    for s in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                        names.push(format!("blk.{i}.{s}"));
                    }
                }
                if is_ssm {
                    for s in ["attn_qkv.weight", "attn_gate.weight", "ssm_conv1d.weight", "ssm_a",
                              "ssm_alpha.weight", "ssm_beta.weight", "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight"] {
                        names.push(format!("blk.{i}.{s}"));
                    }
                } else {
                    for s in ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                              "attn_q_norm.weight", "attn_k_norm.weight"] {
                        names.push(format!("blk.{i}.{s}"));
                    }
                }
            }
            // NextN/MTP block (blk.{n_layers}): a standard gated-attention layer +
            // FFN + the nextn combiner/head tensors.
            if n_nextn > 0 && g.tensors.contains_key(&format!("blk.{n_layers}.nextn.eh_proj.weight")) {
                let i = n_layers;
                for s in ["attn_norm.weight", "post_attention_norm.weight",
                          "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                          "attn_q_norm.weight", "attn_k_norm.weight",
                          "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight"] {
                    names.push(format!("blk.{i}.{s}"));
                }
                if moe_arch {
                    for s in ["ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
                              "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                              "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"] {
                        names.push(format!("blk.{i}.{s}"));
                    }
                } else {
                    for s in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                        names.push(format!("blk.{i}.{s}"));
                    }
                }
                for s in ["nextn.shared_head_norm.weight", "nextn.shared_head_head.weight", "nextn.embed_tokens.weight"] {
                    if g.tensors.contains_key(&format!("blk.{i}.{s}")) { names.push(format!("blk.{i}.{s}")); }
                }
            }
        } else if arch == "qwen4exp" {
            // Hyper-connection hybrid: gated-DeltaNet recurrent layers and
            // indexer-selected sparse-attention layers, each wrapped by two
            // hyper-connection mixers that stand in for the usual attention/FFN
            // norms. A subset of layers additionally carry an n-gram embedding
            // table. Attention layers use the fused qkv projection.
            let interval = g.meta_u32(&format!("{arch}.full_attention_interval")).unwrap_or(4);
            let ple_layers: Vec<u32> = g.int_arr(&format!("{arch}.ple.layers"))
                .map(|a| a.iter().map(|&v| v as u32).collect()).unwrap_or_default();
            for i in 0..n_layers {
                let is_recr = (i as u32 + 1) % interval != 0;
                for s in ["hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
                          "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight"] {
                    names.push(format!("blk.{i}.{s}"));
                }
                for s in ["ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
                          "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                          "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"] {
                    names.push(format!("blk.{i}.{s}"));
                }
                if is_recr {
                    for s in ["attn_qkv.weight", "attn_gate.weight", "ssm_conv1d.weight", "ssm_a",
                              "ssm_alpha.weight", "ssm_beta.weight", "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight"] {
                        names.push(format!("blk.{i}.{s}"));
                    }
                } else {
                    // sparse attention: split q/k/v (the q projection is doubled to carry a
                    // per-head gate) plus the learned indexer's q/k projections and norms.
                    for s in ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                              "attn_q_norm.weight", "attn_k_norm.weight",
                              "indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"] {
                        names.push(format!("blk.{i}.{s}"));
                    }
                }
                if ple_layers.contains(&(i as u32)) {
                    for s in ["ple_key.weight", "ple_value.weight", "ple_norm_key.weight",
                              "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight"] {
                        names.push(format!("blk.{i}.{s}"));
                    }
                }
            }
            // NextN/MTP draft block (blk.{n_layers}). Unlike qwen35's, it is a full
            // qwen4exp layer — hyper-connections, sparse attention, its own 512-expert
            // MoE — plus the nextn combiner and its own terminal mixer, so drafting is
            // one more pass of the same graph rather than a dense block.
            if n_nextn > 0 && g.tensors.contains_key(&format!("blk.{n_layers}.nextn.eh_proj.weight")) {
                let i = n_layers;
                for s in ["hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
                          "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
                          "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                          "attn_q_norm.weight", "attn_k_norm.weight",
                          "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
                          "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                          "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                          "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
                          "nextn.hc_head_norm.weight", "nextn.hc_head_down.weight", "nextn.hc_head_up.weight"] {
                    names.push(format!("blk.{i}.{s}"));
                }
                // A `shared-` head borrows token_embd/output from the model it drafts for.
                for s in ["nextn.embed_tokens.weight", "nextn.shared_head_head.weight"] {
                    if g.tensors.contains_key(&format!("blk.{i}.{s}")) { names.push(format!("blk.{i}.{s}")); }
                }
            }
            if !ple_layers.is_empty() { names.push("per_layer_token_embd.weight".into()); }
            for s in ["output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight"] {
                names.push(s.into());
            }
            if g.tensors.contains_key("output.weight") { names.push("output.weight".into()); }
        } else if let Some(m) = mla_cfg {
            for i in 0..n_layers {
                for s in ["attn_norm.weight", "attn_kv_a_mqa.weight",
                          "attn_kv_a_norm.weight",
                          "attn_output.weight", "ffn_norm.weight"] {
                    names.push(format!("blk.{i}.{s}"));
                }
                if !split_kvb { names.push(format!("blk.{i}.attn_kv_b.weight")); }
                // Query: GLM (q_lora>0) uses attn_q_a → q_a_norm → attn_q_b; deepseek-lite uses attn_q.
                if m.q_lora > 0 {
                    for s in ["attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight"] { names.push(format!("blk.{i}.{s}")); }
                } else {
                    names.push(format!("blk.{i}.attn_q.weight"));
                }
                if (i as u32) < m.leading_dense {
                    for s in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] { names.push(format!("blk.{i}.{s}")); }
                } else {
                    for s in ["ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                              "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                              "ffn_down_shexp.weight"] { names.push(format!("blk.{i}.{s}")); }
                    // GLM DeepSeek-V3 router selection bias (f32).
                    if m.sigmoid_router { names.push(format!("blk.{i}.exp_probs_b.bias")); }
                }
            }
        } else if arch == "gpt-oss" {
            // GPT-OSS (OpenAI MoE): GQA attention with per-head sinks + q/k/v/output
            // biases; 32-expert top-4 MoE with biased router and biased experts (no
            // shared expert); SwiGLU-OAI gate. attn_norm=pre-attn, post_attention_norm=pre-FFN.
            for i in 0..n_layers {
                for s in ["attn_norm.weight", "post_attention_norm.weight",
                          "attn_q.weight", "attn_q.bias", "attn_k.weight", "attn_k.bias",
                          "attn_v.weight", "attn_v.bias", "attn_output.weight", "attn_output.bias",
                          "attn_sinks.weight",
                          "ffn_gate_inp.weight", "ffn_gate_inp.bias",
                          "ffn_gate_exps.weight", "ffn_gate_exps.bias",
                          "ffn_up_exps.weight", "ffn_up_exps.bias",
                          "ffn_down_exps.weight", "ffn_down_exps.bias"] {
                    names.push(format!("blk.{i}.{s}"));
                }
            }
        } else {
        for i in 0..n_layers {
            for s in [
                "attn_norm.weight", "attn_q.weight", "attn_k.weight",
                "attn_output.weight", "ffn_norm.weight",
                "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            ] {
                names.push(format!("blk.{i}.{s}"));
            }
            // attn_v absent in gemma-4 KV-shared layers → zero dummy added after load.
            if g.tensors.contains_key(&format!("blk.{i}.attn_v.weight")) {
                names.push(format!("blk.{i}.attn_v.weight"));
            }
            if qkv_bias {
                for s in ["attn_q.bias", "attn_k.bias", "attn_v.bias"] { names.push(format!("blk.{i}.{s}")); }
            }
            if qk_norm {
                for s in ["attn_q_norm.weight", "attn_k_norm.weight"] { names.push(format!("blk.{i}.{s}")); }
            }
            if sandwich {
                for s in ["post_attention_norm.weight", "post_ffw_norm.weight"] { names.push(format!("blk.{i}.{s}")); }
            }
        }
        }
        // The vision tower, if an mmproj was attached. Uploaded by the same loop as
        // every decoder tensor — the maps are name-keyed with no arch gate, so `mm`
        // and `projm` reach `v.*`/`mm.*` with no new code.
        names.extend(vision_names);
        // PLE is a CPU-only sparse row source at every precision. Remove it before any
        // upload/requant/cache path can materialize the whole table.
        let ple = if names.iter().any(|n| n == ple::TABLE) {
            Some(ple::PleTable::from_gguf(g)?)
        } else { None };
        let mut seen_names = std::collections::HashSet::new();
        names.retain(|n| !matches!(role_of(n), TensorRole::NgramTable) && seen_names.insert(n.clone()));

        // Native fused SwiGLU is the default: with the occupancy fix in Q4K_DOT_L it is
        // faster than the Q8 path on decode and reads half the bytes. OJAS_FFN_Q8=1 forces
        // the requantized route.
        let ffn_native_opt_in = !ojas_core::config::flag("OJAS_FFN_Q8");
        // GGUF type per tensor name, for the same-type-pair check below.
        let g_types: HashMap<String, u32> =
            g.tensors.iter().map(|(k, v)| (k.clone(), v.ggml_type)).collect();
        // Zero-copy source for natively-formatted tensors. The GGUF on disk is already in
        // the layout `gemv_q4k`/`gemv_q6k` read, so the file is mapped and Metal points at
        // those pages instead of reading them into RAM and memcpying them into a buffer.
        // OJAS_NO_MMAP=1 forces the copy.
        //
        // Not used in `stream` mode, which opens its own map below for experts.
        let mut mmap_bytes = 0u64;
        // Not gated on `q4k`: the GPU requantizer reads these pages too, so an arch
        // on the conservative prec=1 path (qwen35 et al.) gets the same zero-copy
        // source even though none of its tensors stay native.
        let native_map: Option<crate::weights::MappedGguf> = if !stream && !ecfg.no_mmap {
            if ple.is_some() {
                // Cached mappings prefetch whole shards, including the lazy PLE
                // range. Leave them demand-paged when a CPU-only table exists.
                Some(crate::weights::MappedGguf::from_files(g.shard_files()?)?)
            } else if g.is_fd_backed() {
                g.shard_files().and_then(crate::weights::MappedGguf::from_files_cached).ok()
            } else {
                crate::weights::MappedGguf::open_cached(&g.shard_paths()).ok()
            }
        } else {
            None
        };
        let _t_upload = std::time::Instant::now();
        // ---- q4 mmap weight cache ("LLM in a flash", see weights.rs):
        // requant once into a page-aligned cache file, then every load mmaps it and wraps
        // zero-copy Metal buffers — instant loads with OS-managed residency. Skeleton
        // tensors get pinned; MoE "_exps." tensors stay lazy (the page cache is the expert
        // cache). OJAS_NO_MMAP=1 forces the in-RAM path.
        let mut wq: HashMap<String, metal::Buffer> = HashMap::new();
        let mut wcache: Option<crate::weights::MappedCache> = None;
        if stream {
            // ---- Direct-mmap streaming: experts zero-copy from the GGUF shards (with byte
            // offset for 16KB alignment); skeleton dequant→f16 in RAM (pinned by residency).
            use crate::weights as wc;
            // Descriptor-backed models cannot name their shards, so they hand
            // over duplicated fds instead of paths.
            let mg = if g.is_fd_backed() {
                wc::MappedGguf::from_files(g.shard_files()?)?
            } else {
                wc::MappedGguf::open(&g.shard_paths())?
            };
            let (mut streamed, mut skeleton_bytes) = (0u64, 0u64);
            // Start with metadata-only experts: the full resident request can only be
            // budgeted once every state/scratch allocation exists.
            for name in &names {
                if matches!(super::role_of(name), super::TensorRole::MoeExpert) {
                    if let Some((part, abs, rawlen, ty)) = g.tensor_meta(name) {
                        // Validated against the same table the graph dispatches from, and per
                        // role: a format valid as gate/up is not necessarily valid as down
                        // (a role-blind list of type ids admits Q4_K down and Q8_0 gate/up,
                        // which the dispatcher's catch-all then decodes with the wrong
                        // walker). Rejecting here lets the graph assume a kernel exists.
                        let role = ojas_core::quant_src::moe_role_of(name);
                        if ojas_core::quant_src::moe_kernel(ty, role).is_none() {
                            anyhow::bail!("stream: expert {name} is GGUF type {ty}, which has no \
                                {role:?} MoE kernel — add a row to quant_src::MOE_FORMATS and the kernel it names");
                        }
                        w_qtype.insert(name.clone(), ty);
                        stream_meta.insert(name.clone(), (part, abs, rawlen, ty));
                        streamed += rawlen;
                        continue;
                    }
                }
                // No requant cache here, so the alternative to the file's own format is
                // f16, which native beats on both axes: IQ4_XS measured 332.8 Gw/s
                // against f16's ~186, at a quarter of the bytes.
                if let Some(info) = g.tensors.get(name) {
                    let ty = info.ggml_type;
                    // A Q6_K token table stays in its blocks: `embed_q6k` gathers from
                    // them, and a tied head reads them through the native Q6_K GEMV at
                    // half the bytes of the f16 copy it would otherwise get (Qwen3.5 4B
                    // Q4_K_M: 636 MB/token instead of 1271).
                    if ty == 14 && name == "token_embd.weight" {
                        if let Some((part, abs, rawlen, _)) = g.tensor_meta(name) {
                            let (buf, off) = mg.buffer(gpu, part, abs, rawlen);
                            if off > 0 { w_off.insert(name.clone(), off); }
                            w6k.insert(name.clone(), buf);
                            skeleton_bytes += rawlen;
                            continue;
                        }
                    }
                    let wpb = ojas_metal::kernels::nat::nat_wpb(ty).unwrap_or(0) as u64;
                    // Flash's shared experts use generic matmul dispatch and its Q8
                    // embedding has a native gather, so both keep their source values:
                    // dequantizing them to F16 adds rounding that can change later MoE
                    // routing even in the "native" precision mode.
                    let native_embed = is_qwen4exp && ty == 8
                        && (name == "token_embd.weight" || name.ends_with("nextn.embed_tokens.weight"));
                    if wpb > 0 && info.dims.len() == 2 && info.dims[0] % wpb == 0
                        && (native_eligible(name) || native_embed) {
                        if let Some((part, abs, rawlen, _)) = g.tensor_meta(name) {
                            let (buf, off) = mg.buffer(gpu, part, abs, rawlen);
                            let q4_head = ecfg.q4_head && !tied_embed && name == &lm_head_name;
                            let down_layer = name.strip_prefix("blk.")
                                .and_then(|s| s.split('.').next())
                                .and_then(|s| s.parse::<usize>().ok());
                            let q4_down = name.ends_with(".ffn_down.weight")
                                && (ecfg.q4_ffn_down || down_layer.is_some_and(|l| {
                                    ecfg.q4_ffn_down_last > 0
                                        && l >= n_layers.saturating_sub(ecfg.q4_ffn_down_last)
                                }));
                            if arch == "qwen35" && ty == 8 && info.dims[1] % 8 == 0
                                && (q4_head || q4_down) {
                                if let Some((nib, sc)) = requant_q80_q4(
                                    &buf, off, info.dims[0] as usize, info.dims[1] as usize,
                                ) {
                                    skeleton_bytes += nib.length() + sc.length();
                                    w4.insert(name.clone(), nib);
                                    scale4.insert(name.clone(), sc);
                                    tracing::warn!(target: "quant", "mixed Q4: requantized {name} Q8_0 -> Q4 (experimental; activations/logits changed)");
                                    continue;
                                }
                            }
                            if off > 0 { w_off.insert(name.clone(), off); }
                            w_qtype.insert(name.clone(), ty);
                            wq.insert(name.clone(), buf);
                            skeleton_bytes += rawlen;
                            continue;
                        }
                    }
                }
                let (_dims, ty, bytes) = g.read_tensor(name)?;
                if ty == 1 { w16.insert(name.clone(), newbuf(&bytes)); } else { w32.insert(name.clone(), newbuf(&bytes)); }
            }
            let copied_bytes: u64 = w16.values().chain(w32.values()).map(|b| b.length()).sum();
            tracing::info!(target: "stream", "expert weights {:.2} GB available for streaming; skeleton {:.2} GB native + {:.3} GB copied ({} tensors)",
                streamed as f64 / 1e9,
                skeleton_bytes as f64 / 1e9, copied_bytes as f64 / 1e9, w16.len() + w32.len());
            mapped_gguf = Some(mg);
        } else if q4mode && ecfg.prewarm {   // legacy .awc cache path, opt-in via OJAS_PREWARM
            use crate::weights as wc;
            let (ssz, smt) = wc::source_meta(&g.path)?;
            let cpath = wc::cache_path(&g.path, ssz);
            let mut mc = wc::MappedCache::open(&cpath, ssz, smt)?;
            if mc.is_none() {
                tracing::info!(target: "cache", "building weight cache (one-time requant) → {}", cpath.display());
                let mut cw = wc::CacheWriter::create(cpath.clone(), ssz, smt)?;
                for name in &names {
            let (dims, ty, bytes) = g.read_tensor(name)?;
                    if name.contains("ssm_conv1d") {
                        cw.add(name, if ty == 1 { wc::K_W16 } else { wc::K_W32 }, &bytes)?;
                        continue;
                    }
                    if ty == 1 {
                        let n = dims[1..].iter().product::<u64>() as usize;
                        let k = dims[0] as usize;
                        if k % 32 != 0 { anyhow::bail!("q4 cache: K%32!=0 for {name}"); }
                        let (nib, sc) = quantize_row_q4(&bytes, n, k);
                        cw.add(name, wc::K_W4, &nib)?;
                        cw.add(name, wc::K_SCALE4, bytemuck_u16(&sc))?;
                    } else {
                        cw.add(name, wc::K_W32, &bytes)?;
                    }
                }
                cw.finish()?;
                mc = wc::MappedCache::open(&cpath, ssz, smt)?;
            }
            let mc = mc.ok_or_else(|| anyhow::anyhow!("weight cache open failed"))?;
            let mut pinned = 0u64;
            let prewarm = ecfg.prewarm;
            for e in &mc.entries {
                if matches!(role_of(&e.name), TensorRole::NgramTable) { continue; }
                let buf = mc.buffer(gpu, e);
                match e.kind {
                    wc::K_W4 => { w4.insert(e.name.clone(), buf); }
                    wc::K_SCALE4 => { scale4.insert(e.name.clone(), buf); }
                    wc::K_W16 => { w16.insert(e.name.clone(), buf); }
                    _ => { w32.insert(e.name.clone(), buf); }
                }
                // pin the dense skeleton; routed experts stay disk-backed
                if prewarm || !e.name.contains("_exps.") { mc.pin(e); pinned += e.len; }
            }
            tracing::info!(target: "cache", "mmap hit: {} tensors zero-copy, {:.1} GB pinned skeleton{}",
                mc.entries.len(), pinned as f64 / 1e9, if prewarm { " (+prewarm all)" } else { "" });
            wcache = Some(mc);
        } else {
        for name in &names {
            // Mixed precision: the vision tower stays F16 whatever `prec` says. The ViT
            // graph reaches its weights through `projm`, which probes w16/wq and never
            // w4/w8, so a requantized mmproj would be unreachable, not merely lossy. The
            // decoder is bandwidth-bound (1.21 GB/token at F16, measured 141 tok/s against
            // 326 at Q4) while the ViT is compute-bound and gains nothing from narrower
            // weights. Shadowing both flags here covers every branch below.
            let vis = ojas_formats::mmproj::keep_tensor(name);
            let quant = quant && !vis;
            let q4mode = q4mode && !vis;
            // MLA absorption needs attn_kv_b in f16 (small; used by qabsorb/ctx matvecs).
            if mla_cfg.is_some() && name.ends_with("attn_kv_b.weight") {
                let (_d, _t, bytes) = g.read_tensor(name)?;   // dequant to f16
                w16.insert(name.clone(), newbuf(&bytes));
                continue;
            }
            // Native ternary Q2_0: keep the raw g128 blocks; gemv_q20/ffn_gu_q20
            // dequant in-kernel. token_embd is a row gather, not a gemv, so it
            // stays on the dequant path.
            if q20 && !name.contains("token_embd") {
                if let Ok((dims, 42, raw)) = g.read_tensor_raw(name) {
                    if dims[0] % 128 == 0 {
                        let nb = raw.len() / 34;               // total blocks in the tensor
                        let mut codes = vec![0u8; nb * 32];
                        let mut scales = vec![0u16; nb];
                        for b in 0..nb {
                            let src = &raw[b * 34..b * 34 + 34];
                            scales[b] = u16::from_le_bytes([src[0], src[1]]);
                            codes[b * 32..(b + 1) * 32].copy_from_slice(&src[2..34]);
                        }
                        w20.insert(name.clone(), newbuf(&codes));
                        s20.insert(name.clone(), newbuf(bytemuck_u16(&scales)));
                        continue;
                    }
                }
            }
            // Native Q4_K (faithful): keep matrix Q4_K tensors in their GGUF super-block
            // format and dequant in gemv_q4k/moe_gu_q4k. token_embd stays on the Q8 embed
            // path.
            //
            // Native Q4_K/Q6_K is reachable only through the generic `mm()` router. The
            // fused kernels — qkv_q8/qkv_q4 (attn_q/k/v) and ffn_gu_* (ffn_gate/ffn_up) —
            // index `w8`/`w4`/`w16` directly, so moving one of their inputs into a native
            // map panics with "no entry found for key"; they also fold a bias inline that
            // the plain/accum native gemv would silently drop. Expert tensors go through
            // moe_gu_q4k rather than the fused dense kernels, which is why native Q4_K has
            // always worked on the MoE path. Everything the router does handle — ffn_down,
            // o_proj, output — goes native here; attn_q/k/v stay off it because of the
            // inline bias.
            //
            // ffn_gate/ffn_up can go native as a same-type pair (ffn_gu_q4k/_q6k) but are
            // off by default. Measured on Mythos-nano Q4_K_S / M2 Max:
            //
            //   ffn_gu Q8 (requantized at load)   54 tok/s
            //   ffn_gu native K-quant            ~39 tok/s   (-28%)
            //
            // These two are the largest matrices per layer, and unpacking 6-bit K-quant
            // scales costs more than Q8's int8*scale: the Q8 path pays that once at load,
            // the native path on every token. OJAS_FFN_NATIVE=1 opts in, for the
            // load-time-dominated case (a big model loaded repeatedly, generating little).
            //
            // The pair must be the same type: ffn_gu_* reads both operands with one block
            // stride, so a mixed pair would read Q6_K blocks at the Q4_K stride. attn_q/k/v
            // may go native only when all three are Q4_K, because qkv_q4l reads that trio
            // in one dispatch, also with one block stride; mixed types stay on Q8.
            let fused_input = (["attn_q.", "attn_k.", "attn_v."].iter().any(|t| name.contains(t))
                    && !qkv_triple_native(&g_types, name))
                || ((name.contains("ffn_gate.") || name.contains("ffn_up."))
                    && !(ffn_native_opt_in && ffn_pair_native(&g_types, name)));
            // token_embd stays on the Q8 requant path even when the file ships it as Q6_K.
            // Native Q6_K reads 255 MB/token against Q8's 331 — 4% of all decode traffic —
            // and is slower on every path, because unpacking Q6_K's split ql/qh costs more
            // ALU than the bytes save on this GPU:
            //
            //     decode lm_head   0.800 -> 0.959 ms   (413 -> 266 GB/s)
            //     decode floor     7.306 -> 7.438 ms
            //     verify M=4      22.50  -> 25.77 ms
            //     prefill M=256  193.4   -> 195.2 ms
            //
            // The Q6_K kernels (embed_q6k, embed_m_q6k, gemm_mm_q6k) are kept and wired: an
            // untied model whose separate output.weight is Q6_K does land in w6k, and the
            // batched lm_head would otherwise index w8[lm_head] and panic.
            //
            // Q6_K ffn_down (the Q4_K_M mix keeps ffn_down at Q6_K) is redirected out of
            // the native path to the Q8 requant below, so it lands in w8, which gemm_mm_q8
            // consumes; batched_dense_ok requires w8/w4l, so this is what lets a Q4_K_M
            // model batch prefill at all. Native it lands in w6k, read only by gemm_mm_q6k
            // — a kernel exercised only by rare untied Q6_K heads, and one that corrupts
            // the residual stream on the batched ffn_down accum path — batched_dense_ok
            // fails, and the whole prompt processes one token at a time (~4x slower,
            // against the ~24x batched prefill buys). Applies to all dense archs. Q4_K
            // ffn_down (type 12, other layers) stays native w4l.
            //
            // qwen35's SSM-layer attn_qkv is Q6_K in the same mix and is redirected for the
            // same reason: in w6k no batched GEMM reads it, which surfaced first as a
            // "missing MTP matrix dispatch" panic in the chunk graph and then, with a
            // per-row fallback, as prefill paying 24 GEMVs per token.
            let q6k_ffn_down = (name.contains("ffn_down") || name.contains("attn_qkv"))
                && g.tensors.get(name).map(|i| i.ggml_type == 14).unwrap_or(false);
            if q4k && !fused_input && !q6k_ffn_down && !name.contains("token_embd") && !name.contains("ssm_") {
                let ty_now = g.tensors.get(name).map(|i| i.ggml_type).unwrap_or(0);
                let k_ok = g.tensors.get(name).map(|i| i.dims[0] % 256 == 0).unwrap_or(false);
                if matches!(ty_now, 12 | 14) && k_ok {
                    // Zero-copy: hand Metal the mmap'd file pages directly. GGUF
                    // tensors are 32-byte aligned but no-copy buffers must start on a
                    // page, so the buffer covers the page-aligned region and the
                    // leftover byte offset rides along in `w_off` for set_buffer.
                    let mapped = native_map.as_ref().and_then(|mg| {
                        g.tensor_meta(name).map(|(part, abs, rawlen, _)| {
                            let (buf, off) = mg.buffer(gpu, part, abs, rawlen);
                            (buf, off, rawlen)
                        })
                    });
                    if let Some((buf, off, rawlen)) = mapped {
                        mmap_bytes += rawlen;
                        if ty_now == 12 {
                            // Q4_K -> Q4L. Same numbers, contiguous layout: the
                            // interleaved 144-byte block form measured 135 GB/s,
                            // this one runs on the tuned 4-row kernel.
                            let kk = g.tensors[name].dims[0] as usize;
                            let nn: usize = g.tensors[name].dims[1..].iter().product::<u64>() as usize;
                            if let Some((nib, qa, qb)) = relayout_q4l(&buf, off, kk, nn) {
                                w4l.insert(name.clone(), nib);
                                q4l_a.insert(name.clone(), qa);
                                q4l_b.insert(name.clone(), qb);
                                continue;
                            }
                        }
                        if off > 0 {
                            w_off.insert(name.clone(), off);
                        }
                        if ty_now == 12 { w4k.insert(name.clone(), buf); } else { w6k.insert(name.clone(), buf); }
                        continue;
                    }
                    if let Ok((_dims, ty_raw, raw)) = g.read_tensor_raw(name) {
                        if ty_raw == 12 { w4k.insert(name.clone(), newbuf(&raw)); } else { w6k.insert(name.clone(), newbuf(&raw)); }
                        continue;
                    }
                }
            }
            // Fused-kernel inputs need Q8, but not the CPU round trip. token_embd is
            // excluded from the native path above (it is a row gather, not a gemv) but
            // belongs here: GPU requant emits the same w8+scale8 layout the gather and the
            // tied lm_head both read, and on a 150k-row vocab the CPU round trip is most of
            // the load time.
            //
            // The `ssm_` exclusion is deliberately narrow. `ssm_conv1d` and the small
            // a/dt/norm params are used elementwise and must stay f16/f32, but they are
            // stored as F32 and never match the type filter below anyway. `ssm_out` /
            // `ssm_alpha` / `ssm_beta` are ordinary gemv weights, gigabytes of them on a
            // 32-layer hybrid, so excluding them wholesale would send that back to the CPU
            // path.
            // ---- native quantized path: keep the file's own format ----------
            // Requantizing to Q8/Q4L costs 8 or 5 bits per weight regardless of what the
            // file held, so a 2-bit model expands ~4x in GPU memory. Here the blocks reach
            // the GPU untouched and `gemv_nat_*` decodes them in-kernel, so resident size
            // tracks the file and the load is a zero-copy map of pages the OS already has,
            // with no conversion pass.
            //
            // The tuned Q4L/Q8 kernels are faster per weight (iq3_probe: IQ3 is
            // dequant-bound at ~1/3 of Q4L's DRAM rate), so a model that comfortably fits
            // should still requantize. OJAS_NATIVE=1 opts in; OJAS_NATIVE=auto uses native
            // only for formats below 4 bits/weight, where the memory argument dominates.
            if native_quant && native_eligible(name) {
                if let Some(info) = g.tensors.get(name).cloned() {
                    let ty = info.ggml_type;
                    let wpb = ojas_metal::kernels::nat::nat_wpb(ty).unwrap_or(0) as u64;
                    // Strictly 2-D. The requant path accepts 3-D expert stacks by folding
                    // dims[1..] into N, but those are consumed by the MoE kernels, which
                    // read their own packed buffers — sending them here would leave those
                    // lookups empty.
                    let use_native = wpb > 0
                        && info.dims.len() == 2
                        && info.dims[0] % wpb == 0
                        && native_wants(ty);
                    if use_native {
                        if let Some((buf, off)) = native_map.as_ref().and_then(|mg| {
                            g.tensor_meta(name).map(|(part, abs, rawlen, _)| {
                                let (b, o) = mg.buffer(gpu, part, abs, rawlen);
                                mmap_bytes += rawlen;
                                (b, o)
                            })
                        }) {
                            let q4_head = ecfg.q4_head && !tied_embed && name == &lm_head_name;
                            let down_layer = name.strip_prefix("blk.")
                                .and_then(|s| s.split('.').next())
                                .and_then(|s| s.parse::<usize>().ok());
                            let q4_down = name.ends_with(".ffn_down.weight")
                                && (ecfg.q4_ffn_down || down_layer.is_some_and(|l| {
                                    ecfg.q4_ffn_down_last > 0
                                        && l >= n_layers.saturating_sub(ecfg.q4_ffn_down_last)
                                }));
                            if arch == "qwen35" && ty == 8 && info.dims[1] % 8 == 0
                                && (q4_head || q4_down) {
                                if let Some((nib, sc)) = requant_q80_q4(
                                    &buf, off, info.dims[0] as usize, info.dims[1] as usize,
                                ) {
                                    w4.insert(name.clone(), nib);
                                    scale4.insert(name.clone(), sc);
                                    tracing::warn!(target: "quant", "mixed Q4: requantized {name} Q8_0 -> Q4 (experimental; activations/logits changed)");
                                    continue;
                                }
                            }
                            if off > 0 { w_off.insert(name.clone(), off); }
                            // The dispatcher picks the kernel from this, so it must
                            // be recorded with the buffer or the lookup panics.
                            w_qtype.insert(name.clone(), ty);
                            wq.insert(name.clone(), buf);
                            continue;
                        }
                    }
                }
            }
            if (quant || q4mode) && !name.contains("ssm_conv1d") {
                if let Some(info) = g.tensors.get(name).cloned() {
                    if matches!(info.ggml_type,
                        8 | 10 | 11 | 12 | 13 | 14 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 29)
                        && info.dims.len() >= 2
                    {
                        let k = info.dims[0] as usize;
                        let n: usize = info.dims[1..].iter().product::<u64>() as usize;
                        // Feed the requantizer straight off the mmap where possible, rather
                        // than reading the blocks into RAM only to memcpy them into a
                        // Metal buffer.
                        let mapped_src = native_map.as_ref().and_then(|mg| {
                            g.tensor_meta(name).map(|(part, abs, rawlen, ty)| {
                                let (buf, off) = mg.buffer(gpu, part, abs, rawlen);
                                (buf, off, ty, rawlen)
                            })
                        });
                        if let Some((buf, off, ty_raw, rawlen)) = mapped_src {
                            if let Some((qb, sb)) = requant_q8(&buf, off, k, n, ty_raw) {
                                mmap_bytes += rawlen;
                                if q4mode {
                                    // q4mode wants the tuned 4-bit family, not Q8.
                                    if let Some((nib, sc4)) = requant_q4(&qb, &sb, k, n) {
                                        w4.insert(name.clone(), nib);
                                        scale4.insert(name.clone(), sc4);
                                        continue;
                                    }
                                }
                                w8.insert(name.clone(), qb);
                                scale8.insert(name.clone(), sb);
                                continue;
                            }
                        }
                        if let Ok((_d, ty_raw, raw)) = g.read_tensor_raw(name) {
                            let wb = newbuf(&raw);
                            if let Some((qb, sb)) = requant_q8(&wb, 0, k, n, ty_raw) {
                                w8.insert(name.clone(), qb);
                                scale8.insert(name.clone(), sb);
                                continue;
                            }
                        }
                    }
                }
            }
            let (dims, ty, bytes) = g.read_tensor(name)?;
            // Small SSM params (conv1d kernel, a/dt/norm) are used elementwise, not as
            // gemv weights — never quantize them; keep as f16 (ty=1) or f32 (ty=0).
            if name.contains("ssm_conv1d") {
                if ty == 1 { w16.insert(name.clone(), newbuf(&bytes)); } else { w32.insert(name.clone(), newbuf(&bytes)); }
                continue;
            }
            // The vision tower (the mmproj's `v.*`/`mm.*`) stays in its f16 at every
            // precision: it runs once per image, so Q8 saves nothing that matters, and
            // requantized it drifts from the reference (surya-2 projector cosine
            // 0.998657 against 0.999999 in f16; see `vision_weights_f16`).
            if ty == 1 && ojas_formats::mmproj::keep_tensor(name) {
                w16.insert(name.clone(), newbuf(&bytes));
                continue;
            }
            if ty == 1 {
                // rows = product of dims[1..]: 2D [K,N] → N; 3D expert tensors
                // [K, ffn_exp, n_expert] → ffn_exp*n_expert stacked expert blocks.
                let n = dims[1..].iter().product::<u64>() as usize;
                let k = dims[0] as usize; // cols (in)
                if q4mode && k % 32 == 0 {
                    let (nib, sc) = quantize_row_q4(&bytes, n, k);
                    w4.insert(name.clone(), newbuf(&nib));
                    scale4.insert(name.clone(), newbuf(bytemuck_u16(&sc)));
                } else if quant || q4mode {
                    // q8 (or q4 fallback for k not mult of 32)
                    let (q, sc) = quantize_row_i8(&bytes, n, k);
                    w8.insert(name.clone(), newbuf(unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q.len()) }));
                    scale8.insert(name.clone(), newbuf(bytemuck_f32(&sc)));
                } else {
                    w16.insert(name.clone(), newbuf(&bytes));
                }
            } else {
                // Gemma's (1+w) RMSNorm is pre-folded into the GGUF norm weights by
                // the reference conversion (gemma3 and gemma4) — use them as-is.
                w32.insert(name.clone(), newbuf(&bytes));
            }
        }
        }
        tracing::info!(target: "gpu", "uploaded {} tensors (prec={}) in {:.1}s{}", names.len(), prec, _t_upload.elapsed().as_secs_f32(),
            if mmap_bytes > 0 { format!(" ({:.2} GB zero-copy mmap)", mmap_bytes as f64 / 1e9) } else { String::new() });

        // Synthesize the combined attn_kv_b [head][nope|v][kvlora] the qabsorb/ctx kernels
        // expect from GLM's split attn_k_b [head][kvlora][nope] (transpose) + attn_v_b
        // [head][v][kvlora] (direct). One-time at load (~30MB, dequant from disk).
        if let Some(m) = &mla_cfg {
            if split_kvb {
                let nope = (m.k_mla - m.qk_rope) as usize;
                let vmla = m.v_mla as usize;
                let kvlora = m.kv_lora as usize;
                let kvpair = nope + vmla;
                let nh = n_head;
                for l in 0..n_layers {
                    let (_kd, _kt, kb) = g.read_tensor(&format!("blk.{l}.attn_k_b.weight"))?; // f16 [h][kvlora][nope]
                    let (_vd, _vt, vb) = g.read_tensor(&format!("blk.{l}.attn_v_b.weight"))?; // f16 [h][vmla][kvlora]
                    // Vec<u8> is 1-byte aligned; reading it as *const u16 is UB. Load each
                    // f16 with from_le_bytes (native LE), same byte semantics, no alignment req.
                    let ru16 = |b: &[u8], i: usize| u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]);
                    let mut comb = vec![0u16; nh * kvpair * kvlora];
                    for h in 0..nh {
                        for dd in 0..nope {            // k_b: kvb[h][dd][k] = kb[h][k][dd]
                            for k in 0..kvlora {
                                comb[h * kvpair * kvlora + dd * kvlora + k] = ru16(&kb, h * kvlora * nope + k * nope + dd);
                            }
                        }
                        for i in 0..vmla {             // v_b: kvb[h][nope+i][k] = vb[h][i][k]
                            for k in 0..kvlora {
                                comb[h * kvpair * kvlora + (nope + i) * kvlora + k] = ru16(&vb, h * vmla * kvlora + i * kvlora + k);
                            }
                        }
                    }
                    let bytes = unsafe { std::slice::from_raw_parts(comb.as_ptr() as *const u8, comb.len() * 2) };
                    w16.insert(format!("blk.{l}.attn_kv_b.weight"), newbuf(bytes));
                }
                tracing::info!(target: "mla", "synthesized combined attn_kv_b from split k_b/v_b for {n_layers} layers");
            }
        }

        // ---- Fold the two patch-embed convolutions into one GEMM weight ----------
        // `temporal_patch_size = 2` means the reference feeds the same pixels to
        // `patch_embeddings_0` and `patch_embeddings_1` and adds the results for a still
        // image (`clip_graph_qwen2vl::build_inp_with_temporal_merge`, qwen2vl.cpp:3), so
        // `conv(w0,x) + conv(w1,x) == conv(w0+w1, x)` and the second convolution is waste.
        // Folding at load rather than per encode also gives the fold its own (K, N)
        // `wshape` entry: the on-disk 4-D `[16,16,3,768]` flattens to (16, 36864), the
        // right record for that tensor and the wrong shape for this GEMM.
        //
        // The sum is taken in f32 (two f16s do not add to an f16) and then stored in the
        // same tier every other vision weight landed in, so the CPU oracle and the GPU
        // round in the same place — `ojas_cpu::cpu_vit::fold_patch_weights` makes the same
        // choice.
        if let Some(v) = &vision {
            let k = (v.channels * v.patch * v.patch) as usize;
            let (_d0, t0, b0) = g.read_tensor("v.patch_embd.weight")?;
            let (_d1, t1, b1) = g.read_tensor("v.patch_embd.weight.1")?;
            anyhow::ensure!(t0 == 1 && t1 == 1 && b0.len() == b1.len(),
                "v.patch_embd.weight/.1 are not a matching f16 pair (types {t0}/{t1}, {} vs {} bytes)",
                b0.len(), b1.len());
            anyhow::ensure!(b0.len() == 2 * k * v.d as usize,
                "v.patch_embd.weight is {} bytes, expected {} ({k} x {})", b0.len(), 2 * k * v.d as usize, v.d);
            let rd = |b: &[u8], i: usize| half::f16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32();
            let mut folded = vec![0u8; b0.len()];
            for i in 0..k * v.d as usize {
                let bits = half::f16::from_f32(rd(&b0, i) + rd(&b1, i)).to_bits().to_le_bytes();
                folded[2 * i] = bits[0];
                folded[2 * i + 1] = bits[1];
            }
            let n = v.d as usize;
            // f16, as the upload loop keeps every other tower weight.
            w16.insert(VIT_PATCH_FOLD.into(), newbuf(&folded));
            wshape.insert(VIT_PATCH_FOLD.into(), (k as u32, v.d));
            tracing::info!(target: "vision", "folded v.patch_embd.weight + .weight.1 -> {VIT_PATCH_FOLD} [{k} x {n}]");
        }

        // Pin the resident skeleton (attention/router/shared-expert weights, ~19GB) in RAM.
        // In stream mode w16/w32/w8 hold only the skeleton (experts are gathered on demand),
        // and the attention GEMVs re-read ~17GB of it per token — if macOS compresses or
        // evicts it under the pread churn, the route phase runs at disk speed (5–13s).
        // mlock keeps it hot so attention stays RAM-bound (~2s). OJAS_NO_SKEL_LOCK disables.
        if stream && !ecfg.no_skel_lock {
            let mut locked = 0u64;
            for b in w16.values().chain(w32.values()).chain(w8.values()) {
                unsafe { if libc::mlock(b.contents(), b.length() as usize) == 0 { locked += b.length(); } }
            }
            tracing::info!(target: "stream", "mlock'd {:.1} GB resident skeleton (OJAS_NO_SKEL_LOCK disables)", locked as f64 / 1e9);
        }

        // No-bias archs (Llama/Qwen3/Gemma): the qkv kernels always add a bias, so
        // provide zero bias buffers → same kernels work, bias contributes nothing.
        if !qkv_bias {
            let zq = vec![0u8; (n_head * hd) * 4];
            let zkv = vec![0u8; kvdim * 4];
            for i in 0..n_layers {
                w32.insert(format!("blk.{i}.attn_q.bias"), newbuf(&zq));
                w32.insert(format!("blk.{i}.attn_k.bias"), newbuf(&zkv));
                w32.insert(format!("blk.{i}.attn_v.bias"), newbuf(&zkv));
            }
        }

        // ---- Build the per-layer plan (reusable, data-driven forward). Uniform archs
        // get N identical plans; gemma3 gets per-layer dual RoPE base. gemma-4's full
        // per-layer variation (mixed geometry + KV sharing) is layered on in build_layers. ----
        let rope_local = if arch == "gemma3" { 10000.0 } else { 0.0 };
        let swa_pattern = if arch == "gemma3" { 6u32 } else { 0 };
        let mut layers = Self::build_layers(g, &arch, n_layers, n_head, n_kv, hd, rope_base, rope_local, swa_pattern);
        // NextN/MTP: append the draft block's LayerPlan (index n_layers, which the main
        // forward loop does not iterate) before the KV caches are sized from `layers`, so
        // the draft block gets its own cache.
        let mtp: Option<MtpConfig> = if n_nextn > 0 && g.tensors.contains_key(&format!("blk.{n_layers}.nextn.eh_proj.weight")) {
            let has = |t: &str| g.tensors.contains_key(&format!("blk.{n_layers}.nextn.{t}.weight"));
            let cfg = MtpConfig {
                layer: n_layers,
                has_head: has("shared_head_head"),
                has_head_norm: has("shared_head_norm"),
                has_embed: has("embed_tokens"),
                hnorm_len: g.tensors.get(&format!("blk.{n_layers}.nextn.hnorm.weight"))
                    .map(|i| i.dims.iter().product::<u64>() as usize).unwrap_or(0),
            };
            layers.push(LayerPlan {
                n_head: n_head as u32, n_kv: n_kv as u32, head_dim: hd as u32,
                qdim: (n_head * hd) as u32, kvdim: (n_kv * hd) as u32,
                rope_base, scale: 1.0 / (hd as f32).sqrt(),
                kv_source: n_layers, has_v: true, is_ssm: false,
            });
            tracing::info!(target: "arch", "NextN/MTP draft block at blk.{n_layers} (head={} head_norm={} embed={} hnorm={})",
                cfg.has_head, cfg.has_head_norm, cfg.has_embed, cfg.hnorm_len);
            Some(cfg)
        } else { None };
        let max_qdim = layers.iter().map(|p| p.qdim as usize).max().unwrap_or(n_head * hd);
        let max_kvdim = layers.iter().map(|p| p.kvdim as usize).max().unwrap_or(kvdim);
        // KV-shared layers (gemma-4 global) have no attn_v.weight — the fused qkv kernel
        // still expects one, so give them a zero V projection (its output lands in an
        // unused cache slot; attention reads vcache[kv_source] instead). These layers' q/k/v
        // biases are zeroed too; no-bias archs are already handled above.
        for (l, lp) in layers.iter().enumerate() {
            if lp.has_v { continue; }
            let (kv, din) = (lp.kvdim as usize, d);
            let name = format!("blk.{l}.attn_v.weight");
            if q4mode && din % 32 == 0 {
                w4.insert(name.clone(), newbuf(&vec![0x88u8; kv * (din / 2)]));       // nib=8 → w=0
                scale4.insert(name.clone(), newbuf(bytemuck_u16(&vec![0u16; kv * (din / 32)])));
            } else if quant || q4mode {
                w8.insert(name.clone(), newbuf(&vec![0u8; kv * din]));
                scale8.insert(name.clone(), newbuf(bytemuck_f32(&vec![0f32; kv])));
            } else {
                w16.insert(name.clone(), newbuf(&vec![0u8; kv * din * 2]));
            }
        }

        // Gemma4: all-1.0 weight for weightless V RMSNorm (sized to the max head_dim).
        let max_hd = layers.iter().map(|p| p.head_dim as usize).max().unwrap_or(hd);
        let ones = newbuf(bytemuck_f32(&vec![1.0f32; max_hd]));
        // Gemma4: per-layer output scalar (layer_output_scale.weight, a [1] f32).
        let out_scale: Vec<f32> = if g.tensors.contains_key("blk.0.layer_output_scale.weight") {
            (0..n_layers).map(|l| {
                g.read_tensor(&format!("blk.{l}.layer_output_scale.weight")).ok()
                    .map(|(_, _, b)| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).unwrap_or(1.0)
            }).collect()
        } else { Vec::new() };
        if !out_scale.is_empty() {
            tracing::info!(target: "arch", "gemma4 layer_output_scale: first3={:?} min={:.4} max={:.4}",
                &out_scale[..3.min(out_scale.len())],
                out_scale.iter().cloned().fold(f32::MAX, f32::min),
                out_scale.iter().cloned().fold(f32::MIN, f32::max));
        }

        // zero-init: the MTP draft block legitimately attends positions its cache
        // never saw (prompt span, accepted-token holes) — zero keys are benign noise,
        // uninitialized memory is not.
        // ---- adaptive context (8 GB Air to 192 GB Ultra) ----
        // max_seq == 0 → auto: the model's trained limit, capped by a RAM budget derived
        // from hw.memsize minus the estimated resident weights. An explicit ctx obeys the
        // same cap, because the KV cache is force-committed at alloc and a "max" pick must
        // not commit past what RAM can hold. OJAS_CTX overrides.
        //
        // Absorbed MLA is the default (compressed latent cache, tiny KV); OJAS_MLA_NAIVE
        // forces the full-K/V streaming path. Decided here so the KV budget below sizes the
        // right cache and absorb context can be capped at the kernel's SC_CAP limit.
        if let Some(m) = mla_cfg.as_mut() { m.absorb = !ecfg.mla_naive; }
        {
            let model_ctx = g.meta_u32(&format!("{arch}.context_length")).unwrap_or(32768) as usize;
            // only attention layers hold real KV (SSM layers' caches are never
            // written — lazy-commit keeps them virtual)
            // MLA absorbed: KV is the compressed latent (kv_lora+rope) f16/token, not the full K/V.
            let kv_per_tok: usize = if let Some(m) = &mla_cfg {
                if !m.absorb { layers.iter().filter(|p| !p.is_ssm).map(|p| p.kvdim as usize * 2 * 2).sum() }
                else { n_layers * (m.kv_lora + m.qk_rope) as usize * 2 }
            } else { layers.iter().filter(|p| !p.is_ssm).map(|p| p.kvdim as usize * 2 * 2).sum() };
            // `physical_ram_bytes` is the sysctlbyname("hw.memsize") query with a fallback,
            // and is deliberately the only copy of it
            let ram = physical_ram_bytes();
            // resident-weight estimate: MoE experts are mmap'd + streamed (page cache),
            // so only the skeleton + a hot-expert fraction actually stays resident.
            let el = |t: &ojas_formats::gguf::TensorInfo| t.dims.iter().product::<u64>();
            let (exp_e, skel_e): (u64, u64) = g.tensors.iter().fold((0, 0), |(a, b), (n, t)| {
                if n.contains("_exps.") { (a + el(t), b) } else { (a, b + el(t)) }
            });
            // Streamed experts live on disk (pread + page cache) and are not resident, so
            // counting them here collapses the budget to the 2048 floor. In stream mode only
            // the mlock'd skeleton is resident (the expert cache is separate, bounded RAM).
            let resident_exp = if stream { 0 } else { exp_e * 3 / 10 };
            let weights_est = (skel_e + resident_exp) * 9 / 16;
            // The KV cache is buf_zeroed, so it is fully force-committed at alloc, not lazy.
            // The auto-ctx therefore caps by a RAM budget and by an absolute KV size ceiling,
            // so one process cannot eat tens of GB (MLA archs store full reconstructed K/V
            // ≈ 324 KB/token → 163840 ctx would be ~53 GB). OJAS_CTX overrides.
            let budget = ((ram as f64 * 0.4) as u64).saturating_sub(weights_est);
            // Stream mode keeps a big skeleton + expert cache resident, so use a tighter
            // default KV ceiling (RAM headroom) — still ~40k+ tokens for MLA's tiny KV.
            let default_kv_gb = if stream { 4 } else { 10 };
            let kv_ceiling: u64 = kv_gb.or(ecfg.kv_gb).unwrap_or(default_kv_gb) << 30;
            let cap = ((budget.min(kv_ceiling)) as usize / kv_per_tok.max(1)).max(2048);
            if max_seq == 0 {
                max_seq = model_ctx.min(cap);
                if let Some(n) = ecfg.ctx { max_seq = n; }
            } else if max_seq > cap {
                tracing::warn!(target: "ctx", "requested ctx {max_seq} exceeds the RAM budget → clamped to {cap}");
                max_seq = cap;
            }
            // Until learned sparse selection is implemented, only expose the
            // context where all candidate positions fit the indexer's budget.
            if arch == "qwen4exp" {
                let limit = g.meta_u32("qwen4exp.attention.indexer.top_k").unwrap_or(0) as usize;
                if limit > 0 {
                    anyhow::ensure!(!ecfg.ctx.is_some_and(|n| n > limit),
                        "qwen4exp context exceeds {limit}: learned sparse selection is not implemented");
                    if max_seq > limit {
                        tracing::warn!(target: "ctx", "qwen4exp context capped at {limit} until learned sparse selection is available");
                        max_seq = limit;
                    }
                }
            }
            // Absorbed-MLA attention buffers all scores in sc[SC_CAP]; clamp so it's never
            // dispatched past that. Longer MLA context needs OJAS_MLA_NAIVE.
            if mla_cfg.as_ref().map(|m| m.absorb).unwrap_or(false) && max_seq > ojas_metal::kernels::mla::SC_CAP {
                tracing::warn!(target: "ctx", "absorbed-MLA caps ctx at {} (OJAS_MLA_NAIVE=1 for longer)", ojas_metal::kernels::mla::SC_CAP);
                max_seq = ojas_metal::kernels::mla::SC_CAP;
            }
            tracing::info!(target: "ctx", "model max {} | KV {} KB/token, ≤{} GB → using ctx {} (~{:.1} GB KV; OJAS_CTX/OJAS_KV_GB override)",
                model_ctx, kv_per_tok / 1024, kv_ceiling >> 30, max_seq, (max_seq * kv_per_tok) as f64 / 1e9);
        }
        // KV caches are half precision (2 bytes/elem — buf() sizes in f32 units, so halve
        // the count). Metal shared buffers lazy-commit pages: allocating a large context
        // costs real memory only as positions fill.
        // MLA absorbed (default) uses the compressed mla_lat cache, not the full reconstructed
        // K/V — so skip the big kcache/vcache alloc (~53 GB) unless naive is forced.
        let mla_naive = mla_cfg.as_ref().map(|m| !m.absorb).unwrap_or(false);
        // Independent sequence slots (MAX_SLOTS documents the ceiling of 4). Read once here,
        // like every other env knob, because it sizes allocations: a slot count that could
        // change after load would leave the arena addressing state it never reserved.
        //
        // The default of 1 keeps every slot offset at 0, so single-sequence graphs encode
        // byte-identical dispatches, and it keeps memory flat: KV is ~12 KB/token, so a
        // 4-slot 32k context would reserve 4x the largest allocation in the arena. A host
        // that wants the throughput opts in.
        let slots: usize = std::env::var("OJAS_SLOTS").ok().and_then(|v| v.parse().ok())
            .unwrap_or(1).clamp(1, MAX_SLOTS);
        if slots > 1 {
            tracing::info!(target: "ctx", "{slots} sequence slots: KV and recurrent state allocated {slots}x");
        }
        let kv_sz = |p: &LayerPlan| if p.is_ssm || mla_cfg.is_some() && !mla_naive { 1 } else { max_seq * p.kvdim as usize / 2 };
        // Slot-strided: slot s's rows live at `s * kv_sz` (see StateArena::kv_stride).
        // The MLA placeholder length of 1 is deliberately not multiplied — nothing
        // indexes it, and `kv_stride` divides the allocation back out, so a 1-long
        // buffer keeps a stride of 1/slots = 0 and stays unaddressable either way.
        let kcache: Vec<metal::Buffer> = layers.iter().map(|p| buf_zeroed(gpu, kv_sz(p) * slots)).collect();
        let vcache = layers.iter().map(|p| buf_zeroed(gpu, kv_sz(p) * slots)).collect();
        // MLA absorbed latent cache: [Lc(kv_lora) | Rc(qk_rope)] f16/token per layer (lazy, tiny).
        let moe: Option<MoeConfig> = if let Some(m) = &mla_cfg {
            // deepseek2/glm-dsa: size the MoE scratch from the MLA geometry so the
            // shared moe_* buffers are big enough for the inline MLA MoE forward.
            Some(MoeConfig { n_expert: m.n_expert, n_used: m.n_used, ffn_exp: m.ffn_exp, ffn_shexp: m.ffn_exp * m.n_shared })
        } else if arch == "qwen35moe" || arch == "qwen4exp" {
            let mc = MoeConfig {
                n_expert: g.meta_u32(&format!("{arch}.expert_count")).unwrap_or(256),
                n_used: g.meta_u32(&format!("{arch}.expert_used_count")).unwrap_or(8),
                ffn_exp: g.meta_u32(&format!("{arch}.expert_feed_forward_length")).unwrap_or(512),
                ffn_shexp: g.meta_u32(&format!("{arch}.expert_shared_feed_forward_length")).unwrap_or(512),
            };
            // Adaptive top-k (AdapMoE), as in the MLA branch: fewer experts/token cuts
            // gather bytes, which are what bound decode on a streamed MoE.
            let mut mc = mc;
            if let Some(k) = ecfg.top_k {
                if k > 0 && k < mc.n_used {
                    tracing::info!(target: "arch", "OJAS_TOPK: top-{k} of {} experts (fewer bytes/token)", mc.n_used);
                    mc.n_used = k;
                }
            }
            tracing::info!(target: "arch", "{arch} MoE: {} experts, top-{}, ffn_exp={} ffn_shexp={}",
                mc.n_expert, mc.n_used, mc.ffn_exp, mc.ffn_shexp);
            Some(mc)
        } else if arch == "gpt-oss" {
            let mut mc = MoeConfig {
                n_expert: g.meta_u32("gpt-oss.expert_count").unwrap_or(32),
                n_used: g.meta_u32("gpt-oss.expert_used_count").unwrap_or(4),
                ffn_exp: g.meta_u32("gpt-oss.expert_feed_forward_length").unwrap_or(2880),
                ffn_shexp: 0, // no shared expert
            };
            if let Some(k) = ecfg.top_k { if k > 0 && k < mc.n_used { mc.n_used = k; } }
            tracing::info!(target: "arch", "gpt-oss MoE: {} experts, top-{}, ffn_exp={} (biased router+experts, SwiGLU-OAI, attn-sinks)",
                mc.n_expert, mc.n_used, mc.ffn_exp);
            Some(mc)
        } else { None };

        // gather scratch: sized to n_used experts × the max per-expert byte stride seen in
        // stream_meta (gate/up are Q4_K; down may be Q5_K/Q6_K/Q8_0 — take the max).
        // Sized from the generic MoE config, not from mla_cfg: falling back to the DeepSeek
        // defaults (256 experts, top-8) sizes this scratch for 8 experts on a 512-expert
        // top-10 model while the gather writes 10, silently overflowing a shared GPU buffer.
        let ne_u = moe.as_ref().map(|m| m.n_expert as u64).unwrap_or(256);
        let nu_u = moe.as_ref().map(|m| m.n_used as usize).unwrap_or(8);
        let (mut max_gu, mut max_dn) = (0u64, 0u64);
        for (nm, &(_, _, rawlen, _)) in &stream_meta {
            let per = rawlen / ne_u.max(1);
            if nm.contains("down") { max_dn = max_dn.max(per); } else { max_gu = max_gu.max(per); }
        }
        let raw_buf = |bytes: u64| gpu.device.new_buffer(bytes.max(64), MTLResourceOptions::StorageModeShared);
        // Room for `ubatch` tokens' worth of routed experts, deduped: M tokens pick
        // at most M*n_used distinct experts, and never more than n_expert.
        let ubatch = ecfg.ubatch;
        let gather_cap = (nu_u * ubatch).min(ne_u as usize);
        let moe_gs = raw_buf(max_gu * gather_cap as u64);
        let moe_us = raw_buf(max_gu * gather_cap as u64);
        let moe_ds = raw_buf(max_dn * gather_cap as u64);
        let moe_slot = raw_buf((nu_u * ubatch * 4) as u64);
        if !stream_meta.is_empty() {
            tracing::info!(target: "stream", "expert gather scratch: {} experts ({:.2} GB) for --ubatch-size {ubatch}",
                gather_cap, (gather_cap as f64 * (2.0 * max_gu as f64 + max_dn as f64)) / (1u64 << 30) as f64);
        }
        // GiB, matching kv_gb and the UI's memory-split math (1024³, not decimal
        // GB) — a "10 GiB" slider must allocate 10 GiB, not 9.31.
        let cache_bytes =
            (expert_cache_gb.unwrap_or(ecfg.expert_cache_gb) * (1u64 << 30) as f64) as usize;
        let mut expert_cache = ExpertCache::new(cache_bytes);
        if ecfg.flash_direct_experts && is_qwen4exp {
            expert_cache.metal_device = Some(gpu.device.clone());
        }
        if ecfg.flash_expert_pool {
            anyhow::ensure!(ecfg.flash_direct_experts && is_qwen4exp && !stream_meta.is_empty(),
                "OJAS_FLASH_EXPERT_POOL requires streamed Flash and OJAS_FLASH_DIRECT_EXPERTS=1");
            anyhow::ensure!(cache_bytes > 0 && cache_bytes as u64 <= gpu.device.max_buffer_length(),
                "expert pool budget must be nonzero and fit maxBufferLength");
            // A provisional exclusion, not a measured safety boundary.
            //
            // Measured on an M2 Max (96 GiB, 77 GB recommended working set) against the
            // 90 GB Flash checkpoint: a 32 GiB pool reproduces llama.cpp's output exactly,
            // while a 48 GiB pool intermittently produces different tokens for the same
            // prompt. The cause is not established — the expert bytes reaching the GPU and
            // the routing decisions were both verified correct in failing runs, and the
            // failure tracks neither free memory nor load. Half the working set is simply a
            // threshold below the configuration known to misbehave; the real boundary may
            // be lower.
            let working_set = gpu.device.recommended_max_working_set_size();
            let ceiling = (working_set / 2) as usize;
            // There is deliberately no override. Forcing a larger pool repeatedly exhausted
            // wired memory and hung the host (MTLCommandBufferError 8, OutOfMemory), so the
            // only way past this bound is to change the bound with a reason.
            anyhow::ensure!(cache_bytes <= ceiling,
                "expert pool budget {:.1} GB exceeds half this device's recommended working set \
                 ({:.1} GB of {:.1} GB). Pools this large have been measured producing INCORRECT \
                 OUTPUT on this model; the cause is not yet understood, so the size is excluded \
                 rather than explained. Lower OJAS_EXPERT_CACHE_GB (32 is the tested setting for \
                 Flash on 96 GiB).",
                cache_bytes as f64 / 1e9, ceiling as f64 / 1e9, working_set as f64 / 1e9);
            expert_cache.pool = Some(expert_pool::ExpertPool::new(&gpu.device, cache_bytes));
        }
        let expert_cache = std::cell::RefCell::new(expert_cache);
        let mla_lat: Vec<metal::Buffer> = if let Some(m) = &mla_cfg {
            let hdk = (m.kv_lora + m.qk_rope) as usize;
            (0..n_layers).map(|_| buf(gpu, (max_seq * hdk + 1) / 2)).collect()   // half elems → /2 f32 slots
        } else { Vec::new() };

        // Quest-style page-sparse decode metadata (only when OJAS_SPARSE=<tokens> is
        // set — approximate method, cross-validate before trusting new models).
        // OJAS_SPARSE=<dense-token-budget>; 0 (or unset) disables page-sparse decode.
        let sparse_budget: Option<u32> = ecfg.sparse;
        let pages_max = (max_seq + ojas_metal::kernels::attn::PAGE - 1) / ojas_metal::kernels::attn::PAGE;
        let pmeta: Vec<metal::Buffer> = layers.iter().map(|p| {
            if p.is_ssm || sparse_budget.is_none() { buf(gpu, 1) }
            else { buf_zeroed(gpu, pages_max * p.kvdim as usize) }   // 2×kvdim half = kvdim f32
        }).collect();
        let plist = buf(gpu, n_head as usize * ojas_metal::kernels::attn::MAXSEL);

        // ---- ViT tower scratch sizing -----------------------------------------
        // Everything is `1` float without an mmproj, so a text model pays nothing.
        // Row counts are padded to 32 because `gemm_mm_f16` stores whole 32-token
        // tiles; `vx` additionally has to survive being reinterpreted as
        // `[rows/4, 4*d_v]` by the projector GEMM, whose padding rounds a quarter of the
        // rows up to 32 and can therefore exceed `pad32(patches)` at small grids (36
        // patches: 9 projector rows pad to 32, i.e. 128 tower rows against pad32(36) = 64)
        // — hence the max. Metal shared buffers lazy-commit, so a small image never faults
        // in the pages a full page would use.
        let vpad = |r: u32| ((r + 31) / 32 * 32) as usize;
        let vsz = |v: &VisionConfig| {
            let p = v.max_patches;
            let rows = vpad(p).max(4 * vpad(p / 4));
            let (dv, k) = (v.d as usize, (v.channels * v.patch * v.patch) as usize);
            let mmrows = vpad(p / 4);
            (
                (v.channels * v.patch * v.patch) as usize * p as usize, // vimg: pixels of `p` patches
                rows * k,                                              // vrows
                rows * dv,                                             // vx / vh / vq
                rows * 3 * dv,                                         // vqkv
                ((rows + 32) * dv + 1) / 2,                            // vkh / vvh: half + 32 pad rows
                (rows * v.ffn as usize).max(mmrows * v.mm_hidden as usize), // vffn, also mm.0's output
                p as usize * dv,                                       // vpe
                4 + 4 * p as usize,                                    // vmpos
                mmrows * v.proj_dim as usize,                          // vout
            )
        };
        let (n_vimg, n_vrows, n_vd, n_vqkv, n_vhalf, n_vffn, n_vpe, n_vmpos, n_vout) =
            vision.as_ref().map(vsz).unwrap_or((1, 1, 1, 1, 1, 1, 1, 1, 1));

        // qwen35 Gated-DeltaNet SSM config + per-layer recurrent state caches.
        let ssm: Option<SsmConfig> = if arch == "qwen35" || arch == "qwen35moe" || arch == "qwen4exp" {
            let cfg = SsmConfig {
                d_state: g.meta_u32(&format!("{arch}.ssm.state_size")).unwrap_or(128),
                n_group: g.meta_u32(&format!("{arch}.ssm.group_count")).unwrap_or(16),
                dt_rank: g.meta_u32(&format!("{arch}.ssm.time_step_rank")).unwrap_or(32),
                d_inner: g.meta_u32(&format!("{arch}.ssm.inner_size")).unwrap_or(d as u32),
                conv_kernel: g.meta_u32(&format!("{arch}.ssm.conv_kernel")).unwrap_or(4),
                attn_interval: g.meta_u32(&format!("{arch}.full_attention_interval")).unwrap_or(4),
                n_rot: g.meta_u32(&format!("{arch}.rope.dimension_count")).unwrap_or(64),
                // M-RoPE section sizes in cos/sin pairs, summing to n_rot/2 (surya-2: [11,11,10,0]).
                // Absent => all zero => plain partial NEOX rope, which is what every text-only
                // position stream degenerates to.
                mrope_sections: {
                    let mut s = [0u32; 4];
                    if let Some(a) = g.int_arr(&format!("{arch}.rope.dimension_sections")) {
                        for (i, v) in a.iter().take(4).enumerate() { s[i] = (*v).max(0) as u32; }
                    }
                    s
                },
            };
            tracing::info!(target: "arch", "qwen35 SSM: state={} n_group={} dt_rank={} d_inner={} conv_k={} attn_every={} \
                       | {} SSM + {} attn layers", cfg.d_state, cfg.n_group, cfg.dt_rank, cfg.d_inner,
                cfg.conv_kernel, cfg.attn_interval, layers.iter().filter(|p| p.is_ssm).count(),
                layers.iter().filter(|p| !p.is_ssm).count());
            Some(cfg)
        } else { None };
        // PLE history shares its recurrent layer's state allocation. Existing
        // reset, prefix snapshots and persistence then carry it atomically.
        let ple_layers: Vec<u32> = if arch == "qwen4exp" {
            g.int_arr("qwen4exp.ple.layers").map(|a| a.iter().map(|&v| v as u32).collect()).unwrap_or_default()
        } else { Vec::new() };
        let ple_history = if ple_layers.is_empty() { 0 } else {
            let kernel = g.meta_u32("qwen4exp.ple.conv_kernel").unwrap_or(0) as usize;
            let dilation = g.meta_u32("qwen4exp.ple.ngram_size").unwrap_or(0) as usize;
            let hc = g.meta_u32("qwen4exp.hyper_connection.count").unwrap_or(0) as usize;
            anyhow::ensure!(kernel > 0 && dilation > 0 && hc > 0, "invalid PLE convolution geometry");
            for &l in &ple_layers {
                anyhow::ensure!(layers.get(l as usize).is_some_and(|p| p.is_ssm), "PLE requires a recurrent layer, got {l}");
            }
            (kernel - 1).checked_mul(dilation).and_then(|n| n.checked_mul(hc))
                .and_then(|n| n.checked_mul(d)).ok_or_else(|| anyhow::anyhow!("PLE history size overflow"))?
        };
        // conv_state: [(conv_k-1) × conv_channels], ssm_state: [S × S × H_v], per SSM layer.
        let (conv_state, ssm_state): (Vec<metal::Buffer>, Vec<metal::Buffer>) = if let Some(c) = ssm {
            let conv_ch = c.d_inner + 2 * c.n_group * c.d_state;
            let conv_sz = ((c.conv_kernel - 1) * conv_ch) as usize;
            let state_sz = (c.d_state * c.d_state * c.dt_rank) as usize;
            // x slots, contiguous per slot — the recurrent half of a sequence's state.
            // 18 SSM layers x 1 MiB ssm_state + 1.3 MiB conv_state is ~19 MB per slot,
            // which is small against the ~355 MB of weights the slots share.
            layers.iter().enumerate().map(|(l, p)| if p.is_ssm { (buf_zeroed(gpu, slots * (conv_sz + if ple_layers.contains(&(l as u32)) { ple_history } else { 0 })), buf_zeroed(gpu, slots * state_sz)) }
                                  else { (buf(gpu, 1), buf(gpu, 1)) }).unzip()
        } else { (Vec::new(), Vec::new()) };
        // MTP speculative rollback snapshots (only when the GGUF ships a draft block)
        let snapshot_rows = if ecfg.mtp_prefix && arch == "qwen4exp" {
            ecfg.mtp_draft.saturating_add(1).min(ubatch).min(MAXM).max(2)
        } else { 1 };
        let (conv_snap, ssm_snap): (Vec<metal::Buffer>, Vec<metal::Buffer>) = if let (Some(c), true) = (ssm, n_nextn > 0) {
            let conv_ch = c.d_inner + 2 * c.n_group * c.d_state;
            let conv_sz = ((c.conv_kernel - 1) * conv_ch) as usize;
            let state_sz = (c.d_state * c.d_state * c.dt_rank) as usize;
            layers.iter().enumerate().map(|(l, p)| if p.is_ssm { (buf(gpu, snapshot_rows * (conv_sz + if ple_layers.contains(&(l as u32)) { ple_history } else { 0 })), buf(gpu, snapshot_rows * state_sz)) }
                                  else { (buf(gpu, 1), buf(gpu, 1)) }).unzip()
        } else { (Vec::new(), Vec::new()) };


        let qwen4exp: Option<Qwen4ExpConfig> = if arch == "qwen4exp" {
            let u = |k: &str| g.meta_u32(&format!("{arch}.{k}"));
            let ple_layers: Vec<u32> = g.int_arr(&format!("{arch}.ple.layers"))
                .map(|a| a.iter().map(|&v| v as u32).collect()).unwrap_or_default();
            let ug = |k: &str| g.int_arr(&format!("{arch}.{k}"));
            let ngram = u("ple.ngram_size").unwrap_or(0);
            let hpg = u("ple.heads_per_ngram").unwrap_or(0);
            let cfg = Qwen4ExpConfig {
                hc_mult: u("hyper_connection.count").unwrap_or(1),
                hc_low_rank: u("hyper_connection.low_rank").unwrap_or(0),
                indexer: IndexerConfig::from_gguf(g, &arch),
                ple_ngram_size: ngram,
                ple_heads_per_ngram: hpg,
                ple_conv_kernel: u("ple.conv_kernel").unwrap_or(0),
                // The `_input` suffix matters: the reference constant is named
                // LLM_KV_EMBEDDING_LENGTH_PER_LAYER but the key it maps to is
                // "%s.embedding_length_per_layer_input". Reading the enum's name instead
                // silently yields 0, and a zero head dim makes ple_gather write nothing,
                // leaving the n-gram path to read stale scratch.
                ple_head_dim: u("embedding_length_per_layer_input").unwrap_or(0),
                ple_layers,
                ple_n_heads: ngram.saturating_sub(1) * hpg,
                ple_eos: u("ple.eos_token_id").unwrap_or(0),
                ple_multipliers: ug("ple.layer_multipliers").map(|a| a.iter().map(|&v| v as u64).collect()).unwrap_or_default(),
                ple_head_offsets: ug("ple.head_offsets").map(|a| a.iter().map(|&v| v as u32).collect()).unwrap_or_default(),
                ple_head_vocab_sizes: ug("ple.head_vocab_sizes").map(|a| a.iter().map(|&v| v as u32).collect()).unwrap_or_default(),
            };
            // The gathered n-gram vector must be exactly d wide (n_heads slices of
            // head_dim). Checking it here turns a silently-zero metadata key into a
            // load error naming that key.
            if !cfg.ple_layers.is_empty() {
                let got = (cfg.ple_head_dim * cfg.ple_n_heads) as usize;
                if cfg.ple_head_dim == 0 || got != d {
                    anyhow::bail!("qwen4exp PLE geometry is wrong: head_dim={} x n_heads={} = {}, expected d={} \
                        (check {arch}.embedding_length_per_layer_input / ple.ngram_size / ple.heads_per_ngram)",
                        cfg.ple_head_dim, cfg.ple_n_heads, got, d);
                }
            }
            tracing::info!(target: "arch", "qwen4exp: hc_mult={} hc_low_rank={} indexer(top_k={}) ple_layers={} ple_head_dim={} ple_n_heads={}",
                cfg.hc_mult, cfg.hc_low_rank, cfg.indexer.top_k, cfg.ple_layers.len(), cfg.ple_head_dim, cfg.ple_n_heads);
            Some(cfg)
        } else { None };

        let hc_mult = qwen4exp.as_ref().map(|q| q.hc_mult as usize).unwrap_or(1);
        let hc_lr = qwen4exp.as_ref().map(|q| q.hc_low_rank as usize).unwrap_or(1);
        let ple_nh = qwen4exp.as_ref().map(|q| q.ple_n_heads as usize).unwrap_or(1);
        let ple_stage_bytes = if let Some(table) = &ple {
            let n_layers = qwen4exp.as_ref().unwrap().ple_layers.len();
            let bytes = table.row_bytes.checked_mul(ple_nh)
                .and_then(|b| b.checked_add(3)).map(|b| b / 4 * 4)
                .and_then(|b| b.checked_mul(MAXM))
                .and_then(|b| b.checked_mul(n_layers)).ok_or_else(|| anyhow::anyhow!("PLE staging size overflow"))?;
            tracing::info!(target: "stream", "PLE: {:.2} GB CPU-only table, {} bytes/token, {} bytes GPU staging ({} rows/layer)",
                table.rows as f64 * table.row_bytes as f64 / 1e9, table.row_bytes * ple_nh, bytes, MAXM);
            bytes
        } else { 4 };
        let ple_indices: Vec<u32> = (0..ple_nh.max(1) as u32).collect();
        let ple_idx = newbuf(unsafe { std::slice::from_raw_parts(ple_indices.as_ptr() as *const u8, ple_indices.len() * 4) });
        let max_tg = gpu.device.max_threads_per_threadgroup().width;
        let mut model = Self {
            flash_trace: std::cell::RefCell::new(None),
            cfg: ecfg.clone(),
            sess: Session {
                model_name: std::path::Path::new(&g.path).file_stem()
                    .map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                session_tokens: std::cell::RefCell::new(Vec::new()),
                last_prefill_reused: std::cell::Cell::new(0),
                snap_pos: std::cell::RefCell::new(Vec::new()),
                snap_buf: std::cell::RefCell::new(Vec::new()),
            },
            gpu, d,
            arch: Arch {
                n_layers, n_head, n_kv, hd, ffn, layers, vocab, rope_base, eps, lm_head: lm_head_name, qkv_bias, qk_norm,
                rope_neox: arch != "llama", embed_scale, sandwich, gelu: is_gemma,
                rope_local, swa_pattern,
                v_rmsnorm: arch == "gemma4", gpt_oss: arch == "gpt-oss", out_scale, ssm, moe, mla: mla_cfg, qwen4exp,
                vision,
                text_encoder,
                sparse_budget,
            },
            ms: MoeScratch {
                moe_lg: buf(gpu, moe.map(|m| m.n_expert as usize).unwrap_or(1)),
                route_lg: if ojas_core::config::var("OJAS_ROUTE_STATS").is_ok() {
                    Some(buf(gpu, n_layers * moe.map(|m| m.n_expert as usize).unwrap_or(1)))
                } else { None },
                // Routing ids, laid out per layer as [n_tokens][n_used] — the shape
                // llama.cpp's mul_mat_id takes ([n_expert_used, n_tokens]). The stride is
                // MAXM, the most rows a batched forward can present (not 2, which only
                // covers MTP verify).
                moe_idx: buf(gpu, moe.map(|m| m.n_used as usize * MAXM * (n_layers + 1)).unwrap_or(1)),
                moe_wgt: buf(gpu, moe.map(|m| m.n_used as usize).unwrap_or(1)),
                moe_act: buf(gpu, moe.map(|m| (m.n_used * m.ffn_exp) as usize).unwrap_or(1)),
                moe_sh: buf(gpu, 1),
                moe_blg: buf(gpu, moe.map(|m| MAXM * m.n_expert as usize).unwrap_or(1)),
                moe_bidx: buf(gpu, moe.map(|m| MAXM * m.n_used as usize).unwrap_or(1)),
                moe_bwgt: buf(gpu, moe.map(|m| MAXM * m.n_used as usize).unwrap_or(1)),
                moe_bact: buf(gpu, moe.map(|m| MAXM * (m.n_used * m.ffn_exp) as usize).unwrap_or(1)),
                moe_bsh: buf(gpu, MAXM),
                moe_bg: buf(gpu, moe.map(|m| MAXM * m.ffn_shexp as usize).unwrap_or(1)),
                moe_bu: buf(gpu, moe.map(|m| MAXM * m.ffn_shexp as usize).unwrap_or(1)),
                moe_btmp: buf(gpu, if moe.is_some() { MAXM * d } else { 1 }),
            },
            sp: SpecState {
                mtp,
                // Two verify rows of the full residual: nextn.hnorm is [d*hc], so the
                // combiner's hidden input is all hc streams, not a collapse of them.
                mtp_h: buf(gpu, if mtp.is_some() { MAXM * d * hc_mult.max(1) as usize } else { 1 }),
                // Row i is the hidden the draft needs at batch row i, i.e. the target's
                // hidden one position earlier; row MAXM carries the last one to the
                // next batch, which is what llama.cpp calls pending_h.
                mtp_hprev: buf(gpu, if mtp.is_some() { (MAXM + 1) * d * hc_mult.max(1) as usize } else { 1 }),
                mtp_cat: buf(gpu, if mtp.is_some() { MAXM * 2 * d * hc_mult.max(1) } else { 1 }),
                mtp_chain: buf(gpu, if mtp.is_some() { d * hc_mult.max(1) as usize } else { 1 }),
                mtp_tok: buf(gpu, 1),
                hrow: std::cell::Cell::new(0),
                ssm_snap, conv_snap,
                snapshot_rows, verified_rows: std::cell::Cell::new(0),
            },
            p,
            wt: Weights {
                q8: quant, gemv_r4: ecfg.gemv_r4, w16, w32, w8, scale8,
                q4: q4mode, q4k, q20, w_off, w_qtype, wshape, wq, mapped: mapped_gguf.or(native_map), ple, w4, scale4, w4k, w6k, w4l, q4l_a, q4l_b, w20, s20,
                wcache: None,
            },
            strm: MoeStream {
                stream, resident_layers: 0,
                stream_meta, stream_bufs: std::cell::RefCell::new(HashMap::new()),
                moe_gs, moe_us, moe_ds, moe_slot, gather_cap, ubatch, expert_cache,
                direct_tables: std::array::from_fn(|_| raw_buf(gather_cap as u64 * 8)),
                direct_live: std::cell::RefCell::new(Vec::new()),
            expert_hash_records: std::cell::RefCell::new(Vec::new()),
                expert_residency: None,
                direct_layer: std::cell::Cell::new(None),
                gather_hits: std::cell::Cell::new(0), gather_reads: std::cell::Cell::new(0),
                expert_stats: std::cell::RefCell::new(std::collections::HashMap::new()),
                prefetch: None, stream_prefetch: None,
            },
            // scratch sized for up to MAXM tokens (batched speculative forward)
            st: StateArena {
                x: buf(gpu, d * MAXM), h: buf(gpu, d * MAXM), q: buf(gpu, max_qdim * MAXM), k: buf(gpu, max_kvdim * MAXM),
                v: buf(gpu, max_kvdim * MAXM), attn: buf(gpu, max_qdim.max(d) * MAXM), tmp: buf(gpu, d.max(vocab)), tokbuf: buf(gpu, MAXM), xh: buf(gpu, MAXM * ffn.max(d) / 2 + 1), qh: buf(gpu, MAXM * max_qdim / 2 + 1), skbuf: buf(gpu, 8 * MAXM * d), mpos: buf(gpu, 4 + 4 * MAXM),
                gate: buf(gpu, ffn * MAXM), up: buf(gpu, ffn * MAXM), act: buf(gpu, ffn * MAXM), ones,
                logits: buf(gpu, vocab.max(1) * MAXM), kcache, vcache, mla_lat, conv_state, ssm_state,
                // sized for both uses: SSM conv channels and the attention layers' joint
                // Q+gate projection (2*qdim) — geometries differ across qwen35 models.
                ssm_qkv: buf(gpu, ssm.map(|c| ((c.d_inner + 2*c.n_group*c.d_state) as usize).max(2*max_qdim) * MAXM).unwrap_or(1)),
                ssm_z: buf(gpu, ssm.map(|c| c.d_inner as usize * MAXM).unwrap_or(1)),
                ssm_beta: buf(gpu, ssm.map(|c| c.dt_rank as usize * MAXM).unwrap_or(1)),
                ssm_gate: buf(gpu, ssm.map(|c| c.dt_rank as usize * MAXM).unwrap_or(1)),
                ssm_o: buf(gpu, ssm.map(|c| c.d_inner as usize * MAXM).unwrap_or(1)),
                // Sized for a batched forward: the hyper-connection scratch is
                // token-major, so a chunk of M needs M copies. ~45 MB at MAXM on this
                // arch, and Metal shared buffers lazy-commit, so decode only ever
                // faults in the first token's pages.
                hc_res: buf(gpu, (d * hc_mult * MAXM).max(1)),
                hc_xn: buf(gpu, (d * hc_mult * MAXM).max(1)),
                hc_lo: buf(gpu, (hc_lr * MAXM).max(1)),
                hc_graw: buf(gpu, (d * hc_mult * MAXM).max(1)),
                hc_gated: buf(gpu, (d * hc_mult * MAXM).max(1)),
                hc_mixed: buf(gpu, (d * MAXM).max(1)),
                hc_inject: buf(gpu, (hc_mult * MAXM).max(1)),
                ple_idx, ple_rows: raw_buf(ple_stage_bytes.max(4) as u64),
                expert_hash_out: buf(gpu, 4096),
                expert_hash_addr: buf(gpu, 3 * 1024 * 2),
                // Holds one query's flash-decoding partials rather than a batch, and slot
                // dispatches run concurrently, so each slot needs its own region.
                attn_part: buf(gpu, slots * n_head * ojas_metal::kernels::attn::ATTN_NWG * (hd + 2)),
                pmeta, plist,
                // ViT tower scratch (see `vsz` above). vkh/vvh are zeroed because the MMA
                // attention reads up to 7 rows past `total` and those rows must be finite;
                // nothing writes a non-finite value there afterwards, so one zeroing at
                // load covers every image size.
                vimg: buf(gpu, n_vimg), vrows: buf(gpu, n_vrows),
                vx: buf(gpu, n_vd), vh: buf(gpu, n_vd), vqkv: buf(gpu, n_vqkv),
                vq: buf(gpu, n_vd),
                vkh: buf_zeroed(gpu, n_vhalf), vvh: buf_zeroed(gpu, n_vhalf),
                vffn: buf(gpu, n_vffn), vpe: buf(gpu, n_vpe), vmpos: buf(gpu, n_vmpos),
                vout: buf(gpu, n_vout),
                max_seq,
                slots,
            },
            gpu_s: std::cell::Cell::new(0.0),
            want_topk: std::cell::Cell::new(false),
            cur_slot: std::cell::Cell::new(0),
            text_arena: std::cell::RefCell::new(None),
            tune: Tune { max_tg, gemv_plan: HashMap::new(), xv_plan: HashMap::new(), q4l_tg: HashMap::new() },
        };
        // OJAS_EXPERT_PRUNE=<saliency.bin>:<pct>: REAP-style prune×IQ2 sub-100GB sim
        // on the GPU router — patch pruned experts' exp_probs_b bias to -1e30 so
        // moe_topk_v3 never selects them (weight = unbiased sigmoid, unaffected).
        if let (Some(m), Ok(spec)) = (moe, ojas_core::config::var("OJAS_EXPERT_PRUNE")) {
            {
                if let Some((path, pct)) = spec.rsplit_once(':').and_then(|(p, q)| q.parse::<f32>().ok().map(|f| (p, f))) {
                    if let Ok(raw) = std::fs::read(path) {
                        let mut by_layer: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
                        for r in raw.chunks_exact(12) {
                            let l = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
                            let e = u32::from_le_bytes([r[4], r[5], r[6], r[7]]);
                            let s = f32::from_le_bytes([r[8], r[9], r[10], r[11]]);
                            by_layer.entry(l).or_default().push((e, s));
                        }
                        let ndrop = (m.n_expert as f32 * pct / 100.0) as usize;
                        let mut total = 0usize;
                        for (l, mut es) in by_layer {
                            es.sort_by(|a, b| a.1.total_cmp(&b.1));
                            if let Some(bias) = model.wt.w32.get(&format!("blk.{l}.exp_probs_b.bias")) {
                                let ptr = bias.contents() as *mut f32;
                                for &(e, _) in es.iter().take(ndrop) {
                                    if (e as usize) >= m.n_expert as usize { continue; } // guard OOB from malformed prune file
                                    unsafe { *ptr.add(e as usize) = -1e30; }
                                    total += 1;
                                }
                            }
                        }
                        tracing::info!(target: "gpu", "EXPERT_PRUNE: bottom {pct}% saliency → {total} experts masked");
                    }
                }
            }
        }
        // attach the mmap cache (buffers reference it — must live as long as model)
        // and build the temporal expert prefetcher for MoE models.
        if let Some(mc) = wcache {
            if let Some(m) = moe {
                if !ecfg.no_prefetch {
                    use std::collections::HashMap as HM;
                    let mut off: HM<&str, u64> = HM::new();
                    let mut regions = Vec::with_capacity(n_layers);
                    for l in 0..n_layers {
                        off.clear();
                        for e in &mc.entries {
                            if e.kind == crate::weights::K_W4 && e.name.starts_with(&format!("blk.{l}.ffn_")) && e.name.contains("_exps.") {
                                let key = if e.name.contains("gate") { "g" } else if e.name.contains("up") { "u" } else { "d" };
                                off.insert(key, e.offset);
                            }
                        }
                        regions.push(match (off.get("g"), off.get("u"), off.get("d")) {
                            (Some(&g0), Some(&u0), Some(&d0)) => Some(crate::weights::ExpertRegion {
                                gate_off: g0, up_off: u0, down_off: d0,
                                gu_bytes: (m.ffn_exp as u64) * (d as u64) / 2,
                                down_bytes: (d as u64) * (m.ffn_exp as u64) / 2,
                            }),
                            _ => None,
                        });
                    }
                    model.strm.prefetch = Some(crate::weights::ExpertPrefetcher::new(mc.base(), regions));
                }
            }
            model.wt.wcache = Some(mc);
        }
        // streaming (GLM/DeepSeek disk-streamed MoE): predictive whole-token expert
        // prefetch + optional frequency pinning (OJAS_PIN_GB GB of RAM). See weights.rs.
        if model.strm.stream && !ecfg.no_prefetch {
            if let Some(m) = moe {
                let bases = match &model.wt.mapped { Some(mg) => mg.shard_bases(), None => Vec::new() };
                if !bases.is_empty() {
                    let mut regions = Vec::with_capacity(n_layers);
                    for l in 0..n_layers {
                        let nm = |s: &str| format!("blk.{l}.{s}");
                        let gm = g.tensor_meta(&nm("ffn_gate_exps.weight"));
                        let um = g.tensor_meta(&nm("ffn_up_exps.weight"));
                        let dm = g.tensor_meta(&nm("ffn_down_exps.weight"));
                        regions.push(match (gm, um, dm) {
                            (Some((gp, ga, gl, _)), Some((_, ua, _, _)), Some((_, da, dl, _))) =>
                                Some(crate::weights::StreamRegion {
                                    part: gp, gate_abs: ga, up_abs: ua, down_abs: da,
                                    gu_stride: gl / m.n_expert as u64,
                                    down_stride: dl / m.n_expert as u64,
                                }),
                            _ => None,
                        });
                    }
                    let pin_budget = ecfg.pin_budget;
                    let npinnable = regions.iter().filter(|r| r.is_some()).count();
                    tracing::info!(target: "stream", "predictive expert prefetch on ({npinnable} MoE layers){}",
                        if pin_budget > 0 { format!(", pinning hot experts into {:.1} GB RAM", pin_budget as f64 / 1e9) } else { String::new() });
                    model.strm.stream_prefetch = Some(crate::weights::StreamPrefetcher::new(bases, regions, pin_budget));
                }
            }
        }
        // Autotune picks decoder GEMV plans by decoder tensor name; an encoder runs
        // only the batched GEMM.
        if (quant || q4mode) && model.arch.text_encoder.is_none() {
            model.autotune();
        }
        if stream && is_qwen4exp && !ecfg.flash_expert_pool {
            let plan = model.resident_memory_plan()?;
            tracing::info!(target: "stream",
                "resident plan: {:.3} GB complete GPU request ({:.3} GB existing + {:.3} GB expert views), {:.2} GB RAM reserve, {:.2} GB Metal limit; fits={}",
                plan.requested_bytes as f64 / 1e9, plan.existing_bytes as f64 / 1e9,
                plan.expert_bytes as f64 / 1e9, plan.reserve_bytes as f64 / 1e9,
                plan.working_set_bytes as f64 / 1e9, plan.fits());
            let streaming_requested = ecfg.flash_resident == Some(false)
                || ecfg.flash_resident.is_none() && (ecfg.flash_direct_experts || ecfg.moe_dbuf);
            if streaming_requested {
                tracing::info!(target: "stream", "expert mode: streaming (requested)");
            } else if !plan.fits() {
                tracing::warn!(target: "stream", "complete resident request exceeds RAM reserve or Metal working-set limit; using streaming");
            } else if !live_residency_fits(plan.requested_bytes, available_ram_bytes()) {
                tracing::warn!(target: "stream", "insufficient live RAM for resident request plus 8 GiB headroom; using streaming");
            } else {
                anyhow::ensure!(!ecfg.flash_direct_experts && !ecfg.flash_expert_pool && !ecfg.moe_dbuf,
                    "resident mode replaces the streaming cache; use --stream with direct experts/prefetch");
                let mut experts: Vec<_> = model.strm.stream_meta.iter().filter(|(name, _)| {
                    expert_layer(name).is_some_and(|l| l < n_layers)
                }).map(|(name, &meta)| (name.clone(), meta)).collect();
                experts.sort_by(|a, b| a.0.cmp(&b.0));
                let mapped = model.wt.mapped.as_ref().ok_or_else(|| anyhow::anyhow!("missing expert mapping"))?;
                // No expert Metal allocation is created before the full planned
                // request passes both budgets above.
                for (name, (part, offset, bytes, _)) in experts {
                    let (buffer, boff) = mapped.buffer(gpu, part, offset, bytes);
                    if boff > 0 { model.wt.w_off.insert(name.clone(), boff); }
                    model.wt.wq.insert(name.clone(), buffer);
                    model.strm.stream_meta.remove(&name);
                }
                let buffers = model.all_gpu_buffers();
                let count = buffers.len();
                let actual = checked_buffer_bytes(&buffers)?;
                anyhow::ensure!(actual == plan.requested_bytes && plan.fits(),
                    "resident plan mismatch: planned {} bytes, actual {}; refusing residency", plan.requested_bytes, actual);
                // PLE's table is not a GPU resource; everything that is referenced,
                // including its bounded staging, is covered here.
                let residency = gpu.make_residency_set(&buffers)
                    .ok_or_else(|| anyhow::anyhow!("resident mode requires MTLResidencySet (macOS 15+); use --stream"))?;
                model.strm.expert_residency = Some(residency);
                model.strm.resident_layers = n_layers;
                gpu.set_unretained_command_buffers(true);
                tracing::info!(target: "stream", "expert mode: resident, {count} buffers, {:.3} GB requested; PLE remains CPU-only", actual as f64 / 1e9);
            }
        }
        Ok(model)
    }

    /// deepseek2 YaRN attention mscale: kq_scale = mscale²/sqrt(head_dim).
    ///
    /// The GGUF stores yarn_log_multiplier = 0.1·mscale_all_dim, so mscale =
    /// yarn_log_multiplier/0.1 (0.707 for V2-Lite, matching llama's "mscale == 0.7" log).
    /// `OJAS_KQMSCALE` overrides.
    pub(crate) fn ds_mscale(g: &Gguf) -> f32 {
        if let Ok(v)=ojas_core::config::var("OJAS_KQMSCALE"){ if let Ok(f)=v.parse::<f32>(){ return f; } }
        let factor = g.meta_f32("deepseek2.rope.scaling.factor").unwrap_or(1.0);
        let log_mul = g.meta_f32("deepseek2.rope.scaling.yarn_log_multiplier").unwrap_or(0.0);
        if factor <= 1.0 || log_mul <= 0.0 { 1.0 } else { log_mul / 0.1 }
    }

    /// The context window this model was allocated with: the requested ctx after the
    /// RAM-budget clamp, or the auto pick. This is the loaded ctx callers should report.
    pub fn max_seq(&self) -> usize {
        self.st.max_seq
    }

    /// Build the per-layer plan from GGUF metadata — the arch-agnostic core. Most archs are
    /// uniform (all layers identical; gemma3 varies only the RoPE base). gemma-4 varies
    /// head_dim/n_kv/rope per layer and shares KV across layers of the same attention type
    /// (global/sliding), so shared layers reuse an earlier layer's V.
    pub(crate) fn build_layers(g: &Gguf, arch: &str, n_layers: usize, n_head: usize, n_kv: usize,
                    hd: usize, rope_base: f32, rope_local: f32, swa_pattern: u32) -> Vec<LayerPlan> {
        // gemma-4 uses attention scale 1.0 (Q,K are RMS-normed); others use 1/sqrt(head_dim).
        let scale_one = arch == "gemma4";
        let plan1 = |nh: usize, nkv: usize, h: usize, rb: f32, src: usize, hasv: bool| LayerPlan {
            n_head: nh as u32, n_kv: nkv as u32, head_dim: h as u32,
            qdim: (nh * h) as u32, kvdim: (nkv * h) as u32,
            rope_base: rb, scale: if scale_one { 1.0 } else { 1.0 / (h as f32).sqrt() },
            kv_source: src, has_v: hasv, is_ssm: false,
        };
        if arch == "qwen35" || arch == "qwen35moe" || arch == "qwen4exp" {
            // Gated-DeltaNet hybrid: (l+1)%interval != 0 → recurrent layer; else attention.
            let interval = g.meta_u32(&format!("{arch}.full_attention_interval")).unwrap_or(4);
            return (0..n_layers).map(|l| {
                let is_ssm = (l as u32 + 1) % interval != 0;
                let mut p = plan1(n_head, n_kv, hd, rope_base, l, true);
                p.is_ssm = is_ssm;
                p
            }).collect();
        }
        if arch == "deepseek2" || arch == "glm-dsa" {
            // MLA (absorbed): per-head k_mla = nope+rope. scale = mscale²/√k_mla (ds_mscale=1
            // for GLM — no YaRN keys → mscale 1 → scale 1/√256, matches the reference glm implementation).
            let k_mla = g.meta_u32(&format!("{arch}.attention.key_length_mla"))
                .or_else(|| g.meta_u32(&format!("{arch}.attention.key_length"))).unwrap_or(192) as usize;
            let mscale = Self::ds_mscale(g);
            let scale = mscale * mscale / (k_mla as f32).sqrt();
            return (0..n_layers).map(|l| LayerPlan {
                n_head: n_head as u32, n_kv: n_head as u32, head_dim: k_mla as u32,
                qdim: (n_head * k_mla) as u32, kvdim: (n_head * k_mla) as u32,
                rope_base, scale, kv_source: l, has_v: true, is_ssm: false,
            }).collect();
        }
        if arch == "gemma4" {
            // per-layer arrays: sliding_window_pattern (1=sliding/local, 0=global),
            // head_count_kv[l]; dual head_dim (key_length full / key_length_swa sliding);
            // dual rope base (freq_base global / freq_base_swa sliding).
            let swp = g.int_arr("gemma4.attention.sliding_window_pattern").cloned().unwrap_or_default();
            let kvarr = g.int_arr("gemma4.attention.head_count_kv").cloned().unwrap_or_default();
            let hd_full = g.meta_u32("gemma4.attention.key_length").map(|v| v as usize).unwrap_or(hd);
            let hd_swa = g.meta_u32("gemma4.attention.key_length_swa").map(|v| v as usize).unwrap_or(hd_full);
            let base_full = g.meta_f32("gemma4.rope.freq_base").unwrap_or(rope_base);
            let base_swa = g.meta_f32("gemma4.rope.freq_base_swa").unwrap_or(10000.0);
            let mut plans = Vec::with_capacity(n_layers);
            for l in 0..n_layers {
                let sliding = swp.get(l).copied().unwrap_or(1) != 0; // default sliding
                let h = if sliding { hd_swa } else { hd_full };
                let nkv = kvarr.get(l).copied().unwrap_or(n_kv as i64) as usize;
                let rb = if sliding { base_swa } else { base_full };
                // has_v=false (global layers): no wv weight → V=K projection, computed
                // into this layer's own V cache (not cross-layer shared), so kv_source=l.
                let has_v = g.tensors.contains_key(&format!("blk.{l}.attn_v.weight"));
                plans.push(plan1(n_head, nkv, h, rb, l, has_v));
            }
            return plans;
        }
        // uniform archs (qwen2/qwen3/llama) + gemma3 (per-layer dual RoPE base).
        (0..n_layers).map(|l| {
            let rb = if swa_pattern > 0 && (l as u32 + 1) % swa_pattern != 0 { rope_local } else { rope_base };
            plan1(n_head, n_kv, hd, rb, l, true)
        }).collect()
    }

}


/// True when attn_q/attn_k/attn_v for this layer are all Q4_K — the precondition for the
/// fused `qkv_q4l` dispatch.
fn qkv_triple_native(types: &HashMap<String, u32>, name: &str) -> bool {
    let base = name
        .replace("attn_q.", "attn_@.")
        .replace("attn_k.", "attn_@.")
        .replace("attn_v.", "attn_@.");
    ["attn_q.", "attn_k.", "attn_v."]
        .iter()
        .all(|t| types.get(&base.replace("attn_@.", t)) == Some(&12))
}

/// True when this tensor's ffn gate/up partner carries the same native K-quant type, which
/// is what `ffn_gu_q4k` / `ffn_gu_q6k` require (one block stride for both).
fn ffn_pair_native(types: &HashMap<String, u32>, name: &str) -> bool {
    let partner = if name.contains("ffn_gate.") {
        name.replace("ffn_gate.", "ffn_up.")
    } else {
        name.replace("ffn_up.", "ffn_gate.")
    };
    match (types.get(name), types.get(&partner)) {
        (Some(a), Some(b)) => a == b && (*a == 12 || *a == 14),
        _ => false,
    }
}

// ============================================================================
// Safetensors + MXFP4 checkpoint loading
// ============================================================================
//
// `DecoderGpu::load` above is the GGUF path. This is the reusable substrate for loading
// checkpoints that ship as sharded safetensors with MXFP4 experts (gpt-oss is the
// motivating case): it detects the arch from `config.json`, walks the `weight_map`, and
// assembles ojas's weight representation — MXFP4 tensors repacked into the native 17-byte
// block layout (`ojas_core::quant_src` `MXFP4_SUB` decodes it), float tensors widened to
// f32.
//
// Built and unit-tested at the formats layer:
//   * multi-shard index parse + per-tensor seekable reads   (ojas_formats::safetensors::SafeIndex)
//   * arch detection from config.json                        (HfConfig / is_gpt_oss)
//   * MXFP4 pair detection (`*.blocks` + `*.scales`) and repack into native
//     ojas blocks ready to upload as a `gemv_nat_mxfp4` weight
//   * float (BF16/F16/F32) widening for the dense skeleton + biases
//
// Not built yet, for a live gpt-oss decode:
//   * uploading these buffers into a `DecoderGpu` and building its kernel
//     pipelines, which belongs in `decoder/mod.rs`
//   * HF→ojas tensor-name remap and the gpt-oss MoE graph (biased router, biased
//     experts, no shared expert, SwiGLU-OAI clamp) + attention sinks; the GGUF
//     `arch == "gpt-oss"` branch above shows the tensor set the graph expects
//   * FP8 / NVFP4 scale decode (kept as `Raw` here) for the FP8 checkpoints

// Dead-code allow: `mod load` is private (owned by decoder/mod.rs), so nothing in the
// non-test build path can name these yet. Exposing them to the engine is a one-line
// `pub use load::{SafeCheckpoint, load_safetensors_checkpoint, ...}` in decoder/mod.rs.
#[allow(dead_code)]
/// A weight materialized from a safetensors checkpoint into ojas's representation.
pub enum SafeWeight {
    /// Native MXFP4: ojas 17-byte blocks (`{u8 scale; u8 qs[16]}`), row-major
    /// with K innermost. `rows` counts every output row across any expert stack
    /// (`E*N` for a 3-D expert tensor), `k` is the per-row reduction dim.
    Mxfp4 { blocks: Vec<u8>, rows: usize, k: usize, orig_shape: Vec<usize> },
    /// A float weight widened to f32 (dense skeleton, norms, biases).
    F32 { data: Vec<f32>, shape: Vec<usize> },
    /// Kept verbatim — a dtype ojas cannot yet decode (FP8 today). Recorded so a
    /// caller can see it exists rather than silently dropping it.
    Raw { dtype: String, bytes: Vec<u8>, shape: Vec<usize> },
}

#[allow(dead_code)]
impl SafeWeight {
    pub fn shape(&self) -> &[usize] {
        match self {
            SafeWeight::Mxfp4 { orig_shape, .. } => orig_shape,
            SafeWeight::F32 { shape, .. } => shape,
            SafeWeight::Raw { shape, .. } => shape,
        }
    }
}

/// A checkpoint loaded (or probed) from safetensors: the detected arch config
/// plus the requested weights in ojas's representation.
#[allow(dead_code)]
pub struct SafeCheckpoint {
    pub cfg: ojas_formats::safetensors::HfConfig,
    pub is_gpt_oss: bool,
    /// True when `config.json` declares MXFP4 quantization.
    pub mxfp4: bool,
    pub weights: HashMap<String, SafeWeight>,
    /// MXFP4 pairs seen in the checkpoint, even ones not materialized:
    /// (blocks-tensor name, rows, k). Useful for a dry-run / manifest.
    pub mxfp4_pairs: Vec<(String, usize, usize)>,
}

/// Detect the `(rows, k)` of an MXFP4 `*.blocks` tensor and its `*.scales`
/// sibling, validating the group geometry. `rows` is the product of the scales'
/// leading dims (all but the last), `k = scales.last() * GROUP`. `blocks` and
/// `scales` need not share a rank: real HF gpt-oss ships `*_blocks` as rank-4
/// `[E, 2*inter, K/32, 16]` (u8) paired with rank-3 `*_scales` `[E, 2*inter,
/// K/32]`. Both are treated as flat buffers, so only the element counts must
/// agree: `blocks.numel() == rows*(k/2)`, `scales.numel() == rows*(k/32)`.
/// Returns `None` when the sibling is missing or the geometry is wrong.
#[allow(dead_code)]
fn mxfp4_geometry(
    si: &ojas_formats::safetensors::SafeIndex,
    blocks_name: &str,
    scales_name: &str,
) -> Option<(usize, usize)> {
    let b = si.get(blocks_name)?;
    let s = si.get(scales_name)?;
    if b.dtype != "U8" || s.dtype != "U8" {
        return None;
    }
    let (bsh, ssh) = (&b.shape, &s.shape);
    if bsh.is_empty() || ssh.is_empty() {
        return None;
    }
    // Geometry is derived from the scales: one scale per GROUP-sized group, so the
    // scales' leading dims are the row count and its last dim is k/GROUP.
    let rows: usize = ssh[..ssh.len() - 1].iter().product();
    let k = *ssh.last().unwrap() * ojas_formats::mxfp4::GROUP;
    if k == 0 || k % ojas_formats::mxfp4::GROUP != 0 {
        return None;
    }
    // Validate both tensors' total element counts against the derived geometry —
    // blocks pack 2 nibbles/byte (k/2 bytes/row), scales are k/GROUP/row.
    let bnum: usize = bsh.iter().product();
    let snum: usize = ssh.iter().product();
    if bnum != rows * (k / 2) || snum != rows * (k / ojas_formats::mxfp4::GROUP) {
        return None;
    }
    Some((rows, k))
}

/// The `*.scales` name paired with a `*.blocks` name, matching the HF gpt-oss
/// convention (`..._blocks` / `..._scales`).
#[allow(dead_code)]
fn scales_name_for(blocks_name: &str) -> Option<String> {
    blocks_name
        .strip_suffix("_blocks")
        .map(|p| format!("{p}_scales"))
        .or_else(|| blocks_name.strip_suffix(".blocks").map(|p| format!("{p}.scales")))
}

/// Probe a safetensors checkpoint directory: parse the index + config and list the MXFP4
/// pairs without reading any tensor data, so it is safe on multi-GB checkpoints.
/// Materialize what you need with [`load_safetensors_checkpoint`].
#[allow(dead_code)]
pub fn probe_safetensors_checkpoint(dir: &str) -> Result<SafeCheckpoint> {
    load_safetensors_checkpoint(dir, |_| false)
}

/// Load a safetensors checkpoint, materializing every tensor whose name passes
/// `want`. MXFP4 `*.blocks` tensors are repacked into native ojas blocks (their
/// `*.scales` sibling is consumed, not returned separately); float tensors are
/// widened to f32; FP8 and other undecoded dtypes are kept as `Raw`.
#[allow(dead_code)]
pub fn load_safetensors_checkpoint(
    dir: &str,
    want: impl Fn(&str) -> bool,
) -> Result<SafeCheckpoint> {
    use ojas_formats::safetensors::SafeIndex;
    let si = SafeIndex::open(dir)?;
    let cfg = si.hf_config().unwrap_or_default();
    let is_gpt_oss = cfg.is_gpt_oss();
    let mxfp4 = cfg.quant_method == "mxfp4"
        || si.names().any(|n| n.ends_with("_blocks") || n.ends_with(".blocks"));

    // Index the MXFP4 pairs first, so the `*.scales` halves can be skipped when
    // materializing and a probe can report them.
    let mut scales_consumed: std::collections::HashSet<String> = Default::default();
    let mut mxfp4_pairs: Vec<(String, usize, usize)> = Vec::new();
    let mut names: Vec<String> = si.names().cloned().collect();
    names.sort();
    for n in &names {
        if n.ends_with("_blocks") || n.ends_with(".blocks") {
            if let Some(sc) = scales_name_for(n) {
                if let Some((rows, k)) = mxfp4_geometry(&si, n, &sc) {
                    mxfp4_pairs.push((n.clone(), rows, k));
                    scales_consumed.insert(sc);
                }
            }
        }
    }

    let mut weights = HashMap::new();
    for n in &names {
        if !want(n) || scales_consumed.contains(n) {
            continue;
        }
        // MXFP4 block tensor → repack with its scales sibling.
        if let Some(sc) = scales_name_for(n) {
            if let Some((rows, k)) = mxfp4_geometry(&si, n, &sc) {
                let blocks_raw = si.read_raw(n)?;
                let scales_raw = si.read_raw(&sc)?;
                let packed = ojas_formats::mxfp4::pack_from_hf(&blocks_raw, &scales_raw, rows, k);
                weights.insert(
                    n.clone(),
                    SafeWeight::Mxfp4 { blocks: packed, rows, k, orig_shape: si.shape(n)?.to_vec() },
                );
                continue;
            }
        }
        let dtype = si.dtype(n)?.to_string();
        let shape = si.shape(n)?.to_vec();
        let w = match dtype.as_str() {
            "F32" | "F16" | "BF16" => SafeWeight::F32 { data: si.read_f32(n)?, shape },
            _ => SafeWeight::Raw { dtype, bytes: si.read_raw(n)?, shape },
        };
        weights.insert(n.clone(), w);
    }

    Ok(SafeCheckpoint { cfg, is_gpt_oss, mxfp4, weights, mxfp4_pairs })
}

#[cfg(test)]
mod resident_memory_tests {
    use super::*;

    #[test]
    fn complete_request_must_pass_both_budgets_and_live_headroom() {
        let mut plan = ResidentMemoryPlan { existing_bytes: 7 << 30, expert_bytes: 55 << 30,
            requested_bytes: 62 << 30, reserve_bytes: 24 << 30,
            ram_bytes: 96 << 30, working_set_bytes: 72 << 30 };
        assert!(plan.fits());
        plan.requested_bytes = 95_500_000_000;
        assert!(!plan.fits(), "historical crashing request must remain rejected");
        plan.requested_bytes = 62 << 30;
        plan.working_set_bytes = 61 << 30;
        assert!(!plan.fits());
        plan.working_set_bytes = 72 << 30;
        plan.reserve_bytes = 23 << 30;
        assert!(!plan.fits());
        plan.reserve_bytes = u64::MAX;
        assert!(!plan.fits());
        assert!(!live_residency_fits(62 << 30, None));
        assert!(!live_residency_fits(62 << 30, Some(69 << 30)));
        assert!(live_residency_fits(62 << 30, Some(70 << 30)));
        assert!(!live_residency_fits(u64::MAX, Some(u64::MAX)));
    }

    #[test]
    #[ignore = "requires a Metal device and macOS 15+"]
    fn small_residency_set_can_detach_and_be_recreated() {
        let gpu = MetalGpu::new().unwrap();
        let buffer = gpu.device.new_buffer(16384, MTLResourceOptions::StorageModeShared);
        for _ in 0..3 {
            let set = gpu.make_residency_set(&[&buffer]).expect("macOS 15 residency support");
            set.renew();
            let cb = gpu.queue.new_command_buffer();
            cb.commit();
            cb.wait_until_completed();
            drop(set);
        }
    }
}

#[cfg(test)]
mod safetensors_load_tests {
    use super::{load_safetensors_checkpoint, probe_safetensors_checkpoint, SafeWeight};
    use ojas_formats::safetensors::serialize;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!("ojas_sload_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A tiny gpt-oss-shaped checkpoint: one BF16 dense weight + one MXFP4 expert
    /// stack (blocks + scales), single shard, with config.json + index.json.
    #[test]
    fn loads_mxfp4_experts_and_dense_skeleton() {
        let dir = tmpdir("mxfp4");

        // MXFP4 expert stack: E=2 experts, N=4 rows, K=64 -> blocks [2,4,32], scales [2,4,2].
        let (e, n, k) = (2usize, 4usize, 64usize);
        let rows = e * n;
        let row_qs = k / 2;
        let gpr = k / ojas_formats::mxfp4::GROUP;
        let mut blocks = vec![0u8; rows * row_qs];
        let mut st: u32 = 0xC0FFEE11;
        for b in blocks.iter_mut() {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            *b = (st >> 8) as u8;
        }
        let mut scales = vec![0u8; rows * gpr];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 120 + (i as u8 % 8);
        }
        // dense BF16 weight
        let dense: Vec<f32> = vec![1.0, -2.0, 0.5, 4.0];
        let dense_bytes: Vec<u8> =
            dense.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();

        let bytes = serialize(&[
            ("model.layers.0.mlp.experts.gate_up_proj_blocks", "U8", vec![e, n, row_qs], blocks.clone()),
            ("model.layers.0.mlp.experts.gate_up_proj_scales", "U8", vec![e, n, gpr], scales.clone()),
            ("model.layers.0.self_attn.q_proj.weight", "BF16", vec![4], dense_bytes),
        ]);
        std::fs::write(dir.join("model.safetensors"), &bytes).unwrap();
        std::fs::write(
            dir.join("config.json"),
            br#"{"architectures":["GptOssForCausalLM"],"model_type":"gpt_oss",
                 "num_local_experts":2,"num_experts_per_tok":1,"hidden_size":8,
                 "num_hidden_layers":1,"quantization_config":{"quant_method":"mxfp4"}}"#,
        )
        .unwrap();

        // probe: no data read, but geometry + arch detected
        let probe = probe_safetensors_checkpoint(dir.to_str().unwrap()).unwrap();
        assert!(probe.is_gpt_oss);
        assert!(probe.mxfp4);
        assert_eq!(probe.cfg.num_local_experts, 2);
        assert_eq!(probe.mxfp4_pairs.len(), 1);
        let (pname, prows, pk) = &probe.mxfp4_pairs[0];
        assert!(pname.ends_with("gate_up_proj_blocks"));
        assert_eq!((*prows, *pk), (rows, k));
        assert!(probe.weights.is_empty(), "probe materializes nothing");

        // full load
        let ck = load_safetensors_checkpoint(dir.to_str().unwrap(), |_| true).unwrap();
        // scales half is consumed, not surfaced
        assert!(!ck.weights.contains_key("model.layers.0.mlp.experts.gate_up_proj_scales"));

        // MXFP4 weight: native blocks, byte-identical to a direct repack, and
        // decodes to the same f32 as the reference HF dequant.
        match ck.weights.get("model.layers.0.mlp.experts.gate_up_proj_blocks").unwrap() {
            SafeWeight::Mxfp4 { blocks: packed, rows: r, k: kk, orig_shape } => {
                assert_eq!((*r, *kk), (rows, k));
                assert_eq!(orig_shape, &vec![e, n, row_qs]);
                assert_eq!(packed.len(), rows * gpr * ojas_formats::mxfp4::BLOCK_BYTES);
                let want = ojas_formats::mxfp4::pack_from_hf(&blocks, &scales, rows, k);
                assert_eq!(packed, &want);
                let via_blocks = ojas_formats::mxfp4::dequant_blocks(packed, rows * k);
                let direct = ojas_formats::mxfp4::dequant(&blocks, &scales, rows * gpr);
                assert_eq!(via_blocks, direct);
            }
            _ => panic!("expected MXFP4 weight"),
        }

        // dense BF16 widened to f32 exactly
        match ck.weights.get("model.layers.0.self_attn.q_proj.weight").unwrap() {
            SafeWeight::F32 { data, .. } => assert_eq!(data, &dense),
            _ => panic!("expected F32 dense weight"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The real HF gpt-oss layout: `*_blocks` is rank-4 `[E, 2*inter, K/32, 16]` (u8)
    /// paired with rank-3 `*_scales` `[E, 2*inter, K/32]`. The ranks differ, so an
    /// equal-rank check drops real experts to undecoded `Raw`: geometry must be derived
    /// from the scales and validated by element count, and the repack (flat buffers) stays
    /// byte-identical.
    #[test]
    fn loads_mxfp4_rank4_blocks_rank3_scales() {
        let dir = tmpdir("mxfp4_rank4");

        // E=2 experts, N=4 rows, K=64 → gpr = K/32 = 2, row_qs = K/2 = 32.
        // blocks rank-4 [E, N, gpr, 16]; scales rank-3 [E, N, gpr].
        let (e, n, k) = (2usize, 4usize, 64usize);
        let rows = e * n;
        let row_qs = k / 2;
        let gpr = k / ojas_formats::mxfp4::GROUP;
        let half_group = ojas_formats::mxfp4::GROUP / 2; // 16
        assert_eq!(gpr * half_group, row_qs);

        let mut blocks = vec![0u8; rows * row_qs];
        let mut st: u32 = 0xC0FFEE11;
        for b in blocks.iter_mut() {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            *b = (st >> 8) as u8;
        }
        let mut scales = vec![0u8; rows * gpr];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 120 + (i as u8 % 8);
        }

        let bytes = serialize(&[
            (
                "model.layers.0.mlp.experts.gate_up_proj_blocks",
                "U8",
                vec![e, n, gpr, half_group],
                blocks.clone(),
            ),
            (
                "model.layers.0.mlp.experts.gate_up_proj_scales",
                "U8",
                vec![e, n, gpr],
                scales.clone(),
            ),
        ]);
        std::fs::write(dir.join("model.safetensors"), &bytes).unwrap();
        std::fs::write(
            dir.join("config.json"),
            br#"{"architectures":["GptOssForCausalLM"],"model_type":"gpt_oss",
                 "num_local_experts":2,"num_experts_per_tok":1,"hidden_size":8,
                 "num_hidden_layers":1,"quantization_config":{"quant_method":"mxfp4"}}"#,
        )
        .unwrap();

        // probe: rank-4/rank-3 pair is detected with the right (rows, k).
        let probe = probe_safetensors_checkpoint(dir.to_str().unwrap()).unwrap();
        assert_eq!(probe.mxfp4_pairs.len(), 1, "rank-4 blocks must still be paired");
        let (_, prows, pk) = &probe.mxfp4_pairs[0];
        assert_eq!((*prows, *pk), (rows, k));

        // full load: repack from the flat buffers is byte-identical, and not Raw.
        let ck = load_safetensors_checkpoint(dir.to_str().unwrap(), |_| true).unwrap();
        match ck.weights.get("model.layers.0.mlp.experts.gate_up_proj_blocks").unwrap() {
            SafeWeight::Mxfp4 { blocks: packed, rows: r, k: kk, orig_shape } => {
                assert_eq!((*r, *kk), (rows, k));
                // orig_shape preserves the on-disk rank-4 shape.
                assert_eq!(orig_shape, &vec![e, n, gpr, half_group]);
                let want = ojas_formats::mxfp4::pack_from_hf(&blocks, &scales, rows, k);
                assert_eq!(packed, &want);
            }
            _ => panic!("expected MXFP4 weight — rank-4 blocks fell through to Raw"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
