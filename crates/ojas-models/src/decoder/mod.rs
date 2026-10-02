//! GPU-resident Qwen2 decode on Metal. Weights live on the GPU; every per-token
//! op (RMSNorm, GEMV+bias, RoPE, GQA attention, SwiGLU, residual, LM head) is a
//! Metal kernel, chained into one command buffer per token with a single sync and
//! no host round-trips.

use ojas_metal::MetalGpu;
use ojas_core::Device as _;
use metal::{ComputePipelineState, MTLResourceOptions};
use std::collections::HashMap;

use ojas_core::config::EngineConfig;

/// Per-layer plan describing each layer's geometry, RoPE and KV-sharing;
/// encode_forward interprets a `Vec<LayerPlan>` instead of branching per arch.
/// Uniform archs get N identical plans; gemma-4 varies per layer (mixed head_dim/n_kv,
/// dual RoPE base, cross-layer KV sharing where a layer reuses another's V cache).
/// qwen35 Gated-DeltaNet (Qwen3-Next) SSM config. Layers where (l+1)%attn_interval != 0
/// are SSM blocks; the rest are standard attention.
#[derive(Clone, Copy)]
pub struct SsmConfig {
    d_state: u32,      // S: state size (128) — the matrix state is [S × S] per head
    n_group: u32,      // H_k: number of k/q heads (16)
    dt_rank: u32,      // H_v: number of v heads (32) = number of matrix-state heads
    d_inner: u32,      // inner size (4096); head_v_dim = d_inner / dt_rank
    conv_kernel: u32,  // depthwise causal conv width (4)
    attn_interval: u32,// every attn_interval-th layer ((l+1)%interval==0) is attention (4)
    n_rot: u32,        // rotary dims for the attention layers (64 of hd=256 — partial rope).
                       // M-RoPE sections [11,11,10,0] reduce to standard NEOX n_dims=64 for
                       // text-only (all position streams equal) — see mrope_sections.
    /// M-RoPE section sizes from `{arch}.rope.dimension_sections`, in cos/sin pairs
    /// (they sum to `n_rot/2`: surya-2 declares [11,11,10,0] against n_rot 64).
    ///
    /// All-zero means the model declared none, i.e. plain partial NEOX rope. A declared
    /// split only takes effect once a token carries a position whose (t,h,w) disagree —
    /// an image span; text positions make every section select the same number, and
    /// `ojas-metal/tests/rope_sections.rs` asserts that reproduces the scalar kernel
    /// bit-for-bit. qwen35 is IMROPE (interleaved [t h w t h w …], llama.cpp
    /// `llama-model.cpp:3034`), not the contiguous qwen2-vl layout; the kernel takes
    /// that as `ojas_metal::kernels::ops::MROPE_INTERLEAVED`.
    #[allow(dead_code)]  // read by the M-RoPE dispatch once the image path is wired
    mrope_sections: [u32; 4],
}

/// What a tensor is used for, decided once from its name.
///
/// One classifier, because the name tests collide: `per_layer_token_embd` contains
/// `token_embd`, so an embedding exclusion also swallows it and sends a 51-billion
/// parameter n-gram table to the requantizer — ~100 GB of RAM to serve about 1.4 KB
/// of reads per token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TensorRole {
    /// Vocabulary embedding table; read by the `embed_*` row gathers, never a gemv.
    Embedding,
    /// N-gram hashed per-layer embedding table (qwen4exp PLE). Enormous and read
    /// a handful of rows at a time, so only selected rows may enter Metal.
    NgramTable,
    /// Routed MoE expert stack — the MoE kernels read their own packed buffers.
    MoeExpert,
    /// Always-on shared expert; an ordinary gemv, but not part of the routed stack.
    SharedExpert,
    /// Depthwise conv / elementwise SSM parameter; must stay f16 or f32.
    Elementwise,
    /// Ordinary 2-D matmul weight.
    Gemv,
}

/// Classify a tensor by name.
///
/// Order matters: `per_layer_token_embd` is tested before `token_embd`, because the
/// former contains the latter.
pub(crate) fn role_of(name: &str) -> TensorRole {
    if name.contains("per_layer_token_embd") { return TensorRole::NgramTable; }
    if name.contains("token_embd") || name.contains("embed_tokens") { return TensorRole::Embedding; }
    if name.contains("_exps.") { return TensorRole::MoeExpert; }
    if name.contains("_shexp") { return TensorRole::SharedExpert; }
    if name.contains("ssm_conv1d") { return TensorRole::Elementwise; }
    TensorRole::Gemv
}

/// Which half of a streamed MoE layer is being encoded.
///
/// Disk-streamed MoE runs each layer in two command buffers: `Route` does attention +
/// router + shared expert (writing `moe_idx`), the CPU then gathers only the routed
/// experts into packed scratch, and `Experts` runs the expert GEMMs over that scratch,
/// so Metal wires ~n_used experts per buffer instead of every expert in the model.
/// `Full` is the ordinary one-buffer path.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum MoePhase { Full, Route, Experts }

/// One missing-expert read job for the parallel pread gather (all fields Copy/Send).
#[derive(Clone, Copy)]
struct GatherJob { key: u64, fd: i32, off: i64, len: usize, dst: usize }

/// Userspace LRU cache of gathered expert weight bytes (native quant), keyed by
/// (layer, expert, kind). Survives the 439GB mmap page-cache thrash so decode — which
/// reuses ~90% of experts token-to-token — mostly hits RAM instead of re-reading disk.
enum ExpertBytes {
    Cpu(Box<[u8]>),
    Metal(std::sync::Arc<expert_pool::MetalExpert>),
}
impl ExpertBytes {
    fn len(&self) -> usize { match self { Self::Cpu(b) => b.len(), Self::Metal(b) => b.len } }
    fn as_ptr(&self) -> *const u8 { match self { Self::Cpu(b) => b.as_ptr(), Self::Metal(b) => b.contents() as *const u8 } }
}
pub(crate) struct ExpertCache {
    map: std::collections::HashMap<u64, (ExpertBytes, u64, u64)>, // key -> (bytes, freq, last_used)
    metal_device: Option<metal::Device>,
    pool: Option<expert_pool::ExpertPool>,
    clock: u64,
    bytes: usize,
    budget: usize,
    dbuf: Option<Dbuf>, // double-buffer (streaming MoE prefetch) staging ring; None until first use
}

/// Staging buffers for the double-buffered gather (OJAS_MOE_DBUF opt-in): one staging
/// slot, drained into moe_gs each layer then refilled, not a ring of two. While the GPU
/// runs layer L's expert GEMMs over the packed scratch moe_gs/us/ds, a background thread
/// preads layer L+1's predicted-routed experts (stale moe_idx, the same ~90%-accurate
/// decode predictor stream_prefetch_routed uses) into these staging buffers; when
/// gather() reaches L+1 it memcpys the staged bytes into moe_gs instead of blocking on
/// disk. A staged byte range is pread from the exact fd/offset a serial gather would use,
/// so it is byte-identical to disk, and a miss or misprediction falls back to a
/// synchronous pread (what OJAS_MOE_DBUF unset always does). The GPU only reads moe_gs
/// and the background thread only writes staging, so the bytes the GPU consumes are
/// never raced.
struct Dbuf {
    stage: [Vec<u8>; 3],                                   // gate/up/down staging (sized == moe_gs/us/ds), allocated once
    staged: std::collections::HashMap<u64, (usize, usize)>, // key -> (kind 0/1/2, byte offset in stage[kind])
    handle: Option<std::thread::JoinHandle<()>>,           // in-flight background prefetch for the next layer
    pf_hits: u64, pf_reads: u64,                           // prefetch telemetry (staging bytes served / staged)
}

impl ExpertCache {
    fn new(budget: usize) -> Self { Self { map: std::collections::HashMap::new(), metal_device: None, pool: None, clock: 0, bytes: 0, budget, dbuf: None } }
    fn budget(&self) -> usize { self.budget }
    fn bytes(&self) -> usize { self.bytes }
    fn contains(&self, key: u64) -> bool { self.map.contains_key(&key) }
    fn get(&mut self, key: u64) -> Option<*const u8> {
        self.clock += 1; let c = self.clock;
        if let Some((b, f, lu)) = self.map.get_mut(&key) { *f += 1; *lu = c; Some(b.as_ptr()) } else { None }
    }
    fn metal_buffer(&self, key: u64) -> Option<std::sync::Arc<expert_pool::MetalExpert>> {
        match &self.map.get(&key)?.0 { ExpertBytes::Metal(b) => Some(b.clone()), _ => None }
    }
    fn insert(&mut self, key: u64, data: Box<[u8]>) {
        if self.budget == 0 { return; }
        self.clock += 1; let c = self.clock;
        let data = if self.pool.is_some() {
            if data.len() > self.budget { return; }
            let allocation = loop {
                if let Some(a) = self.pool.as_ref().unwrap().allocate(data.len()) { break Some(a); }
                if !self.evict_cold() { break None; }
            };
            // Active GPU pins may hold all remaining space. Skip admission;
            // this call already has correct bytes in its scratch slot.
            let Some(a) = allocation else { return; };
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), a.contents(), data.len()); }
            ExpertBytes::Metal(a)
        } else if let Some(device) = &self.metal_device {
            ExpertBytes::Metal(expert_pool::MetalExpert::standalone(device.new_buffer_with_data(data.as_ptr() as *const std::ffi::c_void,
                data.len() as u64, metal::MTLResourceOptions::StorageModeShared)))
        } else { ExpertBytes::Cpu(data) };
        self.bytes += data.len();
        if let Some((old, _, _)) = self.map.insert(key, (data, 1, c)) { self.bytes -= old.len(); }
        // LFU eviction (recency tiebreak): protects the globally-hot experts, which recur
        // heavily in MoE routing. LRU threw them out when a burst of cold experts arrived,
        // dropping the measured hit rate from 57% to 40%.
        while self.bytes > self.budget && self.map.len() > 1 {
            if !self.evict_one() { break; }
        }
    }

    // A new entry starts at frequency one. The ordinary insert-then-evict
    // policy would reject it rather than remove any frequency>1 resident.
    // Preserve that decision when the pool needs space before insertion.
    fn evict_cold(&mut self) -> bool {
        if self.map.values().map(|(_, frequency, _)| *frequency).min() != Some(1) { return false; }
        self.evict_one()
    }

    fn evict_one(&mut self) -> bool {
        let victim = self.map.iter().min_by_key(|(_, (_, f, lu))| (*f, *lu)).map(|(k, _)| *k);
        let Some(victim) = victim else { return false; };
        let (bytes, _, _) = self.map.remove(&victim).unwrap();
        self.bytes -= bytes.len();
        true
    }

    // ---- double-buffer (2-deep gather ring) ----
    /// Allocate the staging buffers once (sized == moe_gs/us/ds). Idempotent.
    fn dbuf_ensure(&mut self, sizes: [usize; 3]) {
        if self.dbuf.is_none() {
            self.dbuf = Some(Dbuf {
                stage: [vec![0u8; sizes[0].max(64)], vec![0u8; sizes[1].max(64)], vec![0u8; sizes[2].max(64)]],
                staged: std::collections::HashMap::new(), handle: None, pf_hits: 0, pf_reads: 0,
            });
        }
    }
    /// Take the in-flight prefetch handle so the caller can join it (making staging safe to read).
    fn dbuf_take_handle(&mut self) -> Option<std::thread::JoinHandle<()>> {
        self.dbuf.as_mut().and_then(|d| d.handle.take())
    }
    fn dbuf_set_handle(&mut self, h: std::thread::JoinHandle<()>) {
        if let Some(d) = self.dbuf.as_mut() {
            // A live handle means a prefetch was spawned without the caller joining the
            // previous one; assigning over it would detach rather than join that thread,
            // leaving two writers on the staging buffers. The gather-then-prefetch call
            // order guarantees this is None; join defensively rather than detach.
            debug_assert!(d.handle.is_none(), "prefetch handle overwritten without join");
            if let Some(old) = d.handle.take() { let _ = old.join(); }
            d.handle = Some(h);
        }
    }
    /// Register the prefetch plan and hand back pread jobs (fd, off, len, dst) whose dst
    /// pointers alias into the staging buffers. `cand` = (key, kind, fd, disk_off, len)
    /// for the predicted-routed experts not already cache-resident (caller filtered).
    fn dbuf_stage_jobs(&mut self, sizes: [usize; 3], cand: &[(u64, usize, i32, i64, usize)]) -> Vec<(i32, i64, usize, usize)> {
        self.dbuf_ensure(sizes);
        let d = self.dbuf.as_mut().unwrap();
        d.staged.clear();
        let mut cursor = [0usize; 3];
        let mut jobs = Vec::with_capacity(cand.len());
        for &(key, kind, fd, off, len) in cand {
            if kind > 2 || d.staged.contains_key(&key) { continue; }
            let base = cursor[kind];
            if base + len > d.stage[kind].len() { continue; } // staging is sized == moe_gs, i.e. gather_cap experts
            cursor[kind] = base + len;
            d.staged.insert(key, (kind, base));
            let dst = d.stage[kind].as_mut_ptr() as usize + base;
            d.pf_reads += 1;
            jobs.push((fd, off, len, dst));
        }
        jobs
    }
    /// If `key` was prefetched into staging, return a pointer to its `len` bytes (else None).
    /// Bumps the prefetch-hit counter. Caller must have joined the prefetch handle first.
    fn dbuf_staged_ptr(&mut self, key: u64) -> Option<*const u8> {
        let d = self.dbuf.as_mut()?;
        let &(kind, off) = d.staged.get(&key)?;
        d.pf_hits += 1;
        Some(unsafe { d.stage[kind].as_ptr().add(off) })
    }
    /// (prefetch-hits, prefetch-reads) since last call; resets the counters.
    fn dbuf_report(&mut self) -> (u64, u64) {
        match self.dbuf.as_mut() { Some(d) => { let r = (d.pf_hits, d.pf_reads); d.pf_hits = 0; d.pf_reads = 0; r }, None => (0, 0) }
    }
}
impl Drop for ExpertCache {
    // A prefetch thread writes raw pointers into `dbuf.stage`; if a token was cancelled with a
    // prefetch still in flight, join it before the staging buffers are freed (no use-after-free).
    fn drop(&mut self) {
        if let Some(d) = self.dbuf.as_mut() { if let Some(h) = d.handle.take() { let _ = h.join(); } }
    }
}

/// qwen35moe (Qwen3.6-A3B) MoE FFN config: softmax router → top-k experts
/// (weights renormalized over the top-k) + an always-on shared expert scaled by
/// sigmoid(gate_inp_shexp·h). Ref: qwen35moe.cpp build_layer_ffn + build_moe_ffn.
#[derive(Clone, Copy)]
pub struct MoeConfig {
    n_expert: u32,   // 256
    n_used: u32,     // 8 (top-k)
    ffn_exp: u32,    // per-expert FFN width (512)
    ffn_shexp: u32,  // shared-expert FFN width (512)
}

/// glm-dsa (GLM-5.2 / DeepSeek-V3.2 class) config: Multi-head Latent Attention
/// (MLA) + DeepSeek Sparse Attention (DSA indexer) + grouped MoE. Ref: the glm-dsa
/// and deepseek2 arches.
/// MLA: q/kv are compressed through low-rank latents (q_lora/kv_lora); the KV
/// cache stores the kv_lora latent + a shared rope key (tiny KV). DSA: a learned
/// indexer scores KV positions and attention runs over the top_k only (the same
/// structure as our Quest page-sparse decode). MoE: experts in groups + shared
/// experts, with the first `leading_dense` layers being plain dense FFN.
#[derive(Clone, Copy)]
#[allow(dead_code)] // DSA-indexer + group-routing fields parsed from GGUF for models not yet run
pub struct MlaConfig {
    q_lora: u32,        // query down-projection rank (attn_q_a → q_lora)
    kv_lora: u32,       // kv down-projection rank (attn_kv_a_mqa → kv_lora latent; the KV cache)
    k_mla: u32,         // per-head key length (n_embd_head_k_mla); qk_nope = k_mla - qk_rope
    v_mla: u32,         // per-head value length (n_embd_head_v_mla)
    qk_rope: u32,       // rope dims per head (n_rot); shared rope key k_pe is qk_rope wide
    // grouped MoE
    n_expert: u32, n_used: u32, n_group: u32, group_used: u32,
    n_shared: u32,      // always-on shared experts
    ffn_exp: u32,       // per-expert FFN width
    leading_dense: u32, // first N layers are dense FFN, rest are MoE
    // learned sparse-attention indexer (top-k KV selection)
    indexer: IndexerConfig,
    // GLM-5.2 (glm-dsa) deltas vs deepseek2
    sigmoid_router: bool,   // DeepSeek-V3 sigmoid gating + exp_probs_b bias (else V2 softmax)
    routed_scale: f32,      // routed_scaling_factor applied to expert weights
    interleaved: bool,      // interleaved-pair RoPE (glm) vs NEOX split-half (deepseek2)
    absorb: bool,       // absorbed latent path (default) vs naive full-K/V; context capped at mla::SC_CAP
}

/// NextN/MTP (multi-token prediction) config: one extra trained draft block after
/// the main stack (blk.{layer}) predicting the token after next. Structure (ref
/// qwen35moe.cpp graph_mtp): combiner rmsnorm(emb)·enorm ‖ rmsnorm(h)·hnorm →
/// eh_proj(2d→d) → standard gated-attention block (own KV) → FFN → head.
#[derive(Clone, Copy)]
pub struct MtpConfig {
    layer: usize,        // blk index of the MTP block (== n_layers of the main stack)
    has_head: bool,      // nextn.shared_head_head present (else main lm_head)
    has_head_norm: bool, // nextn.shared_head_norm present (else output_norm)
    has_embed: bool,     // nextn.embed_tokens present (else main token_embd)
    /// Elements in nextn.hnorm. A hyper-connection model ships d*hc here (a per-stream
    /// gamma, like hc_attn_norm) and a plain one ships d; the two need different combiner
    /// inputs. Read from the tensor rather than inferred from the architecture.
    hnorm_len: usize,
}

/// Per-token scratch + KV/recurrent state arena. conv_state holds the last
/// (conv_kernel-1) tokens' conv input; ssm_state is the [S×S×H_v] matrix state
/// (qwen35, empty for non-SSM archs). attn_part: flash-decoding partials
/// [n_head][NWG][hd+2] f32. pmeta/plist: Quest-style page-sparse decode
/// (OJAS_SPARSE) — per attention layer [page][min|max][kvdim] half / [head][MAXSEL] u32.
///
/// Also the session-cache identity, the token list backing the current sequence state,
/// and the in-memory prefix cache (hybrid models): the KV cache stays addressable by
/// position across prefill calls, so rolling back to a common prefix needs only periodic
/// host snapshots of the small SSM/conv recurrent state, and the next prompt re-prefills
/// just the divergent tail after the longest matching prefix, avoiding O(N²) multi-turn
/// re-prefill. See prefill().
///
/// NextN/MTP draft block state (mtp None when the GGUF ships without it): mtp_h holds
/// the pre-output-norm hidden state captured during the main forward, mtp_cat the
/// [e_norm ‖ h_norm] combiner input (2d), mtp_tok the draft token id (GPU-side
/// draft→verify chaining), and ssm_snap/conv_snap the per-SSM-layer speculative rollback
/// snapshots (state after verify token 0).
pub(crate) struct StateArena {
    pub(crate) x: metal::Buffer,
    pub(crate) h: metal::Buffer,
    pub(crate) q: metal::Buffer,
    pub(crate) k: metal::Buffer,
    pub(crate) v: metal::Buffer,
    pub(crate) attn: metal::Buffer,
    pub(crate) tmp: metal::Buffer,
    /// Pooled token-id buffer for batched forwards (MAXM ids).
    ///
    /// Pooled rather than allocated per call: a Metal buffer allocation per batched
    /// forward is invisible in prefill (a handful of chunks) but costs ~3 ms per step on
    /// the speculative-verify critical path.
    pub(crate) tokbuf: metal::Buffer,
    /// Half-precision copy of the batched activation slab, for the async-copy GEMM
    /// (the DMA moves bytes; it cannot convert f32->half in flight). Sized as f32
    /// elements but holds MAXM*max_k halfs.
    pub(crate) xh: metal::Buffer,
    /// Q as f16 for attention_m_mma_dq (device-Q flash attention).
    pub(crate) qh: metal::Buffer,
    /// Split-K partial-sums scratch: up to 8 partitions x MAXM x d fp32.
    pub(crate) skbuf: metal::Buffer,
    /// Sectioned M-RoPE position descriptor for `rope_qk_store_m` buffer 14:
    /// `[s0,s1,s2,s3]` then `(t,h,w,e)` per token. Allocated once rather than per
    /// dispatch because `forward_chunk_enc` can be handed an external encoder whose
    /// command buffer is committed elsewhere, and `set_unretained_command_buffers(true)`
    /// means a temporary Buffer would not be retained by it. Written host-side like
    /// `st.x`; read only when the rope mode is non-zero, so text models never touch it.
    pub(crate) mpos: metal::Buffer,
    pub(crate) gate: metal::Buffer,
    pub(crate) up: metal::Buffer,
    pub(crate) act: metal::Buffer,
    pub(crate) ones: metal::Buffer, // all-1.0 weight (size max head_dim) for weightless V RMSNorm
    pub(crate) logits: metal::Buffer,
    pub(crate) kcache: Vec<metal::Buffer>,
    pub(crate) mla_lat: Vec<metal::Buffer>, // MLA absorbed: compressed latent KV [Lc|Rc] f16/token
    pub(crate) vcache: Vec<metal::Buffer>,
    pub(crate) conv_state: Vec<metal::Buffer>,
    pub(crate) ssm_state: Vec<metal::Buffer>,
    // qwen35 SSM decode scratch (sized to conv_channels / d_inner / dt_rank; 1 if no SSM).
    pub(crate) ssm_qkv: metal::Buffer,  // [conv_channels] mixed qkv → conv output
    pub(crate) ssm_z: metal::Buffer,    // [d_inner] gate z
    pub(crate) ssm_beta: metal::Buffer, // [dt_rank]
    pub(crate) ssm_gate: metal::Buffer, // [dt_rank]
    pub(crate) ssm_o: metal::Buffer,    // [d_inner] delta-net output → gated norm
    // qwen4exp hyper-connection scratch (sized d*hc / d / hc_low_rank / hc; 1 otherwise).
    pub(crate) hc_res: metal::Buffer,    // [d*hc] wide residual streams (persist across layers)
    pub(crate) hc_xn: metal::Buffer,     // [d*hc] per-stream normalized
    pub(crate) hc_lo: metal::Buffer,     // [hc_low_rank] low-rank vector
    pub(crate) hc_graw: metal::Buffer,   // [d*hc] up-projection before the sigmoid gate
    pub(crate) hc_gated: metal::Buffer,  // [d*hc] gated streams
    pub(crate) hc_mixed: metal::Buffer,  // [d] collapsed mixer output (one block's input)
    pub(crate) hc_inject: metal::Buffer, // [hc] per-stream injection weights
    pub(crate) ple_idx: metal::Buffer,   // constant [0..ple_n_heads) indices into packed staging
    pub(crate) ple_rows: metal::Buffer,  // [PLE layer][MAXM][heads][packed row], host-filled
    /// Diagnostic only (OJAS_EXPERT_HASH_LAYER): GPU-written hashes of the expert
    /// bytes reached through the address table, one per slot per weight kind.
    pub(crate) expert_hash_out: metal::Buffer,
    /// Diagnostic: the effective address each slot resolves to, chosen the same way the
    /// expert consumer chooses — the direct table on a direct layer, the scratch base
    /// plus slot*stride otherwise. Reading `direct_tables` unconditionally would pick up
    /// stale entries on a scratch layer.
    pub(crate) expert_hash_addr: metal::Buffer,
    pub(crate) attn_part: metal::Buffer,
    pub(crate) pmeta: Vec<metal::Buffer>,
    pub(crate) plist: metal::Buffer,
    // ---- qwen3vl ViT tower scratch (all `1` float when no mmproj is attached) ----
    // Sized for `VisionConfig::max_patches` patches, not MAXM tokens: the tower runs at
    // the full patch count (2304 for a 768x768 page) and its width is the mmproj's 768,
    // neither of which the decoder arena knows about. Row counts are padded to a multiple
    // of 32 because `gemm_mm_f16` stores whole 32-token tiles (cooperative
    // `simdgroup_store` cannot skip sub-tile rows) — the same constraint `batch.rs:51`
    // documents. Every one of these must be listed in `all_gpu_buffers`: residency-set
    // mode walks that list, and a buffer missing from it re-faults on every pass.
    /// `[channels * max_pixels]` f32, planar CHW — the `inp_raw` the tower reads.
    pub(crate) vimg: metal::Buffer,
    /// `[rows * channels*patch*patch]` f32 — `vit_patchify`'s im2col output.
    pub(crate) vrows: metal::Buffer,
    /// `[rows * d_v]` f32 residual stream. Also reinterpreted as
    /// `[rows/4 * 4*d_v]` for the projector, hence the max() in its sizing.
    pub(crate) vx: metal::Buffer,
    /// `[rows * d_v]` f32 — normed input, then the attention output (the ln1 value
    /// is dead once the qkv GEMM has consumed it).
    pub(crate) vh: metal::Buffer,
    /// `[rows * 3*d_v]` f32 — the fused `attn_qkv` output before the split.
    pub(crate) vqkv: metal::Buffer,
    pub(crate) vq: metal::Buffer,
    /// K/V as f16 for the attention kernels, which take `device const half*`.
    /// Allocated 32 rows long: `attention_m_mma_bidir_*` loads 8x8 K/V tiles at
    /// `p0 < total`, so it reads up to 7 rows past `total` and those rows must hold
    /// finite values. Zeroed at load and only ever written with finite data, so the
    /// tail is finite for every image size.
    pub(crate) vkh: metal::Buffer,
    pub(crate) vvh: metal::Buffer,
    /// `[rows * ffn_v]` f32 — the MLP hidden state, reused for `mm.0`'s output
    /// (`mm_hidden == ffn_v == 4*d_v` here, over a quarter of the rows).
    pub(crate) vffn: metal::Buffer,
    /// `[max_patches * d_v]` f32 — the bilinearly-resized position embedding,
    /// already in permuted token order. Host-written per image.
    pub(crate) vpe: metal::Buffer,
    /// `[4 + 4*max_patches]` u32 — the M-RoPE descriptor `kernels::ops::mrope_desc`
    /// builds: four section sizes then `(t,h,w,e)` per token. Host-written.
    pub(crate) vmpos: metal::Buffer,
    /// `[rows/4 * proj_dim]` f32 — `mm.2`'s output, what the decoder consumes.
    pub(crate) vout: metal::Buffer,
    pub(crate) max_seq: usize,
    /// Independent sequences this arena holds state for (`OJAS_SLOTS`, ≤ [`MAX_SLOTS`]).
    /// The default of 1 is byte-identical to no slots at all: every slot-strided buffer
    /// is allocated at 1x and every slot offset computes to 0, so the single-sequence
    /// paths encode the dispatches they always did.
    ///
    /// It multiplies five buffers — `conv_state`, `ssm_state`, `kcache`, `vcache` and
    /// `attn_part`, the per-sequence state. Everything else in the arena is already
    /// `MAXM`-sized scratch, and since `MAX_SLOTS <= MAXM` a batch of B slots is just B
    /// rows of buffers that already hold 256.
    pub(crate) slots: usize,
}

impl StateArena {
    /// Byte stride between slots of `kcache[l]` / `vcache[l]`. Derived by dividing the
    /// allocation rather than recomputing `max_seq * kvdim`, because the MLA-absorbed
    /// path allocates these at length 1 as placeholders and a recomputed stride would
    /// address past the end of one.
    fn kv_stride(&self, l: usize) -> u64 { self.kcache[l].length() / self.slots as u64 }
    /// Byte stride between slots of `conv_state[l]`. Divided out of the allocation
    /// because a PLE layer appends n-gram history to its conv state, so the stride is
    /// not `(conv_k - 1) * conv_ch` for every layer.
    fn conv_stride(&self, l: usize) -> u64 { self.conv_state[l].length() / self.slots as u64 }
    fn ssm_stride(&self, l: usize) -> u64 { self.ssm_state[l].length() / self.slots as u64 }
    /// Byte stride between slots of the flash-decoding partials. The one buffer in the
    /// arena that is not already batch-sized: it is `[n_head][NWG][hd+2]` for a single
    /// query, and B concurrent slot dispatches each need their own.
    fn part_stride(&self) -> u64 { self.attn_part.length() / self.slots as u64 }
}

/// Resident weights (f16 raw for matmuls, f32 for norms) + quant-mode flags.
/// w4/scale4: native Q4 (per-32-block symmetric Q4_0): nibbles + f16 scale;
/// w = scale*(nib-8). w20/s20: ternary Q2_0 (g128), repacked at load — codes
/// split from scales so the kernel can do 16-byte loads (the on-disk 34-byte
/// block stride is not a multiple of 4, which forces byte-at-a-time reads and
/// makes the GEMV compute-bound: measured 72 GB/s of a 400 GB/s machine).
/// wcache: mmap'd weight cache (must outlive the zero-copy Metal buffers).
pub(crate) struct Weights {
    pub(crate) q8: bool,
    #[allow(dead_code)] // OJAS_GEMV_R4 variant; parsed, path not yet wired
    pub(crate) gemv_r4: bool,
    pub(crate) w16: HashMap<String, metal::Buffer>,
    pub(crate) w32: HashMap<String, metal::Buffer>,
    pub(crate) w8: HashMap<String, metal::Buffer>,
    pub(crate) scale8: HashMap<String, metal::Buffer>,
    pub(crate) q4: bool,
    pub(crate) q4k: bool,
    pub(crate) q20: bool,
    pub(crate) w_off: HashMap<String, u64>,
    pub(crate) w_qtype: HashMap<String, u32>,
    /// Logical GEMM shape (K, N) = (dims[0], product(dims[1..])) per 2D weight, from
    /// the GGUF header. gemm_named/projm debug_assert the caller's (k, n) against this,
    /// so a dispatch that passes `d` where the weight is really `n_head*hd` — silently
    /// correct only while the two coincide — trips in tests instead of surfacing as
    /// garbage on the first model where the dims diverge.
    pub(crate) wshape: HashMap<String, (u32, u32)>,
    pub(crate) mapped: Option<crate::weights::MappedGguf>,
    pub(crate) ple: Option<ple::PleTable>, // CPU-only mapping; never in a Metal residency set
    pub(crate) w4: HashMap<String, metal::Buffer>,
    pub(crate) scale4: HashMap<String, metal::Buffer>,
    /// Native quantized rows, any GGUF type: the GGUF blocks themselves, mapped
    /// zero-copy, decoded inside the matvec by `gemv_nat_*`. The type is in `w_qtype`
    /// and any page-alignment slack in `w_off`.
    ///
    /// This path keeps a low-bit model low-bit. Requantizing to Q8 costs 8 bits/weight
    /// whatever the file held, expanding an IQ2 model ~4x in GPU memory.
    pub(crate) wq: HashMap<String, metal::Buffer>,
    pub(crate) w4k: HashMap<String, metal::Buffer>,
    /// Native Q6_K rows (GGUF type 14) — the other half of every `*_K_M` file.
    pub(crate) w6k: HashMap<String, metal::Buffer>,
    /// Q4L: Q4_K's exact values relaid out for the tuned Q4 kernel shape —
    /// contiguous nibbles plus per-32-block f32 `d1` and `-m1`.
    pub(crate) w4l: HashMap<String, metal::Buffer>,
    pub(crate) q4l_a: HashMap<String, metal::Buffer>,
    pub(crate) q4l_b: HashMap<String, metal::Buffer>,
    pub(crate) w20: HashMap<String, metal::Buffer>,   // codes: n*nblk*32 B, 32 B/block
    pub(crate) s20: HashMap<String, metal::Buffer>,   // scales: n*nblk f16
    pub(crate) wcache: Option<crate::weights::MappedCache>,
}

impl Weights {
    /// The single place that knows the probe order. Mirrors `mm()`'s if-let chain
    /// exactly; `repr_matches_dispatch` in tests/ asserts they stay in step.
    pub(crate) fn repr(&self, name: &str) -> Repr {
        if self.wq.contains_key(name) { return Repr::Native; }
        if self.w20.contains_key(name) { return Repr::Q20; }
        if self.w4k.contains_key(name) { return Repr::Q4K; }
        if self.w4l.contains_key(name) { return Repr::Q4L; }
        if self.w6k.contains_key(name) { return Repr::Q6K; }
        if self.w4.contains_key(name) { return Repr::Q4; }
        if self.q8 && self.w8.contains_key(name) { return Repr::Q8; }
        if self.w8.contains_key(name) { return Repr::Q8; }
        if self.w16.contains_key(name) { return Repr::F16; }
        if self.w32.contains_key(name) { return Repr::F32; }
        Repr::Missing
    }

    /// Histogram of representations, plus any weight that resolves ambiguously
    /// (present in more than one primary map, so the probe order is silently
    /// deciding) or not at all. Used by `repr_gate`.
    pub(crate) fn audit(&self, names: &[String]) -> (Vec<(String, usize)>, Vec<String>, Vec<String>) {
        use std::collections::BTreeMap;
        let mut hist: BTreeMap<String, usize> = BTreeMap::new();
        let (mut dup, mut missing) = (vec![], vec![]);
        for n in names {
            let r = self.repr(n);
            *hist.entry(format!("{r:?}")).or_default() += 1;
            if !r.present() { missing.push(n.clone()); continue; }
            // count membership across the primary maps (the ones repr() ranks)
            let c = [self.wq.contains_key(n), self.w20.contains_key(n), self.w4k.contains_key(n),
                     self.w4l.contains_key(n), self.w6k.contains_key(n), self.w4.contains_key(n),
                     self.w8.contains_key(n), self.w16.contains_key(n), self.w32.contains_key(n)]
                .iter().filter(|b| **b).count();
            if c > 1 { dup.push(n.clone()); }
        }
        (hist.into_iter().collect(), dup, missing)
    }

    /// gate/up as the fused FFN kernels see them.
    pub(crate) fn fused_repr_ffn(&self, p: &dyn Fn(&str) -> String) -> Option<Repr> {
        self.fused_repr(&[&p("ffn_gate.weight"), &p("ffn_up.weight")])
    }

    /// The representation shared by a group of weights a fused kernel reads in one
    /// dispatch, or None when they disagree or any is missing.
    ///
    /// Each fused kernel walks one hard-coded layout, so the precondition is that every
    /// weight in the group has the same representation.
    pub(crate) fn fused_repr(&self, names: &[&str]) -> Option<Repr> {
        let first = self.repr(names[0]);
        if !first.present() || first == Repr::F32 { return None; }
        names[1..].iter().all(|n| self.repr(n) == first).then_some(first)
    }
}

/// GLM/DeepSeek disk-streamed MoE state. stream: expert tensors are not mapped
/// upfront (439GB >> 77GB Metal working set → allocations fail). Store metadata
/// only; create no-copy buffers per chunk in stream_bind() and drop them in
/// stream_clear() so resident no-copy bytes stay small. Gather path: routed-expert
/// weights memcpy'd from mmap into small packed scratch (slots 0..n_used-1) so
/// Metal only ever wires ~n_used experts, not all 256. Plus the temporal expert
/// prefetcher and the streaming predictive prefetch + freq pinning (weights.rs).
pub(crate) struct MoeStream {
    pub(crate) stream: bool,
    /// Partial/full residency: layers `[0, resident_layers)` hold every expert as
    /// a full zero-copy device buffer (`wt.wq`), wired by `expert_residency`, so
    /// their expert kernels index the tensor by the router's own moe_idx and need
    /// no host gather — consecutive resident layers share a command buffer. Layers
    /// `[resident_layers, n_layers)` stream as before. `stream` stays true either
    /// way. 0 = pure streaming; n_layers = full residency.
    pub(crate) resident_layers: usize,
    pub(crate) stream_meta: HashMap<String, (usize, u64, u64, u32)>, // name → (part, abs, rawlen, ggml_type)
    #[allow(dead_code)]
    pub(crate) stream_bufs: std::cell::RefCell<HashMap<String, (metal::Buffer, u64)>>, // current chunk's live bufs
    pub(crate) moe_gs: metal::Buffer, pub(crate) moe_us: metal::Buffer, pub(crate) moe_ds: metal::Buffer, // packed gate/up/down
    /// Where each (token, k) routed expert landed in the packed scratch:
    /// [ubatch][n_used] u32 slots. The M tokens' expert sets overlap, so the
    /// gather packs their union once and this maps each token's k-th pick onto the
    /// slot that holds it. At M=1 with a distinct in-range top-k it is the identity.
    pub(crate) moe_slot: metal::Buffer,
    /// Per-union-slot GPU addresses; misses point into packed scratch.
    pub(crate) direct_tables: [metal::Buffer; 3],
    /// Retain all indirectly referenced buffers until the next completed gather.
    pub(crate) direct_live: std::cell::RefCell<Vec<std::sync::Arc<expert_pool::MetalExpert>>>,
    /// Diagnostic (OJAS_EXPERT_HASH_LAYER): what the host believes each table slot
    /// points at, recorded at gather time so the GPU's view can be compared with it.
    pub(crate) expert_hash_records: std::cell::RefCell<Vec<audit::SlotRecord>>,
    /// Retained for its lifetime effect: keeps the resident-mode expert buffers
    /// wired so they do not re-fault every token. `None` in streaming mode or when
    /// the OS lacks MTLResidencySet.
    pub(crate) expert_residency: Option<ojas_metal::ResidencySet>,
    pub(crate) direct_layer: std::cell::Cell<Option<usize>>,
    /// Experts the packed scratch has room for: min(n_expert, n_used * ubatch).
    pub(crate) gather_cap: usize,
    /// Tokens a streamed batched forward may carry (`--ubatch-size`), captured at
    /// load because it is what `gather_cap` was sized from.
    pub(crate) ubatch: usize,
    pub(crate) expert_cache: std::cell::RefCell<ExpertCache>, // userspace LRU of gathered experts
    pub(crate) gather_hits: std::cell::Cell<u64>, pub(crate) gather_reads: std::cell::Cell<u64>, // hit-rate telemetry
    /// OJAS_EXPERT_STATS: (layer,expert)→routed-hit census for pack planning
    pub(crate) expert_stats: std::cell::RefCell<std::collections::HashMap<(u32, u32), u64>>,
    pub(crate) prefetch: Option<crate::weights::ExpertPrefetcher>,
    pub(crate) stream_prefetch: Option<crate::weights::StreamPrefetcher>,
}

/// MoE decode scratch: router logits [n_expert], per-layer top-k indices
/// [n_layers × n_used] (read back for the expert prefetcher), weights, expert
/// activations [n_used × ffn_exp], shared-expert gate scalar; plus the batched
/// prefill variants (chunked prefill: all M rows' expert work in flight).
/// route_lg (OJAS_ROUTE_STATS=1): per-layer router-logit snapshots
/// [n_layers × n_expert] read back per token (routing-agreement telemetry).
pub(crate) struct MoeScratch {
    pub(crate) moe_lg: metal::Buffer, pub(crate) moe_idx: metal::Buffer, pub(crate) moe_wgt: metal::Buffer,
    pub(crate) route_lg: Option<metal::Buffer>,
    pub(crate) moe_act: metal::Buffer, pub(crate) moe_sh: metal::Buffer,
    pub(crate) moe_blg: metal::Buffer, pub(crate) moe_bidx: metal::Buffer, pub(crate) moe_bwgt: metal::Buffer,
    pub(crate) moe_bact: metal::Buffer, pub(crate) moe_bsh: metal::Buffer, pub(crate) moe_bg: metal::Buffer,
    pub(crate) moe_bu: metal::Buffer, pub(crate) moe_btmp: metal::Buffer,
}

pub(crate) struct SpecState {
    pub(crate) mtp: Option<MtpConfig>,
    pub(crate) mtp_h: metal::Buffer,
    pub(crate) mtp_hprev: metal::Buffer,
    pub(crate) mtp_cat: metal::Buffer,
    /// The draft block's own hidden state, so a chained draft can condition on it
    /// rather than the target's, which lets one trained head produce more than one
    /// token per step.
    pub(crate) mtp_chain: metal::Buffer,
    pub(crate) mtp_tok: metal::Buffer,
    /// Which mtp_h row the next draft conditions on. Kept here rather than passed in by
    /// callers, because a stale row costs acceptance and nothing else, so getting it
    /// wrong is easy to miss.
    pub(crate) hrow: std::cell::Cell<usize>,
    pub(crate) ssm_snap: Vec<metal::Buffer>,
    pub(crate) conv_snap: Vec<metal::Buffer>,
    pub(crate) snapshot_rows: usize,
    pub(crate) verified_rows: std::cell::Cell<usize>,
}

pub(crate) struct Session {
    pub(crate) model_name: String,
    /// Prompt-prefix blocks shared by every slot (`prefix_cache`).
    pub(crate) cache: std::cell::RefCell<prefix_cache::PrefixCache>,
    /// Spans reusable at any position (`doc_cache`), shared by every slot.
    pub(crate) docs: std::cell::RefCell<doc_cache::DocCache>,
    /// Each slot's sequence, indexed by slot.
    pub(crate) seqs: Vec<SeqState>,
}

/// What the decoder tracks about the sequence one slot holds.
pub(crate) struct SeqState {
    pub(crate) session_tokens: std::cell::RefCell<Vec<u32>>,
    pub(crate) last_prefill_reused: std::cell::Cell<usize>,
    pub(crate) snap_pos: std::cell::RefCell<Vec<usize>>,
    pub(crate) snap_buf: std::cell::RefCell<Vec<Vec<u8>>>,
    /// What the prefill in progress may add to the prefix cache.
    pub(crate) cache_plan: std::cell::RefCell<Option<prefix_cache::Plan>>,
    /// Whether requests may restore from the prefix cache (they always add to it).
    pub(crate) cache_reuse: std::cell::Cell<bool>,
    /// Token positions of boundaries in the next prompt, kept as snapshot points.
    pub(crate) cache_marks: std::cell::RefCell<Vec<usize>>,
    /// What the cache did for the most recent prompt.
    pub(crate) cache_last: std::cell::Cell<ojas_core::PrefixRestore>,
    /// Token ranges of the next prompt for the document cache, and whether it may
    /// serve them.
    pub(crate) cache_docs: std::cell::RefCell<Vec<(usize, usize)>>,
    pub(crate) docs_serve: std::cell::Cell<bool>,
    /// Ranges of the prompt in progress to add to the document cache once processed.
    pub(crate) docs_pending: std::cell::RefCell<Vec<(usize, usize)>>,
}

impl Default for SeqState {
    fn default() -> Self {
        SeqState {
            session_tokens: Default::default(), last_prefill_reused: Default::default(), snap_pos: Default::default(),
            snap_buf: Default::default(), cache_plan: Default::default(), cache_reuse: std::cell::Cell::new(true),
            cache_marks: Default::default(), cache_last: Default::default(), cache_docs: Default::default(),
            docs_serve: Default::default(), docs_pending: Default::default(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct LayerPlan {
    n_head: u32,
    n_kv: u32,
    head_dim: u32,
    qdim: u32,       // n_head * head_dim
    kvdim: u32,      // n_kv * head_dim
    rope_base: f32,  // per-layer (gemma dual-rope: local vs global)
    scale: f32,      // attention scale = 1/sqrt(head_dim)
    kv_source: usize,// layer whose k/v cache to read (== self unless KV-shared)
    has_v: bool,     // computes its own V (false = gemma-4 shared layer, reuse source's V)
    is_ssm: bool,    // qwen35 Gated-DeltaNet: this layer is an SSM block, not attention
}

/// Which representation a weight is actually stored in.
///
/// Weights live in ~14 parallel `HashMap<String, Buffer>`, one per representation, and
/// every dispatch site used to rediscover the representation by probing those maps in
/// priority order — ~280 such probes across the decoder, where a site that does not know
/// about a newly added map takes a wrong branch instead of failing. Adding the
/// native-quant map (`wq`) broke five sites that way: `batched_dense_ok` returned false
/// and silently disabled batched prefill (a ~4x path, so its gate compared the per-token
/// path against itself), `profile_batch` recorded 64 dispatches where 16 did work
/// (understating per-call cost by up to 3.8x), and the fused qkv/ffn sites and
/// graph_chunk's M=2 path panicked on a missing key.
///
/// So the probe order lives in one place, `Weights::repr`, and decision sites `match` on
/// this enum: adding a representation adds a variant, and every non-exhaustive match
/// becomes a compile error instead of a silent wrong branch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Repr {
    /// The GGUF's own blocks, decoded in-kernel (`gemv_nat_*`).
    Native,
    /// Q4_K values in the tuned Q4L layout.
    Q4L,
    /// Native Q4_K blocks.
    Q4K,
    /// Native Q6_K blocks.
    Q6K,
    /// Ternary Q2_0 (g128).
    Q20,
    /// int8 + one f32 scale per row.
    Q8,
    /// 4-bit + per-block scales (q4mode).
    Q4,
    F16,
    F32,
    /// Not in any map — the caller must not dispatch.
    Missing,
}

impl Repr {
    /// True when a matvec can be dispatched for this weight at all.
    pub(crate) fn present(self) -> bool { self != Repr::Missing }
}

/// Model state + host orchestration, generic over the GPU backend.
/// The dispatch/encode paths are Metal today, so all methods live in
/// `impl DecoderGpu<'a>`; the parameter is the socket where a second
/// backend plugs in as its ops are implemented.
pub struct DecoderGpu<'a> {
    flash_trace: std::cell::RefCell<Option<profile::FlashTrace>>,
    gpu: &'a MetalGpu,
    cfg: EngineConfig, // env knobs, read once at load (frozen names)
    pub d: usize,
    arch: Arch,
    // pipelines
    p: HashMap<String, ComputePipelineState>,
    wt: Weights,
    st: StateArena,
    ms: MoeScratch,
    strm: MoeStream,
    sp: SpecState,
    sess: Session,
    tune: Tune,
    gpu_s: std::cell::Cell<f64>, // accumulated GPU execution time
    /// When set, forward_id also encodes a top-8 selection over the logits into the
    /// same command buffer (tmp[1..9]). A separate command buffer for it measured
    /// ~5 ms/token of commit+wait overhead — 60% of a whole forward.
    want_topk: std::cell::Cell<bool>,
    /// Which sequence slot the single-sequence graphs address.
    ///
    /// `decode_slots` names its slot per dispatch and does not read this. The readers are
    /// the paths written for one sequence that must still be pointable at slot s: the
    /// chunked prefill graph, `reset_state`, the snapshot capture/restore. They bind their
    /// state buffers at `slot_*_off()` and this Cell says where that is, rather than
    /// threading a slot argument through all of them (and through `Model::prefill`, which
    /// has no slot to give).
    ///
    /// A Cell rather than an argument because `DecoderGpu` is `!Sync` and single-threaded
    /// by construction: the batch is a data dimension on one thread, as `MAXM` already is.
    /// Every writer restores the previous value, so the resting state is 0 and the default
    /// paths are unchanged.
    cur_slot: std::cell::Cell<usize>,
    /// Row buffers of the text encoder, reused across requests (`text_encoder.rs`).
    text_arena: std::cell::RefCell<Option<text_encoder::TextBuffers>>,
}

impl Drop for DecoderGpu<'_> {
    fn drop(&mut self) {
        // Detach queue residency before fields release their no-copy mappings.
        if let Some(residency) = self.strm.expert_residency.take() {
            self.gpu.set_unretained_command_buffers(false);
            drop(residency);
        }
    }
}

/// Arch/config scalars plus the per-arch feature configs (MoE/SSM/MLA/MTP live in
/// their own state groups).
/// Learned sparse-attention indexer, shared by every architecture that scores KV
/// positions and attends over a retained top-k (deepseek/glm DSA, qwen4exp QSA).
/// Read from `{arch}.attention.indexer.{head_count,key_length,top_k}`.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct IndexerConfig {
    pub(crate) n_head: u32,    // indexer query heads
    pub(crate) head_size: u32, // per-head key/query width
    pub(crate) top_k: u32,     // KV blocks retained per query
}

/// Parameters specific to the qwen4exp family with no analog in the dense, SSM, or
/// MoE configs. Gated-DeltaNet layers reuse [`SsmConfig`], the mixture of experts
/// reuses [`MoeConfig`], and the sparse attention reuses [`IndexerConfig`]; this
/// struct holds only the hyper-connection and per-layer n-gram embedding
/// parameters that those do not cover.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct Qwen4ExpConfig {
    /// Number of parallel residual streams. Between blocks the residual state is
    /// `hc_mult` copies of the `d`-wide hidden state; each block reads one mixed
    /// view and writes back through per-stream injection weights.
    pub(crate) hc_mult: u32,
    /// Rank of the hyper-connection mixer's down- and up-projections.
    pub(crate) hc_low_rank: u32,
    /// Sparse-attention indexer for the full-attention layers.
    pub(crate) indexer: IndexerConfig,
    /// Per-layer n-gram embedding: n-gram order, heads per n-gram, causal-conv
    /// width, and per-layer embedding width. `ple_layers` lists the layers that
    /// carry an n-gram table; an empty list means the feature is absent.
    pub(crate) ple_ngram_size: u32,
    pub(crate) ple_heads_per_ngram: u32,
    pub(crate) ple_conv_kernel: u32,
    pub(crate) ple_head_dim: u32,
    pub(crate) ple_layers: Vec<u32>,
    /// n-gram hash inputs: `(ngram_size-1)*heads_per_ngram` heads; the per-position
    /// multipliers (up to 2^45, so u64), per-head vocab sizes and row offsets, and
    /// the EOS token that resets an n-gram window at a segment boundary.
    pub(crate) ple_n_heads: u32,
    pub(crate) ple_eos: u32,
    pub(crate) ple_multipliers: Vec<u64>,
    pub(crate) ple_head_offsets: Vec<u32>,
    pub(crate) ple_head_vocab_sizes: Vec<u32>,
}

/// qwen3vl ViT tower + `qwen3vl_merger` projector geometry, from the `clip.*` KV an
/// mmproj contributes through `Gguf::attach_with_meta`. Carried on [`Arch`] as
/// `moe`/`ssm`/`mla`/`qwen4exp` are.
///
/// Deliberately not a `Vec<LayerPlan>`: `LayerPlan` carries a KV-cache geometry, a rope
/// base and a KV-sharing source, and `load.rs` allocates a K and a V cache per plan
/// (`load.rs:1367`) — twelve caches nothing would write, for a tower whose K and V are
/// per-layer temporaries. The tower's layers are uniform, so one struct describes all
/// twelve.
#[derive(Clone, Copy)]
pub(crate) struct VisionConfig {
    /// `clip.vision.embedding_length` (768). The tower's width, not the decoder's.
    pub(crate) d: u32,
    pub(crate) layers: u32,
    pub(crate) n_head: u32,
    /// `d / n_head` (64). The M-RoPE sections are `hd/4` each, so `hd % 4 == 0`.
    pub(crate) hd: u32,
    pub(crate) ffn: u32,
    pub(crate) patch: u32,
    /// Colour planes in `inp_raw` (3). From `v.patch_embd.weight`'s own shape.
    pub(crate) channels: u32,
    /// Side of the learned position grid, `image_size / patch_size` (48). A patch
    /// grid of exactly this size takes `resize_position_embeddings`' early return.
    pub(crate) pos_side: u32,
    /// `clip.vision.spatial_merge_size` (2). Patches merge 2x2 for the projector.
    pub(crate) merge: u32,
    /// `clip.vision.projection_dim` (1024) == the decoder's embedding width.
    pub(crate) proj_dim: u32,
    /// `mm.0`'s output width (3072) == `d * merge^2`, read from the tensor rather
    /// than computed.
    pub(crate) mm_hidden: u32,
    /// `clip.vision.attention.layer_norm_epsilon` (1e-6) — not the decoder's eps.
    pub(crate) eps: f32,
    /// `GGML_ROPE_TYPE_VISION` freq base (10000), `qwen3vl.cpp:106`.
    pub(crate) rope_base: f32,
    /// Patches the vision scratch arena is sized for. Larger images still encode:
    /// `encode_image` falls back to on-demand temporaries, the same `mk`/`big`
    /// idiom `forward_diffusion_range` uses past `MAXM` (`batch.rs:44`). Defaults
    /// to `pos_side^2` (a 768x768 page); `OJAS_VIT_MAX_PATCHES` overrides.
    pub(crate) max_patches: u32,
}

/// A ModernBERT text encoder (`general.architecture = "modern-bert"`), read from the
/// same GGUF keys llama.cpp reads. Carried on [`Arch`] as `vision` is: the encoder
/// runs through its own entry (`text_encoder.rs`), not the decoder graph, because it
/// has no KV cache, no causal mask and no LM head.
#[derive(Clone, Debug)]
pub(crate) struct TextEncoderConfig {
    pub(crate) d: u32,
    pub(crate) layers: u32,
    pub(crate) n_head: u32,
    pub(crate) hd: u32,
    /// Per half of the gated MLP: `ffn_up` produces `2 * ffn` columns.
    pub(crate) ffn: u32,
    pub(crate) eps: f32,
    /// RoPE base of the global-attention layers.
    pub(crate) rope_base: f32,
    /// RoPE base of the sliding-window layers.
    pub(crate) rope_base_local: f32,
    /// Keys a local layer's query sees on each side: `|i - j| <= window`.
    /// `attention.sliding_window` is the full width (128), this is half of it.
    pub(crate) window: u32,
    /// Layer `l` is local when `l % swa_pattern != 0` (llama.cpp's dense-first
    /// rule); 0 means every layer is global.
    pub(crate) swa_pattern: u32,
    pub(crate) max_positions: u32,
    /// The Laya decision head, when the file carries one.
    pub(crate) laya: Option<LayaHeadConfig>,
}

impl TextEncoderConfig {
    pub(crate) fn is_local(&self, layer: usize) -> bool {
        self.swa_pattern > 0 && self.window > 0 && layer as u32 % self.swa_pattern != 0
    }
}

/// Geometry of the Laya decision head appended by `scripts/laya_convert.py`: a stack
/// of PyTorch `nn.TransformerEncoderLayer` blocks (pre-norm, ReLU, biased, no RoPE).
#[derive(Clone, Debug)]
pub(crate) struct LayaHeadConfig {
    pub(crate) blocks: u32,
    pub(crate) n_head: u32,
    pub(crate) ffn: u32,
    pub(crate) eps: f32,
}

pub(crate) struct Arch {
    pub(crate) n_layers: usize,
    pub(crate) n_head: usize,
    pub(crate) n_kv: usize,
    pub(crate) hd: usize,
    pub(crate) ffn: usize,
    pub(crate) layers: Vec<LayerPlan>, // per-layer plan (reusable, arch-agnostic forward)
    pub(crate) vocab: usize,
    pub(crate) rope_base: f32,
    pub(crate) eps: f32,
    pub(crate) lm_head: String, // "token_embd.weight" (tied) or "output.weight" (untied)
    pub(crate) qkv_bias: bool,  // Qwen2 has q/k/v bias; Llama/Qwen3/Gemma don't (→ fast split-qkv)
    pub(crate) qk_norm: bool,   // Qwen3/Gemma: per-head RMSNorm on q,k before RoPE (attn_q_norm/attn_k_norm)
    pub(crate) rope_neox: bool, // true=split-half (Qwen/NeoX); false=interleaved (LLaMA GGUF permuted weights)
    pub(crate) embed_scale: f32,// Gemma: multiply input embedding by sqrt(d) (1.0 otherwise)
    pub(crate) sandwich: bool,  // Gemma: extra RMSNorm on attn/ffn output before residual (post_*_norm)
    pub(crate) gelu: bool,      // Gemma: GeLU FFN activation (else SiLU)
    #[allow(dead_code)] // Gemma3 sliding-window rope base; parsed, not yet used
    pub(crate) rope_local: f32, // Gemma3: rope base for sliding-window layers (0.0 = single base)
    #[allow(dead_code)]
    pub(crate) swa_pattern: u32,// Gemma3: layer l is sliding/local unless (l+1)%swa_pattern==0 (0 = none)
    pub(crate) v_rmsnorm: bool, // Gemma4: weightless per-head RMSNorm on V (all layers)
    pub(crate) gpt_oss: bool,   // GPT-OSS: biased GQA + per-head attn sinks + biased SwiGLU-OAI MoE
    pub(crate) out_scale: Vec<f32>, // Gemma4: per-layer output scalar (layer_output_scale); empty = none
    pub(crate) moe: Option<MoeConfig>, // qwen35moe MoE FFN config (None for dense archs)
    pub(crate) ssm: Option<SsmConfig>, // qwen35 Gated-DeltaNet (None for non-SSM archs)
    pub(crate) mla: Option<MlaConfig>, // deepseek2/glm-dsa Multi-head Latent Attention
    pub(crate) qwen4exp: Option<Qwen4ExpConfig>, // hyper-connection + sparse-indexer + n-gram embedding params
    /// qwen3vl ViT tower, when an mmproj sidecar was attached (None otherwise).
    pub(crate) vision: Option<VisionConfig>,
    /// ModernBERT text encoder, when the file is one (None for a decoder).
    pub(crate) text_encoder: Option<TextEncoderConfig>,
    pub(crate) sparse_budget: Option<u32>,
}

/// Device-adaptive tuning (filled by autotune() at load). gemv_plan maps a shape
/// (K,N) -> config; pseudo-keys (0,1)=ffn_gu threads, (0,2)=attention threads.
pub(crate) struct Tune {
    pub(crate) max_tg: u64, // device max threads/threadgroup (clamp target)
    pub(crate) gemv_plan: HashMap<(u32, u32), GemvPlan>,
    /// Lanes-per-row for the lane-partitioned verify matvec, per (K,N). 0 = the
    /// row-blocked kernel wins on that shape. Autotuned per device because the choice
    /// depends on how many threadgroups the shape yields against how many the GPU needs
    /// to saturate, which is what changes from a 7-core M1 to an 80-core Ultra.
    pub(crate) xv_plan: HashMap<(u32, u32), u32>,
    /// Threadgroup size for the Q4L gemv family, per (K,N); replaces a single hardcoded
    /// 256 that applied to every shape on every device. Pseudo-keys: (0,3) = ffn_gu_q4l,
    /// (0,4) = qkv_q4l — fused kernels with no single (K,N). Decode dispatches serially,
    /// so an isolated micro-benchmark is representative here; the batched path is not
    /// tunable this way (see the concurrency caveat on xv_plan).
    pub(crate) q4l_tg: HashMap<(u32, u32), u32>,
}

// A chosen GEMV execution plan for a given (K,N) shape, picked by autotune per device.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GemvPlan {
    ksplit: bool, // K-split kernel (1 row/tg, simdgroups split K) vs standard (rows/tg)
    threads: u64, // threads per threadgroup
}

/// What a batched forward should produce for the logits.
///
/// `Ids` runs the lm_head and an on-device row-wise argmax but copies back nothing but
/// the ids; `Host` additionally copies vocab*M floats back. Speculative verify depends
/// on `Ids` being affordable — see forward_batch_ids.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LogitsOut { None, Ids, IdsTopK, Host }

pub const MAXM: usize = 256; // max tokens per batched forward (speculative verify / prefill chunk)

/// Hard cap on independent sequences decoded per step (`OJAS_SLOTS`).
///
/// 4 is measured. Slot batching amortizes the per-token weight read across the slots,
/// so the cap is set by where the M-row GEMV family stops improving — and on this GPU it
/// does not merely flatten, it regresses:
///
/// ```text
///   M:      1     2     3     4     6     8
///   Q4L: 1.00x 1.34x 1.53x 1.84x 1.52x 1.54x
/// ```
///
/// That is the register-occupancy cliff `kernels/gemv.rs` documents: past ~4 rows the
/// accumulators and the dequantized block stop fitting. All three kernel routings
/// (`OJAS_BR_XV` = 0 / 8 / default) peak at 4, so it is a property of the machine rather
/// than of which kernel `dispatch.rs` routes to. F16 keeps creeping to M=8 (2.24x ->
/// 2.40x), but that is 7% for twice the per-sequence state.
///
/// The table is Q4L, and not every model uses it: surya-2 ships F16 and requantizes to
/// plain `Q4`, never Q4L (`repr_gate`: 188 Q4, zero Q4L), and plain Q4 owns no 4-row
/// kernel at all — see `q4_pair` in `graph_decode.rs` for the cliff that caused and how
/// the slot graph works around it. With that workaround the measured curve on
/// surya-2/prec 2 is monotonic to 4 (1.44x, 1.54x, 1.96x at B=2,3,4), by a different
/// mechanism: pairing makes the per-token weight-read term flat in B, so what keeps
/// improving past B=2 is the per-step fixed cost (submit, barrier serialization, CPU
/// encode) divided among more tokens. That amortization has diminishing returns while
/// per-sequence state cost stays linear, the second, independent reason not to go
/// past 4.
///
/// Raising this constant does not trade memory for speed; it trades memory for less
/// speed.
pub const MAX_SLOTS: usize = 4;


fn buf(gpu: &MetalGpu, floats: usize) -> metal::Buffer {
    gpu.device
        .new_buffer((floats * 4) as u64, MTLResourceOptions::StorageModeShared)
}

// Zero-initialized buffer (Metal's new_buffer does not zero). Required for the SSM
// recurrent state caches — token 0 must see an all-zero conv/matrix state.
fn buf_zeroed(gpu: &MetalGpu, floats: usize) -> metal::Buffer {
    let b = buf(gpu, floats);
    unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, floats * 4); }
    b
}

// ---- autotune cache (gemv_tune.txt under ojas_core::config::cache_dir),
// keyed by device name + shape ----
// Lets the per-device GEMV autotune run once and be reused across launches. Plain
// tab-separated text: dev<TAB>K<TAB>N<TAB>ksplit<TAB>threads.
fn tune_cache_path() -> Option<std::path::PathBuf> {
    ojas_core::config::cache_file("gemv_tune.txt")
}

fn load_tune_cache() -> HashMap<(String, u32, u32), GemvPlan> {
    let mut m = HashMap::new();
    let Some(p) = tune_cache_path() else { return m; };
    let Ok(s) = std::fs::read_to_string(&p) else { return m; };
    for line in s.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 5 { continue; }
        if let (Ok(k), Ok(n), Ok(ks), Ok(t)) =
            (f[1].parse::<u32>(), f[2].parse::<u32>(), f[3].parse::<u8>(), f[4].parse::<u64>()) {
            m.insert((f[0].to_string(), k, n), GemvPlan { ksplit: ks != 0, threads: t });
        }
    }
    m
}

fn save_tune_cache(dev: &str, plans: &HashMap<(u32, u32), GemvPlan>) {
    // read may have come from the legacy root; the write never does.
    let Some(p) = ojas_core::config::cache_dir().map(|d| d.join("gemv_tune.txt")) else { return; };
    let mut all = load_tune_cache(); // keep other devices' entries
    for (&(k, n), &pl) in plans { all.insert((dev.to_string(), k, n), pl); }
    let mut out = String::new();
    for ((d, k, n), pl) in &all {
        out.push_str(&format!("{}\t{}\t{}\t{}\t{}\n", d, k, n, pl.ksplit as u8, pl.threads));
    }
    let _ = std::fs::write(&p, out);
}

/// Per-row symmetric int8 quantization of an f16 weight `[N,K]` (row-major).
/// Returns int8 weights + per-row f32 scale (the scale factors out of the dot).
fn quantize_row_i8(f16_bytes: &[u8], n: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; n * k];
    let mut scale = vec![1.0f32; n];
    // Rows are independent → fan out across cores (the big MoE expert tensors make
    // this the dominant model-load cost; single-threaded it's ~70s for gpt-oss).
    let nthreads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1).clamp(1, n.max(1));
    let rows_per = (n + nthreads - 1) / nthreads;
    std::thread::scope(|s| {
        for (ti, (qc, sc)) in q.chunks_mut(rows_per * k).zip(scale.chunks_mut(rows_per)).enumerate() {
            let row0 = ti * rows_per;
            s.spawn(move || {
                for r in 0..(qc.len() / k) {
                    let gbase = (row0 + r) * k;
                    let mut amax = 0.0f32;
                    for c in 0..k {
                        let bi = (gbase + c) * 2;
                        let v = half::f16::from_bits(u16::from_le_bytes([f16_bytes[bi], f16_bytes[bi + 1]])).to_f32();
                        amax = amax.max(v.abs());
                    }
                    let sv = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                    sc[r] = sv;
                    let lbase = r * k;
                    for c in 0..k {
                        let bi = (gbase + c) * 2;
                        let v = half::f16::from_bits(u16::from_le_bytes([f16_bytes[bi], f16_bytes[bi + 1]])).to_f32();
                        qc[lbase + c] = (v / sv).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            });
        }
    });
    (q, scale)
}

fn bytemuck_u16(v: &[u16]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) } }
fn bytemuck_f32(v: &[f32]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) } }

/// Per-32-block symmetric Q4_0 from f16 bytes. Returns (nibbles[n*k/2], scales_f16
/// [n*k/32]). Dequant: w = scale*(nibble - 8), nibble 0..15. 0.5625 bytes/weight
/// (matches the reference Q4_K), one fewer memory stream than the asymmetric scale+min form.
fn quantize_row_q4(f16_bytes: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u16>) {
    let nblk = k / 32;
    let mut nib = vec![0u8; n * (k / 2)];
    let mut scales = vec![0u16; n * nblk];
    let rd = |bi: usize| half::f16::from_bits(u16::from_le_bytes([f16_bytes[bi], f16_bytes[bi + 1]])).to_f32();
    for row in 0..n {
        for b in 0..nblk {
            let base = row * k + b * 32;
            let mut vals = [0f32; 32];
            let mut amax = 0.0f32;
            for j in 0..32 {
                let v = rd((base + j) * 2);
                vals[j] = v; amax = amax.max(v.abs());
            }
            let sc = if amax > 0.0 { amax / 8.0 } else { 1.0 };
            scales[row * nblk + b] = half::f16::from_f32(sc).to_bits();
            let inv = 1.0 / sc;
            let nb = row * (k / 2) + b * 16;
            for j in 0..16 {
                let q0 = ((vals[2 * j] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                let q1 = ((vals[2 * j + 1] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                nib[nb + j] = q0 | (q1 << 4);
            }
        }
    }
    (nib, scales)
}


mod audit;
mod autotune;
mod batch;
pub(crate) mod dispatch;
mod encoder;
mod entries;
mod graph_attn;
mod graph_chunk;
mod graph_decode;
mod graph_mla;
mod graph_qwen4exp;
mod graph_qwen4exp_m;
mod expert_pool;
mod load;
pub use load::PRECISION_AUTO;
mod moe_stream;
mod pass;
mod doc_cache;
mod prefix_cache;
mod prefix_disk;
mod ple;
mod prefill_session;
mod profile;
pub use profile::{FlashTargetTiming, FlashTraceEvent};
mod span;
mod spec;
mod text_encoder;
mod vision;
pub(crate) use vision::VIT_PATCH_FOLD;

/// Env-gated attention-kernel logger (OJAS_ATTN_LOG=1). Prints each distinct
/// (phase, kernel) once, which is enough to tell which attention path runs
/// (score-array vs flash) without per-token spam. Off costs one atomic load.
pub(crate) fn attn_log(phase: &str, kernel: &str, n: u32) {
    use std::sync::{Mutex, OnceLock};
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("OJAS_ATTN_LOG").is_ok()) {
        return;
    }
    static SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let key = format!("{phase}:{kernel}");
    if SEEN.get_or_init(|| Mutex::new(Default::default())).lock().unwrap().insert(key) {
        tracing::debug!(target: "attn", "{phase}: {kernel} (n={n})");
    }
}

#[cfg(test)]
mod expert_cache_admission_tests {
    use super::*;
    #[test]
    fn decoder_remains_send() {
        fn require_send<T: Send>() {}
        require_send::<DecoderGpu<'static>>();
    }
    #[test]
    fn cold_admission_preserves_hot_entries() {
        let mut cache = ExpertCache::new(8);
        cache.insert(1, vec![1; 4].into_boxed_slice());
        cache.insert(2, vec![2; 4].into_boxed_slice());
        cache.get(2).unwrap();
        assert!(cache.evict_cold());
        assert!(!cache.contains(1));
        assert!(cache.contains(2));
        assert!(!cache.evict_cold());
        assert_eq!(cache.bytes(), 4);
        // Ordinary insertion at full capacity rejects a cold newcomer too.
        cache.insert(3, vec![3; 4].into_boxed_slice());
        cache.get(3).unwrap();
        cache.insert(4, vec![4; 4].into_boxed_slice());
        assert!(!cache.contains(4));
        assert!(cache.contains(2) && cache.contains(3));
    }
}

impl<'a> DecoderGpu<'a> {
    /// The expert tensors only. They dominate the footprint and are what routing
    /// indexes per token, so they are the pages worth pinning. The non-expert weights
    /// are left to page: they are a third of the size, reused every token, and the
    /// embedding tables are read too sparsely to be worth wiring whole. Resident mode
    /// uses unretained command buffers, so whatever is pinned must be in a residency
    /// set or Metal may evict it mid-token.
    pub(crate) fn expert_gpu_buffers(&self) -> Vec<&metal::Buffer> {
        self.wt.wq.iter()
            .filter(|(n, _)| matches!(role_of(n), TensorRole::MoeExpert))
            .map(|(_, b)| b)
            .collect()
    }

    /// Every weight buffer: mmap'd experts and skeleton. Clean file-backed pages,
    /// unlike the state arena and scratch, which are dirty and can only go to swap.
    #[allow(dead_code)]
    pub(crate) fn weight_gpu_buffers(&self) -> Vec<&metal::Buffer> {
        let mut v: Vec<&metal::Buffer> = Vec::new();
        for m in [&self.wt.w16, &self.wt.w32, &self.wt.w8, &self.wt.scale8, &self.wt.w4,
                  &self.wt.scale4, &self.wt.wq, &self.wt.w4k, &self.wt.w6k, &self.wt.w4l,
                  &self.wt.q4l_a, &self.wt.q4l_b, &self.wt.w20, &self.wt.s20] {
            v.extend(m.values());
        }
        v
    }

    /// Independent sequences this decoder can decode per step — [`ojas_core::Model::max_slots`].
    pub fn slots(&self) -> usize { self.st.slots }

    // Byte offsets of slot `s`'s state within the shared per-layer buffers. At
    // `slots == 1` every one of these is 0 for every s, which keeps the
    // single-sequence graphs byte-identical.
    pub(crate) fn kv_slot_off(&self, l: usize, s: usize) -> u64 { s as u64 * self.st.kv_stride(l) }
    pub(crate) fn conv_slot_off(&self, l: usize, s: usize) -> u64 { s as u64 * self.st.conv_stride(l) }
    pub(crate) fn ssm_slot_off(&self, l: usize, s: usize) -> u64 { s as u64 * self.st.ssm_stride(l) }
    pub(crate) fn part_slot_off(&self, s: usize) -> u64 { s as u64 * self.st.part_stride() }
    // The same four for whichever slot the single-sequence graphs are currently
    // pointed at. `cur_slot` rests at 0, so an untouched decoder answers 0.
    pub(crate) fn kv_off(&self, l: usize) -> u64 { self.kv_slot_off(l, self.cur_slot.get()) }
    /// The sequence state of the slot the single-sequence paths point at.
    pub(crate) fn seq(&self) -> &SeqState { &self.sess.seqs[self.cur_slot.get()] }
    pub(crate) fn conv_off(&self, l: usize) -> u64 { self.conv_slot_off(l, self.cur_slot.get()) }
    pub(crate) fn ssm_off(&self, l: usize) -> u64 { self.ssm_slot_off(l, self.cur_slot.get()) }

    // Host views of one slot's recurrent state, for the paths that snapshot, restore
    // and serialize a sequence. These must use the per-slot stride, not
    // `buffer.length()`: that is the whole allocation, so a snapshot taken while slot 0
    // prefilled would restore over slots 1..B and wipe unrelated sequences.
    pub(crate) fn conv_region(&self, l: usize) -> (*mut u8, usize) {
        let n = self.st.conv_stride(l);
        (unsafe { (self.st.conv_state[l].contents() as *mut u8).add(self.conv_off(l) as usize) }, n as usize)
    }
    pub(crate) fn ssm_region(&self, l: usize) -> (*mut u8, usize) {
        let n = self.st.ssm_stride(l);
        (unsafe { (self.st.ssm_state[l].contents() as *mut u8).add(self.ssm_off(l) as usize) }, n as usize)
    }
    /// Host pointer to position 0 of `cur_slot`'s rows of a KV cache.
    pub(crate) fn kv_ptr(&self, b: &metal::Buffer, l: usize) -> *mut u8 {
        unsafe { (b.contents() as *mut u8).add(self.kv_off(l) as usize) }
    }

    /// Point the single-sequence graphs at slot `s` for the duration of `f`, then put
    /// `cur_slot` back however `f` ended. Without the restore, a prefill into slot 2
    /// would leave `cur_slot` at 2 and silently redirect the next plain `forward_id`,
    /// corrupting a whole sequence with no failing call anywhere near it.
    pub(crate) fn with_slot<R>(&self, s: usize, f: impl FnOnce() -> R) -> R {
        let prev = self.cur_slot.replace(s);
        let r = f();
        self.cur_slot.set(prev);
        r
    }

    /// Every GPU buffer the decoder can touch, for `MTLResidencySet` membership.
    ///
    /// The slot-strided state (`kcache`/`vcache`/`conv_state`/`ssm_state`/`attn_part`)
    /// needs nothing added here: slots make those buffers larger, not more numerous, so
    /// the `Buffer` objects already listed cover every slot. A per-slot allocation would
    /// have to be appended — a buffer missing from this list re-faults on every pass.
    pub(crate) fn all_gpu_buffers(&self) -> Vec<&metal::Buffer> {
        let mut v: Vec<&metal::Buffer> = self.weight_gpu_buffers();
        // activation / attention / state arena
        let st = &self.st;
        v.extend([&st.x, &st.h, &st.q, &st.k, &st.v, &st.attn, &st.tmp, &st.tokbuf, &st.xh,
                  &st.qh, &st.skbuf, &st.mpos, &st.gate, &st.up, &st.act, &st.ones, &st.logits,
                  &st.ssm_qkv, &st.ssm_z, &st.ssm_beta, &st.ssm_gate, &st.ssm_o,
                  &st.hc_res, &st.hc_xn, &st.hc_lo, &st.hc_graw, &st.hc_gated, &st.hc_mixed,
                  &st.hc_inject, &st.ple_idx, &st.ple_rows, &st.expert_hash_out, &st.expert_hash_addr,
                  &st.attn_part, &st.plist,
                  // ViT tower scratch. A buffer left off this list re-faults every
                  // pass under MTLResidencySet, which is costly on a 12-layer tower
                  // over thousands of patches.
                  &st.vimg, &st.vrows, &st.vx, &st.vh, &st.vqkv, &st.vq,
                  &st.vkh, &st.vvh, &st.vffn, &st.vpe, &st.vmpos, &st.vout]);
        for vec in [&st.kcache, &st.mla_lat, &st.vcache, &st.conv_state, &st.ssm_state, &st.pmeta] {
            v.extend(vec.iter());
        }
        // MoE scratch
        let ms = &self.ms;
        v.extend([&ms.moe_lg, &ms.moe_idx, &ms.moe_wgt, &ms.moe_act, &ms.moe_sh, &ms.moe_blg,
                  &ms.moe_bidx, &ms.moe_bwgt, &ms.moe_bact, &ms.moe_bsh, &ms.moe_bg, &ms.moe_bu,
                  &ms.moe_btmp]);
        if let Some(b) = &ms.route_lg { v.push(b); }
        // MoE streaming scratch + direct tables (referenced by streamed layers)
        v.extend([&self.strm.moe_gs, &self.strm.moe_us, &self.strm.moe_ds, &self.strm.moe_slot]);
        v.extend(self.strm.direct_tables.iter());
        // MTP / speculative state
        let sp = &self.sp;
        v.extend([&sp.mtp_h, &sp.mtp_hprev, &sp.mtp_cat, &sp.mtp_chain, &sp.mtp_tok]);
        for vec in [&sp.ssm_snap, &sp.conv_snap] { v.extend(vec.iter()); }
        v
    }
}
