//! CUDA qwen35 decoder (surya-2) with its qwen3vl vision tower: `CudaSsm` implements
//! `ojas_core::Model`, so `ojas ocr --device cuda` drives it exactly as it drives the Metal
//! `DecoderGpu` and the CPU `CpuSsm`.
//!
//! The model is the CPU oracle's (`ojas_cpu::cpu_ssm`), layer for layer:
//!
//! - SSM layers (`(l+1) % full_attention_interval != 0`): `attn_qkv` / `attn_gate` /
//!   `ssm_alpha` / `ssm_beta` projections, `ssm_ab` (gate = softplus(alpha+dt)·a, β =
//!   sigmoid), causal conv1d + SiLU with a rolling state (`conv1d_prefill`), the Gated
//!   DeltaNet recurrence with fused q/k L2 norm (`deltanet_fused`), per-head gated RMSNorm
//!   (`gated_rmsnorm`) and `ssm_out` added into the residual.
//! - Attention layers: `attn_q` → per-head `[q | gate]`, QK-RMSNorm, partial NEOX rope over
//!   `rope.dimension_count` dims, sectioned (IMROPE) when the row carries a `(t,h,w,e)`
//!   coordinate (`q35_qk_prep`); causal GQA over an f32 KV cache (`q35_attn_256`),
//!   `× sigmoid(gate)`, `attn_output` into the residual.
//! - SwiGLU FFN with `post_attention_norm` as its pre-norm; final `output_norm` + LM head.
//!
//! ## Matmul precision (`GemmMode`, `OJAS_CUDA_GEMM=fast|split|exact`)
//!
//! Weights are the GGUF's own f16 everywhere. One-token GEMVs (`gemv_f16`) keep activations
//! in f32 in every mode. Prefill differs:
//!
//! * `fast`  — `gemm_mm_f16` once: activations rounded to f16 on their way into the tensor
//!   core tile, as Metal and llama.cpp's cuBLAS path do.
//! * `split` (default) — `gemm_mm_f16(x) + gemm_mm_f16(x - f16(x))`: two tensor-core passes
//!   carrying ~22 bits of every activation — the oracle's f32 activations to within f32
//!   rounding — at twice the (small) GEMM cost.
//! * `exact` — `gemv_m_f16` in 8-row chunks: f32 activations and f32 FMAs like the oracle,
//!   re-reading the weights once per 8 rows. The parity tests' mode.
//!
//! ## Positions
//!
//! Same contract as `Model::prefill_embeds`: `base_pos + i` is the KV cache row, `pos3` moves
//! only the rope angle. Rows without a coordinate (ids, `pos3 = None`, decode) rope from their
//! scalar position with the plain partial rope, bit-for-bit the oracle's `rope_partial`.

use crate::kernels;
use crate::{CuBuf, CudaGpu};
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::CudaStream;
use ojas_core::cancel::STREAM_CANCEL;
use ojas_core::{KernelRuntime, Model};
use ojas_formats::gguf::Gguf;
use std::cell::RefCell;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// How prefill matmuls treat f32 activations. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmMode {
    Fast,
    Split,
    Exact,
}

impl GemmMode {
    /// `OJAS_CUDA_GEMM=fast|split|exact`; unset = `split`.
    pub fn from_env() -> GemmMode {
        match std::env::var("OJAS_CUDA_GEMM").unwrap_or_default().as_str() {
            "fast" => GemmMode::Fast,
            "exact" => GemmMode::Exact,
            _ => GemmMode::Split,
        }
    }
}

/// Vision-tower attention precision (`OJAS_CUDA_VIT_ATTN=f16|x3|f32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VitAttn {
    /// `attention_m_mma_bidir_64`: Q/K/V/P rounded to f16 on the tensor cores (Metal's and
    /// llama.cpp's choice). Fastest; drifts from the f32 oracle at page scale.
    F16,
    /// `q35_vattn_x3_64`: split-f16 tensor cores (hi + lo per operand, three MMAs per product).
    /// Oracle-grade at ~2x the f16 kernel's time. The default.
    X3,
    /// `q35_attn_64`: f32 CUDA cores, no rounding at all. Slowest.
    F32,
}

impl VitAttn {
    pub fn from_env() -> VitAttn {
        match std::env::var("OJAS_CUDA_VIT_ATTN").unwrap_or_default().as_str() {
            "f16" | "fast" => VitAttn::F16,
            "f32" | "exact" => VitAttn::F32,
            _ => VitAttn::X3,
        }
    }
}

/// Load options for [`CudaSsm::load_with`].
#[derive(Clone, Copy, Debug)]
pub struct CudaSsmOpts {
    pub gemm: GemmMode,
    /// KV cache rows to allocate (the context capacity).
    pub context: usize,
    /// Rows per internal prefill chunk (scratch is sized for this).
    pub chunk: usize,
    /// Vision tower attention precision ([`VitAttn`], `OJAS_CUDA_VIT_ATTN`).
    pub vit_attn: VitAttn,
    /// Device ordinal.
    pub ordinal: usize,
    /// Independent sequence slots: each holds its own recurrent state and `context` KV rows.
    /// Generation uses slot 0; the decision readouts prefill one prompt per slot.
    pub slots: usize,
}

impl Default for CudaSsmOpts {
    fn default() -> Self {
        CudaSsmOpts {
            gemm: GemmMode::from_env(),
            context: 8192,
            chunk: 256,
            vit_attn: VitAttn::from_env(),
            ordinal: 0,
            slots: 1,
        }
    }
}

/// What [`CudaSsm::trace_start`] records (the CUDA twin of `cpu_ssm::TraceCfg`).
#[derive(Clone, Debug, Default)]
pub struct TraceCfg {
    pub hidden: bool,
    pub logits: bool,
    pub rows: Option<std::ops::Range<usize>>,
}

/// One forwarded row, laid out like `cpu_ssm::TraceRow`: `hidden[0]` the input row,
/// `hidden[l+1]` the residual after layer `l`, `hidden[n_layers+1]` the post-final-norm row.
#[derive(Clone, Debug)]
pub struct TraceRow {
    pub row: usize,
    pub pos: [u32; 4],
    pub injected: bool,
    pub hidden: Vec<Vec<f32>>,
    pub logits: Option<Vec<f32>>,
}

struct Tracer {
    cfg: TraceCfg,
    rows: Vec<TraceRow>,
}

/// How a matmul weight is held on the device: the file's own blocks where a GEMM kernel reads
/// them in place (`gemm_q.rs`), else f16.
pub(crate) enum Repr {
    F16(CuBuf),
    /// Q4_K relaid by `quant::relayout_q4k_q4l`: nibbles and per-32 f16 `qa`, `qb`.
    Q4L { w4: CuBuf, qa: CuBuf, qb: CuBuf },
    /// Q6_K super-blocks as in the file.
    Q6K(CuBuf),
    /// Q8_0 blocks as in the file.
    Q80(CuBuf),
}

impl Repr {
    fn f16_buf(self) -> Option<CuBuf> { match self { Repr::F16(b) => Some(b), _ => None } }
}

/// A `[n, k]` matmul weight on the device.
pub(crate) struct Lin {
    pub(crate) w: Repr,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

impl Lin {
    /// Several `[n_i, k]` GGUF tensors stacked along n into one `[sum n_i, k]` weight, held in
    /// the file's format when every part shares one the GEMMs read in place (Q4_K, Q6_K,
    /// Q8_0, with the block-aligned K they need), else dequantized to f16.
    pub(crate) fn load(gpu: &CudaGpu, g: &mut Gguf, names: &[String]) -> Result<Lin> { Self::load_as(gpu, g, names, true) }

    /// [`Lin::load`] dequantized to f16 whatever the file holds.
    pub(crate) fn load_f16(gpu: &CudaGpu, g: &mut Gguf, name: &str) -> Result<Lin> { Self::load_as(gpu, g, &[name.to_string()], false) }

    fn load_as(gpu: &CudaGpu, g: &mut Gguf, names: &[String], native: bool) -> Result<Lin> {
        let (mut raw, mut n, mut k0, mut ty0) = (Vec::new(), 0usize, None, None);
        for name in names {
            let info = g.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
            let k = info.dims.first().copied().unwrap_or(1) as usize;
            let rows = info.dims.get(1).copied().unwrap_or(1) as usize;
            ensure!(k0.is_none_or(|k0| k0 == k), "{name}: K {k} differs from the fused group's");
            ensure!(ty0.is_none_or(|t| t == info.ggml_type), "{name}: type differs from the fused group's; the group is dequantized");
            let (_, ty, b) = g.read_tensor_raw(name).with_context(|| format!("reading {name}"))?;
            raw.push(b);
            n += rows;
            k0 = Some(k);
            ty0 = Some(ty);
        }
        let k = k0.unwrap_or(1);
        let in_place = native && match ty0 {
            Some(12) => k % 256 == 0,
            Some(14) => k % 256 == 0,
            Some(8) => k % 32 == 0,
            _ => false,
        };
        let w = if in_place {
            let bytes: Vec<u8> = raw.concat();
            match ty0 {
                Some(12) => {
                    let (nib, qa, qb) = ojas_formats::quant::relayout_q4k_q4l(&bytes, k, n);
                    Repr::Q4L { w4: gpu.upload_bytes(&nib)?, qa: gpu.upload_bytes(ojas_formats::quant::bytes_of_u16(&qa))?, qb: gpu.upload_bytes(ojas_formats::quant::bytes_of_u16(&qb))? }
                }
                Some(14) => Repr::Q6K(gpu.upload_bytes(&bytes)?),
                _ => Repr::Q80(gpu.upload_bytes(&bytes)?),
            }
        } else {
            let mut f16 = Vec::new();
            for name in names {
                let (_, ty, b) = g.read_tensor(name).with_context(|| format!("reading {name}"))?;
                match ty {
                    1 => f16.extend_from_slice(&b),
                    0 => f16.extend(bytes_f32(&b).iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())),
                    t => bail!("{name}: unsupported GGUF type {t}"),
                }
            }
            ensure!(f16.len() == n * k * 2, "{}: {} f16 bytes for [{n}, {k}]", names.join("|"), f16.len());
            Repr::F16(gpu.upload_bytes(&f16)?)
        };
        Ok(Lin { w, n, k })
    }

    pub(crate) fn f16(gpu: &CudaGpu, bytes: &[u8], n: usize, k: usize) -> Result<Lin> {
        Ok(Lin { w: Repr::F16(gpu.upload_bytes(bytes)?), n, k })
    }
}

/// The token table: f16, or a Q4_K file's blocks gathered in place.
enum Embd {
    F16(CuBuf),
    Q4K(CuBuf),
}

/// Input projections are fused at load into one weight each (rows concatenated, so each output
/// is the same dot product) and one matmul fills `st.comb`: SSM
/// `[attn_qkv | attn_gate | ssm_alpha | ssm_beta]`, attention `[attn_q | attn_k | attn_v]`,
/// FFN `[ffn_gate | ffn_up]`.
enum Mixer {
    Ssm { win: Lin, dt: CuBuf, a: CuBuf, conv_w: CuBuf, norm: CuBuf, wout: Lin },
    Attn { wqkv: Lin, q_norm: CuBuf, k_norm: CuBuf, wo: Lin },
}

struct Layer {
    attn_norm: CuBuf,
    post_norm: CuBuf,
    mixer: Mixer,
    gate_up: Lin,
    down: Lin,
}

/// Recurrent state, KV cache and per-chunk scratch. The state buffers hold every slot's
/// region back to back: `conv` / `ssm` / `kc` / `vc` of slot `s` start at
/// `s * conv_bytes` / `s * ssm_bytes` / `s * kv_bytes` ([`CudaSsm::slot_bytes`]); `conv`
/// and `ssm` have one region more than there are slots, the saved state.
struct State {
    /// Fused-projection output rows (see [`Mixer`]).
    comb: CuBuf,
    conv: Vec<Option<CuBuf>>,
    ssm: Vec<Option<CuBuf>>,
    kc: Vec<Option<CuBuf>>,
    vc: Vec<Option<CuBuf>>,
    x: CuBuf,
    h: CuBuf,
    qkv: CuBuf,
    z: CuBuf,
    alpha: CuBuf,
    beta: CuBuf,
    o: CuBuf,
    qfull: CuBuf,
    kb: CuBuf,
    vb: CuBuf,
    q: CuBuf,
    att: CuBuf,
    g: CuBuf,
    tmp: CuBuf,
    lo: CuBuf,
    po: CuBuf,
    pml: CuBuf,
    logits: CuBuf,
    amax: CuBuf,
    ids: CuBuf,
    mpos: CuBuf,
    /// `[base, total]` of the current chunk, read by `q35_qk_prep` / `q35_attn_256`.
    ctl: CuBuf,
    /// One `[base, total, slot, first row]` per segment of a multi-sequence chunk (`slots` entries).
    ctls: CuBuf,
    /// `[dst, src, words]` per region of a state copy (`copy_regions`): two per layer.
    copytab: CuBuf,
    /// The chunk's activations as f16 for the quantized split-K GEMMs, `chunk * widest K`.
    xh: CuBuf,
}

/// The most split-K partitions.
const SPLITK_MAX: usize = 8;

/// Split-K partitions for a chunk on the 64x128 tile: enough to put about
/// [`SPLITK_BLOCKS`] blocks in flight (eight per SM on a 28-SM card, the fastest count on a
/// 4B model's prefill), at most [`SPLITK_MAX`], each at least one 32-block; 1 when the
/// tiles alone fill the card.
const SPLITK_BLOCKS: usize = 224;

fn nsplit_for(m: usize, k: usize, n: usize) -> usize {
    let tiles = m.div_ceil(64) * n.div_ceil(128);
    let mut ns = SPLITK_BLOCKS.div_ceil(tiles.max(1)).clamp(1, SPLITK_MAX);
    while ns > 1 && k / ns < 32 { ns -= 1; }
    ns
}

/// One sequence's run of consecutive rows in a chunk: the rows continue `slot` at cache
/// rows `base..`, and `ctl` holds their `[base, base + rows]` on the device.
struct Seg<'a> {
    rows: std::ops::Range<usize>,
    slot: usize,
    base: usize,
    /// Its `[base, base + rows, slot, rows.start]` u32 words on the device; the segments
    /// of a chunk are consecutive, so the first one's is the table of them all.
    ctl: (&'a CuBuf, u64),
}

/// What a forward leaves behind for its last row.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Want {
    /// Nothing (prefill).
    Nothing,
    /// Logits on the device (`st.logits`), not read back — `forward_id` argmaxes them there.
    Head,
    /// Logits read back to the host.
    Logits,
}

/// Keys per flash-decoding split, and the query count up to which splitting is used.
const KSPLIT: usize = 256;
const SPLIT_MAX_M: usize = 16;

pub struct CudaSsm {
    gpu: CudaGpu,
    enc: Arc<CudaStream>,
    opts: CudaSsmOpts,
    d: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    n_rot: usize,
    s_st: usize,
    h_k: usize,
    h_v: usize,
    d_inner: usize,
    head_v: usize,
    conv_ch: usize,
    conv_k: usize,
    ffn: usize,
    vocab: usize,
    rope_base: f32,
    eps: f32,
    mrope_sections: [u32; 4],
    mrope_mode: u32,
    layers: Vec<Layer>,
    output_norm: CuBuf,
    embd: Embd,
    head: Option<Lin>,
    max_split: usize,
    vit: Option<CudaVit>,
    st: RefCell<State>,
    tracer: RefCell<Option<Tracer>>,
    /// The one-token decode step recorded as a CUDA graph (captured on first use).
    graph: RefCell<Option<crate::Recorded>>,
    /// Accumulated wall time spent in decoder forwards / vision encodes (seconds).
    pub timing: RefCell<Timing>,
    profile: crate::KernelProfile,
}

/// Coarse timing counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub prefill_s: f64,
    pub prefill_rows: usize,
    pub decode_s: f64,
    pub decode_tokens: usize,
    pub vit_s: f64,
}

pub(crate) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub(crate) fn read_f32_tensor(g: &mut Gguf, name: &str) -> Result<Vec<f32>> {
    let (_d, ty, b) = g.read_tensor(name).with_context(|| format!("reading {name}"))?;
    match ty {
        0 => Ok(bytes_f32(&b)),
        1 => Ok(b.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()),
        t => bail!("{name}: unsupported GGUF type {t}"),
    }
}

pub(crate) fn u32s_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `b` moved `r0` rows of `stride` f32 down: a segment's view of a `[M, stride]` buffer.
fn at_rows(b: (&CuBuf, u64), r0: u64, stride: usize) -> (&CuBuf, u64) { (b.0, b.1 + r0 * stride as u64 * 4) }

pub(crate) fn blocks(n: usize, b: usize) -> u32 {
    n.div_ceil(b).max(1) as u32
}

impl CudaSsm {
    /// Load with default options and the given context.
    pub fn load(g: &mut Gguf, context: usize) -> Result<CudaSsm> {
        Self::load_with(g, CudaSsmOpts { context, ..CudaSsmOpts::default() })
    }

    pub fn load_with(g: &mut Gguf, opts: CudaSsmOpts) -> Result<CudaSsm> {
        let arch = g.arch();
        ensure!(arch == "qwen35", "CudaSsm runs dense qwen35 (surya-2); got {arch}");
        let mut gpu = CudaGpu::new(opts.ordinal)?;
        for fam in ["ops", "ssm", "gemm_f16", "gemm_q", "qwen35", "vision", "attn_bidir", "bert"] {
            gpu.ensure_family(fam).with_context(|| format!("compiling CUDA family {fam}"))?;
        }
        let enc = gpu.begin();
        let mu = |g: &Gguf, k: &str| g.meta_u32(&format!("qwen35.{k}")).unwrap_or(0) as usize;
        let d = mu(g, "embedding_length");
        let n_layers = mu(g, "block_count");
        let n_head = mu(g, "attention.head_count");
        let n_kv = mu(g, "attention.head_count_kv");
        let hd = g.meta_u32("qwen35.attention.key_length").unwrap_or(256) as usize;
        let ffn = mu(g, "feed_forward_length");
        let s_st = g.meta_u32("qwen35.ssm.state_size").unwrap_or(128) as usize;
        let h_k = g.meta_u32("qwen35.ssm.group_count").unwrap_or(16) as usize;
        let h_v = g.meta_u32("qwen35.ssm.time_step_rank").unwrap_or(32) as usize;
        let d_inner = g.meta_u32("qwen35.ssm.inner_size").unwrap_or(4096) as usize;
        let conv_k = g.meta_u32("qwen35.ssm.conv_kernel").unwrap_or(4) as usize;
        let interval = g.meta_u32("qwen35.full_attention_interval").unwrap_or(4) as usize;
        let n_rot = g.meta_u32("qwen35.rope.dimension_count").unwrap_or(64) as usize;
        let rope_base = g.meta_f32("qwen35.rope.freq_base").unwrap_or(1e6);
        let eps = g.meta_f32("qwen35.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let vocab = g.str_arr("tokenizer.ggml.tokens").map(|t| t.len()).unwrap_or(1).max(1);
        let head_v = d_inner / h_v.max(1);
        let conv_ch = 2 * h_k * s_st + d_inner;
        let mut mrope_sections = [0u32; 4];
        if let Some(a) = g.int_arr("qwen35.rope.dimension_sections") {
            for (i, v) in a.iter().take(4).enumerate() {
                mrope_sections[i] = (*v).max(0) as u32;
            }
        }
        // The kernels' fixed shapes.
        ensure!(hd == 256, "q35_attn is compiled for head_dim 256 (got {hd})");
        ensure!(s_st == 128, "deltanet_fused holds a 128-wide state column in registers (got {s_st})");
        ensure!(head_v % 32 == 0 && hd <= 512 && n_rot <= hd && n_rot % 2 == 0, "unsupported head geometry");
        ensure!(n_kv > 0 && n_head % n_kv == 0 && 32 % (n_head / n_kv) == 0, "GQA group {}/{} must divide 32", n_head, n_kv);
        ensure!(conv_k >= 2 && conv_k <= 9, "conv1d_prefill keeps <= 8 taps of state (K = {conv_k})");
        ensure!(d % 4 == 0 && (n_kv * hd) % 4 == 0, "f32x4 loads need 4-aligned rows");

        let t0 = std::time::Instant::now();
        let up_lin = |gpu: &CudaGpu, g: &mut Gguf, name: &str| -> Result<Lin> { Lin::load(gpu, g, &[name.to_string()]) };
        let up_f32 = |gpu: &CudaGpu, g: &mut Gguf, name: &str| -> Result<CuBuf> {
            gpu.upload_bytes(&f32_bytes(&read_f32_tensor(g, name)?))
        };
        let up_fused = |gpu: &CudaGpu, g: &mut Gguf, names: &[String]| -> Result<Lin> { Lin::load(gpu, g, names) };
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            let mixer = if (i + 1) % interval == 0 {
                Mixer::Attn {
                    wqkv: up_fused(&gpu, g, &[p("attn_q.weight"), p("attn_k.weight"), p("attn_v.weight")])?,
                    q_norm: up_f32(&gpu, g, &p("attn_q_norm.weight"))?,
                    k_norm: up_f32(&gpu, g, &p("attn_k_norm.weight"))?,
                    wo: up_lin(&gpu, g, &p("attn_output.weight"))?,
                }
            } else {
                Mixer::Ssm {
                    win: up_fused(&gpu, g, &[p("attn_qkv.weight"), p("attn_gate.weight"), p("ssm_alpha.weight"), p("ssm_beta.weight")])?,
                    dt: up_f32(&gpu, g, &p("ssm_dt.bias"))?,
                    a: up_f32(&gpu, g, &p("ssm_a"))?,
                    conv_w: up_f32(&gpu, g, &p("ssm_conv1d.weight"))?,
                    norm: up_f32(&gpu, g, &p("ssm_norm.weight"))?,
                    wout: up_lin(&gpu, g, &p("ssm_out.weight"))?,
                }
            };
            layers.push(Layer {
                attn_norm: up_f32(&gpu, g, &p("attn_norm.weight"))?,
                post_norm: up_f32(&gpu, g, &p("post_attention_norm.weight"))?,
                mixer,
                gate_up: up_fused(&gpu, g, &[p("ffn_gate.weight"), p("ffn_up.weight")])?,
                down: up_lin(&gpu, g, &p("ffn_down.weight"))?,
            });
        }
        let output_norm = up_f32(&gpu, g, "output_norm.weight")?;
        let embd = {
            let info = g.tensors.get("token_embd.weight").context("missing token_embd.weight")?;
            ensure!(info.dims.first().copied() == Some(d as u64) && info.dims.get(1).copied() == Some(vocab as u64),
                "token_embd is {:?}, expected [{d}, {vocab}]", info.dims);
            if info.ggml_type == 12 && d % 256 == 0 {
                Embd::Q4K(gpu.upload_bytes(&g.read_tensor_raw("token_embd.weight")?.2)?)
            } else {
                Embd::F16(Lin::load_f16(&gpu, g, "token_embd.weight")?.w.f16_buf().expect("load_f16 holds f16"))
            }
        };
        let head = if g.tensors.contains_key("output.weight") { Some(up_lin(&gpu, g, "output.weight")?) } else { None };

        let opts = CudaSsmOpts { context: opts.context.max(16), chunk: opts.chunk.clamp(16, 4096), slots: opts.slots.max(1), ..opts };
        let max_split = opts.context.div_ceil(KSPLIT).max(1);
        let kv_elem = if opts.gemm == GemmMode::Exact { 4 } else { 2 };
        let st = Self::alloc_state(&gpu, &layers, &opts, d, n_head, n_kv, hd, conv_ch, conv_k, h_v, s_st, d_inner, ffn, vocab, max_split, kv_elem)?;
        gpu.sync()?;
        tracing::info!(target: "cuda:qwen35",
            "loaded {n_layers} layers in {:.1}s | d={d} attn {n_head}/{n_kv} hd={hd} rot={n_rot} sections={mrope_sections:?} \
             | GDN S={s_st} Hk={h_k} Hv={h_v} d_inner={d_inner} | ctx={} chunk={} gemm={:?}",
            t0.elapsed().as_secs_f64(), opts.context, opts.chunk, opts.gemm);
        Ok(CudaSsm {
            gpu, enc, opts, d, n_head, n_kv, hd, n_rot, s_st, h_k, h_v, d_inner, head_v, conv_ch, conv_k, ffn, vocab,
            rope_base, eps, mrope_sections, mrope_mode: 2, layers, output_norm, embd, head, max_split, vit: None,
            st: RefCell::new(st),
            tracer: RefCell::new(None),
            graph: RefCell::new(None),
            timing: RefCell::new(Timing::default()),
            profile: crate::KernelProfile::from_env(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn alloc_state(gpu: &CudaGpu, layers: &[Layer], o: &CudaSsmOpts, d: usize, n_head: usize, n_kv: usize, hd: usize,
                   conv_ch: usize, conv_k: usize, h_v: usize, s_st: usize, d_inner: usize, ffn: usize, vocab: usize,
                   max_split: usize, kv_elem: usize) -> Result<State> {
        let f = |n: usize| gpu.alloc_bytes(n * 4);
        let c = o.chunk;
        let (qdim, kvdim) = (n_head * hd, n_kv * hd);
        let mut conv = Vec::new();
        let mut ssm = Vec::new();
        let mut kc = Vec::new();
        let mut vc = Vec::new();
        for l in layers {
            match l.mixer {
                Mixer::Ssm { .. } => {
                    // one extra slot: the saved state (`save_state`), on the device
                    conv.push(Some(f((o.slots + 1) * (conv_k - 1) * conv_ch)?));
                    ssm.push(Some(f((o.slots + 1) * h_v * s_st * s_st)?));
                    kc.push(None);
                    vc.push(None);
                }
                Mixer::Attn { .. } => {
                    conv.push(None);
                    ssm.push(None);
                    kc.push(Some(gpu.alloc_bytes(o.slots * o.context * kvdim * kv_elem)?));
                    vc.push(Some(gpu.alloc_bytes(o.slots * o.context * kvdim * kv_elem)?));
                }
            }
        }
        let maxk = d.max(d_inner).max(qdim).max(ffn);
        let comb_w = (conv_ch + d_inner + 2 * h_v).max(2 * qdim + 2 * kvdim).max(2 * ffn);
        Ok(State {
            comb: f(c * comb_w)?,
            conv, ssm, kc, vc,
            x: f(c * d)?,
            h: f(c * d)?,
            qkv: f(c * conv_ch)?,
            z: f(c * d_inner)?,
            alpha: f(c * h_v)?,
            beta: f(c * h_v)?,
            o: f(c * d_inner)?,
            qfull: f(c * 2 * qdim)?,
            kb: f(c * kvdim)?,
            vb: f(c * kvdim)?,
            q: f(c * qdim)?,
            att: f(c * qdim)?,
            g: f(c * ffn)?,
            tmp: f(c * d)?,
            lo: f(c * maxk)?,
            po: f(max_split * SPLIT_MAX_M * qdim)?,
            pml: f(max_split * SPLIT_MAX_M * n_head * 2)?,
            logits: f(vocab)?,
            amax: f(4)?,
            ids: f(c)?,
            mpos: f(c * 4)?,
            ctl: f(4)?,
            ctls: f(4 * o.slots)?,
            copytab: gpu.alloc_bytes(layers.len() * 2 * 24)?,
            xh: gpu.alloc_bytes(c * maxk * 2)?,
        })
    }

    /// Attach the vision tower from an mmproj GGUF (`general.architecture = clip`).
    pub fn attach_vit_gguf(&mut self, mm: &mut Gguf) -> Result<()> {
        let v = CudaVit::load(&self.gpu, mm)?;
        ensure!(v.proj_dim == self.d, "the mmproj projects to {} but the decoder is {}-wide", v.proj_dim, self.d);
        self.vit = Some(v);
        Ok(())
    }

    pub fn gpu(&self) -> &CudaGpu { &self.gpu }
    /// Sequence slots this runner holds state for.
    pub fn slots(&self) -> usize { self.opts.slots }
    pub fn opts(&self) -> CudaSsmOpts { self.opts }
    pub fn vocab(&self) -> usize { self.vocab }
    pub fn mrope(&self) -> ([u32; 4], u32) { (self.mrope_sections, self.mrope_mode) }
    pub fn has_vit(&self) -> bool { self.vit.is_some() }
    /// Change the prefill matmul mode after load (the parity tests flip it).
    pub fn set_gemm(&mut self, m: GemmMode) { self.opts.gemm = m; }
    pub fn set_vit_attn(&mut self, a: VitAttn) { self.opts.vit_attn = a; }

    // -------------------------------------------------------------- tracing

    pub fn trace_start(&self, cfg: TraceCfg) {
        *self.tracer.borrow_mut() = Some(Tracer { cfg, rows: Vec::new() });
    }

    pub fn trace_take(&self) -> Vec<TraceRow> {
        self.tracer.borrow_mut().take().map(|t| t.rows).unwrap_or_default()
    }

    // ------------------------------------------------------------ dispatch

    fn k(&self, name: &str, bufs: &[(&CuBuf, u64)], consts: &[u32], grid: [u32; 3], block: [u32; 3]) -> Result<()> {
        self.gpu.dispatch_profiled(&self.profile, name, bufs, consts, grid, block).with_context(|| format!("launching {name}"))
    }

    /// `x[m] = token_embd[ids[m]]` for `m` rows.
    fn embed(&self, st: &State, m: usize) -> Result<()> {
        let d = self.d;
        let (name, table) = match &self.embd { Embd::F16(t) => ("q35_embed_f16", t), Embd::Q4K(t) => ("q35_embed_q4k", t) };
        self.k(name, &[(table, 0), (&st.ids, 0), (&st.x, 0)], &[d as u32, m as u32], [blocks(m * d, 256), 1, 1], [256, 1, 1])
    }

    /// `st.logits = head · st.h[0]`: the LM head (or the tied embedding) over one row.
    fn head_logits(&self, st: &State) -> Result<()> {
        let (d, vocab) = (self.d as u32, self.vocab as u32);
        match (&self.head, &self.embd) {
            (Some(h), _) => match &h.w {
                Repr::F16(w) => self.k("gemv_f16", &[(&st.h, 0), (w, 0), (&st.logits, 0)], &[d, vocab], [blocks(self.vocab, 8), 1, 1], [256, 1, 1]),
                _ => self.mm(st, &st.h, h, &st.logits, 1, false),
            },
            (None, Embd::F16(w)) => self.k("gemv_f16", &[(&st.h, 0), (w, 0), (&st.logits, 0)], &[d, vocab], [blocks(self.vocab, 8), 1, 1], [256, 1, 1]),
            (None, Embd::Q4K(_)) => bail!("a tied LM head over a Q4_K token table is not implemented"),
        }
    }

    /// `y[m, n] (+)= x[m, :] · W[n, :]` for `m` rows. `accum` adds into `y` (which must then be
    /// the `d`-wide residual; `tmp` stages the GEMV forms). An f16 weight takes the GEMV
    /// forms for a few rows and `GemmMode::Exact`; a quantized one always goes through its
    /// tensor-core GEMM, which rounds the activations to f16 (`Exact` has no quantized form).
    fn mm(&self, st: &State, x: &CuBuf, w: &Lin, y: &CuBuf, m: usize, accum: bool) -> Result<()> {
        let (k, n) = (w.k as u32, w.n as u32);
        let gemv_grid = [blocks(w.n, 8), 1, 1];
        let f16 = match &w.w { Repr::F16(b) => Some(b), _ => None };
        if let Some(wb) = f16.filter(|_| m <= 8 || self.opts.gemm == GemmMode::Exact) {
            let dst = if accum { &st.tmp } else { y };
            if accum {
                ensure!(m * w.n <= self.opts.chunk * self.d, "accumulating GEMV wider than the residual");
            }
            if m == 1 {
                self.k("gemv_f16", &[(x, 0), (wb, 0), (dst, 0)], &[k, n], gemv_grid, [256, 1, 1])?;
            } else {
                self.k("gemv_m_f16", &[(x, 0), (wb, 0), (dst, 0)], &[k, n, m as u32], gemv_grid, [256, 1, 1])?;
            }
            if accum {
                let t = (m * w.n) as u32;
                self.k("add_inplace", &[(y, 0), (&st.tmp, 0)], &[t], [blocks(m * w.n, 256), 1, 1], [256, 1, 1])?;
            }
            return Ok(());
        }
        // A short chunk, or a narrow N, is a few rows of N/128 blocks: split K across the
        // idle SMs instead, each slice adding into y.
        let nsplit = nsplit_for(m, w.k, w.n);
        // the one-pass mode takes the `_h` entries (f16 partial sums), the others every
        // product in f32
        let h = if self.opts.gemm == GemmMode::Fast { "_h" } else { "" };
        let gemm = |x: &CuBuf, accum: u32| -> Result<()> {
            let mu = m as u32;
            if nsplit > 1 {
                let grid = [blocks(w.n, 128), blocks(m, 64), nsplit as u32];
                let ns = nsplit as u32;
                if accum == 0 { self.gpu.zero_bytes(y, 0, m * w.n * 4)?; }
                if !matches!(w.w, Repr::F16(_)) {
                    let t = m * w.k;
                    self.k("copy_f32_half", &[(x, 0), (&st.xh, 0)], &[t as u32], [blocks(t, 256), 1, 1], [256, 1, 1])?;
                }
                let xh = &st.xh;
                return match &w.w {
                    Repr::F16(wb) => self.k("gemm_mm_f16_sk", &[(x, 0), (wb, 0), (y, 0)], &[k, n, mu, ns], grid, [256, 1, 1]),
                    Repr::Q4L { w4, qa, qb } => self.k(&format!("gemm_mm_q4l_sk{h}"), &[(xh, 0), (w4, 0), (y, 0), (qa, 0), (qb, 0)], &[k, n, mu, ns], grid, [256, 1, 1]),
                    Repr::Q6K(wb) => self.k("gemm_mm_q6k_sk", &[(xh, 0), (wb, 0), (y, 0)], &[k, n, mu, ns], grid, [256, 1, 1]),
                    Repr::Q80(wb) => self.k(&format!("gemm_mm_q8_0_sk{h}"), &[(xh, 0), (wb, 0), (y, 0)], &[k, n, mu, ns], grid, [256, 1, 1]),
                };
            }
            match &w.w {
                Repr::F16(wb) => self.k("gemm_mm_f16", &[(x, 0), (wb, 0), (y, 0)], &[k, n, accum, mu], [blocks(w.n, 128), blocks(m, 128), 1], [256, 1, 1]),
                Repr::Q4L { w4, qa, qb } => self.k(&format!("gemm_mm_q4l{h}"), &[(x, 0), (w4, 0), (y, 0), (qa, 0), (qb, 0)], &[k, n, accum, mu], [blocks(w.n, 128), blocks(m, 64), 1], [256, 1, 1]),
                Repr::Q6K(wb) => self.k("gemm_mm_q6k", &[(x, 0), (wb, 0), (y, 0)], &[k, n, accum, mu], [blocks(w.n, 128), blocks(m, 64), 1], [256, 1, 1]),
                Repr::Q80(wb) => self.k(&format!("gemm_mm_q8_0{h}"), &[(x, 0), (wb, 0), (y, 0)], &[k, n, accum, mu], [blocks(w.n, 128), blocks(m, 64), 1], [256, 1, 1]),
            }
        };
        gemm(x, accum as u32)?;
        if self.opts.gemm == GemmMode::Split {
            let tot = m * w.k;
            self.k("q35_split_lo", &[(x, 0), (&st.lo, 0)], &[tot as u32], [blocks(tot, 256), 1, 1], [256, 1, 1])?;
            gemm(&st.lo, 1)?;
        }
        Ok(())
    }

    fn rmsnorm(&self, x: &CuBuf, xoff: u64, w: &CuBuf, out: &CuBuf, ooff: u64, m: usize) -> Result<()> {
        self.k("rmsnorm_m", &[(x, xoff), (w, 0), (out, ooff)], &[self.d as u32, self.eps.to_bits()], [m as u32, 1, 1], [256, 1, 1])
    }

    // --------------------------------------------------------------- state

    /// Bytes of one slot's region of a layer's conv state, SSM state and KV cache.
    fn slot_bytes(&self) -> (usize, usize, usize) {
        ((self.conv_k - 1) * self.conv_ch * 4, self.h_v * self.s_st * self.s_st * 4, self.opts.context * self.n_kv * self.hd * self.kv_elem())
    }

    /// Bytes per K/V cache element: f16, or f32 in the exact mode (no rounding anywhere).
    fn kv_elem(&self) -> usize { if self.opts.gemm == GemmMode::Exact { 4 } else { 2 } }

    /// Zero every slot's recurrent state (fresh sequences). KV rows need no clearing:
    /// attention only ever reads rows it has written in this sequence.
    pub fn reset(&self) {
        let st = &mut *self.st.borrow_mut();
        for b in st.conv.iter_mut().chain(st.ssm.iter_mut()).flatten() {
            self.enc.memset_zeros(&mut b.bytes).expect("cuda memset");
        }
    }

    /// Zero one slot's recurrent state.
    pub fn reset_slot(&self, slot: usize) {
        assert!(slot < self.opts.slots, "slot {slot} of {}", self.opts.slots);
        let (cb, sb, _) = self.slot_bytes();
        let st = &mut *self.st.borrow_mut();
        for (b, n) in st.conv.iter_mut().map(|b| (b, cb)).chain(st.ssm.iter_mut().map(|b| (b, sb))) {
            if let Some(b) = b {
                let mut region = b.bytes.slice_mut(slot * n..(slot + 1) * n);
                self.enc.memset_zeros(&mut region).expect("cuda memset");
            }
        }
    }

    /// Keep slot 0's recurrent state as the saved state: a device-side copy into the
    /// state buffers' extra slot, so the state (50 MB for a 4B model) never crosses to
    /// the host.
    pub fn save_state(&self) -> Result<()> {
        let st = &mut *self.st.borrow_mut();
        self.copy_slots(st, 0, self.opts.slots, 0)
    }

    /// Restore slot 0's recurrent state from the saved one.
    pub fn restore_state(&self) -> Result<()> {
        let st = &mut *self.st.borrow_mut();
        self.copy_slots(st, self.opts.slots, 0, 0)
    }

    /// Copy slot `from`'s recurrent state and its first `rows` cache rows into slot `to`.
    pub fn copy_slot_prefix(&self, from: usize, to: usize, rows: usize) -> Result<()> {
        ensure!(from < self.opts.slots && to < self.opts.slots && rows <= self.opts.context, "copy_slot_prefix: slot or rows out of range");
        if from == to { return Ok(()); }
        let st = &mut *self.st.borrow_mut();
        let kv_rows = rows * self.n_kv * self.hd * self.kv_elem();
        self.copy_slots(st, from, to, kv_rows)
    }

    /// Every SSM layer's conv and SSM state, slot `from` to slot `to` (slot `slots` is the
    /// saved state), and `kv_bytes` of each attention layer's K and V rows. One launch
    /// over a table of regions (a 4B model has 64), stream-ordered like the kernels that
    /// read the state.
    fn copy_slots(&self, st: &mut State, from: usize, to: usize, kv_bytes: usize) -> Result<()> {
        let (cb, sb, kb) = self.slot_bytes();
        let mut tab: Vec<u64> = Vec::with_capacity(self.layers.len() * 6);
        let mut longest = 0;
        let mut add = |b: &CuBuf, region: usize, bytes: usize| {
            if bytes == 0 { return; }
            let p = self.gpu.device_ptr(b);
            tab.extend([p + (to * region) as u64, p + (from * region) as u64, (bytes / 4) as u64]);
            longest = longest.max(bytes / 4);
        };
        for l in 0..self.layers.len() {
            match (&st.conv[l], &st.ssm[l], &st.kc[l], &st.vc[l]) {
                (Some(c), Some(s), _, _) => { add(c, cb, cb); add(s, sb, sb); }
                (_, _, Some(kc), Some(vc)) => { add(kc, kb, kv_bytes); add(vc, kb, kv_bytes); }
                _ => unreachable!("a layer has either a recurrent state or a KV cache"),
            }
        }
        if tab.is_empty() { return Ok(()); }
        let bytes: Vec<u8> = tab.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.write_bytes(&mut st.copytab, 0, &bytes)?;
        let grid_x = blocks(longest, 256).min(256);
        self.k("copy_regions", &[(&st.copytab, 0)], &[], [grid_x, (tab.len() / 3) as u32, 1], [256, 1, 1])
    }

    /// The merged token grid `(columns, rows)` an image of this size encodes to.
    pub fn vision_grid(&self, width: usize, height: usize) -> Option<(usize, usize)> {
        let v = self.vit.as_ref()?;
        Some((width / v.patch / v.merge, height / v.patch / v.merge))
    }

    /// The vision tower's patch side and spatial merge.
    pub fn vision_patch_merge(&self) -> Option<(usize, usize)> { self.vit.as_ref().map(|v| (v.patch, v.merge)) }

    /// Wall time spent in decoder forwards and vision encodes so far, seconds.
    pub fn gpu_seconds(&self) -> f64 {
        let t = self.timing.borrow();
        t.prefill_s + t.decode_s + t.vit_s
    }

    /// Prefill several prompts at once, each in its own slot, and return the final hidden
    /// state (after `output_norm`) of each prompt's `read` rows (ascending prompt indices
    /// inside its span). Every pass holds rows of as many prompts as fit in a chunk, one
    /// segment each ([`CudaSsm::forward_segments`]), so the weights are read once for all
    /// of them. Rows a prompt gives as embeddings enter the residual stream directly; rows
    /// with explicit rotary coordinates are roped at those rather than at their cache row.
    pub fn prefill_hidden_slots(&self, jobs: &[ojas_decision::SlotPrefill]) -> Result<Vec<Vec<Vec<f32>>>> {
        ensure!(jobs.len() <= self.opts.slots && jobs.iter().enumerate().all(|(i, j)| jobs[..i].iter().all(|k| k.slot != j.slot)),
            "prefill_hidden_slots: one slot per prompt, at most {} slots", self.opts.slots);
        for j in jobs {
            ensure!(j.slot < self.opts.slots && j.span.end <= j.prompt.ids.len() && j.span.end <= self.opts.context
                && j.prompt.positions.is_none_or(|p| p.len() == j.prompt.ids.len()),
                "prefill_hidden_slots: the span or the positions do not fit the prompt or the context");
            ensure!(j.read.windows(2).all(|w| w[0] < w[1]) && j.read.iter().all(|r| j.span.contains(r)),
                "prefill_hidden_slots: rows to read must be ascending and inside the span");
        }
        let d = self.d;
        let chunk = self.opts.chunk;
        let positioned = jobs.iter().any(|j| j.prompt.positions.is_some());
        let t0 = std::time::Instant::now();
        let mut next: Vec<usize> = jobs.iter().map(|j| j.span.start).collect();
        let mut out: Vec<Vec<Vec<f32>>> = jobs.iter().map(|j| Vec::with_capacity(j.read.len())).collect();
        let mut total_rows = 0;
        loop {
            let (mut tokens, mut segments, mut given, mut positions) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (i, j) in jobs.iter().enumerate() {
                let room = chunk - tokens.len();
                if next[i] == j.span.end || room == 0 { continue; }
                let take = (j.span.end - next[i]).min(room);
                let (from, r0) = (next[i], tokens.len());
                tokens.extend_from_slice(&j.prompt.ids[from..from + take]);
                for p in from..from + take {
                    positions.push(j.prompt.positions.map_or([p as u32, p as u32, p as u32, 0], |pos| pos[p]));
                }
                for &(first, rows) in j.prompt.embedded {
                    let (lo, hi) = (first.max(from), (first + rows.len() / d).min(from + take));
                    if lo < hi { given.push((r0 + lo - from, &rows[(lo - first) * d..(hi - first) * d])); }
                }
                segments.push((i, r0..r0 + take, j.slot, from));
                next[i] += take;
            }
            if tokens.is_empty() { break; }
            total_rows += tokens.len();
            let reads: Vec<usize> = segments.iter().flat_map(|(i, rows, _, base)| {
                jobs[*i].read.iter().filter(move |&&r| (*base..*base + rows.len()).contains(&r)).map(move |&r| rows.start + r - base)
            }).collect();
            let segs: Vec<(std::ops::Range<usize>, usize, usize)> = segments.iter().map(|(_, rows, slot, base)| (rows.clone(), *slot, *base)).collect();
            let hidden = self.forward_segments(&tokens, &given, positioned.then_some(positions.as_slice()), &segs, &reads)?;
            let mut h = hidden.into_iter();
            for (i, rows, _, base) in &segments {
                for _ in jobs[*i].read.iter().filter(|&&r| (*base..*base + rows.len()).contains(&r)) {
                    out[*i].push(h.next().expect("one hidden row per read"));
                }
            }
        }
        let mut tm = self.timing.borrow_mut();
        tm.prefill_s += t0.elapsed().as_secs_f64();
        tm.prefill_rows += total_rows;
        drop(tm);
        self.profile.report(&self.gpu, &format!("prefill_hidden_slots, {total_rows} rows in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3));
        Ok(out)
    }

    /// One chunk of several sequences: `tokens` tile the chunk's rows, `segs` are
    /// `(rows, slot, base)` runs of one sequence each, `given` are rows written as
    /// embeddings instead of gathered, `pos3` the rotary coordinate of every row. Returns
    /// the post-`output_norm` hidden state of the chunk rows `reads` names.
    fn forward_segments(&self, tokens: &[u32], given: &[(usize, &[f32])], pos3: Option<&[[u32; 4]]>,
                        segs: &[(std::ops::Range<usize>, usize, usize)], reads: &[usize]) -> Result<Vec<Vec<f32>>> {
        let (d, m) = (self.d, tokens.len());
        ensure!(m > 0 && m <= self.opts.chunk && segs.len() <= self.opts.slots, "forward_segments: chunk of {m} rows, {} segments", segs.len());
        ensure!(tokens.iter().all(|&t| (t as usize) < self.vocab), "forward_segments: a token id is outside the {}-token vocabulary", self.vocab);
        let st = &mut *self.st.borrow_mut();
        let (p4, mode): (Vec<u32>, u32) = match pos3 {
            Some(p) => (p.iter().flat_map(|v| v.iter().copied()).collect(), self.mrope_mode),
            None => (segs.iter().flat_map(|(rows, _, base)| (0..rows.len()).flat_map(move |i| { let p = (base + i) as u32; [p, p, p, 0] })).collect(), 0),
        };
        self.gpu.write_bytes(&mut st.mpos, 0, &u32s_bytes(&p4))?;
        let ctls: Vec<u32> = segs.iter().flat_map(|(rows, slot, base)| [*base as u32, (base + rows.len()) as u32, *slot as u32, rows.start as u32]).collect();
        self.gpu.write_bytes(&mut st.ctls, 0, &u32s_bytes(&ctls))?;
        self.gpu.write_bytes(&mut st.ids, 0, &u32s_bytes(tokens))?;
        self.embed(st, m)?;
        for &(row, rows) in given {
            self.gpu.write_bytes(&mut st.x, row * d * 4, &f32_bytes(rows))?;
        }
        let st: &State = st;
        let seg_list: Vec<Seg> = segs.iter().enumerate()
            .map(|(i, (rows, slot, base))| Seg { rows: rows.clone(), slot: *slot, base: *base, ctl: (&st.ctls, 16 * i as u64) }).collect();
        self.run_layers(st, m, mode, false, &seg_list, &mut |_| Ok(()))?;
        let mut out = Vec::with_capacity(reads.len());
        if let (Some(&lo), Some(&hi)) = (reads.iter().min(), reads.iter().max()) {
            // the rows between the first and the last read come along, in one copy
            self.rmsnorm(&st.x, 0, &self.output_norm, &st.h, 0, m)?;
            let mut b = vec![0u8; (hi + 1 - lo) * d * 4];
            self.gpu.read_bytes(&st.h, lo * d * 4, &mut b)?;
            for &r in reads { out.push(bytes_f32(&b[(r - lo) * d * 4..(r + 1 - lo) * d * 4])); }
        }
        self.gpu.sync()?;
        Ok(out)
    }

    // ------------------------------------------------------------- forward

    /// Forward `n` rows (ids or injected residual rows) at cache rows `base..base+n`, in
    /// internal chunks. `pos3 = None` ropes from the scalar cache row. Returns the last row's
    /// logits when asked.
    fn forward(&self, ids: Option<&[u32]>, rows: Option<&[f32]>, n: usize, pos3: Option<&[[u32; 4]]>, base: usize,
               want: Want) -> Result<Option<Vec<f32>>> {
        ensure!(base + n <= self.opts.context,
            "rows {base}..{} exceed the CUDA context of {} (raise -c)", base + n, self.opts.context);
        let t0 = std::time::Instant::now();
        let mut out = None;
        let c = self.opts.chunk;
        let mut c0 = 0;
        while c0 < n {
            if n > 1 && STREAM_CANCEL.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let c1 = (c0 + c).min(n);
            let r = self.forward_chunk(
                ids.map(|v| &v[c0..c1]),
                rows.map(|v| &v[c0 * self.d..c1 * self.d]),
                c1 - c0,
                pos3.map(|p| &p[c0..c1]),
                base + c0,
                if c1 == n { want } else { Want::Nothing },
            )?;
            if c1 == n {
                out = r;
            }
            c0 = c1;
        }
        let dt = t0.elapsed().as_secs_f64();
        let mut tm = self.timing.borrow_mut();
        if n == 1 && ids.is_some() {
            tm.decode_s += dt;
            tm.decode_tokens += 1;
        } else {
            tm.prefill_s += dt;
            tm.prefill_rows += n;
        }
        Ok(out)
    }

    fn forward_chunk(&self, ids: Option<&[u32]>, rows: Option<&[f32]>, m: usize, pos3: Option<&[[u32; 4]]>, base: usize,
                     want: Want) -> Result<Option<Vec<f32>>> {
        let d = self.d;
        let st = &mut *self.st.borrow_mut();
        // positions: (t,h,w,e) per row; mode 0 (plain partial rope from t) without coordinates
        let (p4, mode): (Vec<u32>, u32) = match pos3 {
            Some(p) => (p.iter().flat_map(|v| v.iter().copied()).collect(), self.mrope_mode),
            None => ((0..m).flat_map(|i| { let p = (base + i) as u32; [p, p, p, 0] }).collect(), 0),
        };
        self.gpu.write_bytes(&mut st.mpos, 0, &u32s_bytes(&p4))?;
        self.gpu.write_bytes(&mut st.ctl, 0, &u32s_bytes(&[base as u32, (base + m) as u32, 0, 0]))?;
        match (ids, rows) {
            (Some(ids), _) => {
                self.gpu.write_bytes(&mut st.ids, 0, &u32s_bytes(ids))?;
                self.embed(st, m)?;
            }
            (None, Some(x)) => self.gpu.write_bytes(&mut st.x, 0, &f32_bytes(x))?,
            _ => bail!("forward_chunk needs ids or rows"),
        }
        // which rows the tracer wants
        let (traced, t_hidden, t_logits): (Vec<usize>, bool, bool) = match &*self.tracer.borrow() {
            Some(t) => (
                (0..m).filter(|&i| t.cfg.rows.as_ref().is_none_or(|r| r.contains(&(base + i)))).collect(),
                t.cfg.hidden,
                t.cfg.logits,
            ),
            None => (Vec::new(), false, false),
        };
        let mut hid: Vec<Vec<Vec<f32>>> = vec![Vec::new(); traced.len()];
        let read_rows = |buf: &CuBuf, hid: &mut Vec<Vec<Vec<f32>>>| -> Result<()> {
            for (j, &r) in traced.iter().enumerate() {
                let mut b = vec![0u8; d * 4];
                self.gpu.read_bytes(buf, r * d * 4, &mut b)?;
                hid[j].push(bytes_f32(&b));
            }
            Ok(())
        };
        if t_hidden {
            read_rows(&st.x, &mut hid)?;
        }

        let st: &State = st;
        let one = [Seg { rows: 0..m, slot: 0, base, ctl: (&st.ctl, 0) }];
        self.run_layers(st, m, mode, false, &one, &mut |_l| if t_hidden { read_rows(&st.x, &mut hid) } else { Ok(()) })?;

        // final norm + head for the rows that need it (the last row, and traced rows)
        let head_row = |r: usize| -> Result<()> {
            self.rmsnorm(&st.x, (r * d * 4) as u64, &self.output_norm, &st.h, 0, 1)?;
            self.head_logits(st)
        };
        let read_logits = || -> Result<Vec<f32>> {
            let mut b = vec![0u8; self.vocab * 4];
            self.gpu.read_bytes(&st.logits, 0, &mut b)?;
            Ok(bytes_f32(&b))
        };
        if !traced.is_empty() {
            let mut lg = Vec::with_capacity(traced.len());
            for (j, &r) in traced.iter().enumerate() {
                if t_hidden || t_logits {
                    head_row(r)?;
                }
                if t_hidden {
                    let mut b = vec![0u8; d * 4];
                    self.gpu.read_bytes(&st.h, 0, &mut b)?;
                    hid[j].push(bytes_f32(&b));
                }
                lg.push(if t_logits { Some(read_logits()?) } else { None });
            }
            if let Some(t) = self.tracer.borrow_mut().as_mut() {
                for ((j, &r), l) in traced.iter().enumerate().zip(lg) {
                    t.rows.push(TraceRow {
                        row: base + r,
                        pos: [p4[4 * r], p4[4 * r + 1], p4[4 * r + 2], p4[4 * r + 3]],
                        injected: ids.is_none(),
                        hidden: std::mem::take(&mut hid[j]),
                        logits: l,
                    });
                }
            }
        }
        match want {
            Want::Nothing => {
                self.gpu.sync()?;
                Ok(None)
            }
            Want::Head => {
                head_row(m - 1)?;
                Ok(None)
            }
            Want::Logits => {
                head_row(m - 1)?;
                Ok(Some(read_logits()?))
            }
        }
    }

    /// Every decoder layer over the `m` rows in `st.x` (positions in `st.mpos`), the rows
    /// tiled by `segs`: the projections and every row-wise kernel run once over all rows, and
    /// the work that carries a sequence's state (convolution, recurrence, rope and KV store,
    /// attention) runs per segment at its rows and its slot's state. `fixed_split` sizes
    /// decode attention for the whole context (the CUDA-graph form).
    fn run_layers(&self, st: &State, m: usize, mode: u32, fixed_split: bool, segs: &[Seg],
                  after_layer: &mut dyn FnMut(usize) -> Result<()>) -> Result<()> {
        let qdim = self.n_head * self.hd;
        let mu = m as u32;
        let (conv_bytes, ssm_bytes, kv_bytes) = self.slot_bytes();
        debug_assert!(segs.iter().map(|g| g.rows.len()).sum::<usize>() == m);
        for (l, ly) in self.layers.iter().enumerate() {
            self.rmsnorm(&st.x, 0, &ly.attn_norm, &st.h, 0, m)?;
            match &ly.mixer {
                Mixer::Ssm { win, dt, a, conv_w, norm, wout } => {
                    self.mm(st, &st.h, win, &st.comb, m, false)?;
                    let (nq, nz, hv) = (self.conv_ch, self.d_inner, self.h_v);
                    // one row: the parts in place; a chunk: unpacked to contiguous rows
                    let (qkv, z, al, be) = if m == 1 {
                        ((&st.comb, 0u64), (&st.comb, 4 * nq as u64), (&st.comb, 4 * (nq + nz) as u64), (&st.comb, 4 * (nq + nz + hv) as u64))
                    } else {
                        let tot = m * win.n;
                        self.k("q35_unpack4", &[(&st.comb, 0), (&st.qkv, 0), (&st.z, 0), (&st.alpha, 0), (&st.beta, 0)],
                               &[nq as u32, nz as u32, hv as u32, hv as u32, mu], [blocks(tot, 256), 1, 1], [256, 1, 1])?;
                        ((&st.qkv, 0), (&st.z, 0), (&st.alpha, 0), (&st.beta, 0))
                    };
                    let nab = m * hv;
                    self.k("ssm_ab", &[al, be, (dt, 0), (a, 0)], &[nab as u32, hv as u32], [blocks(nab, 256), 1, 1], [256, 1, 1])?;
                    let cs = st.conv[l].as_ref().unwrap();
                    let ss = st.ssm[l].as_ref().unwrap();
                    // every segment in one launch each, from the table of segments
                    let (tab, nseg) = (segs[0].ctl, segs.len() as u32);
                    self.k("conv1d_prefill", &[qkv, (cs, 0), (conv_w, 0), tab],
                           &[self.conv_ch as u32, self.conv_k as u32, (conv_bytes / 4) as u32], [blocks(self.conv_ch, 128), nseg, 1], [128, 1, 1])?;
                    self.k("deltanet_fused", &[(ss, 0), qkv, al, be, (&st.o, 0), tab],
                           &[self.s_st as u32, self.h_k as u32, self.h_v as u32, self.conv_ch as u32, (ssm_bytes / 4) as u32, self.eps.to_bits()],
                           [(self.s_st / 16) as u32, self.h_v as u32, nseg], [128, 1, 1])?;
                    self.k("gated_rmsnorm", &[(&st.o, 0), (norm, 0), z],
                           &[self.head_v as u32, self.eps.to_bits(), self.d_inner as u32], [self.h_v as u32, mu, 1], [32, 1, 1])?;
                    self.mm(st, &st.o, wout, &st.x, m, true)?;
                }
                Mixer::Attn { wqkv, q_norm, k_norm, wo } => {
                    self.mm(st, &st.h, wqkv, &st.comb, m, false)?;
                    let kvdim = self.n_kv * self.hd;
                    let (qf, kin, vin) = if m == 1 {
                        ((&st.comb, 0u64), (&st.comb, 4 * (2 * qdim) as u64), (&st.comb, 4 * (2 * qdim + kvdim) as u64))
                    } else {
                        let tot = m * wqkv.n;
                        self.k("q35_unpack4", &[(&st.comb, 0), (&st.qfull, 0), (&st.kb, 0), (&st.vb, 0), (&st.vb, 0)],
                               &[(2 * qdim) as u32, kvdim as u32, kvdim as u32, 0, mu], [blocks(tot, 256), 1, 1], [256, 1, 1])?;
                        ((&st.qfull, 0), (&st.kb, 0), (&st.vb, 0))
                    };
                    let kc = st.kc[l].as_ref().unwrap();
                    let vc = st.vc[l].as_ref().unwrap();
                    let [s0, s1, s2, s3] = self.mrope_sections;
                    for g in segs {
                        let (r0, n) = (g.rows.start as u64, g.rows.len());
                        let warps = n * (self.n_head + 2 * self.n_kv);
                        let kv_at = (g.slot * kv_bytes) as u64;
                        self.k(if self.kv_elem() == 2 { "q35_qk_prep_h" } else { "q35_qk_prep" },
                               &[at_rows(qf, r0, 2 * qdim), at_rows(kin, r0, kvdim), at_rows(vin, r0, kvdim), (q_norm, 0), (k_norm, 0), (&st.mpos, r0 * 16),
                                 (&st.q, r0 * qdim as u64 * 4), (kc, kv_at), (vc, kv_at), g.ctl],
                               &[self.hd as u32, self.n_head as u32, self.n_kv as u32, self.n_rot as u32, n as u32,
                                 s0, s1, s2, s3, mode, self.rope_base.to_bits(), self.eps.to_bits()],
                               [blocks(warps, 4), 1, 1], [128, 1, 1])?;
                        self.attention(st, kc, vc, g, fixed_split && segs.len() == 1)?;
                    }
                    self.k("gate_mul_sigmoid", &[(&st.att, 0), qf], &[self.hd as u32, qdim as u32, mu],
                           [blocks(m * qdim, 256), 1, 1], [256, 1, 1])?;
                    self.mm(st, &st.att, wo, &st.x, m, true)?;
                }
            }
            self.rmsnorm(&st.x, 0, &ly.post_norm, &st.h, 0, m)?;
            self.mm(st, &st.h, &ly.gate_up, &st.comb, m, false)?;
            let t = m * self.ffn;
            self.k("q35_swiglu_rows", &[(&st.comb, 0), (&st.g, 0)], &[self.ffn as u32, mu], [blocks(t, 256), 1, 1], [256, 1, 1])?;
            self.mm(st, &st.g, &ly.down, &st.x, m, true)?;
            after_layer(l)?;
        }

        Ok(())
    }

    /// Causal GQA attention of one segment's query rows at its slot's cache rows into
    /// `st.att`. The key range is split (flash-decoding) only for a lone short segment: the
    /// partial buffers are sized for one.
    fn attention(&self, st: &State, kc: &CuBuf, vc: &CuBuf, g: &Seg, fixed_split: bool) -> Result<()> {
        let group = (self.n_head / self.n_kv) as u32;
        let rpt = kernels::qwen35::attn_rows_per_block(group) as usize;
        let kvdim = (self.n_kv * self.hd) as u32;
        let qdim = self.n_head * self.hd;
        let scale = 1.0f32 / (self.hd as f32).sqrt();
        let m = g.rows.len();
        let total = g.base + m;
        let lone = g.rows.start == 0 && g.slot == 0;
        let nsplit = if fixed_split { self.max_split }
            else if lone && m <= SPLIT_MAX_M { total.div_ceil(KSPLIT).clamp(1, self.max_split) } else { 1 };
        let chunk = if nsplit > 1 { KSPLIT } else { total.max(1) };
        let (r0, kv_at) = (g.rows.start as u64, (g.slot * self.slot_bytes().2) as u64);
        self.k(if self.kv_elem() == 2 { "q35_attn_256_h" } else { "q35_attn_256" },
               &[(&st.q, r0 * qdim as u64 * 4), (kc, kv_at), (vc, kv_at), (&st.att, r0 * qdim as u64 * 4), (&st.po, 0), (&st.pml, 0), g.ctl],
               &[kvdim, m as u32, group, self.n_head as u32, chunk as u32, 1, scale.to_bits()],
               [blocks(m, rpt), self.n_kv as u32, nsplit as u32], [128, 1, 1])?;
        if nsplit > 1 {
            self.k("q35_attn_merge", &[(&st.po, 0), (&st.pml, 0), (&st.att, 0)],
                   &[self.hd as u32, self.n_head as u32, m as u32, nsplit as u32], [(m * self.n_head) as u32, 1, 1], [128, 1, 1])?;
        }
        Ok(())
    }

    /// Greedy id of the last forward's logits (on device).
    fn argmax_last(&self) -> Result<u32> {
        let st = self.st.borrow();
        self.k("argmax", &[(&st.logits, 0), (&st.amax, 0)], &[self.vocab as u32], [1, 1, 1], [1024, 1, 1])?;
        let mut b = [0u8; 4];
        self.gpu.read_bytes(&st.amax, 0, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// One greedy decode step. Replays the recorded decode graph (every launch of the step,
    /// with positions read from device memory) unless a trace is running or `OJAS_CUDA_GRAPH=0`.
    pub fn decode_id(&self, token: u32, pos: usize) -> Result<u32> {
        let graphs = std::env::var("OJAS_CUDA_GRAPH").map_or(true, |v| v != "0");
        if !graphs || self.tracer.borrow().is_some() {
            self.forward(Some(&[token]), None, 1, None, pos, Want::Head)?;
            return self.argmax_last();
        }
        ensure!(pos < self.opts.context, "position {pos} is past the CUDA context of {}", self.opts.context);
        let t0 = std::time::Instant::now();
        {
            let st = &mut *self.st.borrow_mut();
            let p = pos as u32;
            self.gpu.write_bytes(&mut st.ids, 0, &u32s_bytes(&[token]))?;
            self.gpu.write_bytes(&mut st.mpos, 0, &u32s_bytes(&[p, p, p, 0]))?;
            self.gpu.write_bytes(&mut st.ctl, 0, &u32s_bytes(&[p, p + 1, 0, 0]))?;
        }
        let mut g = self.graph.borrow_mut();
        if g.is_none() {
            let st = self.st.borrow();
            let rec = self.gpu.capture(&mut |_s| self.decode_body(&st))?;
            *g = Some(rec.context("CUDA graph capture produced no graph")?);
        }
        self.gpu.replay(g.as_ref().unwrap())?;
        let st = self.st.borrow();
        let mut b = [0u8; 4];
        self.gpu.read_bytes(&st.amax, 0, &mut b)?;
        let mut tm = self.timing.borrow_mut();
        tm.decode_s += t0.elapsed().as_secs_f64();
        tm.decode_tokens += 1;
        Ok(u32::from_le_bytes(b))
    }

    /// The launches of one decode step, for graph capture: embed, every layer (attention split
    /// over the whole context), final norm, LM head, on-device argmax.
    fn decode_body(&self, st: &State) -> Result<()> {
        self.embed(st, 1)?;
        let one = [Seg { rows: 0..1, slot: 0, base: 0, ctl: (&st.ctl, 0) }];
        self.run_layers(st, 1, 0, true, &one, &mut |_| Ok(()))?;
        self.rmsnorm(&st.x, 0, &self.output_norm, &st.h, 0, 1)?;
        self.head_logits(st)?;
        self.k("argmax", &[(&st.logits, 0), (&st.amax, 0)], &[self.vocab as u32], [1, 1, 1], [1024, 1, 1])
    }

    /// Forward one token at cache row / scalar position `pos` and return its logits.
    pub fn step_logits(&self, token: u32, pos: usize) -> Result<Vec<f32>> {
        Ok(self.forward(Some(&[token]), None, 1, None, pos, Want::Logits)?.expect("last logits"))
    }

    /// Forward injected rows at explicit coordinates into cache rows `base..`; returns the
    /// last row's logits (the CUDA twin of `CpuSsm::forward_embeds_logits`).
    pub fn forward_embeds_logits(&self, x: &[f32], pos3: Option<&[[u32; 4]]>, base: usize) -> Result<Vec<f32>> {
        let n = x.len() / self.d;
        Ok(self.forward(None, Some(x), n, pos3, base, Want::Logits)?.expect("last logits"))
    }

    /// Encode an image and also return the per-block residual (`layer_out[l]` = `[n_pos, d_v]`
    /// in permuted token order) and the `v.post_ln` output — for parity against `CpuVit`.
    pub fn encode_image_trace(&self, img: &[f32], w: usize, h: usize, want_layers: bool) -> Result<VitOut> {
        let v = self.vit.as_ref().context("no vision tower attached (attach_vit_gguf)")?;
        let t0 = std::time::Instant::now();
        let r = v.encode(self, img, w, h, want_layers);
        self.timing.borrow_mut().vit_s += t0.elapsed().as_secs_f64();
        r
    }
}

impl Model for CudaSsm {
    fn context_capacity(&self) -> usize { self.opts.context }
    fn n_layers(&self) -> usize { self.layers.len() }
    fn hidden_dim(&self) -> usize { self.d }

    fn prefill(&self, tokens: &[u32], base_pos: usize) {
        if tokens.is_empty() { return; }
        if base_pos == 0 { self.reset(); }
        self.forward(Some(tokens), None, tokens.len(), None, base_pos, Want::Nothing).expect("CUDA qwen35 prefill");
    }

    fn prefill_embeds(&self, tokens: &[u32], x: &[f32], base_pos: usize, pos3: Option<&[[u32; 4]]>) -> bool {
        if let Some(p3) = pos3 {
            assert_eq!(p3.len(), tokens.len(), "pos3 must carry one (t,h,w,e) per row");
            if self.mrope_sections.iter().all(|&s| s == 0) { return false; }
        }
        if tokens.is_empty() { return true; }
        assert_eq!(x.len(), tokens.len() * self.d, "prefill_embeds: x must be tokens.len() * hidden_dim");
        if base_pos == 0 { self.reset(); }
        self.forward(None, Some(x), tokens.len(), pos3, base_pos, Want::Nothing).expect("CUDA qwen35 prefill_embeds");
        true
    }

    fn vision_width(&self) -> Option<usize> { self.vit.as_ref().map(|v| v.proj_dim) }

    fn vision_tokens(&self, w: usize, h: usize) -> Option<usize> {
        let v = self.vit.as_ref()?;
        let unit = v.patch * v.merge;
        if w == 0 || h == 0 || w % unit != 0 || h % unit != 0 { return None; }
        Some((w / v.patch) * (h / v.patch) / (v.merge * v.merge))
    }

    fn encode_image(&self, img: &[f32], w: usize, h: usize) -> Option<Result<Vec<f32>>> {
        self.vit.as_ref()?;
        Some(self.encode_image_trace(img, w, h, false).map(|o| o.out))
    }

    fn reset_session(&self) { self.reset(); }

    fn forward_id(&self, token: u32, pos: usize) -> u32 {
        if STREAM_CANCEL.load(Ordering::Relaxed) { return u32::MAX; }
        self.decode_id(token, pos).expect("CUDA qwen35 decode")
    }

    fn forward_logits(&self, token: u32, pos: usize) -> Option<Vec<f32>> {
        if STREAM_CANCEL.load(Ordering::Relaxed) { return None; }
        Some(self.step_logits(token, pos).expect("CUDA qwen35 decode"))
    }
}

// =============================== vision tower ===============================

/// What [`CudaSsm::encode_image_trace`] returns.
pub struct VitOut {
    pub grid: (usize, usize),
    /// `[n_pos, d_v]` after `v.post_ln`, permuted token order.
    pub post_ln: Vec<f32>,
    /// Residual after each block (`want_layers`), `[n_pos, d_v]` each.
    pub layer_out: Vec<Vec<f32>>,
    /// Projector output, `[n_merged, proj_dim]` — what the decoder consumes.
    pub out: Vec<f32>,
}

struct VBlock {
    ln1: (CuBuf, CuBuf),
    ln2: (CuBuf, CuBuf),
    qkv: Lin,
    qkv_b: CuBuf,
    o: Lin,
    o_b: CuBuf,
    up: Lin,
    up_b: CuBuf,
    down: Lin,
    down_b: CuBuf,
}

/// The qwen3vl ViT + merger on CUDA, transcribed from `ojas_cpu::cpu_vit::CpuVit::forward`
/// (and dispatched like Metal's `encode_vit`).
pub struct CudaVit {
    pub n_embd: usize,
    pub n_layers: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub patch: usize,
    pub channels: usize,
    pub pos_side: usize,
    pub merge: usize,
    pub proj_dim: usize,
    pub eps: f32,
    patch_w: Lin,
    patch_b: CuBuf,
    pos_embd: Vec<f32>,
    blocks: Vec<VBlock>,
    post_ln: (CuBuf, CuBuf),
    mm0: (Lin, CuBuf),
    mm2: (Lin, CuBuf),
}

const VIT_ROPE_BASE: f32 = 10000.0;

impl CudaVit {
    pub fn load(gpu: &CudaGpu, g: &mut Gguf) -> Result<CudaVit> {
        ensure!(g.arch() == "clip", "vision tower needs general.architecture=clip (got {})", g.arch());
        let mu = |g: &Gguf, k: &str| g.meta_u32(&format!("clip.vision.{k}")).unwrap_or(0) as usize;
        let n_embd = mu(g, "embedding_length");
        let n_layers = mu(g, "block_count");
        let n_head = mu(g, "attention.head_count");
        let ffn = mu(g, "feed_forward_length");
        let patch = mu(g, "patch_size");
        let image_size = mu(g, "image_size");
        let proj_dim = mu(g, "projection_dim");
        let merge = mu(g, "spatial_merge_size").max(1);
        let eps = g.meta_f32("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6);
        ensure!(n_embd > 0 && n_layers > 0 && n_head > 0 && patch > 0 && image_size > 0, "mmproj is missing clip.vision.* metadata");
        let head_dim = n_embd / n_head;
        ensure!(head_dim % 8 == 0 && head_dim <= 512, "the CUDA tower's attention takes a head_dim that is a multiple of 8 up to 512 (got {head_dim})");
        ensure!(merge == 2, "only spatial_merge_size=2 is implemented (got {merge})");
        ensure!(!g.int_arr("clip.vision.is_deepstack_layers").is_some_and(|v| v.iter().any(|&b| b != 0)),
            "this mmproj declares deepstack layers; the CUDA tower does not implement deepstack");
        let pos_side = image_size / patch;

        let lin = |g: &mut Gguf, name: &str| -> Result<Lin> {
            let (dims, ty, bytes) = g.read_tensor(name).with_context(|| format!("reading {name}"))?;
            let k = dims.first().copied().unwrap_or(1) as usize;
            let b = match ty {
                1 => bytes,
                0 => bytes_f32(&bytes).iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect(),
                t => bail!("{name}: unsupported GGUF type {t}"),
            };
            let n = b.len() / 2 / k.max(1);
            Lin::f16(gpu, &b, n, k)
        };
        let vf = |g: &mut Gguf, name: &str| -> Result<CuBuf> { gpu.upload_bytes(&f32_bytes(&read_f32_tensor(g, name)?)) };
        let ln = |g: &mut Gguf, n: &str| -> Result<(CuBuf, CuBuf)> { Ok((vf(g, &format!("{n}.weight"))?, vf(g, &format!("{n}.bias"))?)) };

        // patch embed: fold the two temporal convs in f32, store f16 (the oracle's tier)
        let (pdims, _, _) = g.read_tensor("v.patch_embd.weight").context("reading v.patch_embd.weight")?;
        ensure!(pdims.len() == 4, "v.patch_embd.weight: expected 4-D, got {pdims:?}");
        let channels = pdims[2] as usize;
        let w0 = read_f32_tensor(g, "v.patch_embd.weight")?;
        let w1 = read_f32_tensor(g, "v.patch_embd.weight.1")?;
        let folded = ojas_cpu::cpu_vit::fold_patch_weights(&w0, &w1)?;
        let fb: Vec<u8> = folded.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();
        let kpatch = channels * patch * patch;
        let patch_w = Lin::f16(gpu, &fb, n_embd, kpatch)?;

        let mut blocks_v = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("v.blk.{i}.{s}");
            blocks_v.push(VBlock {
                ln1: ln(g, &p("ln1"))?,
                ln2: ln(g, &p("ln2"))?,
                qkv: lin(g, &p("attn_qkv.weight"))?,
                qkv_b: vf(g, &p("attn_qkv.bias"))?,
                o: lin(g, &p("attn_out.weight"))?,
                o_b: vf(g, &p("attn_out.bias"))?,
                up: lin(g, &p("ffn_up.weight"))?,
                up_b: vf(g, &p("ffn_up.bias"))?,
                down: lin(g, &p("ffn_down.weight"))?,
                down_b: vf(g, &p("ffn_down.bias"))?,
            });
        }
        let pos_embd = read_f32_tensor(g, "v.position_embd.weight")?;
        ensure!(pos_embd.len() == pos_side * pos_side * n_embd, "v.position_embd.weight size mismatch");
        let v = CudaVit {
            n_embd, n_layers, n_head, head_dim, ffn, patch, channels, pos_side, merge, proj_dim, eps,
            patch_w,
            patch_b: vf(g, "v.patch_embd.bias")?,
            pos_embd,
            blocks: blocks_v,
            post_ln: ln(g, "v.post_ln")?,
            mm0: (lin(g, "mm.0.weight")?, vf(g, "mm.0.bias")?),
            mm2: (lin(g, "mm.2.weight")?, vf(g, "mm.2.bias")?),
        };
        ensure!(v.mm0.0.k == n_embd * merge * merge, "mm.0 takes {} but the 2x2 merge gives {}", v.mm0.0.k, n_embd * merge * merge);
        ensure!(v.mm2.0.n == proj_dim, "mm.2 outputs {} but projection_dim is {proj_dim}", v.mm2.0.n);
        Ok(v)
    }

    fn encode(&self, m: &CudaSsm, img: &[f32], width: usize, height: usize, want_layers: bool) -> Result<VitOut> {
        use ojas_cpu::cpu_vit::{merge_permutation, mrope_positions, resize_position_embeddings};
        let gpu = &m.gpu;
        let (p, dv, hd, nh) = (self.patch, self.n_embd, self.head_dim, self.n_head);
        let step = p * self.merge;
        ensure!(width > 0 && height > 0 && width % step == 0 && height % step == 0,
            "image {width}x{height} must be a non-zero multiple of patch_size*merge = {step}");
        ensure!(img.len() == self.channels * width * height, "image buffer has {} values, expected {}x{height}x{width}", img.len(), self.channels);
        let (pw, ph) = (width / p, height / p);
        let n_pos = pw * ph;
        let n_mm = n_pos / (self.merge * self.merge);
        let kpatch = self.channels * p * p;
        let kmm = dv * self.merge * self.merge;
        let mmh = self.mm0.0.n;
        let pd = self.proj_dim;

        let f = |n: usize| gpu.alloc_bytes(n * 4);
        let imgb = gpu.upload_bytes(&f32_bytes(img))?;
        let rowsb = f((n_pos * kpatch).max(n_mm * mmh))?;
        let x = f(n_pos * dv)?;
        let hb = f(n_pos * dv)?;
        let qkv = f(n_pos * 3 * dv)?;
        let q = f(n_pos * dv)?;
        let kb = f(n_pos * dv)?;
        let vb = f(n_pos * dv)?;
        // The tensor-core kernels are compiled for head_dim 64; any other width streams
        // every key through `attention_m_bidir_span` over one whole-sequence span.
        let attn_mode = if hd == 64 { m.opts.vit_attn } else { VitAttn::F16 };
        let span = if hd == 64 { None } else { Some(gpu.upload_bytes(&u32s_bytes(&vec![0u32, n_pos as u32].repeat(n_pos)))?) };
        let halves = |on: bool| -> Result<Option<CuBuf>> { if on { Ok(Some(gpu.alloc_bytes((n_pos + 64) * dv * 2)?)) } else { Ok(None) } };
        let khb = halves(attn_mode != VitAttn::F32)?;
        let vhb = halves(attn_mode != VitAttn::F32)?;
        let klb = halves(attn_mode == VitAttn::X3)?;
        let vlb = halves(attn_mode == VitAttn::X3)?;
        let ffnb = f((n_pos * self.ffn).max(n_mm * mmh))?;
        let outb = f(n_mm * pd)?;
        let lo_need = n_pos * self.ffn.max(3 * dv).max(kpatch);

        // position embedding, resized and permuted on the host (as Metal)
        let perm = merge_permutation(pw, ph);
        let pe = resize_position_embeddings(&self.pos_embd, dv, self.pos_side, pw, ph);
        let mut pe_perm = vec![0f32; n_pos * dv];
        for (dst, &src) in perm.iter().enumerate() {
            pe_perm[dst * dv..(dst + 1) * dv].copy_from_slice(&pe[src * dv..(src + 1) * dv]);
        }
        let peb = gpu.upload_bytes(&f32_bytes(&pe_perm))?;
        let mut desc: Vec<u32> = vec![(hd / 4) as u32; 4];
        for pp in mrope_positions(pw, ph) {
            desc.extend(pp.iter().map(|&v| v as u32));
        }
        let mposb = gpu.upload_bytes(&u32s_bytes(&desc))?;
        let vctl = gpu.upload_bytes(&u32s_bytes(&[0, n_pos as u32]))?;

        // the matmul helper needs a split-lo buffer as large as the widest [n_pos, k] input
        let lo = if m.opts.gemm == GemmMode::Split { Some(f(lo_need)?) } else { None };
        let mmv = |xb: &CuBuf, w: &Lin, yb: &CuBuf, rows: usize, accum: bool| -> Result<()> {
            let (k, n) = (w.k as u32, w.n as u32);
            let Repr::F16(wb) = &w.w else { bail!("the vision tower's weights are held in f16") };
            if m.opts.gemm == GemmMode::Exact {
                ensure!(!accum, "exact ViT matmuls do not accumulate");
                return m.k("gemv_m_f16", &[(xb, 0), (wb, 0), (yb, 0)], &[k, n, rows as u32], [blocks(w.n, 8), 1, 1], [256, 1, 1]);
            }
            let grid = [blocks(w.n, 128), blocks(rows, 128), 1];
            m.k("gemm_mm_f16", &[(xb, 0), (wb, 0), (yb, 0)], &[k, n, accum as u32, rows as u32], grid, [256, 1, 1])?;
            if let Some(lo) = &lo {
                let tot = rows * w.k;
                m.k("q35_split_lo", &[(xb, 0), (lo, 0)], &[tot as u32], [blocks(tot, 256), 1, 1], [256, 1, 1])?;
                m.k("gemm_mm_f16", &[(lo, 0), (wb, 0), (yb, 0)], &[k, n, 1, rows as u32], grid, [256, 1, 1])?;
            }
            Ok(())
        };
        let el = |n: usize| [blocks(n, 256), 1, 1];
        let bias = |buf: &CuBuf, b: &CuBuf, nn: usize, tot: usize| -> Result<()> {
            m.k("add_rowbias_m", &[(buf, 0), (b, 0)], &[nn as u32, tot as u32], el(tot), [256, 1, 1])
        };
        let (d32, np) = (dv as u32, n_pos as u32);
        let tot = n_pos * dv;

        // 1. patch embed
        m.k("vit_patchify", &[(&imgb, 0), (&rowsb, 0)],
            &[width as u32, height as u32, self.channels as u32, p as u32, (n_pos * kpatch) as u32], el(n_pos * kpatch), [256, 1, 1])?;
        mmv(&rowsb, &self.patch_w, &hb, n_pos, false)?;
        // 2. 2x2 permute before block 0
        m.k("vit_merge_permute", &[(&hb, 0), (&x, 0)], &[d32, pw as u32, tot as u32], el(tot), [256, 1, 1])?;
        // 3. + patch bias, + position embedding
        bias(&x, &self.patch_b, dv, tot)?;
        bias(&x, &peb, tot, tot)?;

        let scale = 1.0f32 / (hd as f32).sqrt();
        let mut layer_out = Vec::new();
        let tmp = if m.opts.gemm == GemmMode::Exact { Some(f(n_pos * dv.max(self.ffn))?) } else { None };
        // x += W·a (+ bias): accumulate on the GEMM, or via a temp in exact mode
        let proj_acc = |a: &CuBuf, w: &Lin, b: &CuBuf| -> Result<()> {
            match &tmp {
                Some(t) => {
                    mmv(a, w, t, n_pos, false)?;
                    bias(t, b, dv, tot)?;
                    m.k("add_inplace", &[(&x, 0), (t, 0)], &[tot as u32], el(tot), [256, 1, 1])
                }
                None => {
                    mmv(a, w, &x, n_pos, true)?;
                    bias(&x, b, dv, tot)
                }
            }
        };
        for blk in &self.blocks {
            m.k("vit_layernorm_m", &[(&x, 0), (&blk.ln1.0, 0), (&hb, 0), (&blk.ln1.1, 0)], &[d32, self.eps.to_bits()], [np, 1, 1], [256, 1, 1])?;
            mmv(&hb, &blk.qkv, &qkv, n_pos, false)?;
            bias(&qkv, &blk.qkv_b, 3 * dv, 3 * tot)?;
            m.k("vit_qkv_split", &[(&qkv, 0), (&q, 0), (&kb, 0), (&vb, 0)], &[d32, tot as u32], el(tot), [256, 1, 1])?;
            let pairs = n_pos * nh * (hd / 2);
            for tgt in [&q, &kb] {
                m.k("vit_rope", &[(tgt, 0), (&mposb, 0)], &[hd as u32, VIT_ROPE_BASE.to_bits(), d32, np], el(pairs), [256, 1, 1])?;
            }
            let (grid, block) = kernels::attn_bidir::mma_bidir_launch(nh as u32, np);
            match (attn_mode, &khb, &vhb, &klb, &vlb) {
                (VitAttn::F32, ..) => {
                    // f32 Q/K/V, bidirectional: q35_attn_64 with causal = 0, group 1
                    let rpt = kernels::qwen35::attn_rows_per_block(1) as usize;
                    m.k("q35_attn_64", &[(&q, 0), (&kb, 0), (&vb, 0), (&hb, 0), (&q, 0), (&q, 0), (&vctl, 0)],
                        &[d32, np, 1, nh as u32, np, 0, scale.to_bits()], [blocks(n_pos, rpt), nh as u32, 1], [128, 1, 1])?;
                }
                (VitAttn::F16, Some(kh), Some(vh), _, _) => {
                    m.k("copy_f32_half", &[(&kb, 0), (kh, 0)], &[tot as u32], el(tot), [256, 1, 1])?;
                    m.k("copy_f32_half", &[(&vb, 0), (vh, 0)], &[tot as u32], el(tot), [256, 1, 1])?;
                    match &span {
                        None => m.k("attention_m_mma_bidir_64", &[(&q, 0), (kh, 0), (vh, 0), (&hb, 0)],
                            &[hd as u32, d32, np, 1, scale.to_bits(), nh as u32, np], grid, block)?,
                        Some(sp) => m.k("attention_m_bidir_span", &[(&q, 0), (kh, 0), (vh, 0), (&hb, 0), (sp, 0)],
                            &[hd as u32, d32, np, 1, scale.to_bits(), nh as u32], [(n_pos * nh) as u32, 1, 1], [256, 1, 1])?,
                    }
                }
                (VitAttn::X3, Some(kh), Some(vh), Some(kl), Some(vl)) => {
                    m.k("q35_split_half", &[(&kb, 0), (kh, 0), (kl, 0)], &[tot as u32], el(tot), [256, 1, 1])?;
                    m.k("q35_split_half", &[(&vb, 0), (vh, 0), (vl, 0)], &[tot as u32], el(tot), [256, 1, 1])?;
                    m.k("q35_vattn_x3_64", &[(&q, 0), (kh, 0), (kl, 0), (vh, 0), (vl, 0), (&hb, 0)],
                        &[d32, np, 1, scale.to_bits(), nh as u32, np], grid, block)?;
                }
                _ => unreachable!("attention buffers are allocated for the mode"),
            }
            proj_acc(&hb, &blk.o, &blk.o_b)?;
            m.k("vit_layernorm_m", &[(&x, 0), (&blk.ln2.0, 0), (&hb, 0), (&blk.ln2.1, 0)], &[d32, self.eps.to_bits()], [np, 1, 1], [256, 1, 1])?;
            mmv(&hb, &blk.up, &ffnb, n_pos, false)?;
            bias(&ffnb, &blk.up_b, self.ffn, n_pos * self.ffn)?;
            m.k("vit_gelu", &[(&ffnb, 0), (&ffnb, 0)], &[(n_pos * self.ffn) as u32], el(n_pos * self.ffn), [256, 1, 1])?;
            proj_acc(&ffnb, &blk.down, &blk.down_b)?;
            if want_layers {
                let mut b = vec![0u8; tot * 4];
                gpu.read_bytes(&x, 0, &mut b)?;
                layer_out.push(bytes_f32(&b));
            }
        }
        // 5. post-LN in place, projector over the 2x2-merged rows (a reshape of x)
        m.k("vit_layernorm_m", &[(&x, 0), (&self.post_ln.0, 0), (&x, 0), (&self.post_ln.1, 0)], &[d32, self.eps.to_bits()], [np, 1, 1], [256, 1, 1])?;
        debug_assert_eq!(self.mm0.0.k, kmm);
        mmv(&x, &self.mm0.0, &ffnb, n_mm, false)?;
        bias(&ffnb, &self.mm0.1, mmh, n_mm * mmh)?;
        m.k("vit_gelu", &[(&ffnb, 0), (&rowsb, 0)], &[(n_mm * mmh) as u32], el(n_mm * mmh), [256, 1, 1])?;
        mmv(&rowsb, &self.mm2.0, &outb, n_mm, false)?;
        bias(&outb, &self.mm2.1, pd, n_mm * pd)?;

        let mut b = vec![0u8; tot * 4];
        gpu.read_bytes(&x, 0, &mut b)?;
        let post_ln = bytes_f32(&b);
        let mut b = vec![0u8; n_mm * pd * 4];
        gpu.read_bytes(&outb, 0, &mut b)?;
        Ok(VitOut { grid: (pw, ph), post_ln, layer_out, out: bytes_f32(&b) })
    }
}
