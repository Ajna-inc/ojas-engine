//! Pure-CPU qwen35 (Qwen3-Next / Gated-DeltaNet hybrid) decoder — SSM on the
//! CPU tier. Faithful to the validated Metal host logic (the ojas-models SSM
//! path) — same math, scalar:
//!
//! - SSM layers ((l+1) % attn_interval != 0): mixed qkv projection →
//!   causal depthwise conv1d (K=4, rolling state) + SiLU → Gated-DeltaNet
//!   recurrence per v-head with fused q/k L2-norm (S_h ← g·S_h + β·k(v−S_hᵀk);
//!   y = S_h·q), gate g = exp(softplus(α+dt)·a), β = sigmoid; then per-head
//!   gated RMSNorm (× silu(z)) and the output projection.
//! - **Attention layers** ((l+1) % interval == 0): gated attention — attn_q
//!   projects per-head [q|gate] chunks; per-head QK-RMSNorm; PARTIAL NEOX
//!   rope (n_rot of head_dim), optionally SECTIONED (M-RoPE, see below); GQA;
//!   attn × sigmoid(gate).
//! - FFN pre-norm is `post_attention_norm` (qwen35 semantics, NOT sandwich).
//!
//! Dense qwen35 only (9B-class, surya-2); qwen35moe bails.
//!
//! ## Weight modes
//!
//! By default F16 matmul weights are repacked to per-row int8 (SDOT fast path).
//! That is a lossy requantization, so it cannot serve as the oracle a GPU port
//! is gated against. The **exact** mode ([`CpuSsmOpts::exact`], or
//! `OJAS_CPU_EXACT=1` for [`CpuSsm::load`]) keeps every F16 tensor as f16 and
//! does all math in f32 (each weight row widened, f32 dot) — the GGUF's own
//! values, no requantization anywhere. F32 tensors are f32 in both modes.
//!
//! ## Sectioned M-RoPE
//!
//! The attention layers rope either from a scalar position (the id path, the
//! decode path, and `prefill_embeds(.., None)`) or from a per-row `(t,h,w,e)`
//! coordinate ([`RopeAt::Sect`]), with the section split from
//! `qwen35.rope.dimension_sections` (surya-2: `[11,11,10,0]` pairs). The
//! selection is a line-for-line transcription of the Metal `mrope_sel`
//! (`ojas-metal/src/kernels/ops.rs`), all four modes ([`MROPE_OFF`],
//! [`MROPE_SECTIONS`], [`MROPE_INTERLEAVED`], [`MROPE_VISION`]); qwen35 is
//! IMROPE, the default. Only the rope ANGLE follows the coordinate — the KV
//! cache row is always `base_pos + i`, exactly as on Metal.
//!
//! ## Oracle hooks
//!
//! [`CpuSsm::trace_start`] / [`CpuSsm::trace_take`] record, per forwarded
//! row, the residual stream at every layer boundary and (optionally) the
//! logits; `OJAS_CPU_TRACE_DIR` streams the same to headerless f32 files so a
//! full `ojas ocr --device cpu` run leaves a comparable dump behind.

use crate::cpu_math::{argmax, matmul, quant_row_i8, rmsnorm, silu, W};
use crate::cpu_vit::CpuVit;
use ojas_core::cancel::STREAM_CANCEL;
use ojas_core::Model as DecoderModel;
use ojas_formats::gguf::Gguf;
use anyhow::{bail, Result};
use half::f16;
use half::slice::HalfFloatSliceExt;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================ M-RoPE contract ===============================
// Mode numbers are the Metal kernel's (`ops.rs` MROPE_*), so a descriptor built
// for one backend means the same thing on the other.

/// Scalar rope: the angle takes `pos[0]` (the t stream) and the plain pair index.
pub const MROPE_OFF: u32 = 0;
/// Contiguous sections `[t t t t | h h | w w]` — qwen2-vl, glm4v.
pub const MROPE_SECTIONS: u32 = 1;
/// Interleaved sections `[t h w t h w …]` — qwen3-vl and **qwen35/surya-2**.
pub const MROPE_INTERLEAVED: u32 = 2;
/// Contiguous sections with theta restarting per section — the ViT layout.
pub const MROPE_VISION: u32 = 3;

/// Which position stream drives rotary pair `j`, and the exponent index its
/// frequency uses. Transcribed from Metal `mrope_sel` (`ops.rs`): `sections`
/// are in cos/sin PAIRS; `pos` is `(t, h, w, e)` for one token.
pub fn mrope_sel(sections: [u32; 4], mode: u32, pos: [u32; 4], j: usize) -> (u32, usize) {
    let [s0, s1, s2, s3] = sections.map(|v| v as usize);
    let sect = s0 + s1 + s2 + s3;
    if sect == 0 { return (pos[0], j); } // no sections declared: t stream, plain theta
    let sector = j % sect;
    let (sel, start) = if mode == MROPE_INTERLEAVED {
        let r = sector % 3;
        let sel = if r == 1 && sector < 3 * s1 { 1 }
            else if r == 2 && sector < 3 * s2 { 2 }
            else if r == 0 && sector < 3 * s0 { 0 }
            else { 3 };
        (sel, 0)
    } else if sector < s0 { (0, 0) }
    else if sector < s0 + s1 { (1, s0) }
    else if sector < s0 + s1 + s2 { (2, s0 + s1) }
    else { (3, s0 + s1 + s2) };
    (pos[sel], if mode == MROPE_VISION { sector - start } else { j })
}

/// Where one row's rope angle comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeAt {
    /// Plain scalar position — the id path and the decode path.
    Scalar(usize),
    /// Sectioned `(t, h, w, e)` coordinate (see `Model::prefill_embeds`).
    Sect([u32; 4]),
}

impl RopeAt {
    /// `(t,h,w,e)` as it would be written to a trace (`Scalar(p)` → `(p,p,p,0)`).
    pub fn as4(self) -> [u32; 4] {
        match self {
            RopeAt::Scalar(p) => [p as u32, p as u32, p as u32, 0],
            RopeAt::Sect(v) => v,
        }
    }
}

/// Text rows at sequence positions `p0 .. p0+n`: `(p, p, p, 0)` each
/// (llama.cpp `llm_graph_input_pos::set_input`).
pub fn text_pos3(p0: u32, n: usize) -> Vec<[u32; 4]> {
    (0..n as u32).map(|i| [p0 + i, p0 + i, p0 + i, 0]).collect()
}

/// Merged image rows of an `nx` x `ny` grid whose span starts at sequence
/// position `pos0`: row `i` → `(pos0, pos0 + i/nx, pos0 + i%nx, 0)`
/// (`mtmd_image_tokens_get_decoder_pos`). The span consumes `max(nx, ny)`
/// sequence positions; the text after it starts at `pos0 + max(nx, ny)`.
pub fn image_pos3(pos0: u32, nx: usize, ny: usize) -> Vec<[u32; 4]> {
    (0..nx * ny)
        .map(|i| [pos0, pos0 + (i / nx) as u32, pos0 + (i % nx) as u32, 0])
        .collect()
}

/// Partial NEOX rope with sectioned positions: rotate the first `rd` dims of
/// one head (pairs `(j, j+rd/2)`); dims ≥ rd pass through. With `t == h == w`
/// (and modes OFF/SECTIONS/INTERLEAVED) this is bit-identical to the scalar
/// rope, because only WHICH position scales the angle changes.
pub fn rope_partial_m(v: &mut [f32], rd: usize, pos: [u32; 4], sections: [u32; 4], mode: u32, base: f32) {
    let half = rd / 2;
    for j in 0..half {
        let (p, je) = if mode == MROPE_OFF { (pos[0], j) } else { mrope_sel(sections, mode, pos, j) };
        let freq = 1.0 / base.powf(2.0 * je as f32 / rd as f32);
        let ang = p as f32 * freq;
        let (s, c) = ang.sin_cos();
        let x0 = v[j];
        let x1 = v[j + half];
        v[j] = x0 * c - x1 * s;
        v[j + half] = x0 * s + x1 * c;
    }
}

/// Partial NEOX rope: rotate only the first `rd` dims of the head
/// (pairs (i, i+rd/2)); dims ≥ rd pass through.
pub fn rope_partial(v: &mut [f32], rd: usize, pos: usize, base: f32) {
    let half = rd / 2;
    for i in 0..half {
        let freq = 1.0 / base.powf(2.0 * i as f32 / rd as f32);
        let ang = pos as f32 * freq;
        let (s, c) = ang.sin_cos();
        let x0 = v[i];
        let x1 = v[i + half];
        v[i] = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
}

// ================================ tracing ===================================

/// What [`CpuSsm::trace_start`] records.
#[derive(Clone, Debug, Default)]
pub struct TraceCfg {
    /// Residual stream at every layer boundary (see [`TraceRow::hidden`]).
    pub hidden: bool,
    /// Logits for every traced row — including PREFILL rows, which otherwise
    /// never reach the LM head. Costs one vocab-wide matvec per traced row.
    pub logits: bool,
    /// Only rows whose KV-cache row falls in this range (None = all).
    pub rows: Option<std::ops::Range<usize>>,
}

/// One forwarded row.
#[derive(Clone, Debug)]
pub struct TraceRow {
    /// KV-cache row (`base_pos + i`).
    pub row: usize,
    /// The rope position the row was forwarded at.
    pub rope: RopeAt,
    /// True when the input row came from `prefill_embeds` rather than `token_embd`.
    pub injected: bool,
    /// `n_layers + 2` vectors of `hidden_dim`: `[0]` the input row (embedding
    /// or injected), `[l+1]` the residual after layer `l`, `[n_layers+1]` the
    /// post-final-norm row (what the LM head reads). Empty unless
    /// `TraceCfg::hidden`.
    pub hidden: Vec<Vec<f32>>,
    pub logits: Option<Vec<f32>>,
}

struct Tracer {
    cfg: TraceCfg,
    rows: Vec<TraceRow>,
    /// When set, rows are appended to files here instead of kept in memory.
    dir: Option<PathBuf>,
}

impl Tracer {
    fn wants(&self, row: usize) -> bool {
        self.cfg.rows.as_ref().map_or(true, |r| r.contains(&row))
    }
    fn push(&mut self, r: TraceRow) {
        match &self.dir {
            None => self.rows.push(r),
            Some(dir) => {
                if let Err(e) = append_trace_row(dir, &r) {
                    tracing::warn!(target: "cpu:qwen35", "trace dump to {} failed: {e}", dir.display());
                }
            }
        }
    }
}

fn append_f32(path: &Path, v: &[f32]) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v { b.extend_from_slice(&x.to_le_bytes()); }
    f.write_all(&b)
}

fn append_trace_row(dir: &Path, r: &TraceRow) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    // rows.u32: (cache row, injected, t, h, w, e) per traced row.
    let p = r.rope.as4();
    let meta = [r.row as u32, r.injected as u32, p[0], p[1], p[2], p[3]];
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("rows.u32"))?;
    let b: Vec<u8> = meta.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.write_all(&b)?;
    for (i, h) in r.hidden.iter().enumerate() {
        append_f32(&dir.join(format!("hidden_{i:03}.f32")), h)?;
    }
    if let Some(l) = &r.logits { append_f32(&dir.join("logits.f32"), l)?; }
    Ok(())
}

/// Write traced rows as headerless little-endian files in `dir` (the
/// `compare_logits.py` convention): `hidden_{k:03}.f32` = `[rows][hidden_dim]`
/// for boundary k (0 = input row, l+1 = after layer l, L+1 = post-final-norm),
/// `logits.f32` = `[rows][vocab]` (rows that have logits), and `rows.u32` =
/// `(cache row, injected, t, h, w, e)` per row. Existing files are appended to.
pub fn write_trace_dir(rows: &[TraceRow], dir: &Path) -> Result<()> {
    for r in rows { append_trace_row(dir, r)?; }
    Ok(())
}

// ================================= model ====================================

enum Mixer {
    Ssm {
        wqkv: W,       // [d → conv_ch]
        wz: W,         // [d → d_inner]   (attn_gate.weight)
        walpha: W,     // [d → H_v]
        wbeta: W,      // [d → H_v]
        dt_bias: Vec<f32>,   // [H_v]
        a: Vec<f32>,         // [H_v]
        conv_w: Vec<f32>,    // [conv_ch × K] (K taps contiguous per channel)
        norm: Vec<f32>,      // [head_v]
        wout: W,       // [d_inner → d]
    },
    Attn {
        wq: W,         // [d → 2*qdim] per-head [q|gate]
        wk: W,         // [d → kvdim]
        wv: W,
        q_norm: Vec<f32>, // [hd]
        k_norm: Vec<f32>,
        wo: W,         // [qdim → d]
    },
}

struct Layer {
    attn_norm: Vec<f32>,
    mixer: Mixer,
    post_norm: Vec<f32>, // post_attention_norm = FFN pre-norm
    ffn_gate: W,
    ffn_up: W,
    ffn_down: W,
}

struct State {
    conv: Vec<Vec<f32>>,      // per SSM layer [(K-1) * conv_ch]
    ssm: Vec<Vec<f32>>,       // per SSM layer [H_v * S * S]
    k: Vec<Vec<f32>>,         // per attn layer flat [row * kvdim]
    v: Vec<Vec<f32>>,
}

/// Load options for [`CpuSsm::load_with`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuSsmOpts {
    /// Keep F16 weights as f16 and compute in f32 (no int8 repack). The oracle mode.
    pub exact: bool,
}

impl CpuSsmOpts {
    /// `OJAS_CPU_EXACT=1` (any value but empty/"0") selects the exact mode.
    pub fn from_env() -> CpuSsmOpts {
        let exact = std::env::var("OJAS_CPU_EXACT").map(|v| !v.is_empty() && v != "0").unwrap_or(false);
        CpuSsmOpts { exact }
    }
}

/// Rows per layer-major prefill chunk. Every row's result is bit-identical to a
/// one-row forward (each matmul output is one fixed-order dot); chunking only
/// amortizes the weight reads.
const CHUNK: usize = 128;

pub struct CpuSsm {
    d: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    n_rot: usize,
    s_st: usize,   // d_state (128)
    h_k: usize,    // n_group: number of k/q heads
    h_v: usize,    // dt_rank: number of v/state heads
    d_inner: usize,
    head_v: usize, // d_inner / h_v
    conv_ch: usize,
    conv_k: usize,
    ffn: usize,
    vocab: usize,
    rope_base: f32,
    eps: f32,
    /// M-RoPE sections in cos/sin PAIRS (`rope.dimension_sections`); all zero =
    /// the model declares none, and `prefill_embeds` refuses coordinates.
    mrope_sections: [u32; 4],
    mrope_mode: u32,
    exact: bool,
    /// GDN grouped-value-attention: v-head → k/q-head. GGUF layout = hh % h_k
    /// (validated vs the reference); HF safetensors = hh / (h_v/h_k)
    /// (repeat_interleave in modeling_qwen3_5).
    kmap: Vec<usize>,
    layers: Vec<Layer>,
    output_norm: Vec<f32>,
    head: W,
    embd: W,
    threads: usize,
    dotprod: bool,
    vit: Option<CpuVit>,
    st: RefCell<State>,
    tracer: RefCell<Option<Tracer>>,
}

fn new_state(layers: &[Layer], conv_k: usize, conv_ch: usize, h_v: usize, s_st: usize) -> State {
    State {
        conv: layers.iter().map(|l| if matches!(l.mixer, Mixer::Ssm { .. }) { vec![0f32; (conv_k - 1) * conv_ch] } else { Vec::new() }).collect(),
        ssm: layers.iter().map(|l| if matches!(l.mixer, Mixer::Ssm { .. }) { vec![0f32; h_v * s_st * s_st] } else { Vec::new() }).collect(),
        k: vec![Vec::new(); layers.len()],
        v: vec![Vec::new(); layers.len()],
    }
}

fn env_tracer() -> Option<Tracer> {
    let dir = std::env::var_os("OJAS_CPU_TRACE_DIR")?;
    let logits = std::env::var("OJAS_CPU_TRACE_LOGITS").map(|v| !v.is_empty() && v != "0").unwrap_or(false);
    tracing::info!(target: "cpu:qwen35", "tracing every row to {} (logits={logits})", Path::new(&dir).display());
    Some(Tracer { cfg: TraceCfg { hidden: true, logits, rows: None }, rows: Vec::new(), dir: Some(dir.into()) })
}

impl CpuSsm {
    /// Load with options from the environment (`OJAS_CPU_EXACT`).
    pub fn load(g: &mut Gguf) -> Result<CpuSsm> {
        Self::load_with(g, CpuSsmOpts::from_env())
    }

    pub fn load_with(g: &mut Gguf, opts: CpuSsmOpts) -> Result<CpuSsm> {
        let arch = g.arch();
        if arch != "qwen35" {
            bail!("CpuSsm supports dense qwen35 for now (got {arch})");
        }
        let exact = opts.exact;
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
        let attn_interval = g.meta_u32("qwen35.full_attention_interval").unwrap_or(4) as usize;
        let n_rot = g.meta_u32("qwen35.rope.dimension_count").unwrap_or(64) as usize;
        let rope_base = g.meta_f32("qwen35.rope.freq_base").unwrap_or(1e6);
        let eps = g.meta_f32("qwen35.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let vocab = g.str_arr("tokenizer.ggml.tokens").map(|t| t.len()).unwrap_or(1).max(1);
        let head_v = d_inner / h_v.max(1);
        let conv_ch = 2 * h_k * s_st + d_inner;
        // Same read as the Metal loader (load.rs): absent => all zero => plain rope.
        let mut mrope_sections = [0u32; 4];
        if let Some(a) = g.int_arr("qwen35.rope.dimension_sections") {
            for (i, v) in a.iter().take(4).enumerate() { mrope_sections[i] = (*v).max(0) as u32; }
        }

        #[cfg(target_arch = "aarch64")]
        let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        #[cfg(not(target_arch = "aarch64"))]
        let dotprod = false;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        tracing::info!(target: "cpu:qwen35", "d={d} L={n_layers} attn {n_head}/{n_kv} hd={hd} rot={n_rot} \
                   sections={mrope_sections:?} | GDN S={s_st} Hk={h_k} Hv={h_v} d_inner={d_inner} conv_k={conv_k} \
                   every={attn_interval} | {}", if exact { "EXACT f16 weights / f32 math".to_string() } else { format!("q8 sdot={dotprod}") });
        let kmap: Vec<usize> = (0..h_v).map(|hh| hh % h_k.max(1)).collect();

        let f16w = |bytes: &[u8], cols: usize| -> W {
            let v: Vec<f16> = bytes.chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
            if exact { return W::F16(v); }
            let mut f = vec![0f32; v.len()];
            v.convert_to_f32_slice(&mut f);
            let rows = f.len() / cols.max(1);
            let mut q = Vec::with_capacity(f.len());
            let mut scale = Vec::with_capacity(rows);
            for r in 0..rows {
                let (qr, sc) = quant_row_i8(&f[r * cols..(r + 1) * cols]);
                q.extend_from_slice(&qr);
                scale.push(sc);
            }
            W::Q8 { q, scale }
        };
        let readw = |g: &mut Gguf, name: &str| -> Result<W> {
            let (dims, ty, bytes) = g.read_tensor(name)?;
            let cols = dims.first().copied().unwrap_or(1) as usize;
            match ty {
                1 => Ok(f16w(&bytes, cols)),
                0 => Ok(W::F32(bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())),
                t => bail!("unexpected type {t} for {name}"),
            }
        };
        let readf32 = |g: &mut Gguf, name: &str| -> Result<Vec<f32>> {
            let (_d, ty, bytes) = g.read_tensor(name)?;
            match ty {
                0 => Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                1 => Ok(bytes.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()),
                t => bail!("unexpected type {t} for {name}"),
            }
        };

        let t0 = std::time::Instant::now();
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            let is_attn = (i + 1) % attn_interval == 0;
            let mixer = if is_attn {
                Mixer::Attn {
                    wq: readw(g, &p("attn_q.weight"))?,
                    wk: readw(g, &p("attn_k.weight"))?,
                    wv: readw(g, &p("attn_v.weight"))?,
                    q_norm: readf32(g, &p("attn_q_norm.weight"))?,
                    k_norm: readf32(g, &p("attn_k_norm.weight"))?,
                    wo: readw(g, &p("attn_output.weight"))?,
                }
            } else {
                Mixer::Ssm {
                    wqkv: readw(g, &p("attn_qkv.weight"))?,
                    wz: readw(g, &p("attn_gate.weight"))?,
                    walpha: readw(g, &p("ssm_alpha.weight"))?,
                    wbeta: readw(g, &p("ssm_beta.weight"))?,
                    dt_bias: readf32(g, &p("ssm_dt.bias"))?,
                    a: readf32(g, &p("ssm_a"))?,
                    conv_w: readf32(g, &p("ssm_conv1d.weight"))?,
                    norm: readf32(g, &p("ssm_norm.weight"))?,
                    wout: readw(g, &p("ssm_out.weight"))?,
                }
            };
            layers.push(Layer {
                attn_norm: readf32(g, &p("attn_norm.weight"))?,
                mixer,
                post_norm: readf32(g, &p("post_attention_norm.weight"))?,
                ffn_gate: readw(g, &p("ffn_gate.weight"))?,
                ffn_up: readw(g, &p("ffn_up.weight"))?,
                ffn_down: readw(g, &p("ffn_down.weight"))?,
            });
        }
        let output_norm = readf32(g, "output_norm.weight")?;
        let head = if g.tensors.contains_key("output.weight") {
            readw(g, "output.weight")?
        } else {
            readw(g, "token_embd.weight")? // tied
        };
        let embd = readw(g, "token_embd.weight")?;
        tracing::info!(target: "cpu:qwen35", "loaded {n_layers} layers in {:.0}s", t0.elapsed().as_secs_f32());

        let st = new_state(&layers, conv_k, conv_ch, h_v, s_st);
        Ok(CpuSsm {
            d, n_head, n_kv, hd, n_rot, s_st, h_k, h_v, d_inner, head_v, conv_ch, conv_k,
            ffn, vocab, rope_base, eps, mrope_sections, mrope_mode: MROPE_INTERLEAVED, exact, kmap,
            layers, output_norm, head, embd, threads, dotprod, vit: None,
            st: RefCell::new(st),
            tracer: RefCell::new(env_tracer()),
        })
    }

    /// Load from an HF safetensors checkpoint dir (Qwen3_5 VL layout, e.g.
    /// 9B-class dense: text decoder under `model.language_model.*`, vision
    /// tower skipped). Weights stay exact f32 (bf16 widened, no re-quant) —
    /// this is the parity/training oracle path, not the fast path.
    pub fn load_safetensors(dir: &str) -> Result<CpuSsm> {
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(format!("{dir}/config.json"))?)?;
        let tc = &cfg["text_config"];
        let u = |v: &serde_json::Value, k: &str| v[k].as_u64().unwrap_or(0) as usize;
        let d = u(tc, "hidden_size");
        let n_layers = u(tc, "num_hidden_layers");
        let n_head = u(tc, "num_attention_heads");
        let n_kv = u(tc, "num_key_value_heads");
        let hd = u(tc, "head_dim");
        let ffn = u(tc, "intermediate_size");
        let s_st = u(tc, "linear_key_head_dim");
        let h_k = u(tc, "linear_num_key_heads");
        let h_v = u(tc, "linear_num_value_heads");
        let d_inner = h_v * u(tc, "linear_value_head_dim");
        let conv_k = u(tc, "linear_conv_kernel_dim");
        let attn_interval = u(tc, "full_attention_interval");
        let vocab = u(tc, "vocab_size");
        let rope_base = tc["rope_parameters"]["rope_theta"].as_f64().unwrap_or(1e7) as f32;
        let partial = tc["rope_parameters"]["partial_rotary_factor"].as_f64().unwrap_or(0.25) as f32;
        let n_rot = (hd as f32 * partial) as usize;
        let eps = tc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let head_v = d_inner / h_v.max(1);
        let conv_ch = 2 * h_k * s_st + d_inner;
        // HF: rope_parameters.mrope_section ([11,11,10] for 64 rot dims); the
        // fourth (e) section is implicit zero, as in the GGUF converter.
        let mut mrope_sections = [0u32; 4];
        if let Some(a) = tc["rope_parameters"]["mrope_section"].as_array() {
            for (i, v) in a.iter().take(4).enumerate() { mrope_sections[i] = v.as_u64().unwrap_or(0) as u32; }
        }

        #[cfg(target_arch = "aarch64")]
        let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        #[cfg(not(target_arch = "aarch64"))]
        let dotprod = false;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        tracing::info!(target: "cpu:qwen35:st", "d={d} L={n_layers} attn {n_head}/{n_kv} hd={hd} rot={n_rot} | \
                   GDN S={s_st} Hk={h_k} Hv={h_v} d_inner={d_inner} conv_k={conv_k} every={attn_interval} | f32 exact");
        let kmap: Vec<usize> = (0..h_v).map(|hh| hh / (h_v / h_k.max(1)).max(1)).collect();

        let t0 = std::time::Instant::now();
        let pre = "model.language_model.";
        let mut ts = ojas_formats::safetensors::load_dir_f32(dir, |n| {
            n.starts_with(pre) || n == "lm_head.weight"
        })?;
        type Ts = HashMap<String, (Vec<usize>, Vec<f32>)>;
        fn take(ts: &mut Ts, name: String) -> Result<(Vec<usize>, Vec<f32>)> {
            ts.remove(&name).ok_or_else(|| anyhow::anyhow!("tensor {name} missing"))
        }
        fn takew(ts: &mut Ts, name: String, out: usize, inp: usize) -> Result<W> {
            let (shape, v) = take(ts, name.clone())?;
            if shape.len() != 2 || shape[0] != out || shape[1] != inp {
                bail!("{name}: shape {shape:?}, expected [{out}, {inp}]");
            }
            Ok(W::F32(v))
        }
        // Qwen3_5RMSNorm stores weights zero-centered: effective scale is
        // (1 + w) (Gemma-style; the GGUF converter bakes the +1 in). Applies
        // to input/post/final norms and attention q/k norms, but not the GDN
        // gated norm (Qwen3_5RMSNormGated uses a plain ones-init weight).
        fn take1p(ts: &mut Ts, name: String) -> Result<Vec<f32>> {
            Ok(take(ts, name)?.1.into_iter().map(|v| v + 1.0).collect())
        }
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("{pre}layers.{i}.{s}");
            let is_attn = (i + 1) % attn_interval == 0;
            let mixer = if is_attn {
                Mixer::Attn {
                    wq: takew(&mut ts, p("self_attn.q_proj.weight"), 2 * n_head * hd, d)?,
                    wk: takew(&mut ts, p("self_attn.k_proj.weight"), n_kv * hd, d)?,
                    wv: takew(&mut ts, p("self_attn.v_proj.weight"), n_kv * hd, d)?,
                    q_norm: take1p(&mut ts, p("self_attn.q_norm.weight"))?,
                    k_norm: take1p(&mut ts, p("self_attn.k_norm.weight"))?,
                    wo: takew(&mut ts, p("self_attn.o_proj.weight"), d, n_head * hd)?,
                }
            } else {
                // gate = softplus(a_proj(x)+dt_bias) * a, with a = -exp(A_log)
                // (the GGUF converter bakes this in as ssm_a; done here instead)
                let a: Vec<f32> = take(&mut ts, p("linear_attn.A_log"))?.1
                    .iter().map(|v| -v.exp()).collect();
                let (cshape, conv_w) = take(&mut ts, p("linear_attn.conv1d.weight"))?;
                if cshape != [conv_ch, 1, conv_k] {
                    bail!("conv1d shape {cshape:?}, expected [{conv_ch}, 1, {conv_k}]");
                }
                Mixer::Ssm {
                    wqkv: takew(&mut ts, p("linear_attn.in_proj_qkv.weight"), conv_ch, d)?,
                    wz: takew(&mut ts, p("linear_attn.in_proj_z.weight"), d_inner, d)?,
                    walpha: takew(&mut ts, p("linear_attn.in_proj_a.weight"), h_v, d)?,
                    wbeta: takew(&mut ts, p("linear_attn.in_proj_b.weight"), h_v, d)?,
                    dt_bias: take(&mut ts, p("linear_attn.dt_bias"))?.1,
                    a,
                    conv_w,
                    norm: take(&mut ts, p("linear_attn.norm.weight"))?.1,
                    wout: takew(&mut ts, p("linear_attn.out_proj.weight"), d, d_inner)?,
                }
            };
            layers.push(Layer {
                attn_norm: take1p(&mut ts, p("input_layernorm.weight"))?,
                mixer,
                post_norm: take1p(&mut ts, p("post_attention_layernorm.weight"))?,
                ffn_gate: takew(&mut ts, p("mlp.gate_proj.weight"), ffn, d)?,
                ffn_up: takew(&mut ts, p("mlp.up_proj.weight"), ffn, d)?,
                ffn_down: takew(&mut ts, p("mlp.down_proj.weight"), d, ffn)?,
            });
        }
        let output_norm = take1p(&mut ts, format!("{pre}norm.weight"))?;
        let head = takew(&mut ts, "lm_head.weight".into(), vocab, d)?;
        let embd = takew(&mut ts, format!("{pre}embed_tokens.weight"), vocab, d)?;
        tracing::info!(target: "cpu:qwen35:st", "loaded {n_layers} layers in {:.0}s", t0.elapsed().as_secs_f32());

        let st = new_state(&layers, conv_k, conv_ch, h_v, s_st);
        Ok(CpuSsm {
            d, n_head, n_kv, hd, n_rot, s_st, h_k, h_v, d_inner, head_v, conv_ch, conv_k,
            ffn, vocab, rope_base, eps, mrope_sections, mrope_mode: MROPE_INTERLEAVED, exact: true, kmap,
            layers, output_norm, head, embd, threads, dotprod, vit: None,
            st: RefCell::new(st),
            tracer: RefCell::new(env_tracer()),
        })
    }

    // ------------------------------------------------------------ accessors

    /// True when weights are kept at the file's precision with f32 math.
    pub fn is_exact(&self) -> bool { self.exact }
    pub fn vocab(&self) -> usize { self.vocab }
    /// `(sections, mode)` the sectioned rope uses.
    pub fn mrope(&self) -> ([u32; 4], u32) { (self.mrope_sections, self.mrope_mode) }
    /// Override the M-RoPE layout (mode is one of the `MROPE_*` constants).
    /// The default is what the file declares, in IMROPE mode (qwen35).
    pub fn set_mrope(&mut self, sections: [u32; 4], mode: u32) {
        assert!(mode <= MROPE_VISION, "unknown M-RoPE mode {mode}");
        self.mrope_sections = sections;
        self.mrope_mode = mode;
    }

    /// The row `token_embd` gathers for `token`, as the decoder sees it
    /// (dequantized in the default mode, the f16 values widened in exact mode).
    /// Injecting these through `prefill_embeds` reproduces an id prefill.
    pub fn embed_row(&self, token: usize) -> Vec<f32> {
        let d = self.d;
        match &self.embd {
            W::Q20 { raw } => {
                let rb = d / 128 * 34;
                crate::cpu_math::dequant_q2_0_row(&raw[token * rb..(token + 1) * rb], d)
            }
            W::Q8 { q, scale } => q[token * d..(token + 1) * d].iter().map(|&b| b as f32 * scale[token]).collect(),
            W::F32(v) => v[token * d..(token + 1) * d].to_vec(),
            W::F16(v) => (0..d).map(|i| v[token * d + i].to_f32()).collect(),
        }
    }

    /// Attach a vision tower so `vision_width` / `vision_tokens` /
    /// `encode_image` answer (a CPU decoder otherwise has no tower and `ojas
    /// ocr` falls back to its own `CpuVitEncoder`, which is the same code).
    pub fn attach_vit(&mut self, vit: CpuVit) -> Result<()> {
        anyhow::ensure!(vit.proj_dim == self.d,
            "ViT projects to {} but the decoder is {}-wide — different checkpoints", vit.proj_dim, self.d);
        self.vit = Some(vit);
        Ok(())
    }

    // -------------------------------------------------------------- tracing

    /// Start recording every forwarded row (prefill, `prefill_embeds` and
    /// decode alike) in memory. Replaces any trace in progress, including an
    /// `OJAS_CPU_TRACE_DIR` one.
    pub fn trace_start(&self, cfg: TraceCfg) {
        *self.tracer.borrow_mut() = Some(Tracer { cfg, rows: Vec::new(), dir: None });
    }

    /// Stop tracing and return what was recorded (in forward order).
    pub fn trace_take(&self) -> Vec<TraceRow> {
        self.tracer.borrow_mut().take().map(|t| t.rows).unwrap_or_default()
    }

    /// Zero all recurrent/KV state (fresh sequence).
    pub fn reset(&self) {
        let st = &mut *self.st.borrow_mut();
        for c in st.conv.iter_mut() { c.iter_mut().for_each(|v| *v = 0.0); }
        for s in st.ssm.iter_mut() { s.iter_mut().for_each(|v| *v = 0.0); }
        for k in st.k.iter_mut() { k.clear(); }
        for v in st.v.iter_mut() { v.clear(); }
    }

    /// Forward one token, recording the residual stream: returns
    /// (trace, logits) where trace = [embedding, after layer 0, ..., after
    /// layer L-1, post-final-norm] — i.e. trace[i] is the input to layer i
    /// for i < L (matches HF `output_hidden_states` indexing; HF's last
    /// entry corresponds to the post-final-norm one).
    pub fn forward_trace(&self, token: usize, pos: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
        let (mut logits, mut hid) = self.forward_rows(vec![self.embed_row(token)], &[RopeAt::Scalar(pos)], pos, true, false, true);
        (hid.take().unwrap().pop().unwrap(), logits.take().unwrap())
    }

    /// Forward rows whose input is supplied directly, at explicit rope
    /// positions, into cache rows `base_row..`. Returns the LAST row's logits.
    /// The general form of `prefill` / `prefill_embeds` / `forward_logits`, for
    /// an oracle driver that wants logits at the end of an injected span.
    pub fn forward_embeds_logits(&self, x: &[f32], rope: &[RopeAt], base_row: usize) -> Vec<f32> {
        assert_eq!(x.len(), rope.len() * self.d, "x must be rope.len() * hidden_dim");
        assert!(!rope.is_empty());
        let n = rope.len();
        let mut last = None;
        for c0 in (0..n).step_by(CHUNK) {
            let c1 = (c0 + CHUNK).min(n);
            let xs = (c0..c1).map(|i| x[i * self.d..(i + 1) * self.d].to_vec()).collect();
            last = self.forward_rows(xs, &rope[c0..c1], base_row + c0, c1 == n, true, false).0;
        }
        last.unwrap()
    }

    fn forward(&self, token: usize, pos: usize, want_logits: bool) -> Option<Vec<f32>> {
        self.forward_rows(vec![self.embed_row(token)], &[RopeAt::Scalar(pos)], pos, want_logits, false, false).0
    }

    /// Prefill rows in layer-major chunks. `x` rows are the residual input.
    fn prefill_rows(&self, x: impl Fn(usize) -> Vec<f32>, n: usize, rope: impl Fn(usize) -> RopeAt,
                    base_pos: usize, injected: bool) {
        for c0 in (0..n).step_by(CHUNK) {
            if STREAM_CANCEL.load(Ordering::Relaxed) { return; }
            let c1 = (c0 + CHUNK).min(n);
            let xs: Vec<Vec<f32>> = (c0..c1).map(&x).collect();
            let rp: Vec<RopeAt> = (c0..c1).map(&rope).collect();
            self.forward_rows(xs, &rp, base_pos + c0, false, injected, false);
        }
    }

    /// Batched `y[r] = W·x[r]`; bit-identical per row to `matvec`.
    fn mm_rows(&self, w: &W, n: usize, k: usize, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let refs: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
        let mut outs = vec![vec![0f32; n]; xs.len()];
        matmul(w, n, k, &refs, None, &mut outs, self.threads, self.dotprod);
        outs
    }

    fn rope_head(&self, seg: &mut [f32], at: RopeAt) {
        match at {
            RopeAt::Scalar(p) => rope_partial(seg, self.n_rot, p, self.rope_base),
            RopeAt::Sect(p4) => rope_partial_m(seg, self.n_rot, p4, self.mrope_sections, self.mrope_mode, self.rope_base),
        }
    }

    /// The forward core: M rows through every layer, layer-major. Row `i` has
    /// input `xs[i]`, rope position `rope[i]`, and KV-cache row `base_row + i`
    /// (the cache is truncated to `base_row` rows first, so re-forwarding a row
    /// overwrites it rather than appending past it). Rows are causal among
    /// themselves exactly as if forwarded one at a time.
    ///
    /// Returns (logits of the LAST row if `last_logits`, per-row hidden trace
    /// if `local_hidden`). Rows selected by an active tracer are recorded too.
    fn forward_rows(&self, xs: Vec<Vec<f32>>, rope: &[RopeAt], base_row: usize, last_logits: bool,
                    injected: bool, local_hidden: bool) -> (Option<Vec<f32>>, Option<Vec<Vec<Vec<f32>>>>) {
        let d = self.d;
        let m = xs.len();
        assert_eq!(rope.len(), m);
        let mut x = xs;
        // Which rows the tracer wants, and what it wants from them.
        let (traced, t_hidden, t_logits): (Vec<bool>, bool, bool) = match &*self.tracer.borrow() {
            Some(t) => ((0..m).map(|i| t.wants(base_row + i)).collect(), t.cfg.hidden, t.cfg.logits),
            None => (vec![false; m], false, false),
        };
        let any_traced = traced.iter().any(|&b| b);
        let keep_hidden = local_hidden || (any_traced && t_hidden);
        let mut hid: Vec<Vec<Vec<f32>>> = if keep_hidden { x.iter().map(|r| vec![r.clone()]).collect() } else { Vec::new() };

        let st = &mut *self.st.borrow_mut();
        for (l, ly) in self.layers.iter().enumerate() {
            let h: Vec<Vec<f32>> = x.iter().map(|r| rmsnorm(r, &ly.attn_norm, self.eps)).collect();
            match &ly.mixer {
                Mixer::Ssm { wqkv, wz, walpha, wbeta, dt_bias, a, conv_w, norm, wout } => {
                    let (s, hk, hv, hvd) = (self.s_st, self.h_k, self.h_v, self.head_v);
                    let mut qkv = self.mm_rows(wqkv, self.conv_ch, d, &h);
                    let z = self.mm_rows(wz, self.d_inner, d, &h);
                    let alpha = self.mm_rows(walpha, hv, d, &h);
                    let beta = self.mm_rows(wbeta, hv, d, &h);
                    // gate = softplus(alpha+dt)·a; β = sigmoid(beta)   (ssm_ab kernel)
                    let mut gate = vec![0f32; m * hv];
                    let mut bet = vec![0f32; m * hv];
                    for r in 0..m {
                        for w in 0..hv {
                            let xg = alpha[r][w] + dt_bias[w];
                            let sp = if xg > 20.0 { xg } else { (1.0 + xg.exp()).ln() };
                            gate[r * hv + w] = sp * a[w];
                            bet[r * hv + w] = 1.0 / (1.0 + (-beta[r][w]).exp());
                        }
                    }
                    // causal conv1d + SiLU + rolling state, row by row   (conv1d kernels)
                    let k_taps = self.conv_k;
                    let cs = &mut st.conv[l];
                    for q in qkv.iter_mut() {
                        for c in 0..self.conv_ch {
                            let mut acc = conv_w[c * k_taps + (k_taps - 1)] * q[c];
                            for j in 0..k_taps - 1 {
                                acc += conv_w[c * k_taps + j] * cs[j * self.conv_ch + c];
                            }
                            for j in 0..k_taps.saturating_sub(2) {
                                cs[j * self.conv_ch + c] = cs[(j + 1) * self.conv_ch + c];
                            }
                            cs[(k_taps - 2) * self.conv_ch + c] = q[c];
                            q[c] = silu(acc);
                        }
                    }
                    // Gated-DeltaNet recurrence (deltanet_fused). Heads are
                    // independent, so each worker owns whole heads and walks
                    // them through the rows in order.
                    let scale = 1.0 / (s as f32).sqrt();
                    let mut o = vec![0f32; m * self.d_inner];
                    let state = &mut st.ssm[l];
                    let (st_addr, o_addr) = (state.as_mut_ptr() as usize, o.as_mut_ptr() as usize);
                    let next = AtomicUsize::new(0);
                    let (qkv, gate, bet) = (&qkv, &gate, &bet);
                    let d_inner = self.d_inner;
                    let eps = self.eps;
                    let kmap = &self.kmap;
                    let body = |_id: usize, _nt: usize| loop {
                        let hh = next.fetch_add(1, Ordering::Relaxed);
                        if hh >= hv { break; }
                        // SAFETY: head hh's state block and its o columns are
                        // touched by exactly this task.
                        let sb = unsafe { std::slice::from_raw_parts_mut((st_addr as *mut f32).add(hh * s * s), s * s) };
                        let khead = kmap[hh];
                        for r in 0..m {
                            let row = &qkv[r];
                            let q = &row[khead * s..(khead + 1) * s];
                            let k = &row[hk * s + khead * s..hk * s + (khead + 1) * s];
                            let v = &row[2 * hk * s + hh * s..2 * hk * s + (hh + 1) * s];
                            let qn = 1.0 / (q.iter().map(|a| a * a).sum::<f32>() + eps).sqrt() * scale;
                            let kn = 1.0 / (k.iter().map(|a| a * a).sum::<f32>() + eps).sqrt();
                            let g = gate[r * hv + hh].exp();
                            let b = bet[r * hv + hh];
                            let orow = unsafe { std::slice::from_raw_parts_mut((o_addr as *mut f32).add(r * d_inner + hh * hvd), hvd) };
                            for col in 0..s {
                                let srow = &mut sb[col * s..(col + 1) * s];
                                let mut sk = 0.0;
                                for j in 0..s {
                                    srow[j] *= g;
                                    sk += srow[j] * k[j];
                                }
                                let dlt = (v[col] - sk * kn) * b;
                                let mut y = 0.0;
                                for j in 0..s {
                                    srow[j] += k[j] * kn * dlt;
                                    y += srow[j] * q[j];
                                }
                                orow[col] = y * qn;
                            }
                        }
                    };
                    let nt = if m * hv * s * s < (1 << 18) { 1 } else { self.threads };
                    crate::cpu_math::parallel(nt, &body);
                    // per-v-head gated RMSNorm × silu(z)   (gated_rmsnorm kernel)
                    let mut orows: Vec<Vec<f32>> = o.chunks_exact(self.d_inner).map(|c| c.to_vec()).collect();
                    for (r, orow) in orows.iter_mut().enumerate() {
                        for hh in 0..hv {
                            let seg = &mut orow[hh * hvd..(hh + 1) * hvd];
                            let ss: f32 = seg.iter().map(|v| v * v).sum::<f32>() / hvd as f32;
                            let inv = 1.0 / (ss + self.eps).sqrt();
                            for i in 0..hvd {
                                let zz = z[r][hh * hvd + i];
                                seg[i] = seg[i] * inv * norm[i] * (zz / (1.0 + (-zz).exp()));
                            }
                        }
                    }
                    let out = self.mm_rows(wout, d, self.d_inner, &orows);
                    for r in 0..m { for i in 0..d { x[r][i] += out[r][i]; } }
                }
                Mixer::Attn { wq, wk, wv, q_norm, k_norm, wo } => {
                    let (nh, nkv, hd) = (self.n_head, self.n_kv, self.hd);
                    let (qdim, kvdim) = (nh * hd, nkv * hd);
                    let qfull = self.mm_rows(wq, 2 * qdim, d, &h);
                    let mut k = self.mm_rows(wk, kvdim, d, &h);
                    let v = self.mm_rows(wv, kvdim, d, &h);
                    // split per-head [q|gate]; per-head QK-RMSNorm; partial NEOX
                    // rope (sectioned when the row carries a coordinate)
                    let mut q = vec![0f32; m * qdim];
                    for r in 0..m {
                        let qr = &mut q[r * qdim..(r + 1) * qdim];
                        for hh in 0..nh {
                            qr[hh * hd..(hh + 1) * hd].copy_from_slice(&qfull[r][hh * 2 * hd..hh * 2 * hd + hd]);
                        }
                        for hh in 0..nh {
                            let seg = &mut qr[hh * hd..(hh + 1) * hd];
                            let ss: f32 = seg.iter().map(|v| v * v).sum::<f32>() / hd as f32;
                            let inv = 1.0 / (ss + self.eps).sqrt();
                            for i in 0..hd { seg[i] *= inv * q_norm[i]; }
                            self.rope_head(seg, rope[r]);
                        }
                        for hh in 0..nkv {
                            let seg = &mut k[r][hh * hd..(hh + 1) * hd];
                            let ss: f32 = seg.iter().map(|v| v * v).sum::<f32>() / hd as f32;
                            let inv = 1.0 / (ss + self.eps).sqrt();
                            for i in 0..hd { seg[i] *= inv * k_norm[i]; }
                            self.rope_head(seg, rope[r]);
                        }
                    }
                    // KV cache: row base_row + r, always contiguous (the rope
                    // coordinate moves the ANGLE only).
                    if st.k[l].len() > base_row * kvdim {
                        st.k[l].truncate(base_row * kvdim);
                        st.v[l].truncate(base_row * kvdim);
                    }
                    let seq0 = st.k[l].len() / kvdim;
                    for r in 0..m {
                        st.k[l].extend_from_slice(&k[r]);
                        st.v[l].extend_from_slice(&v[r]);
                    }
                    let (kc, vc) = (&st.k[l], &st.v[l]);
                    let scale = 1.0 / (hd as f32).sqrt();
                    let group = nh / nkv.max(1);
                    let mut attn = vec![0f32; m * qdim];
                    let a_addr = attn.as_mut_ptr() as usize;
                    let next = AtomicUsize::new(0);
                    let q = &q;
                    let body = |_id: usize, _nt: usize| {
                        let mut scores = Vec::new();
                        loop {
                            let task = next.fetch_add(1, Ordering::Relaxed);
                            if task >= m * nh { break; }
                            let (r, hh) = (task / nh, task % nh);
                            let seq = seq0 + r + 1; // causal: rows up to and including this one
                            let kvh = hh / group;
                            let qh = &q[r * qdim + hh * hd..r * qdim + (hh + 1) * hd];
                            scores.clear();
                            scores.resize(seq, 0f32);
                            for (t, sc) in scores.iter_mut().enumerate() {
                                let kt = &kc[t * kvdim + kvh * hd..t * kvdim + kvh * hd + hd];
                                *sc = crate::cpu_math::dot_f32(qh, kt) * scale;
                            }
                            let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                            let mut denom = 0.0;
                            for sc in scores.iter_mut() { *sc = (*sc - mx).exp(); denom += *sc; }
                            // SAFETY: task (r, hh) owns attn[r*qdim + hh*hd ..][..hd].
                            let out = unsafe { std::slice::from_raw_parts_mut((a_addr as *mut f32).add(r * qdim + hh * hd), hd) };
                            for (i, o) in out.iter_mut().enumerate() {
                                let mut acc = 0.0;
                                for (t, sc) in scores.iter().enumerate() {
                                    acc += sc * vc[t * kvdim + kvh * hd + i];
                                }
                                *o = acc / denom;
                            }
                        }
                    };
                    let work = m * nh * (seq0 + m) * hd;
                    let nt = if work < (1 << 18) { 1 } else { self.threads };
                    crate::cpu_math::parallel(nt, &body);
                    // attn *= sigmoid(per-head gate) — second hd of each qfull chunk
                    let mut arows: Vec<Vec<f32>> = attn.chunks_exact(qdim).map(|c| c.to_vec()).collect();
                    for (r, ar) in arows.iter_mut().enumerate() {
                        for hh in 0..nh {
                            for i in 0..hd {
                                let zz = qfull[r][hh * 2 * hd + hd + i];
                                ar[hh * hd + i] *= 1.0 / (1.0 + (-zz).exp());
                            }
                        }
                    }
                    let o = self.mm_rows(wo, d, qdim, &arows);
                    for r in 0..m { for i in 0..d { x[r][i] += o[r][i]; } }
                }
            }
            // FFN (SwiGLU) with post_attention_norm as pre-norm
            let h2: Vec<Vec<f32>> = x.iter().map(|r| rmsnorm(r, &ly.post_norm, self.eps)).collect();
            let gv = self.mm_rows(&ly.ffn_gate, self.ffn, d, &h2);
            let uv = self.mm_rows(&ly.ffn_up, self.ffn, d, &h2);
            let act: Vec<Vec<f32>> = (0..m).map(|r| (0..self.ffn).map(|i| silu(gv[r][i]) * uv[r][i]).collect()).collect();
            let dv = self.mm_rows(&ly.ffn_down, d, self.ffn, &act);
            for r in 0..m { for i in 0..d { x[r][i] += dv[r][i]; } }
            if keep_hidden { for r in 0..m { hid[r].push(x[r].clone()); } }
        }

        // Final norm + LM head, only for rows that need it.
        let need: Vec<bool> = (0..m).map(|r| (last_logits && r == m - 1) || (traced[r] && t_logits)).collect();
        let mut logits: Vec<Option<Vec<f32>>> = vec![None; m];
        if keep_hidden || need.iter().any(|&b| b) {
            let xn: Vec<Vec<f32>> = x.iter().map(|r| rmsnorm(r, &self.output_norm, self.eps)).collect();
            let idx: Vec<usize> = (0..m).filter(|&r| need[r]).collect();
            if !idx.is_empty() {
                let sel: Vec<Vec<f32>> = idx.iter().map(|&r| xn[r].clone()).collect();
                let out = self.mm_rows(&self.head, self.vocab, d, &sel);
                for (&r, lg) in idx.iter().zip(out) { logits[r] = Some(lg); }
            }
            if keep_hidden { for (r, v) in xn.into_iter().enumerate() { hid[r].push(v); } }
        }
        if any_traced {
            if let Some(t) = self.tracer.borrow_mut().as_mut() {
                for r in 0..m {
                    if !traced[r] { continue; }
                    t.push(TraceRow {
                        row: base_row + r,
                        rope: rope[r],
                        injected,
                        hidden: if t_hidden { hid[r].clone() } else { Vec::new() },
                        logits: if t_logits { logits[r].clone() } else { None },
                    });
                }
            }
        }
        let last = if last_logits { logits[m - 1].take() } else { None };
        (last, if local_hidden { Some(hid) } else { None })
    }
}

impl DecoderModel for CpuSsm {
    fn n_layers(&self) -> usize { self.layers.len() }
    fn hidden_dim(&self) -> usize { self.d }
    fn prefill(&self, tokens: &[u32], base_pos: usize) {
        if tokens.is_empty() { return; }
        if base_pos == 0 { self.reset(); } // fresh sequence, as the Metal prefill does
        self.prefill_rows(|i| self.embed_row(tokens[i] as usize), tokens.len(),
                          |i| RopeAt::Scalar(base_pos + i), base_pos, false);
    }
    fn prefill_embeds(&self, tokens: &[u32], x: &[f32], base_pos: usize,
                      pos3: Option<&[[u32; 4]]>) -> bool {
        if let Some(p3) = pos3 {
            assert_eq!(p3.len(), tokens.len(), "pos3 must carry one (t,h,w,e) per row");
            // No declared sections = nothing to honour the coordinates with; refuse
            // rather than silently drop them (same rule as the Metal decoder).
            if self.mrope_sections.iter().all(|&s| s == 0) { return false; }
        }
        if tokens.is_empty() { return true; }
        assert_eq!(x.len(), tokens.len() * self.d,
            "prefill_embeds: x must be tokens.len() * hidden_dim f32 row-major");
        if base_pos == 0 { self.reset(); }
        let d = self.d;
        self.prefill_rows(|i| x[i * d..(i + 1) * d].to_vec(), tokens.len(),
                          |i| match pos3 { Some(p) => RopeAt::Sect(p[i]), None => RopeAt::Scalar(base_pos + i) },
                          base_pos, true);
        true
    }
    fn vision_width(&self) -> Option<usize> { self.vit.as_ref().map(|v| v.proj_dim) }
    fn vision_tokens(&self, w: usize, h: usize) -> Option<usize> {
        let v = self.vit.as_ref()?;
        let unit = v.patch * v.merge;
        if w == 0 || h == 0 || w % unit != 0 || h % unit != 0 { return None; }
        Some(v.n_merged_tokens(w, h))
    }
    fn encode_image(&self, img: &[f32], w: usize, h: usize) -> Option<Result<Vec<f32>>> {
        Some(self.vit.as_ref()?.forward(img, w, h))
    }
    fn reset_session(&self) { self.reset(); }
    fn forward_id(&self, token: u32, pos: usize) -> u32 {
        if STREAM_CANCEL.load(Ordering::Relaxed) { return u32::MAX; }
        argmax(&self.forward(token as usize, pos, true).unwrap()) as u32
    }
    fn forward_logits(&self, token: u32, pos: usize) -> Option<Vec<f32>> {
        if STREAM_CANCEL.load(Ordering::Relaxed) { return None; }
        self.forward(token as usize, pos, true)
    }
}
