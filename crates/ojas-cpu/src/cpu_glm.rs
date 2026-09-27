//! Pure-CPU GLM-5.2 (glm-dsa) — a 744B MoE chatting with no GPU by treating
//! RAM and disk as one hierarchy.
//!
//! - Dense skeleton in RAM at q8 (~18 GB): MLA attention (q_lora path:
//!   q_a → q_a_norm → q_b), split kv_b (this GGUF stores `attn_k_b`
//!   transposed — absorption layout — so it is transposed per head at load for
//!   the naive path; `attn_v_b` is row-major), router + bias, fused shared
//!   expert, 3 leading dense FFNs, embed/head.
//! - 256 routed experts per MoE layer stay on disk (Q4_K gate/up, Q5_K
//!   down in this GGUF): per token the DeepSeek-V3 sigmoid router picks
//!   top-k, expert slices are `read_at` from the sharded GGUF
//!   (`Gguf::tensor_meta` + expert offset), dequanted, re-quantized to q8
//!   rows, and kept in a byte-budgeted LRU (OJAS_EXPERT_CACHE_GB; the OS
//!   page cache is the free L2 under it).
//! - Router semantics = the GPU moe_topk_v3 kernel: s = sigmoid(logit);
//!   select top-k by s + exp_probs_b bias (lowest-index ties); weights are
//!   the selected s values normalized, × routed_scaling_factor (2.5).
//! - Interleaved-pair RoPE (GLM) — not NEOX split-half. Attention scale
//!   1/√k_mla (no YaRN keys → mscale 1; matches the reference glm implementation).
//! - Scope: dense attention (the DSA lightning indexer is an optimization
//!   — skipped), MTP draft block (blk.78) skipped, naive per-head KV cache
//!   (fine for short contexts).

use crate::cpu_math::{argmax, matvec_kq, matvec as mv, quant_row_i8, rmsnorm, W};
use ojas_core::cancel::STREAM_CANCEL;
use ojas_core::Model as DecoderModel;
use ojas_formats::gguf::Gguf;
use anyhow::{bail, Result};
use half::f16;
use half::slice::HalfFloatSliceExt;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::Ordering;

/// Positional read at an absolute offset, safe to call in parallel on one handle.
#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.read_at(buf, off)
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_read(buf, off)
}

/// On-disk location of one per-expert tensor family (all experts contiguous).
struct ExpMeta {
    part: usize,
    base_off: u64,
    ggml_type: u32,
    bytes_per_expert: u64,
    rows: usize,
    cols: usize,
}

struct MoeLayer {
    gate_inp: W,
    bias: Vec<f32>, // exp_probs_b
    gate_meta: ExpMeta,
    up_meta: ExpMeta,
    down_meta: ExpMeta,
    gate_sh: W,
    up_sh: W,
    down_sh: W,
}

enum Ffn {
    Dense { gate: W, up: W, down: W },
    Moe(MoeLayer),
}

struct Layer {
    attn_norm: Vec<f32>,
    q_a: W,
    q_a_norm: Vec<f32>,
    q_b: W,
    kv_a: W,
    kv_a_norm: Vec<f32>,
    k_b: W, // [nh*nope rows × kv_lora] (transposed from the GGUF's absorption layout)
    v_b: W, // [nh*v_mla rows × kv_lora]
    wo: W,
    ffn_norm: Vec<f32>,
    ffn: Ffn,
}

struct Cached {
    gate_raw: Vec<u8>, // Q4_K blocks, exact as stored
    up_raw: Vec<u8>,
    down_raw: Vec<u8>, // Q5_K in this GGUF
    down_type: u32,    // effective down GGUF type (overridden by OJAS_DOWN_IQ2 requant)
    bytes: usize,
}

pub struct CpuGlm {
    d: usize,
    n_head: usize,
    q_lora: usize,
    kv_lora: usize,
    qk_rope: usize,
    k_mla: usize,
    v_mla: usize,
    nope: usize,
    n_expert: usize,
    n_used: usize,
    ffn_exp: usize,
    routed_scale: f32,
    vocab: usize,
    rope_base: f32,
    eps: f32,
    attn_scale: f32,
    layers: Vec<Layer>,
    output_norm: Vec<f32>,
    head: W,
    embd: W,
    shards: Vec<File>,
    cache_budget: usize,
    threads: usize,
    dotprod: bool,
    kv: RefCell<(Vec<Vec<f32>>, Vec<Vec<f32>>)>, // naive per-head (k, v) per layer
    ecache: RefCell<(HashMap<(u32, u32), Cached>, Vec<(u32, u32)>, usize)>, // map, lru order, bytes
    /// OJAS_CHANNEL_STATS=<path>: per-(layer,expert,channel) decoding-time
    /// energy Σ(w·act)² — the micro-expert census.
    chan_energy: RefCell<std::collections::HashMap<(u32, u32), Vec<f32>>>,
    chan_tokens: std::cell::Cell<u64>,
    h2_energy: RefCell<std::collections::HashMap<u32, Vec<f32>>>, // per-layer ffn-input (d-dim) energy
    h2_tokens: std::cell::Cell<u64>,
    /// OJAS_CHANNEL_PRUNE=<energy.bin>:<pct>: per-expert bottom-pct% channels
    /// by measured energy, zeroed in-memory at compute time (validation mode).
    prune_mask: Option<std::collections::HashMap<(u32, u32), Vec<u32>>>,
    /// OJAS_DOWN_IQ2=<energy.bin>: re-encode down experts IQ3→IQ2 at fetch
    /// (the ~31GB size lever), imatrix = per-expert census channel energy.
    down_imat: Option<std::collections::HashMap<(u32, u32), Vec<f32>>>,
    /// OJAS_EXPERT_PRUNE=<saliency.bin>:<pct>: per-layer set of pruned (bottom
    /// saliency) expert ids, for the REAP-style prune×IQ2 sub-100GB simulation.
    expert_prune: Option<std::collections::HashMap<u32, Vec<u32>>>,
    hits: RefCell<(usize, usize)>, // (hits, loads)
}

impl CpuGlm {
    pub fn load(g: &mut Gguf) -> Result<CpuGlm> {
        let arch = g.arch();
        if arch != "glm-dsa" {
            bail!("CpuGlm supports glm-dsa (got {arch})");
        }
        let mu = |g: &Gguf, k: &str| g.meta_u32(&format!("glm-dsa.{k}")).unwrap_or(0) as usize;
        let d = mu(g, "embedding_length");
        let n_layers = mu(g, "block_count"); // includes the MTP draft block
        let n_head = mu(g, "attention.head_count");
        let q_lora = mu(g, "attention.q_lora_rank");
        let kv_lora = mu(g, "attention.kv_lora_rank");
        let qk_rope = g.meta_u32("glm-dsa.rope.dimension_count").unwrap_or(64) as usize;
        let k_mla = g.meta_u32("glm-dsa.attention.key_length_mla")
            .or_else(|| g.meta_u32("glm-dsa.attention.key_length")).unwrap_or(256) as usize;
        let v_mla = g.meta_u32("glm-dsa.attention.value_length_mla")
            .or_else(|| g.meta_u32("glm-dsa.attention.value_length")).unwrap_or(256) as usize;
        let nope = k_mla - qk_rope;
        let n_expert = mu(g, "expert_count");
        let mut n_used = mu(g, "expert_used_count");
        let ffn_exp = mu(g, "expert_feed_forward_length");
        let leading_dense = mu(g, "leading_dense_block_count");
        let routed_scale = g.meta_f32("glm-dsa.expert_weights_scale").unwrap_or(2.5);
        let rope_base = g.meta_f32("glm-dsa.rope.freq_base").unwrap_or(1e4);
        let eps = g.meta_f32("glm-dsa.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let vocab = g.str_arr("tokenizer.ggml.tokens").map(|t| t.len()).unwrap_or(1).max(1);
        let attn_scale = 1.0 / (k_mla as f32).sqrt(); // no YaRN keys → mscale 1
        // adaptive top-k mirrors the GPU path (LoadOpts.top_k → OJAS_TOPK env)
        if let Some(k) = ojas_core::config::var("OJAS_TOPK").ok().and_then(|v| v.parse::<usize>().ok()) {
            if k > 0 && k < n_used { n_used = k; }
        }
        // main transformer layers only (the NextN/MTP draft block is last)
        let n_main = if g.tensors.contains_key(&format!("blk.{}.nextn.eh_proj.weight", n_layers - 1)) {
            n_layers - 1
        } else {
            n_layers
        };

        #[cfg(target_arch = "aarch64")]
        let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        #[cfg(not(target_arch = "aarch64"))]
        let dotprod = false;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let cache_budget = (ojas_core::config::var("OJAS_EXPERT_CACHE_GB").ok()
            .and_then(|v| v.parse::<f64>().ok()).unwrap_or(32.0) * 1e9) as usize;
        tracing::info!(target: "cpu:glm", "d={d} L={n_main}(+MTP skipped) heads={n_head} q_lora={q_lora} kv_lora={kv_lora} \
                   k={k_mla}(nope {nope}+rope {qk_rope}) v={v_mla} | {n_expert}e top{n_used} scale={routed_scale} \
                   | experts DISK-streamed, q8 LRU {:.0} GB | sdot={dotprod}", cache_budget as f64 / 1e9);

        let quant = |bytes16: &[u8], cols: usize| -> W {
            let v: Vec<f16> = bytes16.chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
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
        // OJAS_CORE_SQUEEZE=1: round-trip core weights through 4-bit
        // (per-32 symmetric int4) at load — simulates the core-squeeze pack
        // stage (attn/shared/dense to ~4 bits) without new storage formats.
        // Norms, embeddings, head, router stay untouched (sensitive set).
        let squeeze_on = ojas_core::config::var("OJAS_CORE_SQUEEZE").is_ok();
        let squeeze4 = move |w: W, name: &str| -> W {
            let target = name.contains("attn_") || name.contains("shexp")
                || name.contains("ffn_gate.") || name.contains("ffn_up.") || name.contains("ffn_down.");
            if !squeeze_on || !target { return w; }
            let round_trip = |row: &mut [f32]| {
                for g32 in row.chunks_mut(32) {
                    let amax = g32.iter().fold(0f32, |m, &v| m.max(v.abs()));
                    if amax == 0.0 { continue; }
                    let sc = amax / 7.0;
                    for v in g32.iter_mut() { *v = (*v / sc).round().clamp(-8.0, 7.0) * sc; }
                }
            };
            match w {
                W::F32(mut v) => { round_trip(&mut v); W::F32(v) }
                W::Q8 { q, scale } => {
                    let cols = q.len() / scale.len();
                    let mut out_q = Vec::with_capacity(q.len());
                    let mut out_s = Vec::with_capacity(scale.len());
                    for (r, &sc) in scale.iter().enumerate() {
                        let mut row: Vec<f32> = q[r * cols..(r + 1) * cols].iter().map(|&b| b as f32 * sc).collect();
                        round_trip(&mut row);
                        let (qr, s2) = quant_row_i8(&row);
                        out_q.extend_from_slice(&qr);
                        out_s.push(s2);
                    }
                    W::Q8 { q: out_q, scale: out_s }
                }
                other => other,
            }
        };
        let readw = |g: &mut Gguf, name: &str| -> Result<W> {
            let (dims, ty, bytes) = g.read_tensor(name)?;
            let cols = dims.first().copied().unwrap_or(1) as usize;
            let w = match ty {
                1 => quant(&bytes, cols),
                0 => W::F32(bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                t => bail!("unexpected type {t} for {name}"),
            };
            Ok(squeeze4(w, name))
        };
        let readf32 = |g: &mut Gguf, name: &str| -> Result<Vec<f32>> {
            let (_d, ty, bytes) = g.read_tensor(name)?;
            match ty {
                0 => Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                1 => Ok(bytes.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()),
                t => bail!("unexpected type {t} for {name}"),
            }
        };
        // attn_k_b is stored [kv_lora rows × nope cols] per head (absorption
        // layout); transpose to [nope rows × kv_lora cols] per head for naive.
        let read_k_b = |g: &mut Gguf, name: &str| -> Result<W> {
            let (dims, ty, bytes) = g.read_tensor(name)?;
            if ty != 1 { bail!("expected f16-dequanted {name}"); }
            let ne0 = dims.first().copied().unwrap_or(1) as usize;       // nope (cols in file)
            let heads = dims.get(2).copied().unwrap_or(1) as usize;
            let v: Vec<f16> = bytes.chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
            let mut f = vec![0f32; v.len()];
            v.convert_to_f32_slice(&mut f);
            let lora = f.len() / heads / ne0;                            // 512
            let mut t = vec![0f32; f.len()];                             // [heads][ne0][lora]
            for h in 0..heads {
                let src = &f[h * lora * ne0..(h + 1) * lora * ne0];
                let dst = &mut t[h * ne0 * lora..(h + 1) * ne0 * lora];
                for l in 0..lora {
                    for i in 0..ne0 {
                        dst[i * lora + l] = src[l * ne0 + i];
                    }
                }
            }
            let rows = heads * ne0;
            let mut q = Vec::with_capacity(t.len());
            let mut scale = Vec::with_capacity(rows);
            for r in 0..rows {
                let (qr, sc) = quant_row_i8(&t[r * lora..(r + 1) * lora]);
                q.extend_from_slice(&qr);
                scale.push(sc);
            }
            Ok(W::Q8 { q, scale })
        };
        // per-expert on-disk metadata (raw bytes stay on disk)
        let exp_meta = |g: &Gguf, name: &str, rows: usize, cols: usize| -> Result<ExpMeta> {
            let Some((part, off, len, ty)) = g.tensor_meta(name) else { bail!("missing {name}") };
            Ok(ExpMeta { part, base_off: off, ggml_type: ty,
                         bytes_per_expert: len / n_expert as u64, rows, cols })
        };

        let t0 = std::time::Instant::now();
        let mut layers = Vec::with_capacity(n_main);
        for i in 0..n_main {
            let p = |s: &str| format!("blk.{i}.{s}");
            let ffn = if i < leading_dense {
                Ffn::Dense {
                    gate: readw(g, &p("ffn_gate.weight"))?,
                    up: readw(g, &p("ffn_up.weight"))?,
                    down: readw(g, &p("ffn_down.weight"))?,
                }
            } else {
                Ffn::Moe(MoeLayer {
                    gate_inp: readw(g, &p("ffn_gate_inp.weight"))?,
                    bias: readf32(g, &p("exp_probs_b.bias"))?,
                    gate_meta: exp_meta(g, &p("ffn_gate_exps.weight"), ffn_exp, d)?,
                    up_meta: exp_meta(g, &p("ffn_up_exps.weight"), ffn_exp, d)?,
                    down_meta: exp_meta(g, &p("ffn_down_exps.weight"), d, ffn_exp)?,
                    gate_sh: readw(g, &p("ffn_gate_shexp.weight"))?,
                    up_sh: readw(g, &p("ffn_up_shexp.weight"))?,
                    down_sh: readw(g, &p("ffn_down_shexp.weight"))?,
                })
            };
            layers.push(Layer {
                attn_norm: readf32(g, &p("attn_norm.weight"))?,
                q_a: readw(g, &p("attn_q_a.weight"))?,
                q_a_norm: readf32(g, &p("attn_q_a_norm.weight"))?,
                q_b: readw(g, &p("attn_q_b.weight"))?,
                kv_a: readw(g, &p("attn_kv_a_mqa.weight"))?,
                kv_a_norm: readf32(g, &p("attn_kv_a_norm.weight"))?,
                k_b: read_k_b(g, &p("attn_k_b.weight"))?,
                v_b: readw(g, &p("attn_v_b.weight"))?,
                wo: readw(g, &p("attn_output.weight"))?,
                ffn_norm: readf32(g, &p("ffn_norm.weight"))?,
                ffn,
            });
            tracing::debug!(target: "cpu:glm", "skeleton {}/{} layers ({:.0}s)", i + 1, n_main, t0.elapsed().as_secs_f32());
        }
        let output_norm = readf32(g, "output_norm.weight")?;
        let head = readw(g, "output.weight")?;
        let embd = readw(g, "token_embd.weight")?;
        tracing::info!(target: "cpu:glm", "skeleton loaded: {n_main} layers in {:.0}s", t0.elapsed().as_secs_f32());

        let shards: Vec<File> = g.shard_paths().iter().map(File::open).collect::<std::io::Result<_>>()?;
        Ok(CpuGlm {
            d, n_head, q_lora, kv_lora, qk_rope, k_mla, v_mla, nope, n_expert, n_used, ffn_exp,
            routed_scale, vocab, rope_base, eps, attn_scale,
            layers, output_norm, head, embd, shards, cache_budget, threads, dotprod,
            kv: RefCell::new((vec![Vec::new(); n_main], vec![Vec::new(); n_main])),
            ecache: RefCell::new((HashMap::new(), Vec::new(), 0)),
            chan_energy: RefCell::new(std::collections::HashMap::new()),
            chan_tokens: std::cell::Cell::new(0),
            h2_energy: RefCell::new(std::collections::HashMap::new()),
            h2_tokens: std::cell::Cell::new(0),
            prune_mask: load_prune_mask(ffn_exp),
            down_imat: load_down_imat(ffn_exp),
            expert_prune: load_expert_prune(n_expert),
            hits: RefCell::new((0, 0)),
        })
    }

    /// Ensure all `wanted` experts of layer `l` are in the q8 LRU.
    /// Missing experts are fetched from disk concurrently — one worker per
    /// expert, gate/up/down in parallel inside each worker (the serial
    /// per-expert dequant→requant pipeline was ~80% of a token).
    fn ensure_experts(&self, l: u32, wanted: &[usize], m: &MoeLayer) {
        let missing: Vec<u32> = {
            let (map, order, _) = &mut *self.ecache.borrow_mut();
            let mut miss = Vec::new();
            for &e in wanted {
                let key = (l, e as u32);
                if map.contains_key(&key) {
                    if let Some(i) = order.iter().position(|&k| k == key) { order.remove(i); }
                    order.push(key);
                    self.hits.borrow_mut().0 += 1;
                } else {
                    miss.push(e as u32);
                }
            }
            miss
        };
        if missing.is_empty() { return; }
        self.hits.borrow_mut().1 += missing.len();
        let shards = &self.shards;
        let read_raw = move |meta: &ExpMeta, e: u32| -> Vec<u8> {
            let mut raw = vec![0u8; meta.bytes_per_expert as usize];
            let _ = read_at(
                &shards[meta.part],
                &mut raw,
                meta.base_off + e as u64 * meta.bytes_per_expert,
            );
            raw
        };
        // parallel reads for the missing experts (pure I/O — the direct
        // K-quant dot removes the per-fetch dequant->requant step)
        let fetched: Vec<(u32, Cached)> = std::thread::scope(|sc| {
            let handles: Vec<_> = missing.iter().map(|&e| {
                let read_raw = &read_raw;
                let down_imat = self.down_imat.as_ref();
                sc.spawn(move || {
                    let gate_raw = read_raw(&m.gate_meta, e);
                    let up_raw = read_raw(&m.up_meta, e);
                    let mut down_raw = read_raw(&m.down_meta, e);
                    let mut down_type = m.down_meta.ggml_type;
                    // OJAS_DOWN_IQ2 sim: re-encode this expert's down IQ3→IQ2 (the
                    // ~31GB size lever). imatrix = census channel energy (the down
                    // input space); uniform fallback. Cached so it's paid once/expert.
                    if let Some(imat) = down_imat {
                        let d = m.down_meta.rows; let ffn = m.down_meta.cols;
                        let f = crate::cpu_math::dequant_iq(down_type, &down_raw, d * ffn);
                        let iw = imat.get(&(l, e));
                        let ones = vec![1f32; ffn];
                        let w = iw.map(|v| v.as_slice()).unwrap_or(&ones);
                        let mut out = Vec::with_capacity(d / 256 * 66 * (ffn / 256).max(1) * 256);
                        out.clear();
                        for r in 0..d {
                            let row = ojas_formats::iq_encode::encode_iq2_xxs(&f[r * ffn..(r + 1) * ffn], ffn, w);
                            out.extend_from_slice(&row);
                        }
                        down_raw = out; down_type = 16;
                    }
                    let bytes = gate_raw.len() + up_raw.len() + down_raw.len();
                    (e, Cached { gate_raw, up_raw, down_raw, down_type, bytes })
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (map, order, bytes) = &mut *self.ecache.borrow_mut();
        for (e, c) in fetched {
            *bytes += c.bytes;
            map.insert((l, e), c);
            order.push((l, e));
        }
        while *bytes > self.cache_budget && order.len() > 1 {
            let victim = order.remove(0);
            if let Some(old) = map.remove(&victim) { *bytes -= old.bytes; }
        }
    }

    fn forward(&self, token: usize, pos: usize, want_logits: bool) -> Option<Vec<f32>> {
        let (d, nh) = (self.d, self.n_head);
        let (nope, rd, k_mla, v_mla) = (self.nope, self.qk_rope, self.k_mla, self.v_mla);
        let mut x: Vec<f32> = match &self.embd {
            W::Q8 { q, scale } => q[token * d..(token + 1) * d].iter().map(|&b| b as f32 * scale[token]).collect(),
            W::F32(v) => v[token * d..(token + 1) * d].to_vec(),
            W::F16(v) => (0..d).map(|i| v[token * d + i].to_f32()).collect(),
            W::Q20 { raw } => {
                let rb = d / 128 * 34;
                crate::cpu_math::dequant_q2_0_row(&raw[token * rb..(token + 1) * rb], d)
            }
        };
        let (kc, vc) = &mut *self.kv.borrow_mut();

        for (l, ly) in self.layers.iter().enumerate() {
            // ---- MLA attention (q_lora path, naive cache) ----
            let h = rmsnorm(&x, &ly.attn_norm, self.eps);
            let qa = mv(&ly.q_a, self.q_lora, d, &h, None, self.threads, self.dotprod);
            let qa = rmsnorm(&qa, &ly.q_a_norm, self.eps);
            let mut q = mv(&ly.q_b, nh * k_mla, self.q_lora, &qa, None, self.threads, self.dotprod);
            let kv_a = mv(&ly.kv_a, self.kv_lora + rd, d, &h, None, self.threads, self.dotprod);
            let ckv = rmsnorm(&kv_a[..self.kv_lora], &ly.kv_a_norm, self.eps);
            let mut kr = kv_a[self.kv_lora..].to_vec();
            rope_interleaved(&mut kr, rd, pos, self.rope_base);
            let kn = mv(&ly.k_b, nh * nope, self.kv_lora, &ckv, None, self.threads, self.dotprod);
            let vv = mv(&ly.v_b, nh * v_mla, self.kv_lora, &ckv, None, self.threads, self.dotprod);
            for hh in 0..nh {
                rope_interleaved(&mut q[hh * k_mla + nope..(hh + 1) * k_mla], rd, pos, self.rope_base);
            }
            for hh in 0..nh {
                kc[l].extend_from_slice(&kn[hh * nope..(hh + 1) * nope]);
                kc[l].extend_from_slice(&kr);
                vc[l].extend_from_slice(&vv[hh * v_mla..(hh + 1) * v_mla]);
            }
            let seq = kc[l].len() / (nh * k_mla);
            let mut ao = vec![0f32; nh * v_mla];
            for hh in 0..nh {
                let qh = &q[hh * k_mla..(hh + 1) * k_mla];
                let mut scores = vec![0f32; seq];
                for (t, s) in scores.iter_mut().enumerate() {
                    let kt = &kc[l][(t * nh + hh) * k_mla..(t * nh + hh + 1) * k_mla];
                    *s = crate::cpu_math::dot_f32(qh, kt) * self.attn_scale;
                }
                let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                let mut denom = 0.0;
                for s in scores.iter_mut() { *s = (*s - mx).exp(); denom += *s; }
                for i in 0..v_mla {
                    let mut acc = 0.0;
                    for (t, s) in scores.iter().enumerate() {
                        acc += s * vc[l][(t * nh + hh) * v_mla + i];
                    }
                    ao[hh * v_mla + i] = acc / denom;
                }
            }
            let o = mv(&ly.wo, d, nh * v_mla, &ao, None, self.threads, self.dotprod);
            for i in 0..d { x[i] += o[i]; }

            // ---- FFN ----
            let h2 = rmsnorm(&x, &ly.ffn_norm, self.eps);
            // OJAS_H2_DUMP=<layer>:<path>: append raw ffn-input vectors (d f32) for
            // one layer — real activation samples X for per-expert healing.
            if let Ok(spec) = ojas_core::config::var("OJAS_H2_DUMP") {
                if let Some((tl, path)) = spec.split_once(':') {
                    if tl.parse::<usize>() == Ok(l) {
                        use std::io::Write;
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(h2.as_ptr() as *const u8, h2.len() * 4) };
                            let _ = f.write_all(bytes);
                        }
                    }
                }
            }
            match &ly.ffn {
                Ffn::Dense { gate, up, down } => {
                    let n = match gate { W::Q8 { scale, .. } => scale.len(), W::F32(v) => v.len() / d, W::F16(v) => v.len() / d,
                        W::Q20 { raw } => raw.len() / (d / 128 * 34) };
                    let gv = mv(gate, n, d, &h2, None, self.threads, self.dotprod);
                    let uv = mv(up, n, d, &h2, None, self.threads, self.dotprod);
                    let act: Vec<f32> = (0..n).map(|i| slu(gv[i]) * uv[i]).collect();
                    let dv = mv(down, d, n, &act, None, self.threads, self.dotprod);
                    for i in 0..d { x[i] += dv[i]; }
                }
                Ffn::Moe(m) => {
                    if ojas_core::config::var("OJAS_H2_STATS").is_ok() {
                        let mut he = self.h2_energy.borrow_mut();
                        let v = he.entry(l as u32).or_insert_with(|| vec![0f32; d]);
                        for i in 0..d { v[i] += h2[i] * h2[i]; }
                    }
                    // DeepSeek-V3 sigmoid router (GPU moe_topk_v3 semantics)
                    let logits = mv(&m.gate_inp, self.n_expert, d, &h2, None, self.threads, self.dotprod);
                    let sig: Vec<f32> = logits.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
                    let mut choose: Vec<f32> = sig.iter().zip(&m.bias).map(|(s, b)| s + b).collect();
                    // OJAS_EXPERT_PRUNE: mask bottom-pct% saliency experts of this layer
                    // (REAP-style) so top-k selects only survivors — the sub-100GB
                    // prune×IQ2 sim. Pruned experts get score -inf (never selected).
                    if let Some(pm) = &self.expert_prune {
                        if let Some(dead) = pm.get(&(l as u32)) {
                            for &e in dead { choose[e as usize] = f32::MIN; }
                        }
                    }
                    let mut picks: Vec<(usize, f32)> = Vec::with_capacity(self.n_used);
                    for _ in 0..self.n_used {
                        let (mut bi, mut bv) = (0usize, f32::MIN);
                        for (e, &cv) in choose.iter().enumerate() {
                            if cv > bv { bv = cv; bi = e; }
                        }
                        picks.push((bi, sig[bi]));
                        choose[bi] = f32::MIN;
                    }
                    let sum: f32 = picks.iter().map(|&(_, s)| s).sum();
                    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
                    let wanted: Vec<usize> = picks.iter().map(|&(e, _)| e).collect();
                    self.ensure_experts(l as u32, &wanted, m);
                    let mut moe = vec![0f32; d];
                    for (pi, &(e, s)) in picks.iter().enumerate() {
                        let cache = self.ecache.borrow();
                        let Some(c) = cache.0.get(&(l as u32, e as u32)) else { continue };
                        let gv = matvec_kq(&c.gate_raw, m.gate_meta.ggml_type, self.ffn_exp, d, &h2, self.threads, self.dotprod);
                        let uv = matvec_kq(&c.up_raw, m.up_meta.ggml_type, self.ffn_exp, d, &h2, self.threads, self.dotprod);
                        let act: Vec<f32> = (0..self.ffn_exp).map(|i| slu(gv[i]) * uv[i]).collect();
                        if pi == 0 && ojas_core::config::var("OJAS_LAYER_DUMP").is_ok() {
                            let al2 = act.iter().map(|v| v * v).sum::<f32>().sqrt();
                            tracing::trace!(target: "adump", "l={l} act_l2={al2:.4} act={:?}", &act[..4]);
                        }
                        let dv = matvec_kq(&c.down_raw, c.down_type, d, self.ffn_exp, &act, self.threads, self.dotprod);
                        let wgt = s * inv * self.routed_scale;
                        if let Some(pm) = &self.prune_mask {
                            if let Some(dead) = pm.get(&(l as u32, e as u32)) {
                                let mut act = act;
                                for &i in dead { act[i as usize] = 0.0; }
                                let dv = matvec_kq(&c.down_raw, c.down_type, d, self.ffn_exp, &act, self.threads, self.dotprod);
                                let wgt = s * inv * self.routed_scale;
                                for i in 0..d { moe[i] += wgt * dv[i]; }
                                continue;
                            }
                        }
                        if ojas_core::config::var("OJAS_CHANNEL_STATS").is_ok() {
                            let mut ce = self.chan_energy.borrow_mut();
                            let v = ce.entry((l as u32, e as u32)).or_insert_with(|| vec![0f32; self.ffn_exp]);
                            for i in 0..self.ffn_exp {
                                let a = wgt * act[i];
                                v[i] += a * a;
                            }
                        }
                        for i in 0..d { moe[i] += wgt * dv[i]; }
                    }
                    // shared expert, weight 1
                    let ns = match &m.gate_sh { W::Q8 { scale, .. } => scale.len(), W::F32(v) => v.len() / d, W::F16(v) => v.len() / d,
                        W::Q20 { raw } => raw.len() / (d / 128 * 34) };
                    let gv = mv(&m.gate_sh, ns, d, &h2, None, self.threads, self.dotprod);
                    let uv = mv(&m.up_sh, ns, d, &h2, None, self.threads, self.dotprod);
                    let act: Vec<f32> = (0..ns).map(|i| slu(gv[i]) * uv[i]).collect();
                    let sv = mv(&m.down_sh, d, ns, &act, None, self.threads, self.dotprod);
                    if ojas_core::config::var("OJAS_LAYER_DUMP").is_ok() {
                        let we: Vec<f32> = picks.iter().map(|&(_, s)| s * inv * self.routed_scale).collect();
                        let ie: Vec<usize> = picks.iter().map(|&(e, _)| e).collect();
                        let shl2 = sv.iter().map(|v| v * v).sum::<f32>().sqrt();
                        tracing::trace!(target: "rdump", "l={l} idx={ie:?} wgt={we:?} sh_l2={shl2:.4} sh={:?} moe={:?}", &sv[..4], &moe[..4]);
                    }
                    for i in 0..d { x[i] += moe[i] + sv[i]; }
                }
            }
            // OJAS_LAYER_DUMP=1: per-layer residual fingerprint (same format as the
            // Metal stream path — diff the logs to bisect CPU-vs-GPU divergence).
            if ojas_core::config::var("OJAS_LAYER_DUMP").is_ok() {
                let l2 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
                tracing::trace!(target: "ldump", "pos={pos} l={l} l2={l2:.4} x={:?}", &x[..4]);
            }
        }
        // OJAS_H2_STATS=<path>: accumulate per-layer ffn-input (h2, d-dim) channel
        // energy — the gate/up-side weighting for compensator fitting (the channel
        // census covers only the 2048-dim down-input space). Layer-level, expert-
        // independent → converges in a few hundred tokens.
        if let Ok(sp) = ojas_core::config::var("OJAS_H2_STATS") {
            let t = self.h2_tokens.get() + 1;
            self.h2_tokens.set(t);
            if t % 8 == 0 {
                let ce = self.h2_energy.borrow();
                let mut out = Vec::with_capacity(ce.len() * (4 + self.d * 4));
                for (&l, v) in ce.iter() {
                    out.extend_from_slice(&l.to_le_bytes());
                    for &f in v { out.extend_from_slice(&f.to_le_bytes()); }
                }
                let _ = std::fs::write(&sp, out);
            }
        }
        if let Ok(sp) = ojas_core::config::var("OJAS_CHANNEL_STATS") {
            let t = self.chan_tokens.get() + 1;
            self.chan_tokens.set(t);
            if t % 8 == 0 {
                // binary dump: [u32 l][u32 e][ffn_exp × f32 energy] records
                let ce = self.chan_energy.borrow();
                let mut out = Vec::with_capacity(ce.len() * (8 + self.ffn_exp * 4));
                for (&(l, e), v) in ce.iter() {
                    out.extend_from_slice(&l.to_le_bytes());
                    out.extend_from_slice(&e.to_le_bytes());
                    for &f in v { out.extend_from_slice(&f.to_le_bytes()); }
                }
                let _ = std::fs::write(&sp, out);
            }
        }
        if !want_logits { return None; }
        let xn = rmsnorm(&x, &self.output_norm, self.eps);
        Some(mv(&self.head, self.vocab, d, &xn, None, self.threads, self.dotprod))
    }
}

impl DecoderModel for CpuGlm {
    fn n_layers(&self) -> usize { self.layers.len() }
    fn hidden_dim(&self) -> usize { self.d }
    fn prefill(&self, tokens: &[u32], base_pos: usize) {
        if base_pos == 0 {
            // new sequence: drop stale KV from prior generates (avoids
            // cross-prompt contamination)
            let (kc, vc) = &mut *self.kv.borrow_mut();
            for k in kc.iter_mut() { k.clear(); }
            for v in vc.iter_mut() { v.clear(); }
        }
        for (i, &t) in tokens.iter().enumerate() {
            if STREAM_CANCEL.load(Ordering::Relaxed) { return; }
            self.forward(t as usize, base_pos + i, false);
        }
    }
    fn forward_id(&self, token: u32, pos: usize) -> u32 {
        if STREAM_CANCEL.load(Ordering::Relaxed) { return u32::MAX; }
        let r = argmax(&self.forward(token as usize, pos, true).unwrap()) as u32;
        let (h, ld) = *self.hits.borrow();
        tracing::debug!(target: "cpu:glm", "pos {pos}: expert cache {h} hits / {ld} disk loads");
        r
    }
    fn forward_logits(&self, token: u32, pos: usize) -> Option<Vec<f32>> {
        if STREAM_CANCEL.load(Ordering::Relaxed) { return None; }
        self.forward(token as usize, pos, true)
    }
}

fn slu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// GLM interleaved-pair RoPE: rotate (v[2i], v[2i+1]) — not NEOX split-half.
fn rope_interleaved(v: &mut [f32], rd: usize, pos: usize, base: f32) {
    for i in 0..rd / 2 {
        let freq = 1.0 / base.powf(2.0 * i as f32 / rd as f32);
        let ang = pos as f32 * freq;
        let (s, c) = ang.sin_cos();
        let x0 = v[2 * i];
        let x1 = v[2 * i + 1];
        v[2 * i] = x0 * c - x1 * s;
        v[2 * i + 1] = x0 * s + x1 * c;
    }
}


/// OJAS_EXPERT_PRUNE=<saliency.bin>:<pct> → per-layer list of the bottom-pct%
/// experts by saliency (Σ gate-weighted output energy). REAP-style prune sim:
/// masked in the router so top-k picks only survivors. saliency.bin = records
/// of [u32 layer][u32 expert][f32 saliency].
fn load_expert_prune(n_expert: usize) -> Option<std::collections::HashMap<u32, Vec<u32>>> {
    let spec = ojas_core::config::var("OJAS_EXPERT_PRUNE").ok()?;
    let (path, pct) = spec.rsplit_once(':')?;
    let pct: f32 = pct.parse().ok()?;
    let raw = std::fs::read(path).ok()?;
    // gather saliency per layer
    let mut by_layer: std::collections::HashMap<u32, Vec<(u32, f32)>> = std::collections::HashMap::new();
    for r in raw.chunks_exact(12) {
        let l = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
        let e = u32::from_le_bytes([r[4], r[5], r[6], r[7]]);
        let s = f32::from_le_bytes([r[8], r[9], r[10], r[11]]);
        by_layer.entry(l).or_default().push((e, s));
    }
    let mut map = std::collections::HashMap::new();
    let mut total = 0usize;
    for (l, mut es) in by_layer {
        es.sort_by(|a, b| a.1.total_cmp(&b.1)); // ascending saliency
        let n_drop = (n_expert as f32 * pct / 100.0) as usize;
        let dead: Vec<u32> = es.iter().take(n_drop).map(|&(e, _)| e).collect();
        total += dead.len();
        map.insert(l, dead);
    }
    tracing::info!(target: "cpu:glm", "EXPERT_PRUNE: bottom {pct}% saliency → {total} experts masked over {} layers", map.len());
    Some(map)
}

/// OJAS_DOWN_IQ2=<energy.bin> → per-(layer,expert) imatrix (census channel
/// energy) used as quant_weights when re-encoding down IQ3→IQ2 at fetch.
fn load_down_imat(ffn_exp: usize) -> Option<std::collections::HashMap<(u32, u32), Vec<f32>>> {
    let path = ojas_core::config::var("OJAS_DOWN_IQ2").ok()?;
    let raw = std::fs::read(&path).ok()?;
    let rec = 8 + ffn_exp * 4;
    let mut map = std::collections::HashMap::new();
    for r in raw.chunks_exact(rec) {
        let l = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
        let e = u32::from_le_bytes([r[4], r[5], r[6], r[7]]);
        let v: Vec<f32> = r[8..].chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).max(1e-6)).collect();
        map.insert((l, e), v);
    }
    tracing::info!(target: "cpu:glm", "DOWN_IQ2 re-encode: imatrix for {} experts", map.len());
    Some(map)
}

/// Parse OJAS_CHANNEL_PRUNE=<energy.bin>:<pct> → per-(layer,expert) list of
/// the bottom-pct% channel indices by measured decoding-time energy.
fn load_prune_mask(ffn_exp: usize) -> Option<std::collections::HashMap<(u32, u32), Vec<u32>>> {
    let spec = ojas_core::config::var("OJAS_CHANNEL_PRUNE").ok()?;
    // <path>:<pct>  OR depth-aware  <path>:<early>:<mid>:<late>
    // (early = layer < 16, mid = 16..52, late = 52+; the census shows early
    // layers holding a ~35% cold tail and mid/late ~5-8%)
    let parts: Vec<&str> = spec.split(':').collect();
    let (path, pcts): (&str, Vec<f32>) = match parts.len() {
        2 => (parts[0], vec![parts[1].parse().ok()?]),
        4 => (parts[0], parts[1..].iter().map(|p| p.parse().unwrap_or(0.0)).collect()),
        _ => return None,
    };
    let raw = std::fs::read(path).ok()?;
    let rec = 8 + ffn_exp * 4;
    let mut map = std::collections::HashMap::new();
    for r in raw.chunks_exact(rec) {
        let l = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
        let e = u32::from_le_bytes([r[4], r[5], r[6], r[7]]);
        let pct = if pcts.len() == 1 { pcts[0] }
            else if l < 16 { pcts[0] } else if l < 52 { pcts[1] } else { pcts[2] };
        let n_drop = (ffn_exp as f32 * pct / 100.0) as usize;
        let mut idx: Vec<u32> = (0..ffn_exp as u32).collect();
        let en = |i: u32| f32::from_le_bytes([r[8 + i as usize * 4], r[9 + i as usize * 4], r[10 + i as usize * 4], r[11 + i as usize * 4]]);
        idx.sort_by(|&a, &b| en(a).total_cmp(&en(b)));
        idx.truncate(n_drop);
        map.insert((l, e), idx);
    }
    tracing::info!(target: "cpu:glm", "channel-prune: {pcts:?}% schedule over {} experts", map.len());
    Some(map)
}
