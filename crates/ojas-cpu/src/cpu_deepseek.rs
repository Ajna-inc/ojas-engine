//! Pure-CPU deepseek2 (DeepSeek-V2/Coder-V2-Lite) decoder — MLA + MoE on the
//! CPU tier.
//!
//! Faithful to the validated Metal host logic (the ojas-models MLA/MoE paths)
//! and the reference `build_deepseek2`:
//! - MLA (lite, naive): q = Wq·h → 16 heads × [nope 128 | rope 64];
//!   kv_a = Wkv_a·h → [kv_lora 512 | k_rope 64]; the latent is RMSNormed then
//!   expanded per head via Wkv_b → [k_nope 128 | v 128]; the 64-dim shared
//!   rope key is broadcast to every head. NEOX rope on the rope dims only.
//!   Attention scale = mscale²/√k_mla (YaRN mscale = yarn_log_multiplier/0.1,
//!   0.707 for V2-Lite → matches the reference "mscale == 0.7").
//! - MoE (V2 semantics): softmax over all experts → top-k raw probs, no
//!   renormalization (norm_topk_prob=false), ties → lowest index (matches the
//!   GPU moe_topk_nonorm kernel), × routed_scaling_factor; always-on fused
//!   shared expert added with weight 1. First `leading_dense` layers are
//!   plain SwiGLU.
//! - Weights are per-row int8 (SDOT) via cpu_math, same as the dense CPU tier;
//!   per-expert tensors quantize per expert slice.

use crate::cpu_math::{argmax, matvec as mv, quant_row_i8, rmsnorm, rope, silu, W};
use ojas_core::cancel::STREAM_CANCEL;
use ojas_core::Model as DecoderModel;
use ojas_formats::gguf::Gguf;
use anyhow::{bail, Result};
use half::f16;
use half::slice::HalfFloatSliceExt;
use std::cell::RefCell;
use std::sync::atomic::Ordering;

struct Kv {
    k: Vec<Vec<f32>>, // [layer] -> flat [pos * n_head * k_mla]
    v: Vec<Vec<f32>>, // [layer] -> flat [pos * n_head * v_mla]
}

struct MoeLayer {
    gate_inp: W,          // router [d → n_expert] (stored f32 in the GGUF)
    gate: Vec<W>,         // per expert [d → ffn_exp]
    up: Vec<W>,
    down: Vec<W>,         // per expert [ffn_exp → d]
    gate_sh: W,           // fused shared experts [d → n_shared*ffn_exp]
    up_sh: W,
    down_sh: W,
}

enum Ffn {
    Dense { gate: W, up: W, down: W },
    Moe(MoeLayer),
}

struct Layer {
    attn_norm: Vec<f32>,
    wq: W,
    wkv_a: W,
    kv_a_norm: Vec<f32>,
    wkv_b: W,
    wo: W,
    ffn_norm: Vec<f32>,
    ffn: Ffn,
}

pub struct CpuDeepseek {
    d: usize,
    n_head: usize,
    kv_lora: usize,
    qk_rope: usize,
    k_mla: usize, // per-head key len (nope + rope)
    v_mla: usize,
    nope: usize,
    n_expert: usize,
    n_used: usize,
    ffn_exp: usize,
    routed_scale: f32,
    vocab: usize,
    rope_base: f32,
    eps: f32,
    attn_scale: f32, // mscale²/√k_mla
    layers: Vec<Layer>,
    output_norm: Vec<f32>,
    head: W,
    embd: W,
    threads: usize,
    dotprod: bool,
    kv: RefCell<Kv>,
}

impl CpuDeepseek {
    pub fn load(g: &mut Gguf) -> Result<CpuDeepseek> {
        let arch = g.arch();
        if arch != "deepseek2" {
            bail!("CpuDeepseek supports deepseek2 (got {arch})");
        }
        let mu = |g: &Gguf, k: &str| g.meta_u32(&format!("deepseek2.{k}")).unwrap_or(0) as usize;
        let d = mu(g, "embedding_length");
        let n_layers = mu(g, "block_count");
        let n_head = mu(g, "attention.head_count");
        let kv_lora = mu(g, "attention.kv_lora_rank");
        let qk_rope = g.meta_u32("deepseek2.rope.dimension_count").unwrap_or(64) as usize;
        let k_mla = g.meta_u32("deepseek2.attention.key_length_mla")
            .or_else(|| g.meta_u32("deepseek2.attention.key_length")).unwrap_or(192) as usize;
        let v_mla = g.meta_u32("deepseek2.attention.value_length_mla")
            .or_else(|| g.meta_u32("deepseek2.attention.value_length")).unwrap_or(128) as usize;
        let nope = k_mla - qk_rope;
        let n_expert = mu(g, "expert_count");
        let n_used = mu(g, "expert_used_count");
        let ffn_exp = mu(g, "expert_feed_forward_length");
        let n_shared = mu(g, "expert_shared_count").max(1);
        let leading_dense = mu(g, "leading_dense_block_count");
        let _ffn_dense = mu(g, "feed_forward_length");
        let routed_scale = g.meta_f32("deepseek2.expert_weights_scale").unwrap_or(1.0);
        let rope_base = g.meta_f32("deepseek2.rope.freq_base").unwrap_or(1e4);
        let eps = g.meta_f32("deepseek2.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let vocab = g.str_arr("tokenizer.ggml.tokens").map(|t| t.len()).unwrap_or(1).max(1);
        // YaRN attention mscale (see ojas-models DecoderGpu::ds_mscale):
        // GGUF stores yarn_log_multiplier = 0.1·mscale_all_dim.
        let factor = g.meta_f32("deepseek2.rope.scaling.factor").unwrap_or(1.0);
        let log_mul = g.meta_f32("deepseek2.rope.scaling.yarn_log_multiplier").unwrap_or(0.0);
        let mscale = if factor <= 1.0 || log_mul <= 0.0 { 1.0 } else { log_mul / 0.1 };
        let attn_scale = mscale * mscale / (k_mla as f32).sqrt();
        let q_lora = mu(g, "attention.q_lora_rank");
        if q_lora != 0 {
            bail!("CpuDeepseek supports the lite arch (q_lora=0) for now; got q_lora={q_lora}");
        }

        #[cfg(target_arch = "aarch64")]
        let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        #[cfg(not(target_arch = "aarch64"))]
        let dotprod = false;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        tracing::info!(target: "cpu:deepseek2", "d={d} L={n_layers} heads={n_head} kv_lora={kv_lora} k={k_mla}(nope {nope}+rope {qk_rope}) v={v_mla} \
                   | moe {n_expert}e top{n_used} +{n_shared}sh ffn_exp={ffn_exp} dense0..{leading_dense} | mscale={mscale:.3} q8 sdot={dotprod}");

        // per-row q8 quantization of an f16-dequanted tensor (rows from dims[0]=cols)
        let quant = |bytes: &[u8], cols: usize| -> W {
            let v: Vec<f16> = bytes.chunks_exact(2)
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
        let readw = |g: &mut Gguf, name: &str| -> Result<W> {
            let (dims, ty, bytes) = g.read_tensor(name)?;
            let cols = dims.first().copied().unwrap_or(1) as usize;
            match ty {
                1 => Ok(quant(&bytes, cols)),
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
        // split a 3D per-expert tensor [cols, rows, n_expert] into n_expert q8 Ws
        let read_exps = |g: &mut Gguf, name: &str, n_expert: usize| -> Result<Vec<W>> {
            let (dims, ty, bytes) = g.read_tensor(name)?;
            if ty != 1 { bail!("expected f16-dequanted experts for {name} (got type {ty})"); }
            let cols = dims.first().copied().unwrap_or(1) as usize;
            let per = bytes.len() / n_expert;
            Ok((0..n_expert).map(|e| quant(&bytes[e * per..(e + 1) * per], cols)).collect())
        };

        let t0 = std::time::Instant::now();
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
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
                    gate: read_exps(g, &p("ffn_gate_exps.weight"), n_expert)?,
                    up: read_exps(g, &p("ffn_up_exps.weight"), n_expert)?,
                    down: read_exps(g, &p("ffn_down_exps.weight"), n_expert)?,
                    gate_sh: readw(g, &p("ffn_gate_shexp.weight"))?,
                    up_sh: readw(g, &p("ffn_up_shexp.weight"))?,
                    down_sh: readw(g, &p("ffn_down_shexp.weight"))?,
                })
            };
            layers.push(Layer {
                attn_norm: readf32(g, &p("attn_norm.weight"))?,
                wq: readw(g, &p("attn_q.weight"))?,
                wkv_a: readw(g, &p("attn_kv_a_mqa.weight"))?,
                kv_a_norm: readf32(g, &p("attn_kv_a_norm.weight"))?,
                wkv_b: readw(g, &p("attn_kv_b.weight"))?,
                wo: readw(g, &p("attn_output.weight"))?,
                ffn_norm: readf32(g, &p("ffn_norm.weight"))?,
                ffn,
            });
            if i % 4 == 3 { tracing::debug!(target: "cpu:deepseek2", "loaded {}/{} layers ({:.0}s)", i + 1, n_layers, t0.elapsed().as_secs_f32()); }
        }
        let output_norm = readf32(g, "output_norm.weight")?;
        let head = readw(g, "output.weight")?;
        let embd = readw(g, "token_embd.weight")?;
        tracing::info!(target: "cpu:deepseek2", "loaded {n_layers} layers in {:.0}s", t0.elapsed().as_secs_f32());

        Ok(CpuDeepseek {
            d, n_head, kv_lora, qk_rope, k_mla, v_mla, nope, n_expert, n_used, ffn_exp,
            routed_scale, vocab, rope_base, eps, attn_scale,
            layers, output_norm, head, embd, threads, dotprod,
            kv: RefCell::new(Kv { k: vec![Vec::new(); n_layers], v: vec![Vec::new(); n_layers] }),
        })
    }

    fn forward(&self, token: usize, pos: usize, want_logits: bool) -> Option<Vec<f32>> {
        let (d, nh) = (self.d, self.n_head);
        let (nope, rope_d, k_mla, v_mla) = (self.nope, self.qk_rope, self.k_mla, self.v_mla);
        let mut x: Vec<f32> = match &self.embd {
            W::Q8 { q, scale } => q[token * d..(token + 1) * d].iter().map(|&b| b as f32 * scale[token]).collect(),
            W::F32(v) => v[token * d..(token + 1) * d].to_vec(),
            W::F16(v) => (0..d).map(|i| v[token * d + i].to_f32()).collect(),
            W::Q20 { raw } => {
                let rb = d / 128 * 34;
                crate::cpu_math::dequant_q2_0_row(&raw[token * rb..(token + 1) * rb], d)
            }
        };
        let kvc = &mut *self.kv.borrow_mut();

        for (l, ly) in self.layers.iter().enumerate() {
            // ---- MLA attention (naive: full per-head K/V cache) ----
            let h = rmsnorm(&x, &ly.attn_norm, self.eps);
            let mut q = mv(&ly.wq, nh * k_mla, d, &h, None, self.threads, self.dotprod);
            let kv_a = mv(&ly.wkv_a, self.kv_lora + rope_d, d, &h, None, self.threads, self.dotprod);
            let ckv = rmsnorm(&kv_a[..self.kv_lora], &ly.kv_a_norm, self.eps);
            let mut kr = kv_a[self.kv_lora..].to_vec();
            rope(&mut kr, rope_d, pos, self.rope_base);
            let kvb = mv(&ly.wkv_b, nh * (nope + v_mla), self.kv_lora, &ckv, None, self.threads, self.dotprod);
            // per-head rope on q's rope dims, then append K/V rows
            for hh in 0..nh {
                rope(&mut q[hh * k_mla + nope..(hh + 1) * k_mla], rope_d, pos, self.rope_base);
            }
            for hh in 0..nh {
                let kn = &kvb[hh * (nope + v_mla)..hh * (nope + v_mla) + nope];
                kvc.k[l].extend_from_slice(kn);
                kvc.k[l].extend_from_slice(&kr); // shared rope key, broadcast per head
                kvc.v[l].extend_from_slice(&kvb[hh * (nope + v_mla) + nope..(hh + 1) * (nope + v_mla)]);
            }
            let seq = kvc.k[l].len() / (nh * k_mla);
            let mut ao = vec![0f32; nh * v_mla];
            for hh in 0..nh {
                let qh = &q[hh * k_mla..(hh + 1) * k_mla];
                let mut scores = vec![0f32; seq];
                for (t, s) in scores.iter_mut().enumerate() {
                    let kt = &kvc.k[l][(t * nh + hh) * k_mla..(t * nh + hh + 1) * k_mla];
                    *s = crate::cpu_math::dot_f32(qh, kt) * self.attn_scale;
                }
                let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                let mut denom = 0.0;
                for s in scores.iter_mut() { *s = (*s - mx).exp(); denom += *s; }
                for i in 0..v_mla {
                    let mut acc = 0.0;
                    for (t, s) in scores.iter().enumerate() {
                        acc += s * kvc.v[l][(t * nh + hh) * v_mla + i];
                    }
                    ao[hh * v_mla + i] = acc / denom;
                }
            }
            let o = mv(&ly.wo, d, nh * v_mla, &ao, None, self.threads, self.dotprod);
            for i in 0..d { x[i] += o[i]; }

            // ---- FFN ----
            let h2 = rmsnorm(&x, &ly.ffn_norm, self.eps);
            match &ly.ffn {
                Ffn::Dense { gate, up, down } => {
                    let n = match gate { W::Q8 { scale, .. } => scale.len(), W::F32(v) => v.len() / d, W::F16(v) => v.len() / d,
                        W::Q20 { raw } => raw.len() / (d / 128 * 34) };
                    let gv = mv(gate, n, d, &h2, None, self.threads, self.dotprod);
                    let uv = mv(up, n, d, &h2, None, self.threads, self.dotprod);
                    let act: Vec<f32> = (0..n).map(|i| silu(gv[i]) * uv[i]).collect();
                    let dv = mv(down, d, n, &act, None, self.threads, self.dotprod);
                    for i in 0..d { x[i] += dv[i]; }
                }
                Ffn::Moe(m) => {
                    // router: softmax over all experts → top-k raw probs (no renorm),
                    // ties → lowest index (matches the GPU moe_topk_nonorm kernel)
                    let logits = mv(&m.gate_inp, self.n_expert, d, &h2, None, self.threads, self.dotprod);
                    let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
                    let exps: Vec<f32> = logits.iter().map(|&v| (v - mx).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    let mut probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
                    let mut picks: Vec<(usize, f32)> = Vec::with_capacity(self.n_used);
                    for _ in 0..self.n_used {
                        let (mut bi, mut bv) = (0usize, -1.0f32);
                        for (e, &pv) in probs.iter().enumerate() {
                            if pv > bv { bv = pv; bi = e; }
                        }
                        picks.push((bi, bv));
                        probs[bi] = -1.0;
                    }
                    let mut moe = vec![0f32; d];
                    for &(e, wgt) in &picks {
                        let gv = mv(&m.gate[e], self.ffn_exp, d, &h2, None, self.threads, self.dotprod);
                        let uv = mv(&m.up[e], self.ffn_exp, d, &h2, None, self.threads, self.dotprod);
                        let act: Vec<f32> = (0..self.ffn_exp).map(|i| silu(gv[i]) * uv[i]).collect();
                        let dv = mv(&m.down[e], d, self.ffn_exp, &act, None, self.threads, self.dotprod);
                        let wsc = wgt * self.routed_scale;
                        for i in 0..d { moe[i] += wsc * dv[i]; }
                    }
                    // fused shared experts, weight 1
                    let ns = match &m.gate_sh { W::Q8 { scale, .. } => scale.len(), W::F32(v) => v.len() / d, W::F16(v) => v.len() / d,
                        W::Q20 { raw } => raw.len() / (d / 128 * 34) };
                    let gv = mv(&m.gate_sh, ns, d, &h2, None, self.threads, self.dotprod);
                    let uv = mv(&m.up_sh, ns, d, &h2, None, self.threads, self.dotprod);
                    let act: Vec<f32> = (0..ns).map(|i| silu(gv[i]) * uv[i]).collect();
                    let sv = mv(&m.down_sh, d, ns, &act, None, self.threads, self.dotprod);
                    for i in 0..d { x[i] += moe[i] + sv[i]; }
                }
            }
        }
        if !want_logits { return None; }
        let xn = rmsnorm(&x, &self.output_norm, self.eps);
        Some(mv(&self.head, self.vocab, d, &xn, None, self.threads, self.dotprod))
    }
}

impl DecoderModel for CpuDeepseek {
    fn n_layers(&self) -> usize { self.layers.len() }
    fn hidden_dim(&self) -> usize { self.d }
    fn prefill(&self, tokens: &[u32], base_pos: usize) {
        for (i, &t) in tokens.iter().enumerate() {
            if STREAM_CANCEL.load(Ordering::Relaxed) { return; }
            self.forward(t as usize, base_pos + i, false);
        }
    }
    fn forward_id(&self, token: u32, pos: usize) -> u32 {
        if STREAM_CANCEL.load(Ordering::Relaxed) { return u32::MAX; }
        argmax(&self.forward(token as usize, pos, true).unwrap()) as u32
    }
    fn forward_logits(&self, token: u32, pos: usize) -> Option<Vec<f32>> {
        if STREAM_CANCEL.load(Ordering::Relaxed) { return None; }
        self.forward(token as usize, pos, true)
    }
}
