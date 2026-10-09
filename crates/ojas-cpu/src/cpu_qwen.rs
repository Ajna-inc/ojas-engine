//! Pure-CPU qwen-family decoder — the CPU tier, no GPU required. Plugged into
//! the shared engine as a [`DecoderModel`], so `Engine::load_with(mode:"cpu")`
//! (or the automatic no-Metal fallback) chats through the same
//! tokenizer/template/cancel/session frontend as the GPU paths.
//!
//! Supports any GGUF the reader can decode (F32/F16/BF16/Q4_0/Q8_0/Q4_K/
//! Q5_K/Q6_K — quantized tensors are dequanted to f16 at load), dense
//! qwen2/qwen3/llama archs (RMSNorm, optional QKV bias, optional per-head
//! QK-norm, NeoX RoPE, GQA, SwiGLU, tied/untied head).
//!
//! Performance shape:
//! - hot dot: per-row f16→f32 hardware slice conversion (half's slice API)
//!   + unrolled f32 FMA that LLVM autovectorizes (NEON fmla).
//! - decode matvecs: deterministic row-parallel across threads.
//! - prefill: chunked — each weight row is converted/read once per chunk of
//!   M prompt tokens instead of once per token (the same weights-read-once
//!   approach as the GPU batched prefill; otherwise prefill is M×
//!   weight-bandwidth bound).

use crate::cpu_math::{argmax, matmul as mm, matvec as mv, dot_f32, rmsnorm, rmsnorm_inplace, rope, silu, W, quant_row_i8};
use ojas_core::cancel::STREAM_CANCEL;
use ojas_core::Model as DecoderModel;
use ojas_formats::gguf::Gguf;
use anyhow::{bail, Result};
use half::f16;
use half::slice::HalfFloatSliceExt;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

/// Prompt tokens processed per prefill chunk (bounds scratch memory: a chunk
/// holds M×ffn f32 activations — 64×11008×4 ≈ 2.8 MB on a 2B model).
const CHUNK: usize = 64;

struct Kv {
    k: Vec<Vec<f32>>, // [layer] -> flat [pos * (n_kv*head_dim)]
    v: Vec<Vec<f32>>,
}

pub struct CpuQwen {
    d: usize,
    n_layers: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    ffn: usize,
    vocab: usize,
    rope_base: f32,
    eps: f32,
    qk_norm: bool,
    w: HashMap<String, W>,           // big weights (f32 fast path or f16)
    f32w: HashMap<String, Vec<f32>>, // norms/biases (f32)
    threads: usize,
    dotprod: bool,
    kv: RefCell<Kv>,
}

impl CpuQwen {
    pub fn load(g: &mut Gguf) -> Result<CpuQwen> {
        let arch = g.arch();
        if !matches!(arch.as_str(), "qwen2" | "qwen3" | "llama") {
            bail!("CPU mode supports dense qwen2/qwen3/llama GGUFs for now (got {arch})");
        }
        let mu = |k: &str| g.meta_u32(&format!("{arch}.{k}")).unwrap_or(0) as usize;
        let d = mu("embedding_length");
        let n_layers = mu("block_count");
        let n_head = mu("attention.head_count");
        let n_kv = mu("attention.head_count_kv");
        let ffn = mu("feed_forward_length");
        // Qwen3 sets attention.key_length independent of d/n_head (128 vs 64); honor it
        // like the GPU path, else fall back to d/n_head (qwen2/llama).
        let key_len = mu("attention.key_length");
        let head_dim = if key_len > 0 { key_len } else { d / n_head.max(1) };
        let rope_base = g.meta_f32(&format!("{arch}.rope.freq_base")).unwrap_or(1e6);
        let eps = g.meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
        let vocab = g.str_arr("tokenizer.ggml.tokens").map(|t| t.len()).unwrap_or(1).max(1);

        // Precision ladder (OJAS_CPU_PREC=q8|f32|f16): default q8 where the CPU
        // has int8 dot-product instructions (fewest bytes/token = fastest on a
        // byte-bound decode; per-row scales, same requant as the GPU q8 path);
        // f32 = exact tier when it fits OJAS_CPU_F32_GB; f16 = half-RAM fallback.
        let dotprod = crate::cpu_math::fast_i8();
        let f32_budget = ojas_core::config::var("OJAS_CPU_F32_GB").ok()
            .and_then(|v| v.parse::<f64>().ok()).unwrap_or(16.0) * 1e9;
        let approx_params = (n_layers * (2 * d * d + 2 * d * (n_kv * head_dim) + 3 * d * ffn) + vocab * d) as f64;
        let prec = ojas_core::config::var("OJAS_CPU_PREC").unwrap_or_else(|_|
            if dotprod { "q8".into() } else if approx_params * 4.0 <= f32_budget { "f32".into() } else { "f16".into() });
        let store_f32 = prec == "f32" && approx_params * 4.0 <= f32_budget;
        // OJAS_CPU_NO_Q20=1 forces ternary weights through the dequant+requant
        // ladder instead (for A/B-ing the native path).
        let native_q2 = ojas_core::config::var("OJAS_CPU_NO_Q20").is_err();
        let mut w: HashMap<String, W> = HashMap::new();
        let mut f32w = HashMap::new();
        // biases (qwen2) and per-head qk-norms (qwen3) are optional per arch
        let mut want: Vec<(String, bool)> =
            vec![("token_embd.weight".into(), false), ("output_norm.weight".into(), false)];
        for i in 0..n_layers {
            for (s, opt) in [
                ("attn_norm.weight", false), ("attn_q.weight", false), ("attn_q.bias", true),
                ("attn_k.weight", false), ("attn_k.bias", true), ("attn_v.weight", false),
                ("attn_v.bias", true), ("attn_output.weight", false),
                ("attn_q_norm.weight", true), ("attn_k_norm.weight", true),
                ("ffn_norm.weight", false), ("ffn_gate.weight", false),
                ("ffn_up.weight", false), ("ffn_down.weight", false),
            ] {
                want.push((format!("blk.{i}.{s}"), opt));
            }
        }
        want.push(("output.weight".into(), true)); // untied head if present
        let n_want = want.len();
        for (name, optional) in &want {
            // Ternary Q2_0 (type 42) stays in its raw GGUF form: 2.125 bits/weight
            // resident instead of the 8 the q8 ladder would expand it to. Decode is
            // byte-bound, so this is ~3.8x fewer bytes/token and ~4x less RAM, which
            // is what lets an 8B model fit on a phone.
            if native_q2 && g.tensor_meta(name).is_some_and(|m| m.3 == 42) {
                match g.read_tensor_raw(name) {
                    Ok((_dims, _ty, raw)) => { w.insert(name.clone(), W::Q20 { raw }); continue; }
                    Err(e) => { if *optional { continue; } bail!("{name}: {e}"); }
                }
            }
            let Ok((dims, ty, bytes)) = g.read_tensor(name) else {
                if *optional { continue; }
                bail!("missing tensor {name}");
            };
            match ty {
                1 => {
                    let v: Vec<f16> = bytes.chunks_exact(2)
                        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
                    // norm-sized tensors stay f32-exact even under q8 (they're tiny)
                    let is_norm = name.ends_with("_norm.weight");
                    if prec == "q8" && !is_norm {
                        let mut f = vec![0f32; v.len()];
                        v.convert_to_f32_slice(&mut f);
                        // dims: [cols(K), rows(N)] for a matmul weight
                        let cols = dims.first().copied().unwrap_or(f.len() as u64) as usize;
                        let rows = f.len() / cols.max(1);
                        let mut q = Vec::with_capacity(f.len());
                        let mut scale = Vec::with_capacity(rows);
                        for r in 0..rows {
                            let (qr, sc) = quant_row_i8(&f[r * cols..(r + 1) * cols]);
                            q.extend_from_slice(&qr);
                            scale.push(sc);
                        }
                        w.insert(name.clone(), W::Q8 { q, scale });
                    } else if store_f32 && !is_norm {
                        let mut f = vec![0f32; v.len()];
                        v.convert_to_f32_slice(&mut f);
                        w.insert(name.clone(), W::F32(f));
                    } else {
                        w.insert(name.clone(), W::F16(v));
                    }
                }
                0 => {
                    let v: Vec<f32> = bytes.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    f32w.insert(name.clone(), v);
                }
                t => bail!("unsupported GGUF type {t} for {name}"),
            }
        }
        let qk_norm = f32w.contains_key("blk.0.attn_q_norm.weight") || w.contains_key("blk.0.attn_q_norm.weight");
        // OJAS_CPU_THREADS pins the worker count. Defaults to all cores, but on
        // big.LITTLE phones the little cores gate every row-block join, so being
        // able to dial this to the big-core count matters.
        let threads = ojas_core::config::var("OJAS_CPU_THREADS").ok().and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .or_else(crate::cpu_math::perf_cores)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
        tracing::info!(target: "cpu", "loaded {n_want} tensors ({d} dim, {n_layers} layers, {n_head}/{n_kv} heads, ffn {ffn}, vocab {vocab}, qk_norm={qk_norm}, prec={prec}, dotprod={dotprod}, {threads} threads)");
        Ok(CpuQwen {
            d, n_layers, n_head, n_kv, head_dim, ffn, vocab, rope_base, eps, qk_norm, w, f32w, threads, dotprod,
            kv: RefCell::new(Kv { k: vec![Vec::new(); n_layers], v: vec![Vec::new(); n_layers] }),
        })
    }

    fn norm_w(&self, name: &str) -> Vec<f32> {
        // per-head norm weights may be stored f16 or f32 depending on the GGUF
        if let Some(v) = self.f32w.get(name) { return v.clone(); }
        match &self.w[name] {
            W::F32(v) => v.clone(),
            W::F16(v) => v.iter().map(|h| h.to_f32()).collect(),
            // norms are F32 in every ternary GGUF observed; safety net only
            W::Q20 { raw } => crate::cpu_math::dequant_q2_0_row(raw, raw.len() / 34 * 128),
            W::Q8 { q, scale } => {
                // norms are stored exact; this path is only a safety net
                let cols = q.len() / scale.len();
                q.iter().enumerate().map(|(i, &b)| b as f32 * scale[i / cols]).collect()
            }
        }
    }

    /// Run `m` tokens through all layers, filling KV. `logits_last` returns
    /// the final token's logits (decode); prefill passes false.
    fn forward_batch(&self, tokens: &[u32], base_pos: usize, logits_last: bool) -> Option<Vec<f32>> {
        let (d, hd) = (self.d, self.head_dim);
        let kvdim = self.n_kv * hd;
        let qdim = self.n_head * hd;   // == d on qwen2/llama, but n_head*key_length on qwen3
        let m = tokens.len();
        let mut xs: Vec<Vec<f32>> = tokens.iter()
            .map(|&t| { let t = t as usize; match &self.w["token_embd.weight"] {
                W::F32(v) => v[t * d..(t + 1) * d].to_vec(),
                W::F16(v) => (0..d).map(|i| v[t * d + i].to_f32()).collect(),
                W::Q8 { q, scale } => q[t * d..(t + 1) * d].iter().map(|&b| b as f32 * scale[t]).collect(),
                W::Q20 { raw } => {
                    let rb = d / 128 * 34;
                    crate::cpu_math::dequant_q2_0_row(&raw[t * rb..(t + 1) * rb], d)
                }
            } })
            .collect();
        let kv = &mut *self.kv.borrow_mut();
        // Attention reads rows 0..base_pos+t by position, so the cache must hold exactly
        // base_pos rows before this batch appends. Rows past base_pos belong to whatever
        // sequence ran before (another request with a different prefix) and would be
        // attended to as if they were this one's. Same rule as `CpuSsm`.
        for l in 0..self.n_layers {
            assert!(kv.k[l].len() >= base_pos * kvdim, "KV holds {} rows, forward at {base_pos}", kv.k[l].len() / kvdim);
            kv.k[l].truncate(base_pos * kvdim);
            kv.v[l].truncate(base_pos * kvdim);
        }
        let mut q = vec![vec![0f32; qdim]; m];
        let mut k = vec![vec![0f32; kvdim]; m];
        let mut v = vec![vec![0f32; kvdim]; m];
        let mut attn = vec![vec![0f32; qdim]; m];
        let mut o = vec![vec![0f32; d]; m];
        let mut gate = vec![vec![0f32; self.ffn]; m];
        let mut up = vec![vec![0f32; self.ffn]; m];
        let mut down = vec![vec![0f32; d]; m];

        for l in 0..self.n_layers {
            let p = |s: &str| format!("blk.{l}.{s}");
            // ---- attention ----
            let hs: Vec<Vec<f32>> = xs.iter().map(|x| rmsnorm(x, &self.f32w[&p("attn_norm.weight")], self.eps)).collect();
            let hrefs: Vec<&[f32]> = hs.iter().map(|h| h.as_slice()).collect();
            mm(&self.w[&p("attn_q.weight")], qdim, d, &hrefs, self.f32w.get(&p("attn_q.bias")).map(|b| b.as_slice()), &mut q, self.threads, self.dotprod);
            mm(&self.w[&p("attn_k.weight")], kvdim, d, &hrefs, self.f32w.get(&p("attn_k.bias")).map(|b| b.as_slice()), &mut k, self.threads, self.dotprod);
            mm(&self.w[&p("attn_v.weight")], kvdim, d, &hrefs, self.f32w.get(&p("attn_v.bias")).map(|b| b.as_slice()), &mut v, self.threads, self.dotprod);
            let (qn, kn) = if self.qk_norm {
                (Some(self.norm_w(&p("attn_q_norm.weight"))), Some(self.norm_w(&p("attn_k_norm.weight"))))
            } else { (None, None) };
            for (t, tok_pos) in (base_pos..base_pos + m).enumerate() {
                for hh in 0..self.n_head {
                    let qv = &mut q[t][hh * hd..(hh + 1) * hd];
                    if let Some(w) = &qn { rmsnorm_inplace(qv, w, self.eps); }
                    rope(qv, hd, tok_pos, self.rope_base);
                }
                for hh in 0..self.n_kv {
                    let kvv = &mut k[t][hh * hd..(hh + 1) * hd];
                    if let Some(w) = &kn { rmsnorm_inplace(kvv, w, self.eps); }
                    rope(kvv, hd, tok_pos, self.rope_base);
                }
                kv.k[l].extend_from_slice(&k[t]);
                kv.v[l].extend_from_slice(&v[t]);
            }
            // causal attention: token t sees positions 0..=base_pos+t
            let t_attn = std::time::Instant::now();
            let scale = 1.0 / (hd as f32).sqrt();
            let group = self.n_head / self.n_kv;
            // One job per (token, head): each writes a disjoint hd-slice of attn,
            // so this parallelizes with no coordination beyond the row counter.
            // Its cost grows with context while the matmuls' does not, so a serial
            // version degrades on long prompts.
            let n_head = self.n_head;
            let jobs = m * n_head;
            let attn_addr = attn.as_mut_ptr() as usize;
            let (kk, vv) = (&kv.k[l], &kv.v[l]);
            let qref = &q;
            let next = std::sync::atomic::AtomicUsize::new(0);
            let body = |_id: usize, _nt: usize| {
                let mut scores = Vec::new();
                loop {
                    let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if j >= jobs { break; }
                    let (t, hh) = (j / n_head, j % n_head);
                    let seq = base_pos + t + 1;
                    let kvh = hh / group;
                    let qh = &qref[t][hh * hd..(hh + 1) * hd];
                    scores.clear();
                    scores.resize(seq, 0.0f32);
                    for (tt, s) in scores.iter_mut().enumerate() {
                        let kt = &kk[tt * kvdim + kvh * hd..tt * kvdim + kvh * hd + hd];
                        *s = dot_f32(qh, kt) * scale;
                    }
                    let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let mut denom = 0.0;
                    for s in scores.iter_mut() { *s = (*s - mx).exp(); denom += *s; }
                    for i in 0..hd {
                        let mut acc = 0.0;
                        for (tt, s) in scores.iter().enumerate() {
                            acc += s * vv[tt * kvdim + kvh * hd + i];
                        }
                        // SAFETY: (t,hh) is claimed by exactly one worker, and
                        // heads write disjoint hd-wide slices of row t.
                        unsafe {
                            let av = &mut *(attn_addr as *mut Vec<f32>).add(t);
                            av[hh * hd + i] = acc / denom;
                        }
                    }
                }
            };
            crate::cpu_math::parallel(self.threads, &body);
            if crate::cpu_math::PROF.load(std::sync::atomic::Ordering::Relaxed) {
                crate::cpu_math::T_ATTN.fetch_add(t_attn.elapsed().as_nanos() as u64,
                                                  std::sync::atomic::Ordering::Relaxed);
            }
            let arefs: Vec<&[f32]> = attn.iter().map(|a| a.as_slice()).collect();
            mm(&self.w[&p("attn_output.weight")], d, qdim, &arefs, None, &mut o, self.threads, self.dotprod);
            for t in 0..m { for i in 0..d { xs[t][i] += o[t][i]; } }
            // ---- FFN (SwiGLU) ----
            let h2s: Vec<Vec<f32>> = xs.iter().map(|x| rmsnorm(x, &self.f32w[&p("ffn_norm.weight")], self.eps)).collect();
            let h2refs: Vec<&[f32]> = h2s.iter().map(|h| h.as_slice()).collect();
            mm(&self.w[&p("ffn_gate.weight")], self.ffn, d, &h2refs, None, &mut gate, self.threads, self.dotprod);
            mm(&self.w[&p("ffn_up.weight")], self.ffn, d, &h2refs, None, &mut up, self.threads, self.dotprod);
            let acts: Vec<Vec<f32>> = (0..m).map(|t| (0..self.ffn).map(|i| silu(gate[t][i]) * up[t][i]).collect()).collect();
            let actrefs: Vec<&[f32]> = acts.iter().map(|a| a.as_slice()).collect();
            mm(&self.w[&p("ffn_down.weight")], d, self.ffn, &actrefs, None, &mut down, self.threads, self.dotprod);
            for t in 0..m { for i in 0..d { xs[t][i] += down[t][i]; } }
        }
        if !logits_last { return None; }
        let xn = rmsnorm(&xs[m - 1], &self.f32w["output_norm.weight"], self.eps);
        let head = self.w.get("output.weight").unwrap_or(&self.w["token_embd.weight"]);
        Some(mv(head, self.vocab, self.d, &xn, None, self.threads, self.dotprod))
    }
}

impl DecoderModel for CpuQwen {
    fn n_layers(&self) -> usize { self.n_layers }
    fn hidden_dim(&self) -> usize { self.d }
    fn prefill(&self, tokens: &[u32], base_pos: usize) {
        let mut pos = base_pos;
        for chunk in tokens.chunks(CHUNK) {
            // CPU prefill takes real time — honor Stop between chunks
            if STREAM_CANCEL.load(Ordering::Relaxed) { return; }
            self.forward_batch(chunk, pos, false);
            pos += chunk.len();
        }
    }
    fn forward_id(&self, token: u32, pos: usize) -> u32 {
        if STREAM_CANCEL.load(Ordering::Relaxed) { return u32::MAX; }
        argmax(&self.forward_batch(&[token], pos, true).unwrap()) as u32
    }
    fn forward_logits(&self, token: u32, pos: usize) -> Option<Vec<f32>> {
        if STREAM_CANCEL.load(Ordering::Relaxed) { return None; }
        self.forward_batch(&[token], pos, true)
    }
    fn reset_session(&self) {
        let kv = &mut *self.kv.borrow_mut();
        kv.k.iter_mut().chain(kv.v.iter_mut()).for_each(Vec::clear);
    }
}
