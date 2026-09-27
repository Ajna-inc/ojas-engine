//! Full-model trainer on Metal. Full fine-tuning of all decoder layers + final norm (embeddings and
//! lm_head frozen). f32 weights + AdamW with blockwise-8-bit moments for big tensors
//! (optimizer-in-backward: each dW is consumed and its buffer reused immediately). One command
//! buffer per layer keeps GPU work sub-second (macOS watchdog). Limits: seq len bounded by
//! per-token GDN state history (~96–256), one sequence per step.

use crate::kernels::{g1, g2, Flce, Kit};
use crate::moe_ffn::{encode_dense_ffn, MoeFfn};
use ojas_metal::{MBuf, MetalGpu};
use ojas_core::Device as _;
use ojas_formats::safetensors::SafeTensors;
use anyhow::{anyhow, bail, Result};
use metal::MTLSize;
use objc::sel;
use objc::sel_impl;
use std::collections::HashMap;

// ---- Qwen3-0.6B (plain attention, all layers) ----
const D: usize = 1024;
// GDN consts retained only to size unused shared scratch (no GDN layers are constructed).
const HK: usize = 16;
const HV: usize = 32;
const S: usize = 128;
const DI: usize = 4096;
const C: usize = 2 * HK * S + DI;
const FFN: usize = 3072;
const NH: usize = 16;
const NKV: usize = 8;
const HD: usize = 128;
const ROT: usize = 128;      // full rope (partial_rotary_factor=None) → ROT = HD; base 1e6 in t_rope
const QDIM: usize = NH * HD;
const KVDIM: usize = NKV * HD;
const NL: usize = 28;
const V: usize = 151936;
fn tg256() -> MTLSize { MTLSize::new(256, 1, 1) }
fn tg128() -> MTLSize { MTLSize::new(128, 1, 1) }

enum Opt {
    F32 { m: MBuf, v: MBuf },
    C8 { mh: MBuf, vq: MBuf, vs: MBuf }, // m f16, v log-u8 blockwise
}

struct FfnA { x1: MBuf, h2: MBuf, rp: MBuf, gl: MBuf, ul: MBuf, act: MBuf }
struct GdnA {
    h: MBuf, rln: MBuf, qkv: MBuf, z: MBuf, ain: MBuf, braw: MBuf,
    bet: MBuf, sp: MBuf, gex: MBuf, acc: MBuf, conv: MBuf,
    o: MBuf, st: MBuf, sk: MBuf, dlt: MBuf, ron: MBuf, og: MBuf, ffn: FfnA,
}
struct AttnA {
    h: MBuf, rln: MBuf, qfull: MBuf, kfull: MBuf, vfull: MBuf,
    q0: MBuf, q2: MBuf, rqn: MBuf, rkn: MBuf, k2: MBuf, p: MBuf,
    aog: MBuf, ao: MBuf, ffn: FfnA,
}
enum LayerA { G(GdnA), A(AttnA) }

pub struct Trainer<'a> {
    gpu: &'a MetalGpu,
    kit: Kit,
    flce: Flce,
    p_logits_mm: metal::ComputePipelineState,
    pub t_max: usize,
    w: HashMap<String, MBuf>,
    opt: HashMap<String, Opt>,
    lm_head16: MBuf,
    layers: Vec<LayerA>,
    x: Vec<MBuf>,       // NL+1 residual-stream buffers
    xn: MBuf, rfin: MBuf,
    tok: MBuf, tgt: MBuf,
    logits: MBuf, loss: MBuf, dh_top: MBuf,
    mix: MBuf,
    // shared backward scratch
    dwbuf: MBuf, dact: MBuf, dgl: MBuf, dul: MBuf, dh2: MBuf, dx1: MBuf,
    dog: MBuf, d_o: MBuf, dz: MBuf, ds_state: MBuf, dqk: MBuf, dconv: MBuf,
    d_gex: MBuf, d_bet: MBuf, d_gexp: MBuf, d_betp: MBuf,
    dqkv: MBuf, d_ain: MBuf, d_braw: MBuf,
    ds_attn: MBuf, dq2: MBuf, dk2: MBuf, dvv: MBuf, dq0: MBuf, dk0: MBuf,
    dqfull: MBuf, dao: MBuf, daog: MBuf,
    dh: MBuf, dxa: MBuf, dxb: MBuf,
    pub step: u32,
    last_t: usize,
    pub lr: f32,
    pub wd: f32,
    flce_chunk: usize,
    /// LISA rotation: how many decoder layers train per period (0 = all,
    /// i.e. strict full FT). Inactive layers still propagate exact dx;
    /// their dW/optimizer work is skipped. Every layer trains over the run.
    pub lisa_n: usize,
    pub lisa_period: u32,
    active: Vec<bool>,
    /// Per-layer carved MoE FFN (None = dense). When Some, the layer's FFN is
    /// replaced by this MoeFfn in the forward — the carved-0.6B run path.
    carved: Vec<Option<MoeFfn>>,
    /// Shared d_h2 scratch for the heal_* paths (backward's dh2 output is
    /// discarded there) — one t_max*D buffer instead of a fresh alloc per call.
    heal_dh2: MBuf,
    /// Bidirectional (masked-diffusion) attention when true; causal (AR) when false.
    pub bidir: bool,
    /// Per-sequence block length for microbatch packing: attention is block-diagonal and RoPE
    /// resets every `block` positions, so B packed sequences of length `block` (T = B*block)
    /// train together but never attend across the boundary. Default t_max = no packing.
    pub block: usize,
}

fn is_attn(_i: usize) -> bool { true }  // Qwen3-0.6B: every layer is full attention (no GDN)

/// Weight keys stored f16 (all >=1M-param projections; matches the C8
/// optimizer class). wa/wb/norms/dt/alog/conv stay f32.
const F16_KEYS: &[&str] = &[".qkv", ".z", ".wout", ".wq", ".wk", ".wv", ".wo",
                            ".wg", ".wu", ".wd"];

/// Checkpoint container magic + format version. Bumped on any change to the on-disk layout;
/// `load_ckpt` rejects a mismatch so an old ckpt never loads silently into an incompatible reader.
const CKPT_MAGIC: &[u8; 4] = b"OJCK";
const CKPT_VERSION: u32 = 1;

impl<'a> Trainer<'a> {
    pub fn new(gpu: &'a MetalGpu, model_dir: &str, t_max: usize, lr: f32) -> Result<Trainer<'a>> {
        let kit = Kit::new(gpu)?;
        let flce = Flce::new(gpu)?;
        let p_logits_mm = gpu.pipeline(ojas_metal::kernels::source_of("gemm_mm_f16").unwrap(), "gemm_mm_f16")?;
        // ---- stream-load weights shard by shard (single-file fallback for small models) ----
        let idx_path = format!("{model_dir}/model.safetensors.index.json");
        let mut shards: Vec<String> = if std::path::Path::new(&idx_path).exists() {
            let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx_path)?)?;
            let map = idx["weight_map"].as_object().ok_or_else(|| anyhow!("bad index"))?;
            map.values().filter_map(|v| v.as_str().map(String::from)).collect()
        } else {
            vec!["model.safetensors".to_string()]  // Qwen3-0.6B: single unsharded file
        };
        shards.sort();
        shards.dedup();
        let mut w: HashMap<String, MBuf> = HashMap::new();
        let mut lm_head16 = None;
        let t0 = std::time::Instant::now();
        for f in &shards {
            let st = SafeTensors::open(&format!("{model_dir}/{f}"))?;
            let names: Vec<String> = st.names().cloned().collect();
            for n in names {
                let Some(key) = map_name(&n) else { continue };
                let v = st.f32(&n)?;
                // plain Qwen3 uses standard RMSNorm (weight ~1) — no zero-centered +1
                if key == "lm_head" {
                    lm_head16 = Some(gpu.upload_f16(&v)); // frozen, f16 for FLCE
                } else if key == "embed" || F16_KEYS.iter().any(|k| key.ends_with(k)) {
                    // big projections: f16 storage (SR AdamW), halves traffic
                    if key == "embed" && lm_head16.is_none() {
                        lm_head16 = Some(gpu.upload_f16(&v)); // tied embeddings → lm_head = embed
                    }
                    w.insert(key, gpu.upload_f16(&v));
                } else {
                    w.insert(key, gpu.upload(&v));
                }
            }
            tracing::info!(target: "trainer", "loaded shard {f} ({:.0}s)", t0.elapsed().as_secs_f32());
        }
        let lm_head16 = lm_head16.ok_or_else(|| anyhow!("lm_head missing"))?;
        // ---- optimizer states for trainable tensors (everything but embed/lm_head) ----
        let mut opt = HashMap::new();
        for (k, b) in &w {
            if k == "embed" { continue; }
            let n = b.len;
            let state = if n >= 1 << 20 {
                let nb = n.div_ceil(256);
                let (mh, vq) = (gpu.alloc(n.div_ceil(2)), gpu.alloc(n.div_ceil(4)));
                let vs = gpu.alloc(nb);
                for z in [&mh, &vq, &vs] {
                    unsafe { std::ptr::write_bytes(z.buf.contents() as *mut u8, 0, z.len * 4); }
                }
                Opt::C8 { mh, vq, vs }
            } else {
                let (m, v) = (gpu.alloc(n), gpu.alloc(n));
                for z in [&m, &v] {
                    unsafe { std::ptr::write_bytes(z.buf.contents() as *mut u8, 0, z.len * 4); }
                }
                Opt::F32 { m, v }
            };
            opt.insert(k.clone(), state);
        }
        // ---- arenas ----
        let t = t_max;
        let a = |n: usize| gpu.alloc(n);
        let ffn_a = |gpu: &MetalGpu| FfnA {
            x1: gpu.alloc(t * D), h2: gpu.alloc(t * D), rp: gpu.alloc(t),
            gl: gpu.alloc(t * FFN), ul: gpu.alloc(t * FFN), act: gpu.alloc(t * FFN),
        };
        let mut layers = Vec::new();
        for i in 0..NL {
            layers.push(if is_attn(i) {
                LayerA::A(AttnA {
                    h: a(t * D), rln: a(t), qfull: a(t * 2 * QDIM), kfull: a(t * KVDIM),
                    vfull: a(t * KVDIM), q0: a(t * QDIM), q2: a(t * QDIM),
                    rqn: a(t * NH), rkn: a(t * NKV), k2: a(t * KVDIM),
                    p: a(NH * t * t), aog: a(t * QDIM), ao: a(t * QDIM), ffn: ffn_a(gpu),
                })
            } else {
                LayerA::G(GdnA {
                    h: a(t * D), rln: a(t), qkv: a(t * C), z: a(t * DI),
                    ain: a(t * HV), braw: a(t * HV), bet: a(t * HV), sp: a(t * HV),
                    gex: a(t * HV), acc: a(t * C), conv: a(t * C), o: a(t * DI),
                    st: a((t + 1) * HV * S * S), sk: a(t * HV * S), dlt: a(t * HV * S),
                    ron: a(t * HV), og: a(t * DI), ffn: ffn_a(gpu),
                })
            });
        }
        let flce_chunk = 32usize;
        let tr = Trainer {
            gpu, kit, flce, p_logits_mm, t_max, w, opt, lm_head16, layers,
            x: (0..NL + 1).map(|_| a(t * D)).collect(),
            xn: a(t * D), rfin: a(t),
            tok: a(t), tgt: a(t),
            logits: a(flce_chunk * V), loss: a(t), dh_top: a(t * D),
            mix: a(t * D),
            dwbuf: a(FFN * D), dact: a(t * FFN), dgl: a(t * FFN), dul: a(t * FFN),
            dh2: a(t * D), dx1: a(t * D), dog: a(t * DI), d_o: a(t * DI), dz: a(t * DI),
            ds_state: a(HV * S * S), dqk: a(t * HV * 16 * S), dconv: a(t * C),
            d_gex: a(t * HV), d_bet: a(t * HV),
            d_gexp: a(t * HV * 4), d_betp: a(t * HV * 4), dqkv: a(t * C),
            d_ain: a(t * HV), d_braw: a(t * HV),
            ds_attn: a(NH * t * t), dq2: a(t * QDIM), dk2: a(t * KVDIM), dvv: a(t * KVDIM),
            dq0: a(t * QDIM), dk0: a(t * KVDIM), dqfull: a(t * 2 * QDIM),
            dao: a(t * QDIM), daog: a(t * QDIM),
            dh: a(t * D), dxa: a(t * D), dxb: a(t * D),
            step: 0, last_t: 0, lr, wd: 0.0, flce_chunk,
            lisa_n: 0, lisa_period: 25, active: vec![true; NL],
            carved: (0..NL).map(|_| None).collect(),
            heal_dh2: a(t * D),
            bidir: ojas_core::config::var("OJAS_BIDIR").is_ok(),
            block: t_max,
        };
        tracing::info!(target: "trainer", "ready in {:.0}s (T_max={t_max}, lr={lr})", t0.elapsed().as_secs_f32());
        Ok(tr)
    }

    /// Sorted list of all persistent buffers: weights + optimizer states. Save and load use the
    /// same order, so raw bytes stream without name lookups.
    fn ckpt_entries(&self) -> Vec<(String, &MBuf)> {
        let mut e: Vec<(String, &MBuf)> = Vec::new();
        let mut wk: Vec<&String> = self.w.keys().collect(); wk.sort();
        for k in wk { e.push((format!("w:{k}"), &self.w[k])); }
        let mut ok: Vec<&String> = self.opt.keys().collect(); ok.sort();
        for k in ok {
            match &self.opt[k] {
                Opt::C8 { mh, vq, vs } => {
                    e.push((format!("o:{k}:mh"), mh)); e.push((format!("o:{k}:vq"), vq)); e.push((format!("o:{k}:vs"), vs));
                }
                Opt::F32 { m, v } => { e.push((format!("o:{k}:m"), m)); e.push((format!("o:{k}:v"), v)); }
            }
        }
        e
    }

    /// Trainable tensors in a deterministic order: everything carrying optimizer
    /// state, which excludes the frozen `embed` / `lm_head`. Shared by the three
    /// DiLoCo accessors below so a peer's parameter vector is laid out identically.
    fn trainable_keys(&self) -> Vec<String> {
        let mut ks: Vec<String> = self.w.keys().filter(|k| self.opt.contains_key(*k)).cloned().collect();
        ks.sort();
        ks
    }

    /// Architecture signature: FNV-1a over the ordered (name, element-count) list.
    /// Two peers whose signatures differ must never exchange deltas — the vectors
    /// would line up by offset while meaning different tensors.
    pub fn weight_sig(&self) -> u32 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for k in self.trainable_keys() {
            for b in k.as_bytes().iter().chain(self.w[&k].len.to_le_bytes().iter()) {
                h ^= *b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        (h ^ (h >> 32)) as u32
    }

    /// Trainable weights flattened to f32. Storage dtype is per-tensor (f16 for the
    /// big projections, f32 elsewhere), so widen on the way out and narrow on the
    /// way back in; element size comes from the buffer, never assumed.
    pub fn weights_f32(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for k in self.trainable_keys() {
            let b = &self.w[&k];
            unsafe {
                if b.buf.length() as usize / b.len.max(1) == 2 {
                    let p = b.buf.contents() as *const half::f16;
                    out.extend(std::slice::from_raw_parts(p, b.len).iter().map(|h| h.to_f32()));
                } else {
                    out.extend_from_slice(std::slice::from_raw_parts(b.buf.contents() as *const f32, b.len));
                }
            }
        }
        out
    }

    /// Inverse of `weights_f32`. Optimizer moments are deliberately untouched: DiLoCo's outer step
    /// replaces parameters only, and the inner AdamW state stays with the worker that built it.
    pub fn set_weights_f32(&mut self, v: &[f32]) -> Result<()> {
        let mut off = 0usize;
        for k in self.trainable_keys() {
            let b = &self.w[&k];
            anyhow::ensure!(off + b.len <= v.len(), "parameter vector too short at {k}");
            unsafe {
                if b.buf.length() as usize / b.len.max(1) == 2 {
                    let p = b.buf.contents() as *mut half::f16;
                    for i in 0..b.len { *p.add(i) = half::f16::from_f32(v[off + i]); }
                } else {
                    std::ptr::copy_nonoverlapping(v[off..].as_ptr(), b.buf.contents() as *mut f32, b.len);
                }
            }
            off += b.len;
        }
        anyhow::ensure!(off == v.len(), "parameter vector has {} entries, model has {off}", v.len());
        Ok(())
    }

    /// Save full training state (weights + AdamW moments + step) as raw buffer bytes, for resume.
    /// Dtype-agnostic: sizes come from the Metal buffer byte length.
    pub fn save_ckpt(&self, path: &str) -> Result<()> {
        use std::io::Write;
        let tmp = format!("{path}.tmp");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(CKPT_MAGIC)?;
        f.write_all(&CKPT_VERSION.to_le_bytes())?;
        f.write_all(&self.step.to_le_bytes())?;
        let entries = self.ckpt_entries();
        f.write_all(&(entries.len() as u32).to_le_bytes())?;
        for (name, buf) in &entries {
            let nb = name.as_bytes();
            f.write_all(&(nb.len() as u32).to_le_bytes())?;
            f.write_all(nb)?;
            let blen = buf.buf.length() as usize;
            f.write_all(&(blen as u64).to_le_bytes())?;
            let bytes = unsafe { std::slice::from_raw_parts(buf.buf.contents() as *const u8, blen) };
            f.write_all(bytes)?;
        }
        f.flush()?; drop(f);
        std::fs::rename(&tmp, path)?;   // atomic: never leave a half-written ckpt
        Ok(())
    }

    /// Restore weights + optimizer + step from a checkpoint (buffers already allocated by new()).
    pub fn load_ckpt(&mut self, path: &str) -> Result<()> {
        let data = std::fs::read(path)?;
        anyhow::ensure!(data.len() >= 16 && &data[0..4] == CKPT_MAGIC, "bad ckpt magic");
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        anyhow::ensure!(version == CKPT_VERSION, "unsupported ckpt version {version} (expected {CKPT_VERSION})");
        let step = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let entries = self.ckpt_entries();
        let mut off = 16usize;
        for (name, buf) in entries {
            let nlen = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize; off += 4;
            let ename = std::str::from_utf8(&data[off..off + nlen])?; off += nlen;
            anyhow::ensure!(ename == name, "ckpt entry mismatch: {ename} != {name}");
            let blen = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize; off += 8;
            anyhow::ensure!(blen == buf.buf.length() as usize, "ckpt size mismatch for {name}");
            unsafe { std::ptr::copy_nonoverlapping(data[off..].as_ptr(), buf.buf.contents() as *mut u8, blen); }
            off += blen;
        }
        self.step = step;
        Ok(())
    }

    fn wk(&self, i: usize, n: &str) -> &MBuf { &self.w[&format!("L{i}.{n}")] }

    fn layer_trains(&self, i: usize) -> bool { self.lisa_n == 0 || self.active[i] }

    fn opt_step(&self, enc: &metal::ComputeCommandEncoderRef, key: &str, grad: &MBuf) {
        let wb = &self.w[key];
        let n = wb.len as u32;
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let bc1 = 1.0 / (1.0 - b1.powi(self.step as i32 + 1));
        let bc2 = 1.0 / (1.0 - b2.powi(self.step as i32 + 1));
        let fc = [self.lr, b1, b2, bc1, bc2, self.wd];
        match &self.opt[key] {
            Opt::F32 { m, v } => self.kit.df(enc, "t_adamw",
                &[(wb, 0), (grad, 0), (m, 0), (v, 0)], &fc, &[n],
                g1((wb.len).div_ceil(256)), tg256()),
            Opt::C8 { mh, vq, vs } => self.kit.df(enc, "t_adamw_8h",
                &[(wb, 0), (grad, 0), (mh, 0), (vq, 0), (vs, 0)], &fc, &[n, self.step],
                g1(wb.len.div_ceil(2048)), tg256()),
        }
    }

    /// One training step on a single sequence. `targets[t] >= V` is ignored.
    /// Returns (mean loss over valid targets, n_valid).
    pub fn train_step(&mut self, tokens: &[u32], targets: &[u32]) -> Result<(f32, usize)> {
        self.step_core(tokens, None, targets, 0, true)
    }

    /// Replace layer `i`'s dense FFN with a carved MoE FFN for the forward pass
    /// (the carved-0.6B run path). D must match the model hidden size.
    pub fn set_carved(&mut self, i: usize, moe: MoeFfn) {
        self.carved[i] = Some(moe);
    }

    pub fn clear_carved(&mut self) {
        for c in self.carved.iter_mut() {
            *c = None;
        }
    }

    /// Temporarily remove all carved MoEs, reverting the model to dense, to run the dense teacher
    /// on new data mid-experiment. Restore with `restore_carved`.
    pub fn take_carved(&mut self) -> Vec<Option<MoeFfn>> {
        std::mem::replace(&mut self.carved, (0..NL).map(|_| None).collect())
    }

    pub fn restore_carved(&mut self, v: Vec<Option<MoeFfn>>) {
        self.carved = v;
    }

    /// The MLP input (post-attention RMSNorm output, h2) captured for layer `i` from the last
    /// forward: the input that layer's FFN sees in the current, possibly carved, model. Used for
    /// on-policy sequential healing.
    pub fn ffn_input(&self, i: usize) -> Vec<f32> {
        let n = self.last_t * D;
        let h2 = self.ffn_input_buf(i);
        unsafe { std::slice::from_raw_parts(h2.buf.contents() as *const f32, n).to_vec() }
    }

    /// GPU-resident form of `ffn_input`: layer `i`'s captured h2 buffer itself, with no CPU copy
    /// or re-upload, to feed straight into MoeFfn fwd/bwd.
    pub(crate) fn ffn_input_buf(&self, i: usize) -> &MBuf {
        match &self.layers[i] {
            LayerA::A(ar) => &ar.ffn.h2,
            LayerA::G(ar) => &ar.ffn.h2,
        }
    }

    /// The trainer's compiled kernel Kit, shared so callers can heal on the same pipelines without
    /// a second 78-kernel compile.
    pub fn kit(&self) -> &Kit { &self.kit }

    /// Greedy next-token id from the last forward: the final position's logits (xn row t-1 ·
    /// lm_head) via one 1×V GEMM, argmax on CPU. The FLCE path never materializes logits to host,
    /// so a generation loop needs this.
    pub fn greedy_next(&self) -> u32 {
        let t = self.last_t;
        let vout = self.gpu.alloc(V);
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.kit.d(enc, "t_mm_xwT_h",
            &[(&self.xn, ((t - 1) * D * 4) as u64), (&self.lm_head16, 0), (&vout, 0)],
            &[D as u32, V as u32, 1u32],
            g2(1, V.div_ceil(64)), MTLSize::new(128, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let lg = self.gpu.read(&vout);
        let mut best = (f32::MIN, 0u32);
        for (i, &l) in lg.iter().enumerate() {
            if l > best.0 { best = (l, i as u32); }
        }
        best.1
    }

    /// Greedy with a repetition penalty: logits of tokens in `recent` are divided by `penalty`
    /// (> 1) before argmax.
    pub fn greedy_next_penalized(&self, recent: &[u32], penalty: f32) -> u32 {
        let t = self.last_t;
        let vout = self.gpu.alloc(V);
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.kit.d(enc, "t_mm_xwT_h",
            &[(&self.xn, ((t - 1) * D * 4) as u64), (&self.lm_head16, 0), (&vout, 0)],
            &[D as u32, V as u32, 1u32],
            g2(1, V.div_ceil(64)), MTLSize::new(128, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let mut lg = self.gpu.read(&vout);
        for &r in recent {
            let l = &mut lg[r as usize];
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
        let mut best = (f32::MIN, 0u32);
        for (i, &l) in lg.iter().enumerate() {
            if l > best.0 { best = (l, i as u32); }
        }
        best.1
    }

    /// Residual-stream hidden x[i] from the last forward (x[0] = embed, x[NL] = final pre-norm).
    /// Captures the teacher trajectory for drift-correcting heal.
    pub fn hidden(&self, i: usize) -> Vec<f32> {
        let n = self.last_t * D;
        unsafe { std::slice::from_raw_parts(self.x[i].buf.contents() as *const f32, n).to_vec() }
    }

    /// Post-attention residual x1 for layer i (the FFN branch's residual base: x[i+1] = x1 +
    /// FFN(h2)). With teacher x̂[i+1], the drift-correcting FFN target is x̂[i+1] − x1.
    pub fn ffn_x1(&self, i: usize) -> Vec<f32> {
        let f = match &self.layers[i] {
            LayerA::A(ar) => &ar.ffn,
            LayerA::G(ar) => &ar.ffn,
        };
        let n = self.last_t * D;
        unsafe { std::slice::from_raw_parts(f.x1.buf.contents() as *const f32, n).to_vec() }
    }

    /// The global output error d(loss)/d(final_hidden) from the last forward's FLCE, the signal
    /// Direct Feedback Alignment broadcasts to every block.
    pub fn output_error(&self) -> Vec<f32> {
        let n = self.last_t * D;
        unsafe { std::slice::from_raw_parts(self.dh_top.buf.contents() as *const f32, n).to_vec() }
    }

    /// DFA adaptation step on carved block `i`. `dout` is the global output error random-projected
    /// into this block's FFN-output space by the caller. Runs MoeFfn.backward on the block's saved
    /// forward state from the last full forward (no re-forward), updating both the experts and the
    /// router, so the global error reaches the block's routing. No cross-block gradient.
    pub fn heal_carved_block_dfa(&self, i: usize, dout: &[f32], lr: f32, step: u32) {
        let moe = match &self.carved[i] {
            Some(m) => m,
            None => return,
        };
        let t = self.last_t;
        let bh2 = self.ffn_input_buf(i);   // GPU-resident captured input, no re-upload
        let bdout = self.gpu.upload(dout);
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        moe.backward(&self.kit, enc, bh2, &bdout, &self.heal_dh2, t);
        moe.opt_step(&self.kit, enc, lr, 0.0, step);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    /// Identity-structured DFA over every carved block in one command buffer: the global output
    /// error `e_global` is broadcast unchanged (residual-net first-order feedback) to every block's
    /// MoeFfn.backward + opt_step. One command buffer keeps the buffer count low, which the
    /// shared-GPU path needs. Uses each block's saved forward state from the last full forward.
    pub fn heal_all_blocks_identity(&self, e_global: &[f32], lr: f32, step: u32) {
        let t = self.last_t;
        let bdout = self.gpu.upload(e_global);
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        for i in 0..NL {
            if let Some(moe) = &self.carved[i] {
                // saved h2 is already GPU-resident; dh2 output is discarded, so one shared
                // scratch serves all blocks (each backward t_fill-zeroes it)
                moe.backward(&self.kit, enc, self.ffn_input_buf(i), &bdout, &self.heal_dh2, t);
                moe.opt_step(&self.kit, enc, lr, 0.0, step);
            }
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    /// Local adaptation step on carved block `i`: distill its MoeFfn toward `dense_target` (the
    /// dense FFN's output on block i's own input) using the input captured in the last forward.
    /// No cross-block gradient; the caller modulates `lr` by the global modulation signal.
    /// Returns the block's MSE before the step.
    pub fn heal_carved_block(&self, i: usize, dense_target: &[f32], steps: usize, lr: f32) -> f32 {
        let moe = match &self.carved[i] {
            Some(m) => m,
            None => return 0.0,
        };
        let t = self.last_t;
        let n = (t * D) as f32;
        let bx = self.ffn_input_buf(i);            // GPU-resident captured input
        let bout = self.gpu.alloc(t * D);
        let btgt = self.gpu.upload(dense_target);
        let bdout = self.gpu.alloc(t * D);
        let bnrm = self.gpu.alloc(1);
        let mut first = 0.0;
        for s in 0..steps {
            objc::rc::autoreleasepool(|| {
                // one fused command buffer: forward + GPU loss grad (dout = (2/n)(out - tgt))
                // + Σdout² + backward + AdamW. Wait only on step 0 (first-MSE read), every 10th
                // step, and the last step.
                let cb = self.gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                moe.forward(&self.kit, enc, bx, &bout, t);
                self.kit.df(enc, "t_lincomb2", &[(&bdout, 0), (&bout, 0), (&btgt, 0)], &[2.0 / n, -2.0 / n], &[(t * D) as u32], g1((t * D).div_ceil(256)), tg256());
                self.kit.d(enc, "t_sumsq", &[(&bdout, 0), (&bnrm, 0)], &[(t * D) as u32], g1(1), tg256());
                moe.backward(&self.kit, enc, bx, &bdout, &self.heal_dh2, t);
                moe.opt_step(&self.kit, enc, lr, 0.0, s as u32);
                enc.end_encoding();
                cb.commit();
                if s == 0 || s % 10 == 9 || s + 1 == steps {
                    cb.wait_until_completed();
                    if s == 0 {
                        // bnrm = Σ((2/n)e)² = (4/n²)·SSE  =>  MSE = SSE/n = (n/4)·bnrm
                        first = (n / 4.0) * self.gpu.read(&bnrm)[0];
                    }
                }
            });
        }
        first
    }

    /// `heal_carved_block` with the dense target computed on GPU internally. The layer's dense
    /// weights stay resident after set_carved (self.w L{i}.wg/wu/wd) and its FfnA gl/ul/act scratch
    /// is unused while the layer is carved (fwd_ffn's carved branch skips it), so the dense SwiGLU
    /// runs from the captured h2 (ffn_input_buf) into a target buffer, then the same fused heal
    /// loop. Removes the per-(seq, layer) CPU dense_forward and upload. Returns the block's MSE
    /// before the first step.
    pub fn heal_carved_block_dense_target(&self, i: usize, steps: usize, lr: f32) -> f32 {
        let moe = match &self.carved[i] {
            Some(m) => m,
            None => return 0.0,
        };
        let t = self.last_t;
        let n = (t * D) as f32;
        let bx = self.ffn_input_buf(i);            // GPU-resident captured input (== f.h2)
        let f = match &self.layers[i] {
            LayerA::A(ar) => &ar.ffn,
            LayerA::G(ar) => &ar.ffn,
        };
        let bout = self.gpu.alloc(t * D);
        let btgt = self.gpu.alloc(t * D);
        let bdout = self.gpu.alloc(t * D);
        let bnrm = self.gpu.alloc(1);
        // setup: dense teacher output on GPU (weights + scratch already resident)
        objc::rc::autoreleasepool(|| {
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            encode_dense_ffn(&self.kit, enc, bx, self.wk(i, "wg"), self.wk(i, "wu"), self.wk(i, "wd"),
                             &f.gl, &f.ul, &f.act, &btgt, t, D, FFN);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });
        let mut first = 0.0;
        for s in 0..steps {
            objc::rc::autoreleasepool(|| {
                // one fused command buffer: forward + GPU loss grad (dout = (2/n)(out - tgt))
                // + Σdout² + backward + AdamW. Wait only on step 0 (first-MSE read), every 10th
                // step, and the last step.
                let cb = self.gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                moe.forward(&self.kit, enc, bx, &bout, t);
                self.kit.df(enc, "t_lincomb2", &[(&bdout, 0), (&bout, 0), (&btgt, 0)], &[2.0 / n, -2.0 / n], &[(t * D) as u32], g1((t * D).div_ceil(256)), tg256());
                self.kit.d(enc, "t_sumsq", &[(&bdout, 0), (&bnrm, 0)], &[(t * D) as u32], g1(1), tg256());
                moe.backward(&self.kit, enc, bx, &bdout, &self.heal_dh2, t);
                moe.opt_step(&self.kit, enc, lr, 0.0, s as u32);
                enc.end_encoding();
                cb.commit();
                if s == 0 || s % 10 == 9 || s + 1 == steps {
                    cb.wait_until_completed();
                    if s == 0 {
                        // bnrm = Σ((2/n)e)² = (4/n²)·SSE  =>  MSE = SSE/n = (n/4)·bnrm
                        first = (n / 4.0) * self.gpu.read(&bnrm)[0];
                    }
                }
            });
        }
        first
    }

    /// Worker core: forward layers [lo, NL) (plus the FLCE loss) with an optional injected
    /// activation. lo == 0 embeds `tokens`; lo > 0 writes `x_in` ([t*D] f32, from the upstream
    /// worker) into x[lo]. do_bwd = false gives the loss only, leaving activations resident for a
    /// later bwd_from(lo). Entry point for the decentralized serve-and-learn path.
    pub fn step_core(&mut self, tokens: &[u32], x_in: Option<&[f32]>, targets: &[u32],
                     lo: usize, do_bwd: bool) -> Result<(f32, usize)> {
        let t = tokens.len();
        if t > self.t_max || t != targets.len() { bail!("bad seq len"); }
        let n_valid = targets.iter().filter(|&&y| (y as usize) < V).count().max(1);
        if self.lisa_n > 0 && self.step % self.lisa_period == 0 {
            // deterministic per-period sample of active layers (LCG on period)
            self.active = vec![false; NL];
            let mut r = (self.step / self.lisa_period) as u64 ^ 0x9e3779b97f4a7c15;
            let mut picked = 0;
            while picked < self.lisa_n.min(NL) {
                r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let l = ((r >> 33) as usize) % NL;
                if !self.active[l] { self.active[l] = true; picked += 1; }
            }
        }
        unsafe {
            let tp = self.tok.buf.contents() as *mut f32;
            let gp = self.tgt.buf.contents() as *mut f32;
            for i in 0..t {
                *tp.add(i) = tokens[i] as f32;
                *gp.add(i) = targets[i] as f32;
            }
        }
        let tt0 = std::time::Instant::now();
        self.last_t = t;
        // ---- forward: embed (stage 0) or inject upstream activation ----
        if lo == 0 {
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.kit.d(enc, "t_embed_fwd", &[(&self.tok, 0), (&self.w["embed"], 0), (&self.x[0], 0)],
                       &[D as u32, (t * D) as u32], g1((t * D).div_ceil(256)), tg256());
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        } else {
            let xi = x_in.ok_or_else(|| anyhow!("lo>0 requires x_in"))?;
            anyhow::ensure!(xi.len() == t * D, "x_in len");
            unsafe { std::ptr::copy_nonoverlapping(xi.as_ptr(), self.x[lo].buf.contents() as *mut f32, t * D); }
        }
        let t2f = ojas_core::config::var("OJAS_TRAIN_TIMING").map(|v| v == "2").unwrap_or(false);
        let fb = if t2f { 1 } else { 8 };
        for i0 in (lo..NL).step_by(fb) {
            let lt = std::time::Instant::now();
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            for i in i0..(i0 + fb).min(NL) {
                match &self.layers[i] {
                    LayerA::G(_) => self.fwd_gdn(enc, i, t),
                    LayerA::A(_) => self.fwd_attn(enc, i, t),
                }
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            if t2f && self.step == 0 {
                tracing::trace!(target: "fwd", "L{i0:2} {:.0}ms", lt.elapsed().as_secs_f32() * 1000.0);
            }
        }
        let t_fwd = tt0.elapsed().as_secs_f32();
        // ---- forward-parity dump (OJAS_DUMP=1, step 0): x[0]=embed, x[1]=after L0, x[NL]=final pre-norm ----
        if ojas_core::config::var("OJAS_DUMP").is_ok() && self.step == 0 {
            let dump = |name: &str, buf: &MBuf| {
                let s = unsafe { std::slice::from_raw_parts(buf.buf.contents() as *const f32, t * D) };
                let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, t * D * 4) };
                std::fs::write(format!("/tmp/tr_{name}.f32"), bytes).unwrap();
            };
            dump("x0", &self.x[0]); dump("x1", &self.x[1]); dump("xNL", &self.x[NL]);
            if let LayerA::A(ar) = &self.layers[0] {
                dump("h0", &ar.h);          // L0 input_layernorm output
                dump("att0", &ar.ffn.x1);   // L0 post-attention residual (x0 + attn_out)
                dump("h2_0", &ar.ffn.h2);   // L0 post_attention_layernorm output
                let dq = |name: &str, buf: &MBuf, w: usize| {
                    let s = unsafe { std::slice::from_raw_parts(buf.buf.contents() as *const f32, t*w) };
                    let b = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, t*w*4) };
                    std::fs::write(format!("/tmp/tr_{name}.f32"), b).unwrap();
                };
                dq("q2", &ar.q2, QDIM); dq("k2", &ar.k2, KVDIM);
            }
            tracing::debug!(target: "dump", "wrote /tmp/tr_x0,x1,xNL,h0,att0,h2_0.f32 (t={t} D={D})");
        }
        let tt1 = std::time::Instant::now();
        // ---- final norm + FLCE ----
        let scale = 1.0 / n_valid as f32;
        {
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.kit.d(enc, "t_rms_fwd", &[(&self.x[NL], 0), (&self.w["final_norm"], 0), (&self.xn, 0), (&self.rfin, 0)],
                       &[D as u32, t as u32], g1(t.div_ceil(8)), tg256());
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let t2c = ojas_core::config::var("OJAS_TRAIN_TIMING").map(|v| v == "2").unwrap_or(false);
        let mut flt = [0f32; 3];
        for c0 in (0..t).step_by(self.flce_chunk) {
            let cn = self.flce_chunk.min(t - c0);
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            let lt = std::time::Instant::now();
            let (du, vu, c0u) = (D as u32, V as u32, c0 as u32);
            let cnu = cn as u32;
            enc.set_compute_pipeline_state(&self.p_logits_mm);
            enc.set_buffer(0, Some(&self.xn.buf), (c0 * D * 4) as u64);
            enc.set_buffer(1, Some(&self.lm_head16.buf), 0);
            enc.set_buffer(2, Some(&self.logits.buf), 0);
            enc.set_bytes(3, 4, &du as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &vu as *const u32 as *const std::ffi::c_void);
            let zero = 0u32;
            enc.set_bytes(6, 4, &zero as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &cnu as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(g2(cn.div_ceil(32), V / 64), MTLSize::new(128, 1, 1));
            let (cb, enc) = if t2c {
                enc.end_encoding(); cb.commit(); cb.wait_until_completed();
                flt[0] += lt.elapsed().as_secs_f32();
                let cb2 = self.gpu.command_buffer();
                let e2 = cb2.new_compute_command_encoder();
                (cb2, e2)
            } else { (cb, enc) };
            let lt = std::time::Instant::now();
            unsafe { let _: () = objc::msg_send![enc, memoryBarrierWithScope: 1u64]; }
            enc.set_compute_pipeline_state(&self.flce.p_row);
            enc.set_buffer(0, Some(&self.logits.buf), 0);
            enc.set_buffer(1, Some(&self.tgt.buf), 0);
            enc.set_buffer(2, Some(&self.loss.buf), 0);
            enc.set_bytes(3, 4, &vu as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &scale as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &c0u as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(g1(cn), MTLSize::new(1024, 1, 1));
            let (cb, enc) = if t2c {
                enc.end_encoding(); cb.commit(); cb.wait_until_completed();
                flt[1] += lt.elapsed().as_secs_f32();
                let cb2 = self.gpu.command_buffer();
                let e2 = cb2.new_compute_command_encoder();
                (cb2, e2)
            } else { (cb, enc) };
            let lt = std::time::Instant::now();
            unsafe { let _: () = objc::msg_send![enc, memoryBarrierWithScope: 1u64]; }
            // dh = dlogits @ lm_head — MMA (t_mm_dx_h), consts [IN=D, OUT=V, accum, M]
            self.kit.d(enc, "t_mm_dx_h",
                &[(&self.logits, 0), (&self.lm_head16, 0), (&self.dh_top, (c0 * D * 4) as u64)],
                &[du, vu, 0, cnu], g2(cn.div_ceil(32), D / 64), MTLSize::new(128, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            if t2c { flt[2] += lt.elapsed().as_secs_f32(); }
        }
        if t2c && self.step == 0 {
            tracing::debug!(target: "flce", "mm {:.2}s  row {:.2}s  dh {:.2}s", flt[0], flt[1], flt[2]);
        }
        let mean_loss: f32 = unsafe {
            std::slice::from_raw_parts(self.loss.buf.contents() as *const f32, t)
        }.iter().sum::<f32>() / n_valid as f32;
        let t_flce = tt1.elapsed().as_secs_f32();
        if !do_bwd {
            if ojas_core::config::var("OJAS_TRAIN_TIMING").is_ok() {
                tracing::debug!(target: "timing", "fwd {t_fwd:.2}s  flce {t_flce:.2}s  (probe only)");
            }
            return Ok((mean_loss, n_valid));
        }
        let _ = t_fwd;
        self.bwd_from(lo)?;
        self.step += 1;
        Ok((mean_loss, n_valid))
    }

    /// Backward + optimizer from the top down to layer `lo` (exclusive floor): the tail
    /// worker's gated update. Uses activations left resident by the last step_core call.
    pub fn bwd_from(&mut self, lo: usize) -> Result<()> {
        let t = self.last_t;
        let tt2 = std::time::Instant::now();
        // ---- backward: final norm, then layers in reverse ----
        {
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            unsafe { std::ptr::write_bytes(self.dxa.buf.contents() as *mut u8, 0, t * D * 4); }
            self.kit.d(enc, "t_rms_bwd", &[(&self.x[NL], 0), (&self.w["final_norm"], 0), (&self.rfin, 0), (&self.dh_top, 0), (&self.dxa, 0)],
                       &[D as u32, t as u32, 1], g1(t.div_ceil(8)), tg256());
            self.kit.d(enc, "t_rms_dw", &[(&self.x[NL], 0), (&self.rfin, 0), (&self.dh_top, 0), (&self.dwbuf, 0)],
                       &[D as u32, t as u32], g1(D.div_ceil(256)), tg256());
            self.opt_step(enc, "final_norm", &self.dwbuf);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let t2 = ojas_core::config::var("OJAS_TRAIN_TIMING").map(|v| v == "2").unwrap_or(false);
        let bb = if t2 { 1 } else { 4 };
        let mut i = NL;
        while i > lo {
            let lt = std::time::Instant::now();
            let n = bb.min(i);
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            for _ in 0..n {
                i -= 1;
                match &self.layers[i] {
                    LayerA::G(_) => self.bwd_gdn(enc, i, t),
                    LayerA::A(_) => self.bwd_attn(enc, i, t),
                }
                // dxb of this layer is dxa of the next-lower layer
                std::mem::swap(&mut self.dxa, &mut self.dxb);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            if t2 && self.step == 0 {
                tracing::trace!(target: "bwd", "L{i:2} {:.0}ms", lt.elapsed().as_secs_f32() * 1000.0);
            }
        }
        if ojas_core::config::var("OJAS_TRAIN_TIMING").is_ok() {
            tracing::debug!(target: "timing", "bwd {:.2}s", tt2.elapsed().as_secs_f32());
        }
        Ok(())
    }

    /// Pure span forward for middle workers: layers [lo, hi), no loss/backward.
    /// Returns the outgoing activation x[hi] ([t*D] f32) for the next worker.
    pub fn fwd_span_range(&mut self, tokens: &[u32], x_in: Option<&[f32]>, lo: usize, hi: usize)
                          -> Result<Vec<f32>> {
        let t = tokens.len();
        anyhow::ensure!(t <= self.t_max && hi <= NL && lo < hi, "bad span");
        self.last_t = t;
        unsafe {
            let tp = self.tok.buf.contents() as *mut f32;
            for i in 0..t { *tp.add(i) = tokens[i] as f32; }
        }
        if lo == 0 {
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.kit.d(enc, "t_embed_fwd", &[(&self.tok, 0), (&self.w["embed"], 0), (&self.x[0], 0)],
                       &[D as u32, (t * D) as u32], g1((t * D).div_ceil(256)), tg256());
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        } else {
            let xi = x_in.ok_or_else(|| anyhow!("lo>0 requires x_in"))?;
            unsafe { std::ptr::copy_nonoverlapping(xi.as_ptr(), self.x[lo].buf.contents() as *mut f32, t * D); }
        }
        for i0 in (lo..hi).step_by(8) {
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            for i in i0..(i0 + 8).min(hi) {
                match &self.layers[i] {
                    LayerA::G(_) => self.fwd_gdn(enc, i, t),
                    LayerA::A(_) => self.fwd_attn(enc, i, t),
                }
            }
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        }
        let ptr = self.x[hi].buf.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(ptr, t * D) }.to_vec())
    }

    // ---------- per-layer encode ----------

    fn fwd_ffn(&self, enc: &metal::ComputeCommandEncoderRef, i: usize, t: usize, f: &FfnA) {
        let (du, t32, f32_) = (D as u32, t as u32, FFN as u32);
        let el = |n: usize| g1(n.div_ceil(256));
        self.kit.d(enc, "t_copy", &[(&f.x1, 0), (&self.x[i], 0)], &[(t * D) as u32], el(t * D), tg256());
        self.kit.d(enc, "t_add", &[(&f.x1, 0), (&self.mix, 0)], &[(t * D) as u32], el(t * D), tg256());
        self.kit.d(enc, "t_rms_fwd", &[(&f.x1, 0), (self.wk(i, "pln"), 0), (&f.h2, 0), (&f.rp, 0)], &[du, t32], g1(t.div_ceil(8)), tg256());
        if let Some(moe) = &self.carved[i] {
            // carved MoE FFN: out = MoE(h2), then residual. Grouped-GEMM top-k fast path
            // (encoder-based, no readbacks; rel ~1e-8 vs the dense-all-expert reference and
            // 2-3x faster). Heal paths train with MoeFfn::forward, which agrees to 1e-8, so
            // healed weights transfer.
            moe.forward_gs_enc(&self.kit, enc, &f.h2, &self.x[i + 1], t);
        } else {
            self.kit.d(enc, "t_mm_xwT_h", &[(&f.h2, 0), (self.wk(i, "wg"), 0), (&f.gl, 0)], &[du, f32_, t32], g2(t.div_ceil(32), (FFN).div_ceil(64)), MTLSize::new(128, 1, 1));
            self.kit.d(enc, "t_mm_xwT_h", &[(&f.h2, 0), (self.wk(i, "wu"), 0), (&f.ul, 0)], &[du, f32_, t32], g2(t.div_ceil(32), (FFN).div_ceil(64)), MTLSize::new(128, 1, 1));
            self.kit.d(enc, "t_swiglu_fwd", &[(&f.gl, 0), (&f.ul, 0), (&f.act, 0)], &[(t * FFN) as u32], el(t * FFN), tg256());
            self.kit.d(enc, "t_mm_xwT_h", &[(&f.act, 0), (self.wk(i, "wd"), 0), (&self.x[i + 1], 0)], &[f32_, du, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        }
        self.kit.d(enc, "t_add", &[(&self.x[i + 1], 0), (&f.x1, 0)], &[(t * D) as u32], el(t * D), tg256());
    }

    fn bwd_ffn(&self, enc: &metal::ComputeCommandEncoderRef, i: usize, t: usize, f: &FfnA) {
        // consumes self.dxa (dOut), leaves dMix in self.dx1
        let (du, t32, f32_) = (D as u32, t as u32, FFN as u32);
        let el = |n: usize| g1(n.div_ceil(256));
        if self.layer_trains(i) {
        self.kit.d(enc, "t_mm_dw", &[(&self.dxa, 0), (&f.act, 0), (&self.dwbuf, 0)], &[f32_, du, t32], g2((FFN).div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        }
        self.kit.d(enc, "t_mm_dx_h", &[(&self.dxa, 0), (self.wk(i, "wd"), 0), (&self.dact, 0)], &[f32_, du, 0, t32], g2(t.div_ceil(32), (FFN).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.wd"), &self.dwbuf); }
        self.kit.d(enc, "t_swiglu_bwd", &[(&self.dact, 0), (&f.gl, 0), (&f.ul, 0), (&self.dgl, 0), (&self.dul, 0)], &[(t * FFN) as u32], el(t * FFN), tg256());
        if self.layer_trains(i) {
        self.kit.d(enc, "t_mm_dw", &[(&self.dgl, 0), (&f.h2, 0), (&self.dwbuf, 0)], &[du, f32_, t32], g2((D).div_ceil(32), (FFN).div_ceil(64)), MTLSize::new(128, 1, 1));
        }
        self.kit.d(enc, "t_mm_dx_h", &[(&self.dgl, 0), (self.wk(i, "wg"), 0), (&self.dh2, 0)], &[du, f32_, 0, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.wg"), &self.dwbuf); }
        if self.layer_trains(i) {
        self.kit.d(enc, "t_mm_dw", &[(&self.dul, 0), (&f.h2, 0), (&self.dwbuf, 0)], &[du, f32_, t32], g2((D).div_ceil(32), (FFN).div_ceil(64)), MTLSize::new(128, 1, 1));
        }
        self.kit.d(enc, "t_mm_dx_h", &[(&self.dul, 0), (self.wk(i, "wu"), 0), (&self.dh2, 0)], &[du, f32_, 1, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.wu"), &self.dwbuf); }
        self.kit.d(enc, "t_copy", &[(&self.dx1, 0), (&self.dxa, 0)], &[(t * D) as u32], el(t * D), tg256());
        self.kit.d(enc, "t_rms_bwd", &[(&f.x1, 0), (self.wk(i, "pln"), 0), (&f.rp, 0), (&self.dh2, 0), (&self.dx1, 0)], &[du, t32, 1], g1(t.div_ceil(8)), tg256());
        if self.layer_trains(i) {
        self.kit.d(enc, "t_rms_dw", &[(&f.x1, 0), (&f.rp, 0), (&self.dh2, 0), (&self.dwbuf, 0)], &[du, t32], g1(D.div_ceil(256)), tg256());
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.pln"), &self.dwbuf); }
        }
    }

    fn fwd_gdn(&self, enc: &metal::ComputeCommandEncoderRef, i: usize, t: usize) {
        let LayerA::G(ar) = &self.layers[i] else { unreachable!() };
        let (du, t32) = (D as u32, t as u32);
        let (c32, di32, hv32, hk32, s32) = (C as u32, DI as u32, HV as u32, HK as u32, S as u32);
        self.kit.d(enc, "t_rms_fwd", &[(&self.x[i], 0), (self.wk(i, "ln"), 0), (&ar.h, 0), (&ar.rln, 0)], &[du, t32], g1(t.div_ceil(8)), tg256());
        self.kit.d(enc, "t_mm_xwT_h", &[(&ar.h, 0), (self.wk(i, "qkv"), 0), (&ar.qkv, 0)], &[du, c32, t32], g2(t.div_ceil(32), (C).div_ceil(64)), MTLSize::new(128, 1, 1));
        self.kit.d(enc, "t_mm_xwT_h", &[(&ar.h, 0), (self.wk(i, "z"), 0), (&ar.z, 0)], &[du, di32, t32], g2(t.div_ceil(32), (DI).div_ceil(64)), MTLSize::new(128, 1, 1));
        self.kit.d(enc, "t_gemm_xwT", &[(&ar.h, 0), (self.wk(i, "wa"), 0), (&ar.ain, 0)], &[du, hv32, t32], g2(HV.div_ceil(8), t.div_ceil(16)), tg256());
        self.kit.d(enc, "t_gemm_xwT", &[(&ar.h, 0), (self.wk(i, "wb"), 0), (&ar.braw, 0)], &[du, hv32, t32], g2(HV.div_ceil(8), t.div_ceil(16)), tg256());
        self.kit.d(enc, "t_gates_fwd", &[(&ar.ain, 0), (&ar.braw, 0), (self.wk(i, "dt"), 0), (self.wk(i, "alog"), 0), (&ar.bet, 0), (&ar.sp, 0), (&ar.gex, 0)],
                   &[hv32, (t * HV) as u32], g1((t * HV).div_ceil(256)), tg256());
        self.kit.d(enc, "t_conv_fwd", &[(&ar.qkv, 0), (self.wk(i, "cw"), 0), (&ar.acc, 0), (&ar.conv, 0)], &[c32, t32], g2(C.div_ceil(64), t), MTLSize::new(64, 1, 1));
        self.kit.d(enc, "t_dn_fwd", &[(&ar.conv, 0), (&ar.gex, 0), (&ar.bet, 0), (&ar.o, 0), (&ar.st, 0), (&ar.sk, 0), (&ar.dlt, 0)],
                   &[s32, hk32, hv32, c32, t32], g2(S / 4, HV), tg128());
        self.kit.d(enc, "t_gnorm_fwd", &[(&ar.o, 0), (&ar.z, 0), (self.wk(i, "nw"), 0), (&ar.og, 0), (&ar.ron, 0)], &[s32, (t * HV) as u32], g1((t * HV).div_ceil(8)), tg256());
        self.kit.d(enc, "t_mm_xwT_h", &[(&ar.og, 0), (self.wk(i, "wout"), 0), (&self.mix, 0)], &[di32, du, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        self.fwd_ffn(enc, i, t, &ar.ffn);
    }

    fn bwd_gdn(&self, enc: &metal::ComputeCommandEncoderRef, i: usize, t: usize) {
        let LayerA::G(ar) = &self.layers[i] else { unreachable!() };
        let (du, t32) = (D as u32, t as u32);
        let (c32, di32, hv32, hk32, s32) = (C as u32, DI as u32, HV as u32, HK as u32, S as u32);
        self.bwd_ffn(enc, i, t, &ar.ffn);
        if self.layer_trains(i) {
        self.kit.d(enc, "t_mm_dw", &[(&self.dx1, 0), (&ar.og, 0), (&self.dwbuf, 0)], &[di32, du, t32], g2((DI).div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        }
        self.kit.d(enc, "t_mm_dx_h", &[(&self.dx1, 0), (self.wk(i, "wout"), 0), (&self.dog, 0)], &[di32, du, 0, t32], g2(t.div_ceil(32), (DI).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.wout"), &self.dwbuf); }
        self.kit.d(enc, "t_gnorm_bwd", &[(&ar.o, 0), (&ar.z, 0), (self.wk(i, "nw"), 0), (&ar.ron, 0), (&self.dog, 0), (&self.d_o, 0), (&self.dz, 0)],
                   &[s32, (t * HV) as u32], g1((t * HV).div_ceil(8)), tg256());
        if self.layer_trains(i) {
        self.kit.d(enc, "t_gnorm_dnw", &[(&ar.o, 0), (&ar.z, 0), (&ar.ron, 0), (&self.dog, 0), (&self.dwbuf, 0)], &[s32, (t * HV) as u32], g1(1), tg128());
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.nw"), &self.dwbuf); }
        }
        self.kit.d(enc, "t_fill", &[(&self.ds_state, 0)], &[(HV * S * S) as u32],
                   g1((HV * S * S).div_ceil(256)), tg256());
        self.kit.d(enc, "t_dn_bwd", &[(&ar.conv, 0), (&ar.gex, 0), (&ar.bet, 0), (&self.d_o, 0), (&ar.st, 0), (&ar.sk, 0), (&ar.dlt, 0),
                   (&self.ds_state, 0), (&self.dqk, 0), (&self.dconv, 0), (&self.d_gexp, 0), (&self.d_betp, 0)],
                   &[s32, hk32, hv32, c32, t32], g2(4, HV), tg128());
        self.kit.d(enc, "t_dng_fold", &[(&self.d_gexp, 0), (&self.d_betp, 0), (&self.d_gex, 0), (&self.d_bet, 0)], &[(t * HV) as u32], g1((t * HV).div_ceil(256)), tg256());
        self.kit.d(enc, "t_dnqk_fold", &[(&self.dqk, 0), (&self.dconv, 0), (&ar.conv, 0)], &[s32, hk32, hv32, c32, t32], g2(S.div_ceil(64), t * HK), MTLSize::new(64, 1, 1));
        self.kit.d(enc, "t_conv_bwd", &[(&self.dconv, 0), (&ar.acc, 0), (self.wk(i, "cw"), 0), (&self.dqkv, 0)], &[c32, t32], g2(C.div_ceil(64), t), MTLSize::new(64, 1, 1));
        if self.layer_trains(i) {
        self.kit.d(enc, "t_conv_dw", &[(&self.dconv, 0), (&ar.acc, 0), (&ar.qkv, 0), (&self.dwbuf, 0)], &[c32, t32], g2(C.div_ceil(64), 4), MTLSize::new(64, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.cw"), &self.dwbuf); }
        }
        self.kit.d(enc, "t_gates_bwd", &[(&self.d_gex, 0), (&self.d_bet, 0), (&ar.gex, 0), (&ar.bet, 0), (&ar.ain, 0), (self.wk(i, "dt"), 0), (self.wk(i, "alog"), 0), (&self.d_ain, 0), (&self.d_braw, 0)],
                   &[hv32, (t * HV) as u32], g1((t * HV).div_ceil(256)), tg256());
        if self.layer_trains(i) {
        self.kit.d(enc, "t_gates_dtb", &[(&self.d_ain, 0), (&self.d_gex, 0), (&ar.gex, 0), (&ar.sp, 0), (self.wk(i, "alog"), 0), (&self.dwbuf, 0), (&self.dwbuf, (HV * 4) as u64)],
                   &[hv32, t32], g1(1), tg128());
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.dt"), &self.dwbuf); }
        // A_log grad sits at offset HV in dwbuf — move to front for opt_step
        self.kit.d(enc, "t_copy", &[(&self.dwbuf, 0), (&self.dwbuf, (HV * 4) as u64)], &[HV as u32], g1(1), tg128());
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.alog"), &self.dwbuf); }
        }
        if self.layer_trains(i) {
        self.kit.d(enc, "t_mm_dw", &[(&self.dqkv, 0), (&ar.h, 0), (&self.dwbuf, 0)], &[du, c32, t32], g2((D).div_ceil(32), (C).div_ceil(64)), MTLSize::new(128, 1, 1));
        }
        self.kit.d(enc, "t_mm_dx_h", &[(&self.dqkv, 0), (self.wk(i, "qkv"), 0), (&self.dh, 0)], &[du, c32, 0, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.qkv"), &self.dwbuf); }
        if self.layer_trains(i) {
        self.kit.d(enc, "t_mm_dw", &[(&self.dz, 0), (&ar.h, 0), (&self.dwbuf, 0)], &[du, di32, t32], g2((D).div_ceil(32), (DI).div_ceil(64)), MTLSize::new(128, 1, 1));
        }
        self.kit.d(enc, "t_mm_dx_h", &[(&self.dz, 0), (self.wk(i, "z"), 0), (&self.dh, 0)], &[du, di32, 1, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.z"), &self.dwbuf); }
        if self.layer_trains(i) {
        self.kit.d(enc, "t_gemm_dw", &[(&self.d_ain, 0), (&ar.h, 0), (&self.dwbuf, 0)], &[du, hv32, t32], g2((D).div_ceil(1024), (HV).div_ceil(8)), tg256());
        }
        self.kit.d(enc, "t_mm_dx", &[(&self.d_ain, 0), (self.wk(i, "wa"), 0), (&self.dh, 0)], &[du, hv32, 1, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.wa"), &self.dwbuf); }
        if self.layer_trains(i) {
        self.kit.d(enc, "t_gemm_dw", &[(&self.d_braw, 0), (&ar.h, 0), (&self.dwbuf, 0)], &[du, hv32, t32], g2((D).div_ceil(1024), (HV).div_ceil(8)), tg256());
        }
        self.kit.d(enc, "t_mm_dx", &[(&self.d_braw, 0), (self.wk(i, "wb"), 0), (&self.dh, 0)], &[du, hv32, 1, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.wb"), &self.dwbuf); }
        self.kit.d(enc, "t_copy", &[(&self.dxb, 0), (&self.dx1, 0)], &[(t * D) as u32], g1((t * D).div_ceil(256)), tg256());
        self.kit.d(enc, "t_rms_bwd", &[(&self.x[i], 0), (self.wk(i, "ln"), 0), (&ar.rln, 0), (&self.dh, 0), (&self.dxb, 0)], &[du, t32, 1], g1(t.div_ceil(8)), tg256());
        if self.layer_trains(i) {
        self.kit.d(enc, "t_rms_dw", &[(&self.x[i], 0), (&ar.rln, 0), (&self.dh, 0), (&self.dwbuf, 0)], &[du, t32], g1(D.div_ceil(256)), tg256());
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.ln"), &self.dwbuf); }
        }
    }

    fn fwd_attn(&self, enc: &metal::ComputeCommandEncoderRef, i: usize, t: usize) {
        let LayerA::A(ar) = &self.layers[i] else { unreachable!() };
        let (du, t32, hd32, nh32, nkv32, rot32) = (D as u32, t as u32, HD as u32, NH as u32, NKV as u32, ROT as u32);
        // Qwen3 attention is ungated: q_proj -> QDIM (no fused gate); q0 = qfull, ao = aog.
        let (qd32, kvd32) = (QDIM as u32, KVDIM as u32);
        let block32 = self.block as u32;   // block-diagonal packing (used by rope below)
        let el = |n: usize| g1(n.div_ceil(256));
        self.kit.d(enc, "t_rms_fwd", &[(&self.x[i], 0), (self.wk(i, "ln"), 0), (&ar.h, 0), (&ar.rln, 0)], &[du, t32], g1(t.div_ceil(8)), tg256());
        self.kit.d(enc, "t_mm_xwT_h", &[(&ar.h, 0), (self.wk(i, "wq"), 0), (&ar.qfull, 0)], &[du, qd32, t32], g2(t.div_ceil(32), (QDIM).div_ceil(64)), MTLSize::new(128, 1, 1));
        self.kit.d(enc, "t_mm_xwT_h", &[(&ar.h, 0), (self.wk(i, "wk"), 0), (&ar.kfull, 0)], &[du, kvd32, t32], g2(t.div_ceil(32), (KVDIM).div_ceil(64)), MTLSize::new(128, 1, 1));
        self.kit.d(enc, "t_mm_xwT_h", &[(&ar.h, 0), (self.wk(i, "wv"), 0), (&ar.vfull, 0)], &[du, kvd32, t32], g2(t.div_ceil(32), (KVDIM).div_ceil(64)), MTLSize::new(128, 1, 1));
        self.kit.d(enc, "t_copy", &[(&ar.q0, 0), (&ar.qfull, 0)], &[(t * QDIM) as u32], el(t * QDIM), tg256()); // q0 = qfull (dst,src)
        self.kit.d(enc, "t_rms_fwd", &[(&ar.q0, 0), (self.wk(i, "qnw"), 0), (&ar.q2, 0), (&ar.rqn, 0)], &[hd32, (t * NH) as u32], g1((t * NH).div_ceil(8)), tg256());
        self.kit.d(enc, "t_rms_fwd", &[(&ar.kfull, 0), (self.wk(i, "knw"), 0), (&ar.k2, 0), (&ar.rkn, 0)], &[hd32, (t * NKV) as u32], g1((t * NKV).div_ceil(8)), tg256());
        self.kit.d(enc, "t_rope", &[(&ar.q2, 0)], &[hd32, nh32, rot32, 0, (t * NH * ROT / 2) as u32, block32], el(t * NH * ROT / 2), tg256());
        self.kit.d(enc, "t_rope", &[(&ar.k2, 0)], &[hd32, nkv32, rot32, 0, (t * NKV * ROT / 2) as u32, block32], el(t * NKV * ROT / 2), tg256());
        let causal = if self.bidir { 0u32 } else { 1u32 };
        let block32 = self.block as u32;
        self.kit.d(enc, "t_attn_fwd", &[(&ar.q2, 0), (&ar.k2, 0), (&ar.vfull, 0), (&ar.p, 0), (&ar.aog, 0)],
                   &[nh32, nkv32, hd32, t32, causal, block32], g2(NH.div_ceil(8), t.div_ceil(8)), MTLSize::new(8, 8, 1));
        self.kit.d(enc, "t_copy", &[(&ar.ao, 0), (&ar.aog, 0)], &[(t * QDIM) as u32], el(t * QDIM), tg256()); // ao = aog (dst,src; no output gate)
        self.kit.d(enc, "t_mm_xwT_h", &[(&ar.ao, 0), (self.wk(i, "wo"), 0), (&self.mix, 0)], &[QDIM as u32, du, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        self.fwd_ffn(enc, i, t, &ar.ffn);
    }

    fn bwd_attn(&self, enc: &metal::ComputeCommandEncoderRef, i: usize, t: usize) {
        let LayerA::A(ar) = &self.layers[i] else { unreachable!() };
        let (du, t32, hd32, nh32, nkv32, rot32) = (D as u32, t as u32, HD as u32, NH as u32, NKV as u32, ROT as u32);
        let (qd32, kvd32) = (QDIM as u32, KVDIM as u32);   // Qwen3 ungated: q_proj = QDIM
        let el = |n: usize| g1(n.div_ceil(256));
        self.bwd_ffn(enc, i, t, &ar.ffn);
        if self.layer_trains(i) {
        self.kit.d(enc, "t_mm_dw", &[(&self.dx1, 0), (&ar.ao, 0), (&self.dwbuf, 0)], &[QDIM as u32, du, t32], g2((QDIM).div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        }
        self.kit.d(enc, "t_mm_dx_h", &[(&self.dx1, 0), (self.wk(i, "wo"), 0), (&self.dao, 0)], &[QDIM as u32, du, 0, t32], g2(t.div_ceil(32), (QDIM).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.wo"), &self.dwbuf); }
        self.kit.d(enc, "t_copy", &[(&self.daog, 0), (&self.dao, 0)], &[(t * QDIM) as u32], el(t * QDIM), tg256()); // daog = dao (dst,src; no output gate)
        let causal = if self.bidir { 0u32 } else { 1u32 };
        let block32 = self.block as u32;
        self.kit.d(enc, "t_attn_dscore", &[(&ar.q2, 0), (&ar.k2, 0), (&ar.vfull, 0), (&ar.p, 0), (&self.daog, 0), (&self.ds_attn, 0), (&self.dq2, 0)],
                   &[nh32, nkv32, hd32, t32, causal, block32], g2(NH.div_ceil(8), t.div_ceil(8)), MTLSize::new(8, 8, 1));
        self.kit.d(enc, "t_attn_dkv", &[(&ar.q2, 0), (&ar.p, 0), (&self.ds_attn, 0), (&self.daog, 0), (&self.dk2, 0), (&self.dvv, 0)],
                   &[nh32, nkv32, hd32, t32, causal, block32], g2(t, NKV), tg256());
        self.kit.d(enc, "t_rope", &[(&self.dq2, 0)], &[hd32, nh32, rot32, 1, (t * NH * ROT / 2) as u32, block32], el(t * NH * ROT / 2), tg256());
        self.kit.d(enc, "t_rope", &[(&self.dk2, 0)], &[hd32, nkv32, rot32, 1, (t * NKV * ROT / 2) as u32, block32], el(t * NKV * ROT / 2), tg256());
        self.kit.d(enc, "t_rms_bwd", &[(&ar.q0, 0), (self.wk(i, "qnw"), 0), (&ar.rqn, 0), (&self.dq2, 0), (&self.dq0, 0)], &[hd32, (t * NH) as u32, 0], g1((t * NH).div_ceil(8)), tg256());
        if self.layer_trains(i) {
        self.kit.d(enc, "t_rms_dw", &[(&ar.q0, 0), (&ar.rqn, 0), (&self.dq2, 0), (&self.dwbuf, 0)], &[hd32, (t * NH) as u32], g1(1), tg256());
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.qnw"), &self.dwbuf); }
        }
        self.kit.d(enc, "t_rms_bwd", &[(&ar.kfull, 0), (self.wk(i, "knw"), 0), (&ar.rkn, 0), (&self.dk2, 0), (&self.dk0, 0)], &[hd32, (t * NKV) as u32, 0], g1((t * NKV).div_ceil(8)), tg256());
        if self.layer_trains(i) {
        self.kit.d(enc, "t_rms_dw", &[(&ar.kfull, 0), (&ar.rkn, 0), (&self.dk2, 0), (&self.dwbuf, 0)], &[hd32, (t * NKV) as u32], g1(1), tg256());
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.knw"), &self.dwbuf); }
        }
        self.kit.d(enc, "t_copy", &[(&self.dqfull, 0), (&self.dq0, 0)], &[(t * QDIM) as u32], el(t * QDIM), tg256()); // dqfull = dq0 (dst,src; no gate half)
        if self.layer_trains(i) {
        self.kit.d(enc, "t_mm_dw", &[(&self.dqfull, 0), (&ar.h, 0), (&self.dwbuf, 0)], &[du, qd32, t32], g2((D).div_ceil(32), (QDIM).div_ceil(64)), MTLSize::new(128, 1, 1));
        }
        self.kit.d(enc, "t_mm_dx_h", &[(&self.dqfull, 0), (self.wk(i, "wq"), 0), (&self.dh, 0)], &[du, qd32, 0, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.wq"), &self.dwbuf); }
        if self.layer_trains(i) {
        self.kit.d(enc, "t_mm_dw", &[(&self.dk0, 0), (&ar.h, 0), (&self.dwbuf, 0)], &[du, kvd32, t32], g2((D).div_ceil(32), (KVDIM).div_ceil(64)), MTLSize::new(128, 1, 1));
        }
        self.kit.d(enc, "t_mm_dx_h", &[(&self.dk0, 0), (self.wk(i, "wk"), 0), (&self.dh, 0)], &[du, kvd32, 1, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.wk"), &self.dwbuf); }
        if self.layer_trains(i) {
        self.kit.d(enc, "t_mm_dw", &[(&self.dvv, 0), (&ar.h, 0), (&self.dwbuf, 0)], &[du, kvd32, t32], g2((D).div_ceil(32), (KVDIM).div_ceil(64)), MTLSize::new(128, 1, 1));
        }
        self.kit.d(enc, "t_mm_dx_h", &[(&self.dvv, 0), (self.wk(i, "wv"), 0), (&self.dh, 0)], &[du, kvd32, 1, t32], g2(t.div_ceil(32), (D).div_ceil(64)), MTLSize::new(128, 1, 1));
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.wv"), &self.dwbuf); }
        self.kit.d(enc, "t_copy", &[(&self.dxb, 0), (&self.dx1, 0)], &[(t * D) as u32], el(t * D), tg256());
        self.kit.d(enc, "t_rms_bwd", &[(&self.x[i], 0), (self.wk(i, "ln"), 0), (&ar.rln, 0), (&self.dh, 0), (&self.dxb, 0)], &[du, t32, 1], g1(t.div_ceil(8)), tg256());
        if self.layer_trains(i) {
        self.kit.d(enc, "t_rms_dw", &[(&self.x[i], 0), (&ar.rln, 0), (&self.dh, 0), (&self.dwbuf, 0)], &[du, t32], g1(D.div_ceil(256)), tg256());
        if self.layer_trains(i) { self.opt_step(enc, &format!("L{i}.ln"), &self.dwbuf); }
        }
    }
}

/// HF tensor name → trainer weight key (None = skip, e.g. vision tower).
fn map_name(n: &str) -> Option<String> {
    if n == "lm_head.weight" { return Some("lm_head".into()); }
    let n = n.strip_prefix("model.")?;
    if n == "embed_tokens.weight" { return Some("embed".into()); }
    if n == "norm.weight" { return Some("final_norm".into()); }
    let rest = n.strip_prefix("layers.")?;
    let (i, rest) = rest.split_once('.')?;
    let key = match rest {
        "input_layernorm.weight" => "ln",
        "post_attention_layernorm.weight" => "pln",
        "linear_attn.in_proj_qkv.weight" => "qkv",
        "linear_attn.in_proj_z.weight" => "z",
        "linear_attn.in_proj_a.weight" => "wa",
        "linear_attn.in_proj_b.weight" => "wb",
        "linear_attn.dt_bias" => "dt",
        "linear_attn.A_log" => "alog",
        "linear_attn.conv1d.weight" => "cw",
        "linear_attn.norm.weight" => "nw",
        "linear_attn.out_proj.weight" => "wout",
        "self_attn.q_proj.weight" => "wq",
        "self_attn.k_proj.weight" => "wk",
        "self_attn.v_proj.weight" => "wv",
        "self_attn.o_proj.weight" => "wo",
        "self_attn.q_norm.weight" => "qnw",
        "self_attn.k_norm.weight" => "knw",
        "mlp.gate_proj.weight" => "wg",
        "mlp.up_proj.weight" => "wu",
        "mlp.down_proj.weight" => "wd",
        _ => return None,
    };
    Some(format!("L{i}.{key}"))
}

impl<'a> ojas_core::Learner for Trainer<'a> {
    fn step_core(&mut self, tokens: &[u32], x_in: Option<&[f32]>, targets: &[u32],
                 lo: usize, do_bwd: bool) -> anyhow::Result<(f32, usize)> {
        Trainer::step_core(self, tokens, x_in, targets, lo, do_bwd)
    }
    fn bwd_from(&mut self, lo: usize) -> anyhow::Result<()> { Trainer::bwd_from(self, lo) }
    fn fwd_span_range(&mut self, tokens: &[u32], x_in: Option<&[f32]>, lo: usize, hi: usize)
                      -> anyhow::Result<Vec<f32>> {
        Trainer::fwd_span_range(self, tokens, x_in, lo, hi)
    }
    fn save_ckpt(&self, path: &str) -> anyhow::Result<()> { Trainer::save_ckpt(self, path) }
    fn load_ckpt(&mut self, path: &str) -> anyhow::Result<()> { Trainer::load_ckpt(self, path) }
}
