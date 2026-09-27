//! Pure-CPU Whisper (speech-to-text) — encoder-decoder over log-mel features
//! (voice stack). Same shape as an encoder-decoder model (bidirectional encoder + causal
//! decoder with cross-attention, cross-K/V precomputed once per utterance),
//! with whisper's specifics: LayerNorm (mean+bias, eps 1e-5) not RMSNorm,
//! biased projections (k_proj bias-less), GELU MLPs, conv stem (k=3: s=1 then
//! s=2), sinusoidal encoder positions and learned decoder positions (both
//! stored in the GGUF), forced decode prefix
//! `<|startoftranscript|><|lang|><|transcribe|><|notimestamps|>`, greedy.
//!
//! The mel frontend lives here too: 16 kHz mono → 400-pt DFT (direct, with
//! precomputed twiddles — n_fft=400 is not radix-2 and 3000 frames × O(N²) is
//! ~0.5 s, acceptable here), hann window, slaney mel filterbank read from the
//! GGUF (computed by the converter in numpy, so there is no mismatch risk),
//! whisper log10 + max-8 clamp + (x+4)/4 normalization, padded/truncated to
//! 30 s (3000 frames).

use crate::cpu_math::{gelu, layernorm, matmul as mm, matvec as mv, W};
use ojas_formats::gguf::Gguf;
use anyhow::{bail, Result};
use std::cell::RefCell;
use std::sync::atomic::Ordering;

const N_FFT: usize = 400;
const HOP: usize = 160;
const CHUNK_FRAMES: usize = 3000; // 30 s @ 10 ms hop
/// Whisper's LayerNorm epsilon, passed explicitly because `cpu_math::layernorm`
/// is shared with the surya-2 ViT, which needs 1e-6.
const WHISPER_EPS: f32 = 1e-5;

struct LayerW {
    attn_ln: (Vec<f32>, Vec<f32>),
    q: W, q_b: Vec<f32>,
    k: W,
    v: W, v_b: Vec<f32>,
    o: W, o_b: Vec<f32>,
    // decoder-only cross attention
    x_ln: Option<(Vec<f32>, Vec<f32>)>,
    xq: Option<(W, Vec<f32>)>,
    xk: Option<W>,
    xv: Option<(W, Vec<f32>)>,
    xo: Option<(W, Vec<f32>)>,
    mlp_ln: (Vec<f32>, Vec<f32>),
    fc1: (W, Vec<f32>),
    fc2: (W, Vec<f32>),
}

pub struct CpuWhisper {
    pub d: usize,
    pub n_head: usize,
    pub hd: usize,
    pub n_enc: usize,
    pub n_dec: usize,
    pub n_mel: usize,
    pub n_audio_ctx: usize,
    pub n_text_ctx: usize,
    pub vocab: usize,
    // special tokens
    pub sot: u32,
    pub eot: u32,
    pub lang_en: u32,
    pub transcribe: u32,
    pub no_timestamps: u32,
    pub ts_begin: u32,
    mel_filters: Vec<f32>, // [n_mel × 201]
    conv1: (Vec<f32>, Vec<f32>), // [d][n_mel][3]
    conv2: (Vec<f32>, Vec<f32>), // [d][d][3]
    enc_pos: Vec<f32>,
    dec_pos: Vec<f32>,
    tok_embd: W,
    enc: Vec<LayerW>,
    dec: Vec<LayerW>,
    enc_ln: (Vec<f32>, Vec<f32>),
    dec_ln: (Vec<f32>, Vec<f32>),
    threads: usize,
    dotprod: bool,
    // per-utterance state
    xkv: RefCell<Vec<(Vec<f32>, Vec<f32>)>>, // cross K/V per dec layer [frames × d]
    kv: RefCell<(Vec<Vec<f32>>, Vec<Vec<f32>>)>, // decoder self K/V
    enc_frames: RefCell<usize>,
}

impl CpuWhisper {
    pub fn load(g: &mut Gguf) -> Result<CpuWhisper> {
        if g.arch() != "whisper" {
            bail!("CpuWhisper needs arch=whisper (got {})", g.arch());
        }
        let mu = |g: &Gguf, k: &str| g.meta_u32(&format!("whisper.{k}")).unwrap_or(0) as usize;
        let d = mu(g, "d_model");
        let n_enc = mu(g, "encoder_layers");
        let n_dec = mu(g, "decoder_layers");
        let n_head = mu(g, "attention.head_count");
        let n_mel = mu(g, "num_mel_bins");
        let n_audio_ctx = mu(g, "max_source_positions");
        let n_text_ctx = mu(g, "max_target_positions");
        let vocab = mu(g, "vocab_size");
        let tok = |g: &Gguf, k: &str| g.meta_u32(&format!("whisper.token.{k}")).unwrap_or(0);
        let (sot, eot) = (tok(g, "sot"), tok(g, "eot"));
        let (lang_en, transcribe) = (tok(g, "lang_en"), tok(g, "transcribe"));
        let (no_timestamps, ts_begin) = (tok(g, "no_timestamps"), tok(g, "ts_begin"));

        #[cfg(target_arch = "aarch64")]
        let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        #[cfg(not(target_arch = "aarch64"))]
        let dotprod = false;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        tracing::info!(target: "cpu:whisper", "d={d} enc={n_enc} dec={n_dec} heads={n_head} mel={n_mel} vocab={vocab} | q8 sdot={dotprod}");

        let readf = |g: &mut Gguf, n: &str| -> Result<Vec<f32>> {
            let (_d, ty, b) = g.read_tensor(n)?;
            if ty != 0 { bail!("{n}: expected f32 (type {ty})"); }
            Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        // big matmul weights → per-row q8 (SDOT); everything else stays f32
        let readw = |g: &mut Gguf, n: &str| -> Result<W> {
            let (dims, _ty, _b) = {
                let (dims, ty, b) = g.read_tensor(n)?;
                if ty != 0 { bail!("{n}: expected f32"); }
                (dims, ty, b)
            };
            let f = {
                let (_d2, _t2, b) = g.read_tensor(n)?;
                b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>()
            };
            if ojas_core::config::flag("OJAS_WHISPER_F32") {
                return Ok(W::F32(f)); // debug: exact reference vs GPU f16
            }
            let cols = dims.first().copied().unwrap_or(1) as usize;
            let rows = f.len() / cols.max(1);
            let mut q = Vec::with_capacity(f.len());
            let mut scale = Vec::with_capacity(rows);
            for r in 0..rows {
                let (qr, sc) = crate::cpu_math::quant_row_i8(&f[r * cols..(r + 1) * cols]);
                q.extend_from_slice(&qr);
                scale.push(sc);
            }
            Ok(W::Q8 { q, scale })
        };
        let ln = |g: &mut Gguf, n: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((readf(g, &format!("{n}.weight"))?, readf(g, &format!("{n}.bias"))?))
        };

        let load_layers = |g: &mut Gguf, pre: &str, n: usize, cross: bool| -> Result<Vec<LayerW>> {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let p = |s: &str| format!("{pre}.{i}.{s}");
                out.push(LayerW {
                    attn_ln: ln(g, &p("attn_ln"))?,
                    q: readw(g, &p("attn_q.weight"))?, q_b: readf(g, &p("attn_q.bias"))?,
                    k: readw(g, &p("attn_k.weight"))?,
                    v: readw(g, &p("attn_v.weight"))?, v_b: readf(g, &p("attn_v.bias"))?,
                    o: readw(g, &p("attn_o.weight"))?, o_b: readf(g, &p("attn_o.bias"))?,
                    x_ln: if cross { Some(ln(g, &p("xattn_ln"))?) } else { None },
                    xq: if cross { Some((readw(g, &p("xattn_q.weight"))?, readf(g, &p("xattn_q.bias"))?)) } else { None },
                    xk: if cross { Some(readw(g, &p("xattn_k.weight"))?) } else { None },
                    xv: if cross { Some((readw(g, &p("xattn_v.weight"))?, readf(g, &p("xattn_v.bias"))?)) } else { None },
                    xo: if cross { Some((readw(g, &p("xattn_o.weight"))?, readf(g, &p("xattn_o.bias"))?)) } else { None },
                    mlp_ln: ln(g, &p("mlp_ln"))?,
                    fc1: (readw(g, &p("fc1.weight"))?, readf(g, &p("fc1.bias"))?),
                    fc2: (readw(g, &p("fc2.weight"))?, readf(g, &p("fc2.bias"))?),
                });
            }
            Ok(out)
        };

        let m = CpuWhisper {
            hd: d / n_head.max(1),
            mel_filters: readf(g, "mel_filters")?,
            conv1: (readf(g, "enc.conv1.weight")?, readf(g, "enc.conv1.bias")?),
            conv2: (readf(g, "enc.conv2.weight")?, readf(g, "enc.conv2.bias")?),
            enc_pos: readf(g, "enc.pos")?,
            dec_pos: readf(g, "dec.pos")?,
            tok_embd: readw(g, "dec.tok_embd")?,
            enc: load_layers(g, "enc", n_enc, false)?,
            dec: load_layers(g, "dec", n_dec, true)?,
            enc_ln: ln(g, "enc_ln")?,
            dec_ln: ln(g, "dec_ln")?,
            d, n_head, n_enc, n_dec, n_mel, n_audio_ctx, n_text_ctx, vocab,
            sot, eot, lang_en, transcribe, no_timestamps, ts_begin,
            threads, dotprod,
            xkv: RefCell::new(Vec::new()),
            kv: RefCell::new((vec![Vec::new(); n_dec], vec![Vec::new(); n_dec])),
            enc_frames: RefCell::new(0),
        };
        Ok(m)
    }

    /// 16 kHz mono PCM → whisper log-mel, padded/truncated to 30 s.
    /// Returns (mel [frames × n_mel] frame-major, n_frames = CHUNK_FRAMES).
    pub fn log_mel(&self, pcm: &[f32]) -> Vec<f32> {
        let n_freq = N_FFT / 2 + 1;
        let mut samples = pcm.to_vec();
        samples.resize(CHUNK_FRAMES * HOP + N_FFT, 0.0); // whisper pads to 30 s
        // hann window + DFT twiddles (precomputed)
        let hann: Vec<f32> = (0..N_FFT)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos())
            .collect();
        let mut cos_t = vec![0f32; n_freq * N_FFT];
        let mut sin_t = vec![0f32; n_freq * N_FFT];
        for k in 0..n_freq {
            for n in 0..N_FFT {
                let a = 2.0 * std::f32::consts::PI * (k * n) as f32 / N_FFT as f32;
                cos_t[k * N_FFT + n] = a.cos();
                sin_t[k * N_FFT + n] = a.sin();
            }
        }
        // frames (reflect-pad start by n_fft/2 like torch.stft center=True)
        let mut mel = vec![0f32; CHUNK_FRAMES * self.n_mel];
        let nt = self.threads;
        let mel_addr = mel.as_mut_ptr() as usize;
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..nt {
                let (samples, hann, cos_t, sin_t) = (&samples, &hann, &cos_t, &sin_t);
                let next = &next;
                let filters = &self.mel_filters;
                let n_mel = self.n_mel;
                sc.spawn(move || {
                    let mut win = vec![0f32; N_FFT];
                    let mut power = vec![0f32; n_freq];
                    loop {
                        let f = next.fetch_add(1, Ordering::Relaxed);
                        if f >= CHUNK_FRAMES { break; }
                        for i in 0..N_FFT {
                            // center=True reflect padding
                            let idx = f as i64 * HOP as i64 + i as i64 - (N_FFT / 2) as i64;
                            let s = if idx < 0 { samples[(-idx) as usize] }
                                    else { samples[idx as usize] };
                            win[i] = s * hann[i];
                        }
                        for k in 0..n_freq {
                            let (mut re, mut im) = (0f32, 0f32);
                            let (ct, st) = (&cos_t[k * N_FFT..(k + 1) * N_FFT], &sin_t[k * N_FFT..(k + 1) * N_FFT]);
                            for n in 0..N_FFT {
                                re += win[n] * ct[n];
                                im -= win[n] * st[n];
                            }
                            power[k] = re * re + im * im;
                        }
                        for m in 0..n_mel {
                            let fr = &filters[m * n_freq..(m + 1) * n_freq];
                            let mut acc = 0f32;
                            for k in 0..n_freq { acc += fr[k] * power[k]; }
                            // SAFETY: each frame written by exactly one thread
                            unsafe { *(mel_addr as *mut f32).add(f * n_mel + m) = acc.max(1e-10).log10(); }
                        }
                    }
                });
            }
        });
        // whisper normalization: clamp to max-8, then (x+4)/4
        let mx = mel.iter().cloned().fold(f32::MIN, f32::max);
        for v in mel.iter_mut() { *v = (v.max(mx - 8.0) + 4.0) / 4.0; }
        mel
    }

    /// Encode 30 s of log-mel; fills the per-layer cross K/V.
    pub fn encode(&self, mel: &[f32]) {
        let x = self.encoder_forward(mel);
        self.attach_encoded(&x);
    }

    /// Conv stem + sinusoidal positions (CPU, threaded):
    /// mel [3000 × n_mel] → [1500 × d]. Public so a GPU encoder can reuse it.
    pub fn conv_stem(&self, mel: &[f32]) -> Vec<f32> {
        let d = self.d;
        let n_mel = self.n_mel; // hoisted: par_rows closures must not capture self
        // conv stem: mel [3000 × n_mel] → conv1(k3,s1)+gelu → conv2(k3,s2)+gelu → [1500 × d]
        let frames = CHUNK_FRAMES;
        let get_mel = move |t: i64, c: usize| -> f32 {
            if t < 0 || t >= frames as i64 { 0.0 } else { mel[t as usize * n_mel + c] }
        };
        let mut h1 = vec![0f32; frames * d];
        let (c1w, c1b) = (&self.conv1.0, &self.conv1.1);
        par_rows(frames, self.threads, |t| {
            let mut row = vec![0f32; d];
            for oc in 0..d {
                let mut acc = c1b[oc];
                for ic in 0..n_mel {
                    let wbase = oc * n_mel * 3 + ic * 3;
                    for kk in 0..3 {
                        acc += c1w[wbase + kk] * get_mel(t as i64 + kk as i64 - 1, ic);
                    }
                }
                row[oc] = gelu(acc);
            }
            (t * d, row)
        }, &mut h1);
        let out_frames = frames / 2;
        let mut x = vec![0f32; out_frames * d];
        let (c2w, c2b) = (&self.conv2.0, &self.conv2.1);
        let h1_ref = &h1;
        par_rows(out_frames, self.threads, |t| {
            let mut row = vec![0f32; d];
            for oc in 0..d {
                let mut acc = c2b[oc];
                for kk in 0..3 {
                    let src = (t * 2 + kk) as i64 - 1;
                    if src < 0 || src >= frames as i64 { continue; }
                    let hrow = &h1_ref[src as usize * d..(src as usize + 1) * d];
                    let wbase = oc * d * 3;
                    for ic in 0..d { acc += c2w[wbase + ic * 3 + kk] * hrow[ic]; }
                }
                row[oc] = gelu(acc);
            }
            (t * d, row)
        }, &mut x);
        // + sinusoidal positions
        for t in 0..out_frames {
            for i in 0..d { x[t * d + i] += self.enc_pos[t * d + i]; }
        }
        x
    }

    /// Full CPU encoder: mel → final LN'd frames [out_frames × d].
    pub fn encoder_forward(&self, mel: &[f32]) -> Vec<f32> {
        let (d, hd, nh) = (self.d, self.hd, self.n_head);
        let mut x = self.conv_stem(mel);
        let out_frames = x.len() / d;
        // transformer encoder (pre-LN, bidirectional)
        let scale = 1.0 / (hd as f32).sqrt();
        let n_layers = ojas_core::config::var("OJAS_WHISPER_LAYERS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(self.enc.len());
        for ly in self.enc.iter().take(n_layers) {
            let hs: Vec<Vec<f32>> = (0..out_frames)
                .map(|t| layernorm(&x[t * d..(t + 1) * d], &ly.attn_ln.0, &ly.attn_ln.1, WHISPER_EPS))
                .collect();
            let hrefs: Vec<&[f32]> = hs.iter().map(|h| h.as_slice()).collect();
            let mut q = vec![vec![0f32; d]; out_frames];
            let mut k = vec![vec![0f32; d]; out_frames];
            let mut v = vec![vec![0f32; d]; out_frames];
            mm(&ly.q, d, d, &hrefs, Some(&ly.q_b), &mut q, self.threads, self.dotprod);
            mm(&ly.k, d, d, &hrefs, None, &mut k, self.threads, self.dotprod);
            mm(&ly.v, d, d, &hrefs, Some(&ly.v_b), &mut v, self.threads, self.dotprod);
            // full bidirectional attention, parallel over frames
            let mut attn = vec![0f32; out_frames * d];
            let (qr, kr, vr) = (&q, &k, &v);
            par_rows(out_frames, self.threads, |t| {
                let mut row = vec![0f32; d];
                for hh in 0..nh {
                    let qh = &qr[t][hh * hd..(hh + 1) * hd];
                    let mut sc: Vec<f32> = (0..out_frames)
                        .map(|u| crate::cpu_math::dot_f32(qh, &kr[u][hh * hd..(hh + 1) * hd]) * scale)
                        .collect();
                    let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                    let mut den = 0.0;
                    for s in sc.iter_mut() { *s = (*s - mx).exp(); den += *s; }
                    for i in 0..hd {
                        let mut acc = 0.0;
                        for (u, s) in sc.iter().enumerate() { acc += s * vr[u][hh * hd + i]; }
                        row[hh * hd + i] = acc / den;
                    }
                }
                (t * d, row)
            }, &mut attn);
            let arefs: Vec<&[f32]> = (0..out_frames).map(|t| &attn[t * d..(t + 1) * d]).collect();
            let mut o = vec![vec![0f32; d]; out_frames];
            mm(&ly.o, d, d, &arefs, Some(&ly.o_b), &mut o, self.threads, self.dotprod);
            for t in 0..out_frames { for i in 0..d { x[t * d + i] += o[t][i]; } }
            // MLP
            let h2: Vec<Vec<f32>> = (0..out_frames)
                .map(|t| layernorm(&x[t * d..(t + 1) * d], &ly.mlp_ln.0, &ly.mlp_ln.1, WHISPER_EPS))
                .collect();
            let h2r: Vec<&[f32]> = h2.iter().map(|h| h.as_slice()).collect();
            let ffn = ly.fc1.1.len();
            let mut a = vec![vec![0f32; ffn]; out_frames];
            mm(&ly.fc1.0, ffn, d, &h2r, Some(&ly.fc1.1), &mut a, self.threads, self.dotprod);
            for row in a.iter_mut() { for v in row.iter_mut() { *v = gelu(*v); } }
            let ar: Vec<&[f32]> = a.iter().map(|r| r.as_slice()).collect();
            let mut f2 = vec![vec![0f32; d]; out_frames];
            mm(&ly.fc2.0, d, ffn, &ar, Some(&ly.fc2.1), &mut f2, self.threads, self.dotprod);
            for t in 0..out_frames { for i in 0..d { x[t * d + i] += f2[t][i]; } }
        }
        for t in 0..out_frames {
            let n = layernorm(&x[t * d..(t + 1) * d], &self.enc_ln.0, &self.enc_ln.1, WHISPER_EPS);
            x[t * d..(t + 1) * d].copy_from_slice(&n);
        }
        x
    }

    /// Precompute the decoder's cross K/V from encoder output frames
    /// [frames × d] and reset the decode state.
    pub fn attach_encoded(&self, x: &[f32]) {
        let d = self.d;
        let out_frames = x.len() / d;
        let xr: Vec<&[f32]> = (0..out_frames).map(|t| &x[t * d..(t + 1) * d]).collect();
        let mut xkv = self.xkv.borrow_mut();
        xkv.clear();
        for ly in &self.dec {
            let mut k = vec![vec![0f32; d]; out_frames];
            let mut v = vec![vec![0f32; d]; out_frames];
            mm(ly.xk.as_ref().unwrap(), d, d, &xr, None, &mut k, self.threads, self.dotprod);
            let (xv, xv_b) = ly.xv.as_ref().unwrap();
            mm(xv, d, d, &xr, Some(xv_b), &mut v, self.threads, self.dotprod);
            xkv.push((k.concat(), v.concat()));
        }
        *self.enc_frames.borrow_mut() = out_frames;
        let (kc, vc) = &mut *self.kv.borrow_mut();
        for l in 0..self.n_dec { kc[l].clear(); vc[l].clear(); }
    }

    /// One greedy decode step (suppresses timestamp tokens; v1 has
    /// <|notimestamps|> in the prefix anyway).
    pub fn decode_step(&self, tok: u32, pos: usize) -> u32 {
        let (d, hd, nh) = (self.d, self.hd, self.n_head);
        let frames = *self.enc_frames.borrow();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut x: Vec<f32> = match &self.tok_embd {
            W::Q8 { q, scale } => q[tok as usize * d..(tok as usize + 1) * d].iter()
                .map(|&b| b as f32 * scale[tok as usize]).collect(),
            _ => unreachable!(),
        };
        for i in 0..d { x[i] += self.dec_pos[pos * d + i]; }
        let (kc, vc) = &mut *self.kv.borrow_mut();
        let xkv = self.xkv.borrow();
        for (l, ly) in self.dec.iter().enumerate() {
            // self attention (causal via cache)
            let h = layernorm(&x, &ly.attn_ln.0, &ly.attn_ln.1, WHISPER_EPS);
            let q = mv(&ly.q, d, d, &h, Some(&ly.q_b), self.threads, self.dotprod);
            let k = mv(&ly.k, d, d, &h, None, self.threads, self.dotprod);
            let v = mv(&ly.v, d, d, &h, Some(&ly.v_b), self.threads, self.dotprod);
            kc[l].extend_from_slice(&k);
            vc[l].extend_from_slice(&v);
            let seq = kc[l].len() / d;
            let mut ao = vec![0f32; d];
            for hh in 0..nh {
                let qh = &q[hh * hd..(hh + 1) * hd];
                let mut sc: Vec<f32> = (0..seq)
                    .map(|u| crate::cpu_math::dot_f32(qh, &kc[l][u * d + hh * hd..u * d + (hh + 1) * hd]) * scale)
                    .collect();
                let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                let mut den = 0.0;
                for s in sc.iter_mut() { *s = (*s - mx).exp(); den += *s; }
                for i in 0..hd {
                    let mut acc = 0.0;
                    for (u, s) in sc.iter().enumerate() { acc += s * vc[l][u * d + hh * hd + i]; }
                    ao[hh * hd + i] = acc / den;
                }
            }
            let o = mv(&ly.o, d, d, &ao, Some(&ly.o_b), self.threads, self.dotprod);
            for i in 0..d { x[i] += o[i]; }
            // cross attention over encoder frames
            let hx = layernorm(&x, &ly.x_ln.as_ref().unwrap().0, &ly.x_ln.as_ref().unwrap().1, WHISPER_EPS);
            let (xqw, xqb) = ly.xq.as_ref().unwrap();
            let q = mv(xqw, d, d, &hx, Some(xqb), self.threads, self.dotprod);
            let (ck, cv) = &xkv[l];
            let mut ao = vec![0f32; d];
            for hh in 0..nh {
                let qh = &q[hh * hd..(hh + 1) * hd];
                let mut sc: Vec<f32> = (0..frames)
                    .map(|u| crate::cpu_math::dot_f32(qh, &ck[u * d + hh * hd..u * d + (hh + 1) * hd]) * scale)
                    .collect();
                let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                let mut den = 0.0;
                for s in sc.iter_mut() { *s = (*s - mx).exp(); den += *s; }
                for i in 0..hd {
                    let mut acc = 0.0;
                    for (u, s) in sc.iter().enumerate() { acc += s * cv[u * d + hh * hd + i]; }
                    ao[hh * hd + i] = acc / den;
                }
            }
            let (xow, xob) = ly.xo.as_ref().unwrap();
            let o = mv(xow, d, d, &ao, Some(xob), self.threads, self.dotprod);
            for i in 0..d { x[i] += o[i]; }
            // MLP
            let h2 = layernorm(&x, &ly.mlp_ln.0, &ly.mlp_ln.1, WHISPER_EPS);
            let ffn = ly.fc1.1.len();
            let a = mv(&ly.fc1.0, ffn, d, &h2, Some(&ly.fc1.1), self.threads, self.dotprod);
            let a: Vec<f32> = a.into_iter().map(gelu).collect();
            let f2 = mv(&ly.fc2.0, d, ffn, &a, Some(&ly.fc2.1), self.threads, self.dotprod);
            for i in 0..d { x[i] += f2[i]; }
        }
        let xn = layernorm(&x, &self.dec_ln.0, &self.dec_ln.1, WHISPER_EPS);
        let mut logits = mv(&self.tok_embd, self.vocab, d, &xn, None, self.threads, self.dotprod);
        // suppress timestamps + sot-class specials that derail greedy decode.
        // Guard: a missing whisper.token.ts_begin resolves to 0 (meta unwrap_or(0)),
        // which would mask the entire vocab → argmax collapses to token 0. Only
        // suppress a valid timestamp range.
        let ts0 = self.ts_begin as usize;
        if ts0 > 0 && ts0 < self.vocab {
            for t in ts0..self.vocab { logits[t] = f32::MIN; }
        }
        for t in [self.sot, self.lang_en, self.transcribe, self.no_timestamps] {
            logits[t as usize] = f32::MIN;
        }
        crate::cpu_math::argmax(&logits) as u32
    }

    /// Full greedy transcription of 16 kHz mono PCM: standard chunked decode.
    /// Each 30 s window is encoded and decoded independently (fresh token
    /// budget per window) and the transcripts concatenate in order, so audio
    /// past the first window is not dropped.
    pub fn transcribe(&self, pcm: &[f32]) -> Vec<u32> {
        const WINDOW: usize = CHUNK_FRAMES * HOP; // 30 s of samples
        let mut out = Vec::new();
        let mut start = 0usize;
        loop {
            let end = (start + WINDOW).min(pcm.len());
            let mel = self.log_mel(&pcm[start..end]);
            self.encode(&mel);
            out.extend(self.decode_transcript());
            if end >= pcm.len() {
                break;
            }
            start = end;
        }
        out
    }

    /// Greedy decode after `encode`/`attach_encoded` (public so a GPU
    /// encoder can drive the same decoder).
    pub fn decode_transcript(&self) -> Vec<u32> {
        let prefix = [self.sot, self.lang_en, self.transcribe, self.no_timestamps];
        let mut pos = 0usize;
        let mut last = 0u32;
        for &t in &prefix {
            last = self.decode_step(t, pos);
            pos += 1;
        }
        let mut out = Vec::new();
        while pos < self.n_text_ctx && out.len() < 224 {
            if last == self.eot { break; }
            out.push(last);
            last = self.decode_step(last, pos);
            pos += 1;
        }
        out
    }
}

/// Parallel map over row indices writing disjoint output spans.
fn par_rows(n: usize, threads: usize, f: impl Fn(usize) -> (usize, Vec<f32>) + Sync, out: &mut [f32]) {
    let out_addr = out.as_mut_ptr() as usize;
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            let next = &next;
            let f = &f;
            sc.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n { break; }
                let (off, row) = f(i);
                // SAFETY: each row index is claimed by exactly one thread and
                // rows write disjoint spans.
                unsafe {
                    std::ptr::copy_nonoverlapping(row.as_ptr(), (out_addr as *mut f32).add(off), row.len());
                }
            });
        }
    });
}
