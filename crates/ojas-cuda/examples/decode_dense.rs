//! First end-to-end decode on CUDA: a dense Qwen/Llama GGUF, one token at a time, every launch
//! through `KernelRuntime::dispatch`.
//!
//! Not a speed path. `ojas-models` cannot target CUDA (21 of its files call `metal::`
//! directly), so this shows the kernel seam carries a real model before that refactor: GGUF in,
//! logits out, checked against `ojas-cpu`'s decoder on the same weights.
//!
//! Weights are read once, dequantised to f32 and re-packed into the row-quantized Q8 layout the
//! `gemv_q8` family reads (K int8 values per row, one f32 scale per row), which keeps this path
//! to one weight format.
//!
//! The graph is complete: NeoX RoPE on q and k, an f32 KV cache appended per position, and GQA
//! attention over everything cached so far (`attention_short`). It lacks prefill batching, the
//! q/k norm Qwen3 adds, and every non-Q8 weight format.
//!
//! `cargo run --release -p ojas-cuda --example decode_dense -- model.gguf [n_tokens]`
use anyhow::{anyhow, Result};
use ojas_core::{Device, KernelRuntime};
use ojas_cuda::{CuBuf, CudaGpu};
use ojas_formats::gguf::Gguf;

/// A weight matrix in the layout `gemv_q8` reads: `rows * k` int8, plus one f32 scale per row.
struct Q8 {
    w: CuBuf,
    scale: CuBuf,
    rows: usize,
    k: usize,
}

/// f32 rows → int8 + per-row scale (absmax). The decoder's own requant path does the same thing.
fn pack_q8(gpu: &CudaGpu, data: &[f32], rows: usize, k: usize) -> Result<Q8> {
    let mut q = vec![0u8; rows * k];
    let mut scale = vec![0f32; rows];
    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        scale[r] = s;
        for (i, v) in row.iter().enumerate() {
            q[r * k + i] = ((v / s).round().clamp(-127.0, 127.0) as i8) as u8;
        }
    }
    Ok(Q8 { w: gpu.upload_bytes(&q)?, scale: gpu.upload(&scale), rows, k })
}

/// Read a tensor as f32 regardless of how it is stored. `read_tensor` already decodes every
/// block-quant type to F16, so no block format is re-implemented here.
fn tensor_f32(g: &mut Gguf, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let (dims, ty, bytes) = g.read_tensor(name)?;
    let dims: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
    let vals: Vec<f32> = match ty {
        0 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        1 => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        other => return Err(anyhow!("tensor '{name}': unexpected type {other} after read_tensor")),
    };
    Ok((vals, dims))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or_else(|| anyhow!("usage: decode_dense <model.gguf> [n_tokens]"))?;
    let steps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(8);

    let check = std::env::args().any(|a| a == "--check");
    let mut g = Gguf::open(path)?;
    let arch = g.arch();
    let n_layer = g.meta_u32(&format!("{arch}.block_count")).ok_or_else(|| anyhow!("no block_count"))? as usize;
    let d = g.meta_u32(&format!("{arch}.embedding_length")).ok_or_else(|| anyhow!("no embedding_length"))? as usize;
    let n_head = g.meta_u32(&format!("{arch}.attention.head_count")).unwrap_or(1) as usize;
    let n_kv = g.meta_u32(&format!("{arch}.attention.head_count_kv")).unwrap_or(n_head as u32) as usize;
    let ffn = g.meta_u32(&format!("{arch}.feed_forward_length")).unwrap_or((4 * d) as u32) as usize;
    let eps = g.meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
    let hd = d / n_head.max(1);
    println!("{arch}: {n_layer} layers, d={d}, heads={n_head}/{n_kv}, hd={hd}, ffn={ffn}, eps={eps}");

    let mut gpu = CudaGpu::new(0)?;
    for fam in ["ops", "gemv_q8", "attn", "attn_core"] {
        gpu.ensure_family(fam)?;
    }
    println!("device: {:?}, families ready", gpu.caps().tier);

    // ---- weights: embedding table and, per layer, the seven matrices a dense block needs
    let (embd, embd_dims) = tensor_f32(&mut g, "token_embd.weight")?;
    let vocab = embd_dims.last().copied().unwrap_or(0);
    println!("vocab {vocab}, embedding table {} MB f32", embd.len() * 4 / 1_000_000);
    let embd_q8 = pack_q8(&gpu, &embd, vocab, d)?;

    struct Layer {
        attn_norm: CuBuf,
        ffn_norm: CuBuf,
        bias: Option<(CuBuf, CuBuf, CuBuf)>,
        wq: Q8,
        wk: Q8,
        wv: Q8,
        wo: Q8,
        wg: Q8,
        wu: Q8,
        wd: Q8,
    }
    // host copies kept only when --check is on: the reference recomputes step 0 from these
    let mut host_w: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut layers = Vec::with_capacity(n_layer);
    for l in 0..n_layer {
        macro_rules! t { ($n:expr) => { tensor_f32(&mut g, &format!("blk.{l}.{}", $n))? }; }
        let (an, _) = t!("attn_norm.weight");
        let (fn_, _) = t!("ffn_norm.weight");
        let (q, qd) = t!("attn_q.weight");
        let (k, kd) = t!("attn_k.weight");
        let (v, vd) = t!("attn_v.weight");
        let (o, od) = t!("attn_output.weight");
        let (gt, gd) = t!("ffn_gate.weight");
        let (u, ud) = t!("ffn_up.weight");
        let (dn, dd) = t!("ffn_down.weight");
        // GGUF stores [in, out]; a row of our gemv is one output
        let rows = |dims: &[usize], len: usize| len / dims[0];
        // Qwen2 carries q/k/v bias; Qwen3, Gemma and Llama do not. A missing bias is fine;
        // dropping one that exists sends the logits to noise.
        let mut bias_host = (Vec::new(), Vec::new(), Vec::new());
        let bias = match (tensor_f32(&mut g, &format!("blk.{l}.attn_q.bias")),
                          tensor_f32(&mut g, &format!("blk.{l}.attn_k.bias")),
                          tensor_f32(&mut g, &format!("blk.{l}.attn_v.bias"))) {
            (Ok((bq, _)), Ok((bk, _)), Ok((bv, _))) => {
                if check { bias_host = (bq.clone(), bk.clone(), bv.clone()); }
                Some((gpu.upload(&bq), gpu.upload(&bk), gpu.upload(&bv)))
            }
            _ => None,
        };
        if check {
            host_w.push(vec![an.clone(), fn_.clone(), q.clone(), k.clone(), v.clone(), o.clone(),
                             gt.clone(), u.clone(), dn.clone(),
                             bias_host.0.clone(), bias_host.1.clone(), bias_host.2.clone()]);
        }
        layers.push(Layer {
            attn_norm: gpu.upload(&an),
            ffn_norm: gpu.upload(&fn_),
            bias,
            wq: pack_q8(&gpu, &q, rows(&qd, q.len()), qd[0])?,
            wk: pack_q8(&gpu, &k, rows(&kd, k.len()), kd[0])?,
            wv: pack_q8(&gpu, &v, rows(&vd, v.len()), vd[0])?,
            wo: pack_q8(&gpu, &o, rows(&od, o.len()), od[0])?,
            wg: pack_q8(&gpu, &gt, rows(&gd, gt.len()), gd[0])?,
            wu: pack_q8(&gpu, &u, rows(&ud, u.len()), ud[0])?,
            wd: pack_q8(&gpu, &dn, rows(&dd, dn.len()), dd[0])?,
        });
        if l == 0 {
            println!("layer 0: q {}x{}, ffn_gate {}x{}, qkv bias {}", layers[0].wq.rows,
                     layers[0].wq.k, layers[0].wg.rows, layers[0].wg.k,
                     if layers[0].bias.is_some() { "yes" } else { "no" });
        }
    }
    let (out_norm, _) = tensor_f32(&mut g, "output_norm.weight")?;
    let out_norm_d = gpu.upload(&out_norm);
    // An untied head when the file has one; otherwise the lm_head is the embedding table (the
    // common case for the small Qwens), packed a second time so the embed step keeps its copy.
    let head = match tensor_f32(&mut g, "output.weight") {
        Ok((w, dims)) => pack_q8(&gpu, &w, w.len() / dims[0], dims[0])?,
        Err(_) => {
            println!("tied lm_head (same weights as token_embd)");
            pack_q8(&gpu, &embd, vocab, d)?
        }
    };
    println!("weights on device: {n_layer} layers + head");

    // ---- scratch
    let x = gpu.alloc(d);
    let xb = gpu.alloc(d);
    let q = gpu.alloc(n_head * hd);
    let k = gpu.alloc(n_kv * hd);
    let v = gpu.alloc(n_kv * hd);
    let att = gpu.alloc(n_head * hd);
    let hb = gpu.alloc(ffn);
    let logits = gpu.alloc(vocab);
    let argd = gpu.alloc(1);
    // f32 KV cache per layer (Metal keeps f16 to halve the bandwidth; this is bring-up)
    let max_seq = 64usize;
    let kvdim = n_kv * hd;
    let kcache: Vec<CuBuf> = (0..n_layer).map(|_| gpu.alloc(max_seq * kvdim)).collect();
    let vcache: Vec<CuBuf> = (0..n_layer).map(|_| gpu.alloc(max_seq * kvdim)).collect();
    let rope_base = g.meta_f32(&format!("{arch}.rope.freq_base")).unwrap_or(10000.0);
    let scale = 1.0f32 / (hd as f32).sqrt();

    let warps = 8u32;
    let rows_grid = |n: usize| [(n as u32).div_ceil(warps), 1, 1];

    // ---- one decode step per token, greedy
    let mut token = 1u32;
    let (mut first_arg, mut first_logit) = (0usize, 0f32);
    let t0 = std::time::Instant::now();
    for step in 0..steps {
        let enc = gpu.begin();
        gpu.dispatch(&enc, "embed_q8", &[(&embd_q8.w, 0), (&embd_q8.scale, 0), (&x, 0)],
                     &[token, d as u32], [8, 1, 1], [128, 1, 1])?;
        for (li, l) in layers.iter().enumerate() {
            gpu.dispatch(&enc, "rmsnorm", &[(&x, 0), (&l.attn_norm, 0), (&xb, 0)],
                         &[d as u32, eps.to_bits()], [1, 1, 1], [256, 1, 1])?;
            match &l.bias {
                Some((bq, bk, bv)) => {
                    for (w, out, b) in [(&l.wq, &q, bq), (&l.wk, &k, bk), (&l.wv, &v, bv)] {
                        gpu.dispatch(&enc, "gemv_q8_bias",
                                     &[(&xb, 0), (&w.w, 0), (out, 0), (&w.scale, 0), (b, 0)],
                                     &[w.k as u32, w.rows as u32], rows_grid(w.rows), [256, 1, 1])?;
                    }
                }
                None => {
                    for (w, out) in [(&l.wq, &q), (&l.wk, &k), (&l.wv, &v)] {
                        gpu.dispatch(&enc, "gemv_q8", &[(&xb, 0), (&w.w, 0), (out, 0), (&w.scale, 0)],
                                     &[w.k as u32, w.rows as u32], rows_grid(w.rows), [256, 1, 1])?;
                    }
                }
            }
            // NeoX RoPE on q and k at this position
            let q_pairs = (n_head * hd / 2) as u32;
            let k_pairs = (n_kv * hd / 2) as u32;
            gpu.dispatch(&enc, "rope", &[(&q, 0)],
                         &[hd as u32, step as u32, rope_base.to_bits(), q_pairs],
                         [q_pairs.div_ceil(128), 1, 1], [128, 1, 1])?;
            gpu.dispatch(&enc, "rope", &[(&k, 0)],
                         &[hd as u32, step as u32, rope_base.to_bits(), k_pairs],
                         [k_pairs.div_ceil(128), 1, 1], [128, 1, 1])?;
            // append this position to the cache, then attend over everything cached
            let off = (step * kvdim) as u32;
            gpu.dispatch(&enc, "store_kv", &[(&kcache[li], 0), (&k, 0)], &[kvdim as u32, off],
                         [(kvdim as u32).div_ceil(256), 1, 1], [256, 1, 1])?;
            gpu.dispatch(&enc, "store_kv", &[(&vcache[li], 0), (&v, 0)], &[kvdim as u32, off],
                         [(kvdim as u32).div_ceil(256), 1, 1], [256, 1, 1])?;
            gpu.dispatch(&enc, "attention_short",
                         &[(&q, 0), (&kcache[li], 0), (&vcache[li], 0), (&att, 0)],
                         &[hd as u32, kvdim as u32, (step + 1) as u32,
                           (n_head / n_kv.max(1)) as u32, scale.to_bits()],
                         [n_head as u32, 1, 1], [256, 1, 1])?;
            gpu.dispatch(&enc, "gemv_q8_accum", &[(&att, 0), (&l.wo.w, 0), (&x, 0), (&l.wo.scale, 0)],
                         &[l.wo.k as u32, l.wo.rows as u32], rows_grid(l.wo.rows), [256, 1, 1])?;
            gpu.dispatch(&enc, "rmsnorm", &[(&x, 0), (&l.ffn_norm, 0), (&xb, 0)],
                         &[d as u32, eps.to_bits()], [1, 1, 1], [256, 1, 1])?;
            gpu.dispatch(&enc, "ffn_gu_q8",
                         &[(&xb, 0), (&l.wg.w, 0), (&l.wu.w, 0), (&hb, 0), (&l.wg.scale, 0), (&l.wu.scale, 0)],
                         &[l.wg.k as u32, l.wg.rows as u32, 0], rows_grid(l.wg.rows), [256, 1, 1])?;
            gpu.dispatch(&enc, "gemv_q8_accum", &[(&hb, 0), (&l.wd.w, 0), (&x, 0), (&l.wd.scale, 0)],
                         &[l.wd.k as u32, l.wd.rows as u32], rows_grid(l.wd.rows), [256, 1, 1])?;
        }
        gpu.dispatch(&enc, "rmsnorm", &[(&x, 0), (&out_norm_d, 0), (&xb, 0)],
                     &[d as u32, eps.to_bits()], [1, 1, 1], [256, 1, 1])?;
        gpu.dispatch(&enc, "gemv_q8", &[(&xb, 0), (&head.w, 0), (&logits, 0), (&head.scale, 0)],
                     &[head.k as u32, head.rows as u32], rows_grid(head.rows), [256, 1, 1])?;
        // argmax on the device: pulling 151,936 floats per token back to the host would measure
        // PCIe rather than the model. `--check` still reads the full vector for the reference.
        gpu.dispatch(&enc, "argmax", &[(&logits, 0), (&argd, 0)], &[vocab as u32],
                     [1, 1, 1], [256, 1, 1])?;
        gpu.submit(enc)?;

        let mut idx_raw = vec![0u8; 4];
        gpu.read_bytes(&argd, 0, &mut idx_raw)?;
        let best = u32::from_le_bytes([idx_raw[0], idx_raw[1], idx_raw[2], idx_raw[3]]) as usize;
        if check || step == 0 {
            let mut host = vec![0.0f32; vocab];
            gpu.read(&logits, &mut host);
            let finite = host.iter().filter(|v| v.is_finite()).count();
            println!("step {step}: token {token} -> argmax {best} (logit {:.4}), {finite}/{vocab} finite",
                     host[best]);
            if step == 0 { first_logit = host[best]; }
        } else if step < 4 || step + 1 == steps {
            println!("step {step}: token {token} -> argmax {best}");
        }
        if step == 0 { first_arg = best; }
        token = best as u32;
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("{steps} steps in {:.1} ms = {:.2} ms/token, {:.1} tok/s",
             secs * 1000.0, secs * 1000.0 / steps as f64, steps as f64 / secs);

    if check {
        // Step 0 on the CPU from the same f32 weights. At position 0 RoPE is the identity and
        // attention over a single key reduces to v, so the reference needs no cache and checks
        // embed, rmsnorm, qkv+bias, o_proj, ffn and the head.
        let rms = |v: &[f32], w: &[f32]| -> Vec<f32> {
            let ss: f32 = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
            let inv = 1.0 / (ss + eps).sqrt();
            v.iter().zip(w).map(|(x, g)| x * inv * g).collect()
        };
        // The GPU runs Q8-packed weights, so the reference must too, or the comparison measures
        // quantisation error instead of kernel correctness. Same absmax rounding as pack_q8,
        // then back to f32.
        let q8_roundtrip = |w: &[f32], rows: usize, k: usize| -> Vec<f32> {
            let mut out = vec![0f32; rows * k];
            for r in 0..rows {
                let row = &w[r * k..(r + 1) * k];
                let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
                let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                for (i, v) in row.iter().enumerate() {
                    out[r * k + i] = (v / sc).round().clamp(-127.0, 127.0) * sc;
                }
            }
            out
        };
        let matvec = |w: &[f32], x: &[f32], rows: usize, bias: &[f32]| -> Vec<f32> {
            (0..rows).map(|r| {
                let acc: f32 = w[r * x.len()..(r + 1) * x.len()].iter().zip(x).map(|(a, b)| a * b).sum();
                acc + bias.get(r).copied().unwrap_or(0.0)
            }).collect()
        };
        let (embd_host_f32, _) = tensor_f32(&mut g, "token_embd.weight")?;
        let embd_host = q8_roundtrip(&embd_host_f32, vocab, d);
        let mut xh: Vec<f32> = embd_host[d..2 * d].to_vec();   // token 1
        let group = n_head / n_kv.max(1);
        for (li, l) in layers.iter().enumerate() {
            let w = &host_w[li];
            let xb = rms(&xh, &w[0]);
            let vv = matvec(&q8_roundtrip(&w[4], l.wv.rows, w[4].len() / l.wv.rows), &xb, l.wv.rows, &w[11]);
            // attention at position 0 == v, routed through GQA
            let att: Vec<f32> = (0..n_head).flat_map(|h| {
                let src = (h / group) * hd;
                vv[src..src + hd].to_vec()
            }).collect();
            let o = matvec(&q8_roundtrip(&w[5], l.wo.rows, w[5].len() / l.wo.rows), &att, l.wo.rows, &[]);
            for (a, b) in xh.iter_mut().zip(&o) { *a += b; }
            let xb2 = rms(&xh, &w[1]);
            let gate = matvec(&q8_roundtrip(&w[6], l.wg.rows, w[6].len() / l.wg.rows), &xb2, l.wg.rows, &[]);
            let up = matvec(&q8_roundtrip(&w[7], l.wu.rows, w[7].len() / l.wu.rows), &xb2, l.wu.rows, &[]);
            let hb: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
            let down = matvec(&q8_roundtrip(&w[8], l.wd.rows, w[8].len() / l.wd.rows), &hb, l.wd.rows, &[]);
            for (a, b) in xh.iter_mut().zip(&down) { *a += b; }
        }
        let xf = rms(&xh, &out_norm);
        let (head_w, hdims) = tensor_f32(&mut g, "output.weight")
            .unwrap_or((embd_host.clone(), vec![d, vocab]));
        let hrows = head_w.len() / hdims[0];
        let want = matvec(&q8_roundtrip(&head_w, hrows, hdims[0]), &xf, hrows, &[]);
        let (want_arg, _) = want.iter().enumerate().fold((0usize, f32::MIN),
            |(bi, bv), (i, v)| if *v > bv { (i, *v) } else { (bi, bv) });
        println!("\n-- CPU reference, step 0 (same Q8-packed weights as the GPU)");
        println!("   cpu argmax {want_arg}, logit {:.4}", want[want_arg]);
        println!("   gpu argmax {first_arg}, logit {first_logit:.4}");
        let rel = (want[want_arg] - first_logit).abs() / want[want_arg].abs().max(1e-6);
        println!("   argmax agrees: {}; top-logit relative difference {:.3}%",
                 want_arg == first_arg, rel * 100.0);
    }
    Ok(())
}
