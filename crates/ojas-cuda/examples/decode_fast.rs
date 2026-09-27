//! Fast dense decode path on CUDA: `block_q8_0` read as the file stores it, `__dp4a` int8
//! arithmetic, an f16 KV cache, and a step whose launch arguments never change, so the whole
//! token records once as a CUDA graph and replays.
//!
//! `decode_dense` is the bring-up path: it dequantises every weight to f32 and repacks it into
//! the engine's row-Q8 layout, which works for any quantization. It measured 25 % slower than
//! llama.cpp on Qwen2.5-0.5B-Instruct Q8_0, from four causes. This file addresses those four, so
//! the two can be compared on one card:
//!
//! | cause | `decode_dense` | here |
//! |---|---|---|
//! | matmul | one warp per row, f32 fma | `__dp4a` on packed int8, four lanes per weight block |
//! | weights | dequantised then repacked to row-Q8 at load | the file's `block_q8_0` bytes, uploaded as they are |
//! | KV cache | f32 | f16 (`store_kv2_g` / `attention_short_g`) |
//! | launches | ~13 per layer, re-recorded every token | same launches, captured once and replayed |
//!
//! Graph capture is why the token id and the position live in a two-int device buffer (`ctl`)
//! rather than in launch arguments: `argmax_ctl` writes the sampled id and bumps the position on
//! the device, so nothing about the step depends on which token it is.
//!
//! The projections must be stored `Q8_0`, since nothing here is dequantised; other formats need
//! their own dp4a kernels and are refused rather than silently taking another path.
//!
//! ```text
//! cargo run --release -p ojas-cuda --example decode_fast -- model.gguf [n_tokens] [--no-graph]
//! ```
use anyhow::{anyhow, bail, Result};
use ojas_core::{Device, KernelRuntime};
use ojas_cuda::{CuBuf, CudaGpu};
use ojas_arch::ArchSpec;
use ojas_formats::gguf::Gguf;

const Q8_0: u32 = 8;

/// A `block_q8_0` matrix as the file stores it: 34 bytes per 32 values, `rows * k/32` blocks,
/// no repack and no scale side-table.
struct Q80 {
    w: CuBuf,
    rows: usize,
    k: usize,
}

/// Upload a tensor's raw bytes, refusing anything but `Q8_0`. GGUF stores `[in, out]`, so the
/// row count is the element count divided by the leading dimension.
fn q80(gpu: &CudaGpu, g: &mut Gguf, name: &str) -> Result<Q80> {
    let (dims, ty, bytes) = g.read_tensor_raw(name)?;
    if ty != Q8_0 {
        bail!("{name} is {} ({ty}), not Q8_0 — decode_fast has no kernel for it yet; \
               use decode_dense, which dequantises",
              ojas_formats::gguf::gguf_type_name(ty));
    }
    let k = dims[0] as usize;
    let elems: u64 = dims.iter().product();
    let rows = elems as usize / k;
    if k % 32 != 0 {
        bail!("{name}: K={k} is not a multiple of 32, which block_q8_0 requires");
    }
    Ok(Q80 { w: gpu.upload_bytes(&bytes)?, rows, k })
}

/// Norm weights and biases stay f32; `read_tensor` already decodes F32/F16.
fn f32_tensor(g: &mut Gguf, name: &str) -> Result<Vec<f32>> {
    let (_, ty, bytes) = g.read_tensor(name)?;
    Ok(match ty {
        0 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        1 => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        other => bail!("tensor '{name}': unexpected type {other} after read_tensor"),
    })
}

struct Layer {
    attn_norm: CuBuf,
    ffn_norm: CuBuf,
    /// Qwen2 has q/k/v bias, concatenated in that order so one fused launch can index it.
    qkv_bias: Option<CuBuf>,
    wq: Q80,
    wk: Q80,
    wv: Q80,
    wo: Q80,
    wg: Q80,
    wu: Q80,
    wd: Q80,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or_else(|| anyhow!("usage: decode_fast <model.gguf> [n_tokens]"))?;
    let steps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(64);
    let use_graph = !args.iter().any(|a| a == "--no-graph");
    let check = args.iter().any(|a| a == "--check");

    let mut g = Gguf::open(path)?;
    // The architecture comes from `ojas-arch` rather than from metadata keys read here, so
    // there is one description of a model shared across platforms.
    let a = ArchSpec::from_gguf(&g)?;
    // This graph has no q/k norm, no embedding scale, no sandwich norms and no interleaved
    // RoPE. Anything that needs them is refused rather than ignored: without the check, a Qwen3
    // or Gemma would run to completion and produce fluent output from the wrong model.
    a.require_dense(&["qkv_bias"])?;
    let (n_layer, d, ffn, vocab, eps) = (a.n_layers, a.d, a.ffn, a.vocab, a.eps);
    let l0 = a.layers[0];
    let (n_head, n_kv, hd) = (l0.n_head as usize, l0.n_kv as usize, l0.head_dim as usize);
    let rope_base = l0.rope_base;
    let group = n_head / n_kv.max(1);
    println!("{}: {n_layer} layers, d={d}, heads={n_head}/{n_kv}, hd={hd}, ffn={ffn}, eps={eps}",
             a.arch);

    // `rmsnorm_q8` stages the normalised row in a 4096-float `__shared__` array, which caps the
    // hidden size; a larger `d` would read past the end.
    if d > 4096 {
        bail!("d={d} exceeds rmsnorm_q8's 4096-float shared staging; the wide-model variant is \
               not ported yet");
    }

    let mut gpu = CudaGpu::new(0)?;
    for fam in ["ops", "gemv", "gemv_q8", "attn_core"] {
        gpu.ensure_family(fam)?;
    }
    println!("device tier {:?}, cuda_graphs={}", gpu.caps().tier, gpu.caps().has("cuda_graphs"));

    // ---- weights, straight from the file
    let t_load = std::time::Instant::now();
    let embd = q80(&gpu, &mut g, "token_embd.weight")?;
    if embd.rows != vocab {
        bail!("token_embd has {} rows but the tokenizer declares {vocab}", embd.rows);
    }
    let mut layers = Vec::with_capacity(n_layer);
    for l in 0..n_layer {
        let m = |g: &mut Gguf, gpu: &CudaGpu, n: &str| q80(gpu, g, &format!("blk.{l}.{n}.weight"));
        // Bias is optional and architecture-dependent; q|k|v are concatenated so
        // `qkv_q80_dp4a` can address them with one pointer. A partial set is an error: dropping
        // a bias that exists sends the logits to noise.
        let bq = f32_tensor(&mut g, &format!("blk.{l}.attn_q.bias")).ok();
        let bk = f32_tensor(&mut g, &format!("blk.{l}.attn_k.bias")).ok();
        let bv = f32_tensor(&mut g, &format!("blk.{l}.attn_v.bias")).ok();
        let qkv_bias = match (bq, bk, bv) {
            (Some(a), Some(b), Some(c)) => {
                let mut all = a;
                all.extend(b);
                all.extend(c);
                Some(gpu.upload(&all))
            }
            (None, None, None) => None,
            _ => bail!("blk.{l}: some but not all of attn_q/k/v.bias are present"),
        };
        layers.push(Layer {
            attn_norm: gpu.upload(&f32_tensor(&mut g, &format!("blk.{l}.attn_norm.weight"))?),
            ffn_norm: gpu.upload(&f32_tensor(&mut g, &format!("blk.{l}.ffn_norm.weight"))?),
            qkv_bias,
            wq: m(&mut g, &gpu, "attn_q")?,
            wk: m(&mut g, &gpu, "attn_k")?,
            wv: m(&mut g, &gpu, "attn_v")?,
            wo: m(&mut g, &gpu, "attn_output")?,
            wg: m(&mut g, &gpu, "ffn_gate")?,
            wu: m(&mut g, &gpu, "ffn_up")?,
            wd: m(&mut g, &gpu, "ffn_down")?,
        });
    }
    // Every matvec's K comes from the metadata. A checkpoint whose tensors disagree with
    // `embedding_length` / `feed_forward_length` would read off the end of a row and produce
    // plausible-looking noise.
    for (l, lay) in layers.iter().enumerate() {
        let want_attn_in = d;
        let want_o_in = lay.wq.rows;              // o_proj consumes the concatenated heads
        let ffn_rows = lay.wg.rows;
        let _ = ffn_rows;
        for (name, got, want) in [
            ("attn_q", lay.wq.k, want_attn_in), ("attn_k", lay.wk.k, want_attn_in),
            ("attn_v", lay.wv.k, want_attn_in), ("attn_output", lay.wo.k, want_o_in),
            ("ffn_gate", lay.wg.k, d), ("ffn_up", lay.wu.k, d), ("ffn_down", lay.wd.k, ffn_rows),
        ] {
            if got != want {
                bail!("blk.{l}.{name}: K={got}, expected {want} from the model metadata");
            }
        }
        if lay.wu.rows != ffn_rows || lay.wd.rows != d || lay.wo.rows != d {
            bail!("blk.{l}: ffn_up/ffn_down/attn_output row counts disagree with d={d}, ffn={ffn_rows}");
        }
    }
    let out_norm = gpu.upload(&f32_tensor(&mut g, "output_norm.weight")?);
    // An untied head when the file has one; otherwise the lm_head is the embedding table, and
    // since nothing is repacked here the same buffer serves. `ArchSpec` already decided tied vs
    // untied from tensor presence.
    if a.tied_head() {
        println!("tied lm_head (shares token_embd)");
    }
    let head = q80(&gpu, &mut g, &a.lm_head)?;
    if head.k != d {
        bail!("lm_head: K={}, expected {d}", head.k);
    }
    if layers[0].wg.rows != ffn {
        bail!("ffn_gate has {} rows but feed_forward_length is {ffn}", layers[0].wg.rows);
    }
    println!("vocab {vocab}, ffn {ffn}, qkv bias {}; weights on device in {:.1} s",
             if layers[0].qkv_bias.is_some() { "yes" } else { "no" }, t_load.elapsed().as_secs_f64());

    // ---- scratch
    let max_seq = steps.max(2);
    let kvdim = n_kv * hd;
    let x = gpu.alloc(d);
    let q = gpu.alloc(n_head * hd);
    let kvec = gpu.alloc(kvdim);
    let vvec = gpu.alloc(kvdim);
    let att = gpu.alloc(n_head * hd);
    let hb = gpu.alloc(ffn);
    let logits = gpu.alloc(vocab);
    // the quantised activation, sized for the widest consumer (the FFN's K is d, down's is ffn)
    let qmax = d.max(ffn);
    let xq = gpu.alloc_bytes(qmax)?;
    let d8 = gpu.alloc(qmax / 32);
    let d8sum = gpu.alloc(qmax / 32);
    // f16 KV cache: half the bandwidth of decode_dense's f32, matching the Metal path
    let kcache: Vec<CuBuf> = (0..n_layer).map(|_| gpu.alloc(max_seq * kvdim / 2)).collect();
    let vcache: Vec<CuBuf> = (0..n_layer).map(|_| gpu.alloc(max_seq * kvdim / 2)).collect();
    println!("KV cache {:.1} MiB f16 ({} positions)",
             2.0 * n_layer as f64 * max_seq as f64 * kvdim as f64 * 2.0 / (1 << 20) as f64, max_seq);

    // stands in for a null bias pointer; sized for the widest q|k|v the model has
    let zero_bias = gpu.alloc(layers.iter().map(|l| l.wq.rows + 2 * l.wk.rows).max().unwrap_or(1));

    // ctl[0] = current token id, ctl[1] = current position. Both on the device.
    let ctl_init: Vec<u8> = [1i32, 0i32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut ctl = gpu.upload_bytes(&ctl_init)?;

    let scale = 1.0f32 / (hd as f32).sqrt();
    let warps = 8u32;
    let rows_grid = |n: usize| [(n as u32).div_ceil(warps), 1, 1];
    let nq_rope = (n_head * hd / 2) as u32;
    let nk_rope = (kvdim / 2) as u32;

    // One decode step. Every argument is fixed for the whole run (the token and position are
    // read from `ctl` inside the kernels), which is the condition for graph capture.
    // `upto = Some(n)` stops after layer n and skips the head, so `--check` can read the residual
    // stream one layer at a time; `None` runs the whole step, as the timed run does.
    let step = |gpu: &CudaGpu, enc: &<CudaGpu as Device>::Enc, ctl: &CuBuf,
                upto: Option<usize>| -> Result<()> {
        gpu.dispatch(enc, "embed_q80_g", &[(&embd.w, 0), (&x, 0), (&ctl, 0)], &[d as u32],
                     [8, 1, 1], [128, 1, 1])?;
        for (li, l) in layers.iter().enumerate() {
            if upto.is_some_and(|n| li > n) { return Ok(()); }
            // norm and quantize in one pass: the dp4a matvecs need q8_1 activations, and a
            // separate launch would re-read the whole residual stream
            gpu.dispatch(enc, "rmsnorm_q8",
                         &[(&x, 0), (&l.attn_norm, 0), (&xq, 0), (&d8, 0), (&d8sum, 0)],
                         &[d as u32, eps.to_bits()], [1, 1, 1], [256, 1, 1])?;
            // `qkv_q80_dp4a` tests `bias` for null, but a null device pointer cannot be
            // expressed through this seam, so an architecture without bias (Qwen3, Gemma,
            // Llama) gets a zero vector: the same arithmetic, one buffer.
            let bias = l.qkv_bias.as_ref().unwrap_or(&zero_bias);
            let nq = l.wq.rows;
            let nkv = l.wk.rows;
            gpu.dispatch(enc, "qkv_q80_dp4a",
                         &[(&l.wq.w, 0), (&l.wk.w, 0), (&l.wv.w, 0), (&xq, 0), (&d8, 0),
                           (&q, 0), (&kvec, 0), (&vvec, 0), (bias, 0)],
                         &[d as u32, nq as u32, nkv as u32],
                         rows_grid(nq + 2 * nkv), [256, 1, 1])?;
            gpu.dispatch(enc, "rope_qk_g", &[(&q, 0), (&kvec, 0), (&ctl, 0)],
                         &[hd as u32, hd as u32, rope_base.to_bits(), nq_rope, nk_rope],
                         [(nq_rope + nk_rope).div_ceil(128), 1, 1], [128, 1, 1])?;
            gpu.dispatch(enc, "store_kv2_g",
                         &[(&kcache[li], 0), (&vcache[li], 0), (&kvec, 0), (&vvec, 0), (&ctl, 0)],
                         &[kvdim as u32, kvdim as u32],
                         [(kvdim as u32).div_ceil(256), 1, 1], [256, 1, 1])?;
            gpu.dispatch(enc, "attention_short_g",
                         &[(&q, 0), (&kcache[li], 0), (&vcache[li], 0), (&att, 0), (&ctl, 0)],
                         &[hd as u32, kvdim as u32, group as u32, scale.to_bits()],
                         [n_head as u32, 1, 1], [256, 1, 1])?;
            gpu.dispatch(enc, "quantize_q8_1", &[(&att, 0), (&xq, 0), (&d8, 0), (&d8sum, 0)],
                         &[d as u32], [(d / 32) as u32, 1, 1], [32, 1, 1])?;
            gpu.dispatch(enc, "gemv_q80_dp4a_accum",
                         &[(&l.wo.w, 0), (&xq, 0), (&d8, 0), (&x, 0)],
                         &[d as u32, l.wo.rows as u32], rows_grid(l.wo.rows), [256, 1, 1])?;
            gpu.dispatch(enc, "rmsnorm_q8",
                         &[(&x, 0), (&l.ffn_norm, 0), (&xq, 0), (&d8, 0), (&d8sum, 0)],
                         &[d as u32, eps.to_bits()], [1, 1, 1], [256, 1, 1])?;
            gpu.dispatch(enc, "ffn_gu_q80_dp4a",
                         &[(&l.wg.w, 0), (&l.wu.w, 0), (&xq, 0), (&d8, 0), (&hb, 0)],
                         &[d as u32, ffn as u32, 0], rows_grid(ffn), [256, 1, 1])?;
            gpu.dispatch(enc, "quantize_q8_1", &[(&hb, 0), (&xq, 0), (&d8, 0), (&d8sum, 0)],
                         &[ffn as u32], [(ffn / 32) as u32, 1, 1], [32, 1, 1])?;
            gpu.dispatch(enc, "gemv_q80_dp4a_accum",
                         &[(&l.wd.w, 0), (&xq, 0), (&d8, 0), (&x, 0)],
                         &[ffn as u32, l.wd.rows as u32], rows_grid(l.wd.rows), [256, 1, 1])?;
        }
        if upto.is_some() { return Ok(()); }
        // the final norm quantises too, so the lm_head matvec needs no separate quantize launch
        gpu.dispatch(enc, "rmsnorm_q8", &[(&x, 0), (&out_norm, 0), (&xq, 0), (&d8, 0), (&d8sum, 0)],
                     &[d as u32, eps.to_bits()], [1, 1, 1], [256, 1, 1])?;
        gpu.dispatch(enc, "gemv_q80_dp4a", &[(&head.w, 0), (&xq, 0), (&d8, 0), (&logits, 0)],
                     &[d as u32, head.rows as u32], rows_grid(head.rows), [256, 1, 1])?;
        // sample and advance on the device, closing the loop without the host
        gpu.dispatch(enc, "argmax_ctl", &[(&logits, 0), (&ctl, 0)], &[vocab as u32],
                     [1, 1, 1], [256, 1, 1])?;
        Ok(())
    };

    let read_ctl = |gpu: &CudaGpu, ctl: &CuBuf| -> Result<(i32, i32)> {
        let mut b = vec![0u8; 8];
        gpu.read_bytes(ctl, 0, &mut b)?;
        Ok((i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            i32::from_le_bytes([b[4], b[5], b[6], b[7]])))
    };

    // ---- warm one step so compilation and first-touch paging are not in the measurement
    {
        let enc = gpu.begin();
        step(&gpu, &enc, &ctl, None)?;
        gpu.submit(enc)?;
    }
    gpu.write_bytes(&mut ctl, 0, &ctl_init)?;

    // --check: one step from position 0, whose logits the CPU reference below reproduces. At
    // position 0 RoPE is the identity and attention sees a single key, which keeps the reference
    // short. Resetting ctl afterwards makes the timed run overwrite cache position 0.
    let mut logits0: Option<Vec<f32>> = None;
    let mut resid: Vec<Vec<f32>> = Vec::new();
    let mut qprobe: Vec<Vec<i8>> = Vec::new();
    if check {
        // The residual stream after each layer, so the reference can say where it diverges.
        // Each probe re-runs from position 0; `embed_q80_g` writes x rather than accumulating
        // into it, and cache position 0 is rewritten, so the runs are independent.
        for li in 0..n_layer {
            gpu.write_bytes(&mut ctl, 0, &ctl_init)?;
            let enc = gpu.begin();
            step(&gpu, &enc, &ctl, Some(li))?;
            gpu.submit(enc)?;
            let mut xh = vec![0f32; d];
            gpu.read(&x, &mut xh);
            resid.push(xh);
            // `xq` still holds this layer's last quantisation, the ffn_down input. Comparing
            // the int8 values directly separates "one activation crossed a quantisation
            // boundary" from "a kernel in this layer is wrong".
            let mut q = vec![0u8; ffn];
            gpu.read_bytes(&xq, 0, &mut q)?;
            qprobe.push(q.into_iter().map(|b| b as i8).collect::<Vec<i8>>());
        }
        gpu.write_bytes(&mut ctl, 0, &ctl_init)?;
        let enc = gpu.begin();
        step(&gpu, &enc, &ctl, None)?;
        gpu.submit(enc)?;
        let mut lg = vec![0f32; vocab];
        gpu.read(&logits, &mut lg);
        logits0 = Some(lg);
        gpu.write_bytes(&mut ctl, 0, &ctl_init)?;
    }

    let mut first: Vec<i32> = Vec::new();
    let t0 = std::time::Instant::now();
    if use_graph {
        // Capture one step and replay it. Every launch argument is fixed and the two values
        // that change per token live in `ctl`, so each replay is a different token.
        let recorded = gpu.capture(&mut |_| {
            let enc = gpu.begin();
            step(&gpu, &enc, &ctl, None)?;
            Ok(())
        })?.ok_or_else(|| anyhow!("stream capture produced no graph"))?;
        // capture records without executing, so the warm state is untouched and position is 0
        for _ in 0..steps {
            gpu.replay(&recorded)?;
        }
        gpu.sync()?;
    } else {
        for _ in 0..steps {
            let enc = gpu.begin();
            step(&gpu, &enc, &ctl, None)?;
            gpu.submit(enc)?;
            if first.len() < 8 {
                first.push(read_ctl(&gpu, &ctl)?.0);
            }
        }
        gpu.sync()?;
    }
    let secs = t0.elapsed().as_secs_f64();
    let (tok, pos) = read_ctl(&gpu, &ctl)?;

    println!("\n{steps} steps{} in {:.1} ms = {:.3} ms/token, {:.1} tok/s",
             if use_graph { " (CUDA graph replay)" } else { " (per-step encode)" },
             secs * 1000.0, secs * 1000.0 / steps as f64, steps as f64 / secs);
    println!("final ctl: token {tok}, position {pos}");
    if pos != steps as i32 {
        bail!("position advanced to {pos} over {steps} steps — the device-side counter is wrong");
    }
    if !first.is_empty() {
        println!("first tokens: {first:?}");
    }
    let mut lg = vec![0f32; vocab];
    gpu.read(&logits, &mut lg);
    let finite = lg.iter().filter(|v| v.is_finite()).count();
    if finite != vocab {
        bail!("{}/{vocab} logits are non-finite", finite);
    }
    println!("logits all finite, peak {:.4}", lg.iter().fold(0f32, |m, v| m.max(v.abs())));

    if let Some(gpu0) = logits0 {
        check_step0(&mut g, &layers, &gpu0, &resid, &qprobe, n_head, n_kv, hd, eps)?;
    }
    Ok(())
}

/// Step 0 on the CPU from the same `block_q8_0` bytes the kernels read, compared against a fresh
/// GPU step 0.
///
/// Position 0 keeps the reference short and exact: RoPE at pos 0 is the identity, and attention
/// over a single key reduces to `v`. It covers embed, rmsnorm, q/k/v + bias, o_proj, SwiGLU,
/// ffn_down and the lm_head; the multi-position paths are exercised by the 64-step run instead.
///
/// The reference performs the kernel's arithmetic rather than a more accurate version of it:
///
/// * f32 activations against the GPU's q8_1 ones moved the argmax (220 vs 14582) because the top
///   two logits were within 0.3 of each other, so the reference quantises activations exactly as
///   `quantize_q8_1` does.
/// * dequantising the weights through `dequant_to_f16` rounds every `d * q` to f16 and loses a
///   couple of mantissa bits per weight; over ~900 terms that produced a 3.2 % deviation, with
///   the reference as the less accurate side. The dot product here is the integer one the kernel
///   performs: `Σ_blocks d_w · d_x · Σ(int8 · int8)`, accumulated in f64.
///
/// What remains is summation order and f32 versus f64 accumulation.
struct HostQ80 {
    bytes: Vec<u8>,
    rows: usize,
    k: usize,
}

impl HostQ80 {
    fn load(g: &mut Gguf, name: &str) -> Result<Self> {
        let (dims, ty, bytes) = g.read_tensor_raw(name)?;
        if ty != Q8_0 {
            bail!("{name}: reference path expects Q8_0, got {ty}");
        }
        let k = dims[0] as usize;
        let elems: usize = dims.iter().map(|v| *v as usize).product();
        Ok(Self { bytes, rows: elems / k, k })
    }

    /// Row `r` dotted with a q8_1 activation, in the kernel's own arithmetic.
    fn dot(&self, r: usize, xq: &[i8], xd: &[f32]) -> f32 {
        let nblk = self.k / 32;
        let mut acc = 0f64;
        for b in 0..nblk {
            let off = (r * nblk + b) * 34;
            let dw = half::f16::from_le_bytes([self.bytes[off], self.bytes[off + 1]]).to_f32();
            let mut sumi = 0i64;
            for i in 0..32 {
                sumi += (self.bytes[off + 2 + i] as i8 as i64) * (xq[b * 32 + i] as i64);
            }
            acc += dw as f64 * xd[b] as f64 * sumi as f64;
        }
        acc as f32
    }

    /// `embed_q80`: dequantise one row, f32 product, no f16 round-trip.
    fn row_f32(&self, r: usize) -> Vec<f32> {
        let nblk = self.k / 32;
        let mut out = vec![0f32; self.k];
        for b in 0..nblk {
            let off = (r * nblk + b) * 34;
            let dw = half::f16::from_le_bytes([self.bytes[off], self.bytes[off + 1]]).to_f32();
            for i in 0..32 {
                out[b * 32 + i] = self.bytes[off + 2 + i] as i8 as f32 * dw;
            }
        }
        out
    }
}

/// `quantize_q8_1` on the host: absmax per 32 values, round to int8, one scale per block.
fn q8_1(v: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; v.len()];
    let mut d = vec![0f32; v.len() / 32];
    for (b, blk) in v.chunks(32).enumerate() {
        let amax = blk.iter().fold(0f32, |m, x| m.max(x.abs()));
        let dd = amax / 127.0;
        let id = if dd > 0.0 { 1.0 / dd } else { 0.0 };
        d[b] = dd;
        for (i, x) in blk.iter().enumerate() {
            // `__float2int_rn` is round-to-nearest-even; Rust's `round()` is ties-away-from-zero.
            // Matching it removes one source of reference-side disagreement (measured: not the
            // cause of the late-layer step below).
            q[b * 32 + i] = (x * id).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, d)
}

#[allow(clippy::too_many_arguments)]
fn check_step0(g: &mut Gguf, layers: &[Layer], gpu_logits: &[f32], gpu_resid: &[Vec<f32>],
               gpu_q: &[Vec<i8>], n_head: usize, n_kv: usize, hd: usize, eps: f32) -> Result<()> {
    let rms = |v: &[f32], w: &[f32]| -> Vec<f32> {
        let ss: f32 = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        v.iter().zip(w).map(|(x, gw)| x * inv * gw).collect()
    };
    let mv = |w: &HostQ80, xq: &[i8], xd: &[f32], bias: &[f32]| -> Vec<f32> {
        (0..w.rows).map(|r| w.dot(r, xq, xd) + bias.get(r).copied().unwrap_or(0.0)).collect()
    };

    let mut drift: Vec<(usize, f32)> = Vec::new();
    let mut qdiff: Vec<(usize, usize, usize, i32)> = Vec::new();
    let embd = HostQ80::load(g, "token_embd.weight")?;
    let mut xh = embd.row_f32(1);                 // ctl starts at token 1
    let group = n_head / n_kv.max(1);
    for (l, lay) in layers.iter().enumerate() {
        let an = f32_tensor(g, &format!("blk.{l}.attn_norm.weight"))?;
        let fnw = f32_tensor(g, &format!("blk.{l}.ffn_norm.weight"))?;
        let bv = f32_tensor(g, &format!("blk.{l}.attn_v.bias")).unwrap_or_default();
        let (xq, xd) = q8_1(&rms(&xh, &an));
        let wv = HostQ80::load(g, &format!("blk.{l}.attn_v.weight"))?;
        let vv = mv(&wv, &xq, &xd, &bv);
        // Attention at position 0 is v, routed through GQA and through the f16 KV cache, so the
        // reference rounds where `store_kv2_g` does. Without that round the reference disagrees
        // by a few percent and the cache precision looks like a kernel error.
        let att: Vec<f32> = (0..n_head)
            .flat_map(|h| vv[(h / group) * hd..(h / group) * hd + hd]
                .iter().map(|v| half::f16::from_f32(*v).to_f32()).collect::<Vec<f32>>())
            .collect();
        let (aq, ad) = q8_1(&att);
        let o = mv(&HostQ80::load(g, &format!("blk.{l}.attn_output.weight"))?, &aq, &ad, &[]);
        for (a, b) in xh.iter_mut().zip(&o) { *a += b; }
        let (x2q, x2d) = q8_1(&rms(&xh, &fnw));
        let gt = mv(&HostQ80::load(g, &format!("blk.{l}.ffn_gate.weight"))?, &x2q, &x2d, &[]);
        let up = mv(&HostQ80::load(g, &format!("blk.{l}.ffn_up.weight"))?, &x2q, &x2d, &[]);
        let hb: Vec<f32> = gt.iter().zip(&up).map(|(a, b)| a / (1.0 + (-a).exp()) * b).collect();
        let (hq, hd_) = q8_1(&hb);
        let dn = mv(&HostQ80::load(g, &format!("blk.{l}.ffn_down.weight"))?, &hq, &hd_, &[]);
        for (a, b) in xh.iter_mut().zip(&dn) { *a += b; }
        let _ = lay;
        if let Some(gq) = gpu_q.get(l) {
            let n = gq.len().min(hq.len());
            let diff = (0..n).filter(|i| gq[*i] != hq[*i]).count();
            let worst = (0..n).map(|i| (gq[i] as i32 - hq[i] as i32).abs()).max().unwrap_or(0);
            qdiff.push((l, diff, n, worst));
        }
        if let Some(gr) = gpu_resid.get(l) {
            let peak = xh.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let dev = gr.iter().zip(&xh).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            drift.push((l, 100.0 * dev / peak));
        }
    }
    if !drift.is_empty() {
        // Reading the output, from Qwen2.5-0.5B Q8_0:
        //
        //   L00..L16   0 int8 values differ.
        //   L17, L20   one and seventeen values differ, each by a single level.
        //   L22, L23   583 then 1347 differ, by up to 9 levels.
        //
        // That shape is quantisation boundaries compounding, not a kernel bug. A last-ulp
        // difference is invisible until one activation lands on the other side of an int8 step;
        // that flip shifts the next layer's input, so more values cross. Qwen's late layers carry
        // large outlier activations, so a per-32-block absmax leaves the other 31 values few
        // levels to sit on, which is why it appears at the end and not the start.
        //
        // A bug looks different: values differing in the early layers, or by many levels before
        // the last few. Judge it on where the first difference appears, not the final number.
        println!("\n-- residual-stream drift per layer (max |gpu - cpu| as % of that layer's peak)");
        for chunk in drift.chunks(6) {
            let line: Vec<String> = chunk.iter().map(|(l, v)| format!("L{l:02}:{v:7.4}")).collect();
            println!("   {}", line.join("  "));
        }
        let first = drift.iter().find(|(_, v)| *v > 0.05).map(|(l, _)| *l);
        let jump = drift.windows(2).map(|w| w[1].1 - w[0].1).fold(0f32, f32::max);
        println!("   first layer above 0.05 %: {}; final {:.2} %; largest single-layer step {jump:.2} pp",
                 first.map_or("none".to_string(), |l| format!("L{l}")), drift.last().unwrap().1);
        let worst_level = qdiff.iter().map(|(_, _, _, w)| *w).max().unwrap_or(0);
        let total: usize = qdiff.iter().map(|(_, d, _, _)| *d).sum();
        let width = qdiff.first().map_or(0, |(_, _, n, _)| *n);
        println!("   ffn_down int8 inputs differing from the reference: {total} of {} across all \
                  layers, never by more than {worst_level} level(s)",
                 width * qdiff.len());
        for (l, d, n, w) in qdiff.iter().filter(|(_, d, _, _)| *d > 0) {
            println!("     L{l}: {d}/{n} values differ, max {w} level(s)");
        }
    }
    let (fq, fd) = q8_1(&rms(&xh, &f32_tensor(g, "output_norm.weight")?));
    let hw = HostQ80::load(g, "output.weight")
        .or_else(|_| HostQ80::load(g, "token_embd.weight"))?;
    let want = mv(&hw, &fq, &fd, &[]);
    let (want_arg, want_val) = want.iter().enumerate()
        .fold((0usize, f32::MIN), |(bi, bv), (i, v)| if *v > bv { (i, *v) } else { (bi, bv) });
    let (got_arg, got_val) = gpu_logits.iter().enumerate()
        .fold((0usize, f32::MIN), |(bi, bv), (i, v)| if *v > bv { (i, *v) } else { (bi, bv) });

    println!("\n-- CPU reference, step 0 (block_q8_0 integer dot, q8_1 activations — the kernel's own arithmetic)");
    println!("   cpu argmax {want_arg}, logit {want_val:.4}");
    println!("   gpu argmax {got_arg}, logit {got_val:.4}");

    // Rank agreement over the head of the distribution says more than one logit does: a subtly
    // wrong kernel usually keeps the argmax and shuffles what is behind it.
    let mut order: Vec<usize> = (0..want.len()).collect();
    order.sort_unstable_by(|a, b| want[*b].total_cmp(&want[*a]));
    let top: Vec<usize> = order.into_iter().take(10).collect();
    let mut gorder: Vec<usize> = (0..gpu_logits.len()).collect();
    gorder.sort_unstable_by(|a, b| gpu_logits[*b].total_cmp(&gpu_logits[*a]));
    let gtop: Vec<usize> = gorder.into_iter().take(10).collect();
    let overlap = top.iter().filter(|t| gtop.contains(t)).count();

    let rel = (want_val - got_val).abs() / want_val.abs().max(1e-6);
    println!("   argmax agrees: {}; top-logit difference {:.4} %; top-10 overlap {overlap}/10",
             want_arg == got_arg, rel * 100.0);
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let max_abs = gpu_logits.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("   full-logit deviation {max_abs:.4} over peak {peak:.2} = {:.4} % of peak",
             100.0 * max_abs / peak);
    if want_arg != got_arg {
        // A disagreement only means something if the candidates are separated, so print each
        // side's logit for both: "the kernel is wrong" and "the top two are within a rounding
        // error" have different fixes.
        println!("   cpu: [{want_arg}]={:.4} [{got_arg}]={:.4}   gap {:.4}",
                 want[want_arg], want[got_arg], want[want_arg] - want[got_arg]);
        println!("   gpu: [{want_arg}]={:.4} [{got_arg}]={:.4}   gap {:.4}",
                 gpu_logits[want_arg], gpu_logits[got_arg],
                 gpu_logits[got_arg] - gpu_logits[want_arg]);
        bail!("step-0 argmax disagrees with the CPU reference: {got_arg} vs {want_arg}");
    }
    Ok(())
}
