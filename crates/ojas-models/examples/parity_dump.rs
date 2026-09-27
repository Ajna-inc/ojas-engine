//! Dump per-layer hidden states + logits for comparison against HuggingFace.
//!
//! `q6k_gate` and `requant_gate` check one kernel's arithmetic, and `decode_gate` checks
//! that a token sequence did not move; neither catches a layer that is subtly wrong in a
//! way the argmax happens to survive. A per-layer cosine against the reference
//! implementation does.
//!
//! Traces are produced with `Span::forward_span`, one layer at a time, reading the
//! residual stream between layers — the same primitive the pipeline-split workers use,
//! so this also exercises that path.
//!
//! Output, matching `scripts/parity.py --gpu`:
//!   <dump>/prompt<i>.trace.f32    [n_layers+1][T][d]  f32, row 0 = embedding output,
//!                                                     row l = input to layer l
//!   <dump>/prompt<i>.logits.f32   [T][vocab]          f32
//!
//! usage: parity_dump <gguf> <dump_dir> <csv_tokens> [<csv_tokens> ...]

use std::io::Write;

use anyhow::{bail, Result};
use ojas_core::Model;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: parity_dump <gguf> <dump_dir> <csv_tokens> [<csv_tokens> ...]");
    }
    let gguf = &args[1];
    let dump = &args[2];
    std::fs::create_dir_all(dump)?;

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, std::env::var("OJAS_PREC").ok().and_then(|v| v.parse().ok()).unwrap_or(2), None, None)?;
    let (n_layers, d) = (m.n_layers(), m.hidden_dim());
    tracing::info!(target: "parity", "{n_layers} layers, d={d}");

    for (i, csv) in args[3..].iter().enumerate() {
        let tokens: Vec<u32> = csv
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<u32>())
            .collect::<Result<_, _>>()?;
        let t = tokens.len();
        tracing::info!(target: "parity", "prompt{i}: {t} tokens");

        // trace[row][pos][dim]; row 0 = post-embedding, row l = input to layer l.
        let mut trace = vec![0f32; (n_layers + 1) * t * d];
        let mut logits_all: Vec<f32> = Vec::new();

        // Each prompt is an independent sequence. KV self-heals as positions are
        // overwritten, but the recurrent state does not: without this reset, prompt2
        // inherits prompt1's tail and every layer's cosine collapses.
        m.reset_state();

        for (pos, &tok) in tokens.iter().enumerate() {
            // Embed only: an empty layer span leaves the embedding in the residual.
            m.forward_span(tok, pos, 0, 0, true, false);
            let h = m.read_hidden();
            trace[pos * d..(pos + 1) * d].copy_from_slice(&h);

            for l in 0..n_layers {
                // One layer at a time: no embed (the residual carries over) and no head
                // until the end.
                m.forward_span(tok, pos, l, l + 1, false, false);
                let h = m.read_hidden();
                let row = l + 1;
                let off = (row * t + pos) * d;
                trace[off..off + d].copy_from_slice(&h);
            }

            // Head on the final residual, for the logits row.
            m.forward_span(tok, pos, n_layers, n_layers, false, true);
            logits_all.extend_from_slice(&m.logits_vec());
        }

        write_f32(&format!("{dump}/prompt{i}.trace.f32"), &trace)?;
        write_f32(&format!("{dump}/prompt{i}.logits.f32"), &logits_all)?;
        tracing::info!(target: "parity",
            "wrote prompt{i}: trace {}x{}x{}, logits {}x{}",
            n_layers + 1,
            t,
            d,
            t,
            logits_all.len() / t.max(1)
        );
    }
    Ok(())
}

fn write_f32(path: &str, v: &[f32]) -> Result<()> {
    let bytes =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(bytes)?;
    f.flush()?;
    Ok(())
}
