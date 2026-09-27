//! Capture batched verification logits along an independent token sequence.
//! usage: OJAS_NO_SPEC=1 flash_verify_logits model reference.ids output.f32 width
use anyhow::{ensure, Result};
use ojas_core::Model;
use std::io::Write;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 5,
        "usage: flash_verify_logits model reference.ids output.f32 width"
    );
    ensure!(
        ojas_core::config::EngineConfig::current().no_spec,
        "set OJAS_NO_SPEC=1"
    );
    let widths: Vec<usize> = a[4]
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(
        !widths.is_empty() && widths.iter().all(|w| (1..=8).contains(w)),
        "widths must be 1..8"
    );
    let ids: Vec<u32> = std::fs::read_to_string(&a[2])?
        .split_whitespace()
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(ids.len() > 1, "missing IDs");
    let n = ids[0] as usize;
    let tokens = &ids[1..];
    ensure!(n > 0 && n < tokens.len(), "invalid prompt length");
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&a[1])?;
    ensure!(g.arch() == "qwen4exp", "Flash only");
    let mut m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 512, 4, None, None)?;
    ensure!(tokens.len() <= m.context_capacity(), "context too small");
    for &width in &widths {
        m.reset_session();
        m.prefill(&tokens[..n - 1], 0);
        let mut output = std::io::BufWriter::new(std::fs::File::create(if widths.len() == 1 {
            a[3].clone()
        } else {
            format!("{}.w{width}.f32", a[3])
        })?);
        for pos in (n - 1..tokens.len()).step_by(width) {
            let count = width.min(tokens.len() - pos);
            let (got, _) = m.flash_target_profile(&tokens[pos..pos + count], pos, None);
            let comparable = count.min(tokens.len() - pos - 1);
            ensure!(
                got[..comparable] == tokens[pos + 1..pos + 1 + comparable],
                "reference mismatch at {pos}"
            );
            let logits = m.flash_target_logits(count);
            ensure!(logits.iter().all(|x| x.is_finite()), "nonfinite logits");
            for v in logits {
                output.write_all(&v.to_le_bytes())?;
            }
        }
        output.flush()?;
        println!("PASS: width={width} rows={}", tokens.len() - n + 1);
    }
    Ok(())
}
