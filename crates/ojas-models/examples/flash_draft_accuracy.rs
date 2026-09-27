//! Teacher-forced draft accuracy at the same token positions, independent of scheduling.
//! usage: flash_draft_accuracy model reference.ids
use anyhow::{ensure, Result};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 3 || (a.len() == 4 && a[3] == "projection-only"),
        "usage: flash_draft_accuracy model reference.ids [projection-only]"
    );
    ensure!(
        !ojas_core::config::EngineConfig::current().no_spec,
        "unset OJAS_NO_SPEC"
    );
    let ids: Vec<u32> = std::fs::read_to_string(&a[2])?
        .split_whitespace()
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(ids.len() > 1, "missing IDs");
    let n = ids[0] as usize;
    let tokens = &ids[1..];
    ensure!(n > 0 && tokens.len() >= n + 32, "need 32 reference outputs");
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&a[1])?;
    ensure!(g.arch() == "qwen4exp", "Flash only");
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 512, 4, None, None)?;
    m.reset_session();
    m.prefill(&tokens[..n - 1], 0);
    let (rmse, max_abs, pooled_rmse) = m.flash_mtp_projection_reference(tokens[n - 1]);
    ensure!(
        rmse.is_finite() && max_abs.is_finite() && rmse < 0.0001 && max_abs < 0.001,
        "per-stream projection reference mismatch: {rmse} / {max_abs}"
    );
    ensure!(
        pooled_rmse > 0.001,
        "fixture does not distinguish pooled from per-stream projection"
    );
    println!("{{\"projection_rmse\":{rmse},\"projection_max_abs\":{max_abs},\"pooled_reference_rmse\":{pooled_rmse}}}");
    if a.len() == 4 {
        return Ok(());
    }
    let (mut eligible, mut correct) = ([0usize; 3], [0usize; 3]);
    for pos in n - 1..n + 31 {
        let mut predictions = Vec::new();
        let mut tok = tokens[pos];
        let count = 3.min(tokens.len() - pos - 1);
        for j in 0..count {
            tok = if j == 0 {
                m.flash_draft_current(tok, pos)
            } else {
                m.flash_draft_probe(tok, pos + j, 0, true)
            };
            predictions.push(tok);
        }
        let expected = &tokens[pos + 1..pos + 1 + count];
        let mut prefix = true;
        for j in 0..count {
            if prefix {
                eligible[j] += 1;
                if predictions[j] == expected[j] {
                    correct[j] += 1;
                } else {
                    prefix = false;
                }
            }
        }
        let actual = m.forward_id(tokens[pos], pos);
        ensure!(actual == tokens[pos + 1], "target changed at {pos}");
        println!("{{\"position\":{pos},\"draft\":{predictions:?},\"expected\":{expected:?}}}");
    }
    println!(
        "{{\"eligible\":{eligible:?},\"correct\":{correct:?},\"target_reference_passed\":true}}"
    );
    Ok(())
}
