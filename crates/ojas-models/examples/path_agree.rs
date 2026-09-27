//! Gate: the single and batched forward paths must agree on the argmax.
//!
//! `forward_id` runs the lm_head as a GEMV and reduces with `argmax`; the batched path
//! runs it as an MMA GEMM (gemm8) and reduces with `argmax_m`. Different kernels and
//! summation orders make the logits differ in the last bits, so wherever the top two
//! are close the argmax can land differently — and then speculative decoding cannot be
//! bit-exact against single decode however correct the accept test is.
//!
//! spec_gate tests the same property end-to-end, but only catches a violation when the
//! drafter happens to fire at the position where the two paths disagree, and whether it
//! fires is decided by wall-clock timing: it fails about one run in five, always at the
//! same position with the same two candidates.
//!
//! This checks the invariant directly and deterministically: every row of a batched
//! forward, at every batch width speculation can produce, against the single path
//! replayed over the same positions.
//!
//! usage: path_agree <gguf> [n_steps] [prec]

use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: path_agree <gguf> [n] [prec]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(96);
    let prec: u8 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    // batch width to compare against: speculation runs M=4 (cur + 3 drafts)
    let mb: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(4);

    anyhow::ensure!(n > 0 && mb > 0, "steps and batch width must be nonzero");
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;

    // spec_gate's prompt: deliberately repetitive so lookup drafting fires.
    let unit: Vec<u32> = vec![785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13];
    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..6 { prompt.extend_from_slice(&unit); }

    m.reset_state();
    m.prefill(&prompt[..prompt.len() - 1], 0);
    let mut pos = prompt.len() - 1;
    let mut cur = prompt[prompt.len() - 1];

    let mut disagree = 0usize;
    println!("comparing EVERY ROW of a batched forward at M={mb} against the");
    println!("single path replayed over the same positions with the same tokens.");
    println!("Row 0 is what a rejected draft emits; rows 1+ are what an ACCEPTED");
    println!("draft emits, and those had never been checked.");
    println!("step row  single  batched   gap");
    for step in 0..n {
        let mut batch = vec![cur];
        for j in 0..(mb - 1) { batch.push(100u32 + ((step * 7 + j * 13) % 400) as u32); }

        let bl = m.forward_batch_logits(&batch, pos).ok_or_else(|| anyhow::anyhow!("batched logits unavailable"))?;
        anyhow::ensure!(bl.len() == batch.len(), "incomplete batched logits");
        // Replay the same tokens one at a time over the same positions. The batched pass
        // already wrote their KV and these rewrite it identically, so each single
        // forward sees the causal context row j had.
        let mut singles = Vec::with_capacity(batch.len());
        for (j, &t) in batch.iter().enumerate() {
            match m.forward_logits(t, pos + j) { Some(v) => singles.push(v), None => break }
        }
        anyhow::ensure!(singles.len() == batch.len(), "incomplete single-token logits");

        for j in 0..batch.len() {
            let sl = &singles[j];
            let mut t1 = 0usize;
            for i in 1..sl.len() { if sl[i] > sl[t1] { t1 = i; } }
            let mut t2 = if t1 == 0 { 1 } else { 0 };
            for i in 0..sl.len() { if i != t1 && sl[i] > sl[t2] { t2 = i; } }
            let gap = sl[t1] - sl[t2];

            let b = &bl[j];
            let mut bt = 0usize;
            for i in 1..b.len() { if b[i] > b[bt] { bt = i; } }

            if bt != t1 {
                disagree += 1;
                if disagree <= 12 {
                    println!("{step:4} {j:3}  {t1:6}  {bt:7}   {gap:.6}  <-- DISAGREE");
                }
            }
        }

        // advance one token along the single-path greedy sequence
        let sl = &singles[0];
        let mut t1 = 0usize;
        for i in 1..sl.len() { if sl[i] > sl[t1] { t1 = i; } }
        pos += 1;
        cur = t1 as u32;
    }
    println!("\ndisagreements: {disagree} across {n} steps x {mb} rows");
    if disagree == 0 {
        println!("GATE: PATH-AGREE PASS (M={mb})");
        Ok(())
    } else {
        // A disagreement means speculative decoding cannot be exact however correct the
        // accept test is, because the two paths disagree on the model's argmax.
        println!("GATE: PATH-AGREE FAIL (M={mb})");
        std::process::exit(1);
    }
}
