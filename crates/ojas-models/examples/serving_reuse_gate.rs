//! Check the serving frontend's chunked prefill and cross-request reuse.
//! usage: OJAS_SNAP=256 serving_reuse_gate model [precision]
use anyhow::{ensure, Result};
use ojas_core::Model;
use ojas_models::decoder::DecoderGpu;

// Keep access to reset_session while exercising the real EngineCore prefill loop.
// Compare an MTP-enabled reused request against a fresh plain-decode request.
struct Borrowed<'a>(&'a DecoderGpu<'a>, bool, &'a std::cell::Cell<usize>);
impl Model for Borrowed<'_> {
    fn context_capacity(&self) -> usize { self.0.context_capacity() }
    fn mtp_verify_width(&self) -> usize { self.0.mtp_verify_width() }
    fn has_mtp(&self) -> bool { self.1 && self.0.has_mtp() }
    fn mtp_step_committed(&self, t: u32, p: usize) -> Option<Vec<u32>> {
        self.2.set(self.2.get()+1); self.0.mtp_step_committed(t,p)
    }
    fn n_layers(&self) -> usize { self.0.n_layers() }
    fn hidden_dim(&self) -> usize { self.0.hidden_dim() }
    fn prefill(&self, t: &[u32], p: usize) { self.0.prefill(t, p) }
    fn reuse_prefix_len(&self, t: &[u32]) -> usize { self.0.reuse_prefix_len(t) }
    fn forward_id(&self, t: u32, p: usize) -> u32 { self.0.forward_id(t, p) }
}
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() >= 2, "usage: serving_reuse_gate model [precision]");
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&args[1])?;
    let prec = args.get(2).map(|v| v.parse()).transpose()?.unwrap_or(4);
    let m = DecoderGpu::load(&gpu, &mut g, 512, prec, None, None)?;
    let calls=std::cell::Cell::new(0);
    let engine = ojas_infer::EngineCore::new(Borrowed(&m,true,&calls));
    let plain = ojas_infer::EngineCore::new(Borrowed(&m,false,&calls));
    let first: Vec<u32> = (0..310).map(|i| 1000 + (i * 7) % 4000).collect();
    let mut second = first.clone();
    second.extend([5100, 5101, 5102]);
    let mut reference = Vec::new();
    for prompt in [&first, &second] {
        m.reset_session();
        reference.push(plain.generate(prompt, 4, true));
    }
    m.reset_session();
    for (i, prompt) in [&first, &second].into_iter().enumerate() {
        let mut reused = None;
        let got = engine.generate_with(prompt, 4, None,
            &mut |done, _| { reused.get_or_insert(done); }, &mut |_| true);
        ensure!(got == reference[i], "serving reuse changed turn {i}: {got:?} != {:?}", reference[i]);
        if i == 1 { ensure!(reused.unwrap_or(0) > 0, "reuse did not engage"); }
        println!("turn {i}: prompt={} reused={} output={got:?} TOKEN-IDENTICAL", prompt.len(), reused.unwrap_or(0));
    }
    ensure!(!m.has_mtp() || calls.get()>0,"MTP never engaged");
    println!("MTP calls={}",calls.get());
    println!("PASS: EngineCore chunked prefill and cross-request reuse");
    Ok(())
}
