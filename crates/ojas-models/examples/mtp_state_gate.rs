//! Deterministic MTP state checks, independent of the runtime's timing gate.
//! usage: mtp_state_gate model [precision]
use anyhow::{ensure, Result};
use ojas_core::Model;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len() >= 2, "usage: mtp_state_gate model [precision]");
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&a[1])?;
    let bpe = ojas_tokenize::Bpe::from_gguf(&g);
    let text = ojas_tokenize::tokenizer::chat_template(&g.arch(), "What is the capital of France?");
    let prompt: Vec<u32> = bpe.encode(&text).into_iter().map(|v| v as u32).collect();
    ensure!(prompt.len() > 1, "empty prompt");
    let vocab = g.str_arr("tokenizer.ggml.tokens").unwrap().len() as u32;
    let prec = a.get(2).map(|v| v.parse()).transpose()?.unwrap_or(4);
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 512, prec, None, None)?;
    ensure!(m.has_mtp(), "MTP required for this gate");
    let pos = prompt.len()-1; let cur=prompt[pos];
    let start = || { m.reset_session(); m.prefill(&prompt[..pos],0); };
    start();
    let mut baseline=Vec::new(); let mut t=cur;
    for i in 0..8 { t=m.forward_id(t,pos+i); baseline.push(t); }
    for accept in [false,true] {
        start();
        let draft=if accept { baseline[0] } else { (baseline[0]+1)%vocab };
        let (a0,a1)=m.mtp_verify(cur,draft,pos);
        ensure!(a0==baseline[0], "verify row 0 differs from plain decode");
        if accept {
            ensure!(a1==baseline[1], "accepted row 1 differs");
            ensure!(m.forward_id(a1,pos+2)==baseline[2], "state after acceptance differs");
        } else {
            m.mtp_rollback();
            ensure!(m.forward_id(a0,pos+1)==baseline[1], "state after forced rejection differs");
        }
    }
    start(); let mut got=Vec::new(); let mut t=cur;
    while got.len()<baseline.len() {
        let next=m.mtp_generate_step(t,pos+got.len());
        ensure!(!next.is_empty(), "empty MTP step");
        for x in next {
            if got.len()==baseline.len() { break; }
            got.push(x);t=x;
        }
    }
    ensure!(got==baseline, "MTP generation differs: {got:?} vs {baseline:?}");
    println!("PASS: forced acceptance, forced rejection/rollback, and {} generated tokens match plain decode",baseline.len());
    Ok(())
}
