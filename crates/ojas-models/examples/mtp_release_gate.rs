//! Full MTP protocol and serving checks; missing draft weights are reported explicitly.
//! usage: mtp_release_gate model [precision] [output_tokens]
use anyhow::{ensure, Result};
use ojas_core::Model;
use ojas_models::decoder::DecoderGpu;
use std::cell::Cell;

struct Observed<'a> {
    model: &'a DecoderGpu<'a>,
    enabled: bool,
    calls: &'a Cell<usize>,
    accepted: &'a Cell<usize>,
}
impl Model for Observed<'_> {
    fn context_capacity(&self) -> usize { self.model.context_capacity() }
    fn mtp_verify_width(&self) -> usize { self.model.mtp_verify_width() }
    fn n_layers(&self) -> usize { self.model.n_layers() }
    fn hidden_dim(&self) -> usize { self.model.hidden_dim() }
    fn prefill(&self, t: &[u32], p: usize) { self.model.prefill(t,p); }
    fn reuse_prefix_len(&self, t: &[u32]) -> usize { self.model.reuse_prefix_len(t) }
    fn forward_id(&self, t: u32, p: usize) -> u32 { self.model.forward_id(t,p) }
    fn forward_logits(&self, t: u32, p: usize) -> Option<Vec<f32>> { self.model.forward_logits(t,p) }
    fn has_mtp(&self) -> bool { self.enabled && self.model.has_mtp() }
    fn mtp_step_committed(&self, t: u32, p: usize) -> Option<Vec<u32>> {
        let v=self.model.mtp_step_committed(t,p)?;
        self.calls.set(self.calls.get()+1);
        self.accepted.set(self.accepted.get()+usize::from(v.len()>1));
        Some(v)
    }
}
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len()>=2,"usage: mtp_release_gate model [precision] [output_tokens]");
    ensure!(std::env::var_os("OJAS_NO_SPEC").is_none(), "unset OJAS_NO_SPEC for this gate");
    let prec=args.get(2).map(|s|s.parse()).transpose()?.unwrap_or(4);
    let count:usize=args.get(3).map(|s|s.parse()).transpose()?.unwrap_or(32);
    ensure!((8..=64).contains(&count),"output_tokens must be 8..=64");
    let gpu=ojas_metal::MetalGpu::new()?;
    let mut g=ojas_formats::gguf::Gguf::open(&args[1])?;
    let bpe=ojas_tokenize::Bpe::from_gguf(&g);
    let arch=g.arch();
    let vocab=g.str_arr("tokenizer.ggml.tokens").unwrap().len() as u32;
    let m=DecoderGpu::load(&gpu,&mut g,128,prec,None,None)?;
    let calls=Cell::new(0);let accepted=Cell::new(0);
    let plain=ojas_infer::EngineCore::new(Observed{model:&m,enabled:false,calls:&calls,accepted:&accepted});
    let mut spec=ojas_infer::EngineCore::new(Observed{model:&m,enabled:true,calls:&calls,accepted:&accepted});
    let has=m.has_mtp();
    println!("architecture={arch} MTP={has} verify_width={}",m.mtp_verify_width());
    let mut first=Vec::new();let mut baseline=Vec::new();
    for (i,text) in ["What is the capital of France?","Write a Python function that adds two integers.","Continue: one two three one two three one two three"].iter().enumerate() {
        let prompt:Vec<u32>=bpe.encode(&ojas_tokenize::tokenizer::chat_template(&arch,text)).into_iter().map(|t|t as u32).collect();
        ensure!(prompt.len()+count<=m.context_capacity(),"fixture exceeds context");
        m.reset_session();let want=plain.generate(&prompt,count,true);
        m.reset_session();calls.set(0);accepted.set(0);let got=spec.generate(&prompt,count,true);
        ensure!(got==want,"serving MTP differs on prompt {i}: {got:?} vs {want:?}");
        ensure!(!has || calls.get()>0,"MTP did not engage");
        ensure!(has || calls.get()==0,"draft-less model attempted MTP");
        println!("prompt {i}: {} tokens match; MTP calls={} accepted_calls={}",got.len(),calls.get(),accepted.get());
        if i==0 {first=prompt;baseline=want;}
    }
    if has {
        let width=m.mtp_verify_width();let pos=first.len()-1;let cur=first[pos];
        ensure!(width<=baseline.len(),"increase output_tokens to cover verify width");
        for bad in 0..width { // last case accepts every draft; others reject at each draft position
            m.reset_session();m.prefill(&first[..pos],0);
            let mut batch=vec![cur];batch.extend_from_slice(&baseline[..width-1]);
            if bad<width-1 {batch[bad+1]=(batch[bad+1]+1)%vocab;}
            let got=m.mtp_verify_n(&batch,pos);
            let correct_rows=if bad==width-1 {width} else {bad+1};
            ensure!(got[..correct_rows]==baseline[..correct_rows],"verify prefix differs at rejection position {bad}");
            let (mut token,offset)=if bad==width-1 {(got[width-1],width)} else if ojas_core::config::EngineConfig::current().mtp_prefix && arch=="qwen4exp" {m.mtp_rollback_to(bad);(got[bad],bad+1)} else {m.mtp_rollback();(got[0],1)};
            for k in offset..(offset+4).min(baseline.len()) {
                token=m.forward_id(token,pos+k);
                ensure!(token==baseline[k],"state after reject/accept case {bad} differs at {k}");
            }
            println!("forced case {bad}: verify prefix and continuation match");
        }
    }
    // A callback stop and EOS may cut a multi-token commit; the next request must start cleanly.
    m.reset_session();let stopped=spec.generate_with(&first,count,None,&mut|_,_|{},&mut |_|false);
    ensure!(stopped==baseline[..1],"callback stop emitted extra tokens");
    spec.eos=Some(baseline[0]);m.reset_session();
    ensure!(spec.generate(&first,count,true)==baseline[..1],"EOS emitted extra tokens");spec.eos=None;
    ensure!(spec.generate(&first,count,true)==baseline,"new request after early stop differs");
    let opts=ojas_infer::SampleOpts{seed:91,..Default::default()};
    m.reset_session();let want=plain.generate_with(&first,8,Some(&opts),&mut|_,_|{},&mut |_|true);
    m.reset_session();calls.set(0);let got=spec.generate_with(&first,8,Some(&opts),&mut|_,_|{},&mut |_|true);
    ensure!(want==got && calls.get()==0,"sampled generation must bypass greedy MTP");
    let boundary:Vec<u32>=(0..m.context_capacity()-2).map(|i|1000+(i as u32%100)).collect();
    m.reset_session();let want=plain.generate(&boundary,32,true);
    m.reset_session();let got=spec.generate(&boundary,32,true);
    ensure!(want==got && got.len()==3,"context-end generation differs");
    m.reset_session();m.prefill(&boundary,0);
    let p=boundary.len();let token=m.forward_id(1000,p);
    let last=m.mtp_generate_step(token,p+1);
    ensure!(last.len()==1,"direct MTP must fall back at last context position");
    println!("PASS: serving, forced branches, callback/EOS, request reset, sampling bypass, and context boundary; MTP present={has}");
    Ok(())
}
