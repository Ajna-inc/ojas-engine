//! Fixed-reference oracle verification and forced-MTP cost/acceptance diagnostics.
//! usage: flash_mtp_cost model reference.ids oracle|maintained|draft [repetitions=3]
use anyhow::{ensure, Result};
use ojas_core::Model;
use ojas_models::decoder::DecoderGpu;
use std::time::Instant;
fn main() -> Result<()> {
    let a:Vec<_>=std::env::args().collect();
    ensure!(a.len()>=4,"usage: flash_mtp_cost model reference.ids oracle|maintained|draft [reps]");
    let oracle=a[3]!="draft";ensure!(["oracle","maintained","draft"].contains(&a[3].as_str()),"unknown mode");
    ensure!(ojas_core::config::EngineConfig::current().no_spec==(a[3]=="oracle"),"oracle requires OJAS_NO_SPEC=1; draft requires it unset");
    let reps:usize=a.get(4).map(|x|x.parse()).transpose()?.unwrap_or(3);ensure!((3..=20).contains(&reps),"use 3..20 repetitions");
    let ids:Vec<u32>=std::fs::read_to_string(&a[2])?.split_whitespace().map(str::parse).collect::<std::result::Result<_,_>>()?;
    ensure!(ids.len()>1,"missing reference inputs");let n=ids[0] as usize;let tokens=&ids[1..];
    ensure!(n>0 && tokens.len()>=n+32,"need prompt plus 32 reference continuation tokens");
    let gpu=ojas_metal::MetalGpu::new()?;let mut g=ojas_formats::gguf::Gguf::open(&a[1])?;
    ensure!(g.arch()=="qwen4exp","Flash-only diagnostic");
    let m=DecoderGpu::load(&gpu,&mut g,512,4,None,None)?;ensure!(m.has_mtp(),"draft weights required");
    ensure!(n+36<=m.context_capacity(),"fixture exceeds context");
    let widths:Vec<usize>=if a[3]=="oracle" {vec![1,2,3,4]} else {vec![2,3,4]};
    // One warmup per configuration, then rotate measured order to reduce order bias.
    for rep in 0..=reps {
      for k in 0..widths.len() {
        let width=widths[(k+rep.saturating_sub(1))%widths.len()];
        m.reset_session();m.prefill(&tokens[..n-1],0);m.gather_stats_reset();
        let load=ojas_models::bench::load_average();let start=Instant::now();
        let(mut emitted,mut calls,mut accepted,mut hrow)=(0,0,0,0);
        let(mut draft_s,mut verify_s,mut rollback_s)=(0.0,0.0,0.0);
        let(mut target_s,mut catchup_s)=(0.0,0.0);
        let mut eligible=vec![0usize;width-1];let mut correct=vec![0usize;width-1];
        while emitted<32 {
          let pos=n-1+emitted;let cur=tokens[pos];calls+=1;
          let got=if oracle {
            let count=width.min(32-emitted);let t=Instant::now();
            let out=if width==1 {let out=vec![m.forward_id(cur,pos)];target_s+=t.elapsed().as_secs_f64();out} else {let(out,ts,cs)=m.flash_verify_probe(&tokens[pos..pos+count],pos);target_s+=ts;catchup_s+=cs;out};
            verify_s+=t.elapsed().as_secs_f64();out
          } else {
            let mut batch=vec![cur];let t=Instant::now();
            for j in 0..width-1 {let id=if emitted==0 && j==0 {m.flash_draft_current(*batch.last().unwrap(),pos)} else {m.flash_draft_probe(*batch.last().unwrap(),pos+j,hrow,j>0)};ensure!(id!=u32::MAX,"draft unavailable");batch.push(id);}
            draft_s+=t.elapsed().as_secs_f64();let t=Instant::now();let(out,ts,cs)=m.flash_verify_probe(&batch,pos);target_s+=ts;catchup_s+=cs;verify_s+=t.elapsed().as_secs_f64();
            let mut all=true;
            for j in 0..width-1 {if all {eligible[j]+=1;if batch[j+1]==out[j] {correct[j]+=1;} else {all=false;}}}
            if all {accepted+=1;hrow=width-1;out} else {let t=Instant::now();m.mtp_rollback();rollback_s+=t.elapsed().as_secs_f64();hrow=0;vec![out[0]]}
          };
          let count=got.len().min(32-emitted);
          ensure!(got[..count]==tokens[pos+1..pos+1+count],"reference mismatch at output {emitted}, width {width}, rep {rep}");emitted+=count;
        }
        let elapsed=start.elapsed().as_secs_f64();let(hits,reads,cache)=m.gather_stats();
        if rep>0 {println!("{{\"rep\":{rep},\"mode\":\"{}\",\"width\":{width},\"tokens\":{emitted},\"seconds\":{elapsed},\"tps\":{},\"draft_s\":{draft_s},\"verify_s\":{verify_s},\"rollback_s\":{rollback_s},\"target_s\":{target_s},\"catchup_s\":{catchup_s},\"calls\":{calls},\"accepted_calls\":{accepted},\"conditional_eligible\":{eligible:?},\"conditional_correct\":{correct:?},\"gather_hits\":{hits},\"gather_reads\":{reads},\"cache_bytes\":{cache},\"load_average\":{}}}",a[3],32.0/elapsed,load.map(|x|x.to_string()).unwrap_or("null".into()));}
      }
    }
    Ok(())
}
