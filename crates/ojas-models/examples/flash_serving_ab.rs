//! Repeated real-serving measurements with fresh recurrent state and output checks.
//! usage: flash_serving_ab model prompt-file plain|mtp [tokens=32] [reps=3]
use anyhow::{ensure, Result};
use ojas_core::Model;
use ojas_models::decoder::DecoderGpu;
use std::{cell::Cell, time::Instant};
use std::io::Write;

struct Observed<'a> { m: &'a DecoderGpu<'a>, calls: &'a Cell<usize>, accepted: &'a Cell<usize> }
impl Model for Observed<'_> {
    fn context_capacity(&self) -> usize { self.m.context_capacity() }
    fn mtp_verify_width(&self) -> usize { self.m.mtp_verify_width() }
    fn n_layers(&self) -> usize { self.m.n_layers() }
    fn hidden_dim(&self) -> usize { self.m.hidden_dim() }
    fn prefill(&self, t: &[u32], p: usize) { self.m.prefill(t,p) }
    fn reuse_prefix_len(&self, t: &[u32]) -> usize { self.m.reuse_prefix_len(t) }
    fn forward_id(&self, t: u32, p: usize) -> u32 { self.m.forward_id(t,p) }
    fn has_mtp(&self) -> bool { self.m.has_mtp() }
    fn mtp_step_committed(&self, t: u32, p: usize) -> Option<Vec<u32>> {
        let out=self.m.mtp_step_committed(t,p)?;
        self.calls.set(self.calls.get()+1);
        self.accepted.set(self.accepted.get()+usize::from(out.len()>1));
        Some(out)
    }
}
fn main() -> Result<()> {
    let a:Vec<_>=std::env::args().collect();
    ensure!(a.len()>=4,"usage: flash_serving_ab model prompt-file plain|mtp [tokens] [reps]");
    let plain=a[3]=="plain";
    ensure!(plain || a[3]=="mtp","mode must be plain or mtp");
    ensure!(ojas_core::config::EngineConfig::current().no_spec==plain,"plain requires OJAS_NO_SPEC=1; mtp requires it unset");
    let count:usize=a.get(4).map(|s|s.parse()).transpose()?.unwrap_or(32);
    let reps:usize=a.get(5).map(|s|s.parse()).transpose()?.unwrap_or(3);
    ensure!(count>=2 && count<=256 && (3..=20).contains(&reps),"use 2..256 output tokens and 3..20 repetitions");
    let start=Instant::now();
    let gpu=ojas_metal::MetalGpu::new()?;
    let mut g=ojas_formats::gguf::Gguf::open(&a[1])?;
    let text=std::fs::read_to_string(&a[2])?;
    let prompt:Vec<u32>=ojas_tokenize::Bpe::from_gguf(&g).encode(&text).into_iter().map(|v|v as u32).collect();
    ensure!(!prompt.is_empty(),"empty prompt");
    let mut m=DecoderGpu::load(&gpu,&mut g,512.max(prompt.len()+count),4,None,None)?;
    ensure!(prompt.len()+count<=m.context_capacity(),"workload exceeds context");
    ensure!(plain || m.has_mtp(),"MTP fixture has no usable draft");
    println!("{{\"load_s\":{},\"prompt_tokens\":{:?},\"mode\":\"{}\",\"precision\":4,\"warmups\":1}}",start.elapsed().as_secs_f64(),prompt,a[3]);
    let calls=Cell::new(0);let accepted=Cell::new(0);
    ensure!(ojas_core::config::EngineConfig::current().flash_direct_experts,"load with OJAS_FLASH_DIRECT_EXPERTS=1");
    // Access counters through the wrapper reference retained outside EngineCore.
    // Output equality against the warmup guards repeated-run state contamination.
    let mut reference=None;
    for run in 0..=reps {
      for direct in if run % 2 == 0 { [false, true] } else { [true, false] } {
        m.flash_direct_probe_mode(direct);
        let engine=ojas_infer::EngineCore::new(Observed{m:&m,calls:&calls,accepted:&accepted});
        m.reset_session();calls.set(0);accepted.set(0);
        m.gather_stats_reset(); m.flash_trace_begin();
        let load=ojas_models::bench::load_average();
        let start=Instant::now();let mut first=None;
        let out=engine.generate_with(&prompt,count,None,&mut|_,_|{},&mut |_|{first.get_or_insert_with(||start.elapsed().as_secs_f64());true});
        let total=start.elapsed().as_secs_f64();
        let events=m.flash_trace_end();
        if run > 0 && direct {
            ensure!(events.iter().any(|e| e.target.direct_expert_bytes > 0), "direct expert path did not engage");
        }
        let ttft=first.ok_or_else(||anyhow::anyhow!("generation emitted no tokens"))?;
        ensure!(out.len()==count,"generation ended before requested output length");
        ensure!(plain == (calls.get()==0),"observed MTP engagement does not match benchmark mode");
        if let Some(want)=&reference {ensure!(&out==want,"repeated fresh run changed output");} else {reference=Some(out.clone());}
        let prefix=std::env::var("OJAS_TRACE_OUTPUT").expect("set OJAS_TRACE_OUTPUT to an output file prefix");
        let mut file=std::io::BufWriter::new(std::fs::File::create(format!("{prefix}-{direct}-{run}.jsonl"))?);
        for e in events {
            let t=e.target;
            writeln!(file,"{{\"name\":\"{}\",\"position\":{},\"rows\":{},\"start_s\":{},\"seconds\":{},\"hits\":{},\"lookups\":{},\"cache_bytes\":{},\"gpu_s\":{},\"gather_s\":{},\"copy_s\":{},\"read_s\":{},\"admit_s\":{},\"encode_s\":{},\"wait_s\":{},\"commands\":{},\"invalid_gpu_timestamps\":{},\"direct_expert_bytes\":{}}}",e.name,e.position,e.rows,e.start_s,e.seconds,e.hits,e.lookups,e.cache_bytes,t.gpu_s,t.gather_s,t.gather_copy_s,t.gather_read_s,t.gather_admit_s,t.encode_s,t.submit_wait_s,t.commands,t.invalid_gpu_timestamps,t.direct_expert_bytes)?;
        }
        file.flush()?;
        if run>0 {println!("{{\"run\":{run},\"direct\":{direct},\"total_s\":{total},\"ttft_s\":{ttft},\"decode_tps\":{},\"load_average\":{},\"mtp_calls\":{},\"accepted_calls\":{},\"output_tokens\":{:?}}}",(count-1) as f64/(total-ttft),load.map(|x|x.to_string()).unwrap_or("null".into()),calls.get(),accepted.get(),out);}
    }
    }
    Ok(())
}
