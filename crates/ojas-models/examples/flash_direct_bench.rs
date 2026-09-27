//! Fixed target inputs, same Metal cache, rotating direct-read/copy A/B. No drafting.
//! usage: OJAS_FLASH_DIRECT_EXPERTS=1 OJAS_NO_SPEC=1 flash_direct_bench model reference.ids [reps=3] [widths=2,4]
use anyhow::{ensure, Result};
use ojas_core::Model;
use ojas_models::decoder::{DecoderGpu, FlashTargetTiming};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() >= 3,
        "usage: flash_direct_bench model reference.ids [reps] [widths]"
    );
    ensure!(
        ojas_core::config::EngineConfig::current().no_spec,
        "set OJAS_NO_SPEC=1"
    );
    let reps: usize = a.get(3).map(|x| x.parse()).transpose()?.unwrap_or(3);
    ensure!((3..=20).contains(&reps), "reps must be 3..20");
    let widths: Vec<usize> = a
        .get(4)
        .map(String::as_str)
        .unwrap_or("2,4")
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(
        !widths.is_empty() && widths.iter().all(|w| (1..=16).contains(w)),
        "widths must be 1..16"
    );
    ensure!(
        a.len() <= 5,
        "usage: flash_direct_bench model reference.ids [reps] [widths]"
    );
    let cases: Vec<_> = widths
        .iter()
        .flat_map(|&w| [false, true].map(|v| (w, v)))
        .collect();
    let ids: Vec<u32> = std::fs::read_to_string(&a[2])?
        .split_whitespace()
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(ids.len() > 1, "missing IDs");
    let n = ids[0] as usize;
    let tokens = &ids[1..];
    ensure!(
        n > 0 && n + 32 <= tokens.len(),
        "need prompt and 32 continuation IDs"
    );
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&a[1])?;
    ensure!(g.arch() == "qwen4exp", "Flash only");
    let mut m = DecoderGpu::load(&gpu, &mut g, 512, 4, None, None)?;
    ensure!(n + 32 <= m.context_capacity(), "context too small");
    for rep in 0..=reps {
        for k in 0..cases.len() {
            let (width, direct) = cases[(k + rep.saturating_sub(1)) % cases.len()];
            m.flash_direct_probe_mode(direct);
            m.reset_session();
            m.prefill(&tokens[..n - 1], 0);
            m.gather_stats_reset();
            let load = ojas_models::bench::load_average();
            let mut total = FlashTargetTiming::default();
            let start = std::time::Instant::now();
            for emitted in (0..32).step_by(width) {
                let pos = n - 1 + emitted;
                let count = width.min(32 - emitted);
                let (out, s) = m.flash_target_profile(&tokens[pos..pos + count], pos, None);
                ensure!(
                    out == tokens[pos + 1..pos + 1 + count],
                    "reference mismatch rep={rep} width={width} position={pos}"
                );
                ensure!(s.invalid_gpu_timestamps == 0, "GPU timing unavailable");
                macro_rules! add {($($field:ident),*)=>{$(total.$field+=s.$field;)*}}
                add!(
                    wall_s,
                    gather_s,
                    gather_setup_s,
                    gather_copy_s,
                    direct_expert_bytes,
                    gather_read_s,
                    gather_admit_s,
                    encode_s,
                    submit_wait_s,
                    gpu_s,
                    commands,
                    invalid_gpu_timestamps
                );
            }
            let elapsed = start.elapsed().as_secs_f64();
            ensure!(!direct || total.direct_expert_bytes > 0, "direct reads did not engage");
            let (hits, reads, cache) = m.gather_stats();
            if rep > 0 {
                println!("{{\"rep\":{rep},\"direct\":{direct},\"width\":{width},\"seconds\":{elapsed},\"tps\":{},\"target_s\":{},\"gather_s\":{},\"gather_setup_s\":{},\"gather_copy_s\":{},\"gather_read_s\":{},\"gather_admit_s\":{},\"encode_s\":{},\"submit_wait_s\":{},\"gpu_s\":{},\"commands\":{},\"direct_expert_bytes\":{},\"hits\":{hits},\"reads\":{reads},\"cache_bytes\":{cache},\"load_average\":{}}}",32.0/elapsed,total.wall_s,total.gather_s,total.gather_setup_s,total.gather_copy_s,total.gather_read_s,total.gather_admit_s,total.encode_s,total.submit_wait_s,total.gpu_s,total.commands,total.direct_expert_bytes,load.map(|x|x.to_string()).unwrap_or("null".into()));
            }
        }
    }
    Ok(())
}
