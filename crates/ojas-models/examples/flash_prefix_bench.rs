//! Interleaved Flash prefix-acceptance and snapshot-cost experiment.
//! usage: OJAS_MTP_PREFIX=1 OJAS_MTP_DRAFT=3 flash_prefix_bench model reference.ids [draft|oracle] [reps=3] [widths=2,3,4]
use anyhow::{ensure, Result};
use ojas_core::Model;
use ojas_models::decoder::DecoderGpu;
use std::time::Instant;

fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() >= 3, "need model and reference.ids");
    let mode = a.get(3).map(String::as_str).unwrap_or("draft");
    ensure!(
        ["draft", "oracle"].contains(&mode),
        "mode must be draft or oracle"
    );
    let oracle = mode == "oracle";
    let cfg = ojas_core::config::EngineConfig::current();
    ensure!(
        cfg.mtp_prefix && cfg.no_spec == oracle,
        "set OJAS_MTP_PREFIX=1; set OJAS_NO_SPEC only for oracle"
    );
    let reps: usize = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(3);
    ensure!((3..=20).contains(&reps), "reps must be 3..20");
    let widths: Vec<usize> = a
        .get(5)
        .map(String::as_str)
        .unwrap_or("2,3,4")
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(
        !widths.is_empty() && widths.iter().all(|w| (2..=4).contains(w)),
        "widths must be 2..4"
    );
    let cases: Vec<_> = widths
        .iter()
        .flat_map(|&w| [(w, false), (w, true)])
        .collect();
    let ids: Vec<u32> = std::fs::read_to_string(&a[2])?
        .split_whitespace()
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(ids.len() > 1, "missing reference IDs");
    let n = ids[0] as usize;
    let tokens = &ids[1..];
    ensure!(n > 0 && n + 32 <= tokens.len(), "need 32 reference outputs");
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&a[1])?;
    ensure!(g.arch() == "qwen4exp", "Flash only");
    let mut m = DecoderGpu::load(&gpu, &mut g, 512, 4, None, None)?;
    ensure!(
        m.has_mtp() && n + 36 <= m.context_capacity(),
        "MTP and sufficient context required"
    );
    eprintln!("snapshot_bytes={}", m.mtp_snapshot_bytes());
    for rep in 0..=reps {
        for k in 0..cases.len() {
            let (width, prefix) = cases[(k + rep.saturating_sub(1)) % cases.len()];
            m.flash_prefix_probe_mode(prefix);
            m.reset_session();
            m.prefill(&tokens[..n - 1], 0);
            m.gather_stats_reset();
            let load = ojas_models::bench::load_average();
            let (mut emitted, mut calls, mut partial_calls, mut kept_drafts, mut hrow) =
                (0, 0, 0, 0, 0);
            let (mut target_s, mut catchup_s, mut draft_s, mut rollback_s) = (0.0, 0.0, 0.0, 0.0);
            let start = Instant::now();
            while emitted < 32 {
                let pos = n - 1 + emitted;
                let count = width.min(32 - emitted);
                let mut batch = vec![tokens[pos]];
                if oracle {
                    batch = tokens[pos..pos + count].to_vec();
                } else {
                    let t = Instant::now();
                    for j in 0..width - 1 {
                        let id = if emitted == 0 && j == 0 { m.flash_draft_current(*batch.last().unwrap(), pos) }
                            else { m.flash_draft_probe(*batch.last().unwrap(), pos + j, hrow, j > 0) };
                        ensure!(id != u32::MAX, "draft unavailable");
                        batch.push(id);
                    }
                    draft_s += t.elapsed().as_secs_f64();
                }
                let (mut out, ts, cs) = m.flash_verify_probe(&batch, pos);
                target_s += ts;
                catchup_s += cs;
                calls += 1;
                if !oracle {
                    let matched = batch[1..]
                        .iter()
                        .zip(&out)
                        .take_while(|(d, g)| d == g)
                        .count();
                    if matched == width - 1 {
                        hrow = matched;
                        kept_drafts += matched;
                    } else {
                        let row = if prefix { matched } else { 0 };
                        let t = Instant::now();
                        m.mtp_rollback_to(row);
                        rollback_s += t.elapsed().as_secs_f64();
                        hrow = row;
                        kept_drafts += row;
                        partial_calls += usize::from(row > 0);
                        out.truncate(row + 1);
                    }
                }
                let take = out.len().min(32 - emitted);
                ensure!(
                    out[..take] == tokens[pos + 1..pos + 1 + take],
                    "reference mismatch rep={rep} prefix={prefix} width={width} output={emitted}"
                );
                emitted += take;
            }
            let seconds = start.elapsed().as_secs_f64();
            let (hits, reads, cache) = m.gather_stats();
            if rep > 0 {
                println!("{{\"rep\":{rep},\"mode\":\"{mode}\",\"prefix\":{prefix},\"width\":{width},\"seconds\":{seconds},\"tps\":{},\"target_s\":{target_s},\"catchup_s\":{catchup_s},\"draft_s\":{draft_s},\"rollback_s\":{rollback_s},\"calls\":{calls},\"partial_calls\":{partial_calls},\"kept_drafts\":{kept_drafts},\"hits\":{hits},\"reads\":{reads},\"cache_bytes\":{cache},\"load_average\":{}}}", 32.0/seconds, load.map(|x|x.to_string()).unwrap_or("null".into()));
            }
        }
    }
    Ok(())
}
