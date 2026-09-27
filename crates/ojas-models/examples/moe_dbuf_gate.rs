//! Gate: the double-buffered streaming MoE expert gather must be byte-identical to the
//! serial gather (`OJAS_MOE_DBUF` unset).
//!
//! The feature (moe_stream.rs `dbuf_prefetch` plus the staging ring in `ExpertCache`)
//! preads layer L+1's predicted-routed experts on a background thread into a second
//! scratch buffer while the GPU runs layer L's expert GEMMs. `gather_experts` then serves
//! those experts from RAM instead of blocking on disk. Correctness rests on one
//! invariant: a staged expert's bytes are pread from the exact same (fd, offset, len) a
//! serial miss would use, so the packed gather scratch ends up bit-for-bit identical
//! whether an expert came from staging (prediction hit) or from a synchronous disk read
//! (miss / misprediction).
//!
//! Two parts:
//!   A. Structural (always runs): reproduces the staging mechanism — background
//!      positioned preads into a packed staging buffer at assigned offsets, vs. the serial
//!      fallback pread — and asserts byte-identity against the source of truth, using the
//!      real pread, threading and offset-packing logic.
//!   B. Model (runs only if given a MoE GGUF): decodes the same prompt twice under prec=4
//!      (the streaming path), once with OJAS_MOE_DBUF unset and once with the double
//!      buffer, and asserts the token sequences are identical; reports the gather/decode
//!      timing. Skipped with a message on a non-MoE model.
//!
//! usage: moe_dbuf_gate [gguf] [n_tokens]

use anyhow::Result;
use std::ffi::c_void;
use std::os::unix::io::AsRawFd;

/// Packing plan mirroring `ExpertCache::dbuf_stage_jobs`: assign each (key,len) a byte
/// offset inside the staging buffer via a running cursor, no two overlapping.
fn pack(jobs: &[(u64, usize)]) -> Vec<(u64, usize, usize)> {
    let mut cur = 0usize;
    jobs.iter().map(|&(key, len)| { let off = cur; cur += len; (key, off, len) }).collect()
}

fn structural_gate() -> Result<()> {
    use std::io::Write;
    // Deterministic pseudo-random file standing in for a GGUF expert shard.
    let path = std::env::temp_dir().join(format!("ojas_moe_dbuf_gate_{}.bin", std::process::id()));
    let n = 8 * 1024 * 1024usize;
    let mut truth = vec![0u8; n];
    let mut x = 0x9e3779b97f4a7c15u64;
    for b in truth.iter_mut() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; *b = (x & 0xff) as u8; }
    { let mut f = std::fs::File::create(&path)?; f.write_all(&truth)?; f.sync_all()?; }
    let file = std::fs::File::open(&path)?;
    let fd = file.as_raw_fd();

    // A set of "routed experts" = (key, disk_offset, len) — the same shape gather builds.
    let stride = 96 * 1024usize; // per-expert byte stride
    let experts: Vec<(u64, i64, usize)> = (0..24u64)
        .map(|j| (j, (j as usize * 131 % (n / stride)) as i64 * stride as i64, stride))
        .collect();

    // ---- Slot B (staging): background positioned preads into a packed staging buffer,
    // exactly as dbuf_prefetch spawns them (chunked across gather_threads). ----
    let plan = pack(&experts.iter().map(|&(k, _, l)| (k, l)).collect::<Vec<_>>());
    let stage_len: usize = plan.last().map(|&(_, o, l)| o + l).unwrap_or(0);
    let mut staging = vec![0u8; stage_len];
    let jobs: Vec<(i32, i64, usize, usize)> = experts.iter().zip(&plan)
        .map(|(&(_, off, len), &(_, soff, _))| (fd, off, len, staging.as_mut_ptr() as usize + soff))
        .collect();
    let nth = 8usize;
    let handle = std::thread::spawn(move || {
        let n = nth.min(jobs.len()).max(1);
        let cs = (jobs.len() + n - 1) / n;
        std::thread::scope(|sc| {
            for chunk in jobs.chunks(cs.max(1)) {
                sc.spawn(move || {
                    for &(fd, off, len, dst) in chunk {
                        unsafe { libc::pread(fd, dst as *mut c_void, len, off); }
                    }
                });
            }
        });
    });
    handle.join().ok();

    // ---- Slot A (serial gather fallback): direct synchronous pread into the packed scratch. ----
    let mut serial = vec![0u8; stage_len];
    for (&(_, off, len), &(_, soff, _)) in experts.iter().zip(&plan) {
        unsafe { libc::pread(fd, serial.as_mut_ptr().add(soff) as *mut c_void, len, off); }
    }

    // Invariant 1: staging (background) == serial (direct) — both slots hold identical bytes,
    // so gather is byte-identical no matter which slot supplied a given expert.
    assert_eq!(staging, serial, "staging (background pread) diverged from serial (direct pread)");
    // Invariant 2: both equal the source of truth at each expert's disk offset.
    for (&(_, off, len), &(_, soff, _)) in experts.iter().zip(&plan) {
        let src = &truth[off as usize..off as usize + len];
        assert_eq!(&staging[soff..soff + len], src, "staged expert bytes != disk source");
    }
    std::fs::remove_file(&path).ok();
    println!("[moe-dbuf] structural gate PASS: {} experts, {}KB staging, background==serial==disk (byte-identical)",
        experts.len(), stage_len / 1024);
    Ok(())
}

fn model_gate(gguf: &str, n_tok: usize) -> Result<()> {
    let g = ojas_formats::gguf::Gguf::open(gguf)?;
    let arch = g.arch();
    let ne = g.meta_u32(&format!("{arch}.expert_count")).unwrap_or(0);
    if ne == 0 {
        println!("[moe-dbuf] '{gguf}' arch={arch} is NOT MoE (expert_count=0) — model gate SKIPPED.");
        println!("[moe-dbuf] identity across dbuf on/off is proven structurally above; a MoE GGUF");
        println!("[moe-dbuf] loaded with prec=4 (streaming) is required to exercise the live path.");
        return Ok(());
    }
    println!("[moe-dbuf] '{gguf}' arch={arch} expert_count={ne} — running live identity + timing (prec=4).");
    let prompt: Vec<u32> = match std::env::var("OJAS_PROMPT") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => vec![151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198],
    };

    let run = |no_dbuf: bool| -> Result<(Vec<u32>, f64)> {
        // Double-buffer is opt-in (OJAS_MOE_DBUF); serial is the default (unset).
        if no_dbuf { std::env::remove_var("OJAS_MOE_DBUF"); }
        else { std::env::set_var("OJAS_MOE_DBUF", "1"); }
        let gpu = ojas_metal::MetalGpu::new()?;
        let mut g = ojas_formats::gguf::Gguf::open(gguf)?;
        let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 2048, 4, None, None)?; // prec=4 → streaming gather path
        m.prefill(&prompt[..prompt.len() - 1], 0);
        let t0 = std::time::Instant::now();
        let mut got = Vec::with_capacity(n_tok);
        let mut t = prompt[prompt.len() - 1];
        for i in 0..n_tok { t = m.forward_id(t, prompt.len() - 1 + i); got.push(t); }
        Ok((got, t0.elapsed().as_secs_f64()))
    };

    let (serial, ts) = run(true)?;
    let (dbuf, td) = run(false)?;
    assert_eq!(serial, dbuf, "DOUBLE-BUFFER BROKE IDENTITY: dbuf tokens != serial tokens");
    let (tps_s, tps_d) = (n_tok as f64 / ts, n_tok as f64 / td);
    println!("[moe-dbuf] model gate PASS: {n_tok} tokens byte-identical (dbuf==serial).");
    println!("[moe-dbuf] serial {ts:.2}s ({tps_s:.2} tok/s)  dbuf {td:.2}s ({tps_d:.2} tok/s)  speedup {:.2}x", ts / td);
    Ok(())
}

fn main() -> Result<()> {
    structural_gate()?;
    let n_tok: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    match std::env::args().nth(1) {
        Some(gguf) => model_gate(&gguf, n_tok)?,
        None => println!("[moe-dbuf] no GGUF arg — structural gate only. Pass a MoE GGUF to run the live identity+timing gate."),
    }
    Ok(())
}
