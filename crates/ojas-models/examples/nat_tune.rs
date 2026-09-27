//! Per-kernel tuning sweep for the native quant matvecs.
//!
//! The `nr0`/`nsg` table in `quant_src` was seeded from the reference N_R0_*/N_SG_*
//! constants, which were measured for a different kernel on different hardware. This
//! repo's work assignment and staging differ, and one of those constants has already
//! turned out to be worth nothing here, so the table has to be measured rather than
//! inherited.
//!
//! Sweeps rows-per-simdgroup (NR, a compile-time constant, hence the `_r*` kernel
//! variants) against threadgroup size, on a shape taken from a real FFN. Reports
//! Gweights/s, which keeps formats comparable where bytes/s flatters the high-bit
//! formats.
//!
//! Timing is best-of-N on GPU timestamps over COPIES distinct weight buffers, so the
//! matrix is not served from cache. Weight bytes are random: the outputs are
//! meaningless and only the timing is read.
//!
//! usage: OJAS_TUNE_KERNELS=1 nat_tune [K] [N] [format-tag]

use anyhow::Result;
use metal::{MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device as _;
use std::ffi::c_void;

fn main() -> Result<()> {
    if std::env::var("OJAS_TUNE_KERNELS").is_err() {
        eprintln!("set OJAS_TUNE_KERNELS=1 (the _r* variants are not emitted otherwise)");
        std::process::exit(2);
    }
    let k: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(5120);
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let only = std::env::args().nth(3);
    let gpu = ojas_metal::MetalGpu::new()?;
    let dev = &gpu.device;
    const COPIES: usize = 4;

    let xs: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
    let xbuf = dev.new_buffer_with_data(
        xs.as_ptr() as *const c_void, (k * 4) as u64, MTLResourceOptions::StorageModeShared);
    let ybuf = dev.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);

    // OJAS_TUNE_M sweeps the batched kernel instead: the same formats at a given M,
    // against the M=1 decode kernel, which gives the per-row cost of batching.
    let sweep_m: Option<u32> = std::env::var("OJAS_TUNE_M").ok().and_then(|v| v.parse().ok());
    println!("  K={k} N={n}, {COPIES} distinct weight sets, best-of-7 GPU timestamps");
    if let Some(mv) = sweep_m {
        println!("  batched sweep at M={mv}: first column is the M=1 decode kernel in ms,");
        println!("  then each variant as a MULTIPLE of it. Below {mv:.2}x means batching wins.");
        for (i, c) in ojas_metal::kernels::nat::M_VARIANTS.iter().enumerate() {
            let (st, ch, mt, nr) = *c;
            let kind = match st {
                1 => format!("staged chunk={ch} tgmem={}KB", ch * mt * 4 / 1024),
                2 => "device-read, x in registers".to_string(),
                3 => "walker-fused (M=1 body, MTILE rows)".to_string(),
                _ => "device-read generic".to_string(),
            };
            let nrs = if st == 0 { "1".to_string() }
                      else if st == 3 && nr == 0 { "format".to_string() }
                      else { nr.to_string() };
            println!("    v{i}: {kind}, Mtile={mt}, rows/sg={nrs}");
        }
        let hdr: String = (0..ojas_metal::kernels::nat::M_VARIANTS.len())
            .map(|i| format!("{:>9}", format!("v{i}"))).collect();
        println!("  {:<9}{:>6}{:>9}{hdr}", "format", "bpw", "M=1 ms");
    } else {
    println!("  {:<9}{:>6}{:>6}{:>9}{:>11}{:>10}{:>9}", "format", "bpw", "NR", "threads", "ms", "Gw/s", "GB/s");
    }
    let mut table: Vec<(String, u32, u32, f64)> = vec![];

    for f in ojas_metal::kernels::nat::FORMATS {
        if only.as_deref().is_some_and(|tag| tag != f.tag) { continue; }
        let (bb, wp) = (f.block_bytes as usize, f.weights as usize);
        if k % wp != 0 { continue; }
        let nblk = k / wp;
        let rowb = nblk * bb;
        let ws: Vec<_> = (0..COPIES)
            .map(|_| dev.new_buffer((n * rowb) as u64, MTLResourceOptions::StorageModeShared))
            .collect();
        // Every byte must be pseudo-random. Filling sparsely left the buffers ~98%
        // zeros, and the values are not inert: zero grid indices, zero nibbles and zero
        // hmask bits make every data-dependent branch in a decoder take the same path,
        // so ALU cost came out uniform and moved between runs with whichever bytes
        // happened to be non-zero — two unchanged kernels swung +41% and -49% across
        // runs before this fix. Real weights are dense and varied.
        for w in &ws {
            let p = w.contents() as *mut u8;
            let mut st: u32 = 0x9E37_79B9;
            for i in 0..n * rowb {
                st ^= st << 13; st ^= st >> 17; st ^= st << 5;
                unsafe { *p.add(i) = (st >> 24) as u8 };
            }
        }
        let bpw = bb as f64 * 8.0 / wp as f64;
        if let Some(mv) = sweep_m {
            // The M=1 decode kernel as the reference, then every batched variant.
            // Per-token cost decides whether speculation can pay: a batched kernel
            // earns its keep only if (M kernel)/(M=1 kernel) < M.
            let mut row = format!("  {:<9}{:>6.2}", f.tag, bpw);
            let mut ref_ms = f64::NAN;
            let nvar = ojas_metal::kernels::nat::M_VARIANTS.len();
            for vi in 0..=nvar {
                let (entry, mm, rows) = if vi == 0 {
                    (format!("gemv_nat_{}", f.tag), None,
                     ojas_metal::kernels::nat::nat_launch(f.ty).map(|p| p.1).unwrap_or(8))
                } else {
                    let cfg = ojas_metal::kernels::nat::M_VARIANTS[vi - 1];
                    (format!("gemv_nat_{}_m_v{}", f.tag, vi - 1), Some(mv),
                     ojas_metal::kernels::nat::m_launch(f, cfg).1)
                };
                let Some(src) = ojas_metal::kernels::source_of(&entry) else { row.push_str(&format!("{:>9}", "-")); continue };
                let Ok(pipe) = gpu.pipeline(src, &entry) else { row.push_str(&format!("{:>9}", "x")); continue };
                let go = || {
                    let cb = gpu.command_buffer();
                    let e = cb.new_compute_command_encoder();
                    for c in 0..COPIES {
                        e.set_compute_pipeline_state(&pipe);
                        e.set_buffer(0, Some(&xbuf), 0);
                        e.set_buffer(1, Some(&ws[c]), 0);
                        e.set_buffer(2, Some(&ybuf), 0);
                        let (ku, nu) = (k as u32, n as u32);
                        e.set_bytes(3, 4, &ku as *const u32 as *const c_void);
                        e.set_bytes(4, 4, &nu as *const u32 as *const c_void);
                        if let Some(v) = mm {
                            let acc = 0u32;
                            e.set_bytes(7, 4, &v as *const u32 as *const c_void);
                            e.set_bytes(8, 4, &acc as *const u32 as *const c_void);
                        }
                        e.dispatch_thread_groups(
                            MTLSize::new(((n as u32 + rows - 1) / rows) as u64, 1, 1),
                            MTLSize::new(256, 1, 1));
                    }
                    e.end_encoding(); cb.commit(); cb.wait_until_completed();
                    let (gs, ge): (f64, f64) = unsafe {
                        (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
                    (ge - gs) * 1e3
                };
                go(); go();
                let mut ms = f64::INFINITY;
                for _ in 0..7 { ms = ms.min(go()); }
                let ms = ms / COPIES as f64;
                if vi == 0 { ref_ms = ms; row.push_str(&format!("{ms:>9.4}")); }
                else { row.push_str(&format!("{:>9.2}", ms / ref_ms)); }
            }
            println!("{row}");
            continue;
        }
        let mut best = (f64::INFINITY, 0u32, 0u32);
        for nr in [1u32, 2, 4, 8] {
            let entry = format!("gemv_nat_{}_r{nr}", f.tag);
            let Some(src) = ojas_metal::kernels::source_of(&entry) else { continue };
            let Ok(pipe) = gpu.pipeline(src, &entry) else { continue };
            let maxt = pipe.max_total_threads_per_threadgroup() as u32;
            for t in [32u32, 64, 128, 256] {
                if t > maxt { continue; }
                // NAT_GEMV_Q80 splits K across SIMDgroups while every threadgroup still
                // owns exactly NR rows; other native bodies assign NR rows per
                // SIMDgroup. Using their grid for Q8 silently skips outputs.
                if f.ty == 8 && t > 128 { continue; } // Q8 reduction has four SG slots.
                let rows = if f.ty == 8 { nr } else { (t / 32).max(1) * nr };
                let go = || {
                    let cb = gpu.command_buffer();
                    let e = cb.new_compute_command_encoder();
                    for c in 0..COPIES {
                        e.set_compute_pipeline_state(&pipe);
                        e.set_buffer(0, Some(&xbuf), 0);
                        e.set_buffer(1, Some(&ws[c]), 0);
                        e.set_buffer(2, Some(&ybuf), 0);
                        let (ku, nu) = (k as u32, n as u32);
                        e.set_bytes(3, 4, &ku as *const u32 as *const c_void);
                        e.set_bytes(4, 4, &nu as *const u32 as *const c_void);
                        e.dispatch_thread_groups(
                            MTLSize::new(((n as u32 + rows - 1) / rows) as u64, 1, 1),
                            MTLSize::new(t as u64, 1, 1));
                    }
                    e.end_encoding(); cb.commit(); cb.wait_until_completed();
                    let (gs, ge): (f64, f64) = unsafe {
                        (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
                    (ge - gs) * 1e3
                };
                go(); go();
                let mut ms = f64::INFINITY;
                for _ in 0..7 { ms = ms.min(go()); }
                let per = ms / COPIES as f64;
                if per < best.0 { best = (per, nr, t); }
            }
        }
        let (per, nr, t) = best;
        let gw = (k * n) as f64 / (per / 1e3) / 1e9;
        let gbs = gw * bpw / 8.0;
        println!("  {:<9}{:>6.2}{:>6}{:>9}{:>11.4}{:>10.1}{:>9.0}", f.tag, bpw, nr, t, per, gw, gbs);
        table.push((f.tag.into(), nr, t / 32, gw));
    }

    // Controls: formats whose kernels did not change between runs. If these move, the
    // machine moved and nothing else in the table means anything — this sweep has read
    // 413 and 328 Gw/s for the same unchanged kernel minutes apart.
    println!("\n  CONTROLS (unchanged kernels — compare against a previous run):");
    for (tag, _, _, gw) in table.iter().filter(|(t, ..)| t == "iq1s" || t == "iq2xxs" || t == "q80") {
        println!("    {tag:<9} {gw:7.1} Gw/s");
    }

    println!("\n  best (nr0, nsg) per format — paste into quant_src::FORMATS:");
    for (tag, nr, nsg, _) in &table {
        println!("    {tag:<9} nr0: {nr}, nsg: {nsg}");
    }
    Ok(())
}
