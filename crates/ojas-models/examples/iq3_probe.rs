//! Times the IQ3_XXS matvec against Q4L's, to see whether IQ3's byte saving survives
//! its dequantization cost.
//!
//! Decode is limited by getting weights out of memory, and Q4L costs 5.0 bits/weight
//! against IQ3_XXS's 3.06 (98 bytes per 256 weights) — 1.63x fewer bits. Measuring Q4L
//! against Q8 on a real model showed decode is byte-sensitive (1.7x fewer bits bought
//! 1.29x speed), so on bytes alone IQ3 should be worth something. Against that, native
//! Q6_K read fewer bytes than the Q8 it replaced and ran slower, because unpacking cost
//! more than the bytes saved; IQ3's dequant is heavier still (a codebook lookup per
//! group of 8 weights plus sign/scale unpacking).
//!
//! Both kernels run on the same (K,N) with the same activations, reported as Gweights/s,
//! the metric that makes formats comparable. Throughput only: the weight bytes are
//! random, so the outputs are garbage and only the timing means anything.
//!
//! usage: iq3_probe [K] [N]

use anyhow::Result;
use metal::{MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use std::ffi::c_void;

fn main() -> Result<()> {
    let k: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(11008);
    let n: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    assert!(k % 256 == 0, "IQ3_XXS blocks are 256 weights");
    let gpu = ojas_metal::MetalGpu::new()?;

    let iq_src = ojas_metal::kernels::family_source("moe_iq")
        .or_else(|| Some(ojas_metal::kernels::moe_iq::MOE_IQ_KERNELS)).unwrap();
    let gemv_src = ojas_metal::kernels::family_source("gemv").unwrap();
    let p_iq = gpu.compile(iq_src, "moe_down_iq3xxs")?;
    let p_q4l = gpu.compile(gemv_src, "gemv_q4l")?;

    let dev = &gpu.device;
    let mk = |bytes: usize| dev.new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared);

    // COPIES so the weight stream exceeds cache: re-dispatching one matrix leaves
    // it resident and flatters both kernels equally but measures the wrong thing.
    const COPIES: usize = 6;
    let rb_iq = (k as usize / 256) * 98;             // IQ3_XXS row bytes
    let rb_q4 = k as usize / 2;                      // Q4L nibble row bytes
    let iq_w: Vec<_> = (0..COPIES).map(|_| mk(n as usize * rb_iq)).collect();
    let q4_w: Vec<_> = (0..COPIES).map(|_| mk(n as usize * rb_q4)).collect();
    let q4_a: Vec<_> = (0..COPIES).map(|_| mk(n as usize * (k as usize / 32) * 2)).collect();
    let q4_b: Vec<_> = (0..COPIES).map(|_| mk(n as usize * (k as usize / 32) * 2)).collect();
    let act = mk(k as usize * 4);
    let out = mk(n as usize * 4);
    let shx = mk(n as usize * 4);
    let shg = mk(4);
    let idx = mk(4);
    let wgt = mk(4);
    unsafe {
        *(wgt.contents() as *mut f32) = 1.0;
        *(idx.contents() as *mut u32) = 0;
        // plausible activations; weights stay random bytes (timing only)
        let a = std::slice::from_raw_parts_mut(act.contents() as *mut f32, k as usize);
        for (i, v) in a.iter_mut().enumerate() { *v = ((i % 17) as f32 - 8.0) * 0.05; }
    }

    let time = |label: &str, bits_per_w: f64, f: &dyn Fn(&metal::ComputeCommandEncoderRef, usize)| {
        for _ in 0..2 {
            let cb = gpu.command_buffer();
            let e = cb.new_compute_command_encoder();
            for i in 0..COPIES { f(&e, i); }
            e.end_encoding(); cb.commit(); cb.wait_until_completed();
        }
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let cb = gpu.command_buffer();
            let e = cb.new_compute_command_encoder();
            for i in 0..COPIES { f(&e, i); }
            e.end_encoding(); cb.commit(); cb.wait_until_completed();
            let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
            best = best.min((ge - gs) * 1e3);
        }
        let per = best / COPIES as f64;                       // ms per matvec
        let weights = k as f64 * n as f64;
        let gw = weights / (per / 1e3) / 1e9;                 // Gweights/s
        let gbs = weights * bits_per_w / 8.0 / (per / 1e3) / 1e9;
        println!("  {label:<16} {per:7.4} ms   {gw:6.1} Gweights/s   {gbs:6.1} GB/s   ({bits_per_w:.2} bit/w)");
        gw
    };

    println!("IQ3_XXS vs Q4L, K={k} N={n}, {COPIES} distinct weight sets:");
    // Sweep the threadgroup size: this kernel does one row per simdgroup, the shape that
    // held Q4_K at 39 tok/s until lanes were made to cooperate inside a block (63.5
    // tok/s). The sweep shows whether IQ3 is occupancy-limited rather than ALU-limited.
    let mut gw_iq = 0.0f64;
    for ts_iq in [64u64, 128, 256, 512] {
    let rows_iq = (ts_iq / 32) as u32;                        // 1 row per simdgroup
    let g = time(&format!("iq3xxs ts={ts_iq}"), 98.0 * 8.0 / 256.0, &|e, i| {
        e.set_compute_pipeline_state(&p_iq);
        e.set_buffer(0, Some(&act), 0);
        e.set_buffer(1, Some(&iq_w[i]), 0);
        e.set_buffer(2, Some(&out), 0);
        e.set_bytes(3, 4, &k as *const u32 as *const c_void);
        e.set_bytes(4, 4, &n as *const u32 as *const c_void);
        e.set_buffer(6, Some(&idx), 0);
        e.set_buffer(7, Some(&wgt), 0);
        let ksel = 1u32;
        e.set_bytes(8, 4, &ksel as *const u32 as *const c_void);
        e.set_buffer(9, Some(&shx), 0);
        e.set_buffer(10, Some(&shg), 0);
        e.dispatch_thread_groups(MTLSize::new(((n + rows_iq - 1) / rows_iq) as u64, 1, 1),
                                 MTLSize::new(ts_iq, 1, 1));
    });
    if g > gw_iq { gw_iq = g; }
    }

    let ts_q4 = 256u64;
    let rows_q4 = (ts_q4 / 32 * 4) as u32;                    // 4 rows per simdgroup
    let gw_q4 = time("gemv_q4l", 160.0 * 8.0 / 256.0, &|e, i| {
        e.set_compute_pipeline_state(&p_q4l);
        e.set_buffer(0, Some(&act), 0);
        e.set_buffer(1, Some(&q4_w[i]), 0);
        e.set_buffer(2, Some(&out), 0);
        e.set_bytes(3, 4, &k as *const u32 as *const c_void);
        e.set_bytes(4, 4, &n as *const u32 as *const c_void);
        e.set_buffer(5, Some(&q4_a[i]), 0);
        e.set_buffer(6, Some(&q4_b[i]), 0);
        e.dispatch_thread_groups(MTLSize::new(((n + rows_q4 - 1) / rows_q4) as u64, 1, 1),
                                 MTLSize::new(ts_q4, 1, 1));
    });

    println!("\n  IQ3 reads {:.2}x fewer bits/weight than Q4L", 160.0 / 98.0);
    println!("  IQ3 runs  {:.2}x Q4L's weights/second", gw_iq / gw_q4);
    // Break-even is not byte-adjusted: both formats hold the same number of weights and
    // matvec time is weights / (weights per second), so one format is faster than the
    // other if and only if its Gweights/s is higher. The byte advantage is not a separate
    // credit to add on top; it is the reason to expect a higher Gw/s.
    println!("  break-even is simply Q4L's {gw_q4:.0} Gw/s (same weight count, so");
    println!("  bytes only matter through the rate they produce). IQ3 needs {:+.0}%.",
        (gw_q4 / gw_iq - 1.0) * 100.0);
    println!("  Its DRAM rate is a third of Q4L's, so it is dequant-bound, not");
    println!("  memory-bound: the byte saving has nothing to convert into speed.");
    Ok(())
}
