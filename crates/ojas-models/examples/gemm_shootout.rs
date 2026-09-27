//! Times the dense-f16 prefill proposal against the Q4L GEMM before building it.
//!
//! The proposal: dequantize each layer's Q4L weights once per prefill into f16 scratch,
//! then run a dense f16xf16 GEMM with no in-loop dequant — the form the reference
//! ~90%-ALU numbers are quoted for. The counter-arithmetic: at M=256 this GEMM re-reads
//! each weight once per 32-wide M-tile (8x), so f16 doubles a 180 MB/layer weight stream
//! to 720, and a layer's f16 gate+up (90 MB) no longer fits the 48 MB SLC while its Q4L
//! nibbles (22.5 MB) do.
//!
//! This times gemm_mm_q4l against gemm_mm_f16 on the same logical matrix (ffn shape,
//! K=2048 N=11008 M=256), cycling 4 weight sets so each dispatch starts SLC-cold like a
//! real layer.
//!
//! usage: gemm_shootout
use anyhow::Result;
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device;

fn main() -> Result<()> {
    let gpu = ojas_metal::MetalGpu::new()?;
    let (k, n, m) = (2048usize, 11008usize, 256u32);
    let nblk = k / 32;
    let nsets = 4usize;

    // synthetic Q4L: random nibbles, mild scales
    let mut seed = 0xABCDEFu32;
    let mut rnd8 = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); (seed >> 13) as u8 };
    let mut q4sets = vec![];
    for _ in 0..nsets {
        let nib: Vec<u8> = (0..n * k / 2).map(|_| rnd8()).collect();
        let qa: Vec<u16> = (0..n * nblk).map(|_| half::f16::from_f32(0.01).to_bits()).collect();
        let qb: Vec<u16> = (0..n * nblk).map(|_| half::f16::from_f32(-0.08).to_bits()).collect();
        let cast = |v: &[u16]| -> Vec<u8> { let mut o = Vec::with_capacity(v.len()*2); for &x in v { o.extend_from_slice(&x.to_le_bytes()); } o };
        q4sets.push((gpu.upload_u8(&nib), gpu.upload_u8(&cast(&qa)), gpu.upload_u8(&cast(&qb))));
    }
    // f16 weight sets, same footprint ratio (values irrelevant to timing)
    let mut f16sets = vec![];
    for _ in 0..nsets {
        let w: Vec<u16> = (0..n * k).map(|_| half::f16::from_f32(0.02).to_bits()).collect();
        let cast = |v: &[u16]| -> Vec<u8> { let mut o = Vec::with_capacity(v.len()*2); for &x in v { o.extend_from_slice(&x.to_le_bytes()); } o };
        f16sets.push(gpu.upload_u8(&cast(&w)));
    }
    let x: Vec<f32> = (0..m as usize * k).map(|i| ((i % 97) as f32) * 0.01 - 0.4).collect();
    let xb = gpu.upload(&x);
    let yb = gpu.alloc(m as usize * n);

    let src = ojas_metal::kernels::source_of("gemm_mm_q4l").expect("src");
    let pq = gpu.pipeline(src, "gemm_mm_q4l")?;
    let pf = gpu.pipeline(src, "gemm_mm_f16")?;
    let (ku, nu) = (k as u32, n as u32);
    let zero = 0u32;

    let time = |f: &dyn Fn(&metal::ComputeCommandEncoderRef, usize)| -> f64 {
        // warm
        for _ in 0..2 {
            let cb = gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            for i in 0..nsets * 2 { f(&enc, i % nsets); }
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        }
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let cb = gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            for i in 0..nsets * 2 { f(&enc, i % nsets); }
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
            let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
            best = best.min((ge - gs) * 1e3 / (nsets * 2) as f64);
        }
        best
    };

    let tq = time(&|enc, i| {
        let (nib, qa, qb) = &q4sets[i];
        enc.set_compute_pipeline_state(&pq);
        enc.set_buffer(0, Some(&xb.buf), 0);
        enc.set_buffer(1, Some(&nib.buf), 0);
        enc.set_buffer(2, Some(&yb.buf), 0);
        enc.set_bytes(3, 4, &ku as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &nu as *const u32 as *const std::ffi::c_void);
        enc.set_buffer(5, Some(&qa.buf), 0);
        enc.set_bytes(6, 4, &zero as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &m as *const u32 as *const std::ffi::c_void);
        enc.set_buffer(8, Some(&qb.buf), 0);
        enc.dispatch_thread_groups(metal::MTLSize::new((m as u64 + 31) / 32, (n as u64) / 64, 1), metal::MTLSize::new(128, 1, 1));
    });
    let tf = time(&|enc, i| {
        enc.set_compute_pipeline_state(&pf);
        enc.set_buffer(0, Some(&xb.buf), 0);
        enc.set_buffer(1, Some(&f16sets[i].buf), 0);
        enc.set_buffer(2, Some(&yb.buf), 0);
        enc.set_bytes(3, 4, &ku as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &nu as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &zero as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &m as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(metal::MTLSize::new((m as u64 + 31) / 32, (n as u64) / 64, 1), metal::MTLSize::new(128, 1, 1));
    });
    let gflop = 2.0 * k as f64 * n as f64 * m as f64 / 1e9;
    println!("shape K={k} N={n} M={m}  ({gflop:.1} GFLOP)");
    println!("  gemm_mm_q4l : {tq:.3} ms  = {:.2} TFLOP/s   (weights 11.3 MB, SLC-resident)", gflop / tq);
    println!("  gemm_mm_f16 : {tf:.3} ms  = {:.2} TFLOP/s   (weights 45.1 MB, SLC-hostile)", gflop / tf);
    println!("  f16/q4l = {:.3}", tf / tq);
    Ok(())
}
