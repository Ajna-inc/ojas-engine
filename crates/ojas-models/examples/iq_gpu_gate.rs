//! Checks the IQ requant kernels against the IQ dequant arms.
//!
//! `requant_gate` does this on a real GGUF, but IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS cannot be
//! produced by the reference quantizer without an importance matrix, and building one
//! needs a working reference inference build (the reference build here aborts on
//! mismatched dylibs). So this reads the same random block file `iq_raw_gate` wrote —
//! bytes already checked value-for-value against gguf-py — and drives the Metal kernel
//! over it.
//!
//! Random blocks are the stronger input anyway: they sweep every entry of the 1024-wide
//! grids and all 128 sign words, where a real tensor touches only the codebook entries it
//! happens to need.
//!
//! As in `requant_gate`, agreement is not exact and must not be asserted so: requant
//! lands on int8 with one scale per row. The bound distinguishes Q8 rounding from a
//! decode bug, which misses by orders of magnitude.
//!
//! usage: iq_gpu_gate <ggml_type> <blocks.bin> <rows>

use anyhow::Result;
use metal::MTLResourceOptions;
use ojas_core::Device as _;
use std::ffi::c_void;

fn spec(t: u32) -> (&'static str, usize) {
    match t {
        16 => ("requant_iq2xxs_q8", 66),
        19 => ("requant_iq1s_q8", 50),
        29 => ("requant_iq1m_q8", 56),
        17 => ("requant_iq2xs_q8", 74),
        18 => ("requant_iq3xxs_q8", 98),
        21 => ("requant_iq3s_q8", 110),
        22 => ("requant_iq2s_q8", 82),
        20 => ("requant_iq4nl_q8", 18),
        23 => ("requant_iq4xs_q8", 136),
        10 => ("requant_q2k_q8", 84),
        11 => ("requant_q3k_q8", 110),
        12 => ("requant_q4k_q8", 144),
        13 => ("requant_q5k_q8", 176),
        14 => ("requant_q6k_q8", 210),
        _ => panic!("iq_gpu_gate: no requant kernel for GGUF type {t}"),
    }
}

fn main() -> Result<()> {
    let ty: u32 = std::env::args().nth(1).expect("usage: iq_gpu_gate <type> <bin> <rows>").parse()?;
    let bin = std::env::args().nth(2).unwrap();
    let rows: usize = std::env::args().nth(3).unwrap_or_else(|| "8".into()).parse()?;

    let (entry, bb) = spec(ty);
    let elems_per_blk = if ty == 20 { 32 } else { 256 };
    let raw = std::fs::read(&bin)?;
    let nblk = raw.len() / bb;
    assert!(nblk % rows == 0, "{nblk} blocks does not divide into {rows} rows");
    let nsb = nblk / rows;
    let k = nsb * elems_per_blk;

    // CPU side: proven against gguf-py by iq_raw_gate.
    let refb = ojas_formats::gguf::dequant_to_f16(&raw, ty, nblk * elems_per_blk);

    let gpu = ojas_metal::MetalGpu::new()?;
    let src = ojas_metal::kernels::source_of(entry)
        .unwrap_or_else(|| panic!("kernel {entry} not found in any family"));
    let pipe = gpu.pipeline(src, entry)?;
    let wbuf = gpu.device.new_buffer_with_data(
        raw.as_ptr() as *const c_void, raw.len() as u64, MTLResourceOptions::StorageModeShared);
    let qb = gpu.device.new_buffer((rows * k).max(4) as u64, MTLResourceOptions::StorageModeShared);
    let sb = gpu.device.new_buffer((rows * 4).max(4) as u64, MTLResourceOptions::StorageModeShared);
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(0, Some(&wbuf), 0);
    enc.set_buffer(1, Some(&qb), 0);
    enc.set_buffer(2, Some(&sb), 0);
    let (ku, nu) = (k as u32, rows as u32);
    enc.set_bytes(3, 4, &ku as *const u32 as *const c_void);
    enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(((rows + 7) / 8) as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let qs = unsafe { std::slice::from_raw_parts(qb.contents() as *const i8, rows * k) };
    let sc = unsafe { std::slice::from_raw_parts(sb.contents() as *const f32, rows) };

    let (mut dot, mut na, mut nb, mut se, mut sr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for r in 0..rows {
        let s = sc[r] as f64;
        for c in 0..k {
            let o = (r * k + c) * 2;
            let a = half::f16::from_bits(u16::from_le_bytes([refb[o], refb[o + 1]])).to_f64();
            let b = qs[r * k + c] as f64 * s;
            dot += a * b; na += a * a; nb += b * b;
            se += (a - b) * (a - b); sr += a * a;
        }
    }
    let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 0.0 };
    let rel = if sr > 0.0 { (se / sr).sqrt() } else { 1.0 };
    let ok = cos > 0.999 && rel < 0.02;
    println!("  {:<18} rows={rows} K={k}   cosine {cos:.6}   rel-RMS {:.3}%   {}",
        ojas_formats::gguf::gguf_type_name(ty), rel * 100.0, if ok { "PASS" } else { "FAIL" });
    if ok { Ok(()) } else { std::process::exit(1) }
}
