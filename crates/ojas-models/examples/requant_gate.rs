//! Checks the GPU requantizer against the CPU dequantizer.
//!
//! Loading a K-quant model fast means never materializing f16: the raw blocks go to the
//! GPU and `requant_qXk_q8` turns them into int8 plus one f32 scale per row in a single
//! resident pass. That kernel reimplements the block math a second time, in Metal, from
//! the same spec, and a transcription slip there produces weights that are wrong but not
//! obviously wrong (see the Qwen3 o_proj K=d bug: silently corrupt, 23x slow, found only
//! because someone read the output).
//!
//! `dequant_to_f16` is the trusted side: `kquant_gate` checks it bit-exact against the
//! reference dequantizer. This checks the GPU against it.
//!
//! The comparison is not exact, and must not be: requant lands on int8 with one scale
//! per row, so a row spanning a wide dynamic range loses real precision. The assertion
//! is that the error looks like Q8 rounding (cosine ~1, relative RMS at the 1/127 level)
//! rather than a decode bug (nibbles swapped, a sub-block mis-scaled, halves transposed),
//! all of which blow past the bound by orders of magnitude while still producing finite,
//! plausible-looking floats.
//!
//! usage: requant_gate <gguf> [tensor]

use anyhow::Result;
use metal::MTLResourceOptions;
use ojas_core::Device as _;
use std::ffi::c_void;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: requant_gate <gguf> [tensor]");
    let want = std::env::args().nth(2);

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&path)?;

    // Every quantized 2-D tensor, or just the one asked for.
    let mut names: Vec<String> = g
        .tensors
        .iter()
        .filter(|(_, i)| i.dims.len() == 2 && matches!(i.ggml_type, 8 | 10 | 11 | 12 | 13 | 14 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 29))
        .map(|(n, _)| n.clone())
        .collect();
    names.sort();
    if let Some(w) = &want { names.retain(|n| n == w); }
    if names.is_empty() { anyhow::bail!("no quantized 2-D tensors matched"); }
    names.truncate(8);

    let mut worst = 0.0f64;
    let mut worst_name = String::new();
    let mut fails = 0usize;
    println!("{:<34} {:>8} {:>10} {:>12} {:>10}", "tensor", "type", "cosine", "rel-RMS", "verdict");
    for name in &names {
        let info = g.tensors.get(name).cloned().unwrap();
        let ty = info.ggml_type;
        let k = info.dims[0] as usize;
        let n = info.dims[1] as usize;
        if k % 256 != 0 && ty != 8 && ty != 20 { continue; }

        // CPU side: the reference, proven against the oracle dequantizer.
        let (_, _, raw) = g.read_tensor_raw(name)?;
        let ref_f16 = ojas_formats::gguf::dequant_to_f16(&raw, ty, k * n);

        // GPU side: upload the same raw blocks and run the shipping kernel.
        let (entry, group) = match ty {
            10 => ("requant_q2k_q8", 256),
            11 => ("requant_q3k_q8", 256),
            12 => ("requant_q4k_q8", 256),
            13 => ("requant_q5k_q8", 256),
            14 => ("requant_q6k_q8", 256),
            16 => ("requant_iq2xxs_q8", 256),
            19 => ("requant_iq1s_q8", 256),
            29 => ("requant_iq1m_q8", 256),
            17 => ("requant_iq2xs_q8", 256),
            18 => ("requant_iq3xxs_q8", 256),
            21 => ("requant_iq3s_q8", 256),
            22 => ("requant_iq2s_q8", 256),
            20 => ("requant_iq4nl_q8", 32),
            23 => ("requant_iq4xs_q8", 256),
            8 => ("requant_q80_q8", 32),
            _ => continue,
        };
        if k % group != 0 { continue; }
        let src = ojas_metal::kernels::source_of(entry).expect("kernel not found");
        let pipe = gpu.pipeline(src, entry)?;
        let wbuf = gpu.device.new_buffer_with_data(
            raw.as_ptr() as *const c_void, raw.len() as u64, MTLResourceOptions::StorageModeShared);
        let qb = gpu.device.new_buffer((n * k).max(4) as u64, MTLResourceOptions::StorageModeShared);
        let sb = gpu.device.new_buffer((n * 4).max(4) as u64, MTLResourceOptions::StorageModeShared);
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&wbuf), 0);
        enc.set_buffer(1, Some(&qb), 0);
        enc.set_buffer(2, Some(&sb), 0);
        let (ku, nu) = (k as u32, n as u32);
        enc.set_bytes(3, 4, &ku as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((n + 7) / 8) as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let qs = unsafe { std::slice::from_raw_parts(qb.contents() as *const i8, n * k) };
        let sc = unsafe { std::slice::from_raw_parts(sb.contents() as *const f32, n) };

        // Compare in the layout the kernel writes: q[row*K + col], row-major,
        // which is how `dequant_to_f16` lays the tensor out too.
        let (mut dot, mut na, mut nb, mut se, mut sr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for r in 0..n {
            let s = sc[r] as f64;
            for c in 0..k {
                let o = (r * k + c) * 2;
                let a = half::f16::from_bits(u16::from_le_bytes([ref_f16[o], ref_f16[o + 1]])).to_f64();
                let b = qs[r * k + c] as f64 * s;
                dot += a * b; na += a * a; nb += b * b;
                se += (a - b) * (a - b); sr += a * a;
            }
        }
        let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 1.0 };
        let rel = if sr > 0.0 { (se / sr).sqrt() } else { 0.0 };
        // Q8 with one scale per row: the quantization step is amax/127, so the
        // RMS error is bounded near (1/127)/sqrt(12) times amax/rms. 2% leaves
        // room for rows with a wide dynamic range; a decode bug is far past it.
        let ok = cos > 0.999 && rel < 0.02;
        if !ok { fails += 1; }
        if rel > worst { worst = rel; worst_name = name.clone(); }
        println!("{:<34} {:>8} {:>10.6} {:>11.3}% {:>10}",
            name, ojas_formats::gguf::gguf_type_name(ty), cos, rel * 100.0,
            if ok { "ok" } else { "FAIL" });
    }

    println!("\nworst rel-RMS {:.3}% on {worst_name}", worst * 100.0);
    if fails == 0 {
        println!("GATE: REQUANT-AGREE PASS");
        Ok(())
    } else {
        println!("GATE: REQUANT-AGREE FAIL ({fails} tensors)");
        std::process::exit(1);
    }
}
