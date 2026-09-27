//! Checks the native IQ MoE expert kernels against the CPU decoders.
//!
//! `moe_gu_*` / `moe_down_*` decode GGUF blocks in-kernel, and their CPU twins in
//! `cpu_math` are pinned against `gguf::dequant_to_f16` (checked value-for-value against
//! gguf-py), but nothing checked the two against each other. A transcription slip in the
//! Metal port — a wrong offset, a swapped nibble half, a sign mask read from the wrong
//! byte — produces plausible-looking numbers and otherwise shows up only as bad text
//! from a model too large to bisect.
//!
//! Rows come from a real tensor rather than random bytes, so they carry the scales and
//! sign patterns the model actually uses.
//!
//! Agreement is exact to f32 rounding — both sides do the same arithmetic in the same
//! order per row — so a relative miss beyond ~1e-3 is a decode bug, not accumulated
//! error.
//!
//! usage: moe_iq_gate <gguf> [layer] [rows]

use anyhow::Result;
use metal::MTLResourceOptions;
use ojas_core::Device as _;
use std::ffi::c_void;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: moe_iq_gate <gguf> [layer] [rows]");
    let layer: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let rows: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&path)?;

    let name = format!("blk.{layer}.ffn_gate_exps.weight");
    let info = g.tensors.get(&name).cloned().expect("no ffn_gate_exps tensor");
    let (k, n_ff, n_exp) = (info.dims[0] as usize, info.dims[1] as usize, info.dims[2] as usize);
    let ty = info.ggml_type;
    let f = ojas_core::quant_src::format_of(ty).expect("no native format for this type");
    let row_bytes = k / f.weights as usize * f.block_bytes as usize;
    let kern = ojas_core::quant_src::moe_kernel(ty, ojas_core::quant_src::MoeRole::GateUp)
        .unwrap_or_else(|| panic!("no gate/up MoE kernel for GGUF type {ty}"));
    println!("{name}: type {ty} ({}), K={k} ffn={n_ff} experts={n_exp}, {row_bytes} B/row -> {}",
             f.tag, kern.entry);

    // Expert 0's first `rows` rows, laid out exactly as the kernel indexes them.
    let (_d, _t, raw) = g.read_tensor_raw(&name)?;
    let rows = rows.min(n_ff);
    let want_bytes = rows * row_bytes;
    let w = &raw[..want_bytes];

    // Deterministic activations, exactly representable in f32 so both sides agree.
    let x: Vec<f32> = (0..k)
        .map(|i| (((i.wrapping_mul(2654435761)) % 2048) as f32 / 1024.0) - 1.0)
        .collect();

    // CPU reference: silu(gate)*up with the same buffer bound to both operands, so
    // `up == gate` and one decoder call per row covers both.
    let cpu: Vec<f32> = (0..rows)
        .map(|r| {
            let row = &w[r * row_bytes..(r + 1) * row_bytes];
            let s = ojas_cpu::cpu_math::dot_iq(ty, row, &x);
            (s / (1.0 + (-s).exp())) * s
        })
        .collect();

    // GPU: one expert, identity index, N = rows. Kernels live in whichever family
    // defines them (moe or moe_iq), so look them up by name.
    let fam = |e: &str| ojas_metal::kernels::family_of(e)
        .and_then(ojas_metal::kernels::family_source)
        .unwrap_or_else(|| panic!("no family defines {e}"));
    let src = fam(kern.entry);
    let pipes = gpu.compile_all(src, |n| n == kern.entry)?;
    let (_, pipe) = pipes.into_iter().find(|(n, _)| n == kern.entry).expect("pipeline missing");
    let dev = gpu.device.clone();
    let wb = dev.new_buffer_with_data(w.as_ptr() as *const c_void, want_bytes as u64,
                                      MTLResourceOptions::StorageModeShared);
    let xb = dev.new_buffer_with_data(x.as_ptr() as *const c_void, (k * 4) as u64,
                                      MTLResourceOptions::StorageModeShared);
    let idx = [0u32];
    let ib = dev.new_buffer_with_data(idx.as_ptr() as *const c_void, 4,
                                      MTLResourceOptions::StorageModeShared);
    let ab = dev.new_buffer(((rows * 4) as u64).max(4), MTLResourceOptions::StorageModeShared);

    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(0, Some(&xb), 0);
    enc.set_buffer(1, Some(&wb), 0);
    enc.set_buffer(2, Some(&wb), 0);
    enc.set_buffer(3, Some(&ab), 0);
    let (kk, nn) = (k as u32, rows as u32);
    enc.set_bytes(4, 4, &kk as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &nn as *const u32 as *const c_void);
    enc.set_buffer(8, Some(&ib), 0);
    let (t, r) = kern.launch;
    enc.dispatch_thread_groups(metal::MTLSize::new(((nn + r - 1) / r) as u64, 1, 1),
                               metal::MTLSize::new(t as u64, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let got = unsafe { std::slice::from_raw_parts(ab.contents() as *const f32, rows) };
    let (mut worst, mut at) = (0f32, 0usize);
    for i in 0..rows {
        let rel = (got[i] - cpu[i]).abs() / cpu[i].abs().max(1e-3);
        if rel > worst {
            worst = rel;
            at = i;
        }
    }
    println!("rows={rows} worst rel diff {worst:.3e} at row {at} (gpu {} vs cpu {})",
             got[at], cpu[at]);
    if worst > 1e-3 {
        println!("GATE: MOE IQ KERNEL MISMATCH FAIL ({})", kern.entry);
        std::process::exit(1);
    }
    println!("GATE: MOE IQ KERNEL PASS ({})", kern.entry);

    // ---- down projection ----------------------------------------------------
    // Different kernel family and a different signature, so it needs its own pass:
    // one expert, weight 1.0, and a shared-expert gate driven to -inf so the
    // sigmoid term vanishes and the output is exactly the row dot product.
    let dname = format!("blk.{layer}.ffn_down_exps.weight");
    let dinfo = g.tensors.get(&dname).cloned().expect("no ffn_down_exps tensor");
    let (dk_, dn_) = (dinfo.dims[0] as usize, dinfo.dims[1] as usize);
    let dty = dinfo.ggml_type;
    let df = ojas_core::quant_src::format_of(dty).expect("no native format");
    let drow = dk_ / df.weights as usize * df.block_bytes as usize;
    let dkern = ojas_core::quant_src::moe_kernel(dty, ojas_core::quant_src::MoeRole::Down)
        .unwrap_or_else(|| panic!("no down MoE kernel for GGUF type {dty}"));
    println!("\n{dname}: type {dty} ({}), K={dk_} N={dn_}, {drow} B/row -> {}", df.tag, dkern.entry);

    let (_d2, _t2, draw) = g.read_tensor_raw(&dname)?;
    let drows = rows.min(dn_);
    let dw = &draw[..drows * drow];
    let a: Vec<f32> = (0..dk_)
        .map(|i| (((i.wrapping_mul(40503)) % 2048) as f32 / 1024.0) - 1.0)
        .collect();
    let dcpu: Vec<f32> = (0..drows)
        .map(|r| ojas_cpu::cpu_math::dot_iq(dty, &dw[r * drow..(r + 1) * drow], &a))
        .collect();

    let dpipes = gpu.compile_all(fam(dkern.entry), |n| n == dkern.entry)?;
    let (_, dpipe) = dpipes.into_iter().find(|(n, _)| n == dkern.entry).expect("pipeline missing");
    let dwb = dev.new_buffer_with_data(dw.as_ptr() as *const c_void, (drows * drow) as u64,
                                       MTLResourceOptions::StorageModeShared);
    let dab = dev.new_buffer_with_data(a.as_ptr() as *const c_void, (dk_ * 4) as u64,
                                       MTLResourceOptions::StorageModeShared);
    let one = [1.0f32];
    let wgt = dev.new_buffer_with_data(one.as_ptr() as *const c_void, 4, MTLResourceOptions::StorageModeShared);
    let zeros = vec![0f32; drows];
    let outb = dev.new_buffer_with_data(zeros.as_ptr() as *const c_void, (drows * 4) as u64,
                                        MTLResourceOptions::StorageModeShared);
    let shx = dev.new_buffer_with_data(zeros.as_ptr() as *const c_void, (drows * 4) as u64,
                                       MTLResourceOptions::StorageModeShared);
    let neg = [-60.0f32];
    let shg = dev.new_buffer_with_data(neg.as_ptr() as *const c_void, 4, MTLResourceOptions::StorageModeShared);

    let cb2 = gpu.command_buffer();
    let e2 = cb2.new_compute_command_encoder();
    e2.set_compute_pipeline_state(&dpipe);
    e2.set_buffer(0, Some(&dab), 0);
    e2.set_buffer(1, Some(&dwb), 0);
    e2.set_buffer(2, Some(&outb), 0);
    let (dkk, dnn, ksel) = (dk_ as u32, drows as u32, 1u32);
    e2.set_bytes(3, 4, &dkk as *const u32 as *const c_void);
    e2.set_bytes(4, 4, &dnn as *const u32 as *const c_void);
    e2.set_buffer(6, Some(&ib), 0);
    e2.set_buffer(7, Some(&wgt), 0);
    e2.set_bytes(8, 4, &ksel as *const u32 as *const c_void);
    e2.set_buffer(9, Some(&shx), 0);
    e2.set_buffer(10, Some(&shg), 0);
    let (dt, dr) = dkern.launch;
    e2.dispatch_thread_groups(metal::MTLSize::new(((dnn + dr - 1) / dr) as u64, 1, 1),
                              metal::MTLSize::new(dt as u64, 1, 1));
    e2.end_encoding();
    cb2.commit();
    cb2.wait_until_completed();

    let dgot = unsafe { std::slice::from_raw_parts(outb.contents() as *const f32, drows) };
    let (mut dworst, mut dat) = (0f32, 0usize);
    for i in 0..drows {
        let rel = (dgot[i] - dcpu[i]).abs() / dcpu[i].abs().max(1e-3);
        if rel > dworst { dworst = rel; dat = i; }
    }
    println!("rows={drows} worst rel diff {dworst:.3e} at row {dat} (gpu {} vs cpu {})",
             dgot[dat], dcpu[dat]);
    if dworst > 1e-3 {
        println!("GATE: MOE IQ KERNEL MISMATCH FAIL ({})", dkern.entry);
        std::process::exit(1);
    }
    println!("GATE: MOE IQ KERNEL PASS ({})", dkern.entry);

    // ---- batched (M>1) gate/up ----------------------------------------------
    // Two tokens with different activations: a batched kernel that ignored the token
    // stride would still pass with identical rows.
    if let Some(mk) = ojas_core::quant_src::moe_kernel_m(ty, ojas_core::quant_src::MoeRole::GateUp) {
        println!("\nbatched: {}", mk.entry);
        let x2: Vec<f32> = (0..k).map(|i| (((i.wrapping_mul(2246822519)) % 2048) as f32 / 1024.0) - 1.0).collect();
        let mut xm = x.clone();
        xm.extend_from_slice(&x2);

        let want: Vec<f32> = [&x, &x2].iter().flat_map(|xx| {
            (0..rows).map(move |r| {
                let row = &w[r * row_bytes..(r + 1) * row_bytes];
                let s = ojas_cpu::cpu_math::dot_iq(ty, row, xx);
                (s / (1.0 + (-s).exp())) * s
            })
        }).collect();

        let mpipes = gpu.compile_all(fam(mk.entry), |n| n == mk.entry)?;
        let (_, mpipe) = mpipes.into_iter().find(|(n, _)| n == mk.entry).expect("pipeline missing");
        let xmb = dev.new_buffer_with_data(xm.as_ptr() as *const c_void, (xm.len() * 4) as u64,
                                           MTLResourceOptions::StorageModeShared);
        let idx2 = [0u32, 0u32];   // both tokens route to expert 0
        let ib2 = dev.new_buffer_with_data(idx2.as_ptr() as *const c_void, 8,
                                           MTLResourceOptions::StorageModeShared);
        let amb = dev.new_buffer((rows * 2 * 4) as u64, MTLResourceOptions::StorageModeShared);

        let cb3 = gpu.command_buffer();
        let e3 = cb3.new_compute_command_encoder();
        e3.set_compute_pipeline_state(&mpipe);
        e3.set_buffer(0, Some(&xmb), 0);
        e3.set_buffer(1, Some(&wb), 0);
        e3.set_buffer(2, Some(&wb), 0);
        e3.set_buffer(3, Some(&amb), 0);
        let (kk, nn, ksel) = (k as u32, rows as u32, 1u32);
        e3.set_bytes(4, 4, &kk as *const u32 as *const c_void);
        e3.set_bytes(5, 4, &nn as *const u32 as *const c_void);
        e3.set_buffer(8, Some(&ib2), 0);
        e3.set_bytes(9, 4, &ksel as *const u32 as *const c_void);
        let (mt, mr) = mk.launch;
        e3.dispatch_thread_groups(metal::MTLSize::new(((nn + mr - 1) / mr) as u64, 2, 1),
                                  metal::MTLSize::new(mt as u64, 1, 1));
        e3.end_encoding();
        cb3.commit();
        cb3.wait_until_completed();

        let mgot = unsafe { std::slice::from_raw_parts(amb.contents() as *const f32, rows * 2) };
        let (mut mworst, mut mat) = (0f32, 0usize);
        for i in 0..rows * 2 {
            let rel = (mgot[i] - want[i]).abs() / want[i].abs().max(1e-3);
            if rel > mworst { mworst = rel; mat = i; }
        }
        println!("rows={} (2 tokens) worst rel diff {mworst:.3e} at {mat} (gpu {} vs cpu {})",
                 rows * 2, mgot[mat], want[mat]);
        if mworst > 1e-3 {
            println!("GATE: MOE IQ BATCHED MISMATCH FAIL ({})", mk.entry);
            std::process::exit(1);
        }
        println!("GATE: MOE IQ BATCHED PASS ({})", mk.entry);
    }

    // ---- batched down --------------------------------------------------------
    if let Some(dmk) = ojas_core::quant_src::moe_kernel_m(dty, ojas_core::quant_src::MoeRole::Down) {
        println!("\nbatched: {}", dmk.entry);
        let a2: Vec<f32> = (0..dk_).map(|i| (((i.wrapping_mul(1103515245)) % 2048) as f32 / 1024.0) - 1.0).collect();
        let mut am = a.clone();
        am.extend_from_slice(&a2);
        let dwant: Vec<f32> = [&a, &a2].iter().flat_map(|aa| {
            (0..drows).map(move |r| ojas_cpu::cpu_math::dot_iq(dty, &dw[r * drow..(r + 1) * drow], aa))
        }).collect();

        let dmp = gpu.compile_all(fam(dmk.entry), |n| n == dmk.entry)?;
        let (_, dmpipe) = dmp.into_iter().find(|(n, _)| n == dmk.entry).expect("pipeline missing");
        let amb2 = dev.new_buffer_with_data(am.as_ptr() as *const c_void, (am.len() * 4) as u64,
                                            MTLResourceOptions::StorageModeShared);
        let idx2 = [0u32, 0u32];
        let ib3 = dev.new_buffer_with_data(idx2.as_ptr() as *const c_void, 8, MTLResourceOptions::StorageModeShared);
        let ones2 = [1.0f32, 1.0f32];
        let wgt2 = dev.new_buffer_with_data(ones2.as_ptr() as *const c_void, 8, MTLResourceOptions::StorageModeShared);
        let z2 = vec![0f32; drows * 2];
        let out2 = dev.new_buffer_with_data(z2.as_ptr() as *const c_void, (drows * 2 * 4) as u64,
                                            MTLResourceOptions::StorageModeShared);
        let shx2 = dev.new_buffer_with_data(z2.as_ptr() as *const c_void, (drows * 2 * 4) as u64,
                                            MTLResourceOptions::StorageModeShared);
        let neg2 = [-60.0f32, -60.0f32];
        let shg2 = dev.new_buffer_with_data(neg2.as_ptr() as *const c_void, 8, MTLResourceOptions::StorageModeShared);

        let cb4 = gpu.command_buffer();
        let e4 = cb4.new_compute_command_encoder();
        e4.set_compute_pipeline_state(&dmpipe);
        e4.set_buffer(0, Some(&amb2), 0);
        e4.set_buffer(1, Some(&dwb), 0);
        e4.set_buffer(2, Some(&out2), 0);
        let (dkk2, dnn2, ks2) = (dk_ as u32, drows as u32, 1u32);
        e4.set_bytes(3, 4, &dkk2 as *const u32 as *const c_void);
        e4.set_bytes(4, 4, &dnn2 as *const u32 as *const c_void);
        e4.set_buffer(6, Some(&ib3), 0);
        e4.set_buffer(7, Some(&wgt2), 0);
        e4.set_bytes(8, 4, &ks2 as *const u32 as *const c_void);
        e4.set_buffer(9, Some(&shx2), 0);
        e4.set_buffer(10, Some(&shg2), 0);
        let (dmt, dmr) = dmk.launch;
        e4.dispatch_thread_groups(metal::MTLSize::new(((dnn2 + dmr - 1) / dmr) as u64, 2, 1),
                                  metal::MTLSize::new(dmt as u64, 1, 1));
        e4.end_encoding();
        cb4.commit();
        cb4.wait_until_completed();

        let dmgot = unsafe { std::slice::from_raw_parts(out2.contents() as *const f32, drows * 2) };
        let (mut w2, mut a2i) = (0f32, 0usize);
        for i in 0..drows * 2 {
            let rel = (dmgot[i] - dwant[i]).abs() / dwant[i].abs().max(1e-3);
            if rel > w2 { w2 = rel; a2i = i; }
        }
        println!("rows={} (2 tokens) worst rel diff {w2:.3e} at {a2i} (gpu {} vs cpu {})",
                 drows * 2, dmgot[a2i], dwant[a2i]);
        if w2 > 1e-3 {
            println!("GATE: MOE IQ BATCHED MISMATCH FAIL ({})", dmk.entry);
            std::process::exit(1);
        }
        println!("GATE: MOE IQ BATCHED PASS ({})", dmk.entry);
    }
    Ok(())
}
