//! GPU ANPR pipeline vs the CPU chain (anpr.rs logic on the CPU executors) on
//! the same images, then GPU throughput on a batch of frames.
//!
//! `pipeline_gate veh_cpu.onnx veh_gpu.onnx plate_cpu.onnx plate_gpu.onnx rec.onnx dict.txt batch image...`
//! (CPU detectors take batch-1 exports; the GPU side takes batched ones:
//! vehicle at `batch`, plate/OCR with a bindable `batch` dim, planned at 32).

use std::time::Instant;

use anyhow::Result;
use ojas_vision::pipeline::{plan_buckets, Buckets, PipelineCfg, PlatePipeline, RgbFrame};
use ojas_vision::plate_ocr::Dictionary;
use ojas_vision::pre::OcrNorm;
use ojas_vision::{Detection, DetectorCfg, Frame, OcrCfg, Runtime, RuntimeCfg};

fn crop(rgb: &[u8], w: usize, h: usize, d: &Detection, pad: f32) -> (usize, usize, usize, usize, Vec<u8>) {
    let pw = (d.x1 - d.x0) * pad;
    let ph = (d.y1 - d.y0) * pad;
    let x0 = (d.x0 - pw).max(0.0) as usize;
    let y0 = (d.y0 - ph).max(0.0) as usize;
    let x1 = ((d.x1 + pw).ceil() as usize).min(w);
    let y1 = ((d.y1 + ph).ceil() as usize).min(h);
    let (cw, ch) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    let mut out = vec![0u8; cw * ch * 3];
    for y in 0..ch {
        out[y * cw * 3..(y + 1) * cw * 3].copy_from_slice(&rgb[((y0 + y) * w + x0) * 3..((y0 + y) * w + x1) * 3]);
    }
    (x0, y0, cw, ch, out)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let batch: usize = a[7].parse()?;
    let images: Vec<(usize, usize, Vec<u8>)> =
        a[8..].iter().map(|p| ojas_cpu::vit_preprocess::decode_rgb8_path(p)).collect::<Result<_>>()?;

    // --- CPU reference chain (anpr.rs) ---
    let rt = Runtime::new(RuntimeCfg::default())?;
    let mut veh = rt.detector(&a[1], DetectorCfg { conf: 0.25, classes: Some(vec![2, 3, 5, 7]), ..Default::default() })?;
    let mut plate = rt.detector(&a[3], DetectorCfg { conf: 0.15, ..Default::default() })?;
    let mut ocr = rt.plate_ocr(&a[5], &a[6], OcrCfg { norm: OcrNorm::Signed, bgr: true, ..Default::default() })?;
    let mut cpu_reads = vec![];
    for (w, h, rgb) in &images {
        let t = Instant::now();
        let vehicles = veh.run(&[Frame::Rgb8 { w: *w, h: *h, data: rgb }])?.remove(0);
        let mut reads = vec![];
        for v in &vehicles {
            let (vx, vy, cw, ch, vc) = crop(rgb, *w, *h, v, 0.03);
            if cw < 40 || ch < 40 { continue; }
            for p in plate.run(&[Frame::Rgb8 { w: cw, h: ch, data: &vc }])?.remove(0) {
                let (_, _, pw, ph, pc) = crop(&vc, cw, ch, &p, 0.10);
                if pw < 20 || ph < 8 { continue; }
                let r = ocr.run(&[Frame::Rgb8 { w: pw, h: ph, data: &pc }])?.remove(0);
                reads.push((vx as f32 + p.x0, vy as f32 + p.y0, r.text, r.mean_conf));
            }
        }
        println!("CPU  {}x{}: {} vehicles, reads {:?}  [{:.0} ms]", w, h, vehicles.len(),
                 reads.iter().map(|r| format!("{}@({:.0},{:.0}) {:.2}", r.2, r.0, r.1, r.3)).collect::<Vec<_>>(),
                 t.elapsed().as_secs_f64() * 1e3);
        cpu_reads.push(reads);
    }

    // --- GPU pipeline ---
    let t = Instant::now();
    let [v, pl, o]: [Buckets; 3] = plan_buckets(&[(&a[2], &[1, 4, batch]), (&a[4], &[1, 4, 8, 16, 32]), (&a[5], &[1, 4, 8, 16, 32])])?
        .try_into().map_err(|_| anyhow::anyhow!("three models"))?;
    let mut pipe = PlatePipeline::new(v, pl, o, Dictionary::load(&a[6])?, PipelineCfg::default())?;
    println!("GPU plans: {:.1} s", t.elapsed().as_secs_f64());
    let frames: Vec<RgbFrame> = images.iter().map(|(w, h, d)| RgbFrame { w: *w, h: *h, data: d }).collect();
    let res = pipe.run(&frames)?;
    let mut agree = true;
    for ((r, (w, h, _)), cpu) in res.iter().zip(&images).zip(&cpu_reads) {
        let gpu: Vec<(f32, f32, String, f32)> = r.plates.iter().filter_map(|p| p.read.as_ref().map(|rd| (p.det.x0, p.det.y0, rd.text.clone(), rd.mean_conf))).collect();
        println!("GPU  {}x{}: {} vehicles, reads {:?}", w, h, r.vehicles.len(),
                 gpu.iter().map(|r| format!("{}@({:.0},{:.0}) {:.2}", r.2, r.0, r.1, r.3)).collect::<Vec<_>>());
        let texts = |v: &Vec<(f32, f32, String, f32)>| { let mut t: Vec<String> = v.iter().map(|x| x.2.clone()).collect(); t.sort(); t };
        agree &= texts(&gpu) == texts(cpu);
    }
    println!("plate texts CPU == GPU: {agree}");

    // --- throughput: `batch` frames per call, cycling the images ---
    let many: Vec<RgbFrame> = (0..batch).map(|i| { let (w, h, d) = &images[i % images.len()]; RgbFrame { w: *w, h: *h, data: d } }).collect();
    for _ in 0..3 { pipe.run(&many)?; }
    let mut ts = vec![];
    for _ in 0..20 {
        let t = Instant::now();
        pipe.run(&many)?;
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let st = &pipe.times;
    println!("GPU pipeline, {batch} frames/call: median {:.2} ms = {:.0} frames/s | upload {:.2} vehicle {:.2} plate {:.2} ({} crops) ocr {:.2} ({} crops) ms",
             ts[10], batch as f64 * 1e3 / ts[10], st.upload, st.vehicle, st.plate, st.plate_crops, st.ocr, st.ocr_crops);
    Ok(())
}
