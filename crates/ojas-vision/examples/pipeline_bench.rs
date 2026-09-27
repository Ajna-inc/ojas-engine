//! Aggregate throughput of N concurrent GPU ANPR pipelines (one per thread,
//! each with its own CUDA streams), fed 1920x1080 "camera" frames.
//!
//! `pipeline_bench veh.onnx plate.onnx rec.onnx dict.txt <pipelines> <frames/call> <seconds> image...`

use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use anyhow::Result;
use ojas_vision::pipeline::{plan_buckets, Buckets, PipelineCfg, PlatePipeline, RgbFrame};
use ojas_vision::plate_ocr::Dictionary;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (np, per_call, secs): (usize, usize, f64) = (a[5].parse()?, a[6].parse()?, a[7].parse()?);
    let frames: Vec<(usize, usize, Vec<u8>)> = a[8..]
        .iter()
        .map(|p| {
            let (w, h, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(p)?;
            Ok((1920, 1080, ojas_vision::pre::resize_bilinear_rgb8(&rgb, w, h, 1920, 1080, 3)))
        })
        .collect::<Result<_>>()?;
    let frames = Arc::new(frames);
    let barrier = Arc::new(Barrier::new(np + 1));
    let mut handles = vec![];
    let t_plan = Instant::now();
    for pi in 0..np {
        let (a, frames, barrier) = (a.clone(), frames.clone(), barrier.clone());
        handles.push(std::thread::spawn(move || -> Result<(usize, Vec<f64>, [f64; 4], [usize; 3])> {
            let [v, pl, o]: [Buckets; 3] = plan_buckets(&[(&a[1], &[per_call]), (&a[2], &[4, 8, 16, 32]), (&a[3], &[1, 4, 16])])?
                .try_into().map_err(|_| anyhow::anyhow!("three models"))?;
            let mut pipe = PlatePipeline::new(v, pl, o, Dictionary::load(&a[4])?, PipelineCfg::default())?;
            let batch: Vec<RgbFrame> = (0..per_call)
                .map(|i| { let (w, h, d) = &frames[(i + pi) % frames.len()]; RgbFrame { w: *w, h: *h, data: d } })
                .collect();
            for _ in 0..3 { pipe.run(&batch)?; }
            barrier.wait();
            let (t0, mut done, mut lat) = (Instant::now(), 0usize, vec![]);
            let mut acc = [0.0f64; 4];
            while t0.elapsed() < Duration::from_secs_f64(secs) {
                let t = Instant::now();
                pipe.run(&batch)?;
                lat.push(t.elapsed().as_secs_f64() * 1e3);
                done += per_call;
                let s = &pipe.times;
                acc[0] += s.upload; acc[1] += s.vehicle; acc[2] += s.plate; acc[3] += s.ocr;
            }
            let s = &pipe.times;
            let n = lat.len() as f64;
            Ok((done, lat, acc.map(|v| v / n), [s.vehicles, s.plate_crops, s.ocr_crops]))
        }));
    }
    barrier.wait();
    println!("plans for {np} pipeline(s): {:.1} s", t_plan.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let mut total = 0;
    for (i, h) in handles.into_iter().enumerate() {
        let (done, mut lat, st, work) = h.join().unwrap()?;
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("  pipeline {i}: {done} frames, call p50 {:.1} ms p90 {:.1} ms | stage avg upload {:.1} vehicle {:.1} plate {:.1} ocr {:.1} ms | per call {} vehicles {} plate crops {} ocr crops",
                 lat[lat.len() / 2], lat[lat.len() * 9 / 10], st[0], st[1], st[2], st[3], work[0], work[1], work[2]);
        total += done;
    }
    let wall = t0.elapsed().as_secs_f64();
    println!("TOTAL {np} pipeline(s) x {per_call} frames/call: {total} frames in {wall:.1} s = {:.0} frames/s (1080p)", total as f64 / wall);
    Ok(())
}
