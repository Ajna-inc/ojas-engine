//! Many cameras, fully on the GPU: every camera is an NVDEC decoder fed an
//! H.264 bitstream (looped), each pipeline thread owns `cams` cameras and runs
//! the ANPR chain on their NV12 frames straight from decoder memory.
//!
//! Per round: the next round's pictures are submitted to the decode engine
//! before this round's inference (decode overlaps compute), and with gating
//! (default; `GATE=0` reads every vehicle every frame) each camera's vehicles
//! go through an IoU tracker + `PlateGate`, so plate detection and OCR run
//! only on tracks still being read. `KEY=1`: keyframe-only decode.
//! `CHURN=<s>`: forget every camera's tracks each `s` seconds of stream time
//! (every vehicle new again: traffic of ~vehicles-in-view / s new tracks per
//! second per camera — the looped clip itself has near-static vehicles).
//!
//! `camera_bench veh.onnx plate.onnx rec.onnx dict.txt <pipelines> <cams/pipeline> <seconds> stream.h264`

use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use anyhow::Result;
use ojas_cuda::nvdec::{collect_all, Codec, Mode, NvDecoder};
use ojas_cuda::CudaGpu;
use ojas_vision::gpu_pre::{DevFrame, PixFmt};
use ojas_vision::pipeline::{plan_buckets, Buckets, PipelineCfg, PlateReq, PlatePipeline, RgbFrame, StageTimes};
use ojas_vision::plate_ocr::Dictionary;
use ojas_vision::track_gate::{GateCfg, IouTracker, PlateGate};

const STREAM_FPS: f64 = 15.0;

/// Looping cameras: decoders + stream positions.
struct Cameras {
    decs: Vec<NvDecoder>,
    pos: Vec<usize>,
    stream: Arc<Vec<u8>>,
}

impl Cameras {
    /// Submit bitstream until every camera has a picture queued.
    fn feed(&mut self) -> Result<()> {
        for (d, p) in self.decs.iter_mut().zip(self.pos.iter_mut()) {
            while d.pending() == 0 {
                if *p >= self.stream.len() {
                    *p = 0; // loop: the next IDR + SPS restart decoding
                }
                let end = (*p + 4096).min(self.stream.len());
                d.feed(&self.stream[*p..end], 0, false)?;
                *p = end;
            }
        }
        Ok(())
    }

    /// One frame per camera (after `feed`): map + copy behind one sync.
    fn collect(&mut self) -> Result<Vec<DevFrame>> {
        Ok(collect_all(&mut self.decs, 1)?
            .into_iter()
            .map(|v| {
                let f = v[0];
                DevFrame { ptr: f.ptr, pitch: f.pitch, w: f.w, h: f.h, fmt: PixFmt::Nv12 { uv_off: f.uv_off } }
            })
            .collect())
    }
}

#[derive(Default)]
struct Stats {
    frames: usize,
    rounds: usize,
    decode: f64,
    vehicle: f64,
    plate: f64,
    ocr: f64,
    plate_crops: usize,
    ocr_crops: usize,
    vehicles: usize,
    settled: usize,
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (np, cams, secs): (usize, usize, f64) = (a[5].parse()?, a[6].parse()?, a[7].parse()?);
    let stream = Arc::new(std::fs::read(&a[8])?);
    let barrier = Arc::new(Barrier::new(np + 1));
    let mut handles = vec![];
    for pi in 0..np {
        let (a, stream, barrier) = (a.clone(), stream.clone(), barrier.clone());
        handles.push(std::thread::spawn(move || -> Result<Stats> {
            let [v, pl, o]: [Buckets; 3] = plan_buckets(&[(&a[1], &[cams]), (&a[2], &[4, 8, 16, 32]), (&a[3], &[1, 4, 16])])?
                .try_into().map_err(|_| anyhow::anyhow!("three models"))?;
            let mut pipe = PlatePipeline::new(v, pl, o, Dictionary::load(&a[4])?, PipelineCfg::default())?;
            let gpu = Arc::new(CudaGpu::new(0)?);
            let key = std::env::var("KEY").is_ok_and(|v| v == "1");
            let gate_on = std::env::var("GATE").map_or(true, |v| v != "0");
            let churn: Option<f64> = std::env::var("CHURN").ok().and_then(|v| v.parse().ok());
            let mut cams_ = Cameras {
                decs: (0..cams)
                    .map(|_| {
                        let mut d = NvDecoder::new(gpu.clone(), Codec::H264, 16)?;
                        d.set_mode(if key { Mode::Keyframes } else { Mode::All });
                        Ok(d)
                    })
                    .collect::<Result<_>>()?,
                pos: vec![0; cams],
                stream: stream.clone(),
            };
            // check once: NV12 input == host RGB input (converted with the cv2 formula)
            cams_.feed()?;
            let frames = cams_.collect()?;
            let dev_res = pipe.run_device(&frames, StageTimes::default())?;
            let f0 = frames[0];
            let mut nv = vec![0u8; f0.pitch * f0.h * 3 / 2];
            gpu.read_ptr(f0.ptr, &mut nv)?;
            let rgb = ojas_vision::pre::nv12_to_rgb8(&nv, f0.w, f0.h, f0.pitch, f0.pitch * f0.h);
            let host_res = pipe.run(&[RgbFrame { w: f0.w, h: f0.h, data: &rgb }])?;
            let texts = |r: &ojas_vision::pipeline::FrameResult| -> Vec<String> {
                r.plates.iter().filter_map(|p| p.read.as_ref().map(|x| x.text.clone())).collect()
            };
            if pi == 0 {
                println!("NV12 path {:?} vs host RGB path {:?} -> {}", texts(&dev_res[0]), texts(&host_res[0]),
                         if texts(&dev_res[0]) == texts(&host_res[0]) { "identical" } else { "DIFFER" });
            }
            let mut trackers: Vec<IouTracker> = (0..cams).map(|_| IouTracker::default()).collect();
            let mut gates: Vec<PlateGate> = (0..cams).map(|_| PlateGate::new(GateCfg::default())).collect();
            let mut st = Stats::default();
            let mut round = |frames: &[DevFrame], pipe: &mut PlatePipeline, fno: usize, st: &mut Stats| -> Result<()> {
                if !gate_on {
                    pipe.run_device(frames, StageTimes::default())?;
                } else {
                    let vehicles = pipe.detect_vehicles(frames)?;
                    let t = fno as f64 / STREAM_FPS;
                    if churn.is_some_and(|c| fno > 0 && (t / c).floor() != ((fno - 1) as f64 / STREAM_FPS / c).floor()) {
                        trackers.iter_mut().for_each(|t| *t = IouTracker::default());
                        gates.iter_mut().for_each(|g| *g = PlateGate::new(GateCfg::default()));
                    }
                    let (mut reqs, mut owners) = (vec![], vec![]);
                    for (ci, vs) in vehicles.iter().enumerate() {
                        let (ids, ended) = trackers[ci].update(vs);
                        for id in ended {
                            gates[ci].end(id);
                        }
                        let cands: Vec<(u64, _)> = ids.iter().zip(vs).filter_map(|(id, v)| id.map(|i| (i, *v))).collect();
                        for k in gates[ci].select(t, &cands) {
                            reqs.push(PlateReq { frame: ci, vehicle: cands[k].1 });
                            owners.push((ci, cands[k].0));
                        }
                    }
                    let found = pipe.read_plates(frames, &reqs)?;
                    for ((ci, id), plates) in owners.into_iter().zip(found) {
                        let reads: Vec<(String, f32)> =
                            plates.iter().filter_map(|p| p.read.as_ref().map(|r| (r.text.clone(), r.mean_conf))).collect();
                        if gates[ci].record(id, &reads).is_some() {
                            st.settled += 1;
                        }
                    }
                }
                let s = &pipe.times;
                st.vehicle += s.vehicle;
                st.plate += s.plate;
                st.ocr += s.ocr;
                st.plate_crops += s.plate_crops;
                st.ocr_crops += s.ocr_crops;
                st.vehicles += s.vehicles;
                Ok(())
            };
            let mut fno = 0;
            for _ in 0..3 {
                cams_.feed()?;
                let f = cams_.collect()?;
                round(&f, &mut pipe, fno, &mut Stats::default())?;
                fno += 1;
            }
            barrier.wait();
            let t0 = Instant::now();
            cams_.feed()?;
            let mut frames = cams_.collect()?;
            while t0.elapsed() < Duration::from_secs_f64(secs) {
                // next round's pictures decode on the engine during this round's inference
                let td = Instant::now();
                cams_.feed()?;
                st.decode += td.elapsed().as_secs_f64() * 1e3;
                round(&frames, &mut pipe, fno, &mut st)?;
                let td = Instant::now();
                frames = cams_.collect()?;
                st.decode += td.elapsed().as_secs_f64() * 1e3;
                st.frames += cams;
                st.rounds += 1;
                fno += 1;
            }
            Ok(st)
        }));
    }
    barrier.wait();
    let t0 = Instant::now();
    let mut total = 0;
    for (i, h) in handles.into_iter().enumerate() {
        let st = h.join().unwrap()?;
        let r = st.rounds.max(1) as f64;
        println!("  pipeline {i}: {} frames ({} rounds) | per round: decode {:.1} vehicle {:.1} ({:.1} vehicles) \
                  plate {:.1} ({:.1} crops) ocr {:.1} ({:.1} crops) ms | gate: {} tracks settled",
                 st.frames, st.rounds, st.decode / r, st.vehicle / r, st.vehicles as f64 / r, st.plate / r,
                 st.plate_crops as f64 / r, st.ocr / r, st.ocr_crops as f64 / r, st.settled);
        total += st.frames;
    }
    let wall = t0.elapsed().as_secs_f64();
    println!("TOTAL {np} pipeline(s) x {cams} cameras: {total} frames in {wall:.1} s = {:.0} frames/s (1080p H.264, decode on GPU)", total as f64 / wall);
    Ok(())
}
