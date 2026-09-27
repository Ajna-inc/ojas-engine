//! The ANPR chain on one GPU, device-resident end to end:
//! vehicle detector → vehicle crops → plate detector → plate crops → OCR.
//!
//! Frames are uploaded once as u8 RGB. Every model input is produced on the
//! device by `gpu_pre` (cv2-exact crops/resizes straight from the frames, the
//! same bytes the CPU path feeds), each stage runs batched across all frames
//! of the call, and only detection heads and CTC probabilities come back.
//! Crop rules (paddings, clipping, minimum sizes) are those of
//! `examples/anpr.rs`, so results match the CPU chain.

use std::time::Instant;

use anyhow::{ensure, Result};


use crate::exec_gpu::{ArenaOf, GpuExecutor};
use crate::gpu::GpuDev;
use crate::gpu_pre::{descriptors, launch, DevFrame, PixFmt, Placement, Roi};
use crate::plate_ocr::{ctc_greedy, Dictionary, PlateRead};
use crate::pre::OcrNorm;
use crate::yolo::{self, DecodeCfg, Detection};

#[derive(Debug, Clone)]
pub struct PipelineCfg {
    pub vehicle_conf: f32,
    /// COCO vehicle classes kept (car, motorcycle, bus, truck)
    pub vehicle_classes: Vec<u16>,
    pub plate_conf: f32,
    pub iou: f32,
    /// crop padding as a fraction of the box (anpr.rs: 3 % vehicles, 10 % plates)
    pub vehicle_pad: f32,
    pub plate_pad: f32,
    pub min_vehicle: usize,
    pub min_plate_w: usize,
    pub min_plate_h: usize,
    pub ocr_norm: OcrNorm,
    pub ocr_bgr: bool,
}

impl Default for PipelineCfg {
    fn default() -> Self {
        PipelineCfg {
            vehicle_conf: 0.25,
            vehicle_classes: vec![2, 3, 5, 7],
            plate_conf: 0.15,
            iou: 0.45,
            vehicle_pad: 0.03,
            plate_pad: 0.10,
            min_vehicle: 40,
            min_plate_w: 20,
            min_plate_h: 8,
            ocr_norm: OcrNorm::Signed,
            ocr_bgr: true,
        }
    }
}

/// One plate: its box in frame pixels, the vehicle it was found in, the read.
#[derive(Debug, Clone)]
pub struct PlateHit {
    pub vehicle: usize,
    pub det: Detection,
    pub read: Option<PlateRead>,
}

/// A vehicle to search for plates: its frame (index into the frames passed)
/// and box in frame pixels.
#[derive(Debug, Clone, Copy)]
pub struct PlateReq {
    pub frame: usize,
    pub vehicle: Detection,
}

/// A plate found for a `PlateReq`: box in frame pixels (`det`), the region
/// its OCR crop is clipped to (the padded vehicle crop, `[x0, y0, w, h]`) and
/// the box relative to that region's origin (`rel`, what the crop is cut
/// from — exact, no float round trip), the read (None when not read or the
/// crop is below the OCR minimum size).
#[derive(Debug, Clone)]
pub struct PlateFound {
    pub det: Detection,
    pub within: [usize; 4],
    pub rel: Detection,
    pub read: Option<PlateRead>,
}

/// A plate to read: its frame, the clip region `[x0, y0, w, h]` and the box
/// relative to the region's origin (`PlateFound::{within, rel}`; for a box in
/// frame pixels pass the whole frame as `within`).
#[derive(Debug, Clone, Copy)]
pub struct OcrReq {
    pub frame: usize,
    pub within: [usize; 4],
    pub rel: Detection,
}

#[derive(Debug, Clone, Default)]
pub struct FrameResult {
    pub vehicles: Vec<Detection>,
    pub plates: Vec<PlateHit>,
}

/// Wall time per stage of the last `run` (ms), and the work counts.
#[derive(Debug, Clone, Default)]
pub struct StageTimes {
    pub upload: f64,
    pub vehicle: f64,
    pub plate: f64,
    pub ocr: f64,
    pub vehicles: usize,
    pub plate_crops: usize,
    pub ocr_crops: usize,
}

/// A frame to process: interleaved RGB8, `w * h * 3` bytes.
pub struct RgbFrame<'a> {
    pub w: usize,
    pub h: usize,
    pub data: &'a [u8],
}

/// Plans of one model at several batch sizes; a stage with `n` inputs runs
/// the smallest plan >= n (the largest, repeatedly, beyond it). Padding a
/// 4-crop OCR batch to a 32-plan costs the whole 32 — measured 42 ms vs ~6.
pub struct BucketsOf<G: GpuDev> {
    pub(crate) plans: Vec<GpuExecutor<G>>, // ascending batch
    /// measured forward time per plan (ms)
    cost: Vec<f64>,
}

/// Plans every `(onnx path, bucket batches)` executor of one pipeline on
/// device 0 with a shared activation arena (sized to the largest plan instead
/// of the sum; `OJAS_ARENA=0`: own buffers), then times the buckets.
#[cfg(feature = "cuda")]
pub fn plan_buckets(models: &[(&str, &[usize])]) -> Result<Vec<Buckets>> {
    plan_buckets_on(models, 0)
}

/// [`plan_buckets`] on any backend's device `ordinal`.
pub fn plan_buckets_on<G: GpuDev>(models: &[(&str, &[usize])], ordinal: usize) -> Result<Vec<BucketsOf<G>>> {
    let arena = std::env::var("OJAS_ARENA").map_or(true, |v| v != "0").then(ArenaOf::<G>::new);
    let mut all = vec![];
    for &(path, batches) in models {
        let model = ojas_formats::onnx::load(path)?;
        let mut plans = vec![];
        for &b in batches {
            let binds = std::collections::HashMap::from([("batch".to_string(), b)]);
            let mut g = crate::import(&model, &binds)?;
            crate::passes::optimize(&mut g);
            plans.push(GpuExecutor::<G>::new_in(&g, ordinal, arena.as_ref())?);
        }
        all.push(plans);
    }
    all.into_iter().map(BucketsOf::new).collect()
}

/// Buckets on the CUDA backend (the original name).
#[cfg(feature = "cuda")]
pub type Buckets = BucketsOf<ojas_cuda::CudaGpu>;

impl<G: GpuDev> BucketsOf<G> {
    /// Sorts the plans and times each forward once (median of 5) so `split`
    /// can pick the cheapest cover of a work count.
    pub fn new(mut plans: Vec<GpuExecutor<G>>) -> Result<Self> {
        ensure!(!plans.is_empty(), "no plans");
        plans.sort_by_key(|p| p.batch());
        let mut cost = vec![];
        for p in plans.iter_mut() {
            p.forward_device()?;
            let mut ts: Vec<f64> = (0..5)
                .map(|_| {
                    let t = Instant::now();
                    p.forward_device().map(|_| t.elapsed().as_secs_f64() * 1e3)
                })
                .collect::<Result<_>>()?;
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            cost.push(ts[2]);
        }
        Ok(BucketsOf { plans, cost })
    }
    pub(crate) fn largest(&self) -> &GpuExecutor<G> {
        self.plans.last().unwrap()
    }
    /// Cheapest sequence of plan runs covering `n` items (plan index, items):
    /// dp over measured costs, e.g. 20 crops -> 16 + 4 instead of one padded 32.
    pub(crate) fn split(&self, n: usize) -> Vec<(usize, usize)> {
        let mut best = vec![(0.0f64, usize::MAX); n + 1];
        for i in 1..=n {
            best[i] = (f64::INFINITY, usize::MAX);
            for (k, p) in self.plans.iter().enumerate() {
                let rest = i.saturating_sub(p.batch());
                let c = self.cost[k] + best[rest].0;
                if c < best[i].0 {
                    best[i] = (c, k);
                }
            }
        }
        let (mut out, mut i) = (vec![], n);
        while i > 0 {
            let k = best[i].1;
            let take = self.plans[k].batch().min(i);
            out.push((k, take));
            i -= take;
        }
        out
    }
}

/// Anchor survivors per image read back by the GPU score filter.
const FILTER_CAP: usize = 1024;

pub(crate) struct DetHead<G: GpuDev> {
    pub(crate) plans: BucketsOf<G>,
    pub(crate) target: usize,
    pub(crate) nc: usize,
    pub(crate) anchors: usize,
    // score filter scratch, sized for the largest plan
    count: G::Buf,
    idx: G::Buf,
    cols: G::Buf,
    mask: Option<(Vec<u16>, G::Buf)>,
}

impl<G: GpuDev> DetHead<G> {
    pub(crate) fn new(plans: BucketsOf<G>) -> Result<Self> {
        let exec = plans.largest();
        let st = exec.input().1;
        ensure!(st.h == st.w, "detector input must be square");
        let oshape = exec.output_shape(0).to_vec();
        ensure!(oshape.len() == 3, "detector head {oshape:?}: expected [N, 4+nc, A]");
        let (nc, anchors, b) = (oshape[1] - 4, oshape[2], exec.batch());
        let g = exec.gpu();
        let (count, idx, cols) = (g.alloc_bytes(b * 4)?, g.alloc_bytes(b * FILTER_CAP * 4)?, g.alloc_bytes(b * FILTER_CAP * (4 + nc) * 2)?);
        Ok(DetHead { target: st.w, nc, anchors, count, idx, cols, mask: None, plans })
    }
}

pub struct PlatePipelineOf<G: GpuDev> {
    vehicle: DetHead<G>,
    plate: DetHead<G>,
    ocr: BucketsOf<G>,
    ocr_steps: usize,
    ocr_classes: usize,
    dict: Dictionary,
    pub cfg: PipelineCfg,
    frames: Option<G::Buf>,
    pub times: StageTimes,
}

/// The pipeline on the CUDA backend (the original name).
#[cfg(feature = "cuda")]
pub type PlatePipeline = PlatePipelineOf<ojas_cuda::CudaGpu>;

impl<G: GpuDev> PlatePipelineOf<G> {
    /// Executors are planned by the caller (their batch sizes are the stage
    /// batch sizes: a stage with more crops runs several passes).
    pub fn new(vehicle: BucketsOf<G>, plate: BucketsOf<G>, ocr: BucketsOf<G>, dict: Dictionary, cfg: PipelineCfg) -> Result<Self> {
        let os = ocr.largest().output_shape(0).to_vec();
        ensure!(os.len() == 3 && os[2] == dict.classes(), "ocr head {os:?} vs dictionary ({} classes)", dict.classes());
        Ok(PlatePipelineOf {
            vehicle: DetHead::new(vehicle)?,
            plate: DetHead::new(plate)?,
            ocr_steps: os[1],
            ocr_classes: os[2],
            ocr,
            dict,
            cfg,
            frames: None,
            times: StageTimes::default(),
        })
    }

    pub fn run(&mut self, frames: &[RgbFrame]) -> Result<Vec<FrameResult>> {
        let mut times = StageTimes::default();
        // 1. frames -> device, once
        let t = Instant::now();
        for f in frames {
            ensure!(f.data.len() == f.w * f.h * 3, "frame {}x{}: {} bytes", f.w, f.h, f.data.len());
        }
        let host: Vec<HostRgb> = frames.iter().map(|f| HostRgb { w: f.w, h: f.h, stride: f.w * 3, data: f.data }).collect();
        let dev = upload_rgb(self.vehicle.plans.largest().gpu(), &mut self.frames, &host)?;
        times.upload = t.elapsed().as_secs_f64() * 1e3;
        self.run_device(&dev, times)
    }

    /// The chain on frames already on the device (e.g. NVDEC output). The
    /// frames must stay valid until this returns.
    pub fn run_device(&mut self, dev: &[DevFrame], times: StageTimes) -> Result<Vec<FrameResult>> {
        self.times = times;
        let vehicles = self.detect_vehicles(dev)?;
        let reqs: Vec<PlateReq> = vehicles
            .iter()
            .enumerate()
            .flat_map(|(fi, vs)| vs.iter().map(move |v| PlateReq { frame: fi, vehicle: *v }))
            .collect();
        let found = self.read_plates(dev, &reqs)?;
        let mut out: Vec<FrameResult> = vehicles.into_iter().map(|v| FrameResult { vehicles: v, plates: vec![] }).collect();
        let mut vi = vec![0usize; out.len()];
        for (req, plates) in reqs.iter().zip(found) {
            for p in plates {
                out[req.frame].plates.push(PlateHit { vehicle: vi[req.frame], det: p.det, read: p.read });
            }
            vi[req.frame] += 1;
        }
        Ok(out)
    }

    /// Stage 1 alone: vehicles on whole frames (frame pixels). With a tracker
    /// in between, follow with `read_plates` for the vehicles worth reading.
    pub fn detect_vehicles(&mut self, dev: &[DevFrame]) -> Result<Vec<Vec<Detection>>> {
        let t = Instant::now();
        let whole: Vec<Roi> = dev.iter().enumerate().map(|(i, f)| Roi { frame: i, x0: 0, y0: 0, w: f.w, h: f.h }).collect();
        let vcfg = DecodeCfg { conf: self.cfg.vehicle_conf, iou: self.cfg.iou, classes: Some(self.cfg.vehicle_classes.clone()), ..Default::default() };
        let vehicles = detect(&mut self.vehicle, dev, &whole, &vcfg)?;
        self.times.vehicle = t.elapsed().as_secs_f64() * 1e3;
        self.times.vehicles = vehicles.iter().map(|v| v.len()).sum();
        Ok(vehicles)
    }

    /// Stages 2-3 for chosen vehicles: plate detection inside each vehicle
    /// box (padded, as anpr.rs), OCR on each plate. One result list per
    /// request, plates in frame pixels; vehicles below `min_vehicle` get none.
    pub fn read_plates(&mut self, dev: &[DevFrame], reqs: &[PlateReq]) -> Result<Vec<Vec<PlateFound>>> {
        let mut out = self.find_plates(dev, reqs)?;
        let mut oreqs = vec![];
        let mut owner = vec![];
        for (ri, plates) in out.iter().enumerate() {
            for (pi, p) in plates.iter().enumerate() {
                oreqs.push(OcrReq { frame: reqs[ri].frame, within: p.within, rel: p.rel });
                owner.push((ri, pi));
            }
        }
        for (read, (ri, pi)) in self.ocr_plates(dev, &oreqs)?.into_iter().zip(owner) {
            out[ri][pi].read = read;
        }
        Ok(out)
    }

    /// Stage 2 alone: plate boxes inside each requested vehicle (for callers
    /// that score crops before choosing which to OCR).
    pub fn find_plates(&mut self, dev: &[DevFrame], reqs: &[PlateReq]) -> Result<Vec<Vec<PlateFound>>> {
        let t = Instant::now();
        let mut out: Vec<Vec<PlateFound>> = reqs.iter().map(|_| vec![]).collect();
        let mut vrois = vec![];
        let mut vowner = vec![]; // request index
        for (ri, r) in reqs.iter().enumerate() {
            let f = &dev[r.frame];
            let roi = pad_roi(&r.vehicle, self.cfg.vehicle_pad, r.frame, 0, 0, f.w, f.h);
            if roi.w < self.cfg.min_vehicle || roi.h < self.cfg.min_vehicle {
                continue;
            }
            vrois.push(roi);
            vowner.push(ri);
        }
        let pcfg = DecodeCfg { conf: self.cfg.plate_conf, iou: self.cfg.iou, ..Default::default() };
        let plates = if vrois.is_empty() { vec![] } else { detect(&mut self.plate, dev, &vrois, &pcfg)? };
        self.times.plate_crops = vrois.len();
        for ((vr, &ri), dets) in vrois.iter().zip(&vowner).zip(plates) {
            for rel in dets {
                let mut d = rel;
                d.x0 += vr.x0 as f32;
                d.x1 += vr.x0 as f32;
                d.y0 += vr.y0 as f32;
                d.y1 += vr.y0 as f32;
                out[ri].push(PlateFound { det: d, within: [vr.x0, vr.y0, vr.w, vr.h], rel, read: None });
            }
        }
        self.times.plate = t.elapsed().as_secs_f64() * 1e3;
        Ok(out)
    }

    /// Stage 3 alone: OCR of plates (crops padded by `plate_pad`, clipped to
    /// `within`). None for crops below `min_plate_w` x `min_plate_h`.
    pub fn ocr_plates(&mut self, dev: &[DevFrame], reqs: &[OcrReq]) -> Result<Vec<Option<PlateRead>>> {
        let t = Instant::now();
        let mut out: Vec<Option<PlateRead>> = vec![None; reqs.len()];
        let mut orois = vec![];
        let mut oowner = vec![];
        for (i, r) in reqs.iter().enumerate() {
            let [x0, y0, w, h] = r.within;
            let roi = pad_roi(&r.rel, self.cfg.plate_pad, r.frame, x0, y0, w, h);
            if roi.w >= self.cfg.min_plate_w && roi.h >= self.cfg.min_plate_h {
                orois.push(roi);
                oowner.push(i);
            }
        }
        let ost = self.ocr.largest().input().1;
        let place = Placement::Ocr { height: ost.h, width: ost.w, norm: self.cfg.ocr_norm, bgr: self.cfg.ocr_bgr };
        let mut at = 0;
        for (k, take) in self.ocr.split(orois.len()) {
            let (chunk, owners) = (&orois[at..at + take], &oowner[at..at + take]);
            at += take;
            let exec = &mut self.ocr.plans[k];
            run_stage(exec, dev, chunk, place)?;
            let probs = exec.read_output(0, chunk.len())?;
            let per = self.ocr_steps * self.ocr_classes;
            for (k, &i) in owners.iter().enumerate() {
                out[i] = Some(ctc_greedy(&probs[k * per..(k + 1) * per], self.ocr_steps, self.ocr_classes, &self.dict)?);
            }
        }
        self.times.ocr_crops = orois.len();
        self.times.ocr = t.elapsed().as_secs_f64() * 1e3;
        Ok(out)
    }
}

/// Host RGB8 frame rows `stride` bytes apart (>= 3 * w).
pub(crate) struct HostRgb<'a> {
    pub w: usize,
    pub h: usize,
    pub stride: usize,
    pub data: &'a [u8],
}

/// Copy host frames into `buf` (grown as needed; stream-ordered, the caller's
/// next sync covers it) and describe them as device frames.
pub(crate) fn upload_rgb<G: GpuDev>(g: &G, buf: &mut Option<G::Buf>, frames: &[HostRgb]) -> Result<Vec<DevFrame>> {
    let sizes: Vec<usize> = frames.iter().map(|f| (f.h - 1) * f.stride + f.w * 3).collect();
    let need: usize = sizes.iter().map(|s| s.div_ceil(256) * 256).sum::<usize>().max(1);
    if buf.as_ref().is_none_or(|b| G::buf_len(b) < need) {
        *buf = Some(g.alloc_bytes(need)?);
    }
    let b = buf.as_mut().unwrap();
    let base = g.device_ptr(b);
    let mut off = 0usize;
    let mut out = Vec::with_capacity(frames.len());
    for (f, &sz) in frames.iter().zip(&sizes) {
        ensure!(f.stride >= f.w * 3 && f.data.len() >= sz, "frame {}x{} stride {}: {} bytes", f.w, f.h, f.stride, f.data.len());
        g.write_bytes(b, off, &f.data[..sz])?;
        out.push(DevFrame { ptr: base + off as u64, pitch: f.stride, w: f.w, h: f.h, fmt: PixFmt::Rgb8 });
        off += sz.div_ceil(256) * 256;
    }
    // stages run on other executors' streams: the copies must have landed
    g.submit(g.begin())?;
    Ok(out)
}

/// anpr.rs crop rule: pad the box by `pad` of its size, truncate the low
/// corner, ceil the high one, clip to the `(ox, oy, w, h)` region (frame
/// coordinates of a crop the box is relative to).
fn pad_roi(d: &Detection, pad: f32, frame: usize, ox: usize, oy: usize, w: usize, h: usize) -> Roi {
    let pw = (d.x1 - d.x0) * pad;
    let ph = (d.y1 - d.y0) * pad;
    let x0 = (d.x0 - pw).max(0.0) as usize;
    let y0 = (d.y0 - ph).max(0.0) as usize;
    let x1 = ((d.x1 + pw).ceil() as usize).min(w);
    let y1 = ((d.y1 + ph).ceil() as usize).min(h);
    Roi { frame, x0: ox + x0, y0: oy + y0, w: x1.saturating_sub(x0), h: y1.saturating_sub(y0) }
}

/// Crops -> model input (device) -> forward, one batch.
pub(crate) fn run_stage<G: GpuDev>(exec: &mut GpuExecutor<G>, dev: &[DevFrame], rois: &[Roi], place: Placement) -> Result<Vec<Option<crate::pre::Letterbox>>> {
    let (d, lbs) = descriptors(dev, rois, place)?;
    let g = exec.gpu();
    let dbuf = g.upload_bytes(&d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    let (input, st) = exec.input();
    let enc = g.begin();
    launch(g, &enc, &dbuf, (input, 0), &st, rois.len(), place)?;
    g.submit(enc)?;
    exec.forward_device()?;
    Ok(lbs)
}

/// Batched detection over `rois`; detections in ROI-relative pixels. The
/// head stays on the device: the score filter compacts surviving anchors and
/// only those columns are read and decoded (by the CPU decode, unchanged).
pub(crate) fn detect<G: GpuDev>(head: &mut DetHead<G>, dev: &[DevFrame], rois: &[Roi], cfg: &DecodeCfg) -> Result<Vec<Vec<Detection>>> {
    let mut all = Vec::with_capacity(rois.len());
    let rows = 4 + head.nc;
    // class mask words, cached per class filter
    let classes: Vec<u16> = cfg.classes.clone().unwrap_or_else(|| (0..head.nc as u16).collect());
    if head.mask.as_ref().is_none_or(|(c, _)| c != &classes) {
        let mut words = vec![0u32; head.nc.div_ceil(32)];
        for &c in &classes {
            if (c as usize) < head.nc {
                words[c as usize / 32] |= 1 << (c % 32);
            }
        }
        let buf = head.plans.largest().gpu().upload_bytes(&words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())?;
        head.mask = Some((classes, buf));
    }
    let mut at = 0;
    for (k, take) in head.plans.split(rois.len()) {
        let chunk = &rois[at..at + take];
        at += take;
        let n = chunk.len();
        let exec = &mut head.plans.plans[k];
        let lbs = run_stage(exec, dev, chunk, Placement::Letterbox { target: head.target })?;
        let g = exec.gpu();
        let mut count = std::mem::replace(&mut head.count, g.alloc_bytes(0)?);
        g.write_bytes(&mut count, 0, &vec![0u8; n * 4])?;
        let enc = g.begin();
        g.dispatch(&enc, "cnn_det_filter_f16",
                   &[(exec.output(0), 0), (&head.mask.as_ref().unwrap().1, 0), (&count, 0), (&head.idx, 0), (&head.cols, 0)],
                   &[rows as u32, head.anchors as u32, FILTER_CAP as u32, cfg.conf.to_bits()],
                   [(head.anchors as u32).div_ceil(256), n as u32, 1], [256, 1, 1])?;
        g.submit(enc)?;
        let mut cb = vec![0u8; n * 4];
        g.read_bytes(&count, 0, &mut cb)?;
        head.count = count;
        for (k, lb) in lbs.into_iter().enumerate() {
            let cnt = u32::from_le_bytes(cb[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
            let pred: Vec<f32>;
            let anchors;
            if cnt > FILTER_CAP {
                // overflow: fall back to the whole head of this image
                let raw = exec.read_output(0, k + 1)?;
                pred = raw[k * rows * head.anchors..(k + 1) * rows * head.anchors].to_vec();
                anchors = head.anchors;
            } else {
                let mut ib = vec![0u8; cnt * 4];
                let mut colb = vec![0u8; cnt * rows * 2];
                if cnt > 0 {
                    g.read_bytes(&head.idx, k * FILTER_CAP * 4, &mut ib)?;
                    g.read_bytes(&head.cols, k * FILTER_CAP * rows * 2, &mut colb)?;
                }
                // deterministic order: by anchor index, as the full decode visits them
                let mut order: Vec<usize> = (0..cnt).collect();
                let aidx = |i: usize| u32::from_le_bytes(ib[i * 4..i * 4 + 4].try_into().unwrap());
                order.sort_by_key(|&i| aidx(i));
                let mut p = vec![0.0f32; rows * cnt];
                for (j, &i) in order.iter().enumerate() {
                    for r in 0..rows {
                        let o = (i * rows + r) * 2;
                        p[r * cnt + j] = half::f16::from_bits(u16::from_le_bytes([colb[o], colb[o + 1]])).to_f32();
                    }
                }
                pred = p;
                anchors = cnt;
            }
            let mut dets = yolo::nms(yolo::decode(&pred, head.nc, anchors, cfg), cfg);
            yolo::to_frame(&mut dets, &lb.unwrap());
            all.push(dets);
        }
    }
    Ok(all)
}
