//! The person lanes on one GPU, device-resident: person boxes (a DETR detector, or boxes the
//! caller already has) → per-track crops cut straight from the frames → an appearance embedding
//! (OSNet), 2-D pose (RTMPose SimCC) and an image–text embedding (SigLIP 2), each batched across
//! every request of the call. Nothing runs per frame here except detection; the lanes run per
//! track, a few times each, through [`PersonGate`].
//!
//! Every model is a `TensorSpec` model (`model.rs`): the crop kernel places each box with the
//! spec's window rule and normalisation (`Placement::Norm`), so the GPU input is the CPU
//! input's bytes, and the heads are the shared `model::head`.

use std::sync::Arc;

use anyhow::{ensure, Result};

use crate::exec_gpu::{ArenaOf, GpuExecutor};
use crate::gpu::GpuDev;
use crate::gpu_pre::{DevFrame, Placement, Roi};
use crate::model::{head, head_outputs, Output, TensorHead, TensorSpec};
use crate::pipeline::{run_stage, upload_rgb, BucketsOf, HostRgb, RgbFrame};
use crate::pre::{window_geom, OcrNorm, Window};
use crate::yolo::{DecodeCfg, Detection};

/// ImageNet mean / std, the normalisation OSNet and RTMPose were trained with.
pub const IMAGENET: ([f32; 3], [f32; 3]) = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);

/// The specs of the three person models as shipped.
pub fn osnet_spec() -> TensorSpec {
    TensorSpec { width: 128, height: 256, mean_std: Some(IMAGENET), window: Window::Stretch, head: TensorHead::Embed, ..Default::default() }
}
pub fn rtmpose_spec() -> TensorSpec {
    TensorSpec { width: 192, height: 256, mean_std: Some(IMAGENET), window: Window::Around { scale: 1.25 }, fill_u8: 0, head: TensorHead::Pose { split: 2.0 }, ..Default::default() }
}
pub fn siglip_spec() -> TensorSpec {
    TensorSpec { width: 224, height: 224, norm: OcrNorm::Signed, window: Window::Stretch, head: TensorHead::Embed, ..Default::default() }
}

/// Which lanes a request wants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Lanes {
    pub reid: bool,
    pub pose: bool,
    pub clip: bool,
}

impl Lanes {
    pub const ALL: Lanes = Lanes { reid: true, pose: true, clip: true };
}

/// One person box to run lanes on: which frame of the call, the box (frame pixels), the lanes.
#[derive(Debug, Clone, Copy)]
pub struct PersonReq {
    pub frame: usize,
    pub bbox: Detection,
    pub lanes: Lanes,
}

/// One keypoint: x, y in frame pixels, score.
pub type Keypoint = [f32; 3];

#[derive(Debug, Clone, Default)]
pub struct PersonOut {
    /// OSNet, L2-normalised
    pub reid: Option<Vec<f32>>,
    /// SigLIP 2 image embedding, L2-normalised
    pub clip: Option<Vec<f32>>,
    /// COCO-17 keypoints in frame pixels
    pub pose: Option<Vec<Keypoint>>,
}

/// Wall time per lane of the last call (ms) and the crops each ran.
#[derive(Debug, Clone, Default)]
pub struct PersonTimes {
    pub upload: f64,
    pub detect: f64,
    pub reid: f64,
    pub pose: f64,
    pub clip: f64,
    pub detections: usize,
    pub reid_crops: usize,
    pub pose_crops: usize,
    pub clip_crops: usize,
}

/// A tensor model's plans plus what the head reads.
struct Lane<G: GpuDev> {
    plans: BucketsOf<G>,
    spec: TensorSpec,
    /// (output index, values per image)
    outs: Vec<(usize, usize)>,
    softmaxed: bool,
}

/// Plans of one ONNX model at `batches` (only 1 for a static export) into `arena`, without
/// running them: every member of an arena must be planned before any runs.
fn plan_execs<G: GpuDev>(path: &str, batches: &[usize], size: Option<(usize, usize)>, ordinal: usize, arena: &Arc<ArenaOf<G>>) -> Result<Vec<GpuExecutor<G>>> {
    let model = ojas_formats::onnx::load(path)?;
    let dynamic = model.graph.inputs.first().and_then(|i| i.dims.first()).is_some_and(|d| matches!(d, ojas_formats::onnx::OnnxDim::Param(_)));
    let sizes: Vec<usize> = if dynamic { batches.to_vec() } else { vec![1] };
    let mut plans = vec![];
    for b in sizes {
        let binds = crate::bind_input_dims(&model, b, size.map(|s| s.0), size.map(|s| s.1))?;
        let mut g = crate::import(&model, &binds)?;
        crate::passes::optimize(&mut g);
        crate::passes::lower_for_gpu(&mut g);
        plans.push(GpuExecutor::<G>::new_in(&g, ordinal, Some(arena))?);
    }
    Ok(plans)
}

impl<G: GpuDev> Lane<G> {
    fn new(path: &str, spec: TensorSpec, plans: Vec<GpuExecutor<G>>) -> Result<Self> {
        let plans = BucketsOf::new(plans)?;
        let exec = plans.largest();
        let st = exec.input().1;
        ensure!(st.h == spec.height && st.w == spec.width, "{path}: input {}x{} vs spec {}x{}", st.h, st.w, spec.height, spec.width);
        let outs = head_outputs(spec.head, exec.output_shapes().len())?.into_iter().map(|i| (i, exec.output_shape(i).iter().skip(1).product::<usize>().max(1))).collect();
        Ok(Lane { plans, spec, outs, softmaxed: false })
    }

    /// Run the lane over `rois` (each ROI is the box; the window rule grows it): the head's
    /// typed output per ROI, in this lane's input coordinates.
    fn run(&mut self, dev: &[DevFrame], rois: &[Roi]) -> Result<Vec<Output>> {
        let place: Placement = self.spec.placement();
        let mut out = Vec::with_capacity(rois.len());
        let mut at = 0;
        for (k, take) in self.plans.split(rois.len()) {
            let chunk = &rois[at..at + take];
            at += take;
            let exec = &mut self.plans.plans[k];
            run_stage(exec, dev, chunk, place)?;
            let mut per_crop: Vec<Vec<Vec<f32>>> = vec![vec![]; chunk.len()];
            for &(oi, per) in &self.outs {
                let raw = exec.read_output(oi, chunk.len())?;
                for (i, c) in raw.chunks(per).take(chunk.len()).enumerate() {
                    per_crop[i].push(c.to_vec());
                }
            }
            out.extend(per_crop.iter().map(|o| head(o, &self.spec, self.softmaxed)));
        }
        Ok(out)
    }
}

/// A DETR-family detector (D-FINE / RT-DETR export: `logits [N,Q,C]`, `boxes [N,Q,4]` cxcywh in
/// input fractions), stretched input in 0–1 RGB, no NMS — the person detector here and the
/// public `Detector`'s GPU backend for such exports (`gpu_models`). The head decode is
/// `crate::detr::decode`, shared with the CPU path.
pub struct DetrHead<G: GpuDev> {
    pub(crate) plans: BucketsOf<G>,
    pub(crate) target: usize,
    pub(crate) nc: usize,
    queries: usize,
    /// output indices of (logits, boxes)
    outs: (usize, usize),
}

impl<G: GpuDev> DetrHead<G> {
    pub(crate) fn new(path: &str, plans: Vec<GpuExecutor<G>>) -> Result<Self> {
        Self::of_buckets(path, BucketsOf::new(plans)?)
    }

    pub(crate) fn of_buckets(path: &str, plans: BucketsOf<G>) -> Result<Self> {
        let exec = plans.largest();
        let st = exec.input().1;
        ensure!(st.h == st.w, "{path}: DETR input must be square");
        ensure!(exec.output_shapes().len() == 2, "{path}: a DETR export has two outputs (logits, boxes), this one has {}", exec.output_shapes().len());
        let (s0, s1) = (exec.output_shape(0).to_vec(), exec.output_shape(1).to_vec());
        let outs = if s1.len() == 3 && s1[2] == 4 { (0, 1) } else { (1, 0) };
        let (ls, bs) = (exec.output_shape(outs.0).to_vec(), exec.output_shape(outs.1).to_vec());
        ensure!(ls.len() == 3 && bs.len() == 3 && bs[2] == 4 && ls[1] == bs[1], "{path}: outputs {s0:?} {s1:?} are not logits [N,Q,C] + boxes [N,Q,4]");
        Ok(DetrHead { target: st.w, nc: ls[2], queries: ls[1], outs, plans })
    }

    /// Detections per ROI (frame pixels, clipped to the ROI), the query's top class among
    /// `cfg.classes`, score ≥ `cfg.conf`, best first, at most `cfg.max_det` per ROI.
    pub(crate) fn detect(&mut self, dev: &[DevFrame], rois: &[Roi], cfg: &DecodeCfg) -> Result<Vec<Vec<Detection>>> {
        let place = Placement::Stretch { height: self.target, width: self.target, norm: OcrNorm::Unit, bgr: false };
        let (q, c) = (self.queries, self.nc);
        let mut all = Vec::with_capacity(rois.len());
        let mut at = 0;
        for (k, take) in self.plans.split(rois.len()) {
            let chunk = &rois[at..at + take];
            at += take;
            let exec = &mut self.plans.plans[k];
            run_stage(exec, dev, chunk, place)?;
            let logits = exec.read_output(self.outs.0, chunk.len())?;
            let boxes = exec.read_output(self.outs.1, chunk.len())?;
            for (i, r) in chunk.iter().enumerate() {
                let (lg, bx) = (&logits[i * q * c..(i + 1) * q * c], &boxes[i * q * 4..(i + 1) * q * 4]);
                let mut dets = crate::detr::decode(lg, bx, c, q, r.w, r.h, cfg);
                for d in dets.iter_mut() {
                    d.x0 += r.x0 as f32;
                    d.x1 += r.x0 as f32;
                    d.y0 += r.y0 as f32;
                    d.y1 += r.y0 as f32;
                }
                all.push(dets);
            }
        }
        Ok(all)
    }
}

/// The model files of a person pipeline; a lane is skipped when its path is `None`.
#[derive(Debug, Clone, Default)]
pub struct PersonModels<'a> {
    /// DETR person detector (D-FINE COCO); `None`: the caller brings boxes
    pub detector: Option<&'a str>,
    pub reid: Option<&'a str>,
    pub pose: Option<&'a str>,
    pub clip: Option<&'a str>,
}

/// Batch buckets per lane: detector at the number of frames per call, crop lanes at the
/// crops a node's cameras produce per call.
#[derive(Debug, Clone)]
pub struct PersonBatches {
    pub detector: Vec<usize>,
    pub reid: Vec<usize>,
    pub pose: Vec<usize>,
    pub clip: Vec<usize>,
}

impl Default for PersonBatches {
    fn default() -> Self {
        PersonBatches { detector: vec![1, 4, 8], reid: vec![4, 16, 32], pose: vec![4, 16], clip: vec![1, 4, 8] }
    }
}

#[derive(Debug, Clone)]
pub struct PersonCfg {
    /// detector score floor and the COCO class ids to keep (person = 0)
    pub conf: f32,
    pub classes: Vec<u16>,
    pub max_det: usize,
    /// a lane skips boxes shorter than this (pixels): ReID ≥ 60, pose ≥ 100, per
    /// `examples/person_survey`
    pub min_reid_h: f32,
    pub min_pose_h: f32,
    pub min_clip_h: f32,
}

impl Default for PersonCfg {
    fn default() -> Self {
        PersonCfg { conf: 0.4, classes: vec![0], max_det: 100, min_reid_h: 60.0, min_pose_h: 100.0, min_clip_h: 60.0 }
    }
}

pub struct PersonPipelineOf<G: GpuDev> {
    detector: Option<DetrHead<G>>,
    reid: Option<Lane<G>>,
    pose: Option<Lane<G>>,
    clip: Option<Lane<G>>,
    frames: Option<G::Buf>,
    pub cfg: PersonCfg,
    pub times: PersonTimes,
}

#[cfg(feature = "cuda")]
pub type PersonPipeline = PersonPipelineOf<ojas_cuda::CudaGpu>;

impl<G: GpuDev> PersonPipelineOf<G> {
    /// Plan every lane on device `ordinal` into one shared activation arena (or the plate
    /// pipeline's, when given: the two then run on one thread, one after the other).
    pub fn load(models: PersonModels, batches: &PersonBatches, ordinal: usize, arena: Option<&Arc<ArenaOf<G>>>, cfg: PersonCfg) -> Result<Self> {
        let own = ArenaOf::<G>::new();
        let arena = arena.unwrap_or(&own);
        // plan everything first (the arena is sized to the largest member and allocated on the
        // first run), then time the buckets
        let (os, rs) = (osnet_spec(), rtmpose_spec());
        let d_plans = models.detector.map(|p| plan_execs(p, &batches.detector, None, ordinal, arena)).transpose()?;
        let r_plans = models.reid.map(|p| plan_execs(p, &batches.reid, Some((os.width, os.height)), ordinal, arena)).transpose()?;
        let p_plans = models.pose.map(|p| plan_execs(p, &batches.pose, Some((rs.width, rs.height)), ordinal, arena)).transpose()?;
        let c_plans = models.clip.map(|p| plan_execs(p, &batches.clip, Some((siglip_spec().width, siglip_spec().height)), ordinal, arena)).transpose()?;
        let detector = d_plans.map(|pl| DetrHead::new(models.detector.unwrap(), pl)).transpose()?;
        let reid = r_plans.map(|pl| Lane::new(models.reid.unwrap(), os, pl)).transpose()?;
        let pose = p_plans.map(|pl| Lane::new(models.pose.unwrap(), rs, pl)).transpose()?;
        let clip = c_plans.map(|pl| Lane::new(models.clip.unwrap(), siglip_spec(), pl)).transpose()?;
        ensure!(detector.is_some() || reid.is_some() || pose.is_some() || clip.is_some(), "person pipeline: no models");
        Ok(PersonPipelineOf { detector, reid, pose, clip, frames: None, cfg, times: PersonTimes::default() })
    }

    /// Persons in each device frame (needs a detector).
    pub fn detect_device(&mut self, dev: &[DevFrame]) -> Result<Vec<Vec<Detection>>> {
        let t = std::time::Instant::now();
        let det = self.detector.as_mut().ok_or_else(|| anyhow::anyhow!("person pipeline: no detector"))?;
        let rois: Vec<Roi> = dev.iter().enumerate().map(|(i, f)| Roi { frame: i, x0: 0, y0: 0, w: f.w, h: f.h }).collect();
        let cfg = DecodeCfg { conf: self.cfg.conf, iou: 1.0, max_det: self.cfg.max_det, classes: Some(self.cfg.classes.clone()), class_agnostic_nms: false };
        let out = det.detect(dev, &rois, &cfg)?;
        self.times.detect = t.elapsed().as_secs_f64() * 1e3;
        self.times.detections = out.iter().map(Vec::len).sum();
        Ok(out)
    }

    /// The lanes for each request, batched per lane across all requests; a lane a box is too
    /// small for (or that is not loaded) stays `None`.
    pub fn lanes_device(&mut self, dev: &[DevFrame], reqs: &[PersonReq]) -> Result<Vec<PersonOut>> {
        let mut out = vec![PersonOut::default(); reqs.len()];
        let roi_of = |r: &PersonReq| -> Option<Roi> {
            let f = dev[r.frame];
            let x0 = r.bbox.x0.max(0.0).min(f.w as f32 - 2.0);
            let y0 = r.bbox.y0.max(0.0).min(f.h as f32 - 2.0);
            let x1 = r.bbox.x1.min(f.w as f32).max(x0 + 1.0);
            let y1 = r.bbox.y1.min(f.h as f32).max(y0 + 1.0);
            let (x0, y0) = (x0 as usize, y0 as usize);
            let (w, h) = ((x1.ceil() as usize).min(f.w) - x0, (y1.ceil() as usize).min(f.h) - y0);
            (w > 0 && h > 0).then_some(Roi { frame: r.frame, x0, y0, w, h })
        };
        // reid
        if let Some(lane) = self.reid.as_mut() {
            let t = std::time::Instant::now();
            let (idx, rois): (Vec<usize>, Vec<Roi>) = reqs.iter().enumerate().filter(|(_, r)| r.lanes.reid && r.bbox.y1 - r.bbox.y0 >= self.cfg.min_reid_h).filter_map(|(i, r)| roi_of(r).map(|roi| (i, roi))).unzip();
            for (i, o) in idx.iter().zip(lane.run(dev, &rois)?) {
                if let Output::Vector(v) = o {
                    out[*i].reid = Some(v);
                }
            }
            self.times.reid_crops = rois.len();
            self.times.reid = t.elapsed().as_secs_f64() * 1e3;
        }
        // clip
        if let Some(lane) = self.clip.as_mut() {
            let t = std::time::Instant::now();
            let (idx, rois): (Vec<usize>, Vec<Roi>) = reqs.iter().enumerate().filter(|(_, r)| r.lanes.clip && r.bbox.y1 - r.bbox.y0 >= self.cfg.min_clip_h).filter_map(|(i, r)| roi_of(r).map(|roi| (i, roi))).unzip();
            for (i, o) in idx.iter().zip(lane.run(dev, &rois)?) {
                if let Output::Vector(v) = o {
                    out[*i].clip = Some(v);
                }
            }
            self.times.clip_crops = rois.len();
            self.times.clip = t.elapsed().as_secs_f64() * 1e3;
        }
        // pose: keypoints come back in input pixels; the same window rule maps them to the frame
        if let Some(lane) = self.pose.as_mut() {
            let t = std::time::Instant::now();
            let (idx, rois): (Vec<usize>, Vec<Roi>) = reqs.iter().enumerate().filter(|(_, r)| r.lanes.pose && r.bbox.y1 - r.bbox.y0 >= self.cfg.min_pose_h).filter_map(|(i, r)| roi_of(r).map(|roi| (i, roi))).unzip();
            let spec = lane.spec;
            for ((i, roi), o) in idx.iter().zip(&rois).zip(lane.run(dev, &rois)?) {
                if let Output::Keypoints(kps) = o {
                    let f = dev[roi.frame];
                    let g = window_geom([roi.x0 as f32, roi.y0 as f32, roi.w as f32, roi.h as f32], f.w, f.h, spec.width, spec.height, spec.window);
                    out[*i].pose = Some(kps.iter().map(|k| { let (x, y) = g.to_frame(k[0], k[1]); [x, y, k[2]] }).collect());
                }
            }
            self.times.pose_crops = rois.len();
            self.times.pose = t.elapsed().as_secs_f64() * 1e3;
        }
        Ok(out)
    }

    /// Host frames: upload, detect (when a detector is loaded) and run every lane on every
    /// detection — the bench / gate path. Per frame: (detections, lanes per detection).
    pub fn run(&mut self, frames: &[RgbFrame]) -> Result<Vec<(Vec<Detection>, Vec<PersonOut>)>> {
        let t = std::time::Instant::now();
        let hs: Vec<HostRgb> = frames.iter().map(|f| HostRgb { w: f.w, h: f.h, stride: f.w * 3, data: f.data }).collect();
        let mut frames_buf = self.frames.take();
        let dev = upload_rgb(self.gpu()?, &mut frames_buf, &hs)?;
        self.frames = frames_buf;
        self.times.upload = t.elapsed().as_secs_f64() * 1e3;
        let dets = self.detect_device(&dev)?;
        let reqs: Vec<PersonReq> = dets.iter().enumerate().flat_map(|(fi, ds)| ds.iter().map(move |d| PersonReq { frame: fi, bbox: *d, lanes: Lanes::ALL })).collect();
        let outs = self.lanes_device(&dev, &reqs)?;
        let mut it = outs.into_iter();
        Ok(dets.into_iter().map(|ds| { let n = ds.len(); (ds, it.by_ref().take(n).collect()) }).collect())
    }

    fn gpu(&self) -> Result<&G> {
        [self.detector.as_ref().map(|d| d.plans.largest().gpu()), self.reid.as_ref().map(|l| l.plans.largest().gpu()), self.pose.as_ref().map(|l| l.plans.largest().gpu()), self.clip.as_ref().map(|l| l.plans.largest().gpu())]
            .into_iter()
            .flatten()
            .next()
            .ok_or_else(|| anyhow::anyhow!("person pipeline: no models"))
    }
}

// ---------------------------------------------------------------------------------------------
// Gating: which tracks get a lane run this frame.

/// Per-lane gate tuning: run a lane on a track a few times while it is in view, spaced out,
/// until enough good samples are in, then stop.
#[derive(Debug, Clone, Copy)]
pub struct PersonGateCfg {
    /// seconds between attempts on one track
    pub min_interval_s: f64,
    /// ... or earlier when the box height grew by this fraction (closer, better crop)
    pub grow: f32,
    /// samples of quality ≥ `good` that settle the track
    pub want: u32,
    pub good: f32,
    /// attempts before giving up
    pub max_attempts: u32,
    /// boxes shorter than this are not attempted
    pub min_h: f32,
    /// attempts per camera frame, tallest first
    pub max_per_frame: usize,
}

impl PersonGateCfg {
    pub fn reid() -> Self {
        PersonGateCfg { min_interval_s: 0.5, grow: 0.25, want: 3, good: 0.5, max_attempts: 6, min_h: 60.0, max_per_frame: 8 }
    }
    pub fn pose() -> Self {
        PersonGateCfg { min_interval_s: 0.2, grow: 0.25, want: 6, good: 0.5, max_attempts: 12, min_h: 100.0, max_per_frame: 8 }
    }
    pub fn clip() -> Self {
        PersonGateCfg { min_interval_s: 1.0, grow: 0.3, want: 1, good: 0.5, max_attempts: 3, min_h: 60.0, max_per_frame: 4 }
    }
}

#[derive(Debug, Default)]
struct GateEntry {
    attempts: u32,
    good: u32,
    last_t: f64,
    last_h: f32,
    done: bool,
}

/// Per-camera, per-lane gate over tracked people (any tracker: it only needs a stable id).
#[derive(Debug)]
pub struct PersonGate {
    pub cfg: PersonGateCfg,
    tracks: std::collections::HashMap<u64, GateEntry>,
}

impl PersonGate {
    pub fn new(cfg: PersonGateCfg) -> Self {
        PersonGate { cfg, tracks: Default::default() }
    }

    /// Which of this frame's tracked people `(track id, box)` to run now (indices into
    /// `cands`), at time `t` seconds. Counts as an attempt.
    pub fn select(&mut self, t: f64, cands: &[(u64, Detection)]) -> Vec<usize> {
        let cfg = self.cfg;
        let mut want: Vec<(usize, f32)> = cands
            .iter()
            .enumerate()
            .filter_map(|(i, (id, d))| {
                let h = d.y1 - d.y0;
                if h < cfg.min_h {
                    return None;
                }
                let due = match self.tracks.get(id) {
                    None => true,
                    Some(e) => !e.done && (t - e.last_t >= cfg.min_interval_s || h >= e.last_h * (1.0 + cfg.grow)),
                };
                due.then_some((i, h))
            })
            .collect();
        want.sort_by(|a, b| b.1.total_cmp(&a.1));
        want.truncate(cfg.max_per_frame);
        for &(i, h) in &want {
            let e = self.tracks.entry(cands[i].0).or_default();
            e.attempts += 1;
            e.last_t = t;
            e.last_h = h;
        }
        want.into_iter().map(|(i, _)| i).collect()
    }

    /// The quality of one attempt's sample on track `id` (0–1; e.g. the box height over 200 px
    /// times the detector score, or the mean keypoint score). Returns true when the track just
    /// settled — enough good samples, or attempts exhausted.
    pub fn record(&mut self, id: u64, quality: f32) -> bool {
        let cfg = self.cfg;
        let Some(e) = self.tracks.get_mut(&id) else { return false };
        if e.done {
            return false;
        }
        e.good += (quality >= cfg.good) as u32;
        if e.good >= cfg.want || e.attempts >= cfg.max_attempts {
            e.done = true;
            return true;
        }
        false
    }

    /// Track `id` left view: (attempts, good samples) if it was ever attempted.
    pub fn end(&mut self, id: u64) -> Option<(u32, u32)> {
        self.tracks.remove(&id).map(|e| (e.attempts, e.good))
    }

    pub fn len(&self) -> usize {
        self.tracks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
}
