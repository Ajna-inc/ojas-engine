//! CUDA backends of the public `Detector` / `PlateOcr` (host RGB frames in,
//! the same results out): frames are uploaded once, letterboxed / placed by
//! the GPU crop kernel, run through batch-bucket plans sharing one activation
//! arena per model, and decoded with the CPU path's own rules (the dense
//! detector's score filter runs on the GPU; decode + NMS and CTC stay on the
//! host; a DETR head is read back whole and decoded by `detr::decode`).

use anyhow::{ensure, Context, Result};

use crate::exec_gpu::{ArenaOf, GpuExecutor};
use crate::gpu::GpuDev;
use crate::gpu_pre::{Placement, Roi};
use crate::person::DetrHead;
use crate::pipeline::{detect, run_stage, upload_rgb, BucketsOf, DetHead, HostRgb};
use crate::plate_ocr::{ctc_greedy, Dictionary, PlateRead};
use crate::yolo::{DecodeCfg, Detection};
use crate::{bind_input_dims, DetectorCfg, Frame, OcrCfg};

/// Batch sizes planned for a dynamic-batch export (static exports: 1 only).
pub(crate) const DETECTOR_BATCHES: [usize; 3] = [1, 4, 8];
pub(crate) const OCR_BATCHES: [usize; 4] = [1, 4, 16, 32];

fn host<'a>(f: &Frame<'a>) -> HostRgb<'a> {
    match *f {
        Frame::Rgb8 { w, h, data } => HostRgb { w, h, stride: w * 3, data },
        Frame::Rgb8Strided { w, h, stride, data } => HostRgb { w, h, stride, data },
    }
}

/// Plan `batches` of one model (only 1 when its batch dim is static) on
/// device `ordinal`, sharing one arena, and time them.
fn plan<G: GpuDev>(model: &ojas_formats::onnx::OnnxModel, batches: &[usize], width: Option<usize>, height: Option<usize>, ordinal: usize) -> Result<BucketsOf<G>> {
    let dynamic = model
        .graph
        .inputs
        .first()
        .and_then(|i| i.dims.first())
        .is_some_and(|d| matches!(d, ojas_formats::onnx::OnnxDim::Param(_)));
    let sizes: Vec<usize> = if dynamic { batches.to_vec() } else { vec![1] };
    let arena = ArenaOf::<G>::new();
    let mut plans = vec![];
    for b in sizes {
        let binds = bind_input_dims(model, b, width, height)?;
        let mut g = crate::import(model, &binds)?;
        crate::passes::optimize(&mut g);
        crate::passes::lower_for_gpu(&mut g);
        plans.push(GpuExecutor::<G>::new_in(&g, ordinal, Some(&arena))?);
    }
    BucketsOf::new(plans)
}

/// The detector head on the device: a dense YOLO head (letterbox, GPU score
/// filter, host NMS) or a DETR head (stretch, no NMS).
enum GpuHead<G: GpuDev> {
    Dense(DetHead<G>),
    Detr(DetrHead<G>),
}

pub(crate) struct GpuDetector<G: GpuDev> {
    head: GpuHead<G>,
    decode: DecodeCfg,
    frames: Option<G::Buf>,
}

impl<G: GpuDev> GpuDetector<G> {
    pub(crate) fn load(model: &ojas_formats::onnx::OnnxModel, cfg: &DetectorCfg, ordinal: usize) -> Result<Self> {
        let plans = plan(model, &DETECTOR_BATCHES, None, None, ordinal)?;
        let head = match model.graph.outputs.len() {
            2 => GpuHead::Detr(DetrHead::of_buckets(&model.graph.name, plans)?),
            _ => GpuHead::Dense(DetHead::new(plans)?),
        };
        let decode = DecodeCfg { conf: cfg.conf, iou: cfg.iou, max_det: cfg.max_det, classes: cfg.classes.clone(), class_agnostic_nms: cfg.class_agnostic_nms };
        Ok(GpuDetector { head, decode, frames: None })
    }

    pub(crate) fn input_size(&self) -> usize {
        match &self.head {
            GpuHead::Dense(h) => h.target,
            GpuHead::Detr(h) => h.target,
        }
    }

    pub(crate) fn classes(&self) -> usize {
        match &self.head {
            GpuHead::Dense(h) => h.nc,
            GpuHead::Detr(h) => h.nc,
        }
    }

    pub(crate) fn run(&mut self, frames: &[Frame]) -> Result<Vec<Vec<Detection>>> {
        if frames.is_empty() {
            return Ok(vec![]);
        }
        let hs: Vec<HostRgb> = frames.iter().map(host).collect();
        let (head, frames_buf, decode) = (&mut self.head, &mut self.frames, &self.decode);
        let gpu = match &*head {
            GpuHead::Dense(h) => h.plans.largest().gpu(),
            GpuHead::Detr(h) => h.plans.largest().gpu(),
        };
        let dev = upload_rgb(gpu, frames_buf, &hs)?;
        let whole: Vec<Roi> = dev.iter().enumerate().map(|(i, f)| Roi { frame: i, x0: 0, y0: 0, w: f.w, h: f.h }).collect();
        match head {
            GpuHead::Dense(h) => detect(h, &dev, &whole, decode),
            GpuHead::Detr(h) => h.detect(&dev, &whole, decode),
        }
    }

    /// Median device forward time (ms) of the batch-1 plan.
    pub(crate) fn bench_ms(&mut self, iters: u32) -> Result<f64> {
        let exec = match &mut self.head {
            GpuHead::Dense(h) => &mut h.plans.plans[0],
            GpuHead::Detr(h) => &mut h.plans.plans[0],
        };
        let mut ts = vec![];
        for _ in 0..iters.max(3) {
            let t = std::time::Instant::now();
            exec.forward_device()?;
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(|a, b| a.total_cmp(b));
        Ok(ts[ts.len() / 2])
    }
}

pub(crate) struct GpuOcr<G: GpuDev> {
    plans: BucketsOf<G>,
    steps: usize,
    classes: usize,
    dict: Dictionary,
    cfg: OcrCfg,
    frames: Option<G::Buf>,
}

impl<G: GpuDev> GpuOcr<G> {
    pub(crate) fn load(model: &ojas_formats::onnx::OnnxModel, dict: Dictionary, cfg: &OcrCfg, ordinal: usize) -> Result<Self> {
        let plans = plan(model, &OCR_BATCHES, Some(cfg.width), Some(cfg.height), ordinal)?;
        let exec = plans.largest();
        let (st, os) = (exec.input().1, exec.output_shape(0).to_vec());
        ensure!(st.h == cfg.height && st.w == cfg.width, "ocr: model input {}x{} vs cfg {}x{}", st.h, st.w, cfg.height, cfg.width);
        ensure!(os.len() == 3 && os[2] == dict.classes(), "ocr head {os:?} vs dictionary ({} classes)", dict.classes());
        Ok(GpuOcr { steps: os[1], classes: os[2], plans, dict, cfg: cfg.clone(), frames: None })
    }

    pub(crate) fn run(&mut self, crops: &[Frame]) -> Result<Vec<PlateRead>> {
        if crops.is_empty() {
            return Ok(vec![]);
        }
        let hs: Vec<HostRgb> = crops.iter().map(host).collect();
        let dev = upload_rgb(self.plans.largest().gpu(), &mut self.frames, &hs)?;
        let rois: Vec<Roi> = dev.iter().enumerate().map(|(i, f)| Roi { frame: i, x0: 0, y0: 0, w: f.w, h: f.h }).collect();
        let place = Placement::Ocr { height: self.cfg.height, width: self.cfg.width, norm: self.cfg.norm, bgr: self.cfg.bgr };
        let per = self.steps * self.classes;
        let mut out = Vec::with_capacity(crops.len());
        let mut at = 0;
        for (k, take) in self.plans.split(rois.len()) {
            let chunk = &rois[at..at + take];
            at += take;
            let exec = &mut self.plans.plans[k];
            run_stage(exec, &dev, chunk, place).context("ocr stage")?;
            let probs = exec.read_output(0, chunk.len())?;
            for i in 0..chunk.len() {
                out.push(ctc_greedy(&probs[i * per..(i + 1) * per], self.steps, self.classes, &self.dict)?);
            }
        }
        Ok(out)
    }
}

/// A GPU detector on whichever backend the device names.
pub(crate) enum AnyGpuDetector {
    #[cfg(feature = "cuda")]
    Cuda(GpuDetector<ojas_cuda::CudaGpu>),
    #[cfg(feature = "vulkan")]
    Vulkan(GpuDetector<ojas_vulkan::VkGpu>),
}

macro_rules! each {
    ($self:expr, $g:ident => $e:expr) => {
        match $self {
            #[cfg(feature = "cuda")]
            Self::Cuda($g) => $e,
            #[cfg(feature = "vulkan")]
            Self::Vulkan($g) => $e,
        }
    };
}

impl AnyGpuDetector {
    /// None: `device` is not a GPU.
    pub(crate) fn load(model: &ojas_formats::onnx::OnnxModel, cfg: &DetectorCfg, device: crate::Device) -> Result<Option<Self>> {
        Ok(match device {
            #[cfg(feature = "cuda")]
            crate::Device::Cuda(o) => Some(Self::Cuda(GpuDetector::load(model, cfg, o)?)),
            #[cfg(feature = "vulkan")]
            crate::Device::Vulkan(o) => Some(Self::Vulkan(GpuDetector::load(model, cfg, o)?)),
            _ => None,
        })
    }
    pub(crate) fn backend(&self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => <ojas_cuda::CudaGpu as GpuDev>::BACKEND,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => <ojas_vulkan::VkGpu as GpuDev>::BACKEND,
        }
    }
    pub(crate) fn input_size(&self) -> usize {
        each!(self, g => g.input_size())
    }
    pub(crate) fn classes(&self) -> usize {
        each!(self, g => g.classes())
    }
    pub(crate) fn run(&mut self, frames: &[Frame]) -> Result<Vec<Vec<Detection>>> {
        each!(self, g => g.run(frames))
    }
    pub(crate) fn bench_ms(&mut self, iters: u32) -> Result<f64> {
        each!(self, g => g.bench_ms(iters))
    }
}

/// A GPU plate reader on whichever backend the device names.
pub(crate) enum AnyGpuOcr {
    #[cfg(feature = "cuda")]
    Cuda(GpuOcr<ojas_cuda::CudaGpu>),
    #[cfg(feature = "vulkan")]
    Vulkan(GpuOcr<ojas_vulkan::VkGpu>),
}

impl AnyGpuOcr {
    pub(crate) fn load(model: &ojas_formats::onnx::OnnxModel, dict: Dictionary, cfg: &OcrCfg, device: crate::Device) -> Result<Option<Self>> {
        Ok(match device {
            #[cfg(feature = "cuda")]
            crate::Device::Cuda(o) => Some(Self::Cuda(GpuOcr::load(model, dict, cfg, o)?)),
            #[cfg(feature = "vulkan")]
            crate::Device::Vulkan(o) => Some(Self::Vulkan(GpuOcr::load(model, dict, cfg, o)?)),
            _ => None,
        })
    }
    pub(crate) fn backend(&self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => <ojas_cuda::CudaGpu as GpuDev>::BACKEND,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => <ojas_vulkan::VkGpu as GpuDev>::BACKEND,
        }
    }
    pub(crate) fn run(&mut self, crops: &[Frame]) -> Result<Vec<PlateRead>> {
        each!(self, g => g.run(crops))
    }
}

/// Batch sizes planned for a dynamic-batch classifier / embedding export.
pub(crate) const TENSOR_BATCHES: [usize; 4] = [1, 4, 16, 32];

/// A classifier / embedding / pose model on the GPU: crops placed into the input by
/// the crop kernel (the spec's window and normalisation), batch-bucket plans, the raw
/// outputs the head reads per image (the head is `model::head`, shared with the CPU path).
pub(crate) struct GpuTensor<G: GpuDev> {
    plans: BucketsOf<G>,
    spec: crate::model::TensorSpec,
    /// (output index, values per image) for each output the head reads
    outs: Vec<(usize, usize)>,
    frames: Option<G::Buf>,
}

impl<G: GpuDev> GpuTensor<G> {
    pub(crate) fn load(model: &ojas_formats::onnx::OnnxModel, spec: &crate::model::TensorSpec, ordinal: usize) -> Result<Self> {
        let plans = plan(model, &TENSOR_BATCHES, Some(spec.width), Some(spec.height), ordinal)?;
        let exec = plans.largest();
        let st = exec.input().1;
        ensure!(st.h == spec.height && st.w == spec.width, "tensor model: input {}x{} vs spec {}x{}", st.h, st.w, spec.height, spec.width);
        let n_out = exec.output_shapes().len();
        let outs = crate::model::head_outputs(spec.head, n_out)?
            .into_iter()
            .map(|i| (i, exec.output_shape(i).iter().skip(1).product::<usize>().max(1)))
            .collect();
        Ok(GpuTensor { plans, spec: *spec, outs, frames: None })
    }

    /// Per crop: the head's outputs, each one image's values.
    pub(crate) fn run(&mut self, crops: &[Frame]) -> Result<Vec<Vec<Vec<f32>>>> {
        if crops.is_empty() {
            return Ok(vec![]);
        }
        let hs: Vec<HostRgb> = crops.iter().map(host).collect();
        let dev = upload_rgb(self.plans.largest().gpu(), &mut self.frames, &hs)?;
        let rois: Vec<Roi> = dev.iter().enumerate().map(|(i, f)| Roi { frame: i, x0: 0, y0: 0, w: f.w, h: f.h }).collect();
        let place = self.spec.placement();
        let mut out: Vec<Vec<Vec<f32>>> = Vec::with_capacity(crops.len());
        let mut at = 0;
        for (k, take) in self.plans.split(rois.len()) {
            let chunk = &rois[at..at + take];
            at += take;
            let exec = &mut self.plans.plans[k];
            run_stage(exec, &dev, chunk, place).context("tensor stage")?;
            let mut per_crop: Vec<Vec<Vec<f32>>> = vec![vec![]; chunk.len()];
            for &(oi, per) in &self.outs {
                let raw = exec.read_output(oi, chunk.len())?;
                for (i, c) in raw.chunks(per).take(chunk.len()).enumerate() {
                    per_crop[i].push(c.to_vec());
                }
            }
            out.extend(per_crop);
        }
        Ok(out)
    }
}

pub(crate) enum AnyGpuTensor {
    #[cfg(feature = "cuda")]
    Cuda(GpuTensor<ojas_cuda::CudaGpu>),
    #[cfg(feature = "vulkan")]
    Vulkan(GpuTensor<ojas_vulkan::VkGpu>),
}

impl AnyGpuTensor {
    /// None: `device` is not a GPU.
    pub(crate) fn load(model: &ojas_formats::onnx::OnnxModel, spec: &crate::model::TensorSpec, device: crate::Device) -> Result<Option<Self>> {
        Ok(match device {
            #[cfg(feature = "cuda")]
            crate::Device::Cuda(o) => Some(Self::Cuda(GpuTensor::load(model, spec, o)?)),
            #[cfg(feature = "vulkan")]
            crate::Device::Vulkan(o) => Some(Self::Vulkan(GpuTensor::load(model, spec, o)?)),
            _ => None,
        })
    }
    pub(crate) fn backend(&self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => <ojas_cuda::CudaGpu as GpuDev>::BACKEND,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => <ojas_vulkan::VkGpu as GpuDev>::BACKEND,
        }
    }
    pub(crate) fn run(&mut self, crops: &[Frame]) -> Result<Vec<Vec<Vec<f32>>>> {
        each!(self, g => g.run(crops))
    }
}
