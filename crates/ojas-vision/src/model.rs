//! One entry point for any vision model: a model file +
//! what it is → `run(frames)` → typed outputs. Four decoders cover the
//! engine's tasks:
//!
//! | Kind | Input placement | Output |
//! |---|---|---|
//! | `Detect` (YOLO head) | letterbox | [`Output::Boxes`], frame pixels |
//! | `Ocr` (CTC) | PP-OCR (height fixed, aspect kept) | [`Output::Text`] |
//! | `Tensor` + `Classify` | stretched to w × h | [`Output::Labels`], top-k (class, prob) |
//! | `Tensor` + `Embed` | stretched to w × h | [`Output::Vector`], L2-normalised |
//!
//! Detect and OCR are the existing [`Detector`] / [`PlateOcr`] (same results,
//! bit for bit). Tensor models run on the CPU executor, or on CUDA / Vulkan
//! through the same batch-bucket plans and GPU crop kernel as OCR. A DETR
//! decoder (NMS-free boxes) joins as a fifth kind once its ops land.

use crate::ir::{Graph, Op};
use crate::pre::{window_geom, window_into, ChanNorm, OcrNorm, Window};
use crate::{bind_input_dims, exec_cpu, import, passes, Detection, Detector, DetectorCfg, Device, Frame, OcrCfg, PlateOcr, PlateRead, Runtime};
use anyhow::{ensure, Context, Result};
use ojas_formats::onnx::OnnxModel;

/// What a tensor model's outputs mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TensorHead {
    /// Class scores (output 0): top-k (class, probability), softmaxed here unless the
    /// graph already ends in Softmax.
    Classify { top_k: usize },
    /// An embedding (the last output — SigLIP's pooled vector follows its hidden states),
    /// L2-normalised.
    Embed,
    /// SimCC 2-D pose (RTMPose): outputs `simcc_x [K, W·split]`, `simcc_y [K, H·split]` →
    /// K keypoints (x, y, score) in input-crop pixels; score = min of the two maxima.
    Pose { split: f32 },
}

/// A classifier / embedding / pose model: a box placed into a `width` × `height` input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorSpec {
    pub width: usize,
    pub height: usize,
    /// same-on-every-channel normalisation; `mean_std` overrides it
    pub norm: OcrNorm,
    /// (x/255 − mean)/std per channel (OSNet, RTMPose: ImageNet)
    pub mean_std: Option<([f32; 3], [f32; 3])>,
    /// how the box maps onto the input
    pub window: Window,
    /// pixel value outside the frame (`Window::Around`), normalised like a pixel
    pub fill_u8: u8,
    /// Feed B, G, R planes.
    pub bgr: bool,
    pub head: TensorHead,
}

impl Default for TensorSpec {
    fn default() -> Self {
        TensorSpec { width: 224, height: 224, norm: OcrNorm::Signed, mean_std: None, window: Window::Stretch, fill_u8: 0, bgr: false, head: TensorHead::Embed }
    }
}

impl TensorSpec {
    pub fn chan_norm(&self) -> ChanNorm {
        match self.mean_std {
            Some((mean, std)) => ChanNorm::mean_std(mean, std),
            None => ChanNorm::of(self.norm),
        }
    }
    /// The device crop placement for this spec.
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    pub fn placement(&self) -> crate::gpu_pre::Placement {
        crate::gpu_pre::Placement::Norm { height: self.height, width: self.width, norm: self.chan_norm(), window: self.window, fill_u8: self.fill_u8, bgr: self.bgr }
    }
}

/// Which graph outputs a head reads, given how many the graph has.
pub(crate) fn head_outputs(head: TensorHead, n_out: usize) -> Result<Vec<usize>> {
    ensure!(n_out >= 1, "tensor model: no outputs");
    Ok(match head {
        TensorHead::Classify { .. } => vec![0],
        TensorHead::Embed => vec![n_out - 1],
        TensorHead::Pose { .. } => {
            ensure!(n_out >= 2, "pose head needs simcc_x and simcc_y outputs");
            vec![0, 1]
        }
    })
}

pub enum ModelKind<'a> {
    Detect(DetectorCfg),
    Ocr { dict: &'a str, cfg: OcrCfg },
    Tensor(TensorSpec),
}

#[derive(Clone, Debug)]
pub enum Output {
    Boxes(Vec<Detection>),
    Text(PlateRead),
    Labels(Vec<(usize, f32)>),
    Vector(Vec<f32>),
    /// (x, y, score) per keypoint, in input-crop pixels (`WindowGeom::to_frame` maps back)
    Keypoints(Vec<[f32; 3]>),
}

pub enum Model {
    Detect(Detector),
    Ocr(PlateOcr),
    Tensor(TensorModel),
}

impl Runtime {
    /// Load any model: `path` is the ONNX file.
    pub fn model(&self, path: &str, kind: ModelKind) -> Result<Model> {
        Ok(match kind {
            ModelKind::Detect(cfg) => Model::Detect(self.detector(path, cfg)?),
            ModelKind::Ocr { dict, cfg } => Model::Ocr(self.plate_ocr(path, dict, cfg)?),
            ModelKind::Tensor(spec) => Model::Tensor(self.tensor_model(&ojas_formats::onnx::load(path)?, spec).with_context(|| format!("loading {path}"))?),
        })
    }

    /// A classifier / embedding model from an ONNX graph already in memory.
    pub fn tensor_model(&self, model: &OnnxModel, spec: TensorSpec) -> Result<TensorModel> {
        TensorModel::load(model, spec, self.threads, self.device)
    }
}

impl Model {
    /// One output per frame (a detector gives each frame's boxes).
    pub fn run(&mut self, frames: &[Frame]) -> Result<Vec<Output>> {
        Ok(match self {
            Model::Detect(d) => d.run(frames)?.into_iter().map(Output::Boxes).collect(),
            Model::Ocr(o) => o.run(frames)?.into_iter().map(Output::Text).collect(),
            Model::Tensor(t) => t.run(frames)?,
        })
    }

    /// "cuda", "vulkan" or "cpu".
    pub fn backend(&self) -> &'static str {
        match self {
            Model::Detect(d) => d.backend(),
            Model::Ocr(o) => o.backend(),
            Model::Tensor(t) => t.backend(),
        }
    }
}

pub struct TensorModel {
    graph: Graph,
    exec: exec_cpu::CpuExecutor,
    spec: TensorSpec,
    /// which graph outputs the head reads
    outs: Vec<usize>,
    /// The graph ends in Softmax: its output is already probabilities.
    softmaxed: bool,
    input_buf: Vec<f32>,
    scratch: Vec<u8>,
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    gpu: Option<crate::gpu_models::AnyGpuTensor>,
}

impl TensorModel {
    fn load(model: &OnnxModel, spec: TensorSpec, threads: usize, device: Device) -> Result<TensorModel> {
        let mut graph = import(model, &bind_input_dims(model, 1, Some(spec.width), Some(spec.height))?)?;
        passes::optimize(&mut graph);
        ensure!(graph.inputs.len() == 1, "tensor model: expected one input");
        let ishape = graph.shape(graph.inputs[0]).to_vec();
        ensure!(ishape == [1, 3, spec.height, spec.width], "tensor model: input {ishape:?}, spec says [1,3,{},{}]", spec.height, spec.width);
        let outs = head_outputs(spec.head, graph.outputs.len())?;
        let out = graph.outputs[outs[0]];
        let softmaxed = graph.nodes.iter().any(|n| n.outputs.contains(&out) && matches!(n.op, Op::Softmax { .. }));
        let exec = exec_cpu::CpuExecutor::new(&graph, threads);
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        let gpu = crate::gpu_models::AnyGpuTensor::load(model, &spec, device)?;
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        let _ = device;
        Ok(TensorModel { graph, exec, spec, outs, softmaxed, input_buf: vec![0.0; 3 * spec.height * spec.width], scratch: vec![], #[cfg(any(feature = "cuda", feature = "vulkan"))] gpu })
    }

    pub fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    pub fn backend(&self) -> &'static str {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if let Some(g) = &self.gpu {
            return g.backend();
        }
        "cpu"
    }

    /// Each frame is the box: placed into the input by the spec's window rule.
    pub fn run(&mut self, frames: &[Frame]) -> Result<Vec<Output>> {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if let Some(g) = self.gpu.as_mut() {
            return Ok(g.run(frames)?.iter().map(|outs| head(outs, &self.spec, self.softmaxed)).collect());
        }
        let mut out = Vec::with_capacity(frames.len());
        let mut scratch = std::mem::take(&mut self.scratch);
        let norm = self.spec.chan_norm();
        for f in frames {
            let (w, h) = f.dims();
            let geom = window_geom([0.0, 0.0, w as f32, h as f32], w, h, self.spec.width, self.spec.height, self.spec.window);
            window_into(f.packed(&mut scratch)?, w, h, &geom, self.spec.width, self.spec.height, &norm, self.spec.fill_u8, self.spec.bgr, &mut self.input_buf);
            let raw = self.exec.run(&self.graph, &[&self.input_buf])?;
            let outs: Vec<Vec<f32>> = self.outs.iter().map(|&i| raw[i].clone()).collect();
            out.push(head(&outs, &self.spec, self.softmaxed));
        }
        self.scratch = scratch;
        Ok(out)
    }
}

/// The head's outputs for one image → its typed result.
pub(crate) fn head(outs: &[Vec<f32>], spec: &TensorSpec, softmaxed: bool) -> Output {
    let raw: &[f32] = &outs[0];
    match spec.head {
        TensorHead::Pose { split } => {
            let (sx, sy) = (&outs[0], &outs[1]);
            let (nx, ny) = ((spec.width as f32 * split) as usize, (spec.height as f32 * split) as usize);
            let k = sx.len() / nx.max(1);
            let argmax = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |m, (i, &x)| if x > m.1 { (i, x) } else { m });
            Output::Keypoints(
                (0..k)
                    .map(|j| {
                        let (bx, vx) = argmax(&sx[j * nx..(j + 1) * nx]);
                        let (by, vy) = argmax(&sy[j * ny..(j + 1) * ny]);
                        [bx as f32 / split, by as f32 / split, vx.min(vy)]
                    })
                    .collect(),
            )
        }
        TensorHead::Classify { top_k } => {
            let probs: Vec<f32> = if softmaxed {
                raw.to_vec()
            } else {
                let m = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = raw.iter().map(|v| (v - m).exp()).collect();
                let s: f32 = e.iter().sum();
                e.iter().map(|v| v / s).collect()
            };
            let mut idx: Vec<usize> = (0..probs.len()).collect();
            // highest first; ties to the lower class id
            idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
            Output::Labels(idx.into_iter().take(top_k.max(1)).map(|i| (i, probs[i])).collect())
        }
        TensorHead::Embed => {
            let n = raw.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            Output::Vector(raw.iter().map(|v| v / n).collect())
        }
    }
}
