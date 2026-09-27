//! ojas-vision — ONNX CNN runtime for detection and plate OCR.
//!
//! CPU-first: the executor runs on `ojas-cpu`'s operators (the oracle), gated
//! against onnxruntime per-node dumps (`examples/cnn_gate.rs`). Device
//! backends (Metal/CUDA) plug in behind the same IR + plan later.
//!
//! ```no_run
//! use ojas_vision::{Runtime, RuntimeCfg, DetectorCfg, Frame};
//! let rt = Runtime::new(RuntimeCfg::default())?;
//! let mut det = rt.detector("models/yolo11n.onnx", DetectorCfg::default())?;
//! let frame = Frame::Rgb8 { w: 1920, h: 1080, data: &[0u8; 1920 * 1080 * 3] };
//! let boxes = det.run(&[frame])?; // Vec<Vec<Detection>>, boxes in frame pixels
//! # anyhow::Ok(())
//! ```

pub mod detr;
pub mod detr_ops;
pub mod exec_cpu;
#[cfg(any(feature = "cuda", feature = "vulkan"))]
pub mod exec_gpu;
/// The CUDA names of the GPU executor (kept for existing callers).
#[cfg(feature = "cuda")]
pub mod exec_cuda {
    pub use crate::exec_gpu::*;
}
#[cfg(any(feature = "cuda", feature = "vulkan"))]
pub mod gpu;
#[cfg(any(feature = "cuda", feature = "vulkan"))]
mod gpu_models;
#[cfg(any(feature = "cuda", feature = "vulkan"))]
pub mod gpu_pre;
#[cfg(any(feature = "cuda", feature = "vulkan"))]
pub mod pipeline;
#[cfg(any(feature = "cuda", feature = "vulkan"))]
pub mod person;
pub mod import;
pub mod ir;
pub mod model;
pub mod passes;
pub mod plate_ocr;
pub mod pre;
pub mod track_gate;
pub mod yolo;

use std::collections::HashMap;

use anyhow::{bail, ensure, Context, Result};
use ojas_formats::onnx::{OnnxDim, OnnxModel};

pub use import::import;
pub use plate_ocr::{Dictionary, PlateRead};
pub use pre::{ChanNorm, Letterbox, OcrNorm, Window, WindowGeom};
pub use yolo::{Detection, DecodeCfg};
pub use model::{Model, ModelKind, Output, TensorHead, TensorModel, TensorSpec};

/// A frame handed to a detector or OCR. Interleaved RGB, 8-bit.
#[derive(Clone, Copy)]
pub enum Frame<'a> {
    Rgb8 { w: usize, h: usize, data: &'a [u8] },
    /// Rows are `stride` bytes apart (≥ 3·w). The decoder's own pitch is
    /// accepted directly; ojas repacks once into a reused scratch buffer.
    Rgb8Strided { w: usize, h: usize, stride: usize, data: &'a [u8] },
}

impl Frame<'_> {
    fn dims(&self) -> (usize, usize) {
        match self {
            Frame::Rgb8 { w, h, .. } | Frame::Rgb8Strided { w, h, .. } => (*w, *h),
        }
    }

    /// Contiguous RGB bytes, using `scratch` only when repacking is needed.
    fn packed<'s>(&'s self, scratch: &'s mut Vec<u8>) -> Result<&'s [u8]> {
        match self {
            Frame::Rgb8 { w, h, data } => {
                ensure!(data.len() == w * h * 3, "frame: {w}x{h} needs {} bytes, got {}", w * h * 3, data.len());
                Ok(data)
            }
            Frame::Rgb8Strided { w, h, stride, data } => {
                ensure!(*stride >= w * 3, "frame: stride {stride} < row bytes {}", w * 3);
                ensure!(data.len() >= (h - 1) * stride + w * 3, "frame: strided buffer too short");
                if *stride == w * 3 {
                    return Ok(&data[..w * h * 3]);
                }
                scratch.clear();
                scratch.reserve(w * h * 3);
                for y in 0..*h {
                    scratch.extend_from_slice(&data[y * stride..y * stride + w * 3]);
                }
                Ok(scratch)
            }
        }
    }
}

/// Where models run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// the ojas-cpu operators (the oracle)
    #[default]
    Cpu,
    /// NVIDIA GPU `ordinal`, fp16 on the ojas-cuda kernels (build feature `cuda`)
    Cuda(usize),
    /// Vulkan device `index` (any vendor: NVIDIA, AMD, Intel, Mesa), fp16 on
    /// the ojas-vulkan kernels (build feature `vulkan`)
    Vulkan(usize),
    /// The first usable CUDA GPU, else the first Vulkan GPU, else CPU
    Auto,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeCfg {
    /// CPU pool size. None = physical performance cores. The library never sizes
    /// its pool from env vars.
    pub threads: Option<usize>,
    pub device: Device,
}

/// An NVIDIA GPU as seen by `Runtime::probe`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuInfo {
    pub ordinal: usize,
    pub name: String,
    pub memory_bytes: usize,
    pub multiprocessors: usize,
    pub compute_capability: (u32, u32),
    /// hardware video decode (libnvcuvid) is available
    pub nvdec: bool,
}

/// What this machine and build can run models on.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DeviceInfo {
    pub cpu_threads: usize,
    /// performance cores, when the platform reports them
    pub cpu_perf_cores: Option<usize>,
    /// this build includes the CUDA backend (feature `cuda`)
    pub cuda_built: bool,
    pub gpus: Vec<GpuInfo>,
    /// this build includes the Vulkan backend (feature `vulkan`)
    #[serde(default)]
    pub vulkan_built: bool,
    /// Vulkan GPUs (CPU implementations like llvmpipe are listed, not picked by `Auto`)
    #[serde(default)]
    pub vulkan: Vec<VulkanInfo>,
    /// why no GPU is usable, when none is
    pub gpu_error: Option<String>,
}

/// A Vulkan device as seen by `Runtime::probe`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VulkanInfo {
    pub index: usize,
    pub name: String,
    pub driver: String,
    /// discrete / integrated / cpu / virtual / other
    pub kind: String,
    pub memory_bytes: u64,
    /// tensor cores / matrix units (`VK_KHR_cooperative_matrix`)
    pub cooperative_matrix: bool,
    /// hardware H.264 decode (`VK_KHR_video_decode_h264`)
    pub video_decode: bool,
}

impl DeviceInfo {
    /// The device `Device::Auto` resolves to: CUDA, else a Vulkan GPU
    /// (discrete first; never a CPU implementation), else the CPU.
    pub fn best(&self) -> Device {
        if let Some(g) = self.gpus.first() {
            return Device::Cuda(g.ordinal);
        }
        let gpu = |k: &str| self.vulkan.iter().find(|v| v.kind == k);
        match gpu("discrete").or_else(|| gpu("integrated")) {
            Some(v) => Device::Vulkan(v.index),
            None => Device::Cpu,
        }
    }
}

/// One runtime per device. Models borrow nothing from it; it only carries
/// configuration.
pub struct Runtime {
    threads: usize,
    device: Device,
}

impl Runtime {
    pub fn new(cfg: RuntimeCfg) -> Result<Runtime> {
        let threads = cfg
            .threads
            .or_else(ojas_cpu::cpu_math::perf_cores)
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(4);
        let device = match cfg.device {
            Device::Auto => Self::probe().best(),
            Device::Cuda(o) => {
                ensure!(cfg!(feature = "cuda"), "Device::Cuda({o}): this build has no CUDA backend (feature `cuda`)");
                Device::Cuda(o)
            }
            Device::Vulkan(i) => {
                ensure!(cfg!(feature = "vulkan"), "Device::Vulkan({i}): this build has no Vulkan backend (feature `vulkan`)");
                Device::Vulkan(i)
            }
            Device::Cpu => Device::Cpu,
        };
        Ok(Runtime { threads: threads.max(1), device })
    }

    /// The device models of this runtime run on (`Auto` resolved).
    pub fn device(&self) -> Device {
        self.device
    }

    /// CPU threads and every usable GPU. A GPU counts as usable when its
    /// context opens and the CNN kernels compile for it.
    pub fn probe() -> DeviceInfo {
        let cpu_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let mut info = DeviceInfo { cpu_threads, cpu_perf_cores: ojas_cpu::cpu_math::perf_cores(), cuda_built: cfg!(feature = "cuda"), gpus: vec![], gpu_error: None,
                                    vulkan_built: cfg!(feature = "vulkan"), vulkan: vec![] };
        #[cfg(feature = "vulkan")]
        {
            info.vulkan = ojas_vulkan::devices()
                .into_iter()
                .map(|d| VulkanInfo { index: d.index, name: d.name, driver: d.driver, kind: d.kind, memory_bytes: d.memory_bytes, cooperative_matrix: d.cooperative_matrix, video_decode: d.video_decode_h264 })
                .collect();
        }
        #[cfg(feature = "cuda")]
        match probe_gpus() {
            Ok(g) => info.gpus = g,
            Err(e) => info.gpu_error = Some(format!("{e:#}")),
        }
        #[cfg(not(feature = "cuda"))]
        {
            info.gpu_error = Some("built without the CUDA backend".into());
        }
        info
    }

    pub fn detector(&self, model_path: &str, cfg: DetectorCfg) -> Result<Detector> {
        Detector::load(model_path, cfg, self.threads, self.device)
    }

    pub fn plate_ocr(&self, model_path: &str, dict_path: &str, cfg: OcrCfg) -> Result<PlateOcr> {
        PlateOcr::load(model_path, Dictionary::load(dict_path)?, cfg, self.threads, self.device)
    }

    pub fn plate_ocr_with_dict(&self, model_path: &str, dict: Dictionary, cfg: OcrCfg) -> Result<PlateOcr> {
        PlateOcr::load(model_path, dict, cfg, self.threads, self.device)
    }
}

#[cfg(feature = "cuda")]
fn probe_gpus() -> Result<Vec<GpuInfo>> {
    let n = ojas_cuda::CudaGpu::device_count()?;
    let nvdec = ojas_cuda::nvdec::available();
    let mut out = vec![];
    for ordinal in 0..n {
        let mut g = ojas_cuda::CudaGpu::new(ordinal)?;
        let p = g.properties()?;
        // the tensor-core kernels use mma.sync m16n8k16 + cp.async: Ampere (8.0) or newer
        anyhow::ensure!(p.compute_capability >= (8, 0), "{} is compute capability {}.{}; the CUDA backend needs 8.0+ (Ampere or newer)",
                        p.name, p.compute_capability.0, p.compute_capability.1);
        ojas_core::KernelRuntime::ensure_family(&mut g, "cnn").context("compiling the CNN kernels (NVRTC)")?;
        out.push(GpuInfo { ordinal, name: p.name, memory_bytes: p.memory_bytes, multiprocessors: p.multiprocessors, compute_capability: p.compute_capability, nvdec });
    }
    anyhow::ensure!(!out.is_empty(), "no NVIDIA GPU found");
    Ok(out)
}

/// Bind a model's input dim_params: dim 0 → batch, dim 1 → 3 channels, dim 2 → `height`
/// and dim 3 → `width` when given (HF exports leave all four symbolic). Errors early on
/// anything still symbolic.
fn bind_input_dims(model: &OnnxModel, batch: usize, width: Option<usize>, height: Option<usize>) -> Result<HashMap<String, usize>> {
    let mut binds = HashMap::new();
    for vi in &model.graph.inputs {
        for (i, d) in vi.dims.iter().enumerate() {
            if let OnnxDim::Param(p) = d {
                let v = match i {
                    0 => batch,
                    1 => 3,
                    2 => height.with_context(|| format!("input {}: symbolic height {p:?} needs a spec height", vi.name))?,
                    3 => width.with_context(|| format!("input {}: symbolic width {p:?} needs OcrCfg.width", vi.name))?,
                    _ => bail!("input {}: symbolic dim {p:?} at position {i} is unsupported", vi.name),
                };
                binds.insert(p.clone(), v);
            }
        }
    }
    Ok(binds)
}

#[derive(Debug, Clone)]
pub struct DetectorCfg {
    /// model input square (informational; the graph's own shape rules)
    pub input: usize,
    pub conf: f32,
    pub iou: f32,
    pub classes: Option<Vec<u16>>,
    pub max_det: usize,
    pub class_agnostic_nms: bool,
}

impl Default for DetectorCfg {
    fn default() -> Self {
        DetectorCfg { input: 640, conf: 0.25, iou: 0.45, classes: None, max_det: 300, class_agnostic_nms: false }
    }
}

pub struct Detector {
    graph: ir::Graph,
    exec: exec_cpu::CpuExecutor,
    cfg: DetectorCfg,
    input_target: usize,
    nc: usize,
    anchors: usize,
    layout: yolo::HeadLayout,
    /// DETR: graph output indices of (logits, boxes)
    detr_outs: (usize, usize),
    input_buf: Vec<f32>,
    scratch: Vec<u8>,
    class_names: Option<Vec<String>>,
    pub pass_stats: passes::PassStats,
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    gpu: Option<gpu_models::AnyGpuDetector>,
}

/// Class names ship beside a detector as `<model>.classes.json` (a JSON
/// string array, e.g. `["plate"]`). Absent file = None; a malformed file or
/// a count mismatch is an error, not a silent fallback.
fn load_class_names(model_path: &str, nc: usize) -> Result<Option<Vec<String>>> {
    let path = match model_path.strip_suffix(".onnx") {
        Some(stem) => format!("{stem}.classes.json"),
        None => return Ok(None),
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(anyhow::anyhow!("{path}: {e}")),
    };
    let names: Vec<String> = serde_json::from_str(&text).with_context(|| format!("{path}: expected a JSON string array"))?;
    ensure!(names.len() == nc, "{path}: {} names for a {nc}-class model", names.len());
    Ok(Some(names))
}

impl Detector {
    fn load(path: &str, cfg: DetectorCfg, threads: usize, device: Device) -> Result<Detector> {
        let t0 = std::time::Instant::now();
        let model = ojas_formats::onnx::load(path)?;
        let binds = bind_input_dims(&model, 1, None, None)?;
        let mut graph = import(&model, &binds).with_context(|| format!("importing {path}"))?;
        let stats = passes::optimize(&mut graph);
        ensure!(graph.inputs.len() == 1, "detector: expected one input, got {}", graph.inputs.len());
        let ishape = graph.shape(graph.inputs[0]).to_vec();
        ensure!(
            ishape.len() == 4 && ishape[0] == 1 && ishape[1] == 3 && ishape[2] == ishape[3],
            "detector: unsupported input shape {ishape:?} (want [1,3,S,S])"
        );
        let input_target = ishape[2];
        // One output: a dense head. Anchors are the (much) larger dim: [1,4+nc,A]
        // Ultralytics vs [1,A,5+nc] YOLOX decode-in-inference (obj·cls scoring).
        // Two outputs: a DETR head, logits [1,Q,C] + boxes [1,Q,4] in either order.
        let (layout, nc, anchors, detr_outs) = match graph.outputs.len() {
            1 => {
                let oshape = graph.shape(graph.outputs[0]).to_vec();
                ensure!(
                    oshape.len() == 3 && oshape[0] == 1 && oshape[1] > 4 && oshape[2] > 4,
                    "detector: unsupported output shape {oshape:?} (want a dense head, no in-graph NMS)"
                );
                if oshape[1] >= oshape[2] {
                    let pyramid = (input_target / 8).pow(2) + (input_target / 16).pow(2) + (input_target / 32).pow(2);
                    ensure!(
                        oshape[1] == pyramid,
                        "detector: [1,A,5+nc] head with A={} but the {input_target}px stride-8/16/32 pyramid has {pyramid} anchors",
                        oshape[1]
                    );
                    (yolo::HeadLayout::AnchorsFirstObj, oshape[2] - 5, oshape[1], (0, 0))
                } else {
                    (yolo::HeadLayout::ChannelsFirst, oshape[1] - 4, oshape[2], (0, 0))
                }
            }
            2 => {
                let (s0, s1) = (graph.shape(graph.outputs[0]).to_vec(), graph.shape(graph.outputs[1]).to_vec());
                let (li, bi) = if s1.len() == 3 && s1[2] == 4 { (0, 1) } else { (1, 0) };
                let (ls, bs) = (graph.shape(graph.outputs[li]).to_vec(), graph.shape(graph.outputs[bi]).to_vec());
                ensure!(
                    ls.len() == 3 && bs.len() == 3 && ls[0] == 1 && bs[0] == 1 && bs[2] == 4 && ls[1] == bs[1] && ls[2] > 0,
                    "detector: outputs {s0:?} {s1:?} are neither a dense head nor DETR logits [1,Q,C] + boxes [1,Q,4]"
                );
                (yolo::HeadLayout::Detr, ls[2], ls[1], (li, bi))
            }
            n => bail!("detector: expected one output (dense head) or two (DETR logits + boxes), got {n}"),
        };
        let exec = exec_cpu::CpuExecutor::new(&graph, threads);
        tracing::info!(
            target: "vision:load",
            path, input = input_target, nc, anchors, head = layout_name(layout),
            nodes = graph.nodes.len(),
            ms = t0.elapsed().as_secs_f64() * 1e3,
            "detector loaded"
        );
        let input_buf = vec![0.0f32; 3 * input_target * input_target];
        let class_names = load_class_names(path, nc)?;
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        let gpu = match (device, layout) {
            (Device::Cuda(_) | Device::Vulkan(_), yolo::HeadLayout::ChannelsFirst | yolo::HeadLayout::Detr) => {
                let g = gpu_models::AnyGpuDetector::load(&model, &cfg, device).with_context(|| format!("{path} on {device:?}"))?;
                if let Some(g) = &g {
                    ensure!(g.input_size() == input_target && g.classes() == nc, "detector: GPU plan disagrees with the CPU graph");
                }
                g
            }
            (Device::Cuda(_) | Device::Vulkan(_), _) => {
                tracing::warn!(target: "vision:load", path, "YOLOX-style head (decode in the graph) runs on the CPU");
                None
            }
            _ => None,
        };
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        let _ = device;
        Ok(Detector {
            graph,
            exec,
            cfg,
            input_target,
            nc,
            anchors,
            layout,
            detr_outs,
            input_buf,
            scratch: Vec::new(),
            class_names,
            pass_stats: stats,
            #[cfg(any(feature = "cuda", feature = "vulkan"))]
            gpu,
        })
    }

    /// Where this detector runs: "cuda", "vulkan" or "cpu".
    pub fn backend(&self) -> &'static str {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if let Some(g) = &self.gpu {
            return g.backend();
        }
        "cpu"
    }

    pub fn classes(&self) -> usize {
        self.nc
    }

    /// The head this export has, which fixes its preprocessing: "yolo" (dense
    /// head, letterbox, NMS) or "detr" (logits + boxes, stretched input, no NMS).
    pub fn head(&self) -> &'static str {
        layout_name(self.layout)
    }

    /// Names from the sibling `<model>.classes.json`, if it exists.
    pub fn class_names(&self) -> Option<&[String]> {
        self.class_names.as_deref()
    }

    pub fn input_size(&self) -> usize {
        self.input_target
    }

    /// Detect on each frame. Boxes come back in frame pixels, clipped.
    pub fn run(&mut self, frames: &[Frame]) -> Result<Vec<Vec<Detection>>> {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if let Some(g) = self.gpu.as_mut() {
            return g.run(frames);
        }
        let decode_cfg = DecodeCfg {
            conf: self.cfg.conf,
            iou: self.cfg.iou,
            max_det: self.cfg.max_det,
            classes: self.cfg.classes.clone(),
            class_agnostic_nms: self.cfg.class_agnostic_nms,
        };
        let mut out = Vec::with_capacity(frames.len());
        let mut scratch = std::mem::take(&mut self.scratch);
        for f in frames {
            let (w, h) = f.dims();
            let rgb = f.packed(&mut scratch)?;
            // Preprocessing is intrinsic to the head: YOLOX decode-in-inference
            // exports were trained on raw 0-255 BGR with a top-left letterbox;
            // DETR exports on the frame stretched to the square, 0-1 RGB.
            let s = self.input_target;
            let lb = match self.layout {
                yolo::HeadLayout::ChannelsFirst => Some(pre::letterbox_rgb8_into(rgb, w, h, s, &mut self.input_buf)),
                yolo::HeadLayout::AnchorsFirstObj => Some(pre::letterbox_yolox_bgr_into(rgb, w, h, s, &mut self.input_buf)),
                yolo::HeadLayout::Detr => {
                    pre::stretch_into(rgb, w, h, s, s, pre::OcrNorm::Unit, false, &mut self.input_buf);
                    None
                }
            };
            let input = std::mem::take(&mut self.input_buf);
            let outs = self.exec.run(&self.graph, &[&input])?;
            self.input_buf = input;
            let dets = match self.layout {
                yolo::HeadLayout::ChannelsFirst => yolo::decode(&outs[0], self.nc, self.anchors, &decode_cfg),
                yolo::HeadLayout::AnchorsFirstObj => yolo::decode_anchors_first_obj(&outs[0], self.nc, self.anchors, s, &decode_cfg),
                yolo::HeadLayout::Detr => {
                    let (li, bi) = self.detr_outs;
                    out.push(detr::decode(&outs[li], &outs[bi], self.nc, self.anchors, w, h, &decode_cfg));
                    continue;
                }
            };
            let mut dets = yolo::nms(dets, &decode_cfg);
            yolo::to_frame(&mut dets, &lb.unwrap());
            out.push(dets);
        }
        self.scratch = scratch;
        Ok(out)
    }

    /// Median wall time per forward at batch 1 (letterbox excluded).
    pub fn bench(&mut self, iters: u32) -> Result<std::time::Duration> {
        Ok(self.bench_detailed(iters, false)?.median)
    }

    /// Full timing: median + min (min is robust on a loaded machine) and,
    /// when `profile`, per-op-kind totals across all iterations.
    pub fn bench_detailed(&mut self, iters: u32, profile: bool) -> Result<BenchResult> {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if let Some(g) = self.gpu.as_mut() {
            let ms = g.bench_ms(iters)?;
            let d = std::time::Duration::from_secs_f64(ms / 1e3);
            return Ok(BenchResult { median: d, min: d, per_op: vec![] });
        }
        let input = vec![0.5f32; 3 * self.input_target * self.input_target];
        for _ in 0..3 {
            self.exec.run(&self.graph, &[&input])?; // warmup
        }
        self.exec.profile = profile;
        self.exec.op_times.clear();
        let mut times: Vec<f64> = Vec::with_capacity(iters as usize);
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            self.exec.run(&self.graph, &[&input])?;
            times.push(t0.elapsed().as_secs_f64());
        }
        self.exec.profile = false;
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut per_op: Vec<(String, f64, usize)> =
            self.exec.op_times.iter().map(|(k, (s, n))| (k.to_string(), *s / iters as f64, *n / iters as usize)).collect();
        per_op.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(BenchResult {
            median: std::time::Duration::from_secs_f64(times[times.len() / 2]),
            min: std::time::Duration::from_secs_f64(times[0]),
            per_op,
        })
    }
}

fn layout_name(l: yolo::HeadLayout) -> &'static str {
    match l {
        yolo::HeadLayout::ChannelsFirst | yolo::HeadLayout::AnchorsFirstObj => "yolo",
        yolo::HeadLayout::Detr => "detr",
    }
}

/// `Detector::bench_detailed` output.
pub struct BenchResult {
    pub median: std::time::Duration,
    pub min: std::time::Duration,
    /// (op kind, seconds per forward, calls per forward), slowest first
    pub per_op: Vec<(String, f64, usize)>,
}

#[derive(Debug, Clone)]
pub struct OcrCfg {
    pub height: usize,
    pub width: usize,
    /// Which normalization this export expects — see `pre::OcrNorm` and the
    /// per-model line in models/MANIFEST.md. The wrong choice produces confident
    /// garbage rather than an error.
    pub norm: pre::OcrNorm,
    /// Feed planes in BGR order (Paddle-native exports trained on cv2 BGR).
    pub bgr: bool,
}

impl Default for OcrCfg {
    fn default() -> Self {
        OcrCfg { height: 48, width: 320, norm: pre::OcrNorm::Signed, bgr: false }
    }
}

/// Batch sizes the OCR plan cache is willing to build.
const OCR_BATCH_SIZES: [usize; 6] = [1, 4, 8, 16, 32, 64];

pub struct PlateOcr {
    /// Parsed ONNX, kept to build further batch plans lazily.
    model: ojas_formats::onnx::OnnxModel,
    /// (graph, executor) per batch size.
    plans: HashMap<usize, (ir::Graph, exec_cpu::CpuExecutor)>,
    dict: Dictionary,
    cfg: OcrCfg,
    threads: usize,
    t_steps: usize,
    classes: usize,
    /// The export's batch dim is a bindable dim_param (> 1). Static batch-1
    /// exports fall back to a per-crop loop.
    batched: bool,
    input_buf: Vec<f32>,
    scratch: Vec<u8>,
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    gpu: Option<gpu_models::AnyGpuOcr>,
}

impl PlateOcr {
    fn build_plan(
        model: &ojas_formats::onnx::OnnxModel,
        cfg: &OcrCfg,
        batch: usize,
        threads: usize,
    ) -> Result<(ir::Graph, exec_cpu::CpuExecutor)> {
        let binds = bind_input_dims(model, batch, Some(cfg.width), Some(cfg.height))?;
        let mut graph = import(model, &binds)?;
        passes::optimize(&mut graph);
        ensure!(graph.inputs.len() == 1 && graph.outputs.len() == 1, "ocr: expected one input and one output");
        let ishape = graph.shape(graph.inputs[0]).to_vec();
        ensure!(
            ishape == [batch, 3, cfg.height, cfg.width],
            "ocr: model input {ishape:?} vs [{batch},3,{},{}]",
            cfg.height,
            cfg.width
        );
        let oshape = graph.shape(graph.outputs[0]).to_vec();
        ensure!(
            oshape.len() == 3 && oshape[0] == batch,
            "ocr: unsupported output {oshape:?} (want [{batch},T,classes])"
        );
        let exec = exec_cpu::CpuExecutor::new(&graph, threads);
        Ok((graph, exec))
    }

    fn load(path: &str, dict: Dictionary, cfg: OcrCfg, threads: usize, device: Device) -> Result<PlateOcr> {
        let model = ojas_formats::onnx::load(path)?;
        let batched = model
            .graph
            .inputs
            .first()
            .and_then(|i| i.dims.first())
            .is_some_and(|d| matches!(d, ojas_formats::onnx::OnnxDim::Param(_)));
        let (graph, exec) = Self::build_plan(&model, &cfg, 1, threads).with_context(|| format!("importing {path}"))?;
        let oshape = graph.shape(graph.outputs[0]).to_vec();
        let (t_steps, classes) = (oshape[1], oshape[2]);
        ensure!(
            classes == dict.classes(),
            "ocr: model emits {classes} classes, dictionary has {} (+1 blank)",
            dict.len()
        );
        let mut plans = HashMap::new();
        plans.insert(1usize, (graph, exec));
        let input_buf = vec![0.0f32; 3 * cfg.height * cfg.width];
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        let gpu = gpu_models::AnyGpuOcr::load(&model, dict.clone(), &cfg, device).with_context(|| format!("{path} on {device:?}"))?;
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        let _ = device;
        Ok(PlateOcr {
            model,
            plans,
            dict,
            cfg,
            threads,
            t_steps,
            classes,
            batched,
            input_buf,
            scratch: Vec::new(),
            #[cfg(any(feature = "cuda", feature = "vulkan"))]
            gpu,
        })
    }

    /// Where this reader runs: "cuda", "vulkan" or "cpu".
    pub fn backend(&self) -> &'static str {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if let Some(g) = &self.gpu {
            return g.backend();
        }
        "cpu"
    }

    /// Compile the plan for each batch size up front so the first real batch runs
    /// at steady-state speed. Sizes are clamped to the cache's allowed set; a no-op
    /// for static batch-1 exports.
    pub fn warmup(&mut self, batches: &[usize]) -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if self.gpu.is_some() {
            return Ok(()); // GPU plans are built and timed at load
        }
        if !self.batched {
            return Ok(());
        }
        for &b in batches {
            let b = Self::plan_size(b);
            if !self.plans.contains_key(&b) {
                let plan = Self::build_plan(&self.model, &self.cfg, b, self.threads)?;
                self.plans.insert(b, plan);
            }
        }
        Ok(())
    }

    /// Smallest cached-plan size that fits n crops.
    fn plan_size(n: usize) -> usize {
        *OCR_BATCH_SIZES.iter().find(|&&b| b >= n).unwrap_or(OCR_BATCH_SIZES.last().unwrap())
    }

    /// Read each plate crop. Text is raw dictionary output. Crops run in
    /// batched forwards when the export allows it (partial batches are
    /// zero-padded; a crop's result never depends on its batch-mates).
    pub fn run(&mut self, crops: &[Frame]) -> Result<Vec<PlateRead>> {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if let Some(g) = self.gpu.as_mut() {
            return g.run(crops);
        }
        let mut out = Vec::with_capacity(crops.len());
        let mut scratch = std::mem::take(&mut self.scratch);
        let plane = 3 * self.cfg.height * self.cfg.width;
        let max_chunk = if self.batched { *OCR_BATCH_SIZES.last().unwrap() } else { 1 };
        for chunk in crops.chunks(max_chunk) {
            let b = if self.batched { Self::plan_size(chunk.len()) } else { 1 };
            if !self.plans.contains_key(&b) {
                let plan = Self::build_plan(&self.model, &self.cfg, b, self.threads)?;
                self.plans.insert(b, plan);
            }
            let mut input = std::mem::take(&mut self.input_buf);
            input.clear();
            input.resize(b * plane, 0.0);
            for (slot, c) in chunk.iter().enumerate() {
                let (w, h) = c.dims();
                let rgb = c.packed(&mut scratch)?;
                pre::ocr_resize_into(
                    rgb,
                    w,
                    h,
                    self.cfg.height,
                    self.cfg.width,
                    self.cfg.norm,
                    self.cfg.bgr,
                    &mut input[slot * plane..(slot + 1) * plane],
                );
            }
            let (graph, exec) = self.plans.get_mut(&b).unwrap();
            let outs = exec.run(graph, &[&input])?;
            self.input_buf = input;
            let step = self.t_steps * self.classes;
            for slot in 0..chunk.len() {
                out.push(plate_ocr::ctc_greedy(
                    &outs[0][slot * step..(slot + 1) * step],
                    self.t_steps,
                    self.classes,
                    &self.dict,
                )?);
            }
        }
        self.scratch = scratch;
        Ok(out)
    }
}
