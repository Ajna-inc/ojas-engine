//! What the GPU executor and pipeline need from a backend: buffers, byte
//! copies, the kernel dispatch seam (`KernelRuntime`) and a convolution plan.
//! Implemented by ojas-cuda (`CudaGpu`, NVIDIA) and ojas-vulkan (`VkGpu`, any
//! Vulkan 1.3 driver); everything above this trait is backend-neutral.

use anyhow::Result;
use ojas_core::conv::{ConvGeom, Storage};
use ojas_core::KernelRuntime;

/// A planned convolution (weights packed for the backend's kernels).
pub trait GpuConv<G: GpuDev>: Send {
    /// Kernel variants the plan can run; executors autotune among them.
    type Variant: Copy + std::fmt::Debug + PartialEq + Send;
    fn geom(&self) -> &ConvGeom;
    fn variants(&self) -> Vec<Self::Variant>;
    fn variant(&self) -> Self::Variant;
    fn set_variant(&mut self, v: Self::Variant);
    /// Enqueue for `n` images; `x` / `y` = (buffer, byte offset of the first channel).
    fn dispatch_at(&self, g: &G, enc: &G::Enc, x: (&G::Buf, u64), y: (&G::Buf, u64), n: usize) -> Result<()>;
    fn device_bytes(&self) -> usize;
}

pub trait GpuDev: KernelRuntime + Sized + Send + 'static {
    type Conv: GpuConv<Self>;
    /// "cuda" / "vulkan" (messages, logs)
    const BACKEND: &'static str;
    /// Conv epilogue code 13 (a per-channel floor after the bias, read from
    /// the bias buffer's second half) is available.
    const CONV_FLOOR: bool = false;
    fn open(ordinal: usize) -> Result<Self>;
    /// Plan a conv reading `input` and writing `output` storage; `act`: 0 none,
    /// 1 SiLU, 2 sigmoid, 3 ReLU (fused with the bias).
    fn conv_plan(&self, geom: ConvGeom, w: &[f32], bias: Option<&[f32]>, act: u32, input: Storage, output: Storage) -> Result<Self::Conv>;
    fn alloc_bytes(&self, len: usize) -> Result<Self::Buf>;
    fn upload_bytes(&self, data: &[u8]) -> Result<Self::Buf>;
    fn write_bytes(&self, b: &mut Self::Buf, off: usize, data: &[u8]) -> Result<()>;
    fn read_bytes(&self, b: &Self::Buf, off: usize, out: &mut [u8]) -> Result<()>;
    fn buf_len(b: &Self::Buf) -> usize;
    /// The address kernels see (frame descriptors carry it).
    fn device_ptr(&self, b: &Self::Buf) -> u64;
    /// A forward pass recorded for replay (CUDA Graphs; `()` where there are none).
    type Graph: Send;
    /// Record what `f` enqueues into a graph; None: this backend has no graphs
    /// (the caller then just runs `f` each time).
    fn capture(&self, f: &mut dyn FnMut(&Self::Enc) -> Result<()>) -> Result<Option<Self::Graph>>;
    /// Run a recorded graph and wait for it.
    fn replay(&self, g: &Self::Graph) -> Result<()>;
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use ojas_cuda::conv::{ConvPlan, ConvVariant};
    use ojas_cuda::{CuBuf, CudaGpu};

    impl GpuConv<CudaGpu> for ConvPlan {
        type Variant = ConvVariant;
        fn geom(&self) -> &ConvGeom {
            &self.geom
        }
        fn variants(&self) -> Vec<ConvVariant> {
            ConvPlan::variants(self)
        }
        fn variant(&self) -> ConvVariant {
            self.variant
        }
        fn set_variant(&mut self, v: ConvVariant) {
            self.variant = v;
        }
        fn dispatch_at(&self, g: &CudaGpu, enc: &<CudaGpu as ojas_core::Device>::Enc, x: (&CuBuf, u64), y: (&CuBuf, u64), n: usize) -> Result<()> {
            ConvPlan::dispatch_at(self, g, enc, x, y, n)
        }
        fn device_bytes(&self) -> usize {
            ConvPlan::device_bytes(self)
        }
    }

    impl GpuDev for CudaGpu {
        type Conv = ConvPlan;
        const BACKEND: &'static str = "cuda";
        const CONV_FLOOR: bool = true;
        fn open(ordinal: usize) -> Result<Self> {
            CudaGpu::new(ordinal)
        }
        fn conv_plan(&self, geom: ConvGeom, w: &[f32], bias: Option<&[f32]>, act: u32, input: Storage, output: Storage) -> Result<ConvPlan> {
            ConvPlan::with_storage(self, geom, w, bias, act, input, output)
        }
        fn alloc_bytes(&self, len: usize) -> Result<CuBuf> {
            CudaGpu::alloc_bytes(self, len)
        }
        fn upload_bytes(&self, data: &[u8]) -> Result<CuBuf> {
            CudaGpu::upload_bytes(self, data)
        }
        fn write_bytes(&self, b: &mut CuBuf, off: usize, data: &[u8]) -> Result<()> {
            CudaGpu::write_bytes(self, b, off, data)
        }
        fn read_bytes(&self, b: &CuBuf, off: usize, out: &mut [u8]) -> Result<()> {
            CudaGpu::read_bytes(self, b, off, out)
        }
        fn buf_len(b: &CuBuf) -> usize {
            b.bytes.len()
        }
        fn device_ptr(&self, b: &CuBuf) -> u64 {
            CudaGpu::device_ptr(self, b)
        }
        type Graph = ojas_cuda::Recorded;
        fn capture(&self, f: &mut dyn FnMut(&<CudaGpu as ojas_core::Device>::Enc) -> Result<()>) -> Result<Option<ojas_cuda::Recorded>> {
            CudaGpu::capture(self, f)
        }
        fn replay(&self, g: &ojas_cuda::Recorded) -> Result<()> {
            CudaGpu::replay(self, g)
        }
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use ojas_vulkan::conv::{VkConvPlan, VkConvVariant};
    use ojas_vulkan::{VkBuf, VkGpu};

    impl GpuConv<VkGpu> for VkConvPlan {
        type Variant = VkConvVariant;
        fn geom(&self) -> &ConvGeom {
            &self.geom
        }
        fn variants(&self) -> Vec<VkConvVariant> {
            VkConvPlan::variants(self)
        }
        fn variant(&self) -> VkConvVariant {
            self.variant
        }
        fn set_variant(&mut self, v: VkConvVariant) {
            self.variant = v;
        }
        fn dispatch_at(&self, g: &VkGpu, enc: &<VkGpu as ojas_core::Device>::Enc, x: (&VkBuf, u64), y: (&VkBuf, u64), n: usize) -> Result<()> {
            VkConvPlan::dispatch_at(self, g, enc, x, y, n)
        }
        fn device_bytes(&self) -> usize {
            VkConvPlan::device_bytes(self)
        }
    }

    impl GpuDev for VkGpu {
        type Conv = VkConvPlan;
        const BACKEND: &'static str = "vulkan";
        fn open(ordinal: usize) -> Result<Self> {
            VkGpu::new(ordinal)
        }
        fn conv_plan(&self, geom: ConvGeom, w: &[f32], bias: Option<&[f32]>, act: u32, input: Storage, output: Storage) -> Result<VkConvPlan> {
            VkConvPlan::with_storage(self, geom, w, bias, act, input, output)
        }
        fn alloc_bytes(&self, len: usize) -> Result<VkBuf> {
            VkGpu::alloc_bytes(self, len)
        }
        fn upload_bytes(&self, data: &[u8]) -> Result<VkBuf> {
            VkGpu::upload_bytes(self, data)
        }
        fn write_bytes(&self, b: &mut VkBuf, off: usize, data: &[u8]) -> Result<()> {
            VkGpu::write_bytes(self, b, off, data)
        }
        fn read_bytes(&self, b: &VkBuf, off: usize, out: &mut [u8]) -> Result<()> {
            VkGpu::read_bytes(self, b, off, out)
        }
        fn buf_len(b: &VkBuf) -> usize {
            b.len
        }
        fn device_ptr(&self, b: &VkBuf) -> u64 {
            VkGpu::device_ptr(self, b)
        }
        type Graph = ();
        fn capture(&self, _f: &mut dyn FnMut(&<VkGpu as ojas_core::Device>::Enc) -> Result<()>) -> Result<Option<()>> {
            Ok(None)
        }
        fn replay(&self, _g: &()) -> Result<()> {
            Ok(())
        }
    }
}
