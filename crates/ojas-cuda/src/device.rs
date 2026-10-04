//! CudaGpu — the CUDA implementation of ojas_core::{Device, KernelRuntime}.
//!
//! Built on cudarc with the "dynamic-loading" feature: the crate compiles on any
//! host (including macOS with no CUDA toolkit); `CudaGpu::new` dlopens the driver
//! at runtime and returns a clear error when no CUDA is present. Kernels are
//! NVRTC-compiled per family from `crate::kernels` at first use.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, ensure, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use ojas_core::{Caps, Device, KernelRuntime, Tier};

/// Untyped device buffer (byte-backed, like Metal's MBuf). `f16` records the
/// element width so `read` can widen back to f32.
pub struct CuBuf {
    pub bytes: CudaSlice<u8>,
    pub f16: bool,
}

/// Launches recorded from this device's stream as a CUDA Graph: a whole
/// forward pass replayed with one launch instead of hundreds.
pub struct Recorded(cudarc::driver::CudaGraph);

// Used by one executor at a time (through &mut); the graph is never shared.
unsafe impl Send for Recorded {}

pub struct CudaGpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// family key -> loaded module
    modules: Mutex<HashMap<String, Arc<CudaModule>>>,
    /// canonical entry name -> function (from ensure_family'd modules)
    funcs: Mutex<HashMap<String, CudaFunction>>,
    caps: Caps,
    arch: &'static str,
    ordinal: usize,
    /// NVRTC include dirs (the PRELUDE needs `cuda_fp16.h`).
    include_paths: Vec<String>,
}

/// Static properties of a device (`CudaGpu::properties`).
#[derive(Debug, Clone)]
pub struct DeviceProps {
    pub name: String,
    pub memory_bytes: usize,
    pub multiprocessors: usize,
    pub compute_capability: (u32, u32),
}

impl CudaGpu {
    /// Number of CUDA devices (0 when the driver is absent).
    pub fn device_count() -> Result<usize> {
        Ok(CudaContext::device_count().map_err(|e| anyhow!("CUDA unavailable: {e:?}"))? as usize)
    }

    /// Name, memory, SM count and compute capability of this device.
    pub fn properties(&self) -> Result<DeviceProps> {
        use cudarc::driver::sys::CUdevice_attribute::*;
        let at = |a| self.ctx.attribute(a).map_err(|e| anyhow!("cuda attribute: {e:?}"));
        let memory_bytes = unsafe { cudarc::driver::result::device::total_mem(self.ctx.cu_device()) }.map_err(|e| anyhow!("cuda total_mem: {e:?}"))?;
        Ok(DeviceProps {
            name: self.ctx.name().map_err(|e| anyhow!("cuda name: {e:?}"))?,
            memory_bytes,
            multiprocessors: at(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize,
            compute_capability: (at(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? as u32, at(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)? as u32),
        })
    }

    /// Open device `ordinal`. Errors with a clear message when the CUDA driver library is
    /// absent (a macOS host, for instance); the crate itself always builds.
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).map_err(|e| {
            anyhow!(
                "CUDA unavailable (driver dlopen/init failed: {e:?}). \
                 ojas-cuda compiles everywhere but runs only on a machine with an \
                 NVIDIA driver; on this host use ojas-metal or ojas-cpu."
            )
        })?;
        // Every submit synchronises its stream, so cudarc's per-buffer event
        // tracking only adds cross-stream waits — which also make stream
        // capture (CUDA Graphs) fail. OJAS_CUDA_EVENT_TRACKING=1 keeps it.
        if !std::env::var("OJAS_CUDA_EVENT_TRACKING").is_ok_and(|v| v == "1") {
            // SAFETY: buffers are only used on this device's own stream, which every submit waits for
            unsafe { ctx.disable_event_tracking() };
        }
        // Own stream per CudaGpu so independent executors / pipelines overlap on
        // the device (the legacy default stream serializes everything).
        let stream = if std::env::var("OJAS_CUDA_DEFAULT_STREAM").is_ok_and(|v| v == "1") {
            ctx.default_stream()
        } else {
            ctx.new_stream().map_err(|e| anyhow!("cuda new_stream: {e:?}"))?
        };
        // sm arch for NVRTC: the device's own compute capability; override
        // with OJAS_CUDA_ARCH (e.g. "sm_89").
        let arch: &'static str = match std::env::var("OJAS_CUDA_ARCH") {
            Ok(a) => Box::leak(a.into_boxed_str()),
            Err(_) => {
                use cudarc::driver::sys::CUdevice_attribute::*;
                let cc = |a| ctx.attribute(a).map_err(|e| anyhow!("cuda attribute: {e:?}"));
                let (major, minor) = (cc(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?, cc(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?);
                Box::leak(format!("sm_{major}{minor}").into_boxed_str())
            }
        };
        // NVRTC has no default search path. OJAS_CUDA_INCLUDE (':'-separated)
        // wins; otherwise the toolkit's include dir from CUDA_HOME / CUDA_PATH.
        let include_paths: Vec<String> = match std::env::var("OJAS_CUDA_INCLUDE") {
            Ok(p) => p.split(':').filter(|s| !s.is_empty()).map(String::from).collect(),
            Err(_) => ["CUDA_HOME", "CUDA_PATH"]
                .iter()
                .find_map(|k| std::env::var(k).ok())
                .map(|root| vec![format!("{root}/include")])
                .unwrap_or_default(),
        };
        let caps = Caps {
            tier: Tier::A,
            f16_compute: true,
            simd_width: 32,
            unified_memory: false,
            max_buffer: usize::MAX,
            features: vec!["cuda_graphs", "kv_context_shift"],
        };
        Ok(CudaGpu { ctx, stream, modules: Mutex::new(HashMap::new()), funcs: Mutex::new(HashMap::new()), caps, arch, include_paths, ordinal })
    }

    /// Compile (NVRTC) and load `source`, once per process per device: every CudaGpu on the
    /// same device shares the primary context, so modules are shared too (a pipeline may plan a
    /// dozen executors on one device).
    fn compile(&self, source: &str) -> Result<Arc<CudaModule>> {
        use std::hash::{Hash, Hasher};
        static CACHE: std::sync::OnceLock<Mutex<HashMap<(usize, &'static str, u64), Arc<CudaModule>>>> = std::sync::OnceLock::new();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut h);
        let key = (self.ordinal, self.arch, h.finish());
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(m) = cache.lock().unwrap().get(&key) {
            return Ok(m.clone());
        }
        let m = self.compile_uncached(source)?;
        cache.lock().unwrap().insert(key, m.clone());
        Ok(m)
    }

    fn compile_uncached(&self, source: &str) -> Result<Arc<CudaModule>> {
        let opts = CompileOptions {
            arch: Some(self.arch),
            include_paths: self.include_paths.clone(),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(source, opts)
            .map_err(|e| anyhow!("NVRTC compile failed: {e:?}"))?;
        self.ctx.load_module(ptx).map_err(|e| anyhow!("module load failed: {e:?}"))
    }

    fn func(&self, name: &str) -> Result<CudaFunction> {
        self.funcs
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("kernel '{name}' not loaded — ensure_family first"))
    }

    /// Opt-in dynamic-shared dispatch for the big-tile kernels (gemm_q4_s8_mma19
    /// family: 48896 B; attention_flash_tc3: ~77 KB). Sets the max-dynamic-shared
    /// attribute and launches with `shared_bytes`. Same arg convention as
    /// `KernelRuntime::dispatch`.
    pub fn dispatch_dyn_shared(
        &self,
        name: &str,
        bufs: &[(&CuBuf, u64)],
        consts: &[u32],
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
    ) -> Result<()> {
        use cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        let f = self.func(name)?;
        f.set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes as i32)
            .map_err(|e| anyhow!("set_attribute({name}): {e:?}"))?;
        self.launch(&f, bufs, consts, grid, block, shared_bytes)
    }

    /// Raw device address of `buf` (for descriptors that point at frames
    /// living in other buffers). Stream-ordered against this CudaGpu's stream.
    pub fn device_ptr(&self, buf: &CuBuf) -> u64 {
        use cudarc::driver::DevicePtr;
        let (p, _sync) = buf.bytes.device_ptr(&self.stream);
        p
    }

    /// Zero `len` bytes of `buf` from byte offset `off`, in stream order.
    pub fn zero_bytes(&self, buf: &CuBuf, off: usize, len: usize) -> Result<()> {
        if len == 0 { return Ok(()); }
        let rc = unsafe { cudarc::driver::sys::cuMemsetD8Async(self.device_ptr(buf) + off as u64, 0, len, self.stream.cu_stream()) };
        ensure!(rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "cuMemsetD8Async: {rc:?}");
        Ok(())
    }

    /// Copy `out.len()` bytes from a raw device address (e.g. a decoder frame).
    pub fn read_ptr(&self, ptr: u64, out: &mut [u8]) -> Result<()> {
        self.bind_thread()?;
        self.stream.synchronize().map_err(|e| anyhow!("cuda sync: {e:?}"))?;
        let rc = unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut _, ptr, out.len()) };
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuMemcpyDtoH: {rc:?}");
        }
        Ok(())
    }

    /// Driver context / stream handles and thread binding, for driver-level
    /// clients such as the NVDEC decoder.
    pub fn cu_ctx(&self) -> cudarc::driver::sys::CUcontext {
        self.ctx.cu_ctx()
    }
    pub fn cu_stream(&self) -> cudarc::driver::sys::CUstream {
        self.stream.cu_stream()
    }
    pub fn bind_thread(&self) -> Result<()> {
        self.ctx.bind_to_thread().map_err(|e| anyhow!("cuda bind_to_thread: {e:?}"))
    }

    /// Raw bytes to a new device buffer (u8 frames, u32 tables, packed f16).
    pub fn upload_bytes(&self, data: &[u8]) -> Result<CuBuf> {
        let bytes = self.stream.memcpy_stod(data).map_err(|e| anyhow!("cuda upload_bytes: {e:?}"))?;
        Ok(CuBuf { bytes, f16: false })
    }

    /// Page-locked host staging buffer.
    ///
    /// Needed by streamed experts: on Apple the weights the GPU reads are already in unified
    /// memory, but on NVIDIA every gathered expert crosses PCIe, and a pageable source forces
    /// the driver to stage through its own pinned bounce buffer, roughly halving bandwidth and
    /// preventing overlap with compute.
    pub fn alloc_pinned(&self, len: usize) -> Result<cudarc::driver::PinnedHostSlice<u8>> {
        // SAFETY: the contents are uninitialised until written, which is why cudarc marks this
        // unsafe; every caller here fills the slice before copying it.
        unsafe { self.ctx.alloc_pinned::<u8>(len.max(1)) }
            .map_err(|e| anyhow!("cuda alloc_pinned({len}): {e:?}"))
    }

    /// Copy a pinned host slice into an existing device buffer on this stream, without waiting.
    /// Pair it with [`CudaGpu::sync`] (or a dependent kernel on the same stream) before reading.
    pub fn upload_pinned_async(&self, src: &cudarc::driver::PinnedHostSlice<u8>, dst: &mut CuBuf)
        -> Result<()> {
        self.stream.memcpy_htod(src, &mut dst.bytes).map_err(|e| anyhow!("cuda async htod: {e:?}"))
    }

    /// Copy pinned staging into `dst` starting at `offset` bytes — the expert-cache slot write.
    pub fn upload_pinned_at(&self, src: &cudarc::driver::PinnedHostSlice<u8>, dst: &mut CuBuf,
                            offset: usize) -> Result<()> {
        let len = src.len();
        let mut view = dst.bytes.slice_mut(offset..offset + len);
        self.stream.memcpy_htod(src, &mut view).map_err(|e| anyhow!("cuda htod at {offset}: {e:?}"))
    }

    /// Wait for everything queued on this stream.
    /// Blocks of `block_size` threads of a compiled kernel that fit on one SM at once (what
    /// its registers and shared memory allow), for the kernel benches.
    pub fn blocks_per_sm(&self, name: &str, block_size: u32) -> Result<u32> {
        let f = self.func(name)?;
        f.occupancy_max_active_blocks_per_multiprocessor(block_size, 0, None).map_err(|e| anyhow!("occupancy: {e:?}"))
    }

    /// Record a timing event on the stream (`CudaEvent::elapsed_ms` between two of them).
    pub fn record_event(&self) -> Result<cudarc::driver::CudaEvent> {
        self.stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(|e| anyhow!("cuda event: {e:?}"))
    }

    /// `dispatch` bracketed by timing events when `prof` is on (`OJAS_CUDA_PROFILE=1`).
    pub fn dispatch_profiled(&self, prof: &KernelProfile, name: &str, bufs: &[(&CuBuf, u64)], consts: &[u32],
                             grid: [u32; 3], block: [u32; 3]) -> Result<()> {
        if prof.launches.borrow().is_none() {
            return self.dispatch(&self.stream, name, bufs, consts, grid, block);
        }
        let start = self.record_event()?;
        self.dispatch(&self.stream, name, bufs, consts, grid, block)?;
        let end = self.record_event()?;
        prof.launches.borrow_mut().as_mut().expect("on").push((name.to_string(), start, end));
        Ok(())
    }
}

/// GPU time per kernel name, from stream events around every launch; on with
/// `OJAS_CUDA_PROFILE=1`, reported and cleared by [`KernelProfile::report`].
pub struct KernelProfile {
    launches: std::cell::RefCell<Option<Vec<(String, cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)>>>,
}

impl KernelProfile {
    pub fn from_env() -> Self {
        KernelProfile { launches: std::cell::RefCell::new(std::env::var("OJAS_CUDA_PROFILE").is_ok().then(Vec::new)) }
    }

    /// Print the time per kernel since the last report, most first, to stderr.
    pub fn report(&self, gpu: &CudaGpu, what: &str) {
        let mut guard = self.launches.borrow_mut();
        let Some(launches) = guard.as_mut() else { return };
        let _ = gpu.sync();
        let mut by_name: HashMap<String, (f64, usize)> = HashMap::new();
        for (name, start, end) in launches.drain(..) {
            let e = by_name.entry(name).or_insert((0.0, 0));
            e.0 += start.elapsed_ms(&end).unwrap_or(0.0) as f64;
            e.1 += 1;
        }
        let mut rows: Vec<(String, (f64, usize))> = by_name.into_iter().collect();
        rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        let total: f64 = rows.iter().map(|r| r.1 .0).sum();
        eprintln!("cuda profile {what}: {total:.1} ms in kernels");
        for (name, (ms, n)) in rows {
            eprintln!("  {name:<24} {ms:>8.2} ms  {n:>5} launches  {:>6.1}%", 100.0 * ms / total);
        }
    }
}

impl CudaGpu {

    pub fn sync(&self) -> Result<()> {
        self.stream.synchronize().map_err(|e| anyhow!("cuda synchronize: {e:?}"))
    }

    /// Zeroed device buffer of `len` bytes.
    pub fn alloc_bytes(&self, len: usize) -> Result<CuBuf> {
        let bytes = self.stream.alloc_zeros::<u8>(len.max(1)).map_err(|e| anyhow!("cuda alloc_bytes: {e:?}"))?;
        Ok(CuBuf { bytes, f16: false })
    }

    /// Keep freed device memory cached in the stream-ordered pool instead of
    /// returning it to the driver at every synchronisation (the default release
    /// threshold is 0): training allocates and frees the same sizes every step.
    pub fn keep_pool_memory(&self) -> Result<()> {
        use cudarc::driver::sys;
        unsafe {
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            sys::cuDeviceGetDefaultMemPool(&mut pool, self.ctx.cu_device()).result().map_err(|e| anyhow!("default mem pool: {e:?}"))?;
            let mut threshold: u64 = u64::MAX;
            sys::cuMemPoolSetAttribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &mut threshold as *mut u64 as *mut std::ffi::c_void)
                .result()
                .map_err(|e| anyhow!("mem pool release threshold: {e:?}"))?;
        }
        Ok(())
    }

    /// Device buffer of `len_f32` floats with unspecified contents — for outputs
    /// a kernel overwrites completely (skips the zero-fill memset).
    pub fn alloc_uninit(&self, len_f32: usize) -> Result<CuBuf> {
        // SAFETY: u8 has no invalid bit patterns; callers only read what a kernel wrote
        let bytes = unsafe { self.stream.alloc::<u8>(len_f32.max(1) * 4) }.map_err(|e| anyhow!("cuda alloc_uninit: {e:?}"))?;
        Ok(CuBuf { bytes, f16: false })
    }

    /// Copy `data` into `buf` at byte offset `off` (stream-ordered: kernels
    /// enqueued afterwards see it). The host slice must stay alive until the
    /// stream reaches the copy; callers pass buffers they own for the call.
    pub fn write_bytes(&self, buf: &mut CuBuf, off: usize, data: &[u8]) -> Result<()> {
        let mut dst = buf.bytes.slice_mut(off..off + data.len());
        self.stream.memcpy_htod(data, &mut dst).map_err(|e| anyhow!("cuda write_bytes: {e:?}"))
    }

    /// Copy `out.len()` bytes from `buf` at byte offset `off` (synchronizes).
    pub fn read_bytes(&self, buf: &CuBuf, off: usize, out: &mut [u8]) -> Result<()> {
        self.stream
            .memcpy_dtoh(&buf.bytes.slice(off..off + out.len()), out)
            .map_err(|e| anyhow!("cuda read_bytes: {e:?}"))?;
        self.stream.synchronize().map_err(|e| anyhow!("cuda sync: {e:?}"))
    }

    /// Launch a function obtained from `Device::pipeline` (ad-hoc source:
    /// probes, benches). Same argument convention as `KernelRuntime::dispatch`.
    pub fn dispatch_pipeline(
        &self,
        f: &CudaFunction,
        bufs: &[(&CuBuf, u64)],
        consts: &[u32],
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
    ) -> Result<()> {
        if shared_bytes > 48 * 1024 {
            use cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
            f.set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes as i32)
                .map_err(|e| anyhow!("set_attribute(max dynamic shared {shared_bytes}): {e:?}"))?;
        }
        self.launch(f, bufs, consts, grid, block, shared_bytes)
    }

    fn launch(
        &self,
        f: &CudaFunction,
        bufs: &[(&CuBuf, u64)],
        consts: &[u32],
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
    ) -> Result<()> {
        let views: Vec<_> = bufs.iter().map(|(b, off)| b.bytes.slice(*off as usize..)).collect();
        let cfg = LaunchConfig {
            grid_dim: (grid[0], grid[1], grid[2]),
            block_dim: (block[0], block[1], block[2]),
            shared_mem_bytes: shared_bytes,
        };
        let mut lb = self.stream.launch_builder(f);
        for v in &views {
            lb.arg(v);
        }
        for c in consts {
            lb.arg(c);
        }
        unsafe { lb.launch(cfg) }.map_err(|e| anyhow!("launch failed: {e:?}"))?;
        Ok(())
    }
}

impl Device for CudaGpu {
    type Buf = CuBuf;
    type Pipeline = CudaFunction;
    type Enc = Arc<CudaStream>;

    fn name(&self) -> String {
        self.ctx.name().unwrap_or_else(|_| "cuda".into())
    }

    fn kernel_source(&self, family: &str) -> Option<&'static str> {
        crate::kernels::family_source(family)
    }

    fn pipeline(&self, source: &str, entry: &str) -> Result<Self::Pipeline> {
        // cache module by source identity (families are 'static, leaked once)
        let key = format!("{:p}:{}", source.as_ptr(), source.len());
        let module = {
            let mut m = self.modules.lock().unwrap();
            match m.get(&key) {
                Some(md) => md.clone(),
                None => {
                    let md = self.compile(source)?;
                    m.insert(key, md.clone());
                    md
                }
            }
        };
        module.load_function(entry).with_context(|| format!("entry '{entry}' not in module"))
    }

    fn alloc(&self, len_f32: usize) -> Self::Buf {
        let bytes = self.stream.alloc_zeros::<u8>(len_f32 * 4).expect("cuda alloc");
        CuBuf { bytes, f16: false }
    }

    fn upload(&self, data: &[f32]) -> Self::Buf {
        let raw: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        let bytes = self.stream.memcpy_stod(raw).expect("cuda upload");
        CuBuf { bytes, f16: false }
    }

    fn upload_f16(&self, data: &[f32]) -> Self::Buf {
        let h: Vec<u8> = data
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let bytes = self.stream.memcpy_stod(&h).expect("cuda upload_f16");
        CuBuf { bytes, f16: true }
    }

    fn read(&self, buf: &Self::Buf, out: &mut [f32]) {
        let elem = if buf.f16 { 2 } else { 4 };
        let mut raw = vec![0u8; out.len() * elem];
        let n = raw.len().min(buf.bytes.len());
        self.stream
            .memcpy_dtoh(&buf.bytes.slice(0..n), &mut raw[..n])
            .expect("cuda read");
        self.stream.synchronize().expect("cuda sync");
        if buf.f16 {
            for (i, o) in out.iter_mut().enumerate() {
                let bits = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
                *o = half::f16::from_bits(bits).to_f32();
            }
        } else {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]]);
            }
        }
    }
}

impl KernelRuntime for CudaGpu {
    fn ensure_family(&mut self, family: &str) -> Result<()> {
        let src = crate::kernels::family_source(family)
            .ok_or_else(|| anyhow!("family '{family}' has no CUDA dialect"))?;
        let module = self.compile(src)?;
        let mut funcs = self.funcs.lock().unwrap();
        for name in crate::kernels::all_names() {
            if crate::kernels::family_of(name) != Some(family) {
                continue;
            }
            let f = module
                .load_function(name)
                .with_context(|| format!("family '{family}' compiled but entry '{name}' missing"))?;
            funcs.insert(name.to_string(), f);
        }
        self.modules.lock().unwrap().insert(format!("family:{family}"), module);
        Ok(())
    }

    fn has_kernel(&self, name: &str) -> bool {
        self.funcs.lock().unwrap().contains_key(name)
    }

    fn caps(&self) -> &Caps {
        &self.caps
    }

    fn begin(&self) -> Self::Enc {
        self.stream.clone()
    }

    fn dispatch(
        &self,
        _enc: &Self::Enc,
        name: &str,
        bufs: &[(&Self::Buf, u64)],
        consts: &[u32],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> Result<()> {
        let f = self.func(name)?;
        self.launch(&f, bufs, consts, grid, block, 0)
    }

    fn submit(&self, enc: Self::Enc) -> Result<()> {
        enc.synchronize().map_err(|e| anyhow!("stream sync failed: {e:?}"))
    }
}

impl CudaGpu {
    /// Record what `f` enqueues on the stream into a graph (nothing runs while recording).
    pub fn capture(&self, f: &mut dyn FnMut(&Arc<CudaStream>) -> Result<()>) -> Result<Option<Recorded>> {
        use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
        self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).map_err(|e| anyhow!("begin capture: {e:?}"))?;
        let r = f(&self.stream);
        let g = self.stream.end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        r?;
        Ok(g.map_err(|e| anyhow!("end capture: {e:?}"))?.map(Recorded))
    }

    /// Run a recorded graph and wait for it.
    pub fn replay(&self, g: &Recorded) -> Result<()> {
        g.0.launch().map_err(|e| anyhow!("graph launch: {e:?}"))?;
        self.stream.synchronize().map_err(|e| anyhow!("stream sync failed: {e:?}"))
    }
}
