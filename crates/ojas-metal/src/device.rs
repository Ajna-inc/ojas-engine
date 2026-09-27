//! The Metal device itself: `MetalGpu`, its `Device`/`KernelRuntime` impls, the compat shim for
//! Apple family 6, and the residency-set helpers. Needs Apple's `metal`/`objc` crates, so it is
//! compiled only on macOS — see `lib.rs` for why `kernels` is not.
#![allow(unexpected_cfgs)] // objc msg_send! expands cfg(cargo-clippy)

use anyhow::{anyhow, Result};
use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device as MtlDevice, MTLResourceOptions};
use objc::{msg_send, sel, sel_impl};
use crate::kernels;
use ojas_core::{Caps, Device, KernelRuntime, Tier};
use std::collections::HashMap;
use std::ffi::c_void;

pub struct MetalGpu {
    pub device: MtlDevice,
    pub queue: CommandQueue,
    /// Resident mode: hand out unretained-reference command buffers.
    unretained_cb: std::sync::atomic::AtomicBool,
    /// Apple family 7+ / Mac2: native simd reductions + simdgroup_matrix MMA.
    pub native_reduce: bool,
    /// Apple family 6: shuffle shim injected at compile; no MMA.
    pub compat_shim: bool,
    caps: Caps,
    families: HashMap<&'static str, &'static str>,
    pipelines: HashMap<String, ComputePipelineState>,
}

/// Pre-family-7 shim: reductions as shuffle butterflies (the builtins exist in the
/// header but fail to lower on those GPUs).
const COMPAT_PRELUDE: &str = r#"
#include <metal_stdlib>
static inline float  __aj_ssum(float v)  { for (ushort o=16;o;o>>=1) v += metal::simd_shuffle_xor(v,o); return v; }
static inline int    __aj_ssum(int v)    { for (ushort o=16;o;o>>=1) v += metal::simd_shuffle_xor(v,o); return v; }
static inline uint   __aj_ssum(uint v)   { for (ushort o=16;o;o>>=1) v += metal::simd_shuffle_xor(v,o); return v; }
static inline float  __aj_smax(float v)  { for (ushort o=16;o;o>>=1) v = metal::max(v, metal::simd_shuffle_xor(v,o)); return v; }
static inline int    __aj_smax(int v)    { for (ushort o=16;o;o>>=1) v = metal::max(v, metal::simd_shuffle_xor(v,o)); return v; }
static inline uint   __aj_smax(uint v)   { for (ushort o=16;o;o>>=1) v = metal::max(v, metal::simd_shuffle_xor(v,o)); return v; }
static inline float  __aj_smin(float v)  { for (ushort o=16;o;o>>=1) v = metal::min(v, metal::simd_shuffle_xor(v,o)); return v; }
static inline int    __aj_smin(int v)    { for (ushort o=16;o;o>>=1) v = metal::min(v, metal::simd_shuffle_xor(v,o)); return v; }
static inline uint   __aj_smin(uint v)   { for (ushort o=16;o;o>>=1) v = metal::min(v, metal::simd_shuffle_xor(v,o)); return v; }
#define simd_sum __aj_ssum
#define simd_max __aj_smax
#define simd_min __aj_smin
"#;

pub struct MBuf {
    pub buf: metal::Buffer,
    pub len: usize,
}

pub struct MetalEnc {
    pub cb: metal::CommandBuffer,
    pub enc: metal::ComputeCommandEncoder,
}

impl MetalGpu {
    pub fn new() -> Result<Self> {
        let device = MtlDevice::system_default().ok_or_else(|| anyhow!("no Metal device"))?;
        let queue = device.new_command_queue();
        // OJAS_GPU_FAMILY=<n> overrides detection (7=native, 6=shim, 5=CPU tier).
        let (native_reduce, compat_shim) = match ojas_core::config::var("OJAS_GPU_FAMILY").ok().and_then(|v| v.parse::<u32>().ok()) {
            Some(fam) => (fam >= 7, fam == 6),
            None => {
                let f7 = device.supports_family(metal::MTLGPUFamily::Apple7)
                      || device.supports_family(metal::MTLGPUFamily::Mac2);
                let f6 = !f7 && device.supports_family(metal::MTLGPUFamily::Apple6);
                (f7, f6)
            }
        };
        let tier = if native_reduce { Tier::A } else if compat_shim { Tier::B } else { Tier::C };
        let caps = Caps {
            tier,
            f16_compute: true,
            simd_width: 32,
            unified_memory: true,
            max_buffer: device.max_buffer_length() as usize,
            features: vec!["metal", "unified_memory"],
        };
        let families: HashMap<&'static str, &'static str> = kernels::families().collect();
        Ok(Self { device, queue, native_reduce, compat_shim, caps,
                  families, pipelines: HashMap::new(),
                  unretained_cb: std::sync::atomic::AtomicBool::new(false) })
    }

    pub fn register_family(&mut self, family: &'static str, source: &'static str) {
        self.families.insert(family, source);
    }

        /// True on Apple7/8 GPUs (M1/M2 generation), where the undocumented
    /// simdgroup async-copy DMA helps GEMM. On Apple9+ (M3/M4) it is documented
    /// as a slowdown, so the async-copy kernels are not even compiled there.
    pub fn async_copy_ok(&self) -> bool {
        // supportsFamily(MTLGPUFamilyApple9 = 1009)
        let apple9: bool = unsafe {
            let r: objc::runtime::BOOL = msg_send![self.device.as_ref(), supportsFamily: 1009u64];
            r != objc::runtime::NO
        };
        !apple9
    }

pub fn compile(&self, src: &str, entry: &str) -> Result<ComputePipelineState> {
        let shimmed;
        let src = if self.compat_shim && !self.native_reduce {
            shimmed = format!("{COMPAT_PRELUDE}\n{src}");
            shimmed.as_str()
        } else { src };
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(true);
        opts.set_language_version(metal::MTLLanguageVersion::V3_1);
        let lib = self.device.new_library_with_source(src, &opts)
            .map_err(|e| anyhow!("MSL compile error: {e}"))?;
        let func = lib.get_function(entry, None)
            .map_err(|e| anyhow!("missing entry {entry}: {e}"))?;
        self.device.new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow!("pipeline error: {e}"))
    }

    pub fn command_buffer(&self) -> &metal::CommandBufferRef {
        // Unretained references (resident mode) don't residency-track referenced
        // buffers, so a graph over 62 GB of experts doesn't re-demand them resident
        // per command buffer. Those buffers must all be in a residency set instead.
        if self.unretained_cb.load(std::sync::atomic::Ordering::Relaxed) {
            self.queue.new_command_buffer_with_unretained_references()
        } else {
            self.queue.new_command_buffer()
        }
    }

    pub fn set_unretained_command_buffers(&self, on: bool) {
        self.unretained_cb.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Replace the command queue with a fresh one. Cheap mitigation for the
    /// shared-gpu corruption that appears after ~1200 command buffers on one
    /// queue: call between independent workloads (e.g. per healed layer) to
    /// start from a clean queue without recompiling any pipelines.
    pub fn reset_queue(&mut self) {
        self.queue = self.device.new_command_queue();
    }

    /// Standalone queue on the same device (for callers that only hold &self).
    pub fn make_queue(&self) -> CommandQueue {
        self.device.new_command_queue()
    }

    /// Compile a family source once and return pipelines for every kernel entry
    /// passing `keep` (vs `compile`, which rebuilds the library per entry).
    pub fn compile_all(&self, src: &str, keep: impl Fn(&str) -> bool) -> Result<Vec<(String, ComputePipelineState)>> {
        let shimmed;
        let src = if self.compat_shim && !self.native_reduce {
            shimmed = format!("{COMPAT_PRELUDE}\n{src}");
            shimmed.as_str()
        } else { src };
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(true);
        opts.set_language_version(metal::MTLLanguageVersion::V3_1);
        let lib = self.device.new_library_with_source(src, &opts)
            .map_err(|e| anyhow!("MSL compile error: {e}"))?;
        let mut out = vec![];
        for name in lib.function_names() {
            if !keep(&name) { continue; }
            let func = lib.get_function(&name, None)
                .map_err(|e| anyhow!("missing entry {name}: {e}"))?;
            let p = self.device.new_compute_pipeline_state_with_function(&func)
                .map_err(|e| anyhow!("pipeline error for {name}: {e}"))?;
            out.push((name.to_string(), p));
        }
        Ok(out)
    }

    pub fn read(&self, b: &MBuf) -> Vec<f32> {
        let ptr = b.buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, b.len) }.to_vec()
    }

    pub fn upload_i8(&self, data: &[i8]) -> MBuf {
        let buf = self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void, data.len() as u64,
            MTLResourceOptions::StorageModeShared);
        MBuf { buf, len: data.len() }
    }

    pub fn upload_u8(&self, data: &[u8]) -> MBuf {
        let buf = self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void, data.len() as u64,
            MTLResourceOptions::StorageModeShared);
        MBuf { buf, len: data.len() }
    }

    /// Zero-filled f16 buffer holding `n` elements (`len` = element count, byte
    /// size = 2n) — same MBuf semantics as `upload_f16` without the Vec churn.
    pub fn alloc_f16_zeroed(&self, n: usize) -> MBuf {
        let buf = self.device.new_buffer((n * 2) as u64, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, n * 2) };
        MBuf { buf, len: n }
    }

    pub fn upload_u32(&self, data: &[u32]) -> MBuf {
        let buf = self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void, (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared);
        MBuf { buf, len: data.len() }
    }
}

impl Device for MetalGpu {
    type Buf = MBuf;
    type Pipeline = ComputePipelineState;
    type Enc = MetalEnc;

    fn name(&self) -> String { self.device.name().to_string() }

    fn kernel_source(&self, family: &str) -> Option<&'static str> {
        self.families.get(family).copied()
    }

    fn pipeline(&self, source: &str, entry: &str) -> Result<ComputePipelineState> {
        self.compile(source, entry)
    }

    fn alloc(&self, len: usize) -> MBuf {
        let buf = self.device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared);
        MBuf { buf, len }
    }

    fn upload(&self, data: &[f32]) -> MBuf {
        let buf = self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void, (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared);
        MBuf { buf, len: data.len() }
    }

    fn upload_f16(&self, data: &[f32]) -> MBuf {
        let h: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
        let buf = self.device.new_buffer_with_data(
            h.as_ptr() as *const c_void, (h.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared);
        MBuf { buf, len: data.len() }
    }

    fn read(&self, b: &MBuf, out: &mut [f32]) {
        let ptr = b.buf.contents() as *const f32;
        let n = out.len().min(b.len);
        out[..n].copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, n) });
    }
}

impl KernelRuntime for MetalGpu {
    fn ensure_family(&mut self, family: &str) -> Result<()> {
        let src = *self.families.get(family)
            .ok_or_else(|| anyhow!("unknown kernel family {family}"))?;
        let entries: Vec<String> = src.match_indices("kernel void ")
            .map(|(i, _)| {
                let rest = &src[i + 12..];
                rest[..rest.find('(').unwrap_or(0)].trim().to_string()
            })
            .collect();
        for e in entries {
            if !self.pipelines.contains_key(&e) {
                let p = self.compile(src, &e)?;
                self.pipelines.insert(e, p);
            }
        }
        Ok(())
    }

    fn has_kernel(&self, name: &str) -> bool { self.pipelines.contains_key(name) }

    fn caps(&self) -> &Caps { &self.caps }

    fn begin(&self) -> MetalEnc {
        let cb = self.queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder().to_owned();
        MetalEnc { cb, enc }
    }

    fn dispatch(&self, enc: &MetalEnc, name: &str,
                bufs: &[(&MBuf, u64)], consts: &[u32],
                grid: [u32; 3], block: [u32; 3]) -> Result<()> {
        let p = self.pipelines.get(name).ok_or_else(|| anyhow!("kernel {name} not compiled"))?;
        enc.enc.set_compute_pipeline_state(p);
        for (i, (b, off)) in bufs.iter().enumerate() {
            enc.enc.set_buffer(i as u64, Some(&b.buf), *off);
        }
        for (j, c) in consts.iter().enumerate() {
            enc.enc.set_bytes((bufs.len() + j) as u64, 4, c as *const u32 as *const c_void);
        }
        enc.enc.dispatch_thread_groups(
            metal::MTLSize::new(grid[0] as u64, grid[1] as u64, grid[2] as u64),
            metal::MTLSize::new(block[0] as u64, block[1] as u64, block[2] as u64));
        unsafe { let _: () = msg_send![&*enc.enc, memoryBarrierWithScope: 1u64]; }
        Ok(())
    }

    fn submit(&self, enc: MetalEnc) -> Result<()> {
        enc.enc.end_encoding();
        enc.cb.commit();
        enc.cb.wait_until_completed();
        Ok(())
    }
}

/// Commit a command buffer, wait for it, and report whether it actually succeeded.
///
/// `wait_until_completed` returns the same for a failed buffer as for a completed one;
/// Metal reports failure only through `status`/`error`. Unchecked, a failed dispatch
/// leaves whatever was already in the output buffers and decoding continues on it, so a
/// real `OutOfMemory` surfaces as fluent, plausible, wrong text instead of an error.
///
/// On failure the fault is latched for this thread (see `ojas_core::device_fault`) so
/// callers that cannot return a `Result` still stop, and the session is marked unusable
/// rather than silently reused.
pub fn commit_and_wait_checked(
    cb: &metal::CommandBufferRef,
    context: &str,
) -> Result<(), ojas_core::device_fault::DeviceError> {
    cb.commit();
    cb.wait_until_completed();
    if cb.status() != metal::MTLCommandBufferStatus::Error {
        return Ok(());
    }
    let detail: *mut objc::runtime::Object = unsafe { msg_send![cb, error] };
    let code: i64 = if detail.is_null() { 0 } else { unsafe { msg_send![detail, code] } };
    let description = match code {
        1 => "Internal", 2 => "Timeout", 3 => "PageFault", 4 => "Blacklisted",
        7 => "NotPermitted", 8 => "OutOfMemory", 9 => "InvalidResource",
        10 => "Memoryless", 11 => "DeviceRemoved", _ => "Unknown",
    };
    // The code alone does not say what happened: "Internal" covers a watchdog
    // timeout, an address fault and a recovery victim alike, and only the
    // driver's text tells them apart ("Caused GPU Timeout Error (...)").
    let driver: String = if detail.is_null() { String::new() } else {
        unsafe {
            let text: *mut objc::runtime::Object = msg_send![detail, localizedDescription];
            if text.is_null() { String::new() } else {
                let utf8: *const std::os::raw::c_char = msg_send![text, UTF8String];
                if utf8.is_null() { String::new() } else {
                    std::ffi::CStr::from_ptr(utf8).to_string_lossy().into_owned()
                }
            }
        }
    };
    let err = ojas_core::device_fault::DeviceError {
        domain: "MTLCommandBufferError".into(),
        code,
        description: if driver.is_empty() { description.into() }
                     else { format!("{description}: {driver}") },
        context: context.into(),
    };
    ojas_core::device_fault::set(err.clone());
    Err(err)
}

/// Compatibility boundary for call sites that cannot yet return a `Result`.
///
/// The fault is latched either way, so the engine stops and the session is marked
/// unusable; this only decides whether the immediate caller learns about it. The public
/// serving path does not rely on catching this, it checks the latch.
pub fn commit_and_wait(cb: &metal::CommandBufferRef) {
    let _ = commit_and_wait_checked(cb, "unlabelled submission");
}

/// A committed `MTLResidencySet` that keeps a set of buffers wired for the process
/// lifetime. Zero-copy mmap buffers are reclaimed between passes unless something
/// requests their residency, so a resident-expert model without one re-faults its whole
/// footprint every token.
///
/// Attaching to the queue makes these allocations resident for every command buffer
/// without naming them. `requestResidency` raises residency priority rather than
/// hard-wiring like `mlock` — the OS still owns eviction under pressure. Requires
/// macOS 15.
pub struct ResidencySet {
    set: *mut objc::runtime::Object,
    queue: metal::CommandQueue,
}
unsafe impl Send for ResidencySet {}
unsafe impl Sync for ResidencySet {}
impl ResidencySet {
    /// Re-request residency. `requestResidency` is a one-shot hint the OS lets
    /// lapse; called each forward to keep the set warm (llama renews on a timer).
    pub fn renew(&self) {
        unsafe { let _: () = msg_send![self.set, requestResidency]; }
    }
}
impl Drop for ResidencySet {
    fn drop(&mut self) {
        use metal::foreign_types::ForeignType;
        unsafe {
            let queue = self.queue.as_ptr() as *mut objc::runtime::Object;
            let _: () = msg_send![queue, removeResidencySet: self.set];
            let _: () = msg_send![self.set, endResidency];
            let _: () = msg_send![self.set, removeAllAllocations];
            let _: () = msg_send![self.set, commit];
            let _: () = msg_send![self.set, release];
        }
    }
}

impl MetalGpu {
    /// Build a residency set over `buffers`, attach it to the queue, and request
    /// residency. `None` if the OS predates `MTLResidencySet` (macOS 15) — the
    /// caller then runs without it and accepts faulting.
    pub fn make_residency_set(&self, buffers: &[&metal::Buffer]) -> Option<ResidencySet> {
        use metal::foreign_types::ForeignType;
        type Obj = *mut objc::runtime::Object;
        unsafe {
            let device: Obj = self.device.as_ptr() as Obj;
            let queue: Obj = self.queue.as_ptr() as Obj;
            let cls = objc::runtime::Class::get("MTLResidencySetDescriptor")?;
            let has: bool = msg_send![device, respondsToSelector: sel!(newResidencySetWithDescriptor:error:)];
            let att: bool = msg_send![queue, respondsToSelector: sel!(addResidencySet:)];
            if !has || !att { return None; }
            let desc: Obj = msg_send![cls, alloc];
            let desc: Obj = msg_send![desc, init];
            if desc.is_null() { return None; }
            let mut err: Obj = std::ptr::null_mut();
            let set: Obj = msg_send![device, newResidencySetWithDescriptor: desc error: &mut err];
            let _: () = msg_send![desc, release];
            if set.is_null() { return None; }
            for b in buffers {
                let bp: Obj = b.as_ptr() as Obj;
                let _: () = msg_send![set, addAllocation: bp];
            }
            let _: () = msg_send![set, commit];
            // Callers decide whether to build a set at all; reaching here means the
            // footprint was judged to fit, so ask for the pages.
            let _: () = msg_send![set, requestResidency];
            let _: () = msg_send![queue, addResidencySet: set];
            Some(ResidencySet { set, queue: self.queue.clone() })
        }
    }
}
