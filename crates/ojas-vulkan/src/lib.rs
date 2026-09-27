//! ojas-vulkan — the Vulkan compute backend: vendor-neutral GPU inference on
//! any Vulkan 1.3 driver (NVIDIA, AMD RADV, Intel ANV, NVK, Mesa llvmpipe).
//!
//! Same dispatch contract as ojas-cuda ([`ojas_core::KernelRuntime`]):
//! kernels by canonical name, buffers first, then u32 constants, a CUDA-style
//! grid of workgroups × workgroup size. Mapping:
//!
//! - **Buffers are device addresses** (`VK_KHR_buffer_device_address`, core
//!   in 1.2): a kernel receives 64-bit pointers like a CUDA kernel, so
//!   `(buffer, byte offset)` is just `address + offset`, and host-side
//!   descriptor tables of frame pointers work unchanged. No descriptor sets.
//! - **Arguments** travel as push constants: the addresses, then the
//!   constants, in argument order (each shader declares the matching block).
//! - **Workgroup size** is a specialization constant: one pipeline per
//!   (kernel, block) pair, created on first use.
//! - **Encoder** = one command buffer (its own pool, so encoders on different
//!   threads never share a pool); every dispatch is followed by a
//!   compute→compute memory barrier, which gives CUDA's in-stream ordering.
//!   `submit` waits on a fence — the single sync per forward.
//! - **Grids** larger than the device's workgroup-count limit (65,535 on some
//!   drivers) are split with `vkCmdDispatchBase`, invisible to the shader.
//! - Transfers (`upload`, `read`, `write_bytes`, …) are synchronous through a
//!   host-visible staging buffer.
//!
//! Shaders are GLSL, compiled to SPIR-V at build time (see `build.rs`).

use anyhow::{anyhow, bail, ensure, Context, Result};
use ash::vk;
use ojas_core::{Caps, Tier};
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

pub mod conv;

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

/// Every kernel entry compiled into this build (name, SPIR-V).
pub fn kernel_table() -> &'static [(&'static str, &'static [u8])] {
    kernels::KERNELS
}

/// Push-constant bytes every pipeline layout reserves (the Vulkan minimum
/// guarantee is 128; all of today's kernels fit in it).
const PUSH_BYTES: u32 = 128;

/// One Vulkan device as the probe sees it.
#[derive(Clone, Debug)]
pub struct VkDeviceInfo {
    pub index: usize,
    pub name: String,
    /// `NVIDIA`, `radv`, `llvmpipe`, … plus the driver version string
    pub driver: String,
    pub api: String,
    /// discrete / integrated / cpu / virtual / other
    pub kind: String,
    pub memory_bytes: u64,
    pub subgroup: u32,
    pub fp16: bool,
    /// Tensor cores / matrix units through `VK_KHR_cooperative_matrix`.
    pub cooperative_matrix: bool,
    /// ... with the fp16 16x16x16 → fp32 subgroup shape the conv kernel uses.
    pub coopmat_f16_16x16x16: bool,
    pub video_decode_h264: bool,
    pub video_decode_h265: bool,
    pub video_decode_av1: bool,
}

struct Inner {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: Mutex<vk::Queue>,
    qfamily: u32,
    mem: vk::PhysicalDeviceMemoryProperties,
    max_groups: [u32; 3],
    /// Buffers dropped while work that may use them is recorded or in flight
    /// (CUDA frees in stream order; here they wait until no encoder is open).
    graveyard: Mutex<Vec<(vk::Buffer, vk::DeviceMemory)>>,
    /// Encoders begun and not yet finished.
    open: std::sync::atomic::AtomicUsize,
    /// finished encoders (pool, command buffer, fence) kept for reuse:
    /// creating them per submit cost ~1 ms on the 3060 driver
    spare: Mutex<Vec<(vk::CommandPool, vk::CommandBuffer, vk::Fence)>>,
}

impl Inner {
    /// Free dropped buffers once nothing recorded or submitted can use them.
    fn reap(&self) {
        if self.open.load(std::sync::atomic::Ordering::SeqCst) != 0 {
            return;
        }
        let dead: Vec<_> = std::mem::take(&mut *self.graveyard.lock().unwrap());
        for (b, m) in dead {
            unsafe {
                self.device.destroy_buffer(b, None);
                self.device.free_memory(m, None);
            }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for (b, m) in self.graveyard.get_mut().unwrap().drain(..) {
                self.device.destroy_buffer(b, None);
                self.device.free_memory(m, None);
            }
            for (pool, _, fence) in self.spare.get_mut().unwrap().drain(..) {
                self.device.destroy_command_pool(pool, None);
                self.device.destroy_fence(fence, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A device buffer: `addr` is its device address (what kernels receive).
pub struct VkBuf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub addr: u64,
    /// Usable bytes (the allocation is rounded up).
    pub len: usize,
    /// Elements are f16 (`read` converts to f32).
    pub f16: bool,
    inner: Arc<Inner>,
}

impl Drop for VkBuf {
    fn drop(&mut self) {
        // a recorded dispatch may still read it: free after the work is done
        self.inner.graveyard.lock().unwrap().push((self.buffer, self.memory));
        self.inner.reap();
    }
}

/// A chain of dispatches recorded into one command buffer. Dropped without
/// `submit` (an error path), it is discarded and returned for reuse.
pub struct VkEnc {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    inner: Arc<Inner>,
    done: bool,
}

impl Drop for VkEnc {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let d = &self.inner.device;
        unsafe {
            let _ = d.end_command_buffer(self.cmd);
            let _ = d.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty());
        }
        self.inner.spare.lock().unwrap().push((self.pool, self.cmd, self.fence));
        self.inner.open.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.reap();
    }
}

struct Staging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    len: usize,
    /// HOST_COHERENT: no explicit invalidate needed before reading
    coherent: bool,
}

// the mapped pointer is only used under the `staging` mutex
unsafe impl Send for Staging {}

pub struct VkGpu {
    inner: Arc<Inner>,
    info: VkDeviceInfo,
    caps: Caps,
    layout: vk::PipelineLayout,
    modules: Mutex<HashMap<&'static str, vk::ShaderModule>>,
    pipes: Mutex<HashMap<(String, [u32; 3]), vk::Pipeline>>,
    staging: Mutex<Option<Staging>>,
    /// readback staging in host-cached memory (reading write-combined memory
    /// back cost 4.3 ms per MB on the 3060)
    readback: Mutex<Option<Staging>>,
}

impl Drop for VkGpu {
    fn drop(&mut self) {
        let d = &self.inner.device;
        unsafe {
            // other handles share the device: its queue must not be in use while idling
            {
                let _q = self.inner.queue.lock().unwrap();
                let _ = d.device_wait_idle();
            }
            for (_, p) in self.pipes.lock().unwrap().drain() {
                d.destroy_pipeline(p, None);
            }
            for (_, m) in self.modules.lock().unwrap().drain() {
                d.destroy_shader_module(m, None);
            }
            d.destroy_pipeline_layout(self.layout, None);
            for st in [&self.staging, &self.readback] {
                if let Some(s) = st.lock().unwrap().take() {
                    d.unmap_memory(s.memory);
                    d.destroy_buffer(s.buffer, None);
                    d.free_memory(s.memory, None);
                }
            }
        }
    }
}

fn cstr(b: &[std::ffi::c_char]) -> String {
    unsafe { CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned()
}

fn instance(entry: &ash::Entry) -> Result<ash::Instance> {
    let app = vk::ApplicationInfo::default().application_name(c"ojas").api_version(vk::API_VERSION_1_3);
    unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }.context("vkCreateInstance")
}

fn describe(instance: &ash::Instance, pdev: vk::PhysicalDevice, index: usize) -> Result<VkDeviceInfo> {
    let p = unsafe { instance.get_physical_device_properties(pdev) };
    let exts: Vec<String> = unsafe { instance.enumerate_device_extension_properties(pdev)? }.iter().map(|e| cstr(&e.extension_name)).collect();
    let has = |n: &str| exts.iter().any(|e| e == n);
    let mut drv = vk::PhysicalDeviceDriverProperties::default();
    let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut drv).push_next(&mut sg);
    unsafe { instance.get_physical_device_properties2(pdev, &mut p2) };
    let mut f16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f16);
    unsafe { instance.get_physical_device_features2(pdev, &mut f2) };
    let mem = unsafe { instance.get_physical_device_memory_properties(pdev) };
    let memory_bytes = (0..mem.memory_heap_count as usize).filter(|&i| mem.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).map(|i| mem.memory_heaps[i].size).max().unwrap_or(0);
    let kind = match p.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::CPU => "cpu",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        _ => "other",
    };
    Ok(VkDeviceInfo {
        index,
        name: cstr(&p.device_name),
        driver: format!("{} {}", cstr(&drv.driver_name), cstr(&drv.driver_info)).trim().to_string(),
        api: format!("{}.{}.{}", vk::api_version_major(p.api_version), vk::api_version_minor(p.api_version), vk::api_version_patch(p.api_version)),
        kind: kind.into(),
        memory_bytes,
        subgroup: sg.subgroup_size,
        fp16: f16.shader_float16 == vk::TRUE,
        cooperative_matrix: has("VK_KHR_cooperative_matrix"),
        coopmat_f16_16x16x16: has("VK_KHR_cooperative_matrix") && {
            let entry = unsafe { ash::Entry::load() }?;
            // Older layers (the 1.3.204 validation layer of Ubuntu 22.04) do not
            // forward this physical-device query: the loader's trampoline then
            // jumps to null. Under an explicitly enabled validation layer the
            // matrix units are treated as absent (the layer could not check
            // cooperative-matrix shaders anyway).
            let layered = ["VK_INSTANCE_LAYERS", "VK_LOADER_LAYERS_ENABLE"]
                .iter()
                .any(|v| std::env::var(v).is_ok_and(|l| l.to_lowercase().contains("validation")));
            let name = c"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR";
            !layered && unsafe { entry.get_instance_proc_addr(instance.handle(), name.as_ptr()) }.is_some() && {
            let cm = ash::khr::cooperative_matrix::Instance::new(&entry, instance);
            unsafe { cm.get_physical_device_cooperative_matrix_properties(pdev) }.unwrap_or_default().iter().any(|c| {
                c.m_size == 16 && c.n_size == 16 && c.k_size == 16 && c.a_type == vk::ComponentTypeKHR::FLOAT16 && c.b_type == vk::ComponentTypeKHR::FLOAT16
                    && c.c_type == vk::ComponentTypeKHR::FLOAT32 && c.result_type == vk::ComponentTypeKHR::FLOAT32 && c.scope == vk::ScopeKHR::SUBGROUP
            })
            }
        },
        video_decode_h264: has("VK_KHR_video_decode_h264"),
        video_decode_h265: has("VK_KHR_video_decode_h265"),
        video_decode_av1: has("VK_KHR_video_decode_av1"),
    })
}

/// The Vulkan devices of this machine (empty when there is no Vulkan loader).
pub fn devices() -> Vec<VkDeviceInfo> {
    let Ok(entry) = (unsafe { ash::Entry::load() }) else { return vec![] };
    let Ok(inst) = instance(&entry) else { return vec![] };
    let out = unsafe { inst.enumerate_physical_devices() }
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .filter_map(|(i, pd)| describe(&inst, pd, i).ok())
        .collect();
    unsafe { inst.destroy_instance(None) };
    out
}

/// Pick a device: an index, or a case-insensitive part of its name or driver
/// (`"llvmpipe"`, `"radv"`, `"nvidia"`); None = the first discrete GPU, else
/// the first device.
pub fn select(spec: Option<&str>) -> Result<usize> {
    let all = devices();
    ensure!(!all.is_empty(), "no Vulkan device (is a Vulkan loader and driver installed?)");
    Ok(match spec.map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => match s.parse::<usize>() {
            Ok(i) if i < all.len() => i,
            Ok(i) => bail!("Vulkan device {i} does not exist ({} devices)", all.len()),
            Err(_) => {
                let l = s.to_lowercase();
                all.iter().find(|d| d.name.to_lowercase().contains(&l) || d.driver.to_lowercase().contains(&l)).map(|d| d.index).ok_or_else(|| anyhow!("no Vulkan device matches {s:?}"))?
            }
        },
        None => all.iter().find(|d| d.kind == "discrete").unwrap_or(&all[0]).index,
    })
}

/// Live logical devices by physical-device index: every `VkGpu` of one GPU
/// shares one `VkDevice`, so buffers work across executors, pipeline stages
/// and threads (as CUDA's primary context gives ojas-cuda). A separate device
/// per handle made shared buffers undefined behaviour (caught by validation:
/// VUID-vkCmdCopyBuffer-commonparent).
static DEVICES: std::sync::OnceLock<Mutex<HashMap<usize, (std::sync::Weak<Inner>, VkDeviceInfo)>>> = std::sync::OnceLock::new();

fn shared_device(index: usize) -> Result<(Arc<Inner>, VkDeviceInfo)> {
    let mut reg = DEVICES.get_or_init(Mutex::default).lock().unwrap();
    if let Some((w, info)) = reg.get(&index) {
        if let Some(inner) = w.upgrade() {
            return Ok((inner, info.clone()));
        }
    }
    let (inner, info) = create_device(index)?;
    reg.insert(index, (Arc::downgrade(&inner), info.clone()));
    Ok((inner, info))
}

fn create_device(index: usize) -> Result<(Arc<Inner>, VkDeviceInfo)> {
        let entry = unsafe { ash::Entry::load() }.context("no Vulkan loader (libvulkan)")?;
        let instance = instance(&entry)?;
        let pdevs = unsafe { instance.enumerate_physical_devices()? };
        let pdev = *pdevs.get(index).ok_or_else(|| anyhow!("Vulkan device {index} does not exist ({} devices)", pdevs.len()))?;
        let info = describe(&instance, pdev, index)?;
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        ensure!(props.api_version >= vk::API_VERSION_1_3, "{}: Vulkan {} (1.3 needed)", info.name, info.api);
        let qfamily = unsafe { instance.get_physical_device_queue_family_properties(pdev) }
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow!("{}: no compute queue", info.name))? as u32;
        let prio = [1.0f32];
        let qinfo = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfamily).queue_priorities(&prio)];
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true).shader_float16(true).shader_int8(true).storage_buffer8_bit_access(true);
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let base = vk::PhysicalDeviceFeatures::default().shader_int64(true).shader_int16(true);
        let mut cm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let exts: Vec<*const std::ffi::c_char> = if info.coopmat_f16_16x16x16 { vec![ash::khr::cooperative_matrix::NAME.as_ptr()] } else { vec![] };
        let mut dinfo = vk::DeviceCreateInfo::default().queue_create_infos(&qinfo).enabled_features(&base).enabled_extension_names(&exts).push_next(&mut f11).push_next(&mut f12).push_next(&mut f13);
        if info.coopmat_f16_16x16x16 {
            dinfo = dinfo.push_next(&mut cm);
        }
        let device = unsafe { instance.create_device(pdev, &dinfo, None) }.with_context(|| format!("{}: vkCreateDevice (needs buffer device address, fp16, int64)", info.name))?;
        let queue = unsafe { device.get_device_queue(qfamily, 0) };
        let mem = unsafe { instance.get_physical_device_memory_properties(pdev) };
        let inner = Arc::new(Inner { _entry: entry, instance, device, queue: Mutex::new(queue), qfamily, mem, max_groups: props.limits.max_compute_work_group_count, graveyard: Mutex::default(), open: Default::default(), spare: Mutex::default() });
        Ok((inner, info))
}

impl VkGpu {
    /// A handle on Vulkan device `index` (see [`devices`] / [`select`]); handles
    /// of one device share its `VkDevice`, queue and buffers.
    pub fn new(index: usize) -> Result<Self> {
        let (inner, info) = shared_device(index)?;
        let range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(PUSH_BYTES)];
        let layout = unsafe { inner.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&range), None)? };
        let mut features = vec!["vulkan"];
        if info.cooperative_matrix {
            features.push("vk_cooperative_matrix");
        }
        if info.video_decode_h264 {
            features.push("vk_video_decode");
        }
        let caps = Caps {
            tier: if info.cooperative_matrix { Tier::A } else { Tier::B },
            f16_compute: info.fp16,
            simd_width: info.subgroup,
            unified_memory: info.kind != "discrete",
            max_buffer: usize::MAX,
            features,
        };
        Ok(VkGpu { inner, info, caps, layout, modules: Mutex::default(), pipes: Mutex::default(), staging: Mutex::new(None), readback: Mutex::new(None) })
    }

    /// Open the device named by `OJAS_VK_DEVICE` (index or name part), else the default.
    pub fn from_env() -> Result<Self> {
        Self::new(select(std::env::var("OJAS_VK_DEVICE").ok().as_deref())?)
    }

    pub fn info(&self) -> &VkDeviceInfo {
        &self.info
    }

    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Result<u32> {
        let m = &self.inner.mem;
        (0..m.memory_type_count).find(|&i| bits & (1 << i) != 0 && m.memory_types[i as usize].property_flags.contains(want)).ok_or_else(|| anyhow!("no memory type with {want:?}"))
    }

    fn raw_buffer(&self, len: usize, usage: vk::BufferUsageFlags, props: vk::MemoryPropertyFlags, address: bool) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let d = &self.inner.device;
        let buffer = unsafe { d.create_buffer(&vk::BufferCreateInfo::default().size(len as u64).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None)? };
        let req = unsafe { d.get_buffer_memory_requirements(buffer) };
        let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(self.memory_type(req.memory_type_bits, props)?);
        if address {
            ai = ai.push_next(&mut flags);
        }
        let memory = unsafe { d.allocate_memory(&ai, None) }.with_context(|| format!("allocate {len} bytes on {}", self.info.name))?;
        unsafe { d.bind_buffer_memory(buffer, memory, 0)? };
        Ok((buffer, memory))
    }

    /// A zeroed device buffer of `len` bytes.
    pub fn alloc_bytes(&self, len: usize) -> Result<VkBuf> {
        let size = len.max(4).div_ceil(256) * 256;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let (buffer, memory) = self.raw_buffer(size, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;
        let addr = unsafe { self.inner.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) };
        let b = VkBuf { buffer, memory, addr, len, f16: false, inner: self.inner.clone() };
        self.one_shot(|d, cmd| unsafe { d.cmd_fill_buffer(cmd, b.buffer, 0, vk::WHOLE_SIZE, 0) })?;
        Ok(b)
    }

    pub fn upload_bytes(&self, data: &[u8]) -> Result<VkBuf> {
        let mut b = self.alloc_bytes(data.len())?;
        self.write_bytes(&mut b, 0, data)?;
        Ok(b)
    }

    /// The device address of a buffer (what a kernel's pointer argument holds).
    pub fn device_ptr(&self, b: &VkBuf) -> u64 {
        b.addr
    }

    /// Record and run commands now, waiting for them.
    fn one_shot(&self, f: impl FnOnce(&ash::Device, vk::CommandBuffer)) -> Result<()> {
        let enc = self.new_enc()?;
        f(&self.inner.device, enc.cmd);
        self.finish(enc)
    }

    fn with_staging<R>(&self, len: usize, f: impl FnOnce(&Staging) -> Result<R>) -> Result<R> {
        self.with_host(&self.staging, false, len, f)
    }

    /// A mapped host buffer of at least `len` bytes: write-combined for
    /// uploads, host-cached for readback (falling back to what exists).
    fn with_host<R>(&self, slot: &Mutex<Option<Staging>>, cached: bool, len: usize, f: impl FnOnce(&Staging) -> Result<R>) -> Result<R> {
        let mut s = slot.lock().unwrap();
        if s.as_ref().is_none_or(|x| x.len < len) {
            let d = &self.inner.device;
            if let Some(old) = s.take() {
                unsafe {
                    d.unmap_memory(old.memory);
                    d.destroy_buffer(old.buffer, None);
                    d.free_memory(old.memory, None);
                }
            }
            let size = len.max(1 << 20).next_power_of_two();
            let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE;
            let tries: &[vk::MemoryPropertyFlags] = if cached {
                &[hv | vk::MemoryPropertyFlags::HOST_CACHED | vk::MemoryPropertyFlags::HOST_COHERENT, hv | vk::MemoryPropertyFlags::HOST_CACHED, hv | vk::MemoryPropertyFlags::HOST_COHERENT]
            } else {
                &[hv | vk::MemoryPropertyFlags::HOST_COHERENT]
            };
            let (mut made, mut coherent) = (None, true);
            for &props in tries {
                if let Ok(bm) = self.raw_buffer(size, usage, props, false) {
                    coherent = props.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
                    made = Some(bm);
                    break;
                }
            }
            let (buffer, memory) = made.ok_or_else(|| anyhow!("{}: no host-visible memory for staging", self.info.name))?;
            let ptr = unsafe { d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? } as *mut u8;
            *s = Some(Staging { buffer, memory, ptr, len: size, coherent });
        }
        f(s.as_ref().unwrap())
    }

    pub fn write_bytes(&self, b: &mut VkBuf, off: usize, data: &[u8]) -> Result<()> {
        ensure!(off + data.len() <= b.len, "write {}..{} past a {}-byte buffer", off, off + data.len(), b.len);
        if data.is_empty() {
            return Ok(());
        }
        self.with_staging(data.len(), |s| {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), s.ptr, data.len()) };
            let region = [vk::BufferCopy { src_offset: 0, dst_offset: off as u64, size: data.len() as u64 }];
            self.one_shot(|d, cmd| unsafe { d.cmd_copy_buffer(cmd, s.buffer, b.buffer, &region) })
        })
    }

    pub fn read_bytes(&self, b: &VkBuf, off: usize, out: &mut [u8]) -> Result<()> {
        ensure!(off + out.len() <= b.len, "read {}..{} past a {}-byte buffer", off, off + out.len(), b.len);
        if out.is_empty() {
            return Ok(());
        }
        self.with_host(&self.readback, true, out.len(), |s| {
            let region = [vk::BufferCopy { src_offset: off as u64, dst_offset: 0, size: out.len() as u64 }];
            self.one_shot(|d, cmd| unsafe { d.cmd_copy_buffer(cmd, b.buffer, s.buffer, &region) })?;
            if !s.coherent {
                let range = [vk::MappedMemoryRange::default().memory(s.memory).offset(0).size(vk::WHOLE_SIZE)];
                unsafe { self.inner.device.invalidate_mapped_memory_ranges(&range)? };
            }
            unsafe { std::ptr::copy_nonoverlapping(s.ptr, out.as_mut_ptr(), out.len()) };
            Ok(())
        })
    }

    fn new_enc(&self) -> Result<VkEnc> {
        let d = &self.inner.device;
        let reused = self.inner.spare.lock().unwrap().pop();
        let (pool, cmd, fence) = match reused {
            Some((pool, cmd, fence)) => {
                unsafe { d.reset_command_pool(pool, vk::CommandPoolResetFlags::empty())? };
                (pool, cmd, fence)
            }
            None => {
                let pool = unsafe { d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(self.inner.qfamily).flags(vk::CommandPoolCreateFlags::TRANSIENT), None)? };
                let cmd = unsafe { d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))? }[0];
                let fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None)? };
                (pool, cmd, fence)
            }
        };
        unsafe { d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))? };
        self.inner.open.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(VkEnc { pool, cmd, fence, inner: self.inner.clone(), done: false })
    }

    fn finish(&self, mut enc: VkEnc) -> Result<()> {
        enc.done = true;
        let d = &self.inner.device;
        let fence = enc.fence;
        let r = (|| -> Result<()> {
            unsafe { d.end_command_buffer(enc.cmd)? };
            let cmds = [enc.cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
            let res = {
                let q = self.inner.queue.lock().unwrap();
                unsafe { d.queue_submit(*q, &submit, fence) }
            };
            let res = res.and_then(|_| unsafe { d.wait_for_fences(&[fence], true, u64::MAX) }).and_then(|_| unsafe { d.reset_fences(&[fence]) });
            res.map_err(|e| anyhow!("{}: queue submit/wait failed: {e:?}", self.info.name))
        })();
        if r.is_ok() {
            self.inner.spare.lock().unwrap().push((enc.pool, enc.cmd, fence));
        } else {
            unsafe {
                d.destroy_command_pool(enc.pool, None);
                d.destroy_fence(fence, None);
            }
        }
        self.inner.open.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.reap();
        r
    }

    fn pipeline(&self, name: &str, block: [u32; 3]) -> Result<vk::Pipeline> {
        let key = (name.to_string(), block);
        if let Some(p) = self.pipes.lock().unwrap().get(&key) {
            return Ok(*p);
        }
        let module = *self.modules.lock().unwrap().get(name).ok_or_else(|| anyhow!("kernel '{name}' not loaded (ensure_family first) or not ported to Vulkan"))?;
        let entries = [
            vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 },
            vk::SpecializationMapEntry { constant_id: 1, offset: 4, size: 4 },
            vk::SpecializationMapEntry { constant_id: 2, offset: 8, size: 4 },
        ];
        let data: Vec<u8> = block.iter().flat_map(|v| v.to_le_bytes()).collect();
        let spec = vk::SpecializationInfo::default().map_entries(&entries).data(&data);
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main").specialization_info(&spec);
        let ci = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.layout).flags(vk::PipelineCreateFlags::DISPATCH_BASE)];
        let p = unsafe { self.inner.device.create_compute_pipelines(vk::PipelineCache::null(), &ci, None) }.map_err(|(_, e)| anyhow!("pipeline '{name}' {block:?}: {e:?}"))?[0];
        self.pipes.lock().unwrap().insert(key, p);
        Ok(p)
    }
}

impl ojas_core::Device for VkGpu {
    type Buf = VkBuf;
    type Pipeline = vk::Pipeline;
    type Enc = VkEnc;

    fn name(&self) -> String {
        format!("{} ({})", self.info.name, self.info.driver)
    }

    fn kernel_source(&self, _family: &str) -> Option<&'static str> {
        None // SPIR-V is compiled at build time, not from source here
    }

    fn pipeline(&self, _source: &str, entry: &str) -> Result<Self::Pipeline> {
        VkGpu::pipeline(self, entry, [256, 1, 1])
    }

    fn alloc(&self, len_f32: usize) -> Self::Buf {
        self.alloc_bytes(len_f32 * 4).expect("vulkan alloc")
    }

    fn upload(&self, data: &[f32]) -> Self::Buf {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload_bytes(&bytes).expect("vulkan upload")
    }

    fn upload_f16(&self, data: &[f32]) -> Self::Buf {
        let bytes: Vec<u8> = data.iter().flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes()).collect();
        let mut b = self.upload_bytes(&bytes).expect("vulkan upload_f16");
        b.f16 = true;
        b
    }

    fn read(&self, buf: &Self::Buf, out: &mut [f32]) {
        let elem = if buf.f16 { 2 } else { 4 };
        let mut raw = vec![0u8; (out.len() * elem).min(buf.len)];
        self.read_bytes(buf, 0, &mut raw).expect("vulkan read");
        for (i, o) in out.iter_mut().enumerate().take(raw.len() / elem) {
            *o = if buf.f16 { half::f16::from_bits(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])).to_f32() } else { f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()) };
        }
    }
}

impl ojas_core::KernelRuntime for VkGpu {
    /// Load every kernel of `family` (entry names start with `<family>_`).
    fn ensure_family(&mut self, family: &str) -> Result<()> {
        let prefix = format!("{family}_");
        let mut mods = self.modules.lock().unwrap();
        let mut n = 0;
        // matrix-unit kernels (`*_cm*`) only where the device runs them
        let coop = self.info.coopmat_f16_16x16x16;
        for (name, spv) in kernels::KERNELS.iter().filter(|(n, _)| n.starts_with(&prefix) && (coop || !n.contains("_cm"))) {
            if mods.contains_key(name) {
                n += 1;
                continue;
            }
            let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))?;
            let m = unsafe { self.inner.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None) }.with_context(|| format!("shader module {name}"))?;
            mods.insert(name, m);
            n += 1;
        }
        ensure!(n > 0, "family '{family}' has no Vulkan kernels");
        Ok(())
    }

    fn has_kernel(&self, name: &str) -> bool {
        self.modules.lock().unwrap().contains_key(name)
    }

    fn caps(&self) -> &Caps {
        &self.caps
    }

    fn begin(&self) -> Self::Enc {
        self.new_enc().expect("vulkan command buffer")
    }

    fn dispatch(&self, enc: &Self::Enc, name: &str, bufs: &[(&Self::Buf, u64)], consts: &[u32], grid: [u32; 3], block: [u32; 3]) -> Result<()> {
        let mut push: Vec<u8> = Vec::with_capacity(PUSH_BYTES as usize);
        for (b, off) in bufs {
            push.extend_from_slice(&(b.addr + off).to_le_bytes());
        }
        for c in consts {
            push.extend_from_slice(&c.to_le_bytes());
        }
        ensure!(push.len() <= PUSH_BYTES as usize, "'{name}': {} bytes of arguments, the push-constant block holds {PUSH_BYTES}", push.len());
        if grid.contains(&0) {
            return Ok(());
        }
        let pipe = self.pipeline(name, block)?;
        let d = &self.inner.device;
        let cmd = enc.cmd;
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            d.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
            // split grids beyond the device limit; the base offsets gl_WorkGroupID
            let m = self.inner.max_groups;
            let mut z = 0;
            while z < grid[2] {
                let gz = (grid[2] - z).min(m[2]);
                let mut y = 0;
                while y < grid[1] {
                    let gy = (grid[1] - y).min(m[1]);
                    let mut x = 0;
                    while x < grid[0] {
                        let gx = (grid[0] - x).min(m[0]);
                        d.cmd_dispatch_base(cmd, x, y, z, gx, gy, gz);
                        x += gx;
                    }
                    y += gy;
                }
                z += gz;
            }
            // CUDA stream order: the next dispatch sees this one's writes
            let barrier = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier, &[], &[]);
        }
        Ok(())
    }

    fn submit(&self, enc: Self::Enc) -> Result<()> {
        self.finish(enc)
    }
}
