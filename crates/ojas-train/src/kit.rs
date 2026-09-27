//! Kit — the training dispatch seam (pipeline table + `d` convention).
use ojas_metal::{MBuf, MetalGpu};
use anyhow::Result;
use metal::{ComputePipelineState, MTLSize};
use objc::msg_send;
use objc::sel;
use objc::sel_impl;
use std::ffi::c_void;
use std::collections::HashMap;

use ojas_metal::kernels::train::TRAIN_KERNELS;

/// Compiled training pipelines + dispatch helper. Kernel convention: buffers
/// first (in declaration order), then u32 constants.
pub struct Kit {
    pub(crate) p: HashMap<&'static str, ComputePipelineState>,
}

const KIT_KERNELS: &[&str] = &[
    "t_gemm_xwT", "t_gemm_dx", "t_gemm_dw", "t_rms_fwd", "t_rms_bwd", "t_rms_dw",
    "t_gnorm_fwd", "t_gnorm_bwd", "t_gnorm_dnw", "t_gates_fwd", "t_gates_bwd",
    "t_gates_dtb", "t_conv_fwd", "t_conv_bwd", "t_conv_dw", "t_dn_fwd", "t_dn_bwd",
    "t_dnqk_fold", "t_dng_fold", "t_swiglu_fwd", "t_swiglu_bwd", "t_add", "t_copy", "t_fill",
    "t_qsplit", "t_rope", "t_attn_fwd", "t_attngate_fwd", "t_attngate_bwd",
    "t_attn_dscore", "t_attn_dkv", "t_embed_fwd", "t_embed_bwd", "t_adamw", "t_adamw_8h", "t_mm_xwT", "t_mm_dx", "t_mm_dx_h", "t_mm_dw", "t_mm_xwT_h",
    "t_moe_gate_fwd", "t_moe_gate_bwd", "t_moe_acc", "t_moe_rowscale", "t_moe_dgate",
    "t_moe_gather", "t_moe_scatter_k",
    "t_moe_gather_scaled", "t_moe_dgate_gs", "t_moe_scatter_dh2", "t_moe_scatter_dgate",
    "t_scale", "t_lincomb2", "t_sumsq", "t_sumsq_g", "t_scale_rnorm", "t_transpose", "t_muon_update", "t_mm_grp_xwT", "t_flash_attn_fwd",
    "t_moe_route_count", "t_moe_route_offset", "t_moe_route_scatter", "t_moe_tilemap",
    "t_rmsrp", "t_mm_rmsnorm_xwT", "t_flash_mma_fwd", "t_flash_drow", "t_flash_mma_dq", "t_flash_mma_dkv",
    "t_pool_fwd", "t_pool_bwd", "t_xattn_fwd", "t_xattn_dq", "t_xattn_dkv",
];

impl Kit {
    pub fn new(gpu: &MetalGpu) -> Result<Kit> {
        let mut p = HashMap::new();
        for (name, pipe) in gpu.compile_all(TRAIN_KERNELS, |n| KIT_KERNELS.contains(&n))? {
            let key = KIT_KERNELS.iter().find(|k| **k == name).unwrap();
            p.insert(*key, pipe);
        }
        Ok(Kit { p })
    }

    pub(crate) fn d(&self, enc: &metal::ComputeCommandEncoderRef, name: &str,
         bufs: &[(&MBuf, u64)], consts: &[u32], grid: MTLSize, tgs: MTLSize) {
        if skip(name) { return; }
        enc.set_compute_pipeline_state(&self.p[name]);
        for (i, (b, off)) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(&b.buf), *off);
        }
        for (j, c) in consts.iter().enumerate() {
            enc.set_bytes((bufs.len() + j) as u64, 4, c as *const u32 as *const c_void);
        }
        enc.dispatch_thread_groups(grid, tgs);
        unsafe { let _: () = msg_send![enc, memoryBarrierWithScope: 1u64]; }
    }

    /// Variant with f32 constants first, then u32 constants (AdamW kernels).
    pub(crate) fn df(&self, enc: &metal::ComputeCommandEncoderRef, name: &str,
         bufs: &[(&MBuf, u64)], fconsts: &[f32], iconsts: &[u32], grid: MTLSize, tgs: MTLSize) {
        if skip(name) { return; }
        enc.set_compute_pipeline_state(&self.p[name]);
        for (i, (b, off)) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(&b.buf), *off);
        }
        for (j, c) in fconsts.iter().enumerate() {
            enc.set_bytes((bufs.len() + j) as u64, 4, c as *const f32 as *const c_void);
        }
        for (j, c) in iconsts.iter().enumerate() {
            enc.set_bytes((bufs.len() + fconsts.len() + j) as u64, 4, c as *const u32 as *const c_void);
        }
        enc.dispatch_thread_groups(grid, tgs);
        unsafe { let _: () = msg_send![enc, memoryBarrierWithScope: 1u64]; }
    }
}

/// OJAS_SKIP=name1,name2 disables kernels by name. Profiling only: results are wrong while set.
fn skip(name: &str) -> bool {
    static SKIP: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let list = SKIP.get_or_init(|| ojas_core::config::var("OJAS_SKIP")
        .map(|v| v.split(',').map(String::from).collect()).unwrap_or_default());
    list.iter().any(|s| s == name)
}

pub(crate) fn g1(n: usize) -> MTLSize { MTLSize::new(n as u64, 1, 1) }
pub(crate) fn g2(x: usize, y: usize) -> MTLSize { MTLSize::new(x as u64, y as u64, 1) }

