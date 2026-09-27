//! Convolution plans for the Vulkan backend: the weights as a zero-padded f16
//! GEMM matrix plus the pixel / tap offset tables of the implicit GEMM
//! (`cnn_vkconv_f16`), for the padded-NHWC storage the executor keeps
//! (`ojas_core::conv::Storage`). Same contract as ojas-cuda's `ConvPlan`:
//! input/output storages may be channel slices of wider buffers (the dispatch
//! offsets address the first channel), bias + activation fused.

use anyhow::{ensure, Result};
use ojas_core::conv::{ConvGeom, Storage};
use ojas_core::{Device, KernelRuntime};

use crate::{VkBuf, VkGpu};

/// Conv kernels this backend can run for a plan (autotuning picks among them).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkConvVariant {
    /// shared-memory tiled implicit GEMM on the ALUs, 128x64 tiles, fp32 accumulation
    Tiled,
    /// the same on matrix units (cooperative matrix, fp16 16x16x16 → fp32)
    Coop,
}

pub struct VkConvPlan {
    pub geom: ConvGeom,
    pub input: Storage,
    pub output: Storage,
    pub variant: VkConvVariant,
    w: VkBuf,
    bias: VkBuf,
    xoff: VkBuf,
    yoff: VkBuf,
    toff: VkBuf,
    kpad: usize,
    has_bias: bool,
    act: u32,
    /// the device runs `Coop`, with this workgroup (4 subgroups)
    coop_block: Option<u32>,
}

fn u32_buf(g: &VkGpu, v: &[u32]) -> Result<VkBuf> {
    g.upload_bytes(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}

impl VkConvPlan {
    /// `w`: `[cout, cin, kh, kw]` f32; `act`: 0 none, 1 SiLU, 2 sigmoid, 3 ReLU.
    /// `input.pad` must cover every conv pad (taps read the zero border).
    pub fn with_storage(g: &VkGpu, geom: ConvGeom, w: &[f32], bias: Option<&[f32]>, act: u32, input: Storage, output: Storage) -> Result<Self> {
        let (cin, cout, kh, kw) = (geom.cin, geom.cout, geom.kh, geom.kw);
        ensure!(w.len() == cout * cin * kh * kw, "conv weights: {} values for {geom:?}", w.len());
        ensure!(input.c == cin && input.h == geom.h && input.w == geom.w, "input storage {input:?} vs {geom:?}");
        ensure!(input.pad >= geom.max_pad(), "input storage pad {} < conv pads {:?}", input.pad, geom.pads);
        let (oh, ow) = geom.out_hw();
        ensure!(output.c == cout && output.h == oh && output.w == ow, "output storage {output:?} vs {geom:?}");
        let [pt, pl, _, _] = geom.pads;
        let (offy, offx) = (input.pad - pt, input.pad - pl);
        // the right-most tap must stay inside the padded row
        ensure!((ow - 1) * geom.sw + offx + kw <= input.wp(), "conv {geom:?} reads past the storage row {input:?}");
        let taps = kh * kw;
        let k = taps * cin;
        // rows of 32-multiples: the matrix-unit kernel reads whole 32-wide chunks
        let kpad = k.div_ceil(32) * 32;
        let mut wm = vec![0.0f32; cout * kpad];
        for oc in 0..cout {
            for ic in 0..cin {
                for t in 0..taps {
                    wm[oc * kpad + t * cin + ic] = w[(oc * cin + ic) * taps + t];
                }
            }
        }
        let opix = oh * ow;
        let (mut xo, mut yo) = (vec![0u32; opix], vec![0u32; opix]);
        for q in 0..opix {
            let (oy, ox) = (q / ow, q % ow);
            xo[q] = (((oy * geom.sh + offy) * input.wp() + ox * geom.sw + offx) * input.cs) as u32;
            yo[q] = (((oy + output.pad) * output.wp() + ox + output.pad) * output.cs) as u32;
        }
        let to: Vec<u32> = (0..taps).map(|t| (((t / kw) * input.wp() + t % kw) * input.cs) as u32).collect();
        let info = g.info();
        // the matrix-unit kernel is written for 32-wide subgroups (4 per workgroup)
        let coop_block = (info.coopmat_f16_16x16x16 && info.subgroup == 32).then_some(128);
        Ok(VkConvPlan {
            geom,
            input,
            output,
            coop_block,
            // matrix units when present (autotuning may still pick Tiled per layer)
            variant: if coop_block.is_some() { VkConvVariant::Coop } else { VkConvVariant::Tiled },
            w: g.upload_f16(&wm),
            bias: g.upload_f16(bias.unwrap_or(&[0.0])),
            xoff: u32_buf(g, &xo)?,
            yoff: u32_buf(g, &yo)?,
            toff: u32_buf(g, &to)?,
            kpad,
            has_bias: bias.is_some(),
            act,
        })
    }

    pub fn variants(&self) -> Vec<VkConvVariant> {
        let mut v = vec![VkConvVariant::Tiled];
        if self.coop_block.is_some() {
            v.push(VkConvVariant::Coop);
        }
        v
    }

    pub fn device_bytes(&self) -> usize {
        [&self.w, &self.bias, &self.xoff, &self.yoff, &self.toff].iter().map(|b| b.len).sum()
    }

    /// Enqueue for `n` images; `x` / `y` are (buffer, byte offset of the first channel).
    pub fn dispatch_at(&self, g: &VkGpu, enc: &<VkGpu as Device>::Enc, x: (&VkBuf, u64), y: (&VkBuf, u64), n: usize) -> Result<()> {
        let geom = &self.geom;
        let (oh, ow) = geom.out_hw();
        let m = n * oh * ow;
        // wide loads of the input: 8-channel groups (never across a tap), 16-byte aligned
        let vec_a = geom.cin % 8 == 0 && self.input.cs % 8 == 0 && (x.0.addr + x.1) % 16 == 0;
        let consts = [m, geom.cout, geom.kh * geom.kw * geom.cin, self.kpad, oh * ow, self.input.img(), self.output.img(), geom.cin, self.act as usize, self.has_bias as usize, vec_a as usize]
            .map(|c| c as u32);
        let bufs = [x, (&self.w, 0), y, (&self.xoff, 0), (&self.yoff, 0), (&self.toff, 0), (&self.bias, 0)];
        match (self.variant, self.coop_block) {
            (VkConvVariant::Coop, Some(block)) => g.dispatch(enc, "cnn_vkconv_cm_f16", &bufs, &consts, [m.div_ceil(128) as u32, geom.cout.div_ceil(64) as u32, 1], [block, 1, 1]),
            _ => g.dispatch(enc, "cnn_vkconv_f16", &bufs, &consts, [m.div_ceil(128) as u32, geom.cout.div_ceil(64) as u32, 1], [256, 1, 1]),
        }
    }
}
