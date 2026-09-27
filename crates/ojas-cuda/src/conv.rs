//! Host-side planning for `cnn_conv_f16` (see `kernels/cnn.rs` for the kernel
//! contract): padded NHWC geometry, the lane-register weight packing, per-tile
//! pixel tables and the launch grid.
//!
//! Storage contract: the input of a k×k, pad-p convolution is an NHWC image
//! padded by `p` on every side plus `spare_cols()` extra zero columns on the
//! right (the "phantom" kernel columns that round `kw·C` up to a multiple of
//! 16 read them; their weights are zero). Output goes into an NHWC image with
//! its own padding `out_pad` (so the next layer can read it in place).

use anyhow::{bail, ensure, Result};
use ojas_core::{Device, KernelRuntime};

use crate::{CuBuf, CudaGpu};

/// Pixels per tile (8 multiply groups of 8).
pub const NPIX: usize = 64;

pub use ojas_core::conv::{ConvGeom, Storage};

/// Weights `[cout, cin, kh, kw]` (NCHW order, f32) → lane register words for
/// output-row groups of 16, patch chunks of 16, 32 lanes, 4 words each.
/// Row groups are padded to a power-of-two multiple so every launch block's
/// groups exist.
pub fn pack_weights(geom: &ConvGeom, w: &[f32]) -> Result<Vec<u32>> {
    let kwe = geom.kw_eff().ok_or_else(|| anyhow::anyhow!("cin {} needs channel padding", geom.cin))?;
    let (kh, kw, cin, cout) = (geom.kh, geom.kw, geom.cin, geom.cout);
    ensure!(w.len() == cout * cin * kh * kw, "weight length {} != {}", w.len(), cout * cin * kh * kw);
    let c16 = geom.patch_len() / 16;
    let groups = packed_groups(cout);
    // KRSC patch order with phantom columns: j = (ky*kwe + kx)*cin + ic
    let bits = |r: usize, j: usize| -> u32 {
        if r >= cout {
            return 0;
        }
        let (kyx, ic) = (j / cin, j % cin);
        let (ky, kx) = (kyx / kwe, kyx % kwe);
        if kx >= kw {
            return 0;
        }
        half::f16::from_f32(w[((r * cin + ic) * kh + ky) * kw + kx]).to_bits() as u32
    };
    let mut out = vec![0u32; groups * c16 * 32 * 4];
    for g in 0..groups {
        for c in 0..c16 {
            for lane in 0..32 {
                let (r, q) = (lane / 4, lane % 4);
                let (r0, r1) = (g * 16 + r, g * 16 + r + 8);
                let (k0, k1) = (c * 16 + 2 * q, c * 16 + 8 + 2 * q);
                let o = ((g * c16 + c) * 32 + lane) * 4;
                out[o] = bits(r0, k0) | bits(r0, k0 + 1) << 16;
                out[o + 1] = bits(r1, k0) | bits(r1, k0 + 1) << 16;
                out[o + 2] = bits(r0, k1) | bits(r0, k1 + 1) << 16;
                out[o + 3] = bits(r1, k1) | bits(r1, k1 + 1) << 16;
            }
        }
    }
    Ok(out)
}

fn packed_groups(cout: usize) -> usize {
    let g = cout.div_ceil(16);
    let ng = g.next_power_of_two().min(8);
    g.div_ceil(ng) * ng
}


/// The tiled implicit-GEMM form of the conv (`cnn_igemm_f16`).
struct GemmPart {
    w: CuBuf,
    xoff: CuBuf,
    yoff: CuBuf,
    toff: CuBuf,
    kpad: usize,
}

/// How a planned conv runs. All variants compute the same values (the conv
/// output is rounded to f16 before the bias in every one); speed differs per
/// layer, so executors autotune (`ConvPlan::variants`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvVariant {
    /// direct kernel with the bias/activation epilogue fused in
    DirectFused,
    /// direct kernel, then the cnn_bias_act_nhwc pass
    DirectSplit,
    /// tiled implicit GEMM with an N tile of 128 / 64 / 32 output channels,
    /// then the cnn_bias_act_nhwc pass
    Gemm(u16),
    /// tiled implicit GEMM with the bias/activation epilogue fused in
    GemmFused(u16),
}

/// A convolution ready to dispatch: uploaded weights/bias/tables + launch data.
pub struct ConvPlan {
    pub geom: ConvGeom,
    /// Standalone-compiled kernel (experiments): used instead of the family entry.
    pub func: Option<cudarc::driver::CudaFunction>,
    wpk: CuBuf,
    bias: CuBuf,
    tpo: CuBuf,
    tpy: CuBuf,
    has_bias: bool,
    act: u32,
    gemm: Option<GemmPart>,
    /// selected variant (default: GEMM when planned, else fused direct)
    pub variant: ConvVariant,
    pub input: Storage,
    pub output: Storage,
    lg: usize,
    tpi: usize,
}

impl ConvPlan {
    /// `act`: 0 none, 1 SiLU, 2 sigmoid (the cnn_bias_act codes).
    /// `out_pad`: zero border of the output image (for the next conv).
    /// Input storage is exactly the padding the conv needs.
    pub fn new(g: &CudaGpu, geom: ConvGeom, w: &[f32], bias: Option<&[f32]>, act: u32, out_pad: usize) -> Result<Self> {
        let input = Storage { c: geom.cin, cs: geom.cin, h: geom.h, w: geom.w, pad: geom.max_pad(), spare: geom.spare_cols() };
        let (oh, ow) = geom.out_hw();
        let output = Storage { c: geom.cout, cs: geom.cout, h: oh, w: ow, pad: out_pad, spare: 0 };
        Self::with_storage(g, geom, w, bias, act, input, output)
    }

    /// General form: read from / write to images with their own padding.
    /// Requires `input.pad` >= every conv pad and enough right-hand columns for the
    /// phantom kernel columns; `input.c == geom.cin`, `output.c == geom.cout`.
    pub fn with_storage(g: &CudaGpu, geom: ConvGeom, w: &[f32], bias: Option<&[f32]>, act: u32, input: Storage, output: Storage) -> Result<Self> {
        if geom.kw_eff().is_none() {
            bail!("conv {geom:?}: kw*cin cannot reach a multiple of 16; widen channel storage");
        }
        ensure!(input.c == geom.cin && input.h == geom.h && input.w == geom.w, "input storage {input:?} vs {geom:?}");
        ensure!(input.cs == input.c || geom.cin % 16 == 0, "conv reading a channel slice needs cin % 16 == 0");
        let [pt, pl, _, pr] = geom.pads;
        ensure!(input.pad >= geom.max_pad(), "input storage pad {} < conv pads {:?}", input.pad, geom.pads);
        // last phantom column read: (ow-1)*sw + (pad_in - pl) + kwe - 1 < wp
        ensure!(input.pad + input.spare >= pr + geom.spare_cols(),
                "input storage lacks {} spare columns", geom.spare_cols());
        let (oh, ow) = geom.out_hw();
        ensure!(output.c == geom.cout && output.h == oh && output.w == ow, "output storage {output:?} vs {geom:?}");
        let wpk = pack_weights(&geom, w)?;
        let (offy, offx) = (input.pad - pt, input.pad - pl);
        let (opix, tpi) = (oh * ow, (oh * ow).div_ceil(NPIX));
        let (mut tpo, mut tpy) = (vec![0u32; tpi * 64], vec![0u32; tpi * 64]);
        for p in 0..opix {
            let (oy, ox) = (p / ow, p % ow);
            tpo[p] = (((oy * geom.sh + offy) * input.wp() + ox * geom.sw + offx) * input.cs) as u32;
            tpy[p] = (((oy + output.pad) * output.wp() + ox + output.pad) * output.cs) as u32;
        }
        let bits = |v: &[u32]| v.iter().map(|&u| f32::from_bits(u)).collect::<Vec<_>>();
        let lg = geom.cout.div_ceil(16).next_power_of_two().min(8).trailing_zeros() as usize;
        // implicit GEMM (chosen at dispatch when the buffer offsets allow
        // 16-byte copies): whole taps of cin % 8 channels
        let taps = geom.kh * geom.kw;
        let gemm_ok = geom.cin % 8 == 0 && input.cs % 8 == 0 && output.cs % 2 == 0 && taps <= 64
            && std::env::var("OJAS_GEMM").map_or(true, |v| v != "0");
        let gemm = if gemm_ok {
            let (cin, cout) = (geom.cin, geom.cout);
            let k = taps * cin;
            let (kpad, npad) = (k.div_ceil(32) * 32, cout.div_ceil(128) * 128);
            let mut wm = vec![0.0f32; npad * kpad];
            for oc in 0..cout {
                for ic in 0..cin {
                    for t in 0..taps {
                        wm[oc * kpad + t * cin + ic] = w[(oc * cin + ic) * taps + t];
                    }
                }
            }
            let (mut xo, mut yo) = (vec![0u32; opix], vec![0u32; opix]);
            for p in 0..opix {
                xo[p] = tpo[p];
                yo[p] = tpy[p];
            }
            let to: Vec<u32> = (0..taps)
                .map(|t| (((t / geom.kw) * input.wp() + t % geom.kw) * input.cs) as u32)
                .collect();
            Some(GemmPart { w: g.upload_f16(&wm), xoff: g.upload(&bits(&xo)), yoff: g.upload(&bits(&yo)),
                            toff: g.upload(&bits(&to)), kpad })
        } else {
            None
        };
        // default before autotuning: the GEMM on wide 1x1 layers (measured)
        let gemm_default = gemm.is_some() && taps == 1 && geom.sh == 1 && geom.sw == 1
            && geom.cin >= 64 && geom.cout >= 64;
        Ok(ConvPlan {
            geom,
            func: None,
            wpk: g.upload(&bits(&wpk)),
            bias: g.upload_f16(bias.unwrap_or(&[0.0])),
            tpo: g.upload(&bits(&tpo)),
            tpy: g.upload(&bits(&tpy)),
            has_bias: bias.is_some(),
            act,
            variant: if gemm_default { ConvVariant::Gemm(128) } else { ConvVariant::DirectFused },
            gemm,
            input,
            output,
            lg,
            tpi,
        })
    }

    /// Device bytes held by this plan (weights in every packed form, tables).
    pub fn device_bytes(&self) -> usize {
        let mut n = [&self.wpk, &self.bias, &self.tpo, &self.tpy].iter().map(|b| b.bytes.len()).sum::<usize>();
        if let Some(gp) = &self.gemm {
            n += [&gp.w, &gp.xoff, &gp.yoff, &gp.toff].iter().map(|b| b.bytes.len()).sum::<usize>();
        }
        n
    }

    /// Variants this plan can run (for autotuning).
    pub fn variants(&self) -> Vec<ConvVariant> {
        let mut v = vec![ConvVariant::DirectFused, ConvVariant::DirectSplit];
        if self.gemm.is_some() {
            let epi = self.has_bias || self.act != 0;
            for bn in [128u16, 64, 32] {
                v.push(ConvVariant::Gemm(bn));
                if epi {
                    v.push(ConvVariant::GemmFused(bn));
                }
            }
        }
        v
    }

    /// Output image size (rows, cols) including its padding.
    pub fn out_padded_hw(&self) -> (usize, usize) {
        (self.output.hp(), self.output.wp())
    }

    /// Enqueue on `enc` for `n` images. `x`: padded NHWC input (see module
    /// docs); `y`: NHWC output of `out_padded_hw()` × cout per image.
    pub fn dispatch(&self, g: &CudaGpu, enc: &<CudaGpu as ojas_core::Device>::Enc, x: &CuBuf, y: &CuBuf, n: usize) -> Result<()> {
        self.dispatch_at(g, enc, (x, 0), (y, 0), n)
    }

    /// `dispatch` with byte offsets into the input/output buffers (channel
    /// slices: offset = first channel × 2 bytes).
    pub fn dispatch_at(&self, g: &CudaGpu, enc: &<CudaGpu as ojas_core::Device>::Enc, x: (&CuBuf, u64), y: (&CuBuf, u64), n: usize) -> Result<()> {
        let geom = &self.geom;
        let (oh, ow) = geom.out_hw();
        let (ng, tpb) = (1usize << self.lg, 8usize >> self.lg);
        let consts = [
            self.input.img(),
            self.input.wp() * self.input.cs,
            geom.kw_eff().unwrap() * geom.cin,
            geom.cout,
            oh * ow,
            geom.patch_len() / 16,
            NPIX,
            n,
            self.lg,
            self.output.img(),
            geom.cin,
            self.input.cs,
        ]
        .map(|c| c as u32);
        let grid = [(n * self.tpi).div_ceil(tpb) as u32, geom.cout.div_ceil(16).div_ceil(ng) as u32, 1];
        let variant = match (self.variant, &self.gemm) {
            (v @ (ConvVariant::Gemm(_) | ConvVariant::GemmFused(_)), Some(_)) if x.1 % 16 == 0 && y.1 % 4 == 0 && self.func.is_none() => v,
            (ConvVariant::Gemm(_), _) => ConvVariant::DirectSplit,
            (ConvVariant::GemmFused(_), _) => ConvVariant::DirectFused,
            (v, _) => v,
        };
        match (&self.gemm, variant) {
            (Some(gp), ConvVariant::Gemm(bn) | ConvVariant::GemmFused(bn)) => {
                let m = n * oh * ow;
                let taps = geom.kh * geom.kw;
                let fused = matches!(variant, ConvVariant::GemmFused(_));
                // 2 stages: 3 or 4 stages cost the second resident block per SM
                let name = format!("cnn_igemm{bn}x2_f16");
                let smem = (2 * (128 + bn as usize) * 40 * 2) as u32;
                let (act, hb) = if fused { (self.act as usize, self.has_bias as usize) } else { (0, 0) };
                let gc = [m, geom.cout, taps * geom.cin, gp.kpad, oh * ow, self.input.img(), self.output.img(),
                          geom.cin, taps, act, hb].map(|c| c as u32);
                let _ = enc;
                g.dispatch_dyn_shared(&name,
                           &[x, (&gp.w, 0), y, (&gp.xoff, 0), (&gp.yoff, 0), (&gp.toff, 0), (&self.bias, 0)], &gc,
                           [m.div_ceil(128) as u32, geom.cout.div_ceil(bn as usize) as u32, 1], [256, 1, 1], smem)?;
                if fused {
                    return Ok(());
                }
            }
            _ => {
                let epi = self.has_bias || self.act != 0;
                let fused = variant == ConvVariant::DirectFused && epi;
                let mut c2 = consts.to_vec();
                c2.extend([self.act, self.has_bias as u32]);
                let bufs = [x, (&self.wpk, 0), y, (&self.tpo, 0), (&self.tpy, 0), (&self.bias, 0)];
                let name = if fused { "cnn_conv_fused_f16" } else { "cnn_conv_f16" };
                match &self.func {
                    Some(f) => g.dispatch_pipeline(f, &bufs, &c2, grid, [256, 1, 1], 0)?,
                    None => g.dispatch(enc, name, &bufs, &c2, grid, [256, 1, 1])?,
                }
                let fused = fused || !epi;
                if fused {
                    return Ok(());
                }
            }
        }
        if self.has_bias || self.act != 0 {
            let o = &self.output;
            let total = n * oh * ow * geom.cout;
            let ep = [total, oh, ow, o.pad, o.wp(), geom.cout, o.img(), self.act as usize, self.has_bias as usize, o.cs]
                .map(|c| c as u32);
            g.dispatch(enc, "cnn_bias_act_nhwc_f16", &[y, (&self.bias, 0)], &ep,
                       [(total as u32).div_ceil(256), 1, 1], [256, 1, 1])?;
        }
        Ok(())
    }
}

/// NCHW f32 → padded NHWC (pad on all sides, `spare` extra right columns),
/// the storage layout `ConvPlan::dispatch` reads. Host helper for tests and
/// the first frame; the executor keeps activations in this layout.
pub fn nchw_to_padded_nhwc(x: &[f32], n: usize, c: usize, h: usize, w: usize, pad: usize, spare: usize, cpad: usize) -> Vec<f32> {
    let (hp, wp, cs) = (h + 2 * pad, w + 2 * pad + spare, cpad.max(c));
    let img = hp * wp * cs;
    let mut out = vec![0.0f32; n * img];
    for i in 0..n {
        for ic in 0..c {
            for r in 0..h {
                let src = ((i * c + ic) * h + r) * w;
                let dst = i * img + ((r + pad) * wp + pad) * cs;
                for cc in 0..w {
                    out[dst + cc * cs + ic] = x[src + cc];
                }
            }
        }
    }
    out
}

/// NHWC (with `pad` border) → NCHW f32, host helper for checks.
pub fn padded_nhwc_to_nchw(y: &[f32], n: usize, c: usize, h: usize, w: usize, pad: usize) -> Vec<f32> {
    let (hp, wp) = (h + 2 * pad, w + 2 * pad);
    let img = hp * wp * c;
    let mut out = vec![0.0f32; n * c * h * w];
    for i in 0..n {
        for r in 0..h {
            for cc in 0..w {
                let src = i * img + ((r + pad) * wp + cc + pad) * c;
                for ic in 0..c {
                    out[((i * c + ic) * h + r) * w + cc] = y[src + ic];
                }
            }
        }
    }
    out
}
