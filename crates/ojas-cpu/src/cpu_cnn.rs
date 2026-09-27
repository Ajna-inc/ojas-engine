//! CPU CNN operators — the reference implementation for the ojas-vision
//! executor. NCHW, f32, static shapes.
//!
//! Conv is a direct (implicit-GEMM) kernel: for each output channel it
//! accumulates weight-scaled input rows into an output row tile, so the inner
//! loop is a contiguous saxpy that autovectorizes. No im2col buffer exists;
//! that keeps peak memory flat across layer shapes. Parallelism comes from
//! `cpu_math::parallel` over (batch × out-channel) rows — never per-op spawns.

use crate::cpu_math::parallel;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Activation fused into conv / gemm epilogues. Mirrors ojas-vision's UnaryOp
/// for the fusable subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None,
    Relu,
    Sigmoid,
    Silu,
    HardSigmoid,
    HardSwish,
    Tanh,
}

#[inline(always)]
pub fn apply_act(a: Act, x: f32) -> f32 {
    match a {
        Act::None => x,
        Act::Relu => x.max(0.0),
        Act::Sigmoid => sigmoid(x),
        // PyTorch 2.8 `silu_kernel` form, not x·σ(x): the two differ in the last bit.
        Act::Silu => x / (1.0 + (-x).exp()),
        Act::HardSigmoid => (x / 6.0 + 0.5).clamp(0.0, 1.0),
        Act::HardSwish => x * (x / 6.0 + 0.5).clamp(0.0, 1.0),
        Act::Tanh => x.tanh(),
    }
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// erf(x), Abramowitz–Stegun 7.1.26 (|err| < 1.5e-7 — below f32 epsilon).
/// ONNX `Erf` / erf-GELU use this, not the ggml tanh-GELU (trap T1).
pub fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// The tanh approximation, as HF / GPT-2 define it.
pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_56 * (x + 0.044715 * x * x * x)).tanh())
}

// ---------------------------------------------------------------------------
// Convolution
// ---------------------------------------------------------------------------

/// Shape/stride description of a 2-D convolution.
#[derive(Debug, Clone, Copy)]
pub struct ConvShape {
    pub n: usize,
    pub cin: usize,
    pub h: usize,
    pub w: usize,
    pub cout: usize,
    pub kh: usize,
    pub kw: usize,
    pub group: usize,
    pub stride: [usize; 2],
    /// [top, left, bottom, right]
    pub pads: [usize; 4],
    pub dilation: [usize; 2],
}

impl ConvShape {
    pub fn out_hw(&self) -> (usize, usize) {
        let eff_h = (self.kh - 1) * self.dilation[0] + 1;
        let eff_w = (self.kw - 1) * self.dilation[1] + 1;
        let oh = (self.h + self.pads[0] + self.pads[2] - eff_h) / self.stride[0] + 1;
        let ow = (self.w + self.pads[1] + self.pads[3] - eff_w) / self.stride[1] + 1;
        (oh, ow)
    }
}

/// Pixels per GEMM microkernel tile (4 NEON vectors worth of f32).
const PR: usize = 16;
/// Output channels per microkernel step.
const MR: usize = 4;
/// Output pixels per im2col panel (one parallel job).
const TILE_P: usize = 128;

/// Direct conv: weights `[cout, cin/group, kh, kw]`, x `[n, cin, h, w]`,
/// output `[n, cout, oh, ow]`, bias per out-channel, `act` fused.
///
/// Depthwise runs a direct loop; everything else goes through a row-tiled
/// im2col panel and a register-blocked GEMM microkernel (`MR`×`PR`
/// accumulators, k innermost), so peak im2col memory stays at one panel per
/// worker regardless of layer shape.
pub fn conv2d(x: &[f32], w: &[f32], bias: Option<&[f32]>, s: &ConvShape, act: Act, threads: usize, out: &mut [f32]) {
    let (oh, ow) = s.out_hw();
    let cin_g = s.cin / s.group;
    let cout_g = s.cout / s.group;
    debug_assert_eq!(x.len(), s.n * s.cin * s.h * s.w);
    debug_assert_eq!(w.len(), s.cout * cin_g * s.kh * s.kw);
    debug_assert_eq!(out.len(), s.n * s.cout * oh * ow);

    if cin_g == 1 && cout_g == 1 {
        return conv2d_depthwise(x, w, bias, s, act, threads, out);
    }

    let plane = oh * ow;
    let k_len = cin_g * s.kh * s.kw;
    let tiles = plane.div_ceil(TILE_P);
    let jobs = s.n * s.group * tiles;
    let out_base = out.as_mut_ptr() as usize;
    let next = AtomicUsize::new(0);

    thread_local! {
        static PANEL: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    let body = |_id: usize, _nt: usize| {
        PANEL.with(|panel_cell| {
            let mut panel = panel_cell.borrow_mut();
            loop {
                let job = next.fetch_add(1, Ordering::Relaxed);
                if job >= jobs {
                    break;
                }
                let bi = job / (s.group * tiles);
                let g = (job / tiles) % s.group;
                let tile = job % tiles;
                let p0 = tile * TILE_P;
                let pt = TILE_P.min(plane - p0);
                panel.clear();
                panel.resize(k_len * pt, 0.0);
                build_panel(x, s, bi, g, cin_g, ow, p0, pt, &mut panel);
                for co0 in (0..cout_g).step_by(MR) {
                    let mr = MR.min(cout_g - co0);
                    let co_abs = g * cout_g + co0;
                    // SAFETY: each (job, co-block) pair owns the disjoint
                    // region out[(bi·cout+co_abs+m)·plane + p0 ..+pt], m < mr.
                    let out_ptr = unsafe { (out_base as *mut f32).add((bi * s.cout + co_abs) * plane + p0) };
                    let wrows = &w[co_abs * k_len..(co_abs + mr) * k_len];
                    let biases: [f32; MR] =
                        std::array::from_fn(|m| bias.map_or(0.0, |b| b[co_abs + m.min(mr - 1)]));
                    gemm_tile(wrows, k_len, mr, &panel, pt, &biases, act, out_ptr, plane);
                }
            }
        });
    };
    parallel(threads.min(jobs).max(1), &body);
}

/// im2col for one tile: panel[k][p] = x(bi, g·cin_g+ci, iy, ix) for the
/// pixel range [p0, p0+pt), zero where the window is in padding.
#[inline]
fn build_panel(x: &[f32], s: &ConvShape, bi: usize, g: usize, cin_g: usize, ow: usize, p0: usize, pt: usize, panel: &mut [f32]) {
    let (sh, sw) = (s.stride[0], s.stride[1]);
    let (dh, dw) = (s.dilation[0], s.dilation[1]);
    let mut k = 0usize;
    for ci in 0..cin_g {
        let xc = &x[(bi * s.cin + g * cin_g + ci) * s.h * s.w..(bi * s.cin + g * cin_g + ci + 1) * s.h * s.w];
        for ky in 0..s.kh {
            for kx in 0..s.kw {
                let row = &mut panel[k * pt..(k + 1) * pt];
                let off = (kx * dw) as isize - s.pads[1] as isize;
                let mut p = 0usize;
                while p < pt {
                    let oy = (p0 + p) / ow;
                    let ox = (p0 + p) % ow;
                    let iy = (oy * sh + ky * dh) as isize - s.pads[0] as isize;
                    // pixels remaining in this output row (and this tile)
                    let run = (ow - ox).min(pt - p);
                    if iy < 0 || iy >= s.h as isize {
                        row[p..p + run].fill(0.0);
                        p += run;
                        continue;
                    }
                    let xrow = &xc[iy as usize * s.w..(iy as usize + 1) * s.w];
                    if sw == 1 {
                        // stride 1: ix = ox + j + off is contiguous — clip once, memcpy
                        let lo = (-(ox as isize + off)).clamp(0, run as isize) as usize;
                        let hi = (s.w as isize - (ox as isize + off)).clamp(0, run as isize) as usize;
                        row[p..p + lo].fill(0.0);
                        if hi > lo {
                            let src0 = (ox as isize + lo as isize + off) as usize;
                            row[p + lo..p + hi].copy_from_slice(&xrow[src0..src0 + hi - lo]);
                        }
                        row[p + hi.max(lo)..p + run].fill(0.0);
                    } else {
                        for (j, slot) in row[p..p + run].iter_mut().enumerate() {
                            let ix = ((ox + j) * sw) as isize + off;
                            *slot = if ix < 0 || ix >= s.w as isize { 0.0 } else { xrow[ix as usize] };
                        }
                    }
                    p += run;
                }
                k += 1;
            }
        }
    }
}

/// C[m][p] = Σ_k W[m][k] · B[k][p] (+bias, act), stored to
/// `out[m·row_stride + p]` for m < mr.
///
/// aarch64 uses explicit NEON fma with an MR×4-vector accumulator block
/// (the scalar form would not stay in registers); other targets take the
/// autovectorized fallback. Dead accumulator rows (m ≥ mr) compute on the last
/// live weight row and are not stored.
///
/// SAFETY: the caller guarantees `out[m·row_stride .. m·row_stride + pt]`
/// for m < mr are valid and unaliased by any concurrent writer.
#[cfg(target_arch = "aarch64")]
#[inline]
fn gemm_tile(
    wrows: &[f32],
    k_len: usize,
    mr: usize,
    panel: &[f32],
    pt: usize,
    biases: &[f32; MR],
    act: Act,
    out: *mut f32,
    row_stride: usize,
) {
    use std::arch::aarch64::*;
    debug_assert_eq!(PR, 16);
    debug_assert!(wrows.len() >= mr * k_len && panel.len() >= k_len * pt);
    let mut p = 0usize;
    unsafe {
        while p + PR <= pt {
            let mut acc: [[float32x4_t; 4]; MR] = [[vdupq_n_f32(0.0); 4]; MR];
            let mut bp = panel.as_ptr().add(p);
            for k in 0..k_len {
                let b0 = vld1q_f32(bp);
                let b1 = vld1q_f32(bp.add(4));
                let b2 = vld1q_f32(bp.add(8));
                let b3 = vld1q_f32(bp.add(12));
                bp = bp.add(pt);
                for m in 0..MR {
                    let wv = vdupq_n_f32(*wrows.get_unchecked(m.min(mr - 1) * k_len + k));
                    acc[m][0] = vfmaq_f32(acc[m][0], wv, b0);
                    acc[m][1] = vfmaq_f32(acc[m][1], wv, b1);
                    acc[m][2] = vfmaq_f32(acc[m][2], wv, b2);
                    acc[m][3] = vfmaq_f32(acc[m][3], wv, b3);
                }
            }
            for (m, a) in acc.iter().enumerate().take(mr) {
                let bv = vdupq_n_f32(biases[m]);
                let dst = out.add(m * row_stride + p);
                if act == Act::None {
                    vst1q_f32(dst, vaddq_f32(a[0], bv));
                    vst1q_f32(dst.add(4), vaddq_f32(a[1], bv));
                    vst1q_f32(dst.add(8), vaddq_f32(a[2], bv));
                    vst1q_f32(dst.add(12), vaddq_f32(a[3], bv));
                } else {
                    let mut tmp = [0f32; PR];
                    vst1q_f32(tmp.as_mut_ptr(), vaddq_f32(a[0], bv));
                    vst1q_f32(tmp.as_mut_ptr().add(4), vaddq_f32(a[1], bv));
                    vst1q_f32(tmp.as_mut_ptr().add(8), vaddq_f32(a[2], bv));
                    vst1q_f32(tmp.as_mut_ptr().add(12), vaddq_f32(a[3], bv));
                    for (j, &v) in tmp.iter().enumerate() {
                        *dst.add(j) = apply_act(act, v);
                    }
                }
            }
            p += PR;
        }
        gemm_tile_tail(wrows, k_len, mr, panel, pt, biases, act, out, row_stride, p);
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn gemm_tile(
    wrows: &[f32],
    k_len: usize,
    mr: usize,
    panel: &[f32],
    pt: usize,
    biases: &[f32; MR],
    act: Act,
    out: *mut f32,
    row_stride: usize,
) {
    let mut p = 0usize;
    while p + PR <= pt {
        let mut acc = [[0.0f32; PR]; MR];
        for k in 0..k_len {
            let b = &panel[k * pt + p..k * pt + p + PR];
            for m in 0..MR {
                let wv = wrows[(m.min(mr - 1)) * k_len + k];
                let a = &mut acc[m];
                for j in 0..PR {
                    a[j] += wv * b[j];
                }
            }
        }
        for (m, a) in acc.iter().enumerate().take(mr) {
            let dst = unsafe { std::slice::from_raw_parts_mut(out.add(m * row_stride + p), PR) };
            for j in 0..PR {
                dst[j] = apply_act(act, a[j] + biases[m]);
            }
        }
        p += PR;
    }
    gemm_tile_tail(wrows, k_len, mr, panel, pt, biases, act, out, row_stride, p);
}

/// Scalar pixel tail shared by both microkernels.
#[allow(clippy::too_many_arguments)]
#[inline]
fn gemm_tile_tail(
    wrows: &[f32],
    k_len: usize,
    mr: usize,
    panel: &[f32],
    pt: usize,
    biases: &[f32; MR],
    act: Act,
    out: *mut f32,
    row_stride: usize,
    mut p: usize,
) {
    while p < pt {
        for m in 0..mr {
            let mut accv = biases[m];
            for k in 0..k_len {
                accv += wrows[m * k_len + k] * panel[k * pt + p];
            }
            unsafe { *out.add(m * row_stride + p) = apply_act(act, accv) };
        }
        p += 1;
    }
}

/// Depthwise (group == cin == cout): direct accumulation per plane.
fn conv2d_depthwise(x: &[f32], w: &[f32], bias: Option<&[f32]>, s: &ConvShape, act: Act, threads: usize, out: &mut [f32]) {
    let (oh, ow) = s.out_hw();
    let jobs = s.n * s.cout;
    let out_base = out.as_mut_ptr() as usize;
    let next = AtomicUsize::new(0);
    let body = |_id: usize, _nt: usize| loop {
        let job = next.fetch_add(1, Ordering::Relaxed);
        if job >= jobs {
            break;
        }
        let (bi, co) = (job / s.cout, job % s.cout);
        // SAFETY: each job writes exactly one disjoint [oh*ow] plane.
        let plane =
            unsafe { std::slice::from_raw_parts_mut((out_base as *mut f32).add((bi * s.cout + co) * oh * ow), oh * ow) };
        let b = bias.map_or(0.0, |b| b[co]);
        plane.fill(b);
        let wg = &w[co * s.kh * s.kw..(co + 1) * s.kh * s.kw];
        let xc = &x[(bi * s.cin + co) * s.h * s.w..];
        for ky in 0..s.kh {
            for kx in 0..s.kw {
                let wv = wg[ky * s.kw + kx];
                if wv == 0.0 {
                    continue;
                }
                accumulate_row(plane, xc, wv, s, oh, ow, ky, kx);
            }
        }
        if act != Act::None {
            for v in plane.iter_mut() {
                *v = apply_act(act, *v);
            }
        }
    };
    parallel(threads.min(jobs).max(1), &body);
}

/// plane[oy, ox] += wv * x[iy, ix] over the valid output range of one (ky,kx).
#[inline]
fn accumulate_row(plane: &mut [f32], xc: &[f32], wv: f32, s: &ConvShape, oh: usize, ow: usize, ky: usize, kx: usize) {
    let (sh, sw) = (s.stride[0], s.stride[1]);
    let (dh, dw) = (s.dilation[0], s.dilation[1]);
    let (pt, pl) = (s.pads[0] as isize, s.pads[1] as isize);
    for oy in 0..oh {
        let iy = oy as isize * sh as isize + (ky * dh) as isize - pt;
        if iy < 0 || iy >= s.h as isize {
            continue;
        }
        // valid ox range: 0 <= ox*sw + kx*dw - pl < w
        let off = (kx * dw) as isize - pl;
        let ox_lo = if off >= 0 { 0 } else { ((-off) as usize).div_ceil(sw) };
        let ox_hi_excl = {
            // ox*sw + off <= w-1  →  ox <= (w-1-off)/sw
            let top = s.w as isize - 1 - off;
            if top < 0 {
                0
            } else {
                ((top as usize) / sw + 1).min(ow)
            }
        };
        if ox_lo >= ox_hi_excl {
            continue;
        }
        let dst = &mut plane[oy * ow + ox_lo..oy * ow + ox_hi_excl];
        let src_start = (iy as usize) * s.w + (ox_lo as isize * sw as isize + off) as usize;
        if sw == 1 {
            let src = &xc[src_start..src_start + dst.len()];
            for (d, &v) in dst.iter_mut().zip(src) {
                *d += wv * v;
            }
        } else {
            let src = &xc[src_start..];
            for (i, d) in dst.iter_mut().enumerate() {
                *d += wv * src[i * sw];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct PoolShape {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub pads: [usize; 4],
    /// ONNX ceil_mode: round the output extent up; windows starting past the
    /// input + left pad are dropped (the torch rule).
    pub ceil: bool,
}

/// One pooled axis extent, floor or ceil mode.
pub fn pool_out(input: usize, kernel: usize, pad_lo: usize, pad_hi: usize, stride: usize, ceil: bool) -> usize {
    let padded = input + pad_lo + pad_hi;
    let span = padded.saturating_sub(kernel);
    let mut out = if ceil { span.div_ceil(stride) + 1 } else { span / stride + 1 };
    if ceil && (out - 1) * stride >= input + pad_lo {
        out -= 1; // last window would start entirely in the right padding
    }
    out
}

impl PoolShape {
    pub fn out_hw(&self) -> (usize, usize) {
        (
            pool_out(self.h, self.kernel[0], self.pads[0], self.pads[2], self.stride[0], self.ceil),
            pool_out(self.w, self.kernel[1], self.pads[1], self.pads[3], self.stride[1], self.ceil),
        )
    }
}

pub fn maxpool2d(x: &[f32], s: &PoolShape, threads: usize, out: &mut [f32]) {
    pool2d(x, s, threads, out, true, false)
}

pub fn avgpool2d(x: &[f32], s: &PoolShape, count_include_pad: bool, threads: usize, out: &mut [f32]) {
    pool2d(x, s, threads, out, false, count_include_pad)
}

fn pool2d(x: &[f32], s: &PoolShape, threads: usize, out: &mut [f32], is_max: bool, count_include_pad: bool) {
    let (oh, ow) = s.out_hw();
    let jobs = s.n * s.c;
    let out_base = out.as_mut_ptr() as usize;
    let next = AtomicUsize::new(0);
    let body = |_id: usize, _nt: usize| loop {
        let job = next.fetch_add(1, Ordering::Relaxed);
        if job >= jobs {
            break;
        }
        let xc = &x[job * s.h * s.w..(job + 1) * s.h * s.w];
        // SAFETY: disjoint planes per job.
        let plane = unsafe { std::slice::from_raw_parts_mut((out_base as *mut f32).add(job * oh * ow), oh * ow) };
        for oy in 0..oh {
            for ox in 0..ow {
                let y0 = oy as isize * s.stride[0] as isize - s.pads[0] as isize;
                let x0 = ox as isize * s.stride[1] as isize - s.pads[1] as isize;
                let mut acc = if is_max { f32::NEG_INFINITY } else { 0.0 };
                let mut cnt = 0usize;
                for ky in 0..s.kernel[0] {
                    let iy = y0 + ky as isize;
                    if iy < 0 || iy >= s.h as isize {
                        continue;
                    }
                    for kx in 0..s.kernel[1] {
                        let ix = x0 + kx as isize;
                        if ix < 0 || ix >= s.w as isize {
                            continue;
                        }
                        let v = xc[iy as usize * s.w + ix as usize];
                        if is_max {
                            // PyTorch `max_pool_forward_nchw`: strictly greater wins, NaN always wins.
                            if v > acc || v.is_nan() {
                                acc = v;
                            }
                        } else {
                            acc += v;
                        }
                        cnt += 1;
                    }
                }
                plane[oy * ow + ox] = if is_max {
                    acc
                } else {
                    let d = if count_include_pad { s.kernel[0] * s.kernel[1] } else { cnt.max(1) };
                    acc / d as f32
                };
            }
        }
    };
    parallel(threads.min(jobs).max(1), &body);
}

pub fn global_avgpool(x: &[f32], n: usize, c: usize, hw: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), n * c);
    for (i, o) in out.iter_mut().enumerate() {
        let p = &x[i * hw..(i + 1) * hw];
        *o = p.iter().sum::<f32>() / hw as f32;
    }
    let _ = (n, c);
}

// ---------------------------------------------------------------------------
// Resize (nearest, asymmetric/floor — the Ultralytics export; trap T4)
// ---------------------------------------------------------------------------

pub fn resize_nearest(x: &[f32], planes: usize, h: usize, w: usize, sh: usize, sw: usize, out: &mut [f32]) {
    let (oh, ow) = (h * sh, w * sw);
    debug_assert_eq!(out.len(), planes * oh * ow);
    for p in 0..planes {
        let src = &x[p * h * w..(p + 1) * h * w];
        let dst = &mut out[p * oh * ow..(p + 1) * oh * ow];
        for oy in 0..oh {
            let iy = oy / sh; // asymmetric + floor
            let srow = &src[iy * w..(iy + 1) * w];
            let drow = &mut dst[oy * ow..(oy + 1) * ow];
            if sw == 2 {
                for (i, &v) in srow.iter().enumerate() {
                    drow[2 * i] = v;
                    drow[2 * i + 1] = v;
                }
            } else {
                for (ox, d) in drow.iter_mut().enumerate() {
                    *d = srow[ox / sw];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Elementwise with numpy broadcast
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
}

#[inline(always)]
fn bin_apply(op: BinOp, a: f32, b: f32) -> f32 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Pow => a.powf(b),
        BinOp::Max => a.max(b),
        BinOp::Min => a.min(b),
    }
}

/// out[shape] = a[sa broadcast] op b[sb broadcast]. `sa`/`sb` are read strides
/// aligned to `shape`'s rank, 0 on broadcast axes (see ir::broadcast_strides).
pub fn binary_bcast(op: BinOp, a: &[f32], sa: &[usize], b: &[f32], sb: &[usize], shape: &[usize], out: &mut [f32]) {
    let rank = shape.len();
    if rank == 0 {
        out[0] = bin_apply(op, a[0], b[0]);
        return;
    }
    // Fast paths: identical layout, or right-hand scalar.
    let numel: usize = shape.iter().product();
    let contig = crate::cpu_cnn::contiguous_strides(shape);
    if sa == contig && sb == contig {
        for i in 0..numel {
            out[i] = bin_apply(op, a[i], b[i]);
        }
        return;
    }
    if sa == contig && sb.iter().all(|&s| s == 0) {
        let bv = b[0];
        for i in 0..numel {
            out[i] = bin_apply(op, a[i], bv);
        }
        return;
    }
    // General: iterate all but the innermost axis with an index vector.
    let inner = shape[rank - 1];
    let (ia, ib) = (sa[rank - 1], sb[rank - 1]);
    let outer: usize = numel / inner.max(1);
    let mut idx = vec![0usize; rank.saturating_sub(1)];
    let mut off_a = 0usize;
    let mut off_b = 0usize;
    let mut o = 0usize;
    for _ in 0..outer {
        let dst = &mut out[o..o + inner];
        if ia == 1 && ib == 1 {
            let (ra, rb) = (&a[off_a..off_a + inner], &b[off_b..off_b + inner]);
            for ((d, &x), &y) in dst.iter_mut().zip(ra).zip(rb) {
                *d = bin_apply(op, x, y);
            }
        } else if ia == 1 && ib == 0 {
            let bv = b[off_b];
            let ra = &a[off_a..off_a + inner];
            for (d, &x) in dst.iter_mut().zip(ra) {
                *d = bin_apply(op, x, bv);
            }
        } else if ia == 0 && ib == 1 {
            let av = a[off_a];
            let rb = &b[off_b..off_b + inner];
            for (d, &y) in dst.iter_mut().zip(rb) {
                *d = bin_apply(op, av, y);
            }
        } else {
            for (j, d) in dst.iter_mut().enumerate() {
                *d = bin_apply(op, a[off_a + j * ia], b[off_b + j * ib]);
            }
        }
        o += inner;
        // odometer over the outer axes
        for ax in (0..rank - 1).rev() {
            idx[ax] += 1;
            off_a += sa[ax];
            off_b += sb[ax];
            if idx[ax] < shape[ax] {
                break;
            }
            off_a -= sa[ax] * shape[ax];
            off_b -= sb[ax] * shape[ax];
            idx[ax] = 0;
        }
    }
}

/// Row-major strides of a contiguous tensor (public for stride comparisons).
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

// ---------------------------------------------------------------------------
// Layout ops
// ---------------------------------------------------------------------------

/// Generic N-d transpose (permute) of a contiguous tensor.
pub fn transpose(x: &[f32], shape: &[usize], perm: &[usize], out: &mut [f32]) {
    let rank = shape.len();
    let in_strides = contiguous_strides(shape);
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let numel: usize = shape.iter().product();
    debug_assert_eq!(out.len(), numel);
    // walk output in order; compute source offset via permuted strides
    let src_stride: Vec<usize> = perm.iter().map(|&p| in_strides[p]).collect();
    let mut idx = vec![0usize; rank];
    let mut src = 0usize;
    let inner = *out_shape.last().unwrap_or(&1);
    let step = *src_stride.last().unwrap_or(&1);
    let mut o = 0;
    while o < numel {
        if step == 1 {
            out[o..o + inner].copy_from_slice(&x[src..src + inner]);
        } else {
            for j in 0..inner {
                out[o + j] = x[src + j * step];
            }
        }
        o += inner;
        for ax in (0..rank - 1).rev() {
            idx[ax] += 1;
            src += src_stride[ax];
            if idx[ax] < out_shape[ax] {
                break;
            }
            src -= src_stride[ax] * out_shape[ax];
            idx[ax] = 0;
        }
    }
}

/// Strided slice copy (steps >= 1).
pub fn slice_copy(x: &[f32], shape: &[usize], starts: &[usize], ends: &[usize], steps: &[usize], out: &mut [f32]) {
    let rank = shape.len();
    let in_strides = contiguous_strides(shape);
    let out_shape: Vec<usize> = (0..rank).map(|a| (ends[a] - starts[a]).div_ceil(steps[a])).collect();
    let numel: usize = out_shape.iter().product();
    debug_assert_eq!(out.len(), numel);
    if numel == 0 {
        return;
    }
    let base: usize = (0..rank).map(|a| starts[a] * in_strides[a]).sum();
    let inner = out_shape[rank - 1];
    let istep = steps[rank - 1] * in_strides[rank - 1];
    let mut idx = vec![0usize; rank - 1];
    let mut src = base;
    let mut o = 0;
    while o < numel {
        if istep == 1 {
            out[o..o + inner].copy_from_slice(&x[src..src + inner]);
        } else {
            for j in 0..inner {
                out[o + j] = x[src + j * istep];
            }
        }
        o += inner;
        for ax in (0..rank - 1).rev() {
            idx[ax] += 1;
            src += steps[ax] * in_strides[ax];
            if idx[ax] < out_shape[ax] {
                break;
            }
            src -= steps[ax] * in_strides[ax] * out_shape[ax];
            idx[ax] = 0;
        }
    }
}

/// Concat along `axis`: inputs share every other dim.
pub fn concat(inputs: &[(&[f32], &[usize])], axis: usize, out: &mut [f32]) {
    let (first_shape,) = (inputs[0].1,);
    let outer: usize = first_shape[..axis].iter().product();
    let inner: usize = first_shape[axis + 1..].iter().product();
    let total_axis: usize = inputs.iter().map(|(_, s)| s[axis]).sum();
    debug_assert_eq!(out.len(), outer * total_axis * inner);
    let mut dst_off = 0usize;
    for (data, shape) in inputs {
        let rows = shape[axis] * inner;
        for oi in 0..outer {
            out[oi * total_axis * inner + dst_off..oi * total_axis * inner + dst_off + rows]
                .copy_from_slice(&data[oi * rows..(oi + 1) * rows]);
        }
        dst_off += rows;
    }
}

/// Split along `axis` into pre-sized output slices.
pub fn split(x: &[f32], shape: &[usize], axis: usize, parts: &[usize], outs: &mut [&mut [f32]]) {
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let total = shape[axis];
    let mut src_off = 0usize;
    for (p, out) in parts.iter().zip(outs.iter_mut()) {
        let rows = p * inner;
        for oi in 0..outer {
            out[oi * rows..(oi + 1) * rows]
                .copy_from_slice(&x[oi * total * inner + src_off..oi * total * inner + src_off + rows]);
        }
        src_off += rows;
    }
}

// ---------------------------------------------------------------------------
// Math tails: softmax / reduce / layernorm / matmul
// ---------------------------------------------------------------------------

/// Numerically-stable softmax along `axis` of a contiguous tensor.
pub fn softmax_axis(x: &[f32], shape: &[usize], axis: usize, out: &mut [f32]) {
    let strides = contiguous_strides(shape);
    let d = shape[axis];
    let stride = strides[axis];
    let numel: usize = shape.iter().product();
    let slices = numel / d;
    for s in 0..slices {
        // origin of the s-th 1-D slice along `axis`
        let mut rem = s;
        let mut origin = 0usize;
        for (ax, (&dim, &st)) in shape.iter().zip(&strides).enumerate() {
            if ax == axis {
                continue;
            }
            let extent = dim;
            let coord = rem % extent;
            rem /= extent;
            origin += coord * st;
        }
        // PyTorch `cunn_SpatialSoftMaxForward` order: max seeded with -FLT_MAX
        // (`a < b ? b : a`), serial Σexp(x-max), then exp(x-max)/sum.
        let mut m = f32::MIN;
        for i in 0..d {
            let v = x[origin + i * stride];
            if m < v {
                m = v;
            }
        }
        let mut sum = 0.0f32;
        for i in 0..d {
            let e = (x[origin + i * stride] - m).exp();
            out[origin + i * stride] = e;
            sum += e;
        }
        for i in 0..d {
            out[origin + i * stride] /= sum;
        }
    }
}

/// Mean over a sorted set of axes (keepdims handled by the caller's shape).
pub fn reduce_mean(x: &[f32], shape: &[usize], axes: &[usize], out: &mut [f32]) {
    let rank = shape.len();
    let reduce: Vec<bool> = (0..rank).map(|a| axes.contains(&a)).collect();
    let out_dims: Vec<usize> = (0..rank).map(|a| if reduce[a] { 1 } else { shape[a] }).collect();
    let out_strides = contiguous_strides(&out_dims);
    let n_reduced: usize = axes.iter().map(|&a| shape[a]).product();
    out.fill(0.0);
    let strides = contiguous_strides(shape);
    let numel: usize = shape.iter().product();
    // Single pass: accumulate into the collapsed index.
    let mut idx = vec![0usize; rank];
    for i in 0..numel {
        let mut oi = 0usize;
        for a in 0..rank {
            if !reduce[a] {
                oi += idx[a] * out_strides[a];
            }
        }
        out[oi] += x[i];
        for ax in (0..rank).rev() {
            idx[ax] += 1;
            if idx[ax] < shape[ax] {
                break;
            }
            idx[ax] = 0;
        }
    }
    let inv = 1.0 / n_reduced as f32;
    for v in out.iter_mut() {
        *v *= inv;
    }
    let _ = strides;
}

/// Last-axis LayerNorm, ONNX semantics: population variance, eps inside sqrt.
pub fn layernorm_lastaxis(x: &[f32], d: usize, w: &[f32], b: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    for (xr, or) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean = xr.iter().sum::<f32>() / d as f32;
        let var = xr.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        match b {
            Some(b) => {
                for i in 0..d {
                    or[i] = (xr[i] - mean) * inv * w[i] + b[i];
                }
            }
            None => {
                for i in 0..d {
                    or[i] = (xr[i] - mean) * inv * w[i];
                }
            }
        }
    }
}

/// Batched matmul: a `[batch, m, k]` · b `[batch(bcast), k, n]` → `[batch, m, n]`.
/// `b_batch` of 1 broadcasts the single right-hand matrix.
pub fn matmul_batched(a: &[f32], b: &[f32], batch: usize, b_batch: usize, m: usize, k: usize, n: usize, threads: usize, out: &mut [f32]) {
    debug_assert!(b_batch == batch || b_batch == 1);
    let jobs = batch * m;
    let out_base = out.as_mut_ptr() as usize;
    let next = AtomicUsize::new(0);
    let body = |_id: usize, _nt: usize| loop {
        let job = next.fetch_add(1, Ordering::Relaxed);
        if job >= jobs {
            break;
        }
        let (bi, mi) = (job / m, job % m);
        let arow = &a[(bi * m + mi) * k..(bi * m + mi + 1) * k];
        let bmat = &b[if b_batch == 1 { 0 } else { bi * k * n }..];
        // SAFETY: one disjoint output row per job.
        let orow = unsafe { std::slice::from_raw_parts_mut((out_base as *mut f32).add((bi * m + mi) * n), n) };
        orow.fill(0.0);
        for (kk, &av) in arow.iter().enumerate() {
            if av == 0.0 {
                continue;
            }
            let brow = &bmat[kk * n..(kk + 1) * n];
            for (o, &bv) in orow.iter_mut().zip(brow) {
                *o += av * bv;
            }
        }
    };
    parallel(threads.min(jobs).max(1), &body);
}

/// Gemm y = x·wᵀ (+bias) with `w` `[n, k]` row-major (transB=1 layout), `act` fused.
pub fn gemm_nt(x: &[f32], w: &[f32], bias: Option<&[f32]>, m: usize, k: usize, n: usize, act: Act, threads: usize, out: &mut [f32]) {
    let jobs = m;
    let out_base = out.as_mut_ptr() as usize;
    let next = AtomicUsize::new(0);
    let body = |_id: usize, _nt: usize| loop {
        let mi = next.fetch_add(1, Ordering::Relaxed);
        if mi >= jobs {
            break;
        }
        let xr = &x[mi * k..(mi + 1) * k];
        // SAFETY: disjoint rows.
        let orow = unsafe { std::slice::from_raw_parts_mut((out_base as *mut f32).add(mi * n), n) };
        for (j, o) in orow.iter_mut().enumerate() {
            let wr = &w[j * k..(j + 1) * k];
            let mut acc = bias.map_or(0.0, |b| b[j]);
            acc += dot(xr, wr);
            *o = apply_act(act, acc);
        }
    };
    parallel(threads.min(jobs).max(1), &body);
}

/// Unrolled dot product (4 independent accumulators — autovectorizes).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..chunks {
        let a8 = &a[i * 8..i * 8 + 8];
        let b8 = &b[i * 8..i * 8 + 8];
        s0 += a8[0] * b8[0] + a8[4] * b8[4];
        s1 += a8[1] * b8[1] + a8[5] * b8[5];
        s2 += a8[2] * b8[2] + a8[6] * b8[6];
        s3 += a8[3] * b8[3] + a8[7] * b8[7];
    }
    let mut tail = 0.0f32;
    for i in chunks * 8..n {
        tail += a[i] * b[i];
    }
    s0 + s1 + s2 + s3 + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive 7-loop conv, the reference the tiled path is checked against.
    fn conv_naive(x: &[f32], w: &[f32], bias: Option<&[f32]>, s: &ConvShape) -> Vec<f32> {
        let (oh, ow) = s.out_hw();
        let cin_g = s.cin / s.group;
        let cout_g = s.cout / s.group;
        let mut out = vec![0.0f32; s.n * s.cout * oh * ow];
        for bi in 0..s.n {
            for co in 0..s.cout {
                let g = co / cout_g;
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = bias.map_or(0.0, |b| b[co]);
                        for ci in 0..cin_g {
                            for ky in 0..s.kh {
                                for kx in 0..s.kw {
                                    let iy = oy as isize * s.stride[0] as isize + (ky * s.dilation[0]) as isize
                                        - s.pads[0] as isize;
                                    let ix = ox as isize * s.stride[1] as isize + (kx * s.dilation[1]) as isize
                                        - s.pads[1] as isize;
                                    if iy < 0 || iy >= s.h as isize || ix < 0 || ix >= s.w as isize {
                                        continue;
                                    }
                                    acc += x[((bi * s.cin + g * cin_g + ci) * s.h + iy as usize) * s.w + ix as usize]
                                        * w[((co * cin_g + ci) * s.kh + ky) * s.kw + kx];
                                }
                            }
                        }
                        out[((bi * s.cout + co) * oh + oy) * ow + ox] = acc;
                    }
                }
            }
        }
        out
    }

    fn tvec(n: usize, seed: u32) -> Vec<f32> {
        // deterministic ragged-value generator (integers → exact float math)
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 16) % 17) as f32 - 8.0
            })
            .collect()
    }

    #[test]
    fn conv_matches_naive_ragged_shapes() {
        // ragged: stem 3->13 channels, k3 s2 p1; depthwise; dilation; group
        let cases = [
            ConvShape { n: 2, cin: 3, h: 13, w: 11, cout: 13, kh: 3, kw: 3, group: 1, stride: [2, 2], pads: [1, 1, 1, 1], dilation: [1, 1] },
            ConvShape { n: 1, cin: 8, h: 9, w: 9, cout: 8, kh: 3, kw: 3, group: 8, stride: [1, 1], pads: [1, 1, 1, 1], dilation: [1, 1] },
            ConvShape { n: 1, cin: 4, h: 12, w: 12, cout: 6, kh: 1, kw: 1, group: 2, stride: [1, 1], pads: [0; 4], dilation: [1, 1] },
            ConvShape { n: 1, cin: 2, h: 15, w: 15, cout: 4, kh: 3, kw: 3, group: 1, stride: [1, 1], pads: [2, 2, 2, 2], dilation: [2, 2] },
            ConvShape { n: 1, cin: 3, h: 8, w: 8, cout: 5, kh: 5, kw: 5, group: 1, stride: [1, 1], pads: [2, 2, 2, 2], dilation: [1, 1] },
        ];
        for s in &cases {
            let x = tvec(s.n * s.cin * s.h * s.w, 7);
            let cin_g = s.cin / s.group;
            let w = tvec(s.cout * cin_g * s.kh * s.kw, 31);
            let bias = tvec(s.cout, 5);
            let want = conv_naive(&x, &w, Some(&bias), s);
            let (oh, ow) = s.out_hw();
            let mut got = vec![0.0f32; s.n * s.cout * oh * ow];
            conv2d(&x, &w, Some(&bias), s, Act::None, 4, &mut got);
            assert_eq!(got, want, "conv mismatch for {s:?}"); // integer inputs → exact
        }
    }

    #[test]
    fn conv_fused_act() {
        let s = ConvShape { n: 1, cin: 2, h: 6, w: 6, cout: 3, kh: 3, kw: 3, group: 1, stride: [1, 1], pads: [1, 1, 1, 1], dilation: [1, 1] };
        let x = tvec(s.n * s.cin * s.h * s.w, 3);
        let w = tvec(s.cout * s.cin * s.kh * s.kw, 9);
        let want: Vec<f32> = conv_naive(&x, &w, None, &s).iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        let mut got = vec![0.0f32; want.len()];
        conv2d(&x, &w, None, &s, Act::Silu, 2, &mut got);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-6 * w.abs().max(1.0), "{g} vs {w}");
        }
    }

    #[test]
    fn ceil_mode_extents() {
        // torch: MaxPool2d(k=3, s=2, ceil_mode=True) on 5 -> ceil((5-3)/2)+1 = 2
        assert_eq!(pool_out(5, 3, 0, 0, 2, true), 2);
        assert_eq!(pool_out(5, 3, 0, 0, 2, false), 2);
        // on 6: floor -> 2, ceil -> 3 (last window starts at 4 < 6, kept)
        assert_eq!(pool_out(6, 3, 0, 0, 2, false), 2);
        assert_eq!(pool_out(6, 3, 0, 0, 2, true), 3);
        // window starting entirely in right padding is dropped:
        // input 4, k2 s2 pad 0/2 ceil: span 4, ceil(4/2)+1 = 3 but start 4 >= 4 -> 2
        assert_eq!(pool_out(4, 2, 0, 2, 2, true), 2);
    }

    #[test]
    fn pools_match_reference() {
        let s = PoolShape { n: 1, c: 2, h: 7, w: 7, kernel: [5, 5], stride: [1, 1], pads: [2, 2, 2, 2], ceil: false };
        let x = tvec(2 * 49, 11);
        let (oh, ow) = s.out_hw();
        assert_eq!((oh, ow), (7, 7)); // SPPF shape-preserving
        let mut mx = vec![0.0f32; 2 * oh * ow];
        maxpool2d(&x, &s, 1, &mut mx);
        // spot-check centre and corner against direct computation
        for c in 0..2 {
            let plane = &x[c * 49..(c + 1) * 49];
            let mut want = f32::NEG_INFINITY;
            for y in 0..3 {
                for xx in 0..3 {
                    want = want.max(plane[y * 7 + xx]);
                }
            }
            assert_eq!(mx[c * 49], want, "corner max, channel {c}");
        }
        let mut av = vec![0.0f32; 2 * oh * ow];
        avgpool2d(&x, &s, false, 1, &mut av);
        let plane = &x[0..49];
        let mut sum = 0.0;
        for y in 0..3 {
            for xx in 0..3 {
                sum += plane[y * 7 + xx];
            }
        }
        assert!((av[0] - sum / 9.0).abs() < 1e-6);
    }

    #[test]
    fn resize_nearest_2x() {
        let x = [1.0, 2.0, 3.0, 4.0]; // 1 plane 2x2
        let mut out = vec![0.0; 16];
        resize_nearest(&x, 1, 2, 2, 2, 2, &mut out);
        assert_eq!(out, vec![1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.]);
    }

    #[test]
    fn broadcast_binary() {
        // [2,3] * [3] (per-channel)
        let a = [1., 2., 3., 4., 5., 6.];
        let b = [10., 100., 1000.];
        let shape = [2usize, 3];
        let sa = contiguous_strides(&shape);
        let sb = [0usize, 1];
        let mut out = vec![0.0; 6];
        binary_bcast(BinOp::Mul, &a, &sa, &b, &sb, &shape, &mut out);
        assert_eq!(out, vec![10., 200., 3000., 40., 500., 6000.]);
    }

    #[test]
    fn transpose_permute() {
        // [2,3] -> [3,2]
        let x = [1., 2., 3., 4., 5., 6.];
        let mut out = vec![0.0; 6];
        transpose(&x, &[2, 3], &[1, 0], &mut out);
        assert_eq!(out, vec![1., 4., 2., 5., 3., 6.]);
        // [1,84,4] -> [1,4,84] round-trip
        let x = tvec(84 * 4, 3);
        let mut once = vec![0.0; x.len()];
        transpose(&x, &[1, 84, 4], &[0, 2, 1], &mut once);
        let mut back = vec![0.0; x.len()];
        transpose(&once, &[1, 4, 84], &[0, 2, 1], &mut back);
        assert_eq!(back, x);
    }

    #[test]
    fn slice_strided() {
        let x: Vec<f32> = (0..24).map(|i| i as f32).collect(); // [2,3,4]
        let mut out = vec![0.0; 2 * 2 * 2];
        // [:, 1:3, 0:4:2]
        slice_copy(&x, &[2, 3, 4], &[0, 1, 0], &[2, 3, 4], &[1, 1, 2], &mut out);
        assert_eq!(out, vec![4., 6., 8., 10., 16., 18., 20., 22.]);
    }

    #[test]
    fn concat_split_roundtrip() {
        let a: Vec<f32> = (0..12).map(|i| i as f32).collect(); // [2,2,3]
        let b: Vec<f32> = (100..112).map(|i| i as f32).collect();
        let mut cat = vec![0.0; 24];
        concat(&[(&a, &[2, 2, 3][..]), (&b, &[2, 2, 3][..])], 1, &mut cat);
        let mut oa = vec![0.0; 12];
        let mut ob = vec![0.0; 12];
        {
            let mut outs: Vec<&mut [f32]> = vec![&mut oa, &mut ob];
            split(&cat, &[2, 4, 3], 1, &[2, 2], &mut outs);
        }
        assert_eq!(oa, a);
        assert_eq!(ob, b);
    }

    #[test]
    fn softmax_any_axis() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let mut out = vec![0.0; 6];
        softmax_axis(&x, &[2, 3], 1, &mut out);
        for r in 0..2 {
            let s: f32 = out[r * 3..(r + 1) * 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-6);
        }
        // axis 0: columns sum to 1
        softmax_axis(&x, &[2, 3], 0, &mut out);
        for c in 0..3 {
            assert!((out[c] + out[3 + c] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn layernorm_matches_onnx_semantics() {
        let d = 8;
        let x = tvec(2 * d, 21);
        let w = vec![1.0f32; d];
        let mut out = vec![0.0; x.len()];
        layernorm_lastaxis(&x, d, &w, None, 1e-5, &mut out);
        for row in out.chunks_exact(d) {
            let mean: f32 = row.iter().sum::<f32>() / d as f32;
            assert!(mean.abs() < 1e-5, "normalized mean {mean}");
        }
        // agrees with cpu_math::layernorm (same population-variance semantics)
        let b = vec![0.0f32; d];
        let want = crate::cpu_math::layernorm(&x[..d], &w, &b, 1e-5);
        for (g, w) in out[..d].iter().zip(&want) {
            assert!((g - w).abs() < 1e-6);
        }
    }

    #[test]
    fn matmul_and_gemm() {
        // [1,2,3]·[1,3,2]
        let a = [1., 2., 3., 4., 5., 6.];
        let b = [7., 8., 9., 10., 11., 12.];
        let mut out = vec![0.0; 4];
        matmul_batched(&a, &b, 1, 1, 2, 3, 2, 1, &mut out);
        assert_eq!(out, vec![58., 64., 139., 154.]);
        // gemm_nt: y = x·wᵀ, w [n,k]
        let x = [1., 2., 3., 4., 5., 6.]; // [2,3]
        let w = [1., 0., 0., 0., 1., 0.]; // [2,3] picks x0, x1
        let mut y = vec![0.0; 4];
        gemm_nt(&x, &w, Some(&[10.0, 20.0]), 2, 3, 2, Act::None, 1, &mut y);
        assert_eq!(y, vec![11., 22., 14., 25.]);
    }

    #[test]
    fn erf_reference_values() {
        // reference values from scipy.special.erf
        for (x, want) in [(0.0f32, 0.0f32), (0.5, 0.5204999), (1.0, 0.8427008), (2.0, 0.9953223), (-1.0, -0.8427008)] {
            assert!((erf(x) - want).abs() < 2e-7, "erf({x}) = {} want {want}", erf(x));
        }
        assert!((gelu_erf(1.0) - 0.8413447).abs() < 1e-6);
    }
}
