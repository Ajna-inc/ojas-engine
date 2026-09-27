//! CPU reference for the DETR-family ops: the
//! ground truth the CUDA and Vulkan kernels are checked against.
//!
//! - [`grid_sample`] ports PyTorch `grid_sampler_2d` (aten/src/ATen/native/
//!   GridSampler.cpp / cuda/GridSampler.cu), bilinear, zeros padding,
//!   `align_corners = false`: the same unnormalise, corner weights and
//!   bounds checks, accumulated in the same order.
//! - [`topk`] / [`topk_gather`] rank by counting: an element's rank is the
//!   number of elements greater than it, or equal with a lower index — exact,
//!   deterministic, and ONNX TopK's tie rule (the lower index first).

use crate::ir::ReduceOp;

/// PyTorch `grid_sampler_compute_source_index` for align_corners = false.
#[inline]
fn unnormalize(coord: f32, size: usize) -> f32 {
    ((coord + 1.0) * size as f32 - 1.0) / 2.0
}

/// x (N,C,H,W), grid (N,Ho,Wo,2) → out (N,C,Ho,Wo).
pub fn grid_sample(x: &[f32], xs: &[usize], grid: &[f32], gs: &[usize], out: &mut [f32]) {
    let (n, c, h, w) = (xs[0], xs[1], xs[2], xs[3]);
    let (ho, wo) = (gs[1], gs[2]);
    for b in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                let g = ((b * ho + oy) * wo + ox) * 2;
                let ix = unnormalize(grid[g], w);
                let iy = unnormalize(grid[g + 1], h);
                let (ix_nw, iy_nw) = (ix.floor(), iy.floor());
                let (ix_ne, iy_ne) = (ix_nw + 1.0, iy_nw);
                let (ix_sw, iy_sw) = (ix_nw, iy_nw + 1.0);
                let (ix_se, iy_se) = (ix_nw + 1.0, iy_nw + 1.0);
                let nw = (ix_se - ix) * (iy_se - iy);
                let ne = (ix - ix_sw) * (iy_sw - iy);
                let sw = (ix_ne - ix) * (iy - iy_ne);
                let se = (ix - ix_nw) * (iy - iy_nw);
                let inside = |xx: f32, yy: f32| xx >= 0.0 && yy >= 0.0 && (xx as i64) < w as i64 && (yy as i64) < h as i64;
                for ch in 0..c {
                    let plane = &x[(b * c + ch) * h * w..][..h * w];
                    let at = |xx: f32, yy: f32| plane[yy as usize * w + xx as usize];
                    let mut acc = 0.0f32;
                    if inside(ix_nw, iy_nw) {
                        acc += at(ix_nw, iy_nw) * nw;
                    }
                    if inside(ix_ne, iy_ne) {
                        acc += at(ix_ne, iy_ne) * ne;
                    }
                    if inside(ix_sw, iy_sw) {
                        acc += at(ix_sw, iy_sw) * sw;
                    }
                    if inside(ix_se, iy_se) {
                        acc += at(ix_se, iy_se) * se;
                    }
                    out[((b * c + ch) * ho + oy) * wo + ox] = acc;
                }
            }
        }
    }
}

/// Reduce `x` over `axes` (sorted, unique); the output is laid out without
/// the reduced axes (keepdims only changes the shape, not the order).
pub fn reduce(x: &[f32], xs: &[usize], axes: &[usize], kind: ReduceOp, out: &mut [f32]) {
    let init = match kind {
        ReduceOp::Max => f32::NEG_INFINITY,
        _ => 0.0,
    };
    out.iter_mut().for_each(|v| *v = init);
    let strides = crate::ir::strides_of(xs);
    let kept: Vec<usize> = (0..xs.len()).filter(|a| !axes.contains(a)).collect();
    let kept_shape: Vec<usize> = kept.iter().map(|&a| xs[a]).collect();
    let kept_strides = crate::ir::strides_of(&kept_shape);
    for (i, &v) in x.iter().enumerate() {
        let mut o = 0;
        for (k, &a) in kept.iter().enumerate() {
            o += (i / strides[a] % xs[a]) * kept_strides[k];
        }
        match kind {
            ReduceOp::Max => out[o] = out[o].max(v),
            _ => out[o] += v,
        }
    }
    if kind == ReduceOp::Mean {
        let n: usize = axes.iter().map(|&a| xs[a]).product();
        out.iter_mut().for_each(|v| *v /= n as f32);
    }
}

/// Indices of the k largest in `row`, largest first (ties: the lower index first).
fn top_indices(row: &[f32], k: usize) -> Vec<usize> {
    let mut idx = vec![0usize; k];
    for (i, &v) in row.iter().enumerate() {
        let rank = row.iter().enumerate().filter(|&(j, &u)| u > v || (u == v && j < i)).count();
        if rank < k {
            idx[rank] = i;
        }
    }
    idx
}

/// x (…, L) → out (…, k): the k largest of each last-axis row, largest first.
pub fn topk(x: &[f32], len: usize, k: usize, out: &mut [f32]) {
    for (r, row) in x.chunks(len).enumerate() {
        for (j, i) in top_indices(row, k).into_iter().enumerate() {
            out[r * k + j] = row[i];
        }
    }
}

/// scores (B,N), data (B,N,C) → out (B,k,C): data's rows at the k largest scores.
pub fn topk_gather(scores: &[f32], data: &[f32], b: usize, n: usize, c: usize, k: usize, out: &mut [f32]) {
    for bb in 0..b {
        for (j, i) in top_indices(&scores[bb * n..][..n], k).into_iter().enumerate() {
            out[(bb * k + j) * c..][..c].copy_from_slice(&data[(bb * n + i) * c..][..c]);
        }
    }
}

/// Constant Pad of NCHW x on its spatial axes: pads = [top, left, bottom, right].
pub fn pad(x: &[f32], xs: &[usize], pads: [usize; 4], value: f32, out: &mut [f32]) {
    let (h, w) = (xs[2], xs[3]);
    let (ho, wo) = (h + pads[0] + pads[2], w + pads[1] + pads[3]);
    for (p, plane) in x.chunks(h * w).enumerate() {
        let o = &mut out[p * ho * wo..][..ho * wo];
        o.iter_mut().for_each(|v| *v = value);
        for r in 0..h {
            o[(r + pads[0]) * wo + pads[1]..][..w].copy_from_slice(&plane[r * w..][..w]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_sample_matches_hand_values() {
        // 1x1x2x2 [[1,2],[3,4]]; grid points: the centre of pixel (0,0), the
        // image centre, and one outside
        let x = [1.0, 2.0, 3.0, 4.0];
        let grid = [-0.5, -0.5, 0.0, 0.0, 2.0, 2.0];
        let mut out = [0.0; 3];
        grid_sample(&x, &[1, 1, 2, 2], &grid, &[1, 1, 3, 2], &mut out);
        assert_eq!(out, [1.0, 2.5, 0.0]);
        // half a pixel outside the right edge: half of the edge value (zeros padding)
        let mut o = [0.0];
        grid_sample(&x, &[1, 1, 2, 2], &[1.0, -0.5], &[1, 1, 1, 2], &mut o);
        assert_eq!(o, [1.0]);
    }

    #[test]
    fn topk_orders_and_breaks_ties_by_index() {
        let row = [0.1, 0.9, 0.5, 0.9, 0.2];
        let mut out = [0.0; 3];
        topk(&row, 5, 3, &mut out);
        assert_eq!(out, [0.9, 0.9, 0.5]);
        assert_eq!(top_indices(&row, 3), vec![1, 3, 2]);
        let data: Vec<f32> = (0..10).map(|v| v as f32).collect(); // 5 rows of 2
        let mut g = [0.0; 6];
        topk_gather(&row, &data, 1, 5, 2, 3, &mut g);
        assert_eq!(g, [2.0, 3.0, 6.0, 7.0, 4.0, 5.0]);
    }

    #[test]
    fn reductions_over_the_last_and_middle_axes() {
        let x: Vec<f32> = (0..12).map(|v| v as f32).collect(); // [2,3,2]
        let mut s = [0.0; 6];
        reduce(&x, &[2, 3, 2], &[2], ReduceOp::Sum, &mut s);
        assert_eq!(s, [1.0, 5.0, 9.0, 13.0, 17.0, 21.0]);
        let mut m = [0.0; 4];
        reduce(&x, &[2, 3, 2], &[1], ReduceOp::Max, &mut m);
        assert_eq!(m, [4.0, 5.0, 10.0, 11.0]);
        let mut a = [0.0; 2];
        reduce(&x, &[2, 3, 2], &[1, 2], ReduceOp::Mean, &mut a);
        assert_eq!(a, [2.5, 8.5]);
    }
}
