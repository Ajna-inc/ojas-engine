//! DETR training targets and losses, ported from RT-DETRv2 (`rtdetrv2_criterion.py`,
//! `matcher.py`, `denoising.py`, `box_ops.py`): the Hungarian matcher (focal
//! class cost + L1 + GIoU), VFL / L1 / GIoU losses on the tape, and the
//! contrastive denoising queries. Matching and target construction run on the
//! host (they need a few hundred values per image); the losses are tape ops.

use anyhow::Result;

use crate::backend::Backend;
use crate::models::rtdetr::Outputs;
use crate::tape::{Tape, Var};
use crate::Unary;

/// Ground truth for one image: class ids and boxes (cx, cy, w, h), normalised to 0..1.
#[derive(Clone, Debug, Default)]
pub struct Target {
    pub labels: Vec<usize>,
    pub boxes: Vec<[f32; 4]>,
}

pub fn xyxy(b: [f32; 4]) -> [f32; 4] {
    [b[0] - 0.5 * b[2], b[1] - 0.5 * b[3], b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]]
}

fn area(b: [f32; 4]) -> f32 {
    (b[2] - b[0]) * (b[3] - b[1])
}

/// (IoU, GIoU) of two xyxy boxes (torchvision box_iou / generalized_box_iou).
pub fn iou_giou(a: [f32; 4], b: [f32; 4]) -> (f32, f32) {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let union = area(a) + area(b) - inter;
    let iou = inter / union;
    let cw = (a[2].max(b[2]) - a[0].min(b[0])).max(0.0);
    let ch = (a[3].max(b[3]) - a[1].min(b[1])).max(0.0);
    let c = cw * ch;
    (iou, iou - (c - union) / c)
}

/// Minimum-cost assignment on a rows × cols matrix (row-major): the (row, col)
/// pairs, min(rows, cols) of them, sorted by row. Shortest augmenting paths
/// with potentials (the Jonker–Volgenant / Kuhn–Munkres scheme SciPy's
/// linear_sum_assignment uses), in f64.
pub fn hungarian(cost: &[f32], rows: usize, cols: usize) -> Vec<(usize, usize)> {
    if rows == 0 || cols == 0 {
        return vec![];
    }
    if rows > cols {
        let mut t = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = cost[r * cols + c];
            }
        }
        let mut p: Vec<(usize, usize)> = hungarian(&t, cols, rows).into_iter().map(|(c, r)| (r, c)).collect();
        p.sort();
        return p;
    }
    // n rows ≤ m cols, 1-indexed (e-maxx formulation)
    let (n, m) = (rows, cols);
    let a = |i: usize, j: usize| cost[(i - 1) * cols + (j - 1)] as f64;
    let (mut u, mut v) = (vec![0.0f64; n + 1], vec![0.0f64; m + 1]);
    let (mut p, mut way) = (vec![0usize; m + 1], vec![0usize; m + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let (i0, mut delta, mut j1) = (p[j0], f64::INFINITY, 0);
            for j in 1..=m {
                if !used[j] {
                    let cur = a(i0, j) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out: Vec<(usize, usize)> = (1..=m).filter(|&j| p[j] != 0).map(|j| (p[j] - 1, j - 1)).collect();
    out.sort();
    out
}

pub struct Matcher {
    pub cost_class: f32,
    pub cost_bbox: f32,
    pub cost_giou: f32,
    pub alpha: f32,
    pub gamma: f32,
}

impl Default for Matcher {
    fn default() -> Self {
        Matcher { cost_class: 2.0, cost_bbox: 5.0, cost_giou: 2.0, alpha: 0.25, gamma: 2.0 }
    }
}

impl Matcher {
    /// One image: logits [Q·C], boxes [Q·4] → (query, target) pairs sorted by query.
    pub fn assign(&self, logits: &[f32], boxes: &[f32], c: usize, tgt: &Target) -> Vec<(usize, usize)> {
        let (q, t) = (boxes.len() / 4, tgt.labels.len());
        if t == 0 {
            return vec![];
        }
        let mut cost = vec![0.0f32; q * t];
        for i in 0..q {
            let b = [boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]];
            for (j, (&lab, &tb)) in tgt.labels.iter().zip(&tgt.boxes).enumerate() {
                let p = 1.0 / (1.0 + (-logits[i * c + lab]).exp());
                let neg = (1.0 - self.alpha) * p.powf(self.gamma) * -(1.0 - p + 1e-8).ln();
                let pos = self.alpha * (1.0 - p).powf(self.gamma) * -(p + 1e-8).ln();
                let l1: f32 = (0..4).map(|k| (b[k] - tb[k]).abs()).sum();
                let (_, g) = iou_giou(xyxy(b), xyxy(tb));
                cost[i * t + j] = self.cost_bbox * l1 + self.cost_class * (pos - neg) + self.cost_giou * -g;
            }
        }
        hungarian(&cost, q, t)
    }
}

/// Contrastive denoising queries for one batch (training only).
pub struct DnGroup {
    /// class id per denoising slot [B·D] (num_classes = padding)
    pub classes: Vec<usize>,
    /// 1 for real slots, 0 for padding [B·D]
    pub pad: Vec<f32>,
    /// noised boxes, inverse-sigmoided [B·D·4]
    pub boxes_unact: Vec<f32>,
    /// additive attention mask [(D+Q)·(D+Q)]: 0 or −∞
    pub mask: Vec<f32>,
    pub num_dn: usize,
    pub num_group: usize,
    /// per image: (dn query, target) pairs
    pub matches: Vec<Vec<(usize, usize)>>,
}

/// Small deterministic generator for the denoising noise (PCG32).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
    }
    pub fn next_u32(&mut self) -> u32 {
        let old = self.0;
        self.0 = old.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let xs = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xs.rotate_right(rot)
    }
    /// uniform in [0, 1)
    pub fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

fn inv_sigmoid(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    (x.max(1e-5) / (1.0 - x).max(1e-5)).ln()
}

/// get_contrastive_denoising_training_group.
pub fn denoising(targets: &[Target], num_classes: usize, num_queries: usize, num_denoising: usize, label_noise: f32, box_noise: f32, rng: &mut Rng) -> Option<DnGroup> {
    if num_denoising == 0 {
        return None;
    }
    let bs = targets.len();
    let gts: Vec<usize> = targets.iter().map(|t| t.labels.len()).collect();
    let mut max_gt = *gts.iter().max().unwrap_or(&0);
    let empty = max_gt == 0;
    let num_group = if empty {
        max_gt = 1;
        1
    } else {
        (num_denoising / max_gt).max(1)
    };
    let d = max_gt * 2 * num_group;
    let (mut classes, mut pad, mut boxes_unact) = (vec![num_classes; bs * d], vec![0.0f32; bs * d], vec![0.0f32; bs * d * 4]);
    let mut matches = vec![];
    for (b, t) in targets.iter().enumerate() {
        let mut m = vec![];
        for slot in 0..d {
            let (j, negative) = (slot % max_gt, (slot / max_gt) % 2 == 1);
            let real = j < gts[b];
            let (mut cls, mut bx) = (num_classes, if empty { [0.5, 0.5, 0.1, 0.1] } else { [0.0; 4] });
            if real {
                cls = t.labels[j];
                bx = t.boxes[j];
                pad[b * d + slot] = 1.0;
                if !negative {
                    m.push((slot, j));
                }
                if label_noise > 0.0 && rng.uniform() < label_noise * 0.5 {
                    cls = (rng.next_u32() as usize) % num_classes;
                }
            }
            if box_noise > 0.0 {
                let mut k = xyxy(bx);
                let diff = [bx[2] * 0.5 * box_noise, bx[3] * 0.5 * box_noise];
                for (c, v) in k.iter_mut().enumerate() {
                    let sign = if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 };
                    let part = rng.uniform() + if negative { 1.0 } else { 0.0 };
                    *v = (*v + sign * part * diff[c % 2]).clamp(0.0, 1.0);
                }
                bx = [(k[0] + k[2]) * 0.5, (k[1] + k[3]) * 0.5, k[2] - k[0], k[3] - k[1]];
            }
            classes[b * d + slot] = cls;
            for c in 0..4 {
                boxes_unact[(b * d + slot) * 4 + c] = inv_sigmoid(bx[c]);
            }
        }
        matches.push(m);
    }
    // queries don't see denoising slots; each group sees only itself
    let n = d + num_queries;
    let mut mask = vec![0.0f32; n * n];
    let mut block = |r0: usize, r1: usize, c0: usize, c1: usize| {
        for r in r0..r1 {
            for c in c0..c1 {
                mask[r * n + c] = f32::NEG_INFINITY;
            }
        }
    };
    block(d, n, 0, d);
    let g = max_gt * 2;
    for i in 0..num_group {
        if i == 0 {
            block(g * i, g * (i + 1), g * (i + 1), d);
        }
        if i == num_group - 1 {
            block(g * i, g * (i + 1), 0, g * i);
        } else {
            block(g * i, g * (i + 1), g * (i + 1), d);
            block(g * i, g * (i + 1), 0, g * i);
        }
    }
    Some(DnGroup { classes, pad, boxes_unact, mask, num_dn: d, num_group, matches })
}

/// L1 and GIoU of the matched boxes, weighted and divided by `num_boxes`; `matches[b]` =
/// (query, target) pairs. Returns (loss_bbox, loss_giou).
pub(crate) fn box_losses<B: Backend>(t: &mut Tape<B>, boxes: Var, matches: &[Vec<(usize, usize)>], targets: &[Target], w_bbox: f32, w_giou: f32, num_boxes: f32) -> Result<[Var; 2]> {
    let (bsz, q) = { let s = t.shape(boxes); (s[0], s[1]) };
    // boxes: gather the matched predictions
    let idx: Vec<usize> = matches.iter().enumerate().flat_map(|(b, m)| m.iter().map(move |&(qi, _)| b * q + qi)).collect();
    if idx.is_empty() {
        let z = t.input(&[0.0], &[1]);
        let z2 = t.input(&[0.0], &[1]);
        return Ok([z, z2]);
    }
    let tb: Vec<f32> = matches.iter().enumerate().flat_map(|(b, m)| m.iter().flat_map(move |&(_, tj)| targets[b].boxes[tj])).collect();
    let txy: Vec<f32> = matches.iter().enumerate().flat_map(|(b, m)| m.iter().flat_map(move |&(_, tj)| xyxy(targets[b].boxes[tj]))).collect();
    let k = idx.len();
    let flat = t.reshape(boxes, &[1, bsz * q, 4])?;
    let src = t.gather_rows(flat, &idx, k)?; // [1, k, 4]
    let tgt = t.input(&tb, &[1, k, 4]);
    let d = t.sub(src, tgt)?;
    let (dr, dn) = (t.relu(d), t.unary(Unary::Neg, d));
    let dn = t.relu(dn);
    let l1 = t.add(dr, dn)?;
    let l1 = t.sum_scaled(l1, w_bbox / num_boxes);
    // GIoU on the tape
    let col = |t: &mut Tape<B>, v: Var, i: usize| t.slice(v, 2, i, i + 1);
    let (cx, cy, w, h) = (col(t, src, 0)?, col(t, src, 1)?, col(t, src, 2)?, col(t, src, 3)?);
    let (hw, hh) = (t.scale(w, 0.5)?, t.scale(h, 0.5)?);
    let (x1, y1, x2, y2) = (t.sub(cx, hw)?, t.sub(cy, hh)?, t.add(cx, hw)?, t.add(cy, hh)?);
    let tx = t.input(&txy, &[1, k, 4]);
    let (tx1, ty1, tx2, ty2) = (col(t, tx, 0)?, col(t, tx, 1)?, col(t, tx, 2)?, col(t, tx, 3)?);
    let zero = t.input(&[0.0], &[1]);
    let (ix1, iy1) = (t.binary(crate::Binary::Max, x1, tx1)?, t.binary(crate::Binary::Max, y1, ty1)?);
    let (ix2, iy2) = (t.binary(crate::Binary::Min, x2, tx2)?, t.binary(crate::Binary::Min, y2, ty2)?);
    let iw = t.sub(ix2, ix1)?;
    let iw = t.binary(crate::Binary::Max, iw, zero)?;
    let ih = t.sub(iy2, iy1)?;
    let ih = t.binary(crate::Binary::Max, ih, zero)?;
    let inter = t.mul(iw, ih)?;
    let a1 = t.mul(w, h)?;
    let (tw, th) = (t.sub(tx2, tx1)?, t.sub(ty2, ty1)?);
    let a2 = t.mul(tw, th)?;
    let un = t.add(a1, a2)?;
    let un = t.sub(un, inter)?;
    let iou = t.div(inter, un)?;
    let (cx1, cy1) = (t.binary(crate::Binary::Min, x1, tx1)?, t.binary(crate::Binary::Min, y1, ty1)?);
    let (cx2, cy2) = (t.binary(crate::Binary::Max, x2, tx2)?, t.binary(crate::Binary::Max, y2, ty2)?);
    let cw = t.sub(cx2, cx1)?;
    let cw = t.binary(crate::Binary::Max, cw, zero)?;
    let chh = t.sub(cy2, cy1)?;
    let chh = t.binary(crate::Binary::Max, chh, zero)?;
    let carea = t.mul(cw, chh)?;
    let extra = t.sub(carea, un)?;
    let extra = t.div(extra, carea)?;
    let giou = t.sub(iou, extra)?;
    // Σ (1 − giou) = k − Σ giou
    let s = t.sum_scaled(giou, -w_giou / num_boxes);
    let lg = t.add_scalar(s, w_giou * k as f32 / num_boxes)?;
    Ok([l1, lg])
}

pub struct Criterion {
    pub num_classes: usize,
    pub alpha: f32,
    pub gamma: f32,
    pub w_vfl: f32,
    pub w_bbox: f32,
    pub w_giou: f32,
    pub matcher: Matcher,
}

impl Criterion {
    /// RTDETRCriterionv2 as configured for RT-DETRv2 (VFL α 0.75, γ 2; weights 1 / 5 / 2).
    pub fn rtdetrv2(num_classes: usize) -> Self {
        Criterion { num_classes, alpha: 0.75, gamma: 2.0, w_vfl: 1.0, w_bbox: 5.0, w_giou: 2.0, matcher: Matcher::default() }
    }

    /// VFL + L1 + GIoU for one output set; `matches[b]` = (query, target) pairs.
    /// Returns the three weighted terms (loss_vfl, loss_bbox, loss_giou).
    fn set_loss<B: Backend>(&self, t: &mut Tape<B>, logits: Var, boxes: Var, matches: &[Vec<(usize, usize)>], targets: &[Target], num_boxes: f32) -> Result<[Var; 3]> {
        let (bsz, q, c) = { let s = t.shape(logits); (s[0], s[1], s[2]) };
        let (lv, bv) = (t.value(logits), t.value(boxes));
        // VFL targets: IoU (detached) at the matched (query, class); weight α·p^γ on negatives
        let mut target = vec![0.0f32; bsz * q * c];
        let mut onehot = vec![0.0f32; bsz * q * c];
        for (b, m) in matches.iter().enumerate() {
            for &(qi, tj) in m {
                let pb = [bv[(b * q + qi) * 4], bv[(b * q + qi) * 4 + 1], bv[(b * q + qi) * 4 + 2], bv[(b * q + qi) * 4 + 3]];
                let (iou, _) = iou_giou(xyxy(pb), xyxy(targets[b].boxes[tj]));
                let o = (b * q + qi) * c + targets[b].labels[tj];
                target[o] = iou;
                onehot[o] = 1.0;
            }
        }
        let weight: Vec<f32> = (0..bsz * q * c).map(|i| self.alpha * (1.0 / (1.0 + (-lv[i]).exp())).powf(self.gamma) * (1.0 - onehot[i]) + target[i]).collect();
        let shape = [bsz, q, c];
        let (tv, wv) = (t.input(&target, &shape), t.input(&weight, &shape));
        // BCE with logits, stable: relu(x) − x·t + log(1 + exp(−|x|))
        let r = t.relu(logits);
        let nx = t.unary(Unary::Neg, logits);
        let rn = t.relu(nx);
        let abs = t.add(r, rn)?;
        let na = t.unary(Unary::Neg, abs);
        let e = t.unary(Unary::Exp, na);
        let e1 = t.add_scalar(e, 1.0)?;
        let sp = t.unary(Unary::Log, e1);
        let xt = t.mul(logits, tv)?;
        let bce = t.sub(r, xt)?;
        let bce = t.add(bce, sp)?;
        let wb = t.mul(bce, wv)?;
        let vfl = t.sum_scaled(wb, self.w_vfl / num_boxes);
        let [l1, lg] = box_losses(t, boxes, matches, targets, self.w_bbox, self.w_giou, num_boxes)?;
        Ok([vfl, l1, lg])
    }

    fn match_set<B: Backend>(&self, t: &Tape<B>, logits: Var, boxes: Var, targets: &[Target]) -> Vec<Vec<(usize, usize)>> {
        let (q, c) = { let s = t.shape(logits); (s[1], s[2]) };
        let (lv, bv) = (t.value(logits), t.value(boxes));
        targets.iter().enumerate().map(|(b, tg)| self.matcher.assign(&lv[b * q * c..(b + 1) * q * c], &bv[b * q * 4..(b + 1) * q * 4], c, tg)).collect()
    }

    /// Total loss of a training forward, and every term by its PyTorch name.
    pub fn forward<B: Backend>(&self, t: &mut Tape<B>, out: &Outputs, targets: &[Target], dn: Option<&DnGroup>) -> Result<(Var, Vec<(String, Var)>)> {
        let num_boxes = (targets.iter().map(|x| x.labels.len()).sum::<usize>() as f32).max(1.0);
        let layers = out.logits.len();
        let mut terms: Vec<(String, Var)> = vec![];
        let names = ["loss_vfl", "loss_bbox", "loss_giou"];
        let push = |suffix: &str, v: [Var; 3], terms: &mut Vec<(String, Var)>| {
            for (n, x) in names.iter().zip(v) {
                terms.push((format!("{n}{suffix}"), x));
            }
        };
        // split denoising queries (they come first) from the matching queries
        let (mut logits, mut boxes, mut dn_sets) = (vec![], vec![], vec![]);
        for i in 0..layers {
            let (l, b) = (out.logits[i], out.boxes[i]);
            match dn {
                Some(g) => {
                    let q = t.shape(l)[1];
                    logits.push(t.slice(l, 1, g.num_dn, q)?);
                    boxes.push(t.slice(b, 1, g.num_dn, q)?);
                    dn_sets.push((t.slice(l, 1, 0, g.num_dn)?, t.slice(b, 1, 0, g.num_dn)?));
                }
                None => {
                    logits.push(l);
                    boxes.push(b);
                }
            }
        }
        let last = layers - 1;
        let m = self.match_set(t, logits[last], boxes[last], targets);
        let v = self.set_loss(t, logits[last], boxes[last], &m, targets, num_boxes)?;
        push("", v, &mut terms);
        for i in 0..last {
            let m = self.match_set(t, logits[i], boxes[i], targets);
            let v = self.set_loss(t, logits[i], boxes[i], &m, targets, num_boxes)?;
            push(&format!("_aux_{i}"), v, &mut terms);
        }
        if let Some(g) = dn {
            for (i, &(l, b)) in dn_sets.iter().enumerate() {
                let v = self.set_loss(t, l, b, &g.matches, targets, num_boxes * g.num_group as f32)?;
                push(&format!("_dn_{i}"), v, &mut terms);
            }
        }
        if let (Some(l), Some(b)) = (out.enc_logits, out.enc_boxes) {
            let m = self.match_set(t, l, b, targets);
            let v = self.set_loss(t, l, b, &m, targets, num_boxes)?;
            push("_enc_0", v, &mut terms);
        }
        let mut total = terms[0].1;
        for &(_, v) in &terms[1..] {
            total = t.add(total, v)?;
        }
        Ok((total, terms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungarian_finds_the_optimum() {
        // brute force over every assignment of 3 rows into 4 columns
        let cost = [4.0, 1.0, 3.0, 2.0, 2.0, 0.0, 5.0, 3.0, 3.0, 2.0, 2.0, 1.0];
        let got = hungarian(&cost, 3, 4);
        let total: f32 = got.iter().map(|&(r, c)| cost[r * 4 + c]).sum();
        let mut best = f32::MAX;
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    if a != b && b != c && a != c {
                        best = best.min(cost[a] + cost[4 + b] + cost[8 + c]);
                    }
                }
            }
        }
        assert_eq!(total, best);
        assert_eq!(got.len(), 3);
        // rows > cols: transposed internally, still one pair per column
        let t = hungarian(&[1.0, 9.0, 8.0, 2.0, 3.0, 3.0], 3, 2);
        assert_eq!(t, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn denoising_mask_and_matches() {
        let tg = vec![Target { labels: vec![1, 2], boxes: vec![[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]] }, Target { labels: vec![3], boxes: vec![[0.6, 0.4, 0.2, 0.3]] }];
        let g = denoising(&tg, 15, 300, 100, 0.5, 1.0, &mut Rng::new(1)).unwrap();
        assert_eq!(g.num_group, 50);
        assert_eq!(g.num_dn, 200);
        // positives: group k slots 4k, 4k+1 for image 0; 4k for image 1
        assert_eq!(g.matches[0][..2], [(0, 0), (1, 1)]);
        assert_eq!(g.matches[1][..2], [(0, 0), (4, 0)]);
        assert_eq!(g.matches[0].len(), 100);
        let n = 500;
        // a matching query never sees denoising slots; group 0 never sees group 1
        assert!(g.mask[300 * n + 5] == f32::NEG_INFINITY && g.mask[300 * n + 300] == 0.0);
        assert!(g.mask[5] == f32::NEG_INFINITY && g.mask[3] == 0.0);
        assert!(g.pad[1] == 1.0 && g.pad[200 + 1] == 0.0);
    }
}
