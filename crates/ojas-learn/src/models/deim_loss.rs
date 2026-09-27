//! The DEIM training loss for D-FINE, ported from Intellindust-AI-Lab/DEIM (Apache-2.0),
//! `engine/deim/deim_criterion.py` and `dfine_utils.py`, as configured by `configs/base/deim.yml`:
//!
//! - MAL (matchability-aware): BCE against IoU^γ at the matched class, weight p^γ on negatives
//!   and 1 on positives; each output set uses its own Hungarian matches.
//! - L1 + GIoU and FGL use the GO union: every image's matches from all decoder layers, the
//!   pre-refinement head and the encoder, one target per query, most-agreed first.
//! - FGL (fine-grained localisation): each edge's distribution against the two bins either
//!   side of the true edge distance, weighted by the bins' proximity and the box IoU.
//! - DDF (decoupled distillation): every non-final layer's edge distributions against the
//!   last layer's at temperature 5, weighted by IoU (matched) or the teacher's confidence
//!   (unmatched), the two groups averaged separately and mixed by √count.
//!
//! Denoising queries get the same terms against their fixed pairs. Matching and target
//! construction are host code; every loss is a tape op.

use anyhow::Result;

use crate::backend::Backend;
use crate::models::detr_loss::{box_losses, iou_giou, xyxy, DnGroup, Matcher, Target};
use crate::models::dfine::{weighting_function, DfineOutputs, HeadOutputs};
use crate::tape::{Tape, Var};
use crate::Unary;

pub struct DeimCriterion {
    pub num_classes: usize,
    /// MAL focusing exponent (deim.yml: 1.5)
    pub gamma: f32,
    pub w_mal: f32,
    pub w_bbox: f32,
    pub w_giou: f32,
    pub w_fgl: f32,
    pub w_ddf: f32,
    pub reg_max: usize,
    pub reg_scale: f32,
    pub up: f32,
    /// DDF temperature
    pub temperature: f32,
    pub matcher: Matcher,
}

impl DeimCriterion {
    /// `DEIMCriterion` as `configs/base/deim.yml` sets it: weights mal 1, bbox 5, giou 2, fgl 0.15,
    /// ddf 1.5; γ 1.5; the focal matcher (class 2, bbox 5, giou 2, α 0.25, γ 2).
    pub fn deim(num_classes: usize) -> Self {
        DeimCriterion {
            num_classes,
            gamma: 1.5,
            w_mal: 1.0,
            w_bbox: 5.0,
            w_giou: 2.0,
            w_fgl: 0.15,
            w_ddf: 1.5,
            reg_max: 32,
            reg_scale: 4.0,
            up: 0.5,
            temperature: 5.0,
            matcher: Matcher::default(),
        }
    }
}

type Matches = Vec<Vec<(usize, usize)>>;

/// log_softmax over the last axis of [R, K], shifted by the (constant) row max.
fn log_softmax_rows<B: Backend>(t: &mut Tape<B>, x: Var) -> Result<Var> {
    let s = t.shape(x).to_vec();
    let (r, k) = (s[0], s[1]);
    let v = t.value(x);
    let m: Vec<f32> = (0..r).map(|i| v[i * k..(i + 1) * k].iter().cloned().fold(f32::NEG_INFINITY, f32::max)).collect();
    let m = t.input(&m, &[r, 1]);
    let y = t.sub(x, m)?;
    let e = t.unary(Unary::Exp, y);
    let se = t.sum_axis(e, 1)?;
    let se = t.reshape(se, &[r, 1])?;
    let ls = t.unary(Unary::Log, se);
    t.sub(y, ls)
}

/// Stable BCE-with-logits · weight, summed and scaled: relu(x) − x·t + log(1 + exp(−|x|)).
fn weighted_bce_sum<B: Backend>(t: &mut Tape<B>, logits: Var, target: &[f32], weight: &[f32], scale: f32) -> Result<Var> {
    let shape = t.shape(logits).to_vec();
    let (tv, wv) = (t.input(target, &shape), t.input(weight, &shape));
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
    Ok(t.sum_scaled(wb, scale))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn box_at(v: &[f32], i: usize) -> [f32; 4] {
    [v[i * 4], v[i * 4 + 1], v[i * 4 + 2], v[i * 4 + 3]]
}

/// `_get_go_indices` for one image: the union of every set's (query, target) pairs, most-agreed
/// pair first (ties in (query, target) order), one target per query.
fn go_union(sets: &[&Vec<(usize, usize)>]) -> Vec<(usize, usize)> {
    let mut all: Vec<(usize, usize)> = sets.iter().flat_map(|s| s.iter().copied()).collect();
    all.sort();
    let mut uniq: Vec<((usize, usize), usize)> = vec![];
    for p in all {
        match uniq.last_mut() {
            Some((q, n)) if *q == p => *n += 1,
            _ => uniq.push((p, 1)),
        }
    }
    uniq.sort_by(|a, b| b.1.cmp(&a.1));
    let mut seen = std::collections::HashSet::new();
    uniq.into_iter().filter(|((r, _), _)| seen.insert(*r)).map(|(p, _)| p).collect()
}

/// One output set to score.
struct Set {
    logits: Var,
    boxes: Var,
    /// FDR corners and their reference boxes (decoder layers only)
    local: Option<(Var, Var)>,
    /// DDF teacher: the last layer's corners and logits (non-final decoder layers only)
    teacher: Option<(Var, Var)>,
}

impl DeimCriterion {
    fn match_set<B: Backend>(&self, t: &Tape<B>, logits: Var, boxes: Var, targets: &[Target]) -> Matches {
        let (q, c) = { let s = t.shape(logits); (s[1], s[2]) };
        let (lv, bv) = (t.value(logits), t.value(boxes));
        targets.iter().enumerate().map(|(b, tg)| self.matcher.assign(&lv[b * q * c..(b + 1) * q * c], &bv[b * q * 4..(b + 1) * q * 4], c, tg)).collect()
    }

    /// Matched IoUs (detached) of a set's boxes, in `matches` order.
    fn ious<B: Backend>(&self, t: &Tape<B>, boxes: Var, matches: &Matches, targets: &[Target]) -> Vec<f32> {
        let q = t.shape(boxes)[1];
        let bv = t.value(boxes);
        matches.iter().enumerate().flat_map(|(b, m)| m.iter().map(|&(qi, tj)| iou_giou(xyxy(box_at(&bv, b * q + qi)), xyxy(targets[b].boxes[tj])).0).collect::<Vec<_>>()).collect()
    }

    /// loss_labels_mal.
    fn mal<B: Backend>(&self, t: &mut Tape<B>, logits: Var, boxes: Var, matches: &Matches, targets: &[Target], num_boxes: f32) -> Result<Var> {
        let (bsz, q, c) = { let s = t.shape(logits); (s[0], s[1], s[2]) };
        let lv = t.value(logits);
        let ious = self.ious(t, boxes, matches, targets);
        let mut target = vec![0.0f32; bsz * q * c];
        let mut onehot = vec![0.0f32; bsz * q * c];
        let mut k = 0;
        for (b, m) in matches.iter().enumerate() {
            for &(qi, tj) in m {
                let o = (b * q + qi) * c + targets[b].labels[tj];
                target[o] = ious[k].powf(self.gamma);
                onehot[o] = 1.0;
                k += 1;
            }
        }
        let weight: Vec<f32> = (0..bsz * q * c).map(|i| sigmoid(lv[i]).powf(self.gamma) * (1.0 - onehot[i]) + onehot[i]).collect();
        weighted_bce_sum(t, logits, &target, &weight, self.w_mal / num_boxes)
    }

    /// bbox2distance + translate_gt for one matched pair: per edge (left bin, weight left,
    /// weight right).
    fn fgl_target(&self, w: &[f32], refb: [f32; 4], tgt: [f32; 4]) -> [(usize, f32, f32); 4] {
        let rs = self.reg_scale.abs();
        let r = self.reg_max;
        let b = xyxy(tgt);
        let (sx, sy) = (refb[2] / rs + 1e-16, refb[3] / rs + 1e-16);
        let d = [(refb[0] - b[0]) / sx - 0.5 * rs, (refb[1] - b[1]) / sy - 0.5 * rs, (b[2] - refb[0]) / sx - 0.5 * rs, (b[3] - refb[1]) / sy - 0.5 * rs];
        let mut out = [(0, 0.0, 0.0); 4];
        for (e, &g) in d.iter().enumerate() {
            let idx = w.iter().filter(|&&v| v - g <= 0.0).count() as i64 - 1;
            let (label, wl, wr) = if idx < 0 {
                (0.0, 1.0, 0.0)
            } else if idx as usize >= r {
                (r as f32 - 0.1, 0.0, 1.0)
            } else {
                let i = idx as usize;
                let (l, rt) = ((g - w[i]).abs(), (w[i + 1] - g).abs());
                let wr = l / (l + rt);
                (i as f32, 1.0 - wr, wr)
            };
            out[e] = (label.clamp(0.0, r as f32 - 0.1) as usize, wl, wr);
        }
        out
    }

    /// loss_local's FGL: Σ −(log p[left]·wl + log p[right]·wr)·IoU / num_boxes.
    fn fgl<B: Backend>(&self, t: &mut Tape<B>, corners: Var, refs: Var, matches: &Matches, ious: &[f32], targets: &[Target], num_boxes: f32) -> Result<Option<Var>> {
        let (bsz, q) = { let s = t.shape(corners); (s[0], s[1]) };
        let idx: Vec<usize> = matches.iter().enumerate().flat_map(|(b, m)| m.iter().map(move |&(qi, _)| b * q + qi)).collect();
        if idx.is_empty() {
            return Ok(None);
        }
        let k = idx.len();
        let bins = self.reg_max + 1;
        let w = weighting_function(self.reg_max, self.up, self.reg_scale);
        let rv = t.value(refs);
        let (mut pick, mut wt) = (Vec::with_capacity(k * 8), Vec::with_capacity(k * 8));
        let mut n = 0;
        for (b, m) in matches.iter().enumerate() {
            for &(qi, tj) in m {
                for (left, wl, wr) in self.fgl_target(&w, box_at(&rv, b * q + qi), targets[b].boxes[tj]) {
                    pick.extend([left, left + 1]);
                    wt.extend([-wl * ious[n] * self.w_fgl / num_boxes, -wr * ious[n] * self.w_fgl / num_boxes]);
                }
                n += 1;
            }
        }
        let flat = t.reshape(corners, &[1, bsz * q, 4 * bins])?;
        let sel = t.gather_rows(flat, &idx, k)?;
        let sel = t.reshape(sel, &[k * 4, bins])?;
        let lp = log_softmax_rows(t, sel)?;
        let g = t.gather_last(lp, &pick, 2)?;
        let wv = t.input(&wt, &[k * 4, 2]);
        let y = t.mul(g, wv)?;
        Ok(Some(t.sum(y)))
    }

    /// loss_local's DDF against the teacher. `pos` = (num_pos, num_neg) from the matching queries.
    #[allow(clippy::too_many_arguments)]
    fn ddf<B: Backend>(&self, t: &mut Tape<B>, corners: Var, teacher: (Var, Var), matches: &Matches, ious: &[f32], pos: (f32, f32)) -> Result<Option<Var>> {
        let (bsz, q) = { let s = t.shape(corners); (s[0], s[1]) };
        let bins = self.reg_max + 1;
        let rows = bsz * q * 4;
        let (tc, tl) = (t.value(teacher.0), t.value(teacher.1));
        let pc = t.value(corners);
        if pc == tc {
            return Ok(None);
        }
        let c = t.shape(teacher.1)[2];
        // per-query weight: teacher confidence, or the IoU where matched
        let mut wq: Vec<f32> = (0..bsz * q).map(|i| tl[i * c..(i + 1) * c].iter().map(|&x| sigmoid(x)).fold(f32::NEG_INFINITY, f32::max)).collect();
        let mut mask = vec![false; bsz * q];
        let mut n = 0;
        for (b, m) in matches.iter().enumerate() {
            for &(qi, _) in m {
                wq[b * q + qi] = ious[n];
                mask[b * q + qi] = true;
                n += 1;
            }
        }
        let n_pos = mask.iter().filter(|&&x| x).count() * 4;
        let n_neg = rows - n_pos;
        let (num_pos, num_neg) = pos;
        let tt = self.temperature;
        // row coefficient: w·T² · (mask ? num_pos/n_pos : num_neg/n_neg) / (num_pos + num_neg)
        let coef = |qi: usize| -> f32 {
            let share = if mask[qi] {
                if n_pos > 0 { num_pos / n_pos as f32 } else { 0.0 }
            } else if n_neg > 0 {
                num_neg / n_neg as f32
            } else {
                0.0
            };
            wq[qi] * tt * tt * share / (num_pos + num_neg)
        };
        // KL(p ‖ q) = Σ p·(log p − log q), p = softmax(teacher/T) (constant); the difference is
        // taken per element, as PyTorch does — the KL is ~1e-4 per row, far below either sum
        let (mut wmat, mut logp) = (vec![0.0f32; rows * bins], vec![0.0f32; rows * bins]);
        for r in 0..rows {
            let z = &tc[r * bins..(r + 1) * bins];
            let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max) / tt;
            let ls = z.iter().map(|&v| (v / tt - m).exp()).sum::<f32>().ln();
            let cr = coef(r / 4);
            for j in 0..bins {
                let lp = z[j] / tt - m - ls;
                logp[r * bins + j] = lp;
                wmat[r * bins + j] = cr * lp.exp() * self.w_ddf;
            }
        }
        let x = t.reshape(corners, &[rows, bins])?;
        let x = t.scale(x, 1.0 / tt)?;
        let lq = log_softmax_rows(t, x)?;
        let lp = t.input(&logp, &[rows, bins]);
        let diff = t.sub(lp, lq)?;
        let wv = t.input(&wmat, &[rows, bins]);
        let y = t.mul(diff, wv)?;
        Ok(Some(t.sum(y)))
    }

    /// Every term of one set. `cls` matches drive MAL; `boxm` (the GO union, or the dn pairs)
    /// drive L1, GIoU, FGL and DDF.
    #[allow(clippy::too_many_arguments)]
    fn set_terms<B: Backend>(&self, t: &mut Tape<B>, s: &Set, cls: &Matches, nb_cls: f32, boxm: &Matches, nb_box: f32, targets: &[Target], pos: (f32, f32), suffix: &str, terms: &mut Vec<(String, Var)>) -> Result<()> {
        terms.push((format!("loss_mal{suffix}"), self.mal(t, s.logits, s.boxes, cls, targets, nb_cls)?));
        let [l1, lg] = box_losses(t, s.boxes, boxm, targets, self.w_bbox, self.w_giou, nb_box)?;
        terms.push((format!("loss_bbox{suffix}"), l1));
        terms.push((format!("loss_giou{suffix}"), lg));
        if let Some((corners, refs)) = s.local {
            let ious = self.ious(t, s.boxes, boxm, targets);
            if let Some(v) = self.fgl(t, corners, refs, boxm, &ious, targets, nb_box)? {
                terms.push((format!("loss_fgl{suffix}"), v));
            }
            if let Some(teacher) = s.teacher {
                if let Some(v) = self.ddf(t, corners, teacher, boxm, &ious, pos)? {
                    terms.push((format!("loss_ddf{suffix}"), v));
                }
            }
        }
        Ok(())
    }

    fn layer_sets(h: &HeadOutputs) -> Vec<Set> {
        let last = h.logits.len() - 1;
        (0..=last)
            .map(|i| Set {
                logits: h.logits[i],
                boxes: h.boxes[i],
                local: Some((h.corners[i], h.refs[i])),
                teacher: if i < last { Some((h.corners[last], h.logits[last])) } else { None },
            })
            .collect()
    }

    /// Total loss of a training forward, and every weighted term by its PyTorch name.
    pub fn forward<B: Backend>(&self, t: &mut Tape<B>, out: &DfineOutputs, targets: &[Target], dn: Option<&DnGroup>) -> Result<(Var, Vec<(String, Var)>)> {
        let num_boxes = (targets.iter().map(|x| x.labels.len()).sum::<usize>() as f32).max(1.0);
        let layers = Self::layer_sets(&out.main);
        let last = layers.len() - 1;
        let pre = Set { logits: out.main.pre_logits, boxes: out.main.pre_boxes, local: None, teacher: None };
        let enc = match (out.enc_logits, out.enc_boxes) {
            (Some(l), Some(b)) => Some(Set { logits: l, boxes: b, local: None, teacher: None }),
            _ => None,
        };
        // own matches: last layer, aux layers, pre head, encoder
        let m_last = self.match_set(t, layers[last].logits, layers[last].boxes, targets);
        let m_aux: Vec<Matches> = layers[..last].iter().map(|s| self.match_set(t, s.logits, s.boxes, targets)).collect();
        let m_pre = self.match_set(t, pre.logits, pre.boxes, targets);
        let m_enc = enc.as_ref().map(|s| self.match_set(t, s.logits, s.boxes, targets));
        let go: Matches = (0..targets.len())
            .map(|b| {
                let mut sets = vec![&m_last[b]];
                sets.extend(m_aux.iter().map(|m| &m[b]));
                sets.push(&m_pre[b]);
                if let Some(m) = &m_enc {
                    sets.push(&m[b]);
                }
                go_union(&sets)
            })
            .collect();
        let num_go = (go.iter().map(|m| m.len()).sum::<usize>() as f32).max(1.0);
        // DDF's positive / negative mix, from the matching queries' GO mask
        let (bsz, q) = { let s = t.shape(layers[last].logits); (s[0], s[1]) };
        let scale = 8.0 / bsz as f32;
        let n_pos = go.iter().map(|m| m.len()).sum::<usize>() as f32 * 4.0;
        let pos = ((n_pos * scale).sqrt(), (((bsz * q * 4) as f32 - n_pos) * scale).sqrt());

        let mut terms: Vec<(String, Var)> = vec![];
        self.set_terms(t, &layers[last], &m_last, num_boxes, &go, num_go, targets, pos, "", &mut terms)?;
        for (i, s) in layers[..last].iter().enumerate() {
            self.set_terms(t, s, &m_aux[i], num_boxes, &go, num_go, targets, pos, &format!("_aux_{i}"), &mut terms)?;
        }
        self.set_terms(t, &pre, &m_pre, num_boxes, &go, num_go, targets, pos, "_pre", &mut terms)?;
        if let (Some(s), Some(m)) = (&enc, &m_enc) {
            self.set_terms(t, s, m, num_boxes, &go, num_go, targets, pos, "_enc_0", &mut terms)?;
        }
        if let (Some(g), Some(h)) = (dn, &out.dn) {
            let nb = num_boxes * g.num_group as f32;
            for (i, s) in Self::layer_sets(h).iter().enumerate() {
                self.set_terms(t, s, &g.matches, nb, &g.matches, nb, targets, pos, &format!("_dn_{i}"), &mut terms)?;
            }
            let s = Set { logits: h.pre_logits, boxes: h.pre_boxes, local: None, teacher: None };
            self.set_terms(t, &s, &g.matches, nb, &g.matches, nb, targets, pos, "_dn_pre", &mut terms)?;
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
    fn go_union_prefers_the_most_agreed_target() {
        let a = vec![(3, 0), (7, 1)];
        let b = vec![(3, 0), (7, 2)];
        let c = vec![(3, 1), (9, 2)];
        let u = go_union(&[&a, &b, &c]);
        // (3, 0) twice beats (3, 1); query 7 ties → lower target first
        assert_eq!(u, vec![(3, 0), (7, 1), (9, 2)]);
    }

    #[test]
    fn fgl_target_brackets_the_edge() {
        let c = DeimCriterion::deim(15);
        let w = weighting_function(32, 0.5, 4.0);
        // target equal to the reference: every edge distance is 0 (W[16]), up to float noise
        let r = c.fgl_target(&w, [0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]);
        for (left, wl, wr) in r {
            assert!(left == 15 || left == 16);
            assert!((wl + wr - 1.0).abs() < 1e-6);
            assert!((w[left] * wl + w[left + 1] * wr).abs() < 1e-5);
        }
        // a real edge: the two weights interpolate it back
        let r = c.fgl_target(&w, [0.5, 0.5, 0.2, 0.2], [0.52, 0.5, 0.3, 0.2]);
        let d = (0.5 - (0.52 - 0.15)) / 0.05 - 2.0;
        assert!((w[r[0].0] * r[0].1 + w[r[0].0 + 1] * r[0].2 - d).abs() < 1e-4);
        // far outside: clamped into the last bin pair, all weight right
        let r = c.fgl_target(&w, [0.5, 0.5, 0.01, 0.01], [0.5, 0.5, 0.9, 0.9]);
        assert_eq!(r[0], (31, 0.0, 1.0));
    }
}
