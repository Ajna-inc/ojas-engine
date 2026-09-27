//! D-FINE / DEIM on the tape — ported from Intellindust-AI-Lab/DEIM (Apache-2.0),
//! `engine/deim/hybrid_encoder.py` (version 'dfine') and `dfine_decoder.py`. Parameter names are
//! the checkpoint's, so a DEIM `.pth` loads without a mapping.
//!
//! The encoder differs from RT-DETR's in three places: the lateral 1×1 has no activation, the
//! fusion blocks are RepNCSPELAN4 (split → two CSP branches of re-parameterisable VGG blocks →
//! concat), and the bottom-up downsample is SCDown (1×1 then a depthwise 3×3 stride 2).

use anyhow::{anyhow, ensure, Result};

use crate::backend::Backend;
use crate::models::hgnetv2::HgConfig;
use crate::models::rtdetr::Store;
use crate::tape::{Tape, Var};
use crate::Unary;

#[derive(Clone, Debug)]
pub struct DfineConfig {
    pub backbone: HgConfig,
    /// backbone channels feeding the encoder, one per level
    pub in_channels: Vec<usize>,
    pub feat_strides: Vec<usize>,
    pub hidden: usize,
    pub heads: usize,
    /// encoder feed-forward width
    pub ffn: usize,
    /// levels that get the transformer layer (the coarsest, for every published config)
    pub use_encoder_idx: Vec<usize>,
    /// RepNCSPELAN4 branch width, `round(expansion · hidden // 2)`
    pub elan_c4: usize,
    /// VGG blocks per CSP branch, `round(3 · depth_mult)`
    pub elan_n: usize,
    pub num_classes: usize,
    pub num_queries: usize,
    pub dec_layers: usize,
    /// decoder feed-forward width
    pub dec_ffn: usize,
    /// deformable sampling points per level
    pub points: Vec<usize>,
    /// FDR: bins per edge − 1
    pub reg_max: usize,
    /// FDR: curvature and upper bound of the weighting function W(n)
    pub reg_scale: f32,
    pub up: f32,
}

impl DfineConfig {
    /// `training/deim/step3_ojas_n32.yml`: D-FINE-N with the encoder and decoder narrowed to 96.
    pub fn ojas_n32() -> Self {
        let hidden = 96;
        let expansion = 0.34f64;
        DfineConfig {
            backbone: HgConfig::b0(&[2, 3]),
            in_channels: vec![512, 1024],
            feat_strides: vec![16, 32],
            hidden,
            heads: 8,
            ffn: 384,
            use_encoder_idx: vec![1],
            elan_c4: py_round((expansion * hidden as f64 / 2.0).floor()),
            elan_n: py_round(3.0 * 0.5),
            num_classes: 15,
            num_queries: 300,
            dec_layers: 4,
            dec_ffn: 384,
            points: vec![6, 6],
            reg_max: 32,
            reg_scale: 4.0,
            up: 0.5,
        }
    }
}

/// `weighting_function(reg_max, up, reg_scale)`: the 33 non-uniform bin positions W(n), dense
/// near 0 and sparse towards ±2·up·reg_scale.
pub fn weighting_function(reg_max: usize, up: f32, reg_scale: f32) -> Vec<f32> {
    let (ub1, ub2) = ((up * reg_scale).abs(), (up * reg_scale).abs() * 2.0);
    let step = (ub1 + 1.0).powf(2.0 / (reg_max as f32 - 2.0));
    let half = reg_max / 2;
    let mut v = vec![-ub2];
    v.extend((1..half).rev().map(|i| -step.powi(i as i32) + 1.0));
    v.push(0.0);
    v.extend((1..half).map(|i| step.powi(i as i32) - 1.0));
    v.push(ub2);
    v
}

/// `_generate_anchors`: per level a grid of (cx, cy, w, h) with w = h = 0.05·2^level, as logits;
/// cells within `eps` of the border are +∞ and masked out of the memory. Returns the anchors
/// [ΣHW·4] and the mask [ΣHW].
pub fn anchors(shapes: &[(usize, usize)], eps: f32) -> (Vec<f32>, Vec<f32>) {
    let (mut a, mut m) = (vec![], vec![]);
    for (lvl, &(h, w)) in shapes.iter().enumerate() {
        let wh = 0.05f32 * 2f32.powi(lvl as i32);
        for y in 0..h {
            for x in 0..w {
                let box4 = [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32, wh, wh];
                let valid = box4.iter().all(|&v| v > eps && v < 1.0 - eps);
                m.push(if valid { 1.0 } else { 0.0 });
                a.extend(box4.iter().map(|&v| if valid { (v / (1.0 - v)).ln() } else { f32::INFINITY }));
            }
        }
    }
    (a, m)
}

/// Top-`k` rows of [B, N, C] by max class logit, per image, largest first.
fn topk_by_max(v: &[f32], b: usize, n: usize, c: usize, k: usize) -> Vec<usize> {
    let mut idx = Vec::with_capacity(b * k);
    for bi in 0..b {
        let mut best: Vec<(f32, usize)> = (0..n).map(|i| (v[(bi * n + i) * c..(bi * n + i + 1) * c].iter().cloned().fold(f32::NEG_INFINITY, f32::max), i)).collect();
        best.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        idx.extend(best[..k].iter().map(|x| x.1));
    }
    idx
}

/// One decoder output set: per layer (training: every layer; eval: the last) the class logits
/// [B, Q, C] (LQE included), boxes [B, Q, 4] (cxcywh, 0..1), FDR corner logits
/// [B, Q, 4·(reg_max+1)] and the reference box they are decoded against; plus layer 0's
/// pre-refinement head.
pub struct HeadOutputs {
    pub logits: Vec<Var>,
    pub boxes: Vec<Var>,
    pub corners: Vec<Var>,
    pub refs: Vec<Var>,
    pub pre_logits: Var,
    pub pre_boxes: Var,
}

pub struct DfineOutputs {
    /// the matching queries
    pub main: HeadOutputs,
    /// training: the denoising queries, when a denoising group was given
    pub dn: Option<HeadOutputs>,
    /// training: encoder top-k proposals — logits [B, Q, C], boxes (sigmoid) [B, Q, 4]
    pub enc_logits: Option<Var>,
    pub enc_boxes: Option<Var>,
}

/// Python's `round`: halves go to the even neighbour.
fn py_round(x: f64) -> usize {
    let r = x.round();
    (if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r }) as usize
}

/// `HybridEncoder.build_2d_sincos_position_embedding(w, h, d, T)`: [w·h, d], w-major.
fn sincos(w: usize, h: usize, d: usize, temperature: f32) -> Vec<f32> {
    let pd = d / 4;
    let omega: Vec<f32> = (0..pd).map(|i| 1.0 / temperature.powf(i as f32 / pd as f32)).collect();
    let mut out = vec![0.0; w * h * d];
    for k in 0..w * h {
        let (gw, gh) = ((k / h) as f32, (k % h) as f32);
        for i in 0..pd {
            let (a, b) = (gw * omega[i], gh * omega[i]);
            out[k * d + i] = a.sin();
            out[k * d + pd + i] = a.cos();
            out[k * d + 2 * pd + i] = b.sin();
            out[k * d + 3 * pd + i] = b.cos();
        }
    }
    out
}

pub struct Dfine<'s, B: Backend> {
    pub cfg: DfineConfig,
    pub st: &'s Store<B>,
    /// BatchNorm in training mode
    pub train: bool,
}

impl<'s, B: Backend> Dfine<'s, B> {
    fn p(&self, t: &mut Tape<B>, name: &str) -> Result<Var> {
        let p = self.st.params.get(name).ok_or_else(|| anyhow!("parameter {name} missing"))?;
        Ok(t.param(p))
    }

    fn linear(&self, t: &mut Tape<B>, x: Var, name: &str) -> Result<Var> {
        let (w, b) = (self.p(t, &format!("{name}.weight"))?, self.p(t, &format!("{name}.bias"))?);
        t.linear(x, w, Some(b))
    }

    fn layer_norm(&self, t: &mut Tape<B>, x: Var, name: &str) -> Result<Var> {
        let (g, b) = (self.p(t, &format!("{name}.weight"))?, self.p(t, &format!("{name}.bias"))?);
        Ok(t.layer_norm(x, g, b, 1e-5))
    }

    /// ConvNormLayer(_fuse): conv (no bias, pad (k−1)/2, `groups`) → BatchNorm → act.
    fn conv_norm(&self, t: &mut Tape<B>, x: Var, name: &str, stride: usize, groups: usize, act: Option<Unary>) -> Result<Var> {
        let w = self.p(t, &format!("{name}.conv.weight"))?;
        let k = t.shape(w)[2];
        let pad = (k - 1) / 2;
        let y = t.conv2d(x, w, None, [stride; 2], [pad; 4], groups)?;
        let (g, b) = (self.p(t, &format!("{name}.norm.weight"))?, self.p(t, &format!("{name}.norm.bias"))?);
        let run = self.st.bns.get(&format!("{name}.norm")).ok_or_else(|| anyhow!("BN stats {name}.norm missing"))?;
        let y = t.batch_norm2d(y, g, b, run, self.train)?;
        Ok(match act {
            Some(a) => t.unary(a, y),
            None => y,
        })
    }

    /// nn.MultiheadAttention (batch_first, packed in_proj), optional additive mask.
    fn mha(&self, t: &mut Tape<B>, q: Var, k: Var, v: Var, name: &str, mask: Option<Var>) -> Result<Var> {
        let d = self.cfg.hidden;
        let (h, hd) = (self.cfg.heads, d / self.cfg.heads);
        let (w, b) = (self.p(t, &format!("{name}.in_proj_weight"))?, self.p(t, &format!("{name}.in_proj_bias"))?);
        let mut proj = |x: Var, i: usize| -> Result<Var> {
            let (wi, bi) = (t.slice(w, 0, i * d, (i + 1) * d)?, t.slice(b, 0, i * d, (i + 1) * d)?);
            let y = t.linear(x, wi, Some(bi))?;
            let s = t.shape(y).to_vec();
            let y = t.reshape(y, &[s[0], s[1], h, hd])?;
            t.permute(y, &[0, 2, 1, 3])
        };
        let (qh, kh, vh) = (proj(q, 0)?, proj(k, 1)?, proj(v, 2)?);
        let sc = t.matmul_opts(qh, kh, true)?;
        let mut sc = t.scale(sc, 1.0 / (hd as f32).sqrt())?;
        if let Some(m) = mask {
            sc = t.add(sc, m)?;
        }
        let pr = t.softmax(sc);
        let o = t.matmul(pr, vh)?;
        let o = t.permute(o, &[0, 2, 1, 3])?;
        let s = t.shape(o).to_vec();
        let o = t.reshape(o, &[s[0], s[1], d])?;
        self.linear(t, o, &format!("{name}.out_proj"))
    }

    /// VGGBlock: 3×3 conv-norm + 1×1 conv-norm, summed, then SiLU.
    fn vgg(&self, t: &mut Tape<B>, x: Var, name: &str) -> Result<Var> {
        let a = self.conv_norm(t, x, &format!("{name}.conv1"), 1, 1, None)?;
        let b = self.conv_norm(t, x, &format!("{name}.conv2"), 1, 1, None)?;
        let s = t.add(a, b)?;
        Ok(t.silu(s))
    }

    /// CSPLayer, expansion 1 (no conv3): conv2(x) + VGG blocks over conv1(x).
    fn csp(&self, t: &mut Tape<B>, x: Var, name: &str) -> Result<Var> {
        let silu = Some(Unary::Silu);
        let x2 = self.conv_norm(t, x, &format!("{name}.conv2"), 1, 1, silu)?;
        let mut x1 = self.conv_norm(t, x, &format!("{name}.conv1"), 1, 1, silu)?;
        for i in 0..self.cfg.elan_n {
            x1 = self.vgg(t, x1, &format!("{name}.bottlenecks.{i}"))?;
        }
        t.add(x1, x2)
    }

    /// RepNCSPELAN4: cv1 → split in half → cv2 on the second half → cv3 on that → cat all four
    /// → cv4.
    fn elan(&self, t: &mut Tape<B>, x: Var, name: &str) -> Result<Var> {
        let silu = Some(Unary::Silu);
        let y = self.conv_norm(t, x, &format!("{name}.cv1"), 1, 1, silu)?;
        let c = t.shape(y)[1] / 2;
        let (y0, y1) = (t.slice(y, 1, 0, c)?, t.slice(y, 1, c, 2 * c)?);
        let a = self.csp(t, y1, &format!("{name}.cv2.0"))?;
        let a = self.conv_norm(t, a, &format!("{name}.cv2.1"), 1, 1, silu)?;
        let b = self.csp(t, a, &format!("{name}.cv3.0"))?;
        let b = self.conv_norm(t, b, &format!("{name}.cv3.1"), 1, 1, silu)?;
        let cat = t.concat(&[y0, y1, a, b], 1)?;
        self.conv_norm(t, cat, &format!("{name}.cv4"), 1, 1, silu)
    }

    /// The post-norm transformer layer on one flattened level, sin-cos positions on q and k.
    fn encoder_layer(&self, t: &mut Tape<B>, x: Var, name: &str) -> Result<Var> {
        let d = self.cfg.hidden;
        let s = t.shape(x).to_vec();
        let (b, h, w) = (s[0], s[2], s[3]);
        let src = t.reshape(x, &[b, d, h * w])?;
        let src = t.permute(src, &[0, 2, 1])?;
        let pos = t.input(&sincos(w, h, d, 10000.0), &[1, h * w, d]);
        let q = t.add(src, pos)?;
        let a = self.mha(t, q, q, src, &format!("{name}.self_attn"), None)?;
        let y = t.add(src, a)?;
        let y = self.layer_norm(t, y, &format!("{name}.norm1"))?;
        let f = self.linear(t, y, &format!("{name}.linear1"))?;
        let f = t.gelu(f);
        let f = self.linear(t, f, &format!("{name}.linear2"))?;
        let y = t.add(y, f)?;
        let y = self.layer_norm(t, y, &format!("{name}.norm2"))?;
        let y = t.permute(y, &[0, 2, 1])?;
        t.reshape(y, &[b, d, h, w])
    }

    /// HybridEncoder (version 'dfine'): input projections, the transformer on the chosen
    /// levels, top-down FPN, bottom-up PAN. Returns one map per level, finest first.
    pub fn encoder(&self, t: &mut Tape<B>, feats: &[Var]) -> Result<Vec<Var>> {
        let l = self.cfg.in_channels.len();
        ensure!(feats.len() == l, "encoder wants {l} levels, got {}", feats.len());
        let mut proj = vec![];
        for (i, &f) in feats.iter().enumerate() {
            proj.push(self.conv_norm(t, f, &format!("encoder.input_proj.{i}"), 1, 1, None)?);
        }
        for (i, &lvl) in self.cfg.use_encoder_idx.iter().enumerate() {
            proj[lvl] = self.encoder_layer(t, proj[lvl], &format!("encoder.encoder.{i}.layers.0"))?;
        }
        // top-down
        let mut inner = vec![proj[l - 1]];
        for idx in (1..l).rev() {
            let j = l - 1 - idx;
            let hi = self.conv_norm(t, inner[0], &format!("encoder.lateral_convs.{j}"), 1, 1, None)?;
            inner[0] = hi;
            let up = t.upsample_nearest(hi, 2, 2)?;
            let c = t.concat(&[up, proj[idx - 1]], 1)?;
            let o = self.elan(t, c, &format!("encoder.fpn_blocks.{j}"))?;
            inner.insert(0, o);
        }
        // bottom-up
        let mut outs = vec![inner[0]];
        for idx in 0..l - 1 {
            let d = format!("encoder.downsample_convs.{idx}.0");
            let x = self.conv_norm(t, *outs.last().unwrap(), &format!("{d}.cv1"), 1, 1, None)?;
            let x = self.conv_norm(t, x, &format!("{d}.cv2"), 2, self.cfg.hidden, None)?;
            let c = t.concat(&[x, inner[idx + 1]], 1)?;
            outs.push(self.elan(t, c, &format!("encoder.pan_blocks.{idx}"))?);
        }
        Ok(outs)
    }

    // ------------------------------------------------------------- decoder ---

    /// MLP of `n` Linear layers, ReLU between (the production config sets `mlp_act: relu`).
    fn mlp(&self, t: &mut Tape<B>, mut x: Var, name: &str, n: usize) -> Result<Var> {
        for i in 0..n {
            x = self.linear(t, x, &format!("{name}.layers.{i}"))?;
            if i + 1 < n {
                x = t.relu(x);
            }
        }
        Ok(x)
    }

    /// D-FINE's MSDeformableAttention: no value or output projection — the memory is split into
    /// heads as it is — and offsets scaled by 1/points · ref_wh · 0.5.
    fn msda(&self, t: &mut Tape<B>, query: Var, refb: Var, memory: Var, shapes: &[(usize, usize)], name: &str) -> Result<Var> {
        let (d, h) = (self.cfg.hidden, self.cfg.heads);
        let hd = d / h;
        let np: usize = self.cfg.points.iter().sum();
        let qs = t.shape(query).to_vec();
        let (b, lq) = (qs[0], qs[1]);
        let lv = t.shape(memory)[1];
        let value = t.reshape(memory, &[b, lv, h, hd])?;
        let value = t.permute(value, &[0, 2, 3, 1])?; // [b, h, hd, lv]
        let value = t.reshape(value, &[b * h, hd, lv])?;
        let off = self.linear(t, query, &format!("{name}.sampling_offsets"))?;
        let off = t.reshape(off, &[b, lq, h, np, 2])?;
        let aw = self.linear(t, query, &format!("{name}.attention_weights"))?;
        let aw = t.reshape(aw, &[b, lq, h, np])?;
        let aw = t.softmax(aw);
        let scale: Vec<f32> = self.cfg.points.iter().flat_map(|&n| std::iter::repeat(1.0 / n as f32).take(n)).collect();
        let nps = t.input(&scale, &[np, 1]);
        let xy = t.slice(refb, 2, 0, 2)?;
        let xy = t.reshape(xy, &[b, lq, 1, 1, 2])?;
        let wh = t.slice(refb, 2, 2, 4)?;
        let wh = t.reshape(wh, &[b, lq, 1, 1, 2])?;
        let o = t.mul(off, nps)?;
        let o = t.mul(o, wh)?;
        let o = t.scale(o, 0.5)?;
        let loc = t.add(xy, o)?;
        let g = t.scale(loc, 2.0)?;
        let g = t.add_scalar(g, -1.0)?;
        let g = t.permute(g, &[0, 2, 1, 3, 4])?;
        let g = t.reshape(g, &[b * h, lq, np, 2])?;
        let mut samples = vec![];
        let (mut start, mut p0) = (0, 0);
        for (lvl, &(fh, fw)) in shapes.iter().enumerate() {
            let vl = t.slice(value, 2, start, start + fh * fw)?;
            let vl = t.reshape(vl, &[b * h, hd, fh, fw])?;
            let gl = t.slice(g, 2, p0, p0 + self.cfg.points[lvl])?;
            samples.push(t.grid_sample(vl, gl)?); // [b·h, hd, lq, p]
            start += fh * fw;
            p0 += self.cfg.points[lvl];
        }
        let sv = t.concat(&samples, 3)?;
        let aw = t.permute(aw, &[0, 2, 1, 3])?;
        let aw = t.reshape(aw, &[b * h, 1, lq, np])?;
        let wv = t.mul(sv, aw)?;
        let o = t.sum_axis(wv, 3)?; // [b·h, hd, lq]
        let o = t.reshape(o, &[b, d, lq])?;
        t.permute(o, &[0, 2, 1])
    }

    /// Gate: sigmoid gates from [x1, x2] mix the self-attended and the cross-attended query,
    /// then LayerNorm (replaces RT-DETR's residual + norm2).
    fn gate(&self, t: &mut Tape<B>, x1: Var, x2: Var, name: &str) -> Result<Var> {
        let d = self.cfg.hidden;
        let cat = t.concat(&[x1, x2], 2)?;
        let g = self.linear(t, cat, &format!("{name}.gate"))?;
        let g = t.sigmoid(g);
        let (g1, g2) = (t.slice(g, 2, 0, d)?, t.slice(g, 2, d, 2 * d)?);
        let a = t.mul(g1, x1)?;
        let b = t.mul(g2, x2)?;
        let y = t.add(a, b)?;
        self.layer_norm(t, y, &format!("{name}.norm"))
    }

    /// Softmax over each edge's bins: [B, Q, 4·(R+1)] → [B, Q, 4, R+1].
    fn edge_probs(&self, t: &mut Tape<B>, corners: Var) -> Result<Var> {
        let s = t.shape(corners).to_vec();
        let x = t.reshape(corners, &[s[0], s[1], 4, self.cfg.reg_max + 1])?;
        Ok(t.softmax(x))
    }

    /// Integral: Σ Pr(n)·W(n) per edge → [B, Q, 4].
    fn integral(&self, t: &mut Tape<B>, probs: Var) -> Result<Var> {
        let s = t.shape(probs).to_vec();
        let w = weighting_function(self.cfg.reg_max, self.cfg.up, self.cfg.reg_scale);
        let w = t.input(&w, &[1, self.cfg.reg_max + 1]);
        let y = t.linear(probs, w, None)?;
        t.reshape(y, &[s[0], s[1], 4])
    }

    /// distance2bbox: edge distances (in units of ref_wh / reg_scale, offset by reg_scale/2)
    /// around the reference centre → cxcywh.
    fn distance2bbox(&self, t: &mut Tape<B>, points: Var, dist: Var) -> Result<Var> {
        let rs = self.cfg.reg_scale.abs();
        let (pxy, pwh) = (t.slice(points, 2, 0, 2)?, t.slice(points, 2, 2, 4)?);
        let unit = t.scale(pwh, 1.0 / rs)?;
        let (lt, rb) = (t.slice(dist, 2, 0, 2)?, t.slice(dist, 2, 2, 4)?);
        let lt = t.add_scalar(lt, 0.5 * rs)?;
        let lt = t.mul(lt, unit)?;
        let rb = t.add_scalar(rb, 0.5 * rs)?;
        let rb = t.mul(rb, unit)?;
        let p1 = t.sub(pxy, lt)?;
        let p2 = t.add(pxy, rb)?;
        let c = t.add(p1, p2)?;
        let c = t.scale(c, 0.5)?;
        let wh = t.sub(p2, p1)?;
        t.concat(&[c, wh], 2)
    }

    /// LQE: the top-4 probabilities of each edge and their mean → MLP → added to the class logits.
    fn lqe(&self, t: &mut Tape<B>, scores: Var, probs: Var, name: &str) -> Result<Var> {
        let s = t.shape(probs).to_vec();
        let (top, _) = t.topk_last(probs, 4)?; // [B, Q, 4, 4]
        let m = t.sum_axis(top, 3)?;
        let m = t.scale(m, 0.25)?;
        let m = t.reshape(m, &[s[0], s[1], 4, 1])?;
        let stat = t.concat(&[top, m], 3)?;
        let stat = t.reshape(stat, &[s[0], s[1], 20])?;
        let q = self.mlp(t, stat, &format!("{name}.reg_conf"), 2)?;
        t.add(scores, q)
    }

    /// One TransformerDecoderLayer: self-attention (post-norm), deformable cross-attention through
    /// the gate, ReLU FFN (post-norm).
    fn decoder_layer(&self, t: &mut Tape<B>, x: Var, qpos: Var, refb: Var, memory: Var, shapes: &[(usize, usize)], mask: Option<Var>, name: &str) -> Result<Var> {
        let q = t.add(x, qpos)?;
        let sa = self.mha(t, q, q, x, &format!("{name}.self_attn"), mask)?;
        let y = t.add(x, sa)?;
        let y = self.layer_norm(t, y, &format!("{name}.norm1"))?;
        let q = t.add(y, qpos)?;
        let ca = self.msda(t, q, refb, memory, shapes, &format!("{name}.cross_attn"))?;
        let y = self.gate(t, y, ca, &format!("{name}.gateway"))?;
        let f = self.linear(t, y, &format!("{name}.linear1"))?;
        let f = t.relu(f);
        let f = self.linear(t, f, &format!("{name}.linear2"))?;
        let y = t.add(y, f)?;
        let y = t.clamp(y, -65504.0, 65504.0)?;
        self.layer_norm(t, y, &format!("{name}.norm3"))
    }

    /// DFINETransformer. `feats`: the encoder's maps (hidden channels, finest first). `dn`:
    /// training-time denoising queries (content [B, D, hidden], boxes-unact [B, D, 4], additive
    /// attention mask [D+Q, D+Q]); their outputs are split off into `DfineOutputs::dn`.
    pub fn decoder(&self, t: &mut Tape<B>, feats: &[Var], dn: Option<(Var, Var, Var)>) -> Result<DfineOutputs> {
        let d = self.cfg.hidden;
        let (mut flat, mut shapes) = (vec![], vec![]);
        for &f in feats {
            let s = t.shape(f).to_vec();
            // feat_channels == hidden_dim: input_proj is the identity
            ensure!(s[1] == d, "decoder input has {} channels, want {d}", s[1]);
            shapes.push((s[2], s[3]));
            let p = t.reshape(f, &[s[0], d, s[2] * s[3]])?;
            flat.push(t.permute(p, &[0, 2, 1])?);
        }
        let memory = t.concat(&flat, 1)?;
        let (b, n) = (t.shape(memory)[0], t.shape(memory)[1]);
        let (anc, valid) = anchors(&shapes, 1e-2);
        let valid = t.input(&valid, &[1, n, 1]);
        let anc = t.input(&anc, &[1, n, 4]);
        let mem_v = t.mul(memory, valid)?;
        let om = self.linear(t, mem_v, "decoder.enc_output.proj")?;
        let om = self.layer_norm(t, om, "decoder.enc_output.norm")?;
        let enc_logits = self.linear(t, om, "decoder.enc_score_head")?;
        let k = self.cfg.num_queries;
        let c = t.shape(enc_logits)[2];
        let idx = topk_by_max(&t.value(enc_logits), b, n, c, k);
        let top_mem = t.gather_rows(om, &idx, k)?;
        let anc = anc_b(t, anc, b)?;
        let top_anc = t.gather_rows(anc, &idx, k)?;
        let coord = self.mlp(t, top_mem, "decoder.enc_bbox_head", 3)?;
        let coord = t.add(coord, top_anc)?;
        let (enc_l, enc_b) = if self.train { (Some(t.gather_rows(enc_logits, &idx, k)?), Some(t.sigmoid(coord))) } else { (None, None) };
        let mut target = t.detach(top_mem);
        let mut ref_unact = t.detach(coord);
        let (mut attn_mask, mut num_dn) = (None, 0);
        if let Some((dn_c, dn_b, m)) = dn {
            num_dn = t.shape(dn_c)[1];
            target = t.concat(&[dn_c, target], 1)?;
            ref_unact = t.concat(&[dn_b, ref_unact], 1)?;
            attn_mask = Some(m);
        }

        let mut ref_detach = t.sigmoid(ref_unact);
        let mut out = target;
        let (mut out_detach, mut prev_corners): (Option<Var>, Option<Var>) = (None, None);
        let mut ref_initial = ref_detach;
        let (mut pre_logits, mut pre_boxes) = (None, None);
        let (mut logits, mut boxes, mut corners, mut refs) = (vec![], vec![], vec![], vec![]);
        let last = self.cfg.dec_layers - 1;
        for i in 0..self.cfg.dec_layers {
            let qpos = self.mlp(t, ref_detach, "decoder.query_pos_head", 2)?;
            let qpos = t.clamp(qpos, -10.0, 10.0)?;
            out = self.decoder_layer(t, out, qpos, ref_detach, memory, &shapes, attn_mask, &format!("decoder.decoder.layers.{i}"))?;
            if i == 0 {
                let delta = self.mlp(t, out, "decoder.pre_bbox_head", 3)?;
                let inv = t.inverse_sigmoid(ref_detach, 1e-5)?;
                let s = t.add(delta, inv)?;
                let pb = t.sigmoid(s);
                pre_logits = Some(self.linear(t, out, "decoder.dec_score_head.0")?);
                pre_boxes = Some(pb);
                ref_initial = t.detach(pb);
            }
            let head_in = match out_detach {
                Some(od) => t.add(out, od)?,
                None => out,
            };
            let mut pc = self.mlp(t, head_in, &format!("decoder.dec_bbox_head.{i}"), 3)?;
            if let Some(prev) = prev_corners {
                pc = t.add(pc, prev)?;
            }
            let probs = self.edge_probs(t, pc)?;
            let dist = self.integral(t, probs)?;
            let inter = self.distance2bbox(t, ref_initial, dist)?;
            if self.train || i == last {
                let sc = self.linear(t, out, &format!("decoder.dec_score_head.{i}"))?;
                let sc = self.lqe(t, sc, probs, &format!("decoder.decoder.lqe_layers.{i}"))?;
                logits.push(sc);
                boxes.push(inter);
                corners.push(pc);
                refs.push(ref_initial);
            }
            prev_corners = Some(pc);
            ref_detach = t.detach(inter);
            out_detach = Some(t.detach(out));
        }
        let all = HeadOutputs { logits, boxes, corners, refs, pre_logits: pre_logits.unwrap(), pre_boxes: pre_boxes.unwrap() };
        if num_dn == 0 {
            return Ok(DfineOutputs { main: all, dn: None, enc_logits: enc_l, enc_boxes: enc_b });
        }
        let q = num_dn + k;
        let mut split = |v: Var| -> Result<(Var, Var)> { Ok((t.slice(v, 1, 0, num_dn)?, t.slice(v, 1, num_dn, q)?)) };
        let mut dn_h = HeadOutputs { logits: vec![], boxes: vec![], corners: vec![], refs: vec![], pre_logits: all.pre_logits, pre_boxes: all.pre_boxes };
        let mut main = HeadOutputs { logits: vec![], boxes: vec![], corners: vec![], refs: vec![], pre_logits: all.pre_logits, pre_boxes: all.pre_boxes };
        for (src, dn_v, main_v) in [(&all.logits, &mut dn_h.logits, &mut main.logits), (&all.boxes, &mut dn_h.boxes, &mut main.boxes), (&all.corners, &mut dn_h.corners, &mut main.corners), (&all.refs, &mut dn_h.refs, &mut main.refs)] {
            for &v in src {
                let (a, b) = split(v)?;
                dn_v.push(a);
                main_v.push(b);
            }
        }
        (dn_h.pre_logits, main.pre_logits) = split(all.pre_logits)?;
        (dn_h.pre_boxes, main.pre_boxes) = split(all.pre_boxes)?;
        Ok(DfineOutputs { main, dn: Some(dn_h), enc_logits: enc_l, enc_boxes: enc_b })
    }

    /// Image batch [B, 3, H, W] (RGB, 0..1) → outputs.
    pub fn forward(&self, t: &mut Tape<B>, x: Var, dn: Option<(Var, Var, Var)>) -> Result<DfineOutputs> {
        let bb = crate::models::hgnetv2::HgNetV2 { cfg: self.cfg.backbone.clone(), st: self.st, prefix: "backbone.".into(), train: self.train };
        let f = bb.forward(t, x)?;
        let e = self.encoder(t, &f)?;
        self.decoder(t, &e, dn)
    }
}

/// The [1, N, 4] anchors repeated over the batch (gather_rows indexes per image).
fn anc_b<B: Backend>(t: &mut Tape<B>, anc: Var, b: usize) -> Result<Var> {
    if b == 1 {
        return Ok(anc);
    }
    let copies = vec![anc; b];
    t.concat(&copies, 0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn python_rounding_of_the_encoder_widths() {
        assert_eq!(super::py_round(1.5), 2);
        assert_eq!(super::py_round(2.5), 2);
        assert_eq!(super::py_round(16.0), 16);
        let c = super::DfineConfig::ojas_n32();
        assert_eq!((c.elan_c4, c.elan_n), (16, 2));
    }

    #[test]
    fn weighting_function_matches_dfine() {
        // weighting_function(32, 0.5, 4): 33 values, ±4 at the ends, 0 in the middle, symmetric
        let w = super::weighting_function(32, 0.5, 4.0);
        assert_eq!(w.len(), 33);
        assert_eq!((w[0], w[16], w[32]), (-4.0, 0.0, 4.0));
        assert!((w[31] - 2.0).abs() < 1e-5, "{}", w[31]); // step^15 − 1 = 3 − 1
        for i in 0..33 {
            assert!((w[i] + w[32 - i]).abs() < 1e-6);
        }
    }
}
