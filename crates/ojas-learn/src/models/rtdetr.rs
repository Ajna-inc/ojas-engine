//! RT-DETRv2 (PResNet-vd backbone + hybrid encoder + deformable decoder) on the
//! tape, ported from lyuwenyu/RT-DETR `rtdetrv2_pytorch` (Apache-2.0):
//! `nn/backbone/presnet.py`, `zoo/rtdetr/hybrid_encoder.py`,
//! `zoo/rtdetr/rtdetrv2_decoder.py`, `zoo/rtdetr/utils.py`. Parameter names are
//! the checkpoint's state-dict names, so a `.pth` loads without a mapping.
//!
//! Every block is a plain function over the tape.

use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, ensure, Result};

use crate::backend::Backend;
use crate::tape::{BnRunning, Param, Tape, Var};
use crate::Unary;

#[derive(Clone, Debug)]
pub struct Config {
    pub depth: usize,
    pub num_classes: usize,
    pub hidden: usize,
    pub heads: usize,
    pub num_queries: usize,
    pub dec_layers: usize,
    pub points: [usize; 3],
    pub csp_blocks: usize,
    pub csp_expansion: f32,
    /// input size the anchors / encoder positions are built for (h, w)
    pub size: [usize; 2],
}

impl Config {
    /// rtdetrv2_r18vd with IISc's UVH-26 head (15 score classes).
    pub fn r18vd(num_classes: usize) -> Self {
        Config { depth: 18, num_classes, hidden: 256, heads: 8, num_queries: 300, dec_layers: 3, points: [4, 4, 4], csp_blocks: 3, csp_expansion: 0.5, size: [640, 640] }
    }
}

/// Trainable parameters, BatchNorm running statistics and constant buffers, by state-dict name.
pub struct Store<B: Backend> {
    pub params: BTreeMap<String, Param<B>>,
    pub bns: HashMap<String, BnRunning<B>>,
    pub consts: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl<B: Backend> Store<B> {
    /// From checkpoint tensors under `prefix` (e.g. "ema.module.").
    pub fn from_tensors(be: &B, tensors: &[(String, ojas_formats::pth::PthTensor)], prefix: &str) -> Self {
        let mut by: HashMap<&str, &ojas_formats::pth::PthTensor> = HashMap::new();
        for (n, t) in tensors {
            if let Some(k) = n.strip_prefix(prefix) {
                by.insert(k, t);
            }
        }
        let mut st = Store { params: BTreeMap::new(), bns: HashMap::new(), consts: HashMap::new() };
        for (&k, t) in &by {
            if k.ends_with("num_batches_tracked") || k.ends_with("running_mean") || k.ends_with("running_var") {
                continue;
            }
            if k.ends_with("anchors") || k.ends_with("valid_mask") || k.ends_with("num_points_scale") {
                st.consts.insert(k.to_string(), (t.shape.clone(), t.data.clone()));
                continue;
            }
            st.params.insert(k.to_string(), Param::new(be, k, &t.shape, &t.data));
        }
        for (&k, t) in &by {
            if let Some(base) = k.strip_suffix(".running_mean") {
                let var = by[format!("{base}.running_var").as_str()];
                st.bns.insert(base.to_string(), BnRunning { mean: be.upload(&t.data), var: be.upload(&var.data), momentum: 0.1, eps: 1e-5 });
            }
        }
        st
    }

    /// Re-initialise the class-dependent tensors for `num_classes` (fine-tuning a COCO checkpoint
    /// on another label set), as RTDETRTransformerv2 initialises them: Linear default init with the
    /// 0.01-prior bias; denoising embedding N(0, 1) with a zero padding row.
    pub fn reset_heads(&mut self, be: &B, num_classes: usize, dec_layers: usize, hidden: usize, seed: u64) {
        let mut rng = crate::models::detr_loss::Rng::new(seed);
        let prior = -((1.0f32 - 0.01) / 0.01).ln();
        let bound = 1.0 / (hidden as f32).sqrt();
        let uni = |n: usize, rng: &mut crate::models::detr_loss::Rng| -> Vec<f32> { (0..n).map(|_| (rng.uniform() * 2.0 - 1.0) * bound).collect() };
        let mut heads = vec!["decoder.enc_score_head".to_string()];
        heads.extend((0..dec_layers).map(|i| format!("decoder.dec_score_head.{i}")));
        for h in heads {
            let w = uni(num_classes * hidden, &mut rng);
            self.params.insert(format!("{h}.weight"), Param::new(be, format!("{h}.weight"), &[num_classes, hidden], &w));
            self.params.insert(format!("{h}.bias"), Param::new(be, format!("{h}.bias"), &[num_classes], &vec![prior; num_classes]));
        }
        // N(0, 1) by Box–Muller, the padding row (index num_classes) zero
        let mut emb = vec![0.0f32; (num_classes + 1) * hidden];
        for v in emb[..num_classes * hidden].iter_mut() {
            let (u1, u2) = (rng.uniform().max(1e-7), rng.uniform());
            *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        }
        let k = "decoder.denoising_class_embed.weight".to_string();
        self.params.insert(k.clone(), Param::new(be, k, &[num_classes + 1, hidden], &emb));
    }

    /// From one of our safetensors checkpoints ("model." or "ema." prefix).
    pub fn from_safetensors(be: &B, path: &str, prefix: &str) -> anyhow::Result<Self> {
        let t: Vec<(String, ojas_formats::pth::PthTensor)> = ojas_formats::safetensors::read_all_f32(path)?
            .into_iter()
            .map(|(n, shape, data)| (n, ojas_formats::pth::PthTensor { shape, data }))
            .collect();
        let st = Self::from_tensors(be, &t, prefix);
        ensure!(!st.params.is_empty(), "no tensors under {prefix} in {path}");
        Ok(st)
    }
}

/// What a forward returns.
pub struct Outputs {
    /// per decoder layer (training: all; eval: the last): logits [B, Q, C], boxes [B, Q, 4] (cxcywh, 0..1)
    pub logits: Vec<Var>,
    pub boxes: Vec<Var>,
    /// encoder top-k proposals (training): logits, boxes (sigmoid)
    pub enc_logits: Option<Var>,
    pub enc_boxes: Option<Var>,
    /// the flattened multi-level memory the decoder attends to [B, ΣHW, hidden]
    pub memory: Var,
}

pub struct RtDetr<'s, B: Backend> {
    pub cfg: Config,
    pub st: &'s Store<B>,
    pub train: bool,
    /// experiment switches: see [`Variant`]
    pub var: Variant,
}

/// Locality and gating experiment switches (all off = the reference model).
#[derive(Clone, Debug, Default)]
pub struct Variant {
    /// block-local: decoder layers from this index on read a detached memory
    /// (Some(0): backbone + encoder learn only from the encoder head's loss)
    pub detach_memory_from: Option<usize>,
    /// block-local: each decoder layer learns only from its own head (its input is detached)
    pub detach_layers: bool,
    /// frozen blocks (parameter-name prefixes): their BatchNorms use running statistics
    pub frozen: Vec<String>,
}

impl<'s, B: Backend> RtDetr<'s, B> {
    fn p(&self, t: &mut Tape<B>, name: &str) -> Result<Var> {
        let p = self.st.params.get(name).ok_or_else(|| anyhow!("parameter {name} missing"))?;
        Ok(t.param(p))
    }

    fn konst(&self, t: &mut Tape<B>, name: &str) -> Result<Var> {
        let (s, d) = self.st.consts.get(name).ok_or_else(|| anyhow!("buffer {name} missing"))?;
        Ok(t.input(d, s))
    }

    fn linear(&self, t: &mut Tape<B>, x: Var, name: &str) -> Result<Var> {
        let (w, b) = (self.p(t, &format!("{name}.weight"))?, self.p(t, &format!("{name}.bias"))?);
        t.linear(x, w, Some(b))
    }

    fn layer_norm(&self, t: &mut Tape<B>, x: Var, name: &str) -> Result<Var> {
        let (g, b) = (self.p(t, &format!("{name}.weight"))?, self.p(t, &format!("{name}.bias"))?);
        Ok(t.layer_norm(x, g, b, 1e-5))
    }

    /// MLP of `n` Linear layers, ReLU between (DETR box / query-pos heads).
    fn mlp(&self, t: &mut Tape<B>, mut x: Var, name: &str, n: usize) -> Result<Var> {
        for i in 0..n {
            x = self.linear(t, x, &format!("{name}.layers.{i}"))?;
            if i + 1 < n {
                x = t.relu(x);
            }
        }
        Ok(x)
    }

    /// ConvNormLayer: conv (no bias, pad (k−1)/2) → BatchNorm → act.
    fn conv_norm(&self, t: &mut Tape<B>, x: Var, name: &str, stride: usize, act: Option<Unary>) -> Result<Var> {
        let w = self.p(t, &format!("{name}.conv.weight"))?;
        let k = t.shape(w)[2];
        let p = (k - 1) / 2;
        let y = t.conv2d(x, w, None, [stride; 2], [p; 4], 1)?;
        let (g, b) = (self.p(t, &format!("{name}.norm.weight"))?, self.p(t, &format!("{name}.norm.bias"))?);
        let run = self.st.bns.get(&format!("{name}.norm")).ok_or_else(|| anyhow!("BN stats {name}.norm missing"))?;
        let train_bn = self.train && !self.var.frozen.iter().any(|f| name.starts_with(f.as_str()));
        let y = t.batch_norm2d(y, g, b, run, train_bn)?;
        Ok(match act {
            Some(a) => t.unary(a, y),
            None => y,
        })
    }

    // ------------------------------------------------------------ backbone ---

    /// PResNet (BasicBlock, variant d): the stride 8 / 16 / 32 feature maps.
    pub fn backbone(&self, t: &mut Tape<B>, x: Var) -> Result<Vec<Var>> {
        self.backbone_through(t, x, 3)
    }

    /// Stop after a residual stage (1..=3), for smaller pretrained crop experts.
    pub fn backbone_through(&self, t: &mut Tape<B>, x: Var, last_stage: usize) -> Result<Vec<Var>> {
        ensure!((1..=3).contains(&last_stage), "backbone stage must be 1..=3");
        ensure!(self.cfg.depth == 18 || self.cfg.depth == 34, "PResNet-{}: BottleNeck depths not ported yet", self.cfg.depth);
        let relu = Some(Unary::Relu);
        let mut h = self.conv_norm(t, x, "backbone.conv1.conv1_1", 2, relu)?;
        h = self.conv_norm(t, h, "backbone.conv1.conv1_2", 1, relu)?;
        h = self.conv_norm(t, h, "backbone.conv1.conv1_3", 1, relu)?;
        h = t.max_pool2d(h, [3, 3], [2, 2], [1; 4], false)?;
        let blocks = if self.cfg.depth == 18 { [2, 2, 2, 2] } else { [3, 4, 6, 3] };
        let mut outs = vec![];
        for (i, &n) in blocks.iter().enumerate() {
            for j in 0..n {
                let name = format!("backbone.res_layers.{i}.blocks.{j}");
                let stride = if j == 0 && i > 0 { 2 } else { 1 };
                let a = self.conv_norm(t, h, &format!("{name}.branch2a"), stride, relu)?;
                let a = self.conv_norm(t, a, &format!("{name}.branch2b"), 1, None)?;
                let short = if j > 0 {
                    h
                } else if stride == 2 {
                    // variant d: AvgPool2d(2, 2, ceil_mode=True) then a 1×1 ConvNormLayer
                    let p = t.avg_pool2d(h, [2, 2], [2, 2], [0; 4], true, true)?;
                    self.conv_norm(t, p, &format!("{name}.short.conv"), 1, None)?
                } else {
                    self.conv_norm(t, h, &format!("{name}.short"), 1, None)?
                };
                let y = t.add(a, short)?;
                h = t.relu(y);
            }
            if i >= 1 {
                outs.push(h);
            }
            if i == last_stage { break; }
        }
        Ok(outs)
    }

    // ------------------------------------------------------------- encoder ---

    /// nn.MultiheadAttention (batch_first, packed in_proj), optional additive mask [Lq, Lk].
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

    /// HybridEncoder.build_2d_sincos_position_embedding(w, h): [w·h, d], w-major.
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

    fn csp(&self, t: &mut Tape<B>, x: Var, name: &str) -> Result<Var> {
        let silu = Some(Unary::Silu);
        let mut x1 = self.conv_norm(t, x, &format!("{name}.conv1"), 1, silu)?;
        for i in 0..self.cfg.csp_blocks {
            let b = format!("{name}.bottlenecks.{i}");
            let a = self.conv_norm(t, x1, &format!("{b}.conv1"), 1, None)?;
            let c = self.conv_norm(t, x1, &format!("{b}.conv2"), 1, None)?;
            let s = t.add(a, c)?;
            x1 = t.silu(s);
        }
        let x2 = self.conv_norm(t, x, &format!("{name}.conv2"), 1, silu)?;
        let s = t.add(x1, x2)?;
        self.conv_norm(t, s, &format!("{name}.conv3"), 1, silu)
    }

    pub fn encoder(&self, t: &mut Tape<B>, feats: &[Var]) -> Result<Vec<Var>> {
        let d = self.cfg.hidden;
        let mut proj = vec![];
        for (i, &f) in feats.iter().enumerate() {
            proj.push(self.conv_norm(t, f, &format!("encoder.input_proj.{i}"), 1, None)?);
        }
        // transformer on the stride-32 level
        let s = t.shape(proj[2]).to_vec();
        let (b, h, w) = (s[0], s[2], s[3]);
        let src = t.reshape(proj[2], &[b, d, h * w])?;
        let src = t.permute(src, &[0, 2, 1])?;
        let pos = t.input(&Self::sincos(w, h, d, 10000.0), &[1, h * w, d]);
        let name = "encoder.encoder.0.layers.0";
        let q = t.add(src, pos)?;
        let a = self.mha(t, q, q, src, &format!("{name}.self_attn"), None)?;
        let x = t.add(src, a)?;
        let x = self.layer_norm(t, x, &format!("{name}.norm1"))?;
        let f = self.linear(t, x, &format!("{name}.linear1"))?;
        let f = t.gelu(f);
        let f = self.linear(t, f, &format!("{name}.linear2"))?;
        let x = t.add(x, f)?;
        let x = self.layer_norm(t, x, &format!("{name}.norm2"))?;
        let x = t.permute(x, &[0, 2, 1])?;
        proj[2] = t.reshape(x, &[b, d, h, w])?;
        // top-down FPN
        let mut inner = vec![proj[2]];
        for idx in (1..3).rev() {
            let j = 2 - idx;
            let hi = self.conv_norm(t, inner[0], &format!("encoder.lateral_convs.{j}"), 1, Some(Unary::Silu))?;
            inner[0] = hi;
            let up = t.upsample_nearest(hi, 2, 2)?;
            let c = t.concat(&[up, proj[idx - 1]], 1)?;
            let o = self.csp(t, c, &format!("encoder.fpn_blocks.{j}"))?;
            inner.insert(0, o);
        }
        // bottom-up PAN
        let mut outs = vec![inner[0]];
        for idx in 0..2 {
            let down = self.conv_norm(t, *outs.last().unwrap(), &format!("encoder.downsample_convs.{idx}"), 2, Some(Unary::Silu))?;
            let c = t.concat(&[down, inner[idx + 1]], 1)?;
            outs.push(self.csp(t, c, &format!("encoder.pan_blocks.{idx}"))?);
        }
        Ok(outs)
    }

    // ------------------------------------------------------------- decoder ---

    /// MSDeformableAttention (v2, 4-d reference boxes), default sampling.
    fn msda(&self, t: &mut Tape<B>, query: Var, refb: Var, memory: Var, shapes: &[(usize, usize)], name: &str) -> Result<Var> {
        let (d, h) = (self.cfg.hidden, self.cfg.heads);
        let hd = d / h;
        let np: usize = self.cfg.points.iter().sum();
        let qs = t.shape(query).to_vec();
        let (b, lq) = (qs[0], qs[1]);
        let lv = t.shape(memory)[1];
        let value = self.linear(t, memory, &format!("{name}.value_proj"))?;
        let value = t.reshape(value, &[b, lv, h, hd])?;
        let value = t.permute(value, &[0, 2, 3, 1])?; // [b, h, hd, lv]
        let value = t.reshape(value, &[b * h, hd, lv])?;
        let off = self.linear(t, query, &format!("{name}.sampling_offsets"))?;
        let off = t.reshape(off, &[b, lq, h, np, 2])?;
        let aw = self.linear(t, query, &format!("{name}.attention_weights"))?;
        let aw = t.reshape(aw, &[b, lq, h, np])?;
        let aw = t.softmax(aw);
        // locations = ref_xy + off · points_scale · ref_wh · 0.5
        let nps = self.konst(t, &format!("{name}.num_points_scale"))?;
        let nps = t.reshape(nps, &[np, 1])?;
        let xy = t.slice(refb, 2, 0, 2)?;
        let xy = t.reshape(xy, &[b, lq, 1, 1, 2])?;
        let wh = t.slice(refb, 2, 2, 4)?;
        let wh = t.reshape(wh, &[b, lq, 1, 1, 2])?;
        let o = t.mul(off, nps)?;
        let o = t.mul(o, wh)?;
        let o = t.scale(o, 0.5)?;
        let loc = t.add(xy, o)?;
        // grid = 2·loc − 1, [b·h, lq, np, 2]
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
        let sv = t.concat(&samples, 3)?; // [b·h, hd, lq, np]
        let aw = t.permute(aw, &[0, 2, 1, 3])?;
        let aw = t.reshape(aw, &[b * h, 1, lq, np])?;
        let wv = t.mul(sv, aw)?;
        let o = t.sum_axis(wv, 3)?; // [b·h, hd, lq]
        let o = t.reshape(o, &[b, d, lq])?;
        let o = t.permute(o, &[0, 2, 1])?;
        self.linear(t, o, &format!("{name}.output_proj"))
    }

    /// The top-k query indices by max class logit, per image (host side).
    fn topk(&self, t: &Tape<B>, logits: Var) -> Vec<usize> {
        let s = t.shape(logits).to_vec();
        let (b, n, c) = (s[0], s[1], s[2]);
        let v = t.value(logits);
        let k = self.cfg.num_queries;
        let mut idx = Vec::with_capacity(b * k);
        for bi in 0..b {
            let mut best: Vec<(f32, usize)> = (0..n).map(|i| (v[(bi * n + i) * c..(bi * n + i + 1) * c].iter().cloned().fold(f32::NEG_INFINITY, f32::max), i)).collect();
            // torch.topk: largest first (ties: lower index first here)
            best.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            idx.extend(best[..k].iter().map(|x| x.1));
        }
        idx
    }

    /// Decoder. `dn`: training-time denoising queries (content [B, D, hidden],
    /// boxes-unact [B, D, 4], attention mask [D+Q, D+Q]); their outputs come first.
    pub fn decoder(&self, t: &mut Tape<B>, feats: &[Var], dn: Option<(Var, Var, Var)>) -> Result<Outputs> {
        let d = self.cfg.hidden;
        let mut flat = vec![];
        let mut shapes = vec![];
        for (i, &f) in feats.iter().enumerate() {
            let p = self.conv_norm(t, f, &format!("decoder.input_proj.{i}"), 1, None)?;
            let s = t.shape(p).to_vec();
            shapes.push((s[2], s[3]));
            let p = t.reshape(p, &[s[0], d, s[2] * s[3]])?;
            flat.push(t.permute(p, &[0, 2, 1])?);
        }
        let memory = t.concat(&flat, 1)?;
        let b = t.shape(memory)[0];
        let mask = self.konst(t, "decoder.valid_mask")?;
        let mem_v = t.mul(memory, mask)?;
        let om = self.linear(t, mem_v, "decoder.enc_output.proj")?;
        let om = self.layer_norm(t, om, "decoder.enc_output.norm")?;
        let enc_logits = self.linear(t, om, "decoder.enc_score_head")?;
        let coord = self.mlp(t, om, "decoder.enc_bbox_head", 3)?;
        let anchors = self.konst(t, "decoder.anchors")?;
        let coord = t.add(coord, anchors)?;
        let idx = self.topk(t, enc_logits);
        let k = self.cfg.num_queries;
        let top_coord = t.gather_rows(coord, &idx, k)?;
        let (enc_l, enc_b) = if self.train {
            (Some(t.gather_rows(enc_logits, &idx, k)?), Some(t.sigmoid(top_coord)))
        } else {
            (None, None)
        };
        let content = t.gather_rows(om, &idx, k)?;
        let mut target = t.detach(content);
        let mut ref_unact = t.detach(top_coord);
        let mut attn_mask = None;
        if let Some((dn_c, dn_b, m)) = dn {
            target = t.concat(&[dn_c, target], 1)?;
            ref_unact = t.concat(&[dn_b, ref_unact], 1)?;
            attn_mask = Some(m);
        }
        let mut ref_detach = t.sigmoid(ref_unact);
        let mut ref_points = ref_detach;
        let mut out = target;
        let (mut logits, mut boxes) = (vec![], vec![]);
        let last = self.cfg.dec_layers - 1;
        for i in 0..self.cfg.dec_layers {
            let name = format!("decoder.decoder.layers.{i}");
            let qpos = self.mlp(t, ref_detach, "decoder.query_pos_head", 2)?;
            let q = t.add(out, qpos)?;
            let sa = self.mha(t, q, q, out, &format!("{name}.self_attn"), attn_mask)?;
            let x = t.add(out, sa)?;
            let x = self.layer_norm(t, x, &format!("{name}.norm1"))?;
            let q = t.add(x, qpos)?;
            let mem = if self.var.detach_memory_from.is_some_and(|k| i >= k) { t.detach(memory) } else { memory };
            let ca = self.msda(t, q, ref_detach, mem, &shapes, &format!("{name}.cross_attn"))?;
            let x = t.add(x, ca)?;
            let x = self.layer_norm(t, x, &format!("{name}.norm2"))?;
            let f = self.linear(t, x, &format!("{name}.linear1"))?;
            let f = t.relu(f);
            let f = self.linear(t, f, &format!("{name}.linear2"))?;
            let x = t.add(x, f)?;
            out = self.layer_norm(t, x, &format!("{name}.norm3"))?;
            let delta = self.mlp(t, out, &format!("decoder.dec_bbox_head.{i}"), 3)?;
            let inv = t.inverse_sigmoid(ref_detach, 1e-5)?;
            let s = t.add(delta, inv)?;
            let inter = t.sigmoid(s);
            if self.train {
                logits.push(self.linear(t, out, &format!("decoder.dec_score_head.{i}"))?);
                if i == 0 {
                    boxes.push(inter);
                } else {
                    // v2: this layer's box through the undetached previous reference
                    let inv = t.inverse_sigmoid(ref_points, 1e-5)?;
                    let s = t.add(delta, inv)?;
                    boxes.push(t.sigmoid(s));
                }
            } else if i == last {
                logits.push(self.linear(t, out, &format!("decoder.dec_score_head.{i}"))?);
                boxes.push(inter);
            }
            ref_points = inter;
            ref_detach = t.detach(inter);
            if self.var.detach_layers {
                out = t.detach(out);
            }
        }
        let _ = b;
        Ok(Outputs { logits, boxes, enc_logits: enc_l, enc_boxes: enc_b, memory })
    }

    /// Image batch [B, 3, H, W] (RGB, 0..1) → outputs.
    pub fn forward(&self, t: &mut Tape<B>, x: Var, dn: Option<(Var, Var, Var)>) -> Result<Outputs> {
        let f = self.backbone(t, x)?;
        let e = self.encoder(t, &f)?;
        self.decoder(t, &e, dn)
    }
}
