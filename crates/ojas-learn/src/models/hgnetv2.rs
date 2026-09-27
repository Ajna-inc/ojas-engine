//! HGNetv2, the backbone of D-FINE / DEIM, on the tape — ported from
//! Peterande/D-FINE `src/nn/backbone/hgnetv2.py` (Apache-2.0), as vendored by DEIM. Parameter
//! names are the checkpoint's (`backbone.stem.stem1.conv.weight`, …), so D-FINE weights load
//! without a mapping.
//!
//! The production model uses B0 with learnable affine blocks and the squeeze-then-excite
//! aggregation ('se'); both are what the D-FINE-N checkpoint contains.

use anyhow::{anyhow, ensure, Result};

use crate::backend::Backend;
use crate::models::rtdetr::Store;
use crate::tape::{Tape, Var};

/// One stage: in, mid, out channels, blocks, downsample, light blocks, kernel, layers per block.
#[derive(Clone, Copy, Debug)]
pub struct Stage {
    pub in_ch: usize,
    pub mid_ch: usize,
    pub out_ch: usize,
    pub blocks: usize,
    pub downsample: bool,
    pub light: bool,
    pub kernel: usize,
    pub layers: usize,
}

#[derive(Clone, Debug)]
pub struct HgConfig {
    /// in, mid, out channels of the stem
    pub stem: [usize; 3],
    pub stages: [Stage; 4],
    pub use_lab: bool,
    /// which stage outputs to return (D-FINE-N: [2, 3], strides 16 and 32)
    pub return_idx: Vec<usize>,
}

impl HgConfig {
    /// `arch_configs['B0']` with learnable affine blocks.
    pub fn b0(return_idx: &[usize]) -> Self {
        let s = |in_ch, mid_ch, out_ch, blocks, downsample, light, kernel, layers| Stage { in_ch, mid_ch, out_ch, blocks, downsample, light, kernel, layers };
        HgConfig {
            stem: [3, 16, 16],
            stages: [
                s(16, 16, 64, 1, false, false, 3, 3),
                s(64, 32, 256, 1, true, false, 3, 3),
                s(256, 64, 512, 2, true, true, 5, 3),
                s(512, 128, 1024, 1, true, true, 5, 3),
            ],
            use_lab: true,
            return_idx: return_idx.to_vec(),
        }
    }

    /// Channels of the returned feature maps.
    pub fn out_channels(&self) -> Vec<usize> {
        self.return_idx.iter().map(|&i| self.stages[i].out_ch).collect()
    }
}

pub struct HgNetV2<'s, B: Backend> {
    pub cfg: HgConfig,
    pub st: &'s Store<B>,
    /// state-dict prefix, "backbone." inside a D-FINE model
    pub prefix: String,
    /// BatchNorm in training mode (D-FINE-N trains its backbone norms: freeze_norm False)
    pub train: bool,
}

impl<'s, B: Backend> HgNetV2<'s, B> {
    fn p(&self, t: &mut Tape<B>, name: &str) -> Result<Var> {
        let full = format!("{}{name}", self.prefix);
        let p = self.st.params.get(&full).ok_or_else(|| anyhow!("parameter {full} missing"))?;
        Ok(t.param(p))
    }

    /// ConvBNAct: conv (no bias, pad (k−1)/2) → BatchNorm → ReLU → learnable affine (the last two
    /// only when `act`; the affine only when the model uses it).
    fn conv_bn_act(&self, t: &mut Tape<B>, x: Var, name: &str, stride: usize, groups: usize, act: bool) -> Result<Var> {
        let w = self.p(t, &format!("{name}.conv.weight"))?;
        let k = t.shape(w)[2];
        let pad = (k - 1) / 2;
        let y = t.conv2d(x, w, None, [stride; 2], [pad; 4], groups)?;
        let (g, b) = (self.p(t, &format!("{name}.bn.weight"))?, self.p(t, &format!("{name}.bn.bias"))?);
        let bn_name = format!("{}{name}.bn", self.prefix);
        let run = self.st.bns.get(&bn_name).ok_or_else(|| anyhow!("BN stats {bn_name} missing"))?;
        let y = t.batch_norm2d(y, g, b, run, self.train)?;
        if !act {
            return Ok(y);
        }
        let y = t.relu(y);
        if !self.cfg.use_lab {
            return Ok(y);
        }
        // LearnableAffineBlock: scale · x + bias, one scalar each
        let (s, c) = (self.p(t, &format!("{name}.lab.scale"))?, self.p(t, &format!("{name}.lab.bias"))?);
        let y = t.mul(y, s)?;
        t.add(y, c)
    }

    /// StemBlock: two 2×2 convs on a zero-padded branch beside a 2×2 stride-1 max pool.
    fn stem(&self, t: &mut Tape<B>, x: Var) -> Result<Var> {
        let x = self.conv_bn_act(t, x, "stem.stem1", 2, 1, true)?;
        let x = t.pad2d(x, 0, 1, 0, 1)?;
        let x2 = self.conv_bn_act(t, x, "stem.stem2a", 1, 1, true)?;
        let x2 = t.pad2d(x2, 0, 1, 0, 1)?;
        let x2 = self.conv_bn_act(t, x2, "stem.stem2b", 1, 1, true)?;
        let x1 = t.max_pool2d(x, [2, 2], [1, 1], [0; 4], true)?;
        let x = t.concat(&[x1, x2], 1)?;
        let x = self.conv_bn_act(t, x, "stem.stem3", 2, 1, true)?;
        self.conv_bn_act(t, x, "stem.stem4", 1, 1, true)
    }

    /// HG_Block: `layers` convs in sequence, every intermediate concatenated with the input, then
    /// the 'se' aggregation (1×1 squeeze to out/2, 1×1 excite to out); residual after the first
    /// block of a stage.
    fn block(&self, t: &mut Tape<B>, x: Var, name: &str, stage: &Stage, residual: bool) -> Result<Var> {
        let mut h = x;
        let mut outs = vec![x];
        for i in 0..stage.layers {
            let l = format!("{name}.layers.{i}");
            h = if stage.light {
                // LightConvBNAct: 1×1 (no act) then a depthwise k×k
                let a = self.conv_bn_act(t, h, &format!("{l}.conv1"), 1, 1, false)?;
                self.conv_bn_act(t, a, &format!("{l}.conv2"), 1, stage.mid_ch, true)?
            } else {
                self.conv_bn_act(t, h, &l, 1, 1, true)?
            };
            outs.push(h);
        }
        let cat = t.concat(&outs, 1)?;
        let y = self.conv_bn_act(t, cat, &format!("{name}.aggregation.0"), 1, 1, true)?;
        let y = self.conv_bn_act(t, y, &format!("{name}.aggregation.1"), 1, 1, true)?;
        if residual {
            t.add(y, x)
        } else {
            Ok(y)
        }
    }

    /// The feature maps at `return_idx`.
    pub fn forward(&self, t: &mut Tape<B>, x: Var) -> Result<Vec<Var>> {
        ensure!(t.shape(x).len() == 4 && t.shape(x)[1] == self.cfg.stem[0], "HGNetv2 wants NCHW with {} channels: {:?}", self.cfg.stem[0], t.shape(x));
        let mut h = self.stem(t, x)?;
        let mut outs = vec![];
        for (i, stage) in self.cfg.stages.iter().enumerate() {
            let name = format!("stages.{i}");
            if stage.downsample {
                // depthwise 3×3 stride 2, no activation
                h = self.conv_bn_act(t, h, &format!("{name}.downsample"), 2, stage.in_ch, false)?;
            }
            for j in 0..stage.blocks {
                h = self.block(t, h, &format!("{name}.blocks.{j}"), stage, j > 0)?;
            }
            if self.cfg.return_idx.contains(&i) {
                outs.push(h);
            }
        }
        Ok(outs)
    }
}
