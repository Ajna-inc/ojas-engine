//! Convolution geometry and the padded-NHWC activation layout, shared by the
//! GPU backends (CUDA, Vulkan) and the executor that plans for them.

/// Convolution geometry (groups == 1): kh×kw kernel, per-axis stride, pads
/// top/left/bottom/right (ONNX order [t, l, b, r]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvGeom {
    pub cin: usize,
    pub h: usize,
    pub w: usize,
    pub cout: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub pads: [usize; 4],
}

impl ConvGeom {
    /// k×k, stride s, pad p on every side.
    pub fn square(cin: usize, h: usize, w: usize, cout: usize, k: usize, s: usize, p: usize) -> Self {
        ConvGeom { cin, h, w, cout, kh: k, kw: k, sh: s, sw: s, pads: [p; 4] }
    }

    pub fn out_hw(&self) -> (usize, usize) {
        let [t, l, b, r] = self.pads;
        ((self.h + t + b - self.kh) / self.sh + 1, (self.w + l + r - self.kw) / self.sw + 1)
    }

    /// Effective kernel width: the smallest `kwe >= kw` with `kwe*cin % 16 == 0`
    /// (zero-weight phantom columns), or None if that needs more than 4 spare
    /// columns — the caller must widen the channel storage instead.
    pub fn kw_eff(&self) -> Option<usize> {
        (self.kw..=self.kw + 4).find(|kw| (kw * self.cin) % 16 == 0)
    }

    pub fn spare_cols(&self) -> usize {
        self.kw_eff().map_or(0, |kw| kw - self.kw)
    }

    /// Largest pad on any side: the storage border `ConvPlan::new` gives the input.
    pub fn max_pad(&self) -> usize {
        self.pads.iter().copied().max().unwrap_or(0)
    }

    /// Patch length J = kh·kwe·C (a multiple of 16).
    pub fn patch_len(&self) -> usize {
        self.kh * self.kw_eff().unwrap_or(0) * self.cin
    }
}

/// Where an NHWC image lives: `pad` zero rows/cols on every side of the h×w
/// interior plus `spare` extra zero columns on the right; `c` channels of a
/// pixel that is `cs` channels wide (`cs > c`: a channel slice of a wider
/// buffer — the slice's first channel is addressed by the dispatch offset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Storage {
    pub c: usize,
    pub cs: usize,
    pub h: usize,
    pub w: usize,
    pub pad: usize,
    pub spare: usize,
}

impl Storage {
    pub fn hp(&self) -> usize {
        self.h + 2 * self.pad
    }
    pub fn wp(&self) -> usize {
        self.w + 2 * self.pad + self.spare
    }
    /// Elements per image (of the whole buffer the slice lives in).
    pub fn img(&self) -> usize {
        self.hp() * self.wp() * self.cs
    }
}
