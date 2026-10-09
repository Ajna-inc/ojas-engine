//! Self-describing tensors: `[u32 ndim][u32 dims...][u32 dtype][data]`, little-endian.
//!
//! The layout and the tag values 0 (f32) and 1 (Q8_0) follow SwarmLLM
//! (MIT OR Apache-2.0, `src/inference/tensor_util.rs` @ b14482d), so a hidden state
//! encoded by either side decodes on the other. Tags 2 (f16) and 3 (bf16) are ours.
//!
//! The shape travels with the data. The previous ojas wire passed `d` out of band,
//! so a peer that disagreed about it desynchronised silently.
//!
//! Decoding bounds the allocation by the bytes actually present, not by a fixed
//! element cap: a 12-byte frame declaring a billion elements is refused before
//! anything is reserved, and a long prompt is never refused for being long.

use crate::quant;
use half::{bf16, f16};

pub const MAX_NDIM: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32 = 0,
    Q8_0 = 1,
    F16 = 2,
    Bf16 = 3,
}

impl DType {
    fn from_tag(t: u32) -> Option<DType> {
        Some(match t {
            0 => DType::F32,
            1 => DType::Q8_0,
            2 => DType::F16,
            3 => DType::Bf16,
            _ => return None,
        })
    }

    /// Bytes needed for `n` elements, or `None` on overflow.
    pub fn byte_len(self, n: usize) -> Option<usize> {
        match self {
            DType::F32 => n.checked_mul(4),
            DType::F16 | DType::Bf16 => n.checked_mul(2),
            DType::Q8_0 => quant::q8_0_len(n),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum TensorError {
    Truncated(&'static str),
    TooManyDims(usize),
    UnknownDType(u32),
    Overflow,
    Empty,
    /// Declared shape needs more bytes than the payload holds.
    Short { need: usize, have: usize },
    /// NaN or Inf in the decoded values: a faulted kernel or a hostile peer, never
    /// something to feed into attention or an optimizer.
    NonFinite,
    ShapeMismatch { want: Vec<usize>, got: Vec<usize> },
    Q8(String),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::Truncated(w) => write!(f, "tensor truncated in {w}"),
            TensorError::TooManyDims(n) => write!(f, "tensor ndim {n} exceeds {MAX_NDIM}"),
            TensorError::UnknownDType(t) => write!(f, "unknown tensor dtype tag {t}"),
            TensorError::Overflow => write!(f, "tensor shape overflows"),
            TensorError::Empty => write!(f, "tensor has zero elements"),
            TensorError::Short { need, have } => write!(f, "tensor needs {need} bytes, {have} present"),
            TensorError::NonFinite => write!(f, "tensor contains NaN or Inf"),
            TensorError::ShapeMismatch { want, got } => write!(f, "tensor shape {got:?}, expected {want:?}"),
            TensorError::Q8(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for TensorError {}

/// A host-side f32 tensor. The wire dtype is chosen at encode time.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Tensor { shape, data }
    }

    pub fn vector(data: Vec<f32>) -> Tensor {
        Tensor { shape: vec![data.len()], data }
    }

    pub fn encode(&self, dtype: DType) -> Vec<u8> {
        let n = self.data.len();
        let mut out = Vec::with_capacity(8 + 4 * self.shape.len() + dtype.byte_len(n).unwrap_or(0));
        out.extend_from_slice(&(self.shape.len() as u32).to_le_bytes());
        for &d in &self.shape {
            out.extend_from_slice(&(d as u32).to_le_bytes());
        }
        out.extend_from_slice(&(dtype as u32).to_le_bytes());
        match dtype {
            DType::F32 => self.data.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
            DType::F16 => self.data.iter().for_each(|v| out.extend_from_slice(&f16::from_f32(*v).to_le_bytes())),
            DType::Bf16 => self.data.iter().for_each(|v| out.extend_from_slice(&bf16::from_f32(*v).to_le_bytes())),
            DType::Q8_0 => out.extend_from_slice(&quant::quantize_q8_0(&self.data)),
        }
        out
    }

    /// Decode one tensor from the front of `bytes`; returns it and the bytes consumed.
    /// Trailing bytes are left for the caller.
    pub fn decode(bytes: &[u8]) -> Result<(Tensor, usize), TensorError> {
        let mut pos = 0usize;
        let mut u32_at = |what: &'static str| -> Result<u32, TensorError> {
            let b = bytes.get(pos..pos + 4).ok_or(TensorError::Truncated(what))?;
            pos += 4;
            Ok(u32::from_le_bytes(b.try_into().unwrap()))
        };
        let ndim = u32_at("ndim")? as usize;
        if ndim > MAX_NDIM {
            return Err(TensorError::TooManyDims(ndim));
        }
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(u32_at("shape")? as usize);
        }
        let tag = u32_at("dtype")?;
        let dtype = DType::from_tag(tag).ok_or(TensorError::UnknownDType(tag))?;
        let n = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).ok_or(TensorError::Overflow)?;
        if n == 0 {
            return Err(TensorError::Empty);
        }
        let need = dtype.byte_len(n).ok_or(TensorError::Overflow)?;
        let have = bytes.len() - pos;
        if need > have {
            return Err(TensorError::Short { need, have });
        }
        let p = &bytes[pos..pos + need];
        let data: Vec<f32> = match dtype {
            DType::F32 => p.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            DType::F16 => p.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            DType::Bf16 => p.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            DType::Q8_0 => quant::dequantize_q8_0(p, n).map_err(TensorError::Q8)?,
        };
        if data.iter().any(|v| !v.is_finite()) {
            return Err(TensorError::NonFinite);
        }
        Ok((Tensor { shape, data }, pos + need))
    }

    /// Decode and require an exact shape.
    pub fn decode_shaped(bytes: &[u8], want: &[usize]) -> Result<Tensor, TensorError> {
        let (t, _) = Tensor::decode(bytes)?;
        if t.shape != want {
            return Err(TensorError::ShapeMismatch { want: want.to_vec(), got: t.shape });
        }
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Tensor {
        Tensor::new(vec![3, 5], (0..15).map(|i| i as f32 * 0.25 - 1.5).collect())
    }

    #[test]
    fn every_dtype_round_trips_with_its_precision() {
        let t = sample();
        for (dt, tol) in [(DType::F32, 0.0), (DType::F16, 1e-3), (DType::Bf16, 1e-2), (DType::Q8_0, 2e-2)] {
            let b = t.encode(dt);
            let (d, used) = Tensor::decode(&b).unwrap();
            assert_eq!(used, b.len());
            assert_eq!(d.shape, t.shape);
            for (a, b) in t.data.iter().zip(&d.data) {
                assert!((a - b).abs() <= tol, "{dt:?}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn f32_layout_matches_the_swarmllm_layout() {
        // ndim=1, dim=2, tag=0, then two LE f32.
        let b = Tensor::vector(vec![1.0, -2.0]).encode(DType::F32);
        let mut want = vec![1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
        want.extend_from_slice(&1.0f32.to_le_bytes());
        want.extend_from_slice(&(-2.0f32).to_le_bytes());
        assert_eq!(b, want);
    }

    #[test]
    fn hostile_headers_are_refused_before_allocating() {
        // One billion elements declared, no data.
        let mut b = vec![1, 0, 0, 0];
        b.extend_from_slice(&1_000_000_000u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        assert!(matches!(Tensor::decode(&b), Err(TensorError::Short { .. })));
        let mut b = 9u32.to_le_bytes().to_vec();
        b.extend_from_slice(&[0; 40]);
        assert_eq!(Tensor::decode(&b), Err(TensorError::TooManyDims(9)));
        let mut b = Tensor::vector(vec![1.0]).encode(DType::F32);
        b[8] = 9;
        assert_eq!(Tensor::decode(&b), Err(TensorError::UnknownDType(9)));
        let mut b = vec![2, 0, 0, 0];
        b.extend_from_slice(&u32::MAX.to_le_bytes());
        b.extend_from_slice(&u32::MAX.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        #[cfg(target_pointer_width = "32")]
        assert_eq!(Tensor::decode(&b), Err(TensorError::Overflow));
        #[cfg(target_pointer_width = "64")]
        assert!(Tensor::decode(&b).is_err());
    }

    #[test]
    fn non_finite_values_and_wrong_shapes_are_refused() {
        let b = Tensor::vector(vec![1.0, f32::NAN]).encode(DType::F32);
        assert_eq!(Tensor::decode(&b), Err(TensorError::NonFinite));
        let b = sample().encode(DType::F32);
        assert!(matches!(Tensor::decode_shaped(&b, &[5, 3]), Err(TensorError::ShapeMismatch { .. })));
        assert!(Tensor::decode_shaped(&b, &[3, 5]).is_ok());
    }

    #[test]
    fn trailing_bytes_are_left_to_the_caller() {
        let mut b = sample().encode(DType::F16);
        let len = b.len();
        b.extend_from_slice(&[7, 7, 7]);
        assert_eq!(Tensor::decode(&b).unwrap().1, len);
    }
}
