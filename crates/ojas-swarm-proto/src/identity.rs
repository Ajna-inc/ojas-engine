//! Model identity: what makes two peers agree they hold the same model.
//!
//! Two hashes, both BLAKE3, both carried on every message that names a model:
//!
//! * [`ArchId`] — the *layout*: architecture name, dimensions, and every tensor's
//!   name, dtype and shape. Two checkpoints of one model share it. DiLoCo refuses a
//!   delta whose `ArchId` differs: matching byte length is not the same schema.
//! * [`ModelId`] — the *content*: the `ArchId` plus every tensor's bytes. Changes when
//!   one weight byte changes. A worker refuses to serve a request for a `ModelId` it
//!   did not load, and DiLoCo refuses a delta computed from a different base.
//!
//! Tensors are hashed in name order, so the id does not depend on file order.
//! The previous `weight_sig` was a u32 FNV over names and sizes: a layout check only,
//! and 32 bits.

use serde::{Deserialize, Serialize};
use std::fmt;

const DOMAIN_ARCH: &str = "ojas-swarm/arch/v1";
const DOMAIN_MODEL: &str = "ojas-swarm/model/v1";

macro_rules! hash_id {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub [u8; 32]);

        impl $name {
            pub fn to_hex(&self) -> String {
                self.0.iter().map(|b| format!("{b:02x}")).collect()
            }
            pub fn from_hex(s: &str) -> Option<$name> {
                if s.len() != 64 || !s.is_ascii() {
                    return None;
                }
                let mut out = [0u8; 32];
                for (i, o) in out.iter_mut().enumerate() {
                    *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
                }
                Some($name(out))
            }
            /// First 8 hex digits, for logs.
            pub fn short(&self) -> String {
                self.to_hex()[..8].to_string()
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.short())
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.to_hex())
            }
        }
        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(&self.to_hex())
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<$name, D::Error> {
                let s = String::deserialize(d)?;
                $name::from_hex(&s).ok_or_else(|| serde::de::Error::custom("expected 64 hex digits"))
            }
        }
    };
}

hash_id!(ArchId);
hash_id!(ModelId);
hash_id!(BlobId);

impl BlobId {
    pub fn of(bytes: &[u8]) -> BlobId {
        BlobId(*blake3::hash(bytes).as_bytes())
    }
}

/// Builds both ids from a model's tensors. Feed every tensor once, in any order.
///
/// `dtype` is the storage type as the file names it ("f16", "q8_0", "f32", ...): a
/// re-quantised copy of the same weights is a different model.
pub struct IdentityBuilder {
    arch: String,
    dims: Vec<(String, u64)>,
    tensors: Vec<(String, String, Vec<u64>, blake3::Hash)>,
}

impl IdentityBuilder {
    pub fn new(arch: &str) -> IdentityBuilder {
        IdentityBuilder { arch: arch.to_string(), dims: Vec::new(), tensors: Vec::new() }
    }

    /// A scalar hyperparameter that defines the layout (`n_layers`, `d_model`, `vocab`,
    /// `n_heads`, ...).
    pub fn dim(mut self, key: &str, value: u64) -> IdentityBuilder {
        self.dims.push((key.to_string(), value));
        self
    }

    pub fn tensor(&mut self, name: &str, dtype: &str, shape: &[u64], bytes: &[u8]) {
        self.tensors.push((name.to_string(), dtype.to_string(), shape.to_vec(), blake3::hash(bytes)));
    }

    /// For callers that already stream large tensors through their own hasher.
    pub fn tensor_hashed(&mut self, name: &str, dtype: &str, shape: &[u64], content: [u8; 32]) {
        self.tensors.push((name.to_string(), dtype.to_string(), shape.to_vec(), blake3::Hash::from(content)));
    }

    pub fn finish(mut self) -> Identity {
        self.dims.sort();
        self.tensors.sort_by(|a, b| a.0.cmp(&b.0));
        let mut a = blake3::Hasher::new();
        put_str(&mut a, DOMAIN_ARCH);
        put_str(&mut a, &self.arch);
        put_u64(&mut a, self.dims.len() as u64);
        for (k, v) in &self.dims {
            put_str(&mut a, k);
            put_u64(&mut a, *v);
        }
        put_u64(&mut a, self.tensors.len() as u64);
        for (name, dtype, shape, _) in &self.tensors {
            put_str(&mut a, name);
            put_str(&mut a, dtype);
            put_u64(&mut a, shape.len() as u64);
            shape.iter().for_each(|d| put_u64(&mut a, *d));
        }
        let arch = ArchId(*a.finalize().as_bytes());

        let mut m = blake3::Hasher::new();
        put_str(&mut m, DOMAIN_MODEL);
        m.update(&arch.0);
        for (name, _, _, h) in &self.tensors {
            put_str(&mut m, name);
            m.update(h.as_bytes());
        }
        Identity { arch, model: ModelId(*m.finalize().as_bytes()) }
    }
}

fn put_u64(h: &mut blake3::Hasher, v: u64) {
    h.update(&v.to_le_bytes());
}

fn put_str(h: &mut blake3::Hasher, s: &str) {
    put_u64(h, s.len() as u64);
    h.update(s.as_bytes());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Identity {
    pub arch: ArchId,
    pub model: ModelId,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(order: &[usize], tweak: bool) -> Identity {
        let t = [("a.weight", vec![1u8, 2, 3, 4]), ("b.weight", vec![5, 6, 7, 8]), ("c.bias", vec![9, 9])];
        let mut b = IdentityBuilder::new("tiny").dim("n_layers", 2).dim("d_model", 8);
        for &i in order {
            let mut bytes = t[i].1.clone();
            if tweak && i == 1 {
                bytes[0] ^= 1;
            }
            b.tensor(t[i].0, "f32", &[bytes.len() as u64], &bytes);
        }
        b.finish()
    }

    #[test]
    fn order_does_not_matter_one_byte_does() {
        let x = build(&[0, 1, 2], false);
        assert_eq!(x, build(&[2, 0, 1], false));
        let y = build(&[0, 1, 2], true);
        assert_eq!(x.arch, y.arch, "same layout");
        assert_ne!(x.model, y.model, "different content");
    }

    #[test]
    fn layout_changes_change_the_arch() {
        let x = build(&[0, 1, 2], false);
        let mut b = IdentityBuilder::new("tiny").dim("n_layers", 3).dim("d_model", 8);
        b.tensor("a.weight", "f32", &[4], &[1, 2, 3, 4]);
        b.tensor("b.weight", "f32", &[4], &[5, 6, 7, 8]);
        b.tensor("c.bias", "f32", &[2], &[9, 9]);
        assert_ne!(x.arch, b.finish().arch);
        let mut b = IdentityBuilder::new("tiny").dim("n_layers", 2).dim("d_model", 8);
        b.tensor("a.weight", "f16", &[4], &[1, 2, 3, 4]);
        b.tensor("b.weight", "f32", &[4], &[5, 6, 7, 8]);
        b.tensor("c.bias", "f32", &[2], &[9, 9]);
        assert_ne!(x.arch, b.finish().arch, "dtype is layout");
    }

    #[test]
    fn hex_round_trips_through_serde() {
        let id = build(&[0, 1, 2], false).model;
        let j = serde_json::to_string(&id).unwrap();
        assert_eq!(serde_json::from_str::<ModelId>(&j).unwrap(), id);
        assert!(serde_json::from_str::<ModelId>("\"abc\"").is_err());
        assert_eq!(ModelId::from_hex(&id.to_hex()), Some(id));
    }
}
