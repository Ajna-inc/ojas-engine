//! Minimal safetensors reader (header JSON + raw little-endian data). Enough
//! to load reference checkpoints (Mimi codec) directly without a GGUF
//! conversion pass: F32 and BF16 tensors → Vec<f32>.

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub struct SafeTensors {
    data: Vec<u8>,
    /// name → (dtype, shape, byte range)
    index: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

impl SafeTensors {
    pub fn open(path: &str) -> Result<SafeTensors> {
        let data = std::fs::read(path)?;
        let n = u64::from_le_bytes(data[..8].try_into()?) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&data[8..8 + n])?;
        let base = 8 + n;
        let mut index = HashMap::new();
        for (k, v) in hdr.as_object().ok_or_else(|| anyhow!("bad header"))? {
            if k == "__metadata__" { continue; }
            let dtype = v["dtype"].as_str().unwrap_or("").to_string();
            let shape: Vec<usize> = v["shape"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect()).unwrap_or_default();
            let offs = v["data_offsets"].as_array().ok_or_else(|| anyhow!("{k}: no offsets"))?;
            let (s, e) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
            index.insert(k.clone(), (dtype, shape, base + s, base + e));
        }
        Ok(SafeTensors { data, index })
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        self.index.get(name).map(|(_, s, _, _)| s.as_slice()).ok_or_else(|| anyhow!("tensor {name} not found"))
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    /// Tensor as f32 (F32 pass-through, BF16 widened).
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, _shape, s, e) = self.index.get(name).ok_or_else(|| anyhow!("tensor {name} not found"))?;
        let b = &self.data[*s..*e];
        Ok(match dtype.as_str() {
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "BF16" => b.chunks_exact(2).map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)).collect(),
            "F16" => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            t => bail!("{name}: unsupported dtype {t}"),
        })
    }
}

/// Load a (possibly sharded) HF checkpoint directory: every tensor whose name
/// passes `want` → (shape, f32 data). Shards are opened one at a time so peak
/// memory is one shard + the harvested tensors. Falls back to a single
/// `model.safetensors` when there is no index.json.
pub fn load_dir_f32(
    dir: &str,
    want: impl Fn(&str) -> bool,
) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>> {
    let idx_path = format!("{dir}/model.safetensors.index.json");
    let shards: Vec<String> = if let Ok(bytes) = std::fs::read(&idx_path) {
        let idx: serde_json::Value = serde_json::from_slice(&bytes)?;
        let map = idx["weight_map"].as_object().ok_or_else(|| anyhow!("bad index.json"))?;
        let mut files: Vec<String> =
            map.values().filter_map(|v| v.as_str().map(String::from)).collect();
        files.sort();
        files.dedup();
        files
    } else {
        vec!["model.safetensors".to_string()]
    };
    let mut out = HashMap::new();
    for f in &shards {
        let st = SafeTensors::open(&format!("{dir}/{f}"))?;
        let names: Vec<String> = st.names().filter(|n| want(n)).cloned().collect();
        for n in names {
            let shape = st.shape(&n)?.to_vec();
            out.insert(n.clone(), (shape, st.f32(&n)?));
        }
    }
    Ok(out)
}

// ============================================================================
// Multi-shard reader for HF checkpoints (`model.safetensors.index.json`)
// ============================================================================
//
// The helper above reads one file fully into RAM and only widens floats. Modern
// open-weight checkpoints (gpt-oss and friends) are sharded across several
// multi-GB `.safetensors` files indexed by `model.safetensors.index.json`, and
// carry non-float dtypes — `U8` for packed 4-bit weights, `F8_E4M3`/`F8_E5M2`
// for FP8 scales/weights, alongside `BF16`/`F16`/`F32`. `SafeIndex` parses the
// index (or a single `model.safetensors`), records every tensor's dtype, shape
// and byte range without reading the data, and reads a tensor's raw bytes on
// demand by seeking into its shard — so peak memory is one tensor, not one
// shard. `config.json` alongside is exposed for arch detection.

/// One tensor's location, as recorded from a shard header. `begin`/`end` are
/// byte offsets into the shard's data section, i.e. already past the header.
#[derive(Clone, Debug)]
pub struct STensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Shard file name (relative to the checkpoint dir).
    pub shard: String,
    pub begin: usize,
    pub end: usize,
}

impl STensor {
    /// Number of elements (product of the shape; scalars are 1).
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(if self.shape.is_empty() { 1 } else { 0 })
    }
    /// Size in bytes of the on-disk tensor.
    pub fn nbytes(&self) -> usize {
        self.end - self.begin
    }
}

/// Bytes per element for a safetensors dtype, or `None` for unknown/bool-packed.
pub fn dtype_size(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "BOOL" => 1,
        _ => return None,
    })
}

/// A parsed (possibly sharded) safetensors checkpoint directory.
pub struct SafeIndex {
    dir: PathBuf,
    /// tensor name -> location.
    tensors: HashMap<String, STensor>,
    /// `__metadata__` from the shard headers / index (merged).
    metadata: HashMap<String, String>,
    /// shard file name -> data-section base offset (`8 + header_len`).
    shard_base: HashMap<String, usize>,
}

impl SafeIndex {
    /// Open a checkpoint directory. Uses `model.safetensors.index.json` when
    /// present (the sharded layout), else a single `model.safetensors`.
    pub fn open(dir: impl AsRef<Path>) -> Result<SafeIndex> {
        let dir = dir.as_ref().to_path_buf();
        let idx_path = dir.join("model.safetensors.index.json");
        let shards: Vec<String> = if let Ok(bytes) = std::fs::read(&idx_path) {
            let idx: serde_json::Value = serde_json::from_slice(&bytes)?;
            let map = idx
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| anyhow!("index.json: no weight_map object"))?;
            let mut files: Vec<String> =
                map.values().filter_map(|v| v.as_str().map(String::from)).collect();
            files.sort();
            files.dedup();
            if files.is_empty() {
                bail!("index.json: weight_map is empty");
            }
            files
        } else {
            let single = dir.join("model.safetensors");
            if !single.exists() {
                bail!("no model.safetensors.index.json and no model.safetensors in {}", dir.display());
            }
            vec!["model.safetensors".to_string()]
        };

        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        let mut shard_base = HashMap::new();
        for shard in &shards {
            let (base, hdr) = read_header(&dir.join(shard))?;
            shard_base.insert(shard.clone(), base);
            let obj = hdr.as_object().ok_or_else(|| anyhow!("{shard}: header not an object"))?;
            for (k, v) in obj {
                if k == "__metadata__" {
                    if let Some(m) = v.as_object() {
                        for (mk, mv) in m {
                            if let Some(s) = mv.as_str() {
                                metadata.insert(mk.clone(), s.to_string());
                            }
                        }
                    }
                    continue;
                }
                let dtype = v["dtype"].as_str().unwrap_or("").to_string();
                let shape: Vec<usize> = v["shape"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
                    .unwrap_or_default();
                let offs = v["data_offsets"]
                    .as_array()
                    .ok_or_else(|| anyhow!("{shard}:{k}: no data_offsets"))?;
                // A malformed header must be a clean error, not a panic on `offs[1]`
                // or a `begin > end` underflow that `nbytes()` turns into a huge
                // `vec!` alloc. Validate the pair before trusting it.
                if offs.len() != 2 {
                    bail!("{shard}:{k}: data_offsets must have exactly 2 elements, got {}", offs.len());
                }
                let begin = offs[0].as_u64().ok_or_else(|| anyhow!("{shard}:{k}: bad begin offset"))? as usize;
                let end = offs[1].as_u64().ok_or_else(|| anyhow!("{shard}:{k}: bad end offset"))? as usize;
                if begin > end {
                    bail!("{shard}:{k}: data_offsets begin {begin} > end {end}");
                }
                let t = STensor { dtype, shape, shard: shard.clone(), begin, end };
                // Cross-check the byte span against dtype × element count when the
                // dtype is known — catches a header whose offsets disagree with the
                // declared shape before a later read walks off the tensor.
                if let Some(esz) = dtype_size(&t.dtype) {
                    let expect = t.numel() * esz;
                    if t.nbytes() != expect {
                        bail!(
                            "{shard}:{}: byte span {} ≠ numel {} × {} = {expect}",
                            k, t.nbytes(), t.numel(), esz
                        );
                    }
                }
                tensors.insert(k.clone(), t);
            }
        }
        Ok(SafeIndex { dir, tensors, metadata, shard_base })
    }

    /// Checkpoint directory.
    pub fn dir(&self) -> &Path { &self.dir }

    /// Every tensor name.
    pub fn names(&self) -> impl Iterator<Item = &String> { self.tensors.keys() }

    /// Number of tensors.
    pub fn len(&self) -> usize { self.tensors.len() }
    pub fn is_empty(&self) -> bool { self.tensors.is_empty() }

    /// Tensor location, if present.
    pub fn get(&self, name: &str) -> Option<&STensor> { self.tensors.get(name) }

    pub fn contains(&self, name: &str) -> bool { self.tensors.contains_key(name) }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice()).ok_or_else(|| anyhow!("{name} not found"))
    }
    pub fn dtype(&self, name: &str) -> Result<&str> {
        self.tensors.get(name).map(|t| t.dtype.as_str()).ok_or_else(|| anyhow!("{name} not found"))
    }

    /// Merged `__metadata__` entries.
    pub fn metadata(&self) -> &HashMap<String, String> { &self.metadata }

    /// Raw on-disk bytes of a tensor, read by seeking into its shard.
    pub fn read_raw(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.tensors.get(name).ok_or_else(|| anyhow!("{name} not found"))?;
        let base = *self
            .shard_base
            .get(&t.shard)
            .ok_or_else(|| anyhow!("{name}: shard {} not indexed", t.shard))?;
        let mut f = std::fs::File::open(self.dir.join(&t.shard))?;
        f.seek(SeekFrom::Start((base + t.begin) as u64))?;
        let mut buf = vec![0u8; t.nbytes()];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Tensor widened to f32. Handles the float dtypes; errors for the packed /
    /// integer dtypes, which have no single f32 interpretation (use `read_raw`
    /// and decode with the format-specific path, e.g. `crate::mxfp4`).
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let dtype = self.dtype(name)?.to_string();
        let b = self.read_raw(name)?;
        Ok(match dtype.as_str() {
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "F16" => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            "BF16" => b.chunks_exact(2).map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)).collect(),
            t => bail!("{name}: read_f32 does not handle dtype {t}"),
        })
    }

    /// Parse `config.json` from the checkpoint dir (the HF sidecar).
    pub fn config(&self) -> Result<serde_json::Value> {
        let bytes = std::fs::read(self.dir.join("config.json"))
            .map_err(|e| anyhow!("config.json: {e}"))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Parse the common `config.json` fields used for arch detection. Keeps
    /// `serde_json` inside this crate so backend crates need not depend on it.
    pub fn hf_config(&self) -> Result<HfConfig> {
        Ok(HfConfig::from_value(&self.config()?))
    }
}

/// A handful of common HF `config.json` fields, parsed once for arch detection.
/// Zero for anything the file does not carry — callers check what they need.
#[derive(Clone, Debug, Default)]
pub struct HfConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: u64,
    pub num_hidden_layers: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub head_dim: u64,
    pub intermediate_size: u64,
    pub num_local_experts: u64,
    pub num_experts_per_tok: u64,
    pub vocab_size: u64,
    /// `quantization_config.quant_method` (e.g. "mxfp4"), lower-cased.
    pub quant_method: String,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    /// `sliding_window` / attention-sink presence hints (gpt-oss).
    pub sliding_window: u64,
}

impl HfConfig {
    pub fn from_value(v: &serde_json::Value) -> HfConfig {
        let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0);
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
        HfConfig {
            architectures: v
                .get("architectures")
                .and_then(|a| a.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            model_type: s("model_type"),
            hidden_size: u("hidden_size"),
            num_hidden_layers: u("num_hidden_layers"),
            num_attention_heads: u("num_attention_heads"),
            num_key_value_heads: u("num_key_value_heads"),
            head_dim: u("head_dim"),
            intermediate_size: u("intermediate_size"),
            num_local_experts: u("num_local_experts"),
            num_experts_per_tok: u("num_experts_per_tok"),
            vocab_size: u("vocab_size"),
            quant_method: v
                .get("quantization_config")
                .and_then(|q| q.get("quant_method"))
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_lowercase(),
            rope_theta: f("rope_theta"),
            rms_norm_eps: f("rms_norm_eps"),
            sliding_window: u("sliding_window"),
        }
    }

    /// True for a gpt-oss checkpoint (arch name or model_type).
    pub fn is_gpt_oss(&self) -> bool {
        self.model_type.contains("gpt_oss")
            || self.architectures.iter().any(|a| a.to_lowercase().contains("gptoss"))
    }
}

/// Read a safetensors file header: returns `(data_base, header_json)` where
/// `data_base = 8 + header_len`. Only the header is read, not the tensor data.
fn read_header(path: &Path) -> Result<(usize, serde_json::Value)> {
    let mut f = std::fs::File::open(path)
        .map_err(|e| anyhow!("open {}: {e}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    f.read_exact(&mut len_bytes)?;
    let n = u64::from_le_bytes(len_bytes) as usize;
    let mut hdr = vec![0u8; n];
    f.read_exact(&mut hdr)?;
    let json: serde_json::Value = serde_json::from_slice(&hdr)
        .map_err(|e| anyhow!("{}: bad header json: {e}", path.display()))?;
    Ok((8 + n, json))
}

/// Serialize tensors into safetensors bytes. Header + little-endian data, in the
/// declaration order given. Exposed so tests (and future writers) can build a
/// checkpoint without a Python dependency. `(name, dtype, shape, data)`.
pub fn serialize(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    let mut meta = serde_json::Map::new();
    let mut off = 0usize;
    for (name, dtype, shape, data) in tensors {
        let begin = off;
        off += data.len();
        meta.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, off],
            }),
        );
    }
    let hdr = serde_json::to_vec(&serde_json::Value::Object(meta)).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr);
    for (_, _, _, data) in tensors {
        out.extend_from_slice(data);
    }
    out
}

/// Write f32 tensors as a safetensors file (loadable by `safetensors.torch.load_file`).
pub fn write_f32(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, shape, data) in tensors {
        let n = data.len() * 4;
        header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [off, off + n]}));
        off += n;
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(header))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + off);
    out.extend((h.len() as u64).to_le_bytes());
    out.extend(&h);
    for (_, _, data) in tensors {
        for v in data {
            out.extend(v.to_le_bytes());
        }
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Every F32 tensor of a safetensors file (name, shape, data).
pub fn read_all_f32(path: &str) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>> {
    let st = SafeTensors::open(path)?;
    let mut names: Vec<String> = st.names().cloned().collect();
    names.sort();
    names.into_iter().map(|n| Ok((n.clone(), st.shape(&n)?.to_vec(), st.f32(&n)?))).collect()
}
