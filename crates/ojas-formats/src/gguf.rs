//! Minimal GGUF reader — enough to load config + f16/f32 tensors from the model
//! files GGUF tools ship. Weights in GGUF are stored `[ne0=in, ne1=out]`
//! row-major, i.e. one contiguous input-vector per output row, which is the
//! `[N,K]` layout the decode GEMV wants.

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug, Clone)]
pub enum Meta {
    U32(u32),
    U64(u64),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    StrArr(Vec<String>),
    IntArr(Vec<i64>), // integer array (e.g. tokenizer.ggml.token_type)
    FloatArr(Vec<f32>), // float array (e.g. tokenizer.ggml.scores for SPM)
    Arr, // other array body skipped
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub rel_offset: u64,
    pub part: usize,     // which split file holds the data (0 for single-file GGUFs)
}

pub struct Gguf {
    files: Vec<File>,            // part 0 first; split GGUFs add more
    pub path: String,
    pub meta: HashMap<String, Meta>,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_offsets: Vec<u64>,  // per-part tensor-data base offset
    /// Filesystem path of each part, in `data_offsets` order. Kept explicitly
    /// rather than derived from the "-00001-of-" naming, because a part need not
    /// be a numbered split — an MTP sidecar is a separate file with its own name.
    part_paths: Vec<String>,
}

// Bound header parsing separately from weight payloads. A model can have a
// 100-GB tensor section without allowing corrupt metadata to allocate 100 GB.
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_HEADER_ITEMS: u64 = 1_048_576;
struct HeaderReader<'a> { file: &'a mut File, pos: u64, end: u64 }
impl Read for HeaderReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = out.len().min(self.end.saturating_sub(self.pos) as usize);
        let got = self.file.read(&mut out[..n])?;
        self.pos += got as u64;
        Ok(got)
    }
}
impl Seek for HeaderReader<'_> {
    fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
        let pos = match from {
            SeekFrom::Start(p) => p as i128,
            SeekFrom::Current(p) => self.pos as i128 + p as i128,
            SeekFrom::End(p) => self.end as i128 + p as i128,
        };
        if pos < 0 || pos > self.end as i128 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "GGUF header exceeds file or metadata budget"));
        }
        self.pos = self.file.seek(SeekFrom::Start(pos as u64))?;
        Ok(self.pos)
    }
    fn stream_position(&mut self) -> std::io::Result<u64> { Ok(self.pos) }
}

fn rd<const N: usize>(f: &mut HeaderReader<'_>) -> Result<[u8; N]> {
    let mut b = [0u8; N];
    f.read_exact(&mut b)?;
    Ok(b)
}
fn u32r(f: &mut HeaderReader<'_>) -> Result<u32> {
    Ok(u32::from_le_bytes(rd::<4>(f)?))
}
fn u64r(f: &mut HeaderReader<'_>) -> Result<u64> {
    Ok(u64::from_le_bytes(rd::<8>(f)?))
}
fn strr(f: &mut HeaderReader<'_>) -> Result<String> {
    let n = u64r(f)?;
    anyhow::ensure!(n <= 16 * 1024 * 1024 && n <= f.end.saturating_sub(f.pos), "GGUF string exceeds metadata bounds");
    let n = usize::try_from(n)?;
    let mut b = vec![0u8; n];
    f.read_exact(&mut b)?;
    Ok(String::from_utf8(b)?)
}

/// Read one metadata value of GGUF type `t`, advancing the file. Scalars are
/// returned; array bodies are consumed (position advanced) and reported as `Arr`.
fn read_meta(f: &mut HeaderReader<'_>, t: u32) -> Result<Meta> {
    Ok(match t {
        0 => Meta::U32(rd::<1>(f)?[0] as u32),        // u8
        1 => Meta::I32(i8::from_le_bytes(rd::<1>(f)?) as i32),
        2 => Meta::U32(u16::from_le_bytes(rd::<2>(f)?) as u32),
        3 => Meta::I32(i16::from_le_bytes(rd::<2>(f)?) as i32),
        4 => Meta::U32(u32r(f)?),
        5 => Meta::I32(i32::from_le_bytes(rd::<4>(f)?)),
        6 => Meta::F32(f32::from_le_bytes(rd::<4>(f)?)),
        7 => Meta::Bool(rd::<1>(f)?[0] != 0),
        8 => Meta::Str(strr(f)?),
        9 => {
            // array: element type + count. Keep string arrays (tokenizer vocab/
            // merges); consume others.
            let et = u32r(f)?;
            let n = u64r(f)?;
            anyhow::ensure!(et <= 12 && et != 9, "invalid or nested GGUF array type {et}");
            anyhow::ensure!(n <= MAX_HEADER_ITEMS && n <= f.end.saturating_sub(f.pos), "GGUF array exceeds metadata bounds");
            if et == 8 {
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(strr(f)?);
                }
                Meta::StrArr(v)
            } else if matches!(et, 0..=5 | 7 | 10 | 11) {
                // integer/bool array (token_type, sliding_window_pattern, ...): keep as i64.
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(match read_meta(f, et)? {
                        Meta::U32(x) => x as i64,
                        Meta::I32(x) => x as i64,
                        Meta::U64(x) => x as i64,
                        Meta::Bool(b) => b as i64,
                        _ => 0,
                    });
                }
                Meta::IntArr(v)
            } else if matches!(et, 6 | 12) {
                // float array (tokenizer scores).
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    if let Meta::F32(x) = read_meta(f, et)? { v.push(x); } else { v.push(0.0); }
                }
                Meta::FloatArr(v)
            } else {
                for _ in 0..n {
                    read_meta(f, et)?;
                }
                Meta::Arr
            }
        }
        10 => Meta::U64(u64r(f)?),
        11 => Meta::U64(u64::from_le_bytes(rd::<8>(f)?)), // i64 (fits)
        12 => Meta::F32(f64::from_le_bytes(rd::<8>(f)?) as f32),
        _ => bail!("unknown GGUF metadata type {t}"),
    })
}

/// Parse one GGUF file's header: (meta, tensors, data_offset).
fn parse_gguf(file: &mut File) -> Result<(HashMap<String, Meta>, HashMap<String, TensorInfo>, u64)> {
    let file_len = file.metadata()?.len();
    let mut input = HeaderReader { pos: file.stream_position()?, end: file_len.min(MAX_HEADER_BYTES), file };
    let file = &mut input;
    if &rd::<4>(file)? != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = u32r(file)?;
    anyhow::ensure!(matches!(version, 2 | 3), "unsupported GGUF version {version}");
    let n_tensors = u64r(file)?;
    let n_kv = u64r(file)?;
    anyhow::ensure!(n_tensors <= MAX_HEADER_ITEMS && n_kv <= MAX_HEADER_ITEMS, "GGUF header item count exceeds limit");
    let mut meta = HashMap::new();
    for _ in 0..n_kv {
        let key = strr(file)?;
        let t = u32r(file)?;
        meta.insert(key, read_meta(file, t)?);
    }
    let mut tensors = HashMap::new();
    for _ in 0..n_tensors {
        let name = strr(file)?;
        let nd = u32r(file)? as usize;
        anyhow::ensure!((1..=4).contains(&nd), "invalid GGUF tensor rank {nd}");
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(u64r(file)?);
        }
        let elements = dims.iter().try_fold(1u64, |n, &d| n.checked_mul(d));
        anyhow::ensure!(elements.is_some_and(|n| n > 0 && n <= u64::MAX / 8), "invalid or overflowing GGUF tensor dimensions");
        let ggml_type = u32r(file)?;
        let rel_offset = u64r(file)?;
        anyhow::ensure!(!tensors.contains_key(&name), "duplicate GGUF tensor {name}");
        tensors.insert(name, TensorInfo { dims, ggml_type, rel_offset, part: 0 });
    }
    // tensor data starts at the next `alignment` boundary.
    let align = match meta.get("general.alignment") {
        Some(Meta::U32(a)) => *a as u64,
        _ => 32,
    };
    let pos = file.stream_position()?;
    anyhow::ensure!(align.is_power_of_two(), "invalid GGUF alignment {align}");
    let data_offset = pos.checked_add(align - 1)
        .ok_or_else(|| anyhow!("GGUF alignment overflow"))? & !(align - 1);
    for (name, tensor) in &tensors {
        anyhow::ensure!(data_offset.checked_add(tensor.rel_offset).is_some_and(|p| p < file_len), "GGUF tensor {name} starts outside file");
    }
    Ok((meta, tensors, data_offset))
}

fn split_integer(meta: &HashMap<String, Meta>, key: &str) -> Result<Option<usize>> {
    let value = match meta.get(key) {
        None => return Ok(None),
        Some(Meta::U32(v)) => *v as u64,
        Some(Meta::U64(v)) => *v,
        Some(Meta::I32(v)) if *v >= 0 => *v as u64,
        _ => bail!("invalid GGUF {key}: expected a nonnegative integer"),
    };
    Ok(Some(usize::try_from(value).map_err(|_| anyhow!("GGUF {key} is too large"))?))
}

fn validate_split(meta: &HashMap<String, Meta>, part: usize, supplied: usize) -> Result<()> {
    let count = split_integer(meta, "split.count")?.unwrap_or(1);
    anyhow::ensure!(count > 0 && count == supplied,
        "GGUF shard {part}: split.count={count}, supplied {supplied} shards");
    let number = split_integer(meta, "split.no")?;
    anyhow::ensure!(count == 1 || number.is_some(), "GGUF shard {part}: missing split.no");
    anyhow::ensure!(number.unwrap_or(0) == part,
        "GGUF shard {part}: split.no={number:?}; supply shards in order starting with the first shard (-00001-of-)");
    Ok(())
}

fn validate_tensor_count(meta: &HashMap<String, Meta>, actual: usize) -> Result<()> {
    if let Some(expected) = split_integer(meta, "split.tensors.count")? {
        anyhow::ensure!(actual == expected,
            "incomplete GGUF: expected {expected} tensors, found {actual}");
    }
    Ok(())
}

impl Gguf {
    /// Parses a model from already-open shards, in shard order.
    ///
    /// The path-based `open` derives sibling shard names from the first
    /// file's name; that is impossible without a filesystem, so here the
    /// caller supplies every part. A sandboxed process gets its shards as
    /// descriptors -- see MappedGguf::from_files for the tensor-data half.
    ///
    /// `path` is left empty, which is how the loader tells the two modes
    /// apart.
    pub fn from_files(files: Vec<File>) -> Result<Self> {
        anyhow::ensure!(!files.is_empty(), "no GGUF shards supplied");
        let mut files = files;
        let mut tensors: HashMap<String, TensorInfo> = HashMap::new();
        let mut data_offsets = Vec::with_capacity(files.len());
        let mut meta_out: Option<HashMap<String, Meta>> = None;
        let supplied = files.len();
        for (part, f) in files.iter_mut().enumerate() {
            f.seek(SeekFrom::Start(0))?;
            let (meta, ptensors, poff) = parse_gguf(f)?;
            validate_split(&meta, part, supplied)?;
            data_offsets.push(poff);
            for (name, mut ti) in ptensors {
                anyhow::ensure!(!tensors.contains_key(&name), "duplicate tensor {name} in shard {part}");
                ti.part = part;
                tensors.insert(name, ti);
            }
            // Part 0 carries the model metadata; later parts repeat only
            // enough of it to be valid GGUF.
            if part == 0 {
                meta_out = Some(meta);
            }
        }
        validate_tensor_count(meta_out.as_ref().unwrap(), tensors.len())?;
        // fd mode has no filesystem paths; shard_paths() is only meaningful for
        // the path-based loader, and `path` being empty is how callers tell the
        // two modes apart (see is_fd_backed).
        let part_paths = vec![String::new(); files.len()];
        let model = Self {
            files,
            path: String::new(),
            meta: meta_out.unwrap_or_default(),
            tensors,
            data_offsets,
            part_paths,
        };
        model.validate_extents()?;
        Ok(model)
    }

    /// Duplicate descriptors for the shards, for a second consumer that needs
    /// its own (the mmap streamer keeps fds and sets F_NOCACHE on them).
    pub fn shard_files(&self) -> Result<Vec<File>> {
        self.files.iter().map(|f| Ok(f.try_clone()?)).collect()
    }

    /// True when this model was opened from descriptors and has no paths.
    pub fn is_fd_backed(&self) -> bool { self.path.is_empty() }

    pub fn open(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let path_owned = path.to_string();
        let (meta, mut tensors, data_offset) = parse_gguf(&mut file)?;
        let mut files = vec![file];
        let mut data_offsets = vec![data_offset];
        let mut part_paths = vec![path_owned.clone()];
        // split GGUFs: "<base>-00001-of-0000N.gguf" parts, each a complete
        // GGUF holding a subset of tensors (split.count in the metadata). Merge the
        // per-part tensor indexes; read_tensor picks the right file.
        let split_count = split_integer(&meta, "split.count")?.unwrap_or(1);
        validate_split(&meta, 0, split_count)?;
        if split_count > 1 {
            let first = std::path::Path::new(path);
            let filename = first.file_name().and_then(|s| s.to_str())
                .ok_or_else(|| anyhow!("invalid GGUF filename"))?;
            anyhow::ensure!(filename.contains("-00001-of-"),
                "split GGUF: pass the first shard (-00001-of-), not {filename}");
            for n in 2..=split_count {
                let ppath = first.with_file_name(filename.replacen("-00001-of-", &format!("-{n:05}-of-"), 1))
                    .to_string_lossy().into_owned();
                let mut pf = File::open(&ppath)
                    .map_err(|e| anyhow!("split part {ppath} missing: {e}"))?;
                let (pmeta, ptensors, poff) = parse_gguf(&mut pf)?;
                let part = files.len();
                validate_split(&pmeta, part, split_count)?;
                files.push(pf);
                data_offsets.push(poff);
                part_paths.push(ppath.clone());
                for (name, mut ti) in ptensors {
                    anyhow::ensure!(!tensors.contains_key(&name), "duplicate tensor {name} in shard {part}");
                    ti.part = part;
                    tensors.insert(name, ti);
                }
            }
            tracing::info!(target: "gguf", "split model: {} parts, {} tensors total", split_count, tensors.len());
        }
        validate_tensor_count(&meta, tensors.len())?;
        let model = Self { files, path: path_owned, meta, tensors, data_offsets, part_paths };
        model.validate_extents()?;
        Ok(model)
    }

    fn validate_extents(&self) -> Result<()> {
        let lengths = self.files.iter().map(|f| Ok(f.metadata()?.len())).collect::<Result<Vec<_>>>()?;
        for name in self.tensors.keys() {
            if let Some((part, start, bytes, _)) = self.tensor_meta(name) {
                anyhow::ensure!(start.checked_add(bytes).is_some_and(|end| end <= lengths[part]),
                    "GGUF tensor {name} payload extends beyond shard {part}");
            }
        }
        Ok(())
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        match self.meta.get(key)? {
            Meta::U32(v) => Some(*v),
            Meta::U64(v) => Some(*v as u32),
            Meta::I32(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        match self.meta.get(key)? {
            Meta::F32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn meta_bool(&self, key: &str) -> Option<bool> {
        match self.meta.get(key)? {
            Meta::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn float_arr(&self, key: &str) -> Option<&Vec<f32>> {
        match self.meta.get(key)? {
            Meta::FloatArr(v) => Some(v),
            _ => None,
        }
    }
    pub fn str_arr(&self, key: &str) -> Option<&Vec<String>> {
        match self.meta.get(key)? {
            Meta::StrArr(v) => Some(v),
            _ => None,
        }
    }
    pub fn f32_arr(&self, key: &str) -> Option<&Vec<f32>> {
        match self.meta.get(key)? {
            Meta::FloatArr(v) => Some(v),
            _ => None,
        }
    }
    pub fn int_arr(&self, key: &str) -> Option<&Vec<i64>> {
        match self.meta.get(key)? {
            Meta::IntArr(v) => Some(v),
            _ => None,
        }
    }
    pub fn arch(&self) -> String {
        match self.meta.get("general.architecture") {
            Some(Meta::Str(s)) => s.clone(),
            _ => "?".into(),
        }
    }

    /// Read a tensor, decoding block-quant types (Q4_0/Q8_0/Q4_K/Q6_K) to F16 so the
    /// engine only ever sees F16 (type 1) or F32 (type 0). Returns (dims, type, bytes).
    pub fn read_tensor(&mut self, name: &str) -> Result<(Vec<u64>, u32, Vec<u8>)> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("tensor {name} not found"))?
            .clone();
        let elems: u64 = info.dims.iter().product();
        let raw: u64 = match info.ggml_type {
            0 => elems * 4,          // F32
            1 => elems * 2,          // F16
            2 => elems / 32 * 18,    // Q4_0
            3 => elems / 32 * 20,    // Q4_1
            6 => elems / 32 * 22,    // Q5_0
            7 => elems / 32 * 24,    // Q5_1
            8 => elems / 32 * 34,    // Q8_0
            10 => elems / 256 * 84,  // Q2_K
            11 => elems / 256 * 110, // Q3_K
            12 => elems / 256 * 144, // Q4_K
            13 => elems / 256 * 176, // Q5_K
            14 => elems / 256 * 210, // Q6_K
            30 => elems * 2,         // BF16
            42 => elems / 128 * 34,  // Q2_0 (ternary g128)
            // IQ family. Sizes are listed for all of them even where
            // `dequant_to_f16` cannot yet decode the block: a wrong size here
            // mis-offsets every later tensor in the file, a worse failure than a
            // panic at decode time.
            16 => elems / 256 * 66,  // IQ2_XXS
            17 => elems / 256 * 74,  // IQ2_XS
            18 => elems / 256 * 98,  // IQ3_XXS
            19 => elems / 256 * 50,  // IQ1_S
            20 => elems / 32 * 18,   // IQ4_NL  (32-weight blocks, not 256)
            21 => elems / 256 * 110, // IQ3_S
            22 => elems / 256 * 82,  // IQ2_S
            23 => elems / 256 * 136, // IQ4_XS
            29 => elems / 256 * 56,  // IQ1_M   (no d field; scales carry it)
            t => bail!("unsupported GGUF type {t} ({})", gguf_type_name(t)),
        };
        let base = self.data_offsets[info.part];
        let f = &mut self.files[info.part];
        f.seek(SeekFrom::Start(base + info.rel_offset))?;
        let mut buf = vec![0u8; raw as usize];
        f.read_exact(&mut buf)?;
        match info.ggml_type {
            0 | 1 => Ok((info.dims, info.ggml_type, buf)),
            30 => {
                // BF16 → F32 (exact: bf16 is the top 16 bits of f32). Routed as F32
                // so e.g. MoE routers stay on the f32 GEMV path.
                let mut out = Vec::with_capacity(buf.len() * 2);
                for c in buf.chunks_exact(2) {
                    let bits = (u16::from_le_bytes([c[0], c[1]]) as u32) << 16;
                    out.extend_from_slice(&f32::from_bits(bits).to_le_bytes());
                }
                Ok((info.dims, 0, out))
            }
            _ => Ok((info.dims, 1, dequant_to_f16(&buf, info.ggml_type, elems as usize))),
        }
    }

    /// Read a tensor's raw quantized bytes (no dequant), returning (dims,
    /// ggml_type, bytes). Used by the native-precision path, which keeps
    /// Q4_K/Q6_K/Q8_0 in their GGUF format and dequants inside the GEMV kernels,
    /// so there is no requant accuracy loss.
    pub fn read_tensor_raw(&mut self, name: &str) -> Result<(Vec<u64>, u32, Vec<u8>)> {
        let info = self.tensors.get(name).ok_or_else(|| anyhow!("tensor {name} not found"))?.clone();
        let elems: u64 = info.dims.iter().product();
        let raw: u64 = match info.ggml_type {
            0 => elems * 4, 1 => elems * 2, 2 => elems / 32 * 18, 6 => elems / 32 * 22,
            // Every type with a native kernel must be listed here. A missing one
            // cannot be mapped zero-copy and falls back to requantization, which
            // on a mixed file leaves attn_v in a different representation from
            // attn_q/attn_k; the fused-qkv fallback then sends a bias dispatch to
            // a kernel with no bias form.
            3 => elems / 32 * 20, 7 => elems / 32 * 24,
            8 => elems / 32 * 34, 10 => elems / 256 * 84, 11 => elems / 256 * 110,
            12 => elems / 256 * 144, 13 => elems / 256 * 176,
            14 => elems / 256 * 210, 30 => elems * 2, 42 => elems / 128 * 34,
            16 => elems / 256 * 66, 17 => elems / 256 * 74, 18 => elems / 256 * 98,
            19 => elems / 256 * 50, 20 => elems / 32 * 18,  21 => elems / 256 * 110,
            22 => elems / 256 * 82, 23 => elems / 256 * 136, 29 => elems / 256 * 56,
            t => bail!("read_tensor_raw: unsupported GGUF type {t} ({})", gguf_type_name(t)),
        };
        let base = self.data_offsets[info.part];
        let f = &mut self.files[info.part];
        f.seek(SeekFrom::Start(base + info.rel_offset))?;
        let mut buf = vec![0u8; raw as usize];
        f.read_exact(&mut buf)?;
        Ok((info.dims, info.ggml_type, buf))
    }

    /// Shard file paths (for direct mmap streaming). Single-file → [path]; split → all parts.
    pub fn shard_paths(&self) -> Vec<std::path::PathBuf> {
        self.part_paths.iter().map(std::path::PathBuf::from).collect()
    }

    /// Merge another GGUF's tensors in as an extra part.
    ///
    /// This is how an MTP/NextN sidecar is loaded: the head ships as its own file
    /// (the reference loads it as a separate draft context; here the tensor names
    /// already match what the draft block looks up, so making it a part is
    /// enough). `keep` selects which tensors to take — a sidecar also carries its
    /// own token_embd/output/output_norm, which would shadow the main model's.
    ///
    /// Existing names are never overwritten: the main model wins, so a sidecar
    /// cannot silently replace a weight the model already has.
    ///
    /// The incoming file's metadata is discarded. That is right for an MTP
    /// sidecar — its KV is a copy of the target's — and wrong for a sidecar
    /// whose KV is the payload, which is what `attach_with_meta` is for.
    pub fn attach(&mut self, path: &str, keep: impl Fn(&str) -> bool) -> Result<usize> {
        self.attach_with_meta(path, keep, |_| false)
    }

    /// `attach`, plus the incoming file's metadata for the keys `keep_kv` accepts.
    ///
    /// A vision projector (`mmproj`) sidecar carries its whole configuration in
    /// KV — ~20 `clip.*` keys describing the ViT geometry, the normalisation
    /// constants and the projector type — and none of it can be recovered from
    /// the tensors. `attach` drops all of it, so an mmproj needs this form.
    ///
    /// KV merges follow the tensor rule: the main model wins. An incoming key
    /// the main model already defines is dropped, never overwritten, so no
    /// sidecar can change `general.architecture`, `<arch>.embedding_length` or
    /// any other key the decoder was loaded on. `keep_kv` is a second line of
    /// defence, not the only one: pass a prefix filter
    /// (`ojas_formats::mmproj::keep_kv`) so unrelated keys never enter the map.
    ///
    /// Returns the number of tensors added, as `attach` does; KV merges are not
    /// counted, because a key already present is a no-op rather than an error.
    pub fn attach_with_meta(
        &mut self,
        path: &str,
        keep: impl Fn(&str) -> bool,
        keep_kv: impl Fn(&str) -> bool,
    ) -> Result<usize> {
        let mut incoming = Self::open(path)?;
        anyhow::ensure!(incoming.files.len() == 1, "sidecar must be a single GGUF file");
        let f = incoming.files.pop().unwrap();
        let off = incoming.data_offsets[0];
        let tensors = std::mem::take(&mut incoming.tensors);
        let meta = std::mem::take(&mut incoming.meta);
        let part = self.files.len();
        self.files.push(f);
        self.data_offsets.push(off);
        self.part_paths.push(path.to_string());
        let mut added = 0usize;
        for (name, mut ti) in tensors {
            if !keep(&name) || self.tensors.contains_key(&name) { continue; }
            ti.part = part;
            self.tensors.insert(name, ti);
            added += 1;
        }
        for (key, value) in meta {
            if !keep_kv(&key) { continue; }
            self.meta.entry(key).or_insert(value);
        }
        Ok(added)
    }

    /// (part, absolute file offset, raw byte length, ggml_type) — for zero-copy mmap streaming
    /// (no read). None if the type has no fixed raw size this reader supports.
    pub fn tensor_meta(&self, name: &str) -> Option<(usize, u64, u64, u32)> {
        let info = self.tensors.get(name)?;
        let elems: u64 = info.dims.iter().product();
        let raw = match info.ggml_type {
            0 => elems * 4, 1 => elems * 2, 2 => elems / 32 * 18, 6 => elems / 32 * 22,
            // Every type with a native kernel must be listed here. A missing one
            // cannot be mapped zero-copy and falls back to requantization, which
            // on a mixed file leaves attn_v in a different representation from
            // attn_q/attn_k; the fused-qkv fallback then sends a bias dispatch to
            // a kernel with no bias form.
            3 => elems / 32 * 20, 7 => elems / 32 * 24,
            8 => elems / 32 * 34, 10 => elems / 256 * 84, 11 => elems / 256 * 110,
            12 => elems / 256 * 144, 13 => elems / 256 * 176,
            14 => elems / 256 * 210, 30 => elems * 2,
            16 => elems / 256 * 66, 17 => elems / 256 * 74, 18 => elems / 256 * 98,
            19 => elems / 256 * 50, 20 => elems / 32 * 18,  21 => elems / 256 * 110,
            22 => elems / 256 * 82, 23 => elems / 256 * 136, 29 => elems / 256 * 56,
            42 => elems / 128 * 34,                           // Q2_0 (ternary g128)
            _ => return None,
        };
        Some((info.part, self.data_offsets[info.part] + info.rel_offset, raw, info.ggml_type))
    }
}

/// Sign for lane `j` of an 8-wide IQ group. The IQ2/IQ3 grids store magnitudes
/// only; the sign of every weight comes from a separate 8-bit word, one bit per
/// lane, and a set bit means negative.
#[inline]
fn iq_sgn(signs: u8, j: usize) -> f32 {
    if signs & crate::iq_tables::KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 }
}

#[inline]
fn h16(v: f32, out: &mut Vec<u8>) {
    out.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
}
#[inline]
fn rdh(b: &[u8], i: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[i], b[i + 1]])).to_f32()
}

/// Decode a block-quantized byte slice to F16 little-endian bytes, `n` elements
/// of GGUF type `ggml_type`. Public for expert streaming.
pub fn dequant_to_f16(b: &[u8], ggml_type: u32, n: usize) -> Vec<u8> {
    // Q5_1 (gpt-oss weights) is the dominant model-load cost — parallelize it with
    // indexed writes into a pre-sized buffer (the serial branch below pushes one
    // f16 at a time). Same math as the ggml_type==7 arm.
    if ggml_type == 7 {
        let nblk = b.len() / 24;
        let mut out = vec![0u8; nblk * 64];
        let nthreads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1).clamp(1, nblk.max(1));
        let blk_per = (nblk + nthreads - 1) / nthreads;
        std::thread::scope(|s| {
            for (ti, oc) in out.chunks_mut(blk_per * 64).enumerate() {
                let blk0 = ti * blk_per;
                s.spawn(move || {
                    for bi in 0..(oc.len() / 64) {
                        let blk = &b[(blk0 + bi) * 24..(blk0 + bi) * 24 + 24];
                        let d = rdh(blk, 0);
                        let m = rdh(blk, 2);
                        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                        let mut wr = |idx: usize, val: f32| {
                            let bits = half::f16::from_f32(val).to_bits().to_le_bytes();
                            oc[bi * 64 + idx * 2] = bits[0];
                            oc[bi * 64 + idx * 2 + 1] = bits[1];
                        };
                        for j in 0..16 {
                            let hb = ((qh >> j) & 1) << 4;
                            let q = (blk[8 + j] & 0x0F) as u32 | hb;
                            wr(j, d * q as f32 + m);
                        }
                        for j in 0..16 {
                            let hb = ((qh >> (j + 16)) & 1) << 4;
                            let q = (blk[8 + j] >> 4) as u32 | hb;
                            wr(16 + j, d * q as f32 + m);
                        }
                    }
                });
            }
        });
        return out;
    }
    let mut out = Vec::with_capacity(n * 2);
    match ggml_type {
        2 => { // Q4_0: block { half d; u8 qs[16]; } = 18 B / 32 weights; w = d*(nib-8)
            // block layout: weights [0..16) are the low nibbles of qs[0..16], weights
            // [16..32) are the high nibbles — not interleaved 2j/2j+1 per byte.
            for blk in b.chunks_exact(18) {
                let d = rdh(blk, 0);
                for j in 0..16 { let byte = blk[2 + j]; h16(d * ((byte & 0xF) as f32 - 8.0), &mut out); }
                for j in 0..16 { let byte = blk[2 + j]; h16(d * ((byte >> 4) as f32 - 8.0), &mut out); }
            }
        }
        6 => { // Q5_0: block { half d; u8 qh[4]; u8 qs[16]; } = 22 B / 32; w = d*(q-16)
            // 5-bit: low 4 bits in qs (like Q4_0), 5th bit in the qh bitfield.
            for blk in b.chunks_exact(22) {
                let d = rdh(blk, 0);
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                for j in 0..16 {
                    let hb = ((qh >> j) & 1) << 4;               // 5th bit for weight j
                    let q = (blk[6 + j] & 0x0F) as u32 | hb;
                    h16(d * (q as f32 - 16.0), &mut out);
                }
                for j in 0..16 {
                    let hb = ((qh >> (j + 16)) & 1) << 4;         // 5th bit for weight j+16
                    let q = (blk[6 + j] >> 4) as u32 | hb;
                    h16(d * (q as f32 - 16.0), &mut out);
                }
            }
        }
        3 => { // Q4_1: block { half d; half m; u8 qs[16]; } = 20 B / 32; w = nib*d + m
            for blk in b.chunks_exact(20) {
                let d = rdh(blk, 0);
                let m = rdh(blk, 2);
                for j in 0..16 { h16(d * (blk[4 + j] & 0x0F) as f32 + m, &mut out); }
                for j in 0..16 { h16(d * (blk[4 + j] >> 4) as f32 + m, &mut out); }
            }
        }
        7 => { // Q5_1: block { half d; half m; u8 qh[4]; u8 qs[16]; } = 24 B / 32; w = q*d + m (q 5-bit)
            for blk in b.chunks_exact(24) {
                let d = rdh(blk, 0);
                let m = rdh(blk, 2);
                let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                for j in 0..16 {
                    let hb = ((qh >> j) & 1) << 4;               // 5th bit for weight j
                    let q = (blk[8 + j] & 0x0F) as u32 | hb;
                    h16(d * q as f32 + m, &mut out);
                }
                for j in 0..16 {
                    let hb = ((qh >> (j + 16)) & 1) << 4;         // 5th bit for weight j+16
                    let q = (blk[8 + j] >> 4) as u32 | hb;
                    h16(d * q as f32 + m, &mut out);
                }
            }
        }
        8 => { // Q8_0: block { half d; i8 qs[32]; } = 34 B / 32; w = d*q
            for blk in b.chunks_exact(34) {
                let d = rdh(blk, 0);
                for j in 0..32 { h16(d * (blk[2 + j] as i8 as f32), &mut out); }
            }
        }
        42 => { // Q2_0 (ternary g128): block { half d; u8 qs[32]; } = 34 B / 128.
            // 2 bits/weight, sequential packing — weight j is at byte j/4, bits (j%4)*2.
            // Codes 00/01/10 = -1/0/+1; 11 (=+2) is reserved and unused for ternary.
            for blk in b.chunks_exact(34) {
                let d = rdh(blk, 0);
                for j in 0..128 {
                    let q = (blk[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
                    h16(d * (q as f32 - 1.0), &mut out);
                }
            }
        }
        0 => { // F32 -> f16
            for c in b.chunks_exact(4) { h16(f32::from_le_bytes([c[0], c[1], c[2], c[3]]), &mut out); }
        }
        1 => { out.extend_from_slice(&b[..(n * 2).min(b.len())]); }   // F16: already the target
        30 => { // BF16 -> f16: bf16 is the top 16 bits of an f32
            for c in b.chunks_exact(2) {
                h16(f32::from_bits(u32::from_le_bytes([0, 0, c[0], c[1]])), &mut out);
            }
        }
        10 => { // Q2_K: block { u8 scales[16]; u8 qs[64]; half d; half dmin; } = 84 B / 256
            // 2-bit weights, 16 sub-blocks of 16. Each scales[] byte packs a 4-bit
            // scale (low nibble) and a 4-bit min (high). w = d*sc*q - dmin*m.
            // Field order: unlike every other K-quant, d/dmin sit at the end of
            // the block, after scales and qs (block_q2_K).
            for blk in b.chunks_exact(84) {
                let sc = &blk[0..16];
                let qs = &blk[16..80];
                let d = rdh(blk, 80);
                let dmin = rdh(blk, 82);
                let mut is = 0usize;
                for n in 0..2 {                       // two halves of 128 weights
                    let q = &qs[n * 32..n * 32 + 32];
                    for j in 0..4 {
                        let shift = 2 * j;
                        let s1 = sc[is]; is += 1;
                        let (dl, ml) = (d * (s1 & 0xF) as f32, dmin * (s1 >> 4) as f32);
                        for l in 0..16 { h16(dl * ((q[l] >> shift) & 3) as f32 - ml, &mut out); }
                        let s2 = sc[is]; is += 1;
                        let (dl, ml) = (d * (s2 & 0xF) as f32, dmin * (s2 >> 4) as f32);
                        for l in 0..16 { h16(dl * ((q[l + 16] >> shift) & 3) as f32 - ml, &mut out); }
                    }
                }
            }
        }
        11 => { // Q3_K: block { u8 hmask[32]; u8 qs[64]; u8 scales[12]; half d; } = 110 B / 256
            // 3-bit: 2 low bits in qs, the 3rd in hmask as a per-weight inverted
            // bit (set = 0, clear = subtract 4). The 12 scale bytes hold sixteen
            // 6-bit signed-biased scales in the same packed form the reference unpacks with
            // kmask1/kmask2; reproduced here so the values match bit for bit.
            for blk in b.chunks_exact(110) {
                let hm = &blk[0..32];
                let qs = &blk[32..96];
                let raw = &blk[96..108];
                let d_all = rdh(blk, 108);
                let mut aux = [0u32; 4];
                for i in 0..3 { aux[i] = u32::from_le_bytes([raw[4*i], raw[4*i+1], raw[4*i+2], raw[4*i+3]]); }
                let (kmask1, kmask2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
                let tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                let mut sc = [0i8; 16];
                for i in 0..4 { sc[4*i..4*i+4].copy_from_slice(&(aux[i].to_le_bytes().map(|v| v as i8))); }
                let mut is = 0usize;
                let mut m = 1u8;
                for n in 0..2 {
                    let q = &qs[n * 32..n * 32 + 32];
                    for j in 0..4 {
                        let shift = 2 * j;
                        let dl = d_all * (sc[is] as i32 - 32) as f32; is += 1;
                        for l in 0..16 {
                            let hi = if hm[l] & m != 0 { 0 } else { 4 };
                            h16(dl * (((q[l] >> shift) & 3) as i32 - hi) as f32, &mut out);
                        }
                        let dl = d_all * (sc[is] as i32 - 32) as f32; is += 1;
                        for l in 0..16 {
                            let hi = if hm[l + 16] & m != 0 { 0 } else { 4 };
                            h16(dl * (((q[l + 16] >> shift) & 3) as i32 - hi) as f32, &mut out);
                        }
                        m <<= 1;
                    }
                }
            }
        }
        20 => { // IQ4_NL: block { half d; u8 qs[16]; } = 18 B / 32 weights
            // "NL" = non-linear: the 4 bits index a fixed 16-entry codebook of
            // unevenly spaced levels rather than a uniform ramp, which is what
            // buys accuracy over Q4_0 at identical size. Low nibbles fill the
            // first half of the block, high nibbles the second — not interleaved.
            for blk in b.chunks_exact(18) {
                let d = rdh(blk, 0);
                let qs = &blk[2..18];
                let mut lo = [0f32; 16];
                let mut hi = [0f32; 16];
                for j in 0..16 {
                    lo[j] = d * crate::iq_tables::KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
                    hi[j] = d * crate::iq_tables::KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                }
                for v in lo { h16(v, &mut out); }
                for v in hi { h16(v, &mut out); }
            }
        }
        23 => { // IQ4_XS: block { half d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; } = 136 B / 256
            // Same codebook as IQ4_NL, but with 8 sub-blocks of 32 sharing one
            // f16 `d` and each carrying a 6-bit scale split across two arrays:
            // 4 low bits in scales_l (nibble-packed) and 2 high bits in scales_h
            // (2 bits per sub-block). Biased by 32, like the K-quants.
            for blk in b.chunks_exact(136) {
                let d = rdh(blk, 0);
                let sh = u16::from_le_bytes([blk[2], blk[3]]);
                let sl = &blk[4..8];
                let qs = &blk[8..136];
                for ib in 0..8 {
                    let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) as u32
                        | ((((sh >> (2 * ib)) & 3) as u32) << 4);
                    let dl = d * (ls as i32 - 32) as f32;
                    let q = &qs[ib * 16..ib * 16 + 16];
                    let mut lo = [0f32; 16];
                    let mut hi = [0f32; 16];
                    for j in 0..16 {
                        lo[j] = dl * crate::iq_tables::KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
                        hi[j] = dl * crate::iq_tables::KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
                    }
                    for v in lo { h16(v, &mut out); }
                    for v in hi { h16(v, &mut out); }
                }
            }
        }
        16 => { // IQ2_XXS: block { half d; u16 qs[32]; } = 66 B / 256
            // The IQ ("importance-quantized") family does not store weights at
            // all: each group of 8 shares one codebook entry giving 8 magnitudes,
            // plus a sign word and a scale. 2.06 bits/weight comes from spending
            // 8 bits on a grid index rather than 2 bits per weight.
            use crate::iq_tables::{IQ2XXS_GRID, KSIGNS_IQ2XS};
            for blk in b.chunks_exact(66) {
                let d = rdh(blk, 0);
                let qs = &blk[2..66];
                for ib32 in 0..8 {
                    let o = 8 * ib32;
                    let a0 = u32::from_le_bytes([qs[o], qs[o + 1], qs[o + 2], qs[o + 3]]);
                    let a1 = u32::from_le_bytes([qs[o + 4], qs[o + 5], qs[o + 6], qs[o + 7]]);
                    // Top nibble of the second word is the sub-block scale; the
                    // remaining 28 bits are four 7-bit sign-table indices.
                    let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                    let gi = a0.to_le_bytes();
                    for l in 0..4 {
                        let g = IQ2XXS_GRID[gi[l] as usize].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                        for j in 0..8 { h16(db * g[j] as f32 * iq_sgn(signs, j), &mut out); }
                    }
                }
            }
        }
        17 => { // IQ2_XS: block { half d; u16 qs[32]; u8 scales[8]; } = 74 B / 256
            // Same idea as IQ2_XXS but the grid is 512 entries and the sign index
            // rides in the same u16 as the grid index (9 bits grid, 7 bits sign),
            // which frees the scale into its own nibble array.
            use crate::iq_tables::{IQ2XS_GRID, KSIGNS_IQ2XS};
            for blk in b.chunks_exact(74) {
                let d = rdh(blk, 0);
                let qs = &blk[2..66];
                let sc = &blk[66..74];
                for ib32 in 0..8 {
                    let db = [
                        d * (0.5 + (sc[ib32] & 0xF) as f32) * 0.25,
                        d * (0.5 + (sc[ib32] >> 4) as f32) * 0.25,
                    ];
                    for l in 0..4 {
                        let i = 4 * ib32 + l;
                        let q = u16::from_le_bytes([qs[2 * i], qs[2 * i + 1]]);
                        let g = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                        for j in 0..8 { h16(db[l / 2] * g[j] as f32 * iq_sgn(signs, j), &mut out); }
                    }
                }
            }
        }
        22 => { // IQ2_S: block { half d; u8 qs[64]; u8 qh[8]; u8 scales[8]; } = 82 B / 256
            // 2.5 bpw. qs splits in half: the first 32 bytes are grid indices
            // (low 8 bits, high 2 bits from qh), the second 32 are raw sign bytes
            // rather than indices into ksigns — the extra bits are what buys the
            // accuracy over IQ2_XS.
            use crate::iq_tables::IQ2S_GRID;
            for blk in b.chunks_exact(82) {
                let d = rdh(blk, 0);
                let qs = &blk[2..66];
                let qh = &blk[66..74];
                let sc = &blk[74..82];
                for ib32 in 0..8 {
                    let db = [
                        d * (0.5 + (sc[ib32] & 0xF) as f32) * 0.25,
                        d * (0.5 + (sc[ib32] >> 4) as f32) * 0.25,
                    ];
                    for l in 0..4 {
                        let gi = qs[4 * ib32 + l] as usize
                            | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                        let g = IQ2S_GRID[gi].to_le_bytes();
                        let signs = qs[32 + 4 * ib32 + l];
                        for j in 0..8 { h16(db[l / 2] * g[j] as f32 * iq_sgn(signs, j), &mut out); }
                    }
                }
            }
        }
        18 => { // IQ3_XXS: block { half d; u8 qs[96]; } = 98 B / 256
            // 3.06 bpw. Grid entries are 4 magnitudes (u32), so a group of 8 needs
            // two indices; the trailing 32 bytes of qs hold scale+signs packed the
            // same way IQ2_XXS packs them.
            use crate::iq_tables::{IQ3XXS_GRID, KSIGNS_IQ2XS};
            for blk in b.chunks_exact(98) {
                let d = rdh(blk, 0);
                let qs = &blk[2..98];
                let ss = &qs[64..96];
                for ib32 in 0..8 {
                    let a = u32::from_le_bytes([
                        ss[4 * ib32], ss[4 * ib32 + 1], ss[4 * ib32 + 2], ss[4 * ib32 + 3]]);
                    let db = d * (0.5 + (a >> 28) as f32) * 0.5;
                    let q = &qs[8 * ib32..8 * ib32 + 8];
                    for l in 0..4 {
                        let signs = KSIGNS_IQ2XS[((a >> (7 * l)) & 127) as usize];
                        let g1 = IQ3XXS_GRID[q[2 * l] as usize].to_le_bytes();
                        let g2 = IQ3XXS_GRID[q[2 * l + 1] as usize].to_le_bytes();
                        // Lanes 0..4 come from the first index, 4..8 from the
                        // second, but both read the sign word at their own lane.
                        for j in 0..4 { h16(db * g1[j] as f32 * iq_sgn(signs, j), &mut out); }
                        for j in 0..4 { h16(db * g2[j] as f32 * iq_sgn(signs, j + 4), &mut out); }
                    }
                }
            }
        }
        21 => { // IQ3_S: block { half d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]; } = 110 B / 256
            // 3.44 bpw. Grid is 512 entries (9th bit from qh), signs are raw, and
            // the scale is a 4-bit field per 64 weights applied as (1 + 2*s),
            // a different scale form from every other IQ type here.
            use crate::iq_tables::IQ3S_GRID;
            for blk in b.chunks_exact(110) {
                let d = rdh(blk, 0);
                let qs = &blk[2..66];
                let qh = &blk[66..74];
                let sg = &blk[74..106];
                let sc = &blk[106..110];
                // Each scale byte covers two sub-blocks of 32, so this walks four
                // pairs rather than eight singles.
                for p in 0..4 {
                    let dbs = [
                        d * (1.0 + 2.0 * (sc[p] & 0xF) as f32),
                        d * (1.0 + 2.0 * (sc[p] >> 4) as f32),
                    ];
                    for half in 0..2 {
                        let db = dbs[half];
                        let h = qh[2 * p + half] as usize;
                        let qo = 16 * p + 8 * half;
                        let so = 8 * p + 4 * half;
                        for l in 0..4 {
                            let g1 = IQ3S_GRID[qs[qo + 2 * l] as usize | ((h << (8 - 2 * l)) & 256)]
                                .to_le_bytes();
                            let g2 = IQ3S_GRID[qs[qo + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256)]
                                .to_le_bytes();
                            let signs = sg[so + l];
                            for j in 0..4 { h16(db * g1[j] as f32 * iq_sgn(signs, j), &mut out); }
                            for j in 0..4 { h16(db * g2[j] as f32 * iq_sgn(signs, j + 4), &mut out); }
                        }
                    }
                }
            }
        }
        19 => { // IQ1_S: block { half d; u8 qs[32]; u16 qh[8]; } = 50 B / 256
            // 1.56 bpw. The grid is 2048 entries of eight signed bytes (the IQ2/
            // IQ3 grids are unsigned; mixing that up silently halves the value
            // range). Each sub-block carries a 3-bit scale and a sign bit that
            // picks the delta, a constant offset added to every magnitude — at one
            // bit per weight there is no room for an offset per weight.
            use crate::iq_tables::IQ1S_GRID;
            const IQ1S_DELTA: f32 = 0.125;
            for blk in b.chunks_exact(50) {
                let d = rdh(blk, 0);
                let qs = &blk[2..34];
                for ib in 0..8 {
                    let qh = u16::from_le_bytes([blk[34 + 2 * ib], blk[35 + 2 * ib]]);
                    let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                    let delta = if qh & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                    for l in 0..4 {
                        let gi = qs[4 * ib + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8);
                        let g = IQ1S_GRID[gi].to_le_bytes();
                        for j in 0..8 { h16(dl * (g[j] as i8 as f32 + delta), &mut out); }
                    }
                }
            }
        }
        29 => { // IQ1_M: block { u8 qs[32]; u8 qh[16]; u8 scales[8]; } = 56 B / 256
            // 1.75 bpw, and the only block in the format family with no `d`
            // field: the f16 scale is reassembled from four nibbles scattered
            // across the four scale words, which is how the extra scale
            // resolution is paid for. Two 3-bit scales per 32 weights.
            use crate::iq_tables::IQ1S_GRID;
            const IQ1S_DELTA: f32 = 0.125;
            for blk in b.chunks_exact(56) {
                let qs = &blk[0..32];
                let qh = &blk[32..48];
                let sc: [u16; 4] = std::array::from_fn(|i| {
                    u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]])
                });
                let dbits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0)
                    | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
                let d = half::f16::from_bits(dbits).to_f32();
                for ib in 0..8 {
                    let w = sc[ib / 2];
                    let sh = 6 * (ib % 2);
                    let dl1 = d * (2 * ((w >> sh) & 7) + 1) as f32;
                    let dl2 = d * (2 * ((w >> (sh + 3)) & 7) + 1) as f32;
                    let (h0, h1) = (qh[2 * ib] as usize, qh[2 * ib + 1] as usize);
                    let idx = [
                        qs[4 * ib] as usize | ((h0 << 8) & 0x700),
                        qs[4 * ib + 1] as usize | ((h0 << 4) & 0x700),
                        qs[4 * ib + 2] as usize | ((h1 << 8) & 0x700),
                        qs[4 * ib + 3] as usize | ((h1 << 4) & 0x700),
                    ];
                    let dq = [
                        if h0 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                        if h0 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                        if h1 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                        if h1 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    ];
                    for l in 0..4 {
                        let dl = if l < 2 { dl1 } else { dl2 };
                        let g = IQ1S_GRID[idx[l]].to_le_bytes();
                        for j in 0..8 { h16(dl * (g[j] as i8 as f32 + dq[l]), &mut out); }
                    }
                }
            }
        }
        12 => { // Q4_K: block { half d; half dmin; u8 scales[12]; u8 qs[128]; } = 144 B / 256
            for blk in b.chunks_exact(144) {
                let d = rdh(blk, 0);
                let dmin = rdh(blk, 2);
                let sc = &blk[4..16];   // scales[12]
                let qs = &blk[16..144]; // qs[128]
                // 8 sub-blocks of 32; process qs in 32-byte chunks (2 sub-blocks each)
                for chunk in 0..4 {
                    let (d1, m1) = get_scale_min_k4(chunk * 2, sc);
                    let (d2, m2) = get_scale_min_k4(chunk * 2 + 1, sc);
                    let (dl1, ml1) = (d * d1, dmin * m1);
                    let (dl2, ml2) = (d * d2, dmin * m2);
                    let q = &qs[chunk * 32..chunk * 32 + 32];
                    for l in 0..32 { h16(dl1 * (q[l] & 0xF) as f32 - ml1, &mut out); }
                    for l in 0..32 { h16(dl2 * (q[l] >> 4) as f32 - ml2, &mut out); }
                }
            }
        }
        13 => { // Q5_K: {half d; half dmin; u8 scales[12]; u8 qh[32]; u8 qs[128];} = 176 B / 256
            for blk in b.chunks_exact(176) {
                let d = rdh(blk, 0);
                let dmin = rdh(blk, 2);
                let sc = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                let (mut u1, mut u2) = (1u8, 2u8);
                for chunk in 0..4 {
                    let (ds1, ms1) = get_scale_min_k4(chunk * 2, sc);
                    let (ds2, ms2) = get_scale_min_k4(chunk * 2 + 1, sc);
                    let (dl1, ml1) = (d * ds1, dmin * ms1);
                    let (dl2, ml2) = (d * ds2, dmin * ms2);
                    let ql = &qs[chunk * 32..chunk * 32 + 32];
                    for l in 0..32 {
                        let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                        h16(dl1 * ((ql[l] & 0xF) as f32 + hi) - ml1, &mut out);
                    }
                    for l in 0..32 {
                        let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                        h16(dl2 * ((ql[l] >> 4) as f32 + hi) - ml2, &mut out);
                    }
                    u1 <<= 2; u2 <<= 2;
                }
            }
        }
        14 => { // Q6_K: block { u8 ql[128]; u8 qh[64]; i8 scales[16]; half d; } = 210 B / 256
            let mut y = [0f32; 128]; // per-half scratch (strided writes)
            for blk in b.chunks_exact(210) {
                let d = rdh(blk, 208);
                for half_i in 0..2 {
                    let ql = &blk[half_i * 64..half_i * 64 + 64];
                    let qh = &blk[128 + half_i * 32..128 + half_i * 32 + 32];
                    let sc = &blk[192 + half_i * 8..192 + half_i * 8 + 8];
                    for l in 0..32 {
                        let is = l / 16; // 0 or 1
                        let q1 = ((ql[l] & 0xF) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) - 32;
                        let q2 = ((ql[l + 32] & 0xF) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
                        let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
                        let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;
                        y[l]      = d * sc[is] as i8 as f32 * q1 as f32;
                        y[l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                        y[l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                        y[l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                    }
                    for &v in y.iter() { h16(v, &mut out); }
                }
            }
        }
        // An unsupported type must fail here: falling through would return the
        // buffer reserved above but never filled, making the tensor silently
        // zero-length instead of failing the load.
        t => panic!("dequant_to_f16: unsupported GGUF type {t} ({}) — \
                     add a block-format arm before loading this model", gguf_type_name(t)),
    }
    out
}

/// Human-readable type name, for diagnostics only.
pub fn gguf_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1",
        8 => "Q8_0", 9 => "Q8_1", 10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K",
        13 => "Q5_K", 14 => "Q6_K", 15 => "Q8_K", 16 => "IQ2_XXS", 17 => "IQ2_XS",
        18 => "IQ3_XXS", 19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S", 22 => "IQ2_S",
        23 => "IQ4_XS", 29 => "IQ1_M", 30 => "BF16", 34 => "TQ1_0", 35 => "TQ2_0",
        39 => "MXFP4", 42 => "Q2_0", 142 => "PQ2_0", _ => "unknown",
    }
}

// Q4_K sub-block scale/min extraction (get_scale_min_k4), returns (scale, min) as f32.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((q[j] & 63) as f32, (q[j + 4] & 63) as f32)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d as f32, m as f32)
    }
}
