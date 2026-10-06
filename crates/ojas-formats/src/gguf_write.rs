//! Writing a GGUF: a copy of an existing file's metadata with new tensors, as a
//! trained model is saved in the format the engine loads.
//!
//! The metadata section is copied byte for byte from the source file, so the
//! tokenizer, the chat templates and every architecture key survive unchanged
//! whatever their types; the values of named keys can be replaced — a float or an
//! integer scalar, or an integer array, which may change length (a model that gains
//! blocks gains entries in its per-block arrays). Tensor data is written F32 or F16,
//! aligned as the header declares.

use anyhow::{anyhow, bail, ensure, Context, Result};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, Write};
use std::path::Path;

/// One tensor to write: its name, dimensions in GGUF order (fastest axis first),
/// and values. `f16` stores the values as half floats.
pub struct TensorOut<'a> {
    pub name: &'a str,
    pub dims: Vec<u64>,
    pub data: &'a [f32],
    pub f16: bool,
}

const GGUF_F32: u32 = 0;
const GGUF_F16: u32 = 1;

/// A new value for an existing metadata key, written in the key's own GGUF type.
#[derive(Clone, Debug)]
pub enum MetaValue {
    /// For an F32 or F64 key.
    Float(f64),
    /// For an integer key of any width.
    Int(i64),
    /// For an integer array, of any length.
    IntArray(Vec<i64>),
}

/// Write `dst`: `src`'s metadata, with the keys in `set` given new values, and
/// `tensors` in the order given. Every key in `set` must exist in `src` with a type
/// its value fits.
pub fn write_gguf(src: &Path, dst: &Path, set: &[(&str, MetaValue)], tensors: &[TensorOut<'_>]) -> Result<()> {
    let mut f = File::open(src).with_context(|| format!("opening {}", src.display()))?;
    let mut head = [0u8; 24];
    f.read_exact(&mut head)?;
    ensure!(&head[..4] == b"GGUF", "{} is not a GGUF file", src.display());
    let version = u32::from_le_bytes(head[4..8].try_into().unwrap());
    ensure!(matches!(version, 2 | 3), "unsupported GGUF version {version}");
    let n_kv = u64::from_le_bytes(head[16..24].try_into().unwrap());

    // The metadata section is at most this long in any file the reader accepts.
    let mut kv = Vec::new();
    f.take(256 * 1024 * 1024).read_to_end(&mut kv)?;
    let mut pos = 0usize;
    let mut align = 32u64;
    let mut replaced = vec![false; set.len()];
    let mut rebuilt = Vec::with_capacity(kv.len());
    for _ in 0..n_kv {
        let entry_at = pos;
        let key = read_str(&kv, &mut pos)?;
        let t = read_u32(&kv, &mut pos)?;
        let value_at = pos;
        if key == "general.alignment" && t == 4 {
            align = read_u32(&kv, &mut { pos })? as u64;
        }
        ensure!(key != "split.count" || matches!(t, 2 | 4) && read_u32(&kv, &mut { pos }).is_ok_and(|n| n <= 1) || t == 0,
            "{} is a split file; write a single file", src.display());
        skip_value(&kv, &mut pos, t)?;
        match set.iter().position(|(k, _)| *k == key) {
            None => rebuilt.extend_from_slice(&kv[entry_at..pos]),
            Some(i) => {
                rebuilt.extend_from_slice(&kv[entry_at..value_at]);
                encode(&mut rebuilt, t, &kv[value_at..pos], &set[i].1).with_context(|| format!("{key} in {}", src.display()))?;
                replaced[i] = true;
            }
        }
    }
    if let Some(i) = replaced.iter().position(|r| !r) {
        bail!("{} has no key {}", src.display(), set[i].0);
    }
    let kv = rebuilt;
    ensure!(align.is_power_of_two(), "invalid GGUF alignment {align}");

    let mut out = BufWriter::new(File::create(dst).with_context(|| format!("creating {}", dst.display()))?);
    out.write_all(b"GGUF")?;
    out.write_all(&version.to_le_bytes())?;
    out.write_all(&(tensors.len() as u64).to_le_bytes())?;
    out.write_all(&n_kv.to_le_bytes())?;
    out.write_all(&kv)?;
    let mut offset = 0u64;
    for t in tensors {
        let elements: u64 = t.dims.iter().product();
        ensure!(elements as usize == t.data.len(), "{}: {} values for dims {:?}", t.name, t.data.len(), t.dims);
        ensure!((1..=4).contains(&t.dims.len()), "{}: rank {} is not 1 to 4", t.name, t.dims.len());
        write_str(&mut out, t.name)?;
        out.write_all(&(t.dims.len() as u32).to_le_bytes())?;
        for d in &t.dims { out.write_all(&d.to_le_bytes())?; }
        out.write_all(&if t.f16 { GGUF_F16 } else { GGUF_F32 }.to_le_bytes())?;
        out.write_all(&offset.to_le_bytes())?;
        let bytes = elements * if t.f16 { 2 } else { 4 };
        offset = (offset + bytes + align - 1) & !(align - 1);
    }
    let here = out.stream_position()?;
    let data_start = (here + align - 1) & !(align - 1);
    out.write_all(&vec![0u8; (data_start - here) as usize])?;
    let mut written = 0u64;
    for t in tensors {
        let mut bytes = Vec::with_capacity(t.data.len() * 4);
        if t.f16 {
            for &v in t.data { bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()); }
        } else {
            for &v in t.data { bytes.extend_from_slice(&v.to_le_bytes()); }
        }
        out.write_all(&bytes)?;
        written += bytes.len() as u64;
        let padded = (written + align - 1) & !(align - 1);
        out.write_all(&vec![0u8; (padded - written) as usize])?;
        written = padded;
    }
    out.flush()?;
    Ok(())
}

/// `value` in GGUF type `t`; `old` is the key's current encoded value, from which an
/// array keeps its element type.
fn encode(out: &mut Vec<u8>, t: u32, old: &[u8], value: &MetaValue) -> Result<()> {
    let int = |out: &mut Vec<u8>, t: u32, v: i64| -> Result<()> {
        match t {
            0 | 1 | 7 => out.push(v as u8),
            2 | 3 => out.extend_from_slice(&(v as u16).to_le_bytes()),
            4 | 5 => out.extend_from_slice(&(v as u32).to_le_bytes()),
            10 | 11 => out.extend_from_slice(&(v as u64).to_le_bytes()),
            _ => bail!("type {t} is not an integer"),
        }
        Ok(())
    };
    match (value, t) {
        (MetaValue::Float(v), 6) => out.extend_from_slice(&(*v as f32).to_le_bytes()),
        (MetaValue::Float(v), 12) => out.extend_from_slice(&v.to_le_bytes()),
        (MetaValue::Int(v), t) => int(out, t, *v)?,
        (MetaValue::IntArray(values), 9) => {
            let element = u32::from_le_bytes(old.get(..4).context("truncated array")?.try_into().unwrap());
            out.extend_from_slice(&element.to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for &v in values { int(out, element, v)?; }
        }
        (v, t) => bail!("a {v:?} does not fit GGUF type {t}"),
    }
    Ok(())
}

fn read_u32(b: &[u8], pos: &mut usize) -> Result<u32> {
    let v = b.get(*pos..*pos + 4).ok_or_else(|| anyhow!("truncated GGUF metadata"))?;
    *pos += 4;
    Ok(u32::from_le_bytes(v.try_into().unwrap()))
}

fn read_u64(b: &[u8], pos: &mut usize) -> Result<u64> {
    let v = b.get(*pos..*pos + 8).ok_or_else(|| anyhow!("truncated GGUF metadata"))?;
    *pos += 8;
    Ok(u64::from_le_bytes(v.try_into().unwrap()))
}

fn read_str(b: &[u8], pos: &mut usize) -> Result<String> {
    let n = read_u64(b, pos)? as usize;
    let s = b.get(*pos..*pos + n).ok_or_else(|| anyhow!("truncated GGUF string"))?;
    *pos += n;
    Ok(String::from_utf8_lossy(s).into_owned())
}

fn write_str(out: &mut impl Write, s: &str) -> Result<()> {
    out.write_all(&(s.len() as u64).to_le_bytes())?;
    out.write_all(s.as_bytes())?;
    Ok(())
}

/// Advance past one metadata value of GGUF type `t`.
fn skip_value(b: &[u8], pos: &mut usize, t: u32) -> Result<()> {
    let fixed = |pos: &mut usize, n: usize| -> Result<()> {
        ensure!(*pos + n <= b.len(), "truncated GGUF metadata");
        *pos += n;
        Ok(())
    };
    match t {
        0 | 1 | 7 => fixed(pos, 1),
        2 | 3 => fixed(pos, 2),
        4..=6 => fixed(pos, 4),
        10..=12 => fixed(pos, 8),
        8 => { read_str(b, pos)?; Ok(()) }
        9 => {
            let et = read_u32(b, pos)?;
            let n = read_u64(b, pos)?;
            ensure!(et != 9, "nested GGUF arrays are not valid");
            for _ in 0..n { skip_value(b, pos, et)?; }
            Ok(())
        }
        _ => bail!("unknown GGUF metadata type {t}"),
    }
}
