//! Writing a GGUF: a copy of an existing file's metadata with new tensors, as a
//! trained model is saved in the format the engine loads.
//!
//! The metadata section is copied byte for byte from the source file, so the
//! tokenizer, the chat templates and every architecture key survive unchanged
//! whatever their types; the scalar values of named keys can be replaced. Tensor
//! data is written F32 or F16, aligned as the header declares.

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

/// Write `dst`: `src`'s metadata, with the F32 scalar keys in `set` replaced, and
/// `tensors` in the order given. Every key in `set` must exist in `src` as an F32.
pub fn write_gguf(src: &Path, dst: &Path, set: &[(&str, f32)], tensors: &[TensorOut<'_>]) -> Result<()> {
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
    for _ in 0..n_kv {
        let key = read_str(&kv, &mut pos)?;
        let t = read_u32(&kv, &mut pos)?;
        let value_at = pos;
        if key == "general.alignment" && t == 4 {
            align = read_u32(&kv, &mut { pos })? as u64;
        }
        ensure!(key != "split.count" || matches!(t, 2 | 4) && read_u32(&kv, &mut { pos }).is_ok_and(|n| n <= 1) || t == 0,
            "{} is a split file; write a single file", src.display());
        skip_value(&kv, &mut pos, t)?;
        if let Some(i) = set.iter().position(|(k, _)| *k == key) {
            ensure!(t == 6, "{key} is not an F32 scalar in {}", src.display());
            kv[value_at..value_at + 4].copy_from_slice(&set[i].1.to_le_bytes());
            replaced[i] = true;
        }
    }
    if let Some(i) = replaced.iter().position(|r| !r) {
        bail!("{} has no F32 key {}", src.display(), set[i].0);
    }
    kv.truncate(pos);
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
