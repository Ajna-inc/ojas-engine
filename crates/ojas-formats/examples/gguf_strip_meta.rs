//! Copy a GGUF file without the metadata keys that start with a prefix, byte for byte
//! otherwise: the remaining header entries, the tensor table and the tensor data are
//! unchanged, so the copy loads like the original minus those keys.
//!
//! Used to derive test models: a decision model's file without its `<arch>.decision.*`
//! keys is the plain generation model it was built from.
//!
//! usage: gguf_strip_meta <in.gguf> <out.gguf> <key prefix>

use anyhow::{bail, ensure, Context, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

fn u32_at(r: &mut impl Read) -> Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn u64_at(r: &mut impl Read) -> Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }

/// A length-prefixed string: its bytes as encoded, and its text.
fn string(r: &mut impl Read) -> Result<(Vec<u8>, String)> {
    let n = u64_at(r)?;
    ensure!(n <= 1 << 24, "GGUF string of {n} bytes");
    let mut bytes = vec![0u8; n as usize];
    r.read_exact(&mut bytes)?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let mut enc = n.to_le_bytes().to_vec();
    enc.extend_from_slice(&bytes);
    Ok((enc, text))
}

/// Bytes of one metadata value of GGUF type `t`, as encoded.
fn value(r: &mut impl Read, t: u32) -> Result<Vec<u8>> {
    let fixed = |n: usize, r: &mut dyn Read| -> Result<Vec<u8>> { let mut b = vec![0u8; n]; r.read_exact(&mut b)?; Ok(b) };
    Ok(match t {
        0 | 1 | 7 => fixed(1, r)?,
        2 | 3 => fixed(2, r)?,
        4..=6 => fixed(4, r)?,
        10..=12 => fixed(8, r)?,
        8 => string(r)?.0,
        9 => {
            let et = u32_at(r)?;
            let n = u64_at(r)?;
            ensure!(et != 9 && n <= 1 << 26, "GGUF array of type {et} with {n} items");
            let mut out = et.to_le_bytes().to_vec();
            out.extend_from_slice(&n.to_le_bytes());
            for _ in 0..n { out.extend(value(r, et)?); }
            out
        }
        other => bail!("unknown GGUF metadata type {other}"),
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(args.len() == 3, "usage: gguf_strip_meta <in.gguf> <out.gguf> <key prefix>");
    let (input, output, prefix) = (&args[0], &args[1], &args[2]);
    let mut r = BufReader::new(File::open(input).with_context(|| format!("opening {input}"))?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    ensure!(&magic == b"GGUF", "{input} is not a GGUF file");
    let version = u32_at(&mut r)?;
    ensure!(matches!(version, 2 | 3), "unsupported GGUF version {version}");
    let n_tensors = u64_at(&mut r)?;
    let n_kv = u64_at(&mut r)?;

    let mut kept: Vec<Vec<u8>> = Vec::new();
    let mut dropped = Vec::new();
    let mut alignment = 32u64;
    for _ in 0..n_kv {
        let (key_bytes, key) = string(&mut r)?;
        let t = u32_at(&mut r)?;
        let val = value(&mut r, t)?;
        if key == "general.alignment" && t == 4 { alignment = u32::from_le_bytes(val[..4].try_into()?) as u64; }
        if key.starts_with(prefix.as_str()) {
            dropped.push(key);
            continue;
        }
        let mut entry = key_bytes;
        entry.extend_from_slice(&t.to_le_bytes());
        entry.extend(val);
        kept.push(entry);
    }
    let mut tensor_table = Vec::new();
    for _ in 0..n_tensors {
        let (name, _) = string(&mut r)?;
        tensor_table.extend(name);
        let nd = u32_at(&mut r)?;
        tensor_table.extend_from_slice(&nd.to_le_bytes());
        for _ in 0..nd { tensor_table.extend_from_slice(&u64_at(&mut r)?.to_le_bytes()); }
        tensor_table.extend_from_slice(&u32_at(&mut r)?.to_le_bytes());
        tensor_table.extend_from_slice(&u64_at(&mut r)?.to_le_bytes());
    }
    let header_end = r.stream_position()?;
    let data_start = header_end.div_ceil(alignment) * alignment;

    let mut w = BufWriter::new(File::create(output).with_context(|| format!("creating {output}"))?);
    w.write_all(b"GGUF")?;
    w.write_all(&version.to_le_bytes())?;
    w.write_all(&n_tensors.to_le_bytes())?;
    w.write_all(&(kept.len() as u64).to_le_bytes())?;
    for entry in &kept { w.write_all(entry)?; }
    w.write_all(&tensor_table)?;
    let written = 4 + 4 + 8 + 8 + kept.iter().map(Vec::len).sum::<usize>() as u64 + tensor_table.len() as u64;
    let pad = written.div_ceil(alignment) * alignment - written;
    w.write_all(&vec![0u8; pad as usize])?;
    r.seek(SeekFrom::Start(data_start))?;
    std::io::copy(&mut r, &mut w)?;
    w.flush()?;
    println!("{output}: {} metadata keys kept, {} dropped ({})", kept.len(), dropped.len(), dropped.join(", "));
    Ok(())
}
