//! PyTorch checkpoints (`torch.save`, the zip format): the tensors of a state
//! dict, by dotted name, as f32 — without Python.
//!
//! A `.pth` is an uncompressed zip: `<root>/data.pkl` (a pickle whose tensors
//! are persistent references to storages) and `<root>/data/<key>` (each
//! storage's raw little-endian bytes). This reads the zip central directory,
//! runs the subset of the pickle VM that state dicts use (protocol 2: dicts,
//! OrderedDict, tuples, lists, scalars, `torch._utils._rebuild_tensor_v2` and
//! `rebuild_parameter`), and returns every tensor found in the nested dicts.

use std::collections::HashMap;

use anyhow::{anyhow, bail, ensure, Context, Result};

/// A tensor from a checkpoint, converted to f32 (ints and bools as numbers).
#[derive(Clone, Debug)]
pub struct PthTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)] // scalar payloads are parsed but only ints are read
enum V {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String, String),
    /// persistent storage: (dtype name, key)
    Storage(String, String),
    Tensor { storage: Box<V>, offset: usize, size: Vec<usize>, stride: Vec<usize> },
    /// anything else REDUCE / NEWOBJ built (kept so the stack stays balanced)
    Obj,
    Mark,
}

fn as_usizes(v: &V) -> Result<Vec<usize>> {
    match v {
        V::Tuple(xs) | V::List(xs) => xs.iter().map(|x| match x {
            V::Int(i) => Ok(*i as usize),
            _ => bail!("expected an int in a size / stride tuple"),
        }).collect(),
        _ => bail!("expected a size / stride tuple"),
    }
}

/// Stored (method 0) zip entries: name → (offset of data, size).
fn zip_entries(z: &[u8]) -> Result<HashMap<String, (usize, usize)>> {
    let u16_at = |o: usize| u16::from_le_bytes([z[o], z[o + 1]]) as usize;
    let u32_at = |o: usize| u32::from_le_bytes([z[o], z[o + 1], z[o + 2], z[o + 3]]) as usize;
    let u64_at = |o: usize| u64::from_le_bytes(z[o..o + 8].try_into().unwrap()) as usize;
    // end of central directory (no comment in torch files, but scan back anyway)
    let eocd = (0..z.len().saturating_sub(21)).rev().find(|&o| z[o..o + 4] == [0x50, 0x4b, 0x05, 0x06]).ok_or_else(|| anyhow!("not a zip file"))?;
    let (mut n, mut cd) = (u16_at(eocd + 10), u32_at(eocd + 16));
    if cd == 0xffff_ffff || n == 0xffff {
        // zip64: locator just before the EOCD points at the zip64 EOCD record
        let loc = eocd - 20;
        ensure!(z[loc..loc + 4] == [0x50, 0x4b, 0x06, 0x07], "zip64 locator missing");
        let rec = u64_at(loc + 8);
        n = u64_at(rec + 32);
        cd = u64_at(rec + 48);
    }
    let mut out = HashMap::new();
    let mut o = cd;
    for _ in 0..n {
        ensure!(z[o..o + 4] == [0x50, 0x4b, 0x01, 0x02], "bad central directory entry");
        let method = u16_at(o + 10);
        let (mut csize, mut size, mut local) = (u32_at(o + 20), u32_at(o + 24), u32_at(o + 42));
        let (nl, el, cl) = (u16_at(o + 28), u16_at(o + 30), u16_at(o + 32));
        let name = String::from_utf8_lossy(&z[o + 46..o + 46 + nl]).to_string();
        // zip64 extra field: sizes / offset that did not fit in 32 bits, in this order
        let mut e = o + 46 + nl;
        let end = e + el;
        while e + 4 <= end {
            let (id, len) = (u16_at(e), u16_at(e + 2));
            if id == 1 {
                let mut p = e + 4;
                if size == 0xffff_ffff { size = u64_at(p); p += 8; }
                if csize == 0xffff_ffff { csize = u64_at(p); p += 8; }
                if local == 0xffff_ffff { local = u64_at(p); }
            }
            e += 4 + len;
        }
        ensure!(method == 0 && csize == size, "zip entry {name} is compressed (method {method}); torch.save writes stored entries");
        let data = local + 30 + u16_at(local + 26) + u16_at(local + 28);
        out.insert(name, (data, size));
        o += 46 + nl + el + cl;
    }
    Ok(out)
}

/// Run the pickle, returning the top-level object.
fn unpickle(p: &[u8]) -> Result<V> {
    let mut stack: Vec<V> = vec![];
    let mut memo: HashMap<u32, V> = HashMap::new();
    let mut i = 0usize;
    let rd = |i: &mut usize, n: usize| -> Result<&[u8]> {
        ensure!(*i + n <= p.len(), "pickle truncated");
        let s = &p[*i..*i + n];
        *i += n;
        Ok(s)
    };
    let pop_mark = |stack: &mut Vec<V>| -> Result<Vec<V>> {
        let m = stack.iter().rposition(|v| matches!(v, V::Mark)).ok_or_else(|| anyhow!("pickle: no MARK"))?;
        let items = stack.split_off(m + 1);
        stack.pop();
        Ok(items)
    };
    loop {
        let op = rd(&mut i, 1)?[0];
        match op {
            0x80 => { rd(&mut i, 1)?; }                        // PROTO
            0x95 => { rd(&mut i, 8)?; }                        // FRAME
            b'.' => break,                                      // STOP
            b'(' => stack.push(V::Mark),                        // MARK
            b'N' => stack.push(V::None),
            0x88 => stack.push(V::Bool(true)),
            0x89 => stack.push(V::Bool(false)),
            b'}' => stack.push(V::Dict(vec![])),                // EMPTY_DICT
            b']' => stack.push(V::List(vec![])),                // EMPTY_LIST
            b')' => stack.push(V::Tuple(vec![])),               // EMPTY_TUPLE
            b'J' => { let b = rd(&mut i, 4)?; stack.push(V::Int(i32::from_le_bytes(b.try_into().unwrap()) as i64)); }
            b'K' => { let b = rd(&mut i, 1)?[0]; stack.push(V::Int(b as i64)); }
            b'M' => { let b = rd(&mut i, 2)?; stack.push(V::Int(u16::from_le_bytes([b[0], b[1]]) as i64)); }
            0x8a => {                                           // LONG1
                let n = rd(&mut i, 1)?[0] as usize;
                let b = rd(&mut i, n)?;
                let mut v: i64 = 0;
                for (k, &byte) in b.iter().enumerate().take(8) { v |= (byte as i64) << (8 * k); }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 { v -= 1i64 << (8 * n); }
                stack.push(V::Int(v));
            }
            b'G' => { let b = rd(&mut i, 8)?; stack.push(V::Float(f64::from_be_bytes(b.try_into().unwrap()))); }
            b'X' => { let n = u32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap()) as usize; stack.push(V::Str(String::from_utf8_lossy(rd(&mut i, n)?).to_string())); }
            0x8c => { let n = rd(&mut i, 1)?[0] as usize; stack.push(V::Str(String::from_utf8_lossy(rd(&mut i, n)?).to_string())); }
            b'C' => { let n = rd(&mut i, 1)?[0] as usize; rd(&mut i, n)?; stack.push(V::Obj); }          // SHORT_BINBYTES
            b'B' => { let n = u32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap()) as usize; rd(&mut i, n)?; stack.push(V::Obj); }
            b'q' => { let k = rd(&mut i, 1)?[0] as u32; memo.insert(k, stack.last().cloned().ok_or_else(|| anyhow!("BINPUT on empty stack"))?); }
            b'r' => { let k = u32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap()); memo.insert(k, stack.last().cloned().ok_or_else(|| anyhow!("LONG_BINPUT on empty stack"))?); }
            0x94 => { let k = memo.len() as u32; memo.insert(k, stack.last().cloned().ok_or_else(|| anyhow!("MEMOIZE on empty stack"))?); }
            b'h' => { let k = rd(&mut i, 1)?[0] as u32; stack.push(memo.get(&k).cloned().ok_or_else(|| anyhow!("BINGET {k} missing"))?); }
            b'j' => { let k = u32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap()); stack.push(memo.get(&k).cloned().ok_or_else(|| anyhow!("LONG_BINGET {k} missing"))?); }
            b't' => { let items = pop_mark(&mut stack)?; stack.push(V::Tuple(items)); }
            0x85 => { let a = stack.pop().unwrap(); stack.push(V::Tuple(vec![a])); }
            0x86 => { let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(V::Tuple(vec![a, b])); }
            0x87 => { let c = stack.pop().unwrap(); let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(V::Tuple(vec![a, b, c])); }
            b'l' => { let items = pop_mark(&mut stack)?; stack.push(V::List(items)); }
            b'a' => { let v = stack.pop().unwrap(); if let Some(V::List(l)) = stack.last_mut() { l.push(v); } }
            b'e' => { let items = pop_mark(&mut stack)?; if let Some(V::List(l)) = stack.last_mut() { l.extend(items); } }
            b's' => { let v = stack.pop().unwrap(); let k = stack.pop().unwrap(); if let Some(V::Dict(d)) = stack.last_mut() { d.push((k, v)); } }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                if let Some(V::Dict(d)) = stack.last_mut() {
                    for kv in items.chunks(2) { if let [k, v] = kv { d.push((k.clone(), v.clone())); } }
                }
            }
            b'c' => {                                           // GLOBAL "module\nname\n"
                let line = |i: &mut usize| -> Result<String> {
                    let s = *i;
                    while *i < p.len() && p[*i] != b'\n' { *i += 1; }
                    let v = String::from_utf8_lossy(&p[s..*i]).to_string();
                    *i += 1;
                    Ok(v)
                };
                let (m, n) = (line(&mut i)?, line(&mut i)?);
                stack.push(V::Global(m, n));
            }
            0x93 => {                                           // STACK_GLOBAL
                let n = stack.pop().unwrap();
                let m = stack.pop().unwrap();
                match (m, n) { (V::Str(m), V::Str(n)) => stack.push(V::Global(m, n)), _ => bail!("STACK_GLOBAL wants two strings") }
            }
            b'Q' => {                                           // BINPERSID
                let pid = stack.pop().unwrap();
                let V::Tuple(t) = pid else { bail!("persistent id is not a tuple") };
                // ('storage', <class FloatStorage>, key, location, numel)
                let dtype = match t.get(1) { Some(V::Global(_, n)) => n.clone(), _ => bail!("persistent id without a storage class") };
                let key = match t.get(2) { Some(V::Str(k)) => k.clone(), _ => bail!("persistent id without a key") };
                stack.push(V::Storage(dtype, key));
            }
            b'R' | 0x81 => {                                    // REDUCE / NEWOBJ
                let args = stack.pop().unwrap();
                let f = stack.pop().unwrap();
                let V::Tuple(a) = args else { stack.push(V::Obj); continue };
                let v = match &f {
                    V::Global(m, n) if m == "torch._utils" && n == "_rebuild_tensor_v2" => V::Tensor {
                        storage: Box::new(a[0].clone()),
                        offset: match a[1] { V::Int(o) => o as usize, _ => 0 },
                        size: as_usizes(&a[2])?,
                        stride: as_usizes(&a[3])?,
                    },
                    // Parameter(tensor, requires_grad, hooks) → the tensor
                    V::Global(m, n) if m == "torch._utils" && n == "_rebuild_parameter" => a[0].clone(),
                    V::Global(m, n) if m == "collections" && n == "OrderedDict" => V::Dict(vec![]),
                    _ => V::Obj,
                };
                stack.push(v);
            }
            b'b' => { stack.pop(); }                            // BUILD: state is ignored (dicts are filled by SETITEMS)
            other => bail!("pickle opcode 0x{other:02x} not supported at byte {}", i - 1),
        }
    }
    stack.pop().ok_or_else(|| anyhow!("empty pickle"))
}

fn walk(prefix: &str, v: &V, out: &mut Vec<(String, V)>) {
    match v {
        V::Dict(d) => {
            for (k, x) in d {
                let key = match k { V::Str(s) => s.clone(), V::Int(i) => i.to_string(), _ => continue };
                let name = if prefix.is_empty() { key } else { format!("{prefix}.{key}") };
                walk(&name, x, out);
            }
        }
        V::Tensor { .. } => out.push((prefix.to_string(), v.clone())),
        _ => {}
    }
}

/// Every tensor in the checkpoint, by dotted path through its nested dicts
/// (e.g. `model.backbone.conv1.weight`, `ema.module.…`).
pub fn load(bytes: &[u8]) -> Result<Vec<(String, PthTensor)>> {
    let zip = zip_entries(bytes)?;
    let pkl = zip.keys().find(|k| k.ends_with("/data.pkl") || *k == "data.pkl").cloned().ok_or_else(|| anyhow!("no data.pkl in the checkpoint"))?;
    let root = pkl.trim_end_matches("data.pkl").to_string();
    let (po, pn) = zip[&pkl];
    let top = unpickle(&bytes[po..po + pn]).context("reading data.pkl")?;
    let mut found = vec![];
    walk("", &top, &mut found);
    let mut out = vec![];
    for (name, t) in found {
        let V::Tensor { storage, offset, size, stride } = t else { continue };
        let V::Storage(dtype, key) = *storage else { bail!("{name}: tensor without a storage") };
        let (o, n) = *zip.get(&format!("{root}data/{key}")).ok_or_else(|| anyhow!("{name}: storage {key} missing"))?;
        let raw = &bytes[o..o + n];
        let (elem, conv): (usize, fn(&[u8]) -> f32) = match dtype.as_str() {
            "FloatStorage" => (4, |b| f32::from_le_bytes(b.try_into().unwrap())),
            "DoubleStorage" => (8, |b| f64::from_le_bytes(b.try_into().unwrap()) as f32),
            "HalfStorage" => (2, |b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()),
            "BFloat16Storage" => (2, |b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32()),
            "LongStorage" => (8, |b| i64::from_le_bytes(b.try_into().unwrap()) as f32),
            "IntStorage" => (4, |b| i32::from_le_bytes(b.try_into().unwrap()) as f32),
            "BoolStorage" | "ByteStorage" => (1, |b| b[0] as f32),
            other => bail!("{name}: storage type {other} not supported"),
        };
        let numel: usize = size.iter().product();
        let mut data = Vec::with_capacity(numel);
        // strided gather (state-dict tensors are almost always contiguous)
        for lin in 0..numel {
            let (mut rem, mut e) = (lin, offset);
            for d in (0..size.len()).rev() {
                e += rem % size[d] * stride[d];
                rem /= size[d];
            }
            ensure!((e + 1) * elem <= raw.len(), "{name}: element {e} outside storage {key}");
            data.push(conv(&raw[e * elem..(e + 1) * elem]));
        }
        out.push((name, PthTensor { shape: size, data }));
    }
    Ok(out)
}
