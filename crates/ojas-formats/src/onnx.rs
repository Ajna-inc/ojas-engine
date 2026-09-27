//! Minimal ONNX reader — a hand-rolled protobuf wire decoder, enough to load
//! the graph, initializers and IO shapes of detector / OCR exports (Ultralytics,
//! paddle2onnx). No `prost`, no `protoc`: same policy as `gguf.rs`.
//!
//! Refused with named errors rather than parsed: external tensor data,
//! subgraph attributes (`If`/`Loop`/`Scan`), and node domains other than the
//! default `ai.onnx`.

use anyhow::{anyhow, bail, ensure, Result};

// ---------------------------------------------------------------------------
// Data types (TensorProto.DataType). Only what the vision exports use.
// ---------------------------------------------------------------------------

pub const DT_FLOAT: u32 = 1;
pub const DT_UINT8: u32 = 2;
pub const DT_INT8: u32 = 3;
pub const DT_INT32: u32 = 6;
pub const DT_INT64: u32 = 7;
pub const DT_BOOL: u32 = 9;
pub const DT_FLOAT16: u32 = 10;
pub const DT_DOUBLE: u32 = 11;

pub fn dtype_name(dt: u32) -> &'static str {
    match dt {
        DT_FLOAT => "f32",
        DT_UINT8 => "u8",
        DT_INT8 => "i8",
        DT_INT32 => "i32",
        DT_INT64 => "i64",
        DT_BOOL => "bool",
        DT_FLOAT16 => "f16",
        DT_DOUBLE => "f64",
        _ => "unsupported",
    }
}

// ---------------------------------------------------------------------------
// Model structs
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub producer_name: String,
    /// (domain, version) pairs from `opset_import`. The default domain is "".
    pub opsets: Vec<(String, i64)>,
    pub graph: OnnxGraph,
}

#[derive(Debug, Default)]
pub struct OnnxGraph {
    pub name: String,
    pub nodes: Vec<OnnxNode>,
    pub initializers: Vec<OnnxTensor>,
    pub inputs: Vec<OnnxValueInfo>,
    pub outputs: Vec<OnnxValueInfo>,
    pub value_info: Vec<OnnxValueInfo>,
}

#[derive(Debug, Default)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<OnnxAttr>,
}

#[derive(Debug)]
pub struct OnnxAttr {
    pub name: String,
    pub value: AttrValue,
}

#[derive(Debug)]
pub enum AttrValue {
    F(f32),
    I(i64),
    S(String),
    T(OnnxTensor),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

/// One tensor's payload, kept in whichever encoding the file used. Accessors
/// below convert on demand; nothing is eagerly widened.
#[derive(Debug)]
enum TensorData {
    /// `raw_data`: little-endian bytes of `dtype`.
    Raw(Vec<u8>),
    /// `float_data` (f32) — also used by FLOAT16 models that store widened values.
    Floats(Vec<f32>),
    /// `int32_data` (i32/i8/u8/bool stored widened).
    Ints32(Vec<i32>),
    /// `int64_data`.
    Ints64(Vec<i64>),
    Empty,
}

#[derive(Debug)]
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<i64>,
    pub dtype: u32,
    data: TensorData,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OnnxDim {
    /// A concrete extent.
    Value(i64),
    /// A named symbolic extent (e.g. "batch"); the caller must bind it.
    Param(String),
    /// Present but neither value nor param (rare, from lax exporters).
    Unknown,
}

#[derive(Debug)]
pub struct OnnxValueInfo {
    pub name: String,
    pub elem_type: u32,
    pub dims: Vec<OnnxDim>,
}

impl OnnxModel {
    /// Version of the default (`ai.onnx`) opset, if declared.
    pub fn default_opset(&self) -> Option<i64> {
        self.opsets
            .iter()
            .find(|(d, _)| d.is_empty() || d == "ai.onnx")
            .map(|(_, v)| *v)
    }

    /// Sorted (op_type, count) histogram over the graph's nodes.
    pub fn op_histogram(&self) -> Vec<(String, usize)> {
        let mut counts = std::collections::HashMap::<&str, usize>::new();
        for n in &self.graph.nodes {
            *counts.entry(n.op_type.as_str()).or_default() += 1;
        }
        let mut v: Vec<(String, usize)> = counts.into_iter().map(|(k, c)| (k.to_string(), c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }

    /// Total bytes of initializer payloads (parameter size on disk).
    pub fn parameter_bytes(&self) -> u64 {
        self.graph.initializers.iter().map(|t| t.payload_bytes() as u64).sum()
    }
}

impl OnnxNode {
    pub fn attr(&self, name: &str) -> Option<&AttrValue> {
        self.attrs.iter().find(|a| a.name == name).map(|a| &a.value)
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: renormalize
            let shift = frac.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((frac << (shift + 1)) & 0x7f_ffff)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000,
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

impl OnnxTensor {
    /// An f32 tensor built in memory (synthetic test graphs; no protobuf needed).
    pub fn from_f32(name: &str, dims: &[i64], data: Vec<f32>) -> OnnxTensor {
        OnnxTensor { name: name.into(), dims: dims.to_vec(), dtype: DT_FLOAT, data: TensorData::Floats(data) }
    }

    pub fn element_count(&self) -> Result<usize> {
        let mut n: usize = 1;
        for &d in &self.dims {
            ensure!(d >= 0, "ONNX tensor {}: negative dim {d}", self.name);
            n = n
                .checked_mul(usize::try_from(d)?)
                .ok_or_else(|| anyhow!("ONNX tensor {}: dim overflow", self.name))?;
        }
        Ok(n)
    }

    fn payload_bytes(&self) -> usize {
        match &self.data {
            TensorData::Raw(b) => b.len(),
            TensorData::Floats(v) => v.len() * 4,
            TensorData::Ints32(v) => v.len() * 4,
            TensorData::Ints64(v) => v.len() * 8,
            TensorData::Empty => 0,
        }
    }

    /// Payload as f32 (FLOAT, FLOAT16 and DOUBLE convert; anything else errors).
    pub fn f32_data(&self) -> Result<Vec<f32>> {
        let n = self.element_count()?;
        let out = match (&self.data, self.dtype) {
            (TensorData::Floats(v), DT_FLOAT | DT_FLOAT16) => v.clone(),
            (TensorData::Raw(b), DT_FLOAT) => {
                ensure!(b.len() == n * 4, "ONNX tensor {}: raw f32 size mismatch", self.name);
                b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
            }
            (TensorData::Raw(b), DT_FLOAT16) => {
                ensure!(b.len() == n * 2, "ONNX tensor {}: raw f16 size mismatch", self.name);
                b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect()
            }
            (TensorData::Raw(b), DT_DOUBLE) => {
                ensure!(b.len() == n * 8, "ONNX tensor {}: raw f64 size mismatch", self.name);
                b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect()
            }
            (TensorData::Ints32(v), DT_FLOAT16) => v.iter().map(|&x| f16_to_f32(x as u16)).collect(),
            (TensorData::Empty, _) if n == 0 => Vec::new(),
            _ => bail!(
                "ONNX tensor {}: cannot read dtype {} as f32",
                self.name,
                dtype_name(self.dtype)
            ),
        };
        ensure!(out.len() == n, "ONNX tensor {}: element count mismatch ({} vs {n})", self.name, out.len());
        Ok(out)
    }

    /// Payload as i64 (INT64, INT32, INT8, UINT8 and BOOL widen; else error).
    pub fn i64_data(&self) -> Result<Vec<i64>> {
        let n = self.element_count()?;
        let out = match (&self.data, self.dtype) {
            (TensorData::Ints64(v), DT_INT64) => v.clone(),
            (TensorData::Ints32(v), DT_INT32) => v.iter().map(|&x| x as i64).collect(),
            (TensorData::Ints32(v), DT_INT8) => v.iter().map(|&x| x as i8 as i64).collect(),
            (TensorData::Ints32(v), DT_UINT8 | DT_BOOL) => v.iter().map(|&x| x as u8 as i64).collect(),
            (TensorData::Raw(b), DT_INT64) => {
                ensure!(b.len() == n * 8, "ONNX tensor {}: raw i64 size mismatch", self.name);
                b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
            }
            (TensorData::Raw(b), DT_INT32) => {
                ensure!(b.len() == n * 4, "ONNX tensor {}: raw i32 size mismatch", self.name);
                b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect()
            }
            (TensorData::Raw(b), DT_INT8) => b.iter().map(|&x| x as i8 as i64).collect(),
            (TensorData::Raw(b), DT_UINT8 | DT_BOOL) => b.iter().map(|&x| x as i64).collect(),
            (TensorData::Empty, _) if n == 0 => Vec::new(),
            _ => bail!(
                "ONNX tensor {}: cannot read dtype {} as i64",
                self.name,
                dtype_name(self.dtype)
            ),
        };
        ensure!(out.len() == n, "ONNX tensor {}: element count mismatch ({} vs {n})", self.name, out.len());
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Protobuf wire decoding. Slice-based: every length is validated against the
// remaining bytes before any allocation, so a corrupt length cannot balloon.
// ---------------------------------------------------------------------------

const WIRE_VARINT: u8 = 0;
const WIRE_FIXED64: u8 = 1;
const WIRE_LEN: u8 = 2;
const WIRE_FIXED32: u8 = 5;

/// Item-count ceilings, same spirit as gguf.rs's header budget: a corrupt
/// count must not drive huge allocations or quadratic work.
const MAX_ITEMS: usize = 4_194_304;
const MAX_MODEL_BYTES: u64 = 2 * 1024 * 1024 * 1024; // protobuf practical limit

struct Pb<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Pb<'a> {
    fn new(b: &'a [u8]) -> Self {
        Pb { b, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.b.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut out: u64 = 0;
        for shift in (0..64).step_by(7) {
            let byte = *self.b.get(self.pos).ok_or_else(|| anyhow!("ONNX: truncated varint"))?;
            self.pos += 1;
            out |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(out);
            }
        }
        bail!("ONNX: varint longer than 10 bytes")
    }

    /// Next (field number, wire type), or None at end of the slice.
    fn tag(&mut self) -> Result<Option<(u32, u8)>> {
        if self.done() {
            return Ok(None);
        }
        let key = self.varint()?;
        let wire = (key & 7) as u8;
        let field = u32::try_from(key >> 3).map_err(|_| anyhow!("ONNX: field number overflow"))?;
        ensure!(field != 0, "ONNX: field number 0");
        Ok(Some((field, wire)))
    }

    fn fixed32(&mut self) -> Result<u32> {
        let end = self.pos.checked_add(4).filter(|&e| e <= self.b.len())
            .ok_or_else(|| anyhow!("ONNX: truncated fixed32"))?;
        let v = u32::from_le_bytes(self.b[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v)
    }

    fn fixed64(&mut self) -> Result<u64> {
        let end = self.pos.checked_add(8).filter(|&e| e <= self.b.len())
            .ok_or_else(|| anyhow!("ONNX: truncated fixed64"))?;
        let v = u64::from_le_bytes(self.b[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v)
    }

    /// Length-delimited payload as a subslice.
    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = usize::try_from(self.varint()?).map_err(|_| anyhow!("ONNX: length overflow"))?;
        let end = self.pos.checked_add(n).filter(|&e| e <= self.b.len())
            .ok_or_else(|| anyhow!("ONNX: length-delimited field exceeds buffer"))?;
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn string(&mut self) -> Result<String> {
        Ok(String::from_utf8_lossy(self.bytes()?).into_owned())
    }

    fn skip(&mut self, wire: u8) -> Result<()> {
        match wire {
            WIRE_VARINT => {
                self.varint()?;
            }
            WIRE_FIXED64 => {
                self.fixed64()?;
            }
            WIRE_LEN => {
                self.bytes()?;
            }
            WIRE_FIXED32 => {
                self.fixed32()?;
            }
            w => bail!("ONNX: unsupported wire type {w} (group encoding?)"),
        }
        Ok(())
    }
}

/// Repeated scalar fields arrive packed (one length-delimited blob) or as
/// individual entries; proto3 readers must accept both.
fn read_packed_i64(p: &mut Pb<'_>, wire: u8, out: &mut Vec<i64>) -> Result<()> {
    match wire {
        WIRE_VARINT => out.push(p.varint()? as i64),
        WIRE_LEN => {
            let mut inner = Pb::new(p.bytes()?);
            while !inner.done() {
                ensure!(out.len() < MAX_ITEMS, "ONNX: packed int64 field exceeds item budget");
                out.push(inner.varint()? as i64);
            }
        }
        w => bail!("ONNX: unexpected wire type {w} for int64 field"),
    }
    Ok(())
}

fn read_packed_f32(p: &mut Pb<'_>, wire: u8, out: &mut Vec<f32>) -> Result<()> {
    match wire {
        WIRE_FIXED32 => out.push(f32::from_bits(p.fixed32()?)),
        WIRE_LEN => {
            let b = p.bytes()?;
            ensure!(b.len() % 4 == 0, "ONNX: packed float field has ragged length");
            out.reserve(b.len() / 4);
            for c in b.chunks_exact(4) {
                out.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
        }
        w => bail!("ONNX: unexpected wire type {w} for float field"),
    }
    Ok(())
}

fn read_packed_i32(p: &mut Pb<'_>, wire: u8, out: &mut Vec<i32>) -> Result<()> {
    match wire {
        WIRE_VARINT => out.push(p.varint()? as i64 as i32),
        WIRE_LEN => {
            let mut inner = Pb::new(p.bytes()?);
            while !inner.done() {
                ensure!(out.len() < MAX_ITEMS, "ONNX: packed int32 field exceeds item budget");
                out.push(inner.varint()? as i64 as i32);
            }
        }
        w => bail!("ONNX: unexpected wire type {w} for int32 field"),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Message parsers. Field numbers follow onnx.proto (IR spec); unknown fields
// are skipped so newer exporters keep loading.
// ---------------------------------------------------------------------------

/// TensorProto: dims=1, data_type=2, float_data=4, int32_data=5, int64_data=7,
/// name=8, raw_data=9, external_data=13, data_location=14.
fn read_tensor(b: &[u8]) -> Result<OnnxTensor> {
    let mut p = Pb::new(b);
    let mut t = OnnxTensor { name: String::new(), dims: Vec::new(), dtype: 0, data: TensorData::Empty };
    let mut has_external = false;
    while let Some((field, wire)) = p.tag()? {
        match field {
            1 => read_packed_i64(&mut p, wire, &mut t.dims)?,
            2 => t.dtype = p.varint()? as u32,
            4 => {
                let mut v = match std::mem::replace(&mut t.data, TensorData::Empty) {
                    TensorData::Floats(v) => v,
                    _ => Vec::new(),
                };
                read_packed_f32(&mut p, wire, &mut v)?;
                t.data = TensorData::Floats(v);
            }
            5 => {
                let mut v = match std::mem::replace(&mut t.data, TensorData::Empty) {
                    TensorData::Ints32(v) => v,
                    _ => Vec::new(),
                };
                read_packed_i32(&mut p, wire, &mut v)?;
                t.data = TensorData::Ints32(v);
            }
            7 => {
                let mut v = match std::mem::replace(&mut t.data, TensorData::Empty) {
                    TensorData::Ints64(v) => v,
                    _ => Vec::new(),
                };
                read_packed_i64(&mut p, wire, &mut v)?;
                t.data = TensorData::Ints64(v);
            }
            8 => t.name = p.string()?,
            9 => t.data = TensorData::Raw(p.bytes()?.to_vec()),
            13 => {
                p.skip(wire)?;
                has_external = true;
            }
            14 => {
                if p.varint()? != 0 {
                    has_external = true;
                }
            }
            _ => p.skip(wire)?,
        }
    }
    ensure!(
        !has_external,
        "ONNX tensor {}: external data is not supported — re-export with weights embedded (save_as_external_data=False)",
        t.name
    );
    ensure!(t.dims.len() <= 8, "ONNX tensor {}: rank {} exceeds limit", t.name, t.dims.len());
    Ok(t)
}

/// AttributeProto: name=1, f=2, i=3, s=4, t=5, g=6, floats=7, ints=8,
/// strings=9, type=20. Subgraph attributes (g / graphs) are refused.
fn read_attr(b: &[u8], node: &str) -> Result<OnnxAttr> {
    let mut p = Pb::new(b);
    let mut name = String::new();
    let mut value: Option<AttrValue> = None;
    let mut floats: Vec<f32> = Vec::new();
    let mut ints: Vec<i64> = Vec::new();
    let mut strings: Vec<String> = Vec::new();
    let mut declared_type: Option<u64> = None;
    while let Some((field, wire)) = p.tag()? {
        match field {
            1 => name = p.string()?,
            2 => value = Some(AttrValue::F(f32::from_bits(p.fixed32()?))),
            3 => value = Some(AttrValue::I(p.varint()? as i64)),
            4 => value = Some(AttrValue::S(p.string()?)),
            5 => value = Some(AttrValue::T(read_tensor(p.bytes()?)?)),
            6 | 11 => bail!(
                "ONNX node {node}: subgraph attribute {name:?} — control flow (If/Loop/Scan) is not supported"
            ),
            7 => read_packed_f32(&mut p, wire, &mut floats)?,
            8 => read_packed_i64(&mut p, wire, &mut ints)?,
            9 => strings.push(p.string()?),
            20 => declared_type = Some(p.varint()?),
            _ => p.skip(wire)?,
        }
    }
    // AttributeType: FLOATS=6, INTS=7, STRINGS=8. An empty declared list is
    // still that list (e.g. `pads: INTS []`), so honour the declared type
    // before falling back on populated repeated fields.
    let value = match (value, declared_type) {
        (Some(v), _) => v,
        (None, Some(6)) => AttrValue::Floats(floats),
        (None, Some(7)) => AttrValue::Ints(ints),
        (None, Some(8)) => AttrValue::Strings(strings),
        (None, _) if !floats.is_empty() => AttrValue::Floats(floats),
        (None, _) if !ints.is_empty() => AttrValue::Ints(ints),
        (None, _) if !strings.is_empty() => AttrValue::Strings(strings),
        (None, t) => bail!("ONNX node {node}: attribute {name:?} has unsupported type {t:?}"),
    };
    Ok(OnnxAttr { name, value })
}

/// NodeProto: input=1, output=2, name=3, op_type=4, attribute=5, domain=7.
fn read_node(b: &[u8]) -> Result<OnnxNode> {
    let mut p = Pb::new(b);
    let mut n = OnnxNode::default();
    let mut attr_slices: Vec<&[u8]> = Vec::new();
    while let Some((field, wire)) = p.tag()? {
        match field {
            1 => n.inputs.push(p.string()?),
            2 => n.outputs.push(p.string()?),
            3 => n.name = p.string()?,
            4 => n.op_type = p.string()?,
            5 => attr_slices.push(p.bytes()?),
            7 => n.domain = p.string()?,
            _ => p.skip(wire)?,
        }
    }
    ensure!(
        n.domain.is_empty() || n.domain == "ai.onnx",
        "ONNX node {} ({}): domain {:?} is not supported (only the default ai.onnx domain)",
        n.name,
        n.op_type,
        n.domain
    );
    let label = if n.name.is_empty() { n.op_type.clone() } else { n.name.clone() };
    for s in attr_slices {
        n.attrs.push(read_attr(s, &label)?);
    }
    Ok(n)
}

/// ValueInfoProto(name=1, type=2) → TypeProto(tensor_type=1)
/// → Tensor(elem_type=1, shape=2) → TensorShapeProto(dim=1)
/// → Dimension(dim_value=1, dim_param=2).
fn read_value_info(b: &[u8]) -> Result<OnnxValueInfo> {
    let mut p = Pb::new(b);
    let mut v = OnnxValueInfo { name: String::new(), elem_type: 0, dims: Vec::new() };
    while let Some((field, wire)) = p.tag()? {
        match field {
            1 => v.name = p.string()?,
            2 => {
                let mut tp = Pb::new(p.bytes()?);
                while let Some((f2, w2)) = tp.tag()? {
                    if f2 != 1 {
                        tp.skip(w2)?; // sequence/map/optional types: not tensors
                        continue;
                    }
                    let mut tt = Pb::new(tp.bytes()?);
                    while let Some((f3, w3)) = tt.tag()? {
                        match f3 {
                            1 => v.elem_type = tt.varint()? as u32,
                            2 => {
                                let mut sh = Pb::new(tt.bytes()?);
                                while let Some((f4, w4)) = sh.tag()? {
                                    if f4 != 1 {
                                        sh.skip(w4)?;
                                        continue;
                                    }
                                    let mut dp = Pb::new(sh.bytes()?);
                                    let mut dim = OnnxDim::Unknown;
                                    while let Some((f5, w5)) = dp.tag()? {
                                        match f5 {
                                            1 => dim = OnnxDim::Value(dp.varint()? as i64),
                                            2 => dim = OnnxDim::Param(dp.string()?),
                                            _ => dp.skip(w5)?,
                                        }
                                    }
                                    v.dims.push(dim);
                                }
                            }
                            _ => tt.skip(w3)?,
                        }
                    }
                }
            }
            _ => p.skip(wire)?,
        }
    }
    Ok(v)
}

/// GraphProto: node=1, name=2, initializer=5, input=11, output=12, value_info=13.
fn read_graph(b: &[u8]) -> Result<OnnxGraph> {
    let mut p = Pb::new(b);
    let mut g = OnnxGraph::default();
    while let Some((field, wire)) = p.tag()? {
        match field {
            1 => {
                ensure!(g.nodes.len() < MAX_ITEMS, "ONNX graph exceeds node budget");
                g.nodes.push(read_node(p.bytes()?)?);
            }
            2 => g.name = p.string()?,
            5 => {
                ensure!(g.initializers.len() < MAX_ITEMS, "ONNX graph exceeds initializer budget");
                g.initializers.push(read_tensor(p.bytes()?)?);
            }
            11 => g.inputs.push(read_value_info(p.bytes()?)?),
            12 => g.outputs.push(read_value_info(p.bytes()?)?),
            13 => g.value_info.push(read_value_info(p.bytes()?)?),
            _ => p.skip(wire)?,
        }
    }
    Ok(g)
}

/// ModelProto: ir_version=1, producer_name=2, graph=7, opset_import=8;
/// OperatorSetIdProto: domain=1, version=2.
pub fn parse(bytes: &[u8]) -> Result<OnnxModel> {
    let mut p = Pb::new(bytes);
    let mut ir_version = 0i64;
    let mut producer_name = String::new();
    let mut opsets: Vec<(String, i64)> = Vec::new();
    let mut graph: Option<OnnxGraph> = None;
    while let Some((field, wire)) = p.tag()? {
        match field {
            1 => ir_version = p.varint()? as i64,
            2 => producer_name = p.string()?,
            7 => graph = Some(read_graph(p.bytes()?)?),
            8 => {
                let mut op = Pb::new(p.bytes()?);
                let (mut domain, mut version) = (String::new(), 0i64);
                while let Some((f2, w2)) = op.tag()? {
                    match f2 {
                        1 => domain = op.string()?,
                        2 => version = op.varint()? as i64,
                        _ => op.skip(w2)?,
                    }
                }
                opsets.push((domain, version));
            }
            _ => p.skip(wire)?,
        }
    }
    let graph = graph.ok_or_else(|| anyhow!("ONNX model has no graph"))?;
    ensure!(ir_version > 0, "ONNX model has no ir_version — not an ONNX file?");
    for (domain, version) in &opsets {
        ensure!(
            domain.is_empty() || domain == "ai.onnx",
            "ONNX model imports unsupported operator domain {domain:?} v{version}"
        );
    }
    Ok(OnnxModel { ir_version, producer_name, opsets, graph })
}

pub fn load(path: &str) -> Result<OnnxModel> {
    let len = std::fs::metadata(path)?.len();
    ensure!(len <= MAX_MODEL_BYTES, "ONNX file {path} is {len} bytes, over the 2 GiB limit");
    let bytes = std::fs::read(path)?;
    parse(&bytes).map_err(|e| anyhow!("{path}: {e}"))
}

// ---------------------------------------------------------------------------
// Tests build models from synthetic protobuf bytes — no downloads.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal wire writer for constructing fixtures.
    fn vint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                break;
            }
            out.push(b | 0x80);
        }
    }
    fn tag(out: &mut Vec<u8>, field: u32, wire: u8) {
        vint(out, ((field as u64) << 3) | wire as u64);
    }
    fn put_len(out: &mut Vec<u8>, field: u32, payload: &[u8]) {
        tag(out, field, 2);
        vint(out, payload.len() as u64);
        out.extend_from_slice(payload);
    }
    fn put_str(out: &mut Vec<u8>, field: u32, s: &str) {
        put_len(out, field, s.as_bytes());
    }
    fn put_varint(out: &mut Vec<u8>, field: u32, v: u64) {
        tag(out, field, 0);
        vint(out, v);
    }

    fn value_info(name: &str, elem: u32, dims: &[Result<i64, &str>]) -> Vec<u8> {
        let mut shape = Vec::new();
        for d in dims {
            let mut dim = Vec::new();
            match d {
                Ok(v) => put_varint(&mut dim, 1, *v as u64),
                Err(p) => put_str(&mut dim, 2, p),
            }
            put_len(&mut shape, 1, &dim);
        }
        let mut tensor_type = Vec::new();
        put_varint(&mut tensor_type, 1, elem as u64);
        put_len(&mut tensor_type, 2, &shape);
        let mut ty = Vec::new();
        put_len(&mut ty, 1, &tensor_type);
        let mut vi = Vec::new();
        put_str(&mut vi, 1, name);
        put_len(&mut vi, 2, &ty);
        vi
    }

    fn f32_tensor_raw(name: &str, dims: &[i64], vals: &[f32]) -> Vec<u8> {
        let mut t = Vec::new();
        let mut packed_dims = Vec::new();
        for &d in dims {
            vint(&mut packed_dims, d as u64);
        }
        put_len(&mut t, 1, &packed_dims);
        put_varint(&mut t, 2, DT_FLOAT as u64);
        put_str(&mut t, 8, name);
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        put_len(&mut t, 9, &raw);
        t
    }

    fn conv_model() -> Vec<u8> {
        // attrs: strides=[2,2] (packed ints), group=1 (int), auto_pad="NOTSET"
        let mut a_strides = Vec::new();
        put_str(&mut a_strides, 1, "strides");
        let mut packed = Vec::new();
        vint(&mut packed, 2);
        vint(&mut packed, 2);
        put_len(&mut a_strides, 8, &packed);
        put_varint(&mut a_strides, 20, 7); // INTS

        let mut a_group = Vec::new();
        put_str(&mut a_group, 1, "group");
        put_varint(&mut a_group, 3, 1);
        put_varint(&mut a_group, 20, 2); // INT

        let mut a_pad = Vec::new();
        put_str(&mut a_pad, 1, "auto_pad");
        put_str(&mut a_pad, 4, "NOTSET");
        put_varint(&mut a_pad, 20, 3); // STRING

        let mut node = Vec::new();
        put_str(&mut node, 1, "images");
        put_str(&mut node, 1, "w");
        put_str(&mut node, 2, "conv_out");
        put_str(&mut node, 3, "conv0");
        put_str(&mut node, 4, "Conv");
        put_len(&mut node, 5, &a_strides);
        put_len(&mut node, 5, &a_group);
        put_len(&mut node, 5, &a_pad);

        let mut graph = Vec::new();
        put_len(&mut graph, 1, &node);
        put_str(&mut graph, 2, "g");
        put_len(&mut graph, 5, &f32_tensor_raw("w", &[1, 3, 1, 1], &[0.5, -1.0, 2.0]));
        put_len(&mut graph, 11, &value_info("images", DT_FLOAT, &[Err("batch"), Ok(3), Ok(640), Ok(640)]));
        put_len(&mut graph, 12, &value_info("conv_out", DT_FLOAT, &[Err("batch"), Ok(1), Ok(320), Ok(320)]));

        let mut opset = Vec::new();
        put_str(&mut opset, 1, "");
        put_varint(&mut opset, 2, 17);

        let mut model = Vec::new();
        put_varint(&mut model, 1, 8); // ir_version
        put_str(&mut model, 2, "test-exporter");
        put_len(&mut model, 7, &graph);
        put_len(&mut model, 8, &opset);
        model
    }

    #[test]
    fn parses_model_graph_and_attrs() {
        let m = parse(&conv_model()).unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "test-exporter");
        assert_eq!(m.default_opset(), Some(17));
        assert_eq!(m.graph.nodes.len(), 1);
        let n = &m.graph.nodes[0];
        assert_eq!(n.op_type, "Conv");
        assert_eq!(n.inputs, vec!["images", "w"]);
        assert_eq!(n.outputs, vec!["conv_out"]);
        match n.attr("strides") {
            Some(AttrValue::Ints(v)) => assert_eq!(v, &[2, 2]),
            other => panic!("strides: {other:?}"),
        }
        match n.attr("group") {
            Some(AttrValue::I(1)) => {}
            other => panic!("group: {other:?}"),
        }
        match n.attr("auto_pad") {
            Some(AttrValue::S(s)) => assert_eq!(s, "NOTSET"),
            other => panic!("auto_pad: {other:?}"),
        }
        assert_eq!(m.op_histogram(), vec![("Conv".to_string(), 1)]);
    }

    #[test]
    fn reads_io_shapes_with_dim_params() {
        let m = parse(&conv_model()).unwrap();
        let input = &m.graph.inputs[0];
        assert_eq!(input.name, "images");
        assert_eq!(input.elem_type, DT_FLOAT);
        assert_eq!(
            input.dims,
            vec![OnnxDim::Param("batch".into()), OnnxDim::Value(3), OnnxDim::Value(640), OnnxDim::Value(640)]
        );
    }

    #[test]
    fn initializer_raw_f32_roundtrip() {
        let m = parse(&conv_model()).unwrap();
        let w = &m.graph.initializers[0];
        assert_eq!(w.name, "w");
        assert_eq!(w.dims, vec![1, 3, 1, 1]);
        assert_eq!(w.f32_data().unwrap(), vec![0.5, -1.0, 2.0]);
        assert_eq!(m.parameter_bytes(), 12);
    }

    #[test]
    fn typed_field_variants_roundtrip() {
        // float_data (packed) instead of raw_data
        let mut t = Vec::new();
        let mut dims = Vec::new();
        vint(&mut dims, 2);
        put_len(&mut t, 1, &dims);
        put_varint(&mut t, 2, DT_FLOAT as u64);
        put_str(&mut t, 8, "fd");
        let mut packed = Vec::new();
        packed.extend_from_slice(&1.5f32.to_le_bytes());
        packed.extend_from_slice(&(-3.0f32).to_le_bytes());
        put_len(&mut t, 4, &packed);
        let tt = read_tensor(&t).unwrap();
        assert_eq!(tt.f32_data().unwrap(), vec![1.5, -3.0]);

        // int64_data unpacked entries
        let mut t = Vec::new();
        let mut dims = Vec::new();
        vint(&mut dims, 3);
        put_len(&mut t, 1, &dims);
        put_varint(&mut t, 2, DT_INT64 as u64);
        put_str(&mut t, 8, "id");
        put_varint(&mut t, 7, 1);
        put_varint(&mut t, 7, u64::MAX); // -1 as two's complement varint
        put_varint(&mut t, 7, 8400);
        let tt = read_tensor(&t).unwrap();
        assert_eq!(tt.i64_data().unwrap(), vec![1, -1, 8400]);
    }

    #[test]
    fn f16_raw_converts() {
        let mut t = Vec::new();
        let mut dims = Vec::new();
        vint(&mut dims, 4);
        put_len(&mut t, 1, &dims);
        put_varint(&mut t, 2, DT_FLOAT16 as u64);
        put_str(&mut t, 8, "h");
        // 1.0, -2.0, 0.0, 0.5 in IEEE f16
        let raw: Vec<u8> = [0x3c00u16, 0xc000, 0x0000, 0x3800].iter().flat_map(|v| v.to_le_bytes()).collect();
        put_len(&mut t, 9, &raw);
        let tt = read_tensor(&t).unwrap();
        assert_eq!(tt.f32_data().unwrap(), vec![1.0, -2.0, 0.0, 0.5]);
    }

    #[test]
    fn refuses_external_data() {
        let mut t = Vec::new();
        put_varint(&mut t, 2, DT_FLOAT as u64);
        put_str(&mut t, 8, "big");
        put_varint(&mut t, 14, 1); // data_location = EXTERNAL
        let err = read_tensor(&t).unwrap_err().to_string();
        assert!(err.contains("external data"), "{err}");
    }

    #[test]
    fn refuses_subgraph_attribute() {
        let mut attr = Vec::new();
        put_str(&mut attr, 1, "body");
        put_len(&mut attr, 6, &[]); // g = empty GraphProto
        let mut node = Vec::new();
        put_str(&mut node, 3, "loop0");
        put_str(&mut node, 4, "Loop");
        put_len(&mut node, 5, &attr);
        let err = read_node(&node).unwrap_err().to_string();
        assert!(err.contains("control flow"), "{err}");
    }

    #[test]
    fn refuses_foreign_node_domain() {
        let mut node = Vec::new();
        put_str(&mut node, 4, "DeformConv");
        put_str(&mut node, 7, "com.microsoft");
        let err = read_node(&node).unwrap_err().to_string();
        assert!(err.contains("com.microsoft"), "{err}");
    }

    #[test]
    fn refuses_foreign_opset_import() {
        let mut opset = Vec::new();
        put_str(&mut opset, 1, "com.microsoft");
        put_varint(&mut opset, 2, 1);
        let mut graph = Vec::new();
        put_str(&mut graph, 2, "g");
        let mut model = Vec::new();
        put_varint(&mut model, 1, 8);
        put_len(&mut model, 7, &graph);
        put_len(&mut model, 8, &opset);
        let err = parse(&model).unwrap_err().to_string();
        assert!(err.contains("com.microsoft"), "{err}");
    }

    #[test]
    fn skips_unknown_fields() {
        let mut model = conv_model();
        // append an unknown length-delimited field (e.g. functions=25)
        put_len(&mut model, 25, b"future stuff");
        let m = parse(&model).unwrap();
        assert_eq!(m.graph.nodes.len(), 1);
    }

    #[test]
    fn truncated_input_errors_cleanly() {
        let model = conv_model();
        for cut in [1, model.len() / 2, model.len() - 1] {
            assert!(parse(&model[..cut]).is_err(), "cut at {cut} should fail");
        }
    }
}

/// Serialise a model to ONNX protobuf bytes — the subset [`parse`] reads
/// (graph, nodes, attributes, f32/i64 initializers, typed inputs/outputs).
/// For synthetic models in tests and fixtures, so no Python or protoc is
/// needed to make one.
pub fn encode(m: &OnnxModel) -> Vec<u8> {
    fn varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    fn key(out: &mut Vec<u8>, field: u64, wire: u64) {
        varint(out, (field << 3) | wire);
    }
    fn bytes(out: &mut Vec<u8>, field: u64, b: &[u8]) {
        key(out, field, 2);
        varint(out, b.len() as u64);
        out.extend_from_slice(b);
    }
    fn int(out: &mut Vec<u8>, field: u64, v: i64) {
        key(out, field, 0);
        varint(out, v as u64);
    }
    fn tensor(t: &OnnxTensor) -> Vec<u8> {
        let mut o = vec![];
        for &d in &t.dims {
            int(&mut o, 1, d);
        }
        int(&mut o, 2, t.dtype as i64);
        match &t.data {
            TensorData::Raw(r) => bytes(&mut o, 9, r),
            TensorData::Floats(f) => bytes(&mut o, 4, &f.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()),
            TensorData::Ints64(v) => {
                let mut p = vec![];
                v.iter().for_each(|&x| varint(&mut p, x as u64));
                bytes(&mut o, 7, &p);
            }
            TensorData::Ints32(v) => {
                let mut p = vec![];
                v.iter().for_each(|&x| varint(&mut p, x as i64 as u64));
                bytes(&mut o, 5, &p);
            }
            _ => {}
        }
        bytes(&mut o, 8, t.name.as_bytes());
        o
    }
    fn attr(a: &OnnxAttr) -> Vec<u8> {
        let mut o = vec![];
        bytes(&mut o, 1, a.name.as_bytes());
        let ty = match &a.value {
            AttrValue::F(f) => {
                key(&mut o, 2, 5);
                o.extend_from_slice(&f.to_bits().to_le_bytes());
                1
            }
            AttrValue::I(i) => {
                int(&mut o, 3, *i);
                2
            }
            AttrValue::S(s) => {
                bytes(&mut o, 4, s.as_bytes());
                3
            }
            AttrValue::T(t) => {
                bytes(&mut o, 5, &tensor(t));
                4
            }
            AttrValue::Floats(v) => {
                bytes(&mut o, 7, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
                6
            }
            AttrValue::Ints(v) => {
                let mut p = vec![];
                v.iter().for_each(|&x| varint(&mut p, x as u64));
                bytes(&mut o, 8, &p);
                7
            }
            AttrValue::Strings(v) => {
                v.iter().for_each(|s| bytes(&mut o, 9, s.as_bytes()));
                8
            }
        };
        int(&mut o, 20, ty);
        o
    }
    fn value_info(v: &OnnxValueInfo) -> Vec<u8> {
        let mut shape = vec![];
        for d in &v.dims {
            let mut dim = vec![];
            match d {
                OnnxDim::Value(x) => int(&mut dim, 1, *x),
                OnnxDim::Param(p) => bytes(&mut dim, 2, p.as_bytes()),
                OnnxDim::Unknown => {}
            }
            bytes(&mut shape, 1, &dim);
        }
        let mut tt = vec![];
        int(&mut tt, 1, v.elem_type as i64);
        bytes(&mut tt, 2, &shape);
        let mut ty = vec![];
        bytes(&mut ty, 1, &tt);
        let mut o = vec![];
        bytes(&mut o, 1, v.name.as_bytes());
        bytes(&mut o, 2, &ty);
        o
    }
    let g = &m.graph;
    let mut graph = vec![];
    for n in &g.nodes {
        let mut o = vec![];
        n.inputs.iter().for_each(|s| bytes(&mut o, 1, s.as_bytes()));
        n.outputs.iter().for_each(|s| bytes(&mut o, 2, s.as_bytes()));
        bytes(&mut o, 3, n.name.as_bytes());
        bytes(&mut o, 4, n.op_type.as_bytes());
        n.attrs.iter().for_each(|a| bytes(&mut o, 5, &attr(a)));
        if !n.domain.is_empty() {
            bytes(&mut o, 7, n.domain.as_bytes());
        }
        bytes(&mut graph, 1, &o);
    }
    bytes(&mut graph, 2, g.name.as_bytes());
    g.initializers.iter().for_each(|t| bytes(&mut graph, 5, &tensor(t)));
    g.inputs.iter().for_each(|v| bytes(&mut graph, 11, &value_info(v)));
    g.outputs.iter().for_each(|v| bytes(&mut graph, 12, &value_info(v)));
    let mut out = vec![];
    int(&mut out, 1, m.ir_version);
    bytes(&mut out, 2, m.producer_name.as_bytes());
    bytes(&mut out, 7, &graph);
    for (domain, version) in &m.opsets {
        let mut o = vec![];
        bytes(&mut o, 1, domain.as_bytes());
        int(&mut o, 2, *version);
        bytes(&mut out, 8, &o);
    }
    out
}

#[cfg(test)]
mod encode_tests {
    use super::*;

    #[test]
    fn encode_round_trips_through_parse() {
        let m = OnnxModel {
            ir_version: 8,
            producer_name: "t".into(),
            opsets: vec![(String::new(), 17)],
            graph: OnnxGraph {
                name: "g".into(),
                nodes: vec![OnnxNode {
                    name: "n".into(),
                    op_type: "Gemm".into(),
                    domain: String::new(),
                    inputs: vec!["x".into(), "w".into()],
                    outputs: vec!["y".into()],
                    attrs: vec![
                        OnnxAttr { name: "transB".into(), value: AttrValue::I(1) },
                        OnnxAttr { name: "alpha".into(), value: AttrValue::F(0.5) },
                        OnnxAttr { name: "pads".into(), value: AttrValue::Ints(vec![]) },
                    ],
                }],
                initializers: vec![OnnxTensor::from_f32("w", &[2, 3], vec![1.0, -2.0, 3.5, 0.0, 1e-3, -7.25])],
                inputs: vec![OnnxValueInfo { name: "x".into(), elem_type: 1, dims: vec![OnnxDim::Param("N".into()), OnnxDim::Value(3)] }],
                outputs: vec![OnnxValueInfo { name: "y".into(), elem_type: 1, dims: vec![OnnxDim::Param("N".into()), OnnxDim::Value(2)] }],
                value_info: vec![],
            },
        };
        let back = parse(&encode(&m)).unwrap();
        assert_eq!(back.opsets, m.opsets);
        assert_eq!(back.graph.nodes[0].op_type, "Gemm");
        assert_eq!(back.graph.nodes[0].inputs, ["x", "w"]);
        assert!(matches!(back.graph.nodes[0].attrs[0].value, AttrValue::I(1)));
        assert!(matches!(back.graph.nodes[0].attrs[1].value, AttrValue::F(v) if v == 0.5));
        assert!(matches!(&back.graph.nodes[0].attrs[2].value, AttrValue::Ints(v) if v.is_empty()));
        assert_eq!(back.graph.initializers[0].dims, [2, 3]);
        assert_eq!(back.graph.initializers[0].f32_data().unwrap(), [1.0, -2.0, 3.5, 0.0, 1e-3, -7.25]);
        assert_eq!(back.graph.inputs[0].dims, [OnnxDim::Param("N".into()), OnnxDim::Value(3)]);
    }
}
