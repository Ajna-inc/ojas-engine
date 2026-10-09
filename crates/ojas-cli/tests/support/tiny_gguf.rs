//! A tiny random-weight qwen3 GGUF, written at test time so no model binary lives in
//! the repository. Small enough to decode thousands of tokens a second on one core,
//! real enough for `CpuQwen::load` and `Bpe::from_gguf`: two layers, d = 64, GQA
//! with per-head QK-norm, and a byte-level vocabulary (one token per byte, no
//! merges) plus `<|im_end|>` as EOS.
//!
//! The EOS row of the output head is zero, so its logit is 0 while some byte's is
//! almost surely positive: greedy decoding never stops on its own, which is what a
//! test of `max_tokens` and `Cancel` needs.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

pub const D: usize = 64;
pub const LAYERS: usize = 2;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 16;
pub const FFN: usize = 128;
pub const VOCAB: usize = 257;
pub const EOS: u32 = 256;
const ALIGN: usize = 32;

/// Where each tensor's data starts in the file, for tests that edit one weight.
pub struct Written {
    pub offsets: HashMap<String, usize>,
}

enum V {
    U32(u32),
    F32(f32),
    Str(String),
    StrArr(Vec<String>),
    I32Arr(Vec<i32>),
}

/// The GPT-2 byte-to-character map the byte-level BPE tokenizers use.
fn byte_chars() -> Vec<char> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = vec!['\0'; 256];
    for (b, c) in bs.iter().zip(&cs) {
        out[*b as usize] = char::from_u32(*c).unwrap();
    }
    out
}

fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let mut h = ((exp as u32) << 10) | (man >> 13);
    if man & 0x1000 != 0 {
        h += 1;
    }
    sign | h as u16
}

struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Write the model to `path`. The same `seed` writes the same bytes.
pub fn write(path: &Path, seed: u64) -> Written {
    let arch = "qwen3";
    let chars = byte_chars();
    let mut tokens: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
    tokens.push("<|im_end|>".into());
    let mut ttype = vec![1i32; 256];
    ttype.push(3);
    let k = |s: &str| format!("{arch}.{s}");
    let meta: Vec<(String, V)> = vec![
        ("general.architecture".into(), V::Str(arch.into())),
        ("general.alignment".into(), V::U32(ALIGN as u32)),
        (k("block_count"), V::U32(LAYERS as u32)),
        (k("context_length"), V::U32(1 << 20)),
        (k("embedding_length"), V::U32(D as u32)),
        (k("feed_forward_length"), V::U32(FFN as u32)),
        (k("attention.head_count"), V::U32(HEADS as u32)),
        (k("attention.head_count_kv"), V::U32(KV_HEADS as u32)),
        (k("attention.key_length"), V::U32(HEAD_DIM as u32)),
        (k("attention.value_length"), V::U32(HEAD_DIM as u32)),
        (k("rope.freq_base"), V::F32(10000.0)),
        (k("attention.layer_norm_rms_epsilon"), V::F32(1e-6)),
        ("tokenizer.ggml.model".into(), V::Str("gpt2".into())),
        ("tokenizer.ggml.pre".into(), V::Str("qwen2".into())),
        ("tokenizer.ggml.tokens".into(), V::StrArr(tokens)),
        ("tokenizer.ggml.token_type".into(), V::I32Arr(ttype)),
        ("tokenizer.ggml.merges".into(), V::StrArr(vec![])),
        ("tokenizer.ggml.eos_token_id".into(), V::U32(EOS)),
    ];

    // (name, dims [in, out], f16?) — matmul weights f16, norms f32, as CpuQwen expects.
    let mut specs: Vec<(String, Vec<usize>, bool)> = vec![
        ("token_embd.weight".into(), vec![D, VOCAB], true),
        ("output_norm.weight".into(), vec![D], false),
        ("output.weight".into(), vec![D, VOCAB], true),
    ];
    let qd = HEADS * HEAD_DIM;
    let kvd = KV_HEADS * HEAD_DIM;
    for i in 0..LAYERS {
        let b = |s: &str| format!("blk.{i}.{s}");
        specs.extend([
            (b("attn_norm.weight"), vec![D], false),
            (b("attn_q.weight"), vec![D, qd], true),
            (b("attn_k.weight"), vec![D, kvd], true),
            (b("attn_v.weight"), vec![D, kvd], true),
            (b("attn_output.weight"), vec![qd, D], true),
            (b("attn_q_norm.weight"), vec![HEAD_DIM], false),
            (b("attn_k_norm.weight"), vec![HEAD_DIM], false),
            (b("ffn_norm.weight"), vec![D], false),
            (b("ffn_gate.weight"), vec![D, FFN], true),
            (b("ffn_up.weight"), vec![D, FFN], true),
            (b("ffn_down.weight"), vec![FFN, D], true),
        ]);
    }

    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    for (name, dims, half) in &specs {
        let n: usize = dims.iter().product();
        let fan_in = dims[0] as f32;
        let vals: Vec<f32> = (0..n)
            .map(|j| {
                if !half {
                    // Norm gains near 1.
                    1.0 + 0.1 * rng.uniform()
                } else if name == "output.weight" && j / D == EOS as usize {
                    0.0
                } else if name == "token_embd.weight" {
                    rng.uniform()
                } else {
                    rng.uniform() * 1.7 / fan_in.sqrt()
                }
            })
            .collect();
        let mut b = Vec::with_capacity(n * 4);
        for v in vals {
            if *half {
                b.extend(f16_bits(v).to_le_bytes());
            } else {
                b.extend(v.to_le_bytes());
            }
        }
        blobs.push(b);
    }

    let s = |out: &mut Vec<u8>, x: &str| {
        out.extend((x.len() as u64).to_le_bytes());
        out.extend(x.as_bytes());
    };
    let mut h = b"GGUF".to_vec();
    h.extend(3u32.to_le_bytes());
    h.extend((specs.len() as u64).to_le_bytes());
    h.extend((meta.len() as u64).to_le_bytes());
    for (key, v) in &meta {
        s(&mut h, key);
        match v {
            V::U32(x) => {
                h.extend(4u32.to_le_bytes());
                h.extend(x.to_le_bytes());
            }
            V::F32(x) => {
                h.extend(6u32.to_le_bytes());
                h.extend(x.to_le_bytes());
            }
            V::Str(x) => {
                h.extend(8u32.to_le_bytes());
                s(&mut h, x);
            }
            V::StrArr(xs) => {
                h.extend(9u32.to_le_bytes());
                h.extend(8u32.to_le_bytes());
                h.extend((xs.len() as u64).to_le_bytes());
                xs.iter().for_each(|x| s(&mut h, x));
            }
            V::I32Arr(xs) => {
                h.extend(9u32.to_le_bytes());
                h.extend(5u32.to_le_bytes());
                h.extend((xs.len() as u64).to_le_bytes());
                xs.iter().for_each(|x| h.extend(x.to_le_bytes()));
            }
        }
    }
    let mut rel = Vec::with_capacity(specs.len());
    let mut off = 0usize;
    for ((name, dims, half), blob) in specs.iter().zip(&blobs) {
        s(&mut h, name);
        h.extend((dims.len() as u32).to_le_bytes());
        dims.iter().for_each(|d| h.extend((*d as u64).to_le_bytes()));
        h.extend((if *half { 1u32 } else { 0 }).to_le_bytes());
        h.extend((off as u64).to_le_bytes());
        rel.push(off);
        off = (off + blob.len()).div_ceil(ALIGN) * ALIGN;
    }
    h.resize(h.len().div_ceil(ALIGN) * ALIGN, 0);
    let base = h.len();
    let mut offsets = HashMap::new();
    for (((name, _, _), blob), r) in specs.iter().zip(&blobs).zip(&rel) {
        h.resize(base + r, 0);
        h.extend(blob);
        offsets.insert(name.clone(), base + r);
    }
    std::fs::write(path, h).unwrap();
    Written { offsets }
}
