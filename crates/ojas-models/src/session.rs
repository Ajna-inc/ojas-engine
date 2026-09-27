//! Prompt/state session cache — the "blink prefill" path.
//!
//! After prefill, the model's entire sequence state (attention KV rows + SSM
//! recurrent/conv states) is written to one file. On the next run, if the saved
//! token sequence is a prefix of the new prompt, the state is memcpy'd back and
//! only the new suffix is prefilled. Hybrid qwen35 models make this cheap: at 8k
//! the state is a few hundred MB (restore ~0.2s) vs minutes of recompute on the
//! big MoEs. Same idea as the reference --prompt-cache / server cache_prompt.
//!
//! Enabled by OJAS_SESSION: "1" → <cache_dir>/sessions, else treated as a
//! directory path. One file per model (last session wins) — covers prompt
//! reruns and growing-prefix chats.
//!
//! Format (little-endian): magic u64, model_key u64, n_tokens u32, n_parts u32,
//! tokens u32×n, then per part: len u64 + raw bytes. Part order is defined by
//! the caller and must be identical between save and restore; any length
//! mismatch aborts the restore (stale file from a different model config).

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

const MAGIC: u64 = 0x4f4a_4153_5345_5331; // "OJASSES1"

/// Session directory from OJAS_SESSION (None = caching disabled).
pub fn dir() -> Option<PathBuf> {
    let v = ojas_core::config::var("OJAS_SESSION").ok()?;
    let d = if v == "1" {
        ojas_core::config::cache_dir()?.join("sessions")
    } else {
        PathBuf::from(v)
    };
    fs::create_dir_all(&d).ok()?;
    Some(d)
}

/// Stable identity for (model, geometry) — mismatch means the file is ignored.
pub fn model_key(name: &str, n_layers: usize, d: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes().chain(n_layers.to_le_bytes()).chain(d.to_le_bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn save(path: &PathBuf, key: u64, tokens: &[u32], parts: &[&[u8]]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(&MAGIC.to_le_bytes())?;
        w.write_all(&key.to_le_bytes())?;
        w.write_all(&(tokens.len() as u32).to_le_bytes())?;
        w.write_all(&(parts.len() as u32).to_le_bytes())?;
        for t in tokens {
            w.write_all(&t.to_le_bytes())?;
        }
        for p in parts {
            w.write_all(&(p.len() as u64).to_le_bytes())?;
            w.write_all(p)?;
        }
        w.flush()?;
    }
    fs::rename(&tmp, path) // atomic: never a half-written session
}

/// An opened session whose header matched; stream parts out in save order.
pub struct Loaded {
    pub tokens: Vec<u32>,
    r: BufReader<File>,
}

pub fn open(path: &PathBuf, key: u64) -> Option<Loaded> {
    let mut r = BufReader::new(File::open(path).ok()?);
    let mut u64b = [0u8; 8];
    let mut u32b = [0u8; 4];
    r.read_exact(&mut u64b).ok()?;
    if u64::from_le_bytes(u64b) != MAGIC {
        return None;
    }
    r.read_exact(&mut u64b).ok()?;
    if u64::from_le_bytes(u64b) != key {
        return None;
    }
    r.read_exact(&mut u32b).ok()?;
    let n = u32::from_le_bytes(u32b) as usize;
    r.read_exact(&mut u32b).ok()?; // n_parts (implied by caller's layer walk)
    let mut tokens = vec![0u32; n];
    for t in tokens.iter_mut() {
        r.read_exact(&mut u32b).ok()?;
        *t = u32::from_le_bytes(u32b);
    }
    Some(Loaded { tokens, r })
}

impl Loaded {
    /// Read the next part into `dst`; fails (→ abort restore) on length mismatch.
    pub fn next_part(&mut self, dst: &mut [u8]) -> bool {
        let mut u64b = [0u8; 8];
        if self.r.read_exact(&mut u64b).is_err() {
            return false;
        }
        if u64::from_le_bytes(u64b) != dst.len() as u64 {
            return false;
        }
        self.r.read_exact(dst).is_ok()
    }
}
