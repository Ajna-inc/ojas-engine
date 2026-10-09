//! Byte-level token files for a TinyGpt run: `train.ojtk` and `heldout.ojtk` from text.
//!
//! ```text
//! cargo run --release -p ojas-swarm --example make_tokens -- <text files...> --out <dir> [--heldout 0.05]
//! ```

use anyhow::{bail, Context, Result};
use ojas_swarm::train::data;

fn main() -> Result<()> {
    let (mut inputs, mut out, mut frac) = (Vec::new(), None, 0.05f64);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = it.next(),
            "--heldout" => frac = it.next().context("--heldout needs a fraction")?.parse()?,
            _ => inputs.push(a),
        }
    }
    let Some(out) = out else { bail!("usage: make_tokens <text files...> --out <dir> [--heldout 0.05]") };
    if inputs.is_empty() {
        bail!("no input files");
    }
    let mut text = Vec::new();
    for p in &inputs {
        text.extend(std::fs::read(p).with_context(|| format!("reading {p}"))?);
        text.push(b'\n');
    }
    let (train, heldout) = data::split_heldout(&data::from_bytes(&text), frac)?;
    std::fs::create_dir_all(&out)?;
    let dir = std::path::Path::new(&out);
    std::fs::write(dir.join("train.ojtk"), &train)?;
    std::fs::write(dir.join("heldout.ojtk"), &heldout)?;
    println!("{} bytes of text -> {}/train.ojtk ({} B), heldout.ojtk ({} B)", text.len(), out, train.len(), heldout.len());
    Ok(())
}
