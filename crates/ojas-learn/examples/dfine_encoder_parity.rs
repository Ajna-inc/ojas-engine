//! Forward parity of the Rust D-FINE hybrid encoder against PyTorch (DEIM `hybrid_encoder.py`,
//! version 'dfine'), eval mode, with the production config. Reference written by
//! `training/rust_parity/enc_ref.py`. `dfine_encoder_parity <ref.safetensors>`
use ojas_learn::cpu::Cpu;
use ojas_learn::models::dfine::{Dfine, DfineConfig};
use ojas_learn::models::rtdetr::Store;
use ojas_learn::Tape;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("dfine_encoder_parity <ref.safetensors>"))?;
    let tensors = ojas_formats::safetensors::read_all_f32(&path)?;
    let get = |name: &str| tensors.iter().find(|(n, _, _)| n == name).map(|(_, s, d)| (s.clone(), d.clone())).ok_or_else(|| anyhow::anyhow!("{name} missing"));
    let be = Cpu;
    let st = Store::from_safetensors(&be, &path, "")?;
    let model = Dfine { cfg: DfineConfig::ojas_n32(), st: &st, train: false };
    let mut t = Tape::new(&be);
    let mut feats = vec![];
    for i in 0..2 {
        let (s, d) = get(&format!("ref.in{i}"))?;
        feats.push(t.input(&d, &s));
    }
    let outs = model.encoder(&mut t, &feats)?;
    let mut worst = 0.0f32;
    for (i, o) in outs.iter().enumerate() {
        let (rs, rd) = get(&format!("ref.out{i}"))?;
        anyhow::ensure!(t.shape(*o) == rs.as_slice(), "out{i}: shape {:?} vs PyTorch {:?}", t.shape(*o), rs);
        let got = t.value(*o);
        let (mut max_abs, mut max_ref) = (0.0f32, 0.0f32);
        for (a, b) in got.iter().zip(&rd) {
            max_abs = max_abs.max((a - b).abs());
            max_ref = max_ref.max(b.abs());
        }
        let rel = max_abs / max_ref.max(1e-6);
        worst = worst.max(rel);
        println!("out{i} {:?}: max |Δ| {max_abs:.3e}, relative {rel:.3e}", rs);
    }
    anyhow::ensure!(worst < 1e-4, "encoder parity failed: worst relative error {worst:.3e}");
    println!("D-FINE encoder matches PyTorch (worst relative error {worst:.3e})");
    Ok(())
}
