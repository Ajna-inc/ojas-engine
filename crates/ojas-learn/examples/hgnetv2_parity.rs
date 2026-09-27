//! Forward parity of the Rust HGNetv2-B0 against PyTorch (D-FINE `hgnetv2.py`), eval mode.
//!
//! The reference file holds the D-FINE-N COCO backbone weights (`backbone.*`), a fixed input
//! (`ref.input`) and PyTorch's two returned feature maps (`ref.out0`, `ref.out1`), written by
//! `training/rust_parity/hg_ref.py`. `hgnetv2_parity <ref.safetensors>`
use ojas_learn::cpu::Cpu;
use ojas_learn::models::hgnetv2::{HgConfig, HgNetV2};
use ojas_learn::models::rtdetr::Store;
use ojas_learn::Tape;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("hgnetv2_parity <ref.safetensors>"))?;
    let tensors = ojas_formats::safetensors::read_all_f32(&path)?;
    let get = |name: &str| tensors.iter().find(|(n, _, _)| n == name).map(|(_, s, d)| (s.clone(), d.clone())).ok_or_else(|| anyhow::anyhow!("{name} missing"));
    let be = Cpu;
    let st = Store::from_safetensors(&be, &path, "")?;
    let model = HgNetV2 { cfg: HgConfig::b0(&[2, 3]), st: &st, prefix: "backbone.".into(), train: false };
    let (xs, xd) = get("ref.input")?;
    let mut t = Tape::new(&be);
    let x = t.input(&xd, &xs);
    let outs = model.forward(&mut t, x)?;
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
        println!("out{i} {:?}: max |Δ| {max_abs:.3e}, relative to max |ref| {rel:.3e}", rs);
    }
    anyhow::ensure!(worst < 1e-4, "HGNetv2 parity failed: worst relative error {worst:.3e}");
    println!("HGNetv2-B0 matches PyTorch (worst relative error {worst:.3e})");
    Ok(())
}
