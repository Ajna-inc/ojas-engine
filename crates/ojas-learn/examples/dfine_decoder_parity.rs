//! Forward parity of the Rust D-FINE decoder against PyTorch (DEIM `dfine_decoder.py`) with the
//! production config, training mode (every layer's logits, boxes, FDR corners and references, the
//! pre-refinement head and the encoder proposals). Reference written by
//! `training/rust_parity/dec_ref.py`. `dfine_decoder_parity <ref.safetensors>`
use ojas_learn::cpu::Cpu;
use ojas_learn::models::dfine::{Dfine, DfineConfig};
use ojas_learn::models::rtdetr::Store;
use ojas_learn::{Tape, Var};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("dfine_decoder_parity <ref.safetensors>"))?;
    let tensors = ojas_formats::safetensors::read_all_f32(&path)?;
    let get = |name: &str| tensors.iter().find(|(n, _, _)| n == name).map(|(_, s, d)| (s.clone(), d.clone())).ok_or_else(|| anyhow::anyhow!("{name} missing"));
    let be = Cpu;
    let st = Store::from_safetensors(&be, &path, "")?;
    let model = Dfine { cfg: DfineConfig::ojas_n32(), st: &st, train: true };
    let mut t = Tape::new(&be);
    let mut feats = vec![];
    for i in 0..2 {
        let (s, d) = get(&format!("ref.in{i}"))?;
        feats.push(t.input(&d, &s));
    }
    let out = model.decoder(&mut t, &feats, None)?;
    let mut checks: Vec<(String, Var)> = vec![];
    for i in 0..out.main.logits.len() {
        checks.push((format!("pred_logits.{i}"), out.main.logits[i]));
        checks.push((format!("pred_boxes.{i}"), out.main.boxes[i]));
        checks.push((format!("pred_corners.{i}"), out.main.corners[i]));
        checks.push((format!("ref_points.{i}"), out.main.refs[i]));
    }
    checks.push(("pre.pred_logits".into(), out.main.pre_logits));
    checks.push(("pre.pred_boxes".into(), out.main.pre_boxes));
    checks.push(("enc.pred_logits".into(), out.enc_logits.unwrap()));
    checks.push(("enc.pred_boxes".into(), out.enc_boxes.unwrap()));
    let mut worst = 0.0f32;
    for (name, v) in checks {
        let (rs, rd) = get(&format!("ref.{name}"))?;
        anyhow::ensure!(t.shape(v) == rs.as_slice(), "{name}: shape {:?} vs PyTorch {:?}", t.shape(v), rs);
        let got = t.value(v);
        let (mut max_abs, mut max_ref) = (0.0f32, 0.0f32);
        for (a, b) in got.iter().zip(&rd) {
            max_abs = max_abs.max((a - b).abs());
            max_ref = max_ref.max(b.abs());
        }
        let rel = max_abs / max_ref.max(1e-6);
        worst = worst.max(rel);
        println!("{name:<18} {:?}: max |Δ| {max_abs:.3e}, relative {rel:.3e}", rs);
    }
    anyhow::ensure!(worst < 1e-4, "decoder parity failed: worst relative error {worst:.3e}");
    println!("D-FINE decoder matches PyTorch (worst relative error {worst:.3e})");
    Ok(())
}
