//! One-step parity of the Rust DEIM criterion against PyTorch: the production D-FINE decoder with
//! contrastive denoising, every weighted loss term and every decoder parameter's gradient.
//! Reference written by `training/rust_parity/loss_ref.py` (which also records the exact
//! denoising group PyTorch drew). `deim_loss_parity <ref.safetensors>`
use ojas_learn::cpu::Cpu;
use ojas_learn::models::deim_loss::DeimCriterion;
use ojas_learn::models::detr_loss::{DnGroup, Target};
use ojas_learn::models::dfine::{Dfine, DfineConfig};
use ojas_learn::models::rtdetr::Store;
use ojas_learn::Tape;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("deim_loss_parity <ref.safetensors>"))?;
    let tensors = ojas_formats::safetensors::read_all_f32(&path)?;
    let get = |name: &str| tensors.iter().find(|(n, _, _)| n == name).map(|(_, s, d)| (s.clone(), d.clone())).ok_or_else(|| anyhow::anyhow!("{name} missing"));
    let be = Cpu;
    let st = Store::from_safetensors(&be, &path, "decoder.")?;
    // the store's names are relative to "decoder."; the model wants the full ones
    let st = Store { params: st.params.into_iter().map(|(k, v)| (format!("decoder.{k}"), v)).collect(), bns: st.bns, consts: st.consts };
    let cfg = DfineConfig::ojas_n32();
    let (hidden, nq, nc) = (cfg.hidden, cfg.num_queries, cfg.num_classes);
    let model = Dfine { cfg, st: &st, train: true };

    let mut targets = vec![];
    for b in 0..2 {
        let (_, labels) = get(&format!("ref.tgt{b}.labels"))?;
        let (_, boxes) = get(&format!("ref.tgt{b}.boxes"))?;
        targets.push(Target { labels: labels.iter().map(|&l| l as usize).collect(), boxes: boxes.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect() });
    }
    // the denoising group PyTorch drew
    let (cs, classes) = get("ref.dn.classes")?;
    let (bsz, d) = (cs[0], cs[1]);
    let classes: Vec<usize> = classes.iter().map(|&c| c as usize).collect();
    let pad: Vec<f32> = classes.iter().map(|&c| if c < nc { 1.0 } else { 0.0 }).collect();
    let (_, boxes_unact) = get("ref.dn.boxes_unact")?;
    let (_, mask) = get("ref.dn.mask")?;
    let mask: Vec<f32> = mask.iter().map(|&m| if m > 0.5 { f32::NEG_INFINITY } else { 0.0 }).collect();
    let num_group = get("ref.dn.num_group")?.1[0] as usize;
    let mut matches = vec![];
    for (b, tg) in targets.iter().enumerate() {
        let (_, pos) = get(&format!("ref.dn.pos{b}"))?;
        let n = tg.labels.len();
        matches.push(pos.iter().enumerate().map(|(i, &p)| (p as usize, i % n)).collect());
    }
    let g = DnGroup { classes: classes.clone(), pad: pad.clone(), boxes_unact: boxes_unact.clone(), mask: mask.clone(), num_dn: d, num_group, matches };

    let mut t = Tape::new(&be);
    let mut feats = vec![];
    for i in 0..2 {
        let (s, v) = get(&format!("ref.in{i}"))?;
        feats.push(t.input(&v, &s));
    }
    let emb = t.param(&st.params["decoder.denoising_class_embed.weight"]);
    let rows = t.shape(emb)[0];
    let emb = t.reshape(emb, &[1, rows, hidden])?;
    let content = t.gather_rows(emb, &g.classes, bsz * d)?;
    let content = t.reshape(content, &[bsz, d, hidden])?;
    let padv = t.input(&g.pad, &[bsz, d, 1]);
    let content = t.mul(content, padv)?;
    let dn_boxes = t.input(&g.boxes_unact, &[bsz, d, 4]);
    let n = d + nq;
    let maskv = t.input(&g.mask, &[n, n]);
    let out = model.decoder(&mut t, &feats, Some((content, dn_boxes, maskv)))?;
    let crit = DeimCriterion::deim(nc);
    let (total, terms) = crit.forward(&mut t, &out, &targets, Some(&g))?;
    for (_, v) in &terms {
        t.keep(*v);
    }
    let total_v = t.value(total)[0];
    t.backward(total)?;

    // loss terms
    let mut worst_loss = 0.0f32;
    let mut seen = 0;
    for (name, v) in &terms {
        let got = t.value(*v)[0];
        let (_, r) = get(&format!("loss.{name}")).map_err(|_| anyhow::anyhow!("PyTorch has no term {name}"))?;
        let rel = (got - r[0]).abs() / r[0].abs().max(1e-3);
        worst_loss = worst_loss.max(rel);
        seen += 1;
        if rel > 1e-4 {
            println!("  term {name:<22} rust {got:.6} torch {:.6} rel {rel:.2e}", r[0]);
        }
    }
    let torch_terms = tensors.iter().filter(|(n, _, _)| n.starts_with("loss.") && n != "loss.total").count();
    let (_, rt) = get("loss.total")?;
    println!("{seen} loss terms (PyTorch {torch_terms}); total rust {total_v:.6} torch {:.6}; worst term relative error {worst_loss:.2e}", rt[0]);
    anyhow::ensure!(seen == torch_terms, "term count differs");

    // gradients
    let mut worst = (0.0f32, String::new());
    let mut count = 0;
    for (name, _, rd) in tensors.iter().filter(|(n, _, _)| n.starts_with("grad.")) {
        let pname = &name["grad.".len()..];
        let p = st.params.get(pname).ok_or_else(|| anyhow::anyhow!("{pname} not in the store"))?;
        let v = t.param_var(p).ok_or_else(|| anyhow::anyhow!("{pname} not on the tape"))?;
        let got = t.grad_vec(v).unwrap_or_else(|| vec![0.0; rd.len()]);
        let (mut max_abs, mut max_ref) = (0.0f32, 0.0f32);
        for (a, b) in got.iter().zip(rd) {
            max_abs = max_abs.max((a - b).abs());
            max_ref = max_ref.max(b.abs());
        }
        let rel = max_abs / max_ref.max(1e-8);
        if rel > worst.0 {
            worst = (rel, pname.to_string());
        }
        if rel > 1e-3 {
            println!("  grad {pname:<55} rel {rel:.2e} (max |ref| {max_ref:.2e})");
        }
        count += 1;
    }
    println!("{count} parameter gradients; worst relative error {:.2e} ({})", worst.0, worst.1);
    anyhow::ensure!(worst_loss < 1e-3 && worst.0 < 1e-2, "DEIM loss parity failed");
    println!("DEIM criterion matches PyTorch");
    Ok(())
}
