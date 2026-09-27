//! RT-DETRv2 on the tape vs the official PyTorch forward (eval mode):
//! `rtdetr_parity model.pth <prefix> input.f32 <ref dir>/<name>`.
//! Compares the backbone and encoder feature maps and the final logits / boxes
//! against `<ref>.{backbone,encoder}{0,1,2}.f32`, `<ref>.{logits,boxes}.f32`.
use ojas_learn::backend::Backend;
use ojas_learn::cuda::{Cuda, Prec};
use ojas_learn::models::rtdetr::{Config, RtDetr, Store};
use ojas_learn::Tape;

fn read(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}")).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn cmp(name: &str, ours: &[f32], want: &[f32]) {
    assert_eq!(ours.len(), want.len(), "{name}: length");
    let max_abs = ours.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let num: f64 = ours.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    println!("  {name:12} max |Δ| {max_abs:.2e}   relative (norm) {:.2e}", num / den);
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let tensors = ojas_formats::pth::load(&std::fs::read(&a[1])?)?;
    let x = read(&a[3]);
    for prec in [Prec::F32, Prec::Bf16] {
        let mut be = Cuda::new(0)?;
        be.prec = prec;
        let st = Store::from_tensors(&be, &tensors, &a[2]);
        let m = RtDetr { cfg: Config::r18vd(15), st: &st, train: false, var: Default::default() };
        let t0 = std::time::Instant::now();
        let mut t = Tape::new(&be);
        let xv = t.input(&x, &[1, 3, 640, 640]);
        let f = m.backbone(&mut t, xv)?;
        let e = m.encoder(&mut t, &f)?;
        let o = m.decoder(&mut t, &e, None)?;
        let logits = t.value(o.logits[0]);
        println!("{prec:?}: forward {:.0} ms ({} params)", t0.elapsed().as_secs_f64() * 1e3, st.params.len());
        for (i, &v) in f.iter().enumerate() {
            cmp(&format!("backbone{i}"), &t.value(v), &read(&format!("{}.backbone{i}.f32", a[4])));
        }
        for (i, &v) in e.iter().enumerate() {
            cmp(&format!("encoder{i}"), &t.value(v), &read(&format!("{}.encoder{i}.f32", a[4])));
        }
        cmp("logits", &logits, &read(&format!("{}.logits.f32", a[4])));
        cmp("boxes", &t.value(o.boxes[0]), &read(&format!("{}.boxes.f32", a[4])));
        let dets = logits.chunks(15).filter(|r| r.iter().any(|&v| 1.0 / (1.0 + (-v).exp()) > 0.5)).count();
        let want = read(&format!("{}.logits.f32", a[4])).chunks(15).filter(|r| r.iter().any(|&v| 1.0 / (1.0 + (-v).exp()) > 0.5)).count();
        println!("  detections > 0.5: ours {dets}, PyTorch {want}");
        let _ = be.len(t.buf(xv));
    }
    Ok(())
}
