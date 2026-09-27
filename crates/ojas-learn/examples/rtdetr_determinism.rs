//! Batched RT-DETRv2 forward run twice on the same input, checking the logits are bit-identical.
//! `rtdetr_determinism model.pth prefix input.f32 batch`
use ojas_learn::cuda::{Cuda, Prec};
use ojas_learn::models::rtdetr::{Config, RtDetr, Store};
use ojas_learn::Tape;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let b: usize = a[4].parse()?;
    let tensors = ojas_formats::pth::load(&std::fs::read(&a[1])?)?;
    let one: Vec<f32> = std::fs::read(&a[3])?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    // a different image per slot: the same frame shifted
    let n = one.len();
    let x: Vec<f32> = (0..b * n).map(|k| { let (i, j) = (k / n, k % n); one[(j + i * 997) % n] * 0.5 + one[j] * 0.5 }).collect();
    for prec in [Prec::F32, Prec::Bf16] {
        let mut be = Cuda::new(0)?;
        be.prec = prec;
        let st = Store::from_tensors(&be, &tensors, &a[2]);
        let m = RtDetr { cfg: Config::r18vd(15), st: &st, train: false, var: Default::default() };
        let run = || -> anyhow::Result<(Vec<f32>, Vec<Vec<f32>>)> {
            let mut t = Tape::new(&be);
            let xv = t.input(&x, &[b, 3, 640, 640]);
            let f = m.backbone(&mut t, xv)?;
            let e = m.encoder(&mut t, &f)?;
            let o = m.decoder(&mut t, &e, None)?;
            Ok((t.value(o.logits[0]), f.iter().chain(e.iter()).map(|&v| t.value(v)).collect()))
        };
        let (l1, s1) = run()?;
        let (l2, s2) = run()?;
        let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let stages: Vec<usize> = s1.iter().zip(&s2).map(|(a, b)| diff(a, b)).collect();
        println!("{prec:?} batch {b}: logits differing {} / {}, per stage (backbone ×3, encoder ×3) {stages:?}", diff(&l1, &l2), l1.len());
    }
    Ok(())
}
