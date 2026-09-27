//! Overfit gate: 40 training steps on a small dataset; loss must collapse.
//! usage: overfit_gate <model_dir> <data.bin>
use ojas_metal::MetalGpu;
use ojas_train::Trainer;

fn main() -> anyhow::Result<()> {
    let model = std::env::args().nth(1).expect("model_dir");
    let data = std::env::args().nth(2).expect("data.bin");
    let bytes = std::fs::read(&data)?;
    let rd = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let (t, n) = (rd(4) as usize, rd(8) as usize);
    let seq = |i: usize| -> (Vec<u32>, Vec<u32>) {
        let base = 12 + i * t * 8;
        ((0..t).map(|j| rd(base + j * 4)).collect(),
         (0..t).map(|j| rd(base + t * 4 + j * 4)).collect())
    };
    let gpu = MetalGpu::new()?;
    let mut tr = Trainer::new(&gpu, &model, t, 1e-4)?;
    let (mut first, mut last) = (0f32, 0f32);
    for step in 0..40 {
        let (toks, tgts) = seq(step % n);
        let (loss, _) = tr.train_step(&toks, &tgts)?;
        if step == 0 { first = loss; }
        last = loss;
        if step % 10 == 0 { println!("step {step} loss {loss:.4}"); }
    }
    println!("GATE: first {first:.4} last {last:.4} -> {}",
             if first > 2.0 && last < 0.4 { "PASS" } else { "FAIL" });
    Ok(())
}
