//! One RT-DETRv2 training step (train-mode BN, no denoising) vs PyTorch: every
//! loss term and every parameter's gradient norm (printed when off by > 2%, or
//! all with ALL=1). `rtdetr_train_parity model.pth prefix input.f32 ref.json`
//! The reference must come from PyTorch on CUDA: PyTorch 2.8's CPU BatchNorm
//! backward returns wrong bias / weight gradients when the incoming gradient is
//! non-contiguous (here: after the decoder's flatten + permute).
use ojas_learn::backend::Backend;
use ojas_learn::cuda::{Cuda, Prec};
use ojas_learn::models::detr_loss::{Criterion, Target};
use ojas_learn::models::rtdetr::{Config, RtDetr, Store};
use ojas_learn::Tape;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let tensors = ojas_formats::pth::load(&std::fs::read(&a[1])?)?;
    let x: Vec<f32> = std::fs::read(&a[3])?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let rf: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[4])?)?;
    let targets = vec![Target { labels: vec![1, 7, 3], boxes: vec![[0.52, 0.61, 0.20, 0.25], [0.30, 0.72, 0.08, 0.15], [0.75, 0.45, 0.30, 0.22]] }];
    for prec in [Prec::F32, Prec::Bf16] {
        let mut be = Cuda::new(0)?;
        be.prec = prec;
        let st = Store::from_tensors(&be, &tensors, &a[2]);
        let m = RtDetr { cfg: Config::r18vd(15), st: &st, train: true, var: Default::default() };
        let crit = Criterion::rtdetrv2(15);
        let t0 = std::time::Instant::now();
        let mut t = Tape::new(&be);
        let xv = t.input(&x, &[1, 3, 640, 640]);
        let out = m.forward(&mut t, xv, None)?;
        let (total, terms) = crit.forward(&mut t, &out, &targets, None)?;
        let tf = t0.elapsed();
        for (_, v) in &terms {
            t.keep(*v);
        }
        t.backward(total)?;
        let tv = t.value(total)[0];
        println!("{prec:?}: forward+loss {:.0} ms, backward {:.0} ms", tf.as_secs_f64() * 1e3, (t0.elapsed() - tf).as_secs_f64() * 1e3);
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
        let mut worst = 0.0f64;
        for (name, v) in &terms {
            let ours = t.value(*v)[0] as f64;
            let want = rf["losses"][name].as_f64().unwrap_or(f64::NAN);
            worst = worst.max(rel(ours, want));
            println!("  {name:18} ours {ours:.6}  torch {want:.6}");
        }
        println!("  total {tv:.6} torch {:.6}   worst loss-term relative error {worst:.2e}", rf["total"].as_f64().unwrap());
        let mut all = 0.0f64;
        for (name, p) in &st.params {
            if let Some(g) = t.param_var(p).and_then(|v| t.grad_vec(v)) {
                let n: f64 = g.iter().map(|v| (*v as f64).powi(2)).sum();
                all += n;
                if let Some(w) = rf["grad_norm"][name].as_f64() {
                    let e = rel(n.sqrt(), w);
                    if e > 2e-2 || std::env::var("ALL").is_ok() {
                        println!("  |grad| {name:62} ours {:.5}  torch {w:.5}  ({:.1e})", n.sqrt(), e);
                    }
                }
            }
        }
        println!("  |grad| all ours {:.4} torch {:.4}", all.sqrt(), rf["grad_norm_all"].as_f64().unwrap());
        let _ = be.len(t.buf(xv));
    }
    Ok(())
}
