//! Train RT-DETRv2 on COCO-format data (the IISc UVH-26 recipe).
//! `rtdetr_train init.pth|init.safetensors <prefix> train.json image_root out_dir [batch] [epochs] [max_steps] [limit]`
//! Env: VAL=json:root:N (EMA eval every epoch on N images), LOG_EVERY (steps).
use std::sync::Arc;

use ojas_learn::backend::Backend;
use ojas_learn::cuda::Cuda;
use ojas_learn::data::{load_coco, loader, Sample};
use ojas_learn::eval::{coco_map, postprocess};
use ojas_learn::models::detr_loss::Rng;
use ojas_learn::models::rtdetr::{Config, RtDetr, Store};
use ojas_learn::train::{TrainCfg, Trainer};
use ojas_learn::Tape;

fn evaluate(be: &Cuda, st: &Store<Cuda>, cfg: &Config, samples: Arc<Vec<Sample>>) -> anyhow::Result<f32> {
    let m = RtDetr { cfg: cfg.clone(), st, train: false, var: Default::default() };
    let (mut gts, mut dets) = (vec![], vec![]);
    for b in loader(samples.clone(), (0..samples.len()).collect(), 8, 640, false, 12, 0) {
        let b = b?;
        let n = b.samples.len();
        let mut t = Tape::new(be);
        let x = t.input(&b.images, &[n, 3, 640, 640]);
        let o = m.forward(&mut t, x, None)?;
        let (lg, bx) = (t.value(o.logits[0]), t.value(o.boxes[0]));
        let (q, c) = (cfg.num_queries, cfg.num_classes);
        for (i, &si) in b.samples.iter().enumerate() {
            let s = &samples[si];
            dets.push(postprocess(&lg[i * q * c..(i + 1) * q * c], &bx[i * q * 4..(i + 1) * q * 4], c, 300, s.width as f32, s.height as f32));
            gts.push(s.boxes.iter().map(|&(c, b)| (c, [b[0], b[1], b[0] + b[2], b[1] + b[3]])).collect());
        }
    }
    Ok(coco_map(&gts, &dets).ap)
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: usize| a.get(i).map(|v| v.parse().unwrap()).unwrap_or(d);
    let (batch, epochs, max_steps, limit) = (arg(6, 8), arg(7, 1), arg(8, usize::MAX), arg(9, usize::MAX));
    let out = std::path::PathBuf::from(&a[5]);
    std::fs::create_dir_all(&out)?;
    let be = Cuda::new(0)?;
    let st = if a[1].ends_with(".safetensors") {
        Store::from_safetensors(&be, &a[1], &a[2])?
    } else {
        Store::from_tensors(&be, &ojas_formats::pth::load(&std::fs::read(&a[1])?)?, &a[2])
    };
    let mut samples = load_coco(a[3].as_ref(), a[4].as_ref())?;
    samples.truncate(limit);
    let samples = Arc::new(samples);
    let val = std::env::var("VAL").ok().map(|v| {
        let p: Vec<&str> = v.split(':').collect();
        let mut s = load_coco(p[0].as_ref(), p[1].as_ref()).expect("val set");
        s.truncate(p.get(2).and_then(|n| n.parse().ok()).unwrap_or(usize::MAX));
        Arc::new(s)
    });
    let log_every: u64 = std::env::var("LOG_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let cfg = Config::r18vd(15);
    println!("{} train images, batch {batch}, {epochs} epochs; {} parameters", samples.len(), st.params.values().map(|p| p.shape.iter().product::<usize>()).sum::<usize>());
    let mut tr = Trainer::new(&be, cfg.clone(), st, TrainCfg::rtdetrv2_uvh26(), 1);
    if let Some(v) = &val {
        println!("epoch 0: EMA mAP {:.4} on {} val images", evaluate(&be, &tr.ema.store, &cfg, v.clone())?, v.len());
    }
    let mut rng = Rng::new(7);
    let t0 = std::time::Instant::now();
    let mut seen = 0usize;
    'outer: for epoch in 1..=epochs {
        let mut order: Vec<usize> = (0..samples.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.next_u32() as usize % (i + 1));
        }
        for b in loader(samples.clone(), order, batch, 640, true, 16, epoch as u64) {
            let b = b?;
            let log = tr.step(&b.images, &b.targets, [640, 640])?;
            seen += b.targets.len();
            if tr.step % log_every == 0 || tr.step == 1 {
                let term = |n: &str| log.terms.iter().find(|t| t.0 == n).map(|t| t.1).unwrap_or(f32::NAN);
                println!(
                    "epoch {epoch} step {:6}: loss {:7.3} (vfl {:.3} bbox {:.3} giou {:.3}) |g| {:7.3} lr {:.2e}  {:.1} img/s",
                    tr.step, log.loss, term("loss_vfl"), term("loss_bbox"), term("loss_giou"), log.grad_norm, log.lr, seen as f64 / t0.elapsed().as_secs_f64()
                );
            }
            if tr.step == 3 {
                be.take_profile(); // drop warm-up steps from the profile
            }
            if tr.step as usize >= max_steps {
                let prof = be.take_profile();
                if !prof.is_empty() {
                    let steps = (tr.step - 3) as f64;
                    println!("per step (synced), over {steps} steps:");
                    for (k, ms, n) in prof.iter().take(18) {
                        println!("  {k:28} {:8.1} ms  x{}", ms / steps, *n as f64 / steps);
                    }
                }
                break 'outer;
            }
        }
        tr.save(&out.join(format!("epoch{epoch}.safetensors")))?;
        if let Some(v) = &val {
            println!("epoch {epoch}: EMA mAP {:.4} on {} val images", evaluate(&be, &tr.ema.store, &cfg, v.clone())?, v.len());
        }
    }
    tr.save(&out.join("last.safetensors"))?;
    let _ = be.len(&be.alloc(1));
    println!("done: {} steps, {:.1} img/s", tr.step, seen as f64 / t0.elapsed().as_secs_f64());
    Ok(())
}
