//! Train the production D-FINE (HGNetv2-B0 + 96-wide encoder / 4-layer decoder, 3.2 M) with the
//! DEIM recipe: MAL / L1 / GIoU / FGL / DDF loss, mosaic and mixup in epochs [4, 40), full
//! augmentation to 64, none after; flat-cosine lr; EMA restart from the best stage-1 checkpoint
//! at the stop epoch (`training/deim/step3_ojas_n32.yml`).
//!
//! `dfine_train init.pth|init.safetensors <prefix> train.json image_root out_dir [batch] [epochs] [max_steps] [limit]`
//!
//! Env: VAL=json:root:N (EMA mAP every epoch on N images), LOG_EVERY (steps), BACKEND=cpu|cuda
//! (default cuda), SIZE (input side, default 640), START_EPOCH (enter the schedule there),
//! EVAL_ONLY (mAP of the init weights on VAL, then exit). Resumes from `out_dir/last.safetensors`
//! when present (weights and EMA; the AdamW moments restart).
use std::sync::Arc;

use ojas_learn::backend::Backend;
use ojas_learn::data::{load_coco, loader, loader_epoch, AugPolicy, Sample};
use ojas_learn::eval::{coco_map, postprocess};
use ojas_learn::models::detr_loss::Rng;
use ojas_learn::models::dfine::{Dfine, DfineConfig};
use ojas_learn::models::rtdetr::Store;
use ojas_learn::train::{DfineTrainer, TrainCfg};
use ojas_learn::Tape;

fn evaluate<B: Backend>(be: &B, st: &Store<B>, cfg: &DfineConfig, samples: Arc<Vec<Sample>>, size: usize) -> anyhow::Result<f32> {
    let m = Dfine { cfg: cfg.clone(), st, train: false };
    let (mut gts, mut dets) = (vec![], vec![]);
    for b in loader(samples.clone(), (0..samples.len()).collect(), 8, size, false, 12, 0) {
        let b = b?;
        let n = b.samples.len();
        let mut t = Tape::new(be);
        let x = t.input(&b.images, &[n, 3, size, size]);
        let o = m.forward(&mut t, x, None)?;
        let (lg, bx) = (t.value(o.main.logits[0]), t.value(o.main.boxes[0]));
        let (q, c) = (cfg.num_queries, cfg.num_classes);
        for (i, &si) in b.samples.iter().enumerate() {
            let s = &samples[si];
            dets.push(postprocess(&lg[i * q * c..(i + 1) * q * c], &bx[i * q * 4..(i + 1) * q * 4], c, 300, s.width as f32, s.height as f32));
            gts.push(s.boxes.iter().map(|&(c, b)| (c, [b[0], b[1], b[0] + b[2], b[1] + b[3]])).collect());
        }
    }
    Ok(coco_map(&gts, &dets).ap)
}

/// Replace the trainer's weights and EMA with a checkpoint's ("model." / "ema.").
fn load_into<B: Backend>(be: &B, tr: &mut DfineTrainer<B>, path: &std::path::Path) -> anyhow::Result<()> {
    let p = path.to_str().unwrap();
    tr.st = Store::from_safetensors(be, p, "model.")?;
    tr.ema.store = Store::from_safetensors(be, p, "ema.")?;
    Ok(())
}

fn run<B: Backend>(be: &B, a: &[String]) -> anyhow::Result<()> {
    let arg = |i: usize, d: usize| a.get(i).map(|v| v.parse().unwrap()).unwrap_or(d);
    let (batch, epochs, max_steps, limit) = (arg(6, 48), arg(7, 72), arg(8, usize::MAX), arg(9, usize::MAX));
    let size: usize = std::env::var("SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(640);
    let out = std::path::PathBuf::from(&a[5]);
    std::fs::create_dir_all(&out)?;
    let st = if a[1].ends_with(".safetensors") {
        Store::from_safetensors(be, &a[1], &a[2])?
    } else {
        Store::from_tensors(be, &ojas_formats::pth::load(&std::fs::read(&a[1])?)?, &a[2])
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
    let cfg = DfineConfig::ojas_n32();
    let iters = (samples.len() / batch).max(1) as u64;
    let mut tcfg = TrainCfg::deim_ojas_n32(iters);
    if let Some(s) = tcfg.schedule.as_mut() {
        *s = ojas_learn::train::FlatCosine::from_epochs(iters, epochs as u64, s.warmup, (epochs as u64 * 40) / 72, (epochs as u64 * 8) / 72, s.gamma);
    }
    // augmentation stages scale with the schedule: [4, 40, 64] of 72
    let (flat, stop) = ((epochs * 40) / 72, epochs - (epochs * 8) / 72);
    let policy = AugPolicy::deim(flat, stop);
    println!(
        "{} train images, batch {batch}, {epochs} epochs ({iters} steps each; mosaic/mixup to {flat}, augmentation to {stop}); {} parameters",
        samples.len(),
        st.params.values().map(|p| p.shape.iter().product::<usize>()).sum::<usize>()
    );
    let mut tr = DfineTrainer::new(be, cfg.clone(), st, tcfg, 1);
    let mut start = 0;
    let last = out.join("last.safetensors");
    if last.exists() {
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("last.json"))?)?;
        load_into(be, &mut tr, &last)?;
        start = meta["epoch"].as_u64().unwrap_or(0) as usize + 1;
        tr.step = meta["step"].as_u64().unwrap_or(0);
        tr.ema.updates = tr.step;
        tr.cfg.ema_decay = meta["ema_decay"].as_f64().unwrap_or(tr.cfg.ema_decay as f64) as f32;
        println!("resumed after epoch {} (step {})", start - 1, tr.step);
    }
    if let Some(e) = std::env::var("START_EPOCH").ok().and_then(|v| v.parse::<usize>().ok()) {
        start = e;
        tr.step = e as u64 * iters;
    }
    if std::env::var("EVAL_ONLY").is_ok() {
        let v = val.as_ref().ok_or_else(|| anyhow::anyhow!("EVAL_ONLY needs VAL"))?;
        println!("mAP {:.4} on {} val images at {size}", evaluate(be, &tr.st, &cfg, v.clone(), size)?, v.len());
        return Ok(());
    }
    let mut best = (f32::NEG_INFINITY, f32::NEG_INFINITY); // (stage 1, stage 2)
    let mut rng = Rng::new(7);
    let t0 = std::time::Instant::now();
    let mut seen = 0usize;
    'outer: for epoch in start..epochs {
        let best1 = out.join("best_stg1.safetensors");
        if epoch == stop && best1.exists() {
            load_into(be, &mut tr, &best1)?;
            tr.restart_ema(0.9999);
            println!("epoch {epoch}: augmentation off — restart from best_stg1, EMA decay {}", tr.cfg.ema_decay);
        }
        let mut order: Vec<usize> = (0..samples.len()).collect();
        for _ in 0..=epoch {
            for i in (1..order.len()).rev() {
                order.swap(i, rng.next_u32() as usize % (i + 1));
            }
        }
        for b in loader_epoch(samples.clone(), order, batch, size, epoch, policy, 16, epoch as u64) {
            let b = b?;
            let log = tr.step(&b.images, &b.targets, [size, size])?;
            seen += b.targets.len();
            if tr.step % log_every == 0 || tr.step == 1 {
                let term = |n: &str| log.terms.iter().find(|t| t.0 == n).map(|t| t.1).unwrap_or(f32::NAN);
                println!(
                    "epoch {epoch} step {:6}: loss {:7.3} (mal {:.3} bbox {:.3} giou {:.3} fgl {:.3} ddf {:.4}) |g| {:7.3} lr {:.2e}  {:.1} img/s",
                    tr.step,
                    log.loss,
                    term("loss_mal"),
                    term("loss_bbox"),
                    term("loss_giou"),
                    term("loss_fgl"),
                    term("loss_ddf_aux_0"),
                    log.grad_norm,
                    log.lr,
                    seen as f64 / t0.elapsed().as_secs_f64()
                );
            }
            if tr.step as usize >= max_steps {
                break 'outer;
            }
        }
        tr.save(&last)?;
        std::fs::write(out.join("last.json"), serde_json::json!({"epoch": epoch, "step": tr.step, "ema_decay": tr.cfg.ema_decay}).to_string())?;
        if let Some(v) = &val {
            let ap = evaluate(be, &tr.ema.store, &cfg, v.clone(), size)?;
            println!("epoch {epoch}: EMA mAP {ap:.4} on {} val images", v.len());
            if epoch < stop {
                if ap > best.0 {
                    best.0 = ap;
                    tr.save(&best1)?;
                }
            } else if ap > best.1 {
                best.1 = ap;
                tr.save(&out.join("best_stg2.safetensors"))?;
            } else if best1.exists() {
                // DEIM: no gain after the stop — back to stage 1's best with a slower EMA
                load_into(be, &mut tr, &best1)?;
                tr.restart_ema(tr.cfg.ema_decay - 0.0001);
                println!("epoch {epoch}: no gain — restart from best_stg1, EMA decay {}", tr.cfg.ema_decay);
            }
        }
    }
    tr.save(&last)?;
    println!("done: {} steps, {:.1} img/s", tr.step, seen as f64 / t0.elapsed().as_secs_f64());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match std::env::var("BACKEND").as_deref() {
        Ok("cpu") => run(&ojas_learn::cpu::Cpu, &a),
        #[cfg(feature = "cuda")]
        _ => run(&ojas_learn::cuda::Cuda::new(0)?, &a),
        #[cfg(not(feature = "cuda"))]
        _ => anyhow::bail!("built without CUDA: set BACKEND=cpu"),
    }
}
