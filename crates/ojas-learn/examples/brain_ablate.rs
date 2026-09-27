//! Small-scale locality and gating ablations: one variant, fine-tuned from COCO RT-DETRv2-S on a
//! fixed UVH-26 subset, EMA mAP on a fixed held-out subset after every epoch.
//! `brain_ablate <variant> <seed> coco.pth train.json image_root out_dir [epochs] [train_n] [val_n] [batch]`
//! variants: control | blocklocal | blocklocal_dec | blocklocal_mem1 | headonly | surprise |
//! surprise_k05 | surprise_k1 | random62 (skip 62% of updates at random: the gate's control)
//! Env: INIT=model.pth (default: the COCO weights argument) with INIT_PREFIX, KEEP_HEADS=1
//! (no class-head reset), LABEL_OFFSET (added to the train set's category ids),
//! EXTRA_VAL=json|root|offset|split_file|first (a second held-out set, for measuring retention:
//! images whose ids are in split_file, from line `first` on).
use std::io::Write;
use std::sync::Arc;

use ojas_learn::cuda::Cuda;
use ojas_learn::data::{load_coco, loader, Sample};
use ojas_learn::eval::{coco_map, postprocess};
use ojas_learn::models::detr_loss::Rng;
use ojas_learn::models::rtdetr::{Config, RtDetr, Store, Variant};
use ojas_learn::train::{TrainCfg, Trainer};
use ojas_learn::Tape;

fn evaluate(be: &Cuda, st: &Store<Cuda>, cfg: &Config, samples: &Arc<Vec<Sample>>) -> anyhow::Result<(f32, f32)> {
    let m = RtDetr { cfg: cfg.clone(), st, train: false, var: Variant::default() };
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
    let s = coco_map(&gts, &dets);
    Ok((s.ap, s.ap50))
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (variant, seed): (&str, u64) = (&a[1], a[2].parse()?);
    let arg = |i: usize, d: usize| a.get(i).map(|v| v.parse().unwrap()).unwrap_or(d);
    let (epochs, train_n, val_n, batch) = (arg(7, 6), arg(8, 2000), arg(9, 500), arg(10, 4));
    let out = std::path::PathBuf::from(&a[6]);
    std::fs::create_dir_all(&out)?;
    // fixed split: lowest image ids among the images on disk, frozen in a file on first use
    let split = out.join(format!("split_{train_n}_{val_n}.txt"));
    let mut all = load_coco(a[4].as_ref(), a[5].as_ref())?;
    let offset: usize = std::env::var("LABEL_OFFSET").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    all.iter_mut().for_each(|s| s.boxes.iter_mut().for_each(|b| b.0 += offset));
    let ids: Vec<i64> = if let Ok(s) = std::fs::read_to_string(&split) {
        s.lines().map(|l| l.parse().unwrap()).collect()
    } else {
        let mut ids: Vec<i64> = all.iter().filter(|s| !s.boxes.is_empty()).map(|s| s.image_id).collect();
        ids.sort();
        ids.truncate(train_n + val_n);
        std::fs::write(&split, ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("\n"))?;
        ids
    };
    let by: std::collections::HashMap<i64, &Sample> = all.iter().map(|s| (s.image_id, s)).collect();
    let pick = |r: &[i64]| -> Arc<Vec<Sample>> { Arc::new(r.iter().map(|i| by[i].clone()).collect()) };
    let (train, val) = (pick(&ids[..train_n]), pick(&ids[train_n..]));
    let be = Cuda::new(0)?;
    let cfg = Config::r18vd(15);
    let init = std::env::var("INIT").unwrap_or_else(|_| a[3].clone());
    let prefix = std::env::var("INIT_PREFIX").unwrap_or_else(|_| "ema.module.".into());
    let mut st = Store::from_tensors(&be, &ojas_formats::pth::load(&std::fs::read(&init)?)?, &prefix);
    if std::env::var("KEEP_HEADS").map_or(true, |v| v != "1") {
        st.reset_heads(&be, cfg.num_classes, cfg.dec_layers, cfg.hidden, seed);
    }
    // a second held-out set (retention)
    let extra: Option<Arc<Vec<Sample>>> = match std::env::var("EXTRA_VAL") {
        Ok(v) => {
            let p: Vec<&str> = v.split('|').collect();
            let mut e = load_coco(p[0].as_ref(), p[1].as_ref())?;
            let off: usize = p[2].parse()?;
            e.iter_mut().for_each(|s| s.boxes.iter_mut().for_each(|b| b.0 += off));
            let want: Vec<i64> = std::fs::read_to_string(p[3])?.lines().skip(p[4].parse()?).map(|l| l.parse().unwrap()).collect();
            let by: std::collections::HashMap<i64, Sample> = e.into_iter().map(|s| (s.image_id, s)).collect();
            Some(Arc::new(want.iter().filter_map(|i| by.get(i).cloned()).collect()))
        }
        Err(_) => None,
    };
    // short schedule: same warmup / EMA warmup for every variant
    let mut tc = TrainCfg::rtdetrv2_uvh26();
    tc.warmup_steps = 200;
    tc.ema_warmup = 200.0;
    let mut tr = Trainer::new(&be, cfg.clone(), st, tc, seed);
    match variant {
        "control" => {}
        "blocklocal" => {
            tr.var.detach_memory_from = Some(0);
            tr.var.detach_layers = true;
        }
        // the first decoder layer keeps its gradient link to the encoder
        "blocklocal_mem1" => {
            tr.var.detach_memory_from = Some(1);
            tr.var.detach_layers = true;
        }
        "blocklocal_dec" => tr.var.detach_layers = true,
        "headonly" => tr.var.frozen = vec!["backbone.".into(), "encoder.".into()],
        "surprise" => tr.gate = Some(ojas_core::Gate::new(0.0)),
        "surprise_k1" => tr.gate = Some(ojas_core::Gate::new(1.0)),
        "surprise_k05" => tr.gate = Some(ojas_core::Gate::new(0.5)),
        "random62" => tr.random_skip = Some(0.62),
        v => anyhow::bail!("unknown variant {v}"),
    }
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(out.join("results.tsv"))?;
    let tag = std::env::var("TAG").unwrap_or_default();
    let extra_eval = |st: &Store<Cuda>| -> anyhow::Result<String> {
        Ok(match &extra {
            Some(x) => {
                let (ap, ap50) = evaluate(&be, st, &cfg, x)?;
                format!("\t{ap:.4}\t{ap50:.4}")
            }
            None => String::new(),
        })
    };
    if std::env::var("EVAL0").is_ok_and(|v| v == "1") {
        let (ap, ap50) = evaluate(&be, &tr.ema.store, &cfg, &val)?;
        let line = format!("{variant}{tag}\t{seed}\t0\t{ap:.4}\t{ap50:.4}\t-\t0\t0\t0{}", extra_eval(&tr.ema.store)?);
        println!("{line}");
        writeln!(log, "{line}")?;
    }
    let t0 = std::time::Instant::now();
    let mut rng = Rng::new(seed ^ 0x5eed);
    let mut seen = 0;
    for epoch in 1..=epochs {
        let mut order: Vec<usize> = (0..train.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.next_u32() as usize % (i + 1));
        }
        let mut loss_sum = 0.0f64;
        let mut n = 0;
        for b in loader(train.clone(), order, batch, 640, true, 16, seed * 1000 + epoch as u64) {
            let b = b?;
            let l = tr.step(&b.images, &b.targets, [640, 640])?;
            loss_sum += l.loss as f64;
            n += 1;
            seen += b.targets.len();
        }
        let (ap, ap50) = evaluate(&be, &tr.ema.store, &cfg, &val)?;
        let line = format!("{variant}{tag}\t{seed}\t{epoch}\t{ap:.4}\t{ap50:.4}\t{:.3}\t{}\t{}\t{:.0}{}", loss_sum / n as f64, tr.step, tr.updates, t0.elapsed().as_secs_f64(), extra_eval(&tr.ema.store)?);
        println!("{line}  ({:.1} img/s)", seen as f64 / t0.elapsed().as_secs_f64());
        writeln!(log, "{line}")?;
    }
    Ok(())
}
