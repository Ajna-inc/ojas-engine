//! Zero-initialized correction added to frozen detector passenger logits. Usage: passenger_residual_train config.json
//! Config: manifest, init, out, seed, epochs, batch, size, lr, head_lr, balanced,
//! overfit_steps (0 for regular training). A .safetensors init starts a new fine-tune from saved
//! model weights with fresh optimizer and RNG state; it is not a resume.
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use anyhow::{ensure, Context, Result};
use ojas_learn::{Backend, Param, Tape};
use ojas_learn::cuda::Cuda;
use ojas_learn::models::{detr_loss::Rng, rtdetr::Store};
use ojas_learn::passenger;

#[derive(Clone)]
struct Row { path: PathBuf, label: usize, group: String, split: String, base: [f32; 4] }

fn batch(rows: &[Row], order: &[usize], size: usize, mut rng: Option<&mut Rng>) -> Result<(Vec<f32>, Vec<usize>, Vec<f32>)> {
    let mut images = Vec::with_capacity(order.len() * 3 * size * size);
    let mut labels = Vec::with_capacity(order.len());
    let mut base = Vec::with_capacity(order.len() * 4);
    for &i in order {
        let img = image::open(&rows[i].path).with_context(|| rows[i].path.display().to_string())?.to_rgb8();
        let (flip, brightness) = match rng.as_deref_mut() { Some(r) => (r.uniform() < 0.5, 0.85 + 0.3 * r.uniform()), None => (false, 1.) };
        images.extend(passenger::preprocess(&img, size, flip, brightness));
        labels.push(rows[i].label);
        base.extend(rows[i].base);
    }
    Ok((images, labels, base))
}

fn evaluate(be: &Cuda, st: &Store<Cuda>, rows: &[Row], batch_size: usize, size: usize, output: &Path) -> Result<serde_json::Value> {
    let (mut predictions, mut labels) = (vec![], vec![]);
    let mut saved = std::io::BufWriter::new(std::fs::File::create(output)?);
    let order: Vec<_> = (0..rows.len()).collect();
    for indices in order.chunks(batch_size) {
        let (images, y, base) = batch(rows, indices, size, None)?;
        let mut t = Tape::new(be);
        let x = t.input(&images, &[indices.len(), 3, size, size]);
        let delta = passenger::forward(&mut t, st, x)?;
        let original = t.input(&base, &[indices.len(), 4]);
        let logits = t.add(delta, original)?;
        let probs = passenger::probabilities(&t.value(logits));
        ensure!(probs.iter().flatten().all(|v| v.is_finite()), "nonfinite predictions");
        for (&i, p) in indices.iter().zip(&probs) {
            writeln!(saved, "{}", serde_json::json!({"path":rows[i].path,"group":rows[i].group,"label":rows[i].label,"probabilities":p}))?;
        }
        predictions.extend(probs);
        labels.extend(y);
    }
    saved.flush()?;
    Ok(passenger::metrics(&predictions, &labels))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 2, "passenger_residual_train config.json");
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let string = |key: &str| -> Result<&str> { config[key].as_str().with_context(|| format!("missing {key}")) };
    let integer = |key: &str, default: usize| config[key].as_u64().map(|n| n as usize).unwrap_or(default);
    let (seed, epochs, batch_size, size, overfit) = (integer("seed", 1) as u64, integer("epochs", 3), integer("batch", 16), integer("size", 224), integer("overfit_steps", 0));
    let lr = config["lr"].as_f64().unwrap_or(3e-5) as f32;
    let head_lr = config["head_lr"].as_f64().unwrap_or(3e-4) as f32;
    let balanced = config["balanced"].as_bool().unwrap_or(false);
    let epoch_samples = integer("epoch_samples", 0);
    ensure!(epoch_samples == 0 || balanced, "fixed draw budget currently requires balanced sampling");
    let stage = integer("backbone_stage", 2);
    let head_epochs = integer("head_only_epochs", 0);
    let sam_rho = config["sam_rho"].as_f64().unwrap_or(0.) as f32;
    ensure!(sam_rho.is_finite() && sam_rho >= 0., "invalid sam_rho");
    ensure!(sam_rho == 0. || head_epochs == 0, "test SAM separately from head-first training");
    ensure!(head_epochs < epochs, "head_only_epochs must leave at least one joint epoch");
    ensure!((1..=3).contains(&stage), "backbone_stage must be 1..=3");
    ensure!(batch_size > 0 && size >= 32 && size % 32 == 0 && epochs > 0 && lr > 0. && head_lr > 0., "invalid training settings");
    let mut all = vec![];
    let mut groups = HashMap::new();
    let mut paths = HashSet::new();
    let mut hashes = HashMap::new();
    for line in std::fs::read_to_string(string("manifest")?)?.lines() {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let label = v["label"].as_u64().context("label")? as usize;
        ensure!(label < 4 && v["class_name"] == passenger::NAMES[label], "category mismatch");
        let split = v["split"].as_str().context("split")?.to_owned();
        if split == "excluded_duplicate" { continue; }
        ensure!(split == "train" || split == "dev", "unknown split");
        let raw = v["detector_logits"].as_array().context("residual training requires original detector logits")?;
        ensure!(raw.len() == 15, "expected original 15-class query logits");
        let base: [f32; 4] = std::array::from_fn(|c| raw[c+1].as_f64().unwrap() as f32);
        ensure!(base.iter().all(|x| x.is_finite()), "nonfinite base logits");
        let row = Row { base, path: v["path"].as_str().context("path")?.into(), label, group: v["group"].as_str().context("group")?.into(), split };
        ensure!(paths.insert(row.path.clone()), "duplicate crop path");
        if let Some(old) = groups.insert(row.group.clone(), row.split.clone()) { ensure!(old == row.split, "source group crosses splits"); }
        for key in ["source_sha256", "crop_sha256"] {
            let hash = v[key].as_str().with_context(|| format!("missing {key}"))?;
            if let Some(old) = hashes.insert((key.to_owned(), hash.to_owned()), row.split.clone()) { ensure!(old == row.split, "duplicate content crosses splits"); }
        }
        all.push(row);
    }
    let mut train: Vec<_> = all.iter().filter(|r| r.split == "train").cloned().collect();
    let mut dev: Vec<_> = all.iter().filter(|r| r.split == "dev").cloned().collect();
    for c in 0..4 { ensure!(train.iter().any(|r| r.label == c) && dev.iter().any(|r| r.label == c), "class missing in partition"); }
    if overfit > 0 {
        train = (0..4).flat_map(|c| train.iter().filter(move |r| r.label == c).take(2).cloned()).collect();
        dev = train.clone(); // explicit correctness gate, NEVER a held-out score
    }
    let out = PathBuf::from(string("out")?);
    std::fs::create_dir(&out).context("output must be a new directory under an existing parent")?;
    std::fs::write(out.join("config.json"), serde_json::to_vec_pretty(&config)?)?;
    let be = Cuda::new(0)?;
    let init = string("init")?;
    let mut st = if init.ends_with(".safetensors") {
        let st = Store::from_safetensors(&be, init, "model.")?;
        ensure!(st.params.get("classifier.weight").is_some_and(|p| p.shape == [4, 64usize << stage]), "checkpoint width does not match backbone_stage");
        st
    } else { passenger::init_stage(&be, init, seed, stage)? };
    ensure!(!init.ends_with(".safetensors") && head_epochs == 0 && sam_rho == 0., "residual experiment starts from original backbone, without SAM or head-first changes");
    let channels = 64usize << stage;
    st.params.insert("classifier.weight".into(), Param::new(&be, "classifier.weight", &[4, channels], &vec![0.; 4*channels]));
    st.params.insert("classifier.bias".into(), Param::new(&be, "classifier.bias", &[4], &[0.; 4]));
    let parameters: usize = st.params.values().map(|p| p.shape.iter().product::<usize>()).sum();
    std::fs::write(out.join("architecture.json"), serde_json::to_vec_pretty(&serde_json::json!({"backbone_stage":stage,"parameters":parameters,"channels":64usize << stage}))?)?;
    let initial: HashMap<String, Vec<f32>> = if head_epochs > 0 {
        st.params.iter().map(|(name, p)| (name.clone(), be.download(&p.val))).collect()
    } else { HashMap::new() };
    let mut rng = Rng::new(seed);
    let mut log = std::fs::File::create(out.join("metrics.jsonl"))?;
    passenger::save(&be, &st, &out.join("zero.safetensors"))?;
    let baseline = evaluate(&be, &st, &dev, batch_size, size, &out.join("epoch0_predictions.jsonl"))?;
    writeln!(log, "{}", serde_json::json!({"epoch":0,"metrics":baseline}))?;
    println!("train {} dev {} seed {seed}; initial metrics {baseline}", train.len(), dev.len());
    let mut best = baseline["macro_f1"].as_f64().unwrap();
    passenger::save(&be, &st, &out.join("best.safetensors"))?;
    std::fs::write(out.join("best.json"), serde_json::to_vec_pretty(&serde_json::json!({"epoch":0,"step":0,"metrics":baseline,"zero_correction":true}))?)?;
    let mut step = 0u32;
    let t0 = std::time::Instant::now();
    let pools: Vec<Vec<usize>> = (0..4).map(|c| (0..train.len()).filter(|&i| train[i].label == c).collect()).collect();
    let epoch_count = if overfit > 0 && head_epochs == 0 { 1 } else { epochs };
    let mut backbone_step = 0u32;
    let mut visited = HashSet::new();
    for epoch in 1..=epoch_count {
        let head_only = epoch <= head_epochs;
        let mut order: Vec<usize> = if overfit > 0 { (0..overfit).flat_map(|_| 0..train.len()).collect() }
            else if balanced { (0..if epoch_samples > 0 { epoch_samples } else { train.len() }).map(|i| { let pool = &pools[i % 4]; pool[rng.next_u32() as usize % pool.len()] }).collect() }
            else { (0..train.len()).collect() };
        if overfit == 0 { for i in (1..order.len()).rev() { order.swap(i, rng.next_u32() as usize % (i + 1)); } }
        let effective_batch = if overfit > 0 { train.len() } else { batch_size };
        let mut loss_sum = 0.;
        let mut perturbed_loss_sum = 0.;
        let mut seen = 0usize;
        for indices in order.chunks(effective_batch) {
            visited.extend(indices.iter().copied());
            let (images, labels, base) = batch(&train, indices, size, if overfit > 0 { None } else { Some(&mut rng) })?;
            let mut t = Tape::new(&be);
            let x = t.input(&images, &[indices.len(), 3, size, size]);
            let delta = passenger::forward(&mut t, &st, x)?;
            let original = t.input(&base, &[indices.len(), 4]);
            let logits = t.add(delta, original)?;
            let loss = passenger::cross_entropy(&mut t, logits, &labels)?;
            let value = t.value(loss)[0];
            ensure!(value.is_finite(), "nonfinite loss at step {step}");
            t.backward(loss)?;
            let mut grads: Vec<_> = st.params.values()
                .filter(|p| !head_only || p.name.starts_with("classifier."))
                .filter_map(|p| t.param_var(p).and_then(|v| t.grad(v)).map(|g| (p, g))).collect();
            let norm = be.alloc(1);
            for (_, g) in &grads { be.sumsq(g, &norm, true); }
            let mut norm = be.download(&norm)[0].sqrt();
            ensure!(norm.is_finite(), "nonfinite gradients at step {step}");
            if step == 0 {
                let backbone_norm = be.alloc(1);
                for (p, g) in &grads {
                    if !p.name.starts_with("classifier.") { be.sumsq(g, &backbone_norm, true); }
                }
                let backbone_norm = be.download(&backbone_norm)[0];
                ensure!(norm > 0. && backbone_norm == 0., "zero-head gradient gate failed");
                std::fs::write(out.join("zero_head_gradient_gate.json"), serde_json::to_vec_pretty(&serde_json::json!({"total_gradient_norm":norm,"backbone_squared_norm":backbone_norm,"head_receives_gradient":true}))?)?;
            }
            if sam_rho > 0. {
                let first_norm = norm;
                let mut perturbation = passenger::SamPerturbation::new(&be, &grads, sam_rho, norm)?;
                let delta_norm = if step == 0 { Some(perturbation.delta_norm()) } else { None };
                drop(grads);
                drop(t);
                t = Tape::new(&be);
                let second_pass: Result<f32> = (|| {
                    let x = t.input(&images, &[indices.len(), 3, size, size]);
                    let delta = passenger::forward(&mut t, &st, x)?;
            let original = t.input(&base, &[indices.len(), 4]);
            let logits = t.add(delta, original)?;
                    let loss = passenger::cross_entropy(&mut t, logits, &labels)?;
                    let value = t.value(loss)[0];
                    ensure!(value.is_finite(), "nonfinite SAM loss at step {step}");
                    t.backward(loss)?;
                    Ok(value)
                })();
                perturbation.restore();
                if step == 0 {
                    ensure!(perturbation.restoration_is_exact(), "SAM weight restoration failed");
                    if first_norm > 0. { ensure!((delta_norm.unwrap() - sam_rho).abs() < 1e-4, "SAM radius mismatch"); }
                    std::fs::write(out.join("sam_gate.json"), serde_json::to_vec_pretty(&serde_json::json!({"rho":sam_rho,"first_gradient_norm":first_norm,"actual_delta_norm":delta_norm,"exact_restoration":true,"same_augmented_batch":true}))?)?;
                }
                drop(perturbation);
                perturbed_loss_sum += second_pass? as f64 * indices.len() as f64;
                grads = st.params.values().filter_map(|p| t.param_var(p).and_then(|v| t.grad(v)).map(|g| (p, g))).collect();
                let sum = be.alloc(1);
                for (_, g) in &grads { be.sumsq(g, &sum, true); }
                norm = be.download(&sum)[0].sqrt();
                ensure!(norm.is_finite(), "nonfinite second SAM gradient");
            }
            if norm > 5. { for (_, g) in &grads { be.scale(g, 5. / norm); } }
            step += 1;
            if !head_only { backbone_step += 1; }
            let warmup = (step as f32 / 20.).min(1.);
            for (p, g) in grads {
                let is_head = p.name.starts_with("classifier.");
                let rate = if is_head { head_lr } else { lr };
                let param_step = if is_head { step } else { backbone_step };
                let param_warmup = if is_head { warmup } else { (param_step as f32 / 20.).min(1.) };
                let decay = if p.name.ends_with("bias") || p.name.contains("norm") { 0. } else { 1e-4 };
                be.adamw(&p.val, g, &p.m, &p.v, rate * param_warmup, 0.9, 0.999, 1e-8, decay, param_step);
            }
            loss_sum += value as f64 * indices.len() as f64;
            seen += indices.len();
            if step % 25 == 0 { println!("epoch {epoch} step {step} loss {value:.4} norm {norm:.3} elapsed {:.1}s", t0.elapsed().as_secs_f64()); }
        }
        let metrics = evaluate(&be, &st, &dev, batch_size, size, &out.join(format!("epoch{epoch}_predictions.jsonl")))?;
        println!("epoch {epoch} metrics {metrics}");
        writeln!(log, "{}", serde_json::json!({"epoch":epoch,"step":step,"backbone_step":backbone_step,"head_only":head_only,"sam_rho":sam_rho,"backward_passes":step * if sam_rho > 0. { 2 } else { 1 },"train_loss":loss_sum/seen as f64,"perturbed_loss":if sam_rho > 0. { Some(perturbed_loss_sum/seen as f64) } else { None },"elapsed_seconds":t0.elapsed().as_secs_f64(),"metrics":metrics}))?;
        log.flush()?;
        if head_epochs > 0 && epoch == head_epochs {
            let mut head_changed = false;
            for (name, p) in &st.params {
                let values = be.download(&p.val);
                if name.starts_with("classifier.") { head_changed |= values != initial[name]; }
                else { ensure!(values == initial[name], "frozen backbone changed: {name}"); }
            }
            ensure!(head_changed && backbone_step == 0, "head-only phase failed");
            std::fs::write(out.join("head_phase_gate.json"), serde_json::to_vec_pretty(&serde_json::json!({"epoch":epoch,"step":step,"backbone_unchanged":true,"head_changed":true,"backbone_step":backbone_step}))?)?;
        }
        if metrics["macro_f1"].as_f64().unwrap() > best {
            best = metrics["macro_f1"].as_f64().unwrap();
            passenger::save(&be, &st, &out.join("best.safetensors"))?;
            std::fs::write(out.join("best.json"), serde_json::to_vec_pretty(&serde_json::json!({"epoch":epoch,"step":step,"metrics":metrics}))?)?;
        }
        if overfit > 0 && !head_only { ensure!(metrics["accuracy"].as_f64().unwrap() == 1. && metrics["nll"].as_f64().unwrap() < 0.1, "tiny-batch gate failed"); }
    }
    if head_epochs > 0 {
        let changed = st.params.iter().filter(|(n, _)| !n.starts_with("classifier."))
            .any(|(n, p)| be.download(&p.val) != initial[n]);
        ensure!(changed && backbone_step > 0, "joint phase did not update backbone");
        std::fs::write(out.join("joint_phase_gate.json"), serde_json::to_vec_pretty(&serde_json::json!({"backbone_changed":changed,"backbone_step":backbone_step,"head_step":step}))?)?;
    }
    passenger::save(&be, &st, &out.join("final.safetensors"))?;
    std::fs::write(out.join("exposure.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "pool_crops":train.len(), "unique_sampled_crops":visited.len(),
        "unique_sampled_groups":visited.iter().map(|&i|train[i].group.as_str()).collect::<HashSet<_>>().len(),
        "updates":step, "epoch_samples":epoch_samples,
        "note":"balanced sampling with replacement; epochs with epoch_samples are draw-budget intervals, not full passes"
    }))?)?;
    if config["evaluate_train"].as_bool().unwrap_or(false) {
        let metrics = evaluate(&be, &st, &train, batch_size, size, &out.join("final_train_predictions.jsonl"))?;
        std::fs::write(out.join("final_train_metrics.json"), serde_json::to_vec_pretty(&metrics)?)?;
    }
    println!("complete: {step} updates; best dev macro-F1 {best:.4}; checkpoints are inference weights, not resumable optimizer state");
    Ok(())
}
