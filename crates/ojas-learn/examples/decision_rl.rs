//! Train an encoder decision model from rewards, with no dataset: tasks are made on
//! demand and judged by one or more `/v1/systemone` servers.
//!
//! ```text
//! cargo run --release -p ojas-learn --features metal --example decision_rl -- \
//!     --model ~/models/decision/Laya-Q8_0.gguf \
//!     --teacher openjev=http://127.0.0.1:8080 [--teacher kev=http://10.0.0.5:8080:0.5] \
//!     --out runs/laya-rl [--steps 500] [--batch 4] [--lr 1e-5] [--anchor 0.1] \
//!     [--families compare,json_lookup] [--requests pool_dir_or.jsonl] [--judge-known] [--freeze token_embd] \
//!     [--max-seq-tokens 1024] [--memory-gb 30] [--distill 1.0] [--lr-min 1e-6] [--judge-threads 4] \
//!     [--grow 6 --stitch-after 1500 --stitch-lr 3e-6] [--surprise-gain 0.5] [--uncertainty-gain 0.5] \
//!     [--replay 20000 --sleep-every 50 --sleep-steps 10 --consolidate 0.5]
//!
//! Requests whose lines carry `background` passages (`hippocampus.py augment --keep`)
//! train both with and without them; sleep replays the most surprising items among
//! random older ones and, with `--consolidate`, teaches the bare request what the model
//! answers when it reads the passages.
//! ```
//!
//! The teacher is swapped by naming another server; several servers with weights
//! form a committee. Judgements are cached in `<out>/teacher-cache.jsonl`.

use anyhow::{bail, Context, Result};
use ojas_decision::DecisionModel;
use ojas_formats::gguf::Gguf;
use ojas_learn::decision_rl::tasks::FAMILIES;
use ojas_learn::decision_rl::teacher::{self, Cache};
use ojas_learn::decision_rl::train::{Config, Trainer};
use ojas_learn::decision_rl::{LearnDecision, ModernBert};
use std::path::PathBuf;

#[cfg(feature = "metal")]
type Device = ojas_learn::metal::Metal;
#[cfg(all(feature = "cuda", not(feature = "metal")))]
type Device = ojas_learn::cuda::Cuda;
#[cfg(not(any(feature = "metal", feature = "cuda")))]
type Device = ojas_learn::cpu::Cpu;

fn device() -> Result<Device> {
    #[cfg(feature = "metal")]
    { ojas_learn::metal::Metal::new() }
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    { ojas_learn::cuda::Cuda::new(0) }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    { Ok(ojas_learn::cpu::Cpu) }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut model, mut teachers, mut requests) = (None, Vec::new(), Vec::new());
    let mut cfg = Config {
        steps: 200, batch: 4, lr: 1e-5, weight_decay: 0.0, anchor: 0.1, distill: 1.0, lr_min: 0.0, judge_threads: 4, clip: 1.0, variants: true,
        teacher_weight: 1.0, gold_weight: 1.0, judge_known: false, families: FAMILIES.iter().map(|f| f.to_string()).collect(),
        files: Vec::new(), eval_every: 25, eval_items: 20, checkpoint_every: 100, seed: 1,
        out: PathBuf::from("runs/decision-rl"), freeze: Vec::new(), max_seq_tokens: 1024, memory_gb: 0.0,
        train_only: Vec::new(), stitch_after: 0, stitch_lr: 3e-6, surprise_gain: 0.0, uncertainty_gain: 0.0,
        replay: 0, sleep_every: 0, sleep_steps: 0, consolidate: 0.0,
    };
    let mut grow = 0usize;
    let mut no_anchor = false;
    while let Some(a) = args.next() {
        let mut value = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = Some(value()?),
            "--teacher" => teachers.push(value()?),
            "--requests" => requests.push(PathBuf::from(value()?)),
            "--out" => cfg.out = PathBuf::from(value()?),
            "--steps" => cfg.steps = value()?.parse()?,
            "--batch" => cfg.batch = value()?.parse()?,
            "--lr" => cfg.lr = value()?.parse()?,
            "--weight-decay" => cfg.weight_decay = value()?.parse()?,
            "--anchor" => { cfg.anchor = value()?.parse()?; no_anchor = cfg.anchor <= 0.0; }
            "--clip" => cfg.clip = value()?.parse()?,
            "--distill" => cfg.distill = value()?.parse()?,
            "--lr-min" => cfg.lr_min = value()?.parse()?,
            "--judge-threads" => cfg.judge_threads = value()?.parse()?,
            "--no-variants" => cfg.variants = false,
            "--teacher-weight" => cfg.teacher_weight = value()?.parse()?,
            "--gold-weight" => cfg.gold_weight = value()?.parse()?,
            "--judge-known" => cfg.judge_known = true,
            "--families" => cfg.families = value()?.split(',').filter(|f| !f.is_empty()).map(str::to_string).collect(),
            "--eval-every" => cfg.eval_every = value()?.parse()?,
            "--eval-items" => cfg.eval_items = value()?.parse()?,
            "--checkpoint-every" => cfg.checkpoint_every = value()?.parse()?,
            "--seed" => cfg.seed = value()?.parse()?,
            "--freeze" => cfg.freeze = value()?.split(',').map(str::to_string).collect(),
            "--max-seq-tokens" => cfg.max_seq_tokens = value()?.parse()?,
            "--grow" => grow = value()?.parse()?,
            "--train-only" => cfg.train_only = value()?.split(',').filter(|p| !p.is_empty()).map(str::to_string).collect(),
            "--stitch-after" => cfg.stitch_after = value()?.parse()?,
            "--stitch-lr" => cfg.stitch_lr = value()?.parse()?,
            "--surprise-gain" => cfg.surprise_gain = value()?.parse()?,
            "--uncertainty-gain" => cfg.uncertainty_gain = value()?.parse()?,
            "--replay" => cfg.replay = value()?.parse()?,
            "--sleep-every" => cfg.sleep_every = value()?.parse()?,
            "--sleep-steps" => cfg.sleep_steps = value()?.parse()?,
            "--consolidate" => cfg.consolidate = value()?.parse()?,
            "--memory-gb" => cfg.memory_gb = value()?.parse()?,
            other => bail!("unknown argument {other}"),
        }
    }
    let model_path = model.context("--model is required")?;
    cfg.files = requests;
    if teachers.is_empty() { cfg.teacher_weight = 0.0; }
    std::fs::create_dir_all(&cfg.out)?;

    let be = device()?;
    let gpu = LearnDecision::grown(&be, grow);
    let model = DecisionModel::load(&gpu, &model_path)?;
    // A grown model trains its new blocks, its head and its scorer first, the old
    // blocks held, until `--stitch-after`; then everything, at `--stitch-lr`.
    if grow > 0 && cfg.train_only.is_empty() {
        let spec = &model.marker_backend().context("not an encoder model")?.model.spec;
        let head = spec.marker_head.as_ref().context("no decision head")?;
        cfg.train_only = (spec.layers as usize - grow..spec.layers as usize).map(|l| format!("blk.{l}."))
            .chain((head.first..head.first + head.blocks).map(|l| format!("blk.{l}.")))
            .chain(["cls.".to_string(), "output_norm".to_string(), "token_types".to_string()])
            .collect();
        eprintln!("grown by {grow} blocks; training {} until step {}", cfg.train_only.join(" "), cfg.stitch_after);
    }
    let anchor = if no_anchor { None } else {
        let mut g = Gguf::open(&model_path)?;
        Some(ModernBert::from_gguf(&be, &mut g)?)
    };
    let teacher: Box<dyn teacher::Teacher> = if teachers.is_empty() {
        Box::new(NoTeacher)
    } else {
        Box::new(Cache::open(teacher::from_specs(&teachers)?, &cfg.out.join("teacher-cache.jsonl"))?)
    };
    eprintln!("{}: {} tensors; teacher {}; families {}", model.name(),
              model.marker_backend().map_or(0, |m| m.model.params().len()), teacher.name(), cfg.families.join(","));
    let mut trainer = Trainer::new(&be, &model, anchor.as_ref(), teacher.as_ref(), &cfg, std::path::Path::new(&model_path))?;
    let report = trainer.run()?;
    if let (Some(first), Some(last)) = (report.evals.first(), report.evals.last()) {
        eprintln!("agreement before -> after:");
        for ((f, a0, _, _), (_, a1, _, _)) in first.families.iter().zip(&last.families) {
            eprintln!("  {f:<14} {:>5.1}% -> {:>5.1}%", a0 * 100.0, a1 * 100.0);
        }
    }
    for c in &report.checkpoints { eprintln!("checkpoint {}", c.display()); }
    Ok(())
}

/// Training on the families that know their answers alone.
struct NoTeacher;

impl teacher::Teacher for NoTeacher {
    fn name(&self) -> &str { "none" }
    fn judge(&self, _: &ojas_decision::json::Json) -> Result<teacher::Judgement> { bail!("no teacher was given") }
}
