//! The training loop: an encoder decision model learns from rewards on questions
//! made on demand, with no dataset.
//!
//! Each question's options are a group. Every option gets a reward — the teacher's
//! probability for it, the right answer where the task knows one, or a mix — and
//! its advantage is the reward less the group's mean. The policy loss is the negative
//! expected advantage under the model's own calibrated distribution: every option's
//! reward is known, so the policy gradient is taken over all of them, weighted by
//! their probabilities, rather than over sampled answers. It is bounded, so the
//! model gains nothing from driving an option's probability to zero once the mass
//! sits on the rewarded options; a KL term to the model as it was loaded (the
//! anchor) keeps it from drifting on what it already answers well. Families the
//! model gets wrong most are drawn most (a curriculum with a floor, so none is
//! forgotten), and every request is also asked in rewritten forms that keep its
//! answers. Every pass runs under a memory budget ([`super::memory`]): a step is
//! split into passes that fit it, with the gradients summed across them.

use super::export::{export, fit_temperatures, Scored};
use super::memory::{self, Budget};
use super::tasks::{self, FileSource, Rng, Task};
use super::teacher::{Judgement, Teacher};
use super::{LearnDecision, MarkerSeq, ModernBert};
use crate::backend::Backend;
use crate::tape::{AdamW, Param, Tape, Var};
use crate::Unary;
use anyhow::{ensure, Context, Result};
use ojas_decision::json::Json;
use ojas_decision::{DecisionModel, Question, QuestionKind};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Config {
    pub steps: usize,
    /// Tasks drawn per step; each adds its rewrites.
    pub batch: usize,
    pub lr: f32,
    pub weight_decay: f32,
    /// Weight of the KL term to the anchor; 0 trains without one.
    pub anchor: f32,
    /// Weight of the distillation term: the cross-entropy of the model's distribution
    /// to the reward distribution, which uses the teacher's probability for every
    /// option rather than only which option is best.
    pub distill: f32,
    /// Learning rate at the last step; the rate follows a cosine from `lr` to it.
    pub lr_min: f32,
    /// Teacher calls in flight at once.
    pub judge_threads: usize,
    /// Largest gradient norm applied; larger gradients are scaled down to it.
    pub clip: f32,
    pub variants: bool,
    /// Weight of the teacher's probabilities in an option's reward.
    pub teacher_weight: f64,
    /// Weight of the right answer, where the task knows it.
    pub gold_weight: f64,
    /// Ask the teacher about questions whose answer is known too; otherwise the
    /// teacher judges only the questions without one.
    pub judge_known: bool,
    /// Task families to draw from.
    pub families: Vec<String>,
    /// Request files, each its own family; a directory stands for every file in it.
    pub files: Vec<PathBuf>,
    pub eval_every: usize,
    /// Held-out questions per family.
    pub eval_items: usize,
    pub checkpoint_every: usize,
    pub seed: u64,
    pub out: PathBuf,
    /// Tensor-name prefixes left untrained.
    pub freeze: Vec<String>,
    /// Longest prompt trained on, in tokens; longer ones are left out.
    pub max_seq_tokens: usize,
    /// Activation memory a pass may take, in GB; 0 is a third of the machine's.
    pub memory_gb: f64,
}

/// What a run reports: the loss and agreement of each step, and each evaluation.
#[derive(Default, Debug)]
pub struct Report {
    pub steps: Vec<StepStats>,
    pub evals: Vec<EvalStats>,
    pub checkpoints: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct StepStats {
    pub step: usize,
    pub loss: f64,
    /// Share of questions whose most probable option is the most rewarded one.
    pub agreement: f64,
    pub questions: usize,
    /// Questions left out: prompts too long, or passes the memory guard stopped.
    pub skipped: usize,
    pub seconds: f64,
}

#[derive(Clone, Debug)]
pub struct EvalStats {
    pub step: usize,
    /// Per family: `(agreement with the reward, mean cross-entropy to it, questions)`.
    pub families: Vec<(String, f64, f64, usize)>,
}

/// One question ready to train on: its prompt, its reward per option in the prompt's
/// option order, its calibration temperature, and its anchor distribution once computed.
struct Item {
    family: String,
    seq: MarkerSeq,
    kind: QuestionKind,
    reward: Vec<f64>,
    temperature: f32,
    anchor: Option<Vec<f64>>,
}

/// Rewards per option of `q` from the teacher's judgement and the right answer.
fn rewards(q: &Question, judged: Option<&HashMap<String, f64>>, gold: Option<&str>, cfg: &Config) -> Result<Vec<f64>> {
    let n = q.options.len();
    let mut r = vec![0.0; n];
    let mut weight = 0.0;
    if let (Some(j), true) = (judged, cfg.teacher_weight > 0.0) {
        for (i, o) in q.options.iter().enumerate() {
            r[i] += cfg.teacher_weight * j.get(&o.key).copied().unwrap_or(0.0);
        }
        weight += cfg.teacher_weight;
    }
    if let (Some(g), true) = (gold, cfg.gold_weight > 0.0) {
        let i = q.options.iter().position(|o| o.key == g).with_context(|| format!("questions.{}: no option {g}", q.id))?;
        r[i] += cfg.gold_weight;
        weight += cfg.gold_weight;
    }
    ensure!(weight > 0.0, "questions.{}: no reward (no teacher judgement and no known answer)", q.id);
    Ok(r.iter().map(|v| v / weight).collect())
}

fn argmax(v: &[f64]) -> usize { v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b }) }

/// The trainer's view of one model: the decision model that builds prompts and the
/// encoder underneath it that the gradients update.
pub struct Trainer<'m, 'b, B: Backend> {
    be: &'b B,
    model: &'m DecisionModel<'m, LearnDecision<'b, B>>,
    anchor: Option<&'m ModernBert<B>>,
    teacher: &'m dyn Teacher,
    cfg: &'m Config,
    source: PathBuf,
    files: Vec<FileSource>,
    /// Per family, a running mean of how often the model disagrees with the reward;
    /// ordered, so a seed reproduces a run's draws.
    error: BTreeMap<String, f64>,
    rng: Rng,
    opt: AdamW,
    budget: Budget,
    /// Summed gradients of the step's passes, one per parameter, in parameter order.
    grads: Vec<Option<B::Buf>>,
    /// The held-out requests, by their canonical text, which training never draws.
    held: HashSet<String>,
}

impl<'m, 'b, B: Backend> Trainer<'m, 'b, B> {
    pub fn new(be: &'b B, model: &'m DecisionModel<'m, LearnDecision<'b, B>>, anchor: Option<&'m ModernBert<B>>,
               teacher: &'m dyn Teacher, cfg: &'m Config, source: &Path) -> Result<Self> {
        ensure!(model.marker_backend().is_some(), "{} is not an encoder model", model.name());
        for f in &cfg.families { ensure!(tasks::FAMILIES.contains(&f.as_str()), "no task family {f}"); }
        let mut files = Vec::new();
        for p in &cfg.files {
            if p.is_dir() { files.extend(FileSource::open_dir(p)?); } else { files.push(FileSource::open(p)?); }
        }
        files.retain(|f| {
            if f.is_empty() { eprintln!("{}: no requests, left out", f.family()); }
            !f.is_empty()
        });
        ensure!(!cfg.families.is_empty() || files.iter().any(|f| !f.is_empty()), "nothing to train on: no families and no requests");
        std::fs::create_dir_all(&cfg.out)?;
        let names: Vec<String> = cfg.families.iter().cloned().chain(files.iter().map(|f| f.family().to_string())).collect();
        let error = names.into_iter().map(|n| (n, 1.0)).collect();
        let opt = AdamW { lr: cfg.lr, wd: cfg.weight_decay, ..AdamW::default() };
        let budget = Budget::new(cfg.memory_gb);
        eprintln!("memory: {:.1} GB per pass, stopping a step past {:.1} GB resident; prompts up to {} tokens",
                  budget.pass as f64 / 1e9, budget.resident as f64 / 1e9, cfg.max_seq_tokens);
        let n = model.marker_backend().expect("checked above").model.params().len();
        Ok(Trainer { be, model, anchor, teacher, cfg, source: source.to_path_buf(), files, error, rng: Rng::new(cfg.seed), opt,
                     budget, grads: (0..n).map(|_| None).collect(), held: HashSet::new() })
    }

    /// The passes `items` are run in under the budget, and the items left out of them.
    fn passes(&self, items: &[Item], train: bool) -> (Vec<std::ops::Range<usize>>, usize) {
        let lengths: Vec<usize> = items.iter().map(|it| it.seq.ids.len()).collect();
        let (passes, dropped) = memory::passes(&self.bert().spec, &lengths, self.budget.pass, train);
        (passes, dropped.len())
    }

    /// Whether another pass may start: the process is under the resident limit.
    fn may_start_pass(&self, what: &str) -> bool {
        let resident = memory::resident_bytes();
        if resident <= self.budget.resident { return true; }
        self.be.trim();
        let resident = memory::resident_bytes();
        if resident <= self.budget.resident { return true; }
        eprintln!("memory guard: {:.1} GB resident, {what} stopped", resident as f64 / 1e9);
        false
    }

    fn bert(&self) -> &'m ModernBert<B> { &self.model.marker_backend().expect("checked at construction").model }

    /// A family, drawn in proportion to a floor plus the model's error on it.
    fn draw_family(&mut self) -> String {
        let weights: Vec<(String, f64)> = self.error.iter().map(|(f, e)| (f.clone(), 0.2 + e)).collect();
        let total: f64 = weights.iter().map(|(_, w)| w).sum();
        let mut x = (self.rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
        for (f, w) in &weights {
            if x < *w { return f.clone(); }
            x -= w;
        }
        weights.last().map(|(f, _)| f.clone()).expect("a family")
    }

    /// A task of `family` that is not held out; after many draws that are, the last.
    fn draw_task(&mut self, family: &str) -> Task {
        let mut task = self.draw_any(family);
        for _ in 0..32 {
            if !self.held.contains(&canonical(&task.body)) { break; }
            task = self.draw_any(family);
        }
        task
    }

    fn draw_any(&mut self, family: &str) -> Task {
        if let Some(f) = self.files.iter().find(|f| f.family() == family) {
            return f.draw(&mut self.rng);
        }
        tasks::generate(family, &mut self.rng)
    }

    /// `task` and its rewrites as items, judged by the teacher where a reward needs it.
    fn items(&mut self, task: &Task, rewrites: bool) -> Result<Vec<Item>> {
        let mut bodies = vec![task.body.clone()];
        if rewrites { bodies.extend(tasks::variants(&task.body, &mut self.rng)); }
        let mut items = Vec::new();
        let needs_teacher = self.cfg.teacher_weight > 0.0 && (self.cfg.judge_known || task.gold.iter().any(Option::is_none));
        // The teacher is asked once per distinct request of the task, the requests at
        // once: a rewrite has the same answers as its original and shares its judgement.
        let judgements: Vec<Option<Judgement>> = if needs_teacher {
            let teacher = self.teacher;
            let keys: Vec<String> = bodies.iter().map(canonical).collect();
            let mut distinct: Vec<usize> = Vec::new();
            for (i, k) in keys.iter().enumerate() {
                if !distinct.iter().any(|&j| keys[j] == *k) { distinct.push(i); }
            }
            let judged: Vec<Judgement> = std::thread::scope(|scope| {
                let handles: Vec<_> = distinct.chunks(distinct.len().div_ceil(self.cfg.judge_threads.max(1))).map(|chunk| {
                    let bodies = &bodies;
                    scope.spawn(move || chunk.iter().map(|&i| teacher.judge(&bodies[i])).collect::<Vec<Result<Judgement>>>())
                }).collect();
                handles.into_iter().flat_map(|h| h.join().expect("a judging thread")).collect::<Result<Vec<_>>>()
            })?;
            keys.iter().map(|k| Some(judged[distinct.iter().position(|&j| keys[j] == *k).expect("every key is judged")].clone())).collect()
        } else {
            bodies.iter().map(|_| None).collect()
        };
        for (body, judged) in bodies.iter().zip(judgements) {
            let req = self.model.request(body)?;
            self.model.validate(&req)?;
            ensure!(judged.is_some() || task.gold.iter().all(Option::is_some),
                "{}: a question has no known answer and no teacher to judge it", task.family);
            let prompts = self.model.marker_prompts(&req)?;
            for (i, (q, p)) in req.questions.iter().zip(prompts).enumerate() {
                if p.ids.len() > self.cfg.max_seq_tokens { continue; }
                let reward = rewards(q, judged.as_ref().map(|j| &j[i]), task.gold.get(i).and_then(|g| g.as_deref()), self.cfg)?;
                items.push(Item {
                    family: task.family.clone(), kind: q.kind, reward, temperature: p.temperature, anchor: None,
                    seq: MarkerSeq { ids: p.ids, qtype: p.qtype, markers: p.markers },
                });
            }
        }
        Ok(items)
    }

    /// The anchor model's calibrated distribution for each item, computed once.
    fn anchor_distributions(&self, items: &mut [Item]) -> Result<()> {
        let Some(anchor) = self.anchor else { return Ok(()) };
        let pending: Vec<usize> = items.iter().enumerate().filter(|(_, it)| it.anchor.is_none()).map(|(i, _)| i).collect();
        if pending.is_empty() { return Ok(()); }
        let lengths: Vec<usize> = pending.iter().map(|&i| items[i].seq.ids.len()).collect();
        let (passes, _) = memory::passes(&self.bert().spec, &lengths, self.budget.pass, false);
        for pass in passes {
            if !self.may_start_pass("the anchor's pass") { break; }
            let seqs: Vec<MarkerSeq> = pending[pass.clone()].iter().map(|&i| items[i].seq.clone()).collect();
            let mut t = Tape::new(self.be);
            let scores = anchor.scores(&mut t, &seqs)?;
            let values = t.value(scores);
            let mut at = 0;
            for &i in &pending[pass] {
                let n = items[i].seq.markers.len();
                items[i].anchor = Some(softmax(&values[at..at + n], items[i].temperature));
                at += n;
            }
        }
        Ok(())
    }

    /// The loss over `items` on one tape, scaled by `1 / divisor` rather than by their
    /// count, and each question's probabilities.
    fn loss(&self, t: &mut Tape<'_, B>, items: &[Item], divisor: usize) -> Result<(Var, Vec<Vec<f64>>)> {
        let seqs: Vec<MarkerSeq> = items.iter().map(|it| it.seq.clone()).collect();
        let scores = self.bert().scores(t, &seqs)?;
        let mut terms = Vec::with_capacity(items.len());
        let mut probabilities = Vec::with_capacity(items.len());
        let mut at = 0;
        for it in items {
            let n = it.seq.markers.len();
            let row = t.slice(scores, 0, at, at + n)?;
            let row = t.scale(row, 1.0 / it.temperature)?;
            let row = t.reshape(row, &[1, n])?;
            let logp = log_softmax(t, row)?;
            let p = t.unary(Unary::Exp, logp);
            probabilities.push(t.value(p).iter().map(|&x| x as f64).collect());
            let mean = it.reward.iter().sum::<f64>() / n as f64;
            let advantage: Vec<f32> = it.reward.iter().map(|r| (mean - r) as f32).collect();
            let a = t.input(&advantage, &[1, n]);
            let weighted = t.mul(p, a)?;
            let mut term = t.sum(weighted);
            if self.cfg.distill > 0.0 {
                let target: Vec<f32> = it.reward.iter().map(|r| -*r as f32).collect();
                let target = t.input(&target, &[1, n]);
                let ce = t.mul(logp, target)?;
                let ce = t.sum_scaled(ce, self.cfg.distill);
                term = t.add(term, ce)?;
            }
            if let (Some(pa), true) = (&it.anchor, self.cfg.anchor > 0.0) {
                let log_anchor: Vec<f32> = pa.iter().map(|x| x.max(1e-9).ln() as f32).collect();
                let la = t.input(&log_anchor, &[1, n]);
                let diff = t.sub(logp, la)?;
                let kl = t.mul(p, diff)?;
                let kl = t.sum_scaled(kl, self.cfg.anchor);
                term = t.add(term, kl)?;
            }
            terms.push(term);
            at += n;
        }
        let total = t.concat(&terms, 0)?;
        let loss = t.sum_scaled(total, 1.0 / divisor as f32);
        Ok((loss, probabilities))
    }

    /// Add the gradients on `t` to the step's sums.
    fn accumulate(&mut self, t: &Tape<'_, B>) {
        let bert = self.bert();
        for (i, p) in bert.params().iter().enumerate() {
            if self.cfg.freeze.iter().any(|f| p.name.starts_with(f.as_str())) { continue; }
            let Some(g) = t.param_var(p).and_then(|v| t.grad(v)) else { continue };
            let acc = self.grads[i].get_or_insert_with(|| self.be.alloc(p.shape.iter().product()));
            self.be.axpby(g, acc, 1.0, 1.0);
        }
    }

    /// Apply the step's summed gradients, scaled by `scale`: clip the global norm, then
    /// AdamW on every tensor that received one. The sums are cleared.
    fn apply(&mut self, scale: f32) {
        let bert = self.bert();
        let grads: Vec<(&Param<B>, B::Buf)> = bert.params().iter().zip(self.grads.iter_mut())
            .filter_map(|(p, g)| g.take().map(|g| (p, g))).collect();
        if grads.is_empty() { return; }
        if scale != 1.0 { for (_, g) in &grads { self.be.scale(g, scale); } }
        let norm_sq = self.be.alloc(1);
        for (_, g) in &grads { self.be.sumsq(g, &norm_sq, true); }
        let norm = self.be.download(&norm_sq)[0].sqrt();
        if !norm.is_finite() {
            eprintln!("the step's gradients are not finite and were discarded");
            return;
        }
        if norm > self.cfg.clip {
            for (_, g) in &grads { self.be.scale(g, self.cfg.clip / norm); }
        }
        self.opt.begin();
        for (p, g) in &grads { self.opt.update(self.be, p, g, self.cfg.weight_decay); }
    }

    pub fn run(&mut self) -> Result<Report> {
        let mut report = Report::default();
        let held_out = self.held_out()?;
        eprintln!("held out: {} questions over {} families", held_out.len(), self.error.len());
        if self.cfg.eval_every > 0 {
            report.evals.push(self.evaluate(0, &held_out)?);
        }
        for step in 1..=self.cfg.steps {
            let start = Instant::now();
            let progress = (step - 1) as f32 / self.cfg.steps.max(1) as f32;
            self.opt.lr = self.cfg.lr_min + (self.cfg.lr - self.cfg.lr_min) * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            let mut items = Vec::new();
            for _ in 0..self.cfg.batch {
                let family = self.draw_family();
                let task = self.draw_task(&family);
                match self.items(&task, self.cfg.variants) {
                    Ok(built) => items.extend(built),
                    Err(e) => eprintln!("step {step}: a {} task was left out: {e:#}", task.family),
                }
            }
            if items.is_empty() { eprintln!("step {step}: nothing to train on"); continue; }
            self.anchor_distributions(&mut items)?;
            let (passes, mut skipped) = self.passes(&items, true);
            let trained: usize = passes.iter().map(|p| p.len()).sum();
            let (mut loss, mut agreed, mut seen) = (0.0, 0, 0);
            for pass in passes {
                if !self.may_start_pass("the step") { skipped += trained - seen; break; }
                let chunk = &items[pass];
                let mut t = Tape::new(self.be);
                let (l, probabilities) = self.loss(&mut t, chunk, trained)?;
                let value = t.value(l)[0];
                if !value.is_finite() {
                    eprintln!("step {step}: a pass of {} questions gave a {value} loss and was left out", chunk.len());
                    skipped += chunk.len();
                    seen += chunk.len();
                    continue;
                }
                t.backward(l)?;
                self.accumulate(&t);
                loss += value as f64;
                for (it, p) in chunk.iter().zip(&probabilities) {
                    let right = argmax(p) == argmax(&it.reward);
                    agreed += right as usize;
                    let e = self.error.entry(it.family.clone()).or_insert(1.0);
                    *e = 0.9 * *e + 0.1 * (!right) as u8 as f64;
                }
                seen += chunk.len();
            }
            // The passes' losses were scaled for every trained question; what the skipped
            // ones would have added is made up by scaling the sum back to a mean.
            let used = trained - skipped;
            self.apply(if used > 0 && used < trained { trained as f32 / used as f32 } else { 1.0 });
            let loss = if used > 0 { loss * trained as f64 / used as f64 } else { loss };
            let stats = StepStats { step, loss, agreement: if used > 0 { agreed as f64 / used as f64 } else { 0.0 },
                                    questions: used, skipped, seconds: start.elapsed().as_secs_f64() };
            eprintln!("step {step}: loss {:.4} agreement {:.0}% ({} questions{}, {:.1} s, {:.1} GB resident)", stats.loss, stats.agreement * 100.0,
                      stats.questions, if skipped > 0 { format!(", {skipped} skipped") } else { String::new() }, stats.seconds,
                      memory::resident_bytes() as f64 / 1e9);
            report.steps.push(stats);
            if self.cfg.eval_every > 0 && step % self.cfg.eval_every == 0 {
                report.evals.push(self.evaluate(step, &held_out)?);
            }
            if (self.cfg.checkpoint_every > 0 && step % self.cfg.checkpoint_every == 0) || step == self.cfg.steps {
                report.checkpoints.push(self.checkpoint(step, &held_out)?);
            }
        }
        Ok(report)
    }

    /// Held-out items of every family, from a seed of their own, so the same ones
    /// are scored at every evaluation.
    fn held_out(&mut self) -> Result<Vec<Item>> {
        let families: Vec<String> = self.error.keys().cloned().collect();
        let saved = self.rng.clone();
        self.rng = Rng::new(self.cfg.seed ^ 0x5EED);
        let mut items = Vec::new();
        for family in families {
            for _ in 0..self.cfg.eval_items {
                let task = self.draw_any(&family);
                self.held.insert(canonical(&task.body));
                items.extend(self.items(&task, false)?);
            }
        }
        self.rng = saved;
        Ok(items)
    }

    /// Score the held-out items without training on them.
    fn evaluate(&self, step: usize, items: &[Item]) -> Result<EvalStats> {
        let mut per: HashMap<String, (f64, f64, usize)> = HashMap::new();
        for pass in self.passes(items, false).0 {
            if !self.may_start_pass("the evaluation") { break; }
            let chunk = &items[pass];
            let mut t = Tape::new(self.be);
            let (_, probabilities) = self.loss(&mut t, chunk, chunk.len())?;
            for (it, p) in chunk.iter().zip(probabilities) {
                let e = per.entry(it.family.clone()).or_default();
                e.0 += (argmax(&p) == argmax(&it.reward)) as u8 as f64;
                e.1 -= it.reward.iter().zip(&p).map(|(r, q)| r * q.max(1e-12).ln()).sum::<f64>();
                e.2 += 1;
            }
        }
        let mut families: Vec<(String, f64, f64, usize)> = per.into_iter()
            .map(|(f, (a, ce, n))| (f, a / n as f64, ce / n as f64, n)).collect();
        families.sort_by(|a, b| a.0.cmp(&b.0));
        eprintln!("eval at step {step}:");
        for (f, a, ce, n) in &families {
            eprintln!("  {f:<14} agreement {:>5.1}%  cross-entropy {ce:.3}  ({n} questions)", a * 100.0);
        }
        Ok(EvalStats { step, families })
    }

    /// Save the model as a GGUF with temperatures refitted on the held-out items.
    fn checkpoint(&self, step: usize, held_out: &[Item]) -> Result<PathBuf> {
        let mut scored = Vec::with_capacity(held_out.len());
        for pass in self.passes(held_out, false).0 {
            if !self.may_start_pass("the calibration") { break; }
            let chunk = &held_out[pass];
            let seqs: Vec<MarkerSeq> = chunk.iter().map(|it| it.seq.clone()).collect();
            let mut t = Tape::new(self.be);
            let s = self.bert().scores(&mut t, &seqs)?;
            let values = t.value(s);
            let mut at = 0;
            for it in chunk {
                let n = it.seq.markers.len();
                scored.push(Scored { kind: it.kind, scores: values[at..at + n].to_vec(), target: it.reward.clone() });
                at += n;
            }
        }
        let temperatures = fit_temperatures(&scored);
        let path = self.cfg.out.join(format!("{}-step{step}.gguf", self.model.name()));
        export(self.be, self.bert(), &self.source, &path, &temperatures)?;
        eprintln!("saved {} (temperatures {})", path.display(),
                  temperatures.iter().map(|(k, t)| format!("{k}={t:.2}")).collect::<Vec<_>>().join(" "));
        Ok(path)
    }
}

/// `log softmax` of one row, finite for every entry: the largest score is taken out
/// before the exponentials, so a far option's log probability is a large negative
/// number rather than the log of an underflowed zero.
fn log_softmax<B: Backend>(t: &mut Tape<'_, B>, row: Var) -> Result<Var> {
    let n = t.shape(row)[1];
    let mx = t.value(row).iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted = t.add_scalar(row, -mx)?;
    let e = t.unary(Unary::Exp, shifted);
    let z = t.sum_axis(e, 1)?;
    let z = t.reshape(z, &[1, 1])?;
    let logz = t.unary(Unary::Log, z);
    let logz = t.concat(&vec![logz; n], 1)?;
    t.sub(shifted, logz)
}

/// A request's text with its keys in one order: the same for a request and its
/// rewrites, which keep its answers.
fn canonical(body: &Json) -> String { body.sorted().to_python(true) }

fn softmax(scores: &[f32], temperature: f32) -> Vec<f64> {
    let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = scores.iter().map(|&s| ((s - mx) / temperature) as f64).map(f64::exp).collect();
    let z: f64 = e.iter().sum();
    e.iter().map(|v| v / z).collect()
}
