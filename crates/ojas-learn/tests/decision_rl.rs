//! The reward-driven trainer end to end on the test models: a saved GGUF answers as
//! the model it came from; teachers are any `/v1/systemone` server, swapped by
//! address and combined by weight; a few steps move the student towards its teacher.
//!
//! ```text
//! OJAS_DECISION_MODELS=~/models/decision cargo test --release -p ojas-learn \
//!     --test decision_rl -- --ignored --nocapture
//! ```
//!
//! The teacher tests start `target/release/ojas serve` on the small causal models.

use ojas_decision::json::Json;
use ojas_decision::{testkit, DecisionModel};
use ojas_formats::gguf::Gguf;
use ojas_learn::cpu::Cpu;
use ojas_learn::decision_rl::export::export;
use ojas_learn::decision_rl::teacher::{Cache, Committee, HttpTeacher, Teacher};
use ojas_learn::decision_rl::train::{Config, Trainer};
use ojas_learn::decision_rl::{LearnDecision, ModernBert};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ojas-decision-rl-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn a_saved_model_answers_as_the_one_it_came_from() {
    let src = testkit::models_dir().join("tinylaya-Q8_0.gguf");
    let be = Cpu;
    let mut g = Gguf::open(src.to_str().unwrap()).unwrap();
    let model = ModernBert::from_gguf(&be, &mut g).unwrap();
    let dir = scratch("export");
    let dst = dir.join("tinylaya-Q8_0.gguf");
    export(&be, &model, &src, &dst, &[("choice".into(), 1.5), ("noul".into(), 1.9)]).unwrap();
    let saved = Gguf::open(dst.to_str().unwrap()).unwrap();
    assert_eq!(saved.tensors.len(), g.tensors.len());
    assert_eq!(saved.meta_f32("modern-bert.decision.temperature.choice"), Some(1.5));
    assert_eq!(saved.meta.len(), g.meta.len(), "every metadata key survives");
    // The file loads through the decision pipeline and matches the reference answers
    // as the original does; the temperatures above were the original's own.
    testkit::decisions_match_the_reference_responses(&LearnDecision(&be), &dir);
}

/// `ojas serve` on a model, on a free port, stopped when dropped.
struct Server {
    child: Child,
    url: String,
}

impl Server {
    fn start(file: &str) -> Option<Server> {
        let model = testkit::models_dir().join(file);
        let binary = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/ojas");
        if !model.is_file() || !binary.is_file() {
            eprintln!("skipped: {} or {} is missing", model.display(), binary.display());
            return None;
        }
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let mut child = Command::new(binary).arg("serve").arg(&model).arg("--port").arg(port.to_string())
            .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("starting ojas serve");
        let url = format!("http://127.0.0.1:{port}");
        let deadline = Instant::now() + Duration::from_secs(120);
        while Instant::now() < deadline {
            if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
                std::thread::sleep(Duration::from_millis(300));
                return Some(Server { child, url });
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        let _ = child.kill();
        let _ = child.wait();
        panic!("{file} did not start serving");
    }
}

impl Drop for Server {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}

fn ticket() -> Json { testkit::read_json(&testkit::fixtures().join("requests/ticket.json")) }

#[test]
#[ignore = "needs the decision model files and target/release/ojas; set OJAS_DECISION_MODELS"]
fn teachers_are_swapped_by_address_and_combined_by_weight() {
    let (Some(kev), Some(lev)) = (Server::start("tinykev-Q8_0.gguf"), Server::start("tinylev-Q8_0.gguf")) else { return };
    let a = HttpTeacher::new("kev", &kev.url).unwrap();
    let b = HttpTeacher::new("lev", &lev.url).unwrap();
    let body = ticket();
    let (ja, jb) = (a.judge(&body).unwrap(), b.judge(&body).unwrap());
    assert_eq!(ja.len(), 3);
    for q in ja.iter().chain(&jb) {
        let total: f64 = q.values().sum();
        assert!((total - 1.0).abs() < 1e-6, "a judgement is a distribution: {q:?}");
    }
    assert!(ja.iter().zip(&jb).any(|(x, y)| x.iter().any(|(k, p)| (p - y[k]).abs() > 1e-3)), "two models judge differently");

    let committee = Committee::new(vec![(Box::new(HttpTeacher::new("kev", &kev.url).unwrap()), 1.0),
                                        (Box::new(HttpTeacher::new("lev", &lev.url).unwrap()), 3.0)]).unwrap();
    assert_eq!(committee.name(), "kev:1+lev:3");
    let jc = committee.judge(&body).unwrap();
    for ((x, y), c) in ja.iter().zip(&jb).zip(&jc) {
        for (k, p) in c { assert!((p - (x[k] + 3.0 * y[k]) / 4.0).abs() < 1e-9, "{k}: {p} is not the weighted mean"); }
    }

    let dir = scratch("cache");
    let cache = Cache::open(Box::new(committee), &dir.join("cache.jsonl")).unwrap();
    assert_eq!(cache.judge(&body).unwrap(), jc);
    assert_eq!(cache.len(), 1);
    drop(kev);
    drop(lev);
    let reopened = Cache::open(Box::new(HttpTeacher::new("kev:1+lev:3", "http://127.0.0.1:1").unwrap()), &dir.join("cache.jsonl")).unwrap();
    assert_eq!(reopened.judge(&body).unwrap(), jc, "a cached judgement needs no server");
}

#[test]
#[ignore = "needs the decision model files and target/release/ojas; set OJAS_DECISION_MODELS"]
fn a_few_steps_move_the_student_towards_its_teacher() {
    let Some(kev) = Server::start("tinykev-Q8_0.gguf") else { return };
    let src = testkit::models_dir().join("tinylaya-Q8_0.gguf");
    let be = Cpu;
    let gpu = LearnDecision(&be);
    let model = DecisionModel::load(&gpu, src.to_str().unwrap()).unwrap();
    let dir = scratch("train");
    let teacher = Cache::open(Box::new(HttpTeacher::new("kev", &kev.url).unwrap()), &dir.join("cache.jsonl")).unwrap();
    let cfg = Config {
        steps: 6, batch: 1, lr: 3e-4, weight_decay: 0.0, anchor: 0.0, distill: 0.0, lr_min: 3e-4, judge_threads: 2, clip: 1.0, variants: false,
        teacher_weight: 1.0, gold_weight: 1.0, judge_known: true, families: vec!["compare".into(), "negation".into()], files: Vec::new(),
        eval_every: 6, eval_items: 3, checkpoint_every: 0, seed: 5, out: dir.clone(), freeze: vec!["token_embd".into()],
        max_seq_tokens: 1024, memory_gb: 0.0,
    };
    let mut trainer = Trainer::new(&be, &model, None, &teacher, &cfg, &src).unwrap();
    let report = trainer.run().unwrap();
    assert_eq!(report.steps.len(), 6);
    assert_eq!(report.evals.len(), 2);
    let (first, last) = (report.steps.first().unwrap().loss, report.steps.last().unwrap().loss);
    eprintln!("loss {first:.4} -> {last:.4}");
    let before: f64 = report.evals[0].families.iter().map(|f| f.2).sum();
    let after: f64 = report.evals[1].families.iter().map(|f| f.2).sum();
    eprintln!("held-out cross-entropy {before:.4} -> {after:.4}");
    assert!(after < before, "the held-out cross-entropy to the reward did not fall");
    assert_eq!(report.checkpoints.len(), 1, "the final step saves the model");
    let saved = report.checkpoints[0].clone();
    let served = DecisionModel::load(&gpu, saved.to_str().unwrap()).unwrap();
    let req = served.request(&ticket()).unwrap();
    assert_eq!(served.decide(&req).unwrap().answers.len(), 3, "the saved model serves requests");
}
