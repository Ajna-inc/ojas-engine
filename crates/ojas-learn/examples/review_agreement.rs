//! Agreement between two detectors on a frame list, before any review. Per set (the frames' `set`
//! field, e.g. day / night / day_new) and per camera: the boxes per frame each model finds at
//! score ≥ `score`, the boxes matched between them (greedy one-to-one at IoU ≥ 0.5, the rule of
//! `review_sheets`), and the share of each model's boxes the other confirms. `only A + only B` is
//! what the dispute sheets will hold, so these numbers size a review round.
//!
//! De-duplication and matching are `review_common`'s, so the counts are the sheets' counts.
//!
//! `review_agreement <frames.json> <A.dets.json> <B.dets.json> [score 0.4]`
use std::collections::BTreeMap;

use serde_json::Value;

#[path = "review_common/mod.rs"]
mod review_common;
use review_common::{load_dets, pair};

#[derive(Default)]
struct Row {
    frames: usize,
    a: usize,
    b: usize,
    agreed: usize,
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 4, "review_agreement <frames.json> <A.dets.json> <B.dets.json> [score 0.4]");
    let min_score: f32 = a.get(4).map(|v| v.parse().unwrap()).unwrap_or(0.4);
    let frames: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let (da, db) = (load_dets(&a[2], min_score)?, load_dets(&a[3], min_score)?);
    let mut by_set: BTreeMap<String, Row> = BTreeMap::new();
    let mut by_cam: BTreeMap<String, Row> = BTreeMap::new();
    let mut all = Row::default();
    for im in frames["images"].as_array().unwrap() {
        let id = im["id"].as_i64().unwrap();
        let empty = vec![];
        let (la, lb) = (da.get(&id).unwrap_or(&empty), db.get(&id).unwrap_or(&empty));
        let (of_a, _) = pair(la, lb);
        let agreed = of_a.iter().filter(|m| m.is_some()).count();
        let set = im["set"].as_str().unwrap_or("all").to_string();
        let cam = im["camera"].as_str().unwrap_or("?").split('@').next().unwrap().to_string();
        for r in [by_set.entry(set).or_default(), by_cam.entry(cam).or_default(), &mut all] {
            r.frames += 1;
            r.a += la.len();
            r.b += lb.len();
            r.agreed += agreed;
        }
    }
    let (na, nb) = (a[2].rsplit('/').next().unwrap(), a[3].rsplit('/').next().unwrap());
    println!("A = {na}\nB = {nb}\nscore ≥ {min_score}, matched at IoU ≥ 0.5\n");
    let head = format!("{:<24}{:>7}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}", "", "frames", "A/frame", "B/frame", "agreed", "only A", "only B", "A conf.", "B conf.");
    let line = |name: &str, r: &Row| {
        let f = r.frames.max(1) as f64;
        println!("{name:<24}{:>7}{:>9.2}{:>9.2}{:>9}{:>9}{:>9}{:>8.1}%{:>8.1}%", r.frames, r.a as f64 / f, r.b as f64 / f, r.agreed, r.a - r.agreed, r.b - r.agreed, 100.0 * r.agreed as f64 / r.a.max(1) as f64, 100.0 * r.agreed as f64 / r.b.max(1) as f64);
    };
    println!("{head}");
    for (s, r) in &by_set {
        line(s, r);
    }
    line("all", &all);
    println!("\n{head}");
    for (c, r) in &by_cam {
        line(c, r);
    }
    println!("\n(A conf. = share of A's boxes that B also found; only A + only B = the dispute cells at this score)");
    Ok(())
}
