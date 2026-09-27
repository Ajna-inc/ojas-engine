//! Resolve the static false positives of a review round from the verdicts given so far. A cell
//! still without a verdict (in `verdicts.tsv` or any `verdicts_*.tsv` part) whose box overlaps
//! (IoU ≥ 0.7) at least `min` reviewed boxes of the same camera, all of them `not_vehicle` and
//! none a vehicle, is appended to `verdicts.tsv` as `not_vehicle` with the note `auto: static`, so
//! a kiosk or a potted tree is only reviewed by eye a few times.
//!
//! Second rule: a cell that repeats an already-judged cell of the same camera exactly — same kind,
//! same classes, same box to the pixel — is a frozen or re-sent frame (moving traffic never repeats
//! to the pixel) and takes that cell's verdict with the note `auto: exact repeat`.
//!
//! `TASK=person` (see `review_common`): the static verdict written is `not_person`.
//!
//! `review_autofill <review_dir> [min 3] [prior_dir ...]`
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use serde_json::Value;

#[path = "review_common/mod.rs"]
mod review_common;
use review_common::{camera_of_entry, iou, load_priors, read_verdicts, task};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 2, "review_autofill <review_dir> [min 3] [prior_dir ...]");
    let task = task()?;
    let dir = Path::new(&a[1]);
    let min: usize = a.get(2).map(|v| v.parse().unwrap()).unwrap_or(3);
    let dirs: Vec<&str> = std::iter::once(a[1].as_str()).chain(a.iter().skip(3).map(String::as_str)).collect();
    let priors = load_priors(&dirs.join(":"))?;
    let manifest: Vec<Value> = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    let done: HashSet<(u64, u64)> = read_verdicts(dir)?.into_keys().collect();
    // judged cells by (camera, kind, a, b, box to the pixel)
    let verdicts = read_verdicts(dir)?;
    let key = |m: &Value| {
        let b: Vec<i64> = m["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap().round() as i64).collect();
        (camera_of_entry(m), m["kind"].as_str().unwrap_or("").to_string(), m["a"].to_string(), m["b"].to_string(), b)
    };
    let mut judged = std::collections::HashMap::new();
    for m in &manifest {
        if let (Some(s), Some(c)) = (m["sheet"].as_u64(), m["cell"].as_u64()) {
            if let Some(v) = verdicts.get(&(s, c)) {
                judged.entry(key(m)).or_insert_with(|| v.clone());
            }
        }
    }
    let mut repeats = 0;
    let empty = vec![];
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("verdicts.tsv"))?;
    let (mut open, mut filled) = (0, 0);
    for m in &manifest {
        let (Some(s), Some(c)) = (m["sheet"].as_u64(), m["cell"].as_u64()) else { continue };
        if done.contains(&(s, c)) {
            continue;
        }
        open += 1;
        if let Some(v) = judged.get(&key(m)) {
            writeln!(out, "{s}\t{c}\t{v}\tauto: exact repeat")?;
            repeats += 1;
            continue;
        }
        let b: Vec<f32> = m["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let same: Vec<bool> = priors.get(&camera_of_entry(m)).unwrap_or(&empty).iter().filter(|(pb, _)| iou(*pb, [b[0], b[1], b[2], b[3]]) >= 0.7).map(|(_, nv)| *nv).collect();
        if same.len() >= min && same.iter().all(|&nv| nv) {
            writeln!(out, "{s}\t{c}\t{}\tauto: static, {} reviews", task.negative, same.len())?;
            filled += 1;
        }
    }
    println!("{open} cells without a verdict; {repeats} exact repeats of judged cells, {filled} resolved as static false positives → {}", dir.join("verdicts.tsv").display());
    Ok(())
}
