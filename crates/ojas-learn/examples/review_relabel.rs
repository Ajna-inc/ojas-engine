//! Bring a review's verdicts in line with the training set's own conventions, after the fact.
//! Reviewers named classes by the UVH-26 README, but UVH's labels (checked with `class_sheet`)
//! differ in places: cargo three-wheelers are LCV, Bolero/Scorpio/Sumo are MUV, pushcarts are
//! sometimes Others. A verdict that contradicts the pre-training labels teaches the model a
//! conflict, so such verdicts are rewritten by rule.
//!
//! `rules.tsv`: `from<TAB>to<TAB>any-of<TAB>none-of` — a cell whose final verdict is `from` and
//! whose note contains one of the `|`-separated `any-of` substrings (case-insensitive) and none of
//! `none-of` becomes `to`; the first matching rule wins. A cell filled by `review_autofill`'s
//! exact-repeat rule has no note of its own and takes the note of the cell it repeats.
//! Writes `verdicts_zzz_relabel.tsv` (read last, so it overrides; rewritten on every run and
//! ignored as input, so re-running is safe).
//!
//! `review_relabel <rules.tsv> <review_dir> [review_dir ...]`
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use serde_json::Value;

#[path = "review_common/mod.rs"]
mod review_common;
use review_common::{camera_of_entry, is_miss_line};

struct Rule {
    from: String,
    to: String,
    any: Vec<String>,
    none: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 3, "review_relabel <rules.tsv> <review_dir> [review_dir ...]");
    let split = |s: &str| s.split('|').map(|x| x.trim().to_lowercase()).filter(|x| !x.is_empty()).collect::<Vec<_>>();
    let rules: Vec<Rule> = std::fs::read_to_string(&a[1])?
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Rule { from: f[0].trim().into(), to: f[1].trim().into(), any: split(f.get(2).unwrap_or(&"")), none: split(f.get(3).unwrap_or(&"")) }
        })
        .collect();
    for d in &a[2..] {
        let dir = Path::new(d);
        let manifest: Vec<Value> = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
        // final (verdict, note) per cell, from every verdict file but this tool's own output
        let mut files: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("verdicts") && n.ends_with(".tsv") && n != "verdicts_zzz_relabel.tsv")).collect();
        files.sort();
        let mut fin: HashMap<(u64, u64), (String, String)> = HashMap::new();
        for f in files {
            for l in std::fs::read_to_string(&f)?.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty() && !is_miss_line(l)) {
                let p: Vec<&str> = l.split('\t').collect();
                fin.insert((p[0].trim().parse()?, p[1].trim().parse()?), (p[2].trim().to_string(), p.get(3).unwrap_or(&"").to_string()));
            }
        }
        let key = |m: &Value| {
            let b: Vec<i64> = m["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap().round() as i64).collect();
            (camera_of_entry(m), m["kind"].as_str().unwrap_or("").to_string(), m["a"].to_string(), m["b"].to_string(), b)
        };
        let cell = |m: &Value| Some((m["sheet"].as_u64()?, m["cell"].as_u64()?));
        // notes of by-eye cells, for exact-repeat copies to inherit
        let mut note_of = HashMap::new();
        for m in &manifest {
            if let Some(c) = cell(m) {
                if let Some((_, n)) = fin.get(&c) {
                    if !n.starts_with("auto:") {
                        note_of.entry(key(m)).or_insert_with(|| n.clone());
                    }
                }
            }
        }
        let mut out = std::fs::File::create(dir.join("verdicts_zzz_relabel.tsv"))?;
        let mut counts: HashMap<(String, String), usize> = HashMap::new();
        for m in &manifest {
            let Some(c) = cell(m) else { continue };
            let Some((v, n)) = fin.get(&c) else { continue };
            let note = if n.starts_with("auto: exact repeat") { note_of.get(&key(m)).cloned().unwrap_or_default() } else { n.clone() };
            let low = note.to_lowercase();
            if let Some(r) = rules.iter().find(|r| r.from == *v && r.any.iter().any(|s| low.contains(s.as_str())) && !r.none.iter().any(|s| low.contains(s.as_str()))) {
                writeln!(out, "{}\t{}\t{}\trelabel to UVH convention: {note}", c.0, c.1, r.to)?;
                *counts.entry((r.from.clone(), r.to.clone())).or_default() += 1;
            }
        }
        let mut cs: Vec<_> = counts.into_iter().collect();
        cs.sort();
        println!("{d}: {}", cs.iter().map(|((f, t), n)| format!("{f}→{t} {n}")).collect::<Vec<_>>().join(", "));
    }
    Ok(())
}
