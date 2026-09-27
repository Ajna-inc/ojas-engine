//! Turn `sample_dataset` corpora (one or more `frames.jsonl`) into a review round and a held-out
//! test set, per camera, from a plan the reviewer writes after looking at `camera_sheet`:
//!
//! `plan.tsv`: `camera-folder<TAB>name<TAB>cap` — cap 0 drops the camera (corrupt, black, no
//! traffic); several folders may share a name (one physical camera recorded twice) and then
//! share the cap. Folders not in the plan are dropped.
//!
//! Per named camera: the last 15 % of its time span is the test block, of which up to `test_n`
//! frames spread evenly are the test set; the training pool is the rest minus anything within
//! `gap_s` of a test frame, thinned evenly in time to `cap` frames. Writes COCO image lists
//! (`<out>/round_frames.json`, `<out>/test_frames.json`), image ids from `id_base`.
//!
//! `corpus_rounds <plan.tsv> <out_dir> <id_base> <test_n> <gap_s> <frames.jsonl> [frames.jsonl ...]`
use std::collections::{BTreeMap, HashMap};

use serde_json::{json, Value};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 7, "corpus_rounds <plan.tsv> <out_dir> <id_base> <test_n> <gap_s> <frames.jsonl> [frames.jsonl ...]");
    let (id_base, test_n): (i64, usize) = (a[3].parse()?, a[4].parse()?);
    let gap_ms = a[5].parse::<i64>()? * 1000;
    let mut plan: HashMap<String, (String, usize)> = HashMap::new();
    for l in std::fs::read_to_string(&a[1])?.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = l.split('\t').collect();
        plan.insert(f[0].to_string(), (f[1].to_string(), f[2].trim().parse()?));
    }
    // frames by named camera: (utc_ms, path, w, h)
    let mut cams: BTreeMap<String, Vec<(i64, String, i64, i64)>> = BTreeMap::new();
    let mut caps: HashMap<String, usize> = HashMap::new();
    for p in &a[6..] {
        for l in std::fs::read_to_string(p)?.lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(l)?;
            let folder = v["camera"].as_str().unwrap_or("");
            let Some((name, cap)) = plan.get(folder) else { continue };
            if *cap == 0 {
                continue;
            }
            caps.insert(name.clone(), *cap);
            cams.entry(name.clone()).or_default().push((v["utc_ms"].as_i64().unwrap_or(0), v["path"].as_str().unwrap().to_string(), v["width"].as_i64().unwrap_or(0), v["height"].as_i64().unwrap_or(0)));
        }
    }
    let even = |v: &[(i64, String, i64, i64)], n: usize| -> Vec<(i64, String, i64, i64)> {
        if v.len() <= n {
            return v.to_vec();
        }
        (0..n).map(|i| v[((i as f64 + 0.5) * v.len() as f64 / n as f64) as usize].clone()).collect()
    };
    let (mut round, mut test) = (vec![], vec![]);
    let mut id = id_base;
    println!("{:<28} {:>7} {:>6} {:>6}", "camera", "frames", "train", "test");
    for (name, mut fr) in cams {
        fr.sort();
        let (t0, t1) = (fr[0].0, fr[fr.len() - 1].0);
        let cut = t1 - (t1 - t0) * 15 / 100;
        let block: Vec<_> = fr.iter().filter(|f| f.0 >= cut).cloned().collect();
        let tset = if fr.len() >= 20 { even(&block, test_n) } else { vec![] };
        let pool: Vec<_> = fr.iter().filter(|f| !tset.iter().any(|t| (t.0 - f.0).abs() <= gap_ms)).cloned().collect();
        let train = even(&pool, caps[&name]);
        println!("{name:<28} {:>7} {:>6} {:>6}", fr.len(), train.len(), tset.len());
        for (dst, set) in [(&mut round, train), (&mut test, tset)] {
            for (t, path, w, h) in set {
                dst.push(json!({"id": id, "file_name": path, "width": w, "height": h, "camera": name, "utc_ms": t}));
                id += 1;
            }
        }
    }
    let out = std::path::Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let cats: Vec<Value> = ["Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"].iter().enumerate().map(|(i, n)| json!({"id": i + 1, "name": n})).collect();
    for (file, set) in [("round_frames.json", &round), ("test_frames.json", &test)] {
        std::fs::write(out.join(file), serde_json::to_string(&json!({"images": set, "categories": cats, "annotations": []}))?)?;
    }
    println!("{} training-pool frames, {} test frames → {}", round.len(), test.len(), out.display());
    Ok(())
}
