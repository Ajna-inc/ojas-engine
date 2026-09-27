//! Correct every reviewed cell at one spot of one camera, after a closer look (`review_spots`)
//! showed the spot was misjudged: e.g. a "stall" that is really parked motorcycles. Cells whose
//! box overlaps the spot (IoU ≥ 0.5) get `verdict`, appended to `verdicts_zz_fix.tsv` — read last,
//! so it overrides the earlier lines. With `--auto-only`, only cells autofilled as static are
//! changed (for a spot that is open road, where by-eye verdicts are per frame and stand).
//!
//! `review_fix <review_dir> <camera prefix> <x> <y> <w> <h> <verdict> <note> [--auto-only]`
use std::io::Write;
use std::path::Path;

use serde_json::Value;

#[path = "review_common/mod.rs"]
mod review_common;
use review_common::{camera_of_entry, iou, is_miss_line};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 9, "review_fix <review_dir> <camera prefix> <x> <y> <w> <h> <verdict> <note> [--auto-only]");
    let dir = Path::new(&a[1]);
    let b: Vec<f32> = a[3..7].iter().map(|s| s.parse().unwrap()).collect();
    let spot = [b[0], b[1], b[2], b[3]];
    let auto_only = a.iter().any(|x| x == "--auto-only");
    let manifest: Vec<Value> = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    // which cells were autofilled: the note of the winning line
    let mut notes = std::collections::HashMap::new();
    let mut files: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("verdicts") && n.ends_with(".tsv"))).collect();
    files.sort();
    for f in files {
        for line in std::fs::read_to_string(&f)?.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty() && !is_miss_line(l)) {
            let p: Vec<&str> = line.split('\t').collect();
            notes.insert((p[0].parse::<u64>()?, p[1].parse::<u64>()?), p.get(3).unwrap_or(&"").to_string());
        }
    }
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("verdicts_zz_fix.tsv"))?;
    let mut n = 0;
    for m in &manifest {
        let (Some(s), Some(c)) = (m["sheet"].as_u64(), m["cell"].as_u64()) else { continue };
        if !camera_of_entry(m).starts_with(a[2].as_str()) {
            continue;
        }
        let mb: Vec<f32> = m["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        if iou(spot, [mb[0], mb[1], mb[2], mb[3]]) < 0.5 {
            continue;
        }
        if auto_only && !notes.get(&(s, c)).is_some_and(|nt| nt.starts_with("auto")) {
            continue;
        }
        writeln!(out, "{s}\t{c}\t{}\tfix: {}", a[7], a[8])?;
        n += 1;
    }
    println!("{n} cells at ({}, {}, {}, {}) on {} → {}", b[0], b[1], b[2], b[3], a[2], a[7]);
    Ok(())
}
