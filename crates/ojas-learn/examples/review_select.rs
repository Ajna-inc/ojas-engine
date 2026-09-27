//! Pick frames for a review round: N frames spread evenly in time over each camera (in proportion
//! to the camera's frame count), skipping frames already in earlier rounds. The field footage is
//! sampled at ~0.56 s, so neighbouring frames show the same vehicles; an even spread over
//! thousands of frames sees each vehicle once and every hour of light.
//!
//! With `TEST=test.json` (and `GAP` seconds, default 15), frames within `GAP` of a test frame of
//! the same physical camera are skipped too, since `review_export` would drop them anyway.
//!
//! `review_select <frames_coco.json> <out.json> <n> [exclude.json ...]`
use std::collections::{BTreeMap, HashSet};

use serde_json::{json, Value};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 4, "review_select <frames_coco.json> <out.json> <n> [exclude.json ...]");
    let n: usize = a[3].parse()?;
    let coco: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let mut exclude = HashSet::new();
    for p in &a[4..] {
        let v: Value = serde_json::from_slice(&std::fs::read(p)?)?;
        exclude.extend(v["images"].as_array().unwrap().iter().map(|im| im["id"].as_i64().unwrap()));
    }
    let cam_of = |im: &Value| im["camera"].as_str().unwrap_or("?").split('@').next().unwrap().to_string();
    let mut test: Vec<(String, i64)> = vec![];
    if let Ok(p) = std::env::var("TEST") {
        let v: Value = serde_json::from_slice(&std::fs::read(p)?)?;
        test = v["images"].as_array().unwrap().iter().map(|im| (cam_of(im), im["utc_ms"].as_i64().unwrap_or(0))).collect();
    }
    let gap_ms = std::env::var("GAP").ok().and_then(|g| g.parse::<i64>().ok()).unwrap_or(15) * 1000;
    let mut by_cam: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for im in coco["images"].as_array().unwrap() {
        let (c, t) = (cam_of(im), im["utc_ms"].as_i64().unwrap_or(0));
        if !exclude.contains(&im["id"].as_i64().unwrap()) && !test.iter().any(|(tc, tt)| *tc == c && (tt - t).abs() <= gap_ms) {
            by_cam.entry(im["camera"].as_str().unwrap_or("?").to_string()).or_default().push(im);
        }
    }
    let total: usize = by_cam.values().map(Vec::len).sum();
    let mut picked: Vec<Value> = vec![];
    for (cam, mut ims) in by_cam {
        ims.sort_by_key(|im| im["utc_ms"].as_i64().unwrap_or(0));
        let want = ((n * ims.len()) as f64 / total as f64).round().max(1.0) as usize;
        let want = want.min(ims.len());
        // the frame nearest each of `want` evenly spaced instants
        let step = ims.len() as f64 / want as f64;
        let chosen: Vec<&Value> = (0..want).map(|i| ims[((i as f64 + 0.5) * step) as usize]).collect();
        println!("{cam:<20} {want:>5} of {:>5}", ims.len());
        picked.extend(chosen.into_iter().cloned());
    }
    picked.sort_by_key(|im| im["id"].as_i64().unwrap());
    let out = json!({"images": picked, "categories": coco["categories"], "annotations": [], "info": {"selected_from": a[1], "excluding": a[4..].to_vec(), "n": picked.len()}});
    std::fs::write(&a[2], serde_json::to_string(&out)?)?;
    println!("{} frames → {}", picked.len(), a[2]);
    Ok(())
}
