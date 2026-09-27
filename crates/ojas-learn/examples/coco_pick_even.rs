//! Pick ~`n` frames of a labelled COCO set for a new labelling round, evenly in time per physical
//! camera, keeping every frame's id and `file_name` so the new round's labels join back onto the
//! set by image id.
//!
//! - Physical camera: the `camera` field before any `@segment`.
//! - Exact repeats are skipped first: a frame whose annotations are exactly (class, box to the
//!   pixel) those of an earlier frame of the same camera, or whose `source` was already seen — a
//!   frozen or re-sent segment (the rule of `review_export`).
//! - Then one global step `k = frames / n`; each camera with `m` frames (sorted by `utc_ms`) gets
//!   `max(1, round(m / k))` frames, the one nearest each of that many evenly spaced instants. The
//!   share of every camera — and so of day and night — is the set's own.
//! - Each picked image gets `path` (`ROOT` joined with `file_name`, checked to exist), `set` from
//!   `SETS=prefix=set,prefix=set` (the first `file_name` prefix that matches; `all` otherwise),
//!   and `luma`, the frame's mean luminance (0–255, on a 1/8 subsample), so dark frames can be
//!   identified independently of the set name.
//! - `EXCLUDE=a.json[,b.json]` (env): frames whose image id is in any of those files — those an
//!   earlier round already labelled — are left out before picking. With `n` at or above the number
//!   of frames left, every frame is picked (step 1).
//!
//! `coco_pick_even <in.json> <out.json> <n>` (env `ROOT`, `SETS`, `EXCLUDE`)
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use serde_json::{json, Value};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() == 4, "coco_pick_even <in.json> <out.json> <n>");
    let n: usize = a[3].parse()?;
    let root = std::env::var("ROOT").unwrap_or_default();
    let sets: Vec<(String, String)> = std::env::var("SETS").unwrap_or_default().split(',').filter_map(|p| p.split_once('=')).map(|(a, b)| (a.to_string(), b.to_string())).collect();
    let v: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let mut sig: HashMap<i64, Vec<(i64, [i64; 4])>> = HashMap::new();
    for an in v["annotations"].as_array().unwrap() {
        let b: Vec<i64> = an["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap().round() as i64).collect();
        sig.entry(an["image_id"].as_i64().unwrap()).or_default().push((an["category_id"].as_i64().unwrap(), [b[0], b[1], b[2], b[3]]));
    }
    let cam_of = |im: &Value| im["camera"].as_str().unwrap_or("?").split('@').next().unwrap().to_string();
    let mut exclude = HashSet::new();
    for f in std::env::var("EXCLUDE").unwrap_or_default().split(',').filter(|f| !f.is_empty()) {
        let e: Value = serde_json::from_slice(&std::fs::read(f)?)?;
        exclude.extend(e["images"].as_array().into_iter().flatten().filter_map(|im| im["id"].as_i64()));
    }
    let mut by_cam: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    let mut excluded = 0usize;
    for im in v["images"].as_array().unwrap() {
        if exclude.contains(&im["id"].as_i64().unwrap()) {
            excluded += 1;
            continue;
        }
        by_cam.entry(cam_of(im)).or_default().push(im);
    }
    let mut repeats = 0usize;
    let mut sources = HashSet::new();
    for ims in by_cam.values_mut() {
        ims.sort_by_key(|im| (im["utc_ms"].as_i64().unwrap_or(0), im["id"].as_i64().unwrap()));
        let mut seen = HashSet::new();
        ims.retain(|im| {
            let mut s = sig.get(&im["id"].as_i64().unwrap()).cloned().unwrap_or_default();
            s.sort();
            let src = im["source"].as_str().unwrap_or(im["file_name"].as_str().unwrap()).to_string();
            let fresh = sources.insert(src) && (s.is_empty() || seen.insert(s));
            repeats += !fresh as usize;
            fresh
        });
    }
    let total: usize = by_cam.values().map(Vec::len).sum();
    if excluded > 0 {
        println!("{excluded} frames excluded (EXCLUDE)");
    }
    let k = (total as f64 / n as f64).max(1.0);
    let mut picked: Vec<Value> = vec![];
    let mut per: BTreeMap<(String, String), (usize, usize)> = BTreeMap::new();
    for (cam, ims) in &by_cam {
        let want = ((ims.len() as f64 / k).round() as usize).clamp(1, ims.len());
        let step = ims.len() as f64 / want as f64;
        for i in 0..want {
            let im = ims[((i as f64 + 0.5) * step) as usize];
            let file = im["file_name"].as_str().unwrap();
            let path = if Path::new(file).is_absolute() { file.to_string() } else { Path::new(&root).join(file).to_string_lossy().into_owned() };
            let img = image::open(&path).map_err(|e| anyhow::anyhow!("{path}: {e}"))?.to_rgb8();
            let (mut sum, mut cnt) = (0f64, 0usize);
            for y in (0..img.height()).step_by(8) {
                for x in (0..img.width()).step_by(8) {
                    let p = img.get_pixel(x, y).0;
                    sum += 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64;
                    cnt += 1;
                }
            }
            let set = sets.iter().find(|(p, _)| file.starts_with(p.as_str())).map(|(_, s)| s.clone()).unwrap_or_else(|| "all".into());
            let e = per.entry((set.clone(), cam.clone())).or_default();
            e.0 += 1;
            e.1 = ims.len();
            let mut im = im.clone();
            im["path"] = json!(path);
            im["set"] = json!(set);
            im["luma"] = json!((sum / cnt.max(1) as f64 * 10.0).round() / 10.0);
            picked.push(im);
        }
    }
    picked.sort_by_key(|im| im["id"].as_i64().unwrap());
    let mut set_tot: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for ((set, cam), (p, of)) in &per {
        println!("{set:<8} {cam:<24} {p:>5} of {of:>5}");
        let e = set_tot.entry(set).or_default();
        e.0 += p;
        e.1 += of;
    }
    for (s, (p, of)) in &set_tot {
        println!("{s:<8} {:<24} {p:>5} of {of:>5} ({:.1} % of the pick, {:.1} % of the set)", "all cameras", 100.0 * *p as f64 / picked.len() as f64, 100.0 * *of as f64 / total as f64);
    }
    let dark = picked.iter().filter(|im| im["luma"].as_f64().unwrap_or(255.0) < 60.0).count();
    let out = json!({"info": {"description": "frames picked evenly in time per camera for a labelling round; ids and file_name as in the source set", "picked_from": a[1], "image_root": root, "step": k, "exact_repeats_skipped": repeats}, "images": picked, "categories": [], "annotations": []});
    std::fs::write(&a[2], serde_json::to_string(&out)?)?;
    println!("{} frames (step {k:.2} over {total} after {repeats} exact repeats; {dark} with mean luma < 60) → {}", picked.len(), a[2]);
    Ok(())
}
