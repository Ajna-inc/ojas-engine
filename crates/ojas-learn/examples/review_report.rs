//! Score two detectors on the boxes they disagree on from a by-eye review of `review_sheets`
//! output, and turn the review into a provisional ground truth.
//!
//! `review_dir/verdicts.tsv` (and any `verdicts_*.tsv` parts) holds one line per sheet cell,
//! `sheet cell verdict note`. A verdict is a class name (the object is real and this is its class),
//! `not_vehicle`, `duplicate` (a second box on an already-boxed object), `unsure_car` (a car,
//! subtype unclear), `unsure_vehicle` (a vehicle, class unclear) or `unsure`. The matching is
//! redone exactly as `review_sheets` did it and checked against `manifest.json`, so every verdict
//! lands on the box it was given for. A manifest entry with `"auto"` (a static false positive
//! resolved from earlier reviews) needs no line.
//!
//! Prints, per model: false positives, duplicates, real vehicles it missed that the other found,
//! and who named the class right — overall, by camera and by class. Writes
//! `review_dir/reviewed_gold.json`, a COCO set of the agreed boxes (same box and class from both
//! models, at the mean box) plus every box the review gave a class, with the unsure boxes as crowd
//! (ignore) regions in each class they might be. Objects both models missed are not in it.
//!
//! Agreed boxes are not reviewed and two models can share a static false positive, so an agreed box
//! overlapping (IoU ≥ 0.7) at least 3 reviewed boxes of its camera that were all `not_vehicle` —
//! this review's and the `PRIORS=dir1:dir2` ones — is dropped.
//!
//! `TASK=person` (see `review_common`): verdicts are `person`, `rider`, `not_person`, `duplicate`
//! or `unsure`; `person` and `rider` both land as category 15 `person`, `unsure` as a crowd region
//! of that category. With `SWEEP=<sweep_dir>` (a `review_sheets --sweep` output and its verdict
//! files) the sweep's `miss` lines — `miss <frame id> <x> <y> <w> <h> [label] [note]`, whole-frame
//! pixels — become boxes as well (`person` / `rider` real, `unsure` crowd), except one overlapping
//! (IoU ≥ 0.5) a box the frame already has, which is reported and skipped; `<sheet> <cell> swept`
//! lines count the frames swept, and sweep-manifest frames without one are listed. `miss` lines in
//! `review_dir` itself are read the same way.
//!
//! Person boxes need a suppression vehicles do not: both COCO DETRs put several queries on one
//! person at 0.3–0.4 and the per-model de-duplication only collapses them at IoU ≥ 0.7, so the
//! extra query is reviewed as a second `person`. Each frame's assembled person boxes are therefore
//! suppressed again: drop a box with IoU ≥ 0.5 against a kept one, or where the smaller of the pair
//! is ≥ 80 % inside the larger; keep an agreed box over a reviewed one, then the higher score, then
//! the larger area. Crowd (unsure) regions take no part. Drops print per set, with per-model false
//! positives (`not_person`) and misses.
//!
//! `review_report <frames.json> <A.dets.json> <B.dets.json> <review_dir> [score 0.3]`
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde_json::{json, Value};

#[path = "review_common/mod.rs"]
mod review_common;
use review_common::{camera_of, dir_dedupe_contain, iou, load_dets_with, load_priors, pair, read_misses, read_verdicts, task, Miss};

#[derive(Default, Clone, Copy)]
struct Side {
    fp: usize,
    dup: usize,
    found: usize,
    found_class_right: usize,
    missed: usize,
    dispute_right: usize,
}

/// A COCO bbox value as floats.
fn key_f(b: &Value) -> [f32; 4] {
    let v: Vec<f32> = b.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    [v[0], v[1], v[2], v[3]]
}

/// A box rounded to whole pixels, to find a manifest entry by box.
fn key(b: &Value) -> [i32; 4] {
    let v: Vec<i32> = b.as_array().unwrap().iter().map(|x| x.as_f64().unwrap().round() as i32).collect();
    [v[0], v[1], v[2], v[3]]
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 5, "review_report <frames.json> <A.dets.json> <B.dets.json> <review_dir> [score]");
    let task = task()?;
    let names = task.names;
    let min_score: f32 = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(0.3);
    let dir = Path::new(&a[4]);
    let frames: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    // the per-model de-duplication the sheets were cut with (`params.json`, see `review_sheets`)
    let contain = dir_dedupe_contain(dir)?;
    let (da, db) = (load_dets_with(&a[2], min_score, contain)?, load_dets_with(&a[3], min_score, contain)?);
    let manifest: Vec<Value> = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    let verdicts = read_verdicts(dir)?;
    let camera: HashMap<i64, String> = frames["images"].as_array().unwrap().iter().map(|im| (im["id"].as_i64().unwrap(), im["camera"].as_str().unwrap_or("?").to_string())).collect();

    let set_of: HashMap<i64, String> = frames["images"].as_array().unwrap().iter().map(|im| (im["id"].as_i64().unwrap(), im["set"].as_str().unwrap_or("all").to_string())).collect();

    let mut anns: Vec<Value> = vec![];
    // per annotation: (agreed by both models, score, origin). The first two give the person
    // suppression its precedence; origin 1 / 2 = a reviewed box only A / only B drew (net misses).
    let mut meta: Vec<(bool, f32, u8)> = vec![];
    let add = |anns: &mut Vec<Value>, meta: &mut Vec<(bool, f32, u8)>, id: i64, b: [f32; 4], c: usize, crowd: bool, agreed: bool, score: f32, origin: u8| {
        let n = anns.len() + 1;
        anns.push(json!({"id": n, "image_id": id, "category_id": task.cat_id(c), "bbox": b, "area": b[2] * b[3], "iscrowd": crowd as u8}));
        meta.push((agreed, score, origin));
    };
    let mut riders = 0usize;
    let (mut sa, mut sb) = (Side::default(), Side::default());
    let (mut both_wrong, mut both_fp, mut unresolved, mut agreed) = (0, 0, 0, 0);
    let mut by_cam: BTreeMap<String, [usize; 4]> = BTreeMap::new(); // fp A, fp B, missed A, missed B
    let mut by_set: BTreeMap<String, [usize; 7]> = BTreeMap::new(); // fp A, fp B, missed A, missed B, suppressed duplicates, net missed A, net missed B
    let mut by_class: BTreeMap<usize, [usize; 4]> = BTreeMap::new(); // missed A, missed B, dispute A right, dispute B right
    let mut fp_as: [BTreeMap<&str, usize>; 2] = Default::default();
    // manifest entries by (frame, box): the cells first, the auto-resolved statics after
    let by_box: HashMap<(i64, [i32; 4]), &Value> = manifest.iter().map(|m| ((m["image_id"].as_i64().unwrap(), key(&m["bbox"])), m)).collect();
    let prior_dirs = std::iter::once(a[4].clone()).chain(std::env::var("PRIORS").ok()).collect::<Vec<_>>().join(":");
    // STATICS=0: no static-spot dropping — for a set whose frames are unrelated scenes (UVH-26),
    // where every frame shares one "camera" and a spot in one frame means nothing in another
    let statics = if std::env::var("STATICS").as_deref() == Ok("0") { Default::default() } else { load_priors(&prior_dirs)? };
    // KEEP_A_BOX=1: an agreed box keeps A's geometry instead of the mean of A and B — when A is a
    // labelled set being fixed, its boxes are the reference and must not drift toward a model's
    let keep_a_box = std::env::var("KEEP_A_BOX").as_deref() == Ok("1");
    let no_statics = vec![];
    let mut dropped_static = 0usize;
    let mut k = 0usize;
    let mut auto_n = 0usize;
    for im in frames["images"].as_array().unwrap() {
        let id = im["id"].as_i64().unwrap();
        let cam = camera[&id].clone();
        let empty = vec![];
        let (la, lb) = (da.get(&id).unwrap_or(&empty), db.get(&id).unwrap_or(&empty));
        // a person round over exported frames keys statics by the original frame (`source`)
        let cam_key = camera_of(if task.name == "person" { im["source"].as_str() } else { None }.unwrap_or(im["file_name"].as_str().unwrap()));
        let (of_a, ob) = pair(la, lb);
        // disputed boxes in manifest order: each A box in turn, then the unmatched B boxes
        let mut items: Vec<(&str, [f32; 4], Option<usize>, Option<usize>, f32)> = vec![];
        for (x, m) in la.iter().zip(of_a) {
            match m {
                Some(j) if x.class == lb[j].class => {
                    let y = lb[j].bbox;
                    let same: Vec<bool> = statics.get(&cam_key).unwrap_or(&no_statics).iter().filter(|(pb, _)| iou(*pb, x.bbox) >= 0.7).map(|(_, nv)| *nv).collect();
                    if same.len() >= 3 && same.iter().all(|&nv| nv) {
                        dropped_static += 1;
                        continue;
                    }
                    let mean = if keep_a_box { x.bbox } else { [(x.bbox[0] + y[0]) / 2.0, (x.bbox[1] + y[1]) / 2.0, (x.bbox[2] + y[2]) / 2.0, (x.bbox[3] + y[3]) / 2.0] };
                    add(&mut anns, &mut meta, id, mean, x.class, false, true, x.score.max(lb[j].score), 0);
                    agreed += 1;
                }
                Some(j) => items.push(("differ", x.bbox, Some(x.class), Some(lb[j].class), x.score.max(lb[j].score))),
                None => items.push(("only_a", x.bbox, Some(x.class), None, x.score)),
            }
        }
        items.extend(ob.iter().map(|&j| ("only_b", lb[j].bbox, None, Some(lb[j].class), lb[j].score)));

        for (kind, bbox, ca, cb, score) in items {
            let m = by_box.get(&(id, key(&json!(bbox)))).ok_or_else(|| anyhow::anyhow!("frame {id} box {bbox:?} is not in the manifest"))?;
            anyhow::ensure!(m["kind"].as_str() == Some(kind), "manifest entry for frame {id} box {bbox:?} does not match the recomputed disputes");
            k += 1;
            let (sheet, cell) = (m["sheet"].as_u64().unwrap_or(u64::MAX), m["cell"].as_u64().unwrap_or(u64::MAX));
            let v = match m["auto"].as_str() {
                Some(v) => {
                    auto_n += 1;
                    v.to_string()
                }
                None => verdicts.get(&(sheet, cell)).ok_or_else(|| anyhow::anyhow!("no verdict for sheet {sheet} cell {cell}"))?.clone(),
            };
            let v = &v;
            let e = by_cam.entry(cam.clone()).or_default();
            let es = by_set.entry(set_of[&id].clone()).or_default();
            match (v.as_str(), task.class_id(v)) {
                (_, Some(c)) => {
                    riders += (v == "rider") as usize;
                    add(&mut anns, &mut meta, id, bbox, c, false, false, score, match (ca, cb) { (Some(_), None) => 1, (None, Some(_)) => 2, _ => 0 });
                    let row = by_class.entry(c).or_default();
                    match (ca, cb) {
                        (Some(x), Some(y)) => {
                            if x == c {
                                sa.dispute_right += 1;
                                row[2] += 1;
                            } else if y == c {
                                sb.dispute_right += 1;
                                row[3] += 1;
                            } else {
                                both_wrong += 1;
                            }
                        }
                        (Some(x), None) => {
                            sa.found += 1;
                            sa.found_class_right += (x == c) as usize;
                            sb.missed += 1;
                            e[3] += 1;
                            es[3] += 1;
                            row[1] += 1;
                        }
                        (None, Some(y)) => {
                            sb.found += 1;
                            sb.found_class_right += (y == c) as usize;
                            sa.missed += 1;
                            e[2] += 1;
                            es[2] += 1;
                            row[0] += 1;
                        }
                        (None, None) => unreachable!(),
                    }
                }
                (neg, _) if neg == task.negative => {
                    if let Some(x) = ca {
                        sa.fp += 1;
                        e[0] += 1;
                        es[0] += 1;
                        *fp_as[0].entry(names[x]).or_default() += 1;
                    }
                    if let Some(y) = cb {
                        sb.fp += 1;
                        e[1] += 1;
                        es[1] += 1;
                        *fp_as[1].entry(names[y]).or_default() += 1;
                    }
                    both_fp += (ca.is_some() && cb.is_some()) as usize;
                }
                ("duplicate", _) => {
                    sa.dup += ca.is_some() as usize;
                    sb.dup += cb.is_some() as usize;
                }
                (u, _) if task.unsure_classes(u).is_some() => {
                    unresolved += 1;
                    for c in task.unsure_classes(u).unwrap() {
                        add(&mut anns, &mut meta, id, bbox, c, true, false, score, 0);
                    }
                }
                (other, _) => anyhow::bail!("unknown verdict {other:?} at sheet {sheet} cell {cell}"),
            }
        }
    }
    anyhow::ensure!(k == manifest.len(), "manifest has {} entries, recomputed {k}", manifest.len());

    // sweep: objects neither model boxed, listed by frame from whole-frame sheets
    let sweep_dir = std::env::var("SWEEP").ok().filter(|d| !d.is_empty());
    let mut misses: Vec<Miss> = read_misses(dir)?;
    let mut sweep_line = None;
    if let Some(sd) = &sweep_dir {
        let sd = Path::new(sd);
        misses.extend(read_misses(sd)?);
        let sweep_manifest: Vec<Value> = serde_json::from_slice(&std::fs::read(sd.join("manifest.json"))?)?;
        let swept = read_verdicts(sd)?;
        let mut unswept = vec![];
        for m in &sweep_manifest {
            let (s, c) = (m["sheet"].as_u64().unwrap(), m["cell"].as_u64().unwrap());
            match swept.get(&(s, c)).map(String::as_str) {
                Some("swept") => {}
                Some(other) => anyhow::bail!("sweep sheet {s} cell {c}: verdict {other:?} (a sweep cell is `swept`; its people are `miss` lines)"),
                None => unswept.push(format!("{s}/{c}")),
            }
        }
        sweep_line = Some(format!("sweep: {} of {} frames swept{}", sweep_manifest.len() - unswept.len(), sweep_manifest.len(), if unswept.is_empty() { String::new() } else { format!("; not yet swept (sheet/cell): {}", unswept.join(" ")) }));
    }
    let (mut miss_added, mut miss_unsure, mut miss_dup) = (0usize, 0usize, 0usize);
    if !misses.is_empty() {
        anyhow::ensure!(task.name == "person", "miss lines are for the person task (TASK=person)");
        for m in &misses {
            anyhow::ensure!(camera.contains_key(&m.image_id), "miss line for frame {} which is not in {}", m.image_id, a[1]);
            let taken = anns.iter().any(|x| x["image_id"] == m.image_id && x["iscrowd"] == 0 && iou(m.bbox, key_f(&x["bbox"])) >= 0.5);
            if taken {
                miss_dup += 1;
                eprintln!("miss on frame {} at {:?} overlaps a box the frame already has; skipped", m.image_id, m.bbox);
                continue;
            }
            match m.label.as_str() {
                "unsure" => {
                    miss_unsure += 1;
                    add(&mut anns, &mut meta, m.image_id, m.bbox, 1, true, false, 0.0, 0);
                }
                l => {
                    riders += (l == "rider") as usize;
                    miss_added += 1;
                    add(&mut anns, &mut meta, m.image_id, m.bbox, 1, false, false, 0.0, 0);
                }
            }
        }
    }

    // person task: one box per person (see the module doc)
    // SAME_CLASS_DEDUPE=1 (vehicle task): two boxes of the same class on one object — a label kept
    // next to a model's box of another extent, or two model boxes — are one vehicle; the person
    // rule below, restricted to the same class so a two-wheeler inside a bus box is never touched
    let same_class_dedupe = task.name == "vehicle" && std::env::var("SAME_CLASS_DEDUPE").as_deref() == Ok("1");
    let mut suppressed = 0usize;
    if task.name == "person" || same_class_dedupe {
        let mut by_frame: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, x) in anns.iter().enumerate() {
            if x["iscrowd"] == 0 {
                by_frame.entry(x["image_id"].as_i64().unwrap()).or_default().push(i);
            }
        }
        let area = |b: [f32; 4]| b[2] * b[3];
        let inter = |a: [f32; 4], b: [f32; 4]| {
            let iw = ((a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0])).max(0.0);
            let ih = ((a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1])).max(0.0);
            iw * ih
        };
        let conflict = |a: [f32; 4], b: [f32; 4]| iou(a, b) >= 0.5 || inter(a, b) / area(a).min(area(b)).max(1e-6) >= 0.8;
        let mut drop = vec![false; anns.len()];
        for (id, idx) in by_frame.iter_mut() {
            // precedence: agreed, then score, then area
            idx.sort_by(|&i, &j| {
                let (bi, bj) = (key_f(&anns[i]["bbox"]), key_f(&anns[j]["bbox"]));
                meta[j].0.cmp(&meta[i].0).then(meta[j].1.total_cmp(&meta[i].1)).then(area(bj).total_cmp(&area(bi)))
            });
            let mut kept: Vec<([f32; 4], u64)> = vec![];
            for &i in idx.iter() {
                let b = key_f(&anns[i]["bbox"]);
                let es = by_set.entry(set_of[id].clone()).or_default();
                let cls = anns[i]["category_id"].as_u64().unwrap_or(0);
                if kept.iter().any(|(k, kc)| conflict(*k, b) && (!same_class_dedupe || *kc == cls)) {
                    drop[i] = true;
                    suppressed += 1;
                    es[4] += 1;
                } else {
                    kept.push((b, cls));
                    // a surviving box only B drew is a person A missed, and the other way round
                    match meta[i].2 {
                        2 => es[5] += 1,
                        1 => es[6] += 1,
                        _ => {}
                    }
                }
            }
        }
        anns = anns.into_iter().enumerate().filter(|(i, _)| !drop[*i]).map(|(_, x)| x).collect();
        for (n, x) in anns.iter_mut().enumerate() {
            x["id"] = json!(n + 1);
        }
    }

    let (na, nb) = (Path::new(&a[2]).file_name().unwrap().to_string_lossy(), Path::new(&a[3]).file_name().unwrap().to_string_lossy());
    let disputes = sa.dispute_right + sb.dispute_right + both_wrong;
    println!("{dropped_static} agreed boxes dropped as shared static false positives");
    println!("{} frames, {agreed} agreed boxes, {} disputes reviewed ({auto_n} static false positives resolved from earlier reviews, {unresolved} left unsure)", camera.len(), manifest.len());
    println!("A = {na}\nB = {nb}\n");
    println!("{:<44}{:>8}{:>8}", "", "A", "B");
    println!("{:<44}{:>8}{:>8}", format!("false positives (not a {})", task.noun), sa.fp, sb.fp);
    println!("{:<44}{:>8}{:>8}", "  of which both models", both_fp, both_fp);
    println!("{:<44}{:>8}{:>8}", "duplicate boxes", sa.dup, sb.dup);
    println!("{:<44}{:>8}{:>8}", format!("real {} only this model found", task.plural), sa.found, sb.found);
    println!("{:<44}{:>8}{:>8}", "  with the right class", sa.found_class_right, sb.found_class_right);
    println!("{:<44}{:>8}{:>8}", format!("real {} missed (the other found)", task.plural), sa.missed, sb.missed);
    println!("{:<44}{:>8}{:>8}", format!("class disputes won (of {disputes}; {both_wrong} neither)"), sa.dispute_right, sb.dispute_right);

    println!("\nfalse positives called as");
    for (i, m) in fp_as.iter().enumerate() {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|x, y| y.1.cmp(x.1));
        println!("  {}: {}", ["A", "B"][i], v.iter().map(|(n, c)| format!("{n} {c}")).collect::<Vec<_>>().join(", "));
    }
    println!("\n{:<16}{:>8}{:>8}{:>10}{:>10}", "camera", "FP A", "FP B", "miss A", "miss B");
    for (c, v) in &by_cam {
        println!("{c:<16}{:>8}{:>8}{:>10}{:>10}", v[0], v[1], v[2], v[3]);
    }
    println!("\n{:<16}{:>8}{:>8}{:>12}{:>12}", "true class", "miss A", "miss B", "dispute A", "dispute B");
    for (c, v) in &by_class {
        println!("{:<16}{:>8}{:>8}{:>12}{:>12}", names[*c], v[0], v[1], v[2], v[3]);
    }
    if task.name == "person" {
        println!("\n{riders} of the person boxes are riders (verdict `rider`, exported as person)");
        if let Some(l) = &sweep_line {
            println!("{l}");
        }
        if !misses.is_empty() || sweep_dir.is_some() {
            println!("sweep misses: {miss_added} people added that neither model boxed, {miss_unsure} unsure regions, {miss_dup} skipped as already boxed");
        }
        println!("{suppressed} duplicate person boxes suppressed (IoU ≥ 0.5, or the smaller ≥ 80 % inside the larger; agreed > score > area kept)");
        println!("\n{:<16}{:>8}{:>8}{:>10}{:>10}{:>12}{:>12}{:>12}", "set", "FP A", "FP B", "miss A", "miss B", "suppressed", "net miss A", "net miss B");
        for (s, v) in &by_set {
            println!("{s:<16}{:>8}{:>8}{:>10}{:>10}{:>12}{:>12}{:>12}", v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
        }
        println!("(miss = reviewed people only the other model boxed; net = those still standing after the duplicate suppression)");
    }

    let cats: Vec<Value> = task.categories();
    let images: Vec<Value> = frames["images"].as_array().unwrap().to_vec();
    let real = anns.iter().filter(|x| x["iscrowd"] == 0).count();
    let out = dir.join("reviewed_gold.json");
    std::fs::write(&out, serde_json::to_string(&json!({"info": {"description": "provisional gold: two-model agreement plus a by-eye review of every disagreement; not human-verified"}, "images": images, "categories": cats, "annotations": anns}))?)?;
    println!("\n{real} boxes ({} crowd/ignore entries for the unsure ones) → {}", anns.len() - real, out.display());
    Ok(())
}
