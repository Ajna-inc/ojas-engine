//! One training file for a vehicle + person model, from sets that label different classes. No image
//! is copied or touched: the output is one COCO json whose `file_name`s point into the existing
//! folders under a shared image root.
//!
//! - `base` (e.g. `field_train/train_bal.json`, vehicles only, repeats kept) is walked in order. A
//!   frame whose `source` is in `labelled` (e.g. `person_r1/merged/train.json`: the same frames with
//!   people added and unsure people greyed) is replaced by that labelled frame and its boxes; every
//!   other frame keeps its own boxes and gets `"unlabelled_cats": [<ids>]`.
//! - Every labelled frame is then added `REPEAT - 1` more times (env, default 1 = no extra copies)
//!   to raise how often the person class is seen; labelled frames `base` skipped still appear once.
//! - Each `extra` input (e.g. UVH-26 train, `path[=prefix/][#k]`: every k-th image) is appended with
//!   `unlabelled_cats` set: it never labels people.
//! - Categories are `labelled`'s, and `base` and `extra` must agree with them on every id they use.
//!   Image and box ids are renumbered from 1. `UNLABELLED` (env, default `15`) is the comma-separated
//!   list of class ids a frame without the labelled round does not label.
//!
//! The trainer must honour `unlabelled_cats` — the ojas DEIM patch drops those classes' terms from
//! that image's classification loss. Without it, those people are learned as background.
//!
//! `labelled` may list several rounds joined by `,` (`r1.json=pre1/,r2.json=pre2/`); a `source` must
//! not appear in two of them.
//!
//! `coco_person_mix <out.json> <base.json[=prefix/]> <labelled.json[=prefix/][,more.json[=prefix/]]> [<extra.json[=prefix/][#k]> ...]`
use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};

fn split(arg: &str) -> (String, String, usize) {
    let (arg, every) = match arg.rsplit_once('#') {
        Some((a, k)) => (a, k.parse().expect("#k must be a number")),
        None => (arg, 1),
    };
    let (path, prefix) = arg.split_once('=').unwrap_or((arg, ""));
    (path.to_string(), prefix.to_string(), every)
}

fn load(path: &str) -> anyhow::Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn by_image(v: &Value) -> HashMap<i64, Vec<Value>> {
    let mut m: HashMap<i64, Vec<Value>> = HashMap::new();
    for an in v["annotations"].as_array().into_iter().flatten() {
        m.entry(an["image_id"].as_i64().unwrap_or(-1)).or_default().push(an.clone());
    }
    m
}

struct Out {
    images: Vec<Value>,
    anns: Vec<Value>,
    cats_used: HashMap<i64, usize>,
}

impl Out {
    fn push(&mut self, im: &Value, prefix: &str, boxes: &[Value], unlabelled: Option<&[i64]>) {
        let id = self.images.len() as i64 + 1;
        let mut im = im.clone();
        im["id"] = json!(id);
        im["file_name"] = json!(format!("{prefix}{}", im["file_name"].as_str().unwrap_or("")));
        match unlabelled {
            Some(u) => im["unlabelled_cats"] = json!(u),
            None => {
                if let Some(o) = im.as_object_mut() {
                    o.remove("unlabelled_cats");
                }
            }
        }
        self.images.push(im);
        for an in boxes {
            let mut an = an.clone();
            an["id"] = json!(self.anns.len() as i64 + 1);
            an["image_id"] = json!(id);
            *self.cats_used.entry(an["category_id"].as_i64().unwrap_or(-1)).or_default() += 1;
            self.anns.push(an);
        }
    }
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 4, "coco_person_mix <out.json> <base.json[=prefix/]> <labelled.json[=prefix/]> [<extra.json[=prefix/][#k]> ...]");
    let unlabelled: Vec<i64> = std::env::var("UNLABELLED").unwrap_or_else(|_| "15".into()).split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
    let repeat: usize = std::env::var("REPEAT").ok().map(|s| s.parse()).transpose()?.unwrap_or(1);
    anyhow::ensure!(repeat >= 1, "REPEAT must be at least 1");

    let (bpath, bprefix, _) = split(&a[2]);
    let base = load(&bpath)?;
    let labs: Vec<(String, String, Value)> = a[3].split(',').map(|s| { let (p, pre, _) = split(s); load(&p).map(|v| (p, pre, v)) }).collect::<anyhow::Result<_>>()?;
    let lpath = labs[0].0.clone();
    let cats = labs[0].2["categories"].clone();
    let names: HashMap<i64, String> = cats.as_array().into_iter().flatten().map(|c| (c["id"].as_i64().unwrap_or(-1), c["name"].as_str().unwrap_or("").to_string())).collect();
    let check_cats = |v: &Value, path: &str| -> anyhow::Result<()> {
        for c in v["categories"].as_array().into_iter().flatten() {
            let id = c["id"].as_i64().unwrap_or(-1);
            let name = c["name"].as_str().unwrap_or("");
            anyhow::ensure!(names.get(&id).map(String::as_str) == Some(name), "{path}: category {id} {name:?} is not in {}", lpath);
        }
        Ok(())
    };
    check_cats(&base, &bpath)?;
    for (p, _, v) in &labs[1..] {
        check_cats(v, p)?;
    }
    for u in &unlabelled {
        anyhow::ensure!(names.contains_key(u), "UNLABELLED id {u} is not a category of {lpath}");
    }

    // source -> (labelled frame, its prefix, its boxes)
    let lab_boxes: Vec<HashMap<i64, Vec<Value>>> = labs.iter().map(|(_, _, v)| by_image(v)).collect();
    let mut lab_by_source: HashMap<String, (&Value, &str, &[Value])> = HashMap::new();
    let mut lab_order: Vec<String> = vec![];
    for (li, (lp, pre, v)) in labs.iter().enumerate() {
        for im in v["images"].as_array().into_iter().flatten() {
            let s = im["source"].as_str().unwrap_or("");
            anyhow::ensure!(!s.is_empty(), "{lp}: image {} has no source", im["id"]);
            let bx = lab_boxes[li].get(&im["id"].as_i64().unwrap_or(-1)).map_or(&[][..], |v| v.as_slice());
            anyhow::ensure!(lab_by_source.insert(s.to_string(), (im, pre.as_str(), bx)).is_none(), "{lp}: source {s} is in two labelled inputs");
            lab_order.push(s.to_string());
        }
    }

    let mut out = Out { images: vec![], anns: vec![], cats_used: HashMap::new() };
    let base_boxes = by_image(&base);
    let (mut n_swapped, mut n_base) = (0usize, 0usize);
    let mut seen = HashSet::new();
    for im in base["images"].as_array().into_iter().flatten() {
        let s = im["source"].as_str().unwrap_or("");
        if let Some((l, pre, bx)) = lab_by_source.get(s) {
            out.push(l, pre, bx, None);
            seen.insert(s.to_string());
            n_swapped += 1;
        } else {
            let id = im["id"].as_i64().unwrap_or(-1);
            out.push(im, &bprefix, base_boxes.get(&id).map_or(&[][..], |v| v), Some(&unlabelled));
            n_base += 1;
        }
    }
    // labelled frames: the extra copies, plus one for any the base skipped
    let mut n_added = 0usize;
    for s in &lab_order {
        let (im, pre, bx) = lab_by_source[s];
        let times = if seen.contains(s) { repeat - 1 } else { repeat };
        for _ in 0..times {
            out.push(im, pre, bx, None);
            n_added += 1;
        }
    }
    println!("{bpath}: {n_base} frames kept with class(es) {unlabelled:?} unlabelled, {n_swapped} swapped for their labelled version");
    println!("{}: {} frames, {n_added} more copies added (REPEAT={repeat})", a[3], lab_by_source.len());

    for arg in &a[4..] {
        let (path, prefix, every) = split(arg);
        let v = load(&path)?;
        check_cats(&v, &path)?;
        let boxes = by_image(&v);
        let mut n = 0;
        for (i, im) in v["images"].as_array().into_iter().flatten().enumerate() {
            if i % every != 0 {
                continue;
            }
            let id = im["id"].as_i64().unwrap_or(-1);
            out.push(im, &prefix, boxes.get(&id).map_or(&[][..], |v| v), Some(&unlabelled));
            n += 1;
        }
        println!("{path}: {n} frames (every {every}), class(es) {unlabelled:?} unlabelled");
    }

    let n_lab = out.images.iter().filter(|im| im.get("unlabelled_cats").is_none()).count();
    let mut used: Vec<_> = out.cats_used.iter().collect();
    used.sort();
    let info = json!({"description": "vehicle + person training mix; frames with unlabelled_cats do not label those classes", "base": a[2], "labelled": a[3], "extra": &a[4..], "repeat": repeat, "unlabelled": unlabelled});
    std::fs::write(&a[1], serde_json::to_vec(&json!({"info": info, "images": out.images, "annotations": out.anns, "categories": cats}))?)?;
    println!("→ {}: {} frames ({n_lab} with every class labelled), {} boxes", a[1], out.images.len(), out.anns.len());
    for (c, n) in used {
        println!("  {:>2} {:<16} {n}", c, names.get(c).map_or("?", String::as_str));
    }
    Ok(())
}
