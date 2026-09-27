//! One image list out of several COCO files, each image tagged with the set it came from. For a
//! review round over frames that already form other sets — the person round R0 reuses the vehicle
//! gold's 852 test frames from three files — so one `frames.json` drives `person_survey` and the
//! review tools while every frame keeps the id it has in its own gold file.
//!
//! Only the images are taken; annotations and categories are not (the round's own labels come
//! from its review). Every image gets `"set": <tag>`; ids must not collide across inputs; a
//! relative `file_name` is resolved against `ROOT` (env) and every file must exist.
//!
//! `coco_tag_frames <out.json> <a.json=tag> [<b.json=tag> ...]`
use std::collections::HashSet;
use std::path::Path;

use serde_json::{json, Value};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 3, "coco_tag_frames <out.json> <a.json=tag> [<b.json=tag> ...]");
    let root = std::env::var("ROOT").unwrap_or_default();
    let mut images: Vec<Value> = vec![];
    let mut ids = HashSet::new();
    let mut sources = vec![];
    for arg in &a[2..] {
        let (path, tag) = arg.split_once('=').ok_or_else(|| anyhow::anyhow!("{arg}: expected <file.json>=<tag>"))?;
        let v: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let mut n = 0;
        for im in v["images"].as_array().ok_or_else(|| anyhow::anyhow!("{path}: no images"))? {
            let id = im["id"].as_i64().ok_or_else(|| anyhow::anyhow!("{path}: image without id"))?;
            anyhow::ensure!(ids.insert(id), "{path}: image id {id} is already in an earlier input");
            let file = im["file_name"].as_str().ok_or_else(|| anyhow::anyhow!("{path}: image {id} without file_name"))?;
            let file = if Path::new(file).is_absolute() { file.to_string() } else { Path::new(&root).join(file).to_string_lossy().into_owned() };
            anyhow::ensure!(Path::new(&file).is_file(), "{path}: image {id}: {file} does not exist");
            let mut im = im.clone();
            im["file_name"] = json!(file);
            im["set"] = json!(tag);
            images.push(im);
            n += 1;
        }
        println!("{tag:<10} {n:>6} frames from {path}");
        sources.push(json!({"file": path, "set": tag, "frames": n}));
    }
    let out = json!({"info": {"description": "frame list for a review round, tagged by source set", "sources": sources}, "images": images, "categories": [], "annotations": []});
    std::fs::write(&a[1], serde_json::to_string(&out)?)?;
    println!("{} frames → {}", images.len(), a[1]);
    Ok(())
}
