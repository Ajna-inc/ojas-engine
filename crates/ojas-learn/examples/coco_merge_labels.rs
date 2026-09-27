//! Join a second labelling onto an existing training set by image id, into a new directory;
//! neither input is modified. Written for the person rounds: `base` is `field_train/train.json`
//! (vehicles, ids 1..14, images already greyed by `review_export`), `extra` is the person round's
//! labels-only export (`review_export` with `KEEP_IDS=1`: category 15, same image ids and
//! `file_name`, the regions to grey as crowd entries).
//!
//! - Frames: by default only those in both files (`--joined`), since a `base` frame the person
//!   round did not label would teach the person class that people are background. `--all` keeps
//!   every `base` frame, for a vehicle-only use of the file.
//! - Annotations: every `base` annotation, then the `extra` labelled boxes (`iscrowd` 0), all
//!   re-numbered from 1 in that order; categories are the union by id (the same id must carry
//!   the same name).
//! - Greying: each `extra` crowd region is filled with grey 114 (the letterbox colour) in a copy of
//!   the `base` image, except pixels inside any labelled box of either file, so an unsure person
//!   region never blanks a labelled vehicle or vice versa. The crowd entries are then left out,
//!   their pixels being grey; the `base` greying stays as it was; a frame with no region to grey is
//!   copied byte for byte.
//! - Images go to `out_dir/images/<file name>`, `file_name` becomes `images/<file name>` (the old
//!   one kept as `base_file_name`), and the set to `out_dir/train.json`. A relative `file_name` of
//!   `base` is read from `ROOT` (env).
//!
//! `coco_merge_labels <base.json> <extra.json> <out_dir> [--joined|--all] [--limit N]`
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use image::Rgb;
use serde_json::{json, Value};

fn fbox(v: &Value) -> [f32; 4] {
    let b: Vec<f32> = v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    [b[0], b[1], b[2], b[3]]
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 4, "coco_merge_labels <base.json> <extra.json> <out_dir> [--joined|--all] [--limit N]");
    let all = a.iter().any(|x| x == "--all");
    let limit: usize = a.iter().position(|x| x == "--limit").and_then(|i| a.get(i + 1)).map(|v| v.parse()).transpose()?.unwrap_or(usize::MAX);
    let root = std::env::var("ROOT").unwrap_or_default();
    let base: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let extra: Value = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    let out = Path::new(&a[3]);
    anyhow::ensure!(!out.join("train.json").exists(), "{} already exists; pick a new out_dir", out.join("train.json").display());
    for input in [&a[1], &a[2]] {
        let p = Path::new(input).canonicalize()?;
        anyhow::ensure!(!p.starts_with(out.canonicalize().unwrap_or_else(|_| out.to_path_buf())), "out_dir must not hold the inputs");
    }
    std::fs::create_dir_all(out.join("images"))?;

    // categories: union by id
    let mut cats: BTreeMap<i64, Value> = BTreeMap::new();
    for c in base["categories"].as_array().unwrap().iter().chain(extra["categories"].as_array().unwrap()) {
        let id = c["id"].as_i64().unwrap();
        if let Some(old) = cats.get(&id) {
            anyhow::ensure!(old["name"] == c["name"], "category {id} is {} in one file and {} in the other", old["name"], c["name"]);
        } else {
            cats.insert(id, c.clone());
        }
    }
    let group = |v: &Value| {
        let mut m: HashMap<i64, Vec<Value>> = HashMap::new();
        for an in v["annotations"].as_array().unwrap() {
            m.entry(an["image_id"].as_i64().unwrap()).or_default().push(an.clone());
        }
        m
    };
    let (ba, ea) = (group(&base), group(&extra));
    let extra_ids: HashSet<i64> = extra["images"].as_array().unwrap().iter().map(|im| im["id"].as_i64().unwrap()).collect();
    let base_ids: HashSet<i64> = base["images"].as_array().unwrap().iter().map(|im| im["id"].as_i64().unwrap()).collect();
    let orphans = extra_ids.difference(&base_ids).count();
    anyhow::ensure!(orphans == 0, "{orphans} frames of {} are not in {}", a[2], a[1]);

    let (mut images, mut anns) = (vec![], vec![]);
    let (mut n_base, mut n_extra, mut n_regions, mut grey_px, mut total_px, mut copied, mut names) = (0usize, 0usize, 0usize, 0u64, 0u64, 0usize, HashSet::new());
    for im in base["images"].as_array().unwrap() {
        if images.len() >= limit {
            break;
        }
        let id = im["id"].as_i64().unwrap();
        let joined = extra_ids.contains(&id);
        if !joined && !all {
            continue;
        }
        let empty = vec![];
        let (bl, el) = (ba.get(&id).unwrap_or(&empty), ea.get(&id).unwrap_or(&empty));
        let labelled: Vec<[f32; 4]> = bl.iter().chain(el).filter(|x| x["iscrowd"].as_i64().unwrap_or(0) == 0).map(|x| fbox(&x["bbox"])).collect();
        let regions: Vec<[f32; 4]> = el.iter().filter(|x| x["iscrowd"].as_i64().unwrap_or(0) == 1).map(|x| fbox(&x["bbox"])).collect();
        let file = im["file_name"].as_str().unwrap();
        let src = if Path::new(file).is_absolute() { file.to_string() } else { Path::new(&root).join(file).to_string_lossy().into_owned() };
        let name = Path::new(file).file_name().unwrap().to_string_lossy().into_owned();
        anyhow::ensure!(names.insert(name.clone()), "two frames share the file name {name}");
        let dst = out.join("images").join(&name);
        let (w, h) = (im["width"].as_u64().unwrap_or(1920), im["height"].as_u64().unwrap_or(1080));
        total_px += w * h;
        if regions.is_empty() {
            std::fs::copy(&src, &dst)?;
            copied += 1;
        } else {
            let mut img = image::open(&src).map_err(|e| anyhow::anyhow!("{src}: {e}"))?.to_rgb8();
            let (iw, ih) = (img.width() as i64, img.height() as i64);
            let mut done = HashSet::new();
            for b in &regions {
                n_regions += 1;
                let (x0, y0) = ((b[0].floor() as i64).max(0), (b[1].floor() as i64).max(0));
                let (x1, y1) = (((b[0] + b[2]).ceil() as i64).min(iw), ((b[1] + b[3]).ceil() as i64).min(ih));
                for y in y0..y1 {
                    for x in x0..x1 {
                        let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                        if labelled.iter().any(|k| fx >= k[0] && fx < k[0] + k[2] && fy >= k[1] && fy < k[1] + k[3]) {
                            continue;
                        }
                        img.put_pixel(x as u32, y as u32, Rgb([114, 114, 114]));
                        if done.insert((x, y)) {
                            grey_px += 1;
                        }
                    }
                }
            }
            img.save(&dst)?;
        }
        let mut im = im.clone();
        im["base_file_name"] = json!(file);
        im["file_name"] = json!(format!("images/{name}"));
        im["persons_labelled"] = json!(joined);
        images.push(im);
        for x in bl {
            let mut x = x.clone();
            x["id"] = json!(anns.len() + 1);
            anns.push(x);
            n_base += 1;
        }
        for x in el.iter().filter(|x| x["iscrowd"].as_i64().unwrap_or(0) == 0) {
            let mut x = x.clone();
            x["id"] = json!(anns.len() + 1);
            anns.push(x);
            n_extra += 1;
        }
    }
    let cats: Vec<Value> = cats.into_values().collect();
    let info = json!({"description": "a training set joined with a second labelling by image id; the second labelling's crowd regions greyed into copies of the images", "base": a[1], "extra": a[2], "frames": if all { "all base frames" } else { "frames in both" }});
    std::fs::write(out.join("train.json"), serde_json::to_string(&json!({"info": info, "images": images, "categories": cats, "annotations": anns}))?)?;
    println!(
        "{} frames ({} with the second labelling; {copied} copied as is), {n_base} base + {n_extra} added boxes; {n_regions} regions greyed ({:.2} % of pixels) → {}",
        images.len(),
        images.iter().filter(|im| im["persons_labelled"] == true).count(),
        100.0 * grey_px as f64 / total_px.max(1) as f64,
        out.join("train.json").display()
    );
    Ok(())
}
