//! Turn a reviewed round (`review_report`'s `reviewed_gold.json`) into a training set a COCO
//! loader can use as is.
//!
//! A review labels every vehicle either model found at score ≥ `review_score`, but a loader has no
//! "don't know": crowd boxes are dropped and anything unlabelled trains as background. The export
//! therefore removes what it cannot vouch for from the pixels:
//!
//! - the unsure boxes (crowd entries of the review) and every detection of either model in
//!   [`low`, `review_score`) that no labelled box covers (IoU < 0.3) — objects nobody reviewed,
//!   often real vehicles both models were unsure of — are filled with grey (114, the letterbox
//!   colour) where they do not overlap a labelled box, provided they are smaller than `max_area`
//!   of the frame (default 2 %; a large low-score box is a whole-road or shop-front hallucination,
//!   and greying it would blank a fifth of a night frame);
//! - a frame whose labelled boxes are exactly (class, box to the pixel) those of an earlier frame
//!   of the same camera is a repeat — a frozen or re-sent segment — and is left out;
//! - frames within `gap_s` seconds of a test frame of the same camera are left out, since frames
//!   0.56 s apart show the same vehicles and would leak the test set into training.
//!
//! Writes `out_dir/images/<camera>_<utc_ms>.jpg` and `out_dir/train.json` (file names relative to
//! `out_dir/images`). Categories are the reviewed file's, so a person round (`TASK=person` in
//! `review_report`: category 15 `person`, unsure people as crowd) exports like a vehicle round
//! under the same greying rules; `coco_merge` joins a person train.json with a vehicle one when
//! the frames differ.
//!
//! Labels only (`KEEP_IDS=1`): no image is written and nothing greyed, and `<out_dir>` is the
//! output JSON file — the reviewed frames with their own ids and `file_name`, the labelled boxes,
//! and every region the export would grey as a crowd entry (`iscrowd` 1, `"ignore": "unsure"` or
//! `"low"`, the reviewed file's first category). A second labelling of frames that already belong
//! to a training set takes this form (the person round over `field_train`); `coco_merge_labels`
//! joins it onto that set by image id and greys the crowd regions there.
//!
//! `review_export <reviewed_gold.json> <A.dets.json> <B.dets.json> <out_dir> [test_frames.json] [gap_s 60] [low 0.1] [review_score 0.3] [max_area 0.02]`
use std::collections::HashMap;
use std::path::Path;

use image::{Rgb, RgbImage};
use serde_json::{json, Value};

#[path = "review_common/mod.rs"]
mod review_common;
use review_common::{iou, load_dets};

fn fbox(v: &Value) -> [f32; 4] {
    let b: Vec<f32> = v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    [b[0], b[1], b[2], b[3]]
}

/// Grey out `b` except the pixels inside any of `keep`.
fn grey(img: &mut RgbImage, b: [f32; 4], keep: &[[f32; 4]]) -> u64 {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (x0, y0) = ((b[0].floor() as i64).max(0), (b[1].floor() as i64).max(0));
    let (x1, y1) = (((b[0] + b[2]).ceil() as i64).min(w), ((b[1] + b[3]).ceil() as i64).min(h));
    let mut n = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            if keep.iter().any(|k| fx >= k[0] && fx < k[0] + k[2] && fy >= k[1] && fy < k[1] + k[3]) {
                continue;
            }
            img.put_pixel(x as u32, y as u32, Rgb([114, 114, 114]));
            n += 1;
        }
    }
    n
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 5, "review_export <reviewed_gold.json> <A.dets.json> <B.dets.json> <out_dir> [test_frames.json] [gap_s 60] [low 0.1] [review_score 0.3] [max_area 0.02]");
    let gap_ms = a.get(6).map(|v| v.parse::<f64>().unwrap()).unwrap_or(60.0) * 1000.0;
    let low: f32 = a.get(7).map(|v| v.parse().unwrap()).unwrap_or(0.1);
    let review_score: f32 = a.get(8).map(|v| v.parse().unwrap()).unwrap_or(0.3);
    let max_area: f32 = a.get(9).map(|v| v.parse().unwrap()).unwrap_or(0.02);
    let reviewed: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let (da, db) = (load_dets(&a[2], low)?, load_dets(&a[3], low)?);
    let out = Path::new(&a[4]);
    let keep_ids = std::env::var("KEEP_IDS").is_ok_and(|v| v == "1");
    if !keep_ids {
        std::fs::create_dir_all(out.join("images"))?;
    }
    let first_cat = reviewed["categories"][0]["id"].clone();

    // test frames per camera, by time
    let mut test: HashMap<String, Vec<i64>> = HashMap::new();
    if let Some(p) = a.get(5).filter(|p| !p.is_empty()) {
        let t: Value = serde_json::from_slice(&std::fs::read(p)?)?;
        for im in t["images"].as_array().unwrap() {
            test.entry(im["camera"].as_str().unwrap_or("?").split('@').next().unwrap().to_string()).or_default().push(im["utc_ms"].as_i64().unwrap_or(0));
        }
    }

    let mut by_image: HashMap<i64, Vec<&Value>> = HashMap::new();
    for an in reviewed["annotations"].as_array().unwrap() {
        by_image.entry(an["image_id"].as_i64().unwrap()).or_default().push(an);
    }
    let (mut images, mut anns) = (vec![], vec![]);
    let mut seen: std::collections::HashSet<(String, Vec<(i64, [i32; 4])>)> = Default::default();
    let mut repeats = 0usize;
    let (mut leaked, mut n_ignore, mut n_low, mut grey_px, mut total_px) = (0usize, 0usize, 0usize, 0u64, 0u64);
    for im in reviewed["images"].as_array().unwrap() {
        let id = im["id"].as_i64().unwrap();
        let cam = im["camera"].as_str().unwrap_or("?").split('@').next().unwrap().to_string();
        let t = im["utc_ms"].as_i64().unwrap_or(0);
        if test.get(&cam).is_some_and(|ts| ts.iter().any(|&u| ((u - t) as f64).abs() <= gap_ms)) {
            leaked += 1;
            continue;
        }
        let empty = vec![];
        let list = by_image.get(&id).unwrap_or(&empty);
        let real: Vec<&Value> = list.iter().copied().filter(|x| x["iscrowd"] == 0).collect();
        let keep: Vec<[f32; 4]> = real.iter().map(|x| fbox(&x["bbox"])).collect();
        let mut sig: Vec<(i64, [i32; 4])> = real.iter().map(|x| {
            let b = fbox(&x["bbox"]);
            (x["category_id"].as_i64().unwrap(), [b[0].round() as i32, b[1].round() as i32, b[2].round() as i32, b[3].round() as i32])
        }).collect();
        sig.sort();
        if !sig.is_empty() && !seen.insert((cam.clone(), sig)) {
            repeats += 1;
            continue;
        }
        // unsure boxes, once each (the review repeats them per possible class)
        let mut ignore: Vec<[f32; 4]> = vec![];
        for x in list.iter().filter(|x| x["iscrowd"] == 1) {
            let b = fbox(&x["bbox"]);
            if !ignore.iter().any(|g| iou(*g, b) > 0.99) {
                ignore.push(b);
                n_ignore += 1;
            }
        }
        // unreviewed low-score detections no labelled box covers
        let frame_area = (im["width"].as_f64().unwrap_or(1920.0) * im["height"].as_f64().unwrap_or(1080.0)) as f32;
        for d in da.get(&id).into_iter().chain(db.get(&id)).flatten().filter(|d| d.score < review_score && d.bbox[2] * d.bbox[3] < max_area * frame_area) {
            if keep.iter().chain(ignore.iter()).all(|k| iou(*k, d.bbox) < 0.3) {
                ignore.push(d.bbox);
                n_low += 1;
            }
        }
        if keep_ids {
            // labels only: regions as crowd entries, the greyed share counted on a mask
            let (w, h) = (im["width"].as_u64().unwrap_or(1920) as usize, im["height"].as_u64().unwrap_or(1080) as usize);
            let mut mask = vec![false; w * h];
            let n_unsure = list.iter().filter(|x| x["iscrowd"] == 1).fold(Vec::<[f32; 4]>::new(), |mut g, x| {
                let b = fbox(&x["bbox"]);
                if !g.iter().any(|q| iou(*q, b) > 0.99) {
                    g.push(b);
                }
                g
            }).len();
            for (i, b) in ignore.iter().enumerate() {
                let (x0, y0) = ((b[0].floor() as i64).clamp(0, w as i64) as usize, (b[1].floor() as i64).clamp(0, h as i64) as usize);
                let (x1, y1) = (((b[0] + b[2]).ceil() as i64).clamp(0, w as i64) as usize, ((b[1] + b[3]).ceil() as i64).clamp(0, h as i64) as usize);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                        if !keep.iter().any(|k| fx >= k[0] && fx < k[0] + k[2] && fy >= k[1] && fy < k[1] + k[3]) {
                            mask[y * w + x] = true;
                        }
                    }
                }
                anns.push(json!({"id": anns.len() + 1, "image_id": id, "category_id": first_cat, "bbox": b, "area": b[2] * b[3], "iscrowd": 1, "ignore": if i < n_unsure { "unsure" } else { "low" }}));
            }
            grey_px += mask.iter().filter(|&&m| m).count() as u64;
            total_px += (w * h) as u64;
            images.push(im.clone());
            for x in real {
                let b = fbox(&x["bbox"]);
                anns.push(json!({"id": anns.len() + 1, "image_id": id, "category_id": x["category_id"], "bbox": b, "area": b[2] * b[3], "iscrowd": 0}));
            }
            continue;
        }
        let src = im["file_name"].as_str().unwrap();
        let name = format!("{cam}_{t}.jpg");
        let mut img = image::open(src)?.to_rgb8();
        total_px += img.width() as u64 * img.height() as u64;
        for b in &ignore {
            grey_px += grey(&mut img, *b, &keep);
        }
        img.save(out.join("images").join(&name))?;
        let new_id = images.len() as i64;
        images.push(json!({"id": new_id, "file_name": name, "width": im["width"], "height": im["height"], "camera": im["camera"], "utc_ms": t, "source": src}));
        for x in real {
            let b = fbox(&x["bbox"]);
            anns.push(json!({"id": anns.len() + 1, "image_id": new_id, "category_id": x["category_id"], "bbox": b, "area": b[2] * b[3], "iscrowd": 0}));
        }
    }
    let cats = reviewed["categories"].clone();
    if keep_ids {
        let real = anns.iter().filter(|x| x["iscrowd"] == 0).count();
        std::fs::write(out, serde_json::to_string(&json!({"info": {"description": "labels of a two-model review on frames of an existing training set: same image ids and file_name; crowd entries are the regions to grey (unsure, and unreviewed low-score detections)", "from": a[1], "test_gap_s": gap_ms / 1000.0, "low": low, "review_score": review_score, "max_area": max_area}, "images": images, "categories": cats, "annotations": anns}))?)?;
        println!("labels only: {} frames ({leaked} left out within {:.0} s of a test frame, {repeats} exact repeats), {real} boxes; {n_ignore} unsure + {n_low} unreviewed low-score regions as crowd ({:.2} % of pixels to grey) → {}", images.len(), gap_ms / 1000.0, 100.0 * grey_px as f64 / total_px.max(1) as f64, out.display());
        return Ok(());
    }
    std::fs::write(out.join("train.json"), serde_json::to_string(&json!({"info": {"description": "Field training set from a two-model review: agreed boxes plus reviewed disputes; unsure and unreviewed low-score regions greyed out", "from": a[1], "test_gap_s": gap_ms / 1000.0, "low": low, "review_score": review_score}, "images": images, "categories": cats, "annotations": anns}))?)?;
    println!(
        "{} frames ({leaked} left out within {:.0} s of a test frame, {repeats} exact repeats), {} boxes; greyed {n_ignore} unsure + {n_low} unreviewed low-score regions ({:.2} % of pixels) → {}",
        images.len(),
        gap_ms / 1000.0,
        anns.len(),
        100.0 * grey_px as f64 / total_px.max(1) as f64,
        out.join("train.json").display()
    );
    Ok(())
}
