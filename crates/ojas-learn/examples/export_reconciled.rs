//! Passenger crops whose label both of UVH-26's labellings agree on.
//!
//! UVH-26 ships two labellings of the training set, MV and ST, which disagree on ~12 % of passenger
//! boxes (`label_agreement`). A box is kept only when the MV and ST annotations match at IoU ≥ 0.7
//! and carry the same class, so training never learns one side of a contradiction and is scored on
//! the other.
//!
//! Writes the JSONL manifest `passenger_train` reads: one crop per line with
//! `path`, `label` (0..4 in passenger order), `class_name`, `split`, `group`
//! (one per source image, so a scene never crosses the split), and the content
//! hashes the trainer uses to refuse leaks. Crops are cut at original resolution
//! with 15 % context; boxes under `min_px` are skipped.
//!
//! `export_reconciled <mv.json> <st.json> <image_root> <out_dir> [dev_share=0.1] [min_px=48]`
use std::collections::HashMap;
use std::io::Write;

use ojas_learn::data::decode;
use sha2::{Digest, Sha256};

/// UVH-26 ids 1..4 → the four-way passenger order used by `passenger::NAMES`.
const PASSENGER: [(usize, &str); 4] = [(1, "Hatchback"), (2, "Sedan"), (3, "SUV"), (4, "MUV")];
const CONTEXT: f32 = 0.15;

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let (ax1, ay1, bx1, by1) = (a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]);
    let iw = (ax1.min(bx1) - a[0].max(b[0])).max(0.0);
    let ih = (ay1.min(by1) - a[1].max(b[1])).max(0.0);
    let i = iw * ih;
    let u = a[2] * a[3] + b[2] * b[3] - i;
    if u <= 0.0 { 0.0 } else { i / u }
}

struct Img {
    file: String,
    boxes: Vec<(i64, usize, [f32; 4])>, // annotation id, category, xywh
}

fn load(path: &str) -> anyhow::Result<HashMap<i64, Img>> {
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut by: HashMap<i64, Img> = HashMap::new();
    for im in j["images"].as_array().into_iter().flatten() {
        let id = im["id"].as_i64().unwrap_or(-1);
        by.insert(id, Img { file: im["file_name"].as_str().unwrap_or("").to_string(), boxes: vec![] });
    }
    for a in j["annotations"].as_array().into_iter().flatten() {
        let img = a["image_id"].as_i64().unwrap_or(-1);
        let c = a["category_id"].as_u64().unwrap_or(0) as usize;
        let id = a["id"].as_i64().unwrap_or(-1);
        if let (Some(b), Some(e)) = (a["bbox"].as_array(), by.get_mut(&img)) {
            let f = |i: usize| b[i].as_f64().unwrap_or(0.0) as f32;
            e.boxes.push((id, c, [f(0), f(1), f(2), f(3)]));
        }
    }
    Ok(by)
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Resolve a COCO file name under the image root: the flat path, then the numbered sub-folders
/// UVH-26 uses.
fn resolve(root: &std::path::Path, file: &str) -> Option<std::path::PathBuf> {
    let direct = root.join(file);
    if direct.exists() {
        return Some(direct);
    }
    let base = std::path::Path::new(file).file_name()?;
    for sub in std::fs::read_dir(root).ok()? {
        let p = sub.ok()?.path().join(base);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() > 4, "export_reconciled <mv.json> <st.json> <image_root> <out_dir> [dev_share] [min_px]");
    let dev_share: f64 = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(0.1);
    let min_px: f32 = a.get(6).and_then(|v| v.parse().ok()).unwrap_or(48.0);
    let root = std::path::PathBuf::from(&a[3]);
    let out = std::path::PathBuf::from(&a[4]);
    let crops_dir = out.join("crops");
    std::fs::create_dir_all(&crops_dir)?;
    let (mv, st) = (load(&a[1])?, load(&a[2])?);
    let label_of: HashMap<usize, (usize, &str)> = PASSENGER.iter().enumerate().map(|(i, (c, n))| (*c, (i, *n))).collect();

    let mut manifest = std::fs::File::create(out.join("reconciled.jsonl"))?;
    let (mut images, mut agreed, mut disagreed, mut unmatched, mut small, mut written) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut per_class = [0usize; 4];
    let mut per_split = [0usize; 2];
    let mut ids: Vec<&i64> = mv.keys().collect();
    ids.sort();
    for img_id in ids {
        let (Some(m), Some(s)) = (mv.get(img_id), st.get(img_id)) else { continue };
        // reconciled boxes for this image: MV box, ST agrees on class
        let mut pairs: Vec<(f32, usize, usize)> = vec![];
        for (i, (_, ca, ba)) in m.boxes.iter().enumerate() {
            if !label_of.contains_key(ca) {
                continue;
            }
            for (j, (_, _, bb)) in s.boxes.iter().enumerate() {
                let v = iou(*ba, *bb);
                if v >= 0.7 {
                    pairs.push((v, i, j));
                }
            }
        }
        pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
        let (mut ua, mut ub) = (vec![false; m.boxes.len()], vec![false; s.boxes.len()]);
        let mut keep: Vec<usize> = vec![];
        for (_, i, j) in pairs {
            if ua[i] || ub[j] {
                continue;
            }
            ua[i] = true;
            ub[j] = true;
            if m.boxes[i].1 == s.boxes[j].1 {
                agreed += 1;
                keep.push(i);
            } else {
                disagreed += 1;
            }
        }
        unmatched += m.boxes.iter().enumerate().filter(|(i, (_, c, _))| label_of.contains_key(c) && !ua[*i]).count();
        if keep.is_empty() {
            continue;
        }
        let Some(path) = resolve(&root, &m.file) else {
            eprintln!("  missing image {}", m.file);
            continue;
        };
        let bytes = std::fs::read(&path)?;
        let source_sha = sha(&bytes);
        let img = match decode(&path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("  skip {}: {e:#}", path.display());
                continue;
            }
        };
        images += 1;
        // the split is a property of the source image, decided by its hash
        let split_idx = if (u64::from_str_radix(&source_sha[..8], 16)? as f64 / u32::MAX as f64) < dev_share { 1 } else { 0 };
        let split = if split_idx == 1 { "dev" } else { "train" };
        for i in keep {
            let (ann_id, cat, b) = m.boxes[i];
            let (label, name) = label_of[&cat];
            if b[2] < min_px || b[3] < min_px {
                small += 1;
                continue;
            }
            let (px, py) = (b[2] * CONTEXT, b[3] * CONTEXT);
            let x0 = (b[0] - px).max(0.0) as usize;
            let y0 = (b[1] - py).max(0.0) as usize;
            let x1 = ((b[0] + b[2] + px) as usize).min(img.w);
            let y1 = ((b[1] + b[3] + py) as usize).min(img.h);
            if x1 <= x0 + 8 || y1 <= y0 + 8 {
                continue;
            }
            let (cw, ch) = (x1 - x0, y1 - y0);
            let mut buf = Vec::with_capacity(cw * ch * 3);
            for y in y0..y1 {
                for x in x0..x1 {
                    let k = (y * img.w + x) * 3;
                    buf.extend_from_slice(&[(img.rgb[k] * 255.0) as u8, (img.rgb[k + 1] * 255.0) as u8, (img.rgb[k + 2] * 255.0) as u8]);
                }
            }
            let crop_path = crops_dir.join(format!("{img_id}_{ann_id}.jpg"));
            image::save_buffer(&crop_path, &buf, cw as u32, ch as u32, image::ColorType::Rgb8)?;
            let crop_sha = sha(&std::fs::read(&crop_path)?);
            let row = serde_json::json!({
                "annotation_id": ann_id, "image_id": img_id, "raw_category_id": cat,
                "label": label, "class_name": name, "dataset": "uvh-reconciled-train",
                "group": format!("uvh:train:{img_id}"), "split": split,
                "source": path.to_string_lossy(), "source_sha256": source_sha,
                "path": crop_path.to_string_lossy(), "crop_sha256": crop_sha,
                "box": b, "crop_xyxy": [x0, y0, x1, y1], "context": CONTEXT, "width": cw, "height": ch,
            });
            writeln!(manifest, "{row}")?;
            per_class[label] += 1;
            per_split[split_idx] += 1;
            written += 1;
        }
        if images % 500 == 0 {
            eprintln!("  {images} images, {written} crops");
        }
    }
    println!("images with reconciled passenger boxes: {images}");
    println!("passenger boxes: {agreed} agreed, {disagreed} disagreed (dropped), {unmatched} unmatched in ST (dropped), {small} agreed but under {min_px} px");
    println!("\n| class | crops |\n|---|---:|");
    for (i, (_, n)) in PASSENGER.iter().enumerate() {
        println!("| {n} | {} |", per_class[i]);
    }
    println!("\ntrain {} · dev {} (split by source-image hash, share {dev_share})", per_split[0], per_split[1]);
    println!("manifest: {}", out.join("reconciled.jsonl").display());
    Ok(())
}
