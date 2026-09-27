//! Contact sheets of the boxes two detectors disagree on, for review by eye.
//!
//! Each model's detections are de-duplicated (class-agnostic suppression at IoU 0.7, top score
//! kept), then the two sets are matched greedily by IoU (highest score first, IoU ≥ 0.5 — the rule
//! of `training/field/agreement.py`). Every non-agreement — same box with different classes, or a
//! box only one model found — is cropped with context, outlined and tiled row-major onto numbered
//! sheets. `manifest.json` maps sheet/cell → frame, box and each model's class and score.
//!
//! With `PRIORS=dir1:dir2` (earlier review dirs), a dispute overlapping (IoU ≥ 0.7) at least
//! `PRIOR_MIN` reviewed boxes of the same camera, all `not_vehicle` and none a vehicle, is resolved
//! as `not_vehicle` with no cell (`"auto": "not_vehicle"` in the manifest). That clears the static
//! false positives — a kiosk, a bench, a timestamp overlay — recurring at one spot in every frame.
//!
//! `THIRD=<dets.json>` (with `THIRD_MIN`, default 0.5): a third detector's box at the same place
//! (IoU ≥ 0.5) whose class is A's or B's settles that dispute as that class — manifest `"auto"`,
//! no cell. For fixing a labelled set: A = the labels as detections (score 1), B = one model, the
//! third = another, so the cells left are what the two models and the labels cannot settle.
//!
//! `TASK=person` (see `review_common`) names the one class `person`, and the auto verdict is
//! `not_person`. A frame list whose images carry `source` (the original frame of an exported
//! training image) gets it copied into the person manifest, and the camera of a prior is read
//! from it, so priors from a round over original frames apply to a round over exported ones.
//!
//! Env: `DEDUPE=contain` — the stricter suppression of `review_common::load_dets_with` (IoU ≥ 0.5
//! or ≥ 80 % containment), recorded in `out_dir/params.json` so `review_report` recomputes the same
//! disputes; `PRIOR_MIN` — default 2; `DRY=1` — print counts, write nothing.
//!
//! Sweep mode (`--sweep`) covers objects neither model boxed, which never reach a dispute sheet.
//! It tiles whole frames `cols × rows` per sheet (default 2 × 2), each scaled to `cell_w` px wide
//! (default 960), with every box of either model at score ≥ `score`: green where both agree (the
//! mean box), cyan for A only, magenta for B only. A faint grid every 100 frame pixels is labelled
//! with the frame coordinate every 200 px along the top and left edges, under the banner
//! `FRAME <id>  SHEET <s> CELL <c>` and a line of set, camera and box counts. Reviewers add
//! `miss <frame id> <x> <y> <w> <h> [label] [note]` lines (whole-frame pixels off the grid; label
//! `person` by default, `rider` or `unsure`) and mark each frame `<sheet> <cell> swept`. A sweep
//! entry in the manifest carries `"kind": "sweep"`, the frame, its scale, its cell origin in the
//! sheet and the box counts.
//!
//! `review_sheets <frames.json> <A.dets.json> <B.dets.json> <out_dir> [score 0.3] [per_sheet 20]`
//! `review_sheets --sweep <frames.json> <A.dets.json> <B.dets.json> <out_dir> [score 0.3] [cols 2] [rows 2] [cell_w 960]`
use std::collections::HashMap;
use std::path::Path;

use image::{imageops, Rgb, RgbImage};
use serde_json::{json, Value};

#[path = "review_common/mod.rs"]
mod review_common;
use review_common::{camera_of_entry, draw_box, draw_text, iou, load_dets, load_dets_with, load_priors, pair, task};

const CELL: u32 = 256;
const COLS: u32 = 5;
const BANNER: u32 = 48;
const GREEN: Rgb<u8> = Rgb([0, 230, 60]);
const CYAN: Rgb<u8> = Rgb([0, 220, 255]);
const MAGENTA: Rgb<u8> = Rgb([255, 0, 255]);

/// The box with 40 % context each side (at least 24 px), cropped from the frame, scaled to fit a
/// cell, and outlined.
fn crop_cell(img: &RgbImage, b: [f32; 4], color: Rgb<u8>) -> RgbImage {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let pad_x = (b[2] * 0.4).max(24.0);
    let pad_y = (b[3] * 0.4).max(24.0);
    let x0 = (b[0] - pad_x).max(0.0);
    let y0 = (b[1] - pad_y).max(0.0);
    let x1 = (b[0] + b[2] + pad_x).min(w);
    let y1 = (b[1] + b[3] + pad_y).min(h);
    let (cw, ch) = ((x1 - x0).max(1.0), (y1 - y0).max(1.0));
    let crop = imageops::crop_imm(img, x0 as u32, y0 as u32, cw as u32, ch as u32).to_image();
    let s = (CELL as f32 - 8.0) / cw.max(ch);
    let (nw, nh) = (((cw * s) as u32).max(1), ((ch * s) as u32).max(1));
    let mut small = imageops::resize(&crop, nw, nh, imageops::FilterType::CatmullRom);
    let (bx0, by0) = (((b[0] - x0) * s) as i64, ((b[1] - y0) * s) as i64);
    let (bx1, by1) = (((b[0] + b[2] - x0) * s) as i64, ((b[1] + b[3] - y0) * s) as i64);
    for t in 0..2i64 {
        for x in bx0..=bx1 {
            for y in [by0 + t, by1 - t] {
                if x >= 0 && y >= 0 && (x as u32) < nw && (y as u32) < nh {
                    small.put_pixel(x as u32, y as u32, color);
                }
            }
        }
        for y in by0..=by1 {
            for x in [bx0 + t, bx1 - t] {
                if x >= 0 && y >= 0 && (x as u32) < nw && (y as u32) < nh {
                    small.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
    let mut cell = RgbImage::from_pixel(CELL, CELL, Rgb([30, 30, 30]));
    imageops::overlay(&mut cell, &small, ((CELL - nw) / 2) as i64, ((CELL - nh) / 2) as i64);
    cell
}

/// One whole frame for a sweep cell: scaled to `cell_w` wide, with the grid, boxes and banner.
fn sweep_cell(img: &RgbImage, cell_w: u32, agreed: &[[f32; 4]], only_a: &[[f32; 4]], only_b: &[[f32; 4]], banner: [&str; 2]) -> (RgbImage, f32) {
    let s = cell_w as f32 / img.width() as f32;
    let cell_h = ((img.height() as f32 * s).round() as u32).max(1);
    let mut small = imageops::resize(img, cell_w, cell_h, imageops::FilterType::CatmullRom);
    // grid every 100 frame px; the frame coordinate every 200 px along the edges
    for k in 1.. {
        let v = (k * 100) as f32 * s;
        if v >= cell_w as f32 && v >= cell_h as f32 {
            break;
        }
        let blend = |p: &mut Rgb<u8>| {
            for c in p.0.iter_mut() {
                *c = (*c as u32 * 2 / 3 + 85) as u8;
            }
        };
        if (v as u32) < cell_w {
            for y in 0..cell_h {
                blend(small.get_pixel_mut(v as u32, y));
            }
        }
        if (v as u32) < cell_h {
            for x in 0..cell_w {
                blend(small.get_pixel_mut(x, v as u32));
            }
        }
        if k % 2 == 0 {
            let label = (k * 100).to_string();
            if (v as u32) < cell_w {
                draw_text(&mut small, v as i64 + 2, 1, 2, Rgb([255, 255, 255]), &label);
            }
            if (v as u32) < cell_h {
                draw_text(&mut small, 1, v as i64 + 2, 2, Rgb([255, 255, 255]), &label);
            }
        }
    }
    let scaled = |b: &[f32; 4]| [b[0] * s, b[1] * s, b[2] * s, b[3] * s];
    for b in agreed {
        draw_box(&mut small, scaled(b), 2, GREEN);
    }
    for b in only_a {
        draw_box(&mut small, scaled(b), 2, CYAN);
    }
    for b in only_b {
        draw_box(&mut small, scaled(b), 2, MAGENTA);
    }
    let mut cell = RgbImage::from_pixel(cell_w, cell_h + BANNER, Rgb([0, 0, 0]));
    draw_text(&mut cell, 4, 4, 3, Rgb([255, 255, 0]), banner[0]);
    draw_text(&mut cell, 4, 29, 2, Rgb([220, 220, 220]), banner[1]);
    imageops::overlay(&mut cell, &small, 0, BANNER as i64);
    (cell, s)
}

/// Sweep sheets: whole frames with both models' boxes, for listing what neither boxed.
fn sweep(a: &[String]) -> anyhow::Result<()> {
    anyhow::ensure!(a.len() >= 5, "review_sheets --sweep <frames.json> <A.dets.json> <B.dets.json> <out_dir> [score 0.3] [cols 2] [rows 2] [cell_w 960]");
    let min_score: f32 = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(0.3);
    let cols: usize = a.get(6).map(|v| v.parse().unwrap()).unwrap_or(2);
    let rows: usize = a.get(7).map(|v| v.parse().unwrap()).unwrap_or(2);
    let cell_w: u32 = a.get(8).map(|v| v.parse().unwrap()).unwrap_or(960);
    let out = Path::new(&a[4]);
    std::fs::create_dir_all(out)?;
    let frames: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let (da, db) = (load_dets(&a[2], min_score)?, load_dets(&a[3], min_score)?);
    let per_sheet = cols * rows;
    let images = frames["images"].as_array().unwrap();
    let mut manifest = vec![];
    let (mut n_agree, mut n_a, mut n_b) = (0usize, 0usize, 0usize);
    for (sh, chunk) in images.chunks(per_sheet).enumerate() {
        let mut sheet: Option<RgbImage> = None;
        let mut cell_h = 0u32;
        for (c, im) in chunk.iter().enumerate() {
            let id = im["id"].as_i64().unwrap();
            let file = im["file_name"].as_str().unwrap();
            let empty = vec![];
            let (la, lb) = (da.get(&id).unwrap_or(&empty), db.get(&id).unwrap_or(&empty));
            let (of_a, ob) = pair(la, lb);
            let (mut agreed, mut only_a) = (vec![], vec![]);
            for (x, m) in la.iter().zip(of_a) {
                match m {
                    Some(j) => {
                        let y = lb[j].bbox;
                        agreed.push([(x.bbox[0] + y[0]) / 2.0, (x.bbox[1] + y[1]) / 2.0, (x.bbox[2] + y[2]) / 2.0, (x.bbox[3] + y[3]) / 2.0]);
                    }
                    None => only_a.push(x.bbox),
                }
            }
            let only_b: Vec<[f32; 4]> = ob.iter().map(|&j| lb[j].bbox).collect();
            n_agree += agreed.len();
            n_a += only_a.len();
            n_b += only_b.len();
            let set = im["set"].as_str().unwrap_or("");
            let cam = im["camera"].as_str().unwrap_or("?");
            let banner = [format!("FRAME {id}   SHEET {sh} CELL {c}"), format!("{set} {cam}   green both {}  cyan A only {}  magenta B only {}", agreed.len(), only_a.len(), only_b.len())];
            let img = image::open(file)?.to_rgb8();
            let (cell, scale) = sweep_cell(&img, cell_w, &agreed, &only_a, &only_b, [&banner[0], &banner[1]]);
            cell_h = cell.height();
            let sheet = sheet.get_or_insert_with(|| RgbImage::from_pixel(cols as u32 * (cell_w + 2), rows as u32 * (cell_h + 2), Rgb([255, 255, 255])));
            let (ox, oy) = (((c % cols) as u32 * (cell_w + 2)) as i64, ((c / cols) as u32 * (cell_h + 2)) as i64);
            imageops::overlay(sheet, &cell, ox, oy);
            manifest.push(json!({"sheet": sh, "cell": c, "kind": "sweep", "image_id": id, "file": file, "set": set, "camera": cam, "scale": scale, "origin": [ox, oy + BANNER as i64], "agreed": agreed.len(), "only_a": only_a.len(), "only_b": only_b.len()}));
        }
        // empty cells of a partial last sheet stay white
        let sheet = sheet.unwrap();
        let _ = cell_h;
        sheet.save(out.join(format!("sheet_{sh:04}.jpg")))?;
    }
    std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
    println!("sweep: {} frames at score ≥ {min_score}: {n_agree} agreed boxes, {n_a} only A, {n_b} only B; {} sheets of {cols}×{rows} at {cell_w} px wide → {}", images.len(), images.len().div_ceil(per_sheet), out.display());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut a: Vec<String> = std::env::args().collect();
    let task = task()?;
    if a.get(1).is_some_and(|f| f == "--sweep") {
        a.remove(1);
        return sweep(&a);
    }
    anyhow::ensure!(a.len() >= 5, "review_sheets [--sweep] <frames.json> <A.dets.json> <B.dets.json> <out_dir> [score] [per_sheet]");
    let min_score: f32 = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(0.3);
    let per_sheet: usize = a.get(6).map(|v| v.parse().unwrap()).unwrap_or(20);
    let out = Path::new(&a[4]);
    let dry = std::env::var("DRY").is_ok_and(|v| v == "1");
    let contain = std::env::var("DEDUPE").is_ok_and(|v| v == "contain");
    let prior_min: usize = std::env::var("PRIOR_MIN").ok().map(|v| v.parse()).transpose()?.unwrap_or(2);
    if !dry {
        std::fs::create_dir_all(out)?;
        if contain {
            std::fs::write(out.join("params.json"), serde_json::to_string_pretty(&json!({"dedupe": "contain", "score": min_score, "prior_min": prior_min, "priors": std::env::var("PRIORS").unwrap_or_default()}))?)?;
        }
    }
    let frames: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let (da, db) = (load_dets_with(&a[2], min_score, contain)?, load_dets_with(&a[3], min_score, contain)?);
    // what the per-model suppression removed from A: for a labelled set, the double labels
    // (the same box under two classes, or twice)
    let raw_a: Value = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    let raw_a_n = raw_a.as_array().map_or(0, |v| v.iter().filter(|d| d["score"].as_f64().unwrap_or(0.0) as f32 >= min_score).count());
    let kept_a_n: usize = da.values().map(Vec::len).sum();
    if raw_a_n != kept_a_n {
        println!("A: {} of {raw_a_n} boxes dropped as overlapping duplicates (IoU ≥ {}) before matching", raw_a_n - kept_a_n, if contain { 0.5 } else { 0.7 });
    }
    let names = task.names;

    let mut items: Vec<Value> = vec![];
    let (mut agree, mut differ, mut only_a, mut only_b) = (0, 0, 0, 0);
    for im in frames["images"].as_array().unwrap() {
        let id = im["id"].as_i64().unwrap();
        let file = im["file_name"].as_str().unwrap();
        let empty = vec![];
        let (la, lb) = (da.get(&id).unwrap_or(&empty), db.get(&id).unwrap_or(&empty));
        let (of_a, ob) = pair(la, lb);
        let first = items.len();
        for (x, m) in la.iter().zip(of_a) {
            match m {
                Some(j) if x.class == lb[j].class => agree += 1,
                Some(j) => {
                    let y = &lb[j];
                    differ += 1;
                    items.push(json!({"image_id": id, "file": file, "bbox": x.bbox, "kind": "differ", "a": names[x.class], "a_score": x.score, "b": names[y.class], "b_score": y.score}));
                }
                None => {
                    only_a += 1;
                    items.push(json!({"image_id": id, "file": file, "bbox": x.bbox, "kind": "only_a", "a": names[x.class], "a_score": x.score, "b": null, "b_score": null}));
                }
            }
        }
        for j in ob {
            let y = &lb[j];
            only_b += 1;
            items.push(json!({"image_id": id, "file": file, "bbox": y.bbox, "kind": "only_b", "a": null, "a_score": null, "b": names[y.class], "b_score": y.score}));
        }
        if let (Some(src), "person") = (im["source"].as_str(), task.name) {
            for it in &mut items[first..] {
                it["source"] = json!(src);
            }
        }
    }
    let mut auto = vec![];
    // a third detector settles a dispute when it confidently sides with one side (THIRD=<dets.json>,
    // THIRD_MIN score, default 0.5): same box (IoU ≥ 0.5) and the class of A or of B → that class,
    // no cell. Made for fixing a labelled set: A = the labels (score 1), B and the third = two models.
    let mut third_n = 0usize;
    if let Ok(path) = std::env::var("THIRD") {
        let tmin: f32 = std::env::var("THIRD_MIN").ok().map(|v| v.parse()).transpose()?.unwrap_or(0.5);
        let dt = load_dets_with(&path, tmin, contain)?;
        let empty = vec![];
        items.retain(|it| {
            let b: Vec<f32> = it["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let b = [b[0], b[1], b[2], b[3]];
            let best = dt.get(&it["image_id"].as_i64().unwrap()).unwrap_or(&empty).iter().filter(|d| iou(d.bbox, b) >= 0.5).max_by(|x, y| iou(x.bbox, b).total_cmp(&iou(y.bbox, b)));
            let Some(d) = best else { return true };
            let third = names[d.class];
            let settled = [it["a"].as_str(), it["b"].as_str()].into_iter().flatten().any(|c| c == third);
            if settled {
                let mut m = it.clone();
                m["auto"] = json!(third);
                m["auto_by"] = json!("third");
                auto.push(m);
                third_n += 1;
            }
            !settled
        });
        println!("{third_n} disputes settled by the third detector ({path}, score ≥ {tmin})");
    }
    // static false positives seen before on this camera need no cell
    let priors = load_priors(&std::env::var("PRIORS").unwrap_or_default())?;
    if !priors.is_empty() {
        let empty = vec![];
        items.retain(|it| {
            let b: Vec<f32> = it["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let same: Vec<bool> = priors.get(&camera_of_entry(it)).unwrap_or(&empty).iter().filter(|(pb, _)| iou(*pb, [b[0], b[1], b[2], b[3]]) >= 0.7).map(|(_, nv)| *nv).collect();
            if same.len() >= prior_min && same.iter().all(|&nv| nv) {
                let mut m = it.clone();
                m["auto"] = json!(task.negative);
                auto.push(m);
                false
            } else {
                true
            }
        });
    }
    println!("frames {}: agree {agree}, class differs {differ}, only A {only_a}, only B {only_b} → {} boxes to review ({} resolved as static false positives from earlier reviews)", frames["images"].as_array().unwrap().len(), items.len(), auto.len());
    if dry {
        println!("dry run ({}dedupe {}, priors at ≥ {prior_min}): {} sheets of {per_sheet} would be drawn", if contain { "" } else { "plain " }, if contain { "contain" } else { "IoU 0.7" }, items.len().div_ceil(per_sheet));
        return Ok(());
    }

    // sheets: yellow outline = class dispute, cyan = only A, magenta = only B
    let mut cache: HashMap<String, RgbImage> = HashMap::new();
    let mut manifest = vec![];
    for (s, chunk) in items.chunks(per_sheet).enumerate() {
        let rows = (chunk.len() as u32).div_ceil(COLS);
        let mut sheet = RgbImage::from_pixel(COLS * CELL, rows * CELL, Rgb([0, 0, 0]));
        for (c, it) in chunk.iter().enumerate() {
            let file = it["file"].as_str().unwrap().to_string();
            if !cache.contains_key(&file) {
                if cache.len() > 32 {
                    cache.clear();
                }
                cache.insert(file.clone(), image::open(&file)?.to_rgb8());
            }
            let b: Vec<f32> = it["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let color = match it["kind"].as_str().unwrap() {
                "differ" => Rgb([255, 220, 0]),
                "only_a" => Rgb([0, 220, 255]),
                _ => Rgb([255, 0, 255]),
            };
            let cell = crop_cell(&cache[&file], [b[0], b[1], b[2], b[3]], color);
            imageops::overlay(&mut sheet, &cell, ((c as u32 % COLS) * CELL) as i64, ((c as u32 / COLS) * CELL) as i64);
            let mut m = it.clone();
            m["sheet"] = json!(s);
            m["cell"] = json!(c);
            manifest.push(m);
        }
        sheet.save(out.join(format!("sheet_{s:04}.jpg")))?;
    }
    manifest.extend(auto);
    std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
    println!("{} sheets of {per_sheet} → {}", items.len().div_ceil(per_sheet), out.display());
    Ok(())
}
