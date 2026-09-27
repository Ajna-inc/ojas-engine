//! Detection loading and two-model matching shared by `review_sheets` (which draws the disputes)
//! and `review_report` (which scores them), so both see the same agreements.
//!
//! Two labelling tasks, chosen by the env var `TASK` (default `vehicle`):
//!
//! - `vehicle`: the 14 UVH classes (detection `category_id` 1..14 = [`NAMES`]); verdicts are a
//!   class name, `not_vehicle`, `duplicate`, `unsure_car`, `unsure_vehicle` or `unsure`.
//! - `person`: one class, `category_id` 1 in the detection files (`person_survey` writes COCO
//!   `person` as 1), exported as category 15 `person` so it sits after the vehicle ids in one
//!   train.json; verdicts are `person`, `rider` (a person on a two-wheeler or bicycle — the same
//!   box in the export, kept apart in the verdict for the record), `not_person`, `duplicate` or
//!   `unsure`.
//!
//! Verdict files (`verdicts.tsv`, `verdicts_*.tsv`) are tab-separated; `#` starts a comment:
//!
//! - `<sheet>\t<cell>\t<verdict>[\t<note>]` — one dispute cell (or, on a sweep sheet, a swept frame:
//!   verdict `swept`);
//! - `miss\t<frame id>\t<x>\t<y>\t<w>\t<h>[\t<label>[\t<note>]]` — an object neither model boxed,
//!   found on a sweep sheet, in whole-frame pixels (x y = top-left, w h = size); `label` is
//!   `person` (default), `rider` or `unsure`. [`read_verdicts`] skips these lines and
//!   [`read_misses`] collects them.
#![allow(dead_code)] // each example uses a different part
use std::collections::HashMap;
use std::path::Path;

use image::{Rgb, RgbImage};
use serde_json::{json, Value};

pub const NAMES: [&str; 15] = ["?", "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"];

/// The UVH ids of the car subtypes an `unsure_car` verdict may be.
pub const CARS: [usize; 5] = [1, 2, 3, 4, 13];

/// The category id of `person` in the combined train.json: after the 14 UVH vehicle ids.
pub const PERSON_CAT: usize = 15;

/// One labelling task: its class list, verdict vocabulary and export ids.
pub struct Task {
    /// `vehicle` or `person`.
    pub name: &'static str,
    /// Class names by detection class index (index 0 unused).
    pub names: &'static [&'static str],
    /// The verdict for a box that is not an object of this task at all.
    pub negative: &'static str,
    /// Singular and plural nouns for the report's wording.
    pub noun: &'static str,
    pub plural: &'static str,
}

/// The task named by `TASK` (default `vehicle`).
pub fn task() -> anyhow::Result<Task> {
    match std::env::var("TASK").as_deref().unwrap_or("vehicle") {
        "vehicle" => Ok(Task { name: "vehicle", names: &NAMES, negative: "not_vehicle", noun: "vehicle", plural: "vehicles" }),
        "person" => Ok(Task { name: "person", names: &["?", "person"], negative: "not_person", noun: "person", plural: "people" }),
        other => anyhow::bail!("TASK={other:?}: expected vehicle or person"),
    }
}

impl Task {
    /// The detection class index a verdict names, if it names a class (`rider` is a `person`).
    pub fn class_id(&self, verdict: &str) -> Option<usize> {
        if self.name == "person" && verdict == "rider" {
            return Some(1);
        }
        self.names.iter().position(|&n| n == verdict).filter(|&i| i > 0)
    }

    /// The COCO category id a detection class index is exported as.
    pub fn cat_id(&self, class: usize) -> usize {
        if self.name == "person" {
            PERSON_CAT
        } else {
            class
        }
    }

    /// The `categories` list of a COCO file of this task.
    pub fn categories(&self) -> Vec<Value> {
        (1..self.names.len()).map(|i| json!({"id": self.cat_id(i), "name": self.names[i]})).collect()
    }

    /// The classes an unsure verdict may be; `None` if the verdict is not an unsure kind of this
    /// task.
    pub fn unsure_classes(&self, verdict: &str) -> Option<Vec<usize>> {
        match (self.name, verdict) {
            ("vehicle", "unsure_car") => Some(CARS.to_vec()),
            ("vehicle", "unsure" | "unsure_vehicle") | ("person", "unsure") => Some((1..self.names.len()).collect()),
            _ => None,
        }
    }
}

/// Whether a verdict means "not an object of the task", for either task, so priors from a person
/// round and a vehicle round can be read by one loader.
pub fn is_negative(verdict: &str) -> bool {
    verdict == "not_vehicle" || verdict == "not_person"
}

#[derive(Clone)]
pub struct Det {
    pub bbox: [f32; 4],
    pub class: usize,
    pub score: f32,
}

pub fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let (ax2, ay2, bx2, by2) = (a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]);
    let iw = (ax2.min(bx2) - a[0].max(b[0])).max(0.0);
    let ih = (ay2.min(by2) - a[1].max(b[1])).max(0.0);
    let i = iw * ih;
    i / (a[2] * a[3] + b[2] * b[3] - i).max(1e-6)
}

pub fn load_dets(path: &str, min_score: f32) -> anyhow::Result<HashMap<i64, Vec<Det>>> {
    load_dets_with(path, min_score, false)
}

/// The share of the smaller box's area that lies inside the larger one.
pub fn containment(a: [f32; 4], b: [f32; 4]) -> f32 {
    let iw = ((a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0])).max(0.0);
    let ih = ((a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1])).max(0.0);
    iw * ih / (a[2] * a[3]).min(b[2] * b[3]).max(1e-6)
}

/// [`load_dets`], with `contain` selecting a stricter per-model suppression: a box is also dropped
/// if it has IoU ≥ 0.5 with a kept box or the smaller of the two is ≥ 80 % inside the larger. That
/// is the rule `review_report` applies to the person gold, applied here before matching so a
/// second query on one person never becomes a dispute. Set per review dir (`DEDUPE=contain` on
/// `review_sheets`, recorded in the dir's `params.json`) so earlier rounds recompute as they
/// were cut.
pub fn load_dets_with(path: &str, min_score: f32, contain: bool) -> anyhow::Result<HashMap<i64, Vec<Det>>> {
    let v: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut out: HashMap<i64, Vec<Det>> = HashMap::new();
    for d in v.as_array().ok_or_else(|| anyhow::anyhow!("{path}: not a list"))? {
        let s = d["score"].as_f64().unwrap_or(0.0) as f32;
        if s < min_score {
            continue;
        }
        let b: Vec<f32> = d["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        out.entry(d["image_id"].as_i64().unwrap()).or_default().push(Det { bbox: [b[0], b[1], b[2], b[3]], class: d["category_id"].as_u64().unwrap() as usize, score: s });
    }
    // a DETR emits several class hypotheses for one object; keep each object's top-scoring one
    // (class-agnostic suppression at IoU 0.7), as a deployment would
    for v in out.values_mut() {
        v.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept: Vec<Det> = vec![];
        for d in v.drain(..) {
            if kept.iter().all(|k| iou(k.bbox, d.bbox) < if contain { 0.5 } else { 0.7 } && !(contain && containment(k.bbox, d.bbox) >= 0.8)) {
                kept.push(d);
            }
        }
        *v = kept;
    }
    Ok(out)
}

/// Greedy one-to-one matching of one frame's boxes (A in score order, each taking the unused B box
/// of highest IoU ≥ 0.5, the rule of `training/field/agreement.py`): each A box's B match, and
/// the indices of the B boxes left unmatched.
pub fn pair(la: &[Det], lb: &[Det]) -> (Vec<Option<usize>>, Vec<usize>) {
    let mut used_b = vec![false; lb.len()];
    let mut of_a = vec![];
    for x in la {
        let mut best = (0.5f32, None);
        for (j, y) in lb.iter().enumerate() {
            let v = iou(x.bbox, y.bbox);
            if !used_b[j] && v >= best.0 {
                best = (v, Some(j));
            }
        }
        if let Some(j) = best.1 {
            used_b[j] = true;
        }
        of_a.push(best.1);
    }
    let only_b = (0..lb.len()).filter(|&j| !used_b[j]).collect();
    (of_a, only_b)
}

/// The physical camera of a frame: the directory the file sits in (`day/<camera>@<segment>/<utc_ms>.jpg`)
/// without the time segment, so a static object carries across one camera's segments.
pub fn camera_of(file: &str) -> String {
    let dir = Path::new(file).parent().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    dir.split('@').next().unwrap_or_default().to_string()
}

/// The physical camera of a manifest entry: from its `source` (the original frame's path) when it
/// has one — a round over exported training images, whose `file` is `train_rN/images/...` —
/// otherwise from its `file`.
pub fn camera_of_entry(m: &Value) -> String {
    camera_of(m["source"].as_str().or(m["file"].as_str()).unwrap_or_default())
}

/// The dedupe mode a review dir was cut with: `params.json` `{"dedupe": "contain"}`, else plain.
pub fn dir_dedupe_contain(dir: &Path) -> anyhow::Result<bool> {
    match std::fs::read(dir.join("params.json")) {
        Ok(b) => Ok(serde_json::from_slice::<Value>(&b)?["dedupe"].as_str() == Some("contain")),
        Err(_) => Ok(false),
    }
}

/// Reviewed boxes of earlier rounds, per camera: (box, verdict was `not_vehicle` / `not_person`).
pub fn load_priors(dirs: &str) -> anyhow::Result<HashMap<String, Vec<([f32; 4], bool)>>> {
    let mut out: HashMap<String, Vec<([f32; 4], bool)>> = HashMap::new();
    for d in dirs.split(':').filter(|d| !d.is_empty()) {
        let d = Path::new(d);
        let manifest: Vec<Value> = serde_json::from_slice(&std::fs::read(d.join("manifest.json"))?)?;
        let verdicts = read_verdicts(d)?;
        for m in &manifest {
            let v = match (m["sheet"].as_u64(), m["cell"].as_u64()) {
                (Some(s), Some(c)) => verdicts.get(&(s, c)).cloned(),
                _ => m["auto"].as_str().map(String::from),
            };
            let Some(v) = v else { continue };
            if v == "duplicate" || v.starts_with("unsure") {
                continue;
            }
            let b: Vec<f32> = m["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            out.entry(camera_of_entry(m)).or_default().push(([b[0], b[1], b[2], b[3]], is_negative(&v)));
        }
    }
    Ok(out)
}

/// The verdict files of a review dir, in name order: `verdicts.tsv` and any `verdicts_*.tsv`
/// (parallel reviewers write their own part).
pub fn verdict_files(dir: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("verdicts") && n.ends_with(".tsv"))).collect();
    files.sort();
    Ok(files)
}

/// Whether a verdict-file line is a `miss` line (see the module doc): its first field is `miss`.
pub fn is_miss_line(line: &str) -> bool {
    line.split('\t').next().map(str::trim) == Some("miss")
}

/// A review's verdicts, `(sheet, cell) → verdict`, from `verdicts.tsv` and any `verdicts_*.tsv`
/// (parallel reviewers write their own part); files are read in name order and a later line for a
/// cell wins, so a correction can be appended anywhere. `miss` lines are left to [`read_misses`].
pub fn read_verdicts(dir: &Path) -> anyhow::Result<HashMap<(u64, u64), String>> {
    let mut out = HashMap::new();
    for f in verdict_files(dir)? {
        for line in std::fs::read_to_string(&f)?.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty() && !is_miss_line(l)) {
            let p: Vec<&str> = line.split('\t').collect();
            anyhow::ensure!(p.len() >= 3, "{}: bad verdict line: {line}", f.display());
            out.insert((p[0].trim().parse()?, p[1].trim().parse()?), p[2].trim().to_string());
        }
    }
    Ok(out)
}

/// An object neither model boxed, listed by a reviewer from a sweep sheet.
#[derive(Clone, Debug)]
pub struct Miss {
    pub image_id: i64,
    pub bbox: [f32; 4],
    /// `person`, `rider` or `unsure`.
    pub label: String,
    pub note: String,
}

/// The `miss` lines of a review dir's verdict files, in file then line order.
pub fn read_misses(dir: &Path) -> anyhow::Result<Vec<Miss>> {
    let mut out = vec![];
    for f in verdict_files(dir)? {
        for line in std::fs::read_to_string(&f)?.lines().filter(|l| is_miss_line(l)) {
            let p: Vec<&str> = line.split('\t').map(str::trim).collect();
            anyhow::ensure!(p.len() >= 6, "{}: bad miss line (need miss<TAB>frame<TAB>x<TAB>y<TAB>w<TAB>h): {line}", f.display());
            let n = |i: usize| p[i].parse::<f32>().map_err(|e| anyhow::anyhow!("{}: miss line field {i}: {e}: {line}", f.display()));
            let bbox = [n(2)?, n(3)?, n(4)?, n(5)?];
            anyhow::ensure!(bbox[2] > 0.0 && bbox[3] > 0.0, "{}: miss line with an empty box: {line}", f.display());
            let label = p.get(6).filter(|l| !l.is_empty()).unwrap_or(&"person").to_string();
            anyhow::ensure!(matches!(label.as_str(), "person" | "rider" | "unsure"), "{}: miss label {label:?} is not person, rider or unsure: {line}", f.display());
            out.push(Miss { image_id: p[1].parse()?, bbox, label, note: p.get(7).unwrap_or(&"").to_string() });
        }
    }
    Ok(out)
}

/// A 5×7 bitmap font (digits, capitals, a little punctuation; lower case is drawn as capitals),
/// for frame ids and grid labels on sheets without a font dependency.
fn glyph(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        ':' => [0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08],
        '/' => [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10],
        '@' => [0x0E, 0x11, 0x01, 0x0D, 0x15, 0x15, 0x0E],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        '#' => [0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        ' ' => [0; 7],
        _ => [0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F],
    }
}

/// Draw `text` with its top-left corner at (x, y), each font pixel `scale` px wide; returns the
/// width drawn. Glyphs are 5 px wide with a 1 px gap, 7 px tall.
pub fn draw_text(img: &mut RgbImage, x: i64, y: i64, scale: u32, color: Rgb<u8>, text: &str) -> i64 {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let s = scale as i64;
    let mut cx = x;
    for c in text.chars() {
        let g = glyph(c);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..5 {
                if bits & (0x10 >> col) != 0 {
                    for dy in 0..s {
                        for dx in 0..s {
                            let (px, py) = (cx + col as i64 * s + dx, y + row as i64 * s + dy);
                            if px >= 0 && py >= 0 && px < w && py < h {
                                img.put_pixel(px as u32, py as u32, color);
                            }
                        }
                    }
                }
            }
        }
        cx += 6 * s;
    }
    cx - x
}

/// Outline `b` (x, y, w, h in the image's pixels) `thick` px wide, clipped to the image.
pub fn draw_box(img: &mut RgbImage, b: [f32; 4], thick: u32, color: Rgb<u8>) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (x0, y0) = (b[0].round() as i64, b[1].round() as i64);
    let (x1, y1) = ((b[0] + b[2]).round() as i64, (b[1] + b[3]).round() as i64);
    let put = |img: &mut RgbImage, x: i64, y: i64| {
        if x >= 0 && y >= 0 && x < w && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
    };
    for t in 0..thick as i64 {
        for x in x0..=x1 {
            put(img, x, y0 + t);
            put(img, x, y1 - t);
        }
        for y in y0..=y1 {
            put(img, x0 + t, y);
            put(img, x1 - t, y);
        }
    }
}
