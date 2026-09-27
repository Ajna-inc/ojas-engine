//! One-time preparation of the UVH-26 / BMD-45 training sets.
//!
//! The source disk reads ~41 of the 1080p PNGs a second, slower than the GPU trains, so every
//! image is decoded once, resized to `long_side` (default 1280 — 2× the 640 training resolution,
//! so multi-scale up to 800 still downsamples), and written as JPEG q92 into `out/images/<set>/`.
//! Each COCO json is rewritten beside it with the boxes rescaled and the categories moved onto the
//! UVH ids 1..14 (BMD-45 is 0-based and has no `Others`, so it takes `offset` 1). Originals are
//! hashed so a frame appearing in two sets — the UVH and BMD cameras overlap — is reported in
//! `out/leaks.tsv` before anything trains on it.
//!
//! Idempotent: images already converted (present in `out/hashes.tsv` with the JPEG on disk)
//! are skipped, so several jsons over one image root cost one pass.
//!
//! `prepare_dataset <out_dir> <long_side> <set>=<coco.json>:<image_root>:<category_offset> ...`
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

const NAMES: [&str; 14] = ["Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"];
const QUALITY: u8 = 92;
const DEFAULT_WORKERS: usize = 12;

#[derive(Clone)]
struct Record {
    set: String,
    file: String,
    sha: String,
    orig_w: u32,
    orig_h: u32,
    new_w: u32,
    new_h: u32,
}

/// `file_name` under `root`, or one level down (UVH's `data/000/`, BMD's `images_000/`).
fn resolve(root: &Path, index: &HashMap<String, PathBuf>, file: &str) -> Option<PathBuf> {
    let direct = root.join(file);
    if direct.exists() {
        return Some(direct);
    }
    index.get(Path::new(file).file_name()?.to_str()?).cloned()
}

fn index_root(root: &Path) -> anyhow::Result<HashMap<String, PathBuf>> {
    let mut files = HashMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d)? {
            let p = e?.path();
            if p.is_dir() {
                stack.push(p);
            } else if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
                files.insert(n.to_string(), p.clone());
            }
        }
    }
    Ok(files)
}

fn convert(src: &Path, dst: &Path, long_side: u32) -> anyhow::Result<(String, u32, u32, u32, u32)> {
    let bytes = std::fs::read(src)?;
    let sha = format!("{:x}", Sha256::digest(&bytes));
    let img = image::load_from_memory(&bytes)?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    let scale = long_side as f64 / w.max(h) as f64;
    let (nw, nh) = if scale < 1.0 { (((w as f64) * scale).round() as u32, ((h as f64) * scale).round() as u32) } else { (w, h) };
    let out = if (nw, nh) != (w, h) { image::imageops::resize(&img, nw, nh, image::imageops::FilterType::CatmullRom) } else { img };
    let tmp = dst.with_extension("jpg.part");
    {
        let f = std::fs::File::create(&tmp)?;
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(f), QUALITY);
        enc.encode(out.as_raw(), nw, nh, image::ExtendedColorType::Rgb8)?;
    }
    std::fs::rename(&tmp, dst)?;
    Ok((sha, w, h, nw, nh))
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 4, "prepare_dataset <out_dir> <long_side> <set>=<coco.json>:<image_root>:<offset> ...");
    let out = PathBuf::from(&a[1]);
    let long_side: u32 = a[2].parse()?;
    // an HDD serves fewer concurrent readers faster; PREPARE_WORKERS tunes it without a rebuild
    let workers: usize = std::env::var("PREPARE_WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_WORKERS);
    std::fs::create_dir_all(out.join("annotations"))?;

    // previous work, keyed by set/file
    let hashes_path = out.join("hashes.tsv");
    let mut done: HashMap<(String, String), Record> = HashMap::new();
    if let Ok(s) = std::fs::read_to_string(&hashes_path) {
        for l in s.lines().skip(1) {
            let f: Vec<&str> = l.split('\t').collect();
            if f.len() == 7 {
                let r = Record { set: f[0].into(), file: f[1].into(), sha: f[2].into(), orig_w: f[3].parse()?, orig_h: f[4].parse()?, new_w: f[5].parse()?, new_h: f[6].parse()? };
                done.insert((r.set.clone(), r.file.clone()), r);
            }
        }
    }
    let hashes = Arc::new(Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&hashes_path)?));
    if done.is_empty() {
        writeln!(hashes.lock().unwrap(), "set\tfile\tsha256\torig_w\torig_h\tnew_w\tnew_h")?;
    }
    let done = Arc::new(Mutex::new(done));

    for spec in &a[3..] {
        let (set, rest) = spec.split_once('=').ok_or_else(|| anyhow::anyhow!("bad spec {spec}"))?;
        let mut parts = rest.rsplitn(3, ':');
        let offset: i64 = parts.next().unwrap().parse()?;
        let root = PathBuf::from(parts.next().ok_or_else(|| anyhow::anyhow!("bad spec {spec}"))?);
        let json = PathBuf::from(parts.next().ok_or_else(|| anyhow::anyhow!("bad spec {spec}"))?);
        // the image set is the root's name, shared by every json over the same images
        let img_set = root.file_name().unwrap().to_string_lossy().to_string();
        let img_dir = out.join("images").join(&img_set);
        std::fs::create_dir_all(&img_dir)?;
        eprintln!("== {set}: {} → images/{img_set}", json.display());
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&json)?)?;
        let index = index_root(&root)?;
        let images = v["images"].as_array().cloned().unwrap_or_default();
        let jobs: Vec<(usize, String, PathBuf, PathBuf)> = images
            .iter()
            .enumerate()
            .filter_map(|(i, im)| {
                let file = im["file_name"].as_str()?.to_string();
                let stem = Path::new(&file).file_stem()?.to_string_lossy().to_string();
                let dst = img_dir.join(format!("{stem}.jpg"));
                let key = (img_set.clone(), stem.clone());
                if done.lock().unwrap().contains_key(&key) && dst.exists() {
                    return None;
                }
                let src = resolve(&root, &index, &file)?;
                Some((i, stem, src, dst))
            })
            .collect();
        eprintln!("   {} images, {} to convert", images.len(), jobs.len());
        let jobs = Arc::new(Mutex::new(jobs.into_iter()));
        let progress = Arc::new(Mutex::new((0usize, 0usize)));
        std::thread::scope(|s| {
            for _ in 0..workers {
                let (jobs, done, hashes, progress, img_set) = (jobs.clone(), done.clone(), hashes.clone(), progress.clone(), img_set.clone());
                s.spawn(move || loop {
                    let job = jobs.lock().unwrap().next();
                    let Some((_, stem, src, dst)) = job else { break };
                    match convert(&src, &dst, long_side) {
                        Ok((sha, w, h, nw, nh)) => {
                            let r = Record { set: img_set.clone(), file: stem.clone(), sha, orig_w: w, orig_h: h, new_w: nw, new_h: nh };
                            writeln!(hashes.lock().unwrap(), "{}\t{}\t{}\t{}\t{}\t{}\t{}", r.set, r.file, r.sha, r.orig_w, r.orig_h, r.new_w, r.new_h).ok();
                            done.lock().unwrap().insert((img_set.clone(), stem), r);
                            let mut p = progress.lock().unwrap();
                            p.0 += 1;
                            if p.0 % 500 == 0 {
                                eprintln!("   {} converted, {} failed", p.0, p.1);
                            }
                        }
                        Err(e) => {
                            eprintln!("   FAIL {}: {e:#}", src.display());
                            progress.lock().unwrap().1 += 1;
                        }
                    }
                });
            }
        });
        // rewrite the json onto the converted images and unified ids
        let done_now = done.lock().unwrap().clone();
        let mut new_images = vec![];
        let mut scale_of: HashMap<i64, (f64, f64)> = HashMap::new();
        let mut missing = 0usize;
        for im in &images {
            let file = im["file_name"].as_str().unwrap_or("");
            let stem = Path::new(file).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let Some(r) = done_now.get(&(img_set.clone(), stem.clone())) else { missing += 1; continue };
            let id = im["id"].as_i64().unwrap_or(-1);
            scale_of.insert(id, (r.new_w as f64 / r.orig_w as f64, r.new_h as f64 / r.orig_h as f64));
            new_images.push(serde_json::json!({"id": id, "file_name": format!("{img_set}/{stem}.jpg"), "width": r.new_w, "height": r.new_h, "source_sha256": r.sha}));
        }
        let mut new_anns = vec![];
        let mut per_class: BTreeMap<i64, usize> = BTreeMap::new();
        for an in v["annotations"].as_array().into_iter().flatten() {
            let Some(&(sx, sy)) = scale_of.get(&an["image_id"].as_i64().unwrap_or(-1)) else { continue };
            let b = an["bbox"].as_array().map(|b| b.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect::<Vec<_>>()).unwrap_or_default();
            if b.len() != 4 {
                continue;
            }
            let cat = an["category_id"].as_i64().unwrap_or(0) + offset;
            anyhow::ensure!((1..=14).contains(&cat), "{set}: category {} + offset {offset} is outside 1..14", an["category_id"]);
            let bbox = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy];
            *per_class.entry(cat).or_default() += 1;
            new_anns.push(serde_json::json!({"id": an["id"], "image_id": an["image_id"], "category_id": cat, "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0}));
        }
        let categories: Vec<_> = NAMES.iter().enumerate().map(|(i, n)| serde_json::json!({"id": i + 1, "name": n})).collect();
        let coco_out = serde_json::json!({"images": new_images, "annotations": new_anns, "categories": categories,
            "info": {"source": json.to_string_lossy(), "long_side": long_side, "category_offset": offset}});
        let out_json = out.join("annotations").join(format!("{set}.json"));
        std::fs::write(&out_json, serde_json::to_vec(&coco_out)?)?;
        println!("{set}: {} images ({missing} missing), {} boxes → {}", coco_out["images"].as_array().unwrap().len(), coco_out["annotations"].as_array().unwrap().len(), out_json.display());
        println!("   {}", per_class.iter().map(|(c, n)| format!("{} {n}", NAMES[(*c - 1) as usize])).collect::<Vec<_>>().join(", "));
    }

    // leak report: one original hash in more than one image set
    let done_now = done.lock().unwrap().clone();
    let mut by_sha: HashMap<&str, Vec<&Record>> = HashMap::new();
    for r in done_now.values() {
        by_sha.entry(&r.sha).or_default().push(r);
    }
    let mut leaks = std::fs::File::create(out.join("leaks.tsv"))?;
    writeln!(leaks, "sha256\tsets\tfiles")?;
    let mut pairs: BTreeMap<String, usize> = BTreeMap::new();
    let mut n = 0usize;
    for (sha, rs) in &by_sha {
        let sets: HashSet<&str> = rs.iter().map(|r| r.set.as_str()).collect();
        if sets.len() > 1 {
            n += 1;
            let mut s: Vec<&str> = sets.into_iter().collect();
            s.sort();
            *pairs.entry(s.join("+")).or_default() += 1;
            writeln!(leaks, "{sha}\t{}\t{}", s.join(","), rs.iter().map(|r| format!("{}/{}", r.set, r.file)).collect::<Vec<_>>().join(","))?;
        }
    }
    println!("leaks: {n} identical frames across image sets → {}", out.join("leaks.tsv").display());
    for (k, v) in pairs {
        println!("   {k}: {v}");
    }
    Ok(())
}
