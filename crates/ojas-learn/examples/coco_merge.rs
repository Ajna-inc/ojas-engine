//! Concatenate prepared COCO jsons into one training file. Image and annotation ids of the
//! second and later files are offset so they cannot collide (UVH ids are six digits, BMD ids
//! start at 0). Categories must already agree — `prepare_dataset` puts every set on the UVH
//! ids 1..14 — and the first file's category list is kept.
//!
//! An input written `a.json=prefix/` has `prefix/` put in front of its images' `file_name`, so
//! sets whose images live in different folders can share one image root without symlinks, which
//! the exFAT image volume does not support.
//!
//! `#k` after an input keeps every k-th of its images (with their boxes), and listing an input twice
//! repeats it — together they set each source's weight in a training mix.
//!
//! `coco_merge <out.json> <a.json[=prefix/][#k]> <b.json[=prefix/][#k]> [...]`
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 4, "coco_merge <out.json> <a.json> <b.json> ...");
    let mut images = vec![];
    let mut anns = vec![];
    let mut categories = serde_json::Value::Null;
    let mut sources = vec![];
    for (k, arg) in a[2..].iter().enumerate() {
        let (arg, every) = match arg.rsplit_once('#') {
            Some((a, k)) => (a, k.parse::<usize>()?),
            None => (arg.as_str(), 1),
        };
        let (path, prefix) = arg.split_once('=').unwrap_or((arg, ""));
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        if categories.is_null() {
            categories = v["categories"].clone();
        } else {
            anyhow::ensure!(v["categories"] == categories, "{path}: categories differ from the first input");
        }
        let offset = (k as i64) * 10_000_000;
        let (mut ni, mut na) = (0, 0);
        let mut kept = std::collections::HashSet::new();
        for (i, im) in v["images"].as_array().into_iter().flatten().enumerate() {
            if i % every != 0 {
                continue;
            }
            kept.insert(im["id"].as_i64().unwrap_or(0));
            let mut im = im.clone();
            im["id"] = serde_json::json!(im["id"].as_i64().unwrap_or(0) + offset);
            if !prefix.is_empty() {
                im["file_name"] = serde_json::json!(format!("{prefix}{}", im["file_name"].as_str().unwrap_or("")));
            }
            images.push(im);
            ni += 1;
        }
        for an in v["annotations"].as_array().into_iter().flatten() {
            if !kept.contains(&an["image_id"].as_i64().unwrap_or(0)) {
                continue;
            }
            let mut an = an.clone();
            an["id"] = serde_json::json!(an["id"].as_i64().unwrap_or(0) + offset);
            an["image_id"] = serde_json::json!(an["image_id"].as_i64().unwrap_or(0) + offset);
            anns.push(an);
            na += 1;
        }
        println!("{path}: {ni} images, {na} boxes (id offset {offset})");
        sources.push(serde_json::json!({"path": path, "id_offset": offset, "images": ni, "annotations": na}));
    }
    let out = serde_json::json!({"images": images, "annotations": anns, "categories": categories, "info": {"merged_from": sources}});
    std::fs::write(&a[1], serde_json::to_vec(&out)?)?;
    println!("→ {}: {} images, {} boxes", a[1], out["images"].as_array().unwrap().len(), out["annotations"].as_array().unwrap().len());
    Ok(())
}
