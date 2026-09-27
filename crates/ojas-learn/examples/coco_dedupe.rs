//! Drop from one prepared COCO json every image whose source frame also appears in any of
//! the other prepared jsons, matched by `source_sha256` (written by `prepare_dataset`).
//!
//! UVH-26 and BMD-45 were cut from the same camera network and share 10,841 frames: 1,698 of
//! UVH's validation images sit in BMD's training set, and 1,929 of BMD's validation images sit in
//! UVH's training set. A cross-dataset number not computed on de-leaked splits is partly
//! memorisation.
//!
//! `coco_dedupe <in.json> <out.json> <exclude.json> [<exclude.json> ...]`
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 4, "coco_dedupe <in.json> <out.json> <exclude.json> ...");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let mut banned = std::collections::HashSet::new();
    for path in &a[3..] {
        let e: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        for im in e["images"].as_array().into_iter().flatten() {
            if let Some(s) = im["source_sha256"].as_str() {
                banned.insert(s.to_string());
            }
        }
    }
    let (ni, na) = (v["images"].as_array().map_or(0, Vec::len), v["annotations"].as_array().map_or(0, Vec::len));
    let mut dropped_ids = std::collections::HashSet::new();
    if let Some(ims) = v["images"].as_array_mut() {
        ims.retain(|im| {
            let leak = im["source_sha256"].as_str().is_some_and(|s| banned.contains(s));
            if leak {
                if let Some(id) = im["id"].as_i64() {
                    dropped_ids.insert(id);
                }
            }
            !leak
        });
    }
    if let Some(anns) = v["annotations"].as_array_mut() {
        anns.retain(|an| !an["image_id"].as_i64().is_some_and(|id| dropped_ids.contains(&id)));
    }
    v["info"]["deduped_against"] = serde_json::Value::Array(a[3..].iter().map(|p| serde_json::Value::String(p.clone())).collect());
    std::fs::write(&a[2], serde_json::to_vec(&v)?)?;
    println!(
        "{ni} → {} images ({} shared frames dropped), {na} → {} boxes → {}",
        v["images"].as_array().map_or(0, Vec::len),
        dropped_ids.len(),
        v["annotations"].as_array().map_or(0, Vec::len),
        a[2]
    );
    Ok(())
}
