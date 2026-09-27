//! Restrict a COCO json to the images another COCO json covers, so two labellings of one
//! set can be compared on identical frames. UVH-26's ST train json covers 17,387 of the
//! 21,349 MV train images; an MV-vs-ST arm pair must train on the same 17,387.
//!
//! `coco_subset <in.json> <images_from.json> <out.json>`
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() == 4, "coco_subset <in.json> <images_from.json> <out.json>");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let from: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    let keep: std::collections::HashSet<i64> = from["images"].as_array().into_iter().flatten().filter_map(|im| im["id"].as_i64()).collect();
    let (ni, na) = (v["images"].as_array().map_or(0, Vec::len), v["annotations"].as_array().map_or(0, Vec::len));
    if let Some(ims) = v["images"].as_array_mut() {
        ims.retain(|im| im["id"].as_i64().is_some_and(|id| keep.contains(&id)));
    }
    if let Some(anns) = v["annotations"].as_array_mut() {
        anns.retain(|an| an["image_id"].as_i64().is_some_and(|id| keep.contains(&id)));
    }
    v["info"]["images_from"] = serde_json::Value::String(a[2].clone());
    std::fs::write(&a[3], serde_json::to_vec(&v)?)?;
    println!("{ni} → {} images, {na} → {} boxes → {}", v["images"].as_array().map_or(0, Vec::len), v["annotations"].as_array().map_or(0, Vec::len), a[3]);
    Ok(())
}
