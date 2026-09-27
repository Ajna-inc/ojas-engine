//! Make a COCO file self-contained: copy every image it names (by absolute path) into
//! `<out_dir>/<sub>/` as `<camera>_<utc_ms>.jpg` (or the source's basename when the image has no
//! camera or time) and rewrite `file_name` to `<sub>/<name>`, relative to `out_dir`, so the set can
//! sit next to a training set and be read with one image root.
//!
//! `coco_localize <in.json> <out_dir> <sub> <out.json>`
use serde_json::{json, Value};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() == 5, "coco_localize <in.json> <out_dir> <sub> <out.json>");
    let mut v: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let dir = std::path::Path::new(&a[2]).join(&a[3]);
    std::fs::create_dir_all(&dir)?;
    let mut n = 0;
    for im in v["images"].as_array_mut().unwrap() {
        let src = im["file_name"].as_str().unwrap().to_string();
        let name = match (im["camera"].as_str(), im["utc_ms"].as_i64()) {
            (Some(c), Some(t)) => format!("{}_{t}.jpg", c.replace(['/', '@'], "-")),
            _ => std::path::Path::new(&src).file_name().unwrap().to_string_lossy().into_owned(),
        };
        let dst = dir.join(&name);
        if !dst.exists() {
            std::fs::copy(&src, &dst)?;
        }
        im["source"] = json!(src);
        im["file_name"] = json!(format!("{}/{name}", a[3]));
        n += 1;
    }
    std::fs::write(&a[4], serde_json::to_string(&v)?)?;
    println!("{n} images → {}, {}", dir.display(), a[4]);
    Ok(())
}
