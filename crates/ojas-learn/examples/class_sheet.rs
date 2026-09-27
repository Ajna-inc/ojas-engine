//! Inspect what a dataset calls one category: `n` boxes of it, sampled evenly over the annotation
//! file (deterministic), cropped from the original image with 30 % context, outlined and tiled 6
//! per row at 200 px. Checks a labelling convention against the data — for instance whether
//! UVH-26 files cargo autos under Three-wheeler or LCV — before adding labels that contradict it.
//!
//! `class_sheet <ann.json> <image_root> <category_id> <out.jpg> [n 36] [min_w 40]`
use image::{imageops, Rgb, RgbImage};
use serde_json::Value;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 5, "class_sheet <ann.json> <image_root> <category_id> <out.jpg> [n 36] [min_w 40]");
    let cat: i64 = a[3].parse()?;
    let n: usize = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(36);
    let min_w: f64 = a.get(6).map(|v| v.parse().unwrap()).unwrap_or(40.0);
    let v: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let files: std::collections::HashMap<i64, String> = v["images"].as_array().unwrap().iter().map(|im| (im["id"].as_i64().unwrap(), im["file_name"].as_str().unwrap().to_string())).collect();
    let boxes: Vec<&Value> = v["annotations"].as_array().unwrap().iter().filter(|x| x["category_id"].as_i64() == Some(cat) && x["iscrowd"].as_i64().unwrap_or(0) == 0 && x["bbox"][2].as_f64().unwrap_or(0.0) >= min_w).collect();
    anyhow::ensure!(!boxes.is_empty(), "no boxes of category {cat} at least {min_w} px wide");
    let pick: Vec<&Value> = (0..n.min(boxes.len())).map(|i| boxes[i * boxes.len() / n.min(boxes.len())]).collect();
    let (t, cols) = (200u32, 6u32);
    let rows = (pick.len() as u32).div_ceil(cols);
    let mut sheet = RgbImage::from_pixel(cols * t, rows * t, Rgb([20, 20, 20]));
    for (i, x) in pick.iter().enumerate() {
        let path = std::path::Path::new(&a[2]).join(&files[&x["image_id"].as_i64().unwrap()]);
        let img = image::open(&path)?.to_rgb8();
        let b: Vec<f32> = x["bbox"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let (px, py) = (b[2] * 0.3, b[3] * 0.3);
        let x0 = (b[0] - px).max(0.0) as u32;
        let y0 = (b[1] - py).max(0.0) as u32;
        let x1 = ((b[0] + b[2] + px) as u32).min(img.width());
        let y1 = ((b[1] + b[3] + py) as u32).min(img.height());
        let mut c = imageops::crop_imm(&img, x0, y0, (x1 - x0).max(1), (y1 - y0).max(1)).to_image();
        let (bx0, by0) = ((b[0] as u32).saturating_sub(x0), (b[1] as u32).saturating_sub(y0));
        let (bx1, by1) = (((b[0] + b[2]) as u32).saturating_sub(x0).min(c.width() - 1), ((b[1] + b[3]) as u32).saturating_sub(y0).min(c.height() - 1));
        for xx in bx0..=bx1 {
            c.put_pixel(xx, by0.min(c.height() - 1), Rgb([255, 220, 0]));
            c.put_pixel(xx, by1, Rgb([255, 220, 0]));
        }
        for yy in by0..=by1 {
            c.put_pixel(bx0.min(c.width() - 1), yy, Rgb([255, 220, 0]));
            c.put_pixel(bx1, yy, Rgb([255, 220, 0]));
        }
        let s = (t as f32 - 4.0) / c.width().max(c.height()) as f32;
        let r = imageops::resize(&c, ((c.width() as f32 * s) as u32).max(1), ((c.height() as f32 * s) as u32).max(1), imageops::FilterType::Triangle);
        imageops::overlay(&mut sheet, &r, ((i as u32 % cols) * t + 2) as i64, ((i as u32 / cols) * t + 2) as i64);
    }
    sheet.save(&a[4])?;
    println!("{} of {} boxes of category {cat} → {}", pick.len(), boxes.len(), a[4]);
    Ok(())
}
