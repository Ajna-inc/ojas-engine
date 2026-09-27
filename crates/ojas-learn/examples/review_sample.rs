//! Audit a review: `n` randomly chosen cells that were judged by eye (their note does not start
//! with `auto:`), re-cropped from the original frames and tiled on one sheet with the recorded
//! verdict printed per tile, so a second look can count how many verdicts it would change.
//!
//! `review_sample <review_dir> <out.jpg> [n 40] [seed 1]`
use image::{imageops, Rgb, RgbImage};
use serde_json::Value;

#[path = "review_common/mod.rs"]
mod review_common;
use review_common::is_miss_line;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 3, "review_sample <review_dir> <out.jpg> [n 40] [seed 1]");
    let dir = std::path::Path::new(&a[1]);
    let n: usize = a.get(3).map(|v| v.parse().unwrap()).unwrap_or(40);
    let mut seed: u64 = a.get(4).map(|v| v.parse().unwrap()).unwrap_or(1);
    let manifest: Vec<Value> = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    // final (verdict, note) per cell, later files and lines win
    let mut files: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("verdicts") && n.ends_with(".tsv"))).collect();
    files.sort();
    let mut fin = std::collections::HashMap::new();
    for f in files {
        for l in std::fs::read_to_string(&f)?.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty() && !is_miss_line(l)) {
            let p: Vec<&str> = l.split('\t').collect();
            fin.insert((p[0].trim().parse::<u64>()?, p[1].trim().parse::<u64>()?), (p[2].trim().to_string(), p.get(3).unwrap_or(&"").to_string()));
        }
    }
    let judged: Vec<(&Value, &(String, String))> = manifest
        .iter()
        .filter_map(|m| Some((m, fin.get(&(m["sheet"].as_u64()?, m["cell"].as_u64()?))?)))
        .filter(|(_, (_, note))| !note.starts_with("auto:") && !note.contains("static rule") && !note.starts_with("relabel"))
        .collect();
    anyhow::ensure!(!judged.is_empty(), "no by-eye verdicts in {}", dir.display());
    // xorshift pick without replacement
    let mut idx: Vec<usize> = (0..judged.len()).collect();
    let k = n.min(idx.len());
    for i in 0..k {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let j = i + (seed as usize) % (idx.len() - i);
        idx.swap(i, j);
    }
    let (t, cols) = (256u32, 5u32);
    let mut sheet = RgbImage::from_pixel(cols * t, (k as u32).div_ceil(cols) * t, Rgb([20, 20, 20]));
    for (tile, &i) in idx[..k].iter().enumerate() {
        let (m, (v, note)) = judged[i];
        let img = image::open(m["file"].as_str().unwrap())?.to_rgb8();
        let b: Vec<f32> = m["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let (px, py) = ((b[2] * 0.4).max(24.0), (b[3] * 0.4).max(24.0));
        let x0 = (b[0] - px).max(0.0) as u32;
        let y0 = (b[1] - py).max(0.0) as u32;
        let x1 = ((b[0] + b[2] + px) as u32).min(img.width());
        let y1 = ((b[1] + b[3] + py) as u32).min(img.height());
        let mut c = imageops::crop_imm(&img, x0, y0, (x1 - x0).max(1), (y1 - y0).max(1)).to_image();
        let (bx0, by0) = ((b[0].max(0.0) as u32 - x0).min(c.width() - 1), (b[1].max(0.0) as u32 - y0).min(c.height() - 1));
        let (bx1, by1) = ((((b[0] + b[2]) as u32).saturating_sub(x0)).min(c.width() - 1), (((b[1] + b[3]) as u32).saturating_sub(y0)).min(c.height() - 1));
        for x in bx0..=bx1 {
            c.put_pixel(x, by0, Rgb([255, 220, 0]));
            c.put_pixel(x, by1, Rgb([255, 220, 0]));
        }
        for y in by0..=by1 {
            c.put_pixel(bx0, y, Rgb([255, 220, 0]));
            c.put_pixel(bx1, y, Rgb([255, 220, 0]));
        }
        let s = (t as f32 - 8.0) / c.width().max(c.height()) as f32;
        let r = imageops::resize(&c, ((c.width() as f32 * s) as u32).max(1), ((c.height() as f32 * s) as u32).max(1), imageops::FilterType::CatmullRom);
        imageops::overlay(&mut sheet, &r, ((tile as u32 % cols) * t + 4) as i64, ((tile as u32 / cols) * t + 4) as i64);
        println!("{tile:>3}  {}/{:<3} {:<16} a={:<14} b={:<14} {}", m["sheet"], m["cell"], v, m["a"].as_str().unwrap_or("-"), m["b"].as_str().unwrap_or("-"), &note.chars().take(50).collect::<String>());
    }
    sheet.save(&a[2])?;
    println!("{k} of {} by-eye verdicts → {}", judged.len(), a[2]);
    Ok(())
}
