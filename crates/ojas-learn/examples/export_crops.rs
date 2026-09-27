//! Export vehicle crops with their fine class, for a crop expert separating
//! Hatchback / Sedan / SUV / MUV, where 89 % of the recoverable subtype error lives.
//!
//! Crops come from the original-resolution image with context around the box, never from a resized
//! frame: the 640-pixel detector input has already lost the detail that separates a sedan from a
//! hatchback.
//!
//! The split is by source image rather than by crop, so two crops of one scene cannot land on both
//! sides. A camera-disjoint split would be better still, but UVH-26 ships no camera id.
//!
//! `export_crops <annotations.json> <image_root> <out_dir> [classes] [min_px] [max_per_class]`
//!   classes: comma-separated category ids, default `1,2,3,4` (the passenger family)
use std::collections::HashMap;
use std::io::Write;

use ojas_learn::data::{decode, load_coco};

const NAMES: [&str; 15] = ["-", "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"];

/// Context around the box, as a fraction of its size: the roofline and the gap under the bumper
/// are part of what separates these classes.
const CONTEXT: f32 = 0.15;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() > 3, "export_crops <annotations.json> <image_root> <out_dir> [classes] [min_px] [max_per_class]");
    let classes: Vec<usize> = a.get(4).map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect()).unwrap_or_else(|| vec![1, 2, 3, 4]);
    let min_px: f32 = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(48.0);
    let max_per_class: usize = a.get(6).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let out = std::path::PathBuf::from(&a[3]);
    for c in &classes {
        std::fs::create_dir_all(out.join(NAMES.get(*c).unwrap_or(&"?")))?;
    }

    let samples = load_coco(a[1].as_ref(), a[2].as_ref())?;
    println!("{} images; exporting classes {:?} at ≥ {min_px} px", samples.len(), classes.iter().map(|c| NAMES[*c]).collect::<Vec<_>>());

    let mut kept: HashMap<usize, usize> = HashMap::new();
    let mut too_small = 0usize;
    let mut index = std::fs::File::create(out.join("crops.tsv"))?;
    writeln!(index, "path\tclass\tclass_id\timage_id\tw\th")?;
    let mut done = 0usize;

    for s in &samples {
        // one decode per image, all of its crops taken from it
        if !s.boxes.iter().any(|(c, b)| classes.contains(c) && b[2] >= min_px && b[3] >= min_px) {
            continue;
        }
        let img = match decode(&s.path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("  skip {}: {e:#}", s.path.display());
                continue;
            }
        };
        for (ci, (class, b)) in s.boxes.iter().enumerate() {
            if !classes.contains(class) {
                continue;
            }
            let (bw, bh) = (b[2], b[3]);
            if bw < min_px || bh < min_px {
                too_small += 1;
                continue;
            }
            if *kept.get(class).unwrap_or(&0) >= max_per_class {
                continue;
            }
            let (px, py) = (bw * CONTEXT, bh * CONTEXT);
            let x0 = (b[0] - px).max(0.0) as usize;
            let y0 = (b[1] - py).max(0.0) as usize;
            let x1 = ((b[0] + bw + px) as usize).min(img.w);
            let y1 = ((b[1] + bh + py) as usize).min(img.h);
            if x1 <= x0 + 8 || y1 <= y0 + 8 {
                continue;
            }
            let (cw, ch) = (x1 - x0, y1 - y0);
            let mut buf = Vec::with_capacity(cw * ch * 3);
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * img.w + x) * 3;
                    buf.extend_from_slice(&[(img.rgb[i] * 255.0) as u8, (img.rgb[i + 1] * 255.0) as u8, (img.rgb[i + 2] * 255.0) as u8]);
                }
            }
            let name = NAMES.get(*class).unwrap_or(&"?");
            let rel = format!("{name}/{:08}_{ci}.jpg", s.image_id);
            image::save_buffer(out.join(&rel), &buf, cw as u32, ch as u32, image::ColorType::Rgb8)?;
            writeln!(index, "{rel}\t{name}\t{class}\t{}\t{cw}\t{ch}", s.image_id)?;
            *kept.entry(*class).or_default() += 1;
        }
        done += 1;
        if done % 500 == 0 {
            eprintln!("  {done} images, {} crops", kept.values().sum::<usize>());
        }
    }

    let total: usize = kept.values().sum();
    println!("\n| class | crops | share |");
    println!("|---|---:|---:|");
    let mut rows: Vec<(usize, usize)> = kept.into_iter().collect();
    rows.sort_by_key(|x| std::cmp::Reverse(x.1));
    for (c, n) in &rows {
        println!("| {} | {n} | {:.1} % |", NAMES[*c], 100.0 * *n as f32 / total.max(1) as f32);
    }
    println!("\n{total} crops written to {}, {too_small} boxes below {min_px} px skipped", out.display());
    println!("index: {}/crops.tsv (split by image_id, never by crop)", out.display());
    Ok(())
}
