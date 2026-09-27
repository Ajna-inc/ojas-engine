//! Inspect one fixed spot of one camera: the box with generous context, cropped from `n` frames
//! spread over the camera's time span, scaled up 2× and tiled on one sheet, to settle whether a
//! static detection is a parked vehicle or street furniture.
//!
//! `review_spots <frames.json> <camera prefix> <x> <y> <w> <h> <out.jpg> [n 8]`
use image::{imageops, Rgb, RgbImage};
use serde_json::Value;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 8, "review_spots <frames.json> <camera prefix> <x> <y> <w> <h> <out.jpg> [n 8]");
    let v: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let b: Vec<f32> = a[3..7].iter().map(|s| s.parse().unwrap()).collect();
    let n: usize = a.get(8).map(|s| s.parse().unwrap()).unwrap_or(8);
    let mut ims: Vec<&Value> = v["images"].as_array().unwrap().iter().filter(|im| im["camera"].as_str().unwrap_or("").starts_with(a[2].as_str())).collect();
    ims.sort_by_key(|im| im["utc_ms"].as_i64().unwrap_or(0));
    anyhow::ensure!(!ims.is_empty(), "no frames for camera {}", a[2]);
    let pick: Vec<&Value> = (0..n.min(ims.len())).map(|i| ims[i * ims.len() / n.min(ims.len())]).collect();
    // context: 1.5× the box each side, at least 60 px
    let (px, py) = ((b[2] * 1.5).max(60.0), (b[3] * 1.5).max(60.0));
    let tiles: Vec<RgbImage> = pick
        .iter()
        .map(|im| {
            let img = image::open(im["file_name"].as_str().unwrap()).unwrap().to_rgb8();
            let (x0, y0) = ((b[0] - px).max(0.0) as u32, (b[1] - py).max(0.0) as u32);
            let x1 = ((b[0] + b[2] + px) as u32).min(img.width());
            let y1 = ((b[1] + b[3] + py) as u32).min(img.height());
            let mut t = imageops::crop_imm(&img, x0, y0, x1 - x0, y1 - y0).to_image();
            let (bx0, by0, bx1, by1) = (b[0] as i64 - x0 as i64, b[1] as i64 - y0 as i64, (b[0] + b[2]) as i64 - x0 as i64, (b[1] + b[3]) as i64 - y0 as i64);
            for x in bx0..=bx1 {
                for y in [by0, by1] {
                    if x >= 0 && y >= 0 && (x as u32) < t.width() && (y as u32) < t.height() {
                        t.put_pixel(x as u32, y as u32, Rgb([255, 0, 255]));
                    }
                }
            }
            for y in by0..=by1 {
                for x in [bx0, bx1] {
                    if x >= 0 && y >= 0 && (x as u32) < t.width() && (y as u32) < t.height() {
                        t.put_pixel(x as u32, y as u32, Rgb([255, 0, 255]));
                    }
                }
            }
            imageops::resize(&t, t.width() * 2, t.height() * 2, imageops::FilterType::CatmullRom)
        })
        .collect();
    let (tw, th) = (tiles.iter().map(|t| t.width()).max().unwrap(), tiles.iter().map(|t| t.height()).max().unwrap());
    let cols = 4u32.min(tiles.len() as u32);
    let rows = (tiles.len() as u32).div_ceil(cols);
    let mut sheet = RgbImage::from_pixel(cols * tw, rows * th, Rgb([0, 0, 0]));
    for (i, t) in tiles.iter().enumerate() {
        imageops::overlay(&mut sheet, t, ((i as u32 % cols) * tw) as i64, ((i as u32 / cols) * th) as i64);
    }
    sheet.save(&a[7])?;
    println!("{} frames of {} → {}", tiles.len(), a[2], a[7]);
    Ok(())
}
