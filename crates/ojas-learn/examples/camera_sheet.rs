//! Contact sheet of every camera in a sampled corpus: for each camera folder under `root` (the
//! `sample_dataset` layout, `<root>/<camera>/<utc_ms>.jpg`), `k` frames spread over its time span,
//! scaled to a 320 px wide tile with the camera's index burnt in as a bar of dots (index = row of
//! the printed table), tiled one camera per row. Prints the index, folder, frame count and mean
//! luma of each camera, so day, dusk, night and IR are visible before choosing what to label.
//!
//! `camera_sheet <root> <out.jpg> [k 4]`
use image::{imageops, Rgb, RgbImage};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 3, "camera_sheet <root> <out.jpg> [k 4]");
    let k: usize = a.get(3).map(|v| v.parse().unwrap()).unwrap_or(4);
    let mut cams: Vec<_> = std::fs::read_dir(&a[1])?.filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).map(|e| e.path()).collect();
    cams.sort();
    let (tw, th) = (320u32, 180u32);
    let mut sheet = RgbImage::from_pixel(tw * k as u32, th * cams.len() as u32, Rgb([0, 0, 0]));
    println!("{:>3}  {:<48} {:>7} {:>6}", "row", "camera", "frames", "luma");
    for (r, cam) in cams.iter().enumerate() {
        let mut files: Vec<_> = std::fs::read_dir(cam)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "jpg")).collect();
        files.sort();
        if files.is_empty() {
            println!("{r:>3}  {:<48} {:>7}", cam.file_name().unwrap().to_string_lossy(), 0);
            continue;
        }
        let mut luma = 0.0;
        for i in 0..k {
            let f = &files[(i * files.len() / k).min(files.len() - 1)];
            let img = image::open(f)?.to_rgb8();
            let t = imageops::resize(&img, tw, th, imageops::FilterType::Triangle);
            luma += t.pixels().map(|p| 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64).sum::<f64>() / (tw * th) as f64 / k as f64;
            imageops::overlay(&mut sheet, &t, (i as u32 * tw) as i64, (r as u32 * th) as i64);
        }
        // row index as white dots in the first tile's corner: tens as big dots, units as small
        for (n, size) in [(r / 10, 8u32), (r % 10, 4u32)].iter().copied().enumerate().map(|(j, (n, s))| ((n, s), j)).map(|(x, _)| x) {
            let row_y = if size == 8 { 4 } else { 16 };
            for d in 0..n as u32 {
                for y in 0..size {
                    for x in 0..size {
                        sheet.put_pixel(4 + d * (size + 3) + x, r as u32 * th + row_y + y, Rgb([255, 255, 0]));
                    }
                }
            }
        }
        println!("{r:>3}  {:<48} {:>7} {:>6.0}", cam.file_name().unwrap().to_string_lossy(), files.len(), luma);
    }
    sheet.save(&a[2])?;
    println!("→ {}", a[2]);
    Ok(())
}
