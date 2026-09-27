//! How well 2-D pose holds up at the person sizes a camera actually sees. Runs an
//! RTMPose (SimCC) export on person crops taken from a detection file, stratified by
//! box height, and reports how many keypoints come out confident per height band; the
//! gait and height lanes need the ankles, knees and hips. Draws the skeletons on
//! contact sheets for review by eye.
//!
//! Preprocessing follows the export's `pipeline.json`: box centre and 1.25× scale, aspect fixed
//! to 192:256, affine crop (zero outside the frame), RGB, mean/std normalisation. Decode: argmax
//! of the SimCC x/y vectors at split ratio 2, score = min of the two maxima.
//!
//! `pose_probe <frames.json> <persons.dets.json> <rtmpose.onnx> <out_dir> [per_bin 30] [conf 0.5]`
//! Env: `OJAS_DEVICE=cuda:0|cpu`.
use std::collections::HashMap;

use image::{imageops, Rgb, RgbImage};
use serde_json::Value;

const W: usize = 192;
const H: usize = 256;
const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];
const BINS: [(f32, &str); 6] = [(60.0, "<60"), (100.0, "60-100"), (150.0, "100-150"), (200.0, "150-200"), (280.0, "200-280"), (f32::MAX, "≥280")];
// COCO-17 skeleton
const BONES: [(usize, usize); 16] = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (3, 5)];
const LOWER: [usize; 6] = [11, 12, 13, 14, 15, 16]; // hips, knees, ankles

/// The crop window for a box: centre, and a 1.25× box with the input's aspect.
fn window(b: [f32; 4]) -> (f32, f32, f32, f32) {
    let (cx, cy) = (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0);
    let (mut sw, mut sh) = (b[2] * 1.25, b[3] * 1.25);
    let aspect = W as f32 / H as f32;
    if sw > sh * aspect {
        sh = sw / aspect;
    } else {
        sw = sh * aspect;
    }
    (cx, cy, sw, sh)
}

/// Bilinear crop of the window into a W×H RGB image (black outside the frame).
fn crop(img: &RgbImage, win: (f32, f32, f32, f32)) -> RgbImage {
    let (cx, cy, sw, sh) = win;
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let mut out = RgbImage::new(W as u32, H as u32);
    for y in 0..H {
        for x in 0..W {
            let sx = cx - sw / 2.0 + (x as f32 + 0.5) * sw / W as f32 - 0.5;
            let sy = cy - sh / 2.0 + (y as f32 + 0.5) * sh / H as f32 - 0.5;
            if sx < 0.0 || sy < 0.0 || sx >= iw - 1.0 || sy >= ih - 1.0 {
                continue;
            }
            let (x0, y0) = (sx.floor() as u32, sy.floor() as u32);
            let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
            let p = |dx: u32, dy: u32| img.get_pixel(x0 + dx, y0 + dy).0;
            let (a, b, c, d) = (p(0, 0), p(1, 0), p(0, 1), p(1, 1));
            let mut px = [0u8; 3];
            for k in 0..3 {
                let v = a[k] as f32 * (1.0 - fx) * (1.0 - fy) + b[k] as f32 * fx * (1.0 - fy) + c[k] as f32 * (1.0 - fx) * fy + d[k] as f32 * fx * fy;
                px[k] = v.round() as u8;
            }
            out.put_pixel(x as u32, y as u32, Rgb(px));
        }
    }
    out
}

fn normalise(c: &RgbImage, out: &mut [f32]) {
    for ch in 0..3 {
        for y in 0..H {
            for x in 0..W {
                out[ch * W * H + y * W + x] = (c.get_pixel(x as u32, y as u32).0[ch] as f32 - MEAN[ch]) / STD[ch];
            }
        }
    }
}

/// SimCC decode: 17 keypoints in crop pixels with a score.
fn decode(sx: &[f32], sy: &[f32]) -> [(f32, f32, f32); 17] {
    let (nx, ny) = (sx.len() / 17, sy.len() / 17);
    let mut out = [(0.0, 0.0, 0.0); 17];
    for k in 0..17 {
        let (mut bx, mut vx) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in sx[k * nx..(k + 1) * nx].iter().enumerate() {
            if v > vx {
                (bx, vx) = (i, v);
            }
        }
        let (mut by, mut vy) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in sy[k * ny..(k + 1) * ny].iter().enumerate() {
            if v > vy {
                (by, vy) = (i, v);
            }
        }
        out[k] = (bx as f32 / 2.0, by as f32 / 2.0, vx.min(vy));
    }
    out
}

fn line(img: &mut RgbImage, a: (f32, f32), b: (f32, f32), c: Rgb<u8>) {
    let n = ((a.0 - b.0).abs().max((a.1 - b.1).abs()) as usize).max(1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
        if x >= 0.0 && y >= 0.0 && (x as u32) < img.width() && (y as u32) < img.height() {
            img.put_pixel(x as u32, y as u32, c);
        }
    }
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 5, "pose_probe <frames.json> <persons.dets.json> <rtmpose.onnx> <out_dir> [per_bin] [conf]");
    let per_bin: usize = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(30);
    let kp_conf: f32 = a.get(6).and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let out_dir = std::path::Path::new(&a[4]);
    std::fs::create_dir_all(out_dir)?;

    let frames: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let files: HashMap<i64, String> = frames["images"].as_array().unwrap().iter().map(|im| (im["id"].as_i64().unwrap(), im["file_name"].as_str().unwrap().to_string())).collect();
    let dets: Vec<Value> = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    // stratified sample: every k-th detection of each height band, spread over the whole file
    let mut by_bin: Vec<Vec<(i64, [f32; 4])>> = vec![vec![]; BINS.len()];
    for d in &dets {
        let b: Vec<f32> = d["bbox"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let bi = BINS.iter().position(|(top, _)| b[3] < *top).unwrap();
        by_bin[bi].push((d["image_id"].as_i64().unwrap(), [b[0], b[1], b[2], b[3]]));
    }
    let mut sample: Vec<(usize, i64, [f32; 4])> = vec![];
    for (bi, v) in by_bin.iter().enumerate() {
        let step = (v.len() / per_bin).max(1);
        sample.extend(v.iter().step_by(step).take(per_bin).map(|(id, b)| (bi, *id, *b)));
    }
    sample.sort_by_key(|s| s.1); // by frame, so each image is decoded once

    // the model, through the raw executors (two outputs: simcc_x, simcc_y)
    let model = ojas_formats::onnx::load(&a[3])?;
    let mut g = ojas_vision::import(&model, &HashMap::from([("batch".to_string(), 1usize)]))?;
    ojas_vision::passes::optimize(&mut g);
    let device = std::env::var("OJAS_DEVICE").unwrap_or_else(|_| "cpu".into());
    let mut cpu = None;
    #[cfg(feature = "cuda")]
    let mut gpu = None;
    match device.as_str() {
        #[cfg(feature = "cuda")]
        d if d.starts_with("cuda") => {
            ojas_vision::passes::lower_for_gpu(&mut g);
            gpu = Some(ojas_vision::exec_gpu::CudaExecutor::new(&g, 0)?);
        }
        _ => cpu = Some(ojas_vision::exec_cpu::CpuExecutor::new(&g, std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))),
    }
    let mut run = |input: &[f32]| -> anyhow::Result<Vec<Vec<f32>>> {
        #[cfg(feature = "cuda")]
        if let Some(ex) = gpu.as_mut() {
            return ex.run(input);
        }
        cpu.as_mut().unwrap().run(&g, &[input])
    };
    println!("{} crops ({} per band) through {} on {device}", sample.len(), per_bin, a[3].rsplit('/').next().unwrap_or(""));

    let mut input = vec![0.0f32; 3 * W * H];
    let mut stats: Vec<(usize, usize, usize, usize)> = vec![(0, 0, 0, 0); BINS.len()]; // crops, confident kps, lower-body confident, full-body crops
    let mut cells: Vec<(usize, RgbImage)> = vec![];
    let mut cache: Option<(i64, RgbImage)> = None;
    let mut ms = 0.0f64;
    let mut dumped = false;
    let verbose = std::env::var("VERBOSE").is_ok();
    for (bi, id, b) in &sample {
        if cache.as_ref().map(|c| c.0) != Some(*id) {
            cache = Some((*id, image::open(&files[id])?.to_rgb8()));
        }
        let img = &cache.as_ref().unwrap().1;
        let win = window(*b);
        let c = crop(img, win);
        normalise(&c, &mut input);
        // `DUMP=path`: the first crop's input as raw f32, for graph_run / parity checks
        // (`DUMP_BAND=k`: the first crop of height band k instead)
        let want_band: usize = std::env::var("DUMP_BAND").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
        if let (Ok(p), true) = (std::env::var("DUMP"), !dumped && (*bi == want_band || (want_band == usize::MAX && cells.is_empty()))) {
            std::fs::write(p, input.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            dumped = true;
        }
        let t = std::time::Instant::now();
        let outs = run(&input)?;
        ms += t.elapsed().as_secs_f64() * 1e3;
        let kps = decode(&outs[0], &outs[1]);
        let n_conf = kps.iter().filter(|k| k.2 >= kp_conf).count();
        let lower = LOWER.iter().filter(|&&k| kps[k].2 >= kp_conf).count();
        if verbose {
            println!("crop frame {id} band {bi} box {:?}: {n_conf} kps, lower {lower}, scores {}", b.map(|v| v.round()), kps.iter().map(|k| format!("{:.2}", k.2)).collect::<Vec<_>>().join(" "));
        }
        let s = &mut stats[*bi];
        s.0 += 1;
        s.1 += n_conf;
        s.2 += lower;
        s.3 += (lower == 6) as usize;
        // draw
        let mut cell = c.clone();
        for &(p, q) in &BONES {
            if kps[p].2 >= kp_conf && kps[q].2 >= kp_conf {
                line(&mut cell, (kps[p].0, kps[p].1), (kps[q].0, kps[q].1), Rgb([0, 255, 0]));
            }
        }
        for k in kps.iter() {
            let col = if k.2 >= kp_conf { Rgb([255, 255, 0]) } else { Rgb([255, 0, 0]) };
            for dx in -1..=1i32 {
                for dy in -1..=1i32 {
                    let (x, y) = (k.0 as i32 + dx, k.1 as i32 + dy);
                    if x >= 0 && y >= 0 && (x as u32) < W as u32 && (y as u32) < H as u32 {
                        cell.put_pixel(x as u32, y as u32, col);
                    }
                }
            }
        }
        cells.push((*bi, cell));
    }

    println!("\n{:.2} ms per crop on {device}\n", ms / sample.len().max(1) as f64);
    println!("{:<10}{:>7}{:>16}{:>18}{:>16}", "height px", "crops", "kps ≥ conf /17", "lower-body kps /6", "full lower body");
    for (bi, (n, kp, lo, full)) in stats.iter().enumerate() {
        if *n > 0 {
            println!("{:<10}{n:>7}{:>16.1}{:>18.1}{:>15.0}%", BINS[bi].1, *kp as f64 / *n as f64, *lo as f64 / *n as f64, 100.0 * *full as f64 / *n as f64);
        }
    }
    // contact sheets per band, 10 columns
    for bi in 0..BINS.len() {
        let band: Vec<&RgbImage> = cells.iter().filter(|(b, _)| *b == bi).map(|(_, c)| c).collect();
        if band.is_empty() {
            continue;
        }
        let cols = 10u32;
        let rows = (band.len() as u32).div_ceil(cols);
        let mut sheet = RgbImage::from_pixel(cols * W as u32, rows * H as u32, Rgb([0, 0, 0]));
        for (i, c) in band.iter().enumerate() {
            imageops::overlay(&mut sheet, *c, ((i as u32 % cols) * W as u32) as i64, ((i as u32 / cols) * H as u32) as i64);
        }
        sheet.save(out_dir.join(format!("pose_{}.jpg", BINS[bi].1.replace('≥', "ge").replace('<', "lt"))))?;
    }
    println!("\nsheets → {}", out_dir.display());
    Ok(())
}
