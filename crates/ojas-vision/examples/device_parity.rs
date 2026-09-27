//! The public API on `Device::Cpu` vs a GPU: vehicle / plate boxes (matched
//! by IoU) and plate texts on the same images, plus timing. The GPU:
//! `OJAS_PARITY_DEVICE=cuda:0` (default), `vulkan:<index or name part>`
//! (e.g. `vulkan:nvidia`, `vulkan:radv`, `vulkan:llvmpipe`).
//!
//! `device_parity <vehicle.onnx> <plate.onnx> <rec.onnx> <dict.txt> <image|dir>...`
//! Plates are detected on the whole image (the images are plate close-ups or
//! street scenes) and read from the padded plate crop, as anpr.rs does.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use ojas_vision::{Detection, DetectorCfg, Device, Frame, OcrCfg, OcrNorm, Runtime, RuntimeCfg};

/// `OJAS_PARITY_DEVICE`: `cuda:N` or `vulkan:<index | name part>`.
fn parity_device() -> Result<Device> {
    let spec = std::env::var("OJAS_PARITY_DEVICE").unwrap_or_else(|_| "cuda:0".into());
    let (kind, which) = spec.split_once(':').unwrap_or((spec.as_str(), "0"));
    Ok(match kind {
        "cuda" => Device::Cuda(which.parse()?),
        "vulkan" => match which.parse::<usize>() {
            Ok(i) => Device::Vulkan(i),
            Err(_) => {
                let info = Runtime::probe();
                let w = which.to_lowercase();
                let v = info.vulkan.iter().find(|d| d.name.to_lowercase().contains(&w) || d.driver.to_lowercase().contains(&w))
                    .ok_or_else(|| anyhow::anyhow!("no Vulkan device matches {which:?}: {:?}", info.vulkan.iter().map(|d| &d.name).collect::<Vec<_>>()))?;
                Device::Vulkan(v.index)
            }
        },
        other => anyhow::bail!("OJAS_PARITY_DEVICE: {other}? (cuda:N or vulkan:<index|name>)"),
    })
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let iw = (a.x1.min(b.x1) - a.x0.max(b.x0)).max(0.0);
    let ih = (a.y1.min(b.y1) - a.y0.max(b.y0)).max(0.0);
    let i = iw * ih;
    let u = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - i;
    if u > 0.0 { i / u } else { 0.0 }
}

/// Boxes of `a` with a same-class partner in `b` at IoU >= 0.9.
fn matched(a: &[Detection], b: &[Detection]) -> usize {
    a.iter().filter(|x| b.iter().any(|y| y.class == x.class && iou(x, y) >= 0.9)).count()
}

fn crop(rgb: &[u8], w: usize, h: usize, d: &Detection, pad: f32) -> Option<(usize, usize, Vec<u8>)> {
    let (pw, ph) = ((d.x1 - d.x0) * pad, (d.y1 - d.y0) * pad);
    let (x0, y0) = ((d.x0 - pw).max(0.0) as usize, (d.y0 - ph).max(0.0) as usize);
    let (x1, y1) = (((d.x1 + pw).ceil() as usize).min(w), ((d.y1 + ph).ceil() as usize).min(h));
    let (cw, ch) = (x1.checked_sub(x0)?, y1.checked_sub(y0)?);
    if cw < 8 || ch < 8 {
        return None;
    }
    let mut out = Vec::with_capacity(cw * ch * 3);
    for y in y0..y1 {
        out.extend_from_slice(&rgb[(y * w + x0) * 3..(y * w + x1) * 3]);
    }
    Some((cw, ch, out))
}

struct Side {
    veh: ojas_vision::Detector,
    plate: ojas_vision::Detector,
    ocr: ojas_vision::PlateOcr,
    ms: f64,
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let mut images: Vec<PathBuf> = vec![];
    for p in &a[5..] {
        let p = PathBuf::from(p);
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(&p)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "jpg" || x == "png" || x == "jpeg")).collect();
            v.sort();
            images.extend(v);
        } else {
            images.push(p);
        }
    }
    let info = Runtime::probe();
    println!("probe: {}", serde_json::to_string(&info)?);
    let load = |device| -> Result<Side> {
        let rt = Runtime::new(RuntimeCfg { threads: None, device })?;
        let veh = rt.detector(&a[1], DetectorCfg { classes: Some(vec![2, 3, 5, 7]), ..Default::default() })?;
        let plate = rt.detector(&a[2], DetectorCfg { conf: 0.15, ..Default::default() })?;
        let ocr = rt.plate_ocr(&a[3], &a[4], OcrCfg { norm: OcrNorm::Signed, bgr: true, ..Default::default() })?;
        println!("{device:?}: detector {} / plate {} / ocr {}", veh.backend(), plate.backend(), ocr.backend());
        Ok(Side { veh, plate, ocr, ms: 0.0 })
    };
    let gpu = parity_device()?;
    eprintln!("GPU side: {gpu:?}");
    let mut sides = [load(Device::Cpu)?, load(gpu)?];
    let (mut vb, mut vm, mut pb, mut pm, mut reads, mut same) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut diffs = vec![];
    for path in &images {
        let (w, h, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(path.to_str().unwrap())?;
        let frame = Frame::Rgb8 { w, h, data: &rgb };
        let mut res = vec![];
        // OCR input: crops from the CPU's plate boxes on both sides, so text
        // differences are the reader's, not a pixel of crop border
        let mut ref_boxes: Option<Vec<Detection>> = None;
        for s in sides.iter_mut() {
            let t = Instant::now();
            let v = s.veh.run(&[frame])?.remove(0);
            let p = s.plate.run(&[frame])?.remove(0);
            let boxes = ref_boxes.get_or_insert_with(|| p.clone()).clone();
            let crops: Vec<(usize, usize, Vec<u8>)> = boxes.iter().filter_map(|d| crop(&rgb, w, h, d, 0.10)).collect();
            let frames: Vec<Frame> = crops.iter().map(|(cw, ch, c)| Frame::Rgb8 { w: *cw, h: *ch, data: c }).collect();
            let texts: Vec<String> = s.ocr.run(&frames)?.into_iter().map(|r| format!("{} {:.2}", r.text, r.mean_conf)).collect();
            if let (Ok(dir), true) = (std::env::var("DUMP_DIR"), res.is_empty()) {
                // the CPU reader's exact input tensors, for an onnxruntime cross-check
                for (k, (cw, ch, c)) in crops.iter().enumerate() {
                    let mut t = vec![0f32; 3 * 48 * 320];
                    ojas_vision::pre::ocr_resize_into(c, *cw, *ch, 48, 320, OcrNorm::Signed, true, &mut t);
                    let name = format!("{}/{}_{k}.f32", dir, path.file_stem().unwrap().to_string_lossy());
                    std::fs::write(name, t.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                }
            }
            s.ms += t.elapsed().as_secs_f64() * 1e3;
            res.push((v, p, texts));
        }
        let ((cv, cp, ct), (gv, gp, gt)) = (&res[0], &res[1]);
        vb += cv.len();
        vm += matched(cv, gv).min(matched(gv, cv));
        pb += cp.len();
        pm += matched(cp, gp).min(matched(gp, cp));
        {
            reads += ct.len();
            same += ct.iter().zip(gt).filter(|(x, y)| x.split(' ').next() == y.split(' ').next()).count();
        }
        if let Ok(dir) = std::env::var("DUMP_DIR") {
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            for (k, (c, g)) in ct.iter().zip(gt).enumerate() {
                use std::io::Write;
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/texts.tsv"))?;
                writeln!(f, "{stem}_{k}\t{}\t{}", c.split(' ').next().unwrap(), g.split(' ').next().unwrap())?;
            }
        }
        if ct.iter().map(|x| x.split(' ').next()).ne(gt.iter().map(|x| x.split(' ').next())) {
            diffs.push(format!("{}: cpu {ct:?} gpu {gt:?}", path.file_name().unwrap().to_string_lossy()));
        }
    }
    let n = images.len().max(1) as f64;
    println!("{} images", images.len());
    println!("vehicle boxes: {vm}/{vb} matched at IoU>=0.9 | plate boxes: {pm}/{pb} | plate texts identical: {same}/{reads}");
    println!("time per image (detect + plates + OCR, host frames): cpu {:.1} ms, gpu {:.1} ms", sides[0].ms / n, sides[1].ms / n);
    for d in diffs.iter().take(10) {
        println!("  differs: {d}");
    }
    Ok(())
}
