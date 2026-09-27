//! Full ANPR chain on real images: vehicle detector → vehicle crops → plate
//! detector → plate crops → OCR — the deployed pipeline shape, run standalone for
//! end-to-end validation.
//!
//! cargo run -p ojas-vision --release --example anpr -- \
//!     vehicle.onnx plate.onnx rec.onnx dict.txt image...

use anyhow::{bail, Result};
use ojas_vision::{Detection, DetectorCfg, Frame, OcrCfg, Runtime, RuntimeCfg};

/// COCO vehicle classes: car, motorcycle, bus, truck.
const VEHICLES: [u16; 4] = [2, 3, 5, 7];

fn crop(rgb: &[u8], w: usize, h: usize, d: &Detection, pad: f32) -> (usize, usize, usize, usize, Vec<u8>) {
    let pw = (d.x1 - d.x0) * pad;
    let ph = (d.y1 - d.y0) * pad;
    let x0 = (d.x0 - pw).max(0.0) as usize;
    let y0 = (d.y0 - ph).max(0.0) as usize;
    let x1 = ((d.x1 + pw).ceil() as usize).min(w);
    let y1 = ((d.y1 + ph).ceil() as usize).min(h);
    let (cw, ch) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    let mut out = vec![0u8; cw * ch * 3];
    for y in 0..ch {
        let src = &rgb[((y0 + y) * w + x0) * 3..((y0 + y) * w + x1) * 3];
        out[y * cw * 3..(y + 1) * cw * 3].copy_from_slice(src);
    }
    (x0, y0, cw, ch, out)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        bail!("usage: anpr <vehicle.onnx> <plate.onnx> <rec.onnx> <dict.txt> <image>...");
    }
    let rt = Runtime::new(RuntimeCfg::default())?;
    let mut veh = rt.detector(
        &args[1],
        DetectorCfg { conf: 0.25, classes: Some(VEHICLES.to_vec()), ..Default::default() },
    )?;
    let mut plate = rt.detector(&args[2], DetectorCfg { conf: 0.15, ..Default::default() })?;
    let mut ocr = rt.plate_ocr(&args[3], &args[4], OcrCfg::default())?;

    for path in &args[5..] {
        let (w, h, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(path)?;
        let t0 = std::time::Instant::now();
        let vehicles = veh.run(&[Frame::Rgb8 { w, h, data: &rgb }])?.remove(0);
        println!("\n{path} ({w}x{h}): {} vehicles [{:.0} ms]", vehicles.len(), t0.elapsed().as_secs_f64() * 1e3);

        // plates inside each vehicle crop (the deployed path)
        for (vi, v) in vehicles.iter().enumerate() {
            let (vx, vy, cw, ch, vcrop) = crop(&rgb, w, h, v, 0.03);
            if cw < 40 || ch < 40 {
                continue;
            }
            let plates = plate.run(&[Frame::Rgb8 { w: cw, h: ch, data: &vcrop }])?.remove(0);
            println!("  vehicle[{vi}] class {} conf {:.2} at [{:.0},{:.0},{:.0},{:.0}]: {} plates", v.class, v.score, v.x0, v.y0, v.x1, v.y1, plates.len());
            for p in &plates {
                let (_, _, pw2, ph2, pcrop) = crop(&vcrop, cw, ch, p, 0.10);
                if pw2 < 20 || ph2 < 8 {
                    println!("    plate conf {:.2} too small ({pw2}x{ph2})", p.score);
                    continue;
                }
                let read = ocr.run(&[Frame::Rgb8 { w: pw2, h: ph2, data: &pcrop }])?.remove(0);
                println!(
                    "    plate conf {:.2} at frame [{:.0},{:.0}] ({pw2}x{ph2}) -> {:?} (mean {:.2})",
                    p.score,
                    vx as f32 + p.x0,
                    vy as f32 + p.y0,
                    read.text,
                    read.mean_conf
                );
            }
        }

        // plates on the full frame (close-ups with no vehicle box)
        let plates = plate.run(&[Frame::Rgb8 { w, h, data: &rgb }])?.remove(0);
        for p in &plates {
            let (_, _, pw2, ph2, pcrop) = crop(&rgb, w, h, p, 0.10);
            if pw2 < 20 || ph2 < 8 {
                continue;
            }
            let read = ocr.run(&[Frame::Rgb8 { w: pw2, h: ph2, data: &pcrop }])?.remove(0);
            println!(
                "  full-frame plate conf {:.2} [{:.0},{:.0},{:.0},{:.0}] -> {:?} (mean {:.2})",
                p.score, p.x0, p.y0, p.x1, p.y1, read.text, read.mean_conf
            );
        }
    }
    Ok(())
}
