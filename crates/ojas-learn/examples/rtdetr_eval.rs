//! COCO mAP of an RT-DETRv2 checkpoint on a COCO-format set (resize 640, eval mode):
//! `rtdetr_eval model.pth prefix annotations.json image_root [limit] [batch]`
use std::sync::Arc;

use ojas_learn::cuda::Cuda;
use ojas_learn::data::{load_coco, loader};
use ojas_learn::eval::{coco_map, postprocess};
use ojas_learn::models::rtdetr::{Config, RtDetr, Store};
use ojas_learn::Tape;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let limit: usize = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(usize::MAX);
    let batch: usize = a.get(6).map(|v| v.parse().unwrap()).unwrap_or(8);
    let mut samples = load_coco(a[3].as_ref(), a[4].as_ref())?;
    samples.truncate(limit);
    println!("{} images, {} boxes", samples.len(), samples.iter().map(|s| s.boxes.len()).sum::<usize>());
    let tensors = ojas_formats::pth::load(&std::fs::read(&a[1])?)?;
    let be = Cuda::new(0)?;
    let st = Store::from_tensors(&be, &tensors, &a[2]);
    let cfg = Config::r18vd(15);
    let m = RtDetr { cfg: cfg.clone(), st: &st, train: false, var: Default::default() };
    let samples = Arc::new(samples);
    let order: Vec<usize> = (0..samples.len()).collect();
    let t0 = std::time::Instant::now();
    let (mut gts, mut dets) = (vec![], vec![]);
    for b in loader(samples.clone(), order, batch, 640, false, 12, 0) {
        let b = b?;
        let n = b.samples.len();
        let mut t = Tape::new(&be);
        let x = t.input(&b.images, &[n, 3, 640, 640]);
        let o = m.forward(&mut t, x, None)?;
        let (lg, bx) = (t.value(o.logits[0]), t.value(o.boxes[0]));
        if let Ok(p) = std::env::var("LOGITS_DUMP") {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p)?;
            f.write_all(&lg.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            f.write_all(&b.images.iter().flat_map(|v| v.to_le_bytes()).take(4 * 3 * 640 * 10).collect::<Vec<u8>>())?;
        }
        for (i, &si) in b.samples.iter().enumerate() {
            let s = &samples[si];
            let q = cfg.num_queries;
            dets.push(postprocess(&lg[i * q * 15..(i + 1) * q * 15], &bx[i * q * 4..(i + 1) * q * 4], 15, 300, s.width as f32, s.height as f32));
            gts.push(s.boxes.iter().map(|&(c, b)| (c, [b[0], b[1], b[0] + b[2], b[1] + b[3]])).collect());
        }
        if dets.len() % (batch * 25) == 0 {
            eprintln!("  {} / {} ({:.1} img/s)", dets.len(), samples.len(), dets.len() as f64 / t0.elapsed().as_secs_f64());
        }
    }
    let s = coco_map(&gts, &dets);
    if let Ok(path) = std::env::var("DETS_JSON") {
        // COCO results format, to cross-check with pycocotools
        let mut rows = vec![];
        for (d, s) in dets.iter().zip(samples.iter()) {
            for x in d {
                rows.push(serde_json::json!({"image_id": s.image_id, "category_id": x.label, "score": x.score,
                    "bbox": [x.xyxy[0], x.xyxy[1], x.xyxy[2] - x.xyxy[0], x.xyxy[3] - x.xyxy[1]]}));
            }
        }
        std::fs::write(&path, serde_json::to_vec(&rows)?)?;
    }
    println!("mAP@[.5:.95] {:.4}  AP50 {:.4}  AP75 {:.4}  ({} categories, {} images, {:.1} img/s)", s.ap, s.ap50, s.ap75, s.categories, dets.len(), dets.len() as f64 / t0.elapsed().as_secs_f64());
    Ok(())
}
