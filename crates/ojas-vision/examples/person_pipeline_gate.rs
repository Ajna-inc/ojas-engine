//! The GPU person pipeline end to end on real frames — D-FINE person detection, then OSNet,
//! RTMPose and SigLIP on every detection, batched — against the CPU `Model` API run on the
//! same boxes: embeddings by cosine, keypoints by pixels; then the stage times per frame.
//!
//! `person_pipeline_gate <dfine.onnx> <osnet.onnx> <rtmpose.onnx> <siglip.onnx> <image>...`
use ojas_vision::model::{ModelKind, Output};
use ojas_vision::person::{osnet_spec, rtmpose_spec, siglip_spec, PersonBatches, PersonCfg, PersonModels, PersonPipeline};
use ojas_vision::pipeline::RgbFrame;
use ojas_vision::pre::window_geom;
use ojas_vision::{Device, Frame, Runtime, RuntimeCfg};

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 6, "person_pipeline_gate <dfine.onnx> <osnet.onnx> <rtmpose.onnx> <siglip.onnx> <image>...");
    let imgs: Vec<(usize, usize, Vec<u8>)> = a[5..].iter().map(|p| ojas_cpu::vit_preprocess::decode_rgb8_path(p)).collect::<anyhow::Result<_>>()?;
    let frames: Vec<RgbFrame> = imgs.iter().map(|(w, h, d)| RgbFrame { w: *w, h: *h, data: d }).collect();

    let t = std::time::Instant::now();
    let mut pp = PersonPipeline::load(PersonModels { detector: Some(&a[1]), reid: Some(&a[2]), pose: Some(&a[3]), clip: Some(&a[4]) }, &PersonBatches { detector: vec![1], ..Default::default() }, 0, None, PersonCfg::default())?;
    println!("plans: {:.1} s", t.elapsed().as_secs_f64());
    pp.run(&frames)?; // warm-up
    let res = pp.run(&frames)?;
    let tm = pp.times.clone();
    println!("{} frames: {} people; upload {:.1} detect {:.1} reid {:.1} ({} crops) pose {:.1} ({}) clip {:.1} ({}) ms", frames.len(), tm.detections, tm.upload, tm.detect, tm.reid, tm.reid_crops, tm.pose, tm.pose_crops, tm.clip, tm.clip_crops);

    // the CPU reference on the same boxes, through the Model API (each box is the frame handed in)
    let cpu = Runtime::new(RuntimeCfg { threads: None, device: Device::Cpu })?;
    let mut mo = cpu.model(&a[2], ModelKind::Tensor(osnet_spec()))?;
    let mut mp = cpu.model(&a[3], ModelKind::Tensor(rtmpose_spec()))?;
    let mut ms = cpu.model(&a[4], ModelKind::Tensor(siglip_spec()))?;
    let (mut worst_reid, mut worst_clip, mut kp_n, mut kp_far, mut n_reid, mut n_clip) = (1.0f32, 1.0f32, 0usize, 0usize, 0usize, 0usize);
    for ((w, h, rgb), (dets, outs)) in imgs.iter().zip(&res) {
        for (d, o) in dets.iter().zip(outs) {
            // the pipeline's ROI: the box clipped to the frame on whole pixels
            let x0 = d.x0.max(0.0).min(*w as f32 - 2.0) as usize;
            let y0 = d.y0.max(0.0).min(*h as f32 - 2.0) as usize;
            let x1 = (d.x1.min(*w as f32).max(x0 as f32 + 1.0).ceil() as usize).min(*w);
            let y1 = (d.y1.min(*h as f32).max(y0 as f32 + 1.0).ceil() as usize).min(*h);
            let (cw, ch) = (x1 - x0, y1 - y0);
            // the CPU Model sees only the box crop; for the pose window (which reads beyond the
            // box) hand it the 1.25× window region of the frame instead, as the same box
            let mut crop = Vec::with_capacity(cw * ch * 3);
            for y in y0..y1 {
                crop.extend_from_slice(&rgb[(y * w + x0) * 3..(y * w + x1) * 3]);
            }
            if let Some(v) = &o.reid {
                if let Output::Vector(c) = &mo.run(&[Frame::Rgb8 { w: cw, h: ch, data: &crop }])?[0] {
                    worst_reid = worst_reid.min(cos(v, c));
                    n_reid += 1;
                }
            }
            if let Some(v) = &o.clip {
                if let Output::Vector(c) = &ms.run(&[Frame::Rgb8 { w: cw, h: ch, data: &crop }])?[0] {
                    worst_clip = worst_clip.min(cos(v, c));
                    n_clip += 1;
                }
            }
            if let Some(kps) = &o.pose {
                // CPU twin of the window: the same geometry over the whole frame
                let spec = rtmpose_spec();
                let g = window_geom([x0 as f32, y0 as f32, cw as f32, ch as f32], *w, *h, spec.width, spec.height, spec.window);
                let mut win = Vec::with_capacity(g.w * g.h * 3);
                for y in g.y0..g.y0 + g.h {
                    win.extend_from_slice(&rgb[(y * w + g.x0) * 3..(y * w + g.x0 + g.w) * 3]);
                }
                // the crop already is the clipped window: stretch it into the input at the same
                // place the pipeline put it, by handing the Model the window as its box
                let spec_stretch = ojas_vision::model::TensorSpec { window: ojas_vision::pre::Window::Stretch, ..spec };
                let mut mp2 = cpu.model(&a[3], ModelKind::Tensor(spec_stretch))?;
                let _ = &mut mp;
                if let Output::Keypoints(c) = &mp2.run(&[Frame::Rgb8 { w: g.w, h: g.h, data: &win }])?[0] {
                    // c is in input pixels of a stretched window == the pipeline's placement when
                    // the window lies inside the frame (left = top = 0, nw = width)
                    if g.left == 0 && g.top == 0 && g.nw == spec.width && g.nh == spec.height {
                        for (p, q) in kps.iter().zip(c) {
                            let (qx, qy) = g.to_frame(q[0], q[1]);
                            if p[2] >= 0.5 && q[2] >= 0.5 {
                                let dd = ((p[0] - qx).powi(2) + (p[1] - qy).powi(2)).sqrt();
                                kp_n += 1;
                                kp_far += (dd > 2.0) as usize;
                            }
                        }
                    }
                }
            }
        }
    }
    println!("vs CPU Model API: reid worst cosine {worst_reid:.5} over {n_reid}; clip worst cosine {worst_clip:.5} over {n_clip}; pose {kp_n} confident keypoints, {kp_far} moved > 2 px");
    let ok = worst_reid >= 0.995 && worst_clip >= 0.995 && kp_far * 200 <= kp_n.max(1);
    println!("person pipeline GPU == CPU: {ok}");
    Ok(())
}
