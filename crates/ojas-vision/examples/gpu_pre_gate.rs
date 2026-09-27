//! GPU preprocessing vs the CPU path, byte for byte (after the f16 rounding
//! the CPU input gets on upload): random frames, random crops, both
//! placements (letterbox RGB x/255 and OCR BGR signed).

use ojas_core::KernelRuntime;
use ojas_cuda::conv::Storage;
use ojas_cuda::CudaGpu;
use ojas_vision::gpu_pre::{descriptors, launch, DevFrame, PixFmt, Placement, Roi};
use ojas_vision::pre::{letterbox_rgb8_into, nv12_to_rgb8, ocr_resize_into, stretch_into, OcrNorm, window_geom, window_into, ChanNorm, Window};

fn main() -> anyhow::Result<()> {
    let mut g = CudaGpu::new(0)?;
    g.ensure_family("cnn")?;
    let mut s = 12345u64;
    let mut rnd = |n: usize| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 33) as usize) % n.max(1) };
    // RGB frames (one with row pitch > 3w) and NV12 frames (pitch > w, UV plane
    // after an aligned luma height), packed in one device buffer
    struct Src { w: usize, h: usize, pitch: usize, nv12: bool }
    let srcs = [
        Src { w: 1920, h: 1080, pitch: 1920 * 3 + 64, nv12: false },
        Src { w: 1280, h: 720, pitch: 1280 * 3, nv12: false },
        Src { w: 1920, h: 1080, pitch: 2048, nv12: true },
        Src { w: 640, h: 360, pitch: 640, nv12: true },
    ];
    let mut host = vec![];
    let mut specs = vec![]; // (offset, fmt)
    let mut rgb_frames = vec![];
    for (si, sd) in srcs.iter().enumerate() {
        let off = host.len();
        if sd.nv12 {
            let luma_rows = sd.h.div_ceil(16) * 16; // decoder surfaces align the luma height
            let uv_off = luma_rows * sd.pitch;
            let mut buf = vec![0u8; uv_off + (sd.h / 2) * sd.pitch];
            for (i, v) in buf.iter_mut().enumerate() { *v = (((i * 2654435761usize) >> 9) as u8).wrapping_add(si as u8 * 37); }
            rgb_frames.push(nv12_to_rgb8(&buf, sd.w, sd.h, sd.pitch, uv_off));
            specs.push((off, PixFmt::Nv12 { uv_off }));
            host.extend_from_slice(&buf);
        } else {
            let mut rgb = vec![0u8; sd.w * sd.h * 3];
            for (i, v) in rgb.iter_mut().enumerate() { *v = ((i * 2654435761usize) >> 7) as u8 ^ (i / (sd.w * 3)) as u8; }
            for y in 0..sd.h {
                host.extend_from_slice(&rgb[y * sd.w * 3..(y + 1) * sd.w * 3]);
                host.extend(std::iter::repeat(0).take(sd.pitch - sd.w * 3));
            }
            specs.push((off, PixFmt::Rgb8));
            rgb_frames.push(rgb);
        }
    }
    let fbuf = g.upload_bytes(&host)?;
    let base = g.device_ptr(&fbuf);
    let frames: Vec<DevFrame> = srcs.iter().zip(&specs)
        .map(|(sd, &(off, fmt))| DevFrame { ptr: base + off as u64, pitch: sd.pitch, w: sd.w, h: sd.h, fmt })
        .collect();
    let dims: Vec<(usize, usize)> = srcs.iter().map(|s| (s.w, s.h)).collect();
    let mut rois: Vec<Roi> = dims.iter().enumerate().map(|(i, &(w, h))| Roi { frame: i, x0: 0, y0: 0, w, h }).collect();
    for _ in 0..40 {
        let fi = rnd(dims.len());
        let (w, h) = dims[fi];
        let (cw, ch) = (8 + rnd(400), 8 + rnd(300));
        rois.push(Roi { frame: fi, x0: rnd(w - cw), y0: rnd(h - ch), w: cw, h: ch });
    }
    let crop = |r: &Roi| -> Vec<u8> {
        let (w, _) = dims[r.frame];
        let mut out = vec![];
        for y in r.y0..r.y0 + r.h { out.extend_from_slice(&rgb_frames[r.frame][(y * w + r.x0) * 3..(y * w + r.x0 + r.w) * 3]); }
        out
    };
    let imagenet = ChanNorm::mean_std([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    for place in [
        Placement::Letterbox { target: 384 },
        Placement::Ocr { height: 48, width: 320, norm: OcrNorm::Signed, bgr: true },
        Placement::Stretch { height: 64, width: 96, norm: OcrNorm::Unit, bgr: false },
        // the person models: OSNet's stretched box with ImageNet mean/std, RTMPose's 1.25× window with zero fill
        Placement::Norm { height: 256, width: 128, norm: imagenet, window: Window::Stretch, fill_u8: 0, bgr: false },
        Placement::Norm { height: 256, width: 192, norm: imagenet, window: Window::Around { scale: 1.25 }, fill_u8: 0, bgr: false },
    ] {
        let (tw, th) = match place { Placement::Letterbox { target } => (target, target), Placement::Ocr { height, width, .. } | Placement::Stretch { height, width, .. } | Placement::Norm { height, width, .. } => (width, height) };
        let st = Storage { c: 3, cs: 4, h: th, w: tw, pad: 1, spare: 1 };
        let (d, _) = descriptors(&frames, &rois, place)?;
        let dbuf = g.upload_bytes(&d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let out = g.alloc_bytes(rois.len() * st.img() * 2)?;
        let enc = g.begin();
        launch(&g, &enc, &dbuf, (&out, 0), &st, rois.len(), place)?;
        g.submit(enc)?;
        let mut raw = vec![0u8; rois.len() * st.img() * 2];
        g.read_bytes(&out, 0, &mut raw)?;
        let (mut bad, mut total, mut nv_total) = (0usize, 0usize, 0usize);
        for (i, r) in rois.iter().enumerate() {
            let c = crop(r);
            let mut cpu = vec![0.0f32; 3 * tw * th];
            match place {
                Placement::Letterbox { target } => { letterbox_rgb8_into(&c, r.w, r.h, target, &mut cpu); }
                Placement::Ocr { height, width, norm, bgr } => ocr_resize_into(&c, r.w, r.h, height, width, norm, bgr, &mut cpu),
                Placement::Stretch { height, width, norm, bgr } => stretch_into(&c, r.w, r.h, height, width, norm, bgr, &mut cpu),
                Placement::Norm { height, width, norm, window, fill_u8, bgr } => {
                    // the window may read beyond the ROI, so the CPU twin runs on the whole frame
                    let (fw, fh) = dims[r.frame];
                    let geom = window_geom([r.x0 as f32, r.y0 as f32, r.w as f32, r.h as f32], fw, fh, width, height, window);
                    window_into(&rgb_frames[r.frame], fw, fh, &geom, width, height, &norm, fill_u8, bgr, &mut cpu);
                }
            }
            for ch in 0..3 { for y in 0..th { for x in 0..tw {
                let want = half::f16::from_f32(cpu[(ch * th + y) * tw + x]).to_bits();
                let o = (i * st.img() + ((y + 1) * st.wp() + x + 1) * st.cs + ch) * 2;
                let got = u16::from_le_bytes([raw[o], raw[o + 1]]);
                total += 1;
                if srcs[r.frame].nv12 { nv_total += 1; }
                if got != want { bad += 1; }
            } } }
        }
        println!("{place:?}: {} crops, {total} values ({nv_total} from NV12 frames), {bad} differ", rois.len());
    }
    Ok(())
}
