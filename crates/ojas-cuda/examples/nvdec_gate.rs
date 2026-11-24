//! NVDEC vs a software decode, byte for byte, then decode throughput.
//!
//! `nvdec_gate <stream.h264|.h265> <h264|hevc> <ref.nv12> [chunk bytes]`
//! (ref.nv12: `ffmpeg -i stream -f rawvideo -pix_fmt nv12 ref.nv12`)
//!
//! Decoder-scaled output vs a full-size decode resized on the host:
//!
//! `nvdec_gate <stream> <h264|hevc> scaled <W>x<H> [stretch|letterbox] [min PSNR dB]`
//!
//! The two scalers are different filters, so this is a closeness check (PSNR
//! of the picture in RGB, default floor 24 dB), plus exact checks of the
//! surface layout: size, pitch, chroma offset, the mapping back to the stream.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{ensure, Result};
use ojas_cuda::nvdec::{output_geom, Codec, DecodedFrame, DecoderOptions, Fit, NvDecoder, OutputSize};
use ojas_cuda::CudaGpu;
use ojas_vision::pre::{nv12_to_rgb8, resize_bilinear_rgb8};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let stream = std::fs::read(&a[1])?;
    let codec = if a[2] == "hevc" { Codec::Hevc } else { Codec::H264 };
    let gpu = Arc::new(CudaGpu::new(0)?);
    if a[3] == "scaled" {
        let (w, h) = a[4].split_once('x').ok_or_else(|| anyhow::anyhow!("size {}: expected WxH", a[4]))?;
        let fit = if a.get(5).is_some_and(|s| s == "letterbox") { Fit::Letterbox } else { Fit::Stretch };
        let floor: f64 = a.get(6).map(|s| s.parse()).transpose()?.unwrap_or(24.0);
        return scaled(gpu, &stream, codec, OutputSize { width: w.parse()?, height: h.parse()?, fit }, floor);
    }
    let reference = std::fs::read(&a[3])?;
    let chunk: usize = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(4096);

    // correctness: arbitrary chunking, every frame compared as it arrives
    let mut dec = NvDecoder::new(gpu.clone(), codec, 8)?;
    let (mut frames, mut bad_frames, mut bad_bytes) = (0usize, 0usize, 0usize);
    let pieces: Vec<&[u8]> = stream.chunks(chunk).collect();
    for (i, piece) in pieces.iter().enumerate() {
        let eos = i + 1 == pieces.len();
        for f in dec.decode(piece, i as i64, eos)? {
            let size = f.w * f.h * 3 / 2;
            let mut got = vec![0u8; size];
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(got.as_mut_ptr() as *mut _, f.ptr, size);
            }
            let want = &reference[frames * size..(frames + 1) * size];
            let diff = got.iter().zip(want).filter(|(x, y)| x != y).count();
            if diff > 0 {
                bad_frames += 1;
                bad_bytes += diff;
            }
            frames += 1;
        }
    }
    let (w, h) = dec.size();
    println!("{} {}x{}: {frames} frames decoded, reference has {}; {bad_frames} frames differ ({bad_bytes} bytes)",
             a[2], w, h, reference.len() / (w * h * 3 / 2));

    // throughput: whole stream, access-unit-agnostic 64 KB chunks
    let reps = 5;
    let t = Instant::now();
    let mut n = 0;
    for _ in 0..reps {
        let mut d = NvDecoder::new(gpu.clone(), codec, 8)?;
        let pieces: Vec<&[u8]> = stream.chunks(65536).collect();
        for (i, p) in pieces.iter().enumerate() {
            n += d.decode(p, 0, i + 1 == pieces.len())?.len();
        }
    }
    let s = t.elapsed().as_secs_f64();
    println!("decode throughput: {n} frames in {s:.2} s = {:.0} frames/s ({w}x{h}, one decoder)", n as f64 / s);
    Ok(())
}

/// A decoded frame's NV12 bytes on the host (frames are only valid until the ring wraps).
fn download(gpu: &CudaGpu, f: &DecodedFrame) -> Result<(DecodedFrame, Vec<u8>)> {
    // an odd height has a half chroma row less on the device than the conversion indexes
    let mut nv = vec![128u8; f.uv_off + f.pitch * f.h.div_ceil(2)];
    gpu.read_ptr(f.ptr, &mut nv[..f.uv_off + f.pitch * (f.h / 2)])?;
    Ok((*f, nv))
}

/// The same stream through a full-size decoder and a scaling one: every scaled frame
/// against the full-size frame resized on the host (cv2-linear) to the picture rectangle.
fn scaled(gpu: Arc<CudaGpu>, stream: &[u8], codec: Codec, out: OutputSize, floor: f64) -> Result<()> {
    let mut full = NvDecoder::new(gpu.clone(), codec, 8)?;
    let mut small = NvDecoder::with(gpu.clone(), codec, DecoderOptions { output: Some(out), ..Default::default() })?;
    let (mut qa, mut qb) = (VecDeque::new(), VecDeque::new());
    let (mut frames, mut sum_psnr, mut worst, mut sum_abs) = (0usize, 0.0f64, f64::INFINITY, 0.0f64);
    let (mut bar_lo, mut bar_hi) = (255u8, 0u8);
    let pieces: Vec<&[u8]> = stream.chunks(4096).collect();
    let t = Instant::now();
    for (i, piece) in pieces.iter().enumerate() {
        let eos = i + 1 == pieces.len();
        for f in full.decode(piece, i as i64, eos)? {
            qa.push_back(download(&gpu, &f)?);
        }
        for f in small.decode(piece, i as i64, eos)? {
            qb.push_back(download(&gpu, &f)?);
        }
        while !qa.is_empty() && !qb.is_empty() {
            let ((fa, na), (fb, nb)) = (qa.pop_front().unwrap(), qb.pop_front().unwrap());
            let g = small.geom().ok_or_else(|| anyhow::anyhow!("frames before a sequence header"))?;
            // layout: the surface is the asked size, tightly packed, and remembers the stream's
            ensure!(g == output_geom(fa.w, fa.h, Some(out)), "geometry {g:?} for a {}x{} stream", fa.w, fa.h);
            ensure!(full.size() == (g.src_w, g.src_h) && small.size() == (g.w, g.h), "sizes {:?} / {:?} vs {g:?}", full.size(), small.size());
            ensure!((fb.w, fb.h, fb.pitch, fb.uv_off) == (g.w, g.h, g.w, g.w * g.h), "scaled frame {fb:?} vs {g:?}");
            ensure!(fa.timestamp == fb.timestamp, "frame order: timestamps {} vs {}", fa.timestamp, fb.timestamp);
            let [x0, y0, x1, y1] = g.rect;
            let (cw, ch) = (x1 - x0, y1 - y0);
            let want = resize_bilinear_rgb8(&nv12_to_rgb8(&na, fa.w, fa.h, fa.pitch, fa.uv_off), fa.w, fa.h, cw, ch, 3);
            let got = nv12_to_rgb8(&nb, fb.w, fb.h, fb.pitch, fb.uv_off);
            let (mut se, mut abs) = (0u64, 0u64);
            for y in 0..ch {
                let row = &got[((y + y0) * fb.w + x0) * 3..((y + y0) * fb.w + x1) * 3];
                for (p, q) in row.iter().zip(&want[y * cw * 3..(y + 1) * cw * 3]) {
                    let d = (*p as i64 - *q as i64).unsigned_abs();
                    se += d * d;
                    abs += d;
                }
            }
            let n = (cw * ch * 3) as f64;
            let psnr = if se == 0 { 99.0 } else { 10.0 * (255.0 * 255.0 * n / se as f64).log10() };
            sum_psnr += psnr;
            sum_abs += abs as f64 / n;
            worst = worst.min(psnr);
            // letterbox bars: luma outside the picture (reported, not judged)
            for y in 0..fb.h {
                for x in 0..fb.w {
                    if x < x0 || x >= x1 || y < y0 || y >= y1 {
                        let v = nb[y * fb.pitch + x];
                        (bar_lo, bar_hi) = (bar_lo.min(v), bar_hi.max(v));
                    }
                }
            }
            frames += 1;
        }
    }
    ensure!(frames > 0, "no frames decoded");
    ensure!(qa.is_empty() && qb.is_empty(), "frame counts differ: {} full-size / {} scaled left over", qa.len(), qb.len());
    let g = small.geom().unwrap();
    println!("{}x{} -> {}x{} {:?}, picture at {:?}: {frames} frames in {:.1} s", g.src_w, g.src_h, g.w, g.h, out.fit, g.rect, t.elapsed().as_secs_f64());
    println!("decoder-scaled vs full-size decode + host resize (RGB): PSNR mean {:.2} dB, worst {worst:.2} dB, mean |diff| {:.2}", sum_psnr / frames as f64, sum_abs / frames as f64);
    if bar_lo <= bar_hi {
        println!("bars: luma {bar_lo}..{bar_hi}");
    }
    ensure!(worst >= floor, "worst frame {worst:.2} dB is below the {floor} dB floor");
    println!("ok");
    Ok(())
}
