//! Many cameras on one GPU's decode engine: C looping decoders, one frame
//! each per round, either decoded camera by camera (`decode`: map in turn)
//! or fed together and collected behind one sync (`feed` + `collect_all`).
//!
//! `nvdec_multi <stream> <h264|hevc> <cameras> <seconds> [keep_every] [key]`
//! (`key`: keyframe-only decode)

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use ojas_cuda::nvdec::{collect_all, Codec, Mode, NvDecoder};
use ojas_cuda::CudaGpu;

const PIECE: usize = 16384;

/// Feed pieces until a picture is queued (the stream loops).
fn feed_one(dec: &mut NvDecoder, pos: &mut usize, s: &[u8]) -> Result<()> {
    while dec.pending() == 0 {
        if *pos >= s.len() {
            *pos = 0;
        }
        let end = (*pos + PIECE).min(s.len());
        dec.feed(&s[*pos..end], 0, false)?;
        *pos = end;
    }
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let stream = std::fs::read(&a[1])?;
    let codec = if a[2] == "hevc" { Codec::Hevc } else { Codec::H264 };
    let (cams, secs): (usize, f64) = (a[3].parse()?, a[4].parse()?);
    let keep: u32 = a.get(5).map_or(Ok(1), |v| v.parse())?;
    let dmode = if a.get(6).is_some_and(|m| m == "key") { Mode::Keyframes } else { Mode::All };
    let gpu = Arc::new(CudaGpu::new(0)?);
    for mode in ["sequential", "batched"] {
        let mut decs: Vec<NvDecoder> = (0..cams)
            .map(|_| NvDecoder::with_options(gpu.clone(), codec, 8, keep).map(|mut d| { d.set_mode(dmode); d }))
            .collect::<Result<_>>()?;
        // cameras start at different points of the loop
        let mut pos: Vec<usize> = (0..cams).map(|i| (i * 7919 * PIECE / 3) % stream.len()).collect();
        let (t, mut frames) = (Instant::now(), 0usize);
        while t.elapsed().as_secs_f64() < secs {
            if mode == "sequential" {
                for (d, p) in decs.iter_mut().zip(pos.iter_mut()) {
                    feed_one(d, p, &stream)?;
                    frames += d.collect(1)?.len();
                }
            } else {
                for (d, p) in decs.iter_mut().zip(pos.iter_mut()) {
                    feed_one(d, p, &stream)?;
                }
                frames += collect_all(&mut decs, 1)?.iter().map(|v| v.len()).sum::<usize>();
            }
        }
        let el = t.elapsed().as_secs_f64();
        let pics: u64 = decs.iter().map(|d| d.pictures()).sum();
        println!("{mode:>10}: {cams} cameras, {frames} frames in {el:.1} s = {:.0} analysed frames/s, {:.0} stream frames/s \
                  (= {:.0} cameras at 15 fps; keep 1 in {keep}, {dmode:?})",
                 frames as f64 / el, pics as f64 / el, pics as f64 / el / 15.0);
    }
    Ok(())
}
