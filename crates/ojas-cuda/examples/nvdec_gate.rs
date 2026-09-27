//! NVDEC vs a software decode, byte for byte, then decode throughput.
//!
//! `nvdec_gate <stream.h264|.h265> <h264|hevc> <ref.nv12> [chunk bytes]`
//! (ref.nv12: `ffmpeg -i stream -f rawvideo -pix_fmt nv12 ref.nv12`)

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use ojas_cuda::nvdec::{Codec, NvDecoder};
use ojas_cuda::CudaGpu;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let stream = std::fs::read(&a[1])?;
    let codec = if a[2] == "hevc" { Codec::Hevc } else { Codec::H264 };
    let reference = std::fs::read(&a[3])?;
    let chunk: usize = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(4096);
    let gpu = Arc::new(CudaGpu::new(0)?);

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
