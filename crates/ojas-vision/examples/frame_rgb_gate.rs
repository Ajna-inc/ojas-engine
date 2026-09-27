//! `gpu_pre::frame_to_rgb8` on NVDEC frames vs the host reference
//! (`pre::nv12_to_rgb8`, cv2 BT.601), full size and strided thumbnails.
//!
//! `frame_rgb_gate stream.h264`

use std::sync::Arc;

use anyhow::Result;
use ojas_cuda::nvdec::{Codec, NvDecoder};
use ojas_cuda::CudaGpu;
use ojas_vision::gpu_pre::{frame_to_rgb8, DevFrame, PixFmt};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let stream = std::fs::read(&a[1])?;
    let mut gpu = CudaGpu::new(0)?;
    ojas_core::KernelRuntime::ensure_family(&mut gpu, "cnn")?;
    let gpu = Arc::new(gpu);
    let mut dec = NvDecoder::new(gpu.clone(), Codec::H264, 64)?;
    let mut frames = vec![];
    for piece in stream.chunks(4096) {
        frames.extend(dec.decode(piece, 0, false)?);
        if frames.len() >= 10 {
            break;
        }
    }
    let (mut scratch, mut out, mut bad, mut n) = (None, vec![], 0usize, 0usize);
    for f in &frames {
        let df = DevFrame { ptr: f.ptr, pitch: f.pitch, w: f.w, h: f.h, fmt: PixFmt::Nv12 { uv_off: f.uv_off } };
        let mut nv = vec![0u8; f.pitch * f.h * 3 / 2];
        gpu.read_ptr(f.ptr, &mut nv)?;
        let host = ojas_vision::pre::nv12_to_rgb8(&nv, f.w, f.h, f.pitch, f.uv_off);
        for step in [1usize, 2, 4] {
            let (ow, oh) = frame_to_rgb8(&*gpu, &df, step, &mut scratch, &mut out)?;
            for y in 0..oh {
                for x in 0..ow {
                    for c in 0..3 {
                        n += 1;
                        bad += (out[(y * ow + x) * 3 + c] != host[((y * step) * f.w + x * step) * 3 + c]) as usize;
                    }
                }
            }
        }
    }
    println!("{} frames x steps 1/2/4: {bad} of {n} values differ", frames.len());
    anyhow::ensure!(bad == 0, "mismatch");
    Ok(())
}
