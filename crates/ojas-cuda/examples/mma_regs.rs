//! What `ldmatrix` puts in each lane's registers. A[r][c] = 16r + c + 1 and
//! B[k][n] = 1000 + 16k + n are stored as f16; each lane's raw register words are
//! dumped and every 16-bit half inside is decoded back to its matrix cell.

use ojas_core::{Device, KernelRuntime};
use ojas_cuda::CudaGpu;

const SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void regs(const float* a, const float* b, unsigned* out) {
    int tid = threadIdx.x, lane = tid & 31;
    __shared__ __half sA[16 * 16];
    __shared__ __half sB[16 * 8];
    for (int idx = tid; idx < 256; idx += 32) sA[idx] = __float2half(a[idx]);
    for (int idx = tid; idx < 128; idx += 32) sB[idx] = __float2half(b[idx]);
    __syncthreads();
    unsigned a0, a1, a2, a3, b0, b1;
    const __half* aptr = &sA[(lane % 16) * 16 + (lane / 16) * 8];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(aptr));
    const __half* bptr = &sB[(lane % 16) * 8];
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];"
        : "=r"(b0), "=r"(b1) : "l"(bptr));
    out[lane * 6 + 0] = a0; out[lane * 6 + 1] = a1; out[lane * 6 + 2] = a2;
    out[lane * 6 + 3] = a3; out[lane * 6 + 4] = b0; out[lane * 6 + 5] = b1;
}
"#;

fn main() -> anyhow::Result<()> {
    let g = CudaGpu::new(0)?;
    let f = g.pipeline(SRC, "regs")?;
    let a: Vec<f32> = (0..256).map(|i| i as f32 + 1.0).collect();
    let b: Vec<f32> = (0..128).map(|i| 1000.0 + i as f32).collect();
    let (ad, bd) = (g.upload(&a), g.upload(&b));
    let od = g.upload(&vec![0.0; 32 * 6]);
    g.dispatch_pipeline(&f, &[(&ad, 0), (&bd, 0), (&od, 0)], &[], [1, 1, 1], [32, 1, 1], 0)?;
    g.submit(g.begin())?;
    let mut raw = vec![0.0f32; 32 * 6];
    g.read(&od, &mut raw);
    let cell = |w: u32, shift: u32| {
        let v = half::f16::from_bits((w >> shift) as u16).to_f32();
        if v >= 1000.0 { let i = (v - 1000.0) as usize; format!("B{},{}", i / 8, i % 8) }
        else if v >= 1.0 { let i = v as usize - 1; format!("A{},{}", i / 16, i % 16) }
        else { format!("{v}") }
    };
    for lane in [0usize, 1, 2, 3, 4, 15, 16, 17, 31] {
        let w: Vec<u32> = raw[lane * 6..lane * 6 + 6].iter().map(|v| v.to_bits()).collect();
        let s: Vec<String> = w.iter().map(|&x| format!("[{} {}]", cell(x, 0), cell(x, 16))).collect();
        println!("lane {lane:2}: A {} {} {} {} | B {} {}", s[0], s[1], s[2], s[3], s[4], s[5]);
    }
    Ok(())
}
