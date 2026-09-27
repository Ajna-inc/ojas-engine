//! Raw fp16 `mma` throughput ceiling: every lane loads its registers once and
//! then runs `reps` m16n8k16 instructions. Reports multiplies per second
//! (64 per lane instruction) for several block sizes and grid shapes.

use std::time::Instant;

use ojas_core::{Device, KernelRuntime};
use ojas_cuda::CudaGpu;

const SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void rate(const float* in, float* out, int reps, int mode) {
    int tid = threadIdx.x, lane = tid & 31;
    __shared__ __half sA[16 * 32];
    __shared__ __half sB[16 * 8];
    for (int idx = tid; idx < 16 * 32; idx += 32) sA[idx] = __float2half(in[idx % 64]);
    for (int idx = tid; idx < 16 * 8; idx += 32) sB[idx] = __float2half(in[idx % 64]);
    __syncthreads();
    unsigned a0, a1, a2, a3, b0, b1;
    const __half* aptr = &sA[(lane % 16) * 32 + (lane / 16) * 8];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(aptr));
    const __half* bptr = &sB[(lane % 16) * 8];
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];"
        : "=r"(b0), "=r"(b1) : "l"(bptr));
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int r = 0; r < reps; r++) {
        if (mode >= 1) asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];"
            : "=r"(b0), "=r"(b1) : "l"(bptr));
        if (mode >= 2) asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(aptr));
        if (mode != 3) asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    if (lane == 0) out[blockIdx.x] = c0 + c1 + c2 + c3;
}
"#;

fn main() -> anyhow::Result<()> {
    let g = CudaGpu::new(0)?;
    let f = g.pipeline(SRC, "rate")?;
    let input: Vec<f32> = (0..64).map(|i| (i % 7) as f32 * 0.1).collect();
    let ind = g.upload(&input);
    for (mode, name) in [(0u32, "mma only"), (1, "B load + mma"), (2, "A+B load + mma"), (3, "B+A loads only")] {
    for (block, blocks, reps) in [(256u32, 128u32, 256u32), (256, 1024, 64)] {
        let od = g.upload(&vec![0.0; blocks as usize]);
        let go = || {
            g.dispatch_pipeline(&f, &[(&ind, 0), (&od, 0)], &[reps, mode], [blocks, 1, 1], [block, 1, 1], 0).unwrap();
            g.submit(g.begin()).unwrap();
        };
        go();
        let t = Instant::now();
        for _ in 0..5 { go(); }
        let dt = t.elapsed().as_secs_f64() / 5.0;
        let mults = blocks as f64 * block as f64 * reps as f64 * 64.0;
        println!("{name:<16} block {block:>3} x {blocks:>5} blocks, {reps:>4} reps: {:>8.3} ms  -> {:>6.2} T mults/s-equiv", dt * 1e3, mults / dt / 1e12);
    }
    }
    Ok(())
}
