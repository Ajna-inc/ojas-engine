//! Matrix-unit ceiling: vk_gemm_cm_f16 on M = N = K = size (multiples of 128).
//! `gemm_bench <device> [size]`
use ojas_core::KernelRuntime;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let s: usize = a.get(2).map(|v| v.parse().unwrap()).unwrap_or(2048);
    let mut g = ojas_vulkan::VkGpu::new(ojas_vulkan::select(Some(&a[1]))?)?;
    g.ensure_family("vk")?;
    let block = 4 * g.info().subgroup;
    let (am, bm, cm) = (g.alloc_bytes(s * s * 2)?, g.alloc_bytes(s * s * 2)?, g.alloc_bytes(s * s * 4)?);
    let run = |reps: usize| -> anyhow::Result<f64> {
        let enc = g.begin();
        let t = Instant::now();
        for _ in 0..reps {
            g.dispatch(&enc, "vk_gemm_cm_f16", &[(&am, 0), (&bm, 0), (&cm, 0)], &[s as u32; 3], [(s / 128) as u32, (s / 128) as u32, 1], [block, 1, 1])?;
        }
        g.submit(enc)?;
        Ok(t.elapsed().as_secs_f64() / reps as f64)
    };
    let t0 = Instant::now();
    while t0.elapsed().as_millis() < 400 {
        run(5)?;
    }
    let mut ts: Vec<f64> = (0..7).map(|_| run(10).unwrap()).collect();
    ts.sort_by(|a, b| a.total_cmp(b));
    let t = ts[3];
    println!("{} gemm {s}^3: {:.0} µs = {:.1} TFLOP/s", g.info().name, t * 1e6, 2.0 * (s as f64).powi(3) / t / 1e12);
    Ok(())
}
