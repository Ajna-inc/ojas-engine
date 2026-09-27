//! Memory bandwidth through ojas-vulkan buffers: a pure f16 copy kernel
//! (cnn_axis_copy_f16) over `mb` MB, 20 dispatches per submit.
//! `bw_bench <device> [mb]`
use ojas_core::KernelRuntime;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let mb: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(64);
    let mut g = ojas_vulkan::VkGpu::new(ojas_vulkan::select(Some(&a[1]))?)?;
    g.ensure_family("cnn")?;
    g.ensure_family("vk")?;
    let n = mb * (1 << 20) / 2;
    let x = g.alloc_bytes(n * 2)?;
    let y = g.alloc_bytes(n * 2)?;
    let cases: Vec<(&str, Vec<u32>, u32)> = vec![
        ("cnn_axis_copy_f16 (index math)", vec![n as u32, 1, 1, 1, 0, 1, 0], n as u32),
        ("vk_copy_f16 (2 B/invocation)", vec![n as u32], n as u32),
        ("vk_copy16_f16 (16 B/invocation)", vec![(n / 8) as u32], (n / 8) as u32),
    ];
    for (name, consts, items) in &cases {
    let block = 256u32;
    let kname = name.split(' ').next().unwrap();
        let mut ts = vec![];
        for _ in 0..6 {
            let enc = g.begin();
            let t = Instant::now();
            for _ in 0..20 {
                g.dispatch(&enc, kname, &[(&x, 0), (&y, 0)], consts, [items.div_ceil(block), 1, 1], [block, 1, 1])?;
            }
            g.submit(enc)?;
            ts.push(t.elapsed().as_secs_f64() / 20.0);
        }
        ts.remove(0);
        ts.sort_by(|a, b| a.total_cmp(b));
        let t = ts[ts.len() / 2];
        println!("{name:34} {mb} MB in {:6.0} µs = {:4.0} GB/s (read + write)", t * 1e6, 2.0 * (n * 2) as f64 / t / 1e9);
    }
    Ok(())
}
