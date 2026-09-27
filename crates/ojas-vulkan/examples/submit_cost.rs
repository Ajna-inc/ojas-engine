//! Cost of an empty submit+wait and of a 1 MB upload on a Vulkan device.
//! `submit_cost [device index or name part]`
use ojas_core::KernelRuntime;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let g = ojas_vulkan::VkGpu::new(ojas_vulkan::select(std::env::args().nth(1).as_deref())?)?;
    let med = |f: &mut dyn FnMut()| {
        let mut t: Vec<f64> = (0..50).map(|_| { let s = Instant::now(); f(); s.elapsed().as_secs_f64() * 1e6 }).collect();
        t.sort_by(|a, b| a.total_cmp(b));
        t[25]
    };
    let empty = med(&mut || g.submit(g.begin()).unwrap());
    let mut b = g.alloc_bytes(1 << 20)?;
    let data = vec![7u8; 1 << 20];
    let up = med(&mut || g.write_bytes(&mut b, 0, &data).unwrap());
    let mut out = vec![0u8; 1 << 20];
    let down = med(&mut || g.read_bytes(&b, 0, &mut out).unwrap());
    println!("{}: empty submit {empty:.0} µs, 1 MB upload {up:.0} µs, 1 MB read {down:.0} µs", g.info().name);
    Ok(())
}
