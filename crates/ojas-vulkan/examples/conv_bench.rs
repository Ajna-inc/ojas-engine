//! One convolution on a Vulkan device: time of each kernel variant (20
//! dispatches per submit, median of 7) and their agreement.
//! `conv_bench <device> <n> <cin> <h> <w> <cout> <k> <stride>`
use ojas_core::conv::{ConvGeom, Storage};
use ojas_core::{Device, KernelRuntime};
use ojas_vulkan::conv::{VkConvPlan, VkConvVariant};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let v: Vec<usize> = a[2..9].iter().map(|s| s.parse().unwrap()).collect();
    let (n, cin, h, w, cout, k, s) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    let mut g = ojas_vulkan::VkGpu::new(ojas_vulkan::select(Some(&a[1]))?)?;
    g.ensure_family("cnn")?;
    let geom = ConvGeom::square(cin, h, w, cout, k, s, k / 2);
    let (oh, ow) = geom.out_hw();
    let ist = Storage { c: cin, cs: cin, h, w, pad: 1.max(k / 2), spare: 1 };
    let ost = Storage { c: cout, cs: cout, h: oh, w: ow, pad: 1, spare: 1 };
    let mut r = 0x9e37u32;
    let mut rnd = || { r ^= r << 13; r ^= r >> 17; r ^= r << 5; (r % 2001) as f32 / 1000.0 - 1.0 };
    let wt: Vec<f32> = (0..cout * cin * k * k).map(|_| rnd() * 0.1).collect();
    let bias: Vec<f32> = (0..cout).map(|_| rnd()).collect();
    let x: Vec<f32> = (0..n * ist.img()).map(|_| rnd()).collect();
    let xd = g.upload_f16(&x);
    let mut plan = VkConvPlan::with_storage(&g, geom, &wt, Some(&bias), 1, ist, ost)?;
    let flops = 2.0 * (n * oh * ow) as f64 * cout as f64 * (cin * k * k) as f64;
    // bring the GPU to its boost clock first (short bursts run at idle clocks)
    {
        let yd = g.upload_f16(&vec![0.0; n * ost.img()]);
        let t = Instant::now();
        while t.elapsed().as_millis() < 400 {
            let enc = g.begin();
            for _ in 0..20 {
                plan.dispatch_at(&g, &enc, (&xd, 0), (&yd, 0), n)?;
            }
            g.submit(enc)?;
        }
    }
    let mut outs = vec![];
    for var in plan.variants() {
        plan.variant = var;
        let yd = g.upload_f16(&vec![0.0; n * ost.img()]);
        let mut ts = vec![];
        for _ in 0..8 {
            let enc = g.begin();
            let t = Instant::now();
            for _ in 0..20 {
                plan.dispatch_at(&g, &enc, (&xd, 0), (&yd, 0), n)?;
            }
            g.submit(enc)?;
            ts.push(t.elapsed().as_secs_f64() / 20.0);
        }
        ts.remove(0);
        ts.sort_by(|a, b| a.total_cmp(b));
        let t = ts[ts.len() / 2];
        let mut y = vec![0.0f32; n * ost.img()];
        g.read(&yd, &mut y);
        println!("{var:?}: {:8.1} µs  {:6.2} TFLOP/s", t * 1e6, flops / t / 1e12);
        outs.push((var, y));
    }
    if outs.len() == 2 {
        let (a, b) = (&outs[0].1, &outs[1].1);
        let worst = a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        println!("max |{:?} - {:?}| = {worst:.4}", outs[0].0, outs[1].0);
    }
    let _ = VkConvVariant::Tiled;
    Ok(())
}
