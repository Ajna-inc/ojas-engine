//! Data loader throughput alone: `loader_bench train.json root [workers] [batches]`
use std::sync::Arc;
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let workers: usize = a.get(3).map(|v| v.parse().unwrap()).unwrap_or(16);
    let n: usize = a.get(4).map(|v| v.parse().unwrap()).unwrap_or(50);
    let s = Arc::new(ojas_learn::data::load_coco(a[1].as_ref(), a[2].as_ref())?);
    let t = std::time::Instant::now();
    let one = ojas_learn::data::prepare(&s[0], 640, true, &mut ojas_learn::models::detr_loss::Rng::new(1))?;
    println!("one sample: {:.0} ms ({} boxes)", t.elapsed().as_secs_f64() * 1e3, one.1.labels.len());
    let d = std::time::Instant::now();
    let _ = ojas_learn::data::decode(&s[1].path)?;
    println!("decode alone: {:.0} ms", d.elapsed().as_secs_f64() * 1e3);
    let t = std::time::Instant::now();
    let mut k = 0;
    for b in ojas_learn::data::loader(s.clone(), (0..s.len()).collect(), 4, 640, true, workers, 1).into_iter().take(n) {
        k += b?.targets.len();
    }
    println!("{workers} workers: {k} images in {:.1} s = {:.1} img/s", t.elapsed().as_secs_f64(), k as f64 / t.elapsed().as_secs_f64());
    Ok(())
}
