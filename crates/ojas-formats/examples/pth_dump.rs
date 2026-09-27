//! List a PyTorch checkpoint's tensors: `pth_dump file.pth [prefix]`.
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1])?;
    let t = std::time::Instant::now();
    let ts = ojas_formats::pth::load(&bytes)?;
    let prefix = a.get(2).map(String::as_str).unwrap_or("");
    let mut tops: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    for (n, x) in &ts {
        let top = n.split('.').next().unwrap_or("").to_string();
        let e = tops.entry(top).or_default();
        e.0 += 1;
        e.1 += x.data.len();
        if !prefix.is_empty() && n.starts_with(prefix) {
            println!("{n:70} {:?}", x.shape);
        }
    }
    for (k, (n, p)) in tops {
        eprintln!("{k}: {n} tensors, {p} values");
    }
    eprintln!("loaded in {:.2} s", t.elapsed().as_secs_f64());
    Ok(())
}
