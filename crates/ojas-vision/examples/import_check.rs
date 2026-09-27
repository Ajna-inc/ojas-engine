//! Imports an ONNX export into the IR and reports what it became: the bound input
//! shape, the op histogram after optimisation and the outputs, or the first error.
//!
//! `import_check model.onnx [dim=value ...]`   e.g. `import_check vit.onnx batch_size=1 height=224 width=224`
use std::collections::{BTreeMap, HashMap};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 2, "import_check model.onnx [dim=value ...]");
    let binds: HashMap<String, usize> = a[2..].iter().filter_map(|kv| kv.split_once('=').and_then(|(k, v)| v.parse().ok().map(|v| (k.to_string(), v)))).collect();
    let model = ojas_formats::onnx::load(&a[1])?;
    let t = std::time::Instant::now();
    let mut g = ojas_vision::import(&model, &binds)?;
    let n0 = g.nodes.len();
    ojas_vision::passes::optimize(&mut g);
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for n in &g.nodes {
        let name = format!("{:?}", n.op);
        *hist.entry(name.split(|c: char| c == ' ' || c == '(' || c == '{').next().unwrap().to_string()).or_default() += 1;
    }
    println!("imported in {:.0} ms: {n0} nodes → {} after optimisation", t.elapsed().as_secs_f64() * 1e3, g.nodes.len());
    for &i in &g.inputs {
        println!("input  {} {:?}", g.tensors[i].name, g.shape(i));
    }
    for &o in &g.outputs {
        println!("output {} {:?}", g.tensors[o].name, g.shape(o));
    }
    for (k, v) in &hist {
        println!("  {k:<16}{v:>5}");
    }
    Ok(())
}
