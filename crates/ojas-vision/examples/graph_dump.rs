//! Print the optimized IR of an ONNX model: one line per node with op, input
//! and output shapes. `graph_dump model.onnx [batch]`
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1])?;
    let model = ojas_formats::onnx::parse(&bytes)?;
    let mut binds = HashMap::new();
    if let Some(b) = args.get(2) {
        binds.insert("batch".to_string(), b.parse()?);
    }
    let mut g = ojas_vision::import(&model, &binds)?;
    let stats = ojas_vision::passes::optimize(&mut g);
    eprintln!("{stats:?}");
    let mut kinds: HashMap<String, usize> = HashMap::new();
    for n in &g.nodes {
        let op = format!("{:?}", n.op);
        let short = op.split(|c| c == ' ' || c == '(' || c == '{').next().unwrap().to_string();
        *kinds.entry(short.clone()).or_default() += 1;
        let ins: Vec<String> = n.inputs.iter().map(|&i| format!("{:?}{}", g.shape(i), if g.is_weight(i) { "w" } else { "" })).collect();
        let outs: Vec<String> = n.outputs.iter().map(|&o| format!("{:?}", g.shape(o))).collect();
        println!("{:<60} {} -> {}", op.chars().take(60).collect::<String>(), ins.join(" "), outs.join(" "));
    }
    // conv variants: what a backend must implement (kind, kernel, stride, dilation, fused act)
    let mut convs: HashMap<String, usize> = HashMap::new();
    for n in &g.nodes {
        if let ojas_vision::ir::Op::Conv { group, strides, dilations, act, .. } = &n.op {
            let w = g.shape(n.inputs[1]);
            let kind = if *group > 1 && w[1] == 1 { "depthwise" } else if *group > 1 { "grouped" } else { "dense" };
            let key = format!("{kind} {}x{} s{} d{} act {:?}", w[2], w[3], strides[0], dilations[0], act);
            *convs.entry(key).or_default() += 1;
        }
    }
    let mut cv: Vec<_> = convs.into_iter().collect();
    cv.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in &cv {
        eprintln!("conv {n:4}  {k}");
    }
    let mut k: Vec<_> = kinds.into_iter().collect();
    k.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("{k:?}");
    Ok(())
}
