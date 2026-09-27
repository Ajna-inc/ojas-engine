//! Run any ONNX model through ojas on one raw input and compare with reference
//! outputs: the gate for bringing up a new model family.
//!
//! `graph_run model.onnx input.f32 <prefix> [dim=value ...]` — input: raw little-endian f32 of
//! the model's (single) input. Writes `<prefix>.ojas.<i>.f32` per output and,
//! when `<prefix>.ref.<i>.f32` exists (onnxruntime / PyTorch), prints the
//! largest difference. `OJAS_DEVICE=cuda:0` runs the CUDA executor (needs
//! `--features cuda`); default: the CPU executor.
use std::collections::HashMap;

fn read_f32(path: &str) -> anyhow::Result<Vec<f32>> {
    let b = std::fs::read(path)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 4, "graph_run model.onnx input.f32 <prefix> [dim=value ...]");
    let model = ojas_formats::onnx::parse(&std::fs::read(&args[1])?)?;
    // symbolic input dims (`batch_size=1 height=224 ...`)
    let binds: HashMap<String, usize> = args[4..].iter().filter_map(|kv| kv.split_once('=').and_then(|(k, v)| v.parse().ok().map(|v| (k.to_string(), v)))).collect();
    let mut g = ojas_vision::import(&model, &binds)?;
    ojas_vision::passes::optimize(&mut g);
    let input = read_f32(&args[2])?;
    let want: usize = g.shape(g.inputs[0]).iter().product();
    anyhow::ensure!(input.len() == want, "input has {} values, the model wants {want} ({:?})", input.len(), g.shape(g.inputs[0]));
    let device = std::env::var("OJAS_DEVICE").unwrap_or_else(|_| "cpu".into());
    let t0 = std::time::Instant::now();
    let outs = match device.as_str() {
        "cpu" => ojas_vision::exec_cpu::CpuExecutor::new(&g, std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)).run(&g, &[&input])?,
        #[cfg(feature = "cuda")]
        d if d.starts_with("cuda") => {
            let ordinal = d.split_once(':').map(|(_, i)| i.parse()).transpose()?.unwrap_or(0);
            ojas_vision::passes::lower_for_gpu(&mut g);
            let mut ex = ojas_vision::exec_gpu::CudaExecutor::new(&g, ordinal)?;
            ex.run(&input)?; // warm-up (kernels compile, buffers settle)
            let t = std::time::Instant::now();
            let o = ex.run(&input)?;
            eprintln!("cuda run: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
            o
        }
        other => anyhow::bail!("OJAS_DEVICE {other:?}: cpu or cuda:N (with --features cuda)"),
    };
    eprintln!("{device}: {:.1} ms total", t0.elapsed().as_secs_f64() * 1e3);
    for (i, (o, &id)) in outs.iter().zip(&g.outputs).enumerate() {
        let path = format!("{}.ojas.{i}.f32", args[3]);
        std::fs::write(&path, o.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let reference = format!("{}.ref.{i}.f32", args[3]);
        let diff = match read_f32(&reference) {
            Ok(r) if r.len() == o.len() => format!("max |Δ| vs reference {:.5}", o.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)),
            Ok(r) => format!("reference has {} values, ours {}", r.len(), o.len()),
            Err(_) => "no reference".into(),
        };
        println!("output {i} {:?}: {diff}", g.shape(id));
    }
    // DETR-style outputs (logits [1,Q,C], boxes [1,Q,4] cxcywh): compare detections, not rows
    // (in f16 near-equal query scores can select or order differently)
    if outs.len() == 2 && g.shape(g.outputs[1]).last() == Some(&4) {
        let c = *g.shape(g.outputs[0]).last().unwrap();
        let dets = |logits: &[f32], boxes: &[f32]| -> Vec<(usize, f32, [f32; 4])> {
            let mut d = vec![];
            for q in 0..boxes.len() / 4 {
                for k in 0..c {
                    let s = 1.0 / (1.0 + (-logits[q * c + k]).exp());
                    if s > 0.5 {
                        d.push((k, s, [boxes[q * 4], boxes[q * 4 + 1], boxes[q * 4 + 2], boxes[q * 4 + 3]]));
                    }
                }
            }
            d
        };
        let iou = |a: &[f32; 4], b: &[f32; 4]| {
            let r = |v: &[f32; 4]| [v[0] - v[2] / 2.0, v[1] - v[3] / 2.0, v[0] + v[2] / 2.0, v[1] + v[3] / 2.0];
            let (a, b) = (r(a), r(b));
            let i = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
            i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)
        };
        let ours = dets(&outs[0], &outs[1]);
        if let (Ok(rl), Ok(rb)) = (read_f32(&format!("{}.ref.0.f32", args[3])), read_f32(&format!("{}.ref.1.f32", args[3]))) {
            let refs = dets(&rl, &rb);
            let mut worst = (1.0f32, 0.0f32);
            let matched = refs
                .iter()
                .filter(|r| {
                    ours.iter().filter(|o| o.0 == r.0).map(|o| (iou(&o.2, &r.2), (o.1 - r.1).abs())).max_by(|a, b| a.0.total_cmp(&b.0)).is_some_and(|(i, ds)| {
                        worst = (worst.0.min(i), worst.1.max(ds));
                        i > 0.9
                    })
                })
                .count();
            println!("detections > 0.5: reference {}, ours {}, matched (same class, IoU > 0.9) {matched}; worst IoU {:.4}, worst |Δscore| {:.4}", refs.len(), ours.len(), worst.0, worst.1);
        } else {
            println!("detections > 0.5: {}", ours.len());
        }
    }
    Ok(())
}
