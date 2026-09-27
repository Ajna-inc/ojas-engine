//! CPU executor parity gate against onnxruntime golden dumps.
//!
//! 1. `scripts/release/onnx_reference.py dump model.onnx --out golden/dir`
//! 2. `cargo run -p ojas-vision --release --example cnn_gate -- model.onnx golden/dir`
//!
//! The gate feeds the same input the reference used (read back from the dump),
//! compares every intermediate tensor whose ONNX name survived optimization, and
//! holds the graph outputs to cosine ≥ 0.9999 and max-abs ≤ 1e-4. Exit code 0 =
//! pass.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use ojas_vision::exec_cpu::CpuExecutor;
use ojas_vision::ir::TensorKind;

fn sanitize(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || ".-_".contains(c) { c } else { '_' }).collect()
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    ensure!(bytes.len() % 4 == 0, "{}: ragged f32 file", path.display());
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

struct Stats {
    cosine: f64,
    max_abs: f64,
    rmse: f64,
    /// max |reference| — tolerances scale with value magnitude (box coords
    /// are pixel-scale; 1e-4 absolute only makes sense for logit-scale data).
    scale: f64,
}

impl Stats {
    fn tol(&self) -> f64 {
        1e-4_f64.max(2e-5 * self.scale)
    }
    fn pass(&self) -> bool {
        self.cosine >= 0.9999 && self.max_abs <= self.tol()
    }
}

fn compare(a: &[f32], b: &[f32]) -> Stats {
    let n = a.len();
    let (mut dot, mut na, mut nb, mut max_abs, mut sq, mut scale) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        if y.abs() > scale {
            scale = y.abs();
        }
        sq += (x - y) * (x - y);
    }
    let denom = (na * nb).sqrt();
    Stats { cosine: if denom == 0.0 { 1.0 } else { dot / denom }, max_abs, rmse: (sq / n as f64).sqrt(), scale }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: cnn_gate <model.onnx> <golden_dir> [dim=value ...]");
    }
    let (model_path, golden) = (&args[1], Path::new(&args[2]));

    let model = ojas_formats::onnx::load(model_path)?;
    let mut binds = HashMap::new();
    for b in ["batch", "N", "n"] {
        binds.insert(b.to_string(), 1usize);
    }
    for kv in &args[3..] {
        let (k, v) = kv.split_once('=').with_context(|| format!("expected dim=value, got {kv:?}"))?;
        binds.insert(k.to_string(), v.parse()?);
    }
    let t0 = std::time::Instant::now();
    let mut g = ojas_vision::import(&model, &binds)?;
    let stats = ojas_vision::passes::optimize(&mut g);
    println!(
        "import: {} nodes after passes (silu {}, bn folds {}, act fusions {}, dce {}) in {:.1} ms",
        g.nodes.len(),
        stats.silu,
        stats.bn_folds,
        stats.act_fusions,
        stats.dce,
        t0.elapsed().as_secs_f64() * 1e3
    );

    // Inputs come from the dump so both runtimes see identical bytes.
    let mut inputs = Vec::new();
    for &id in &g.inputs {
        let name = &g.tensors[id].name;
        let f = golden.join(format!("{}.f32", sanitize(name)));
        let data = read_f32(&f).with_context(|| format!("input {name} — run onnx_reference.py dump first"))?;
        ensure!(data.len() == g.tensors[id].numel(), "input {name}: {} vs expected {}", data.len(), g.tensors[id].numel());
        inputs.push(data);
    }
    let input_refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();

    // Reference intermediates on disk, keyed by sanitized name.
    let mut golden_files: HashMap<String, std::path::PathBuf> = HashMap::new();
    for e in std::fs::read_dir(golden)? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "f32") {
            golden_files.insert(p.file_stem().unwrap().to_string_lossy().into_owned(), p);
        }
    }

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let mut exec = CpuExecutor::new(&g, threads);
    let mut checked = 0usize;
    let mut worst: Option<(String, Stats)> = None;
    let mut failed_nodes: Vec<String> = Vec::new();
    let outs = exec.run_with_capture(&g, &input_refs, &mut |name, data| {
        if let Some(p) = golden_files.get(&sanitize(name)) {
            if let Ok(want) = read_f32(p) {
                if want.len() == data.len() {
                    let s = compare(data, &want);
                    checked += 1;
                    if !s.pass() {
                        failed_nodes.push(format!("{name}: cosine {:.6} max_abs {:.2e} rmse {:.2e}", s.cosine, s.max_abs, s.rmse));
                    }
                    if worst.as_ref().is_none_or(|(_, w)| s.cosine < w.cosine) {
                        worst = Some((name.to_string(), s));
                    }
                }
            }
        }
    })?;
    println!("intermediates compared: {checked}");
    if let Some((name, s)) = &worst {
        println!("worst intermediate: {name} (cosine {:.6}, max_abs {:.2e})", s.cosine, s.max_abs);
    }
    for f in failed_nodes.iter().take(10) {
        println!("DIVERGED: {f}");
    }

    // Graph outputs: the hard gate.
    let mut pass = true;
    for (i, (&oid, got)) in g.outputs.iter().zip(&outs).enumerate() {
        let name = &g.tensors[oid].name;
        let p = golden_files
            .get(&sanitize(name))
            .with_context(|| format!("golden dump missing output {name}"))?;
        let want = read_f32(p)?;
        ensure!(want.len() == got.len(), "output {name}: {} vs {}", got.len(), want.len());
        let s = compare(got, &want);
        let ok = s.pass();
        pass &= ok;
        println!(
            "output[{i}] {name} ({:?} {:?}): cosine {:.7} max_abs {:.3e} (tol {:.1e}) rmse {:.3e} — {}",
            g.tensors[oid].kind,
            g.shape(oid),
            s.cosine,
            s.max_abs,
            s.tol(),
            s.rmse,
            if ok { "PASS" } else { "FAIL" }
        );
        let _ = TensorKind::Value;
    }
    if !pass {
        bail!("parity gate FAILED");
    }
    println!("parity gate PASSED");
    Ok(())
}
