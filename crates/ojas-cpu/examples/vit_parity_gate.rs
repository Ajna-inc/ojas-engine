//! Stage-by-stage parity check of the CPU ViT against llama.cpp.
//!
//! `cpu_vit.rs` is the numerical oracle the Metal ViT is validated against, so
//! its unit tests are not sufficient alone: they can all pass against a
//! transcription that is internally consistent and externally wrong. This gate
//! runs the same image through `VitPreproc` + `CpuVit::forward_tapped`, dumping
//! at exactly the points `clip_graph::cb()` fires, and compares against a
//! `reference_probe` `OJAS_REFERENCE_VISION_TRACE` directory.
//!
//! Run the reference with GPU offload. `ggml_flash_attn_ext` (Metal)
//! accumulates-then-divides and shares this file's softmax convention;
//! `ggml_compute_forward_soft_max_f32` (the `-ngl 0` CPU fallback) normalizes
//! before the KQV matmul and differs in the last bits of every attention
//! output. That convention difference still moves the cosines.
//!
//! Files are written as `0-<tensor>.f32` / `.i32`, headerless little-endian, in
//! the reference's own layout (`[n_pos][row]`, ggml `ne = [row, n_pos]`), so
//! `scripts/release/parity.py` and `compare_logits.py` compare the two
//! directories untouched. The verdict below uses the same thresholds.
//!
//! Four reference checkpoints are never emitted: `ffn_up-<il>` /
//! `ffn_down-<il>` (and their bare projector twins) are cb'd before their bias
//! add, and `cpu_math::matmul` folds the bias into the same expression as the
//! dot product, so that intermediate is not a value this implementation holds;
//! `ffn_up_b` / `ffn_out` are the post-bias twins of the same matmul. Likewise
//! `norm_w-<il>` (layernorm before the bias add) and the `mm.*.bias` leaves.
//! parity.py lists them under `missing`; pass `--allow-missing`.
//!
//! The reference's final projector output is not in the trace directory: it is
//! `<prefix>.mmproj.f32` from `mtmd_get_output_embd`. This gate writes its twin
//! as `<out-dir>.mmproj.f32` and compares against it when given.
//!
//! Check the oracle's revision before reading any number here.
//! `vit_preprocess.rs` targets llama.cpp `434ddbbc0`, where Qwen3-VL sets
//! `image_resize_algo = RESIZE_ALGO_BICUBIC` and every algo routes through
//! `resize_pillow`. llama.cpp 0.2.0 is built from `bb4caa754`, where Qwen3-VL
//! sets `RESIZE_ALGO_BILINEAR`, which dispatches to the naive align-corners
//! `resize_bilinear` — a different image, by up to 8/255 on a text page. Any
//! image whose target size differs from its source size therefore fails at
//! `inp_raw` against a 0.2.0 oracle however correct the ViT is.
//! `OJAS_VIT_INP_RAW=<reference inp_raw dump>` substitutes the oracle's own
//! pixels so the encoder can still be judged; use it whenever the `inp_raw`
//! row is not bit-exact.
//!
//! usage: vit_parity_gate <mmproj.gguf> <image> <out-dir> [reference-dir] [reference.mmproj.f32]
//!        OJAS_VIT_INP_RAW=<path> to bypass preprocessing with the reference's plane

use anyhow::{bail, Context, Result};
use ojas_cpu::cpu_vit::{mrope_positions, CpuVit};
use ojas_cpu::vit_preprocess::{decode_rgb8_path, VitPreproc};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Thresholds, fixed before the run.
const PIXEL_MAX_ABS: f64 = 1e-5;
const ACT_MIN_COSINE: f64 = 0.9999;
const PROJ_MIN_COSINE: f64 = 0.999;
const PROJ_MAX_RMSE: f64 = 0.05;

/// What kind of tolerance a dumped tensor is judged against.
#[derive(PartialEq, Clone, Copy)]
enum Kind {
    /// preprocessed pixels — an absolute target
    Pixels,
    /// m-rope position ids — exact or nothing
    Ints,
    /// a ViT activation — judged on direction
    Act,
    /// a projector stage — direction and magnitude
    Proj,
}

struct Dump {
    key: String,
    path: PathBuf,
    rows: usize,
    kind: Kind,
}

fn write_f32(dir: &Path, key: &str, v: &[f32]) -> Result<PathBuf> {
    let path = dir.join(format!("0-{key}.f32"));
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?,
    );
    for x in v {
        out.write_all(&x.to_le_bytes())?;
    }
    out.flush()?;
    Ok(path)
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if b.len() % 4 != 0 {
        bail!("{}: {} bytes is not a whole number of f32", path.display(), b.len());
    }
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Worst-row (rmse, cosine, max_abs) over `rows` equal slices — the same
/// arithmetic `compare_logits.py` does, in f64, so the two agree to printing
/// precision. A row identically zero on both sides has no direction and would
/// yield a NaN cosine, so it is skipped rather than poisoning the minimum.
fn compare(a: &[f32], b: &[f32], rows: usize) -> (f64, f64, f64) {
    let w = a.len() / rows;
    let (mut rmse, mut cos, mut mx) = (0f64, 1f64, 0f64);
    for r in 0..rows {
        let (x, y) = (&a[r * w..(r + 1) * w], &b[r * w..(r + 1) * w]);
        let (mut se, mut dot, mut na, mut nb) = (0f64, 0f64, 0f64, 0f64);
        for i in 0..w {
            let (u, v) = (x[i] as f64, y[i] as f64);
            se += (u - v) * (u - v);
            dot += u * v;
            na += u * u;
            nb += v * v;
            mx = mx.max((u - v).abs());
        }
        rmse = rmse.max((se / w as f64).sqrt());
        let den = (na * nb).sqrt();
        if den > 0.0 {
            cos = cos.min(dot / den);
        }
    }
    (rmse, cos, mx)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage =
        "usage: vit_parity_gate <mmproj.gguf> <image> <out-dir> [reference-dir] [reference.mmproj.f32]";
    let mmproj = args.next().expect(usage);
    let image = args.next().expect(usage);
    let outdir = PathBuf::from(args.next().expect(usage));
    let reference = args.next().map(PathBuf::from);
    let ref_final = args.next().map(PathBuf::from);

    std::fs::create_dir_all(&outdir)?;

    // Pair on the tensor name with the leading "<pos>-" stripped, as parity.py's
    // key_of does: the reference's position index is its own graph-pass counter
    // and need not match this one. Built before the forward pass so a narrow
    // OJAS_REFERENCE_VISION_FILTER also narrows what is written — a full-page
    // default-filter trace is ~4.8 GB a side.
    let mut refs: BTreeMap<String, PathBuf> = BTreeMap::new();
    if let Some(dir) = &reference {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
            let Some((head, tail)) = name.split_once('-') else { continue };
            if !head.is_empty() && head.chars().all(|c| c.is_ascii_digit()) {
                let key = tail.trim_end_matches(".f32").trim_end_matches(".i32").to_string();
                refs.insert(key, path);
            }
        }
        if refs.is_empty() {
            bail!("{} holds no <pos>-<tensor>.f32 trace files", dir.display());
        }
    }
    let wanted = |key: &str| reference.is_none() || refs.contains_key(key);

    // --- preprocessing -----------------------------------------------------
    let (w, h, rgb) = decode_rgb8_path(&image)?;
    let pre = VitPreproc::default();
    let (tw, th, mut planar) = pre.preprocess(&rgb, w, h)?;

    // `mtmd_debug_encode_image` takes pre-processed planes, so the ViT can be
    // checked independently of resize/normalize. Substituting the reference's
    // own `inp_raw` dump gives the same separation here: geometry still comes
    // from VitPreproc, pixels do not. Use it when the oracle's llama.cpp
    // revision preprocesses differently from the revision vit_preprocess.rs
    // targets, otherwise a preprocessing mismatch masks every ViT number
    // downstream.
    if let Ok(path) = std::env::var("OJAS_VIT_INP_RAW") {
        let given = read_f32(Path::new(&path))?;
        if given.len() != planar.len() {
            bail!(
                "OJAS_VIT_INP_RAW {path} has {} floats, but {tw}x{th}x3 needs {}",
                given.len(),
                planar.len()
            );
        }
        println!("pixels substituted from {path} (preprocessing bypassed)");
        planar = given;
    }

    let mut g = ojas_formats::gguf::Gguf::open(&mmproj)?;
    let vit = CpuVit::load(&mut g)?;
    let (pw, ph) = (tw / vit.patch, th / vit.patch);
    let n_pos = pw * ph;
    let n_merged = n_pos / (vit.merge * vit.merge);
    println!(
        "image {w}x{h} -> {tw}x{th} | patches {pw}x{ph} = {n_pos} | merged {n_merged} | d={} layers={}",
        vit.n_embd, vit.n_layers
    );

    let mut dumps: Vec<Dump> = Vec::new();

    // `inp_raw` is a graph input: the planar-CHW pixel plane, and the only
    // observable output of preprocessing, since mtmd_debug_preprocess_image logs
    // geometry and dumps nothing. Rows are the three colour planes.
    dumps.push(Dump {
        key: "inp_raw".into(),
        path: write_f32(&outdir, "inp_raw", &planar)?,
        rows: 3,
        kind: Kind::Pixels,
    });

    // `positions` is section-major in the reference — [t..][h..][w..][e..],
    // clip.cpp's PROJECTOR_TYPE_QWEN3VL arm — while `mrope_positions` returns
    // one [t,h,w,e] per token, so this writes the transpose.
    {
        let mpos = mrope_positions(pw, ph);
        let path = outdir.join("0-positions.i32");
        let mut out = std::io::BufWriter::new(std::fs::File::create(&path)?);
        for sect in 0..4 {
            for p in &mpos {
                out.write_all(&p[sect].to_le_bytes())?;
            }
        }
        out.flush()?;
        dumps.push(Dump { key: "positions".into(), path, rows: 1, kind: Kind::Ints });
    }

    // --- the encoder itself ------------------------------------------------
    let mut tapped: Vec<Dump> = Vec::new();
    let mut tap_err: Option<anyhow::Error> = None;
    let trace = vit.forward_tapped(&planar, tw, th, &mut |name, il, v| {
        if tap_err.is_some() {
            return;
        }
        let key = match il {
            Some(i) => format!("{name}-{i}"),
            None => name.to_string(),
        };
        if !wanted(&key) {
            return;
        }
        // The projector's build_ffn runs with il == -1 over the 2x2-merged
        // sequence, so its stages carry bare names and n_pos/4 rows.
        let proj = il.is_none() && name.starts_with("ffn_");
        match write_f32(&outdir, &key, v) {
            Ok(path) => tapped.push(Dump {
                key,
                path,
                rows: if proj { n_merged } else { n_pos },
                kind: if proj { Kind::Proj } else { Kind::Act },
            }),
            Err(e) => tap_err = Some(e),
        }
    })?;

    if let Some(e) = tap_err {
        return Err(e);
    }
    dumps.extend(tapped);

    // The reference's projector output comes from mtmd_get_output_embd, not the
    // trace callback, so its twin goes beside the directory under the same name.
    let final_path = PathBuf::from(format!("{}.mmproj.f32", outdir.display()));
    {
        let mut out = std::io::BufWriter::new(std::fs::File::create(&final_path)?);
        for x in &trace.out {
            out.write_all(&x.to_le_bytes())?;
        }
        out.flush()?;
    }
    println!(
        "wrote {} tensors to {} (+ {} : {} f32)",
        dumps.len(),
        outdir.display(),
        final_path.display(),
        trace.out.len()
    );
    if !trace.out.iter().all(|v| v.is_finite()) {
        println!("GATE: VIT PARITY FAIL (non-finite projector output)");
        std::process::exit(1);
    }

    // --- comparison --------------------------------------------------------
    if reference.is_none() {
        println!("no reference directory given; dumps only, no verdict");
        return Ok(());
    }

    let mut first_fail: Option<(String, f64, f64)> = None;
    let mut failures = 0usize;
    let mut missing: Vec<&str> = Vec::new();
    println!("{:<22} {:>6} {:>11} {:>13} {:>11}", "tensor", "rows", "max_abs", "cosine", "rmse");
    for d in &dumps {
        let Some(rp) = refs.get(&d.key) else {
            missing.push(&d.key);
            continue;
        };
        if d.kind == Kind::Ints {
            let same = std::fs::read(rp)? == std::fs::read(&d.path)?;
            println!("{:<22} {:>6} {:>11}", d.key, "i32", if same { "exact" } else { "DIFFER" });
            if !same {
                failures += 1;
            }
            continue;
        }
        let (a, b) = (read_f32(rp)?, read_f32(&d.path)?);
        if a.len() != b.len() {
            println!("{:<22} {:>6} SHAPE {} vs {}", d.key, d.rows, a.len(), b.len());
            failures += 1;
            continue;
        }
        let (rmse, cos, mx) = compare(&a, &b, d.rows);
        let ok = match d.kind {
            Kind::Pixels => mx <= PIXEL_MAX_ABS,
            Kind::Proj => cos >= PROJ_MIN_COSINE && rmse <= PROJ_MAX_RMSE,
            _ => cos >= ACT_MIN_COSINE,
        };
        println!(
            "{:<22} {:>6} {:>11.3e} {:>13.9} {:>11.3e} {}",
            d.key,
            d.rows,
            mx,
            cos,
            rmse,
            if ok { "" } else { "FAIL" }
        );
        if !ok {
            failures += 1;
            first_fail.get_or_insert((d.key.clone(), cos, mx));
        }
    }
    if !missing.is_empty() {
        println!("not in reference trace (filter or pre-bias checkpoint): {}", missing.join(" "));
    }

    match ref_final {
        Some(rf) => {
            let a = read_f32(&rf)?;
            if a.len() != trace.out.len() {
                println!("mm.2 (mmproj)          SHAPE {} vs {}", a.len(), trace.out.len());
                failures += 1;
            } else {
                let (rmse, cos, mx) = compare(&a, &trace.out, n_merged);
                let ok = cos >= PROJ_MIN_COSINE && rmse <= PROJ_MAX_RMSE;
                println!(
                    "{:<22} {:>6} {:>11.3e} {:>13.9} {:>11.3e} {}",
                    "mm.2 (mmproj)",
                    n_merged,
                    mx,
                    cos,
                    rmse,
                    if ok { "" } else { "FAIL" }
                );
                if !ok {
                    failures += 1;
                    first_fail.get_or_insert(("mm.2 (mmproj)".into(), cos, mx));
                }
            }
        }
        None => println!("{:<22} {:>6} {:>11}", "mm.2 (mmproj)", n_merged, "NO REF"),
    }

    if failures > 0 {
        if let Some((k, c, m)) = first_fail {
            println!("first divergence: {k} (cosine {c:.9}, max_abs {m:.3e})");
        }
        println!("GATE: VIT PARITY FAIL ({failures} tensors)");
        std::process::exit(1);
    }
    println!("GATE: VIT PARITY PASS");
    Ok(())
}
