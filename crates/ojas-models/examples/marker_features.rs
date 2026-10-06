//! Record an encoder decision model's view of every option of every request in a set
//! of JSONL request files: the scorer's hidden vector at the option's marker, the
//! option's raw score, the question's calibration temperature, and the answer the row
//! carries. The vectors go to one f32 file, one row per option; an index line per
//! question says where its options are.
//!
//! usage: marker_features <model.gguf> <out_dir> <requests.jsonl>... [--rows N]
//!
//! A request file holds `{"body": request, "gold": {question id: key}}` lines, the
//! form `decision_rl` trains on; lines without a gold answer are skipped.

use anyhow::{bail, Context, Result};
use ojas_decision::json::Json;
use ojas_decision::{DecisionModel, MarkerBackend};
use ojas_formats::gguf::Gguf;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Sequences per GPU pass.
const PASS: usize = 32;

struct Pending {
    family: String,
    row: usize,
    kind: String,
    temperature: f32,
    gold: usize,
    ids: Vec<u32>,
    qtype: u32,
    markers: Vec<usize>,
}

fn read_vector(g: &mut Gguf, name: &str) -> Result<Vec<f32>> {
    let (_, ty, bytes) = g.read_tensor(name)?;
    Ok(match ty {
        0 => bytes.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
        _ => bytes.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c).to_f32()).collect(),
    })
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().context("model path")?;
    let out = PathBuf::from(args.next().context("output directory")?);
    let (mut files, mut rows_per_file) = (Vec::new(), usize::MAX);
    while let Some(a) = args.next() {
        if a == "--rows" { rows_per_file = args.next().context("--rows N")?.parse()?; } else { files.push(PathBuf::from(a)); }
    }
    if files.is_empty() { bail!("usage: marker_features <model.gguf> <out_dir> <requests.jsonl>... [--rows N]"); }
    std::fs::create_dir_all(&out)?;

    let metal = ojas_metal::MetalGpu::new()?;
    let gpu = ojas_models::decision_backend::MetalDecision(&metal);
    let model = DecisionModel::load(&gpu, &model_path)?;
    let encoder = model.marker_backend().context("not an encoder decision model")?;
    let mut g = Gguf::open(&model_path)?;
    let (out_w, out_b) = (read_vector(&mut g, "cls.output.weight")?, read_vector(&mut g, "cls.output.bias")?[0]);
    let d = encoder.width();

    let mut vectors = BufWriter::new(std::fs::File::create(out.join("vectors.f32"))?);
    let mut index = BufWriter::new(std::fs::File::create(out.join("index.jsonl"))?);
    let mut written = 0usize;
    let mut pending: Vec<Pending> = Vec::new();
    let flush = |pending: &mut Vec<Pending>, vectors: &mut BufWriter<std::fs::File>, index: &mut BufWriter<std::fs::File>, written: &mut usize| -> Result<()> {
        if pending.is_empty() { return Ok(()); }
        let seqs: Vec<Vec<u32>> = pending.iter().map(|p| p.ids.clone()).collect();
        let qtypes: Vec<u32> = pending.iter().map(|p| p.qtype).collect();
        let markers: Vec<Vec<usize>> = pending.iter().map(|p| p.markers.clone()).collect();
        let fwd = encoder.marker_head_forward(&seqs, &qtypes, &markers)?;
        for (p, hidden) in pending.iter().zip(fwd.scorer_hidden) {
            let scores: Vec<f32> = hidden.iter()
                .map(|h| out_b + h.iter().zip(&out_w).map(|(a, w)| a * w).sum::<f32>()).collect();
            for h in &hidden {
                debug_assert_eq!(h.len(), d);
                for v in h { vectors.write_all(&v.to_le_bytes())?; }
            }
            let line = Json::Object(vec![
                ("family".into(), Json::Str(p.family.clone())),
                ("row".into(), Json::Int(p.row.to_string())),
                ("kind".into(), Json::Str(p.kind.clone())),
                ("temperature".into(), Json::Float(p.temperature as f64)),
                ("gold".into(), Json::Int(p.gold.to_string())),
                ("first".into(), Json::Int(written.to_string())),
                ("scores".into(), Json::Array(scores.iter().map(|&s| Json::Float(s as f64)).collect())),
            ]);
            writeln!(index, "{}", line.to_python(true))?;
            *written += hidden.len();
        }
        pending.clear();
        Ok(())
    };

    for file in &files {
        let family = Path::new(file).file_stem().and_then(|s| s.to_str()).unwrap_or("file").to_string();
        let text = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
        let mut taken = 0;
        for (row, line) in text.lines().enumerate() {
            if taken >= rows_per_file { break; }
            let Ok(entry) = Json::parse(line) else { continue };
            let (Some(body), Some(gold)) = (entry.get("body"), entry.get("gold")) else { continue };
            let Ok(req) = model.request(body) else { continue };
            if model.validate(&req).is_err() || req.questions.len() != 1 { continue; }
            let q = &req.questions[0];
            let key = match gold.get(&q.id) {
                Some(Json::Str(s)) => s.clone(),
                Some(Json::Bool(b)) => b.to_string(),
                Some(Json::Int(i)) => i.clone(),
                _ => continue,
            };
            let Some(gold) = q.options.iter().position(|o| o.key == key) else { continue };
            let Ok(mut prompts) = model.marker_prompts(&req) else { continue };
            let p = prompts.swap_remove(0);
            if p.ids.len() > 1024 { continue; }
            pending.push(Pending { family: family.clone(), row, kind: q.kind.name().into(), temperature: p.temperature, gold,
                                   ids: p.ids, qtype: p.qtype, markers: p.markers });
            taken += 1;
            if pending.len() == PASS { flush(&mut pending, &mut vectors, &mut index, &mut written)?; }
        }
        flush(&mut pending, &mut vectors, &mut index, &mut written)?;
        eprintln!("{family}: {taken} questions");
    }
    vectors.flush()?;
    index.flush()?;
    std::fs::write(out.join("meta.json"), format!("{{\"width\": {d}, \"options\": {written}, \"model\": {:?}}}\n", model_path))?;
    eprintln!("{written} option vectors of width {d} in {}", out.display());
    Ok(())
}
