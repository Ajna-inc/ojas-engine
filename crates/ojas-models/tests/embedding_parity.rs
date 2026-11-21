//! A text-embedding GGUF on Metal against the reference model's sentence embeddings.
//!
//! `scripts/embedding_reference.py` writes the reference file (texts of varied length,
//! some past the sliding-window reach, embedded on the CPU in float32 under each named
//! prompt, with their token ids):
//!
//! ```text
//! OJAS_EMBED_GGUF=~/models/embed/embeddinggemma-2-f16.gguf \
//! OJAS_EMBED_REFERENCE=~/models/embed/embeddinggemma-2-reference.jsonl \
//!     cargo test --release -p ojas-models --test embedding_parity -- --ignored --nocapture
//! ```
//!
//! Every text must tokenize to the reference's ids and embed within cosine 0.999 of the
//! reference. Each prompt's texts go through the encoder in one packed pass, and again
//! one text per pass, which must agree with the packed result.

use ojas_decision::json::Json;
use ojas_formats::gguf::{Gguf, Meta};
use ojas_models::decoder::DecoderGpu;
use ojas_tokenize::Bpe;

struct Case {
    prompt_name: String,
    prompt: String,
    text: String,
    ids: Vec<u32>,
    embedding: Vec<f32>,
}

fn path(var: &str) -> String {
    let p = std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
    match p.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").expect("HOME")),
        None => p,
    }
}

fn cases(file: &str) -> Vec<Case> {
    let nums = |j: &Json, k: &str| -> Vec<f64> {
        j.get(k).and_then(Json::as_array).unwrap_or_else(|| panic!("reference line without {k}"))
            .iter().map(|v| v.as_f64().expect("a number")).collect()
    };
    let text = |j: &Json, k: &str| j.get(k).and_then(Json::as_str).unwrap_or_else(|| panic!("reference line without {k}")).to_string();
    std::fs::read_to_string(file).expect("reference file").lines().filter(|l| !l.trim().is_empty()).map(|l| {
        let j = Json::parse(l).expect("a JSON line");
        Case {
            prompt_name: text(&j, "prompt_name"), prompt: text(&j, "prompt"), text: text(&j, "text"),
            ids: nums(&j, "ids").into_iter().map(|v| v as u32).collect(),
            embedding: nums(&j, "embedding").into_iter().map(|v| v as f32).collect(),
        }
    }).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(&p, &q)| p as f64 * q as f64).sum::<f64>();
    dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt()).max(1e-30)
}

#[test]
#[ignore = "needs an embedding GGUF and its reference file; set OJAS_EMBED_GGUF and OJAS_EMBED_REFERENCE"]
fn embeddings_match_the_reference_model() {
    let cases = cases(&path("OJAS_EMBED_REFERENCE"));
    assert!(!cases.is_empty(), "the reference file is empty");
    let mut g = Gguf::open(&path("OJAS_EMBED_GGUF")).expect("the GGUF");
    let tok = Bpe::from_gguf(&g);
    let flag = |k: &str| matches!(g.meta.get(&format!("tokenizer.ggml.add_{k}_token")), Some(Meta::Bool(true)));
    let bos = flag("bos").then(|| g.meta_u32("tokenizer.ggml.bos_token_id").expect("bos id"));
    let eos = flag("eos").then(|| g.meta_u32("tokenizer.ggml.eos_token_id").expect("eos id"));
    g.sidecars = false;
    let gpu = ojas_metal::MetalGpu::new().expect("a Metal GPU");
    let enc = DecoderGpu::load(&gpu, &mut g, 64, 0, None, None).expect("load");

    let mut names: Vec<&str> = cases.iter().map(|c| c.prompt_name.as_str()).collect();
    names.dedup();
    let mut failures = Vec::new();
    for name in names {
        let group: Vec<&Case> = cases.iter().filter(|c| c.prompt_name == name).collect();
        let seqs: Vec<Vec<u32>> = group.iter().map(|c| {
            let mut ids: Vec<u32> = bos.into_iter().collect();
            ids.extend(tok.encode(&format!("{}{}", c.prompt, c.text)).into_iter().map(|t| t as u32));
            ids.extend(eos);
            ids
        }).collect();
        for (c, ids) in group.iter().zip(&seqs) {
            if *ids != c.ids {
                failures.push(format!("{name}: tokens differ ({} vs {} reference) for {:?}", ids.len(), c.ids.len(),
                    c.text.chars().take(60).collect::<String>()));
            }
        }
        let packed = enc.embed_text(&seqs).expect("embed");
        let cos: Vec<f64> = group.iter().zip(&packed).map(|(c, v)| cosine(v, &c.embedding)).collect();
        let (min, mean) = (cos.iter().cloned().fold(f64::MAX, f64::min), cos.iter().sum::<f64>() / cos.len() as f64);
        let longest = seqs.iter().map(Vec::len).max().unwrap_or(0);
        let long: Vec<String> = seqs.iter().zip(&cos).filter(|(s, _)| s.len() > 512)
            .map(|(s, c)| format!("{} tokens {c:.7}", s.len())).collect();
        println!("{name}: {} texts in one pass ({} tokens, longest {longest}): cosine min {min:.7} mean {mean:.7}; past 512 tokens: {}",
            seqs.len(), seqs.iter().map(Vec::len).sum::<usize>(), long.join(", "));
        for ((c, cs), s) in group.iter().zip(&cos).zip(&seqs) {
            if *cs < 0.999 {
                failures.push(format!("{name}: cosine {cs:.6} at {} tokens for {:?}", s.len(), c.text.chars().take(60).collect::<String>()));
            }
        }
        // One text per pass: packing must not change a text's embedding.
        let mut worst = f64::MAX;
        for (s, v) in seqs.iter().zip(&packed) {
            let alone = enc.embed_text(std::slice::from_ref(s)).expect("embed");
            worst = worst.min(cosine(&alone[0], v));
        }
        println!("{name}: packed against one text per pass: cosine min {worst:.7}");
        if worst < 0.99999 { failures.push(format!("{name}: packing changes an embedding (cosine {worst:.7})")); }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
