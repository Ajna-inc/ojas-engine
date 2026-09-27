//! Index parse + tensor read on a synthetic multi-shard safetensors checkpoint.
//!
//! Builds a two-shard checkpoint (with an `index.json`, a `config.json`, and a
//! mix of F32 / BF16 / U8 dtypes) entirely in a temp dir — no Python, no real
//! model — then checks that `SafeIndex` recovers each tensor's dtype, shape and
//! exact bytes, resolving the right tensor from the right shard.

use ojas_formats::safetensors::{self, SafeIndex};

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let mut d = std::env::temp_dir();
    d.push(format!("ojas_st_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn index_parse_and_reads_across_two_shards() {
    let dir = tmpdir("multi");

    // Shard 1: an F32 weight and a BF16 weight.
    let f32_data: Vec<f32> = vec![1.0, -2.5, 3.25, 0.0, 42.0, -0.5];
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    // BF16 = top 16 bits of the f32 bit pattern.
    let bf_src: Vec<f32> = vec![1.0, 2.0, -4.0, 0.5];
    let bf_bytes: Vec<u8> = bf_src
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let shard1 = safetensors::serialize(&[
        ("model.dense.weight", "F32", vec![2, 3], f32_bytes.clone()),
        ("model.norm.weight", "BF16", vec![4], bf_bytes.clone()),
    ]);
    std::fs::write(dir.join("model-00001-of-00002.safetensors"), &shard1).unwrap();

    // Shard 2: a U8 packed tensor (e.g. MXFP4 blocks) — no float widening.
    let u8_data: Vec<u8> = (0..48u8).collect();
    let shard2 = safetensors::serialize(&[
        ("model.experts.blocks", "U8", vec![3, 16], u8_data.clone()),
    ]);
    std::fs::write(dir.join("model-00002-of-00002.safetensors"), &shard2).unwrap();

    // The index maps each tensor name to its shard.
    let index = serde_json::json!({
        "metadata": { "total_size": shard1.len() + shard2.len() },
        "weight_map": {
            "model.dense.weight": "model-00001-of-00002.safetensors",
            "model.norm.weight": "model-00001-of-00002.safetensors",
            "model.experts.blocks": "model-00002-of-00002.safetensors",
        }
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .unwrap();

    // A config.json sidecar for arch detection.
    std::fs::write(
        dir.join("config.json"),
        br#"{"architectures":["GptOssForCausalLM"],"num_local_experts":32,"hidden_size":2880}"#,
    )
    .unwrap();

    let si = SafeIndex::open(&dir).unwrap();
    assert_eq!(si.len(), 3, "all three tensors indexed across two shards");

    // dtype + shape recovered
    assert_eq!(si.dtype("model.dense.weight").unwrap(), "F32");
    assert_eq!(si.shape("model.dense.weight").unwrap(), &[2, 3]);
    assert_eq!(si.dtype("model.norm.weight").unwrap(), "BF16");
    assert_eq!(si.dtype("model.experts.blocks").unwrap(), "U8");
    assert_eq!(si.shape("model.experts.blocks").unwrap(), &[3, 16]);

    // the U8 tensor lives in shard 2
    assert_eq!(si.get("model.experts.blocks").unwrap().shard, "model-00002-of-00002.safetensors");

    // raw bytes are byte-exact
    assert_eq!(si.read_raw("model.dense.weight").unwrap(), f32_bytes);
    assert_eq!(si.read_raw("model.experts.blocks").unwrap(), u8_data);

    // float widening is exact for F32 and BF16
    let got = si.read_f32("model.dense.weight").unwrap();
    assert_eq!(got, f32_data);
    let got_bf = si.read_f32("model.norm.weight").unwrap();
    assert_eq!(got_bf, bf_src, "BF16 round-trips exactly for these values");

    // widening refuses the packed dtype
    assert!(si.read_f32("model.experts.blocks").is_err());

    // config.json is reachable
    let cfg = si.config().unwrap();
    assert_eq!(cfg["architectures"][0], "GptOssForCausalLM");
    assert_eq!(cfg["num_local_experts"], 32);

    // missing tensor errors, not panics
    assert!(si.read_raw("nope").is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn single_file_fallback_without_index() {
    let dir = tmpdir("single");
    let data: Vec<u8> = vec![9, 8, 7, 6];
    let bytes = safetensors::serialize(&[("w", "U8", vec![4], data.clone())]);
    std::fs::write(dir.join("model.safetensors"), &bytes).unwrap();

    let si = SafeIndex::open(&dir).unwrap();
    assert_eq!(si.len(), 1);
    assert_eq!(si.read_raw("w").unwrap(), data);
    let _ = std::fs::remove_dir_all(&dir);
}
