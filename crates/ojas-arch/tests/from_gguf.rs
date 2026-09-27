//! `ArchSpec` against a real file.
//!
//! The failure modes are quiet ones: a key that silently defaults, a head_dim derived when
//! the file states it, a vocab read from `vocab_size` when the tokenizer disagrees. So this
//! reads an actual GGUF rather than a hand-built one, and is skipped loudly when none is
//! available rather than quietly passing.
//!
//! `OJAS_TEST_GGUF=/path/to/model.gguf cargo test -p ojas-arch`

use ojas_arch::{Act, ArchSpec};
use ojas_formats::gguf::Gguf;

fn open() -> Option<Gguf> {
    let p = std::env::var("OJAS_TEST_GGUF").ok().or_else(|| {
        let d = format!("{}/models/qwen2.5-0.5b-instruct-q8_0.gguf", std::env::var("HOME").ok()?);
        std::path::Path::new(&d).exists().then_some(d)
    })?;
    Some(Gguf::open(&p).expect("OJAS_TEST_GGUF is set but could not be opened"))
}

#[test]
fn qwen2_shape_and_features() {
    let Some(g) = open() else {
        eprintln!("skipped: set OJAS_TEST_GGUF to a dense GGUF to run this");
        return;
    };
    let a = ArchSpec::from_gguf(&g).expect("parse");
    println!("{}: L={} d={} ffn={} vocab={}", a.arch, a.n_layers, a.d, a.ffn, a.vocab);

    assert_eq!(a.layers.len(), a.n_layers, "one LayerSpec per block");
    // MTP: the main stack excludes the draft blocks, and `block_count` keeps the raw value
    assert_eq!(a.n_layers, a.block_count - a.n_nextn);
    let l = a.layers[0];
    assert_eq!(l.qdim, l.n_head * l.head_dim);
    assert_eq!(l.kvdim, l.n_kv * l.head_dim);
    assert_eq!(l.n_head % l.n_kv, 0, "GQA grouping must divide");
    assert!((l.scale - 1.0 / (l.head_dim as f32).sqrt()).abs() < 1e-9);
    assert!(a.vocab > 0 && a.d > 0 && a.ffn > 0 && a.eps > 0.0);

    // the lm_head name must name a tensor that exists: the tied/untied decision comes from
    // tensor presence, and inverting it gives a runner a missing-tensor error at load or, worse,
    // the embedding table used as a head on a model that ships its own
    assert!(g.tensors.contains_key(&a.lm_head), "lm_head '{}' is not in the file", a.lm_head);
    assert_eq!(a.tied_head(), !g.tensors.contains_key("output.weight"));

    // feature flags must agree with the tensors they are derived from
    assert_eq!(a.qkv_bias, g.tensors.contains_key("blk.0.attn_q.bias"));
    assert_eq!(a.qk_norm, g.tensors.contains_key("blk.0.attn_q_norm.weight"));
    if a.arch == "qwen2" {
        assert!(a.qkv_bias, "qwen2 ships q/k/v bias");
        assert!(!a.qk_norm, "qwen2 has no per-head q/k norm");
        assert_eq!(a.act, Act::Silu);
        assert_eq!(a.embed_scale, 1.0);
        assert!(a.rope_neox);
        assert!(a.non_dense.is_none() && a.n_experts == 0);
    }
}

#[test]
fn require_dense_refuses_what_it_cannot_run() {
    let Some(g) = open() else { return };
    let a = ArchSpec::from_gguf(&g).expect("parse");

    // the runner that supports nothing optional must refuse a model needing something
    if a.qkv_bias {
        let e = a.require_dense(&[]).unwrap_err().to_string();
        assert!(e.contains("qkv_bias"), "refusal should name the feature: {e}");
        a.require_dense(&["qkv_bias"]).expect("accepted once the feature is declared");
    }

    // and a structural mismatch is refused however many flags are declared
    let mut moe = a.clone();
    moe.n_experts = 64;
    assert!(moe.require_dense(&["qkv_bias", "qk_norm", "gelu"]).is_err(),
            "a dense runner must not accept an MoE file");
    let mut ssm = a.clone();
    ssm.non_dense = Some("Gated-DeltaNet recurrent layers");
    let e = ssm.require_dense(&["qkv_bias"]).unwrap_err().to_string();
    assert!(e.contains("Gated-DeltaNet"), "refusal should name what is missing: {e}");
}
