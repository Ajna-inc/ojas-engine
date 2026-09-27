//! Metal-vs-CUDA parity ledger: which canonical kernel entries exist on each backend.
//!
//! Both backends' entry tables are read at runtime, including the macro- and
//! generator-produced names a text scan misses, and the difference is printed.
//!
//! A name present on both sides means the entry exists, not that it computes the same thing;
//! the conformance tests cover that. Read this as which model paths are reachable on CUDA, not
//! as a quality score.
//!
//! `--formats` compares the other axis, GGUF quantization coverage against llama.cpp. That
//! column is transcribed (see `LLAMA_CUDA_MMVQ`) rather than read from their tree, so it carries
//! a commit and has to be refreshed by hand.
//!
//! ```text
//! cargo run --release -p ojas-cuda --example parity              # summary + gap by family
//! cargo run --release -p ojas-cuda --example parity -- --all     # every missing entry
//! cargo run --release -p ojas-cuda --example parity -- --cuda-only
//! cargo run --release -p ojas-cuda --example parity -- --formats # vs llama.cpp, per quant type
//! ```
use std::collections::{BTreeMap, BTreeSet};

/// Group an entry by purpose, which is what decides porting order. The prefix is a better key
/// than the family: Metal splits `gemv` across `gemv.rs` and `nat.rs` while CUDA splits it across
/// `gemv`, `gemv_q8` and `nat`, so grouping by family would report differences that are purely
/// organisational.
fn group(name: &str) -> &'static str {
    const PREFIXES: &[(&str, &str)] = &[
        ("gemv_nat_", "gemv (native quant)"),
        ("gemv_", "gemv"),
        ("gemm_", "gemm / prefill"),
        ("moe_", "moe"),
        ("mla_", "mla (DeepSeek)"),
        ("attention_", "attention"),
        ("attn_", "attention"),
        ("flash_", "attention"),
        ("requant_", "requant"),
        ("ffn_", "ffn"),
        ("qkv_", "qkv"),
        ("embed_", "embed"),
        ("rope", "rope"),
        ("rmsnorm", "norm"),
        ("layernorm", "norm"),
        ("store_kv", "kv cache"),
        ("kv_", "kv cache"),
        ("learn_", "train (tape)"),
        ("t_", "train"),
        ("flce_", "train (fused CE)"),
        ("cnn_", "vision cnn"),
        ("vit_", "vision vit"),
        ("patch", "vision vit"),
        ("ssm_", "ssm / deltanet"),
        ("deltanet", "ssm / deltanet"),
        ("conv1d", "ssm / deltanet"),
        ("iq", "iq grids"),
        ("quantize_", "quantize"),
    ];
    for (p, g) in PREFIXES {
        if name.starts_with(p) {
            return g;
        }
    }
    "misc / glue"
}

/// Quantization types with a CUDA `mul_mat_vec_q` kernel in llama.cpp, transcribed from
/// `ggml/src/ggml-cuda/mmvq.cu` at `66fba63` (23 Sep 2026). Paired with our own tag for the same
/// format, or `""` for a format neither backend implements.
const LLAMA_CUDA_MMVQ: &[(&str, &str)] = &[
    ("Q4_0", "q40"), ("Q4_1", "q41"), ("Q5_0", "q50"), ("Q5_1", "q51"), ("Q8_0", "q80"),
    ("Q2_K", "q2k"), ("Q3_K", "q3k"), ("Q4_K", "q4k"), ("Q5_K", "q5k"), ("Q6_K", "q6k"),
    ("IQ1_S", "iq1s"), ("IQ1_M", "iq1m"), ("IQ2_XXS", "iq2xxs"), ("IQ2_XS", "iq2xs"),
    ("IQ2_S", "iq2s"), ("IQ3_XXS", "iq3xxs"), ("IQ3_S", "iq3s"), ("IQ4_NL", "iq4nl"),
    ("IQ4_XS", "iq4xs"), ("MXFP4", "mxfp4"),
    // newer than the GGUF reader here: no format on either backend
    ("NVFP4", ""), ("Q1_0", ""), ("Q2_0", ""),
];

/// llama.cpp's KV cache types (`common/arg.cpp::kv_cache_types`, same commit). This engine
/// supports F16 only.
const LLAMA_KV_TYPES: &[&str] =
    &["F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "IQ4_NL", "Q5_0", "Q5_1"];

fn formats_report(metal: &BTreeSet<String>, cuda: &BTreeSet<String>) {
    println!("quantization formats, against llama.cpp CUDA (mmvq @ 66fba63)");
    println!("  {:<10} {:>8} {:>8} {:>8}", "format", "llama", "metal", "cuda");
    let (mut only_llama, mut metal_only) = (vec![], vec![]);
    for (llama_name, tag) in LLAMA_CUDA_MMVQ {
        let (m, c) = if tag.is_empty() {
            (false, false)
        } else {
            let e = format!("gemv_nat_{tag}");
            (metal.contains(&e), cuda.contains(&e))
        };
        println!("  {:<10} {:>8} {:>8} {:>8}", llama_name, "yes",
                 if m { "yes" } else { "-" }, if c { "yes" } else { "-" });
        if !m && !c {
            only_llama.push(*llama_name);
        } else if m && !c {
            metal_only.push(*llama_name);
        }
    }
    println!("\n  in llama.cpp, in neither of ours: {}",
             if only_llama.is_empty() { "none".into() } else { only_llama.join(", ") });
    println!("  on Metal but not CUDA:            {}",
             if metal_only.is_empty() { "none".into() } else { metal_only.join(", ") });
    println!("\nKV cache types");
    println!("  llama.cpp  {}", LLAMA_KV_TYPES.join(", "));
    println!("  ojas       F16 only, both backends");
    println!("  At long context the KV cache is the memory, so this one is on OUR axis.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let show_all = args.iter().any(|a| a == "--all");
    let show_cuda_only = args.iter().any(|a| a == "--cuda-only");
    let show_formats = args.iter().any(|a| a == "--formats");

    let metal: BTreeSet<String> =
        ojas_metal::kernels::all_names().into_iter().map(str::to_string).collect();
    let cuda: BTreeSet<String> =
        ojas_cuda::kernels::all_names().map(str::to_string).collect();

    if show_formats {
        formats_report(&metal, &cuda);
        return;
    }

    let missing: BTreeSet<&String> = metal.difference(&cuda).collect();
    let extra: BTreeSet<&String> = cuda.difference(&metal).collect();
    let shared = metal.intersection(&cuda).count();

    println!("canonical kernel entries");
    println!("  metal       {:4}", metal.len());
    println!("  cuda        {:4}", cuda.len());
    println!("  on both     {:4}   ({:.0} % of Metal)", shared,
             100.0 * shared as f64 / metal.len() as f64);
    println!("  metal only  {:4}   <- the port's remaining surface", missing.len());
    println!("  cuda only   {:4}   (graphs/_g, W4A8, dp4a decode, cnn+learn training)", extra.len());

    // ---- the gap, by what it is for
    let mut by_group: BTreeMap<&str, (usize, usize, Vec<&String>)> = BTreeMap::new();
    for n in &metal {
        by_group.entry(group(n)).or_default().0 += 1;
    }
    for n in &missing {
        let e = by_group.entry(group(n)).or_default();
        e.1 += 1;
        e.2.push(n);
    }
    let mut rows: Vec<_> = by_group.into_iter().collect();
    rows.sort_by_key(|(_, (_, miss, _))| std::cmp::Reverse(*miss));

    println!("\nthe gap, by what the kernels are for");
    println!("  {:<22} {:>6} {:>8} {:>8}", "group", "metal", "missing", "done");
    for (g, (total, miss, names)) in &rows {
        if *miss == 0 && !show_all {
            continue;
        }
        println!("  {:<22} {:>6} {:>8} {:>7.0}%", g, total, miss,
                 100.0 * (total - miss) as f64 / *total as f64);
        if show_all {
            for chunk in names.chunks(4) {
                let line: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
                println!("      {}", line.join("  "));
            }
        }
    }
    let complete: Vec<&str> = rows.iter().filter(|(_, (_, m, _))| *m == 0).map(|(g, _)| *g).collect();
    if !complete.is_empty() {
        println!("\n  complete: {}", complete.join(", "));
    }

    if show_cuda_only {
        println!("\ncuda-only entries (no Metal twin)");
        let mut g: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for n in &extra {
            g.entry(group(n)).or_default().push(n.as_str());
        }
        for (k, v) in g {
            println!("  {k} ({})", v.len());
            for chunk in v.chunks(4) {
                println!("      {}", chunk.join("  "));
            }
        }
    }

    if !show_all {
        println!("\n--all lists every missing entry; --cuda-only lists what CUDA has and Metal does not;");
        println!("--formats compares quantization coverage against llama.cpp.");
    }
}
