//! Reports where a MoE model's parameters and bytes are.
//!
//! "176.94 B params, A3B" says nothing about residency: what a token reads and what has
//! to be resident are different sets, and for a streamed MoE they differ by two orders of
//! magnitude. This splits the file three ways -- routed experts, embeddings, and the
//! dense per-layer skeleton -- and prices each at both its on-disk quant and the f16 the
//! streaming loader dequantizes the skeleton to.
//!
//! usage: param_census <gguf>
use anyhow::Result;
use std::collections::BTreeMap;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: param_census <gguf>");
    let g = ojas_formats::gguf::Gguf::open(&path)?;

    // bytes-per-weight for the ggml types this file uses
    let bpw = |t: u32| -> f64 {
        match t {
            0 => 4.0, 1 => 2.0,                 // F32, F16
            8 => 1.0625,                        // Q8_0  (34B / 32w)
            12 => 0.5625, 14 => 0.8203125,      // Q4_K, Q6_K
            20 => 0.5625,                       // IQ4_NL (18B / 32w)
            23 => 0.53125,                      // IQ4_XS (136B / 256w)
            _ => 0.5,
        }
    };
    let mut cls: BTreeMap<&str, (u64, f64)> = BTreeMap::new();   // class -> (params, bytes)
    let mut per_type: BTreeMap<u32, u64> = BTreeMap::new();
    for (name, t) in &g.tensors {
        let n: u64 = t.dims.iter().product();
        let class = if name.contains("_exps.") { "routed experts" }
            else if name.contains("per_layer_token_embd") { "PLE n-gram table" }
            else if name.contains("token_embd") { "token_embd" }
            else if name == "output.weight" { "lm_head" }
            else if name.starts_with(&format!("blk.{}.", g.tensors.keys()
                     .filter_map(|k| k.strip_prefix("blk.").and_then(|s| s.split('.').next()))
                     .filter_map(|s| s.parse::<usize>().ok()).max().unwrap_or(0))) { "MTP draft block" }
            else { "dense skeleton" };
        let e = cls.entry(class).or_insert((0, 0.0));
        e.0 += n;
        e.1 += n as f64 * bpw(t.ggml_type);
        *per_type.entry(t.ggml_type).or_insert(0) += n;
    }
    let tot_p: u64 = cls.values().map(|v| v.0).sum();
    let tot_b: f64 = cls.values().map(|v| v.1).sum();

    println!("\n{:<20} {:>12} {:>8} {:>10} {:>8}", "class", "params", "share", "on disk", "share");
    for (k, (p, b)) in &cls {
        println!("{k:<20} {:>10.2} B {:>7.1}% {:>8.2} GB {:>7.1}%",
                 *p as f64 / 1e9, *p as f64 / tot_p as f64 * 100.0,
                 b / 1e9, b / tot_b * 100.0);
    }
    println!("{:<20} {:>10.2} B {:>7}  {:>8.2} GB", "TOTAL", tot_p as f64 / 1e9, "", tot_b / 1e9);

    // What one token reads: every dense weight, one embedding row, and n_used of
    // n_expert experts per layer.
    let (ne, nu) = (512u64, 10u64);
    let (ep, eb) = cls.get("routed experts").copied().unwrap_or((0, 0.0));
    let dense = cls.get("dense skeleton").copied().unwrap_or((0, 0.0));
    let lm = cls.get("lm_head").copied().unwrap_or((0, 0.0));
    let act_exp_p = ep * nu / ne;
    println!("\nactive per token (the 'A3B'):");
    println!("  dense skeleton + lm_head  {:>6.2} B params", (dense.0 + lm.0) as f64 / 1e9);
    println!("  {nu} of {ne} experts x layers {:>6.2} B params", act_exp_p as f64 / 1e9);
    println!("  ACTIVE TOTAL              {:>6.2} B params", (dense.0 + lm.0 + act_exp_p) as f64 / 1e9);
    println!("\nbytes one token must READ:");
    println!("  dense as f16 (prec=4 dequantizes it)  {:>6.2} GB", (dense.0 + lm.0) as f64 * 2.0 / 1e9);
    println!("  dense at its on-disk quant            {:>6.2} GB", (dense.1 + lm.1) / 1e9);
    println!("  routed experts (native, mmap'd)       {:>6.2} GB", eb * nu as f64 / ne as f64 / 1e9);
    println!("\nbytes that must be RESIDENT to avoid disk:");
    println!("  dense as f16                          {:>6.2} GB", (dense.0 + lm.0 + cls.get("token_embd").map(|v| v.0).unwrap_or(0)) as f64 * 2.0 / 1e9);
    println!("  ALL experts (working set grows to it) {:>6.2} GB", eb / 1e9);
    Ok(())
}
