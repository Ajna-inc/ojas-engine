//! Diffs the KV cache written by per-token prefill against the batched prefill path.
//!
//! Batched prefill produces garbage on qk-norm archs (Qwen3/Qwen3.5/Gemma3) while the
//! per-token path is correct, and the forward-over-cache has been cleared (path_agree
//! passes). That isolates the fault to the batched cache write, but token output cannot
//! say whether K or V is wrong, or at which position.
//!
//! This prefills the same tokens both ways in one process and diffs layer 0's
//! cache position by position, splitting K from V and reporting where the first
//! divergence is and how large. The per-token path is the reference.
//!
//! usage: kv_diff <gguf> [n_tokens] [prec] [layer]

use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: kv_diff <gguf> [n] [prec] [layer]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(9);
    let prec: u8 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let layer: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;
    let toks: Vec<u32> = (0..n).map(|i| 100u32 + (i * 37 % 400) as u32).collect();

    // Reference: feed the tokens one at a time through the decode path, the path known
    // to produce correct output on these models.
    m.reset_state();
    for (i, &t) in toks.iter().enumerate() { m.forward_id(t, i); }
    let (k_ref, v_ref) = m.dump_kv(layer, 0, n);

    // Under test: the batched path writing the whole cache from base_pos = 0.
    m.reset_state();
    m.prefill(&toks, 0);
    let (k_bat, v_bat) = m.dump_kv(layer, 0, n);

    let kvdim = k_ref.len() / n;
    println!("layer {layer}, {n} positions, kvdim {kvdim}, prec {prec}");
    println!("pos    K maxdiff   K nbad     V maxdiff   V nbad");
    let mut first_bad = None;
    for p in 0..n {
        let sl = |v: &Vec<f32>| v[p * kvdim..(p + 1) * kvdim].to_vec();
        let (kr, kb, vr, vb) = (sl(&k_ref), sl(&k_bat), sl(&v_ref), sl(&v_bat));
        // f16 storage, so treat a bit of noise as equal; a real divergence is huge
        let cmp = |a: &Vec<f32>, b: &Vec<f32>| -> (f32, usize) {
            let mut mx = 0.0f32; let mut nb = 0usize;
            for (x, y) in a.iter().zip(b) {
                let d = (x - y).abs();
                if d > 1e-3 { nb += 1; }
                if d > mx { mx = d; }
            }
            (mx, nb)
        };
        let (km, kn) = cmp(&kr, &kb);
        let (vm, vn) = cmp(&vr, &vb);
        println!("{p:3}   {km:9.5}   {kn:5}     {vm:9.5}   {vn:5}");
        if first_bad.is_none() && (kn > 0 || vn > 0) { first_bad = Some((p, kn > 0, vn > 0)); }
    }
    match first_bad {
        None => println!("\nCACHES MATCH — the batched write is faithful; look elsewhere."),
        Some((p, k, v)) => {
            println!("\nfirst divergence at position {p}: K={} V={}", if k {"WRONG"} else {"ok"}, if v {"WRONG"} else {"ok"});
            // Head and dims within it separate a rope-pairing bug (a specific dim
            // pattern) from an indexing bug (whole head shifted).
            let hd_guess = [64usize, 96, 128, 256].into_iter().find(|h| kvdim % h == 0).unwrap_or(kvdim);
            let heads = kvdim / hd_guess;
            print!("  per-head bad-dim counts (hd={hd_guess}, {heads} heads) K:");
            for h in 0..heads {
                let bad = (0..hd_guess).filter(|&i| {
                    let idx = p * kvdim + h * hd_guess + i;
                    (k_ref[idx] - k_bat[idx]).abs() > 1e-3
                }).count();
                print!(" {bad}");
            }
            println!();
            print!("  first 8 K dims of head 0  ref:");
            for i in 0..8.min(hd_guess) { print!(" {:.4}", k_ref[p * kvdim + i]); }
            print!("\n                            bat:");
            for i in 0..8.min(hd_guess) { print!(" {:.4}", k_bat[p * kvdim + i]); }
            println!();
        }
    }
    Ok(())
}
