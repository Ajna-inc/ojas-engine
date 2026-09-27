//! Prints what the MTP draft head predicts next to what the model does.
//!
//! Acceptance alone cannot tell a mis-wired draft block from a weak one: both land well
//! above chance and well below useful. Printing both tokens as text separates them — a
//! wiring bug produces tokens unrelated to the context, a weak head produces plausible
//! continuations that simply differ.
//!
//! usage: mtp_probe <gguf> "<prompt>" [n] [prec]
use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: mtp_probe <gguf> \"<prompt>\" [n] [prec]");
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "What is the capital of France?".into());
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(24);
    let prec: u8 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    ojas_core::logging::init();
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let arch = g.arch();
    let bpe = ojas_tokenize::tokenizer::Bpe::from_gguf(&g);
    let ids: Vec<u32> = bpe.encode(&ojas_tokenize::tokenizer::chat_template(&arch, &prompt))
        .into_iter().map(|v| v as u32).collect();
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;
    if !m.has_mtp() { eprintln!("no MTP draft block"); std::process::exit(2); }

    m.reset_state();
    if std::env::var("OJAS_MTP_WARM").is_ok() {
        // Walk the prompt one token at a time so each forward leaves mtp_h holding that
        // position's hidden, then run the draft block at the next position so its own KV
        // cache ends up covering the prompt. The draft block at p consumes
        // (emb(t_p), h_{p-1}), hence the offset.
        for i in 0..ids.len() - 1 {
            m.forward_id(ids[i], i);
            if i + 1 < ids.len() { m.mtp_draft(ids[i + 1], i + 1, 0, false); }
        }
    } else {
        m.prefill(&ids[..ids.len() - 1], 0);
    }
    let mut pos = ids.len() - 1;
    let mut cur = ids[ids.len() - 1];
    let (mut hit, mut tot, mut hit2, mut echo) = (0usize, 0usize, 0usize, 0usize);
    // OJAS_MTP_HROW pins the verify row the draft conditions on, so the off-by-one can
    // be tested.
    let pin: Option<usize> = std::env::var("OJAS_MTP_HROW").ok().and_then(|v| v.parse().ok());
    let fill = std::env::var("OJAS_MTP_FILL").is_ok();
    let mut hrow = pin.unwrap_or(0);
    println!("\n  {:>4}  {:<18} {:<18} {:<7} {}", "pos", "model says", "draft says", "", "combiner h input");
    for _ in 0..n {
        let (h0, h1) = (m.mtp_h_norm(0), m.mtp_h_norm(1));
        let (a0, a1, draft) = m.mtp_step(cur, pos, hrow);
        let show = |t: u32| {
            let s = bpe.decode(t as usize);
            let s = s.replace('\n', "\\n");
            if s.trim().is_empty() && s != "\\n" { format!("{t}") } else { format!("{s:?}") }
        };
        tot += 1;
        let ok = draft == a0;
        if ok { hit += 1; }
        if draft == a1 { hit2 += 1; }
        if draft == cur { echo += 1; }
        println!("  {pos:>4}  {:<18} {:<18} {:<7} |h0|={h0:8.2} |h1|={h1:8.2} hrow={hrow}",
                 show(a0), show(draft), if ok { "match" } else if draft == cur { "ECHO" } else { "" });
        if ok {
            // Accepting skips a position where the draft block never ran, so its KV
            // cache would carry a hole that every later draft attends over; mtp_draft's
            // `head=false` form fills it.
            if fill { m.mtp_draft(draft, pos + 1, 0, false); }
            pos += 2; cur = a1; hrow = pin.unwrap_or(1);
        } else { m.mtp_rollback(); pos += 1; cur = a0; hrow = pin.unwrap_or(0); }
    }
    let pc = |x: usize| x as f64 * 100.0 / tot as f64;
    println!("\n  draft == next token   (t+1, what we verify as)  {hit}/{tot} = {:.0}%", pc(hit));
    println!("  draft == token AFTER  (t+2, MTP's own target)   {hit2}/{tot} = {:.0}%", pc(hit2));
    println!("  draft == its own input (echo)                   {echo}/{tot} = {:.0}%", pc(echo));
    Ok(())
}
