//! Repeated real-serving measurements with fresh recurrent state and output checks.
//! usage: serving_bench model prompt-file plain|mtp|ocr [tokens=32] [reps=3] [page-image]
//!
//! Three modes, one measurement shape:
//!
//! * `plain` — speculation off (`OJAS_NO_SPEC=1`), text prompt.
//! * `mtp` — speculation on, and the model must have a usable draft head.
//! * `ocr` — speculation on, and the prompt carries an image span: the page is
//!   preprocessed and run through the CPU ViT, and the projector rows enter the
//!   decoder through `Model::prefill_embeds` instead of through token ids.
//!
//! `ocr` reports a per-page number: the encode sits inside every repetition's timed
//! window, so `total_s` is encode + prefill + decode, matching how the baseline to
//! beat is stated (32.4 s/page median, 23.1 s/page sustained). A decode-only rate is
//! not comparable to it. `ttft_s` is page start to first token, so prefill alone is
//! `ttft_s - encode_s`.
//!
//! Two clock-independent gates, so they hold on a contended machine:
//!
//! * every repetition must emit the same output ids as the warmup; greedy decode is
//!   deterministic and speculation is exact, so a difference is state contamination
//!   between runs;
//! * every repetition's projector rows must hash the same (`rows_fnv`); the ViT is
//!   pure, so a difference means the encoder carried state or read uninitialized
//!   memory, and the per-page throughput would describe a transcription nobody can
//!   reproduce.
//!
//! `eos`/`eog` are left unset so every repetition decodes exactly `tokens` steps
//! instead of stopping at end-of-generation: comparable repetitions need an identical
//! amount of work.
use anyhow::{bail, ensure, Context, Result};
use ojas_core::Model;
use ojas_models::decoder::DecoderGpu;
use std::{cell::Cell, cell::RefCell, ops::Range, path::Path, time::Instant};

/// Substituted for the image span while the chat template is rendered, then split on.
/// Must be a string an instruction cannot contain and the tokenizer cannot merge
/// across; NUL bytes are both. Same marker as `ojas-cli/src/ocr.rs`, whose prompt
/// builder is crate-private, so the assembly is repeated here while the three things
/// that must not drift stay shared: `VitPreproc` (pixels), `CpuVit` (rows),
/// `chat_template` (scaffolding).
const SPAN_MARKER: &str = "\u{0}\u{0}ojas-image-span\u{0}\u{0}";

/// FNV-1a over the row bits, so two encodes differing by one ulp hash differently.
fn rows_fnv(v: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// Merged-token budget, read exactly as `ojas-cli/src/ocr.rs` reads it so the
/// benchmark and the command agree on how many patches a page is.
fn image_budget() -> (i32, i32) {
    let get = |k: &str, d: i32| -> i32 {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(d)
    };
    (get("OJAS_IMAGE_MIN_TOKENS", 8), get("OJAS_IMAGE_MAX_TOKENS", 4096))
}

/// The wrapped model in every respect except two: MTP calls are counted, and
/// prefill calls landing inside an image span are served from precomputed
/// projector rows. `span` empty = an ordinary text prompt and every call forwards.
struct Observed<'a> {
    m: &'a DecoderGpu<'a>,
    calls: &'a Cell<usize>,
    accepted: &'a Cell<usize>,
    /// Set if the decoder refuses an injection. Also checked up front, so it should be
    /// unreachable; if it fires, the run is not a page and must fail.
    refused: &'a Cell<bool>,
    /// Absolute prompt positions the image occupies.
    span: Range<usize>,
    /// `span.len() * d` f32, row-major. Behind a `RefCell` because the encode is re-run
    /// per repetition while the `EngineCore` must not be rebuilt: it carries the Token
    /// Recycling adjacency table, which `ojas-infer` persists across generate calls, and
    /// a fresh engine each run would reset it and make these samples incomparable with
    /// every serving-bench result already recorded.
    rows: RefCell<Vec<f32>>,
    d: usize,
    /// Whether the batched-verify primitives are forwarded to the real decoder.
    ///
    /// False for `plain` and `mtp`: this wrapper has never forwarded them, so those two
    /// modes measure what every serving-bench result already under `evidence/` measured
    /// — MTP drafts only, with prompt-lookup unable to verify because
    /// `forward_batch_ids` answers None. Forwarding them would reprice that history.
    ///
    /// True for `ocr`, because `ojas-cli`'s own `VisionPrefill` wrapper forwards all of
    /// them, so a per-page decode rate taken without them would not be the rate
    /// `ojas ocr` produces on the same page. It does not change the output: greedy
    /// speculation keeps a drafted token only where it equals the argmax, and the same
    /// page transcribed with and without these forwarded emitted byte-identical ids.
    forward_spec: bool,
}
impl Model for Observed<'_> {
    fn context_capacity(&self) -> usize { self.m.context_capacity() }
    fn mtp_verify_width(&self) -> usize { self.m.mtp_verify_width() }
    fn n_layers(&self) -> usize { self.m.n_layers() }
    fn hidden_dim(&self) -> usize { self.m.hidden_dim() }
    /// `EngineCore` prefills in 256-token chunks against an absolute position, so a
    /// chunk can be all text, all image, or straddle either edge — and a 4 096-row span
    /// straddles both. Split into the three sub-ranges and serve the middle one from
    /// rows.
    fn prefill(&self, t: &[u32], p: usize) {
        if self.span.is_empty() { self.m.prefill(t, p); return; }
        let end = p + t.len();
        let head_end = end.min(self.span.start).max(p);
        let img_end = end.min(self.span.end).max(head_end);
        let at = |lo: usize, hi: usize| &t[lo - p..hi - p];
        if head_end > p { self.m.prefill(at(p, head_end), p); }
        if img_end > head_end {
            let lo = (head_end - self.span.start) * self.d;
            let hi = (img_end - self.span.start) * self.d;
            let rows = self.rows.borrow();
            if !self.m.prefill_embeds(at(head_end, img_end), &rows[lo..hi], head_end, None) {
                self.refused.set(true);
            }
        }
        if end > img_end { self.m.prefill(at(img_end, end), img_end); }
    }
    fn prefill_embeds(&self, t: &[u32], x: &[f32], p: usize, p3: Option<&[[u32; 4]]>) -> bool {
        self.m.prefill_embeds(t, x, p, p3)
    }
    /// Zero while an image span is present. Reuse is a longest-common-prefix over raw
    /// ids and every image token is the same placeholder, so a match here would serve
    /// page N-1's KV rows for page N.
    fn reuse_prefix_len(&self, t: &[u32]) -> usize {
        if self.span.is_empty() { self.m.reuse_prefix_len(t) } else { 0 }
    }
    fn forward_id(&self, t: u32, p: usize) -> u32 { self.m.forward_id(t,p) }
    fn forward_logits(&self, t: u32, p: usize) -> Option<Vec<f32>> {
        if self.forward_spec { self.m.forward_logits(t,p) } else { None }
    }
    fn forward_batch_logits(&self, t: &[u32], p: usize) -> Option<Vec<Vec<f32>>> {
        if self.forward_spec { self.m.forward_batch_logits(t,p) } else { None }
    }
    fn forward_batch_ids(&self, t: &[u32], p: usize) -> Option<Vec<u32>> {
        if self.forward_spec { self.m.forward_batch_ids(t,p) } else { None }
    }
    fn forward_batch_topk(&self, t: &[u32], p: usize) -> Option<(Vec<u32>, Vec<u32>)> {
        if self.forward_spec { self.m.forward_batch_topk(t,p) } else { None }
    }
    fn forward_id_topk(&self, t: u32, p: usize) -> Option<(u32, [u32; 8])> {
        if self.forward_spec { self.m.forward_id_topk(t,p) } else { None }
    }
    fn has_mtp(&self) -> bool { self.m.has_mtp() }
    fn mtp_step_committed(&self, t: u32, p: usize) -> Option<Vec<u32>> {
        let out=self.m.mtp_step_committed(t,p)?;
        self.calls.set(self.calls.get()+1);
        self.accepted.set(self.accepted.get()+usize::from(out.len()>1));
        Some(out)
    }
    fn reset_session(&self) { self.m.reset_session() }
}
fn main() -> Result<()> {
    let a:Vec<_>=std::env::args().collect();
    ensure!(a.len()>=4,"usage: serving_bench model prompt-file plain|mtp|ocr [tokens] [reps] [page-image]");
    let mode=a[3].as_str();
    ensure!(matches!(mode,"plain"|"mtp"|"ocr"),"mode must be plain, mtp or ocr");
    let plain=mode=="plain";
    let ocr=mode=="ocr";
    // `ocr` keeps speculation on and forwards the primitives it needs (see
    // `Observed::forward_spec`): a page of `<div data-bbox=...>` is the repetitive text
    // prompt-lookup wins on, so forcing it off would measure a path `ojas ocr` never
    // takes. It stays exact either way — a drafted token is kept only where it equals
    // the argmax — so the output gates still hold.
    ensure!(ojas_core::config::EngineConfig::current().no_spec==plain,"plain requires OJAS_NO_SPEC=1; mtp and ocr require it unset");
    let count:usize=a.get(4).map(|s|s.parse()).transpose()?.unwrap_or(32);
    let reps:usize=a.get(5).map(|s|s.parse()).transpose()?.unwrap_or(3);
    // A page is 2 100-3 200 output tokens, so the 256-token text ceiling would cover a
    // tenth of one; and one encode per repetition is minutes on the CPU ViT, so an OCR
    // case may run a single measured sample. The determinism gates still cover it:
    // every run is compared against the warmup.
    let (max_count, min_reps) = if ocr { (4096, 1) } else { (256, 3) };
    ensure!(count>=2 && count<=max_count && (min_reps..=20).contains(&reps),"use 2..{max_count} output tokens and {min_reps}..20 repetitions");
    let page=a.get(6).map(String::as_str).filter(|s|!s.is_empty());
    ensure!(ocr==page.is_some(),"ocr mode requires a page image; plain and mtp must not be given one");

    let start=Instant::now();
    let gpu=ojas_metal::MetalGpu::new()?;
    let mut g=ojas_formats::gguf::Gguf::open(&a[1])?;
    let arch=g.arch();
    let bpe=ojas_tokenize::Bpe::from_gguf(&g);
    let text=std::fs::read_to_string(&a[2])?;

    // ---- the vision half, only for `ocr` ----------------------------------
    // Loaded before the decoder because the prompt length depends on how many
    // merged tokens this page is, and the decoder's context is sized from that.
    let mut vision: Option<(ojas_cpu::CpuVit, Vec<f32>, usize, usize)> = None;   // (vit, planar, w, h)
    let (mut n_image, mut patches, mut proj_w) = (0usize, 0usize, 0usize);
    if let Some(p)=page {
        let mmproj=ojas_formats::mmproj::discover(Path::new(&a[1]), ojas_core::config::EngineConfig::current().mmproj.as_deref())?
            .context("no vision projector: put a *mmproj*.gguf beside the model, or set OJAS_MMPROJ")?;
        let mut mg=ojas_formats::gguf::Gguf::open(mmproj.to_str().context("non-UTF8 mmproj path")?)?;
        // Catches a projector from a different checkpoint, which would otherwise surface
        // as fluent wrong text rather than as an error.
        ojas_formats::mmproj::validate(&g,&mg)?;
        let vit=ojas_cpu::CpuVit::load(&mut mg)?;
        proj_w=mg.meta_u32("clip.vision.projection_dim").context("mmproj has no clip.vision.projection_dim")? as usize;
        let (min_tok,max_tok)=image_budget();
        let pre=ojas_cpu::VitPreproc::qwen3vl().with_token_budget(min_tok,max_tok);
        let (w0,h0,rgb)=ojas_cpu::vit_preprocess::decode_rgb8_path(p).with_context(||format!("decoding {p}"))?;
        let (w,h,planar)=pre.preprocess(&rgb,w0,h0)?;
        n_image=vit.n_merged_tokens(w,h);
        patches=(w/pre.patch_size as usize)*(h/pre.patch_size as usize);
        ensure!(n_image>0,"page preprocessed to zero image tokens");
        // The preprocessor decides the pixel grid, the ViT how many rows come out of it.
        // If their patch/merge geometry disagreed, the prompt would reserve a different
        // number of placeholder positions than the encoder returns rows — caught by the
        // seam check below, but only after paying for a whole ViT pass.
        ensure!(patches==n_image*(pre.n_merge*pre.n_merge) as usize,
            "preprocessor and ViT disagree about the patch grid: {patches} patches for {n_image} merged tokens at merge {}",pre.n_merge);
        eprintln!("page {p}: {w0}x{h0} -> {w}x{h} | {n_image} image tokens | {patches} patches | budget {min_tok}..{max_tok}");
        vision=Some((vit,planar,w,h));
    }

    // ---- the prompt --------------------------------------------------------
    // Text modes encode the file verbatim, with no chat template. An OCR prompt is the
    // template with the image span spliced in as placeholder ids; the span never reaches
    // the tokenizer, because `Bpe::encode` rescans every special token per iteration and
    // 4 096 literal `<|image_pad|>` is millions of substring searches.
    let enc=|s:&str|->Vec<u32>{bpe.encode(s).into_iter().map(|v|v as u32).collect()};
    let (prompt, span): (Vec<u32>, Range<usize>) = if ocr {
        let user=format!("<|vision_start|>{SPAN_MARKER}<|vision_end|>{text}");
        let full=ojas_tokenize::chat_template(&arch,&user);
        let (head,tail)=full.split_once(SPAN_MARKER).context("chat template dropped the image-span marker")?;
        let one=|s:&str|->Result<u32>{match bpe.encode(s).as_slice(){[id]=>Ok(*id as u32),o=>bail!("no single {s} token (encodes to {} pieces) — not a vision model",o.len())}};
        let (start_id,pad,end_id)=(one("<|vision_start|>")?,one("<|image_pad|>")?,one("<|vision_end|>")?);
        let (hi,ti)=(enc(head),enc(tail));
        // The halves were split around the markers, so they must still carry them; an
        // unmarked span is read by the model as ordinary text.
        ensure!(hi.last()==Some(&start_id),"templated prompt does not end its prefix with <|vision_start|>");
        ensure!(ti.first()==Some(&end_id),"templated prompt does not begin its suffix with <|vision_end|>");
        let mut ids=hi; let at=ids.len();
        ids.resize(at+n_image,pad); ids.extend_from_slice(&ti);
        (ids, at..at+n_image)
    } else {
        (enc(&text), 0..0)
    };
    ensure!(!prompt.is_empty(),"empty prompt");
    let m=DecoderGpu::load(&gpu,&mut g,512.max(prompt.len()+count),4,None,None)?;
    ensure!(prompt.len()+count<=m.context_capacity(),"workload exceeds context");
    ensure!(plain || ocr || m.has_mtp(),"MTP fixture has no usable draft");

    // ---- the encoder seam, checked once before any timed run ---------------
    // `prefill_embeds` answers per call, which is too late: by then the prompt is half
    // prefilled and whatever comes out is nonsense. One throwaway row is cheap next to
    // a ViT pass.
    let mut encode_s=0.0f64;
    let mut rows_ref=0u64;
    if let Some((vit,planar,w,h))=&vision {
        ensure!(m.hidden_dim()==proj_w,"projector writes {proj_w}-wide rows but the decoder's hidden dim is {} — mmproj and model are from different checkpoints",m.hidden_dim());
        m.reset_session();
        let ok=m.prefill_embeds(&[0u32],&vec![0.0f32;m.hidden_dim()],0,None);
        m.reset_session();
        ensure!(ok,"this decoder has no embedding-injection path (Model::prefill_embeds returned false), so an image span cannot enter it");
        let t=Instant::now();
        let rows=vit.forward(planar,*w,*h)?;
        encode_s=t.elapsed().as_secs_f64();
        ensure!(rows.len()==n_image*m.hidden_dim(),"encoder returned {} f32 for {n_image} rows of {} — the seam's contract is n_tokens * hidden_dim, row-major",rows.len(),m.hidden_dim());
        rows_ref=rows_fnv(&rows);
        eprintln!("encoded in {encode_s:.1}s (cpu-vit), rows_fnv {rows_ref:#x}");
    }
    println!("{{\"load_s\":{},\"prompt_tokens\":{:?},\"mode\":\"{}\",\"precision\":4,\"warmups\":1,\"encode_s\":{},\"image_tokens\":{},\"patches\":{},\"image\":{}}}",start.elapsed().as_secs_f64(),prompt,mode,encode_s,n_image,patches,page.map(|p|format!("{:?}",p)).unwrap_or("null".into()));
    let calls=Cell::new(0);let accepted=Cell::new(0);let refused=Cell::new(false);
    let engine=ojas_infer::EngineCore::new(Observed{m:&m,calls:&calls,accepted:&accepted,refused:&refused,span:span.clone(),rows:RefCell::new(Vec::new()),d:m.hidden_dim(),forward_spec:ocr});
    // Access counters through the wrapper reference retained outside EngineCore.
    // Output equality against the warmup guards repeated-run state contamination.
    let mut reference=None;
    for run in 0..=reps {
        m.reset_session();calls.set(0);accepted.set(0);refused.set(false);
        let load=ojas_models::bench::load_average();
        // The timer opens before the encode: an OCR sample is a whole page and the ViT
        // is the largest single compute item in the pipeline. Text modes have no encode,
        // so their fields are unchanged.
        let start=Instant::now();let mut first=None;
        let (rows,enc_s)=match &vision {
            Some((vit,planar,w,h))=>{let t=Instant::now();let r=vit.forward(planar,*w,*h)?;(r,t.elapsed().as_secs_f64())}
            None=>(Vec::new(),0.0),
        };
        let fnv=if vision.is_some() {rows_fnv(&rows)} else {0};
        ensure!(vision.is_none() || fnv==rows_ref,"repeated encode of the same page changed the projector rows");
        *engine.model().rows.borrow_mut()=rows;
        let out=engine.generate_with(&prompt,count,None,&mut|_,_|{},&mut |_|{first.get_or_insert_with(||start.elapsed().as_secs_f64());true});
        let total=start.elapsed().as_secs_f64();let ttft=first.ok_or_else(||anyhow::anyhow!("generation emitted no tokens"))?;
        ensure!(!refused.get(),"the decoder refused an embedding injection mid-prompt; this run is not a transcription of that page");
        ensure!(out.len()==count,"generation ended before requested output length");
        // MTP engagement is only asserted where the mode claims it. `ocr` does not:
        // surya-2 has no draft head, and a page-throughput case does not measure it.
        ensure!(!plain || calls.get()==0,"plain mode observed MTP engagement");
        ensure!(mode!="mtp" || calls.get()>0,"mtp mode observed no MTP engagement");
        if let Some(want)=&reference {ensure!(&out==want,"repeated fresh run changed output");} else {reference=Some(out.clone());}
        if run>0 {println!("{{\"run\":{run},\"total_s\":{total},\"ttft_s\":{ttft},\"encode_s\":{enc_s},\"decode_tps\":{},\"load_average\":{},\"mtp_calls\":{},\"accepted_calls\":{},\"image_tokens\":{},\"patches\":{},\"rows_fnv\":{fnv},\"output_tokens\":{:?}}}",(count-1) as f64/(total-ttft),load.map(|x|x.to_string()).unwrap_or("null".into()),calls.get(),accepted.get(),n_image,patches,out);}
    }
    Ok(())
}
