//! surya-2 page gate: every page in the fixture directory through the CUDA runner and through
//! the CPU oracle (exact F16), with line-level agreement between the two and against the
//! llama.cpp CUDA reference text (`<pages>/ref_llamacpp/<page>.txt`).
//!
//! ```text
//! SP=~/.local/lib/python3.10/site-packages/nvidia
//! LD_LIBRARY_PATH=$SP/cuda_nvrtc/lib:$SP/cublas/lib OJAS_CUDA_INCLUDE=$SP/cuda_runtime/include \
//!   cargo run --release -p ojas-cuda --example surya_page_gate -- [flags] [page.jpg ...]
//! ```
//!
//! Flags:
//! * `--pages DIR`      fixture directory (default: `<HDD>/ocr-pages`)
//! * `--out DIR`        where texts and the summary go (default: `<HDD>/surya-cuda/gate`)
//! * `--cpu-max N`      oracle decode cap in tokens (default 512; 0 = skip the CPU side)
//! * `--cuda-max N`     CUDA decode cap (default 12000)
//! * `--vit-attn MODE`  CUDA tower attention: x3 (default, split-f16 tensor cores) | f16 | f32
//! * `--cpu-from DIR`   reuse the oracle outputs a previous run left in DIR (`<page>.cpu.ids`,
//!   else `<page>.cpu.txt`) instead of running the CPU side again
//! * `--gemm MODE`      fast | split | exact (default split, `OJAS_CUDA_GEMM`)
//! * `--budget N`       merged-token budget of the preprocessor (default 4096, as `ojas ocr`)
//!
//! The prompt, the anchored M-RoPE layout, the stop ids and the repetition guard are
//! `ojas ocr`'s (`crates/ojas-cli/src/ocr.rs`), transcribed: ChatML head injected at positions
//! shifted by `nx*ny - max(nx,ny)`, image rows at `(t0, t0 + i/nx, t0 + i%nx, 0)`, tail by id
//! at its cache rows, greedy decode from there.
//!
//! Agreement is reported three ways per pair of outputs:
//! * first difference: the first generated token (CPU vs CUDA) or character (vs the
//!   reference) where the two part;
//! * lines: the HTML is flattened to text lines (block tags and `<br/>` end a line, cell tags
//!   become ` | `), and `exact` is the longest common subsequence of identical lines over the
//!   longer side's line count;
//! * CER: character edit distance over the flattened text, and over the reference length.
//!
//! The oracle is capped, so CUDA-vs-CPU compares the CUDA output truncated to the oracle's
//! token count. Pages whose reference loops (llama.cpp has no loop guard) are compared on the
//! reference prefix as long as ours.

use anyhow::{bail, Context, Result};
use ojas_core::Model;
use ojas_cpu::cpu_ssm::{image_pos3, text_pos3, CpuSsm, CpuSsmOpts};
use ojas_cuda::{CudaSsm, CudaSsmOpts, GemmMode, VitAttn};
use ojas_formats::gguf::Gguf;
use std::path::{Path, PathBuf};
use std::time::Instant;

const HDD: &str = "/data/dev-cache";
const INSTRUCTION: &str = "OCR this image to HTML. Each block is a div with data-label and data-bbox \
     (x0 y0 x1 y1, normalized 0-1000).";
const LOOP_MAX_PERIOD: usize = 128;
const LOOP_MIN_RUN: usize = 24;

fn find_surya(file: &str, env: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env) {
        return Some(PathBuf::from(p));
    }
    let snaps = Path::new(HDD).join("hf/hub/models--datalab-to--surya-ocr-2-gguf/snapshots");
    std::fs::read_dir(snaps).ok()?.flatten().map(|e| e.path().join(file)).find(|p| p.exists())
}

/// `ocr.rs::loop_period`, verbatim.
fn loop_period(gen: &[u32], max_period: usize, min_run: usize) -> Option<usize> {
    for p in 1..=max_period {
        let reps = std::cmp::max(4, min_run.div_ceil(p));
        let need = p * reps;
        if gen.len() < need {
            continue;
        }
        let tail = &gen[gen.len() - need..];
        if tail.chunks_exact(p).all(|c| c == &tail[..p]) {
            return Some(p);
        }
    }
    None
}

struct Prompt {
    head: Vec<u32>,
    tail: Vec<u32>,
    pad: u32,
}

impl Prompt {
    fn new(bpe: &ojas_tokenize::Bpe) -> Result<Prompt> {
        const MARK: &str = "\u{0}\u{0}ojas-image-span\u{0}\u{0}";
        let full = ojas_tokenize::chat_template("qwen35", &format!("<|vision_start|>{MARK}<|vision_end|>{INSTRUCTION}"));
        let (h, t) = full.split_once(MARK).context("template dropped the span marker")?;
        let enc = |s: &str| bpe.encode(s).into_iter().map(|v| v as u32).collect::<Vec<u32>>();
        let pad = enc("<|image_pad|>");
        anyhow::ensure!(pad.len() == 1, "no single <|image_pad|> token");
        Ok(Prompt { head: enc(h), tail: enc(t), pad: pad[0] })
    }
}

struct Run {
    ids: Vec<u32>,
    text: String,
    stop: &'static str,
    vit_s: f64,
    prefill_s: f64,
    decode_s: f64,
    prompt_len: usize,
}

/// One page through one model: encode, prefill the anchored prompt, greedy decode.
#[allow(clippy::too_many_arguments)]
fn run_page(m: &dyn Model, bpe: &ojas_tokenize::Bpe, pr: &Prompt, embd: &dyn Fn(u32) -> Vec<f32>, img: &[f32], w: usize, h: usize,
            max_new: usize, eog: &[u32]) -> Result<Run> {
    let t0 = Instant::now();
    let rows = m.encode_image(img, w, h).context("model has no vision tower")??;
    let vit_s = t0.elapsed().as_secs_f64();
    let (nx, ny) = (w / 32, h / 32);
    let n = nx * ny;
    anyhow::ensure!(rows.len() == n * m.hidden_dim(), "tower returned {} values for {n} rows", rows.len());
    let delta = (n - nx.max(ny)) as u32;
    let head_rows: Vec<f32> = pr.head.iter().flat_map(|&t| embd(t)).collect();
    let t0 = Instant::now();
    m.reset_session();
    anyhow::ensure!(m.prefill_embeds(&pr.head, &head_rows, 0, Some(&text_pos3(delta, pr.head.len()))), "head refused");
    let at = pr.head.len();
    anyhow::ensure!(m.prefill_embeds(&vec![pr.pad; n], &rows, at, Some(&image_pos3(at as u32 + delta, nx, ny))), "image refused");
    let tail_at = at + n;
    m.prefill(&pr.tail[..pr.tail.len() - 1], tail_at);
    let prefill_s = t0.elapsed().as_secs_f64();
    let mut cur = pr.tail[pr.tail.len() - 1];
    let mut pos = tail_at + pr.tail.len() - 1;
    let t0 = Instant::now();
    let mut ids = Vec::new();
    let mut stop = "cap";
    while ids.len() < max_new {
        let id = m.forward_id(cur, pos);
        if eog.contains(&id) {
            stop = "eog";
            break;
        }
        ids.push(id);
        if loop_period(&ids, LOOP_MAX_PERIOD, LOOP_MIN_RUN).is_some() {
            stop = "loop";
            break;
        }
        cur = id;
        pos += 1;
    }
    let decode_s = t0.elapsed().as_secs_f64();
    let bytes: Vec<u8> = ids.iter().flat_map(|&i| bpe.decode_bytes(i as usize)).collect();
    Ok(Run { text: String::from_utf8_lossy(&bytes).into_owned(), ids, stop, vit_s, prefill_s, decode_s, prompt_len: tail_at + pr.tail.len() })
}

fn ids_bytes(ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ------------------------------------------------------------------ agreement

/// Flatten surya's HTML to text lines: block ends and `<br>` end a line, table cells become
/// ` | `, other tags vanish, the common entities are decoded.
fn html_lines(s: &str) -> Vec<String> {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find('<') {
        out.push_str(&rest[..i]);
        let Some(j) = rest[i..].find('>') else { out.push_str(&rest[i..]); rest = ""; break };
        let tag = rest[i + 1..i + j].trim_start_matches('/').to_ascii_lowercase();
        let name: String = tag.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
        match name.as_str() {
            "br" | "p" | "div" | "tr" | "li" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "table" | "ul" | "ol" => out.push('\n'),
            "td" | "th" => out.push_str(" | "),
            _ => {}
        }
        rest = &rest[i + j + 1..];
    }
    out.push_str(rest);
    let out = out.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'").replace("&nbsp;", " ");
    out.lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .map(|l| l.trim_matches(|c| c == '|' || c == ' ').to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn lcs(a: &[String], b: &[String]) -> usize {
    let mut prev = vec![0usize; b.len() + 1];
    for x in a {
        let mut cur = vec![0usize; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y { prev[j] + 1 } else { prev[j + 1].max(cur[j]) };
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Levenshtein distance over chars (two rows, O(n*m) time).
fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &x) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + (x != y) as usize).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

struct Agree {
    lines_a: usize,
    lines_b: usize,
    exact_lines: usize,
    edit: usize,
    ref_chars: usize,
    first_char_diff: Option<usize>,
}

impl Agree {
    fn of(got: &str, want: &str) -> Agree {
        let (a, b) = (html_lines(got), html_lines(want));
        let ta: Vec<char> = a.join("\n").chars().collect();
        let tb: Vec<char> = b.join("\n").chars().collect();
        let first = got.chars().zip(want.chars()).position(|(x, y)| x != y)
            .or_else(|| (got.chars().count() != want.chars().count()).then(|| got.chars().count().min(want.chars().count())));
        Agree { lines_a: a.len(), lines_b: b.len(), exact_lines: lcs(&a, &b), edit: edit_distance(&ta, &tb), ref_chars: tb.len(), first_char_diff: first }
    }
    fn line_rate(&self) -> f64 {
        let n = self.lines_a.max(self.lines_b);
        if n == 0 { 1.0 } else { self.exact_lines as f64 / n as f64 }
    }
    fn cer(&self) -> f64 { self.edit as f64 / self.ref_chars.max(1) as f64 }
    fn cell(&self) -> String {
        format!("{}/{} lines ({:.1}%), edit {} (CER {:.2}%), first diff {}", self.exact_lines, self.lines_a.max(self.lines_b),
                100.0 * self.line_rate(), self.edit, 100.0 * self.cer(),
                self.first_char_diff.map_or("none".to_string(), |c| format!("@char {c}")))
    }
}

// ----------------------------------------------------------------------- main

struct Args {
    pages: PathBuf,
    out: PathBuf,
    cpu_max: usize,
    cuda_max: usize,
    vit_attn: VitAttn,
    cpu_from: Option<PathBuf>,
    gemm: GemmMode,
    budget: i32,
    only: Vec<String>,
}

fn args() -> Result<Args> {
    let mut a = Args {
        pages: Path::new(HDD).join("ocr-pages"),
        out: Path::new(HDD).join("surya-cuda/gate"),
        cpu_max: 512,
        cuda_max: 12000,
        vit_attn: VitAttn::from_env(),
        cpu_from: None,
        gemm: GemmMode::from_env(),
        budget: 4096,
        only: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        let mut val = || it.next().context("flag needs a value");
        match x.as_str() {
            "--pages" => a.pages = val()?.into(),
            "--out" => a.out = val()?.into(),
            "--cpu-max" => a.cpu_max = val()?.parse()?,
            "--cuda-max" => a.cuda_max = val()?.parse()?,
            "--budget" => a.budget = val()?.parse()?,
            "--cpu-from" => a.cpu_from = Some(val()?.into()),
            "--vit-attn" => a.vit_attn = match val()?.as_str() {
                "f16" => VitAttn::F16,
                "x3" => VitAttn::X3,
                "f32" => VitAttn::F32,
                o => bail!("--vit-attn {o}: want x3, f16 or f32"),
            },
            "--gemm" => a.gemm = match val()?.as_str() {
                "fast" => GemmMode::Fast,
                "split" => GemmMode::Split,
                "exact" => GemmMode::Exact,
                o => bail!("--gemm {o}: want fast, split or exact"),
            },
            f if f.starts_with("--") => bail!("unknown flag {f}"),
            p => a.only.push(p.to_string()),
        }
    }
    Ok(a)
}

fn page_list(a: &Args) -> Result<Vec<String>> {
    if !a.only.is_empty() {
        return Ok(a.only.clone());
    }
    // manifest order when there is one, else every image in the directory
    if let Ok(s) = std::fs::read_to_string(a.pages.join("manifest.json")) {
        let v: serde_json::Value = serde_json::from_str(&s)?;
        if let Some(ps) = v["pages"].as_array() {
            return Ok(ps.iter().filter_map(|p| p["file"].as_str().map(String::from)).collect());
        }
    }
    let mut v: Vec<String> = std::fs::read_dir(&a.pages)?.flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| [".jpg", ".jpeg", ".png"].iter().any(|x| n.to_ascii_lowercase().ends_with(x)))
        .collect();
    v.sort();
    Ok(v)
}

fn main() -> Result<()> {
    let a = args()?;
    let model = find_surya("surya-2.gguf", "OJAS_SURYA_GGUF").context("surya-2.gguf not found (OJAS_SURYA_GGUF)")?;
    let mmproj = find_surya("surya-2-mmproj.gguf", "OJAS_SURYA_MMPROJ").context("surya-2-mmproj.gguf not found (OJAS_SURYA_MMPROJ)")?;
    std::fs::create_dir_all(&a.out)?;
    let pages = page_list(&a)?;

    let mut g = Gguf::open(model.to_str().unwrap())?;
    let bpe = ojas_tokenize::Bpe::from_gguf(&g);
    let eog = ojas_tokenize::eog_token_ids(&g, "qwen35");
    let pr = Prompt::new(&bpe)?;

    // the ChatML head's rows, as `ocr.rs::EmbedTable` gathers them (the file's f16, widened)
    let (dims, ty, emb) = g.read_tensor("token_embd.weight")?;
    anyhow::ensure!(ty == 1, "token_embd is not f16");
    let d = dims[0] as usize;
    let embd = move |t: u32| -> Vec<f32> {
        emb[t as usize * d * 2..(t as usize + 1) * d * 2].chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()
    };

    let mut cuda = CudaSsm::load_with(&mut g, CudaSsmOpts { gemm: a.gemm, context: 16384, ..CudaSsmOpts::default() })?;
    cuda.set_vit_attn(a.vit_attn);
    cuda.attach_vit_gguf(&mut Gguf::open(mmproj.to_str().unwrap())?)?;
    let cpu = if a.cpu_max > 0 && a.cpu_from.is_none() {
        let mut g = Gguf::open(model.to_str().unwrap())?;
        let mut c = CpuSsm::load_with(&mut g, CpuSsmOpts { exact: true })?;
        c.attach_vit(ojas_cpu::CpuVit::load(&mut Gguf::open(mmproj.to_str().unwrap())?)?)?;
        Some(c)
    } else {
        None
    };
    let pre = ojas_cpu::VitPreproc::qwen3vl().with_token_budget(8, a.budget);
    let cfg = format!("gemm {:?}, ViT attention {:?}, budget {}, oracle {}", a.gemm, a.vit_attn, a.budget,
                      match &a.cpu_from { Some(d) => format!("from {}", d.display()), None => format!("cap {} tokens", a.cpu_max) });
    eprintln!("surya page gate | {cfg}");

    let mut rows = Vec::new();
    for name in &pages {
        let path = a.pages.join(name);
        let stem = Path::new(name).file_stem().unwrap().to_string_lossy().into_owned();
        let (w0, h0, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(&path).with_context(|| format!("decoding {}", path.display()))?;
        let (w, h, img) = pre.preprocess(&rgb, w0, h0)?;
        eprintln!("\n[{name}] {w0}x{h0} -> {w}x{h} ({} image tokens)", (w / 32) * (h / 32));

        let g_run = run_page(&cuda, &bpe, &pr, &embd, &img, w, h, a.cuda_max, &eog)?;
        std::fs::write(a.out.join(format!("{stem}.cuda.txt")), &g_run.text)?;
        std::fs::write(a.out.join(format!("{stem}.cuda.ids")), ids_bytes(&g_run.ids))?;
        let tps = g_run.ids.len() as f64 / g_run.decode_s.max(1e-9);
        eprintln!("  cuda: ViT {:.2}s | prefill {} rows {:.2}s | {} tokens ({}) {:.1}s = {tps:.1} tok/s",
                  g_run.vit_s, g_run.prompt_len - 1, g_run.prefill_s, g_run.ids.len(), g_run.stop, g_run.decode_s);

        let refp = a.pages.join("ref_llamacpp").join(format!("{stem}.txt"));
        let reference = std::fs::read_to_string(&refp).ok().map(|s| s.trim().to_string());
        let vs_ref = reference.as_ref().map(|r| {
            // the reference loops on some pages (no guard): judge on its prefix as long as ours
            let ours = g_run.text.trim();
            let r = if g_run.stop != "eog" { r.chars().take(ours.chars().count()).collect::<String>() } else { r.clone() };
            Agree::of(ours, &r)
        });

        // the oracle: run now, or read back from an earlier run (ids, else text)
        enum Oracle { Ids(Vec<u32>), Text(String) }
        let oracle: Option<Oracle> = match (&cpu, &a.cpu_from) {
            (Some(c), _) => {
                let c_run = run_page(c, &bpe, &pr, &embd, &img, w, h, a.cpu_max, &eog)?;
                std::fs::write(a.out.join(format!("{stem}.cpu.txt")), &c_run.text)?;
                std::fs::write(a.out.join(format!("{stem}.cpu.ids")), ids_bytes(&c_run.ids))?;
                eprintln!("  cpu : ViT {:.1}s | prefill {:.1}s | {} tokens ({}) {:.1}s", c_run.vit_s, c_run.prefill_s,
                          c_run.ids.len(), c_run.stop, c_run.decode_s);
                Some(Oracle::Ids(c_run.ids))
            }
            (None, Some(dir)) => match std::fs::read(dir.join(format!("{stem}.cpu.ids"))) {
                Ok(b) => Some(Oracle::Ids(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())),
                Err(_) => std::fs::read_to_string(dir.join(format!("{stem}.cpu.txt"))).ok().map(Oracle::Text),
            },
            _ => None,
        };
        let (vs_cpu, cpu_note) = match &oracle {
            Some(Oracle::Ids(c_ids)) => {
                let first_tok = g_run.ids.iter().zip(c_ids).position(|(x, y)| x != y);
                let k = c_ids.len().min(g_run.ids.len());
                let g_prefix: Vec<u8> = g_run.ids[..k].iter().flat_map(|&i| bpe.decode_bytes(i as usize)).collect();
                let c_prefix: Vec<u8> = c_ids[..k].iter().flat_map(|&i| bpe.decode_bytes(i as usize)).collect();
                let ag = Agree::of(&String::from_utf8_lossy(&g_prefix), &String::from_utf8_lossy(&c_prefix));
                let note = format!("{} tok compared, first token diff {}", k, first_tok.map_or("none".into(), |t| format!("@{t}")));
                (Some(ag), note)
            }
            Some(Oracle::Text(t)) => {
                // an older run saved text only: compare our text cut to the oracle's length
                let n = t.chars().count();
                let ours: String = g_run.text.chars().take(n).collect();
                (Some(Agree::of(&ours, t)), format!("{n} chars compared (text)"))
            }
            None => (None, "skipped".into()),
        };
        if let Some(r) = &vs_ref { eprintln!("  vs llama.cpp: {}", r.cell()); }
        if let Some(c) = &vs_cpu { eprintln!("  vs cpu oracle: {} | {cpu_note}", c.cell()); }
        rows.push((name.clone(), g_run.prompt_len, g_run.ids.len(), g_run.stop, tps, g_run.vit_s, g_run.prefill_s, vs_cpu, cpu_note, vs_ref));
    }

    let mut md = format!("# surya-2 CUDA page gate\n\n{cfg}\n\n| page | prompt | out tok | stop | tok/s | ViT s | prefill s | vs CPU oracle (prefix) | vs llama.cpp |\n|---|---:|---:|---|---:|---:|---:|---|---|\n");
    for (name, pl, n, stop, tps, vs, ps, vc, cn, vr) in &rows {
        md += &format!("| {name} | {pl} | {n} | {stop} | {tps:.1} | {vs:.2} | {ps:.2} | {} | {} |\n",
                       vc.as_ref().map_or("-".to_string(), |c| format!("{} ; {cn}", c.cell())),
                       vr.as_ref().map_or("-".to_string(), |r| r.cell()));
    }
    std::fs::write(a.out.join("summary.md"), &md)?;
    println!("{md}");
    Ok(())
}
