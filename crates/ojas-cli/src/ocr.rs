//! `ojas ocr` — transcribe a page image (or a directory of them) with a
//! vision-projector GGUF attached to a text decoder.
//!
//! This module is the surface around a vision encoder, not the encoder. The
//! encoder sits behind [`VisionEncoder`], with two implementors:
//! [`ModelVitEncoder`] (the decoder's own tower via `Model::encode_image`, used by
//! real runs) and [`CpuVitEncoder`] (a wrapper over [`ojas_cpu::CpuVit`]: correct,
//! deterministic, ~500x slower, used as a numerical oracle). [`open_encoder`] is
//! the only place that choice is made, and the banner names the result.
//!
//! Five non-obvious constraints:
//!
//! 1. The image span never reaches the tokenizer. `Bpe::encode` scans ~1 948
//!    specials per iteration (`tokenizer.rs:94`), so 4 096 literal `<|image_pad|>`
//!    occurrences would be ~8 M substring searches over a 55 KB string. Text
//!    scaffolding is templated and encoded; the span is assembled as ids.
//! 2. The span enters the decoder through `Model::prefill_embeds`, not ids.
//!    [`VisionPrefill`] wraps the real `Model` and rewrites only the prefill calls
//!    that land inside the image span, so `EngineCore` and
//!    [`crate::cmds::stream`] are reused verbatim: the engine keeps chunking,
//!    positions, sampling, speculation, TTFT accounting and the partial-UTF-8
//!    detokenizer, and never learns that part of the prompt was pixels.
//! 3. Prefix reuse is a correctness hazard here. Every image token is the literal
//!    id 11, so a longest-common-prefix over ids matches across different pages
//!    and would transcribe page N as page N-1. `prefill_embeds` disowns its span;
//!    this file also reports `reuse_prefix_len == 0` unconditionally and calls
//!    `reset_session()` before every page.
//! 4. A page prompt does not fit the default context: 4 096 image tokens plus
//!    ChatML scaffolding is ~4 220 ids against a 4 096 default, so the `ocr` arm
//!    defaults to [`DEFAULT_CTX`]. A prompt that still does not fit errors,
//!    naming both levers.
//! 5. Image rows are not at sequential positions, and the text around them is not
//!    where its cache row says. llama.cpp gives merged token `i` the M-RoPE
//!    coordinate `(t, t + i/nx, t + i%nx, 0)` and advances the sequence counter
//!    by `max(nx, ny)`, not `nx*ny`; the cache row must stay contiguous, so the
//!    counters diverge by `nx*ny - max(nx,ny)` (552 on a 24x24 page). The decode
//!    path (`Model::forward_id`) has no M-RoPE argument, so every generated
//!    token's angle is its cache row. [`span_positions`] anchors the prompt on
//!    the counter the decode loop can express.
//!
//! Portability: `ojas-cpu` is not gated behind macOS, so this command compiles
//! and runs everywhere. What it needs from the *decoder* is
//! `Model::prefill_embeds`, which `DecoderGpu` and the CPU `CpuSsm` (qwen35)
//! implement; a decoder without it fails at an explicit capability probe with a
//! message saying so, rather than silently transcribing the placeholder ids.

use crate::backend::{with_model, ModelInfo};
use crate::cmds::{banner, check_fits, stop_ids, stream};
use crate::flags::RunOpts;
use anyhow::{bail, Context, Result};
use ojas_core::Model;
use ojas_cpu::VitPreproc;
use ojas_formats::gguf::Gguf;
use ojas_infer::{EngineCore, SampleOpts};
use ojas_tokenize::Bpe;
use std::ops::Range;
use std::path::{Path, PathBuf};

/// Context default for `ocr` only. A full page is 4 096 image tokens plus ~125
/// of ChatML scaffolding and instruction, which does not fit the engine-wide
/// 4 096 default (`main.rs:77`) — and `check_fits` hard-errors rather than
/// clipping. `-c/--ctx-size` still wins; this only changes what "unset" means on
/// this one path.
pub const DEFAULT_CTX: usize = 8192;

/// The prompt surya-2 is trained on for full-page OCR, transcribed from
/// `surya/recognition/prompts.py`. `-p/--prompt` or `-f/--file` replaces it.
pub const DEFAULT_INSTRUCTION: &str = "OCR this image to HTML. Each block is a div with \
     data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).";

/// Extensions [`expand_pages`] will pick up from a directory. Narrower than
/// "anything the `image` crate can decode": page directories routinely also hold
/// `.json` sidecars, `.txt` ground truth and `.DS_Store`, and trying to decode
/// those produces a per-file error storm instead of a run.
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif", "ppm", "tga"];

/// Substituted for the image span while the chat template is rendered, then
/// split on. It has to be a string that cannot appear in an instruction and
/// cannot tokenize into anything — NUL bytes are both.
const SPAN_MARKER: &str = "\u{0}\u{0}ojas-image-span\u{0}\u{0}";

// ---------------------------------------------------------------------------
// the encoder seam
// ---------------------------------------------------------------------------

/// What the command needs from a vision encoder.
///
/// Both halves of the contract are narrow:
///
/// * input is an already-preprocessed planar CHW f32 buffer plus its
///   `(width, height)`. The resize/normalize policy lives in
///   [`ojas_cpu::VitPreproc`] and is shared, so two encoders cannot disagree
///   about pixels.
/// * output is `n_tokens(w, h) * width()` f32 row-major, which is what
///   `Model::prefill_embeds` consumes. No embedding scale, no normalization; the
///   rows are used verbatim.
///
/// Identical input and output make the two implementors an oracle pair: the same
/// page through `OJAS_OCR_ENCODER=cpu` and `=model` must transcribe the same way,
/// and any divergence is the GPU tower's, because nothing else differs.
pub trait VisionEncoder {
    /// Shown in the banner so a log says which encoder produced the rows.
    fn name(&self) -> &'static str;
    /// Projector output width. Must equal the decoder's `hidden_dim`.
    fn width(&self) -> usize;
    /// Rows this image will produce — known before encoding, so the prompt can
    /// be built and the context checked before paying for the ViT.
    fn n_tokens(&self, width: usize, height: usize) -> usize;
    /// Post-merge token grid `(nx, ny)` = (columns, rows), with
    /// `nx * ny == n_tokens(width, height)`.
    ///
    /// Row order is part of the contract: row `i` is grid cell `(i / nx, i % nx)`,
    /// and the M-RoPE coordinates in [`span_positions`] depend on it. The 2x2
    /// spatial merge visits `(y_block, x_block, dy, dx)` with `dx` fastest
    /// (`ojas_cpu::cpu_vit::merge_permutation`, transcribed from qwen3vl.cpp), so
    /// four consecutive pre-merge patches form one merged token and the merged
    /// tokens are raster-ordered over the half-size grid.
    fn grid(&self, width: usize, height: usize) -> (usize, usize);
    /// `n_tokens(w, h) * width()` f32, row-major.
    fn encode(&self, img: &[f32], width: usize, height: usize) -> Result<Vec<f32>>;
}

/// Post-merge grid from the mmproj's own geometry. One definition, two encoders.
fn merged_grid(width: usize, height: usize, patch: usize, merge: usize) -> (usize, usize) {
    ((width / patch) / merge, (height / patch) / merge)
}

/// The decoder's own vision tower behind the seam — what a real run uses.
///
/// Borrows `&dyn Model` rather than owning anything: the tower is not a separate
/// object, it shares the decoder's device, weight arena, precision flag and
/// scratch buffers. `Model`'s three vision methods are the whole interface
/// (`ojas-core/src/lib.rs`).
pub struct ModelVitEncoder<'m> {
    m: &'m dyn Model,
    width: usize,
    patch: usize,
    merge: usize,
}

impl VisionEncoder for ModelVitEncoder<'_> {
    fn name(&self) -> &'static str {
        "model-vit"
    }
    fn width(&self) -> usize {
        self.width
    }
    fn n_tokens(&self, w: usize, h: usize) -> usize {
        // `open_encoder` only builds this after `vision_width()` answered Some, so
        // the tower exists; a None here is a geometry it cannot encode, and the
        // grid below is the fallback answer.
        self.m.vision_tokens(w, h).unwrap_or_else(|| {
            let (nx, ny) = self.grid(w, h);
            nx * ny
        })
    }
    fn grid(&self, w: usize, h: usize) -> (usize, usize) {
        merged_grid(w, h, self.patch, self.merge)
    }
    fn encode(&self, img: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
        // `open_encoder` proved the tower exists before constructing this, so None
        // cannot happen; reported rather than unwrapped to distinguish "no tower"
        // from "the tower failed".
        self.m.encode_image(img, w, h).context(
            "the decoder reported a vision tower and then declined to use it \
             (Model::encode_image returned None)",
        )?
    }
}

/// [`ojas_cpu::CpuVit`] behind the seam. Correct and deterministic, but minutes
/// per 16 384-patch page — an oracle rather than a production encoder. Also the
/// only encoder on a non-macOS build, and the only cross-check for the GPU tower
/// on the same page.
pub struct CpuVitEncoder {
    vit: ojas_cpu::CpuVit,
    width: usize,
}

impl VisionEncoder for CpuVitEncoder {
    fn name(&self) -> &'static str {
        "cpu-vit"
    }
    fn width(&self) -> usize {
        self.width
    }
    fn n_tokens(&self, w: usize, h: usize) -> usize {
        self.vit.n_merged_tokens(w, h)
    }
    fn grid(&self, w: usize, h: usize) -> (usize, usize) {
        merged_grid(w, h, self.vit.patch, self.vit.merge)
    }
    fn encode(&self, img: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
        self.vit.forward(img, w, h)
    }
}

/// Choose the encoder for `mmproj`. The only place a backend is selected.
///
/// Takes `&dyn Model` because the preferred encoder is the decoder's own tower,
/// reachable only once the model is loaded, so this must run inside `with_model`.
/// Selection order:
///
/// 1. the model's own tower, when `Model::vision_width()` says it has one;
/// 2. otherwise [`CpuVitEncoder`], loaded from the mmproj — the portable fallback,
///    and what `--device cpu` gets.
///
/// `OJAS_OCR_ENCODER=cpu|model` forces one. Running the same page through both is
/// the only end-to-end check available at page scale, where activation cosine
/// against llama.cpp is not usable. An env var rather than a CLI flag because
/// `flags.rs` is shared surface and this knob serves one command.
fn open_encoder<'m>(m: &'m dyn Model, mmproj: &Path) -> Result<Box<dyn VisionEncoder + 'm>> {
    let mut g = Gguf::open(mmproj.to_string_lossy().as_ref())
        .with_context(|| format!("opening mmproj {}", mmproj.display()))?;
    let width = g
        .meta_u32("clip.vision.projection_dim")
        .context("mmproj has no clip.vision.projection_dim")? as usize;
    let patch = g
        .meta_u32("clip.vision.patch_size")
        .context("mmproj has no clip.vision.patch_size")? as usize;
    // Absent = no spatial merge, so the post-merge grid is the raw patch grid.
    let merge = g.meta_u32("clip.vision.spatial_merge_size").unwrap_or(1).max(1) as usize;
    anyhow::ensure!(patch > 0, "mmproj declares patch_size 0");

    let forced = std::env::var("OJAS_OCR_ENCODER").unwrap_or_default();
    match forced.as_str() {
        "" | "auto" | "model" | "cpu" => {}
        other => bail!("OJAS_OCR_ENCODER={other:?} (want model, cpu or auto)"),
    }
    if forced != "cpu" {
        if let Some(w) = m.vision_width() {
            // The two widths come from different files' metadata; if they disagree
            // the projector and the decoder are from different checkpoints and every
            // row would be misaligned rather than merely wrong.
            anyhow::ensure!(
                w == width,
                "the decoder's tower writes {w}-wide rows but {} declares projection_dim {width}",
                mmproj.display()
            );
            return Ok(Box::new(ModelVitEncoder { m, width, patch, merge }));
        }
        if forced == "model" {
            bail!(
                "OJAS_OCR_ENCODER=model but this decoder has no vision tower \
                 (Model::vision_width returned None) — it is a CPU decoder, or it was \
                 loaded without an mmproj"
            );
        }
    }
    let vit = ojas_cpu::CpuVit::load(&mut g)
        .with_context(|| format!("loading the CPU ViT from {}", mmproj.display()))?;
    Ok(Box::new(CpuVitEncoder { vit, width }))
}

// ---------------------------------------------------------------------------
// page selection — pure where it can be
// ---------------------------------------------------------------------------

/// `*` (any run) and `?` (one character), on one path component. Iterative with
/// a backtrack point, so a pathological pattern cannot blow the stack.
pub(crate) fn glob_match(pat: &str, name: &str) -> bool {
    let (p, n): (Vec<char>, Vec<char>) = (pat.chars().collect(), name.chars().collect());
    let (mut pi, mut ni) = (0usize, 0usize);
    // Where to resume if the current `*` expansion turns out to be too short.
    let (mut star, mut resume) = (usize::MAX, 0usize);
    while ni < n.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == n[ni]) {
            pi += 1;
            ni += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = pi;
            resume = ni;
            pi += 1;
        } else if star != usize::MAX {
            resume += 1;
            ni = resume;
            pi = star + 1;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

/// Whether a file name looks like a page image. Extension only; the contents are
/// not sniffed.
pub(crate) fn is_image_name(name: &str) -> bool {
    match name.rsplit_once('.') {
        Some((_, ext)) => IMAGE_EXTS.contains(&ext.to_ascii_lowercase().as_str()),
        None => false,
    }
}

/// Filter to images and sort, so a 1 000-page run is reproducible and resumable
/// rather than following `read_dir`'s arbitrary order.
pub(crate) fn pick_images(mut names: Vec<String>) -> Vec<String> {
    names.retain(|n| is_image_name(n));
    names.sort();
    names
}

/// Resolve the positional into the pages to transcribe: one file, every image in
/// a directory, or the matches of a `*`/`?` pattern on the last component.
pub fn expand_pages(arg: &str) -> Result<Vec<PathBuf>> {
    let path = Path::new(arg);
    if path.is_dir() {
        let names: Vec<String> = std::fs::read_dir(path)
            .with_context(|| format!("reading directory {arg}"))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        let picked = pick_images(names);
        if picked.is_empty() {
            bail!("no page images in {arg} (looked for {})", IMAGE_EXTS.join(", "));
        }
        return Ok(picked.into_iter().map(|n| path.join(n)).collect());
    }
    let last = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    if last.contains('*') || last.contains('?') {
        let dir = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        };
        let names: Vec<String> = std::fs::read_dir(&dir)
            .with_context(|| format!("reading directory {}", dir.display()))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| glob_match(&last, n))
            .collect();
        // A pattern is an explicit selection: honour it even for extensions the
        // directory walk would skip, but still order it.
        let mut names = names;
        names.sort();
        if names.is_empty() {
            bail!("no files match {arg}");
        }
        return Ok(names.into_iter().map(|n| dir.join(n)).collect());
    }
    if !path.is_file() {
        bail!("no such page image: {arg}");
    }
    Ok(vec![path.to_path_buf()])
}

// ---------------------------------------------------------------------------
// prompt construction
// ---------------------------------------------------------------------------

/// The three vision marker token ids, resolved through the model's own vocabulary
/// rather than hardcoded. They are 9 / 11 / 10 in surya-2, but an id that is right
/// for one checkpoint is a silent transcription bug on the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VisionTokens {
    pub start: u32,
    pub pad: u32,
    pub end: u32,
}

impl VisionTokens {
    /// Each marker must be exactly one token. `Bpe::encode` is expensive per
    /// call, but these are 15-byte strings, so its special-token scan is
    /// negligible here.
    fn resolve(bpe: &Bpe) -> Result<VisionTokens> {
        let one = |s: &str| -> Result<u32> {
            match bpe.encode(s).as_slice() {
                [id] => Ok(*id as u32),
                other => bail!(
                    "this model's tokenizer has no single {s} token (it encodes to {} pieces) — \
                     it is probably not a vision model",
                    other.len()
                ),
            }
        };
        Ok(VisionTokens {
            start: one("<|vision_start|>")?,
            pad: one("<|image_pad|>")?,
            end: one("<|vision_end|>")?,
        })
    }
}

/// A prompt with one image span in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VisionPrompt {
    /// The full id sequence, image span included as placeholder ids.
    pub ids: Vec<u32>,
    /// Index of the first image row in `ids`.
    pub image_at: usize,
    /// How many rows the image occupies.
    pub n_image: usize,
}

impl VisionPrompt {
    fn span(&self) -> Range<usize> {
        self.image_at..self.image_at + self.n_image
    }
}

/// The chat scaffolding around an image turn, split at the image span.
///
/// Built by rendering [`ojas_tokenize::chat_template`] (verified byte-identical to
/// surya's Jinja for the text parts) over a user message carrying a marker where
/// the pads go, then splitting on the marker. Rendering and splicing keeps one
/// definition of the scaffolding; writing the ChatML by hand here would be a
/// second one free to drift.
///
/// Splitting is safe at this boundary: `head` ends with the complete
/// `<|vision_start|>` special and `tail` begins with `<|vision_end|>`, and
/// `Bpe::encode` splits on specials, so encoding the halves separately cannot
/// lose a merge that encoding the whole string would have found.
pub(crate) fn template_halves(arch: &str, instruction: &str) -> Result<(String, String)> {
    if instruction.contains(SPAN_MARKER) {
        bail!("the instruction contains the internal image-span marker");
    }
    let user = format!("<|vision_start|>{SPAN_MARKER}<|vision_end|>{instruction}");
    let full = ojas_tokenize::chat_template(arch, &user);
    let (head, tail) = full
        .split_once(SPAN_MARKER)
        .context("chat template dropped the image span marker")?;
    Ok((head.to_string(), tail.to_string()))
}

/// Splice `n` placeholder ids between two already-encoded text halves.
///
/// Defines the span's position: everything downstream (the prefill split, the row
/// offsets) derives from `image_at`/`n_image`.
pub(crate) fn splice_span(head: &[u32], pad: u32, n: usize, tail: &[u32]) -> VisionPrompt {
    let mut ids = Vec::with_capacity(head.len() + n + tail.len());
    ids.extend_from_slice(head);
    let image_at = ids.len();
    ids.resize(image_at + n, pad);
    ids.extend_from_slice(tail);
    VisionPrompt { ids, image_at, n_image: n }
}

/// The prompt's two text halves as ids, encoded once for a whole batch.
///
/// The halves depend only on the arch and the instruction, not the image, so a
/// 1 000-page run encodes them once. `Bpe::encode` scans ~1 948 specials per
/// iteration, and the head's ids are also what [`EmbedTable`] gathers rows for,
/// which must be decided before the first page.
pub(crate) struct Scaffold {
    pub head: Vec<u32>,
    pub tail: Vec<u32>,
    pub pad: u32,
}

impl Scaffold {
    fn splice(&self, n_image: usize) -> VisionPrompt {
        splice_span(&self.head, self.pad, n_image, &self.tail)
    }
}

fn encode_scaffold(
    bpe: &Bpe,
    arch: &str,
    instruction: &str,
    vtok: &VisionTokens,
) -> Result<Scaffold> {
    let (head, tail) = template_halves(arch, instruction)?;
    let enc = |s: &str| -> Vec<u32> { bpe.encode(s).into_iter().map(|v| v as u32).collect() };
    let (h, t) = (enc(&head), enc(&tail));
    // The halves were split around the markers, so they must still carry them.
    // If a template stops emitting them the span is unmarked and the model reads
    // the rows as ordinary text.
    if h.last() != Some(&vtok.start) {
        bail!("templated prompt does not end its prefix with <|vision_start|>");
    }
    if t.first() != Some(&vtok.end) {
        bail!("templated prompt does not begin its suffix with <|vision_end|>");
    }
    Ok(Scaffold { head: h, tail: t, pad: vtok.pad })
}

// ---------------------------------------------------------------------------
// M-RoPE positions
// ---------------------------------------------------------------------------

/// Which position space the prompt is laid out in.
///
/// Picked once per run by [`choose_layout`] from what the decoder can honour, and
/// overridable with `OJAS_OCR_POS` so the three are comparable on one page. Ordered
/// worst to best.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PosLayout {
    /// `pos3 = None`. Image rows take sequential text positions, so the page is
    /// 4 096 tokens in a line. Pre-M-RoPE behaviour.
    Sequential,
    /// 2-D image coordinates anchored at the span's own first cache row. The image
    /// is a grid and its distance to the text before it is right, but the text
    /// after it — and every generated token — sits `nx*ny - max(nx,ny)` positions
    /// too far away.
    Grid,
    /// [`PosLayout::Grid`] shifted so the text after the image lands where
    /// llama.cpp puts it. Requires the ChatML head to be injected too.
    Anchored,
}

/// Positions for one image span plus the shift its preceding text needs.
pub(crate) struct SpanPositions {
    /// `(t,h,w,e)` per image row, aligned to the span's cache rows. `None` =
    /// [`PosLayout::Sequential`].
    pub img: Option<Vec<[u32; 4]>>,
    /// Added to the sequence position of every text row before the span. Zero
    /// except under [`PosLayout::Anchored`].
    pub shift: u32,
}

/// Lay out one image span's M-RoPE coordinates, and say where the text before it
/// has to go.
///
/// `Model::prefill_embeds` keeps the two counters separate: `base_pos + i` is the
/// KV cache row and stays contiguous, `pos3` moves only the rope angle.
/// `Model::prefill` and `Model::forward_id` each have one scalar that is both, so
/// the text after the image — and every generated token — ropes at its cache row:
/// the decode loop calls `forward_id(cur, pos)` with `pos` counting cache rows
/// (`ojas-infer/src/lib.rs`, `pos = prompt_ids.len() - 1` then `pos += 1`), and
/// `rope_qk_store` has no coordinate argument.
///
/// On a 24x24 post-merge grid llama.cpp puts the instruction at `pos_0 + 24` where
/// the cache row puts it at `pos_0 + 576`: every image-to-text rope distance is off
/// by 552 in all three sections. With `rope.freq_base` 1e7 and `n_rot` 64 the fast
/// channels (theta = 1) wrap ~88 full turns over that gap, so those pairs become
/// noise rather than merely stretched.
///
/// Accepting the offset is not an option: each of several hundred generated tokens
/// attends to all `nx*ny` image rows, so image-to-generated carries most of the run's
/// attention mass, which is where the 552 lands. Nor is routing the tail through
/// `prefill_embeds` with correct `pos3` and host-side `token_embd` rows (as
/// `ojas-models/examples/embed_inject_gate.rs` demonstrates) — the tail would sit at
/// llama.cpp's positions while the generated tokens after it still run at cache rows,
/// inserting the whole 552 as a discontinuity at the generation boundary and breaking
/// local text-to-text distances where they are strongest. Making that coherent needs
/// a coordinate argument on `forward_id`, i.e. a kernel change.
///
/// RoPE scores depend only on position differences, so absolute numbering is free
/// and the decode loop has already fixed one end of it. This anchors everything on
/// that end, with `delta = nx*ny - max(nx, ny)`:
///
/// ```text
/// head text row p  ->  p + delta                    (injected, pos3 = (q,q,q,0))
/// image row i      ->  (t0, t0 + i/nx, t0 + i%nx, 0),  t0 = pos_0 + delta
/// tail text row k  ->  pos_0 + nx*ny + k            (its cache row, for free)
/// generated row k  ->  its cache row                (for free)
/// ```
///
/// The image's largest coordinate is `t0 + max(nx,ny) - 1 == pos_0 + nx*ny - 1`, the
/// span's last cache row, so the next text position is `t0 + max(nx,ny)` —
/// llama.cpp's rule (`mtmd_image_tokens_get_n_pos`) without the decode path knowing
/// anything. The result is llama.cpp's layout plus a uniform `+delta`, which rope
/// cannot see; nothing is approximated.
///
/// The price is a shifted head, which `prefill` cannot express, so the head's seven
/// rows are injected (see [`EmbedTable`]). Failing that, [`PosLayout::Grid`] is the
/// fallback and the deviation is printed with its measured magnitude.
pub(crate) fn span_positions(pos_0: usize, nx: usize, ny: usize, layout: PosLayout) -> SpanPositions {
    let n = nx * ny;
    if layout == PosLayout::Sequential || n == 0 {
        return SpanPositions { img: None, shift: 0 };
    }
    // The sequence positions an image consumes: max(nx,ny), not nx*ny.
    let n_pos = nx.max(ny);
    let shift = if layout == PosLayout::Anchored { (n - n_pos) as u32 } else { 0 };
    let t0 = pos_0 as u32 + shift;
    let (nxu, mut img) = (nx as u32, Vec::with_capacity(n));
    for i in 0..n as u32 {
        // Fourth stream is zero: it carries the image index in llama.cpp's XD-RoPE
        // (`pos.z = image_idx`), and one page is one image.
        img.push([t0, t0 + i / nxu, t0 + i % nxu, 0]);
    }
    SpanPositions { img: Some(img), shift }
}

/// Resolve the layout from what the decoder honours and what the caller asked
/// for. `OJAS_OCR_POS=seq|grid|anchor` pins it, which is how the three are
/// compared on one page.
pub(crate) fn choose_layout(mrope_ok: bool, head_rows_ok: bool) -> Result<PosLayout> {
    layout_from(&std::env::var("OJAS_OCR_POS").unwrap_or_default(), mrope_ok, head_rows_ok)
}

/// [`choose_layout`] without the environment, so the ladder is testable. What the
/// decoder cannot do wins over what the caller asked for, except `seq`, which is
/// honoured even on a decoder that could do better.
pub(crate) fn layout_from(want: &str, mrope_ok: bool, head_rows_ok: bool) -> Result<PosLayout> {
    let best = match want {
        "seq" | "sequential" => return Ok(PosLayout::Sequential),
        "grid" => PosLayout::Grid,
        "" | "auto" | "anchor" | "anchored" => PosLayout::Anchored,
        other => bail!("OJAS_OCR_POS={other:?} (want seq, grid, anchor or auto)"),
    };
    if !mrope_ok {
        return Ok(PosLayout::Sequential);
    }
    Ok(if best == PosLayout::Anchored && head_rows_ok { PosLayout::Anchored } else { PosLayout::Grid })
}

/// Host-side `token_embd.weight` rows.
///
/// Needed only for the ~7-token ChatML head, which [`PosLayout::Anchored`] must
/// place at shifted positions and `Model::prefill` cannot. `prefill_embeds` can,
/// but it wants rows rather than ids, so they are read from the GGUF the way
/// `embed_inject_gate.rs` does.
///
/// The table is read once and dropped as soon as the head is gathered: ~134 MB on
/// surya-2, of which the head needs 28 KB. These are the file's own values, so they
/// match what the decoder would gather only at a precision that does not requantize
/// `token_embd` (prec 4, the `ocr` default, keeps an F16 file's table in F16). At a
/// requantizing precision the head's seven rows are the pre-quantization values —
/// a ~1e-3 relative perturbation on seven tokens, far smaller than a 552-position
/// rope error.
struct EmbedTable {
    d: usize,
    ty: u32,
    bytes: Vec<u8>,
}

/// IEEE binary16 -> binary32, exact (every f16 is representable in f32).
///
/// Written out rather than pulled from `half`: `ojas-cli`'s dependency list is what
/// keeps the Linux build alive (see `Cargo.toml`'s macOS block), and the one caller
/// reads a handful of embedding rows.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let man = (bits & 0x3ff) as u32;
    f32::from_bits(match (exp, man) {
        (0, 0) => sign,                                     // +-0
        (0, _) => {
            // Subnormal: f16 value is man * 2^-24, which f32 represents as a
            // normal number, so it must be renormalized rather than shifted.
            let h = 31 - man.leading_zeros();               // top set bit, 0..=9
            sign | ((h + 103) << 23) | ((man - (1 << h)) << (23 - h))
        }
        (0x1f, _) => sign | 0x7f80_0000 | (man << 13),      // inf / NaN
        _ => sign | ((exp + 112) << 23) | (man << 13),      // 127 - 15 = 112
    })
}

impl EmbedTable {
    fn open(model: &str) -> Result<EmbedTable> {
        let mut g = Gguf::open(model).with_context(|| format!("reopening {model}"))?;
        let (dims, ty, bytes) = g
            .read_tensor("token_embd.weight")
            .context("reading token_embd.weight")?;
        anyhow::ensure!(dims.len() == 2, "token_embd.weight is not 2-D: {dims:?}");
        anyhow::ensure!(ty == 0 || ty == 1, "token_embd.weight decoded to type {ty}, want F16 or F32");
        Ok(EmbedTable { d: dims[0] as usize, ty, bytes })
    }

    /// `ids.len() * d` f32 row-major — `prefill_embeds`' input.
    fn gather(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let w = if self.ty == 1 { 2 } else { 4 };
        let mut out = Vec::with_capacity(ids.len() * self.d);
        for &t in ids {
            let o = t as usize * self.d * w;
            let end = o + self.d * w;
            anyhow::ensure!(end <= self.bytes.len(), "token {t} is outside token_embd.weight");
            let src = &self.bytes[o..end];
            if self.ty == 1 {
                out.extend(src.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))));
            } else {
                out.extend(src.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
            }
        }
        Ok(out)
    }
}

pub(crate) fn layout_name(l: PosLayout) -> &'static str {
    match l {
        PosLayout::Sequential => "sequential (1-D)",
        PosLayout::Grid => "grid (2-D, unanchored)",
        PosLayout::Anchored => "grid (2-D, anchored)",
    }
}

/// Describes how `l` deviates from the anchored layout, with the actual grid
/// numbers.
pub(crate) fn layout_caveat(l: PosLayout, mrope_ok: bool, head_rows_failed: bool) -> String {
    // `describe_positions` carries the exact per-page deviation, so this only names
    // the problem and its cause.
    match l {
        PosLayout::Anchored => String::new(),
        PosLayout::Grid => format!(
            "DEVIATION: the image is a 2-D grid but anchored at its own first cache row, so \
             the instruction and every generated token sit `nx*ny - max(nx,ny)` rope \
             positions further from the image than llama.cpp puts them — see the per-page \
             `delta` below for the size on this page. Cause: {}.",
            if head_rows_failed {
                "no host-side token_embd rows, so the ChatML head cannot be shifted"
            } else {
                "OJAS_OCR_POS=grid was requested"
            }
        ),
        PosLayout::Sequential => format!(
            "DEVIATION: image rows take SEQUENTIAL text positions, so the page is a line \
             rather than a grid — image geometry is lost AND image-to-text distances are \
             `nx*ny - max(nx,ny)` too large. This is the pre-M-RoPE behaviour; measured \
             67.3% -> 21.8% ground-truth word recall against the anchored layout on a \
             synthetic page. Cause: {}.",
            if mrope_ok { "OJAS_OCR_POS=seq was requested" } else { "the decoder declined pos3" }
        ),
    }
}

/// One log line showing the coordinates actually in use.
fn describe_positions(
    sp: &SpanPositions, prompt: &VisionPrompt, nx: usize, ny: usize, layout: PosLayout,
) -> String {
    // The gap between where the text after the image actually ropes (its cache row)
    // and where llama.cpp would put it (image_t + max(nx,ny)). Printed either way,
    // including the zero the anchored layout produces.
    // Saturating because a degenerate grid (a dimension under patch*merge, so nx or
    // ny is 0) makes `max(nx,ny)` exceed `nx*ny`; an underflow would panic in debug
    // and print a 19-digit number in release.
    let off = (nx * ny).saturating_sub(nx.max(ny)).saturating_sub(sp.shift as usize);
    let Some(p3) = &sp.img else {
        return format!(
            "positions: {} | image rows {}..{} take their cache-row positions | \
             image-to-text distances {off} too large",
            layout_name(layout), prompt.image_at, prompt.image_at + prompt.n_image
        );
    };
    let show = |i: usize| -> String {
        let c = p3[i.min(p3.len() - 1)];
        format!("[{}]=({},{},{},{})", i, c[0], c[1], c[2], c[3])
    };
    // Row 1 must advance `w`; row `nx` must advance `h` and reset `w`. Both are
    // printed so the log shows a grid rather than merely a non-empty pos3.
    format!(
        "positions: {} | delta {} | head {}..{} -> {}..{} | image {} {} {} | after image {} \
         (llama.cpp: this image consumes {} positions, not {}; image-to-text error {off})",
        layout_name(layout),
        sp.shift,
        0, prompt.image_at,
        sp.shift, sp.shift + prompt.image_at as u32,
        show(0), show(1), show(nx),
        prompt.image_at + prompt.n_image,
        nx.max(ny), nx * ny,
    )
}

/// Sequential `(p,p,p,0)` coordinates for `n` text rows starting at sequence
/// position `from`.
///
/// The fourth stream is zero, not `p`: llama.cpp's text convention
/// (`llm_graph_input_pos::set_input`, "the 3 first dims are the same, and 4th dim
/// is all 0"). With `t == h == w` every section reads the same number, so a
/// sectioned kernel reproduces plain rope bit-for-bit.
pub(crate) fn text_positions(from: u32, n: usize) -> Vec<[u32; 4]> {
    (0..n as u32).map(|i| { let p = from + i; [p, p, p, 0] }).collect()
}

// ---------------------------------------------------------------------------
// the injection wrapper
// ---------------------------------------------------------------------------

/// Split an absolute prefill range against the image span.
///
/// `EngineCore` prefills in 256-token chunks with an absolute `base_pos`, so a
/// chunk can be all text, all image, or straddle either edge, and a 4 096-row span
/// straddles both. Returns the three absolute sub-ranges in order; any may be
/// empty.
pub(crate) fn split_prefill(
    base: usize,
    len: usize,
    span: Range<usize>,
) -> (Range<usize>, Range<usize>, Range<usize>) {
    let end = base + len;
    // Three consecutive cuts: text up to the span, the span, text after it.
    // Each boundary is clamped into [base, end], so the three ranges always
    // partition the chunk even when the span lies entirely outside it.
    let head_end = end.min(span.start).max(base);
    let img_end = end.min(span.end).max(head_end);
    (base..head_end, head_end..img_end, img_end..end)
}

/// A `Model` that is the wrapped model in every respect except that prefill
/// calls landing inside the image span are served from precomputed rows.
///
/// Lets `EngineCore` and [`crate::cmds::stream`] be reused unchanged. Prefilling by
/// hand and driving the decode loop here would fork sampling, speculative decoding,
/// EOG handling, TTFT accounting and the detokenizer out of the one place they are
/// implemented.
struct VisionPrefill<'m> {
    inner: &'m dyn Model,
    /// Absolute cache rows the image occupies.
    span: Range<usize>,
    /// `span.len() * d` f32, row-major.
    rows: Vec<f32>,
    d: usize,
    /// `(t,h,w,e)` per image row, indexed by `row - span.start`. `None` =
    /// [`PosLayout::Sequential`] — the rows take their cache-row positions.
    img_pos3: Option<Vec<[u32; 4]>>,
    /// The ChatML head — cache rows `0..span.start` — as injected rows plus their
    /// shifted coordinates, indexed by absolute cache row. `Some` only under
    /// [`PosLayout::Anchored`], the one layout that moves the text before the
    /// image; otherwise the head is an ordinary id prefill.
    head: Option<(Vec<f32>, Vec<[u32; 4]>)>,
    /// Set if the decoder refuses an injection. [`probe_injection`] checks up front,
    /// so this should be unreachable; if it fires, the page is garbage and the
    /// caller must report that rather than print it.
    refused: std::cell::Cell<bool>,
}

impl Model for VisionPrefill<'_> {
    fn prefill(&self, tokens: &[u32], base_pos: usize) {
        let (head, img, tail) = split_prefill(base_pos, tokens.len(), self.span.clone());
        let at = |r: &Range<usize>| &tokens[r.start - base_pos..r.end - base_pos];
        if !head.is_empty() {
            // `head` is always a sub-range of `0..span.start`, so the absolute
            // cache row indexes the head arrays directly.
            match &self.head {
                Some((rows, pos3)) => {
                    let (lo, hi) = (head.start * self.d, head.end * self.d);
                    if !self.inner.prefill_embeds(
                        at(&head), &rows[lo..hi], head.start, Some(&pos3[head.clone()]),
                    ) {
                        self.refused.set(true);
                    }
                }
                None => self.inner.prefill(at(&head), head.start),
            }
        }
        if !img.is_empty() {
            let lo = (img.start - self.span.start) * self.d;
            let hi = (img.end - self.span.start) * self.d;
            // Slice the coordinates the same way as the rows — one per row, not one
            // per `d` floats — and by span-relative index, because
            // `forward_chunk_enc_embed` reads them by `m` and not by position.
            let p3 = self.img_pos3.as_ref()
                .map(|p| &p[img.start - self.span.start..img.end - self.span.start]);
            if !self.inner.prefill_embeds(at(&img), &self.rows[lo..hi], img.start, p3) {
                self.refused.set(true);
            }
        }
        if !tail.is_empty() {
            // The id path with a scalar position, deliberately. Under
            // `PosLayout::Anchored` the cache row is the sequence position llama.cpp
            // assigns here, and it must be: the decode loop continuing from this
            // point has no coordinate argument, so anything else would put a
            // discontinuity at the generation boundary. See [`span_positions`].
            self.inner.prefill(at(&tail), tail.start);
        }
    }

    /// Always 0. Reuse is a longest-common-prefix over raw ids and every image token
    /// is the same placeholder, so a match here would serve page N-1's KV cache for
    /// page N. `prefill_embeds` already disowns its span; answering 0 unconditionally
    /// also keeps a recurrent model from being asked, since there the query restores
    /// a snapshot as a side effect.
    fn reuse_prefix_len(&self, _full_prompt: &[u32]) -> usize {
        0
    }

    // ---- pure forwarding below -------------------------------------------
    fn context_capacity(&self) -> usize { self.inner.context_capacity() }
    fn mtp_verify_width(&self) -> usize { self.inner.mtp_verify_width() }
    fn n_layers(&self) -> usize { self.inner.n_layers() }
    fn hidden_dim(&self) -> usize { self.inner.hidden_dim() }
    fn prefill_embeds(&self, t: &[u32], x: &[f32], base_pos: usize, pos3: Option<&[[u32; 4]]>) -> bool {
        self.inner.prefill_embeds(t, x, base_pos, pos3)
    }
    fn vision_width(&self) -> Option<usize> { self.inner.vision_width() }
    fn vision_tokens(&self, w: usize, h: usize) -> Option<usize> { self.inner.vision_tokens(w, h) }
    fn encode_image(&self, img: &[f32], w: usize, h: usize) -> Option<Result<Vec<f32>>> {
        self.inner.encode_image(img, w, h)
    }
    fn forward_id(&self, token: u32, pos: usize) -> u32 { self.inner.forward_id(token, pos) }
    fn forward_logits(&self, token: u32, pos: usize) -> Option<Vec<f32>> {
        self.inner.forward_logits(token, pos)
    }
    fn forward_batch_logits(&self, t: &[u32], base_pos: usize) -> Option<Vec<Vec<f32>>> {
        self.inner.forward_batch_logits(t, base_pos)
    }
    fn forward_batch_ids(&self, t: &[u32], base_pos: usize) -> Option<Vec<u32>> {
        self.inner.forward_batch_ids(t, base_pos)
    }
    fn forward_batch_topk(&self, t: &[u32], base_pos: usize) -> Option<(Vec<u32>, Vec<u32>)> {
        self.inner.forward_batch_topk(t, base_pos)
    }
    fn has_mtp(&self) -> bool { self.inner.has_mtp() }
    fn mtp_step_committed(&self, cur: u32, pos: usize) -> Option<Vec<u32>> {
        self.inner.mtp_step_committed(cur, pos)
    }
    fn forward_id_topk(&self, token: u32, pos: usize) -> Option<(u32, [u32; 8])> {
        self.inner.forward_id_topk(token, pos)
    }
    fn reset_session(&self) { self.inner.reset_session() }
}

/// Does this decoder have an embedding-injection path, and does it honour M-RoPE
/// coordinates? Returns `(rows, pos3)`.
///
/// `Model::prefill_embeds` answers per call, which is too late: by the time it says
/// no, the prompt is half prefilled and the output is nonsense. This asks once, on a
/// single throwaway row, before any page is encoded.
///
/// The `pos3` probe uses the degenerate coordinate `[0,0,0,0]` — the text convention
/// `(p,p,p,0)` at `p = 0` — so a decoder that honours it runs bit-identically to
/// plain rope and the probe cannot perturb anything even before the reset.
/// `prefill_embeds` is documented to return false for a `pos3` it cannot honour
/// rather than dropping the coordinates, which is what makes this answerable.
fn probe_injection(m: &dyn Model, d: usize) -> (bool, bool) {
    let x = vec![0.0f32; d];
    m.reset_session();
    let rows = m.prefill_embeds(&[0u32], &x, 0, None);
    m.reset_session();
    let pos3 = rows && m.prefill_embeds(&[0u32], &x, 0, Some(&[[0, 0, 0, 0]]));
    m.reset_session();
    (rows, pos3)
}

// ---------------------------------------------------------------------------
// repetition guard
// ---------------------------------------------------------------------------

/// Longest block period considered by [`loop_period`].
///
/// 128 rather than a handful because of this vocabulary: surya-2's
/// `tokenizer.ggml.merges` has exactly one entry, so BPE never merges and text
/// encodes to roughly one token per byte. A repeated `<div data-bbox=...>...` block,
/// the shape a looping page takes, is 60-80 tokens here where it would be a dozen on
/// a merged vocabulary — and a ceiling tuned for one would never see it.
pub(crate) const LOOP_MAX_PERIOD: usize = 128;
/// Minimum number of repeated tokens before a loop is called, so a short period
/// needs many more repeats than a long one: `</div>` twice is HTML, the same token
/// twenty-four times is not.
pub(crate) const LOOP_MIN_RUN: usize = 24;

/// Detect a decode that has fallen into an n-gram loop, returning the period.
///
/// Fires when the tail of the output is `reps` back-to-back copies of one
/// `p`-token block, with `reps = max(4, ceil(min_run / p))`. The `max(4)` floor
/// keeps a long block from tripping on two repeats (tables and lists legitimately
/// repeat structure); the `min_run` term keeps a short block from tripping on a
/// handful (`p = 1` needs 24 identical tokens in a row).
///
/// Surya takes the same approach, abandoning full-page mode for layout-guided block
/// recognition when the output "devolves into a repetition loop"
/// (`surya/recognition/__init__.py:167`). Without a guard, one bad page in a thousand
/// burns the whole `-n` budget: a 271 s outlier against a 32 s median in the server
/// log.
///
/// Soft loops are not caught — the model re-emitting the same structure with
/// different numbers, e.g. `<div data-bbox="47 654 ...">Total</div>` then
/// `<div data-bbox="47 759 ...">`, which this model does. Catching that needs the
/// comparison to ignore digits, which makes a table of 40 numeric cells
/// indistinguishable from a loop. Killing a good page is worse than finishing a bad
/// one, so the test stays exact.
pub(crate) fn loop_period(gen: &[u32], max_period: usize, min_run: usize) -> Option<usize> {
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

// ---------------------------------------------------------------------------
// the command
// ---------------------------------------------------------------------------

/// Merged-token budget for the preprocessor (`--image-max-tokens` upstream). An env
/// var rather than a CLI flag because `flags.rs` is shared surface and only OCR needs
/// this knob. It is also the second half of the "prompt does not fit" remedy: halving
/// the budget quarters the ViT's attention cost as well as shortening the prompt.
fn token_budget() -> (i32, i32) {
    let get = |k: &str, d: i32| -> i32 {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(d)
    };
    (get("OJAS_IMAGE_MIN_TOKENS", 8), get("OJAS_IMAGE_MAX_TOKENS", 4096))
}

fn instruction_for(opts: &RunOpts) -> Result<String> {
    if let Some(f) = &opts.prompt_file {
        return std::fs::read_to_string(f).with_context(|| format!("reading prompt file {f}"));
    }
    Ok(opts.prompt.clone().unwrap_or_else(|| DEFAULT_INSTRUCTION.to_string()))
}

/// Find and validate the mmproj before anything is loaded. Discovery mirrors the MTP
/// sidecar rules (one file beside the model, ambiguity refused).
/// [`ojas_formats::mmproj::validate`] catches a projector from a different
/// checkpoint, which would otherwise surface as fluent wrong text.
fn resolve_mmproj(model: &str) -> Result<PathBuf> {
    let explicit = ojas_core::config::EngineConfig::current().mmproj;
    let found = ojas_formats::mmproj::discover(Path::new(model), explicit.as_deref())?.context(
        "no vision projector: put a *mmproj*.gguf next to the model, or pass \
         --mmproj /path/to/mmproj.gguf (OJAS_MMPROJ also works)",
    )?;
    let base = Gguf::open(model).with_context(|| format!("opening {model}"))?;
    let mm = Gguf::open(found.to_string_lossy().as_ref())
        .with_context(|| format!("opening mmproj {}", found.display()))?;
    ojas_formats::mmproj::validate(&base, &mm)
        .with_context(|| format!("{} is not a usable projector for {model}", found.display()))?;
    Ok(found)
}

// ---------------------------------------------------------------------------
// per-page preparation — one definition, two drivers
// ---------------------------------------------------------------------------

/// Everything about a page that does not depend on where it decodes: the prompt, the
/// encoder's rows, and the span's coordinates.
///
/// Shared so the sequential and batched paths cannot drift. A second copy of "decode
/// the scan, check the grid against the row count, splice the span, lay out the
/// coordinates" is a second place for the transpose bug and the context check to go
/// wrong.
struct Prepared {
    prompt: VisionPrompt,
    /// `n_image * hidden_dim` f32, row-major — the encoder's own output.
    rows: Vec<f32>,
    sp: SpanPositions,
}

/// What page preparation needs that is the same for every page in a run.
struct PageCtx<'a> {
    enc: &'a dyn VisionEncoder,
    pre: &'a VitPreproc,
    scaffold: &'a Scaffold,
    layout: PosLayout,
    info: &'a ModelInfo,
    opts: &'a RunOpts,
    max_tok: i32,
    pages: &'a [PathBuf],
    /// The ChatML head's residual rows, when [`PosLayout::Anchored`] is live.
    head_rows: Option<&'a [f32]>,
    /// The banner is printed once, from whichever page gets there first.
    banner_shown: std::cell::Cell<bool>,
}

impl PageCtx<'_> {
    /// Read, resize, encode and prompt-build page `i`. `Ok(None)` means the scan
    /// could not be read and the page is skipped, which only happens in a batch; a
    /// single-page run reports the error.
    ///
    /// Shared with the sequential loop, log lines included, so a batched run leaves
    /// the same evidence.
    fn prepare(&self, i: usize) -> Result<Option<Prepared>> {
        let page = &self.pages[i];
        let (enc, info, opts) = (self.enc, self.info, self.opts);
        let te = std::time::Instant::now();
        // Reading and resizing is the one stage whose failure is about the input
        // rather than the engine: an unreadable scan, an unknown format, a
        // zero-pixel image. A thousand-page batch must not die on one of those, so
        // it is skipped and counted. Everything after this point is fatal, since a
        // failure there means the engine is wrong.
        let loaded = (|| -> Result<(usize, usize, usize, usize, Vec<f32>)> {
            let (w0, h0, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(page)
                .with_context(|| format!("decoding {}", page.display()))?;
            let (w, h, planar) = self.pre.preprocess(&rgb, w0, h0)?;
            Ok((w0, h0, w, h, planar))
        })();
        let (w0, h0, w, h, planar) = match loaded {
            Ok(v) => v,
            Err(e) if self.pages.len() > 1 => {
                eprintln!("\n  [{}/{}] {}: SKIPPED — {e:#}", i + 1, self.pages.len(), page.display());
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        let n_image = enc.n_tokens(w, h);
        let (nx, ny) = enc.grid(w, h);
        // The grid and the row count come from the same geometry by different
        // arithmetic, and everything positional below indexes rows by
        // `(i / nx, i % nx)`. A disagreement means the M-RoPE coordinates are a
        // transpose or a wrap of the real layout, which yields plausible output for
        // the wrong page — so it is an error, not a warning.
        if nx * ny != n_image {
            bail!(
                "{} reports {n_image} rows for {w}x{h} but a {nx}x{ny} post-merge grid \
                 ({} rows) — the row order and the M-RoPE coordinates would disagree",
                enc.name(),
                nx * ny
            );
        }
        let prompt = self.scaffold.splice(n_image);

        if prompt.ids.len() >= info.context {
            bail!(
                "page prompt is {} tokens ({n_image} image + {} text) but the context holds {} \
                 — raise it with -c {}, or shrink the image with OJAS_IMAGE_MAX_TOKENS={}",
                prompt.ids.len(),
                prompt.ids.len() - n_image,
                info.context,
                (prompt.ids.len() + opts.n_predict).next_power_of_two(),
                (self.max_tok / 2).max(8),
            );
        }
        check_fits(prompt.ids.len(), opts.n_predict, info)?;
        if !self.banner_shown.replace(true) {
            banner(info, prompt.ids.len());
        }

        // Before the encode, not after: the CPU ViT takes a minute at 768 image
        // tokens and several at a full page, and a silent terminal reads as a hang.
        let sp = span_positions(prompt.image_at, nx, ny, self.layout);
        eprintln!(
            "\n  [{}/{}] {} | {w0}x{h0} -> {w}x{h} | {n_image} image tokens ({nx}x{ny} grid) | {} prompt tokens",
            i + 1,
            self.pages.len(),
            page.display(),
            prompt.ids.len(),
        );
        // Printed every page, not behind a flag: the coordinates are the one part of
        // the pipeline with no oracle at page scale. Rows are checkable against the
        // CPU ViT, but a wrong position space produces fluent wrong text and nothing
        // else in the output says so. The first descriptors show the grid in the log
        // — `w` advancing at row 1 and `h` at row `nx`.
        eprintln!("  {}", describe_positions(&sp, &prompt, nx, ny, self.layout));
        let rows = enc.encode(&planar, w, h)?;
        if rows.len() != n_image * info.hidden_dim {
            bail!(
                "encoder returned {} f32 for {n_image} rows of {} — the seam's contract is \
                 n_tokens * hidden_dim, row-major",
                rows.len(),
                info.hidden_dim
            );
        }
        eprintln!("  encoded in {:.1}s ({})", te.elapsed().as_secs_f64(), enc.name());
        Ok(Some(Prepared { prompt, rows, sp }))
    }

    /// The ChatML head as injected rows plus its shifted coordinates, for the one
    /// layout that moves it. `None` otherwise: the other two layouts prefill the head
    /// by id at its cache rows.
    fn head_for(&self, p: &Prepared) -> Option<(Vec<f32>, Vec<[u32; 4]>)> {
        self.head_rows
            .filter(|_| self.layout == PosLayout::Anchored)
            .map(|r| (r.to_vec(), text_positions(p.sp.shift, p.prompt.image_at)))
    }

    /// The page separator a multi-page run writes to stdout. Single-page output stays
    /// exactly what `run` produces, with nothing prepended.
    fn page_marker(&self, i: usize) {
        if self.pages.len() > 1 {
            use std::io::Write;
            let mut o = std::io::stdout();
            let _ = writeln!(
                o, "<!-- ojas-ocr page {}/{}: {} -->", i + 1, self.pages.len(), self.pages[i].display()
            );
            let _ = o.flush();
        }
    }
}

// ---------------------------------------------------------------------------
// the batched path
// ---------------------------------------------------------------------------

/// The decoder's slot methods, reached through one adapter.
///
/// `crate::sched` is written against its own three-method [`crate::sched::SlotModel`]
/// rather than against `Model`, so the scheduler can be tested against a software mock
/// with no GPU in the process. This is the other implementor: the real decoder,
/// forwarded verbatim. All five slot methods land here and nowhere else in the file.
mod slots {
    use ojas_core::Model;

    /// A `&dyn Model` viewed as a bank of decode slots.
    pub(super) struct ModelSlots<'m> {
        pub m: &'m dyn Model,
    }

    impl ModelSlots<'_> {
        /// Prefill ids into one slot. `prefill_slot(0, ..)` is `prefill`, so slot 0
        /// takes the same path as the sequential driver.
        pub fn prefill_ids(&self, s: usize, t: &[u32], base_pos: usize) -> bool {
            self.m.prefill_slot(s, t, base_pos)
        }
        /// Inject rows into one slot — how a page's ViT output enters a batch.
        pub fn prefill_rows(&self, s: usize, t: &[u32], x: &[f32], base_pos: usize,
                            pos3: Option<&[[u32; 4]]>) -> bool {
            self.m.prefill_embeds_slot(s, t, x, base_pos, pos3)
        }
    }

    impl crate::sched::SlotModel for ModelSlots<'_> {
        /// 1 on every decoder that has not implemented slots (the trait's own
        /// default), and on the Metal decoder unless `OJAS_SLOTS>1` allocated the
        /// per-slot KV and recurrent state at load time. 1 selects the sequential
        /// path, so an unbatched decoder needs no special case.
        fn max_slots(&self) -> usize { self.m.max_slots() }
        fn reset_slot(&self, s: usize) { self.m.reset_slot(s) }
        fn decode_slots(&self, steps: &[(usize, u32, usize)]) -> Option<Vec<u32>> {
            self.m.decode_slots(steps)
        }
    }
}

/// Slots to use when the decoder offers more than one, before `OJAS_OCR_SLOTS`.
///
/// Two, not four: B=4 only beats B=2 via the pair-tiled Q4 path, which is not
/// reproducible (see `graph_decode::q4_pair`). B=2 is the fastest configuration that
/// decodes the same page the same way twice — 8.47 s/page against 12.53 at B=1 on
/// four real 300-dpi pages. Raise it with `OJAS_OCR_SLOTS` once a 4-row plain-Q4
/// kernel lands.
///
/// 4 is the hard ceiling: sweeping M = 1..8 on this model's real shapes under three
/// kernel routings, all three peak at M=4 — Q4L regresses at M=6 and M=8
/// (1.84x -> 1.52x), and F16 gains only 7% from M=4 to M=8 for twice the per-sequence
/// state. Past 2 the per-sequence recurrent state (20.2 MB, read and written every
/// token) falls out of the system-level cache: 0.040 ms/token/sequence at N=2 against
/// 0.105 at N=4. Over five interleaved rounds on four 300-dpi pages (prec 2,
/// `OJAS_IMAGE_MAX_TOKENS=2048`), `OJAS_OCR_SLOTS=2` beat `=4` in every round — best
/// wall 33.9 s against 40.2 s for the same pages, against 50.1 s sequential.
pub(crate) const DEFAULT_SLOTS: usize = 2;

/// How many slots this run should use, where three separate ceilings meet. Pure so
/// the policy is testable.
pub(crate) fn batch_slots(max_slots: usize, pages: usize, want: Option<usize>) -> usize {
    // More slots than pages is not merely wasteful: an empty slot still costs a row
    // in every batched dispatch.
    max_slots.min(pages).min(want.unwrap_or(DEFAULT_SLOTS)).max(1)
}

/// `OJAS_OCR_SLOTS=N`. An env var rather than a CLI flag because `flags.rs` is shared
/// surface and this knob serves one command. `0` or unparseable means unset.
fn slots_env() -> Option<usize> {
    std::env::var("OJAS_OCR_SLOTS").ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0)
}

/// Prefill one prepared page into `slot`.
///
/// Two details keep this byte-identical to the sequential path:
///
/// * the prompt is chunked at 256, which is `EngineCore`'s `PREFILL_STEP` and the
///   model's own `MAXM`, so the decoder sees the same dispatch shapes; a different
///   chunking is a different reduction order and a different last bit;
/// * each chunk is split against the image span by [`split_prefill`], the same
///   function [`VisionPrefill`] uses, and the three parts are routed the same way
///   (injected head under [`PosLayout::Anchored`], injected image rows, id tail).
///
/// The final prompt id is not prefilled; it drives the slot's first decode step, as
/// `EngineCore` forwards it by hand.
fn prefill_page_into_slot(
    sl: &slots::ModelSlots,
    slot: usize,
    p: &Prepared,
    head: Option<&(Vec<f32>, Vec<[u32; 4]>)>,
    d: usize,
) -> bool {
    const PREFILL_STEP: usize = 256;
    let ids = &p.prompt.ids;
    let span = p.prompt.span();
    let pre = &ids[..ids.len() - 1];
    let mut base = 0usize;
    while base < pre.len() {
        let end = (base + PREFILL_STEP).min(pre.len());
        let chunk = &pre[base..end];
        let (h, img, tail) = split_prefill(base, chunk.len(), span.clone());
        let at = |r: &Range<usize>| &chunk[r.start - base..r.end - base];
        if !h.is_empty() {
            let ok = match head {
                Some((rows, pos3)) => sl.prefill_rows(
                    slot, at(&h), &rows[h.start * d..h.end * d], h.start, Some(&pos3[h.clone()]),
                ),
                None => sl.prefill_ids(slot, at(&h), h.start),
            };
            if !ok {
                return false;
            }
        }
        if !img.is_empty() {
            let lo = (img.start - span.start) * d;
            let hi = (img.end - span.start) * d;
            let p3 = p.sp.img.as_ref()
                .map(|v| &v[img.start - span.start..img.end - span.start]);
            if !sl.prefill_rows(slot, at(&img), &p.rows[lo..hi], img.start, p3) {
                return false;
            }
        }
        if !tail.is_empty() && !sl.prefill_ids(slot, at(&tail), tail.start) {
            return false;
        }
        base = end;
    }
    true
}

/// The batch driver: prepare-and-prefill on admission, emit on completion.
///
/// Tokens are not streamed to stdout as they arrive: four pages interleaved token by
/// token would be an unsplittable blob, so a page's text is buffered and written
/// whole, with its `<!-- ojas-ocr page i/n -->` marker, when it finishes. That is the
/// one user-visible difference from the sequential path.
struct BatchDriver<'a> {
    sl: &'a slots::ModelSlots<'a>,
    ctx: &'a PageCtx<'a>,
    bpe: &'a Bpe,
    d: usize,
    /// Wall clock spent inside `prepare` — the ViT, which is per page and does not
    /// batch (`clip_support_batch` is false for this architecture in the reference
    /// too). It stalls every other slot, so it is measured rather than assumed small.
    encode_secs: f64,
    tokens: usize,
    looped: usize,
    truncated: usize,
    skipped: usize,
    done: usize,
}

impl crate::sched::PageSource for BatchDriver<'_> {
    fn admit(&mut self, slot: usize, page: usize) -> Result<Option<crate::sched::Seed>> {
        let t0 = std::time::Instant::now();
        let prepared = self.ctx.prepare(page)?;
        self.encode_secs += t0.elapsed().as_secs_f64();
        let Some(p) = prepared else {
            self.skipped += 1;
            return Ok(None);
        };
        let head = self.ctx.head_for(&p);
        if !prefill_page_into_slot(self.sl, slot, &p, head.as_ref(), self.d) {
            bail!(
                "the decoder refused a slot prefill for {}; output would not be a \
                 transcription of that page",
                self.ctx.pages[page].display()
            );
        }
        Ok(Some(crate::sched::Seed {
            first_token: *p.prompt.ids.last().expect("a spliced prompt is never empty"),
            prompt_len: p.prompt.ids.len(),
        }))
    }

    fn finished(&mut self, done: crate::sched::Done) -> Result<()> {
        use crate::sched::Finish;
        if done.finish == Finish::Skipped {
            return Ok(()); // already reported by `prepare`
        }
        // A failed command buffer leaves its outputs undefined, so every token of
        // every slot in flight is suspect — not only this page's.
        if let Some(err) = ojas_core::device_fault::peek() {
            eprintln!();
            bail!(
                "device fault while decoding {}; the batch is truncated and the session must \
                 be reloaded: {err}",
                self.ctx.pages[done.page].display()
            );
        }
        self.done += 1;
        // Emitted tokens, not produced: the sequential path's per-page count comes
        // from `stream`, which never sees the terminator. Counting `produced` here
        // would make the batched path look 1 token/page faster and the two summaries
        // incomparable.
        self.tokens += done.tokens.len();
        // Detokenized whole, which is what streaming it token by token would have
        // produced: `Detok` is a UTF-8 continuation buffer, not a stateful decoder.
        let mut dt = crate::detok::Detok::default();
        let mut text = String::new();
        for &t in &done.tokens {
            text.push_str(&dt.push(self.bpe, t));
        }
        text.push_str(&dt.finish());
        {
            use std::io::Write;
            self.ctx.page_marker(done.page);
            let mut o = std::io::stdout();
            let _ = o.write_all(text.as_bytes());
            let _ = o.write_all(b"\n");
            let _ = o.flush();
        }
        if let Finish::Looped(p) = done.finish {
            self.looped += 1;
            let block: String = done.tokens[done.tokens.len() - p..]
                .iter()
                .map(|&t| self.bpe.decode(t as usize))
                .collect();
            eprintln!(
                "\n  REPETITION GUARD: output looped on a {p}-token block {block:?} after \
                 {} tokens — stopped. The page is incomplete. Surya handles this case by \
                 falling back to layout-guided block recognition; there is no such fallback \
                 here yet.",
                done.tokens.len()
            );
        }
        if done.finish == Finish::Budget {
            self.truncated += 1;
        }
        eprintln!(
            "  [{}/{}] {} | slot {} | {} tokens in {} rounds | {}",
            done.page + 1,
            self.ctx.pages.len(),
            self.ctx.pages[done.page].display(),
            done.slot,
            done.tokens.len(),
            done.rounds,
            match done.finish {
                Finish::Stop(t) => format!("ended at token {t}"),
                Finish::Budget => "TRUNCATED at -n/context".to_string(),
                Finish::Looped(p) => format!("stopped by the repetition guard ({p}-token block)"),
                Finish::Skipped => unreachable!(),
            }
        );
        Ok(())
    }
}

/// `OJAS_OCR_ADMIT=continuous|static`. `static` runs fixed batches to completion, the
/// A/B control for continuous batching.
pub(crate) fn admission_from(want: &str) -> Result<crate::sched::Admission> {
    use crate::sched::Admission;
    Ok(match want {
        "" | "auto" | "continuous" => Admission::Continuous,
        "static" | "fixed" => Admission::StaticRounds,
        other => bail!("OJAS_OCR_ADMIT={other:?} (want continuous or static)"),
    })
}

/// The throughput line, printed by both drivers so a B=1 run and a B=4 run are
/// directly comparable.
///
/// `s/page` is wall over pages transcribed, not a mean of per-page timings: in a batch
/// the per-page clock overlaps and summing it would double-count. Encode is broken out
/// because the ViT takes one image at a time on both paths, so the part batching can
/// improve is `wall - encode`.
#[allow(clippy::too_many_arguments)]
fn summary(
    label: &str,
    done: usize,
    total: usize,
    tokens: usize,
    wall: f64,
    encode: f64,
    skipped: usize,
    looped: usize,
    truncated: usize,
    occupancy: Option<(usize, f64)>,
) {
    let per = |n: usize| if n == 0 || wall <= 0.0 { 0.0 } else { n as f64 / wall };
    eprintln!(
        "\n  ocr {label}: {done} of {total} pages in {wall:.1}s | {:.3} pages/s | {:.2} s/page \
         | {tokens} tokens | {:.1} tok/s | encode {encode:.1}s ({:.0}%){}{}{}",
        per(done),
        if done == 0 { 0.0 } else { wall / done as f64 },
        per(tokens),
        if wall > 0.0 { encode * 100.0 / wall } else { 0.0 },
        match occupancy {
            Some((n, o)) => format!(" | {n} slots, occupancy {o:.3}"),
            None => String::new(),
        },
        if skipped > 0 { format!(" | {skipped} skipped") } else { String::new() },
        match (looped, truncated) {
            (0, 0) => String::new(),
            (l, t) => format!(" | {l} looped | {t} truncated"),
        },
    );
}

/// Run the batch through `n_slots` decoder slots.
///
/// The scheduler is `crate::sched`, which is where the policy and its tests are;
/// this function is only the wiring and the clock.
fn run_batched(
    sl: &slots::ModelSlots,
    ctx: &PageCtx,
    bpe: &Bpe,
    n_slots: usize,
) -> Result<()> {
    use crate::sched::{drive, Admission, SchedCfg, Scheduler};
    let (primary, secondary) = stop_ids(bpe, ctx.info, false);
    let admission = admission_from(&std::env::var("OJAS_OCR_ADMIT").unwrap_or_default())?;
    let cfg = SchedCfg {
        n_predict: ctx.opts.n_predict,
        ctx: ctx.info.context,
        eog: ctx.info.eog.clone(),
        primary,
        also_stop: secondary,
        // The same guard the sequential path runs, through the same function.
        loop_probe: Some(|g| loop_period(g, LOOP_MAX_PERIOD, LOOP_MIN_RUN)),
        admission,
    };
    eprintln!(
        "  batched decode: {n_slots} slots | {} admission | {} pages | the ViT still runs one \
         page at a time",
        if admission == Admission::Continuous { "continuous" } else { "static" },
        ctx.pages.len(),
    );
    // `reset_slot` covers the recurrent half only; KV is positional and is overwritten
    // as a slot's new page advances, so that suffices between pages. It does not touch
    // the single-sequence bookkeeping slot 0 shares with `prefill`/`prefill_embeds`
    // (the reuse token log, the prefilled high-water mark), so clear that once here:
    // a batch following `probe_injection`, or anything else that touched slot 0, then
    // starts from the state a fresh process would. Per-page resets remain `drive`'s
    // job.
    sl.m.reset_session();
    let mut sched = Scheduler::new(n_slots, 0..ctx.pages.len(), cfg);
    let mut drv = BatchDriver {
        sl,
        ctx,
        bpe,
        d: ctx.info.hidden_dim,
        encode_secs: 0.0,
        tokens: 0,
        looped: 0,
        truncated: 0,
        skipped: 0,
        done: 0,
    };
    let t0 = std::time::Instant::now();
    let r = drive(sl, &mut drv, &mut sched);
    let wall = t0.elapsed().as_secs_f64();
    let st = sched.stats();
    // Printed even when the run failed: a batch that died on page 900 still measured
    // 899, and the numbers say where it died.
    summary(
        "batched",
        drv.done,
        ctx.pages.len(),
        drv.tokens,
        wall,
        drv.encode_secs,
        drv.skipped,
        drv.looped,
        drv.truncated,
        Some((sched.n_slots(), st.occupancy(sched.n_slots()))),
    );
    r
}

pub fn ocr(model: &str, target: &str, opts: &RunOpts, context: usize) -> Result<()> {
    // Transcription is greedy unless the caller says otherwise. `RunOpts::default`
    // carries llama.cpp's chat defaults (temp 0.8, top_p 0.95, top_k 40,
    // repeat_penalty 1.1), and all four are wrong here: sampling invents text that was
    // never on the page, and the repeat penalty fights the `<div data-bbox=` that every
    // block legitimately starts with. `sample_set` is true only if a sampling flag was
    // passed, so `--temp 0.7` still wins.
    let greedy;
    let opts = if opts.sample_set { opts } else {
        greedy = RunOpts { sample: SampleOpts { temperature: 0.0, ..opts.sample.clone() }, ..opts.clone() };
        &greedy
    };
    let pages = expand_pages(target)?;
    let instruction = instruction_for(opts)?;
    let mmproj = resolve_mmproj(model)?;
    let (min_tok, max_tok) = token_budget();
    let pre = VitPreproc::qwen3vl().with_token_budget(min_tok, max_tok);

    // One `with_model` call for the whole batch: the load is the expensive part
    // (60-90 s on a big checkpoint) and a per-page process would pay it per page. Same
    // structure `serve` uses to hold a model across requests.
    with_model(model, opts.device, context, opts.precision, |m, bpe, info| {
        // The encoder is chosen here rather than before the load, which is why
        // `open_encoder` takes a `&dyn Model`: the preferred encoder is the decoder's
        // own vision tower, which does not exist until the decoder does. Selecting
        // earlier could only pick the CPU oracle.
        let t0 = std::time::Instant::now();
        let enc = open_encoder(m, &mmproj)?;
        if enc.width() != info.hidden_dim {
            bail!(
                "projector writes {}-wide rows but the decoder's hidden dim is {} — \
                 the mmproj and the model are from different checkpoints",
                enc.width(),
                info.hidden_dim
            );
        }
        let (rows_ok, mrope_ok) = probe_injection(m, info.hidden_dim);
        if !rows_ok {
            bail!(
                "the {} backend for {} has no embedding-injection path \
                 (Model::prefill_embeds returned false), so an image span cannot enter the \
                 decoder. `ojas ocr` needs the Metal decoder; the CPU decoders and \
                 disk-streamed MoE models cannot run it.",
                info.backend,
                info.arch
            );
        }
        let vtok = VisionTokens::resolve(bpe)?;
        let scaffold = encode_scaffold(bpe, &info.arch, &instruction, &vtok)?;

        // The anchored layout needs the ChatML head's residual rows: it puts that head
        // at shifted positions and only `prefill_embeds` can carry a coordinate. Read
        // once for the batch, then the table is dropped — the head is ~7 rows of a
        // ~134 MB tensor. A failure here is not fatal; it costs exactness on seven
        // scaffolding tokens, so the run degrades to `Grid` and says so.
        let wanted_anchor = choose_layout(mrope_ok, true)? == PosLayout::Anchored;
        let head_rows: Option<Vec<f32>> = if wanted_anchor {
            match EmbedTable::open(model).and_then(|t| t.gather(&scaffold.head)) {
                Ok(r) => Some(r),
                Err(e) => {
                    eprintln!("  note: no host-side token_embd rows ({e:#}); \
                               the ChatML head cannot be shifted");
                    None
                }
            }
        } else {
            None
        };
        let layout = choose_layout(mrope_ok, head_rows.is_some())?;
        // "could not read the rows" and "was not asked to" are different causes and the
        // caveat must name the right one. `head_rows` alone cannot tell them apart,
        // because the read is skipped unless anchoring was wanted.
        let head_rows_failed = wanted_anchor && head_rows.is_none();
        eprintln!(
            "  {} | {} | proj {} | image budget {}..{} tokens | positions {} | encoder ready in {:.1}s",
            mmproj.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            enc.name(),
            enc.width(),
            min_tok,
            max_tok,
            layout_name(layout),
            t0.elapsed().as_secs_f64(),
        );
        if layout != PosLayout::Anchored {
            eprintln!("  {}", layout_caveat(layout, mrope_ok, head_rows_failed));
        }

        let ctx = PageCtx {
            enc: &*enc,
            pre: &pre,
            scaffold: &scaffold,
            layout,
            info,
            opts,
            max_tok,
            pages: &pages,
            head_rows: head_rows.as_deref(),
            banner_shown: std::cell::Cell::new(false),
        };

        // The batching gate, driven by capability rather than a flag:
        // `Model::max_slots()` is 1 on every decoder that has not implemented slots,
        // and 1 slot is the sequential path. A decoder without batching, a single-page
        // run and `OJAS_OCR_SLOTS=1` therefore all land on the same code, with no
        // second implementation to keep in step.
        let sl = slots::ModelSlots { m };
        let n_slots = batch_slots(
            crate::sched::SlotModel::max_slots(&sl), pages.len(), slots_env());
        if n_slots > 1 {
            return run_batched(&sl, &ctx, bpe, n_slots);
        }

        let (primary, secondary) = stop_ids(bpe, info, false);
        let t_run = std::time::Instant::now();
        let (mut looped, mut skipped, mut truncated) = (0usize, 0usize, 0usize);
        let (mut tokens, mut encode_secs, mut done_pages) = (0usize, 0.0f64, 0usize);
        for (i, page) in pages.iter().enumerate() {
            // Unconditional: a recurrent decoder carries SSM/GDN and MTP state
            // forward, and prefix reuse would match this page's placeholder ids
            // against the last page's cache.
            m.reset_session();

            let te = std::time::Instant::now();
            let prepared = ctx.prepare(i)?;
            encode_secs += te.elapsed().as_secs_f64();
            let Some(p) = prepared else {
                skipped += 1;
                continue;
            };
            ctx.page_marker(i);

            // Taken before the fields are moved into the wrapper.
            let head = ctx.head_for(&p);
            let Prepared { prompt, rows, sp } = p;
            let wrapper = VisionPrefill {
                inner: m,
                span: prompt.span(),
                rows,
                d: info.hidden_dim,
                img_pos3: sp.img,
                // Only the anchored layout moves the head; the other two prefill it by
                // id at its cache rows.
                head,
                refused: std::cell::Cell::new(false),
            };
            let mut core = EngineCore::new(&wrapper as &dyn Model);
            core.eos = primary;
            core.eog = info.eog.clone();

            // The guard runs inside `stream`'s token callback and stops generation the
            // way an EOG would, so the partial page is still emitted and detokenized
            // cleanly.
            let mut seen: Vec<u32> = Vec::with_capacity(opts.n_predict);
            let mut fired: Option<usize> = None;
            let mut guard = |t: u32| -> bool {
                seen.push(t);
                match loop_period(&seen, LOOP_MAX_PERIOD, LOOP_MIN_RUN) {
                    Some(p) => {
                        fired = Some(p);
                        false
                    }
                    None => true,
                }
            };
            let (n, ttft, decode) =
                stream(&core, bpe, &prompt.ids, opts.n_predict, opts, secondary, &mut guard);
            tokens += n;
            done_pages += 1;
            // The same clamp the scheduler applies per slot, through the same function:
            // `-n` alone would miscount a page the context truncated.
            if n >= crate::sched::budget_for(opts.n_predict, info.context, prompt.ids.len()) {
                truncated += 1;
            }

            if wrapper.refused.get() {
                bail!("the decoder refused an embedding injection mid-prompt; \
                       output for {} is not a transcription of that page", page.display());
            }
            if let Some(err) = ojas_core::device_fault::peek() {
                eprintln!();
                bail!(
                    "device fault after {n} tokens on {}; output is truncated and the session \
                     must be reloaded: {err}",
                    page.display()
                );
            }
            if let Some(p) = fired {
                looped += 1;
                let block: String = seen[seen.len() - p..]
                    .iter()
                    .map(|&t| bpe.decode(t as usize))
                    .collect();
                eprintln!(
                    "\n  REPETITION GUARD: output looped on a {p}-token block {block:?} after \
                     {n} tokens — stopped. The page is incomplete. Surya handles this case by \
                     falling back to layout-guided block recognition; there is no such fallback \
                     here yet."
                );
            }
            {
                use std::io::Write;
                let mut o = std::io::stdout();
                let _ = o.write_all(b"\n");
                let _ = o.flush();
            }
            eprintln!(
                "  {n} tokens | first token {ttft:.2}s | {:.2} tok/s",
                if n < 2 || decode <= 0.0 { 0.0 } else { (n - 1) as f64 / decode }
            );
        }
        if pages.len() > 1 {
            summary(
                "sequential",
                done_pages,
                pages.len(),
                tokens,
                t_run.elapsed().as_secs_f64(),
                encode_secs,
                skipped,
                looped,
                truncated,
                None,
            );
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- prompt construction ----------------------------------------------

    /// The scaffolding must be surya's byte for byte: `<|im_start|>user\n`, the
    /// markers around the span, the instruction, then an open assistant turn.
    #[test]
    fn template_halves_wrap_the_span_in_surya_chatml() {
        let (h, t) = template_halves("qwen35", "OCR this.").unwrap();
        assert_eq!(h, "<|im_start|>user\n<|vision_start|>");
        assert_eq!(t, "<|vision_end|>OCR this.<|im_end|>\n<|im_start|>assistant\n");
    }

    /// Rejoining the halves must reproduce the ordinary template exactly.
    #[test]
    fn template_halves_rejoin_to_the_shared_template() {
        let (h, t) = template_halves("qwen35", "hi").unwrap();
        let want = ojas_tokenize::chat_template("qwen35", "<|vision_start|><|vision_end|>hi");
        assert_eq!(format!("{h}{t}"), want);
    }

    #[test]
    fn an_instruction_carrying_the_marker_is_refused() {
        let bad = format!("x{SPAN_MARKER}y");
        assert!(template_halves("qwen35", &bad).is_err());
    }

    // ---- id-span assembly --------------------------------------------------

    #[test]
    fn splice_puts_the_pads_between_the_halves() {
        let p = splice_span(&[1, 2, 9], 11, 4, &[10, 5]);
        assert_eq!(p.ids, vec![1, 2, 9, 11, 11, 11, 11, 10, 5]);
        assert_eq!(p.image_at, 3);
        assert_eq!(p.n_image, 4);
        assert_eq!(p.span(), 3..7);
    }

    /// The span must land strictly between the markers, or the decoder reads image
    /// rows as text (or text rows as image).
    #[test]
    fn the_span_sits_exactly_between_the_markers() {
        let p = splice_span(&[1, 9], 11, 3, &[10, 2]);
        assert_eq!(p.ids[p.image_at - 1], 9, "row before the span is <|vision_start|>");
        assert_eq!(p.ids[p.image_at + p.n_image], 10, "row after the span is <|vision_end|>");
        assert!(p.ids[p.span()].iter().all(|&t| t == 11));
    }

    /// A zero-token image splices nothing and leaves `<|vision_start|>` against
    /// `<|vision_end|>`; the arithmetic must still hold.
    #[test]
    fn an_empty_span_is_well_formed() {
        let p = splice_span(&[1, 9], 11, 0, &[10]);
        assert_eq!(p.ids, vec![1, 9, 10]);
        assert_eq!(p.span(), 2..2);
    }

    /// The last id drives the first decode step, so the span must never reach it:
    /// `EngineCore` prefills `ids[..len-1]` and forwards the final id by hand.
    #[test]
    fn the_span_never_touches_the_last_id() {
        let p = splice_span(&[1, 9], 11, 4096, &[10, 2, 3]);
        assert!(p.span().end < p.ids.len() - 1);
    }

    // ---- prefill splitting -------------------------------------------------

    #[test]
    fn a_chunk_entirely_before_the_span_is_all_text() {
        let (h, i, t) = split_prefill(0, 7, 7..4103);
        assert_eq!((h, i, t), (0..7, 7..7, 7..7));
    }

    #[test]
    fn a_chunk_entirely_inside_the_span_is_all_image() {
        let (h, i, t) = split_prefill(256, 256, 7..4103);
        assert_eq!((h, i, t), (256..256, 256..512, 512..512));
    }

    #[test]
    fn a_chunk_straddling_the_start_splits_in_two() {
        let (h, i, t) = split_prefill(0, 256, 7..4103);
        assert_eq!((h, i, t), (0..7, 7..256, 256..256));
    }

    #[test]
    fn a_chunk_straddling_the_end_splits_in_two() {
        let (h, i, t) = split_prefill(3840, 256, 7..4103);
        assert_eq!((h, i, t), (3840..3840, 3840..4096, 4096..4096));
        let (h, i, t) = split_prefill(4096, 100, 7..4103);
        assert_eq!((h, i, t), (4096..4096, 4096..4103, 4103..4196));
    }

    /// A short prompt can have one chunk covering text, image and text.
    #[test]
    fn a_chunk_covering_everything_splits_in_three() {
        let (h, i, t) = split_prefill(0, 20, 7..11);
        assert_eq!((h, i, t), (0..7, 7..11, 11..20));
    }

    /// The split must partition the chunk — no row prefilled twice, none dropped —
    /// which is what the row-offset arithmetic depends on.
    #[test]
    fn the_split_always_partitions_the_chunk() {
        let span = 7..4103;
        for base in [0usize, 3, 7, 8, 255, 256, 4102, 4103, 4104] {
            for len in [0usize, 1, 5, 256, 4200] {
                let (h, i, t) = split_prefill(base, len, span.clone());
                assert!(h.start >= base && t.end <= base + len, "inside the chunk");
                assert_eq!(h.end, i.start, "head meets image at {base}+{len}");
                assert_eq!(i.end, t.start, "image meets tail at {base}+{len}");
                assert_eq!(h.start, base);
                assert_eq!(t.end, base + len);
                assert!(
                    i.is_empty() || (i.start >= span.start && i.end <= span.end),
                    "a non-empty image part must lie inside the span ({base}+{len})"
                );
            }
        }
    }

    // ---- M-RoPE positions --------------------------------------------------

    /// A 1x1 grid must produce the text convention `(p,p,p,0)`: every rope section
    /// then reads the same number and the sectioned kernel stays bit-identical to
    /// plain rope. Changing this changes every text-only model.
    #[test]
    fn a_one_by_one_grid_degenerates_to_the_text_convention() {
        for &l in &[PosLayout::Grid, PosLayout::Anchored] {
            let sp = span_positions(9, 1, 1, l);
            assert_eq!(sp.shift, 0, "a 1-row image consumes exactly 1 position");
            assert_eq!(sp.img.unwrap(), vec![[9, 9, 9, 0]], "{l:?}");
        }
    }

    #[test]
    fn text_positions_put_zero_in_the_fourth_stream() {
        assert_eq!(text_positions(5, 3), vec![[5, 5, 5, 0], [6, 6, 6, 0], [7, 7, 7, 0]]);
    }

    /// `w` advances every row, `h` every `nx` rows — the ordering the merge
    /// permutation produces, and what makes the span a grid rather than a line.
    #[test]
    fn image_rows_get_two_dimensional_coordinates() {
        let sp = span_positions(7, 3, 2, PosLayout::Grid);
        assert_eq!(sp.shift, 0);
        assert_eq!(
            sp.img.unwrap(),
            vec![
                [7, 7, 7, 0], [7, 7, 8, 0], [7, 7, 9, 0],   // row 0: w = 7,8,9
                [7, 8, 7, 0], [7, 8, 8, 0], [7, 8, 9, 0],   // row 1: h stepped, w reset
            ]
        );
    }

    /// A dimension smaller than `patch * merge` gives a zero-row grid. The
    /// preprocessor should never produce one, but the reporting path must not
    /// underflow if it ever does.
    #[test]
    fn a_degenerate_grid_reports_rather_than_underflows() {
        let prompt = splice_span(&[1, 9], 11, 0, &[10]);
        for &(nx, ny) in &[(0usize, 0usize), (5, 0), (0, 5)] {
            for &l in &[PosLayout::Sequential, PosLayout::Grid, PosLayout::Anchored] {
                let sp = span_positions(prompt.image_at, nx, ny, l);
                assert!(sp.img.is_none() && sp.shift == 0, "{nx}x{ny} {l:?}");
                let line = describe_positions(&sp, &prompt, nx, ny, l);
                assert!(line.contains("positions:"), "{line}");
            }
        }
    }

    #[test]
    fn the_sequential_layout_supplies_no_coordinates_at_all() {
        let sp = span_positions(7, 48, 32, PosLayout::Sequential);
        assert!(sp.img.is_none(), "Sequential must pass None, not a rebuilt identity");
        assert_eq!(sp.shift, 0);
    }

    /// The anchored layout's central invariant. The decode loop and the tail prefill
    /// both put a text row at its cache row, so the span's largest coordinate must be
    /// `pos_0 + nx*ny - 1`, one less than the next cache row; otherwise the text after
    /// the image is not at `image_t + max(nx,ny)` and llama.cpp's rule is broken.
    /// Checked over ragged grids, including the `nx != ny` ones where `max` matters.
    #[test]
    fn anchoring_makes_the_next_cache_row_the_next_sequence_position() {
        for &(nx, ny) in &[(1, 1), (2, 1), (1, 2), (3, 2), (24, 24), (48, 32), (32, 48), (7, 5)] {
            for &pos_0 in &[0usize, 1, 7, 300] {
                let sp = span_positions(pos_0, nx, ny, PosLayout::Anchored);
                let p3 = sp.img.unwrap();
                assert_eq!(p3.len(), nx * ny);
                let hi = p3.iter().flat_map(|c| [c[0], c[1], c[2]]).max().unwrap();
                assert_eq!(
                    hi as usize + 1, pos_0 + nx * ny,
                    "{nx}x{ny}@{pos_0}: the image must end exactly at its last cache row"
                );
                // The temporal position is shared by the whole image, and the text
                // after it must be max(nx,ny) beyond it, not nx*ny.
                let t = p3[0][0] as usize;
                assert_eq!(t + nx.max(ny), pos_0 + nx * ny, "{nx}x{ny}@{pos_0}");
                assert_eq!(sp.shift as usize, nx * ny - nx.max(ny));
                assert!(p3.iter().all(|c| c[3] == 0), "one image = image index 0");
            }
        }
    }

    /// Pins the size of the gap the unanchored grid leaves, which is the deviation the
    /// caveat prints.
    #[test]
    fn the_unanchored_grid_leaves_a_measurable_gap() {
        let (nx, ny) = (24usize, 24usize);
        let sp = span_positions(7, nx, ny, PosLayout::Grid);
        let p3 = sp.img.unwrap();
        let hi = p3.iter().flat_map(|c| [c[0], c[1], c[2]]).max().unwrap() as usize;
        // Reference: text resumes at image_t + max(nx,ny) = 7 + 24 = 31.
        assert_eq!(hi + 1, 7 + nx.max(ny));
        // Cache rows put it at 7 + 576 = 583, so 552 positions too far.
        assert_eq!(7 + nx * ny - (hi + 1), 552);
    }

    #[test]
    fn the_layout_ladder_prefers_what_the_decoder_can_do() {
        use PosLayout::*;
        assert_eq!(layout_from("", true, true).unwrap(), Anchored);
        assert_eq!(layout_from("auto", true, true).unwrap(), Anchored);
        // No head rows -> the head cannot be shifted, so the grid stays unanchored.
        assert_eq!(layout_from("", true, false).unwrap(), Grid);
        // A decoder that refuses pos3 gets the pre-M-RoPE layout whatever is asked.
        assert_eq!(layout_from("anchor", false, true).unwrap(), Sequential);
        assert_eq!(layout_from("grid", false, true).unwrap(), Sequential);
        // ...except `seq`, which is an explicit A/B against the pre-M-RoPE layout.
        assert_eq!(layout_from("seq", true, true).unwrap(), Sequential);
        assert_eq!(layout_from("grid", true, true).unwrap(), Grid);
        assert!(layout_from("2d", true, true).is_err(), "a typo must not silently mean auto");
    }

    /// The log line is the only evidence a run leaves that the coordinates form a
    /// grid, so assert it shows both steps.
    #[test]
    fn the_position_log_shows_w_and_h_stepping() {
        let prompt = splice_span(&[1, 9], 11, 6, &[10, 2]);
        let sp = span_positions(prompt.image_at, 3, 2, PosLayout::Anchored);
        let line = describe_positions(&sp, &prompt, 3, 2, PosLayout::Anchored);
        // head is 2 ids, so pos_0 = 2; nx*ny - max = 6 - 3 = 3, so t0 = 5.
        assert!(line.contains("[0]=(5,5,5,0)"), "{line}");
        assert!(line.contains("[1]=(5,5,6,0)"), "w must step at row 1: {line}");
        assert!(line.contains("[3]=(5,6,5,0)"), "h must step at row nx: {line}");
        assert!(line.contains("delta 3"), "{line}");
        assert!(line.contains("head 0..2 -> 3..5"), "the head must be shifted too: {line}");
        // The deviation the anchored layout drives to zero.
        assert!(line.contains("image-to-text error 0"), "{line}");
        let grid = describe_positions(
            &span_positions(prompt.image_at, 3, 2, PosLayout::Grid), &prompt, 3, 2, PosLayout::Grid);
        assert!(grid.contains("image-to-text error 3"), "unanchored must report the gap: {grid}");
    }

    // ---- f16 decode --------------------------------------------------------

    /// The embedding rows are read as raw f16 bytes, so a wrong decode perturbs the
    /// ChatML head silently. Exact values, including the subnormal branch that a
    /// shift-only implementation gets wrong.
    #[test]
    fn f16_decodes_exactly() {
        for (bits, want) in [
            (0x0000u16, 0.0f32),
            (0x8000, -0.0),
            (0x3c00, 1.0),
            (0xbc00, -1.0),
            (0x3555, 0.333251953125),
            (0x7bff, 65504.0),                 // largest normal
            (0x0400, 6.103515625e-5),          // smallest normal
            (0x0001, 5.960464477539063e-8),    // smallest subnormal
            (0x03ff, 6.0975551605224609e-5),   // largest subnormal
            (0x0200, 3.0517578125e-5),         // mid subnormal
        ] {
            assert_eq!(f16_to_f32(bits), want, "f16 {bits:#06x}");
        }
        assert!(f16_to_f32(0x7c00).is_infinite() && f16_to_f32(0x7c00) > 0.0);
        assert!(f16_to_f32(0xfc00).is_infinite() && f16_to_f32(0xfc00) < 0.0);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    // ---- geometry ----------------------------------------------------------

    /// `clip_n_output_tokens_x` halves the patch grid, so the M-RoPE grid is the
    /// post-merge one. Getting this wrong is a 2x transpose of every coordinate.
    #[test]
    fn the_grid_is_post_merge_not_the_raw_patch_grid() {
        assert_eq!(merged_grid(768, 768, 16, 2), (24, 24));
        assert_eq!(merged_grid(1536, 1024, 16, 2), (48, 32));
        // merge 1 (a projector with no spatial merge) is the raw patch grid.
        assert_eq!(merged_grid(768, 768, 16, 1), (48, 48));
    }

    // ---- repetition detection ---------------------------------------------

    #[test]
    fn a_single_token_run_is_a_loop_only_once_it_is_long() {
        let short: Vec<u32> = vec![5; LOOP_MIN_RUN - 1];
        assert_eq!(loop_period(&short, LOOP_MAX_PERIOD, LOOP_MIN_RUN), None);
        let long: Vec<u32> = vec![5; LOOP_MIN_RUN];
        assert_eq!(loop_period(&long, LOOP_MAX_PERIOD, LOOP_MIN_RUN), Some(1));
    }

    #[test]
    fn a_repeated_block_is_detected_at_its_own_period() {
        let block = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let g: Vec<u32> = block.iter().cycle().take(block.len() * 4).copied().collect();
        assert_eq!(loop_period(&g, LOOP_MAX_PERIOD, LOOP_MIN_RUN), Some(8));
    }

    /// Three repeats of a long block is structure (a table row), not a loop.
    #[test]
    fn three_repeats_of_a_long_block_are_not_a_loop() {
        let block: Vec<u32> = (100..116).collect();
        let g: Vec<u32> = block.iter().cycle().take(block.len() * 3).copied().collect();
        assert_eq!(loop_period(&g, LOOP_MAX_PERIOD, LOOP_MIN_RUN), None);
    }

    /// Only the tail matters: a loop that has started must be caught even though the
    /// first few hundred tokens were fine, and prose that merely contains a repeated
    /// phrase must not trip it.
    #[test]
    fn the_guard_reads_the_tail_not_the_history() {
        let mut g: Vec<u32> = (0..500).collect();
        assert_eq!(loop_period(&g, LOOP_MAX_PERIOD, LOOP_MIN_RUN), None);
        g.extend(std::iter::repeat_n([7u32, 8, 9], 8).flatten());
        assert_eq!(loop_period(&g, LOOP_MAX_PERIOD, LOOP_MIN_RUN), Some(3));
    }

    #[test]
    fn ordinary_output_does_not_trip_the_guard() {
        // `<div>..</div>` shaped: repeated structure with differing content.
        let mut g = Vec::new();
        for i in 0..40u32 {
            g.extend_from_slice(&[60, 100, 62, 1000 + i, 60, 47, 100, 62]);
        }
        assert_eq!(loop_period(&g, LOOP_MAX_PERIOD, LOOP_MIN_RUN), None);
    }

    #[test]
    fn a_period_longer_than_the_window_is_not_reported() {
        let block: Vec<u32> = (0..LOOP_MAX_PERIOD as u32 + 1).collect();
        let g: Vec<u32> = block.iter().cycle().take(block.len() * 6).copied().collect();
        assert_eq!(loop_period(&g, LOOP_MAX_PERIOD, LOOP_MIN_RUN), None);
    }

    // ---- page selection ----------------------------------------------------

    #[test]
    fn glob_matches_star_and_question() {
        assert!(glob_match("*.png", "page-001.png"));
        assert!(glob_match("page-*.png", "page-001.png"));
        assert!(glob_match("page-00?.png", "page-001.png"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("page-001.png", "page-001.png"));
        assert!(!glob_match("*.png", "page-001.jpg"));
        assert!(!glob_match("page-00?.png", "page-0012.png"));
        assert!(!glob_match("a*b", "ac"));
    }

    /// The backtracking case: a greedy `*` has to give characters back.
    #[test]
    fn glob_backtracks_out_of_a_greedy_star() {
        assert!(glob_match("*abc", "zzabcabc"));
        assert!(glob_match("*a*b", "xxaybzzb"));
        assert!(!glob_match("*abc", "zzabcd"));
    }

    #[test]
    fn only_image_extensions_are_picked_up() {
        assert!(is_image_name("scan.PNG"), "extension match is case-insensitive");
        assert!(is_image_name("a.jpeg"));
        assert!(!is_image_name("page.json"));
        assert!(!is_image_name("README"));
        assert!(!is_image_name(".DS_Store"), "a dotfile is not an extension");
    }

    /// Ordering is part of the contract: a thousand-page run must be reproducible and
    /// resumable, and `read_dir` order is not.
    #[test]
    fn directory_expansion_filters_and_sorts() {
        let got = pick_images(vec![
            "b.png".into(),
            "notes.txt".into(),
            "a.jpg".into(),
            ".DS_Store".into(),
            "c.tiff".into(),
        ]);
        assert_eq!(got, vec!["a.jpg", "b.png", "c.tiff"]);
    }

    #[test]
    fn a_directory_of_non_images_yields_nothing() {
        assert!(pick_images(vec!["a.txt".into(), "b.json".into()]).is_empty());
    }

    // ---- batching policy ---------------------------------------------------

    /// Three ceilings meet in `batch_slots`: what the decoder allocated, how many pages
    /// there are, and the measured M=4 peak.
    #[test]
    fn the_slot_count_is_the_smallest_of_three_ceilings() {
        // An unbatched decoder is 1 whatever is asked for, which keeps `ojas ocr` on
        // its sequential path on every CPU backend and on a Metal decoder loaded
        // without OJAS_SLOTS.
        assert_eq!(batch_slots(1, 100, None), 1);
        assert_eq!(batch_slots(1, 100, Some(8)), 1);
        // A one-page run has nothing to batch with.
        assert_eq!(batch_slots(4, 1, None), 1);
        // More slots than pages would put an empty row in every dispatch. Asserted
        // with an explicit `want` so the page ceiling is tested independently of
        // whatever DEFAULT_SLOTS happens to be.
        assert_eq!(batch_slots(8, 3, Some(8)), 3);
        assert_eq!(batch_slots(8, 3, None), DEFAULT_SLOTS.min(3));
        // The default stops at the fastest reproducible configuration: B=4 needs the
        // pair-tiled Q4 path to beat B=2, and that path disagrees with itself run to
        // run (see graph_decode::q4_pair).
        assert_eq!(batch_slots(8, 100, None), DEFAULT_SLOTS);
        // ...and the env var can go past it, or below it, deliberately.
        assert_eq!(batch_slots(8, 100, Some(8)), 8);
        assert_eq!(batch_slots(8, 100, Some(2)), 2);
        assert_eq!(batch_slots(8, 100, Some(1)), 1, "1 slot is the sequential path");
        // Degenerate inputs still produce a usable slot count.
        assert_eq!(batch_slots(0, 0, None), 1);
    }

    #[test]
    fn the_admission_policy_is_selectable_and_typos_are_refused() {
        use crate::sched::Admission;
        assert_eq!(admission_from("").unwrap(), Admission::Continuous);
        assert_eq!(admission_from("auto").unwrap(), Admission::Continuous);
        assert_eq!(admission_from("continuous").unwrap(), Admission::Continuous);
        assert_eq!(admission_from("static").unwrap(), Admission::StaticRounds);
        assert_eq!(admission_from("fixed").unwrap(), Admission::StaticRounds);
        assert!(admission_from("batched").is_err(), "a typo must not silently mean continuous");
    }

    // ---- misc --------------------------------------------------------------

    #[test]
    fn the_default_context_holds_a_full_page_prompt() {
        // 4 096 image tokens + ChatML scaffolding + the default instruction, all
        // one-byte-per-token on this vocabulary, which the engine-wide 4 096 default
        // cannot hold.
        let scaffold = 7 + 1 + DEFAULT_INSTRUCTION.len() + 12;
        assert!(4096 + scaffold > 4096);
        assert!(4096 + scaffold < DEFAULT_CTX, "a page must fit the ocr default");
    }
}
