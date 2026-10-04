//! Readouts of a causal language model, whose prompt ends where its answer would
//! begin.
//!
//! - Labels: the prompt names every option by a label, and an option's score is the
//!   next-token logit of its label after the prompt.
//! - Pointer: each option ends with a marker token, and the model projects its
//!   final hidden states (`cls.output`) to a query half and a key half; an option's
//!   score is the scaled dot product of the last token's query with its marker's
//!   key.
//!
//! Both read the final hidden states ([`CausalBackend::prefill_hidden_slots`]), so
//! only the label rows of the output projection are ever multiplied, on the host.
//!
//! A question may be asked in more than one option order ([`Causal::variants`]),
//! each its own prompt. The prefix every prompt of a request shares, the state as a
//! rule, is evaluated once.
//!
//! Images: the template puts one media marker per image. The text between markers
//! is tokenized piece by piece, and each image enters as `<|vision_start|>`, one
//! row per merged patch from the vision tower, then `<|vision_end|>`. Every row
//! then takes a multi-axis rotary coordinate: text at `(p, p, p)`, the image's row
//! `r` and column `c` at `(p, p + r, p + c)`, and the text after an image resumes at
//! `p + max(columns, rows)`.

use super::json::Json;
use super::media::Image;
use super::template::{OptionView, PromptInput, Template};
use super::{invalid, read_f32, Question, QuestionKind, Scored};
use crate::backend::{CausalBackend, CausalLoad, DecisionGpu, PromptRows, SlotPrefill, MAX_SLOTS};
use anyhow::{bail, ensure, Context, Result};
use ojas_cpu::VitPreproc;
use ojas_formats::gguf::{Gguf, Meta};
use ojas_tokenize::Bpe;

/// How options are labelled in the prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LabelSet {
    /// `A`..`Z` then `a`..`z`, each of which must be one token.
    Letters,
    /// `A`..`Z` then `AA`..`ZZ`, keeping the codes that are one token, at most 255;
    /// the template is given each option's code as `label`.
    Codes,
}

/// Where the scores are read.
#[derive(Clone, Copy, Debug)]
pub(super) enum Read {
    /// The next-token logits of the option labels.
    Labels(LabelSet),
    /// The projected hidden states at the token that ends each option.
    Pointer { marker: &'static str },
}

/// What a causal model's profile fixes about its prompts and readout.
#[derive(Clone, Copy, Debug)]
pub(super) struct Causal {
    pub(super) read: Read,
    /// A choice is also asked with its options reversed, and the two averaged.
    pub(super) reversed_choice: bool,
    /// A noul is read as a rating on this many levels, at the first labels, rather
    /// than at its two options.
    pub(super) noul_ratings: Option<usize>,
    /// Object keys are sorted before the prompt is rendered.
    pub(super) sorted_keys: bool,
    /// The template takes text only: the state, the instructions and the options
    /// are flattened to text ([`plain_text`]) before it is rendered.
    pub(super) text_only: bool,
    /// Requests may carry images, when the model's vision projector is beside it.
    pub(super) images: bool,
}

impl Causal {
    /// Option orders `q` is asked in: the request's, then possibly reversed.
    pub(super) fn variants(&self, q: &Question) -> usize {
        if self.reversed_choice && q.kind == QuestionKind::Choice && q.options.len() > 1 { 2 } else { 1 }
    }

    /// Scores read per prompt of `q`.
    pub(super) fn outputs(&self, q: &Question) -> usize {
        match (q.kind, self.noul_ratings) {
            (QuestionKind::Noul, Some(levels)) => levels,
            _ => q.options.len(),
        }
    }
}

/// The pointer projection: `cls.output`, `[out][d]` row-major, and its bias. The
/// first half of the output is the query, the second the key.
struct Pointer {
    marker: u32,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Pointer {
    fn half(&self) -> usize { self.b.len() / 2 }

    /// Rows `range` of the projection of `h`.
    fn project(&self, h: &[f32], range: std::ops::Range<usize>) -> Vec<f32> {
        let d = h.len();
        range.map(|o| self.b[o] + self.w[o * d..(o + 1) * d].iter().zip(h).map(|(w, x)| w * x).sum::<f32>()).collect()
    }
}

/// The vision tower's tokens and preprocessing.
struct Vision {
    start: u32,
    end: u32,
    /// The placeholder id an image row carries; it is never embedded.
    pad: u32,
    preproc: VitPreproc,
}

/// An image through the vision tower: one row per merged patch, `columns x rows`.
struct EncodedImage {
    rows: Vec<f32>,
    columns: usize,
    grid_rows: usize,
}

/// One prompt: its tokens, their rotary coordinates when it holds images, where
/// each image's rows begin, the scores it yields, and the rows they are read at.
struct Prompt {
    ids: Vec<u32>,
    positions: Option<Vec<[u32; 4]>>,
    /// `(first row, image index)`.
    images: Vec<(usize, usize)>,
    outputs: usize,
    /// Rows whose hidden states are read: the last row, after a pointer readout's
    /// option markers.
    reads: Vec<usize>,
}

/// A prompt's rows as they are laid out: ids, a rotary coordinate per row, and
/// where each image's rows begin.
#[derive(Default)]
struct Layout {
    ids: Vec<u32>,
    positions: Vec<[u32; 4]>,
    /// `(first row, image index)`.
    images: Vec<(usize, usize)>,
    /// The position the next row takes.
    next: u32,
}

impl Layout {
    /// Text rows, one position each.
    fn text(&mut self, run: &[u32]) {
        for &t in run {
            self.ids.push(t);
            self.positions.push([self.next, self.next, self.next, 0]);
            self.next += 1;
        }
    }

    /// An image's `columns x rows` grid in raster order, at `(p, p + row, p + column)`;
    /// the text after it resumes `max(columns, rows)` positions on.
    fn image(&mut self, index: usize, pad: u32, columns: usize, rows: usize) {
        let p = self.next;
        self.images.push((self.ids.len(), index));
        for r in 0..columns * rows {
            self.ids.push(pad);
            self.positions.push([p, p + (r / columns) as u32, p + (r % columns) as u32, 0]);
        }
        self.next = p + columns.max(rows) as u32;
    }
}

/// The text the template stands in for each image, split out after rendering.
const MEDIA_MARKER: &str = "<__media__>";
/// Longest prompt, in tokens.
const MAX_PROMPT_TOKENS: usize = 8192;
/// Most options when the readout has no label set to bound them.
const MAX_POINTER_OPTIONS: usize = 255;

pub(super) struct CausalHead<C> {
    dec: C,
    tok: Bpe,
    template: Template,
    spec: Causal,
    /// Label texts, given to the template when the set names them.
    label_texts: Vec<String>,
    /// The output projection's row for each label, in label order.
    label_rows: Vec<Vec<f32>>,
    pointer: Option<Pointer>,
    vision: Option<Vision>,
    max_seq: usize,
    /// The prefix the last request shared, still in the cache: its ids; the recurrent
    /// state after it is the backend's saved state. Its KV rows stay valid, since a
    /// request writes only the rows after the prefix it continues from.
    held: std::cell::RefCell<Option<HeldPrefix>>,
}

/// A prompt prefix left in the cache by an earlier request.
struct HeldPrefix {
    ids: Vec<u32>,
}

impl<C: CausalBackend> CausalHead<C> {
    pub(super) fn load<'a, G: DecisionGpu<Causal<'a> = C> + 'a>(gpu: &'a G, g: &mut Gguf, template: Template, spec: Causal) -> Result<Self> {
        let tok = Bpe::from_gguf(g);
        let single = |text: &str| -> Option<u32> {
            match tok.encode(text).as_slice() { [one] => Some(*one as u32), _ => None }
        };
        let (mut labels, mut label_texts, mut pointer) = (Vec::new(), Vec::new(), None);
        match spec.read {
            Read::Labels(LabelSet::Letters) => {
                for c in ('A'..='Z').chain('a'..='z') {
                    labels.push(single(&c.to_string()).with_context(|| format!("label {c:?} is not a single token"))?);
                }
            }
            Read::Labels(LabelSet::Codes) => {
                let pairs = ('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}")));
                for code in ('A'..='Z').map(String::from).chain(pairs) {
                    if labels.len() == 255 { break; }
                    if let Some(id) = single(&code) {
                        labels.push(id);
                        label_texts.push(code);
                    }
                }
            }
            Read::Pointer { marker } => {
                let marker = single(marker).with_context(|| format!("the option marker {marker} is not a single token"))?;
                let w = read_f32(g, "cls.output.weight")?;
                let b = read_f32(g, "cls.output.bias")?;
                let d = g.meta_u32(&format!("{}.embedding_length", g.arch())).unwrap_or(0) as usize;
                ensure!(!b.is_empty() && b.len() % 2 == 0 && w.len() == b.len() * d,
                    "cls.output must project the {d}-wide hidden state to an even width");
                pointer = Some(Pointer { marker, w, b });
            }
        }
        if let Some(levels) = spec.noul_ratings { ensure!(labels.len() >= levels, "fewer labels than rating levels"); }
        // An untied model has its own output projection; a tied one reads the
        // token embeddings.
        let head = if g.tensors.contains_key("output.weight") { "output.weight" } else { "token_embd.weight" };
        let label_rows = g.read_rows(head, &labels.iter().map(|&t| t as usize).collect::<Vec<_>>())?;

        let max_seq = g.meta_u32(&format!("{}.context_length", g.arch())).map_or(MAX_PROMPT_TOKENS, |c| (c as usize).min(MAX_PROMPT_TOKENS));
        // One slot per prompt run at once: a request's prompts continue their shared
        // prefix a slot each, and one-prompt requests share passes a slot each. The
        // host's `parallel` (else `OJAS_SLOTS`) sets the count; unset, every slot the
        // backend allows. A projector beside the file is loaded only for a model that
        // takes images.
        let slots = ojas_core::config::EngineConfig::current().parallel
            .or_else(|| std::env::var("OJAS_SLOTS").ok()?.parse().ok()).unwrap_or(MAX_SLOTS).max(1);
        let dec = gpu.load_causal(g, CausalLoad { max_seq, slots, images: spec.images })?;
        ensure!(dec.slots() >= 1, "a causal decision model needs at least one sequence slot");
        let vision = match dec.vision_patch_merge() {
            Some((patch, merge)) if spec.images => {
                let floats = |k: &str| match g.meta.get(k) {
                    Some(Meta::FloatArr(v)) if v.len() == 3 => Some([v[0], v[1], v[2]]),
                    _ => None,
                };
                let base = VitPreproc::qwen3vl();
                Some(Vision {
                    start: single("<|vision_start|>").context("no single <|vision_start|> token")?,
                    end: single("<|vision_end|>").context("no single <|vision_end|> token")?,
                    pad: single("<|image_pad|>").context("no single <|image_pad|> token")?,
                    preproc: VitPreproc {
                        patch_size: patch as i32, n_merge: merge as i32,
                        mean: floats("clip.vision.image_mean").unwrap_or(base.mean),
                        std: floats("clip.vision.image_std").unwrap_or(base.std),
                        ..base
                    },
                })
            }
            _ => None,
        };
        Ok(CausalHead { dec, tok, template, spec, label_texts, label_rows, pointer, vision, max_seq, held: Default::default() })
    }

    /// Most options a question may have.
    pub(super) fn max_options(&self) -> usize {
        if self.pointer.is_some() { MAX_POINTER_OPTIONS } else { self.label_rows.len() }
    }

    /// Whether requests may carry images.
    pub(super) fn takes_images(&self) -> bool { self.vision.is_some() }

    /// Each image through the vision tower.
    fn encode_images(&self, images: &[Image]) -> Result<Vec<EncodedImage>> {
        let Some(v) = &self.vision else {
            ensure!(images.is_empty(), "this model takes no images");
            return Ok(Vec::new());
        };
        images.iter().map(|img| {
            let (w, h, planar) = v.preproc.preprocess(&img.rgb, img.width, img.height)?;
            let (columns, grid_rows) = self.dec.vision_grid(w, h).context("the model has no vision tower")?;
            let rows = self.dec.encode_image(&planar, w, h)?;
            ensure!(rows.len() == columns * grid_rows * self.dec.width(), "the vision tower returned {} values for a {columns}x{grid_rows} grid", rows.len());
            ensure!(rows.iter().all(|v| v.is_finite()), "the vision tower produced non-finite values");
            Ok(EncodedImage { rows, columns, grid_rows })
        }).collect()
    }

    fn prompt(&self, state: &Json, q: &Question, variant: usize, images: &[EncodedImage]) -> Result<Prompt> {
        // With images, the marker text may not occur in what the request wrote.
        let scrub = |j: Json| if images.is_empty() { j } else { j.map_strings(&|s: &str| s.replace(MEDIA_MARKER, " ")) };
        let prepare = |j: &Json| scrub(match (self.spec.text_only, j) {
            (true, Json::Null) => Json::Null,
            (true, j) => Json::Str(plain_text(j)),
            (false, j) if self.spec.sorted_keys => j.sorted(),
            (false, j) => j.clone(),
        });
        let (state, instructions) = (prepare(state), prepare(&q.instructions));
        let n = q.options.len();
        let shown: Vec<(String, Json)> = (0..n)
            .map(|i| &q.options[if variant == 0 { i } else { n - 1 - i }])
            .map(|o| {
                let key = if self.spec.text_only { escape_specials(&o.key) } else { o.key.clone() };
                let key = if images.is_empty() { key } else { key.replace(MEDIA_MARKER, " ") };
                (key, prepare(&o.description))
            })
            .collect();
        let text = self.template.render(&PromptInput {
            id: &q.id, kind: q.kind.name(), instructions: &instructions, state: &state,
            options: shown.iter().enumerate().map(|(i, (key, description))| OptionView {
                key, description, label: self.label_texts.get(i).map(String::as_str),
            }).collect(),
            images: vec![MEDIA_MARKER.to_string(); images.len()],
        })?;

        let pieces: Vec<&str> = text.split(MEDIA_MARKER).collect();
        ensure!(pieces.len() == images.len() + 1, "the template placed {} of {} images", pieces.len() - 1, images.len());
        let mut layout = Layout::default();
        for (i, piece) in pieces.iter().enumerate() {
            layout.text(&self.tok.encode(piece).into_iter().map(|t| t as u32).collect::<Vec<_>>());
            let Some(img) = images.get(i) else { continue };
            let v = self.vision.as_ref().context("this model takes no images")?;
            layout.text(&[v.start]);
            layout.image(i, v.pad, img.columns, img.grid_rows);
            layout.text(&[v.end]);
        }
        let Layout { ids, positions, images: spans, .. } = layout;
        if ids.is_empty() { bail!("questions.{}: the prompt is empty", q.id); }
        if ids.len() > self.max_seq {
            return Err(invalid(format!("questions.{}: the prompt is {} tokens, this model reads at most {}",
                q.id, ids.len(), self.max_seq)));
        }
        let last = ids.len() - 1;
        let reads = match &self.pointer {
            Some(p) => {
                let markers: Vec<usize> = (0..last).filter(|&i| ids[i] == p.marker).collect();
                ensure!(markers.len() == n, "unexpected layout of the decision prompt");
                markers.into_iter().chain([last]).collect()
            }
            None => vec![last],
        };
        Ok(Prompt { ids, positions: (!images.is_empty()).then_some(positions), images: spans, outputs: self.spec.outputs(q), reads })
    }

    /// Each prompt's scores, and how many prompt tokens were reused rather than
    /// computed.
    ///
    /// The prefix every prompt shares, up to the first row any prompt reads, is
    /// evaluated once, in slot 0. The prompts then continue from it a slot each,
    /// as many at a time as there are slots, in shared passes: slot 0 restores the
    /// prefix's recurrent state (the backend's saved state), and every other slot
    /// starts from a copy of the prefix's state and KV rows. The prefix is held for
    /// the next request, which continues from it when its own prompts begin with it;
    /// slot 0's prefix rows stay in place, since a prompt writes only the rows after
    /// them. A prompt holding an image is never continued from: its image rows carry
    /// the same placeholder id whatever the image.
    fn read(&self, prompts: &[Prompt], images: &[EncodedImage]) -> (Vec<Vec<f32>>, usize) {
        let first_read = prompts.iter().map(|p| p.reads[0]).min().unwrap_or(0);
        let shared = prompts[1..].iter().map(|p| ojas_tokenize::shared_prefix(&prompts[0].ids, &p.ids))
            .min().unwrap_or(prompts[0].ids.len()).min(first_read);
        let reusable = images.is_empty();
        let embedded = |p: &Prompt| -> Vec<(usize, &[f32])> {
            p.images.iter().map(|&(first, i)| (first, images[i].rows.as_slice())).collect()
        };
        let mut held = self.held.borrow_mut();
        let resumed = match held.take() {
            Some(h) if reusable && h.ids.len() <= shared && prompts[0].ids.starts_with(&h.ids) => {
                self.dec.restore_state();
                h.ids.len()
            }
            _ => {
                self.dec.reset_session();
                0
            }
        };
        if shared > resumed {
            let runs = embedded(&prompts[0]);
            let rows = PromptRows { ids: &prompts[0].ids, embedded: &runs, positions: prompts[0].positions.as_deref() };
            self.dec.prefill_hidden_slots(&[SlotPrefill { prompt: &rows, span: resumed..shared, slot: 0, read: &[] }]);
        }
        self.dec.save_state();
        if reusable && shared > 0 {
            *held = Some(HeldPrefix { ids: prompts[0].ids[..shared].to_vec() });
        }
        let reused = resumed + shared * (prompts.len() - 1);
        // The prompts continue from the prefix a slot each, as many at once as there
        // are slots: slot 0 holds the prefix (the first group finds it there), and
        // every other slot starts from a copy.
        let mut hidden: Vec<Vec<Vec<f32>>> = Vec::with_capacity(prompts.len());
        for (gi, group) in prompts.chunks(self.dec.slots()).enumerate() {
            if gi > 0 { self.dec.restore_state(); }
            for slot in 1..group.len() { self.dec.copy_slot_prefix(0, slot, shared); }
            let runs: Vec<Vec<(usize, &[f32])>> = group.iter().map(embedded).collect();
            let rows: Vec<PromptRows> = group.iter().zip(&runs)
                .map(|(p, runs)| PromptRows { ids: &p.ids, embedded: runs, positions: p.positions.as_deref() }).collect();
            let jobs: Vec<SlotPrefill> = group.iter().zip(&rows).enumerate()
                .map(|(slot, (p, rows))| SlotPrefill { prompt: rows, span: shared..p.ids.len(), slot, read: &p.reads }).collect();
            hidden.extend(self.dec.prefill_hidden_slots(&jobs));
        }
        (prompts.iter().zip(&hidden).map(|(p, h)| self.score_prompt(p, h)).collect(), reused)
    }

    /// One prompt's scores from the hidden states of its read rows.
    fn score_prompt(&self, p: &Prompt, hidden: &[Vec<f32>]) -> Vec<f32> {
        let (markers, last) = hidden.split_at(hidden.len() - 1);
        match &self.pointer {
            None => self.label_rows[..p.outputs].iter()
                .map(|row| row.iter().zip(&last[0]).map(|(w, h)| w * h).sum::<f32>()).collect(),
            Some(ptr) => {
                let half = ptr.half();
                let query = ptr.project(&last[0], 0..half);
                let scale = (half as f32).sqrt();
                markers.iter().map(|h| {
                    ptr.project(h, half..2 * half).iter().zip(&query).map(|(k, q)| k * q).sum::<f32>() / scale
                }).collect()
            }
        }
    }

    /// A request's prompts, a question at a time and an option order at a time.
    fn prompts(&self, state: &Json, questions: &[Question], images: &[EncodedImage]) -> Result<Vec<Prompt>> {
        let mut prompts = Vec::new();
        for q in questions {
            for v in 0..self.spec.variants(q) { prompts.push(self.prompt(state, q, v, images)?); }
        }
        Ok(prompts)
    }

    /// Scores in prompt order, regrouped per question and option order.
    fn regroup(&self, questions: &[Question], scores: Vec<Vec<f32>>) -> Vec<Vec<Vec<f32>>> {
        let mut scores = scores.into_iter();
        questions.iter().map(|q| (0..self.spec.variants(q)).map(|_| scores.next().unwrap()).collect()).collect()
    }

    /// The scores of several requests, each its own result.
    ///
    /// A request asked in one prompt shares GPU passes with up to `slots - 1` others,
    /// each in a slot of its own from an empty state, so a pass reads the weights once
    /// for all of them. A request asked in several prompts runs on its own, its
    /// prompts continuing their shared prefix (`read`). A pass's GPU time is shared
    /// among its requests by token count.
    pub(super) fn scores_many(&self, reqs: &[(&Json, &[Question], &[Image])]) -> Vec<Result<Scored>> {
        let built: Vec<Result<(Vec<EncodedImage>, Vec<Prompt>)>> = reqs.iter().map(|&(state, questions, images)| {
            let encoded = self.encode_images(images)?;
            let prompts = self.prompts(state, questions, &encoded)?;
            Ok((encoded, prompts))
        }).collect();
        let mut out: Vec<Option<Result<Scored>>> = built.iter().map(|_| None).collect();
        let mut singles = Vec::new();
        for (i, b) in built.iter().enumerate() {
            match b {
                Err(e) => out[i] = Some(Err(anyhow::anyhow!("{e:#}"))),
                Ok((_, prompts)) if prompts.len() == 1 => singles.push(i),
                Ok((encoded, prompts)) => {
                    let gpu_before = self.dec.gpu_seconds();
                    let (scores, reused) = self.read(prompts, encoded);
                    out[i] = Some(Ok(Scored {
                        scores: self.regroup(reqs[i].1, scores), tokens: prompts.iter().map(|p| p.ids.len()).sum(), reused,
                        gpu_s: self.dec.gpu_seconds() - gpu_before,
                    }));
                }
            }
        }
        if !singles.is_empty() {
            // The slots are about to be overwritten from row 0.
            self.held.borrow_mut().take();
        }
        for group in singles.chunks(self.dec.slots()) {
            let items: Vec<(&Prompt, &[EncodedImage])> = group.iter().map(|&i| {
                let Ok((encoded, prompts)) = &built[i] else { unreachable!("only built requests are grouped") };
                (&prompts[0], encoded.as_slice())
            }).collect();
            let runs: Vec<Vec<(usize, &[f32])>> = items.iter()
                .map(|(p, images)| p.images.iter().map(|&(first, k)| (first, images[k].rows.as_slice())).collect()).collect();
            let rows: Vec<PromptRows> = items.iter().zip(&runs)
                .map(|((p, _), runs)| PromptRows { ids: &p.ids, embedded: runs, positions: p.positions.as_deref() }).collect();
            let jobs: Vec<SlotPrefill> = items.iter().zip(&rows).enumerate()
                .map(|(slot, ((p, _), rows))| SlotPrefill { prompt: rows, span: 0..p.ids.len(), slot, read: &p.reads }).collect();
            for slot in 0..group.len() { self.dec.reset_slot(slot); }
            let gpu_before = self.dec.gpu_seconds();
            let hidden = self.dec.prefill_hidden_slots(&jobs);
            let gpu_s = self.dec.gpu_seconds() - gpu_before;
            let total: usize = items.iter().map(|(p, _)| p.ids.len()).sum();
            for ((&i, (p, _)), h) in group.iter().zip(&items).zip(&hidden) {
                let scores = vec![self.score_prompt(p, h)];
                out[i] = Some(Ok(Scored {
                    scores: self.regroup(reqs[i].1, scores), tokens: p.ids.len(), reused: 0,
                    gpu_s: gpu_s * p.ids.len() as f64 / total as f64,
                }));
            }
        }
        out.into_iter().map(|r| r.expect("every request answered")).collect()
    }
}

/// A JSON value as indented text, the form a text-only template was trained on:
/// strings as they are, `True`/`False`, nothing for `null`, an array as `- item`
/// lines and an object as `key: value` lines, nested values indented two spaces
/// further. Special-token spellings in the text are escaped ([`escape_specials`]).
pub(super) fn plain_text(j: &Json) -> String { escape_specials(&render_text(j, 0)) }

fn render_text(j: &Json, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match j {
        Json::Null => String::new(),
        Json::Str(s) => s.clone(),
        Json::Bool(b) => if *b { "True" } else { "False" }.into(),
        Json::Array(items) => items.iter().map(|v| {
            format!("{pad}- {}", render_text(v, indent + 1).trim_start_matches([' ', '\t', '\n', '\r']))
        }).collect::<Vec<_>>().join("\n"),
        Json::Object(kv) => kv.iter().map(|(k, v)| {
            let nested = matches!(v, Json::Array(_) | Json::Object(_));
            if nested { format!("{pad}{k}:\n{}", render_text(v, indent + 1)) }
            else { format!("{pad}{k}: {}", render_text(v, 0)) }
        }).collect::<Vec<_>>().join("\n"),
        number => number.to_python(false),
    }
}

/// `<|name|>` written as `<¦name¦>`, for a name of ASCII letters, digits and `_`, so
/// text a request carries cannot be read as a special token.
pub(super) fn escape_specials(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(at) = rest.find("<|") {
        out.push_str(&rest[..at]);
        let after = &rest[at + 2..];
        let name_len = after.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(after.len());
        if name_len > 0 && after[name_len..].starts_with("|>") {
            out.push_str("<¦");
            out.push_str(&after[..name_len]);
            out.push_str("¦>");
            rest = &after[name_len + 2..];
        } else {
            out.push_str("<|");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_flattens_json_as_the_text_templates_expect() {
        let j = Json::parse(r#"{"user": "Ann", "vip": true, "note": null, "n": 3,
            "items": ["a", {"sku": "x1", "qty": 2}], "meta": {"tags": []}}"#).unwrap();
        assert_eq!(plain_text(&j), "user: Ann\nvip: True\nnote: \nn: 3\nitems:\n  - a\n  - sku: x1\n    qty: 2\nmeta:\n  tags:\n");
    }

    #[test]
    fn image_rows_take_grid_positions_and_text_resumes_past_the_grid() {
        let mut layout = Layout::default();
        layout.text(&[10, 11]);
        layout.image(0, 99, 3, 2);
        layout.text(&[12]);
        assert_eq!(layout.ids, vec![10, 11, 99, 99, 99, 99, 99, 99, 12]);
        assert_eq!(layout.images, vec![(2, 0)]);
        assert_eq!(layout.positions, vec![
            [0, 0, 0, 0], [1, 1, 1, 0],
            [2, 2, 2, 0], [2, 2, 3, 0], [2, 2, 4, 0], [2, 3, 2, 0], [2, 3, 3, 0], [2, 3, 4, 0],
            [5, 5, 5, 0],
        ]);
    }

    #[test]
    fn special_token_spellings_are_escaped() {
        assert_eq!(escape_specials("a <|box_end|> b <|not one|> <||> <|x"), "a <¦box_end¦> b <|not one|> <||> <|x");
    }
}
