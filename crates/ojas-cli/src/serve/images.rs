//! Images in chat requests: `image_url` content parts on `/v1/chat/completions`.
//!
//! A part is a `data:image/...;base64,...` URL (or a Messages API `image` block
//! with a base64 source); a remote URL is refused, never fetched. Each image is
//! decoded and resized here, on the thread that reads the request, to at most the
//! request's token budget, and the model's own vision tower encodes it when prompt
//! processing reaches it (`ojas_infer::image`). This is the pipeline `ojas ocr`
//! runs: the same preprocessor and budget, the same markers resolved through the
//! model's vocabulary, the same anchored rotary layout, extended to several images
//! among several turns.
//!
//! The prefix cache sees an image prompt by key, not by id. An image's rows carry
//! a hash of its decoded pixels and of the budget that sized them, and every row
//! placed off its cache position carries that offset, so two images never share a
//! cached prefix, the same image in a later request finds its own, and a prefix is
//! only reused where it would be computed to the same values.

use super::completion::Ctx;
use crate::backend::ModelInfo;
use crate::flags::RunOpts;
use crate::ocr::{resolve_mmproj, span_positions, text_positions, token_budget, PosLayout, VisionTokens};
use anyhow::{bail, ensure, Context, Result};
use ojas_core::Model;
use ojas_cpu::VitPreproc;
use ojas_decision::{Image, MAX_IMAGES};
use ojas_formats::gguf::{Gguf, Meta};
use ojas_infer::image::{PromptImage, PromptImages};
use ojas_tokenize::Bpe;
use serde_json::Value;

/// Stands for image `k` while the chat template is rendered: this, `k`, then
/// [`SPAN_END`]. NUL bytes cannot tokenize into anything, and a request whose text
/// holds the marker is refused.
const SPAN: &str = "\u{0}\u{0}ojas-image:";
const SPAN_END: &str = "\u{0}\u{0}";

/// Set on the key of every row whose token id does not say what it holds; no
/// vocabulary reaches it, so such a key never equals a token id.
const KEY_BIT: u32 = 1 << 31;

/// What the server needs to take images: the projector's geometry, the markers an
/// image sits between, and the budget.
pub(crate) struct Vision {
    /// Resize policy, with the projector's patch, merge and normalisation.
    pre: VitPreproc,
    tokens: VisionTokens,
    /// The text of the tokens before and after an image's rows.
    open: String,
    close: String,
    /// Fewest and most rows one image may take. A request may lower the most.
    min_tokens: i32,
    pub(crate) max_tokens: i32,
    /// Whether the model places rows by rotary coordinate.
    mrope: bool,
}

impl Vision {
    /// What `model` offers for images, or why it takes none: it needs its vision
    /// tower (a projector beside the model, `--mmproj` or `OJAS_MMPROJ`) and a path
    /// for image rows beside text rows.
    pub(crate) fn open(path: &str, model: &dyn Model, bpe: &Bpe, info: &ModelInfo, opts: &RunOpts) -> Result<Vision, String> {
        let Some(width) = model.vision_width() else {
            return Err("this server has no vision projector: start it with --mmproj <mmproj.gguf> (or OJAS_MMPROJ) to send images".into());
        };
        let d = info.hidden_dim;
        let probe = |pos3: Option<&[[u32; 4]]>| {
            let taken = model.prefill_rows(&[0], &[KEY_BIT], 0, &[(0, &vec![0.0f32; d])], pos3);
            model.reset_session();
            taken
        };
        if width != d || !probe(None) {
            return Err(format!("the {} backend for {} cannot place image rows in a prompt", info.backend, info.arch));
        }
        let mrope = probe(Some(&[[0; 4]]));
        (|| -> Result<Vision> {
            let mm = Gguf::open(resolve_mmproj(path)?.to_string_lossy().as_ref())?;
            let base = VitPreproc::qwen3vl();
            let floats = |k: &str| match mm.meta.get(k) {
                Some(Meta::FloatArr(v)) if v.len() == 3 => Some([v[0], v[1], v[2]]),
                _ => None,
            };
            let patch_size = mm.meta_u32("clip.vision.patch_size").context("the projector has no clip.vision.patch_size")? as i32;
            ensure!(patch_size > 0, "the projector declares patch_size 0");
            let pre = VitPreproc {
                patch_size, n_merge: mm.meta_u32("clip.vision.spatial_merge_size").unwrap_or(1).max(1) as i32,
                mean: floats("clip.vision.image_mean").unwrap_or(base.mean),
                std: floats("clip.vision.image_std").unwrap_or(base.std),
                ..base
            };
            let template = match Gguf::open(path)?.meta.get("tokenizer.chat_template") {
                Some(Meta::Str(t)) => Some(t.clone()),
                _ => None,
            };
            let (tokens, open, close) = markers(template.as_deref(), bpe)?;
            // The budget `ojas ocr` reads, capped at half the context unless the
            // flag sets it, so one image always leaves room for the conversation.
            let (min_tokens, env_max) = token_budget();
            let max_tokens = match opts.image_max_tokens {
                Some(n) => n.clamp(1, i32::MAX as usize) as i32,
                None => env_max.min((info.context / 2).max(1) as i32),
            };
            Ok(Vision { pre, tokens, open, close, min_tokens: min_tokens.min(max_tokens), max_tokens, mrope })
        })().map_err(|e| format!("this server cannot take images: {e:#}"))
    }

    /// The most rows one image of this request may take: the server's, lowered by
    /// the request's `image_max_tokens` or `image_max_pixels`.
    fn budget(&self, body: &Value) -> Result<i32> {
        let field = |k: &str| -> Result<Option<u64>> {
            match body.get(k) {
                None | Some(Value::Null) => Ok(None),
                Some(v) => Ok(Some(v.as_u64().filter(|&n| n > 0).with_context(|| format!("\"{k}\" must be a positive integer"))?)),
            }
        };
        let pixels = field("image_max_pixels")?.map(|p| (p / self.pre.patch_area() as u64).max(1));
        let asked = [field("image_max_tokens")?, pixels].into_iter().flatten().min();
        Ok(asked.map_or(self.max_tokens, |n| n.min(self.max_tokens as u64) as i32))
    }

    /// The merged token grid `(columns, rows)` of a preprocessed image.
    fn grid(&self, width: usize, height: usize) -> (usize, usize) {
        let side = (self.pre.patch_size * self.pre.n_merge) as usize;
        (width / side, height / side)
    }
}

/// The markers an image's rows sit between and the placeholder they carry, with
/// the text of the first and last. The model's chat template names them where it
/// writes three adjacent special tokens for an image; a model without a template,
/// or whose template writes none, uses the vision tokens of its vocabulary.
fn markers(template: Option<&str>, bpe: &Bpe) -> Result<(VisionTokens, String, String)> {
    let single = |s: &str| match bpe.encode(s).as_slice() {
        [id] if bpe.is_control(*id) => Some(*id as u32),
        _ => None,
    };
    for [open, pad, close] in template.map(special_triples).unwrap_or_default() {
        if let (Some(start), Some(pad), Some(end)) = (single(open), single(pad), single(close)) {
            return Ok((VisionTokens { start, pad, end }, open.into(), close.into()));
        }
    }
    let tokens = VisionTokens::resolve(bpe)?;
    Ok((tokens, bpe.decode(tokens.start as usize), bpe.decode(tokens.end as usize)))
}

/// Every run of exactly three adjacent `<|name|>` tokens in a chat template, those
/// whose placeholder names an image first.
fn special_triples(template: &str) -> Vec<[&str; 3]> {
    let token = |s: &str| -> Option<usize> {
        let body = s.strip_prefix("<|")?;
        let end = body.find("|>")?;
        body[..end].bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_').then_some(end + 4)
    };
    let (mut found, mut i) = (Vec::new(), 0);
    while let Some(at) = template[i..].find("<|").map(|k| i + k) {
        let mut run = Vec::new();
        let mut end = at;
        while let Some(len) = token(&template[end..]) {
            run.push(&template[end..end + len]);
            end += len;
        }
        if let [a, b, c] = run[..] { found.push([a, b, c]); }
        i = end.max(at + 2);
    }
    found.sort_by_key(|t| !t[1].contains("image"));
    found
}

/// One part of a message's content.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Part<'a> {
    Text(&'a str),
    /// A `data:` URL holding an image.
    Image(String),
}

/// A message's content as parts: a string, or an array of `text`, `image_url` and
/// `image` parts. An image that is not inline data is an error, as is a part of
/// any other type: neither is dropped.
pub(crate) fn parts(content: Option<&Value>) -> Result<Vec<Part<'_>>> {
    let remote = "images must be data URLs (data:image/...;base64,...); remote URLs are not fetched";
    match content {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![Part::Text(s)]),
        Some(Value::Array(items)) => items.iter().map(|item| {
            match item.get("type").and_then(Value::as_str) {
                Some("text") => Ok(Part::Text(item.get("text").and_then(Value::as_str).context("a text part needs \"text\"")?)),
                Some("image_url") => {
                    let url = item.get("image_url").map(|u| u.get("url").unwrap_or(u)).and_then(Value::as_str)
                        .context("an image_url part needs image_url.url")?;
                    ensure!(url.starts_with("data:"), "{remote}");
                    Ok(Part::Image(url.to_string()))
                }
                Some("image") => {
                    let source = item.get("source").context("an image part needs \"source\"")?;
                    ensure!(source.get("type").and_then(Value::as_str) == Some("base64"), "{remote}");
                    let data = source.get("data").and_then(Value::as_str).context("an image source needs \"data\"")?;
                    let media = source.get("media_type").and_then(Value::as_str).unwrap_or("image/png");
                    Ok(Part::Image(format!("data:{media};base64,{data}")))
                }
                Some(other) => bail!("unsupported content part type \"{other}\""),
                None => bail!("a content part needs a \"type\""),
            }
        }).collect(),
        Some(_) => bail!("message content must be a string or an array of parts"),
    }
}

/// The text of a message's content, its text parts joined. For the paths that
/// take no images: an image part contributes nothing, and a malformed one is the
/// completion request's to refuse.
pub(crate) fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        other => parts(other).unwrap_or_default().iter()
            .map(|p| if let Part::Text(t) = p { t } else { "" }).collect(),
    }
}

/// Whether any message of `body` carries an image part.
pub(crate) fn has_images(body: &Value) -> bool {
    body.get("messages").and_then(Value::as_array).is_some_and(|msgs| msgs.iter().any(|m| {
        m.get("content").and_then(Value::as_array).is_some_and(|items| items.iter().any(|i| {
            matches!(i.get("type").and_then(Value::as_str), Some("image_url" | "image"))
        }))
    }))
}

/// A hash of an image as the model will see it: its decoded pixels and size, and
/// the token budget that sizes them. Two words, mixed apart, so the rows of one
/// image carry 128 bits between them.
fn image_key(image: &Image, min_tokens: i32, max_tokens: i32) -> [u64; 2] {
    let mut key = [0x6f6a_6173_2d69_6d67u64, 0x243f_6a88_85a3_08d3];
    let mut absorb = |word: u64| {
        key[0] = mix(key[0] ^ word);
        key[1] = mix(key[1].rotate_left(23) ^ word.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    };
    for v in [image.width as u64, image.height as u64, min_tokens as u64, max_tokens as u64] { absorb(v); }
    let (words, rest) = image.rgb.as_chunks::<8>();
    for w in words { absorb(u64::from_le_bytes(*w)); }
    let mut last = [0u8; 8];
    last[..rest.len()].copy_from_slice(rest);
    absorb(u64::from_le_bytes(last) ^ (rest.len() as u64) << 56);
    key
}

/// A 64-bit finalizer (splitmix64): every input bit reaches every output bit.
fn mix(z: u64) -> u64 {
    let z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// An image in the prompt: its first row, its token grid and its content key.
#[derive(Clone, Copy, Debug)]
struct Span {
    at: usize,
    columns: usize,
    rows: usize,
    key: [u64; 2],
}

impl Span {
    fn len(&self) -> usize { self.columns * self.rows }
    /// Positions an image's rows hold beyond those its coordinates advance by:
    /// a grid advances the sequence by its longer side.
    fn excess(&self) -> usize { self.len() - self.columns.max(self.rows).min(self.len()) }
}

/// A rotary coordinate for each of `n` prompt rows holding `spans` (ascending).
///
/// The rows after the last image, and every generated token, sit at their cache
/// row: that is the only position the decode loop can express. Everything before
/// is shifted up to meet them, as `ojas ocr` anchors a page: a row is offset by
/// the excess of every image at or after it, so each image is a grid whose
/// following text resumes `max(columns, rows)` positions on, and every distance
/// in the prompt is the one the model was trained with.
fn positions(n: usize, spans: &[Span]) -> Vec<[u32; 4]> {
    let mut shift: usize = spans.iter().map(Span::excess).sum();
    let mut out = Vec::with_capacity(n);
    for s in spans {
        out.extend(text_positions((out.len() + shift) as u32, s.at - out.len()));
        out.extend(span_positions(s.at + shift, s.columns, s.rows, PosLayout::Grid).img.unwrap_or_default());
        shift -= s.excess();
    }
    out.extend(text_positions(out.len() as u32, n - out.len()));
    out
}

/// The prefix-cache key of every prompt row. A text row at its own cache position
/// is its token id, so text before any image is shared with prompts of text. An
/// image row is drawn from the image's key and its index, and any row placed off
/// its cache position also from that offset, all outside the vocabulary.
fn keys(ids: &[u32], spans: &[Span], pos3: Option<&[[u32; 4]]>) -> Vec<u32> {
    let offset = |i: usize| pos3.map_or(0, |p| (p[i][0] as u64).wrapping_sub(i as u64));
    let mut keys: Vec<u32> = ids.iter().enumerate().map(|(i, &t)| match offset(i) {
        0 => t,
        off => KEY_BIT | mix(t as u64 ^ off << 32) as u32,
    }).collect();
    for s in spans {
        // Every row of an image shares its first coordinate, so the offset of the
        // first row places them all.
        let off = offset(s.at);
        for r in 0..s.len() {
            keys[s.at + r] = KEY_BIT | mix(s.key[r % 2] ^ mix(r as u64 ^ off << 32)) as u32;
        }
    }
    keys
}

/// Encode rendered text holding image markers: each piece of text on its own, and
/// `rows[k]` placeholder ids where image `k`'s marker stood. Returns the ids and
/// each image's `(first row, k)`. The pieces split at special tokens, which the
/// tokenizer splits at too, so encoding them apart loses no merge.
fn splice(text: &str, pad: u32, rows: &[usize], encode: &dyn Fn(&str) -> Vec<u32>) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
    let mut pieces = text.split(SPAN);
    let mut ids = encode(pieces.next().unwrap_or(""));
    let mut placed = Vec::new();
    for piece in pieces {
        let (k, rest) = piece.split_once(SPAN_END).context("the chat template broke an image marker")?;
        let k: usize = k.parse().ok().filter(|&k| k < rows.len()).context("the chat template broke an image marker")?;
        placed.push((ids.len(), k));
        ids.resize(ids.len() + rows[k], pad);
        ids.extend(encode(rest));
    }
    Ok((ids, placed))
}

/// A chat request's prompt when it carries images.
pub(crate) struct Built {
    pub(crate) ids: Vec<u32>,
    /// Message boundaries, for the prefix cache.
    pub(crate) marks: Vec<usize>,
    pub(crate) images: PromptImages,
}

/// The prompt of a chat request with image parts; `None` for one without, which
/// the text path serves once its content parts are known to be well formed.
pub(crate) fn build(ctx: &Ctx, body: &Value) -> Result<Option<Built>> {
    let Some(msgs) = body.get("messages").and_then(Value::as_array) else { return Ok(None) };
    if !has_images(body) {
        for m in msgs { parts(m.get("content"))?; }
        return Ok(None);
    }
    let vision = ctx.images.as_ref().map_err(|why| anyhow::anyhow!("{why}"))?;
    let budget = vision.budget(body)?;
    let pre = vision.pre.with_token_budget(vision.min_tokens.min(budget), budget);

    // Each message's text, an indexed marker between the model's own where an
    // image stands; the images decoded and resized to the budget.
    let (mut system, mut turns) = (ctx.opts.system.clone(), Vec::new());
    let mut decoded: Vec<(PromptImage, Span)> = Vec::new();
    for m in msgs {
        let mut text = String::new();
        for part in parts(m.get("content"))? {
            match part {
                Part::Text(t) => {
                    ensure!(!t.contains(SPAN), "message text holds the internal image marker");
                    text.push_str(t);
                }
                Part::Image(url) => {
                    ensure!(decoded.len() < MAX_IMAGES, "too many images, the maximum is {MAX_IMAGES}");
                    let image = Image::from_data_url(&url)?;
                    let (w, h, pixels) = pre.preprocess(&image.rgb, image.width, image.height)?;
                    let (columns, rows) = vision.grid(w, h);
                    ensure!(columns * rows > 0, "an image of {}x{} is too small to encode", image.width, image.height);
                    text.push_str(&format!("{}{SPAN}{}{SPAN_END}{}", vision.open, decoded.len(), vision.close));
                    let key = image_key(&image, pre.min_tokens, pre.max_tokens);
                    decoded.push((PromptImage::new(0, columns * rows, pixels, w, h), Span { at: 0, columns, rows, key }));
                }
            }
        }
        match m.get("role").and_then(Value::as_str).unwrap_or("user") {
            "system" => system = text,
            role => turns.push((role.to_string(), text)),
        }
    }

    let rows: Vec<usize> = decoded.iter().map(|(_, s)| s.len()).collect();
    let encode = |text: &str| -> Vec<u32> { ctx.bpe.encode(text).into_iter().map(|v| v as u32).collect() };
    let arch = &ctx.info.arch;
    let (ids, placed) = splice(&ojas_tokenize::chat_transcript(arch, &system, &turns), vision.tokens.pad, &rows, &encode)?;
    ensure!(placed.len() == decoded.len(), "the chat template placed {} of {} images", placed.len(), decoded.len());
    for &(at, k) in &placed {
        // An image the template left without its markers would read as text.
        ensure!(at > 0 && ids[at - 1] == vision.tokens.start && ids.get(at + rows[k]) == Some(&vision.tokens.end),
            "the chat template did not keep an image between its markers");
        (decoded[k].0.at, decoded[k].1.at) = (at, at);
    }
    decoded.sort_by_key(|(_, s)| s.at);
    let (images, spans): (Vec<PromptImage>, Vec<Span>) = decoded.into_iter().unzip();

    let marks = if !ctx.prefix_cache { Vec::new() } else {
        ojas_tokenize::transcript_boundaries(arch, &system, &turns, super::MARKED_TURNS,
            |text| splice(text, vision.tokens.pad, &rows, &encode).map(|(ids, _)| ids).unwrap_or_default())
    };
    let pos3 = vision.mrope.then(|| positions(ids.len(), &spans));
    let keys = keys(&ids, &spans, pos3.as_deref());
    let until = spans.last().map_or(0, |s| s.at + s.len());
    Ok(Some(Built { ids, marks, images: PromptImages { keys, pos3, until, images } }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A 2x1 PNG: one red pixel, one blue.
    const PNG_2X1: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGP4zwAE/wEHAAH/4iOeWQAAAABJRU5ErkJggg==";

    fn vision(max_tokens: i32) -> Vision {
        Vision {
            pre: VitPreproc::qwen3vl(), tokens: VisionTokens { start: 9, pad: 11, end: 10 },
            open: "<|vision_start|>".into(), close: "<|vision_end|>".into(), min_tokens: 8, max_tokens, mrope: true,
        }
    }

    fn span(at: usize, columns: usize, rows: usize, key: u64) -> Span { Span { at, columns, rows, key: [key, !key] } }

    #[test]
    fn content_is_a_string_or_text_and_image_parts() {
        let url = format!("data:image/png;base64,{PNG_2X1}");
        assert_eq!(parts(Some(&json!("hi"))).unwrap(), vec![Part::Text("hi")]);
        assert!(parts(None).unwrap().is_empty() && parts(Some(&Value::Null)).unwrap().is_empty());
        let content = json!([{"type": "text", "text": "what is "}, {"type": "image_url", "image_url": {"url": url}},
                             {"type": "text", "text": "this?"}]);
        assert_eq!(parts(Some(&content)).unwrap(), vec![Part::Text("what is "), Part::Image(url.clone()), Part::Text("this?")]);
        assert_eq!(content_text(Some(&content)), "what is this?");
        // The Messages API's block for the same image, and a bare URL string.
        let block = json!([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_2X1}},
                           {"type": "image_url", "image_url": url}]);
        assert_eq!(parts(Some(&block)).unwrap(), vec![Part::Image(url.clone()), Part::Image(url)]);
    }

    #[test]
    fn remote_urls_and_unknown_parts_are_refused_not_dropped() {
        for bad in [json!([{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]),
                    json!([{"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}]),
                    json!([{"type": "input_audio", "input_audio": {}}]), json!([{"text": "untyped"}]),
                    json!([{"type": "text"}]), json!([{"type": "image_url"}]), json!(3)] {
            assert!(parts(Some(&bad)).is_err(), "{bad}");
        }
        let remote = parts(Some(&json!([{"type": "image_url", "image_url": {"url": "http://host/a.png"}}]))).unwrap_err();
        assert!(remote.to_string().contains("remote URLs are not fetched"), "{remote}");
    }

    #[test]
    fn an_image_without_a_projector_is_refused_and_text_parts_pass() {
        let (bpe, opts) = (Bpe::default(), RunOpts::default());
        let info = ModelInfo {
            arch: "qwen3".into(), eos: Some(2), eog: vec![2], vocab: 16, context: 64, backend: "cpu",
            has_mtp: false, n_layers: 1, hidden_dim: 1, load_secs: 0.0,
        };
        let vocab = std::sync::Arc::new(ojas_grammar::TokenVocab::new(Vec::new(), Vec::new(), &[]));
        let none = Err("this server has no vision projector".to_string());
        let ctx = Ctx { bpe: &bpe, info: &info, opts: &opts, vocab: &vocab, primary: None, secondary: None,
                        prefix_cache: false, images: &none };
        let ask = |content: Value| build(&ctx, &json!({"messages": [{"role": "user", "content": content}]}));
        let url = format!("data:image/png;base64,{PNG_2X1}");
        let refused = ask(json!([{"type": "image_url", "image_url": {"url": url}}])).err().expect("no projector");
        assert!(refused.to_string().contains("no vision projector"), "{refused}");
        assert!(ask(json!([{"type": "text", "text": "hi"}])).unwrap().is_none(), "text parts go the text way");
        assert!(ask(json!("hi")).unwrap().is_none());
        assert!(ask(json!([{"type": "input_audio"}])).is_err(), "a part of an unknown type is not dropped");
        assert!(build(&ctx, &json!({"prompt": "hi"})).unwrap().is_none());
    }

    #[test]
    fn a_request_has_images_when_any_message_holds_an_image_part() {
        let url = format!("data:image/png;base64,{PNG_2X1}");
        assert!(has_images(&json!({"messages": [{"role": "user", "content": "a"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]})));
        assert!(has_images(&json!({"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]})));
        assert!(!has_images(&json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "image_url"}]}]})));
        assert!(!has_images(&json!({"prompt": "image_url"})));
    }

    #[test]
    fn data_urls_decode_and_anything_else_does_not() {
        let image = Image::from_data_url(&format!("data:image/png;base64,{PNG_2X1}")).unwrap();
        assert_eq!((image.width, image.height, image.rgb), (2, 1, vec![255, 0, 0, 0, 0, 255]));
        // What a Messages API block becomes decodes the same way.
        let block = json!([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_2X1}}]);
        let Part::Image(url) = &parts(Some(&block)).unwrap()[0] else { panic!("an image part") };
        assert_eq!(Image::from_data_url(url).unwrap().rgb.len(), 6);
        for bad in ["data:image/png;base64,!!!!", "data:image/png;base64,aGVsbG8=", "data:text/plain;base64,aGk=", "data:image/png,raw"] {
            assert!(Image::from_data_url(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_request_lowers_the_budget_but_never_raises_it() {
        let v = vision(1024);
        assert_eq!(v.budget(&json!({})).unwrap(), 1024);
        assert_eq!(v.budget(&json!({"image_max_tokens": 256})).unwrap(), 256);
        assert_eq!(v.budget(&json!({"image_max_tokens": 100000})).unwrap(), 1024, "the server's cap holds");
        // A pixel cap is a token cap at the projector's pixels per token (32x32).
        assert_eq!(v.budget(&json!({"image_max_pixels": 64 * 1024})).unwrap(), 64);
        assert_eq!(v.budget(&json!({"image_max_pixels": 10})).unwrap(), 1);
        assert_eq!(v.budget(&json!({"image_max_tokens": 512, "image_max_pixels": 128 * 1024})).unwrap(), 128, "the tighter of two");
        for bad in [json!({"image_max_tokens": 0}), json!({"image_max_tokens": -4}), json!({"image_max_tokens": "many"}),
                    json!({"image_max_pixels": 1.5})] {
            assert!(v.budget(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_budget_bounds_the_rows_an_image_takes() {
        let v = vision(4096);
        let rgb = vec![128u8; 1920 * 1080 * 3];
        for budget in [64, 256, 1024] {
            let (w, h, pixels) = v.pre.with_token_budget(8, budget).preprocess(&rgb, 1920, 1080).unwrap();
            let (columns, rows) = v.grid(w, h);
            assert!(columns * rows <= budget as usize && columns * rows > budget as usize / 2, "{columns}x{rows} for {budget}");
            assert_eq!((pixels.len(), w % 32, h % 32), (w * h * 3, 0, 0));
        }
    }

    #[test]
    fn the_key_is_the_pixels_their_size_and_the_budget() {
        let image = |rgb: Vec<u8>, width, height| Image { width, height, rgb };
        let a = image(vec![7; 24], 4, 2);
        let key = image_key(&a, 8, 256);
        assert_eq!(key, image_key(&a.clone(), 8, 256), "the same image under the same budget");
        let mut other = a.clone();
        other.rgb[23] ^= 1;
        assert_ne!(key, image_key(&other, 8, 256), "one bit of one pixel");
        assert_ne!(key, image_key(&image(vec![7; 24], 2, 4), 8, 256), "the same bytes in another shape");
        assert_ne!(key, image_key(&a, 8, 128), "another budget");
        // A tail shorter than a word still counts, and by its length.
        assert_ne!(image_key(&image(vec![0; 3], 1, 1), 8, 256), image_key(&image(vec![0; 6], 2, 1), 8, 256));
        assert_ne!(key[0], key[1]);
    }

    #[test]
    fn one_image_is_laid_out_as_the_page_layout_lays_it() {
        // Five rows of text, a 4x3 image, three rows of text: the image's rows hold
        // 12 positions and advance the sequence by 4, so everything through it sits
        // 8 further on and the text after it sits at its cache row.
        let spans = [span(5, 4, 3, 1)];
        let p = positions(20, &spans);
        assert_eq!(p[0], [8, 8, 8, 0]);
        assert_eq!(p[4], [12, 12, 12, 0]);
        assert_eq!((p[5], p[6], p[9], p[16]), ([13, 13, 13, 0], [13, 13, 14, 0], [13, 14, 13, 0], [13, 15, 16, 0]));
        assert_eq!((p[17], p[19]), ([17, 17, 17, 0], [19, 19, 19, 0]), "13 + max(4, 3) is the next cache row");
        // The same as the single-page layout's own answer.
        let page = span_positions(5, 4, 3, PosLayout::Anchored);
        assert_eq!((p[5..17].to_vec(), p[0][0]), (page.img.unwrap(), page.shift));
    }

    #[test]
    fn each_image_shifts_what_comes_before_it() {
        // text 0..2, a 2x2 image (excess 2) at 2..6, text 6..8, a 3x3 (excess 6) at 8..17, text 17..19.
        let spans = [span(2, 2, 2, 1), span(8, 3, 3, 2)];
        let p = positions(19, &spans);
        assert_eq!(p[0], [8, 8, 8, 0], "before both: offset by both");
        assert_eq!((p[2], p[5]), ([10, 10, 10, 0], [10, 11, 11, 0]));
        assert_eq!(p[6], [12, 12, 12, 0], "the first image advanced the sequence by its side, 2");
        assert_eq!((p[8], p[16]), ([14, 14, 14, 0], [14, 16, 16, 0]));
        assert_eq!(p[17], [17, 17, 17, 0], "14 + 3: the tail is at its cache row");
        assert_eq!(p.len(), 19);
        assert_eq!(positions(4, &[]), text_positions(0, 4));
    }

    #[test]
    fn keys_tell_images_apart_and_find_the_same_one_again() {
        let ids: Vec<u32> = (0..20).map(|i| if (5..17).contains(&i) { 11 } else { 100 + i }).collect();
        let key_of = |key: u64, mrope: bool| {
            let spans = [span(5, 4, 3, key)];
            let p = positions(20, &spans);
            keys(&ids, &spans, mrope.then_some(&p[..]))
        };
        let a = key_of(1, true);
        assert_eq!(a, key_of(1, true), "the same image keys the same rows");
        let b = key_of(2, true);
        assert!((5..17).all(|i| a[i] != b[i] && a[i] & KEY_BIT != 0), "every row of another image differs, off the vocabulary");
        assert_eq!((&a[..5], &a[17..]), (&b[..5], &b[17..]), "the text around them does not");
        // Shifted text is not the text a prompt without the image caches; the tail,
        // at its cache rows, is its ids.
        assert!((0..5).all(|i| a[i] != ids[i] && a[i] & KEY_BIT != 0));
        assert_eq!(a[17..], ids[17..]);
        // Rows of one image differ from each other, though they share a placeholder.
        assert!((6..17).all(|i| a[i] != a[5]));
        // Without coordinates nothing is shifted: text keeps its ids, images their keys.
        let flat = key_of(1, false);
        assert_eq!((&flat[..5], &flat[17..]), (&ids[..5], &ids[17..]));
        assert!((5..17).all(|i| flat[i] & KEY_BIT != 0 && flat[i] != a[i]), "and a row at another coordinate is another key");
    }

    #[test]
    fn an_image_at_another_offset_is_another_key() {
        // The same image and leading text, with and without a second image after:
        // every row before the second image moves, so none may be shared.
        let ids = vec![11u32; 19];
        let (one, two) = ([span(2, 2, 2, 1)], [span(2, 2, 2, 1), span(8, 3, 3, 2)]);
        let (p1, p2) = (positions(19, &one), positions(19, &two));
        let (k1, k2) = (keys(&ids, &one, Some(&p1)), keys(&ids, &two, Some(&p2)));
        assert!((0..6).all(|i| k1[i] != k2[i]));
    }

    #[test]
    fn markers_are_replaced_by_placeholder_rows() {
        // One id per byte, so positions are easy to read.
        let encode = |t: &str| t.bytes().map(u32::from).collect::<Vec<u32>>();
        let text = format!("ab{SPAN}1{SPAN_END}c{SPAN}0{SPAN_END}");
        let (ids, placed) = splice(&text, 7, &[3, 2], &encode).unwrap();
        assert_eq!(ids, vec![97, 98, 7, 7, 99, 7, 7, 7]);
        assert_eq!(placed, vec![(2, 1), (5, 0)]);
        assert_eq!(splice("plain", 7, &[], &encode).unwrap(), (encode("plain"), Vec::new()));
        for broken in [format!("a{SPAN}2{SPAN_END}"), format!("a{SPAN}x{SPAN_END}"), format!("a{SPAN}0")] {
            assert!(splice(&broken, 7, &[3, 2], &encode).is_err(), "{broken:?}");
        }
    }

    #[test]
    fn a_template_names_its_image_markers() {
        let template = "{% if content.type == 'video' %}<|vision_start|><|video_pad|><|vision_end|>{% endif %}\
            {{ '<|im_start|>' + role }}{% if 'image' in content %}Picture: <|vision_start|><|image_pad|><|vision_end|>{% endif %}\
            <|a|><|b|><|c|><|d|> <|im_end|>";
        assert_eq!(special_triples(template), vec![["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"],
                                                   ["<|vision_start|>", "<|video_pad|>", "<|vision_end|>"]]);
        assert!(special_triples("<|im_start|>user\n{{ content }}<|im_end|> <| not a token |>").is_empty());
    }
}
