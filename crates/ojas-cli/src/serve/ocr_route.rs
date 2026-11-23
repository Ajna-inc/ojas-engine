//! `POST /v1/ocr`: a page image in, its lines with boxes and confidences out.
//!
//! The request is `{"image": "<data URL or bare base64>", "max_tokens": N,
//! "return": ["text", "lines", "html"]}`, or the bytes of a PNG or JPEG as the
//! whole body. It becomes a chat completion of one user turn, the page then the
//! instruction `ojas ocr` gives, decoded greedily under the same repetition guard,
//! so it takes a slot, a timeout and its place in the metrics as any completion
//! does (`completion::start`). The reply is the model's layout markup read back:
//!
//! ```json
//! {"text": "...", "lines": [{"text": "...", "box": [x0, y0, x1, y1], "confidence": 0.98}],
//!  "width": W, "height": H, "finish_reason": "stop", "usage": {...}}
//! ```
//!
//! `box` is in pixels of the image as sent (after its EXIF orientation), `width`
//! by `height`. The model boxes blocks, not lines: a block of several lines gives
//! each an equal horizontal band of the block's box, which is exact for one line
//! and an estimate for more.
//!
//! `confidence` is the mean probability of the tokens that wrote the line: every
//! token from the line's first character to its last, inline markup between them
//! included, the block's tag and box excluded. A token's probability is the
//! model's own for it among the whole vocabulary, before sampling
//! (`super::logprobs`). It says how sure the model was of the characters, not
//! where they are. `null` when no step of the line was observed.
//!
//! Whether a model can do this is decided by what it has and what it writes: a
//! server without a vision projector answers 501 before generating, and so does
//! one whose output for the page carries no block this route can read.

use super::completion::Ctx;
use super::{metrics, send_err, Conn};
use crate::ocr::{loop_period, DEFAULT_INSTRUCTION, LOOP_MAX_PERIOD, LOOP_MIN_RUN};
use ojas_decision::Image;
use serde_json::{json, Value};
use std::ops::Range;

/// Marks a completion body built from an OCR request, and holds what its reply needs.
const MARK: &str = "__ocr";

/// Most bytes of an encoded image; a larger one is answered 413.
const MAX_IMAGE: usize = 20 * 1024 * 1024;

/// The parts of a reply a request asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Want {
    text: bool,
    lines: bool,
    html: bool,
}

/// An OCR request as sent.
#[derive(Debug)]
struct Asked {
    /// The image as a `data:` URL.
    url: String,
    max_tokens: Option<u64>,
    want: Want,
    /// Fields handed to the completion as they are.
    passed: Vec<(&'static str, Value)>,
}

/// A status and the message that goes with it.
type Refusal = (&'static str, String);

fn bad(msg: impl Into<String>) -> Refusal { ("400 Bad Request", msg.into()) }

/// Whether `body` is an encoded PNG or JPEG rather than JSON.
fn is_image(body: &[u8]) -> bool {
    body.starts_with(b"\x89PNG\r\n\x1a\n") || body.starts_with(&[0xff, 0xd8, 0xff])
}

/// Read a request body: JSON, or an image's own bytes.
fn parse(body: &[u8]) -> Result<Asked, Refusal> {
    let large = |n: usize| ("413 Payload Too Large", format!("the image is {n} bytes; the most is {MAX_IMAGE}"));
    let all = Want { text: true, lines: true, html: false };
    if is_image(body) {
        if body.len() > MAX_IMAGE { return Err(large(body.len())); }
        return Ok(Asked { url: format!("data:image/png;base64,{}", base64(body)), max_tokens: None, want: all, passed: Vec::new() });
    }
    let v: Value = serde_json::from_slice(body).map_err(|e| bad(format!("the body is neither a PNG, a JPEG nor JSON: {e}")))?;
    let image = v.get("image").and_then(Value::as_str).ok_or_else(|| bad("\"image\" must be a data URL or base64"))?;
    if image.starts_with("http://") || image.starts_with("https://") {
        return Err(bad("\"image\" must be a data URL or base64; remote URLs are not fetched"));
    }
    let payload = match image.strip_prefix("data:") {
        Some(rest) => rest.split_once(";base64,").map(|(_, p)| p).ok_or_else(|| bad("\"image\" must be a base64 data URL"))?,
        None => image,
    };
    if payload.len() / 4 * 3 > MAX_IMAGE { return Err(large(payload.len() / 4 * 3)); }
    let url = if image.starts_with("data:") { image.to_string() } else { format!("data:image/png;base64,{}", image.trim()) };
    let max_tokens = match v.get("max_tokens") {
        None | Some(Value::Null) => None,
        Some(n) => Some(n.as_u64().filter(|&n| n > 0).ok_or_else(|| bad("\"max_tokens\" must be a positive integer"))?),
    };
    let want = match v.get("return") {
        None | Some(Value::Null) => all,
        Some(Value::Array(names)) => {
            let mut want = Want { text: false, lines: false, html: false };
            for name in names {
                match name.as_str() {
                    Some("text") => want.text = true,
                    Some("lines") => want.lines = true,
                    Some("html") => want.html = true,
                    _ => return Err(bad(format!("\"return\" holds {name}; it takes \"text\", \"lines\" and \"html\""))),
                }
            }
            want
        }
        Some(_) => return Err(bad("\"return\" must be an array")),
    };
    let passed = ["timeout_ms", "image_max_tokens", "image_max_pixels"].into_iter()
        .filter_map(|k| v.get(k).filter(|f| !f.is_null()).map(|f| (k, f.clone()))).collect();
    Ok(Asked { url, max_tokens, want, passed })
}

/// Standard base64 with padding.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let word = chunk.iter().fold(0u32, |w, &b| w << 8 | b as u32) << (8 * (3 - chunk.len()));
        for k in 0..4 {
            out.push(if k <= chunk.len() { ALPHABET[(word >> (18 - 6 * k)) as usize & 63] as char } else { '=' });
        }
    }
    out
}

/// The completion body an OCR request runs as: the page, then the instruction, in
/// one user turn under no system prompt, as `ojas ocr` writes its prompt. Greedy,
/// with every step's distribution observed for the confidences, and as long as the
/// context allows unless the request says less.
fn completion(asked: &Asked, width: usize, height: usize) -> Value {
    let want = [("text", asked.want.text), ("lines", asked.want.lines), ("html", asked.want.html)];
    let mut body = json!({
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": asked.url}},
                {"type": "text", "text": DEFAULT_INSTRUCTION},
            ]},
        ],
        "temperature": 0.0,
        "logprobs": true,
        "max_tokens": asked.max_tokens.unwrap_or(u32::MAX as u64),
        MARK: {"width": width, "height": height,
               "return": want.iter().filter(|w| w.1).map(|w| w.0).collect::<Vec<_>>()},
    });
    for (k, v) in &asked.passed { body[*k] = v.clone(); }
    body
}

/// The completion body of the OCR request in `body`, answering a malformed one
/// itself. The image is decoded here for its size and to refuse what is no image.
pub(crate) fn request(stream: &mut Conn, body: &[u8]) -> Option<Value> {
    let built = parse(body).and_then(|asked| {
        let image = Image::from_data_url(&asked.url).map_err(|e| bad(format!("{e:#}")))?;
        if image.width == 0 || image.height == 0 { return Err(bad("the image has no pixels")); }
        Ok(completion(&asked, image.width, image.height))
    });
    match built {
        Ok(body) => Some(body),
        Err((status, msg)) => {
            metrics::failed();
            send_err(stream, status, &msg);
            None
        }
    }
}

/// How an OCR request is answered: the page's size and the parts asked for.
#[derive(Debug)]
pub(crate) struct Shape {
    width: usize,
    height: usize,
    want: Want,
    /// Whether the repetition guard ended the page.
    looped: bool,
}

impl Shape {
    /// The shape of the reply to `body`, `None` when it is no OCR request. An
    /// error, answered 501, when this server takes no images.
    pub(crate) fn of(ctx: &Ctx, body: &Value) -> Result<Option<Shape>, String> {
        let Some(mark) = body.get(MARK) else { return Ok(None) };
        if let Err(why) = ctx.images { return Err(format!("this model cannot do OCR: {why}")); }
        let side = |k: &str| mark.get(k).and_then(Value::as_u64).unwrap_or(0) as usize;
        let has = |name: &str| mark.get("return").and_then(Value::as_array).is_some_and(|r| r.iter().any(|v| v == name));
        Ok(Some(Shape {
            width: side("width"), height: side("height"), looped: false,
            want: Want { text: has("text"), lines: has("lines"), html: has("html") },
        }))
    }

    /// Take the tokens generated so far; false ends the page, which has fallen
    /// into a loop (`crate::ocr::loop_period`).
    pub(crate) fn token(&mut self, out: &[u32]) -> bool {
        self.looped = loop_period(out, LOOP_MAX_PERIOD, LOOP_MIN_RUN).is_some();
        !self.looped
    }

    /// The reply for the generated `html`, all but its usage. `steps` are each
    /// token's length in bytes and its log-probability; `length` says the page
    /// ran out of tokens. An error, answered 501, when the model wrote something
    /// that holds no block.
    pub(crate) fn page(&self, html: &str, steps: Option<&[(usize, Option<f64>)]>, length: bool) -> Result<Value, String> {
        let blocks = blocks(html);
        if blocks.is_empty() && !html.trim().is_empty() {
            return Err("this model cannot do OCR: its output for the page has no block with a data-bbox".into());
        }
        // Token offsets are offsets into `html` only when the two are the same bytes.
        let steps = steps.filter(|s| s.iter().map(|t| t.0).sum::<usize>() == html.len());
        let mut out = json!({
            "width": self.width, "height": self.height,
            "finish_reason": if self.looped { "loop" } else if length { "length" } else { "stop" },
        });
        if self.want.text {
            let text: Vec<String> = blocks.iter().filter(|b| !b.lines.is_empty())
                .map(|b| b.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n")).collect();
            out["text"] = json!(text.join("\n\n"));
        }
        if self.want.lines {
            let lines: Vec<Value> = blocks.iter().flat_map(|b| b.lines.iter().enumerate().map(|(k, line)| json!({
                "text": line.text,
                "box": line_box(b.bbox, k, b.lines.len(), self.width, self.height),
                "confidence": steps.and_then(|s| confidence(&line.span, s)).map(|c| (c * 1e4).round() / 1e4),
            }))).collect();
            out["lines"] = json!(lines);
        }
        if self.want.html { out["html"] = json!(html); }
        Ok(out)
    }
}

/// One block of the model's markup: a `div` with a `data-bbox`.
#[derive(Debug, PartialEq)]
struct Block {
    /// `x0 y0 x1 y1` as written, in thousandths of the page's sides.
    bbox: [f64; 4],
    lines: Vec<Line>,
}

#[derive(Debug, PartialEq)]
struct Line {
    text: String,
    /// The bytes of the markup from the line's first character to its last.
    span: Range<usize>,
}

/// The blocks of `html`, in the order written. A block's content runs to the next
/// block's tag, so output cut short mid-block still gives what was written of it.
fn blocks(html: &str) -> Vec<Block> {
    let (mut opens, mut at) = (Vec::new(), 0);
    while let Some(start) = html[at..].find("<div").map(|k| at + k) {
        let Some(end) = html[start..].find('>').map(|k| start + k + 1) else { break };
        if let Some(bbox) = bbox_of(&html[start..end]) { opens.push((start, end, bbox)); }
        at = end;
    }
    opens.iter().enumerate().map(|(k, &(_, from, bbox))| {
        let to = opens.get(k + 1).map_or(html.len(), |next| next.0);
        Block { bbox, lines: lines(html, from..to) }
    }).collect()
}

/// The four numbers of a tag's `data-bbox` attribute.
fn bbox_of(tag: &str) -> Option<[f64; 4]> {
    let value = tag.split_once("data-bbox=")?.1;
    let quote = value.chars().next().filter(|c| matches!(c, '"' | '\''))?;
    let value = value[1..].split_once(quote)?.0;
    let mut numbers = value.split(|c: char| c.is_whitespace() || c == ',').filter(|s| !s.is_empty())
        .map(|s| s.parse::<f64>().ok().filter(|v| v.is_finite()));
    let bbox = [numbers.next()??, numbers.next()??, numbers.next()??, numbers.next()??];
    numbers.next().is_none().then_some(bbox)
}

/// The lines of text in `html[within]`: tags removed, entities decoded, runs of
/// white space made one space. A `br` and the edges of block-level elements end a
/// line; a table row is a line, its cells a space apart.
fn lines(html: &str, within: Range<usize>) -> Vec<Line> {
    fn end(out: &mut Vec<Line>, text: &mut String, span: &mut Option<Range<usize>>) {
        let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if let Some(span) = span.take() { out.push(Line { text: joined, span }); }
        text.clear();
    }
    let (mut out, mut text, mut span) = (Vec::new(), String::new(), None);
    let mut i = within.start;
    while i < within.end {
        let rest = &html[i..within.end];
        if rest.starts_with('<') {
            // Output that ends inside a tag has no more text.
            let Some(close) = rest.find('>') else { break };
            let name: String = rest[1..close].trim_start_matches('/').chars()
                .take_while(char::is_ascii_alphanumeric).map(|c| c.to_ascii_lowercase()).collect();
            match name.as_str() {
                "br" | "p" | "div" | "li" | "ul" | "ol" | "tr" | "table" | "caption" | "blockquote" | "pre"
                | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => end(&mut out, &mut text, &mut span),
                "td" | "th" => text.push(' '),
                _ => {}
            }
            i += close + 1;
            continue;
        }
        let (c, len) = entity(rest).unwrap_or_else(|| {
            let c = rest.chars().next().unwrap_or(' ');
            (c, c.len_utf8())
        });
        if !c.is_whitespace() { span = Some(span.map_or(i, |s: Range<usize>| s.start)..i + len); }
        text.push(c);
        i += len;
    }
    end(&mut out, &mut text, &mut span);
    out
}

/// The character a leading `&...;` stands for, and the reference's length.
fn entity(text: &str) -> Option<(char, usize)> {
    let name = text.strip_prefix('&')?.split_once(';').map(|(name, _)| name).filter(|n| n.len() <= 8)?;
    let c = match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        _ => {
            let code = name.strip_prefix('#')?;
            let code = match code.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => code.parse().ok()?,
            };
            char::from_u32(code)?
        }
    };
    Some((c, name.len() + 2))
}

/// The box of line `k` of the `n` in a block, in pixels of a `width` by `height`
/// image: the block's box, kept inside the page, cut into `n` equal bands.
fn line_box(bbox: [f64; 4], k: usize, n: usize, width: usize, height: usize) -> [f64; 4] {
    let unit = |a: f64, b: f64| (a.min(b).clamp(0.0, 1000.0) / 1000.0, a.max(b).clamp(0.0, 1000.0) / 1000.0);
    let ((x0, x1), (y0, y1)) = (unit(bbox[0], bbox[2]), unit(bbox[1], bbox[3]));
    let band = (y1 - y0) / n.max(1) as f64;
    let pixels = |v: f64, side: usize| (v * side as f64 * 10.0).round() / 10.0;
    [pixels(x0, width), pixels(y0 + band * k as f64, height), pixels(x1, width), pixels(y0 + band * (k + 1) as f64, height)]
}

/// The mean probability of the tokens holding any byte of `span`; `None` when
/// none of them was observed.
fn confidence(span: &Range<usize>, steps: &[(usize, Option<f64>)]) -> Option<f64> {
    let (mut at, mut sum, mut n) = (0, 0.0, 0);
    for &(len, logprob) in steps {
        let token = at..at + len;
        at += len;
        if token.start >= span.end { break; }
        if let (true, Some(logprob)) = (token.end > span.start, logprob) {
            sum += logprob.exp();
            n += 1;
        }
    }
    (n > 0).then(|| sum / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x1 PNG: one red pixel, one blue.
    const PNG_2X1: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGP4zwAE/wEHAAH/4iOeWQAAAABJRU5ErkJggg==";

    const PAGE: &str = "<div data-bbox=\"100 50 500 100\" data-label=\"Section-Header\"><h1>Title &amp; more</h1></div>\
        <div data-bbox=\"100 200 400 500\" data-label=\"Text\"><p>first line<br/>second <b>bold</b> line<br/>third</p></div>";

    fn shape(want: Want) -> Shape { Shape { width: 2000, height: 1000, want, looped: false } }

    const ALL: Want = Want { text: true, lines: true, html: true };

    #[test]
    fn a_request_is_json_with_a_data_url_or_base64_or_the_image_itself() {
        let url = format!("data:image/png;base64,{PNG_2X1}");
        let asked = parse(json!({"image": url}).to_string().as_bytes()).unwrap();
        assert_eq!((asked.url.as_str(), asked.max_tokens, asked.want), (url.as_str(), None, Want { text: true, lines: true, html: false }));
        let bare = parse(json!({"image": PNG_2X1, "max_tokens": 64, "return": ["html"], "timeout_ms": 500}).to_string().as_bytes()).unwrap();
        assert_eq!((bare.url.as_str(), bare.max_tokens, bare.want), (url.as_str(), Some(64), Want { text: false, lines: false, html: true }));
        assert_eq!(bare.passed, vec![("timeout_ms", json!(500))]);
        // The bytes of a PNG as the body decode to the same image.
        let png = Image::from_data_url(&url).unwrap();
        let mut raw = b"\x89PNG\r\n\x1a\n".to_vec();
        raw.extend_from_slice(b"rest");
        assert!(is_image(&raw) && is_image(&[0xff, 0xd8, 0xff, 0xe0]) && !is_image(b"{\"image\": 1}"));
        assert_eq!(parse(&raw).unwrap().url, format!("data:image/png;base64,{}", base64(&raw)));
        assert_eq!((png.width, png.height), (2, 1));
    }

    #[test]
    fn a_malformed_request_is_400_and_an_oversized_image_413() {
        for body in [json!({}), json!({"image": 3}), json!({"image": "https://example.com/a.png"}),
                     json!({"image": "data:image/png,raw"}), json!({"image": PNG_2X1, "max_tokens": 0}),
                     json!({"image": PNG_2X1, "max_tokens": "many"}), json!({"image": PNG_2X1, "return": "text"}),
                     json!({"image": PNG_2X1, "return": ["text", "boxes"]})] {
            assert_eq!(parse(body.to_string().as_bytes()).unwrap_err().0, "400 Bad Request", "{body}");
        }
        assert_eq!(parse(b"not json").unwrap_err().0, "400 Bad Request");
        let big = "A".repeat(MAX_IMAGE / 3 * 4 + 8);
        assert_eq!(parse(json!({"image": big}).to_string().as_bytes()).unwrap_err().0, "413 Payload Too Large");
        let mut raw = b"\x89PNG\r\n\x1a\n".to_vec();
        raw.resize(MAX_IMAGE + 1, 0);
        assert_eq!(parse(&raw).unwrap_err().0, "413 Payload Too Large");
    }

    #[test]
    fn base64_pads_to_whole_groups() {
        for (bytes, text) in [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("foobar", "Zm9vYmFy")] {
            assert_eq!(base64(bytes.as_bytes()), text);
        }
        assert_eq!(base64(&[0xfb, 0xff]), "+/8=");
    }

    #[test]
    fn the_completion_is_the_page_then_the_instruction_decoded_greedily() {
        let asked = parse(json!({"image": PNG_2X1, "return": ["lines"], "image_max_tokens": 256}).to_string().as_bytes()).unwrap();
        let body = completion(&asked, 2, 1);
        let content = &body["messages"][1]["content"];
        assert_eq!((content[0]["type"].as_str(), content[1]["text"].as_str()), (Some("image_url"), Some(DEFAULT_INSTRUCTION)));
        assert_eq!(body["messages"][0], json!({"role": "system", "content": ""}), "a server's system prompt is not the page's");
        assert_eq!((body["temperature"].as_f64(), body["logprobs"].as_bool(), body["image_max_tokens"].as_u64()), (Some(0.0), Some(true), Some(256)));
        assert_eq!(body[MARK], json!({"width": 2, "height": 1, "return": ["lines"]}));
        assert_eq!(body["max_tokens"].as_u64(), Some(u32::MAX as u64), "the context bounds a page, not a default");
    }

    #[test]
    fn blocks_are_divs_with_a_box_and_their_lines() {
        let found = blocks(PAGE);
        assert_eq!(found.len(), 2);
        assert_eq!((found[0].bbox, found[0].lines[0].text.as_str()), ([100.0, 50.0, 500.0, 100.0], "Title & more"));
        let texts: Vec<&str> = found[1].lines.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(texts, ["first line", "second bold line", "third"]);
        // A line's span is its characters in the markup, inline tags between them included.
        assert_eq!(&PAGE[found[0].lines[0].span.clone()], "Title &amp; more");
        assert_eq!(&PAGE[found[1].lines[1].span.clone()], "second <b>bold</b> line");
    }

    #[test]
    fn odd_markup_is_read_for_what_it_holds() {
        // Single quotes, commas, a div without a box, white space, a numeric entity.
        let found = blocks("<div class=\"page\"><div data-label='Text' data-bbox='1, 2, 3, 4'>\n  a&#65;&#x42; \n b </div></div>");
        assert_eq!((found.len(), found[0].bbox, found[0].lines[0].text.as_str()), (1, [1.0, 2.0, 3.0, 4.0], "aAB b"));
        // A table row is a line.
        let table = blocks("<div data-bbox=\"0 0 9 9\"><table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table></div>");
        assert_eq!(table[0].lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), ["a b", "1 2"]);
        // A box of the wrong shape is no block; an empty block has no lines.
        assert!(blocks("<div data-bbox=\"1 2 3\">x</div><div data-bbox=\"1 2 3 4 5\">y</div><div data-bbox=\"a b c d\">z</div>").is_empty());
        assert!(blocks("<div data-bbox=\"1 2 3 4\" data-label=\"Picture\"></div>")[0].lines.is_empty());
        assert!(blocks("plain text & no tags").is_empty());
    }

    #[test]
    fn output_cut_short_keeps_what_was_written() {
        for (cut, lines) in [("<div data-bbox=\"1 2 3 4\"><p>whole<br/>par", vec!["whole", "par"]),
                             ("<div data-bbox=\"1 2 3 4\"><p>whole</p></div><div data-bbox=\"5 6", vec!["whole"]),
                             ("<div data-bbox=\"1 2 3 4\"><p>whole<br", vec!["whole"])] {
            let found = blocks(cut);
            assert_eq!(found.len(), 1, "{cut}");
            assert_eq!(found[0].lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), lines, "{cut}");
        }
    }

    #[test]
    fn a_box_maps_from_thousandths_to_the_pixels_of_the_image_sent() {
        assert_eq!(line_box([100.0, 50.0, 500.0, 100.0], 0, 1, 2000, 1000), [200.0, 50.0, 1000.0, 100.0]);
        // Three lines share a block's box in equal bands, top to bottom.
        let bands: Vec<[f64; 4]> = (0..3).map(|k| line_box([100.0, 200.0, 400.0, 500.0], k, 3, 2000, 1000)).collect();
        assert_eq!(bands, [[200.0, 200.0, 800.0, 300.0], [200.0, 300.0, 800.0, 400.0], [200.0, 400.0, 800.0, 500.0]]);
        // Out of range or reversed corners still give a box inside the page.
        assert_eq!(line_box([900.0, -20.0, 100.0, 1200.0], 0, 1, 1240, 1100), [124.0, 0.0, 1116.0, 1100.0]);
        // Sides that are no multiple of a thousand keep a tenth of a pixel.
        assert_eq!(line_box([65.0, 56.0, 510.0, 100.0], 0, 1, 1240, 1100), [80.6, 61.6, 632.4, 110.0]);
    }

    #[test]
    fn confidence_is_the_mean_probability_of_a_lines_tokens() {
        // Four tokens of 3, 2, 4 and 1 bytes at probabilities 1, 1/2, 1/4 and unobserved.
        let steps = [(3, Some(0.0)), (2, Some(0.5f64.ln())), (4, Some(0.25f64.ln())), (1, None)];
        let near = |got: Option<f64>, want: f64| assert!((got.unwrap() - want).abs() < 1e-12, "{got:?} for {want}");
        near(confidence(&(0..3), &steps), 1.0);
        near(confidence(&(2..4), &steps), 0.75);
        near(confidence(&(4..6), &steps), 0.375);
        near(confidence(&(0..10), &steps), 1.75 / 3.0);
        assert_eq!(confidence(&(9..10), &steps), None, "a line no step was observed for");
        assert_eq!(confidence(&(0..3), &[]), None);
    }

    #[test]
    fn a_reply_holds_what_was_asked_for() {
        // One token per byte, every one at probability 1/2.
        let steps = vec![(1usize, Some(0.5f64.ln())); PAGE.len()];
        let page = shape(ALL).page(PAGE, Some(&steps), false).unwrap();
        assert_eq!(page["text"], "Title & more\n\nfirst line\nsecond bold line\nthird");
        assert_eq!((page["width"].as_u64(), page["height"].as_u64(), page["finish_reason"].as_str()), (Some(2000), Some(1000), Some("stop")));
        assert_eq!(page["html"], PAGE);
        let lines = page["lines"].as_array().unwrap();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], json!({"text": "Title & more", "box": [200.0, 50.0, 1000.0, 100.0], "confidence": 0.5}));
        assert_eq!(lines[2]["box"], json!([200.0, 300.0, 800.0, 400.0]));
        // Only the parts named are sent.
        let text_only = shape(Want { text: true, lines: false, html: false }).page(PAGE, None, true).unwrap();
        assert!(text_only.get("lines").is_none() && text_only.get("html").is_none());
        assert_eq!(text_only["finish_reason"], "length");
    }

    #[test]
    fn confidence_is_null_without_steps_that_cover_the_text() {
        for steps in [None, Some(vec![(1usize, Some(0.0)); PAGE.len() - 1])] {
            let page = shape(ALL).page(PAGE, steps.as_deref(), false).unwrap();
            assert!(page["lines"].as_array().unwrap().iter().all(|l| l["confidence"].is_null()));
        }
    }

    #[test]
    fn a_blank_page_is_no_lines_and_output_without_blocks_is_not_ocr() {
        let blank = shape(ALL).page(" \n", None, false).unwrap();
        assert_eq!((blank["text"].as_str(), blank["lines"].as_array().map(Vec::len)), (Some(""), Some(0)));
        let refused = shape(ALL).page("The image shows a report.", None, false).unwrap_err();
        assert!(refused.contains("cannot do OCR"), "{refused}");
    }

    #[test]
    fn the_guard_ends_a_page_that_loops_and_the_reply_says_so() {
        let mut shape = shape(ALL);
        assert!(shape.token(&[1, 2, 3, 4]));
        assert!(!shape.token(&[7; LOOP_MIN_RUN]));
        assert_eq!(shape.page(PAGE, None, false).unwrap()["finish_reason"], "loop");
    }
}
