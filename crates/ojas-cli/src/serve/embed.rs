//! `ojas serve` for a sentence-embedding model: `POST /v1/embeddings`, in the
//! OpenAI shape.
//!
//! ```text
//! {"input": "text" | ["a", "b"], "model": "...", "encoding_format": "float" | "base64", "dimensions": N}
//! ```
//!
//! Each input becomes one unit-length vector, pooled the way the file declares
//! (`<arch>.pooling_type`). `dimensions` keeps the first N values and normalises
//! again. `base64` returns each vector as little-endian f32 bytes. An input longer
//! than the model's context is refused, not cut.
//!
//! A bidirectional encoder embeds every input of the requests that arrived together
//! in shared GPU passes. A causal decoder that pools its last token runs one input
//! at a time.

// The encoders run on Metal. Elsewhere `serve` refuses before loading, and the code
// past the refusal is never reached.
#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables, unused_imports))]

use super::{health, ready, send, send_err, send_json, serve_http_batches, set_shared_headers, Conn, Front, Identity, Loading};
use crate::flags::RunOpts;
use anyhow::{Context, Result};
use ojas_formats::gguf::{Gguf, Meta};
use ojas_tokenize::Bpe;
use serde_json::{json, Value};
use std::fmt::Write as _;

/// Most requests answered together.
const MAX_BATCH: usize = 32;

/// Most inputs one request may carry.
const MAX_INPUTS: usize = 2048;

/// Tokens a bidirectional encoder takes in one GPU pass; an input longer than this
/// gets a pass to itself.
const PASS_TOKENS: usize = 8192;

/// `<arch>.pooling_type` in the reference's numbering: 1 mean, 2 first token, 3 last
/// token, 4 rank. `None` when the file declares no pooling.
fn pooling_type(g: &Gguf) -> Option<i64> {
    match g.meta.get(&format!("{}.pooling_type", g.arch())) {
        Some(Meta::U32(v)) => Some(*v as i64),
        Some(Meta::I32(v)) => Some(*v as i64),
        _ => None,
    }.filter(|v| *v > 0)
}

/// Whether `path` is an embedding model, read from the header alone: its metadata
/// says how per-token states pool into one vector.
pub(crate) fn is_embedding_model(path: &str) -> bool {
    Gguf::open(path).is_ok_and(|g| pooling_type(&g).is_some())
}

fn pooling_name(kind: i64) -> &'static str {
    match kind {
        1 => "mean",
        2 => "cls",
        3 => "last",
        _ => "rank",
    }
}

/// A loaded encoder.
trait Encoder {
    /// Width of the vectors it returns.
    fn width(&self) -> usize;
    /// Longest input it takes, in tokens, the special tokens included.
    fn max_tokens(&self) -> usize;
    /// One vector per sequence, not normalised.
    fn embed(&self, seqs: &[Vec<u32>]) -> Result<Vec<Vec<f32>>>;
}

/// The special tokens the file asks to surround a text with.
#[derive(Clone, Copy, Debug, Default)]
struct Specials {
    bos: Option<u32>,
    eos: Option<u32>,
}

impl Specials {
    fn from_gguf(g: &Gguf) -> Result<Specials> {
        let flag = |k: &str| g.meta_bool(&format!("tokenizer.ggml.add_{k}_token")) == Some(true);
        let id = |k: &str| g.meta_u32(&format!("tokenizer.ggml.{k}_token_id")).with_context(|| format!("no tokenizer.ggml.{k}_token_id"));
        Ok(Specials {
            bos: if flag("bos") { Some(id("bos")?) } else { None },
            eos: if flag("eos") { Some(id("eos")?) } else { None },
        })
    }

    fn wrap(&self, text: impl Iterator<Item = u32>) -> Vec<u32> {
        self.bos.into_iter().chain(text).chain(self.eos).collect()
    }
}

/// A parsed request body.
#[derive(Debug, PartialEq)]
struct Embeddings {
    inputs: Vec<String>,
    base64: bool,
    /// Values kept of each vector; the model's width when absent.
    dimensions: Option<usize>,
}

/// An error reply: the status and the message.
type Refusal = (&'static str, String);

fn bad(msg: impl Into<String>) -> Refusal { ("400 Bad Request", msg.into()) }

/// Read a request body for a model whose vectors are `width` wide. `model` is
/// accepted and not checked: one model is served.
fn parse(body: &[u8], width: usize) -> Result<Embeddings, Refusal> {
    let body: Value = serde_json::from_slice(body).map_err(|_| bad("the request body is not valid JSON"))?;
    let body = body.as_object().ok_or_else(|| bad("the request body must be a JSON object"))?;
    let shape = "input must be a string or a non-empty array of strings";
    let inputs: Vec<String> = match body.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) if !items.is_empty() => {
            if items.len() > MAX_INPUTS {
                return Err(bad(format!("input has {} entries; at most {MAX_INPUTS} are taken in one request", items.len())));
            }
            items.iter().map(|v| v.as_str().map(str::to_string).ok_or_else(|| bad(shape))).collect::<Result<_, _>>()?
        }
        _ => return Err(bad(shape)),
    };
    if let Some(i) = inputs.iter().position(String::is_empty) {
        return Err(bad(format!("input {i} is empty")));
    }
    let base64 = match body.get("encoding_format") {
        None | Some(Value::Null) => false,
        Some(Value::String(s)) if s == "float" => false,
        Some(Value::String(s)) if s == "base64" => true,
        Some(_) => return Err(bad("encoding_format must be \"float\" or \"base64\"")),
    };
    let dimensions = match body.get("dimensions") {
        None | Some(Value::Null) => None,
        Some(v) => match v.as_u64().filter(|n| (1..=width as u64).contains(n)) {
            Some(n) => Some(n as usize),
            None => return Err(bad(format!("dimensions {v} is outside 1..={width}, the model's width"))),
        },
    };
    Ok(Embeddings { inputs, base64, dimensions })
}

/// Each input as token ids with the file's special tokens around it, refusing one
/// the model has no room for.
fn tokenize(inputs: &[String], tok: &Bpe, specials: Specials, max: usize) -> Result<Vec<Vec<u32>>, Refusal> {
    inputs.iter().enumerate().map(|(i, text)| {
        let ids = specials.wrap(tok.encode(text).into_iter().map(|t| t as u32));
        if ids.len() > max {
            return Err(bad(format!("input {i} is {} tokens; the model takes at most {max}", ids.len())));
        }
        Ok(ids)
    }).collect()
}

/// The first `keep` values of `v`, scaled to unit length.
fn unit(v: &[f32], keep: usize) -> Vec<f32> {
    let v = &v[..keep.min(v.len())];
    let norm = v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt().max(1e-12);
    v.iter().map(|&x| (x as f64 / norm) as f32).collect()
}

fn base64_into(out: &mut String, bytes: &[u8]) {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for chunk in bytes.chunks(3) {
        let n = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            if i <= chunk.len() { out.push(ALPHABET[(n >> shift) as usize & 63] as char) } else { out.push('=') }
        }
    }
}

/// The response body. Written directly: a vector of a thousand floats through a
/// JSON tree costs more than embedding a short text.
fn response_body(model: &str, rows: &[Vec<f32>], base64: bool, tokens: usize) -> String {
    let mut s = String::from("{\"object\":\"list\",\"data\":[");
    for (i, v) in rows.iter().enumerate() {
        if i > 0 { s.push(','); }
        let _ = write!(s, "{{\"object\":\"embedding\",\"index\":{i},\"embedding\":");
        if base64 {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            s.push('"');
            base64_into(&mut s, &bytes);
            s.push('"');
        } else {
            s.push('[');
            for (j, x) in v.iter().enumerate() {
                if j > 0 { s.push(','); }
                let _ = write!(s, "{x}");
            }
            s.push(']');
        }
        s.push('}');
    }
    let _ = write!(s, "],\"model\":{},\"usage\":{{\"prompt_tokens\":{tokens},\"total_tokens\":{tokens}}}}}", json!(model));
    s
}

/// Counters `GET /metrics` reports, in the Prometheus text format.
#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    /// Requests answered.
    requests: u64,
    /// Requests refused or failed.
    errors: u64,
    /// Texts embedded.
    inputs: u64,
    /// Tokens embedded, the special tokens included.
    input_tokens: u64,
    /// Time spent embedding, seconds.
    seconds: f64,
}

impl Metrics {
    fn render(&self) -> String {
        let counters: [(&str, &str, String); 5] = [
            ("requests_total", "Embedding requests answered.", self.requests.to_string()),
            ("request_errors_total", "Embedding requests refused or failed.", self.errors.to_string()),
            ("inputs_total", "Texts embedded.", self.inputs.to_string()),
            ("input_tokens_total", "Tokens embedded.", self.input_tokens.to_string()),
            ("seconds_total", "Time spent embedding, in seconds.", format!("{:.6}", self.seconds)),
        ];
        counters.iter().map(|(name, help, value)| {
            format!("# HELP ojas_embedding_{name} {help}\n# TYPE ojas_embedding_{name} counter\nojas_embedding_{name} {value}\n")
        }).collect()
    }
}

/// Write a reply on its own thread, so a client slow to read holds up no one.
fn reply(mut stream: Conn, result: Result<String, Refusal>) {
    std::thread::spawn(move || match result {
        Ok(body) => send(&mut stream, "200 OK", "application/json", body.as_bytes()),
        Err((status, msg)) => send_err(&mut stream, status, &msg),
    });
}

/// One request's vectors as its response body.
fn answer(model: &str, req: &Embeddings, vectors: &[Vec<f32>], width: usize, tokens: usize) -> Result<String, Refusal> {
    if vectors.iter().flatten().any(|x| !x.is_finite()) {
        return Err(("500 Internal Server Error", "the model produced a non-finite embedding".to_string()));
    }
    let rows: Vec<Vec<f32>> = vectors.iter().map(|v| unit(v, req.dimensions.unwrap_or(width))).collect();
    Ok(response_body(model, &rows, req.base64, tokens))
}

/// What the server says about the model it loaded.
struct About {
    /// The id `/v1/models` and every response carry.
    name: String,
    pooling: &'static str,
    /// Whether several inputs share a GPU pass.
    packed: bool,
    load_s: f64,
}

/// Serve `enc` on `front`. Requests that arrive while others are being answered
/// wait, and are then embedded together.
#[allow(clippy::too_many_arguments)]
fn serve_encoder(enc: &dyn Encoder, tok: &Bpe, specials: Specials, about: &About, opts: &RunOpts, front: &Front,
                 loading: Loading, identity: &Identity) -> Result<()> {
    let (width, max_tokens) = (enc.width(), enc.max_tokens());
    eprintln!("  {} | embedding model | {} pooling | width {width} | ctx {max_tokens} | loaded in {:.1}s",
        about.name, about.pooling, about.load_s);
    eprintln!("  {} sha256 {} | engine {}", identity.file, identity.sha256, super::REVISION);
    eprintln!("  listening on {}  ({}; {})", front.url(),
        if about.packed { "inputs embedded together" } else { "one input at a time" }, front.auth.describe());
    eprintln!("  POST /v1/embeddings  GET /health  GET /ready  GET /v1/models  GET /props  GET /metrics");
    let props = json!({
        "model": about.name, "kind": "embedding", "backend": "metal", "pooling": about.pooling,
        "embedding_length": width, "n_ctx": max_tokens, "max_inputs": MAX_INPUTS, "max_batch": MAX_BATCH,
        "load_seconds": about.load_s,
    });
    let mut metrics = Metrics::default();
    loading.finish();
    super::drain_on_sigterm(std::time::Duration::from_secs_f64(opts.drain_secs.max(0.0)));
    serve_http_batches(front, MAX_BATCH, |batch| {
        let mut pending: Vec<(Conn, Embeddings, Vec<Vec<u32>>)> = Vec::new();
        for (mut stream, req, route) in batch {
            match (req.method.as_str(), route.as_str()) {
                ("POST", "/v1/embeddings" | "/embeddings") => {
                    match parse(&req.body, width).and_then(|r| Ok((tokenize(&r.inputs, tok, specials, max_tokens)?, r))) {
                        Ok((seqs, r)) => pending.push((stream, r, seqs)),
                        Err(refused) => {
                            metrics.errors += 1;
                            reply(stream, Err(refused));
                        }
                    }
                }
                ("GET", "/metrics") => send(&mut stream, "200 OK", "text/plain; version=0.0.4", metrics.render().as_bytes()),
                ("GET", "/props") => send_json(&mut stream, "200 OK", &props),
                ("GET", "/health") => health(&mut stream),
                ("GET", "/ready") => ready(&mut stream),
                ("GET", "/v1/models") => {
                    let mut entry = json!({"id": about.name, "object": "model", "owned_by": "ojas"});
                    identity.describe(&mut entry);
                    send_json(&mut stream, "200 OK", &json!({"object": "list", "data": [entry]}))
                }
                _ => send_err(&mut stream, "404 Not Found", &format!("no route for {} {}", req.method, route)),
            }
        }
        if pending.is_empty() { return; }
        // Every pending request's inputs go to the encoder as one list.
        let all: Vec<Vec<u32>> = pending.iter().flat_map(|(_, _, seqs)| seqs.iter().cloned()).collect();
        let t = std::time::Instant::now();
        let result = enc.embed(&all).and_then(|v| {
            anyhow::ensure!(v.len() == all.len(), "the encoder returned {} vectors for {} inputs", v.len(), all.len());
            Ok(v)
        });
        metrics.seconds += t.elapsed().as_secs_f64();
        let mut at = 0usize;
        for (stream, req, seqs) in pending {
            let mine = at..at + seqs.len();
            at = mine.end;
            let tokens: usize = seqs.iter().map(Vec::len).sum();
            let body = match &result {
                Ok(vectors) => answer(&about.name, &req, &vectors[mine], width, tokens),
                Err(e) => Err(("500 Internal Server Error", format!("{e:#}"))),
            };
            match &body {
                Ok(_) => {
                    metrics.requests += 1;
                    metrics.inputs += seqs.len() as u64;
                    metrics.input_tokens += tokens as u64;
                }
                Err(_) => metrics.errors += 1,
            }
            reply(stream, body);
        }
    })
}

#[cfg(target_os = "macos")]
mod metal {
    use super::*;
    use ojas_core::Model;
    use ojas_models::decoder::{DecoderGpu, PRECISION_AUTO};

    struct MetalEncoder<'a> {
        dec: DecoderGpu<'a>,
        /// A bidirectional encoder, which pools on the GPU and takes many sequences a
        /// pass; otherwise a causal decoder read at its last token.
        packed: bool,
    }

    impl Encoder for MetalEncoder<'_> {
        fn width(&self) -> usize {
            if self.packed { self.dec.text_embedding_width().unwrap_or(0) } else { self.dec.hidden_dim() }
        }

        fn max_tokens(&self) -> usize {
            if self.packed { self.dec.text_encoder_max_positions().unwrap_or(0) } else { self.dec.max_seq() }
        }

        fn embed(&self, seqs: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
            if !self.packed {
                return seqs.iter().map(|s| {
                    let v = self.dec.embed_ids(s).map_err(anyhow::Error::msg)?;
                    match ojas_core::device_fault::peek() {
                        Some(fault) => Err(anyhow::anyhow!("{fault}")),
                        None => Ok(v),
                    }
                }).collect();
            }
            let mut out = Vec::with_capacity(seqs.len());
            let mut from = 0usize;
            while from < seqs.len() {
                let (mut to, mut used) = (from + 1, seqs[from].len());
                while to < seqs.len() && used + seqs[to].len() <= PASS_TOKENS {
                    used += seqs[to].len();
                    to += 1;
                }
                out.extend(self.dec.embed_text(&seqs[from..to])?);
                from = to;
            }
            Ok(out)
        }
    }

    pub(super) fn serve(path: &str, opts: &RunOpts, context: usize, front: Front, loading: Loading,
                        hashing: std::thread::JoinHandle<Result<Identity>>) -> Result<()> {
        use crate::backend::Device;
        anyhow::ensure!(matches!(opts.device, Device::Auto | Device::Metal),
            "embedding models run on Metal; --device {:?} cannot serve one", opts.device);
        let t = std::time::Instant::now();
        let mut g = Gguf::open(path)?;
        let arch = g.arch();
        let pooling = pooling_type(&g).context("this file declares no pooling")?;
        anyhow::ensure!(pooling <= 3, "{arch}: rank pooling (a reranker) is not served; /v1/embeddings needs mean, first-token or last-token pooling");
        let tok = Bpe::from_gguf(&g);
        let specials = Specials::from_gguf(&g)?;
        let name = match g.meta.get("general.name") {
            Some(Meta::Str(s)) if !s.is_empty() => s.clone(),
            _ => std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| arch.clone()),
        };
        // A causal decoder keeps keys and values for the positions it has seen, so its
        // longest input is the context it is loaded with. A bidirectional encoder keeps none.
        let causal = g.meta_bool(&format!("{arch}.attention.causal")) != Some(false);
        let trained = g.meta_u32(&format!("{arch}.context_length")).map_or(context, |n| n as usize);
        let max_seq = if causal { context.min(trained).max(1) } else { 64 };
        g.sidecars = false;
        let gpu = ojas_metal::MetalGpu::new().map_err(|e| anyhow::anyhow!("{e}")).context("opening the Metal GPU")?;
        let dec = DecoderGpu::load(&gpu, &mut g, max_seq, opts.precision.unwrap_or(PRECISION_AUTO), None, None)
            .with_context(|| format!("loading {path} as an embedding model"))?;
        let packed = dec.has_text_encoder();
        anyhow::ensure!(packed || pooling == 3,
            "{arch}: {} pooling over a causal decoder is not implemented; only last-token pooling is", pooling_name(pooling));
        let enc = MetalEncoder { dec, packed };
        anyhow::ensure!(enc.width() > 0 && enc.max_tokens() > 0, "{arch}: the encoder reports no width or context");
        let identity = Identity::finish(hashing)?;
        set_shared_headers(opts, &identity);
        let about = About { name, pooling: pooling_name(pooling), packed, load_s: t.elapsed().as_secs_f64() };
        serve_encoder(&enc, &tok, specials, &about, opts, &front, loading, &identity)
    }
}

/// `ojas serve` for an embedding model. `front` is already listening and `loading`
/// answers its probes until the model is in. `context` bounds a causal encoder's input.
pub(crate) fn serve(path: &str, opts: &RunOpts, context: usize, front: Front, loading: Loading,
                    hashing: std::thread::JoinHandle<Result<Identity>>) -> Result<()> {
    #[cfg(target_os = "macos")]
    return metal::serve(path, opts, context, front, loading, hashing);
    #[cfg(not(target_os = "macos"))]
    anyhow::bail!("embedding models are served on macOS (Metal) only in this build")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(body: &str) -> Result<Embeddings, Refusal> { parse(body.as_bytes(), 8) }

    #[test]
    fn input_is_one_string_or_a_list_of_them() {
        let one = parsed(r#"{"input": "a cat", "model": "any"}"#).unwrap();
        assert_eq!(one, Embeddings { inputs: vec!["a cat".into()], base64: false, dimensions: None });
        let many = parsed(r#"{"input": ["a", "b", "c"], "encoding_format": "float", "dimensions": null}"#).unwrap();
        assert_eq!(many.inputs, ["a", "b", "c"]);
        assert!(!many.base64);
    }

    #[test]
    fn malformed_requests_are_bad_requests() {
        for body in ["not json", "[]", "{}", r#"{"input": 3}"#, r#"{"input": []}"#, r#"{"input": ["a", 1]}"#,
                     r#"{"input": [[1, 2]]}"#, r#"{"input": ""}"#, r#"{"input": ["a", ""]}"#,
                     r#"{"input": "a", "encoding_format": "hex"}"#, r#"{"input": "a", "encoding_format": 1}"#,
                     r#"{"input": "a", "dimensions": 0}"#, r#"{"input": "a", "dimensions": 9}"#,
                     r#"{"input": "a", "dimensions": 2.5}"#, r#"{"input": "a", "dimensions": "4"}"#] {
            let (status, msg) = parsed(body).expect_err(body);
            assert_eq!(status, "400 Bad Request", "{body}");
            assert!(!msg.is_empty());
        }
        let many = json!({"input": vec!["a"; MAX_INPUTS + 1]}).to_string();
        assert!(parsed(&many).unwrap_err().1.contains("at most"));
    }

    #[test]
    fn dimensions_up_to_the_width_and_base64_are_read() {
        let r = parsed(r#"{"input": "a", "encoding_format": "base64", "dimensions": 8}"#).unwrap();
        assert_eq!((r.base64, r.dimensions), (true, Some(8)));
        assert!(parsed(r#"{"input": "a", "dimensions": 9}"#).unwrap_err().1.contains("1..=8"));
    }

    #[test]
    fn a_shortened_vector_is_unit_length_again() {
        let v = unit(&[3.0, 4.0, 12.0], 3);
        assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(unit(&[3.0, 4.0, 12.0], 2), [0.6, 0.8]);
        assert_eq!(unit(&[0.0, 0.0], 2), [0.0, 0.0]);
    }

    #[test]
    fn base64_pads_to_whole_groups() {
        for (bytes, want) in [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("foobar", "Zm9vYmFy")] {
            let mut s = String::new();
            base64_into(&mut s, bytes.as_bytes());
            assert_eq!(s, want);
        }
    }

    #[test]
    fn the_response_has_the_openai_shape() {
        let rows = vec![vec![0.5f32, -0.25], vec![1.0, 1e-7]];
        let v: Value = serde_json::from_str(&response_body("m \"1\"", &rows, false, 7)).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["model"], "m \"1\"");
        assert_eq!(v["usage"], json!({"prompt_tokens": 7, "total_tokens": 7}));
        let data = v["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        for (i, (entry, row)) in data.iter().zip(&rows).enumerate() {
            assert_eq!(entry["object"], "embedding");
            assert_eq!(entry["index"], i);
            let got: Vec<f32> = entry["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            assert_eq!(&got, row);
        }
    }

    #[test]
    fn base64_embeddings_are_little_endian_f32() {
        let v: Value = serde_json::from_str(&response_body("m", &[vec![1.0f32, -2.0]], true, 1)).unwrap();
        // 1.0f32 is 00 00 80 3f, -2.0f32 is 00 00 00 c0.
        assert_eq!(v["data"][0]["embedding"], "AACAPwAAAMA=");
    }

    #[test]
    fn an_answer_cuts_then_normalises_and_refuses_non_finite_values() {
        let req = Embeddings { inputs: vec!["a".into()], base64: false, dimensions: Some(2) };
        let v: Value = serde_json::from_str(&answer("m", &req, &[vec![3.0, 4.0, 12.0]], 3, 5).unwrap()).unwrap();
        assert_eq!(v["data"][0]["embedding"], json!([0.6, 0.8]));
        assert_eq!(answer("m", &req, &[vec![f32::NAN, 1.0, 0.0]], 3, 5).unwrap_err().0, "500 Internal Server Error");
    }

    #[test]
    fn special_tokens_surround_a_text_as_the_file_asks() {
        let text = || [7u32, 8].into_iter();
        assert_eq!(Specials { bos: None, eos: Some(2) }.wrap(text()), [7, 8, 2]);
        assert_eq!(Specials { bos: Some(1), eos: Some(2) }.wrap(text()), [1, 7, 8, 2]);
        assert_eq!(Specials::default().wrap(text()), [7, 8]);
    }

    #[test]
    fn metrics_render_as_prometheus_counters() {
        let m = Metrics { requests: 3, errors: 1, inputs: 5, input_tokens: 42, seconds: 0.25 };
        let text = m.render();
        for line in ["ojas_embedding_requests_total 3", "ojas_embedding_request_errors_total 1", "ojas_embedding_inputs_total 5",
                     "ojas_embedding_input_tokens_total 42", "ojas_embedding_seconds_total 0.250000",
                     "# TYPE ojas_embedding_requests_total counter"] {
            assert!(text.lines().any(|l| l == line), "missing {line:?} in\n{text}");
        }
    }
}
