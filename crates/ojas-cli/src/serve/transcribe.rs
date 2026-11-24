//! `ojas serve` for a speech model: `POST /v1/audio/transcriptions`, in the
//! OpenAI shape.
//!
//! ```text
//! curl -F file=@speech.wav -F response_format=verbose_json http://host:port/v1/audio/transcriptions
//! curl -H 'Content-Type: audio/wav' --data-binary @speech.wav 'http://host:port/v1/audio/transcriptions?language=en'
//! ```
//!
//! The body is `multipart/form-data` with `file` and the optional fields
//! `language` (a code the vocabulary has, detected when absent), `prompt`,
//! `response_format` (`json`, `text`, `verbose_json`, `srt`, `vtt`),
//! `temperature` and `timestamp_granularities[]` (`segment`); or the audio
//! itself, with the same options in the query string.
//!
//! Audio is WAV, decoded and resampled here; any other container is answered
//! 415. Audio longer than one model window is decoded window by window, each
//! resuming at the last timestamp of the one before. The model runs on the
//! CPU, one request at a time.

mod multipart;
mod wav;

use super::{health, ready, send, send_err, send_json, serve_http_batches, set_shared_headers, Conn, Front, Identity, Loading, Request};
use crate::flags::RunOpts;
use anyhow::{Context, Result};
use ojas_cpu::whisper_decode::{self, DecodeOpts, SAMPLE_RATE, WINDOW};
use ojas_cpu::CpuWhisper;
use ojas_formats::gguf::{Gguf, Meta};
use ojas_tokenize::Bpe;
use serde_json::{json, Value};

/// Whether `path` is a speech-to-text model, read from the header alone: the file
/// declares the audio encoder-decoder architecture and its mel frontend.
pub(crate) fn is_speech_model(path: &str) -> bool {
    Gguf::open(path).is_ok_and(|g| {
        let arch = g.arch();
        arch == "whisper" && g.meta_u32(&format!("{arch}.num_mel_bins")).is_some()
    })
}

/// An error reply: the status and the message.
type Refusal = (&'static str, String);

fn bad(msg: impl Into<String>) -> Refusal { ("400 Bad Request", msg.into()) }

#[derive(Clone, Copy, Debug, PartialEq)]
enum Format {
    Json,
    Text,
    Verbose,
    Srt,
    Vtt,
}

/// What a request asks for beside its audio.
#[derive(Debug, PartialEq)]
struct Options {
    /// Lower-case language code; `None` to detect it.
    language: Option<String>,
    prompt: String,
    format: Format,
    temperature: f32,
}

/// Read the form fields or query pairs of a request. `model` is accepted and not
/// checked: one model is served.
fn options(fields: &[(String, String)]) -> Result<Options, Refusal> {
    let mut o = Options { language: None, prompt: String::new(), format: Format::Json, temperature: 0.0 };
    for (key, value) in fields {
        let value = value.trim();
        match key.as_str() {
            "language" => o.language = (!value.is_empty()).then(|| value.to_ascii_lowercase()),
            "prompt" => o.prompt = value.to_string(),
            "response_format" => o.format = match value {
                "" | "json" => Format::Json,
                "text" => Format::Text,
                "verbose_json" => Format::Verbose,
                "srt" => Format::Srt,
                "vtt" => Format::Vtt,
                other => return Err(bad(format!("response_format {other:?} is not one of json, text, verbose_json, srt, vtt"))),
            },
            "temperature" if !value.is_empty() => o.temperature = match value.parse::<f32>() {
                Ok(t) if (0.0..=1.0).contains(&t) => t,
                _ => return Err(bad(format!("temperature {value:?} is not a number from 0 to 1"))),
            },
            "timestamp_granularities[]" | "timestamp_granularities" => match value {
                "" | "segment" => {}
                "word" => return Err(bad("word timestamps are not produced; timestamp_granularities[] may only be \"segment\"")),
                other => return Err(bad(format!("timestamp_granularities[] {other:?} is not \"segment\""))),
            },
            "stream" if matches!(value, "true" | "1") => return Err(bad("streamed transcriptions are not served; send stream=false")),
            _ => {}
        }
    }
    Ok(o)
}

/// `a=b&c=d` with percent escapes and `+` decoded.
fn query_pairs(query: &str) -> Vec<(String, String)> {
    let decode = |s: &str| {
        let b = s.as_bytes();
        let (mut out, mut i) = (Vec::with_capacity(b.len()), 0);
        while i < b.len() {
            let hex = b.get(i + 1..i + 3).filter(|h| b[i] == b'%' && h.iter().all(u8::is_ascii_hexdigit))
                .and_then(|h| u8::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok());
            match (b[i], hex) {
                (_, Some(byte)) => { out.push(byte); i += 3; }
                (b'+', _) => { out.push(b' '); i += 1; }
                (c, _) => { out.push(c); i += 1; }
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    };
    query.split('&').filter(|p| !p.is_empty()).map(|p| {
        let (k, v) = p.split_once('=').unwrap_or((p, ""));
        (decode(k), decode(v))
    }).collect()
}

/// The audio bytes and the options of a request, from either body shape.
fn read_request(req: &Request) -> Result<(&[u8], Options), Refusal> {
    let kind = req.content_type.as_deref().unwrap_or("");
    let base = kind.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    if base == "multipart/form-data" {
        let boundary = multipart::boundary(kind).ok_or_else(|| bad("the multipart Content-Type names no boundary"))?;
        let parts = multipart::parse(&req.body, &boundary).map_err(bad)?;
        let file = parts.iter().find(|p| p.name == "file").ok_or_else(|| bad("the form has no \"file\" field"))?;
        let fields: Vec<(String, String)> = parts.iter().filter(|p| p.name != "file")
            .map(|p| (p.name.clone(), String::from_utf8_lossy(p.data).into_owned())).collect();
        return Ok((file.data, options(&fields)?));
    }
    if matches!(base.as_str(), "application/json" | "application/x-www-form-urlencoded" | "text/plain") {
        return Err(("415 Unsupported Media Type",
            format!("a body of type {base} carries no audio; send multipart/form-data with a \"file\" field, or a raw audio/wav body")));
    }
    let query = req.path.split_once('?').map_or("", |(_, q)| q);
    Ok((&req.body, options(&query_pairs(query))?))
}

/// One segment of a transcript, as answered.
#[derive(Debug)]
struct Piece {
    seek: f32,
    start: f32,
    end: f32,
    text: String,
    tokens: Vec<u32>,
    temperature: f32,
    avg_logprob: f32,
    no_speech_prob: f32,
}

/// A transcript, as answered.
#[derive(Debug)]
struct Output {
    /// The language's code, when the model names languages.
    language: Option<String>,
    /// Length of the audio, seconds.
    duration: f32,
    pieces: Vec<Piece>,
}

impl Output {
    fn text(&self) -> String {
        self.pieces.iter().map(|p| p.text.as_str()).collect::<String>().trim().to_string()
    }
}

/// `HH:MM:SS` and milliseconds after `sep`.
fn clock(secs: f32, sep: char) -> String {
    let ms = (secs.max(0.0) as f64 * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}{sep}{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

fn round(v: f32, places: i32) -> f64 {
    let k = 10f64.powi(places);
    (v as f64 * k).round() / k
}

/// The response's content type and body.
fn render(format: Format, out: &Output) -> (&'static str, String) {
    match format {
        Format::Json => ("application/json", json!({"text": out.text()}).to_string()),
        Format::Text => ("text/plain; charset=utf-8", out.text() + "\n"),
        Format::Verbose => {
            let segments: Vec<Value> = out.pieces.iter().enumerate().map(|(id, p)| json!({
                "id": id, "seek": (p.seek * 100.0).round() as u64, "start": round(p.start, 2), "end": round(p.end, 2),
                "text": p.text, "tokens": p.tokens, "temperature": round(p.temperature, 1),
                "avg_logprob": round(p.avg_logprob, 4), "no_speech_prob": round(p.no_speech_prob, 4),
            })).collect();
            ("application/json", json!({
                "task": "transcribe", "language": out.language, "duration": round(out.duration, 2),
                "text": out.text(), "segments": segments,
            }).to_string())
        }
        Format::Srt => ("text/plain; charset=utf-8", out.pieces.iter().enumerate().map(|(i, p)| {
            format!("{}\n{} --> {}\n{}\n\n", i + 1, clock(p.start, ','), clock(p.end, ','), p.text.trim())
        }).collect()),
        Format::Vtt => ("text/vtt; charset=utf-8", std::iter::once("WEBVTT\n\n".to_string()).chain(out.pieces.iter().map(|p| {
            format!("{} --> {}\n{}\n\n", clock(p.start, '.'), clock(p.end, '.'), p.text.trim())
        })).collect()),
    }
}

/// Counters `GET /metrics` reports, in the Prometheus text format.
#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    /// Requests answered.
    requests: u64,
    /// Requests refused or failed.
    errors: u64,
    /// Audio transcribed, seconds.
    audio_seconds: f64,
    /// Time spent decoding audio and running the model, seconds.
    compute_seconds: f64,
}

impl Metrics {
    fn render(&self) -> String {
        let counters: [(&str, &str, String); 4] = [
            ("requests_total", "Transcription requests answered.", self.requests.to_string()),
            ("request_errors_total", "Transcription requests refused or failed.", self.errors.to_string()),
            ("audio_seconds_total", "Audio transcribed, in seconds.", format!("{:.3}", self.audio_seconds)),
            ("compute_seconds_total", "Time spent transcribing, in seconds.", format!("{:.6}", self.compute_seconds)),
        ];
        counters.iter().map(|(name, help, value)| {
            format!("# HELP ojas_transcription_{name} {help}\n# TYPE ojas_transcription_{name} counter\nojas_transcription_{name} {value}\n")
        }).collect()
    }
}

/// The marker tokens the file's metadata does not carry, found in its vocabulary.
#[derive(Debug, Default, PartialEq)]
struct Markers {
    /// `(code, token)` for every `<|xx|>` between start-of-transcript and the task tokens.
    languages: Vec<(String, u32)>,
    sot_prev: Option<u32>,
    no_speech: Option<u32>,
}

impl Markers {
    /// Scan the marker range `eot + 1 .. ts_begin`; `text` is a token's literal text.
    fn find(sot: u32, eot: u32, transcribe: u32, ts_begin: u32, text: impl Fn(u32) -> String) -> Markers {
        let mut m = Markers::default();
        for id in eot + 1..ts_begin {
            let t = text(id);
            let Some(inner) = t.strip_prefix("<|").and_then(|t| t.strip_suffix("|>")) else { continue };
            match inner {
                "startofprev" => m.sot_prev = Some(id),
                "nospeech" | "nocaptions" => m.no_speech = Some(id),
                code if id > sot && id < transcribe && (2..=3).contains(&code.len())
                    && code.bytes().all(|b| b.is_ascii_lowercase()) => m.languages.push((code.to_string(), id)),
                _ => {}
            }
        }
        m
    }
}

/// A loaded speech model and what it takes to turn its tokens into text.
struct Speech {
    model: CpuWhisper,
    bpe: Bpe,
    markers: Markers,
    /// The id `/v1/models` carries.
    name: String,
}

impl Speech {
    fn load(path: &str) -> Result<Speech> {
        let mut g = Gguf::open(path).with_context(|| format!("opening {path}"))?;
        let bpe = Bpe::from_gguf(&g);
        let name = match g.meta.get("general.name") {
            Some(Meta::Str(s)) if !s.is_empty() => s.clone(),
            _ => std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| g.arch()),
        };
        let model = CpuWhisper::load(&mut g).with_context(|| format!("loading {path} as a speech model"))?;
        anyhow::ensure!(bpe.len() >= model.vocab && model.vocab > 0,
            "{path}: the file carries {} vocabulary entries (tokenizer.ggml.tokens) for a model of {}; \
             a transcript cannot be turned into text without them", bpe.len(), model.vocab);
        anyhow::ensure!(model.eot < model.ts_begin && (model.ts_begin as usize) < model.vocab,
            "{path}: the end-of-text and first timestamp tokens are missing or out of order");
        let markers = Markers::find(model.sot, model.eot, model.transcribe, model.ts_begin, |id| bpe.decode(id as usize));
        Ok(Speech { model, bpe, markers, name })
    }

    fn text(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| self.bpe.decode_bytes(*t as usize)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The token of a requested language.
    fn language(&self, code: &str) -> Result<u32, Refusal> {
        if let Some((_, id)) = self.markers.languages.iter().find(|(c, _)| c == code) { return Ok(*id); }
        if self.markers.languages.is_empty() {
            return Err(bad(format!("this model transcribes one language and takes no \"language\" (got {code:?})")));
        }
        Err(bad(format!("language {code:?} is not one this model knows; send a code such as \"en\", or none to detect it")))
    }

    /// Decode the audio bytes and transcribe them.
    fn transcribe(&self, audio: &[u8], o: &Options) -> Result<Output, Refusal> {
        if audio.is_empty() { return Err(bad("the request carries no audio")); }
        let decoded = wav::decode(audio).map_err(|e| match e {
            wav::AudioError::Unsupported(m) => ("415 Unsupported Media Type", m),
            wav::AudioError::Invalid(m) => ("400 Bad Request", m),
        })?;
        if decoded.samples.is_empty() { return Err(bad("the WAV file holds no samples")); }
        let pcm = wav::resample(&decoded.samples, decoded.rate, SAMPLE_RATE as u32);
        let language = o.language.as_deref().map(|code| self.language(code)).transpose()?;
        // The prompt opens with a space, as text in the middle of a transcript does.
        let prompt: Vec<u32> = if o.prompt.is_empty() { Vec::new() }
            else { self.bpe.encode(&format!(" {}", o.prompt)).into_iter().map(|t| t as u32).filter(|t| *t < self.model.eot).collect() };
        let languages: Vec<u32> = self.markers.languages.iter().map(|(_, id)| *id).collect();
        // Timestamps when the answer shows them, and whenever audio spans windows:
        // they are what lets the next window start between words.
        let timestamps = !matches!(o.format, Format::Json | Format::Text) || pcm.len() > WINDOW;
        let t = whisper_decode::transcribe(&self.model, &pcm, &DecodeOpts {
            language, languages: &languages, prompt: &prompt, sot_prev: self.markers.sot_prev,
            no_speech: self.markers.no_speech, timestamps, temperature: o.temperature,
        });
        let language = t.language.and_then(|id| self.markers.languages.iter().find(|(_, l)| *l == id)).map(|(code, _)| code.clone());
        let pieces = t.segments.into_iter().map(|s| Piece {
            seek: s.seek, start: s.start, end: s.end, text: self.text(&s.tokens), tokens: s.tokens,
            temperature: s.temperature, avg_logprob: s.avg_logprob, no_speech_prob: s.no_speech_prob,
        }).collect();
        Ok(Output { language, duration: pcm.len() as f32 / SAMPLE_RATE as f32, pieces })
    }
}

/// Write a reply on its own thread, so a client slow to read holds up no one.
fn reply(mut stream: Conn, result: Result<(&'static str, String), Refusal>) {
    std::thread::spawn(move || match result {
        Ok((kind, body)) => send(&mut stream, "200 OK", kind, body.as_bytes()),
        Err((status, msg)) => send_err(&mut stream, status, &msg),
    });
}

/// `ojas serve` for a speech model. `front` is already listening and `loading`
/// answers its probes until the model is in.
pub(crate) fn serve(path: &str, opts: &RunOpts, front: Front, loading: Loading,
                    hashing: std::thread::JoinHandle<Result<Identity>>) -> Result<()> {
    use crate::backend::Device;
    anyhow::ensure!(matches!(opts.device, Device::Auto | Device::Cpu),
        "speech models run on the CPU; --device {:?} cannot serve one", opts.device);
    let t = std::time::Instant::now();
    let speech = Speech::load(path)?;
    let load_s = t.elapsed().as_secs_f64();
    let identity = Identity::finish(hashing)?;
    set_shared_headers(opts, &identity);
    let m = &speech.model;
    eprintln!("  {} | speech model | cpu | {} encoder + {} decoder layers, width {} | {} languages | loaded in {load_s:.1}s",
        speech.name, m.n_enc, m.n_dec, m.d, speech.markers.languages.len().max(1));
    eprintln!("  {} sha256 {} | engine {}", identity.file, identity.sha256, super::REVISION);
    eprintln!("  listening on {}  (one request at a time; {})", front.url(), front.auth.describe());
    eprintln!("  POST /v1/audio/transcriptions  GET /health  GET /ready  GET /v1/models  GET /props  GET /metrics");
    let props = json!({
        "model": speech.name, "kind": "speech", "backend": "cpu", "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW / SAMPLE_RATE, "audio": ["wav"],
        "languages": speech.markers.languages.iter().map(|(code, _)| code.as_str()).collect::<Vec<_>>(),
        "response_formats": ["json", "text", "verbose_json", "srt", "vtt"], "load_seconds": load_s,
    });
    let mut metrics = Metrics::default();
    loading.finish();
    super::drain_on_sigterm(std::time::Duration::from_secs_f64(opts.drain_secs.max(0.0)));
    serve_http_batches(&front, 1, |batch| {
        for (mut stream, req, route) in batch {
            match (req.method.as_str(), route.as_str()) {
                ("POST", "/v1/audio/transcriptions" | "/audio/transcriptions") => {
                    let t = std::time::Instant::now();
                    let result = read_request(&req).and_then(|(audio, o)| Ok((o.format, speech.transcribe(audio, &o)?)));
                    let took = t.elapsed().as_secs_f64();
                    match &result {
                        Ok((_, out)) => {
                            metrics.requests += 1;
                            metrics.audio_seconds += out.duration as f64;
                            metrics.compute_seconds += took;
                            eprintln!("  transcribed {:.1}s of audio in {took:.2}s ({} segments)", out.duration, out.pieces.len());
                        }
                        Err(_) => metrics.errors += 1,
                    }
                    reply(stream, result.map(|(format, out)| render(format, &out)));
                }
                ("GET", "/metrics") => send(&mut stream, "200 OK", "text/plain; version=0.0.4", metrics.render().as_bytes()),
                ("GET", "/props") => send_json(&mut stream, "200 OK", &props),
                ("GET", "/health") => health(&mut stream),
                ("GET", "/ready") => ready(&mut stream),
                ("GET", "/v1/models") => {
                    let mut entry = json!({"id": speech.name, "object": "model", "owned_by": "ojas"});
                    identity.describe(&mut entry);
                    send_json(&mut stream, "200 OK", &json!({"object": "list", "data": [entry]}))
                }
                _ => send_err(&mut stream, "404 Not Found", &format!("no route for {} {}", req.method, route)),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    fn request(content_type: Option<&str>, path: &str, body: &[u8]) -> Request {
        let head = format!("POST {path} HTTP/1.1\r\n{}Content-Length: {}\r\n\r\n",
            content_type.map(|c| format!("Content-Type: {c}\r\n")).unwrap_or_default(), body.len());
        let bytes = [head.as_bytes(), body].concat();
        super::super::read_request(&mut std::io::BufReader::new(&bytes[..])).unwrap().unwrap()
    }

    #[test]
    fn options_default_and_read_every_field() {
        assert_eq!(options(&[]).unwrap(), Options { language: None, prompt: String::new(), format: Format::Json, temperature: 0.0 });
        let o = options(&fields(&[("model", "any"), ("language", "EN"), ("prompt", "Names: Ojas."), ("response_format", "verbose_json"),
                                  ("temperature", "0.4"), ("timestamp_granularities[]", "segment")])).unwrap();
        assert_eq!(o, Options { language: Some("en".into()), prompt: "Names: Ojas.".into(), format: Format::Verbose, temperature: 0.4 });
        for (value, format) in [("json", Format::Json), ("text", Format::Text), ("srt", Format::Srt), ("vtt", Format::Vtt)] {
            assert_eq!(options(&fields(&[("response_format", value)])).unwrap().format, format);
        }
    }

    #[test]
    fn options_that_cannot_be_honoured_are_bad_requests() {
        for pair in [("response_format", "xml"), ("temperature", "hot"), ("temperature", "1.5"), ("temperature", "-0.1"),
                     ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "phoneme"), ("stream", "true")] {
            let (status, msg) = options(&fields(&[pair])).expect_err(pair.1);
            assert_eq!(status, "400 Bad Request");
            assert!(!msg.is_empty());
        }
    }

    #[test]
    fn a_multipart_request_gives_its_file_and_fields() {
        let body = b"--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n\x00\x01\r\n\xff\r\n\
                     --B\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\ntext\r\n\
                     --B\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nde\r\n--B--\r\n";
        let req = request(Some("multipart/form-data; boundary=B"), "/v1/audio/transcriptions", body);
        let (audio, o) = read_request(&req).unwrap();
        assert_eq!(audio, b"\x00\x01\r\n\xff");
        assert_eq!((o.format, o.language.as_deref()), (Format::Text, Some("de")));
        let none = request(Some("multipart/form-data; boundary=B"), "/v1/audio/transcriptions",
            b"--B\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nde\r\n--B--\r\n");
        assert!(read_request(&none).unwrap_err().1.contains("file"));
        let unbounded = request(Some("multipart/form-data"), "/v1/audio/transcriptions", body);
        assert_eq!(read_request(&unbounded).unwrap_err().0, "400 Bad Request");
    }

    #[test]
    fn a_raw_body_takes_its_options_from_the_query() {
        let req = request(Some("audio/wav"), "/v1/audio/transcriptions?response_format=vtt&prompt=two+words%2C%20more&timestamp_granularities%5B%5D=segment", b"RIFF");
        let (audio, o) = read_request(&req).unwrap();
        assert_eq!(audio, b"RIFF");
        assert_eq!((o.format, o.prompt.as_str()), (Format::Vtt, "two words, more"));
        assert_eq!(read_request(&request(None, "/v1/audio/transcriptions", b"RIFF")).unwrap().1.format, Format::Json);
        let json = request(Some("application/json"), "/v1/audio/transcriptions", b"{}");
        assert_eq!(read_request(&json).unwrap_err().0, "415 Unsupported Media Type");
        assert_eq!(query_pairs("a=1&b&c=%zz%4"), fields(&[("a", "1"), ("b", ""), ("c", "%zz%4")]));
    }

    fn output() -> Output {
        let piece = |seek: f32, start: f32, end: f32, text: &str| Piece {
            seek, start, end, text: text.into(), tokens: vec![1, 2], temperature: 0.0, avg_logprob: -0.25, no_speech_prob: 0.01,
        };
        Output { language: Some("en".into()), duration: 31.5,
                 pieces: vec![piece(0.0, 0.0, 2.5, " Hello there."), piece(0.0, 2.5, 3661.25, " General \"quote\".")] }
    }

    #[test]
    fn json_and_text_carry_the_whole_text() {
        let (kind, body) = render(Format::Json, &output());
        assert_eq!(kind, "application/json");
        assert_eq!(serde_json::from_str::<Value>(&body).unwrap(), json!({"text": "Hello there. General \"quote\"."}));
        assert_eq!(render(Format::Text, &output()), ("text/plain; charset=utf-8", "Hello there. General \"quote\".\n".to_string()));
        let empty = Output { language: None, duration: 1.0, pieces: Vec::new() };
        assert_eq!(render(Format::Json, &empty).1, r#"{"text":""}"#);
    }

    #[test]
    fn verbose_json_has_the_openai_shape() {
        let v: Value = serde_json::from_str(&render(Format::Verbose, &output()).1).unwrap();
        assert_eq!((&v["task"], &v["language"], &v["duration"]), (&json!("transcribe"), &json!("en"), &json!(31.5)));
        assert_eq!(v["text"], "Hello there. General \"quote\".");
        let segments = v["segments"].as_array().unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[1]["id"], 1);
        assert_eq!((&segments[0]["start"], &segments[0]["end"], &segments[0]["text"]), (&json!(0.0), &json!(2.5), &json!(" Hello there.")));
        assert_eq!(segments[0]["tokens"], json!([1, 2]));
        assert_eq!(segments[0]["avg_logprob"], json!(-0.25));
    }

    #[test]
    fn subtitles_number_and_time_their_segments() {
        assert_eq!(render(Format::Srt, &output()).1,
            "1\n00:00:00,000 --> 00:00:02,500\nHello there.\n\n2\n00:00:02,500 --> 01:01:01,250\nGeneral \"quote\".\n\n");
        let (kind, vtt) = render(Format::Vtt, &output());
        assert_eq!(kind, "text/vtt; charset=utf-8");
        assert!(vtt.starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello there.\n\n00:00:02.500 --> 01:01:01.250\n"));
    }

    #[test]
    fn markers_are_found_in_the_vocabulary() {
        // end 10, start 11, languages 12..=14, tasks 15 and 16, then the rest; timestamps from 20.
        let vocab = ["<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|de|>", "<|yue|>", "<|translate|>", "<|transcribe|>",
                     "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>"];
        let m = Markers::find(11, 10, 16, 21, |id| vocab[id as usize - 10].to_string());
        assert_eq!(m, Markers { languages: vec![("en".into(), 12), ("de".into(), 13), ("yue".into(), 14)],
                                sot_prev: Some(18), no_speech: Some(19) });
        // A one-language vocabulary names none.
        let vocab = ["<|endoftext|>", "<|startoftranscript|>", "<|transcribe|>", "<|startofprev|>", "<|nocaptions|>", "<|notimestamps|>"];
        let m = Markers::find(11, 10, 12, 16, |id| vocab[id as usize - 10].to_string());
        assert_eq!(m, Markers { languages: Vec::new(), sot_prev: Some(13), no_speech: Some(14) });
    }

    #[test]
    fn metrics_render_as_prometheus_counters() {
        let m = Metrics { requests: 3, errors: 1, audio_seconds: 42.5, compute_seconds: 0.25 };
        let text = m.render();
        for line in ["ojas_transcription_requests_total 3", "ojas_transcription_request_errors_total 1",
                     "ojas_transcription_audio_seconds_total 42.500", "ojas_transcription_compute_seconds_total 0.250000",
                     "# TYPE ojas_transcription_requests_total counter"] {
            assert!(text.lines().any(|l| l == line), "missing {line:?} in\n{text}");
        }
    }

    #[test]
    fn audio_that_is_not_wav_is_named_in_the_refusal() {
        // The decoding step alone: no model is needed to refuse a container.
        assert!(matches!(wav::decode(b"ID3\x04\0\0\0\0\0\0"), Err(wav::AudioError::Unsupported(m)) if m.contains("mp3")));
        let ok = wav::tests::wav(1, 2, 44100, 16, &[0u8; 4 * 441], false);
        let a = wav::decode(&ok).unwrap();
        assert_eq!(wav::resample(&a.samples, a.rate, SAMPLE_RATE as u32).len(), 160);
    }
}
