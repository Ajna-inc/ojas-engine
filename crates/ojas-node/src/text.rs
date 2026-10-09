//! Tokenising on the node: prompts are templated and encoded here, tokens decoded
//! here, so workers and remote peers only ever see ids.
//!
//! Chat templating reuses `ojas_tokenize::chat_transcript`, the same function the
//! CLI's `serve` uses, which picks the template from the GGUF architecture.
//! [`Detok`] follows `ojas-cli/src/detok.rs`: a token is a byte string, so text is
//! released only as complete UTF-8.

use anyhow::{Context, Result};
use ojas_tokenize::Bpe;
use std::path::Path;

pub struct Tok {
    pub bpe: Bpe,
    pub arch: String,
    pub eog: Vec<u32>,
    pub vocab: usize,
}

impl Tok {
    pub fn open(path: &Path) -> Result<Tok> {
        let g = ojas_formats::gguf::Gguf::open(&path.to_string_lossy()).with_context(|| format!("opening {}", path.display()))?;
        let arch = g.arch();
        let bpe = Bpe::from_gguf(&g);
        let eog = ojas_tokenize::eog_token_ids(&g, &arch);
        Ok(Tok { vocab: bpe.len(), bpe, arch, eog })
    }

    pub fn encode(&self, s: &str) -> Vec<u32> {
        self.bpe.encode(s).into_iter().map(|v| v as u32).collect()
    }

    pub fn chat(&self, system: &str, turns: &[(String, String)]) -> Vec<u32> {
        self.encode(&ojas_tokenize::chat_transcript(&self.arch, system, turns))
    }

    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }
}

#[derive(Default)]
pub struct Detok {
    buf: Vec<u8>,
}

impl Detok {
    pub fn push(&mut self, bpe: &Bpe, id: u32) -> String {
        self.buf.extend_from_slice(&bpe.decode_bytes(id as usize));
        self.take()
    }

    /// End of stream: a dangling partial sequence is broken output, shown as a
    /// replacement character rather than dropped.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        out
    }

    fn take(&mut self) -> String {
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    return out;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(&String::from_utf8_lossy(&self.buf[..good]));
                    match e.error_len() {
                        None => {
                            self.buf.drain(..good);
                            return out;
                        }
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.buf.drain(..good + bad);
                        }
                    }
                }
            }
        }
    }
}

/// Detokenised output with stop strings applied.
pub struct TextStream {
    detok: Detok,
    stop: ojas_infer::StopMatcher,
    stopped: bool,
}

impl TextStream {
    pub fn new(stops: &[String]) -> TextStream {
        TextStream { detok: Detok::default(), stop: ojas_infer::StopMatcher::new(stops.iter().cloned()), stopped: false }
    }

    /// Text now safe to show, and whether a stop string ended the output.
    pub fn push(&mut self, bpe: &Bpe, id: u32) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }
        let piece = self.detok.push(bpe, id);
        let (out, hit) = self.stop.push(&piece);
        self.stopped = hit;
        (out, hit)
    }

    pub fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        let tail = self.detok.finish();
        let (mut out, hit) = self.stop.push(&tail);
        self.stopped = hit;
        if !hit {
            out.push_str(&self.stop.finish());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_multibyte_character_is_held_until_complete() {
        let mut d = Detok::default();
        let r = "🚀".as_bytes();
        d.buf.extend_from_slice(&r[..2]);
        assert_eq!(d.take(), "");
        d.buf.extend_from_slice(&r[2..]);
        assert_eq!(d.take(), "🚀");
        d.buf.extend_from_slice(&[0xff, b'o', b'k']);
        assert!(d.take().ends_with("ok"));
    }
}
