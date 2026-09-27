//! Incremental detokenizer.
//!
//! A token is a byte string, not a character: emoji, CJK and accented text often
//! span several tokens, so decoding each id separately yields replacement
//! characters at every split. This buffers bytes and releases only complete
//! UTF-8, holding a partial sequence until the token that finishes it arrives.

use ojas_tokenize::Bpe;

#[derive(Default)]
pub struct Detok {
    buf: Vec<u8>,
}

impl Detok {
    /// Add one token and return whatever is now printable. May return "".
    pub fn push(&mut self, bpe: &Bpe, id: u32) -> String {
        self.buf.extend_from_slice(&bpe.decode_bytes(id as usize));
        self.take()
    }

    /// Flush at end of stream. A trailing partial sequence is broken output rather
    /// than a pending boundary, so it surfaces as a replacement character instead
    /// of being dropped.
    pub fn finish(&mut self) -> String {
        if self.buf.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        out
    }

    /// Drain everything currently printable. Loops because one buffer can hold an
    /// invalid sequence followed by valid text, which stopping at the first error
    /// would hold back until the next token arrived.
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
                        // Truncated but still valid so far: keep the tail and
                        // wait for the token that completes it. Without this
                        // branch every multi-byte character would be mangled.
                        None => {
                            self.buf.drain(..good);
                            return out;
                        }
                        // Invalid: emit a replacement and step over it, so the
                        // buffer always makes progress.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive `take` directly; the byte behaviour needs no real vocabulary.
    fn feed(d: &mut Detok, bytes: &[u8]) -> String {
        d.buf.extend_from_slice(bytes);
        d.take()
    }

    #[test]
    fn ascii_passes_straight_through() {
        let mut d = Detok::default();
        assert_eq!(feed(&mut d, b"hello"), "hello");
    }

    /// The bug this type exists to prevent: a 4-byte emoji arriving in pieces.
    #[test]
    fn multibyte_split_across_tokens_is_held_then_released() {
        let mut d = Detok::default();
        let rocket = "🚀".as_bytes();
        assert_eq!(feed(&mut d, &rocket[..2]), "", "partial must not be emitted");
        assert_eq!(feed(&mut d, &rocket[2..]), "🚀", "completing bytes release it whole");
    }

    #[test]
    fn text_before_a_partial_sequence_still_flows() {
        let mut d = Detok::default();
        let mut bytes = b"hi ".to_vec();
        bytes.extend_from_slice(&"é".as_bytes()[..1]);
        assert_eq!(feed(&mut d, &bytes), "hi ");
        assert_eq!(feed(&mut d, &"é".as_bytes()[1..]), "é");
    }

    /// Invalid bytes must not wedge the stream; later text still has to print.
    #[test]
    fn invalid_bytes_are_replaced_and_do_not_stall_the_stream() {
        let mut d = Detok::default();
        let out = feed(&mut d, &[0xff, b'o', b'k']);
        assert!(out.contains(char::REPLACEMENT_CHARACTER), "got {out:?}");
        assert!(out.ends_with("ok"), "stream must continue past bad bytes: {out:?}");
    }

    #[test]
    fn finish_surfaces_a_dangling_partial_rather_than_dropping_it() {
        let mut d = Detok::default();
        assert_eq!(feed(&mut d, &"é".as_bytes()[..1]), "");
        assert!(!d.finish().is_empty(), "a truncated tail must be visible, not silent");
    }

    #[test]
    fn finish_on_empty_is_empty() {
        assert_eq!(Detok::default().finish(), "");
    }
}
