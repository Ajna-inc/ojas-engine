//! Running a grammar against generated tokens.
//!
//! The matcher state is the set of parse stacks the text so far leaves open. A
//! stack is a path of positions `(rule, alternative, element)`, innermost on top;
//! after expansion every top points at a character element, and an empty stack
//! means the grammar can end here. A code point advances each stack whose top
//! accepts it. Tokens are byte strings that may split a code point, so the state
//! also carries a partially decoded code point between tokens. Bytes are decoded as
//! strict UTF-8: an overlong form, a surrogate or a value past U+10FFFF is
//! rejected, so a model cannot spell an allowed character with bytes no reader
//! would decode to it.

use crate::gbnf::{Elem, Grammar};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Pos {
    rule: u32,
    alt: u32,
    idx: u32,
}

type Stack = Vec<Pos>;

/// Where a grammar is after some text: the open parse stacks and any partial
/// UTF-8 sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    stacks: Vec<Stack>,
    cp: u32,
    need: u8,
    /// The smallest code point the partial one may encode, by its length.
    min: u32,
}

impl State {
    /// The text so far can end here.
    pub fn accepting(&self) -> bool { self.need == 0 && self.stacks.iter().any(|s| s.is_empty()) }

    /// The text so far must end here: no character can follow.
    pub fn finished(&self) -> bool { self.need == 0 && self.stacks.iter().all(|s| s.is_empty()) }
}

impl Grammar {
    /// The state before any text.
    pub fn start(&self) -> State {
        let mut stacks = Vec::new();
        for a in 0..self.rules[self.root as usize].len() as u32 {
            self.expand(vec![Pos { rule: self.root, alt: a, idx: 0 }], &mut stacks);
        }
        normalize(&mut stacks);
        State { stacks, cp: 0, need: 0, min: 0 }
    }

    /// Push every stack `stack` can become with a character element (or nothing)
    /// on top: finished alternatives pop to their parent and rule references open
    /// each of their alternatives.
    fn expand(&self, mut stack: Stack, out: &mut Vec<Stack>) {
        loop {
            let Some(&top) = stack.last() else { out.push(stack); return };
            let seq = &self.rules[top.rule as usize][top.alt as usize];
            if top.idx as usize >= seq.len() {
                stack.pop();
                if let Some(parent) = stack.last_mut() { parent.idx += 1; }
                continue;
            }
            match &seq[top.idx as usize] {
                Elem::Chars { .. } => { out.push(stack); return; }
                Elem::Rule(r) => {
                    let n = self.rules[*r as usize].len() as u32;
                    for a in 0..n {
                        let mut s = stack.clone();
                        s.push(Pos { rule: *r, alt: a, idx: 0 });
                        self.expand(s, out);
                    }
                    return;
                }
            }
        }
    }

    /// The stacks that remain after code point `c`.
    fn advance(&self, stacks: &[Stack], c: u32) -> Vec<Stack> {
        let mut out = Vec::new();
        for s in stacks {
            let Some(top) = s.last() else { continue };
            let elem = &self.rules[top.rule as usize][top.alt as usize][top.idx as usize];
            if elem.matches(c) {
                let mut n = s.clone();
                n.last_mut().unwrap().idx += 1;
                self.expand(n, &mut out);
            }
        }
        normalize(&mut out);
        out
    }

    /// Feed raw bytes. `None` when the grammar rejects them.
    pub fn feed(&self, state: &State, bytes: &[u8]) -> Option<State> {
        let mut st = state.clone();
        for &b in bytes {
            if st.need == 0 {
                let (cp, need, min) = match b {
                    0x00..=0x7F => (b as u32, 0, 0),
                    0xC2..=0xDF => ((b & 0x1F) as u32, 1, 0x80),
                    0xE0..=0xEF => ((b & 0x0F) as u32, 2, 0x800),
                    0xF0..=0xF4 => ((b & 0x07) as u32, 3, 0x1_0000),
                    _ => return None,
                };
                (st.cp, st.need, st.min) = (cp, need, min);
            } else {
                if b & 0xC0 != 0x80 { return None; }
                st.cp = (st.cp << 6) | (b & 0x3F) as u32;
                st.need -= 1;
            }
            if st.need == 0 {
                if st.cp < st.min || st.cp > 0x10_FFFF || (0xD800..=0xDFFF).contains(&st.cp) { return None; }
                st.stacks = self.advance(&st.stacks, st.cp);
                if st.stacks.is_empty() { return None; }
            } else if !self.can_continue(&st) {
                return None;
            }
        }
        Some(st)
    }

    /// Whether some valid code point beginning with the partial one in `state` is
    /// accepted, so a token ending mid-character is allowed only where the
    /// character it starts can follow.
    fn can_continue(&self, state: &State) -> bool {
        let shift = 6 * state.need as u32;
        let lo = (state.cp << shift).max(state.min);
        let hi = ((state.cp << shift) | ((1 << shift) - 1)).min(0x10_FFFF);
        if lo > hi || (0xD800 <= lo && hi <= 0xDFFF) { return false; }
        state.stacks.iter().filter_map(|s| s.last())
            .any(|top| self.rules[top.rule as usize][top.alt as usize][top.idx as usize].matches_any(lo, hi))
    }

    /// Whether `text` is a complete sentence of the grammar.
    pub fn accepts(&self, text: &str) -> bool {
        self.feed(&self.start(), text.as_bytes()).is_some_and(|s| s.accepting())
    }
}

/// Order and deduplicate stacks so equal states compare equal and the set does
/// not grow with ambiguous grammars.
fn normalize(stacks: &mut Vec<Stack>) {
    stacks.sort_unstable();
    stacks.dedup();
}

/// The byte string of every token, and which tokens are special. Built once per
/// model and shared by every constrained request.
#[derive(Clone, Debug)]
pub struct TokenVocab {
    bytes: Vec<Vec<u8>>,
    special: Vec<bool>,
    eog: Vec<bool>,
}

impl TokenVocab {
    /// `bytes[id]` is the text token `id` decodes to; `special[id]` marks control
    /// tokens, which a grammar never produces; `eog` lists the end-of-generation
    /// ids, allowed exactly when the grammar can end.
    pub fn new(bytes: Vec<Vec<u8>>, special: Vec<bool>, eog: &[u32]) -> Self {
        let mut e = vec![false; bytes.len()];
        for &t in eog { if let Some(x) = e.get_mut(t as usize) { *x = true; } }
        TokenVocab { special, eog: e, bytes }
    }

    /// The vocabulary of a GGUF tokenizer. `n_vocab` is the model's logit count,
    /// which can exceed the tokenizer's (padded output rows decode to nothing and
    /// are never allowed); `eog` is the model's end-of-generation set.
    pub fn from_bpe(bpe: &ojas_tokenize::Bpe, n_vocab: usize, eog: &[u32]) -> Self {
        let n = n_vocab.max(bpe.len());
        let bytes = (0..n).map(|i| if i < bpe.len() { bpe.decode_bytes(i) } else { Vec::new() }).collect();
        let special = (0..n).map(|i| bpe.is_control(i)).collect();
        TokenVocab::new(bytes, special, eog)
    }

    pub fn len(&self) -> usize { self.bytes.len() }
    pub fn is_empty(&self) -> bool { self.bytes.is_empty() }
}

/// A grammar applied to generation: an [`ojas_infer::LogitProcessor`] that allows
/// a token only when the grammar can continue with its bytes, and an
/// end-of-generation token only when the grammar can end.
pub struct GrammarProcessor {
    grammar: Arc<Grammar>,
    vocab: Arc<TokenVocab>,
    state: State,
    /// The state after the last token `allows` checked, reused by `accept`.
    checked: Option<(u32, State)>,
    ended: bool,
}

impl GrammarProcessor {
    pub fn new(grammar: Arc<Grammar>, vocab: Arc<TokenVocab>) -> Self {
        let state = grammar.start();
        GrammarProcessor { grammar, vocab, state, checked: None, ended: false }
    }

    fn next_state(&self, token: u32) -> Option<State> {
        let t = token as usize;
        if t >= self.vocab.len() || self.vocab.special[t] || self.vocab.bytes[t].is_empty() { return None; }
        self.grammar.feed(&self.state, &self.vocab.bytes[t])
    }
}

impl ojas_infer::LogitProcessor for GrammarProcessor {
    fn allows(&mut self, token: u32) -> bool {
        if self.vocab.eog.get(token as usize).copied().unwrap_or(false) { return self.state.accepting(); }
        match self.next_state(token) {
            Some(s) => { self.checked = Some((token, s)); true }
            None => false,
        }
    }

    fn accept(&mut self, token: u32) {
        if self.vocab.eog.get(token as usize).copied().unwrap_or(false) { self.ended = true; return; }
        let next = match self.checked.take() {
            Some((t, s)) if t == token => Some(s),
            _ => self.next_state(token),
        };
        match next {
            Some(s) => self.state = s,
            None => self.ended = true,
        }
    }

    fn finished(&self) -> bool { self.ended || self.state.finished() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_infer::LogitProcessor;

    fn g(src: &str) -> Grammar { Grammar::parse(src).unwrap() }

    #[test]
    fn accepts_exactly_the_language() {
        let gr = g("root ::= \"a\" [0-9]+ ( \"x\" | \"yz\" )?");
        for ok in ["a1", "a123", "a7x", "a09yz"] { assert!(gr.accepts(ok), "{ok}"); }
        for bad in ["a", "ax", "a1y", "a1xx", "b1", ""] { assert!(!gr.accepts(bad), "{bad}"); }
    }

    #[test]
    fn bounded_repetition_counts() {
        let gr = g("root ::= [a-c]{2,3}");
        assert!(!gr.accepts("a") && gr.accepts("ab") && gr.accepts("abc") && !gr.accepts("abca"));
        let exact = g("root ::= \"z\"{3}");
        assert!(exact.accepts("zzz") && !exact.accepts("zz") && !exact.accepts("zzzz"));
    }

    #[test]
    fn negated_classes_and_any_char_cover_unicode() {
        let gr = g("root ::= \"\\\"\" [^\"\\\\]* \"\\\"\" .");
        assert!(gr.accepts("\"héllo → 世界\"!"));
        assert!(!gr.accepts("\"a\"b\"!"));
    }

    #[test]
    fn recursive_rules() {
        let gr = g("root ::= item\nitem ::= \"(\" item* \")\"");
        assert!(gr.accepts("(()(()))") && !gr.accepts("(()") && !gr.accepts("())"));
    }

    #[test]
    fn code_points_split_across_tokens() {
        let gr = g("root ::= \"é\" \"!\"");
        let bytes = "é!".as_bytes();
        let s1 = gr.feed(&gr.start(), &bytes[..1]).expect("first byte of é is a valid prefix");
        assert!(!s1.accepting());
        let s2 = gr.feed(&s1, &bytes[1..]).unwrap();
        assert!(s2.accepting() && s2.finished());
        assert!(gr.feed(&gr.start(), &[0xFF]).is_none());
    }

    #[test]
    fn a_partial_character_is_allowed_only_where_it_can_complete() {
        let ascii = g("root ::= \"{\" [a-z]+");
        assert!(ascii.feed(&ascii.start(), &[0xEA]).is_none(), "no character starting 0xEA fits here");
        let accented = g("root ::= \"é\"");
        assert!(accented.feed(&accented.start(), &[0xC3]).is_some(), "é is C3 A9");
        assert!(accented.feed(&accented.start(), &[0xC4]).is_none());
        assert!(accented.feed(&accented.start(), &[0xC3, 0xA8]).is_none(), "è is not é");
        let not_quote = g("root ::= [^\"]");
        assert!(not_quote.feed(&not_quote.start(), &[0xE4, 0xB8]).is_some());
        let gap = g("root ::= [^a-z\\u0080-\\u00ff]");
        assert!(gap.feed(&gap.start(), &[0xC3]).is_none(), "every code point from C3 is excluded");
    }

    #[test]
    fn only_strict_utf8_is_accepted() {
        let brace = g("root ::= \"{\"");
        assert!(brace.feed(&brace.start(), &[0xF0, 0x80, 0x81, 0xBB]).is_none(), "overlong form of {{");
        assert!(brace.feed(&brace.start(), &[0xF0, 0x80]).is_none(), "an overlong prefix is refused at once");
        let any = g("root ::= .");
        assert!(any.feed(&any.start(), &[0xE0, 0x9F]).is_none(), "overlong three-byte prefix");
        assert!(any.feed(&any.start(), &[0xED, 0xA0, 0x80]).is_none(), "surrogate");
        assert!(any.feed(&any.start(), &[0xF4, 0x90, 0x80, 0x80]).is_none(), "past U+10FFFF");
        assert!(any.accepts("世") && any.accepts("😀") && any.accepts("é"));
    }

    #[test]
    fn processor_gates_tokens_and_end_of_generation() {
        // ids: 0 "{", 1 "}", 2 "{}", 3 "x", 4 <eog> (special), 5 <ctrl> (special)
        let vocab = Arc::new(TokenVocab::new(
            vec![b"{".to_vec(), b"}".to_vec(), b"{}".to_vec(), b"x".to_vec(), b"<|end|>".to_vec(), b"<|c|>".to_vec()],
            vec![false, false, false, false, true, true], &[4]));
        let gram = Arc::new(g("root ::= \"{\" \"}\" \" \"?"));
        let mut p = GrammarProcessor::new(gram, vocab);
        assert!(p.allows(0) && p.allows(2) && !p.allows(1) && !p.allows(3) && !p.allows(4) && !p.allows(5));
        p.accept(2);
        assert!(p.allows(4), "end of generation is allowed once the grammar can end");
        assert!(!p.finished(), "a trailing space may still follow");
        p.accept(4);
        assert!(p.finished());
    }
}
