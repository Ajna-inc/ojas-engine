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
use std::collections::HashMap;
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

    /// This state without what a finished code point leaves behind, so two states
    /// that accept the same continuations compare equal.
    fn canonical(mut self) -> State {
        if self.need == 0 { (self.cp, self.min) = (0, 0); }
        self
    }
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
    /// `allowed_tokens` for states already asked about.
    masks: HashMap<State, Arc<[bool]>>,
}

/// States whose token masks a processor keeps; text inside one string or one
/// repetition returns to the same state token after token.
const MASKS: usize = 64;

/// States one pass over the vocabulary tabulates byte transitions for.
const TABULATED: usize = 4096;

impl GrammarProcessor {
    pub fn new(grammar: Arc<Grammar>, vocab: Arc<TokenVocab>) -> Self {
        let state = grammar.start();
        GrammarProcessor { grammar, vocab, state, checked: None, ended: false, masks: HashMap::new() }
    }

    /// Whether each token id may come next: what [`allows`](ojas_infer::LogitProcessor::allows)
    /// answers, for the whole vocabulary at once. Generation checks only the few
    /// candidates it ranks first; this is for a caller that needs the full set, such
    /// as one normalising probabilities over the allowed tokens.
    pub fn allowed_tokens(&mut self) -> Arc<[bool]> {
        let state = self.state.clone().canonical();
        if let Some(mask) = self.masks.get(&state) { return mask.clone(); }
        let mask: Arc<[bool]> = self.sweep(&state).into();
        if self.masks.len() >= MASKS { self.masks.clear(); }
        self.masks.insert(state, mask.clone());
        mask
    }

    /// `allowed_tokens` computed from `start`. Tokens share prefixes and most bytes
    /// lead back to a state already seen, so each (state, byte) step is taken
    /// through the grammar once and every token then walks a table.
    fn sweep(&self, start: &State) -> Vec<bool> {
        const UNKNOWN: u32 = u32::MAX;
        const REJECTED: u32 = u32::MAX - 1;
        let (grammar, vocab) = (&*self.grammar, &*self.vocab);
        let mut states = vec![start.clone()];
        let mut ids: HashMap<State, u32> = HashMap::from([(start.clone(), 0)]);
        let mut next: Vec<[u32; 256]> = vec![[UNKNOWN; 256]];
        let can_end = start.accepting();
        (0..vocab.len()).map(|t| {
            if vocab.eog[t] { return can_end; }
            if vocab.special[t] || vocab.bytes[t].is_empty() { return false; }
            let bytes = &vocab.bytes[t];
            let mut at = 0usize;
            for (i, &b) in bytes.iter().enumerate() {
                let mut to = next[at][b as usize];
                if to == UNKNOWN {
                    to = match grammar.feed(&states[at], &[b]).map(State::canonical) {
                        None => REJECTED,
                        Some(s) => match ids.get(&s) {
                            Some(&id) => id,
                            None if states.len() < TABULATED => {
                                let id = states.len() as u32;
                                ids.insert(s.clone(), id);
                                states.push(s);
                                next.push([UNKNOWN; 256]);
                                id
                            }
                            // Too many states to tabulate: finish this token directly.
                            None => return grammar.feed(&s, &bytes[i + 1..]).is_some(),
                        },
                    };
                    next[at][b as usize] = to;
                }
                if to == REJECTED { return false; }
                at = to as usize;
            }
            true
        }).collect()
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

    /// `allowed_tokens` against `allows` on every id, at every state along `path`.
    fn masks_agree(grammar: &str, pieces: &[&[u8]], path: &[u32]) {
        let mut bytes: Vec<Vec<u8>> = pieces.iter().map(|p| p.to_vec()).collect();
        let n = bytes.len() as u32;
        // Two specials after the text tokens: end of generation, and a control token.
        bytes.extend([b"<|end|>".to_vec(), b"<|c|>".to_vec(), Vec::new()]);
        let special = (0..n + 3).map(|t| t == n || t == n + 1).collect();
        let vocab = Arc::new(TokenVocab::new(bytes, special, &[n]));
        let mut p = GrammarProcessor::new(Arc::new(g(grammar)), vocab.clone());
        for step in 0..=path.len() {
            let mask = p.allowed_tokens();
            assert_eq!(mask.len(), vocab.len());
            for t in 0..vocab.len() as u32 {
                assert_eq!(mask[t as usize], p.allows(t), "{grammar}: token {t} after {:?}", &path[..step]);
            }
            assert!(Arc::ptr_eq(&mask, &p.allowed_tokens()), "a state asked about twice is not recomputed");
            if let Some(&t) = path.get(step) {
                assert!(mask[t as usize], "{grammar}: the path must be valid");
                p.accept(t);
            }
        }
    }

    #[test]
    fn the_token_mask_is_what_allows_answers() {
        let e = "é".as_bytes();
        let zh = "世".as_bytes();
        let pieces: Vec<&[u8]> = vec![
            b"{", b"}", b"{\"", b"\"", b"\":", b"a", b"ab", b"abc", b" ", b"1", b"12", b"yes", b"no", b"y", b"es", b"x\"y",
            e, &e[..1], &e[1..], zh, &zh[..1], &zh[..2], &zh[1..], &zh[2..], &[0xFF], b"a\xC3", "é世".as_bytes(),
        ];
        // A string of anything, entered and left; ids 2 `{"`, 6 `ab`, 17/18 the halves of é, 20/23 a split 世.
        masks_agree("root ::= \"{\" \"\\\"\" [^\"]* \"\\\"\" \":\" [0-9]+ \"}\"", &pieces, &[2, 6, 17, 18, 20, 22, 4, 10, 9, 1]);
        masks_agree("root ::= (\"yes\" | \"no\") \" \"?", &pieces, &[13, 14, 8]);
        masks_agree("root ::= [a-c]{2,5} \"é\"", &pieces, &[7, 6, 17, 18]);
        masks_agree("root ::= item\nitem ::= \"{\" item* \"}\"", &pieces, &[0, 0, 1, 0, 1, 1]);
    }

    #[test]
    fn the_token_mask_survives_more_states_than_it_tabulates() {
        // Every `a` consumed is a new state, more of them than one pass tabulates.
        let vocab = Arc::new(TokenVocab::new(vec![vec![b'a'; 3 * TABULATED], b"a".to_vec(), b"b".to_vec()], vec![false; 3], &[]));
        let grammar = format!("root ::= \"a\"{{{}}} \"b\"", 2 * TABULATED);
        let mut p = GrammarProcessor::new(Arc::new(g(&grammar)), vocab);
        assert_eq!(&*p.allowed_tokens(), [false, true, false]);
        let exact = Arc::new(TokenVocab::new(vec![vec![b'a'; 2 * TABULATED], b"b".to_vec()], vec![false; 2], &[]));
        let mut p = GrammarProcessor::new(Arc::new(g(&grammar)), exact);
        assert_eq!(&*p.allowed_tokens(), [true, false]);
        p.accept(0);
        assert_eq!(&*p.allowed_tokens(), [false, true]);
    }
}
