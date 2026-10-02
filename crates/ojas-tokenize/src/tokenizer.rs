//! BPE tokenizer (GPT2/Qwen/Llama/SPM families) + chat templates — one
//! implementation shared by every embedder.

use std::collections::HashMap;

// ---- byte-level tokenizer (GPT2 map) ----
pub fn byte_maps() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bs: Vec<u16> = Vec::new();
    for r in [(b'!' as u16, b'~' as u16), (0xA1, 0xAC), (0xAE, 0xFF)] {
        for b in r.0..=r.1 {
            bs.push(b);
        }
    }
    let mut cs = bs.clone();
    let mut n = 0u16;
    for b in 0u16..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut enc = HashMap::new();
    let mut dec = HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(*c as u32).unwrap();
        enc.insert(*b as u8, ch);
        dec.insert(ch, *b as u8);
    }
    (enc, dec)
}

/// Proper merge-based BPE tokenizer (GPT2/Qwen/Llama family) — greedy
/// longest-match mis-splits vs real BPE, so merges are required.
/// Pipeline: split off special tokens → GPT2 pre-tokenize → byte-encode →
/// apply merges in priority (rank) order within each pre-token.
/// `Default` is the empty tokenizer: no vocabulary, no merges. It encodes and
/// decodes nothing, which makes it a usable stand-in for code paths that take a
/// `&Bpe` without consulting it (a request that supplies token ids directly).
#[derive(Default)]
pub struct Bpe {
    tokens: Vec<String>,
    vocab: HashMap<String, usize>,
    ranks: HashMap<(String, String), usize>, // merge pair -> priority (lower = earlier)
    scores: Vec<f32>,               // SPM: per-token score (higher merges first)
    spm: bool,                      // true = SentencePiece score-BPE (model="llama", gemma3)
    g4: bool,                       // gemma-4: SPM-style BPE (▁ spaces, merge-by-rank, <0xXX> fallback)
    /// SPM-style BPE options, as Hugging Face's `Metaspace` pre-tokenizer spells them:
    /// prepend `▁` to each text segment that does not start with one
    /// (`tokenizer.ggml.add_space_prefix`), and split into words before every `▁` so
    /// merges never cross a word (`tokenizer.ggml.pre = "metaspace"`). Both are off for
    /// gemma-4, which does neither.
    g4_prefix: bool,
    g4_split: bool,
    enc: HashMap<u8, char>,
    dec: HashMap<char, u8>,
    specials: Vec<(String, usize)>, // (literal string, id), longest first
    /// The pre-tokenizer pattern, selected from `tokenizer.ggml.pre`.
    split: PreSplit,
    /// Ids of CONTROL(3)/USER_DEFINED(4) tokens. Their GGUF text is literal UTF-8
    /// rather than gpt2 byte-encoded, so it must not be mapped back through `dec`:
    /// a real 0x20 in the text has no `dec` entry (space is `\u{0120}` there) and is
    /// dropped. One shipped vocab has 71 such tokens, where `<div data-bbox="`
    /// (id 1168) came out as `<divdata-bbox="`. llama.cpp's `token_to_piece` copies
    /// USER_DEFINED text raw for the same reason.
    literal: std::collections::HashSet<usize>,
    /// Ids of CONTROL(3) and UNUSED(5) tokens: markers such as `<|im_end|>` that
    /// are never text, which constrained decoding must not produce.
    control: std::collections::HashSet<usize>,
}

impl Bpe {
    pub fn from_gguf(g: &ojas_formats::gguf::Gguf) -> Self {
        let tokens = g.str_arr("tokenizer.ggml.tokens").cloned().unwrap_or_default();
        let vocab: HashMap<String, usize> = tokens.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let mut ranks = HashMap::new();
        if let Some(merges) = g.str_arr("tokenizer.ggml.merges") {
            for (rank, m) in merges.iter().enumerate() {
                // "a b" — split on the first space (byte-encoded tokens never contain ' ').
                if let Some(sp) = m.find(' ') {
                    ranks.insert((m[..sp].to_string(), m[sp + 1..].to_string()), rank);
                }
            }
        }
        // SPM (SentencePiece): tokenizer.ggml.model is "llama"/"gemma*"; uses scores not merges.
        let model = match g.meta.get("tokenizer.ggml.model") {
            Some(ojas_formats::gguf::Meta::Str(s)) => s.clone(),
            _ => String::new(),
        };
        let spm = model == "llama" || model == "gemma" || model == "gemma2"; // score-based SPM
        let g4 = model == "gemma4";                                          // rank-based SPM-style BPE
        let pre = match g.meta.get("tokenizer.ggml.pre") {
            Some(ojas_formats::gguf::Meta::Str(p)) => p.clone(),
            _ => String::new(),
        };
        let g4_prefix = g4 && matches!(g.meta.get("tokenizer.ggml.add_space_prefix"), Some(ojas_formats::gguf::Meta::Bool(true)));
        let g4_split = g4 && pre == "metaspace";
        let scores = g.float_arr("tokenizer.ggml.scores").cloned().unwrap_or_default();
        // special tokens: CONTROL(3) / USER_DEFINED(4) — matched atomically in raw text.
        let mut specials: Vec<(String, usize)> = Vec::new();
        let mut literal: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut control: std::collections::HashSet<usize> = std::collections::HashSet::new();
        if let Some(tt) = g.int_arr("tokenizer.ggml.token_type") {
            for (id, &t) in tt.iter().enumerate() {
                if t == 3 || t == 5 { control.insert(id); }
                if (t == 3 || t == 4) && id < tokens.len() && !tokens[id].is_empty() {
                    specials.push((tokens[id].clone(), id));
                    literal.insert(id);
                }
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // longest first
        let (enc, dec) = byte_maps();
        let split = PreSplit::for_pre(&pre);
        Bpe { tokens, vocab, ranks, scores, spm, g4, g4_prefix, g4_split, enc, dec, specials, split, literal, control }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        let mut rest = text;
        let mut at_start = true; // SPM add_dummy_prefix applies at the very beginning
        while !rest.is_empty() {
            // earliest special-token occurrence in the remaining text
            let mut hit: Option<(usize, usize, usize)> = None; // (byte_pos, len, id)
            for (s, id) in &self.specials {
                if let Some(pos) = rest.find(s.as_str()) {
                    if hit.map_or(true, |(hp, _, _)| pos < hp) { hit = Some((pos, s.len(), *id)); }
                }
            }
            match hit {
                Some((pos, len, id)) => {
                    if pos > 0 { self.encode_seg(&rest[..pos], at_start, &mut ids); }
                    ids.push(id);
                    rest = &rest[pos + len..];
                    at_start = false;
                }
                None => { self.encode_seg(rest, at_start, &mut ids); break; }
            }
        }
        ids
    }

    fn encode_seg(&self, text: &str, first: bool, out: &mut Vec<usize>) {
        if self.g4 { self.encode_g4(text, first, out); }
        else if self.spm { self.encode_spm(text, first, out); }
        else { self.encode_ordinary(text, out); }
    }

    /// Gemma-4: SPM-style BPE. ▁ space substitution (plus dummy prefix) like SPM, but
    /// merges by rank (from tokenizer.ggml.merges) on raw UTF-8 chars — no GPT2 byte
    /// encoding, no word pre-splitting — with <0xXX> byte fallback for unknown symbols.
    fn encode_g4(&self, text: &str, _first: bool, out: &mut Vec<usize>) {
        // gemma-4 does not add a dummy ▁ prefix, unlike SPM: "The capital" → "The▁capital"
        // (first word has no ▁; internal spaces become ▁).
        let mut norm = String::new();
        for c in text.chars() { norm.push(if c == ' ' { '\u{2581}' } else { c }); }
        if self.g4_prefix && !norm.starts_with('\u{2581}') { norm.insert(0, '\u{2581}'); }
        if !self.g4_split { return self.bpe_g4_word(&norm, out); }
        // Split before every ▁, each word keeping its leading ▁ ("▁a▁▁b" → "▁a", "▁", "▁b").
        let mut start = 0usize;
        for (i, c) in norm.char_indices() {
            if c == '\u{2581}' && i > start { self.bpe_g4_word(&norm[start..i], out); start = i; }
        }
        if start < norm.len() { self.bpe_g4_word(&norm[start..], out); }
    }

    /// Rank-ordered merges over one span of characters, then `<0xXX>` byte fallback
    /// for any piece the vocabulary lacks.
    fn bpe_g4_word(&self, norm: &str, out: &mut Vec<usize>) {
        let mut parts: Vec<String> = norm.chars().map(|c| c.to_string()).collect();
        if parts.is_empty() { return; }
        while parts.len() > 1 {
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..parts.len() - 1 {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.map_or(true, |(br, _)| r < br) { best = Some((r, i)); }
                }
            }
            let Some((_, idx)) = best else { break };
            let merged = format!("{}{}", parts[idx], parts[idx + 1]);
            parts.splice(idx..idx + 2, [merged]);
        }
        for s in &parts {
            if let Some(&id) = self.vocab.get(s.as_str()) { out.push(id); }
            else { for b in s.bytes() { if let Some(&id) = self.vocab.get(&format!("<0x{b:02X}>")) { out.push(id); } } }
        }
    }

    /// SentencePiece: normalize spaces to ▁, then greedily merge the adjacent pair
    /// with the highest vocab score (priority queue) until none remain; byte-fallback
    /// (<0xXX>) for symbols not in vocab.
    fn encode_spm(&self, text: &str, first: bool, out: &mut Vec<usize>) {
        let mut norm = String::new();
        if first { norm.push('\u{2581}'); } // add_dummy_prefix
        for c in text.chars() { norm.push(if c == ' ' { '\u{2581}' } else { c }); }
        let mut syms: Vec<String> = norm.chars().map(|c| c.to_string()).collect();
        if syms.is_empty() { return; }
        loop {
            let mut best: Option<(f32, usize)> = None; // (score, index)
            for i in 0..syms.len() - 1 {
                let merged = format!("{}{}", syms[i], syms[i + 1]);
                if let Some(&id) = self.vocab.get(&merged) {
                    let sc = self.scores.get(id).copied().unwrap_or(f32::MIN);
                    if best.map_or(true, |(bs, _)| sc > bs) { best = Some((sc, i)); }
                }
            }
            let Some((_, idx)) = best else { break };
            let merged = format!("{}{}", syms[idx], syms[idx + 1]);
            syms.splice(idx..idx + 2, [merged]);
            if syms.len() == 1 { break; }
        }
        for s in &syms {
            if let Some(&id) = self.vocab.get(s.as_str()) { out.push(id); }
            else {
                for b in s.bytes() {
                    if let Some(&id) = self.vocab.get(&format!("<0x{b:02X}>")) { out.push(id); }
                }
            }
        }
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<usize>) {
        for word in pretokenize(text, self.split) {
            // byte-encode the pre-token to the GPT2 byte-char alphabet
            let piece: String = word.bytes().map(|b| self.enc[&b]).collect();
            self.bpe_word(&piece, out);
        }
    }

    fn bpe_word(&self, piece: &str, out: &mut Vec<usize>) {
        let mut parts: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
        if parts.is_empty() { return; }
        while parts.len() > 1 {
            // find the adjacent pair with the lowest merge rank
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..parts.len() - 1 {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.map_or(true, |(br, _)| r < br) { best = Some((r, i)); }
                }
            }
            let Some((_, idx)) = best else { break };
            let merged = format!("{}{}", parts[idx], parts[idx + 1]);
            parts.splice(idx..idx + 2, [merged]);
        }
        for p in &parts {
            if let Some(&id) = self.vocab.get(p.as_str()) { out.push(id); }
            // unknown piece: byte-encoded chars are always in vocab for these models,
            // so a miss is rare; silently drop.
        }
    }

    /// Raw bytes for one token. A multi-byte UTF-8 character is routinely split
    /// across two tokens, so anything streaming token-by-token must concatenate
    /// these and decode at the boundary — `decode` alone turns each half into
    /// U+FFFD. See `Utf8Stream`.
    pub fn decode_bytes(&self, id: usize) -> Vec<u8> {
        let Some(tok) = self.tokens.get(id) else { return Vec::new() };
        if self.literal.contains(&id) { return tok.as_bytes().to_vec(); }
        if self.spm || self.g4 {
            if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                    return vec![b];
                }
            }
            return tok.replace('\u{2581}', " ").into_bytes();
        }
        let mut bytes = Vec::new();
        for ch in tok.chars() {
            if let Some(&b) = self.dec.get(&ch) {
                bytes.push(b);
            }
        }
        bytes
    }

    /// Number of token ids in the vocabulary.
    pub fn len(&self) -> usize { self.tokens.len() }

    pub fn is_empty(&self) -> bool { self.tokens.is_empty() }

    /// Whether `id` is a control marker (`<|im_end|>`, `<|endoftext|>`, unused
    /// slots) rather than text.
    pub fn is_control(&self, id: usize) -> bool { self.control.contains(&id) }

    pub fn decode(&self, id: usize) -> String {
        let Some(tok) = self.tokens.get(id) else { return String::new() };
        if self.literal.contains(&id) { return tok.clone(); }
        if self.spm || self.g4 {
            // SPM tokens are literal UTF-8: ▁ = space; <0xXX> = raw byte fallback.
            if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                    return String::from_utf8_lossy(&[b]).into_owned();
                }
            }
            return tok.replace('\u{2581}', " ");
        }
        // gpt2 BPE: chars are byte-encoded; map back through the byte decoder.
        let mut bytes = Vec::new();
        for ch in tok.chars() {
            if let Some(&b) = self.dec.get(&ch) { bytes.push(b); }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Which BPE pre-tokenizer pattern a vocabulary uses, selected from
/// `tokenizer.ggml.pre`:
///
/// ```text
/// GPT-2    's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
/// Llama-3  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
///          | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// Qwen2    Llama-3 with \p{N}: one digit per piece
/// Qwen3.5  Qwen2 with [\p{L}\p{M}] wherever Qwen2 has \p{L}
/// ```
///
/// The differences are not cosmetic. Qwen merges were trained on single digits, on a
/// word keeping one leading punctuation character (`-call`) and on newline runs as one
/// piece (`\n\n`); split any other way the model reads token sequences it never saw.
/// Names not listed take the Llama-3 pattern, the common one among recent vocabularies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreSplit {
    Gpt2,
    #[default]
    Llama3,
    Qwen2,
    Qwen35,
}

impl PreSplit {
    const GPT2: &'static [&'static str] = &[
        "gpt-2", "phi-2", "modern-bert", "roberta-bpe", "olmo", "mpt", "jais", "trillion",
        "jina-es", "jina-de", "jina-v1-en", "jina-v2-es", "jina-v2-de", "jina-v2-code",
        "gigachat", "mellum", "exaone4", "a.x-4.0", "granite-docling",
    ];
    const QWEN2: &'static [&'static str] = &[
        "qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "stablelm2", "hunyuan", "solar-open",
    ];

    pub fn for_pre(pre: &str) -> Self {
        if pre == "qwen35" { Self::Qwen35 }
        else if Self::QWEN2.contains(&pre) { Self::Qwen2 }
        else if Self::GPT2.contains(&pre) { Self::Gpt2 }
        else { Self::Llama3 }
    }

    /// `\p{L}`, or `[\p{L}\p{M}]` for Qwen3.5.
    fn letter(self, c: char) -> bool {
        use unicode_general_category::{get_general_category, GeneralCategory as G};
        match get_general_category(c) {
            G::UppercaseLetter | G::LowercaseLetter | G::TitlecaseLetter | G::ModifierLetter
            | G::OtherLetter => true,
            G::NonspacingMark | G::SpacingMark | G::EnclosingMark => self == Self::Qwen35,
            _ => false,
        }
    }

    /// `\p{N}`.
    fn number(c: char) -> bool {
        use unicode_general_category::{get_general_category, GeneralCategory as G};
        matches!(get_general_category(c), G::DecimalNumber | G::LetterNumber | G::OtherNumber)
    }

    /// `[^\s\p{L}\p{N}]` (Qwen3.5: `[^\s\p{L}\p{M}\p{N}]`).
    fn other(self, c: char) -> bool { !c.is_whitespace() && !self.letter(c) && !Self::number(c) }

    /// Length in chars of the contraction starting at `ch[i]` (an apostrophe), or 0.
    /// GPT-2 matches lowercase only; the others either case, and the piece keeps the
    /// case it was written in.
    fn contraction_len(self, ch: &[char], i: usize) -> usize {
        let fold = |c: char| if self == Self::Gpt2 { c } else { c.to_ascii_lowercase() };
        let at = |k: usize| ch.get(i + k).copied().map(fold);
        match (at(1), at(2)) {
            (Some('l'), Some('l')) | (Some('r'), Some('e')) | (Some('v'), Some('e')) => 3,
            (Some('s' | 't' | 'm' | 'd'), _) => 2,
            _ => 0,
        }
    }

    /// Length in chars of the piece the pattern matches at `ch[i]`: its alternatives
    /// tried in order, each greedy, as the regex engine would.
    fn piece_len(self, ch: &[char], i: usize) -> usize {
        let n = ch.len();
        let run = |mut k: usize, f: &dyn Fn(char) -> bool| { while k < n && f(ch[k]) { k += 1; } k };
        let newline = |c: char| c == '\r' || c == '\n';
        let c = ch[i];
        if c == '\'' {
            let len = self.contraction_len(ch, i);
            if len > 0 { return len; }
        }
        let letter = |c: char| self.letter(c);
        let other = |c: char| self.other(c);
        let space = |c: char| c.is_whitespace();
        if self == Self::Gpt2 {
            // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`
            let s = if c == ' ' && i + 1 < n { i + 1 } else { i };
            for class in [&letter as &dyn Fn(char) -> bool, &Self::number, &other] {
                if class(ch[s]) { return run(s, class) - i; }
            }
            // `\s+(?!\S)`; a lone space before a word matches nothing and stays
            // its own piece, as unmatched text does.
            let j = run(i, &space);
            return if j == n || j - i < 2 { j - i } else { j - i - 1 };
        }
        // `[^\r\n\p{L}\p{N}]?\p{L}+`: the optional prefix excludes letters and numbers
        // only, so under Qwen3.5 a mark may lead too (it also matches the run itself).
        if letter(c) { return run(i, &letter) - i; }
        let pure_letter = PreSplit::Qwen2.letter(c);
        if !newline(c) && !pure_letter && !Self::number(c) && i + 1 < n && letter(ch[i + 1]) {
            return run(i + 1, &letter) - i;
        }
        // `\p{N}{1,3}` / `\p{N}`
        if Self::number(c) {
            let max = if self == Self::Llama3 { 3 } else { 1 };
            let mut k = i;
            while k < n && k - i < max && Self::number(ch[k]) { k += 1; }
            return k - i;
        }
        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        let s = if c == ' ' { i + 1 } else { i };
        if s < n && other(ch[s]) {
            let k = run(s, &other);
            return run(k, &newline) - i;
        }
        // Whitespace from here on. `\s*[\r\n]+` backtracks to the last newline in the
        // run; `\s+(?!\S)` leaves the run's last character for the next word; `\s+`.
        let j = run(i, &space);
        if let Some(last) = (i..j).rev().find(|&k| newline(ch[k])) { return last + 1 - i; }
        if j == n || j - i < 2 { j - i } else { j - i - 1 }
    }
}

/// Split `text` into the pieces BPE merges within, by the vocabulary's pattern.
fn pretokenize(text: &str, split: PreSplit) -> Vec<String> {
    let ch: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < ch.len() {
        let len = split.piece_len(&ch, i).max(1);
        out.push(ch[i..i + len].iter().collect());
        i += len;
    }
    out
}


/// Wrap a user message in Qwen/ChatML so instruct/chat/reasoning models behave
/// (a raw prompt makes them degenerate). The assistant turn is left open.
pub fn chatml(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

/// Arch-aware chat template. Llama3 uses header-id tokens + <|eot_id|>; everyone
/// else (Qwen2/Qwen3) uses ChatML. BOS (if required) is prepended separately.
pub fn chat_template(arch: &str, user: &str) -> String {
    match arch {
        "llama" => format!("<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
        a if a.starts_with("gemma") => format!("<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"),
        // gpt-oss "harmony" format: <|start|>role<|message|>…<|end|>. The assistant
        // header is opened on the `final` channel so the model emits its answer
        // directly (skipping the analysis/reasoning channel) and stops at <|return|>.
        "gpt-oss" => format!("<|start|>user<|message|>{user}<|end|><|start|>assistant<|channel|>final<|message|>"),
        _ => chatml(user),
    }
}

/// The EOS token string that ends an assistant turn for this arch.
pub fn chat_eos(arch: &str) -> &'static str {
    match arch {
        "llama" => "<|eot_id|>",
        a if a.starts_with("gemma") => "<end_of_turn>",
        "gpt-oss" => "<|return|>", // harmony end-of-final-message (GGUF eos = 200002)
        _ => "<|im_end|>",
    }
}


/// Arch-aware multi-turn transcript with an open assistant turn at the end.
/// `turns` are (role, text) with role ∈ user/assistant; other roles are treated
/// as user. ChatML for the Qwen family; Llama3 / Gemma get their own formats.
pub fn chat_transcript(arch: &str, system: &str, turns: &[(String, String)]) -> String {
    let mut s = String::new();
    match arch {
        "llama" => {
            if !system.is_empty() {
                s.push_str(&format!("<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"));
            }
            for (role, text) in turns {
                let r = if role == "assistant" { "assistant" } else { "user" };
                s.push_str(&format!("<|start_header_id|>{r}<|end_header_id|>\n\n{text}<|eot_id|>"));
            }
            s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        a if a.starts_with("gemma") => {
            // Gemma has no system role: fold the system prompt into the first turn.
            let mut sys = if system.is_empty() { None } else { Some(system) };
            for (role, text) in turns {
                let r = if role == "assistant" { "model" } else { "user" };
                let body = match (r, sys.take()) {
                    ("user", Some(sy)) => format!("{sy}\n\n{text}"),
                    _ => text.clone(),
                };
                s.push_str(&format!("<start_of_turn>{r}\n{body}<end_of_turn>\n"));
            }
            s.push_str("<start_of_turn>model\n");
        }
        "gpt-oss" => {
            // Harmony format. gpt-oss requires a system message declaring the reasoning
            // level and valid channels; without it, forcing a channel is out of
            // distribution and the model degenerates, especially with tool text in the
            // prompt. The app's system prompt (including the Hermes tool preamble) goes
            // into a `developer` message. The assistant answers on the `final` channel
            // and stops at <|return|>.
            s.push_str("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nReasoning: low\n# Valid channels: analysis, final. Channel must be included for every message.<|end|>");
            if !system.is_empty() {
                s.push_str(&format!("<|start|>developer<|message|>{system}<|end|>"));
            }
            for (role, text) in turns {
                if role == "assistant" {
                    s.push_str(&format!("<|start|>assistant<|channel|>final<|message|>{text}<|end|>"));
                } else {
                    s.push_str(&format!("<|start|>user<|message|>{text}<|end|>"));
                }
            }
            s.push_str("<|start|>assistant<|channel|>final<|message|>");
        }
        _ => {
            if !system.is_empty() {
                s.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
            }
            for (role, text) in turns {
                let r = if role == "assistant" { "assistant" } else { "user" };
                s.push_str(&format!("<|im_start|>{r}\n{text}<|im_end|>\n"));
            }
            s.push_str("<|im_start|>assistant\n");
        }
    }
    s
}

/// Token positions in `chat_transcript(arch, system, turns)` where a later request
/// is likely to stop matching it: the end of the system prompt, and the start of
/// each assistant turn, where a template may re-render the reply differently.
/// Each is the token prefix the full transcript shares with a shorter rendering,
/// so it holds for any template and any tokenizer. Ascending, without zeros.
pub fn transcript_boundaries(arch: &str, system: &str, turns: &[(String, String)],
                             encode: impl Fn(&str) -> Vec<u32>) -> Vec<usize> {
    let shared = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    let after_system = |next: &str| encode(&chat_transcript(arch, system, &[("user".into(), next.into())]));
    let full = encode(&chat_transcript(arch, system, turns));
    let mut marks = vec![shared(&after_system("a"), &after_system("b"))];
    for k in (0..turns.len()).filter(|&k| turns[k].0 != "assistant") {
        marks.push(shared(&full, &encode(&chat_transcript(arch, system, &turns[..=k]))));
    }
    marks.retain(|&m| m > 0);
    marks.sort_unstable();
    marks.dedup();
    marks
}

/// Token range `[start, end)` of turn `k`'s text in the tokens of
/// `chat_transcript(arch, system, turns)`: where the transcript stops matching one
/// whose turn `k` text has a character added at its start, and at its end. Holds for
/// any template; at either edge a token the text shares with the template is left
/// out, so the range never includes template tokens.
pub fn transcript_span(arch: &str, system: &str, turns: &[(String, String)], k: usize,
                       encode: impl Fn(&str) -> Vec<u32>) -> (usize, usize) {
    let shared = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    let full = encode(&chat_transcript(arch, system, turns));
    let altered = |text: String| {
        let mut t = turns.to_vec();
        t[k].1 = text;
        encode(&chat_transcript(arch, system, &t))
    };
    let start = shared(&full, &altered(format!("\u{7}{}", turns[k].1)));
    let end = shared(&full, &altered(format!("{}\u{7}", turns[k].1)));
    (start, end.max(start))
}

/// Reassembles a byte stream into valid UTF-8 as tokens arrive.
///
/// Byte-level BPE happily splits one character across token boundaries — a token
/// can end mid-sequence. Decoding each token on its own therefore emits U+FFFD
/// for both halves of, say, a non-breaking space. This holds the incomplete tail
/// back until the continuation bytes show up.
#[derive(Default)]
pub struct Utf8Stream {
    tail: Vec<u8>,
}

impl Utf8Stream {
    /// Feed one token's bytes; get back everything that is now decodable.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.tail.extend_from_slice(bytes);
        let mut out = String::new();
        // Loop rather than handle one error: a chunk can contain an invalid byte
        // followed by valid text, and returning after the first error would defer that
        // text to the next token, which may never come.
        loop {
            match std::str::from_utf8(&self.tail) {
                Ok(s) => {
                    out.push_str(s);
                    self.tail.clear();
                    return out;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(unsafe { std::str::from_utf8_unchecked(&self.tail[..good]) });
                    match e.error_len() {
                        // Truncated: hold the tail for the next token.
                        None => {
                            self.tail.drain(..good);
                            return out;
                        }
                        // Genuinely invalid: emit the replacement char and continue.
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.tail.drain(..good + n);
                        }
                    }
                }
            }
        }
    }

    /// Flush anything still buffered at end of stream (lossy — it never completed).
    pub fn finish(&mut self) -> String {
        if self.tail.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.tail).into_owned();
        self.tail.clear();
        out
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::{chat_transcript, transcript_boundaries, transcript_span};

    fn chars(s: &str) -> Vec<u32> { s.chars().map(|c| c as u32).collect() }

    #[test]
    fn a_turn_span_covers_exactly_its_text() {
        let turns: Vec<(String, String)> = [("user", "the page"), ("user", "a question")]
            .map(|(r, t)| (r.to_string(), t.to_string())).to_vec();
        let text = chat_transcript("qwen35", "be brief", &turns);
        let (s, e) = transcript_span("qwen35", "be brief", &turns, 0, chars);
        let chars: Vec<char> = text.chars().collect();
        assert_eq!(chars[s..e].iter().collect::<String>(), "the page");
    }

    #[test]
    fn marks_the_system_prompt_end_and_each_assistant_turn_start() {
        let turns: Vec<(String, String)> = [("user", "hi"), ("assistant", "hello"), ("user", "more")]
            .map(|(r, t)| (r.to_string(), t.to_string())).to_vec();
        let text = chat_transcript("qwen35", "be brief", &turns);
        let marks = transcript_boundaries("qwen35", "be brief", &turns, chars);
        let system_end = "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\n".chars().count();
        let first_reply = text.find("hello").unwrap();
        assert_eq!(marks, vec![system_end, first_reply, text.chars().count()]);
    }
}

#[cfg(test)]
mod utf8_tests {
    use super::Utf8Stream;

    #[test]
    fn splits_multibyte_across_pushes() {
        let mut s = Utf8Stream::default();
        // "€" is E2 82 AC — arriving one byte per token.
        assert_eq!(s.push(&[0xE2]), "");
        assert_eq!(s.push(&[0x82]), "");
        assert_eq!(s.push(&[0xAC]), "\u{20ac}");
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn passes_ascii_straight_through() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(b"ok"), "ok");
    }

    #[test]
    fn drops_genuinely_invalid_rather_than_stalling() {
        let mut s = Utf8Stream::default();
        let out = s.push(&[0xFF, b'a']);
        assert!(out.ends_with('a'), "got {out:?}");
    }
}

/// The model's complete end-of-generation set.
///
/// A model has a set of stop tokens rather than one: the GGUF names some in
/// metadata (`eos`, `eot`, `eom`) and the rest are architecture convention.
/// Treating only `tokenizer.ggml.eos_token_id` as terminal makes Qwen-family
/// generation run past its own ending and degenerate, because the argmax reaches
/// `<|endoftext|>` before `<|im_end|>` and a generator watching one id emits the
/// other and keeps going.
///
/// Two jobs use this, and both need the same set or they disagree: deciding when
/// generation has finished, and (when a caller asks to ignore the ending) deciding
/// which ids may not be selected at all.
pub fn eog_token_ids(g: &ojas_formats::gguf::Gguf, arch: &str) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    let mut push = |v: Option<u32>| {
        if let Some(v) = v {
            if !ids.contains(&v) {
                ids.push(v);
            }
        }
    };
    for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                "tokenizer.ggml.eom_token_id"] {
        push(g.meta_u32(key));
    }
    // Conventional end markers, resolved through the vocabulary, so a model without
    // one contributes nothing.
    let tokens = g.str_arr("tokenizer.ggml.tokens");
    if let Some(tokens) = tokens {
        let mut by_text = |want: &str| {
            if let Some(i) = tokens.iter().position(|t| t == want) {
                push(Some(i as u32));
            }
        };
        for marker in ["<|im_end|>", "<|endoftext|>", "<|end|>", "<|eot_id|>",
                       "<|end_of_text|>", "<end_of_turn>", "<|return|>"] {
            by_text(marker);
        }
        by_text(chat_eos(arch));
    }
    ids
}

#[cfg(test)]
mod pretokenize_tests {
    use super::{pretokenize, PreSplit};

    #[test]
    fn contractions_keep_their_case() {
        assert_eq!(pretokenize("THEY'RE here", PreSplit::Llama3), ["THEY", "'RE", " here"]);
    }

    #[test]
    fn gpt2_contractions_match_lowercase_only() {
        assert_eq!(pretokenize("THEY'RE", PreSplit::Gpt2), ["THEY", "'", "RE"]);
        assert_eq!(pretokenize("they're", PreSplit::Gpt2), ["they", "'re"]);
        assert_eq!(pretokenize("don't I'll", PreSplit::Gpt2), ["don", "'t", " I", "'ll"]);
    }

    #[test]
    fn gpt2_spaces_attach_only_as_a_literal_space() {
        assert_eq!(pretokenize("a\nb  c ", PreSplit::Gpt2), ["a", "\n", "b", " ", " c", " "]);
        assert_eq!(pretokenize("x 12 -y", PreSplit::Gpt2), ["x", " 12", " -", "y"]);
    }

    #[test]
    fn qwen_keeps_newline_runs_leading_punctuation_and_single_digits() {
        assert_eq!(pretokenize("a-call\n\nb 1440", PreSplit::Qwen2),
            ["a", "-call", "\n\n", "b", " ", "1", "4", "4", "0"]);
        assert_eq!(pretokenize("x  \n  y", PreSplit::Qwen2), ["x", "  \n", " ", " y"]);
        assert_eq!(pretokenize("ok!!\n\tz", PreSplit::Qwen2), ["ok", "!!\n", "\tz"]);
    }

    #[test]
    fn llama3_groups_digits_in_threes() {
        assert_eq!(pretokenize("1440", PreSplit::Llama3), ["144", "0"]);
    }

    #[test]
    fn qwen35_keeps_combining_marks_in_the_word() {
        // नमस्ते: the virama and the vowel sign are marks, not letters.
        assert_eq!(pretokenize("नमस्ते", PreSplit::Qwen35), ["नमस्ते"]);
        assert_eq!(pretokenize("नमस्ते", PreSplit::Qwen2), ["नमस", "्त", "े"]);
    }

    #[test]
    fn pre_names_select_the_pattern() {
        assert_eq!(PreSplit::for_pre("modern-bert"), PreSplit::Gpt2);
        assert_eq!(PreSplit::for_pre("gpt-2"), PreSplit::Gpt2);
        assert_eq!(PreSplit::for_pre("llama-bpe"), PreSplit::Llama3);
        assert_eq!(PreSplit::for_pre("qwen2"), PreSplit::Qwen2);
        assert_eq!(PreSplit::for_pre("qwen35"), PreSplit::Qwen35);
    }
}

#[cfg(test)]
mod metaspace_tests {
    use super::{byte_maps, Bpe, PreSplit};
    use std::collections::HashMap;

    /// A vocabulary where the merges could join words if nothing stopped them:
    /// `▁▁` exists, so "a  b" without the word split would merge the two spaces.
    fn bpe(prefix: bool, split: bool) -> Bpe {
        let tokens: Vec<String> = ["▁", "a", "b", "▁a", "▁b", "▁▁", "<0x21>"].iter().map(|s| s.to_string()).collect();
        let vocab: HashMap<String, usize> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
        let ranks: HashMap<(String, String), usize> = [("▁", "▁"), ("▁", "a"), ("▁", "b")].iter().enumerate()
            .map(|(r, (x, y))| ((x.to_string(), y.to_string()), r)).collect();
        let (enc, dec) = byte_maps();
        Bpe {
            tokens, vocab, ranks, scores: Vec::new(), spm: false, g4: true, g4_prefix: prefix, g4_split: split,
            enc, dec, specials: Vec::new(), split: PreSplit::default(), literal: Default::default(),
            control: Default::default(),
        }
    }

    fn pieces(b: &Bpe, text: &str) -> Vec<String> {
        b.encode(text).into_iter().map(|i| b.tokens[i].clone()).collect()
    }

    #[test]
    fn metaspace_prefixes_and_splits_words() {
        let b = bpe(true, true);
        assert_eq!(pieces(&b, "a  b"), ["▁a", "▁", "▁b"]);
        // An existing leading space is the prefix; no second one is added.
        assert_eq!(pieces(&b, " a"), ["▁a"]);
        // Unknown characters fall back to bytes.
        assert_eq!(pieces(&b, "a!"), ["▁a", "<0x21>"]);
    }

    #[test]
    fn gemma4_mode_is_unchanged() {
        // No prefix and no split: the ranks may merge across the double space.
        let b = bpe(false, false);
        assert_eq!(pieces(&b, "a  b"), ["a", "▁▁", "b"]);
    }
}
