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
    enc: HashMap<u8, char>,
    dec: HashMap<char, u8>,
    specials: Vec<(String, usize)>, // (literal string, id), longest first
    /// Ids of CONTROL(3)/USER_DEFINED(4) tokens. Their GGUF text is literal UTF-8
    /// rather than gpt2 byte-encoded, so it must not be mapped back through `dec`:
    /// a real 0x20 in the text has no `dec` entry (space is `\u{0120}` there) and is
    /// dropped. One shipped vocab has 71 such tokens, where `<div data-bbox="`
    /// (id 1168) came out as `<divdata-bbox="`. llama.cpp's `token_to_piece` copies
    /// USER_DEFINED text raw for the same reason.
    literal: std::collections::HashSet<usize>,
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
        let scores = g.float_arr("tokenizer.ggml.scores").cloned().unwrap_or_default();
        // special tokens: CONTROL(3) / USER_DEFINED(4) — matched atomically in raw text.
        let mut specials: Vec<(String, usize)> = Vec::new();
        let mut literal: std::collections::HashSet<usize> = std::collections::HashSet::new();
        if let Some(tt) = g.int_arr("tokenizer.ggml.token_type") {
            for (id, &t) in tt.iter().enumerate() {
                if (t == 3 || t == 4) && id < tokens.len() && !tokens[id].is_empty() {
                    specials.push((tokens[id].clone(), id));
                    literal.insert(id);
                }
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // longest first
        let (enc, dec) = byte_maps();
        Bpe { tokens, vocab, ranks, scores, spm, g4, enc, dec, specials, literal }
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
        for word in pretokenize(text) {
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

/// GPT2-family pre-tokenizer (hand-rolled; the `regex` crate lacks the lookahead
/// the real pattern uses). Splits into: contractions, optional-single-leading-space
/// + a same-category run (letters / digits / other), and whitespace runs (BPE then
/// merges "ĠĠ…"). Good enough that prompt tokenization matches real BPE on ordinary text.
fn pretokenize(text: &str) -> Vec<String> {
    let ch: Vec<char> = text.chars().collect();
    let n = ch.len();
    let cat = |c: char| -> u8 { if c.is_alphabetic() { 1 } else if c.is_numeric() { 2 } else { 3 } };
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < n {
        let c = ch[i];
        // contractions: 's 't 're 've 'm 'll 'd (case-insensitive, like llama-bpe)
        if c == '\'' && i + 1 < n {
            let two: String = ch[i + 1..(i + 3).min(n)].iter().collect::<String>().to_lowercase();
            let one = two.chars().next().unwrap_or(' ');
            let m2 = ["ll", "re", "ve"].iter().find(|p| two.starts_with(*p));
            if let Some(p) = m2 { out.push(format!("'{p}")); i += 3; continue; }
            if "std".contains(one) || one == 'm' {
                out.push(format!("'{one}")); i += 2; continue;
            }
        }
        if c.is_whitespace() {
            // maximal whitespace run; if followed by a non-ws char, the last ws char
            // attaches to the next word as its single leading space (GPT2 ` ?`).
            let s = i;
            while i < n && ch[i].is_whitespace() { i += 1; }
            let followed = i < n;
            let run_end = if followed { i - 1 } else { i };
            if run_end > s { out.push(ch[s..run_end].iter().collect()); }
            if followed {
                let lead = ch[run_end]; // the single attached space
                let k = cat(ch[i]);
                let ws_start = i;
                while i < n && !ch[i].is_whitespace() && cat(ch[i]) == k { i += 1; }
                let mut w = String::new();
                w.push(lead);
                w.extend(&ch[ws_start..i]);
                out.push(w);
            }
            continue;
        }
        // non-space run of one category
        let k = cat(c);
        let s = i;
        while i < n && !ch[i].is_whitespace() && cat(ch[i]) == k { i += 1; }
        out.push(ch[s..i].iter().collect());
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
