//! SentencePiece tokenizer over the GGUF-embedded vocab
//! (`tokenizer.ggml.tokens` + `.scores` + `.token_type`) — what
//! encoder-decoder GGUFs ship.
//!
//! Encode is greedy pairwise merging (the reference SPM algorithm — see
//! [`Sp::encode`]): pieces score against the ▁-normalized text, unseen bytes
//! fall back to `<0xXX>` byte pieces, and control pieces (`<s>`, `</s>`, …)
//! never match plain text. Decode streams piece-by-piece (▁ → space, byte
//! pieces → raw bytes) which is what a token callback wants.

use ojas_formats::gguf::Gguf;
use std::collections::HashMap;

const SP_SPACE: char = '\u{2581}'; // ▁

// token_type values (the standard GGUF convention)
const T_CONTROL: i64 = 3;
const T_BYTE: i64 = 6;

pub struct Sp {
    pieces: Vec<String>,
    score: Vec<f32>,
    ttype: Vec<i64>,
    /// text-matchable pieces only (control pieces excluded)
    id_of: HashMap<String, u32>,
    byte_id: [i32; 256], // id of <0xXX>, -1 if absent
    unk: u32,
}

impl Sp {
    /// None if this GGUF has no SPM vocab (no scores array).
    pub fn from_gguf(g: &Gguf) -> Option<Sp> {
        let pieces = g.str_arr("tokenizer.ggml.tokens")?.clone();
        let score = g.f32_arr("tokenizer.ggml.scores")?.clone();
        let ttype: Vec<i64> = g.int_arr("tokenizer.ggml.token_type")
            .cloned()
            .unwrap_or_else(|| vec![1; pieces.len()]);
        let unk = g.meta_u32("tokenizer.ggml.unknown_token_id").unwrap_or(0);

        let mut id_of = HashMap::with_capacity(pieces.len());
        let mut byte_id = [-1i32; 256];
        for (i, p) in pieces.iter().enumerate() {
            match ttype.get(i).copied().unwrap_or(1) {
                T_CONTROL => {} // never matches plain text
                T_BYTE => {
                    if let Some(b) = parse_byte_piece(p) { byte_id[b as usize] = i as i32; }
                }
                _ => {
                    id_of.insert(p.clone(), i as u32);
                }
            }
        }
        Some(Sp { pieces, score, ttype, id_of, byte_id, unk })
    }

    /// SPM encode by greedy pairwise merging (the reference SPM algorithm):
    /// start from single characters and repeatedly merge the adjacent pair
    /// whose concatenation is the best-scoring vocab piece. Correct for both
    /// score conventions (unigram log-probs and BPE-mode −rank scores — this
    /// vocab is the latter, where Viterbi-sum picks pathological splits).
    /// Normalization: spaces → ▁ with a leading dummy prefix.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let norm: String = {
            let mut s = String::with_capacity(text.len() + 4);
            s.push(SP_SPACE);
            for c in text.chars() { s.push(if c == ' ' { SP_SPACE } else { c }); }
            s
        };
        // symbols = byte ranges over `norm`, one per char initially
        let mut sym: Vec<(usize, usize)> = Vec::new(); // (start, end)
        let mut it = norm.char_indices().peekable();
        while let Some((i, c)) = it.next() {
            sym.push((i, i + c.len_utf8()));
        }
        loop {
            let mut best: Option<(f32, usize)> = None; // (score, left index)
            for i in 0..sym.len().saturating_sub(1) {
                let merged = &norm[sym[i].0..sym[i + 1].1];
                if let Some(&id) = self.id_of.get(merged) {
                    let s = self.score[id as usize];
                    if best.map_or(true, |(bs, _)| s > bs) { best = Some((s, i)); }
                }
            }
            let Some((_, i)) = best else { break };
            sym[i] = (sym[i].0, sym[i + 1].1);
            sym.remove(i + 1);
        }
        // map symbols to ids; unmatched symbols fall back to byte pieces
        let mut ids = Vec::with_capacity(sym.len());
        for &(a, b) in &sym {
            if let Some(&id) = self.id_of.get(&norm[a..b]) {
                ids.push(id);
            } else {
                for &byte in norm[a..b].as_bytes() {
                    ids.push(if self.byte_id[byte as usize] >= 0 {
                        self.byte_id[byte as usize] as u32
                    } else {
                        self.unk
                    });
                }
            }
        }
        ids
    }

    /// Id of an exact piece, including control pieces (which never match plain
    /// text in encode) — for injecting special separators such as `<tools>`
    /// directly into a token stream.
    pub fn piece_id(&self, piece: &str) -> Option<u32> {
        self.pieces.iter().position(|p| p == piece).map(|i| i as u32)
    }

    /// Streaming decode of one token: ▁ → space, byte pieces → their byte
    /// (lossy across split UTF-8 sequences), control pieces → nothing.
    pub fn decode(&self, id: usize) -> String {
        if id >= self.pieces.len() { return String::new(); }
        match self.ttype[id] {
            T_CONTROL => String::new(),
            T_BYTE => parse_byte_piece(&self.pieces[id])
                .map(|b| String::from_utf8_lossy(&[b]).into_owned())
                .unwrap_or_default(),
            _ => self.pieces[id].replace(SP_SPACE, " "),
        }
    }
}

fn parse_byte_piece(p: &str) -> Option<u8> {
    // "<0xAB>"
    let hex = p.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}
