//! CTC decoding for plate OCR (PaddleOCR-style recognition heads).
//!
//! The rec model emits `[T, C]` probabilities per crop, `C = dictionary + 1`
//! with the CTC blank at index 0 (the PaddleOCR convention). Greedy decode:
//! argmax per step, collapse repeats, drop blanks. Text comes out raw;
//! normalisation, plate grammar and confusables are the caller's job.

use anyhow::{ensure, Context, Result};

#[derive(Debug, Clone)]
pub struct PlateRead {
    /// Raw model output over the dictionary alphabet, no normalisation.
    pub text: String,
    /// One confidence per character of `text` (max prob over its CTC frames).
    pub char_conf: Vec<f32>,
    pub mean_conf: f32,
    /// Lines the splitter produced: 1 or 2, and 1 until two-line reading lands.
    pub lines: u8,
    /// Top-2 candidate characters per position of `text` with their probabilities,
    /// from the frame that produced the character. A matcher uses these to resolve
    /// O/0, B/8 and S/5 confusions.
    pub alternatives: Vec<[(char, f32); 2]>,
}

/// Dictionary: one character per line; CTC class `i + 1` maps to line `i`.
#[derive(Debug, Clone)]
pub struct Dictionary {
    chars: Vec<char>,
}

impl Dictionary {
    pub fn load(path: &str) -> Result<Dictionary> {
        let text = std::fs::read_to_string(path).with_context(|| format!("dictionary {path}"))?;
        Ok(Self::from_text(&text))
    }

    pub fn from_text(text: &str) -> Dictionary {
        let chars: Vec<char> = text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.chars().next().unwrap())
            .collect();
        Dictionary { chars }
    }

    /// `0-9A-Z`, the plate alphabet.
    pub fn plate_default() -> Dictionary {
        Dictionary { chars: ('0'..='9').chain('A'..='Z').collect() }
    }

    pub fn len(&self) -> usize {
        self.chars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// Number of CTC classes this dictionary expects (blank + chars).
    pub fn classes(&self) -> usize {
        self.chars.len() + 1
    }

    fn char_at(&self, class: usize) -> Option<char> {
        // class 0 = blank
        self.chars.get(class.checked_sub(1)?).copied()
    }
}

/// Greedy CTC over `[t_steps, classes]` probabilities.
pub fn ctc_greedy(probs: &[f32], t_steps: usize, classes: usize, dict: &Dictionary) -> Result<PlateRead> {
    ensure!(probs.len() == t_steps * classes, "CTC: {} values for [{t_steps}, {classes}]", probs.len());
    ensure!(
        classes == dict.classes(),
        "CTC: model emits {classes} classes but dictionary has {} (+1 blank) — wrong dictionary file?",
        dict.len()
    );
    // Top-2 non-blank classes of one probability row.
    let top2 = |row: &[f32]| -> [(usize, f32); 2] {
        let (mut a, mut pa, mut b, mut pb) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
        for (c, &p) in row.iter().enumerate().skip(1) {
            if p > pa {
                (b, pb) = (a, pa);
                (a, pa) = (c, p);
            } else if p > pb {
                (b, pb) = (c, p);
            }
        }
        [(a, pa), (b, pb)]
    };
    let alt_of = |row: &[f32]| -> [(char, f32); 2] {
        let [(a, pa), (b, pb)] = top2(row);
        [
            (dict.char_at(a).unwrap_or('\u{fffd}'), pa),
            (dict.char_at(b).unwrap_or('\u{fffd}'), pb),
        ]
    };

    let mut text = String::new();
    let mut char_conf: Vec<f32> = Vec::new();
    let mut alternatives: Vec<[(char, f32); 2]> = Vec::new();
    let mut prev = 0usize; // blank
    for t in 0..t_steps {
        let row = &probs[t * classes..(t + 1) * classes];
        let (mut arg, mut best) = (0usize, f32::NEG_INFINITY);
        for (c, &p) in row.iter().enumerate() {
            if p > best {
                best = p;
                arg = c;
            }
        }
        if arg != 0 {
            if arg == prev {
                // repeat frame of the same character: keep the best frame
                if let Some(last) = char_conf.last_mut() {
                    if best > *last {
                        *last = best;
                        *alternatives.last_mut().unwrap() = alt_of(row);
                    }
                }
            } else if let Some(ch) = dict.char_at(arg) {
                text.push(ch);
                char_conf.push(best);
                alternatives.push(alt_of(row));
            }
        }
        prev = arg;
    }
    let mean_conf = if char_conf.is_empty() { 0.0 } else { char_conf.iter().sum::<f32>() / char_conf.len() as f32 };
    Ok(PlateRead { text, char_conf, mean_conf, lines: 1, alternatives })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probs_from_classes(seq: &[usize], classes: usize, p: f32) -> Vec<f32> {
        let mut out = vec![(1.0 - p) / (classes - 1) as f32; seq.len() * classes];
        for (t, &c) in seq.iter().enumerate() {
            out[t * classes + c] = p;
        }
        out
    }

    #[test]
    fn collapses_repeats_and_blanks() {
        let dict = Dictionary::plate_default(); // 36 chars, 37 classes
        // G G _ J 0 0 _ 1 -> "GJ01"  (G=17, J=20, 0=1, 1=2)
        let seq = [17, 17, 0, 20, 1, 1, 0, 2];
        let probs = probs_from_classes(&seq, dict.classes(), 0.9);
        let r = ctc_greedy(&probs, seq.len(), dict.classes(), &dict).unwrap();
        assert_eq!(r.text, "GJ01");
        assert_eq!(r.char_conf.len(), 4);
        assert!((r.mean_conf - 0.9).abs() < 1e-5);
        // alternatives: one pair per character, first entry = the chosen char
        assert_eq!(r.alternatives.len(), 4);
        assert_eq!(r.alternatives[0][0].0, 'G');
        assert!((r.alternatives[0][0].1 - 0.9).abs() < 1e-5);
        assert!(r.alternatives[0][1].1 <= r.alternatives[0][0].1);
    }

    #[test]
    fn alternatives_capture_runner_up() {
        let dict = Dictionary::plate_default();
        let classes = dict.classes();
        // one frame: 0 ('0'->class 1) at 0.6, O (class 25) at 0.35 — classic confusable
        let mut probs = vec![0.0f32; classes];
        probs[1] = 0.6;
        probs[25] = 0.35;
        let r = ctc_greedy(&probs, 1, classes, &dict).unwrap();
        assert_eq!(r.text, "0");
        assert_eq!(r.alternatives[0][0], ('0', 0.6));
        assert_eq!(r.alternatives[0][1].0, dict_char(&dict, 25));
        assert!((r.alternatives[0][1].1 - 0.35).abs() < 1e-6);
    }

    fn dict_char(d: &Dictionary, class: usize) -> char {
        // test helper mirroring the internal mapping (class 1 = first dict char)
        ('0'..='9').chain('A'..='Z').nth(class - 1).unwrap_or_else(|| panic!("class {class} out of range for {}", d.len()))
    }

    #[test]
    fn blank_separated_double_letter_survives() {
        let dict = Dictionary::plate_default();
        // A _ A must stay "AA" (blank separates identical chars)  A=11
        let seq = [11, 0, 11];
        let probs = probs_from_classes(&seq, dict.classes(), 0.8);
        let r = ctc_greedy(&probs, 3, dict.classes(), &dict).unwrap();
        assert_eq!(r.text, "AA");
    }

    #[test]
    fn wrong_dictionary_is_a_named_error() {
        let dict = Dictionary::from_text("0\n1\n2\n");
        let err = ctc_greedy(&[0.0; 10], 2, 5, &dict).unwrap_err().to_string();
        assert!(err.contains("dictionary"), "{err}");
    }

    #[test]
    fn empty_read_has_zero_conf() {
        let dict = Dictionary::plate_default();
        let probs = probs_from_classes(&[0, 0, 0], dict.classes(), 0.99);
        let r = ctc_greedy(&probs, 3, dict.classes(), &dict).unwrap();
        assert_eq!(r.text, "");
        assert_eq!(r.mean_conf, 0.0);
    }
}
