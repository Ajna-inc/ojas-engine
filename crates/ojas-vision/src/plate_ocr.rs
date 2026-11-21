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
    /// Per-timestep top-k, before the collapse. None unless the caller asked for
    /// it (`OcrCfg::top_k`, `ctc_greedy_topk`).
    pub steps: Option<CtcSteps>,
}

/// The `k` most probable classes of every CTC timestep, blank included: what a
/// caller reading one plate across many frames needs to align the reads and vote
/// per character, which the collapsed text cannot give back.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcSteps {
    /// Candidates kept per timestep: the requested k, capped at the class count.
    pub k: usize,
    /// `[t_steps * k]` (class, probability), each timestep's best first; equal
    /// probabilities order by class, so a step's first entry is the class the
    /// greedy decode took. Class 0 is the CTC blank (`CTC_BLANK`), any other
    /// class maps to its character through `Dictionary::char_at`.
    pub cands: Vec<(u32, f32)>,
}

/// The CTC blank's class index (the PaddleOCR convention).
pub const CTC_BLANK: u32 = 0;

impl CtcSteps {
    pub fn t_steps(&self) -> usize {
        self.cands.len() / self.k
    }

    /// Timestep `t`'s candidates, best first.
    pub fn step(&self, t: usize) -> &[(u32, f32)] {
        &self.cands[t * self.k..(t + 1) * self.k]
    }

    pub fn iter(&self) -> impl Iterator<Item = &[(u32, f32)]> {
        self.cands.chunks_exact(self.k)
    }
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

    /// The character of a CTC class; None for the blank (class 0) and out of range.
    pub fn char_at(&self, class: usize) -> Option<char> {
        self.chars.get(class.checked_sub(1)?).copied()
    }
}

/// Greedy CTC over `[t_steps, classes]` probabilities.
pub fn ctc_greedy(probs: &[f32], t_steps: usize, classes: usize, dict: &Dictionary) -> Result<PlateRead> {
    ctc_greedy_topk(probs, t_steps, classes, dict, 0)
}

/// `ctc_greedy` that also keeps each timestep's `top_k` classes in
/// `PlateRead::steps` (k is capped at the class count; 0 keeps nothing and costs
/// nothing). The read itself is the same either way.
pub fn ctc_greedy_topk(probs: &[f32], t_steps: usize, classes: usize, dict: &Dictionary, top_k: usize) -> Result<PlateRead> {
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
    let k = top_k.min(classes);
    let mut cands: Vec<(u32, f32)> = Vec::with_capacity(t_steps * k);
    let mut order: Vec<u32> = Vec::new();
    let mut prev = 0usize; // blank
    for t in 0..t_steps {
        let row = &probs[t * classes..(t + 1) * classes];
        if k > 0 {
            // Probability descending, then class ascending: the argmax below takes
            // the first maximum, so entry 0 is the class the decode acts on. A NaN
            // never wins the argmax, so it sorts last here.
            let key = |c: u32| if row[c as usize].is_nan() { f32::NEG_INFINITY } else { row[c as usize] };
            let by = |a: &u32, b: &u32| key(*b).total_cmp(&key(*a)).then(a.cmp(b));
            order.clear();
            order.extend(0..classes as u32);
            if k < classes {
                order.select_nth_unstable_by(k - 1, by);
            }
            order[..k].sort_unstable_by(by);
            cands.extend(order[..k].iter().map(|&c| (c, row[c as usize])));
        }
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
    let steps = (k > 0).then_some(CtcSteps { k, cands });
    Ok(PlateRead { text, char_conf, mean_conf, lines: 1, alternatives, steps })
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

    #[test]
    fn steps_are_off_unless_asked() {
        let dict = Dictionary::plate_default();
        let seq = [17, 17, 0, 20];
        let probs = probs_from_classes(&seq, dict.classes(), 0.9);
        let plain = ctc_greedy(&probs, seq.len(), dict.classes(), &dict).unwrap();
        assert!(plain.steps.is_none());
        assert!(ctc_greedy_topk(&probs, seq.len(), dict.classes(), &dict, 0).unwrap().steps.is_none());
        // asking for steps does not change the read
        let with = ctc_greedy_topk(&probs, seq.len(), dict.classes(), &dict, 3).unwrap();
        assert_eq!((with.text.as_str(), &with.char_conf, &with.alternatives), (plain.text.as_str(), &plain.char_conf, &plain.alternatives));
    }

    #[test]
    fn steps_keep_repeats_and_blanks() {
        let dict = Dictionary::plate_default();
        // G G _ J 0 0 _ 1: every frame survives, blanks as class 0
        let seq = [17, 17, 0, 20, 1, 1, 0, 2];
        let probs = probs_from_classes(&seq, dict.classes(), 0.9);
        let r = ctc_greedy_topk(&probs, seq.len(), dict.classes(), &dict, 1).unwrap();
        let s = r.steps.unwrap();
        assert_eq!((s.k, s.t_steps()), (1, seq.len()));
        let top: Vec<u32> = s.iter().map(|c| c[0].0).collect();
        assert_eq!(top, seq.iter().map(|&c| c as u32).collect::<Vec<_>>());
        assert!(s.iter().all(|c| (c[0].1 - 0.9).abs() < 1e-6));
        assert_eq!(s.step(2)[0].0, CTC_BLANK);
        assert_eq!(dict.char_at(s.step(2)[0].0 as usize), None);
        assert_eq!(dict.char_at(s.step(3)[0].0 as usize), Some('J'));
        // collapsing the top-1 track by the CTC rule gives the text back
        let mut text = String::new();
        let mut prev = CTC_BLANK;
        for c in top {
            if c != CTC_BLANK && c != prev {
                text.push(dict.char_at(c as usize).unwrap());
            }
            prev = c;
        }
        assert_eq!(text, r.text);
    }

    #[test]
    fn steps_rank_by_probability_with_blank_in_place() {
        let dict = Dictionary::from_text("A\nB\nC\n"); // 4 classes
        #[rustfmt::skip]
        let probs = [
            0.10, 0.60, 0.05, 0.25, // A, C, blank, B
            0.70, 0.10, 0.15, 0.05, // blank, B, A, C
        ];
        let r = ctc_greedy_topk(&probs, 2, 4, &dict, 3).unwrap();
        assert_eq!(r.text, "A");
        let s = r.steps.unwrap();
        assert_eq!(s.step(0), &[(1, 0.60), (3, 0.25), (0, 0.10)]);
        assert_eq!(s.step(1), &[(0, 0.70), (2, 0.15), (1, 0.10)]);
    }

    #[test]
    fn step_ties_order_by_class_and_agree_with_the_decode() {
        let dict = Dictionary::from_text("A\nB\nC\n");
        #[rustfmt::skip]
        let probs = [
            0.20, 0.30, 0.20, 0.30, // A ties C: the decode takes A
            0.25, 0.25, 0.25, 0.25, // all equal: the decode takes blank
            0.10, 0.10, 0.40, 0.40, // B ties C: the decode takes B
        ];
        let r = ctc_greedy_topk(&probs, 3, 4, &dict, 4).unwrap();
        assert_eq!(r.text, "AB");
        let s = r.steps.unwrap();
        assert_eq!(s.step(0), &[(1, 0.30), (3, 0.30), (0, 0.20), (2, 0.20)]);
        assert_eq!(s.step(1), &[(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)]);
        assert_eq!(s.step(2), &[(2, 0.40), (3, 0.40), (0, 0.10), (1, 0.10)]);
        // a cut through a tie keeps the lower classes
        let s2 = ctc_greedy_topk(&probs, 3, 4, &dict, 2).unwrap().steps.unwrap();
        assert_eq!(s2.step(1), &[(0, 0.25), (1, 0.25)]);
        assert_eq!(s2.step(2), &[(2, 0.40), (3, 0.40)]);
    }

    #[test]
    fn k_beyond_the_class_count_is_capped() {
        let dict = Dictionary::from_text("A\nB\nC\n");
        let probs = [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1];
        for k in [4usize, 5, 1000, usize::MAX] {
            let s = ctc_greedy_topk(&probs, 2, 4, &dict, k).unwrap().steps.unwrap();
            assert_eq!((s.k, s.cands.len(), s.t_steps()), (4, 8, 2));
            assert_eq!(s.step(0), &[(3, 0.4), (2, 0.3), (1, 0.2), (0, 0.1)]);
            // the whole row comes back, so each step sums to the row's mass
            assert!(s.iter().all(|c| (c.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-6));
        }
    }

    #[test]
    fn nan_sorts_last_and_no_timesteps_is_an_empty_record() {
        let dict = Dictionary::from_text("A\nB\n");
        let probs = [f32::NAN, 0.2, 0.7];
        let r = ctc_greedy_topk(&probs, 1, 3, &dict, 3).unwrap();
        assert_eq!(r.text, "B");
        let s = r.steps.unwrap();
        assert_eq!((s.step(0)[0], s.step(0)[1], s.step(0)[2].0), ((2, 0.7), (1, 0.2), 0));
        let empty = ctc_greedy_topk(&[], 0, 3, &dict, 2).unwrap().steps.unwrap();
        assert_eq!((empty.k, empty.t_steps(), empty.iter().count()), (2, 0, 0));
    }
}
