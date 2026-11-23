//! Segment-level transcription over [`CpuWhisper`]: a chosen or detected
//! language, an optional text prompt, timestamp tokens under the reference's
//! pairing rules, temperature fallback, and windowing that seeks to the last
//! timestamp of each window so long audio is neither cut mid-word nor dropped.
//!
//! Token ids the model's metadata does not carry (language tokens, the
//! previous-text marker, the no-speech marker) come from the caller, who has
//! the vocabulary.

use crate::CpuWhisper;

/// The sample rate the mel frontend expects.
pub const SAMPLE_RATE: usize = 16_000;
/// One model window, in samples.
pub const WINDOW: usize = 30 * SAMPLE_RATE;
/// Seconds per timestamp token step.
const TICK: f32 = 0.02;
/// A window this likely to be silence, decoded this unsurely, is skipped.
const NO_SPEECH: f32 = 0.6;
/// A decode below this mean log-probability is retried warmer.
const MIN_LOGPROB: f32 = -1.0;

pub struct DecodeOpts<'a> {
    /// The language token, or `None` to detect it among `languages`.
    pub language: Option<u32>,
    /// Every language token of the vocabulary; empty for a one-language model,
    /// whose prefix carries neither a language nor a task token.
    pub languages: &'a [u32],
    /// Text the transcript should continue from, as tokens.
    pub prompt: &'a [u32],
    /// The token that opens previous text; without it the prompt is not used.
    pub sot_prev: Option<u32>,
    /// The token whose probability after start-of-transcript marks silence.
    pub no_speech: Option<u32>,
    /// Decode timestamp tokens. Off, every window is one segment.
    pub timestamps: bool,
    /// 0 is greedy. A decode that fails its checks is retried 0.2 warmer, up to 1.
    pub temperature: f32,
}

pub struct Segment {
    /// Start of the window this segment was decoded in, seconds.
    pub seek: f32,
    pub start: f32,
    pub end: f32,
    /// Text tokens only.
    pub tokens: Vec<u32>,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
}

pub struct Transcript {
    /// The language token used, when the model has them.
    pub language: Option<u32>,
    pub segments: Vec<Segment>,
    /// Windows encoded and decoded.
    pub windows: usize,
}

/// Which tokens the timestamp rules speak of.
#[derive(Clone, Copy)]
struct Rules {
    eot: usize,
    ts_begin: usize,
    timestamps: bool,
}

/// One decode of one window.
struct Pass {
    /// Sampled tokens, timestamps included, without the end token.
    tokens: Vec<u32>,
    avg_logprob: f32,
    no_speech_prob: f32,
    temperature: f32,
}

fn log_sum_exp(v: &[f32]) -> f32 {
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if mx == f32::NEG_INFINITY { return mx; }
    mx + v.iter().map(|x| (x - mx).exp()).sum::<f32>().ln()
}

/// Mask what may not follow `sampled`: markers always; timestamps when they
/// are off; and, when on, the reference's rules: a window opens with a
/// timestamp no later than 1 s, timestamps come in pairs and never decrease,
/// and once timestamps together outweigh every text token one is forced.
fn mask(logits: &mut [f32], sampled: &[u32], r: Rules) {
    const OFF: f32 = f32::NEG_INFINITY;
    let (eot, ts) = (r.eot, r.ts_begin.min(logits.len()));
    for l in &mut logits[(eot + 1).min(ts)..ts] { *l = OFF; }
    if !r.timestamps {
        for l in &mut logits[ts..] { *l = OFF; }
        return;
    }
    let is_ts = |t: u32| t as usize >= ts;
    let n = sampled.len();
    let last = n >= 1 && is_ts(sampled[n - 1]);
    let before = n < 2 || is_ts(sampled[n - 2]);
    if last {
        if before { for l in &mut logits[ts..] { *l = OFF; } } else { for l in &mut logits[..eot] { *l = OFF; } }
    }
    if let Some(&prev) = sampled.iter().rev().find(|t| is_ts(**t)) {
        // A closing timestamp may repeat as the next segment's opening one.
        let floor = if last && !before { prev as usize } else { prev as usize + 1 };
        let floor = floor.min(logits.len());
        for l in &mut logits[ts..floor] { *l = OFF; }
    }
    if n == 0 {
        for l in &mut logits[..ts] { *l = OFF; }
        let latest = ts + (1.0 / TICK) as usize;
        if latest + 1 < logits.len() { for l in &mut logits[latest + 1..] { *l = OFF; } }
    }
    let text = logits[..ts].iter().cloned().fold(OFF, f32::max);
    if log_sum_exp(&logits[ts..]) > text {
        for l in &mut logits[..ts] { *l = OFF; }
    }
}

/// Cut a window's tokens into segments at each pair of consecutive timestamps:
/// `(start, end, text tokens)` in seconds from the window's start, and how far
/// to move on when the window ended mid-segment (`None`: the whole window).
fn split(tokens: &[u32], r: Rules, window_secs: f32) -> (Vec<(f32, f32, Vec<u32>)>, Option<f32>) {
    let is_ts = |t: u32| t as usize >= r.ts_begin;
    let secs = |t: u32| (t as usize - r.ts_begin) as f32 * TICK;
    let text = |s: &[u32]| s.iter().copied().filter(|t| (*t as usize) < r.eot).collect::<Vec<u32>>();
    let n = tokens.len();
    let mut cuts: Vec<usize> = (1..n).filter(|&i| is_ts(tokens[i - 1]) && is_ts(tokens[i])).collect();
    let mut out = Vec::new();
    if cuts.is_empty() {
        let end = tokens.iter().rev().find(|t| is_ts(**t)).map(|t| secs(*t)).filter(|e| *e > 0.0).unwrap_or(window_secs);
        let start = tokens.first().filter(|t| is_ts(**t)).map(|t| secs(*t)).unwrap_or(0.0);
        out.push((start.min(end), end, text(tokens)));
        return (out, None);
    }
    // `.. text <|t|>` at the end closes the last segment; the window is spent.
    let closed = is_ts(tokens[n - 1]) && !is_ts(tokens[n - 2]);
    if closed { cuts.push(n); }
    let mut from = 0;
    for &cut in &cuts {
        let piece = &tokens[from..cut];
        let start = piece.iter().find(|t| is_ts(**t)).map(|t| secs(*t)).unwrap_or(0.0);
        let end = piece.iter().rev().find(|t| is_ts(**t)).map(|t| secs(*t)).unwrap_or(window_secs);
        out.push((start, end.max(start), text(piece)));
        from = cut;
    }
    let advance = (!closed).then(|| secs(tokens[from - 1]));
    (out, advance)
}

/// Whether text has fallen into a loop: few of its token trigrams are distinct.
fn looped(tokens: &[u32], r: Rules) -> bool {
    let text: Vec<u32> = tokens.iter().copied().filter(|t| (*t as usize) < r.eot).collect();
    if text.len() < 32 { return false; }
    let distinct: std::collections::HashSet<&[u32]> = text.windows(3).collect();
    (distinct.len() as f32) < 0.4 * (text.len() - 2) as f32
}

/// xorshift64*, uniform in [0, 1).
fn uniform(state: &mut u64) -> f32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
}

fn pick(logits: &[f32], temperature: f32, rng: &mut u64) -> usize {
    if temperature <= 0.0 { return crate::cpu_math::argmax(logits); }
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f32> = logits.iter().map(|l| ((l - mx) / temperature).exp()).collect();
    let mut at = uniform(rng) * p.iter().sum::<f32>();
    for (i, w) in p.iter().enumerate() {
        at -= w;
        if at <= 0.0 && *w > 0.0 { return i; }
    }
    crate::cpu_math::argmax(logits)
}

/// The language whose token is likeliest right after start-of-transcript.
fn detect_language(m: &CpuWhisper, languages: &[u32]) -> Option<u32> {
    m.reset_decoder();
    let logits = m.logits(&m.step_hidden(m.sot, 0));
    languages.iter().copied().filter(|l| (*l as usize) < logits.len())
        .max_by(|a, b| logits[*a as usize].total_cmp(&logits[*b as usize]))
}

/// Decode the encoded window once at `temperature`.
fn decode_window(m: &CpuWhisper, o: &DecodeOpts, language: Option<u32>, temperature: f32, rng: &mut u64) -> Pass {
    let r = Rules { eot: m.eot as usize, ts_begin: m.ts_begin as usize, timestamps: o.timestamps };
    let budget = m.n_text_ctx / 2;
    let mut prefix = Vec::new();
    if let (Some(prev), false) = (o.sot_prev, o.prompt.is_empty()) {
        prefix.push(prev);
        prefix.extend_from_slice(&o.prompt[o.prompt.len().saturating_sub(budget.saturating_sub(1))..]);
    }
    let sot_at = prefix.len();
    prefix.push(m.sot);
    if let Some(l) = language { prefix.extend([l, m.transcribe]); }
    if !o.timestamps { prefix.push(m.no_timestamps); }

    m.reset_decoder();
    let (mut h, mut no_speech_prob) = (Vec::new(), 0.0);
    for (pos, &t) in prefix.iter().enumerate() {
        h = m.step_hidden(t, pos);
        if let (true, Some(ns)) = (pos == sot_at, o.no_speech) {
            let logits = m.logits(&h);
            no_speech_prob = (logits[ns as usize] - log_sum_exp(&logits)).exp();
        }
    }
    let (mut tokens, mut sum, mut steps, mut pos) = (Vec::new(), 0.0f32, 0usize, prefix.len());
    loop {
        let mut logits = m.logits(&h);
        mask(&mut logits, &tokens, r);
        let tok = pick(&logits, temperature, rng);
        sum += logits[tok] - log_sum_exp(&logits);
        steps += 1;
        if tok == r.eot { break; }
        tokens.push(tok as u32);
        if tokens.len() >= budget || pos >= m.n_text_ctx { break; }
        h = m.step_hidden(tok as u32, pos);
        pos += 1;
    }
    Pass { tokens, avg_logprob: sum / steps as f32, no_speech_prob, temperature }
}

/// Decode the encoded window, warmer each time the result is unsure or loops.
fn decode_checked(m: &CpuWhisper, o: &DecodeOpts, language: Option<u32>, rng: &mut u64) -> Pass {
    let r = Rules { eot: m.eot as usize, ts_begin: m.ts_begin as usize, timestamps: o.timestamps };
    let mut temperature = o.temperature.clamp(0.0, 1.0);
    loop {
        let pass = decode_window(m, o, language, temperature, rng);
        let silent = pass.no_speech_prob > NO_SPEECH && pass.avg_logprob < MIN_LOGPROB;
        let failed = !silent && (pass.avg_logprob < MIN_LOGPROB || looped(&pass.tokens, r));
        temperature += 0.2;
        if !failed || temperature > 1.0 + 1e-3 { return pass; }
    }
}

/// Transcribe 16 kHz mono PCM of any length.
pub fn transcribe(m: &CpuWhisper, pcm: &[f32], o: &DecodeOpts) -> Transcript {
    let r = Rules { eot: m.eot as usize, ts_begin: m.ts_begin as usize, timestamps: o.timestamps };
    let rate = SAMPLE_RATE as f32;
    let mut language = o.language;
    let (mut segments, mut seek, mut windows) = (Vec::new(), 0usize, 0usize);
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    while seek < pcm.len() {
        let end = (seek + WINDOW).min(pcm.len());
        // A sliver left after the last timestamp holds no word.
        if windows > 0 && end - seek < SAMPLE_RATE / 10 { break; }
        // A window without signal (below -80 dBFS) holds no speech; the model would invent some.
        if pcm[seek..end].iter().all(|x| x.abs() < 1e-4) {
            seek = end;
            continue;
        }
        m.encode(&m.log_mel(&pcm[seek..end]));
        windows += 1;
        if language.is_none() && !o.languages.is_empty() { language = detect_language(m, o.languages); }
        let pass = decode_checked(m, o, language, &mut rng);
        let (at, window_secs) = (seek as f32 / rate, (end - seek) as f32 / rate);
        let mut advance = end - seek;
        if !(pass.no_speech_prob > NO_SPEECH && pass.avg_logprob < MIN_LOGPROB) {
            let (pieces, moved) = if o.timestamps { split(&pass.tokens, r, window_secs) }
                                  else { (vec![(0.0, window_secs, pass.tokens.clone())], None) };
            for (start, stop, tokens) in pieces {
                if tokens.is_empty() { continue; }
                segments.push(Segment {
                    seek: at, start: at + start.min(window_secs), end: at + stop.min(window_secs), tokens,
                    temperature: pass.temperature, avg_logprob: pass.avg_logprob, no_speech_prob: pass.no_speech_prob,
                });
            }
            // Resume at the last timestamp, moving at least a second so every window makes progress.
            if let Some(secs) = moved.filter(|s| *s > 0.0) {
                advance = advance.min(((secs * rate) as usize).max(SAMPLE_RATE));
            }
        }
        seek += advance;
    }
    Transcript { language, segments, windows }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A toy vocabulary: text 0..10, end 10, markers 11..15, timestamps 15.. (0.00, 0.02, ..).
    const R: Rules = Rules { eot: 10, ts_begin: 15, timestamps: true };
    fn ts(secs: f32) -> u32 { 15 + (secs / TICK).round() as u32 }
    fn open(logits: &[f32]) -> Vec<usize> { (0..logits.len()).filter(|i| logits[*i] > f32::NEG_INFINITY).collect() }
    fn flat() -> Vec<f32> { vec![0.0; 15 + 1501] }

    #[test]
    fn a_window_opens_with_an_early_timestamp() {
        let mut l = flat();
        mask(&mut l, &[], R);
        assert_eq!(open(&l), (15..=65).collect::<Vec<_>>());
    }

    #[test]
    fn without_timestamps_only_text_and_the_end_remain() {
        let mut l = flat();
        mask(&mut l, &[], Rules { timestamps: false, ..R });
        assert_eq!(open(&l), (0..=10).collect::<Vec<_>>());
    }

    #[test]
    fn timestamps_pair_up_and_never_go_back() {
        // After an opening timestamp: text or the end, no timestamp. One strong text
        // logit keeps the summed timestamps from outweighing it.
        let mut l = flat();
        l[3] = 20.0;
        mask(&mut l, &[ts(0.0)], R);
        assert_eq!(open(&l), (0..=10).collect::<Vec<_>>());
        // After text: text, the end, or a timestamp later than the opening one.
        let mut l = flat();
        l[3] = 20.0;
        mask(&mut l, &[ts(1.0), 3], R);
        let o = open(&l);
        assert!(o.contains(&3) && o.contains(&10));
        assert!(!o.contains(&(ts(1.0) as usize)) && o.contains(&(ts(1.02) as usize)));
        // After a closing timestamp: no text; the same timestamp may open the next segment.
        let mut l = flat();
        l[10] = 20.0;
        mask(&mut l, &[ts(1.0), 3, ts(2.0)], R);
        let o = open(&l);
        assert_eq!((o[0], o[1]), (10, ts(2.0) as usize));
    }

    type Cut = (Vec<(f32, f32, Vec<u32>)>, Option<f32>);
    fn same(got: Cut, want: Cut) {
        let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
        assert_eq!(got.0.len(), want.0.len(), "{got:?}");
        for (g, w) in got.0.iter().zip(&want.0) { assert!(near(g.0, w.0) && near(g.1, w.1) && g.2 == w.2, "{g:?} vs {w:?}"); }
        assert!(match (got.1, want.1) { (Some(g), Some(w)) => near(g, w), (None, None) => true, _ => false }, "{:?}", got.1);
    }

    #[test]
    fn timestamps_that_outweigh_text_are_forced() {
        // Flat logits: 1500 timestamps together outweigh any one text token.
        let mut l = flat();
        mask(&mut l, &[ts(1.0), 3], R);
        assert!(open(&l).iter().all(|i| *i >= 15));
    }

    #[test]
    fn segments_are_cut_at_timestamp_pairs() {
        // Two closed segments: the window is spent.
        same(split(&[ts(0.0), 1, 2, ts(2.0), ts(2.5), 3, ts(4.0)], R, 30.0),
             (vec![(0.0, 2.0, vec![1, 2]), (2.5, 4.0, vec![3])], None));
        // The second segment is cut by the window: resume where the first closed.
        same(split(&[ts(0.0), 1, ts(26.0), ts(26.0), 3, 4], R, 30.0), (vec![(0.0, 26.0, vec![1])], Some(26.0)));
        // No pair at all: one segment to the last timestamp, or to the window's end.
        same(split(&[ts(0.5), 1, 2, ts(3.0)], R, 30.0), (vec![(0.5, 3.0, vec![1, 2])], None));
        same(split(&[ts(0.0), 1, 2], R, 12.0), (vec![(0.0, 12.0, vec![1, 2])], None));
    }

    #[test]
    fn a_loop_is_told_from_text() {
        assert!(looped(&[1, 2, 3].repeat(20), R));
        assert!(!looped(&[1, 2, 3].repeat(5), R), "too short to judge");
        let mut x = 1u32;
        let varied: Vec<u32> = (0..60).map(|_| { x = x.wrapping_mul(1664525).wrapping_add(1013904223); (x >> 16) % 10 }).collect();
        assert!(!looped(&varied, R));
    }

    #[test]
    fn greedy_picks_the_top_and_sampling_stays_in_the_support() {
        let mut l = vec![f32::NEG_INFINITY; 8];
        l[2] = 1.0;
        l[5] = 3.0;
        let mut rng = 7u64;
        assert_eq!(pick(&l, 0.0, &mut rng), 5);
        for _ in 0..200 { assert!(matches!(pick(&l, 1.0, &mut rng), 2 | 5)); }
    }
}
