//! WAV decoding and resampling: RIFF/WAVE with integer PCM (8, 16, 24, 32 bit)
//! or float (32, 64 bit) samples, any channel count and sample rate, to mono
//! `f32` at the rate a model wants.

/// Why audio was not decoded.
#[derive(Debug, PartialEq)]
pub(crate) enum AudioError {
    /// A container or sample encoding this build does not decode (415).
    Unsupported(String),
    /// A WAV file that is cut short or contradicts itself (400).
    Invalid(String),
}

/// Decoded audio, channels averaged to one.
#[derive(Debug)]
pub(crate) struct Audio {
    pub(crate) rate: u32,
    pub(crate) samples: Vec<f32>,
}

/// The container `bytes` open with, by their magic numbers.
fn container(b: &[u8]) -> &'static str {
    let at = |i: usize, m: &[u8]| b.len() >= i + m.len() && &b[i..i + m.len()] == m;
    if at(0, b"RIFF") && at(8, b"WAVE") { "wav" }
    else if at(0, b"RF64") || at(0, b"BW64") { "RF64" }
    else if at(0, b"ID3") || (b.len() > 1 && b[0] == 0xFF && b[1] & 0xE0 == 0xE0) { "mp3" }
    else if at(4, b"ftyp") { "mp4/m4a" }
    else if at(0, b"OggS") { "ogg" }
    else if at(0, b"fLaC") { "flac" }
    else if at(0, b"FORM") && (at(8, b"AIFF") || at(8, b"AIFC")) { "aiff" }
    else if at(0, &[0x1A, 0x45, 0xDF, 0xA3]) { "webm/matroska" }
    else { "unknown" }
}

fn u16_at(b: &[u8], i: usize) -> u16 { u16::from_le_bytes([b[i], b[i + 1]]) }
fn u32_at(b: &[u8], i: usize) -> u32 { u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) }

/// Decode a WAV file. Any other container is `Unsupported`, named.
pub(crate) fn decode(b: &[u8]) -> Result<Audio, AudioError> {
    let invalid = |m: &str| AudioError::Invalid(format!("WAV: {m}"));
    match container(b) {
        "wav" => {}
        "unknown" => return Err(AudioError::Unsupported(
            "the audio is not a WAV file (no RIFF/WAVE header); send WAV with 16-bit, 24-bit or 32-bit float PCM".into())),
        other => return Err(AudioError::Unsupported(format!(
            "{other} audio is not decoded here; convert it to WAV (16-bit, 24-bit or 32-bit float PCM, any sample rate)"))),
    }
    // (format tag, channels, rate, block align, bits per sample)
    let mut fmt: Option<(u16, u16, u32, usize, u16)> = None;
    let mut at = 12usize;
    while at + 8 <= b.len() {
        let size = u32_at(b, at + 4) as usize;
        let body = at + 8;
        match &b[at..at + 4] {
            b"fmt " => {
                if size < 16 || body + 16 > b.len() { return Err(invalid("the fmt chunk is cut short")); }
                let mut tag = u16_at(b, body);
                // WAVE_FORMAT_EXTENSIBLE: the encoding is the sub-format's first two bytes.
                if tag == 0xFFFE {
                    if size < 26 || body + 26 > b.len() { return Err(invalid("the extensible fmt chunk is cut short")); }
                    tag = u16_at(b, body + 24);
                }
                fmt = Some((tag, u16_at(b, body + 2), u32_at(b, body + 4), u16_at(b, body + 12) as usize, u16_at(b, body + 14)));
            }
            b"data" => {
                let (tag, channels, rate, align, bits) = fmt.ok_or_else(|| invalid("the data chunk comes before the fmt chunk"))?;
                // A streamed file leaves the size 0 or all ones: the data runs to the end.
                let end = if size == 0 || size == u32::MAX as usize { b.len() } else { (body + size).min(b.len()) };
                return samples(&b[body..end], tag, channels, rate, align, bits);
            }
            _ => {}
        }
        at = body.saturating_add(size).saturating_add(size & 1);
    }
    Err(invalid("no data chunk"))
}

fn samples(data: &[u8], tag: u16, channels: u16, rate: u32, align: usize, bits: u16) -> Result<Audio, AudioError> {
    let invalid = |m: String| AudioError::Invalid(format!("WAV: {m}"));
    if channels == 0 || rate == 0 { return Err(invalid(format!("{channels} channels at {rate} Hz"))); }
    let read: fn(&[u8]) -> f32 = match (tag, bits) {
        (1, 8) => |s| (s[0] as f32 - 128.0) / 128.0,
        (1, 16) => |s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
        (1, 24) => |s| (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8_388_608.0,
        (1, 32) => |s| i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
        (3, 32) => |s| f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        (3, 64) => |s| f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]) as f32,
        (1 | 3, _) => return Err(AudioError::Unsupported(format!("WAV with {bits}-bit samples is not decoded here"))),
        _ => return Err(AudioError::Unsupported(format!(
            "WAV encoding {tag:#06x} is not decoded here; send PCM (16-bit, 24-bit) or 32-bit float"))),
    };
    let width = bits as usize / 8;
    let frame = width * channels as usize;
    // A container wider than the sample (24 bits in 4 bytes) is not handled.
    if align != 0 && align != frame { return Err(invalid(format!("block align {align} does not match {channels} x {bits}-bit"))); }
    let scale = 1.0 / channels as f32;
    let samples = data.chunks_exact(frame).map(|f| {
        let sum: f32 = f.chunks_exact(width).map(read).sum();
        let v = sum * scale;
        if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 }
    }).collect();
    Ok(Audio { rate, samples })
}

/// Zero crossings of the resampling filter on each side of its centre.
const LOBES: f64 = 24.0;

fn gcd(a: u64, b: u64) -> u64 { if b == 0 { a } else { gcd(b, a % b) } }

/// Resample by a windowed-sinc low-pass at just under the lower of the two
/// Nyquist rates, as a polyphase filter: one set of taps per output phase.
pub(crate) fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() || from == 0 || to == 0 { return x.to_vec(); }
    let g = gcd(from as u64, to as u64);
    let (step, phases) = (from as u64 / g, to as u64 / g);
    // Cutoff in cycles per input sample.
    let cutoff = 0.5 * 0.96 * (to as f64 / from as f64).min(1.0);
    let half = (LOBES / (2.0 * cutoff)).ceil() as i64;
    let width = (2 * half + 1) as usize;
    let mut taps = vec![0f32; phases as usize * width];
    for p in 0..phases as usize {
        let frac = p as f64 / phases as f64;
        let row = &mut taps[p * width..(p + 1) * width];
        let mut sum = 0.0f64;
        for (i, t) in row.iter_mut().enumerate() {
            // Distance from the output instant to input sample `i - half`.
            let d = (i as i64 - half) as f64 - frac;
            let a = 2.0 * std::f64::consts::PI * cutoff * d;
            let sinc = if a.abs() < 1e-9 { 1.0 } else { a.sin() / a };
            let w = d / (half as f64 + 1.0);
            let hann = if w.abs() >= 1.0 { 0.0 } else { 0.5 + 0.5 * (std::f64::consts::PI * w).cos() };
            let v = sinc * hann;
            sum += v;
            *t = v as f32;
        }
        // Unit gain at DC for every phase.
        for t in row.iter_mut() { *t = (*t as f64 / sum) as f32; }
    }
    let n = (x.len() as u64 * phases).div_ceil(step) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n as u64 {
        let (centre, p) = ((i * step / phases) as i64, (i * step % phases) as usize);
        let row = &taps[p * width..(p + 1) * width];
        let first = centre - half;
        let (lo, hi) = (first.max(0), (first + width as i64).min(x.len() as i64));
        let mut acc = 0.0f32;
        if lo < hi {
            let xs = &x[lo as usize..hi as usize];
            let ts = &row[(lo - first) as usize..(lo - first) as usize + xs.len()];
            for (a, b) in xs.iter().zip(ts) { acc += a * b; }
        }
        out.push(acc);
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A WAV file around `data`: `tag` 1 (PCM) or 3 (float), optionally extensible.
    pub(crate) fn wav(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8], extensible: bool) -> Vec<u8> {
        let align = channels * bits / 8;
        let mut fmt = Vec::new();
        fmt.extend((if extensible { 0xFFFE } else { tag }).to_le_bytes());
        fmt.extend(channels.to_le_bytes());
        fmt.extend(rate.to_le_bytes());
        fmt.extend((rate * align as u32).to_le_bytes());
        fmt.extend(align.to_le_bytes());
        fmt.extend(bits.to_le_bytes());
        if extensible {
            fmt.extend(22u16.to_le_bytes());
            fmt.extend(bits.to_le_bytes());
            fmt.extend(0u32.to_le_bytes());
            fmt.extend(tag.to_le_bytes());
            fmt.extend([0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
        }
        let mut out = b"RIFF\0\0\0\0WAVE".to_vec();
        // An odd-sized chunk before the format, padded to an even offset.
        out.extend(b"LIST\x03\0\0\0abc\0");
        out.extend(b"fmt ");
        out.extend((fmt.len() as u32).to_le_bytes());
        out.extend(fmt);
        out.extend(b"data");
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(data);
        let size = out.len() as u32 - 8;
        out[4..8].copy_from_slice(&size.to_le_bytes());
        out
    }

    fn near(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) { assert!((g - w).abs() < 1e-4, "{got:?} vs {want:?}"); }
    }

    #[test]
    fn each_sample_format_decodes_to_the_same_values() {
        let want = [0.0f32, 0.5, -0.5, 0.25, -1.0];
        let i16s: Vec<u8> = want.iter().flat_map(|v| ((v * 32768.0) as i16).to_le_bytes()).collect();
        let i24s: Vec<u8> = want.iter().flat_map(|v| ((v * 8_388_608.0) as i32).to_le_bytes()[..3].to_vec()).collect();
        let i32s: Vec<u8> = want.iter().flat_map(|v| ((*v as f64 * 2_147_483_648.0) as i32).to_le_bytes()).collect();
        let f32s: Vec<u8> = want.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f64s: Vec<u8> = want.iter().flat_map(|v| (*v as f64).to_le_bytes()).collect();
        let u8s: Vec<u8> = want.iter().map(|v| (v * 128.0 + 128.0) as u8).collect();
        for (tag, bits, data) in [(1, 16, &i16s), (1, 24, &i24s), (1, 32, &i32s), (3, 32, &f32s), (3, 64, &f64s), (1, 8, &u8s)] {
            for extensible in [false, true] {
                let a = decode(&wav(tag, 1, 22050, bits, data, extensible)).unwrap_or_else(|e| panic!("{tag}/{bits}: {e:?}"));
                assert_eq!(a.rate, 22050);
                near(&a.samples, &want);
            }
        }
    }

    #[test]
    fn channels_are_averaged() {
        let frames: [(i16, i16); 3] = [(16384, 0), (-16384, -16384), (8192, -8192)];
        let data: Vec<u8> = frames.iter().flat_map(|(l, r)| [l.to_le_bytes(), r.to_le_bytes()].concat()).collect();
        let a = decode(&wav(1, 2, 44100, 16, &data, false)).unwrap();
        assert_eq!(a.rate, 44100);
        near(&a.samples, &[0.25, -0.5, 0.0]);
    }

    #[test]
    fn a_streamed_file_runs_to_its_end() {
        let data: Vec<u8> = [1000i16, -1000, 500].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut file = wav(1, 1, 16000, 16, &data, false);
        let size_at = file.len() - data.len() - 4;
        file[size_at..size_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(&file).unwrap().samples.len(), 3);
    }

    #[test]
    fn other_containers_and_encodings_are_unsupported_and_named() {
        for (bytes, name) in [(&b"ID3\x04\0\0\0\0\0\0"[..], "mp3"), (&[0xFF, 0xFB, 0x90, 0x00][..], "mp3"),
                              (&b"\0\0\0\x20ftypM4A \0\0\0\0"[..], "m4a"), (&b"OggS\0\x02\0\0"[..], "ogg"),
                              (&b"fLaC\0\0\0\x22"[..], "flac"), (&b"FORM\0\0\0\x10AIFF"[..], "aiff")] {
            match decode(bytes) {
                Err(AudioError::Unsupported(msg)) => assert!(msg.contains(name) && msg.contains("WAV"), "{msg}"),
                other => panic!("{name}: {other:?}"),
            }
        }
        assert!(matches!(decode(b"hello there"), Err(AudioError::Unsupported(_))));
        assert!(matches!(decode(b""), Err(AudioError::Unsupported(_))));
        // mu-law, and 12-bit PCM.
        assert!(matches!(decode(&wav(7, 1, 8000, 8, &[0; 8], false)), Err(AudioError::Unsupported(_))));
        assert!(matches!(decode(&wav(1, 1, 8000, 12, &[0; 8], false)), Err(AudioError::Unsupported(_))));
    }

    #[test]
    fn broken_wav_files_are_invalid() {
        assert!(matches!(decode(b"RIFF\x04\0\0\0WAVE"), Err(AudioError::Invalid(_))));
        assert!(matches!(decode(b"RIFF\x10\0\0\0WAVEdata\x04\0\0\0\0\0\0\0"), Err(AudioError::Invalid(_))));
        assert!(matches!(decode(b"RIFF\x10\0\0\0WAVEfmt \x10\0\0\0\x01\0"), Err(AudioError::Invalid(_))));
        assert!(matches!(decode(&wav(1, 0, 16000, 16, &[0; 8], false)), Err(AudioError::Invalid(_))));
    }

    fn sine(hz: f64, rate: u32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / rate as f64).sin() as f32).collect()
    }

    fn rms(v: &[f32]) -> f32 { (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt() }

    #[test]
    fn a_resampled_sine_is_the_same_sine_at_the_new_rate() {
        for (from, to) in [(44100u32, 16000u32), (48000, 16000), (8000, 16000), (22050, 16000), (11025, 16000)] {
            let got = resample(&sine(1000.0, from, from as usize), from, to);
            let want = sine(1000.0, to, to as usize);
            assert_eq!(got.len(), want.len(), "{from} -> {to}");
            // Away from the edges, where the filter runs off the signal.
            let mid = 1000..want.len() - 1000;
            let err: Vec<f32> = got[mid.clone()].iter().zip(&want[mid]).map(|(g, w)| g - w).collect();
            assert!(rms(&err) < 2e-3, "{from} -> {to}: rms error {}", rms(&err));
        }
    }

    #[test]
    fn resampling_down_removes_what_would_alias() {
        // 10 kHz cannot exist at 16 kHz; unfiltered it would fold to 6 kHz at full level.
        let got = resample(&sine(10_000.0, 44100, 44100), 44100, 16000);
        assert!(rms(&got[1000..15000]) < 1e-3, "rms {}", rms(&got[1000..15000]));
        // 7 kHz is below the new Nyquist rate and passes.
        let got = resample(&sine(7_000.0, 44100, 44100), 44100, 16000);
        assert!((rms(&got[1000..15000]) - 0.7071).abs() < 0.02);
    }

    #[test]
    fn the_same_rate_and_empty_audio_pass_through() {
        assert_eq!(resample(&[0.1, 0.2], 16000, 16000), [0.1, 0.2]);
        assert!(resample(&[], 44100, 16000).is_empty());
        assert_eq!(resample(&[0.5; 441], 44100, 16000).len(), 160);
    }
}
