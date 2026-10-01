//! Stop strings over streamed text.
//!
//! Text arrives a piece at a time, and a stop string can span pieces, so a
//! streaming frontend cannot print a piece the moment it arrives: its tail might
//! be the start of a stop string. [`StopMatcher`] holds back exactly the longest
//! tail that could still grow into one and releases everything before it.

/// Matches a set of stop strings against generated text.
#[derive(Clone, Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
}

impl StopMatcher {
    /// Empty strings are ignored; with no stop strings every piece passes through.
    pub fn new<I: IntoIterator<Item = S>, S: Into<String>>(stops: I) -> Self {
        let stops = stops.into_iter().map(Into::into).filter(|s: &String| !s.is_empty()).collect();
        StopMatcher { stops, held: String::new() }
    }

    pub fn is_empty(&self) -> bool { self.stops.is_empty() }

    /// Add a piece of generated text. Returns the text that is now safe to show
    /// and whether a stop string was reached; on a match the returned text ends
    /// just before it, and the stop string itself is never returned.
    pub fn push(&mut self, piece: &str) -> (String, bool) {
        self.held.push_str(piece);
        if let Some(at) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            let out = self.held[..at].to_string();
            self.held.clear();
            return (out, true);
        }
        let keep = self.stops.iter().map(|s| {
            s.char_indices().map(|(i, _)| i).filter(|&k| k > 0 && self.held.ends_with(&s[..k])).max().unwrap_or(0)
        }).max().unwrap_or(0);
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        (out, false)
    }

    /// End of generation: release whatever was held back.
    pub fn finish(&mut self) -> String { std::mem::take(&mut self.held) }
}

#[cfg(test)]
mod tests {
    use super::StopMatcher;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut m = StopMatcher::new(stops.iter().copied());
        let mut out = String::new();
        for p in pieces {
            let (t, hit) = m.push(p);
            out.push_str(&t);
            if hit { return (out, true); }
        }
        out.push_str(&m.finish());
        (out, false)
    }

    #[test]
    fn stops_across_pieces_and_drops_the_stop_string() {
        assert_eq!(run(&["</answer>"], &["The result", " is 4</ans", "wer> and more"]), ("The result is 4".into(), true));
    }

    #[test]
    fn releases_a_held_prefix_that_does_not_complete() {
        let mut m = StopMatcher::new(["STOP"]);
        assert_eq!(m.push("go ST"), ("go ".into(), false));
        assert_eq!(m.push("ART"), ("START".into(), false));
        assert_eq!(m.finish(), "");
    }

    #[test]
    fn earliest_of_several_stops_wins() {
        assert_eq!(run(&["\n\n", "User:"], &["a User: b\n\nc"]), ("a ".into(), true));
    }

    #[test]
    fn multibyte_stop_strings_hold_at_character_boundaries() {
        assert_eq!(run(&["→end"], &["x →e", "nd y"]), ("x ".into(), true));
        assert_eq!(run(&["→end"], &["x →", "e!"]), ("x →e!".into(), false));
    }

    #[test]
    fn no_stops_passes_everything_through() {
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), false));
    }
}
