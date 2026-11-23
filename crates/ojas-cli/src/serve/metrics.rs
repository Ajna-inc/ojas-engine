//! Counters `GET /metrics` reports for a completion server, in the Prometheus text
//! format. Process-wide: one server runs per process.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

static REQUESTS: AtomicU64 = AtomicU64::new(0);
static ERRORS: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);
static PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static CACHED_TOKENS: AtomicU64 = AtomicU64::new(0);
static COMPLETION_TOKENS: AtomicU64 = AtomicU64::new(0);
static FIRST_TOKEN_US: AtomicU64 = AtomicU64::new(0);
static FIRST_TOKENS: AtomicU64 = AtomicU64::new(0);
static GENERATION_US: AtomicU64 = AtomicU64::new(0);
static REQUEST_US: AtomicU64 = AtomicU64::new(0);
static RUNNING: AtomicU64 = AtomicU64::new(0);
static WAITING: AtomicU64 = AtomicU64::new(0);
static SLOTS: AtomicU64 = AtomicU64::new(1);

/// What one answered request cost.
pub(crate) struct Answered {
    pub(crate) prompt_tokens: usize,
    pub(crate) cached_tokens: usize,
    pub(crate) completion_tokens: usize,
    /// From the request's arrival to its first token, when it produced one.
    pub(crate) first_token: Option<Duration>,
    pub(crate) total: Duration,
}

pub(crate) fn answered(a: &Answered) {
    REQUESTS.fetch_add(1, Relaxed);
    PROMPT_TOKENS.fetch_add((a.prompt_tokens - a.cached_tokens.min(a.prompt_tokens)) as u64, Relaxed);
    CACHED_TOKENS.fetch_add(a.cached_tokens as u64, Relaxed);
    COMPLETION_TOKENS.fetch_add(a.completion_tokens as u64, Relaxed);
    REQUEST_US.fetch_add(a.total.as_micros() as u64, Relaxed);
    if let Some(first) = a.first_token {
        FIRST_TOKENS.fetch_add(1, Relaxed);
        FIRST_TOKEN_US.fetch_add(first.as_micros() as u64, Relaxed);
        GENERATION_US.fetch_add(a.total.saturating_sub(first).as_micros() as u64, Relaxed);
    }
}

/// A request that failed, or was malformed.
pub(crate) fn failed() { ERRORS.fetch_add(1, Relaxed); }

/// A request turned away because the server was full or stopping.
pub(crate) fn refused() { REFUSED.fetch_add(1, Relaxed); }

/// Requests decoding and requests waiting for a slot, as of now.
pub(crate) fn load(running: usize, waiting: usize) {
    RUNNING.store(running as u64, Relaxed);
    WAITING.store(waiting as u64, Relaxed);
}

pub(crate) fn slots(n: usize) { SLOTS.store(n as u64, Relaxed); }

pub(crate) fn render() -> String {
    let secs = |us: &AtomicU64| format!("{:.6}", us.load(Relaxed) as f64 / 1e6);
    let n = |v: &AtomicU64| v.load(Relaxed).to_string();
    let rows: [(&str, &str, &str, String); 15] = [
        ("requests_total", "counter", "Completion requests answered.", n(&REQUESTS)),
        ("request_errors_total", "counter", "Completion requests that failed or were malformed.", n(&ERRORS)),
        ("requests_refused_total", "counter", "Requests turned away because the server was full or stopping.", n(&REFUSED)),
        ("prompt_tokens_total", "counter", "Prompt tokens computed.", n(&PROMPT_TOKENS)),
        ("prompt_tokens_cached_total", "counter", "Prompt tokens restored from the prefix cache.", n(&CACHED_TOKENS)),
        ("completion_tokens_total", "counter", "Tokens generated.", n(&COMPLETION_TOKENS)),
        ("first_token_seconds_total", "counter", "Time from a request's arrival to its first token, summed.", secs(&FIRST_TOKEN_US)),
        ("first_token_requests_total", "counter", "Requests that produced a first token.", n(&FIRST_TOKENS)),
        ("generation_seconds_total", "counter", "Time from first token to last, summed; completion tokens over this is the decode rate.", secs(&GENERATION_US)),
        ("request_seconds_total", "counter", "Time from a request's arrival to its end, summed.", secs(&REQUEST_US)),
        ("requests_running", "gauge", "Requests holding a slot.", n(&RUNNING)),
        ("requests_waiting", "gauge", "Requests waiting for a slot.", n(&WAITING)),
        ("slots", "gauge", "Requests the model decodes together.", n(&SLOTS)),
        ("connections_open", "gauge", "Requests read and not yet fully answered, on any route.", super::conn::active().to_string()),
        ("draining", "gauge", "1 once the server was asked to stop.", (super::conn::draining() as u8).to_string()),
    ];
    rows.iter().map(|(name, kind, help, value)| {
        format!("# HELP ojas_serve_{name} {help}\n# TYPE ojas_serve_{name} {kind}\nojas_serve_{name} {value}\n")
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_answered_request_counts_computed_and_restored_prompt_tokens_apart() {
        let before = (PROMPT_TOKENS.load(Relaxed), CACHED_TOKENS.load(Relaxed), FIRST_TOKENS.load(Relaxed));
        answered(&Answered { prompt_tokens: 100, cached_tokens: 60, completion_tokens: 8,
                             first_token: Some(Duration::from_millis(50)), total: Duration::from_millis(250) });
        answered(&Answered { prompt_tokens: 10, cached_tokens: 0, completion_tokens: 0, first_token: None,
                             total: Duration::from_millis(5) });
        assert!(PROMPT_TOKENS.load(Relaxed) - before.0 >= 50);
        assert!(CACHED_TOKENS.load(Relaxed) - before.1 >= 60);
        assert!(FIRST_TOKENS.load(Relaxed) - before.2 >= 1);
        let text = render();
        assert!(text.contains("# TYPE ojas_serve_requests_total counter\nojas_serve_requests_total "));
        assert!(text.contains("# TYPE ojas_serve_requests_waiting gauge\n"));
    }
}
