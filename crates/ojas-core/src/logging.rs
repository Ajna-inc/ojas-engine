//! Central logging for the engine.
//!
//! Engine code emits through the `tracing` facade — `tracing::{error, warn, info,
//! debug, trace}!` — never `eprintln!`. All crates feed one global dispatcher, so
//! the host decides where the output goes:
//!
//!   * standalone / CLI: call [`init`] once at startup for the default stderr sink.
//!   * embedded in an app: the app installs its own `tracing` subscriber and the
//!     engine's events flow into it automatically; do not call [`init`].
//!
//! Level convention (so a host can filter by level or target):
//!   * `error!` — the operation failed.
//!   * `warn!`  — degraded / partially-supported / falling back.
//!   * `info!`  — one-time lifecycle a user wants to see (model geometry, chosen
//!                context, device, session restore).
//!   * `debug!` — per-load / per-layer detail, autotune, cache, timing.
//!   * `trace!` — high-volume dumps (per-token tensor stats), gated at the call site.
//!
//! Each event carries a `target:` naming its subsystem (e.g. `target: "arch"`,
//! `"ctx"`, `"mla"`, `"autotune"`), so a host can filter per-subsystem, e.g.
//! `OJAS_LOG=info,autotune=debug,rdump=trace`.

/// Install the default stderr subscriber, honoring the `OJAS_LOG` filter
/// (a `tracing` EnvFilter directive, e.g. `info`, `debug`, `info,mla=trace`).
/// Defaults to `info`. Idempotent and host-safe: if a subscriber is already
/// installed (e.g. by an embedding app) this is a no-op.
#[cfg(feature = "subscriber")]
pub fn init() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_env("OJAS_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .try_init();
}

/// No-op when the `subscriber` feature is disabled: the host is expected to
/// install its own `tracing` subscriber.
#[cfg(not(feature = "subscriber"))]
pub fn init() {}
