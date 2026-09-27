//! `EngineConfig::install` is the host's alternative to the environment.
//!
//! A host that is not a shell — a sandboxed utility process, say — has no
//! meaningful environment to set, so it installs a config instead. Install is
//! once per process, because knobs are captured at model load and a second install
//! would not apply, so this lives in its own integration test binary and therefore
//! its own process.

use ojas_core::config::EngineConfig;

#[test]
fn install_overrides_env_and_only_wins_once() {
    // Before anything is installed, `current` is just the environment.
    let from_env = EngineConfig::from_env();
    let before = EngineConfig::current();
    assert_eq!(
        before.expert_cache_gb, from_env.expert_cache_gb,
        "current() should fall back to the environment before install"
    );

    let mut cfg = EngineConfig::from_env();
    cfg.top_k = Some(3);
    cfg.ctx = Some(8192);
    cfg.expert_cache_gb = 7.5;
    cfg.no_spec = true;
    assert!(EngineConfig::install(cfg).is_ok(), "first install should win");

    let now = EngineConfig::current();
    assert_eq!(now.top_k, Some(3));
    assert_eq!(now.ctx, Some(8192));
    assert_eq!(now.expert_cache_gb, 7.5);
    assert!(now.no_spec);

    // A second install must fail rather than appear to work: the engine has already
    // read these values by the time anyone would try.
    let mut second = EngineConfig::from_env();
    second.top_k = Some(99);
    assert!(
        EngineConfig::install(second).is_err(),
        "second install must be rejected"
    );
    assert_eq!(
        EngineConfig::current().top_k,
        Some(3),
        "a rejected install must not have changed anything"
    );
}
