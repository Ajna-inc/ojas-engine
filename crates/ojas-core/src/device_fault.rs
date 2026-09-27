//! Device failures, and the sessions they poison.
//!
//! A GPU command buffer can fail — most commonly `OutOfMemory` when the declared
//! working set exceeds what the device can keep resident. Metal reports this only
//! through the command buffer's `status`/`error`; code that does not check it keeps
//! decoding from whatever is already in the output buffers, so the failure surfaces
//! as fluent, plausible, wrong text rather than as an error.
//!
//! Two separate consequences:
//!
//!   1. Output buffers from failed work must never be consumed.
//!   2. The session is untrustworthy even if later work succeeds, because a failed
//!      command buffer may have partially updated recurrent state, KV cache or
//!      expert tables. Recovery means reinitialising the model, not retrying.
//!
//! So a fault latches: once set, generation refuses to continue until the fault is
//! explicitly taken and the session rebuilt.
//!
//! The latch is per-thread. A decoder is not `Sync` and each serving thread owns
//! its model, so a fault on one thread must not condemn another's session.

use std::cell::RefCell;

/// What failed, with enough context to act on it rather than just log it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceError {
    /// Backend-specific error domain, e.g. "MTLCommandBufferError".
    pub domain: String,
    pub code: i64,
    /// Human-readable cause from the driver, when it supplies one.
    pub description: String,
    /// What the engine was doing — which graph, which layer, which phase.
    pub context: String,
}

impl std::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} ({}) during {}", self.domain, self.code, self.description, self.context)
    }
}

impl std::error::Error for DeviceError {}

thread_local! {
    static FAULT: RefCell<Option<DeviceError>> = const { RefCell::new(None) };
}

/// Latch a fault. The first fault is kept: later failures are usually consequences
/// of it, and the original identifies the cause.
pub fn set(err: DeviceError) {
    FAULT.with(|f| {
        let mut slot = f.borrow_mut();
        if slot.is_none() {
            tracing::error!(target: "device", "device fault latched: {err}");
            *slot = Some(err);
        }
    });
}

/// Whether this thread's session has been poisoned.
pub fn is_faulted() -> bool {
    FAULT.with(|f| f.borrow().is_some())
}

/// Read the fault without clearing it.
pub fn peek() -> Option<DeviceError> {
    FAULT.with(|f| f.borrow().clone())
}

/// Take the fault, clearing the latch.
///
/// Call this only when rebuilding the session. Clearing it while reusing a model a
/// failed command buffer may have partially written reintroduces the silent
/// corruption this module prevents.
pub fn take() -> Option<DeviceError> {
    FAULT.with(|f| f.borrow_mut().take())
}

/// Test seam: inject a fault so error propagation can be exercised without
/// provoking a real device failure (which would need a machine-sized allocation).
pub fn inject_for_test(context: &str) {
    set(DeviceError {
        domain: "TestInjected".into(),
        code: -1,
        description: "injected device fault".into(),
        context: context.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear() { let _ = take(); }

    #[test]
    fn fault_latches_and_keeps_the_first() {
        clear();
        assert!(!is_faulted());
        set(DeviceError { domain: "A".into(), code: 1, description: "first".into(), context: "x".into() });
        set(DeviceError { domain: "B".into(), code: 2, description: "second".into(), context: "y".into() });
        // The second failure is usually a consequence of the first; keep the cause.
        assert_eq!(peek().unwrap().description, "first");
        assert!(is_faulted());
        assert_eq!(take().unwrap().code, 1);
        assert!(!is_faulted(), "take clears the latch");
    }

    #[test]
    fn display_carries_enough_to_act_on() {
        clear();
        let e = DeviceError {
            domain: "MTLCommandBufferError".into(), code: 8,
            description: "OutOfMemory".into(), context: "qwen4exp layer 12 experts".into(),
        };
        let s = e.to_string();
        for want in ["MTLCommandBufferError", "8", "OutOfMemory", "layer 12"] {
            assert!(s.contains(want), "{s:?} should mention {want}");
        }
    }
}
