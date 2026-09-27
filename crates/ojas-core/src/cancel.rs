use std::sync::atomic::{AtomicBool, Ordering};

pub static STREAM_CANCEL: AtomicBool = AtomicBool::new(false);

pub fn request_cancel() { STREAM_CANCEL.store(true, Ordering::Relaxed); }
pub fn clear_cancel() { STREAM_CANCEL.store(false, Ordering::Relaxed); }
