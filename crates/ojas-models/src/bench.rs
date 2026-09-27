//! Shared guards for the benchmark examples.
//!
//! A timing is only worth keeping if the machine was idle, was not throttling, and
//! both sides of an A/B ran against the same cache state. Load is the cheapest of
//! the three to check, so it lives here rather than in each example.

/// 1-minute load average, or None if the platform will not report it.
pub fn load_average() -> Option<f64> {
    let mut avg = [0f64; 3];
    // getloadavg returns how many of the 3 samples it filled, or -1.
    let n = unsafe { libc::getloadavg(avg.as_mut_ptr(), 3) };
    if n >= 1 { Some(avg[0]) } else { None }
}

/// Print the machine's load and say whether timings taken now are worth quoting.
/// Returns true if the machine looks idle enough to measure on.
///
/// The threshold is 4. Past that, other work competes for the same cores, memory
/// bandwidth and page cache, enough to invert an A/B rather than just widen it.
pub fn report_load() -> bool {
    match load_average() {
        Some(l) if l > 4.0 => {
            eprintln!("** load average {l:.1} — the machine is busy. Correctness results still \
                       hold (they do not depend on the clock); TIMINGS DO NOT. **");
            false
        }
        Some(l) => { eprintln!("load average {l:.1} — ok to measure"); true }
        None => true,
    }
}
