//! Records the source revision the binary was built from, for a server to report.

use std::process::Command;

fn main() {
    let revision = Command::new("git").args(["rev-parse", "--short=12", "HEAD"]).output().ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=OJAS_REVISION={revision}");
    // The revision moves when HEAD does, or when the branch HEAD names gains a commit.
    for path in ["../../.git/HEAD", "../../.git/refs", "../../.git/packed-refs"] {
        println!("cargo:rerun-if-changed={path}");
    }
}
