//! Snapshot gate: the GGUF reader must keep reading the same bytes.
//!
//! Compares against a recorded digest rather than against the reference engine's
//! reader, so no second checkout is needed. The invariant is that the reader has
//! not changed what it returns.
//!
//! Models come from OJAS_TEST_GGUF (colon-separated); the test skips when unset
//! rather than hard-coding one machine's absolute paths.

// Fixed FNV-1a digest: unlike DefaultHasher this algorithm is stable across
// toolchains. This is a regression checksum, not artifact authentication.
fn digest(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3))
}

#[test]
fn reader_snapshot() {
    let Ok(list) = std::env::var("OJAS_TEST_GGUF") else {
        assert!(std::env::var_os("OJAS_REQUIRE_MODEL_TESTS").is_none(), "required model tests need OJAS_TEST_GGUF");
        eprintln!("skip: set OJAS_TEST_GGUF=<path>[:<path>...] to run the reader snapshot");
        return;
    };
    assert!(!list.trim().is_empty(), "OJAS_TEST_GGUF is empty");
    for path in list.split(':').filter(|p| !p.is_empty()) {
        if !std::path::Path::new(path).exists() {
            panic!("configured model is missing: {path}");
        }
        let mut g = ojas_formats::gguf::Gguf::open(path).unwrap();
        let mut names: Vec<String> = g.tensors.keys().cloned().collect();
        names.sort();
        assert!(!names.is_empty(), "{path}: no tensors");

        // first / middle / last, so a reordering or an offset slip shows up
        let mut acc = names.len() as u64;
        for idx in [0, names.len() / 2, names.len() - 1] {
            let n = names[idx].clone();
            let (shape, ty, bytes) = g.read_tensor(&n).unwrap();
            assert!(!bytes.is_empty(), "{path}:{n} read empty");
            acc = acc.wrapping_mul(31).wrapping_add(digest(&bytes));
            acc = acc.wrapping_mul(31).wrapping_add(ty as u64);
            for d in shape { acc = acc.wrapping_mul(31).wrapping_add(d as u64); }
        }
        let stem = std::path::Path::new(path).file_stem().unwrap().to_string_lossy();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/formats");
        let baseline = dir.join(format!("{stem}.txt"));
        let got = format!("{acc:016x}");
        if std::env::var("OJAS_REGOLD").as_deref() == Ok("1") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&baseline, format!("{got}\n")).unwrap();
        } else {
            let want = std::fs::read_to_string(&baseline).expect("missing reader baseline; explicitly set OJAS_REGOLD=1 to record");
            assert_eq!(got, want.trim(), "reader digest changed for {stem}");
        }
        eprintln!("ok: {path} ({} tensors) digest {acc:#x}", names.len());
    }
}
