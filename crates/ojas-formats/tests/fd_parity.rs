//! Opening a model from descriptors must be indistinguishable from opening it
//! by path.
//!
//! `Gguf::from_files` exists because a sandboxed process cannot call `open()`
//! on a model path — the browser opens the shards and passes the descriptors
//! in. That path is only ever exercised inside the sandbox, where a mistake
//! surfaces as a model that will not load and no useful diagnostic, so the
//! equivalence is pinned here instead.
//!
//! Models come from OJAS_TEST_GGUF (colon-separated), matching the other
//! golden tests; the test skips when unset rather than hard-coding one
//! machine's paths.

use ojas_formats::gguf::Gguf;

fn each_model(mut f: impl FnMut(&str)) {
    let Ok(list) = std::env::var("OJAS_TEST_GGUF") else {
        assert!(std::env::var_os("OJAS_REQUIRE_MODEL_TESTS").is_none(), "required model tests need OJAS_TEST_GGUF");
        eprintln!("skip: set OJAS_TEST_GGUF=<path>[:<path>...] to run fd parity");
        return;
    };
    assert!(!list.trim().is_empty(), "OJAS_TEST_GGUF is empty");
    for path in list.split(':').filter(|p| !p.is_empty()) {
        if !std::path::Path::new(path).exists() {
            panic!("configured model is missing: {path}");
        }
        f(path);
    }
}

#[test]
fn fd_open_matches_path_open() {
    each_model(|path| {
        let by_path = Gguf::open(path).unwrap_or_else(|e| panic!("path open {path}: {e}"));
        let by_fd = Gguf::from_files(by_path.shard_files().unwrap())
            .unwrap_or_else(|e| panic!("fd open {path}: {e}"));

        assert_eq!(by_fd.arch(), by_path.arch(), "arch differs for {path}");
        assert_eq!(
            by_fd.data_offsets, by_path.data_offsets,
            "tensor-data base offsets differ for {path}"
        );
        assert_eq!(
            by_fd.tensors.len(),
            by_path.tensors.len(),
            "tensor count differs for {path}"
        );
        assert_eq!(
            by_fd.meta.len(),
            by_path.meta.len(),
            "metadata key count differs for {path}"
        );

        // Every tensor must be found at the same place in the same part.
        for name in by_path.tensors.keys() {
            assert_eq!(
                by_fd.tensor_meta(name),
                by_path.tensor_meta(name),
                "tensor {name} located differently in {path}"
            );
        }

        // The mode flag the loader branches on.
        assert!(by_fd.is_fd_backed(), "fd-opened model should report fd-backed");
        assert!(!by_path.is_fd_backed(), "path-opened model should not");
    });
}

#[test]
fn fd_open_reads_identical_tensor_bytes() {
    each_model(|path| {
        let mut by_path = Gguf::open(path).unwrap();
        let mut by_fd = Gguf::from_files(by_path.shard_files().unwrap()).unwrap();

        // Metadata agreeing is not the same as the bytes agreeing: the
        // descriptor path could read from the right offset in the wrong file.
        let mut names: Vec<String> = by_path.tensors.keys().cloned().collect();
        names.sort();
        for name in names.iter().take(8) {
            let a = by_path.read_tensor_raw(name).unwrap();
            let b = by_fd.read_tensor_raw(name).unwrap();
            assert_eq!(a.0, b.0, "shape differs for {name} in {path}");
            assert_eq!(a.1, b.1, "dtype differs for {name} in {path}");
            assert_eq!(a.2, b.2, "raw bytes differ for {name} in {path}");
        }
    });
}

#[test]
fn shard_files_are_independent_descriptors() {
    each_model(|path| {
        let by_path = Gguf::open(path).unwrap();
        let g = Gguf::from_files(by_path.shard_files().unwrap()).unwrap();
        // The streamer needs its own descriptors: it sets F_NOCACHE, and a
        // `dup` shares the file description, so handing over the originals
        // would make the reader's own reads uncached too.
        let a = g.shard_files().unwrap();
        let b = g.shard_files().unwrap();
        assert_eq!(a.len(), by_path.data_offsets.len(), "shard count differs for {path}");
        assert_eq!(b.len(), a.len());
        use std::os::unix::io::AsRawFd;
        assert_ne!(
            a[0].as_raw_fd(),
            b[0].as_raw_fd(),
            "shard_files must hand out distinct descriptors"
        );
    });
}
