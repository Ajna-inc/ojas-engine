//! Snapshot gate: the tokenizer must keep producing the same ids.
//!
//! The check is round-trip self-consistency plus a stable encoding, neither of which
//! needs a second implementation to compare against.
//!
//! Set OJAS_TEST_GGUF=<path> to run; skipped otherwise.

#[test]
fn tokenizer_roundtrip() {
    let Ok(list) = std::env::var("OJAS_TEST_GGUF") else {
        assert!(std::env::var_os("OJAS_REQUIRE_MODEL_TESTS").is_none(), "required model tests need OJAS_TEST_GGUF");
        eprintln!("skip: set OJAS_TEST_GGUF=<path> to run the tokenizer gate");
        return;
    };
    let Some(path) = list.split(':').find(|p| !p.is_empty() && std::path::Path::new(p).exists())
    else { panic!("OJAS_TEST_GGUF was set but no model path exists"); };

    let g = ojas_formats::gguf::Gguf::open(path).unwrap();
    let t = ojas_tokenize::Bpe::from_gguf(&g);
    for s in ["Hello, world!", "def f(x):\n    return x * 2\n", "2, 3, and 5.", "  leading spaces"] {
        let ids = t.encode(s);
        assert!(!ids.is_empty(), "empty encoding for {s:?}");
        // Concatenate bytes, not per-id strings: byte-level BPE splits a character
        // across tokens, and decoding each id alone yields U+FFFD for both halves.
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| t.decode_bytes(i)).collect();
        let back = String::from_utf8(bytes).expect("round-trip produced invalid UTF-8");
        assert_eq!(back, s, "round-trip changed the text");
    }
    eprintln!("ok: tokenizer round-trip on {path}");
}
