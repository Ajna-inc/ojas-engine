//! Snapshot gate: the CPU decoder must keep emitting the same greedy tokens.
//!
//! Compares against a recorded sequence rather than against the reference
//! engine's CPU path, so the test needs no second checkout and no absolute
//! model path.
//!
//! Set OJAS_TEST_GGUF=<path> to run; skipped otherwise. OJAS_REGOLD=1 re-records.

const PROMPT: &[u32] = &[151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198];
const N_GEN: usize = 16;

#[test]
fn greedy_snapshot() {
    let Ok(list) = std::env::var("OJAS_TEST_GGUF") else {
        assert!(std::env::var_os("OJAS_REQUIRE_MODEL_TESTS").is_none(), "required model tests need OJAS_TEST_GGUF");
        eprintln!("skip: set OJAS_TEST_GGUF=<path> to run the CPU gate");
        return;
    };
    let Some(path) = list.split(':').find(|p| !p.is_empty() && std::path::Path::new(p).exists())
    else { panic!("OJAS_TEST_GGUF was set but no model path exists"); };

    use ojas_core::Model;
    let mut g = ojas_formats::gguf::Gguf::open(path).unwrap();
    let m = ojas_cpu::CpuQwen::load(&mut g).unwrap();
    m.prefill(&PROMPT[..PROMPT.len() - 1], 0);
    let mut got = Vec::new();
    let mut t = PROMPT[PROMPT.len() - 1];
    for i in 0..N_GEN {
        t = m.forward_id(t, PROMPT.len() - 1 + i);
        got.push(t);
    }

    let stem = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().into_owned();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/cpu");
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join(format!("{stem}-n{N_GEN}.txt"));
    let line = got.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");

    match std::fs::read_to_string(&f).ok().map(|s| s.trim().to_string()) {
        Some(want) if std::env::var("OJAS_REGOLD").as_deref() != Ok("1") => {
            assert_eq!(line, want, "greedy decode changed for {stem}");
            eprintln!("ok: {N_GEN} tokens match the recording");
        }
        _ => {
            assert_eq!(std::env::var("OJAS_REGOLD").as_deref(), Ok("1"), "missing CPU baseline {}; explicitly set OJAS_REGOLD=1 to record", f.display());
            std::fs::write(&f, format!("{line}\n")).unwrap();
            eprintln!("recorded {}", f.display());
        }
    }
}
