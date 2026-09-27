//! Tiny on-disk fixtures exercise the same loader as multi-gigabyte models.
use ojas_formats::gguf::Gguf;
use std::{
    fs::{self, File},
    io::{Seek, SeekFrom},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        // The parent deliberately contains the shard marker: only filenames may change.
        let p = std::env::temp_dir().join(format!(
            "ojas-00001-of-fixture-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn write(&self, part: u32, count: u32, total: u32, names: &[&str], align: u32) -> PathBuf {
        fn string(b: &mut Vec<u8>, s: &str) {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.as_bytes());
        }
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend((names.len() as u64).to_le_bytes());
        b.extend(4u64.to_le_bytes());
        for (k, v) in [
            ("split.no", part),
            ("split.count", count),
            ("split.tensors.count", total),
            ("general.alignment", align),
        ] {
            string(&mut b, k);
            b.extend(4u32.to_le_bytes());
            b.extend(v.to_le_bytes());
        }
        for (i, name) in names.iter().enumerate() {
            string(&mut b, name);
            b.extend(1u32.to_le_bytes());
            b.extend(1u64.to_le_bytes());
            b.extend(0u32.to_le_bytes());
            b.extend((i as u64 * 32).to_le_bytes());
        }
        b.resize(b.len().div_ceil(32) * 32, 0);
        for _ in names {
            b.extend((part as f32).to_le_bytes());
            b.extend([0; 28]);
        }
        let p = self
            .0
            .join(format!("model-{:05}-of-{count:05}.gguf", part + 1));
        fs::write(&p, b).unwrap();
        p
    }
    fn three(&self) -> Vec<PathBuf> {
        vec![
            self.write(0, 3, 2, &[], 32),
            self.write(1, 3, 2, &["a"], 32),
            self.write(2, 3, 2, &["b"], 32),
        ]
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn error(p: &std::path::Path) -> String {
    Gguf::open(p.to_str().unwrap())
        .err()
        .expect("must reject fixture")
        .to_string()
}

#[test]
fn metadata_only_first_shard_reads_weights_from_correct_files() {
    let f = Fixture::new();
    let paths = f.three();
    let mut g = Gguf::open(paths[0].to_str().unwrap()).unwrap();
    assert_eq!(g.tensors.len(), 2);
    for (name, value) in [("a", 1f32), ("b", 2f32)] {
        let (_, ty, bytes) = g.read_tensor(name).unwrap();
        assert_eq!(ty, 0);
        assert_eq!(bytes, value.to_le_bytes());
    }
}
#[test]
fn descriptors_are_rewound_and_shards_must_be_complete_and_ordered() {
    let f = Fixture::new();
    let paths = f.three();
    let files = || {
        paths
            .iter()
            .map(|p| File::open(p).unwrap())
            .collect::<Vec<_>>()
    };
    let mut advanced = files();
    for file in &mut advanced {
        file.seek(SeekFrom::End(0)).unwrap();
    }
    assert_eq!(Gguf::from_files(advanced).unwrap().tensors.len(), 2);
    let mut missing = files();
    missing.pop();
    assert!(Gguf::from_files(missing)
        .err()
        .unwrap()
        .to_string()
        .contains("split.count"));
    let mut reversed = files();
    reversed.swap(1, 2);
    assert!(Gguf::from_files(reversed)
        .err()
        .unwrap()
        .to_string()
        .contains("split.no"));
}
#[test]
fn later_shard_is_not_a_standalone_model() {
    let f = Fixture::new();
    let paths = f.three();
    assert!(error(&paths[1]).contains("first shard"));
}
#[test]
fn missing_and_inconsistent_shards_are_rejected() {
    let f = Fixture::new();
    let paths = f.three();
    fs::remove_file(&paths[2]).unwrap();
    assert!(error(&paths[0]).contains("missing"));
    let wrong = f.write(2, 4, 2, &["b"], 32);
    fs::rename(wrong, &paths[2]).unwrap();
    assert!(error(&paths[0]).contains("split.count"));
}
#[test]
fn duplicates_within_and_across_shards_are_rejected() {
    let f = Fixture::new();
    let paths = f.three();
    f.write(2, 3, 2, &["a"], 32);
    assert!(error(&paths[0]).contains("duplicate tensor"));
    let p = f.write(0, 1, 2, &["a", "a"], 32);
    assert!(error(&p).contains("duplicate GGUF tensor"));
}
#[test]
fn incorrect_total_and_zero_alignment_return_errors() {
    let f = Fixture::new();
    let p = f.write(0, 1, 2, &["a"], 32);
    assert!(error(&p).contains("expected 2 tensors, found 1"));
    let p = f.write(0, 1, 1, &["a"], 0);
    assert!(error(&p).contains("alignment"));
}

#[test]
fn corrupt_header_lengths_fail_before_allocation() {
    let f = Fixture::new();
    let path = f.0.join("bad.gguf");
    let header = |version: u32, tensors: u64, metadata: u64| {
        let mut b = b"GGUF".to_vec();
        b.extend(version.to_le_bytes());
        b.extend(tensors.to_le_bytes());
        b.extend(metadata.to_le_bytes());
        b
    };
    let mut huge_string = header(3, 0, 1);
    huge_string.extend(u64::MAX.to_le_bytes());
    let mut nested_array = header(3, 0, 1);
    nested_array.extend(1u64.to_le_bytes()); nested_array.push(b'x');
    nested_array.extend(9u32.to_le_bytes()); nested_array.extend(9u32.to_le_bytes());
    nested_array.extend(1u64.to_le_bytes());
    let mut rank = header(3, 1, 0);
    rank.extend(1u64.to_le_bytes()); rank.push(b'x'); rank.extend(u32::MAX.to_le_bytes());
    for bytes in [header(99, 0, 0), header(3, u64::MAX, 0), huge_string, nested_array, rank] {
        fs::write(&path, bytes).unwrap();
        assert!(Gguf::open(path.to_str().unwrap()).is_err());
    }
}

#[test]
fn truncated_weight_payload_is_rejected_before_mapping() {
    let f = Fixture::new();
    let path = f.write(0, 1, 1, &["a"], 32);
    let len = fs::metadata(&path).unwrap().len();
    fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(len - 31).unwrap();
    assert!(error(&path).contains("payload extends beyond"));
}
