//! `Gguf::read_rows` reads single rows of a 2-D table and agrees with a full read.
use ojas_formats::gguf::Gguf;
use std::path::PathBuf;

/// A GGUF holding `f32_table` (F32, 4 x 3) and `q8_table` (Q8_0, 32 x 3), written
/// under a name of its own for each test.
fn fixture(test: &str) -> PathBuf {
    fn string(b: &mut Vec<u8>, s: &str) {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    }
    let f32_rows: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 2.0).collect();
    let mut q8 = Vec::new();
    for r in 0..3 {
        q8.extend(half::f16::from_f32(0.25 * (r + 1) as f32).to_le_bytes());
        q8.extend((0..32).map(|j| (j as i8 - 16 + r as i8) as u8));
    }
    let tensors: [(&str, [u64; 2], u32, Vec<u8>); 2] = [
        ("f32_table", [4, 3], 0, f32_rows.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ("q8_table", [32, 3], 8, q8),
    ];

    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend(1u64.to_le_bytes());
    string(&mut b, "general.alignment");
    b.extend(4u32.to_le_bytes());
    b.extend(32u32.to_le_bytes());
    let mut offset = 0u64;
    for (name, dims, ty, data) in &tensors {
        string(&mut b, name);
        b.extend(2u32.to_le_bytes());
        for d in dims { b.extend(d.to_le_bytes()); }
        b.extend(ty.to_le_bytes());
        b.extend(offset.to_le_bytes());
        offset += (data.len() as u64).div_ceil(32) * 32;
    }
    b.resize(b.len().div_ceil(32) * 32, 0);
    for (_, _, _, data) in &tensors {
        b.extend(data);
        b.resize(b.len().div_ceil(32) * 32, 0);
    }
    let path = std::env::temp_dir().join(format!("ojas-read-rows-{test}-{}.gguf", std::process::id()));
    std::fs::write(&path, b).unwrap();
    path
}

fn as_f32(ty: u32, bytes: &[u8]) -> Vec<f32> {
    match ty {
        0 => bytes.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
        _ => bytes.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c).to_f32()).collect(),
    }
}

#[test]
fn rows_match_the_full_tensor_for_float_and_quantized_tables() {
    let path = fixture("match");
    let mut g = Gguf::open(path.to_str().unwrap()).unwrap();
    for (name, width) in [("f32_table", 4), ("q8_table", 32)] {
        let (_, ty, bytes) = g.read_tensor(name).unwrap();
        let full = as_f32(ty, &bytes);
        let rows = g.read_rows(name, &[2, 0]).unwrap();
        assert_eq!(rows, vec![full[2 * width..3 * width].to_vec(), full[..width].to_vec()], "{name}");
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn rows_past_the_table_are_refused() {
    let path = fixture("bounds");
    let mut g = Gguf::open(path.to_str().unwrap()).unwrap();
    let err = g.read_rows("q8_table", &[3]).unwrap_err().to_string();
    assert!(err.contains("row 3 is past its 3 rows"), "{err}");
    let _ = std::fs::remove_file(path);
}
