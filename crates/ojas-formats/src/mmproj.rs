//! Deterministic vision-projector (`mmproj`) selection and metadata checks,
//! before GPU allocation.
//!
//! Sibling of [`crate::mtp`], and the same shape: `discover`
//! picks exactly one file or refuses, `validate` proves the file matches the
//! decoder it is about to be attached to, and both run before a single byte is
//! uploaded. The difference is that an mmproj's configuration lives entirely in
//! its KV, so it must be merged in with [`crate::gguf::Gguf::attach_with_meta`]
//! rather than [`crate::gguf::Gguf::attach`].
use crate::gguf::{Gguf, Meta};
use anyhow::{ensure, Context, Result};
use std::path::{Path, PathBuf};

/// Tensor filter for `attach_with_meta`: the ViT ships as `v.*` and the
/// projector as `mm.*`. Nothing else in an mmproj belongs to the decoder, and the
/// prefixes cannot collide with a decoder's `blk.*`/`token_embd`/`output*`.
pub fn keep_tensor(name: &str) -> bool {
    name.starts_with("v.") || name.starts_with("mm.")
}

/// KV filter for `attach_with_meta`. Every key the vision path reads is under
/// `clip.`; `general.*` is the decoder's and must stay the decoder's.
/// `attach_with_meta`'s "main model wins" rule would protect `general.*`
/// anyway; this states the intent rather than relying on it.
pub fn keep_kv(key: &str) -> bool {
    key.starts_with("clip.")
}

/// The `clip.vision.*` integer keys the encoder cannot be built without.
/// Checked for presence here so a truncated or foreign file fails with a key
/// name rather than with an `unwrap` inside graph construction.
const REQUIRED_U32: [&str; 7] = [
    "clip.vision.projection_dim",
    "clip.vision.image_size",
    "clip.vision.patch_size",
    "clip.vision.embedding_length",
    "clip.vision.feed_forward_length",
    "clip.vision.block_count",
    "clip.vision.attention.head_count",
];

/// Pick the mmproj to load: an explicit path if one was given, else the single
/// `*mmproj*.gguf` sitting next to the model.
///
/// The naming convention is llama.cpp's (`mmproj-model-f16.gguf`,
/// `surya-2-mmproj.gguf`) and the file lives beside the decoder rather than in
/// a subdirectory, which is the one way this differs from [`crate::mtp::discover`].
/// Ambiguity is refused, never resolved by sort order.
pub fn discover(model: &Path, explicit: Option<&str>) -> Result<Option<PathBuf>> {
    if let Some(path) = explicit {
        let path = PathBuf::from(path);
        ensure!(path.is_file(), "explicit mmproj does not exist: {}", path.display());
        return Ok(Some(path));
    }
    if model.as_os_str().is_empty() { return Ok(None); }
    let dir = match model.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let entries = match std::fs::read_dir(&dir) {
        Ok(v) => v,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading model directory {}", dir.display())),
    };
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if path.file_name() == model.file_name() { continue; }
        if path.extension().is_some_and(|s| s == "gguf")
            && path.file_name().is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().contains("mmproj"))
            && path.is_file()
        {
            paths.push(path);
        }
    }
    paths.sort();
    ensure!(paths.len() <= 1, "ambiguous mmproj in {}: {} candidate GGUF files; select one with --mmproj /path/to/mmproj.gguf", dir.display(), paths.len());
    Ok(paths.pop())
}

/// Prove `mmproj` is a vision projector for `base`, before anything is allocated.
///
/// Three classes of failure, each with its own message: wrong kind of file (a
/// decoder passed to `--mmproj`), incomplete file (a key the encoder needs is
/// absent, or the tensors do not cover the declared block count), and wrong
/// model (the projector's output width does not match the decoder's embedding,
/// so the mmproj belongs to a different checkpoint).
pub fn validate(base: &Gguf, mmproj: &Gguf) -> Result<()> {
    let ty = match mmproj.meta.get("general.type") {
        Some(Meta::Str(s)) => s.as_str(),
        _ => "?",
    };
    let arch = mmproj.arch();
    ensure!(ty == "mmproj" || arch == "clip",
        "not an mmproj GGUF ({}): general.type={ty:?}, general.architecture={arch:?}", mmproj.path);
    ensure!(mmproj.meta_bool("clip.has_vision_encoder") != Some(false),
        "mmproj declares clip.has_vision_encoder=false: no vision encoder to load");
    for key in REQUIRED_U32 {
        ensure!(mmproj.meta_u32(key).is_some(), "mmproj missing required key {key}");
    }

    // The one cross-file invariant: the projector writes rows straight into the
    // decoder's embedding stream, so its output width is the decoder's width. A
    // mismatch means the two files come from different checkpoints, which would
    // otherwise surface as garbage text.
    let base_arch = base.arch();
    let d = base.meta_u32(&format!("{base_arch}.embedding_length"))
        .with_context(|| format!("missing {base_arch}.embedding_length on the decoder"))? as u64;
    let proj = mmproj.meta_u32("clip.vision.projection_dim").unwrap() as u64;
    ensure!(proj == d, "mmproj projection_dim {proj} != {base_arch}.embedding_length {d}");

    // Geometry -> tensors. patch_embd is stored as [patch, patch, channels, d_v]
    // by the reference converter; accept the flattened form too, because that is
    // the same weight and some exporters write it that way.
    let d_v = mmproj.meta_u32("clip.vision.embedding_length").unwrap() as u64;
    let ps = mmproj.meta_u32("clip.vision.patch_size").unwrap() as u64;
    let patch = mmproj.tensors.get("v.patch_embd.weight").context("mmproj missing v.patch_embd.weight")?;
    ensure!([vec![ps, ps, 3, d_v], vec![ps * ps * 3, d_v]].contains(&patch.dims),
        "mmproj patch embedding shape mismatch: v.patch_embd.weight: {:?} (patch_size {ps}, embedding_length {d_v})", patch.dims);

    // A declared block count the tensors do not cover is a truncated download;
    // catching it here beats a missing-weight panic 60 s into the load.
    let nb = mmproj.meta_u32("clip.vision.block_count").unwrap();
    let last = format!("v.blk.{}.ln1.weight", nb.saturating_sub(1));
    ensure!(nb > 0 && mmproj.tensors.contains_key(&last),
        "mmproj declares {nb} vision blocks but {last} is missing");
    ensure!(mmproj.tensors.keys().any(|k| k.starts_with("mm.")),
        "mmproj has no mm.* projector tensors");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_requires_an_unambiguous_file() {
        let dir = std::env::temp_dir().join(format!("ojas-mmproj-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        std::fs::write(&model, []).unwrap();
        // A directory with only the decoder in it has no mmproj, and that is not
        // an error: every text-only model takes this path.
        assert!(discover(&model, None).unwrap().is_none());
        // A sibling that is not named mmproj is not a candidate.
        std::fs::write(dir.join("draft.gguf"), []).unwrap();
        assert!(discover(&model, None).unwrap().is_none());
        let one = dir.join("model-mmproj.gguf");
        std::fs::write(&one, []).unwrap();
        assert_eq!(discover(&model, None).unwrap(), Some(one.clone()));
        std::fs::write(dir.join("mmproj-f16.gguf"), []).unwrap();
        assert!(discover(&model, None).unwrap_err().to_string().contains("ambiguous"));
        // Explicit always wins over ambiguity, and must exist.
        assert_eq!(discover(&model, Some(one.to_str().unwrap())).unwrap(), Some(one));
        assert!(discover(&model, Some("/nonexistent/ojas-mmproj.gguf")).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_decoder_named_mmproj_is_not_discovered_as_its_own_sidecar() {
        let dir = std::env::temp_dir().join(format!("ojas-mmproj-self-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("mmproj-model.gguf");
        std::fs::write(&model, []).unwrap();
        assert!(discover(&model, None).unwrap().is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use crate::gguf::TensorInfo;

    fn empty_gguf(tag: &str) -> (Gguf, PathBuf) {
        let path = std::env::temp_dir().join(format!("ojas-mmproj-meta-{}-{tag}.gguf", std::process::id()));
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend([0; 24]);
        std::fs::write(&path, bytes).unwrap();
        (Gguf::open(path.to_str().unwrap()).unwrap(), path)
    }

    fn good_pair(tag: &str) -> (Gguf, Gguf, PathBuf) {
        let (mut base, path) = empty_gguf(tag);
        let mut mm = Gguf::open(path.to_str().unwrap()).unwrap();
        base.meta.insert("general.architecture".into(), Meta::Str("qwen35".into()));
        base.meta.insert("qwen35.embedding_length".into(), Meta::U32(1024));
        mm.meta.insert("general.architecture".into(), Meta::Str("clip".into()));
        mm.meta.insert("general.type".into(), Meta::Str("mmproj".into()));
        mm.meta.insert("clip.has_vision_encoder".into(), Meta::Bool(true));
        for (k, v) in [("projection_dim", 1024u32), ("image_size", 768), ("patch_size", 16),
                       ("embedding_length", 768), ("feed_forward_length", 3072),
                       ("block_count", 12), ("attention.head_count", 12)] {
            mm.meta.insert(format!("clip.vision.{k}"), Meta::U32(v));
        }
        mm.tensors.insert("v.patch_embd.weight".into(), TensorInfo { dims: vec![16, 16, 3, 768], ggml_type: 1, rel_offset: 0, part: 0 });
        mm.tensors.insert("v.blk.11.ln1.weight".into(), TensorInfo { dims: vec![768], ggml_type: 0, rel_offset: 0, part: 0 });
        mm.tensors.insert("mm.2.weight".into(), TensorInfo { dims: vec![3072, 1024], ggml_type: 1, rel_offset: 0, part: 0 });
        (base, mm, path)
    }

    #[test]
    fn accepts_a_well_formed_projector() {
        let (base, mm, path) = good_pair("accepts_a_we");
        validate(&base, &mm).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_a_decoder_passed_as_an_mmproj() {
        let (base, mut mm, path) = good_pair("rejects_a_de");
        mm.meta.insert("general.type".into(), Meta::Str("model".into()));
        mm.meta.insert("general.architecture".into(), Meta::Str("qwen35".into()));
        assert!(validate(&base, &mm).unwrap_err().to_string().contains("not an mmproj"));
        std::fs::remove_file(path).unwrap();
    }

    /// architecture=clip alone is enough: not every converter writes general.type.
    #[test]
    fn accepts_clip_architecture_without_general_type() {
        let (base, mut mm, path) = good_pair("accepts_clip");
        mm.meta.remove("general.type");
        validate(&base, &mm).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_a_projector_for_a_different_checkpoint() {
        let (base, mut mm, path) = good_pair("rejects_a_pr");
        mm.meta.insert("clip.vision.projection_dim".into(), Meta::U32(2048));
        let err = validate(&base, &mm).unwrap_err().to_string();
        assert!(err.contains("projection_dim 2048") && err.contains("1024"), "{err}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn names_the_missing_key_a_missing_tensor_and_a_short_block_stack() {
        let (base, mut mm, path) = good_pair("names_the_mi");
        mm.meta.remove("clip.vision.feed_forward_length");
        assert!(validate(&base, &mm).unwrap_err().to_string().contains("clip.vision.feed_forward_length"));
        mm.meta.insert("clip.vision.feed_forward_length".into(), Meta::U32(3072));

        mm.tensors.get_mut("v.patch_embd.weight").unwrap().dims = vec![14, 14, 3, 768];
        assert!(validate(&base, &mm).unwrap_err().to_string().contains("patch embedding shape"));
        mm.tensors.remove("v.patch_embd.weight");
        assert!(validate(&base, &mm).unwrap_err().to_string().contains("v.patch_embd.weight"));
        mm.tensors.insert("v.patch_embd.weight".into(), TensorInfo { dims: vec![16 * 16 * 3, 768], ggml_type: 1, rel_offset: 0, part: 0 });
        validate(&base, &mm).unwrap(); // flattened patch embedding is the same weight

        mm.meta.insert("clip.vision.block_count".into(), Meta::U32(24));
        assert!(validate(&base, &mm).unwrap_err().to_string().contains("v.blk.23.ln1.weight"));
        mm.meta.insert("clip.vision.block_count".into(), Meta::U32(12));

        mm.tensors.remove("mm.2.weight");
        assert!(validate(&base, &mm).unwrap_err().to_string().contains("no mm.* projector tensors"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_a_decoder_with_no_embedding_length() {
        let (mut base, mm, path) = good_pair("no-embd-len");
        base.meta.remove("qwen35.embedding_length");
        assert!(validate(&base, &mm).unwrap_err().to_string().contains("embedding_length"));
        std::fs::remove_file(path).unwrap();
    }
}
