//! Deterministic sidecar selection and metadata checks, before GPU allocation.
use crate::gguf::Gguf;
use anyhow::{ensure, Context, Result};
use std::path::{Path, PathBuf};

pub fn discover(model: &Path, explicit: Option<&str>) -> Result<Option<PathBuf>> {
    if let Some(path) = explicit {
        let path = PathBuf::from(path);
        ensure!(path.is_file(), "explicit MTP sidecar does not exist: {}", path.display());
        return Ok(Some(path));
    }
    if model.as_os_str().is_empty() { return Ok(None); }
    let dir = model.parent().unwrap_or(Path::new(".")).join("MTP");
    let entries = match std::fs::read_dir(&dir) {
        Ok(v) => v,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading MTP directory {}", dir.display())),
    };
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if path.extension().is_some_and(|s| s == "gguf") && path.is_file() { paths.push(path); }
    }
    paths.sort();
    ensure!(paths.len() <= 1, "ambiguous MTP directory {}: {} GGUF files; select one with OJAS_MTP=/path/to/draft.gguf", dir.display(), paths.len());
    Ok(paths.pop())
}

pub fn validate(base: &Gguf, draft: &Gguf, layer: usize) -> Result<()> {
    let arch = base.arch();
    ensure!(draft.arch() == arch, "MTP architecture differs from target");
    for field in ["embedding_length", "attention.head_count", "attention.head_count_kv", "attention.key_length", "attention.value_length", "expert_count", "expert_used_count", "expert_feed_forward_length", "hyper_connection.count"] {
        let key = format!("{arch}.{field}");
        ensure!(base.meta_u32(&key) == draft.meta_u32(&key), "MTP geometry mismatch: {key}");
    }
    let d = base.meta_u32(&format!("{arch}.embedding_length")).context("missing target embedding length")? as u64;
    let hc = base.meta_u32(&format!("{arch}.hyper_connection.count")).unwrap_or(1) as u64;
    for (suffix, shapes) in [
        ("nextn.eh_proj.weight", vec![vec![2*d,d]]),
        ("nextn.enorm.weight", vec![vec![d]]),
        ("nextn.hnorm.weight", vec![vec![d],vec![d*hc]]),
    ] {
        let name = format!("blk.{layer}.{suffix}");
        let tensor = draft.tensors.get(&name).with_context(|| format!("MTP sidecar missing {name}"))?;
        ensure!(shapes.contains(&tensor.dims), "MTP tensor shape mismatch: {name}: {:?}", tensor.dims);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_requires_an_unambiguous_file() {
        let dir=std::env::temp_dir().join(format!("ojas-mtp-{}",std::process::id()));
        std::fs::create_dir_all(dir.join("MTP")).unwrap();
        let model=dir.join("model.gguf");
        assert!(discover(&model,None).unwrap().is_none());
        let one=dir.join("MTP/a.gguf"); std::fs::write(&one,[]).unwrap();
        assert_eq!(discover(&model,None).unwrap(),Some(one.clone()));
        std::fs::write(dir.join("MTP/b.gguf"),[]).unwrap();
        assert!(discover(&model,None).unwrap_err().to_string().contains("ambiguous"));
        assert_eq!(discover(&model,Some(one.to_str().unwrap())).unwrap(),Some(one));
        assert!(discover(&model,Some("/nonexistent/ojas-mtp.gguf")).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use crate::gguf::{Meta, TensorInfo};
    #[test]
    fn rejects_wrong_architecture_geometry_and_missing_or_misshaped_combiner() {
        let path=std::env::temp_dir().join(format!("ojas-mtp-meta-{}.gguf",std::process::id()));
        let mut bytes=b"GGUF".to_vec(); bytes.extend(3u32.to_le_bytes()); bytes.extend([0;24]);
        std::fs::write(&path,bytes).unwrap();
        let mut base=Gguf::open(path.to_str().unwrap()).unwrap();
        let mut draft=Gguf::open(path.to_str().unwrap()).unwrap();
        for g in [&mut base,&mut draft] {
            g.meta.insert("general.architecture".into(),Meta::Str("qwen4exp".into()));
            g.meta.insert("qwen4exp.embedding_length".into(),Meta::U32(8));
            g.meta.insert("qwen4exp.hyper_connection.count".into(),Meta::U32(4));
        }
        for (name,dims) in [("eh_proj",vec![16,8]),("enorm",vec![8]),("hnorm",vec![32])] {
            draft.tensors.insert(format!("blk.48.nextn.{name}.weight"),TensorInfo { dims,ggml_type:0,rel_offset:0,part:0 });
        }
        validate(&base,&draft,48).unwrap();
        draft.meta.insert("general.architecture".into(),Meta::Str("qwen3".into()));
        assert!(validate(&base,&draft,48).is_err());
        draft.meta.insert("general.architecture".into(),Meta::Str("qwen4exp".into()));
        draft.meta.insert("qwen4exp.embedding_length".into(),Meta::U32(16));
        assert!(validate(&base,&draft,48).is_err());
        draft.meta.insert("qwen4exp.embedding_length".into(),Meta::U32(8));
        draft.tensors.get_mut("blk.48.nextn.hnorm.weight").unwrap().dims=vec![7];
        assert!(validate(&base,&draft,48).is_err());
        draft.tensors.remove("blk.48.nextn.eh_proj.weight");
        assert!(validate(&base,&draft,48).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
