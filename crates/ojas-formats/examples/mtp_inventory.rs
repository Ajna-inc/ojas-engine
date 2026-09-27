//! Read architecture and MTP tensor metadata without loading model weights.
use anyhow::Result;
fn main() -> Result<()> {
    for path in std::env::args().skip(1) {
        let g = ojas_formats::gguf::Gguf::open(&path)?;
        let arch = g.arch();
        let mut combiners: Vec<_> = g.tensors.keys().filter(|n| n.ends_with("nextn.eh_proj.weight")).collect();
        combiners.sort();
        println!("{}", serde_json::json!({
            "file": std::path::Path::new(&path).file_name().unwrap().to_string_lossy(),
            "architecture": arch, "layers": g.meta_u32(&format!("{arch}.block_count")),
            "nextn_layers": g.meta_u32(&format!("{arch}.nextn_predict_layers")).unwrap_or(0),
            "combiners": combiners,
            "sidecar": ojas_formats::mtp::discover(std::path::Path::new(&path), None)?.map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        }));
    }
    Ok(())
}
