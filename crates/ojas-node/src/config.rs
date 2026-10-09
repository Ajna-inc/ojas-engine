//! Node configuration: a TOML or JSON file (by extension), overridden by flags.
//! Relative paths in the file resolve against the file's directory, so a config
//! and its key, pool and models can move together.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub key: Option<PathBuf>,
    pub pool: Option<PathBuf>,
    /// An invite token, or `@path` to a file holding one. Not needed on the admin.
    pub invite: Option<String>,
    pub listen: Vec<String>,
    pub bootstrap: Vec<String>,
    /// Relays to listen through when this node has no inbound reachability.
    pub relays: Vec<String>,
    pub external: Vec<String>,
    pub relay_server: bool,
    pub mdns: Option<bool>,
    pub autonat: Option<bool>,
    /// Devices to run an engine worker on: "metal", "cuda", "cpu".
    pub devices: Vec<String>,
    pub worker_bin: Option<PathBuf>,
    /// Arguments before `--socket ... --device ...`. Default `["engine-worker"]`.
    pub worker_args: Option<Vec<String>>,
    /// Extra environment for worker processes.
    pub worker_env: BTreeMap<String, String>,
    pub models: Vec<ModelCfg>,
    /// Local API port on 127.0.0.1. 0 disables the API.
    pub api_port: Option<u16>,
    pub api_token_file: Option<PathBuf>,
    pub announce_secs: Option<u64>,
    /// Working directory for worker sockets and training data.
    pub data_dir: Option<PathBuf>,
    /// Run config of a DiLoCo run this node coordinates.
    pub coordinator: Option<PathBuf>,
    pub train: Option<TrainCfg>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelCfg {
    pub path: PathBuf,
    /// API name; defaults to the file stem.
    pub name: Option<String>,
    /// The model's ModelId (hex), for a model this node routes but does not load:
    /// identity is a content hash only a worker computes.
    pub id: Option<String>,
    /// Load it on the local workers. Default true.
    pub load: Option<bool>,
    pub context: Option<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainCfg {
    /// Coordinator multiaddr including `/p2p/<PeerId>`.
    pub coordinator: String,
    pub run: String,
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut c: Config = match path.extension().and_then(|e| e.to_str()) {
            Some("json") => serde_json::from_str(&s).with_context(|| format!("parsing {}", path.display()))?,
            Some("toml") => toml::from_str(&s).with_context(|| format!("parsing {}", path.display()))?,
            _ => bail!("{}: config must be .toml or .json", path.display()),
        };
        let base = path.parent().unwrap_or(Path::new("."));
        let rel = |p: &mut PathBuf| {
            if p.is_relative() {
                *p = base.join(&*p);
            }
        };
        for p in [&mut c.key, &mut c.pool, &mut c.worker_bin, &mut c.api_token_file, &mut c.data_dir, &mut c.coordinator].into_iter().flatten() {
            rel(p);
        }
        for m in &mut c.models {
            rel(&mut m.path);
        }
        if let Some(inv) = c.invite.as_mut() {
            if let Some(p) = inv.strip_prefix('@') {
                let mut p = PathBuf::from(p);
                rel(&mut p);
                *inv = format!("@{}", p.display());
            }
        }
        Ok(c)
    }

    pub fn data_dir(&self) -> PathBuf {
        self.data_dir.clone().unwrap_or_else(|| PathBuf::from(".ojas-node"))
    }

    pub fn key_path(&self) -> PathBuf {
        self.key.clone().unwrap_or_else(|| self.data_dir().join("node.key"))
    }

    pub fn invite_token(&self) -> Result<Option<String>> {
        Ok(match &self.invite {
            None => None,
            Some(s) => match s.strip_prefix('@') {
                Some(p) => Some(std::fs::read_to_string(p).with_context(|| format!("reading invite {p}"))?.trim().to_string()),
                None => Some(s.trim().to_string()),
            },
        })
    }
}

impl ModelCfg {
    pub fn name(&self) -> String {
        self.name.clone().unwrap_or_else(|| self.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_and_json_agree_and_paths_resolve_against_the_file() {
        let dir = std::env::temp_dir().join(format!("ojas-node-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let t = dir.join("n.toml");
        std::fs::write(&t, "key = \"k.key\"\ndevices = [\"cpu\"]\n[[models]]\npath = \"m.gguf\"\n").unwrap();
        let j = dir.join("n.json");
        std::fs::write(&j, r#"{"key": "k.key", "devices": ["cpu"], "models": [{"path": "m.gguf"}]}"#).unwrap();
        for p in [t, j] {
            let c = Config::load(&p).unwrap();
            assert_eq!(c.key.unwrap(), dir.join("k.key"));
            assert_eq!(c.models[0].path, dir.join("m.gguf"));
            assert_eq!(c.models[0].name(), "m");
        }
        std::fs::write(dir.join("bad.toml"), "nonsense_field = 1\n").unwrap();
        assert!(Config::load(&dir.join("bad.toml")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
