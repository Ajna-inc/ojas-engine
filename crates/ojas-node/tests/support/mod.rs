//! Helpers shared by the multi-process tests.
#![allow(dead_code)]

use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub const BIN: &str = env!("CARGO_BIN_EXE_ojas-node");

pub struct Proc(Child);
impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub fn cli(args: &[&str]) -> String {
    let o = Command::new(BIN).args(args).output().unwrap();
    assert!(o.status.success(), "ojas-node {args:?}: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().trim().to_string()
}

pub fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// A GGUF with a byte-level vocabulary and ChatML markers, and no tensors: enough
/// for the node to tokenise and detokenise.
pub fn write_gguf(path: &Path) {
    let (enc, _) = ojas_tokenize::byte_maps();
    let mut tokens: Vec<String> = (0..=255u8).map(|b| enc[&b].to_string()).collect();
    let mut types = vec![1i32; 256];
    for s in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] {
        tokens.push(s.into());
        types.push(3);
    }
    let mut b = Vec::new();
    let s = |b: &mut Vec<u8>, x: &str| {
        b.extend_from_slice(&(x.len() as u64).to_le_bytes());
        b.extend_from_slice(x.as_bytes());
    };
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&6u64.to_le_bytes());
    s(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes());
    s(&mut b, "mocklm");
    s(&mut b, "tokenizer.ggml.model");
    b.extend_from_slice(&8u32.to_le_bytes());
    s(&mut b, "gpt2");
    s(&mut b, "tokenizer.ggml.tokens");
    b.extend_from_slice(&9u32.to_le_bytes());
    b.extend_from_slice(&8u32.to_le_bytes());
    b.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for t in &tokens {
        s(&mut b, t);
    }
    s(&mut b, "tokenizer.ggml.token_type");
    b.extend_from_slice(&9u32.to_le_bytes());
    b.extend_from_slice(&5u32.to_le_bytes());
    b.extend_from_slice(&(types.len() as u64).to_le_bytes());
    for t in &types {
        b.extend_from_slice(&t.to_le_bytes());
    }
    s(&mut b, "tokenizer.ggml.merges");
    b.extend_from_slice(&9u32.to_le_bytes());
    b.extend_from_slice(&8u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    s(&mut b, "tokenizer.ggml.eos_token_id");
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&257u32.to_le_bytes());
    while b.len() % 32 != 0 {
        b.push(0);
    }
    std::fs::File::create(path).unwrap().write_all(&b).unwrap();
}

pub struct N {
    pub dir: PathBuf,
    pub p2p: u16,
    pub api: u16,
    pub id: String,
    proc: Option<Proc>,
}

impl N {
    pub fn new(root: &Path, name: &str) -> N {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let id = cli(&["id", "--key", dir.join("node.key").to_str().unwrap()]);
        N { dir, p2p: free_port(), api: free_port(), id, proc: None }
    }

    pub fn addr(&self) -> String {
        format!("/ip4/127.0.0.1/tcp/{}/p2p/{}", self.p2p, self.id)
    }

    pub fn start(&mut self, mut cfg: Value) {
        let d = &self.dir;
        let base = json!({
            "key": d.join("node.key"), "listen": [format!("/ip4/127.0.0.1/tcp/{}", self.p2p)],
            "mdns": false, "autonat": false, "api_port": self.api, "api_token_file": d.join("api.token"),
            "data_dir": d, "announce_secs": 1,
        });
        for (k, v) in base.as_object().unwrap() {
            cfg.as_object_mut().unwrap().entry(k.clone()).or_insert(v.clone());
        }
        let path = d.join("node.json");
        std::fs::write(&path, cfg.to_string()).unwrap();
        let log = std::fs::File::create(d.join("node.log")).unwrap();
        let child = Command::new(BIN)
            .args(["--config", path.to_str().unwrap()])
            .env("RUST_LOG", "info,ojas_net=debug,libp2p=warn")
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap();
        self.proc = Some(Proc(child));
    }

    pub fn token(&self) -> String {
        std::fs::read_to_string(self.dir.join("api.token")).unwrap_or_default()
    }

    pub async fn get(&self, c: &reqwest::Client, path: &str) -> Option<Value> {
        let r = c.get(format!("http://127.0.0.1:{}{path}", self.api)).bearer_auth(self.token()).send().await.ok()?;
        r.json().await.ok()
    }

    pub async fn post(&self, c: &reqwest::Client, path: &str, body: Value) -> (u16, String) {
        let r = c.post(format!("http://127.0.0.1:{}{path}", self.api)).bearer_auth(self.token()).json(&body).send().await.unwrap();
        (r.status().as_u16(), r.text().await.unwrap())
    }

    pub fn log(&self) -> String {
        std::fs::read_to_string(self.dir.join("node.log")).unwrap_or_default()
    }
}

/// Poll `f` on `n`'s /status until it holds.
pub async fn until(c: &reqwest::Client, n: &N, what: &str, secs: u64, f: impl Fn(&Value) -> bool) -> Value {
    let t = Instant::now();
    loop {
        if let Some(s) = n.get(c, "/status").await {
            if f(&s) {
                return s;
            }
        }
        if t.elapsed() > Duration::from_secs(secs) {
            panic!("timed out waiting for {what}; status: {:?}\nlog:\n{}", n.get(c, "/status").await, n.log());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

