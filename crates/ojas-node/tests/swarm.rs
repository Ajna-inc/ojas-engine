//! Four `ojas-node` processes on loopback with mock workers; no GPU, no model.
//!
//! A is the pool admin with the model's tokenizer but no worker. B and C are
//! invited members whose mock workers serve the model; B's worker crashes on its
//! third generation. D belongs to an impostor pool of the same name.

mod support;

use serde_json::{json, Value};
use std::time::Duration;
use support::*;

fn announces<'a>(s: &'a Value, peer: &str) -> Option<&'a Value> {
    s["table"].as_array()?.iter().find(|e| e["peer"] == peer).map(|e| &e["announce"])
}

fn serves(s: &Value, peer: &str, model: &str) -> bool {
    announces(s, peer).and_then(|a| a["models"].as_array()).is_some_and(|m| m.iter().any(|m| m["model"] == model))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pool_routes_fails_over_and_refuses_strangers() {
    let root = std::env::temp_dir().join(format!("ojnit-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let gguf = root.join("tiny.gguf");
    write_gguf(&gguf);
    let gguf_s = gguf.to_str().unwrap().to_string();
    // The mock worker's identity for a model is BLAKE3 of its path.
    let model_id = blake3::hash(gguf_s.as_bytes()).to_hex().to_string();

    let mut a = N::new(&root, "a");
    let mut b = N::new(&root, "b");
    let mut c = N::new(&root, "c");
    let mut d = N::new(&root, "d");
    let pool = root.join("pool.json");
    let admin = cli(&["pool", "init", "--name", "it", "--key", a.dir.join("node.key").to_str().unwrap(), "--out", pool.to_str().unwrap()]);
    assert_eq!(admin, a.id);
    let invite = |who: &N| cli(&["pool", "invite", &who.id, "--days", "1", "--key", a.dir.join("node.key").to_str().unwrap(), "--pool", pool.to_str().unwrap()]);
    let (inv_b, inv_c) = (invite(&b), invite(&c));
    // D: an impostor pool with the same name, whose own admin invited it.
    let fake = root.join("fake.json");
    let fake_key = root.join("fake-admin.key");
    cli(&["pool", "init", "--name", "it", "--key", fake_key.to_str().unwrap(), "--out", fake.to_str().unwrap()]);
    let inv_d = cli(&["pool", "invite", &d.id, "--key", fake_key.to_str().unwrap(), "--pool", fake.to_str().unwrap()]);

    let worker = |env: Value| json!({"devices": ["cpu"], "worker_bin": BIN, "worker_args": ["mock-worker"], "worker_env": env, "models": [{"path": gguf_s}]});
    a.start(json!({"pool": pool, "models": [{"path": gguf_s, "id": model_id, "load": false}]}));
    let mut bc = worker(json!({"OJAS_MOCK_CRASH_AT_REQ": "3"}));
    bc["pool"] = json!(pool);
    bc["invite"] = json!(inv_b);
    bc["bootstrap"] = json!([a.addr()]);
    b.start(bc);

    let http = reqwest::Client::new();
    until(&http, &a, "B's announce at A", 20, |s| serves(s, &b.id, &model_id)).await;

    // 1. A has no worker: B serves it, over PROTO_STREAM.
    let (code, body) = a.post(&http, "/v1/completions", json!({"model": "tiny", "prompt": "Hello world", "max_tokens": 5})).await;
    assert_eq!(code, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["text"], "Hello");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["ojas"]["served_by"], b.id);

    // 2. Streaming chat, also through B.
    let (code, body) = a.post(&http, "/v1/chat/completions", json!({"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 6, "stream": true})).await;
    assert_eq!(code, 200, "{body}");
    let events: Vec<&str> = body.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
    assert_eq!(events.last(), Some(&"[DONE]"), "{body}");
    let text: String = events.iter().filter_map(|e| serde_json::from_str::<Value>(e).ok()).filter_map(|e| e["choices"][0]["delta"]["content"].as_str().map(str::to_string)).collect();
    assert!(text.contains("user"), "streamed {text:?}");
    assert!(body.contains(&b.id));

    // 3. C joins. B has a measured decode rate and C none, so B ranks first, and
    // B's worker dies before its first token: A must fail over to C.
    let mut cc = worker(json!({}));
    cc["pool"] = json!(pool);
    cc["invite"] = json!(inv_c);
    cc["bootstrap"] = json!([a.addr()]);
    c.start(cc);
    until(&http, &a, "C's announce and B's rate at A", 20, |s| {
        serves(s, &c.id, &model_id) && announces(s, &b.id).is_some_and(|a| a["models"][0]["tok_s"].is_number())
    })
    .await;
    let (code, body) = a.post(&http, "/v1/completions", json!({"model": "tiny", "prompt": "Hello world", "max_tokens": 5})).await;
    assert_eq!(code, 200, "{body}\nA log:\n{}", a.log());
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["ojas"]["served_by"], c.id, "{body}");
    assert!(a.log().contains("failing over"), "C served without B being tried first");
    assert_eq!(v["choices"][0]["text"], "Hello");
    until(&http, &b, "B's worker restarted", 20, |s| s["workers"][0]["restarts"].as_u64() >= Some(1) && s["workers"][0]["alive"] == true).await;

    // 4. Gossip round trip: C learns what B serves without asking B.
    until(&http, &c, "B's announce at C", 20, |s| serves(s, &b.id, &model_id)).await;

    // 5. D is refused by every member, and its gossip never lands.
    let mut dc = json!({"pool": fake, "invite": inv_d, "bootstrap": [a.addr()]});
    dc["models"] = json!([]);
    d.start(dc);
    until(&http, &d, "D running", 20, |_| true).await;
    tokio::time::sleep(Duration::from_secs(3)).await;
    let sa = a.get(&http, "/status").await.unwrap();
    let members: Vec<&str> = sa["members"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(members.contains(&b.id.as_str()) && members.contains(&c.id.as_str()), "{members:?}");
    assert!(!members.contains(&d.id.as_str()), "{members:?}");
    assert!(announces(&sa, &d.id).is_none());
    assert!(a.log().contains("refusing peer") || d.log().contains("refusing peer"), "nobody refused D");
    let sd = d.get(&http, "/status").await.unwrap();
    assert_eq!(sd["members"].as_array().unwrap().len(), 0, "{sd}");
    assert!(sd["table"].as_array().unwrap().is_empty(), "{sd}");

    // The API wants its token.
    let r = http.get(format!("http://127.0.0.1:{}/status", a.api)).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 401);
    drop((a, b, c, d));
    let _ = std::fs::remove_dir_all(&root);
}
