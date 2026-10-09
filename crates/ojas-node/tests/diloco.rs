//! A DiLoCo run over the swarm: the admin node coordinates, an invited member
//! with a mock worker pulls rounds, trains and pushes deltas until the run is done.
#![cfg(feature = "coordinator")]

mod support;

use serde_json::json;
use support::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_trains_every_round_through_the_coordinator() {
    let root = std::env::temp_dir().join(format!("ojdil-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let text: Vec<u8> = b"the quick brown fox jumps over the lazy dog. ".iter().copied().cycle().take(8192).collect();
    std::fs::write(root.join("train.ojtk"), ojas_swarm::train::data::from_bytes(&text)).unwrap();
    let run = json!({
        "run": "it-run",
        "model": {"kind": "tiny_gpt", "vocab": 256, "ctx": 16, "d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "init_seed": 1},
        "data": "train.ojtk", "inner_steps": 2, "batch": 2, "inner_lr": 0.001,
        "rounds": 3, "min_members": 1, "max_members": 1, "round_deadline_secs": 60,
        "checkpoint_dir": "ckpt", "eval_seqs": 2,
    });
    std::fs::write(root.join("run.json"), run.to_string()).unwrap();

    let mut a = N::new(&root, "a");
    let mut b = N::new(&root, "b");
    let pool = root.join("pool.json");
    cli(&["pool", "init", "--name", "it", "--key", a.dir.join("node.key").to_str().unwrap(), "--out", pool.to_str().unwrap()]);
    let inv = cli(&["pool", "invite", &b.id, "--key", a.dir.join("node.key").to_str().unwrap(), "--pool", pool.to_str().unwrap()]);

    a.start(json!({"pool": pool, "coordinator": root.join("run.json")}));
    b.start(json!({
        "pool": pool, "invite": inv, "bootstrap": [a.addr()],
        "devices": ["cpu"], "worker_bin": BIN, "worker_args": ["mock-worker"],
        "train": {"coordinator": a.addr(), "run": "it-run"},
    }));
    let http = reqwest::Client::new();
    until(&http, &a, "coordinator up", 20, |s| s["coordinator"] == true).await;
    let s = until(&http, &b, "the run to finish at the member", 40, |s| s["train"]["state"] == "done").await;
    assert_eq!(s["train"]["run"], "it-run");
    assert!(b.log().contains("round 2 delta accepted"), "{}", b.log());
    assert!(root.join("ckpt").read_dir().unwrap().next().is_some(), "the coordinator checkpointed");
    drop((a, b));
    let _ = std::fs::remove_dir_all(&root);
}
