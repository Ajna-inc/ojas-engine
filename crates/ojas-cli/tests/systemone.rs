//! `POST /v1/systemone` end to end: `ojas serve` on a decision model, over HTTP.
//!
//! The model files are not in the repository. These tests run when
//! `OJAS_DECISION_MODELS` names the directory holding them (the models of
//! `crates/ojas-decision/tests/decision/cases.json`):
//!
//! ```text
//! OJAS_DECISION_MODELS=~/models/decision cargo test --release -p ojas-cli \
//!     --test systemone -- --ignored
//! ```

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const STATE: &str = "I was charged twice for my order last week and nobody has replied.";

fn questions() -> Value {
    json!({
        "route": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "payments and refunds", "shipping": null, "technical": null}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["can wait", "this week", "today", "right now"]},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    })
}

/// A request fixture shared with the model-level tests.
fn fixture(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../ojas-decision/tests/decision/requests/{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

/// `ojas serve` on one model, stopped when dropped.
struct Server {
    child: Child,
    port: u16,
}

impl Server {
    /// Start the server on `file`, or `None` when the model is not available.
    fn start(file: &str) -> Option<Server> {
        let dir = PathBuf::from(std::env::var("OJAS_DECISION_MODELS").expect("OJAS_DECISION_MODELS names the model directory"));
        let model = dir.join(file);
        if !model.is_file() {
            eprintln!("skipped: {file} is not in {}", dir.display());
            return None;
        }
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let child = Command::new(env!("CARGO_BIN_EXE_ojas"))
            .args(["serve", model.to_str().unwrap(), "--port", &port.to_string()])
            .stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
        let server = Server { child, port };
        let deadline = Instant::now() + Duration::from_secs(120);
        while server.get("/health").map(|(s, _)| s) != Some(200) {
            assert!(Instant::now() < deadline, "{file}: the server did not come up");
            std::thread::sleep(Duration::from_millis(100));
        }
        Some(server)
    }

    fn call(&self, method: &str, path: &str, body: &str) -> Option<(u16, String)> {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).ok()?;
        write!(stream, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
                        Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).ok()?;
        let mut text = String::new();
        stream.read_to_string(&mut text).ok()?;
        let status = text.split(' ').nth(1)?.parse().ok()?;
        Some((status, text.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default()))
    }

    fn get(&self, path: &str) -> Option<(u16, String)> { self.call("GET", path, "") }

    /// `POST /v1/systemone`: the status and the parsed body.
    fn decide(&self, body: &Value) -> (u16, Value) {
        let (status, text) = self.call("POST", "/v1/systemone", &body.to_string()).expect("the server answers");
        (status, serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}")))
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn keys(v: &Value) -> Vec<String> { v.as_object().unwrap().keys().cloned().collect() }

fn sum(v: &Value) -> f64 { v.as_object().unwrap().values().map(|p| p.as_f64().unwrap()).sum() }

#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn answers_have_the_systemone_shape() {
    for file in ["tinylaya-Q8_0.gguf", "tinyopenjev-Q8_0.gguf"] {
        let Some(server) = Server::start(file) else { continue };
        let (status, res) = server.decide(&json!({"state": STATE, "questions": questions()}));
        assert_eq!(status, 200, "{file}: {res}");
        assert!(res["usage"]["input_tokens"].as_u64().unwrap() > 0);
        assert_eq!(res["usage"]["output_tokens"], 0);
        let answers = &res["answers"];
        assert_eq!(keys(answers), ["route", "urgency", "angry"]);

        let route = &answers["route"];
        assert_eq!(route["type"], "choice");
        assert_eq!(keys(&route["probabilities"]), ["billing", "shipping", "technical"]);
        assert!((sum(&route["probabilities"]) - 1.0).abs() < 1e-4);
        let best = route["probabilities"].as_object().unwrap().iter()
            .max_by(|a, b| a.1.as_f64().unwrap().total_cmp(&b.1.as_f64().unwrap())).unwrap().0.clone();
        assert_eq!(route["choice"], best);
        assert!((0.0..=1.0).contains(&route["confidence"].as_f64().unwrap()));

        let urgency = &answers["urgency"];
        assert_eq!(urgency["type"], "score");
        assert_eq!(urgency["legend"], json!({"0": "can wait", "1": "this week", "2": "today", "3": "right now"}));
        assert_eq!(keys(&urgency["probabilities"]), ["0", "1", "2", "3"]);
        assert!((sum(&urgency["probabilities"]) - 1.0).abs() < 1e-4);
        let expected: f64 = urgency["probabilities"].as_object().unwrap().values().enumerate()
            .map(|(i, p)| i as f64 * p.as_f64().unwrap()).sum();
        assert!((urgency["score"].as_f64().unwrap() - expected).abs() < 1e-4);
        assert!((0.0..=1.0).contains(&urgency["confidence"].as_f64().unwrap()));

        let angry = &answers["angry"];
        assert_eq!(angry["type"], "noul");
        assert!((0.0..=1.0).contains(&angry["noul"].as_f64().unwrap()));
    }
}

#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn a_json_state_reads_as_its_json_text() {
    let Some(server) = Server::start("tinylaya-Q8_0.gguf") else { return };
    let questions = json!({"refund": {"type": "noul", "instructions": "Is a refund requested?",
                                      "criteria": {"false": "no refund is asked", "true": "a refund is asked"}}});
    let (s1, object) = server.decide(&json!({"state": {"ticket": STATE, "plan": "pro"}, "questions": questions}));
    let (s2, text) = server.decide(&json!({"state": format!("{{\"ticket\": \"{STATE}\", \"plan\": \"pro\"}}"), "questions": questions}));
    assert_eq!((s1, s2), (200, 200));
    assert_eq!(object["usage"], text["usage"]);
    let noul = |r: &Value| r["answers"]["refund"]["noul"].as_f64().unwrap();
    assert!((noul(&object) - noul(&text)).abs() < 1e-4);
}

#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn invalid_requests_are_refused() {
    let Some(server) = Server::start("tinylaya-Q8_0.gguf") else { return };
    let q = |question: Value| json!({"state": STATE, "questions": {"q": question}});
    for body in [
        json!({"questions": questions()}),
        json!({"state": STATE}),
        json!({"state": STATE, "questions": {}}),
        q(json!({"type": "unknown", "instructions": "x"})),
        q(json!({"type": "noul"})),
        q(json!({"type": "choice", "instructions": "x"})),
        q(json!({"type": "choice", "instructions": "x", "criteria": {}})),
        q(json!({"type": "score", "instructions": "x", "criteria": ["only one"]})),
    ] {
        let (status, res) = server.decide(&body);
        assert_eq!(status, 400, "{body}: {res}");
        assert_eq!(res["error"]["type"], "invalid_request_error", "{body}");
    }
    let (status, _) = server.call("POST", "/v1/systemone", "{not json").unwrap();
    assert_eq!(status, 400);
}

#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn the_shared_prompt_is_evaluated_once_and_reported() {
    let Some(server) = Server::start("tinyopenjev-Q8_0.gguf") else { return };
    let (status, res) = server.decide(&json!({"state": STATE, "questions": questions()}));
    assert_eq!(status, 200);
    let (_, metrics) = server.get("/metrics").unwrap();
    let counter = |name: &str| -> u64 {
        metrics.lines().find_map(|l| l.strip_prefix(&format!("ojas_decision_{name} "))).unwrap().parse().unwrap()
    };
    let (computed, cached) = (counter("prompt_tokens_total"), counter("prompt_tokens_cached_total"));
    assert!(cached > 0, "nothing was shared:\n{metrics}");
    assert_eq!(computed + cached, res["usage"]["input_tokens"].as_u64().unwrap());
    assert_eq!(counter("requests_total"), 1);
}

#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn images_are_read_from_the_array_and_from_message_parts() {
    let Some(server) = Server::start("tinyopenjev-Q8_0.gguf") else { return };
    let (_, text) = server.decide(&fixture("ticket"));
    let tokens = |r: &Value| r["usage"]["input_tokens"].as_u64().unwrap();
    for name in ["one_image", "image_in_message"] {
        let (status, res) = server.decide(&fixture(name));
        assert_eq!(status, 200, "{name}: {res}");
        assert_eq!(keys(&res["answers"]), ["route", "urgency", "angry"]);
        assert!(tokens(&res) > tokens(&text), "{name}: the image added no tokens");
    }
    let mut nine = fixture("one_image");
    let image = nine["images"][0].clone();
    nine["images"] = json!(vec![image; 9]);
    let mut url = fixture("one_image");
    url["images"] = json!(["https://example.com/image.png"]);
    for body in [nine, url] {
        let (status, res) = server.decide(&body);
        assert_eq!(status, 400, "{res}");
    }
    let (_, props) = server.get("/props").unwrap();
    assert_eq!(serde_json::from_str::<Value>(&props).unwrap()["images"], true);
}

#[test]
#[ignore = "needs the decision model files; set OJAS_DECISION_MODELS"]
fn images_need_a_model_that_takes_them() {
    let Some(server) = Server::start("tinylaya-Q8_0.gguf") else { return };
    let (status, res) = server.decide(&fixture("one_image"));
    assert_eq!(status, 501, "{res}");
    assert_eq!(res["error"]["type"], "not_supported_error");
    assert_eq!(res["error"]["code"], 501);
}
