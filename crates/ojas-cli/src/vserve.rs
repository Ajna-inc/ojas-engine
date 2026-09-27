//! `ojas vision-serve` — HTTP inference for vision models, so a detector can run
//! on a different machine from the one decoding the video.
//!
//! One model per process, served sequentially on the thread that owns it
//! (`Detector` keeps interior-mutable state and is not `Sync`), the same shape as
//! the LLM `serve` command. Concurrency belongs to several processes, not several
//! threads.
//!
//! Access control: peers on a trusted network are served without a token, so a
//! LAN caller — typically a video pipeline sending frames at 5 fps per camera —
//! pays no per-request auth cost. Everyone else must present
//! `Authorization: Bearer <token>`. The defaults are narrow (bind loopback, trust
//! loopback only), so exposing the service is an explicit act:
//!
//! ```text
//! ojas vision-serve models/yolo11n.onnx                      # 127.0.0.1:8730, loopback only
//! ojas vision-serve m.onnx --host 0.0.0.0 --token-file t.txt # token required from anywhere
//! ojas vision-serve m.onnx --host 0.0.0.0 --token-file t.txt \
//!     --trust 192.168.29.0/24                                # that LAN is unauthenticated
//! ```
//!
//! A token is required whenever the bind address is not loopback; the server
//! refuses to start rather than serve an open detector to a network.

use anyhow::{bail, Context, Result};
use ojas_vision::{DetectorCfg, Frame, Runtime, RuntimeCfg};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};

use crate::flags::RunOpts;

const MAX_BODY: usize = 64 * 1024 * 1024;

/// A parsed request: the bits this server acts on.
struct Request {
    method: String,
    path: String,
    query: String,
    bearer: Option<String>,
    body: Vec<u8>,
}

/// An address range that may call without a token.
#[derive(Debug, Clone)]
struct Trusted {
    net: IpAddr,
    bits: u32,
}

impl Trusted {
    fn parse(s: &str) -> Result<Trusted> {
        let (addr, bits) = match s.split_once('/') {
            Some((a, b)) => (a, b.parse::<u32>().context("CIDR prefix")?),
            None => (s, if s.contains(':') { 128 } else { 32 }),
        };
        let net: IpAddr = addr.parse().with_context(|| format!("{s:?}: not an IP address or CIDR range"))?;
        let max = if net.is_ipv6() { 128 } else { 32 };
        anyhow::ensure!(bits <= max, "{s:?}: /{bits} is too long for this address family");
        Ok(Trusted { net, bits })
    }

    fn contains(&self, ip: IpAddr) -> bool {
        // An IPv4 peer arriving on a dual-stack socket looks like ::ffff:a.b.c.d.
        let ip = match ip {
            IpAddr::V6(v6) => v6.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(IpAddr::V6(v6)),
            v4 => v4,
        };
        match (self.net, ip) {
            (IpAddr::V4(a), IpAddr::V4(b)) => prefix_eq(&a.octets(), &b.octets(), self.bits),
            (IpAddr::V6(a), IpAddr::V6(b)) => prefix_eq(&a.octets(), &b.octets(), self.bits),
            _ => false,
        }
    }
}

fn prefix_eq(a: &[u8], b: &[u8], bits: u32) -> bool {
    let (whole, rest) = ((bits / 8) as usize, bits % 8);
    if a[..whole] != b[..whole] {
        return false;
    }
    if rest == 0 {
        return true;
    }
    let mask = 0xffu8 << (8 - rest);
    a[whole] & mask == b[whole] & mask
}

/// Read one HTTP/1.1 request, answering `Expect: 100-continue` before reading the
/// body. curl announces it for any body over 1 KB and a frame always exceeds
/// that; a client that announced it waits for the interim response, so without
/// one every request pays curl's one-second timeout, 13× the inference itself.
fn read_request(stream: &mut BufReader<&TcpStream>) -> Result<Option<Request>> {
    let mut line = String::new();
    if stream.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let (method, target) = match (parts.next(), parts.next()) {
        (Some(m), Some(p)) => (m.to_string(), p.to_string()),
        _ => return Ok(None),
    };
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target, String::new()),
    };
    let (mut len, mut bearer, mut expect_continue) = (0usize, None, false);
    loop {
        let mut h = String::new();
        if stream.read_line(&mut h)? == 0 {
            return Ok(None);
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            } else if k.eq_ignore_ascii_case("authorization") {
                bearer = v.trim().strip_prefix("Bearer ").map(|t| t.trim().to_string());
            } else if k.eq_ignore_ascii_case("expect") && v.trim().eq_ignore_ascii_case("100-continue") {
                expect_continue = true;
            }
        }
    }
    if len > MAX_BODY {
        bail!("request body of {len} bytes exceeds the {MAX_BODY}-byte limit");
    }
    if expect_continue {
        let mut w: &TcpStream = stream.get_ref();
        let _ = w.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
        let _ = w.flush();
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body)?;
    Ok(Some(Request { method, path, query, bearer, body }))
}

fn send(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn send_json(stream: &mut TcpStream, status: &str, v: &Value) {
    send(stream, status, "application/json", v.to_string().as_bytes());
}

fn send_err(stream: &mut TcpStream, status: &str, msg: &str) {
    send_json(stream, status, &json!({ "error": msg }));
}

/// `key=value&…` from the query string.
fn param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query.split('&').find_map(|kv| kv.split_once('=').filter(|(k, _)| *k == key).map(|(_, v)| v))
}

/// Token comparison with no early exit on the first differing byte.
fn token_ok(want: &str, got: &str) -> bool {
    if want.len() != got.len() {
        return false;
    }
    want.bytes().zip(got.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

pub struct Auth {
    token: Option<String>,
    trusted: Vec<Trusted>,
}

impl Auth {
    /// Why a request is allowed, or the reason it is not.
    fn check(&self, peer: Option<IpAddr>, bearer: Option<&str>) -> Result<&'static str, &'static str> {
        if let Some(ip) = peer {
            if self.trusted.iter().any(|t| t.contains(ip)) {
                return Ok("trusted network");
            }
        }
        match (&self.token, bearer) {
            (Some(want), Some(got)) if token_ok(want, got) => Ok("token"),
            (Some(_), Some(_)) => Err("bad token"),
            (Some(_), None) => Err("this address is not trusted: send Authorization: Bearer <token>"),
            (None, _) => Err("this address is not trusted and no token is configured"),
        }
    }
}

/// "cpu" | "auto" | "cuda[:N]" | "vulkan[:N]" → the device the model runs on.
fn vision_device(s: Option<&str>) -> Result<ojas_vision::Device> {
    let ordinal = |v: &str| -> Result<usize> { Ok(v.split_once(':').map(|(_, n)| n.parse()).transpose().context("device ordinal")?.unwrap_or(0)) };
    Ok(match s.unwrap_or("cpu") {
        "cpu" => ojas_vision::Device::Cpu,
        "auto" => ojas_vision::Device::Auto,
        d if d.starts_with("cuda") => ojas_vision::Device::Cuda(ordinal(d)?),
        d if d.starts_with("vulkan") => ojas_vision::Device::Vulkan(ordinal(d)?),
        other => bail!("--vision-device {other:?}: want cpu, auto, cuda[:N] or vulkan[:N]"),
    })
}

/// `ojas vision-serve <model.onnx> [--host H] [--port P] [--token T | --token-file F] [--trust CIDR,…] [--vision-device cuda:0]`
pub fn vision_serve(model_path: &str, opts: &RunOpts) -> Result<()> {
    let host = opts.host.clone();
    let port = if opts.port_set { opts.port } else { 8730 };
    let token = match (&opts.token, &opts.token_file) {
        (Some(_), Some(_)) => bail!("--token and --token-file are alternatives"),
        (Some(t), None) => Some(t.trim().to_string()),
        (None, Some(f)) => Some(std::fs::read_to_string(f).with_context(|| format!("reading {f}"))?.trim().to_string()),
        (None, None) => None,
    };
    if let Some(t) = &token {
        anyhow::ensure!(t.len() >= 16, "the token is {} characters; use at least 16", t.len());
    }
    let mut trusted = vec![Trusted::parse("127.0.0.1/32")?, Trusted::parse("::1/128")?];
    if let Some(list) = &opts.trust {
        for part in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            trusted.push(Trusted::parse(part)?);
        }
    }
    let loopback_only = host == "127.0.0.1" || host == "::1" || host == "localhost";
    if !loopback_only && token.is_none() {
        bail!(
            "binding {host} without a token would serve this detector to the network.\n\
             Pass --token-file <file> (a caller on a --trust range still needs no token), \
             or bind 127.0.0.1."
        );
    }
    let auth = Auth { token, trusted };

    let rt = Runtime::new(RuntimeCfg { device: vision_device(opts.vision_device.as_deref())?, ..Default::default() })?;
    let cfg = DetectorCfg { conf: opts.conf.unwrap_or(0.25), iou: opts.iou.unwrap_or(0.45), ..Default::default() };
    let mut det = rt.detector(model_path, cfg.clone()).with_context(|| format!("loading {model_path}"))?;
    // Without a `.classes.json` sidecar an 80-class head falls back to COCO names,
    // the same fallback `ojas detect` prints.
    let nc = det.class_names().map(|n| n.len()).unwrap_or(if det.classes() == 80 { 80 } else { det.classes() });
    let classes: Vec<String> = (0..nc).map(|i| crate::vision::class_name(det.class_names(), nc, i as u16)).collect();
    let model_id = std::path::Path::new(model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.to_string());

    let listener = TcpListener::bind((host.as_str(), port)).with_context(|| format!("binding {host}:{port}"))?;
    let started = std::time::Instant::now();
    let (mut served, mut frames_seen) = (0u64, 0u64);
    eprintln!(
        "ojas vision-serve: {model_id} on http://{host}:{port}  ({} classes, auth: {}, trusted: {})",
        classes.len(),
        if auth.token.is_some() { "token" } else { "trusted networks only" },
        auth.trusted.iter().map(|t| format!("{}/{}", t.net, t.bits)).collect::<Vec<_>>().join(" ")
    );

    for conn in listener.incoming() {
        let mut stream = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let peer = stream.peer_addr().ok().map(|a| a.ip());
        let req = {
            let mut reader = BufReader::new(&stream);
            match read_request(&mut reader) {
                Ok(Some(r)) => r,
                Ok(None) => continue,
                Err(e) => {
                    send_err(&mut stream, "400 Bad Request", &format!("{e:#}"));
                    continue;
                }
            }
        };
        // Health needs no credentials: it says nothing a port scan cannot see.
        if req.path == "/v1/health" {
            send_json(
                &mut stream,
                "200 OK",
                &json!({"ok": true, "model": model_id, "classes": classes.len(),
                        "uptime_s": started.elapsed().as_secs(), "served": served, "frames": frames_seen}),
            );
            continue;
        }
        if let Err(why) = auth.check(peer, req.bearer.as_deref()) {
            eprintln!("refused {peer:?} {} {}: {why}", req.method, req.path);
            send_err(&mut stream, "401 Unauthorized", why);
            continue;
        }
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/v1/models") => send_json(
                &mut stream,
                "200 OK",
                &json!({"models": [{"id": model_id, "task": "detect", "input": cfg.input,
                                    "classes": classes, "conf": cfg.conf, "iou": cfg.iou}]}),
            ),
            // Raw RGB8 in the body, dimensions in the query: the caller sends the
            // pixels it already has, with no base64 and no re-encode.
            //   POST /v1/detect?width=1920&height=1080[&conf=0.35]
            ("POST", "/v1/detect") => {
                let dims = (param(&req.query, "width").and_then(|v| v.parse::<usize>().ok()), param(&req.query, "height").and_then(|v| v.parse::<usize>().ok()));
                let (w, h) = match dims {
                    (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
                    _ => {
                        send_err(&mut stream, "400 Bad Request", "need ?width=<px>&height=<px>");
                        continue;
                    }
                };
                if req.body.len() != w * h * 3 {
                    send_err(&mut stream, "400 Bad Request", &format!("{} bytes of body for a {w}×{h} RGB8 frame (want {})", req.body.len(), w * h * 3));
                    continue;
                }
                let t0 = std::time::Instant::now();
                match det.run(&[Frame::Rgb8 { w, h, data: &req.body }]) {
                    Ok(out) => {
                        let ms = t0.elapsed().as_secs_f64() * 1e3;
                        served += 1;
                        frames_seen += 1;
                        let dets: Vec<Value> = out
                            .first()
                            .map(|v| {
                                v.iter()
                                    .map(|d| {
                                        json!({"class": d.class, "name": classes.get(d.class as usize).cloned().unwrap_or_else(|| d.class.to_string()),
                                               "score": d.score, "box": [d.x0, d.y0, d.x1, d.y1]})
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        send_json(&mut stream, "200 OK", &json!({"model": model_id, "ms": ms, "detections": dets}));
                    }
                    Err(e) => send_err(&mut stream, "500 Internal Server Error", &format!("{e:#}")),
                }
            }
            ("OPTIONS", _) => send(&mut stream, "204 No Content", "text/plain", b""),
            _ => send_err(&mut stream, "404 Not Found", "GET /v1/health, GET /v1/models, POST /v1/detect"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cidr_membership() {
        let lan = Trusted::parse("192.168.29.0/24").unwrap();
        assert!(lan.contains("192.168.29.103".parse().unwrap()));
        assert!(!lan.contains("192.168.30.1".parse().unwrap()));
        let host = Trusted::parse("10.1.2.3").unwrap();
        assert!(host.contains("10.1.2.3".parse().unwrap()));
        assert!(!host.contains("10.1.2.4".parse().unwrap()));
        // An odd prefix length.
        let odd = Trusted::parse("10.0.0.0/12").unwrap();
        assert!(odd.contains("10.15.255.1".parse().unwrap()));
        assert!(!odd.contains("10.16.0.1".parse().unwrap()));
        // A v4 peer on a dual-stack socket.
        assert!(lan.contains("::ffff:192.168.29.7".parse().unwrap()));
        assert!(Trusted::parse("127.0.0.1/32").unwrap().contains("127.0.0.1".parse().unwrap()));
    }

    #[test]
    fn auth_rules() {
        let lan: IpAddr = "192.168.29.5".parse().unwrap();
        let far: IpAddr = "8.8.8.8".parse().unwrap();
        let a = Auth { token: Some("0123456789abcdef".into()), trusted: vec![Trusted::parse("192.168.29.0/24").unwrap()] };
        assert!(a.check(Some(lan), None).is_ok(), "the trusted LAN needs no token");
        assert!(a.check(Some(far), Some("0123456789abcdef")).is_ok());
        assert!(a.check(Some(far), Some("wrong")).is_err());
        assert!(a.check(Some(far), None).is_err());
        assert!(a.check(None, None).is_err(), "an unknown peer is not trusted");
        // No token configured: only trusted ranges get in.
        let b = Auth { token: None, trusted: vec![Trusted::parse("127.0.0.1/32").unwrap()] };
        assert!(b.check(Some("127.0.0.1".parse().unwrap()), None).is_ok());
        assert!(b.check(Some(far), Some("anything")).is_err());
    }

    #[test]
    fn token_compare_is_length_safe() {
        assert!(token_ok("abcd", "abcd"));
        assert!(!token_ok("abcd", "abce"));
        assert!(!token_ok("abcd", "abcde"));
        assert!(!token_ok("", "x"));
    }
}
