//! Load a decision server (`ojas serve` or any `/v1/systemone` server) with concurrent
//! clients and report throughput, latency and the answers.
//!
//! Each client sends the request bodies of a JSONL file in turn, one at a time, for
//! `--seconds` (default 30) after a warm-up of one pass. Reported: requests and prompt
//! tokens per second, wall latency percentiles, failures. `--save out.jsonl` writes one
//! response per request body (from a single-client pass, in file order) for
//! `--compare a.jsonl b.jsonl`, which prints the largest probability difference per
//! request between two servers' answers to the same bodies.
//!
//! usage: decision_load <url> <requests.jsonl> [--clients N] [--seconds S] [--save out.jsonl]
//!        decision_load --compare a.jsonl b.jsonl

use anyhow::{bail, ensure, Context, Result};
use ojas_decision::json::Json;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

/// One POST, the response body (status 200) or an error.
fn post(host: &str, port: u16, path: &str, body: &str) -> Result<String> {
    let mut s = TcpStream::connect((host, port)).with_context(|| format!("connecting to {host}:{port}"))?;
    s.set_read_timeout(Some(Duration::from_secs(600)))?;
    write!(s, "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw)?;
    let text = String::from_utf8_lossy(&raw);
    let (head, rest) = text.split_once("\r\n\r\n").context("no response header")?;
    let status = head.lines().next().unwrap_or("");
    ensure!(status.contains(" 200 "), "{status}: {}", rest.chars().take(200).collect::<String>());
    // chunked transfer: join the chunks
    if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        let mut out = String::new();
        let mut rest = rest;
        loop {
            let (len, after) = rest.split_once("\r\n").context("bad chunk")?;
            let n = usize::from_str_radix(len.trim(), 16).context("bad chunk size")?;
            if n == 0 { break; }
            out.push_str(&after[..n]);
            rest = &after[n + 2..];
        }
        return Ok(out);
    }
    Ok(rest.to_string())
}

fn url_parts(url: &str) -> Result<(String, u16, String)> {
    let u = url.trim_end_matches('/').trim_start_matches("http://");
    let (hostport, path) = u.split_once('/').map(|(h, p)| (h, format!("/{p}"))).unwrap_or((u, "/v1/systemone".into()));
    let (host, port) = hostport.split_once(':').context("url needs host:port")?;
    Ok((host.to_string(), port.parse()?, path))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn compare(a: &str, b: &str) -> Result<()> {
    let read = |p: &str| -> Result<Vec<Json>> {
        std::fs::read_to_string(p)?.lines().filter(|l| !l.trim().is_empty()).map(Json::parse).collect()
    };
    let (ra, rb) = (read(a)?, read(b)?);
    ensure!(ra.len() == rb.len(), "{} vs {} responses", ra.len(), rb.len());
    let mut worst_all = 0.0f64;
    for (i, (x, y)) in ra.iter().zip(&rb).enumerate() {
        let (ax, ay) = (x.get("answers").context("no answers")?, y.get("answers").context("no answers")?);
        let Json::Object(qs) = ax else { bail!("answers is not an object") };
        let mut worst = 0.0f64;
        let mut where_ = String::new();
        for (id, ans) in qs {
            let other = ay.get(id).with_context(|| format!("response {i}: no answer {id} in {b}"))?;
            let pairs: Vec<(f64, f64)> = match ans.get("noul").and_then(Json::as_f64) {
                Some(p) => vec![(p, other.get("noul").and_then(Json::as_f64).unwrap_or(f64::NAN))],
                None => {
                    let Some(Json::Object(pa)) = ans.get("probabilities") else { continue };
                    pa.iter().map(|(k, v)| (v.as_f64().unwrap_or(f64::NAN),
                        other.get("probabilities").and_then(|o| o.get(k)).and_then(Json::as_f64).unwrap_or(f64::NAN))).collect()
                }
            };
            for (p, q) in pairs {
                let d = (p - q).abs();
                if d.is_nan() || d > worst { worst = if d.is_nan() { f64::INFINITY } else { d }; where_ = id.clone(); }
            }
        }
        let tokens = |r: &Json| r.get("usage").and_then(|u| u.get("input_tokens")).and_then(Json::as_f64).unwrap_or(f64::NAN);
        println!("request {i}: worst |dp| {worst:.2e} ({where_}), input tokens {} vs {}", tokens(x), tokens(y));
        worst_all = worst_all.max(worst);
    }
    println!("worst over all requests: {worst_all:.2e}");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--compare") {
        ensure!(args.len() == 3, "usage: decision_load --compare a.jsonl b.jsonl");
        return compare(&args[1], &args[2]);
    }
    ensure!(args.len() >= 2, "usage: decision_load <url> <requests.jsonl> [--clients N] [--seconds S] [--save out.jsonl]");
    let (host, port, path) = url_parts(&args[0])?;
    let bodies: Vec<String> = std::fs::read_to_string(&args[1])?.lines().filter(|l| !l.trim().is_empty()).map(String::from).collect();
    ensure!(!bodies.is_empty(), "no request bodies in {}", args[1]);
    let (mut clients, mut seconds, mut save) = (1usize, 30.0f64, None);
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--clients" => { clients = args[i + 1].parse()?; i += 2; }
            "--seconds" => { seconds = args[i + 1].parse()?; i += 2; }
            "--save" => { save = Some(args[i + 1].clone()); i += 2; }
            other => bail!("unknown flag {other}"),
        }
    }
    let tokens_of = |resp: &str| -> f64 {
        Json::parse(resp).ok().and_then(|j| j.get("usage").and_then(|u| u.get("input_tokens")).and_then(Json::as_f64)).unwrap_or(0.0)
    };

    // warm-up and the saved pass: every body once, one client
    let mut saved = Vec::new();
    for b in &bodies {
        let r = post(&host, port, &path, b).context("warm-up request failed")?;
        saved.push(r);
    }
    if let Some(p) = &save {
        std::fs::write(p, saved.join("\n") + "\n")?;
        eprintln!("saved {} responses to {p}", saved.len());
    }

    let deadline = Instant::now() + Duration::from_secs_f64(seconds);
    let t0 = Instant::now();
    let results: Vec<(Vec<f64>, f64, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..clients).map(|c| {
            let (host, path, bodies) = (host.clone(), path.clone(), &bodies);
            s.spawn(move || {
                let (mut lat, mut toks, mut fails) = (Vec::new(), 0.0, 0usize);
                let mut k = c;
                while Instant::now() < deadline {
                    let t = Instant::now();
                    match post(&host, port, &path, &bodies[k % bodies.len()]) {
                        Ok(r) => { lat.push(t.elapsed().as_secs_f64() * 1e3); toks += tokens_of(&r); }
                        Err(e) => { fails += 1; if fails <= 3 { eprintln!("client {c}: {e:#}"); } }
                    }
                    k += clients;
                }
                (lat, toks, fails)
            })
        }).collect();
        handles.into_iter().map(|h| h.join().expect("client thread")).collect()
    });
    let wall = t0.elapsed().as_secs_f64();
    let mut lat: Vec<f64> = results.iter().flat_map(|(l, _, _)| l.iter().copied()).collect();
    lat.sort_by(f64::total_cmp);
    let n = lat.len();
    let toks: f64 = results.iter().map(|(_, t, _)| t).sum();
    let fails: usize = results.iter().map(|(_, _, f)| f).sum();
    println!("{} | {clients} client(s), {n} requests in {wall:.1} s: {:.2} req/s, {:.0} prompt tokens/s | latency ms p50 {:.1} p95 {:.1} max {:.1} | {fails} failed",
        args[0], n as f64 / wall, toks / wall, percentile(&lat, 0.5), percentile(&lat, 0.95), lat.last().copied().unwrap_or(f64::NAN));
    Ok(())
}
